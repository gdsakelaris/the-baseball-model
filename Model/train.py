"""Train the MLB prediction models.

Models (all LightGBM):
  batter props (binary + isotonic calibration):
    hr, hit, hits2 (2+ hits), tb2 (2+ total bases), run (run scored),
    rbi (1+ RBI), bb (walk), sb (stolen base)
  k     starter strikeouts in the game              Poisson regression
  runs  game total runs                             Poisson regression

Honest evaluation protocol (no leakage):
  train on 2020-2024  ->  early-stop & calibrate on 2025  ->  test on 2026.
The shipped artifacts are exactly the models those 2026 numbers describe.
A secondary backtest (train<=2023 / cal 2024 / test 2025) checks stability.

Usage:
    python Model/train.py [--rebuild]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)

LGB_CLS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=127,
               min_child_samples=80, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=3.0, max_bin=255,
               objective="binary", verbose=-1)
LGB_POIS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                min_child_samples=60, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=3.0, objective="poisson",
                verbose=-1)
# The winner model trains on ~10k games, not ~190k batter-games: batter-scale
# capacity overfit instantly (v1 early-stopped at 26 trees, test AUC 0.52).
# Small trees + heavy regularization let boosting actually accumulate signal.
LGB_WIN = dict(n_estimators=3000, learning_rate=0.02, num_leaves=15,
               min_child_samples=150, subsample=0.9, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=10.0, objective="binary",
               verbose=-1)

# Per-prop feature exclusions. Stolen bases regressed when the platoon-split
# features arrived (iteration 2) — platoon matchups don't drive steals, the
# extra columns just diluted that model. Each prop's actual column list is
# saved in its artifact, so predict/evaluate pick it up automatically.
PROP_EXCLUDE = {
    "sb": ["vsh_PA", "vsh_hr_pa_sh", "vsh_tb_ab_sh", "vsh_k_pct_sh"],
}

# batter prop -> (target column, description)
PROPS = {
    "hr": ("y_hr", "home run"),
    "hit": ("y_hit", "1+ hit"),
    "hits2": ("y_hits2", "2+ hits"),
    "tb2": ("y_tb2", "2+ total bases"),
    "run": ("y_run", "run scored"),
    "rbi": ("y_rbi", "1+ RBI"),
    "bb": ("y_bb", "1+ walk"),
    "sb": ("y_sb", "stolen base"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_categories(df, cat_levels):
    for c, levels in cat_levels.items():
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=levels)
    return df


def _fit_logistic(tr, cols, target):
    """Regularized logistic on numeric features (categoricals dropped) — a
    learner diverse from the trees, so blending the two helps."""
    num_cols = [c for c in cols if c not in F.CAT_COLS]
    pipe = Pipeline([
        # F.inf_to_nan lives in the features module so the pickled pipeline
        # resolves it from predict.py/evaluate.py too (not just __main__).
        ("clean", FunctionTransformer(F.inf_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=0.3, solver="lbfgs")),
    ])
    pipe.fit(tr[num_cols], tr[target])
    return pipe, num_cols


def fit_classifier(df, cols, target, train_yrs, cal_yr, test_yr, name,
                   params=None):
    tr = df[df["Season"].isin(train_yrs)]
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr]
    model = lgb.LGBMClassifier(**(params or LGB_CLS))
    model.fit(tr[cols], tr[target],
              eval_set=[(ca[cols], ca[target])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])

    # diverse second learner + blend weight chosen on the calibration year
    lr, num_cols = _fit_logistic(tr, cols, target)
    g_cal = model.predict_proba(ca[cols])[:, 1]
    l_cal = lr.predict_proba(ca[num_cols])[:, 1]
    yca = ca[target].to_numpy()
    best_w, best_ll = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        ll = log_loss(yca, np.clip(w * g_cal + (1 - w) * l_cal, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll, best_w = ll, w

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(best_w * g_cal + (1 - best_w) * l_cal, yca)

    g_te = model.predict_proba(te[cols])[:, 1]
    l_te = lr.predict_proba(te[num_cols])[:, 1]
    p_te = iso.predict(best_w * g_te + (1 - best_w) * l_te)
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": int(model.best_iteration_ or 0),
        "blend_gbm_weight": round(float(best_w), 2),
        "auc": float(roc_auc_score(y, p_te)),
        "acc": float(((p_te >= 0.5).astype(float) == y).mean()),
        "logloss": float(log_loss(y, p_te)),
        "logloss_baserate": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p_te)),
        "brier_baserate": float(brier_score_loss(y, base)),
    }
    # calibration by decile
    q = pd.qcut(p_te, 10, duplicates="drop")
    cal_tab = pd.DataFrame({"pred": p_te, "y": y}).groupby(q, observed=True).agg(
        pred_mean=("pred", "mean"), actual=("y", "mean"), n=("y", "size"))
    metrics["calibration"] = [
        {"pred": round(r.pred_mean, 4), "actual": round(r.actual, 4), "n": int(r.n)}
        for r in cal_tab.itertuples()]
    # daily top-10 lift (ranking value: does the top of the list hit?)
    day = pd.DataFrame({"d": te["Date"].values, "p": p_te, "y": y})
    top = day.sort_values("p", ascending=False).groupby("d").head(10)
    metrics["top10_daily_hit_rate"] = float(top["y"].mean())
    log(f"{name} [{test_yr}]: AUC {metrics['auc']:.4f} | "
        f"logloss {metrics['logloss']:.4f} (base {metrics['logloss_baserate']:.4f}) | "
        f"brier {metrics['brier']:.4f} (base {metrics['brier_baserate']:.4f}) | "
        f"top10/day {metrics['top10_daily_hit_rate']:.3f} vs base "
        f"{metrics['base_rate']:.3f} | gbm wt {best_w:.2f}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols, "w": best_w, "iso": iso}
    return prop, metrics


def fit_winner(wf, cols, target, mu_map, train_yrs, cal_yr, test_yr, name):
    """Home-win model: small-capacity GBM + logistic, then a second blend
    with the runs-model Poisson win probability (a diverse signal — park,
    weather, starter run prevention), then isotonic calibration. Both blend
    weights are chosen on the calibration year.

    mu_map: per-GamePk expected runs (mu_away, mu_home) from the runs
    model, or None to skip the Poisson component. The runs model trains on
    2020-2024, so pass None whenever the calibration year falls inside that
    range: its in-sample predictions look falsely sharp there, the blend
    collapses onto them, and the isotonic miscalibrates (this corrupted the
    first 2025 backtest — cal 2024 is training data for the runs model)."""
    from predict import poisson_win
    tr = wf[wf["Season"].isin(train_yrs)]
    ca = wf[wf["Season"] == cal_yr]
    te = wf[wf["Season"] == test_yr]
    model = lgb.LGBMClassifier(**LGB_WIN)
    model.fit(tr[cols], tr[target],
              eval_set=[(ca[cols], ca[target])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    lr, num_cols = _fit_logistic(tr, cols, target)

    def parts(d):
        g = model.predict_proba(d[cols])[:, 1]
        l = lr.predict_proba(d[num_cols])[:, 1]
        if mu_map is None:
            return g, l, np.full(len(d), np.nan)
        mus = mu_map.reindex(d["GamePk"])
        pois = np.array([poisson_win(h, a) for h, a in
                         zip(mus["mu_home"], mus["mu_away"])])
        return g, l, pois

    yca = ca[target].to_numpy()

    def pick_w(a, b):
        best_w, best_ll = 1.0, np.inf
        for w in np.linspace(0.0, 1.0, 21):
            ll = log_loss(yca, np.clip(w * a + (1 - w) * b, 1e-6, 1 - 1e-6))
            if ll < best_ll:
                best_ll, best_w = ll, w
        return best_w

    g_cal, l_cal, pois_cal = parts(ca)
    w1 = pick_w(g_cal, l_cal)
    s_cal = w1 * g_cal + (1 - w1) * l_cal
    pois_cal = np.where(np.isfinite(pois_cal), pois_cal, s_cal)
    w_ml = 1.0 if mu_map is None else pick_w(s_cal, pois_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(w_ml * s_cal + (1 - w_ml) * pois_cal, yca)

    g_te, l_te, pois_te = parts(te)
    s_te = w1 * g_te + (1 - w1) * l_te
    pois_te = np.where(np.isfinite(pois_te), pois_te, s_te)
    p_te = iso.predict(w_ml * s_te + (1 - w_ml) * pois_te)
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": int(model.best_iteration_ or 0),
        "blend_gbm_weight": round(float(w1), 2),
        "blend_ml_weight": round(float(w_ml), 2),
        "auc": float(roc_auc_score(y, p_te)),
        "acc": float(((p_te >= 0.5).astype(float) == y).mean()),
        "logloss": float(log_loss(y, p_te)),
        "logloss_baserate": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p_te)),
        "brier_baserate": float(brier_score_loss(y, base)),
    }
    log(f"{name} [{test_yr}]: AUC {metrics['auc']:.4f} | acc "
        f"{metrics['acc']:.3f} | logloss {metrics['logloss']:.4f} "
        f"(base {metrics['logloss_baserate']:.4f}) | gbm wt {w1:.2f} | "
        f"ML-vs-poisson wt {w_ml:.2f}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols, "w": w1,
            "w_ml": w_ml, "iso": iso}
    return prop, metrics


def fit_poisson(df, cols, target, train_yrs, cal_yr, test_yr, name, baseline):
    tr = df[df["Season"].isin(train_yrs)]
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr].copy()
    model = lgb.LGBMRegressor(**LGB_POIS)
    model.fit(tr[cols], tr[target],
              eval_set=[(ca[cols], ca[target])], eval_metric="poisson",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    pred = model.predict(te[cols])
    y = te[target].to_numpy()
    bl = baseline(te)
    metrics = {
        "n_train": len(tr), "n_test": len(te),
        "best_iter": int(model.best_iteration_ or 0),
        "mae": float(mean_absolute_error(y, pred)),
        "mae_baseline": float(mean_absolute_error(y, bl)),
        "mean_actual": float(y.mean()), "mean_pred": float(pred.mean()),
    }
    log(f"{name} [{test_yr}]: MAE {metrics['mae']:.3f} "
        f"(baseline {metrics['mae_baseline']:.3f}) | "
        f"mean pred {metrics['mean_pred']:.2f} vs actual {metrics['mean_actual']:.2f}")
    return model, metrics


def naive_hr_baseline(te, slot_pa, league_hr_pa):
    """P(HR) if you only used season HR/PA and lineup slot."""
    rate = te["s_hr_pa"].fillna(league_hr_pa).clip(0, 0.15)
    exp_pa = te["slot"].map(slot_pa).fillna(4.1)
    return 1 - (1 - rate) ** exp_pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild feature frames even if cached")
    args = ap.parse_args()

    cache = ART / "frames.joblib"
    if cache.exists() and not args.rebuild:
        log("loading cached feature frames")
        frames = joblib.load(cache)
    else:
        log("loading raw data")
        raw = F.load_raw()
        log("building batter frame (this is the big one)")
        bf = F.build_batter_frame(raw)
        log(f"batter frame: {len(bf):,} rows")
        log("building starts frame")
        sf = F.build_starts_frame(raw)
        log(f"starts frame: {len(sf):,} rows")
        log("building game frame")
        gf = F.build_game_frame(raw)
        log(f"game frame: {len(gf):,} rows")
        frames = {"bf": bf, "sf": sf, "gf": gf}
        joblib.dump(frames, cache, compress=3)
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]

    # exclude 7-inning doubleheaders from training grain
    bf = bf[~bf["ShortGame"].fillna(False)].copy()
    sf = sf[~sf["ShortGame"].fillna(False)].copy()

    bat_cols = F.batter_feature_cols()
    st_cols = F.starts_feature_cols()
    gm_cols = F.game_feature_cols()

    cat_levels = {}
    for c in F.CAT_COLS:
        vals = set()
        for frame in (bf, sf, gf):
            if c in frame.columns:
                vals |= set(frame[c].dropna().astype(str).unique())
        cat_levels[c] = sorted(vals)
    for frame in (bf, sf, gf):
        set_categories(frame, cat_levels)

    train_yrs = [2020, 2021, 2022, 2023, 2024]
    slot_pa = bf[bf["Season"].isin(train_yrs)].groupby("slot")["PA"].mean().to_dict()
    league_hr_pa = (bf.loc[bf["Season"].isin(train_yrs), "HR"].sum()
                    / bf.loc[bf["Season"].isin(train_yrs), "PA"].sum())

    metrics = {}

    # -- stability backtest on 2025 -----------------------------------
    log("=== stability backtest (train<=2023, cal 2024, test 2025) ===")
    _, m = fit_classifier(bf, bat_cols, "y_hr",
                          [2020, 2021, 2022, 2023], 2024, 2025, "HR")
    metrics["hr_backtest_2025"] = m

    # -- final models, tested on the 2026 holdout ---------------------
    log("=== final models (train<=2024, cal 2025, test 2026 holdout) ===")
    props = {}
    for name, (target, _desc) in PROPS.items():
        cols = [c for c in bat_cols if c not in PROP_EXCLUDE.get(name, ())]
        prop, m = fit_classifier(bf, cols, target,
                                 train_yrs, 2025, 2026, name.upper())
        prop["cols"] = cols
        props[name] = prop
        if name == "hr":
            te = bf[bf["Season"] == 2026]
            nb = naive_hr_baseline(te, slot_pa, league_hr_pa)
            m["logloss_naive_seasonrate"] = float(
                log_loss(te["y_hr"], nb.clip(1e-4, 1 - 1e-4)))
            m["brier_naive_seasonrate"] = float(brier_score_loss(te["y_hr"], nb))
        metrics[f"{name}_2026"] = m

    def k_baseline(te):
        league = sf.loc[sf["Season"].isin(train_yrs), "y_so"].mean()
        per_start = te["ps_k_bf"] * (te["ps_BF"] / te["p_starts_season"])
        return per_start.fillna(league).clip(0, 15)

    k_model, m = fit_poisson(sf, st_cols, "y_so", train_yrs, 2025, 2026,
                             "K", k_baseline)
    metrics["k_2026"] = m

    # per-team runs model; game totals and win probability derive from it.
    # Canonical row order: LightGBM's bagging draws depend on row order, so
    # without this, unrelated upstream merge changes shuffle rows and move
    # MAE by ~0.005-0.01 — pure noise that pollutes baseline diffs.
    tg = F.build_team_game_frame(gf.dropna(subset=["total_runs"]))
    tg = tg.dropna(subset=["y_runs"])
    tg = tg.sort_values(["GamePk", "Home"]).reset_index(drop=True)
    set_categories(tg, cat_levels)
    tg_cols = F.team_game_feature_cols()

    def team_baseline(te):
        league = tg.loc[tg["Season"].isin(train_yrs), "y_runs"].mean()
        return te["off_r_pg"].fillna(league)

    team_runs_model, m = fit_poisson(tg, tg_cols, "y_runs", train_yrs, 2025,
                                     2026, "TEAM RUNS", team_baseline)
    metrics["team_runs_2026"] = m

    # Game-total dispersion, fit on the CALIBRATION year (2025 — never the
    # holdout): real totals ran ~2.3x Poisson variance, which made pure
    # Poisson P(over) worse than the base rate at low lines. predict.py
    # switches to a negative binomial with this factor.
    tg25 = tg[tg["Season"] == 2025]
    pr25 = team_runs_model.predict(tg25[tg_cols])
    per_game = pd.DataFrame({"g": tg25["GamePk"].to_numpy(), "mu": pr25,
                             "y": tg25["y_runs"].to_numpy()}).groupby("g").sum()
    total_disp = float(np.mean((per_game["y"] - per_game["mu"]) ** 2)
                       / np.mean(per_game["mu"]))
    metrics["total_dispersion_2025"] = total_disp
    log(f"game-total dispersion (2025 cal year): {total_disp:.2f} "
        f"(Poisson assumes 1.00)")

    # dedicated winner model: home-win classifier on team strength (win%,
    # run diff, pythag, Elo with winter carryover), recent form, both
    # starters, bullpens, and rest — blended with the runs-model Poisson
    # win prob (a diverse signal). The 2025 backtest is the tuning
    # yardstick for this model: iterate against IT, and only confirm on
    # 2026, so the holdout stays honest.
    mu_all = team_runs_model.predict(tg[tg_cols])
    mu_map = (pd.DataFrame({"GamePk": tg["GamePk"].to_numpy(),
                            "Home": tg["Home"].to_numpy(), "mu": mu_all})
              .pivot_table(index="GamePk", columns="Home", values="mu")
              .rename(columns={0: "mu_away", 1: "mu_home"}))
    wf = gf[~gf["ShortGame"].fillna(False)].dropna(subset=["y_home_win"])
    wf = wf.sort_values("GamePk").reset_index(drop=True)  # canonical order
    win_cols = F.win_feature_cols()
    # backtest: no Poisson component — its calibration year (2024) is
    # inside the runs model's training data (see fit_winner docstring)
    _, m_bt = fit_winner(wf, win_cols, "y_home_win", None,
                         [2020, 2021, 2022, 2023], 2024, 2025, "WINNER-BT")
    m_bt["acc_home_baseline"] = float(
        wf.loc[wf["Season"] == 2025, "y_home_win"].mean())
    log(f"WINNER backtest [2025]: pick accuracy {m_bt['acc']:.3f} vs "
        f"always-home {m_bt['acc_home_baseline']:.3f}")
    metrics["winner_backtest_2025"] = m_bt
    win_model, m = fit_winner(wf, win_cols, "y_home_win", mu_map,
                              train_yrs, 2025, 2026, "WINNER")
    win_model["cols"] = win_cols
    te = wf[wf["Season"] == 2026]
    m["acc_home_baseline"] = float(te["y_home_win"].mean())
    log(f"WINNER [2026]: pick accuracy {m['acc']:.3f} vs always-home "
        f"{m['acc_home_baseline']:.3f}")
    metrics["winner_2026"] = m

    # multi-HR correction: E[HR | HR>=1], for expected-HR outputs
    tr_hr = bf[bf["Season"].isin(train_yrs) & (bf["hr_count"] >= 1)]
    multi_hr = float(tr_hr["hr_count"].mean())

    artifacts = {
        "props": props,
        "k_model": k_model, "team_runs_model": team_runs_model,
        "win_model": win_model, "total_disp": total_disp,
        "bat_cols": bat_cols, "st_cols": st_cols, "tg_cols": tg_cols,
        "cat_levels": cat_levels, "multi_hr": multi_hr,
        "slot_pa": slot_pa, "league_hr_pa": league_hr_pa,
        "metrics": metrics,
        "trained_on": "2020-2024, calibrated 2025, holdout-tested 2026 YTD",
    }
    joblib.dump(artifacts, ART / "models.joblib", compress=3)
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"saved artifacts to {ART}")

    # feature importances for the HR model (top 25)
    imp = pd.Series(props["hr"]["gbm"].feature_importances_, index=bat_cols)
    log("top HR-model features:\n" +
        imp.sort_values(ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
