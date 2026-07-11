"""Per-head hyperparameter sweep via day-block cross-validation.

Only `run` and `hrr2` carry tuned LightGBM params today (train.PROP_PARAMS);
every other binary shares LGB_CLS and every count head shares LGB_POIS. This
script sweeps a small, curated ladder of regularization/capacity profiles for
each head and picks the best by an HONEST out-of-fold score, so the choice is
made on generalization, not on the 2025 test year.

Design (why it doesn't leak the verification set):
  * The selection suite is train<=2023 / cal 2024 / TEST 2025. This sweep touches
    ONLY Season <= 2024 (the data the selection model is allowed to see). 2025
    stays pristine so `evaluate_deep.py --paired` remains an independent read on
    whatever params we adopt.
  * Day-block CV: rows are grouped by calendar Date (a day is atomic — never
    split across folds), GroupKFold gives K out-of-fold blocks. Same grouping the
    paired day-block bootstrap uses, so the CV and the verification agree on the
    unit of independence.
  * Honest early stopping + calibration: inside each fold's TRAIN rows we carve a
    15% day-grouped watch set. The GBM early-stops on it and (binary heads) an
    isotonic is fit on it; both are then applied to the held-out val block. The
    val block is never seen during fitting, ES, or calibration, so the OOF score
    is clean.

Scoring is balanced (per the north-star): binary heads by OOF logloss (a proper
score that rewards ranking AND calibration) with AUC reported alongside; count
heads by OOF MAE with the matching Poisson/Tweedie deviance alongside.

Output: Model/artifacts/prop_params_sweep.json — the full CV table for every
head x profile plus a gated `recommended` profile. Nothing is applied to
train.py automatically; review the table, then wire winners into PROP_PARAMS and
confirm on 2025 with `evaluate_deep.py --paired`.

--ensemble (families-aware): by default the CV tunes the LGBM member in
ISOLATION. With --ensemble it MeanBags each LGBM profile with the head's XGB +
CatBoost members and scores the ENSEMBLE's OOF — the honest objective now that
the shipped model is the bag, not LGBM alone (heavy LGBM regularization can be
redundant once the families also cut variance). The families' predictions do
NOT depend on the LGBM profile, so they are fit ONCE per fold and cached, then
reused across every profile — that keeps it to ~one families pass + the LGBM
sweep (~2-4 hr) instead of re-fitting all members per profile (~a day). LGBM is
weighted by its real bag size (PROP_BAGS/COUNT_BAGS) in the MeanBag. Needs
train.XGB_BAGS/CB_BAGS > 0 and the GPU; writes prop_params_sweep_ensemble.json.

RE-SWEEP CADENCE: hyperparameter optima are broad basins and drift slowly, so
re-run this on EVENTS, not on the calendar — (1) a large feature-set change,
(2) an era rollover / a new season graduating into the training window, (3) ship
(families back on is itself a re-validation; use --ensemble then). Do NOT
re-sweep on daily data accretion: the CV's own ~+-0.0005 jitter would flip
profiles on noise. Tie it to the era audit cadence.

Usage:
  python Model/param_sweep.py                 # LGBM-only, all heads, K=4
  python Model/param_sweep.py --heads run,rbi,hit
  python Model/param_sweep.py --ensemble --max-rows 80000   # families-aware (GPU)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, log_loss, mean_absolute_error,
                             mean_poisson_deviance, mean_tweedie_deviance)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# import the training module so the sweep reuses its EXACT configs + helpers
# (LGB_CLS/LGB_POIS/LGB_WIN, PROPS/COUNT_HEADS, PROP_EXCLUDE, MONOTONE,
# _apply_keep, set_categories, suite_years) — no risk of the two drifting.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F          # noqa: E402
import train as T             # noqa: E402

ART = T.ART


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Profile ladders. Each dict OVERRIDES the head's global base params. "default"
# = the incumbent base exactly, included so every head is scored against its own
# current config under the identical CV. The ladder spans heavy regularization
# (small leaves / large min_child / low colsample) -> high capacity.
# --------------------------------------------------------------------------
BIN_PROFILES = {
    "default":   {},                                   # LGB_CLS: nl127 mcs80 cs.8 rl3
    "reg_light": dict(min_child_samples=160),
    "reg_med":   dict(num_leaves=63, min_child_samples=160),
    "reg_med2":  dict(num_leaves=63, min_child_samples=300,
                      colsample_bytree=0.7, reg_lambda=6.0),
    "reg_heavy": dict(num_leaves=31, min_child_samples=300,
                      colsample_bytree=0.7, reg_lambda=6.0),
    "cap_high":  dict(num_leaves=255),
    "cap_wide":  dict(colsample_bytree=0.9, min_child_samples=120,
                      reg_lambda=2.0),
    "lr_slow":   dict(learning_rate=0.02),             # finer steps, more trees via ES
}

CNT_PROFILES = {
    "default":   {},                                   # LGB_POIS: nl63 mcs60 cs.8 rl3
    "reg_light": dict(min_child_samples=120),
    "reg_med":   dict(num_leaves=31, min_child_samples=120),
    "reg_heavy": dict(num_leaves=31, min_child_samples=300,
                      colsample_bytree=0.7, reg_lambda=6.0),
    "cap_high":  dict(num_leaves=127, min_child_samples=80),
    "cap_wide":  dict(num_leaves=95, colsample_bytree=0.9, reg_lambda=2.0),
    "lr_slow":   dict(learning_rate=0.02),
}

WIN_PROFILES = {
    "default":  {},                                    # LGB_WIN: nl15 mcs150 lr.02
    "cap_up":   dict(num_leaves=31),
    "cap_up2":  dict(num_leaves=31, min_child_samples=100),
    "reg_up":   dict(num_leaves=15, min_child_samples=300, reg_lambda=6.0),
    "leaves7":  dict(num_leaves=7, min_child_samples=200),
    "lr_up":    dict(learning_rate=0.03),
}

# gates for the "recommended" flag (evidence only — nothing auto-applies).
# A profile must beat default by at least the margin on the PRIMARY metric and
# not regress the SECONDARY metric past its band.
EPS_LL = 0.0004     # binary: min logloss improvement
AUC_BAND = 0.0010   # binary: max tolerated AUC regression
EPS_MAE = 0.0030    # count: min MAE improvement
DEV_BAND = 0.0030   # count: max tolerated deviance regression (relative-ish)


# --------------------------------------------------------------------------
def load_frames():
    """Replicate train.main()'s frame prep so the sweep sees identical data:
    ShortGame excluded, categoricals set to shared levels, tg/wf built the same
    canonical way."""
    cache = ART / "frames.joblib"
    if not cache.exists():
        raise SystemExit("no artifacts/frames.joblib — run train.py --rebuild first")
    log("loading cached feature frames")
    frames = joblib.load(cache)
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]

    bf = bf[~bf["ShortGame"].fillna(False)].copy()
    sf = sf[~sf["ShortGame"].fillna(False)].copy()

    cat_levels = {}
    for c in F.CAT_COLS:
        vals = set()
        for frame in (bf, sf, gf):
            if c in frame.columns:
                vals |= set(frame[c].dropna().astype(str).unique())
        cat_levels[c] = sorted(vals)
    for frame in (bf, sf, gf):
        T.set_categories(frame, cat_levels)

    tg = F.build_team_game_frame(gf.dropna(subset=["total_runs"]))
    tg = tg.dropna(subset=["y_runs"]).sort_values(["GamePk", "Home"]).reset_index(drop=True)
    T.set_categories(tg, cat_levels)

    wf = gf[~gf["ShortGame"].fillna(False)].dropna(subset=["y_home_win"])
    wf = wf.sort_values("GamePk").reset_index(drop=True)
    return bf, sf, tg, wf


def _prep(df, cols, target, cutoff, max_rows, seed):
    """Rows the selection model may see (Season<=cutoff), target present, columns
    sliced, day groups factorized. Row-subsampled (day integrity preserved — each
    surviving row keeps its Date, folds still group by day) when over budget."""
    d = df[df["Season"] <= cutoff].dropna(subset=[target]).copy()
    if max_rows and len(d) > max_rows:
        d = d.sample(n=max_rows, random_state=seed)
    d = d.reset_index(drop=True)
    X = d[cols]
    y = d[target].to_numpy()
    groups = pd.factorize(d["Date"])[0]
    return X, y, groups


def _fit_families_fold(kind, X, y, fit_idx, watch_idx, va_idx, cat_here,
                       xgb_params, cb_params, n_xgb, n_cb, tweedie):
    """Fit the XGBoost + CatBoost members ONCE for a fold and return the SUM of
    their watch- and val-block predictions plus the member count. Their outputs
    are independent of the LGBM profile, so this is cached and reused across
    every profile — fitting the GPU families once per fold instead of once per
    profile is the trick that keeps an ensemble-aware sweep tractable. Mirrors
    train.py's family fit path exactly (InfSafe/CatSafe, Poisson/Tweedie)."""
    Xf, yf = X.iloc[fit_idx], y[fit_idx]
    Xw, Xv, yw = X.iloc[watch_idx], X.iloc[va_idx], y[watch_idx]
    watch_sum, val_sum, n = (np.zeros(len(watch_idx)), np.zeros(len(va_idx)), 0)
    for b in range(n_xgb):
        if kind == "binary":
            m = F.InfSafe(T.xgb_lib.XGBClassifier(**xgb_params, random_state=b))
            m.fit(Xf, yf, eval_set=[(Xw, yw)], verbose=False)
            watch_sum += m.predict_proba(Xw)[:, 1]
            val_sum += m.predict_proba(Xv)[:, 1]
        else:
            p = dict(xgb_params)
            if tweedie is not None:
                p = dict(p, objective="reg:tweedie",
                         tweedie_variance_power=tweedie,
                         eval_metric=f"tweedie-nloglik@{tweedie}")
            m = F.InfSafe(T.xgb_lib.XGBRegressor(**p, random_state=b))
            m.fit(Xf, yf, eval_set=[(Xw, yw)], verbose=False)
            watch_sum += np.clip(m.predict(Xw), 1e-6, None)
            val_sum += np.clip(m.predict(Xv), 1e-6, None)
        n += 1
    for b in range(n_cb):
        if kind == "binary":
            m = F.CatSafe(T.CatBoostClassifier(**cb_params, random_seed=b,
                                               cat_features=cat_here), cat_here)
            m.fit(Xf, yf, eval_set=[(Xw, yw)])
            watch_sum += m.predict_proba(Xw)[:, 1]
            val_sum += m.predict_proba(Xv)[:, 1]
        else:
            p = dict(cb_params)
            if tweedie is not None:
                tw = f"Tweedie:variance_power={tweedie}"
                p = dict(p, loss_function=tw, eval_metric=tw)
            m = F.CatSafe(T.CatBoostRegressor(**p, random_seed=b,
                                              cat_features=cat_here),
                          cat_here, exponent=True)
            m.fit(Xf, yf, eval_set=[(Xw, yw)])
            watch_sum += np.clip(m.predict(Xw), 1e-6, None)
            val_sum += np.clip(m.predict(Xv), 1e-6, None)
        n += 1
    return watch_sum, val_sum, n


def _fold_fit_binary(X, y, fit_idx, watch_idx, va_idx, params, seed,
                     fam=None, w_lgbm=1):
    """Fit one fold: LGBM early-stopped on watch. When `fam` (cached XGB+CB
    prediction sums) is given, MeanBag the LGBM member with them — LGBM weighted
    `w_lgbm` to match its real bag size — so the scored preds are the ENSEMBLE's.
    Isotonic on watch, applied to the held-out val block. Returns calibrated val
    predictions."""
    m = lgb.LGBMClassifier(**params)
    m.fit(X.iloc[fit_idx], y[fit_idx],
          eval_set=[(X.iloc[watch_idx], y[watch_idx])],
          eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(150, verbose=False)])
    raw_w = m.predict_proba(X.iloc[watch_idx])[:, 1]
    raw_v = m.predict_proba(X.iloc[va_idx])[:, 1]
    if fam is not None:
        fw, fv, n_fam = fam
        raw_w = (w_lgbm * raw_w + fw) / (w_lgbm + n_fam)
        raw_v = (w_lgbm * raw_v + fv) / (w_lgbm + n_fam)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(raw_w, y[watch_idx])
    return iso.predict(raw_v)


def _fold_fit_count(X, y, fit_idx, watch_idx, va_idx, params, tweedie, seed,
                    fam=None, w_lgbm=1):
    p = dict(params)
    if tweedie is not None:
        p = dict(p, objective="tweedie", tweedie_variance_power=tweedie)
    m = lgb.LGBMRegressor(**p)
    m.fit(X.iloc[fit_idx], y[fit_idx],
          eval_set=[(X.iloc[watch_idx], y[watch_idx])],
          eval_metric=("tweedie" if tweedie is not None else "poisson"),
          callbacks=[lgb.early_stopping(150, verbose=False)])
    raw_v = np.clip(m.predict(X.iloc[va_idx]), 1e-6, None)
    if fam is not None:                # MeanBag with the cached XGB+CB val sums
        _, fv, n_fam = fam
        raw_v = (w_lgbm * raw_v + fv) / (w_lgbm + n_fam)
    return raw_v


def cv_head(kind, X, y, groups, base, profiles, folds, seed,
            mono_cols=None, tweedie=None, ensemble=False, cat_cols=None,
            xgb_params=None, cb_params=None, n_xgb=0, n_cb=0, w_lgbm=1):
    """Run day-block CV for every profile on one head. Returns {profile: metrics}.
    ensemble=True MeanBags each LGBM profile with the head's XGB+CatBoost members
    (fit once per fold, cached) so the objective is the ENSEMBLE's OOF error —
    the honest target when the shipped model is the bag, not LGBM alone."""
    gkf = GroupKFold(n_splits=folds)
    # carve each fold's 15% day-grouped ES/calibration watch up front
    fold_splits = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        sub_fit, sub_watch = next(gss.split(tr_idx, groups=groups[tr_idx]))
        fold_splits.append((tr_idx[sub_fit], tr_idx[sub_watch], va_idx))
    # families don't depend on the LGBM profile -> fit once per fold, then cache
    fam_cache = [None] * len(fold_splits)
    if ensemble and (n_xgb + n_cb) > 0:
        cat_here = [c for c in X.columns if cat_cols and c in cat_cols]
        for k, (fit_idx, watch_idx, va_idx) in enumerate(fold_splits):
            fam_cache[k] = _fit_families_fold(
                kind, X, y, fit_idx, watch_idx, va_idx, cat_here,
                xgb_params, cb_params, n_xgb, n_cb, tweedie)
    out = {}
    for pname, override in profiles.items():
        params = dict(base, **override)
        if mono_cols is not None:      # hr monotone: constraints track the col order
            params["monotone_constraints"] = mono_cols
            params["monotone_constraints_method"] = "advanced"
        oof = np.full(len(y), np.nan)
        for k, (fit_idx, watch_idx, va_idx) in enumerate(fold_splits):
            if kind == "binary":
                oof[va_idx] = _fold_fit_binary(X, y, fit_idx, watch_idx, va_idx,
                                               params, seed, fam=fam_cache[k],
                                               w_lgbm=w_lgbm)
            else:
                oof[va_idx] = _fold_fit_count(X, y, fit_idx, watch_idx, va_idx,
                                              params, tweedie, seed,
                                              fam=fam_cache[k], w_lgbm=w_lgbm)
        m = {}
        if kind == "binary":
            p = np.clip(oof, 1e-6, 1 - 1e-6)
            m["logloss"] = float(log_loss(y, p))
            m["auc"] = float(roc_auc_score(y, p))
        else:
            m["mae"] = float(mean_absolute_error(y, oof))
            try:
                if tweedie is not None:
                    m["deviance"] = float(mean_tweedie_deviance(y, oof, power=tweedie))
                else:
                    m["deviance"] = float(mean_poisson_deviance(y, oof))
            except ValueError:
                m["deviance"] = float("nan")
            m["mean_pred"] = float(oof.mean())
            m["mean_actual"] = float(y.mean())
        out[pname] = m
    return out


def _recommend(kind, table):
    """Gated winner: best primary metric that also clears the margin over default
    without regressing the secondary past its band. Ties -> keep 'default'."""
    d = table["default"]
    if kind == "binary":
        best, best_ll = "default", d["logloss"]
        for name, m in table.items():
            if name == "default":
                continue
            if (m["logloss"] <= best_ll - EPS_LL
                    and m["auc"] >= d["auc"] - AUC_BAND):
                best, best_ll = name, m["logloss"]
        return best
    best, best_mae = "default", d["mae"]
    for name, m in table.items():
        if name == "default":
            continue
        dev_ok = (np.isnan(m.get("deviance", np.nan))
                  or m["deviance"] <= d["deviance"] + DEV_BAND)
        if m["mae"] <= best_mae - EPS_MAE and dev_ok:
            best, best_mae = name, m["mae"]
    return best


# --------------------------------------------------------------------------
def build_jobs(bf, sf, tg, wf):
    """One job per head: (name, kind, frame, cols, target, base_params, profiles,
    mono_cols, tweedie). Column routing mirrors train.train_suite exactly."""
    bat_cols = F.batter_feature_cols()
    st_cols = F.starts_feature_cols()
    jobs = []

    # binary batter props -------------------------------------------------
    for name, (target, _desc) in T.PROPS.items():
        cols = T._apply_keep(name, [c for c in bat_cols
                                    if c not in T.PROP_EXCLUDE.get(name, ())])
        mono = None
        mc = T.MONOTONE.get(name)
        if mc:
            mono = [mc.get(c, 0) for c in cols]
        jobs.append(dict(name=name, kind="binary", frame=bf, cols=cols,
                         target=target, base=T.LGB_CLS, profiles=BIN_PROFILES,
                         mono=mono, tweedie=None,
                         xgb_params=T.XGB_CLS, cb_params=T.CB_CLS,
                         n_xgb=T.XGB_BAGS, n_cb=T.CB_BAGS,
                         w_lgbm=T.PROP_BAGS.get(name, 1)))

    # count heads ---------------------------------------------------------
    for cname, ch in T.COUNT_HEADS.items():
        if ch["frame"] == "bat":
            frame = bf
            cols = [c for c in bat_cols
                    if c not in T.PROP_EXCLUDE.get(ch["exclude"], ())]
        else:
            frame = sf
            cols = [c for c in st_cols if c not in ch.get("st_exclude", ())]
        cols = T._apply_keep(cname, cols)
        jobs.append(dict(name=cname, kind="count", frame=frame, cols=cols,
                         target=ch["target"], base=T.LGB_POIS,
                         profiles=CNT_PROFILES, mono=None,
                         tweedie=ch.get("tweedie"),
                         xgb_params=T.XGB_POIS, cb_params=T.CB_POIS,
                         n_xgb=T.XGB_BAGS, n_cb=T.CB_BAGS,
                         w_lgbm=T.COUNT_BAGS.get(cname, 1)))

    # team runs (count) + winner (binary) ---------------------------------
    tg_cols = T._apply_keep("total", F.team_game_feature_cols())
    jobs.append(dict(name="total", kind="count", frame=tg, cols=tg_cols,
                     target="y_runs", base=T.LGB_POIS, profiles=CNT_PROFILES,
                     mono=None, tweedie=None,
                     xgb_params=T.XGB_POIS, cb_params=T.CB_POIS,
                     n_xgb=T.XGB_BAGS, n_cb=T.CB_BAGS, w_lgbm=1))
    win_cols = T._apply_keep("winner", F.win_feature_cols())
    jobs.append(dict(name="winner", kind="binary", frame=wf, cols=win_cols,
                     target="y_home_win", base=T.LGB_WIN, profiles=WIN_PROFILES,
                     mono=None, tweedie=None,
                     xgb_params=T.XGB_WIN, cb_params=T.CB_WIN,
                     n_xgb=T.XGB_BAGS, n_cb=T.CB_BAGS, w_lgbm=1))
    return jobs


# priority order (user's dynamic priority: run-production + hit first)
HEAD_ORDER = ["run", "rbi", "hit", "hits2", "tb2", "hr", "single", "double",
              "bb", "sb", "bk", "bk2", "hrr2", "hrr3",
              "xhrr", "xtb", "xbk", "outs", "pbb", "pha", "per",
              "total", "winner"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="", help="comma list to restrict (else all)")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=120_000,
                    help="row cap per head for the big batter-frame heads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ensemble", action="store_true",
                    help="MeanBag each LGBM profile with the head's XGB+CatBoost "
                         "members (fit once per fold, cached) so the objective is "
                         "the ENSEMBLE OOF, not LGBM alone -- GPU + much slower. "
                         "Use at the next re-sweep now that the families ship.")
    ap.add_argument("--out", default=None,
                    help="output json (default prop_params_sweep[_ensemble].json)")
    args = ap.parse_args()

    want = {h.strip() for h in args.heads.split(",") if h.strip()}
    bf, sf, tg, wf = load_frames()
    train_yrs, cal_yr, hold_yr = T.suite_years(bf)
    cutoff = train_yrs[-1]     # selection suite may see Season <= this (2024)
    mode = (f"ENSEMBLE (LGBM + {T.XGB_BAGS} XGB + {T.CB_BAGS} CatBoost per fold)"
            if args.ensemble else "LGBM-only")
    log(f"selection window: Season <= {cutoff} (test {cal_yr} stays untouched); "
        f"folds={args.folds} max_rows={args.max_rows} | mode: {mode}")
    if args.ensemble and (T.XGB_BAGS + T.CB_BAGS) == 0:
        log("  ! --ensemble but XGB_BAGS=CB_BAGS=0 in train.py -> no families fit "
            "(falls back to LGBM-only); flip them on for a real ensemble run")
    out_path = Path(args.out) if args.out else ART / (
        f"prop_params_sweep{'_ensemble' if args.ensemble else ''}.json")

    jobs = {j["name"]: j for j in build_jobs(bf, sf, tg, wf)}
    order = [h for h in HEAD_ORDER if h in jobs and (not want or h in want)]

    results = {}
    for name in order:
        j = jobs[name]
        X, y, groups = _prep(j["frame"], j["cols"], j["target"], cutoff,
                             args.max_rows, args.seed)
        n_days = len(np.unique(groups))
        log(f"[{name}] {j['kind']} | rows {len(y):,} | days {n_days} | "
            f"cols {len(j['cols'])} | base_rate/mean "
            f"{y.mean():.4f} | sweeping {len(j['profiles'])} profiles x "
            f"{args.folds} folds")
        t0 = time.time()
        table = cv_head(j["kind"], X, y, groups, j["base"], j["profiles"],
                        args.folds, args.seed, mono_cols=j["mono"],
                        tweedie=j["tweedie"], ensemble=args.ensemble,
                        cat_cols=F.CAT_COLS, xgb_params=j["xgb_params"],
                        cb_params=j["cb_params"], n_xgb=j["n_xgb"],
                        n_cb=j["n_cb"], w_lgbm=j["w_lgbm"])
        rec = _recommend(j["kind"], table)
        d = table["default"]
        if j["kind"] == "binary":
            delta = {"logloss": round(table[rec]["logloss"] - d["logloss"], 5),
                     "auc": round(table[rec]["auc"] - d["auc"], 5)}
            tag = (f"ll {table[rec]['logloss']:.4f} (def {d['logloss']:.4f}, "
                   f"{delta['logloss']:+.4f}) | auc {table[rec]['auc']:.4f} "
                   f"({delta['auc']:+.4f})")
        else:
            delta = {"mae": round(table[rec]["mae"] - d["mae"], 5),
                     "deviance": round(table[rec]["deviance"] - d["deviance"], 5)}
            tag = (f"mae {table[rec]['mae']:.4f} (def {d['mae']:.4f}, "
                   f"{delta['mae']:+.4f}) | dev {delta['deviance']:+.5f}")
        results[name] = {"kind": j["kind"], "n_rows": int(len(y)),
                         "n_cols": len(j["cols"]), "tweedie": j["tweedie"],
                         "recommended": rec, "delta_vs_default": delta,
                         "profiles": table}
        star = "  <-- CHANGE" if rec != "default" else ""
        log(f"[{name}] DONE in {time.time()-t0:.0f}s | recommend '{rec}' | "
            f"{tag}{star}")
        # checkpoint after every head so a long run's partial results survive
        with open(out_path, "w") as f:
            json.dump({"cutoff": int(cutoff), "test_year_untouched": int(cal_yr),
                       "folds": args.folds, "max_rows": args.max_rows,
                       "ensemble": bool(args.ensemble),
                       "families": {"n_xgb": T.XGB_BAGS, "n_cb": T.CB_BAGS}
                       if args.ensemble else None,
                       "profiles": {"binary": BIN_PROFILES, "count": CNT_PROFILES,
                                    "winner": WIN_PROFILES},
                       "results": results}, f, indent=2)

    # summary ------------------------------------------------------------
    print("\n===== SWEEP SUMMARY (recommended profile per head) =====")
    changes = [n for n in order if results[n]["recommended"] != "default"]
    for name in order:
        r = results[name]
        d = r["delta_vs_default"]
        prim = f"{d.get('logloss', d.get('mae')):+.5f}"
        metric = "ll" if r["kind"] == "binary" else "mae"
        flag = " *" if r["recommended"] != "default" else ""
        print(f"  {name:8s} -> {r['recommended']:10s}  d{metric} {prim}{flag}")
    print(f"\n{len(changes)}/{len(order)} heads recommend a change: "
          f"{', '.join(changes) if changes else '(none)'}")
    print(f"wrote {out_path}")
    print("Review, then wire winners into train.PROP_PARAMS / count params and "
          "confirm on 2025 with evaluate_deep.py --paired.")


if __name__ == "__main__":
    main()
