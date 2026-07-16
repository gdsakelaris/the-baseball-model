"""Deep evaluation: statistical confidence, drift, segments, and betting realism.

The full accuracy + betting workup on a held-out season, answering the
questions a single accuracy table can't:

  1. Is the edge REAL?         bootstrap 95% CIs on AUC and the logloss edge
                               (resampling whole days, since games cluster)
  2. Is it CALIBRATED?         ECE + calibration slope per prop (slope ~1 is
                               ideal; <1 means overconfident)
  3. Where's the VALUE?        top-N sweep and pick-threshold tables with the
                               break-even American odds each hit rate implies
  4. Is it DRIFTING?           month-by-month actual-minus-predicted for every
                               prop (catches a changing run environment)
  5. Where does it WIN/LOSE?   segment breakdown (home/away, platoon, slot,
                               experience, temperature, arsenal coverage)
  6. Are picks CORRELATED?     how many distinct games the daily top-10 spans
  7. K / totals / winner       P(over) line calibration, Poisson dispersion
                               checks, win-prob calibration, and significance
                               of the winner model vs always-picking-home
  8. Did a change HELP?        --set-baseline snapshots the headline metrics;
                               later runs print a per-metric better/worse/same
                               diff so a retrain's effect is visible at a glance

Usage:
    python Model/evaluate_deep.py [--prop hr] [--boot 400]   # SELECTION suite
    python Model/evaluate_deep.py --confirm                  # shipping suite

The years come from the trained artifacts (train.py derives them from the
data, e.g. selection on 2025 / holdout 2026 while 2026 is the newest
season). The DEFAULT run scores the selection suite on its test year — safe
to run as often as you like. The newest season is CONFIRM-ONLY: iterating
features/params against its numbers quietly overfits the holdout, so
looking at it takes an explicit --confirm. The iteration loop is:

    python Model/evaluate_deep.py --set-baseline   # snapshot before changing
    ...make a change...
    python Model/train.py --rebuild --select       # retrain the selection suite
    python Model/evaluate_deep.py                  # Section 11 diff, selection year
    ...iterate; only when satisfied:
    python Model/train.py                          # both suites
    python Model/evaluate_deep.py --confirm        # ONE confirming holdout look
"""

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import odds as O  # noqa: E402
import recalibrate as R  # noqa: E402
from predict import (american_odds, apply_stack, count_over,  # noqa: E402
                     k_over, total_over,
                     nb_over, poisson_over, poisson_win, predict_prop,
                     predict_win)
from train import PROPS  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"

# How the --set-baseline diff judges each metric's movement, matched by key
# suffix. Anything not listed here is "higher is better" (AUC, edge, top10,
# winner_acc). Keeps the verdict honest for metrics where up isn't good.
LOWER_BETTER = ("_ece", "_mae", "_logloss")   # smaller = better
TARGET_ONE = ("_dispersion",)                 # ideal is 1.0; closer = better

# Retrain-to-retrain noise bands, matched by key suffix. Two runs of train.py
# on IDENTICAL code differ by this much anyway (LightGBM bagging draws depend
# on row order — train.py documents ~0.005-0.01 MAE from row order alone — and
# the daily job adds a day of data), so a delta inside the band says nothing
# about whether a change helped. Sized to observed jitter: top-10 daily hit
# rate on ~1,000 picks has a ±1.4pp binomial SE before any model change.
NOISE_BAND = {"_auc": 0.004, "_edge": 0.001, "_ece": 0.003, "_top10": 0.02,
              "_mae": 0.008, "_dispersion": 0.02, "_acc": 0.008,
              "_logloss": 0.002}
# exact-key overrides where the metric's scale differs from its suffix class
# (outs run ~16.5 a start, hits allowed ~5 — their MAE jitter is larger than
# the K model's; provisional sizes, tune after a few retrains)
NOISE_BAND_EXACT = {"outs_mae": 0.03, "pha_mae": 0.015, "xhrr_mae": 0.01,
                    "per_mae": 0.015, "xtb_mae": 0.01}


def noise_band(metric):
    if metric in NOISE_BAND_EXACT:
        return NOISE_BAND_EXACT[metric]
    for suf, band in NOISE_BAND.items():
        if metric.endswith(suf):
            return band
    return 5e-5


def prep(df, cols, cat_levels):
    X = df[cols].copy()
    for c, levels in cat_levels.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
    return X


def wilson(k, n, z=1.96):
    """95% CI for a proportion (Wilson interval)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def ece(p, y, bins=10):
    """Expected calibration error: pick-weighted |predicted - actual| by decile."""
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True)
    tab = g.agg(pred=("p", "mean"), act=("y", "mean"), n=("y", "size"))
    return float((tab["n"] * (tab["pred"] - tab["act"]).abs()).sum() / tab["n"].sum())


def calibration_slope(p, y):
    """Logistic fit of outcome on logit(pred). Slope 1 = perfect; <1 means the
    model's extremes are too extreme (overconfident); >1 too timid."""
    z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0][0])


def day_bootstrap(dates, p, y, n_boot, seed=0):
    """Resample whole DAYS with replacement -> 95% CIs for AUC and for the
    logloss edge over the base rate. Day-level blocks respect the fact that
    outcomes within a slate share weather/park/lineup context."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(dates)
    idx_by_day = {d: np.flatnonzero(dates == d) for d in uniq}
    aucs, edges = [], []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in take])
        yy, pp = y[idx], p[idx]
        if yy.min() == yy.max():
            continue
        aucs.append(roc_auc_score(yy, pp))
        base = np.full_like(pp, yy.mean())
        edges.append(log_loss(yy, base) - log_loss(yy, pp))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    elo, ehi = np.percentile(edges, [2.5, 97.5])
    return (lo, hi), (elo, ehi)


# ------------------------------------------------------------ prop sections


def section_confidence(results, n_boot):
    print(f"\n=== 1. Statistical confidence (day-block bootstrap, "
          f"{n_boot} resamples) ===")
    rows = []
    for name, r in results.items():
        (alo, ahi), (elo, ehi) = day_bootstrap(
            r["d"], r["p"], r["y"], n_boot)
        rows.append({
            "prop": name,
            "AUC [95% CI]": f'{r["auc"]:.3f} [{alo:.3f}, {ahi:.3f}]',
            "AUC>0.5": "yes" if alo > 0.5 else "NO",
            "logloss edge [95% CI]": f'{r["edge"]:+.4f} [{elo:+.4f}, {ehi:+.4f}]',
            "edge>0": "yes" if elo > 0 else "NO",
            "ECE": f'{ece(r["p"], r["y"]):.4f}',
            "cal slope": f'{calibration_slope(r["p"], r["y"]):.2f}',
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("  edge = base-rate logloss minus model logloss (bigger = better).")
    print("  If any CI column says NO, treat that prop's edge as unproven.")


def section_topn(results):
    print("\n=== 2. Daily top-N sweep — hit rate [95% CI] and lift ===")
    rows = []
    for name, r in results.items():
        day = pd.DataFrame({"d": r["d"], "p": r["p"], "y": r["y"]})
        row = {"prop": name, "base": f'{r["y"].mean():.1%}'}
        for n in (1, 3, 5, 10, 20):
            top = day.sort_values("p", ascending=False).groupby("d").head(n)
            k, m = int(top["y"].sum()), len(top)
            lo, hi = wilson(k, m)
            row[f"top{n}"] = (f'{k / m:.1%} [{lo:.1%},{hi:.1%}] '
                              f'{k / m / r["y"].mean():.2f}x')
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("  Read: does the edge shrink from top-1 to top-20? A model whose "
          "top-1 isn't its best is ranking noise at the extreme.")


def section_thresholds(results, prop):
    r = results[prop]
    print(f"\n=== 3. Pick-threshold table, {prop} — if you bet everything "
          f"above a cutoff ===")
    order = np.argsort(-r["p"])
    p, y = r["p"][order], r["y"][order]
    rows = []
    for pct in (1, 2, 5, 10, 20):
        n = max(1, int(len(p) * pct / 100))
        hit = y[:n].mean()
        lo, hi = wilson(int(y[:n].sum()), n)
        rows.append({
            "picks": f"top {pct}%", "n": n,
            "min model prob": f"{p[n - 1]:.3f}",
            "mean model prob": f"{p[:n].mean():.3f}",
            "actual hit rate": f"{hit:.1%} [{lo:.1%}, {hi:.1%}]",
            "break-even odds at actual": american_odds(hit),
            "model fair odds": american_odds(p[:n].mean()),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("  'break-even odds at actual' is the worst sportsbook price at "
          "which these picks\n  historically profit. Compare it to the "
          "prices actually offered for this prop.")


def section_drift(results):
    print("\n=== 4. Monthly drift — actual minus predicted, percentage points ===")
    mats = {}
    for name, r in results.items():
        m = pd.DataFrame({"month": pd.to_datetime(r["d"]).month,
                          "p": r["p"], "y": r["y"]})
        g = m.groupby("month")
        mats[name] = ((g["y"].mean() - g["p"].mean()) * 100).round(1)
    print(pd.DataFrame(mats).T.to_string())
    print("  Positive = model UNDER-predicts that month (environment hotter "
          "than history);\n  a growing positive trend means the league has "
          "drifted since training.")

    print("\n--- AUC by month (ranking skill over the season) ---")
    mats = {}
    for name, r in results.items():
        m = pd.DataFrame({"month": pd.to_datetime(r["d"]).month,
                          "p": r["p"], "y": r["y"]})
        aucs = {}
        for month, x in m.groupby("month"):
            aucs[month] = (roc_auc_score(x["y"], x["p"])
                           if x["y"].nunique() > 1 else np.nan)
        mats[name] = pd.Series(aucs).round(3)
    print(pd.DataFrame(mats).T.to_string())


def section_segments(bf_y, results, prop):
    r = results[prop]
    df = bf_y.reset_index(drop=True)
    p, y = r["p"], r["y"]
    temp = pd.to_numeric(df["Temp"], errors="coerce")

    segs = [
        ("home", df["Home"] == 1), ("away", df["Home"] == 0),
        ("day game", df["DayNight"].astype(str) == "day"),
        ("night game", df["DayNight"].astype(str) == "night"),
        ("platoon edge (opp hand)", df["same_hand"] == 0),
        ("same hand", df["same_hand"] == 1),
        ("slot 1-3", df["slot"] <= 3), ("slot 4-6", df["slot"].between(4, 6)),
        ("slot 7-9", df["slot"] >= 7),
        ("rookie-ish (<50 career G)", df["g_career"] < 50),
        ("50-199 career G", df["g_career"].between(50, 199)),
        ("veteran (200+ G)", df["g_career"] >= 200),
        ("temp <60F", temp < 60), ("temp 60-75F", temp.between(60, 75)),
        ("temp >75F", temp > 75),
        ("arsenal matchup known", df["m_coverage"].fillna(0) > 0.5),
        ("arsenal matchup missing", df["m_coverage"].fillna(0) <= 0.5),
    ]
    print(f"\n=== 5. Segment breakdown, {prop} — where the model wins/loses ===")
    rows = []
    for label, mask in segs:
        mask = mask.fillna(False).to_numpy()
        n = int(mask.sum())
        if n < 200 or y[mask].min() == y[mask].max():
            continue
        base = np.full(n, y[mask].mean(), dtype=float)
        rows.append({
            "segment": label, "n": n, "base": f"{y[mask].mean():.1%}",
            "pred": f"{p[mask].mean():.1%}",
            "AUC": f"{roc_auc_score(y[mask], p[mask]):.3f}",
            "logloss edge": f"{log_loss(y[mask], base) - log_loss(y[mask], p[mask]):+.4f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("  Segments where AUC ~0.5 or pred is far from base are where NOT "
          "to trust picks.")


def section_concentration(results, prop="hr", top_n=10):
    r = results[prop]
    day = pd.DataFrame({"d": r["d"], "g": r["g"], "p": r["p"], "y": r["y"]})
    top = day.sort_values("p", ascending=False).groupby("d").head(top_n)
    per_day = top.groupby("d").agg(
        games=("g", "nunique"), biggest=("g", lambda s: s.value_counts().iloc[0]))
    print(f"\n=== 6. Slate concentration — daily top-{top_n} {prop} picks ===")
    print(f"  distinct games spanned: mean {per_day['games'].mean():.1f} | "
          f"min {per_day['games'].min()}")
    print(f"  days where one game supplies 4+ picks: "
          f"{(per_day['biggest'] >= 4).mean():.0%}")
    print("  Correlated picks (same game) win and lose together — size bets "
          "accordingly.")


# --------------------------------------------------- pitcher/game sections


def section_strikeouts(sf_y, art):
    Xs = prep(sf_y, art["st_cols"], art["cat_levels"])
    mu = art["k_model"].predict(Xs)
    y = sf_y["y_so"].to_numpy().astype(float)
    resid = y - mu
    print(f"\n=== 7. Starter strikeouts deep dive ({len(y):,} starts) ===")
    print(f"  MAE {mean_absolute_error(y, mu):.3f} | RMSE "
          f"{math.sqrt(np.mean(resid ** 2)):.3f} | bias (pred-actual) "
          f"{mu.mean() - y.mean():+.3f}")
    disp = float(np.mean(resid ** 2) / np.mean(mu))
    print(f"  dispersion index {disp:.2f} (Poisson assumes 1.00; >1 means "
          f"real K counts are wilder\n  than the model's P(over) math "
          f"assumes -> over/under probs too confident)")

    m = pd.DataFrame({"month": pd.to_datetime(sf_y["Date"]).dt.month.values,
                      "mu": mu, "y": y})
    g = m.groupby("month").agg(n=("y", "size"), pred=("mu", "mean"),
                               actual=("y", "mean"))
    g["bias"] = (g["pred"] - g["actual"]).round(2)
    print("\n--- monthly bias ---")
    print(g.round(2).to_string())

    t = pd.qcut(mu, 3, labels=["low-K pred", "mid", "high-K pred"])
    tiers = pd.DataFrame({"tier": t, "mu": mu, "y": y})
    print("\n--- by predicted-K tier ---")
    rows = []
    for tier, d in tiers.groupby("tier", observed=True):
        rows.append({"tier": tier, "n": len(d),
                     "pred": round(d["mu"].mean(), 2),
                     "actual": round(d["y"].mean(), 2),
                     "mae": round(mean_absolute_error(d["y"], d["mu"]), 2)})
    print(pd.DataFrame(rows).to_string(index=False))

    k_disp = float(art.get("k_disp", 1.0))
    dist = ("cal-year per-line logit calibrators" if art.get("k_line_cals")
            else f"negative binomial, disp {k_disp:.2f} from cal year"
            if k_disp > 1.001 else "Poisson")
    print(f"\n--- P(over) line calibration (the sellable output; uses {dist}) ---")
    rows = []
    for line in (3.5, 4.5, 5.5, 6.5, 7.5):
        p_over = k_over(art, mu, line)
        actual = (y > line).astype(int)
        rows.append({"line": line, "mean P(over)": f"{p_over.mean():.3f}",
                     "actual over rate": f"{actual.mean():.3f}",
                     "logloss": f"{log_loss(actual, np.clip(p_over, 1e-4, 1 - 1e-4)):.4f}",
                     "base logloss": f"{log_loss(actual, np.full_like(p_over, actual.mean())):.4f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    return {"k_mae": round(float(mean_absolute_error(y, mu)), 4),
            "k_dispersion": round(float(disp), 4)}


def section_count_heads(bf_y, sf_y, art):
    """Count-prop heads (batter Ks, H+R+RBI, starter outs/walks/hits):
    mean quality, dispersion, and NB line calibration — the starter-K deep
    dive generalized. Batter HALF-POINT lines are priced in serving by the
    calibrated binary heads (Section 1 grades those); these tables check the
    count means (xSO/xHRR/xOuts/...) and their P(over) distributions."""
    cm = art.get("count_models")
    if not cm:
        return {}
    print("\n=== 7b. Count props — means, dispersion, line calibration ===")
    out = {}
    for name, head in cm.items():
        d = bf_y if head["frame"] == "bat" else sf_y
        X = prep(d, head["cols"], art["cat_levels"])
        mu = head["model"].predict(X)
        y = d[head["target"]].to_numpy().astype(float)
        disp_obs = float(np.mean((y - mu) ** 2) / np.mean(mu))
        print(f"\n--- {name} ({head['desc']}; {len(y):,} rows) ---")
        print(f"  MAE {mean_absolute_error(y, mu):.3f} | bias (pred-actual) "
              f"{mu.mean() - y.mean():+.3f} | observed dispersion "
              f"{disp_obs:.2f} | model disp {head['disp']:.2f} (cal year)")
        rows = []
        for line in head["lines"]:
            p_over = count_over(head, mu, line)
            actual = (y > line).astype(int)
            if actual.min() == actual.max():
                continue
            rows.append({
                "line": line,
                "cal": "logit" if line in head.get("line_cals", {}) else "nb",
                "mean P(over)": f"{p_over.mean():.3f}",
                "actual over rate": f"{actual.mean():.3f}",
                "logloss": f"{log_loss(actual, np.clip(p_over, 1e-4, 1 - 1e-4)):.4f}",
                "base logloss": f"{log_loss(actual, np.full_like(p_over, actual.mean())):.4f}"})
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
        out[f"{name}_mae"] = round(float(mean_absolute_error(y, mu)), 4)
        out[f"{name}_dispersion"] = round(disp_obs, 4)
    print("\n  P(over) per line comes from a cal-year logistic on mu "
          "('logit' — handles the\n  under-Poisson variance of bounded "
          "counts like outs and batter Ks), falling\n  back to nb_over "
          "('nb') where a line had no calibration data. For batter\n  "
          "half-point lines the binary heads (Section 1) stay the sellable "
          "numbers.")
    return out


def section_h1_bars(results, bf_y, art):
    """H1 acceptance bars (2026-07-14): each deep binary head must BEAT the
    banked count-calibrator pricing of the same line (count_vs_binary
    verdict — a loser ships count-priced, a measured mixed board is fine).
    Recomputed LIVE on this year's rows: the count head's banked per-line
    calibrator vs the binary head, same rows, logloss + AUC."""
    pairs = [("bk3", "xbk", 2.5), ("tb3", "xtb", 2.5),
             ("tb4", "xtb", 3.5), ("hrr4", "xhrr", 3.5)]
    cm = art.get("count_models", {})
    have = [(b, c, ln) for b, c, ln in pairs
            if b in results and c in cm and ln in cm[c].get("line_cals", {})]
    if not have:
        return
    print("\n=== 7c. H1 acceptance bars: deep binaries vs their banked "
          "count calibrators ===")
    rows = []
    for bname, cname, line in have:
        head = cm[cname]
        d = bf_y
        mu = head["model"].predict(prep(d, head["cols"], art["cat_levels"]))
        p_cal = count_over(head, mu, line)
        r = results[bname]
        y = r["y"]
        p_bin = np.clip(r["p"], 1e-4, 1 - 1e-4)
        p_cal = np.clip(p_cal, 1e-4, 1 - 1e-4)
        ll_b, ll_c = log_loss(y, p_bin), log_loss(y, p_cal)
        auc_b, auc_c = roc_auc_score(y, p_bin), roc_auc_score(y, p_cal)
        rows.append({"head": bname, "vs": f"{cname}>{line}",
                     "logloss bin": f"{ll_b:.5f}", "logloss cal": f"{ll_c:.5f}",
                     "AUC bin": f"{auc_b:.4f}", "AUC cal": f"{auc_c:.4f}",
                     "verdict": ("BINARY wins" if ll_b < ll_c and auc_b > auc_c
                                 else "count wins" if ll_c < ll_b and auc_c > auc_b
                                 else "MIXED")})
    print(pd.DataFrame(rows).to_string(index=False))
    print("  A head that loses BOTH metrics ships count-priced instead "
          "(measured mixed board, per the 07-13 verdict).")


def section_coherence(bf_y, gf_y, art):
    """H6 coherence read (2026-07-14): the lineup's summed expected runs
    (xrun, batter grain) against the team-runs model's per-team mean (team
    grain) — two grains, one quantity; a large systematic gap flags a
    frame-plumbing or dispersion problem neither grain sees alone."""
    cm = art.get("count_models", {})
    if "xrun" not in cm or art.get("team_runs_model") is None:
        return
    head = cm["xrun"]
    mu = head["model"].predict(prep(bf_y, head["cols"], art["cat_levels"]))
    lu = (pd.DataFrame({"GamePk": bf_y["GamePk"].to_numpy(),
                        "Team": bf_y["Team"].to_numpy(), "xrun": mu})
          .groupby(["GamePk", "Team"], as_index=False)["xrun"].sum())
    tg = F.build_team_game_frame(gf_y)
    tp = art["team_runs_model"].predict(prep(tg, art["tg_cols"],
                                             art["cat_levels"]))
    n_g = len(gf_y)
    team = pd.concat([
        pd.DataFrame({"GamePk": gf_y["GamePk"], "Team": gf_y["AwayTeam"],
                      "mu_team": tp[:n_g]}),
        pd.DataFrame({"GamePk": gf_y["GamePk"], "Team": gf_y["HomeTeam"],
                      "mu_team": tp[n_g:]})])
    m = lu.merge(team, on=["GamePk", "Team"])
    if not len(m):
        return
    diff = m["xrun"] - m["mu_team"]
    corr = float(np.corrcoef(m["xrun"], m["mu_team"])[0, 1])
    print("\n=== 7d. Coherence: sum(lineup xrun) vs team_total mean ===")
    print(f"  {len(m):,} team-games | mean gap (lineup - team) "
          f"{diff.mean():+.3f} runs | SD {diff.std():.3f} | corr {corr:.3f}")
    print("  (lineup xrun omits bench/pinch runs, so a small negative gap "
          "is expected; watch for drift or low corr)")


def section_games(gf_y, art):
    tg = F.build_team_game_frame(gf_y)
    Xt = prep(tg, art["tg_cols"], art["cat_levels"])
    tp = art["team_runs_model"].predict(Xt)
    n_g = len(gf_y)
    mu_away, mu_home = tp[:n_g], tp[n_g:]
    away_y = pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy()
    home_y = pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy()
    total_mu, total_y = mu_away + mu_home, away_y + home_y

    print(f"\n=== 8. Totals & winner deep dive ({n_g:,} games) ===")
    resid = total_y - total_mu
    disp = float(np.mean(resid ** 2) / np.mean(total_mu))
    model_disp = float(art.get("total_disp", 1.0))
    dist = ("cal-year per-line logit calibrators"
            if art.get("total_line_cals")
            else f"negative binomial, disp {model_disp:.2f} from cal year"
            if model_disp > 1.001 else "Poisson")
    print(f"  totals bias (pred-actual) {total_mu.mean() - total_y.mean():+.2f} "
          f"runs | observed dispersion {disp:.2f} | P(over) uses {dist}")

    print("\n--- P(over) run-line calibration ---")
    rows = []
    for line in (6.5, 7.5, 8.5, 9.5, 10.5):
        p_over = total_over(art, total_mu, line)
        actual = (total_y > line).astype(int)
        rows.append({"line": line, "mean P(over)": f"{p_over.mean():.3f}",
                     "actual over rate": f"{actual.mean():.3f}",
                     "logloss": f"{log_loss(actual, np.clip(p_over, 1e-4, 1 - 1e-4)):.4f}",
                     "base logloss": f"{log_loss(actual, np.full_like(p_over, actual.mean())):.4f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # H5 team_total (2026-07-14): the per-TEAM line surface off the same
    # per-team means — its own dispersion + calibrators (the game total's
    # do not transfer)
    if art.get("team_line_cals") or art.get("team_total_disp"):
        from predict import team_over, TEAM_TOTAL_LINES
        team_mu = np.concatenate([mu_away, mu_home])
        team_y = np.concatenate([away_y, home_y])
        t_disp = float(np.mean((team_y - team_mu) ** 2) / np.mean(team_mu))
        print(f"\n--- team_total (per-team lines): observed dispersion "
              f"{t_disp:.2f} | model disp "
              f"{float(art.get('team_total_disp', 1.0)):.2f} (cal year) ---")
        rows = []
        for line in TEAM_TOTAL_LINES:
            p_over = team_over(art, team_mu, line)
            actual = (team_y > line).astype(int)
            rows.append({
                "line": line,
                "cal": "logit" if line in art.get("team_line_cals", {})
                       else "nb",
                "mean P(over)": f"{p_over.mean():.3f}",
                "actual over rate": f"{actual.mean():.3f}",
                "logloss": f"{log_loss(actual, np.clip(p_over, 1e-4, 1 - 1e-4)):.4f}",
                "base logloss": f"{log_loss(actual, np.full_like(p_over, actual.mean(), dtype=float)):.4f}"})
        print(pd.DataFrame(rows).to_string(index=False))

    # winner: dedicated model when the artifact has one
    win_cols = art.get("win_model", {}).get("cols", [])
    if _win_scoreable(win_cols, gf_y):
        _ensure_persp(gf_y, win_cols)
        Xw = prep(gf_y, win_cols, art["cat_levels"])
        p_home = predict_win(art["win_model"], Xw, mu_home, mu_away)
        print("\n  (winner probabilities from the dedicated win model)")
    else:
        p_home = np.array([poisson_win(h, a) for h, a in zip(mu_home, mu_away)])
        print("\n  (winner probabilities from Poisson means — no win model "
              "in artifacts)")
    actual_home = (home_y > away_y).astype(int)
    ll = log_loss(actual_home, np.clip(p_home, 1e-4, 1 - 1e-4))
    ll_base = log_loss(actual_home, np.full_like(p_home, actual_home.mean(),
                                                 dtype=float))
    print("\n--- winner: PROBABILITY quality (the intended output) ---")
    print(f"  win-prob log loss {ll:.4f} vs always-home base rate "
          f"{ll_base:.4f}  (edge {ll_base - ll:+.4f})")
    print("  The product is a calibrated home-win probability, NOT a side to "
          "bet.")

    model_pick = (p_home >= 0.5).astype(int)
    model_right = model_pick == actual_home
    home_right = actual_home == 1
    acc = model_right.mean()
    lo, hi = wilson(int(model_right.sum()), len(model_right))
    b = int((model_right & ~home_right).sum())
    c = int((~model_right & home_right).sum())
    if b + c == 0:
        pval = 1.0
    else:
        try:
            pval = stats.binomtest(b, b + c).pvalue
        except AttributeError:  # scipy < 1.7
            pval = stats.binom_test(b, b + c)
    print(f"\n  [reference only, not a bet] straight-up pick accuracy "
          f"{acc:.1%} [{lo:.1%}, {hi:.1%}] vs always-home "
          f"{home_right.mean():.1%};")
    print(f"  beats the home pick on {b} games, loses on {c}, McNemar "
          f"p={pval:.3f} -> {'no' if pval > 0.05 else 'some'} evidence of an "
          f"actual moneyline edge.")

    q = pd.qcut(p_home, 5, duplicates="drop")
    tab = pd.DataFrame({"p": p_home, "y": actual_home}).groupby(q, observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    print("\n--- win-prob calibration (quintiles: predicted should ~ actual) ---")
    print(tab.round(3).to_string())

    return {
        "total_mae": round(float(mean_absolute_error(total_y, total_mu)), 4),
        "total_dispersion": round(float(disp), 4),
        "winner_acc": round(float(acc), 4),
        "winner_logloss": round(
            float(log_loss(actual_home, np.clip(p_home, 1e-4, 1 - 1e-4))), 4),
    }


# --------------------------------------------------------- vs. the market


def _dayblock_roi_ci(days, profit, n_boot, seed=0):
    """Day-block bootstrap 95% CI for ROI (mean profit per bet): resample whole
    days, since a slate's bets share weather/park/pitcher context."""
    rng = np.random.default_rng(seed)
    uniq = pd.unique(days)
    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    rois = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in take])
        if len(idx):
            rois.append(profit[idx].mean())
    if not rois:
        return float("nan"), float("nan")
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return lo, hi


def _market_consensus(store, api, line):
    """Per (Date, PlayerId): de-vig each book's two-sided price to a fair
    P(over), then keep the reference fair prob (Pinnacle's where it posts the
    line, else the median — odds.sharp_fair, shared with MktEdge% and the Bets
    sheet) plus the best (most generous) over/under price across books for ROI
    line-shopping."""
    m = store[(store["Market"] == api) & ((store["Line"] - line).abs() < 1e-6)]
    recs = []
    for _, r in m.iterrows():
        fo, hold = O.devig_two_way(r["OverPrice"], r["UnderPrice"])
        if np.isnan(fo):
            continue
        recs.append((r["Date"], r["PlayerId"], fo, hold,
                     r["OverPrice"], r["UnderPrice"], r["Book"]))
    cols = ["Date", "PlayerId", "fair", "hold", "best_over", "best_under",
            "n_books"]
    if not recs:
        return pd.DataFrame(columns=cols)
    md = pd.DataFrame(recs, columns=["Date", "PlayerId", "fair", "hold",
                                     "over", "under", "Book"])
    out = md.groupby(["Date", "PlayerId"], dropna=False).agg(
        hold=("hold", "median"),
        best_over=("over", "max"), best_under=("under", "max"),
        n_books=("fair", "size"))
    out["fair"] = md.groupby(["Date", "PlayerId"], dropna=False)[
        ["fair", "Book"]].apply(O.sharp_fair)
    return out.reset_index()[cols]


def _grade_against_market(name, model, cons, n_boot, key="PlayerId"):
    """Join a model's per-event P(over) to the de-vigged market consensus and
    grade it: model vs market log loss on the shared events, plus flat-1u ROI
    on every side the model prices +EV at the best book price. `model` carries
    Date, the join key (`key`: PlayerId for player props, Team for game
    markets), p (model P(over)) and y (0/1 outcome). Returns a summary row
    dict, or None if fewer than 20 events join."""
    cons = cons.copy()
    if key == "PlayerId":
        cons["PlayerId"] = pd.to_numeric(cons["PlayerId"], errors="coerce")
    j = model.merge(cons, on=["Date", key], how="inner")
    if len(j) < 20:
        return None
    yv = j["y"].to_numpy()
    mp = np.clip(j["p"].to_numpy(), 1e-4, 1 - 1e-4)
    fq = np.clip(j["fair"].to_numpy(), 1e-4, 1 - 1e-4)
    both = len(np.unique(yv)) > 1
    m_ll = log_loss(yv, mp) if both else float("nan")
    k_ll = log_loss(yv, fq) if both else float("nan")
    # flat 1u on every side the model prices +EV, at the best book price,
    # settled at the real outcome
    profit, bdays = [], []
    for _, row in j.iterrows():
        side, _ev = O.pick_side(row["p"], row["best_over"], row["best_under"])
        if side is None:
            continue
        price = row["best_over"] if side == "over" else row["best_under"]
        won = (row["y"] == 1) if side == "over" else (row["y"] == 0)
        profit.append(O.settle(price, bool(won)))
        bdays.append(row["Date"])
    if profit:
        profit, days = np.array(profit), np.array(bdays)
        lo, hi = _dayblock_roi_ci(days, profit, n_boot)
        roi_s = f"{profit.mean():+.1%} [{lo:+.1%},{hi:+.1%}]"
    else:
        roi_s = "—"
    holds = j["hold"].dropna()
    return {
        "prop": name, "events": len(j),
        "med hold": f"{holds.median():.1%}" if len(holds) else "—",
        "model ll": f"{m_ll:.4f}" if both else "—",
        "market ll": f"{k_ll:.4f}" if both else "—",
        "model<mkt": ("yes" if (both and m_ll < k_ll) else "no"),
        "+EV bets": len(profit),
        "flat ROI [95% CI]": roi_s,
    }


def _starter_market_rows(sf_y, art, store, n_boot):
    """Section 9 for the pitcher count props (strikeouts, outs, walks, hits,
    earned runs). These are multi-line, so each line the book posts is graded
    on its own: the model's P(over line) comes from the K model (nb_over) or
    the matching count head (count_over), the outcome is the real start's count
    > line, joined to the market on (Date, PlayerId). Prop labels are the
    scraper's keys plus the line, e.g. 'pk 6.5', 'per 2.5'."""
    if sf_y is None or not len(sf_y):
        return []
    date = pd.to_datetime(sf_y["Date"]).dt.date.to_numpy()
    pid = pd.to_numeric(sf_y["PlayerId"], errors="coerce").to_numpy()
    # api market -> (mu, actual-count array, P(over line) fn)
    providers = {}
    if art.get("k_model") is not None and "y_so" in sf_y.columns:
        mu_k = art["k_model"].predict(prep(sf_y, art["st_cols"], art["cat_levels"]))
        providers["pitcher_strikeouts"] = (
            mu_k, sf_y["y_so"].to_numpy().astype(float),
            lambda mu, line: k_over(art, mu, line))
    cm = art.get("count_models", {})
    api_of = {"outs": "pitcher_outs", "pbb": "pitcher_walks",
              "pha": "pitcher_hits_allowed", "per": "pitcher_earned_runs"}
    for cname, api in api_of.items():
        head = cm.get(cname)
        if head is None or head.get("frame") != "starts":
            continue
        mu = head["model"].predict(prep(sf_y, head["cols"], art["cat_levels"]))
        y = sf_y[head["target"]].to_numpy().astype(float)
        providers[api] = (mu, y, (lambda h: lambda mu, line:
                                  count_over(h, mu, line))(head))
    key_of = {m["api"]: k for k, m in O.STARTER_MARKET.items()}
    rows = []
    for api, (mu, yv, prob_fn) in providers.items():
        lines = sorted({float(l) for l in
                        store.loc[store["Market"] == api, "Line"].dropna()})
        for line in lines:
            cons = _market_consensus(store, api, line)
            if cons.empty:
                continue
            model = pd.DataFrame({"Date": date, "PlayerId": pid,
                                  "p": prob_fn(mu, line),
                                  "y": (yv > line).astype(int)})
            row = _grade_against_market(f"{key_of.get(api, api)} {line:g}",
                                        model, cons, n_boot)
            if row:
                rows.append(row)
    return rows


def _game_consensus(store, api, line):
    """Like _market_consensus but for game markets, keyed on (Date, Team=home)
    since game rows carry no PlayerId. de-vig gives fair P(over) for totals and
    fair P(home) for h2h (the store puts OverPrice=home, UnderPrice=away).
    `line` is None for h2h (a moneyline has no point)."""
    m = store[store["Market"] == api]
    if line is not None:
        m = m[(m["Line"] - line).abs() < 1e-6]
    recs = []
    for _, r in m.iterrows():
        fo, hold = O.devig_two_way(r["OverPrice"], r["UnderPrice"])
        if np.isnan(fo):
            continue
        recs.append((r["Date"], r["Team"], fo, hold,
                     r["OverPrice"], r["UnderPrice"], r["Book"]))
    cols = ["Date", "Team", "fair", "hold", "best_over", "best_under", "n_books"]
    if not recs:
        return pd.DataFrame(columns=cols)
    md = pd.DataFrame(recs, columns=["Date", "Team", "fair", "hold",
                                     "over", "under", "Book"])
    out = md.groupby(["Date", "Team"], dropna=False).agg(
        hold=("hold", "median"),
        best_over=("over", "max"), best_under=("under", "max"),
        n_books=("fair", "size"))
    out["fair"] = md.groupby(["Date", "Team"], dropna=False)[
        ["fair", "Book"]].apply(O.sharp_fair)
    return out.reset_index()[cols]


def _game_market_rows(gf_y, art, store, n_boot):
    """Section 9 for the game markets. totals: model P(over runs) from the
    team-runs Poisson (nb_over, cal-year dispersion). h2h: model P(home win)
    from the win model — betting the 'over' side backs the home team. Both join
    to the game holdout on (Date, home team) since game rows carry no PlayerId."""
    if gf_y is None or not len(gf_y):
        return []
    if store[store["Market"].isin(("totals", "h2h"))].empty:
        return []
    tg = F.build_team_game_frame(gf_y)
    tp = art["team_runs_model"].predict(prep(tg, art["tg_cols"], art["cat_levels"]))
    n = len(gf_y)
    mu_away, mu_home = tp[:n], tp[n:]
    total_mu = mu_away + mu_home
    disp = float(art.get("total_disp", 1.0))
    home_y = pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy()
    away_y = pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy()
    date = pd.to_datetime(gf_y["Date"]).dt.date.to_numpy()
    team = gf_y["HomeTeam"].to_numpy()
    total_y = home_y + away_y
    rows = []
    for line in sorted({float(l) for l in
                        store.loc[store["Market"] == "totals", "Line"].dropna()}):
        cons = _game_consensus(store, "totals", line)
        if cons.empty:
            continue
        model = pd.DataFrame({
            "Date": date, "Team": team,
            "p": total_over(art, total_mu, line),
            "y": (total_y > line).astype(int)})
        row = _grade_against_market(f"totals {line:g}", model, cons, n_boot,
                                    key="Team")
        if row:
            rows.append(row)
    win = art.get("win_model", {})
    if not store[store["Market"] == "h2h"].empty \
            and _win_scoreable(win.get("cols"), gf_y):
        _ensure_persp(gf_y, win["cols"])
        p_home = predict_win(win, prep(gf_y, win["cols"], art["cat_levels"]),
                             mu_home, mu_away)
        cons = _game_consensus(store, "h2h", None)
        if not cons.empty:
            model = pd.DataFrame({"Date": date, "Team": team, "p": p_home,
                                  "y": (home_y > away_y).astype(int)})
            row = _grade_against_market("h2h (home)", model, cons, n_boot,
                                        key="Team")
            if row:
                rows.append(row)
    return rows


def _batter_count_market_rows(bf_y, art, store, n_boot):
    """Section 9 for batter count-head lines the binary props DON'T cover —
    3+/4+ total bases (xtb), 4+ H+R+RBI (xhrr). P(over line) from the count
    head's count_over; any line already owned by a binary prop (in PROP_MARKET)
    is skipped, so each line is graded once by its better-calibrated head."""
    cm = art.get("count_models", {})
    if bf_y is None or not len(bf_y) or not cm:
        return []
    bat_api = {"xtb": "batter_total_bases", "xhrr": "batter_hits_runs_rbis",
               "xbk": "batter_strikeouts"}
    owned = {(m["api"], m["line"]) for m in O.PROP_MARKET.values()}
    date = pd.to_datetime(bf_y["Date"]).dt.date.to_numpy()
    pid = pd.to_numeric(bf_y["PlayerId"], errors="coerce").to_numpy()
    rows = []
    for cname, api in bat_api.items():
        head = cm.get(cname)
        if head is None or head.get("frame") != "bat":
            continue
        mu = head["model"].predict(prep(bf_y, head["cols"], art["cat_levels"]))
        y = bf_y[head["target"]].to_numpy().astype(float)
        for line in sorted({float(l) for l in
                            store.loc[store["Market"] == api, "Line"].dropna()}):
            if (api, line) in owned:      # a binary prop already grades this
                continue
            cons = _market_consensus(store, api, line)
            if cons.empty:
                continue
            model = pd.DataFrame({"Date": date, "PlayerId": pid,
                                  "p": count_over(head, mu, line),
                                  "y": (y > line).astype(int)})
            row = _grade_against_market(f"{cname} {line:g}", model, cons, n_boot)
            if row:
                rows.append(row)
    return rows


def section_market(results, bf_y, sf_y, gf_y, art, year, store_path, n_boot):
    print("\n=== 9. Model vs. the market ===")
    store = O.load_odds(store_path, year=year)
    if store.empty:
        print(f"  No odds captured for {year} (store: {store_path}).")
        print("  Run `python Tools/2_scrape_odds.py` near game time to start "
              "collecting closing\n  lines; this section grades the model "
              "against them as they accrue. Everything\n  above is model vs. a "
              "naive base rate — this is the only section that asks whether\n"
              "  the edge survives a real sportsbook price.")
        return
    n_days = store["Date"].nunique()
    if n_days < 15:
        print(f"  WARNING: the odds store covers only {n_days} day(s). The "
              f"day-block bootstrap\n  resamples whole days, so with this few "
              f"the ROI confidence intervals collapse\n  toward the point "
              f"estimate (with 1 day they ARE the point estimate). Nothing\n"
              f"  below is statistically meaningful until ~15+ days of lines "
              f"accrue — treat it\n  as anecdote.")
    rows = []
    for name, meta in O.PROP_MARKET.items():
        r = results.get(name)
        if r is None:  # prop not scored on this artifact (e.g. older model)
            continue
        model = pd.DataFrame({"Date": pd.to_datetime(r["d"]).date,
                              "PlayerId": pd.to_numeric(r["pid"]),
                              "p": r["p"], "y": r["y"]})
        cons = _market_consensus(store, meta["api"], meta["line"])
        if cons.empty:
            continue
        row = _grade_against_market(name, model, cons, n_boot)
        if row:
            rows.append(row)
    rows += _starter_market_rows(sf_y, art, store, n_boot)
    rows += _batter_count_market_rows(bf_y, art, store, n_boot)
    rows += _game_market_rows(gf_y, art, store, n_boot)
    if not rows:
        print("  Odds present, but none matched the holdout predictions on "
              "(date, player).\n  Check that scraped player names resolved to "
              "PlayerIds (Tools/2_scrape_odds.py).")
        return
    print(pd.DataFrame(rows).to_string(index=False))
    print("  'model ll' vs 'market ll': log loss on the SAME events (de-vigged "
          "market as the\n  forecast) — lower wins. If the market is sharper, "
          "base-rate lift is not an edge.")
    print("  'flat ROI' bets 1u on every side the model prices +EV at the best "
          "book price and\n  settles at the outcome. Negative = the vig eats "
          "the edge.")
    print("  Labels: batter binary props ('hr','tb2',...); batter count-head "
          "lines the binaries\n  don't cover ('xtb 2.5' = 3+ TB, 'xhrr 3.5'); "
          "pitcher lines ('pk 6.5','per 2.5'); and\n  game markets ('totals "
          "8.5'; 'h2h (home)' = model P(home win) vs the moneyline).")


# ------------------------------------------------ in-season recalibration


# drift-guard variants compared side by side; "expand" is what train.py
# currently stores for `predict.py --recal`.
# VERDICT (2026-07-07, both years): no variant beats the incumbent. On the
# one genuinely drifting prop (bb, 2026) all three offsets tie at +0.0005 —
# right at the noise line — and temp loses everywhere (1/14). The Section-4
# drift (~1-2pp) is real but too small to correct profitably in-season.
# Recal stays default-OFF; this table stays as the standing monitor — if bb
# clears +0.0005 decisively by August, expand is the variant to enable.
RECAL_VARIANTS = [
    ("expand", {}),
    ("trail30", {"window_days": 30}),
    ("trail45", {"window_days": 45}),
    ("temp", {"temp": True}),           # expanding window + temperature term
]


def section_inseason(results, min_n=300):
    print("\n=== 10. In-season recalibration (drift guard) — variant "
          "shoot-out ===")
    rows = []
    wins = {v: 0 for v, _ in RECAL_VARIANTS}
    for name, r in results.items():
        row = {"prop": name}
        best_v, best_gain = "raw", 0.0
        mask = y_s = p_s = None
        for vname, kw in RECAL_VARIANTS:
            kwargs = dict(kw)
            if kwargs.pop("temp", False):
                kwargs["temp"] = r["temp"]
            corrected, applied, order = R.inseason_correct(
                r["d"], r["p"], r["y"], min_n=min_n, **kwargs)
            if mask is None:            # identical across variants
                mask, y_s = applied, r["y"][order]
                p_s = np.clip(r["p"][order], 1e-4, 1 - 1e-4)
                if mask.sum() < 50 or len(np.unique(y_s[mask])) < 2:
                    break
                row["n"] = int(mask.sum())
                row["ll raw"] = f"{log_loss(y_s[mask], p_s[mask]):.4f}"
            gain = (log_loss(y_s[mask], p_s[mask])
                    - log_loss(y_s[mask], corrected[mask]))
            row[vname] = f"{gain:+.4f}"
            wins[vname] += int(gain > 1e-6)
            if gain > best_gain + 1e-6:
                best_v, best_gain = vname, gain
        if "n" not in row:
            row.update({"n": 0, "ll raw": "-", "best": "n/a"})
        else:
            row["best"] = best_v
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    helped = " | ".join(f"{v} {wins[v]}/{len(rows)}"
                        for v, _ in RECAL_VARIANTS)
    print(f"  Per-prop drift corrections, each refit ONLY on in-season games "
          f"before each date\n  (after {min_n} games accrue) — strictly "
          f"causal. Columns are log-loss GAIN over the\n  raw frozen model "
          f"on the same rows (positive = the guard helped):\n"
          f"    expand   one offset, all season-to-date (what --recal serves "
          f"today)\n"
          f"    trail30/45  one offset, last 30/45 days only (tracks a "
          f"summer surge)\n"
          f"    temp     offset + temperature slope, season-to-date\n"
          f"  Helped: {helped}.")
    print("  Read: gains under ~0.0005 are noise. A variant earns serving "
          "(`predict.py --recal`)\n  only by winning on the props that "
          "drift (bb/hr/run), here AND on the --confirm year.")


# ----------------------------------------------------- baseline tracking


def prop_summary(results):
    """Flat, JSON-able headline scalars per prop for the --set-baseline diff:
    AUC and logloss edge (higher = better), ECE (lower = better), and the
    daily top-10 hit rate (higher = better)."""
    s = {}
    for name, r in results.items():
        day = pd.DataFrame({"d": r["d"], "p": r["p"], "y": r["y"]})
        top10 = day.sort_values("p", ascending=False).groupby(
            "d").head(10)["y"].mean()
        s[f"{name}_auc"] = round(float(r["auc"]), 4)
        s[f"{name}_edge"] = round(float(r["edge"]), 4)
        s[f"{name}_ece"] = round(ece(r["p"], r["y"]), 4)
        s[f"{name}_top10"] = round(float(top10), 4)
    return s


def verdict(metric, was, now):
    """better / worse / within-noise for one metric, honoring its direction.
    For dispersion-type metrics, "better" means moving closer to the ideal
    1.0. A delta inside the metric's NOISE_BAND is "within noise" — retrain
    jitter alone moves it that much, so it must not drive a keep/revert."""
    band = noise_band(metric)
    if any(metric.endswith(s) for s in TARGET_ONE):
        was_d, now_d = abs(was - 1.0), abs(now - 1.0)
        if abs(now_d - was_d) <= band:
            return "within noise"
        return "better" if now_d < was_d else "worse"
    delta = now - was
    if abs(delta) <= band:
        return "within noise"
    lower = any(metric.endswith(s) for s in LOWER_BETTER)
    return "better" if (delta < 0) == lower else "worse"


def handle_baseline(summary, year, set_baseline, select=False):
    """Save this run's headline metrics as the baseline, or (default) diff the
    current run against a saved one and report what improved / regressed.
    Selection mode keeps its own baseline file (and both names carry the
    scored year), so iterating on the selection year never collides with
    the confirm-only holdout snapshot."""
    tag = "select_" if select else ""
    base_path = ART / f"eval_baseline_{tag}{year}.json"
    if set_baseline:
        base_path.write_text(json.dumps(summary, indent=2))
        print("\n=== Baseline set ===")
        print(f"  snapshotted {len(summary)} metrics -> {base_path.name}")
        print("  Re-run without --set-baseline after a change to see the diff.")
        return

    print("\n=== 11. Change vs baseline ===")
    if not base_path.exists():
        print(f"  No baseline for {year} yet. Run with --set-baseline to "
              f"snapshot this\n  model, then re-run after a change to see what "
              f"moved.")
        return
    base = json.loads(base_path.read_text())
    rows = []
    for k in summary:
        if k not in base:
            continue
        was, now = float(base[k]), float(summary[k])
        rows.append({"metric": k, "baseline": round(was, 4),
                     "now": round(now, 4), "delta": f"{now - was:+.4f}",
                     "band": noise_band(k),
                     "result": verdict(k, was, now)})
    print(f"  (baseline: {base_path.name})")
    print(pd.DataFrame(rows).to_string(index=False))
    n_better = sum(1 for r in rows if r["result"] == "better")
    n_worse = sum(1 for r in rows if r["result"] == "worse")
    print(f"\n  {n_better} better, {n_worse} worse, "
          f"{len(rows) - n_better - n_worse} within noise.")
    print("  'band' is the retrain-to-retrain jitter for that metric type; "
          "only deltas\n  beyond it say anything about the change. A real "
          "improvement should move\n  several related metrics beyond their "
          "bands, not one metric barely past it.")
    new = [k for k in summary if k not in base]
    missing = [k for k in base if k not in summary]
    if new:
        print(f"  new since baseline ({len(new)}): {', '.join(new)}")
    if missing:
        print(f"  in baseline but not reported now ({len(missing)}): "
              f"{', '.join(missing)}  (re-run --set-baseline to refresh)")


# --------------------------------------------- paired verdict (policy v2)
# The static-band Section 11 above compares two frozen point estimates; it is
# a screen. --paired is the VERDICT: a paired day-block bootstrap of the change
# itself. --set-baseline snapshots the pre-change model's per-row predictions
# (eval_paired_*.joblib); --paired reloads them, scores the candidate on the
# same rows, and CIs the per-resample delta. Keep by default; a north-star
# metric whose delta CI lies entirely on the harmful side is the only bench
# signal (and only when it corroborates on the other year).
#
# ADVISORY Score rows (step 1 of the Score-north-star plan, user 2026-07-15):
# each head also gets a paired delta of the 5_prop_rankings v4 rank-weighted
# composite (the user's betting priorities: Lift-led, per-class weights) —
# the SAME day-block draws price baseline and candidate, so the CI is on the
# change in the composite itself. These rows are ADVISORY: raw-CI verdicts
# tagged "(adv)", EXCLUDED from the BH-FDR family and the net keep/bench
# vote. The starred multi-metric bar stays the arbiter; the Score read rides
# along for a few builds first because (a) its qualities CLIP at fixed
# anchors, so ELITE heads can sit in saturated zones where a real move shows
# zero delta, and (b) its heaviest term (Lift) is its noisiest. Coverage:
# binary heads (self-contained, p/y only) + the O/U families the board
# prices (count families price BOTH mu sets through the CANDIDATE's serving
# calibrators — the delta isolates the mu change, holding pricing constant).
# Mean heads are skipped (their board Score is MAE-led; the starred *MAE row
# already reads it); winner has no candidate predictions in this read.
SCORE_BOOT_DEFAULT = 400   # matches 5_prop_rankings.BOOT_B


def _score_delta_row(prop, n, s_b, s_c):
    """One advisory row from paired per-draw composite scores (same weight
    matrix both sides). None when every draw is degenerate."""
    d = np.asarray(s_c, float) - np.asarray(s_b, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return None
    mean = float(d.mean())
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"prop": prop, "metric": "Score", "delta": f"{mean:+.2f}",
            "95% CI": f"[{lo:+.2f}, {hi:+.2f}]", "n": n,
            "verdict": paired_verdict(lo, hi) + " (adv)",
            "q": "(adv)", "_star": False, "_p": None, "_mean": mean}


def build_binary_results(art, bf_y):
    """Per-row calibrated probabilities for every binary prop, in serving order
    (raw pass first so stacked props see their donors, then the threshold-
    ladder coherence projection predict.py also applies) — the `results` dict
    the print sections and the paired test both consume."""
    X = prep(bf_y, art["bat_cols"], art["cat_levels"])
    raw_p = {name: predict_prop(art["props"][name], X)
             for name in PROPS if name in art["props"]}
    fin_p = {name: apply_stack(art["props"][name], raw_p[name], raw_p)
             for name in raw_p}
    fin_p = F.enforce_ladders(fin_p)   # serve/eval identical prices
    results = {}
    for name, (target, _desc) in PROPS.items():
        if name not in fin_p:  # artifact predates this prop
            continue
        p = fin_p[name]
        y = bf_y[target].to_numpy()
        base = np.full_like(p, y.mean())
        results[name] = {
            "p": p, "y": y, "d": bf_y["Date"].to_numpy(),
            "g": bf_y["GamePk"].to_numpy(),
            "pid": bf_y["PlayerId"].to_numpy(),
            "temp": pd.to_numeric(bf_y["Temp"], errors="coerce").to_numpy(),
            "auc": roc_auc_score(y, p),
            "edge": log_loss(y, base) - log_loss(y, p),
        }
    return results


def build_count_preds(art, bf_y, sf_y, gf_y):
    """Per-row count means (mu) + outcomes, keyed for the paired MAE test:
    the batter/starter count heads, the starter-K model, and game totals."""
    cat = art["cat_levels"]
    KEY = ["Date", "GamePk", "PlayerId"]
    out = {}
    for name, head in art.get("count_models", {}).items():
        d = bf_y if head["frame"] == "bat" else sf_y
        df = d[KEY].copy()
        df["mu"] = head["model"].predict(prep(d, head["cols"], cat))
        df["y"] = d[head["target"]].to_numpy().astype(float)
        out[name] = {"df": df.reset_index(drop=True), "keycols": KEY}
    if art.get("k_model") is not None:
        df = sf_y[KEY].copy()
        df["mu"] = art["k_model"].predict(prep(sf_y, art["st_cols"], cat))
        df["y"] = sf_y["y_so"].to_numpy().astype(float)
        out["k"] = {"df": df.reset_index(drop=True), "keycols": KEY}
    if art.get("team_runs_model") is not None:
        tg = F.build_team_game_frame(gf_y)
        tp = art["team_runs_model"].predict(prep(tg, art["tg_cols"], cat))
        n_g = len(gf_y)
        gk = ["Date", "GamePk"]
        df = gf_y[gk].copy()
        df["mu"] = tp[:n_g] + tp[n_g:]
        df["y"] = (pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy()
                   + pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy())
        out["total"] = {"df": df.reset_index(drop=True), "keycols": gk}
        # H5 team_total (2026-07-14): the per-TEAM line family, two rows per
        # game — the paired MAE read + 5_prop_rankings' 'Team Runs > x' family
        if art.get("team_line_cals") or art.get("team_total_disp"):
            tk = ["Date", "GamePk", "Home"]
            tdf = pd.DataFrame({
                "Date": np.concatenate([gf_y["Date"].to_numpy()] * 2),
                "GamePk": np.concatenate([gf_y["GamePk"].to_numpy()] * 2),
                "Home": np.concatenate([np.zeros(n_g, dtype=int),
                                        np.ones(n_g, dtype=int)]),
                "mu": tp,
                "y": np.concatenate([
                    pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy(),
                    pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy()])})
            out["team_total"] = {"df": tdf, "keycols": tk}
    return out


def build_binary_snapshot(results):
    """Per-row p + y + keys per binary prop, for the paired-baseline file."""
    return {name: {"df": pd.DataFrame({
        "Date": r["d"], "GamePk": r["g"], "PlayerId": r["pid"],
        "p": r["p"], "y": r["y"]}),
        "keycols": ["Date", "GamePk", "PlayerId"]}
        for name, r in results.items()}


def score_snapshot(gf_y, art):
    """Per-TEAM expected runs + actual score for the paired snapshot — the
    workbook's Away Score / Home Score columns are these means, so
    5_prop_rankings grades them like every other displayed column. Two rows
    per game (Home flag distinguishes them)."""
    if art.get("team_runs_model") is None:
        return None
    tg = F.build_team_game_frame(gf_y)
    tp = art["team_runs_model"].predict(prep(tg, art["tg_cols"],
                                             art["cat_levels"]))
    n_g = len(gf_y)
    away_y = pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy()
    home_y = pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy()
    return {"df": pd.DataFrame({
        "Date": np.concatenate([gf_y["Date"].to_numpy()] * 2),
        "GamePk": np.concatenate([gf_y["GamePk"].to_numpy()] * 2),
        "Home": np.concatenate([np.zeros(n_g, dtype=int),
                                np.ones(n_g, dtype=int)]),
        "mu": tp, "y": np.concatenate([away_y, home_y])}),
        "keycols": ["Date", "GamePk", "Home"]}


def _win_scoreable(win_cols, frame):
    """Can the dedicated win model score this frame? persp_home is EXEMPT
    from the presence check (2026-07-16 hotfix): it is the WINNER_MIRROR
    augmentation flag that predict_win pins to 1.0 itself, so its absence
    from an eval frame must not disqualify the model. Without the
    exemption, every winner section here silently fell back to
    Poisson-means and the paired snapshot lost its winner block (the
    07-16 ship-morning reads scored winner from the fallback)."""
    return bool(win_cols) and all(c in frame.columns for c in win_cols
                                  if c != "persp_home")


def _ensure_persp(frame, win_cols):
    """Materialize persp_home before prep()'s strict column selection —
    the VALUE is irrelevant (predict_win pins 1.0 itself), the COLUMN must
    exist. In-place; a constant column on an eval frame is harmless."""
    if "persp_home" in (win_cols or []) and "persp_home" not in frame.columns:
        frame["persp_home"] = 1.0


def winner_snapshot(gf_y, art):
    """Per-game served home-win probability + outcome for the paired
    snapshot — lets 5_prop_rankings grade the Win Prob column on the same
    internal diagnostics as every other probability (it previously had
    nothing to grade from). None when the artifact predates the dedicated
    win model. Same computation as section_games."""
    win_cols = art.get("win_model", {}).get("cols", [])
    if not _win_scoreable(win_cols, gf_y):
        return None
    _ensure_persp(gf_y, win_cols)
    tg = F.build_team_game_frame(gf_y)
    tp = art["team_runs_model"].predict(prep(tg, art["tg_cols"],
                                             art["cat_levels"]))
    n_g = len(gf_y)
    Xw = prep(gf_y, win_cols, art["cat_levels"])
    p_home = predict_win(art["win_model"], Xw, tp[n_g:], tp[:n_g])
    away_y = pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy()
    home_y = pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy()
    return {"df": pd.DataFrame({
        "Date": gf_y["Date"].to_numpy(),
        "GamePk": gf_y["GamePk"].to_numpy(),
        "p": p_home, "y": (home_y > away_y).astype(int)}),
        "keycols": ["Date", "GamePk"]}


def _all_boosters(art):
    """(label, LightGBM model) for every booster in an artifact, so feature
    gain can be summed across the props/heads that share a column."""
    pairs = [(name, p["gbm"]) for name, p in art.get("props", {}).items()
             if "gbm" in p]
    pairs += [(name, h["model"]) for name, h in art.get("count_models", {}).items()]
    if art.get("k_model") is not None:
        pairs.append(("k", art["k_model"]))
    if art.get("team_runs_model") is not None:
        pairs.append(("total", art["team_runs_model"]))
    wm = art.get("win_model") or {}
    if "gbm" in wm:
        pairs.append(("winner", wm["gbm"]))
    return pairs


def declared_features(art):
    """Every feature column the artifact's heads DECLARE (bagging-independent —
    unlike booster introspection, which misses columns used only by bagged
    heads). The stable basis for 'what columns are new since the baseline'."""
    feats = (set(art.get("bat_cols", [])) | set(art.get("st_cols", []))
             | set(art.get("tg_cols", [])))
    for h in art.get("count_models", {}).values():
        feats.update(h.get("cols", []))
    wm = art.get("win_model") or {}
    feats.update(wm.get("cols", []) or [])
    feats.update(wm.get("lr_cols", []) or [])
    return feats


def feature_gains(art):
    """({col: total gain}, {col: heads that split on it}, {all feature cols})
    across every booster — the (2) usage gate that labels a flat feature inert
    (gain 0, safe ballast) vs active (used but nets out). Bagged heads
    (features.MeanBag) are expanded into their seed members with gain averaged,
    so a bagged prop weighs like a single model."""
    gains, users, feats = {}, {}, set()
    for label, model in _all_boosters(art):
        members = getattr(model, "models", None) or [model]
        w = 1.0 / len(members)
        for m in members:
            b = getattr(m, "booster_", None)
            if b is None:
                continue
            for n, g in zip(b.feature_name(),
                            b.feature_importance(importance_type="gain")):
                feats.add(n)
                gains[n] = gains.get(n, 0.0) + w * float(g)
                if g > 0:
                    users.setdefault(n, set()).add(label)
    return gains, users, feats


CALSLOPE_BAND = 0.03   # |slope-1| move to count cal-slope (point est, no CI)


def paired_verdict(lo, hi):
    """CI decides (policy v2): entirely good = improved, entirely bad = harm,
    straddling 0 = no effect (keep). Metrics are oriented so + is always good.
    RAW verdict only — run_paired re-verdicts under the BH-FDR gate (audit #1)
    unless --fdr 0."""
    if lo > 0:
        return "IMPROVED"
    if hi < 0:
        return "HARM"
    return "no effect"


DEFAULT_FDR = 0.10


def bh_qvalues(ps):
    """Benjamini-Hochberg q-values (audit fix #1). A --paired read makes
    ~100+ simultaneous CI calls; at plain 95% CIs ~5 spurious 'CI-clear'
    verdicts are EXPECTED under the null per read. Verdicts therefore gate
    on BH q <= --fdr instead of raw CI exclusion. (Sequential re-testing of
    the same season across many reads is NOT corrected here — that residual
    risk is why the forward record, not any backtest, is the true test.)"""
    ps = np.asarray(ps, dtype=float)
    m = len(ps)
    order = np.argsort(ps)
    q = np.empty(m)
    prev = 1.0
    for rank_pos in range(m - 1, -1, -1):
        i = order[rank_pos]
        prev = min(prev, ps[i] * m / (rank_pos + 1))
        q[i] = prev
    return q


def paired_day_bootstrap(dates, y, base_val, cand_val, kind, n_boot, seed=0):
    """Paired day-block bootstrap of (candidate - baseline), oriented so POSITIVE
    = candidate better for every kind (auc/edge higher-better; ece/mae
    lower-better, flipped to base-minus-cand). Each draw resamples whole days
    ONCE and scores both models on the SAME rows, so shared-day noise cancels
    and the CI is on the CHANGE, not on either level."""
    rng = np.random.default_rng(seed)
    dates = np.asarray(dates)
    uniq = pd.unique(dates)
    idx_by_day = {d: np.flatnonzero(dates == d) for d in uniq}
    out = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_day[d] for d in take])
        yy, pb, pc = y[idx], base_val[idx], cand_val[idx]
        if kind in ("auc", "edge"):
            if yy.min() == yy.max():
                continue
            out.append(roc_auc_score(yy, pc) - roc_auc_score(yy, pb)
                       if kind == "auc"
                       else log_loss(yy, pb) - log_loss(yy, pc))
        elif kind == "ece":                     # + = candidate lower ECE
            if yy.min() == yy.max():
                continue
            out.append(ece(pb, yy) - ece(pc, yy))
        else:  # mae
            out.append(mean_absolute_error(yy, pb) - mean_absolute_error(yy, pc))
    a = np.asarray(out)
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    if sd > 0:      # bootstrap-normal two-sided p (finer than 1/n_boot)
        pval = float(2.0 * stats.norm.sf(abs(a.mean()) / sd))
    else:
        pval = 1.0 if a.mean() == 0 else 0.0
    return (float(a.mean()), float(np.percentile(a, 2.5)),
            float(np.percentile(a, 97.5)), pval)


# The Model sources that define what a train run produces (train imports
# features + predict + recalibrate; predict imports recalibrate). Their md5s
# are written to baseline_code_fp.json at --set-baseline so the daily
# update_all --retrain can tell "shipped code" from "experiment in flight"
# and stand down rather than re-baseline a candidate as its own reference.
CODE_FP_FILES = ("features.py", "train.py", "predict.py", "recalibrate.py",
                 # Scrapers/seasons.py decides which seasons feed the frames
                 # (FIRST_SEASON); changing it is a modeling change like any
                 # other, so the daily job must go scrape-only until the
                 # experiment is confirmed and re-baselined.
                 "../Scrapers/seasons.py")


def _code_fingerprint():
    import hashlib
    here = Path(__file__).resolve().parent
    return {f: hashlib.md5((here / f).read_bytes()).hexdigest()
            for f in CODE_FP_FILES if (here / f).exists()}


def _data_fingerprint():
    """(name, size, mtime) for every frame-feeding Data/*.csv — written into
    the paired snapshot so run_paired can refuse a contaminated read. A daily
    scrape rewrites the CSVs; scoring a candidate on refreshed data against a
    baseline snapshotted on the old data conflates data drift with the change
    under test (every prop moves a little, even ones the change can't reach —
    2026-07-09 park-factor read). mlb_odds.csv is grading-only, never a frame
    input, so it can't contaminate a paired read and is exempt."""
    return sorted((p.name, p.stat().st_size, int(p.stat().st_mtime))
                  for p in F.DATA_DIR.glob("*.csv")
                  if p.name != "mlb_odds.csv")


def _frames_fingerprint():
    """(size, mtime) of frames.joblib — the input the eval ACTUALLY consumes.
    The CSV fingerprint above is a conservative proxy: a scrape can rewrite
    the CSVs without the cached frames changing (train without --rebuild),
    and that read is still clean. Stored in new snapshots for a direct
    check; run_paired falls back to an mtime-ordering inference for
    snapshots written before this existed."""
    p = ART / "frames.joblib"
    return (p.stat().st_size, int(p.stat().st_mtime)) if p.exists() else None


def run_paired(art, results, count_preds, tag, year, n_boot, era=None,
               fdr=DEFAULT_FDR, stale_ok=False,
               score_boot=SCORE_BOOT_DEFAULT):
    """Print the paired keep/bench verdict against the --set-baseline snapshot,
    plus the ② feature-usage gate for columns added since that snapshot.
    `era` names a frozen archive dir under artifacts/ (e.g.
    era_2026-07-09_queue_closed): the snapshot loads from there instead of
    the rolling baseline — the CUMULATIVE 'did everything kept since this
    era sum to harm?' read. Era reads warn on data drift instead of
    refusing (drift is expected across months; a CI-clear harm on an era
    read is grounds for the RIGOROUS check — retrain the archived era
    sources on current data — before pruning anything)."""
    base_dir = ART / era if era else ART
    path = base_dir / f"eval_paired_{tag}{year}.joblib"
    label = f"era '{era}'" if era else "baseline"
    print(f"\n=== 11p. Paired change vs {label} (day-block paired bootstrap, "
          f"{n_boot} draws; + = candidate better; * = north-star; "
          f"Score = advisory composite) ===")
    PRK, score_rng = None, None
    if score_boot:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Tools"))
        try:
            # lazy, fail-soft; the digit-leading tool name (renamed from
            # prop_rankings.py 2026-07-15) needs importlib
            import importlib
            PRK = importlib.import_module("5_prop_rankings")
            score_rng = np.random.default_rng(20260715)
        except Exception as e:            # advisory rows must never kill a read
            print(f"  (advisory Score rows unavailable: {e})")
            PRK = None
    if not path.exists():
        pre = "" if tag == "select_" else "--confirm "
        print(f"  No paired snapshot at {path}. "
              + ("Check the era dir name." if era else
                 "Snapshot the PRE-change model first:\n    python "
                 f"Model/evaluate_deep.py {pre}--set-baseline"
                 "\n  (--set-baseline now also writes the per-row paired "
                 "snapshot.)"))
        return
    comp = joblib.load(path)
    if comp.get("data_fp") is not None:
        base = {x[0]: tuple(x[1:]) for x in comp["data_fp"]}
        now = {x[0]: tuple(x[1:]) for x in _data_fingerprint()}
        changed = sorted(k for k in base.keys() | now.keys()
                         if base.get(k) != now.get(k))
        if changed and era:
            print(f"  NOTE: {len(changed)} data file(s) changed since this era "
                  "was frozen — deltas below mix\n  code keeps with data "
                  "drift. Treat CI-clear harm as a signal to run the "
                  "rigorous\n  check (retrain the era dir's archived sources "
                  "on current data), not as a verdict.")
        elif changed:
            # The CSVs drifted — but if the cached frames (the only data this
            # eval consumes) are the ones the snapshot was scored on, the
            # read is still clean: a scrape after the snapshot can't reach a
            # model that never re-reads the CSVs. Direct check when the
            # snapshot stored frames_fp; mtime-ordering inference otherwise
            # (frames older than the snapshot + never rebuilt since = same
            # frames both sides).
            fr = ART / "frames.joblib"
            if comp.get("frames_fp") is not None:
                frames_ok = comp["frames_fp"] == _frames_fingerprint()
            else:
                frames_ok = (fr.exists()
                             and fr.stat().st_mtime < path.stat().st_mtime)
            if frames_ok:
                print(f"  NOTE: {len(changed)} Data/*.csv rescraped since the "
                      "snapshot, but frames.joblib is unchanged —\n  baseline "
                      "and candidate score the same cached frames, so the "
                      "read is clean.")
            elif stale_ok:
                # user-accepted stale read (2026-07-15): deltas below mix
                # data drift with the change under test — treat them as a
                # SCREEN, not a clean attribution; the forward record is
                # the true test either way.
                print(f"  !! STALE BASELINE ACCEPTED (--stale-ok): "
                      f"{len(changed)} Data/*.csv changed since the")
                print("  !! snapshot — deltas MIX data drift with the change "
                      "under test.")
            else:
                pre = "" if tag == "select_" else "--confirm "
                print("  !! STALE BASELINE — Data/*.csv changed since the snapshot was")
                print("  !! written (a scrape ran). This read would conflate data drift")
                print("  !! with the change under test, so it is SKIPPED. Re-baseline:")
                print("  !!   1. set the working tree back to the shipped code")
                print("  !!   2. python Model/train.py")
                print(f"  !!   3. python Model/evaluate_deep.py {pre}--set-baseline")
                print("  !!   4. restore the candidate, retrain, re-run --paired")
                print(f"  !! changed: {', '.join(changed[:8])}"
                      + (f" (+{len(changed) - 8} more)" if len(changed) > 8 else ""))
                print("  !! (or accept the confound explicitly: --paired --stale-ok)")
                return
    rows = []
    for name, r in results.items():
        if name not in comp.get("binary", {}):
            continue
        m = comp["binary"][name]["df"].merge(
            pd.DataFrame({"Date": r["d"], "GamePk": r["g"], "PlayerId": r["pid"],
                          "p": r["p"], "y": r["y"]}),
            on=["Date", "GamePk", "PlayerId"], suffixes=("_b", "_c"))
        d, y = m["Date"].to_numpy(), m["y_c"].to_numpy()
        pb, pc = m["p_b"].to_numpy(), m["p_c"].to_numpy()
        for metric, kind in (("AUC", "auc"), ("edge", "edge"), ("ECE", "ece")):
            mean, lo, hi, pv = paired_day_bootstrap(d, y, pb, pc, kind, n_boot)
            rows.append({"prop": name, "metric": "*" + metric,
                         "delta": f"{mean:+.4f}",
                         "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "n": len(m),
                         "verdict": paired_verdict(lo, hi), "_star": True,
                         "_p": pv, "_mean": mean})
        # cal-slope: point estimate (a logistic per resample would triple the
        # run); classified against CALSLOPE_BAND since there's no CI. Counted in
        # the net vote like the others. delta + = candidate nearer the ideal 1.0.
        sb, sc = calibration_slope(pb, y), calibration_slope(pc, y)
        dcs = abs(sb - 1.0) - abs(sc - 1.0)
        rows.append({"prop": name, "metric": "*cal-slp",
                     "delta": f"{dcs:+.3f}",
                     "95% CI": f"{sb:.2f}->{sc:.2f} /1.0", "n": len(m),
                     "verdict": ("IMPROVED" if dcs >= CALSLOPE_BAND
                                 else "HARM" if dcs <= -CALSLOPE_BAND
                                 else "no effect"),
                     "_star": True})
        # advisory paired delta of the v4 composite (same draws both sides)
        if PRK is not None:
            did = pd.factorize(m["Date"], sort=False)[0]
            D = int(did.max()) + 1 if len(did) else 0
            if D >= 2:
                Wm = score_rng.multinomial(
                    D, np.full(D, 1.0 / D), size=score_boot).astype(float)
                yf = y.astype(float)
                s_b = PRK._binlike_boot(PRK._prep_binlike(did, D, pb, yf),
                                        Wm)[0]
                s_c = PRK._binlike_boot(PRK._prep_binlike(did, D, pc, yf),
                                        Wm)[0]
                r_ = _score_delta_row(name, len(m), s_b, s_c)
                if r_:
                    rows.append(r_)
    for name, cp in count_preds.items():
        if name not in comp.get("count", {}):
            continue
        kc = comp["count"][name]["keycols"]
        m = comp["count"][name]["df"].merge(cp["df"], on=kc, suffixes=("_b", "_c"))
        d, y = m["Date"].to_numpy(), m["y_c"].to_numpy()
        mean, lo, hi, pv = paired_day_bootstrap(
            d, y, m["mu_b"].to_numpy(), m["mu_c"].to_numpy(), "mae", n_boot)
        rows.append({"prop": name, "metric": "*MAE", "delta": f"{mean:+.4f}",
                     "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "n": len(m),
                     "verdict": paired_verdict(lo, hi), "_star": True,
                     "_p": pv, "_mean": mean})
        # advisory composite for the O/U families the board prices — both mu
        # sets priced through the CANDIDATE's serving calibrators, so the
        # delta isolates the mu change (mean heads read through *MAE above)
        if PRK is not None and name in PRK.CNT_MARKETS:
            try:
                head = art.get("count_models", {}).get(name)
                kd = float(art.get("k_disp", 1.0))
                td = float(art.get("total_disp", 1.0))
                yf = y.astype(float)
                p_b = PRK._prep_count_family(
                    name, {"count": {name: {"df": pd.DataFrame(
                        {"Date": m["Date"], "mu": m["mu_b"].to_numpy(float),
                         "y": yf})}}}, head, kd, td, art)
                p_c = PRK._prep_count_family(
                    name, {"count": {name: {"df": pd.DataFrame(
                        {"Date": m["Date"], "mu": m["mu_c"].to_numpy(float),
                         "y": yf})}}}, head, kd, td, art)
                if p_b["D"] >= 2 and p_b["lines"] and p_c["lines"]:
                    Wm = score_rng.multinomial(
                        p_b["D"], np.full(p_b["D"], 1.0 / p_b["D"]),
                        size=score_boot).astype(float)
                    r_ = _score_delta_row(name, len(m),
                                          PRK._countfam_boot(p_b, Wm),
                                          PRK._countfam_boot(p_c, Wm))
                    if r_:
                        rows.append(r_)
            except Exception as e:        # advisory: never kills the read
                print(f"  (Score row skipped for {name}: {e})")
    if not rows:
        print("  (no props overlap between this model and the snapshot)")
        return
    # ---- multiplicity gate (audit fix #1): a read this wide makes ~100+
    # simultaneous tests; raw 95% CIs alone EXPECT ~5 false CI-clears under
    # the null. Verdicts on the CI'd metrics therefore require BH q <= fdr;
    # cal-slope has no CI and keeps its band rule (marked "(band)").
    pidx = [i for i, r in enumerate(rows)
            if r.get("_p") is not None and np.isfinite(r["_p"])]
    if fdr and fdr > 0 and pidx:
        qv = bh_qvalues([rows[i]["_p"] for i in pidx])
        n_rawclear = sum(1 for i in pidx
                         if rows[i]["verdict"] in ("IMPROVED", "HARM"))
        n_pass = 0
        for j, i in enumerate(pidx):
            rows[i]["q"] = f"{qv[j]:.3f}"
            sig = qv[j] <= fdr
            n_pass += int(sig)
            rows[i]["verdict"] = ("IMPROVED" if sig and rows[i]["_mean"] > 0
                                  else "HARM" if sig and rows[i]["_mean"] < 0
                                  else "no effect")
        for r in rows:
            if r.get("_p") is None:
                r.setdefault("q", "(band)")   # Score rows keep their "(adv)"
        print(f"  multiplicity gate ON: {len(pidx)} simultaneous tests; "
              f"~{0.05 * len(pidx):.1f} raw CI-clears expected under the "
              f"null; verdicts require Benjamini-Hochberg q <= {fdr:g} "
              f"(--fdr 0 restores raw-CI verdicts). {n_rawclear} raw "
              f"CI-clear, {n_pass} survive the gate.")
        print("  NOTE: repeated reads against the same season are sequential "
              "tests this gate does NOT correct — the forward record is the "
              "only uncorrectable-free test.")
    df = pd.DataFrame(rows)
    if "q" in df.columns:
        df["q"] = df["q"].fillna("-")
    drop = [c for c in ("_star", "_p", "_mean") if c in df.columns]
    print(df.drop(columns=drop).to_string(index=False))
    star = df[df["_star"]]
    # Balanced NET VOTE per prop across the gated metrics — AUC (rank), edge
    # (score), ECE + cal-slope (calibration) for binaries; MAE for counts.
    # net = #improved - #regressed: net>0 = WIN, net<0 = REGRESSION, net==0
    # (incl. 2-2) = flat (user's rule, 07-10). Shown with the net in parens.
    imp, harm = [], []
    for prop, grp in star.groupby("prop", sort=False):
        net = int((grp["verdict"] == "IMPROVED").sum()
                  - (grp["verdict"] == "HARM").sum())
        if net > 0:
            imp.append(f"{prop}(+{net})")
        elif net < 0:
            harm.append(f"{prop}({net})")
    print("\n  balanced net vote (AUC rank + edge/ECE/cal-slope probability "
          "quality; net = #improved - #regressed, ties = flat):")
    print(f"    net wins:        {imp or '[]'}")
    print(f"    net regressions: {harm or '[]'}")
    sc = df[df["metric"] == "Score"]
    if not sc.empty:
        s_win = sc[sc["verdict"].str.startswith("IMPROVED")]["prop"].tolist()
        s_harm = sc[sc["verdict"].str.startswith("HARM")]["prop"].tolist()
        print("\n  ADVISORY Score read (step 1 of the Score-north-star plan, "
              "2026-07-15): paired delta of the")
        print(f"  5_prop_rankings v4 rank-weighted composite ({score_boot} "
              f"draws, raw-CI verdicts). NOT in the FDR")
        print("  family or the net vote — the starred bar stays the arbiter "
              "while this read rides along.")
        print(f"    Score CI-clear wins:  {s_win or '[]'}")
        print(f"    Score CI-clear harms: {s_harm or '[]'}")
        print("  caveats: qualities clip at fixed anchors (an ELITE head in "
              "a saturated zone shows zero Score")
        print("  delta on a real move); count families price both mu sets "
              "through the CANDIDATE's calibrators;")
        print("  mean heads read through *MAE; winner has no candidate "
              "predictions in this read.")
    print("  NOTE: dev phase (superset, no selection) — this read is "
          "directional; the binding verdict is post-selection + families at "
          "ship, so don't over-weight dev-phase moves.")
    gains, users, _ = feature_gains(art)
    added = sorted(declared_features(art) - set(comp.get("features", [])),
                   key=lambda c: -gains.get(c, 0.0))
    print("\n--- (2) feature-usage gate: columns added since the paired baseline ---")
    if not added:
        print("  (none — candidate uses the same feature columns as the baseline)")
    for c in added:
        g = gains.get(c, 0.0)
        u = ", ".join(sorted(users.get(c, []))) or "-"
        print(f"  {c:<26} gain {g:>12.1f}  "
              f"[{'used' if g > 0 else 'INERT ballast'}]  ({u})")


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=None,
                    help="season to score (default: the scored suite's own "
                         "test year, stored in its artifact)")
    ap.add_argument("--prop", default="hr",
                    help="prop for the threshold/segment sections")
    ap.add_argument("--boot", type=int, default=400,
                    help="bootstrap resamples (lower = faster)")
    ap.add_argument("--set-baseline", action="store_true",
                    help="snapshot this run's headline metrics as the baseline "
                         "to diff future runs against")
    ap.add_argument("--select", action="store_true",
                    help="(default) score the SELECTION suite "
                         "(models_bt.joblib) on its own test year. "
                         "Iterate here.")
    ap.add_argument("--confirm", action="store_true",
                    help="score the SHIPPING suite (models.joblib) on the "
                         "confirm-only holdout (the newest season). Use once "
                         "per finished change, not to iterate.")
    ap.add_argument("--paired", action="store_true",
                    help="policy-v2 verdict: paired day-block bootstrap CI of "
                         "(candidate - baseline) per prop vs the snapshot a "
                         "prior --set-baseline saved, plus the feature-usage "
                         "gate. Fast (skips the verbose sections).")
    ap.add_argument("--era", default=None, metavar="DIR",
                    help="with --paired: read the snapshot from a frozen "
                         "archive under artifacts/ (e.g. "
                         "era_2026-07-09_queue_closed) instead of the rolling "
                         "baseline — the cumulative 'did everything kept "
                         "since sum to harm?' check. Warns on data drift "
                         "instead of refusing.")
    ap.add_argument("--odds", default=str(O.DEFAULT_STORE),
                    help="odds store CSV for the vs-market section "
                         "(Tools/2_scrape_odds.py writes it)")
    ap.add_argument("--fdr", type=float, default=DEFAULT_FDR,
                    help="Benjamini-Hochberg false-discovery-rate level for "
                         "the --paired verdicts (audit #1). A read makes "
                         "100+ simultaneous tests, so raw 95%% CIs alone "
                         "expect ~5 false CI-clears; 0 disables the gate "
                         "(legacy raw-CI verdicts).")
    ap.add_argument("--stale-ok", action="store_true",
                    help="with --paired: run against a snapshot whose data "
                         "fingerprint is stale (a scrape ran since) instead "
                         "of refusing — the deltas then MIX data drift with "
                         "the change under test; treat as a screen only")
    ap.add_argument("--score-boot", type=int, default=SCORE_BOOT_DEFAULT,
                    metavar="N",
                    help="with --paired: draws for the ADVISORY paired Score "
                         "rows (5_prop_rankings v4 rank-weighted composite; "
                         "step 1 of the Score-north-star plan, excluded from "
                         "the FDR family and the net vote). 0 disables. "
                         f"Default {SCORE_BOOT_DEFAULT}.")
    ap.add_argument("--superset", action="store_true",
                    help="score the SUPERSET electorate artifacts "
                         "(models_superset*.joblib) instead of the serving "
                         "keep-train artifacts — the dev-loop read under "
                         "the 2026-07-15 artifact-role split")
    args = ap.parse_args()
    if args.select and args.confirm:
        sys.exit("--select and --confirm are mutually exclusive")
    if args.paired and args.set_baseline:
        sys.exit("--paired reads the snapshot that --set-baseline writes; "
                 "run them in separate steps (set-baseline before the change, "
                 "paired after)")
    select = not args.confirm
    tag = "select_" if select else ""

    if select:
        art_path = ART / ("models_superset_bt.joblib" if args.superset
                          else "models_bt.joblib")
        if not art_path.exists():
            sys.exit(f"no artifacts at {art_path.name} — run "
                     "`python Model/train.py --select` (or a plain train.py "
                     "run) first")
        art = joblib.load(art_path)
        # years stored by train.py; fall back for pre-rollover artifacts
        yrs = art.get("years") or {"train": [2020, 2021, 2022, 2023],
                                   "cal": 2024, "test": 2025}
        if args.year is not None and args.year != yrs["test"]:
            print(f"(the default run scores the selection suite on its own "
                  f"{yrs['test']} test year; ignoring --year {args.year}. "
                  f"For the shipping suite use --confirm.)")
        args.year = yrs["test"]
        print(f"*** SELECTION mode (default): train<={max(yrs['train'])}, "
              f"cal {yrs['cal']}, scored on {yrs['test']}. Iterate\n*** "
              f"freely — the confirm-only holdout stays untouched until you "
              f"confirm a finished\n*** change there once, with --confirm. "
              f"***")
    else:
        art_path = ART / ("models_superset.joblib" if args.superset
                          else "models.joblib")
        art = joblib.load(art_path)
        yrs = art.get("years") or {"train": [2020, 2021, 2022, 2023, 2024],
                                   "cal": 2025, "test": 2026}
        if args.year is None:
            args.year = yrs["test"]
        if args.year <= max(yrs["train"]):
            print("*** WARNING: training year -> in-sample, inflated numbers ***")
        elif args.year == yrs["cal"]:
            print(f"*** NOTE: calibration year -> mildly optimistic numbers "
                  f"(early stop, blend\n*** weights and isotonic all fit on "
                  f"{yrs['cal']}). For honest {yrs['cal']} numbers use the\n"
                  f"*** default (selection) run. ***")
        else:
            print(f"*** CONFIRM mode: {args.year} is the confirm-only "
                  f"holdout. Use this to confirm a\n*** finished change once "
                  f"— iterating against these numbers quietly overfits\n*** "
                  f"the holdout. Day-to-day, run without --confirm "
                  f"(selection suite, {yrs['cal']}). ***")
    frames = joblib.load(ART / "frames.joblib")
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]

    # shadow-contract guard (audit #8, approved 07-15): a shadow-superset
    # artifact whose serving lists need shdw_ columns these frames don't
    # carry would die deep in prep() with a raw KeyError. Refuse up front
    # with a message instead. tg/wf are re-derived here WITHOUT shadows, so
    # any shdw_ in the game-head contracts is an automatic refusal.
    stamp = art.get("meta_stamp") or {}
    if stamp.get("role") == "superset":
        print(f"note: scoring a SUPERSET artifact (role=superset, created "
              f"{stamp.get('created')}) — the dev electorate, not the "
              f"serving model")
    problems = [f"bf:{c}" for c in art.get("bat_cols", [])
                if c.startswith("shdw_") and c not in bf.columns]
    problems += [f"sf:{c}" for c in art.get("st_cols", [])
                 if c.startswith("shdw_") and c not in sf.columns]
    problems += [f"tg:{c}" for c in art.get("tg_cols", [])
                 if c.startswith("shdw_")]
    problems += [f"wf:{c}" for c in (art.get("win_model") or {}).get(
        "cols", []) if c.startswith("shdw_")]
    if problems:
        sys.exit(
            f"REFUSING (audit #8 guard): {art_path.name} "
            f"carries a shadow-superset serving contract these frames can't "
            f"score ({len(problems)} shdw_ columns, e.g. "
            f"{', '.join(problems[:4])}). tg/wf are rebuilt shadow-free "
            f"here. Score the keep-train artifacts instead, or regenerate "
            f"the keep-lists (feature_select.py --write) and rerun the "
            f"keep train. This replaces what used to be a raw KeyError.")

    bf_y = bf[(bf["Season"] == args.year) & ~bf["ShortGame"].fillna(False)]
    print(f"Scoring {len(bf_y):,} batter-games, {args.year}...")
    # raw pass first, so stacked props (single/double borrow hit/tb2 scores,
    # predict.apply_stack) see their donors — same order of operations as serving
    results = build_binary_results(art, bf_y)

    sf_y = sf[(sf["Season"] == args.year) & ~sf["ShortGame"].fillna(False)]
    gf_y = gf[gf["Season"] == args.year].dropna(subset=["total_runs"])

    if args.paired:
        count_preds = build_count_preds(art, bf_y, sf_y, gf_y)
        run_paired(art, results, count_preds, tag, args.year, args.boot,
                   era=args.era, fdr=args.fdr, stale_ok=args.stale_ok,
                   score_boot=args.score_boot)
        return

    section_confidence(results, args.boot)
    section_topn(results)
    section_thresholds(results, args.prop)
    section_drift(results)
    section_segments(bf_y, results, args.prop)
    section_concentration(results)

    summary = prop_summary(results)

    summary.update(section_strikeouts(sf_y, art))
    summary.update(section_count_heads(bf_y, sf_y, art))
    section_h1_bars(results, bf_y, art)       # H1 deep-binary acceptance
    section_coherence(bf_y, gf_y, art)        # H6 grain-coherence read
    summary.update(section_games(gf_y, art))

    if select:
        print(f"\n=== 9. Model vs. the market === (skipped in selection "
              f"mode: no odds store\n  exists for {args.year}, and market "
              f"grading belongs to the --confirm run anyway)")
    else:
        section_market(results, bf_y, sf_y, gf_y, art, args.year, args.odds,
                       args.boot)
    section_inseason(results)

    handle_baseline(summary, args.year, args.set_baseline, select=select)
    if args.set_baseline:
        comp = {"binary": build_binary_snapshot(results),
                "count": build_count_preds(art, bf_y, sf_y, gf_y),
                "features": sorted(declared_features(art)),
                "data_fp": _data_fingerprint(),
                "frames_fp": _frames_fingerprint()}
        win = winner_snapshot(gf_y, art)
        if win is not None:
            comp["winner"] = win
        sc = score_snapshot(gf_y, art)
        if sc is not None:
            comp["score"] = sc
        joblib.dump(comp, ART / f"eval_paired_{tag}{args.year}.joblib", compress=3)
        print(f"  paired snapshot -> eval_paired_{tag}{args.year}.joblib "
              f"({len(comp['binary'])} binary props + {len(comp['count'])} "
              f"count heads{' + winner' if win is not None else ''}, "
              f"{len(comp['features'])} feature cols)")
        (ART / "baseline_code_fp.json").write_text(
            json.dumps(_code_fingerprint(), indent=1))
        if not select:
            # Record the code this confirm stamp was made under. The daily
            # job (update_all.confirm_is_stale) compares it to the shipped
            # sources and re-stamps the 2026 confirm exactly once per ship
            # — audit #6 as amended 2026-07-15 (auto-on-ship, user
            # directive). Between ships the daily job never touches 2026.
            (ART / "confirm_code_fp.json").write_text(
                json.dumps(_code_fingerprint(), indent=1))

    print("""
How to read this:
  - Section 1 is the headline: a prop is only bankable if BOTH the AUC CI
    stays above 0.5 and the logloss-edge CI stays above 0. Cal slope near
    1.00 and small ECE mean the probabilities can price bets directly.
  - Section 3 turns hit rates into the worst odds you could accept. If books
    routinely post better prices than break-even, the prop has real value.
  - Section 4 catches environment drift the models can't see (juiced ball,
    hot summer). Consistent positive rows = consider recalibrating in-season.
  - Sections 7-8 dispersion: if the index is well above 1.00, Poisson-based
    P(over) is overconfident at extreme lines - shade those probabilities.
  - Section 9 is the real-money test: model vs. de-vigged sportsbook prices on
    the same events, plus flat-stake ROI. Needs an odds store (2_scrape_odds.py);
    everything above only beats a naive base rate.
  - Section 10 asks whether a strictly-causal in-season correction would beat
    the frozen model as the environment drifts (a growing Section-4 miss) —
    and which variant: season-to-date offset, trailing-window, or
    temperature-aware.
  - Section 11 diffs this run against the --set-baseline snapshot. Deltas
    inside each metric's noise band (retrain jitter) are "within noise" —
    only movement beyond the band says a change did anything. Dispersion
    counts as "better" when it moves toward 1.0.
  - The default run scores the selection suite on its own test year —
    iterate here. Look at the newest season (--confirm) only to confirm a
    finished change.""")


if __name__ == "__main__":
    main()
