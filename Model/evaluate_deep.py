"""Deep evaluation: statistical confidence, drift, segments, and betting realism.

Extends evaluate.py with the questions it can't answer:

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

Usage:
    python Model/evaluate_deep.py [--year 2026] [--prop hr] [--boot 400]
"""

import argparse
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
from predict import (american_odds, nb_over, poisson_over,  # noqa: E402
                     poisson_win, predict_prop, predict_win)
from train import PROPS  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"


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

    print("\n--- P(over) line calibration (the sellable output) ---")
    rows = []
    for line in (3.5, 4.5, 5.5, 6.5, 7.5):
        p_over = np.array([poisson_over(l, line) for l in mu])
        actual = (y > line).astype(int)
        rows.append({"line": line, "mean P(over)": f"{p_over.mean():.3f}",
                     "actual over rate": f"{actual.mean():.3f}",
                     "logloss": f"{log_loss(actual, np.clip(p_over, 1e-4, 1 - 1e-4)):.4f}",
                     "base logloss": f"{log_loss(actual, np.full_like(p_over, actual.mean())):.4f}"})
    print(pd.DataFrame(rows).to_string(index=False))


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
    dist = (f"negative binomial, disp {model_disp:.2f} from cal year"
            if model_disp > 1.001 else "Poisson")
    print(f"  totals bias (pred-actual) {total_mu.mean() - total_y.mean():+.2f} "
          f"runs | observed dispersion {disp:.2f} | P(over) uses {dist}")

    print("\n--- P(over) run-line calibration ---")
    rows = []
    for line in (6.5, 7.5, 8.5, 9.5, 10.5):
        p_over = np.array([nb_over(l, line, model_disp) for l in total_mu])
        actual = (total_y > line).astype(int)
        rows.append({"line": line, "mean P(over)": f"{p_over.mean():.3f}",
                     "actual over rate": f"{actual.mean():.3f}",
                     "logloss": f"{log_loss(actual, np.clip(p_over, 1e-4, 1 - 1e-4)):.4f}",
                     "base logloss": f"{log_loss(actual, np.full_like(p_over, actual.mean())):.4f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # winner: dedicated model when the artifact has one
    win_cols = art.get("win_model", {}).get("cols", [])
    if win_cols and all(c in gf_y.columns for c in win_cols):
        Xw = prep(gf_y, win_cols, art["cat_levels"])
        p_home = predict_win(art["win_model"], Xw, mu_home, mu_away)
        print("\n  (winner probabilities from the dedicated win model)")
    else:
        p_home = np.array([poisson_win(h, a) for h, a in zip(mu_home, mu_away)])
        print("\n  (winner probabilities from Poisson means — no win model "
              "in artifacts)")
    actual_home = (home_y > away_y).astype(int)
    model_pick = (p_home >= 0.5).astype(int)
    model_right = model_pick == actual_home
    home_right = actual_home == 1
    acc = model_right.mean()
    lo, hi = wilson(int(model_right.sum()), len(model_right))
    print(f"\n--- winner ---")
    print(f"  model accuracy {acc:.1%} [{lo:.1%}, {hi:.1%}] | always-home "
          f"baseline {home_right.mean():.1%}")
    b = int((model_right & ~home_right).sum())
    c = int((~model_right & home_right).sum())
    if b + c == 0:
        pval = 1.0
    else:
        try:
            pval = stats.binomtest(b, b + c).pvalue
        except AttributeError:  # scipy < 1.7
            pval = stats.binom_test(b, b + c)
    print(f"  games model wins vs home-pick: {b} | loses: {c} | "
          f"McNemar p-value {pval:.3f}")
    print("  (p > 0.05 = no statistical evidence the winner model beats "
          "just picking the home team)")

    q = pd.qcut(p_home, 5, duplicates="drop")
    tab = pd.DataFrame({"p": p_home, "y": actual_home}).groupby(q, observed=True).agg(
        predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    print("\n--- win-prob calibration (quintiles) ---")
    print(tab.round(3).to_string())


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--prop", default="hr",
                    help="prop for the threshold/segment sections")
    ap.add_argument("--boot", type=int, default=400,
                    help="bootstrap resamples (lower = faster)")
    args = ap.parse_args()

    art = joblib.load(ART / "models.joblib")
    frames = joblib.load(ART / "frames.joblib")
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]

    if args.year <= 2024:
        print("*** WARNING: training year -> in-sample, inflated numbers ***")
    elif args.year == 2025:
        print("*** NOTE: calibration year -> mildly optimistic numbers ***")

    bf_y = bf[(bf["Season"] == args.year) & ~bf["ShortGame"].fillna(False)]
    X = prep(bf_y, art["bat_cols"], art["cat_levels"])
    print(f"Scoring {len(bf_y):,} batter-games, {args.year}...")

    results = {}
    for name, (target, _desc) in PROPS.items():
        p = predict_prop(art["props"][name], X)
        y = bf_y[target].to_numpy()
        base = np.full_like(p, y.mean())
        results[name] = {
            "p": p, "y": y, "d": bf_y["Date"].to_numpy(),
            "g": bf_y["GamePk"].to_numpy(),
            "auc": roc_auc_score(y, p),
            "edge": log_loss(y, base) - log_loss(y, p),
        }

    section_confidence(results, args.boot)
    section_topn(results)
    section_thresholds(results, args.prop)
    section_drift(results)
    section_segments(bf_y, results, args.prop)
    section_concentration(results)

    sf_y = sf[(sf["Season"] == args.year) & ~sf["ShortGame"].fillna(False)]
    section_strikeouts(sf_y, art)

    gf_y = gf[gf["Season"] == args.year].dropna(subset=["total_runs"])
    section_games(gf_y, art)

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
    P(over) is overconfident at extreme lines - shade those probabilities.""")


if __name__ == "__main__":
    main()
