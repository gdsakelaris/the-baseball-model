"""Evaluate the trained models' accuracy and betting effectiveness.

Runs the shipped models over a full season of real games and reports, per
prop: discrimination (AUC), accuracy vs baselines (log loss / Brier),
calibration (do predicted probabilities match reality?), and ranking value
(daily top-N hit rate — what a bettor cares about). Also evaluates the
strikeout and total-runs models against naive baselines, and a monthly
stability breakdown for the HR model.

The shipped models were trained on 2020-2024 and calibrated on 2025, so:
  --year 2026 (default)  true holdout — honest out-of-sample numbers
  --year 2025            calibration year — mildly optimistic
  --year <=2024          training data — IN-SAMPLE, numbers are inflated

To measure whether a change helped:
    python Model/evaluate.py --set-baseline   # snapshot current model
    ...make changes, retrain...
    python Model/evaluate.py                   # prints Δ vs the baseline

Usage:
    python Model/evaluate.py [--year 2026] [--top 10] [--prop hr]
                             [--set-baseline]
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
from train import PROPS  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"


def prep(df, cols, cat_levels):
    X = df[cols].copy()
    for c, levels in cat_levels.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
    return X


def eval_prop(name, target, bf_year, art, top_n):
    from predict import predict_prop
    X = prep(bf_year, art["bat_cols"], art["cat_levels"])
    p = predict_prop(art["props"][name], X)
    y = bf_year[target].to_numpy()
    base = np.full_like(p, y.mean())
    day = pd.DataFrame({"d": bf_year["Date"].values, "p": p, "y": y})
    top = day.sort_values("p", ascending=False).groupby("d").head(top_n)
    return {
        "prop": name, "n": len(y), "base_rate": y.mean(),
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, p), "logloss_base": log_loss(y, base),
        "brier": brier_score_loss(y, p), "brier_base": brier_score_loss(y, base),
        f"top{top_n}_day": top["y"].mean(),
        "p": p, "y": y,
    }


def calibration_table(p, y, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"pred": p, "actual": y}).groupby(q, observed=True).agg(
        predicted=("pred", "mean"), actual=("actual", "mean"), n=("actual", "size"))


# metrics where LOWER is better (everything else: higher is better)
LOWER_BETTER = {"logloss", "mae"}


def _handle_baseline(summary, args):
    base_path = ART / f"eval_baseline_{args.year}.json"
    if args.set_baseline:
        base_path.write_text(json.dumps(summary, indent=2))
        print(f"\nBaseline saved to {base_path.name} "
              f"({len(summary)} metrics). Re-run without --set-baseline "
              f"after changes to see deltas.")
        return
    if not base_path.exists():
        print(f"\n(No baseline yet. Run `python Model/evaluate.py "
              f"--set-baseline` to snapshot this model for future comparison.)")
        return
    base = json.loads(base_path.read_text())
    rows = []
    for k, now in summary.items():
        if k not in base:
            continue
        was = base[k]
        delta = now - was
        lower = any(k.endswith(s) for s in LOWER_BETTER)
        better = (delta < 0) if lower else (delta > 0)
        mark = "same" if abs(delta) < 5e-5 else ("better" if better else "worse")
        rows.append({"metric": k, "baseline": was, "now": now,
                     "delta": f"{delta:+.4f}", "": mark})
    print(f"\n=== Change vs baseline (eval_baseline_{args.year}.json) ===")
    print(pd.DataFrame(rows).to_string(index=False))
    improved = sum(1 for r in rows if r[""] == "better")
    worse = sum(1 for r in rows if r[""] == "worse")
    print(f"\n{improved} metrics improved, {worse} worse, "
          f"{len(rows) - improved - worse} unchanged.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--top", type=int, default=10,
                    help="daily top-N for the ranking metric")
    ap.add_argument("--prop", default="hr",
                    help="prop to show the calibration table for")
    ap.add_argument("--set-baseline", action="store_true",
                    help="save this run's metrics as the baseline to diff against")
    args = ap.parse_args()

    summary = {}  # flat scalar metrics for baseline comparison

    art = joblib.load(ART / "models.joblib")
    frames = joblib.load(ART / "frames.joblib")
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]

    if args.year <= 2024:
        print("*** WARNING: this year was TRAINING data - numbers are "
              "in-sample and inflated. Use 2026 for honest evaluation. ***\n")
    elif args.year == 2025:
        print("*** NOTE: 2025 is the calibration year - numbers are mildly "
              "optimistic. Use 2026 for honest evaluation. ***\n")

    bf_y = bf[(bf["Season"] == args.year) & ~bf["ShortGame"].fillna(False)]
    print(f"=== Batter props, {args.year} ({len(bf_y):,} batter-games, "
          f"daily top-{args.top}) ===")
    rows = []
    results = {}
    for name, (target, desc) in PROPS.items():
        r = eval_prop(name, target, bf_y, art, args.top)
        results[name] = r
        rows.append({
            "prop": f"{name} ({desc})", "base rate": f'{r["base_rate"]:.1%}',
            "AUC": f'{r["auc"]:.3f}',
            "logloss vs base": f'{r["logloss"]:.4f} / {r["logloss_base"]:.4f}',
            "Brier vs base": f'{r["brier"]:.4f} / {r["brier_base"]:.4f}',
            f"top{args.top}/day": f'{r[f"top{args.top}_day"]:.1%}',
            "lift": f'{r[f"top{args.top}_day"] / r["base_rate"]:.2f}x',
        })
        summary[f"{name}_auc"] = round(r["auc"], 4)
        summary[f"{name}_logloss"] = round(r["logloss"], 4)
        summary[f"{name}_top{args.top}"] = round(r[f"top{args.top}_day"], 4)
    print(pd.DataFrame(rows).to_string(index=False))

    r = results[args.prop]
    print(f"\n=== Calibration, {args.prop} ({args.year}) — predicted vs actual "
          f"by decile ===")
    print(calibration_table(r["p"], r["y"]).to_string())

    print(f"\n=== HR model monthly stability ({args.year}) ===")
    hr = results["hr"]
    m = pd.DataFrame({"month": pd.to_datetime(bf_y["Date"]).dt.month.values,
                      "p": hr["p"], "y": hr["y"]})
    print(m.groupby("month").agg(n=("y", "size"), base=("y", "mean"),
                                 pred=("p", "mean")).round(4).to_string())

    # strikeouts
    sf_y = sf[(sf["Season"] == args.year) & ~sf["ShortGame"].fillna(False)]
    Xs = prep(sf_y, art["st_cols"], art["cat_levels"])
    kp = art["k_model"].predict(Xs)
    ky = sf_y["y_so"].to_numpy()
    per_start = sf_y["ps_k_bf"] * (sf_y["ps_BF"] / sf_y["p_starts_season"])
    k_base = per_start.fillna(ky.mean()).clip(0, 15)
    summary["k_mae"] = round(mean_absolute_error(ky, kp), 4)
    print(f"\n=== Starter strikeouts ({args.year}, {len(ky):,} starts) ===")
    print(f"  model MAE {mean_absolute_error(ky, kp):.3f} | baseline "
          f"(pitcher season rate) {mean_absolute_error(ky, k_base):.3f} | "
          f"mean pred {kp.mean():.2f} vs actual {ky.mean():.2f}")

    # per-team runs, game totals, and winner picks
    from predict import poisson_win, predict_win
    gf_y = gf[(gf["Season"] == args.year)].dropna(subset=["total_runs"])
    tg = F.build_team_game_frame(gf_y)
    Xt = prep(tg, art["tg_cols"], art["cat_levels"])
    tp = art["team_runs_model"].predict(Xt)
    ty = tg["y_runs"].to_numpy()
    n_g = len(gf_y)
    mu_away, mu_home = tp[:n_g], tp[n_g:]
    away_y = pd.to_numeric(gf_y["AwayScore"], errors="coerce").to_numpy()
    home_y = pd.to_numeric(gf_y["HomeScore"], errors="coerce").to_numpy()

    print(f"\n=== Team runs / totals / winner ({args.year}, {n_g:,} games) ===")
    print(f"  per-team runs MAE {mean_absolute_error(ty, tp):.3f} | baseline "
          f"(team season rate) "
          f"{mean_absolute_error(ty, tg['off_r_pg'].fillna(ty.mean())):.3f}")
    total_p = mu_away + mu_home
    total_y = away_y + home_y
    print(f"  game total MAE {mean_absolute_error(total_y, total_p):.3f} | "
          f"baseline (league mean) "
          f"{mean_absolute_error(total_y, np.full_like(total_p, total_y.mean())):.3f}")
    # winner: the dedicated home-win classifier when the artifact has one
    # (falls back to comparing Poisson means for pre-upgrade artifacts)
    win_cols = art.get("win_model", {}).get("cols", [])
    if win_cols and all(c in gf_y.columns for c in win_cols):
        Xw = prep(gf_y, win_cols, art["cat_levels"])
        p_home = predict_win(art["win_model"], Xw, mu_home, mu_away)
        win_src = "win model"
    else:
        p_home = np.array([poisson_win(h, a) for h, a in zip(mu_home, mu_away)])
        win_src = "poisson means (no win model in artifacts)"
    pick_home = p_home >= 0.5
    actual_home = home_y > away_y
    acc = (pick_home == actual_home).mean()
    summary["team_runs_mae"] = round(mean_absolute_error(ty, tp), 4)
    summary["total_mae"] = round(mean_absolute_error(total_y, total_p), 4)
    summary["winner_acc"] = round(acc, 4)
    summary["winner_logloss"] = round(
        log_loss(actual_home, np.clip(p_home, 1e-4, 1 - 1e-4)), 4)
    print(f"  winner pick accuracy {acc:.1%} ({win_src}) | baseline "
          f"(always home team) {actual_home.mean():.1%} | win-prob log loss "
          f"{log_loss(actual_home, np.clip(p_home, 1e-4, 1 - 1e-4)):.4f} "
          f"(base {log_loss(actual_home, np.full_like(p_home, actual_home.mean())):.4f})")

    _handle_baseline(summary, args)

    print("""
How to read this:
  - AUC > 0.5 = the model ranks players better than chance; 0.60+ is strong
    for per-game player props, which are extremely noisy.
  - logloss/Brier below the base-rate baseline = probabilities carry real
    information, not just ranking.
  - Calibration: 'predicted' and 'actual' should track closely by decile.
    That means a 20% prediction really hits ~20% of the time.
  - topN/day lift = how much better the model's daily favorites hit than a
    random pick. This is the number closest to betting value, but beating
    the vig also requires beating the sportsbook's own line - track model
    picks against real prices before staking anything.""")


if __name__ == "__main__":
    main()
