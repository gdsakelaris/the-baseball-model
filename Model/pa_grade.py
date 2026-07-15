"""PA-sim Phase 2 gate — grade the shadow backtest against the incumbent
heads on identical rows.

Incumbent per-row predictions come from the paired snapshots the baselines
already store (eval_paired_select_2025 / eval_paired_2026); sim predictions
from sim_backtest_*.parquet. Every comparison is an inner merge on row keys,
so both models are scored on exactly the same games, then a day-block
bootstrap CI on the per-resample delta (+ = SIM better) — the house paired
methodology, applied read-only.

Read across the board — every batter binary (sb included since the
2026-07-13 steal layer), the count means, total, winner, score. This is
evidence for the Phase-3 go/no-go, not a ship gate by itself.

    python Model/pa_grade.py --year 2025
"""

import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
BOOT = 400

BAT_PROPS = ["hr", "hit", "hits2", "tb2", "run", "rbi", "bb", "single",
             "double", "bk", "bk2", "hrr2", "hrr3",
             "sb"]                       # sb: steal layer (2026-07-13)
COUNT_MEANS = {"xbk": "xbk", "xhrr": "xhrr", "xtb": "xtb", "k": "xk",
               "outs": "xouts", "pbb": "xpbb", "pha": "xpha", "per": "xper"}


def _sim(year, kind):
    # Prefer part files (pa_backtest --part) over a same-year combined file:
    # the loose glob once concatenated a superseded combined parquet WITH the
    # fresh parts, silently doubling every game in the 07-15 blend fit.
    all_ = sorted(glob.glob(str(ART / f"sim_backtest_{kind}_{year}*.parquet")))
    parts = [p for p in all_ if "of" in Path(p).stem.rsplit("_", 1)[-1]]
    use = parts or all_
    df = pd.concat([pd.read_parquet(p) for p in use], ignore_index=True)
    keys = [k for k in ("GamePk", "PlayerId", "Home") if k in df.columns]
    ndup = int(df.duplicated(subset=keys).sum())
    if ndup:
        raise RuntimeError(
            f"sim_backtest_{kind}_{year}: {ndup} duplicate rows across "
            f"{[Path(p).name for p in use]} — stale/overlapping parquets; "
            "delete the superseded files and re-run.")
    return df


def _block_delta(dates, inc_rows, sim_rows, stat, y=None):
    """Bootstrap days; stat over resampled rows for both models; CI on
    delta (+ = sim better)."""
    rng = np.random.default_rng(0)
    d = pd.factorize(dates)[0]
    idx_by_day = [np.flatnonzero(d == i) for i in range(d.max() + 1)]
    deltas = []
    for _ in range(BOOT):
        take = rng.integers(0, len(idx_by_day), len(idx_by_day))
        rows = np.concatenate([idx_by_day[t] for t in take])
        if stat == "auc":
            yv = y[rows]
            if yv.min() == yv.max():
                continue
            deltas.append(roc_auc_score(yv, sim_rows[rows])
                          - roc_auc_score(yv, inc_rows[rows]))
        elif stat == "logloss":
            yv = y[rows]
            deltas.append(log_loss(yv, np.clip(inc_rows[rows], 1e-6, 1-1e-6))
                          - log_loss(yv, np.clip(sim_rows[rows], 1e-6, 1-1e-6)))
        else:                                   # mae rowwise abs errors
            deltas.append(inc_rows[rows].mean() - sim_rows[rows].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def verdict(lo, hi):
    return "SIM+" if lo > 0 else "INC+" if hi < 0 else "tie"


def main(year):
    tag = "select_2025" if year == 2025 else "2026"
    snap = joblib.load(ART / f"eval_paired_{tag}.joblib")
    bat, st, gm = _sim(year, "bat"), _sim(year, "starter"), _sim(year, "game")
    rows = []

    for prop in BAT_PROPS:
        e = snap["binary"][prop]["df"]
        m = e.merge(bat[["GamePk", "PlayerId", f"p_{prop}"]],
                    on=["GamePk", "PlayerId"], how="inner")
        y = m["y"].to_numpy(float)
        inc, sim = m["p"].to_numpy(), m[f"p_{prop}"].to_numpy()
        d_auc, lo_a, hi_a = _block_delta(m["Date"].to_numpy(), inc, sim,
                                         "auc", y)
        d_ll, lo_l, hi_l = _block_delta(m["Date"].to_numpy(), inc, sim,
                                        "logloss", y)
        rows.append({
            "head": prop, "n": len(m),
            "inc_auc": round(roc_auc_score(y, inc), 4),
            "sim_auc": round(roc_auc_score(y, sim), 4),
            "dAUC": round(d_auc, 4), "auc_v": verdict(lo_a, hi_a),
            "dLL": round(d_ll, 4), "ll_v": verdict(lo_l, hi_l)})

    for head, col in COUNT_MEANS.items():
        e = snap["count"][head]["df"]
        src = bat if head in ("xbk", "xhrr", "xtb") else st
        scol = col if col in src.columns else None
        if scol is None:
            continue
        m = e.merge(src[["GamePk", "PlayerId", scol]],
                    on=["GamePk", "PlayerId"], how="inner")
        y = m["y"].to_numpy(float)
        inc_err = np.abs(y - m["mu"].to_numpy())
        sim_err = np.abs(y - m[scol].to_numpy())
        d, lo, hi = _block_delta(m["Date"].to_numpy(), inc_err, sim_err,
                                 "mae")
        rows.append({"head": head, "n": len(m),
                     "inc_auc": round(inc_err.mean(), 4),
                     "sim_auc": round(sim_err.mean(), 4),
                     "dAUC": round(d, 4), "auc_v": verdict(lo, hi),
                     "dLL": np.nan, "ll_v": "(MAE)"})

    e = snap["count"]["total"]["df"]
    m = e.merge(gm[["GamePk", "x_total"]], on="GamePk", how="inner")
    y = m["y"].to_numpy(float)
    inc_err = np.abs(y - m["mu"].to_numpy())
    sim_err = np.abs(y - m["x_total"].to_numpy())
    d, lo, hi = _block_delta(m["Date"].to_numpy(), inc_err, sim_err, "mae")
    rows.append({"head": "total", "n": len(m),
                 "inc_auc": round(inc_err.mean(), 4),
                 "sim_auc": round(sim_err.mean(), 4),
                 "dAUC": round(d, 4), "auc_v": verdict(lo, hi),
                 "dLL": np.nan, "ll_v": "(MAE)"})

    e = snap["winner"]["df"]
    m = e.merge(gm[["GamePk", "p_home_win"]], on="GamePk", how="inner")
    y = m["y"].to_numpy(float)
    inc, sim = m["p"].to_numpy(), m["p_home_win"].to_numpy()
    d_auc, lo_a, hi_a = _block_delta(m["Date"].to_numpy(), inc, sim,
                                     "auc", y)
    d_ll, lo_l, hi_l = _block_delta(m["Date"].to_numpy(), inc, sim,
                                    "logloss", y)
    rows.append({"head": "winner", "n": len(m),
                 "inc_auc": round(roc_auc_score(y, inc), 4),
                 "sim_auc": round(roc_auc_score(y, sim), 4),
                 "dAUC": round(d_auc, 4), "auc_v": verdict(lo_a, hi_a),
                 "dLL": round(d_ll, 4), "ll_v": verdict(lo_l, hi_l)})

    e = snap["score"]["df"]
    sides = gm.melt(id_vars=["GamePk"], value_vars=["x_away", "x_home"],
                    var_name="side", value_name="x_score")
    sides["Home"] = (sides["side"] == "x_home").astype(e["Home"].dtype)
    m = e.merge(sides[["GamePk", "Home", "x_score"]],
                on=["GamePk", "Home"], how="inner")
    y = m["y"].to_numpy(float)
    inc_err = np.abs(y - m["mu"].to_numpy())
    sim_err = np.abs(y - m["x_score"].to_numpy())
    d, lo, hi = _block_delta(m["Date"].to_numpy(), inc_err, sim_err, "mae")
    rows.append({"head": "score", "n": len(m),
                 "inc_auc": round(inc_err.mean(), 4),
                 "sim_auc": round(sim_err.mean(), 4),
                 "dAUC": round(d, 4), "auc_v": verdict(lo, hi),
                 "dLL": np.nan, "ll_v": "(MAE)"})

    df = pd.DataFrame(rows)
    print(f"\n=== SIM vs INCUMBENT, {year} (identical rows; + = sim better; "
          f"binaries: AUC & logloss deltas w/ day-block CI; counts: MAE — "
          f"inc/sim cols show MAE there) ===")
    print(df.to_string(index=False))
    df.to_csv(ART / f"sim_grade_{year}.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    main(ap.parse_args().year)
