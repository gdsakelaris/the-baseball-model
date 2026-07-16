"""Forward-record calibration monitor (2026-07-15 PM batch): the user's #2/#3
diagnostics — calibration slope/intercept and ECE — measured directly on the
TIMESTAMPED FORWARD RECORD, per head, pooled across every graded workbook in
Predictions/. The true-test doctrine says that record is the only real test;
this wires the betting-critical calibration read to it instead of waiting for
an era audit.

Reuses 4_grade_results.grade() (the same cell->outcome settlement the daily
report uses, painting included — idempotent, same as --all), so this monitor
and the day reports can never disagree about what happened.

Reads honestly at small n: per-head slope/ECE need a few hundred graded cells
to mean anything, so heads below the support floor print with a '~' flag and
the summary counts them separately. A drift column compares the last 14 days
vs the full record once both have support.

Usage:
    python Tools/6_forward_calibration.py            # full record
    python Tools/6_forward_calibration.py --days 30  # restrict window
    python Tools/6_forward_calibration.py --json out.json
"""
import argparse
import importlib
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
GR = importlib.import_module("4_grade_results")

MIN_N = 200          # cells below this -> '~' (directional only)
DRIFT_DAYS = 14


def _slope_intercept(p, y):
    """Logistic recalibration fit y ~ logit(p): (slope, intercept); the
    2-parameter Newton from features.PlattCal's math, dependency-free."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    y = np.asarray(y, float)
    a, b = 1.0, 0.0
    for _ in range(50):
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        w = np.clip(q * (1 - q), 1e-6, None)
        g0, g1 = float(((q - y) * z).sum()), float((q - y).sum())
        h00 = float((w * z * z).sum()) + 1e-4
        h01 = float((w * z).sum())
        h11 = float(w.sum()) + 1e-4
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or det <= 0:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a, b = a - da, b - db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return float(a), float(b)


def _ece(p, y, n_bins=10):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _head_metrics(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    s, i = _slope_intercept(p, y)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    ll0 = float(-(y * np.log(p0) + (1 - y) * np.log(1 - p0)).mean())
    return {"n": int(len(y)), "actual": round(float(y.mean()), 4),
            "stated": round(float(p.mean()), 4),
            "slope": round(s, 3), "intercept": round(i, 3),
            "ece": round(_ece(p, y), 4),
            "logloss": round(ll, 4), "logloss_base": round(ll0, 4),
            "auc": (None if GR._rank_auc(p, y.astype(bool)) is None
                    else round(GR._rank_auc(p, y.astype(bool)), 3))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=0,
                    help="only the most recent N days (0 = full record)")
    ap.add_argument("--json", default=None, help="also write metrics here")
    args = ap.parse_args()

    books = sorted(GR.PRED_DIR.glob("[0-9]*.xlsx"))
    if not books:
        sys.exit(f"no workbooks in {GR.PRED_DIR}")
    recs = []          # (date, sheet, head, p, occ)
    for path in books:
        try:
            date, _stats, rows, _painted = GR.grade(path)
        except GR.GradeError:
            continue
        d = pd.Timestamp(date)
        recs += [(d, s, h, p, o) for s, h, p, o in rows]
    if not recs:
        sys.exit("nothing gradeable yet — the record has no box scores")

    df = pd.DataFrame(recs, columns=["date", "sheet", "head", "p", "y"])
    last = df["date"].max()
    if args.days:
        df = df[df["date"] > last - timedelta(days=args.days)]
    n_days = df["date"].nunique()
    print(f"\n=== FORWARD-RECORD CALIBRATION ({n_days} day(s), "
          f"{len(df):,} graded cells, through {last.date()}) ===")
    print("slope/intercept: 1, 0 = perfectly calibrated; heads under "
          f"{MIN_N} cells print '~' (directional only)")
    hdr = (f"  {'head':14s} {'n':>6s} {'stated%':>8s} {'actual%':>8s}"
           f" {'slope':>7s} {'icept':>7s} {'ece':>7s} {'logloss':>8s}"
           f" {'base_ll':>8s} {'auc':>6s} {'drift':>7s}")
    out = {}
    cut = last - timedelta(days=DRIFT_DAYS)
    for sheet in dict.fromkeys(df["sheet"]):
        print(f"\n  -- {sheet} --")
        print(hdr)
        sub = df[df["sheet"] == sheet]
        for head in dict.fromkeys(sub["head"]):
            g = sub[sub["head"] == head]
            m = _head_metrics(g["p"], g["y"])
            recent = g[g["date"] > cut]
            drift = ""
            if len(recent) >= MIN_N and len(g) - len(recent) >= MIN_N:
                s_r, _ = _slope_intercept(recent["p"], recent["y"])
                drift = f"{s_r - m['slope']:+.2f}"
            flag = "~" if m["n"] < MIN_N else " "
            out[f"{sheet}/{head}"] = dict(m, drift_slope=drift or None)
            print(f" {flag}{head:14s} {m['n']:6d} {m['stated']:8.1%}"
                  f" {m['actual']:8.1%} {m['slope']:7.3f}"
                  f" {m['intercept']:7.3f} {m['ece']:7.4f}"
                  f" {m['logloss']:8.4f} {m['logloss_base']:8.4f}"
                  + (f" {m['auc']:6.3f}" if m['auc'] is not None
                     else "      -")
                  + f" {drift:>7s}")
    thin = sum(1 for v in out.values() if v["n"] < MIN_N)
    print(f"\n{len(out)} head-surfaces; {thin} still under the {MIN_N}-cell "
          f"support floor. Drift = last-{DRIFT_DAYS}-day slope minus "
          f"full-record slope (needs {MIN_N}+ cells on both sides).")
    print("Feeds: the Aug bb-recal check (Section-10 evidence), era-audit "
          "watch ledger, and any in-season offset decision.")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"through": str(last.date()), "days": n_days,
             "heads": out}, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
