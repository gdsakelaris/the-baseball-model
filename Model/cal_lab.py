"""Offline pricing-layer lab (2026-07-15 PM batch): re-fit the binary heads'
stack + calibrator under a knob grid — CAL_POOL_DECAY x FSTACK_C x
CAL_BAG_B — from the persisted calibration stash (artifacts/cal_stash.joblib,
written by every keep-chain), scored on the suite's TEST-year designs.
ZERO booster retrains: the boosters' cal/test logit designs are frozen in
the sidecar; only the small pricing fits (a logistic stack + a calibrator)
are re-run per combo, using train.py's OWN functions (_pool_cal,
_pick_calibrator, LogisticRegression stack) so the lab's arithmetic can
never drift from the chain's.

Discipline (true-test doctrine): the DEFAULT suite is SELECTION — its test
year (2025) is the iterate-freely year. --shipping scores the 2026 designs
and is a DELIBERATE confirm look; don't run it casually.

The knobs are GLOBAL train.py constants, so the actionable output is the
GLOBAL combo ranking (mean per-head deltas); the per-head tables are
evidence/diagnostics. Wire a winning combo into train.py's constants and
the next chain's paired read verdicts it.

Usage:
    python Model/cal_lab.py                  # selection suite, full grid
    python Model/cal_lab.py --heads hr,hit   # restrict
    python Model/cal_lab.py --shipping       # 2026 confirm look (deliberate)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T               # noqa: E402

ART = T.ART

# (pool_years, decay): 1 = current-year-only; 2 = + prior year; 3 = + two
# prior years (needs a --prestash chain's stash — prior k back weighted
# decay**k). C = stack ridge; bag = calibrator day-block bootstrap size.
POOL_GRID = ((1, None), (2, 0.5), (2, 0.75), (2, 1.0),
             (3, 0.5), (3, 0.75), (3, 1.0))
C_GRID = (10.0, 50.0, 200.0, 1e6)
BAG_GRID = (0, 25)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ece(y, p, n_bins=10):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _cal_slope(y, p):
    """(slope, intercept) of the logistic recalibration fit y ~ logit(p) —
    the standard cal-slope diagnostic (1, 0 = perfectly calibrated)."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def _score(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    s, i = _cal_slope(y, p)
    return {"logloss": float(log_loss(y, p)),
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "ece": _ece(y, p), "slope": round(s, 4), "intercept": round(i, 4)}


def _refit(cur, priors, te, pool_years, pool_decay, c, bag_b, name):
    """One knob combo: pool -> stack -> calibrator -> test scores. Reuses
    train's _pool_cal/_pick_calibrator by temporarily setting its globals
    (restored by the caller loop). priors = [(entry, years_back)] newest
    first; the combo uses the first pool_years-1 of them."""
    use = priors[:pool_years - 1]
    if not use or pool_decay is None:
        pool = dict(cur, w=None)
    else:
        T.CAL_POOL_DECAY = pool_decay
        pool = T._pool_cal(cur, use)
    stack = LogisticRegression(C=c, max_iter=1000)
    stack.fit(pool["Z"], pool["y"], sample_weight=pool["w"])
    s_pool = stack.predict_proba(pool["Z"])[:, 1]
    T.CAL_BAG_B = bag_b
    iso, kind = T._pick_calibrator(s_pool, pool["y"], pool["gamepk"], name,
                                   dates=pool["dates"], w=pool["w"])
    p_te = iso.predict(stack.predict_proba(te["Z"])[:, 1])
    m = _score(te["y"], p_te)
    m["cal_kind"] = kind
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="", help="comma list (else all)")
    ap.add_argument("--shipping", action="store_true",
                    help="score the SHIPPING suite's 2026 designs — a "
                         "deliberate confirm-only look, not for iteration")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    side = joblib.load(ART / "cal_stash.joblib")
    stash, suites = side["stash"], side["suites"]
    shipped = side["flags"]
    suite = suites["shipping" if args.shipping else "selection"]
    cal_yr, te_yr, prior_yr = suite["cal"], suite["test"], suite["train_last"]
    if args.shipping:
        log("!! SHIPPING suite: this is a deliberate 2026 confirm look")
    log(f"suite cal {cal_yr} / test {te_yr} / pooled prior {prior_yr} | "
        f"shipped flags {shipped}")

    heads = sorted({k[1] for k in stash
                    if k[0] == "prop" and k[2] == cal_yr})
    if ("winner", "winner", cal_yr) in stash:
        heads.append("winner")
    want = {h.strip() for h in args.heads.split(",") if h.strip()}
    if want:
        heads = [h for h in heads if h in want]

    saved = (T.CAL_POOL_DECAY, T.CAL_BAG_B)
    shipped_combo = ((2 if shipped["multi_year_cal"] else 1,
                      shipped["cal_pool_decay"] if shipped["multi_year_cal"]
                      else None),
                     shipped["fstack_c"], shipped["cal_bag_b"])
    results, combo_deltas = {}, {}
    try:
        for h in heads:
            kind = "winner" if h == "winner" else "prop"
            cur = stash[(kind, h, cal_yr)]
            te = stash[(f"{kind}_te", h, te_yr)]
            priors = []
            for k, yr in ((1, prior_yr), (2, suite.get("train_prev"))):
                e = stash.get((kind, h, yr)) if yr is not None else None
                if e is None or e["Z"].shape[1] != cur["Z"].shape[1]:
                    break
                priors.append((e, k))
            rows = {}
            for years, pd_ in POOL_GRID:
                if years - 1 > len(priors):
                    continue
                for c in C_GRID:
                    for bag in BAG_GRID:
                        m = _refit(cur, priors, te, years, pd_, c, bag, h)
                        rows[f"pool({years}, {pd_})_C{c:g}_bag{bag}"] = m
                        combo_deltas.setdefault(((years, pd_), c, bag),
                                                []).append(m["logloss"])
            base_key = (f"pool{shipped_combo[0]}_C{shipped_combo[1]:g}"
                        f"_bag{shipped_combo[2]}")
            base = rows.get(base_key)
            best_key = min(rows, key=lambda k: rows[k]["logloss"])
            results[h] = {"shipped": base_key, "rows": rows,
                          "best": best_key}
            b = rows[best_key]
            ship_str = (f"shipped {base_key}: ll {base['logloss']:.5f} "
                        f"slope {base['slope']} ece {base['ece']:.4f}"
                        if base else f"shipped combo {base_key} n/a")
            log(f"[{h}] {ship_str} | best {best_key}: "
                f"ll {b['logloss']:.5f} slope {b['slope']} "
                f"ece {b['ece']:.4f}")
    finally:
        T.CAL_POOL_DECAY, T.CAL_BAG_B = saved

    # global combo ranking — the actionable read (knobs are global consts)
    print("\n===== GLOBAL COMBO RANKING (mean logloss across heads) =====")
    ranked = sorted(combo_deltas.items(), key=lambda kv: np.mean(kv[1]))
    for (pool, c, bag), lls in ranked[:8]:
        tag = " <-- shipped" if (pool, c, bag) == shipped_combo else ""
        print(f"  pool={pool} C={c:g} bag={bag}: mean ll "
              f"{np.mean(lls):.5f} over {len(lls)} heads{tag}")
    out_path = Path(args.out) if args.out else ART / (
        f"cal_lab{'_shipping' if args.shipping else ''}.json")
    with open(out_path, "w") as f:
        json.dump({"suite": suite, "shipped_flags": shipped,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")
    print("Wire a winning GLOBAL combo into train.py constants "
          "(CAL_POOL_DECAY / FSTACK_C / CAL_BAG_B); the next chain's "
          "paired read verdicts it.")


if __name__ == "__main__":
    main()
