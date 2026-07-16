"""total_vmr_exp evidence fit (2026-07-15 PM batch — the Phase-5 pass the
audit-wave rank-32 mechanism has been waiting for since it shipped gated).

predict._vmr_scaled_disp already serves disp * (park_vmr/VMR0)^a for the
NB-priced game-total tails, activating ONLY when an exponent exists; this
script produces that exponent's 2025 evidence. Scope honesty: the STANDARD
total lines ship through the per-line logistic calibrators, so the vmr
scaling only prices the NB FALLBACK surface (exotic/odds-store lines); the
fit is scored on exactly that path — NB P(over) at the standard lines on
the SELECTION suite's test year (iterate-freely), pooled logloss over
game x line outcomes vs the constant-dispersion incumbent (a=0).

Writes artifacts/total_vmr_exp.json {exp, recommended, ...}; predict.py
falls back to this sidecar when the artifact carries no exponent, and
`recommended: false` (or deleting the file) keeps the mechanism inert.

Usage:  python Model/vmr_fit.py          (after a keep-chain)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                  # noqa: E402
import param_sweep as PS           # noqa: E402
from features import PARK_VMR0     # noqa: E402
from predict import nb_over, TOTAL_LINES  # noqa: E402

ART = T.ART
GRID = np.round(np.arange(0.0, 1.51, 0.1), 2)
EPS_LL = 0.0004        # same acceptance margin family as the sweeps
CLIP = (1.6, 3.0)      # predict._vmr_scaled_disp's dispersion clip


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    art = joblib.load(ART / "models_bt.joblib")
    stamp = art.get("meta_stamp") or {}
    if stamp.get("role") == "superset":
        sys.exit("models_bt.joblib is a superset intermediate — run after "
                 "a keep-chain")
    yrs = art["years"]
    te_yr = int(yrs["test"])
    log(f"selection artifact: test year {te_yr} (iterate-freely) | "
        f"total_disp {art['total_disp']:.3f}")

    bf, sf, tg, wf = PS.load_frames()
    T.set_categories(tg, art["cat_levels"])
    d = tg[tg["Season"] == te_yr]
    mu = art["team_runs_model"].predict(d[art["tg_cols"]])
    g = pd.DataFrame({"g": d["GamePk"].to_numpy(), "mu": mu,
                      "y": d["y_runs"].to_numpy(),
                      "vmr": d["park_vmr"].to_numpy()})
    per = g.groupby("g").agg(mu=("mu", "sum"), y=("y", "sum"),
                             vmr=("vmr", "first")).dropna(subset=["mu", "y"])
    vmr = per["vmr"].to_numpy(dtype=float)
    vmr = np.where(np.isfinite(vmr), vmr, PARK_VMR0)   # missing -> neutral
    mu_g, y_g = per["mu"].to_numpy(), per["y"].to_numpy()
    disp0 = float(art["total_disp"])
    log(f"{len(per)} games | park vmr range {vmr.min():.2f}-{vmr.max():.2f} "
        f"(VMR0 {PARK_VMR0})")

    def pooled_ll(a):
        ll, n = 0.0, 0
        disp = np.clip(disp0 * (vmr / PARK_VMR0) ** a, *CLIP)
        for line in TOTAL_LINES:
            over = (y_g > line).astype(float)
            p = np.array([nb_over(m, line, dd)
                          for m, dd in zip(mu_g, disp)])
            p = np.clip(p, 1e-6, 1 - 1e-6)
            ll += float(-(over * np.log(p)
                          + (1 - over) * np.log(1 - p)).sum())
            n += len(p)
        return ll / n

    lls = {}
    for a in GRID:
        lls[float(a)] = pooled_ll(float(a))
        log(f"  a={a:4.1f}: NB-path pooled logloss {lls[float(a)]:.5f}")
    base = lls[0.0]
    best_a = min(lls, key=lls.get)
    gain = base - lls[best_a]
    recommended = bool(best_a > 0 and gain >= EPS_LL)
    log(f"best a={best_a:.1f} | ll {lls[best_a]:.5f} vs a=0 {base:.5f} "
        f"(gain {gain:+.5f}, gate {EPS_LL}) -> "
        f"{'RECOMMEND' if recommended else 'keep constant dispersion'}")

    out = {"exp": float(best_a), "fit_year": te_yr, "n_games": len(per),
           "ll_best": lls[best_a], "ll_a0": base,
           "gain": gain, "recommended": recommended,
           "grid": {f"{a:.1f}": v for a, v in lls.items()},
           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ART / "total_vmr_exp.json").write_text(json.dumps(out, indent=1))
    log("wrote artifacts/total_vmr_exp.json"
        + (" (ACTIVE via predict sidecar fallback)" if recommended
           else " (inert — recommended: false)"))


if __name__ == "__main__":
    main()
