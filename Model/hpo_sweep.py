"""Per-head Optuna TPE hyperparameter search (2026-07-15 PM batch, item 5).

param_sweep.py picks from a curated 6-8 profile ladder; this script searches
the CONTINUOUS LightGBM space (lr / leaves / min_child / colsample / lambda /
subsample / max_bin / min_split_gain) with a TPE sampler, reusing
param_sweep's exact machinery so the two sweeps can never disagree on
methodology: same frames (load_frames), same Season<=cutoff window (the
selection test year stays pristine for evaluate_deep --paired), same
day-block GroupKFold with the 15% day-grouped ES/calibration watch, same
honest-OOF scoring (_fold_fit_binary / _fold_fit_count), and the same
deviance/ECE-guarded recommendation gates (EPS_LL / AUC_BAND / ECE_BAND /
EPS_DEV / MAE_BAND).

Where to aim it (the board's soft spots, 07-15 baselines): double (AUC .55),
hit/single/rbi (~.57), the hrr family's 2025->2026 fade, run2/rbi2. The
DEFAULT_HEADS list below encodes that; --heads overrides.

CAVEATS (same as param_sweep, sharpened):
  * Single-bag LGBM CV vs the multi-family bag that ships — winners are
    EVIDENCE, not law; wire them into train.PROP_PARAMS / COUNT_PARAMS and
    the chain's evaluate_deep --paired read verdicts them.
  * TPE on 50-100 trials has its own selection optimism; the gates require
    the best trial to beat the head's CURRENT config (default profile =
    train.PROP_PARAMS entry when present, else the global base) by the
    same margins param_sweep demands, on the same folds.
  * Deterministic: TPESampler(seed=--seed), fixed folds — a re-run
    reproduces bit-identical trials.

Usage:
  python Model/hpo_sweep.py                          # weak-head list, 60 trials
  python Model/hpo_sweep.py --heads double,hit --trials 100
  python Model/hpo_sweep.py --trials 30 --max-rows 80000   # quick pass
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F           # noqa: E402
import train as T              # noqa: E402
import param_sweep as PS       # noqa: E402

ART = T.ART

# The board's current soft spots (07-15 eval baselines) — the compute goes
# where the AUC/edge headroom is, per the dynamic-priority rule.
DEFAULT_HEADS = ["double", "hit", "single", "rbi", "hrr2", "hrr3", "hrr4",
                 "run2", "rbi2"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _space(trial, kind):
    """Sampled LightGBM overrides. Ranges bracket every profile the curated
    ladders span, plus min_split_gain/max_bin/subsample which the ladders
    never touched."""
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06,
                                             log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 600,
                                               log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "max_bin": trial.suggest_categorical("max_bin", [127, 255, 511]),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
    }
    return p


def _oof(job, params, folds, seed, X, y, groups, fold_splits):
    """Honest OOF metrics for one params dict — param_sweep's cv_head inner
    loop for a single profile (no family cache: LGBM-only isolation, the
    same regime the curated sweep runs in)."""
    oof = np.full(len(y), np.nan)
    for (fit_idx, watch_idx, va_idx) in fold_splits:
        if job["kind"] == "binary":
            oof[va_idx] = PS._fold_fit_binary(X, y, fit_idx, watch_idx,
                                              va_idx, params, seed)
        else:
            oof[va_idx] = PS._fold_fit_count(X, y, fit_idx, watch_idx,
                                             va_idx, params, job["tweedie"],
                                             seed)
    m = {}
    if job["kind"] == "binary":
        from sklearn.metrics import log_loss, roc_auc_score
        p = np.clip(oof, 1e-6, 1 - 1e-6)
        m["logloss"] = float(log_loss(y, p))
        m["auc"] = float(roc_auc_score(y, p))
        m["ece"] = float(PS._ece(y, p))
    else:
        from sklearn.metrics import (mean_absolute_error,
                                     mean_poisson_deviance,
                                     mean_tweedie_deviance)
        m["mae"] = float(mean_absolute_error(y, oof))
        try:
            if job["tweedie"] is not None:
                m["deviance"] = float(mean_tweedie_deviance(
                    y, oof, power=job["tweedie"]))
            else:
                m["deviance"] = float(mean_poisson_deviance(y, oof))
        except ValueError:
            m["deviance"] = float("nan")
    return m


def _fold_splits(X, y, groups, folds, seed):
    """param_sweep.cv_head's fold construction, extracted verbatim so both
    sweeps score on identical splits."""
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    gkf = GroupKFold(n_splits=folds)
    out = []
    for tr_idx, va_idx in gkf.split(X, y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        sub_fit, sub_watch = next(gss.split(tr_idx, groups=groups[tr_idx]))
        out.append((tr_idx[sub_fit], tr_idx[sub_watch], va_idx))
    return out


def main():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="",
                    help="comma list (default: the weak-head list)")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    want = ([h.strip() for h in args.heads.split(",") if h.strip()]
            or DEFAULT_HEADS)
    bf, sf, tg, wf = PS.load_frames()
    train_yrs, cal_yr, hold_yr = T.suite_years(bf)
    cutoff = train_yrs[-1]
    log(f"selection window: Season <= {cutoff} (test {cal_yr} untouched) | "
        f"trials {args.trials} folds {args.folds} max_rows {args.max_rows}")
    out_path = Path(args.out) if args.out else ART / "hpo_sweep.json"

    jobs = {j["name"]: j for j in PS.build_jobs(bf, sf, tg, wf)}
    order = [h for h in PS.HEAD_ORDER if h in jobs and h in set(want)]
    skipped = [h for h in want if h not in order]
    if skipped:
        log(f"  ! not sweepable here, skipped: {skipped}")

    results = {}
    for name in order:
        j = jobs[name]
        X, y, groups = PS._prep(j["frame"], j["cols"], j["target"], cutoff,
                                args.max_rows, args.seed)
        fold_splits = _fold_splits(X, y, groups, args.folds, args.seed)
        # the bar to clear = the head's CURRENT shipped config, not the
        # global base: PROP_PARAMS/COUNT_PARAMS override when present
        current = dict(T.PROP_PARAMS.get(name)
                       or T.COUNT_PARAMS.get(name)
                       or j["base"])
        t0 = time.time()
        m_def = _oof(j, current, args.folds, args.seed, X, y, groups,
                     fold_splits)
        log(f"[{name}] {j['kind']} | rows {len(y):,} | current-config OOF "
            + (f"ll {m_def['logloss']:.5f} auc {m_def['auc']:.4f} "
               f"ece {m_def['ece']:.4f}" if j["kind"] == "binary" else
               f"dev {m_def['deviance']:.5f} mae {m_def['mae']:.4f}"))

        def objective(trial, _j=j, _cur=current, _fs=fold_splits,
                      _X=X, _y=y, _g=groups):
            params = dict(_j["base"], **_space(trial, _j["kind"]))
            m = _oof(_j, params, args.folds, args.seed, _X, _y, _g, _fs)
            for k, v in m.items():
                trial.set_user_attr(k, v)
            return (m["logloss"] if _j["kind"] == "binary"
                    else m["deviance"])

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.trials,
                       show_progress_bar=False)
        best = study.best_trial
        m_best = dict(best.user_attrs)

        # param_sweep's gates, vs the CURRENT config
        if j["kind"] == "binary":
            passed = (m_best["logloss"] <= m_def["logloss"] - PS.EPS_LL
                      and m_best["auc"] >= m_def["auc"] - PS.AUC_BAND
                      and m_best["ece"] <= m_def["ece"] + PS.ECE_BAND)
            tag = (f"ll {m_best['logloss']:.5f} ({m_best['logloss']-m_def['logloss']:+.5f}) "
                   f"auc {m_best['auc']:.4f} ({m_best['auc']-m_def['auc']:+.4f}) "
                   f"ece {m_best['ece']:.4f}")
        else:
            passed = (not np.isnan(m_best.get("deviance", np.nan))
                      and m_best["deviance"] <= m_def["deviance"] - PS.EPS_DEV
                      and m_best["mae"] <= m_def["mae"] + PS.MAE_BAND)
            tag = (f"dev {m_best['deviance']:.5f} "
                   f"({m_best['deviance']-m_def['deviance']:+.5f}) "
                   f"mae {m_best['mae']:.4f}")
        results[name] = {"kind": j["kind"], "n_rows": int(len(y)),
                         "trials": args.trials,
                         "current_config": {k: v for k, v in current.items()
                                            if k != "verbose"},
                         "current_oof": m_def,
                         "best_params": best.params,
                         "best_oof": m_best,
                         "recommended": bool(passed)}
        log(f"[{name}] DONE in {time.time()-t0:.0f}s | best trial "
            f"#{best.number} | {tag} | "
            f"{'RECOMMEND' if passed else 'keep current (gates not cleared)'}")
        with open(out_path, "w") as f:
            json.dump({"cutoff": int(cutoff),
                       "test_year_untouched": int(cal_yr),
                       "folds": args.folds, "max_rows": args.max_rows,
                       "seed": args.seed, "gates": {
                           "EPS_LL": PS.EPS_LL, "AUC_BAND": PS.AUC_BAND,
                           "ECE_BAND": PS.ECE_BAND, "EPS_DEV": PS.EPS_DEV,
                           "MAE_BAND": PS.MAE_BAND},
                       "results": results}, f, indent=2)

    print("\n===== HPO SUMMARY =====")
    winners = [n for n in order if results[n]["recommended"]]
    for name in order:
        r = results[name]
        flag = " *" if r["recommended"] else ""
        prim = ("logloss" if r["kind"] == "binary" else "deviance")
        print(f"  {name:8s} {prim} {r['current_oof'][prim]:.5f} -> "
              f"{r['best_oof'][prim]:.5f}{flag}")
    print(f"\n{len(winners)}/{len(order)} heads cleared the gates: "
          f"{', '.join(winners) if winners else '(none)'}")
    if winners:
        print("\npaste-ready PROP_PARAMS / COUNT_PARAMS overrides "
              "(dict(LGB_CLS/LGB_POIS, **...) form):")
        for n in winners:
            print(f'    "{n}": dict({"LGB_CLS" if results[n]["kind"] == "binary" else "LGB_POIS"}, '
                  + ", ".join(f"{k}={v!r}" for k, v in
                              results[n]["best_params"].items()) + "),")
    print(f"\nwrote {out_path}")
    print("Wire winners into train.py, then the chain's evaluate_deep "
          "--paired read verdicts them (single-bag CV vs the multi-family "
          "bag caveat applies).")


if __name__ == "__main__":
    main()
