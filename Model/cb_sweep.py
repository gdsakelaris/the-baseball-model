"""CatBoost-side hyperparameter sweep (2026-07-15 PM batch, chain-2 cargo).

The asymmetry this fixes: LGBM has had three sweep generations, but
CB_CLS/CB_POIS are hand-set 07-10 defaults that have NEVER been searched —
and the per-family stack now assigns CB real weight. param_sweep's caching
trick, inverted: per fold the LGBM member (the head's CURRENT shipped
params) is fit ONCE and cached; each CB profile fits one CB member per
fold, scored as the weighted ensemble (LGBM x LGBM_BAGS + CB x CB_BAGS —
the single members proxy their bags, param_sweep's convention).

CB params are GLOBAL in train.py (one CB_CLS for all binaries, one CB_POIS
for all counts — no per-head CB override mechanism exists), so the
actionable output is the GLOBAL profile ranking over a representative head
panel: mean primary-metric delta + heads-won, gated by param_sweep's
EPS/band rules per head. Wire ONE winning fragment into CB_CLS / CB_POIS;
the next keep-chain's paired read verdicts it.

Same honesty rules as every sweep here: day-block GroupKFold, Season<=cutoff
only, watch-set ES + isotonic, single members proxy bags (documented
caveat), nothing auto-applies.

Usage:
  python Model/cb_sweep.py                    # representative panel
  python Model/cb_sweep.py --heads hr,k --folds 4
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (log_loss, roc_auc_score, mean_absolute_error,
                             mean_poisson_deviance, mean_tweedie_deviance)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F           # noqa: E402
import train as T              # noqa: E402
import param_sweep as PS       # noqa: E402

ART = T.ART

# Representative panel: thick + thin binaries across stat families, the
# flagship counts, the game heads — a GLOBAL decision needs breadth, not
# the full 35-head board (each head x fold x profile = one GPU CB fit).
REP_HEADS = ["hr", "hit", "tb2", "bk", "hrr2", "sb", "triple",
             "k", "xtb", "total", "winner"]

CB_PROFILES = {
    "default":      {},                                  # depth8 lr.03 l2:3
    "depth6":       dict(depth=6),
    "depth10":      dict(depth=10),
    "lr_slow":      dict(learning_rate=0.02),
    "lr_fast":      dict(learning_rate=0.05),
    "l2_up":        dict(l2_leaf_reg=9.0),
    "l2_dn":        dict(l2_leaf_reg=1.0),
    "deep_reg":     dict(depth=10, l2_leaf_reg=9.0),
    "shallow_fast": dict(depth=6, learning_rate=0.05),
    "bayes_hot":    dict(bagging_temperature=2.0),
    "bayes_cold":   dict(bagging_temperature=0.25),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _lgbm_params(name, kind, base):
    cur = T.PROP_PARAMS.get(name) or T.COUNT_PARAMS.get(name)
    return dict(cur or base)


def _fit_lgbm_fold(kind, params, tweedie, X, y, fit_idx, watch_idx, va_idx):
    p = dict(params)
    if kind != "binary" and tweedie is not None:
        p = dict(p, objective="tweedie", tweedie_variance_power=tweedie)
    cls = lgb.LGBMClassifier if kind == "binary" else lgb.LGBMRegressor
    m = cls(**p)
    m.fit(X.iloc[fit_idx], y[fit_idx],
          eval_set=[(X.iloc[watch_idx], y[watch_idx])],
          eval_metric=("binary_logloss" if kind == "binary"
                       else ("tweedie" if tweedie is not None else "poisson")),
          callbacks=[lgb.early_stopping(150, verbose=False)])
    if kind == "binary":
        return (m.predict_proba(X.iloc[watch_idx])[:, 1],
                m.predict_proba(X.iloc[va_idx])[:, 1])
    return (np.clip(m.predict(X.iloc[watch_idx]), 1e-6, None),
            np.clip(m.predict(X.iloc[va_idx]), 1e-6, None))


def _fit_cb_fold(kind, cb_base, override, tweedie, cat_here,
                 X, y, fit_idx, watch_idx, va_idx):
    p = dict(cb_base, **override)
    Xf, yf = X.iloc[fit_idx], y[fit_idx]
    Xw, yw = X.iloc[watch_idx], y[watch_idx]
    if kind == "binary":
        m = F.CatSafe(T.CatBoostClassifier(**p, random_seed=0,
                                           cat_features=cat_here), cat_here)
        m.fit(Xf, yf, eval_set=[(Xw, yw)])
        return (m.predict_proba(Xw)[:, 1],
                m.predict_proba(X.iloc[va_idx])[:, 1])
    if tweedie is not None:
        tw = f"Tweedie:variance_power={tweedie}"
        p = dict(p, loss_function=tw, eval_metric=tw)
    m = F.CatSafe(T.CatBoostRegressor(**p, random_seed=0,
                                      cat_features=cat_here),
                  cat_here, exponent=True)
    m.fit(Xf, yf, eval_set=[(Xw, yw)])
    return (np.clip(m.predict(Xw), 1e-6, None),
            np.clip(m.predict(X.iloc[va_idx]), 1e-6, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    want = ([h.strip() for h in args.heads.split(",") if h.strip()]
            or REP_HEADS)
    bf, sf, tg, wf = PS.load_frames()
    train_yrs, cal_yr, _ = T.suite_years(bf)
    cutoff = train_yrs[-1]
    log(f"Season <= {cutoff} | folds {args.folds} | profiles "
        f"{len(CB_PROFILES)} | panel {want}")
    out_path = Path(args.out) if args.out else ART / "cb_sweep.json"
    jobs = {j["name"]: j for j in PS.build_jobs(bf, sf, tg, wf)}
    w_l, w_c = float(T.LGBM_BAGS), float(T.CB_BAGS)

    results = {}
    for name in [h for h in PS.HEAD_ORDER if h in jobs and h in set(want)]:
        j = jobs[name]
        kind = j["kind"]
        cb_base = dict(T.CB_WIN if name == "winner" else
                       (T.CB_CLS if kind == "binary" else T.CB_POIS))
        X, y, groups = PS._prep(j["frame"], j["cols"], j["target"], cutoff,
                                args.max_rows, args.seed)
        cat_here = [c for c in X.columns if c in F.CAT_COLS]
        splits = []
        from sklearn.model_selection import GroupKFold, GroupShuffleSplit
        gkf = GroupKFold(n_splits=args.folds)
        for tr_idx, va_idx in gkf.split(X, y, groups):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.15,
                                    random_state=args.seed)
            sub_fit, sub_watch = next(gss.split(tr_idx,
                                                groups=groups[tr_idx]))
            splits.append((tr_idx[sub_fit], tr_idx[sub_watch], va_idx))
        t0 = time.time()
        lgbm_cache = [
            _fit_lgbm_fold(kind, _lgbm_params(name, kind, j["base"]),
                           j["tweedie"], X, y, *s) for s in splits]
        table = {}
        for pname, ov in CB_PROFILES.items():
            oof = np.full(len(y), np.nan)
            for (s, (lw, lv)) in zip(splits, lgbm_cache):
                fit_idx, watch_idx, va_idx = s
                cw, cv = _fit_cb_fold(kind, cb_base, ov, j["tweedie"],
                                      cat_here, X, y, *s)
                bw = (w_l * lw + w_c * cw) / (w_l + w_c)
                bv = (w_l * lv + w_c * cv) / (w_l + w_c)
                if kind == "binary":
                    iso = IsotonicRegression(out_of_bounds="clip",
                                             y_min=1e-4, y_max=1 - 1e-4)
                    iso.fit(bw, y[watch_idx])
                    oof[va_idx] = iso.predict(bv)
                else:
                    oof[va_idx] = bv
            m = {}
            if kind == "binary":
                p = np.clip(oof, 1e-6, 1 - 1e-6)
                m = {"logloss": float(log_loss(y, p)),
                     "auc": float(roc_auc_score(y, p)),
                     "ece": float(PS._ece(y, p))}
            else:
                m["mae"] = float(mean_absolute_error(y, oof))
                try:
                    m["deviance"] = float(
                        mean_tweedie_deviance(y, oof, power=j["tweedie"])
                        if j["tweedie"] is not None
                        else mean_poisson_deviance(y, oof))
                except ValueError:
                    m["deviance"] = float("nan")
            table[pname] = m
        rec = PS._recommend(kind, table)
        results[name] = {"kind": kind, "n_rows": int(len(y)),
                         "recommended": rec, "profiles": table}
        d = table["default"]
        r = table[rec]
        prim = "logloss" if kind == "binary" else "deviance"
        log(f"[{name}] DONE in {time.time()-t0:.0f}s | recommend '{rec}' "
            f"| {prim} {r.get(prim):.5f} (def {d.get(prim):.5f})")
        with open(out_path, "w") as fh:
            json.dump({"cutoff": int(cutoff), "folds": args.folds,
                       "panel": want, "profiles": CB_PROFILES,
                       "results": results}, fh, indent=2)

    # global ranking: CB params are global constants, so aggregate
    print("\n===== GLOBAL CB PROFILE RANKING =====")
    for kind, prim in (("binary", "logloss"), ("count", "deviance")):
        rows = {n: r for n, r in results.items() if r["kind"] == kind}
        if not rows:
            continue
        print(f"  -- {kind} heads ({len(rows)}) --")
        agg = {}
        for pname in CB_PROFILES:
            ds = [r["profiles"][pname].get(prim, np.nan)
                  - r["profiles"]["default"].get(prim, np.nan)
                  for r in rows.values()]
            ds = [d for d in ds if np.isfinite(d)]
            wins = sum(r["recommended"] == pname for r in rows.values())
            agg[pname] = (float(np.mean(ds)) if ds else np.nan, wins)
        for pname, (md, wins) in sorted(agg.items(), key=lambda kv: kv[1][0]):
            print(f"    {pname:14s} mean d{prim} {md:+.5f} | "
                  f"recommended by {wins} head(s)")
    print(f"\nwrote {out_path}")
    print("Wire ONE winning fragment into train.CB_CLS / CB_POIS (global); "
          "the next keep-chain's paired read verdicts it.")


if __name__ == "__main__":
    main()
