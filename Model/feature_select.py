"""Per-head feature selection from the trained bags (stability selection).

Every head trains as a features.MeanBag whose LightGBM seed members each
vote on which columns matter; the same head in the OTHER suite's artifact
(a different training era) votes independently. A column is KEPT for a
head when

    equal-family-weighted use-fraction  >=  PI    ...in BOTH suites.

Two voting statistics (--stat):

  shap (default): mean |SHAP contribution| share on the suite's OWN
      CALIBRATION year — held-out evidence, so a feature the trees used
      only to overfit the training era gets no vote, and gain's bias
      toward high-cardinality columns doesn't apply.
  gain: train-time split-gain share from the boosters — no frames
      needed, runs anywhere, but measures "was used", not "helped".

All three families vote (LGBM split-gain / SHAP, XGBoost split-gain /
SHAP, CatBoost PredictionValuesChange / ShapValues). Each member's share
is normalized to sum to 1, so the eps threshold means the same
"essentially unused" regardless of family. Families are then combined
with EQUAL weight: a column's vote is the average, across the families
present, of the fraction of THAT family's members using it — so a head's
larger LGBM bag can't outvote its 2 XGB or 2 CB members. A floor keeps
selection from starving a head: every head keeps at least MIN_KEEP
columns, topped up by mean vote share. Heads with no scorable members
stay unrestricted.

Writes artifacts/feature_keep.json  {head: [kept cols, ...]}  with
--write; always prints the per-head table. train.py applies the file as
an include-list when its SELECT_FEATURES flag is on.

Usage:
    python Model/feature_select.py                     # SHAP report
    python Model/feature_select.py --stat gain         # gain report
    python Model/feature_select.py --write             # emit feature_keep.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"

EPS = 0.0005      # vote share below this = "member didn't really use it"
PI = 0.75         # fraction of a bag's LGBM members that must use it
MIN_KEEP = 40     # no head drops below this many columns
MAX_ROWS = 5000   # cal-year rows per SHAP pass (subsampled, fixed seed);
                  # vote shares only need ~0.05% resolution — 5k is plenty


def _heads(art):
    """(head_name, model) for every trained head."""
    out = [(name, p["gbm"]) for name, p in art.get("props", {}).items()
           if "gbm" in p]
    out += [(name, h["model"])
            for name, h in art.get("count_models", {}).items()]
    if art.get("k_model") is not None:
        out.append(("k", art["k_model"]))
    if art.get("team_runs_model") is not None:
        out.append(("total", art["team_runs_model"]))
    wm = art.get("win_model") or {}
    if "gbm" in wm:
        out.append(("winner", wm["gbm"]))
    return out


def _members(model):
    return getattr(model, "models", None) or [model]


def _family(member):
    """('lgbm'|'xgb'|'cb', underlying estimator). Unwraps InfSafe/CatSafe;
    (None, member) for anything with no importance to read."""
    if getattr(member, "booster_", None) is not None:
        return "lgbm", member
    inner = getattr(member, "model", None)
    mod = type(inner).__module__ if inner is not None else ""
    if mod.startswith("xgboost"):
        return "xgb", inner
    if mod.startswith("catboost"):
        return "cb", inner
    return None, member


def _members_by_family(model):
    """{'lgbm': [...], 'xgb': [...], 'cb': [...]} of scorable members."""
    fam = {}
    for m in _members(model):
        f, _ = _family(m)
        if f is not None:
            fam.setdefault(f, []).append(m)
    return fam


def _shares_gain(member):
    """{col: share of this member's total train-time importance}. LGBM/XGB
    split gain; CatBoost PredictionValuesChange. No frames needed."""
    fam, est = _family(member)
    if fam == "lgbm":
        b = est.booster_
        names = b.feature_name()
        g = np.asarray(b.feature_importance(importance_type="gain"), float)
    elif fam == "xgb":
        b = est.get_booster()
        names = list(b.feature_names)
        score = b.get_score(importance_type="gain")     # used cols only
        g = np.asarray([score.get(n, 0.0) for n in names], float)
    elif fam == "cb":
        names = list(est.feature_names_)
        g = np.asarray(est.get_feature_importance(), float)
    else:
        return {}
    tot = g.sum() or 1.0
    return dict(zip(names, g / tot))


def _shares_shap(member, X):
    """{col: share of mean |SHAP|} on held-out rows, per family."""
    fam, est = _family(member)
    if fam == "lgbm":
        b = est.booster_
        names = b.feature_name()
        c = b.predict(X[names], pred_contrib=True)
        if isinstance(c, list):                 # multiclass safety
            c = np.abs(np.asarray(c)).sum(axis=0)
        mean_abs = np.abs(np.asarray(c))[:, :-1].mean(axis=0)   # drop bias
    elif fam == "xgb":
        import xgboost as xgb_lib
        b = est.get_booster()
        names = list(b.feature_names)
        Xc = X[names].replace([np.inf, -np.inf], np.nan)
        dm = xgb_lib.DMatrix(Xc, enable_categorical=True)
        c = np.asarray(b.predict(dm, pred_contribs=True))
        mean_abs = np.abs(c)[:, :-1].mean(axis=0)
    elif fam == "cb":
        from catboost import Pool
        names = list(est.feature_names_)
        cats = [c for c in getattr(member, "cat_cols", []) if c in names]
        Xc = X[names].copy()                    # mirror CatSafe._clean
        num = Xc.select_dtypes(include=[np.number]).columns
        Xc[num] = Xc[num].replace([np.inf, -np.inf], np.nan)
        for c in cats:
            Xc[c] = Xc[c].astype(str).fillna("missing")
        sv = np.asarray(est.get_feature_importance(
            Pool(Xc, cat_features=cats), type="ShapValues"))
        mean_abs = np.abs(sv)[:, :-1].mean(axis=0)
    else:
        return {}
    tot = mean_abs.sum() or 1.0
    return dict(zip(names, mean_abs / tot))


def _votes(model, eps, X=None, families=None):
    """Equal-family-weighted votes. Within each family the fraction of its
    members using a column is taken; the families present are averaged with
    equal weight (numerous LGBM bag can't outvote 2 XGB / 2 CB). Returns
    ({col: weighted use-fraction}, {col: weighted mean share}, member tag
    like 'L6/X2/C2'). X=None -> gain votes. families=None -> all present;
    pass e.g. {'lgbm'} to weight on LightGBM alone even if the bag carries
    other families (in the LGBM-only regime the bags are pure LGBM, so this
    is already automatic)."""
    fam_members = _members_by_family(model)
    frac_lists, mean_lists, tag = {}, {}, []
    for key, letter in (("lgbm", "L"), ("xgb", "X"), ("cb", "C")):
        if families is not None and key not in families:
            continue
        members = fam_members.get(key, [])
        if not members:
            continue
        tag.append(f"{letter}{len(members)}")
        counts, sums = {}, {}
        for m in members:
            try:
                sh = _shares_shap(m, X) if X is not None else _shares_gain(m)
            except Exception as e:
                stat = "shap" if X is not None else "gain"
                print(f"    ! {key} member {stat} scoring failed ({e}); "
                      f"gain fallback")
                sh = _shares_gain(m)
            for col, s in sh.items():
                sums[col] = sums.get(col, 0.0) + s
                if s > eps:
                    counts[col] = counts.get(col, 0) + 1
        n = len(members)
        for col in sums:
            frac_lists.setdefault(col, []).append(counts.get(col, 0) / n)
            mean_lists.setdefault(col, []).append(sums[col] / n)
    nfam = len(tag)
    if nfam == 0:
        return {}, {}, ""
    votes = {c: sum(v) / nfam for c, v in frac_lists.items()}
    means = {c: sum(v) / nfam for c, v in mean_lists.items()}
    return votes, means, "/".join(tag)


def _cal_rows(frames, art, head, rng):
    """The held-out slice one head's SHAP votes are measured on: the
    suite's calibration year, batter or starter frame, ShortGame out."""
    cal = art.get("years", {}).get("cal")
    if cal is None:
        return None
    bat_heads = set(art.get("props", {})) | {"xbk", "xtb", "xhrr"}
    frame = frames["bf"] if head in bat_heads else \
        frames["sf"] if head in ("k", "outs", "pbb", "pha", "per") else None
    if frame is None:                       # total/winner: gain votes
        return None
    f = frame[(frame["Season"] == cal) & ~frame["ShortGame"].fillna(False)]
    if len(f) > MAX_ROWS:
        f = f.iloc[rng.choice(len(f), MAX_ROWS, replace=False)]
    return f


def select(art_a, art_b, frames=None, eps=EPS, pi=PI, families=None):
    """{head: sorted kept cols} stable in BOTH artifacts (suites)."""
    heads_a, heads_b = dict(_heads(art_a)), dict(_heads(art_b))
    rng = np.random.default_rng(0)
    keep, report = {}, []
    for name in sorted(set(heads_a) & set(heads_b)):
        Xa = _cal_rows(frames, art_a, name, rng) if frames else None
        Xb = _cal_rows(frames, art_b, name, rng) if frames else None
        va, sa, na = _votes(heads_a[name], eps, Xa, families)
        vb, sb, nb = _votes(heads_b[name], eps, Xb, families)
        visible = len(set(sa) | set(sb))
        if not va or not vb:
            report.append((name, visible, None, na, nb, "-"))
            continue    # no LGBM members somewhere -> leave head unrestricted
        stable = {c for c, f in va.items() if f >= pi
                  and vb.get(c, 0.0) >= pi}
        if len(stable) < MIN_KEEP:      # top up by mean share, both suites
            pool = sorted(set(sa) | set(sb),
                          key=lambda c: -(sa.get(c, 0) + sb.get(c, 0)))
            for c in pool:
                if len(stable) >= MIN_KEEP:
                    break
                stable.add(c)
        keep[name] = sorted(stable)
        report.append((name, visible, len(stable), na, nb,
                       "shap" if Xa is not None else "gain"))
    return keep, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write artifacts/feature_keep.json")
    ap.add_argument("--stat", choices=("shap", "gain"), default="shap",
                    help="voting statistic (default shap = held-out)")
    ap.add_argument("--families", default="lgbm,xgb,cb",
                    help="comma-separated families that may vote; pass 'lgbm' "
                         "for LightGBM-only weighting. In the LGBM-only regime "
                         "the bags are pure LGBM so this is already automatic.")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--pi", type=float, default=PI)
    args = ap.parse_args()
    families = {f.strip() for f in args.families.split(",") if f.strip()}

    art_ship = joblib.load(ART / "models.joblib")      # later-era suite
    art_sel = joblib.load(ART / "models_bt.joblib")    # selection suite
    frames = None
    if args.stat == "shap":
        print("loading cached feature frames for held-out SHAP votes...")
        frames = joblib.load(ART / "frames.joblib")
        # mirror train.py: unify categorical levels across frames so the
        # boosters' stored category mappings line up at predict time
        import features as F
        from train import set_categories
        cat_levels = {}
        for c in F.CAT_COLS:
            vals = set()
            for fr in frames.values():
                if c in fr.columns:
                    vals |= set(fr[c].dropna().astype(str).unique())
            cat_levels[c] = sorted(vals)
        for fr in frames.values():
            set_categories(fr, cat_levels)
    keep, report = select(art_ship, art_sel, frames, args.eps, args.pi,
                          families)

    print(f"\n=== Stability selection ({args.stat} share > {args.eps:g} in "
          f">= {args.pi:.0%} of each family's members, equal-family weighted, "
          f"BOTH suites; floor {MIN_KEEP}) ===")
    print(f"  {'head':8s} {'visible':>8s} {'kept':>8s}  {'members':10s} stat")
    for name, n_all, n_keep, na, nb, stat in report:
        kept = "unrestr." if n_keep is None else f"{n_keep}"
        mem = na if na == nb else f"{na}|{nb}"     # suites share bag counts
        print(f"  {name:8s} {n_all:8d} {kept:>8s}  {mem:10s} {stat}")

    if args.write:
        out = ART / "feature_keep.json"
        out.write_text(json.dumps(keep, indent=1), encoding="utf-8")
        print(f"\nwrote {out} ({len(keep)} heads)")
    else:
        print("\n(report only — pass --write to emit feature_keep.json)")


if __name__ == "__main__":
    main()
