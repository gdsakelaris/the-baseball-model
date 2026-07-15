"""Per-head feature selection from the trained bags (stability selection).

Every head trains as a features.MeanBag whose members — the LGBM seed bag
plus the XGB and CatBoost members — each vote on which columns matter; the
same head in the OTHER suite's artifact (a different training era) votes
independently. A column is KEPT for a head when

    equal-family-weighted use-fraction  >=  PI    ...in BOTH suites.

HONESTY CAVEATS (2026-07-15 audit, recorded so the vote is never oversold):
  - The two suites share every training season but one, so their votes are
    highly correlated — "stable in BOTH suites" is closer to 1.2 independent
    votes than 2. The per-head Spearman of the two vote vectors is printed
    and persisted (vote_corr) so the overlap is visible, not assumed.
  - The electorate (superset train) runs a CHEAP regime — row-sampled,
    fewer bags, possibly LGBM-only — while the keep-train ships full
    fidelity (user-accepted trade, re-affirmed 07-15). Both regimes are
    read from the artifacts' meta_stamp and persisted in the report.
  - The MIN_KEEP floor no longer force-feeds failures: top-up candidates
    must clear TOPUP_MIN in both suites; a head may legitimately keep
    fewer than MIN_KEEP columns (floor_short in the report).

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
--write; always prints the per-head table and persists the full evidence
(kept/pre-top-up counts, per-suite eps, PI-grid keep-sizes, per-family
support, co-failure groups, keep-diff vs the newest feature_keep*.bak)
to artifacts/selection_report.json (report-only runs write
selection_report.report_only.json so they never clobber a chain's
evidence). train.py applies feature_keep.json as an include-list when
its SELECT_FEATURES flag is on.

Usage:
    python Model/feature_select.py                     # SHAP report
    python Model/feature_select.py --stat gain         # gain report
    python Model/feature_select.py --write             # emit feature_keep.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ART = Path(__file__).resolve().parent / "artifacts"

EPS = 0.0005      # vote share below this = "member didn't really use it"
                  # (fallback when a head's bags carry no shadow columns)
PI = 0.75         # fraction of each family's members that must use it
MIN_KEEP = 40     # top-up target — but only near-misses may fill it (see
                  # TOPUP_MIN); a head can end below this legitimately
TOPUP_MIN = 0.5   # 2026-07-15 (audit #3): floor top-ups must clear this
                  # vote in BOTH suites — columns that outright failed the
                  # stability test are never re-admitted by the floor
MAX_ROWS = 5000   # cal-year rows per SHAP pass (subsampled, fixed seed);
                  # vote shares only need ~0.05% resolution — 5k is plenty
N_SHAP_SAMPLES = 3  # date-stratified subsample repeats per SHAP vote when
                    # the cal year exceeds MAX_ROWS; member shares are
                    # AVERAGED across repeats (07-14 F1) — damps
                    # single-draw sampling noise at ~3x SHAP cost on the
                    # batter frame (the only frame over MAX_ROWS)
PI_GRID = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0)
                    # free keep-size curve in the report — evidence for
                    # any future PI adjudication, nothing reads it here

# 1F selection upgrades (2026-07-14):
# - shadow-calibrated eps (Boruta-lite): train.py plants shuffled copies of
#   superset columns (SHADOW_PREFIX); each head's eps becomes the
#   SHADOW_Q quantile of its own shadow-column shares — an empirical
#   "essentially unused" floor per head. Keep-lists are written
#   shadow-free, so nothing shadow-flavored can ship.
# - cluster co-failure REPORT: Spearman-clustered near-duplicates whose
#   members EACH narrowly miss PI while the group's combined share is
#   large — the clones-voting-each-other-out failure. Report only; flagged
#   groups are gray zones for the user, never auto-kept.
# 07-14 adopt package (pre-chain riders on 1F):
# - A1: total/winner vote on held-out SHAP too — tg/wf aren't cached, so
#   main() re-derives them from gf exactly as train.py does (canonical
#   sorts, same shadow seed => bit-identical rows and shadow values).
# - F1: repeated date-stratified SHAP subsamples, averaged (above).
# - A3/F2: pre-top-up counts + the whole report persisted to
#   artifacts/selection_report.json (a multi-hour chain's stdout is not
#   durable evidence).
# - A2b: per-family shadow-quantile DIAGNOSTIC — printed and persisted,
#   but eps stays pooled across families; per-family eps remains an
#   evidence question for a post-batch, user-adjudicated amendment.
# - keep-diff vs the newest feature_keep*.bak — the decline ledger's
#   churn evidence for the time-block-subsampling revisit trigger.
# DECLINED from the same review (recorded so they aren't re-proposed):
# hard correlation pre-cut (kills the multi-horizon c_/s_/r*/d_ variants
# and standalone-useless interaction feeders); time-block subsampling of
# the vote (two-suite era stability + held-out cal-year SHAP + the paired
# day-block CI already cover it, in stronger form); LGBM-only voting (the
# vote is per-family-normalized equal-weight use-fractions, and the
# serving XGB/CB members need their vote) — adjudicable anytime via
# --families lgbm, deliberately NOT scheduled.
SHADOW_PREFIX = "shdw_"
SHADOW_Q = 0.85          # 0.85 per user 07-15 (drop to 0.80 if the keep is timid)
COFAIL_RHO = 0.8          # |spearman| that clusters two columns
COFAIL_LO = 0.55          # narrow-miss vote band: [COFAIL_LO, PI)
COFAIL_SHARE = 0.004      # combined mean share that makes a group "large"


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


def _votes(model, eps, Xs=None, families=None):
    """Equal-family-weighted votes. Within each family the fraction of its
    members using a column is taken; the families present are averaged with
    equal weight (numerous LGBM bag can't outvote 2 XGB / 2 CB). Returns
    ({col: weighted use-fraction}, {col: weighted mean share}, member tag
    like 'L6/X2/C2', eps_used, diagnostics). Xs=None -> gain votes; else a
    list of held-out row subsets whose SHAP shares are AVERAGED per member
    (F1 repeated subsamples). families=None -> all present; pass e.g.
    {'lgbm'} to weight on LightGBM alone.

    1F: when the bags carry shadow columns, eps is re-derived as the
    SHADOW_Q quantile of every member's shadow shares (pooled — shares are
    normalized per member, so they are on one scale); the passed eps is
    only the shadow-free fallback. Shadow columns never appear in the
    returned votes/means. diagnostics = {fam_shadow_q, fam_fracs,
    n_shadow_values}: the per-family shadow floors (A2b, print/report
    only) and per-family use-fractions (F2 family support)."""
    fam_members = _members_by_family(model)
    member_shares = []          # (family key, {col: share})
    tag = []
    for key, letter in (("lgbm", "L"), ("xgb", "X"), ("cb", "C")):
        if families is not None and key not in families:
            continue
        members = fam_members.get(key, [])
        if not members:
            continue
        tag.append(f"{letter}{len(members)}")
        for m in members:
            try:
                if Xs is not None:
                    reps = [_shares_shap(m, X) for X in Xs]
                    cols = set().union(*reps)
                    sh = {c: sum(r.get(c, 0.0) for r in reps) / len(reps)
                          for c in cols}
                else:
                    sh = _shares_gain(m)
            except Exception as e:
                stat = "shap" if Xs is not None else "gain"
                print(f"    ! {key} member {stat} scoring failed ({e}); "
                      f"gain fallback")
                sh = _shares_gain(m)
            member_shares.append((key, sh))
    if not tag:
        return {}, {}, "", eps, {}
    shadow_pool = [s for _, sh in member_shares for c, s in sh.items()
                   if c.startswith(SHADOW_PREFIX)]
    if shadow_pool:
        eps = max(float(np.quantile(shadow_pool, SHADOW_Q)), 1e-12)
    # per-family shadow floor (A2b): DIAGNOSTIC only — eps stays pooled.
    # Floors within ~2x of each other = per-family eps is empirically
    # dead; ~10x divergence = evidence for a post-batch, user-adjudicated
    # amendment.
    fam_shadow = {}
    for key, sh in member_shares:
        fam_shadow.setdefault(key, []).extend(
            s for c, s in sh.items() if c.startswith(SHADOW_PREFIX))
    fam_shadow_q = {k: float(np.quantile(v, SHADOW_Q))
                    for k, v in fam_shadow.items() if v}
    fam_fracs, fam_means = {}, {}
    for key, _letter in (("lgbm", "L"), ("xgb", "X"), ("cb", "C")):
        shares = [sh for k, sh in member_shares if k == key]
        if not shares:
            continue
        counts, sums = {}, {}
        for sh in shares:
            for col, s in sh.items():
                if col.startswith(SHADOW_PREFIX):
                    continue
                sums[col] = sums.get(col, 0.0) + s
                if s > eps:
                    counts[col] = counts.get(col, 0) + 1
        n = len(shares)
        fam_fracs[key] = {c: counts.get(c, 0) / n for c in sums}
        fam_means[key] = {c: sums[c] / n for c in sums}
    nfam = len(fam_fracs)
    allcols = set().union(*fam_fracs.values())
    votes = {c: sum(f.get(c, 0.0) for f in fam_fracs.values()) / nfam
             for c in allcols}
    means = {c: sum(m.get(c, 0.0) for m in fam_means.values()) / nfam
             for c in allcols}
    diag = {"fam_shadow_q": fam_shadow_q, "fam_fracs": fam_fracs,
            "n_shadow_values": len(shadow_pool)}
    return votes, means, "/".join(tag), eps, diag


def _cal_rows(frames, art, head, rng):
    """Held-out slices one head's SHAP votes are measured on: the suite's
    calibration year of the head's own frame. ShortGame rows drop wherever
    the frame carries the flag — tg deliberately doesn't (the runs model
    trains on all games, mirroring train.py). Count heads carry their
    frame key in the artifact, so new heads route automatically.
    Returns (frame key, [row subsets]): several date-stratified subsamples
    when the year exceeds MAX_ROWS (F1, shares averaged in _votes), else
    the whole year once. None -> gain votes."""
    cal = art.get("years", {}).get("cal")
    if cal is None:
        return None
    cm = art.get("count_models", {}).get(head, {})
    if head in art.get("props", {}) or cm.get("frame") == "bat":
        key = "bf"
    elif head == "k" or cm.get("frame") == "starts":
        key = "sf"
    elif head == "total":                   # A1: game heads vote SHAP too
        key = "tg"
    elif head == "winner":
        key = "wf"
    else:
        return None
    frame = frames.get(key)
    if frame is None:                       # cache predates game frames
        return None
    f = frame[frame["Season"] == cal]
    if "ShortGame" in f.columns:            # tg lacks the flag — guard
        f = f[~f["ShortGame"].fillna(False)]
    if len(f) <= MAX_ROWS:
        return key, [f]
    # F1: date-ordered equal-count strata, one uniform draw per stratum —
    # every subsample covers the whole season instead of clumping
    f = f.sort_values("Date")
    bins = np.linspace(0, len(f), MAX_ROWS + 1).astype(int)
    lo, hi = bins[:-1], bins[1:]
    ok = hi > lo
    return key, [f.iloc[np.sort(rng.integers(lo[ok], hi[ok]))]
                 for _ in range(N_SHAP_SAMPLES)]


def _spearman_clusters(X, cols):
    """Greedy |spearman| >= COFAIL_RHO clusters (union-find) over numeric
    columns — the near-duplicate map for the co-failure report. Diagnostic
    only; computed once per frame on the (already subsampled) SHAP rows."""
    import pandas as pd
    num = [c for c in cols if c in X.columns
           and pd.api.types.is_numeric_dtype(X[c])]
    if len(num) < 2:
        return {}
    R = X[num].rank().to_numpy(dtype="float64")
    R -= np.nanmean(R, axis=0)
    R = np.nan_to_num(R)
    norm = np.sqrt((R ** 2).sum(axis=0))
    norm[norm == 0] = 1.0
    C = (R / norm).T @ (R / norm)
    parent = list(range(len(num)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ii, jj = np.where(np.triu(np.abs(C) >= COFAIL_RHO, k=1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = {}
    for i, c in enumerate(num):
        groups.setdefault(find(i), []).append(c)
    return {c: tuple(sorted(g)) for g in groups.values() if len(g) > 1
            for c in g}


def _cofail(name, va, vb, sa, sb, clusters, pi):
    """Flagged clone groups for one head: >=2 members each narrowly missing
    PI (vote in [COFAIL_LO, pi) in BOTH suites) while the group's combined
    mean share is large. Returns [(group, members, share)]."""
    out = []
    seen = set()
    for c, grp in clusters.items():
        if grp in seen:
            continue
        seen.add(grp)
        near = [m for m in grp
                if COFAIL_LO <= min(va.get(m, 0), vb.get(m, 0)) < pi]
        if len(near) < 2:
            continue
        share = sum((sa.get(m, 0) + sb.get(m, 0)) / 2 for m in grp)
        if share >= COFAIL_SHARE:
            out.append((name, near, share))
    return out


def select(art_a, art_b, frames=None, eps=EPS, pi=PI, families=None):
    """({head: sorted kept cols} stable in BOTH suites, report rows
    (one dict per head), co-failure rows)."""
    heads_a, heads_b = dict(_heads(art_a)), dict(_heads(art_b))
    rng = np.random.default_rng(0)
    keep, report, cofail = {}, [], []
    cluster_cache = {}
    for name in sorted(set(heads_a) & set(heads_b)):
        ra = _cal_rows(frames, art_a, name, rng) if frames else None
        rb = _cal_rows(frames, art_b, name, rng) if frames else None
        fkey, Xa = ra if ra else (None, None)
        Xb = rb[1] if rb else None
        va, sa, na, ea, da = _votes(heads_a[name], eps, Xa, families)
        vb, sb, nb, eb, db = _votes(heads_b[name], eps, Xb, families)
        row = {"head": name, "visible": len(set(sa) | set(sb)),
               "kept": None, "pre_topup": None, "members": (na, nb),
               "stat": "shap" if Xa is not None else "gain",
               "eps": (ea, eb),
               "fam_shadow_q": (da.get("fam_shadow_q", {}),
                                db.get("fam_shadow_q", {}))}
        if not va or not vb:
            report.append(row)
            continue    # no scorable members somewhere -> head unrestricted
        stable = {c for c, f in va.items() if f >= pi
                  and vb.get(c, 0.0) >= pi}
        # A3: pre-top-up count — the evidence for any future MIN_KEEP
        # floor call (0 = would flip unrestricted without the floor;
        # 0 < n < MIN_KEEP = the floor is doing real work)
        row["pre_topup"] = len(stable)
        if len(stable) < MIN_KEEP:
            # 07-15 (audit #3): top up ONLY from near-misses — columns with
            # vote >= TOPUP_MIN in BOTH suites, by mean share. Columns that
            # failed outright stay out; the head may end below MIN_KEEP.
            pool = sorted((c for c in set(sa) | set(sb)
                           if min(va.get(c, 0.0), vb.get(c, 0.0)) >= TOPUP_MIN
                           and c not in stable),
                          key=lambda c: -(sa.get(c, 0) + sb.get(c, 0)))
            for c in pool:
                if len(stable) >= MIN_KEEP:
                    break
                stable.add(c)
        row["floor_short"] = max(0, MIN_KEEP - len(stable))
        keep[name] = sorted(stable)
        row["kept"] = len(stable)
        # audit #3 diagnostic: how correlated the two suites' votes really
        # are (they share all training seasons but one — near 1.0 means the
        # both-suite requirement added little)
        union = sorted(set(va) | set(vb))
        vc = (float(pd.Series(
            [va.get(c, 0.0) for c in union]).corr(
            pd.Series([vb.get(c, 0.0) for c in union]), method="spearman"))
            if len(union) >= 3 else float("nan"))
        row["vote_corr"] = round(vc, 4) if np.isfinite(vc) else None
        # F2 evidence riders (nothing here changes the vote): the free
        # PI-grid keep-size curve and each family's solo both-suite pass
        # count — future PI / per-family adjudications read these
        row["pi_grid"] = {
            f"{g:g}": sum(1 for c in set(va) | set(vb)
                          if va.get(c, 0.0) >= g and vb.get(c, 0.0) >= g)
            for g in PI_GRID}
        fams = set(da.get("fam_fracs", {})) & set(db.get("fam_fracs", {}))
        row["family_support"] = {
            fam: sum(1 for c, fr in da["fam_fracs"][fam].items()
                     if fr >= pi and db["fam_fracs"][fam].get(c, 0.0) >= pi)
            for fam in sorted(fams)}
        report.append(row)
        # co-failure report (1F): cluster the head's frame once, then flag
        # near-miss clone groups. SHAP-voted heads only (needs frames);
        # fkey comes from _cal_rows routing — the old bf/sf column sniff
        # would have mis-keyed the game heads' cluster maps
        if Xa is not None:
            if fkey not in cluster_cache:
                cluster_cache[fkey] = _spearman_clusters(
                    Xa[0], sorted(set(sa) | set(sb)))
            cofail += _cofail(name, va, vb, sa, sb, cluster_cache[fkey], pi)
    return keep, report, cofail


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

    # electorate = the SUPERSET artifacts (2026-07-15 layout: superset trains
    # write models_superset*.joblib and never touch the serving files);
    # legacy names are the fallback for pre-07-15 artifacts.
    sup_ship = ART / "models_superset.joblib"
    sup_sel = ART / "models_superset_bt.joblib"
    if sup_ship.exists() and sup_sel.exists():
        print("electorate: models_superset*.joblib")
        art_ship = joblib.load(sup_ship)
        art_sel = joblib.load(sup_sel)
    else:
        print("electorate: legacy models.joblib / models_bt.joblib "
              "(no models_superset*.joblib yet — pre-07-15 layout)")
        art_ship = joblib.load(ART / "models.joblib")      # later-era suite
        art_sel = joblib.load(ART / "models_bt.joblib")    # selection suite
    # audit #3: surface the electorate-vs-ship regime gap loudly (the vote
    # comes from cheaper models than the one that ships — user-accepted)
    stamps = {"ship_suite": art_ship.get("meta_stamp"),
              "sel_suite": art_sel.get("meta_stamp")}
    st = stamps["ship_suite"] or {}
    if st.get("role") == "superset" and (st.get("superset_sample")
                                         or st.get("lgbm_only_temp")):
        print(f"NOTE (audit #3): electorate regime = {st.get('bags')} bags, "
              f"row cap {st.get('superset_sample')}, lgbm_only="
              f"{st.get('lgbm_only_temp')} — cheaper than the keep-train "
              f"that ships; votes may not perfectly transfer.")
    frames = None
    if args.stat == "shap":
        print("loading cached feature frames for held-out SHAP votes...")
        frames = joblib.load(ART / "frames.joblib")
        # mirror train.py: unify categorical levels across frames so the
        # boosters' stored category mappings line up at predict time
        import features as F
        from train import set_categories, add_shadow_cols, SHADOW_N
        cat_levels = {}
        for c in F.CAT_COLS:
            vals = set()
            for fr in frames.values():
                if c in fr.columns:
                    vals |= set(fr[c].dropna().astype(str).unique())
            cat_levels[c] = sorted(vals)
        for fr in frames.values():
            set_categories(fr, cat_levels)
        # A1: tg/wf aren't cached — re-derive from gf EXACTLY as train.py
        # does (same dropna/sort/reset, same shadow seed), so the held-out
        # rows and shadow values match training bit-for-bit. wf filters
        # ShortGame at derivation because training does; tg trains on all
        # games and never gets the filter.
        gf = frames.get("gf")
        if gf is not None:
            print("deriving tg/wf game-head frames from gf (A1)...")
            tg = F.build_team_game_frame(gf.dropna(subset=["total_runs"]))
            tg = tg.dropna(subset=["y_runs"])
            tg = tg.sort_values(["GamePk", "Home"]).reset_index(drop=True)
            set_categories(tg, cat_levels)
            add_shadow_cols(tg, F.team_game_feature_cols(), SHADOW_N["tg"])
            wf = gf[~gf["ShortGame"].fillna(False)].dropna(
                subset=["y_home_win"])
            wf = wf.sort_values("GamePk").reset_index(drop=True).copy()
            add_shadow_cols(wf, F.win_feature_cols(), SHADOW_N["wf"])
            frames["tg"], frames["wf"] = tg, wf
    keep, report, cofail = select(art_ship, art_sel, frames, args.eps,
                                  args.pi, families)

    print(f"\n=== Stability selection ({args.stat} share > eps in "
          f">= {args.pi:.0%} of each family's members, equal-family weighted, "
          f"BOTH suites; floor {MIN_KEEP}; eps = p{SHADOW_Q * 100:.0f} of "
          f"shadow shares per head/suite, fallback {args.eps:g}) ===")
    print(f"  {'head':8s} {'visible':>8s} {'kept':>8s} {'pre-top':>8s} "
          f"{'short':>6s} {'vcorr':>6s}  "
          f"{'members':10s} {'stat':5s} {'eps A':>9s} {'eps B':>9s}")
    for r in report:
        kept = "unrestr." if r["kept"] is None else f"{r['kept']}"
        pre = "-" if r["pre_topup"] is None else f"{r['pre_topup']}"
        sh = r.get("floor_short")
        vc = r.get("vote_corr")
        na, nb = r["members"]
        mem = na if na == nb else f"{na}|{nb}"     # suites share bag counts
        ea, eb = r["eps"]
        print(f"  {r['head']:8s} {r['visible']:8d} {kept:>8s} {pre:>8s} "
              f"{('-' if sh is None else str(sh)):>6s} "
              f"{('-' if vc is None or not np.isfinite(vc) else f'{vc:.2f}'):>6s}  "
              f"{mem:10s} {r['stat']:5s} {ea:9.2e} {eb:9.2e}")
    print("  short = columns under the MIN_KEEP target after the restricted "
          "top-up (legitimate);\n  vcorr = Spearman corr of the two suites' "
          "vote vectors — near 1.0 means the both-suite\n  requirement adds "
          "little independence (the suites share all training seasons but "
          "one).")

    # A2b: per-family shadow floors — print-only; the vote uses pooled eps
    fam_rows = [r for r in report if any(r["fam_shadow_q"])]
    if fam_rows:
        print(f"\n=== Shadow noise floor by family (p{SHADOW_Q * 100:.0f} "
              f"of shadow shares; DIAGNOSTIC — eps stays pooled) ===")
        for r in fam_rows:
            parts = []
            for side, qs in zip(("A", "B"), r["fam_shadow_q"]):
                if qs:
                    parts.append(side + ": " + " ".join(
                        f"{k[0].upper()}{v:.1e}"
                        for k, v in sorted(qs.items())))
            print(f"  {r['head']:8s} {'  |  '.join(parts)}")

    if cofail:
        print("\n=== Co-failure report (1F): near-duplicate groups whose "
              "members EACH narrowly miss PI ===")
        print("  (gray zones — user adjudicates; nothing is auto-kept)")
        for name, near, share in cofail:
            print(f"  {name:8s} share {share:.4f}  {', '.join(near)}")
    else:
        print("\n(co-failure report: no flagged clone groups)")

    # keep-diff vs the newest pre-chain backup: per-head adds/drops from
    # the regen — the decline ledger's churn evidence for the time-block
    # subsampling revisit trigger. Counts print here; full column lists
    # persist in the report JSON.
    diff = None
    baks = sorted(ART.glob("feature_keep*.bak"),
                  key=lambda p: p.stat().st_mtime)
    if baks:
        old = json.loads(baks[-1].read_text(encoding="utf-8"))
        diff = {"bak": baks[-1].name,
                "new_heads": sorted(set(keep) - set(old)),
                "gone_heads": sorted(set(old) - set(keep)),
                "heads": {}}
        print(f"\n=== Keep-list diff vs {baks[-1].name} ===")
        unchanged = []
        for name in sorted(set(keep) & set(old)):
            added = sorted(set(keep[name]) - set(old[name]))
            dropped = sorted(set(old[name]) - set(keep[name]))
            diff["heads"][name] = {
                "n_old": len(old[name]), "n_new": len(keep[name]),
                "added": added, "dropped": dropped}
            if added or dropped:
                print(f"  {name:8s} {len(old[name]):3d} -> "
                      f"{len(keep[name]):3d}  (+{len(added)} / "
                      f"-{len(dropped)})")
            else:
                unchanged.append(name)
        if unchanged:
            print(f"  unchanged: {', '.join(unchanged)}")
        if diff["new_heads"]:
            print(f"  new heads (not in .bak): "
                  f"{', '.join(diff['new_heads'])}")
        if diff["gone_heads"]:
            print(f"  heads only in .bak: {', '.join(diff['gone_heads'])}")
    else:
        print("\n(keep-diff: no feature_keep*.bak to diff against)")

    # F2: persist the evidence — a multi-hour chain's stdout evaporates;
    # Phase-5 reads this file back. Report-only runs write a separate
    # file so they never clobber a --write chain's evidence.
    rep = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "stat": args.stat, "pi": args.pi, "eps_fallback": args.eps,
        "shadow_q": SHADOW_Q, "min_keep": MIN_KEEP, "max_rows": MAX_ROWS,
        "n_shap_samples": N_SHAP_SAMPLES,
        "families": sorted(families), "write": bool(args.write),
        "electorate_regime": stamps,   # audit #3: cheap electorate on record
        "topup_min": TOPUP_MIN,
        "heads": {r["head"]: {
            "visible": r["visible"], "kept": r["kept"],
            "pre_topup": r["pre_topup"],
            "floor_short": r.get("floor_short"),
            "vote_corr": r.get("vote_corr"),
            "members": {"a": r["members"][0], "b": r["members"][1]},
            "stat": r["stat"],
            "eps": {"a": r["eps"][0], "b": r["eps"][1]},
            "shadow_q_by_family": {"a": r["fam_shadow_q"][0],
                                   "b": r["fam_shadow_q"][1]},
            "pi_grid": r.get("pi_grid"),
            "family_support": r.get("family_support"),
        } for r in report},
        "cofail": [{"head": n, "share": round(s, 6), "members": list(m)}
                   for n, m, s in cofail],
        "keep_diff": diff,
    }
    rep_path = ART / ("selection_report.json" if args.write
                      else "selection_report.report_only.json")
    rep_path.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\nwrote {rep_path}")

    if args.write:
        out = ART / "feature_keep.json"
        out.write_text(json.dumps(keep, indent=1), encoding="utf-8")
        print(f"wrote {out} ({len(keep)} heads)")
    else:
        print("(report only — pass --write to emit feature_keep.json)")


if __name__ == "__main__":
    main()
