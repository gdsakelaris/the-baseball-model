"""Train the MLB prediction models.

Models (LightGBM seed bags; XGBoost + CatBoost family members join when
LGBM_ONLY_TEMP is False — the shipped 3-family ensemble):
  batter props (binary, Platt-calibrated): the 24 heads in PROPS below
    (hit/run/RBI/TB/K/H+R+RBI thresholds, single/double/triple, sb, bb)
  k     starter strikeouts in the game              Poisson regression
  count heads (mean + cal-year per-line calibrators): batter xSO/xHRR/xTB/
    xH/xR/xRBI/xBB and starter outs/pbb/pha/per (P(over) lines; per also
    drives a derived expected ERA in predict.py)
  runs  game total runs (per-team)                  Poisson regression
  winner  dedicated home-win classifier, blended with the runs model

Honest evaluation protocol (no leakage):
  train on every season but the newest two (boosters early-stop on a
  held-out ~10% GamePk slice of the TRAINING rows — see _es_split)  ->
  blend + calibrate on the next-to-newest  ->  test on the newest
  (e.g. 2015-2024 / 2025 / 2026).
The split is DERIVED from the seasons present in the data (suite_years), so
the annual rollover needs no code edit: once a new season accrues real
games it becomes the holdout, the old holdout graduates to calibration, and
one more season enters training. The shipped artifacts are exactly the
models the holdout numbers describe.

The holdout season is CONFIRM-ONLY. Iterating on features/params against
its numbers quietly overfits it, so model selection runs on a separate
suite shifted one season back (e.g. train<=2023, cal 2024, test 2025) that
the default run also refreshes:

    python Model/train.py --rebuild --select   # selection suite only (fast loop)
    python Model/evaluate_deep.py              # full workup on the selection
    ...iterate until satisfied, then...        #   test year (default)
    python Model/train.py                      # BOTH suites (frames cached)
    python Model/evaluate_deep.py --confirm    # ONE confirming look at the holdout

Usage:
    python Model/train.py [--rebuild] [--select]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb_lib
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(exist_ok=True)

LGB_CLS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=127,
               min_child_samples=80, subsample=0.8, subsample_freq=1,
               colsample_bytree=0.8, reg_lambda=3.0, max_bin=255,
               objective="binary", verbose=-1)
LGB_POIS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                min_child_samples=60, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=3.0, objective="poisson",
                verbose=-1)
# The winner model trains on ~10k games, not ~190k batter-games: batter-scale
# capacity overfit instantly (v1 early-stopped at 26 trees, test AUC 0.52).
# Small trees + heavy regularization let boosting actually accumulate signal.
# num_leaves 15->7 / mcs 300->200 = the 2026-07-15 LGBM-only sweep's "leaves7"
# (winner's largest single-bag CV win: logloss -0.0016, AUC +0.0023 vs the
# prior reg_up base). CAVEAT: a single-bag pick for a 5-bag ship on a
# ~10k-row head — the one most prone to over-regularization — so confirm on
# the keep-train via evaluate_deep --paired before trusting the deeper cut.
LGB_WIN = dict(n_estimators=3000, learning_rate=0.02, num_leaves=7,
               min_child_samples=200, subsample=0.9, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=6.0, objective="binary",
               verbose=-1)

# Recency sample-weighting (2026-07-15, tier-1 mechanics batch): every
# booster and LR fit weights row i by RECENCY_DECAY ** (cal_yr - Season_i),
# so 2015 stops counting as much as 2024 across a decade of era shifts
# (juiced ball, pitch clock, shift ban, bigger bases). 1.0 = OFF (all-ones
# weights are skipped entirely — bit-identical to the incumbent). The value
# is swept on the SELECTION suite by Model/decay_sweep.py (--decay overrides
# per run); bake the sweep winner here before the ship chain. Blend weights,
# calibrators, and dispersions still fit UNWEIGHTED on the cal year — decay
# shapes what the boosters learn, not how they are priced.
RECENCY_DECAY = 1.0


def _recency_w(frame, cal_yr):
    """Per-row training weights RECENCY_DECAY**(cal_yr - Season); None when
    the decay is off so every fit call stays bit-identical to the incumbent."""
    if RECENCY_DECAY >= 1.0:
        return None
    yrs = np.clip(cal_yr - frame["Season"].to_numpy(dtype=float), 0, None)
    return RECENCY_DECAY ** yrs


# GBM-vs-LR blend space (2026-07-15, tier-1 mechanics batch): the 21-point
# grid over w now combines the two members' LOG-ODDS (see features.blend).
# With the Platt/beta calibrator fit downstream on the blended score, the
# logit-space grid is exactly the full 2-member logistic stack (free
# per-member coefficients), which probability-space blending cannot express.
# Artifacts carry blend_space so predict.py serves the same arithmetic;
# absent key = old artifact = probability space. "prob" restores incumbent.
BLEND_SPACE = "logit"

# Monotonic constraints (HR only): physics/domain says HR probability can
# only rise with these — exit velo, barrel rate, own HR rate, HR-friendly
# park, heat, altitude. Constraining the GBM is pure regularization: it
# cannot create signal, only stop trees fitting noise wiggles in thin
# regions. "advanced" is the least accuracy-costly enforcement.
# BENCHED (2026-07-07, cal-2024 fit previewed on 2025): within-noise on
# everything — AUC +0.0003, logloss -0.0008 (edge +0.0008 < .001 band),
# but top10 -0.0087 (the pick metric got slightly worse). Neutral change,
# so the simpler unconstrained model ships. The wiring stays: repopulate
# HR_MONOTONE below into MONOTONE to re-enable (e.g. if a future serving
# robustness guarantee on odd GUI inputs is wanted — monotonicity is
# defensible even at flat metrics, it just didn't earn its way in on
# accuracy). Fill the dict to re-enable (Experiment 4 of the program).
HR_MONOTONE = {
    "bip_ev": 1, "bipd_ev": 1, "bip_brl": 1, "bipd_brl": 1, "bip_hh": 1,
    "hrq_ev_avg": 1, "hrq_dist_avg": 1, "hrq_dist_max": 1,
    "c_hr_pa_sh": 1, "s_hr_pa_sh": 1, "d_hr_pa_sh": 1,
    "park_hr_pg": 1, "Temp": 1, "Elevation_ft": 1,
}
# re-accept test 2026-07-09 (queue Tier A1, keep-leaning bar): flat on
# accuracy when benched 07-07, but monotonicity is a serving-robustness
# guarantee on odd GUI inputs; keep unless the paired read shows harm.
MONOTONE = {"hr": HR_MONOTONE}

# LightGBM seed bagging: LGBM_BAGS GBMs differing only in random_state,
# predictions averaged (features.MeanBag) before the LR blend + isotonic.
# Variance reduction, not new signal — it also SHRINKS retrain jitter,
# making every future Section-11 paired read sharper. Bag 0 keeps
# LightGBM's default seed (the pre-bagging incumbent exactly); bags
# 1..N-1 reseed.
# History: added 2026-07-08 at 5 for a hand-picked "weak/target" prop set
# only (the old PROP_BAGS/COUNT_BAGS dicts); made UNIFORM across EVERY head
# at 6 on 2026-07-14 (user). The targeted scope predated the full-board
# family bags (XGB/CB went everywhere 07-10) and left the selection vote
# coarse for the 27 unbagged heads (a lone LGBM member votes a hard 0/1 in
# feature_select). Every head carries a multi-member LGBM bag so the feature
# vote is granular (0/.17/.33/.../1). 2026-07-15 (user): SPLIT by pass — the
# shipped KEEP-train keeps a 5-member bag (LGBM_BAGS below); the SUPERSET
# train (no keep-list; feeds only feature_select) drops to 3 bags + a row
# sample — see the _KEEP_TRAIN block just after FEATURE_KEEP. Set to 1 to
# revert to a single member.
LGBM_BAGS = 6   # shipped keep-train (6 bags, user 07-15); superset overridden to 3 below

# Family bagging (2026-07-10 experiment, FULL BOARD per user): XGBoost
# members appended to every GBM head's bag — binaries, count heads (k
# included), team runs, winner. A different split policy and regularization
# path decorrelates members more than reseeding LightGBM can. Existing LGBM
# members are untouched (bit-identical to the incumbents); XGB_BAGS members
# join each mean. Params mirror the matching LGB_* block: hist + lossguide
# + max_leaves = LightGBM's leaf-wise growth; min_child_weight ~
# min_child_samples x the objective's per-row hessian scale (~0.2 binary,
# ~mu Poisson). XGBoost rejects the ±inf the frames carry (LightGBM
# tolerates them), hence features.InfSafe. Set XGB_BAGS = 0 to revert to
# pure-LightGBM everywhere.
#
# 2026-07-14 (user): LGBM_ONLY_TEMP drops the XGB+CB families so the
# FINISH_PLAN build batch can iterate on retrains cheaply (6 LGBM bags per
# head only). Downstream adapts on its own: feature_select votes only the
# families present, predict's _force_xgb_cpu no-ops, param_sweep reads
# these constants.
# 2026-07-15 (user): NO LONGER TEMP — the shipped ensemble stays LGBM-only
# (6-bag + LR blend) for the foreseeable future; the FINAL chain ships with
# this flag True. The XGB/CB wiring stays intact (weights included) so
# flipping False deliberately brings the 3-family ensemble back whole.
LGBM_ONLY_TEMP = True
XGB_BAGS = 0 if LGBM_ONLY_TEMP else 2  # 2 = SHIPPED 07-13: 3-family ensemble on the 07-12 features + 3-family keep-list
XGB_CLS = dict(n_estimators=3000, learning_rate=0.03, tree_method="hist",
               grow_policy="lossguide", max_leaves=127, max_depth=0,
               min_child_weight=16, subsample=0.8, colsample_bytree=0.8,
               reg_lambda=3.0, max_bin=255, objective="binary:logistic",
               eval_metric="logloss", early_stopping_rounds=150,
               enable_categorical=True, n_jobs=-1, verbosity=0,
               device="cuda")
XGB_POIS = dict(n_estimators=2000, learning_rate=0.03, tree_method="hist",
                grow_policy="lossguide", max_leaves=63, max_depth=0,
                min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=3.0, max_bin=255, objective="count:poisson",
                eval_metric="poisson-nloglik", early_stopping_rounds=150,
                enable_categorical=True, n_jobs=-1, verbosity=0,
                device="cuda")
XGB_WIN = dict(n_estimators=3000, learning_rate=0.02, tree_method="hist",
               grow_policy="lossguide", max_leaves=15, max_depth=0,
               min_child_weight=30, subsample=0.9, colsample_bytree=0.7,
               reg_lambda=10.0, max_bin=255, objective="binary:logistic",
               eval_metric="logloss", early_stopping_rounds=150,
               enable_categorical=True, n_jobs=-1, verbosity=0,
               device="cuda")
# CatBoost members (iteration 2 of the same experiment): symmetric
# depth-wise trees + ordered target statistics for categoricals — a third
# split policy in the bag. Depth ~8 approximates the LGBM/XGB capacity;
# features.CatSafe handles its quirks (no NaN cats, no inf, Poisson/Tweedie
# predict needs Exponent). Set CB_BAGS = 0 to drop the family. Wired and
# smoke-tested 2026-07-10 but held at 0 for the XGB confirm (CatBoost fits
# cost 2-4x XGB — a three-family full train couldn't land before the 06:00
# job); flip to 2 as the next selection-year iteration.
CB_BAGS = 0 if LGBM_ONLY_TEMP else 2  # 2 = SHIPPED 07-13: 3-family ensemble on the 07-12 features + 3-family keep-list
CB_CLS = dict(iterations=3000, learning_rate=0.03, depth=8,
              l2_leaf_reg=3.0, loss_function="Logloss",
              eval_metric="Logloss", early_stopping_rounds=150,
              verbose=0, allow_writing_files=False,
              task_type="GPU", devices="0")
CB_POIS = dict(iterations=2000, learning_rate=0.03, depth=7,
               l2_leaf_reg=3.0, loss_function="Poisson",
               eval_metric="Poisson", early_stopping_rounds=150,
               verbose=0, allow_writing_files=False,
              task_type="GPU", devices="0")
CB_WIN = dict(iterations=3000, learning_rate=0.02, depth=4,
              l2_leaf_reg=10.0, loss_function="Logloss",
              eval_metric="Logloss", early_stopping_rounds=150,
              verbose=0, allow_writing_files=False,
              task_type="GPU", devices="0")

# Per-prop LightGBM overrides — RE-SWEPT 2026-07-15 with param_sweep
# (LGBM-only, single-bag CV, Season<=2024 with 2025 untouched) on the
# fresh pi=0.6 keep-lists. The leaner keep-lists shifted every binary
# head's optimum toward more regularization: 17/21 binary props took a
# reg_med / reg_med2 / reg_heavy / lr_slow profile (all -0.0005..-0.0016
# CV logloss vs their raw base); every count head + hr/tb4/triple/run2
# kept default (default won their CV). winner's move (leaves7) lives in
# LGB_WIN above. Heads not listed = LGB_CLS/LGB_POIS.
# CAVEAT: single-bag CV over-prefers regularization vs the 5-bag ship —
# these are recommendations, confirmed as a package by evaluate_deep
# --paired on the actual keep-train before they are trusted.
_REG_MED  = dict(num_leaves=63, min_child_samples=160)
_REG_MED2 = dict(num_leaves=63, min_child_samples=300,
                 colsample_bytree=0.7, reg_lambda=6.0)
_REG_HEAVY = dict(num_leaves=31, min_child_samples=300,
                  colsample_bytree=0.7, reg_lambda=6.0)
PROP_PARAMS = {
    "run":    dict(LGB_CLS, **_REG_MED2),
    "rbi":    dict(LGB_CLS, **_REG_MED),
    "hit":    dict(LGB_CLS, **_REG_MED),
    "hits2":  dict(LGB_CLS, **_REG_HEAVY),
    "tb2":    dict(LGB_CLS, **_REG_HEAVY),
    "single": dict(LGB_CLS, learning_rate=0.02),
    "double": dict(LGB_CLS, **_REG_MED2),
    "bb":     dict(LGB_CLS, **_REG_MED),
    "sb":     dict(LGB_CLS, **_REG_MED2),
    "bk":     dict(LGB_CLS, **_REG_MED),
    "bk2":    dict(LGB_CLS, **_REG_MED),
    "hrr2":   dict(LGB_CLS, **_REG_MED2),
    "hrr3":   dict(LGB_CLS, **_REG_HEAVY),
    "bk3":    dict(LGB_CLS, **_REG_MED2),
    "tb3":    dict(LGB_CLS, **_REG_MED),
    "hrr4":   dict(LGB_CLS, **_REG_MED2),
    "rbi2":   dict(LGB_CLS, **_REG_HEAVY),
}

# Per-prop feature routing was REMOVED 2026-07-15 (audit fix #11). The
# hand-curated PROP_EXCLUDE tables had been dead code since the 2026-07-10
# probe emptied this dict and made automated stability selection
# (feature_keep.json) the sole decider of what each head trains on. Git
# history (pre-338e91a) keeps the tables and their bench/re-accept notes.
PROP_EXCLUDE = {}

# Stability-selection include-lists (feature_select.py --write): with
# SELECT_FEATURES on, EVERY head — batter props, count heads, k, total,
# winner — trains only on the columns its bags voted stable (applied AFTER
# PROP_EXCLUDE). Per user 2026-07-10, automated selection is the SOLE
# decider of which features each head keeps; no manual carve-outs.
# Inert by default: flag off, or no feature_keep.json -> no restriction.
# 2026-07-10 SHIP: ON. Keep-list regenerated from the final both-suite
# superset bags (LGBM+XGB+CB, feature_select --write, all-3-family equal
# weight) after adding BvP + park-handed-HR (#8). Every head trains on the
# columns its bags voted stable.
SELECT_FEATURES = True


def _feature_keep():
    p = ART / "feature_keep.json"
    if not (SELECT_FEATURES and p.exists()):
        return {}
    import json
    keep = {k: set(v) for k, v in json.loads(p.read_text()).items()}
    # Since the 2026-07-15 top-up restriction (audit #3), heads may sit
    # legitimately below feature_select.MIN_KEEP; only an EMPTY entry is a
    # corrupt file. Sub-floor counts print informationally.
    from feature_select import MIN_KEEP
    empty = [k for k, v in keep.items() if not v]
    if empty:
        print(f"WARNING: feature_keep.json has EMPTY entries for {empty} — "
              f"corrupt file? Those heads would train on nothing.", flush=True)
    small = {k: len(v) for k, v in keep.items() if 0 < len(v) < MIN_KEEP}
    if small:
        print(f"note: keep-list heads below the old {MIN_KEEP}-column floor "
              f"(legitimate since the 07-15 top-up restriction): {small}",
              flush=True)
    return keep


FEATURE_KEEP = _feature_keep()

# 2026-07-15 (user): two-pass bag/row split. A keep-list present => this run is
# the shipped KEEP-train (high fidelity: full bags, all rows). Absent => the
# SUPERSET train, whose only consumer is feature_select's stability vote, so it
# runs cheap: 3 bags + a 120k train-row sample per head.
# KNOWN REGIME MISMATCH (audit #3, user re-affirmed the cheap electorate
# 2026-07-15): the models that VOTE on feature stability are trained on ~a
# tenth of the rows and half the bags of the model that SHIPS, so votes may
# not perfectly transfer. Accepted for chain speed; feature_select.py reads
# the meta_stamp below and records both regimes in selection_report.json so
# every regen carries the caveat.
_KEEP_TRAIN = bool(FEATURE_KEEP)
if not _KEEP_TRAIN:
    LGBM_BAGS = 3
_SUPERSET_SAMPLE = 0 if _KEEP_TRAIN else 120_000   # train-row cap; 0 = all rows

# Early-stopping holdout (2026-07-15, audit fix #2): boosters used to
# early-stop on the CALIBRATION year, so the same rows chose each head's
# iteration count AND its blend weight, calibrator, and held-out SHAP votes —
# leaving cal-year predictions systematically optimistic. Boosters now fit on
# ~90% of the training rows and early-stop on the held-out ~10%, split by
# GamePk (deterministic, row-order independent, and a game never straddles
# the line). The calibration year is only touched AFTER the boosters are
# frozen, so the blend/calibrator/SHAP reads are clean of iteration choice.
# NOTE: changes every retrain vs pre-07-15 baselines — re-baseline after the
# first train with this in (the chain does that anyway).
ES_MOD, ES_BUCKET = 10, 3


def _es_split(tr):
    """(fit_rows, early_stop_rows): GamePk-bucketed ~10% early-stop slice."""
    es = (tr["GamePk"] % ES_MOD) == ES_BUCKET
    if not es.any() or es.all():        # degenerate tiny frame -> no split
        return tr, tr
    return tr[~es], tr[es]


def _apply_keep(name, cols):
    keep = FEATURE_KEEP.get(name)
    return [c for c in cols if c in keep] if keep else cols

# batter prop -> (target column, description)
PROPS = {
    "hr": ("y_hr", "home run"),
    "hit": ("y_hit", "1+ hit"),
    "hits2": ("y_hits2", "2+ hits"),
    "tb2": ("y_tb2", "2+ total bases"),
    "run": ("y_run", "run scored"),
    "rbi": ("y_rbi", "1+ RBI"),
    "bb": ("y_bb", "1+ walk"),
    "sb": ("y_sb", "stolen base"),
    "single": ("y_1b", "1+ single"),
    "double": ("y_2b", "1+ double"),
    "bk": ("y_bk1", "1+ batter strikeout"),
    "bk2": ("y_bk2", "2+ batter strikeouts"),
    "hrr2": ("y_hrr2", "2+ hits+runs+RBIs"),
    "hrr3": ("y_hrr3", "3+ hits+runs+RBIs"),
    # 2026-07-14 finish batch — H1 deep binaries (must beat their banked
    # count-calibrator bars, count_vs_binary.py table in the backlog; a
    # loser ships count-priced instead), H3 triple (1.21% base rate —
    # thinnest board binary, Platt load-bearing), H4 deeper thresholds
    "bk3": ("y_bk3", "3+ batter strikeouts"),
    "tb3": ("y_tb3", "3+ total bases"),
    "tb4": ("y_tb4", "4+ total bases"),
    "hrr4": ("y_hrr4", "4+ hits+runs+RBIs"),
    "triple": ("y_3b", "1+ triple"),
    "rbi2": ("y_rbi2", "2+ RBIs"),
    "run2": ("y_run2", "2+ runs scored"),
}

# Calibration-layer stacking for the thin-signal props: a logistic on
# logits blends a thin prop's score with thick-prop donors, fit ONLY on
# the calibration year (donor scores there are honest out-of-sample —
# donors never train on cal_yr). Applied identically by evaluate_deep and
# serving through predict.apply_stack; artifacts without a "stack" key
# pass through unchanged.
# BENCHED (2026-07-07, cal-2024 fit previewed on 2025 with the incumbent
# selection artifacts): self coefs ~0.95 with donors ~0 or canceling
# (double: hit -0.25 / tb2 +0.26); double got WORSE on AUC/logloss/ECE,
# single flat with worse ECE. The thin props' own models already extract
# what the donors know — they see the same features. Machinery stays
# (predict.apply_stack + the two-pass loops are pass-through no-ops);
# repopulate this dict to retry with different donors.
STACK_DONORS = {}

# Calibrator choice per head. Default = cal-year isotonic. PLATT_CAL heads
# use features.PlattCal (2-parameter logistic on the blended logit) instead:
# isotonic's free-form steps memorize thin-support extremes and serve
# overconfident tails on the holdout (worst where support is thinnest:
# winner ~2.4k cal games slope .68, double weakest signal slope .76, both
# years). Applied to the FULL binary surface per the no-pre-declared-targets
# policy (2026-07-13); the calibrator is per-head local, so the paired read
# verdicts each head's swap independently — keep Platt where it wins, restore
# isotonic per head where it harms. Platt is monotone, so ranking/AUC are
# untouched by construction everywhere; the read is pure pricing. Count
# heads are unaffected (their per-line calibrators are already logistic).
PLATT_CAL = set(PROPS) | {"winner"}

# Automated per-head calibrator choice (2026-07-15, tier-1 mechanics batch):
# instead of the global PLATT_CAL policy, each binary head picks its own
# calibrator — Platt (2-param), beta (3-param, fixes asymmetric
# miscalibration Platt can't), or isotonic (free-form) — by 5-fold
# GamePk-grouped CV log loss WITHIN the calibration year, then refits the
# winner on the full cal year. Deterministic folds (GamePk % 5), candidate
# order breaks ties toward the simplest monotone map. All three candidates
# serve through the artifact's "iso" key, so predict/evaluate need no
# changes. AUTO_CAL = False restores the PLATT_CAL-set policy exactly.
AUTO_CAL = True
CAL_CANDIDATES = ("platt", "beta", "iso")


def _make_cal(kind):
    if kind == "platt":
        return F.PlattCal()
    if kind == "beta":
        return F.BetaCal()
    return IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)


def _pick_calibrator(s_cal, y, gamepk, name):
    """(fitted calibrator, kind) for one head's cal-year blended scores."""
    if not AUTO_CAL:
        kind = "platt" if name.lower() in PLATT_CAL else "iso"
        return _make_cal(kind).fit(s_cal, y), kind
    folds = np.asarray(gamepk).astype(np.int64) % 5
    y = np.asarray(y, dtype=float)
    best_kind, best_ll = "platt", np.inf
    for kind in CAL_CANDIDATES:
        lls = []
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            # degenerate fold (single class either side, or too thin) -> skip
            if va_m.sum() < 50 or y[tr_m].std() == 0 or y[va_m].std() == 0:
                continue
            c = _make_cal(kind).fit(s_cal[tr_m], y[tr_m])
            p = np.clip(c.predict(s_cal[va_m]), 1e-6, 1 - 1e-6)
            lls.append(log_loss(y[va_m], p))
        if lls and np.mean(lls) < best_ll - 1e-9:
            best_ll, best_kind = float(np.mean(lls)), kind
    return _make_cal(best_kind).fit(s_cal, y), best_kind

# Count-style props: Poisson LGBM (starter-K pattern) + per-line logistic
# calibrators fit on the calibration year (predict.count_over). Batter heads
# exist for the MEANS (xSO, xHRR) — their half-point lines are priced by the
# calibrated binary heads above; starter heads price their own lines.
# `exclude` names the PROP_EXCLUDE entry supplying the column routing (inert
# while routing is retired — every head sees the full superset minus its
# keep-list; see the PROP_EXCLUDE note above).
COUNT_HEADS = {
    "xbk":  dict(frame="bat", target="bk_count", exclude="xbk",
                 lines=[0.5, 1.5, 2.5], desc="batter strikeouts"),
    # xhrr/xtb run ~2x Poisson variance (over-dispersed); a Tweedie objective
    # (power 1.3) models that heavier tail in the mean instead of only in the
    # post-hoc dispersion. The other heads stay Poisson (outs/xbk are UNDER
    # Poisson variance — Tweedie would push the wrong way).
    "xhrr": dict(frame="bat", target="hrr_count", exclude="xhrr",
                 tweedie=1.3, lines=[1.5, 2.5, 3.5], desc="hits+runs+RBIs"),
    "xtb":  dict(frame="bat", target="tb_count", exclude="xtb",
                 tweedie=1.3, lines=[1.5, 2.5, 3.5], desc="total bases"),
    "outs": dict(frame="starts", target="y_outs", exclude=None,
                 lines=[14.5, 15.5, 16.5, 17.5, 18.5],
                 desc="starter outs recorded"),
    "pbb":  dict(frame="starts", target="y_pbb", exclude=None,
                 lines=[0.5, 1.5, 2.5], desc="starter walks allowed"),
    "pha":  dict(frame="starts", target="y_pha", exclude=None,
                 lines=[3.5, 4.5, 5.5, 6.5], desc="starter hits allowed"),
    "per":  dict(frame="starts", target="y_per", exclude=None,
                 lines=[1.5, 2.5, 3.5, 4.5], desc="starter earned runs"),
    # 2026-07-14 finish batch — H6: the rest of the expected-stat-line.
    # MEANS ONLY: their per-line calibrators are banked by fit_line_cals
    # but never ship as prices (binaries own the batter lines, 07-13
    # shoot-out). xrbi runs over-dispersed (var/mean 1.61 measured 2025)
    # -> Tweedie 1.3 like xhrr/xtb; the others stay Poisson.
    "xh":   dict(frame="bat", target="h_count", exclude="hit",
                 lines=[0.5, 1.5, 2.5], desc="hits"),
    "xrun": dict(frame="bat", target="run_count", exclude="run",
                 lines=[0.5, 1.5], desc="runs scored"),
    "xrbi": dict(frame="bat", target="rbi_count", exclude="rbi",
                 tweedie=1.3, lines=[0.5, 1.5, 2.5], desc="RBIs"),
    "xbb":  dict(frame="bat", target="bb_count", exclude="bb",
                 lines=[0.5, 1.5], desc="walks"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Shadow-calibrated eps (2026-07-14, 1F selection upgrade — Boruta-lite):
# shuffled copies of representative superset columns ride the SUPERSET
# train at zero marginal cost; feature_select.py sets each head's eps to a
# high quantile of its shadow-column SHAP shares — the "essentially unused"
# floor becomes empirical per head instead of the fixed EPS constant.
# Deterministic (fixed seed + fixed donor stride), so both suites and any
# re-run see identical shadow values; keep-lists are written shadow-free
# (feature_select strips the prefix), so the keep train drops them and
# NOTHING shadow-flavored can ever ship.
SHADOW_PREFIX = "shdw_"
# tg/wf bumped 10/8 -> 16 (07-14 A2b): the thin frames' pooled shadow
# pools were the noisiest p95 estimates on the board; bf/sf stay put so
# the cached frames (which carry their shadows) remain valid. tg/wf
# shadows are added after the cache loads, so this needs no rebuild —
# feature_select re-derives them from the same SHADOW_N, staying in sync.
SHADOW_N = {"bf": 20, "sf": 12, "tg": 16, "wf": 16}


def add_shadow_cols(frame, cols, n, seed=0):
    """Append n shuffled copies of evenly-spaced numeric superset columns.
    Idempotent (skips if shadows already present, e.g. a cached frame)."""
    if any(c.startswith(SHADOW_PREFIX) for c in frame.columns):
        return [c for c in frame.columns if c.startswith(SHADOW_PREFIX)]
    num = [c for c in cols if c in frame.columns and c not in F.CAT_COLS
           and pd.api.types.is_numeric_dtype(frame[c])]
    if not num or n <= 0:
        return []
    donors = [num[int(i * len(num) / n) % len(num)] for i in range(n)]
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(donors):
        name = f"{SHADOW_PREFIX}{i}_{c}"
        frame[name] = rng.permutation(frame[c].to_numpy())
        out.append(name)
    return out


def shadow_cols_of(frame):
    return [c for c in frame.columns if c.startswith(SHADOW_PREFIX)]


def set_categories(df, cat_levels):
    for c, levels in cat_levels.items():
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=levels)
    return df


def _fit_logistic(tr, cols, target, w=None):
    """Regularized logistic on numeric features (categoricals dropped) — a
    learner diverse from the trees, so blending the two helps. w = per-row
    sample weights (recency decay), routed to the LR step only."""
    num_cols = [c for c in cols if c not in F.CAT_COLS]
    pipe = Pipeline([
        # F.inf_to_nan lives in the features module so the pickled pipeline
        # resolves it from predict.py/evaluate_deep.py too (not just __main__).
        ("clean", FunctionTransformer(F.inf_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=0.3, solver="lbfgs")),
    ])
    pipe.fit(tr[num_cols], tr[target],
             **({"lr__sample_weight": w} if w is not None else {}))
    return pipe, num_cols


def fit_classifier(df, cols, target, train_yrs, cal_yr, test_yr, name,
                   params=None, n_bags=1, n_xgb=0, n_cb=0):
    tr = df[df["Season"].isin(train_yrs)]
    if _SUPERSET_SAMPLE and len(tr) > _SUPERSET_SAMPLE:   # superset train only
        tr = tr.sample(n=_SUPERSET_SAMPLE, random_state=0)
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr]
    fit, es = _es_split(tr)         # cal year never picks iteration counts
    w_fit, w_es = _recency_w(fit, cal_yr), _recency_w(es, cal_yr)
    models = []
    for b in range(n_bags):
        p = dict(params or LGB_CLS)
        if b:                       # bag 0 = the incumbent default seed
            p["random_state"] = b
        m = lgb.LGBMClassifier(**p)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])],
              eval_sample_weight=None if w_es is None else [w_es],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        models.append(m)
    for b in range(n_xgb):          # family members join AFTER the LGBM bag
        m = F.InfSafe(xgb_lib.XGBClassifier(**XGB_CLS, random_state=b))
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              sample_weight_eval_set=None if w_es is None else [w_es],
              eval_set=[(es[cols], es[target])], verbose=False)
        models.append(m)
    cat_here = [c for c in cols if c in F.CAT_COLS]
    for b in range(n_cb):
        # CatBoost: train rows weighted; the ES eval stays unweighted (a
        # weighted eval needs a Pool, which would bypass CatSafe's cleaning)
        m = F.CatSafe(CatBoostClassifier(**CB_CLS, random_seed=b,
                                         cat_features=cat_here), cat_here)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])])
        models.append(m)
    model = F.MeanBag(models) if len(models) > 1 else models[0]

    # diverse second learner + blend weight chosen on the calibration year
    # (the LR has no early stopping, so it may use the full training slice)
    lr, num_cols = _fit_logistic(tr, cols, target, w=_recency_w(tr, cal_yr))
    g_cal = model.predict_proba(ca[cols])[:, 1]
    l_cal = lr.predict_proba(ca[num_cols])[:, 1]
    yca = ca[target].to_numpy()
    best_w, best_ll = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        ll = log_loss(yca, np.clip(F.blend(g_cal, l_cal, w, BLEND_SPACE),
                                   1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll, best_w = ll, w

    s_cal = F.blend(g_cal, l_cal, best_w, BLEND_SPACE)
    iso, cal_kind = _pick_calibrator(s_cal, yca, ca["GamePk"].to_numpy(), name)

    g_te = model.predict_proba(te[cols])[:, 1]
    l_te = lr.predict_proba(te[num_cols])[:, 1]
    p_te = iso.predict(F.blend(g_te, l_te, best_w, BLEND_SPACE))
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": int(model.best_iteration_ or 0),
        "blend_gbm_weight": round(float(best_w), 2),
        "blend_space": BLEND_SPACE,
        "calibrator": cal_kind,
        "auc": float(roc_auc_score(y, p_te)),
        "acc": float(((p_te >= 0.5).astype(float) == y).mean()),
        "logloss": float(log_loss(y, p_te)),
        "logloss_baserate": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p_te)),
        "brier_baserate": float(brier_score_loss(y, base)),
    }
    # calibration by decile
    q = pd.qcut(p_te, 10, duplicates="drop")
    cal_tab = pd.DataFrame({"pred": p_te, "y": y}).groupby(q, observed=True).agg(
        pred_mean=("pred", "mean"), actual=("y", "mean"), n=("y", "size"))
    metrics["calibration"] = [
        {"pred": round(r.pred_mean, 4), "actual": round(r.actual, 4), "n": int(r.n)}
        for r in cal_tab.itertuples()]
    # daily top-10 lift (ranking value: does the top of the list hit?)
    day = pd.DataFrame({"d": te["Date"].values, "p": p_te, "y": y})
    top = day.sort_values("p", ascending=False).groupby("d").head(10)
    metrics["top10_daily_hit_rate"] = float(top["y"].mean())
    log(f"{name} [{test_yr}]: AUC {metrics['auc']:.4f} | "
        f"logloss {metrics['logloss']:.4f} (base {metrics['logloss_baserate']:.4f}) | "
        f"brier {metrics['brier']:.4f} (base {metrics['brier_baserate']:.4f}) | "
        f"top10/day {metrics['top10_daily_hit_rate']:.3f} vs base "
        f"{metrics['base_rate']:.3f} | gbm wt {best_w:.2f} | cal {cal_kind}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols, "w": best_w,
            "iso": iso, "blend_space": BLEND_SPACE}
    return prop, metrics


def fit_winner(wf, cols, target, mu_map, train_yrs, cal_yr, test_yr, name):
    """Home-win model: small-capacity GBM + logistic, then a second blend
    with the runs-model Poisson win probability (a diverse signal — park,
    weather, starter run prevention), then isotonic calibration. Both blend
    weights are chosen on the calibration year.

    mu_map: per-GamePk expected runs (mu_away, mu_home) from the runs
    model, or None to skip the Poisson component. The runs model trains on
    2020-2024, so pass None whenever the calibration year falls inside that
    range: its in-sample predictions look falsely sharp there, the blend
    collapses onto them, and the isotonic miscalibrates (this corrupted the
    first 2025 backtest — cal 2024 is training data for the runs model)."""
    from predict import poisson_win
    tr = wf[wf["Season"].isin(train_yrs)]
    ca = wf[wf["Season"] == cal_yr]
    te = wf[wf["Season"] == test_yr]
    fit, es = _es_split(tr)         # cal year never picks iteration counts
    w_fit, w_es = _recency_w(fit, cal_yr), _recency_w(es, cal_yr)
    members = []
    for b in range(LGBM_BAGS):      # bag 0 = the pre-bagging incumbent seed
        p = dict(LGB_WIN)
        if b:
            p["random_state"] = b
        m = lgb.LGBMClassifier(**p)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])],
              eval_sample_weight=None if w_es is None else [w_es],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        members.append(m)
    for b in range(XGB_BAGS):       # family members join AFTER the incumbent
        m = F.InfSafe(xgb_lib.XGBClassifier(**XGB_WIN, random_state=b))
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              sample_weight_eval_set=None if w_es is None else [w_es],
              eval_set=[(es[cols], es[target])], verbose=False)
        members.append(m)
    cat_here = [c for c in cols if c in F.CAT_COLS]
    for b in range(CB_BAGS):
        # CatBoost: train rows weighted; ES eval unweighted (Pool would
        # bypass CatSafe's cleaning) — same note as fit_classifier
        m = F.CatSafe(CatBoostClassifier(**CB_WIN, random_seed=b,
                                         cat_features=cat_here), cat_here)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])])
        members.append(m)
    model = F.MeanBag(members) if len(members) > 1 else members[0]
    lr, num_cols = _fit_logistic(tr, cols, target, w=_recency_w(tr, cal_yr))

    def parts(d):
        g = model.predict_proba(d[cols])[:, 1]
        l = lr.predict_proba(d[num_cols])[:, 1]
        if mu_map is None:
            return g, l, np.full(len(d), np.nan)
        mus = mu_map.reindex(d["GamePk"])
        pois = np.array([poisson_win(h, a) for h, a in
                         zip(mus["mu_home"], mus["mu_away"])])
        return g, l, pois

    yca = ca[target].to_numpy()

    def pick_w(a, b):
        best_w, best_ll = 1.0, np.inf
        for w in np.linspace(0.0, 1.0, 21):
            ll = log_loss(yca, np.clip(F.blend(a, b, w, BLEND_SPACE),
                                       1e-6, 1 - 1e-6))
            if ll < best_ll:
                best_ll, best_w = ll, w
        return best_w

    g_cal, l_cal, pois_cal = parts(ca)
    w1 = pick_w(g_cal, l_cal)
    s_cal = F.blend(g_cal, l_cal, w1, BLEND_SPACE)
    pois_cal = np.where(np.isfinite(pois_cal), pois_cal, s_cal)
    w_ml = 1.0 if mu_map is None else pick_w(s_cal, pois_cal)
    s2_cal = F.blend(s_cal, pois_cal, w_ml, BLEND_SPACE)
    iso, cal_kind = _pick_calibrator(s2_cal, yca, ca["GamePk"].to_numpy(),
                                     "winner")

    g_te, l_te, pois_te = parts(te)
    s_te = F.blend(g_te, l_te, w1, BLEND_SPACE)
    pois_te = np.where(np.isfinite(pois_te), pois_te, s_te)
    p_te = iso.predict(F.blend(s_te, pois_te, w_ml, BLEND_SPACE))
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": int(model.best_iteration_ or 0),
        "blend_gbm_weight": round(float(w1), 2),
        "blend_ml_weight": round(float(w_ml), 2),
        "blend_space": BLEND_SPACE,
        "calibrator": cal_kind,
        "auc": float(roc_auc_score(y, p_te)),
        "acc": float(((p_te >= 0.5).astype(float) == y).mean()),
        "logloss": float(log_loss(y, p_te)),
        "logloss_baserate": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p_te)),
        "brier_baserate": float(brier_score_loss(y, base)),
    }
    log(f"{name} [{test_yr}]: AUC {metrics['auc']:.4f} | acc "
        f"{metrics['acc']:.3f} | logloss {metrics['logloss']:.4f} "
        f"(base {metrics['logloss_baserate']:.4f}) | gbm wt {w1:.2f} | "
        f"ML-vs-poisson wt {w_ml:.2f}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols, "w": w1,
            "w_ml": w_ml, "iso": iso, "blend_space": BLEND_SPACE}
    return prop, metrics


def fit_poisson(df, cols, target, train_yrs, cal_yr, test_yr, name, baseline,
                n_bags=1, tweedie_power=None, n_xgb=0, n_cb=0):
    """Poisson (default) count regression, or Tweedie when tweedie_power is set
    (a compound Poisson-Gamma objective, variance power in (1,2)). Tweedie lets
    the MEAN model an over-dispersed right tail directly — total bases / H+R+RBI
    run ~2x Poisson variance — instead of leaning entirely on the post-hoc
    cal-year dispersion. Serving is unchanged: .predict() still returns E[y]."""
    tr = df[df["Season"].isin(train_yrs)]
    if _SUPERSET_SAMPLE and len(tr) > _SUPERSET_SAMPLE:   # superset train only
        tr = tr.sample(n=_SUPERSET_SAMPLE, random_state=0)
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr].copy()
    fit, es = _es_split(tr)         # cal year never picks iteration counts
    w_fit, w_es = _recency_w(fit, cal_yr), _recency_w(es, cal_yr)
    tweedie = tweedie_power is not None
    models = []
    for b in range(n_bags):
        p = dict(LGB_POIS)
        if tweedie:
            p = dict(p, objective="tweedie",
                     tweedie_variance_power=tweedie_power)
        if b:                       # bag 0 = the incumbent default seed
            p["random_state"] = b
        m = lgb.LGBMRegressor(**p)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])],
              eval_sample_weight=None if w_es is None else [w_es],
              eval_metric=("tweedie" if tweedie else "poisson"),
              callbacks=[lgb.early_stopping(150, verbose=False)])
        models.append(m)
    for b in range(n_xgb):          # family members join AFTER the LGBM bag
        p = dict(XGB_POIS)
        if tweedie:
            p = dict(p, objective="reg:tweedie",
                     tweedie_variance_power=tweedie_power,
                     eval_metric=f"tweedie-nloglik@{tweedie_power}")
        m = F.InfSafe(xgb_lib.XGBRegressor(**p, random_state=b))
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              sample_weight_eval_set=None if w_es is None else [w_es],
              eval_set=[(es[cols], es[target])], verbose=False)
        models.append(m)
    cat_here = [c for c in cols if c in F.CAT_COLS]
    for b in range(n_cb):
        p = dict(CB_POIS)
        if tweedie:
            tw = f"Tweedie:variance_power={tweedie_power}"
            p = dict(p, loss_function=tw, eval_metric=tw)
        # CatBoost: train rows weighted; ES eval unweighted (Pool would
        # bypass CatSafe's cleaning) — same note as fit_classifier
        m = F.CatSafe(CatBoostRegressor(**p, random_seed=b,
                                        cat_features=cat_here),
                      cat_here, exponent=True)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])])
        models.append(m)
    model = F.MeanBag(models) if len(models) > 1 else models[0]
    pred = model.predict(te[cols])
    y = te[target].to_numpy()
    bl = baseline(te)
    metrics = {
        "n_train": len(tr), "n_test": len(te),
        "best_iter": int(model.best_iteration_ or 0),
        "mae": float(mean_absolute_error(y, pred)),
        "mae_baseline": float(mean_absolute_error(y, bl)),
        "mean_actual": float(y.mean()), "mean_pred": float(pred.mean()),
    }
    log(f"{name} [{test_yr}]: MAE {metrics['mae']:.3f} "
        f"(baseline {metrics['mae_baseline']:.3f}) | "
        f"mean pred {metrics['mean_pred']:.2f} vs actual {metrics['mean_actual']:.2f}")
    return model, metrics


def naive_hr_baseline(te, slot_pa, league_hr_pa):
    """P(HR) if you only used season HR/PA and lineup slot."""
    rate = te["s_hr_pa"].fillna(league_hr_pa).clip(0, 0.15)
    exp_pa = te["slot"].map(slot_pa).fillna(4.1)
    return 1 - (1 - rate) ** exp_pa


def fit_line_cals(mu_cal, y_cal, lines):
    """Per-line logistic calibrators on the CAL year: P(over line) as a
    direct monotone 2-parameter function of mu. One shared implementation
    for every count-family (count heads, starter K, game total) so the
    pricing mechanism is identical across the whole line surface. Degenerate
    lines (single-class cal year) are skipped — consumers fall back to
    nb_over."""
    mu_cal = np.asarray(mu_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    out = {}
    for line in lines:
        over = (y_cal > line).astype(int)
        if 0 < over.mean() < 1:
            out[line] = LogisticRegression(C=1e6, max_iter=1000).fit(
                mu_cal.reshape(-1, 1), over)
    return out


def train_suite(bf, sf, tg, wf, cat_levels, train_yrs, cal_yr, test_yr):
    """Fit the full model suite (every PROPS binary head, the count heads,
    starter K, team runs, winner) on one train/cal/test split. Returns (artifacts, metrics) with the same
    artifact keys regardless of split, so evaluate_deep can score either the
    shipping suite or the selection suite identically."""
    # shadow columns (1F) join every superset; _apply_keep drops them on a
    # keep train because keep-lists are written shadow-free
    bat_cols = F.batter_feature_cols() + shadow_cols_of(bf)
    st_cols = F.starts_feature_cols() + shadow_cols_of(sf)
    tg_cols = _apply_keep("total",
                          F.team_game_feature_cols() + shadow_cols_of(tg))
    metrics, props = {}, {}

    for name, (target, _desc) in PROPS.items():
        cols = _apply_keep(name, [c for c in bat_cols
                                  if c not in PROP_EXCLUDE.get(name, ())])
        params = PROP_PARAMS.get(name)
        mono = MONOTONE.get(name)
        if mono:    # categoricals and unlisted cols get 0 (unconstrained)
            params = dict(params or LGB_CLS,
                          monotone_constraints=[mono.get(c, 0) for c in cols],
                          monotone_constraints_method="advanced")
        prop, m = fit_classifier(bf, cols, target,
                                 train_yrs, cal_yr, test_yr, name.upper(),
                                 params=params,
                                 n_bags=LGBM_BAGS,
                                 n_xgb=XGB_BAGS, n_cb=CB_BAGS)
        prop["cols"] = cols
        props[name] = prop
        metrics[f"{name}_{test_yr}"] = m

    # thin-prop stacking (STACK_DONORS): fit on the calibration year, then
    # log the test-year effect for a first read — evaluate_deep applies the
    # same stacker (predict.apply_stack) and its Section 11 is the verdict.
    # The per-prop metrics above stay PLAIN; the STACK log line shows both.
    if STACK_DONORS:
        from predict import apply_stack, predict_prop  # local: avoids cycle
        from recalibrate import _logit
        ca = bf[bf["Season"] == cal_yr]
        te = bf[bf["Season"] == test_yr]
        p_ca, p_te = {}, {}
        for name, donors in STACK_DONORS.items():
            for d in {name, *donors}:
                if d not in p_ca:
                    p_ca[d] = predict_prop(props[d], ca)
                    p_te[d] = predict_prop(props[d], te)
            y_ca = ca[PROPS[name][0]].to_numpy()
            Z = np.column_stack([_logit(p_ca[name])]
                                + [_logit(p_ca[d]) for d in donors])
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(Z, y_ca)
            props[name]["stack"] = {"donors": list(donors), "lr": lr}
            y_te = te[PROPS[name][0]].to_numpy()
            p0 = np.clip(p_te[name], 1e-4, 1 - 1e-4)
            p1 = apply_stack(props[name], p_te[name], p_te)
            coefs = " ".join(f"{n}:{c:+.2f}" for n, c in
                             zip(["self", *donors], lr.coef_[0]))
            log(f"STACK {name.upper()} [{test_yr}]: AUC "
                f"{roc_auc_score(y_te, p1):.4f} (plain "
                f"{roc_auc_score(y_te, p_te[name]):.4f}) | logloss "
                f"{log_loss(y_te, p1):.4f} (plain {log_loss(y_te, p0):.4f}) "
                f"| coefs {coefs}")

    def k_baseline(te):
        league = sf.loc[sf["Season"].isin(train_yrs), "y_so"].mean()
        per_start = te["ps_k_bf"] * (te["ps_BF"] / te["p_starts_season"])
        return per_start.fillna(league).clip(0, 15)

    # k_cols is what the artifact ships as st_cols — the K model's serving
    # contract (full starts superset minus the keep-list; routing retired).
    k_cols = _apply_keep("k", list(st_cols))
    k_model, m = fit_poisson(sf, k_cols, "y_so", train_yrs, cal_yr, test_yr,
                             "K", k_baseline, n_bags=LGBM_BAGS,
                             n_xgb=XGB_BAGS, n_cb=CB_BAGS)
    metrics[f"k_{test_yr}"] = m

    # Starter-K dispersion on the CALIBRATION year (never the holdout): real K
    # counts run a touch over Poisson variance, so predict.py prices K P(over)
    # with a negative binomial (nb_over) using this factor.
    sf_cal = sf[sf["Season"] == cal_yr]
    kp_cal = k_model.predict(sf_cal[k_cols])
    k_disp = float(np.mean((sf_cal["y_so"].to_numpy() - kp_cal) ** 2)
                   / np.mean(kp_cal))
    metrics[f"k_dispersion_{cal_yr}"] = k_disp
    log(f"starter-K dispersion ({cal_yr} cal year): {k_disp:.2f} "
        f"(Poisson assumes 1.00)")

    # K per-line calibrators (2026-07-13 full-surface calibration pass): K
    # lines were the only starter family still priced by the raw NB/Poisson
    # tail; they now get the same cal-year logistic pricing as outs/pbb/pha
    # (predict.k_over consumes, NB fallback for old artifacts).
    from predict import K_LINES, TOTAL_LINES
    k_line_cals = fit_line_cals(kp_cal, sf_cal["y_so"].to_numpy(), K_LINES)

    # count heads (starter-K pattern): Poisson mean + cal-year dispersion
    count_models = {}
    for cname, ch in COUNT_HEADS.items():
        frame = bf if ch["frame"] == "bat" else sf
        cols = ([c for c in bat_cols
                 if c not in PROP_EXCLUDE.get(ch["exclude"], ())]
                if ch["frame"] == "bat" else list(st_cols))
        cols = _apply_keep(cname, cols)
        tr_mean = frame.loc[frame["Season"].isin(train_yrs),
                            ch["target"]].mean()

        def cbase(te, _m=tr_mean, _n=cname):
            if _n == "xbk":
                return (te["s_k_pct_sh"] * 4.1).fillna(_m)
            if _n == "outs":
                return (te["p_ip_per_start"] * 3).fillna(_m).clip(0, 27)
            if _n == "pbb":
                return (te["ps_bb_bf"] * (te["ps_BF"] / te["p_starts_season"])
                        ).fillna(_m).clip(0, 8)
            if _n == "pha":
                return (te["ps_h_bf"] * (te["ps_BF"] / te["p_starts_season"])
                        ).fillna(_m).clip(0, 12)
            if _n == "per":
                # season ERA (ER per 9 IP) scaled to this start's expected IP
                return (te["ps_era"] * te["p_ip_per_start"] / 9
                        ).fillna(_m).clip(0, 10)
            return pd.Series(_m, index=te.index)  # xhrr: league mean

        model, m = fit_poisson(frame, cols, ch["target"], train_yrs, cal_yr,
                               test_yr, cname.upper(), cbase,
                               n_bags=LGBM_BAGS,
                               tweedie_power=ch.get("tweedie"),
                               n_xgb=XGB_BAGS, n_cb=CB_BAGS)
        ca = frame[frame["Season"] == cal_yr]
        mu_cal = model.predict(ca[cols])
        y_cal = ca[ch["target"]].to_numpy()
        disp = float(np.mean((y_cal - mu_cal) ** 2) / np.mean(mu_cal))
        m["dispersion_cal"] = round(disp, 4)
        # per-line logistic calibrators on the CAL year (the count-head
        # analog of the binary props' isotonic): P(over line) as a direct
        # monotone function of mu. Outs/batter-K counts run UNDER Poisson
        # variance (bounded by PA / the manager's hook), so nb_over — which
        # can only widen, never narrow — misprices their tails; consumers
        # fall back to nb_over only when a line has no calibrator.
        line_cals = fit_line_cals(mu_cal, y_cal, ch["lines"])
        metrics[f"{cname}_{test_yr}"] = m
        count_models[cname] = {"model": model, "cols": cols, "disp": disp,
                               "lines": ch["lines"], "line_cals": line_cals,
                               "frame": ch["frame"],
                               "target": ch["target"], "desc": ch["desc"]}

    def team_baseline(te):
        league = tg.loc[tg["Season"].isin(train_yrs), "y_runs"].mean()
        return te["off_r_pg"].fillna(league)

    team_runs_model, m = fit_poisson(tg, tg_cols, "y_runs", train_yrs, cal_yr,
                                     test_yr, "TEAM RUNS", team_baseline,
                                     n_bags=LGBM_BAGS,
                                     n_xgb=XGB_BAGS, n_cb=CB_BAGS)
    metrics[f"team_runs_{test_yr}"] = m

    # Game-total dispersion, also on the calibration year: real totals ran
    # ~2.3x Poisson variance, which made pure Poisson P(over) worse than the
    # base rate at low lines. predict.py switches to a negative binomial.
    tg_cal = tg[tg["Season"] == cal_yr]
    pr_cal = team_runs_model.predict(tg_cal[tg_cols])
    per_game = pd.DataFrame({"g": tg_cal["GamePk"].to_numpy(), "mu": pr_cal,
                             "y": tg_cal["y_runs"].to_numpy()}).groupby("g").sum()
    total_disp = float(np.mean((per_game["y"] - per_game["mu"]) ** 2)
                       / np.mean(per_game["mu"]))
    metrics[f"total_dispersion_{cal_yr}"] = total_disp
    log(f"game-total dispersion ({cal_yr} cal year): {total_disp:.2f} "
        f"(Poisson assumes 1.00)")

    # total-runs per-line calibrators (same 2026-07-13 pass): the raw NB
    # tail left the total lines the worst-calibrated family on the board
    # (slopes .84-.95); per-game cal-year mu vs actual totals, predict.
    # total_over consumes with NB fallback for exotic odds-store lines.
    total_line_cals = fit_line_cals(per_game["mu"].to_numpy(),
                                    per_game["y"].to_numpy(), TOTAL_LINES)

    # H5 team_total head-ification (2026-07-14): the per-TEAM line surface
    # off the same runs model. TEAM-level cal-year NB dispersion (the game
    # total's ~2.3 does NOT transfer — team variance is its own number) +
    # per-line calibrators for the team-total lines the books post.
    from predict import TEAM_TOTAL_LINES
    team_total_disp = float(np.mean((tg_cal["y_runs"].to_numpy()
                                     - pr_cal) ** 2) / np.mean(pr_cal))
    metrics[f"team_total_dispersion_{cal_yr}"] = round(team_total_disp, 4)
    log(f"team-total dispersion ({cal_yr} cal year): {team_total_disp:.2f} "
        f"(Poisson assumes 1.00)")
    team_line_cals = fit_line_cals(pr_cal, tg_cal["y_runs"].to_numpy(),
                                   TEAM_TOTAL_LINES)

    # dedicated winner model, blended with the runs-model Poisson win prob.
    # The suite's own runs model never trains on cal_yr (it early-stops
    # there), so the mu_map is safe for the blend-weight fit — see the
    # fit_winner docstring for why cal-year-in-training is the failure mode.
    mu_all = team_runs_model.predict(tg[tg_cols])
    mu_map = (pd.DataFrame({"GamePk": tg["GamePk"].to_numpy(),
                            "Home": tg["Home"].to_numpy(), "mu": mu_all})
              .pivot_table(index="GamePk", columns="Home", values="mu")
              .rename(columns={0: "mu_away", 1: "mu_home"}))
    win_cols = _apply_keep("winner",
                           F.win_feature_cols() + shadow_cols_of(wf))
    win_model, m = fit_winner(wf, win_cols, "y_home_win", mu_map,
                              train_yrs, cal_yr, test_yr, "WINNER")
    win_model["cols"] = win_cols
    te = wf[wf["Season"] == test_yr]
    m["acc_home_baseline"] = float(te["y_home_win"].mean())
    log(f"WINNER [{test_yr}]: pick accuracy {m['acc']:.3f} vs always-home "
        f"{m['acc_home_baseline']:.3f}")
    metrics[f"winner_{test_yr}"] = m

    # guard (07-14 F3): on a keep train NOTHING shadow-flavored may persist
    # in a booster's serving column list. The live hazard is a head MISSING
    # from feature_keep.json (e.g. a new head added after the last selection
    # regen): it trains unrestricted on the shadowed superset here, then
    # serving NaN-fills the absent shdw_ columns (predict._prep) — silent
    # train/serve skew. bat_cols stays exempt: it is only the frame-prep
    # superset; every booster reads its own per-head list.
    if FEATURE_KEEP:
        serving_lists = {**{n: p["cols"] for n, p in props.items()},
                         **{n: cm["cols"] for n, cm in count_models.items()},
                         "k": k_cols, "total": tg_cols, "winner": win_cols}
        shadowed = sorted(n for n, cols in serving_lists.items()
                          if any(c.startswith(SHADOW_PREFIX) for c in cols))
        assert not shadowed, (
            f"shadow columns persist in the serving column lists of "
            f"{shadowed} — these heads are missing from feature_keep.json; "
            f"regenerate it over all heads (feature_select.py --write) or "
            f"set SELECT_FEATURES = False")

    artifacts = {
        # role/version stamp (2026-07-15, audit #8/#16): predict.py refuses
        # role="superset" (and any shdw_ serving contract), so a superset
        # intermediate can never silently serve again.
        "meta_stamp": {
            "artifact_version": 2,
            "role": "keep" if _KEEP_TRAIN else "superset",
            "lgbm_only_temp": LGBM_ONLY_TEMP,
            "bags": {"lgbm": LGBM_BAGS, "xgb": XGB_BAGS, "cb": CB_BAGS},
            "superset_sample": _SUPERSET_SAMPLE,
            "recency_decay": RECENCY_DECAY,
            "blend_space": BLEND_SPACE,
            "auto_cal": AUTO_CAL,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "props": props,
        "k_model": k_model, "team_runs_model": team_runs_model,
        "win_model": win_model, "total_disp": total_disp, "k_disp": k_disp,
        "k_line_cals": k_line_cals, "total_line_cals": total_line_cals,
        "team_total_disp": team_total_disp, "team_line_cals": team_line_cals,
        "count_models": count_models,
        # st_cols = the K model's column contract (predict/evaluate feed it
        # to k_model); the count heads carry their own cols. k_cols drops
        # _PARK_OFF from the shared starts superset.
        "bat_cols": bat_cols, "st_cols": k_cols, "tg_cols": tg_cols,
        "cat_levels": cat_levels,
        "metrics": metrics,
        # evaluate_deep reads these instead of hardcoding seasons
        "years": {"train": list(train_yrs), "cal": int(cal_yr),
                  "test": int(test_yr)},
    }
    return artifacts, metrics


def suite_years(bf, min_rows=2000):
    """Derive the shipping split from the seasons actually in the data:
    the newest season with at least min_rows batter-games is the
    confirm-only holdout, the season before it calibrates, everything
    earlier trains. A brand-new season graduates in automatically once
    ~2 weeks of games accrue (below that its rows are simply not scored,
    and the previous split keeps shipping). The selection suite is the
    same split shifted one season back."""
    counts = bf["Season"].value_counts()
    seasons = sorted(int(s) for s in counts.index if counts[s] >= min_rows)
    if len(seasons) < 4:
        raise SystemExit(f"need at least 4 seasons of data to form the "
                         f"train/cal/holdout splits, have {seasons}")
    return seasons[:-2], seasons[-2], seasons[-1]


def main():
    global RECENCY_DECAY
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild feature frames even if cached")
    ap.add_argument("--select", action="store_true",
                    help="train ONLY the model-selection suite (one season "
                         "back from shipping) — the fast iteration loop. "
                         "The default run trains it too, then the shipping "
                         "models on top.")
    ap.add_argument("--decay", type=float, default=None,
                    help="override RECENCY_DECAY for this run (used by "
                         "decay_sweep.py; 1.0 = no recency weighting)")
    args = ap.parse_args()
    if args.decay is not None:
        RECENCY_DECAY = args.decay
    if RECENCY_DECAY < 1.0:
        log(f"recency sample-weighting ON: decay {RECENCY_DECAY} per season "
            f"back from each suite's cal year")

    # role banner (audit #8): superset runs write models_superset*.joblib and
    # never touch the serving artifacts; the temp-regime flag prints loudly so
    # a forgotten flip can't silently ship an LGBM-only ensemble.
    role = "keep" if _KEEP_TRAIN else "superset"
    log(f"run role: {role.upper()} "
        + ("(keep-list applied; writes models_bt.joblib / models.joblib)"
           if _KEEP_TRAIN else
           "(no keep-list; writes models_superset_bt.joblib / "
           "models_superset.joblib — serving artifacts untouched)"))
    if LGBM_ONLY_TEMP:
        log("LGBM_ONLY_TEMP is ON: XGB/CB families disabled. STANDING "
            "regime per user 2026-07-15 — the shipped ensemble is the LGBM "
            "6-bag + LR blend for the foreseeable future; flip False only "
            "when the 3-family ensemble is deliberately brought back.")

    cache = ART / "frames.joblib"
    if cache.exists() and not args.rebuild:
        log("loading cached feature frames")
        frames = joblib.load(cache)
    else:
        log("loading raw data")
        raw = F.load_raw()
        log("building batter frame (this is the big one)")
        bf = F.build_batter_frame(raw)
        log(f"batter frame: {len(bf):,} rows")
        log("building starts frame")
        # bf supplies the opposing-lineup aggregates (lu_*) for the K model
        sf = F.build_starts_frame(raw, bf)
        log(f"starts frame: {len(sf):,} rows")
        log("building game frame")
        # bf also supplies the posted-lineup quality/style aggregates
        # (2026-07-14 #19/#30/#32)
        gf = F.build_game_frame(raw, bf)
        log(f"game frame: {len(gf):,} rows")
        # shadow columns (1F): persisted in the cache so feature_select's
        # held-out SHAP pass sees the exact columns the boosters trained on
        add_shadow_cols(bf, F.batter_feature_cols(), SHADOW_N["bf"])
        add_shadow_cols(sf, F.starts_feature_cols(), SHADOW_N["sf"])
        frames = {"bf": bf, "sf": sf, "gf": gf}
        joblib.dump(frames, cache, compress=3)
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]
    # cached pre-1F frames: add the (deterministic) shadows on the fly
    add_shadow_cols(bf, F.batter_feature_cols(), SHADOW_N["bf"])
    add_shadow_cols(sf, F.starts_feature_cols(), SHADOW_N["sf"])

    # exclude 7-inning doubleheaders from training grain
    bf = bf[~bf["ShortGame"].fillna(False)].copy()
    sf = sf[~sf["ShortGame"].fillna(False)].copy()

    cat_levels = {}
    for c in F.CAT_COLS:
        vals = set()
        for frame in (bf, sf, gf):
            if c in frame.columns:
                vals |= set(frame[c].dropna().astype(str).unique())
        cat_levels[c] = sorted(vals)
    for frame in (bf, sf, gf):
        set_categories(frame, cat_levels)

    # per-team runs frame; game totals and win probability derive from it.
    # Canonical row order: LightGBM's bagging draws depend on row order, so
    # without this, unrelated upstream merge changes shuffle rows and move
    # MAE by ~0.005-0.01 — pure noise that pollutes baseline diffs.
    tg = F.build_team_game_frame(gf.dropna(subset=["total_runs"]))
    tg = tg.dropna(subset=["y_runs"])
    tg = tg.sort_values(["GamePk", "Home"]).reset_index(drop=True)
    set_categories(tg, cat_levels)
    add_shadow_cols(tg, F.team_game_feature_cols(), SHADOW_N["tg"])

    wf = gf[~gf["ShortGame"].fillna(False)].dropna(subset=["y_home_win"])
    wf = wf.sort_values("GamePk").reset_index(drop=True)  # canonical order
    wf = wf.copy()
    add_shadow_cols(wf, F.win_feature_cols(), SHADOW_N["wf"])

    # season splits derived from the data — no code edit at the annual
    # rollover; the holdout promotes itself once the new season has games
    train_yrs, cal_yr, hold_yr = suite_years(bf)
    sel_tr, sel_cal, sel_te = train_yrs[:-1], train_yrs[-1], cal_yr

    # -- selection suite (always refreshed): iterate here, never vs the
    # holdout --
    log(f"=== SELECTION suite (train<={sel_tr[-1]}, cal {sel_cal}, test "
        f"{sel_te}) — {hold_yr} stays untouched ===")
    sel_art, sel_metrics = train_suite(bf, sf, tg, wf, cat_levels,
                                       sel_tr, sel_cal, sel_te)
    sel_art["trained_on"] = (f"selection suite: {sel_tr[0]}-{sel_tr[-1]}, "
                             f"calibrated {sel_cal}, tested {sel_te} "
                             f"({hold_yr} untouched)")
    sel_path = ART / ("models_bt.joblib" if _KEEP_TRAIN
                      else "models_superset_bt.joblib")
    sel_metrics_path = ART / ("metrics_select.json" if _KEEP_TRAIN
                              else "metrics_select_superset.json")
    joblib.dump(sel_art, sel_path, compress=3)
    with open(sel_metrics_path, "w") as f:
        json.dump(sel_metrics, f, indent=2)
    log(f"saved selection artifacts to {sel_path}")
    if args.select:
        log(f"next: python Model/evaluate_deep.py   (scores this suite on "
            f"{sel_te})")
        return

    # -- shipping suite, tested (confirm-only) on the holdout ------
    log(f"=== final models (train<={train_yrs[-1]}, cal {cal_yr}, test "
        f"{hold_yr} holdout) ===")
    artifacts, metrics = train_suite(bf, sf, tg, wf, cat_levels,
                                     train_yrs, cal_yr, hold_yr)
    bat_cols = artifacts["bat_cols"]
    props = artifacts["props"]

    # naive season-rate HR baseline, for context in metrics.json
    slot_pa = bf[bf["Season"].isin(train_yrs)].groupby("slot")["PA"].mean().to_dict()
    league_hr_pa = (bf.loc[bf["Season"].isin(train_yrs), "HR"].sum()
                    / bf.loc[bf["Season"].isin(train_yrs), "PA"].sum())
    te = bf[bf["Season"] == hold_yr]
    nb = naive_hr_baseline(te, slot_pa, league_hr_pa)
    metrics[f"hr_{hold_yr}"]["logloss_naive_seasonrate"] = float(
        log_loss(te["y_hr"], nb.clip(1e-4, 1 - 1e-4)))
    metrics[f"hr_{hold_yr}"]["brier_naive_seasonrate"] = float(
        brier_score_loss(te["y_hr"], nb))

    # In-season drift offsets for serving: a per-prop log-odds shift fit on
    # the current (holdout) season's PAST games. 2026-07-15 (audit #16):
    # written to artifacts/inseason_offsets.json, NOT into models.joblib —
    # the serving artifact stays free of holdout-fit parameters; predict.py
    # reads the sidecar only under --recal. Keep-trains only (a superset
    # model's offsets would describe a model that never serves).
    if _KEEP_TRAIN:
        import recalibrate as R
        from predict import predict_prop as _predict_prop
        te_hold = bf[bf["Season"] == hold_yr]
        inseason_offsets = {}
        for name, (target, _desc) in PROPS.items():
            y_h = te_hold[target].to_numpy()
            if len(y_h) > 200 and 0 < y_h.mean() < 1:
                p_h = _predict_prop(props[name], te_hold[bat_cols])
                inseason_offsets[name] = round(
                    float(R.fit_logit_offset(p_h, y_h)), 4)
            else:
                inseason_offsets[name] = 0.0
        metrics["inseason_offsets"] = inseason_offsets
        (ART / "inseason_offsets.json").write_text(json.dumps(
            {"year": int(hold_yr), "offsets": inseason_offsets,
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
        log(f"in-season drift offsets ({hold_yr}) -> inseason_offsets.json")
    else:
        log("superset run: in-season offsets skipped (keep-trains only)")

    # multi-HR correction: E[HR | HR>=1], for expected-HR outputs
    tr_hr = bf[bf["Season"].isin(train_yrs) & (bf["hr_count"] >= 1)]
    multi_hr = float(tr_hr["hr_count"].mean())

    artifacts.update({
        "multi_hr": multi_hr,
        "slot_pa": slot_pa, "league_hr_pa": league_hr_pa,
        "metrics": metrics,
        "trained_on": (f"{train_yrs[0]}-{train_yrs[-1]}, calibrated "
                       f"{cal_yr}, holdout-tested {hold_yr} YTD"),
    })
    ship_path = ART / ("models.joblib" if _KEEP_TRAIN
                       else "models_superset.joblib")
    ship_metrics_path = ART / ("metrics.json" if _KEEP_TRAIN
                               else "metrics_superset.json")
    joblib.dump(artifacts, ship_path, compress=3)
    with open(ship_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"saved artifacts to {ship_path}")

    # feature importances for the HR model (top 25). Family bags have no
    # blended importances — read the incumbent LGBM member (bag 0).
    hr_gbm = props["hr"]["gbm"]
    if isinstance(hr_gbm, F.MeanBag):
        hr_gbm = hr_gbm.models[0]
    imp = pd.Series(hr_gbm.feature_importances_,
                    index=props["hr"]["cols"])
    log("top HR-model features:\n" +
        imp.sort_values(ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
