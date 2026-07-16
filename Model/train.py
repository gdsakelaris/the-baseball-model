"""Train the MLB prediction models.

Models (LightGBM seed bags + CatBoost family members — the 2-family
ensemble, 2026-07-15 PM: XGBoost retired permanently, wiring kept):
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
                             mean_poisson_deviance, mean_tweedie_deviance,
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
LGB_WIN = dict(n_estimators=3000, learning_rate=0.03, num_leaves=15,
               min_child_samples=300, subsample=0.9, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=6.0, objective="binary",
               verbose=-1)
# lr .02->.03 = the 2026-07-15 chain sweep's 'lr_up' (winner's biggest
# guarded win: CV logloss -0.0030, AUC +0.0007 on the new 18-col keep-list)
# nl 7->15, mcs 200->300 = the 2026-07-16 chain-2 'reg_up' from the FIRST
# MIRROR-AWARE winner sweep (B2: fold-train+watch rows mirrored, persp_home
# in cols — the regime the winner actually ships in): ensemble ll -0.0006
# vs the shipped config, gates cleared; lr_up/leaves7 scored identical to
# default because they're already baked in here. USER-approved 07-16.

# Recency sample-weighting (2026-07-15, tier-1 mechanics batch): every
# booster and LR fit weights row i by RECENCY_DECAY ** (cal_yr - Season_i),
# so 2015 stops counting as much as 2024 across a decade of era shifts
# (juiced ball, pitch clock, shift ban, bigger bases). 1.0 = OFF (all-ones
# weights are skipped entirely — bit-identical to the incumbent). The value
# is swept on the SELECTION suite by Model/decay_sweep.py (--decay overrides
# per run); bake the sweep winner here before the ship chain. Blend weights,
# calibrators, and dispersions still fit UNWEIGHTED on the cal year — decay
# shapes what the boosters learn, not how they are priced.
RECENCY_DECAY = 0.95   # swept 2026-07-15 {1.0,.95,.9,.8} on the selection
                       # suite (decay_sweep.py, banked): best mean delta
                       # (-0.00017) at the widest breadth (23/35 heads,
                       # incl. k/outs/pbb/pha), worst harm +0.0005. The
                       # ship chain's paired read verdicts it in-batch.


def _decay_for(head_key):
    """Effective recency decay for one head: the per-head override when
    baked (RECENCY_HEAD_DECAY, from decay_sweep --per-head), else the
    global RECENCY_DECAY. head_key = lowercase prop/count name, or
    "k"/"total"/"winner"."""
    return RECENCY_HEAD_DECAY.get(head_key, RECENCY_DECAY)


def _recency_w(frame, cal_yr, decay=None):
    """Per-row training weights decay**(cal_yr - Season); None when the
    decay is off so every fit call stays bit-identical to the incumbent."""
    if decay is None:
        decay = RECENCY_DECAY
    if decay >= 1.0:
        return None
    yrs = np.clip(cal_yr - frame["Season"].to_numpy(dtype=float), 0, None)
    return decay ** yrs


# GBM-vs-LR blend space (2026-07-15, tier-1 mechanics batch): the 21-point
# grid over w now combines the two members' LOG-ODDS (see features.blend).
# With the Platt/beta calibrator fit downstream on the blended score, the
# logit-space grid is exactly the full 2-member logistic stack (free
# per-member coefficients), which probability-space blending cannot express.
# Artifacts carry blend_space so predict.py serves the same arithmetic;
# absent key = old artifact = probability space. "prob" restores incumbent.
BLEND_SPACE = "logit"

# ---------------------------------------------------------------------------
# 2026-07-15 PM diversity/calibration batch (user-directed, one ship chain):
# every mechanism below lands together and the chain's evaluate_deep --paired
# read adjudicates the package. Each flag reverts its piece independently.
# ---------------------------------------------------------------------------
# Per-family logistic stacking: instead of the 21-point grid over ONE weight
# between the pooled GBM bag mean and the LR, a small logistic regression on
# [per-family mean logits..., LR logit] fit on the calibration year gives
# free PER-FAMILY weights (the N-member generalization of the BLEND_SPACE
# note above — the grid was exactly the 2-member stack). Members' logits are
# highly correlated, so a mild ridge (FSTACK_C) keeps coefficients stable
# across retrains; the downstream calibrator absorbs any global shrink.
# False restores the incumbent grid path exactly.
FAMILY_STACK = True
FSTACK_C = 50.0
# Bagged calibrators: B bootstrap resamples of the calibration DAYS, one
# calibrator (of the AUTO_CAL-chosen kind) per resample, served as the mean
# curve (features.BaggedCal). Targets the observed 2025->2026 ECE decay:
# part of it is single-year calibrator variance, which bagging shrinks.
# 0 = off (single full-year fit, the incumbent).
# 2026-07-16 chain 2: 25 -> 0 per cal_lab on the chain-1 ship stash — every
# top global combo is bag=0 (the mean-logloss cost of bagging is small but
# consistent; per-head bests are mixed, so the paired read arbitrates).
# USER-approved 07-16; revert = 25.
CAL_BAG_B = 0
# Multi-year calibration support: pool the PRIOR year's honest out-of-sample
# scores (from the suite trained one season back — the selection suite's
# boosters, captured in _CAL_STASH at zero extra train cost) into the
# stack/calibrator/line-cal/dispersion fits, older year discounted by
# CAL_POOL_DECAY. Doubles the pricing support without touching what the
# boosters learn. The SELECTION suite has no earlier sibling in a standard
# run, so it pools only under --prestash (an extra throwaway suite train) —
# without it the 2025 paired read sees every OTHER change but not this one,
# and the 2026 confirm is the multi-year read. False = off.
MULTI_YEAR_CAL = True
CAL_POOL_DECAY = 0.75
# Pooling depth (2026-07-15 late PM): how many YEARS of support the pricing
# fits may pool — 2 = current + one prior (the shipped default), 3 adds a
# second prior year at CAL_POOL_DECAY**2 when the stash has it (a --prestash
# chain leaves the shipping suite three deep: prestash-cal + selection-cal +
# own cal). Adjudicate a depth change OFFLINE first via cal_lab.py's
# pool-years knob — the stash sidecar makes it a minutes-scale experiment.
# 2026-07-16 chain 2: 2 -> 1 per cal_lab TWICE (chain-1 eve run + the
# 07-16 ship-stash run): pool=1 wins mean logloss and posts broad ECE
# gains (hr .0037->.0005, rbi .0063->.0015, tb2 .0083->.0019). Depth 1 =
# current-year-only support; the stash machinery stays live for reverts
# and for cal_lab experiments. USER-approved 07-16; revert = 2.
CAL_POOL_YEARS = 1
# Early-stop refit: after each booster member early-stops on its ~10%
# GamePk holdout (audit fix #2), refit that member at its chosen iteration
# count on 100% of the training rows — recovering the 10% data sacrifice.
# Keep-trains only (the superset electorate stays cheap); roughly doubles
# booster fit time. False = serve the ES-fit members (the incumbent).
ES_REFIT = True
# Winner mirror augmentation: the winner trains on ~10k games — every other
# head's smallest data by 20x. Each TRAIN row is duplicated with home/away
# swapped (paired cols exchanged, d_* diffs negated, elo_prob_home flipped,
# y inverted) and persp_home 1->0 marking the flipped perspective, doubling
# effective rows; cal/test rows are NEVER mirrored and serving always sends
# persp_home=1. False = off.
WINNER_MIRROR = True
# Per-head recency decay: overrides RECENCY_DECAY for listed heads (keys =
# prop names / count names / "k" / "total" / "winner"). Populate from
# decay_sweep.py --per-head output; --decay CLI overrides EVERYTHING (sweep
# isolation).
# BAKED 2026-07-16 (chain-1 5-value sweep on the regen lists): 26 raw
# per-head argmins gated on margin (>= the head kind's EPS vs the global
# 0.95) + shape (adjacent-value corroboration) -> 3 clear winners, all
# counts (faster-drifting surfaces); the winner's 0.80 missed (gain .0005,
# non-adjacent runner-up = jitter). Evidence: artifacts/decay_sweep/.
RECENCY_HEAD_DECAY = {"outs": 0.85, "per": 0.80, "xtb": 0.80}

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
# Bag diversification (2026-07-15 late PM, chain-2 candidate, OFF for the
# 07-15 PM adjudication chain): the first N reseeded LGBM bags (never bag 0,
# the incumbent seed) train with extra_trees=True — randomized split
# thresholds decorrelate members beyond what reseeding can, cheap
# within-family diversity with no new dependencies. Flip to e.g. 2 and
# adjudicate on the next keep-chain's paired read.
# 2026-07-16 chain 2 (B3): flipped 0 -> 2 for this chain's adjudication,
# per the ratified CHAIN2_PLAN. Members b=1,2 of every LGBM bag train with
# extra_trees=True; member 0 stays the bit-identical incumbent seed.
BAG_DIVERSIFY = 2

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
# 2026-07-14 (user): LGBM_ONLY_TEMP dropped the XGB+CB families so the
# FINISH_PLAN build batch could iterate on retrains cheaply.
# 2026-07-15 PM (user, diversity batch): the flag is RETIRED and the family
# roster is now explicit — CatBoost returns at 2 bags (the exact quantity
# shipped+validated 07-13; a third split policy the LGBM reseeds cannot
# express), XGBoost is retired PERMANENTLY at 0 (user call; the InfSafe
# wiring below stays intact as rollback insurance, and feature_select /
# predict._force_xgb_cpu / param_sweep all adapt to whatever families are
# present). NOTE: the current keep-lists + PROP_PARAMS were swept under the
# LGBM-only regime, so CB rejoins on a keep-list it didn't vote on — the
# chain's evaluate_deep --paired read is the arbiter, and the next
# feature_select regen (fresh superset train) lets CB vote.
XGB_BAGS = 0    # RETIRED permanently (user 2026-07-15 PM); wiring kept
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
# predict needs Exponent). Set CB_BAGS = 0 to drop the family. CatBoost
# fits cost 2-4x an XGB fit (GPU) — time the first keep-train against the
# 06:00 window (see the 07-15 PM batch runbook) before trusting the cadence.
CB_BAGS = 2     # RESTORED 2026-07-15 PM (user): the 07-13 shipped quantity
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
              bagging_temperature=2.0,
              verbose=0, allow_writing_files=False,
              task_type="GPU", devices="0")
# bagging_temperature=2.0 = cb_sweep 'bayes_hot' (2026-07-16, the first
# CB-side sweep): winner-panel ll -0.0020 vs default, gates cleared —
# hotter Bayesian bootstrap buys the data-starved winner more member
# diversity. CB_CLS/CB_POIS stay default (6/8 binary + 3/3 count panel
# heads recommended no change). USER-approved 07-16.

# Per-prop LightGBM overrides — RE-SWEPT 2026-07-15 (chain sweep) with
# param_sweep (LGBM-only isolation, K=4 day-grouped CV, Season<=2024) on
# the chain's 0.85-shadow keep-lists under the deviance/ECE-guarded
# objective. Every entry cleared EPS_LL 0.0004 inside the AUC/ECE bands
# (prop_params_sweep.json holds the evidence). Notables vs the morning
# sweep: hits2/double DROPPED their overrides (raw LGB_CLS won), several
# established heads re-priced lighter (run med2->light, sb med2->med),
# and the 7 new heads took their first tuned profiles. winner's lr_up
# (.02->.03) lives in LGB_WIN above; count winners in COUNT_PARAMS below.
# CAVEAT unchanged: single-bag CV vs the 6-bag ship — the package is
# confirmed by evaluate_deep --paired on the actual keep-train.
_REG_LIGHT = dict(min_child_samples=160)
_REG_MED  = dict(num_leaves=63, min_child_samples=160)
_REG_MED2 = dict(num_leaves=63, min_child_samples=300,
                 colsample_bytree=0.7, reg_lambda=6.0)
_REG_HEAVY = dict(num_leaves=31, min_child_samples=300,
                  colsample_bytree=0.7, reg_lambda=6.0)
# RE-WIRED 2026-07-15 PM (regen batch): param_sweep --ensemble on the
# CB-voted keep-lists (first families-aware sweep — the objective is the
# 2-family bag's OOF, not LGBM solo), reconciled against hpo_sweep's
# LGBM-solo winners via the ensemble-scored tiebreak
# (artifacts/tiebreak_0715pm.json). Wholesale-replace semantics: heads
# recommending 'default' DROPPED their overrides (hit, bk3, tb3, triple,
# rbi2, run2 — with CB in the bag, their old LGBM-only regularization no
# longer earns its keep). hrr3 = the one tiebreak head where the Optuna
# winner beat the profile ladder ensemble-scored.
PROP_PARAMS = {
    "run":    dict(LGB_CLS, **_REG_MED2),
    "rbi":    dict(LGB_CLS, **_REG_MED),
    "hits2":  dict(LGB_CLS, **_REG_LIGHT),
    "tb2":    dict(LGB_CLS, **_REG_HEAVY),
    "single": dict(LGB_CLS, **_REG_MED2),
    "bb":     dict(LGB_CLS, **_REG_LIGHT),
    "sb":     dict(LGB_CLS, **_REG_HEAVY),
    "bk":     dict(LGB_CLS, **_REG_MED2),
    "bk2":    dict(LGB_CLS, **_REG_MED2),
    "hrr2":   dict(LGB_CLS, **_REG_MED2),
    "hrr3":   dict(LGB_CLS,               # hpo_sweep winner (tiebreak-confirmed)
                   learning_rate=0.023073442963913587, num_leaves=24,
                   min_child_samples=306,
                   colsample_bytree=0.6541935159879131,
                   reg_lambda=3.2013064859813976,
                   subsample=0.7163126478294832, max_bin=127,
                   min_split_gain=0.04568969383667694),
    "tb4":    dict(LGB_CLS, **_REG_MED),
    "hrr4":   dict(LGB_CLS, **_REG_MED),
}

# Count-head LightGBM overrides (2026-07-15 chain sweep — the first count
# winners ever; fragments mirror param_sweep.CNT_PROFILES). Applied by
# fit_poisson via its params argument; tweedie heads keep their objective
# overlay (it applies AFTER these). All cleared EPS_DEV 0.0015 inside the
# MAE band (xrbi's MAE +0.0009 is within the 0.0050 band by design —
# deviance prices the lines, MAE only guards the point estimate).
_CNT_MED = dict(num_leaves=31, min_child_samples=120)
_CNT_HEAVY = dict(num_leaves=31, min_child_samples=300,
                  colsample_bytree=0.7, reg_lambda=6.0)
# EMPTIED 2026-07-15 PM (regen batch): the families-aware ensemble re-sweep
# recommended 'default' for EVERY count head incl. k — with the 2 CB
# members cutting variance, none of the 07-15 AM LGBM-only overrides
# (k lr_slow, total med, xhrr/xrbi heavy, xtb med) cleared the deviance
# gate. Wholesale-replace semantics -> all dropped; raw LGB_POIS ships.
# The fragments above stay (param_sweep.CNT_PROFILES mirrors them).
# RE-SWEPT 2026-07-16 (chain 2): hpo_sweep 60-trial Optuna over the 13
# count heads — 9 cleared LGBM-solo gates, then the ensemble tiebreak
# (2 CB cached/fold, identical folds) killed 6, the same macro-lesson as
# chain 1. The 3 survivors below cleared deviance+MAE gates ENSEMBLE-
# scored (outs dev -.0015, per -.0017, k -.0017). xhrr/xtb/pbb/pha/total/
# xrbi stay on raw LGB_POIS; for xrbi (chain 1's replicated MAE harm)
# BOTH its Optuna winner and the old _CNT_HEAVY failed the ensemble read
# — base is best on MAE, so the harm suspect moves elsewhere (chain-2
# paired read re-measures). USER-approved 07-16; delete an entry to
# revert that head to LGB_POIS.
COUNT_PARAMS = {   # Optuna dicts verbatim from artifacts/hpo_sweep.json
    "outs": dict(LGB_POIS, learning_rate=0.019936849536100608,
                 num_leaves=33, min_child_samples=24,
                 colsample_bytree=0.5322641785198797,
                 reg_lambda=3.119486926286563,
                 subsample=0.8553474347283905, max_bin=127,
                 min_split_gain=0.05407620105358256),
    "per":  dict(LGB_POIS, learning_rate=0.019329001499361304,
                 num_leaves=8, min_child_samples=25,
                 colsample_bytree=0.984140790945196,
                 reg_lambda=9.98763058156031,
                 subsample=0.6008065801810717, max_bin=127,
                 min_split_gain=0.09042365691059093),
    "k":    dict(LGB_POIS, learning_rate=0.023690837108114676,
                 num_leaves=9, min_child_samples=44,
                 colsample_bytree=0.7230251013296163,
                 reg_lambda=20.616220635769253,
                 subsample=0.7021555285549907, max_bin=127,
                 min_split_gain=0.2511961562678737),
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
SELECT_FEATURES = True    # 2026-07-15 PM regen: keep-lists regenerated from
                          # the 2-family superset (3 LGBM + 2 CB electorate,
                          # CB voting for the first time; 1895->1999 cols)


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
    # Superset electorate keeps the FULL 2 CatBoost members (2026-07-15 PM,
    # user-directed selection regen): a lone member votes a coarse 0/1
    # use-fraction — the exact granularity problem that made the LGBM bags
    # uniform on 07-14 — and this regen exists precisely so the cb family's
    # vote is real. The cheap-electorate regime (3 LGBM bags + the 120k row
    # sample) still applies; the audit-#3 mismatch note stands.
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
# RE-TEST (2026-07-15 PM batch, doctrine: benched verdicts are time-stamped,
# not law): the 07-07 bench tried CROSS-stat donors on medium-thin heads and
# predates the 36-head board entirely. This retry is a different hypothesis —
# SAME-LADDER donors for the genuinely thin deep-threshold heads (triple
# 1.2% base rate, the H4 4+/3+ family, the H1 deep binaries): the lower rung
# sees the same features but at 5-20x the positive-label support, so its
# logit is a shrinkage target the thin head's own model cannot manufacture.
# Ladder PAV runs downstream of the stack (predict/evaluate order:
# predict_prop -> apply_stack -> enforce_ladders), so coherence is preserved.
# Each head's verdict is per-head local in the paired read — keep winners,
# re-bench losers.
STACK_DONORS = {
    "triple": ("double", "hit"),
    "bk3":    ("bk2",),
    "tb4":    ("tb3",),
    "hrr4":   ("hrr3",),
    "rbi2":   ("rbi",),
    "run2":   ("run",),
}

# B1 init_score donor warm-starts (2026-07-16, chain 2) — the STACK_DONORS
# idea INSIDE the trees. Each thin deep-threshold head's LGBM members boost
# from scale * (its ladder lower rung's bag-mean logit) and learn only the
# residual: the lower rung sees the same features at 5-20x the positive
# support, so its logit is a shrinkage prior the thin head's own trees
# cannot manufacture. The donor's train-row logits are IN-SAMPLE (the donor
# trained on those rows), which makes the raw offset optimistically sharp —
# so the per-head scale is picked on the CAL YEAR (donors never train
# there) by one probe member per grid point; scale 0.0 is always probed
# first and recovers the incumbent member exactly, so the mechanism
# self-gates per head. (USER design call 07-16: in-sample + scale grid,
# over a K-fold OOF donor pass; if winners plateau at the low scales,
# the honest-OOF pass is the chain-3 upgrade.) Serving: the offset rides
# at the FAMILY-LOGIT level — residual members' mean logit + scale *
# donor bag-mean logit — computed by the same arithmetic in
# fit_classifier's cal/test designs and predict.predict_prop's
# "init_donor" branch (donor member references in the artifact; joblib
# dedupes shared objects, ~zero size cost). CB members stay offset-free;
# the family stack re-weights around the sharpened LGBM family. Keep-trains
# only (superset/selection-regen paths untouched); empty dict = OFF.
INIT_SCORE_DONORS = {
    "hits2":  "hit",
    "tb4":    "tb3",
    "hrr4":   "hrr3",
    "rbi2":   "rbi",
    "run2":   "run",
    "triple": "double",
}
INIT_SCALE_GRID = (0.25, 0.5, 0.75, 1.0)    # 0.0 (incumbent) always probed
# serve-side donor logits are rebuilt from the donor's members alone, so a
# donor that is itself offset-boosted would serve wrong — forbid chaining
assert not set(INIT_SCORE_DONORS) & set(INIT_SCORE_DONORS.values())
_DONOR_STASH = {}   # donor head_key -> {models, cols, z_tr, z_ca, z_te}

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


def _fit_cal(kind, s, y, w=None):
    """Fit one calibrator of `kind`, routing optional sample weights to the
    right argument per implementation (isotonic takes sample_weight; the
    custom Platt/beta fits take w)."""
    c = _make_cal(kind)
    if w is None:
        return c.fit(s, y)
    if kind in ("platt", "beta"):
        return c.fit(s, y, w=w)
    return c.fit(s, y, sample_weight=w)


def _bag_cal(kind, s, y, dates, w=None):
    """Day-block bootstrap bag of CAL_BAG_B calibrators (features.BaggedCal),
    the mean curve served. Resamples DAYS with replacement — the paired
    read's unit of independence — so the bag reflects between-day variance,
    not just row noise. Degenerate resamples (single class) are skipped; if
    everything degenerates, falls back to the single full fit."""
    if not CAL_BAG_B:
        return _fit_cal(kind, s, y, w), kind
    dates = np.asarray(dates)
    days = np.unique(dates)
    idx_of_day = {d: np.flatnonzero(dates == d) for d in days}
    rng = np.random.default_rng(0)
    members = []
    for _ in range(CAL_BAG_B):
        take = np.concatenate([idx_of_day[d] for d in
                               rng.choice(days, size=len(days), replace=True)])
        yb = y[take]
        if yb.std() == 0:
            continue
        members.append(_fit_cal(kind, s[take], yb,
                                None if w is None else w[take]))
    if not members:
        return _fit_cal(kind, s, y, w), kind
    kind_b = f"{kind}+bag{len(members)}"
    return F.BaggedCal(members, kind=kind_b), kind_b


def _pick_calibrator(s_cal, y, gamepk, name, dates=None, w=None):
    """(fitted calibrator, kind) for one head's cal blended scores. The CV
    pick runs on the (possibly multi-year pooled, weighted) support; the
    winner is then day-block bagged when CAL_BAG_B > 0 and `dates` are
    given, else fit once on the full support (the incumbent path)."""
    y = np.asarray(y, dtype=float)
    if not AUTO_CAL:
        kind = "platt" if name.lower() in PLATT_CAL else "iso"
    else:
        folds = np.asarray(gamepk).astype(np.int64) % 5
        best_kind, best_ll = "platt", np.inf
        for kind in CAL_CANDIDATES:
            lls = []
            for f in range(5):
                tr_m, va_m = folds != f, folds == f
                # degenerate fold (single class either side, too thin) -> skip
                if va_m.sum() < 50 or y[tr_m].std() == 0 or y[va_m].std() == 0:
                    continue
                c = _fit_cal(kind, s_cal[tr_m], y[tr_m],
                             None if w is None else w[tr_m])
                p = np.clip(c.predict(s_cal[va_m]), 1e-6, 1 - 1e-6)
                lls.append(log_loss(y[va_m], p,
                                    sample_weight=None if w is None
                                    else w[va_m]))
            if lls and np.mean(lls) < best_ll - 1e-9:
                best_ll, best_kind = float(np.mean(lls)), kind
        kind = best_kind
    if dates is not None:
        return _bag_cal(kind, s_cal, y, dates, w)
    return _fit_cal(kind, s_cal, y, w), kind


# Multi-year calibration stash (2026-07-15 PM batch): the SELECTION suite
# trains first and its calibration-year scores are honest out-of-sample for
# the SHIPPING suite's train_yrs[-1] — so the shipping suite's stack /
# calibrator / line-cal / dispersion fits can pool two years of support at
# ZERO extra booster cost. Keyed ("prop"|"winner"|"count"|"lines", head,
# cal_yr); populated by every fit, consumed when MULTI_YEAR_CAL and the
# matching prior-year entry exists (i.e. the suites ran in this process,
# same family/bag regime by construction).
_CAL_STASH = {}


def _stash_priors(kind, head_key, train_yrs, ref, mat=None):
    """[(entry, years_back)] for up to CAL_POOL_YEARS-1 stashed prior years,
    NEWEST first, stopping at the first missing year or a design-shape
    mismatch vs ref[mat] (regime guard)."""
    out = []
    if not MULTI_YEAR_CAL:
        return out
    for k in range(1, CAL_POOL_YEARS):
        if k > len(train_yrs):
            break
        e = _CAL_STASH.get((kind, head_key, train_yrs[-k]))
        if e is None:
            break
        if (mat and mat in ref and mat in e
                and np.shape(e[mat])[1:] != np.shape(ref[mat])[1:]):
            break
        out.append((e, k))
    return out


def _pool_cal(cur, priors):
    """Concatenate (discounted) prior-year support with the current year.
    priors = [(entry, years_back)] newest-first (from _stash_priors); a
    prior k years back gets weight CAL_POOL_DECAY**k, the current year 1.
    Rows are ordered oldest -> current."""
    entries = [e for e, _ in reversed(priors)] + [cur]
    ws = [CAL_POOL_DECAY ** k for _, k in reversed(priors)] + [1.0]
    out = {}
    for key in cur:
        out[key] = np.concatenate([np.asarray(e[key]) for e in entries])
    out["w"] = np.concatenate([np.full(len(np.asarray(e["y"])), w)
                               for e, w in zip(entries, ws)])
    return out


def _pool_years(priors, cal_yr, train_yrs):
    return [int(train_yrs[-k]) for _, k in reversed(priors)] + [int(cal_yr)]

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


# LR blend-member ridge grid (2026-07-15 PM batch, user add): C is picked
# PER HEAD on cal-year logloss instead of the hand-set 0.3. The per-family
# stack downstream prices the LR member with a free coefficient, so member
# quality matters more than it did under the fixed-weight blend. 0.3 = the
# incumbent, listed first so exact ties keep it.
LR_C_GRID = (0.3, 0.1, 1.0)


def _fit_logistic(tr, cols, target, w=None, ca=None):
    """Regularized logistic on numeric features (categoricals dropped) — a
    learner diverse from the trees, so blending the two helps. w = per-row
    sample weights (recency decay), routed to the LR step only. ca = the
    calibration-year frame: when given, the ridge strength is picked from
    LR_C_GRID by cal-year logloss; None (or a degenerate cal year) keeps
    the fixed incumbent C=0.3. Returns (pipeline, num_cols, C)."""
    num_cols = [c for c in cols if c not in F.CAT_COLS]

    def _pipe(C):
        return Pipeline([
            # F.inf_to_nan lives in the features module so the pickled
            # pipeline resolves it from predict.py/evaluate_deep.py too
            # (not just __main__).
            ("clean", FunctionTransformer(F.inf_to_nan)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=C, solver="lbfgs")),
        ])

    fitkw = {"lr__sample_weight": w} if w is not None else {}
    grid = LR_C_GRID
    if ca is None or len(ca) < 100 or ca[target].nunique() < 2:
        grid = (0.3,)
    best_ll, best_pipe, best_C = np.inf, None, None
    for C in grid:
        pipe = _pipe(C).fit(tr[num_cols], tr[target], **fitkw)
        if len(grid) == 1:
            return pipe, num_cols, C
        p = np.clip(pipe.predict_proba(ca[num_cols])[:, 1], 1e-6, 1 - 1e-6)
        ll = log_loss(ca[target].to_numpy(), p)
        if ll < best_ll - 1e-9:
            best_ll, best_pipe, best_C = ll, pipe, C
    return best_pipe, num_cols, best_C


def _refit_lgbm(ctor, p, best_iter, X, y, w, init_score=None):
    """ES-refit: same params/seed at the ES-chosen iteration count, fit on
    100% of the training rows (recovers the ~10% ES holdout sacrifice).
    init_score: scaled donor offset on the refit rows (INIT_SCORE_DONORS)."""
    m = ctor(**{**p, "n_estimators": max(int(best_iter), 1)})
    kw = {} if init_score is None else {"init_score": init_score}
    m.fit(X, y, sample_weight=w, **kw)
    return m


def _refit_cb(ctor, p, best_iter, X, y, w, cat_here, exponent=False):
    p = {k: v for k, v in p.items() if k != "early_stopping_rounds"}
    p["iterations"] = max(int(best_iter), 1)
    seed = p.pop("_seed")
    m = F.CatSafe(ctor(**p, random_seed=seed, cat_features=cat_here),
                  cat_here, exponent=exponent)
    m.fit(X, y, sample_weight=w)
    return m


def fit_classifier(df, cols, target, train_yrs, cal_yr, test_yr, name,
                   params=None, n_bags=1, n_xgb=0, n_cb=0, head_key=None):
    head_key = head_key or name.lower()
    decay = _decay_for(head_key)
    tr = df[df["Season"].isin(train_yrs)]
    if _SUPERSET_SAMPLE and len(tr) > _SUPERSET_SAMPLE:   # superset train only
        tr = tr.sample(n=_SUPERSET_SAMPLE, random_state=0)
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr]
    fit, es = _es_split(tr)         # cal year never picks iteration counts
    w_fit, w_es = (_recency_w(fit, cal_yr, decay),
                   _recency_w(es, cal_yr, decay))
    w_tr = _recency_w(tr, cal_yr, decay)
    do_refit = ES_REFIT and _KEEP_TRAIN

    # B1 donor offset: resolve the lower rung's stashed logits and pick the
    # offset scale on the cal year BEFORE the bag loop — one probe member
    # per grid point, scored on raw-sigmoid cal-year logloss (pre-cal
    # sharpness is what the offset buys; the calibrator layer is fit later
    # on whatever wins). The winning probe IS bag member 0 (its s=0 probe
    # is bit-identical to the incumbent member-0 fit), so the grid costs
    # len(INIT_SCALE_GRID) extra fits per recipient head, nothing more.
    init_scale, probe_keep = 0.0, None
    zd_fit = zd_es = zd_tr = zd_ca = zd_te = None
    donor_key = INIT_SCORE_DONORS.get(head_key) if _KEEP_TRAIN else None
    donor = _DONOR_STASH.get(donor_key) if donor_key else None
    if donor is not None:
        zd_fit = donor["z_tr"].loc[fit.index].to_numpy()
        zd_es = donor["z_tr"].loc[es.index].to_numpy()
        zd_tr = donor["z_tr"].loc[tr.index].to_numpy()
        zd_ca, zd_te = donor["z_ca"], donor["z_te"]
        y_ca_probe = ca[target].to_numpy()
        base_p = dict(params or LGB_CLS)
        best_ll = np.inf
        for s in (0.0,) + INIT_SCALE_GRID:
            m = lgb.LGBMClassifier(**base_p)
            ikw = ({"init_score": s * zd_fit,
                    "eval_init_score": [s * zd_es]} if s else {})
            m.fit(fit[cols], fit[target], sample_weight=w_fit,
                  eval_set=[(es[cols], es[target])],
                  eval_sample_weight=None if w_es is None else [w_es],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(150, verbose=False)], **ikw)
            z = m.predict(ca[cols], raw_score=True) + s * zd_ca
            ll = log_loss(y_ca_probe, 1.0 / (1.0 + np.exp(-z)))
            if ll < best_ll - 1e-7:     # ties break toward the smaller scale
                best_ll, init_scale = ll, s
                probe_keep = (m, int(m.best_iteration_ or 0))

    models, best_iters = [], []
    slices, pos = {}, 0             # family -> (start, end) into models
    for b in range(n_bags):
        p = dict(params or LGB_CLS)
        if b:                       # bag 0 = the incumbent default seed
            p["random_state"] = b
            if b <= BAG_DIVERSIFY:
                p["extra_trees"] = True
        if b == 0 and probe_keep is not None:
            m, bi = probe_keep      # the winning scale probe IS member 0
        else:
            m = lgb.LGBMClassifier(**p)
            ikw = ({"init_score": init_scale * zd_fit,
                    "eval_init_score": [init_scale * zd_es]}
                   if init_scale else {})
            m.fit(fit[cols], fit[target], sample_weight=w_fit,
                  eval_set=[(es[cols], es[target])],
                  eval_sample_weight=None if w_es is None else [w_es],
                  eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(150, verbose=False)], **ikw)
            bi = int(m.best_iteration_ or 0)
        if do_refit and bi:
            m = _refit_lgbm(lgb.LGBMClassifier, p, bi,
                            tr[cols], tr[target], w_tr,
                            init_score=(init_scale * zd_tr
                                        if init_scale else None))
        models.append(m)
        best_iters.append(bi)
    slices["lgbm"] = (pos, len(models))
    pos = len(models)
    for b in range(n_xgb):          # family members join AFTER the LGBM bag
        m = F.InfSafe(xgb_lib.XGBClassifier(**XGB_CLS, random_state=b))
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              sample_weight_eval_set=None if w_es is None else [w_es],
              eval_set=[(es[cols], es[target])], verbose=False)
        models.append(m)
        best_iters.append(int(m.best_iteration_ or 0))
    if n_xgb:
        slices["xgb"] = (pos, len(models))
        pos = len(models)
    cat_here = [c for c in cols if c in F.CAT_COLS]
    for b in range(n_cb):
        # CatBoost: train rows weighted; the ES eval stays unweighted (a
        # weighted eval needs a Pool, which would bypass CatSafe's cleaning)
        m = F.CatSafe(CatBoostClassifier(**CB_CLS, random_seed=b,
                                         cat_features=cat_here), cat_here)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])])
        bi = int(m.best_iteration_ or 0)
        if do_refit and bi:
            m = _refit_cb(CatBoostClassifier, dict(CB_CLS, _seed=b), bi,
                          tr[cols], tr[target], w_tr, cat_here)
        models.append(m)
        best_iters.append(bi)
    if n_cb:
        slices["cb"] = (pos, len(models))
    model = F.MeanBag(models) if len(models) > 1 else models[0]

    # diverse second learner (the LR has no early stopping, so it may use
    # the full training slice); ridge strength picked per head on cal year
    lr, num_cols, lr_C = _fit_logistic(tr, cols, target, w=w_tr, ca=ca)

    # per-family cal design matrix [fam logits..., LR logit] — the fit-side
    # twin of predict.predict_prop's serving arithmetic (features.family_logits)
    fam_order = list(slices)
    yca = ca[target].to_numpy()
    zf_cal = F.family_logits(models, slices, ca[cols])
    if init_scale:      # B1: residual members' family logit + donor offset
        zf_cal["lgbm"] = zf_cal["lgbm"] + init_scale * zd_ca
    zl_cal = F.logit(lr.predict_proba(ca[num_cols])[:, 1])
    Z_cal = np.column_stack([zf_cal[f] for f in fam_order] + [zl_cal])

    # multi-year calibration support: stash this suite's cal-year design for
    # the next suite; pool the prior year's when available (selection suite
    # feeds shipping at zero extra train cost)
    cur = {"Z": Z_cal, "y": yca, "gamepk": ca["GamePk"].to_numpy(),
           "dates": ca["Date"].to_numpy()}
    _CAL_STASH[("prop", head_key, cal_yr)] = cur
    priors = _stash_priors("prop", head_key, train_yrs, cur, mat="Z")
    if priors:
        pool = _pool_cal(cur, priors)
    else:
        pool = dict(cur, w=None)
    cal_years = _pool_years(priors, cal_yr, train_yrs)

    n_fam = np.array([slices[f][1] - slices[f][0] for f in fam_order], float)
    best_w, fstack = None, None
    if FAMILY_STACK:
        fstack = LogisticRegression(C=FSTACK_C, max_iter=1000)
        fstack.fit(pool["Z"], pool["y"], sample_weight=pool["w"])
        s_pool = fstack.predict_proba(pool["Z"])[:, 1]
    else:
        # incumbent 21-point grid (rollback path); the member-count-weighted
        # mean of family logits == MeanBag's pooled logit mean
        zg = pool["Z"][:, :-1] @ (n_fam / n_fam.sum())
        g_pool = 1.0 / (1.0 + np.exp(-zg))
        l_pool = 1.0 / (1.0 + np.exp(-pool["Z"][:, -1]))
        best_w, best_ll = 1.0, np.inf
        for w in np.linspace(0.0, 1.0, 21):
            ll = log_loss(pool["y"],
                          np.clip(F.blend(g_pool, l_pool, w, BLEND_SPACE),
                                  1e-6, 1 - 1e-6), sample_weight=pool["w"])
            if ll < best_ll:
                best_ll, best_w = ll, w
        s_pool = F.blend(g_pool, l_pool, best_w, BLEND_SPACE)

    iso, cal_kind = _pick_calibrator(s_pool, pool["y"], pool["gamepk"], name,
                                     dates=pool["dates"], w=pool["w"])

    zf_te = F.family_logits(models, slices, te[cols])
    if init_scale:      # B1 twin of the cal-design offset
        zf_te["lgbm"] = zf_te["lgbm"] + init_scale * zd_te
    zl_te = F.logit(lr.predict_proba(te[num_cols])[:, 1])
    Z_te = np.column_stack([zf_te[f] for f in fam_order] + [zl_te])
    # test-year design stashed too (2026-07-15 PM): with cal+test designs
    # persisted (cal_stash.joblib), the whole pricing layer (stack C, pool
    # decay, calibrator bagging/kind) is re-fittable and SCOREABLE offline
    # by Model/cal_lab.py — no booster retrains. Lab discipline: iterate on
    # the SELECTION suite's test year only; 2026 stays confirm-only.
    _CAL_STASH[("prop_te", head_key, test_yr)] = {
        "Z": Z_te, "y": te[target].to_numpy(),
        "dates": te["Date"].to_numpy()}
    if FAMILY_STACK:
        s_te = fstack.predict_proba(Z_te)[:, 1]
    else:
        zg_te = Z_te[:, :-1] @ (n_fam / n_fam.sum())
        s_te = F.blend(1.0 / (1.0 + np.exp(-zg_te)),
                       1.0 / (1.0 + np.exp(-Z_te[:, -1])), best_w, BLEND_SPACE)
    p_te = iso.predict(s_te)
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": best_iters[0],
        "families": {f: int(slices[f][1] - slices[f][0]) for f in fam_order},
        "es_refit": bool(do_refit),
        "recency_decay": decay,
        "lr_C": lr_C,
        "cal_pool_years": cal_years,
        "blend_space": BLEND_SPACE,
        "calibrator": cal_kind,
        "auc": float(roc_auc_score(y, p_te)),
        "acc": float(((p_te >= 0.5).astype(float) == y).mean()),
        "logloss": float(log_loss(y, p_te)),
        "logloss_baserate": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p_te)),
        "brier_baserate": float(brier_score_loss(y, base)),
    }
    if FAMILY_STACK:
        coefs = dict(zip(fam_order + ["lr"],
                         [round(float(c), 3) for c in fstack.coef_[0]]))
        coefs["intercept"] = round(float(fstack.intercept_[0]), 3)
        metrics["fstack_coefs"] = coefs
        wt_str = " ".join(f"{k}:{v:+.2f}" for k, v in coefs.items()
                          if k != "intercept")
    else:
        metrics["blend_gbm_weight"] = round(float(best_w), 2)
        wt_str = f"gbm wt {best_w:.2f}"
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
    # B1 donor bookkeeping: recipients record the pick (scale 0.0 = the
    # grid chose the incumbent — self-gate visible in metrics); donor heads
    # stash their LGBM bag-mean logits for the recipients that follow them
    # in the PROPS loop. z_ca/z_te are free (the pure LGBM family logits
    # just computed — donors are never offset themselves, the module-level
    # assert forbids chaining); z_tr costs one bag predict on the train
    # rows, donors-with-recipients only.
    init_str = ""
    if donor is not None:
        metrics["init_donor"] = {"donor": donor_key, "scale": init_scale}
        init_str = f" | init {donor_key}@{init_scale:g}"
    if _KEEP_TRAIN and head_key in set(INIT_SCORE_DONORS.values()):
        lo, hi = slices["lgbm"]
        lgbm_members = models[lo:hi]
        z_tr_d = np.mean([F.logit(m.predict_proba(tr[cols])[:, 1], lo=1e-7)
                          for m in lgbm_members], axis=0)
        _DONOR_STASH[head_key] = {
            "models": lgbm_members, "cols": list(cols),
            "z_tr": pd.Series(z_tr_d, index=tr.index),
            "z_ca": zf_cal["lgbm"], "z_te": zf_te["lgbm"]}
    log(f"{name} [{test_yr}]: AUC {metrics['auc']:.4f} | "
        f"logloss {metrics['logloss']:.4f} (base {metrics['logloss_baserate']:.4f}) | "
        f"brier {metrics['brier']:.4f} (base {metrics['brier_baserate']:.4f}) | "
        f"top10/day {metrics['top10_daily_hit_rate']:.3f} vs base "
        f"{metrics['base_rate']:.3f} | {wt_str} | cal {cal_kind} | "
        f"cal-yrs {cal_years}{init_str}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols,
            "iso": iso, "blend_space": BLEND_SPACE}
    if FAMILY_STACK:
        prop.update({"fstack": fstack, "fstack_fams": fam_order,
                     "fam_slices": slices})
    else:
        prop["w"] = best_w
    if init_scale:      # serve-side twin data: predict.predict_prop re-adds
        prop["init_donor"] = {"head": donor_key, "scale": init_scale,
                              "models": donor["models"],
                              "cols": donor["cols"]}
    return prop, metrics


def _mirror_win(df):
    """Home/away-mirrored copy of winner-frame rows (WINNER_MIRROR): paired
    away_*/home_* columns exchanged, d_* (home-minus-away) diffs negated,
    elo_prob_home complemented, y_home_win inverted, persp_home -> 0 so the
    model can keep home-field advantage on the flag. TRAIN rows only —
    cal/test rows are never mirrored and serving always sends persp_home=1
    (predict.predict_win)."""
    out = df.copy()
    for c in df.columns:
        if c.startswith("home_"):
            twin = "away_" + c[5:]
            if twin in df.columns:
                out[c] = df[twin].to_numpy()
                out[twin] = df[c].to_numpy()
        elif c.startswith("d_"):
            out[c] = -df[c]
    if "elo_prob_home" in df.columns:
        out["elo_prob_home"] = 1.0 - df["elo_prob_home"]
    out["y_home_win"] = 1 - df["y_home_win"]
    out["persp_home"] = 0.0
    return out


def fit_winner(wf, cols, target, mu_map, train_yrs, cal_yr, test_yr, name):
    """Home-win model: small-capacity GBM bag + logistic + (when trained
    with one) the runs-model Poisson win probability, combined by the
    per-family logistic stack (FAMILY_STACK; the legacy path is the old
    two-stage 21-point grid), then calibration. Everything downstream of
    the boosters is chosen on the calibration support (multi-year pooled
    when the stash has the prior year).

    mu_map: per-GamePk expected runs (mu_away, mu_home) from the runs
    model, or None to skip the Poisson component. The runs model trains on
    2020-2024, so pass None whenever the calibration year falls inside that
    range: its in-sample predictions look falsely sharp there, the blend
    collapses onto them, and the calibrator miscalibrates (this corrupted
    the first 2025 backtest — cal 2024 is training data for the runs model).

    WINNER_MIRROR doubles the ~10k TRAIN rows with home/away-swapped copies
    (_mirror_win) — the winner is the board's most data-starved head by 20x
    and the one the sweeps flag as most over-regularization-prone."""
    from predict import poisson_win
    decay = _decay_for("winner")
    tr = wf[wf["Season"].isin(train_yrs)]
    ca = wf[wf["Season"] == cal_yr]
    te = wf[wf["Season"] == test_yr]
    fit, es = _es_split(tr)         # cal year never picks iteration counts
    if WINNER_MIRROR:
        fit = pd.concat([fit, _mirror_win(fit)], ignore_index=True)
        es = pd.concat([es, _mirror_win(es)], ignore_index=True)
        tr = pd.concat([tr, _mirror_win(tr)], ignore_index=True)
    w_fit, w_es = (_recency_w(fit, cal_yr, decay),
                   _recency_w(es, cal_yr, decay))
    w_tr = _recency_w(tr, cal_yr, decay)
    do_refit = ES_REFIT and _KEEP_TRAIN
    members, best_iters = [], []
    slices, pos = {}, 0
    for b in range(LGBM_BAGS):      # bag 0 = the pre-bagging incumbent seed
        p = dict(LGB_WIN)
        if b:
            p["random_state"] = b
            if b <= BAG_DIVERSIFY:
                p["extra_trees"] = True
        m = lgb.LGBMClassifier(**p)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])],
              eval_sample_weight=None if w_es is None else [w_es],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        bi = int(m.best_iteration_ or 0)
        if do_refit and bi:
            m = _refit_lgbm(lgb.LGBMClassifier, p, bi,
                            tr[cols], tr[target], w_tr)
        members.append(m)
        best_iters.append(bi)
    slices["lgbm"] = (pos, len(members))
    pos = len(members)
    for b in range(XGB_BAGS):       # family members join AFTER the incumbent
        m = F.InfSafe(xgb_lib.XGBClassifier(**XGB_WIN, random_state=b))
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              sample_weight_eval_set=None if w_es is None else [w_es],
              eval_set=[(es[cols], es[target])], verbose=False)
        members.append(m)
        best_iters.append(int(m.best_iteration_ or 0))
    if XGB_BAGS:
        slices["xgb"] = (pos, len(members))
        pos = len(members)
    cat_here = [c for c in cols if c in F.CAT_COLS]
    for b in range(CB_BAGS):
        # CatBoost: train rows weighted; ES eval unweighted (Pool would
        # bypass CatSafe's cleaning) — same note as fit_classifier
        m = F.CatSafe(CatBoostClassifier(**CB_WIN, random_seed=b,
                                         cat_features=cat_here), cat_here)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])])
        bi = int(m.best_iteration_ or 0)
        if do_refit and bi:
            m = _refit_cb(CatBoostClassifier, dict(CB_WIN, _seed=b), bi,
                          tr[cols], tr[target], w_tr, cat_here)
        members.append(m)
        best_iters.append(bi)
    if CB_BAGS:
        slices["cb"] = (pos, len(members))
    model = F.MeanBag(members) if len(members) > 1 else members[0]
    lr, num_cols, lr_C = _fit_logistic(tr, cols, target, w=w_tr, ca=ca)
    fam_order = list(slices)
    n_fam = np.array([slices[f][1] - slices[f][0] for f in fam_order], float)

    def zparts(d):
        """[fam logits..., LR logit(, Poisson logit)] design matrix — the
        fit-side twin of predict.predict_win's serving arithmetic. Rows with
        no finite Poisson prob fall back to the member-count-weighted mean
        of the family logits (the pure-GBM view)."""
        zf = F.family_logits(members, slices, d[cols])
        zcols = [zf[f] for f in fam_order]
        zcols.append(F.logit(lr.predict_proba(d[num_cols])[:, 1]))
        if mu_map is not None:
            mus = mu_map.reindex(d["GamePk"])
            pois = np.array([poisson_win(h, a) for h, a in
                             zip(mus["mu_home"], mus["mu_away"])])
            zg = np.column_stack(zcols[:len(fam_order)]) @ (n_fam / n_fam.sum())
            with np.errstate(invalid="ignore"):
                zp = np.where(np.isfinite(pois),
                              F.logit(np.nan_to_num(pois, nan=0.5)), zg)
            zcols.append(zp)
        return np.column_stack(zcols)

    yca = ca[target].to_numpy()
    Z_cal = zparts(ca)
    cur = {"Z": Z_cal, "y": yca, "gamepk": ca["GamePk"].to_numpy(),
           "dates": ca["Date"].to_numpy()}
    _CAL_STASH[("winner", "winner", cal_yr)] = cur
    priors = _stash_priors("winner", "winner", train_yrs, cur, mat="Z")
    if priors:
        pool = _pool_cal(cur, priors)
    else:
        pool = dict(cur, w=None)
    cal_years = _pool_years(priors, cal_yr, train_yrs)

    w1, w_ml, fstack = 1.0, 1.0, None
    n_z = len(fam_order)            # fam cols; then lr; then optional pois
    if FAMILY_STACK:
        fstack = LogisticRegression(C=FSTACK_C, max_iter=1000)
        fstack.fit(pool["Z"], pool["y"], sample_weight=pool["w"])
        s_pool = fstack.predict_proba(pool["Z"])[:, 1]
    else:
        def pick_w(a, b):
            best_w, best_ll = 1.0, np.inf
            for w in np.linspace(0.0, 1.0, 21):
                ll = log_loss(pool["y"],
                              np.clip(F.blend(a, b, w, BLEND_SPACE),
                                      1e-6, 1 - 1e-6),
                              sample_weight=pool["w"])
                if ll < best_ll:
                    best_ll, best_w = ll, w
            return best_w

        zg = pool["Z"][:, :n_z] @ (n_fam / n_fam.sum())
        g_pool = 1.0 / (1.0 + np.exp(-zg))
        l_pool = 1.0 / (1.0 + np.exp(-pool["Z"][:, n_z]))
        w1 = pick_w(g_pool, l_pool)
        s_pool = F.blend(g_pool, l_pool, w1, BLEND_SPACE)
        if mu_map is not None:
            pois_pool = 1.0 / (1.0 + np.exp(-pool["Z"][:, n_z + 1]))
            w_ml = pick_w(s_pool, pois_pool)
            s_pool = F.blend(s_pool, pois_pool, w_ml, BLEND_SPACE)

    iso, cal_kind = _pick_calibrator(s_pool, pool["y"], pool["gamepk"],
                                     "winner", dates=pool["dates"],
                                     w=pool["w"])

    Z_te = zparts(te)
    _CAL_STASH[("winner_te", "winner", test_yr)] = {
        "Z": Z_te, "y": te[target].to_numpy(),
        "dates": te["Date"].to_numpy()}
    if FAMILY_STACK:
        s_te = fstack.predict_proba(Z_te)[:, 1]
    else:
        zg_te = Z_te[:, :n_z] @ (n_fam / n_fam.sum())
        s_te = F.blend(1.0 / (1.0 + np.exp(-zg_te)),
                       1.0 / (1.0 + np.exp(-Z_te[:, n_z])), w1, BLEND_SPACE)
        if mu_map is not None:
            s_te = F.blend(s_te, 1.0 / (1.0 + np.exp(-Z_te[:, n_z + 1])),
                           w_ml, BLEND_SPACE)
    p_te = iso.predict(s_te)
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": best_iters[0],
        "families": {f: int(slices[f][1] - slices[f][0]) for f in fam_order},
        "es_refit": bool(do_refit),
        "winner_mirror": bool(WINNER_MIRROR),
        "recency_decay": decay,
        "lr_C": lr_C,
        "cal_pool_years": cal_years,
        "blend_space": BLEND_SPACE,
        "calibrator": cal_kind,
        "auc": float(roc_auc_score(y, p_te)),
        "acc": float(((p_te >= 0.5).astype(float) == y).mean()),
        "logloss": float(log_loss(y, p_te)),
        "logloss_baserate": float(log_loss(y, base)),
        "brier": float(brier_score_loss(y, p_te)),
        "brier_baserate": float(brier_score_loss(y, base)),
    }
    if FAMILY_STACK:
        zn = fam_order + ["lr"] + (["pois"] if mu_map is not None else [])
        coefs = dict(zip(zn, [round(float(c), 3) for c in fstack.coef_[0]]))
        coefs["intercept"] = round(float(fstack.intercept_[0]), 3)
        metrics["fstack_coefs"] = coefs
        wt_str = " ".join(f"{k}:{v:+.2f}" for k, v in coefs.items()
                          if k != "intercept")
    else:
        metrics["blend_gbm_weight"] = round(float(w1), 2)
        metrics["blend_ml_weight"] = round(float(w_ml), 2)
        wt_str = f"gbm wt {w1:.2f} | ML-vs-poisson wt {w_ml:.2f}"
    log(f"{name} [{test_yr}]: AUC {metrics['auc']:.4f} | acc "
        f"{metrics['acc']:.3f} | logloss {metrics['logloss']:.4f} "
        f"(base {metrics['logloss_baserate']:.4f}) | {wt_str}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols,
            "iso": iso, "blend_space": BLEND_SPACE}
    if FAMILY_STACK:
        prop.update({"fstack": fstack, "fstack_fams": fam_order,
                     "fam_slices": slices,
                     "fstack_pois": mu_map is not None})
    else:
        prop.update({"w": w1, "w_ml": w_ml})
    return prop, metrics


def fit_poisson(df, cols, target, train_yrs, cal_yr, test_yr, name, baseline,
                n_bags=1, tweedie_power=None, n_xgb=0, n_cb=0, params=None,
                head_key=None):
    """Poisson (default) count regression, or Tweedie when tweedie_power is set
    (a compound Poisson-Gamma objective, variance power in (1,2)). Tweedie lets
    the MEAN model an over-dispersed right tail directly — total bases / H+R+RBI
    run ~2x Poisson variance — instead of leaning entirely on the post-hoc
    cal-year dispersion. Serving is unchanged: .predict() still returns E[y].

    With 2 families present and FAMILY_STACK on, the families' cal-year means
    are combined by a deviance-chosen weight (features.FamilyBlendBag — the
    count analog of the binary per-family stack; deviance is the proper score
    the per-line calibrators ride on). Weight support pools the prior year
    when the _CAL_STASH has it."""
    head_key = head_key or name.lower()
    decay = _decay_for(head_key)
    tr = df[df["Season"].isin(train_yrs)]
    if _SUPERSET_SAMPLE and len(tr) > _SUPERSET_SAMPLE:   # superset train only
        tr = tr.sample(n=_SUPERSET_SAMPLE, random_state=0)
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr].copy()
    fit, es = _es_split(tr)         # cal year never picks iteration counts
    w_fit, w_es = (_recency_w(fit, cal_yr, decay),
                   _recency_w(es, cal_yr, decay))
    w_tr = _recency_w(tr, cal_yr, decay)
    do_refit = ES_REFIT and _KEEP_TRAIN
    tweedie = tweedie_power is not None
    models, best_iters = [], []
    slices, pos = {}, 0
    for b in range(n_bags):
        p = dict(params or LGB_POIS)    # COUNT_PARAMS override or the base
        if tweedie:
            p = dict(p, objective="tweedie",
                     tweedie_variance_power=tweedie_power)
        if b:                       # bag 0 = the incumbent default seed
            p["random_state"] = b
            if b <= BAG_DIVERSIFY:
                p["extra_trees"] = True
        m = lgb.LGBMRegressor(**p)
        m.fit(fit[cols], fit[target], sample_weight=w_fit,
              eval_set=[(es[cols], es[target])],
              eval_sample_weight=None if w_es is None else [w_es],
              eval_metric=("tweedie" if tweedie else "poisson"),
              callbacks=[lgb.early_stopping(150, verbose=False)])
        bi = int(m.best_iteration_ or 0)
        if do_refit and bi:
            m = _refit_lgbm(lgb.LGBMRegressor, p, bi,
                            tr[cols], tr[target], w_tr)
        models.append(m)
        best_iters.append(bi)
    slices["lgbm"] = (pos, len(models))
    pos = len(models)
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
        best_iters.append(int(m.best_iteration_ or 0))
    if n_xgb:
        slices["xgb"] = (pos, len(models))
        pos = len(models)
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
        bi = int(m.best_iteration_ or 0)
        if do_refit and bi:
            m = _refit_cb(CatBoostRegressor, dict(p, _seed=b), bi,
                          tr[cols], tr[target], w_tr, cat_here,
                          exponent=True)
        models.append(m)
        best_iters.append(bi)
    if n_cb:
        slices["cb"] = (pos, len(models))
    fam_order = list(slices)
    model = F.MeanBag(models) if len(models) > 1 else models[0]
    fam_w = None
    if FAMILY_STACK and len(fam_order) == 2:
        # per-family cal-year means -> deviance-chosen weight, prior year
        # pooled when stashed. >2 families would need a simplex search;
        # the 2-family grid covers the current roster (lgbm + cb).
        mus_cal = np.column_stack(
            [np.mean([models[i].predict(ca[cols])
                      for i in range(a, b)], axis=0)
             for f, (a, b) in slices.items()])
        cur = {"mus": mus_cal, "y": ca[target].to_numpy()}
        _CAL_STASH[("count", head_key, cal_yr)] = cur
        priors = _stash_priors("count", head_key, train_yrs, cur, mat="mus")
        if priors:
            pool = _pool_cal(cur, priors)
        else:
            pool = dict(cur, w=None)
        y_pool = np.asarray(pool["y"], float)
        best_w, best_dev = 1.0, np.inf
        for w in np.linspace(0.0, 1.0, 21):
            mu = np.clip(w * pool["mus"][:, 0] + (1 - w) * pool["mus"][:, 1],
                         1e-6, None)
            if tweedie:
                dev = mean_tweedie_deviance(y_pool, mu, power=tweedie_power,
                                            sample_weight=pool["w"])
            else:
                dev = mean_poisson_deviance(y_pool, mu,
                                            sample_weight=pool["w"])
            if dev < best_dev:
                best_dev, best_w = dev, w
        fam_w = {fam_order[0]: round(float(best_w), 3),
                 fam_order[1]: round(float(1 - best_w), 3)}
        model = F.FamilyBlendBag(models, slices, fam_w)
    pred = model.predict(te[cols])
    y = te[target].to_numpy()
    bl = baseline(te)
    metrics = {
        "n_train": len(tr), "n_test": len(te),
        "best_iter": best_iters[0],
        "families": {f: int(slices[f][1] - slices[f][0]) for f in fam_order},
        "es_refit": bool(do_refit),
        "recency_decay": decay,
        "mae": float(mean_absolute_error(y, pred)),
        "mae_baseline": float(mean_absolute_error(y, bl)),
        "mean_actual": float(y.mean()), "mean_pred": float(pred.mean()),
    }
    if fam_w is not None:
        metrics["fam_w"] = fam_w
    log(f"{name} [{test_yr}]: MAE {metrics['mae']:.3f} "
        f"(baseline {metrics['mae_baseline']:.3f}) | "
        f"mean pred {metrics['mean_pred']:.2f} vs actual "
        f"{metrics['mean_actual']:.2f}"
        + (f" | fam_w {fam_w}" if fam_w is not None else ""))
    return model, metrics


def naive_hr_baseline(te, slot_pa, league_hr_pa):
    """P(HR) if you only used season HR/PA and lineup slot."""
    rate = te["s_hr_pa"].fillna(league_hr_pa).clip(0, 0.15)
    exp_pa = te["slot"].map(slot_pa).fillna(4.1)
    return 1 - (1 - rate) ** exp_pa


def fit_line_cals(mu_cal, y_cal, lines, w=None, dates=None):
    """Per-line logistic calibrators on the CAL support: P(over line) as a
    direct monotone 2-parameter function of mu. One shared implementation
    for every count-family (count heads, starter K, game total) so the
    pricing mechanism is identical across the whole line surface. Degenerate
    lines (single-class cal support) are skipped — consumers fall back to
    nb_over. w = optional sample weights (multi-year pooled support).
    dates + CAL_BAG_B > 0 = day-block bootstrap bagging (features.
    BaggedLineCal), the count-line analog of the binary heads' BaggedCal;
    dates=None keeps the single full-support fit."""
    mu_cal = np.asarray(mu_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)

    def _fit(mu, over, sw):
        return LogisticRegression(C=1e6, max_iter=1000).fit(
            mu.reshape(-1, 1), over, sample_weight=sw)

    out = {}
    for line in lines:
        over = (y_cal > line).astype(int)
        if not 0 < over.mean() < 1:
            continue
        if dates is None or not CAL_BAG_B:
            out[line] = _fit(mu_cal, over, w)
            continue
        days = np.unique(np.asarray(dates))
        idx_of_day = {d: np.flatnonzero(np.asarray(dates) == d)
                      for d in days}
        rng = np.random.default_rng(0)
        members = []
        for _ in range(CAL_BAG_B):
            take = np.concatenate(
                [idx_of_day[d] for d in
                 rng.choice(days, size=len(days), replace=True)])
            ob = over[take]
            if not 0 < ob.mean() < 1:
                continue
            members.append(_fit(mu_cal[take], ob,
                                None if w is None else w[take]))
        out[line] = (F.BaggedLineCal(members) if members
                     else _fit(mu_cal, over, w))
    return out


def _disp(y, mu, w=None):
    """(Possibly weighted) variance-to-mean dispersion factor."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if w is None:
        return float(np.mean((y - mu) ** 2) / np.mean(mu))
    w = np.asarray(w, dtype=float)
    return float(np.sum(w * (y - mu) ** 2) / np.sum(w * mu))


def _pooled_lines(head_key, mu_cal, y_cal, cal_yr, train_yrs_for_pool,
                  dates=None):
    """(mu, y, w, dates) line-calibrator/dispersion support for one count
    family: stashes this suite's cal-year (mu, y, dates), pools up to
    CAL_POOL_YEARS-1 stashed prior years (selection suite feeds shipping;
    a --prestash chain reaches one deeper), prior year k back discounted
    CAL_POOL_DECAY**k. w=None when nothing pooled. dates ride along for
    fit_line_cals' day-block bagging."""
    cur = {"mu": np.asarray(mu_cal, dtype=float),
           "y": np.asarray(y_cal, dtype=float),
           "dates": np.asarray(dates) if dates is not None
           else np.zeros(len(np.asarray(mu_cal)))}
    _CAL_STASH[("lines", head_key, cal_yr)] = cur
    priors = _stash_priors("lines", head_key, train_yrs_for_pool, cur)
    if not priors:
        return cur["mu"], cur["y"], None, (dates if dates is not None
                                           else None)
    pool = _pool_cal(cur, priors)
    return pool["mu"], pool["y"], pool["w"], (pool["dates"]
                                              if dates is not None else None)


def train_suite(bf, sf, tg, wf, cat_levels, train_yrs, cal_yr, test_yr):
    """Fit the full model suite (every PROPS binary head, the count heads,
    starter K, team runs, winner) on one train/cal/test split. Returns (artifacts, metrics) with the same
    artifact keys regardless of split, so evaluate_deep can score either the
    shipping suite or the selection suite identically."""
    # shadow columns (1F) join every SUPERSET train's contracts; a keep
    # train ships shadow-free END-TO-END — the per-head lists via the
    # shadow-free keep-lists, and the frame-prep contracts here too.
    # 2026-07-15: the audit-#8 serving guard rightly refused the first
    # post-audit keep-train because bat_cols still carried the planted
    # shadow names (the old F3 exemption predates the loader guard);
    # serving never reads shadow values, so a keep artifact must not
    # list them anywhere.
    _shadow = (lambda f: []) if _KEEP_TRAIN else shadow_cols_of
    bat_cols = F.batter_feature_cols() + _shadow(bf)
    st_cols = F.starts_feature_cols() + _shadow(sf)
    tg_cols = _apply_keep("total",
                          F.team_game_feature_cols() + _shadow(tg))
    metrics, props = {}, {}
    _DONOR_STASH.clear()    # B1 donors are per-suite (tr/cal rows differ)

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
                                 n_xgb=XGB_BAGS, n_cb=CB_BAGS,
                                 head_key=name)
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
                             n_xgb=XGB_BAGS, n_cb=CB_BAGS,
                             params=COUNT_PARAMS.get("k"), head_key="k")
    metrics[f"k_{test_yr}"] = m

    # Starter-K dispersion on the CALIBRATION support (never the holdout):
    # real K counts run a touch over Poisson variance, so predict.py prices
    # K P(over) with a negative binomial (nb_over) using this factor.
    # 2026-07-15 PM: support is multi-year pooled when the stash has the
    # prior year (_pooled_lines), like every line calibrator below.
    sf_cal = sf[sf["Season"] == cal_yr]
    kp_cal = k_model.predict(sf_cal[k_cols])
    mu_k, y_k, w_k, d_k = _pooled_lines("k", kp_cal,
                                        sf_cal["y_so"].to_numpy(),
                                        cal_yr, train_yrs,
                                        dates=sf_cal["Date"].to_numpy())
    k_disp = _disp(y_k, mu_k, w_k)
    metrics[f"k_dispersion_{cal_yr}"] = k_disp
    log(f"starter-K dispersion ({cal_yr} cal support, "
        f"{'pooled' if w_k is not None else 'single-year'}): {k_disp:.2f} "
        f"(Poisson assumes 1.00)")

    # K per-line calibrators (2026-07-13 full-surface calibration pass): K
    # lines were the only starter family still priced by the raw NB/Poisson
    # tail; they now get the same cal-year logistic pricing as outs/pbb/pha
    # (predict.k_over consumes, NB fallback for old artifacts).
    from predict import K_LINES, TOTAL_LINES
    k_line_cals = fit_line_cals(mu_k, y_k, K_LINES, w=w_k, dates=d_k)

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
                               params=COUNT_PARAMS.get(cname),
                               tweedie_power=ch.get("tweedie"),
                               n_xgb=XGB_BAGS, n_cb=CB_BAGS,
                               head_key=cname)
        ca = frame[frame["Season"] == cal_yr]
        mu_cal = model.predict(ca[cols])
        y_cal = ca[ch["target"]].to_numpy()
        mu_p, y_p, w_p, d_p = _pooled_lines(cname, mu_cal, y_cal,
                                            cal_yr, train_yrs,
                                            dates=ca["Date"].to_numpy())
        disp = _disp(y_p, mu_p, w_p)
        m["dispersion_cal"] = round(disp, 4)
        # per-line logistic calibrators on the CAL support (the count-head
        # analog of the binary props' isotonic): P(over line) as a direct
        # monotone function of mu. Outs/batter-K counts run UNDER Poisson
        # variance (bounded by PA / the manager's hook), so nb_over — which
        # can only widen, never narrow — misprices their tails; consumers
        # fall back to nb_over only when a line has no calibrator.
        line_cals = fit_line_cals(mu_p, y_p, ch["lines"], w=w_p, dates=d_p)
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
                                     n_xgb=XGB_BAGS, n_cb=CB_BAGS,
                                     params=COUNT_PARAMS.get("total"),
                                     head_key="total")
    metrics[f"team_runs_{test_yr}"] = m

    # Game-total dispersion, also on the calibration year: real totals ran
    # ~2.3x Poisson variance, which made pure Poisson P(over) worse than the
    # base rate at low lines. predict.py switches to a negative binomial.
    tg_cal = tg[tg["Season"] == cal_yr]
    pr_cal = team_runs_model.predict(tg_cal[tg_cols])
    per_game = pd.DataFrame({"g": tg_cal["GamePk"].to_numpy(), "mu": pr_cal,
                             "y": tg_cal["y_runs"].to_numpy()}).groupby("g").sum()
    dates_pg = (tg_cal.groupby("GamePk")["Date"].first()
                .reindex(per_game.index).to_numpy())
    mu_tot, y_tot, w_tot, d_tot = _pooled_lines("total_game",
                                                per_game["mu"].to_numpy(),
                                                per_game["y"].to_numpy(),
                                                cal_yr, train_yrs,
                                                dates=dates_pg)
    total_disp = _disp(y_tot, mu_tot, w_tot)
    metrics[f"total_dispersion_{cal_yr}"] = total_disp
    log(f"game-total dispersion ({cal_yr} cal support): {total_disp:.2f} "
        f"(Poisson assumes 1.00)")

    # total-runs per-line calibrators (same 2026-07-13 pass): the raw NB
    # tail left the total lines the worst-calibrated family on the board
    # (slopes .84-.95); per-game cal mu vs actual totals, predict.
    # total_over consumes with NB fallback for exotic odds-store lines.
    total_line_cals = fit_line_cals(mu_tot, y_tot, TOTAL_LINES, w=w_tot,
                                    dates=d_tot)

    # H5 team_total head-ification (2026-07-14): the per-TEAM line surface
    # off the same runs model. TEAM-level cal-year NB dispersion (the game
    # total's ~2.3 does NOT transfer — team variance is its own number) +
    # per-line calibrators for the team-total lines the books post.
    from predict import TEAM_TOTAL_LINES
    mu_tt, y_tt, w_tt, d_tt = _pooled_lines("team_total", pr_cal,
                                            tg_cal["y_runs"].to_numpy(),
                                            cal_yr, train_yrs,
                                            dates=tg_cal["Date"].to_numpy())
    team_total_disp = _disp(y_tt, mu_tt, w_tt)
    metrics[f"team_total_dispersion_{cal_yr}"] = round(team_total_disp, 4)
    log(f"team-total dispersion ({cal_yr} cal support): "
        f"{team_total_disp:.2f} (Poisson assumes 1.00)")
    team_line_cals = fit_line_cals(mu_tt, y_tt, TEAM_TOTAL_LINES, w=w_tt,
                                   dates=d_tt)

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
    # WINNER_MIRROR's perspective flag rides OUTSIDE the keep-list (it is an
    # augmentation artifact, not a selected feature); predict.predict_win
    # pins it to 1.0 at serve time.
    if WINNER_MIRROR and "persp_home" not in win_cols:
        win_cols = win_cols + ["persp_home"]
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
            "artifact_version": 3,
            "role": "keep" if _KEEP_TRAIN else "superset",
            "bags": {"lgbm": LGBM_BAGS, "xgb": XGB_BAGS, "cb": CB_BAGS},
            "superset_sample": _SUPERSET_SAMPLE,
            "recency_decay": RECENCY_DECAY,
            "recency_head_decay": dict(RECENCY_HEAD_DECAY),
            "blend_space": BLEND_SPACE,
            "auto_cal": AUTO_CAL,
            # 2026-07-15 PM diversity/calibration batch flags
            "family_stack": FAMILY_STACK,
            "cal_bag_b": CAL_BAG_B,
            "multi_year_cal": MULTI_YEAR_CAL,
            "cal_pool_decay": CAL_POOL_DECAY,
            "es_refit": ES_REFIT and _KEEP_TRAIN,
            "winner_mirror": WINNER_MIRROR,
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
                         "decay_sweep.py; 1.0 = no recency weighting). "
                         "Clears RECENCY_HEAD_DECAY too (sweep isolation).")
    ap.add_argument("--prestash", action="store_true",
                    help="train an extra throwaway suite one season further "
                         "back FIRST, so the SELECTION suite gets multi-year "
                         "calibration support too and the 2025 paired read "
                         "adjudicates it (otherwise only the shipping suite "
                         "pools and the 2026 confirm is the multi-year "
                         "read). Costs ~one extra suite train.")
    args = ap.parse_args()
    if args.decay is not None:
        RECENCY_DECAY = args.decay
        RECENCY_HEAD_DECAY.clear()
    if RECENCY_DECAY < 1.0 or RECENCY_HEAD_DECAY:
        log(f"recency sample-weighting ON: decay {RECENCY_DECAY} per season "
            f"back from each suite's cal year"
            + (f"; per-head overrides {RECENCY_HEAD_DECAY}"
               if RECENCY_HEAD_DECAY else ""))

    # role banner (audit #8): superset runs write models_superset*.joblib and
    # never touch the serving artifacts; the temp-regime flag prints loudly so
    # a forgotten flip can't silently ship an LGBM-only ensemble.
    role = "keep" if _KEEP_TRAIN else "superset"
    log(f"run role: {role.upper()} "
        + ("(keep-list applied; writes models_bt.joblib / models.joblib)"
           if _KEEP_TRAIN else
           "(no keep-list; writes models_superset_bt.joblib / "
           "models_superset.joblib — serving artifacts untouched)"))
    log(f"families: {LGBM_BAGS} LGBM + {XGB_BAGS} XGB + {CB_BAGS} CB "
        f"(XGB retired permanently, CB restored — user 2026-07-15 PM) | "
        f"fstack {'ON' if FAMILY_STACK else 'off'} | "
        f"cal-bag {CAL_BAG_B or 'off'} | "
        f"multi-year-cal {'ON' if MULTI_YEAR_CAL else 'off'} "
        f"(pool decay {CAL_POOL_DECAY}) | "
        f"es-refit {'ON (keep-trains)' if ES_REFIT else 'off'} | "
        f"winner-mirror {'ON' if WINNER_MIRROR else 'off'}")

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
    # WINNER_MIRROR perspective flag: 1 = the real home orientation (every
    # served row), 0 = a train-time mirrored copy (_mirror_win flips it)
    wf["persp_home"] = 1.0

    # season splits derived from the data — no code edit at the annual
    # rollover; the holdout promotes itself once the new season has games
    train_yrs, cal_yr, hold_yr = suite_years(bf)
    sel_tr, sel_cal, sel_te = train_yrs[:-1], train_yrs[-1], cal_yr

    # -- optional prestash suite (multi-year cal for the SELECTION suite):
    # one season further back again, trained ONLY to leave its cal-year
    # scores in _CAL_STASH; nothing is written. Without it the selection
    # suite prices on single-year support (its paired read still verdicts
    # every non-pooling change) and only the shipping suite pools. --
    if args.prestash and MULTI_YEAR_CAL:
        ps_tr, ps_cal, ps_te = sel_tr[:-1], sel_tr[-1], sel_cal
        if len(ps_tr) >= 2:
            log(f"=== PRESTASH suite (train<={ps_tr[-1]}, cal {ps_cal}, "
                f"test {ps_te}) — throwaway, feeds the selection suite's "
                f"multi-year calibration ===")
            train_suite(bf, sf, tg, wf, cat_levels, ps_tr, ps_cal, ps_te)
        else:
            log("--prestash skipped: not enough seasons for a third suite")

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

    # Calibration-stash sidecar (2026-07-15 PM): persist every suite's cal
    # AND test pricing designs so Model/cal_lab.py can re-fit the whole
    # pricing layer (stack C, pool decay, calibrator bag/kind) offline —
    # zero booster retrains. Keep-trains only; regenerated every chain.
    if _KEEP_TRAIN:
        stash_path = ART / "cal_stash.joblib"
        joblib.dump({
            "stash": _CAL_STASH,
            "suites": {
                "selection": {"train_last": int(sel_tr[-1]),
                              "train_prev": int(sel_tr[-2]),
                              "cal": int(sel_cal), "test": int(sel_te)},
                "shipping": {"train_last": int(train_yrs[-1]),
                             "train_prev": int(train_yrs[-2]),
                             "cal": int(cal_yr), "test": int(hold_yr)},
            },
            "flags": {"family_stack": FAMILY_STACK, "fstack_c": FSTACK_C,
                      "cal_bag_b": CAL_BAG_B, "multi_year_cal": MULTI_YEAR_CAL,
                      "cal_pool_decay": CAL_POOL_DECAY, "auto_cal": AUTO_CAL},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, stash_path, compress=3)
        log(f"calibration stash sidecar -> {stash_path.name} "
            f"({len(_CAL_STASH)} entries)")

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
