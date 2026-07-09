"""Train the MLB prediction models.

Models (all LightGBM):
  batter props (binary + isotonic calibration):
    hr, hit, hits2 (2+ hits), tb2 (2+ total bases), run (run scored),
    rbi (1+ RBI), bb (walk), sb (stolen base), single (1+), double (1+),
    bk/bk2 (1+/2+ batter strikeouts), hrr2/hrr3 (2+/3+ hits+runs+RBIs)
  k     starter strikeouts in the game              Poisson regression
  count heads (starter-K pattern, mean + cal-year NB dispersion):
    xbk/xhrr (batter K and H+R+RBI means), outs, pbb, pha, per (starter
    outs / walks allowed / hits allowed / earned runs, with P(over) lines;
    per also drives a derived expected ERA in predict.py)
  runs  game total runs                             Poisson regression

Honest evaluation protocol (no leakage):
  train on every season but the newest two  ->  early-stop & calibrate on
  the next-to-newest  ->  test on the newest (e.g. 2020-2024 / 2025 / 2026).
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
LGB_WIN = dict(n_estimators=3000, learning_rate=0.02, num_leaves=15,
               min_child_samples=150, subsample=0.9, subsample_freq=1,
               colsample_bytree=0.7, reg_lambda=10.0, objective="binary",
               verbose=-1)

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
MONOTONE = {}

# Seed bagging (2026-07-08) for the weak/target props: N GBMs differing
# only in random_state, predictions averaged (features.MeanBag) before the
# LR blend + isotonic. Variance reduction, not new signal — kept under the
# strictly-not-worse standard because it also SHRINKS these props' retrain
# jitter, making every future Section-11 read on them sharper. Bag 0 keeps
# LightGBM's default seed (the incumbent model exactly).
PROP_BAGS = {"hit": 5, "tb2": 5, "run": 5, "rbi": 5, "hrr2": 5, "hrr3": 5}
COUNT_BAGS = {"xhrr": 5, "xtb": 5}

# Per-prop LightGBM overrides (2026-07-08 grid: 2 regularization configs x
# 6 bagged target props). Heavier regularization paid ONLY as calibration,
# on two props: run (heavy: ece -.0038 vs the bagged incumbent, past band)
# and hrr2 (medium: ece -.0035 past band, top10 +.0070 — and it undoes the
# hrr2_ece drift the bagging introduced). Everything else was flat-to-worse
# (hit/tb2/rbi ece all tilted UP under more regularization) — incumbent
# LGB_CLS kept there.
PROP_PARAMS = {
    "run":  dict(LGB_CLS, num_leaves=31, min_child_samples=300,
                 colsample_bytree=0.7, reg_lambda=6.0),
    "hrr2": dict(LGB_CLS, num_leaves=63, min_child_samples=160),
}

# Per-prop feature routing. The batter frame carries a SUPERSET of columns;
# each prop trains on the superset minus the groups that don't speak to it.
# The SB prop is the cautionary tale: it regressed when the platoon-split
# columns arrived (iteration 2) — models with thin true signal are the most
# sensitive to dilution, so specialized groups only reach the props they
# describe. Each prop's actual column list is saved in its artifact, so
# predict/evaluate pick it up automatically.
_SB_FEATS = ["c_sb_pa_sh", "s_sb_pa_sh", "r7_sb_pa_sh", "r15_sb_pa_sh",
             "r30_sb_pa_sh", "d_sb_pa_sh", "c_sb_succ", "psb_sb27",
             "psb_stop", "tsb_sb_g", "tsb_stop"]
_RUNRBI = ["c_r_pa_sh", "s_r_pa_sh", "c_rbi_pa_sh", "s_rbi_pa_sh"]
# Productive/unproductive outs (features.SHRINK gidp_pa/sf_pa, 2026-07-09):
# EB-shrunk career+season GIDP/PA and SF/PA rates. GIDP kills rallies -> hurts
# both the batter's run and RBI, so it reaches run AND rbi; SF is a run-cashing
# out (RBI without a hit) that says nothing about the batter scoring, so it
# reaches rbi ONLY (run excludes _SF). Every other prop excludes _PRODOUT.
_GIDP = ["c_gidp_pa_sh", "s_gidp_pa_sh"]
_SF = ["c_sf_pa_sh", "s_sf_pa_sh"]
_PRODOUT = _GIDP + _SF
# NOTE: own H+R+RBI joint-threshold history (c/s/d_hrr{2,3}_g_sh, routed
# here as _HRR_HIST to hrr2/hrr3/xhrr only, 2026-07-08) was BENCHED —
# hrr2_ece 1.5x past band, AUC/edge/top10 flat everywhere; the features
# live on in the frames + inference path (see features.HRR_SHRINK note).
# NOTE: recency form for run/rbi (rolling + decayed own R/RBI rates,
# routed here as _RUNRBI_FORM, 2026-07-08) was BENCHED — run_ece +.0063
# marginal (past band), rbi mixed; see features.RUNRBI_FORM_BENCHED.
# teammates ahead/behind. NOTE: the 90-day-decayed variants (ctx_*_d,
# 2026-07-08) were tried on run/rbi/hrr and BENCHED — 0/0/76 within noise;
# they carry ~the same information as the career rates (corr with targets
# nearly identical). Computed in both paths, out of the superset.
_CTX = ["ctx_ahead_obp", "ctx_behind_slg"]
# rbi_opp_obp (full-order / deeper-order OBP of hitters ahead) BENCHED 2026-07-09
# after three designs (0.5 & 0.7 full-order decay, 3rd-5th-ahead isolation) all
# came back flat-to-slightly-negative on rbi/run/hrr — the order beyond the 2
# men on adds nothing the model can't already infer from ctx_ahead_obp + own
# rates + team offense. Frame + serving still COMPUTE it (features.RBI_OPP_AHEAD,
# out of the superset); re-add here + to batter_feature_cols to re-enable.
_OBP = ["c_obp", "s_obp"]
_PWR = ["hrpt_score", "phrq_n", "phrq_ev_avg", "hrq_angle_avg",
        "bat_goao", "pit_goao"]              # power-quality / fly-ball
_XBH = ["c_xbh_ab", "s_xbh_ab"]
_IBB = ["c_ibb_pa"]
_VSH = ["vsh_PA", "vsh_hr_pa_sh", "vsh_tb_ab_sh", "vsh_k_pct_sh"]
_VLOC = ["vloc_PA", "vloc_hr_pa_sh", "vloc_h_pa_sh", "vloc_tb_ab_sh",
         "vloc_k_pct_sh"]
_POS = ["pos_c_share", "pos_dh_share"]
_PEN2 = ["pen_h_bf", "pen_hl_era", "pen_hl_k_bf", "pen_np_l3"]
_TLOC = ["toff_loc_hr_pa", "toff_loc_r_pg"]
_HBF = ["pc_h_bf", "ps_h_bf", "p5_h_bf"]     # starter hit suppression
# Statcast contact quality (scrape_statcast.py). Split power vs hit-type so
# the same dilution discipline applies: barrels/EV speak to power props,
# xBA/xwOBA/GB to anything needing contact, nothing to walks or steals.
_BIP_PWR = ["bip_ev", "bip_la", "bip_hh", "bip_brl",
            "bip_pull", "bip_pullair",
            "bipd_ev", "bipd_brl", "bipd_pullair",
            "pbip_ev", "pbip_la", "pbip_hh", "pbip_brl",
            "pbipd_ev", "pbipd_brl"]
_BIP_HIT = ["bip_n", "bip_xba", "bip_xwoba", "bip_gb",
            "bipd_n", "bipd_xwoba", "bipd_gb",
            "pbip_n", "pbip_xba", "pbip_xwoba", "pbip_gb",
            "pbipd_n", "pbipd_xwoba", "pbipd_gb"]
_PLATE = ["bd_wsw_c", "bd_wsw_d", "bd_chase_c", "bd_chase_d"]
# NOTE: hand-split contact quality (bvh_*/pvh_*) was routed here as
# _VHB_PWR/_VHB_CON (2026-07-07), came back within noise on every prop and
# pushed tb2 ECE past its band; it now lives in the frames only (see the
# NOTE in features.batter_feature_cols). tsb_* (battery SB-allowed, same
# batch) stays: sb-only routing, positive tilt, no regressions.
_SPD = ["bat_sprint", "bat_hp1b"]            # raw footspeed: SB + run only
_DEF = ["opp_oaa"]                           # opponent defense: BABIP props
_PSW = ["p_swstr_d"]                         # opposing starter whiff form
_UMP = ["ump_k_pct", "ump_bb_pct"]           # HP-ump zone tendency: K/BB only
# Multi-dimensional park factors (features._attach_context): as-of per-game
# R/H/2B/TB rates at the venue — the offensive run-environment the lone HR
# park factor (park_hr_pg, on every prop) can't carry for doubles/TB/runs.
# Routed to the OFFENSIVE props only: excluded from bb, sb and the K heads
# (_BK_EXC) below, where park scoring says nothing about a batter's whiffs.
# 2026-07-09: also routed to the starter run-environment heads (outs/pha/per)
# and the team-runs model — the same whiff/walk exclusion applies there via
# k_cols (starter K) and pbb's st_exclude.
_PARK_OFF = ["park_r_pg", "park_h_pg", "park_2b_pg", "park_tb_pg"]
# Statcast bat tracking = swing quality, a fundamental skill that touches
# essentially every offensive outcome (power, contact/BABIP, whiff, and even
# walk/steal propensity indirectly). Deliberately routed to ALL batter props
# (no PROP_EXCLUDE entry) rather than pre-guessing which it helps: it is
# INERT until ~2027 (2023+ coverage vs the <=2023 selection training window),
# so broad routing costs nothing now and confounds nothing, and once it
# activates the standard eval + strictly-not-worse rule prunes any prop it
# actually dilutes (sb/bb the likeliest candidates) — empirically, not by
# prior. Kept as a named group so that pruning is a one-line exclude later.
_BAT = list(F.BAT_TRACK_COLS)

# batter strikeouts: keep only K-flavored signal (k rates, plate discipline,
# starter/bullpen whiff, arsenal) — everything else is dilution risk
_BK_EXC = (_SB_FEATS + _RUNRBI + _CTX + _OBP + _XBH + _IBB + _PWR + _HBF
           + _PEN2 + _TLOC + _BIP_PWR + _BIP_HIT + _SPD + _DEF + _PARK_OFF
           + _PRODOUT)

# Routing flips (2026-07-08), kept under the strictly-not-worse standard
# (tsb precedent): hit+footspeed (4/4 metrics tilted positive on 2025),
# run+plate-discipline (ece -.0022, top10 +.0109), rbi+Statcast-power
# (auc +.0015, ece -.0027) — all within noise but principled and harmless.
# tb2+footspeed REVERTED (ece +.0029, 0.97x band, for nothing).
# _UMP (HP-ump zone tendency) reaches ONLY the K/BB props (bb, bk, bk2, and
# the xbk head via bk's routing); every other batter prop excludes it.
# _BAT (bat tracking) appears in NO exclude list -> it reaches every batter
# prop (see the _BAT note above; inert until ~2027, pruned empirically then).
PROP_EXCLUDE = {
    "hr":    _SB_FEATS + _RUNRBI + _CTX + _OBP + _XBH + _SPD + _DEF + _UMP
             + _PRODOUT,
    # hit keeps footspeed (beat-out grounders, like single). hits2 stays speed-
    # free (2-hit game is contact, not legs — SPD tested 2026-07-09, flat).
    # tb2 GAINED footspeed 2026-07-09 (KEPT: tb2 AUC +0.0009, xtb MAE +0.0012 —
    # legs stretch singles / turn outs into extra total bases).
    "hit":   _SB_FEATS + _RUNRBI + _CTX + _OBP + _XBH + _IBB + _PWR
             + _BIP_PWR + _UMP + _PRODOUT,
    "hits2": _SB_FEATS + _RUNRBI + _CTX + _OBP + _XBH + _IBB + _PWR
             + _BIP_PWR + _SPD + _UMP + _PRODOUT,
    "tb2":   _SB_FEATS + _RUNRBI + _CTX + _OBP + _IBB + _UMP + _PRODOUT,
    # run keeps plate discipline (chase feeds OBP -> runs); GIDP suppresses
    # runs (batter erased), but SF cashes OTHERS' runs, not his -> exclude _SF
    "run":   _SB_FEATS + _PWR + _XBH + _IBB + _BIP_PWR + _UMP + _SF,
    # rbi keeps Statcast power (own HR = automatic RBI; hard contact cashes
    # runners) — box-score ISO alone lags it. (_PWR + _PLATE tested 2026-07-09,
    # both flat — reverted.) KEEPS _PRODOUT (GIDP kills RBI, SF is an RBI).
    "rbi":   _SB_FEATS + _PWR + _SPD + _PLATE + _UMP,
    # bb KEEPS _UMP (a tight zone drives walks)
    "bb":    _SB_FEATS + _RUNRBI + _CTX + _PWR + _XBH + _HBF + _PEN2 + _TLOC
             + _BIP_PWR + _BIP_HIT + _SPD + _DEF + _PARK_OFF + _PRODOUT,
    "sb":    _VSH + _RUNRBI + _CTX + _OBP + _PWR + _XBH + _IBB + _PEN2
             + _TLOC + _HBF + _VLOC + _POS + _BIP_PWR + _BIP_HIT + _DEF
             + _PLATE + _PSW + _UMP + _PARK_OFF + _PRODOUT,
    # singles = contact + footspeed (beat-out grounders), no power groups
    "single": _SB_FEATS + _RUNRBI + _CTX + _OBP + _XBH + _IBB + _PWR
              + _BIP_PWR + _UMP + _PRODOUT,
    # doubles = gap power + speed (stretching); HR-log quality stays out
    # (_PWR tested 2026-07-09, FAILED: double AUC -0.0017 — BIP_PWR already
    # carries double's power, HR-log just diluted the weakest prop).
    "double": _SB_FEATS + _RUNRBI + _CTX + _OBP + _IBB + _PWR + _UMP + _PRODOUT,
    # bk/bk2 KEEP _UMP (a generous zone drives strikeouts)
    "bk":    _BK_EXC,
    "bk2":   _BK_EXC,
    # H+R+RBI is a broad, high-base-rate target (tb2-like robustness):
    # only the steal columns clearly don't speak to it; ump is zone-only
    "hrr2":  _SB_FEATS + _UMP + _PRODOUT,
    "hrr3":  _SB_FEATS + _UMP + _PRODOUT,
}

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

# Count-style props: Poisson LGBM (starter-K pattern) + per-line logistic
# calibrators fit on the calibration year (predict.count_over). Batter heads
# exist for the MEANS (xSO, xHRR) — their half-point lines are priced by the
# calibrated binary heads above; starter heads price their own lines.
# `exclude` names the PROP_EXCLUDE entry supplying the column routing (batter
# heads); `st_exclude` drops columns from the shared starts col set (starter
# heads) — used to keep the HP-ump zone tendency on K/walks but off the
# outs/hits/earned-run heads it doesn't speak to.
COUNT_HEADS = {
    "xbk":  dict(frame="bat", target="bk_count", exclude="bk",
                 lines=[0.5, 1.5, 2.5], desc="batter strikeouts"),
    # xhrr/xtb run ~2x Poisson variance (over-dispersed); a Tweedie objective
    # (power 1.3) models that heavier tail in the mean instead of only in the
    # post-hoc dispersion. The other heads stay Poisson (outs/xbk are UNDER
    # Poisson variance — Tweedie would push the wrong way).
    "xhrr": dict(frame="bat", target="hrr_count", exclude="hrr2",
                 tweedie=1.3, lines=[1.5, 2.5, 3.5], desc="hits+runs+RBIs"),
    "xtb":  dict(frame="bat", target="tb_count", exclude="tb2",
                 tweedie=1.3, lines=[1.5, 2.5, 3.5], desc="total bases"),
    "outs": dict(frame="starts", target="y_outs", exclude=None,
                 st_exclude=_UMP,
                 lines=[14.5, 15.5, 16.5, 17.5, 18.5],
                 desc="starter outs recorded"),
    # pbb keeps _UMP (zone tendency drives walks) but drops the park run
    # environment (_PARK_OFF says nothing about walks — batter-bb precedent);
    # outs/pha/per keep _PARK_OFF (venue run/hit environment is directly
    # on-target for how deep a starter goes and what he allows).
    "pbb":  dict(frame="starts", target="y_pbb", exclude=None,
                 st_exclude=_PARK_OFF,
                 lines=[0.5, 1.5, 2.5], desc="starter walks allowed"),
    "pha":  dict(frame="starts", target="y_pha", exclude=None,
                 st_exclude=_UMP,
                 lines=[3.5, 4.5, 5.5, 6.5], desc="starter hits allowed"),
    "per":  dict(frame="starts", target="y_per", exclude=None,
                 st_exclude=_UMP,
                 lines=[1.5, 2.5, 3.5, 4.5], desc="starter earned runs"),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_categories(df, cat_levels):
    for c, levels in cat_levels.items():
        if c in df.columns:
            df[c] = pd.Categorical(df[c], categories=levels)
    return df


def _fit_logistic(tr, cols, target):
    """Regularized logistic on numeric features (categoricals dropped) — a
    learner diverse from the trees, so blending the two helps."""
    num_cols = [c for c in cols if c not in F.CAT_COLS]
    pipe = Pipeline([
        # F.inf_to_nan lives in the features module so the pickled pipeline
        # resolves it from predict.py/evaluate_deep.py too (not just __main__).
        ("clean", FunctionTransformer(F.inf_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=0.3, solver="lbfgs")),
    ])
    pipe.fit(tr[num_cols], tr[target])
    return pipe, num_cols


def fit_classifier(df, cols, target, train_yrs, cal_yr, test_yr, name,
                   params=None, n_bags=1):
    tr = df[df["Season"].isin(train_yrs)]
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr]
    models = []
    for b in range(n_bags):
        p = dict(params or LGB_CLS)
        if b:                       # bag 0 = the incumbent default seed
            p["random_state"] = b
        m = lgb.LGBMClassifier(**p)
        m.fit(tr[cols], tr[target],
              eval_set=[(ca[cols], ca[target])],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        models.append(m)
    model = F.MeanBag(models) if n_bags > 1 else models[0]

    # diverse second learner + blend weight chosen on the calibration year
    lr, num_cols = _fit_logistic(tr, cols, target)
    g_cal = model.predict_proba(ca[cols])[:, 1]
    l_cal = lr.predict_proba(ca[num_cols])[:, 1]
    yca = ca[target].to_numpy()
    best_w, best_ll = 1.0, np.inf
    for w in np.linspace(0.0, 1.0, 21):
        ll = log_loss(yca, np.clip(w * g_cal + (1 - w) * l_cal, 1e-6, 1 - 1e-6))
        if ll < best_ll:
            best_ll, best_w = ll, w

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(best_w * g_cal + (1 - best_w) * l_cal, yca)

    g_te = model.predict_proba(te[cols])[:, 1]
    l_te = lr.predict_proba(te[num_cols])[:, 1]
    p_te = iso.predict(best_w * g_te + (1 - best_w) * l_te)
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": int(model.best_iteration_ or 0),
        "blend_gbm_weight": round(float(best_w), 2),
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
        f"{metrics['base_rate']:.3f} | gbm wt {best_w:.2f}")
    prop = {"gbm": model, "lr": lr, "lr_cols": num_cols, "w": best_w, "iso": iso}
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
    model = lgb.LGBMClassifier(**LGB_WIN)
    model.fit(tr[cols], tr[target],
              eval_set=[(ca[cols], ca[target])], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    lr, num_cols = _fit_logistic(tr, cols, target)

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
            ll = log_loss(yca, np.clip(w * a + (1 - w) * b, 1e-6, 1 - 1e-6))
            if ll < best_ll:
                best_ll, best_w = ll, w
        return best_w

    g_cal, l_cal, pois_cal = parts(ca)
    w1 = pick_w(g_cal, l_cal)
    s_cal = w1 * g_cal + (1 - w1) * l_cal
    pois_cal = np.where(np.isfinite(pois_cal), pois_cal, s_cal)
    w_ml = 1.0 if mu_map is None else pick_w(s_cal, pois_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(w_ml * s_cal + (1 - w_ml) * pois_cal, yca)

    g_te, l_te, pois_te = parts(te)
    s_te = w1 * g_te + (1 - w1) * l_te
    pois_te = np.where(np.isfinite(pois_te), pois_te, s_te)
    p_te = iso.predict(w_ml * s_te + (1 - w_ml) * pois_te)
    y = te[target].to_numpy()
    base = np.full_like(p_te, tr[target].mean())
    metrics = {
        "n_train": len(tr), "n_test": len(te), "base_rate": float(y.mean()),
        "best_iter": int(model.best_iteration_ or 0),
        "blend_gbm_weight": round(float(w1), 2),
        "blend_ml_weight": round(float(w_ml), 2),
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
            "w_ml": w_ml, "iso": iso}
    return prop, metrics


def fit_poisson(df, cols, target, train_yrs, cal_yr, test_yr, name, baseline,
                n_bags=1, tweedie_power=None):
    """Poisson (default) count regression, or Tweedie when tweedie_power is set
    (a compound Poisson-Gamma objective, variance power in (1,2)). Tweedie lets
    the MEAN model an over-dispersed right tail directly — total bases / H+R+RBI
    run ~2x Poisson variance — instead of leaning entirely on the post-hoc
    cal-year dispersion. Serving is unchanged: .predict() still returns E[y]."""
    tr = df[df["Season"].isin(train_yrs)]
    ca = df[df["Season"] == cal_yr]
    te = df[df["Season"] == test_yr].copy()
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
        m.fit(tr[cols], tr[target],
              eval_set=[(ca[cols], ca[target])],
              eval_metric=("tweedie" if tweedie else "poisson"),
              callbacks=[lgb.early_stopping(150, verbose=False)])
        models.append(m)
    model = F.MeanBag(models) if n_bags > 1 else models[0]
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


def train_suite(bf, sf, tg, wf, cat_levels, train_yrs, cal_yr, test_yr):
    """Fit the full model suite (8 batter props, starter K, team runs, winner)
    on one train/cal/test split. Returns (artifacts, metrics) with the same
    artifact keys regardless of split, so evaluate_deep can score either the
    shipping suite or the selection suite identically."""
    bat_cols = F.batter_feature_cols()
    st_cols = F.starts_feature_cols()
    tg_cols = F.team_game_feature_cols()
    metrics, props = {}, {}

    for name, (target, _desc) in PROPS.items():
        cols = [c for c in bat_cols if c not in PROP_EXCLUDE.get(name, ())]
        params = PROP_PARAMS.get(name)
        mono = MONOTONE.get(name)
        if mono:    # categoricals and unlisted cols get 0 (unconstrained)
            params = dict(params or LGB_CLS,
                          monotone_constraints=[mono.get(c, 0) for c in cols],
                          monotone_constraints_method="advanced")
        prop, m = fit_classifier(bf, cols, target,
                                 train_yrs, cal_yr, test_yr, name.upper(),
                                 params=params,
                                 n_bags=PROP_BAGS.get(name, 1))
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

    # the K model keeps its whiff-only diet: the multi-dim park run
    # environment reaches outs/pha/per + the runs model, but venue scoring
    # says nothing about strikeouts (same reasoning as _BK_EXC). k_cols is
    # what the artifact ships as st_cols — the K model's serving contract.
    k_cols = [c for c in st_cols if c not in _PARK_OFF]
    k_model, m = fit_poisson(sf, k_cols, "y_so", train_yrs, cal_yr, test_yr,
                             "K", k_baseline)
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

    # count heads (starter-K pattern): Poisson mean + cal-year dispersion
    count_models = {}
    for cname, ch in COUNT_HEADS.items():
        frame = bf if ch["frame"] == "bat" else sf
        cols = ([c for c in bat_cols
                 if c not in PROP_EXCLUDE.get(ch["exclude"], ())]
                if ch["frame"] == "bat"
                else [c for c in st_cols if c not in ch.get("st_exclude", ())])
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
                               n_bags=COUNT_BAGS.get(cname, 1),
                               tweedie_power=ch.get("tweedie"))
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
        line_cals = {}
        for line in ch["lines"]:
            over = (y_cal > line).astype(int)
            if 0 < over.mean() < 1:
                line_cals[line] = LogisticRegression(
                    C=1e6, max_iter=1000).fit(mu_cal.reshape(-1, 1), over)
        metrics[f"{cname}_{test_yr}"] = m
        count_models[cname] = {"model": model, "cols": cols, "disp": disp,
                               "lines": ch["lines"], "line_cals": line_cals,
                               "frame": ch["frame"],
                               "target": ch["target"], "desc": ch["desc"]}

    def team_baseline(te):
        league = tg.loc[tg["Season"].isin(train_yrs), "y_runs"].mean()
        return te["off_r_pg"].fillna(league)

    team_runs_model, m = fit_poisson(tg, tg_cols, "y_runs", train_yrs, cal_yr,
                                     test_yr, "TEAM RUNS", team_baseline)
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

    # dedicated winner model, blended with the runs-model Poisson win prob.
    # The suite's own runs model never trains on cal_yr (it early-stops
    # there), so the mu_map is safe for the blend-weight fit — see the
    # fit_winner docstring for why cal-year-in-training is the failure mode.
    mu_all = team_runs_model.predict(tg[tg_cols])
    mu_map = (pd.DataFrame({"GamePk": tg["GamePk"].to_numpy(),
                            "Home": tg["Home"].to_numpy(), "mu": mu_all})
              .pivot_table(index="GamePk", columns="Home", values="mu")
              .rename(columns={0: "mu_away", 1: "mu_home"}))
    win_cols = F.win_feature_cols()
    win_model, m = fit_winner(wf, win_cols, "y_home_win", mu_map,
                              train_yrs, cal_yr, test_yr, "WINNER")
    win_model["cols"] = win_cols
    te = wf[wf["Season"] == test_yr]
    m["acc_home_baseline"] = float(te["y_home_win"].mean())
    log(f"WINNER [{test_yr}]: pick accuracy {m['acc']:.3f} vs always-home "
        f"{m['acc_home_baseline']:.3f}")
    metrics[f"winner_{test_yr}"] = m

    artifacts = {
        "props": props,
        "k_model": k_model, "team_runs_model": team_runs_model,
        "win_model": win_model, "total_disp": total_disp, "k_disp": k_disp,
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


# --------- routing audit: features a prop's siblings get but it doesn't --------
# The batter frame is a superset; PROP_EXCLUDE drops groups from each prop. This
# audit prints the group x prop routing and flags "sibling gaps" — a group most
# of a prop-family receives but some members exclude — so Agenda-A candidates
# surface automatically instead of by hand. It changes nothing; it just
# generates a list to run through evaluate_deep.py --paired.
FEATURE_GROUPS = {
    "SB": _SB_FEATS, "RUNRBI": _RUNRBI, "CTX": _CTX, "OBP": _OBP, "PWR": _PWR,
    "XBH": _XBH, "IBB": _IBB, "VSH": _VSH, "VLOC": _VLOC, "POS": _POS,
    "PEN2": _PEN2, "TLOC": _TLOC, "HBF": _HBF, "BIP_PWR": _BIP_PWR,
    "BIP_HIT": _BIP_HIT, "PLATE": _PLATE, "SPD": _SPD, "DEF": _DEF,
    "PSW": _PSW, "UMP": _UMP, "PARK_OFF": _PARK_OFF, "BAT": _BAT,
}

# Semantic clusters of batter props. A group some members get and others exclude
# is flagged as a testable gap. Props may sit in several families (tb2 is both a
# contact and an extra-base prop) — each family is judged on its own.
PROP_FAMILIES = {
    "contact/hits":   ["hit", "hits2", "single", "tb2", "double"],
    "extra-base":     ["hr", "tb2", "double"],
    "run-production": ["run", "rbi", "hrr2", "hrr3"],
    "K/discipline":   ["bb", "bk", "bk2"],
}

# Deliberately minimal props (dilution-sensitive — see the _SB_FEATS/_BK_EXC
# notes): a gap whose only missing members are these is by design, so the audit
# suppresses it to keep the candidate list about genuinely under-fed props.
LEAN_PROPS = {"sb", "bk", "bk2"}


def _group_status(prop, cols):
    """incl (prop trains on the whole group) / excl (none) / part (some)."""
    ex = set(PROP_EXCLUDE.get(prop, ()))
    kept = sum(c not in ex for c in cols)
    return "incl" if kept == len(cols) else "excl" if kept == 0 else "part"


def audit_routing():
    props = list(PROPS)
    print("=== Feature-group routing (batter props): ok = trains on it, "
          ". = excluded, ~ = partial ===\n")
    print("  " + "group".ljust(9) + "".join(p[:4].rjust(6) for p in props))
    mark = {"incl": "ok", "excl": ".", "part": "~"}
    for g, cols in FEATURE_GROUPS.items():
        print("  " + g.ljust(9)
              + "".join(mark[_group_status(p, cols)].rjust(6) for p in props))

    print("\n=== Sibling routing gaps: a group most of a family gets, but "
          "some members exclude ===")
    print("  (candidates to test via evaluate_deep.py --paired; nothing is "
          "applied here)\n")
    found = 0
    for fam, members in PROP_FAMILIES.items():
        for g, cols in FEATURE_GROUPS.items():
            got = [p for p in members if _group_status(p, cols) == "incl"]
            missing = [p for p in members if _group_status(p, cols) == "excl"
                       and p not in LEAN_PROPS]
            if got and missing:
                found += 1
                print(f"  [{fam:<14}] {g:<8} has: {', '.join(got):<26} "
                      f"GAP: {', '.join(missing)}")
    if not found:
        print("  (no sibling gaps — every group is all-in or all-out per family)")
    print("\n  Count heads inherit their base prop's routing (xtb<-tb2, "
          "xhrr<-hrr2, xbk<-bk), so a base-prop gap propagates to its head.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild feature frames even if cached")
    ap.add_argument("--select", action="store_true",
                    help="train ONLY the model-selection suite (one season "
                         "back from shipping) — the fast iteration loop. "
                         "The default run trains it too, then the shipping "
                         "models on top.")
    ap.add_argument("--audit-routing", action="store_true",
                    help="print the feature-group x prop routing matrix and "
                         "flag sibling gaps (Agenda-A candidates), then exit; "
                         "trains nothing")
    args = ap.parse_args()

    if args.audit_routing:
        audit_routing()
        return

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
        gf = F.build_game_frame(raw)
        log(f"game frame: {len(gf):,} rows")
        frames = {"bf": bf, "sf": sf, "gf": gf}
        joblib.dump(frames, cache, compress=3)
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]

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

    wf = gf[~gf["ShortGame"].fillna(False)].dropna(subset=["y_home_win"])
    wf = wf.sort_values("GamePk").reset_index(drop=True)  # canonical order

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
    joblib.dump(sel_art, ART / "models_bt.joblib", compress=3)
    with open(ART / "metrics_select.json", "w") as f:
        json.dump(sel_metrics, f, indent=2)
    log(f"saved selection artifacts to {ART / 'models_bt.joblib'}")
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
    # the holdout-year games available at train time, so the daily retrain
    # keeps it current as the run environment drifts (evaluate_deep Section
    # 4). STORED, not applied here — evaluate_deep scores the raw props so
    # the holdout stays honest; the Predictor uses these only under --recal,
    # and Section 10 is the leakage-free backtest that says whether they
    # actually help.
    import recalibrate as R
    from predict import predict_prop as _predict_prop
    te_hold = bf[bf["Season"] == hold_yr]
    inseason_offsets = {}
    for name, (target, _desc) in PROPS.items():
        y_h = te_hold[target].to_numpy()
        if len(y_h) > 200 and 0 < y_h.mean() < 1:
            p_h = _predict_prop(props[name], te_hold[bat_cols])
            inseason_offsets[name] = round(float(R.fit_logit_offset(p_h, y_h)), 4)
        else:
            inseason_offsets[name] = 0.0
    metrics["inseason_offsets"] = inseason_offsets
    log(f"in-season drift offsets ({hold_yr}): {inseason_offsets}")

    # multi-HR correction: E[HR | HR>=1], for expected-HR outputs
    tr_hr = bf[bf["Season"].isin(train_yrs) & (bf["hr_count"] >= 1)]
    multi_hr = float(tr_hr["hr_count"].mean())

    artifacts.update({
        "multi_hr": multi_hr,
        "slot_pa": slot_pa, "league_hr_pa": league_hr_pa,
        "inseason_offsets": inseason_offsets,
        "metrics": metrics,
        "trained_on": (f"{train_yrs[0]}-{train_yrs[-1]}, calibrated "
                       f"{cal_yr}, holdout-tested {hold_yr} YTD"),
    })
    joblib.dump(artifacts, ART / "models.joblib", compress=3)
    with open(ART / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"saved artifacts to {ART}")

    # feature importances for the HR model (top 25)
    imp = pd.Series(props["hr"]["gbm"].feature_importances_,
                    index=props["hr"]["cols"])
    log("top HR-model features:\n" +
        imp.sort_values(ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
