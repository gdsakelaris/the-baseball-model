"""Internal-quality rankings for every market the model predicts.

Produces the "what should I trust" table: every prediction column ranked by
the quality of its own served numbers on the two held-out test years (2025
selection test + 2026 holdout) — how trustworthy the probabilities are
(calibration), how much real skill sits behind them (edge/AUC/lift), and
whether both years agree. NO scraped odds, market prices, or de-vigged
references enter anywhere; the true market test lives in Section 9 of
evaluate_deep.py and stays there.

SCORE v4 (2026-07-15, second same-day re-ranking): the Score, the sort,
and the column order all follow the user's ranked diagnostic priorities
for betting, most to least significant — now PER CLASS:
  O/U count heads (13): Lift; Edge%; Slope; Intercept; Dispersion;
      Brier; ECE; Bias; MAE; AUC; Top10%; LogLoss; Accuracy.
  Binary heads (11, the effective order — no Disp/Bias): Lift; Edge%;
      Slope; Intercept; Brier; ECE; AUC; Top10%; LogLoss; MAE; Accuracy.
  Child columns (same day, user): each parent's baseline children sit
      beside it on the sheets — Disp = obs/cal ratio with DispCal/DispObs,
      Brier + BrierBase, MAE + MAEBase/MAEGain, LogLoss + LLBase. These
      are the exact numbers the parents' qualities were ALREADY scored
      against (folded), so the weights are unchanged; Base%, MeanAct,
      MeanPred ride as UNRANKED context columns in the user's positions
      (adjudicated 07-15: they hold sheet positions but do not score —
      base rate is a market property, MeanPred-MeanAct is Bias).

  Sort   — TIER first; tiers are cut ON THE SCORE ITSELF (user: "score
           and tier should be closely connected — tiers are separated by
           score"), so tiers are contiguous Score bands. Within tier:
           Score desc, then the ranked diagnostics as tie-breakers.
  Score  — the rank-WEIGHTED composite of the class's ranked diagnostics
           (weights linear in rank; each metric 0-1 on fixed anchors).
           Lift enters in its base-rate-fair odds-ratio form; raw Top10%
           participates at its own (low) rank; standalone log loss reads
           through its base-relative form, which IS the edge. All computed
           through the actual serving prices (per-line calibrators via
           predict.count_over, negative binomial for starter K / totals).
  Stability — Score = mean(S25, S26) - 0.25*|S25 - S26|: a market that
           performs in only one year is down-ranked for exactly that.
  Uncertainty — a weighted day-block bootstrap (the same philosophy as the
           model's paired accept bar) gives each Score a lower bound
           (Score_lo); the displayed tier reads the Score, but blue-mark
           DEPTH still caps on Score_lo — a thin market can't buy deep
           picks on a lucky point estimate.
  Tiers  — cut on Score on FROZEN semantic anchors (never re-fit to the
           run; re-anchored for v4), and WITHIN class: probability
           markets and expected-count means rank on separate ladders and
           are never compared.

Inputs (all written by the standard loop — no extra steps):
  - eval_paired_select_2025.joblib / eval_paired_2026.joblib   per-row
    (Date, p|mu, y) snapshots from `evaluate_deep.py [--confirm]
    --set-baseline`
  - models_bt.joblib / models.joblib   count heads (line calibrators,
    dispersions) matching each year's suite

Usage:
    python Tools/5_prop_rankings.py          # print + write the workbook
    python Tools/5_prop_rankings.py --out FILE

(Renamed from prop_rankings.py 2026-07-15, taking the retired
5_performance.py's slot in the game-day tool numbering. The digit-leading
filename can't be a plain `import` statement — predict.py and
evaluate_deep.py load it via importlib.import_module("5_prop_rankings").)
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Model"))
import predict as P  # noqa: E402  (count_over / nb_over / K_LINES)

ART = Path(__file__).resolve().parents[1] / "Model" / "artifacts"

# ---------------------------------------------------------------- markets
# One row per prediction COLUMN (family) in the prediction workbook, named
# exactly as the workbook names it — nothing is ranked that the Excel does
# not display, and nothing displayed is missing.
BIN_NAMES = {  # Batter Props sheet, probability columns
    "sb":     "Batter SB",
    "hr":     "Batter HR",
    "bb":     "Batter BB",
    "bk":     "Batter K",
    "bk2":    "Batter 2+ K",
    "bk3":    "Batter 3+ K",
    "tb3":    "Batter 3+ TB",
    "tb4":    "Batter 4+ TB",
    "hrr4":   "Batter H+R+RBI 4+",
    "triple": "Batter Triple",
    "rbi2":   "Batter 2+ RBI",
    "run2":   "Batter 2+ Runs",
    "hit":    "Batter Hit",
    "hits2":  "Batter 2+ Hits",
    "single": "Batter Single",
    "double": "Batter Double",
    "tb2":    "Batter 2+ TB",
    "run":    "Batter Run",
    "rbi":    "Batter RBI",
    "hrr2":   "Batter H+R+RBI 2+",
    "hrr3":   "Batter H+R+RBI 3+",
}
# O/U count columns, split by what the column IS in the workbook:
#   PITCHER_CNT — standalone O/U markets; the model is the SOLE pricer of
#     every line the book posts -> AUC/ECE/Edge% grade the FULL line family.
#   BATTER_X — RETIRED from the board 2026-07-14 (G3): the H1 deep binary
#     heads (bk3/tb3/tb4/hrr4) took the last lines the x-rows uniquely
#     priced, so they have nothing left to grade as line families. Their
#     MEANS still display and still grade (MEAN_MARKETS below).
#   GAME_CNT — game totals (no binary overlap; full family) + the H5
#     per-team total lines (2026-07-14).
PITCHER_CNT = {
    "k":     "Pitcher K > x",
    "outs":  "Pitcher Outs > x",
    "pha":   "Pitcher Hits > x",
    "pbb":   "Pitcher BB > x",
    "per":   "Pitcher ER > x",
}
BATTER_X = {}
GAME_CNT = {"total": "Game Runs > x", "team_total": "Team Runs > x"}
CNT_MARKETS = {**PITCHER_CNT, **BATTER_X, **GAME_CNT}

# The displayed expected-count (mean) columns, named as the workbook names
# them. Each is the MEAN output of a count head — a separate displayed
# prediction from the "> x" probabilities the same head prices, so it gets
# its own row graded on mean quality (MAE beat over a no-skill constant,
# rank correlation, bias) instead of line-pricing quality.
MEAN_MARKETS = {
    "xbk":   "Batter xSO",
    "xtb":   "Batter xTB",
    "xhrr":  "Batter xHRR",
    # H6 (2026-07-14): the rest of the expected-stat-line — means only,
    # their banked line calibrators never ship (binaries own the lines)
    "xh":    "Batter xH",
    "xrun":  "Batter xR",
    "xrbi":  "Batter xRBI",
    "xbb":   "Batter xBB",
    "k":     "Pitcher xK",
    "outs":  "Pitcher xOuts",
    "pha":   "Pitcher xHits",
    "pbb":   "Pitcher xBB",
    "per":   "Pitcher xER",
    "total": "Game Total Runs",
}

# Which of an x-column's lines the BINARY columns already price. After the
# H1 heads (2026-07-14) the binaries own EVERY line the x-heads quote
# (bk/bk2/bk3 -> 0.5/1.5/2.5; tb2/tb3/tb4 -> 1.5/2.5/3.5; hrr2/3/4 ->
# 1.5/2.5/3.5) — which is why BATTER_X retired from the line board (G3).
BINARY_OWNED_LINES = {"xbk": {0.5, 1.5, 2.5}, "xtb": {1.5, 2.5, 3.5},
                      "xhrr": {1.5, 2.5, 3.5}}

# ---------------------------------------------------- score construction
# SCORE v4 (user re-ranking 2026-07-15, second same-day revision). The
# Score is the rank-weighted average of the user's diagnostic priorities
# for BETTING, most to least significant — now PER CLASS, because binary
# heads have no Dispersion/Bias so their EFFECTIVE order differs:
#   O/U count heads (13):  1 Lift  2 Edge%  3 Slope  4 Intercept
#     5 Dispersion  6 Brier  7 ECE  8 Bias  9 MAE  10 AUC  11 Top10%
#     12 LogLoss  13 Accuracy
#   Binary heads (11):     1 Lift  2 Edge%  3 Slope  4 Intercept
#     5 Brier  6 ECE  7 AUC  8 Top10%  9 LogLoss  10 MAE  11 Accuracy
# Each diagnostic maps to a 0-1 quality on FIXED anchors and the composite
# is 100 * sum(w_i * q_i), weights linear in rank. Technicalities:
#   - Lift (rank 1) enters in its base-rate-fair ODDS-RATIO form
#     (ADJUDICATED 2026-07-15: read Lift, not raw Top10%, across markets;
#     the raw ratio is capped at 1/base so high-base markets can't show
#     skill through it).
#   - raw Top10% now participates AT ITS OWN RANK (11 binary / 11 count),
#     as ranked: it is a plain hit rate, base-inflated by construction,
#     which is exactly why the user ranked it low.
#   - standalone log loss is cross-market comparable only through its
#     base-rate-relative form, which IS the edge — so LogLoss reads the
#     same underlying quality as Edge% at its own (much lower) weight.
#   - MAE: binaries score the mean |p - y| beat over the base-rate
#     constant; O/U families the mean-count MAE beat over the no-skill
#     constant (the same number their mean row leads with).
#   - Bias / Dispersion (count heads only) are PROMOTED from the old
#     unranked x0.85-1.0 trust modifier to full ranked participants.
#   - the child columns (BrierBase, LLBase, MAEBase/MAEGain, DispCal/
#     DispObs) DISPLAY the exact baselines those parent qualities fold
#     in — brier_q = beat over BrierBase, ll/edge = relative gap to
#     LLBase, mae_q = MAEGain/MAEBase, disp_q = the DispObs-vs-DispCal
#     excess — so nothing scores twice and the weights are unchanged.
#     Base%/MeanAct/MeanPred are UNRANKED context (user 07-15).
# The stability haircut, day-block bootstrap lower bound, frozen-anchor
# tiers, and blue-mark depth machinery are unchanged — they now run on
# this composite. (v3's top-10-led single ladder is in git history.)
BIN_RANK_W = {           # weight ~ (12 - rank)/66, ranks 1-11
    "lift": 11 / 66, "edge": 10 / 66, "slope": 9 / 66, "int": 8 / 66,
    "brier": 7 / 66, "ece": 6 / 66, "auc": 5 / 66, "top10": 4 / 66,
    "ll": 3 / 66, "mae": 2 / 66, "acc": 1 / 66,
}
CNT_RANK_W = {           # weight ~ (14 - rank)/91, ranks 1-13
    "lift": 13 / 91, "edge": 12 / 91, "slope": 11 / 91, "int": 10 / 91,
    "disp": 9 / 91, "brier": 8 / 91, "ece": 7 / 91, "bias": 6 / 91,
    "mae": 5 / 91, "auc": 4 / 91, "top10": 3 / 91, "ll": 2 / 91,
    "acc": 1 / 91,
}

# Day-block bootstrap. The rest of the pipeline lives by paired day-block
# CIs; the ranking should too. We resample DAYS (multinomial day counts =
# per-day weights) B times, recompute each market's whole composite Score,
# and TIER ON THE LOWER CONFIDENCE BOUND (the SCORE_LCB_Q percentile). This
# makes the ranking sample-size aware for free — a thin market (SB, a deep K
# line) earns a wide CI and a demoted tier, exactly like the model's accept
# bar treats a thin edge. The point Score is still shown (full sample); the
# LCB is what tiers and what caps blue depth. Weighted metrics make a
# resample an O(n) reweight (validated == the multiplicity bootstrap).
BOOT_B, SCORE_LCB_Q, BOOT_SEED = 400, 0.10, 20260714

# Tier cuts are FROZEN semantic anchors on the 0-100 composite scale —
# NOT re-fit to each run's distribution. That matters: because the Score is
# built only from a market's OWN edge / calibration / CI, a frozen cut makes
# a market's tier depend on ITSELF alone, never on how other markets moved
# in a retrain. v4: applied to the SCORE ITSELF (user: "score and tier
# should be closely connected — tiers are separated by score"), so tiers
# are contiguous Score bands on the board; Score_lo remains the displayed
# uncertainty and still caps blue-mark DEPTH. Probability markets (binaries,
# O/U lines and families, winner) and MEAN markets (expected-count columns)
# sit on DIFFERENT scales, so each class has its own ladder and its own
# block in the workbook — a mean Score and a binary Score are never compared.
# (To re-anchor these is a deliberate, documented, versioned decision.)
# Prob anchors: ELITE = strong proper-score edge AND trustworthy probabilities
# (~.65 Skill x ~.85 Trust); each step down relaxes one of the two. Mean
# anchors run higher because the mean Skill anchors (MAE beat /10, rho /.45)
# saturate sooner — which is exactly WHY the two classes cannot share a ladder.
# An empty tier is honest (the means genuinely bifurcate: xK/xOuts, then a
# cliff); anchors are NEVER nudged to make tiers look populated.
#
# 4 DECENT / 5 LOW CEILING replace the old single "MARGINAL" band, split at
# that band's MIDPOINT (a semantic rule, not a fit to where this run's
# distribution happens to break).
#
# The rename is not politeness, and it is NOT "these props are near their
# ceiling" — that would name nothing, since the oracle test found EVERY batter
# binary is near its ceiling (a leave-one-out oracle knowing each batter's TRUE
# full-season rate scores LOWER AUC than the shipped head on all 14, elite ones
# included). What actually separates this band is that its ceiling is LOW: the
# event is ~4 Bernoulli trials with a compressed true-p spread, so even a
# perfect model can only separate players a little. A thin Score on
# hit/run/rbi/tb2/hrr is therefore a fact about the MARKET, not a defect to fix
# — "MARGINAL" implied a fixable deficiency, LOW CEILING says the true thing.
# (6 AVOID keeps its name: it holds markets genuinely BELOW their achievable
# bar — winner and total sit under the ~.60-.62 market ceiling — next to the
# thinnest binaries.)
# RE-ANCHORED for Score v4 (a deliberate, documented, versioned decision —
# the composite changed weights AND the cut moved from Score_lo to Score,
# so the v3 anchors are meaningless on it). Anchors remain SEMANTIC:
# ELITE = strong base-fair top-pick lift AND a near-ideal calibration line
# AND a clear proper-score beat; each step down relaxes one axis. The
# bottom of the PROB ladder is deliberately tighter than the top: the
# DECENT / LOW CEILING / AVOID boundaries sit at the thin-spread
# batter-event cluster and the below-market-ceiling game heads — the
# adjudicated trust story those tiers exist to express. They are NOT
# re-fit per run.
PROB_TIER_CUTS = ((72.0, "1 ELITE"), (60.0, "2 STRONG"), (48.0, "3 SOLID"),
                  (44.0, "4 DECENT"), (41.0, "5 LOW CEILING"),
                  (float("-inf"), "6 AVOID"))
MEAN_TIER_CUTS = ((76.0, "1 ELITE"), (62.0, "2 STRONG"), (46.0, "3 SOLID"),
                  (36.0, "4 DECENT"), (28.0, "5 LOW CEILING"),
                  (float("-inf"), "6 AVOID"))


def tier_of(score, cuts):
    """Tier label for a score on a frozen ladder; '-' for ungraded rows."""
    if not np.isfinite(score):
        return "-"
    return next(t for c, t in cuts if score >= c)


# ------------------------------------------------------------ diagnostics
def ece(p, y, bins=10):
    """Expected calibration error, equal-count bins (evaluate_deep's ece)."""
    q = pd.qcut(p, bins, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "q": q})
    g = df.groupby("q", observed=True)
    return float((g["p"].mean().sub(g["y"].mean()).abs()
                  * g.size().div(len(df))).sum())


def cal_line(p, y):
    """Calibration line: logistic refit of the outcome on the served logit
    -> (slope, intercept). Slope 1.0 = the probability moves exactly as much
    as it should (< 1 overconfident, > 1 underconfident); intercept 0 = no
    systematic log-odds shift at the middle of the scale."""
    z = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def cal_slope(p, y):
    return cal_line(p, y)[0]


def top10_lift(df, p_col="p", y_col="y"):
    """Daily top-10 hit rate over the base rate (how much the best picks
    beat blind betting)."""
    day = df.assign(d=pd.to_datetime(df["Date"]).dt.date)
    top = day.sort_values(p_col, ascending=False).groupby("d").head(10)
    base = df[y_col].mean()
    return float(top[y_col].mean() / base) if base > 0 else np.nan


def _binary_diags(p, y, lift):
    """The full ranked-diagnostic set for one binary-like unit (a prop, a
    priced O/U line, or the winner): everything the v3 Score and the ranked
    sort read, from the same served p/y rows."""
    base = float(y.mean())
    base_ll = log_loss(y, np.full_like(p, base))
    ll = float(log_loss(y, p))
    brier = float(np.mean((p - y) ** 2))
    brier_base = base * (1 - base)
    e = ece(p, y)
    slope, intc = cal_line(p, y)
    acc = float(np.mean((p >= 0.5) == (y > 0.5)))
    mae = float(np.mean(np.abs(p - y)))
    mae_base = 2 * base * (1 - base)     # MAE of the base-rate constant
    return {
        "rel": 100 * (base_ll - ll) / base_ll,
        "ll": ll,
        "ll_base": float(base_ll),
        "auc": float(roc_auc_score(y, p)),
        "brier": brier,
        "brier_base": brier_base,
        "brier_rel": (100 * (brier_base - brier) / brier_base
                      if brier_base > 0 else np.nan),
        "ece": e,
        "ece_rel": e / (base * (1 - base)),
        "slope": slope,
        "int": intc,
        "acc": acc,
        "acc_base": max(base, 1 - base),
        "mae": mae,
        "mae_base": mae_base,
        "mae_gain": mae_base - mae,
        "mae_rel": (100 * (mae_base - mae) / mae_base
                    if mae_base > 0 else np.nan),
        "lift": lift,
        "top10": lift * base if np.isfinite(lift) else np.nan,
        "base": base,
        "mean_actual": base,
        "mean_pred": float(np.mean(p)),
    }


def binary_year(snap):
    """Per binary prop, one year: the full ranked-diagnostic set (top-10
    raw + lift, slope/intercept, ECE, Brier, log loss, edge, AUC, acc)."""
    out = {}
    for name, blob in snap["binary"].items():
        df = blob["df"]
        y, p = df["y"].to_numpy(), np.clip(df["p"].to_numpy(), 1e-4, 1 - 1e-4)
        out[name] = _binary_diags(p, y, top10_lift(df))
    return out


def mean_metrics(df):
    """One year of a displayed MEAN column (xK, xOuts, ..., Total Runs,
    Away/Home Score): relative MAE beat over the no-skill constant (the
    count analog of Edge%), Spearman rank correlation (the ranking-skill
    analog of AUC), bias and observed dispersion."""
    mu, y = df["mu"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)
    mae_c = float(np.mean(np.abs(y - y.mean())))
    mae_m = float(np.mean(np.abs(y - mu)))
    return {
        "rel": 100 * (mae_c - mae_m) / mae_c if mae_c else np.nan,
        "mae": mae_m,
        "mae_base": mae_c,
        "mae_gain": mae_c - mae_m,
        "rho": float(pd.Series(mu).corr(pd.Series(y), method="spearman")),
        "bias": float(mu.mean() - y.mean()),
        "disp": float(np.mean((y - mu) ** 2) / mu.mean()),
        "mean_y": float(y.mean()),
        "mean_actual": float(y.mean()),
        "mean_pred": float(mu.mean()),
    }


def mean_score(m):
    """Mean-column composite: MAE is the highest-ranked diagnostic a mean
    column carries (count rank 9; the probability metrics don't exist for
    a mean), so the MAE beat over the no-skill constant leads at 60%; rank
    correlation (25%) and bias trust (15%) stay as unranked auxiliaries —
    a mean that can't order outcomes or sits off-level is less usable
    however small its MAE. Own class, own ladder (MEAN_TIER_CUTS) — never
    compared to a probability Score."""
    bias_q = _clip(1 - abs(m["bias"]) / (0.05 * m["mean_y"]))
    return 100 * (0.60 * _clip(m["rel"] / 10.0)
                  + 0.25 * _clip(m["rho"] / 0.45)
                  + 0.15 * bias_q)


def mean_year(snap):
    """{key: mean_metrics} for every displayed mean column one snapshot
    carries: the count heads + game total, plus 'score' (Away/Home Score,
    per-team rows) when the snapshot has it."""
    out = {}
    for key in MEAN_MARKETS:
        blob = snap.get("count", {}).get(key)
        if blob is not None:
            out[key] = mean_metrics(blob["df"])
    sc = snap.get("score")
    if sc is not None:
        out["score"] = mean_metrics(sc["df"])
    return out


def winner_year(snap):
    """Win Prob graded on the same internal diagnostics as a binary prop.
    Lift = the day's single most CONFIDENT game (top-10 makes no sense on a
    ~15-game slate): picked-side hit rate over the always-home hit rate.
    None when the snapshot predates winner rows (re-run --set-baseline)."""
    blob = snap.get("winner")
    if blob is None:
        return None
    df = blob["df"]
    y = df["y"].to_numpy()
    p = np.clip(df["p"].to_numpy(), 1e-4, 1 - 1e-4)
    base = y.mean()
    day = df.assign(d=pd.to_datetime(df["Date"]).dt.date,
                    conf=np.maximum(p, 1 - p),
                    hit=((p >= 0.5) == (y == 1)).astype(float))
    top1 = day.sort_values("conf", ascending=False).groupby("d").head(1)
    lift = float(top1["hit"].mean() / base) if base > 0 else np.nan
    return _binary_diags(p, y, lift)


def count_lines(name, head, k_disp, total_disp, art=None):
    """The lines a count column prices and its P(over) pricer, mirroring
    serving: per-line calibrators via count_over (which itself prices the
    NB_PRICED_TARGETS heads, e.g. per, with the negative binomial), NB for
    starter K and game totals; team_over for the H5 per-team lines."""
    if name == "k":
        return P.K_LINES, lambda mu, ln: np.array(
            [P.nb_over(m, ln, k_disp) for m in mu])
    if name == "total":
        return [6.5, 7.5, 8.5, 9.5, 10.5], lambda mu, ln: np.array(
            [P.nb_over(m, ln, total_disp) for m in mu])
    if name == "team_total":
        return P.TEAM_TOTAL_LINES, lambda mu, ln: P.team_over(art or {},
                                                              mu, ln)
    return head["lines"], lambda mu, ln: P.count_over(head, mu, ln)


def count_year(snap, art):
    """Per expected-count column, one year: per-line edge/AUC/ECE/slope
    through the actual serving prices (averaged over the family), plus
    bias, dispersion-vs-priced and a flagship-line top-10 lift."""
    out = {}
    heads = art.get("count_models", {})
    k_disp = float(art.get("k_disp", 1.0))
    total_disp = float(art.get("total_disp", 1.0))
    for name, blob in snap["count"].items():
        df = blob["df"]
        mu, y = df["mu"].to_numpy(), df["y"].to_numpy()
        head = heads.get(name)
        lines, pricer = count_lines(name, head, k_disp, total_disp, art)
        # the dispersion the head's pricer ASSUMES: NB-priced heads bake
        # their cal-year dispersion into P(over); calibrator heads price
        # each listed line empirically but extremes beyond extrapolate a
        # mean-variance view (1.0). play_note warns on observed EXCESS
        # over this, not over raw Poisson.
        priced_disp = (k_disp if name == "k" else
                       total_disp if name == "total" else
                       head["disp"] if head is not None
                       and head.get("target") in P.NB_PRICED_TARGETS
                       else 1.0)
        # each line is a binary market graded through the actual serving
        # price; the family average is directly comparable to the binary
        # rows (pricers are monotone in mu, so per-line AUC is pure
        # within-line ranking; pooling lines would inflate it). BATTER_X
        # rows skip the lines their binary siblings already sell
        # (BINARY_OWNED_LINES) for AUC/ECE/slope; Edge% still averages
        # every line the column quotes.
        owned_ln = BINARY_OWNED_LINES.get(name, set())
        rels = []
        agg = defaultdict(list)      # non-owned lines' diagnostics, averaged
        flagship = None              # first uniquely-priced line's diags
        per_line = {}
        for ln in lines:
            yy = (y > ln).astype(int)
            base = yy.mean()
            if base in (0.0, 1.0):
                continue
            pp = np.clip(pricer(mu, ln), 1e-4, 1 - 1e-4)
            d = _binary_diags(pp, yy.astype(float), top10_lift(
                pd.DataFrame({"Date": df["Date"], "p": pp, "y": yy})))
            d["owned"] = ln in owned_ln
            per_line[ln] = d
            rels.append(d["rel"])    # Edge% averages EVERY quoted line
            if not d["owned"]:
                for k in ("auc", "ece", "ece_rel", "slope", "int", "ll",
                          "ll_base", "brier", "brier_base", "brier_rel",
                          "acc", "acc_base"):
                    agg[k].append(d[k])
                if flagship is None:
                    flagship = d
        fl = flagship or {}
        mae_c = float(np.mean(np.abs(y - y.mean())))
        mae_m = float(np.mean(np.abs(y - mu)))
        disp_v = float(np.mean((y - mu) ** 2) / mu.mean())
        out[name] = {
            "lines": per_line,
            **{k: float(np.nanmean(v)) if v else np.nan
               for k, v in agg.items()},
            "rel": float(np.mean(rels)) if rels else np.nan,
            "lift": float(fl.get("lift", np.nan)),
            "top10": float(fl.get("top10", np.nan)),
            "base": float(fl.get("base", np.nan)),
            "bias": float(mu.mean() - y.mean()),
            "disp": disp_v,
            # the folded number the Score's disp term reads: observed over
            # what the pricing assumes, 1.0 ideal (user call 07-15: the
            # parent Disp column shows this; Cal/Obs are its children)
            "disp_ratio": (disp_v / float(priced_disp)
                           if priced_disp > 0 else np.nan),
            "mae": mae_m,
            "mae_base": mae_c,
            "mae_gain": mae_c - mae_m,
            "mae_rel": 100 * (mae_c - mae_m) / mae_c if mae_c else np.nan,
            "priced_disp": float(priced_disp),
            "mean_y": float(y.mean()),
            "mean_actual": float(y.mean()),
            "mean_pred": float(mu.mean()),
            "n_lines": len(rels),
        }
    return out


# ------------------------------------------------------- composite score
# Score(year) = 100 * sum(w_i * q_i) over the class's user-ranked
# diagnostics (BIN_RANK_W / CNT_RANK_W); Score = mean(S25, S26) -
# 0.25 * |S25 - S26| (stability haircut, unchanged). Each q is a 0-1
# quality on a FIXED anchor:
#   lift_q   odds-ratio top-pick lift (base-rate-fair form of the Lift
#            the board shows): 1.0 -> 0, 2.5 -> full marks
#   edge_q   relative log-loss beat over base rate (8% binary, O/U 10% =
#            full) — read again at the LogLoss rank (standalone log loss
#            is only cross-market comparable through this form)
#   slope_q  calibration slope: full at 1.0, zero +/-0.5 away
#   int_q    calibration intercept: full at 0, zero at +/-0.4 log-odds
#   disp_q   (counts) observed/priced dispersion: excess 0 = full, 50% = 0
#   brier_q  relative Brier beat over the base-rate forecast; 6% = full
#   ece_q    scaled ECE / (base*(1-base)); 10% of Bernoulli scale = zero
#   bias_q   (counts) |bias| as a share of the mean; 5% of mean = zero
#   mae_q    relative MAE beat over the base-rate / no-skill constant
#            (5% = full binaries, 10% = full counts)
#   auc_q    (AUC - .5)/.20 (.70 = full)
#   top10_q  the RAW daily top-10 hit rate, as ranked (base-inflated by
#            construction — exactly why it sits at rank 11)
#   acc_q    accuracy beat over the trivial always-majority pick; 5pp = full
def _clip(x):
    return float(max(0.0, min(1.0, x)))


def _q(v, fn):
    return fn(v) if np.isfinite(v) else 0.0


def _quals(m, rel_full=8.0, mae_full=5.0):
    orl = or_lift(m.get("lift", np.nan), m.get("base", np.nan)) \
        if "orl" not in m else m["orl"]
    q = {
        "lift": _q(orl, lambda o: _clip((o - 1) / 1.5)),
        "edge": _q(m.get("rel", np.nan), lambda r: _clip(r / rel_full)),
        "slope": _q(m.get("slope", np.nan),
                    lambda s: _clip(1 - abs(s - 1) / 0.5)),
        "int": _q(m.get("int", np.nan), lambda a: _clip(1 - abs(a) / 0.4)),
        "brier": _q(m.get("brier_rel", np.nan), lambda b: _clip(b / 6.0)),
        "ece": _q(m.get("ece_rel", np.nan), lambda e: _clip(1 - e / 0.10)),
        "auc": _q(m.get("auc", np.nan), lambda a: _clip((a - 0.5) / 0.20)),
        "top10": _q(m.get("top10", np.nan), _clip),
        "ll": _q(m.get("rel", np.nan), lambda r: _clip(r / rel_full)),
        "mae": _q(m.get("mae_rel", np.nan), lambda r: _clip(r / mae_full)),
        "acc": _q(m.get("acc", np.nan) - m.get("acc_base", np.nan),
                  lambda d: _clip(d / 0.05)),
    }
    if "bias" in m:
        q["bias"] = _q(m.get("bias", np.nan),
                       lambda b: _clip(1 - abs(b) / (0.05 * m["mean_y"])))
        q["disp"] = _q(m.get("disp", np.nan),
                       lambda d: _clip(1 - max(0.0, d / max(
                           m.get("priced_disp", 1.0), 1.0) - 1.0) / 0.5))
    return q


def binary_score(m):
    """Binary head: the 11-rank effective-order composite (BIN_RANK_W)."""
    q = _quals(m, rel_full=8.0, mae_full=5.0)
    return 100 * sum(BIN_RANK_W[k] * q[k] for k in BIN_RANK_W)


def count_score(m):
    """O/U family: the full 13-rank composite (CNT_RANK_W) — Dispersion
    (rank 5) and Bias (rank 8) participate as ranked terms now, not as the
    old unranked x0.85-1.0 trust modifier; MAE (rank 9) is the mean-count
    MAE beat over the no-skill constant."""
    q = _quals(m, rel_full=10.0, mae_full=10.0)
    return 100 * sum(CNT_RANK_W[k] * q[k] for k in CNT_RANK_W)


def final_score(s25, s26):
    return max(0.0, (s25 + s26) / 2 - 0.25 * abs(s25 - s26))


def play_note(kind, m25, m26):
    """Data-driven guidance: shading direction, calibration trust, pick
    depth — derived from the averaged diagnostics, no hand judgment."""
    notes = []
    slope = np.nanmean([m25.get("slope", np.nan), m26.get("slope", np.nan)])
    if kind == "cnt":
        bias = (m25["bias"] + m26["bias"]) / 2
        disp = (m25["disp"] + m26["disp"]) / 2
        if bias > 0.1:
            notes.append(f"over-predicts +{bias:.2f} -> lean unders")
        elif bias < -0.1:
            notes.append(f"under-predicts {bias:.2f} -> lean overs")
        else:
            notes.append("unbiased")
        priced = (m25.get("priced_disp", 1.0) + m26.get("priced_disp", 1.0)) / 2
        excess = disp / max(priced, 1.0)
        if excess > 1.5:
            notes.append(f"wild tails (disp {disp:.1f} vs {priced:.1f} priced)"
                         " -> shade extreme lines")
        elif excess <= 1.05:
            notes.append("tails priced right"
                         + (f" (NB {priced:.1f})" if priced > 1.05 else ""))
        else:
            notes.append(f"tails a bit wilder than priced (disp {disp:.1f} "
                         f"vs {priced:.1f}) -> light shade on extremes")
    else:
        ec = (m25["ece"] + m26["ece"]) / 2
        lift = (m25["lift"] + m26["lift"]) / 2
        orl = np.nanmean([or_lift(m25.get("lift", np.nan),
                                  m25.get("base", np.nan)),
                          or_lift(m26.get("lift", np.nan),
                                  m26.get("base", np.nan))])
        if ec <= 0.007:
            notes.append("calibrated -> price bets directly")
        elif ec >= 0.011:
            notes.append(f"probability level drifts (ECE {ec:.3f}) -> "
                         "trust picks more than prices")
        # selection power judged on the base-rate-fair ODDS-RATIO lift (the
        # Score's top-10 form; raw ratio understates high-base markets) —
        # thresholds = the blue-mark gates
        if orl >= BLUE_OR_DEEP:
            notes.append(f"top picks {lift:.1f}x base (OR {orl:.1f}) -> "
                         "follow the list deep")
        elif orl >= BLUE_OR_TOP:
            notes.append(f"top picks {lift:.1f}x (OR {orl:.1f}) -> "
                         "top 3-10 only")
        else:
            notes.append(f"picks {lift:.1f}x base (OR {orl:.1f}) -> "
                         "no selection power")
    if np.isfinite(slope):
        if slope < 0.85:
            notes.append(f"overconfident (slope {slope:.2f}) -> shade "
                         "extreme probabilities toward the middle")
        elif slope > 1.15:
            notes.append(f"underconfident (slope {slope:.2f}) -> extremes "
                         "are even better than stated")
    return "; ".join(notes)


# ============================================================ bootstrap CIs
# Weighted day-block bootstrap. A resample draws multinomial day COUNTS (the
# day-block bootstrap) and every metric becomes an O(n) weighted mean /
# weighted rank-sum, so nothing is re-sorted or re-priced per resample. The
# weighted primitives were validated == a physical multiplicity bootstrap
# (AUC to machine precision; ECE/slope to <1e-3). Each market's whole
# composite Score is recomputed B times; we keep the SCORE_LCB_Q lower bound.
def _binent(b):
    b = np.clip(b, 1e-12, 1 - 1e-12)
    return -(b * np.log(b) + (1 - b) * np.log(1 - b))


def or_lift(lift, base):
    """Odds-ratio top-pick lift: the top picks' hit ODDS over the base odds.
    Base-rate-fair where raw lift (capped at 1/base) is not, so a high-base
    column like Hit competes with a low-base one. NaN when undefined."""
    if not (np.isfinite(lift) and np.isfinite(base)) or not 0 < base < 1:
        return np.nan
    top = min(lift * base, 1 - 1e-6)
    return (top / (1 - top)) / (base / (1 - base))


def _slope_irls(z, y, w, a=0.0, b=1.0, iters=25):
    """1-D weighted logistic fit by Newton/IRLS -> (intercept, slope);
    matches cal_line's near-unregularized fit and is warm-started in the
    bootstrap for ~6-iter cost."""
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(a + b * z)))
        Wd = w * mu * (1 - mu)
        r = w * (y - mu)
        h00, h01, h11 = Wd.sum(), (Wd * z).sum(), (Wd * z * z).sum()
        det = h00 * h11 - h01 * h01
        if abs(det) < 1e-12:
            break
        g0, g1 = r.sum(), (r * z).sum()
        da, db = (h11 * g0 - h01 * g1) / det, (-h01 * g0 + h00 * g1) / det
        a, b = a + da, b + db
        if abs(da) + abs(db) < 1e-10:
            break
    return float(a), float(b)


def _prep_binlike(day_id, D, p, y):
    """Precompute everything a binary-like unit (a binary prop, a priced O/U
    line, or the winner column) needs so a resample is pure reweighting:
    per-day aggregates for edge/base/lift, and fixed sort orders + tie blocks
    for weighted AUC/ECE, plus a warm-start slope."""
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    z = np.log(p / (1 - p))
    nll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    sumy = np.bincount(day_id, y, D)
    cnt = np.bincount(day_id, minlength=D).astype(float)
    nllsum = np.bincount(day_id, nll, D)
    sqsum = np.bincount(day_id, (p - y) ** 2, D)          # Brier numerator
    aesum = np.bincount(day_id, np.abs(p - y), D)         # MAE numerator
    hitsum = np.bincount(day_id,                          # accuracy numerator
                         ((p >= 0.5) == (y > 0.5)).astype(float), D)
    # per-day top-10-by-p y-sum (fixed under day resampling): sort by day then
    # -p, keep the first 10 of each day-group
    o = np.lexsort((-p, day_id))
    did_s, y_s = day_id[o], y[o]
    gstart = np.flatnonzero(np.concatenate(([True], did_s[1:] != did_s[:-1])))
    pos = np.arange(len(o)) - np.repeat(gstart, np.diff(np.append(gstart, len(o))))
    tm = pos < 10
    picks_sumy = np.bincount(did_s[tm], y_s[tm], D)
    picks_k = np.bincount(did_s[tm], minlength=D).astype(float)
    od = np.argsort(-p, kind="mergesort")           # AUC: descending, tie blocks
    ps_d = p[od]
    starts = np.concatenate(([0], np.nonzero(np.diff(ps_d))[0] + 1))
    oa = np.argsort(p, kind="mergesort")            # ECE: ascending
    return {"D": D, "z": z, "y": y, "day_id": day_id,
            "sumy": sumy, "cnt": cnt, "nllsum": nllsum,
            "sqsum": sqsum, "aesum": aesum, "hitsum": hitsum,
            "picks_sumy": picks_sumy, "picks_k": picks_k,
            "ys_d": y[od], "did_d": day_id[od], "starts": starts,
            "pa": p[oa], "ya": y[oa], "did_a": day_id[oa],
            "ab0": _slope_irls(z, y, np.ones_like(y))}


def _wauc(pp, w):
    wr = w[pp["did_d"]]
    dTP = np.add.reduceat(wr * pp["ys_d"], pp["starts"])
    dFP = np.add.reduceat(wr * (1 - pp["ys_d"]), pp["starts"])
    P, N = dTP.sum(), dFP.sum()
    if P <= 0 or N <= 0:
        return np.nan
    return float(((np.cumsum(dTP) - dTP) * dFP + 0.5 * dTP * dFP).sum() / (P * N))


def _wece(pp, w, base):
    wa = w[pp["did_a"]]
    cw = np.cumsum(wa)
    tot = cw[-1]
    if tot <= 0:
        return np.nan
    edges = np.linspace(0, tot, 11)[1:-1]
    bnd = np.unique(np.concatenate(([0], np.searchsorted(cw, edges), [len(wa)])))
    ece = 0.0
    for a, b in zip(bnd[:-1], bnd[1:]):
        sw = wa[a:b].sum()
        if sw <= 0:
            continue
        mp = (wa[a:b] * pp["pa"][a:b]).sum() / sw
        my = (wa[a:b] * pp["ya"][a:b]).sum() / sw
        ece += (sw / tot) * abs(mp - my)
    return float(ece)


def _wslope(pp, w):
    return _slope_irls(pp["z"], pp["y"], w[pp["day_id"]],
                       a=pp["ab0"][0], b=pp["ab0"][1], iters=6)


def _wdiags(pp, Wm, rel_full=8.0):
    """B full ranked-diagnostic dicts for one binary-like unit. The per-day
    quantities vectorize across all B draws; only AUC/ECE/cal-line loop."""
    Wtot = Wm @ pp["cnt"]
    base = (Wm @ pp["sumy"]) / Wtot
    base_ll = _binent(base)
    rel = np.where(base_ll > 0, 100 * (base_ll - (Wm @ pp["nllsum"]) / Wtot)
                   / base_ll, np.nan)
    brier = (Wm @ pp["sqsum"]) / Wtot
    bvar = base * (1 - base)
    brier_rel = np.where(bvar > 0, 100 * (bvar - brier) / bvar, np.nan)
    mae = (Wm @ pp["aesum"]) / Wtot
    mae_base = 2 * base * (1 - base)
    mae_rel = np.where(mae_base > 0, 100 * (mae_base - mae) / mae_base,
                       np.nan)
    acc = (Wm @ pp["hitsum"]) / Wtot
    pkw = Wm @ pp["picks_k"]
    lift = np.where((pkw > 0) & (base > 0),
                    (Wm @ pp["picks_sumy"]) / pkw / base, np.nan)
    ms = []
    for b, w in enumerate(Wm):
        bb = base[b]
        er = _wece(pp, w, bb) / (bb * (1 - bb)) if 0 < bb < 1 else np.nan
        a_, s_ = _wslope(pp, w)
        ms.append({"rel": rel[b], "auc": _wauc(pp, w), "ece_rel": er,
                   "slope": s_, "int": a_, "brier_rel": brier_rel[b],
                   "acc": acc[b], "acc_base": max(bb, 1 - bb),
                   "mae_rel": mae_rel[b],
                   "top10": (lift[b] * bb if np.isfinite(lift[b])
                             else np.nan),
                   "orl": or_lift(lift[b], bb)})
    return ms, lift, base


def _binlike_boot(pp, Wm):
    """B binary Scores -> arrays (score, lift, base)."""
    ms, lift, base = _wdiags(pp, Wm, rel_full=8.0)
    return np.array([binary_score(m) for m in ms]), lift, base


def _prep_count_family(name, snap, head, k_disp, total_disp, art=None):
    df = snap["count"][name]["df"]
    day_id, uniq = pd.factorize(df["Date"], sort=False)
    D = len(uniq)
    mu, y = df["mu"].to_numpy(float), df["y"].to_numpy(float)
    lines, pricer = count_lines(name, head, k_disp, total_disp, art)
    owned = BINARY_OWNED_LINES.get(name, set())
    priced_disp = (k_disp if name == "k" else total_disp if name == "total"
                   else head["disp"] if head is not None
                   and head.get("target") in P.NB_PRICED_TARGETS else 1.0)
    lp = {}
    for ln in lines:
        yy = (y > ln).astype(float)
        base = yy.mean()
        if base in (0.0, 1.0):
            continue
        pp = np.clip(pricer(mu, ln), 1e-4, 1 - 1e-4)
        lp[ln] = {"prep": _prep_binlike(day_id, D, pp, yy),
                  "owned": ln in owned}
    return {"D": D, "lines": lp, "priced_disp": float(priced_disp),
            "d_mu": np.bincount(day_id, mu, D), "d_y": np.bincount(day_id, y, D),
            "d_sq": np.bincount(day_id, (y - mu) ** 2, D),
            # MAE numerators: model error + deviation from the FULL-sample
            # mean (the no-skill constant is held fixed across resamples —
            # second-order vs re-deriving it per draw)
            "d_ae": np.bincount(day_id, np.abs(y - mu), D),
            "d_dev": np.bincount(day_id, np.abs(y - y.mean()), D),
            "d_cnt": np.bincount(day_id, minlength=D).astype(float)}


def _countfam_boot(prep, Wm):
    """Family composite Score per resample: non-owned lines' ranked
    diagnostics averaged (as count_year does), bias/dispersion/MAE
    re-derived under the weights, and the flagship (first uniquely-priced)
    line's odds-ratio lift + raw top-10 as the family's Lift/Top10% terms."""
    W = Wm @ prep["d_cnt"]
    mu_m, y_m = (Wm @ prep["d_mu"]) / W, (Wm @ prep["d_y"]) / W
    bias = mu_m - y_m
    disp = (Wm @ prep["d_sq"]) / W / mu_m
    mae_m = (Wm @ prep["d_ae"]) / W
    mae_c = (Wm @ prep["d_dev"]) / W
    mae_rel = np.where(mae_c > 0, 100 * (mae_c - mae_m) / mae_c, np.nan)
    non = [lp for lp in prep["lines"].values() if not lp["owned"]]
    per_line = [_wdiags(lp["prep"], Wm, rel_full=10.0)[0] for lp in non]
    sc = np.empty(len(Wm))
    for b in range(len(Wm)):
        ds = [pl[b] for pl in per_line]
        m = {k: (float(np.nanmean([d[k] for d in ds])) if ds else np.nan)
             for k in ("rel", "auc", "ece_rel", "slope", "int",
                       "brier_rel", "acc", "acc_base")}
        m["orl"] = ds[0]["orl"] if ds else np.nan     # flagship line
        m["top10"] = ds[0]["top10"] if ds else np.nan
        m.update(bias=bias[b], disp=disp[b], mae_rel=mae_rel[b],
                 priced_disp=prep["priced_disp"], mean_y=y_m[b])
        sc[b] = count_score(m)
    return sc


def _prep_mean(df, rho):
    day_id, uniq = pd.factorize(df["Date"], sort=False)
    return {"day_id": day_id, "mu": df["mu"].to_numpy(float),
            "y": df["y"].to_numpy(float), "rho": rho, "D": len(uniq)}


def _mean_boot(prep, Wm):
    mu, y, rho = prep["mu"], prep["y"], prep["rho"]
    sc = np.empty(len(Wm))
    for b, w in enumerate(Wm):
        wr = w[prep["day_id"]]
        W = wr.sum()
        ybar = (wr * y).sum() / W
        mae_c = (wr * np.abs(y - ybar)).sum() / W
        mae_m = (wr * np.abs(y - mu)).sum() / W
        sc[b] = mean_score({"rel": 100 * (mae_c - mae_m) / mae_c if mae_c else np.nan,
                            "rho": rho, "bias": (wr * (mu - y)).sum() / W,
                            "mean_y": ybar})
    return sc


def _combine(s25, s26):
    """final_score applied elementwise across paired resample draws."""
    return np.maximum(0.0, (s25 + s26) / 2 - 0.25 * np.abs(s25 - s26))


def _lcb(arr):
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, SCORE_LCB_Q * 100)) if len(a) else np.nan


def _draws(D, rng):
    return rng.multinomial(D, np.full(D, 1.0 / D), size=BOOT_B).astype(float)


def bootstrap_lcb(snap25, snap26, art25, art26):
    """Per-entity Score lower bounds (+ odds-ratio lift point/LCB for the
    blue-eligible ones). Entities: 14 binaries, 9 O/U families, every priced
    O/U line, the winner column, and the mean columns. Independent day-block
    resamples per year, combined by the stability haircut, LCB at SCORE_LCB_Q."""
    rng = np.random.default_rng(BOOT_SEED)
    heads25 = art25.get("count_models", {})
    heads26 = art26.get("count_models", {})
    kd25, td25 = float(art25.get("k_disp", 1.0)), float(art25.get("total_disp", 1.0))
    kd26, td26 = float(art26.get("k_disp", 1.0)), float(art26.get("total_disp", 1.0))
    out = {"binary": {}, "count_fam": {}, "count_line": {}, "mean": {}}

    # binaries + winner (binary-like)
    binlike = [(k, "binary", k) for k in snap25["binary"]] + [("winner", "winner", None)]
    for key, kind, sub in binlike:
        df25 = (snap25["binary"][key]["df"] if kind == "binary"
                else snap25.get("winner", {}).get("df"))
        df26 = (snap26["binary"][key]["df"] if kind == "binary"
                else snap26.get("winner", {}).get("df"))
        if df25 is None or df26 is None:
            continue
        pr = {}
        for tag, df in (("25", df25), ("26", df26)):
            did, uniq = pd.factorize(df["Date"], sort=False)
            pr[tag] = (_prep_binlike(did, len(uniq), df["p"].to_numpy(float),
                                     df["y"].to_numpy(float)), len(uniq))
        s25, l25, b25 = _binlike_boot(pr["25"][0], _draws(pr["25"][1], rng))
        s26, l26, b26 = _binlike_boot(pr["26"][0], _draws(pr["26"][1], rng))
        orl = np.array([np.nanmean([or_lift(l25[i], b25[i]),
                                    or_lift(l26[i], b26[i])]) for i in range(BOOT_B)])
        rec = {"score_lo": _lcb(_combine(s25, s26)), "or_lift_lo": _lcb(orl)}
        (out["binary"] if kind == "binary" else out)[key if kind == "binary" else "winner"] = rec

    # O/U families + their individual lines
    for name in snap25["count"]:
        if name not in snap26["count"]:
            continue
        p25 = _prep_count_family(name, snap25, heads25.get(name), kd25, td25,
                                 art25)
        p26 = _prep_count_family(name, snap26, heads26.get(name), kd26, td26,
                                 art26)
        W25, W26 = _draws(p25["D"], rng), _draws(p26["D"], rng)
        out["count_fam"][name] = {
            "score_lo": _lcb(_combine(_countfam_boot(p25, W25),
                                      _countfam_boot(p26, W26)))}
        for ln in set(p25["lines"]) & set(p26["lines"]):
            lp25, lp26 = p25["lines"][ln], p26["lines"][ln]
            s25, l25, b25 = _binlike_boot(lp25["prep"], W25)
            s26, l26, b26 = _binlike_boot(lp26["prep"], W26)
            orl = np.array([np.nanmean([or_lift(l25[i], b25[i]),
                                        or_lift(l26[i], b26[i])])
                            for i in range(BOOT_B)])
            out["count_line"][(name, ln)] = {
                "score_lo": _lcb(_combine(s25, s26)), "or_lift_lo": _lcb(orl),
                "owned": lp25["owned"]}

    # mean columns
    mm25, mm26 = mean_year(snap25), mean_year(snap26)
    for key in mm25:
        if key not in mm26:
            continue
        df25 = (snap25["count"][key]["df"] if key != "score"
                else snap25["score"]["df"])
        df26 = (snap26["count"][key]["df"] if key != "score"
                else snap26["score"]["df"])
        s25 = _mean_boot(_prep_mean(df25, mm25[key]["rho"]), _draws(
            len(pd.factorize(df25["Date"], sort=False)[1]), rng))
        s26 = _mean_boot(_prep_mean(df26, mm26[key]["rho"]), _draws(
            len(pd.factorize(df26["Date"], sort=False)[1]), rng))
        out["mean"][key] = {"score_lo": _lcb(_combine(s25, s26))}
    return out


# The bootstrap is ~1-2 min; the blue marks need it at SERVE time. Cache the
# result keyed by the snapshot data/frame fingerprints (already stored in
# each snapshot) + the score-construction constants, so it is computed ONCE
# per baseline (whoever runs first — this tool or the day's first predict)
# and reused instantly after. A fingerprint miss recomputes and rewrites.
_BOOT_VERSION = 4          # v4: per-class Lift-led rank weights (2026-07-15)
_BOOT_CACHE = ART / "quality_boot.joblib"


def _boot_key(snap25, snap26):
    return (str(snap25.get("data_fp")), str(snap25.get("frames_fp")),
            str(snap26.get("data_fp")), str(snap26.get("frames_fp")),
            BOOT_B, SCORE_LCB_Q, BOOT_SEED,
            tuple(sorted(BIN_RANK_W.items())),
            tuple(sorted(CNT_RANK_W.items())), _BOOT_VERSION)


def bootstrap_lcb_cached(snap25, snap26, art25, art26):
    key = _boot_key(snap25, snap26)
    try:
        blob = joblib.load(_BOOT_CACHE)
        if blob.get("key") == list(key) or blob.get("key") == key:
            return blob["result"]
    except Exception:
        pass
    result = bootstrap_lcb(snap25, snap26, art25, art26)
    try:
        joblib.dump({"key": key, "result": result}, _BOOT_CACHE)
    except Exception:
        pass
    return result


# ------------------------------------------ blue-mark playbook (for predict)
# The single source of truth for how deep the prediction sheets paint each
# column light blue — computed here, next to the metrics, so the rankings and
# the marks can never disagree. Everything is CI-aware: a column earns depth
# only if its odds-ratio top-pick lift's LOWER bound clears the floor and its
# LCB tier allows it. predict.quality_marks then just paints the top-`depth`
# OVER rows that also clear the informedness floor and the sharp-line veto.
BLUE_OR_TOP, BLUE_OR_DEEP = 1.55, 2.5     # odds-ratio lift gates (on the LB)
# base depth INTERPOLATES linearly between the gates (2026-07-15, user;
# same-day second revision): 5 cells at an OR-lift LB of 1.55 rising to 10
# at 2.5. The old 5-or-10 step at 2.0 doubled a column's depth on a
# hairline LB difference (K > 8.5 at 2.01 painted 10 while K > 4.5-7.5 at
# 1.93-1.99 painted 5 — the thinnest tail line the deepest). The first fix
# ramped to 10 AT 2.0, which pulled every near-2.0 column toward full
# depth and grew the board ~189->227 slots — against the validated
# selectivity lever — so the user stretched the ceiling to 2.5: an LB
# near 2.0 now lands mid-ramp (~7), full 10-deep is reserved for the
# truly exceptional columns (SB/Triple/3+K/2+K-class LBs), and total
# volume sits back where the +19% blue validation was run.
BLUE_N_TOP, BLUE_N_DEEP = 5, 10
BLUE_SLOPE = (0.80, 1.20)                 # calibration-slope sanity gate
# depth cap keyed to the frozen PROB tiers applied to the LCB (v4: the
# DISPLAYED tier cuts on the point Score, but depth reads the tier the
# row's LOWER BOUND would earn — deliberately stricter, so a thin market
# can't buy deep picks on a lucky point estimate): STRONG+ -> 10, SOLID
# -> 7, DECENT -> 5, LOW CEILING -> 4, AVOID -> 0 (2026-07-15, user: blue
# means TRULY recommended — a market whose PROVEN tier is AVOID paints
# nothing, even though its lift gate passed; the old 2-cell floor was the
# least-proven blue on the sheet).
# DERIVED from PROB_TIER_CUTS so a ladder re-anchor can never desync
# the blue-mark depth semantics from the tiers.
BLUE_DEPTH_CAPS = tuple(
    (cut, cap) for (cut, _t), cap in zip(PROB_TIER_CUTS[1:], (10, 7, 5, 4))
) + ((float("-inf"), 0),)
# pitcher serving-column prefix -> count-head key (predict maps its columns
# through the same table)
QUAL_STARTER_KEY = {"pk": "k", "pouts": "outs", "phits": "pha",
                    "pbb": "pbb", "per": "per"}


def _blue_depth(or_lift_lo, slope_ok, score_lo):
    """Blue depth from CI-aware inputs (0 = ineligible): the odds-ratio lift
    LOWER bound must clear BLUE_OR_TOP and the slope gate must pass; the base
    depth ramps linearly from BLUE_N_TOP at the gate to BLUE_N_DEEP at
    BLUE_OR_DEEP (no cliff), then the LCB tier caps it — and an AVOID lower
    bound zeroes it."""
    if not slope_ok or not np.isfinite(or_lift_lo) or or_lift_lo < BLUE_OR_TOP:
        return 0
    frac = min(1.0, (or_lift_lo - BLUE_OR_TOP) / (BLUE_OR_DEEP - BLUE_OR_TOP))
    base = int(round(BLUE_N_TOP + (BLUE_N_DEEP - BLUE_N_TOP) * frac))
    q = score_lo if np.isfinite(score_lo) else float("-inf")
    return min(base, next(c for s, c in BLUE_DEPTH_CAPS if q >= s))


def quality_playbook(snap25=None, snap26=None, art25=None, art26=None):
    """CI-aware blue-mark depth for every paintable column. Returns
    {"binary": {prop_key: PB}, "pitch_line": {(count_key, line): PB}} with
    PB = {depth, base, slope_ok, score_lo, or_lift, or_lift_lo}; `depth` is
    final (0 = paint nothing). Loads snapshots/artifacts itself if not passed.
    Fails soft: if the bootstrap is unavailable the depth falls back to the
    point estimate (still usable, just not CI-shrunk)."""
    if snap25 is None:
        snap25 = joblib.load(ART / "eval_paired_select_2025.joblib")
        snap26 = joblib.load(ART / "eval_paired_2026.joblib")
        art25 = joblib.load(ART / "models_bt.joblib")
        art26 = joblib.load(ART / "models.joblib")
    b25, b26 = binary_year(snap25), binary_year(snap26)
    c25, c26 = count_year(snap25, art25), count_year(snap26, art26)
    try:
        lcb = bootstrap_lcb_cached(snap25, snap26, art25, art26)
    except Exception:
        lcb = {"binary": {}, "count_fam": {}, "count_line": {}}

    def _slope_ok(*ms):
        s = np.nanmean([m.get("slope", np.nan) for m in ms])
        return bool(np.isfinite(s) and BLUE_SLOPE[0] <= s <= BLUE_SLOPE[1])

    def _orl(m25, m26):
        return float(np.nanmean([or_lift(m25.get("lift", np.nan),
                                         m25.get("base", np.nan)),
                                 or_lift(m26.get("lift", np.nan),
                                         m26.get("base", np.nan))]))

    def _base(m25, m26):
        return float(np.nanmean([m25.get("base", np.nan),
                                 m26.get("base", np.nan)]))

    def _fin(fn, m25, m26):
        return final_score(fn(m25), fn(m26))

    pb = {"binary": {}, "pitch_line": {}}
    for key in b25:
        if key not in b26:
            continue
        m25, m26 = b25[key], b26[key]
        rec = lcb.get("binary", {}).get(key, {})
        orl_lo = rec.get("or_lift_lo", np.nan)
        orl_pt = _orl(m25, m26)
        orl_lo = orl_lo if np.isfinite(orl_lo) else orl_pt
        s_lo = rec.get("score_lo", np.nan)
        s_lo = s_lo if np.isfinite(s_lo) else _fin(binary_score, m25, m26)
        ok = _slope_ok(m25, m26)
        pb["binary"][key] = {
            "depth": _blue_depth(orl_lo, ok, s_lo), "base": _base(m25, m26),
            "slope_ok": ok, "score_lo": s_lo, "or_lift": orl_pt,
            "or_lift_lo": orl_lo}
    for skey, ckey in QUAL_STARTER_KEY.items():
        if ckey not in c25 or ckey not in c26:
            continue
        f25, f26 = c25[ckey], c26[ckey]
        fam_lo = lcb.get("count_fam", {}).get(ckey, {}).get("score_lo", np.nan)
        fam_lo = fam_lo if np.isfinite(fam_lo) else _fin(count_score, f25, f26)
        for ln in set(f25.get("lines", {})) & set(f26.get("lines", {})):
            l25, l26 = f25["lines"][ln], f26["lines"][ln]
            rec = lcb.get("count_line", {}).get((ckey, ln), {})
            line_lo = rec.get("score_lo", np.nan)
            line_lo = (line_lo if np.isfinite(line_lo)
                       else _fin(binary_score, l25, l26))
            orl_lo = rec.get("or_lift_lo", np.nan)
            orl_pt = _orl(l25, l26)
            orl_lo = orl_lo if np.isfinite(orl_lo) else orl_pt
            q_lo = float(np.nanmean([fam_lo, line_lo]))   # both tables vote
            ok = _slope_ok(l25, l26)
            pb["pitch_line"][(ckey, ln)] = {
                "depth": _blue_depth(orl_lo, ok, q_lo), "base": _base(l25, l26),
                "slope_ok": ok, "score_lo": q_lo, "or_lift": orl_pt,
                "or_lift_lo": orl_lo}
    return pb


# The board SORT (user 2026-07-15, v4): TIER first, and the tier is cut on
# the SCORE ITSELF ("score and tier should be closely connected — tiers are
# separated by score"), so the board reads as contiguous Score bands.
# Within a tier: Score descending (the same number the tier is cut from),
# then the ranked diagnostics as pre-declared tie-breakers in the user's
# order (Lift, Edge%, slope closest to 1, |Int|, Disp, Brier, ECE, |Bias|,
# MAE, AUC, raw Top10%, LogLoss, Acc). NaNs sort last on every key.
RANK_SORT = (("Lift", False), ("Edge%", False), ("_slope_dev", True),
             ("_int_dev", True), ("Disp", True), ("Brier", True),
             ("ECE", True), ("_bias_dev", True), ("MAE", True),
             ("AUC", False), ("Top10%", False), ("LogLoss", True),
             ("Acc", False))


def _rank_and_tier(df):
    """Order rows into class blocks (probability markets, then means, then
    informational), tier WITHIN class on the frozen ladders — cut on the
    SCORE itself (v4) so tiers are contiguous Score bands — and sort each
    block by TIER, then Score, then the ranked-diagnostic tie-breakers —
    a mean Score and a binary Score are never compared."""
    crank = {"Prob": 0, "Mean": 1}
    d = df.copy()
    d["_c"] = d["Class"].map(lambda c: crank.get(c, 2))
    d["Tier"] = [tier_of(s, MEAN_TIER_CUTS) if c == "Mean"
                 else tier_of(s, PROB_TIER_CUTS) if c == "Prob" else "-"
                 for c, s in zip(d["Class"],
                                 pd.to_numeric(d["Score"],
                                               errors="coerce"))]
    d["_t"] = d["Tier"].map(
        lambda t: int(t[0]) if str(t)[:1].isdigit() else 9)

    def _col(name):
        return (pd.to_numeric(d[name], errors="coerce") if name in d
                else pd.Series(np.nan, index=d.index))

    d["_s"] = _col("Score").fillna(-np.inf)
    d["_slope_dev"] = (_col("Slope") - 1).abs()
    d["_int_dev"] = _col("Int").abs()
    d["_bias_dev"] = _col("Bias").abs()
    keys, asc = ["_c", "_t", "_s"], [True, True, False]
    for i, (col, ascending) in enumerate(RANK_SORT):
        v = _col(col)
        d[f"_k{i}"] = v.fillna(np.inf if ascending else -np.inf)
        keys.append(f"_k{i}")
        asc.append(ascending)
    d = (d.sort_values(keys, ascending=asc)
           .drop(columns=[c for c in d.columns if c.startswith("_")])
           .reset_index(drop=True))
    d.insert(0, "#", range(1, len(d) + 1))
    return d


def build_table():
    snap25 = joblib.load(ART / "eval_paired_select_2025.joblib")
    snap26 = joblib.load(ART / "eval_paired_2026.joblib")
    # audit #6 (user decision 07-15): the 2026 snapshot refreshes only on a
    # DELIBERATE `evaluate_deep --confirm --set-baseline` (the daily job no
    # longer touches it), so it legitimately ages between confirms.
    try:
        age = ((ART / "eval_paired_select_2025.joblib").stat().st_mtime
               - (ART / "eval_paired_2026.joblib").stat().st_mtime) / 86400.0
        if age > 2:
            print(f"note: 2026 snapshot is ~{age:.0f} day(s) older than the "
                  f"2025 one — expected under manual-confirm-only (refresh "
                  f"via evaluate_deep --confirm --set-baseline after a "
                  f"finished change).")
    except OSError:
        pass
    art25 = joblib.load(ART / "models_bt.joblib")
    art26 = joblib.load(ART / "models.joblib")

    b25, b26 = binary_year(snap25), binary_year(snap26)
    c25 = count_year(snap25, art25)
    c26 = count_year(snap26, art26)
    lcb = bootstrap_lcb_cached(snap25, snap26, art25, art26)
    # the Blue column: the paint depth quality_playbook hands predict.py
    # (same snapshots, same cached bootstrap), shown beside Score/Score_lo
    # so the paint rule is legible on the board instead of implicit
    playbook = quality_playbook(snap25, snap26, art25, art26)

    def blue_of(kind, key):
        rec = playbook.get(kind, {}).get(key)
        return float(rec["depth"]) if rec else np.nan

    def blue_fam(ckey):
        """Family rows paint per line — show the family's deepest line."""
        ds = [r["depth"] for (k, _ln), r in playbook["pitch_line"].items()
              if k == ckey]
        return float(max(ds)) if ds else np.nan

    def bin_lo(key):
        return lcb.get("binary", {}).get(key, {}).get("score_lo", np.nan)

    def _avg(m25, m26, key, scale=1.0):
        vals = [v for v in (m25.get(key, np.nan), m26.get(key, np.nan))
                if np.isfinite(v)]
        return scale * float(np.mean(vals)) if vals else np.nan

    def _diag_cols(m25, m26):
        """The ranked-diagnostic display columns + their child/context
        columns (user 07-15: children sit beside their parents; Base%,
        MeanAct, MeanPred are unranked context), averaged over the years,
        in the user's priority order (count-head superset; columns a row's
        metrics don't carry stay empty)."""
        return {
            "Lift": _avg(m25, m26, "lift"),
            "Edge%": _avg(m25, m26, "rel"),
            "Slope": _avg(m25, m26, "slope"),
            "Int": _avg(m25, m26, "int"),
            "Disp": _avg(m25, m26, "disp_ratio"),
            "DispCal": _avg(m25, m26, "priced_disp"),
            "DispObs": _avg(m25, m26, "disp"),
            "Brier": _avg(m25, m26, "brier"),
            "BrierBase": _avg(m25, m26, "brier_base"),
            "ECE": _avg(m25, m26, "ece"),
            "Bias": _avg(m25, m26, "bias"),
            "MAE": _avg(m25, m26, "mae"),
            "MAEBase": _avg(m25, m26, "mae_base"),
            "MAEGain": _avg(m25, m26, "mae_gain"),
            "AUC": _avg(m25, m26, "auc"),
            "Base%": _avg(m25, m26, "base", 100.0),
            "Top10%": _avg(m25, m26, "top10", 100.0),
            "LogLoss": _avg(m25, m26, "ll"),
            "LLBase": _avg(m25, m26, "ll_base"),
            "MeanAct": _avg(m25, m26, "mean_actual"),
            "MeanPred": _avg(m25, m26, "mean_pred"),
            "Acc": _avg(m25, m26, "acc"),
        }

    _NAN_DIAGS = {k: np.nan for k in (
        "Lift", "Edge%", "Slope", "Int", "Disp", "DispCal", "DispObs",
        "Brier", "BrierBase", "ECE", "Bias", "MAE", "MAEBase", "MAEGain",
        "AUC", "Base%", "Top10%", "LogLoss", "LLBase", "MeanAct",
        "MeanPred", "Acc")}

    rows = []
    for key, name in BIN_NAMES.items():
        m25, m26 = b25[key], b26[key]
        rows.append({
            "Market": name, "Key": key, "Class": "Prob",
            "Score": final_score(binary_score(m25), binary_score(m26)),
            "Score_lo": bin_lo(key),
            "Blue": blue_of("binary", key),
            "S25": binary_score(m25), "S26": binary_score(m26),
            **_diag_cols(m25, m26),
            "Notes": play_note("bin", m25, m26),
        })
    # line FAMILIES on the headline sheet: the markets whose displayed
    # columns ARE the "> x" probabilities. Batter x-columns are displayed
    # as MEANS, so they rank as mean rows below (their deep lines stay on
    # the Lines sheet).
    for key, name in {**PITCHER_CNT, **GAME_CNT}.items():
        m25, m26 = c25[key], c26[key]
        rows.append({
            "Market": name, "Key": key, "Class": "Prob",
            "Score": final_score(count_score(m25), count_score(m26)),
            "Score_lo": lcb.get("count_fam", {}).get(key, {}).get("score_lo", np.nan),
            "Blue": blue_fam(key),
            "S25": count_score(m25), "S26": count_score(m26),
            **_diag_cols(m25, m26),
            "Notes": play_note("cnt", m25, m26),
        })
    # displayed MEAN columns (xK, xOuts, ..., xSO, Total Runs, Away/Home
    # Score): graded on mean quality — MAE beat over the no-skill constant
    # (Edge%), Spearman rank correlation, bias trust. Their OWN class and
    # ladder (MEAN_TIER_CUTS) — never compared to the probability rows.
    mm25, mm26 = mean_year(snap25), mean_year(snap26)
    mean_names = dict(MEAN_MARKETS)
    mean_names["score"] = "Away/Home Score"
    for key, name in mean_names.items():
        if key not in mm25 or key not in mm26:
            if key == "score":
                rows.append({
                    "Market": name, "Key": "score", "Class": "Mean",
                    "Score": np.nan, "Score_lo": np.nan,
                    "S25": np.nan, "S26": np.nan, **_NAN_DIAGS,
                    "Notes": "snapshot predates per-team score rows - "
                             "re-run --set-baseline (both suites) to grade"})
            continue
        m25, m26 = mm25[key], mm26[key]
        bias = (m25["bias"] + m26["bias"]) / 2
        rho = (m25["rho"] + m26["rho"]) / 2
        note = (f"over-predicts +{bias:.2f} -> lean unders" if bias > 0.1
                else f"under-predicts {bias:.2f} -> lean overs" if bias < -0.1
                else "unbiased")
        note += f"; ranks real outcomes rho {rho:.2f}"
        if key in BATTER_X:
            note += ("; context mean - its half-point lines are sold by the "
                     "binary columns, deep lines graded on the Lines sheet")
        rows.append({
            "Market": name, "Key": f"x:{key}", "Class": "Mean",
            "Score": final_score(mean_score(m25), mean_score(m26)),
            "Score_lo": lcb.get("mean", {}).get(key, {}).get("score_lo", np.nan),
            "S25": mean_score(m25), "S26": mean_score(m26),
            **_NAN_DIAGS,
            "Bias": bias, "DispObs": (m25["disp"] + m26["disp"]) / 2,
            "MAE": _avg(m25, m26, "mae"),
            "MAEBase": _avg(m25, m26, "mae_base"),
            "MAEGain": _avg(m25, m26, "mae_gain"),
            "MeanAct": _avg(m25, m26, "mean_actual"),
            "MeanPred": _avg(m25, m26, "mean_pred"),
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "Notes": note,
        })
    # Lineup HRs is arithmetic on the Batter HR probabilities (no model or
    # snapshot of its own) — shown so nothing displayed is missing
    rows.append({
        "Market": "Lineup HRs", "Key": "lineup_hr", "Class": "-",
        "Score": np.nan, "Score_lo": np.nan,
        "S25": np.nan, "S26": np.nan, **_NAN_DIAGS,
        "Notes": "sum of the lineup's Batter HR probabilities - quality is "
                 "graded on the Batter HR row"})
    # winner: graded like any probability column when the snapshot carries
    # its rows (Win Prob IS a served %). The McNemar caution — no proven
    # SIDE-PICKING edge vs always-home — stays as usage guidance in Notes;
    # it is a statement about moneyline betting, not probability quality.
    w25, w26 = winner_year(snap25), winner_year(snap26)
    if w25 is not None and w26 is not None:
        rows.append({
            "Market": "Game Winner (Win Prob)", "Key": "winner", "Class": "Prob",
            "Score": final_score(binary_score(w25), binary_score(w26)),
            "Score_lo": lcb.get("winner", {}).get("score_lo", np.nan),
            "S25": binary_score(w25), "S26": binary_score(w26),
            **_diag_cols(w25, w26),
            "Notes": play_note("bin", w25, w26)
                     + "; no proven side edge vs always-home (McNemar "
                       "n.s.) -> win% quality, not a moneyline endorsement",
        })
    else:
        rows.append({"Market": "Game Winner (Win Prob)",
                     "Key": "winner", "Class": "Prob",
                     "Score": np.nan, "Score_lo": np.nan,
                     "S25": np.nan, "S26": np.nan, **_NAN_DIAGS,
                     "Notes": "snapshot predates winner rows — re-run "
                              "evaluate_deep --set-baseline (both suites) "
                              "to grade this column"})
    df = _rank_and_tier(pd.DataFrame(rows))

    # ---- per-line breakout: every individual priced column, own grade.
    # Binary props ARE single lines (their family row repeats here for a
    # complete per-column view); count families split into their lines.
    # Trust varies BY LINE (tails run less calibrated than mid lines) and
    # the family average hides exactly that.
    line_rows = []
    for key, name in BIN_NAMES.items():
        m25, m26 = b25[key], b26[key]
        line_rows.append({
            "Market": name, "Class": "Prob",
            "Score": final_score(binary_score(m25), binary_score(m26)),
            "Score_lo": bin_lo(key),
            "Blue": blue_of("binary", key),
            "S25": binary_score(m25), "S26": binary_score(m26),
            **_diag_cols(m25, m26),
            "Sold": "this column",
        })
    for key, name in CNT_MARKETS.items():
        l25, l26 = c25[key]["lines"], c26[key]["lines"]
        stem = name.replace(" > x", "")
        for ln in sorted(set(l25) & set(l26)):
            m25, m26 = l25[ln], l26[ln]
            line_rows.append({
                "Market": f"{stem} > {ln}", "Class": "Prob",
                "Score": final_score(binary_score(m25), binary_score(m26)),
                "Score_lo": lcb.get("count_line", {}).get(
                    (key, ln), {}).get("score_lo", np.nan),
                "Blue": blue_of("pitch_line", (key, ln)),
                "S25": binary_score(m25), "S26": binary_score(m26),
                **_diag_cols(m25, m26),
                "Sold": ("binary column" if m25["owned"] else "this column"),
            })
    if w25 is not None and w26 is not None:
        line_rows.append({
            "Market": "Game Winner (Win Prob)", "Class": "Prob",
            "Score": final_score(binary_score(w25), binary_score(w26)),
            "Score_lo": lcb.get("winner", {}).get("score_lo", np.nan),
            "S25": binary_score(w25), "S26": binary_score(w26),
            **_diag_cols(w25, w26),
            "Sold": "this column",
        })
    # the displayed mean columns, on the per-column sheet too (the
    # probability diagnostics are blank for means; Edge% is the MAE beat
    # over the no-skill constant)
    for key, name in mean_names.items():
        if key not in mm25 or key not in mm26:
            continue
        m25, m26 = mm25[key], mm26[key]
        line_rows.append({
            "Market": name, "Class": "Mean",
            "Score": final_score(mean_score(m25), mean_score(m26)),
            "Score_lo": lcb.get("mean", {}).get(key, {}).get("score_lo", np.nan),
            "S25": mean_score(m25), "S26": mean_score(m26),
            **_NAN_DIAGS,
            "Bias": (m25["bias"] + m26["bias"]) / 2,
            "DispObs": (m25["disp"] + m26["disp"]) / 2,
            "MAE": _avg(m25, m26, "mae"),
            "MAEBase": _avg(m25, m26, "mae_base"),
            "MAEGain": _avg(m25, m26, "mae_gain"),
            "MeanAct": _avg(m25, m26, "mean_actual"),
            "MeanPred": _avg(m25, m26, "mean_pred"),
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "Sold": "mean (context)",
        })
    ldf = _rank_and_tier(pd.DataFrame(line_rows))
    # column order = the user's diagnostic ranking (v4), left to right,
    # child columns beside their parents and the unranked context columns
    # (Base%, MeanAct, MeanPred) in the user's positions — every column is
    # present for every row even where a row leaves it empty
    _DIAG_ORDER = ["Lift", "Edge%", "Slope", "Int", "Disp", "DispCal",
                   "DispObs", "Brier", "BrierBase", "ECE", "Bias", "MAE",
                   "MAEBase", "MAEGain", "AUC", "Base%", "Top10%",
                   "LogLoss", "LLBase", "MeanAct", "MeanPred", "Acc"]
    ldf = ldf[["#", "Market", "Class", "Tier", "Score", "Score_lo", "Blue",
               "S25", "S26"] + _DIAG_ORDER + ["Sold"]]

    return df[["#", "Market", "Key", "Class", "Tier", "Score", "Score_lo",
               "Blue", "S25", "S26"] + _DIAG_ORDER + ["Notes"]], ldf


LEGEND = [
    ("Sort order", "TIER first, and the tier is cut on the SCORE itself "
     "(user, 2026-07-15 v4: 'score and tier should be closely connected - "
     "tiers are separated by score'), so the board reads as contiguous "
     "Score bands. Within tier: Score descending, then the ranked "
     "diagnostics as pre-declared tie-breakers. Column order follows the "
     "ranking, left to right, with each parent's child columns beside it "
     "and the unranked context columns in the user's positions (07-15): "
     "Lift, Edge%, Slope, Int, Disp (+DispCal/DispObs), Brier "
     "(+BrierBase), ECE, Bias, MAE (+MAEBase/MAEGain), AUC, Base%, "
     "Top10%, LogLoss (+LLBase), MeanAct, MeanPred, Acc — every column is "
     "present for every row even where a row leaves it empty."),
    ("Score", "The rank-WEIGHTED composite of the class's ranked "
     "diagnostics on a 0-100 scale, weights linear in rank. O/U count "
     "heads use the full 13-metric ranking (Lift 14%, Edge% 13%, Slope "
     "12%, Int 11%, Disp 10%, Brier 9%, ECE 8%, Bias 7%, MAE 5%, AUC 4%, "
     "Top10% 3%, LogLoss 2%, Acc 1%); binary heads have no Disp/Bias, so "
     "they use the 11-metric EFFECTIVE order (Lift 17%, Edge% 15%, Slope "
     "14%, Int 12%, Brier 11%, ECE 9%, AUC 8%, Top10% 6%, LogLoss 5%, "
     "MAE 3%, Acc 2%). Computed per held-out test year, then mean(S25, "
     "S26) minus 0.25x their gap (a market that performs in only one year "
     "is down-ranked for exactly that). Lift enters in its base-rate-fair "
     "odds-ratio form; raw Top10% participates at its own low rank; "
     "standalone log loss reads through its base-relative form, which IS "
     "the edge. No scraped odds anywhere. TIERS ARE CUT ON THIS COLUMN."),
    ("Score_lo", "The day-block-bootstrap LOWER bound of Score (10th "
     "percentile over resampled days) - the SAME bootstrap philosophy as the "
     "model's paired accept bar. A thin market (SB, a deep line) earns a wide "
     "CI and a lower Score_lo. v4: the displayed tier follows Score, but "
     "BLUE-MARK PICK DEPTH still caps on this column - a thin market can't "
     "buy deep picks on a lucky point estimate."),
    ("Blue", "How many cells the prediction workbook may paint light blue "
     "for this market today - the quality_playbook depth predict.py "
     "applies, shown here so the paint rule is legible instead of "
     "implicit. Eligibility: the bootstrap LOWER bound of odds-ratio "
     "top-pick lift must clear 1.55 and the calibration slope must sit in "
     "0.80-1.20. Base depth then ramps LINEARLY from 5 at a lift LB of "
     "1.55 to 10 at 2.5 (2026-07-15: no more 5-or-10 cliff at 2.0, and "
     "full 10-deep is reserved for truly exceptional selection power - "
     "an LB near 2.0 lands mid-ramp at ~7), and is "
     "capped by the tier Score_lo earns on the frozen ladder: STRONG+ 10, "
     "SOLID 7, DECENT 5, LOW CEILING 4, AVOID 0 (2026-07-15: an AVOID "
     "lower bound paints NOTHING - blue means truly recommended, and a "
     "market that can't prove more than AVOID isn't). 0 = ineligible; "
     "blank = not a paintable column (means, game totals, winner). O/U "
     "family rows show their DEEPEST line; each line's own depth is on "
     "the Lines sheet. At serve time the informedness floor and the "
     "sharp-line veto still trim below this number."),
    ("Class / block", "Prob = probability markets (binaries, O/U lines and "
     "families, the winner); Mean = the displayed expected-count columns. "
     "They sit on DIFFERENT scales, so each is its own block with its OWN "
     "tier ladder - a mean Score and a binary Score are never compared or "
     "interleaved."),
    ("S25 / S26", "The same score computed on each test year alone - "
     "agreement between them means the ranking is stable, not one-year "
     "noise. The gap directly reduces the final Score."),
    ("Lift", "RANKED #1 (both classes). Top10% over the base rate - the "
     "day's best picks, expressed as selection power. Scored in its "
     "base-rate-fair odds-ratio form (ADJUDICATED 2026-07-15: read Lift, "
     "not raw Top10%, when comparing across markets), and (with a "
     "bootstrap lower bound) it sizes blue-mark depth on the prediction "
     "sheets."),
    ("Edge%", "RANKED #2 (both classes). How much better the column "
     "prices the event than the base-rate guess (relative log-loss beat; "
     "O/U columns averaged across every line priced). Held-out years "
     "only."),
    ("Slope", "RANKED #3 (both classes). Calibration slope: refit of "
     "reality on the served logit. 1.0 = probabilities move exactly as "
     "much as they should; below 1 = overconfident (shade extremes toward "
     "the middle), above 1 = underconfident (extremes better than "
     "stated)."),
    ("Int", "RANKED #4 (both classes). Calibration intercept (log-odds): "
     "the systematic shift left after the slope. 0 = level; positive = "
     "events happen more often than stated across the board."),
    ("Disp", "RANKED #5 (O/U count heads; binaries don't have it). The "
     "FOLDED number: observed dispersion over what the P(over) pricing "
     "ASSUMES, 1.0 ideal (NB heads price extra variance already); above "
     "1 = real tails wilder than priced -> don't trust extreme-line "
     "probabilities. Its children show the two raw values: DispCal = the "
     "dispersion the pricing assumes (cal-year model), DispObs = observed "
     "error variance/mean (mean rows carry DispObs only — a mean doesn't "
     "price tails). Promoted in v4 from an unranked trust modifier to a "
     "full ranked term."),
    ("Brier", "RANKED #5 binary / #6 count. Mean squared error of the "
     "stated probability (lower better); scored via its relative beat "
     "over BrierBase, its child column = the Brier a flat base-rate "
     "forecast would score."),
    ("ECE", "RANKED #6 binary / #7 count. Calibration: average gap "
     "between stated probability and reality. 0 = perfect. Inside Score "
     "it is scaled by the market's base-rate variance (an ECE of .006 is "
     "far worse on an 11% event than on a 60% one); the column shows the "
     "raw value."),
    ("Bias", "RANKED #8 (O/U count heads; binaries don't have it). "
     "Predicted count minus actual, on average. Positive = model "
     "over-predicts -> lean unders. Promoted in v4 from an unranked trust "
     "modifier to a full ranked term."),
    ("MAE", "RANKED #10 binary / #9 count. Mean absolute error - for "
     "binaries, of the stated probability against the 0/1 outcome; for "
     "O/U and mean rows, of the predicted count. Its children fold into "
     "the score: MAEBase = the no-skill constant's MAE (binaries: the "
     "base-rate forecast; counts: always guessing the mean), MAEGain = "
     "MAEBase - MAE, and the Score reads the gain relative to the "
     "baseline (the same number the mean rows lead with)."),
    ("AUC", "RANKED #7 binary / #10 count. Ranking skill (0.5 = coin "
     "flip). Can it put the players who DID do it above the ones who "
     "didn't? O/U columns: computed per line through the actual quoted "
     "P(over), averaged across the column's line family - comparable to "
     "the binary rows."),
    ("Top10%", "RANKED #8 binary / #11 count. The RAW daily top-10 hit "
     "rate - how often the column's ten best picks of a day actually "
     "happen. Not base-rate-fair across markets: a 61%-base market posts "
     "a high Top10% with little skill - which is exactly why the user "
     "ranked it low and Lift first. Pitcher/game families also draw from "
     "a ~30-candidate daily pool, so their top-10 is the top THIRD (vs "
     "the top ~4% of ~270 batter slots). ADJUDICATED 2026-07-15 (user): "
     "pick depth stays uniform; read Lift, not raw Top10%, when comparing "
     "across markets. O/U columns: measured on the first line the column "
     "uniquely prices."),
    ("LogLoss", "RANKED #9 binary / #12 count. The column's standalone "
     "log loss (lower better) - only cross-market comparable through its "
     "base-relative form, which IS Edge%, so the Score reads one signal "
     "at both weights. Its child LLBase = the log loss of always "
     "guessing the base rate; Edge% is exactly the relative gap between "
     "the two."),
    ("Acc", "RANKED #11 binary / #13 count (last, per the user's "
     "ranking). Plain accuracy of the >=50% call; scored as the beat over "
     "the trivial always-majority pick - near-zero for low-base props by "
     "construction (always-no wins)."),
    ("Base% / MeanAct / MeanPred", "UNRANKED CONTEXT columns (user call "
     "07-15: they appear in the ranked positions but do not score). "
     "Base% = how often the event actually happens - the blind-bet rate "
     "Top10% must be read against (O/U rows: the flagship line's over "
     "rate). MeanAct = the mean actual outcome (binaries: identical to "
     "the base rate; counts: the average count). MeanPred = the mean "
     "prediction; MeanPred - MeanAct is exactly the Bias the count "
     "Score already ranks at #8, which is why these don't score "
     "separately."),
    ("Tier", "1 ELITE / 2 STRONG: trust and act on these. 3 SOLID: usable "
     "with the noted caveat. 4 DECENT: a real, playable edge - shallower, but "
     "the picks are honest. 5 LOW CEILING: not a weak model - a low ceiling. "
     "The event is ~4 coin flips with barely any spread between players, so "
     "even a PERFECT forecaster could only separate them a little: a "
     "leave-one-out oracle that knows each batter's true season rate scores no "
     "better than these heads do. The thin Score is a fact about the market, "
     "not a defect. Play the top of the list, size for the variance. 6 AVOID: "
     "don't act - either genuinely below the achievable bar (winner, total sit "
     "under the market ceiling) or too thin to separate at all. Cut from "
     "the SCORE itself (v4: tiers are contiguous Score bands) on FROZEN "
     "semantic anchors WITHIN each class - a market's tier depends only on "
     "its own edge/calibration, never on how other markets moved in a "
     "retrain, and probability and mean rows use separate ladders. The CI "
     "still bites where it matters: blue-mark depth caps on Score_lo."),
    ("Lines sheet", "Every individual priced column graded on its own — "
     "binary props are single lines (repeated from Rankings for a complete "
     "view); count families split into their lines, each with its own "
     "Score/Tier. 'Sold: binary column' = the workbook sells that number "
     "from the binary row; the x-column's quote is shown for completeness. "
     "Base% = how often the over actually hits."),
    ("Mean columns", "The displayed expected-count columns (xK, xOuts, "
     "xHits, xBB, xER, xSO, xTB, xHRR, Total Runs, Away/Home Score) are "
     "separate predictions from the '> x' probabilities, so they get their "
     "own rows: Edge% = MAE beat over a no-skill constant guess, MAE/Bias/"
     "Disp show the raw mean-quality numbers, Notes carry the Spearman "
     "rank correlation (the AUC analog for counts) and bias lean; AUC/ECE/"
     "Slope/Lift are probability diagnostics and stay blank. They are the Mean CLASS: their own block, their own tier "
     "ladder, never scored against a probability row. Lineup HRs is "
     "arithmetic on Batter HR - graded there."),
]


def save_excel(df, ldf, path):
    """Write the rankings workbook (Rankings + Lines + Legend), styled like
    the prediction workbooks via predict._polish."""
    _ND = [("Score", 0), ("Score_lo", 0), ("Blue", 0), ("S25", 0), ("S26", 0),
           ("Base%", 1), ("Lift", 2), ("Edge%", 2), ("Slope", 2),
           ("Int", 2), ("Disp", 2), ("DispCal", 2), ("DispObs", 2),
           ("Brier", 4), ("BrierBase", 4), ("ECE", 4), ("Bias", 2),
           ("MAE", 3), ("MAEBase", 3), ("MAEGain", 3), ("AUC", 3),
           ("Top10%", 1), ("LogLoss", 4), ("LLBase", 4), ("MeanAct", 2),
           ("MeanPred", 2), ("Acc", 3)]
    xl = df.copy()
    for c, nd in _ND:
        if c in xl:
            xl[c] = xl[c].round(nd)
    xll = ldf.copy()
    for c, nd in _ND:
        if c in xll:
            xll[c] = xll[c].round(nd)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        xl.to_excel(xw, sheet_name="Rankings", index=False)
        xll.to_excel(xw, sheet_name="Lines", index=False)
        pd.DataFrame(LEGEND, columns=["Term", "Meaning"]).to_excel(
            xw, sheet_name="Legend", index=False)
    P._polish(path)


def warm_cache():
    """Compute + cache the day-block bootstrap and exit — no workbook.

    The daily job calls this right after its two --set-baseline runs, which
    rewrite the paired snapshots and therefore invalidate quality_boot.joblib
    (it is keyed by their fingerprints). Without it the FIRST predict of the
    day would pay the ~60s bootstrap before it could paint blue marks. Writing
    no Excel is deliberate: the workbook save would raise PermissionError if
    PROP_RANKINGS.xlsx happened to be open, and a perf nicety must never be
    able to fail the nightly run."""
    snap25 = joblib.load(ART / "eval_paired_select_2025.joblib")
    snap26 = joblib.load(ART / "eval_paired_2026.joblib")
    art25 = joblib.load(ART / "models_bt.joblib")
    art26 = joblib.load(ART / "models.joblib")
    bootstrap_lcb_cached(snap25, snap26, art25, art26)
    print(f"quality bootstrap cached -> {_BOOT_CACHE}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", metavar="FILE",
                    default=str(Path(__file__).resolve().parent
                                / "PROP_RANKINGS.xlsx"),
                    help="output Excel file "
                         "(default: Tools/PROP_RANKINGS.xlsx)")
    ap.add_argument("--warm-cache", action="store_true",
                    help="only compute + cache the bootstrap (no workbook); "
                         "the daily job runs this after --set-baseline so the "
                         "first predict of the day is instant")
    args = ap.parse_args()

    if args.warm_cache:
        warm_cache()
        return

    df, ldf = build_table()

    _FMT = {"Score": ".0f", "Score_lo": ".0f", "Blue": ".0f", "S25": ".0f",
            "S26": ".0f",
            "Base%": ".1f", "Lift": ".2f", "Edge%": ".2f", "Slope": ".2f",
            "Int": "+.2f", "Disp": ".2f", "DispCal": ".2f",
            "DispObs": ".2f", "Brier": ".4f", "BrierBase": ".4f",
            "ECE": ".4f", "Bias": "+.2f", "MAE": ".3f", "MAEBase": ".3f",
            "MAEGain": ".3f", "AUC": ".3f", "Top10%": ".1f",
            "LogLoss": ".4f", "LLBase": ".4f", "MeanAct": ".2f",
            "MeanPred": ".2f", "Acc": ".3f"}

    def _fmt(frame):
        o = frame.copy()
        for c, f in _FMT.items():
            if c in o:
                o[c] = o[c].map(
                    lambda v, f=f: format(v, f) if pd.notna(v) else "-")
        return o

    print("\n=== Prediction-column quality rankings — held-out test years "
          "2025 + 2026, internal measurements only ===\n")
    print("  KEY (each column, in the ranked order they appear; ranks "
          "shown binary/count):")
    print("  Tier     trust band, cut on SCORE — THE SORT KEY: 1 ELITE "
          "bet it ... 6 AVOID don't; Prob and Mean ladders are separate")
    print("  Score    rank-weighted composite of the ranked diagnostics, "
          "0-100 | Score_lo its bootstrap lower bound | S25/S26 per year")
    print("  Blue     today's blue-mark paint depth (lift-LB ramp 5->10, "
          "capped by Score_lo tier; AVOID lower bound = 0; blank = not "
          "paintable; families show their deepest line)")
    print("  Lift     [#1/#1] Top10% / base rate (1.0x = no better than "
          "betting blind); scored in its base-rate-fair odds-ratio form")
    print("  Edge%    [#2/#2] relative log-loss beat over always guessing "
          "the base rate (higher better)")
    print("  Slope    [#3/#3] calibration slope: 1.00 = stated "
          "probabilities move exactly as much as reality (<1 overconfident)")
    print("  Int      [#4/#4] calibration intercept (log-odds): 0 = level "
          "overall; positive = events happen more often than stated")
    print("  Disp     [-/#5] O/U rows: observed / priced dispersion, 1.0 "
          "ideal (its children: DispCal = what the pricing assumes,")
    print("           DispObs = observed error variance/mean; mean rows "
          "carry DispObs only)")
    print("  Brier    [#5/#6] mean squared error of the stated probability "
          "(lower better; BrierBase = the base-rate forecast's Brier —")
    print("           the Score reads Brier's beat over it)")
    print("  ECE      [#6/#7] average |stated% - actual%| across deciles "
          "(0 = perfectly calibrated)")
    print("  Bias     [-/#8] O/U + mean rows: predicted minus actual count "
          "(positive = over-predicts -> lean unders)")
    print("  MAE      [#10/#9] mean absolute error (binaries: of the "
          "stated probability; O/U + mean rows: of the predicted count);")
    print("           MAEBase = the no-skill constant's MAE, MAEGain = "
          "MAEBase - MAE — the Score reads the gain over the baseline")
    print("  AUC      [#7/#10] ranking skill: chance a random yes-case "
          "outranks a random no-case (0.5 = coin flip)")
    print("  Base%    context (unranked): how often the event actually "
          "happens — the blind-bet rate Top10% must be read against")
    print("  Top10%   [#8/#11] raw daily top-10 hit rate — base-inflated "
          "across markets, which is why it ranks low and Lift ranks first")
    print("  LogLoss  [#9/#12] standalone log loss (lower better; LLBase = "
          "the base-rate guess's log loss — their relative gap IS Edge%)")
    print("  MeanAct  context (unranked): mean actual outcome (binaries: "
          "= base rate; counts: the average count)")
    print("  MeanPred context (unranked): mean prediction — MeanPred - "
          "MeanAct is the Bias the count Score already ranks")
    print("  Acc      [#11/#13] accuracy of the >=50% call (sits near the "
          "base rate on low-base props by construction)")
    print()
    print(_fmt(df).to_string(index=False))
    print("\n  SORT (2026-07-15, v4): TIER first — cut on the SCORE itself, "
          "so tiers are contiguous Score bands — then Score desc, then the "
          "ranked diagnostics as tie-breakers. Column order = the ranking.")
    print("  Score = the rank-WEIGHTED composite of the class's ranked "
          "diagnostics — binaries use the 11-metric effective order, O/U "
          "count heads the full 13 (Lift in base-rate-fair odds-ratio")
    print("  form; LogLoss reads through its base-relative form = Edge%), "
          "per year, then mean(S25, S26) - 0.25x|S25 - S26| (two-year "
          "stability haircut).")
    print("  Score_lo is the day-block-bootstrap lower bound — it no longer "
          "cuts the tier (v4: tiers follow Score) but still caps blue-mark "
          "pick depth, so thin markets can't buy depth on luck.")
    print("  Probability rows (Prob) and expected-count means (Mean) rank "
          "on separate ladders and never share a cut (means score on their "
          "MAE-led mean-quality composite).")
    print("  No scraped odds or market prices enter this table; the "
          "market test is Section 9 of evaluate_deep.py.")

    lo = _fmt(ldf)
    print("\n=== Per-line breakout — every priced column graded on its own "
          "===\n")
    print(lo.to_string(index=False))
    print("\n  'Sold: binary column' = the workbook sells that number from "
          "the binary row (graded there); the x-column's")
    print("  quote for it is shown for completeness. Trust varies by line "
          "— tails are usually less calibrated than mid lines.")

    save_excel(df, ldf, Path(args.out))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
