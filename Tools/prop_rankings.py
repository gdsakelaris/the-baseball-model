"""Internal-quality rankings for every market the model predicts.

Produces the "what should I trust" table: every prediction column ranked by
the quality of its own served numbers on the two held-out test years (2025
selection test + 2026 holdout) — how trustworthy the probabilities are
(calibration), how much real skill sits behind them (edge/AUC/lift), and
whether both years agree. NO scraped odds, market prices, or de-vigged
references enter anywhere; the true market test lives in Section 9 of
evaluate_deep.py and stays there.

The score is Skill x Trust with a cross-year stability haircut, then a
day-block-bootstrap lower bound, tiered within class on frozen anchors:

  Trust  — can the stated percentage be believed at face value?
           Calibration error normalized to the market's base-rate scale
           (an ECE of .006 on an 11% event is far worse than on a 60%
           event), calibration slope (over/under-confidence ECE can miss),
           and for O/U families bias + tails-vs-priced dispersion.
  Skill  — is there anything behind the number? EDGE-LED: the relative
           log-loss edge over the base rate is a proper score (it already
           rewards ranking AND calibration), so it carries 80%; AUC rides at
           20% as a ranking check. Top-10 lift is NO longer scored (too
           noisy — a winner top-1/day lift is nearly a coin flip); it is a
           reported diagnostic and sizes only blue-mark depth. All computed
           through the actual serving prices (per-line calibrators via
           predict.count_over, negative binomial for starter K / totals).
  Stability — Score = mean(S25, S26) - 0.25*|S25 - S26|: a market that
           performs in only one year is down-ranked for exactly that.
  Uncertainty — a weighted day-block bootstrap (the same philosophy as the
           model's paired accept bar) gives each Score a lower bound
           (Score_lo); a thin market earns a wide CI and a demoted tier, so
           depth is never bought with a lucky point estimate.
  Tiers  — cut on Score_lo on FROZEN semantic anchors (never re-fit to the
           run), and WITHIN class: probability markets and expected-count
           means rank on separate ladders and are never compared.

Inputs (all written by the standard loop — no extra steps):
  - eval_paired_select_2025.joblib / eval_paired_2026.joblib   per-row
    (Date, p|mu, y) snapshots from `evaluate_deep.py [--confirm]
    --set-baseline`
  - models_bt.joblib / models.joblib   count heads (line calibrators,
    dispersions) matching each year's suite

Usage:
    python Tools/prop_rankings.py            # print + write the workbook
    python Tools/prop_rankings.py --out FILE
"""
import argparse
import sys
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
# Skill is EDGE-LED. The relative log-loss beat is a PROPER score — it
# already rewards both ranking and calibration — so it carries the Skill
# term; AUC rides at a small weight purely as a calibration-robust ranking
# check (it can catch skill that a miscalibrated log-loss understates).
# Top-10 lift is NO LONGER in the Score: it is a thresholded, high-variance
# function of the same ranking (the winner top-1/day lift is nearly a coin
# flip), so it is reported as a diagnostic and used only to size blue-mark
# depth, where hard caps bound its noise.
SKILL_EDGE_W, SKILL_AUC_W = 0.80, 0.20

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

# Tier cuts are FROZEN semantic anchors on the 0-100 Skill x Trust scale —
# NOT re-fit to each run's distribution. That matters: because the Score is
# built only from a market's OWN edge / calibration / CI, a frozen cut makes
# a market's tier depend on ITSELF alone, never on how other markets moved
# in a retrain. Applied to the LCB. Probability markets (binaries, O/U
# lines and families, winner) and MEAN markets (expected-count columns) sit
# on DIFFERENT scales, so each class has its own ladder and its own block in
# the workbook — a mean Score and a binary Score are never compared.
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
PROB_TIER_CUTS = ((45.0, "1 ELITE"), (33.0, "2 STRONG"), (22.0, "3 SOLID"),
                  (17.0, "4 DECENT"), (12.0, "5 LOW CEILING"),
                  (float("-inf"), "6 AVOID"))
MEAN_TIER_CUTS = ((60.0, "1 ELITE"), (42.0, "2 STRONG"), (26.0, "3 SOLID"),
                  (20.0, "4 DECENT"), (14.0, "5 LOW CEILING"),
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


def cal_slope(p, y):
    """Calibration slope: logistic refit of the outcome on the served
    logit. 1.0 = the probability moves exactly as much as it should;
    < 1 = overconfident (extremes too extreme), > 1 = underconfident."""
    z = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan


def top10_lift(df, p_col="p", y_col="y"):
    """Daily top-10 hit rate over the base rate (how much the best picks
    beat blind betting)."""
    day = df.assign(d=pd.to_datetime(df["Date"]).dt.date)
    top = day.sort_values(p_col, ascending=False).groupby("d").head(10)
    base = df[y_col].mean()
    return float(top[y_col].mean() / base) if base > 0 else np.nan


def binary_year(snap):
    """Per binary prop, one year: edge, AUC, ECE (raw + base-rate scaled),
    calibration slope, top-10 lift."""
    out = {}
    for name, blob in snap["binary"].items():
        df = blob["df"]
        y, p = df["y"].to_numpy(), np.clip(df["p"].to_numpy(), 1e-4, 1 - 1e-4)
        base = y.mean()
        base_ll = log_loss(y, np.full_like(p, base))
        e = ece(p, y)
        out[name] = {
            "rel": 100 * (base_ll - log_loss(y, p)) / base_ll,
            "auc": roc_auc_score(y, p),
            "ece": e,
            "ece_rel": e / (base * (1 - base)),
            "slope": cal_slope(p, y),
            "lift": top10_lift(df),
            "base": float(base),
        }
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
        "rho": float(pd.Series(mu).corr(pd.Series(y), method="spearman")),
        "bias": float(mu.mean() - y.mean()),
        "disp": float(np.mean((y - mu) ** 2) / mu.mean()),
        "mean_y": float(y.mean()),
    }


def mean_score(m):
    """Skill x Trust for a mean column, EDGE-LED like the probability rows:
    MAE beat over the no-skill constant (the proper-score analog) 70% + rank
    corr 30%, trust from bias vs the count's own scale. Its own class — mean
    Scores are compared only to other means (MEAN_TIER_CUTS)."""
    bias_q = _clip(1 - abs(m["bias"]) / (0.05 * m["mean_y"]))
    trust = 0.35 + 0.65 * bias_q
    skill = (0.70 * _clip(m["rel"] / 10.0)
             + 0.30 * _clip(m["rho"] / 0.45))
    return 100 * skill * trust


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
    base_ll = log_loss(y, np.full_like(p, base))
    e = ece(p, y)
    day = df.assign(d=pd.to_datetime(df["Date"]).dt.date,
                    conf=np.maximum(p, 1 - p),
                    hit=((p >= 0.5) == (y == 1)).astype(float))
    top1 = day.sort_values("conf", ascending=False).groupby("d").head(1)
    return {
        "rel": 100 * (base_ll - log_loss(y, p)) / base_ll,
        "auc": roc_auc_score(y, p),
        "ece": e,
        "ece_rel": e / (base * (1 - base)),
        "slope": cal_slope(p, y),
        "lift": float(top1["hit"].mean() / base) if base > 0 else np.nan,
    }


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
        rels, aucs, eces, ece_rels, slopes = [], [], [], [], []
        lift = np.nan
        per_line = {}
        for ln in lines:
            yy = (y > ln).astype(int)
            base = yy.mean()
            if base in (0.0, 1.0):
                continue
            pp = np.clip(pricer(mu, ln), 1e-4, 1 - 1e-4)
            base_ll = log_loss(yy, np.full_like(pp, base))
            rel = 100 * (base_ll - log_loss(yy, pp)) / base_ll
            rels.append(rel)
            e = ece(pp, yy)
            per_line[ln] = {
                "rel": rel, "auc": roc_auc_score(yy, pp), "ece": e,
                "ece_rel": e / (base * (1 - base)),
                "slope": cal_slope(pp, yy), "base": float(base),
                "lift": top10_lift(
                    pd.DataFrame({"Date": df["Date"], "p": pp, "y": yy})),
                "owned": ln in owned_ln,
            }
            if ln not in owned_ln:
                aucs.append(per_line[ln]["auc"])
                eces.append(e)
                ece_rels.append(per_line[ln]["ece_rel"])
                slopes.append(per_line[ln]["slope"])
                if np.isnan(lift):     # flagship = first uniquely-priced line
                    lift = per_line[ln]["lift"]
        out[name] = {
            "lines": per_line,
            "auc": float(np.mean(aucs)) if aucs else np.nan,
            "ece": float(np.mean(eces)) if eces else np.nan,
            "ece_rel": float(np.mean(ece_rels)) if ece_rels else np.nan,
            "slope": float(np.nanmean(slopes)) if slopes else np.nan,
            "rel": float(np.mean(rels)) if rels else np.nan,
            "lift": float(lift),
            "bias": float(mu.mean() - y.mean()),
            "disp": float(np.mean((y - mu) ** 2) / mu.mean()),
            "priced_disp": float(priced_disp),
            "mean_y": float(y.mean()),
            "n_lines": len(rels),
        }
    return out


# ------------------------------------------------------- composite score
# Score(year) = 100 * Skill * Trust; Score = mean(S25, S26)
#               - 0.25 * |S25 - S26|  (stability haircut).
#
# Trust — can the percentage be believed? — multiplies Skill because an
# uncalibrated probability cannot be sized no matter how well it ranks:
#   cal_q   scaled ECE: ECE / (base*(1-base)), anchored so 10% of the
#           market's Bernoulli scale = zero credit
#   slope_q calibration slope: full credit at 1.0, zero at +/-0.5 away
#   (O/U)   bias_q * disp_q — mean bias vs the count's own scale and
#           observed tails vs what the P(over) pricing assumes
# Trust spans [0.35, 1.0]: even a badly calibrated market keeps a third
# of its skill (the ranking may still pick), never all of it.
#
# Skill — is there anything behind the number? EDGE-LED (see SKILL_*_W):
#   edge_q  relative log-loss beat over base rate (8% binary / 10% O/U
#           = full marks) — the proper-score backbone, 80% of Skill
#   auc_q   (AUC - .5) / .20  (.70 = full marks) — 20%, a ranking check
#   (top-10 lift is NO LONGER scored — reported only, and used to size blue
#    depth; see SKILL_*_W for why)
def _clip(x):
    return float(max(0.0, min(1.0, x)))


def _skill(edge_q, auc_q):
    """Edge-led Skill: the proper-score edge carries it, AUC rides small."""
    return SKILL_EDGE_W * edge_q + SKILL_AUC_W * auc_q


def _trust_parts(m):
    cal_q = _clip(1 - m["ece_rel"] / 0.10) if np.isfinite(m["ece_rel"]) else 0.0
    slope_q = (_clip(1 - abs(m["slope"] - 1.0) / 0.5)
               if np.isfinite(m["slope"]) else 0.0)
    return cal_q, slope_q


def binary_score(m):
    cal_q, slope_q = _trust_parts(m)
    trust = 0.35 + 0.65 * (0.7 * cal_q + 0.3 * slope_q)
    skill = _skill(_clip(m["rel"] / 8.0), _clip((m["auc"] - 0.5) / 0.20))
    return 100 * skill * trust


def count_score(m):
    cal_q, slope_q = _trust_parts(m)
    bias_q = _clip(1 - abs(m["bias"]) / (0.05 * m["mean_y"]))
    disp_q = _clip(1 - max(0.0, m["disp"] / max(m["priced_disp"], 1.0) - 1.0)
                   / 0.5)
    trust = 0.35 + 0.65 * (0.5 * cal_q + 0.2 * slope_q + 0.3 * bias_q * disp_q)
    skill = _skill(_clip(m["rel"] / 10.0), _clip((m["auc"] - 0.5) / 0.20))
    return 100 * skill * trust


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
        if ec <= 0.007:
            notes.append("calibrated -> price bets directly")
        elif ec >= 0.011:
            notes.append(f"probability level drifts (ECE {ec:.3f}) -> "
                         "trust picks more than prices")
        if lift >= 2.0:
            notes.append(f"top picks {lift:.1f}x base -> follow the list deep")
        elif lift >= 1.4:
            notes.append(f"top picks {lift:.1f}x -> top 3-10 only")
        else:
            notes.append(f"picks only {lift:.1f}x base -> no selection power")
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
    """1-D weighted logistic slope by Newton/IRLS (matches cal_slope's
    near-unregularized fit; warm-started in the bootstrap for ~6-iter cost)."""
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
    return float(b)


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
            "picks_sumy": picks_sumy, "picks_k": picks_k,
            "ys_d": y[od], "did_d": day_id[od], "starts": starts,
            "pa": p[oa], "ya": y[oa], "did_a": day_id[oa],
            "b0": _slope_irls(z, y, np.ones_like(y))}


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
    return _slope_irls(pp["z"], pp["y"], w[pp["day_id"]], b=pp["b0"], iters=6)


def _binlike_boot(pp, Wm):
    """B composite-input metric dicts -> arrays (score, lift, base). The
    per-day quantities vectorize across all B draws; only AUC/ECE/slope loop."""
    Wtot = Wm @ pp["cnt"]
    base = (Wm @ pp["sumy"]) / Wtot
    base_ll = _binent(base)
    rel = np.where(base_ll > 0, 100 * (base_ll - (Wm @ pp["nllsum"]) / Wtot)
                   / base_ll, np.nan)
    pkw = Wm @ pp["picks_k"]
    lift = np.where((pkw > 0) & (base > 0),
                    (Wm @ pp["picks_sumy"]) / pkw / base, np.nan)
    sc = np.empty(len(Wm))
    for b, w in enumerate(Wm):
        bb = base[b]
        er = _wece(pp, w, bb) / (bb * (1 - bb)) if 0 < bb < 1 else np.nan
        sc[b] = binary_score({"rel": rel[b], "auc": _wauc(pp, w),
                              "ece_rel": er, "slope": _wslope(pp, w)})
    return sc, lift, base


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
            "d_cnt": np.bincount(day_id, minlength=D).astype(float)}


def _countfam_boot(prep, Wm):
    """Family composite Score per resample: non-owned lines' metrics averaged
    (as count_year does), plus bias/dispersion re-derived under the weights."""
    W = Wm @ prep["d_cnt"]
    mu_m, y_m = (Wm @ prep["d_mu"]) / W, (Wm @ prep["d_y"]) / W
    bias = mu_m - y_m
    disp = (Wm @ prep["d_sq"]) / W / mu_m
    sc = np.empty(len(Wm))
    non = [lp for lp in prep["lines"].values() if not lp["owned"]]
    for b, w in enumerate(Wm):
        rels, aucs, ers, sls = [], [], [], []
        for lp in non:
            pp = lp["prep"]
            bs = (w @ pp["sumy"]) / (w @ pp["cnt"])
            if not 0 < bs < 1:
                continue
            bl = _binent(bs)
            rels.append(100 * (bl - (w @ pp["nllsum"]) / (w @ pp["cnt"])) / bl
                        if bl > 0 else np.nan)
            aucs.append(_wauc(pp, w))
            ers.append(_wece(pp, w, bs) / (bs * (1 - bs)))
            sls.append(_wslope(pp, w))
        sc[b] = count_score({
            "rel": np.nanmean(rels) if rels else np.nan,
            "auc": np.nanmean(aucs) if aucs else np.nan,
            "ece_rel": np.nanmean(ers) if ers else np.nan,
            "slope": np.nanmean(sls) if sls else np.nan,
            "bias": bias[b], "disp": disp[b],
            "priced_disp": prep["priced_disp"], "mean_y": y_m[b]})
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
_BOOT_VERSION = 2
_BOOT_CACHE = ART / "quality_boot.joblib"


def _boot_key(snap25, snap26):
    return (str(snap25.get("data_fp")), str(snap25.get("frames_fp")),
            str(snap26.get("data_fp")), str(snap26.get("frames_fp")),
            BOOT_B, SCORE_LCB_Q, BOOT_SEED, SKILL_EDGE_W, SKILL_AUC_W,
            _BOOT_VERSION)


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
BLUE_OR_TOP, BLUE_OR_DEEP = 1.55, 2.0     # odds-ratio lift gates (on the LB)
BLUE_N_TOP, BLUE_N_DEEP = 5, 10
BLUE_SLOPE = (0.80, 1.20)                 # calibration-slope sanity gate
# depth cap keyed to the frozen PROB tiers on the LCB: STRONG+ -> 10, SOLID
# -> 7, DECENT -> 5, LOW CEILING -> 4, AVOID -> 2. Nothing is ever zeroed by
# tier alone — a low-ceiling market still shows its best picks, just fewer.
BLUE_DEPTH_CAPS = ((33.0, 10), (22.0, 7), (17.0, 5), (12.0, 4),
                   (float("-inf"), 2))
# pitcher serving-column prefix -> count-head key (predict maps its columns
# through the same table)
QUAL_STARTER_KEY = {"pk": "k", "pouts": "outs", "phits": "pha",
                    "pbb": "pbb", "per": "per"}


def _blue_depth(or_lift_lo, slope_ok, score_lo):
    """Blue depth from CI-aware inputs (0 = ineligible): the odds-ratio lift
    LOWER bound must clear BLUE_OR_TOP and the slope gate must pass; the base
    depth (deep if the lift LB is strong) is then capped by the LCB tier."""
    if not slope_ok or not np.isfinite(or_lift_lo) or or_lift_lo < BLUE_OR_TOP:
        return 0
    base = BLUE_N_DEEP if or_lift_lo >= BLUE_OR_DEEP else BLUE_N_TOP
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


def _rank_and_tier(df):
    """Order rows into class blocks (probability markets, then means, then
    informational), sort each block by the LCB, and tier WITHIN class on the
    frozen ladders — so a mean Score and a binary Score are never compared."""
    crank = {"Prob": 0, "Mean": 1}
    df = df.copy()
    df["_c"] = df["Class"].map(lambda c: crank.get(c, 2))
    df["_s"] = pd.to_numeric(df["Score_lo"], errors="coerce").fillna(-np.inf)
    df = (df.sort_values(["_c", "_s"], ascending=[True, False])
            .drop(columns=["_c", "_s"]).reset_index(drop=True))
    df["Tier"] = [tier_of(s, MEAN_TIER_CUTS) if c == "Mean"
                  else tier_of(s, PROB_TIER_CUTS) if c == "Prob" else "-"
                  for c, s in zip(df["Class"], df["Score_lo"])]
    df.insert(0, "#", range(1, len(df) + 1))
    return df


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

    def bin_lo(key):
        return lcb.get("binary", {}).get(key, {}).get("score_lo", np.nan)

    rows = []
    for key, name in BIN_NAMES.items():
        m25, m26 = b25[key], b26[key]
        rows.append({
            "Market": name, "Key": key, "Class": "Prob",
            "Score": final_score(binary_score(m25), binary_score(m26)),
            "Score_lo": bin_lo(key),
            "S25": binary_score(m25), "S26": binary_score(m26),
            "AUC": (m25["auc"] + m26["auc"]) / 2,
            "ECE": (m25["ece"] + m26["ece"]) / 2,
            "Slope": np.nanmean([m25["slope"], m26["slope"]]),
            "Lift": (m25["lift"] + m26["lift"]) / 2,
            "Bias": np.nan, "Disp": np.nan,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
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
            "S25": count_score(m25), "S26": count_score(m26),
            "AUC": (m25["auc"] + m26["auc"]) / 2,
            "ECE": (m25["ece"] + m26["ece"]) / 2,
            "Slope": np.nanmean([m25["slope"], m26["slope"]]),
            "Lift": np.nanmean([m25["lift"], m26["lift"]]),
            "Bias": (m25["bias"] + m26["bias"]) / 2,
            "Disp": (m25["disp"] + m26["disp"]) / 2,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
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
                    "S25": np.nan, "S26": np.nan, "AUC": np.nan,
                    "ECE": np.nan, "Slope": np.nan, "Lift": np.nan,
                    "Bias": np.nan, "Disp": np.nan, "Edge%": np.nan,
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
            "AUC": np.nan, "ECE": np.nan, "Slope": np.nan, "Lift": np.nan,
            "Bias": bias, "Disp": (m25["disp"] + m26["disp"]) / 2,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "Notes": note,
        })
    # Lineup HRs is arithmetic on the Batter HR probabilities (no model or
    # snapshot of its own) — shown so nothing displayed is missing
    rows.append({
        "Market": "Lineup HRs", "Key": "lineup_hr", "Class": "-",
        "Score": np.nan, "Score_lo": np.nan,
        "S25": np.nan, "S26": np.nan, "AUC": np.nan, "ECE": np.nan,
        "Slope": np.nan, "Lift": np.nan, "Bias": np.nan, "Disp": np.nan,
        "Edge%": np.nan,
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
            "AUC": (w25["auc"] + w26["auc"]) / 2,
            "ECE": (w25["ece"] + w26["ece"]) / 2,
            "Slope": np.nanmean([w25["slope"], w26["slope"]]),
            "Lift": (w25["lift"] + w26["lift"]) / 2,
            "Bias": np.nan, "Disp": np.nan,
            "Edge%": (w25["rel"] + w26["rel"]) / 2,
            "Notes": play_note("bin", w25, w26)
                     + "; no proven side edge vs always-home (McNemar "
                       "n.s.) -> win% quality, not a moneyline endorsement",
        })
    else:
        rows.append({"Market": "Game Winner (Win Prob)",
                     "Key": "winner", "Class": "Prob",
                     "Score": np.nan, "Score_lo": np.nan,
                     "S25": np.nan, "S26": np.nan,
                     "AUC": np.nan, "ECE": np.nan, "Slope": np.nan,
                     "Lift": np.nan, "Bias": np.nan,
                     "Disp": np.nan, "Edge%": np.nan,
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
            "S25": binary_score(m25), "S26": binary_score(m26),
            "Base%": np.nan,
            "AUC": (m25["auc"] + m26["auc"]) / 2,
            "ECE": (m25["ece"] + m26["ece"]) / 2,
            "Slope": np.nanmean([m25["slope"], m26["slope"]]),
            "Lift": (m25["lift"] + m26["lift"]) / 2,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
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
                "S25": binary_score(m25), "S26": binary_score(m26),
                "Base%": 100 * (m25["base"] + m26["base"]) / 2,
                "AUC": (m25["auc"] + m26["auc"]) / 2,
                "ECE": (m25["ece"] + m26["ece"]) / 2,
                "Slope": np.nanmean([m25["slope"], m26["slope"]]),
                "Lift": np.nanmean([m25["lift"], m26["lift"]]),
                "Edge%": (m25["rel"] + m26["rel"]) / 2,
                "Sold": ("binary column" if m25["owned"] else "this column"),
            })
    if w25 is not None and w26 is not None:
        line_rows.append({
            "Market": "Game Winner (Win Prob)", "Class": "Prob",
            "Score": final_score(binary_score(w25), binary_score(w26)),
            "Score_lo": lcb.get("winner", {}).get("score_lo", np.nan),
            "S25": binary_score(w25), "S26": binary_score(w26),
            "Base%": np.nan,
            "AUC": (w25["auc"] + w26["auc"]) / 2,
            "ECE": (w25["ece"] + w26["ece"]) / 2,
            "Slope": np.nanmean([w25["slope"], w26["slope"]]),
            "Lift": (w25["lift"] + w26["lift"]) / 2,
            "Edge%": (w25["rel"] + w26["rel"]) / 2,
            "Sold": "this column",
        })
    # the displayed mean columns, on the per-column sheet too (AUC/ECE/
    # Slope/Lift are probability diagnostics — blank for means; Edge% is
    # the MAE beat over the no-skill constant)
    for key, name in mean_names.items():
        if key not in mm25 or key not in mm26:
            continue
        m25, m26 = mm25[key], mm26[key]
        line_rows.append({
            "Market": name, "Class": "Mean",
            "Score": final_score(mean_score(m25), mean_score(m26)),
            "Score_lo": lcb.get("mean", {}).get(key, {}).get("score_lo", np.nan),
            "S25": mean_score(m25), "S26": mean_score(m26),
            "Base%": np.nan, "AUC": np.nan, "ECE": np.nan,
            "Slope": np.nan, "Lift": np.nan,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "Sold": "mean (context)",
        })
    ldf = _rank_and_tier(pd.DataFrame(line_rows))
    ldf = ldf[["#", "Market", "Class", "Tier", "Score", "Score_lo", "S25",
               "S26", "Base%", "AUC", "ECE", "Slope", "Lift", "Edge%", "Sold"]]

    return df[["#", "Market", "Key", "Class", "Tier", "Score", "Score_lo",
               "S25", "S26", "AUC", "ECE", "Slope", "Lift", "Bias", "Disp",
               "Edge%", "Notes"]], ldf


LEGEND = [
    ("Score", "Skill x Trust on a 0-100 scale, per held-out test year, then "
     "mean(S25, S26) minus 0.25x their gap (a market that performs in only "
     "one year is down-ranked for exactly that). Skill = edge 80% + AUC 20% "
     "(edge, the relative log-loss beat, is a proper score that already "
     "rewards ranking AND calibration, so it leads; top-10 lift is NO longer "
     "scored - too noisy - and is shown only as a diagnostic). Trust = "
     "calibration (base-rate-scaled ECE + slope; O/U also bias and "
     "tails-vs-priced dispersion), spanning 0.35-1.0. No scraped odds "
     "anywhere. This is the point estimate; Tier uses Score_lo."),
    ("Score_lo", "The day-block-bootstrap LOWER bound of Score (10th "
     "percentile over resampled days) - the SAME bootstrap philosophy as the "
     "model's paired accept bar. A thin market (SB, a deep line) earns a wide "
     "CI and a lower Score_lo, so it can't buy a high tier on a lucky point "
     "estimate. TIERS AND BLUE-MARK DEPTH READ THIS COLUMN, not Score."),
    ("Class / block", "Prob = probability markets (binaries, O/U lines and "
     "families, the winner); Mean = the displayed expected-count columns. "
     "They sit on DIFFERENT scales, so each is its own block with its OWN "
     "tier ladder - a mean Score and a binary Score are never compared or "
     "interleaved."),
    ("S25 / S26", "The same score computed on each test year alone - "
     "agreement between them means the ranking is stable, not one-year "
     "noise. The gap directly reduces the final Score."),
    ("AUC", "Ranking skill (0.5 = coin flip). Can it put the players who "
     "DID do it above the ones who didn't? O/U columns: computed per line "
     "through the actual quoted P(over), averaged across the column's line "
     "family - comparable to the binary rows. Batter x-columns cover ONLY "
     "the deep lines they uniquely price (their half-point lines belong to "
     "the binary rows)."),
    ("ECE", "Calibration: average gap between stated probability and "
     "reality. 0 = perfect. Inside Score it is scaled by the market's "
     "base-rate variance (an ECE of .006 is far worse on an 11% event "
     "than on a 60% one); the column shows the raw value."),
    ("Slope", "Calibration slope: refit of reality on the served logit. "
     "1.0 = probabilities move exactly as much as they should; below 1 = "
     "overconfident (shade extremes toward the middle), above 1 = "
     "underconfident (extremes better than stated)."),
    ("Lift", "Daily top-10 hit rate over the base rate - the selection "
     "power behind 'follow the list'. O/U columns: measured on the first "
     "line the column uniquely prices. DIAGNOSTIC ONLY now (removed from "
     "Score for noise); its base-rate-fair odds-ratio form, with a bootstrap "
     "lower bound, is still what sizes blue-mark depth on the prediction "
     "sheets."),
    ("Bias", "O/U columns: predicted count minus actual, on average. "
     "Positive = model over-predicts -> lean unders."),
    ("Disp", "O/U columns: error variance / mean. Compared inside Score "
     "to what the P(over) pricing ASSUMES (NB heads price extra variance "
     "already); higher observed = real tails wilder than priced -> don't "
     "trust extreme-line probabilities."),
    ("Edge%", "How much better the column prices the event than the "
     "base-rate guess (relative log-loss beat; O/U columns averaged "
     "across every line priced). Held-out years only."),
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
     "Score_lo (the CI lower bound) on FROZEN semantic anchors WITHIN each "
     "class - a market's tier depends only on its own edge/calibration/CI, "
     "never on how other markets moved in a retrain, and probability and mean "
     "rows use separate ladders."),
    ("Lines sheet", "Every individual priced column graded on its own — "
     "binary props are single lines (repeated from Rankings for a complete "
     "view); count families split into their lines, each with its own "
     "Score/Tier. 'Sold: binary column' = the workbook sells that number "
     "from the binary row; the x-column's quote is shown for completeness. "
     "Base% = how often the over actually hits."),
    ("Mean columns", "The displayed expected-count columns (xK, xOuts, "
     "xHits, xBB, xER, xSO, xTB, xHRR, Total Runs, Away/Home Score) are "
     "separate predictions from the '> x' probabilities, so they get their "
     "own rows: Edge% = MAE beat over a no-skill constant guess, Notes "
     "carry the Spearman rank correlation (the AUC analog for counts) and "
     "bias lean; AUC/ECE/Slope/Lift are probability diagnostics and stay "
     "blank. They are the Mean CLASS: their own block, their own tier "
     "ladder, never scored against a probability row. Lineup HRs is "
     "arithmetic on Batter HR - graded there."),
]


def save_excel(df, ldf, path):
    """Write the rankings workbook (Rankings + Lines + Legend), styled like
    the prediction workbooks via predict._polish."""
    xl = df.copy()
    for c, nd in [("Score", 0), ("Score_lo", 0), ("S25", 0), ("S26", 0),
                  ("AUC", 3), ("ECE", 4), ("Slope", 2), ("Lift", 2),
                  ("Bias", 2), ("Disp", 2), ("Edge%", 2)]:
        xl[c] = xl[c].round(nd)
    xll = ldf.copy()
    for c, nd in [("Score", 0), ("Score_lo", 0), ("S25", 0), ("S26", 0),
                  ("Base%", 1), ("AUC", 3), ("ECE", 4), ("Slope", 2),
                  ("Lift", 2), ("Edge%", 2)]:
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
    out = df.copy()
    for c in ("Score", "Score_lo", "S25", "S26"):
        out[c] = out[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    out["Edge%"] = out["Edge%"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    out["AUC"] = out["AUC"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    out["ECE"] = out["ECE"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    out["Slope"] = out["Slope"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    out["Lift"] = out["Lift"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    out["Bias"] = out["Bias"].map(
        lambda v: f"{v:+.2f}" if pd.notna(v) else "-")
    out["Disp"] = out["Disp"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")

    print("\n=== Prediction-column quality rankings — held-out test years "
          "2025 + 2026, internal measurements only ===\n")
    print(out.to_string(index=False))
    print("\n  Score = Skill x Trust per year, then mean(S25, S26) - "
          "0.25x|S25 - S26| (two-year stability haircut). Score_lo is its "
          "day-block-bootstrap lower bound;")
    print("  Tier cuts on Score_lo (thin markets carry a wider CI and a "
          "demoted tier), on FROZEN semantic anchors, WITHIN class —")
    print("  probability rows (Prob) and expected-count means (Mean) rank on "
          "separate ladders and never share a cut.")
    print("  Skill: Edge% (log-loss beat over base rate) 80% + AUC 20%; "
          "top-10 Lift is a diagnostic only (too noisy to score).")
    print("  Trust: base-rate-scaled ECE + calibration Slope (O/U also "
          "Bias + Disp vs what the pricing assumes), spanning 0.35-1.0 —")
    print("         an uncalibrated probability can't be sized no matter "
          "how well it ranks.")
    print("  No scraped odds or market prices enter this table; the "
          "market test is Section 9 of evaluate_deep.py.")

    lo = ldf.copy()
    for c in ("Score", "Score_lo", "S25", "S26"):
        lo[c] = lo[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    lo["Base%"] = lo["Base%"].map(
        lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    lo["AUC"] = lo["AUC"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    lo["ECE"] = lo["ECE"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    lo["Slope"] = lo["Slope"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    lo["Lift"] = lo["Lift"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    lo["Edge%"] = lo["Edge%"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
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
