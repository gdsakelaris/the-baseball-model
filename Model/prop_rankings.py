"""Internal-quality rankings for every market the model predicts.

Produces the "what should I trust" table: every prediction column ranked by
the quality of its own served numbers on the two held-out test years (2025
selection test + 2026 holdout) — how trustworthy the probabilities are
(calibration), how much real skill sits behind them (edge/AUC/lift), and
whether both years agree. NO scraped odds, market prices, or de-vigged
references enter anywhere; the true market test lives in Section 9 of
evaluate_deep.py and stays there.

The score is Skill x Trust with a cross-year stability haircut:

  Trust  — can the stated percentage be believed at face value?
           Calibration error normalized to the market's base-rate scale
           (an ECE of .006 on an 11% event is far worse than on a 60%
           event), calibration slope (over/under-confidence ECE can miss),
           and for O/U families bias + tails-vs-priced dispersion.
  Skill  — is there anything behind the number? Relative log-loss edge
           over the base rate, AUC, and daily top-10 lift, each computed
           through the actual serving prices (per-line calibrators via
           predict.count_over, negative binomial for starter K / totals).
  Stability — Score = mean(S25, S26) - 0.25*|S25 - S26|: a market that
           performs in only one year is down-ranked for exactly that.

Inputs (all written by the standard loop — no extra steps):
  - eval_paired_select_2025.joblib / eval_paired_2026.joblib   per-row
    (Date, p|mu, y) snapshots from `evaluate_deep.py [--confirm]
    --set-baseline`
  - models_bt.joblib / models.joblib   count heads (line calibrators,
    dispersions) matching each year's suite

Usage:
    python Model/prop_rankings.py            # print + write the workbook
    python Model/prop_rankings.py --out FILE
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict as P  # noqa: E402  (count_over / nb_over / K_LINES)

ART = Path(__file__).resolve().parent / "artifacts"

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
#   BATTER_X — expected-count context columns; their half-point lines are
#     priced (better) by the binary rows -> AUC/ECE grade ONLY the deep
#     lines they uniquely price (3+ K, 3/4+ TB, 4+ HRR); Edge% still
#     averages every line they quote.
#   GAME_CNT — game totals (NB-priced, no binary overlap; full family).
PITCHER_CNT = {
    "k":     "Pitcher K > x",
    "outs":  "Pitcher Outs > x",
    "pha":   "Pitcher Hits > x",
    "pbb":   "Pitcher BB > x",
    "per":   "Pitcher ER > x",
}
BATTER_X = {
    "xbk":   "Batter xSO",
    "xtb":   "Batter xTB",
    "xhrr":  "Batter xHRR",
}
GAME_CNT = {"total": "Game Runs > x"}
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
    "k":     "Pitcher xK",
    "outs":  "Pitcher xOuts",
    "pha":   "Pitcher xHits",
    "pbb":   "Pitcher xBB",
    "per":   "Pitcher xER",
    "total": "Game Total Runs",
}

# Which of an x-column's lines the BINARY columns already price (bk = 1+ K
# -> 0.5, bk2 -> 1.5; tb2 -> 1.5; hrr2/hrr3 -> 1.5/2.5). Pure workbook
# structure — which column sells which number — NOT market data: the
# x-row's AUC/ECE describe only the deep lines it uniquely prices.
BINARY_OWNED_LINES = {"xbk": {0.5, 1.5}, "xtb": {1.5}, "xhrr": {1.5, 2.5}}


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
    """Skill x Trust for a mean column: MAE beat 55% + rank corr 45%,
    trust from bias vs the count's own scale."""
    bias_q = _clip(1 - abs(m["bias"]) / (0.05 * m["mean_y"]))
    trust = 0.35 + 0.65 * bias_q
    skill = (0.55 * _clip(m["rel"] / 10.0)
             + 0.45 * _clip(m["rho"] / 0.45))
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


def count_lines(name, head, k_disp, total_disp):
    """The lines a count column prices and its P(over) pricer, mirroring
    serving: per-line calibrators via count_over (which itself prices the
    NB_PRICED_TARGETS heads, e.g. per, with the negative binomial), NB for
    starter K and game totals."""
    if name == "k":
        return P.K_LINES, lambda mu, ln: np.array(
            [P.nb_over(m, ln, k_disp) for m in mu])
    if name == "total":
        return [6.5, 7.5, 8.5, 9.5, 10.5], lambda mu, ln: np.array(
            [P.nb_over(m, ln, total_disp) for m in mu])
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
        lines, pricer = count_lines(name, head, k_disp, total_disp)
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
# Skill — is there anything behind the number?
#   edge_q  relative log-loss beat over base rate (8% binary / 10% O/U
#           = full marks)
#   auc_q   (AUC - .5) / .20  (.70 = full marks)
#   lift_q  (top-10 lift - 1) / 2  (3x = full marks)
def _clip(x):
    return float(max(0.0, min(1.0, x)))


def _trust_parts(m):
    cal_q = _clip(1 - m["ece_rel"] / 0.10) if np.isfinite(m["ece_rel"]) else 0.0
    slope_q = (_clip(1 - abs(m["slope"] - 1.0) / 0.5)
               if np.isfinite(m["slope"]) else 0.0)
    return cal_q, slope_q


def binary_score(m):
    cal_q, slope_q = _trust_parts(m)
    trust = 0.35 + 0.65 * (0.7 * cal_q + 0.3 * slope_q)
    skill = (0.45 * _clip(m["rel"] / 8.0)
             + 0.35 * _clip((m["auc"] - 0.5) / 0.20)
             + 0.20 * _clip((m["lift"] - 1.0) / 2.0))
    return 100 * skill * trust


def count_score(m):
    cal_q, slope_q = _trust_parts(m)
    bias_q = _clip(1 - abs(m["bias"]) / (0.05 * m["mean_y"]))
    disp_q = _clip(1 - max(0.0, m["disp"] / max(m["priced_disp"], 1.0) - 1.0)
                   / 0.5)
    trust = 0.35 + 0.65 * (0.5 * cal_q + 0.2 * slope_q + 0.3 * bias_q * disp_q)
    skill = (0.45 * _clip(m["rel"] / 10.0)
             + 0.35 * _clip((m["auc"] - 0.5) / 0.20)
             + 0.20 * _clip((m["lift"] - 1.0) / 2.0))
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


def build_table():
    snap25 = joblib.load(ART / "eval_paired_select_2025.joblib")
    snap26 = joblib.load(ART / "eval_paired_2026.joblib")
    art25 = joblib.load(ART / "models_bt.joblib")
    art26 = joblib.load(ART / "models.joblib")

    b25, b26 = binary_year(snap25), binary_year(snap26)
    c25 = count_year(snap25, art25)
    c26 = count_year(snap26, art26)

    rows = []
    for key, name in BIN_NAMES.items():
        m25, m26 = b25[key], b26[key]
        s25, s26 = binary_score(m25), binary_score(m26)
        rows.append({
            "Market": name, "Key": key,
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
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
        s25, s26 = count_score(m25), count_score(m26)
        rows.append({
            "Market": name, "Key": key,
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
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
    # (Edge%), Spearman rank correlation, bias trust
    mm25, mm26 = mean_year(snap25), mean_year(snap26)
    mean_names = dict(MEAN_MARKETS)
    mean_names["score"] = "Away/Home Score"
    for key, name in mean_names.items():
        if key not in mm25 or key not in mm26:
            if key == "score":
                rows.append({
                    "Market": name, "Key": "score", "Score": np.nan,
                    "S25": np.nan, "S26": np.nan, "AUC": np.nan,
                    "ECE": np.nan, "Slope": np.nan, "Lift": np.nan,
                    "Bias": np.nan, "Disp": np.nan, "Edge%": np.nan,
                    "Notes": "snapshot predates per-team score rows - "
                             "re-run --set-baseline (both suites) to grade"})
            continue
        m25, m26 = mm25[key], mm26[key]
        s25, s26 = mean_score(m25), mean_score(m26)
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
            "Market": name, "Key": f"x:{key}",
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
            "AUC": np.nan, "ECE": np.nan, "Slope": np.nan, "Lift": np.nan,
            "Bias": bias, "Disp": (m25["disp"] + m26["disp"]) / 2,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "Notes": note,
        })
    # Lineup HRs is arithmetic on the Batter HR probabilities (no model or
    # snapshot of its own) — shown so nothing displayed is missing
    rows.append({
        "Market": "Lineup HRs", "Key": "lineup_hr", "Score": np.nan,
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
        s25, s26 = binary_score(w25), binary_score(w26)
        rows.append({
            "Market": "Game Winner (Win Prob)", "Key": "winner",
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
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
                     "Key": "winner",
                     "Score": 0.0, "S25": np.nan, "S26": np.nan,
                     "AUC": np.nan, "ECE": np.nan, "Slope": np.nan,
                     "Lift": np.nan, "Bias": np.nan,
                     "Disp": np.nan, "Edge%": np.nan,
                     "Notes": "snapshot predates winner rows — re-run "
                              "evaluate_deep --set-baseline (both suites) "
                              "to grade this column"})
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    # tiers cut on the composite (0-100). The Skill x Trust x stability
    # product is a stricter scale than the old additive one — the cuts
    # sit where the observed distribution actually breaks. Ungraded
    # informational rows (NaN score) tier as "—".
    cuts = [(55, "1 ELITE"), (42, "2 STRONG"), (28, "3 SOLID"),
            (15, "4 MARGINAL"), (-1, "5 AVOID")]

    def _tier(s):
        if not np.isfinite(s):
            return "-"
        return next(t for c, t in cuts if s >= c)

    df["Tier"] = df["Score"].map(_tier)
    df.insert(0, "#", range(1, len(df) + 1))

    # ---- per-line breakout: every individual priced column, own grade.
    # Binary props ARE single lines (their family row repeats here for a
    # complete per-column view); count families split into their lines.
    # Trust varies BY LINE (tails run less calibrated than mid lines) and
    # the family average hides exactly that.
    line_rows = []
    for key, name in BIN_NAMES.items():
        m25, m26 = b25[key], b26[key]
        s25, s26 = binary_score(m25), binary_score(m26)
        line_rows.append({
            "Market": name,
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
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
            s25, s26 = binary_score(m25), binary_score(m26)
            line_rows.append({
                "Market": f"{stem} > {ln}",
                "Score": final_score(s25, s26), "S25": s25, "S26": s26,
                "Base%": 100 * (m25["base"] + m26["base"]) / 2,
                "AUC": (m25["auc"] + m26["auc"]) / 2,
                "ECE": (m25["ece"] + m26["ece"]) / 2,
                "Slope": np.nanmean([m25["slope"], m26["slope"]]),
                "Lift": np.nanmean([m25["lift"], m26["lift"]]),
                "Edge%": (m25["rel"] + m26["rel"]) / 2,
                "Sold": ("binary column" if m25["owned"] else "this column"),
            })
    if w25 is not None and w26 is not None:
        s25, s26 = binary_score(w25), binary_score(w26)
        line_rows.append({
            "Market": "Game Winner (Win Prob)",
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
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
        s25, s26 = mean_score(m25), mean_score(m26)
        line_rows.append({
            "Market": name,
            "Score": final_score(s25, s26), "S25": s25, "S26": s26,
            "Base%": np.nan, "AUC": np.nan, "ECE": np.nan,
            "Slope": np.nan, "Lift": np.nan,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "Sold": "mean (context)",
        })
    ldf = pd.DataFrame(line_rows).sort_values("Score", ascending=False)
    ldf["Tier"] = ldf["Score"].map(_tier)
    ldf.insert(0, "#", range(1, len(ldf) + 1))
    ldf = ldf[["#", "Market", "Tier", "Score", "S25", "S26",
               "Base%", "AUC", "ECE", "Slope", "Lift", "Edge%", "Sold"]]

    return df[["#", "Market", "Key", "Tier", "Score", "S25", "S26",
               "AUC", "ECE", "Slope", "Lift", "Bias", "Disp", "Edge%",
               "Notes"]], ldf


LEGEND = [
    ("Score", "Skill x Trust on a 0-100 scale, per held-out test year, "
     "then mean(S25, S26) minus 0.25x their gap (a market that performs "
     "in only one year is down-ranked for exactly that). Skill = edge 45% "
     "+ AUC 35% + top-10 lift 20%. Trust = calibration (base-rate-scaled "
     "ECE + slope; O/U also bias and tails-vs-priced dispersion), spanning "
     "0.35-1.0. Internal measurements only — no scraped odds anywhere."),
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
     "line the column uniquely prices."),
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
     "with the noted caveat. 4 MARGINAL / 5 AVOID: the model cannot "
     "separate players well enough to act on."),
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
     "blank; their Scores use count anchors, so compare mean rows "
     "primarily against other mean rows. Lineup HRs is arithmetic on "
     "Batter HR - graded there."),
]


def save_excel(df, ldf, path):
    """Write the rankings workbook (Rankings + Lines + Legend), styled like
    the prediction workbooks via predict._polish."""
    xl = df.copy()
    for c, nd in [("Score", 0), ("S25", 0), ("S26", 0), ("AUC", 3),
                  ("ECE", 4), ("Slope", 2), ("Lift", 2), ("Bias", 2),
                  ("Disp", 2), ("Edge%", 2)]:
        xl[c] = xl[c].round(nd)
    xll = ldf.copy()
    for c, nd in [("Score", 0), ("S25", 0), ("S26", 0), ("Base%", 1),
                  ("AUC", 3), ("ECE", 4), ("Slope", 2), ("Lift", 2),
                  ("Edge%", 2)]:
        xll[c] = xll[c].round(nd)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        xl.to_excel(xw, sheet_name="Rankings", index=False)
        xll.to_excel(xw, sheet_name="Lines", index=False)
        pd.DataFrame(LEGEND, columns=["Term", "Meaning"]).to_excel(
            xw, sheet_name="Legend", index=False)
    P._polish(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", metavar="FILE",
                    default=str(Path(__file__).resolve().parent
                                / "PROP_RANKINGS.xlsx"),
                    help="output Excel file "
                         "(default: Model/PROP_RANKINGS.xlsx)")
    args = ap.parse_args()

    df, ldf = build_table()
    out = df.copy()
    for c in ("Score", "S25", "S26"):
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
          "0.25x|S25 - S26| (two-year stability haircut).")
    print("  Skill: Edge% (log-loss beat over base rate) 45% + AUC 35% + "
          "top-10 Lift 20%.")
    print("  Trust: base-rate-scaled ECE + calibration Slope (O/U also "
          "Bias + Disp vs what the pricing assumes), spanning 0.35-1.0 —")
    print("         an uncalibrated probability can't be sized no matter "
          "how well it ranks.")
    print("  No scraped odds or market prices enter this table; the "
          "market test is Section 9 of evaluate_deep.py.")

    lo = ldf.copy()
    for c in ("Score", "S25", "S26"):
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
