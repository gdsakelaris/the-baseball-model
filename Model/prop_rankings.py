"""Betting-merit rankings for every market the model predicts.

Produces the "what should I bet on" table: every bettable market ranked by
the model's PRICING EDGE vs a no-skill price, averaged across the two test
years (2025 selection test + 2026 holdout), with calibration/bias annotations
that say how far to trust each probability when sizing.

The universal score: relative log-loss edge — how much better the model
prices the event than the base rate does, as a share of the base-rate log
loss. Binary props score their single event; O/U count markets average the
edge across every line they price (each line's P(over) computed exactly as
serving does: the artifact's per-line calibrator via predict.count_over, or
negative binomial for starter K / game totals). One scale for all markets.

Inputs (all written by the standard loop — no extra steps):
  - eval_paired_select_2025.joblib / eval_paired_2026.joblib   per-row
    (Date, p|mu, y) snapshots from `evaluate_deep.py [--confirm]
    --set-baseline`
  - models_bt.joblib / models.joblib   count heads (line calibrators,
    dispersions) matching each year's suite

Usage:
    python Model/prop_rankings.py            # print the ranked table
    python Model/prop_rankings.py --md FILE  # also write a markdown copy

Note: this measures edge vs a NO-SKILL price, not vs the market. Section 9
of evaluate_deep.py (model vs de-vigged sportsbook lines) is the true
market test once ~15+ days of scraped odds accrue.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import odds as O  # noqa: E402  (load_odds / devig_two_way / PROP_MARKET)
import predict as P  # noqa: E402  (count_over / nb_over / K_LINES)

ART = Path(__file__).resolve().parent / "artifacts"
ODDS_STORE = Path(__file__).resolve().parent.parent / "Data" / "mlb_odds.csv"

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
GAME_CNT = {"total": "Game Total Runs + Runs > x"}
CNT_MARKETS = {**PITCHER_CNT, **BATTER_X, **GAME_CNT}


def ece(p, y, bins=10):
    """Expected calibration error, equal-count bins (evaluate_deep's ece)."""
    q = pd.qcut(p, bins, duplicates="drop")
    df = pd.DataFrame({"p": p, "y": y, "q": q})
    g = df.groupby("q", observed=True)
    return float((g["p"].mean().sub(g["y"].mean()).abs()
                  * g.size().div(len(df))).sum())


def top10_lift(df):
    """Daily top-10 hit rate over the base rate (how much the best picks
    beat blind betting)."""
    day = df.assign(d=pd.to_datetime(df["Date"]).dt.date)
    top = day.sort_values("p", ascending=False).groupby("d").head(10)
    base = df["y"].mean()
    return float(top["y"].mean() / base) if base > 0 else np.nan


def binary_year(snap):
    """Per binary prop: relative edge, AUC, ECE, top-10 lift for one year."""
    out = {}
    for name, blob in snap["binary"].items():
        df = blob["df"]
        y, p = df["y"].to_numpy(), np.clip(df["p"].to_numpy(), 1e-4, 1 - 1e-4)
        base = np.full_like(p, y.mean())
        base_ll = log_loss(y, base)
        out[name] = {
            "rel": 100 * (base_ll - log_loss(y, p)) / base_ll,
            "auc": roc_auc_score(y, p),
            "ece": ece(p, y),
            "lift": top10_lift(df),
        }
    return out


# ------------------------------------------------------- composite score
# No single metric decides the ranking (a lone log-loss edge can flatter a
# high-base-rate prop; AUC alone ignores whether the probability level is
# priceable). Each market blends the INDEPENDENT skill signals available for
# its kind, then a calibration-trust gate scales the result — an
# uncalibrated probability can't be sized no matter how well it ranks.
#   binary:  edge (40%) + discrimination/AUC (35%) + top-pick lift (25%),
#            gated by ECE
#   O/U:     edge (55%) + discrimination/rank-corr (45%),
#            gated by bias + dispersion sanity
# Anchors (the value earning full marks): edge 8%/10%, AUC .70, lift 3x,
# rank-corr .45 — chosen so the best observed market lands near 100.
_clip = lambda x: float(max(0.0, min(1.0, x)))


def binary_score(m):
    edge = _clip(m["rel"] / 8.0)
    disc = _clip((m["auc"] - 0.5) / 0.20)
    lift = _clip((m["lift"] - 1.0) / 2.0)
    trust = 0.6 + 0.4 * _clip(1 - m["ece"] / 0.015)
    return 100 * (0.40 * edge + 0.35 * disc + 0.25 * lift) * trust


def count_score(m):
    edge = _clip(m["rel"] / 10.0)
    disc = _clip(m["rho"] / 0.45)
    bias_q = _clip(1 - abs(m["bias"]) / (0.05 * m["mean_y"]))
    disp_q = _clip(1 - max(0.0, m["disp"] - 1.0) / 1.3)
    trust = 0.6 + 0.4 * bias_q * disp_q
    return 100 * (0.55 * edge + 0.45 * disc) * trust


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
    """Per expected-count column: mean per-line relative edge +
    bias/dispersion for one year."""
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
        # each line is a binary market: AUC/ECE per line through the actual
        # serving prices, averaged over the family -> directly comparable to
        # the binary columns (pricers are monotone in mu, so per-line AUC is
        # pure within-line ranking; pooling lines would inflate it).
        # BATTER_X rows skip the lines their binary siblings already price
        # (xbk 0.5/1.5 = bk/bk2, ...): the workbook sells those from the
        # binaries, so the x-row's AUC/ECE describes only the deep lines it
        # uniquely prices. PITCHER_CNT/GAME_CNT are the sole pricer of
        # every line they quote -> full family.
        owned_ln = ({m["line"] for m in O.PROP_MARKET.values()
                     if m["api"] == _CNT_API.get(name)}
                    if name in BATTER_X else set())
        rels, aucs, eces = [], [], []
        for ln in lines:
            yy = (y > ln).astype(int)
            if yy.mean() in (0.0, 1.0):
                continue
            pp = np.clip(pricer(mu, ln), 1e-4, 1 - 1e-4)
            base_ll = log_loss(yy, np.full_like(pp, yy.mean()))
            rels.append(100 * (base_ll - log_loss(yy, pp)) / base_ll)
            if ln not in owned_ln:
                aucs.append(roc_auc_score(yy, pp))
                eces.append(ece(pp, yy))
        out[name] = {
            "auc": float(np.mean(aucs)) if aucs else np.nan,
            "ece": float(np.mean(eces)) if eces else np.nan,
            "rel": float(np.mean(rels)) if rels else np.nan,
            "bias": float(mu.mean() - y.mean()),
            "disp": float(np.mean((y - mu) ** 2) / mu.mean()),
            "priced_disp": float(priced_disp),
            "rho": float(pd.Series(mu).corr(pd.Series(y),
                                            method="spearman")),
            "mean_y": float(y.mean()),
            "n_lines": len(rels),
        }
    return out


# ------------------------------------------------- market edge (Section 9 lite)
# Model vs the de-vigged sportsbook consensus, graded on whatever days the
# odds store has captured. This is the ONLY column that touches scraped odds;
# it is display-only (never part of Score) and fills in automatically as the
# daily scrape accrues — treat it as anecdote until ~15+ days.

# count-head -> Odds API market (starter heads from odds.STARTER_MARKET,
# batter heads from their binary siblings' apis; total/winner skipped — the
# store carries no GamePk to join game rows on)
_CNT_API = {"k": "pitcher_strikeouts", "outs": "pitcher_outs",
            "pha": "pitcher_hits_allowed", "pbb": "pitcher_walks",
            "per": "pitcher_earned_runs",
            "xbk": "batter_strikeouts", "xtb": "batter_total_bases",
            "xhrr": "batter_hits_runs_rbis"}


def _devig_consensus(store, api, line):
    """Median de-vigged fair P(over) per (Date, PlayerId) for one market line."""
    m = store[(store["Market"] == api) & ((store["Line"] - line).abs() < 1e-6)]
    recs = []
    for _, r in m.iterrows():
        fair, _hold = O.devig_two_way(r["OverPrice"], r["UnderPrice"])
        if not np.isnan(fair):
            recs.append((r["Date"], r["PlayerId"], fair))
    if not recs:
        return pd.DataFrame(columns=["Date", "PlayerId", "fair"])
    md = pd.DataFrame(recs, columns=["Date", "PlayerId", "fair"])
    md["PlayerId"] = pd.to_numeric(md["PlayerId"], errors="coerce")
    return (md.groupby(["Date", "PlayerId"], dropna=False)["fair"]
            .median().reset_index())


def _mkt_edge(y, p, fair, min_n=20):
    """Relative log-loss edge of the model over the de-vigged consensus on
    the SAME events (+ = model prices them better than the market)."""
    y = np.asarray(y, dtype=float)
    if len(y) < min_n or len(np.unique(y)) < 2:
        return {"mkt": np.nan, "n": int(len(y))}
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    q = np.clip(np.asarray(fair, dtype=float), 1e-4, 1 - 1e-4)
    m_ll, k_ll = log_loss(y, p), log_loss(y, q)
    return {"mkt": 100 * (k_ll - m_ll) / k_ll, "n": int(len(y))}


def _snap_rows(blob):
    df = blob["df"].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["PlayerId"] = pd.to_numeric(df["PlayerId"], errors="coerce")
    return df


def market_year(snap, art, year):
    """{key: {'mkt': rel%, 'n': events}} per market vs the odds store, plus
    '_days' = captured days. Binary props grade their single line; count
    heads grade every captured line not owned by a binary column, priced
    exactly as serving prices it (count_lines)."""
    store = O.load_odds(ODDS_STORE, year=year)
    out = {"_days": int(store["Date"].nunique()) if len(store) else 0}
    if store.empty:
        return out
    heads = art.get("count_models", {})
    k_disp = float(art.get("k_disp", 1.0))
    total_disp = float(art.get("total_disp", 1.0))
    for key in BIN_NAMES:
        meta, blob = O.PROP_MARKET.get(key), snap["binary"].get(key)
        if meta is None or blob is None:
            continue
        cons = _devig_consensus(store, meta["api"], meta["line"])
        if cons.empty:
            continue
        j = _snap_rows(blob).merge(cons, on=["Date", "PlayerId"], how="inner")
        if len(j):
            out[key] = _mkt_edge(j["y"], j["p"], j["fair"])
    owned = {(m["api"], m["line"]) for m in O.PROP_MARKET.values()}
    for key, api in _CNT_API.items():
        blob = snap["count"].get(key)
        if blob is None:
            continue
        lines = sorted(store.loc[store["Market"] == api, "Line"]
                       .dropna().unique())
        if not lines:
            continue
        df = _snap_rows(blob)
        _, pricer = count_lines(key, heads.get(key), k_disp, total_disp)
        ys, ps, fs = [], [], []
        for ln in lines:
            if (api, ln) in owned:
                continue    # that line is graded under its binary column
            cons = _devig_consensus(store, api, ln)
            if cons.empty:
                continue
            j = df.merge(cons, on=["Date", "PlayerId"], how="inner")
            if not len(j):
                continue
            ys.append((j["y"].to_numpy() > ln).astype(float))
            ps.append(np.asarray(pricer(j["mu"].to_numpy(), ln), dtype=float))
            fs.append(j["fair"].to_numpy())
        if ys:
            out[key] = _mkt_edge(np.concatenate(ys), np.concatenate(ps),
                                 np.concatenate(fs))
    return out


def _mkt_cell(mkt, key):
    m = mkt.get(key)
    if m is None:
        return "—"
    if np.isnan(m["mkt"]):
        return f"— (n={m['n']})"
    return f"{m['mkt']:+.1f} (n={m['n']})"


def play_note(kind, m25, m26):
    """Data-driven guidance: shading direction, calibration trust, pick
    depth — derived from the averaged diagnostics, no hand judgment."""
    notes = []
    if kind == "cnt":
        bias = (m25["bias"] + m26["bias"]) / 2
        disp = (m25["disp"] + m26["disp"]) / 2
        rel_scale = max(abs(m25["bias"]), abs(m26["bias"]))
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
        _ = rel_scale
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
    return "; ".join(notes)


def build_table():
    snap25 = joblib.load(ART / "eval_paired_select_2025.joblib")
    snap26 = joblib.load(ART / "eval_paired_2026.joblib")
    art25 = joblib.load(ART / "models_bt.joblib")
    art26 = joblib.load(ART / "models.joblib")

    b25, b26 = binary_year(snap25), binary_year(snap26)
    c25 = count_year(snap25, art25)
    c26 = count_year(snap26, art26)
    # market edge exists only where odds were captured (2026 season store)
    mkt = market_year(snap26, art26, 2026)
    mkt_days = mkt.pop("_days", 0)

    rows = []
    for key, name in BIN_NAMES.items():
        m25, m26 = b25[key], b26[key]
        s25, s26 = binary_score(m25), binary_score(m26)
        rows.append({
            "Market": name, "Key": key, "Score": (s25 + s26) / 2,
            "S25": s25, "S26": s26,
            "AUC": (m25["auc"] + m26["auc"]) / 2,
            "ECE": (m25["ece"] + m26["ece"]) / 2,
            "Bias": np.nan, "Disp": np.nan,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "MktEdge%": _mkt_cell(mkt, key),
            "Notes": play_note("bin", m25, m26),
        })
    for key, name in CNT_MARKETS.items():
        m25, m26 = c25[key], c26[key]
        s25, s26 = count_score(m25), count_score(m26)
        note = play_note("cnt", m25, m26)
        # batter x-columns share their half-point lines with the binary
        # columns, which the workbook glossary calls the sellable numbers
        if key in ("xbk", "xtb", "xhrr"):
            note += "; half-point lines overlap the binary columns"
        rows.append({
            "Market": name, "Key": key, "Score": (s25 + s26) / 2,
            "S25": s25, "S26": s26,
            "AUC": (m25["auc"] + m26["auc"]) / 2,
            "ECE": (m25["ece"] + m26["ece"]) / 2,
            "Bias": (m25["bias"] + m26["bias"]) / 2,
            "Disp": (m25["disp"] + m26["disp"]) / 2,
            "Edge%": (m25["rel"] + m26["rel"]) / 2,
            "MktEdge%": _mkt_cell(mkt, key),
            "Notes": note,
        })
    # moneyline: no per-row snapshot and, more to the point, no proven side
    # edge (McNemar vs always-home is not significant) — pinned informational
    rows.append({"Market": "Games 'Winner' / 'Win Prob' (moneyline)",
                 "Key": "winner",
                 "Score": 0.0, "S25": np.nan, "S26": np.nan,
                 "AUC": np.nan, "ECE": np.nan, "Bias": np.nan,
                 "Disp": np.nan, "Edge%": np.nan, "MktEdge%": "—",
                 "Notes": "no side edge vs always-home (McNemar n.s.) "
                          "-> probability is context, never a bet"})
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    # tiers cut on the composite (0-100) instead of raw edge
    cuts = [(70, "1 ELITE"), (45, "2 STRONG"), (30, "3 SOLID"),
            (18, "4 MARGINAL"), (-1, "5 AVOID")]
    df["Tier"] = df["Score"].map(
        lambda s: next(t for c, t in cuts if s >= c))
    df.insert(0, "#", range(1, len(df) + 1))
    return df[["#", "Market", "Key", "Tier", "Score", "S25", "S26",
               "AUC", "ECE", "Bias", "Disp", "Edge%", "MktEdge%",
               "Notes"]], mkt_days


LEGEND = [
    ("Score", "Overall performance 0-100, averaged over the 2025 and 2026 "
     "held-out test years. Binary columns: edge 40% + AUC 35% + top-pick "
     "lift 25%, scaled by calibration (ECE). O/U columns: per-line edge "
     "55% + rank-correlation 45%, scaled by bias + dispersion sanity."),
    ("S25 / S26", "The same score computed on each test year alone - "
     "agreement between them means the ranking is stable, not one-year "
     "noise."),
    ("AUC", "Ranking skill (0.5 = coin flip). Can it put the players who "
     "DID do it above the ones who didn't? O/U columns: computed per line "
     "through the actual quoted P(over), averaged across the column's line "
     "family - comparable to the binary rows. Batter x-columns cover ONLY "
     "the deep lines they uniquely price (their half-point lines belong to "
     "the binary rows)."),
    ("ECE", "Calibration: average gap between stated probability and "
     "reality. 0 = perfect; above ~0.010 the probability level drifts - "
     "trust the ranking more than the number. O/U columns: per-line ECE of "
     "the quoted prices, averaged across the same line set as AUC."),
    ("Bias", "O/U columns: predicted count minus actual, on average. "
     "Positive = model over-predicts -> lean unders."),
    ("Disp", "O/U columns: error variance / mean. 1.0 = what the P(over) "
     "pricing assumes; higher = real tails wilder than priced -> don't "
     "trust extreme-line probabilities."),
    ("Edge%", "How much better the column prices the event than the "
     "base-rate guess (relative log-loss beat; O/U columns averaged "
     "across every line priced). Held-out years only, no scraped odds."),
    ("MktEdge%", "Model vs the de-vigged sportsbook consensus on the SAME "
     "events, from the scraped odds store (+ = model prices them better "
     "than the market). Display-only - never part of Score - and pooled "
     "over every captured line a column owns. Anecdote until ~15+ scraped "
     "days accrue; '(n=...)' is the graded event count."),
    ("Tier", "1 ELITE / 2 STRONG: trust and act on these. 3 SOLID: usable "
     "with the noted caveat. 4 MARGINAL / 5 AVOID: the model cannot "
     "separate players well enough to act on."),
]


def save_excel(df, path):
    """Write the rankings workbook (Rankings + Legend), styled like the
    prediction workbooks via predict._polish."""
    xl = df.copy()
    for c, nd in [("Score", 0), ("S25", 0), ("S26", 0), ("AUC", 3),
                  ("ECE", 4), ("Bias", 2), ("Disp", 2), ("Edge%", 2)]:
        xl[c] = xl[c].round(nd)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        xl.to_excel(xw, sheet_name="Rankings", index=False)
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

    df, mkt_days = build_table()
    out = df.copy()
    for c in ("Score", "S25", "S26"):
        out[c] = out[c].map(lambda v: f"{v:.0f}" if pd.notna(v) else "-")
    out["Edge%"] = out["Edge%"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")
    out["AUC"] = out["AUC"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    out["ECE"] = out["ECE"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "-")
    out["Bias"] = out["Bias"].map(
        lambda v: f"{v:+.2f}" if pd.notna(v) else "-")
    out["Disp"] = out["Disp"].map(
        lambda v: f"{v:.2f}" if pd.notna(v) else "-")

    print("\n=== Prediction-column performance rankings — held-out test "
          "years 2025 + 2026 averaged ===\n")
    print(out.to_string(index=False))
    print(f"\n  MktEdge%: model vs de-vigged book consensus on the same "
          f"events — {mkt_days} scraped day(s) in the odds store"
          + ("; ANECDOTE until ~15+ days accrue." if mkt_days < 15
             else "."))
    print("\n  Calibration: ECE = mean |predicted % - actual %| (binary "
          "columns; 0 = perfect, >.010 = probability level off).")
    print("               Bias = predicted minus actual count; Disp = "
          "variance/mean of the errors (O/U columns; 1.0 = the")
    print("               P(over) math's assumption — higher means real "
          "tails are wilder than priced).")
    print("  Performance: AUC = ranking skill (binary), Edge% = log-loss "
          "beat over the base-rate guess, from the held-out")
    print("               test years only — no training-year data, no "
          "scraped odds.")
    print("  Score: binary = edge 40% + AUC 35% + top-pick lift 25%, "
          "scaled by calibration (ECE);")
    print("         O/U    = per-line edge 55% + rank-correlation 45%, "
          "scaled by bias + dispersion sanity.")
    print("  S25/S26 = the score per test year; agreement across years = "
          "stability.")

    save_excel(df, Path(args.out))
    print(f"\n  written to {args.out}")


if __name__ == "__main__":
    main()
