"""MiLB priors — the translation layer that feeds BOTH models (built once).

From Data/milb_batting.csv / milb_pitching.csv (Scrapers/scrape_milb.py):
level-translated MLB-equivalent 8-class PA-outcome rates per (player,
serve-season), for

  (a) the PA model's empirical-Bayes career prior: a thin-MLB-history
      player's career rates shrink toward his translated MiLB line instead
      of the league mix (consumers: pa_model.add_asof_features,
      pa_backtest.build_features, pa_engine.MatchupFeatures);
  (b) main-model feature columns riding the Phase-3 retrain batch
      (genuinely new information; the exhausted-features verdict does not
      cover it) — staged in Model/FEATURE_BACKLOG.md, features.py is
      deliberately untouched until that batch (code-fingerprint guard).

TRANSLATION. Per (level, class) log-odds offsets fit on movers: player-
seasons with >=MIN_PAIR_PA at a level and >=MIN_PAIR_PA in the majors the
same or following year (the PA frame supplies the MLB side). Offset =
logit(pooled MLB rate) - logit(pooled MiLB rate), each player weighted by
min(milb_n, mlb_n) on both sides so selection cancels: the SAME players'
lines sit on each side of the difference. Levels whose paired weight is
below MIN_LEVEL_W never fitted -> never served (empirically answers
"which levels carry signal"). Offsets are fit on MLB seasons <=
FIT_MAX_MLB_SEASON only, so the 2025/2026 shadow backtests never see a
table informed by their own years.

SERVING (leakage rule from the scraper docstring: season aggregates ->
join season <= serve_season - 1 ONLY). A serve-season row pools the
player's translated level-lines from the previous SERVE_LOOKBACK seasons,
weighted PA * SERVE_DECAY^(years back - 1); n_eff = that weight sum *
MILB_N_DISCOUNT (a translated MiLB PA is worth half an MLB PA of prior
evidence). The EB blend toward league is prior_blend(); K_LG_MIX anchors
thin MiLB careers to the league mix.

Usage:
    python Model/milb_priors.py [--rebuild]   # build + diagnostics
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pa_model import CLASSES, FRAME_CACHE

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "Data"
ART = HERE / "artifacts"
CACHE = ART / "milb_priors.joblib"

FIT_MAX_MLB_SEASON = 2024     # pairs never see the backtest years
MIN_PAIR_PA = 100             # both sides of a fitting pair
MIN_LEVEL_W = 10_000          # paired PAs to trust a level's offsets
SERVE_LOOKBACK = 3            # MiLB seasons Y-1 .. Y-LOOKBACK feed season Y
SERVE_DECAY = 0.6             # per extra year back
MILB_N_DISCOUNT = 0.5         # translated PA -> prior-evidence PA
K_LG_MIX = 300.0              # league anchor inside the player prior

# ---- audit-wave v1 extras (2026-07-14, fit-free — no level offsets) ----
# Steal prior: constants MEASURED on milb_batting.csv; success is flat
# 0.720-0.723 across the kept levels, the empirical license for skipping
# per-level translation. Attempt-rate level drift is left to GBM ordering.
K_ATT, LG_ATT = 40.0, 0.145   # league MiLB attempt rate per opportunity
K_SUCC, LG_SUCC = 20.0, 0.72  # kept-level success rate
# Workload: starter-role filter — season IP includes relief innings, so
# outs/start is only meaningful on predominantly-starting seasons (naive
# 3*IP/GS is impossible >30 on 17.3% of unfiltered GS>=1 rows).
GS_ROLE_SHARE = 0.6           # GS >= this share of G to count as a starter


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ------------------------------------------------------- class counts ---

def _milb_counts(kind):
    """One row per (PlayerId, Season, Level): n + the 8 class counts,
    plus the audit-wave v1 extras — bat: SB/CS/OPP steal counts; pit:
    G/GS/IPthirds workload counts (IP converted to outs BEFORE pooling,
    since the .1/.2 thirds notation does not sum)."""
    if kind == "bat":
        d = pd.read_csv(DATA / "milb_batting.csv")
        d["n"] = pd.to_numeric(d["PA"], errors="coerce")
    else:
        d = pd.read_csv(DATA / "milb_pitching.csv")
        d["n"] = pd.to_numeric(d["TBF"], errors="coerce")
    for c in ("H", "2B", "3B", "HR", "BB", "HBP", "SO"):
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d = d[d["n"] > 0].copy()
    d["K"] = d["SO"]
    d["1B"] = (d["H"] - d["2B"] - d["3B"] - d["HR"]).clip(lower=0)
    d["OUT"] = (d["n"] - d["SO"] - d["BB"] - d["HBP"] - d["H"]).clip(lower=0)
    if kind == "bat":
        for c in ("SB", "CS"):
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
        d["OPP"] = d["1B"] + d["BB"] + d["HBP"]   # times on first-ish
        extras = ["SB", "CS", "OPP"]
    else:
        for c in ("G", "GS"):
            d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
        ip = pd.to_numeric(d["IP"], errors="coerce").fillna(0)
        d["IPthirds"] = np.floor(ip) * 3 + ((ip % 1) * 10).round()
        extras = ["G", "GS", "IPthirds"]
    out = d[["PlayerId", "Year", "Level", "n"] + CLASSES + extras].rename(
        columns={"Year": "Season"})
    return (out.groupby(["PlayerId", "Season", "Level"], as_index=False)
               .sum())                       # (rare) duplicate splits pooled


def _mlb_counts(pa, key):
    """Per (player, Season) MLB class counts + n from the PA frame."""
    g = (pa.groupby([key, "Season", "cls"], observed=True).size()
           .unstack("cls", fill_value=0)
           .reindex(columns=CLASSES, fill_value=0))
    g["n"] = g.sum(axis=1)
    return g.reset_index().rename(columns={key: "PlayerId"})


# ------------------------------------------------------------ fitting ---

def fit_offsets(milb, mlb):
    """{level: offsets[8]} + per-level diagnostics from mover pairs."""
    pairs = []
    for gap in (0, 1):
        m = mlb.copy()
        m["Season"] = m["Season"] - gap    # MiLB season this MLB year pairs
        pairs.append(milb.merge(m, on=["PlayerId", "Season"],
                                suffixes=("_mi", "_ml")))
    p = pd.concat(pairs, ignore_index=True)
    p = p[(p["n_mi"] >= MIN_PAIR_PA) & (p["n_ml"] >= MIN_PAIR_PA)
          & (p["Season"] + 1 <= FIT_MAX_MLB_SEASON)]
    p["w"] = np.minimum(p["n_mi"], p["n_ml"])

    offsets, diag = {}, []
    for lvl, g in p.groupby("Level"):
        w = g["w"].to_numpy()[:, None]
        r_mi = (g[[f"{c}_mi" for c in CLASSES]].to_numpy() + 0.5) \
            / (g["n_mi"].to_numpy()[:, None] + 4.0)
        r_ml = (g[[f"{c}_ml" for c in CLASSES]].to_numpy() + 0.5) \
            / (g["n_ml"].to_numpy()[:, None] + 4.0)
        pool_mi = (w * r_mi).sum(0) / w.sum()
        pool_ml = (w * r_ml).sum(0) / w.sum()
        off = _logit(pool_ml) - _logit(pool_mi)
        kept = w.sum() >= MIN_LEVEL_W
        if kept:
            offsets[lvl] = off
        diag.append({"Level": lvl, "pairs": len(g), "w": int(w.sum()),
                     "kept": kept,
                     **{f"off_{c}": round(float(o), 3)
                        for c, o in zip(CLASSES, off)}})
    return offsets, pd.DataFrame(diag).set_index("Level")


# ------------------------------------------------------------ serving ---

def build_serving(milb, offsets, seasons):
    """One row per (PlayerId, serve Season): translated pooled rates t_* +
    n_eff. Only levels with fitted offsets contribute."""
    d = milb[milb["Level"].isin(offsets)].copy()
    rates = (d[CLASSES].to_numpy() + 0.5) / (d["n"].to_numpy()[:, None] + 4.0)
    off = np.stack([offsets[l] for l in d["Level"]])
    tr = 1 / (1 + np.exp(-(_logit(rates) + off)))
    tr /= tr.sum(axis=1, keepdims=True)
    d[[f"t_{c}" for c in CLASSES]] = tr

    parts = []
    for y in seasons:
        win = d[(d["Season"] >= y - SERVE_LOOKBACK) & (d["Season"] <= y - 1)]
        if not len(win):
            continue
        w = (win["n"] * SERVE_DECAY ** (y - 1 - win["Season"])).to_numpy()
        g = win[["PlayerId"]].copy()
        g["w"] = w
        for c in CLASSES:
            g[f"t_{c}"] = win[f"t_{c}"].to_numpy() * w
        agg = g.groupby("PlayerId", as_index=False).sum()
        for c in CLASSES:
            agg[f"t_{c}"] /= agg["w"]
        agg["n_eff"] = agg["w"] * MILB_N_DISCOUNT
        agg["Season"] = y
        parts.append(agg.drop(columns=["w"]))
    return pd.concat(parts, ignore_index=True)


def prior_blend(lg, t, n_eff, k=K_LG_MIX):
    """EB mix of the league prior toward translated MiLB rates.
    lg, t: (n, 8) arrays (t rows may be NaN = no MiLB); n_eff: (n,)."""
    t = np.where(np.isnan(t), lg, t)
    ne = np.nan_to_num(np.asarray(n_eff, float))[..., None]
    return (ne * t + k * lg) / (ne + k)


# ------------------------------------------- audit-wave v1 extras -------

def _bat_steal_extras(milb, offsets, seasons):
    """Fit-free steal prior per (PlayerId, serve Season): shrunk attempt
    rate per opportunity + shrunk success rate from decayed kept-level
    COUNT sums (decay weight only — counts are already mass). No level
    offsets (success measured flat 0.720-0.723 across kept levels)."""
    d = milb[milb["Level"].isin(offsets)]
    parts = []
    for y in seasons:
        win = d[(d["Season"] >= y - SERVE_LOOKBACK) & (d["Season"] <= y - 1)]
        if not len(win):
            continue
        w = SERVE_DECAY ** (y - 1 - win["Season"]).to_numpy()
        g = pd.DataFrame({"PlayerId": win["PlayerId"].to_numpy(),
                          "sb_w": win["SB"].to_numpy() * w,
                          "cs_w": win["CS"].to_numpy() * w,
                          "opp_w": win["OPP"].to_numpy() * w})
        agg = g.groupby("PlayerId", as_index=False).sum()
        att = agg["sb_w"] + agg["cs_w"]
        agg["milb_att"] = (att + K_ATT * LG_ATT) / (agg["opp_w"] + K_ATT)
        agg["milb_sb_succ"] = ((agg["sb_w"] + K_SUCC * LG_SUCC)
                               / (att + K_SUCC))
        agg["Season"] = y
        parts.append(agg[["PlayerId", "Season", "milb_att", "milb_sb_succ"]])
    return pd.concat(parts, ignore_index=True)


def _pit_workload_extras(milb, offsets, seasons):
    """Workload pedigree per (PlayerId, serve Season): pmilb_outs_ps =
    decayed outs per start over starter-role seasons only (GS >= 1 and
    GS >= GS_ROLE_SHARE * G — season IP includes relief innings, so
    mixed-role rows are excluded); pmilb_gs_share = decayed GS/G over ALL
    kept-level rows (a pure-reliever pedigree of 0 is the signal)."""
    d = milb[milb["Level"].isin(offsets)]
    starter = d[(d["GS"] >= 1) & (d["GS"] >= GS_ROLE_SHARE * d["G"])]
    parts = []
    for y in seasons:
        win = d[(d["Season"] >= y - SERVE_LOOKBACK) & (d["Season"] <= y - 1)]
        if not len(win):
            continue
        w = SERVE_DECAY ** (y - 1 - win["Season"]).to_numpy()
        g = pd.DataFrame({"PlayerId": win["PlayerId"].to_numpy(),
                          "gs_w": win["GS"].to_numpy() * w,
                          "g_w": win["G"].to_numpy() * w})
        agg = g.groupby("PlayerId", as_index=False).sum()
        agg["pmilb_gs_share"] = agg["gs_w"] / agg["g_w"].where(agg["g_w"] > 0)
        st = starter[(starter["Season"] >= y - SERVE_LOOKBACK)
                     & (starter["Season"] <= y - 1)]
        if len(st):
            ws = SERVE_DECAY ** (y - 1 - st["Season"]).to_numpy()
            s = pd.DataFrame({"PlayerId": st["PlayerId"].to_numpy(),
                              "outs_w": st["IPthirds"].to_numpy() * ws,
                              "gs_w2": st["GS"].to_numpy() * ws})
            s = s.groupby("PlayerId", as_index=False).sum()
            s["pmilb_outs_ps"] = s["outs_w"] / s["gs_w2"].where(s["gs_w2"] > 0)
            agg = agg.merge(s[["PlayerId", "pmilb_outs_ps"]],
                            on="PlayerId", how="left")
        else:
            agg["pmilb_outs_ps"] = np.nan
        agg["Season"] = y
        parts.append(agg[["PlayerId", "Season",
                          "pmilb_gs_share", "pmilb_outs_ps"]])
    return pd.concat(parts, ignore_index=True)


# -------------------------------------------------------------- build ---

def build_all(force=False):
    if CACHE.exists() and not force:
        return joblib.load(CACHE)
    pa = pd.read_parquet(FRAME_CACHE)
    seasons = range(int(pa["Season"].min()), int(pa["Season"].max()) + 1)
    out = {}
    for kind, key in (("bat", "batter"), ("pit", "pitcher")):
        milb = _milb_counts(kind)
        offsets, diag = fit_offsets(milb, _mlb_counts(pa, key))
        serve = build_serving(milb, offsets, seasons)
        # audit-wave v1 extras (2026-07-14): fit-free, same window/levels
        extras = (_bat_steal_extras if kind == "bat"
                  else _pit_workload_extras)(milb, offsets, seasons)
        serve = serve.merge(extras, on=["PlayerId", "Season"], how="left")
        out[kind] = {"serve": serve, "offsets": offsets, "diag": diag}
        log(f"{kind}: levels kept {sorted(offsets)} | serve rows "
            f"{len(serve):,} ({serve['PlayerId'].nunique():,} players)")
    joblib.dump(out, CACHE, compress=3)
    log(f"cached -> {CACHE.name}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    t = build_all(force=args.rebuild)
    for kind in ("bat", "pit"):
        print(f"\n=== {kind}: per-level offsets (logit, MLB minus MiLB; "
              f"kept = weight >= {MIN_LEVEL_W:,}) ===")
        print(t[kind]["diag"].to_string())
        s = t[kind]["serve"]
        cov = s[s["Season"] == 2025]
        print(f"2025 coverage: {len(cov):,} players | median n_eff "
              f"{cov['n_eff'].median():.0f} | mean t_K "
              f"{cov['t_K'].mean():.3f} | mean t_HR {cov['t_HR'].mean():.3f}")
