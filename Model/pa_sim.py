"""PA-sim Phase 2, data layer — the pieces the Monte-Carlo game engine
composes around the Phase-1 PA outcome model (pa_model.joblib):

  1. ADVANCEMENT TRANSITIONS  P(d_outs, runs, next_bases | class, bases, outs)
     estimated empirically from the pitch archive (every terminal PA carries
     base-out state, and bat_score/post_bat_score give runs on the PA; the
     next PA in the same half-inning gives the post state). Fit on <=2024
     ONLY so the 2025/2026 shadow backtest stays pure. Hierarchical fallback
     for thin cells: full cell -> (class, outs) -> class marginal.
  2. STARTER HOOK  per-pitcher as-of batters-faced history (sim samples a
     start's BF from the pitcher's own recent starts, league fallback).
  3. BULLPEN CHAIN  per-team season-to-date day-start relief-allowed class
     rates from the same archive — the "team pen" pseudo-pitcher whose rates
     feed the PA model's pitcher slots once the starter is hooked.

Everything caches to artifacts/pa_sim_tables.joblib. Run standalone to
build + sanity-check (advancement numbers print against known league
baselines, e.g. runner-on-2nd scores on a single ~60%).
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pa_model import CLASSES, EVENT_CLS, RAW

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
TABLE_CACHE = ART / "pa_sim_tables.joblib"

TRANS_MAX_SEASON = 2024      # transitions never see the backtest years
MAX_RUNS = 4                 # runs on one PA capped (grand slam)
STEAL_K_ATT = 50.0           # EB prior strength, in times-on-first
STEAL_K_SUCC = 15.0          # EB prior strength, in attempts

# starter-hook hazard table dimensions: workload tier (trailing 30-start
# mean BF, bucketed at TIER_EDGES), batters faced so far (clipped to
# HOOK_BF_MAX), runs allowed so far (clipped to HOOK_RUN_MAX). Fit on
# seasons <= TRANS_MAX_SEASON with exponential recency weights (usage
# norms drift: openers, 3-batter rule, pitch clock).
TIER_EDGES = [18.0, 21.0, 23.0, 25.0]     # 5 tiers
HOOK_BF_MAX = 40
HOOK_RUN_MAX = 7
HOOK_HALF_LIFE = 3.0                      # seasons
# hazard v2 (2026-07-14, the queued outs fix): the hook is indexed by BF
# RELATIVE to the pitcher's OWN expected BF (trailing 30-start mean), not
# absolute BF within coarse tiers — a 19-BF opener and a 21-BF fifth
# starter shared a v1 tier and identical curves; v2 centers every curve on
# the pitcher's own leash. rel = bf - round(exp_bf), clipped to
# [-HOOK_REL_LO, +HOOK_REL_HI]; consumers slice back to an absolute-BF
# table via hazard_slice_v2 so the engine is unchanged.
HOOK_REL_LO = 16
HOOK_REL_HI = 8
EXP_BF_DEFAULT = 23.0
EXP_BF_MIN_STARTS = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------- transitions ----

def _load_terminal(season_max=TRANS_MAX_SEASON):
    cols = ["game_pk", "game_date", "events", "at_bat_number", "inning",
            "inning_topbot", "on_1b", "on_2b", "on_3b", "outs_when_up",
            "bat_score", "post_bat_score"]
    parts = []
    for f in sorted(RAW.glob("pitches_*.parquet")):
        year = int(f.stem.split("_")[1])
        if year > season_max:
            continue
        df = pd.read_parquet(f, columns=cols)
        df = df[df["events"].isin(EVENT_CLS)].copy()
        df["cls"] = df["events"].map(EVENT_CLS)
        parts.append(df.drop(columns=["events"]))
    pa = pd.concat(parts, ignore_index=True)
    pa["bases"] = (pa["on_1b"].notna().astype(int)
                   + 2 * pa["on_2b"].notna().astype(int)
                   + 4 * pa["on_3b"].notna().astype(int))
    pa["runs"] = (pa["post_bat_score"] - pa["bat_score"]).clip(0, MAX_RUNS)
    return pa


def build_transitions():
    """(cls, bases, outs) -> arrays of outcomes (d_outs, runs, next_bases)
    with probabilities; packed for vectorized sampling."""
    pa = _load_terminal()
    pa = pa.sort_values(["game_pk", "inning", "inning_topbot",
                         "at_bat_number"], kind="stable")
    grp = ["game_pk", "inning", "inning_topbot"]
    nxt = pa.groupby(grp, sort=False)[["bases", "outs_when_up"]].shift(-1)
    pa["next_bases"] = nxt["bases"]
    pa["next_outs"] = nxt["outs_when_up"]
    # half-inning ended (no next PA): inning ends at 3 outs, bases moot
    ended = pa["next_bases"].isna()
    pa.loc[ended, "next_bases"] = 0
    pa.loc[ended, "next_outs"] = 3
    pa["d_outs"] = (pa["next_outs"] - pa["outs_when_up"]).clip(0, 3)
    # walk-off endings can leave d_outs 0 on an inning-ending row; the
    # empirical mass there is tiny and harmless (sim ends half-innings on
    # its own outs counter)
    pa["cls_id"] = pa["cls"].map({c: i for i, c in enumerate(CLASSES)})

    counts = (pa.groupby(["cls_id", "bases", "outs_when_up",
                          "d_outs", "runs", "next_bases"], observed=True)
                .size().rename("n").reset_index())

    # hierarchical fallback distributions
    by_cls_outs = (pa.groupby(["cls_id", "outs_when_up",
                               "d_outs", "runs"], observed=True)
                     .size().rename("n").reset_index())
    by_cls = (pa.groupby(["cls_id", "d_outs", "runs"], observed=True)
                .size().rename("n").reset_index())

    cells = {}
    for (ci, b, o), g in counts.groupby(["cls_id", "bases", "outs_when_up"]):
        if g["n"].sum() < 30:        # thin cell -> fallback below
            continue
        p = (g["n"] / g["n"].sum()).to_numpy()
        cells[(int(ci), int(b), int(o))] = {
            "p": p.astype(np.float64),
            "d_outs": g["d_outs"].to_numpy(np.int8),
            "runs": g["runs"].to_numpy(np.int8),
            "next_bases": g["next_bases"].to_numpy(np.int8)}

    fallback = {}
    for (ci, o), g in by_cls_outs.groupby(["cls_id", "outs_when_up"]):
        p = (g["n"] / g["n"].sum()).to_numpy()
        fallback[(int(ci), int(o))] = {
            "p": p.astype(np.float64),
            "d_outs": g["d_outs"].to_numpy(np.int8),
            "runs": g["runs"].to_numpy(np.int8)}
    fallback_cls = {}
    for ci, g in by_cls.groupby("cls_id"):
        p = (g["n"] / g["n"].sum()).to_numpy()
        fallback_cls[int(ci)] = {
            "p": p.astype(np.float64),
            "d_outs": g["d_outs"].to_numpy(np.int8),
            "runs": g["runs"].to_numpy(np.int8)}

    log(f"transitions: {len(cells)} full cells "
        f"(of {len(CLASSES) * 8 * 3}), {len(fallback)} class-outs fallbacks")
    return {"cells": cells, "fallback": fallback,
            "fallback_cls": fallback_cls,
            "fit_through": TRANS_MAX_SEASON}


# ----------------------------------------------------- starter hook -----

def build_starter_bf():
    """Per-pitcher chronological (Date, BF) start history + the league BF
    list per season — the sim samples a start's length from the pitcher's
    trailing starts as-of the sim date (strictly earlier dates only)."""
    gp = pd.read_csv(HERE.parent / "Data" / "mlb_game_pitching.csv",
                     usecols=["PlayerId", "Date", "Season", "GS", "BF"])
    st = gp[(gp["GS"] == 1) & gp["BF"].notna()].copy()
    st["Date"] = pd.to_datetime(st["Date"])
    st = st.sort_values("Date")
    hist = {int(pid): (g["Date"].to_numpy(), g["BF"].to_numpy(np.int16))
            for pid, g in st.groupby("PlayerId", sort=False)}
    league = {int(s): g["BF"].to_numpy(np.int16)
              for s, g in st.groupby("Season", sort=False)}
    log(f"starter BF: {len(hist)} pitchers, {len(st):,} starts")
    return {"by_pitcher": hist, "league_by_season": league}


# ----------------------------------------------------- bullpen rates ----

def build_pen_rates():
    """Team relief-allowed class rates, season-to-date day-start (the game's
    own day excluded), shrunk toward the league pen mix. Relief PA = any PA
    whose pitcher is not the fielding team's first pitcher of that game.
    Returns per (team, date) rows -> 8 class rates + n."""
    cols = ["game_pk", "game_date", "events", "pitcher", "inning_topbot",
            "home_team", "away_team"]
    parts = []
    for f in sorted(RAW.glob("pitches_*.parquet")):
        df = pd.read_parquet(f, columns=cols)
        df = df[df["events"].isin(EVENT_CLS)].copy()
        df["cls"] = df["events"].map(EVENT_CLS)
        parts.append(df.drop(columns=["events"]))
    pa = pd.concat(parts, ignore_index=True)
    pa["Date"] = pd.to_datetime(pa["game_date"])
    pa["Season"] = pa["Date"].dt.year
    pa["fld_team"] = np.where(pa["inning_topbot"] == "Top",
                              pa["home_team"], pa["away_team"])
    pa = pa.sort_values(["game_pk", "Date"], kind="stable")
    # the fielding team's starter = its first recorded pitcher in the game
    first = (pa.groupby(["game_pk", "fld_team"], sort=False)["pitcher"]
               .transform("first"))
    rel = pa[pa["pitcher"] != first]

    day = (rel.groupby(["fld_team", "Season", "Date", "cls"], observed=True)
              .size().unstack("cls", fill_value=0)
              .reindex(columns=CLASSES, fill_value=0).sort_index())
    g = day.groupby(level=[0, 1], sort=False)
    cum = g.cumsum() - day                      # day-start season-to-date
    n = cum.sum(axis=1)

    lg_day = (rel.groupby(["Season", "Date", "cls"], observed=True)
                 .size().unstack("cls", fill_value=0)
                 .reindex(columns=CLASSES, fill_value=0).sort_index())
    lg_cum = lg_day.groupby(level=0, sort=False).cumsum() - lg_day
    lg_rate = lg_cum.div(lg_cum.sum(axis=1), axis=0)

    out = cum.reset_index()
    lg_on_rows = lg_rate.reindex(
        pd.MultiIndex.from_arrays([out["Season"], out["Date"]])).to_numpy()
    K = 300.0                                   # prior strength in PAs
    rates = (out[CLASSES].to_numpy() + K * np.nan_to_num(lg_on_rows, nan=1/8)
             ) / (n.to_numpy()[:, None] + K)
    rates /= rates.sum(axis=1, keepdims=True)
    out[[f"pen_{c}" for c in CLASSES]] = rates
    out["pen_n"] = n.to_numpy()
    out = out.rename(columns={"fld_team": "Team"})
    log(f"pen rates: {len(out):,} team-days")
    return out[["Team", "Season", "Date", "pen_n"]
               + [f"pen_{c}" for c in CLASSES]]


# ----------------------------------------------------- starter hook v2 --

def build_starter_hazard(starter_bf):
    """P(starter is hooked before facing another batter | workload tier,
    batters faced so far, runs allowed so far) — empirical from every
    start <= TRANS_MAX_SEASON, recency-weighted (half-life
    HOOK_HALF_LIFE seasons). Each PA a starter faced contributes one row
    at its POST-PA state with y = "that was his last batter". Hierarchical
    fallback tier -> pooled (bf, runs) -> pooled bf for thin cells.
    Returns hazard[tier, bf, runs] (float32)."""
    cols = ["game_pk", "game_date", "events", "pitcher", "inning_topbot",
            "at_bat_number", "bat_score", "post_bat_score"]
    parts = []
    for f in sorted(RAW.glob("pitches_*.parquet")):
        if int(f.stem.split("_")[1]) > TRANS_MAX_SEASON:
            continue
        df = pd.read_parquet(f, columns=cols)
        df = df[df["events"].isin(EVENT_CLS)]
        parts.append(df)
    pa = pd.concat(parts, ignore_index=True)
    pa["Date"] = pd.to_datetime(pa["game_date"])
    pa["Season"] = pa["Date"].dt.year
    pa = pa.sort_values(["game_pk", "inning_topbot", "at_bat_number"],
                        kind="stable")
    # the half-inning's first recorded pitcher = that side's starter
    grp = ["game_pk", "inning_topbot"]
    first = pa.groupby(grp, sort=False)["pitcher"].transform("first")
    st = pa[pa["pitcher"] == first].copy()
    st["bf"] = st.groupby(grp, sort=False).cumcount() + 1   # post-PA count
    st["run1"] = (st["post_bat_score"] - st["bat_score"]).clip(0, MAX_RUNS)
    st["runs"] = st.groupby(grp, sort=False)["run1"].cumsum()
    st["last"] = (st["bf"] == st.groupby(grp, sort=False)["bf"]
                  .transform("max")).astype(np.int8)

    # workload tier from the pitcher's trailing 30-start mean BF, as-of
    tiers = np.zeros(len(st), np.int8)
    by_p = starter_bf["by_pitcher"]
    keys = st[["pitcher", "Date"]].drop_duplicates()
    tmap = {}
    for pid, date in keys.itertuples(index=False):
        dates, bfs = by_p.get(int(pid), (None, None))
        mean_bf = 23.0
        if dates is not None:
            i = np.searchsorted(dates, np.datetime64(date))
            hist = bfs[max(0, i - 30):i]
            if len(hist) >= 5:
                mean_bf = float(hist.mean())
        tmap[(pid, date)] = int(np.searchsorted(TIER_EDGES, mean_bf))
    tiers = np.array([tmap[(p, d)] for p, d in
                      zip(st["pitcher"], st["Date"])], np.int8)

    bf = st["bf"].clip(upper=HOOK_BF_MAX).to_numpy()
    rn = st["runs"].clip(upper=HOOK_RUN_MAX).to_numpy(np.int64)
    y = st["last"].to_numpy()
    w = 0.5 ** ((TRANS_MAX_SEASON - st["Season"].to_numpy())
                / HOOK_HALF_LIFE)

    n_tier = len(TIER_EDGES) + 1
    shape = (n_tier, HOOK_BF_MAX + 1, HOOK_RUN_MAX + 1)
    num = np.zeros(shape)
    den = np.zeros(shape)
    np.add.at(num, (tiers, bf, rn), y * w)
    np.add.at(den, (tiers, bf, rn), w)
    pool_num, pool_den = num.sum(0), den.sum(0)       # tier-pooled
    bf_num, bf_den = pool_num.sum(1), pool_den.sum(1)  # bf marginal
    MIN_W = 200.0
    hz = np.where(den >= MIN_W, num / np.maximum(den, 1e-9),
                  np.where(pool_den >= MIN_W,
                           pool_num / np.maximum(pool_den, 1e-9),
                           (bf_num / np.maximum(bf_den, 1e-9))[None, :, None]))
    hz = np.clip(hz, 0.0, 1.0)
    hz[:, HOOK_BF_MAX, :] = np.maximum(hz[:, HOOK_BF_MAX, :], 0.5)
    log(f"starter hazard: {len(st):,} starter PAs, "
        f"{int((den >= MIN_W).sum())}/{den.size} full cells; league mean "
        f"BF check in engine smoke")
    return {"hazard": hz.astype(np.float32), "tier_edges": TIER_EDGES}


def starter_exp_bf(bf_tables, starter, date, k_last=30):
    """The pitcher's trailing k_last-start mean BF as-of `date` — his own
    leash. League default under EXP_BF_MIN_STARTS. Shared by the v2 table
    build, the backtest and serving (one definition, no drift)."""
    dates, bfs = bf_tables["by_pitcher"].get(int(starter), (None, None))
    if dates is not None:
        i = np.searchsorted(dates, np.datetime64(date))
        hist = bfs[max(0, i - k_last):i]
        if len(hist) >= EXP_BF_MIN_STARTS:
            return float(hist.mean())
    return EXP_BF_DEFAULT


def build_starter_hazard_v2(starter_bf):
    """Hazard v2 (2026-07-14): P(hooked before the next batter | bf RELATIVE
    to his own expected BF, runs allowed so far). Same PA rows, recency
    weights and fallback idiom as v1; the tier dimension is REPLACED by the
    relative-BF index, which is what the tiers were coarsely approximating.
    Returns hazard[rel (HOOK_REL_LO+HOOK_REL_HI+1), runs] (float32)."""
    cols = ["game_pk", "game_date", "events", "pitcher", "inning_topbot",
            "at_bat_number", "bat_score", "post_bat_score"]
    parts = []
    for f in sorted(RAW.glob("pitches_*.parquet")):
        if int(f.stem.split("_")[1]) > TRANS_MAX_SEASON:
            continue
        df = pd.read_parquet(f, columns=cols)
        df = df[df["events"].isin(EVENT_CLS)]
        parts.append(df)
    pa = pd.concat(parts, ignore_index=True)
    pa["Date"] = pd.to_datetime(pa["game_date"])
    pa["Season"] = pa["Date"].dt.year
    pa = pa.sort_values(["game_pk", "inning_topbot", "at_bat_number"],
                        kind="stable")
    grp = ["game_pk", "inning_topbot"]
    first = pa.groupby(grp, sort=False)["pitcher"].transform("first")
    st = pa[pa["pitcher"] == first].copy()
    st["bf"] = st.groupby(grp, sort=False).cumcount() + 1   # post-PA count
    st["run1"] = (st["post_bat_score"] - st["bat_score"]).clip(0, MAX_RUNS)
    st["runs"] = st.groupby(grp, sort=False)["run1"].cumsum()
    st["last"] = (st["bf"] == st.groupby(grp, sort=False)["bf"]
                  .transform("max")).astype(np.int8)

    # each start's expected BF (the pitcher's own leash, as-of)
    keys = st[["pitcher", "Date"]].drop_duplicates()
    emap = {(p, d): starter_exp_bf(starter_bf, p, d)
            for p, d in keys.itertuples(index=False)}
    exp_bf = np.array([emap[(p, d)] for p, d in
                       zip(st["pitcher"], st["Date"])])
    rel = np.clip(st["bf"].to_numpy() - np.round(exp_bf).astype(int),
                  -HOOK_REL_LO, HOOK_REL_HI) + HOOK_REL_LO
    rn = st["runs"].clip(upper=HOOK_RUN_MAX).to_numpy(np.int64)
    y = st["last"].to_numpy()
    w = 0.5 ** ((TRANS_MAX_SEASON - st["Season"].to_numpy())
                / HOOK_HALF_LIFE)

    shape = (HOOK_REL_LO + HOOK_REL_HI + 1, HOOK_RUN_MAX + 1)
    num = np.zeros(shape)
    den = np.zeros(shape)
    np.add.at(num, (rel, rn), y * w)
    np.add.at(den, (rel, rn), w)
    rel_num, rel_den = num.sum(1), den.sum(1)          # rel marginal
    MIN_W = 200.0
    hz = np.where(den >= MIN_W, num / np.maximum(den, 1e-9),
                  (rel_num / np.maximum(rel_den, 1e-9))[:, None])
    hz = np.clip(hz, 0.0, 1.0)
    # hard stop at the top of the relative range: 8+ batters past his own
    # norm, the manager is coming regardless of the scoreboard
    hz[-1, :] = np.maximum(hz[-1, :], 0.5)
    log(f"starter hazard v2: {len(st):,} starter PAs, "
        f"{int((den >= MIN_W).sum())}/{den.size} full cells (rel-BF x runs)")
    return {"hazard": hz.astype(np.float32),
            "rel_lo": HOOK_REL_LO, "rel_hi": HOOK_REL_HI}


def hazard_slice_v2(hz2, exp_bf):
    """Absolute-BF hook table for ONE starter from the v2 relative table:
    hz_abs[bf, runs] = hz2[clip(bf - round(exp_bf))]. The engine's hook
    check (hz[bf, runs]) is unchanged — only the slice differs. The v1
    absolute-BF hard stop at HOOK_BF_MAX is re-applied."""
    bf_idx = np.arange(HOOK_BF_MAX + 1)
    rel = np.clip(bf_idx - int(round(exp_bf)), -HOOK_REL_LO, HOOK_REL_HI) \
        + HOOK_REL_LO
    out = hz2[rel, :].copy()
    out[HOOK_BF_MAX, :] = np.maximum(out[HOOK_BF_MAX, :], 0.5)
    return out


# ------------------------------------------------------- steal layer ----

def build_steal_tables():
    """Per-runner as-of steal profile from the game logs: day-start
    cumulative SB successes, attempts (SB+CS) and times-on-first
    (1B+BB+HBP — the opportunity proxy). Event-day-only table -> consumers
    must join merge_asof, never exact-date. Trailing-365d league attempt/
    success rates ride along as the EB prior."""
    gb = pd.read_csv(HERE.parent / "Data" / "mlb_game_batting.csv",
                     usecols=["Date", "PlayerId", "H", "2B", "3B", "HR",
                              "BB", "HBP", "SB", "CS"])
    for c in gb.columns.drop("Date"):
        gb[c] = pd.to_numeric(gb[c], errors="coerce").fillna(0)
    gb["Date"] = pd.to_datetime(gb["Date"])
    gb["on1"] = (gb["H"] - gb["2B"] - gb["3B"] - gb["HR"]).clip(lower=0) \
        + gb["BB"] + gb["HBP"]
    gb["att"] = gb["SB"] + gb["CS"]

    day = (gb.groupby(["PlayerId", "Date"])
             .agg(sb=("SB", "sum"), att=("att", "sum"), on1=("on1", "sum"))
             .sort_index())
    g = day.groupby(level=0, sort=False)
    cum = (g.cumsum() - day).reset_index()          # day-start
    cum = cum.sort_values("Date", kind="stable")

    daily = gb.groupby("Date")[["SB", "att", "on1"]].sum()
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()),
                         fill_value=0)
    roll = full.rolling(365, min_periods=1).sum().shift(1)
    lg = pd.DataFrame({
        "lg_att": (roll["att"] / roll["on1"]).clip(0.01, 0.30),
        "lg_succ": (roll["SB"] / roll["att"]).clip(0.5, 0.95)})
    log(f"steal tables: {cum['PlayerId'].nunique():,} runners, "
        f"league att/opp {lg['lg_att'].iloc[-1]:.3f}, "
        f"succ {lg['lg_succ'].iloc[-1]:.3f}")
    return {"players": cum, "league": lg}


# ----------------------------------------------------------- build ------

def build_all(force=False):
    if TABLE_CACHE.exists() and not force:
        return joblib.load(TABLE_CACHE)
    bf_tables = build_starter_bf()
    tables = {"transitions": build_transitions(),
              "starter_bf": bf_tables,
              "pen_rates": build_pen_rates(),
              "steals": build_steal_tables(),
              "starter_hazard": build_starter_hazard(bf_tables),
              "starter_hazard_v2": build_starter_hazard_v2(bf_tables)}
    joblib.dump(tables, TABLE_CACHE, compress=3)
    log(f"cached -> {TABLE_CACHE.name}")
    return tables


def _sanity(tables):
    tr = tables["transitions"]
    ci = {c: i for i, c in enumerate(CLASSES)}

    def p_scored_from_2nd(outs):
        cell = tr["cells"].get((ci["1B"], 2, outs))   # runner on 2nd only
        return float((cell["p"] * (cell["runs"] >= 1)).sum())

    print("\n--- sanity vs known league advancement rates ---")
    for o in (0, 1, 2):
        print(f"  P(runner on 2nd scores on a single | {o} out): "
              f"{p_scored_from_2nd(o):.3f}   (league ~.55-.70, rises w/outs)")
    cell = tr["cells"].get((ci["OUT"], 1, 0))         # runner on 1st, 0 out
    print(f"  P(double play on ball-in-play out | 1st, 0 out): "
          f"{float((cell['p'] * (cell['d_outs'] >= 2)).sum()):.3f} "
          f"(league ~.15-.20 of outs in that state)")
    cell = tr["cells"].get((ci["HR"], 7, 0))          # slam check
    print(f"  E[runs | HR, bases loaded]: "
          f"{float((cell['p'] * cell['runs']).sum()):.2f} (must be 4.00)")
    bf = tables["starter_bf"]["league_by_season"]
    print(f"  league mean starter BF 2024: {bf[2024].mean():.1f} "
          f"(~22-24); 2015: {bf[2015].mean():.1f} (higher, pre-opener era)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    t = build_all(force=args.rebuild)
    _sanity(t)
