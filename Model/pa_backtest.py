"""PA-sim Phase 2, shadow backtest — simulate every 2025/2026 slate from
pregame information only and write per-entity sim probabilities/means to
parquet for grading against the incumbent heads' paired snapshots.

Pregame-legal inputs per game: posted lineup (the 9 starters + slots from
the box score — the same lineup information the incumbent heads serve
from), each side's starting pitcher, park, and every rate strictly as-of
an earlier date (merge_asof on day-start tables; the game's own day never
contributes). The PA model is the Phase-1 artifact (trained <=2023, so
BOTH backtest years are out-of-sample for it).

Feature assembly is batch merge_asof — one pass per table for all
game-batter pairs — then one predict_proba call per (starter/pen) side.

Usage:
    python Model/pa_backtest.py --year 2025 [--sims 1500] [--part i/k]
`--part i/k` simulates every k-th game date (parallel processes each take
a part; outputs suffix the part and concatenate at grading time).
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pa_model import CLASSES, EB_K, FRAME_CACHE, _asof_counts, _cq_shrink, \
    _cq_tables, _eb, _league_trailing
from milb_priors import build_all as milb_build, prior_blend
from pa_sim import (STEAL_K_ATT, STEAL_K_SUCC, TIER_EDGES,  # noqa: F401
                    battery_adjust, battery_context,
                    hazard_slice_v2, starter_exp_bf)
from pa_engine import PackedTransitions, GameSim, CI, TB_OF  # noqa: F401

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DATA = HERE.parent / "Data"

K_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
OUT_LINES = [14.5, 15.5, 16.5, 17.5, 18.5]
PHA_LINES = [3.5, 4.5, 5.5, 6.5]
PBB_LINES = [0.5, 1.5, 2.5]
PER_LINES = [1.5, 2.5, 3.5, 4.5]
TOTAL_LINES = [6.5, 7.5, 8.5, 9.5, 10.5]
XBK_LINES = [0.5, 1.5, 2.5]
XTB_LINES = [1.5, 2.5, 3.5]
XHRR_LINES = [1.5, 2.5, 3.5]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------ slate build -----

def build_slates(year):
    """One row per (game, side, slot): batter, stand vs the opposing
    starter, park, and the opposing starter/pen identifiers."""
    gb = pd.read_csv(DATA / "mlb_game_batting.csv",
                     usecols=["GamePk", "Season", "Date", "PlayerId", "Team",
                              "Opponent", "Home", "BattingOrder"])
    gb = gb[gb["Season"] == year]
    slot = pd.to_numeric(gb["BattingOrder"], errors="coerce")
    gb = gb[slot % 100 == 0].copy()
    gb["slot"] = (slot[slot % 100 == 0] // 100).astype(int) - 1

    gp = pd.read_csv(DATA / "mlb_game_pitching.csv",
                     usecols=["GamePk", "Season", "Date", "PlayerId", "Team",
                              "GS"])
    st = gp[(gp["Season"] == year) & (gp["GS"] == 1)][
        ["GamePk", "Team", "PlayerId"]].rename(
        columns={"PlayerId": "starter"})
    # opposing starter: merge on the OTHER team
    gb = gb.merge(st.rename(columns={"Team": "Opponent"}),
                  on=["GamePk", "Opponent"], how="inner")
    # drop games without exactly 9 slots a side (suspended/short data)
    ok = (gb.groupby(["GamePk", "Team"])["slot"].transform("count") == 9)
    gb = gb[ok]
    both = gb.groupby("GamePk")["Home"].transform("nunique") == 2
    gb = gb[both].copy()
    gb["park"] = np.where(gb["Home"] == 1, gb["Team"], gb["Opponent"])
    gb["Date"] = pd.to_datetime(gb["Date"])
    log(f"slates {year}: {gb['GamePk'].nunique():,} games, "
        f"{len(gb):,} lineup slots")
    return gb.sort_values(["Date", "GamePk", "Home", "slot"])


# ------------------------------------------------- batch features -------

def _asof_merge(slates, tab, keys, left_keys):
    """merge_asof the day-start table onto slate rows (strictly as-of:
    table rows AT the game date are day-start, so <= is leakage-free).
    merge_asof resets the index, so the original one rides along in a
    column and is restored — output aligns row-for-row with `slates`."""
    t = tab.rename(columns=dict(zip(keys, left_keys)))
    s = slates.reset_index()
    for k in left_keys:                    # match by-key dtypes exactly
        if pd.api.types.is_numeric_dtype(s[k]):
            s[k] = s[k].astype("int64")
            t[k] = t[k].astype("int64")
        else:
            s[k] = s[k].astype(str)
            t[k] = t[k].astype(str)
    t = t.sort_values("Date", kind="stable")
    s = s.sort_values("Date", kind="stable")
    m = pd.merge_asof(s, t, on="Date", by=left_keys, direction="backward")
    return m.set_index("index").sort_index()


def _eb_block(cnt_df, n_ser, prior_df, k_mult=1):
    k = {c: k_mult * v for c, v in EB_K.items()}
    return _eb(cnt_df.fillna(0), n_ser.fillna(0), prior_df, k)


def build_features(slates, pa, pen_rates):
    """78 PA-model feature columns for every slate row, twice (vs starter
    and vs team pen). Returns (feat_st, feat_pen) aligned to slates."""
    lg = _league_trailing(pa)
    lgf = lg.reindex(pd.date_range(lg.index.min(), lg.index.max())).ffill()

    # batter stance vs a given pitcher hand (switch hitters resolved by
    # their own history); starter throwing hand = his modal
    hand_tab = (pa.groupby(["batter", "p_throws"])["stand"]
                  .agg(lambda s: s.mode().iat[0]).rename("stand")
                  .reset_index())
    throws = (pa.groupby("pitcher")["p_throws"]
                .agg(lambda s: s.mode().iat[0]).to_dict())

    s = slates.reset_index(drop=True).copy()
    s["p_throws"] = s["starter"].map(throws).fillna("R")
    s = s.merge(hand_tab, left_on=["PlayerId", "p_throws"],
                right_on=["batter", "p_throws"], how="left")
    s["stand"] = s["stand"].fillna("R")
    s = s.drop(columns=["batter"])
    lg_rows = lgf.reindex(s["Date"]).to_numpy()
    lg_prior = pd.DataFrame(lg_rows, columns=CLASSES, index=s.index)

    # career priors: league blended toward translated MiLB (same tables the
    # PA frame trains on; serve rows only use seasons <= Y-1)
    milb = milb_build()

    def _milb_prior(key_col, kind):
        serve = milb[kind]["serve"]
        m = s[[key_col, "Season"]].merge(
            serve, left_on=[key_col, "Season"],
            right_on=["PlayerId", "Season"], how="left")
        t = m[[f"t_{c}" for c in CLASSES]].to_numpy()
        ne = m["n_eff"].fillna(0).to_numpy()
        return (pd.DataFrame(prior_blend(lg_rows, t, ne),
                             columns=CLASSES, index=s.index), np.log1p(ne))

    b_prior, b_milb_n = _milb_prior("PlayerId", "bat")
    p_prior, p_milb_n = _milb_prior("starter", "pit")

    feats = {}
    specs = [("b", ["batter"], ["PlayerId"], None, 1),
             ("p", ["pitcher"], ["starter"], None, 1),
             ("bh", ["batter", "p_throws"], ["PlayerId", "p_throws"],
              "b", 2),
             ("ph", ["pitcher", "stand"], ["starter", "stand"], "p", 2),
             ("bs", ["batter", "Season"], ["PlayerId", "Season"], "b", 1),
             ("ps", ["pitcher", "Season"], ["starter", "Season"], "p", 1)]
    for pre, keys, lkeys, prior_pre, mult in specs:
        tab = _asof_counts(pa, keys)
        need = s[["Date"] + lkeys].copy()
        m = _asof_merge(need, tab, keys, lkeys)
        cnt = m[CLASSES]
        n = m["n"]
        if prior_pre is None:
            prior = b_prior if pre == "b" else p_prior
        else:
            prior = pd.DataFrame(feats[prior_pre][0], columns=CLASSES,
                                 index=s.index)
        rates = _eb_block(cnt, n, prior, mult).to_numpy()
        feats[pre] = (rates, np.log1p(n.fillna(0)).to_numpy())
        log(f"  features: {pre} done")

    # contact quality: as-of cumulative barrel/EV per batter and starter,
    # EB-shrunk to the trailing league values — the same tables the PA
    # frame trains on, joined merge_asof (event-day rows only; see
    # _cq_tables). The pen pseudo-pitcher carries the league prior at
    # n=0: no pitcher identity is exactly what the EB shrink encodes.
    cq_tabs, lg_brl, lg_ev = _cq_tables(pa)
    lb = lg_brl.reindex(s["Date"]).to_numpy()
    le = lg_ev.reindex(s["Date"]).to_numpy()
    mb = _asof_merge(s[["Date", "PlayerId"]].copy(), cq_tabs["b"],
                     ["batter"], ["PlayerId"])
    b_cq = _cq_shrink(mb["brl"], mb["ev"], mb["n"], lb, le)
    mp = _asof_merge(s[["Date", "starter"]].copy(), cq_tabs["p"],
                     ["pitcher"], ["starter"])
    p_cq = _cq_shrink(mp["brl"], mp["ev"], mp["n"], lb, le)
    zeros = np.zeros(len(s))
    pen_cq = _cq_shrink(zeros, zeros, zeros, lb, le)
    log("  features: cq done")

    # pen pseudo-pitcher: as-of team pen rates fill every pitcher slot
    pen = pen_rates.copy()
    pen["Date"] = pd.to_datetime(pen["Date"])
    m = pd.merge_asof(
        s[["Date", "Opponent"]].reset_index().sort_values("Date"),
        pen.sort_values("Date").rename(columns={"Team": "Opponent"}),
        on="Date", by="Opponent", direction="backward").set_index("index")
    m = m.sort_index()
    pen_rates_arr = m[[f"pen_{c}" for c in CLASSES]].to_numpy()
    pen_rates_arr = np.where(np.isnan(pen_rates_arr), 1 / 8, pen_rates_arr)
    pen_n = np.log1p(m["pen_n"].fillna(0).to_numpy())

    def assemble(vs_pen):
        df = {}
        for pre in ("b", "bh", "bs"):
            r, n = feats[pre]
            for i, c in enumerate(CLASSES):
                df[f"{pre}_{c}"] = r[:, i]
            df[f"{pre}_n"] = n
        for pre in ("p", "ph", "ps"):
            r, n = feats[pre]
            for i, c in enumerate(CLASSES):
                df[f"{pre}_{c}"] = pen_rates_arr[:, i] if vs_pen else r[:, i]
            df[f"{pre}_n"] = pen_n if vs_pen else n
        for i, c in enumerate(CLASSES):
            df[f"lg_{c}"] = lg_rows[:, i]
        df["b_brl"], df["b_ev"], df["b_bip"] = b_cq
        df["p_brl"], df["p_ev"], df["p_bip"] = pen_cq if vs_pen else p_cq
        df["b_milb_n"] = b_milb_n
        df["p_milb_n"] = np.zeros(len(s)) if vs_pen else p_milb_n
        out = pd.DataFrame(df, index=s.index)
        p_throws = pd.Series("R", index=s.index) if vs_pen else s["p_throws"]
        out["same_hand"] = (s["stand"] == p_throws).astype(int)
        out["bat_home"] = s["Home"].to_numpy()
        out["outs_when_up"] = 1
        out["inning"] = 7 if vs_pen else 3
        out["on1b"] = out["on2b"] = out["on3b"] = 0
        park_levels = sorted(pa["park"].astype(str).unique())
        out["park"] = pd.Categorical(s["park"].astype(str),
                                     categories=park_levels)
        out["stand"] = pd.Categorical(s["stand"], categories=["L", "R", "S"])
        out["p_throws"] = pd.Categorical(p_throws, categories=["L", "R", "S"])
        return out

    return s, assemble(False), assemble(True)


# ------------------------------------------------------- simulate -------

def steal_params(s, steal_tables):
    """(att, succ) EB steal rates per slate row: the runner's day-start
    cumulative record shrunk to the trailing league rates (merge_asof —
    the player table only has rows on days he played)."""
    m = _asof_merge(s[["Date", "PlayerId"]].copy(), steal_tables["players"],
                    ["PlayerId"], ["PlayerId"])
    lg = steal_tables["league"].reindex(s["Date"])
    lg_att = np.nan_to_num(lg["lg_att"].to_numpy(), nan=0.06)
    lg_succ = np.nan_to_num(lg["lg_succ"].to_numpy(), nan=0.78)
    att_c, sb_c = m["att"].fillna(0), m["sb"].fillna(0)
    on1 = m["on1"].fillna(0)
    att = ((att_c + STEAL_K_ATT * lg_att) / (on1 + STEAL_K_ATT)).to_numpy()
    succ = ((sb_c + STEAL_K_SUCC * lg_succ)
            / (att_c + STEAL_K_SUCC)).to_numpy()
    return att, succ


def starter_bf_draws(bf_tables, starter, date, season, k_last=30):
    dates, bfs = bf_tables["by_pitcher"].get(int(starter), (None, None))
    if dates is not None:
        i = np.searchsorted(dates, np.datetime64(date))
        hist = bfs[max(0, i - k_last):i]
        if len(hist) >= 5:
            return hist
    lg = bf_tables["league_by_season"].get(int(season))
    if lg is None or len(lg) == 0:
        lg = np.array([22] * 10, np.int16)
    return lg


def starter_tier(bf_tables, starter, date, k_last=30):
    """Workload tier for the hazard table: trailing k_last-start mean BF
    as-of `date` (same as-of window the table was fit with)."""
    dates, bfs = bf_tables["by_pitcher"].get(int(starter), (None, None))
    mean_bf = 23.0
    if dates is not None:
        i = np.searchsorted(dates, np.datetime64(date))
        hist = bfs[max(0, i - k_last):i]
        if len(hist) >= 5:
            mean_bf = float(hist.mean())
    return int(np.searchsorted(TIER_EDGES, mean_bf))


def run_backtest(year, n_sims, part_i=0, part_k=1):
    from pa_model import feature_cols
    art = joblib.load(ART / "pa_model.joblib")
    model, cols = art["model"], art["cols"]
    assert cols == feature_cols(), "pa_model artifact/feature drift"
    tables = joblib.load(ART / "pa_sim_tables.joblib")
    packed = PackedTransitions(tables["transitions"])
    pa = pd.read_parquet(FRAME_CACHE)

    slates = build_slates(year)
    s, f_st, f_pen = build_features(slates, pa, tables["pen_rates"])
    log("predicting matchup probabilities ...")
    p_st = model.predict_proba(f_st[cols])
    p_pen = model.predict_proba(f_pen[cols])
    steal_att, steal_succ = steal_params(s, tables["steals"])

    dates = sorted(s["Date"].unique())
    take = set(dates[part_i::part_k])
    games = [g for g, d in s.groupby("GamePk")["Date"].first().items()
             if d in take]
    log(f"simulating {len(games):,} games (part {part_i + 1}/{part_k}, "
        f"{n_sims} sims each) ...")

    bat_rows, st_rows, game_rows = [], [], []
    s_idx = {g: gg for g, gg in s.groupby("GamePk")}
    for gi, gpk in enumerate(games):
        gg = s_idx[gpk]
        date = gg["Date"].iat[0]
        season = int(gg["Season"].iat[0])
        sides = {}
        steal = {}
        meta = {}
        okay = True
        for side, home_flag in (("away", 0), ("home", 1)):
            rows = gg[gg["Home"] == home_flag].sort_values("slot")
            if len(rows) != 9:
                okay = False
                break
            idx = rows.index
            sides[side] = {"st": p_st[idx], "pen": p_pen[idx]}
            # battery modulation (#35): rows["starter"]/["Opponent"] are
            # this batting side's OPPOSING starter and fielding team
            r_att, stp = battery_context(tables.get("battery"), season,
                                         rows["Opponent"].iat[0],
                                         rows["starter"].iat[0])
            steal[side] = battery_adjust(steal_att[idx], steal_succ[idx],
                                         r_att, stp)
            meta[side] = rows
        if not okay:
            continue
        # meta[side]["starter"] = that batting side's OPPOSING starter, so
        # keying by the opposite side gives each FIELDING team's starter —
        # exactly what GameSim's bf dict expects
        bf = {"away": starter_bf_draws(tables["starter_bf"],
                                       meta["home"]["starter"].iat[0],
                                       date, season),
              "home": starter_bf_draws(tables["starter_bf"],
                                       meta["away"]["starter"].iat[0],
                                       date, season)}
        # hazard v2 (2026-07-14): the relative-BF table sliced back to an
        # absolute-BF table per starter (his own leash); v1 tier slice
        # stays available under tables["starter_hazard"] for rollback
        hz2 = tables["starter_hazard_v2"]["hazard"]
        bf_t = tables["starter_bf"]
        hazard = {"away": hazard_slice_v2(hz2, starter_exp_bf(
                      bf_t, meta["home"]["starter"].iat[0], date)),
                  "home": hazard_slice_v2(hz2, starter_exp_bf(
                      bf_t, meta["away"]["starter"].iat[0], date))}
        sim = GameSim(packed, sides, bf, steal=steal, hazard=hazard,
                      n_sims=n_sims, seed=int(gpk) & 0x7fffffff)
        out = sim.run()

        for side in ("away", "home"):
            rows = meta[side]
            stt = out["stats"]
            h = (stt["h1"][side] + stt["h2"][side] + stt["h3"][side]
                 + stt["hr"][side])
            hrr = h + stt["r"][side] + stt["rbi"][side]
            for j in range(9):
                bat_rows.append({
                    "GamePk": gpk, "Date": date,
                    "PlayerId": int(rows["PlayerId"].iat[j]),
                    "p_hit": (h[j] >= 1).mean(),
                    "p_hits2": (h[j] >= 2).mean(),
                    "p_hr": (stt["hr"][side][j] >= 1).mean(),
                    "p_tb2": (stt["tb"][side][j] >= 2).mean(),
                    "p_run": (stt["r"][side][j] >= 1).mean(),
                    "p_rbi": (stt["rbi"][side][j] >= 1).mean(),
                    "p_bb": (stt["bb"][side][j] >= 1).mean(),
                    "p_sb": (stt["sb"][side][j] >= 1).mean(),
                    "p_bk": (stt["k"][side][j] >= 1).mean(),
                    "p_bk2": (stt["k"][side][j] >= 2).mean(),
                    "p_single": (stt["h1"][side][j] >= 1).mean(),
                    "p_double": (stt["h2"][side][j] >= 1).mean(),
                    "p_hrr2": (hrr[j] >= 2).mean(),
                    "p_hrr3": (hrr[j] >= 3).mean(),
                    "xbk": stt["k"][side][j].mean(),
                    "xtb": stt["tb"][side][j].mean(),
                    "xhrr": hrr[j].mean(),
                    **{f"p_xbk_{l}": (stt["k"][side][j] > l).mean()
                       for l in XBK_LINES},
                    **{f"p_xtb_{l}": (stt["tb"][side][j] > l).mean()
                       for l in XTB_LINES},
                    **{f"p_xhrr_{l}": (hrr[j] > l).mean()
                       for l in XHRR_LINES},
                })
            fld = "home" if side == "away" else "away"
            stp = out["starter"]
            st_rows.append({
                "GamePk": gpk, "Date": date,
                "PlayerId": int(meta[fld]["starter"].iat[0]),
                "xk": stp["k"][side].mean(),
                "xouts": stp["outs"][side].mean(),
                "xpha": stp["h"][side].mean(),
                "xpbb": stp["bb"][side].mean(),
                "xper": stp["er"][side].mean(),
                **{f"p_k_{l}": (stp["k"][side] > l).mean()
                   for l in K_LINES},
                **{f"p_outs_{l}": (stp["outs"][side] > l).mean()
                   for l in OUT_LINES},
                **{f"p_pha_{l}": (stp["h"][side] > l).mean()
                   for l in PHA_LINES},
                **{f"p_pbb_{l}": (stp["bb"][side] > l).mean()
                   for l in PBB_LINES},
                **{f"p_per_{l}": (stp["er"][side] > l).mean()
                   for l in PER_LINES},
            })
        tot = out["score"]["home"] + out["score"]["away"]
        game_rows.append({
            "GamePk": gpk, "Date": date,
            "p_home_win": out["home_win"].mean(),
            "x_total": tot.mean(),
            "x_away": out["score"]["away"].mean(),
            "x_home": out["score"]["home"].mean(),
            **{f"p_total_{l}": (tot > l).mean() for l in TOTAL_LINES},
        })
        if (gi + 1) % 200 == 0:
            log(f"  {gi + 1}/{len(games)} games")

    suff = f"_{part_i + 1}of{part_k}" if part_k > 1 else ""
    for name, rows in (("bat", bat_rows), ("starter", st_rows),
                       ("game", game_rows)):
        df = pd.DataFrame(rows)
        p = ART / f"sim_backtest_{name}_{year}{suff}.parquet"
        df.to_parquet(p, index=False)
        log(f"wrote {p.name}: {len(df):,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--sims", type=int, default=1500)
    ap.add_argument("--part", default="1/1",
                    help="i/k — simulate every k-th game date, offset i")
    args = ap.parse_args()
    i, k = (int(x) for x in args.part.split("/"))
    run_backtest(args.year, args.sims, part_i=i - 1, part_k=k)
