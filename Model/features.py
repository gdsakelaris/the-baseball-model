"""Feature engineering for the MLB prediction models.

Builds leakage-free features: every feature for a game on date D uses only
data from strictly before D. Two paths share the same definitions:

  - build_batter_frame / build_starts_frame / build_game_frame:
      vectorized, over all of 2020-2026, for training (cumsum/rolling
      shifted by one game so the current game never sees itself).
  - Stores.asof_*:
      per-entity, for an arbitrary future/hypothetical game, for inference.

Data used (all files in Data/):
  game logs (3 files)      -> form, rest, bullpen (incl. high-leverage arms
                              and trailing-3-day fatigue), team offense
                              (overall + home/away), park factor, home/away
                              batter splits, position (C/DH) shares,
                              strike rate, defense proxy (unearned runs),
                              battery SB-allowed (team steals surrendered)
  season stats (2 files)   -> prior-season GO/AO (groundball/flyball) and
                              pitcher SB/CS/PK stolen-base control
  pitch arsenals (2 files) -> batter-vs-arsenal matchup scores and pitcher
                              arsenal quality (two-year decay blend)
  homeruns                 -> batter HR power quality (exit velo, launch
                              angle, elevation-adjusted distance), pitcher
                              HR quality allowed, batter HR-by-pitch-type
                              profile vs the starter's usage
  rosters                  -> handedness, platoon, age, height, weight
  ballparks                -> dimensions, elevation
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

ROLL_WINDOWS = (7, 15, 30)
CAT_COLS = ["DayNight", "Condition", "WindDir", "bat_hand", "pit_throws"]

# Decay-weighted "current form": every prior game weighted exp(-days_ago *
# DECAY_LAM), half-life DECAY_HL_DAYS. Sits between the rolling windows
# (hard cliff at N games) and season-to-date (all games equal): skill drift
# shows up gradually and old games fade instead of falling off an edge. The
# decayed PA total is a real effective sample size in PA units, so the same
# SHRINK constants apply to the decayed rates.
DECAY_HL_DAYS = 90.0
DECAY_LAM = np.log(2.0) / DECAY_HL_DAYS
DECAY_EPOCH = pd.Timestamp("2020-01-01")
DECAY_STATS = ["PA", "AB", "H", "HR", "TB", "SO", "BB", "SB", "R", "RBI",
               "HBP", "HRR2", "HRR3"]
DECAY_SHRINK = ("hr_pa", "tb_ab", "k_pct", "bb_pct", "sb_pa",
                "r_pa", "rbi_pa")

# Expected exposure (xPA): every binary prop is really P(gets N plate
# appearances) x P(event per PA), but the trees only see per-PA rates and
# slot — they cannot form a PA-per-game ratio from cumulative sums. Two
# explicit exposure features: xpa_bat (the batter's own decayed PA per
# game — pinch-hit / platoon-removal / early-pull risk) and xpa_slot
# (as-of league PA per game at tonight's lineup slot — leadoff 4.48 vs
# nine hole 3.31).
# BENCHED (2026-07-07 selection run): 1 better (rbi_top10, barely past
# band) vs 3 worse (rbi/single/bk ECE past band), every AUC/edge within
# noise — calibration harm without ranking gain; slot + per-PA rates
# evidently already carry the usable exposure signal. Both stay in the
# frames and the Stores/predict wiring, but out of batter_feature_cols.
XPA_PRIOR = 4.3          # league PA per batter-game (shrinkage target)
XPA_K = 10.0             # games of shrinkage for xpa_bat

# ---------------------------------------------------------------- loading


class MeanBag:
    """Average-prediction ensemble over seed-bagged LightGBM models: quacks
    like a single model (predict / predict_proba / best_iteration_) so every
    consumer — predict_prop, count heads, selftest, GUI — works unchanged.
    Lives here so pickled artifacts resolve it from any entry point."""

    def __init__(self, models):
        self.models = models
        self.best_iteration_ = models[0].best_iteration_

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def inf_to_nan(X):
    """sklearn rejects inf (from divide-by-zero rates); make them missing so
    the imputer handles them. Lives here (not in train's __main__) so a
    pickled pipeline referencing it loads from predict.py/evaluate_deep.py too."""
    X = np.asarray(X, dtype="float64").copy()
    X[~np.isfinite(X)] = np.nan
    return X


def _read(name, **kw):
    return pd.read_csv(DATA_DIR / name, encoding="utf-8-sig", **kw)


def ip_to_outs(ip):
    """'5.2' -> 17 outs. MLB notation: .1/.2 are thirds of an inning."""
    ip = pd.to_numeric(ip, errors="coerce")
    whole = np.floor(ip)
    return (whole * 3 + np.round((ip - whole) * 10)).astype("float64")


def height_to_inches(ht):
    m = re.match(r"(\d+)'\s*(\d+)", str(ht))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def load_raw():
    games = _read("mlb_games.csv", parse_dates=["Date"])
    gb = _read("mlb_game_batting.csv", parse_dates=["Date"])
    # per-game H+R+RBI threshold indicators: own-target history for the hrr
    # props (share of past games clearing the line — the joint clustering
    # that the marginal H/R/RBI rates can't express). Lives on gb so the
    # vectorized and inference paths inherit identical values.
    _hrr = gb["H"] + gb["R"] + gb["RBI"]
    gb["HRR2"] = (_hrr >= 2).astype("float64")
    gb["HRR3"] = (_hrr >= 3).astype("float64")
    gp = _read("mlb_game_pitching.csv", parse_dates=["Date"])
    gp["Outs"] = ip_to_outs(gp["IP"])
    gp["NP"] = pd.to_numeric(gp["NP"], errors="coerce")
    gp["Strikes"] = pd.to_numeric(gp["Strikes"], errors="coerce")

    rosters = _read("mlb_rosters.csv")
    rosters["height_in"] = rosters["Ht"].map(height_to_inches)
    rosters["DOB"] = pd.to_datetime(rosters["DOB"], format="%m/%d/%Y", errors="coerce")

    parks = _read("mlb_ballparks.csv")

    hr = _read("mlb_homeruns.csv", parse_dates=["Date"])
    for c in ["Angle", "Exit Velo", "Distance"]:
        hr[c] = pd.to_numeric(hr[c], errors="coerce")
    hr["BatterId"] = pd.to_numeric(hr["BatterId"], errors="coerce")
    # normalize the HR log's pitch names to the arsenal file's vocabulary so
    # the HR-by-pitch-type matchup join actually connects (e.g. the HR log
    # says "Splitter", the arsenal says "Split-Finger"). Names with no
    # arsenal bucket (Eephus) stay as-is and simply never match usage.
    hr["Pitch"] = hr["Pitch"].replace({
        "Splitter": "Split-Finger", "Forkball": "Split-Finger",
        "2-Seam Fastball": "Sinker",
        "Four-Seam Fastball": "4-Seam Fastball", "Fastball": "4-Seam Fastball",
        "Knuckle Curve": "Curveball", "Slow Curve": "Curveball",
        "Knuckle Ball": "Knuckleball",
    })

    def _ars(name):
        a = _read(name)
        for c in ["RV/100", "%", "xSLG", "xwOBA", "Whiff %", "Hard Hit %",
                  "K%", "Put Away %", "Pitches"]:
            a[c] = pd.to_numeric(a[c], errors="coerce")
        return a

    ars_p = _ars("mlb_pitch_arsenals.csv")
    ars_b = _ars("mlb_pitch_arsenals_batters.csv")

    # season stat lines: the only sources of GO/AO (groundball/flyball
    # tendency) and pitcher SB/CS/PK (stolen-base control) — neither is
    # reconstructible from the game logs. Used as PRIOR-season values.
    bat_season = _read("mlb_batting_stats.csv")
    for c in ["GO/AO", "PA"]:
        bat_season[c] = pd.to_numeric(bat_season[c], errors="coerce")
    pit_season = _read("mlb_pitching_stats.csv")
    for c in ["GO/AO", "SB", "CS", "PK", "TBF"]:
        pit_season[c] = pd.to_numeric(pit_season[c], errors="coerce")
    pit_season["Outs"] = ip_to_outs(pit_season["IP"])

    hands = _read("mlb_handedness.csv")
    hands["PlayerId"] = pd.to_numeric(hands["PlayerId"], errors="coerce")
    hands = hands.dropna(subset=["PlayerId"])
    hands["PlayerId"] = hands["PlayerId"].astype("int64")

    # Statcast batted balls (every tracked ball in play): contact-quality
    # "process" stats that stabilize far faster than outcomes. Optional —
    # frames build without the file (features stay NaN) so a checkout
    # without the backfill still runs. Scripts/scrape_statcast.py creates it.
    bip_path = DATA_DIR / "mlb_statcast_bip.csv"
    bip = None
    if bip_path.exists():
        bip = _read(bip_path.name, parse_dates=["Date"])
        for c in ["HcX", "HcY"]:        # spray coords: absent in old scrapes
            if c not in bip.columns:
                bip[c] = np.nan
        for c in ["BatterId", "PitcherId", "ExitVelo", "LaunchAngle", "LSA",
                  "xBA", "xwOBA", "GamePk", "HcX", "HcY"]:
            bip[c] = pd.to_numeric(bip[c], errors="coerce")

    # pitch-level daily aggregates (whiffs/chases/velo; scrape_pitches.py),
    # sprint speed and team OAA leaderboards — all optional like bip
    def _opt(name, **kw):
        p = DATA_DIR / name
        return _read(name, **kw) if p.exists() else None

    pdp = _opt("mlb_pitch_daily_pitchers.csv", parse_dates=["Date"])
    pdb = _opt("mlb_pitch_daily_batters.csv", parse_dates=["Date"])
    for d in (pdp, pdb):
        if d is not None:
            d["csw_n"] = d["wh_n"] + d["cs_n"]      # called strikes + whiffs
    sprint = _opt("mlb_sprint_speed.csv")
    oaa = _opt("mlb_oaa.csv")
    umps = _opt("mlb_umpires.csv", parse_dates=["Date"])
    bat_track = _opt("mlb_bat_tracking.csv")   # bat speed / swing (2023+ only)

    # 7-inning doubleheaders (2020-21) bias per-game rates; flag to exclude.
    outs_per_game = gp.groupby("GamePk")["Outs"].sum()
    short = set(outs_per_game[outs_per_game < 45].index) & set(
        games.loc[games["Season"] <= 2021, "GamePk"]
    )
    games["ShortGame"] = games["GamePk"].isin(short)

    for df in (games,):
        df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
        df["WindSpeed"] = pd.to_numeric(df["WindSpeed"], errors="coerce")
        df["WindDir"] = df["WindDir"].fillna("").str.strip().str.title()
        df["Condition"] = df["Condition"].fillna("").str.strip().str.title()
    raw = dict(games=games, gb=gb, gp=gp, rosters=rosters, parks=parks,
               hr=hr, ars_p=ars_p, ars_b=ars_b, hands=hands,
               bat_season=bat_season, pit_season=pit_season, bip=bip,
               pdp=pdp, pdb=pdb, sprint=sprint, oaa=oaa, umps=umps,
               bat_track=bat_track)
    raw["gb"] = annotate_opp_hand(gb, gp, hands)
    return raw


def annotate_opp_hand(gb, gp, hands):
    """Add opp_hand to each batter-game row: the throwing hand of the
    OPPOSING STARTER that game (who faces most of a batter's PAs). This is
    what the platoon-split features condition on. Shared by the training
    and inference paths so both see identical history."""
    throws = hands.set_index("PlayerId")["Throws"]
    st = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]].copy()
    st["opp_hand"] = st["PlayerId"].map(throws).replace("", np.nan)
    st = st.rename(columns={"Team": "Opponent"})[["GamePk", "Opponent", "opp_hand"]]
    return gb.merge(st, on=["GamePk", "Opponent"], how="left")


# ------------------------------------------------- vectorized as-of tables

BAT_STATS = ["PA", "AB", "H", "HR", "TB", "SO", "BB",
             "SB", "CS", "R", "RBI", "2B", "3B", "HBP", "IBB",
             "HRR2", "HRR3", "GIDP", "SF"]
PIT_STATS = ["BF", "HR", "SO", "BB", "Outs", "ER", "H", "Strikes", "NP"]
VSH_STATS = ["PA", "HR", "TB", "SO"]  # platoon splits track these
LOC_STATS = ["PA", "H", "HR", "TB", "SO"]  # home/away venue splits
ROLL_STATS = ["PA", "HR", "H", "TB", "SO", "SB", "R", "RBI"]  # rolling sums


def _snap_to_day_start(df, keys, cols):
    """Doubleheader parity: row-order as-of features let game 2 of a same-day
    doubleheader see game 1, but the inference path only knows data from
    strictly before the DATE. Overwrite every same-(entity, day) row with the
    day's first row, whose history is strictly pre-date by construction."""
    return df.groupby([*keys, "Date"], sort=False)[cols].transform("first")


def _batter_asof(gb):
    """Per batter-game row: career/season/rolling form, all as-of (strictly
    pre-DATE, matching Stores.batter_feats)."""
    df = gb.sort_values(["PlayerId", "Date", "GamePk"]).reset_index(drop=True)
    g = df.groupby("PlayerId", sort=False)
    df["g_career"] = g.cumcount()
    for s in BAT_STATS:
        df[f"c_{s}"] = g[s].cumsum() - df[s]
    gs = df.groupby(["PlayerId", "Season"], sort=False)
    df["g_season"] = gs.cumcount()
    for s in BAT_STATS:
        df[f"s_{s}"] = gs[s].cumsum() - df[s]
    for w in ROLL_WINDOWS:
        for s in ROLL_STATS:
            df[f"r{w}_{s}"] = g[s].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).sum())
    df["days_rest"] = g["Date"].diff().dt.days
    # decay-weighted as-of sums: sum_j x_j * exp(-lam*(t_i - t_j)) over prior
    # games j, computed as an exp(lam*t) cumsum discounted back to the row's
    # own date (t spans ~2,400 days -> e^18.5, comfortably inside float64)
    t = (df["Date"] - DECAY_EPOCH).dt.days.to_numpy(dtype="float64")
    wup, wdn = np.exp(DECAY_LAM * t), np.exp(-DECAY_LAM * t)
    for s in DECAY_STATS:
        xw = df[s].to_numpy(dtype="float64") * wup
        cs = pd.Series(xw, index=df.index).groupby(df["PlayerId"]).cumsum() - xw
        df[f"_dk_{s}"] = cs * wdn
    # decayed count of games played (weight 1 per game): the denominator of
    # xpa_bat, the batter's decayed PA per game (exposure)
    cs = pd.Series(wup, index=df.index).groupby(df["PlayerId"]).cumsum() - wup
    df["_dk_G"] = cs * wdn
    # platoon splits: career as-of sums in games vs L / vs R opposing starters
    for hand in ("L", "R"):
        mask = df["opp_hand"] == hand
        for s in VSH_STATS:
            tmp = df[s].where(mask, 0)
            df[f"_vs{hand}_{s}"] = tmp.groupby(df["PlayerId"]).cumsum() - tmp
    # home/away venue splits: career as-of sums in home vs road games
    for flag in (0, 1):
        mask = df["Home"] == flag
        for s in LOC_STATS:
            tmp = df[s].where(mask, 0)
            df[f"_loc{flag}_{s}"] = tmp.groupby(df["PlayerId"]).cumsum() - tmp
    # position wear: career as-of counts of games caught / DH'd
    for pos in ("C", "DH"):
        ind = (df["Position"] == pos).astype(float)
        df[f"_pos{pos}_n"] = ind.groupby(df["PlayerId"]).cumsum() - ind
    asof_cols = (["g_career", "g_season", "days_rest"]
                 + [f"c_{s}" for s in BAT_STATS] + [f"s_{s}" for s in BAT_STATS]
                 + [f"r{w}_{s}" for w in ROLL_WINDOWS for s in ROLL_STATS]
                 + [f"_dk_{s}" for s in DECAY_STATS] + ["_dk_G"]
                 + [f"_vs{h}_{s}" for h in ("L", "R") for s in VSH_STATS]
                 + [f"_loc{f}_{s}" for f in (0, 1) for s in LOC_STATS]
                 + ["_posC_n", "_posDH_n"])
    df[asof_cols] = _snap_to_day_start(df, ["PlayerId"], asof_cols)
    return df


def _starter_asof(gp):
    """Per start row: career/season/rolling starter form, as-of."""
    st = gp[gp["GS"] == 1].sort_values(["PlayerId", "Date", "GamePk"]).reset_index(drop=True)
    g = st.groupby("PlayerId", sort=False)
    st["p_starts_career"] = g.cumcount()
    for s in PIT_STATS:
        st[f"pc_{s}"] = g[s].cumsum() - st[s]
    gs = st.groupby(["PlayerId", "Season"], sort=False)
    st["p_starts_season"] = gs.cumcount()
    for s in PIT_STATS:
        st[f"ps_{s}"] = gs[s].cumsum() - st[s]
    for s in ["BF", "HR", "SO", "BB", "H"]:
        st[f"p5_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    st["p_days_rest"] = g["Date"].diff().dt.days
    # fatigue: pitch counts in the last start and last three starts
    st["p_np_last"] = g["NP"].shift(1)
    st["p_np_l3"] = g["NP"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # a pitcher can't start twice in a day, but snap anyway so any data quirk
    # (suspended/resumed games) can't break train/serve parity
    asof_cols = (["p_starts_career", "p_starts_season", "p_days_rest",
                  "p_np_last", "p_np_l3"]
                 + [f"pc_{s}" for s in PIT_STATS] + [f"ps_{s}" for s in PIT_STATS]
                 + [f"p5_{s}" for s in ["BF", "HR", "SO", "BB", "H"]])
    st[asof_cols] = _snap_to_day_start(st, ["PlayerId"], asof_cols)
    return st


def _daily_cum(df, keys, stats, date_col="Date"):
    """Per-key daily totals -> cumulative as-of table for merge_asof.

    Returned rows are keyed (keys..., Date) with cum_<stat> = totals through
    that date INCLUSIVE; consumers merge with allow_exact_matches=False so a
    game on date D sees totals through D-1.
    """
    day = df.groupby([*keys, date_col], as_index=False)[stats].sum()
    day = day.sort_values([*keys, date_col])
    g = day.groupby(list(keys), sort=False)
    for s in stats:
        day[f"cum_{s}"] = g[s].cumsum()
    day["cum_n"] = g.cumcount() + 1
    return day.drop(columns=stats)


def _team_offense_table(gb):
    return _daily_cum(gb, ["Team", "Season"], ["PA", "HR", "R", "SO", "BB"])


def _team_offense_loc_table(gb):
    """Team offense split by venue context (home vs road): park-driven teams
    (Coors above all) hit very differently at home, and the one cumulative
    table can't express that."""
    return _daily_cum(gb, ["Team", "Season", "Home"], ["PA", "HR", "R"])


def _bullpen_table(gp):
    pen = gp[gp["GS"] == 0]
    return _daily_cum(pen, ["Team", "Season"],
                      ["BF", "HR", "SO", "BB", "ER", "Outs", "H"])


def _bullpen_hl_table(gp):
    """High-leverage bullpen quality: only relief appearances that finished
    the game or earned a save/hold (the arms that actually pitch the 7th-9th
    of close games). The flat cumulative table weights mop-up innings
    equally; these are the innings a batter prop actually meets late."""
    pen = gp[(gp["GS"] == 0)
             & ((gp["SV"] == 1) | (gp["HLD"] == 1) | (gp["GF"] == 1))]
    return _daily_cum(pen, ["Team", "Season"], ["BF", "SO", "ER", "Outs"])


PEN_FATIGUE_DAYS = 3


def _pen_fatigue_table(gp):
    """Team bullpen pitches thrown over the trailing PEN_FATIGUE_DAYS days,
    strictly before each date (one row per team per calendar day). A worn
    bullpen is a late-innings run/hit signal the season-cumulative rates
    can't see. The index runs a week past the last data date so tomorrow's
    slate still resolves at predict time."""
    pen = gp[gp["GS"] == 0]
    day = pen.pivot_table(index="Date", columns="Team", values="NP",
                          aggfunc="sum")
    idx = pd.date_range(day.index.min(), day.index.max() + pd.Timedelta(days=7),
                        freq="D")
    day = day.reindex(idx).fillna(0.0)
    roll = day.rolling(f"{PEN_FATIGUE_DAYS}D").sum().shift(1)
    out = roll.stack().rename("pen_np_l3").reset_index()
    out.columns = ["Date", "Team", "pen_np_l3"]
    return out


def _team_defense_table(gp):
    """Unearned-run rate allowed, per team-season (as-of): (R - ER) * 27 /
    outs across the whole staff. The only defense-quality signal in the
    data; everything else assumes all gloves are equal."""
    return _daily_cum(gp, ["Team", "Season"], ["R", "ER", "Outs"])


def _team_sb_allowed_table(gb):
    """Stolen bases allowed per team, season-cumulative as-of: every SB in
    the batter game logs debits the OPPONENT's battery (pitcher + catcher
    jointly — the logs can't split them). Complements the starter-only
    prior-season psb_* with a current-season, catcher-inclusive rate."""
    d = gb.groupby(["Opponent", "Season", "Date"], as_index=False).agg(
        SB=("SB", "sum"), CS=("CS", "sum"), G=("GamePk", "nunique"))
    d = d.rename(columns={"Opponent": "Team"})
    return _daily_cum(d, ["Team", "Season"], ["SB", "CS", "G"])


def _park_table(gb, games):
    # per-game venue totals for the park factors: HR (legacy) plus runs, hits,
    # doubles and total bases, so the offensive props get the run-environment
    # signal a lone HR factor misses.
    stats = ["HR", "R", "H", "2B", "TB"]
    per_game = gb.groupby("GamePk", as_index=False)[stats].sum()
    gv = games.merge(per_game, on="GamePk", how="left")
    gv[stats] = gv[stats].fillna(0)
    return _daily_cum(gv, ["Venue"], stats)


# rbi opportunity, DEEPER-ORDER ISOLATION: mean career OBP of the hitters 3-5
# spots AHEAD in the order (wrapping). ctx_ahead_obp already covers the 1st-2nd
# ahead, so this deliberately isolates ONLY the deeper table those two miss —
# the honest test of whether the order beyond the immediate two men on carries
# RBI signal of its own. (The earlier full-order decayed version was dominated
# by the -1/-2 terms = a ctx_ahead_obp duplicate, and came back flat.) Equal
# weights => a plain NaN-skipping mean. Shared by the vectorized frame and the
# serving path so the two agree (selftest parity).
RBI_OPP_AHEAD = ((-3, 1.0), (-4, 1.0), (-5, 1.0))


def rbi_opp_from_slots(obp_by_slot, slot):
    """Weighted, NaN-skipping mean of the career OBP of the hitters ahead of
    `slot` (obp_by_slot maps slot -> OBP), renormalized over the hitters that
    have a value. Serving path; the frame computes the same value vectorized."""
    num = den = 0.0
    for off, w in RBI_OPP_AHEAD:
        v = obp_by_slot.get(((slot + off - 1) % 9) + 1)
        if v is not None and not pd.isna(v):
            num += w * v
            den += w
    return num / den if den > 0 else np.nan


ENV_COLS = ["lg_hr_pa", "lg_r_pa", "lg_k_pa", "lg_bb_pa", "lg_sb_pa"]


def _league_env_table(gb):
    """League-wide trailing-30-day offensive environment, one row per
    calendar day. The row for date D covers the 30 days ENDING D-1
    (strictly pre-game), so models can track the CURRENT run environment
    (hot summer, a lively ball) instead of assuming the training years'
    average — that lag showed up as June/July under-prediction in the
    2026 drift eval. NaN when the window is too thin (offseason, opening
    days) so the trees fall back to their other features."""
    stats = ["PA", "HR", "R", "SO", "BB", "SB"]
    day = gb.groupby("Date")[stats].sum().asfreq("D", fill_value=0)
    roll = day.rolling("30D").sum().shift(1)
    out = pd.DataFrame(index=roll.index)
    out["lg_hr_pa"] = roll["HR"] / roll["PA"]
    out["lg_r_pa"] = roll["R"] / roll["PA"]
    out["lg_k_pa"] = roll["SO"] / roll["PA"]
    out["lg_bb_pa"] = roll["BB"] / roll["PA"]
    out["lg_sb_pa"] = roll["SB"] / roll["PA"]
    out[roll["PA"] < 2000] = np.nan
    return out.reset_index()


def _team_results_rows(games):
    """One row per team per game: runs for/against and win flag, sorted for
    as-of lookups. Basis for both season records and recent form."""
    rows = []
    for team_col, score, opp_score in (("AwayTeam", "AwayScore", "HomeScore"),
                                       ("HomeTeam", "HomeScore", "AwayScore")):
        rows.append(pd.DataFrame({
            "GamePk": games["GamePk"], "Team": games[team_col],
            "Season": games["Season"], "Date": games["Date"],
            "RF": pd.to_numeric(games[score], errors="coerce"),
            "RA": pd.to_numeric(games[opp_score], errors="coerce")}))
    res = pd.concat(rows, ignore_index=True).dropna(subset=["RF", "RA"])
    res["W"] = (res["RF"] > res["RA"]).astype(int)
    return res.sort_values(["Team", "Season", "Date", "GamePk"]).reset_index(drop=True)


def _team_results_table(games):
    """Per team per day: cumulative season wins, runs scored and allowed
    (as-of via merge_asof / _cum). Feeds the winner model: win%, run
    differential, and pythagorean expectation are the strongest plain
    team-strength signals and none of them existed in the game frame."""
    return _daily_cum(_team_results_rows(games).drop(columns=["GamePk"]),
                      ["Team", "Season"], ["W", "RF", "RA"])


# Elo: cross-season team strength. K and home-field are standard MLB
# values (538-style); carryover regresses 30% toward the mean each winter.
ELO_BASE, ELO_K, ELO_HFA, ELO_CARRY = 1500.0, 4.0, 24.0, 0.7


def elo_expected(elo_home, elo_away):
    """Expected home-win probability from pre-game Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** (-((elo_home - elo_away + ELO_HFA) / 400.0)))


def build_elo(games):
    """One chronological Elo pass over every game. Returns
    (per-game pre-ratings keyed by GamePk, per-team post-game history).

    This is the winner model's answer to April: win%/run-diff/form all
    reset to NaN each season, but true team strength carries over the
    winter. Elo starts each season at 1500 + ELO_CARRY * (last - 1500)."""
    g = games.copy()
    g["AwayR"] = pd.to_numeric(g["AwayScore"], errors="coerce")
    g["HomeR"] = pd.to_numeric(g["HomeScore"], errors="coerce")
    g = g.dropna(subset=["AwayR", "HomeR"]).sort_values(["Date", "GamePk"])
    state = {}  # team -> (season, elo after last game)
    pre, hist = [], []

    def rating(team, season):
        s_prev, e = state.get(team, (None, ELO_BASE))
        if s_prev is not None and s_prev != season:
            e = ELO_BASE + ELO_CARRY * (e - ELO_BASE)
        return e

    cur_date, day_start = None, {}
    for r in g.itertuples():
        # feature ratings are DAY-START: inference (Stores.team_elo) can only
        # see games strictly before the date, so game 1 of a doubleheader
        # must not leak into game 2's pre-game features
        if r.Date != cur_date:
            cur_date, day_start = r.Date, {}
        for team in (r.AwayTeam, r.HomeTeam):
            day_start.setdefault(team, rating(team, r.Season))
        ea_f, eh_f = day_start[r.AwayTeam], day_start[r.HomeTeam]
        pre.append((r.GamePk, ea_f, eh_f, elo_expected(eh_f, ea_f)))
        # the update chain itself stays game-by-game (running ratings)
        ea = rating(r.AwayTeam, r.Season)
        eh = rating(r.HomeTeam, r.Season)
        exp_h = elo_expected(eh, ea)
        out = 1.0 if r.HomeR > r.AwayR else 0.0
        delta = ELO_K * (out - exp_h)
        state[r.HomeTeam] = (r.Season, eh + delta)
        state[r.AwayTeam] = (r.Season, ea - delta)
        hist.append((r.AwayTeam, r.Season, r.Date, ea - delta))
        hist.append((r.HomeTeam, r.Season, r.Date, eh + delta))
    pre = pd.DataFrame(pre, columns=["GamePk", "away_elo", "home_elo",
                                     "elo_prob_home"])
    hist = pd.DataFrame(hist, columns=["Team", "Season", "Date", "elo_post"])
    return pre, hist


FORM_N, FORM_MIN = 20, 5  # recent form: last 20 games, need at least 5


def _team_form_table(games):
    """Per team per game: win% and run diff over the previous FORM_N games
    (as-of). Season records alone are stale by August; recent form is the
    standard complement for winner models."""
    form = _team_results_rows(games)
    form["rd"] = form["RF"] - form["RA"]
    grp = form.groupby(["Team", "Season"], sort=False)
    form["w20"] = grp["W"].transform(
        lambda x: x.shift(1).rolling(FORM_N, min_periods=FORM_MIN).mean())
    form["rd20"] = grp["rd"].transform(
        lambda x: x.shift(1).rolling(FORM_N, min_periods=FORM_MIN).mean())
    form[["w20", "rd20"]] = _snap_to_day_start(form, ["Team"], ["w20", "rd20"])
    return form[["GamePk", "Team", "w20", "rd20"]]


PYTH_EXP = 1.83  # standard pythagorean exponent for MLB


def _record_feats(cum_w, cum_rf, cum_ra, cum_n):
    rf_p = cum_rf ** PYTH_EXP
    ra_p = cum_ra ** PYTH_EXP
    return {"win_pct": cum_w / cum_n, "rd_pg": (cum_rf - cum_ra) / cum_n,
            "ra_pg": cum_ra / cum_n, "pyth": rf_p / (rf_p + ra_p)}


# Fly-ball carry gain per foot of elevation (~+26 ft at Coors' 5,200 ft):
# HR distances are normalized to sea level so a thin-air 440 doesn't read
# as more raw power than a sea-level 415.
ELEV_DIST_FT = 0.005


def _hr_quality_table(hr, parks):
    """Batter's HR exit velo / launch angle / elevation-adjusted distance
    profile as-of (from the HR log)."""
    h = hr.dropna(subset=["BatterId"]).copy()
    h["BatterId"] = h["BatterId"].astype("int64")
    elev = parks.set_index("Ballpark")["Elevation_ft"]
    h["dist_adj"] = h["Distance"] - ELEV_DIST_FT * h["Ballpark"].map(elev).fillna(0)
    h = h.sort_values(["BatterId", "Date"])
    g = h.groupby("BatterId", sort=False)
    h["cum_n"] = g.cumcount() + 1
    for c, name in [("Exit Velo", "ev"), ("dist_adj", "dist"),
                    ("Angle", "angle")]:
        h[f"cum_{name}"] = g[c].cumsum()
    h["cum_dist_max"] = g["dist_adj"].cummax()
    day = h.groupby(["BatterId", "Date"], as_index=False).last()
    return day[["BatterId", "Date", "cum_n", "cum_ev", "cum_dist",
                "cum_dist_max", "cum_angle"]]


# Statcast contact quality: league priors + stabilization Ks (in batted
# balls) for EB shrinkage of the per-player means/shares. Fixed constants
# (like SHRINK) so training and inference compute identically.
#   name -> (prior, K, numerator day-sum, denominator day-sum)
BIP_SHRINK = {
    "ev":    (88.3, 40, "ev_sum", "ev_n"),    # avg exit velo (mph)
    "la":    (12.7, 60, "la_sum", "la_n"),    # avg launch angle (deg)
    "hh":    (0.39, 60, "hh_n", "ev_n"),      # hard-hit share (EV >= 95)
    "brl":   (0.08, 80, "brl_n", "ev_n"),     # barrel share (Savant LSA 6)
    "xba":   (0.323, 60, "xba_sum", "xba_n"),  # xBA on contact
    "xwoba": (0.371, 60, "xw_sum", "xw_n"),   # xwOBA on contact
    "gb":    (0.43, 60, "gb_n", "n"),         # ground-ball share
    "pull":  (0.436, 60, "pull_n", "hc_n"),   # pull-side share (spray)
    "pullair": (0.166, 80, "pullair_n", "hc_n"),  # pulled fly/line share
}
BIP_DECAYED = ("ev", "brl", "xwoba", "gb", "pullair")  # 90-day decay too

# pitch-level daily aggregates (scrape_pitches.py): swing-and-miss and
# plate discipline. Same (prior, K, numerator, denominator) convention;
# priors are league rates measured on the 2020-2026 file.
PD_SHRINK = {
    "swstr": (0.112, 250, "wh_n", "n"),      # swinging strikes per pitch
    "csw":   (0.275, 250, "csw_n", "n"),     # called strikes + whiffs
    "wsw":   (0.235, 120, "wh_n", "sw_n"),   # whiffs per swing
    "chase": (0.286, 150, "oz_sw", "oz_n"),  # out-of-zone swing share
    "zone":  (0.488, 250, "z_n", "n"),       # in-zone pitch share
    "fbv":   (93.9, 50, "fb_v", "fb_n"),     # avg fastball velo (FF+SI)
}

# spray-angle pull cutoff: hit coordinates -> signed degrees off center
# (negative = LF); a RHB pull = LF side, LHB = RF. Verified empirically:
# RHB ground balls average -14deg, LHB +19deg.
PULL_DEG = 15.0


def _shrunk_rates(get, pre, spec, names=None):
    """Shrunk rates from cumulative (or decayed) sums. `get(day_sum_name)`
    returns a Series (vectorized) or scalar (inference); the same shrinkage
    applies to both because decayed counts are effective sample sizes. NaN
    sums (no history) propagate to NaN."""
    out = {}
    for name in (names or spec):
        prior, k, num, den = spec[name]
        out[f"{pre}_{name}"] = (get(num) + k * prior) / (get(den) + k)
    return out


def bip_feats(get, pre, names=None):
    return _shrunk_rates(get, pre, BIP_SHRINK, names)


def pd_feats(get, pre, names=None):
    return _shrunk_rates(get, pre, PD_SHRINK, names)


def _spray_flags(b):
    """(pull, pull-air) booleans per batted ball from hit coordinates and
    batter side. Shared by both paths so the cutoff math is identical."""
    ang = np.degrees(np.arctan2(b["HcX"] - 125.42, 198.27 - b["HcY"]))
    pull = np.where(b["Stand"] == "R", ang < -PULL_DEG,
                    np.where(b["Stand"] == "L", ang > PULL_DEG, False))
    pull = pull & b["HcX"].notna().to_numpy()
    air = b["BBType"].isin(("fly_ball", "line_drive")).to_numpy()
    return pull, pull & air


def _bip_day_sums(bip, id_col):
    """Per (id, day) sums of contact-quality numerators/denominators. Each
    metric carries its own count: EV/angle/x-stats are missing on a small
    share of batted balls (tracking gaps, especially 2020)."""
    b = bip.dropna(subset=[id_col]).copy()
    b[id_col] = b[id_col].astype("int64")
    ev, la = b["ExitVelo"], b["LaunchAngle"]
    pull, pullair = _spray_flags(b)
    day = pd.DataFrame({
        id_col: b[id_col], "Date": b["Date"],
        "n": 1.0,
        "ev_n": ev.notna().astype(float), "ev_sum": ev.fillna(0.0),
        "hh_n": (ev >= 95).astype(float),
        "brl_n": (b["LSA"] == 6).astype(float),
        "la_n": la.notna().astype(float), "la_sum": la.fillna(0.0),
        "xba_n": b["xBA"].notna().astype(float),
        "xba_sum": b["xBA"].fillna(0.0),
        "xw_n": b["xwOBA"].notna().astype(float),
        "xw_sum": b["xwOBA"].fillna(0.0),
        "gb_n": (b["BBType"] == "ground_ball").astype(float),
        "hc_n": b["HcX"].notna().astype(float),
        "pull_n": pull.astype(float),
        "pullair_n": pullair.astype(float),
    })
    return day.groupby([id_col, "Date"], as_index=False).sum()


def _cum_decay_table(day, id_col):
    """Per (id, day): inclusive career cumsums (cum_*) and exp-decay cumsums
    (dk_*, stored as exp(lam*t)-weighted sums) of day-level sums. Consumers
    merge_asof with allow_exact_matches=False and must discount dk_* to the
    consuming row's date (multiply by exp(-lam * row date)) before using
    them in shrinkage — the +K*prior terms need real batted-ball units."""
    day = day.sort_values([id_col, "Date"])
    g = day.groupby(id_col, sort=False)
    t = (day["Date"] - DECAY_EPOCH).dt.days.to_numpy(dtype="float64")
    wup = np.exp(DECAY_LAM * t)
    out = day[[id_col, "Date"]].copy()
    cols = [c for c in day.columns if c not in (id_col, "Date")]
    for c in cols:
        out[f"cum_{c}"] = g[c].cumsum()
        xw = day[c].to_numpy(dtype="float64") * wup
        out[f"dk_{c}"] = pd.Series(xw, index=day.index).groupby(
            day[id_col]).cumsum()
    return out


def _bip_table(bip, id_col):
    return _cum_decay_table(_bip_day_sums(bip, id_col), id_col)


def _bip_team_tables(bip, gb, gp):
    """Team-level contact quality, season-cumulative as-of (like every other
    team table): OFFENSE — the team's own batters' xwOBA-on-contact and
    barrel share (season scoring runs through contact quality before it
    shows up in R/PA) — and BULLPEN contact allowed (relief appearances
    only). The BIP file carries no team column, so rows are mapped through
    the game logs on (GamePk, PlayerId)."""
    def day_rows(side_df, id_col, log, log_flag=None):
        d = side_df.dropna(subset=[id_col])
        d = d[["GamePk", id_col, "Date", "xwOBA", "LSA", "ExitVelo"]].copy()
        d[id_col] = d[id_col].astype("int64")
        m = log if log_flag is None else log[log_flag]
        m = m[["GamePk", "PlayerId", "Team", "Season"]].rename(
            columns={"PlayerId": id_col})
        d = d.merge(m, on=["GamePk", id_col], how="inner")
        return pd.DataFrame({
            "Team": d["Team"], "Season": d["Season"], "Date": d["Date"],
            "xw_n": d["xwOBA"].notna().astype(float),
            "xw_sum": d["xwOBA"].fillna(0.0),
            "brl_n": (d["LSA"] == 6).astype(float),
            "ev_n": d["ExitVelo"].notna().astype(float),
        })
    stats = ["xw_n", "xw_sum", "brl_n", "ev_n"]
    off = _daily_cum(day_rows(bip, "BatterId", gb), ["Team", "Season"], stats)
    pen = _daily_cum(day_rows(bip, "PitcherId", gp, gp["GS"] == 0),
                     ["Team", "Season"], stats)
    return off, pen


# Hand-split contact quality: the same xwOBA-on-contact / barrel-share
# shrinkage as BIP_SHRINK, keyed by the OTHER side's hand. This is the
# process-stat platoon split — the outcome-based vsh_* rates need hundreds
# of PA to stabilize; contact quality vs a hand gets there in a fraction.
BVH_METRICS = ("xwoba", "brl")
BVH_SUMS = ("n", "ev_n", "brl_n", "xw_n", "xw_sum")


def _bip_hand_table(bip, id_col, hand_col):
    """Per (id, hand, day): inclusive cumulative sums for the hand-split
    shrinkage — batter contact vs pitcher hand (hand_col='PThrows') or
    pitcher contact allowed by batter side (hand_col='Stand'). Consumers
    merge_asof with allow_exact_matches=False, like every BIP table."""
    b = bip.dropna(subset=[id_col])
    b = b[b[hand_col].isin(("L", "R"))].copy()
    b[id_col] = b[id_col].astype("int64")
    day = pd.DataFrame({
        id_col: b[id_col], hand_col: b[hand_col], "Date": b["Date"],
        "n": 1.0,
        "ev_n": b["ExitVelo"].notna().astype(float),
        "brl_n": (b["LSA"] == 6).astype(float),
        "xw_n": b["xwOBA"].notna().astype(float),
        "xw_sum": b["xwOBA"].fillna(0.0),
    }).groupby([id_col, hand_col, "Date"], as_index=False).sum()
    day = day.sort_values([id_col, hand_col, "Date"])
    g = day.groupby([id_col, hand_col], sort=False)
    out = day[[id_col, hand_col, "Date"]].copy()
    for c in BVH_SUMS:
        out[f"cum_{c}"] = g[c].cumsum()
    return out


def _pitcher_name_map(gp):
    """(Name, Team) -> PlayerId from the pitching log, with a name-only
    fallback for unambiguous names. The HR log identifies the pitcher only
    by name+team; this recovers the id for the pitcher-side quality table."""
    by_team = (gp.drop_duplicates(["Name", "Team"])
               .set_index(["Name", "Team"])["PlayerId"])
    uniq = gp.groupby("Name")["PlayerId"].nunique()
    by_name = (gp[gp["Name"].isin(uniq[uniq == 1].index)]
               .drop_duplicates("Name").set_index("Name")["PlayerId"])
    return by_team, by_name


def _pitcher_hr_allowed_table(hr, gp, parks):
    """HR quality ALLOWED per pitcher as-of: how many, how hard, how far
    (elevation-adjusted). pc_hr_bf says how often a starter is taken deep;
    this says whether those homers are wall-scrapers or moonshots."""
    by_team, by_name = _pitcher_name_map(gp)
    h = hr.copy()
    key = pd.MultiIndex.from_frame(h[["Pitcher", "PitcherTeam"]])
    h["PitcherId"] = by_team.reindex(key).to_numpy()
    miss = h["PitcherId"].isna()
    h.loc[miss, "PitcherId"] = by_name.reindex(h.loc[miss, "Pitcher"]).to_numpy()
    h = h.dropna(subset=["PitcherId"])
    h["PitcherId"] = h["PitcherId"].astype("int64")
    elev = parks.set_index("Ballpark")["Elevation_ft"]
    h["dist_adj"] = h["Distance"] - ELEV_DIST_FT * h["Ballpark"].map(elev).fillna(0)
    h = h.sort_values(["PitcherId", "Date"])
    g = h.groupby("PitcherId", sort=False)
    h["cum_n"] = g.cumcount() + 1
    h["cum_ev"] = g["Exit Velo"].cumsum()
    h["cum_dist"] = g["dist_adj"].cumsum()
    day = h.groupby(["PitcherId", "Date"], as_index=False).last()
    return day[["PitcherId", "Date", "cum_n", "cum_ev", "cum_dist"]]


def _hr_pitch_counts(hr):
    """Cumulative career HR count per (BatterId, Pitch name, Date), for the
    HR-by-pitch-type matchup score. Rows are day-end totals; consumers join
    with allow_exact_matches=False so a game on D sees counts through D-1."""
    h = hr.dropna(subset=["BatterId"]).copy()
    h["BatterId"] = h["BatterId"].astype("int64")
    h = h.sort_values(["BatterId", "Pitch", "Date"])
    h["cnt"] = h.groupby(["BatterId", "Pitch"], sort=False).cumcount() + 1
    return h.groupby(["BatterId", "Pitch", "Date"], as_index=False)["cnt"].max()


def hrpt_from_counts(counts, usage, total_hr):
    """HR-by-pitch-type matchup score from a batter's per-pitch HR counts, a
    pitcher's usage fractions {pitch name: fraction}, and the batter's total
    career HR. Shared by the vectorized and inference paths: the share of
    the batter's homers that came on the pitches this starter actually
    throws, weighted by how often he throws them."""
    if not total_hr or pd.isna(total_hr) or not usage:
        return np.nan
    num = sum(frac * counts.get(pitch, 0) for pitch, frac in usage.items())
    return num / total_hr


def _hrpt_scores(df, hr, ars_p):
    """Vectorized hrpt: one score per batter-frame row (df needs PlayerId,
    StarterId, Season, Date). Prior-season usage; as-of HR counts."""
    u = ars_p.dropna(subset=["%"])[["PlayerId", "Year", "Pitch", "%"]].rename(
        columns={"PlayerId": "StarterId"})
    u["StarterId"] = u["StarterId"].astype(df["StarterId"].dtype)
    u["Season"] = u["Year"] + 1
    u["frac"] = u["%"] / 100.0
    pairs = df[["PlayerId", "StarterId", "Season", "Date"]].reset_index()
    m = pairs.merge(u[["StarterId", "Season", "Pitch", "frac"]],
                    on=["StarterId", "Season"], how="inner")
    m = m.rename(columns={"PlayerId": "BatterId"})
    cnts = _hr_pitch_counts(hr)
    m = pd.merge_asof(m.sort_values("Date"), cnts.sort_values("Date"),
                      on="Date", by=["BatterId", "Pitch"],
                      direction="backward", allow_exact_matches=False)
    m["w"] = m["frac"] * m["cnt"].fillna(0)
    return m.groupby("index")["w"].sum()


def _asof_merge(left, right, by, date_col="Date"):
    left = left.sort_values(date_col)
    right = right.sort_values(date_col)
    out = pd.merge_asof(left, right, on=date_col, by=by,
                        direction="backward", allow_exact_matches=False)
    return out


# ------------------------------------------------------- matchup features

ARS_B_METRICS = {"xSLG": "m_xslg", "xwOBA": "m_xwoba",
                 "Whiff %": "m_whiff", "Hard Hit %": "m_hh"}
ARS_P_METRICS = {"xSLG": "pars_xslg", "Whiff %": "pars_whiff",
                 "Hard Hit %": "pars_hh", "RV/100": "pars_rv100"}
# K-model view of a starter's arsenal: the strikeout-predictive metrics
ARS_K_METRICS = {"Whiff %": "pars_whiff", "K%": "pars_kpct",
                 "Put Away %": "pars_paway", "RV/100": "pars_rv100"}
# two-year decay blend: last season dominates, the season before stabilizes
# small samples; either year alone is used when the other is missing
ARS_W1, ARS_W2 = 0.7, 0.3


def _blend_years(out1, out2, cols):
    """ARS_W1*y-1 + ARS_W2*y-2 per column, falling back to whichever year
    exists when the other is NaN. Inputs share an index."""
    out = out1[cols].copy()
    for c in cols:
        b = ARS_W1 * out1[c] + ARS_W2 * out2[c]
        out[c] = b.fillna(out1[c]).fillna(out2[c])
    return out


def _matchup_one(pairs, ars_p, ars_b, lag):
    """Batter-vs-arsenal scores against the season `lag` years back."""
    pairs = pairs.copy()
    pairs["ArsYear"] = pairs["Season"] - lag

    p = ars_p.rename(columns={"PlayerId": "PitcherId", "Year": "ArsYear"})
    p = p[["PitcherId", "ArsYear", "PitchType", "%",
           *ARS_P_METRICS.keys()]].rename(columns={"%": "usage"})
    b = ars_b.rename(columns={"PlayerId": "BatterId", "Year": "ArsYear"})
    b = b[["BatterId", "ArsYear", "PitchType", *ARS_B_METRICS.keys()]]
    b = b.rename(columns={k: v for k, v in ARS_B_METRICS.items()})

    m = pairs.merge(p, on=["PitcherId", "ArsYear"], how="left")
    m = m.merge(b, on=["BatterId", "ArsYear", "PitchType"], how="left")

    keys = ["BatterId", "PitcherId", "Season"]
    out = pairs[keys].copy().set_index(keys)
    usage = m["usage"].fillna(0)

    for src, dst in ARS_P_METRICS.items():
        w = usage * m[src]
        grp = m.assign(_w=w, _u=usage.where(m[src].notna()))
        agg = grp.groupby(keys)[["_w", "_u"]].sum(min_count=1)
        out[dst] = agg["_w"] / agg["_u"]
    for dst in ARS_B_METRICS.values():
        w = usage * m[dst]
        grp = m.assign(_w=w, _u=usage.where(m[dst].notna()))
        agg = grp.groupby(keys)[["_w", "_u"]].sum(min_count=1)
        out[dst] = agg["_w"] / agg["_u"]
    cov = m.assign(_c=usage.where(m["m_xslg"].notna(), 0)).groupby(keys)["_c"].sum()
    out["m_coverage"] = cov / 100.0
    return out


def matchup_features(pairs, ars_p, ars_b):
    """Batter-vs-starter-arsenal scores from prior-season Statcast data,
    decay-blended over the last two seasons (ARS_W1/ARS_W2).

    pairs: DataFrame[BatterId, PitcherId, Season]. For each pair, weight the
    batter's per-pitch-type results by how often the pitcher threw each
    pitch; also aggregate the pitcher's own arsenal quality.
    """
    pairs = pairs.drop_duplicates().copy()
    cols = [*ARS_P_METRICS.values(), *ARS_B_METRICS.values(), "m_coverage"]
    out1 = _matchup_one(pairs, ars_p, ars_b, 1)
    out2 = _matchup_one(pairs, ars_p, ars_b, 2)
    return _blend_years(out1, out2, cols).reset_index()


def pitcher_arsenal_feats(pitchers, ars_p):
    """Usage-weighted arsenal quality per (PitcherId, Season) for the K
    model — whiff, K%, put-away (two-strike conversion), run value — with
    the same two-year decay blend as matchup_features. pars_cov is the
    usage share the blend actually covered."""
    pitchers = pitchers.drop_duplicates().copy()

    def one(lag):
        pit = pitchers.copy()
        pit["ArsYear"] = pit["Season"] - lag
        p = ars_p.rename(columns={"PlayerId": "PitcherId", "Year": "ArsYear"})
        p = p[["PitcherId", "ArsYear", "PitchType", "%",
               *ARS_K_METRICS.keys()]].rename(columns={"%": "usage"})
        m = pit.merge(p, on=["PitcherId", "ArsYear"], how="left")
        keys = ["PitcherId", "Season"]
        out = pit[keys].copy().set_index(keys)
        usage = m["usage"].fillna(0)
        for src, dst in ARS_K_METRICS.items():
            w = usage * m[src]
            grp = m.assign(_w=w, _u=usage.where(m[src].notna()))
            agg = grp.groupby(keys)[["_w", "_u"]].sum(min_count=1)
            out[dst] = agg["_w"] / agg["_u"]
        cov = m.assign(_c=usage.where(m["Whiff %"].notna(), 0)).groupby(keys)["_c"].sum()
        out["pars_cov"] = cov / 100.0
        return out

    cols = [*ARS_K_METRICS.values(), "pars_cov"]
    return _blend_years(one(1), one(2), cols).reset_index()


# ------------------------------------------------------------ rate helpers

# Empirical-Bayes shrinkage of batter rate stats toward league priors.
# shrunk = (successes + K*prior) / (trials + K); K ~ the stat's stabilization
# sample size, so a small-sample rate is pulled to league average and a
# large-sample rate barely moves. Denoises the most important features
# (early-season and rolling windows) without leaking. Constants are fixed
# league averages so the training and inference paths compute identically.
#   name -> (prior, K, numerator stat, denominator stat)
SHRINK = {
    "hr_pa":  (0.032, 170, "HR", "PA"),
    "tb_ab":  (0.410, 320, "TB", "AB"),
    "h_ab":   (0.245, 460, "H",  "AB"),
    "k_pct":  (0.225, 60,  "SO", "PA"),
    "bb_pct": (0.085, 120, "BB", "PA"),
    "iso":    (0.165, 160, None, "AB"),   # numerator = TB - H
    "sb_pa":  (0.016, 200, "SB", "PA"),   # steals are player-idiosyncratic
    "r_pa":   (0.125, 250, "R",  "PA"),   # runs scored (lineup context)
    "rbi_pa": (0.115, 250, "RBI", "PA"),
    # productive/unproductive outs (2026-07-09, train._PRODOUT -> run/rbi):
    # GIDP is a rally-killer that erases baserunners and the batter (suppresses
    # both his run and the runners' RBI); SF is a productive out that cashes a
    # run (an RBI without a hit). Both rare + noisy -> strong prior K; league
    # rates measured off the full game-batting log (GIDP/PA 0.018, SF/PA 0.007).
    "gidp_pa": (0.018,  200, "GIDP", "PA"),
    "sf_pa":   (0.0067, 250, "SF",   "PA"),
}
SHRINK_ROLL = ("hr_pa", "tb_ab", "k_pct", "sb_pa",
               "r_pa", "rbi_pa")  # rolling: PA denominator

# H+R+RBI per-game threshold shares (own-target history, hrr props only):
# prior = league share of batter-games clearing the line (full game-batting
# log, pinch-hit games included — the same population the histories sum
# over), K in games. hrr2/hrr3 were the only props with no joint target
# history; the marginals (c_r_pa_sh etc.) miss how a batter's H/R/RBI
# cluster within games (a cleanup hitter's H and RBI arrive together).
# BENCHED (2026-07-08, selection run): 0 better / 1 worse (hrr2_ece +.0044,
# 1.5x band) / 75 within noise — hrr2/hrr3 AUC/edge/top10 all flat, xhrr
# MAE flat; the same calibration-harm-without-ranking-gain signature that
# benched xpa_*. The marginals evidently already carry the joint signal.
# Everything stays in the frames + inference path (coverage 100%, parity
# 1.7e-16, target corr ~+0.11); re-enable = re-add the six columns to
# batter_feature_cols and re-route in train.py (a d_-only retry is the
# cheapest variant if ever revisited).
HRR_SHRINK = {"hrr2_g": (0.395, 40.0), "hrr3_g": (0.250, 40.0)}

# Home-plate umpire zone tendency: his career K% and BB% per batter-faced
# over PRIOR games (both teams' pitching lines), EB-shrunk toward the league
# rate. A tight zone inflates walks and suppresses strikeouts and vice
# versa; this is the only zone-authority signal, routed to the K/BB props.
# Priors are the measured league K/BF and BB/BF; K (~5 games of BF) only
# bites for an ump's first handful of games — they accrue ~75 BF/game and
# hundreds of games across the dataset, so the estimate is otherwise firm.
UMP_K_PRIOR, UMP_BB_PRIOR, UMP_K = 0.226, 0.085, 400.0

# Statcast bat tracking (scrape_bat_tracking.py, 2023+): raw CSV column ->
# feature name. Prior-season, routed to the power props (train._BAT). BANKED
# BUT INERT until the training window covers a bat-tracking season (~2027);
# wired now so it self-activates at the rollover with no code change.
BAT_TRACK_REN = {"BatSpeed": "bt_speed", "SwingLength": "bt_swlen",
                 "HardSwingRate": "bt_hardsw", "BlastPerSwing": "bt_blast"}
BAT_TRACK_COLS = list(BAT_TRACK_REN.values())

# stolen-base success rate: prior ~ league SB% ; small K (fast to stabilize)
SB_SUCC_PRIOR, SB_SUCC_K = 0.75, 20.0
TSB_STOP_PRIOR = 1.0 - SB_SUCC_PRIOR  # league share of attempts cut down
PSB_MIN_ATT = 5  # attempts needed before a pitcher's stop-rate means much
SHRINK_COLS = ([f"c_{n}_sh" for n in SHRINK] + [f"s_{n}_sh" for n in SHRINK]
               + [f"r{w}_{n}_sh" for w in ROLL_WINDOWS for n in SHRINK_ROLL])

# rolling + decayed own R/RBI rates: computed in BOTH paths (parity
# 5.6e-17) but BENCHED out of the model superset (2026-07-08, routed to
# run/rbi as _RUNRBI_FORM): run_ece +.0063 marginal (past band) against
# ~nothing on AUC/edge; rbi mixed (top10 up, auc/ece down vs flips-alone).
# Same calibration-harm signature as xpa_*/hrr-history. Re-enable = drop
# this filter from batter_feature_cols and re-route in train.py.
RUNRBI_FORM_BENCHED = ([f"r{w}_{n}_pa_sh" for w in ROLL_WINDOWS
                        for n in ("r", "rbi")]
                       + ["d_r_pa_sh", "d_rbi_pa_sh"])


def _shrink(numer, denom, name):
    prior, k = SHRINK[name][0], SHRINK[name][1]
    return (numer + k * prior) / (denom + k)


def shrunk_from_sums(sums, pre, roll=False):
    """Shrunk rates from a mapping of stat sums (scalar path, for inference).
    `roll` uses PA as the denominator (rolling windows track no AB)."""
    out = {}
    names = SHRINK_ROLL if roll else SHRINK
    for name in names:
        _, _, num_s, den_s = SHRINK[name]
        numer = (sums["TB"] - sums["H"]) if name == "iso" else sums[num_s]
        denom = sums["PA"] if roll else sums[den_s]
        out[f"{pre}_{name}_sh"] = _shrink(numer, denom, name)
    return out


def decayed_feats(sums):
    """Decay-weighted shrunk rates from {stat: decayed as-of sum}. Shared by
    the vectorized (Series) and inference (scalar) paths so both compute
    identically. d_PA is the effective (decayed) sample size."""
    out = {"d_PA": sums["PA"]}
    for name in DECAY_SHRINK:
        _, _, num_s, _ = SHRINK[name]
        out[f"d_{name}_sh"] = _shrink(sums[num_s], sums["PA"], name)
    return out


def hrr_hist_feats(sums, den, pre):
    """Shrunk H+R+RBI threshold shares from {HRR2, HRR3} sums and a game
    count. Shared by the vectorized (Series) and inference (scalar) paths so
    both compute identically; zero sums land exactly on the league prior."""
    out = {}
    for line in (2, 3):
        prior, k = HRR_SHRINK[f"hrr{line}_g"]
        out[f"{pre}_hrr{line}_g_sh"] = (sums[f"HRR{line}"] + k * prior) / (den + k)
    return out


def _ump_game_totals(gp):
    """Per-game strikeout / walk / batters-faced totals (both teams'
    pitching lines) — the raw material for a home-plate ump's zone
    tendency. Shared by the vectorized frame and the inference store."""
    g = gp[["GamePk", "SO", "BB", "BF"]].copy()
    for c in ("SO", "BB", "BF"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g.groupby("GamePk").agg(g_SO=("SO", "sum"), g_BB=("BB", "sum"),
                                   g_BF=("BF", "sum")).reset_index()


def _ump_shrink(so, bb, bf):
    """Shrunk (K%, BB%) per batter-faced from as-of ump totals — the single
    definition both paths call so training and serving agree exactly."""
    return ((so + UMP_K * UMP_K_PRIOR) / (bf + UMP_K),
            (bb + UMP_K * UMP_BB_PRIOR) / (bf + UMP_K))


def _ump_asof(umps, gp):
    """Per-GamePk home-plate-umpire tendency, as-of (his K%/BB% over PRIOR
    games only, leakage-free). An umpire never works the plate twice in one
    day, so a strictly-prior cumsum is already date-clean — no doubleheader
    snap needed."""
    tot = _ump_game_totals(gp)
    u = umps.merge(tot, on="GamePk", how="left")
    u["HpUmpId"] = pd.to_numeric(u["HpUmpId"], errors="coerce")
    u = u.dropna(subset=["HpUmpId"]).copy()
    u["HpUmpId"] = u["HpUmpId"].astype("int64")
    for s in ("g_SO", "g_BB", "g_BF"):
        u[s] = pd.to_numeric(u[s], errors="coerce").fillna(0.0)
    u = u.sort_values(["HpUmpId", "Date", "GamePk"]).reset_index(drop=True)
    g = u.groupby("HpUmpId", sort=False)
    cso = g["g_SO"].cumsum() - u["g_SO"]
    cbb = g["g_BB"].cumsum() - u["g_BB"]
    cbf = g["g_BF"].cumsum() - u["g_BF"]
    u["ump_k_pct"], u["ump_bb_pct"] = _ump_shrink(cso, cbb, cbf)
    return u[["GamePk", "ump_k_pct", "ump_bb_pct"]]


def _merge_ump(df, raw):
    """Attach ump_k_pct/ump_bb_pct by GamePk (NaN when the umpire file is
    absent — old cache / not yet scraped — so the models impute harmlessly)."""
    if raw.get("umps") is not None:
        return df.merge(_ump_asof(raw["umps"], raw["gp"]), on="GamePk",
                        how="left")
    df["ump_k_pct"] = np.nan
    df["ump_bb_pct"] = np.nan
    return df


def add_bat_trends(d):
    """Form/trend deltas from existing shrunk rates: rolling-15 vs season
    (hot/cold streak) and season vs career (breakout/decline year). The trees
    could build these from the levels, but handing them the direction of
    change directly is denoised signal. Works on a DataFrame (training) or a
    plain row dict (inference); NaN inputs propagate to NaN."""
    d["tr15_hr"] = d["r15_hr_pa_sh"] - d["s_hr_pa_sh"]
    d["tr15_tb"] = d["r15_tb_ab_sh"] - d["s_tb_ab_sh"]
    d["tr15_k"] = d["r15_k_pct_sh"] - d["s_k_pct_sh"]
    d["dev_hr"] = d["s_hr_pa_sh"] - d["c_hr_pa_sh"]
    d["dev_tb"] = d["s_tb_ab_sh"] - d["c_tb_ab_sh"]
    d["dev_k"] = d["s_k_pct_sh"] - d["c_k_pct_sh"]
    return d


def add_pit_trends(d):
    """Starter form deltas: last-5-starts K/HR rates vs season, season ERA vs
    career — in-season improvement/decline the flat rates hide. Same
    DataFrame-or-dict duality as add_bat_trends."""
    d["p5_k_trend"] = d["p5_k_bf"] - d["ps_k_bf"]
    d["p5_hr_trend"] = d["p5_hr_bf"] - d["ps_hr_bf"]
    d["p_era_trend"] = d["ps_era"] - d["pc_era"]
    return d


def _bat_rates(d, pre):
    """Turn cumulative sums with prefix `pre` into rate features."""
    pa = d[f"{pre}_PA"]
    ab = d[f"{pre}_AB"]
    return pd.DataFrame({
        f"{pre}_hr_pa": d[f"{pre}_HR"] / pa,
        f"{pre}_tb_ab": d[f"{pre}_TB"] / ab,
        f"{pre}_h_ab": d[f"{pre}_H"] / ab,
        f"{pre}_k_pct": d[f"{pre}_SO"] / pa,
        f"{pre}_bb_pct": d[f"{pre}_BB"] / pa,
        f"{pre}_iso": (d[f"{pre}_TB"] - d[f"{pre}_H"]) / ab,
    })


def _bat_rates_shrunk(d, pre):
    """Vectorized empirical-Bayes shrunk rates (career/season prefixes)."""
    out = {}
    for name in SHRINK:
        _, _, num_s, den_s = SHRINK[name]
        numer = (d[f"{pre}_TB"] - d[f"{pre}_H"]) if name == "iso" else d[f"{pre}_{num_s}"]
        out[f"{pre}_{name}_sh"] = _shrink(numer, d[f"{pre}_{den_s}"], name)
    return pd.DataFrame(out)


def _pit_rates(d, pre):
    bf = d[f"{pre}_BF"]
    outs = d[f"{pre}_Outs"]
    return pd.DataFrame({
        f"{pre}_hr_bf": d[f"{pre}_HR"] / bf,
        f"{pre}_k_bf": d[f"{pre}_SO"] / bf,
        f"{pre}_bb_bf": d[f"{pre}_BB"] / bf,
        f"{pre}_h_bf": d[f"{pre}_H"] / bf,
        f"{pre}_era": d[f"{pre}_ER"] * 27 / outs,
        f"{pre}_strike_pct": d[f"{pre}_Strikes"] / d[f"{pre}_NP"],
    })


# ------------------------------------------- prior-season lookup tables


def _batter_season_table(bat_season):
    """Per (PlayerId, Year): GO/AO, PA-weighted across team stints. The
    season files are the only GO/AO source (fly-ball tendency)."""
    t = bat_season.dropna(subset=["PlayerId"]).copy()
    t["_w"] = t["PA"].fillna(0) * t["GO/AO"].notna()
    t["_wg"] = t["GO/AO"] * t["_w"]
    g = t.groupby(["PlayerId", "Year"])[["_w", "_wg"]].sum(min_count=1)
    out = pd.DataFrame({"bat_goao": g["_wg"] / g["_w"]})
    return out.reset_index()


def _pitcher_season_table(pit_season):
    """Per (PlayerId, Year): GO/AO (TBF-weighted) plus stolen-base control —
    SB allowed per 27 outs and the stop rate (CS+PK)/attempts, NaN under
    PSB_MIN_ATT attempts. Feeds the SB prop's 'can you run on him' side."""
    t = pit_season.dropna(subset=["PlayerId"]).copy()
    t["_w"] = t["TBF"].fillna(0) * t["GO/AO"].notna()
    t["_wg"] = t["GO/AO"] * t["_w"]
    g = t.groupby(["PlayerId", "Year"]).agg(
        _w=("_w", "sum"), _wg=("_wg", "sum"), SB=("SB", "sum"),
        CS=("CS", "sum"), PK=("PK", "sum"), Outs=("Outs", "sum"))
    out = pd.DataFrame(index=g.index)
    out["pit_goao"] = g["_wg"] / g["_w"].replace(0, np.nan)
    out["psb_sb27"] = g["SB"] * 27 / g["Outs"].replace(0, np.nan)
    att = g["SB"] + g["CS"] + g["PK"]
    out["psb_stop"] = np.where(att >= PSB_MIN_ATT,
                               (g["CS"] + g["PK"]) / att, np.nan)
    return out.reset_index()


def _merge_prior_season(df, tab, id_col, cols):
    """Attach prior-season values by (id, Season-1), falling back to
    Season-2 when the player has no line the season before (injury year,
    rookie call-up mid-history). Shared join semantics for both paths."""
    for lag in (1, 2):
        t = tab.rename(columns={"PlayerId": id_col})
        t = t.assign(Season=t["Year"] + lag)[[id_col, "Season", *cols]]
        t[id_col] = t[id_col].astype(df[id_col].dtype)  # ids go float on left-merge
        t = t.rename(columns={c: f"{c}__l{lag}" for c in cols})
        df = df.merge(t, on=[id_col, "Season"], how="left")
    for c in cols:
        df[c] = df.pop(f"{c}__l1").fillna(df.pop(f"{c}__l2"))
    return df


# --------------------------------------------------------- frame assembly


def _attach_context(rows, raw, team_tab, pen_tab, park_tab,
                    pen_hl_tab=None, pen_fat_tab=None):
    """Merge game weather/park + as-of team offense, opp bullpen (overall,
    high-leverage, trailing fatigue), park factor."""
    games = raw["games"]
    rows = rows.merge(
        games[["GamePk", "Venue", "DayNight", "Temp", "Condition",
               "WindSpeed", "WindDir", "ShortGame"]], on="GamePk", how="left")
    parks = raw["parks"].rename(columns={"Ballpark": "Venue"})
    rows = rows.merge(parks[["Venue", "LF", "CF", "RF", "Elevation_ft"]],
                      on="Venue", how="left")

    team = team_tab.rename(columns={c: f"toff_{c}" for c in team_tab.columns
                                    if c.startswith("cum")})
    rows = _asof_merge(rows, team, by=["Team", "Season"])
    opp_pen = pen_tab.rename(columns={"Team": "Opponent"})
    opp_pen = opp_pen.rename(columns={c: f"pen_{c}" for c in opp_pen.columns
                                      if c.startswith("cum")})
    rows = _asof_merge(rows, opp_pen, by=["Opponent", "Season"])
    if pen_hl_tab is not None:
        hl = pen_hl_tab.rename(columns={"Team": "Opponent"})
        hl = hl.rename(columns={c: f"penhl_{c}" for c in hl.columns
                                if c.startswith("cum")})
        rows = _asof_merge(rows, hl, by=["Opponent", "Season"])
        rows["pen_hl_era"] = rows["penhl_cum_ER"] * 27 / rows["penhl_cum_Outs"]
        rows["pen_hl_k_bf"] = rows["penhl_cum_SO"] / rows["penhl_cum_BF"]
    if pen_fat_tab is not None:
        fat = pen_fat_tab.rename(columns={"Team": "Opponent"})
        rows = rows.merge(fat, on=["Opponent", "Date"], how="left")
    park = park_tab.rename(columns={c: f"park_{c}" for c in park_tab.columns
                                    if c.startswith("cum")})
    rows = _asof_merge(rows, park, by=["Venue"])

    rows["toff_hr_pa"] = rows["toff_cum_HR"] / rows["toff_cum_PA"]
    rows["toff_r_pg"] = rows["toff_cum_R"] / rows["toff_cum_n"]
    rows["toff_k_pct"] = rows["toff_cum_SO"] / rows["toff_cum_PA"]
    rows["toff_bb_pct"] = rows["toff_cum_BB"] / rows["toff_cum_PA"]
    rows["pen_hr_bf"] = rows["pen_cum_HR"] / rows["pen_cum_BF"]
    rows["pen_k_bf"] = rows["pen_cum_SO"] / rows["pen_cum_BF"]
    rows["pen_h_bf"] = rows["pen_cum_H"] / rows["pen_cum_BF"]
    rows["pen_era"] = rows["pen_cum_ER"] * 27 / rows["pen_cum_Outs"]
    # per-game venue rates over all PRIOR games, gated at 30 games so a new or
    # renamed park (Rate/Daikin) falls back to NaN. park_hr_pg reaches every
    # batter prop (legacy); R/H/2B/TB route to the offensive props only
    # (train._PARK_OFF).
    ok = rows["park_cum_n"] >= 30
    for stat, col in (("HR", "park_hr_pg"), ("R", "park_r_pg"),
                      ("H", "park_h_pg"), ("2B", "park_2b_pg"),
                      ("TB", "park_tb_pg")):
        rows[col] = np.where(ok, rows[f"park_cum_{stat}"] / rows["park_cum_n"],
                             np.nan)
    return rows


def _attach_bio(rows, rosters, player_col, prefix):
    r = rosters[["PlayerId", "B", "T", "height_in", "Wt", "DOB"]].rename(
        columns={"PlayerId": player_col})
    rows = rows.merge(r, on=player_col, how="left")
    rows[f"{prefix}_height"] = rows.pop("height_in")
    rows[f"{prefix}_weight"] = pd.to_numeric(rows.pop("Wt"), errors="coerce")
    rows[f"{prefix}_age"] = (rows["Date"] - rows.pop("DOB")).dt.days / 365.25
    return rows


def _platoon(rows):
    """Effective batter hand vs pitcher throws (switch hitters bat opposite)."""
    b, t = rows["bat_hand"], rows["pit_throws"]
    eff = np.where(b == "S", np.where(t == "L", "R", "L"), b)
    same = np.where(b.isna() | t.isna(), np.nan,
                    (pd.Series(eff, index=rows.index) == t).astype(float))
    rows["same_hand"] = same
    # eff hand unknown if batter hand unknown, or switch hitter vs unknown arm
    rows["eff_hand"] = pd.Series(eff, index=rows.index).where(
        b.notna() & ~((b == "S") & t.isna()))
    return rows


def _slot_pa_table(gb):
    """Per (slot, Date): as-of league PA per game at that lineup slot —
    strictly before the date (a day's own games never inform it). Built
    from the RAW game logs by both the training merge and the Stores
    lookup, so the two paths share one population by construction.
    cum_pa/cum_n are INCLUSIVE cumsums (through the row's own day);
    xpa_slot subtracts the own day out. First-ever day per slot is NaN."""
    slot = pd.to_numeric(gb["BattingOrder"], errors="coerce")
    starters = pd.DataFrame({
        "slot": (slot // 100), "Date": gb["Date"],
        "PA": pd.to_numeric(gb["PA"], errors="coerce").fillna(0.0),
    })[(slot % 100 == 0)]
    starters["slot"] = starters["slot"].astype(int)
    day = (starters.groupby(["slot", "Date"], sort=True)["PA"]
           .agg(day_pa="sum", day_n="size").reset_index())
    g = day.groupby("slot", sort=False)
    day["cum_pa"] = g["day_pa"].cumsum()
    day["cum_n"] = g["day_n"].cumsum()
    day["xpa_slot"] = ((day["cum_pa"] - day["day_pa"])
                       / (day["cum_n"] - day["day_n"]))
    return day


BATTER_FEATURES = None  # populated below


def build_batter_frame(raw):
    """Training frame: one row per starting batter per game, with targets."""
    gb, gp = raw["gb"], raw["gp"]
    df = _batter_asof(gb)

    # starters only (matches what the GUI knows before a game)
    df = df[df["BattingOrder"].notna()].copy()
    df["slot"] = pd.to_numeric(df["BattingOrder"], errors="coerce")
    df = df[(df["slot"] % 100 == 0)].copy()
    df["slot"] = (df["slot"] // 100).astype(int)

    # expected exposure at tonight's slot: as-of league PA/G at that
    # lineup slot (see XPA_PRIOR note; the shared table guarantees the
    # inference lookup computes the identical number)
    df = df.merge(_slot_pa_table(gb)[["slot", "Date", "xpa_slot"]],
                  on=["slot", "Date"], how="left")

    # opposing starter
    starters = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]].rename(
        columns={"Team": "Opponent", "PlayerId": "StarterId"})
    df = df.merge(starters, on=["GamePk", "Opponent"], how="left")

    st = _starter_asof(gp)
    st_feats = st[["GamePk", "PlayerId", "p_starts_career", "p_starts_season",
                   "p_days_rest", "p_np_last", "p_np_l3",
                   *[f"pc_{s}" for s in PIT_STATS], *[f"ps_{s}" for s in PIT_STATS],
                   *[f"p5_{s}" for s in ["BF", "HR", "SO", "BB", "H"]]]].rename(
        columns={"PlayerId": "StarterId"})
    df = df.merge(st_feats, on=["GamePk", "StarterId"], how="left")

    team_tab = _team_offense_table(gb)
    pen_tab = _bullpen_table(gp)
    park_tab = _park_table(gb, raw["games"])
    df = _attach_context(df, raw, team_tab, pen_tab, park_tab,
                         pen_hl_tab=_bullpen_hl_table(gp),
                         pen_fat_tab=_pen_fatigue_table(gp))
    df = df.merge(_league_env_table(gb), on="Date", how="left")

    # own team's offense in TODAY's venue context (home or road split)
    loc = _team_offense_loc_table(gb)
    loc = loc.rename(columns={c: f"tloc_{c}" for c in loc.columns
                              if c.startswith("cum")})
    df = _asof_merge(df, loc, by=["Team", "Season", "Home"])
    df["toff_loc_hr_pa"] = df["tloc_cum_HR"] / df["tloc_cum_PA"]
    df["toff_loc_r_pg"] = df["tloc_cum_R"] / df["tloc_cum_n"]

    hrq = _hr_quality_table(raw["hr"], raw["parks"]).rename(
        columns={"BatterId": "PlayerId"})
    hrq = hrq.rename(columns={"cum_n": "hrq_n", "cum_ev": "hrq_ev",
                              "cum_dist": "hrq_dist", "cum_dist_max": "hrq_dist_max",
                              "cum_angle": "hrq_angle"})
    df = _asof_merge(df, hrq, by=["PlayerId"])
    df["hrq_ev_avg"] = df["hrq_ev"] / df["hrq_n"]
    df["hrq_dist_avg"] = df["hrq_dist"] / df["hrq_n"]
    df["hrq_angle_avg"] = df["hrq_angle"] / df["hrq_n"]

    # opposing starter's HR quality ALLOWED (how hard/far he gets hit)
    phrq = _pitcher_hr_allowed_table(raw["hr"], gp, raw["parks"]).rename(
        columns={"PitcherId": "StarterId", "cum_n": "phrq_n",
                 "cum_ev": "phrq_ev", "cum_dist": "phrq_dist"})
    phrq["StarterId"] = phrq["StarterId"].astype(df["StarterId"].dtype)
    df = _asof_merge(df, phrq, by=["StarterId"])
    df["phrq_ev_avg"] = df["phrq_ev"] / df["phrq_n"]

    # HR-by-pitch-type matchup: share of the batter's career homers that
    # came on the pitches this starter actually throws (usage-weighted)
    df = df.reset_index(drop=True)
    num = _hrpt_scores(df, raw["hr"], raw["ars_p"]).reindex(df.index)
    df["hrpt_score"] = np.where(df["hrq_n"] > 0, num / df["hrq_n"], np.nan)

    # Statcast contact quality (every batted ball, not just homers): the
    # batter's career + 90-day-decayed profile, and the opposing starter's
    # contact quality ALLOWED. Frames build without the file (all NaN).
    if raw.get("bip") is not None:
        bt = _bip_table(raw["bip"], "BatterId").rename(
            columns={"BatterId": "PlayerId"})
        bt = bt.rename(columns={c: f"bb_{c}" for c in bt.columns
                                if c.startswith(("cum_", "dk_"))})
        df = _asof_merge(df, bt, by=["PlayerId"])
        pt = _bip_table(raw["bip"], "PitcherId").rename(
            columns={"PitcherId": "StarterId"})
        pt["StarterId"] = pt["StarterId"].astype(df["StarterId"].dtype)
        pt = pt.rename(columns={c: f"pb_{c}" for c in pt.columns
                                if c.startswith(("cum_", "dk_"))})
        df = _asof_merge(df, pt, by=["StarterId"])
        # dk_* sums are exp(lam*t)-scaled to each TABLE date; discount them
        # to the game row's date so the shrink K operates in real
        # batted-ball units (rates alone would cancel the scale, but the
        # +K*prior terms would not)
        wdn_row = np.exp(-DECAY_LAM * (df["Date"] - DECAY_EPOCH)
                         .dt.days.to_numpy(dtype="float64"))
        for side, tag in (("bb", "bip"), ("pb", "pbip")):
            feats = bip_feats(lambda c, s=side: df[f"{s}_cum_{c}"], tag)
            feats.update(bip_feats(
                lambda c, s=side: df[f"{s}_dk_{c}"] * wdn_row,
                f"{tag}d", BIP_DECAYED))
            for k, v in feats.items():
                df[k] = v
            df[f"{tag}_n"] = df[f"{side}_cum_n"]
            df[f"{tag}d_n"] = df[f"{side}_dk_n"] * wdn_row
    else:
        for tag in ("bip", "pbip"):
            for name in BIP_SHRINK:
                df[f"{tag}_{name}"] = np.nan
            for name in BIP_DECAYED:
                df[f"{tag}d_{name}"] = np.nan
            df[f"{tag}_n"] = np.nan
            df[f"{tag}d_n"] = np.nan

    # plate discipline (pitch-level dailies): the batter's whiff-per-swing
    # and chase rates (career + 90-day decay) and the opposing starter's
    # decayed swinging-strike rate — swing decisions are the fastest-
    # stabilizing skills in the sport and none of them are in box scores
    if raw.get("pdb") is not None:
        bt2 = _cum_decay_table(raw["pdb"], "PlayerId")
        bt2 = bt2.rename(columns={c: f"bd_{c}" for c in bt2.columns
                                  if c.startswith(("cum_", "dk_"))})
        df = _asof_merge(df, bt2, by=["PlayerId"])
    if raw.get("pdp") is not None:
        po = _cum_decay_table(raw["pdp"], "PlayerId")[
            ["PlayerId", "Date", "dk_wh_n", "dk_n"]].rename(
            columns={"PlayerId": "StarterId", "dk_wh_n": "pdo_dk_wh",
                     "dk_n": "pdo_dk_n"})
        po["StarterId"] = po["StarterId"].astype(df["StarterId"].dtype)
        df = _asof_merge(df, po, by=["StarterId"])
    wdn2 = np.exp(-DECAY_LAM * (df["Date"] - DECAY_EPOCH)
                  .dt.days.to_numpy(dtype="float64"))
    if raw.get("pdb") is not None:
        for name in ("wsw", "chase"):
            prior, k, num, den = PD_SHRINK[name]
            df[f"bd_{name}_c"] = ((df[f"bd_cum_{num}"] + k * prior)
                                  / (df[f"bd_cum_{den}"] + k))
            df[f"bd_{name}_d"] = ((df[f"bd_dk_{num}"] * wdn2 + k * prior)
                                  / (df[f"bd_dk_{den}"] * wdn2 + k))
    else:
        for name in ("wsw", "chase"):
            df[f"bd_{name}_c"] = np.nan
            df[f"bd_{name}_d"] = np.nan
    if raw.get("pdp") is not None:
        prior, k, _, _ = PD_SHRINK["swstr"]
        df["p_swstr_d"] = ((df["pdo_dk_wh"] * wdn2 + k * prior)
                           / (df["pdo_dk_n"] * wdn2 + k))
    else:
        df["p_swstr_d"] = np.nan

    # prior-season sprint speed (raw footspeed for the SB/run props) and
    # the OPPONENT's prior-season team defense (outs above average) —
    # leakage-free like GO/AO: a 2026 game sees 2025 measurements
    if raw.get("sprint") is not None:
        sp = raw["sprint"].rename(columns={"SprintSpeed": "bat_sprint",
                                           "HPto1B": "bat_hp1b"})
        df = _merge_prior_season(
            df, sp[["PlayerId", "Year", "bat_sprint", "bat_hp1b"]],
            "PlayerId", ["bat_sprint", "bat_hp1b"])
    else:
        df["bat_sprint"] = np.nan
        df["bat_hp1b"] = np.nan
    # prior-season Statcast bat tracking (bat speed / swing) — power signal
    # for the HR / total-base props. COVERAGE: 2023+ only, so under the
    # current selection suite (train <=2023) every training row is NaN and
    # the feature is INERT (unlearnable) — it is wired now so it activates
    # automatically once the season rollover puts a covered year into the
    # training window (~2027); see BAT_TRACK_COLS / train._BAT.
    if raw.get("bat_track") is not None:
        bt = raw["bat_track"].rename(columns=BAT_TRACK_REN)
        df = _merge_prior_season(df, bt[["PlayerId", "Year", *BAT_TRACK_COLS]],
                                 "PlayerId", BAT_TRACK_COLS)
    else:
        for c in BAT_TRACK_COLS:
            df[c] = np.nan
    if raw.get("oaa") is not None:
        oa = raw["oaa"].rename(columns={"Team": "PlayerId",
                                        "OAA_per162": "opp_oaa"})
        df = _merge_prior_season(df, oa[["PlayerId", "Year", "opp_oaa"]],
                                 "Opponent", ["opp_oaa"])
    else:
        df["opp_oaa"] = np.nan

    # prior-season GO/AO (fly-ball tendency) and pitcher SB control
    df = _merge_prior_season(df, _batter_season_table(raw["bat_season"]),
                             "PlayerId", ["bat_goao"])
    df = _merge_prior_season(df, _pitcher_season_table(raw["pit_season"]),
                             "StarterId", ["pit_goao", "psb_sb27", "psb_stop"])

    # opponent battery: SB allowed per game and the shrunk caught-stealing
    # rate, season-to-date (catcher-inclusive, unlike the psb_* priors)
    tsb = _team_sb_allowed_table(gb).rename(columns={"Team": "Opponent"})
    tsb = tsb.rename(columns={c: f"tsb_{c}" for c in tsb.columns
                              if c.startswith("cum_")})
    df = _asof_merge(df, tsb, by=["Opponent", "Season"])
    df["tsb_sb_g"] = df["tsb_cum_SB"] / df["tsb_cum_G"]
    df["tsb_stop"] = ((df["tsb_cum_CS"] + SB_SUCC_K * TSB_STOP_PRIOR)
                      / (df["tsb_cum_SB"] + df["tsb_cum_CS"] + SB_SUCC_K))

    pairs = df[["PlayerId", "StarterId", "Season"]].dropna().rename(
        columns={"PlayerId": "BatterId", "StarterId": "PitcherId"})
    mu = matchup_features(pairs, raw["ars_p"], raw["ars_b"])
    df = df.merge(mu, left_on=["PlayerId", "StarterId", "Season"],
                  right_on=["BatterId", "PitcherId", "Season"], how="left")

    df = _attach_bio(df, raw["rosters"], "PlayerId", "bat")
    df = df.rename(columns={"B": "bat_hand"}).drop(columns=["T"], errors="ignore")
    ros_p = raw["rosters"][["PlayerId", "T"]].rename(
        columns={"PlayerId": "StarterId", "T": "pit_throws"})
    df = df.merge(ros_p, on="StarterId", how="left")
    # statsapi handedness covers ALL players (rosters only current ones)
    bats = raw["hands"].set_index("PlayerId")["Bats"].replace("", np.nan)
    throws = raw["hands"].set_index("PlayerId")["Throws"].replace("", np.nan)
    df["bat_hand"] = df["PlayerId"].map(bats).fillna(df["bat_hand"])
    df["pit_throws"] = df["StarterId"].map(throws).fillna(df["pit_throws"])
    df = _platoon(df)

    # platoon-split features vs today's starter hand (shrunk rates)
    for s in VSH_STATS:
        df[f"vsh_{s}"] = np.where(
            df["opp_hand"] == "L", df[f"_vsL_{s}"],
            np.where(df["opp_hand"] == "R", df[f"_vsR_{s}"], np.nan))
    df["vsh_hr_pa_sh"] = _shrink(df["vsh_HR"], df["vsh_PA"], "hr_pa")
    df["vsh_tb_ab_sh"] = _shrink(df["vsh_TB"], df["vsh_PA"], "tb_ab")
    df["vsh_k_pct_sh"] = _shrink(df["vsh_SO"], df["vsh_PA"], "k_pct")

    # hand-split contact quality (process-stat platoon): the batter's career
    # contact vs pitchers of TODAY's starter's hand, and the starter's
    # contact allowed to batters of THIS batter's effective side. Unknown
    # hands merge through a never-matching sentinel -> NaN features.
    if raw.get("bip") is not None:
        df["_oh"] = df["opp_hand"].fillna("?")
        df["_eh"] = df["eff_hand"].fillna("?")
        bvh = _bip_hand_table(raw["bip"], "BatterId", "PThrows").rename(
            columns={"BatterId": "PlayerId", "PThrows": "_oh"})
        bvh = bvh.rename(columns={c: f"bvh_{c}" for c in bvh.columns
                                  if c.startswith("cum_")})
        df = _asof_merge(df, bvh, by=["PlayerId", "_oh"])
        pvh = _bip_hand_table(raw["bip"], "PitcherId", "Stand").rename(
            columns={"PitcherId": "StarterId", "Stand": "_eh"})
        pvh["StarterId"] = pvh["StarterId"].astype(df["StarterId"].dtype)
        pvh = pvh.rename(columns={c: f"pvh_{c}" for c in pvh.columns
                                  if c.startswith("cum_")})
        df = _asof_merge(df, pvh, by=["StarterId", "_eh"])
        df = df.drop(columns=["_oh", "_eh"])
        for tag in ("bvh", "pvh"):
            feats = bip_feats(lambda c, t=tag: df[f"{t}_cum_{c}"], tag,
                              BVH_METRICS)
            for k, v in feats.items():
                df[k] = v
            df[f"{tag}_n"] = df[f"{tag}_cum_n"]
    else:
        for tag in ("bvh", "pvh"):
            for name in BVH_METRICS:
                df[f"{tag}_{name}"] = np.nan
            df[f"{tag}_n"] = np.nan

    # pull-porch: the fence on the batter's pull side (lefties pull to RF),
    # and whether his typical career HR clears it
    df["pull_fence"] = np.where(df["eff_hand"] == "L", df["RF"],
                                np.where(df["eff_hand"] == "R", df["LF"],
                                         np.nan))
    df["porch_margin"] = df["hrq_dist_avg"] - df["pull_fence"]

    for pre in ["c", "s"]:
        df = pd.concat([df, _bat_rates(df, pre), _bat_rates_shrunk(df, pre)], axis=1)
    for w in ROLL_WINDOWS:
        df[f"r{w}_hr_pa"] = df[f"r{w}_HR"] / df[f"r{w}_PA"]
        df[f"r{w}_tb_ab"] = df[f"r{w}_TB"] / df[f"r{w}_PA"]
        df[f"r{w}_k_pct"] = df[f"r{w}_SO"] / df[f"r{w}_PA"]
        # shrunk rolling rates (PA denominator)
        df[f"r{w}_hr_pa_sh"] = _shrink(df[f"r{w}_HR"], df[f"r{w}_PA"], "hr_pa")
        df[f"r{w}_tb_ab_sh"] = _shrink(df[f"r{w}_TB"], df[f"r{w}_PA"], "tb_ab")
        df[f"r{w}_k_pct_sh"] = _shrink(df[f"r{w}_SO"], df[f"r{w}_PA"], "k_pct")
        df[f"r{w}_sb_pa_sh"] = _shrink(df[f"r{w}_SB"], df[f"r{w}_PA"], "sb_pa")
        df[f"r{w}_r_pa_sh"] = _shrink(df[f"r{w}_R"], df[f"r{w}_PA"], "r_pa")
        df[f"r{w}_rbi_pa_sh"] = _shrink(df[f"r{w}_RBI"], df[f"r{w}_PA"],
                                        "rbi_pa")
    # decay-weighted current-form rates + trend deltas (shared helpers)
    for k, v in decayed_feats({s: df[f"_dk_{s}"] for s in DECAY_STATS}).items():
        df[k] = v
    add_bat_trends(df)
    # batter's own decayed PA per game (exposure: pinch-hit/platoon-removal
    # risk), shrunk toward the league PA/G; a debut player sits at the prior
    df["xpa_bat"] = ((df["_dk_PA"] + XPA_K * XPA_PRIOR)
                     / (df["_dk_G"] + XPA_K))
    # H+R+RBI joint-threshold history (hrr props only — train.py _HRR_HIST):
    # career / season / 90-day-decayed share of games with 2+/3+ H+R+RBI
    for pre, src, den in (("c", "c", df["g_career"]),
                          ("s", "s", df["g_season"]),
                          ("d", "_dk", df["_dk_G"])):
        sums = {f"HRR{l}": df[f"{src}_HRR{l}"] for l in (2, 3)}
        for k, v in hrr_hist_feats(sums, den, pre).items():
            df[k] = v
    # stolen-base success rate (career, shrunk toward the league rate)
    df["c_sb_succ"] = ((df["c_SB"] + SB_SUCC_K * SB_SUCC_PRIOR)
                       / (df["c_SB"] + df["c_CS"] + SB_SUCC_K))
    # extra-base-hit rate, full OBP (incl. HBP), feared-hitter IBB rate
    for pre in ["c", "s"]:
        df[f"{pre}_xbh_ab"] = ((df[f"{pre}_2B"] + df[f"{pre}_3B"]
                                + df[f"{pre}_HR"]) / df[f"{pre}_AB"])
        df[f"{pre}_obp"] = ((df[f"{pre}_H"] + df[f"{pre}_BB"]
                             + df[f"{pre}_HBP"]) / df[f"{pre}_PA"])
    df["c_ibb_pa"] = df["c_IBB"] / df["c_PA"]
    # position wear: career share of games caught / DH'd
    df["pos_c_share"] = df["_posC_n"] / df["g_career"]
    df["pos_dh_share"] = df["_posDH_n"] / df["g_career"]
    # career splits in TODAY's venue context (home or road), shrunk
    for s in LOC_STATS:
        df[f"vloc_{s}"] = np.where(df["Home"] == 1, df[f"_loc1_{s}"],
                                   df[f"_loc0_{s}"])
    df["vloc_hr_pa_sh"] = _shrink(df["vloc_HR"], df["vloc_PA"], "hr_pa")
    df["vloc_h_pa_sh"] = _shrink(df["vloc_H"], df["vloc_PA"], "h_ab")
    df["vloc_tb_ab_sh"] = _shrink(df["vloc_TB"], df["vloc_PA"], "tb_ab")
    df["vloc_k_pct_sh"] = _shrink(df["vloc_SO"], df["vloc_PA"], "k_pct")
    for pre in ["pc", "ps"]:
        df = pd.concat([df, _pit_rates(df, pre)], axis=1)
    df["p5_hr_bf"] = df["p5_HR"] / df["p5_BF"]
    df["p5_k_bf"] = df["p5_SO"] / df["p5_BF"]
    df["p5_h_bf"] = df["p5_H"] / df["p5_BF"]
    df["p_ip_per_start"] = df["ps_Outs"] / 3 / df["p_starts_season"]
    add_pit_trends(df)

    # teammate context: career on-base of the two hitters AHEAD (they load
    # the bases for RBI) and slugging of the two BEHIND (they drive you in),
    # wrapping around the order. Uses each teammate's as-of career rates.
    df["_obpp"] = (df["c_H"] + df["c_BB"] + df["c_HBP"]) / df["c_PA"]
    df["_slgp"] = df["c_TB"] / df["c_AB"]
    # decayed variants (90-day half-life): the neighbor's CURRENT form —
    # the career rates miss a mid-season acquisition or a surging/slumping
    # neighbor entirely
    df["_obpp_d"] = (df["_dk_H"] + df["_dk_BB"] + df["_dk_HBP"]) / df["_dk_PA"]
    df["_slgp_d"] = df["_dk_TB"] / df["_dk_AB"]
    lt = df[["GamePk", "Team", "slot", "_obpp", "_slgp", "_obpp_d",
             "_slgp_d"]].drop_duplicates(["GamePk", "Team", "slot"])
    for off in (-2, -1, 1, 2):
        nb = lt.rename(columns={"slot": "_nslot", "_obpp": f"_obpp{off}",
                                "_slgp": f"_slgp{off}",
                                "_obpp_d": f"_obpp_d{off}",
                                "_slgp_d": f"_slgp_d{off}"})
        df["_nslot"] = ((df["slot"] + off - 1) % 9) + 1
        df = df.merge(nb, on=["GamePk", "Team", "_nslot"], how="left")
    df = df.drop(columns=["_nslot"])
    df["ctx_ahead_obp"] = df[["_obpp-2", "_obpp-1"]].mean(axis=1)
    df["ctx_behind_slg"] = df[["_slgp1", "_slgp2"]].mean(axis=1)
    df["ctx_ahead_obp_d"] = df[["_obpp_d-2", "_obpp_d-1"]].mean(axis=1)
    df["ctx_behind_slg_d"] = df[["_slgp_d1", "_slgp_d2"]].mean(axis=1)
    # rbi opportunity: full-order proximity-decayed OBP of the hitters ahead
    # (RBI_OPP_AHEAD) — expected runners on base the 2-slot ctx can't see.
    obp_lt = df[["GamePk", "Team", "slot", "_obpp"]].drop_duplicates(
        ["GamePk", "Team", "slot"])
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for off, w in RBI_OPP_AHEAD:
        nb = obp_lt.rename(columns={"slot": "_nslot", "_obpp": "_ahd_obp"})
        df["_nslot"] = ((df["slot"] + off - 1) % 9) + 1
        df = df.merge(nb, on=["GamePk", "Team", "_nslot"], how="left")
        v = df["_ahd_obp"].to_numpy()
        m = ~np.isnan(v)
        num[m] += w * v[m]
        den[m] += w
        df = df.drop(columns=["_nslot", "_ahd_obp"])
    df["rbi_opp_obp"] = np.where(den > 0, num / den, np.nan)

    # home-plate umpire zone tendency (as-of), merged by game
    df = _merge_ump(df, raw)

    df["month"] = df["Date"].dt.month
    df["y_hr"] = (df["HR"] >= 1).astype(int)
    df["y_hit"] = (df["H"] >= 1).astype(int)
    df["y_run"] = (df["R"] >= 1).astype(int)
    df["y_rbi"] = (df["RBI"] >= 1).astype(int)
    df["y_hits2"] = (df["H"] >= 2).astype(int)
    df["y_tb2"] = (df["TB"] >= 2).astype(int)
    df["y_bb"] = (df["BB"] >= 1).astype(int)
    df["y_sb"] = (df["SB"] >= 1).astype(int)
    df["hr_count"] = df["HR"]
    # count-market targets: singles/doubles (real O/U markets), batter
    # strikeouts (0.5/1.5 lines) and H+R+RBI (1.5/2.5 lines). H+R+RBI is
    # modeled DIRECTLY (never derived from the H/R/RBI marginals: the three
    # are strongly positively correlated — a solo HR is 3 by itself — so
    # independent marginals understate the tails).
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    hrr = df["H"] + df["R"] + df["RBI"]
    df["y_1b"] = (singles >= 1).astype(int)
    df["y_2b"] = (df["2B"] >= 1).astype(int)
    df["y_bk1"] = (df["SO"] >= 1).astype(int)
    df["y_bk2"] = (df["SO"] >= 2).astype(int)
    df["y_hrr2"] = (hrr >= 2).astype(int)
    df["y_hrr3"] = (hrr >= 3).astype(int)
    df["bk_count"] = df["SO"]
    df["hrr_count"] = hrr
    df["tb_count"] = df["TB"]   # total bases -> expected-TB head (xTB)
    return df


def batter_feature_cols():
    cols = ["slot", "Home", "Season", "month", "days_rest",
            "g_career", "g_season",
            "c_PA", "c_hr_pa", "c_tb_ab", "c_h_ab", "c_k_pct", "c_bb_pct", "c_iso",
            "s_PA", "s_hr_pa", "s_tb_ab", "s_h_ab", "s_k_pct", "s_bb_pct", "s_iso"]
    for w in ROLL_WINDOWS:
        cols += [f"r{w}_PA", f"r{w}_hr_pa", f"r{w}_tb_ab", f"r{w}_k_pct"]
    cols += ["p_starts_career", "p_starts_season", "p_days_rest",
             "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_h_bf", "pc_era",
             "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_h_bf", "ps_era",
             "p5_hr_bf", "p5_k_bf", "p5_h_bf", "p_ip_per_start",
             "toff_hr_pa", "toff_r_pg", "toff_k_pct", "toff_bb_pct",
             "toff_loc_hr_pa", "toff_loc_r_pg",
             "pen_hr_bf", "pen_k_bf", "pen_h_bf", "pen_era",
             "pen_hl_era", "pen_hl_k_bf", "pen_np_l3",
             "park_hr_pg", "park_r_pg", "park_h_pg", "park_2b_pg", "park_tb_pg",
             "LF", "CF", "RF", "Elevation_ft",
             "Temp", "WindSpeed",
             "hrq_n", "hrq_ev_avg", "hrq_dist_avg", "hrq_dist_max",
             "hrq_angle_avg", "hrpt_score", "phrq_n", "phrq_ev_avg",
             "bat_goao", "pit_goao",
             *ARS_P_METRICS.values(), *ARS_B_METRICS.values(), "m_coverage",
             "bat_height", "bat_weight", "bat_age", "same_hand",
             "vsh_PA", "vsh_hr_pa_sh", "vsh_tb_ab_sh", "vsh_k_pct_sh",
             "vloc_PA", "vloc_hr_pa_sh", "vloc_h_pa_sh", "vloc_tb_ab_sh",
             "vloc_k_pct_sh",
             "c_sb_succ", "psb_sb27", "psb_stop", "tsb_sb_g", "tsb_stop",
             "c_xbh_ab", "s_xbh_ab", "c_obp", "s_obp", "c_ibb_pa",
             "pos_c_share", "pos_dh_share",
             "ctx_ahead_obp", "ctx_behind_slg",
             # rbi_opp_obp BENCHED 2026-07-09 (train._CTX note): computed in
             # both paths but out of the superset — deeper order added no RBI
             # signal across three designs.
             # NOTE: decayed teammate ctx (ctx_*_d, 2026-07-08) BENCHED —
             # 0/0/76 within noise on run/rbi/hrr (corr with targets ~=
             # the career ctx already in the models: nothing new to add);
             # stays computed in both paths, out of the superset.
             "d_PA", "d_hr_pa_sh", "d_tb_ab_sh", "d_k_pct_sh",
             "d_bb_pct_sh", "d_sb_pa_sh",
             "tr15_hr", "tr15_tb", "tr15_k", "dev_hr", "dev_tb", "dev_k",
             "p5_k_trend", "p5_hr_trend", "p_era_trend",
             "bip_n", "bip_ev", "bip_la", "bip_hh", "bip_brl", "bip_xba",
             "bip_xwoba", "bip_gb", "bip_pull", "bip_pullair",
             "bipd_n", "bipd_ev", "bipd_brl", "bipd_xwoba", "bipd_gb",
             "bipd_pullair",
             "pbip_n", "pbip_ev", "pbip_la", "pbip_hh", "pbip_brl",
             "pbip_xba", "pbip_xwoba", "pbip_gb",
             "pbipd_n", "pbipd_ev", "pbipd_brl", "pbipd_xwoba", "pbipd_gb",
             "bd_wsw_c", "bd_wsw_d", "bd_chase_c", "bd_chase_d",
             "p_swstr_d", "bat_sprint", "bat_hp1b", "opp_oaa",
             # HP-umpire zone tendency — routed to the K/BB props only
             # (train.py _UMP); other batter props exclude it
             "ump_k_pct", "ump_bb_pct",
             # Statcast bat tracking (power) — routed to hr/tb2/xtb only
             # (train.py _BAT); INERT until ~2027 (2023+ coverage vs the
             # <=2023 training window), banked+wired to self-activate
             *BAT_TRACK_COLS,
             # NOTE: pull_fence/porch_margin and batter-side fatigue were
             # tried (iteration 3) and hurt the batter props on the holdout;
             # the league-environment lg_* columns (iteration 4) were flat
             # to slightly negative for every prop; hand-split contact
             # quality (bvh_*/pvh_*, 2026-07-07) was within noise everywhere
             # and pushed tb2 ECE past its band — ECE drifted worse in most
             # props that received it; exposure features (xpa_bat/xpa_slot,
             # 2026-07-07 eve) likewise — rbi/single/bk ECE past band, no
             # AUC/edge movement (see the XPA_PRIOR note); own H+R+RBI
             # threshold-share history (c/s/d_hrr{2,3}_g_sh, 2026-07-08)
             # likewise — hrr2_ece 1.5x past band, everything else flat
             # (see the HRR_SHRINK note). All stay in the
             # frames but out of the batter models. Fatigue (p_np_*) and
             # lg_* help the K model and live in starts cols.
             # This is the SUPERSET; per-prop trimming (e.g. SB features
             # only reach the SB model) lives in train.py PROP_EXCLUDE.
             *CAT_COLS]
    # r{w}_{r,rbi}_pa_sh benched (see the RUNRBI_FORM_BENCHED note)
    cols += [c for c in SHRINK_COLS if c not in RUNRBI_FORM_BENCHED]
    return cols


def build_starts_frame(raw, batter_frame=None):
    """Training frame for starter strikeouts: one row per start."""
    gp, gb = raw["gp"], raw["gb"]
    st = _starter_asof(gp)
    team_tab = _team_offense_table(gb)
    pen_tab = _bullpen_table(gp)
    park_tab = _park_table(gb, raw["games"])
    # opposing team offense: rename for join
    st = _attach_context(st, raw, team_tab, pen_tab, park_tab)
    # NOTE: for a pitcher, "toff_*" above is HIS team's offense; what matters
    # is the OPPONENT's. Swap: recompute against Opponent.
    opp = team_tab.rename(columns={"Team": "Opponent"})
    opp = opp.rename(columns={c: f"vs_{c}" for c in opp.columns if c.startswith("cum")})
    st = _asof_merge(st, opp, by=["Opponent", "Season"])
    st["vs_k_pct"] = st["vs_cum_SO"] / st["vs_cum_PA"]
    st["vs_bb_pct"] = st["vs_cum_BB"] / st["vs_cum_PA"]
    st["vs_hr_pa"] = st["vs_cum_HR"] / st["vs_cum_PA"]
    st["vs_r_pg"] = st["vs_cum_R"] / st["vs_cum_n"]

    for pre in ["pc", "ps"]:
        st = pd.concat([st, _pit_rates(st, pre)], axis=1)
    st["p5_hr_bf"] = st["p5_HR"] / st["p5_BF"]
    st["p5_k_bf"] = st["p5_SO"] / st["p5_BF"]
    st["p5_h_bf"] = st["p5_H"] / st["p5_BF"]
    st["p_ip_per_start"] = st["ps_Outs"] / 3 / st["p_starts_season"]
    add_pit_trends(st)
    st = st.merge(_league_env_table(gb), on="Date", how="left")

    # the starter's own arsenal, K-model view (whiff/K%/put-away, blended
    # over the last two Statcast seasons)
    pa = pitcher_arsenal_feats(
        st[["PlayerId", "Season"]].rename(columns={"PlayerId": "PitcherId"}),
        raw["ars_p"]).rename(columns={"PitcherId": "PlayerId"})
    st = st.merge(pa, on=["PlayerId", "Season"], how="left")

    # pitch-level dailies: as-of swinging-strike / CSW / whiff-per-swing /
    # chase-induced / zone rates (90-day decay + career) and the fastball
    # velocity trend (decayed minus career — the classic decline signal).
    # Unlike the prior-season arsenal blend, these move DURING the season.
    PD_C = ("swstr", "fbv")
    PD_D = ("swstr", "csw", "wsw", "chase", "zone", "fbv")
    if raw.get("pdp") is not None:
        pt2 = _cum_decay_table(raw["pdp"], "PlayerId")
        pt2 = pt2.rename(columns={c: f"pd_{c}" for c in pt2.columns
                                  if c.startswith(("cum_", "dk_"))})
        st = _asof_merge(st, pt2, by=["PlayerId"])
        wdn_pd = np.exp(-DECAY_LAM * (st["Date"] - DECAY_EPOCH)
                        .dt.days.to_numpy(dtype="float64"))
        for name in PD_C:
            prior, k, num, den = PD_SHRINK[name]
            st[f"pd_{name}_c"] = ((st[f"pd_cum_{num}"] + k * prior)
                                  / (st[f"pd_cum_{den}"] + k))
        for name in PD_D:
            prior, k, num, den = PD_SHRINK[name]
            st[f"pd_{name}_d"] = ((st[f"pd_dk_{num}"] * wdn_pd + k * prior)
                                  / (st[f"pd_dk_{den}"] * wdn_pd + k))
        st["pd_fbv_tr"] = st["pd_fbv_d"] - st["pd_fbv_c"]
    else:
        for name in PD_C:
            st[f"pd_{name}_c"] = np.nan
        for name in PD_D:
            st[f"pd_{name}_d"] = np.nan
        st["pd_fbv_tr"] = np.nan

    # the ACTUAL opposing lineup (not just team-season rates): mean as-of
    # shrunk K%/BB%, whiff vs this starter's arsenal, and K% vs his hand,
    # aggregated over the nine batters he faces. Needs the built batter
    # frame; columns stay NaN without it (old cache compatibility).
    if batter_frame is not None:
        lu = (batter_frame.groupby(["GamePk", "Team"])
              .agg(lu_k_sh=("s_k_pct_sh", "mean"),
                   lu_bb_sh=("s_bb_pct_sh", "mean"),
                   lu_whiff=("m_whiff", "mean"),
                   lu_vsh_k=("vsh_k_pct_sh", "mean"),
                   lu_wsw=("bd_wsw_d", "mean"),
                   lu_chase=("bd_chase_d", "mean"))
              .reset_index().rename(columns={"Team": "Opponent"}))
        st = st.merge(lu, on=["GamePk", "Opponent"], how="left")

    # home-plate umpire zone tendency (as-of), merged by game
    st = _merge_ump(st, raw)

    st["month"] = st["Date"].dt.month
    st["y_so"] = st["SO"]
    # count-market targets for the starter prop heads
    st["y_outs"] = st["Outs"]
    st["y_pbb"] = st["BB"]
    st["y_pha"] = st["H"]
    st["y_per"] = st["ER"]  # earned runs allowed -> expected-ER head (xER)
    return st


def starts_feature_cols():
    return ["Season", "month", "Home", "p_starts_career", "p_starts_season",
            "p_days_rest", "p_np_last", "p_np_l3",
            "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_era",
            "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_era",
            "pc_strike_pct", "ps_strike_pct",
            "p5_hr_bf", "p5_k_bf", "p_ip_per_start",
            "p5_k_trend", "p_era_trend",
            "pars_whiff", "pars_kpct", "pars_paway", "pars_rv100", "pars_cov",
            "pd_swstr_c", "pd_swstr_d", "pd_csw_d", "pd_wsw_d", "pd_chase_d",
            "pd_zone_d", "pd_fbv_c", "pd_fbv_d", "pd_fbv_tr",
            "lu_k_sh", "lu_bb_sh", "lu_whiff", "lu_vsh_k",
            "lu_wsw", "lu_chase",
            "vs_k_pct", "vs_bb_pct", "vs_hr_pa", "vs_r_pg",
            "park_hr_pg", "Elevation_ft", "Temp", "WindSpeed",
            "lg_k_pa", "lg_r_pa", "lg_hr_pa",
            # HP-umpire zone tendency: the K model uses both; the count
            # heads that don't speak to the zone (outs/pha/per) drop them
            # via train.py COUNT_HEADS st_exclude
            "ump_k_pct", "ump_bb_pct",
            "DayNight", "Condition", "WindDir"]


def build_game_frame(raw):
    """Training frame for game totals and the winner model: one row per game."""
    games, gb, gp = raw["games"], raw["gb"], raw["gp"]
    team_tab = _team_offense_table(gb)
    loc_tab = _team_offense_loc_table(gb)
    pen_tab = _bullpen_table(gp)
    pen_hl_tab = _bullpen_hl_table(gp)
    pen_fat_tab = _pen_fatigue_table(gp)
    def_tab = _team_defense_table(gp)
    park_tab = _park_table(gb, games)
    res_tab = _team_results_table(games)
    form_tab = _team_form_table(games)
    st = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]]
    stf = _starter_asof(gp)
    stf = pd.concat([stf, _pit_rates(stf, "ps"), _pit_rates(stf, "pc")], axis=1)
    stf["p5_k_bf"] = stf["p5_SO"] / stf["p5_BF"]

    # starter contact quality ALLOWED (Statcast): career + 90-day-decayed
    # xwOBA-on-contact, decayed barrel and ground-ball shares. Same shrink
    # constants as the batter frame's pbip_* (bip_feats), so the inference
    # path (Stores.bip_pitcher) matches by construction.
    ST_BIP = ["ps_xwcon", "ps_xwcon_d", "ps_brl_d", "ps_gb_d"]
    if raw.get("bip") is not None:
        pt = _bip_table(raw["bip"], "PitcherId").rename(
            columns={"PitcherId": "PlayerId"})
        pt = pt.rename(columns={c: f"pb_{c}" for c in pt.columns
                                if c.startswith(("cum_", "dk_"))})
        stf = _asof_merge(stf, pt, by=["PlayerId"])
        stf["ps_xwcon"] = bip_feats(
            lambda c: stf[f"pb_cum_{c}"], "t", ("xwoba",))["t_xwoba"]
        # discount dk_* to each start's date (see build_batter_frame note)
        wdn_st = np.exp(-DECAY_LAM * (stf["Date"] - DECAY_EPOCH)
                        .dt.days.to_numpy(dtype="float64"))
        dk = bip_feats(lambda c: stf[f"pb_dk_{c}"] * wdn_st, "t",
                       ("xwoba", "brl", "gb"))
        stf["ps_xwcon_d"] = dk["t_xwoba"]
        stf["ps_brl_d"] = dk["t_brl"]
        stf["ps_gb_d"] = dk["t_gb"]
    else:
        for c in ST_BIP:
            stf[c] = np.nan

    # team-level contact quality: own offense + bullpen allowed
    if raw.get("bip") is not None:
        bip_off_tab, bip_pen_tab = _bip_team_tables(raw["bip"], gb, gp)
    else:
        bip_off_tab = bip_pen_tab = None

    g = games[~games["ShortGame"]].copy()
    away_sc = pd.to_numeric(g["AwayScore"], errors="coerce")
    home_sc = pd.to_numeric(g["HomeScore"], errors="coerce")
    g["total_runs"] = away_sc + home_sc
    g["y_home_win"] = np.where(away_sc.notna() & home_sc.notna(),
                               (home_sc > away_sc).astype(float), np.nan)

    ST_COLS = ["ps_era", "ps_k_bf", "ps_hr_bf", "ps_bb_bf", "ps_h_bf",
               "pc_era", "pc_hr_bf", "p_days_rest", "p5_k_bf", *ST_BIP]
    for side, team_col in [("away", "AwayTeam"), ("home", "HomeTeam")]:
        t = team_tab.rename(columns={"Team": team_col})
        t = t.rename(columns={c: f"{side}_{c}" for c in t.columns if c.startswith("cum")})
        g = _asof_merge(g, t, by=[team_col, "Season"])
        g[f"{side}_hr_pa"] = g[f"{side}_cum_HR"] / g[f"{side}_cum_PA"]
        g[f"{side}_r_pg"] = g[f"{side}_cum_R"] / g[f"{side}_cum_n"]
        g[f"{side}_k_pct"] = g[f"{side}_cum_SO"] / g[f"{side}_cum_PA"]
        # offense split for the venue context this side plays in tonight
        lo = loc_tab[loc_tab["Home"] == (1 if side == "home" else 0)]
        lo = lo.drop(columns=["Home"]).rename(columns={"Team": team_col})
        lo = lo.rename(columns={c: f"{side}_loc_{c}" for c in lo.columns
                                if c.startswith("cum")})
        g = _asof_merge(g, lo, by=[team_col, "Season"])
        g[f"{side}_loc_hr_pa"] = g[f"{side}_loc_cum_HR"] / g[f"{side}_loc_cum_PA"]
        g[f"{side}_loc_r_pg"] = g[f"{side}_loc_cum_R"] / g[f"{side}_loc_cum_n"]
        p = pen_tab.rename(columns={"Team": team_col})
        p = p.rename(columns={c: f"{side}_pen_{c}" for c in p.columns if c.startswith("cum")})
        g = _asof_merge(g, p, by=[team_col, "Season"])
        g[f"{side}_pen_era"] = g[f"{side}_pen_cum_ER"] * 27 / g[f"{side}_pen_cum_Outs"]
        if bip_off_tab is not None:
            bo = bip_off_tab.rename(columns={"Team": team_col})
            bo = bo.rename(columns={c: f"{side}_bo_{c}" for c in bo.columns
                                    if c.startswith("cum")})
            g = _asof_merge(g, bo, by=[team_col, "Season"])
            g[f"{side}_xwcon"] = (g[f"{side}_bo_cum_xw_sum"]
                                  / g[f"{side}_bo_cum_xw_n"])
            g[f"{side}_brl_con"] = (g[f"{side}_bo_cum_brl_n"]
                                    / g[f"{side}_bo_cum_ev_n"])
            bp = bip_pen_tab.rename(columns={"Team": team_col})
            bp = bp.rename(columns={c: f"{side}_bp_{c}" for c in bp.columns
                                    if c.startswith("cum")})
            g = _asof_merge(g, bp, by=[team_col, "Season"])
            g[f"{side}_pen_xwcon"] = (g[f"{side}_bp_cum_xw_sum"]
                                      / g[f"{side}_bp_cum_xw_n"])
        else:
            for c in (f"{side}_xwcon", f"{side}_brl_con",
                      f"{side}_pen_xwcon"):
                g[c] = np.nan
        if raw.get("oaa") is not None:
            oa = raw["oaa"].rename(columns={"Team": "PlayerId",
                                            "OAA_per162": f"{side}_oaa"})
            g = _merge_prior_season(
                g, oa[["PlayerId", "Year", f"{side}_oaa"]], team_col,
                [f"{side}_oaa"])
        else:
            g[f"{side}_oaa"] = np.nan
        hl = pen_hl_tab.rename(columns={"Team": team_col})
        hl = hl.rename(columns={c: f"{side}_penhl_{c}" for c in hl.columns
                                if c.startswith("cum")})
        g = _asof_merge(g, hl, by=[team_col, "Season"])
        g[f"{side}_pen_hl_era"] = (g[f"{side}_penhl_cum_ER"] * 27
                                   / g[f"{side}_penhl_cum_Outs"])
        fat = pen_fat_tab.rename(columns={"Team": team_col,
                                          "pen_np_l3": f"{side}_pen_np_l3"})
        g = g.merge(fat, on=[team_col, "Date"], how="left")
        d = def_tab.rename(columns={"Team": team_col})
        d = d.rename(columns={c: f"{side}_def_{c}" for c in d.columns
                              if c.startswith("cum")})
        g = _asof_merge(g, d, by=[team_col, "Season"])
        g[f"{side}_def_uer"] = ((g[f"{side}_def_cum_R"] - g[f"{side}_def_cum_ER"])
                                * 27 / g[f"{side}_def_cum_Outs"])
        r = res_tab.rename(columns={"Team": team_col})
        r = r.rename(columns={c: f"{side}_res_{c}" for c in r.columns if c.startswith("cum")})
        g = _asof_merge(g, r, by=[team_col, "Season"])
        rec = _record_feats(g[f"{side}_res_cum_W"], g[f"{side}_res_cum_RF"],
                            g[f"{side}_res_cum_RA"], g[f"{side}_res_cum_n"])
        for k, v in rec.items():
            g[f"{side}_{k}"] = v
        fm = form_tab.rename(columns={"Team": team_col, "w20": f"{side}_w20",
                                      "rd20": f"{side}_rd20"})
        g = g.merge(fm, on=["GamePk", team_col], how="left")
        sm = st.rename(columns={"Team": team_col, "PlayerId": f"{side}_starter"})
        g = g.merge(sm, on=["GamePk", team_col], how="left")
        sf = stf[["GamePk", "PlayerId", *ST_COLS]].rename(
            columns={"PlayerId": f"{side}_starter"})
        sf = sf.rename(columns={c: f"{side}_{c}" for c in ST_COLS})
        g = g.merge(sf, on=["GamePk", f"{side}_starter"], how="left")

    elo_pre, _ = build_elo(games)
    g = g.merge(elo_pre, on="GamePk", how="left")
    g["d_elo"] = g["home_elo"] - g["away_elo"]

    # winner-model differentials (home minus away)
    for f in ["win_pct", "rd_pg", "pyth", "ra_pg", "r_pg", "w20", "rd20",
              "ps_era", "pc_era", "pen_era", "ps_k_bf"]:
        g[f"d_{f}"] = g[f"home_{f}"] - g[f"away_{f}"]
    g["d_rest"] = g["home_p_days_rest"] - g["away_p_days_rest"]
    # starter K-BB rate diff: the single most predictive starter skill
    g["d_ps_kbb"] = ((g["home_ps_k_bf"] - g["home_ps_bb_bf"])
                     - (g["away_ps_k_bf"] - g["away_ps_bb_bf"]))
    # starter contact-quality-allowed diff (decayed xwOBA on contact)
    g["d_ps_xwcon_d"] = g["home_ps_xwcon_d"] - g["away_ps_xwcon_d"]

    park = park_tab.rename(columns={c: f"park_{c}" for c in park_tab.columns
                                    if c.startswith("cum")})
    g = _asof_merge(g, park, by=["Venue"])
    g["park_hr_pg"] = np.where(g["park_cum_n"] >= 30,
                               g["park_cum_HR"] / g["park_cum_n"], np.nan)
    parks = raw["parks"].rename(columns={"Ballpark": "Venue"})
    g = g.merge(parks[["Venue", "LF", "CF", "RF", "Elevation_ft"]], on="Venue", how="left")
    g = g.merge(_league_env_table(gb), on="Date", how="left")
    g["month"] = g["Date"].dt.month
    return g


def game_feature_cols():
    cols = ["Season", "month", "Temp", "WindSpeed", "park_hr_pg",
            "LF", "CF", "RF", "Elevation_ft", "DayNight", "Condition", "WindDir"]
    for side in ["away", "home"]:
        cols += [f"{side}_hr_pa", f"{side}_r_pg", f"{side}_k_pct", f"{side}_pen_era",
                 f"{side}_ps_era", f"{side}_ps_k_bf", f"{side}_ps_hr_bf",
                 f"{side}_pc_era", f"{side}_pc_hr_bf"]
    return cols


def win_feature_cols():
    """Features for the dedicated home-win classifier: team strength (win%,
    run diff, pythag, run prevention), recent form (last 20 games), both
    starters, both bullpens, rest — and explicit home-minus-away
    differentials, which is what actually decides a winner. Deliberately
    compact (~45 columns, no weather/park/categoricals): the game frame has
    ~10k training rows, and the v1 winner model showed that batter-scale
    capacity + wide features just overfits (early-stopped at 26 trees,
    test AUC 0.52)."""
    cols = ["Season", "month"]
    for side in ["away", "home"]:
        cols += [f"{side}_win_pct", f"{side}_rd_pg", f"{side}_pyth",
                 f"{side}_ra_pg", f"{side}_r_pg", f"{side}_w20",
                 f"{side}_rd20", f"{side}_pen_era",
                 f"{side}_ps_era", f"{side}_ps_k_bf", f"{side}_ps_hr_bf",
                 f"{side}_ps_bb_bf", f"{side}_pc_era", f"{side}_pc_hr_bf",
                 f"{side}_p_days_rest", f"{side}_p5_k_bf"]
    # NOTE: the starter contact-allowed diff (d_ps_xwcon_d) was tried here
    # (2026-07 Statcast batch) and pushed the winner's logloss BELOW the
    # always-home base rate — the 10k-row winner overfits when widened, same
    # as v1. It stays in the frames and the runs model; the winner sees the
    # Statcast signal only through the runs-model Poisson blend.
    cols += ["d_win_pct", "d_rd_pg", "d_pyth", "d_ra_pg", "d_r_pg",
             "d_w20", "d_rd20", "d_ps_era", "d_pc_era", "d_pen_era",
             "d_ps_k_bf", "d_ps_kbb", "d_rest",
             "away_elo", "home_elo", "d_elo", "elo_prob_home"]
    return cols


def build_team_game_frame(gf):
    """Two rows per game (one per team): own offense vs opposing pitching.
    Target = runs that team scored. One symmetric model serves both sides;
    the Home flag carries home-field advantage. Total game runs and win
    probability are derived from the two per-team predictions, so all game
    outputs stay coherent."""
    frames = []
    for side, opp, home, score in (("away", "home", 0, "AwayScore"),
                                   ("home", "away", 1, "HomeScore")):
        d = pd.DataFrame({
            "GamePk": gf["GamePk"], "Season": gf["Season"], "Date": gf["Date"],
            "month": gf["month"], "Home": home,
            "off_hr_pa": gf[f"{side}_hr_pa"], "off_r_pg": gf[f"{side}_r_pg"],
            "off_k_pct": gf[f"{side}_k_pct"],
            "off_loc_hr_pa": gf[f"{side}_loc_hr_pa"],
            "off_loc_r_pg": gf[f"{side}_loc_r_pg"],
            "off_xwcon": gf[f"{side}_xwcon"],
            "off_brl_con": gf[f"{side}_brl_con"],
            "opp_pen_era": gf[f"{opp}_pen_era"],
            "opp_pen_hl_era": gf[f"{opp}_pen_hl_era"],
            "opp_pen_np_l3": gf[f"{opp}_pen_np_l3"],
            "opp_def_uer": gf[f"{opp}_def_uer"],
            "opp_def_oaa": gf[f"{opp}_oaa"],
            "opp_pen_xwcon": gf[f"{opp}_pen_xwcon"],
            "opp_ps_era": gf[f"{opp}_ps_era"],
            "opp_ps_k_bf": gf[f"{opp}_ps_k_bf"],
            "opp_ps_hr_bf": gf[f"{opp}_ps_hr_bf"],
            "opp_ps_h_bf": gf[f"{opp}_ps_h_bf"],
            "opp_ps_xwcon": gf[f"{opp}_ps_xwcon"],
            "opp_ps_xwcon_d": gf[f"{opp}_ps_xwcon_d"],
            "opp_ps_brl_d": gf[f"{opp}_ps_brl_d"],
            "opp_ps_gb_d": gf[f"{opp}_ps_gb_d"],
            "opp_pc_era": gf[f"{opp}_pc_era"],
            "opp_pc_hr_bf": gf[f"{opp}_pc_hr_bf"],
            "park_hr_pg": gf["park_hr_pg"], "LF": gf["LF"], "CF": gf["CF"],
            "RF": gf["RF"], "Elevation_ft": gf["Elevation_ft"],
            "Temp": gf["Temp"], "WindSpeed": gf["WindSpeed"],
            "lg_r_pa": gf["lg_r_pa"], "lg_hr_pa": gf["lg_hr_pa"],
            "DayNight": gf["DayNight"], "Condition": gf["Condition"],
            "WindDir": gf["WindDir"],
            "y_runs": pd.to_numeric(gf[score], errors="coerce"),
        })
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def team_game_feature_cols():
    # NOTE: the lg_* environment columns were tried here (iteration 4) and
    # cost the runs model a little MAE on the holdout; they stay in the
    # frame but out of this model. They remain in the batter/K models.
    return ["Season", "month", "Home", "off_hr_pa", "off_r_pg", "off_k_pct",
            "off_loc_hr_pa", "off_loc_r_pg", "off_xwcon", "off_brl_con",
            "opp_pen_era", "opp_pen_hl_era", "opp_pen_np_l3", "opp_def_uer",
            "opp_def_oaa", "opp_pen_xwcon",
            "opp_ps_era", "opp_ps_k_bf", "opp_ps_hr_bf", "opp_ps_h_bf",
            "opp_ps_xwcon", "opp_ps_xwcon_d", "opp_ps_brl_d", "opp_ps_gb_d",
            "opp_pc_era", "opp_pc_hr_bf", "park_hr_pg", "LF", "CF", "RF",
            "Elevation_ft", "Temp", "WindSpeed", "DayNight", "Condition",
            "WindDir"]


# --------------------------------------------------------------- inference


class _LazyGroups:
    """Lazy per-player history lookup; avoids materializing thousands of
    per-player DataFrames at startup."""

    def __init__(self, df, key):
        self._g = df.sort_values("Date").groupby(key, sort=False)

    def get(self, key):
        try:
            return self._g.get_group(key)
        except KeyError:
            return None


class Stores:
    """Pre-indexed history for fast as-of feature computation at predict time."""

    def __init__(self, raw=None, progress=None):
        tick = progress or (lambda msg: None)
        tick("reading CSVs...")
        self.raw = raw or load_raw()
        r = self.raw
        tick("indexing player history...")
        self.gb_by_player = _LazyGroups(r["gb"], "PlayerId")
        self.starts_by_player = _LazyGroups(r["gp"][r["gp"]["GS"] == 1], "PlayerId")
        hrb = r["hr"].dropna(subset=["BatterId"]).copy()
        hrb["BatterId"] = hrb["BatterId"].astype("int64")
        self.hr_by_batter = _LazyGroups(hrb, "BatterId")
        self.bip_by_batter = self.bip_by_pitcher = None
        self.bip_off_tab = self.bip_pen_tab = None
        if r.get("bip") is not None:
            bip = r["bip"]
            bb = bip.dropna(subset=["BatterId"]).copy()
            bb["BatterId"] = bb["BatterId"].astype("int64")
            self.bip_by_batter = _LazyGroups(bb, "BatterId")
            pb = bip.dropna(subset=["PitcherId"]).copy()
            pb["PitcherId"] = pb["PitcherId"].astype("int64")
            self.bip_by_pitcher = _LazyGroups(pb, "PitcherId")
            self.bip_off_tab, self.bip_pen_tab = _bip_team_tables(
                bip, r["gb"], r["gp"])
        self.pd_pitcher_hist = (_LazyGroups(r["pdp"], "PlayerId")
                                if r.get("pdp") is not None else None)
        self.pd_batter_hist = (_LazyGroups(r["pdb"], "PlayerId")
                               if r.get("pdb") is not None else None)
        self.sprint_prior = (
            r["sprint"].drop_duplicates(["PlayerId", "Year"], keep="last")
            .set_index(["PlayerId", "Year"])
            if r.get("sprint") is not None else None)
        self.bat_track_prior = (
            r["bat_track"].drop_duplicates(["PlayerId", "Year"], keep="last")
            .set_index(["PlayerId", "Year"])
            if r.get("bat_track") is not None else None)
        self.oaa_prior = (
            r["oaa"].drop_duplicates(["Team", "Year"], keep="last")
            .set_index(["Team", "Year"])
            if r.get("oaa") is not None else None)
        # HP-umpire history: per-game K/BB/BF totals grouped by ump for
        # as-of tendency lookups (Stores.ump_feats)
        self.ump_hist = None
        if r.get("umps") is not None:
            uh = r["umps"].merge(_ump_game_totals(r["gp"]), on="GamePk",
                                 how="left")
            uh["HpUmpId"] = pd.to_numeric(uh["HpUmpId"], errors="coerce")
            uh = uh.dropna(subset=["HpUmpId"]).copy()
            uh["HpUmpId"] = uh["HpUmpId"].astype("int64")
            for s in ("g_SO", "g_BB", "g_BF"):
                uh[s] = pd.to_numeric(uh[s], errors="coerce").fillna(0.0)
            self.ump_hist = _LazyGroups(uh, "HpUmpId")
        tick("building team/park tables...")
        self.team_tab = _team_offense_table(r["gb"])
        self.team_loc_tab = _team_offense_loc_table(r["gb"])
        self.pen_tab = _bullpen_table(r["gp"])
        self.pen_hl_tab = _bullpen_hl_table(r["gp"])
        self._pen_fat = _pen_fatigue_table(r["gp"]).set_index(
            ["Team", "Date"])["pen_np_l3"]
        self._pen_fat_max = self._pen_fat.index.get_level_values("Date").max()
        self.def_tab = _team_defense_table(r["gp"])
        self.tsb_tab = _team_sb_allowed_table(r["gb"])
        self.park_tab = _park_table(r["gb"], r["games"])
        self.env_tab = _league_env_table(r["gb"])
        self.slot_pa_tab = _slot_pa_table(r["gb"])
        self.res_rows = _team_results_rows(r["games"])
        self.res_tab = _daily_cum(self.res_rows.drop(columns=["GamePk"]),
                                  ["Team", "Season"], ["W", "RF", "RA"])
        _, self.elo_hist = build_elo(r["games"])
        self.hrq = _hr_quality_table(r["hr"], r["parks"])
        self.phrq = _pitcher_hr_allowed_table(r["hr"], r["gp"], r["parks"])
        self.bat_prior = _batter_season_table(r["bat_season"]).set_index(
            ["PlayerId", "Year"])
        self.pit_prior = _pitcher_season_table(r["pit_season"]).set_index(
            ["PlayerId", "Year"])
        self.rosters = r["rosters"].set_index("PlayerId")
        self.parks = r["parks"].set_index("Ballpark")
        h = r["hands"].replace("", np.nan)
        self.bats = h.set_index("PlayerId")["Bats"].to_dict()
        self.throws = h.set_index("PlayerId")["Throws"].to_dict()

    # -- entity helpers -----------------------------------------------

    def _cum(self, table, keys, date):
        m = table
        for k, v in keys.items():
            m = m[m[k] == v]
        m = m[m["Date"] < date]
        return m.iloc[-1] if len(m) else None

    def _vsh(self, h, opp_hand):
        """Platoon-split features vs today's starter hand (matches the
        vectorized _vsL/_vsR-then-select computation exactly)."""
        if opp_hand not in ("L", "R"):
            return {"vsh_PA": np.nan, "vsh_hr_pa_sh": np.nan,
                    "vsh_tb_ab_sh": np.nan, "vsh_k_pct_sh": np.nan}
        hh = h[h["opp_hand"] == opp_hand] if h is not None and len(h) else None
        s = (hh[VSH_STATS].sum() if hh is not None
             else {k: 0 for k in VSH_STATS})
        return {"vsh_PA": s["PA"],
                "vsh_hr_pa_sh": _shrink(s["HR"], s["PA"], "hr_pa"),
                "vsh_tb_ab_sh": _shrink(s["TB"], s["PA"], "tb_ab"),
                "vsh_k_pct_sh": _shrink(s["SO"], s["PA"], "k_pct")}

    def _vloc(self, h, home):
        """Career splits in today's venue context (home/road), matching the
        vectorized _loc0/_loc1-then-select computation exactly."""
        if home not in (0, 1):
            return {"vloc_PA": np.nan, "vloc_hr_pa_sh": np.nan,
                    "vloc_h_pa_sh": np.nan, "vloc_tb_ab_sh": np.nan,
                    "vloc_k_pct_sh": np.nan}
        hh = h[h["Home"] == home] if h is not None and len(h) else None
        s = (hh[LOC_STATS].sum() if hh is not None
             else {k: 0 for k in LOC_STATS})
        return {"vloc_PA": s["PA"],
                "vloc_hr_pa_sh": _shrink(s["HR"], s["PA"], "hr_pa"),
                "vloc_h_pa_sh": _shrink(s["H"], s["PA"], "h_ab"),
                "vloc_tb_ab_sh": _shrink(s["TB"], s["PA"], "tb_ab"),
                "vloc_k_pct_sh": _shrink(s["SO"], s["PA"], "k_pct")}

    def xpa_slot(self, slot, date):
        """As-of league PA per game at a lineup slot: the shared
        _slot_pa_table's inclusive cumsums through the last game-day
        strictly before `date` — identical to the training merge, where a
        row's value excludes its own day."""
        t = self.slot_pa_tab
        m = t[(t["slot"] == slot) & (t["Date"] < date)]
        if not len(m):
            return np.nan
        last = m.iloc[-1]
        return float(last["cum_pa"] / last["cum_n"])

    def _prior_val(self, tab, pid, season, col):
        """Prior-season lookup with the same y-1-then-y-2 fallback as
        _merge_prior_season."""
        for lag in (1, 2):
            try:
                v = tab.loc[(pid, season - lag), col]
            except KeyError:
                continue
            if pd.notna(v):
                return float(v)
        return np.nan

    def batter_feats(self, pid, date, season, opp_hand=None, home=None):
        hist = self.gb_by_player.get(pid)
        out = {}
        if hist is not None:
            h = hist[hist["Date"] < date]
        else:
            h = None
        out.update(self._vsh(h, opp_hand))
        out.update(self._vloc(h, home))
        zero = {k: 0 for k in BAT_STATS}
        if h is None or h.empty:
            for pre in ["c", "s"]:
                for k in ["PA", "hr_pa", "tb_ab", "h_ab", "k_pct", "bb_pct",
                          "iso", "xbh_ab", "obp"]:
                    out[f"{pre}_{k}"] = np.nan
                out.update(shrunk_from_sums(zero, pre))  # 0 sums -> priors
            out.update(g_career=0, g_season=0, days_rest=np.nan,
                       c_ibb_pa=np.nan, pos_c_share=np.nan,
                       pos_dh_share=np.nan, _obpp=np.nan, _slgp=np.nan,
                       _obpp_d=np.nan, _slgp_d=np.nan)
            out["c_sb_succ"] = SB_SUCC_PRIOR  # shrink of zero sums
            for w in ROLL_WINDOWS:
                for k in ["PA", "hr_pa", "tb_ab", "k_pct"]:
                    out[f"r{w}_{k}"] = np.nan
                out.update(shrunk_from_sums(zero, f"r{w}", roll=True))
            out.update(decayed_feats({s: 0.0 for s in DECAY_STATS}))
            out["xpa_bat"] = XPA_PRIOR      # zero decayed sums -> the prior
            for pre in ("c", "s", "d"):     # zero sums -> the league priors
                out.update(hrr_hist_feats({"HRR2": 0.0, "HRR3": 0.0}, 0.0, pre))
        else:
            hs = h[h["Season"] == season]
            for pre, frame in [("c", h), ("s", hs)]:
                sums = frame[BAT_STATS].sum()
                out[f"{pre}_PA"] = sums["PA"]
                out[f"{pre}_hr_pa"] = sums["HR"] / sums["PA"] if sums["PA"] else np.nan
                out[f"{pre}_tb_ab"] = sums["TB"] / sums["AB"] if sums["AB"] else np.nan
                out[f"{pre}_h_ab"] = sums["H"] / sums["AB"] if sums["AB"] else np.nan
                out[f"{pre}_k_pct"] = sums["SO"] / sums["PA"] if sums["PA"] else np.nan
                out[f"{pre}_bb_pct"] = sums["BB"] / sums["PA"] if sums["PA"] else np.nan
                out[f"{pre}_iso"] = ((sums["TB"] - sums["H"]) / sums["AB"]
                                     if sums["AB"] else np.nan)
                out[f"{pre}_xbh_ab"] = ((sums["2B"] + sums["3B"] + sums["HR"])
                                        / sums["AB"] if sums["AB"] else np.nan)
                out[f"{pre}_obp"] = ((sums["H"] + sums["BB"] + sums["HBP"])
                                     / sums["PA"] if sums["PA"] else np.nan)
                out.update(shrunk_from_sums(sums, pre))
                out.update(hrr_hist_feats(sums, len(frame), pre))
                if pre == "c":
                    out["c_ibb_pa"] = (sums["IBB"] / sums["PA"]
                                       if sums["PA"] else np.nan)
                    out["c_sb_succ"] = ((sums["SB"] + SB_SUCC_K * SB_SUCC_PRIOR)
                                        / (sums["SB"] + sums["CS"] + SB_SUCC_K))
                    out["_obpp"] = ((sums["H"] + sums["BB"] + sums["HBP"])
                                    / sums["PA"] if sums["PA"] else np.nan)
                    out["_slgp"] = (sums["TB"] / sums["AB"]
                                    if sums["AB"] else np.nan)
            out["g_career"] = len(h)
            out["g_season"] = len(hs)
            out["days_rest"] = (date - h["Date"].iloc[-1]).days
            out["pos_c_share"] = float((h["Position"] == "C").sum()) / len(h)
            out["pos_dh_share"] = float((h["Position"] == "DH").sum()) / len(h)
            for w in ROLL_WINDOWS:
                tail = h.tail(w)[["PA", "HR", "TB", "SO", "SB",
                                  "R", "RBI"]].sum()
                out[f"r{w}_PA"] = tail["PA"]
                out[f"r{w}_hr_pa"] = tail["HR"] / tail["PA"] if tail["PA"] else np.nan
                out[f"r{w}_tb_ab"] = tail["TB"] / tail["PA"] if tail["PA"] else np.nan
                out[f"r{w}_k_pct"] = tail["SO"] / tail["PA"] if tail["PA"] else np.nan
                out.update(shrunk_from_sums(tail, f"r{w}", roll=True))
            # decay-weighted sums: exp(-lam * days_ago), matching the
            # vectorized exp-cumsum-then-discount computation exactly
            wd = np.exp(-DECAY_LAM
                        * (date - h["Date"]).dt.days.to_numpy(dtype="float64"))
            dks = {s: float((h[s].to_numpy(dtype="float64") * wd).sum())
                   for s in DECAY_STATS}
            out.update(decayed_feats(dks))
            # decayed PA per game (exposure), same shrink as the frame
            out["xpa_bat"] = ((dks["PA"] + XPA_K * XPA_PRIOR)
                              / (float(wd.sum()) + XPA_K))
            # decayed H+R+RBI threshold shares (denominator = decayed game
            # count, matching the frame's _dk_G)
            out.update(hrr_hist_feats(dks, float(wd.sum()), "d"))
            # decayed own OBP/SLG: teammate-context inputs (predict.py
            # assembles ctx_*_d from the lineup's values)
            out["_obpp_d"] = ((dks["H"] + dks["BB"] + dks["HBP"]) / dks["PA"]
                              if dks["PA"] else np.nan)
            out["_slgp_d"] = dks["TB"] / dks["AB"] if dks["AB"] else np.nan
        out["bat_goao"] = self._prior_val(self.bat_prior, pid, season, "bat_goao")
        out["bat_sprint"] = (
            self._prior_val(self.sprint_prior, pid, season, "SprintSpeed")
            if self.sprint_prior is not None else np.nan)
        out["bat_hp1b"] = (
            self._prior_val(self.sprint_prior, pid, season, "HPto1B")
            if self.sprint_prior is not None else np.nan)
        # prior-season bat tracking (BatSpeed etc.) -> bt_* (parity with the
        # frame's BAT_TRACK_REN); NaN when unavailable / uncovered season
        for raw_c, feat in BAT_TRACK_REN.items():
            out[feat] = (self._prior_val(self.bat_track_prior, pid, season,
                                         raw_c)
                         if self.bat_track_prior is not None else np.nan)

        q = self.hrq[(self.hrq["BatterId"] == pid) & (self.hrq["Date"] < date)]
        if len(q):
            q = q.iloc[-1]
            out["hrq_n"] = q["cum_n"]
            out["hrq_ev_avg"] = q["cum_ev"] / q["cum_n"]
            out["hrq_dist_avg"] = q["cum_dist"] / q["cum_n"]
            out["hrq_dist_max"] = q["cum_dist_max"]
            out["hrq_angle_avg"] = q["cum_angle"] / q["cum_n"]
        else:
            out.update(hrq_n=np.nan, hrq_ev_avg=np.nan, hrq_dist_avg=np.nan,
                       hrq_dist_max=np.nan, hrq_angle_avg=np.nan)
        return out

    def _bip_entity(self, groups, pid, date, tag):
        """Contact-quality features for one batter or pitcher, as-of `date`
        (career shrunk rates + 90-day-decayed versions), mirroring the
        vectorized _bip_table math exactly."""
        nan = {f"{tag}_{n}": np.nan for n in BIP_SHRINK}
        nan.update({f"{tag}d_{n}": np.nan for n in BIP_DECAYED})
        nan.update({f"{tag}_n": np.nan, f"{tag}d_n": np.nan})
        if groups is None:
            return nan
        hist = groups.get(pid)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return nan
        ev, la = h["ExitVelo"], h["LaunchAngle"]
        pull, pullair = _spray_flags(h)
        parts = {
            "n": np.ones(len(h)),
            "ev_n": ev.notna().to_numpy(dtype="float64"),
            "ev_sum": ev.fillna(0.0).to_numpy(dtype="float64"),
            "hh_n": (ev >= 95).to_numpy(dtype="float64"),
            "brl_n": (h["LSA"] == 6).to_numpy(dtype="float64"),
            "la_n": la.notna().to_numpy(dtype="float64"),
            "la_sum": la.fillna(0.0).to_numpy(dtype="float64"),
            "xba_n": h["xBA"].notna().to_numpy(dtype="float64"),
            "xba_sum": h["xBA"].fillna(0.0).to_numpy(dtype="float64"),
            "xw_n": h["xwOBA"].notna().to_numpy(dtype="float64"),
            "xw_sum": h["xwOBA"].fillna(0.0).to_numpy(dtype="float64"),
            "gb_n": (h["BBType"] == "ground_ball").to_numpy(dtype="float64"),
            "hc_n": h["HcX"].notna().to_numpy(dtype="float64"),
            "pull_n": pull.astype(float),
            "pullair_n": pullair.astype(float),
        }
        cs = {k: float(v.sum()) for k, v in parts.items()}
        wd = np.exp(-DECAY_LAM
                    * (date - h["Date"]).dt.days.to_numpy(dtype="float64"))
        dk = {k: float((v * wd).sum()) for k, v in parts.items()}
        out = bip_feats(lambda c: cs[c], tag)
        out.update(bip_feats(lambda c: dk[c], f"{tag}d", BIP_DECAYED))
        out[f"{tag}_n"] = cs["n"]
        out[f"{tag}d_n"] = dk["n"]
        return out

    def bip_batter(self, pid, date):
        return self._bip_entity(self.bip_by_batter, pid, date, "bip")

    def bip_pitcher(self, pid, date):
        return self._bip_entity(self.bip_by_pitcher, pid, date, "pbip")

    def _bip_hand_entity(self, groups, pid, date, hand, hand_col, tag):
        """Hand-split contact quality vs one hand (career shrunk xwOBA-on-
        contact and barrel share), mirroring _bip_hand_table exactly."""
        nan = {f"{tag}_{n}": np.nan for n in BVH_METRICS}
        nan[f"{tag}_n"] = np.nan
        if groups is None or hand not in ("L", "R"):
            return nan
        hist = groups.get(pid)
        h = (hist[(hist["Date"] < date) & (hist[hand_col] == hand)]
             if hist is not None else None)
        if h is None or h.empty:
            return nan
        cs = {"n": float(len(h)),
              "ev_n": float(h["ExitVelo"].notna().sum()),
              "brl_n": float((h["LSA"] == 6).sum()),
              "xw_n": float(h["xwOBA"].notna().sum()),
              "xw_sum": float(h["xwOBA"].fillna(0.0).sum())}
        out = bip_feats(lambda c: cs[c], tag, BVH_METRICS)
        out[f"{tag}_n"] = cs["n"]
        return out

    def bip_batter_vs_hand(self, pid, date, opp_hand):
        """Batter's career contact quality vs pitchers of `opp_hand`."""
        return self._bip_hand_entity(self.bip_by_batter, pid, date,
                                     opp_hand, "PThrows", "bvh")

    def bip_pitcher_vs_hand(self, pid, date, bat_side):
        """Pitcher's career contact allowed to batters of `bat_side`."""
        return self._bip_hand_entity(self.bip_by_pitcher, pid, date,
                                     bat_side, "Stand", "pvh")

    def team_sb_allowed(self, team, season, date):
        """Battery SB control: steals allowed per game and shrunk
        caught-stealing rate by `team`, season-to-date as-of."""
        row = self._cum(self.tsb_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {"tsb_sb_g": np.nan, "tsb_stop": np.nan}
        return {"tsb_sb_g": row["cum_SB"] / row["cum_G"],
                "tsb_stop": ((row["cum_CS"] + SB_SUCC_K * TSB_STOP_PRIOR)
                             / (row["cum_SB"] + row["cum_CS"] + SB_SUCC_K))}

    def team_bip_offense(self, team, season, date):
        """Team offense contact quality (xwOBA-on-contact, barrel share),
        season-cumulative as-of."""
        row = (None if self.bip_off_tab is None else
               self._cum(self.bip_off_tab, {"Team": team, "Season": season},
                         date))
        if row is None:
            return {"off_xwcon": np.nan, "off_brl_con": np.nan}
        return {"off_xwcon": (row["cum_xw_sum"] / row["cum_xw_n"]
                              if row["cum_xw_n"] else np.nan),
                "off_brl_con": (row["cum_brl_n"] / row["cum_ev_n"]
                                if row["cum_ev_n"] else np.nan)}

    def team_bip_pen(self, team, season, date):
        """Bullpen contact quality ALLOWED (xwOBA-on-contact), as-of."""
        row = (None if self.bip_pen_tab is None else
               self._cum(self.bip_pen_tab, {"Team": team, "Season": season},
                         date))
        if row is None:
            return {"pen_xwcon": np.nan}
        return {"pen_xwcon": (row["cum_xw_sum"] / row["cum_xw_n"]
                              if row["cum_xw_n"] else np.nan)}

    def team_oaa(self, team, season):
        """Opponent team defense: prior-season outs above average per 162."""
        if self.oaa_prior is None:
            return np.nan
        return self._prior_val(self.oaa_prior, team, season, "OAA_per162")

    def ump_feats(self, ump_id, date):
        """HP-umpire zone tendency (K%/BB% per batter faced) over his games
        strictly before `date`. Unknown ump / no file -> the league prior
        (neutral), matching the vectorized _ump_asof exactly."""
        so = bb = bf = 0.0
        if ump_id is not None and self.ump_hist is not None:
            h = self.ump_hist.get(int(ump_id))
            if h is not None:
                h = h[h["Date"] < date]
                so = float(h["g_SO"].sum())
                bb = float(h["g_BB"].sum())
                bf = float(h["g_BF"].sum())
        k, w = _ump_shrink(so, bb, bf)
        return {"ump_k_pct": k, "ump_bb_pct": w}

    def _pd_sums(self, groups, pid, date):
        """(career sums, decayed sums) of the pitch-daily rows before
        `date`, or (None, None) without history. Matches _cum_decay_table
        plus the row-date discount exactly."""
        if groups is None:
            return None, None
        hist = groups.get(pid)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return None, None
        cols = [c for c in h.columns if c not in ("PlayerId", "Date")]
        cs = {c: float(h[c].sum()) for c in cols}
        wd = np.exp(-DECAY_LAM
                    * (date - h["Date"]).dt.days.to_numpy(dtype="float64"))
        dk = {c: float((h[c].to_numpy(dtype="float64") * wd).sum())
              for c in cols}
        return cs, dk

    def pd_pitcher_feats(self, pid, date):
        """Starter swing-and-miss form from the pitch-level dailies."""
        names_c, names_d = ("swstr", "fbv"), ("swstr", "csw", "wsw",
                                              "chase", "zone", "fbv")
        nan = {f"pd_{n}_c": np.nan for n in names_c}
        nan.update({f"pd_{n}_d": np.nan for n in names_d})
        nan["pd_fbv_tr"] = np.nan
        cs, dk = self._pd_sums(self.pd_pitcher_hist, pid, date)
        if cs is None:
            return nan
        out = {}
        for name in names_c:
            prior, k, num, den = PD_SHRINK[name]
            out[f"pd_{name}_c"] = (cs[num] + k * prior) / (cs[den] + k)
        for name in names_d:
            prior, k, num, den = PD_SHRINK[name]
            out[f"pd_{name}_d"] = (dk[num] + k * prior) / (dk[den] + k)
        out["pd_fbv_tr"] = out["pd_fbv_d"] - out["pd_fbv_c"]
        return out

    def pd_batter_feats(self, pid, date):
        """Batter plate discipline (whiff per swing, chase), career + decay."""
        nan = {"bd_wsw_c": np.nan, "bd_wsw_d": np.nan,
               "bd_chase_c": np.nan, "bd_chase_d": np.nan}
        cs, dk = self._pd_sums(self.pd_batter_hist, pid, date)
        if cs is None:
            return nan
        out = {}
        for name in ("wsw", "chase"):
            prior, k, num, den = PD_SHRINK[name]
            out[f"bd_{name}_c"] = (cs[num] + k * prior) / (cs[den] + k)
            out[f"bd_{name}_d"] = (dk[num] + k * prior) / (dk[den] + k)
        return out

    def hrpt(self, batter_id, pitcher_id, season, date):
        """HR-by-pitch-type matchup score (see hrpt_from_counts)."""
        hist = self.hr_by_batter.get(batter_id)
        h = hist[hist["Date"] < date] if hist is not None else None
        total = len(h) if h is not None else 0
        a = self.raw["ars_p"]
        u = a[(a["PlayerId"] == pitcher_id) & (a["Year"] == season - 1)]
        u = u.dropna(subset=["%"])
        usage = dict(zip(u["Pitch"], u["%"] / 100.0))
        counts = h["Pitch"].value_counts().to_dict() if total else {}
        return hrpt_from_counts(counts, usage, total)

    def pitcher_hr_quality(self, pid, date):
        """HR quality allowed by this pitcher, as-of (from the HR log)."""
        q = self.phrq[(self.phrq["PitcherId"] == pid)
                      & (self.phrq["Date"] < date)]
        if not len(q):
            return {"phrq_n": np.nan, "phrq_ev_avg": np.nan}
        q = q.iloc[-1]
        return {"phrq_n": q["cum_n"], "phrq_ev_avg": q["cum_ev"] / q["cum_n"]}

    def pitcher_prior(self, pid, season):
        """Prior-season GO/AO and stolen-base control for a pitcher."""
        return {c: self._prior_val(self.pit_prior, pid, season, c)
                for c in ["pit_goao", "psb_sb27", "psb_stop"]}

    def starter_feats(self, pid, date, season, prefix=""):
        hist = self.starts_by_player.get(pid)
        out = {}
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            keys = ["p_starts_career", "p_starts_season", "p_days_rest",
                    "p_np_last", "p_np_l3",
                    "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_h_bf",
                    "pc_era", "pc_strike_pct",
                    "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_h_bf",
                    "ps_era", "ps_strike_pct",
                    "p5_hr_bf", "p5_k_bf", "p5_h_bf", "p_ip_per_start"]
            return {k: np.nan for k in keys}
        hs = h[h["Season"] == season]
        out["p_starts_career"] = len(h)
        out["p_starts_season"] = len(hs)
        out["p_days_rest"] = (date - h["Date"].iloc[-1]).days
        for pre, frame in [("pc", h), ("ps", hs)]:
            s = frame[PIT_STATS].sum()
            out[f"{pre}_BF"] = s["BF"]
            out[f"{pre}_hr_bf"] = s["HR"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_k_bf"] = s["SO"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_bb_bf"] = s["BB"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_h_bf"] = s["H"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_era"] = s["ER"] * 27 / s["Outs"] if s["Outs"] else np.nan
            out[f"{pre}_strike_pct"] = (s["Strikes"] / s["NP"]
                                        if s["NP"] else np.nan)
        t5 = h.tail(5)[["BF", "HR", "SO", "H"]].sum()
        out["p5_hr_bf"] = t5["HR"] / t5["BF"] if t5["BF"] else np.nan
        out["p5_k_bf"] = t5["SO"] / t5["BF"] if t5["BF"] else np.nan
        out["p5_h_bf"] = t5["H"] / t5["BF"] if t5["BF"] else np.nan
        out["p_ip_per_start"] = (hs["Outs"].sum() / 3 / len(hs)) if len(hs) else np.nan
        out["p_np_last"] = h["NP"].iloc[-1]
        out["p_np_l3"] = h.tail(3)["NP"].mean()
        return out

    def team_offense(self, team, season, date, prefix="toff"):
        row = self._cum(self.team_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {f"{prefix}_hr_pa": np.nan, f"{prefix}_r_pg": np.nan,
                    f"{prefix}_k_pct": np.nan, f"{prefix}_bb_pct": np.nan}
        return {f"{prefix}_hr_pa": row["cum_HR"] / row["cum_PA"],
                f"{prefix}_r_pg": row["cum_R"] / row["cum_n"],
                f"{prefix}_k_pct": row["cum_SO"] / row["cum_PA"],
                f"{prefix}_bb_pct": row["cum_BB"] / row["cum_PA"]}

    def bullpen(self, team, season, date, prefix="pen"):
        row = self._cum(self.pen_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {f"{prefix}_hr_bf": np.nan, f"{prefix}_k_bf": np.nan,
                    f"{prefix}_h_bf": np.nan, f"{prefix}_era": np.nan}
        return {f"{prefix}_hr_bf": row["cum_HR"] / row["cum_BF"],
                f"{prefix}_k_bf": row["cum_SO"] / row["cum_BF"],
                f"{prefix}_h_bf": row["cum_H"] / row["cum_BF"],
                f"{prefix}_era": row["cum_ER"] * 27 / row["cum_Outs"]}

    def bullpen_hl(self, team, season, date, prefix="pen_hl"):
        """High-leverage bullpen quality (save/hold/game-finishing arms)."""
        row = self._cum(self.pen_hl_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {f"{prefix}_era": np.nan, f"{prefix}_k_bf": np.nan}
        return {f"{prefix}_era": row["cum_ER"] * 27 / row["cum_Outs"],
                f"{prefix}_k_bf": row["cum_SO"] / row["cum_BF"]}

    def pen_fatigue(self, team, date):
        """Bullpen pitches thrown over the trailing PEN_FATIGUE_DAYS days.
        Uses the same table as training; dates past the table's horizon
        (predicting more than a week out) mean a fully rested pen -> 0."""
        try:
            return float(self._pen_fat.loc[(team, date)])
        except KeyError:
            return 0.0 if date > self._pen_fat_max else np.nan

    def team_offense_loc(self, team, season, home, date, prefix="toff_loc"):
        """Team offense in one venue context (home=1 / road=0), as-of."""
        row = self._cum(self.team_loc_tab,
                        {"Team": team, "Season": season, "Home": home}, date)
        if row is None:
            return {f"{prefix}_hr_pa": np.nan, f"{prefix}_r_pg": np.nan}
        return {f"{prefix}_hr_pa": row["cum_HR"] / row["cum_PA"],
                f"{prefix}_r_pg": row["cum_R"] / row["cum_n"]}

    def team_defense(self, team, season, date):
        """Unearned-run rate allowed (defense proxy), as-of."""
        row = self._cum(self.def_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return np.nan
        return (row["cum_R"] - row["cum_ER"]) * 27 / row["cum_Outs"]

    def league_env(self, date):
        """Trailing-30-day league rates for `date`. Exact row for dates the
        data covers (train/serve parity); the latest row for future dates."""
        t = self.env_tab
        m = t[t["Date"] <= date]
        if m.empty:
            return {c: np.nan for c in ENV_COLS}
        return {c: m.iloc[-1][c] for c in ENV_COLS}

    def team_record(self, team, season, date):
        """As-of season W%, run differential, runs allowed, pythag."""
        row = self._cum(self.res_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {"win_pct": np.nan, "rd_pg": np.nan, "ra_pg": np.nan,
                    "pyth": np.nan}
        return _record_feats(row["cum_W"], row["cum_RF"], row["cum_RA"],
                             row["cum_n"])

    def team_form(self, team, season, date):
        """Win% and run diff over the previous FORM_N games (as-of)."""
        h = self.res_rows
        m = h[(h["Team"] == team) & (h["Season"] == season)
              & (h["Date"] < date)].tail(FORM_N)
        if len(m) < FORM_MIN:
            return {"w20": np.nan, "rd20": np.nan}
        return {"w20": m["W"].mean(), "rd20": (m["RF"] - m["RA"]).mean()}

    def team_elo(self, team, season, date):
        """As-of Elo rating, with the winter carryover regression applied
        when the team's last game was in an earlier season."""
        h = self.elo_hist
        m = h[(h["Team"] == team) & (h["Date"] < date)]
        if m.empty:
            return ELO_BASE
        last = m.iloc[-1]
        e = last["elo_post"]
        if last["Season"] != season:
            e = ELO_BASE + ELO_CARRY * (e - ELO_BASE)
        return e

    def park(self, venue, date):
        out = {"LF": np.nan, "CF": np.nan, "RF": np.nan, "Elevation_ft": np.nan,
               "park_hr_pg": np.nan, "park_r_pg": np.nan, "park_h_pg": np.nan,
               "park_2b_pg": np.nan, "park_tb_pg": np.nan}
        if venue in self.parks.index:
            p = self.parks.loc[venue]
            out.update(LF=p["LF"], CF=p["CF"], RF=p["RF"],
                       Elevation_ft=p["Elevation_ft"])
        row = self._cum(self.park_tab, {"Venue": venue}, date)
        if row is not None and row["cum_n"] >= 30:
            n = row["cum_n"]
            for stat, col in (("HR", "park_hr_pg"), ("R", "park_r_pg"),
                              ("H", "park_h_pg"), ("2B", "park_2b_pg"),
                              ("TB", "park_tb_pg")):
                out[col] = row[f"cum_{stat}"] / n
        return out

    def bio(self, pid):
        r = self.rosters.loc[pid] if pid in self.rosters.index else None
        bat = self.bats.get(pid)
        thr = self.throws.get(pid)
        if pd.isna(bat) or bat is None:
            bat = r["B"] if r is not None else np.nan
        if pd.isna(thr) or thr is None:
            thr = r["T"] if r is not None else np.nan
        if r is not None:
            return {"bat_hand": bat, "pit_throws": thr,
                    "height": r["height_in"],
                    "weight": pd.to_numeric(r["Wt"], errors="coerce"),
                    "dob": r["DOB"]}
        return {"bat_hand": bat, "pit_throws": thr, "height": np.nan,
                "weight": np.nan, "dob": pd.NaT}
