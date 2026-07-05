"""Feature engineering for the MLB prediction models.

Builds leakage-free features: every feature for a game on date D uses only
data from strictly before D. Two paths share the same definitions:

  - build_batter_frame / build_starts_frame / build_game_frame:
      vectorized, over all of 2020-2026, for training (cumsum/rolling
      shifted by one game so the current game never sees itself).
  - Stores.asof_*:
      per-entity, for an arbitrary future/hypothetical game, for inference.

Data used (all files in Data/):
  game logs (3 files)      -> form, rest, bullpen, team offense, park factor
  season stats (2 files)   -> (validation; game logs supersede for features)
  pitch arsenals (2 files) -> prior-season batter-vs-arsenal matchup scores
  homeruns                 -> batter HR power quality (exit velo, distance)
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

# ---------------------------------------------------------------- loading


def inf_to_nan(X):
    """sklearn rejects inf (from divide-by-zero rates); make them missing so
    the imputer handles them. Lives here (not in train's __main__) so a
    pickled pipeline referencing it loads from predict.py/evaluate.py too."""
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
    games = _read("mlb_games_2020_2026.csv", parse_dates=["Date"])
    gb = _read("mlb_game_batting_2020_2026.csv", parse_dates=["Date"])
    gp = _read("mlb_game_pitching_2020_2026.csv", parse_dates=["Date"])
    gp["Outs"] = ip_to_outs(gp["IP"])
    gp["NP"] = pd.to_numeric(gp["NP"], errors="coerce")

    rosters = _read("mlb_rosters_2026.csv")
    rosters["height_in"] = rosters["Ht"].map(height_to_inches)
    rosters["DOB"] = pd.to_datetime(rosters["DOB"], format="%m/%d/%Y", errors="coerce")

    parks = _read("mlb_ballparks.csv")

    hr = _read("mlb_homeruns_2020_2026.csv", parse_dates=["Date"])
    for c in ["Angle", "Exit Velo", "Distance"]:
        hr[c] = pd.to_numeric(hr[c], errors="coerce")
    hr["BatterId"] = pd.to_numeric(hr["BatterId"], errors="coerce")

    def _ars(name):
        a = _read(name)
        for c in ["RV/100", "%", "xSLG", "xwOBA", "Whiff %", "Hard Hit %", "Pitches"]:
            a[c] = pd.to_numeric(a[c], errors="coerce")
        return a

    ars_p = _ars("mlb_pitch_arsenals_2020_2026.csv")
    ars_b = _ars("mlb_pitch_arsenals_batters_2020_2026.csv")

    hands = _read("mlb_handedness.csv")
    hands["PlayerId"] = pd.to_numeric(hands["PlayerId"], errors="coerce")
    hands = hands.dropna(subset=["PlayerId"])
    hands["PlayerId"] = hands["PlayerId"].astype("int64")

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
               hr=hr, ars_p=ars_p, ars_b=ars_b, hands=hands)
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

BAT_STATS = ["PA", "AB", "H", "HR", "TB", "SO", "BB"]
PIT_STATS = ["BF", "HR", "SO", "BB", "Outs", "ER"]
VSH_STATS = ["PA", "HR", "TB", "SO"]  # platoon splits track these


def _batter_asof(gb):
    """Per batter-game row: career/season/rolling form, all as-of (pre-game)."""
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
        for s in ["PA", "HR", "H", "TB", "SO"]:
            df[f"r{w}_{s}"] = g[s].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).sum())
    df["days_rest"] = g["Date"].diff().dt.days
    # platoon splits: career as-of sums in games vs L / vs R opposing starters
    for hand in ("L", "R"):
        mask = df["opp_hand"] == hand
        for s in VSH_STATS:
            tmp = df[s].where(mask, 0)
            df[f"_vs{hand}_{s}"] = tmp.groupby(df["PlayerId"]).cumsum() - tmp
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
    for s in ["BF", "HR", "SO", "BB"]:
        st[f"p5_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    st["p_days_rest"] = g["Date"].diff().dt.days
    # fatigue: pitch counts in the last start and last three starts
    st["p_np_last"] = g["NP"].shift(1)
    st["p_np_l3"] = g["NP"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
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


def _bullpen_table(gp):
    pen = gp[gp["GS"] == 0]
    return _daily_cum(pen, ["Team", "Season"], ["BF", "HR", "SO", "BB", "ER", "Outs"])


def _park_table(gb, games):
    hr_per_game = gb.groupby("GamePk", as_index=False)["HR"].sum()
    gv = games.merge(hr_per_game, on="GamePk", how="left")
    gv["HR"] = gv["HR"].fillna(0)
    return _daily_cum(gv, ["Venue"], ["HR"])


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

    for r in g.itertuples():
        ea = rating(r.AwayTeam, r.Season)
        eh = rating(r.HomeTeam, r.Season)
        exp_h = elo_expected(eh, ea)
        pre.append((r.GamePk, ea, eh, exp_h))
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
    return form[["GamePk", "Team", "w20", "rd20"]]


PYTH_EXP = 1.83  # standard pythagorean exponent for MLB


def _record_feats(cum_w, cum_rf, cum_ra, cum_n):
    rf_p = cum_rf ** PYTH_EXP
    ra_p = cum_ra ** PYTH_EXP
    return {"win_pct": cum_w / cum_n, "rd_pg": (cum_rf - cum_ra) / cum_n,
            "ra_pg": cum_ra / cum_n, "pyth": rf_p / (rf_p + ra_p)}


def _hr_quality_table(hr):
    """Batter's HR exit velo / distance profile as-of (from the HR log)."""
    h = hr.dropna(subset=["BatterId"]).copy()
    h["BatterId"] = h["BatterId"].astype("int64")
    h = h.sort_values(["BatterId", "Date"])
    g = h.groupby("BatterId", sort=False)
    h["cum_n"] = g.cumcount() + 1
    for c, name in [("Exit Velo", "ev"), ("Distance", "dist")]:
        h[f"cum_{name}"] = g[c].cumsum()
    h["cum_dist_max"] = g["Distance"].cummax()
    day = h.groupby(["BatterId", "Date"], as_index=False).last()
    return day[["BatterId", "Date", "cum_n", "cum_ev", "cum_dist", "cum_dist_max"]]


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


def matchup_features(pairs, ars_p, ars_b):
    """Batter-vs-starter-arsenal scores from the PRIOR season's Statcast data.

    pairs: DataFrame[BatterId, PitcherId, Season]. For each pair, weight the
    batter's per-pitch-type results by how often the pitcher threw each
    pitch last season; also aggregate the pitcher's own arsenal quality.
    """
    pairs = pairs.drop_duplicates().copy()
    pairs["ArsYear"] = pairs["Season"] - 1

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
    return out.reset_index()


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
}
SHRINK_ROLL = ("hr_pa", "tb_ab", "k_pct")  # rolling windows use PA as denom
SHRINK_COLS = ([f"c_{n}_sh" for n in SHRINK] + [f"s_{n}_sh" for n in SHRINK]
               + [f"r{w}_{n}_sh" for w in ROLL_WINDOWS for n in SHRINK_ROLL])


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
        f"{pre}_era": d[f"{pre}_ER"] * 27 / outs,
    })


# --------------------------------------------------------- frame assembly


def _attach_context(rows, raw, team_tab, pen_tab, park_tab):
    """Merge game weather/park + as-of team offense, opp bullpen, park factor."""
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
    park = park_tab.rename(columns={"cum_HR": "park_cum_HR", "cum_n": "park_cum_n"})
    rows = _asof_merge(rows, park, by=["Venue"])

    rows["toff_hr_pa"] = rows["toff_cum_HR"] / rows["toff_cum_PA"]
    rows["toff_r_pg"] = rows["toff_cum_R"] / rows["toff_cum_n"]
    rows["toff_k_pct"] = rows["toff_cum_SO"] / rows["toff_cum_PA"]
    rows["toff_bb_pct"] = rows["toff_cum_BB"] / rows["toff_cum_PA"]
    rows["pen_hr_bf"] = rows["pen_cum_HR"] / rows["pen_cum_BF"]
    rows["pen_k_bf"] = rows["pen_cum_SO"] / rows["pen_cum_BF"]
    rows["pen_era"] = rows["pen_cum_ER"] * 27 / rows["pen_cum_Outs"]
    rows["park_hr_pg"] = np.where(rows["park_cum_n"] >= 30,
                                  rows["park_cum_HR"] / rows["park_cum_n"], np.nan)
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

    # opposing starter
    starters = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]].rename(
        columns={"Team": "Opponent", "PlayerId": "StarterId"})
    df = df.merge(starters, on=["GamePk", "Opponent"], how="left")

    st = _starter_asof(gp)
    st_feats = st[["GamePk", "PlayerId", "p_starts_career", "p_starts_season",
                   "p_days_rest", "p_np_last", "p_np_l3",
                   *[f"pc_{s}" for s in PIT_STATS], *[f"ps_{s}" for s in PIT_STATS],
                   *[f"p5_{s}" for s in ["BF", "HR", "SO", "BB"]]]].rename(
        columns={"PlayerId": "StarterId"})
    df = df.merge(st_feats, on=["GamePk", "StarterId"], how="left")

    team_tab = _team_offense_table(gb)
    pen_tab = _bullpen_table(gp)
    park_tab = _park_table(gb, raw["games"])
    df = _attach_context(df, raw, team_tab, pen_tab, park_tab)
    df = df.merge(_league_env_table(gb), on="Date", how="left")

    hrq = _hr_quality_table(raw["hr"]).rename(columns={"BatterId": "PlayerId"})
    hrq = hrq.rename(columns={"cum_n": "hrq_n", "cum_ev": "hrq_ev",
                              "cum_dist": "hrq_dist", "cum_dist_max": "hrq_dist_max"})
    df = _asof_merge(df, hrq, by=["PlayerId"])
    df["hrq_ev_avg"] = df["hrq_ev"] / df["hrq_n"]
    df["hrq_dist_avg"] = df["hrq_dist"] / df["hrq_n"]

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
    for pre in ["pc", "ps"]:
        df = pd.concat([df, _pit_rates(df, pre)], axis=1)
    df["p5_hr_bf"] = df["p5_HR"] / df["p5_BF"]
    df["p5_k_bf"] = df["p5_SO"] / df["p5_BF"]
    df["p_ip_per_start"] = df["ps_Outs"] / 3 / df["p_starts_season"]

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
    return df


def batter_feature_cols():
    cols = ["slot", "Home", "Season", "month", "days_rest",
            "g_career", "g_season",
            "c_PA", "c_hr_pa", "c_tb_ab", "c_h_ab", "c_k_pct", "c_bb_pct", "c_iso",
            "s_PA", "s_hr_pa", "s_tb_ab", "s_h_ab", "s_k_pct", "s_bb_pct", "s_iso"]
    for w in ROLL_WINDOWS:
        cols += [f"r{w}_PA", f"r{w}_hr_pa", f"r{w}_tb_ab", f"r{w}_k_pct"]
    cols += ["p_starts_career", "p_starts_season", "p_days_rest",
             "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_era",
             "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_era",
             "p5_hr_bf", "p5_k_bf", "p_ip_per_start",
             "toff_hr_pa", "toff_r_pg", "toff_k_pct", "toff_bb_pct",
             "pen_hr_bf", "pen_k_bf", "pen_era",
             "park_hr_pg", "LF", "CF", "RF", "Elevation_ft",
             "Temp", "WindSpeed",
             "hrq_n", "hrq_ev_avg", "hrq_dist_avg", "hrq_dist_max",
             *ARS_P_METRICS.values(), *ARS_B_METRICS.values(), "m_coverage",
             "bat_height", "bat_weight", "bat_age", "same_hand",
             "vsh_PA", "vsh_hr_pa_sh", "vsh_tb_ab_sh", "vsh_k_pct_sh",
             # NOTE: pull_fence/porch_margin and batter-side fatigue were
             # tried (iteration 3) and hurt the batter props on the holdout;
             # the league-environment lg_* columns (iteration 4) were flat
             # to slightly negative for every prop. All stay in the frames
             # but out of the batter models. Fatigue (p_np_*) and lg_* help
             # the K model and live in starts cols.
             *CAT_COLS]
    cols += SHRINK_COLS
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
    st["p_ip_per_start"] = st["ps_Outs"] / 3 / st["p_starts_season"]
    st = st.merge(_league_env_table(gb), on="Date", how="left")
    st["month"] = st["Date"].dt.month
    st["y_so"] = st["SO"]
    return st


def starts_feature_cols():
    return ["Season", "month", "Home", "p_starts_career", "p_starts_season",
            "p_days_rest", "p_np_last", "p_np_l3",
            "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_era",
            "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_era",
            "p5_hr_bf", "p5_k_bf", "p_ip_per_start",
            "vs_k_pct", "vs_bb_pct", "vs_hr_pa", "vs_r_pg",
            "park_hr_pg", "Elevation_ft", "Temp", "WindSpeed",
            "lg_k_pa", "lg_r_pa", "lg_hr_pa",
            "DayNight", "Condition", "WindDir"]


def build_game_frame(raw):
    """Training frame for game totals and the winner model: one row per game."""
    games, gb, gp = raw["games"], raw["gb"], raw["gp"]
    team_tab = _team_offense_table(gb)
    pen_tab = _bullpen_table(gp)
    park_tab = _park_table(gb, games)
    res_tab = _team_results_table(games)
    form_tab = _team_form_table(games)
    st = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]]
    stf = _starter_asof(gp)
    stf = pd.concat([stf, _pit_rates(stf, "ps"), _pit_rates(stf, "pc")], axis=1)
    stf["p5_k_bf"] = stf["p5_SO"] / stf["p5_BF"]

    g = games[~games["ShortGame"]].copy()
    away_sc = pd.to_numeric(g["AwayScore"], errors="coerce")
    home_sc = pd.to_numeric(g["HomeScore"], errors="coerce")
    g["total_runs"] = away_sc + home_sc
    g["y_home_win"] = np.where(away_sc.notna() & home_sc.notna(),
                               (home_sc > away_sc).astype(float), np.nan)

    ST_COLS = ["ps_era", "ps_k_bf", "ps_hr_bf", "ps_bb_bf",
               "pc_era", "pc_hr_bf", "p_days_rest", "p5_k_bf"]
    for side, team_col in [("away", "AwayTeam"), ("home", "HomeTeam")]:
        t = team_tab.rename(columns={"Team": team_col})
        t = t.rename(columns={c: f"{side}_{c}" for c in t.columns if c.startswith("cum")})
        g = _asof_merge(g, t, by=[team_col, "Season"])
        g[f"{side}_hr_pa"] = g[f"{side}_cum_HR"] / g[f"{side}_cum_PA"]
        g[f"{side}_r_pg"] = g[f"{side}_cum_R"] / g[f"{side}_cum_n"]
        g[f"{side}_k_pct"] = g[f"{side}_cum_SO"] / g[f"{side}_cum_PA"]
        p = pen_tab.rename(columns={"Team": team_col})
        p = p.rename(columns={c: f"{side}_pen_{c}" for c in p.columns if c.startswith("cum")})
        g = _asof_merge(g, p, by=[team_col, "Season"])
        g[f"{side}_pen_era"] = g[f"{side}_pen_cum_ER"] * 27 / g[f"{side}_pen_cum_Outs"]
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

    park = park_tab.rename(columns={"cum_HR": "park_cum_HR", "cum_n": "park_cum_n"})
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
            "opp_pen_era": gf[f"{opp}_pen_era"],
            "opp_ps_era": gf[f"{opp}_ps_era"],
            "opp_ps_k_bf": gf[f"{opp}_ps_k_bf"],
            "opp_ps_hr_bf": gf[f"{opp}_ps_hr_bf"],
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
            "opp_pen_era", "opp_ps_era", "opp_ps_k_bf", "opp_ps_hr_bf",
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
        tick("building team/park tables...")
        self.team_tab = _team_offense_table(r["gb"])
        self.pen_tab = _bullpen_table(r["gp"])
        self.park_tab = _park_table(r["gb"], r["games"])
        self.env_tab = _league_env_table(r["gb"])
        self.res_rows = _team_results_rows(r["games"])
        self.res_tab = _daily_cum(self.res_rows.drop(columns=["GamePk"]),
                                  ["Team", "Season"], ["W", "RF", "RA"])
        _, self.elo_hist = build_elo(r["games"])
        self.hrq = _hr_quality_table(r["hr"])
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

    def batter_feats(self, pid, date, season, opp_hand=None):
        hist = self.gb_by_player.get(pid)
        out = {}
        if hist is not None:
            h = hist[hist["Date"] < date]
        else:
            h = None
        out.update(self._vsh(h, opp_hand))
        zero = {k: 0 for k in BAT_STATS}
        if h is None or h.empty:
            for pre in ["c", "s"]:
                for k in ["PA", "hr_pa", "tb_ab", "h_ab", "k_pct", "bb_pct", "iso"]:
                    out[f"{pre}_{k}"] = np.nan
                out.update(shrunk_from_sums(zero, pre))  # 0 sums -> priors
            out.update(g_career=0, g_season=0, days_rest=np.nan)
            for w in ROLL_WINDOWS:
                for k in ["PA", "hr_pa", "tb_ab", "k_pct"]:
                    out[f"r{w}_{k}"] = np.nan
                out.update(shrunk_from_sums(zero, f"r{w}", roll=True))
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
                out.update(shrunk_from_sums(sums, pre))
            out["g_career"] = len(h)
            out["g_season"] = len(hs)
            out["days_rest"] = (date - h["Date"].iloc[-1]).days
            for w in ROLL_WINDOWS:
                tail = h.tail(w)[["PA", "HR", "TB", "SO"]].sum()
                out[f"r{w}_PA"] = tail["PA"]
                out[f"r{w}_hr_pa"] = tail["HR"] / tail["PA"] if tail["PA"] else np.nan
                out[f"r{w}_tb_ab"] = tail["TB"] / tail["PA"] if tail["PA"] else np.nan
                out[f"r{w}_k_pct"] = tail["SO"] / tail["PA"] if tail["PA"] else np.nan
                out.update(shrunk_from_sums(tail, f"r{w}", roll=True))

        q = self.hrq[(self.hrq["BatterId"] == pid) & (self.hrq["Date"] < date)]
        if len(q):
            q = q.iloc[-1]
            out["hrq_n"] = q["cum_n"]
            out["hrq_ev_avg"] = q["cum_ev"] / q["cum_n"]
            out["hrq_dist_avg"] = q["cum_dist"] / q["cum_n"]
            out["hrq_dist_max"] = q["cum_dist_max"]
        else:
            out.update(hrq_n=np.nan, hrq_ev_avg=np.nan,
                       hrq_dist_avg=np.nan, hrq_dist_max=np.nan)
        return out

    def starter_feats(self, pid, date, season, prefix=""):
        hist = self.starts_by_player.get(pid)
        out = {}
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            keys = ["p_starts_career", "p_starts_season", "p_days_rest",
                    "p_np_last", "p_np_l3",
                    "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_era",
                    "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_era",
                    "p5_hr_bf", "p5_k_bf", "p_ip_per_start"]
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
            out[f"{pre}_era"] = s["ER"] * 27 / s["Outs"] if s["Outs"] else np.nan
        t5 = h.tail(5)[["BF", "HR", "SO"]].sum()
        out["p5_hr_bf"] = t5["HR"] / t5["BF"] if t5["BF"] else np.nan
        out["p5_k_bf"] = t5["SO"] / t5["BF"] if t5["BF"] else np.nan
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
                    f"{prefix}_era": np.nan}
        return {f"{prefix}_hr_bf": row["cum_HR"] / row["cum_BF"],
                f"{prefix}_k_bf": row["cum_SO"] / row["cum_BF"],
                f"{prefix}_era": row["cum_ER"] * 27 / row["cum_Outs"]}

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
               "park_hr_pg": np.nan}
        if venue in self.parks.index:
            p = self.parks.loc[venue]
            out.update(LF=p["LF"], CF=p["CF"], RF=p["RF"],
                       Elevation_ft=p["Elevation_ft"])
        row = self._cum(self.park_tab, {"Venue": venue}, date)
        if row is not None and row["cum_n"] >= 30:
            out["park_hr_pg"] = row["cum_HR"] / row["cum_n"]
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
