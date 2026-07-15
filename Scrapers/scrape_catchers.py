"""Scrape catcher defense per season from Baseball Savant — framing,
throwing (caught-stealing value, pop time, exchange, arm), and the pop-time
leaderboard — rolled up to a TEAM battery view.

Three leaderboards, one row per (Year, PlayerId) in the player file:

  catcher-framing (2015+):  called pitches, framing runs (rv_tot), strike
                            rate on takes (pct_tot)
  catcher-throwing (2016+): SB attempts against, caught-stealing above
                            average (season total + per throw), realized
                            CS rate, pop time, exchange time, arm strength
  poptime (2015+):          pop time to 2B + attempt counts + max-effort
                            arm — fills the 2015 gap and thin seasons

Team file (mlb_catchers_team.csv): playing-time-weighted battery quality
per (Year, Team) — framing runs per 2000 called pitches (framing is a
volume stat), CS-above-average per attempt, and attempt-weighted pop time.
This is the serving-safe grain: the model cannot know tonight's starting
catcher, but the team's weighted battery is known before the game — the
same dodge mlb_oaa.csv uses for team defense. Team mapping: the throwing
leaderboard's team_name (2016+), the poptime leaderboard's team_id (2015),
via the statsapi teams endpoint (rename-aware, like scrape_oaa).

The model consumes these as PRIOR-season values (leakage-free, like OAA
and sprint speed). Catcher framing/arm are among the most stable
year-to-year defensive skills, so the lag costs little.

Completed seasons are served from the existing output CSVs; only the
current season hits the network, and a failed current-season fetch keeps
the previous run's rows. --backfill forces a full refetch.

Usage:
    python scrape_catchers.py [--backfill]
"""

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from seasons import CURRENT_SEASON, YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUT_PLAYERS = DATA_DIR / "mlb_catchers.csv"
OUT_TEAM = DATA_DIR / "mlb_catchers_team.csv"

FRAMING_URL = "https://baseballsavant.mlb.com/leaderboard/catcher-framing"
THROWING_URL = "https://baseballsavant.mlb.com/leaderboard/catcher-throwing"
POPTIME_URL = "https://baseballsavant.mlb.com/leaderboard/poptime"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
FRAME_PER = 2000.0               # framing runs per 2000 called pitches

PLAYER_COLS = ["Year", "PlayerId", "Name", "Team", "Pitches", "FrameRV",
               "StrikePct", "SBAtt", "CSAA", "CSAAThrow", "CSRate",
               "PopTime", "Exchange", "ArmStrength"]
TEAM_COLS = ["Year", "Team", "Pitches", "FrameRV", "FrameRV_pt",
             "SBAtt", "CSAA", "CSAA_att", "PopTime", "ArmStrength"]


def _fetch_csv(url, params, marker, tries=4):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=120)
            r.raise_for_status()
            # Savant throttling returns HTTP 200 with an HTML page; make
            # that a retryable error instead of a pandas parse crash
            if "<html" in r.text[:200].lower() or marker not in r.text[:2000]:
                raise ValueError(f"response is not the {marker} CSV "
                                 "(throttled?)")
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 15 * 2 ** attempt                # 15s, 30s, 60s
            print(f"    retry in {wait}s ({e})", flush=True)
            time.sleep(wait)


def _teams(season):
    r = requests.get(TEAMS_URL, params={"sportId": 1, "season": season},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()["teams"]


def team_maps(year):
    """(savant-label -> that season's abbrev, team_id -> abbrev). The
    throwing leaderboard labels teams with their CURRENT franchise abbrev
    even for old seasons (2016 shows ATH, not OAK), so the map goes
    current-abbrev -> franchise id -> that season's abbreviation
    (statsapi; team ids are stable through renames)."""
    by_id = {t["id"]: t["abbreviation"] for t in _teams(year)}
    by_name = {}
    for t in _teams(CURRENT_SEASON):
        ab = by_id.get(t["id"])
        if ab is None:
            continue
        by_name[t["abbreviation"]] = ab
        for k in ("name", "teamName", "clubName", "shortName"):
            if t.get(k):
                by_name[t[k]] = ab
    return by_name, by_id


def fetch_year(year):
    """Player-level catcher table for one season (throwing may be empty
    pre-2016; poptime fills pop/arm there)."""
    fr = _fetch_csv(FRAMING_URL,
                    {"type": "catcher", "seasonStart": year,
                     "seasonEnd": year, "team": "", "minPitches": 1,
                     "csv": "true"},
                    "pitches")
    time.sleep(1.0)
    th = _fetch_csv(THROWING_URL,
                    {"game_type": "Regular", "n": 1, "season_start": year,
                     "season_end": year, "split": "no", "team": "",
                     "type": "Cat", "with_team_only": 1, "csv": "true"},
                    "player_id")
    time.sleep(1.0)
    pt = _fetch_csv(POPTIME_URL,
                    {"year": year, "team": "", "min2b": 1, "min3b": 0,
                     "csv": "true"},
                    "entity_id")
    by_name, by_id = team_maps(year)

    out = pd.DataFrame({
        "Year": year,
        "PlayerId": pd.to_numeric(fr["id"], errors="coerce"),
        "Name": fr["name"],
        "Pitches": pd.to_numeric(fr["pitches"], errors="coerce"),
        "FrameRV": pd.to_numeric(fr["rv_tot"], errors="coerce"),
        "StrikePct": pd.to_numeric(fr["pct_tot"], errors="coerce"),
    }).dropna(subset=["PlayerId"])
    out["PlayerId"] = out["PlayerId"].astype("int64")

    if len(th):
        t = pd.DataFrame({
            "PlayerId": pd.to_numeric(th["player_id"], errors="coerce"),
            "Team": th["team_name"].map(by_name),
            "SBAtt": pd.to_numeric(th["sb_attempts"], errors="coerce"),
            "CSAA": pd.to_numeric(th["caught_stealing_above_average"],
                                  errors="coerce"),
            "CSAAThrow": pd.to_numeric(th["cs_aa_per_throw"],
                                       errors="coerce"),
            "CSRate": pd.to_numeric(th["rate_cs"], errors="coerce"),
            "PopTime": pd.to_numeric(th["pop_time"], errors="coerce"),
            "Exchange": pd.to_numeric(th["exchange_time"], errors="coerce"),
            "ArmStrength": pd.to_numeric(th["arm_strength"], errors="coerce"),
        }).dropna(subset=["PlayerId"])
        t["PlayerId"] = t["PlayerId"].astype("int64")
        unmapped = t.loc[t["Team"].isna()]
        if len(unmapped):
            raise SystemExit(f"{year}: unmapped throwing team names "
                             f"{th.loc[unmapped.index, 'team_name'].unique()}")
        out = out.merge(t, on="PlayerId", how="outer")
    else:
        for c in ("Team", "SBAtt", "CSAA", "CSAAThrow", "CSRate", "PopTime",
                  "Exchange", "ArmStrength"):
            out[c] = np.nan

    # poptime fallback: fills Team / PopTime / SBAtt / arm where the
    # throwing leaderboard has no row (all of 2015, thin seasons after)
    p = pd.DataFrame({
        "PlayerId": pd.to_numeric(pt["entity_id"], errors="coerce"),
        "_team": pd.to_numeric(pt["team_id"], errors="coerce").map(by_id),
        "_pop": pd.to_numeric(pt["pop_2b_sba"], errors="coerce"),
        "_att": pd.to_numeric(pt["pop_2b_sba_count"], errors="coerce"),
        "_arm": pd.to_numeric(pt["maxeff_arm_2b_3b_sba"], errors="coerce"),
    }).dropna(subset=["PlayerId"])
    p["PlayerId"] = p["PlayerId"].astype("int64")
    out = out.merge(p, on="PlayerId", how="left")
    out["Team"] = out["Team"].fillna(out["_team"])
    out["PopTime"] = out["PopTime"].fillna(out["_pop"])
    out["SBAtt"] = out["SBAtt"].fillna(out["_att"])
    out["ArmStrength"] = out["ArmStrength"].fillna(out["_arm"])
    out = out.drop(columns=["_team", "_pop", "_att", "_arm"])
    out["Year"] = year
    return out[PLAYER_COLS]


def team_rollup(players):
    """Playing-time-weighted battery quality per (Year, Team). Framing
    weights by called pitches; the throwing metrics weight by attempts.
    Catchers without a team mapping (no throwing/poptime row — negligible
    playing time) drop out of the weighting."""
    p = players.dropna(subset=["Team"]).copy()
    for c in ("Pitches", "FrameRV", "SBAtt", "CSAA"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["_pop_w"] = p["PopTime"] * p["SBAtt"]
    p["_arm_w"] = p["ArmStrength"] * p["SBAtt"]
    p["_att_pop"] = p["SBAtt"].where(p["PopTime"].notna())
    p["_att_arm"] = p["SBAtt"].where(p["ArmStrength"].notna())
    # min_count=1 keeps all-NaN groups NaN (a plain sum would turn the
    # 2015 CSAA — no throwing leaderboard that year — into a fake 0.0)
    _sum = lambda s: s.sum(min_count=1)                 # noqa: E731
    g = p.groupby(["Year", "Team"], as_index=False).agg(
        Pitches=("Pitches", _sum), FrameRV=("FrameRV", _sum),
        SBAtt=("SBAtt", _sum), CSAA=("CSAA", _sum),
        _pop_w=("_pop_w", _sum), _att_pop=("_att_pop", _sum),
        _arm_w=("_arm_w", _sum), _att_arm=("_att_arm", _sum))
    g["FrameRV_pt"] = FRAME_PER * g["FrameRV"] / g["Pitches"].replace(0, np.nan)
    g["CSAA_att"] = g["CSAA"] / g["SBAtt"].replace(0, np.nan)
    g["PopTime"] = g["_pop_w"] / g["_att_pop"].replace(0, np.nan)
    g["ArmStrength"] = g["_arm_w"] / g["_att_arm"].replace(0, np.nan)
    return g[TEAM_COLS]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()

    stored = None
    if OUT_PLAYERS.exists() and not args.backfill:
        stored = pd.read_csv(OUT_PLAYERS, encoding="utf-8-sig")
    have = set() if stored is None else \
        set(pd.to_numeric(stored["Year"], errors="coerce").dropna().astype(int))

    frames = []
    for year in YEARS:
        if year != CURRENT_SEASON and year in have:
            rows = stored[stored["Year"] == year]
            frames.append(rows[PLAYER_COLS])
            print(f"{year}: {len(rows):,} catchers (stored)", flush=True)
            continue
        try:
            df = fetch_year(year)
        except Exception as e:                      # noqa: BLE001
            if year in have:
                rows = stored[stored["Year"] == year]
                frames.append(rows[PLAYER_COLS])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(rows):,} rows (model uses "
                      f"prior-season values, so this costs nothing)",
                      flush=True)
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?", flush=True)
                continue
            sys.exit(f"{year}: FAILED ({e}) and no stored rows to fall "
                     f"back on — run --backfill once the source recovers")
        frames.append(df)
        print(f"{year}: {len(df):,} catchers", flush=True)
        time.sleep(1.0)

    players = pd.concat(frames, ignore_index=True)
    players = (players.drop_duplicates(["Year", "PlayerId"], keep="last")
               .sort_values(["Year", "PlayerId"]))
    OUT_PLAYERS.parent.mkdir(exist_ok=True)
    players.to_csv(OUT_PLAYERS, index=False, encoding="utf-8-sig")
    print(f"wrote {len(players):,} rows -> {OUT_PLAYERS}", flush=True)

    team = team_rollup(players).sort_values(["Year", "Team"])
    team.to_csv(OUT_TEAM, index=False, encoding="utf-8-sig")
    print(f"wrote {len(team):,} rows -> {OUT_TEAM}", flush=True)


if __name__ == "__main__":
    main()
