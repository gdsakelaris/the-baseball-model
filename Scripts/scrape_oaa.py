"""Scrape team Outs Above Average (defense) per season from Baseball Savant.

OAA is Statcast's range-based defense metric: outs recorded above what an
average fielder converts from the same batted balls. It is the only direct
defense-quality measurement available to the model — the game logs'
unearned-run rate (opp_def_uer) is a weak proxy that misses everything a
bad defense turns into "earned" hits.

Savant keys teams by MLB team_id; this maps them to the per-season
abbreviations the game logs use (statsapi teams endpoint, which handles
renames like OAK -> ATH in 2025). OAA is also scaled to a per-162-game
figure so the 60-game 2020 season is comparable.

The model consumes these as PRIOR-season values (leakage-free, like GO/AO
and sprint speed): a game sees the opponent's previous-season defense.

One row per (Year, Team).

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network. If the current-season
fetch fails (Savant throttling), the previous run's rows for it are kept
and the job still succeeds — the model only ever consumes prior-season
OAA, so a one-day-stale current season costs nothing. --backfill forces
a full refetch of every season.

Usage:
    python scrape_oaa.py [-o output.csv] [--backfill]
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from seasons import CURRENT_SEASON, YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_oaa.csv"

API_URL = "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
SEASON_GAMES = {2020: 60}        # per-162 scaling; every other year is 162
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}


def team_abbrevs(year):
    """MLB team_id -> that season's abbreviation (statsapi, rename-aware)."""
    r = requests.get(TEAMS_URL, params={"sportId": 1, "season": year},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    return {t["id"]: t["abbreviation"] for t in r.json()["teams"]}


def fetch_year(year, tries=4):
    params = {"type": "Fielding_Team", "year": year, "team": "",
              "range": "year", "min": "q", "pos": "", "roles": "",
              "viz": "hide", "csv": "true"}
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=120)
            r.raise_for_status()
            # Savant throttling returns HTTP 200 with an HTML page; make
            # that a retryable error instead of a pandas parse crash
            if "outs_above_average" not in r.text[:2000]:
                raise ValueError("response is not the OAA CSV (throttled?)")
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 15 * 2 ** attempt                # 15s, 30s, 60s
            print(f"    retry {year} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()

    out_path = Path(args.output)
    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
    have = set() if stored is None else \
        set(pd.to_numeric(stored["Year"], errors="coerce").dropna().astype(int))

    cols = ["Year", "Team", "TeamId", "TeamName", "OAA", "OAA_per162"]
    frames = []
    for year in YEARS:
        if year != CURRENT_SEASON and year in have:
            rows = stored[stored["Year"] == year]
            frames.append(rows[cols])
            print(f"{year}: {len(rows):,} teams (stored)", flush=True)
            continue
        try:
            df = fetch_year(year)
        except Exception as e:                      # noqa: BLE001
            if year in have:
                rows = stored[stored["Year"] == year]
                frames.append(rows[cols])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(rows):,} rows (model uses "
                      f"prior-season OAA, so this costs nothing)",
                      flush=True)
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?", flush=True)
                continue
            sys.exit(f"{year}: FAILED ({e}) and no stored rows to fall "
                     f"back on — run --backfill once the source recovers")
        abbrev = team_abbrevs(year)
        games = SEASON_GAMES.get(year, 162)
        out = pd.DataFrame({
            "Year": year,
            "TeamId": pd.to_numeric(df["team_id"], errors="coerce"),
            "TeamName": df["team_name"],
            "OAA": pd.to_numeric(df["outs_above_average"], errors="coerce"),
        }).dropna(subset=["TeamId"])
        out["TeamId"] = out["TeamId"].astype("int64")
        out["Team"] = out["TeamId"].map(abbrev)
        out["OAA_per162"] = out["OAA"] * 162.0 / games
        missing = out[out["Team"].isna()]
        if len(missing):
            raise SystemExit(f"{year}: no abbreviation for team ids "
                             f"{missing['TeamId'].tolist()}")
        frames.append(out[cols])
        print(f"{year}: {len(out):,} teams", flush=True)
        time.sleep(1.0)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.drop_duplicates(["Year", "Team"], keep="last")
    all_rows = all_rows.sort_values(["Year", "Team"])
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    all_rows.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(all_rows):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
