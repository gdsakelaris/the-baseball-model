"""Scrape Statcast pitch arsenal stats from Baseball Savant.

Pulls https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats for
every covered season (Scrapers/seasons.py) via the leaderboard's CSV export
and writes one combined CSV.
One row per player per pitch type per season. Minimum PA is set to 1 so
every row the leaderboard tracks is included (the site defaults to 10).

--type pitcher (default) is the arsenal view: how each pitcher's pitches
perform. --type batter is the same leaderboard's batter view: how each
batter fares against each pitch type; there, % is the share of pitches the
batter saw of that type, and positive run values favor the batter.

The site's "Rk." column is just the row number of the current sort, so it
is not reproduced here.

Relational keys shared with the other CSVs:
  - PlayerId: MLB's stable player ID (PlayerId in the roster/stats files)
  - Team: MLB team abbreviation (Team in the stats files)

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network, and a failed
current-season fetch falls back to the previous run's rows (the model
consumes arsenals as prior-season values). --backfill forces a full
refetch of every season.

Usage:
    python scrape_pitch_arsenals.py [--type pitcher|batter] [-o output.csv]
                                    [--backfill]
"""

import argparse
import csv
import io
import sys
import time
from pathlib import Path

import requests

from seasons import CURRENT_SEASON, YEARS, stored_rows_by_season

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

API_URL = "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"

# Savant's pitch-arsenal-stats leaderboard has no data before 2017 (probed
# 2026-07-09: 2015/2016 return 0 rows, 2017 returns 3,063). Earlier years
# are legitimately absent, not an outage, so they're skipped rather than
# treated as a fatal empty fetch.
SOURCE_FLOOR = 2017

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# CSV column -> savant export field, in output order.
COLUMNS = {
    "Year": None,  # derived from the request year
    "PlayerId": "player_id",
    "Player": "last_name, first_name",
    "Team": "team_name_alt",
    "PitchType": "pitch_type",
    "Pitch": "pitch_name",
    "RV/100": "run_value_per_100",
    "Run Value": "run_value",
    "Pitches": "pitches",
    "%": "pitch_usage",
    "PA": "pa",
    "BA": "ba",
    "SLG": "slg",
    "wOBA": "woba",
    "Whiff %": "whiff_percent",
    "K%": "k_percent",
    "Put Away %": "put_away",
    "xBA": "est_ba",
    "xSLG": "est_slg",
    "xwOBA": "est_woba",
    "Hard Hit %": "hard_hit_percent",
}


def fetch_year(session, year, player_type):
    params = {
        "type": player_type,
        "pitchType": "",
        "year": str(year),
        "team": "",
        "min": "1",       # include every row; site default hides PA < 10
        "csv": "true",
    }
    resp = session.get(API_URL, params=params, timeout=120)
    resp.raise_for_status()
    # utf-8-sig: savant's export starts with a BOM that would otherwise
    # corrupt the first header name.
    text = resp.content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--type", choices=["pitcher", "batter"], default="pitcher",
                    dest="player_type")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()
    if args.output is None:
        suffix = "" if args.player_type == "pitcher" else "_batters"
        args.output = str(DATA_DIR / f"mlb_pitch_arsenals{suffix}.csv")

    session = requests.Session()
    session.headers.update(HEADERS)

    stored = {} if args.backfill else stored_rows_by_season(args.output, "Year")

    all_rows = []
    for year in YEARS:
        if year < SOURCE_FLOOR:
            print(f"{year}: skipped (source leaderboard starts "
                  f"{SOURCE_FLOOR})")
            continue
        if year != CURRENT_SEASON and year in stored:
            all_rows.extend(stored[year])
            print(f"{year}: {len(stored[year])} {args.player_type}-pitch "
                  f"rows (stored)")
            continue
        try:
            year_rows = fetch_year(session, year, args.player_type)
            if not year_rows:
                raise ValueError("empty response")
        except Exception as e:
            if year in stored:
                all_rows.extend(stored[year])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(stored[year])} rows (model uses "
                      f"prior-season arsenals, so this costs nothing)")
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?")
                continue
            print(f"{year}: FAILED ({e}) and no stored rows to fall back "
                  f"on — run --backfill once the source recovers",
                  file=sys.stderr)
            sys.exit(1)
        for r in year_rows:
            row = {col: r.get(field, "") for col, field in COLUMNS.items() if field}
            row["Year"] = year
            all_rows.append(row)
        print(f"{year}: {len(year_rows)} {args.player_type}-pitch rows")
        time.sleep(2)  # be polite to baseballsavant.mlb.com

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
