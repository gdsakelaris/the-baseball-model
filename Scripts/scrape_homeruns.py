"""Scrape every MLB home run since 2020 from onlyhomers.com.

The OnlyHomers database page (https://www.onlyhomers.com/database) is fed by
a JSON API that returns a full season per request; this pulls each covered
year (Scripts/seasons.py) and writes one combined CSV in chronological order
(2020's first homer through the current season's latest). Running Total
numbers every homer sequentially across all years; Total is the site's
running count within a season; HR is that batter's season home run number;
ROB is runners on base (source nulls, meaning solo shots in older seasons,
are written as 0).

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network, with the previous run's
rows as the stale fallback if that fetch fails (validate_data.py's
freshness check catches a fallback that persists for days). --backfill
refetches every season.

Known quirks in the source database, kept as-is to stay faithful to the site:
  - 2020 includes the postseason (66 homers dated after 2020-09-27, played at
    neutral sites), and its regular-season coverage is a few short of MLB's
    official total; other years are regular season and match official counts.
  - 2024 includes two All-Star Futures Game homers (Team NLF vs ALF).
  - One 2021 row (Total 5645) has the batter's name in the BatterId field.

Relational keys shared with the other CSVs:
  - BatterId: MLB's stable player ID (PlayerId in the roster/stats files)
  - Team / PitcherTeam: MLB team abbreviations (Team in the stats files)
  - Ballpark: normalized to the canonical park names used by the game-log
    and ballparks CSVs (the source uses stale names like Marlins Park and
    Minute Maid Park); former and special-event venues (e.g. Oakland
    Coliseum, Field of Dreams, London Stadium) have no ballparks row, and a
    few source rows say "Unknown".

Usage:
    python scrape_homeruns.py [-o output.csv] [--backfill]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

from seasons import CURRENT_SEASON, YEARS, stored_rows_by_season

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

API_URL = "https://zuriteapi.com/homers/api/homeruns"

# One canonical name per physical park across all years and all CSVs.
# Keep in sync with scrape_gamelogs.py and build_ballparks.py.
VENUE_ALIASES = {
    "Minute Maid Park": "Daikin Park",
    "Guaranteed Rate Field": "Rate Field",
    "Marlins Park": "loanDepot Park",
    "loanDepot park": "loanDepot Park",
    "UNIQLO Field at Dodger Stadium": "Dodger Stadium",
    "Camden Yards": "Oriole Park at Camden Yards",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# CSV column -> API field, in output order.
COLUMNS = {
    "Year": None,           # derived from the request year
    "Running Total": None,  # sequential number across all years, oldest first
    "Total": "hr_count",
    "Team": "team_batting",
    "BatterId": "batter",
    "Batter": "batter_name",
    "HR": "player_hr_num",
    "ROB": "on_base_total",
    "Inning": "inning",
    "Outs": "outs",
    "Angle": "hit_angle",
    "Exit Velo": "hit_speed",
    "Distance": "hit_distance",
    "Pitch": "pitch_name",
    "Pitcher": "pitcher_name",
    "PitcherTeam": "team_fielding",
    "Ballpark": "venue",
    "Date": "date",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output",
                    default=str(DATA_DIR / "mlb_homeruns.csv"))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    stored = {} if args.backfill else stored_rows_by_season(args.output, "Year")

    all_rows = []
    for year in YEARS:
        if year != CURRENT_SEASON and year in stored:
            all_rows.extend(stored[year])
            print(f"{year}: {len(stored[year])} home runs (stored)")
            continue
        try:
            resp = session.get(
                API_URL, params={"year": year, "format": "json"}, timeout=120
            )
            resp.raise_for_status()
            homers = resp.json()
            if not isinstance(homers, list) or not homers:
                raise ValueError("empty or unexpected response")
        except Exception as e:
            if year in stored:
                all_rows.extend(stored[year])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(stored[year])} rows (one day "
                      f"stale at most; the freshness check catches worse)")
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?")
                continue
            print(f"{year}: FAILED ({e}) and no stored rows to fall back "
                  f"on — run --backfill once the source recovers",
                  file=sys.stderr)
            sys.exit(1)
        # The API returns each season newest-first; reverse it so the file
        # runs chronologically and the running total can count forward.
        for h in reversed(homers):
            row = {col: h.get(field, "") for col, field in COLUMNS.items() if field}
            row["Year"] = year
            row["ROB"] = row["ROB"] or "0"
            row["Ballpark"] = VENUE_ALIASES.get(row["Ballpark"], row["Ballpark"])
            all_rows.append(row)
        print(f"{year}: {len(homers)} home runs")
        time.sleep(1)  # be polite

    # sequential across all years regardless of which came from the cache
    for i, row in enumerate(all_rows, start=1):
        row["Running Total"] = i

    # utf-8-sig so Excel renders accented names (Domínguez, García) correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} home runs to {args.output}")


if __name__ == "__main__":
    main()
