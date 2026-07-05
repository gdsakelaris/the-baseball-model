"""Scrape every MLB home run from 2020-2026 from onlyhomers.com.

The OnlyHomers database page (https://www.onlyhomers.com/database) is fed by
a JSON API that returns a full season per request; this pulls each year and
writes one combined CSV in chronological order (2020's first homer through
2026's latest). Running Total numbers every homer sequentially across all
years; Total is the site's running count within a season; HR is that
batter's season home run number; ROB is runners on base (source nulls,
meaning solo shots in older seasons, are written as 0).

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
    python scrape_homeruns.py [-o output.csv]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

API_URL = "https://zuriteapi.com/homers/api/homeruns"
YEARS = range(2020, 2027)

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
                    default=str(DATA_DIR / "mlb_homeruns_2020_2026.csv"))
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    for year in YEARS:
        try:
            resp = session.get(
                API_URL, params={"year": year, "format": "json"}, timeout=120
            )
            resp.raise_for_status()
            homers = resp.json()
            if not isinstance(homers, list) or not homers:
                raise ValueError("empty or unexpected response")
        except Exception as e:
            print(f"{year}: FAILED ({e})", file=sys.stderr)
            sys.exit(1)
        # The API returns each season newest-first; reverse it so the file
        # runs chronologically and the running total can count forward.
        for h in reversed(homers):
            row = {col: h.get(field, "") for col, field in COLUMNS.items() if field}
            row["Year"] = year
            row["Running Total"] = len(all_rows) + 1
            row["ROB"] = row["ROB"] or "0"
            row["Ballpark"] = VENUE_ALIASES.get(row["Ballpark"], row["Ballpark"])
            all_rows.append(row)
        print(f"{year}: {len(homers)} home runs")
        time.sleep(1)  # be polite

    # utf-8-sig so Excel renders accented names (Domínguez, García) correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} home runs to {args.output}")


if __name__ == "__main__":
    main()
