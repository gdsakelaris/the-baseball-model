"""Scrape MLB player pitching stats (standard + expanded) from MLB.com.

Pulls season-level pitching stats for every covered season (Scripts/
seasons.py) from the same data service that powers
https://www.mlb.com/stats/pitching (bdfed.stitch.mlbinfra.com) and
writes one combined CSV. Regular season, all teams, all positions, all
players, no splits. Players who changed teams mid-season get a single
aggregated row for the year, attributed to their most recent team.

Relational keys shared with the other CSVs:
  - PlayerId: MLB's stable player ID (also in the roster and batting files)
  - TeamName: full team name matching the roster file's Team column
    (Team holds MLB's abbreviation, e.g. PHI)

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network, with the previous run's
rows as the stale fallback if that fetch fails. --backfill refetches all.

Usage:
    python scrape_pitching_stats.py [-o output.csv] [--backfill]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

from seasons import CURRENT_SEASON, YEARS, stored_rows_by_season

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

API_URL = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
PAGE_SIZE = 1000

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# CSV column -> API field, in output order.
# Standard stats match MLB.com's "Standard" tab, expanded match "Expanded".
COLUMNS = {
    "Year": "year",
    "PlayerId": "playerId",
    "Name": "playerFullName",
    "Pos": "positionAbbrev",
    "Team": "teamAbbrev",
    "TeamName": "teamName",
    # Standard
    "W": "wins",
    "L": "losses",
    "ERA": "era",
    "G": "gamesPlayed",
    "GS": "gamesStarted",
    "CG": "completeGames",
    "SHO": "shutouts",
    "SV": "saves",
    "SVO": "saveOpportunities",
    "IP": "inningsPitched",
    "H": "hits",
    "R": "runs",
    "ER": "earnedRuns",
    "HR": "homeRuns",
    "HB": "hitBatsmen",
    "BB": "baseOnBalls",
    "SO": "strikeOuts",
    "WHIP": "whip",
    "AVG": "avg",
    # Expanded
    "TBF": "battersFaced",
    "NP": "numberOfPitches",
    "P/IP": "pitchesPerInning",
    "QS": "qualityStarts",
    "GF": "gamesFinished",
    "HLD": "holds",
    "IBB": "intentionalWalks",
    "WP": "wildPitches",
    "BK": "balks",
    "GDP": "gidp",
    "GO/AO": "groundOutsToAirouts",
    "SO/9": "strikeoutsPer9Inn",
    "BB/9": "walksPer9Inn",
    "K/BB": "strikeoutWalkRatio",
    "BABIP": "babip",
    "SB": "stolenBases",
    "CS": "caughtStealing",
    "PK": "pickoffs",
}


def fetch_season(session, year):
    """Return every player's aggregated pitching line for one season."""
    rows = []
    offset = 0
    while True:
        params = {
            "stitch_env": "prod",
            "season": str(year),
            "sportId": "1",          # MLB
            "stats": "season",
            "group": "pitching",
            "gameType": "R",         # regular season
            "playerPool": "ALL",     # every player, not just qualified
            "sortStat": "inningsPitched",
            "order": "desc",
            "limit": str(PAGE_SIZE),
            "offset": str(offset),
        }
        resp = session.get(API_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("stats", [])
        rows.extend(batch)
        total = data.get("totalSplits", len(rows))
        offset += len(batch)
        if not batch or offset >= total:
            return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output",
                    default=str(DATA_DIR / "mlb_pitching_stats.csv"))
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
            print(f"{year}: {len(stored[year])} players (stored)")
            continue
        try:
            season_rows = fetch_season(session, year)
        except Exception as e:
            if year in stored:
                all_rows.extend(stored[year])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(stored[year])} rows")
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?")
                continue
            print(f"{year}: FAILED ({e}) and no stored rows to fall back "
                  f"on — run --backfill once the source recovers",
                  file=sys.stderr)
            sys.exit(1)
        for r in season_rows:
            all_rows.append({col: r.get(field, "") for col, field in COLUMNS.items()})
        print(f"{year}: {len(season_rows)} players")
        time.sleep(1)  # be polite to MLB.com

    # utf-8-sig so Excel renders accented names (Sánchez, Berríos) correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} player-season rows to {args.output}")


if __name__ == "__main__":
    main()
