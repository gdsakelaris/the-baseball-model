"""Scrape MLB player batting stats (standard + expanded) from MLB.com.

Pulls season-level hitting stats for 2020-2026 from the same data service
that powers https://www.mlb.com/stats/ (bdfed.stitch.mlbinfra.com) and writes
one combined CSV. Regular season, all teams, all positions, all players, no
splits. Players who changed teams mid-season get a single aggregated row for
the year, attributed to their most recent team.

Relational keys shared with mlb_rosters_2026.csv:
  - PlayerId: MLB's stable player ID (also in the roster file)
  - TeamName: full team name matching the roster file's Team column
    (Team holds MLB's abbreviation, e.g. PHI)

Usage:
    python scrape_batting_stats.py [-o output.csv]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

API_URL = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
YEARS = range(2020, 2027)
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
    "G": "gamesPlayed",
    "AB": "atBats",
    "R": "runs",
    "H": "hits",
    "2B": "doubles",
    "3B": "triples",
    "HR": "homeRuns",
    "RBI": "rbi",
    "BB": "baseOnBalls",
    "SO": "strikeOuts",
    "SB": "stolenBases",
    "CS": "caughtStealing",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    # Expanded
    "PA": "plateAppearances",
    "HBP": "hitByPitch",
    "SAC": "sacBunts",
    "SF": "sacFlies",
    "GIDP": "gidp",
    "GO/AO": "groundOutsToAirouts",
    "XBH": "extraBaseHits",
    "TB": "totalBases",
    "IBB": "intentionalWalks",
    "BABIP": "babip",
    "ISO": "iso",
    "AB/HR": "atBatsPerHomeRun",
    "BB/K": "walksPerStrikeout",
    "BB%": "walksPerPlateAppearance",
    "K%": "strikeoutsPerPlateAppearance",
}


def fetch_season(session, year):
    """Return every player's aggregated hitting line for one season."""
    rows = []
    offset = 0
    while True:
        params = {
            "stitch_env": "prod",
            "season": str(year),
            "sportId": "1",          # MLB
            "stats": "season",
            "group": "hitting",
            "gameType": "R",         # regular season
            "playerPool": "ALL",     # every player, not just qualified
            "sortStat": "hits",
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
                    default=str(DATA_DIR / "mlb_batting_stats_2020_2026.csv"))
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    for year in YEARS:
        try:
            season_rows = fetch_season(session, year)
        except Exception as e:
            print(f"{year}: FAILED ({e})", file=sys.stderr)
            sys.exit(1)
        for r in season_rows:
            all_rows.append({col: r.get(field, "") for col, field in COLUMNS.items()})
        print(f"{year}: {len(season_rows)} players")
        time.sleep(1)  # be polite to MLB.com

    # utf-8-sig so Excel renders accented names (José, Díaz) correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} player-season rows to {args.output}")


if __name__ == "__main__":
    main()
