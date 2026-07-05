"""Scrape MLB player pitching stats (standard + expanded) from MLB.com.

Pulls season-level pitching stats for 2020-2026 from the same data service
that powers https://www.mlb.com/stats/pitching (bdfed.stitch.mlbinfra.com) and
writes one combined CSV. Regular season, all teams, all positions, all
players, no splits. Players who changed teams mid-season get a single
aggregated row for the year, attributed to their most recent team.

Relational keys shared with the other CSVs:
  - PlayerId: MLB's stable player ID (also in the roster and batting files)
  - TeamName: full team name matching the roster file's Team column
    (Team holds MLB's abbreviation, e.g. PHI)

Usage:
    python scrape_pitching_stats.py [-o output.csv]
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
                    default=str(DATA_DIR / "mlb_pitching_stats_2020_2026.csv"))
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

    # utf-8-sig so Excel renders accented names (Sánchez, Berríos) correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} player-season rows to {args.output}")


if __name__ == "__main__":
    main()
