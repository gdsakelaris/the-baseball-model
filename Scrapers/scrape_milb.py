"""Scrape minor-league player season stats (hitting + pitching) from the
same data service that powers MLB.com (bdfed.stitch.mlbinfra.com), for the
MiLB-priors program: level-translated rates feed (a) the PA model's EB
prior for thin-MLB-history players and (b) main-model feature columns
riding the Phase-3 retrain batch.

One row per player-season-LEVEL (a player promoted mid-year gets one row
per level; teams within a level are aggregated by the service). Regular
season only. Age and league ride along — age-at-level is the strongest
translation covariate, league tags park/environment context (PCL vs INT).

Seasons start at MILB_FIRST_SEASON (2010, defined HERE — deliberately not
Scrapers/seasons.py: that file is code-fingerprinted for the daily guard
and its FIRST_SEASON drives the MLB frames, which is a different decision).
Players debuting in 2015 (the frame's first season) thus carry up to five
years of minors history. 2020 is skipped: the minor-league season was
canceled outright.

LEAKAGE NOTE for consumers: these are season AGGREGATES. A season's line
is only as-of-safe for MLB rows dated AFTER that season ended — join
season <= row_season - 1. Never feed a row its own in-progress MiLB
season from this file (the aggregate includes future games). In-season
as-of signal needs a date-grain scrape (byDateRange splits) — a designed
upgrade, not this file.

Completed seasons are served from the existing output CSVs (they never
change); only the current season hits the network, with the previous
run's rows as the stale fallback if that fetch fails. --backfill
refetches everything.

Usage:
    python scrape_milb.py [--backfill]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

from seasons import CURRENT_SEASON, stored_rows_by_season

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

API_URL = "https://bdfed.stitch.mlbinfra.com/bdfed/stats/player"
PAGE_SIZE = 2000
MILB_FIRST_SEASON = 2010

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# statsapi sportIds for the affiliated minors. 15 (short-season A) folded
# after 2020 and 16 covers the complex/rookie leagues (ACL/FCL/DSL) — both
# scraped anyway; the translation fitting decides what carries signal.
LEVELS = {11: "AAA", 12: "AA", 13: "A+", 14: "A", 15: "A-", 16: "Rk"}

META = {
    "Year": "year",
    "SportId": None,                 # filled by the scraper, not the API
    "Level": None,
    "League": "leagueName",
    "PlayerId": "playerId",
    "Name": "playerFullName",
    "Pos": "positionAbbrev",
    "Age": "age",
    "Team": "teamAbbrev",
    "TeamName": "teamName",
}

BAT_COLUMNS = {
    **META,
    "G": "gamesPlayed",
    "PA": "plateAppearances",
    "AB": "atBats",
    "R": "runs",
    "H": "hits",
    "2B": "doubles",
    "3B": "triples",
    "HR": "homeRuns",
    "RBI": "rbi",
    "BB": "baseOnBalls",
    "IBB": "intentionalWalks",
    "HBP": "hitByPitch",
    "SO": "strikeOuts",
    "SF": "sacFlies",
    "SAC": "sacBunts",
    "SB": "stolenBases",
    "CS": "caughtStealing",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
}

PIT_COLUMNS = {
    **META,
    "G": "gamesPlayed",
    "GS": "gamesStarted",
    "IP": "inningsPitched",
    "TBF": "battersFaced",
    "H": "hits",
    "2B": "doubles",
    "3B": "triples",
    "HR": "homeRuns",
    "R": "runs",
    "ER": "earnedRuns",
    "BB": "baseOnBalls",
    "IBB": "intentionalWalks",
    "HBP": "hitByPitch",
    "SO": "strikeOuts",
    "WP": "wildPitches",
    "BK": "balks",
    "ERA": "era",
    "WHIP": "whip",
}

GROUPS = {
    "hitting": (BAT_COLUMNS, "hits", DATA_DIR / "milb_batting.csv"),
    "pitching": (PIT_COLUMNS, "inningsPitched", DATA_DIR / "milb_pitching.csv"),
}


def fetch_level_season(session, group, sort_stat, year, sport_id):
    """Every player's aggregated line for one (level, season)."""
    rows = []
    offset = 0
    while True:
        params = {
            "stitch_env": "prod",
            "season": str(year),
            "sportId": str(sport_id),
            "stats": "season",
            "group": group,
            "gameType": "R",
            "playerPool": "ALL",
            "sortStat": sort_stat,
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


def fetch_season(session, group, columns, sort_stat, year):
    """One season across all levels -> output-shaped dicts."""
    out = []
    for sport_id, level in LEVELS.items():
        raw = fetch_level_season(session, group, sort_stat, year, sport_id)
        for r in raw:
            row = {col: r.get(field, "")
                   for col, field in columns.items() if field}
            row["SportId"] = sport_id
            row["Level"] = level
            out.append(row)
        time.sleep(0.5)  # be polite to MLB.com
    return out


def scrape_group(session, group, backfill):
    columns, sort_stat, out_path = GROUPS[group]
    stored = {} if backfill else stored_rows_by_season(out_path, "Year")

    all_rows = []
    for year in range(MILB_FIRST_SEASON, CURRENT_SEASON + 1):
        if year == 2020:             # minor-league season canceled
            continue
        if year != CURRENT_SEASON and year in stored:
            all_rows.extend(stored[year])
            print(f"{group} {year}: {len(stored[year])} rows (stored)")
            continue
        try:
            season_rows = fetch_season(session, group, columns, sort_stat,
                                       year)
        except Exception as e:
            if year in stored:
                all_rows.extend(stored[year])
                print(f"WARNING: {group} {year} fetch failed ({e}); keeping "
                      f"the previous run's {len(stored[year])} rows")
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {group} {year} fetch failed and no stored "
                      f"rows yet ({e}); season not started?")
                continue
            print(f"{group} {year}: FAILED ({e}) and no stored rows to fall "
                  f"back on — run --backfill once the source recovers",
                  file=sys.stderr)
            sys.exit(1)
        all_rows.extend(season_rows)
        print(f"{group} {year}: {len(season_rows)} rows")

    # utf-8-sig so Excel renders accented names (José, Díaz) correctly.
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} player-season-level rows to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)
    for group in GROUPS:
        scrape_group(session, group, args.backfill)


if __name__ == "__main__":
    main()
