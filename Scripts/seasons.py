"""Single source of truth for which MLB seasons the project covers.

Scrapers import YEARS / CURRENT_SEASON from here instead of hardcoding a
range, so the annual rollover (2026 -> 2027 -> ...) needs no code edits:
from March 1 the new calendar year becomes the current season and every
scraper starts fetching it alongside the stored history.

The model side deliberately does NOT import this. train.py derives its
train/calibration/holdout years from the seasons actually present in the
data (which flips only once real new-season games accrue), and
features.py / predict.py are already season-relative (prior-season
lookups use Season-1, serving uses the game date's year).

Also provides the stored-season cache helper the per-year scrapers share:
a completed season's stats never change once scraped, so a scraper only
needs to hit the network for the current season. That both speeds the
daily job and removes the failure mode where an upstream hiccup on a
HISTORICAL fetch (e.g. Savant throttling the 2021 OAA page, 2026-07-07)
kills the whole job and blocks the retrain.
"""

import csv
from datetime import date
from pathlib import Path

FIRST_SEASON = 2020


def current_season(today=None):
    """The season scrapers treat as in progress.

    January/February belong to the PREVIOUS season — no games have been
    played and the new year's leaderboards are empty. From March 1 the
    new season exists for scraping purposes (Opening Day is late March;
    empty early fetches are cheap and harmless)."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def years(today=None):
    """Every season the project covers, oldest first."""
    return range(FIRST_SEASON, current_season(today) + 1)


YEARS = years()
CURRENT_SEASON = current_season()


def stored_rows_by_season(csv_path, season_col):
    """Rows of an existing combined CSV grouped {season: [dict, ...]}, or
    {} if the file doesn't exist. Completed seasons are served straight
    from this cache; the current season's rows are the stale fallback
    when its fetch fails (better a day-old snapshot than a dead job)."""
    path = Path(csv_path)
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                yr = int(row[season_col])
            except (KeyError, TypeError, ValueError):
                continue
            out.setdefault(yr, []).append(row)
    return out
