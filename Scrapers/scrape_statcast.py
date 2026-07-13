"""Scrape Statcast batted-ball data (every ball in play) from Baseball Savant.

The HR log covers only home runs — a sample censored to each batter's best
contact. This pulls EVERY tracked ball in play 2020-2026 from the public
statcast_search CSV endpoint (the same service behind
baseballsavant.mlb.com/statcast_search): exit velocity, launch angle, barrel
classification, expected stats on contact (xBA/xwOBA), batted-ball type, and
hit distance, one row per batted ball.

Why it matters for the model: contact-quality ("process") stats stabilize
far faster than outcome stats — exit velo is reliable in ~40 batted balls vs
hundreds of AB for batting average — so they detect real skill and real
in-season change earlier than anything in the box scores. The pitcher side
(contact quality ALLOWED) is equally informative.

Relational keys: BatterId/PitcherId are MLBAM ids matching PlayerId in every
other CSV; GamePk matches the game logs. (GamePk, AtBat, PitchNum) uniquely
identifies a batted ball.

Default run is incremental — the output CSV doubles as the cache: stored
seasons are reused, and only dates after the newest stored row (minus a
2-day refetch window for Statcast's own corrections) plus any season missing
from the file are downloaded. Seconds in the daily job. --backfill ignores
the cache and rescrapes all seasons (~10 minutes, ~730k rows); only needed
when the file itself is suspect.

Usage:
    python scrape_statcast.py [-o output.csv] [--backfill]
"""

import argparse
import io
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from seasons import YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_statcast_bip.csv"

API_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# output column -> Savant CSV field
COLUMNS = {
    "GamePk": "game_pk",
    "Season": None,               # derived from Date
    "Date": "game_date",
    "BatterId": "batter",
    "PitcherId": "pitcher",
    "Stand": "stand",
    "PThrows": "p_throws",
    "Events": "events",
    "BBType": "bb_type",
    "ExitVelo": "launch_speed",
    "LaunchAngle": "launch_angle",
    "LSA": "launch_speed_angle",  # Savant contact code; 6 = barrel
    "xBA": "estimated_ba_using_speedangle",
    "xwOBA": "estimated_woba_using_speedangle",
    "HitDistance": "hit_distance_sc",
    "HcX": "hc_x",              # hit coordinates -> spray direction
    "HcY": "hc_y",
    "AtBat": "at_bat_number",
    "PitchNum": "pitch_number",
}
KEY = ["GamePk", "AtBat", "PitchNum"]

CHUNK_DAYS = 14        # ~10k rows in-season; well under Savant's result cap
CAP_ROWS = 24000       # a chunk this big was probably truncated -> split it
SLEEP = 2.0            # politeness between requests
REFETCH_DAYS = 2       # re-pull the newest days: Savant back-corrects data


def fetch_range(d0, d1, tries=3):
    """One CSV request for batted balls in [d0, d1] (regular season only).
    Returns a raw Savant DataFrame; recursively splits ranges that hit the
    result cap so nothing is silently truncated."""
    params = {
        "all": "true", "type": "details", "player_type": "batter",
        "hfBBT": "fly_ball|ground_ball|line_drive|popup|",
        "hfGT": "R|", "minors": "false",
        "game_date_gt": str(d0), "game_date_lt": str(d1),
    }
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=180)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            break
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    retry {d0}..{d1} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    if len(df) >= CAP_ROWS and d0 != d1:
        mid = d0 + (d1 - d0) / 2
        print(f"    {d0}..{d1}: {len(df):,} rows (cap?) -> splitting",
              flush=True)
        time.sleep(SLEEP)
        left = fetch_range(d0, mid)
        time.sleep(SLEEP)
        right = fetch_range(mid + timedelta(days=1), d1)
        return pd.concat([left, right], ignore_index=True)
    return df


def to_schema(raw):
    """Savant frame -> our column schema (empty frame if no rows)."""
    if raw.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    out = pd.DataFrame()
    for col, src in COLUMNS.items():
        if src is None:
            continue
        out[col] = raw[src] if src in raw.columns else pd.NA
    d = pd.to_datetime(out["Date"])
    out["Date"] = d.dt.date
    out["Season"] = d.dt.year
    return out[list(COLUMNS)]


def season_windows(year, start=None):
    """[(d0, d1), ...] CHUNK_DAYS-sized windows covering the season's
    possible dates (Mar 1 - Nov 30; empty off-season chunks are cheap),
    optionally clipped to begin at `start`."""
    d0 = date(year, 3, 1) if start is None else max(start, date(year, 3, 1))
    end = min(date(year, 11, 30), date.today())
    windows = []
    while d0 <= end:
        d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), end)
        windows.append((d0, d1))
        d0 = d1 + timedelta(days=1)
    return windows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="rescrape all seasons from scratch (default run is "
                         "incremental from the newest stored date)")
    args = ap.parse_args()
    out_path = Path(args.output)

    # The output CSV is the cache: stored rows are reused, so the default
    # run only refetches (a) the newest REFETCH_DAYS (Savant back-corrects
    # recent games) and (b) any whole season somehow missing from the file.
    # Completed seasons already stored are never re-downloaded; --backfill
    # is the escape hatch when the file itself is suspect.
    existing = None
    start = None
    have = set()
    if out_path.exists() and not args.backfill:
        existing = pd.read_csv(out_path, encoding="utf-8-sig",
                               low_memory=False)
        existing["Date"] = pd.to_datetime(existing["Date"]).dt.date
        have = set(pd.to_numeric(existing["Season"],
                                 errors="coerce").dropna().astype(int))
        newest = existing["Date"].max()
        start = newest - timedelta(days=REFETCH_DAYS)
        existing = existing[existing["Date"] < start]
        print(f"incremental: {len(existing):,} rows kept, refetching from "
              f"{start}", flush=True)

    frames = []
    for year in YEARS:
        if start is not None and year < start.year and year in have:
            continue                    # completed season already stored
        clip = start if start is not None and year >= start.year else None
        windows = season_windows(year, clip)
        got = 0
        for d0, d1 in windows:
            raw = fetch_range(d0, d1)
            df = to_schema(raw)
            got += len(df)
            if len(df):
                frames.append(df)
            time.sleep(SLEEP)
        print(f"{year}: {got:,} batted balls", flush=True)

    new = pd.concat(frames, ignore_index=True) if frames else \
        pd.DataFrame(columns=list(COLUMNS))
    if existing is not None:
        new = pd.concat([existing, new], ignore_index=True)
    n0 = len(new)
    new = new.drop_duplicates(KEY, keep="last")
    if len(new) < n0:
        print(f"dropped {n0 - len(new):,} duplicate rows on {KEY}", flush=True)
    new = new.sort_values(["Date", "GamePk", "AtBat", "PitchNum"])
    out_path.parent.mkdir(exist_ok=True)
    new.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(new):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
