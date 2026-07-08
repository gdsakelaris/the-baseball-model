"""Scrape pitch-level Statcast data, stored as DAILY PER-PLAYER AGGREGATES.

The batted-ball file (scrape_statcast.py) covers only balls put in play —
no whiffs, no called strikes, no chases. This pulls EVERY pitch 2020-2026
from the same statcast_search CSV endpoint, but ~700k pitches per season is
too much to keep raw, so each chunk is aggregated on arrival into one row
per player per day of sufficient statistics:

  pitcher file: pitches, swings, whiffs, called strikes, zone/out-of-zone
                counts, chases induced, fastball velo sum/count (FF+SI)
  batter file:  the same seen/against counts (no velo)

From these, features can rebuild any rate (swinging-strike%, CSW%, whiff
per swing, chase%, zone%) as career or decay-weighted as-of values —
swing-and-miss and plate discipline are the fastest-stabilizing skills in
baseball, and fastball-velocity trend is the classic in-season decline
signal. None of this exists in the box scores or the batted-ball file.

Relational keys: PlayerId is the MLBAM id used everywhere; one row per
(PlayerId, Date).

Default run is incremental — the output CSVs double as the cache: stored
seasons are reused; only the newest REFETCH_DAYS (Statcast back-corrects)
plus any missing season are downloaded. Seconds in the daily job.
--backfill re-downloads all seasons (~25-35 minutes).

Usage:
    python scrape_pitches.py [--outdir DIR] [--backfill]
"""

import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from seasons import YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUT_PITCHERS = "mlb_pitch_daily_pitchers.csv"
OUT_BATTERS = "mlb_pitch_daily_batters.csv"

API_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# description buckets (verified against live data 2026-07)
SWINGS = {"foul", "foul_tip", "hit_into_play", "swinging_strike",
          "swinging_strike_blocked", "foul_bunt", "missed_bunt",
          "bunt_foul_tip"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FASTBALLS = {"FF", "SI"}         # velo trend tracked on four-seam + sinker

CHUNK_DAYS = 4                   # ~18k pitches in-season; cap guard splits
CAP_ROWS = 24000
SLEEP = 2.0
REFETCH_DAYS = 2


def fetch_range(d0, d1, tries=3):
    """All pitches in [d0, d1] (regular season). Splits on the result cap."""
    params = {
        "all": "true", "type": "details", "player_type": "batter",
        "hfGT": "R|", "minors": "false",
        "game_date_gt": str(d0), "game_date_lt": str(d1),
    }
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=240)
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


def aggregate(raw):
    """One Savant chunk -> (pitcher day rows, batter day rows)."""
    if raw.empty:
        return None, None
    desc = raw["description"].astype(str)
    zone = pd.to_numeric(raw["zone"], errors="coerce")
    swing = desc.isin(SWINGS)
    in_zone = zone.between(1, 9)
    out_zone = zone >= 11
    is_fb = raw["pitch_type"].isin(FASTBALLS)
    velo = pd.to_numeric(raw["release_speed"], errors="coerce")
    base = pd.DataFrame({
        "Date": pd.to_datetime(raw["game_date"]).dt.date,
        "PitcherId": pd.to_numeric(raw["pitcher"], errors="coerce"),
        "BatterId": pd.to_numeric(raw["batter"], errors="coerce"),
        "n": 1.0,
        "sw_n": swing.astype(float),
        "wh_n": desc.isin(WHIFFS).astype(float),
        "cs_n": (desc == "called_strike").astype(float),
        "z_n": in_zone.astype(float),
        "oz_n": out_zone.astype(float),
        "oz_sw": (out_zone & swing).astype(float),
        "fb_n": (is_fb & velo.notna()).astype(float),
        "fb_v": velo.where(is_fb).fillna(0.0),
    })
    stats = ["n", "sw_n", "wh_n", "cs_n", "z_n", "oz_n", "oz_sw",
             "fb_n", "fb_v"]
    pit = (base.dropna(subset=["PitcherId"])
           .astype({"PitcherId": "int64"})
           .groupby(["PitcherId", "Date"], as_index=False)[stats].sum()
           .rename(columns={"PitcherId": "PlayerId"}))
    bat = (base.dropna(subset=["BatterId"])
           .astype({"BatterId": "int64"})
           .groupby(["BatterId", "Date"], as_index=False)[stats[:-2]].sum()
           .rename(columns={"BatterId": "PlayerId"}))
    return pit, bat


def season_windows(year, start=None):
    d0 = date(year, 3, 1) if start is None else max(start, date(year, 3, 1))
    end = min(date(year, 11, 30), date.today())
    windows = []
    while d0 <= end:
        d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), end)
        windows.append((d0, d1))
        d0 = d1 + timedelta(days=1)
    return windows


def load_existing(path, backfill):
    if backfill or not path.exists():
        return None, None, set()
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    have = set(pd.to_datetime(df["Date"]).map(lambda d: d.year))
    newest = df["Date"].max()
    start = newest - timedelta(days=REFETCH_DAYS)
    return df[df["Date"] < start], start, have


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(DATA_DIR))
    ap.add_argument("--backfill", action="store_true",
                    help="re-download all seasons (default is incremental)")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    p_path, b_path = outdir / OUT_PITCHERS, outdir / OUT_BATTERS

    # the pitcher file drives incrementality; both files are written together
    kept_p, start, have = load_existing(p_path, args.backfill)
    kept_b, _, _ = load_existing(b_path, args.backfill)
    if start is not None:
        print(f"incremental: {len(kept_p):,} pitcher-day rows kept, "
              f"refetching from {start}", flush=True)

    pit_frames, bat_frames = [], []
    for year in YEARS:
        if start is not None and year < start.year and year in have:
            continue
        clip = start if start is not None and year >= start.year else None
        got = 0
        for d0, d1 in season_windows(year, clip):
            raw = fetch_range(d0, d1)
            pit, bat = aggregate(raw)
            if pit is not None:
                got += int(pit["n"].sum())
                pit_frames.append(pit)
                bat_frames.append(bat)
            time.sleep(SLEEP)
        print(f"{year}: {got:,} pitches", flush=True)

    def finish(kept, frames, path, key=("PlayerId", "Date")):
        new = pd.concat(frames, ignore_index=True) if frames else None
        if new is not None and kept is not None:
            new = pd.concat([kept, new], ignore_index=True)
        elif new is None:
            new = kept
        new = (new.drop_duplicates(list(key), keep="last")
               .sort_values(list(key)))
        path.parent.mkdir(exist_ok=True)
        new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {len(new):,} rows -> {path}", flush=True)

    finish(kept_p, pit_frames, p_path)
    finish(kept_b, bat_frames, b_path)


if __name__ == "__main__":
    main()
