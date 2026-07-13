"""Scrape pitch-level Statcast data, stored as DAILY PER-PLAYER AGGREGATES.

The batted-ball file (scrape_statcast.py) covers only balls put in play —
no whiffs, no called strikes, no chases. This pulls EVERY pitch 2020-2026
from the same statcast_search CSV endpoint, but ~700k pitches per season is
too much to keep raw, so each chunk is aggregated on arrival into one row
per player per day of sufficient statistics:

  both files:   pitches, swings, whiffs (total + out-of-zone), called
                strikes, zone/out-of-zone counts, graded velo-band buckets
                on FF/SI (fblo_ <92 / fbmid_ 92-95 / fb95_ 95+, each
                n/sw/wh), breaking + offspeed buckets (brk_/off_ n/sw/wh),
                edge-of-zone count (edge_n, shadow band 0.67-1.33 of
                scaled zone), first-pitch counts (fp_n seen, fp_sw swung,
                fp_s strike)
  pitcher file: + fastball velo sum/count (FF+SI)

From these, features can rebuild any rate (swinging-strike%, CSW%, whiff
per swing, chase%, zone%) as career or decay-weighted as-of values —
swing-and-miss and plate discipline are the fastest-stabilizing skills in
baseball, and fastball-velocity trend is the classic in-season decline
signal. oz_wh splits whiffs by zone, so IN-ZONE contact rate (the most
stable hit-tool skill: z_contact = 1 - (wh_n-oz_wh)/(sw_n-oz_sw)) is
derivable; the fb95 buckets give a batter's performance against elite
velocity, to cross with tonight's starter's fastball velo; the brk/off
buckets give whiff-vs-pitch-class splits to collide with the opposing
pitcher's usage mix; edge_n/n is a command proxy; fp_s/fp_n is
first-pitch-strike% (pitcher) and fp_sw/fp_n first-pitch aggression
(batter). None of this exists in the box scores or the batted-ball file.

--backfill also archives every raw pitch to Data/raw_pitches/
pitches_{year}.parquet (~all Savant detail columns, zstd), so future
schema changes re-aggregate from disk via --from-raw instead of paying
another 6-hour Savant download.

Relational keys: PlayerId is the MLBAM id used everywhere; one row per
(PlayerId, Date).

Default run is incremental — the output CSVs double as the cache: stored
seasons are reused; only the newest REFETCH_DAYS (Statcast back-corrects)
plus any missing season are downloaded. Seconds in the daily job.
--backfill re-downloads all seasons (~25-35 minutes).

Usage:
    python scrape_pitches.py [--outdir DIR] [--backfill] [--from-raw]
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
RAW_DIR = DATA_DIR / "raw_pitches"
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
ELITE_VELO = 95.0                # fb95 buckets: fastballs at/above this mph
BAND_LO = 92.0                   # graded velo bands: <92 / 92-95 / 95+
# Savant pitch-class grouping; cutters/others fall in the fastball remainder
BREAKING = {"SL", "ST", "SV", "CU", "KC", "CS", "SC", "KN"}
OFFSPEED = {"CH", "FS", "FO", "EP"}
# shadow band of the scaled zone (1.0 = edge; x incl. ball radius = 0.83 ft)
EDGE_X_HALF = 0.83
EDGE_LO, EDGE_HI = 0.67, 1.33

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
    whiff = desc.isin(WHIFFS)
    in_zone = zone.between(1, 9)
    out_zone = zone >= 11
    is_fb = raw["pitch_type"].isin(FASTBALLS)
    is_brk = raw["pitch_type"].isin(BREAKING)
    is_off = raw["pitch_type"].isin(OFFSPEED)
    velo = pd.to_numeric(raw["release_speed"], errors="coerce")
    is_fb95 = is_fb & (velo >= ELITE_VELO)
    is_fbmid = is_fb & (velo >= BAND_LO) & (velo < ELITE_VELO)
    is_fblo = is_fb & (velo < BAND_LO)
    balls = pd.to_numeric(raw["balls"], errors="coerce")
    strikes = pd.to_numeric(raw["strikes"], errors="coerce")
    first = (balls == 0) & (strikes == 0)
    two_k = strikes == 2
    full = (balls == 3) & (strikes == 2)
    is_ball = desc.isin(("ball", "blocked_ball"))
    # release-point coords (mechanical repeatability -> sum + sum-of-
    # squares so consumers can rebuild the scatter from cumulative sums)
    rx = pd.to_numeric(raw["release_pos_x"], errors="coerce")
    rz = pd.to_numeric(raw["release_pos_z"], errors="coerce")
    rp_ok = rx.notna() & rz.notna()
    # shadow-band location: scale so 1.0 = zone edge on each axis
    px = pd.to_numeric(raw["plate_x"], errors="coerce")
    pz = pd.to_numeric(raw["plate_z"], errors="coerce")
    top = pd.to_numeric(raw["sz_top"], errors="coerce")
    bot = pd.to_numeric(raw["sz_bot"], errors="coerce")
    x_sc = px.abs() / EDGE_X_HALF
    z_sc = (pz - (top + bot) / 2).abs() / ((top - bot) / 2).clip(lower=0.1)
    loc = pd.concat([x_sc, z_sc], axis=1).max(axis=1)
    edge = (loc > EDGE_LO) & (loc <= EDGE_HI)
    base = pd.DataFrame({
        "Date": pd.to_datetime(raw["game_date"]).dt.date,
        "PitcherId": pd.to_numeric(raw["pitcher"], errors="coerce"),
        "BatterId": pd.to_numeric(raw["batter"], errors="coerce"),
        "n": 1.0,
        "sw_n": swing.astype(float),
        "wh_n": whiff.astype(float),
        "cs_n": (desc == "called_strike").astype(float),
        "z_n": in_zone.astype(float),
        "oz_n": out_zone.astype(float),
        "oz_sw": (out_zone & swing).astype(float),
        "oz_wh": (out_zone & whiff).astype(float),
        "fb95_n": is_fb95.astype(float),
        "fb95_sw": (is_fb95 & swing).astype(float),
        "fb95_wh": (is_fb95 & whiff).astype(float),
        "fbmid_n": is_fbmid.astype(float),
        "fbmid_sw": (is_fbmid & swing).astype(float),
        "fbmid_wh": (is_fbmid & whiff).astype(float),
        "fblo_n": is_fblo.astype(float),
        "fblo_sw": (is_fblo & swing).astype(float),
        "fblo_wh": (is_fblo & whiff).astype(float),
        "brk_n": is_brk.astype(float),
        "brk_sw": (is_brk & swing).astype(float),
        "brk_wh": (is_brk & whiff).astype(float),
        "off_n": is_off.astype(float),
        "off_sw": (is_off & swing).astype(float),
        "off_wh": (is_off & whiff).astype(float),
        "edge_n": edge.astype(float),
        "fp_n": first.astype(float),
        "fp_sw": (first & swing).astype(float),
        "fp_s": (first & (swing | (desc == "called_strike"))).astype(float),
        "ts_n": two_k.astype(float),
        "ts_sw": (two_k & swing).astype(float),
        "ts_wh": (two_k & whiff).astype(float),
        "f32_n": full.astype(float),
        "f32_z": (full & in_zone).astype(float),
        "f32_b": (full & is_ball).astype(float),
        "f32_sw": (full & swing).astype(float),
        "f32_wh": (full & whiff).astype(float),
        "fb_n": (is_fb & velo.notna()).astype(float),
        "fb_v": velo.where(is_fb).fillna(0.0),
        "fb_v2": (velo ** 2).where(is_fb).fillna(0.0),
        "rp_n": rp_ok.astype(float),
        "rp_x": rx.where(rp_ok).fillna(0.0),
        "rp_x2": (rx ** 2).where(rp_ok).fillna(0.0),
        "rp_z": rz.where(rp_ok).fillna(0.0),
        "rp_z2": (rz ** 2).where(rp_ok).fillna(0.0),
    })
    shared = ["n", "sw_n", "wh_n", "cs_n", "z_n", "oz_n", "oz_sw", "oz_wh",
              "fb95_n", "fb95_sw", "fb95_wh",
              "fbmid_n", "fbmid_sw", "fbmid_wh",
              "fblo_n", "fblo_sw", "fblo_wh",
              "brk_n", "brk_sw", "brk_wh", "off_n", "off_sw", "off_wh",
              "edge_n", "fp_n", "fp_sw", "fp_s", "ts_n", "ts_sw", "ts_wh",
              "f32_n", "f32_z", "f32_b", "f32_sw", "f32_wh"]
    pit_stats = shared + ["fb_n", "fb_v", "fb_v2",
                          "rp_n", "rp_x", "rp_x2", "rp_z", "rp_z2"]
    bat_stats = shared
    pit = (base.dropna(subset=["PitcherId"])
           .astype({"PitcherId": "int64"})
           .groupby(["PitcherId", "Date"], as_index=False)[pit_stats].sum()
           .rename(columns={"PitcherId": "PlayerId"}))
    bat = (base.dropna(subset=["BatterId"])
           .astype({"BatterId": "int64"})
           .groupby(["BatterId", "Date"], as_index=False)[bat_stats].sum()
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


def write_raw(year, frames):
    """Archive one season of raw pitches so schema changes never need a
    re-download. Object columns are stringified (mixed-type chunks)."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).where(df[c].notna())
    path = RAW_DIR / f"pitches_{year}.parquet"
    try:
        df.to_parquet(path, index=False, compression="zstd")
    except Exception as e:                              # noqa: BLE001
        path = RAW_DIR / f"pitches_{year}.csv.gz"
        print(f"    parquet failed ({e}); falling back to csv.gz", flush=True)
        df.to_csv(path, index=False, compression="gzip")
    print(f"    raw archive: {len(df):,} pitches -> {path.name}", flush=True)


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
    ap.add_argument("--from-raw", action="store_true", dest="from_raw",
                    help="re-aggregate seasons from Data/raw_pitches archives "
                         "where present (implies --backfill for those years)")
    args = ap.parse_args()
    if args.from_raw:
        args.backfill = True
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
        raw_path = RAW_DIR / f"pitches_{year}.parquet"
        if args.from_raw and clip is None and raw_path.exists():
            pit, bat = aggregate(pd.read_parquet(raw_path))
            if pit is not None:
                pit_frames.append(pit)
                bat_frames.append(bat)
            print(f"{year}: {int(pit['n'].sum()) if pit is not None else 0:,}"
                  f" pitches (raw archive)", flush=True)
            continue
        got = 0
        raw_chunks = [] if args.backfill else None
        for d0, d1 in season_windows(year, clip):
            raw = fetch_range(d0, d1)
            if raw_chunks is not None and not raw.empty:
                raw_chunks.append(raw)
            pit, bat = aggregate(raw)
            if pit is not None:
                got += int(pit["n"].sum())
                pit_frames.append(pit)
                bat_frames.append(bat)
            time.sleep(SLEEP)
        if raw_chunks is not None:
            write_raw(year, raw_chunks)
            raw_chunks.clear()
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
