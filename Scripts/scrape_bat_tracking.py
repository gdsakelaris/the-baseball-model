"""Scrape Statcast bat-tracking (bat speed, swing length) per season from
Baseball Savant.

Hawkeye bat tracking measures the swing itself: average bat speed (mph),
swing length (ft), hard-swing rate, squared-up and "blast" rates, and a
per-swing run value. Bat speed is a direct, stable power signal — it speaks
to the total-bases and home-run props in a way the batted-ball outcome
stats (which only see balls put in play) do not.

The model would consume these as PRIOR-season values (leakage-free, like
sprint speed / GO-AO / OAA): a game sees the batter's previous-season
swing profile.

COVERAGE CAVEAT: bat tracking only exists from 2023 on. As a prior-season
feature that leaves every pre-2024 game without a value, so it is NOT yet
wired into the models — under the selection suite (train <=2023) it would
be NaN across the entire training set and thus inert / unevaluable. This
scraper BANKS the data (and the daily job keeps it current) so the full
history is ready the moment the training window rolls forward enough to
make it testable (~2027+). See the queue notes; do not wire it into
features.py until the selection suite carries covered seasons.

One row per (Year, PlayerId). Completed seasons are served from the stored
CSV; only the current season hits the network. --backfill forces a full
refetch.

Usage:
    python scrape_bat_tracking.py [-o output.csv] [--backfill]
"""

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from seasons import CURRENT_SEASON, YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_bat_tracking.csv"

API_URL = "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
FIRST_TRACKED = 2023            # bat tracking begins this season
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
# Savant CSV column -> our column
FIELDS = {
    "id": "PlayerId", "name": "Name",
    "swings_competitive": "Swings",
    "avg_bat_speed": "BatSpeed",
    "hard_swing_rate": "HardSwingRate",
    "swing_length": "SwingLength",
    "squared_up_per_swing": "SquaredUpPerSwing",
    "blast_per_swing": "BlastPerSwing",
    "batter_run_value": "BatterRunValue",
}
COLS = ["Year", *FIELDS.values()]


def fetch_year(year, tries=4):
    params = {"minSwings": "q", "minGroupSwings": "1",
              "seasonStart": year, "seasonEnd": year, "type": "batter",
              "csv": "true"}
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=120)
            r.raise_for_status()
            if "html" in r.text[:200].lower():      # throttle / error page
                raise ValueError("response is not the CSV (throttled?)")
            df = pd.read_csv(io.StringIO(r.text))
            if "avg_bat_speed" not in df.columns:
                raise ValueError("no bat-tracking columns in response")
            return df
        except Exception as e:                       # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 15 * 2 ** attempt
            print(f"    retry {year} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every tracked season, ignoring stored rows")
    args = ap.parse_args()

    years = [y for y in YEARS if y >= FIRST_TRACKED]
    out_path = Path(args.output)
    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
    have = set() if stored is None else \
        set(pd.to_numeric(stored["Year"], errors="coerce").dropna().astype(int))

    frames = []
    for year in years:
        if year != CURRENT_SEASON and year in have:
            rows = stored[stored["Year"] == year]
            frames.append(rows[COLS])
            print(f"{year}: {len(rows):,} batters (stored)", flush=True)
            continue
        try:
            df = fetch_year(year)
        except Exception as e:                       # noqa: BLE001
            if year in have:
                rows = stored[stored["Year"] == year]
                frames.append(rows[COLS])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(rows):,} rows", flush=True)
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?", flush=True)
                continue
            sys.exit(f"{year}: FAILED ({e}) and no stored rows to fall back "
                     f"on — run --backfill once the source recovers")
        out = df[list(FIELDS)].rename(columns=FIELDS)
        out.insert(0, "Year", year)
        out["PlayerId"] = pd.to_numeric(out["PlayerId"], errors="coerce")
        out = out.dropna(subset=["PlayerId"])
        out["PlayerId"] = out["PlayerId"].astype("int64")
        frames.append(out[COLS])
        print(f"{year}: {len(out):,} batters", flush=True)
        time.sleep(1.0)

    if not frames:
        print("no seasons in range (bat tracking starts 2023)", flush=True)
        return
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows.drop_duplicates(["Year", "PlayerId"], keep="last")
    allrows = allrows.sort_values(["Year", "PlayerId"])
    out_path.parent.mkdir(exist_ok=True)
    allrows.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(allrows):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
