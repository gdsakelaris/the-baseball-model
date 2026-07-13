"""Scrape Statcast sprint speed per player per season from Baseball Savant.

Sprint speed (ft/s over a player's fastest one-second window, competitive
runs only) is the most direct raw-speed measurement available — the primary
skill behind stolen bases and taking the extra base, neither of which the
box-score rates isolate from opportunity. hp_to_1b (home-to-first time, s)
is the complementary burst measure.

The model consumes these as PRIOR-season values (same leakage-free pattern
as GO/AO): a game sees the previous season's measurement, falling back one
more year. Sprint speed is among the most stable year-to-year skills, so
the lag costs little.

Relational keys: PlayerId is the MLBAM id used everywhere; one row per
(Year, PlayerId).

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network, and a failed
current-season fetch falls back to the previous run's rows (the model
only consumes prior-season values). --backfill forces a full refetch.

Usage:
    python scrape_sprint_speed.py [-o output.csv] [--backfill]
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
DEFAULT_OUT = DATA_DIR / "mlb_sprint_speed.csv"

API_URL = "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
MIN_RUNS = 5                     # competitive-run floor; low to catch bench
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}


def fetch_year(year, tries=4):
    params = {"year": year, "position": "", "team": "", "min": MIN_RUNS,
              "csv": "true"}
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=120)
            r.raise_for_status()
            # Savant throttling returns HTTP 200 with an HTML page; make
            # that a retryable error instead of a pandas parse crash
            if "sprint_speed" not in r.text[:2000]:
                raise ValueError("response is not the sprint CSV "
                                 "(throttled?)")
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 15 * 2 ** attempt                # 15s, 30s, 60s
            print(f"    retry {year} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every season, ignoring stored rows")
    args = ap.parse_args()

    out_path = Path(args.output)
    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
    have = set() if stored is None else \
        set(pd.to_numeric(stored["Year"], errors="coerce").dropna().astype(int))

    cols = ["Year", "PlayerId", "Name", "Team", "CompetitiveRuns",
            "SprintSpeed", "HPto1B"]
    frames = []
    for year in YEARS:
        if year != CURRENT_SEASON and year in have:
            rows = stored[stored["Year"] == year]
            frames.append(rows[cols])
            print(f"{year}: {len(rows):,} players (stored)", flush=True)
            continue
        try:
            df = fetch_year(year)
        except Exception as e:                      # noqa: BLE001
            if year in have:
                rows = stored[stored["Year"] == year]
                frames.append(rows[cols])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(rows):,} rows (model uses "
                      f"prior-season sprint speed, so this costs nothing)",
                      flush=True)
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?", flush=True)
                continue
            sys.exit(f"{year}: FAILED ({e}) and no stored rows to fall "
                     f"back on — run --backfill once the source recovers")
        out = pd.DataFrame({
            "Year": year,
            "PlayerId": pd.to_numeric(df["player_id"], errors="coerce"),
            "Name": df["last_name, first_name"],
            "Team": df["team"],
            "CompetitiveRuns": pd.to_numeric(df["competitive_runs"],
                                             errors="coerce"),
            "SprintSpeed": pd.to_numeric(df["sprint_speed"],
                                         errors="coerce"),
            "HPto1B": pd.to_numeric(df["hp_to_1b"], errors="coerce"),
        }).dropna(subset=["PlayerId"])
        out["PlayerId"] = out["PlayerId"].astype("int64")
        frames.append(out[cols])
        print(f"{year}: {len(out):,} players", flush=True)
        time.sleep(1.0)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.drop_duplicates(["Year", "PlayerId"], keep="last")
    all_rows = all_rows.sort_values(["Year", "PlayerId"])
    out_path.parent.mkdir(exist_ok=True)
    all_rows.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(all_rows):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
