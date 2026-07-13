"""Scrape Statcast baserunning run value per season from Baseball Savant.

Runner runs measure everything a runner does on the bases in run-value
terms: extra-base advancement (going first-to-third, scoring from second,
tagging up — runner_runs_XB) plus basestealing value (runner_runs_SBX).
Sprint speed says how FAST a player is; this says how much his running
actually PRODUCES — the advancement component is the skill behind scoring
runs that raw speed only proxies.

The leaderboard starts in 2016 and lists qualified runners (~190/season);
players below the opportunity floor are simply absent and the model treats
them as league-average. The model consumes these as PRIOR-season values
(leakage-free, like sprint speed and OAA).

One row per (Year, PlayerId).

Completed seasons are served from the existing output CSV (they never
change); only the current season hits the network; a failed current-season
fetch keeps the previous run's rows and the job still succeeds. --backfill
forces a full refetch of every season.

Usage:
    python scrape_baserunning.py [-o output.csv] [--backfill]
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
DEFAULT_OUT = DATA_DIR / "mlb_baserunning.csv"

API_URL = "https://baseballsavant.mlb.com/leaderboard/baserunning-run-value"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
COLS = ["Year", "PlayerId", "Name", "RunnerRuns", "RunnerRunsXB",
        "RunnerRunsSB", "Opportunities"]


def fetch_year(year, tries=4):
    params = {"game_type": "Regular", "season_start": year,
              "season_end": year, "csv": "true"}
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=120)
            r.raise_for_status()
            # Savant throttling returns HTTP 200 with an HTML page; make
            # that a retryable error instead of a pandas parse crash
            if "runner_runs" not in r.text[:2000]:
                raise ValueError("response is not the runner-runs CSV "
                                 "(throttled?)")
            return pd.read_csv(io.StringIO(r.text))
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 15 * 2 ** attempt                # 15s, 30s, 60s
            print(f"    retry {year} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def build(year, df):
    out = pd.DataFrame({
        "Year": year,
        "PlayerId": pd.to_numeric(df["player_id"], errors="coerce"),
        "Name": df["entity_name"],
        "RunnerRuns": pd.to_numeric(df["runner_runs_tot"], errors="coerce"),
        "RunnerRunsXB": pd.to_numeric(df["runner_runs_XB"], errors="coerce"),
        "RunnerRunsSB": pd.to_numeric(df["runner_runs_SBX"], errors="coerce"),
        "Opportunities": pd.to_numeric(df["N_runner_moved"], errors="coerce"),
    }).dropna(subset=["PlayerId"])
    out["PlayerId"] = out["PlayerId"].astype("int64")
    return out


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

    frames = []
    for year in YEARS:
        if year != CURRENT_SEASON and year in have:
            rows = stored[stored["Year"] == year]
            frames.append(rows[COLS])
            print(f"{year}: {len(rows):,} runners (stored)", flush=True)
            continue
        try:
            df = fetch_year(year)
        except Exception as e:                      # noqa: BLE001
            if year in have:
                rows = stored[stored["Year"] == year]
                frames.append(rows[COLS])
                print(f"WARNING: {year} fetch failed ({e}); keeping the "
                      f"previous run's {len(rows):,} rows (model uses "
                      f"prior-season values, so this costs nothing)",
                      flush=True)
                continue
            if year == CURRENT_SEASON:
                print(f"WARNING: {year} fetch failed and no stored rows yet "
                      f"({e}); season not started?", flush=True)
                continue
            sys.exit(f"{year}: FAILED ({e}) and no stored rows to fall "
                     f"back on — run --backfill once the source recovers")
        if df.empty:
            print(f"{year}: no rows at source (leaderboard starts 2016)",
                  flush=True)
            continue
        frames.append(build(year, df)[COLS])
        print(f"{year}: {len(frames[-1]):,} runners", flush=True)
        time.sleep(1.0)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = (all_rows.drop_duplicates(["Year", "PlayerId"], keep="last")
                .sort_values(["Year", "PlayerId"]))
    out_path.parent.mkdir(exist_ok=True)
    all_rows.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(all_rows):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
