"""Scrape per-inning linescores for every game from the MLB StatsAPI.

The First-5-innings (F5) markets — F5 moneyline and F5 totals — grade on
runs through five innings, which `mlb_games.csv` (final scores only) cannot
provide. This accumulates that grading history so the PA-sim's F5 outputs
(backlog #H2) can be sold once the sim blend has earned trust: the market
stays parked, the DATA accrues (G5 decision, 2026-07-14).

Source: statsapi `/api/v1/game/{GamePk}/linescore`, whose `innings` list
carries away/home runs per inning. Stored LONG — one row per (GamePk,
Inning) — so any partial-game market (F3, F5, F7) can be graded from the
same file.

The game universe is `mlb_games.csv` (the authoritative list of played
games). Games already in the output CSV are cached — only new gamePks hit
the network, so completed seasons never refetch. --backfill forces a full
refetch.

Usage:
    python scrape_linescores.py [-o output.csv] [--backfill] [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_linescores.csv"
GAMES_CSV = DATA_DIR / "mlb_games.csv"

LS_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/linescore"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
COLS = ["GamePk", "Date", "Season", "Inning", "AwayRuns", "HomeRuns"]


def linescore(pk, tries=3):
    """[(inning, away runs, home runs), ...] for one game; [] when the
    endpoint has no innings (very old data gaps)."""
    for attempt in range(tries):
        try:
            r = requests.get(LS_URL.format(pk=pk), headers=HEADERS,
                             timeout=30)
            r.raise_for_status()
            out = []
            for inn in r.json().get("innings", []):
                out.append((inn.get("num"),
                            inn.get("away", {}).get("runs"),
                            inn.get("home", {}).get("runs")))
            return out
        except Exception as e:           # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 5 * 2 ** attempt      # 5s, 10s
            print(f"    retry {pk} in {wait}s ({e})", flush=True)
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every game, ignoring the cache")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N network fetches (smoke testing)")
    ap.add_argument("--sleep", type=float, default=0.12,
                    help="politeness delay between fetches (seconds)")
    args = ap.parse_args()

    if not GAMES_CSV.exists():
        sys.exit(f"{GAMES_CSV} not found — run scrape_gamelogs_3F.py first")
    games = pd.read_csv(GAMES_CSV, encoding="utf-8-sig",
                        usecols=["GamePk", "Date", "Season"])
    games = games.drop_duplicates("GamePk").sort_values("GamePk")

    out_path = Path(args.output)
    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
    have = set() if stored is None else set(
        pd.to_numeric(stored["GamePk"], errors="coerce").dropna()
        .astype("int64"))

    todo = games[~games["GamePk"].isin(have)]
    print(f"{len(games):,} games in universe; {len(have):,} cached; "
          f"{len(todo):,} to fetch", flush=True)

    fetched, fail = [], 0
    for i, g in enumerate(todo.itertuples(index=False)):
        if args.limit and i >= args.limit:
            print(f"--limit {args.limit} reached; stopping", flush=True)
            break
        try:
            innings = linescore(g.GamePk)
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  WARNING: {g.GamePk} failed ({e}); skipping (retried "
                  f"next run)", flush=True)
            continue
        for num, away, home in innings:
            fetched.append({"GamePk": g.GamePk, "Date": g.Date,
                            "Season": g.Season, "Inning": num,
                            "AwayRuns": away, "HomeRuns": home})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1:,}/{len(todo):,} fetched", flush=True)
        time.sleep(args.sleep)

    parts = [df for df in (stored, pd.DataFrame(fetched, columns=COLS))
             if df is not None and len(df)]
    if not parts:
        print("nothing to write", flush=True)
        return
    allrows = pd.concat(parts, ignore_index=True)[COLS]
    for c in ("GamePk", "Inning"):
        allrows[c] = pd.to_numeric(allrows[c], errors="coerce")
    allrows = (allrows.dropna(subset=["GamePk", "Inning"])
               .drop_duplicates(["GamePk", "Inning"], keep="last")
               .sort_values(["GamePk", "Inning"]))
    allrows["GamePk"] = allrows["GamePk"].astype("int64")
    allrows["Inning"] = allrows["Inning"].astype("int64")
    out_path.parent.mkdir(exist_ok=True)
    allrows.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(allrows):,} rows ({allrows['GamePk'].nunique():,} "
          f"games) -> {out_path} ({fail:,} fetch failures this run)",
          flush=True)


if __name__ == "__main__":
    main()
