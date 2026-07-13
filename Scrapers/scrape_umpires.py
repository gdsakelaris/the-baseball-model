"""Scrape the home-plate umpire for every game from the MLB StatsAPI.

The home-plate umpire calls the strike zone, and umpires have measurable,
persistent tendencies in how many strikeouts and walks their zones produce
(a tight zone inflates walks, a generous one inflates strikeouts). This is
the only zone-authority signal available to the model; it speaks directly
to the strikeout and walk props (the tier-1/2 markets).

Source: statsapi `/api/v1/game/{GamePk}/boxscore`, whose `officials` list
carries each crew member and position. We keep only the Home Plate ump
(the others don't touch the zone). Coverage is clean back to 2020.

One row per GamePk. The game universe is `mlb_games.csv` (the authoritative
list of played games, produced by scrape_gamelogs_3F.py, which sorts before
this script so its output is fresh). Games already in the output CSV are
cached — only new gamePks (the current season's fresh games) hit the
network, so completed seasons never refetch and an upstream hiccup can't
re-scrape six years of history. --backfill forces a full refetch.

The model consumes this as-of and leakage-free: a game sees only the HP
ump's tendency over his PRIOR games (features.py `_ump_asof`).

Usage:
    python scrape_umpires.py [-o output.csv] [--backfill] [--limit N]
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_umpires.csv"
GAMES_CSV = DATA_DIR / "mlb_games.csv"

BOX_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
COLS = ["GamePk", "Date", "Season", "HpUmpId", "HpUmp"]


def hp_umpire(pk, tries=3):
    """(id, name) of the home-plate ump for one game, or (NaN, '') if the
    boxscore has no HP official (rare — data gaps on old postponements)."""
    for attempt in range(tries):
        try:
            r = requests.get(BOX_URL.format(pk=pk), headers=HEADERS, timeout=30)
            r.raise_for_status()
            for o in r.json().get("officials", []):
                if o.get("officialType") == "Home Plate":
                    off = o.get("official", {})
                    return off.get("id"), off.get("fullName", "")
            return float("nan"), ""      # boxscore present, no HP listed
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
        pd.to_numeric(stored["GamePk"], errors="coerce").dropna().astype("int64"))

    todo = games[~games["GamePk"].isin(have)]
    print(f"{len(games):,} games in universe; {len(have):,} cached; "
          f"{len(todo):,} to fetch", flush=True)

    fetched, fail = [], 0
    for i, g in enumerate(todo.itertuples(index=False)):
        if args.limit and i >= args.limit:
            print(f"--limit {args.limit} reached; stopping", flush=True)
            break
        try:
            uid, name = hp_umpire(g.GamePk)
        except Exception as e:                       # noqa: BLE001
            fail += 1
            print(f"  WARNING: {g.GamePk} failed ({e}); skipping (retried "
                  f"next run)", flush=True)
            continue
        fetched.append({"GamePk": g.GamePk, "Date": g.Date, "Season": g.Season,
                        "HpUmpId": uid, "HpUmp": name})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1:,}/{len(todo):,} fetched", flush=True)
        time.sleep(args.sleep)

    parts = [df for df in (stored, pd.DataFrame(fetched, columns=COLS))
             if df is not None and len(df)]
    if not parts:
        print("nothing to write", flush=True)
        return
    allrows = pd.concat(parts, ignore_index=True)[COLS]
    allrows["GamePk"] = pd.to_numeric(allrows["GamePk"], errors="coerce")
    allrows = (allrows.dropna(subset=["GamePk"])
               .drop_duplicates("GamePk", keep="last")
               .sort_values("GamePk"))
    allrows["GamePk"] = allrows["GamePk"].astype("int64")
    out_path.parent.mkdir(exist_ok=True)
    allrows.to_csv(out_path, index=False, encoding="utf-8-sig")
    miss = int(allrows["HpUmpId"].isna().sum())
    print(f"wrote {len(allrows):,} rows -> {out_path} "
          f"({miss:,} without an HP ump; {fail:,} fetch failures this run)",
          flush=True)


if __name__ == "__main__":
    main()
