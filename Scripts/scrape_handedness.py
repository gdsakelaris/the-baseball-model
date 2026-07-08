"""Fetch bats/throws handedness for every player in the game logs.

The roster file only covers currently rostered players, which leaves
handedness (platoon features) missing for most historical batter-games.
This pulls batSide/pitchHand from the MLB Stats API for every PlayerId
appearing in the game logs or rosters (~35 batched requests) and writes
Data/mlb_handedness.csv: PlayerId, Name, Bats, Throws.

Usage:
    python Scripts/scrape_handedness.py
"""

import csv
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
API = "https://statsapi.mlb.com/api/v1/people"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def collect_ids():
    ids = set()
    for name, col in [("mlb_game_batting.csv", "PlayerId"),
                      ("mlb_game_pitching.csv", "PlayerId"),
                      ("mlb_rosters.csv", "PlayerId")]:
        with open(DATA_DIR / name, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    ids.add(int(row[col]))
                except (ValueError, KeyError):
                    pass
    return sorted(ids)


def main():
    ids = collect_ids()
    print(f"fetching handedness for {len(ids)} players...")
    rows = []
    session = requests.Session()
    session.headers.update(HEADERS)
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        try:
            r = session.get(API, params={
                "personIds": ",".join(map(str, chunk)),
                "fields": "people,id,fullName,batSide,pitchHand,code"},
                timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"chunk {i}: FAILED ({e})", file=sys.stderr)
            sys.exit(1)
        for p in r.json().get("people", []):
            rows.append({
                "PlayerId": p["id"], "Name": p.get("fullName", ""),
                "Bats": p.get("batSide", {}).get("code", ""),
                "Throws": p.get("pitchHand", {}).get("code", ""),
            })
        if (i // 100) % 10 == 9:
            print(f"  {i + len(chunk)}/{len(ids)}")
        time.sleep(0.3)

    out = DATA_DIR / "mlb_handedness.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["PlayerId", "Name", "Bats", "Throws"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} players to {out}")


if __name__ == "__main__":
    main()
