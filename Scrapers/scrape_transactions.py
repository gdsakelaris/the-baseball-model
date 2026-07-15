"""Scrape MLB injured-list transactions (statsapi, sportId=1) and pair
them into IL STINTS — the layoff-cause data the days-rest gap features
cannot see (an IL stint, a skipped start, and the All-Star break have very
different return-performance profiles).

Two outputs:

  mlb_il_events.csv — one row per raw IL-related transaction (place /
      transfer / activate / rehab), per (PlayerId, Date, Kind). This is
      the incremental cache: completed years never change, only the
      current year is refetched.
  mlb_il.csv — one row per PAIRED STINT: PlayerId, PlaceDate, ActDate
      (activation; empty while still on the IL), StintDays, IL60 (reached
      the 60-day IL), Rehab (stint included a rehab assignment). Rebuilt
      from ALL events on every run because stints span season boundaries
      (placed in September, activated in April). THIS is the file the
      model consumes.

Leakage note for consumers: a transaction is announced BEFORE the game
(roster moves precede lineups), so activation-date joins may use
allow_exact_matches — a player activated this morning plays tonight with
il_ret_days = 0.

Usage:
    python scrape_transactions.py [--backfill]
"""

import argparse
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from seasons import CURRENT_SEASON, YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUT_EVENTS = DATA_DIR / "mlb_il_events.csv"
OUT_STINTS = DATA_DIR / "mlb_il.csv"

API_URL = "https://statsapi.mlb.com/api/v1/transactions"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
EVENT_COLS = ["PlayerId", "Date", "Kind", "ILDays"]

# "injured list" from 2019; "disabled list" before the 2019 rename
_DAYS_RE = re.compile(r"(\d+)-day (?:injured|disabled) list")


def classify(desc):
    """(kind, il_days) from a transaction description, or (None, None).
    Kinds: place / transfer / activate / rehab."""
    d = desc.lower()
    if "rehab assignment" in d:
        return "rehab", None
    if "injured list" not in d and "disabled list" not in d:
        return None, None
    m = _DAYS_RE.search(d)
    days = int(m.group(1)) if m else None
    if "transferred" in d:
        return "transfer", days
    if "placed" in d:
        return "place", days
    if "activated" in d or "reinstated" in d or "returned" in d:
        return "activate", days
    return None, None


def fetch_year(year, tries=4):
    """All MLB IL-related events for one calendar year (monthly windows —
    the endpoint caps long ranges)."""
    rows = []
    for month in range(1, 13):
        d0 = date(year, month, 1)
        d1 = (date(year, month + 1, 1) if month < 12
              else date(year + 1, 1, 1))
        if d0 > date.today():
            break
        params = {"startDate": str(d0),
                  "endDate": str(min(d1, date.today())), "sportId": 1}
        for attempt in range(tries):
            try:
                r = requests.get(API_URL, params=params, headers=HEADERS,
                                 timeout=120)
                r.raise_for_status()
                tx = r.json().get("transactions", [])
                break
            except Exception as e:                  # noqa: BLE001
                if attempt == tries - 1:
                    raise
                wait = 10 * (attempt + 1)
                print(f"    retry {d0} in {wait}s ({e})", flush=True)
                time.sleep(wait)
        for t in tx:
            pid = t.get("person", {}).get("id")
            desc = t.get("description") or ""
            if pid is None:
                continue
            kind, days = classify(desc)
            if kind is None:
                continue
            rows.append({"PlayerId": pid, "Date": t.get("date"),
                         "Kind": kind, "ILDays": days})
        time.sleep(0.5)
    df = pd.DataFrame(rows, columns=EVENT_COLS)
    return df.drop_duplicates(EVENT_COLS)


def build_stints(events):
    """Pair place -> activate per player into stints. Transfers extend the
    open stint (60-day flag); rehab events inside an open stint set the
    Rehab flag; an activation without an open stint is dropped; a stint
    still open at the end gets an empty ActDate."""
    ev = events.copy()
    ev["Date"] = pd.to_datetime(ev["Date"])
    ev = ev.sort_values(["PlayerId", "Date"], kind="mergesort")
    out = []
    for pid, g in ev.groupby("PlayerId", sort=False):
        open_stint = None
        for _, e in g.iterrows():
            k = e["Kind"]
            if k in ("place", "transfer"):
                if open_stint is None:
                    open_stint = {"PlayerId": pid, "PlaceDate": e["Date"],
                                  "IL60": 0, "Rehab": 0}
                if (e["ILDays"] or 0) >= 60:
                    open_stint["IL60"] = 1
            elif k == "rehab":
                if open_stint is not None:
                    open_stint["Rehab"] = 1
            elif k == "activate":
                if open_stint is not None:
                    open_stint["ActDate"] = e["Date"]
                    open_stint["StintDays"] = (
                        e["Date"] - open_stint["PlaceDate"]).days
                    out.append(open_stint)
                    open_stint = None
        if open_stint is not None:
            open_stint["ActDate"] = pd.NaT
            open_stint["StintDays"] = pd.NA
            out.append(open_stint)
    st = pd.DataFrame(out, columns=["PlayerId", "PlaceDate", "ActDate",
                                    "StintDays", "IL60", "Rehab"])
    return st.sort_values(["PlayerId", "PlaceDate"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every year, ignoring stored events")
    args = ap.parse_args()

    stored = None
    if OUT_EVENTS.exists() and not args.backfill:
        stored = pd.read_csv(OUT_EVENTS, encoding="utf-8-sig")
    have = set() if stored is None else \
        set(pd.to_datetime(stored["Date"]).dt.year)

    frames = []
    for year in YEARS:
        if year != CURRENT_SEASON and year in have:
            rows = stored[pd.to_datetime(stored["Date"]).dt.year == year]
            frames.append(rows[EVENT_COLS])
            print(f"{year}: {len(rows):,} IL events (stored)", flush=True)
            continue
        df = fetch_year(year)
        frames.append(df)
        print(f"{year}: {len(df):,} IL events", flush=True)

    events = (pd.concat(frames, ignore_index=True)
              .drop_duplicates(EVENT_COLS)
              .sort_values(["PlayerId", "Date"]))
    OUT_EVENTS.parent.mkdir(exist_ok=True)
    events.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig")
    print(f"wrote {len(events):,} events -> {OUT_EVENTS}", flush=True)

    stints = build_stints(events)
    stints.to_csv(OUT_STINTS, index=False, encoding="utf-8-sig")
    done = stints["ActDate"].notna().sum()
    print(f"wrote {len(stints):,} stints ({done:,} completed) -> "
          f"{OUT_STINTS}", flush=True)


if __name__ == "__main__":
    main()
