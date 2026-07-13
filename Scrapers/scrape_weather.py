"""Scrape per-game weather (humidity, pressure, precipitation) from Open-Meteo.

The MLB boxscores already carry Temp/WindSpeed/WindDir, but not humidity or
barometric pressure — and denser air (cold, humid-low, high-pressure days)
measurably suppresses carry. This fills that gap: one row per GamePk with

  Humidity  relative humidity (%) at the game-start hour
  Pressure  surface (station-level) pressure, hPa, at the game-start hour —
            embeds the park's elevation, so it is the physical air-density
            input directly
  Precip    total precipitation (mm) over the first three game hours

Open-Meteo (open-meteo.com) is keyless and free for non-commercial use:
the archive endpoint covers 1940->about 5 days ago; anything newer comes
from the forecast endpoint's past_days window. Coordinates come from
mlb_ballparks.csv (Lat/Lon, written by build_ballparks.py); former and
special-event venues (Oakland Coliseum, Miller Park, London Stadium, ...)
have their coordinates here. The game-start hour is approximated from
DayNight in park-local time (day -> 13:00, night -> 19:00); Open-Meteo's
timezone=auto returns hourly arrays already in local time.

Default run is incremental — only games missing from the output CSV are
fetched (seconds in the daily job, right after scrape_gamelogs_3F adds
yesterday's finals). --backfill refetches everything (~5-10 minutes).

Usage:
    python scrape_weather.py [-o output.csv] [--backfill]
"""

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
DEFAULT_OUT = DATA_DIR / "mlb_weather.csv"
GAMES_CSV = DATA_DIR / "mlb_games.csv"
PARKS_CSV = DATA_DIR / "mlb_ballparks.csv"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "relative_humidity_2m,surface_pressure,precipitation"
ARCHIVE_LAG_DAYS = 6      # archive endpoint trails realtime by ~5 days
GAP_DAYS = 45             # date runs closer than this share one request
HOUR_BY_DAYNIGHT = {"day": 13, "night": 19}
SLEEP = 0.3

# Venues in mlb_games.csv that are NOT current parks (mlb_ballparks.csv is
# deliberately current-only). Former names of current parks map onto the
# ballparks row via CURRENT_ALIASES; everything else gets coordinates here.
EXTRA_VENUES = {
    "Oakland Coliseum": (37.7516, -122.2005),
    "Globe Life Park in Arlington": (32.7512, -97.0832),
    "Turner Field": (33.7350, -84.3900),
    "George M. Steinbrenner Field": (27.9795, -82.5064),
    "Sahlen Field": (42.8804, -78.8738),
    "TD Ballpark": (28.0028, -82.7873),
    "Estadio de Beisbol Monterrey": (25.7205, -100.3117),
    "London Stadium": (51.5386, -0.0166),
    "Estadio Alfredo Harp Helu": (19.4042, -99.0907),
    "Las Vegas Ballpark": (36.1495, -115.3384),
    "Tokyo Dome": (35.7056, 139.7519),
    "Gocheok Sky Dome": (37.4982, 126.8672),
    "Hiram Bithorn Stadium": (18.4155, -66.0735),
    "Fort Bragg Field": (35.1270, -79.0180),
    "Bristol Motor Speedway": (36.5157, -82.2570),
    "Rickwood Field": (33.5021, -86.8296),
    "TD Ameritrade Park": (41.2665, -95.9245),
    # Williamsport Little League Classic park (Bowman Field), renamed twice
    "BB&T Ballpark": (41.2404, -77.0480),
    "Muncy Bank Ballpark": (41.2404, -77.0480),
    "Journey Bank Ballpark": (41.2404, -77.0480),
}
CURRENT_ALIASES = {
    "Miller Park": "American Family Field",
    "Safeco Field": "T-Mobile Park",
    "AT&T Park": "Oracle Park",
    "Angel Stadium of Anaheim": "Angel Stadium",
    "U.S. Cellular Field": "Rate Field",
    "SunTrust Park": "Truist Park",
    "O.co Coliseum": "Oakland Coliseum",
}


def venue_coords():
    """Venue name (as it appears in mlb_games.csv) -> (lat, lon)."""
    parks = pd.read_csv(PARKS_CSV, encoding="utf-8-sig")
    if "Lat" not in parks.columns:
        raise SystemExit("mlb_ballparks.csv has no Lat/Lon columns — run "
                         "build_ballparks.py first")
    coords = {r.Ballpark: (r.Lat, r.Lon) for r in parks.itertuples()}
    coords.update(EXTRA_VENUES)
    for old, new in CURRENT_ALIASES.items():
        if new in coords:
            coords[old] = coords[new]
    return coords


def fetch_hourly(url, lat, lon, params, tries=4):
    """One Open-Meteo request -> {'YYYY-MM-DDTHH': (hum, pres, prec)}."""
    q = {"latitude": lat, "longitude": lon, "hourly": HOURLY_VARS,
         "timezone": "auto", **params}
    for attempt in range(tries):
        try:
            r = requests.get(url, params=q, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            return dict(zip(
                h["time"],
                zip(h["relative_humidity_2m"], h["surface_pressure"],
                    h["precipitation"])))
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"    retry in {wait}s ({e})", flush=True)
            time.sleep(wait)


def date_runs(dates, gap=GAP_DAYS):
    """Sorted unique dates -> [(start, end), ...] merging near neighbors."""
    runs = []
    for d in sorted(set(dates)):
        if runs and (d - runs[-1][1]).days <= gap:
            runs[-1][1] = d
        else:
            runs.append([d, d])
    return [tuple(r) for r in runs]


def rows_for_games(games, hours):
    """Index one venue's fetched hours for each game -> output row dicts."""
    out = []
    for g in games.itertuples():
        h = HOUR_BY_DAYNIGHT.get(g.DayNight, 19)
        key = f"{g.Date:%Y-%m-%d}T{h:02d}:00"
        hum, pres, prec3 = None, None, None
        got = hours.get(key)
        if got is not None:
            hum, pres, prec3 = got[0], got[1], 0.0
            for hh in range(h, h + 3):
                v = hours.get(f"{g.Date:%Y-%m-%d}T{hh:02d}:00")
                if v is not None and v[2] is not None:
                    prec3 += v[2]
        out.append({"GamePk": g.GamePk, "Date": f"{g.Date:%Y-%m-%d}",
                    "Venue": g.Venue, "Humidity": hum, "Pressure": pres,
                    "Precip": prec3})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DEFAULT_OUT))
    ap.add_argument("--backfill", action="store_true",
                    help="refetch every game, ignoring stored rows")
    args = ap.parse_args()
    out_path = Path(args.output)

    games = pd.read_csv(GAMES_CSV, encoding="utf-8-sig",
                        usecols=["GamePk", "Date", "DayNight", "Venue"])
    games["Date"] = pd.to_datetime(games["Date"])

    stored = None
    if out_path.exists() and not args.backfill:
        stored = pd.read_csv(out_path, encoding="utf-8-sig")
        games = games[~games["GamePk"].isin(set(stored["GamePk"]))]
    print(f"{len(games):,} games need weather", flush=True)

    coords = venue_coords()
    unmatched = games[~games["Venue"].isin(coords)]
    if len(unmatched):
        print("WARNING: no coordinates for these venues (rows skipped):")
        for v, n in unmatched["Venue"].value_counts().items():
            print(f"    {v}: {n} game(s)")
        games = games[games["Venue"].isin(coords)]

    cutoff = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    rows = []
    for venue, vg in games.groupby("Venue"):
        lat, lon = coords[venue]
        old = vg[vg["Date"].dt.date < cutoff]
        new = vg[vg["Date"].dt.date >= cutoff]
        for d0, d1 in date_runs(old["Date"].dt.date):
            hours = fetch_hourly(ARCHIVE_URL, lat, lon,
                                 {"start_date": str(d0), "end_date": str(d1)})
            sel = old[(old["Date"].dt.date >= d0) & (old["Date"].dt.date <= d1)]
            rows.extend(rows_for_games(sel, hours))
            time.sleep(SLEEP)
        if len(new):
            past = (date.today() - new["Date"].dt.date.min()).days + 1
            hours = fetch_hourly(FORECAST_URL, lat, lon,
                                 {"past_days": min(max(past, 1), 92),
                                  "forecast_days": 1})
            rows.extend(rows_for_games(new, hours))
            time.sleep(SLEEP)
        print(f"{venue}: {len(vg):,} games", flush=True)

    fresh = pd.DataFrame(rows, columns=["GamePk", "Date", "Venue",
                                        "Humidity", "Pressure", "Precip"])
    if stored is not None:
        fresh = pd.concat([stored, fresh], ignore_index=True)
    fresh = (fresh.drop_duplicates("GamePk", keep="last")
             .sort_values(["Date", "GamePk"]))
    out_path.parent.mkdir(exist_ok=True)
    fresh.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(fresh):,} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
