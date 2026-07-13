"""Build a ballpark dimensions + elevation CSV.

Dimensions come from the MLB.com ballpark table (left/center/right field
distances in feet). Elevation above sea level is looked up per stadium from
the USGS National Map elevation service (epqs.nationalmap.gov) using each
park's coordinates; Rogers Centre (Canada) uses opentopodata.org instead,
since USGS only covers the US. Values are rounded to the nearest foot.

Team names match the TeamName column of the stats CSVs and the Team column
of the roster CSV, so this file joins to them directly.

Also emitted per park: latitude/longitude (the weather scraper keys
Open-Meteo requests on these) and Roof (open / retractable / dome — lets
consumers ignore outdoor weather when a game is played under a roof).

If an elevation lookup fails, the value from the existing output CSV is
reused instead of failing the run (elevations never change).

Usage:
    python build_ballparks.py [-o output.csv]
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

# Ballpark, Team, Dimensions (as published), latitude, longitude, roof.
# Names are the canonical 2026 venue names used across the game-log and
# homerun CSVs: Daikin Park (ex-Minute Maid), Rate Field (ex-Guaranteed
# Rate), Oriole Park at Camden Yards, and Tropicana Field (Rays returned in
# 2026 after playing 2025 at Steinbrenner Field during hurricane repairs).
# Roof: open / retractable / dome (fixed).
BALLPARKS = [
    ("American Family Field", "Milwaukee Brewers", "332-L, 400-C, 325-R", 43.0280, -87.9712, "retractable"),
    ("Angel Stadium", "Los Angeles Angels", "330-L, 396-C, 330-R", 33.8003, -117.8827, "open"),
    ("Busch Stadium", "St. Louis Cardinals", "336-L, 400-C, 335-R", 38.6226, -90.1928, "open"),
    ("Oriole Park at Camden Yards", "Baltimore Orioles", "337-L, 406-C, 320-R", 39.2839, -76.6217, "open"),
    ("Chase Field", "Arizona Diamondbacks", "330-L, 407-C, 335-R", 33.4455, -112.0667, "retractable"),
    ("Citi Field", "New York Mets", "335-L, 405-C, 330-R", 40.7571, -73.8458, "open"),
    # Sampled at the stadium's street level; dead center hits the excavated
    # field bowl in the USGS bare-earth model and returns a negative value.
    ("Citizens Bank Park", "Philadelphia Phillies", "330-L, 401-C, 329-R", 39.9075, -75.1682, "open"),
    ("Comerica Park", "Detroit Tigers", "345-L, 420-C, 330-R", 42.3390, -83.0485, "open"),
    ("Coors Field", "Colorado Rockies", "347-L, 415-C, 350-R", 39.7559, -104.9942, "open"),
    ("Dodger Stadium", "Los Angeles Dodgers", "330-L, 400-C, 300-R", 34.0739, -118.2400, "open"),
    ("Fenway Park", "Boston Red Sox", "310-L, 420-C, 302-R", 42.3467, -71.0972, "open"),
    ("Globe Life Field", "Texas Rangers", "329-L, 407-C, 326-R", 32.7473, -97.0841, "retractable"),
    ("Great American Ball Park", "Cincinnati Reds", "325-R, 404-C, 328-L", 39.0975, -84.5066, "open"),
    ("Rate Field", "Chicago White Sox", "330-L, 400-C, 335-R", 41.8300, -87.6339, "open"),
    ("Kauffman Stadium", "Kansas City Royals", "330-L, 400-C, 330-R", 39.0517, -94.4803, "open"),
    ("loanDepot Park", "Miami Marlins", "340-L, 420-C, 335-R", 25.7781, -80.2196, "retractable"),
    ("Daikin Park", "Houston Astros", "315-L, 435-C, 326-R", 29.7573, -95.3555, "retractable"),
    ("Nationals Park", "Washington Nationals", "336-L, 403-C, 335-R", 38.8730, -77.0074, "open"),
    ("Oracle Park", "San Francisco Giants", "339-L, 399-C, 309-R", 37.7786, -122.3893, "open"),
    ("Petco Park", "San Diego Padres", "336-L, 396-C, 322-R", 32.7076, -117.1570, "open"),
    ("PNC Park", "Pittsburgh Pirates", "325-L, 399-C, 320-R", 40.4469, -80.0057, "open"),
    ("Progressive Field", "Cleveland Guardians", "325-L, 405-C, 325-R", 41.4962, -81.6852, "open"),
    # Sampled at street level beside the park; SRTM is a surface model, so
    # dead center measures the top of the dome (~380 ft), not the ground.
    ("Rogers Centre", "Toronto Blue Jays", "328-L, 400-C, 328-R", 43.6398, -79.3820, "retractable"),
    # Sampled at the west parking lot; the Trop is a dome, so a dead-center
    # sample could hit the roof or the below-grade bowl in elevation models.
    ("Tropicana Field", "Tampa Bay Rays", "315-L, 404-C, 322-R", 27.7690, -82.6570, "dome"),
    ("Sutter Health Park", "Athletics", "330-L, 403-C, 325-R", 38.5802, -121.5133, "open"),
    ("T-Mobile Park", "Seattle Mariners", "331-L, 405-C, 327-R", 47.5914, -122.3325, "retractable"),
    ("Target Field", "Minnesota Twins", "339-L, 404-C, 328-R", 44.9817, -93.2776, "open"),
    ("Truist Park", "Atlanta Braves", "335-L, 400-C, 325-R", 33.8908, -84.4678, "open"),
    ("Wrigley Field", "Chicago Cubs", "355-L, 400-C, 353-R", 41.9484, -87.6553, "open"),
    ("Yankee Stadium", "New York Yankees", "318-L, 404-C, 314-R", 40.8296, -73.9262, "open"),
]

USGS_URL = "https://epqs.nationalmap.gov/v1/json"
OPENTOPO_URL = "https://api.opentopodata.org/v1/srtm30m"
FEET_PER_METER = 3.28084


def parse_dimensions(dims):
    """Parse '332-L, 400-C, 325-R' into (LF, CF, RF) regardless of order."""
    fields = {}
    for feet, side in re.findall(r"(\d+)-([LCR])", dims):
        fields[side] = int(feet)
    if set(fields) != {"L", "C", "R"}:
        raise ValueError(f"bad dimensions string: {dims!r}")
    return fields["L"], fields["C"], fields["R"]


def usgs_elevation_ft(session, lat, lon):
    resp = session.get(
        USGS_URL,
        params={"x": lon, "y": lat, "units": "Feet", "wkid": "4326"},
        timeout=30,
    )
    resp.raise_for_status()
    return round(float(resp.json()["value"]))


def opentopo_elevation_ft(session, lat, lon):
    resp = session.get(OPENTOPO_URL, params={"locations": f"{lat},{lon}"}, timeout=30)
    resp.raise_for_status()
    meters = resp.json()["results"][0]["elevation"]
    return round(meters * FEET_PER_METER)


def stored_elevations(path):
    """Ballpark -> Elevation_ft from the existing output CSV (fallback when
    an elevation service is down; elevations never change)."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig") as f:
        return {r["Ballpark"]: int(r["Elevation_ft"])
                for r in csv.DictReader(f) if r.get("Elevation_ft")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DATA_DIR / "mlb_ballparks.csv"))
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    stored = stored_elevations(args.output)

    rows = []
    for name, team, dims, lat, lon, roof in BALLPARKS:
        lf, cf, rf = parse_dimensions(dims)
        try:
            if lon < -60 and lat < 49.5 and name != "Rogers Centre":
                elev = usgs_elevation_ft(session, lat, lon)
            else:
                elev = opentopo_elevation_ft(session, lat, lon)
            time.sleep(1)  # both elevation services are rate-limited
        except Exception as e:
            if name in stored:
                elev = stored[name]
                print(f"{name}: elevation lookup failed ({e}); reusing "
                      f"stored {elev} ft", file=sys.stderr)
            else:
                print(f"{name}: elevation lookup FAILED ({e})", file=sys.stderr)
                sys.exit(1)
        rows.append({
            "Ballpark": name,
            "Team": team,
            "LF": lf,
            "CF": cf,
            "RF": rf,
            "Elevation_ft": elev,
            "Lat": lat,
            "Lon": lon,
            "Roof": roof,
        })
        print(f"{name}: LF {lf} / CF {cf} / RF {rf}, {elev} ft, {roof}")

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Ballpark", "Team", "LF", "CF",
                                               "RF", "Elevation_ft", "Lat",
                                               "Lon", "Roof"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} ballparks to {args.output}")


if __name__ == "__main__":
    main()
