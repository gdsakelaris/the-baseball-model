"""Scrape every MLB team's roster from the MLB.com depth chart pages.

Fetches https://www.mlb.com/<team>/roster/depth-chart for all 30 teams and
writes one CSV with: PlayerId, Name, Team, Position, B, T, Ht, Wt, DOB.
PlayerId is MLB's stable player ID, shared with mlb_batting_stats_*.csv so
the two files can be joined on it.

Players listed under multiple position groups on a depth chart are kept once,
under the first group they appear in (their primary slot on the page).
Run with --all-positions to instead keep one row per position listing.

Usage:
    python scrape_rosters.py [--all-positions] [-o output.csv]
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

TEAMS = {
    "dbacks": "Arizona Diamondbacks",
    "braves": "Atlanta Braves",
    "orioles": "Baltimore Orioles",
    "redsox": "Boston Red Sox",
    "cubs": "Chicago Cubs",
    "whitesox": "Chicago White Sox",
    "reds": "Cincinnati Reds",
    "guardians": "Cleveland Guardians",
    "rockies": "Colorado Rockies",
    "tigers": "Detroit Tigers",
    "astros": "Houston Astros",
    "royals": "Kansas City Royals",
    "angels": "Los Angeles Angels",
    "dodgers": "Los Angeles Dodgers",
    "marlins": "Miami Marlins",
    "brewers": "Milwaukee Brewers",
    "twins": "Minnesota Twins",
    "mets": "New York Mets",
    "yankees": "New York Yankees",
    "athletics": "Athletics",
    "phillies": "Philadelphia Phillies",
    "pirates": "Pittsburgh Pirates",
    "padres": "San Diego Padres",
    "giants": "San Francisco Giants",
    "mariners": "Seattle Mariners",
    "cardinals": "St. Louis Cardinals",
    "rays": "Tampa Bay Rays",
    "rangers": "Texas Rangers",
    "bluejays": "Toronto Blue Jays",
    "nationals": "Washington Nationals",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}


def parse_depth_chart(html, team_name, keep_all_positions=False):
    """Parse one depth-chart page into a list of row dicts."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    seen = set()

    for table in soup.select("table.roster__table"):
        thead = table.find("thead")
        if not thead:
            continue
        position = thead.find("td").get_text(strip=True)

        tbody = table.find("tbody")
        if not tbody:
            continue
        for tr in tbody.find_all("tr"):
            info = tr.find("td", class_="info")
            if not info:
                continue
            link = info.find("a")
            if not link:
                continue
            name = link.get_text(strip=True)
            # Player-page URL is https://www.mlb.com/player/<id>.
            player_id = link.get("href", "").rstrip("/").rsplit("/", 1)[-1]
            if not keep_all_positions:
                if player_id in seen:
                    continue
                seen.add(player_id)

            def cell(cls):
                td = tr.find("td", class_=cls)
                return td.get_text(strip=True) if td else ""

            bats, _, throws = cell("bat-throw").partition("/")
            rows.append({
                "PlayerId": player_id,
                "Name": name,
                "Team": team_name,
                "Position": position,
                "B": bats,
                "T": throws,
                "Ht": cell("height"),
                "Wt": cell("weight"),
                "DOB": cell("birthday"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", default=str(DATA_DIR / "mlb_rosters_2026.csv"))
    ap.add_argument(
        "--all-positions",
        action="store_true",
        help="keep a row for every position group a player is listed under",
    )
    args = ap.parse_args()

    all_rows = []
    failures = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for i, (slug, team_name) in enumerate(TEAMS.items()):
        url = f"https://www.mlb.com/{slug}/roster/depth-chart"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            rows = parse_depth_chart(resp.text, team_name, args.all_positions)
            if not rows:
                raise ValueError("no roster tables found on page")
            all_rows.extend(rows)
            print(f"[{i + 1:2}/30] {team_name}: {len(rows)} players")
        except Exception as e:
            failures.append((team_name, url, e))
            print(f"[{i + 1:2}/30] {team_name}: FAILED ({e})", file=sys.stderr)
        time.sleep(1)  # be polite to MLB.com

    fieldnames = ["PlayerId", "Name", "Team", "Position", "B", "T", "Ht", "Wt", "DOB"]
    # utf-8-sig so Excel renders accented names (José, Berríos) correctly.
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} players to {args.output}")
    if failures:
        print(f"{len(failures)} team(s) failed:", file=sys.stderr)
        for team_name, url, e in failures:
            print(f"  {team_name}: {url} ({e})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
