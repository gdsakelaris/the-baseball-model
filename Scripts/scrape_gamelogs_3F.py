"""Scrape per-game boxscore logs for every MLB regular-season game 2020-2026.

Uses MLB's public Stats API (statsapi.mlb.com): one schedule call per season
to enumerate final games, then one boxscore call per game (~14,000 games,
fetched concurrently). Writes three relational CSVs:

  mlb_games_2020_2026.csv          one row per game: teams, score, venue,
                                   day/night, temperature, wind, conditions
  mlb_game_batting_2020_2026.csv   one row per batter-game: lineup slot,
                                   position, PA/AB/H/HR/BB/SO/... (the
                                   per-game labels for modeling)
  mlb_game_pitching_2020_2026.csv  one row per pitcher-game: started or
                                   relieved, IP, batters faced, pitches,
                                   H/R/ER/HR/BB/SO, decisions

Relational keys: PlayerId matches every other CSV; Team/Opponent are MLB
abbreviations (per-season correct, e.g. OAK through 2024, ATH from 2025);
GamePk links the three files; Venue names match Ballpark in
mlb_ballparks.csv for current parks.

Completed seasons are cached under <outdir>/cache/ as JSON after first
fetch; the newest (in-progress) season is always re-fetched. Delete a cache
file to force a season's re-fetch.

BattingOrder is MLB's slot code: 100 = leadoff starter, 401 = first
substitute into the 4th slot, etc. Starters end in 00.

Usage:
    python scrape_gamelogs.py [--outdir DIR] [--workers N]
"""

import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import requests

API = "https://statsapi.mlb.com/api/v1"
YEARS = range(2020, 2027)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

GAME_COLS = ["GamePk", "Season", "Date", "DayNight", "AwayTeam", "HomeTeam",
             "AwayScore", "HomeScore", "Venue", "Temp", "Condition",
             "WindSpeed", "WindDir"]

# One canonical name per physical park across all years and all CSVs
# (renames, sponsor wrappers, stale/case variants). Keep in sync with
# scrape_homeruns.py and build_ballparks.py.
VENUE_ALIASES = {
    "Minute Maid Park": "Daikin Park",
    "Guaranteed Rate Field": "Rate Field",
    "Marlins Park": "loanDepot Park",
    "loanDepot park": "loanDepot Park",
    "UNIQLO Field at Dodger Stadium": "Dodger Stadium",
    "Camden Yards": "Oriole Park at Camden Yards",
}

BAT_STATS = {  # CSV column -> boxscore batting field
    "PA": "plateAppearances", "AB": "atBats", "R": "runs", "H": "hits",
    "2B": "doubles", "3B": "triples", "HR": "homeRuns", "RBI": "rbi",
    "BB": "baseOnBalls", "IBB": "intentionalWalks", "SO": "strikeOuts",
    "HBP": "hitByPitch", "SB": "stolenBases", "CS": "caughtStealing",
    "SAC": "sacBunts", "SF": "sacFlies", "GIDP": "groundIntoDoublePlay",
    "TB": "totalBases", "LOB": "leftOnBase",
}
BAT_COLS = ["GamePk", "Season", "Date", "PlayerId", "Name", "Team", "Opponent",
            "Home", "BattingOrder", "Position"] + list(BAT_STATS)

PIT_STATS = {  # CSV column -> boxscore pitching field
    "GS": "gamesStarted", "GF": "gamesFinished", "IP": "inningsPitched",
    "BF": "battersFaced", "NP": "numberOfPitches", "Strikes": "strikes",
    "H": "hits", "R": "runs", "ER": "earnedRuns", "HR": "homeRuns",
    "BB": "baseOnBalls", "IBB": "intentionalWalks", "SO": "strikeOuts",
    "HBP": "hitBatsmen", "WP": "wildPitches", "BK": "balks",
    "W": "wins", "L": "losses", "SV": "saves", "HLD": "holds",
}
PIT_COLS = ["GamePk", "Season", "Date", "PlayerId", "Name", "Team", "Opponent",
            "Home"] + list(PIT_STATS)

_local = threading.local()


def get_session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update(HEADERS)
    return _local.session


def get_json(url, params=None, tries=4):
    for attempt in range(tries):
        try:
            resp = get_session().get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def team_abbrevs(season):
    """teamId -> abbreviation for one season (handles renames like OAK->ATH)."""
    data = get_json(f"{API}/teams", {"sportId": 1, "season": season})
    return {t["id"]: t["abbreviation"] for t in data["teams"]}


def season_schedule(season):
    """Final regular-season games: [(gamePk, date, dayNight, venue, scores, team ids)]."""
    data = get_json(f"{API}/schedule",
                    {"sportId": 1, "season": season, "gameType": "R"})
    games, seen = [], set()
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g["status"].get("codedGameState") != "F":
                continue
            pk = g["gamePk"]
            if pk in seen:  # resumed/suspended games list twice
                continue
            seen.add(pk)
            games.append({
                "GamePk": pk,
                "Date": g["officialDate"],
                "DayNight": g.get("dayNight", ""),
                "Venue": g.get("venue", {}).get("name", ""),
                "AwayScore": g["teams"]["away"].get("score", ""),
                "HomeScore": g["teams"]["home"].get("score", ""),
                "away_id": g["teams"]["away"]["team"]["id"],
                "home_id": g["teams"]["home"]["team"]["id"],
            })
    return games


def parse_weather(info):
    """Boxscore info: 'Weather: 72 degrees, Partly Cloudy.' 'Wind: 10 mph, Out To CF.'"""
    temp = cond = speed = wdir = ""
    for item in info:
        label, value = item.get("label"), item.get("value", "")
        if label == "Weather":
            m = re.match(r"(-?\d+)\s*degrees,?\s*(.*?)\.?$", value)
            if m:
                temp, cond = m.group(1), m.group(2).strip()
        elif label == "Wind":
            m = re.match(r"(\d+)\s*mph,?\s*(.*?)\.?$", value)
            if m:
                speed, wdir = m.group(1), m.group(2).strip()
    return temp, cond, speed, wdir


def parse_boxscore(game, box, abbrevs, season):
    """Return (game_row, batting_rows, pitching_rows) for one game."""
    away = abbrevs.get(game["away_id"], str(game["away_id"]))
    home = abbrevs.get(game["home_id"], str(game["home_id"]))
    temp, cond, speed, wdir = parse_weather(box.get("info", []))
    game_row = {
        "GamePk": game["GamePk"], "Season": season, "Date": game["Date"],
        "DayNight": game["DayNight"], "AwayTeam": away, "HomeTeam": home,
        "AwayScore": game["AwayScore"], "HomeScore": game["HomeScore"],
        "Venue": game["Venue"], "Temp": temp, "Condition": cond,
        "WindSpeed": speed, "WindDir": wdir,
    }

    bat_rows, pit_rows = [], []
    for side, team, opp, is_home in (("away", away, home, 0), ("home", home, away, 1)):
        for p in box["teams"][side]["players"].values():
            common = {
                "GamePk": game["GamePk"], "Season": season, "Date": game["Date"],
                "PlayerId": p["person"]["id"], "Name": p["person"]["fullName"],
                "Team": team, "Opponent": opp, "Home": is_home,
            }
            bat = p.get("stats", {}).get("batting", {})
            if bat.get("gamesPlayed") or p.get("battingOrder"):
                row = dict(common)
                row["BattingOrder"] = p.get("battingOrder", "")
                row["Position"] = p.get("position", {}).get("abbreviation", "")
                for col, field in BAT_STATS.items():
                    row[col] = bat.get(field, 0)
                bat_rows.append(row)
            pit = p.get("stats", {}).get("pitching", {})
            if pit.get("gamesPlayed"):
                row = dict(common)
                for col, field in PIT_STATS.items():
                    row[col] = pit.get(field, 0)
                pit_rows.append(row)
    return game_row, bat_rows, pit_rows


def fetch_season(season, workers):
    """Fetch and parse a whole season. Returns dict of the three row lists."""
    abbrevs = team_abbrevs(season)
    games = season_schedule(season)
    print(f"{season}: {len(games)} final games, fetching boxscores...", flush=True)

    game_rows, bat_rows, pit_rows, failed = [], [], [], []

    def work(game):
        box = get_json(f"{API}/game/{game['GamePk']}/boxscore")
        return parse_boxscore(game, box, abbrevs, season)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, g): g for g in games}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            g = futures[fut]
            try:
                game_row, bats, pits = fut.result()
                game_rows.append(game_row)
                bat_rows.extend(bats)
                pit_rows.extend(pits)
            except Exception as e:
                failed.append((g["GamePk"], str(e)))
            done += 1
            if done % 500 == 0:
                print(f"  {season}: {done}/{len(games)}", flush=True)

    if failed:
        raise RuntimeError(f"{season}: {len(failed)} boxscores failed, e.g. {failed[:3]}")

    # deterministic order: by date then gamePk
    game_rows.sort(key=lambda r: (r["Date"], r["GamePk"]))
    order = {r["GamePk"]: i for i, r in enumerate(game_rows)}
    bat_rows.sort(key=lambda r: (order[r["GamePk"]], r["Home"], str(r["BattingOrder"] or "999"), r["PlayerId"]))
    pit_rows.sort(key=lambda r: (order[r["GamePk"]], r["Home"], -int(r["GS"] or 0), r["PlayerId"]))
    return {"games": game_rows, "batting": bat_rows, "pitching": pit_rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[1] / "Data"))
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    cache_dir = os.path.join(args.outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    newest = max(YEARS)

    all_data = {"games": [], "batting": [], "pitching": []}
    for season in YEARS:
        cache_file = os.path.join(cache_dir, f"gamelogs_{season}.json")
        if season != newest and os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as f:
                data = json.load(f)
            print(f"{season}: loaded {len(data['games'])} games from cache")
        else:
            try:
                data = fetch_season(season, args.workers)
            except Exception as e:
                print(f"{season}: FAILED ({e})", file=sys.stderr)
                sys.exit(1)
            if season != newest:  # completed seasons never change
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
            print(f"{season}: {len(data['games'])} games, "
                  f"{len(data['batting'])} batting lines, "
                  f"{len(data['pitching'])} pitching lines")
        for row in data["games"]:
            row["Venue"] = VENUE_ALIASES.get(row["Venue"], row["Venue"])
        for key in all_data:
            all_data[key].extend(data[key])

    outputs = [
        ("mlb_games_2020_2026.csv", GAME_COLS, all_data["games"]),
        ("mlb_game_batting_2020_2026.csv", BAT_COLS, all_data["batting"]),
        ("mlb_game_pitching_2020_2026.csv", PIT_COLS, all_data["pitching"]),
    ]
    for name, cols, rows in outputs:
        path = os.path.join(args.outdir, name)
        # utf-8-sig so Excel renders accented names correctly
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
