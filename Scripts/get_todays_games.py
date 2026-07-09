"""Scrape today's slate into an input file the GUI auto-loads.

Primary source is mlb.com (teams, stadium, starters, ordered lineups with
MLB player IDs, date, start time). Weather and any lineups mlb.com hasn't
posted yet are filled from two fallbacks:

  fantasypros.com/mlb/lineups   wind speed + direction (only source for wind),
                                gametime temperature (temp fallback), and
                                full-name lineups (lineup fallback)
  rotowire.com/baseball/daily-lineups.php   temperature + sky condition,
                                and full-name lineups (secondary fallback)

Fallback lineups arrive as player names; they're resolved to MLB player IDs
via a name index built from the roster and recent game logs, so they slot
straight into the model. Indoor/retractable-roof parks with the roof closed
report no weather on the sources; those are set to Dome / calm / ~72 F.

Writes Data/todays_games.json in exactly the format Model/predict.py
consumes. Anything still unknown is left null; the model tolerates it.

Usage:
    python Scripts/get_todays_games.py
"""

import csv
import datetime as dt
import json
import re
import sys
import unicodedata
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
OUT_FILE = DATA_DIR / "todays_games.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

NICKNAME_TO_ABBREV = {
    "D-backs": "AZ", "Diamondbacks": "AZ", "Braves": "ATL", "Orioles": "BAL",
    "Red Sox": "BOS", "Cubs": "CHC", "White Sox": "CWS", "Reds": "CIN",
    "Guardians": "CLE", "Rockies": "COL", "Tigers": "DET", "Astros": "HOU",
    "Royals": "KC", "Angels": "LAA", "Dodgers": "LAD", "Marlins": "MIA",
    "Brewers": "MIL", "Twins": "MIN", "Mets": "NYM", "Yankees": "NYY",
    "Athletics": "ATH", "A's": "ATH", "Phillies": "PHI", "Pirates": "PIT",
    "Padres": "SD", "Giants": "SF", "Mariners": "SEA", "Cardinals": "STL",
    "Rays": "TB", "Rangers": "TEX", "Blue Jays": "TOR", "Nationals": "WSH",
}
ABBREV_ALIASES = {"CHW": "CWS", "ARI": "AZ", "WAS": "WSH", "KCR": "KC",
                  "SDP": "SD", "SFG": "SF", "TBR": "TB", "OAK": "ATH"}

VENUE_ALIASES = {
    "Minute Maid Park": "Daikin Park",
    "Guaranteed Rate Field": "Rate Field",
    "Marlins Park": "loanDepot Park",
    "loanDepot park": "loanDepot Park",
    "UNIQLO Field at Dodger Stadium": "Dodger Stadium",
    "Camden Yards": "Oriole Park at Camden Yards",
}

# parks with a roof; when the sources report no weather the roof is closed
DOME_VENUES = {
    "Daikin Park", "Globe Life Field", "T-Mobile Park", "Chase Field",
    "Rogers Centre", "American Family Field", "loanDepot Park",
    "Tropicana Field",
}
INDOOR = {"temp": 72.0, "wind_speed": 0.0, "wind_dir": "Calm",
          "condition": "Dome"}

WIND_SECTORS = ["Out To Cf", "Out To Rf", "L To R", "In From Lf",
                "In From Cf", "In From Rf", "R To L", "Out To Lf"]

CONDITION_MAP = {
    "clear": "Clear", "sunny": "Sunny", "partly-cloudy": "Partly Cloudy",
    "cloudy": "Cloudy", "overcast": "Overcast", "rain": "Rain",
    "drizzle": "Drizzle", "snow": "Snow", "dome": "Dome",
}


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


# ------------------------------------------------------- name resolution

def norm_name(name):
    """'José Ramírez Jr.' / 'jose-ramirez' -> 'jose ramirez'."""
    name = name.replace("-", " ")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", name)
    name = re.sub(r"[^a-z ]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


# FantasyPros suffixes some lineup slugs with a position ('josh-lowe-3b',
# 'nick-gonzales-if') to disambiguate; strip it before name resolution.
POS_TOKENS = {"p", "c", "1b", "2b", "3b", "ss", "lf", "cf", "rf", "of",
              "dh", "if", "sp", "rp", "ph", "pr", "util"}


def strip_pos_slug(name):
    parts = name.replace("-", " ").split()
    if len(parts) > 2 and parts[-1].lower() in POS_TOKENS:
        parts = parts[:-1]
    return " ".join(parts)


def full_name_to_abbrev():
    m = {}
    with open(DATA_DIR / "mlb_batting_stats.csv",
              encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            m[row["TeamName"]] = row["Team"]
    return m


def build_name_index():
    """Layered name lookup for resolving fallback lineup names to pids. Beyond
    exact (team, norm) and globally-unique norm, it adds a space-collapsed key
    (so 'O'Hearn'->'o hearn' still matches slug 'ohearn', and 'J.T.'->'j t'
    matches 'jt') and a team-scoped last-name + first-initial map (so 'caleb'
    resolves 'Cal Raleigh', 'lucas'->'Luke Raley', 'jung-lee'->'Jung Hoo Lee').
    Every non-exact tier is consulted only when it lands on a single pid, so a
    loose match can never pick the wrong player."""
    idx = {"exact": {}, "glob": {}, "team_col": {}, "col": {}, "team_lfi": {}}
    full2ab = full_name_to_abbrev()

    def add(team, name, pid):
        n = norm_name(name)
        if not n:
            return
        toks = n.split()
        c = "".join(toks)
        idx["exact"][(team, n)] = pid
        idx["glob"].setdefault(n, set()).add(pid)
        idx["team_col"].setdefault((team, c), set()).add(pid)
        idx["col"].setdefault(c, set()).add(pid)
        idx["team_lfi"].setdefault((team, toks[-1], toks[0][0]), set()).add(pid)

    with open(DATA_DIR / "mlb_rosters.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            add(full2ab.get(r["Team"]), r["Name"], int(r["PlayerId"]))
    # recent game-log batters (newest season in the file = current),
    # attributed to last team seen
    path = DATA_DIR / "mlb_game_batting.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    latest = max(r["Season"] for r in rows)
    rows = [r for r in rows if r["Season"] == latest]
    rows.sort(key=lambda r: r["Date"])
    for r in rows:
        add(r["Team"], r["Name"], int(r["PlayerId"]))
    return idx


def resolve(name, team, idx):
    """Slug/name -> pid via progressively looser but still-unambiguous tiers."""
    n = norm_name(strip_pos_slug(name))
    if not n:
        return None
    if (team, n) in idx["exact"]:
        return idx["exact"][(team, n)]
    toks = n.split()
    c = "".join(toks)
    for m, key in ((idx["team_col"], (team, c)),   # collapsed, unique in team
                   (idx["glob"], n),               # exact norm, globally unique
                   (idx["col"], c),                # collapsed, globally unique
                   (idx["team_lfi"],               # last name + first initial
                    (team, toks[-1], toks[0][0]))):
        hit = m.get(key)
        if hit and len(hit) == 1:
            return next(iter(hit))
    return None


def build_umpire_index():
    """normalized HP-ump name -> HpUmpId, from mlb_umpires.csv (the same file
    the model keys its ump tendency on). This bridges tonight's scraped ump
    NAME (rotowire) to the ID the model needs. Latest id per name."""
    path = DATA_DIR / "mlb_umpires.csv"
    if not path.exists():
        return {}
    idx = {}
    rows = sorted(csv.DictReader(open(path, encoding="utf-8-sig")),
                  key=lambda r: r.get("Date", ""))
    for r in rows:
        n = norm_name(r.get("HpUmp", ""))
        uid = r.get("HpUmpId")
        if n and uid:
            try:
                idx[n] = int(float(uid))
            except ValueError:
                pass
    return idx


def player_id_from_href(href):
    m = re.search(r"/player/(?:[a-z0-9-]*?-)?(\d+)", href or "")
    return int(m.group(1)) if m else None


def slug_from_href(href, pattern):
    m = re.search(pattern, href or "")
    return m.group(1) if m else None


def parse_time_daynight(text):
    m = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)", text or "", re.I)
    if not m:
        return None
    hour = int(m.group(1)) % 12 + (12 if m.group(3).upper() == "PM" else 0)
    return "day" if hour < 17 else "night"


# ------------------------------------------------------------------ mlb.com

def scrape_mlb():
    soup = fetch("https://www.mlb.com/starting-lineups")
    date_el = soup.find(class_=re.compile("date-title--current"))
    date = dt.date.today().isoformat()
    if date_el:
        m = re.search(r"(\w+) (\d+)\w*, (\d{4})", date_el.get_text(" ", strip=True))
        if m:
            date = dt.datetime.strptime(
                f"{m.group(1)} {m.group(2)} {m.group(3)}", "%B %d %Y"
            ).date().isoformat()

    games = []
    for g in soup.find_all(class_="starting-lineups__matchup"):
        def team(side):
            el = g.find(class_=f"starting-lineups__team-name--{side}")
            return NICKNAME_TO_ABBREV.get(el.get_text(strip=True)) if el else None

        away, home = team("away"), team("home")
        if not away or not home:
            continue
        loc = g.find(class_="starting-lineups__game-location")
        venue = loc.get_text(" ", strip=True) if loc else ""
        venue = VENUE_ALIASES.get(venue, venue)
        t = g.find(class_="starting-lineups__game-date-time")
        day_night = parse_time_daynight(t.get_text(" ", strip=True) if t else "")

        names = {}  # pid -> display name, straight from the page links

        pitchers, seen = [], set()
        for p in g.find_all(class_="starting-lineups__pitcher-name"):
            a = p.find("a")
            pid = player_id_from_href(a["href"]) if a else None
            if pid:
                names[str(pid)] = a.get_text(strip=True)
            key = pid or p.get_text(strip=True)
            if key not in seen:
                seen.add(key)
                pitchers.append(pid)
        pitchers += [None, None]

        def lineup(side):
            ol = g.find("ol", class_=f"starting-lineups__team--{side}")
            if not ol:
                return []
            pids = []
            for li in ol.find_all("li"):
                a = li.find("a", href=re.compile("/player/"))
                if not a:
                    continue
                pid = player_id_from_href(a["href"])
                if pid:
                    pids.append(pid)
                    names[str(pid)] = a.get_text(strip=True)
            return [[pid, slot] for slot, pid in enumerate(pids, start=1)][:9]

        games.append({
            "date": date, "away_team": away, "home_team": home,
            "venue": venue, "day_night": day_night,
            "away_starter": pitchers[0], "home_starter": pitchers[1],
            "away_lineup": lineup("away"), "home_lineup": lineup("home"),
            "names": names,
            "temp": None, "wind_speed": None, "wind_dir": "", "condition": "",
        })
    return games


# ------------------------------------------------------------- fantasypros

def scrape_fantasypros():
    """(away, home) -> weather + full-name lineups."""
    soup = fetch("https://www.fantasypros.com/mlb/lineups/")
    name_map = full_name_to_abbrev()
    out = {}
    for box in soup.find_all(class_="gamebox"):
        names = []
        for el in box.find_all(["h2", "h3", "a", "span"]):
            ab = name_map.get(el.get_text(strip=True))
            if ab and ab not in names:
                names.append(ab)
            if len(names) == 2:
                break
        if len(names) != 2:
            continue
        away, home = names

        temp = None
        gd = box.find(class_="game-details")
        if gd:
            m = re.search(r"Gametime Temp:\s*(\d+)", gd.get_text(" ", strip=True))
            temp = float(m.group(1)) if m else None
        mph = None
        m = box.find(string=re.compile("mph"))
        if m:
            n = m.find_parent("p")
            mm = re.search(r"(\d+)", n.get_text()) if n else None
            mph = float(mm.group(1)) if mm else None
        wind_dir = ""
        line = box.find("line")
        if line and line.get("transform"):
            mm = re.search(r"rotate\((-?\d+(?:\.\d+)?)", line["transform"])
            if mm:
                angle = float(mm.group(1)) % 360
                wind_dir = WIND_SECTORS[round(angle / 45) % 8]
        if mph == 0:
            wind_dir = "Calm"

        lineups = box.select(".team-lineup")
        def names_of(container):
            slugs = []
            for a in container.find_all("a", href=re.compile(r"/mlb/players/")):
                slug = slug_from_href(a["href"], r"/mlb/players/([a-z0-9-]+)")
                if slug:
                    slugs.append(slug)
            return slugs[:9]
        away_l = names_of(lineups[0]) if len(lineups) >= 1 else []
        home_l = names_of(lineups[1]) if len(lineups) >= 2 else []

        out[(away, home)] = {"wind_speed": mph, "wind_dir": wind_dir,
                             "temp": temp, "away_lineup": away_l,
                             "home_lineup": home_l}
    return out


# --------------------------------------------------------------- rotowire

def scrape_rotowire():
    """(away, home) -> temp, condition, full-name lineups."""
    soup = fetch("https://www.rotowire.com/baseball/daily-lineups.php")
    out = {}
    for g in soup.select(".lineup.is-mlb"):
        abbrs = [ABBREV_ALIASES.get(x.get_text(strip=True), x.get_text(strip=True))
                 for x in g.select(".lineup__abbr")]
        if len(abbrs) != 2:
            continue
        away, home = abbrs
        temp = None
        wx = g.select_one(".lineup__weather-text")
        if wx:
            m = re.search(r"(\d+)\s*°", wx.get_text(" ", strip=True))
            temp = float(m.group(1)) if m else None
        tm = g.select_one(".lineup__time")
        day_night = parse_time_daynight(tm.get_text(strip=True) if tm else "")
        # home-plate umpire: name is the <a> inside .lineup__umpire (falls
        # back to a regex on the div text: "Umpire: <name> 9.1 R/G ...")
        ump_a = g.select_one(".lineup__umpire a")
        umpire = ump_a.get_text(strip=True) if ump_a else None
        if not umpire:
            ud = g.select_one(".lineup__umpire")
            m = re.search(r"Umpire:\s*(.+?)\s+[\d.]+\s*R/G",
                          ud.get_text(" ", strip=True)) if ud else None
            umpire = m.group(1).strip() if m else None
        condition = ""
        icon = g.select_one(".lineup__weather-icon")
        alt = (icon.get("alt") or "").lower() if icon else ""
        for frag, cond in CONDITION_MAP.items():
            if frag in alt:
                condition = cond
                break

        def names_of(sel):
            lst = g.select_one(f".lineup__list.{sel}")
            if not lst:
                return []
            slugs = []
            for a in lst.select(".lineup__player a"):
                slug = slug_from_href(a.get("href"),
                                      r"/baseball/player/([a-z0-9-]+?)-\d+")
                if slug:
                    slugs.append(slug)
            return slugs[:9]

        out[(away, home)] = {"temp": temp, "condition": condition,
                             "day_night": day_night, "umpire": umpire,
                             "away_lineup": names_of("is-visit"),
                             "home_lineup": names_of("is-home")}
    return out


# --------------------------------------------------------------- assembly

def resolve_lineup(slugs, team, idx):
    """Resolve posted names to pids, keeping each player's slot = his spot in
    the posted batting order. An unresolved name leaves that slot empty rather
    than shifting everyone up, so the GUI shows exactly which slot to fill by
    hand and the other eight stay in their real positions."""
    out, used = [], set()
    for slot, slug in enumerate(slugs, start=1):
        pid = resolve(slug, team, idx)
        if pid and pid not in used:
            used.add(pid)
            out.append([pid, slot])
    return out


def classify_side(before, after):
    """Classify one lineup side by diffing the pids mlb.com supplied (`before`)
    against the pids after fallbacks were applied (`after`). Reliance is judged
    by an actual player change, NOT by reaching a full 9 — a fallback that
    fills only a partial lineup still counts. Returns
    (display_tag, contributed, fully, topped, completed) where the four flags
    are 0/1 counters:
      contributed - a fallback added/changed at least one player
      fully       - the whole side came from a fallback (mlb.com had none)
      topped      - a fallback added to a partial mlb.com lineup
      completed   - a fallback brought the side from <9 up to a full 9
    """
    before, after = set(before), set(after)
    contributed = bool(after) and after != before
    fully = contributed and not before
    topped = contributed and bool(before)
    completed = contributed and len(after) == 9
    if not after:
        tag = "none"
    elif not contributed:
        tag = "mlb"
    else:
        tag = "full" if fully else "top"
    return tag, int(contributed), int(fully), int(topped), int(completed)


def lineup_report(games, sources, lineup_src):
    """Human-readable fallback-accounting lines. Split out from main() so the
    reconciliation can be tested without a live scrape.

    'fully sourced' (mlb.com posted none for that side) and 'completed to 9'
    are DIFFERENT axes, so they can differ: a side can be fully fallback-sourced
    yet still under 9 when even the fallback posted fewer than 9 resolvable
    names. This spells that out and lists every side that ended under 9, tagged
    with where it came from (mlb / full / top)."""
    sides = 2 * len(games)
    under9 = lineup_src["contributed"] - lineup_src["completed"]
    lines = [
        f"  fallback: contributed to {lineup_src['contributed']}/{sides} "
        f"side(s) ({lineup_src['fully']} fully sourced, "
        f"{lineup_src['topped']} topped up); of those, "
        f"{lineup_src['completed']} reached a full 9, {under9} still under 9"
    ]
    incomplete = [(g, side, src[side]) for g, src in zip(games, sources)
                  for side in ("away", "home")
                  if len(g[f"{side}_lineup"]) < 9]
    if incomplete:
        lines.append(f"  incomplete lineups — {len(incomplete)} side(s) under "
                     f"9 batters (tag = source):")
        for g, side, tag in incomplete:
            lines.append(f"      {g['away_team']:>3} @ {g['home_team']:<3}  "
                         f"{side:<4} {g[f'{side}_team']:>3}  "
                         f"{len(g[f'{side}_lineup'])}/9  ({tag})")
    return lines


def main():
    print("scraping mlb.com/starting-lineups ...")
    games = scrape_mlb()
    print(f"  {len(games)} games")
    try:
        print("scraping fantasypros (wind, temp fallback, lineup fallback) ...")
        fp = scrape_fantasypros()
        print(f"  data for {len(fp)} games")
    except Exception as e:
        print(f"  fantasypros failed ({e})", file=sys.stderr)
        fp = {}
    try:
        print("scraping rotowire (temp, condition, lineup fallback) ...")
        roto = scrape_rotowire()
        print(f"  data for {len(roto)} games")
    except Exception as e:
        print(f"  rotowire failed ({e})", file=sys.stderr)
        roto = {}

    idx = build_name_index()
    ump_idx = build_umpire_index()
    filled = {"temp": 0, "wind": 0, "cond": 0, "dome": 0, "ump": 0}
    # Lineup-source accounting, per TEAM-SIDE (there are 2 per game), decided
    # by comparing the pids mlb.com gave against the pids after fallbacks —
    # NOT by whether a side reached a full 9 (a fallback that fills in only a
    # partial lineup still "relied on the fallback", and the old count missed
    # exactly that):
    #   contributed - a fallback added/changed at least one player
    #   fully       - the whole side came from a fallback (mlb.com had none)
    #   topped      - a fallback added to a partial mlb.com lineup
    #   completed   - a fallback brought the side from <9 up to a full 9
    lineup_src = {"contributed": 0, "fully": 0, "topped": 0, "completed": 0}
    sources = []   # per-game {"away": tag, "home": tag}, aligned to `games`

    for g in games:
        key = (g["away_team"], g["home_team"])
        f, r = fp.get(key, {}), roto.get(key, {})

        # weather: rotowire temp/condition, fantasypros wind; FP temp as backup
        g["temp"] = r.get("temp") or f.get("temp")
        g["condition"] = r.get("condition") or ""
        g["wind_speed"] = f.get("wind_speed")
        g["wind_dir"] = f.get("wind_dir") or ""
        if not g["day_night"]:  # fall back to rotowire's game time
            g["day_night"] = r.get("day_night")

        # home-plate umpire (rotowire name -> HpUmpId the model uses); the
        # GUI loads hp_ump_id into the spec so ump_feats has real history
        g["hp_ump"] = r.get("umpire")
        g["hp_ump_id"] = (ump_idx.get(norm_name(g["hp_ump"]))
                          if g.get("hp_ump") else None)
        if g["hp_ump_id"] is not None:
            filled["ump"] += 1

        # roof closed: sources report no weather for indoor games
        if g["venue"] in DOME_VENUES and g["temp"] is None and \
                g["wind_speed"] is None:
            g.update(INDOOR)
            filled["dome"] += 1
        if g["temp"] is not None:
            filled["temp"] += 1
        if g["wind_speed"] is not None:
            filled["wind"] += 1
        if g["condition"]:
            filled["cond"] += 1

        # lineup fallback: fantasypros first, then rotowire. Record what the
        # fallback actually did per side by diffing pids before vs. after,
        # so partial fills (never reaching 9) are still counted as reliance.
        src_tags = {}
        for side, team in (("away", g["away_team"]), ("home", g["home_team"])):
            before = [pid for pid, _ in g[f"{side}_lineup"]]
            if len(before) < 9:
                for src in (f, r):
                    cand = resolve_lineup(src.get(f"{side}_lineup", []), team, idx)
                    if len(cand) > len(g[f"{side}_lineup"]):
                        g[f"{side}_lineup"] = cand
                    if len(g[f"{side}_lineup"]) == 9:
                        break
            after = [pid for pid, _ in g[f"{side}_lineup"]]
            tag, contributed, fully, topped, completed = classify_side(before, after)
            src_tags[side] = tag
            lineup_src["contributed"] += contributed
            lineup_src["fully"] += fully
            lineup_src["topped"] += topped
            lineup_src["completed"] += completed
        sources.append(src_tags)

    payload = {"scraped_at": dt.datetime.now().isoformat(timespec="seconds"),
               "date": games[0]["date"] if games else None,
               "games": games}
    with open(OUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)

    posted = sum(1 for g in games if len(g["away_lineup"]) == 9
                 and len(g["home_lineup"]) == 9)
    print(f"\nwrote {len(games)} games to {OUT_FILE}")
    print(f"  both lineups set: {posted}/{len(games)} games")
    for line in lineup_report(games, sources, lineup_src):
        print(line)
    print(f"  weather: temp {filled['temp']}/{len(games)}, "
          f"wind {filled['wind']}/{len(games)}, "
          f"condition {filled['cond']}/{len(games)}, "
          f"dome defaults {filled['dome']}")
    print(f"  home-plate umpire resolved: {filled['ump']}/{len(games)}")
    for g, src in zip(games, sources):
        print(f'  {g["away_team"]:>3} @ {g["home_team"]:<3} {g["venue"]:<28} '
              f'{g["day_night"] or "?":<5} temp {str(g["temp"]) or "?":>5}  '
              f'wind {str(g["wind_speed"]) or "?":>4} {g["wind_dir"] or "-":<10} '
              f'{g["condition"] or "-":<14} '
              f'lineups {len(g["away_lineup"])}({src["away"]})+'
              f'{len(g["home_lineup"])}({src["home"]})')


if __name__ == "__main__":
    main()
