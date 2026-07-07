"""Capture real sportsbook lines into Data/mlb_odds.csv (canonical schema in
Model/odds.py), so the model can finally be graded against the market instead
of only a naive base rate (evaluate_deep Section 9).

Source: The Odds API (https://the-odds-api.com). Set an API key once:

    setx ODDS_API_KEY "your_key_here"      # Windows, new shells pick it up
    # or pass --key on the command line

Then run it near game time to snapshot that day's closing-ish lines:

    python Scripts/scrape_odds.py                        # today, home runs only (default)
    python Scripts/scrape_odds.py --markets all          # every prop + totals + h2h
    python Scripts/scrape_odds.py --markets hr,bb,totals --date 2026-07-04

IMPORTANT on the free tier: it covers moneyline/totals for current & upcoming
games, but player props ("additional markets") and historical snapshots are
paid add-ons. So the honest workflow is GOING FORWARD capture — run this daily
and you accumulate your own closing-line history; Section 9 lights up over the
games you've collected. A purchased historical dump or a scrape from elsewhere
also works: just write rows in the Model/odds.py schema to the same CSV.
Each requested market costs one credit PER GAME, so --markets defaults to home
runs only; use --markets all for everything, or a custom list to fit the quota.

Without a key the script explains this and exits 0 (nothing to do), so it is
safe to wire into a scheduler before you have props access.
"""

import argparse
import csv
import datetime as dt
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import odds as O  # noqa: E402  (canonical schema + market map)
from get_todays_games import (NICKNAME_TO_ABBREV, build_name_index,  # noqa: E402
                              norm_name)

API = "https://api.the-odds-api.com/v4/sports/baseball_mlb"
PROP_APIS = sorted({m["api"] for m in O.PROP_MARKET.values()})
GAME_MARKETS = ["totals", "h2h"]           # over/under total runs; moneyline

# Each market costs 1 credit PER GAME per region (a 14-game slate x 9 markets =
# 126 of a 500/month free quota). Default is home runs only (~14 credits/slate,
# so 500/month covers ~35 daily runs); pass --markets all, or a custom list
# like hr,bb,sb,totals, for more.
DEFAULT_MARKETS = "hr"


def resolve_markets(spec):
    """'hr,bb,totals' (our prop keys + game markets) or 'all' -> the ordered,
    de-duped list of Odds API market keys to request."""
    valid = list(O.PROP_MARKET) + GAME_MARKETS
    keys = (valid if spec.strip().lower() == "all"
            else [k.strip() for k in spec.split(",") if k.strip()])
    unknown = [k for k in keys if k not in valid]
    if unknown:
        raise SystemExit(f"unknown market(s): {', '.join(unknown)}. "
                         f"valid: {', '.join(valid)}, or 'all'.")
    apis, seen = [], set()
    for k in keys:
        api = O.PROP_MARKET[k]["api"] if k in O.PROP_MARKET else k
        if api not in seen:
            seen.add(api)
            apis.append(api)
    return apis


def full_name_to_abbrev(full):
    """The Odds API uses full club names ('New York Yankees'); map to our
    abbreviations by matching the nickname suffix ('Yankees', 'Red Sox')."""
    for nick, ab in NICKNAME_TO_ABBREV.items():
        if full == nick or full.endswith(" " + nick) or full.endswith(nick):
            return ab
    return None


def _side(name):
    """Normalize an outcome label to 'over'/'under' (props use Over/Under or
    Yes/No; both mean the same 'does it clear the line' bet)."""
    n = (name or "").strip().lower()
    if n in ("over", "yes"):
        return "over"
    if n in ("under", "no"):
        return "under"
    return None


def make_resolver(idx, home_abbr, away_abbr):
    """name -> (PlayerId, team_abbr). An exact (club, name) hit on either team
    gives the player's REAL team; a globally-unique name still resolves the
    PlayerId but leaves the team unknown (None) rather than guessing it (the
    prop outcome doesn't say which club the player is on)."""
    by_team, by_name = idx
    def r(name):
        n = norm_name(name)
        for ab in (home_abbr, away_abbr):
            if ab and (ab, n) in by_team:
                return by_team[(ab, n)], ab
        pids = by_name.get(n)
        if pids and len(pids) == 1:
            return next(iter(pids)), None
        return None, None
    return r


def parse_event_props(event, resolver, date, gamepk, captured_at,
                      prop_apis=PROP_APIS):
    """Pure: an Odds API event JSON -> list of canonical prop rows (one per
    player/market/line/book, with matched over & under prices). Unresolved
    player names are skipped."""
    rows = []
    for bk in event.get("bookmakers", []):
        book = bk.get("key")
        for mkt in bk.get("markets", []):
            if mkt.get("key") not in prop_apis:
                continue
            market = mkt["key"]
            # group the market's outcomes by (player, line) -> {over, under}
            pairs = {}
            for o in mkt.get("outcomes", []):
                side = _side(o.get("name"))
                player = o.get("description")
                if side is None or not player:
                    continue
                line = o.get("point")
                if line is None:  # Yes/No HR markets omit the point
                    line = O.API_DEFAULT_LINE.get(market, 0.5)
                key = (player, float(line))
                pairs.setdefault(key, {})[side] = o.get("price")
            for (player, line), pr in pairs.items():
                pid, team = resolver(player)
                if pid is None:
                    continue
                rows.append({
                    "Date": date, "GamePk": gamepk, "Team": team,
                    "PlayerId": pid, "PlayerName": player, "Market": market,
                    "Line": line, "OverPrice": pr.get("over"),
                    "UnderPrice": pr.get("under"), "Book": book,
                    "CapturedAt": captured_at,
                })
    return rows


def parse_event_games(event, home_abbr, away_abbr, date, gamepk, captured_at):
    """Game-level totals (over/under runs) and moneyline (home vs away) rows,
    stored in the same schema: PlayerId blank, Team = home club. Totals ->
    OverPrice/UnderPrice; h2h -> OverPrice = home, UnderPrice = away."""
    rows = []
    for bk in event.get("bookmakers", []):
        book = bk.get("key")
        for mkt in bk.get("markets", []):
            k = mkt.get("key")
            outs = mkt.get("outcomes", [])
            if k == "totals":
                by = {_side(o.get("name")): o for o in outs}
                if "over" in by and "under" in by:
                    rows.append(_game_row(date, gamepk, home_abbr, "totals",
                                          by["over"].get("point"),
                                          by["over"].get("price"),
                                          by["under"].get("price"), book,
                                          captured_at))
            elif k == "h2h":
                price = {full_name_to_abbrev(o.get("name")): o.get("price")
                         for o in outs}
                rows.append(_game_row(date, gamepk, home_abbr, "h2h", None,
                                      price.get(home_abbr),
                                      price.get(away_abbr), book, captured_at))
    return rows


def _game_row(date, gamepk, team, market, line, over, under, book, captured):
    return {"Date": date, "GamePk": gamepk, "Team": team, "PlayerId": "",
            "PlayerName": "", "Market": market, "Line": line,
            "OverPrice": over, "UnderPrice": under, "Book": book,
            "CapturedAt": captured}


def fetch(url, params):
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:200]}")
    remain = r.headers.get("x-requests-remaining")
    return r.json(), remain


def write_store(rows, out):
    """Append rows, then de-dupe on (Date, PlayerId, Market, Line, Book)
    keeping the latest CapturedAt — so re-running closer to first pitch
    upgrades each line to its closing value."""
    out = Path(out)
    existing = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
    combined = existing + rows
    best = {}
    for row in combined:
        key = (str(row.get("Date")), str(row.get("PlayerId")),
               row.get("Market"), str(row.get("Line")), row.get("Book"))
        prev = best.get(key)
        if prev is None or str(row.get("CapturedAt")) >= str(prev.get("CapturedAt")):
            best[key] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=O.ODDS_COLUMNS)
        w.writeheader()
        for row in best.values():
            w.writerow({c: row.get(c, "") for c in O.ODDS_COLUMNS})
    return len(best)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="slate date YYYY-MM-DD (default today)")
    ap.add_argument("--key", default=os.environ.get("ODDS_API_KEY"))
    ap.add_argument("--regions", default="us")
    ap.add_argument("--out", default=str(O.DEFAULT_STORE))
    ap.add_argument("--markets", default=DEFAULT_MARKETS,
                    help="comma-separated markets: prop keys ("
                         + ", ".join(O.PROP_MARKET) + ") and/or totals, h2h; "
                         "or 'all'. Each costs 1 credit per game. Default: "
                         + DEFAULT_MARKETS)
    args = ap.parse_args()
    markets = resolve_markets(args.markets)
    prop_markets = [m for m in markets if m not in GAME_MARKETS]

    if not args.key:
        print("No ODDS_API_KEY found in this shell. Get a free key at "
              "https://the-odds-api.com, then either:\n"
              "  this session:  $env:ODDS_API_KEY = \"...\"   (PowerShell)\n"
              "  persistent:    setx ODDS_API_KEY \"...\"  then open a NEW terminal\n"
              "  one-off:       python Scripts/scrape_odds.py --key <key>\n"
              "(setx does NOT affect the shell you run it in.) Nothing to "
              "capture — exiting.")
        return  # exit 0: safe to schedule before you have a key

    captured_at = dt.datetime.now().isoformat(timespec="seconds")
    day = args.date
    try:
        events, remain = fetch(f"{API}/events", {"apiKey": args.key})
    except Exception as e:
        print(f"could not list events: {e}", file=sys.stderr)
        sys.exit(1)
    events = [e for e in events if str(e.get("commence_time", "")).startswith(day)]
    n_reg = len(args.regions.split(","))
    print(f"{len(events)} MLB events on {day} (api requests left: {remain})")
    print(f"markets [{', '.join(markets)}] x {len(events)} games"
          f"{f' x {n_reg} regions' if n_reg > 1 else ''} "
          f"= ~{len(markets) * len(events) * n_reg} credits")

    idx = build_name_index()
    all_rows, n_prop, n_game = [], 0, 0
    for e in events:
        home_abbr = full_name_to_abbrev(e.get("home_team", ""))
        away_abbr = full_name_to_abbrev(e.get("away_team", ""))
        resolver = make_resolver(idx, home_abbr, away_abbr)
        try:
            data, remain = fetch(
                f"{API}/events/{e['id']}/odds",
                {"apiKey": args.key, "regions": args.regions,
                 "markets": ",".join(markets), "oddsFormat": "american"})
        except Exception as ex:
            print(f"  {away_abbr}@{home_abbr}: odds fetch failed ({ex})",
                  file=sys.stderr)
            continue
        pr = parse_event_props(data, resolver, day, "", captured_at,
                               prop_apis=prop_markets)
        gm = parse_event_games(data, home_abbr, away_abbr, day, "", captured_at)
        all_rows += pr + gm
        n_prop += len(pr)
        n_game += len(gm)
        print(f"  {away_abbr}@{home_abbr}: {len(pr)} prop + {len(gm)} game "
              f"rows (left: {remain})")

    total = write_store(all_rows, args.out)
    print(f"\nwrote {n_prop} prop + {n_game} game rows this run; "
          f"store now holds {total} rows -> {args.out}")


if __name__ == "__main__":
    main()
