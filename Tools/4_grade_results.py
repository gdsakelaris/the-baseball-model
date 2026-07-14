"""Grade a predictions workbook against the actual box scores.

Re-colors the workbook IN PLACE once games are final, using the project's
own scraped per-player box scores (Data/mlb_game_batting.csv /
mlb_game_pitching.csv / mlb_games.csv) joined on the workbook's ID column —
PlayerId-exact, so no third-party box-score scraping or name matching:

  stat occurred + cell was plain (white / red tint)  -> yellow
  stat occurred + cell was a light-blue quality pick -> darker blue
  stat occurred + cell was a light-green +EV bet     -> darker green
  stat occurred + cell was light purple (blue + green) -> dark purple

On the Bets sheet, every WINNING bet has its whole row painted solid dark
green (#00B050); losing / not-yet-settled rows keep the light board. Under
bets settle too (the row carries its Side). Re-running is idempotent.

Every graded cell answers the same literal question: DID THE STAT OCCUR.
Binary batter columns light up if the event happened (the HR cell if he
homered); O/U line columns light up if the OVER hit (K > 8.5 only if he
actually struck out 9+); the Winner cell (and its Win Prob) if the named
team won. Mean columns (xK, xTB, Away/Home Score, ...) have no yes/no
event and are left untouched.

Re-running is safe and REPAIRS earlier grades: pass 1 reverts every
occurred-color cell back to its base color (magenta/yellow -> plain, dark blue/
green/purple -> the light pick shade), then pass 2 grades fresh.

Doubleheaders: a batter's stats are summed across the day's games (the
slate carries one row per matchup, so the day total is the honest read).
Games-sheet rows are matched per game: the tag's i-th row grades against
the day's i-th final for that matchup (schedule order); if only one of
the two games is final the tag's rows are skipped until both are in.

If some games are missing (they weren't final at the last scrape), their
rows are skipped and counted — run  python Scrapers/scrape_gamelogs_3F.py
to pull the late finals, then grade again.

Usage:
    python Tools/4_grade_results.py                  # newest in Predictions/
    python Tools/4_grade_results.py path\\to\\file.xlsx
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
PRED_DIR = Path(__file__).resolve().parent.parent / "Predictions"

# The visual grammar: LIGHT color = pick pending, DARK fill + white bold =
# pick HIT, pale-gray fill + gray italic = pick MISSED, yellow = a stat
# occurred with no pick on it, white = nothing predicted, nothing happened.
BLUE, GREEN, PURPLE = "00B0F0", "92D050", "B1A0C7"
# earlier palettes, still recognized when repairing old workbooks
OLD_BASES = {"9DC3E6": "00B0F0", "C6E0B4": "92D050",
             "C6EFCE": "92D050", "CCC0DA": "B1A0C7"}
# fill + text color; per-cell font weight follows the bold-over-50% rule
# (a HIT is never bolded unless its stated probability was above 50%)
OCCURRED = {
    BLUE:   (PatternFill("solid", fgColor="0070C0"), "FFFFFF"),  # dark blue
    GREEN:  (PatternFill("solid", fgColor="00B050"), "FFFFFF"),  # dark green
    PURPLE: (PatternFill("solid", fgColor="7030A0"), "FFFFFF"),  # dark purple
    None:   (PatternFill("solid", fgColor="FFFF00"), "000000"),  # yellow
}
# a pick that did NOT hit: grayed out (each fill keeps a faint machine-
# readable hue so re-grading can restore the base color; to the eye they
# all read "gray = miss"). Font is per-cell: probabilities over 50% stay
# bold even in a miss.
MISSED = {
    BLUE:   PatternFill("solid", fgColor="B8CCE4"),
    GREEN:  PatternFill("solid", fgColor="C4D79B"),
    PURPLE: PatternFill("solid", fgColor="D2C8DE"),
}

# Bets sheet: a WINNING bet paints its whole row solid dark green; a losing
# or not-yet-settled row shows the light-green board. #00B050 is the same
# dark green a HIT +EV cell gets on the prop grids — one "good outcome" hue.
BETS_WIN = PatternFill("solid", fgColor="00B050")      # winning bet row
BETS_BOARD = PatternFill("solid", fgColor="E7F3E2")    # the light-green board
BET_PCT_COLS = {"Model %", "Mkt %", "Edge", "EV%"}     # bold over 50% (as _polish)
# Bets 'Prop' label (odds.PROP_MARKET / STARTER_MARKET) -> how to settle it.
# Batter labels map to the BAT_EVENTS check above; pitcher labels prefix a
# "... o<line>" string and settle off the row's own Line column.
BET_LABEL_TO_BATCOL = {
    "1+ HR": "HR", "1+ hit": "Hit", "2+ hits": "2+ Hits",
    "2+ total bases": "2+ TB", "run scored": "Run", "1+ RBI": "RBI",
    "1+ walk": "BB", "stolen base": "SB", "1+ single": "Single",
    "1+ double": "Double", "1+ batter K": "K", "2+ batter K": "2+ K",
    "2+ H+R+RBI": "H+R+RBI 2+", "3+ H+R+RBI": "H+R+RBI 3+",
}
BET_PIT_STAT = {"pitcher strikeouts": "SO", "pitcher outs": "outs",
                "pitcher hits allowed": "H", "pitcher walks": "BB",
                "pitcher earned runs": "ER"}

# batter columns -> did the event happen, from the day's summed line
BAT_EVENTS = {
    "HR":          lambda s: s["HR"] >= 1,
    "Hit":         lambda s: s["H"] >= 1,
    "2+ Hits":     lambda s: s["H"] >= 2,
    "Single":      lambda s: (s["H"] - s["2B"] - s["3B"] - s["HR"]) >= 1,
    "Double":      lambda s: s["2B"] >= 1,
    "2+ TB":       lambda s: s["TB"] >= 2,
    "Run":         lambda s: s["R"] >= 1,
    "RBI":         lambda s: s["RBI"] >= 1,
    "H+R+RBI 2+":  lambda s: (s["H"] + s["R"] + s["RBI"]) >= 2,
    "H+R+RBI 3+":  lambda s: (s["H"] + s["R"] + s["RBI"]) >= 3,
    "BB":          lambda s: s["BB"] >= 1,
    "SB":          lambda s: s["SB"] >= 1,
    "K":           lambda s: s["SO"] >= 1,
    "2+ K":        lambda s: s["SO"] >= 2,
}
BAT_SUM_COLS = ["PA", "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "SO",
                "SB", "TB"]

# pitcher O/U column pattern -> the actual-stat key it grades
LINE_RE = re.compile(r"^(K|Outs|Hits|BB|ER) > (\d+(?:\.\d+)?)$")
LINE_STAT = {"K": "SO", "Outs": "outs", "Hits": "H", "BB": "BB", "ER": "ER"}
RUNS_RE = re.compile(r"^Runs > (\d+(?:\.\d+)?)$")


def ip_to_outs(ip):
    """'5.2' -> 17 outs (MLB notation: .1/.2 = thirds)."""
    ip = float(ip)
    whole = int(ip)
    return whole * 3 + round((ip - whole) * 10)


def load_actuals(date):
    """(batters {pid: summed Series}, starters {pid: dict},
    games {'AWY@HOM': [dict, ...]}) for one date, from the scraped logs.
    The games lists keep mlb_games.csv row order (the schedule's game
    order), so a doubleheader's game 1 is entry 0 and game 2 entry 1."""
    gb = pd.read_csv(DATA_DIR / "mlb_game_batting.csv", encoding="utf-8-sig",
                     low_memory=False)
    gb = gb[gb["Date"] == date]
    batters = {int(pid): grp[BAT_SUM_COLS].sum()
               for pid, grp in gb.groupby("PlayerId")}

    gp = pd.read_csv(DATA_DIR / "mlb_game_pitching.csv", encoding="utf-8-sig",
                     low_memory=False)
    gp = gp[(gp["Date"] == date) & (gp["GS"] == 1)]
    starters = {}
    for pid, grp in gp.groupby("PlayerId"):
        r = grp.iloc[0]
        starters[int(pid)] = {"SO": float(r["SO"]), "H": float(r["H"]),
                              "BB": float(r["BB"]), "ER": float(r["ER"]),
                              "outs": ip_to_outs(r["IP"])}

    g = pd.read_csv(DATA_DIR / "mlb_games.csv", encoding="utf-8-sig")
    g = g[g["Date"] == date].dropna(subset=["AwayScore", "HomeScore"])
    games = {}
    for _, r in g.iterrows():
        a, h = float(r["AwayScore"]), float(r["HomeScore"])
        games.setdefault(f'{r["AwayTeam"]}@{r["HomeTeam"]}', []).append({
            "total": a + h,
            "winner": r["HomeTeam"] if h > a else r["AwayTeam"]})
    return batters, starters, games


# graded-color -> the base to restore on re-grade (yellow was plain).
# Covers tonight's palette AND the earlier one, so previously graded
# workbooks repair cleanly.
UNGRADE = {"0070C0": BLUE, "2E75B6": BLUE,          # hits (new, old)
           "00B050": GREEN, "538135": GREEN, "548235": GREEN,
           "7030A0": PURPLE,
           "FFFF00": None, "CB21C3": None, "FFD966": None, "FFE699": None,
           "B8CCE4": BLUE, "DCE6F1": BLUE, "D8E4BC": GREEN, "C4D79B": GREEN,
           "D2C8DE": PURPLE,
           "E8EDF3": BLUE, "EAF0E4": GREEN, "EDE8F3": PURPLE,
           "9DC3E6": BLUE, "C6E0B4": GREEN,          # old bases
           "C6EFCE": GREEN, "CCC0DA": PURPLE,
           "F5DBE2": None}   # the retired red column tint -> plain white
# base fill + its tinted text color (None = plain cell, default font)
BASE_FILL = {
    BLUE:   (PatternFill("solid", fgColor=BLUE), "0B2E4F"),
    GREEN:  (PatternFill("solid", fgColor=GREEN), "1E4620"),
    PURPLE: (PatternFill("solid", fgColor=PURPLE), "3B2151"),
    None:   (PatternFill(fill_type=None), None),
}


def _ungrade(ws):
    """Revert every occurred-color cell to its base color so grading is
    idempotent (and repairs workbooks graded by the old side-of-the-line
    logic)."""
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            try:
                rgb = cell.fill.start_color.rgb
            except AttributeError:
                continue
            if isinstance(rgb, str) and rgb[-6:].upper() in UNGRADE:
                base = UNGRADE[rgb[-6:].upper()]
                fill, color = BASE_FILL[base]
                cell.fill = fill
                # probabilities follow the bold-over-50% rule everywhere;
                # text cells on a pick fill (e.g. Winner) stay bold
                num = isinstance(cell.value, (int, float))
                bold = (cell.value > 0.5) if num else (base is not None)
                cell.font = (Font(bold=bold, color=color) if color
                             else Font(bold=bold))


def _base_color(cell):
    """The pick color a cell was painted with, or None for plain."""
    try:
        rgb = cell.fill.start_color.rgb
    except AttributeError:
        return None
    if not isinstance(rgb, str):
        return None
    tail = rgb[-6:].upper()
    tail = OLD_BASES.get(tail, tail)
    return tail if tail in (BLUE, GREEN, PURPLE) else None


def _mark(cell, occurred):
    """HIT -> the dark fill; MISS on a pick -> grayed out; MISS on a
    plain cell -> untouched. Font weight follows the bold-over-50% rule
    in EVERY case (hit or miss); text cells (e.g. Winner) stay bold on
    fills for contrast."""
    base = _base_color(cell)
    num = isinstance(cell.value, (int, float))
    bold = bool(cell.value > 0.5) if num else True
    if occurred:
        fill, color = OCCURRED[base]
        cell.fill = fill
        cell.font = Font(bold=bold, color=color)
        return
    if base is None:
        return
    cell.fill = MISSED[base]
    cell.font = Font(italic=True, color="7F7F7F", bold=num and bold)




def _name_pid(ws):
    """{Name: pid} and {(Game, Name): pid} from a graded prop sheet, so a
    Bets row (which carries the player NAME, not the ID) can be settled off
    the same PlayerId-keyed actuals the prop grids use. (Game, Name) wins
    when present; plain Name is the single-game fallback."""
    hidx = {str(c.value): j for j, c in enumerate(ws[1], start=1)}
    byname, bygame = {}, {}
    if "Name" not in hidx or "ID" not in hidx:
        return byname, bygame
    gj = hidx.get("Game")
    for i in range(2, ws.max_row + 1):
        nm = ws.cell(row=i, column=hidx["Name"]).value
        pid = ws.cell(row=i, column=hidx["ID"]).value
        if nm is None or pid is None:
            continue
        byname[str(nm)] = int(pid)
        if gj is not None:
            bygame[(str(ws.cell(row=i, column=gj).value), str(nm))] = int(pid)
    return byname, bygame


def _settle_bet(row, batters, starters, games, bat_pid, pit_pid):
    """Did this Bets row win? True / False / None (can't settle yet: no
    final, unmatched player, or a doubleheader game bet we can't pin to one
    game from an EV-sorted board). `row` is {header: value}; `bat_pid` /
    `pit_pid` resolve a (game, name) to a PlayerId."""
    game, prop = str(row.get("Game", "")), str(row.get("Prop", ""))
    side, line = str(row.get("Side", "")), row.get("Line")

    def _line():
        try:
            return float(line)
        except (TypeError, ValueError):
            return None

    if prop == "moneyline":                     # Side is the picked team
        finals = games.get(game, [])
        return finals[0]["winner"] == side if len(finals) == 1 else None
    if prop == "total runs":
        finals, ln = games.get(game, []), _line()
        if len(finals) != 1 or ln is None:
            return None
        occ = finals[0]["total"] > ln
        return occ if side == "Over" else not occ
    if prop in BET_LABEL_TO_BATCOL:
        pid = bat_pid(game, row.get("Player"))
        s = batters.get(pid) if pid is not None else None
        if s is None:
            return None
        occ = bool(BAT_EVENTS[BET_LABEL_TO_BATCOL[prop]](s))
        return occ if side == "Over" else not occ
    for lbl, stat in BET_PIT_STAT.items():      # "pitcher strikeouts o6.5"
        if prop.startswith(lbl):
            pid = pit_pid(game, row.get("Player"))
            a = starters.get(pid) if pid is not None else None
            ln = _line()
            if a is None or ln is None:
                return None
            occ = a[stat] > ln
            return occ if side == "Over" else not occ
    return None


def _grade_bets(wb, batters, starters, games, stats):
    """Paint every WINNING bet row solid #00B050; reset the rest to the
    light board first, so re-running is idempotent (the Bets sheet is
    skipped by _ungrade for exactly this reason). No-op on the 'no bets'
    note sheet (it has no Game/Prop/Side columns)."""
    if "Bets" not in wb.sheetnames:
        return
    ws = wb["Bets"]
    headers = [str(c.value) for c in ws[1]]
    hidx = {h: j for j, h in enumerate(headers, start=1)}
    if not {"Game", "Prop", "Side"} <= set(hidx):
        return                                  # the "No bets to show." note
    bp_name, bp_game = (_name_pid(wb["Batter Props"])
                        if "Batter Props" in wb.sheetnames else ({}, {}))
    pp_name, pp_game = (_name_pid(wb["Pitching Props"])
                        if "Pitching Props" in wb.sheetnames else ({}, {}))

    def bat_pid(g, nm):
        nm = None if nm is None else str(nm)
        return bp_game.get((g, nm)) or bp_name.get(nm)

    def pit_pid(g, nm):
        nm = None if nm is None else str(nm)
        return pp_game.get((g, nm)) or pp_name.get(nm)

    ncol = ws.max_column
    for i in range(2, ws.max_row + 1):
        # reset to board (undo any earlier win paint) — bold-over-50% on the
        # percent columns, matching predict._polish
        for j, h in enumerate(headers, start=1):
            c = ws.cell(row=i, column=j)
            c.fill = BETS_BOARD
            v = c.value
            c.font = Font(bold=(h in BET_PCT_COLS
                                and isinstance(v, (int, float)) and v > 0.5))
        row = {h: ws.cell(row=i, column=hidx[h]).value for h in headers}
        won = _settle_bet(row, batters, starters, games, bat_pid, pit_pid)
        stats["bets"] = stats.get("bets", 0) + 1
        if won:
            stats["bets_won"] = stats.get("bets_won", 0) + 1
            for j in range(1, ncol + 1):
                c = ws.cell(row=i, column=j)
                c.fill = BETS_WIN
                c.font = Font(bold=True, color="FFFFFF")


def grade(path):
    m = re.match(r"(\d{4}-\d{2}-\d{2})", Path(path).stem)
    if not m:
        sys.exit(f"can't read the game date from the filename: {path}")
    date = m.group(1)
    batters, starters, games = load_actuals(date)
    if not batters:
        sys.exit(f"no box scores for {date} in Data/mlb_game_batting.csv — "
                 f"run  python Scrapers/scrape_gamelogs_3F.py  first")

    wb = openpyxl.load_workbook(path)
    for ws in wb.worksheets:
        if ws.title == "Bets":
            continue          # graded by _grade_bets (whole-row), not _ungrade
        _ungrade(ws)
    stats = {"cells": 0, "hit": 0, "missing_rows": 0}

    def headers_of(ws):
        return {str(c.value): j for j, c in enumerate(ws[1], start=1)}

    if "Batter Props" in wb.sheetnames:
        ws = wb["Batter Props"]
        hidx = headers_of(ws)
        cols = {h: j for h, j in hidx.items() if h in BAT_EVENTS}
        for i in range(2, ws.max_row + 1):
            pid = ws.cell(row=i, column=hidx["ID"]).value
            s = batters.get(int(pid)) if pid is not None else None
            if s is None:
                stats["missing_rows"] += 1
                continue
            for h, j in cols.items():
                stats["cells"] += 1
                occ = bool(BAT_EVENTS[h](s))
                stats["hit"] += occ
                _mark(ws.cell(row=i, column=j), occ)

    if "Pitching Props" in wb.sheetnames:
        ws = wb["Pitching Props"]
        hidx = headers_of(ws)
        line_cols = [(h, j, LINE_RE.match(h)) for h, j in hidx.items()
                     if LINE_RE.match(h)]
        for i in range(2, ws.max_row + 1):
            pid = ws.cell(row=i, column=hidx["ID"]).value
            a = starters.get(int(pid)) if pid is not None else None
            if a is None:
                stats["missing_rows"] += 1
                continue
            for h, j, mm in line_cols:
                stats["cells"] += 1
                occ = a[LINE_STAT[mm.group(1)]] > float(mm.group(2))
                stats["hit"] += occ
                _mark(ws.cell(row=i, column=j), occ)

    if "Games" in wb.sheetnames:
        ws = wb["Games"]
        hidx = headers_of(ws)
        run_cols = [(j, float(RUNS_RE.match(h).group(1)))
                    for h, j in hidx.items() if RUNS_RE.match(h)]
        # Doubleheaders: the same "AWY@HOM" tag appears once per game, in
        # start-time order, and the finals list keeps the schedule's game
        # order — so match the sheet's i-th row for a tag to the i-th final.
        # If the finals don't yet cover every predicted game of the tag
        # (game 2 not final at the last scrape), skip the tag's rows rather
        # than grade two predictions against one game.
        need = Counter(str(ws.cell(row=i, column=hidx["Game"]).value)
                       for i in range(2, ws.max_row + 1))
        seen = Counter()
        for i in range(2, ws.max_row + 1):
            tag = str(ws.cell(row=i, column=hidx["Game"]).value)
            finals = games.get(tag, [])
            k = seen[tag]
            seen[tag] += 1
            g = finals[k] if len(finals) == need[tag] else None
            if g is None:
                stats["missing_rows"] += 1
                continue
            if "Winner" in hidx:
                cell = ws.cell(row=i, column=hidx["Winner"])
                stats["cells"] += 1
                occ = str(cell.value) == g["winner"]
                stats["hit"] += occ
                _mark(cell, occ)
                # Win Prob belongs to the named winner -> same outcome
                if "Win Prob" in hidx:
                    _mark(ws.cell(row=i, column=hidx["Win Prob"]), occ)
            for j, line in run_cols:
                stats["cells"] += 1
                occ = g["total"] > line
                stats["hit"] += occ
                _mark(ws.cell(row=i, column=j), occ)

    _grade_bets(wb, batters, starters, games, stats)

    try:
        wb.save(path)
    except PermissionError:
        sys.exit(f"{path} is open in Excel (it holds the file lock) — "
                 f"close it there, then run this again.")
    return date, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workbook", nargs="?", default=None,
                    help="predictions .xlsx (default: newest in Predictions/)")
    args = ap.parse_args()
    path = args.workbook
    if path is None:
        books = sorted(PRED_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime)
        if not books:
            sys.exit(f"no workbooks in {PRED_DIR}")
        path = books[-1]
    date, s = grade(path)
    print(f"graded {path}")
    print(f"  {date}: {s['cells']:,} cells checked, {s['hit']:,} stats "
          f"occurred (now yellow / dark blue / dark green / dark purple)")
    if s.get("bets"):
        print(f"  Bets sheet: {s.get('bets_won', 0)} of {s['bets']} bet(s) "
              f"won -> row highlighted solid green")
    if s["missing_rows"]:
        print(f"  {s['missing_rows']} row(s) had no final box score yet — "
              f"run  python Scrapers/scrape_gamelogs_3F.py  and grade again")


if __name__ == "__main__":
    main()
