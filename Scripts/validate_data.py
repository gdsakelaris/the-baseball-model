"""Schema validation for the CSVs in Data/ — the data pipeline's safety net.

Every scraper rewrites its CSV in full each morning, so a silent upstream
format change (renamed column, half-empty response, truncated download)
would poison the daily retrain and everything downstream. This module
declares, per file, what the MODEL actually requires — the columns
features.py reads, the relational keys, expected row-count behavior — and
checks a freshly scraped file against that spec plus the previous known-good
copy before the pipeline accepts it.

Used two ways:
  - update_all.py: after each scraper runs, its output is validated; on
    failure the previous file is restored from Data/backups/ and the job is
    marked FAILED (which also blocks --retrain).
  - standalone:  python Scripts/validate_data.py            # check all files
                 python Scripts/validate_data.py mlb_odds.csv ...

A validation problem is a human-readable string; a file passes when its
problem list is empty. Checks are deliberately about *shape and plumbing*
(columns, keys, row counts, parseability, date sanity), not statistical
content — the model's own evaluation covers that.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
BACKUP_DIR = DATA_DIR / "backups"

# Spec fields (all optional except required_cols):
#   required_cols   columns the model reads (scrapers may write more)
#   key             columns that should be (near-)unique per row
#   max_dup_frac    tolerated duplicate-key fraction (statsapi has the odd
#                   double-listed player; today's data has 1 dup in 311k rows)
#   date_col        must parse as dates (>=99%) and not run past tomorrow
#   numeric         [(col, max_nan_frac)]: col must parse as numeric with at
#                   most this fraction NaN (catches a text column shifting in)
#   min_rows        absolute floor (a fraction of today's size; catches an
#                   empty/truncated response regardless of backup state)
#   shrink_tol      new_rows >= shrink_tol * old_rows vs the previous copy
#                   (cumulative multi-season files should only grow; snapshot
#                   files like rosters may shrink a little)
#   season_col      seasons present before must still be present (catches a
#                   scrape that silently dropped history)
#   fresh_days      during the unambiguous in-season months (May-September),
#                   max(date_col) must be within this many days of today.
#                   This is the staleness tripwire: a scraper that "succeeds"
#                   but stops ingesting new games (frozen season list, upstream
#                   format change, permanent fallback to stored rows) passes
#                   every shape check — only a freshness check makes that
#                   loud. 6 days clears the All-Star break.
SPECS = {
    "mlb_games.csv": dict(
        required_cols=["GamePk", "Season", "Date", "DayNight", "AwayTeam",
                       "HomeTeam", "AwayScore", "HomeScore", "Venue", "Temp",
                       "Condition", "WindSpeed", "WindDir"],
        key=["GamePk"], max_dup_frac=0.0, date_col="Date", fresh_days=6,
        numeric=[("GamePk", 0.0), ("AwayScore", 0.02), ("HomeScore", 0.02)],
        min_rows=13000, shrink_tol=0.999, season_col="Season"),
    "mlb_game_batting.csv": dict(
        required_cols=["GamePk", "Season", "Date", "PlayerId", "Name", "Team",
                       "Opponent", "Home", "BattingOrder", "Position", "PA",
                       "AB", "R", "H", "2B", "3B", "HR", "RBI", "BB", "IBB",
                       "SO", "HBP", "SB", "CS", "TB"],
        key=["GamePk", "PlayerId"], max_dup_frac=0.001, date_col="Date",
        numeric=[("PlayerId", 0.0), ("PA", 0.001), ("HR", 0.001)],
        min_rows=280000, shrink_tol=0.999, season_col="Season"),
    "mlb_game_pitching.csv": dict(
        required_cols=["GamePk", "Season", "Date", "PlayerId", "Name", "Team",
                       "Opponent", "Home", "GS", "GF", "IP", "BF", "NP",
                       "Strikes", "H", "R", "ER", "HR", "BB", "SO", "SV",
                       "HLD"],
        key=["GamePk", "PlayerId"], max_dup_frac=0.001, date_col="Date",
        numeric=[("PlayerId", 0.0), ("BF", 0.001), ("SO", 0.001)],
        min_rows=110000, shrink_tol=0.999, season_col="Season"),
    "mlb_homeruns.csv": dict(
        required_cols=["Year", "Date", "BatterId", "Batter", "Angle",
                       "Exit Velo", "Distance", "Pitch", "Pitcher",
                       "PitcherTeam", "Ballpark"],
        date_col="Date", fresh_days=6,
        numeric=[("BatterId", 0.01), ("Distance", 0.05), ("Exit Velo", 0.05)],
        min_rows=30000, shrink_tol=0.999, season_col="Year"),
    "mlb_pitch_arsenals.csv": dict(
        required_cols=["Year", "PlayerId", "PitchType", "Pitch", "%",
                       "RV/100", "Pitches", "xSLG", "xwOBA", "Whiff %",
                       "Hard Hit %", "K%", "Put Away %"],
        key=["Year", "PlayerId", "PitchType"], max_dup_frac=0.001,
        numeric=[("PlayerId", 0.0), ("%", 0.05)],
        min_rows=20000, shrink_tol=0.999, season_col="Year"),
    "mlb_pitch_arsenals_batters.csv": dict(
        required_cols=["Year", "PlayerId", "PitchType", "Pitch", "%",
                       "RV/100", "Pitches", "xSLG", "xwOBA", "Whiff %",
                       "Hard Hit %", "K%", "Put Away %"],
        key=["Year", "PlayerId", "PitchType"], max_dup_frac=0.001,
        numeric=[("PlayerId", 0.0), ("%", 0.05)],
        min_rows=32000, shrink_tol=0.999, season_col="Year"),
    "mlb_batting_stats.csv": dict(
        required_cols=["Year", "PlayerId", "Name", "Team", "PA", "GO/AO"],
        key=["Year", "PlayerId"], max_dup_frac=0.001,
        numeric=[("PlayerId", 0.0), ("PA", 0.001)],
        min_rows=5000, shrink_tol=0.999, season_col="Year"),
    "mlb_pitching_stats.csv": dict(
        required_cols=["Year", "PlayerId", "Name", "Team", "IP", "TBF",
                       "GO/AO", "SB", "CS", "PK"],
        key=["Year", "PlayerId"], max_dup_frac=0.001,
        numeric=[("PlayerId", 0.0), ("TBF", 0.001)],
        min_rows=5000, shrink_tol=0.999, season_col="Year"),
    "mlb_rosters.csv": dict(
        required_cols=["PlayerId", "Name", "Team", "Position", "B", "T",
                       "Ht", "Wt", "DOB"],
        key=["PlayerId"], max_dup_frac=0.001,
        numeric=[("PlayerId", 0.0)],
        min_rows=700, shrink_tol=0.9),          # snapshot: may shrink a bit
    "mlb_handedness.csv": dict(
        required_cols=["PlayerId", "Name", "Bats", "Throws"],
        key=["PlayerId"], max_dup_frac=0.001,
        numeric=[("PlayerId", 0.0)],
        min_rows=2500, shrink_tol=0.95),        # snapshot, grows slowly
    "mlb_ballparks.csv": dict(                  # static; standalone runs only
        required_cols=["Ballpark", "Team", "LF", "CF", "RF", "Elevation_ft"],
        key=["Ballpark"], max_dup_frac=0.0,
        numeric=[("LF", 0.0), ("CF", 0.0), ("RF", 0.0)],
        min_rows=30, shrink_tol=1.0),
    "mlb_statcast_bip.csv": dict(
        required_cols=["GamePk", "Season", "Date", "BatterId", "PitcherId",
                       "Events", "BBType", "ExitVelo", "LaunchAngle", "LSA",
                       "xBA", "xwOBA", "AtBat", "PitchNum"],
        key=["GamePk", "AtBat", "PitchNum"], max_dup_frac=0.0005,
        date_col="Date", fresh_days=6,
        numeric=[("BatterId", 0.0), ("ExitVelo", 0.05), ("xwOBA", 0.05)],
        min_rows=700000, shrink_tol=0.999, season_col="Season"),
    "mlb_pitch_daily_pitchers.csv": dict(
        required_cols=["PlayerId", "Date", "n", "sw_n", "wh_n", "cs_n",
                       "z_n", "oz_n", "oz_sw", "fb_n", "fb_v"],
        key=["PlayerId", "Date"], max_dup_frac=0.0, date_col="Date",
        fresh_days=6,
        numeric=[("PlayerId", 0.0), ("n", 0.001)],
        min_rows=120000, shrink_tol=0.999),
    "mlb_pitch_daily_batters.csv": dict(
        required_cols=["PlayerId", "Date", "n", "sw_n", "wh_n", "cs_n",
                       "z_n", "oz_n", "oz_sw"],
        key=["PlayerId", "Date"], max_dup_frac=0.0, date_col="Date",
        fresh_days=6,
        numeric=[("PlayerId", 0.0), ("n", 0.001)],
        min_rows=250000, shrink_tol=0.999),
    "mlb_sprint_speed.csv": dict(
        required_cols=["Year", "PlayerId", "SprintSpeed", "HPto1B",
                       "CompetitiveRuns"],
        key=["Year", "PlayerId"], max_dup_frac=0.0,
        numeric=[("PlayerId", 0.0), ("SprintSpeed", 0.01)],
        min_rows=3500, shrink_tol=0.999, season_col="Year"),
    "mlb_oaa.csv": dict(
        required_cols=["Year", "Team", "OAA", "OAA_per162"],
        key=["Year", "Team"], max_dup_frac=0.0,
        numeric=[("OAA", 0.0)],
        min_rows=200, shrink_tol=0.999, season_col="Year"),
    "mlb_odds.csv": dict(                       # append-only store
        required_cols=["Date", "GamePk", "PlayerId", "Market", "Line",
                       "OverPrice", "UnderPrice", "Book", "CapturedAt"],
        date_col="Date",
        numeric=[("Line", 0.05), ("OverPrice", 0.20)],
        min_rows=1, shrink_tol=0.999),
}


def validate_file(path, prev_path=None, spec=None):
    """Check one CSV against its spec (and its previous copy, if given).
    Returns a list of problem strings; empty means the file passed."""
    path = Path(path)
    spec = spec or SPECS.get(path.name)
    if spec is None:
        return [f"no schema spec for {path.name}"]
    if not path.exists():
        return [f"{path.name}: file does not exist"]
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except Exception as e:                          # noqa: BLE001
        return [f"{path.name}: unreadable as CSV ({e})"]

    problems = []
    missing = [c for c in spec["required_cols"] if c not in df.columns]
    if missing:
        problems.append(f"{path.name}: missing required columns {missing}")
        return problems  # everything else would cascade off this

    n = len(df)
    if n < spec.get("min_rows", 1):
        problems.append(f"{path.name}: only {n:,} rows "
                        f"(floor {spec['min_rows']:,})")

    key = spec.get("key")
    if key and n:
        dup = df.duplicated(key).mean()
        if dup > spec.get("max_dup_frac", 0.0):
            problems.append(f"{path.name}: {dup:.2%} duplicate rows on key "
                            f"{key} (tolerance {spec.get('max_dup_frac'):.2%})")

    for col, max_nan in spec.get("numeric", ()):
        bad = pd.to_numeric(df[col], errors="coerce").isna().mean()
        if bad > max_nan:
            problems.append(f"{path.name}: column {col!r} is {bad:.1%} "
                            f"non-numeric/blank (tolerance {max_nan:.1%})")

    dc = spec.get("date_col")
    if dc and n:
        d = pd.to_datetime(df[dc], errors="coerce")
        if d.isna().mean() > 0.01:
            problems.append(f"{path.name}: {d.isna().mean():.1%} of {dc!r} "
                            f"fails to parse as a date")
        elif d.max() > pd.Timestamp(date.today() + timedelta(days=1)):
            problems.append(f"{path.name}: max {dc} {d.max().date()} is in "
                            f"the future")
        else:
            # staleness tripwire: May-September there is always MLB within
            # fresh_days (6 clears the All-Star break); a file whose newest
            # date lags means the scraper "succeeds" without ingesting new
            # games (frozen season list, upstream change, permanent fallback)
            fd = spec.get("fresh_days")
            if fd and 5 <= date.today().month <= 9 and \
                    d.max() < pd.Timestamp(date.today() - timedelta(days=fd)):
                problems.append(
                    f"{path.name}: newest {dc} is {d.max().date()} — more "
                    f"than {fd} days old mid-season; the scraper runs but "
                    f"ingests no new games")

    if prev_path is not None and Path(prev_path).exists():
        try:
            prev = pd.read_csv(prev_path, encoding="utf-8-sig",
                               low_memory=False)
        except Exception:                           # noqa: BLE001
            prev = None                             # bad backup: skip diffs
        if prev is not None and len(prev):
            tol = spec.get("shrink_tol", 0.999)
            if n < tol * len(prev):
                problems.append(
                    f"{path.name}: shrank {len(prev):,} -> {n:,} rows "
                    f"(tolerance {tol:.1%} of previous)")
            sc = spec.get("season_col")
            if sc and sc in prev.columns:
                lost = set(prev[sc].dropna().unique()) - set(
                    df[sc].dropna().unique())
                if lost:
                    problems.append(f"{path.name}: seasons vanished vs "
                                    f"previous copy: {sorted(lost)}")
    return problems


def main():
    names = sys.argv[1:] or sorted(SPECS)
    n_bad = 0
    for name in names:
        path = DATA_DIR / name
        prev = BACKUP_DIR / name
        problems = validate_file(path, prev if prev.exists() else None)
        if problems:
            n_bad += 1
            for p in problems:
                print(f"  FAIL  {p}")
        else:
            print(f"  ok    {name}")
    if n_bad:
        sys.exit(f"\n{n_bad} file(s) failed validation")
    print("\nall files passed")


if __name__ == "__main__":
    main()
