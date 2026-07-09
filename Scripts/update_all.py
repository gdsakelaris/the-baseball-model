"""Run every scraper to bring all CSVs in Data/ up to date.

Discovers scrape_*.py in this directory and runs each one with its default
output (all default to Data/). The pitch-arsenal scraper runs twice (pitcher
and batter views). build_ballparks.py is intentionally excluded: park
dimensions and elevations don't change daily. scrape_odds.py is also excluded:
betting lines must be captured near game time (closing lines), not in this
morning data job — run it alongside get_todays_games.py near first pitch.

Each scraper is fault-isolated: one failing doesn't stop the rest, and the
exit code is non-zero if anything failed.

Safety net (validate_data.py): before each scraper runs, its current
known-good CSVs are copied to Data/backups/; after it runs, the fresh files
are schema-validated (required columns, keys, row counts vs the backup, date
sanity). A file that fails validation is REPLACED by its backup, the job is
marked FAILED, and the retrain is skipped — a silent upstream format change
can no longer poison the daily retrain. The last log line is always
"RESULT: OK" or "RESULT: FAILED" for easy scanning of Logs/update_*.log.

Usage:
    python Scripts/update_all.py [--retrain]

    --retrain    also rebuild feature frames and retrain the models
                 (Model/train.py --rebuild) after a fully successful update
"""

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import validate_data as V

SCRIPTS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN = SCRIPTS_DIR.parent / "Model" / "train.py"
DATA_DIR = V.DATA_DIR
BACKUP_DIR = V.BACKUP_DIR
# machine-readable outcome of the last run; the GUI reads this at startup
# and warns when the morning job failed (otherwise the only signal is a
# log line nobody looks at until predictions have quietly gone stale)
STATUS_FILE = SCRIPTS_DIR.parent / "Logs" / "last_run_status.json"


# matches the scrape_*.py glob but must NOT run in the 6 AM data job: betting
# lines are captured near game time (closing lines), and a morning run would
# grab opening/empty markets and burn the odds-API quota. Run it near first
# pitch with get_todays_games.py instead.
EXCLUDE = {"scrape_odds.py"}

# which Data/ files each job owns (backed up before the run, validated after)
JOB_FILES = {
    "scrape_batting_stats.py": ["mlb_batting_stats.csv"],
    "scrape_gamelogs_3F.py": ["mlb_games.csv",
                              "mlb_game_batting.csv",
                              "mlb_game_pitching.csv"],
    "scrape_handedness.py": ["mlb_handedness.csv"],
    "scrape_homeruns.py": ["mlb_homeruns.csv"],
    "scrape_pitch_arsenals_2F.py (pitchers)":
        ["mlb_pitch_arsenals.csv"],
    "scrape_pitch_arsenals_2F.py (batters)":
        ["mlb_pitch_arsenals_batters.csv"],
    "scrape_pitching_stats.py": ["mlb_pitching_stats.csv"],
    "scrape_rosters.py": ["mlb_rosters.csv"],
    "scrape_statcast.py": ["mlb_statcast_bip.csv"],
    "scrape_pitches.py": ["mlb_pitch_daily_pitchers.csv",
                          "mlb_pitch_daily_batters.csv"],
    "scrape_sprint_speed.py": ["mlb_sprint_speed.csv"],
    "scrape_oaa.py": ["mlb_oaa.csv"],
    "scrape_umpires.py": ["mlb_umpires.csv"],
    "scrape_bat_tracking.py": ["mlb_bat_tracking.csv"],
}


def discover_jobs():
    jobs = []  # (label, [args])
    for script in sorted(SCRIPTS_DIR.glob("scrape_*.py")):
        if script.name in EXCLUDE:
            continue
        if "pitch_arsenals" in script.name:
            jobs.append((f"{script.name} (pitchers)", [str(script)]))
            jobs.append((f"{script.name} (batters)", [str(script), "--type", "batter"]))
        else:
            jobs.append((script.name, [str(script)]))
    return jobs


def run(label, args):
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, *args])
    took = time.time() - t0
    ok = proc.returncode == 0
    print(f">>> {label}: {'OK' if ok else f'FAILED (exit {proc.returncode})'} "
          f"in {took:.0f}s", flush=True)
    return ok, took


def backup_known_good(files):
    """Copy each currently-valid file to Data/backups/ before its scraper
    rewrites it. A file that is ALREADY invalid is not backed up — that would
    clobber the last good backup with a bad copy."""
    BACKUP_DIR.mkdir(exist_ok=True)
    for name in files:
        src = DATA_DIR / name
        if not src.exists():
            continue
        if V.validate_file(src):        # current copy itself fails validation
            print(f"    (not backing up {name}: current copy already fails "
                  f"validation; keeping the existing backup)", flush=True)
            continue
        shutil.copy2(src, BACKUP_DIR / name)


def validate_and_restore(files):
    """Validate a job's fresh output against the backups. On failure, restore
    the backup so downstream consumers keep working. Returns True if every
    file passed."""
    all_ok = True
    for name in files:
        prev = BACKUP_DIR / name
        problems = V.validate_file(DATA_DIR / name,
                                   prev if prev.exists() else None)
        if not problems:
            continue
        all_ok = False
        for p in problems:
            print(f"    VALIDATION FAIL: {p}", flush=True)
        if prev.exists():
            shutil.copy2(prev, DATA_DIR / name)
            print(f"    restored {name} from backup", flush=True)
        else:
            print(f"    no backup available for {name}; the bad file was "
                  f"left in place for inspection", flush=True)
    return all_ok


def experiment_in_flight():
    """True when the Model sources differ from the ones the current paired
    baselines were snapshotted from (evaluate_deep --set-baseline writes
    baseline_code_fp.json). The daily retrain must train SHIPPED code —
    quietly retraining and re-baselining an in-flight candidate would make
    the experiment its own reference — so a mismatch turns the run
    scrape-only. No fingerprint file (pre-feature snapshots) = proceed."""
    import hashlib
    fp_file = MODEL_TRAIN.parent / "artifacts" / "baseline_code_fp.json"
    if not fp_file.exists():
        return False
    try:
        base = json.loads(fp_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    for name, digest in base.items():
        p = MODEL_TRAIN.parent / name
        if not p.exists() or hashlib.md5(p.read_bytes()).hexdigest() != digest:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the models after a successful update")
    args = ap.parse_args()

    results = []
    for label, cmd in discover_jobs():
        files = JOB_FILES.get(label, [])
        if files:
            backup_known_good(files)
        ok, took = run(label, cmd)
        if ok and files:
            ok = validate_and_restore(files)
            if not ok:
                print(f">>> {label}: data FAILED validation", flush=True)
        results.append((label, ok, took))

    all_ok = all(ok for _, ok, _ in results)
    if args.retrain:
        if all_ok and experiment_in_flight():
            print("\n>>> retrain SKIPPED: EXPERIMENT IN FLIGHT — the Model "
                  "sources differ from the last --set-baseline snapshot "
                  "(baseline_code_fp.json). Data was still refreshed, so the "
                  "next --paired read will demand a re-baseline: finish or "
                  "revert the candidate, then run "
                  "update_all.py --retrain (or wait for tomorrow's run).",
                  flush=True)
            results.append(("retrain models (skipped: experiment in flight)",
                            True, 0.0))
        elif all_ok:
            ok, took = run("Model/train.py --rebuild",
                           [str(MODEL_TRAIN), "--rebuild"])
            results.append(("retrain models", ok, took))
            all_ok = all_ok and ok
            # refresh the paired-eval baselines to match the just-trained
            # models: the retrain above makes yesterday's snapshots stale
            # (evaluate_deep --paired refuses a stale read via its data
            # fingerprint), and a snapshot is only valid when it captures
            # the CURRENT artifacts on the CURRENT data — which is exactly
            # the state right here. The experiment_in_flight() gate above
            # guarantees this only ever snapshots SHIPPED code.
            if ok:
                ev = str(MODEL_TRAIN.parent / "evaluate_deep.py")
                for label, cmd in (
                        ("evaluate_deep.py --set-baseline",
                         [ev, "--set-baseline"]),
                        ("evaluate_deep.py --confirm --set-baseline",
                         [ev, "--confirm", "--set-baseline"])):
                    ok, took = run(label, cmd)
                    results.append((label, ok, took))
                    all_ok = all_ok and ok
        else:
            print("\nskipping retrain: at least one scraper failed or "
                  "produced invalid data", file=sys.stderr)

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for label, ok, took in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {took:6.0f}s  {label}")

    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(json.dumps({
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "ok": all_ok,
        "failed_jobs": [label for label, ok, _ in results if not ok],
    }, indent=1))

    if not all_ok:
        print("\nRESULT: FAILED", flush=True)
        sys.exit(1)
    print("\nRESULT: OK")


if __name__ == "__main__":
    main()
