"""Run every scraper to bring all CSVs in Data/ up to date.

Discovers scrape_*.py in this directory and runs each one with its default
output (all default to Data/). The pitch-arsenal scraper runs twice (pitcher
and batter views). build_ballparks.py is intentionally excluded: park
dimensions and elevations don't change daily. scrape_odds.py is also excluded:
betting lines must be captured near game time (closing lines), not in this
morning data job — run it alongside get_todays_games.py near first pitch.

Each scraper is fault-isolated: one failing doesn't stop the rest, and the
exit code is non-zero if anything failed.

Usage:
    python Scripts/update_all.py [--retrain]

    --retrain    also rebuild feature frames and retrain the models
                 (Model/train.py --rebuild) after a fully successful update
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
MODEL_TRAIN = SCRIPTS_DIR.parent / "Model" / "train.py"


# matches the scrape_*.py glob but must NOT run in the 6 AM data job: betting
# lines are captured near game time (closing lines), and a morning run would
# grab opening/empty markets and burn the odds-API quota. Run it near first
# pitch with get_todays_games.py instead.
EXCLUDE = {"scrape_odds.py"}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrain", action="store_true",
                    help="retrain the models after a successful update")
    args = ap.parse_args()

    results = []
    for label, cmd in discover_jobs():
        ok, took = run(label, cmd)
        results.append((label, ok, took))

    all_ok = all(ok for _, ok, _ in results)
    if args.retrain:
        if all_ok:
            ok, took = run("Model/train.py --rebuild",
                           [str(MODEL_TRAIN), "--rebuild"])
            results.append(("retrain models", ok, took))
            all_ok = all_ok and ok
        else:
            print("\nskipping retrain: at least one scraper failed",
                  file=sys.stderr)

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for label, ok, took in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {took:6.0f}s  {label}")
    if not all_ok:
        print("\nsome steps FAILED", file=sys.stderr)
        sys.exit(1)
    print("\nall data up to date")


if __name__ == "__main__":
    main()
