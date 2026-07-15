"""Sweep RECENCY_DECAY on the SELECTION suite (train<=2023, cal 2024, test
2025 — the split that exists precisely so choices like this never touch the
holdout). For each decay it runs `train.py --select --decay D`, banks that
run's metrics_select.json under artifacts/decay_sweep/, then prints a
per-head comparison table. Bake the winner into train.RECENCY_DECAY before
the ship chain.

Resumable: a decay whose banked metrics file already exists is skipped, so a
killed sweep continues where it stopped (delete artifacts/decay_sweep/ to
force a full re-run).

CAVEATS
  * Each run OVERWRITES artifacts/metrics_select.json and models_bt.joblib —
    the ship chain regenerates both, but do not read them mid-sweep expecting
    the incumbent. The last run leaves the LAST grid value in place, not
    necessarily the winner.
  * Runs under whatever keep-list / family-bag regime train.py currently
    has — the read is RELATIVE across decays, which is what the choice needs.
  * The table is a screen (point estimates on the selection test year); the
    baked winner is still verdicted by the chain's evaluate_deep --paired
    read like every other change in the batch.

Usage:
    python Model/decay_sweep.py             # full grid
    python Model/decay_sweep.py 1.0 0.9     # explicit grid values
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OUT = ART / "decay_sweep"
GRID = [1.0, 0.95, 0.9, 0.85, 0.8]

# metrics whose LOWER value wins, per head kind (binary vs count/MAE);
# auc prints alongside for the binary heads but logloss decides
BINARY_KEY, COUNT_KEY = "logloss", "mae"


def _bank_path(d):
    return OUT / f"metrics_select_decay{d:.2f}.json"


def run_grid(grid):
    OUT.mkdir(exist_ok=True)
    for d in grid:
        dst = _bank_path(d)
        if dst.exists():
            print(f"decay {d:.2f}: banked metrics found, skipping", flush=True)
            continue
        print(f"=== decay {d:.2f}: train.py --select --decay {d} ===",
              flush=True)
        r = subprocess.run([sys.executable, str(HERE / "train.py"),
                            "--select", "--decay", str(d)])
        if r.returncode:
            sys.exit(f"train.py failed at decay {d:.2f} — sweep aborted "
                     f"(banked runs are kept; re-run to resume)")
        shutil.copy2(ART / "metrics_select.json", dst)
        print(f"decay {d:.2f}: banked -> {dst.name}", flush=True)


def _head_rows(metrics):
    """(head, key, value) for every per-head metric block in one run."""
    for head, m in metrics.items():
        if not isinstance(m, dict):
            continue
        if BINARY_KEY in m:
            yield head, BINARY_KEY, m[BINARY_KEY]
        elif COUNT_KEY in m:
            yield head, COUNT_KEY, m[COUNT_KEY]


def report(grid):
    runs = {}
    for d in grid:
        p = _bank_path(d)
        if p.exists():
            runs[d] = json.loads(p.read_text())
    if len(runs) < 2:
        print("fewer than 2 banked runs — nothing to compare yet")
        return
    decays = sorted(runs, reverse=True)   # 1.00 first (the incumbent)
    heads = {}
    for d in decays:
        for head, key, val in _head_rows(runs[d]):
            heads.setdefault((head, key), {})[d] = val

    colw = max(len(h) for h, _ in heads) + 2
    print("\nper-head " + "/".join(f"{BINARY_KEY}|{COUNT_KEY}".split("|"))
          + " by decay (* = best, lower wins; delta vs decay 1.00 where "
            "banked)")
    print(" " * colw + "".join(f"{d:>10.2f}" for d in decays))
    wins = {d: 0 for d in decays}
    for (head, key), vals in sorted(heads.items()):
        best = min(vals, key=vals.get)
        wins[best] += 1
        cells = "".join(
            f"{vals[d]:>9.4f}{'*' if d == best else ' '}" if d in vals
            else f"{'—':>10}" for d in decays)
        print(f"{head:<{colw}}{cells}")
    print("\nheads won: " + ", ".join(f"{d:.2f}: {wins[d]}" for d in decays))
    if 1.0 in runs:
        base = {hk: v[1.0] for hk, v in heads.items() if 1.0 in v}
        for d in decays:
            if d == 1.0:
                continue
            deltas = [heads[hk][d] - base[hk] for hk in base
                      if d in heads[hk]]
            if deltas:
                mean = sum(deltas) / len(deltas)
                print(f"decay {d:.2f}: mean per-head delta vs 1.00 = "
                      f"{mean:+.5f} ({sum(x < 0 for x in deltas)}/"
                      f"{len(deltas)} heads improved)")
    # per-head winners (2026-07-15 PM batch): a paste-ready
    # train.RECENCY_HEAD_DECAY dict. Only heads whose winner differs from
    # the global train.RECENCY_DECAY are listed; the CV-jitter caveat in the
    # module docstring applies PER HEAD here, so treat small margins as
    # noise and prefer the global value unless the win is clear across
    # adjacent grid values too.
    try:
        import train as T
        global_d = T.RECENCY_DECAY
    except Exception:
        global_d = 0.95
    head_key = {}
    for (head, key), vals in heads.items():
        # metrics keys look like "hr_2025" / "team_runs_2025" — strip the
        # trailing year to recover the train-side head key
        base_name = head.rsplit("_", 1)[0]
        base_name = {"team_runs": "total"}.get(base_name, base_name)
        best = min(vals, key=vals.get)
        if abs(best - global_d) > 1e-9:
            head_key[base_name] = best
    if head_key:
        print(f"\nper-head winners differing from the global "
              f"{global_d:.2f} — paste into train.RECENCY_HEAD_DECAY "
              f"(the chain's paired read verdicts the dict as a package):")
        print("RECENCY_HEAD_DECAY = {")
        for k in sorted(head_key):
            print(f'    "{k}": {head_key[k]:.2f},')
        print("}")
    else:
        print(f"\nno per-head winner differs from the global {global_d:.2f} "
              f"— keep RECENCY_HEAD_DECAY empty")
    print("\nnext: bake the global winner into train.RECENCY_DECAY and any "
          "clear per-head winners into train.RECENCY_HEAD_DECAY — the ship "
          "chain's paired read verdicts them.")


def main():
    grid = [float(a) for a in sys.argv[1:]] or GRID
    run_grid(grid)
    report(grid)


if __name__ == "__main__":
    main()
