# FINISH-BATCH HANDOFF — 2026-07-14 (paste into a fresh session to continue)

You are continuing the one-shot "finish the model" batch for an MLB betting-prediction
model. All the build work is done and parity-verified; the training chain was
**deliberately killed mid-run** because the user wants to make *additional* changes that
require re-running the superset. Your job is to help make those changes, re-run the
chain, then carry the finish plan to completion.

**Read first, in this order:** `Model/FINISH_PLAN.md` (the executable spec — gates,
per-item specs, phases), `Model/FEATURE_BACKLOG.md` (item specs + the Decline ledger),
and the auto-memory index `MEMORY.md`. This file is the state snapshot that ties them
together.

---

## 0. STANDING RULES (do not re-litigate — the user set these today)

- **Policy v4 — Claude decides NOTHING unilaterally.** Every accept/decline/ship/revert/
  skip gets pros+cons + a recommendation, and the USER decides. The old ship bar
  (≥1 CI-clear win etc.) is now a *recommendation framework*, not an auto-trigger.
  See memory `accept-bar-and-keep-policy` (v4) and `no-unilateral-declines`.
- **No auto-declines.** Anything you'd skip goes to the user with reasoning, and gets
  logged in `Model/FEATURE_BACKLOG.md` § "Decline ledger". Current ledger: the 1F
  selection trio (correlation pre-cut, time-block subsampling, LGBM-only voting) + the
  shrinkage-prior leakage fixes.
- **No pre-declared targets** at ideation/scope/eval — improve the whole 36-head surface;
  read every head. Memory verdicts are time-stamped, not law (cheap re-tests win).
- **Data-role = GRADED TIERS (not "holdout"/"validation year"):** 2015–2024 training /
  2025 calibration + feature-selection information / 2026→2026-07-14 partial season
  (~59%), lightly touched (enumerate the touches) / 2026-07-15+ timestamped forward
  predictions = the only leakage-free test. Backtest numbers are UPPER-BOUND estimates.
  Memory `true-test-doctrine`.
- **2026 read is VETO-ONLY.** Phase-5 runs a single 2026 `--confirm`, but it may only
  VETO, never TUNE: **no mechanism-bisecting against 2026** (bisect only against 2025).
  If 2026 disagrees → user picks {ship anyway documented | revert whole batch}. All
  weight/param FITTING (pa_blend, sweeps) stays **2025-only**.
- **2027 = forward-record only, no lockbox** (revisitable before Opening Day 2027).

---

## 1. CURRENT TREE STATE (what's dirty — READ CAREFULLY)

- **`Model/artifacts/feature_keep.json` = the ORIGINAL 24-head pre-batch keep-list**
  (restored by hand after the kill). Its backup `feature_keep.pre_chain_0714.bak` was
  consumed by the restore and no longer exists — the chain rerun recreates it. Do NOT
  hand-create a bak; let the chain's first `mv` do it.
- **`models.joblib` (16:xx) and `models_bt.joblib` are STALE SUPERSET intermediates.**
  Stage 1 of the killed chain overwrote the shipped models in place; there is **no backup
  of the pre-batch shipped models**. So the repo is **NOT serving-ready** — do not run
  `predict.py`/GUI for real output until a full chain completes. (The daily 06:00 task
  has a guard that sends it scrape-only when an experiment is in the tree; still, warn the
  user their serving is down until the rerun finishes.)
- **`frames.joblib` (2026-07-14 12:30, 1.29 GB) = cached feature frames WITH all the new
  batch columns.** Reuse it (chain loads it in ~10 s) **only if the user's new changes do
  NOT touch feature construction** (features.py build_* / add_*_derived / any scrape
  schema). If they DO, rebuild: delete `frames.joblib` (the rebuild script is
  `scratchpad/rebuild_frames.py` in the ORIGINAL session, or `train.py` rebuilds frames
  when the cache is absent). After any feature change: re-run `predict.py --selftest`
  parity before trusting the chain.
- Killed chain had: stage 1 ✅ both suites, stage 2 ✅ (wrote a 35-head keep-list +
  printed the shadow-eps table & co-failure report — now discarded), stage 3 had just
  STARTED (selection suite) when killed. Nothing from stage 3 survived.
- **`LGBM_ONLY_TEMP = True` in train.py (added 07-14 post-abort, user):** XGB_BAGS and
  CB_BAGS resolve to 0 so iteration retrains fit only the 6-bag LGBM family. Flip to
  False before the FINAL chain (Phase-4 step 0 in FINISH_PLAN.md) — interim LGBM-only
  models/keep-lists are iteration scaffolding, not the shipped 3-family recipe.

## 2. WHAT IS DONE (built, compiled, parity-verified — don't rebuild)

- **All backlog features #15–32 + 1E precip + 1F selection upgrades** built through the
  shared train/serve helpers (parity by construction). Batter superset 385 cols, starts
  134, team 64, winner 55.
- **24→36 outputs:** new heads bk3/tb3/tb4/hrr4 (H1), triple (H3), rbi2/run2 (H4),
  xh/xrun/xrbi/xbb count means (H6, banked cals never shipped), team_total (H5). All
  wired through predict/grading/hit-rate/rankings/Props.txt/glossary.
- **1F selection machinery:** shadow columns (`shdw_` Boruta-lite, per-head eps = p95 of
  shadow SHAP shares) + Spearman co-failure clone report (report-only). In train.py +
  feature_select.py.
- **Hazard v2** (bf-relative outs fix) built + wired into pa_backtest/pa_serve, v1 kept
  for rollback. **Backtest regrade done both years: v2 ≈ v1 raw (slightly worse outs)** —
  v1-vs-v2 keep is a USER call at blend time (Phase 3.3).
- **G5 linescore scraper** built + **backfill DONE** (243,112 inning rows, 26,635 games,
  2015+, 0 fetch failures → `Data/mlb_linescores.csv`, 7.9 MB). Unblocks F5/inning markets.
- **`predict.py --selftest` PARITY OK** at last check (worst rel diff 1.4e-12, 0 NaN
  mismatches) — including the wind_carry serving-gap fix the new team-row selftest caught.
- **Tools:** `Tools/5_performance.py` (every head × every metric, both suites →
  `Tools/performance.txt`); `Tools/4_grade_results.py --all` (cumulative forward
  backtest-style read); day-report added to the single-workbook grade too.

## 3. REMAINING SEQUENCE (after the user's new changes land)

1. **If features changed:** rebuild frames, then `predict.py --selftest` until PARITY OK.
2. **Re-run the chain** (live-buffered; ~4–4.5 h on the 385-col superset):
   ```
   cd "c:/Users/gdsak/OneDrive/Desktop/MLB" && \
   { [ ! -f Model/artifacts/feature_keep.json ] || mv Model/artifacts/feature_keep.json Model/artifacts/feature_keep.pre_chain_0714.bak; } && \
   echo "=== [1/3] SUPERSET TRAIN ===" && \
   python -u Model/train.py 2>&1 | grep --line-buffered -Ev "PerformanceWarning|frame.insert|DtypeWarning|low_memory" && \
   echo "=== [2/3] FEATURE SELECT (--write) ===" && \
   python -u Model/feature_select.py --stat shap --write 2>&1 | grep --line-buffered -Ev "DtypeWarning|low_memory" && \
   echo "=== [3/3] KEEP-LIST TRAIN ===" && \
   python -u Model/train.py 2>&1 | grep --line-buffered -Ev "PerformanceWarning|frame.insert|DtypeWarning|low_memory" && \
   echo "=== CHAIN COMPLETE ==="
   ```
   Watch live: `Get-Content <task .output file> -Wait -Tail 30`. (`python -u` +
   `grep --line-buffered` is REQUIRED — plain grep block-buffers and hides progress.)
3. **Selection report → user.** Present per-head shadow eps, kept + pre-top-up counts,
   per-family shadow floors, any co-failure clone groups (gray zones the user
   adjudicates), and the keep-diff vs the pre-chain .bak. All of it persists to
   `Model/artifacts/selection_report.json` (07-14 adopt package) — the stdout report
   is no longer the only copy.
4. **Phase 5 verdict.** `evaluate_deep.py --paired` on 2025 (all 36 heads; balanced
   AUC/logloss/ECE gates) + the H1 acceptance-bar table + the Σxrun-vs-team_total
   coherence read; THEN `evaluate_deep.py --confirm` ONCE on 2026 **veto-only** (no
   bisect against 2026). Present evidence + pros/cons + recommendation → **USER decides
   ship/revert** and any 2025-only route-arounds.
5. **On ship — ORDER MATTERS:** G4 dead-code deletion in train.py FIRST, then README
   rewrite (36 heads, SIM_BLEND, hazard, graded-tiers data-role table with touches
   enumerated, multiple-testing caveat, prior-leakage limitation), THEN
   `evaluate_deep.py --set-baseline` + `--confirm --set-baseline` (so baseline_code_fp
   fingerprints the final sources), then `prop_rankings.py --warm-cache`.
6. **PA-sim finish (2025-only fitting):** `pa_blend.py` re-sweep → update
   `predict.SIM_BLEND` weights (user call on gray zones); hazard v1-vs-v2 keep = user
   call on the sweep numbers; steal-layer re-sweep; `pa_serve` smoke (15/15 precedent).
7. **Smokes:** post-chain `predict.py --selftest`, GUI slate, tools end-to-end on a
   graded day.
8. **Phase 7:** artifacts hygiene (prune stale .bak/pre_*), FEATURE_BACKLOG verdicts
   (#15–32 IMPLEMENTED, H1–H6 verdicts, Decline ledger), memory updates, **commit**.

## 4. GOTCHAS

- **>100 MB files will fail `git push`.** `.gitignore` currently un-ignores everything
  under `Data/` (user's own edit — do NOT revert). `Data/mlb_statcast_bip.csv` (146 MB)
  and the `Data/raw_pitches/*.parquet` (12 files) exceed GitHub's 100 MB limit. Do NOT
  `git add` them; flag to the user at commit time (Git LFS or re-ignore = their call).
- **Grader mtime quirk:** `4_grade_results.py` no-arg grades the NEWEST-by-mtime workbook;
  regrading an old one makes it "newest". Not a bug — just know it.
- **Never touch the user's scheduler / daily-automation config** (standing lesson).
- **Commit trailer:** `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- An unrelated `bag_diversity.py` python process from another session may be running —
  it is NOT part of this batch; leave it alone.

## 5. KEY PATHS

- Plan/specs: `Model/FINISH_PLAN.md`, `Model/FEATURE_BACKLOG.md`
- Metrics artifacts: `Model/artifacts/metrics_select.json` (selection/2025),
  `metrics.json` (shipping/2026), `eval_baseline_select_2025.json`,
  `eval_baseline_2026.json`; paired snapshots `eval_paired_*.joblib`
- Sim: `Model/pa_*.py`, `Model/artifacts/pa_sim_tables.joblib`, `sim_grade_*.csv`
- Memory dir: the auto-memory folder (index `MEMORY.md`); relevant entries —
  `finish-plan`, `accept-bar-and-keep-policy` (v4), `no-unilateral-declines`,
  `true-test-doctrine`, `mlb-model-current-state`, `pa-sim-program`, `feature-backlog`,
  `daily-pipeline-automation`.
