# 2026-07-15 PM diversity/calibration batch — ship-chain runbook

Everything below is IMPLEMENTED and smoke-tested (unit suite 43/43 green,
synthetic two-suite parity smokes green, hpo_sweep wiring verified on real
data). Nothing has been trained or shipped — per the batch decision, all
changes land in ONE final retrain chain, adjudicated together.

## What is in the batch

| # | Change | Flag / knob (train.py) | Revert |
|---|--------|------------------------|--------|
| 1 | CatBoost restored (2 bags), XGBoost retired permanently | `CB_BAGS = 2`, `XGB_BAGS = 0` | `CB_BAGS = 0` |
| 2 | Per-family logistic stack replaces the 21-pt blend grid (binaries + winner; counts get a deviance-chosen family weight) | `FAMILY_STACK` / `FSTACK_C = 50` | `FAMILY_STACK = False` |
| 3 | Day-block bootstrap-bagged calibrators | `CAL_BAG_B = 25` | `CAL_BAG_B = 0` |
| 4 | Multi-year pooled calibration support (stack, calibrator, line cals, dispersions) — selection suite feeds shipping at zero extra train cost | `MULTI_YEAR_CAL` / `CAL_POOL_DECAY = 0.75` | `MULTI_YEAR_CAL = False` |
| 5 | ES-refit: every member refit on 100% of train at its early-stop iteration (keep-trains only) | `ES_REFIT` | `ES_REFIT = False` |
| 6 | Winner home/away mirror augmentation (+ `persp_home` flag) | `WINNER_MIRROR` | `WINNER_MIRROR = False` |
| 7 | Per-head recency decay mechanism (empty until swept) | `RECENCY_HEAD_DECAY = {}` | keep empty |
| 8 | STACK_DONORS re-test: same-ladder donors for the thin deep heads | `STACK_DONORS` dict | `STACK_DONORS = {}` |
| 9 | Optuna HPO for the weak heads (evidence only, wired by hand) | `hpo_sweep.py` | don't wire |
| 10 | LR member ridge picked per head on cal-year logloss (user add) | `LR_C_GRID = (0.3, 0.1, 1.0)` | `LR_C_GRID = (0.3,)` |
| 11 | Full selection regen so CB votes on the keep-lists (user add) | chain steps 1-3 below; superset electorate carries both CB members | keep old `feature_keep.json` |

Artifact contract: `meta_stamp.artifact_version = 3`; new prop keys
`fstack`/`fstack_fams`/`fam_slices` (binaries/winner), `FamilyBlendBag`
(counts), `BaggedCal` under the `iso` key. predict.py serves all of it and
stays backward-compatible with the current `models.joblib`.

## Order of operations (regen first, then sweeps on the NEW lists, then chain)

The regen reorders everything: keep-lists change, so every sweep that reads
`_apply_keep` (param_sweep, hpo_sweep, decay_sweep) must run AFTER
feature_select --write, or its winners describe dead column sets.

Timing probes: after step 1, extrapolate wall-clock per head x 36 heads x 2
suites and check the eventual keep-chain against the 06:00 window. CB is
2-4x an LGBM fit (GPU) and ES-refit roughly doubles booster time — expect
~3-5x the LGBM-only chain. If the window is at risk: drop `ES_REFIT` first
(cheapest accuracy give-back), then `CB_BAGS = 1` (keep-train only — leave
the superset electorate at 2).

0. **Back up the serving artifacts** (serving continuity if adjudication
   rejects — the chain OVERWRITES them and they are not in git):
   copy `models.joblib`, `models_bt.joblib`, `metrics.json`,
   `metrics_select.json`, `inseason_offsets.json`,
   `eval_baseline_2026.json`, `eval_baseline_select_2025.json` to
   `Model/artifacts/pre_0715pm_backup/`. Restoring the incumbent is then a
   copy-back, not a flags-off retrain, and the GUI keeps a servable
   artifact through the whole adjudication window. (Baseline guard note:
   with uncommitted changes in the tree, the 06:00 job already goes
   scrape-only via `baseline_code_fp.json` — do NOT commit until the ship
   decision.)
1. **Superset train** (fresh electorate, per the approved shadow-superset
   design — CB votes with BOTH members): set `train.SELECT_FEATURES =
   False`, run `python Model/train.py`. Writes `models_superset_bt.joblib`
   + `models_superset.joblib` only — serving artifacts untouched, and the
   audit-#8 guard refuses them if they ever reach predict. This run doubles
   as the timing probe.
2. **Selection regen**: `python Model/feature_select.py --write` — it
   auto-prefers the `models_superset*.joblib` pair. Review
   `selection_report.json` (per-family votes now include cb), then flip
   `SELECT_FEATURES = True` back. The new `feature_keep.json` is now the
   sole decider again.
3. **Param re-sweep on the new lists** (the RE-SWEEP CADENCE events "large
   feature-set change" + "ship" both fire): 
   `python Model/param_sweep.py --ensemble`
   — families-aware is the honest objective now that CB ships (its
   docstring's own note). Wire gate-clearing winners into
   `PROP_PARAMS`/`COUNT_PARAMS`, wholesale-replace semantics as usual.
4. **HPO sweep on the weak heads, on top of the re-swept baseline**
   (hours): `python Model/hpo_sweep.py --trials 60`
   (default list: double, hit, single, rbi, hrr2/3/4, run2, rbi2). Wire
   gate-clearing winners; hpo gates compare against the CURRENT (step-3)
   config automatically.
5. **Per-head decay sweep** (5 x `--select` train, resumable):
   `python Model/decay_sweep.py` — prints a paste-ready
   `RECENCY_HEAD_DECAY` dict; bake clear winners only (per-head CV jitter
   caveat is printed with it). Delete `artifacts/decay_sweep/` first — the
   banked runs predate the regen.
6. **The chain** (the one decision-relevant run):
   `python Model/train.py --prestash`
   - `--prestash` trains one extra throwaway suite so the SELECTION suite
     also gets multi-year cal support — without it the 2025 paired read
     cannot see change #4 (everything else it sees). Costs ~+50% chain time
     ONCE; the daily job keeps running plain (shipping suite still pools).
7. **Re-fit SIM_BLEND against the new heads** (gap the first runbook draft
   missed): the shipped {score .60, total .55, winner .30} weights were fit
   on 2025 against the OLD GBM game heads; the batch moves those heads
   (stack + CB + mirror + refit + new keep-lists), so the sim-vs-GBM
   tradeoff shifts. Re-run the pa_grade/pa_blend 2025 fit after the
   keep-train and take its weights (fail-safe unchanged: sim failure
   degrades to the GBM alone). evaluate_deep verdicts the raw GBM heads
   either way — this step only affects the serving blend layer.
8. **Adjudication** (the standard bar, all 24+ heads read, no pre-declared
   targets):
   - `python Model/evaluate_deep.py --paired` (2025, selection suite)
   - `python Model/evaluate_deep.py --confirm` (2026, deliberate look)
   - `python Model/predict.py --selftest` + a multi-game slate smoke
     through the GUI's Predictor (the 07-14 chain pattern) — proves the
     new artifact contract serves end-to-end, not just in eval.
   - Per-flag verdicts are independent: a harmed mechanism reverts by its
     flag without unwinding the batch. STACK_DONORS heads verdict per head.
9. **USER DECIDES ship/revert.** If ship: re-stamp baselines
   (`--set-baseline` x2 per the standard chain), commit, and let the 06:00
   job resume. First retrain shifts every baseline (new keep-lists + ES
   refit + stack change the whole surface — a bigger shift than the 07-15
   AM ship). If revert: restore `pre_0715pm_backup/`, restore the old
   `feature_keep.json` (git has it), and flip the rejected flags.

## Decision points resolved by the user (2026-07-15 PM)

- **Feature-selection regen: IN THE CHAIN** (steps 1-2) — fresh superset
  per the approved shadow-superset design; superset electorate carries
  both CB members so the cb vote is granular (0/.5/1), not a coarse 0/1.
- **param_sweep --ensemble: IN THE CHAIN** (step 3) — the regen is a
  "large feature-set change" under the RE-SWEEP CADENCE, so the re-sweep
  fires now rather than at the era audit.
- **LR member upgrade: IN** — per-head C from `LR_C_GRID` on cal-year
  logloss (`lr_C` lands in each head's metrics block).

## Decision points still open (deliberately not decided in code)

- **CAL_POOL_DECAY = 0.75** is a reasoned default, not swept. Sweepable
  later the same way RECENCY_DECAY was.
- **Winner re-tune under mirroring**: LGB_WIN params predate the mirrored
  regime; if the mirror survives adjudication, a later sweep can re-tune
  the winner on mirrored frames (the sweeps here run unmirrored).

## Interactions worth knowing

- Order of serving ops is unchanged: `predict_prop` -> `apply_stack`
  (STACK_DONORS) -> `enforce_ladders` (PAV), identical in evaluate_deep.
- `--decay X` (decay_sweep) clears `RECENCY_HEAD_DECAY` for isolation.
- The winner's `persp_home` rides OUTSIDE feature_keep.json (augmentation
  artifact, not a selected feature); predict_win pins it to 1.0 at serve.
- Old artifacts keep serving: every new key is opt-in at read time.
