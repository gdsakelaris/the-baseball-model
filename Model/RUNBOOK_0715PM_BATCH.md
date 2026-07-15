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

Artifact contract: `meta_stamp.artifact_version = 3`; new prop keys
`fstack`/`fstack_fams`/`fam_slices` (binaries/winner), `FamilyBlendBag`
(counts), `BaggedCal` under the `iso` key. predict.py serves all of it and
stays backward-compatible with the current `models.joblib`.

## Order of operations (pre-chain sweeps first — their winners bake in)

Timing probes: after step 1, extrapolate wall-clock per head x 36 heads x 2
suites and check it fits the 06:00 window. CB is 2-4x an LGBM fit (GPU) and
ES-refit roughly doubles booster time — expect ~3-5x the LGBM-only chain.
If the window is at risk: drop `ES_REFIT` first (cheapest accuracy give-back),
then `CB_BAGS = 1`.

0. **Back up the serving artifacts** (serving continuity if adjudication
   rejects — the chain OVERWRITES them and they are not in git):
   copy `models.joblib`, `models_bt.joblib`, `metrics.json`,
   `metrics_select.json`, `inseason_offsets.json`,
   `eval_baseline_2026.json`, `eval_baseline_select_2025.json` to
   `Model/artifacts/pre_0715pm_backup/`. Restoring the incumbent is then a
   copy-back, not a flags-off retrain, and the GUI keeps a servable
   artifact through the whole adjudication window.
1. **Timing probe** (~minutes): `python Model/train.py --select` and watch
   the first few heads' per-head wall clock. Abort freely — nothing ships
   from a `--select` run. (Baseline guard note: with uncommitted changes in
   the tree, tomorrow's 06:00 job already goes scrape-only via
   `baseline_code_fp.json`, so iterating today is safe. Do NOT commit
   until the ship decision.)
2. **HPO sweep** (hours, GPU idle otherwise): 
   `python Model/hpo_sweep.py --trials 60`
   (weak-head default list: double, hit, single, rbi, hrr2/3/4, run2, rbi2).
   Wire gate-clearing winners into `PROP_PARAMS` / `COUNT_PARAMS`.
3. **Per-head decay sweep** (5 x `--select` train, resumable):
   `python Model/decay_sweep.py` — now prints a paste-ready
   `RECENCY_HEAD_DECAY` dict; bake clear winners only (per-head CV jitter
   caveat is printed with it).
4. **The chain** (the one decision-relevant run):
   `python Model/train.py --prestash`
   - `--prestash` trains one extra throwaway suite so the SELECTION suite
     also gets multi-year cal support — without it the 2025 paired read
     cannot see change #4 (everything else it sees). Costs ~+50% chain time
     ONCE; the daily job keeps running plain (shipping suite still pools).
5. **Re-fit SIM_BLEND against the new heads** (gap the first runbook draft
   missed): the shipped {score .60, total .55, winner .30} weights were fit
   on 2025 against the OLD GBM game heads; the batch moves those heads
   (stack + CB + mirror + refit), so the sim-vs-GBM tradeoff shifts. Re-run
   the pa_grade/pa_blend 2025 fit after the keep-train and take its
   weights (fail-safe unchanged: sim failure degrades to the GBM alone).
   evaluate_deep verdicts the raw GBM heads either way — this step only
   affects the serving blend layer.
6. **Adjudication** (the standard bar, all 24+ heads read, no pre-declared
   targets):
   - `python Model/evaluate_deep.py --paired` (2025, selection suite)
   - `python Model/evaluate_deep.py --confirm` (2026, deliberate look)
   - `python Model/predict.py --selftest` + a multi-game slate smoke
     through the GUI's Predictor (the 07-14 chain pattern) — proves the
     new artifact contract serves end-to-end, not just in eval.
   - Per-flag verdicts are independent: a harmed mechanism reverts by its
     flag without unwinding the batch. STACK_DONORS heads verdict per head.
7. **USER DECIDES ship/revert.** If ship: re-stamp baselines
   (`--set-baseline` x2 per the standard chain), commit, and let the 06:00
   job resume. First retrain shifts every baseline (ES split + refit +
   stack change the whole surface — same situation as the 07-15 AM ship).
   If revert: restore `pre_0715pm_backup/` and flip the rejected flags.

## Decision points still open (deliberately not decided in code)

- **Feature-selection regen**: CB rejoined on a keep-list it never voted
  on. Mechanically fine (feature_select votes whatever families are
  present, `_family` handles CatSafe), but the next regen should follow the
  approved shadow-superset design (fresh superset train ->
  `models_superset*.joblib`). The superset electorate now carries 1 CB bag
  (`CB_BAGS` capped in the `_KEEP_TRAIN` block) — cheap-electorate
  regime mismatch widens accordingly; era-audit-time call.
- **param_sweep --ensemble re-run**: the PROP_PARAMS profiles were swept
  LGBM-only. A families-aware re-sweep (`param_sweep.py --ensemble`) is the
  honest objective now that CB ships — tie it to the era audit per the
  RE-SWEEP CADENCE note, not to this chain.
- **CAL_POOL_DECAY = 0.75** is a reasoned default, not swept. Sweepable
  later the same way RECENCY_DECAY was.

## Interactions worth knowing

- Order of serving ops is unchanged: `predict_prop` -> `apply_stack`
  (STACK_DONORS) -> `enforce_ladders` (PAV), identical in evaluate_deep.
- `--decay X` (decay_sweep) clears `RECENCY_HEAD_DECAY` for isolation.
- The winner's `persp_home` rides OUTSIDE feature_keep.json (augmentation
  artifact, not a selected feature); predict_win pins it to 1.0 at serve.
- Old artifacts keep serving: every new key is opt-in at read time.
