# CHAIN 2 plan — the second no-regen improvement wave

## EXECUTION LOG (2026-07-16 afternoon — Phases A+B COMPLETE, wired, C pending)

Phase A: A1 cal_lab (ship stash) -> pool=1/bag=0 global winner, WIRED
(CAL_POOL_YEARS=1, CAL_BAG_B=0). A2 vmr_fit -> a=0.0, inert json written,
no wire. A3 cb_sweep -> CB_CLS/CB_POIS keep default (9/11 panel heads);
CB_WIN 'bayes_hot' (bagging_temperature=2.0, ll -.0020) WIRED. A4 count
HPO 60-trial -> 9/13 solo RECOMMENDs, ensemble tiebreak
(tiebreak_0716.json, same method as chain 1) killed 6 — outs/per/k
Optuna dicts WIRED into COUNT_PARAMS verbatim. A5 decay refinement:
USER call = skip (park for era audit).

Phase B: B1 init_score donors BUILT — USER design call = in-sample donor
logits + per-head scale probe on cal year, grid (0,.25,.5,.75,1.0),
scale 0 self-gates; offset rides at family-logit level, serve twin in
predict_prop "init_donor" branch; 6 ladder pairs live in
INIT_SCORE_DONORS; 47/47 tests incl. exact fit/serve parity. B2 mirror-
aware winner sweep BUILT (param_sweep/_prep(mirror)/_fold_fit_binary/
_fit_families_fold + hpo passthrough) and RUN -> LGB_WIN 'reg_up'
(nl15/mcs300, ll -.0006) WIRED. B3 BAG_DIVERSIFY=2 flipped.

xrbi (chain-1 replicated MAE harm): Optuna winner AND old _CNT_HEAVY both
failed the ensemble read (base best on MAE) — stays on base, harm suspect
moves elsewhere, chain-2 paired read re-measures.

Also: SIM_BLEND {.40/.15/.30} baked into predict.py pre-slate (USER call,
07-16 first pitch serves the fitted weights); fp-guard consequently stale
until Phase C re-stamps.

Phase C RESULT (2026-07-16 evening — USER RATIFIED "ship all"): keep-train
16:17->~17:30 (fast — bag-off removed the 25x calibrator fits), ALL 12
B1 recipient fits picked NONZERO donor scales (hits2 hit@1/.5, tb4
tb3@1/.75, hrr4 hrr3@.5/.75, triple double@1/.25, rbi2 rbi@.75/.5, run2
run@1/1 — zero self-gates). Paired 2025 (FDR q<=.1, 77 tests): IMPROVED
= sb edge +.0005 + sb ECE +.0037 + xhrr/xtb/xbb MAE + pha MAE (q=.078)
+ **xrbi MAE +.0013 q=0 — CHAIN 1's REPLICATED HARM REVERSED with xrbi
untouched** (leave-base vindicated; the old suspect, dropped _CNT_HEAVY,
was innocent) + hit slope 1.04->1.00 and triple slope 1.07->.97 (band).
HARM = bk2 edge -.0002 (q=.078), sb slope band 1.00->1.04, double slope
band .87->.83. Confirm 2026: 0 better / 1 worse (hits2 ECE +.0033
band-cross) / 111 noise — NO 2025 harm replicates, no win flips sign,
winner leans positive (acc +.0056, ll -.0005, noise). Selftest PARITY OK
3.8e-12. WATCH LEDGER (era audit): bk2 edge tick, double slope drift
(.87->.83, worst-slope head again), hits2 ECE band-cross (donor head),
sb slope 1.04. Close-out: baselines x2 -> pa_blend re-fit (bake = its
own USER call) -> commit.

Written 2026-07-15 late PM, while chain 1 (the 20-mechanism diversity/
calibration batch, RUNBOOK_0715PM_BATCH.md) was still running. Chain 2 is
ONE ordinary keep-train (~2.5h, no superset/selection regen — the feature
set and family roster don't change), carrying everything below that clears
its evidence pass. Every item reverts by its own flag/constant; the paired
read + user adjudication bar is unchanged.

PRECONDITION: chain 1 adjudicated (user ship/revert per flag). Chain-2
sweeps run against whatever survived — a reverted chain-1 flag changes the
baseline the sweeps measure from, so run the sweeps AFTER the verdicts.

## Phase A — daytime RUNS (free GPU/CPU after the chain-1 package)

| Run | Command | Produces | Wire target |
|---|---|---|---|
| A1 Pricing-knob sweep (offline, minutes) | `python Model/cal_lab.py` | global combo ranking over CAL_POOL_YEARS (2 vs 3) x CAL_POOL_DECAY x FSTACK_C x CAL_BAG_B, scored on 2025 | the four constants in train.py |
| A2 Park-dispersion evidence (minutes) | `python Model/vmr_fit.py` | `artifacts/total_vmr_exp.json`; predict's sidecar fallback activates only on `recommended: true` | nothing — self-activating, delete file to revert |
| A3 CatBoost profile sweep (~2-5h GPU) | `python Model/cb_sweep.py` | first-ever CB_CLS/CB_POIS evidence (11 profiles, representative 11-head panel, ensemble objective, global decision) | `CB_CLS` / `CB_POIS` fragments |
| A4 Count-head HPO (~1-2h CPU, parallel with A3) | `python Model/hpo_sweep.py --heads k,xbk,xhrr,xtb,xrbi,xh,xrun,xbb,outs,pbb,pha,per,total --trials 60` | deviance-gated Optuna winners vs the (now empty) COUNT_PARAMS baseline | `COUNT_PARAMS` |
| A5 Decay refinements (only if warranted) | finer `decay_sweep.py` grid around any per-head winner from chain 1's 5-grid | refined `RECENCY_HEAD_DECAY` entries | `RECENCY_HEAD_DECAY` |

Conflict rule (learned from chain 1's rbi/hit/hrr3 tiebreak): if A3 and A4
touch the same count heads' ensembles, reconcile ensemble-scored before
wiring — solo-objective winners mostly don't survive the ensemble read.

## Phase B — daytime BUILDS (clear-headed design work, not midnight code)

- **B1 init_score donor warm-starts** (train-time transfer for the thin
  heads: hits2/tb4/hrr4/rbi2/run2/triple boost FROM the lower rung's logit,
  learning only the residual — the STACK_DONORS idea inside the trees).
  OPEN DESIGN QUESTION to settle first: the donor's logits on the TRAIN
  rows are in-sample (the donor trained on them), which makes the offset
  optimistically sharp — options are (a) donor's ES-fit member scores,
  (b) a K-fold OOF donor pass (extra fits), (c) accept in-sample with a
  shrunk offset scale. Decide, then implement flag-gated
  (`INIT_SCORE_DONORS = {}`), OFF until its own paired read.
- **B2 Winner sweep under mirroring** — ONLY if chain 1's paired read keeps
  WINNER_MIRROR: teach param_sweep/hpo_sweep's winner job to mirror its
  fold-train rows (persp flag included) so LGB_WIN is tuned for the data
  regime it actually ships in.
- **B3 `BAG_DIVERSIFY = 2`** — no build needed (wired, OFF); just flip for
  the chain-2 adjudication.

## Phase C — the chain

Wire every gate-clearing Phase-A winner + B1 (if built) + B3 → run
`python Model/train.py` (plain; add `--prestash` only if the cal-lab
verdicts made depth-3 pooling a candidate for the SELECTION suite's read
too) → SIM_BLEND re-fit if the game heads moved → `evaluate_deep --paired`
+ `--confirm` → **USER adjudicates per flag** → baselines + commit.

## Parked with explicit triggers (NOT chain 2)

- **Forward-record calibration monitor** (`Tools/6_forward_calibration.py`,
  built): first meaningful read ~2 weeks of graded record (~200+ cells/
  head); weekly cadence after that; feeds the Aug bb-recal check and the
  era-audit watch ledger. No model change — a monitor.
- **In-season recal offsets**: August, per the standing monitors ledger
  (needs Section-10 forward evidence; recal stays off by phase rule).
- **Betting-decision layer** (bet sizing from calibrated edge;
  correlation-aware same-slate exposure priced from the PA sim's JOINT
  outcomes — the sim asset no marginal-calibration work can substitute):
  separate Tools project after the ship settles. Scope sketch: sizing
  first (pure Tools, odds stay grading-only), sim-joint extraction second
  (pa_engine surface work).
- **TabM 4th family**: era-audit-scale project; requires the full regen
  (new family = electorate change). The chain-1 evidence (CB earning
  large stack weights, e.g. count fam_w pbb = 1.0 cb) strengthens the
  diversity case; the stack makes a weak 4th member near-free to carry.
- **Era audit** ~mid-Aug: standing watch-ledger items + any chain-1/2
  reverts owed a re-look.

## Standing constraints (unchanged)

STATS-ONLY (odds never features); 2026 confirm-only; no pre-declared
targets — all heads read; user decides every accept/decline; first
retrain after any wire shifts all baselines.
