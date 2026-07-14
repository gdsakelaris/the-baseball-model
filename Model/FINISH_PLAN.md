# FINISH PLAN — the one-shot completion batch (2026-07-14)

**Goal (user directive):** bring the model to "finished" — no executable improvement,
polish, or cleanup left on the board. Everything in `FEATURE_BACKLOG.md` that CAN be built
goes in **one batch, one training chain, one verdict** — the all-at-once attribution risk
is explicitly accepted (time over bisectability).

**What "finished" means here, honestly:** every item that can be executed *now* is built,
evaluated, and either shipped or adjudicated based on user choice; tools, docs,
and the repo are consistent with what ships. It does NOT mean frozen: the daily 06:00
retrain continues, and a short list of **time-gated** items (Phase 8) cannot be collapsed
into now by any amount of effort — they wait on calendar, data coverage, or trust that only
accumulates. When Phases 0–7 are done and Phase 8 is the only thing left, the model is
finished in the only sense that exists.

---

## Decision gates (settle BEFORE starting — each blocks a phase)

| Gate         | Question                                                                                                                                                                                        | Options / recommendation                                                                                                                                                                                                                                                                                                                                 |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **G1** | Mega-batch verdict protocol: the paired read covers ~36 heads at once; MIXED is likely and CANNOT be cheaply bisected. Pre-commit now.                                                          | (a) ship-if-net-positive under the standing policy (≥1 CI-clear north-star win, same-sign 2026, no CI-clear harm — harm anywhere blocks); (b) revert-all on MIXED; (c) budget N hours of mechanism-level bisect (new heads and sim changes ARE separable; feature riders are not). Recommend (a) with (c) as fallback for a single-head CI-clear harm. |
| **G2** | Re-run the families-aware PROP_PARAMS sweep after the batch? Current overrides were swept 07-13 against the pre-batch frames.                                                                   | Skip in-chain (hours); re-sweep only if the verdict is MIXED with calibration-flavored harm.                                                                                                                                                                                                                                                             |
| **G3** | After H1 ships,`prop_rankings.BATTER_X` rows lose their only uniquely-priced lines. What do the x-rows grade then?                                                                            | Recommend: retire the x-rows from the board (means still feed the binary display); alternative: keep them grading the raw count means.                                                                                                                                                                                                                   |
| **G4** | Dead routing machinery:`PROP_EXCLUDE = {}` probe line ([train.py:398](train.py#L398)) + `st_exclude` pop loop ([train.py:516](train.py#L516)) sit on top of ~70 lines of dead curated tables. | Recommend: DELETE probe lines and tables both (selection is the sole decider per 2026-07-10 policy; git history keeps the tables). Alternative: leave as-is, documented.                                                                                                                                                                                 |
| **G5** | H2 (F5 markets) needs a per-inning linescore scrape for grading. Build the scrape in this batch so history accrues while sim-blend trust builds?                                                | Recommend yes (small scraper, statsapi linescore endpoint, new`Data/mlb_linescores.csv` + validate_data entry) — but the F5 MARKET stays parked either way.                                                                                                                                                                                           |
| **G6** | Repo data policy:`Data/*.csv` and `Data/raw_pitches/` (hundreds of MB) are untracked.                                                                                                       | Recommend: gitignore`raw_pitches/`; user decides CSV tracking. Needed so Phase 7's commit is deliberate, not accidental.                                                                                                                                                                                                                               |

---

## Phase 0 — Preconditions (½ hour)

1. **Commit the current tree** as the restore point (working tree is dirty; the last clean
   state is commit `6cd02de`). Settle G6 first so the commit is intentional.
2. **Snapshot artifacts**: `models.joblib`, `models_bt.joblib`, `feature_keep.json` →
   dated `.bak` (existing precedent: `*.lgbm-ship-0712.bak`).
3. **Baselines are current** (both set 07-14 post-Phase-3; verify 06:00 ran clean today).
4. Note: while experiment code is in the tree, `baseline_code_fp.json` sends the 06:00
   task scrape-only — designed behavior, no action. Plan the chain to finish before a
   morning; set baselines after ship so the daily resumes full runs.

## Phase 1 — Build every buildable feature (backlog #15–32 + one leftover)

The engineering lift. Every column: vectorized frame path + `Stores` serving path
(parity by construction via the shared-helper idiom), then into the relevant
`*_feature_cols()` superset. ~50–55 new columns total.

**1A. Raw-pitch re-agg batch** (schema v6, one `--from-raw` pass ≈ 4 min, 12 seasons):

- #25 pitcher-side sequencing + count states: 0-2 waste rate, ahead/behind usage shift,
  pitch-class transition shares (tunneling proxy) → `pd_*`.
- #26 in-game velo fade: per-start OLS slope of FF/SI velo vs pitch number, decayed →
  `pd_fbv_fade`.
- #27 movement axes (archive verified — pfx/api_break/spin/arm_angle all present):
  `pd_ivb_d` + ride-vs-flyball collision with `bd_fbwh`/`bip_air`.

**1B. Game-log features** (no new data):

- #15 bullpen exposure: `xpa_pen` + `xpa_pen ×` (pen−starter) K/HR/H deltas.
- #16 outing shape/policy: `p_outs_sd`, `p_short_share`, `team_st_outs_pg`.
- #17 layoff/ramp flags: 15+ day gap, `p_np_last < 60`, short-rest-after-100+NP.
- #18 schedule: `day_after_night`, `travel_km` + `tz_delta` (ballparks Lat/Lon).
- #20 BaseRuns cluster-luck residual (trailing-30d, team grain).
- #21 ump run environment: shrunk as-of R/G with this HP ump (`_ump_shrink` idiom).
- #23 starter venue splits: `pvloc_era/k/hr` (shrunk, as-of).
- #24 doubleheader flags: `is_dh`, `dh_game2`.

**1C. Posted-lineup / game-grain plumbing** (the one real plumbing job — a shared
lineup-aggregate table consumed by the game frame AND serving):

- #19 B-lineup detector: posted-lineup mean as-of career OBP/SLG minus `toff_` norm.
- #30 `air_dens ×` lineup air profile.
- #32 cross-grain arsenal collision: lineup class-whiff × opposing starter usage
  (the `lu_mix_k` design at team grain — highest-value unbuilt composite).

**1D. Composites on existing columns** (cheap, same-day):

- #28 `per` conversion chain: `(pc_h_bf + pc_bb_bf) × pbipd_xwoba` + decayed sibling.
- #29 `ump_k_pct × lu_k_sh` for the K head.
- #31 form-weighted exposure: `xpa_slot × d_{hr_pa,tb_ab,k_pct}_sh` (credit-splitting
  expected — let the vote pick career vs decayed per head).

**1E. Close the last data-gap leftover:** Precip is in `mlb_weather.csv` but is not a
feature and has no serving forecast (backlog #11 note). Either wire it (forecast field in
`1_get_todays_games.py` + GUI + one feature col) or adjudicate "not a feature" and mark
#11 CLOSED. Recommend wiring it — it's the outs/total head's rain-shortening signal.

**1F. Selection-machinery upgrades** (adopted 2026-07-14 from the selection-process
review; changes to `feature_select.py` + frame assembly, built before the Phase-4 chain):

- **Shadow-calibrated eps** (Boruta-lite): append ~20 shuffled copies of representative
  superset columns (fixed seed, spread across frames and dtypes) to the frames;
  `feature_select` sets each head's eps to a high quantile (start p95) of its
  shadow-column SHAP shares instead of the fixed `EPS = 0.0005` constant — the
  "essentially unused" floor becomes empirical per head. Shadows ride the Phase-4
  superset train at zero marginal training cost; keep-lists are written shadow-free.
- **Cluster co-failure report**: Spearman-cluster the superset once (diagnostic only);
  the selection report flags groups whose members EACH narrowly miss PI (vote in
  ~[0.55, 0.75)) while the group's combined mean share is large — the clones-voting-
  each-other-out failure (risk R3, and #31's expected regime). REPORT only, never
  auto-keep: flagged groups are gray zones and go to the user.
- **Declined from the same review** (recorded so they aren't re-proposed): hard
  correlation pre-cut (kills the multi-horizon `c_/s_/r*/d_` variants and standalone-
  useless interaction feeders); time-block subsampling of the vote (two-suite era
  stability + held-out cal-year SHAP + the day-block paired CI already cover it, in
  stronger form); LGBM-only voting (the vote is normalized per-family use-fractions with
  equal weight — not cross-architecture raw-SHAP averaging — and the serving XGB/CB
  members need their vote). LGBM-only stays adjudicable anytime via the existing
  `--families lgbm` flag + one keep-train + paired read; deliberately NOT scheduled.

## Phase 2 — New heads: H1 + H3 + H4 + H5 + H6 — 24 → 36

**Head census (current 24 → finished 36):**

| Tier            | Current                                                                         | Added by this plan                                 |
| --------------- | ------------------------------------------------------------------------------- | -------------------------------------------------- |
| Batter binaries | hr, hit, hits2, tb2, single, double, run, rbi, bb, sb, bk, bk2, hrr2, hrr3 (14) | bk3, tb3, tb4, hrr4, triple, rbi2, run2 (+7 → 21) |
| Batter counts   | xbk, xhrr, xtb (3)                                                              | xh, xrun, xrbi, xbb (+4 → 7)                      |
| Starter counts  | k, outs, pbb, pha, per (5)                                                      | —                                                 |
| Game / team     | total, winner (2)                                                               | team_total (+1 → 3)                               |

35 trained GBM heads + `team_total`, which rides the already-trained team-runs GBM
(shared with `total`) but gets full head identity: its own dispersion, line calibrators,
eval read, and board row.

Per the specs in FEATURE_BACKLOG Part 3 #H1/#H3/#H4/#H5/#H6:

- Targets `y_bk3/y_tb3/y_tb4/y_hrr4` next to siblings in features.py; 4 `PROPS` entries
  ([train.py:429](train.py#L429)); `PLATT_CAL` covers them automatically (`set(PROPS)`).
- **Acceptance bars are pre-measured** (`count_vs_binary.py` table in the backlog): each
  head must beat its banked count-calibrator's logloss/AUC on BOTH years, else that line
  ships count-priced (a measured mixed board is fine; an accidental one was not).
- The keep-list clobber trap is MOOT in this batch — the full-board selection regen is
  intended (Phase 4). The H1 "one train or two" open question dissolves the same way.
- **H3 `triple`** (user ask 07-14): `y_3b` target one-liner (game logs carry `3B`,
  verified), one `PROPS` entry, `park_3b_pg` as-of factor rider. Base rate 1.21% per
  batter-game (2025) — thinnest board binary; Platt calibration is automatic and
  load-bearing. No banked count-calibrator bar exists, so acceptance = the standing
  gates (CI-clear edge over base rate, honest ECE, no harm elsewhere).
- **H4 `rbi2` + `run2`** (user ask 07-14): one-liner targets next to their 1+ siblings
  (`y_rbi2`, `y_run2`); measured base rates 9.31% / 7.23% (2025) — thicker than bk3.
  No banked count-calibrator bars → standing gates. Platt automatic.
- **H5 `team_total`** (user ask 07-14): head-ify the trained team-runs GBM (per-team
  means already computed and sim-blended at serving, [predict.py:748](predict.py#L748)):
  measure a TEAM-level cal-year NB dispersion (the game total's 2.28 does NOT transfer),
  NB P(over) for team lines (2.5–5.5), display the blended per-team means, grade from
  final scores (`mlb_games`), evaluate_deep team-line read.
- **H6 `xh` / `xrun` / `xrbi` / `xbb`** (user ask 07-14): four `COUNT_HEADS` entries
  completing the expected-stat-line; `xrbi` Tweedie 1.3 (measured var/mean 1.61),
  the rest Poisson. **Means only** — per-line calibrators are BANKED, never shipped
  (binaries own batter lines, 07-13 verdict). Standing count-head gates. Declined +
  recorded: xhr/xsb/hit-type counts (rare events — the binary already is the mean).
- Downstream wiring (all known, shared by the eleven new batter heads + team_total):
  `predict.PROP_COLS/BAT_HEADERS/BAT_ORDER/PCT_COLS/GLOSSARY`;
  `4_grade_results.BAT_EVENTS` (hit_rate_report follows for free) + Games-sheet rows for
  team totals; `prop_rankings.BIN_NAMES` + `BINARY_OWNED_LINES` + the G3 decision;
  `Props.txt` → 62 + team-total lines.

## Phase 3 — PA-sim finishing

1. **Hazard v2** (bf-relative) — the queued outs fix; the single biggest unknown in this
   plan (engine work in `pa_engine`/`pa_sim`, then `pa_backtest` regrade).
2. **Steal-layer blend re-sweep** — currently w=0; re-sweep after hazard v2 + the new sb
   context lands (`pa_blend`).
3. **SIM_BLEND re-sweep** — the {.35/.20/.30} weights were fit against the incumbent
   GBMs; after the mega-batch retrain they are stale by construction. Re-run `pa_blend`
   cross-year, update `predict.py` weights.
4. `pa_grade` + backtest parquets refresh; `pa_serve` smoke (15/15 precedent).

## Phase 4 — The one training chain (~4 h, overnight)

1. `train.py --rebuild` (frame schema changed everywhere).
2. **Full two-train superset recipe** (this IS the wholesale re-litigation case): superset
   train both suites (frames carrying the 1F shadow columns) →
   `feature_select.py --stat shap --write` (all three families, both suites — intended
   full-board regen, includes the 5 new heads; shadow-quantile eps + co-failure report
   per 1F) → keep-list train both suites. Line calibrators, NB dispersions, quality boot,
   MiLB rider joins all ride the normal train.
3. G2 says skip the PROP_PARAMS re-sweep unless the verdict demands it.

## Phase 5 — Evaluation and the verdict

1. `evaluate_deep.py --paired` on 2025 (selection suite) vs the Phase-0 baseline — read
   **all 36 heads** (35 GBM-trained + the team_total line surface), balanced
   AUC/logloss/ECE gates, day-block CI as arbiter.
2. H1 heads graded against their pre-measured count-calibrator bars (not the baseline);
   H3/H4 against the standing gates (no banked bars exist); H6 against the standing
   count-head gates (MAE vs naive, honest dispersion); H5's team-line calibration read
   is new in evaluate_deep, plus the H6 coherence read (Σ lineup xrun vs team_total).
3. `evaluate_deep.py --confirm` ONCE on 2026 (same-sign requirement).
4. Review the 1F co-failure report: any flagged clone group (members individually below
   PI, group share large) is a gray zone the user adjudicates before the verdict is final.
5. Apply G1's pre-committed protocol. Gray zones → user decides, per standing policy.
6. On ship: set BOTH baselines; verify next 06:00 daily runs full.

## Phase 6 — Serving & tools

- `predict.py --selftest` parity for every new column + head (the gate for 1A–1E; budget
  real debugging time here — ~55 cols × two paths is where mistakes will surface).
- GUI slate smoke (blend path + new prop columns + precip field if 1E wired).
- `4_grade_results` / `hit_rate_report` / `prop_rankings` run end-to-end on a graded day.
- `Tools/1_get_todays_games.py` serving inputs for #18/#24 derive from the schedule — no
  new scrape-time inputs needed (travel/day-night/DH from games history + today's slate).

## Phase 7 — Cleanup, docs, repo

- **G4**: delete the nuclear-probe lines + dead routing tables (or the documented
  alternative). Also delete `STACK_DONORS = {}` machinery? NO — benched levers stay
  (adjudicated, documented); only unreachable-by-policy code goes.
- **README.md refresh**: metrics table is pre-2026-07 ("~100 features per batter-game",
  "~2 min retrain", old holdout table) — regenerate from current `metrics.json`; document
  28 heads, SIM_BLEND serving, MiLB riders, selection-as-curation, the 06:00 task.
- **Artifacts hygiene**: prune stale `.bak`/`pre_*` snapshots, keep only the Phase-0 pair.
- **Backlog + memory**: mark #15–32 and #H1 with their verdicts; log entry; update memory.
- **Commit** (per G6 policy), message documenting the batch scope.

## Phase 8 — What remains after "finished" (time-gated; no effort collapses these)

| Item                                                                                               | Gates on                                                                  | When                         |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------- |
| Era audit (xhrr/xtb accepted harm ×2, winner keep-list, double shield, hits2 softness, sb repair) | a month of 2026 in-season data                                            | ~mid-Aug                     |
| bb recal check (recal stays OFF until then)                                                        | in-season sample size                                                     | Aug                          |
| H2: F5 markets via the sim                                                                         | linescore grading history (G5 scrape) + sim-blend trust from weekly reads | weeks                        |
| Bat tracking unshelf (pitch-level`bat_speed`/`swing_length` already in the archive)            | data coverage vs training window                                          | ~2027                        |
| Catcher framing / pop time                                                                         | a serving-time catcher source that doesn't exist                          | shelved                      |
| Odds as features                                                                                   | program-level phase rule (STATS-ONLY) — a policy decision, not a task    | user's call, not this plan's |

## Risk register

- **R1 — attribution loss** (accepted): one chain, ~28-head verdict; G1 governs.
- **R2 — selection churn**: the full keep-list regen re-litigates every head; expect
  keep-list diffs even where nothing changed. Paired day-block CI absorbs the jitter.
- **R3 — credit-splitting**: #31 (and #15 vs existing `xpa_x_*`) compete with parents in
  the vote; either surviving is success. Mitigated by the 1F co-failure report, which
  surfaces the both-members-die case for a user call instead of a silent drop.
- **R4 — small-n frames**: many new columns target winner/runs (~10k rows, documented
  overfit history). The paired-CI read, not the keep-vote, is the arbiter there.
- **R5 — parity surface**: ~55 new columns × two code paths; `--selftest` is the gate.
- **R6 — rollback**: Phase-0 commit + artifact `.bak` pair + the contamination-recovery
  recipe. Reverting = restore artifacts, revert commit, retrain nothing.

## Sequencing (aggressive, time-is-of-the-essence)

- **Day 1:** Phase 1 (1A–1E) + Phase 2 wiring + Phase 3.1 hazard v2. The build is the
  bottleneck — realistically 1–2 focused days; hazard v2 is the wildcard.
- **Night 1:** Phase 4 chain (~4 h) + sim backtests/sweeps queued behind it.
- **Day 2:** Phase 5 reads + verdict; Phase 3.2–3.4 blend re-sweeps; Phase 6 smokes;
  Phase 7 cleanup, docs, commit; set baselines.
- **Done** = Phases 0–7 complete, Phase 8 table is the entire remaining surface.
