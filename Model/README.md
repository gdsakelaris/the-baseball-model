# MLB Prediction Model

Seed-bagged LightGBM ensembles (6 LGBM bags + 2 CatBoost family members
per head — CatBoost restored and XGBoost permanently retired 2026-07-15 PM
— combined with a per-head logistic-regression member by a **per-family
logistic stack** fit on the calibration support) trained on the CSVs
in `Data/`, predicting per game (or per slate of games — the GUI's "Add
game to slate" runs many at once with one combined, cross-game-ranked
board). **36 heads** ship (21 batter binaries + 7 batter counts + 5
starter counts + 3 game-level), every one calibrated:

- **Per lineup batter — 21 binary props:** home run (with fair American
  odds), 1+/2+ hits, single, double, triple, 2+/3+/4+ total bases, run
  scored, 2+ runs, 1+/2+ RBIs, 1+ walk, stolen base, 1+/2+/3+
  strikeouts, and 2+/3+/4+ hits+runs+RBIs. **Plus 7 expected counts**
  (xH, xR, xRBI, xBB, xSO, xTB, xHRR) — means with honest dispersion;
  their half-point lines are banked as calibrators but the binaries
  remain the sellable prices (the count-vs-binary shoot-out verdict,
  07-13). A career-games flag marks players under ~50 MLB games (`*` in
  the GUI) — the least reliable segment.
- **Per starter:** expected strikeouts and P(over) for K lines 3.5–8.5,
  negative-binomial-shaded by the calibration-year K dispersion; plus
  outs recorded, walks allowed, hits allowed, and earned runs with
  per-line P(over) from cal-year logistic line calibrators; and an
  expected ERA derived from the earned-runs and outs heads
  (27 × xER / xOuts). Single-start ERA is inherently noisy — the
  earned-run over/under lines are the sellable output.
- **Per game:** expected total runs with P(over) from a negative binomial
  whose dispersion is measured on the calibration year (real totals run
  ~2x Poisson variance); **per-team totals** with their own line
  calibrators; expected lineup home runs; and a home-win probability
  from a dedicated winner model (team win%, run differential,
  pythagorean expectation, Elo with cross-season carryover, recent form,
  both starters, bullpens, rest). Game-level outputs are then blended
  with a **plate-appearance-level Monte Carlo simulation** of the game
  (`pa_serve.SlateSim`, 1,500 sims): `SIM_BLEND` weights
  {score 0.60, total 0.55, winner 0.30}, fit on 2025 only (2026-07-15);
  any sim failure degrades cleanly to the GBM alone. The winner output
  is presented as a **calibrated probability, not a pick**: on the
  holdout it shows no significant edge over always taking the home team.

## Files

| File | Role |
|---|---|
| `features.py` | Feature engineering. Everything is **as-of date**: a game on date D only uses data from before D (no leakage). Same definitions serve training (vectorized) and prediction (per-entity); `predict.py --selftest` proves the two paths agree. |
| `train.py` | Builds frames, trains everything. TWO ROLES (2026-07-15): with `feature_keep.json` present it is the KEEP-train and writes the serving artifacts (`models_bt.joblib` selection suite, `models.joblib` shipping suite); without it it is the SUPERSET train and writes `models_superset*.joblib` (feature_select's electorate — cheaper regime: fewer bags + a row sample, documented mismatch). Train/cal/holdout years are DERIVED from the data (`suite_years`) — currently ≤2023/2024/2025 and ≤2024/2025/2026; boosters early-stop on a held-out ~10% GamePk slice of the TRAINING rows (`_es_split`), never the calibration year. Heads are seed-bagged LightGBM (`LGBM_BAGS = 6`) + CatBoost members (`CB_BAGS = 2`; XGBoost retired), with **recency-decayed sample weights** (`RECENCY_DECAY = 0.95` global, per-head overrides in `RECENCY_HEAD_DECAY`), **ES-refit** (each member refit on 100% of train at its early-stop iteration, keep-trains only), a **per-family logistic stack** over [LGBM logit, CB logit, LR logit(, Poisson logit for the winner)] replacing the single blend weight (`FAMILY_STACK`; counts use a deviance-chosen family weight), a per-head **auto-selected calibrator** (`AUTO_CAL`: Platt / beta / isotonic) that is **day-block bootstrap-bagged** (`CAL_BAG_B = 25`) and fit on **multi-year pooled support** when available (`MULTI_YEAR_CAL`: the selection suite's honest cal-year scores feed the shipping suite at zero extra train cost, prior year discounted `CAL_POOL_DECAY = 0.75`), a **mirrored winner augmentation** (`WINNER_MIRROR`: home/away-swapped train copies + `persp_home` flag, doubling the 10k-row winner's data), and **ladder coherence** (PAV projection so e.g. P(2+ hits) ≤ P(1+ hit) always). Per-head hyperparameter overrides live in `PROP_PARAMS` (binaries) and `COUNT_PARAMS` (counts), tuned by `param_sweep.py`. Artifacts carry a `meta_stamp` (role/regime/created) that predict.py's serving guard enforces. `--rebuild` refreshes cached frames; `--select` stops after the selection suite. |
| `feature_select.py` | Shadow-calibrated stability selection (Boruta-style): the superset train plants shuffled `shdw_` decoys, per-head SHAP shares are measured against the shadow quantile (`SHADOW_Q = 0.85`), and the surviving columns are written to `feature_keep.json` — the **sole decider** of what each head trains on. The old hand-curated per-prop routing tables were retired 2026-07-10 and deleted 2026-07-15 (git history keeps them). |
| `param_sweep.py` | Per-head LightGBM hyperparameter sweep: day-grouped GroupKFold OOF on the training years, single-member isolation, deviance/ECE-guarded acceptance gates. Winners are wired into `PROP_PARAMS`/`COUNT_PARAMS` by hand. |
| `predict.py` | `Predictor.predict_game(spec)`; CLI: `--game <GamePk>` replays a historical game, `--selftest` checks train/serve parity. Applies the `SIM_BLEND` layer to game-level outputs. Refuses superset-role or shadow-contract artifacts (serving guard). Every run saves a workbook to `Predictions/`. |
| `pa_sim.py` / `pa_engine.py` / `pa_model.py` | The plate-appearance simulator: matchup-calibrated PA outcome model, base-running/steal layer with **battery modulation** (opposing starter's steal permissiveness × catcher CSAA/pop, prior-season, `STEAL_BATTERY`), starter-hazard pulls (v2, bf-relative), MiLB-informed priors for thin MLB histories (`milb_priors.py`). |
| `pa_serve.py` | Serves the simulator at predict time (`SlateSim`); `pa_backtest.py` replays full seasons through it; `pa_grade.py` + `pa_blend.py` grade the sim vs the GBM incumbent and fit the per-head blend weights (2025 only). |
| `odds.py` | Sportsbook-odds utilities: American↔probability, de-vig, EV/ROI, and the canonical odds-store schema. Bridges the model's fair probabilities to the prices books actually post; shared by `Tools/2_scrape_odds.py` and evaluate_deep Section 9. |
| `recalibrate.py` | In-season drift correction: a per-prop log-odds shift fit only on recent in-season games (leakage-free). Stored by `train.py`, backtested in evaluate_deep Section 10, applied at serving with `predict.py --recal` (currently off by default — the backtest says the base calibration is already honest). |
| `evaluate_deep.py` | Full accuracy + betting workup on a held-out season: bootstrap CIs on AUC and the logloss edge, calibration, daily top-N hit rates, pick-threshold economics, monthly drift, segment reliability, K/totals/winner deep dives, **model-vs-market ROI (Section 9)**, **in-season recalibration backtest (Section 10)**, and `--set-baseline` regression diffing with per-metric noise bands (Section 11) plus `--paired` day-block bootstrap CIs with a Benjamini–Hochberg FDR gate. The DEFAULT run scores the selection suite on 2025 (iterate freely); `--confirm` scores the shipping suite on 2026 — deliberate looks only, except the one auto ship-confirm the daily job takes per ship (audit #6 as amended 07-15). |
| `decay_sweep.py` | Sweep that chose `RECENCY_DECAY` (banked runs under `artifacts/decay_sweep/`); now also prints a paste-ready per-head `RECENCY_HEAD_DECAY` dict (07-15 PM). |
| `hpo_sweep.py` | Optuna TPE search over the continuous LightGBM space for the weak heads, reusing `param_sweep.py`'s folds/scoring/gates verbatim (07-15 PM). Winners are wired into `PROP_PARAMS`/`COUNT_PARAMS` by hand. |
| `artifacts/` | Trained models (`models*.joblib`), metrics (`metrics.json`), cached feature frames, selection reports, sim tables/backtests, and the odds store / eval baselines. |

The game-day tools that USE these models live in `Tools/`, numbered in
run order: `1_get_todays_games.py`, `2_scrape_odds.py`, `3_gui.py` (the
Tkinter app), `4_grade_results.py` (`--all` grades every workbook + the
accumulated forward record), and `5_prop_rankings.py` (the per-market
quality board → `PROP_RANKINGS.xlsx`; renamed from `prop_rankings.py`
2026-07-15, replacing the retired `5_performance.py`/`PERFORMANCE.xlsx`).

## What the model knows (superset: 522 batter / 207 starter / 74 team-game / 55 winner columns; stability selection picks each head's ~20–90)

- **Batter form:** career / season-to-date / rolling 7-15-30-game HR rate,
  ISO, contact, K/BB rates, XBH rate, full OBP (incl. HBP); days rest;
  lineup slot; home/away; career home/road splits (venue-context rates);
  position wear (career share of games caught / DH'd); **90-day
  decay-weighted rates** (every past game weighted by recency — skill drift
  without a rolling window's cliff) and **trend deltas** (rolling-15 vs
  season, season vs career; last-5 vs season for the opposing starter).
- **Contact quality** (Statcast, every tracked ball in play — not just
  homers): shrunk career + 90-day-decayed exit velo, hard-hit%, barrel
  rate, launch angle, xBA / xwOBA on contact, ground-ball share,
  spray-based pull% / pulled-air% (which interact with the park's fence
  distances), and damage-on-contact splits by trajectory — for the batter
  AND allowed by the opposing starter. Process stats stabilize in ~40
  batted balls vs hundreds of AB for outcomes, so these see skill (and
  in-season change) far earlier than box-score rates.
- **Plate discipline** (Statcast, every pitch, stored as daily aggregates):
  the batter's whiff-per-swing and chase rates (career + decayed; chase
  feeds the walk prop, whiff the contact props) and the opposing starter's
  decayed swinging-strike rate.
- **Pitch-level shape** (raw-pitch archive): the starter's pitch movement
  profile and usage drift, re-aggregated from stored per-pitch data.
- **Raw speed** (Statcast sprint leaderboard, prior-season): sprint speed
  and home-to-first time for the SB and run-scored props.
- **Team defense** (Statcast outs above average, prior-season, per-162):
  the opponent's range quality, for the BABIP-driven hit props and the
  runs model — the unearned-run proxy only ever saw errors.
- **Battery defense** (Savant catcher leaderboards, team grain,
  prior-season): opposing-catcher framing runs per called pitch, caught-
  stealing above average, pop time (era-centered — pop drifted ~0.05s
  over the sample), and interactions with the batter's take profile and
  the running game. Player-grain catcher data is scraped but shelved.
- **Roster availability** (transactions scrape): IL stints and returns,
  as-of date.
- **Prop-specific history:** stolen-base rate + success rate (career,
  season, rolling) and the starter's SB-allowed / stop rate for the SB
  prop; the batter's own runs-scored and RBI rates for the run/RBI props;
  IBB rate (feared-hitter marker).
- **Teammate context (run/RBI props):** as-of career on-base of the two
  hitters ahead and slugging of the two behind (wrapping the order),
  plus composite traffic/conversion interactions (validated by
  residualized-probe partial correlations before shipping).
- **Power quality** (from the homerun log): average exit velo, launch
  angle, average and max **elevation-adjusted** distance of all prior
  career homers; the opposing starter's HR quality *allowed*; an
  HR-by-pitch-type score (which pitches the batter homers off, weighted by
  this starter's actual usage); prior-season GO/AO fly-ball tendency for
  both batter and starter.
- **Opposing starter:** career/season/last-5 HR / K / BB / hits-allowed
  rates, ERA, strike rate, innings per start, rest, expected bullpen
  handoff (starter-length / bullpen-exposure context).
- **Arsenal matchup** (Statcast, two-year decay blend): the batter's xSLG /
  xwOBA / whiff / hard-hit vs each pitch type, weighted by how often this
  specific starter throws each pitch; plus the starter's own arsenal
  quality.
- **Minor-league priors** (MiLB batting/pitching scrapes): translated
  performance priors that carry information for players with thin MLB
  histories — rider columns in both frames, kept by selection in 22/24
  original heads.
- **Environment:** park dimensions + elevation, empirical park HR factor
  (as-of), temperature, wind speed/direction (including wind × pull /
  carry interactions), condition, day/night.
- **League environment (strikeout model only):** trailing-30-day
  league-wide rates as-of the game date. Tried in the batter and runs
  models as a drift guard; flat-to-negative on the holdout, so only the
  K model (where it helped) kept them.
- **Context:** opposing bullpen HR/K/hit rates, high-leverage bullpen
  quality (save/hold/finishing arms), bullpen fatigue (trailing-3-day
  pitch count), own team offense (overall + home/road split), platoon
  (handedness matchup), schedule/travel context, age/height/weight,
  season, month.
- **Strikeout model only:** the starter's usage-weighted arsenal whiff /
  K% / put-away (two-year blend), his strike rate, his as-of decayed
  swinging-strike / CSW / chase-induced / zone rates and fastball-velocity
  trend (pitch-level dailies — these move DURING the season), and the
  ACTUAL opposing lineup's aggregate K%, BB%, whiff-vs-his-arsenal,
  K%-vs-his-hand, and decayed whiff/chase (not just team-season rates).
- **Runs/totals model only:** venue-split offense, opponent high-leverage
  bullpen + fatigue, opponent defense proxy (unearned-run rate), starter
  hits-allowed rate.
- **Winner model only:** each team's as-of win%, run differential per
  game, runs allowed per game, pythagorean expectation, last-20-games form,
  Elo rating (carries over the winter, so April isn't blind), and
  home-minus-away differentials for all of the above plus starter/bullpen
  quality and rest.

Which columns each head actually trains on is decided by automated
stability selection alone (`feature_select.py` → `feature_keep.json`).

## Honest evaluation — data roles are GRADED TIERS, not clean labels

There is no untouched "holdout year". Every season that has ever
influenced a decision is compromised to some degree; the honest statement
is *how much*:

| Tier | Data | Role and contamination level |
|---|---|---|
| Training | 2015–2024 | Fit directly (boosters early-stop on a held-out ~10% GamePk slice of the training rows, never the calibration year). |
| Calibration + selection | 2025 | Calibrators, blend weights, feature selection, and iteration verdicts all read 2025 — **heavily mined; treat 2025 numbers as upper bounds.** Known contamination path: the keep-lists carry shipping-suite SHAP votes measured on cal-2025. |
| Partial season, lightly touched | 2026 through 07-14 (~59% of games) | Never used for fitting, but it has been LOOKED AT — each decision-influencing look is enumerated below. |
| **Forward record** | **2026-07-15 onward** | **Timestamped predictions graded after the fact — the ONLY fully untouched test.** |

The enumerated 2026 looks (audit #6 doctrine, amended 07-15 to
auto-on-ship — every look is deliberate-by-policy and on this ledger;
the daily job adds at most ONE look per ship, never one per day):

1. Daily automated `--confirm --set-baseline` refreshes ran until
   2026-07-14 (closed by audit #6; 2026 snapshots now refresh only on
   deliberate confirms).
2. 07-13: SIM_BLEND weights were user-moderated after seeing a 2026 read
   (ledgered as a violation-in-substance; resolved 07-15 by refitting
   all three weights on 2025 only).
3. 07-13: the count-vs-binary pricing shoot-out verdict read both years.
4. 07-13/14: PA-sim Phase-3 grading (steal-layer w=0, hazard keep) read
   both years.
5. 07-14: one confirm of the tier-1-mechanics + MiLB chain.
6. 07-15: the finish-chain **veto-only** confirm (2026 may veto a shipped
   change, never tune one) — no veto; plus pa_blend's passive 2026
   column (no weight was set from it).
7. 07-15 (standing policy, user-adopted): **auto ship-confirm** — the
   first daily train after each ship re-stamps the confirm snapshot
   (`update_all.confirm_is_stale` vs `confirm_code_fp.json`), one look
   per ship. Bootstrap stamp taken 07-15 PM against the post-recovery
   retrain.

`--paired` verdicts gate on a Benjamini–Hochberg false-discovery
correction (`--fdr`, default 0.10) because one read makes 100+
simultaneous CI tests; repeated reads against the same season remain
sequential testing that no correction repairs — hence the forward record.
Each head also prints an **advisory paired `Score` row** (07-15, step 1 of
the Score-north-star plan): the same day-block draws CI the delta of the
`5_prop_rankings` v4 rank-weighted composite (the betting-priority Score the
board tiers on). Advisory means raw-CI verdicts tagged `(adv)`, excluded
from the FDR family and the net vote — the starred multi-metric bar stays
the arbiter while the Score read rides along (its qualities clip at fixed
anchors, so saturated ELITE heads can hide real moves; its top-weighted
Lift term is its noisiest). `--score-boot 0` disables.

**Known limitation (Decline ledger #4):** the empirical-Bayes shrinkage
priors in `features.py` (`SHRINK`) are pooled-era constants computed over
all years, a second-order leak. The principled fix (sequential as-of
priors) is queued as a post-ship experiment; expected effect is small but
nonzero.

Headline numbers from the 2026 confirm of the shipped model (07-15,
partial season — see the tier table above for what this number is worth):

| Output | 2026 confirm | Naive base |
|---|---|---|
| HR | AUC 0.635 [0.624, 0.647]; top-10/day 25.5% | 11.6% base rate |
| 1+ hit | AUC 0.573; top-10/day 72.0% | 60.7% |
| Stolen base | AUC 0.735; top-10/day 21.0% (3.2x) | 6.5% |
| 2+ strikeouts | AUC 0.650; top-10/day 42.0% | 21.8% |
| Strikeouts (starter) | MAE 1.74, dispersion honest (0.98) | — |
| Total runs | MAE 3.55, dispersion honest | — |

All 33 graded binary heads clear BOTH bankability gates (AUC CI > 0.5
AND logloss-edge CI > 0) on 2025 and 2026 simultaneously. Binary
probabilities are calibrated per head by `AUTO_CAL` (Platt / beta /
isotonic, chosen on the calibration year; count heads use cal-year
per-line logistic calibrators): when the model says 20%, it happens ~20%
of the time (ECE ≤ ~0.01 on every head). Current numbers always:
`metrics.json` and `python Model/evaluate_deep.py`.

## Reality check for betting

The model roughly doubles the base rate at the top of its daily HR board —
genuinely informative. That is **not** the same as beating a sportsbook:
books price HR props with an 8–15% margin and set lines with more
information (confirmed lineups, real-time weather, sharp money). Use the
fair-odds column to find prices that look off, track results against
closing lines before staking anything, and treat small edges as noise.

**Backtests are structurally rosier than live serving (audit #4).** Two
named channels beyond the usual caveats: (1) training weather is the
ACTUAL observed game weather, but live slates are served pre-game
FORECASTS (rotowire/fantasypros/open-meteo) — the model learned the value
of true temp/wind and gets noisier proxies live, so weather-sensitive
heads (HR especially) will under-deliver their backtest numbers;
`1_get_todays_games.py` archives every served forecast to
`Data/mlb_weather_forecast.csv` so this gap becomes measurable against
the actuals. (2) Backtests always know the HP umpire; live slates often
don't until near game time (served as neutral league priors).

## Workflow

```
# one-time: backfill the Statcast histories (batted balls ~10 min, pitch
# dailies ~30 min; the daily job keeps both current incrementally)
python Scrapers/scrape_statcast.py --backfill
python Scrapers/scrape_pitches.py --backfill

# refresh all data AND retrain, one command (scrapers default to Data/).
# Every scraped file is schema-validated (Scrapers/validate_data.py) against
# the previous copy in Data/backups/; a failing file is restored from backup
# and blocks the retrain, so a broken scrape can't poison the models.
python Scrapers/update_all.py --retrain

# or just the data
python Scrapers/update_all.py

# check the data files by hand (same validation the pipeline runs)
python Scrapers/validate_data.py

# retrain manually — selection suite + shipping models. A run with
# feature_keep.json present is a KEEP-train and writes the serving
# artifacts (models*.joblib); without it it is a SUPERSET train and
# writes models_superset*.joblib — the serving artifacts are never
# touched by experiments (audit #8).
python Model/train.py --rebuild

# game day: scrape today's matchups/lineups/weather into an input file
# (mlb.com starting lineups + fantasypros wind + rotowire temp/condition);
# the GUI auto-loads it into the slate on startup
python Tools/1_get_todays_games.py

# capture real closing lines near game time (needs a free ODDS_API_KEY);
# accrues the history that evaluate_deep Section 9 grades the model against
python Tools/2_scrape_odds.py

# predict
python Tools/3_gui.py                   # the GUI (single game or slate)
python Model/predict.py --game 822716   # replay a historical game
python Model/predict.py --game 822716 --recal   # + in-season drift correction
python Model/predict.py --selftest      # verify train/serve parity

# iterate on a model change WITHOUT touching 2026 (the selection loop)
python Model/evaluate_deep.py --set-baseline    # snapshot before changing
# ...edit features/params...
python Model/train.py --rebuild --select        # selection suite only (fast)
python Model/evaluate_deep.py                   # Section 11 diff on 2025
# (deltas inside the noise band are retrain jitter, not signal)

# when a change is accepted: retrain everything and confirm ONCE on 2026
python Model/train.py                           # both suites (frames cached)
python Model/evaluate_deep.py --confirm         # veto-only; goes on the ledger
```

## Automated daily refresh (Windows Task Scheduler)

A scheduled task named **"MLB Daily Update"** runs
`Scrapers/run_daily_update.cmd` every morning at **6:00 AM**, which calls
`update_all.py --retrain` (rescrape all data, retrain both suites, refresh
the 2025 selection baseline, warm the prop-rankings cache) and logs each
run to `Logs/update_<date>.log`. `-StartWhenAvailable` means a missed run
(laptop asleep at 6 AM) fires as soon as the machine is next on. Per
audit #6 as amended 07-15 (auto-on-ship), the daily job touches 2026
**only on the first train after a ship**: when the shipped Model sources
differ from the ones recorded at the last confirm stamp
(`confirm_code_fp.json`), it runs `--confirm --set-baseline` once, then
stands down until the next ship. A deliberate manual
`evaluate_deep.py --confirm --set-baseline` also refreshes the stamp and
resets the trigger.

Failure guards, in the order they act:

- **Experiment-in-flight guard** — if the Model sources differ from the
  last `--set-baseline` snapshot (`baseline_code_fp.json`), the daily run
  goes scrape-only rather than silently retraining and re-baselining an
  unfinished candidate.
- **Completed-season caching** — per-year scrapers (season stats, homeruns,
  arsenals, sprint speed, OAA, catchers) serve finished seasons from their
  own output CSV and only fetch the current season, so an upstream hiccup
  on historical data (e.g. Savant throttling a 2021 page) can no longer
  kill a job. A failed current-season fetch falls back to the previous
  run's rows with a WARNING (harmless for prior-season-consumed
  leaderboards). `--backfill` on any of them forces a full refetch.
- **Schema validation + backup restore** (`validate_data.py`) — a file that
  fails shape checks is restored from `Data/backups/` and the retrain is
  blocked.
- **Freshness tripwire** — May–September, the game logs / HR log / Statcast
  files must contain games newer than 6 days, otherwise validation FAILS.
  This catches the "scraper succeeds but ingests nothing new" failure mode
  (which every shape check passes) — the exact way a frozen season list
  would fail silently.
- **Status surfacing** — `update_all.py` writes
  `Logs/last_run_status.json`; the GUI reads it at startup and pops a
  warning when the morning job failed or the data is stale mid-season.

Manage it:

```powershell
Get-ScheduledTask -TaskName "MLB Daily Update"          # status
Start-ScheduledTask -TaskName "MLB Daily Update"        # run now
Get-ScheduledTaskInfo -TaskName "MLB Daily Update"      # last/next run time
Disable-ScheduledTask -TaskName "MLB Daily Update"      # pause
Unregister-ScheduledTask -TaskName "MLB Daily Update"   # remove
```

To change the time, edit the trigger or re-register with a new
`-At` value. Lineups for *today's* games (`get_todays_games.py`) and closing
betting lines (`scrape_odds.py`) are deliberately **not** part of this job —
run them near game time, since lineups and props only post a few hours before
first pitch, and you want closing (not opening) prices in the odds store.
