# MLB Prediction Model

Gradient-boosted ensembles (LightGBM seed bags; XGBoost + CatBoost family
members in the shipped 3-family configuration — `train.py LGBM_ONLY_TEMP`
gates a temporary LGBM-only iteration regime) trained on the CSVs in
`Data/`, predicting per game (or per slate of games — the GUI's "Add game
to slate" runs many at once with one combined, cross-game-ranked board):

- **Per lineup batter:** calibrated probabilities for 14 props — home
  run (with fair American odds), 1+ hit, 2+ hits, 2+ total bases, run
  scored, 1+ RBI, 1+ walk, stolen base, single, double, 1+/2+ strikeouts,
  2+/3+ hits+runs+RBIs — plus expected HR / K / H+R+RBI counts and a
  career-games flag (players under ~50 MLB games are the least reliable
  segment; they're marked `*` in the GUI).
- **Per starter:** expected strikeouts and P(over) for K lines 3.5–8.5,
  negative-binomial-shaded by the calibration-year K dispersion; plus
  outs recorded, walks allowed, hits allowed, and earned runs with
  per-line P(over) from cal-year logistic line calibrators; and an
  expected ERA derived from the earned-runs and outs heads
  (27 × xER / xOuts). Single-start ERA is inherently noisy — the
  earned-run over/under lines are the sellable output.
- **Per game:** expected lineup home runs; expected total runs with
  P(over) from a negative binomial whose dispersion is measured on the
  calibration year (real totals run ~2x Poisson variance); and a home-win
  probability from a dedicated winner model (team win%, run differential,
  pythagorean expectation, Elo with cross-season carryover, recent form,
  both starters, bullpens, rest — blended with the runs-model Poisson win
  probability, weights chosen on the calibration year). The winner output is
  presented as a **calibrated probability, not a pick**: on the holdout it
  shows no significant edge over always taking the home team (Section 8).

## Files

| File | Role |
|---|---|
| `features.py` | Feature engineering. Everything is **as-of date**: a game on date D only uses data from before D (no leakage). Same definitions serve training (vectorized) and prediction (per-entity); `predict.py --selftest` proves the two paths agree. |
| `train.py` | Builds frames, trains everything. TWO ROLES (2026-07-15): with `feature_keep.json` present it is the KEEP-train and writes the serving artifacts (`models_bt.joblib` selection suite, `models.joblib` shipping suite); without it it is the SUPERSET train and writes `models_superset*.joblib` (feature_select's electorate — cheaper regime: fewer bags + a row sample, documented mismatch). Train/cal/holdout years are DERIVED from the data (`suite_years`) — currently ≤2023/2024/2025 and ≤2024/2025/2026; boosters early-stop on a held-out ~10% GamePk slice of the TRAINING rows (`_es_split`), never the calibration year. Heads are seed-bagged LightGBM (`LGBM_BAGS`) + XGB/CB family members when `LGBM_ONLY_TEMP` is off; per-head overrides in `PROP_PARAMS`. Artifacts carry a `meta_stamp` (role/regime/created) that predict.py's serving guard enforces. `--rebuild` refreshes cached frames; `--select` stops after the selection suite. |
| `predict.py` | `Predictor.predict_game(spec)`; CLI: `--game <GamePk>` replays a historical game, `--selftest` checks train/serve parity. Every run saves a workbook to `Predictions/`. |
| `odds.py` | Sportsbook-odds utilities: American↔probability, de-vig, EV/ROI, and the canonical odds-store schema. Bridges the model's fair probabilities to the prices books actually post; shared by `Tools/2_scrape_odds.py` and evaluate_deep Section 9. |
| `recalibrate.py` | In-season drift correction: a per-prop log-odds shift fit only on recent in-season games (leakage-free). Stored by `train.py`, backtested in evaluate_deep Section 10, applied at serving with `predict.py --recal`. |
| `evaluate_deep.py` | Full accuracy + betting workup on a held-out season: bootstrap CIs on AUC and the logloss edge, calibration, daily top-N hit rates, pick-threshold economics, monthly drift, segment reliability, K/totals/winner deep dives, **model-vs-market ROI (Section 9)**, **in-season recalibration backtest (Section 10)**, and `--set-baseline` regression diffing with per-metric noise bands (Section 11). The DEFAULT run scores the selection suite on 2025 (iterate freely); `--confirm` scores the shipping suite on the confirm-only 2026 holdout. |
| `artifacts/` | Trained models (`models.joblib`), metrics (`metrics.json`), cached feature frames, and the odds store / eval baselines. |

The game-day tools that USE these models live in `Tools/`, numbered in
run order: `1_get_todays_games.py`, `2_scrape_odds.py`, `3_gui.py` (the
Tkinter app), `4_grade_results.py`, plus `hit_rate_report.py` and
`prop_rankings.py`.

## What the model knows (~575-column batter-game superset; stability selection picks each head's columns)

- **Batter form:** career / season-to-date / rolling 7-15-30-game HR rate,
  ISO, contact, K/BB rates, XBH rate, full OBP (incl. HBP); days rest;
  lineup slot; home/away; career home/road splits (venue-context rates);
  position wear (career share of games caught / DH'd); **90-day
  decay-weighted rates** (every past game weighted by recency — skill drift
  without a rolling window's cliff) and **trend deltas** (rolling-15 vs
  season, season vs career; last-5 vs season for the opposing starter).
- **Contact quality** (Statcast, every tracked ball in play — not just
  homers): shrunk career + 90-day-decayed exit velo, hard-hit%, barrel
  rate, launch angle, xBA / xwOBA on contact, ground-ball share, and
  spray-based pull% / pulled-air% (which interact with the park's fence
  distances) — for the batter AND allowed by the opposing starter. Process
  stats stabilize in ~40 batted balls vs hundreds of AB for outcomes, so
  these see skill (and in-season change) far earlier than box-score rates.
- **Plate discipline** (Statcast, every pitch, stored as daily aggregates):
  the batter's whiff-per-swing and chase rates (career + decayed; chase
  feeds the walk prop, whiff the contact props) and the opposing starter's
  decayed swinging-strike rate.
- **Raw speed** (Statcast sprint leaderboard, prior-season): sprint speed
  and home-to-first time for the SB and run-scored props.
- **Team defense** (Statcast outs above average, prior-season, per-162):
  the opponent's range quality, for the BABIP-driven hit props and the
  runs model — the unearned-run proxy only ever saw errors.
- **Prop-specific history:** stolen-base rate + success rate (career,
  season, rolling) and the starter's SB-allowed / stop rate for the SB
  prop; the batter's own runs-scored and RBI rates for the run/RBI props;
  IBB rate (feared-hitter marker).
- **Teammate context (run/RBI props):** as-of career on-base of the two
  hitters ahead and slugging of the two behind (wrapping the order).
- **Power quality** (from the homerun log): average exit velo, launch
  angle, average and max **elevation-adjusted** distance of all prior
  career homers; the opposing starter's HR quality *allowed*; an
  HR-by-pitch-type score (which pitches the batter homers off, weighted by
  this starter's actual usage); prior-season GO/AO fly-ball tendency for
  both batter and starter.
- **Opposing starter:** career/season/last-5 HR / K / BB / hits-allowed
  rates, ERA, strike rate, innings per start, rest.
- **Arsenal matchup** (Statcast, two-year decay blend): the batter's xSLG /
  xwOBA / whiff / hard-hit vs each pitch type, weighted by how often this
  specific starter throws each pitch; plus the starter's own arsenal
  quality.
- **Environment:** park dimensions + elevation, empirical park HR factor
  (as-of), temperature, wind speed/direction, condition, day/night.
- **League environment (strikeout model only):** trailing-30-day
  league-wide rates as-of the game date. Tried in the batter and runs
  models as a drift guard; flat-to-negative on the holdout, so only the
  K model (where it helped) kept them.
- **Context:** opposing bullpen HR/K/hit rates, high-leverage bullpen
  quality (save/hold/finishing arms), bullpen fatigue (trailing-3-day
  pitch count), own team offense (overall + home/road split), platoon
  (handedness matchup), age/height/weight, season, month.
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
stability selection alone (`feature_select.py` → `feature_keep.json`); the
old hand-curated per-prop routing tables were retired 2026-07-10 and
deleted 2026-07-15 (git history keeps them).

## Honest evaluation (2026 season = confirm-only holdout)

Trained on 2015–2024 (boosters early-stop on a held-out ~10% GamePk slice
of the training rows, never the calibration year — audit fix #2),
calibrated on 2025, tested on 2026. **2026 is confirm-only**: iterating
features/params against its numbers quietly overfits the holdout, so
day-to-day model selection runs on a separate suite (train ≤2023, cal
2024, test 2025) that both `train.py` and `evaluate_deep.py` use **by
default**; 2026 gets one confirming look per finished change via
`evaluate_deep.py --confirm --set-baseline` — run DELIBERATELY, never by
the daily job (audit #6).

Data roles are graded tiers, not clean labels: 2015–2024 train / 2025
calibration + selection + iteration verdicts (heavily mined — treat 2025
numbers as upper bounds) / 2026 lightly touched / the timestamped forward
record from 2026-07-15 is the ONLY fully untouched test. `--paired`
verdicts gate on a Benjamini–Hochberg false-discovery correction
(`--fdr`, default 0.10) because one read makes 100+ simultaneous CI
tests (audit #1); repeated reads against the same season remain
sequential testing that no correction repairs — hence the forward record.

| Model | Metric | Model | Baseline |
|---|---|---|---|
| HR | log loss | **0.3478** | 0.3574 (base rate) |
| HR | AUC | **0.625** | 0.500 |
| HR | top-10/day hit rate | **22.2%** | 11.5% (base rate) |
| Hit | AUC | **0.567** | 0.500 |
| Strikeouts | MAE | **1.78** | 1.93 (pitcher season rate) |
| Total runs | MAE | **3.55** | 3.63 (team scoring rates) |

The 2025 backtest (train ≤2023) gives nearly identical numbers, so the edge
is stable across seasons. Binary probabilities are Platt-calibrated
(`train.py PLATT_CAL` — every binary head + the winner; count heads use
cal-year per-line logistic calibrators): when the model says 20%, it
happens ~20% of the time.

The table above is a HISTORIC snapshot (it predates the 2026-07 upgrades:
3-family ensemble, 24 binary heads, PA-sim blend, 2015+ data) — after
retraining, `metrics.json` and `python Model/evaluate_deep.py` have the
current numbers — confidence intervals, drift, segments, betting-threshold
views, and a `--set-baseline` diff of what a change moved.

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
`1_get_todays_games.py` now archives every served forecast to
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

# retrain manually — selection suite + shipping models (a FULL two-suite
# retrain on the current ~575-column superset runs ~1.5h; LGBM-only
# iteration regimes are faster). A run with feature_keep.json present is a
# KEEP-train and writes the serving artifacts (models*.joblib); without it
# it is a SUPERSET train and writes models_superset*.joblib — the serving
# artifacts are never touched by experiments (audit #8).
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
python Model/evaluate_deep.py --confirm
```

## Automated daily refresh (Windows Task Scheduler)

A scheduled task named **"MLB Daily Update"** runs
`Scrapers/run_daily_update.cmd` every morning at **6:00 AM**, which calls
`update_all.py --retrain` (rescrape all data, then retrain the models) and
logs each run to `Logs/update_<date>.log`. `-StartWhenAvailable` means a
missed run (laptop asleep at 6 AM) fires as soon as the machine is next on.

Failure guards, in the order they act:

- **Completed-season caching** — per-year scrapers (season stats, homeruns,
  arsenals, sprint speed, OAA) serve finished seasons from their own output
  CSV and only fetch the current season, so an upstream hiccup on
  historical data (e.g. Savant throttling a 2021 page) can no longer kill a
  job. A failed current-season fetch falls back to the previous run's rows
  with a WARNING (harmless for prior-season-consumed leaderboards).
  `--backfill` on any of them forces a full refetch.
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
