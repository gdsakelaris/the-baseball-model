# MLB Prediction Model

Gradient-boosted (LightGBM) models trained on the CSVs in `Data/`, predicting
per game (or per slate of games — the GUI's "Add game to slate" runs many at
once with one combined, cross-game-ranked board):

- **Per lineup batter:** calibrated probabilities for eight props — home
  run (with fair American odds), 1+ hit, 2+ hits, 2+ total bases, run
  scored, 1+ RBI, 1+ walk, stolen base — plus expected HR count and a
  career-games flag (players under ~50 MLB games are the least reliable
  segment; they're marked `*` in the GUI).
- **Per starter:** expected strikeouts and P(over) for K lines 3.5–8.5,
  negative-binomial-shaded by the calibration-year K dispersion.
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
| `train.py` | Builds frames, trains everything: the model-SELECTION suite (train ≤2023, cal 2024, test 2025 → `models_bt.joblib`) and then the shipping models (train ≤2024, cal 2025 → `models.joblib`). `--rebuild` refreshes cached frames after re-scraping data; `--select` stops after the selection suite (the fast iteration loop). |
| `predict.py` | `Predictor.predict_game(spec)`; CLI: `--game <GamePk>` replays a historical game, `--selftest` checks train/serve parity. Every run saves a workbook to `Predictions/`. |
| `gui.py` | Tkinter app: dropdowns for teams, date, stadium, starters, two ordered 9-man lineups, day/night, weather. Auto-fill pulls each team's most recent real lineup. Results are shown and auto-saved to `Predictions/<date>_<away>_at_<home>_<time>.xlsx`. |
| `odds.py` | Sportsbook-odds utilities: American↔probability, de-vig, EV/ROI, and the canonical odds-store schema. Bridges the model's fair probabilities to the prices books actually post; shared by `Scripts/scrape_odds.py` and evaluate_deep Section 9. |
| `recalibrate.py` | In-season drift correction: a per-prop log-odds shift fit only on recent in-season games (leakage-free). Stored by `train.py`, backtested in evaluate_deep Section 10, applied at serving with `predict.py --recal`. |
| `evaluate_deep.py` | Full accuracy + betting workup on a held-out season: bootstrap CIs on AUC and the logloss edge, calibration, daily top-N hit rates, pick-threshold economics, monthly drift, segment reliability, K/totals/winner deep dives, **model-vs-market ROI (Section 9)**, **in-season recalibration backtest (Section 10)**, and `--set-baseline` regression diffing with per-metric noise bands (Section 11). The DEFAULT run scores the selection suite on 2025 (iterate freely); `--confirm` scores the shipping suite on the confirm-only 2026 holdout. |
| `artifacts/` | Trained models (`models.joblib`), metrics (`metrics.json`), cached feature frames, and the odds store / eval baselines. |

## What the model knows (features, ~100 per batter-game)

- **Batter form:** career / season-to-date / rolling 7-15-30-game HR rate,
  ISO, contact, K/BB rates, XBH rate, full OBP (incl. HBP); days rest;
  lineup slot; home/away; career home/road splits (venue-context rates);
  position wear (career share of games caught / DH'd).
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
  K% / put-away (two-year blend), his strike rate, and the ACTUAL opposing
  lineup's aggregate K%, BB%, whiff-vs-his-arsenal and K%-vs-his-hand
  (not just team-season rates).
- **Runs/totals model only:** venue-split offense, opponent high-leverage
  bullpen + fatigue, opponent defense proxy (unearned-run rate), starter
  hits-allowed rate.
- **Winner model only:** each team's as-of win%, run differential per
  game, runs allowed per game, pythagorean expectation, last-20-games form,
  Elo rating (carries over the winter, so April isn't blind), and
  home-minus-away differentials for all of the above plus starter/bullpen
  quality and rest.

Feature routing is per-prop (`train.py PROP_EXCLUDE`): specialized groups
only reach the props they describe (SB features only the SB model, teammate
context only run/RBI, power-quality only the power props), so thin-signal
models aren't diluted by irrelevant columns.

## Honest evaluation (2026 season = confirm-only holdout)

Trained on 2020–2024, calibrated on 2025, tested on 2026. **2026 is
confirm-only**: iterating features/params against its numbers quietly
overfits the holdout, so day-to-day model selection runs on a separate
suite (train ≤2023, cal 2024, test 2025) that both `train.py` and
`evaluate_deep.py` use **by default**; 2026 gets one confirming look per
finished change via `evaluate_deep.py --confirm`.

| Model | Metric | Model | Baseline |
|---|---|---|---|
| HR | log loss | **0.3478** | 0.3574 (base rate) |
| HR | AUC | **0.625** | 0.500 |
| HR | top-10/day hit rate | **22.2%** | 11.5% (base rate) |
| Hit | AUC | **0.567** | 0.500 |
| Strikeouts | MAE | **1.78** | 1.93 (pitcher season rate) |
| Total runs | MAE | **3.55** | 3.63 (team scoring rates) |

The 2025 backtest (train ≤2023) gives nearly identical numbers, so the edge
is stable across seasons. Probabilities are isotonic-calibrated: when the
model says 20%, it happens ~20% of the time (see `metrics.json` calibration
tables).

The table above predates the 2026-07 upgrade (dedicated winner model,
negative-binomial totals, league-environment drift features) — after
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

## Workflow

```
# refresh all data AND retrain, one command (scrapers default to Data/)
python Scripts/update_all.py --retrain

# or just the data
python Scripts/update_all.py

# retrain manually — selection suite + shipping models (~2 min total)
python Model/train.py --rebuild

# game day: scrape today's matchups/lineups/weather into an input file
# (mlb.com starting lineups + fantasypros wind + rotowire temp/condition);
# the GUI auto-loads it into the slate on startup
python Scripts/get_todays_games.py

# capture real closing lines near game time (needs a free ODDS_API_KEY);
# accrues the history that evaluate_deep Section 9 grades the model against
python Scripts/scrape_odds.py

# predict
python Model/gui.py                     # the GUI (single game or slate)
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
`Scripts/run_daily_update.cmd` every morning at **6:00 AM**, which calls
`update_all.py --retrain` (rescrape all data, then retrain the models) and
logs each run to `Logs/update_<date>.log`. `-StartWhenAvailable` means a
missed run (laptop asleep at 6 AM) fires as soon as the machine is next on.

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
