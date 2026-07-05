# MLB Prediction Model

Gradient-boosted (LightGBM) models trained on the CSVs in `Data/`, predicting
per game (or per slate of games — the GUI's "Add game to slate" runs many at
once with one combined, cross-game-ranked board):

- **Per lineup batter:** calibrated probabilities for eight props — home
  run (with fair American odds), 1+ hit, 2+ hits, 2+ total bases, run
  scored, 1+ RBI, 1+ walk, stolen base — plus expected HR count and a
  career-games flag (players under ~50 MLB games are the least reliable
  segment; they're marked `*` in the GUI).
- **Per starter:** expected strikeouts and P(over) for K lines 3.5–8.5.
- **Per game:** expected lineup home runs; expected total runs with
  P(over) from a negative binomial whose dispersion is measured on the
  calibration year (real totals run ~2x Poisson variance); and a home-win
  probability from a dedicated winner model (team win%, run differential,
  pythagorean expectation, Elo with cross-season carryover, recent form,
  both starters, bullpens, rest — blended with the runs-model Poisson win
  probability, weights chosen on the calibration year).

## Files

| File | Role |
|---|---|
| `features.py` | Feature engineering. Everything is **as-of date**: a game on date D only uses data from before D (no leakage). Same definitions serve training (vectorized) and prediction (per-entity); `predict.py --selftest` proves the two paths agree. |
| `train.py` | Builds frames, trains all four models, calibrates, backtests. `--rebuild` refreshes cached frames after re-scraping data. |
| `predict.py` | `Predictor.predict_game(spec)`; CLI: `--game <GamePk>` replays a historical game, `--selftest` checks train/serve parity. Every run saves a workbook to `Predictions/`. |
| `gui.py` | Tkinter app: dropdowns for teams, date, stadium, starters, two ordered 9-man lineups, day/night, weather. Auto-fill pulls each team's most recent real lineup. Results are shown and auto-saved to `Predictions/<date>_<away>_at_<home>_<time>.xlsx`. |
| `evaluate.py` | Accuracy/effectiveness report on a held-out season: AUC, log loss and Brier vs baselines, calibration by decile, daily top-N hit rates with lift, monthly stability. `--year 2026` (default) is the honest holdout. |
| `artifacts/` | Trained models (`models.joblib`), metrics (`metrics.json`), cached feature frames. |

## What the model knows (features, ~75 per batter-game)

- **Batter form:** career / season-to-date / rolling 7-15-30-game HR rate,
  ISO, contact, K/BB rates; days rest; lineup slot; home/away.
- **Power quality** (from the homerun log): average exit velo, average and
  max distance of all prior career homers.
- **Opposing starter:** career/season/last-5 HR allowed, K/BB rates, ERA,
  innings per start, rest.
- **Arsenal matchup** (Statcast, prior season): the batter's xSLG / xwOBA /
  whiff / hard-hit vs each pitch type, weighted by how often this specific
  starter throws each pitch; plus the starter's own arsenal quality.
- **Environment:** park dimensions + elevation, empirical park HR factor
  (as-of), temperature, wind speed/direction, condition, day/night.
- **League environment (strikeout model only):** trailing-30-day
  league-wide rates as-of the game date. Tried in the batter and runs
  models as a drift guard; flat-to-negative on the holdout, so only the
  K model (where it helped) kept them.
- **Context:** opposing bullpen HR/K rates, own team offense, platoon
  (handedness matchup), age/height/weight, season, month.
- **Winner model only:** each team's as-of win%, run differential per
  game, runs allowed per game, pythagorean expectation, last-20-games form,
  Elo rating (carries over the winter, so April isn't blind), and
  home-minus-away differentials for all of the above plus starter/bullpen
  quality and rest.

## Honest evaluation (2026 season = untouched holdout)

Trained on 2020–2024, calibrated on 2025, tested on 2026:

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
retraining, `metrics.json` and `python Model/evaluate.py` have the current
numbers, and `python Model/evaluate_deep.py` adds confidence intervals,
drift, segments, and betting-threshold views.

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

# retrain manually (frames rebuild ~1 min, training ~1 min)
python Model/train.py --rebuild

# game day: scrape today's matchups/lineups/weather into an input file
# (mlb.com starting lineups + fantasypros wind + rotowire temp/condition);
# the GUI auto-loads it into the slate on startup
python Scripts/get_todays_games.py

# predict
python Model/gui.py                  # the GUI (single game or slate)
python Model/predict.py --game 822716   # replay a historical game
python Model/predict.py --selftest      # verify train/serve parity

# evaluate accuracy on the held-out season
python Model/evaluate.py --year 2026 --top 10
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
`-At` value. Lineups for *today's* games (`get_todays_games.py`) are not part
of this job — run that near game time, since lineups only post a few hours
before first pitch.
