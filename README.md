# MLB Prediction Engine

End-to-end machine learning system that predicts MLB player props and game
outcomes: calibrated LightGBM models trained on 300K+ player-games across
seven seasons (2020–2026), with leakage-free feature engineering, a strict
temporal holdout, and automated daily retraining.

## What it predicts

Per game (or a full slate of games, ranked cross-game):

- **Batter props** — calibrated probability of a home run (with fair
  American odds), 1+ hit, 2+ hits, 2+ total bases, run scored, RBI, walk,
  and stolen base for every lineup batter, plus expected HR count.
- **Starter strikeouts** — expected Ks and P(over) for lines 3.5–8.5.
- **Game outcomes** — expected runs per team, total-runs P(over) from a
  negative binomial (real totals run ~2× Poisson variance), and a home-win
  probability from a dedicated winner model (Elo with cross-season
  carryover, win%, run differential, pythagorean expectation, recent form,
  both starters, bullpens, rest).

Predictions come from a Tkinter GUI (dropdown teams/lineups/weather, with
auto-fill from each team's latest real lineup) and are saved as formatted
Excel workbooks.

## Results (2026 season, true holdout)

Models train on 2020–2024, calibrate on 2025, and are evaluated on 2026 —
a season they never saw. All eight prop edges are confirmed by day-block
bootstrap CIs (both the AUC and the log-loss edge exclude chance at 95%).

| Model | Metric | Model | Baseline |
|---|---|---|---|
| Home run | daily top-10 hit rate | **24.1%** | 11.5% (base rate) |
| Home run | AUC / log loss | **0.633 / 0.3469** | 0.500 / 0.3578 |
| Stolen base | AUC (top-10 lift) | **0.662 (2.19×)** | 0.500 |
| Walk | AUC (top-10 lift) | **0.616 (1.68×)** | 0.500 |
| Strikeouts | MAE | **1.76** | 1.93 (pitcher season rate) |
| Game total | MAE | **3.53** | 3.60 (league mean) |
| Winner | accuracy / log loss | **53.8% / 0.6880** | 52.5% / 0.6919 (always home) |

Probabilities are isotonic-calibrated and honest: expected calibration
error is under 0.01 on every prop (a "20%" pick hits ~20% of the time),
and strikeout P(over) tracks actual over-rates at every line. Full
metrics, calibration tables, and a 2025 stability backtest live in
`Model/artifacts/metrics.json`.

## Why the numbers can be trusted

- **Leakage-free by construction** — every feature for a game on date D
  uses only data from strictly before D (as-of cumulative/rolling stats,
  prior-season Statcast arsenals, as-of park factors and Elo).
- **Train/serve parity self-test** — `predict.py --selftest` recomputes a
  real game's features through the live inference path and requires exact
  agreement with the training frame.
- **Temporal holdout protocol** — train ≤2024, calibrate 2025, test 2026;
  distribution constants (isotonic maps, NB dispersion, blend weights) are
  all fit on the calibration year, never the holdout.
- **2026 is confirm-only** — model selection runs on a separate suite
  (train ≤2023, calibrate 2024, test 2025) that `train.py` and
  `evaluate_deep.py` use **by default**. Feature/param changes are accepted
  or reverted on 2025; 2026 requires an explicit `evaluate_deep.py --confirm`
  and is looked at once per finished change, so its numbers stay an honest
  out-of-sample estimate.
- **Regression guardrails** — `evaluate_deep.py --set-baseline` snapshots the
  accepted model; every change is diffed against it per metric, with a
  retrain-jitter noise band so only movement beyond noise counts as
  better/worse — alongside bootstrap CIs, monthly drift, segment-level
  reliability, slate concentration, and betting-threshold economics.

## Repo layout

| Path | Role |
|---|---|
| `Scripts/` | 10+ scrapers (MLB Stats API, Baseball Savant, rosters, weather) + `scrape_odds.py` (real closing lines) + `update_all.py` one-command refresh |
| `Data/` | Scraped CSVs (~55 MB, gitignored — regenerate with `update_all.py`); schema documented in `Data/GLOSSARY.md` |
| `Model/features.py` | Feature engineering: ~110 batter features, starter/game/winner frames, Elo, shared by training and inference |
| `Model/train.py` | Builds frames, trains all models (8 props, K, runs, winner) for both suites: model-selection (≤2023/2024/2025) and shipping (≤2024/2025/2026); `--select` stops after the selection suite for fast iteration |
| `Model/odds.py`, `recalibrate.py` | Odds/de-vig/EV math + odds-store schema; and the leakage-free in-season drift correction |
| `Model/predict.py` | Prediction engine + Excel reports; `--game` replays history, `--selftest` checks parity, `--recal` applies in-season drift offsets |
| `Model/gui.py` | Tkinter app for single games or full slates |
| `Model/evaluate_deep.py` | Full holdout workup: bootstrap CIs, drift, segments, betting thresholds, K/totals/winner, model-vs-market ROI (Section 9), in-season recalibration backtest (Section 10), and `--set-baseline` regression diffing with noise bands (Section 11). Default run scores the selection suite on 2025 (iterate freely); `--confirm` scores shipping on 2026 |

## Quickstart

```bash
pip install pandas numpy scikit-learn scipy lightgbm joblib openpyxl requests beautifulsoup4

# 1. scrape all data (Data/ is not committed) — takes a while first time
python Scripts/update_all.py

# 2. build features and train everything (~2 min)
python Model/train.py --rebuild

# 3. verify train/serve parity and see the numbers
python Model/predict.py --selftest
python Model/evaluate_deep.py            # selection suite on 2025 (iterate here)
python Model/evaluate_deep.py --confirm  # shipping suite on 2026 (confirm-only)

# 4. predict
python Scripts/get_todays_games.py   # today's matchups/lineups/weather
python Model/gui.py                  # the GUI auto-loads today's slate
```

A Windows Task Scheduler job runs `update_all.py --retrain` every morning,
so data and models stay current with zero manual upkeep (see
`Model/README.md` for details and management commands).

## Honest limitations

The model beats naive baselines; whether that survives the sportsbook's
8–15% margin is now measured directly rather than assumed — `scrape_odds.py`
captures closing lines and `evaluate_deep` Section 9 grades the model against
de-vigged market prices with realized flat-stake ROI. Until that store fills,
the fair-odds output is a tool for finding mispriced lines, and picks should
be tracked against real closing prices before staking anything. The winner
model is presented as a **calibrated probability, not a pick**: its edge over
always-home is not statistically significant on half a season. Rookie picks
(<50 career games) are flagged in the GUI as low-confidence. Bet responsibly.
