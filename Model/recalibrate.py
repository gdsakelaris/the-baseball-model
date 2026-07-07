"""In-season drift correction for the batter props.

The shipped models train on 2020-2024 and calibrate isotonic maps on 2025;
2026 is a pure holdout they never learn from. So when the current season runs
hotter than history (evaluate_deep Section 4 shows the model UNDER-predicting
HR/runs/walks as summer arrives), nothing adapts. This is a thin correction on
top of the frozen model: a single per-prop shift in log-odds, fit so recent
in-season predictions match recent in-season outcomes.

It is deliberately minimal (one parameter per prop) — the season provides only
a few thousand games, so a global bias fix is what the data can support without
over-fitting. Everything here is leakage-free by construction: a correction for
a game on date D is fit only on in-season games strictly before D.

Two entry points:
  fit_logit_offset(p, y)  -> the single shift delta (used at train time to
                             store a serving offset that the daily retrain
                             refreshes with all in-season data through today)
  inseason_correct(...)   -> a rolling, strictly-causal backtest of the above,
                             for evaluate_deep Section 10 to prove it helps
                             before anyone turns it on.
"""

import numpy as np
import pandas as pd


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_offset(p, delta):
    """Shift probabilities by `delta` in log-odds space and clip back."""
    z = _logit(p) + delta
    return np.clip(1.0 / (1.0 + np.exp(-z)), 1e-4, 1 - 1e-4)


def fit_logit_offset(p, y, lo=-6.0, hi=6.0, iters=50):
    """The log-odds shift delta that makes mean(apply_offset(p, delta)) equal
    the observed base rate mean(y). Mean predicted is monotonic in delta, so a
    bisection nails it. Returns 0.0 for empty/degenerate input."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0 or len(y) == 0:
        return 0.0
    z = _logit(p)
    target = float(y.mean())

    def gap(delta):
        return float((1.0 / (1.0 + np.exp(-(z + delta)))).mean()) - target

    ga = gap(lo)
    if ga >= 0:          # even the biggest downshift can't get low enough
        return lo
    if gap(hi) <= 0:     # even the biggest upshift can't get high enough
        return hi
    a, b = lo, hi
    for _ in range(iters):
        m = 0.5 * (a + b)
        if gap(m) < 0:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def inseason_correct(dates, p, y, min_n=300):
    """Strictly-causal rolling correction. Walking dates in order, the delta
    applied to a given date is fit ONLY on in-season games before it; a date's
    own games never inform their own correction. Dates before `min_n` games
    have accumulated keep the raw probability.

    Returns (corrected, applied, order) all aligned to the CHRONOLOGICAL sort
    of the inputs (order is the argsort, if you need to map back)."""
    dates = np.asarray(dates)
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(dates, kind="stable")
    ds, ps, ys = dates[order], p[order], y[order]

    corrected = ps.copy()
    applied = np.zeros(len(ps), dtype=bool)
    df = pd.DataFrame({"p": ps, "y": ys})            # index 0..n-1 == sorted pos
    seen_p, seen_y = [], []
    for _d, g in df.groupby(ds, sort=True):
        idx = g.index.to_numpy()
        if len(seen_y) >= min_n:
            delta = fit_logit_offset(np.asarray(seen_p), np.asarray(seen_y))
            corrected[idx] = apply_offset(ps[idx], delta)
            applied[idx] = True
        seen_p.extend(ps[idx].tolist())
        seen_y.extend(ys[idx].tolist())
    return corrected, applied, order
