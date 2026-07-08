"""In-season drift correction for the batter props.

The shipped models train on completed seasons and calibrate isotonic maps on
the calibration year; the current season is a pure holdout they never learn
from. So when the current season runs hotter than history (evaluate_deep
Section 4 shows the model UNDER-predicting HR/runs/walks as summer arrives),
nothing adapts. This is a thin correction on top of the frozen model: a
per-prop shift in log-odds, fit so recent in-season predictions match recent
in-season outcomes.

It is deliberately minimal — the season provides only a few thousand games,
so a global bias fix is what the data can support without over-fitting.
Everything here is leakage-free by construction: a correction for a game on
date D is fit only on in-season games strictly before D.

The correction comes in variants, compared side by side by evaluate_deep
Section 10 before anyone turns one on:

  expanding      one offset fit on ALL in-season games so far (the original;
                 this is what train.py stores for `predict.py --recal`)
  trailing-N     one offset fit only on the last N days — an expanding
                 window dilutes a summer surge with April data, a trailing
                 window tracks the drift's shape
  temp-aware     two parameters (intercept + temperature slope), fit on the
                 season so far — lets the correction vary with game-time
                 temperature instead of shifting every park equally

Entry points:
  fit_logit_offset(p, y)  -> the single shift delta (used at train time to
                             store a serving offset that the daily retrain
                             refreshes with all in-season data through today)
  inseason_correct(...)   -> a rolling, strictly-causal backtest of one
                             variant (window_days / temp select which), for
                             Section 10 to prove it helps before enabling.
"""

import numpy as np


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def apply_offset(p, delta):
    """Shift probabilities by `delta` in log-odds space and clip back."""
    z = _logit(p) + delta
    return np.clip(_sigmoid(z), 1e-4, 1 - 1e-4)


def fit_logit_offset(p, y, lo=-6.0, hi=6.0, iters=50):
    """The log-odds shift delta that makes mean(apply_offset(p, delta)) equal
    the observed base rate mean(y). Mean predicted is monotonic in delta, so a
    bisection nails it. Returns 0.0 for empty/degenerate input."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0 or len(y) == 0 or len(np.unique(y)) < 2:
        return 0.0
    z = _logit(p)
    target = float(y.mean())

    def gap(delta):
        return float(_sigmoid(z + delta).mean()) - target

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


def _fit_offset_temp(z, y, x, iters=8, ridge=1e-4):
    """Two-parameter recalibration: minimize log loss of sigmoid(z + a + b*x)
    over (a, b) by Newton's method. x must already be standardized (the
    caller owns the mean/std so serving rows use the same scaling). Starts
    from the plain moment-matched offset with b=0, so it can only refine it.
    Returns (a, b), with b capped at +/-1 per std-unit of temperature —
    anything larger on a few thousand games is noise."""
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    a = fit_logit_offset(_sigmoid(z), y)
    b = 0.0
    for _ in range(iters):
        p = _sigmoid(z + a + b * x)
        w = np.clip(p * (1 - p), 1e-6, None)
        g0 = float((p - y).sum())              # d(nll)/da
        g1 = float(((p - y) * x).sum())        # d(nll)/db
        h00 = float(w.sum()) + ridge
        h01 = float((w * x).sum())
        h11 = float((w * x * x).sum()) + ridge
        det = h00 * h11 - h01 * h01
        if not np.isfinite(det) or det <= 0:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a -= da
        b -= db
        if abs(da) < 1e-8 and abs(db) < 1e-8:
            break
    return float(np.clip(a, -6.0, 6.0)), float(np.clip(b, -1.0, 1.0))


def inseason_correct(dates, p, y, min_n=300, window_days=None, temp=None):
    """Strictly-causal rolling correction, one variant per call. Walking
    dates in order, the correction applied to a given date is fit ONLY on
    in-season games before it; a date's own games never inform their own
    correction. Dates before `min_n` games have accumulated keep the raw
    probability.

      window_days=None  fit on all prior in-season games (expanding)
      window_days=N     fit only on games in the last N days
      temp=array        two-parameter fit (intercept + standardized game
                        temperature); NaN temps are imputed with the fit
                        window's mean

    A fit window that is degenerate (single-class y, or fewer than 50 rows
    for the windowed variants) leaves that date's rows raw but still marks
    them applied, so every variant is scored on the same rows.

    Returns (corrected, applied, order) all aligned to the CHRONOLOGICAL
    sort of the inputs (order is the argsort, if you need to map back)."""
    dates = np.asarray(dates)
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(dates, kind="stable")
    ds, ps, ys = dates[order], p[order], y[order]
    zs = _logit(ps)
    xs = None
    if temp is not None:
        xs = np.asarray(temp, dtype=float)[order]

    corrected = ps.copy()
    applied = np.zeros(len(ps), dtype=bool)
    _, day_starts = np.unique(ds, return_index=True)
    day_starts = np.append(day_starts, len(ds))

    for i in range(len(day_starts) - 1):
        s, e = day_starts[i], day_starts[i + 1]
        if s < min_n:                       # warmup: not enough history yet
            continue
        applied[s:e] = True                 # scored rows, even if left raw
        lo = 0
        if window_days is not None:
            cutoff = ds[s] - np.timedelta64(window_days, "D")
            lo = int(np.searchsorted(ds[:s], cutoff, side="left"))
        yw = ys[lo:s]
        if len(np.unique(yw)) < 2 or (window_days is not None and
                                      len(yw) < 50):
            continue                        # degenerate window: stay raw
        if xs is None:
            delta = fit_logit_offset(ps[lo:s], yw)
            corrected[s:e] = apply_offset(ps[s:e], delta)
        else:
            xw = xs[lo:s]
            mu = float(np.nanmean(xw)) if np.isfinite(xw).any() else 0.0
            sd = float(np.nanstd(xw))
            if not np.isfinite(mu):
                mu = 0.0
            if not np.isfinite(sd) or sd < 1e-6:
                sd = 1.0
            xw_z = (np.where(np.isfinite(xw), xw, mu) - mu) / sd
            a, b = _fit_offset_temp(zs[lo:s], yw, xw_z)
            xd = xs[s:e]
            xd_z = (np.where(np.isfinite(xd), xd, mu) - mu) / sd
            corrected[s:e] = np.clip(_sigmoid(zs[s:e] + a + b * xd_z),
                                     1e-4, 1 - 1e-4)
    return corrected, applied, order
