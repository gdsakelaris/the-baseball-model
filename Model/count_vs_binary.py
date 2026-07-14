"""Count-head vs binary-head pricing on the SAME batter threshold (v1).

Five batter thresholds are priced twice today by two independently-trained
estimators, and only one of them ships:

    threshold        SHIPPED (incumbent)      BANKED, unused
    1+ K             binary head  bk          xbk  line 0.5
    2+ K             binary head  bk2         xbk  line 1.5
    2+ H+R+RBI       binary head  hrr2        xhrr line 1.5
    3+ H+R+RBI       binary head  hrr3        xhrr line 2.5
    2+ TB            binary head  tb2         xtb  line 1.5

train.fit_line_cals fits a per-line logistic for EVERY count head (batter and
starter alike) and banks it in the artifacts; predict.BAT_COUNT_COLS then
ships only the batter means, so the batter line_cals are trained and never
read. This grades them head-to-head on the held-out test year, plus a logit
blend of the two at fixed weights (w = weight on the COUNT side; w=0 IS the
incumbent, so the w=0 row is a self-check and must read delta 0.000).

Blend weights are NOT fit here on purpose: both calibrators (the binary
head's Platt, the count head's line logistic) are fit on the cal year, so a
cal-year weight fit flatters both sides. Fixed-w + held-out day-block CI is
the same read that shipped SIM_BLEND.

Also prices the four count lines that have NO binary counterpart and are
shipped nowhere today (xbk 2.5 = 3+ K, xhrr 3.5 = 4+ H+R+RBI, xtb 2.5/3.5 =
3+/4+ TB) against their base rate, to say whether they are shippable at all.

    + delta = challenger beats the shipped binary head (log loss)

Read-only: loads artifacts, writes nothing. Does not touch the Model sources
in baseline_code_fp.json, so it cannot send the daily run scrape-only.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("Model").resolve()))
from predict import predict_prop            # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402

ART = Path("Model/artifacts")
BOOT = 400
WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]      # w = weight on the COUNT side
rng = np.random.default_rng(0)

# (binary head, its target col per train.PROPS, count head, count line) — the
# same event described two ways. Target cols are pinned from train.PROPS, not
# guessed from the head name: bk's target is y_bk1, not y_bk.
PAIRS = [
    ("bk",    "y_bk1",  "xbk",  0.5),
    ("bk2",   "y_bk2",  "xbk",  1.5),
    ("hrr2",  "y_hrr2", "xhrr", 1.5),
    ("hrr3",  "y_hrr3", "xhrr", 2.5),
    ("tb2",   "y_tb2",  "xtb",  1.5),
]


def prep(df, cols, cat_levels):             # evaluate_deep.prep, inlined
    X = df[cols].copy()
    for c, levels in cat_levels.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
    return X


def rowwise_ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ece(y, p, bins=10):
    """Expected calibration error, 10 equal-width bins."""
    idx = np.clip((np.asarray(p) * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(y) * abs(y[m].mean() - p[m].mean())
    return float(tot)


def block_ci(dates, ll_inc, ll_chl):
    """Day-block bootstrap CI on mean(ll_incumbent) - mean(ll_challenger).
    + = the challenger prices this threshold better than the shipped head."""
    d = pd.factorize(dates)[0]
    n_days = d.max() + 1
    idx_by_day = [np.flatnonzero(d == i) for i in range(n_days)]
    deltas = []
    for _ in range(BOOT):
        take = rng.integers(0, n_days, n_days)
        rows = np.concatenate([idx_by_day[t] for t in take])
        deltas.append(ll_inc[rows].mean() - ll_chl[rows].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def verdict(lo, hi):
    return "WIN" if lo > 0 else "HARM" if hi < 0 else "tie"


frames = joblib.load(ART / "frames.joblib")
bf = frames["bf"]

for fname in ("models_bt.joblib", "models.joblib"):
    path = ART / fname
    if not path.exists():
        print(f"\n!! {fname} missing — skipped")
        continue
    art = joblib.load(path)
    yr = int(art["years"]["test"])
    cl = art["cat_levels"]
    counts = art.get("count_models", {})
    props = art.get("props", {})

    bf_y = bf[(bf["Season"] == yr) & ~bf["ShortGame"].fillna(False)]
    print(f"\n{'=' * 78}\n=== {fname} — test {yr} — {len(bf_y):,} batter-games"
          f"\n=== + delta = challenger beats the SHIPPED binary head (logloss)"
          f"\n{'=' * 78}")

    rows, unshipped = [], []
    for bname, btgt, cname, line in PAIRS:
        head, prop = counts.get(cname), props.get(bname)
        if head is None or prop is None:
            print(f"!! {bname}/{cname} missing from artifact — skipped")
            continue
        lc = head.get("line_cals", {}).get(line)
        if lc is None:
            print(f"!! {cname} has no banked calibrator at {line} — skipped")
            continue

        d = bf_y.dropna(subset=[head["target"], btgt])
        if not len(d):
            print(f"!! {bname}: no rows with both targets — skipped")
            continue

        y_bin = d[btgt].to_numpy().astype(float)
        y_cnt = (d[head["target"]].to_numpy() > line).astype(float)
        # HARD GATE: the two heads must be describing the SAME event, or the
        # whole comparison is meaningless. Never soften this to a warning.
        mismatch = float((y_bin != y_cnt).mean())
        if mismatch > 0:
            print(f"!! {bname} vs {cname}>{line}: targets disagree on "
                  f"{mismatch:.2%} of rows — NOT the same event, skipped")
            continue

        y = y_bin
        dates = d["Date"].to_numpy()
        mu = head["model"].predict(prep(d, head["cols"], cl))
        p_cnt = lc.predict_proba(np.asarray(mu).reshape(-1, 1))[:, 1]
        p_bin = predict_prop(prop, prep(d, prop["cols"], cl))

        ll_bin = rowwise_ll(y, p_bin)
        for w in WEIGHTS:
            p_w = sigmoid(w * logit(p_cnt) + (1 - w) * logit(p_bin))
            ll_w = rowwise_ll(y, p_w)
            dlt, lo, hi = block_ci(dates, ll_bin, ll_w)
            rows.append({
                "threshold": f"{bname} ({cname}>{line})", "w_count": w,
                "base%": round(float(y.mean()), 3),
                "logloss": round(float(ll_w.mean()), 5),
                "auc": round(float(roc_auc_score(y, p_w)), 4),
                "ece": round(ece(y, p_w), 4),
                "delta": round(dlt, 5), "ci_lo": round(lo, 5),
                "ci_hi": round(hi, 5),
                "verdict": "—(incumbent)" if w == 0 else verdict(lo, hi)})

    # count lines with no binary head: shippable at all, or noise?
    for cname, head in counts.items():
        if head["frame"] != "bat":
            continue
        priced = {ln for _bn, _bt, cn, ln in PAIRS if cn == cname}
        for line in head["lines"]:
            if line in priced:
                continue
            lc = head.get("line_cals", {}).get(line)
            if lc is None:
                continue
            d = bf_y.dropna(subset=[head["target"]])
            y = (d[head["target"]].to_numpy() > line).astype(float)
            if not 0 < y.mean() < 1:
                continue
            mu = head["model"].predict(prep(d, head["cols"], cl))
            p = lc.predict_proba(np.asarray(mu).reshape(-1, 1))[:, 1]
            base = np.full_like(p, y.mean())
            unshipped.append({
                "line": f"{cname}>{line}", "base%": round(float(y.mean()), 4),
                "logloss": round(float(log_loss(y, p)), 5),
                "ll_baserate": round(float(log_loss(y, base)), 5),
                "lift": round(float(log_loss(y, base) - log_loss(y, p)), 5),
                "auc": round(float(roc_auc_score(y, p)), 4),
                "ece": round(ece(y, p), 4)})

    if rows:
        print("\n-- SHIPPED THRESHOLDS: binary (w=0) vs count vs blend --")
        print(pd.DataFrame(rows).to_string(index=False))
    if unshipped:
        print("\n-- COUNT LINES WITH NO BINARY HEAD (shipped nowhere today) --")
        print(pd.DataFrame(unshipped).to_string(index=False))

print("\nw_count = weight on the count head in the logit blend. w=0 is the "
      "shipped board (delta must read 0.000 — that row is the self-check); "
      "w=1 is a straight flip to count-priced batter lines.")
