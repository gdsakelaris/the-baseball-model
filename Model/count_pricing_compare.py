"""Calibrator-vs-NB pricing comparison, FULL count surface (v3.7): for every
count family (6 count heads + starter K + game total), price every line both
ways from the SAME mu — cal-year per-line logistic vs NB/Poisson tail — and
grade held-out log loss with a day-block bootstrap CI on the delta.

Re-opens per's 07-09 NB verdict on today's model (user: "who is to say it
wouldn't be different now?") and validates the new K/total calibrators in the
same sweep. mu is identical on both sides, so this isolates pure pricing.
Run AFTER the train + inject_line_cals (needs k/total line_cals present).

  + delta = calibrator better (log loss, sign flipped so + = cal wins)
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("Model").resolve()))
import features as F                       # noqa: E402
from predict import nb_over                # noqa: E402
from sklearn.metrics import log_loss       # noqa: E402

ART = Path("Model/artifacts")
BOOT = 400
rng = np.random.default_rng(0)


def prep(df, cols, cat_levels):            # evaluate_deep.prep, inlined
    X = df[cols].copy()
    for c, levels in cat_levels.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
    return X


def block_ci(dates, ll_cal_rows, ll_nb_rows):
    """Day-block bootstrap CI on mean(ll_nb) - mean(ll_cal): + = cal wins."""
    d = pd.factorize(dates)[0]
    n_days = d.max() + 1
    idx_by_day = [np.flatnonzero(d == i) for i in range(n_days)]
    deltas = []
    for _ in range(BOOT):
        take = rng.integers(0, n_days, n_days)
        rows = np.concatenate([idx_by_day[t] for t in take])
        deltas.append(ll_nb_rows[rows].mean() - ll_cal_rows[rows].mean())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(np.mean(deltas)), float(lo), float(hi)


def rowwise_ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def families(art, frames):
    """Yield (name, dates, mu, y, lines, line_cals, disp) per count family."""
    bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]
    yr = int(art["years"]["test"])
    cl = art["cat_levels"]
    sf_y = sf[(sf["Season"] == yr) & ~sf["ShortGame"].fillna(False)]
    bf_y = bf[(bf["Season"] == yr) & ~bf["ShortGame"].fillna(False)]
    gf_y = gf[(gf["Season"] == yr) & ~gf["ShortGame"].fillna(False)]
    gf_y = gf_y.dropna(subset=["total_runs"])

    from predict import K_LINES, TOTAL_LINES
    mu_k = art["k_model"].predict(prep(sf_y, art["st_cols"], cl))
    yield ("k", sf_y["Date"].to_numpy(), mu_k,
           sf_y["y_so"].to_numpy().astype(float), K_LINES,
           art.get("k_line_cals", {}), float(art.get("k_disp", 1.0)))

    for cname, head in art.get("count_models", {}).items():
        fr = bf_y if head["frame"] == "bat" else sf_y
        d = fr.dropna(subset=[head["target"]])
        mu = head["model"].predict(prep(d, head["cols"], cl))
        yield (cname, d["Date"].to_numpy(), mu,
               d[head["target"]].to_numpy().astype(float), head["lines"],
               head.get("line_cals", {}), float(head["disp"]))

    tg = F.build_team_game_frame(gf_y)
    tp = art["team_runs_model"].predict(prep(tg, art["tg_cols"], cl))
    n = len(gf_y)
    total_mu = tp[:n] + tp[n:]
    total_y = (pd.to_numeric(gf_y["HomeScore"], errors="coerce")
               + pd.to_numeric(gf_y["AwayScore"], errors="coerce")).to_numpy()
    ok = np.isfinite(total_y)
    yield ("total", gf_y["Date"].to_numpy()[ok], total_mu[ok], total_y[ok],
           TOTAL_LINES, art.get("total_line_cals", {}),
           float(art.get("total_disp", 1.0)))


frames = joblib.load(ART / "frames.joblib")
for fname in ("models_bt.joblib", "models.joblib"):
    art = joblib.load(ART / fname)
    yr = art["years"]["test"]
    print(f"\n=== {fname} (test {yr}) — logloss, + delta = calibrator wins "
          f"===")
    rows = []
    for name, dates, mu, y, lines, cals, disp in families(art, frames):
        for line in lines:
            lc = cals.get(line)
            if lc is None:
                continue
            over = (y > line).astype(float)
            if not 0 < over.mean() < 1:
                continue
            p_cal = lc.predict_proba(np.asarray(mu).reshape(-1, 1))[:, 1]
            p_nb = np.array([nb_over(m, line, disp) for m in mu])
            llc, lln = rowwise_ll(over, p_cal), rowwise_ll(over, p_nb)
            d, lo, hi = block_ci(dates, llc, lln)
            verdict = ("CAL" if lo > 0 else "NB" if hi < 0 else "tie")
            rows.append({
                "family": name, "line": line, "base%": round(over.mean(), 3),
                "ll_cal": round(llc.mean(), 5), "ll_nb": round(lln.mean(), 5),
                "delta": round(d, 5), "ci_lo": round(lo, 5),
                "ci_hi": round(hi, 5), "verdict": verdict})
    print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: families currently NB-priced at serving: per (+ any line "
      "without a calibrator). Everything else serves the calibrator today — "
      "for those, an 'NB' verdict would argue the REVERSE flip.")
