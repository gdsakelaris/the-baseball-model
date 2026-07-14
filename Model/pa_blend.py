"""PA-sim Phase 3 preview — does blending the sim with the incumbent beat
the incumbent alone?  Fit the blend on ONE year, evaluate on the OTHER
(fit-2025 → confirm-2026 is the honest house direction; the reverse run is
printed for completeness).

Binaries + winner: p = sigmoid(w·logit(sim) + (1−w)·logit(inc)), w chosen
on the fit year by log loss over a grid (w=0 ≡ incumbent alone, so the fit
can only choose the sim's help voluntarily). Count means + total + score:
linear mu blend, alpha by MAE on the fit year. Evaluation on the confirm
year uses the house day-block bootstrap CI (+ = blend better than the
incumbent alone).

    python Model/pa_blend.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from pa_grade import BAT_PROPS, COUNT_MEANS, _sim, _block_delta, verdict

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
GRID = np.linspace(0.0, 1.0, 21)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _merged(year):
    tag = "select_2025" if year == 2025 else "2026"
    snap = joblib.load(ART / f"eval_paired_{tag}.joblib")
    bat, st, gm = _sim(year, "bat"), _sim(year, "starter"), _sim(year, "game")
    out = {}
    for prop in BAT_PROPS:
        e = snap["binary"][prop]["df"]
        m = e.merge(bat[["GamePk", "PlayerId", f"p_{prop}"]],
                    on=["GamePk", "PlayerId"], how="inner")
        out[prop] = ("bin", m["Date"].to_numpy(), m["p"].to_numpy(),
                     m[f"p_{prop}"].to_numpy(), m["y"].to_numpy(float))
    for head, col in COUNT_MEANS.items():
        e = snap["count"][head]["df"]
        src = bat if head in ("xbk", "xhrr", "xtb") else st
        m = e.merge(src[["GamePk", "PlayerId", col]],
                    on=["GamePk", "PlayerId"], how="inner")
        out[head] = ("mu", m["Date"].to_numpy(), m["mu"].to_numpy(),
                     m[col].to_numpy(), m["y"].to_numpy(float))
    e = snap["count"]["total"]["df"]
    m = e.merge(gm[["GamePk", "x_total"]], on="GamePk", how="inner")
    out["total"] = ("mu", m["Date"].to_numpy(), m["mu"].to_numpy(),
                    m["x_total"].to_numpy(), m["y"].to_numpy(float))
    e = snap["winner"]["df"]
    m = e.merge(gm[["GamePk", "p_home_win"]], on="GamePk", how="inner")
    out["winner"] = ("bin", m["Date"].to_numpy(), m["p"].to_numpy(),
                     m["p_home_win"].to_numpy(), m["y"].to_numpy(float))
    e = snap["score"]["df"]
    sides = gm.melt(id_vars=["GamePk"], value_vars=["x_away", "x_home"],
                    var_name="side", value_name="x_score")
    sides["Home"] = (sides["side"] == "x_home").astype(e["Home"].dtype)
    m = e.merge(sides[["GamePk", "Home", "x_score"]],
                on=["GamePk", "Home"], how="inner")
    out["score"] = ("mu", m["Date"].to_numpy(), m["mu"].to_numpy(),
                    m["x_score"].to_numpy(), m["y"].to_numpy(float))
    return out


def fit_w(kind, inc, sim, y):
    best_w, best = 0.0, np.inf
    for w in GRID:
        if kind == "bin":
            p = _sig(w * _logit(sim) + (1 - w) * _logit(inc))
            v = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
        else:
            v = np.abs(y - (w * sim + (1 - w) * inc)).mean()
        if v < best - 1e-12:
            best, best_w = v, w
    return best_w


def run(fit_year, conf_year):
    fit, conf = _merged(fit_year), _merged(conf_year)
    rows = []
    for head in fit:
        kind, _, inc_f, sim_f, y_f = fit[head]
        w = fit_w(kind, inc_f, sim_f, y_f)
        kind, dates, inc_c, sim_c, y_c = conf[head]
        if kind == "bin":
            blend = _sig(w * _logit(sim_c) + (1 - w) * _logit(inc_c))
            d_ll, lo, hi = _block_delta(dates, inc_c, blend, "logloss", y_c)
            d_auc, lo_a, hi_a = _block_delta(dates, inc_c, blend, "auc", y_c)
            rows.append({"head": head, "w_sim": w,
                         "inc": round(log_loss(y_c, np.clip(inc_c, 1e-6,
                                                            1 - 1e-6)), 5),
                         "blend": round(log_loss(y_c, np.clip(blend, 1e-6,
                                                              1 - 1e-6)), 5),
                         "dLL": round(d_ll, 5), "ll_v": verdict(lo, hi),
                         "dAUC": round(d_auc, 4),
                         "auc_v": verdict(lo_a, hi_a)})
        else:
            blend = w * sim_c + (1 - w) * inc_c
            inc_err = np.abs(y_c - inc_c)
            bl_err = np.abs(y_c - blend)
            d, lo, hi = _block_delta(dates, inc_err, bl_err, "mae")
            rows.append({"head": head, "w_sim": w,
                         "inc": round(inc_err.mean(), 4),
                         "blend": round(bl_err.mean(), 4),
                         "dLL": round(d, 4), "ll_v": verdict(lo, hi),
                         "dAUC": np.nan, "auc_v": "(MAE)"})
    df = pd.DataFrame(rows)
    print(f"\n=== BLEND (w fit on {fit_year}) vs incumbent alone, evaluated "
          f"{conf_year} (+ = blend better) ===")
    print(df.to_string(index=False))
    df.to_csv(ART / f"sim_blend_{fit_year}to{conf_year}.csv", index=False)


if __name__ == "__main__":
    run(2025, 2026)
    run(2026, 2025)
