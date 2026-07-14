"""One-off: inject K + total per-line calibrators into the artifacts the
2026-07-13 PlattCal train produced, so the full-surface calibration pass
needs no third retrain. Fits are IDENTICAL to what the new train.py code
does natively (same fit_line_cals helper, same cal-year frames from the
same frames.joblib cache, deterministic logistic) — tomorrow's 06:00
rebuild reproduces these from train.py and this script is never needed
again. Run AFTER the train completes, from the MLB repo root."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("Model").resolve()))
import features as F                      # noqa: E402
from train import fit_line_cals           # noqa: E402
from predict import K_LINES, TOTAL_LINES  # noqa: E402

ART = Path("Model/artifacts")


def prep(df, cols, cat_levels):           # evaluate_deep.prep, inlined
    X = df[cols].copy()
    for c, levels in cat_levels.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c].astype("object"), categories=levels)
    return X


frames = joblib.load(ART / "frames.joblib")
bf, sf, gf = frames["bf"], frames["sf"], frames["gf"]
# team-game frame exactly as train.py main builds it (canonical order)
tg = F.build_team_game_frame(gf.dropna(subset=["total_runs"]))
tg = tg.dropna(subset=["y_runs"])
tg = tg.sort_values(["GamePk", "Home"]).reset_index(drop=True)

for fname in ("models_bt.joblib", "models.joblib"):
    art = joblib.load(ART / fname)
    cal = int(art["years"]["cal"])

    sf_cal = sf[sf["Season"] == cal]
    mu_k = art["k_model"].predict(
        prep(sf_cal, art["st_cols"], art["cat_levels"]))
    art["k_line_cals"] = fit_line_cals(
        mu_k, sf_cal["y_so"].to_numpy(), K_LINES)

    tg_cal = tg[tg["Season"] == cal]
    mu_t = art["team_runs_model"].predict(
        prep(tg_cal, art["tg_cols"], art["cat_levels"]))
    per_game = pd.DataFrame({"g": tg_cal["GamePk"].to_numpy(), "mu": mu_t,
                             "y": tg_cal["y_runs"].to_numpy()}
                            ).groupby("g").sum()
    art["total_line_cals"] = fit_line_cals(
        per_game["mu"].to_numpy(), per_game["y"].to_numpy(), TOTAL_LINES)

    joblib.dump(art, ART / fname, compress=3)
    ks = {ln: (round(float(c.coef_[0][0]), 3), round(float(c.intercept_[0]), 3))
          for ln, c in art["k_line_cals"].items()}
    ts = {ln: (round(float(c.coef_[0][0]), 3), round(float(c.intercept_[0]), 3))
          for ln, c in art["total_line_cals"].items()}
    print(f"{fname} (cal {cal}): K {ks}")
    print(f"{'':>{len(fname)}}  total {ts}")
print("done — artifacts now match what the updated train.py produces")
