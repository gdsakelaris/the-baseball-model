"""PA-level outcome model — Phase 1 of the PA-sim program (2026-07-13).

One multinomial model over the 8 terminal plate-appearance outcomes

    K  BB  HBP  1B  2B  3B  HR  OUT

conditioned on batter talent, pitcher talent, platoon, base-out state and
park, trained on the raw Statcast pitch archive (Data/raw_pitches/, 2015-).
This is the engine a Monte-Carlo game simulation composes into every prop,
the winner and the total (Phase 2); Phase 1 only has to prove the engine
prices single PAs well.

KILL-GATE (decided before the sim is built):
  1. multiclass log loss on the 2025 holdout must CLEARLY beat both the
     league-rate baseline and the classic log5 batter x pitcher matchup
     (log5 already contains everything "talent-only"); and
  2. its implied per-game binaries (P(1+ hit), P(HR), P(2+ K) from the
     actual PA rows, exact Poisson-binomial DP) must land near the shipped
     game-grain heads' 2025 AUC. Exposure uses the game's ACTUAL PA count,
     an oracle no pregame model has — read those AUCs as an upper bound of
     the talent+matchup component, not a paired comparison.
If (1) fails, nothing downstream can save the sim: stop there.

Leakage rules match the main pipeline: every rate is as-of DAY-START
(a game's own day never informs its rows — doubleheader parity), the
league environment is a trailing-365-day window, 2026 is never touched.

Usage:
    python Model/pa_model.py            # build/refresh frame, train, gate
    python Model/pa_model.py --rebuild  # force PA-frame rebuild from parquet
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "Data" / "raw_pitches"
ART = HERE / "artifacts"
FRAME_CACHE = ART / "pa_frame.parquet"

CLASSES = ["K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT"]
EVENT_CLS = {
    "strikeout": "K", "strikeout_double_play": "K",
    "walk": "BB", "intent_walk": "BB",
    "hit_by_pitch": "HBP",
    "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    # reached-on-error counts as OUT here: the batter's bat made an out; the
    # sim can layer defense back on top (it has the lineup-OAA machinery)
    "field_out": "OUT", "force_out": "OUT", "field_error": "OUT",
    "grounded_into_double_play": "OUT", "double_play": "OUT",
    "triple_play": "OUT", "fielders_choice": "OUT",
    "fielders_choice_out": "OUT", "sac_fly": "OUT",
    "sac_fly_double_play": "OUT", "sac_bunt": "OUT",
}
# truncated_pa / catcher_interf are dropped: no batter outcome to learn.

# Per-class empirical-Bayes prior strength, in PAs — the classic
# stabilization points (K fastest, triples slowest). The vs-hand split
# shrinks toward the player's own overall rate with 2x the strength
# (half the effective sample -> twice the prior).
EB_K = {"K": 60, "BB": 120, "HBP": 240, "1B": 290, "2B": 500,
        "3B": 800, "HR": 170, "OUT": 120}

TRAIN_YRS = range(2015, 2024)   # train <= 2023
VAL_YR = 2024                   # early stopping
TEST_YR = 2025                  # kill-gate holdout; 2026 never touched

LGB_PA = dict(
    objective="multiclass", num_class=len(CLASSES),
    n_estimators=4000, learning_rate=0.03, num_leaves=127,
    min_child_samples=100, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=5.0, n_jobs=-1, verbosity=-1,
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- frame --

def build_pa_frame():
    """Terminal pitch of every PA -> one row with outcome class + context."""
    cols = ["game_pk", "game_date", "events", "batter", "pitcher", "stand",
            "p_throws", "inning_topbot", "on_1b", "on_2b", "on_3b",
            "outs_when_up", "inning", "home_team", "at_bat_number",
            "launch_speed", "launch_speed_angle"]
    parts = []
    for f in sorted(RAW.glob("pitches_*.parquet")):
        df = pd.read_parquet(f, columns=cols)
        df = df[df["events"].isin(EVENT_CLS)].copy()
        df["cls"] = df["events"].map(EVENT_CLS)
        parts.append(df.drop(columns=["events"]))
        log(f"  {f.name}: {len(df):,} PAs")
    pa = pd.concat(parts, ignore_index=True)
    pa["Date"] = pd.to_datetime(pa["game_date"])
    pa["Season"] = pa["Date"].dt.year
    pa["bat_home"] = (pa["inning_topbot"] == "Bot").astype(np.int8)
    for b in ("1b", "2b", "3b"):
        pa[f"on{b}"] = pa[f"on_{b}"].notna().astype(np.int8)
    pa = pa.drop(columns=["game_date", "inning_topbot",
                          "on_1b", "on_2b", "on_3b"])
    pa = pa.rename(columns={"home_team": "park"})
    pa = pa.sort_values(["Date", "game_pk"]).reset_index(drop=True)
    return pa


# ------------------------------------------------------- as-of EB rates --

def _asof_counts(pa, keys):
    """Day-start cumulative class counts per `keys` (+ Date): everything the
    entity did BEFORE today. Returns (Date-sorted frame keyed by keys+Date)
    with one count column per class and n = their sum."""
    day = (pa.groupby(keys + ["Date", "cls"], observed=True)
             .size().unstack("cls", fill_value=0)
             .reindex(columns=CLASSES, fill_value=0))
    day = day.sort_index()
    g = day.groupby(level=list(range(len(keys))), sort=False)
    cum = g.cumsum() - day          # exclude today: day-start parity
    cum["n"] = cum.sum(axis=1)
    return cum.reset_index()


def _league_trailing(pa, days=365):
    """Trailing-`days` league class mix as-of day-start, per date."""
    daily = (pa.groupby(["Date", "cls"], observed=True).size()
               .unstack("cls", fill_value=0)
               .reindex(columns=CLASSES, fill_value=0))
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()),
                         fill_value=0)
    roll = full.rolling(days, min_periods=1).sum().shift(1)
    mix = roll.div(roll.sum(axis=1), axis=0)
    mix.columns = [f"lg_{c}" for c in CLASSES]
    return mix


def _eb(counts, n, prior, k):
    """Shrink per-class counts toward `prior` (rate columns) with per-class
    strength k[c], then renormalize the 8 rates to sum to 1."""
    out = {}
    for c in CLASSES:
        kc = k[c]
        out[c] = (counts[c] + kc * prior[c]) / (n + kc)
    rates = pd.DataFrame(out)
    return rates.div(rates.sum(axis=1), axis=0)


def add_asof_features(pa):
    """Attach leakage-free talent + environment columns to every PA row."""
    # lazy import: milb_priors imports CLASSES/FRAME_CACHE from this module
    from milb_priors import build_all as _milb_build, prior_blend
    lg = _league_trailing(pa)
    pa = pa.merge(lg, left_on="Date", right_index=True, how="left")
    lg_prior = pa[[f"lg_{c}" for c in CLASSES]].rename(
        columns=lambda s: s[3:])
    milb = _milb_build()

    # (prefix, group keys, prior prefix or None=league, prior-strength mult).
    # Career rates shrink to league; the vs-hand split shrinks to the
    # player's own career (2x strength: half the sample, twice the prior);
    # season-to-date shrinks to career (recency without a decay parameter —
    # a full season overrides the career, April barely moves it).
    specs = [
        ("b",  ["batter"],             None, 1),   # batter career
        ("p",  ["pitcher"],            None, 1),   # pitcher-allowed career
        ("bh", ["batter", "p_throws"], "b",  2),   # batter vs pitcher hand
        ("ph", ["pitcher", "stand"],   "p",  2),   # pitcher vs batter side
        ("bs", ["batter", "Season"],   "b",  1),   # batter season-to-date
        ("ps", ["pitcher", "Season"],  "p",  1),   # pitcher season-to-date
    ]
    milb_n = {}
    for pre, keys, prior_pre, mult in specs:
        tab = _asof_counts(pa, keys)
        pa = pa.merge(tab, on=keys + ["Date"], how="left",
                      suffixes=("", "_t"))
        cnt = pa[CLASSES].fillna(0)
        n = pa["n"].fillna(0)
        if prior_pre is None:
            # career prior = league blended toward the player's translated
            # MiLB line (leakage-safe: serve rows only use seasons <= Y-1)
            serve = milb["bat" if pre == "b" else "pit"]["serve"]
            m = pa[[keys[0], "Season"]].merge(
                serve, left_on=[keys[0], "Season"],
                right_on=["PlayerId", "Season"], how="left")
            t = m[[f"t_{c}" for c in CLASSES]].to_numpy()
            ne = m["n_eff"].fillna(0).to_numpy()
            prior = pd.DataFrame(prior_blend(lg_prior.to_numpy(), t, ne),
                                 columns=CLASSES, index=pa.index)
            milb_n[pre] = np.log1p(ne)
        else:
            prior = pa[[f"{prior_pre}_{c}" for c in CLASSES]].rename(
                columns=lambda s: s[len(prior_pre) + 1:])
        k = {c: mult * v for c, v in EB_K.items()}
        rates = _eb(cnt, n, prior, k)
        pa[[f"{pre}_{c}" for c in CLASSES]] = rates.values
        pa[f"{pre}_n"] = np.log1p(n)
        pa = pa.drop(columns=CLASSES + ["n"])
        log(f"  as-of rates: {pre} ({'+'.join(keys)}) attached")
    pa["b_milb_n"] = milb_n["b"]
    pa["p_milb_n"] = milb_n["p"]
    pa["same_hand"] = (pa["stand"] == pa["p_throws"]).astype(np.int8)
    pa = _add_contact_quality(pa)
    return pa


# Contact-quality EB prior strengths, in balls in play. Barrel rate
# stabilizes fast (~50 BIP); mean exit velocity even faster.
CQ_K_BRL, CQ_K_EV = 50.0, 40.0


def _cq_tables(pa):
    """Day-start contact-quality state, shared by frame construction, the
    batch backtest and the live engine: cumulative barrel/EV/BIP sums per
    batter and per pitcher-allowed + the trailing-365d league values.
    Tables only have rows on dates the player put a ball in play —
    consumers must join merge_asof (backward), never exact-date."""
    bip = pa["launch_speed"].notna()
    d = pa[bip][["batter", "pitcher", "Date"]].copy()
    d["brl"] = (pa.loc[bip, "launch_speed_angle"] == 6).astype(float)
    d["ev"] = pa.loc[bip, "launch_speed"].astype(float)

    daily = d.groupby("Date")[["brl", "ev"]].agg(["sum", "count"])
    full = daily.reindex(pd.date_range(daily.index.min(),
                                       daily.index.max()), fill_value=0)
    roll = full.rolling(365, min_periods=1).sum().shift(1)
    lg_brl = (roll[("brl", "sum")] / roll[("brl", "count")]).rename("lg")
    lg_ev = (roll[("ev", "sum")] / roll[("ev", "count")]).rename("lg")

    tabs = {}
    for pre, key in (("b", "batter"), ("p", "pitcher")):
        day = (d.groupby([key, "Date"])
                 .agg(brl=("brl", "sum"), ev=("ev", "sum"),
                      n=("brl", "count")).sort_index())
        g = day.groupby(level=0, sort=False)
        cum = (g.cumsum() - day).reset_index()          # day-start
        tabs[pre] = cum.sort_values("Date", kind="stable")
    return tabs, lg_brl, lg_ev


def _cq_shrink(brl, ev, n, lg_brl, lg_ev):
    """EB-shrink cumulative barrel/EV sums toward the league values.
    Returns the three feature arrays: barrel rate, mean EV, log1p BIP."""
    n = np.nan_to_num(np.asarray(n, float))
    brl = np.nan_to_num(np.asarray(brl, float))
    ev = np.nan_to_num(np.asarray(ev, float))
    lb = np.nan_to_num(np.asarray(lg_brl, float), nan=0.06)
    le = np.nan_to_num(np.asarray(lg_ev, float), nan=88.5)
    return ((brl + CQ_K_BRL * lb) / (n + CQ_K_BRL),
            (ev + CQ_K_EV * le) / (n + CQ_K_EV),
            np.log1p(n))


def _add_contact_quality(pa):
    """As-of empirical-Bayes contact quality per batter and per pitcher-
    allowed, from the BIP rows already in the frame (launch_speed present):
    barrel rate (launch_speed_angle == 6) and mean exit velocity, day-start,
    shrunk toward the trailing-365d league values. Adds 6 columns:
    {b,p}_brl, {b,p}_ev, {b,p}_bip (log1p BIP count)."""
    tabs, lg_brl, lg_ev = _cq_tables(pa)
    lb = lg_brl.reindex(pa["Date"]).to_numpy()
    le = lg_ev.reindex(pa["Date"]).to_numpy()
    for pre, key in (("b", "batter"), ("p", "pitcher")):
        # merge_asof, not exact-date: the table only has rows on dates the
        # player put a ball in play — an all-K/BB day must still see the
        # last KNOWN career cumulative, not collapse to the league prior
        left = pa[[key, "Date"]].reset_index()
        left = left.sort_values("Date", kind="stable")
        m = pd.merge_asof(left, tabs[pre], on="Date", by=key,
                          direction="backward")
        m = m.set_index("index").sort_index()
        pa[[f"{pre}_brl", f"{pre}_ev", f"{pre}_bip"]] = np.column_stack(
            _cq_shrink(m["brl"], m["ev"], m["n"], lb, le))
    return pa


def feature_cols():
    pres = ("b", "bh", "bs", "p", "ph", "ps")
    rate = [f"{pre}_{c}" for pre in pres for c in CLASSES]
    ns = [f"{pre}_n" for pre in pres]
    lg = [f"lg_{c}" for c in CLASSES]
    cq = [f"{pre}_{s}" for pre in ("b", "p") for s in ("brl", "ev", "bip")]
    milb = ["b_milb_n", "p_milb_n"]
    ctx = ["same_hand", "bat_home", "outs_when_up", "inning",
           "on1b", "on2b", "on3b", "park", "stand", "p_throws"]
    return rate + ns + lg + cq + milb + ctx


# ------------------------------------------------------------ baselines --

def log5_probs(pa):
    """Classic multinomial log5: p_c ∝ (batter_c * pitcher_c) / league_c,
    on the vs-hand EB rates — the strongest 'talent-only' reference."""
    b = pa[[f"bh_{c}" for c in CLASSES]].to_numpy()
    p = pa[[f"ph_{c}" for c in CLASSES]].to_numpy()
    l = pa[[f"lg_{c}" for c in CLASSES]].to_numpy()
    raw = b * p / np.clip(l, 1e-6, None)
    return raw / raw.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ kill-gate --

def _per_game_binaries(te, proba):
    """Exact per-game event probabilities from per-PA class probs via the
    Poisson-binomial DP, grouped on the game's ACTUAL PA rows."""
    d = te[["game_pk", "batter"]].copy()
    d["p_hit"] = proba[:, [CLASSES.index(c)
                           for c in ("1B", "2B", "3B", "HR")]].sum(axis=1)
    d["p_hr"] = proba[:, CLASSES.index("HR")]
    d["p_k"] = proba[:, CLASSES.index("K")]
    cls = te["cls"].to_numpy()
    d["y_hit"] = np.isin(cls, ("1B", "2B", "3B", "HR")).astype(int)
    d["y_hr"] = (cls == "HR").astype(int)
    d["y_k"] = (cls == "K").astype(int)

    g = d.groupby(["game_pk", "batter"], sort=False)
    out = g.agg(hit0=("p_hit", lambda p: np.prod(1 - p)),
                hr0=("p_hr", lambda p: np.prod(1 - p)),
                k0=("p_k", lambda p: np.prod(1 - p)),
                k_r=("p_k", lambda p: np.sum(p / (1 - p))),
                y_hit=("y_hit", "max"), y_hr=("y_hr", "max"),
                y_k2=("y_k", lambda y: int(y.sum() >= 2)))
    out["p_hit1"] = 1 - out["hit0"]
    out["p_hr1"] = 1 - out["hr0"]
    out["p_k2"] = 1 - out["k0"] * (1 + out["k_r"])   # P(K>=2)
    return out


def _pregame_game_binaries(te, proba):
    """Fully pregame-legal per-game read: STARTERS only (BattingOrder from
    the box scores), the batter's FIRST PA of the game (no in-game base/out
    context), a flat 4.3-PA exposure (no actual-PA oracle). Outcomes are the
    batter's real full game. This is the honest number to hold next to the
    shipped game-grain heads — same event, pregame information only (still
    not row-paired with them, so read direction, not decimals)."""
    d = te[["game_pk", "batter", "at_bat_number"]].copy()
    d["p_hit"] = proba[:, [CLASSES.index(c)
                           for c in ("1B", "2B", "3B", "HR")]].sum(axis=1)
    d["p_hr"] = proba[:, CLASSES.index("HR")]
    d["p_k"] = proba[:, CLASSES.index("K")]
    cls = te["cls"].to_numpy()
    d["y_hit"] = np.isin(cls, ("1B", "2B", "3B", "HR")).astype(int)
    d["y_hr"] = (cls == "HR").astype(int)
    d["y_k"] = (cls == "K").astype(int)

    first = (d.sort_values("at_bat_number")
              .groupby(["game_pk", "batter"], sort=False)
              [["p_hit", "p_hr", "p_k"]].first())
    ys = d.groupby(["game_pk", "batter"], sort=False).agg(
        y_hit=("y_hit", "max"), y_hr=("y_hr", "max"),
        y_k2=("y_k", lambda y: int(y.sum() >= 2)))
    gb = first.join(ys).reset_index()

    starters = pd.read_csv(HERE.parent / "Data" / "mlb_game_batting.csv",
                           usecols=["GamePk", "PlayerId", "BattingOrder"])
    starters = starters[pd.to_numeric(starters["BattingOrder"],
                                      errors="coerce") % 100 == 0]
    gb = gb.merge(starters, left_on=["game_pk", "batter"],
                  right_on=["GamePk", "PlayerId"], how="inner")

    n = 4.3                                     # flat starter exposure
    gb["p_hit1"] = 1 - (1 - gb["p_hit"]) ** n
    gb["p_hr1"] = 1 - (1 - gb["p_hr"]) ** n
    q = gb["p_k"]
    gb["p_k2"] = 1 - (1 - q) ** n - n * q * (1 - q) ** (n - 1)
    return gb


def gate_report(te, model_proba, log5_p, lg_p):
    y = te["cls_id"].to_numpy()
    print("\n=== KILL-GATE: per-PA multiclass log loss (2025 holdout) ===")
    for name, pr in [("league mix", lg_p), ("log5 b x p", log5_p),
                     ("PA model", model_proba)]:
        ll = log_loss(y, np.clip(pr, 1e-9, 1), labels=range(len(CLASSES)))
        print(f"  {name:12} {ll:.5f}")

    print("\n=== per-PA HR / K discrimination ===")
    for name, pr in [("log5 b x p", log5_p), ("PA model", model_proba)]:
        auc_hr = roc_auc_score((y == CLASSES.index("HR")).astype(int),
                               pr[:, CLASSES.index("HR")])
        auc_k = roc_auc_score((y == CLASSES.index("K")).astype(int),
                              pr[:, CLASSES.index("K")])
        print(f"  {name:12} HR {auc_hr:.4f} | K {auc_k:.4f}")

    print("\n=== implied per-game binaries (ACTUAL-PA oracle exposure; "
          "upper bound, not paired) ===")
    print("  shipped 2025 reference (pooled workbook): "
          "hit .576 | hr .636 | bk2 .651")
    for name, pr in [("log5 b x p", log5_p), ("PA model", model_proba)]:
        gb = _per_game_binaries(te, pr)
        print(f"  {name:12} "
              f"hit {roc_auc_score(gb['y_hit'], gb['p_hit1']):.4f} | "
              f"hr {roc_auc_score(gb['y_hr'], gb['p_hr1']):.4f} | "
              f"bk2 {roc_auc_score(gb['y_k2'], gb['p_k2']):.4f}")

    print("\n=== implied per-game binaries, PREGAME-LEGAL (starters, "
          "first-PA prob, flat 4.3 PA) ===")
    print("  shipped 2025 reference (pooled workbook): "
          "hit .576 | hr .636 | bk2 .651")
    for name, pr in [("log5 b x p", log5_p), ("PA model", model_proba)]:
        gb = _pregame_game_binaries(te, pr)
        print(f"  {name:12} "
              f"hit {roc_auc_score(gb['y_hit'], gb['p_hit1']):.4f} | "
              f"hr {roc_auc_score(gb['y_hr'], gb['p_hr1']):.4f} | "
              f"bk2 {roc_auc_score(gb['y_k2'], gb['p_k2']):.4f} "
              f"(n={len(gb):,})")


# ----------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild the PA frame from Data/raw_pitches")
    args = ap.parse_args()

    if FRAME_CACHE.exists() and not args.rebuild:
        pa = pd.read_parquet(FRAME_CACHE)
        log(f"PA frame from cache: {len(pa):,} rows "
            f"({pa['Season'].min()}-{pa['Season'].max()})")
    else:
        log("building PA frame from raw_pitches ...")
        pa = build_pa_frame()
        pa.to_parquet(FRAME_CACHE, index=False)
        log(f"cached {len(pa):,} PAs -> {FRAME_CACHE.name}")

    log("attaching as-of EB talent/environment features ...")
    pa = add_asof_features(pa)
    pa["cls_id"] = pa["cls"].map({c: i for i, c in enumerate(CLASSES)})
    for c in ("park", "stand", "p_throws"):
        pa[c] = pa[c].astype("category")

    cols = feature_cols()
    tr = pa[pa["Season"].isin(TRAIN_YRS)]
    va = pa[pa["Season"] == VAL_YR]
    te = pa[pa["Season"] == TEST_YR]
    log(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} "
        f"| {len(cols)} features")

    m = lgb.LGBMClassifier(**LGB_PA)
    m.fit(tr[cols], tr["cls_id"],
          eval_set=[(va[cols], va["cls_id"])], eval_metric="multi_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False),
                     lgb.log_evaluation(200)])
    log(f"best iteration: {m.best_iteration_}")

    import joblib
    joblib.dump({"model": m, "cols": cols, "classes": CLASSES,
                 "eb_k": EB_K, "trained_on": f"{TRAIN_YRS[0]}-{TRAIN_YRS[-1]}"
                 f", val {VAL_YR}, gate-tested {TEST_YR}"},
                ART / "pa_model.joblib", compress=3)
    log(f"saved {ART / 'pa_model.joblib'}")

    proba = m.predict_proba(te[cols])
    lgp = te[[f"lg_{c}" for c in CLASSES]].to_numpy()
    gate_report(te, proba, log5_probs(te), lgp)

    imp = pd.Series(m.feature_importances_, index=cols)
    print("\ntop 20 features:")
    print(imp.sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
