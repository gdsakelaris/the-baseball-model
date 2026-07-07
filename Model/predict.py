"""Prediction engine: turn a game spec into betting-relevant predictions.

Given a game (teams, date, venue, starters, ordered lineups, weather), the
Predictor outputs, per run:
  - every lineup batter's calibrated P(home run), P(1+ hit), expected HRs,
    and fair American odds for the HR prop
  - each starter's expected strikeouts and P(over) for common K lines
  - game totals: expected lineup home runs and expected total runs

Usage:
    python Model/predict.py --game 745444      # replay a historical game
    python Model/predict.py --selftest         # train/serve parity check
"""

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import recalibrate as R  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"

K_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

# prop key -> output column, in display order
PROP_COLS = {"hr": "P_HR", "hit": "P_Hit", "hits2": "P_2Hits",
             "tb2": "P_TB2", "run": "P_Run", "rbi": "P_RBI",
             "bb": "P_BB", "sb": "P_SB"}


def american_odds(p):
    if not (0 < p < 1):
        return ""
    if p >= 0.5:
        return f"-{round(100 * p / (1 - p)):d}"
    return f"+{round(100 * (1 - p) / p):d}"


def poisson_over(lam, line):
    """P(count > line) for a half-point line under Poisson(lam)."""
    k = int(math.floor(line))
    cdf = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k + 1))
    return 1 - cdf


def nb_over(lam, line, disp=1.0):
    """P(count > line) when variance = disp * mean (negative binomial).

    Real MLB game totals are ~2x more variable than Poisson (blowouts,
    extras, shared park/weather), which made pure-Poisson P(over) worse
    than the base rate at low run lines. `disp` is measured on the
    calibration year at train time. Falls back to Poisson at disp <= 1
    (starter strikeouts measured ~1.04, so K lines stay Poisson)."""
    if disp <= 1.001 or lam <= 0:
        return poisson_over(lam, line)
    r = lam / (disp - 1.0)          # NB: mean lam, variance disp*lam
    log_p, log_q = -math.log(disp), math.log(1.0 - 1.0 / disp)
    lg_r = math.lgamma(r)
    k = int(math.floor(line))
    cdf = sum(math.exp(math.lgamma(i + r) - lg_r - math.lgamma(i + 1)
                       + r * log_p + i * log_q) for i in range(k + 1))
    return 1 - cdf


def predict_prop(prop, X):
    """Calibrated probability from a prop's GBM+logistic blend. X is prepped
    with the full batter column set; props trained on a subset (per-prop
    feature selection) carry their own column list. Backwards-compatible
    with older {model, iso} artifacts."""
    if "gbm" in prop:
        Xp = X[prop["cols"]] if "cols" in prop else X
        g = prop["gbm"].predict_proba(Xp)[:, 1]
        l = prop["lr"].predict_proba(Xp[prop["lr_cols"]])[:, 1]
        w = prop["w"]
        return prop["iso"].predict(w * g + (1 - w) * l)
    return prop["iso"].predict(prop["model"].predict_proba(X)[:, 1])


def poisson_win(mu_a, mu_b, kmax=30):
    """P(team A outscores team B) for independent Poisson scores; ties
    (games headed to extras) are split evenly."""
    pa = [math.exp(-mu_a) * mu_a ** i / math.factorial(i) for i in range(kmax)]
    pb = [math.exp(-mu_b) * mu_b ** i / math.factorial(i) for i in range(kmax)]
    win = sum(pa[i] * sum(pb[:i]) for i in range(1, kmax))
    tie = sum(pa[i] * pb[i] for i in range(kmax))
    return win + 0.5 * tie


def predict_win(win_art, X, mu_home, mu_away):
    """Home-win probability from the winner artifact: GBM+logistic blend,
    then (when trained with one) a second blend with the runs-model Poisson
    win probability, then isotonic calibration. Backwards-compatible with
    artifacts that have no Poisson blend (w_ml defaults to 1)."""
    g = win_art["gbm"].predict_proba(X[win_art["cols"]])[:, 1]
    l = win_art["lr"].predict_proba(X[win_art["lr_cols"]])[:, 1]
    s = win_art["w"] * g + (1 - win_art["w"]) * l
    w_ml = win_art.get("w_ml", 1.0)
    if w_ml < 1.0:
        pois = np.array([poisson_win(h, a) for h, a in
                         zip(np.atleast_1d(mu_home), np.atleast_1d(mu_away))])
        s = w_ml * s + (1 - w_ml) * np.where(np.isfinite(pois), pois, s)
    return win_art["iso"].predict(s)


class Predictor:
    def __init__(self, stores=None, progress=None, recal=False):
        tick = progress or (lambda msg: None)
        tick("loading models...")
        self.art = joblib.load(ART / "models.joblib")
        # opt-in in-season drift correction (evaluate_deep Section 10 is the
        # evidence for turning it on); offsets are refreshed by the daily retrain
        self.recal = recal
        self.offsets = self.art.get("inseason_offsets") or {}
        self.stores = stores or F.Stores(progress=progress)
        tick("indexing names...")
        # Name lookup, broadest to most current: batting AND pitching logs
        # (pitchers never bat post-DH, so batting logs alone miss them),
        # then the handedness file, then rosters.
        self.names = {}
        for frame in (self.stores.raw["gb"], self.stores.raw["gp"]):
            last = frame.sort_values("Date").groupby("PlayerId")["Name"].last()
            self.names.update(last.to_dict())
        hands = self.stores.raw["hands"]
        for pid, name in zip(hands["PlayerId"], hands["Name"]):
            if name:
                self.names.setdefault(pid, name)
        for pid, row in self.stores.rosters.iterrows():
            self.names[pid] = row["Name"]

    def _name(self, pid, spec=None):
        """Display name; falls back to names scraped into the game spec
        (covers debut players with no MLB history yet), then the raw id."""
        n = self.names.get(pid)
        if n:
            return n
        names = (spec or {}).get("names") or {}
        return names.get(str(pid)) or names.get(pid) or str(pid)

    # ---------------------------------------------------------- rows

    def _weather(self, spec):
        """Missing weather inputs become NaN/blank; the models handle both."""
        def num(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return np.nan
        def txt(v):
            return str(v).strip().title() if v not in (None, "") else ""
        return {"Temp": num(spec.get("temp")),
                "WindSpeed": num(spec.get("wind_speed")),
                "WindDir": txt(spec.get("wind_dir")),
                "Condition": txt(spec.get("condition")),
                "DayNight": spec.get("day_night") or ""}

    def _batter_rows(self, spec):
        date = pd.Timestamp(spec["date"])
        season = date.year
        s = self.stores
        wx = self._weather(spec)
        park = s.park(spec["venue"], date)
        env = s.league_env(date)

        rows, meta = [], []
        sides = [("away", spec["away_team"], spec["home_team"],
                  spec["away_lineup"], spec["home_starter"], 0),
                 ("home", spec["home_team"], spec["away_team"],
                  spec["home_lineup"], spec["away_starter"], 1)]
        for side, team, opp, lineup, opp_starter, home in sides:
            # unknown opposing starter -> sentinel id; every starter-derived
            # feature (incl. arsenal matchup, platoon) comes back NaN
            opp_starter = opp_starter if opp_starter else -1
            st_feats = s.starter_feats(opp_starter, date, season)
            toff = s.team_offense(team, season, date)
            toff_loc = s.team_offense_loc(team, season, home, date)
            pen = s.bullpen(opp, season, date)
            pen_hl = s.bullpen_hl(opp, season, date)
            pen_fat = {"pen_np_l3": s.pen_fatigue(opp, date)}
            phrq = s.pitcher_hr_quality(opp_starter, date)
            pprior = s.pitcher_prior(opp_starter, season)
            p_bio = s.bio(opp_starter)
            opp_hand = p_bio["pit_throws"]
            side_rows = []
            for pid, slot in lineup:
                b = s.batter_feats(pid, date, season, opp_hand=opp_hand,
                                   home=home)
                bio = s.bio(pid)
                row = {"slot": slot, "Home": home, "Season": season,
                       "month": date.month, **b, **st_feats, **toff,
                       **toff_loc, **pen, **pen_hl, **pen_fat, **phrq,
                       **pprior, **park, **wx, **env,
                       "hrpt_score": s.hrpt(pid, opp_starter, season, date),
                       "bat_hand": bio["bat_hand"], "pit_throws": p_bio["pit_throws"],
                       "bat_height": bio["height"], "bat_weight": bio["weight"],
                       "bat_age": ((date - bio["dob"]).days / 365.25
                                   if pd.notna(bio["dob"]) else np.nan)}
                bh, pt = bio["bat_hand"], p_bio["pit_throws"]
                eff = np.nan
                if pd.isna(bh):
                    row["same_hand"] = np.nan
                elif bh == "S":
                    eff = ("R" if pt == "L" else "L") if pd.notna(pt) else np.nan
                    row["same_hand"] = float(eff == pt) if pd.notna(pt) else np.nan
                else:
                    eff = bh
                    row["same_hand"] = float(eff == pt) if pd.notna(pt) else np.nan
                # pull-porch: the fence on the batter's pull side, and whether
                # his typical career HR distance clears it
                if eff == "L":
                    row["pull_fence"] = park["RF"]
                elif eff == "R":
                    row["pull_fence"] = park["LF"]
                else:
                    row["pull_fence"] = np.nan
                row["porch_margin"] = row["hrq_dist_avg"] - row["pull_fence"]
                side_rows.append(row)
                meta.append({"Team": team, "PlayerId": pid, "slot": slot,
                             "Name": self._name(pid, spec),
                             "BatterId": pid, "PitcherId": opp_starter,
                             "Season": season})
            # teammate context: career on-base of the two slots ahead,
            # slugging of the two behind, wrapping the order (mirrors the
            # vectorized ctx_* computation; missing slots skipna)
            omap = {r["slot"]: r.get("_obpp") for r in side_rows}
            smap = {r["slot"]: r.get("_slgp") for r in side_rows}

            def _nmean(vals):
                vals = [v for v in vals if v is not None and pd.notna(v)]
                return float(np.mean(vals)) if vals else np.nan

            for r in side_rows:
                r["ctx_ahead_obp"] = _nmean(
                    [omap.get(((r["slot"] + off - 1) % 9) + 1)
                     for off in (-2, -1)])
                r["ctx_behind_slg"] = _nmean(
                    [smap.get(((r["slot"] + off - 1) % 9) + 1)
                     for off in (1, 2)])
            rows.extend(side_rows)
        df = pd.DataFrame(rows)
        mdf = pd.DataFrame(meta)
        # arsenal matchup features for the 18 pairs
        mu = F.matchup_features(mdf[["BatterId", "PitcherId", "Season"]],
                                self.stores.raw["ars_p"], self.stores.raw["ars_b"])
        mdf2 = mdf.merge(mu, on=["BatterId", "PitcherId", "Season"], how="left")
        for c in [*F.ARS_P_METRICS.values(), *F.ARS_B_METRICS.values(), "m_coverage"]:
            df[c] = mdf2[c].to_numpy()
        return df, mdf

    _LU_COLS = {"lu_k_sh": "s_k_pct_sh", "lu_bb_sh": "s_bb_pct_sh",
                "lu_whiff": "m_whiff", "lu_vsh_k": "vsh_k_pct_sh"}

    def _starter_rows(self, spec, bdf=None, bmeta=None):
        """K-model rows. bdf/bmeta (the batter rows) supply the
        opposing-lineup aggregates; without them the lu_* features are NaN."""
        date = pd.Timestamp(spec["date"])
        season = date.year
        s = self.stores
        wx = self._weather(spec)
        park = s.park(spec["venue"], date)
        env = s.league_env(date)
        rows, meta = [], []
        for pid, team, opp, home in [
                (spec["away_starter"], spec["away_team"], spec["home_team"], 0),
                (spec["home_starter"], spec["home_team"], spec["away_team"], 1)]:
            if not pid:  # starter not specified -> no K projection for side
                continue
            self._starter_sanity(pid, spec)
            f = s.starter_feats(pid, date, season)
            vs = s.team_offense(opp, season, date, prefix="vs")
            row = {"Season": season, "month": date.month, "Home": home,
                   **f, **vs, **park, **wx, **env}
            # the starter's own arsenal, K-model view (same helper as training)
            pa = F.pitcher_arsenal_feats(
                pd.DataFrame({"PitcherId": [pid], "Season": [season]}),
                s.raw["ars_p"])
            for c in [*F.ARS_K_METRICS.values(), "pars_cov"]:
                row[c] = pa[c].iloc[0]
            # the actual opposing lineup he faces (mean of the batter rows)
            for dst, src in self._LU_COLS.items():
                if bdf is not None and bmeta is not None and len(bdf):
                    mask = (bmeta["Team"] == opp).to_numpy()
                    row[dst] = bdf.loc[mask, src].mean() if mask.any() else np.nan
                else:
                    row[dst] = np.nan
            rows.append(row)
            meta.append({"PlayerId": pid, "Team": team, "Opponent": opp,
                         "Name": self._name(pid, spec)})
        return pd.DataFrame(rows), pd.DataFrame(meta)

    def _starter_sanity(self, pid, spec):
        """Soft check: warn when a listed starter is a bullpen arm on the
        current roster (likely a mis-picked player in the GUI)."""
        try:
            pos = self.stores.rosters.loc[pid, "Position"]
        except KeyError:
            return
        if isinstance(pos, str) and pos.strip().lower() == "bullpen":
            print(f"note: listed starter {self._name(pid, spec)} is a "
                  f"bullpen arm on the current roster — check the pick")

    def _team_rows(self, spec):
        """One row per team: own offense vs opposing pitching (order:
        away first, home second)."""
        date = pd.Timestamp(spec["date"])
        season = date.year
        s = self.stores
        wx = self._weather(spec)
        park = s.park(spec["venue"], date)
        env = s.league_env(date)
        rows = []
        sides = [(spec["away_team"], spec["home_starter"],
                  spec["home_team"], 0),
                 (spec["home_team"], spec["away_starter"],
                  spec["away_team"], 1)]
        for team, opp_starter, opp, home in sides:
            opp_starter = opp_starter if opp_starter else -1
            toff = s.team_offense(team, season, date, prefix="off")
            toff_loc = s.team_offense_loc(team, season, home, date,
                                          prefix="off_loc")
            pen = s.bullpen(opp, season, date, prefix="opp_pen")
            pen_hl = s.bullpen_hl(opp, season, date, prefix="opp_pen_hl")
            stf = s.starter_feats(opp_starter, date, season)
            rows.append({
                "Season": season, "month": date.month, "Home": home,
                "off_hr_pa": toff["off_hr_pa"], "off_r_pg": toff["off_r_pg"],
                "off_k_pct": toff["off_k_pct"],
                "off_loc_hr_pa": toff_loc["off_loc_hr_pa"],
                "off_loc_r_pg": toff_loc["off_loc_r_pg"],
                "opp_pen_era": pen["opp_pen_era"],
                "opp_pen_hl_era": pen_hl["opp_pen_hl_era"],
                "opp_pen_np_l3": s.pen_fatigue(opp, date),
                "opp_def_uer": s.team_defense(opp, season, date),
                "opp_ps_era": stf["ps_era"], "opp_ps_k_bf": stf["ps_k_bf"],
                "opp_ps_hr_bf": stf["ps_hr_bf"], "opp_ps_h_bf": stf["ps_h_bf"],
                "opp_pc_era": stf["pc_era"],
                "opp_pc_hr_bf": stf["pc_hr_bf"], **park, **wx, **env,
            })
        return pd.DataFrame(rows)

    # starter features the winner model consumes, per side
    _WIN_ST = ["ps_era", "ps_k_bf", "ps_hr_bf", "ps_bb_bf",
               "pc_era", "pc_hr_bf", "p_days_rest", "p5_k_bf"]

    def _win_row(self, spec):
        """One row for the home-win classifier: both teams' records,
        offenses, bullpens, and starters, mirroring build_game_frame."""
        date = pd.Timestamp(spec["date"])
        season = date.year
        s = self.stores
        row = {"Season": season, "month": date.month,
               **s.park(spec["venue"], date), **self._weather(spec)}
        for side, team, starter in (("away", spec["away_team"],
                                     spec["away_starter"]),
                                    ("home", spec["home_team"],
                                     spec["home_starter"])):
            for k, v in s.team_record(team, season, date).items():
                row[f"{side}_{k}"] = v
            for k, v in s.team_form(team, season, date).items():
                row[f"{side}_{k}"] = v
            row[f"{side}_elo"] = s.team_elo(team, season, date)
            off = s.team_offense(team, season, date, prefix="off")
            row[f"{side}_r_pg"] = off["off_r_pg"]
            row[f"{side}_pen_era"] = s.bullpen(team, season, date)["pen_era"]
            stf = s.starter_feats(starter if starter else -1, date, season)
            for k in self._WIN_ST:
                row[f"{side}_{k}"] = stf[k]
        for f in ["win_pct", "rd_pg", "pyth", "ra_pg", "r_pg", "w20", "rd20",
                  "ps_era", "pc_era", "pen_era", "ps_k_bf"]:
            row[f"d_{f}"] = row[f"home_{f}"] - row[f"away_{f}"]
        row["d_rest"] = row["home_p_days_rest"] - row["away_p_days_rest"]
        row["d_ps_kbb"] = ((row["home_ps_k_bf"] - row["home_ps_bb_bf"])
                           - (row["away_ps_k_bf"] - row["away_ps_bb_bf"]))
        row["d_elo"] = row["home_elo"] - row["away_elo"]
        row["elo_prob_home"] = F.elo_expected(row["home_elo"], row["away_elo"])
        return row

    # ------------------------------------------------------- predict

    def _prep(self, df, cols):
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        df = df[cols].copy()
        for c, levels in self.art["cat_levels"].items():
            if c in df.columns:
                df[c] = pd.Categorical(df[c].astype("object"), categories=levels)
        for c in df.columns:
            if c not in self.art["cat_levels"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def predict_game(self, spec):
        a = self.art
        if not spec.get("away_lineup") and not spec.get("home_lineup"):
            raise ValueError("at least one lineup player is required")
        bdf, bmeta = self._batter_rows(spec)
        X = self._prep(bdf, a["bat_cols"])

        batters = bmeta[["Team", "slot", "PlayerId", "Name"]].copy()
        # career-games flag: picks on players with under ~50 MLB games were
        # the weakest segment in the holdout eval (AUC 0.58 vs 0.63 overall)
        batters["CareerG"] = pd.to_numeric(
            bdf["g_career"], errors="coerce").fillna(0).astype(int).to_numpy()
        offs = self.offsets if self.recal else {}
        for prop, col in PROP_COLS.items():
            p = predict_prop(a["props"][prop], X)
            if offs.get(prop):
                p = R.apply_offset(p, offs[prop])
            batters[col] = np.round(p, 4)
        batters["xHR"] = np.round(batters["P_HR"] * a["multi_hr"], 4)
        batters["HR_fair_odds"] = [american_odds(p) for p in batters["P_HR"]]
        batters = batters.sort_values("P_HR", ascending=False).reset_index(drop=True)

        sdf, smeta = self._starter_rows(spec, bdf, bmeta)
        if len(sdf):
            Xs = self._prep(sdf, a["st_cols"])
            k_pred = a["k_model"].predict(Xs)
        else:
            k_pred = np.array([])
        starters = smeta if len(smeta) else pd.DataFrame(
            columns=["PlayerId", "Team", "Opponent", "Name"])
        starters = starters.copy()
        starters["xK"] = np.round(k_pred, 2)
        # K P(over): negative binomial shaded by the calibration-year K
        # dispersion (nb_over falls back to Poisson when k_disp <= ~1)
        k_disp = float(a.get("k_disp", 1.0))
        for line in K_LINES:
            starters[f"P_over_{line}"] = [round(nb_over(l, line, k_disp), 3)
                                          for l in k_pred]

        gdf = self._team_rows(spec)
        Xg = self._prep(gdf, a["tg_cols"])
        mu_away, mu_home = a["team_runs_model"].predict(Xg)
        total_runs = float(mu_away + mu_home)
        # winner: the dedicated home-win model (team strength/Elo/form +
        # both starters, blended with the Poisson win prob); the bare
        # Poisson comparison is only the fallback for old artifacts
        if "win_model" in a:
            Xw = self._prep(pd.DataFrame([self._win_row(spec)]),
                            a["win_model"]["cols"])
            p_home = float(predict_win(a["win_model"], Xw,
                                       mu_home, mu_away)[0])
        else:
            p_home = poisson_win(mu_home, mu_away)
        disp = float(a.get("total_disp", 1.0))
        # Home-win PROBABILITY, not a pick: on the holdout the winner model
        # has no statistically significant edge over always taking the home
        # team, so it's presented as a calibrated number, never a bet.
        totals = {
            "exp_lineup_HR": round(float(batters["xHR"].sum()), 2),
            "exp_away_runs": round(float(mu_away), 2),
            "exp_home_runs": round(float(mu_home), 2),
            "exp_total_runs": round(total_runs, 2),
            "home_team": spec["home_team"], "away_team": spec["away_team"],
            "home_win_prob": round(float(p_home), 3),
            "P_over_runs": {str(l): round(nb_over(total_runs, l, disp), 3)
                            for l in [6.5, 7.5, 8.5, 9.5, 10.5]},
        }
        return {"batters": batters, "starters": starters, "totals": totals}

    def predict_slate(self, specs):
        """Predict several games; batter/starter boards are combined and the
        HR board is ranked across the whole slate."""
        all_b, all_s, games = [], [], []
        for spec in specs:
            out = self.predict_game(spec)
            tag = f'{spec["away_team"]}@{spec["home_team"]}'
            b = out["batters"].copy()
            b.insert(0, "Game", tag)
            s = out["starters"].copy()
            s.insert(0, "Game", tag)
            t = out["totals"]
            p_home = t["home_win_prob"]
            fav_home = p_home >= 0.5
            games.append({"Game": tag, "Date": spec["date"],
                          "Venue": spec["venue"],
                          "Winner": t["home_team"] if fav_home else t["away_team"],
                          "WinProb": p_home if fav_home else round(1 - p_home, 3),
                          "HomeWinProb": t["home_win_prob"],
                          "exp_away_runs": t["exp_away_runs"],
                          "exp_home_runs": t["exp_home_runs"],
                          "exp_lineup_HR": t["exp_lineup_HR"],
                          "exp_total_runs": t["exp_total_runs"],
                          **{f"P_runs_over_{k}": v
                             for k, v in t["P_over_runs"].items()}})
            all_b.append(b)
            all_s.append(s)
        return {
            "batters": pd.concat(all_b).sort_values(
                "P_HR", ascending=False).reset_index(drop=True),
            "starters": pd.concat(all_s).reset_index(drop=True),
            "games": pd.DataFrame(games),
        }


# -------------------------------------------------------------- reporting

PRED_DIR = Path(__file__).resolve().parents[1] / "Predictions"

GLOSSARY = [
    ("How to read this workbook",
     "Every number is the model's estimated chance that something happens "
     "in this game. 25% means: in 100 similar situations, expect it about "
     "25 times. Nothing is ever certain."),
    ("P_HR", "Chance the batter hits a home run in this game."),
    ("HR_fair_odds", "The break-even sportsbook price for that HR chance. "
     "If a book offers longer odds (bigger + number), the bet pays more "
     "than the risk suggests; shorter odds pay less."),
    ("xHR", "Expected number of home runs (accounts for multi-HR games)."),
    ("P_Hit", "Chance of at least one hit."),
    ("P_2Hits", "Chance of two or more hits."),
    ("P_TB2", "Chance of two or more total bases (e.g. a double, or two "
     "singles)."),
    ("P_Run", "Chance the batter scores a run."),
    ("P_RBI", "Chance the batter drives in at least one run."),
    ("P_BB", "Chance of at least one walk."),
    ("P_SB", "Chance of at least one stolen base."),
    ("CareerG", "The batter's career MLB games before today. Predictions "
     "for players under ~50 games are the least reliable (the model has "
     "little history to work from) - treat their picks with extra caution."),
    ("xK", "Projected strikeouts for the starting pitcher."),
    ("P_over_X", "Chance the starter records more than X strikeouts."),
    ("exp_lineup_HR", "Expected total home runs by the players entered."),
    ("exp_total_runs", "Expected combined runs scored by both teams."),
    ("Winner / WinProb", "The team the model favors and its win "
     "probability (always the bigger side of the home/away split). Treat it "
     "as a probability, not a pick: on a half-season holdout the winner "
     "model shows no statistically significant edge over always taking the "
     "home team."),
    ("HomeWinProb", "The same probability expressed from the home team's "
     "side, for reference."),
    ("Slot", "Batting-order position (1 = leadoff)."),
    ("A caution", "These are model estimates from historical data. They do "
     "not account for injuries, breaking news, or the sportsbook's own "
     "information. Bet responsibly."),
]


def _top_str(df, col, n=3, extra=None):
    rows = df.nlargest(n, col)
    parts = []
    for _, r in rows.iterrows():
        s = f'{r["Name"]} ({r[col]:.0%}'
        if extra:
            s += f', fair odds {r[extra]}'
        parts.append(s + ")")
    return ";  ".join(parts)


def summary_frame(specs, out):
    """Plain-English 'what the model thinks' sheet, one block per game."""
    lines = []
    b_all, s_all = out["batters"], out["starters"]
    for i, spec in enumerate(specs):
        tag = f'{spec["away_team"]}@{spec["home_team"]}'
        b = b_all[b_all["Game"] == tag] if "Game" in b_all.columns else b_all
        s = s_all[s_all["Game"] == tag] if "Game" in s_all.columns else s_all
        t = _slate_totals(out, i) if "games" in out else out["totals"]
        lines += [
            (f'{spec["away_team"]} @ {spec["home_team"]}',
             f'{spec["date"]} — {spec["venue"] or "stadium TBD"}'),
            ("Most likely to homer", _top_str(b, "P_HR", 3, "HR_fair_odds")),
            ("Most likely to get a hit", _top_str(b, "P_Hit")),
            ("Most likely to score a run", _top_str(b, "P_Run")),
            ("Most likely to drive in a run", _top_str(b, "P_RBI")),
            ("Most likely to walk", _top_str(b, "P_BB")),
            ("Stolen base watch", _top_str(b, "P_SB")),
        ]
        for _, r in s.iterrows():
            lines.append((f'{r["Name"]} strikeouts',
                          f'projected {r["xK"]:.1f} — over 4.5: '
                          f'{r["P_over_4.5"]:.0%}, over 5.5: '
                          f'{r["P_over_5.5"]:.0%}, over 6.5: '
                          f'{r["P_over_6.5"]:.0%}'))
        hwp = t.get("home_win_prob", 0)
        lines.append(("Home win probability",
                      f'{spec["home_team"]} {hwp:.0%} vs '
                      f'{spec["away_team"]} {1 - hwp:.0%} — a calibrated '
                      f'estimate, not a pick (no proven edge over always '
                      f'taking the home team). Expected score '
                      f'{spec["away_team"]} {t.get("exp_away_runs", "?")}, '
                      f'{spec["home_team"]} {t.get("exp_home_runs", "?")}'))
        lines.append(("Game total",
                      f'expect about {t["exp_total_runs"]} combined runs '
                      f'(over 8.5 runs: {t["P_over_runs"]["8.5"]:.0%}) and '
                      f'{t["exp_lineup_HR"]} home runs from these lineups'))
        lines.append(("", ""))
    if "Game" in b_all.columns and len(specs) > 1:
        top = b_all.nlargest(min(15, len(b_all)), "P_HR")
        counts = top["Game"].value_counts()
        big_game, big_n = counts.index[0], int(counts.iloc[0])
        # effective independent picks if same-game picks are treated as one
        eff = int(round(counts.shape[0]))
        lines.append(("Slate exposure (HR board)",
                      f"the top {len(top)} HR picks span {counts.shape[0]} of "
                      f"{len(specs)} games; {big_game} supplies {big_n} of them. "
                      f"Same-game picks share park, weather and pitcher — they "
                      f"win and lose together, so size a game's picks as roughly "
                      f"ONE bet (~{eff} independent edges here, not {len(top)}) "
                      f"and cap combined stake per game."))
    return pd.DataFrame(lines, columns=["What", "The model says"])


def _polish(path):
    """Make the workbook readable: bold frozen navy headers, percent formats,
    autofit column widths, thin borders around the whole data block, a light
    zebra stripe, and filter/sort dropdowns on the data sheets."""
    import openpyxl
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    wb = openpyxl.load_workbook(path)
    head_font = Font(bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="041E42")
    stripe_fill = PatternFill("solid", fgColor="EAF0F8")
    thin = Side(style="thin", color="B7C2D4")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    # left-aligned text columns; everything else centers
    text_cols = {"Team", "Name", "Game", "Venue", "Winner",
                 "Predicted winner", "Expected score", "What",
                 "The model says", "Field", "Value", "Term", "Meaning"}

    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.sheet_view.showGridLines = False
        if ws.title in ("Batter Props", "Starter Ks", "Games"):
            ws.auto_filter.ref = ws.dimensions  # sort/filter dropdowns
        max_row, max_col = ws.max_row, ws.max_column

        headers = [str(c.value) for c in ws[1]]
        for j, cell in enumerate(ws[1], start=1):
            cell.font = head_font
            cell.fill = head_fill
            cell.alignment = center if headers[j - 1] not in text_cols else left
            cell.border = border

        for j, h in enumerate(headers, start=1):
            is_text = h in text_cols
            for i in range(2, max_row + 1):
                c = ws.cell(row=i, column=j)
                c.border = border
                c.alignment = left if is_text else center
                if h.startswith("P_") or h in ("WinProb", "HomeWinProb"):
                    c.number_format = "0.0%"
                if i % 2 == 1:
                    c.fill = stripe_fill

        # autofit: widest of header / any cell, clamped; percents count as ~6
        for j, h in enumerate(headers, start=1):
            longest = len(h)
            for i in range(2, max_row + 1):
                v = ws.cell(row=i, column=j).value
                if v is None:
                    continue
                longest = max(longest, 6 if isinstance(v, float) else len(str(v)))
            ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = \
                min(max(longest + 2, 6), 90)
    wb.save(path)


def default_excel_path(spec):
    PRED_DIR.mkdir(exist_ok=True)
    import time as _t
    stamp = _t.strftime("%H%M%S")
    name = f'{spec["date"]}_{spec["away_team"]}_at_{spec["home_team"]}_{stamp}.xlsx'
    return PRED_DIR / name


def save_excel(spec, out, path=None):
    """Write one workbook per prediction run: game info, HR board, starters."""
    path = Path(path) if path else default_excel_path(spec)
    t = out["totals"]
    p_home = t.get("home_win_prob", 0.5)
    fav = spec["home_team"] if p_home >= 0.5 else spec["away_team"]
    info_rows = [
        ("Matchup", f'{spec["away_team"]} @ {spec["home_team"]}'),
        ("Date", spec["date"]), ("Stadium", spec["venue"]),
        ("Day/Night", spec["day_night"]), ("Temp (F)", spec["temp"]),
        ("Wind (mph)", spec["wind_speed"]), ("Wind dir", spec["wind_dir"]),
        ("Condition", spec["condition"]),
        ("Away starter", spec["away_starter"]),
        ("Home starter", spec["home_starter"]),
        ("Expected winner", f'{fav} {max(p_home, 1 - p_home):.0%} '
         f'(calibrated estimate, not a pick)'),
        ("Expected score",
         f'{spec["away_team"]} {t.get("exp_away_runs", "?")} — '
         f'{spec["home_team"]} {t.get("exp_home_runs", "?")}'),
        ("Expected lineup HRs", t["exp_lineup_HR"]),
        ("Expected total runs", t["exp_total_runs"]),
    ] + [(f"P(runs over {k})", v) for k, v in t["P_over_runs"].items()]
    info = pd.DataFrame(info_rows, columns=["Field", "Value"])
    summary = summary_frame([spec], out)
    batters = out["batters"].sort_values("P_HR", ascending=False)
    starters = out["starters"].sort_values("xK", ascending=False)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        batters.to_excel(xw, sheet_name="Batter Props", index=False)
        starters.to_excel(xw, sheet_name="Starter Ks", index=False)
        info.to_excel(xw, sheet_name="Game", index=False)
        summary.to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(GLOSSARY, columns=["Term", "Meaning"]).to_excel(
            xw, sheet_name="Glossary", index=False)
    _polish(path)
    return path


def _single_from_slate(out, i, tag):
    b, s = out["batters"], out["starters"]
    return {"batters": b[b["Game"] == tag].drop(columns=["Game"])
            .reset_index(drop=True),
            "starters": s[s["Game"] == tag].drop(columns=["Game"])
            .reset_index(drop=True),
            "totals": _slate_totals(out, i)}


def save_excel_slate(specs, out, path=None):
    """Combined slate workbook: Summary, per-game totals, cross-game boards."""
    if len(specs) == 1:
        return save_excel(specs[0], _single_from_slate(
            out, 0, f'{specs[0]["away_team"]}@{specs[0]["home_team"]}'), path)
    if path is None:
        PRED_DIR.mkdir(exist_ok=True)
        import time as _t
        date = min(s["date"] for s in specs)
        path = PRED_DIR / (f'{date}_slate_{len(specs)}games_'
                           f'{_t.strftime("%H%M%S")}.xlsx')
    batters = out["batters"].sort_values("P_HR", ascending=False)
    starters = out["starters"].sort_values("xK", ascending=False)
    games = out["games"].sort_values("WinProb", ascending=False)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        batters.to_excel(xw, sheet_name="Batter Props", index=False)
        starters.to_excel(xw, sheet_name="Starter Ks", index=False)
        games.to_excel(xw, sheet_name="Games", index=False)
        summary_frame(specs, out).to_excel(xw, sheet_name="Summary", index=False)
        pd.DataFrame(GLOSSARY, columns=["Term", "Meaning"]).to_excel(
            xw, sheet_name="Glossary", index=False)
    _polish(path)
    return Path(path)


def save_excel_per_game(specs, out, folder=None):
    """One workbook per game of a slate. Returns the list of paths."""
    if folder is None:
        PRED_DIR.mkdir(exist_ok=True)
        folder = PRED_DIR
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    paths = []
    for i, spec in enumerate(specs):
        tag = f'{spec["away_team"]}@{spec["home_team"]}'
        p = folder / (f'{spec["date"]}_{spec["away_team"]}_at_'
                      f'{spec["home_team"]}.xlsx')
        n = 2
        while p in paths or (p.exists() and p not in paths):
            p = folder / (f'{spec["date"]}_{spec["away_team"]}_at_'
                          f'{spec["home_team"]}_{n}.xlsx')
            n += 1
        save_excel(spec, _single_from_slate(out, i, tag), p)
        paths.append(p)
    return paths


def _slate_totals(out, i):
    g = out["games"].iloc[i]
    return {"exp_lineup_HR": g["exp_lineup_HR"],
            "exp_total_runs": g["exp_total_runs"],
            "exp_away_runs": g["exp_away_runs"],
            "exp_home_runs": g["exp_home_runs"],
            "home_win_prob": g["HomeWinProb"],
            "P_over_runs": {k.replace("P_runs_over_", ""): g[k]
                            for k in out["games"].columns
                            if k.startswith("P_runs_over_")}}


# ------------------------------------------------------------ replay/test


def spec_from_game(stores, gamepk):
    """Rebuild the input spec of a real game from the data (for replay/tests)."""
    r = stores.raw
    g = r["games"][r["games"]["GamePk"] == gamepk].iloc[0]
    gb = r["gb"][r["gb"]["GamePk"] == gamepk]
    gp = r["gp"][(r["gp"]["GamePk"] == gamepk) & (r["gp"]["GS"] == 1)]

    def lineup(team):
        rows = gb[(gb["Team"] == team) & gb["BattingOrder"].notna()].copy()
        rows["bo"] = pd.to_numeric(rows["BattingOrder"], errors="coerce")
        rows = rows[rows["bo"] % 100 == 0].sort_values("bo")
        return [(int(p), int(b // 100)) for p, b in zip(rows["PlayerId"], rows["bo"])]

    def starter(team):
        m = gp[gp["Team"] == team]
        return int(m["PlayerId"].iloc[0]) if len(m) else None

    return {
        "date": str(g["Date"].date()), "away_team": g["AwayTeam"],
        "home_team": g["HomeTeam"], "venue": g["Venue"],
        "day_night": g["DayNight"], "temp": g["Temp"],
        "wind_speed": g["WindSpeed"], "wind_dir": g["WindDir"],
        "condition": g["Condition"],
        "away_starter": starter(g["AwayTeam"]),
        "home_starter": starter(g["HomeTeam"]),
        "away_lineup": lineup(g["AwayTeam"]),
        "home_lineup": lineup(g["HomeTeam"]),
    }


def _compare_row(train_row, serve_row, cols, label, worst, counter):
    """Fold one train-vs-serve row comparison into (worst, n_checked)."""
    for c in cols:
        a = train_row.get(c, np.nan)
        b = serve_row.get(c, np.nan)
        a = np.nan if pd.isna(a) else float(a)
        b = np.nan if pd.isna(b) else float(b)
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) != np.isnan(b):
            print(f"  NaN mismatch {label} {c}: train={a} serve={b}")
            continue
        d = abs(a - b) / max(1e-9, abs(a))
        counter += 1
        if d > worst[1]:
            worst = (f"{label} {c} (train={a:.6g} serve={b:.6g})", d)
    return worst, counter


def selftest(pred):
    """Compare inference-path features against the training frames for a
    real 2026 game: they must match, or training and serving have drifted.
    Covers the batter rows, the starter (K-model) rows, and (when the
    artifact has one) the winner row."""
    frames = joblib.load(ART / "frames.joblib")
    bf = frames["bf"]
    b26 = bf[(bf["Season"] == 2026) & bf["StarterId"].notna()]
    gamepk = int(b26["GamePk"].iloc[-1])
    spec = spec_from_game(pred.stores, gamepk)
    print(f"selftest on GamePk {gamepk}: {spec['away_team']} @ "
          f"{spec['home_team']} {spec['date']}")

    bdf, bmeta = pred._batter_rows(spec)
    check_cols = [c for c in pred.art["bat_cols"] if c not in F.CAT_COLS]
    worst = ("", 0.0)
    n_checked = 0
    for i, m in bmeta.iterrows():
        trow = bf[(bf["GamePk"] == gamepk) & (bf["PlayerId"] == m["PlayerId"])]
        if trow.empty:
            continue
        worst, n_checked = _compare_row(trow.iloc[0], bdf.iloc[i], check_cols,
                                        m["Name"], worst, n_checked)

    if "sf" in frames:
        sf = frames["sf"]
        sdf, smeta = pred._starter_rows(spec, bdf, bmeta)
        st_cols = [c for c in pred.art["st_cols"] if c not in F.CAT_COLS]
        for i, m in smeta.iterrows():
            trow = sf[(sf["GamePk"] == gamepk)
                      & (sf["PlayerId"] == m["PlayerId"])]
            if trow.empty:
                continue
            worst, n_checked = _compare_row(trow.iloc[0], sdf.iloc[i], st_cols,
                                            f'K:{m["Name"]}', worst, n_checked)

    if "win_model" in pred.art and "gf" in frames:
        grow = frames["gf"][frames["gf"]["GamePk"] == gamepk]
        if not grow.empty:
            wcols = [c for c in pred.art["win_model"]["cols"]
                     if c not in F.CAT_COLS]
            worst, n_checked = _compare_row(grow.iloc[0], pred._win_row(spec),
                                            wcols, "win-row", worst, n_checked)

    print(f"  compared {n_checked} feature values; "
          f"worst relative diff: {worst[1]:.2e} [{worst[0]}]")
    ok = worst[1] < 1e-6
    print("  PARITY OK" if ok else "  PARITY FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, help="replay a historical GamePk")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--recal", action="store_true",
                    help="apply stored in-season drift offsets to batter props")
    args = ap.parse_args()

    pred = Predictor(recal=args.recal)
    if args.selftest:
        sys.exit(0 if selftest(pred) else 1)

    if args.game:
        spec = spec_from_game(pred.stores, args.game)
        out = pred.predict_game(spec)
        print(f"\n{spec['away_team']} @ {spec['home_team']}  {spec['date']}  "
              f"{spec['venue']}  {spec['temp']}F wind {spec['wind_speed']} "
              f"{spec['wind_dir']}\n")
        print("=== Home run probabilities (lineup, ranked) ===")
        print(out["batters"].to_string(index=False))
        print("\n=== Starter strikeouts ===")
        print(out["starters"].to_string(index=False))
        print("\n=== Game totals ===")
        for k, v in out["totals"].items():
            print(f"  {k}: {v}")
        path = save_excel(spec, out)
        print(f"\nsaved workbook: {path}")


if __name__ == "__main__":
    main()
