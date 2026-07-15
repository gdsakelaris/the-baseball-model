"""PA-sim live-slate path — Phase 3 serving (2026-07-13).

SlateSim runs the Monte-Carlo game engine for TODAY'S games from the same
spec dict predict.py consumes (posted lineups + starters + venue), using
the identical machinery the shadow backtests were graded on:
MatchupFeatures (as-of EB rates incl. contact quality + MiLB priors),
the steal layer, and the starter-hazard hook. predict.py blends the
game-level outputs (score/total/winner) with the incumbent heads at the
fixed SIM_BLEND weights; batter/starter heads stay incumbent (w=0 per the
2026-07-13 evidence).

Loaded once per Predictor (frame + as-of tables ~10s); each game is two
feature queries + one 2,000-sim run (~1s).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pa_model import FRAME_CACHE
from pa_sim import (STEAL_K_ATT, STEAL_K_SUCC, battery_adjust,
                    battery_context, hazard_slice_v2, starter_exp_bf)
from pa_engine import MatchupFeatures, PackedTransitions, GameSim

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

N_SIMS = 2000


class SlateSim:
    """Game-level sim outputs for live slates."""

    def __init__(self):
        art = joblib.load(ART / "pa_model.joblib")
        self.model, self.cols = art["model"], art["cols"]
        self.tables = joblib.load(ART / "pa_sim_tables.joblib")
        self.packed = PackedTransitions(self.tables["transitions"])
        pa = pd.read_parquet(FRAME_CACHE)
        self.mf = MatchupFeatures(pa)
        self.pen = self.tables["pen_rates"].copy()
        self.pen["Date"] = pd.to_datetime(self.pen["Date"])
        st = self.tables["steals"]
        self.steal_players = (st["players"].sort_values("Date")
                              .groupby("PlayerId"))
        self.steal_lg = st["league"]

    def _pen_row(self, team, date):
        m = self.pen[(self.pen["Team"] == team) & (self.pen["Date"] <= date)]
        if len(m):
            return m.iloc[-1]
        row = {f"pen_{c}": 1 / 8 for c in
               ("K", "BB", "HBP", "1B", "2B", "3B", "HR", "OUT")}
        row["pen_n"] = 0.0
        return pd.Series(row)

    def _steal(self, batters, date):
        lg = self.steal_lg[self.steal_lg.index <= date]
        lg_att = float(lg["lg_att"].iloc[-1]) if len(lg) else 0.06
        lg_succ = float(lg["lg_succ"].iloc[-1]) if len(lg) else 0.78
        att, succ = [], []
        for pid in batters:
            sb = at = on1 = 0.0
            try:
                h = self.steal_players.get_group(int(pid))
                h = h[h["Date"] <= date]
                if len(h):
                    r = h.iloc[-1]
                    sb, at, on1 = float(r["sb"]), float(r["att"]), \
                        float(r["on1"])
            except KeyError:
                pass
            att.append((at + STEAL_K_ATT * lg_att) / (on1 + STEAL_K_ATT))
            succ.append((sb + STEAL_K_SUCC * lg_succ) / (at + STEAL_K_SUCC))
        return np.array(att), np.array(succ)

    def game(self, spec):
        """{'x_away','x_home','x_total','p_home_win'} for one game spec,
        or None when either side lacks a full posted lineup/starter."""
        date = pd.Timestamp(spec["date"]).normalize()
        season = int(date.year)
        park = spec["home_team"]
        lus = {"away": [p for p, _ in spec.get("away_lineup", [])],
               "home": [p for p, _ in spec.get("home_lineup", [])]}
        sts = {"away": spec.get("away_starter"),
               "home": spec.get("home_starter")}
        if any(len(lus[s]) != 9 or not sts[s] for s in ("away", "home")):
            return None

        sides, steal = {}, {}
        for side, home_flag in (("away", 0), ("home", 1)):
            opp = "home" if side == "away" else "away"
            opp_team = spec[f"{opp}_team"]
            pitcher = int(sts[opp])
            p_throws = self.mf.pit_hand.get(pitcher, "R")
            stands = [self.mf.bat_hand.get(int(b), "R") for b in lus[side]]
            pen_row = self._pen_row(opp_team, date)
            common = dict(date=date, season=season, park=park,
                          bat_home=home_flag, batters=lus[side],
                          stands=stands, pitcher=pitcher,
                          p_throws=p_throws, pen_rates_row=pen_row)
            sides[side] = {
                "st": self.model.predict_proba(
                    self.mf.rows(**common, vs_pen=False)[self.cols]),
                "pen": self.model.predict_proba(
                    self.mf.rows(**common, vs_pen=True)[self.cols])}
            # battery modulation (#35): opp_team's battery + starter face
            # this batting side — same helper as the backtest path
            r_att, stp = battery_context(self.tables.get("battery"),
                                         season, opp_team, pitcher)
            steal[side] = battery_adjust(*self._steal(lus[side], date),
                                         ratio=r_att, stop=stp)

        bf_t = self.tables["starter_bf"]
        # hazard v2 (2026-07-14): per-starter absolute-BF slice of the
        # relative-BF hook table (his own leash)
        hz2 = self.tables["starter_hazard_v2"]["hazard"]
        bf = {s: np.array([22, 23, 24], np.int16) for s in ("away", "home")}
        hazard = {"away": hazard_slice_v2(hz2, starter_exp_bf(
                      bf_t, sts["away"], date)),
                  "home": hazard_slice_v2(hz2, starter_exp_bf(
                      bf_t, sts["home"], date))}
        # deterministic across processes (audit #9): builtin hash() is
        # salted per interpreter (PYTHONHASHSEED), which made the same game
        # sim differently in every session — ±1.1pp Monte-Carlo jitter on
        # p(home win) at N_SIMS=2000. crc32 is stable everywhere.
        import zlib
        seed = int(date.value // 10 ** 9) ^ (zlib.crc32(
            f"{spec['away_team']}|{spec['home_team']}".encode())
            & 0x7fffffff)
        sim = GameSim(self.packed, sides, bf, steal=steal, hazard=hazard,
                      n_sims=N_SIMS, seed=seed & 0x7fffffff)
        out = sim.run()
        tot = out["score"]["home"] + out["score"]["away"]
        return {"x_away": float(out["score"]["away"].mean()),
                "x_home": float(out["score"]["home"].mean()),
                "x_total": float(tot.mean()),
                "p_home_win": float(out["home_win"].mean())}
