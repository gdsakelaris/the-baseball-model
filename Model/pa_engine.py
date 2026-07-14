"""PA-sim Phase 2, engine — Monte-Carlo game simulation on top of the
Phase-1 PA outcome model (pa_model.joblib) and the pa_sim data tables.

  MatchupFeatures  as-of feature rows for ANY (batter, pitcher, date) —
                   the same 3-level EB rates the PA model trained on,
                   queried by merge_asof so any game date works; the team
                   bullpen enters as a pseudo-pitcher from pen_rates.
  GameSim          vectorized game simulator: N sims of one game in numpy.
                   State machine over (slot, runner IDENTITIES on bases,
                   outs, score); outcomes sampled from the PA model's
                   matchup distributions, advancement from the empirical
                   transition tables. Runner identity -> per-player runs /
                   RBI / pitcher ER attribute exactly (FIFO advancement:
                   lead runners score first). Starter hook = BF sampled
                   from the pitcher's own as-of start history; walk-offs;
                   extra innings use the ghost-runner rule, hard cap 20.

v1 simplifications (upgrade path, not dogma): PA class probabilities are
matchup-conditioned but base-out-NEUTRAL (bases empty / 1 out; inning 3 vs
starter, 7 vs pen) — game state affects advancement, not the class mix; one
aggregate pen pseudo-pitcher per team (league-prior contact quality at
n=0 — no pitcher identity is exactly what the EB shrink encodes); 9
starters bat the whole game; steals are runner-side only (1B->2B, one
Bernoulli per PA at the runner's as-of EB rate — league-average catcher/
pitcher hold, no 2B->3B); all sim runs are earned.
"""

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pa_model import (CLASSES, EB_K, FRAME_CACHE, _asof_counts, _cq_shrink,
                      _cq_tables, _eb, _league_trailing, feature_cols)
from milb_priors import build_all as milb_build, prior_blend

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

CI = {c: i for i, c in enumerate(CLASSES)}
HIT_IDS = np.array([CI["1B"], CI["2B"], CI["3B"], CI["HR"]])
# The steal tables measure attempts per time-ON-first; the sim rolls one
# Bernoulli per PA while a runner stands on 1B with 2B open — a window
# that usually closes within a single PA (forced up, erased, or blocked),
# so the per-PA hazard must RAISE the per-reach rate slightly. Calibrated
# so a league-average synthetic game reproduces league SB/team/game
# (~0.75 in the 2024-25 pitch-clock era) — see the steal smoke.
STEAL_ATT_DIV = 0.8
TB_OF = np.zeros(8, np.int8)
for c, tb in (("1B", 1), ("2B", 2), ("3B", 3), ("HR", 4)):
    TB_OF[CI[c]] = tb


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------- matchup features -----

class MatchupFeatures:
    """As-of PA-model feature rows for arbitrary matchups on any date."""

    def __init__(self, pa=None):
        if pa is None:
            pa = pd.read_parquet(FRAME_CACHE)
        self.lg = _league_trailing(pa)                    # Date -> lg_ mix
        self.park_levels = sorted(pa["park"].astype(str).unique())
        self.tabs = {}
        for pre, keys in (("b", ["batter"]), ("bh", ["batter", "p_throws"]),
                          ("bs", ["batter", "Season"]),
                          ("p", ["pitcher"]), ("ph", ["pitcher", "stand"]),
                          ("ps", ["pitcher", "Season"])):
            t = _asof_counts(pa, keys).sort_values("Date", kind="stable")
            self.tabs[pre] = (keys, t)
        self.cq_tabs, self.lg_brl, self.lg_ev = _cq_tables(pa)
        self.milb = {}
        milb = milb_build()
        for kind in ("bat", "pit"):
            sv = milb[kind]["serve"]
            keys = sv[["PlayerId", "Season"]].to_numpy(int)
            vals = sv[[f"t_{c}" for c in CLASSES]].to_numpy(float)
            nes = sv["n_eff"].to_numpy(float)
            self.milb[kind] = {(int(k[0]), int(k[1])): (vals[i],
                                                        float(nes[i]))
                               for i, k in enumerate(keys)}
        hands = (pa.sort_values("Date")
                   .groupby("batter")["stand"].last().to_dict())
        throws = (pa.sort_values("Date")
                    .groupby("pitcher")["p_throws"].last().to_dict())
        self.bat_hand, self.pit_hand = hands, throws

    def _asof(self, pre, key_vals, date):
        """Last day-start row at or before `date` for the exact keys."""
        keys, t = self.tabs[pre]
        m = np.ones(len(t), bool)
        for k, v in zip(keys, key_vals):
            m &= (t[k] == v).to_numpy()
        sub = t[m & (t["Date"] <= date)]
        if len(sub) == 0:
            return np.zeros(len(CLASSES)), 0.0
        r = sub.iloc[-1]
        return r[CLASSES].to_numpy(dtype=float), float(r["n"])

    def _milb_prior(self, kind, pid, season, lg):
        """Career-prior row (league blended toward the player's translated
        MiLB line) + the log1p n_eff feature."""
        t, ne = self.milb[kind].get((int(pid), int(season)), (None, 0.0))
        tt = t[None] if t is not None else np.full((1, len(CLASSES)), np.nan)
        p0 = prior_blend(lg[None], tt, [ne])[0]
        return dict(zip(CLASSES, p0)), float(np.log1p(ne))

    def _asof_cq(self, pre, key, date):
        """Last day-start cumulative (brl, ev, n) at or before `date`."""
        t = self.cq_tabs[pre]
        kcol = "batter" if pre == "b" else "pitcher"
        sub = t[(t[kcol] == key).to_numpy() & (t["Date"] <= date).to_numpy()]
        if len(sub) == 0:
            return 0.0, 0.0, 0.0
        r = sub.iloc[-1]
        return float(r["brl"]), float(r["ev"]), float(r["n"])

    def rows(self, date, season, park, bat_home, batters, stands,
             pitcher, p_throws, pen_rates_row, vs_pen):
        """One feature row per batter vs either the starter (`vs_pen=False`)
        or the team-pen pseudo-pitcher. Returns df[feature_cols()]."""
        date = pd.Timestamp(date)
        lg_row = self.lg[self.lg.index <= date].iloc[-1]
        lg = lg_row.to_numpy(dtype=float)
        lg_ser = {c: lg[i] for i, c in enumerate(CLASSES)}
        lgb = self.lg_brl[self.lg_brl.index <= date]
        lge = self.lg_ev[self.lg_ev.index <= date]
        lg_b = float(lgb.iloc[-1]) if len(lgb) else np.nan
        lg_e = float(lge.iloc[-1]) if len(lge) else np.nan

        if vs_pen:
            pr = pen_rates_row[[f"pen_{c}" for c in CLASSES]].to_numpy(float)
            p_rates = ph_rates = ps_rates = pr
            p_n = ph_n = ps_n = float(pen_rates_row["pen_n"])
            p_throws = "R"                       # pen aggregate: neutral tag
            p_cq = _cq_shrink(0.0, 0.0, 0.0, lg_b, lg_e)   # league prior n=0
            p_mn = 0.0                           # pen has no MiLB identity
        else:
            p_prior, p_mn = self._milb_prior("pit", pitcher, season, lg)
            cnt, p_n = self._asof("p", (pitcher,), date)
            p_rates = _eb(pd.DataFrame([dict(zip(CLASSES, cnt))]),
                          pd.Series([p_n]), pd.DataFrame([p_prior]),
                          EB_K).iloc[0].to_numpy()
            ps_cnt, ps_n = self._asof("ps", (pitcher, season), date)
            ps_rates = _eb(pd.DataFrame([dict(zip(CLASSES, ps_cnt))]),
                           pd.Series([ps_n]),
                           pd.DataFrame([dict(zip(CLASSES, p_rates))]),
                           EB_K).iloc[0].to_numpy()
            p_cq = _cq_shrink(*self._asof_cq("p", pitcher, date),
                              lg_b, lg_e)

        out = []
        for bat, stand in zip(batters, stands):
            b_prior, b_mn = self._milb_prior("bat", bat, season, lg)
            cnt, b_n = self._asof("b", (bat,), date)
            b_rates = _eb(pd.DataFrame([dict(zip(CLASSES, cnt))]),
                          pd.Series([b_n]), pd.DataFrame([b_prior]),
                          EB_K).iloc[0].to_numpy()
            bh_cnt, bh_n = self._asof("bh", (bat, p_throws), date)
            k2 = {c: 2 * v for c, v in EB_K.items()}
            bh_rates = _eb(pd.DataFrame([dict(zip(CLASSES, bh_cnt))]),
                           pd.Series([bh_n]),
                           pd.DataFrame([dict(zip(CLASSES, b_rates))]),
                           k2).iloc[0].to_numpy()
            bs_cnt, bs_n = self._asof("bs", (bat, season), date)
            bs_rates = _eb(pd.DataFrame([dict(zip(CLASSES, bs_cnt))]),
                           pd.Series([bs_n]),
                           pd.DataFrame([dict(zip(CLASSES, b_rates))]),
                           EB_K).iloc[0].to_numpy()
            if vs_pen:
                ph_rates2, ph_n2 = p_rates, p_n
            else:
                ph_cnt, ph_n2 = self._asof("ph", (pitcher, stand), date)
                ph_rates2 = _eb(pd.DataFrame([dict(zip(CLASSES, ph_cnt))]),
                                pd.Series([ph_n2]),
                                pd.DataFrame([dict(zip(CLASSES, p_rates))]),
                                k2).iloc[0].to_numpy()
            row = {}
            for pre, rates, n in (("b", b_rates, b_n), ("bh", bh_rates, bh_n),
                                  ("bs", bs_rates, bs_n),
                                  ("p", p_rates, p_n),
                                  ("ph", ph_rates2, ph_n2),
                                  ("ps", ps_rates, ps_n)):
                for i, c in enumerate(CLASSES):
                    row[f"{pre}_{c}"] = rates[i]
                row[f"{pre}_n"] = np.log1p(n)
            for i, c in enumerate(CLASSES):
                row[f"lg_{c}"] = lg[i]
            b_cq = _cq_shrink(*self._asof_cq("b", bat, date), lg_b, lg_e)
            for pre, cq in (("b", b_cq), ("p", p_cq)):
                for suf, v in zip(("brl", "ev", "bip"), cq):
                    row[f"{pre}_{suf}"] = float(v)
            row["b_milb_n"], row["p_milb_n"] = b_mn, p_mn
            row.update(same_hand=int(stand == p_throws), bat_home=bat_home,
                       outs_when_up=1, inning=7 if vs_pen else 3,
                       on1b=0, on2b=0, on3b=0, park=park, stand=stand,
                       p_throws=p_throws)
            out.append(row)
        df = pd.DataFrame(out)
        df["park"] = pd.Categorical(df["park"].astype(str),
                                    categories=self.park_levels)
        for c in ("stand", "p_throws"):
            df[c] = pd.Categorical(df[c].astype(str),
                                   categories=["L", "R", "S"])
        return df[feature_cols()]


# --------------------------------------------- packed transitions -------

class PackedTransitions:
    """(cls, bases, outs) -> padded arrays for vectorized sampling."""

    def __init__(self, tr):
        cells, fb, fbc = tr["cells"], tr["fallback"], tr["fallback_cls"]
        K = max(len(v["p"]) for v in cells.values())
        n_cell = len(CLASSES) * 8 * 3
        self.cum = np.zeros((n_cell, K))
        self.d_outs = np.zeros((n_cell, K), np.int8)
        self.runs = np.zeros((n_cell, K), np.int8)
        self.nb = np.zeros((n_cell, K), np.int8)
        for ci in range(len(CLASSES)):
            for b in range(8):
                for o in range(3):
                    cell = cells.get((ci, b, o))
                    if cell is None:             # thin cell: nearest filled
                        for cand in [(ci, b, oo) for oo in (o - 1, o + 1)
                                     if 0 <= oo <= 2] + [(ci, 0, o)]:
                            cell = cells.get(cand)
                            if cell is not None:
                                break
                    if cell is None:             # class-level last resort
                        f = fb.get((ci, o)) or fbc[ci]
                        cell = {"p": f["p"], "d_outs": f["d_outs"],
                                "runs": f["runs"],
                                "next_bases": np.full(len(f["p"]), b,
                                                      np.int8)}
                    idx = (ci * 8 + b) * 3 + o
                    k = len(cell["p"])
                    self.cum[idx, :k] = np.cumsum(cell["p"])
                    self.cum[idx, k:] = 1.0
                    self.d_outs[idx, :k] = cell["d_outs"]
                    self.runs[idx, :k] = cell["runs"]
                    self.nb[idx, :k] = cell["next_bases"]

    def sample(self, cls, bases, outs, rng):
        idx = (cls * 8 + bases) * 3 + outs
        u = rng.random(len(idx))
        j = (u[:, None] > self.cum[idx]).sum(axis=1)
        return (self.d_outs[idx, j], self.runs[idx, j], self.nb[idx, j])


# ------------------------------------------------------- game sim -------

class GameSim:
    """N Monte-Carlo runs of one game. probs_*: [9, 8] class distributions
    per lineup slot (vs starter / vs pen), per side; bf_*: arrays of
    plausible starter BF draws (sampled with replacement)."""

    def __init__(self, packed, probs, bf_draws, n_sims=2000, seed=0,
                 steal=None, hazard=None):
        self.T = packed
        self.probs = probs        # {"away"|"home": {"st": [9,8], "pen": [9,8]}}
        self.bf = bf_draws        # {"away"|"home": np.array of BF samples}
        self.steal = steal        # {"away"|"home": (att[9], succ[9])} or None
        # hazard: {"away"|"home": hz[bf, runs]} — that team's OWN starter's
        # tier slice of the hook table. When given, the starter is hooked
        # by in-game state (v2); when None, legacy sampled-BF countdown.
        self.hazard = hazard
        self.n = n_sims
        self.rng = np.random.default_rng(seed)

    def run(self):
        n, rng, T = self.n, self.rng, self.T
        cum = {s: {k: np.cumsum(v, axis=1) for k, v in d.items()}
               for s, d in self.probs.items()}
        score = {"away": np.zeros(n, int), "home": np.zeros(n, int)}
        slot = {"away": np.zeros(n, np.int8), "home": np.zeros(n, np.int8)}
        use_hz = self.hazard is not None
        hooked = {s: np.zeros(n, bool) for s in ("away", "home")}
        if not use_hz:
            bf_left = {s: self.bf[s][rng.integers(0, len(self.bf[s]), n)]
                       .astype(np.int16) for s in ("away", "home")}
        # per-player per-sim stat blocks [9, n]
        z = lambda: {s: np.zeros((9, n), np.int16) for s in ("away", "home")}
        stats = {k: z() for k in
                 ("pa", "k", "bb", "h1", "h2", "h3", "hr", "tb", "r", "rbi",
                  "sb")}
        stt = lambda: {s: np.zeros(n, np.int16) for s in ("away", "home")}
        st = {k: stt() for k in ("k", "outs", "h", "bb", "er", "bf")}

        done = np.zeros(n, bool)
        for inning in range(1, 21):
            for side in ("away", "home"):
                live = ~done
                # bottom of 9th+ skipped where home already leads
                if inning >= 9 and side == "home":
                    live &= ~(score["home"] > score["away"])
                if not live.any():
                    continue
                outs = np.zeros(n, np.int8)
                bases = np.full((n, 3), -1, np.int8)       # slot ids
                base_pen = np.zeros((n, 3), bool)          # on vs pen?
                if inning >= 10:                           # ghost runner
                    ghost = (slot[side] - 1) % 9
                    bases[live, 1] = ghost[live]
                    base_pen[live, 1] = False              # uncharged run ok
                fld = "home" if side == "away" else "away"
                active = live & (outs < 3)
                while active.any():
                    a = np.flatnonzero(active)
                    # ---- steal layer: runner on 1B, 2B open, one Bernoulli
                    # per PA at the runner's per-reach rate / STEAL_ATT_DIV;
                    # success moves him up, CS removes him + charges the out
                    # to whoever is pitching. Ghost runners (2B) never steal.
                    if self.steal is not None:
                        att_p, succ_p = self.steal[side]
                        can = (bases[a, 0] >= 0) & (bases[a, 1] < 0)
                        if can.any():
                            r_sl = np.where(can, bases[a, 0], 0)
                            attempt = can & (rng.random(len(a))
                                             < att_p[r_sl] / STEAL_ATT_DIV)
                            succ = attempt & (rng.random(len(a))
                                              < succ_p[r_sl])
                            fail = attempt & ~succ
                            if succ.any():
                                si = a[succ]
                                np.add.at(stats["sb"][side],
                                          (r_sl[succ], si), 1)
                                bases[si, 1] = bases[si, 0]
                                base_pen[si, 1] = base_pen[si, 0]
                                bases[si, 0] = -1
                                base_pen[si, 0] = False
                            if fail.any():
                                fi = a[fail]
                                bases[fi, 0] = -1
                                base_pen[fi, 0] = False
                                outs[fi] += 1
                                cs_pen = (hooked[fld][fi] if use_hz
                                          else bf_left[fld][fi] <= 0)
                                st["outs"][fld][fi] += ~cs_pen
                                active &= outs < 3
                                a = np.flatnonzero(active)
                                if len(a) == 0:
                                    break
                    sl = slot[side][a]
                    pen_in = (hooked[fld][a] if use_hz
                              else bf_left[fld][a] <= 0)
                    occ = ((bases[a] >= 0)
                           * np.array([1, 2, 4], np.int8)).sum(axis=1)
                    # sample outcome class per active sim
                    u = rng.random(len(a))
                    p_st, p_pen = cum[side]["st"], cum[side]["pen"]
                    cdf = np.where(pen_in[:, None], p_pen[sl], p_st[sl])
                    cls = (u[:, None] > cdf).sum(axis=1).astype(np.int8)
                    d_o, runs, nb = T.sample(cls, occ, outs[a], rng)

                    # cap runs at runners+batter(HR); count scoring runners
                    n_run = (bases[a] >= 0).sum(axis=1)
                    max_r = n_run + (cls == CI["HR"])
                    runs = np.minimum(runs, max_r).astype(np.int8)

                    # per-batter stats
                    st_b = stats
                    np.add.at(st_b["pa"][side], (sl, a), 1)
                    for name, cid in (("k", CI["K"]), ("bb", CI["BB"]),
                                      ("h1", CI["1B"]), ("h2", CI["2B"]),
                                      ("h3", CI["3B"]), ("hr", CI["HR"])):
                        np.add.at(st_b[name][side], (sl, a), cls == cid)
                    np.add.at(st_b["tb"][side], (sl, a), TB_OF[cls])
                    np.add.at(st_b["rbi"][side], (sl, a), runs)

                    # FIFO scoring: lead runners (3rd,2nd,1st) score first,
                    # batter last (HR). Credit runs + charge starter ER.
                    for j, (ai, r) in enumerate(zip(a, runs)):
                        if r == 0:
                            continue
                        chain = [(bases[ai, 2], base_pen[ai, 2]),
                                 (bases[ai, 1], base_pen[ai, 1]),
                                 (bases[ai, 0], base_pen[ai, 0]),
                                 (sl[j], pen_in[j])]
                        scored = [c for c in chain if c[0] >= 0][:r]
                        for who, was_pen in scored:
                            stats["r"][side][who, ai] += 1
                            if not was_pen:
                                st["er"][fld][ai] += 1
                    score[side][a] += runs

                    # rebuild base occupancy: survivors advance FIFO into
                    # the new occupancy pattern (front-most first)
                    for j, ai in enumerate(a):
                        r = runs[j]
                        chain = [bases[ai, 2], bases[ai, 1], bases[ai, 0],
                                 sl[j]]
                        pens = [base_pen[ai, 2], base_pen[ai, 1],
                                base_pen[ai, 0], pen_in[j]]
                        alive = [(w, p) for w, p in zip(chain, pens)
                                 if w >= 0][r:]
                        newb, newp = [-1, -1, -1], [False] * 3
                        want = nb[j]
                        spots = [k for k in (2, 1, 0) if want & (1 << k)]
                        for spot, (w, p) in zip(spots, alive):
                            newb[spot], newp[spot] = w, p
                        bases[ai] = newb
                        base_pen[ai] = newp

                    # starter accounting (before hook check)
                    st["bf"][fld][a] += ~pen_in
                    st["k"][fld][a] += (cls == CI["K"]) & ~pen_in
                    st["bb"][fld][a] += (cls == CI["BB"]) & ~pen_in
                    is_hit = np.isin(cls, HIT_IDS)
                    st["h"][fld][a] += is_hit & ~pen_in
                    st["outs"][fld][a] += d_o * ~pen_in
                    if use_hz:
                        # v2 hook: hazard on the post-PA state (batters
                        # faced, runs allowed) of sims whose starter is in
                        hz = self.hazard[fld]
                        ai = a[~pen_in]
                        if len(ai):
                            bfi = np.minimum(st["bf"][fld][ai],
                                             hz.shape[0] - 1)
                            rni = np.minimum(st["er"][fld][ai],
                                             hz.shape[1] - 1)
                            hook = rng.random(len(ai)) < hz[bfi, rni]
                            hooked[fld][ai[hook]] = True
                    else:
                        bf_left[fld][a] -= ~pen_in

                    outs[a] = np.minimum(outs[a] + d_o, 3)
                    slot[side][a] = (sl + 1) % 9
                    # walk-off: bottom 9+ ends the moment home leads
                    if side == "home" and inning >= 9:
                        active &= ~(score["home"] > score["away"])
                    active &= outs < 3
            if inning >= 9:
                # a completed inning from the 9th on ends every sim whose
                # score is no longer tied (covers walk-offs via the skip
                # rules above)
                done |= score["home"] != score["away"]
                if done.all():
                    break

        return {"stats": stats, "starter": st, "score": score,
                "home_win": (score["home"] > score["away"]).astype(int)}


if __name__ == "__main__":
    log("smoke test: packing transitions + one synthetic game")
    tables = joblib.load(ART / "pa_sim_tables.joblib")
    packed = PackedTransitions(tables["transitions"])
    lg = np.array([.22, .085, .011, .14, .044, .004, .032, .464])
    lg = lg / lg.sum()
    probs = {s: {"st": np.tile(lg, (9, 1)), "pen": np.tile(lg, (9, 1))}
             for s in ("away", "home")}
    bf = {s: np.array([22, 24, 20, 26, 23]) for s in ("away", "home")}
    sim = GameSim(packed, probs, bf, n_sims=4000, seed=1)
    out = sim.run()
    tot = out["score"]["home"] + out["score"]["away"]
    log(f"mean total {tot.mean():.2f} (league ~8.5-9.0) | "
        f"P(home win) {out['home_win'].mean():.3f} (~.50 for equal teams) | "
        f"mean batter-1 hits "
        f"{(out['stats']['h1']['home'][0] + out['stats']['h2']['home'][0] + out['stats']['h3']['home'][0] + out['stats']['hr']['home'][0]).mean():.2f} | "
        f"mean starter K {out['starter']['k']['home'].mean():.2f} "
        f"(league ~5.3) | mean starter ER {out['starter']['er']['home'].mean():.2f}")
