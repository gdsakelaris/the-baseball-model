"""Unit tests for the money-adjacent pure functions (audit fix #14).

Covers the pricing math (Poisson/NB tails, de-vig, EV, settlement), the
grading/settlement logic that writes the forward record, the parsing
helpers, the calibration map, and the new audit machinery (BH q-values,
the early-stop split, line calibrators). No data files, no artifacts, no
network — runs in seconds.

    python -m unittest discover -s Tests -v
"""
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for sub in ("Model", "Tools", "Scrapers"):
    sys.path.insert(0, str(ROOT / sub))

import features as F                     # noqa: E402
import odds as O                         # noqa: E402
import predict as P                      # noqa: E402

GR = importlib.import_module("4_grade_results")


class TestParsers(unittest.TestCase):
    def test_ip_to_outs(self):
        s = F.ip_to_outs(pd.Series(["5.2", "6.0", "0.1", "7"]))
        self.assertEqual(list(s), [17.0, 18.0, 1.0, 21.0])
        # the grader's scalar twin must agree
        for ip, outs in (("5.2", 17), ("6.0", 18), ("0.1", 1)):
            self.assertEqual(GR.ip_to_outs(ip), outs)

    def test_height_to_inches(self):
        self.assertEqual(F.height_to_inches("6' 2\""), 74)
        self.assertEqual(F.height_to_inches("5' 11\""), 71)
        self.assertTrue(np.isnan(F.height_to_inches("??")))

    def test_grading_regexes(self):
        m = GR.LINE_RE.match("K > 6.5")
        self.assertEqual((m.group(1), m.group(2)), ("K", "6.5"))
        self.assertIsNone(GR.LINE_RE.match("xK"))
        self.assertEqual(GR.RUNS_RE.match("Runs > 8.5").group(1), "8.5")
        m = GR.TEAM_RUNS_RE.match("Away Runs > 3.5")
        self.assertEqual((m.group(1), m.group(2)), ("Away", "3.5"))
        self.assertIsNone(GR.RUNS_RE.match("Away Runs > 3.5"))

    def test_bat_events(self):
        s = pd.Series({"H": 2, "2B": 1, "3B": 0, "HR": 1, "TB": 7, "R": 1,
                       "RBI": 2, "BB": 0, "SO": 3, "SB": 0})
        self.assertFalse(GR.BAT_EVENTS["Single"](s))   # 2 - 1 - 0 - 1 = 0
        self.assertTrue(GR.BAT_EVENTS["Double"](s))
        self.assertFalse(GR.BAT_EVENTS["Triple"](s))
        self.assertTrue(GR.BAT_EVENTS["4+ TB"](s))
        self.assertTrue(GR.BAT_EVENTS["H+R+RBI 4+"](s))   # 2+1+2 = 5
        self.assertTrue(GR.BAT_EVENTS["3+ K"](s))
        self.assertFalse(GR.BAT_EVENTS["BB"](s))
        self.assertTrue(GR.BAT_EVENTS["2+ RBI"](s))


class TestPricing(unittest.TestCase):
    def test_american_odds(self):
        self.assertEqual(P.american_odds(0.5), "-100")
        self.assertEqual(P.american_odds(2 / 3), "-200")
        self.assertEqual(P.american_odds(1 / 3), "+200")
        self.assertEqual(P.american_odds(0.0), "")
        self.assertEqual(P.american_odds(1.5), "")

    def test_poisson_over(self):
        self.assertAlmostEqual(P.poisson_over(1.0, 0.5),
                               1 - math.exp(-1.0), places=12)
        # P(X > 1.5) = 1 - P(0) - P(1)
        self.assertAlmostEqual(P.poisson_over(2.0, 1.5),
                               1 - math.exp(-2) * (1 + 2), places=12)

    def test_nb_over(self):
        # disp <= 1 falls back to Poisson exactly
        self.assertEqual(P.nb_over(3.0, 2.5, 1.0), P.poisson_over(3.0, 2.5))
        # against scipy's negative binomial with the same mean/variance
        from scipy import stats
        lam, disp, line = 4.0, 2.3, 5.5
        r = lam / (disp - 1.0)
        self.assertAlmostEqual(P.nb_over(lam, line, disp),
                               float(stats.nbinom(r, 1.0 / disp).sf(
                                   math.floor(line))), places=10)
        # monotone decreasing in the line
        self.assertGreater(P.nb_over(4.0, 2.5, 2.0), P.nb_over(4.0, 6.5, 2.0))

    def test_poisson_win(self):
        self.assertAlmostEqual(P.poisson_win(4.5, 4.5), 0.5, places=9)
        self.assertAlmostEqual(P.poisson_win(5.0, 3.0)
                               + P.poisson_win(3.0, 5.0), 1.0, places=9)
        self.assertGreater(P.poisson_win(6.0, 3.0), 0.5)

    def test_elo_expected(self):
        self.assertGreater(F.elo_expected(1500.0, 1500.0), 0.5)  # HFA
        self.assertAlmostEqual(
            F.elo_expected(1500.0 - F.ELO_HFA / 2, 1500.0 + F.ELO_HFA / 2),
            0.5, places=9)

    def test_shrink_zero_sums_hit_prior(self):
        prior = F.SHRINK["hr_pa"][0]
        self.assertAlmostEqual(F._shrink(0.0, 0.0, "hr_pa"), prior, places=12)

    def test_plattcal(self):
        rng = np.random.default_rng(0)
        z = rng.normal(0, 1.5, 20000)
        p = 1 / (1 + np.exp(-z))
        y = (rng.random(20000) < p).astype(float)
        cal = F.PlattCal().fit(p, y)
        self.assertTrue(0.9 < cal.a < 1.1, cal.a)
        self.assertTrue(abs(cal.b) < 0.1, cal.b)
        out = cal.predict(np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
        self.assertTrue(np.all(np.diff(out) > 0))   # monotone


class TestOdds(unittest.TestCase):
    def test_american_to_prob(self):
        self.assertAlmostEqual(O.american_to_prob(150), 0.4, places=12)
        self.assertAlmostEqual(O.american_to_prob(-200), 2 / 3, places=12)
        self.assertTrue(np.isnan(O.american_to_prob(None)))

    def test_devig_two_way(self):
        fair, hold = O.devig_two_way(-110, -110)
        self.assertAlmostEqual(fair, 0.5, places=12)
        self.assertAlmostEqual(hold, 2 * (110 / 210) - 1, places=12)
        fair, hold = O.devig_two_way(150, None)   # one-sided: raw implied
        self.assertAlmostEqual(fair, 0.4, places=12)
        self.assertTrue(np.isnan(hold))

    def test_ev_and_settle(self):
        self.assertAlmostEqual(O.ev_per_unit(0.5, 100), 0.0, places=12)
        self.assertGreater(O.ev_per_unit(0.55, 100), 0.0)
        self.assertAlmostEqual(O.settle(150, True), 1.5, places=12)
        self.assertAlmostEqual(O.settle(-200, True), 0.5, places=12)
        self.assertEqual(O.settle(150, False), -1.0)

    def test_pick_side(self):
        side, ev = O.pick_side(0.6, 100, 100)   # over is clearly +EV
        self.assertEqual(side, "over")
        self.assertAlmostEqual(ev, 0.2, places=12)
        side, ev = O.pick_side(0.5, -110, -110)  # vig eats both sides
        self.assertIsNone(side)
        side, _ = O.pick_side(0.3, -300, 120)    # under side value
        self.assertEqual(side, "under")


class TestFeaturesHelpers(unittest.TestCase):
    def test_haversine(self):
        km = float(F.haversine_km(40.75, -74.0, 34.05, -118.25))
        self.assertTrue(3800 < km < 4050, km)     # NYC -> LA
        self.assertAlmostEqual(float(F.haversine_km(40.0, -75.0, 40.0, -75.0)),
                               0.0, places=9)

    def test_sched_from_prev(self):
        coords = {"A": (40.0, -75.0), "B": (34.0, -118.0)}
        out = F.sched_from_prev(pd.Timestamp("2026-07-14"), "A", "N",
                                pd.Timestamp("2026-07-15"), "D", "B", coords)
        self.assertEqual(out["day_after_night"], 1.0)
        self.assertGreater(out["travel_km"], 3000)
        self.assertLess(out["tz_delta"], 0)       # traveling west
        out = F.sched_from_prev(None, None, None,
                                pd.Timestamp("2026-07-15"), "D", "B", coords)
        self.assertTrue(np.isnan(out["travel_km"]))

    def test_hrpt_from_counts(self):
        self.assertAlmostEqual(
            F.hrpt_from_counts({"Slider": 2}, {"Slider": 0.5}, 4), 0.25)
        self.assertTrue(np.isnan(F.hrpt_from_counts({}, {"Slider": 0.5}, 0)))

    def test_outs_sd_from_sums(self):
        outs = np.array([15.0, 18, 21, 12, 18, 15, 21, 18])
        sd = float(F.outs_sd_from_sums(len(outs), outs.sum(),
                                       (outs ** 2).sum()))
        self.assertAlmostEqual(sd, float(outs.std()), places=9)
        self.assertTrue(np.isnan(float(F.outs_sd_from_sums(2, 30.0, 500.0))))


class TestGradingSettlement(unittest.TestCase):
    def setUp(self):
        self.s_hr = pd.Series({"H": 1, "2B": 0, "3B": 0, "HR": 1, "TB": 4,
                               "R": 1, "RBI": 1, "BB": 0, "SO": 1, "SB": 0})
        self.s_quiet = self.s_hr * 0
        self.bg = {(7, 101): self.s_hr, (7, 102): self.s_quiet}
        self.sg = {(9, 101): {"SO": 7.0, "H": 4.0, "BB": 1.0, "ER": 2.0,
                              "outs": 18}}
        self.one = {"A@B": [{"total": 9.0, "away": 4.0, "home": 5.0,
                             "gamepk": 101, "winner": "B"}]}
        self.two = {"A@B": [{"total": 9.0, "away": 4.0, "home": 5.0,
                             "gamepk": 101, "winner": "B"},
                            {"total": 3.0, "away": 2.0, "home": 1.0,
                             "gamepk": 102, "winner": "A"}]}

    def test_row_stats_per_game(self):
        day = {7: self.s_hr + self.s_quiet}
        # G# routes to the right game of a DH
        self.assertIs(GR._row_stats(self.bg, day, self.two, 7, "A@B", 1),
                      self.s_hr)
        self.assertIs(GR._row_stats(self.bg, day, self.two, 7, "A@B", 2),
                      self.s_quiet)
        # game not final yet -> None, never misgraded
        self.assertIsNone(GR._row_stats(self.bg, day, self.one, 7, "A@B", 2))
        # legacy book (no tag/G#) -> day-sum fallback
        self.assertIs(GR._row_stats(self.bg, day, self.two, 7, None, None),
                      day[7])
        self.assertIsNone(GR._row_stats(self.bg, day, self.two, "x", "A@B", 1))

    def _settle(self, row, games):
        return GR._settle_bet(row, {7: self.s_hr}, {9: self.sg[(9, 101)]},
                              games, lambda g, n: 7, lambda g, n: 9,
                              bg=self.bg, sg=self.sg)

    def test_settle_moneyline_total(self):
        self.assertTrue(self._settle(
            {"Game": "A@B", "Prop": "moneyline", "Side": "B"}, self.one))
        self.assertIsNone(self._settle(       # DH day: can't pin the game
            {"Game": "A@B", "Prop": "moneyline", "Side": "B"}, self.two))
        self.assertTrue(self._settle(
            {"Game": "A@B", "Prop": "total runs", "Side": "Over",
             "Line": 8.5}, self.one))
        self.assertFalse(self._settle(
            {"Game": "A@B", "Prop": "total runs", "Side": "Under",
             "Line": 8.5}, self.one))

    def test_settle_player_props(self):
        self.assertTrue(self._settle(
            {"Game": "A@B", "Prop": "1+ HR", "Side": "Over",
             "Player": "X"}, self.one))
        self.assertFalse(self._settle(
            {"Game": "A@B", "Prop": "1+ HR", "Side": "Under",
             "Player": "X"}, self.one))
        self.assertIsNone(self._settle(       # ambiguous DH -> unsettled
            {"Game": "A@B", "Prop": "1+ HR", "Side": "Over",
             "Player": "X"}, self.two))
        self.assertTrue(self._settle(
            {"Game": "A@B", "Prop": "pitcher strikeouts o6.5",
             "Side": "Over", "Line": 6.5, "Player": "Y"}, self.one))


class TestTrainEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = importlib.import_module("train")
        cls.E = importlib.import_module("evaluate_deep")

    def test_es_split(self):
        df = pd.DataFrame({"GamePk": np.arange(1000), "x": np.arange(1000)})
        fit, es = self.T._es_split(df)
        self.assertEqual(len(fit) + len(es), len(df))
        self.assertEqual(len(set(fit["GamePk"]) & set(es["GamePk"])), 0)
        self.assertAlmostEqual(len(es) / len(df), 0.10, places=2)
        # deterministic + row-order independent
        fit2, es2 = self.T._es_split(df.sample(frac=1, random_state=1))
        self.assertEqual(set(es2["GamePk"]), set(es["GamePk"]))

    def test_fit_line_cals_monotone(self):
        mu = np.linspace(0.0, 10.0, 400)
        y = mu + np.sin(mu)               # deterministic, monotone-ish
        cals = self.T.fit_line_cals(mu, y, [2.5, 5.5, 100.5])
        self.assertIn(2.5, cals)
        self.assertNotIn(100.5, cals)     # degenerate line skipped
        p_lo = cals[5.5].predict_proba([[2.0]])[0, 1]
        p_hi = cals[5.5].predict_proba([[8.0]])[0, 1]
        self.assertLess(p_lo, p_hi)

    def test_bh_qvalues(self):
        q = self.E.bh_qvalues([0.01, 0.02, 0.03, 0.04])
        np.testing.assert_allclose(q, [0.04, 0.04, 0.04, 0.04], atol=1e-12)
        q = self.E.bh_qvalues([0.001, 0.5, 0.9])
        self.assertAlmostEqual(q[0], 0.003, places=9)
        self.assertTrue(q[0] <= q[1] <= q[2])

    def test_verdict_bands(self):
        self.assertEqual(self.E.verdict("hr_auc", 0.600, 0.601),
                         "within noise")
        self.assertEqual(self.E.verdict("hr_auc", 0.600, 0.620), "better")
        self.assertEqual(self.E.verdict("hr_ece", 0.010, 0.020), "worse")
        self.assertEqual(self.E.verdict("k_dispersion", 1.30, 1.05), "better")


class TestBattery(unittest.TestCase):
    """Steal-layer battery modulation (#35) — the pure application math;
    the table BUILD reads Data CSVs and is exercised by the Phase-3
    regrade, not here."""

    @classmethod
    def setUpClass(cls):
        cls.PS = importlib.import_module("pa_sim")
        cls.tables = {"ratio": {(2025, 123): 1.40},
                      "stop": {(2025, "NYY"): 0.05}}

    def test_context_lookup_and_fallbacks(self):
        PS = self.PS
        self.assertEqual(PS.battery_context(self.tables, 2025, "NYY", 123),
                         (1.40, 0.05))
        # missing starter / team / both -> neutral halves
        self.assertEqual(PS.battery_context(self.tables, 2025, "NYY", 999),
                         (1.0, 0.05))
        self.assertEqual(PS.battery_context(self.tables, 2025, "BOS", 123),
                         (1.40, 0.0))
        self.assertEqual(PS.battery_context(self.tables, 2024, "NYY", 123),
                         (1.0, 0.0))
        # stale cache (no battery tables) and unparseable starter -> neutral
        self.assertEqual(PS.battery_context(None, 2025, "NYY", 123),
                         (1.0, 0.0))
        self.assertEqual(PS.battery_context(self.tables, 2025, "NYY",
                                            float("nan")),
                         (1.0, 0.0))

    def test_flag_off_is_exact_noop(self):
        PS = self.PS
        old = PS.STEAL_BATTERY
        try:
            PS.STEAL_BATTERY = False
            self.assertEqual(PS.battery_context(self.tables, 2025, "NYY",
                                                123), (1.0, 0.0))
        finally:
            PS.STEAL_BATTERY = old
        att = np.full(9, 0.08)
        succ = np.full(9, 0.78)
        a2, s2 = PS.battery_adjust(att, succ, 1.0, 0.0)
        np.testing.assert_allclose(a2, att)
        np.testing.assert_allclose(s2, succ)

    def test_adjust_math_and_clips(self):
        PS = self.PS
        att = np.array([0.05, 0.30, 0.50])
        succ = np.array([0.40, 0.78, 0.96])
        a2, s2 = PS.battery_adjust(att, succ, 1.5, 0.05)
        np.testing.assert_allclose(a2, [0.075, 0.45, 0.6])   # cap 0.6
        np.testing.assert_allclose(s2, [0.35, 0.73, 0.91])   # floor 0.35
        a3, s3 = PS.battery_adjust(att, succ, 0.6, -0.05)
        np.testing.assert_allclose(a3, [0.03, 0.18, 0.30])
        np.testing.assert_allclose(s3, [0.45, 0.83, 0.98])   # ceil 0.98


class TestConfirmIsStale(unittest.TestCase):
    """update_all.confirm_is_stale — the auto-ship-confirm trigger (audit
    #6 as amended 07-15). Hermetic: MODEL_TRAIN is repointed at a temp
    tree so no real artifacts are read."""

    def setUp(self):
        import hashlib
        import json as _json
        import tempfile
        self.UA = importlib.import_module("update_all")
        self._saved = self.UA.MODEL_TRAIN
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "artifacts").mkdir()
        self.UA.MODEL_TRAIN = self.tmp / "train.py"
        (self.tmp / "features.py").write_text("shipped = 1\n")
        digest = hashlib.md5(
            (self.tmp / "features.py").read_bytes()).hexdigest()
        self.fp_file = self.tmp / "artifacts" / "confirm_code_fp.json"
        self.fp_json = _json.dumps({"features.py": digest})

    def tearDown(self):
        import shutil as _shutil
        self.UA.MODEL_TRAIN = self._saved
        _shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_file_bootstraps(self):
        self.assertTrue(self.UA.confirm_is_stale())

    def test_matching_fp_stands_down(self):
        self.fp_file.write_text(self.fp_json)
        self.assertFalse(self.UA.confirm_is_stale())

    def test_changed_source_triggers(self):
        self.fp_file.write_text(self.fp_json)
        (self.tmp / "features.py").write_text("shipped = 2\n")
        self.assertTrue(self.UA.confirm_is_stale())

    def test_vanished_source_triggers(self):
        self.fp_file.write_text(self.fp_json)
        (self.tmp / "features.py").unlink()
        self.assertTrue(self.UA.confirm_is_stale())

    def test_corrupt_fp_bootstraps(self):
        self.fp_file.write_text("{not json")
        self.assertTrue(self.UA.confirm_is_stale())


class TestDiversityBatchUnits(unittest.TestCase):
    """2026-07-15 PM batch: the new pure pieces — BaggedCal, FamilyBlendBag,
    family_logits, weighted Platt/beta fits, the winner mirror, weighted
    line cals."""

    def test_bagged_cal_is_mean_of_members(self):
        class C:
            def __init__(self, k):
                self.k = k

            def predict(self, p):
                return np.asarray(p) * self.k

        bag = F.BaggedCal([C(0.5), C(1.5)], kind="test+bag2")
        np.testing.assert_allclose(bag.predict([0.2, 0.4]), [0.2, 0.4])

    def test_family_blend_bag_weights_and_contract(self):
        class M:
            def __init__(self, v):
                self.v = v
                self.best_iteration_ = 7

            def predict(self, X):
                return np.full(len(X), self.v)

        models = [M(1.0), M(3.0), M(10.0)]        # lgbm x2, cb x1
        bag = F.FamilyBlendBag(models, {"lgbm": (0, 2), "cb": (2, 3)},
                               {"lgbm": 0.75, "cb": 0.25})
        self.assertIsInstance(bag, F.MeanBag)      # members-aware consumers
        self.assertEqual(len(bag.models), 3)
        X = pd.DataFrame({"a": [0, 1]})
        # 0.75 * mean(1,3) + 0.25 * 10 = 1.5 + 2.5 = 4.0
        np.testing.assert_allclose(bag.predict(X), [4.0, 4.0])

    def test_family_logits_per_family_mean(self):
        class M:
            def __init__(self, p):
                self.p = p

            def predict_proba(self, X):
                p = np.full(len(X), self.p)
                return np.column_stack([1 - p, p])

        X = pd.DataFrame({"a": [0, 1, 2]})
        zf = F.family_logits([M(0.2), M(0.4), M(0.8)],
                             {"lgbm": (0, 2), "cb": (2, 3)}, X)
        want_lgbm = 0.5 * (F.logit(0.2) + F.logit(0.4))
        np.testing.assert_allclose(zf["lgbm"], want_lgbm, rtol=1e-9)
        np.testing.assert_allclose(zf["cb"], F.logit(0.8), rtol=1e-9)

    def test_weighted_cals_reduce_to_unweighted(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 400)
        y = (rng.uniform(size=400) < p).astype(float)
        ones = np.ones_like(y)
        for cls in (F.PlattCal, F.BetaCal):
            c0 = cls().fit(p, y)
            c1 = cls().fit(p, y, w=ones)
            np.testing.assert_allclose(c0.predict(p), c1.predict(p),
                                       atol=1e-8)
        # weights actually bite: upweighting the y=1 rows raises the curve
        c2 = F.PlattCal().fit(p, y, w=np.where(y > 0, 5.0, 1.0))
        self.assertGreater(c2.predict(p).mean(),
                           F.PlattCal().fit(p, y).predict(p).mean())

    def test_mirror_win_swaps_negates_and_flips(self):
        T = importlib.import_module("train")
        df = pd.DataFrame({
            "GamePk": [10, 11], "Season": [2024, 2024],
            "home_x": [1.0, 2.0], "away_x": [3.0, 4.0],
            "d_x": [-2.0, -2.0], "elo_prob_home": [0.6, 0.55],
            "persp_home": [1.0, 1.0], "y_home_win": [1, 0]})
        m = T._mirror_win(df)
        np.testing.assert_allclose(m["home_x"], [3.0, 4.0])
        np.testing.assert_allclose(m["away_x"], [1.0, 2.0])
        np.testing.assert_allclose(m["d_x"], [2.0, 2.0])
        np.testing.assert_allclose(m["elo_prob_home"], [0.4, 0.45])
        np.testing.assert_allclose(m["persp_home"], [0.0, 0.0])
        self.assertEqual(list(m["y_home_win"]), [0, 1])
        # double mirror = identity on everything except the persp flag
        mm = T._mirror_win(m)
        for c in ["home_x", "away_x", "d_x", "elo_prob_home", "y_home_win"]:
            np.testing.assert_allclose(mm[c], df[c])

    def test_dag_coherence(self):
        rng = np.random.default_rng(3)
        n = 200
        # deliberately incoherent: hr above hit/run/rbi on many rows
        p = {"hit": rng.uniform(.4, .8, n), "single": rng.uniform(.3, .9, n),
             "double": rng.uniform(.1, .5, n), "triple": rng.uniform(0, .2, n),
             "hr": rng.uniform(.05, .6, n), "tb2": rng.uniform(.2, .6, n),
             "tb3": rng.uniform(.1, .4, n), "tb4": rng.uniform(.02, .3, n),
             "run": rng.uniform(.2, .6, n), "run2": rng.uniform(.05, .3, n),
             "rbi": rng.uniform(.2, .6, n), "rbi2": rng.uniform(.05, .3, n),
             "hits2": rng.uniform(.1, .5, n), "hrr2": rng.uniform(.2, .7, n),
             "hrr3": rng.uniform(.1, .5, n), "hrr4": rng.uniform(.02, .3, n)}
        F.enforce_ladders(p)
        tol = 1e-9
        for child, parent in F.PROP_DAG_EDGES:
            self.assertTrue((p[child] <= p[parent] + 1e-6).all(),
                            f"{child} <= {parent} violated")
        # within-family ladders stay exact after the DAG passes
        for ladder in F.PROP_LADDERS:
            for a, b in zip(ladder, ladder[1:]):
                if a in p and b in p:
                    self.assertTrue((p[b] <= p[a] + tol).all(),
                                    f"ladder {a}>={b} violated")

    def test_bagged_line_cal_contract(self):
        class C:
            def __init__(self, k):
                self.k = k

            def predict_proba(self, X):
                p = np.clip(np.asarray(X).ravel() * self.k, 0.01, 0.99)
                return np.column_stack([1 - p, p])

        bag = F.BaggedLineCal([C(0.1), C(0.3)])
        p = bag.predict_proba(np.array([[1.0], [2.0]]))[:, 1]
        np.testing.assert_allclose(p, [0.2, 0.4])

    def test_line_cals_weighted_and_degenerate_skip(self):
        T = importlib.import_module("train")
        mu = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])
        cals = T.fit_line_cals(mu, y, [1.5, 99.5], w=np.ones(6))
        self.assertIn(1.5, cals)          # priced
        self.assertNotIn(99.5, cals)      # single-class -> skipped
        p = cals[1.5].predict_proba(np.array([[0.5], [5.0]]))[:, 1]
        self.assertLess(p[0], p[1])       # monotone in mu


class TestDiversityBatchEndToEnd(unittest.TestCase):
    """Synthetic two-suite train -> serve parity for the fstack/multi-year/
    ES-refit/bagged-cal path (fit_classifier + predict_prop), the mirrored
    winner (fit_winner + predict_win), and the count family blend
    (fit_poisson). Tiny frames + tiny booster params: seconds, no GPU (the
    CatBoost test members are re-pointed at CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.T = importlib.import_module("train")
        rng = np.random.default_rng(0)
        rows_per = 260
        frames = []
        gpk = 0
        for season in range(2021, 2027):
            n = rows_per
            x1 = rng.normal(size=n)
            x2 = rng.normal(size=n)
            p = 1 / (1 + np.exp(-(0.9 * x1 - 0.3)))
            frames.append(pd.DataFrame({
                "Season": season,
                "GamePk": np.arange(gpk, gpk + n),
                "Date": pd.Timestamp(f"{season}-05-01")
                + pd.to_timedelta(np.arange(n) % 40, unit="D"),
                "x1": x1, "x2": x2,
                "y": (rng.uniform(size=n) < p).astype(int),
                "y_cnt": rng.poisson(np.exp(0.3 * x1 + 0.2)),
            }))
            gpk += n
        cls.df = pd.concat(frames, ignore_index=True)
        cls.cols = ["x1", "x2"]
        cls.small = dict(n_estimators=40, learning_rate=0.1, num_leaves=7,
                         min_child_samples=5, objective="binary", verbose=-1)
        # CPU CatBoost stand-ins for the GPU training configs
        cls._saved = {k: getattr(cls.T, k) for k in
                      ("CB_CLS", "CB_POIS", "CB_WIN", "LGB_WIN",
                       "LGBM_BAGS", "XGB_BAGS", "CB_BAGS")}
        cb_small = dict(iterations=25, learning_rate=0.1, depth=3,
                        verbose=0, allow_writing_files=False,
                        early_stopping_rounds=25)
        cls.T.CB_CLS = dict(cb_small, loss_function="Logloss")
        cls.T.CB_POIS = dict(cb_small, loss_function="Poisson")
        cls.T.CB_WIN = dict(cb_small, loss_function="Logloss")
        cls.T.LGB_WIN = dict(cls.small)
        cls.T.LGBM_BAGS, cls.T.XGB_BAGS, cls.T.CB_BAGS = 2, 0, 1

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._saved.items():
            setattr(cls.T, k, v)
        cls.T._CAL_STASH.clear()

    def test_fit_classifier_two_suites_and_serve_parity(self):
        T = self.T
        T._CAL_STASH.clear()
        # selection-like suite first: train<=2023, cal 2024, test 2025
        prop_sel, m_sel = T.fit_classifier(
            self.df, self.cols, "y", [2021, 2022, 2023], 2024, 2025,
            "SMOKE", params=self.small, n_bags=2, n_cb=1, head_key="smoke")
        self.assertEqual(m_sel["cal_pool_years"], [2024])
        # shipping-like suite pools the stash: train<=2024, cal 2025, test 2026
        prop, m = T.fit_classifier(
            self.df, self.cols, "y", [2021, 2022, 2023, 2024], 2025, 2026,
            "SMOKE", params=self.small, n_bags=2, n_cb=1, head_key="smoke")
        if T.MULTI_YEAR_CAL:
            self.assertEqual(m["cal_pool_years"], [2024, 2025])
        self.assertIn("fstack", prop)
        # depth-3 pooling (CAL_POOL_YEARS=3): a third suite one season back
        # feeds two prior years into the newest suite's support
        saved_depth = T.CAL_POOL_YEARS
        try:
            T.CAL_POOL_YEARS = 3
            T._CAL_STASH.clear()
            T.fit_classifier(self.df, self.cols, "y", [2021, 2022], 2023,
                             2024, "SMOKE", params=self.small, n_bags=2,
                             n_cb=1, head_key="smoke3")
            T.fit_classifier(self.df, self.cols, "y", [2021, 2022, 2023],
                             2024, 2025, "SMOKE", params=self.small,
                             n_bags=2, n_cb=1, head_key="smoke3")
            _, m3 = T.fit_classifier(self.df, self.cols, "y",
                                     [2021, 2022, 2023, 2024], 2025, 2026,
                                     "SMOKE", params=self.small, n_bags=2,
                                     n_cb=1, head_key="smoke3")
            if T.MULTI_YEAR_CAL:
                self.assertEqual(m3["cal_pool_years"], [2023, 2024, 2025])
        finally:
            T.CAL_POOL_YEARS = saved_depth
            T._CAL_STASH.clear()
        self.assertEqual(m["families"], {"lgbm": 2, "cb": 1})
        self.assertIn(m["lr_C"], T.LR_C_GRID)   # per-head ridge pick landed
        if T.CAL_BAG_B:
            self.assertIn("+bag", m["calibrator"])
        # serve parity: predict_prop must run the fstack path end-to-end
        prop["cols"] = self.cols
        te = self.df[self.df["Season"] == 2026]
        p = P.predict_prop(prop, te)
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.all((p > 0) & (p < 1)))
        # ranking sanity: the true signal is x1
        self.assertGreater(np.corrcoef(p, te["x1"])[0, 1], 0.1)

    def test_fit_winner_mirror_and_serve(self):
        T = self.T
        T._CAL_STASH.clear()
        rng = np.random.default_rng(1)
        frames = []
        gpk = 0
        for season in range(2021, 2027):
            n = 200
            hs = rng.normal(size=n)
            as_ = rng.normal(size=n)
            d = hs - as_
            p = 1 / (1 + np.exp(-(1.2 * d + 0.25)))   # home edge
            frames.append(pd.DataFrame({
                "Season": season, "GamePk": np.arange(gpk, gpk + n),
                "Date": pd.Timestamp(f"{season}-05-01")
                + pd.to_timedelta(np.arange(n) % 40, unit="D"),
                "home_str": hs, "away_str": as_, "d_str": d,
                "elo_prob_home": np.clip(p + rng.normal(0, .05, n), .02, .98),
                "persp_home": 1.0,
                "y_home_win": (rng.uniform(size=n) < p).astype(int)}))
            gpk += n
        wf = pd.concat(frames, ignore_index=True)
        cols = ["home_str", "away_str", "d_str", "elo_prob_home",
                "persp_home"]
        win, m = T.fit_winner(wf, cols, "y_home_win", None,
                              [2021, 2022, 2023, 2024], 2025, 2026, "WSMOKE")
        if T.WINNER_MIRROR:
            # mirrored copies double the training rows
            self.assertEqual(m["n_train"], 2 * 4 * 200)
        self.assertIn("fstack", win)
        self.assertFalse(win["fstack_pois"])
        win["cols"] = cols
        te = wf[wf["Season"] == 2026].drop(columns=["persp_home"])
        p = P.predict_win(win, te, None, None)   # persp pinned to 1.0 inside
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.all((p > 0) & (p < 1)))
        self.assertGreater(np.corrcoef(p, te["d_str"])[0, 1], 0.1)

    def test_fit_poisson_family_blend(self):
        T = self.T
        T._CAL_STASH.clear()
        small = dict(n_estimators=40, learning_rate=0.1, num_leaves=7,
                     min_child_samples=5, objective="poisson", verbose=-1)
        model, m = T.fit_poisson(
            self.df, self.cols, "y_cnt", [2021, 2022, 2023, 2024], 2025,
            2026, "CSMOKE", lambda te: pd.Series(1.0, index=te.index),
            n_bags=2, n_cb=1, params=small, head_key="csmoke")
        if T.FAMILY_STACK:
            self.assertIsInstance(model, F.FamilyBlendBag)
            self.assertAlmostEqual(sum(model.fam_w.values()), 1.0, places=6)
        mu = model.predict(self.df[self.df["Season"] == 2026][self.cols])
        self.assertTrue(np.all(np.isfinite(mu)) and np.all(mu > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
