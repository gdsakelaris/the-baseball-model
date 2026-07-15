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


if __name__ == "__main__":
    unittest.main(verbosity=2)
