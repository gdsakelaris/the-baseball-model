"""Feature engineering for the MLB prediction models.

Builds leakage-free features: every feature for a game on date D uses only
data from strictly before D. Two paths share the same definitions:

  - build_batter_frame / build_starts_frame / build_game_frame:
      vectorized, over every season since Scrapers/seasons.FIRST_SEASON
      (2015), for training (cumsum/rolling shifted by one game so the
      current game never sees itself).
  - Stores.asof_*:
      per-entity, for an arbitrary future/hypothetical game, for inference.

Data used (all files in Data/):
  game logs (3 files)      -> form, rest, bullpen (incl. high-leverage arms
                              and trailing-3-day fatigue), team offense
                              (overall + home/away), park factor, home/away
                              batter splits, position (C/DH) shares,
                              strike rate, defense proxy (unearned runs),
                              battery SB-allowed (team steals surrendered)
  season stats (2 files)   -> prior-season GO/AO (groundball/flyball) and
                              pitcher SB/CS/PK stolen-base control
  pitch arsenals (2 files) -> batter-vs-arsenal matchup scores and pitcher
                              arsenal quality (two-year decay blend)
  homeruns                 -> batter HR power quality (exit velo, launch
                              angle, elevation-adjusted distance), pitcher
                              HR quality allowed, batter HR-by-pitch-type
                              profile vs the starter's usage
  rosters                  -> handedness, platoon, age, height, weight
  ballparks                -> dimensions, elevation
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"

ROLL_WINDOWS = (7, 15, 30)
CAT_COLS = ["DayNight", "Condition", "WindDir", "bat_hand", "pit_throws"]

# Decay-weighted "current form": every prior game weighted exp(-days_ago *
# DECAY_LAM), half-life DECAY_HL_DAYS. Sits between the rolling windows
# (hard cliff at N games) and season-to-date (all games equal): skill drift
# shows up gradually and old games fade instead of falling off an edge. The
# decayed PA total is a real effective sample size in PA units, so the same
# SHRINK constants apply to the decayed rates.
DECAY_HL_DAYS = 90.0
DECAY_LAM = np.log(2.0) / DECAY_HL_DAYS
DECAY_EPOCH = pd.Timestamp("2020-01-01")
DECAY_STATS = ["PA", "AB", "H", "HR", "TB", "SO", "BB", "SB", "R", "RBI",
               "HBP", "HRR2", "HRR3", "HRR4", "RBI2", "RUN2", "LOB"]
DECAY_SHRINK = ("hr_pa", "tb_ab", "k_pct", "bb_pct", "sb_pa",
                "r_pa", "rbi_pa")

# Expected exposure (xPA): every binary prop is really P(gets N plate
# appearances) x P(event per PA), but the trees only see per-PA rates and
# slot — they cannot form a PA-per-game ratio from cumulative sums. Two
# explicit exposure features: xpa_bat (the batter's own decayed PA per
# game — pinch-hit / platoon-removal / early-pull risk) and xpa_slot
# (as-of league PA per game at tonight's lineup slot — leadoff 4.48 vs
# nine hole 3.31).
# BENCHED (2026-07-07 selection run): 1 better (rbi_top10, barely past
# band) vs 3 worse (rbi/single/bk ECE past band), every AUC/edge within
# noise — calibration harm without ranking gain; slot + per-PA rates
# evidently already carry the usable exposure signal. Both stay in the
# frames and the Stores/predict wiring, but out of batter_feature_cols.
XPA_PRIOR = 4.3          # league PA per batter-game (shrinkage target)
XPA_K = 10.0             # games of shrinkage for xpa_bat

# ---------------------------------------------------------------- loading


class MeanBag:
    """Average-prediction ensemble over seed-bagged LightGBM models: quacks
    like a single model (predict / predict_proba / best_iteration_) so every
    consumer — predict_prop, count heads, selftest, GUI — works unchanged.
    Lives here so pickled artifacts resolve it from any entry point."""

    def __init__(self, models):
        self.models = models
        self.best_iteration_ = models[0].best_iteration_

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


class InfSafe:
    """Wrap a model whose library rejects ±inf (XGBoost) for use in a MeanBag
    next to LightGBM members that tolerate them: numeric ±inf -> NaN at
    fit/predict, categorical dtypes preserved. Lives here so pickled
    artifacts resolve it from any entry point (MeanBag precedent)."""

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _clean(X):
        X = X.copy()
        num = X.select_dtypes(include=[np.number]).columns
        X[num] = X[num].replace([np.inf, -np.inf], np.nan)
        return X

    def fit(self, X, y, **kw):
        es = kw.pop("eval_set", None)
        if es is not None:
            kw["eval_set"] = [(self._clean(a), b) for a, b in es]
        self.model.fit(self._clean(X), y, **kw)
        return self

    @property
    def best_iteration_(self):
        return getattr(self.model, "best_iteration", None)

    def predict(self, X):
        return self.model.predict(self._clean(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._clean(X))


class CatSafe:
    """Wrap a CatBoost model for use in a MeanBag next to LightGBM members:
    numeric ±inf -> NaN (CatBoost tolerates numeric NaN but not inf),
    categoricals -> plain strings with NaN as its own 'missing' level
    (CatBoost rejects NaN in cat features). Poisson/Tweedie regressors need
    prediction_type='Exponent' to return means instead of log-means —
    `exponent=True` handles it. Lives here so pickles resolve everywhere."""

    def __init__(self, model, cat_cols, exponent=False):
        self.model = model
        self.cat_cols = list(cat_cols)
        self.exponent = exponent

    def _clean(self, X):
        X = X.copy()
        num = X.select_dtypes(include=[np.number]).columns
        X[num] = X[num].replace([np.inf, -np.inf], np.nan)
        for c in self.cat_cols:
            if c in X.columns:
                X[c] = X[c].astype(str).fillna("missing")
        return X

    def fit(self, X, y, **kw):
        es = kw.pop("eval_set", None)
        if es is not None:
            kw["eval_set"] = [(self._clean(a), b) for a, b in es]
        self.model.fit(self._clean(X), y, **kw)
        return self

    @property
    def best_iteration_(self):
        return self.model.get_best_iteration()

    def predict(self, X):
        if self.exponent:
            return self.model.predict(self._clean(X),
                                      prediction_type="Exponent")
        return self.model.predict(self._clean(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._clean(X))


class PlattCal:
    """Two-parameter Platt map for heads where cal-year isotonic overfits:
    p_out = sigmoid(a * logit(p_in) + b), fit by Newton on cal-year log loss.
    Isotonic's free-form steps memorize the thin-support extremes (winner has
    ~2.4k cal games; double is the weakest batter signal), which serves
    overconfident tails on the holdout — a 2-parameter sigmoid can only
    shrink or shift, and a > 0 keeps it monotone so ranking/AUC are identical
    by construction. Same predict() contract as IsotonicRegression, so it
    drops in under the artifact's "iso" key. Lives here so pickles resolve
    from any entry point (MeanBag precedent)."""

    def __init__(self, lo=1e-4):
        self.a, self.b, self.lo = 1.0, 0.0, lo

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def fit(self, p, y, iters=25, ridge=1e-4):
        z = self._logit(p)
        y = np.asarray(y, dtype=float)
        a, b = 1.0, 0.0
        for _ in range(iters):
            q = 1.0 / (1.0 + np.exp(-(a * z + b)))
            w = np.clip(q * (1 - q), 1e-6, None)
            g0 = float(((q - y) * z).sum())        # d(nll)/da
            g1 = float((q - y).sum())              # d(nll)/db
            h00 = float((w * z * z).sum()) + ridge
            h01 = float((w * z).sum())
            h11 = float(w.sum()) + ridge
            det = h00 * h11 - h01 * h01
            if not np.isfinite(det) or det <= 0:
                break
            da = (h11 * g0 - h01 * g1) / det
            db = (h00 * g1 - h01 * g0) / det
            a -= da
            b -= db
            if abs(da) < 1e-9 and abs(db) < 1e-9:
                break
        self.a = float(np.clip(a, 1e-3, 10.0))     # a > 0: stay monotone
        self.b = float(np.clip(b, -10.0, 10.0))
        return self

    def predict(self, p):
        q = 1.0 / (1.0 + np.exp(-(self.a * self._logit(p) + self.b)))
        return np.clip(q, self.lo, 1 - self.lo)


def inf_to_nan(X):
    """sklearn rejects inf (from divide-by-zero rates); make them missing so
    the imputer handles them. Lives here (not in train's __main__) so a
    pickled pipeline referencing it loads from predict.py/evaluate_deep.py too."""
    X = np.asarray(X, dtype="float64").copy()
    X[~np.isfinite(X)] = np.nan
    return X


def _read(name, **kw):
    return pd.read_csv(DATA_DIR / name, encoding="utf-8-sig", **kw)


def ip_to_outs(ip):
    """'5.2' -> 17 outs. MLB notation: .1/.2 are thirds of an inning."""
    ip = pd.to_numeric(ip, errors="coerce")
    whole = np.floor(ip)
    return (whole * 3 + np.round((ip - whole) * 10)).astype("float64")


def height_to_inches(ht):
    m = re.match(r"(\d+)'\s*(\d+)", str(ht))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


# franchise renames (2026-07-15): the team-keyed PRIOR-season files miss
# across an abbreviation change — 2025 ATH games look up 2024 rows keyed
# OAK and get NaN (y-2 misses the same way). Pre-rename rows are duplicated
# under the successor abbrev at LOAD time so every consumer (vectorized
# joins and the serving Stores) sees the aliased table — parity by
# construction, and the on-disk files stay the untouched source of truth.
# One entry per rename: successor abbrev -> pre-rename abbrev.
TEAM_RENAMES = {"ATH": "OAK"}                   # 2025 A's


def _alias_renamed_teams(tab):
    """Duplicate pre-rename team rows under the successor abbreviation so
    prior-season joins land across a franchise rename (see TEAM_RENAMES)."""
    if tab is None:
        return None
    extra = []
    for new, old in TEAM_RENAMES.items():
        rows = tab[tab["Team"] == old].copy()
        if len(rows):
            rows["Team"] = new
            extra.append(rows)
    return pd.concat([tab, *extra], ignore_index=True) if extra else tab


def load_raw():
    games = _read("mlb_games.csv", parse_dates=["Date"])
    gb = _read("mlb_game_batting.csv", parse_dates=["Date"])
    # per-game H+R+RBI threshold indicators: own-target history for the hrr
    # props (share of past games clearing the line — the joint clustering
    # that the marginal H/R/RBI rates can't express). Lives on gb so the
    # vectorized and inference paths inherit identical values.
    _hrr = gb["H"] + gb["R"] + gb["RBI"]
    gb["HRR2"] = (_hrr >= 2).astype("float64")
    gb["HRR3"] = (_hrr >= 3).astype("float64")
    # audit wave 2026-07-14: deeper thresholds for the new heads (rbi2/
    # run2/hrr4 shipped without the own-target history their hrr2/hrr3
    # siblings have; RBI runs var/mean 1.64 so the mean rate can't price
    # the tail)
    gb["HRR4"] = (_hrr >= 4).astype("float64")
    gb["RBI2"] = (gb["RBI"] >= 2).astype("float64")
    gb["RUN2"] = (gb["R"] >= 2).astype("float64")
    gp = _read("mlb_game_pitching.csv", parse_dates=["Date"])
    gp["Outs"] = ip_to_outs(gp["IP"])
    gp["NP"] = pd.to_numeric(gp["NP"], errors="coerce")
    gp["Strikes"] = pd.to_numeric(gp["Strikes"], errors="coerce")

    rosters = _read("mlb_rosters.csv")
    rosters["height_in"] = rosters["Ht"].map(height_to_inches)
    rosters["DOB"] = pd.to_datetime(rosters["DOB"], format="%m/%d/%Y", errors="coerce")

    parks = _read("mlb_ballparks.csv")

    hr = _read("mlb_homeruns.csv", parse_dates=["Date"])
    for c in ["Angle", "Exit Velo", "Distance"]:
        hr[c] = pd.to_numeric(hr[c], errors="coerce")
    hr["BatterId"] = pd.to_numeric(hr["BatterId"], errors="coerce")
    # normalize the HR log's pitch names to the arsenal file's vocabulary so
    # the HR-by-pitch-type matchup join actually connects (e.g. the HR log
    # says "Splitter", the arsenal says "Split-Finger"). Names with no
    # arsenal bucket (Eephus) stay as-is and simply never match usage.
    hr["Pitch"] = hr["Pitch"].replace({
        "Splitter": "Split-Finger", "Forkball": "Split-Finger",
        "2-Seam Fastball": "Sinker",
        "Four-Seam Fastball": "4-Seam Fastball", "Fastball": "4-Seam Fastball",
        "Knuckle Curve": "Curveball", "Slow Curve": "Curveball",
        "Knuckle Ball": "Knuckleball",
    })

    def _ars(name):
        a = _read(name)
        for c in ["RV/100", "%", "xSLG", "xwOBA", "Whiff %", "Hard Hit %",
                  "K%", "Put Away %", "Pitches"]:
            a[c] = pd.to_numeric(a[c], errors="coerce")
        return a

    ars_p = _ars("mlb_pitch_arsenals.csv")
    ars_b = _ars("mlb_pitch_arsenals_batters.csv")

    # season stat lines: the only sources of GO/AO (groundball/flyball
    # tendency) and pitcher SB/CS/PK (stolen-base control) — neither is
    # reconstructible from the game logs. Used as PRIOR-season values.
    bat_season = _read("mlb_batting_stats.csv")
    for c in ["GO/AO", "PA"]:
        bat_season[c] = pd.to_numeric(bat_season[c], errors="coerce")
    pit_season = _read("mlb_pitching_stats.csv")
    for c in ["GO/AO", "SB", "CS", "PK", "TBF"]:
        pit_season[c] = pd.to_numeric(pit_season[c], errors="coerce")
    pit_season["Outs"] = ip_to_outs(pit_season["IP"])

    hands = _read("mlb_handedness.csv")
    hands["PlayerId"] = pd.to_numeric(hands["PlayerId"], errors="coerce")
    hands = hands.dropna(subset=["PlayerId"])
    hands["PlayerId"] = hands["PlayerId"].astype("int64")

    # Statcast batted balls (every tracked ball in play): contact-quality
    # "process" stats that stabilize far faster than outcomes. Optional —
    # frames build without the file (features stay NaN) so a checkout
    # without the backfill still runs. Scrapers/scrape_statcast.py creates it.
    bip_path = DATA_DIR / "mlb_statcast_bip.csv"
    bip = None
    if bip_path.exists():
        bip = _read(bip_path.name, parse_dates=["Date"])
        for c in ["HcX", "HcY"]:        # spray coords: absent in old scrapes
            if c not in bip.columns:
                bip[c] = np.nan
        for c in ["BatterId", "PitcherId", "ExitVelo", "LaunchAngle", "LSA",
                  "xBA", "xwOBA", "GamePk", "HcX", "HcY", "HitDistance"]:
            bip[c] = pd.to_numeric(bip[c], errors="coerce")
        # elevation-adjusted hit distance (same sea-level normalization as
        # the HR-quality table) so a Coors 340-ft fly doesn't read as more
        # carry than a sea-level 330; former venues without a park row
        # adjust by 0. Fly-ball distance on ALL flies is the UNCENSORED
        # power measure — the HR log only sees each batter's best contact.
        elev = games.set_index("GamePk")["Venue"].map(
            parks.set_index("Ballpark")["Elevation_ft"])
        bip["DistAdj"] = (bip["HitDistance"]
                          - ELEV_DIST_FT * bip["GamePk"].map(elev).fillna(0.0))

    # pitch-level daily aggregates (whiffs/chases/velo; scrape_pitches.py),
    # sprint speed and team OAA leaderboards — all optional like bip
    def _opt(name, **kw):
        p = DATA_DIR / name
        return _read(name, **kw) if p.exists() else None

    pdp = _opt("mlb_pitch_daily_pitchers.csv", parse_dates=["Date"])
    pdb = _opt("mlb_pitch_daily_batters.csv", parse_dates=["Date"])
    for d in (pdp, pdb):
        if d is not None:
            d["csw_n"] = d["wh_n"] + d["cs_n"]      # called strikes + whiffs
            # zone-split whiffs (oz_wh, 2026-07 schema): in-zone swings and
            # whiffs, for zone-contact. Pre-backfill files lack the column;
            # the derived sums go NaN and every zwsw feature stays NaN.
            if "oz_wh" in d.columns:
                d["z_sw"] = d["sw_n"] - d["oz_sw"]
                d["z_wh"] = d["wh_n"] - d["oz_wh"]
            else:
                d["z_sw"] = np.nan
                d["z_wh"] = np.nan
    # v2/v3 scrape schema (2026-07): elite-velo buckets, pitch-class buckets
    # (breaking/offspeed; fastball = remainder incl. cutters), shadow-band
    # location share, first-pitch counts. Ensure-NaN under an old file so
    # every downstream feature gates the same way as z_sw/z_wh above.
    for d in (pdp, pdb):
        if d is None:
            continue
        for c in ("fb95_n", "fb95_sw", "fb95_wh",
                  "fbmid_n", "fbmid_sw", "fbmid_wh",
                  "fblo_n", "fblo_sw", "fblo_wh",
                  "brk_n", "brk_sw", "brk_wh", "off_n", "off_sw", "off_wh",
                  "edge_n", "fp_n", "fp_sw", "fp_s",
                  "ts_n", "ts_sw", "ts_wh",
                  "f32_n", "f32_z", "f32_b", "f32_sw", "f32_wh",
                  # v7 audit wave (2026-07-14): 2K x breaking cell
                  "ts_brk_n", "ts_brk_sw", "ts_brk_wh",
                  # v8 damage-on-contact wave (2026-07-15): xwOBA-per-BBE
                  # sums by velo band / pitch class + the 2K x elite-velo
                  # cell — NaN under a pre-v8 file like every schema wave
                  "con_n", "con_xw",
                  "fblo_bip", "fblo_xw", "fbmid_bip", "fbmid_xw",
                  "fb95_bip", "fb95_xw",
                  "brk_bip", "brk_xw", "off_bip", "off_xw",
                  "ts_fb95_n", "ts_fb95_sw", "ts_fb95_wh"):
            if c not in d.columns:
                d[c] = np.nan
        # fastball-bucket swings/whiffs are derived (NaN pre-backfill)
        d["fbk_sw"] = d["sw_n"] - d["brk_sw"] - d["off_sw"]
        d["fbk_wh"] = d["wh_n"] - d["brk_wh"] - d["off_wh"]
        # v8: fastball-remainder class damage, derived the same way
        d["fbk_bip"] = d["con_n"] - d["brk_bip"] - d["off_bip"]
        d["fbk_xw"] = d["con_xw"] - d["brk_xw"] - d["off_xw"]
    if pdp is not None:                # pitcher-only v5 dispersion sums
        for c in ("fb_v2", "rp_n", "rp_x", "rp_x2", "rp_z", "rp_z2",
                  # v6 sequencing/count-state/movement sums (2026-07-14):
                  # NaN under a pre-v6 file so every derived feature gates
                  # to NaN the same way as the earlier schema waves
                  "c02_n", "c02_w", "ah_n", "ah_brk", "ah_off",
                  "bh_n", "bh_brk", "bh_off", "tr_n", "tr_same", "tr_fbbrk",
                  "ivb_n", "ivb_sum", "fade_w", "fade_num",
                  # v7 audit wave (2026-07-14): stretch split, perceived
                  # velo, per-class release centroids, breaking movement
                  "fbstr_n", "fbstr_v", "fbe_n", "fbe_sum",
                  "rpf_n", "rpf_x", "rpf_z", "rpf_x2", "rpf_z2",
                  "rpb_n", "rpb_x", "rpb_z", "rpb_x2", "rpb_z2",
                  "brkmov_n", "brkmov_sum"):
            if c not in pdp.columns:
                pdp[c] = np.nan
        # spin (breaking+offspeed) usage by count state — the derived sums
        # the ahsp/bhsp shrinkage reads (csw_n idiom)
        pdp["_ah_sp"] = pdp["ah_brk"] + pdp["ah_off"]
        pdp["_bh_sp"] = pdp["bh_brk"] + pdp["bh_off"]
    sprint = _opt("mlb_sprint_speed.csv")
    oaa = _alias_renamed_teams(_opt("mlb_oaa.csv"))
    oaa_players = _opt("mlb_oaa_players.csv")   # per-fielder OAA (2016+)
    baserun = _opt("mlb_baserunning.csv")       # runner run value (2016+)
    weather = _opt("mlb_weather.csv")           # humidity/pressure per game
    if weather is not None:
        for c in ("Humidity", "Pressure", "Precip"):
            weather[c] = pd.to_numeric(weather[c], errors="coerce")
    umps = _opt("mlb_umpires.csv", parse_dates=["Date"])
    bat_track = _opt("mlb_bat_tracking.csv")   # bat speed / swing (2023+ only)
    # team battery quality (2026-07-15): playing-time-weighted catcher
    # framing / CS value / pop time per (Year, Team) — the serving-safe
    # grain (tonight's catcher is unknown; the team battery is)
    cat = _opt("mlb_catchers_team.csv")
    if cat is not None:
        # era-centered pop time (league mean per Year): pop improved
        # ~0.05s 2015->2025 — comparable to the cross-team spread — so the
        # SB-chain collision needs the same prior-season-league centering
        # as sb_chain_env's lg_sb27_prior (regime drift must not flip the
        # product's sign era-wide). Computed HERE so both paths share it —
        # and BEFORE the rename aliasing, so duplicated rows can't
        # double-count a franchise in the league mean.
        cat["PopC"] = (cat["PopTime"]
                       - cat.groupby("Year")["PopTime"].transform("mean"))
        cat = _alias_renamed_teams(cat)
    # paired IL stints (2026-07-15): the layoff CAUSE the days-rest gap
    # features cannot see
    il = _opt("mlb_il.csv", parse_dates=["PlaceDate", "ActDate"])
    # MiLB translated priors — a joblib artifact, not a CSV; optional like
    # bip so a tree without the PA-sim program still builds frames
    milb_path = Path(__file__).resolve().parent / "artifacts" / \
        "milb_priors.joblib"
    if milb_path.exists():
        import joblib as _jl
        _m = _jl.load(milb_path)
        milb = {k: _m[k]["serve"] for k in ("bat", "pit")}
    else:
        milb = None

    # 7-inning doubleheaders (2020-21) bias per-game rates; flag to exclude.
    outs_per_game = gp.groupby("GamePk")["Outs"].sum()
    short = set(outs_per_game[outs_per_game < 45].index) & set(
        games.loc[games["Season"] <= 2021, "GamePk"]
    )
    games["ShortGame"] = games["GamePk"].isin(short)

    for df in (games,):
        df["Temp"] = pd.to_numeric(df["Temp"], errors="coerce")
        df["WindSpeed"] = pd.to_numeric(df["WindSpeed"], errors="coerce")
        df["WindDir"] = df["WindDir"].fillna("").str.strip().str.title()
        df["Condition"] = df["Condition"].fillna("").str.strip().str.title()
    raw = dict(games=games, gb=gb, gp=gp, rosters=rosters, parks=parks,
               hr=hr, ars_p=ars_p, ars_b=ars_b, hands=hands,
               bat_season=bat_season, pit_season=pit_season, bip=bip,
               pdp=pdp, pdb=pdb, sprint=sprint, oaa=oaa,
               oaa_players=oaa_players, baserun=baserun, weather=weather,
               umps=umps, bat_track=bat_track, milb=milb, cat=cat, il=il)
    raw["gb"] = annotate_opp_hand(gb, gp, hands)
    return raw


def annotate_opp_hand(gb, gp, hands):
    """Add opp_hand to each batter-game row: the throwing hand of the
    OPPOSING STARTER that game (who faces most of a batter's PAs). This is
    what the platoon-split features condition on. Shared by the training
    and inference paths so both see identical history."""
    throws = hands.set_index("PlayerId")["Throws"]
    st = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]].copy()
    st["opp_hand"] = st["PlayerId"].map(throws).replace("", np.nan)
    st = st.rename(columns={"Team": "Opponent"})[["GamePk", "Opponent", "opp_hand"]]
    return gb.merge(st, on=["GamePk", "Opponent"], how="left")


# ------------------------------------------------- vectorized as-of tables

BAT_STATS = ["PA", "AB", "H", "HR", "TB", "SO", "BB",
             "SB", "CS", "R", "RBI", "2B", "3B", "HBP", "IBB",
             "HRR2", "HRR3", "GIDP", "SF",
             "HRR4", "RBI2", "RUN2", "LOB"]     # audit wave 2026-07-14
PIT_STATS = ["BF", "HR", "SO", "BB", "Outs", "ER", "H", "Strikes", "NP",
             "WP", "BK", "HBP"]                 # audit wave: chaos channel
VSH_STATS = ["PA", "HR", "TB", "SO"]  # platoon splits track these
LOC_STATS = ["PA", "H", "HR", "TB", "SO"]  # home/away venue splits
ROLL_STATS = ["PA", "HR", "H", "TB", "SO", "SB", "R", "RBI"]  # rolling sums


def _snap_to_day_start(df, keys, cols):
    """Doubleheader parity: row-order as-of features let game 2 of a same-day
    doubleheader see game 1, but the inference path only knows data from
    strictly before the DATE. Overwrite every same-(entity, day) row with the
    day's first row, whose history is strictly pre-date by construction."""
    return df.groupby([*keys, "Date"], sort=False)[cols].transform("first")


def _batter_asof(gb):
    """Per batter-game row: career/season/rolling form, all as-of (strictly
    pre-DATE, matching Stores.batter_feats)."""
    df = gb.sort_values(["PlayerId", "Date", "GamePk"]).reset_index(drop=True)
    g = df.groupby("PlayerId", sort=False)
    df["g_career"] = g.cumcount()
    for s in BAT_STATS:
        df[f"c_{s}"] = g[s].cumsum() - df[s]
    gs = df.groupby(["PlayerId", "Season"], sort=False)
    df["g_season"] = gs.cumcount()
    for s in BAT_STATS:
        df[f"s_{s}"] = gs[s].cumsum() - df[s]
    for w in ROLL_WINDOWS:
        for s in ROLL_STATS:
            df[f"r{w}_{s}"] = g[s].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).sum())
    df["days_rest"] = g["Date"].diff().dt.days
    # fatigue: games in the trailing 7 / 14 calendar days, strictly prior to
    # today (day-start convention, like the other as-of features). A dense
    # stretch tires a bat in ways the game-count windows can't see.
    def _prior_games(day_idx, win):
        v = day_idx.to_numpy()
        out = np.empty(len(v))
        lo = 0
        for i in range(len(v)):
            while v[i] - v[lo] > win:
                lo += 1
            hi = i                       # exclude same-day games (day-start)
            while hi > lo and v[hi - 1] == v[i]:
                hi -= 1
            out[i] = hi - lo
        return pd.Series(out, index=day_idx.index)
    _gd = ((df["Date"] - DECAY_EPOCH).dt.days).astype("float64").groupby(
        df["PlayerId"], sort=False)
    df["g_l7d"] = _gd.transform(lambda s: _prior_games(s, 7))
    df["g_l14d"] = _gd.transform(lambda s: _prior_games(s, 14))
    # decay-weighted as-of sums: sum_j x_j * exp(-lam*(t_i - t_j)) over prior
    # games j, computed as an exp(lam*t) cumsum discounted back to the row's
    # own date (t spans ~2,400 days -> e^18.5, comfortably inside float64)
    t = (df["Date"] - DECAY_EPOCH).dt.days.to_numpy(dtype="float64")
    wup, wdn = np.exp(DECAY_LAM * t), np.exp(-DECAY_LAM * t)
    for s in DECAY_STATS:
        xw = df[s].to_numpy(dtype="float64") * wup
        cs = pd.Series(xw, index=df.index).groupby(df["PlayerId"]).cumsum() - xw
        df[f"_dk_{s}"] = cs * wdn
    # decayed count of games played (weight 1 per game): the denominator of
    # xpa_bat, the batter's decayed PA per game (exposure)
    cs = pd.Series(wup, index=df.index).groupby(df["PlayerId"]).cumsum() - wup
    df["_dk_G"] = cs * wdn
    # platoon splits: career as-of sums in games vs L / vs R opposing starters
    for hand in ("L", "R"):
        mask = df["opp_hand"] == hand
        for s in VSH_STATS:
            tmp = df[s].where(mask, 0)
            df[f"_vs{hand}_{s}"] = tmp.groupby(df["PlayerId"]).cumsum() - tmp
    # home/away venue splits: career as-of sums in home vs road games
    for flag in (0, 1):
        mask = df["Home"] == flag
        for s in LOC_STATS:
            tmp = df[s].where(mask, 0)
            df[f"_loc{flag}_{s}"] = tmp.groupby(df["PlayerId"]).cumsum() - tmp
    # position wear: career as-of counts of games caught / DH'd
    for pos in ("C", "DH"):
        ind = (df["Position"] == pos).astype(float)
        df[f"_pos{pos}_n"] = ind.groupby(df["PlayerId"]).cumsum() - ind
    asof_cols = (["g_career", "g_season", "days_rest"]
                 + [f"c_{s}" for s in BAT_STATS] + [f"s_{s}" for s in BAT_STATS]
                 + [f"r{w}_{s}" for w in ROLL_WINDOWS for s in ROLL_STATS]
                 + [f"_dk_{s}" for s in DECAY_STATS] + ["_dk_G"]
                 + [f"_vs{h}_{s}" for h in ("L", "R") for s in VSH_STATS]
                 + [f"_loc{f}_{s}" for f in (0, 1) for s in LOC_STATS]
                 + ["_posC_n", "_posDH_n", "g_l7d", "g_l14d"])
    df[asof_cols] = _snap_to_day_start(df, ["PlayerId"], asof_cols)
    return df


def _starter_asof(gp):
    """Per start row: career/season/rolling starter form, as-of."""
    st = gp[gp["GS"] == 1].sort_values(["PlayerId", "Date", "GamePk"]).reset_index(drop=True)
    g = st.groupby("PlayerId", sort=False)
    st["p_starts_career"] = g.cumcount()
    for s in PIT_STATS:
        st[f"pc_{s}"] = g[s].cumsum() - st[s]
    gs = st.groupby(["PlayerId", "Season"], sort=False)
    st["p_starts_season"] = gs.cumcount()
    for s in PIT_STATS:
        st[f"ps_{s}"] = gs[s].cumsum() - st[s]
    for s in ["BF", "HR", "SO", "BB", "H", "WP", "BK", "HBP"]:
        st[f"p5_{s}"] = g[s].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    st["p_days_rest"] = g["Date"].diff().dt.days
    # fatigue: pitch counts in the last start and last three starts
    st["p_np_last"] = g["NP"].shift(1)
    st["p_np_l3"] = g["NP"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    # outing shape (2026-07-14 #16): career as-of sum of Outs^2 (with
    # pc_Outs/p_starts_career it rebuilds the outing-length SD via the
    # shared outs_sd_from_sums) and the share of the last SHORT_SHARE_WIN
    # starts at <= SHORT_START_OUTS outs (opener/bulk/quick-hook detection)
    _o2 = st["Outs"] ** 2
    st["_c_Outs2"] = _o2.groupby(st["PlayerId"]).cumsum() - _o2
    _short = (st["Outs"] <= SHORT_START_OUTS).astype(float)
    st["p_short_share"] = _short.groupby(st["PlayerId"]).transform(
        lambda x: x.shift(1).rolling(SHORT_SHARE_WIN,
                                     min_periods=SHORT_SHARE_MIN).mean())
    # starter venue splits (2026-07-14 #23): career as-of sums in home vs
    # road starts (batter _loc idiom); the shrunk pvloc_* rates for THIS
    # start's venue context are derived after the day-start snap
    for flag in (0, 1):
        mask = st["Home"] == flag
        for s in PVLOC_STATS:
            tmp = st[s].where(mask, 0)
            st[f"_ploc{flag}_{s}"] = tmp.groupby(st["PlayerId"]).cumsum() - tmp
    # a pitcher can't start twice in a day, but snap anyway so any data quirk
    # (suspended/resumed games) can't break train/serve parity
    asof_cols = (["p_starts_career", "p_starts_season", "p_days_rest",
                  "p_np_last", "p_np_l3", "_c_Outs2", "p_short_share"]
                 + [f"pc_{s}" for s in PIT_STATS] + [f"ps_{s}" for s in PIT_STATS]
                 + [f"p5_{s}" for s in ["BF", "HR", "SO", "BB", "H",
                                        "WP", "BK", "HBP"]]
                 + [f"_ploc{f}_{s}" for f in (0, 1) for s in PVLOC_STATS])
    st[asof_cols] = _snap_to_day_start(st, ["PlayerId"], asof_cols)
    st["p_outs_sd"] = outs_sd_from_sums(st["p_starts_career"], st["pc_Outs"],
                                        st["_c_Outs2"])
    for s in PVLOC_STATS:
        st[f"_pv_{s}"] = np.where(st["Home"] == 1, st[f"_ploc1_{s}"],
                                  st[f"_ploc0_{s}"])
    for k, v in pvloc_from_sums(st["_pv_BF"], st["_pv_HR"], st["_pv_SO"],
                                st["_pv_ER"], st["_pv_Outs"]).items():
        st[k] = v
    return st


def _daily_cum(df, keys, stats, date_col="Date"):
    """Per-key daily totals -> cumulative as-of table for merge_asof.

    Returned rows are keyed (keys..., Date) with cum_<stat> = totals through
    that date INCLUSIVE; consumers merge with allow_exact_matches=False so a
    game on date D sees totals through D-1.
    """
    day = df.groupby([*keys, date_col], as_index=False)[stats].sum()
    day = day.sort_values([*keys, date_col])
    g = day.groupby(list(keys), sort=False)
    for s in stats:
        day[f"cum_{s}"] = g[s].cumsum()
    day["cum_n"] = g.cumcount() + 1
    return day.drop(columns=stats)


def _team_offense_table(gb):
    # AB/H/TB/HBP joined 2026-07-14 (#19): the team's season OBP/SLG norm is
    # the baseline the posted-lineup quality gap is measured against
    return _daily_cum(gb, ["Team", "Season"],
                      ["PA", "HR", "R", "SO", "BB", "AB", "H", "TB", "HBP"])


def _team_offense_loc_table(gb):
    """Team offense split by venue context (home vs road): park-driven teams
    (Coors above all) hit very differently at home, and the one cumulative
    table can't express that."""
    return _daily_cum(gb, ["Team", "Season", "Home"], ["PA", "HR", "R"])


def _bullpen_table(gp):
    pen = gp[gp["GS"] == 0]
    return _daily_cum(pen, ["Team", "Season"],
                      ["BF", "HR", "SO", "BB", "ER", "Outs", "H"])


def _bullpen_hl_table(gp):
    """High-leverage bullpen quality: only relief appearances that finished
    the game or earned a save/hold (the arms that actually pitch the 7th-9th
    of close games). The flat cumulative table weights mop-up innings
    equally; these are the innings a batter prop actually meets late."""
    pen = gp[(gp["GS"] == 0)
             & ((gp["SV"] == 1) | (gp["HLD"] == 1) | (gp["GF"] == 1))]
    return _daily_cum(pen, ["Team", "Season"], ["BF", "SO", "ER", "Outs"])


PEN_FATIGUE_DAYS = 3


def _pen_fatigue_table(gp):
    """Team bullpen pitches thrown over the trailing PEN_FATIGUE_DAYS days,
    strictly before each date (one row per team per calendar day). A worn
    bullpen is a late-innings run/hit signal the season-cumulative rates
    can't see. The index runs a week past the last data date so tomorrow's
    slate still resolves at predict time."""
    pen = gp[gp["GS"] == 0]
    day = pen.pivot_table(index="Date", columns="Team", values="NP",
                          aggfunc="sum")
    idx = pd.date_range(day.index.min(), day.index.max() + pd.Timedelta(days=7),
                        freq="D")
    day = day.reindex(idx).fillna(0.0)
    roll = day.rolling(f"{PEN_FATIGUE_DAYS}D").sum().shift(1)
    out = roll.stack().rename("pen_np_l3").reset_index()
    out.columns = ["Date", "Team", "pen_np_l3"]
    return out


def _pen_unavail_table(gp):
    """Backlog #22 (audit-wave rebuild): per (Team, Date) — quality-
    weighted share of the team's high-leverage relief capacity likely
    UNAVAILABLE tonight: arms that pitched on BOTH of the two previous
    calendar days (bullpen convention — almost never a third straight
    day). Weight = the arm's as-of season SV+HLD; denominator = the
    team's as-of season total, so the value is a [0,1] share (NaN until
    the team banks any saves/holds — early April). The daily index runs a
    week past the last data date so tomorrow's slate resolves
    (pen-fatigue idiom); pen_np_l3 is aggregate fatigue, this is
    arm-specific availability."""
    pen = gp[gp["GS"] == 0].copy()
    pen["_hl"] = (pd.to_numeric(pen["SV"], errors="coerce").fillna(0.0)
                  + pd.to_numeric(pen["HLD"], errors="coerce").fillna(0.0))
    # per (arm, day) appearance rows with the arm's strictly-prior season
    # SV+HLD weight (weight travels with a traded arm; clip below guards
    # the cross-team edge)
    day = (pen.groupby(["PlayerId", "Team", "Season", "Date"],
                       as_index=False).agg(_hl=("_hl", "sum")))
    day = day.sort_values(["PlayerId", "Season", "Date"])
    g = day.groupby(["PlayerId", "Season"], sort=False)
    day["_w"] = g["_hl"].cumsum() - day["_hl"]
    # arms that pitched BOTH of the two previous days, keyed to tonight
    a1 = day[["Team", "Season", "PlayerId", "Date", "_w"]].copy()
    a1["Date"] = a1["Date"] + pd.Timedelta(days=1)
    a2 = day[["Team", "Season", "PlayerId", "Date"]].copy()
    a2["Date"] = a2["Date"] + pd.Timedelta(days=2)
    both = a1.merge(a2, on=["Team", "Season", "PlayerId", "Date"])
    num = (both.groupby(["Team", "Date"], as_index=False)
           .agg(_unav=("_w", "sum")))
    # team's strictly-prior season SV+HLD on a full daily index (+7d)
    tw = pen.pivot_table(index="Date", columns="Team", values="_hl",
                         aggfunc="sum")
    idx = pd.date_range(tw.index.min(),
                        tw.index.max() + pd.Timedelta(days=7), freq="D")
    tw = tw.reindex(idx).fillna(0.0)
    prior = tw.groupby(tw.index.year).cumsum() - tw
    out = prior.stack().rename("_team_w").reset_index()
    out.columns = ["Date", "Team", "_team_w"]
    out = out.merge(num, on=["Team", "Date"], how="left")
    out["pen_hl_unavail"] = ((out["_unav"].fillna(0.0)
                              / out["_team_w"].where(out["_team_w"] > 0))
                             .clip(0.0, 1.0))
    return out[["Date", "Team", "pen_hl_unavail"]]


def _team_defense_table(gp):
    """Unearned-run rate allowed, per team-season (as-of): (R - ER) * 27 /
    outs across the whole staff. The only defense-quality signal in the
    data; everything else assumes all gloves are equal."""
    return _daily_cum(gp, ["Team", "Season"], ["R", "ER", "Outs"])


def _team_starter_outs_table(gp):
    """The manager's leash (2026-07-14 #16): the TEAM's as-of season average
    starter outs per game — organizational quick-hook/bulk policy, distinct
    from any one pitcher's own history."""
    return _daily_cum(gp[gp["GS"] == 1], ["Team", "Season"], ["Outs"])


def _team_sched_table(games, parks):
    """Per (GamePk, Team): schedule context off the team's PREVIOUS game on
    an earlier DATE (day-start convention, so game 2 of a doubleheader sees
    the same 'previous game' the serving path can see): day_after_night,
    travel_km (great-circle between venues), tz_delta (approx hours, east
    positive) — plus the doubleheader flags is_dh / dh_game2 (two GamePks,
    same team+date; game 2 = the higher GamePk of the day). 2026-07-14
    #18/#24."""
    dn = games["DayNight"].astype(str).str.strip().str.title().str[:1]
    dn = dn.where(dn.isin(("D", "N")), "")     # matches the scalar _dn1
    rows = []
    for tc in ("AwayTeam", "HomeTeam"):
        rows.append(pd.DataFrame({
            "GamePk": games["GamePk"], "Team": games[tc],
            "Date": games["Date"], "Venue": games["Venue"],
            "DayNight": dn}))
    t = pd.concat(rows, ignore_index=True).sort_values(
        ["Team", "Date", "GamePk"], kind="mergesort").reset_index(drop=True)
    # doubleheader flags: games per (Team, Date), ordered by GamePk
    grp_day = t.groupby(["Team", "Date"], sort=False)
    n_day = grp_day["GamePk"].transform("size")
    t["is_dh"] = (n_day >= 2).astype(float)
    t["dh_game2"] = (grp_day.cumcount() >= 1).astype(float)
    # previous game on an EARLIER date: take each team-day's FIRST row's
    # previous row (day-start), so both DH games share one predecessor
    g = t.groupby("Team", sort=False)
    for c in ("Date", "Venue", "DayNight"):
        t[f"_prev_{c}"] = g[c].shift(1)
    prev_cols = ["_prev_Date", "_prev_Venue", "_prev_DayNight"]
    t[prev_cols] = _snap_to_day_start(t, ["Team"], prev_cols)
    coords = parks.set_index("Ballpark")[["Lat", "Lon"]]
    la_p = t["_prev_Venue"].map(coords["Lat"])
    lo_p = t["_prev_Venue"].map(coords["Lon"])
    la_n = t["Venue"].map(coords["Lat"])
    lo_n = t["Venue"].map(coords["Lon"])
    t["travel_km"] = haversine_km(la_p, lo_p, la_n, lo_n)
    t["tz_delta"] = (lo_n - lo_p) / _TZ_DEG_PER_HR
    gap1 = (t["Date"] - t["_prev_Date"]).dt.days == 1
    known = (t["_prev_DayNight"].isin(("D", "N"))
             & t["DayNight"].isin(("D", "N")))
    t["day_after_night"] = np.where(
        known & t["_prev_Date"].notna(),
        (gap1 & (t["_prev_DayNight"] == "N")
         & (t["DayNight"] == "D")).astype(float), np.nan)
    return t[["GamePk", "Team", "day_after_night", "travel_km", "tz_delta",
              "is_dh", "dh_game2"]]


# BaseRuns team-run expectation from trailing components — the sequencing-
# luck residual's denominator model. Standard BsR: A = baserunners,
# B = advancement, C = outs, D = HR; E[R] = A*B/(B+C) + D.
def _baseruns(h, bb, hbp, tb, hr, ab):
    a = h + bb + hbp - hr
    b = 1.02 * (1.4 * tb - 0.6 * h - 3.0 * hr + 0.1 * (bb + hbp))
    c = ab - h
    return a * b / (b + c) + hr


def _team_bsr_table(gb):
    """Per (Team, Date): trailing-30-day cluster-luck residual — actual runs
    minus the BaseRuns expectation from components, per game, strictly
    before the date (2026-07-14 #20). Positive = the offense has been
    sequencing-lucky and is due to regress. NaN under BSR_MIN_G games. The
    index runs a week past the last data date so tomorrow's slate resolves
    (pen-fatigue idiom)."""
    stats = ["H", "BB", "HBP", "TB", "HR", "AB", "R"]
    day = gb.groupby(["Team", "Date"])[stats].sum()
    day["G"] = gb.groupby(["Team", "Date"])["GamePk"].nunique()
    wide = day.unstack("Team")
    idx = pd.date_range(wide.index.min(),
                        wide.index.max() + pd.Timedelta(days=7), freq="D")
    wide = wide.reindex(idx).fillna(0.0)
    roll = wide.rolling("30D").sum().shift(1)
    bsr = _baseruns(roll["H"], roll["BB"], roll["HBP"], roll["TB"],
                    roll["HR"], roll["AB"])
    luck = (roll["R"] - bsr) / roll["G"]
    luck = luck.where(roll["G"] >= BSR_MIN_G)
    out = luck.stack().rename("bsr_luck").reset_index()   # NaN rows drop;
    out.columns = ["Date", "Team", "bsr_luck"]            # merges re-NaN them
    return out


def _team_sb_allowed_table(gb):
    """Stolen bases allowed per team, season-cumulative as-of: every SB in
    the batter game logs debits the OPPONENT's battery (pitcher + catcher
    jointly — the logs can't split them). Complements the starter-only
    prior-season psb_* with a current-season, catcher-inclusive rate."""
    d = gb.groupby(["Opponent", "Season", "Date"], as_index=False).agg(
        SB=("SB", "sum"), CS=("CS", "sum"), G=("GamePk", "nunique"))
    d = d.rename(columns={"Opponent": "Team"})
    return _daily_cum(d, ["Team", "Season"], ["SB", "CS", "G"])


def _park_table(gb, games):
    # per-game venue totals for the park factors: HR (legacy) plus runs, hits,
    # doubles, triples (2026-07-14, the H3 triple head's venue rider — deep
    # gaps and quirky walls) and total bases, so the offensive props get the
    # run-environment signal a lone HR factor misses.
    stats = ["HR", "R", "H", "2B", "3B", "TB"]
    per_game = gb.groupby("GamePk", as_index=False)[stats].sum()
    gv = games.merge(per_game, on="GamePk", how="left")
    gv[stats] = gv[stats].fillna(0)
    # audit wave (rank 32): per-game runs SUM-OF-SQUARES so consumers can
    # rebuild the venue variance-to-mean ratio (the second moment of runs —
    # nothing else in the pipeline encodes one) from cumulative sums.
    gv["R2"] = gv["R"] ** 2
    return _daily_cum(gv, ["Venue"], stats + ["R2"])


# audit wave (rank 32): venue run variance-to-mean ratio, EB-shrunk toward
# the league VMR. The trait is real but weak (split-half ~0.15), so the
# shrinkage is heavy (measured noise/signal implies K~900, not the K~300
# the proposal guessed). VMR0 = pooled league VMR measured on the file.
PARK_VMR0, PARK_VMR_K, PARK_VMR_MIN_N = 2.278, 900.0, 30


def park_vmr_from_sums(n, r, r2):
    """Venue runs variance-to-mean ratio from cumulative (n games, sum R,
    sum R^2), EB-shrunk toward the league VMR0. Shared by the game frame
    and Stores.park so both paths compute identically; NaN under
    PARK_VMR_MIN_N games."""
    n = np.asarray(n, dtype="float64")
    r = np.asarray(r, dtype="float64")
    r2 = np.asarray(r2, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = r / n
        var = r2 / n - mean ** 2
        vmr_raw = var / mean
        shr = (n * vmr_raw + PARK_VMR_K * PARK_VMR0) / (n + PARK_VMR_K)
    return np.where(n >= PARK_VMR_MIN_N, shr, np.nan)


def _park_hand_hr_table(gb, games, hands):
    """Per (Venue, Date): as-of cumulative HR and PA split by BATTER bats-hand
    (L/R), wide (phh_{L,R}_{HR,PA}). Feeds the handedness-specific park HR edge
    that the hand-agnostic park_hr_pg misses — short-RF parks help LHB in a way
    fence distance and the overall park factor don't capture. Switch hitters are
    dropped from the build (their effective hand depends on the pitcher);
    consumers key the edge on eff_hand and merge_asof by Venue (exclusive)."""
    bats = (hands.dropna(subset=["PlayerId"]).drop_duplicates("PlayerId")
            .set_index("PlayerId")["Bats"])
    g = gb[["GamePk", "PlayerId", "Date", "HR", "PA"]].copy()
    g["bh"] = g["PlayerId"].map(bats)
    g = g[g["bh"].isin(("L", "R"))]
    g = g.merge(games[["GamePk", "Venue"]], on="GamePk", how="left")
    g = g.dropna(subset=["Venue"])
    day = g.groupby(["Venue", "Date", "bh"], as_index=False)[["HR", "PA"]].sum()
    wide = day.pivot_table(index=["Venue", "Date"], columns="bh",
                           values=["HR", "PA"], fill_value=0.0)
    wide.columns = [f"{h}_{s}" for s, h in wide.columns]   # -> L_HR, R_HR, L_PA, R_PA
    wide = wide.reset_index().sort_values(["Venue", "Date"])
    for c in ("L_HR", "L_PA", "R_HR", "R_PA"):
        if c not in wide.columns:
            wide[c] = 0.0
    grp = wide.groupby("Venue", sort=False)
    out = wide[["Venue", "Date"]].copy()
    for c in ("L_HR", "L_PA", "R_HR", "R_PA"):
        out[f"phh_{c}"] = grp[c].cumsum()
    return out


# rbi opportunity, DEEPER-ORDER ISOLATION: mean career OBP of the hitters 3-5
# spots AHEAD in the order (wrapping). ctx_ahead_obp already covers the 1st-2nd
# ahead, so this deliberately isolates ONLY the deeper table those two miss —
# the honest test of whether the order beyond the immediate two men on carries
# RBI signal of its own. (The earlier full-order decayed version was dominated
# by the -1/-2 terms = a ctx_ahead_obp duplicate, and came back flat.) Equal
# weights => a plain NaN-skipping mean. Shared by the vectorized frame and the
# serving path so the two agree (selftest parity).
RBI_OPP_AHEAD = ((-3, 1.0), (-4, 1.0), (-5, 1.0))


def rbi_opp_from_slots(obp_by_slot, slot):
    """Weighted, NaN-skipping mean of the career OBP of the hitters ahead of
    `slot` (obp_by_slot maps slot -> OBP), renormalized over the hitters that
    have a value. Serving path; the frame computes the same value vectorized."""
    num = den = 0.0
    for off, w in RBI_OPP_AHEAD:
        v = obp_by_slot.get(((slot + off - 1) % 9) + 1)
        if v is not None and not pd.isna(v):
            num += w * v
            den += w
    return num / den if den > 0 else np.nan


ENV_COLS = ["lg_hr_pa", "lg_r_pa", "lg_k_pa", "lg_bb_pa", "lg_sb_pa"]


def _league_env_table(gb):
    """League-wide trailing-30-day offensive environment, one row per
    calendar day. The row for date D covers the 30 days ENDING D-1
    (strictly pre-game), so models can track the CURRENT run environment
    (hot summer, a lively ball) instead of assuming the training years'
    average — that lag showed up as June/July under-prediction in the
    2026 drift eval. NaN when the window is too thin (offseason, opening
    days) so the trees fall back to their other features."""
    stats = ["PA", "HR", "R", "SO", "BB", "SB"]
    day = gb.groupby("Date")[stats].sum().asfreq("D", fill_value=0)
    roll = day.rolling("30D").sum().shift(1)
    out = pd.DataFrame(index=roll.index)
    out["lg_hr_pa"] = roll["HR"] / roll["PA"]
    out["lg_r_pa"] = roll["R"] / roll["PA"]
    out["lg_k_pa"] = roll["SO"] / roll["PA"]
    out["lg_bb_pa"] = roll["BB"] / roll["PA"]
    out["lg_sb_pa"] = roll["SB"] / roll["PA"]
    out[roll["PA"] < 2000] = np.nan
    return out.reset_index()


def _team_results_rows(games):
    """One row per team per game: runs for/against and win flag, sorted for
    as-of lookups. Basis for both season records and recent form."""
    rows = []
    for team_col, score, opp_score in (("AwayTeam", "AwayScore", "HomeScore"),
                                       ("HomeTeam", "HomeScore", "AwayScore")):
        rows.append(pd.DataFrame({
            "GamePk": games["GamePk"], "Team": games[team_col],
            "Season": games["Season"], "Date": games["Date"],
            "RF": pd.to_numeric(games[score], errors="coerce"),
            "RA": pd.to_numeric(games[opp_score], errors="coerce")}))
    res = pd.concat(rows, ignore_index=True).dropna(subset=["RF", "RA"])
    res["W"] = (res["RF"] > res["RA"]).astype(int)
    return res.sort_values(["Team", "Season", "Date", "GamePk"]).reset_index(drop=True)


def _team_results_table(games):
    """Per team per day: cumulative season wins, runs scored and allowed
    (as-of via merge_asof / _cum). Feeds the winner model: win%, run
    differential, and pythagorean expectation are the strongest plain
    team-strength signals and none of them existed in the game frame."""
    return _daily_cum(_team_results_rows(games).drop(columns=["GamePk"]),
                      ["Team", "Season"], ["W", "RF", "RA"])


# Elo: cross-season team strength. K and home-field are standard MLB
# values (538-style); carryover regresses 30% toward the mean each winter.
ELO_BASE, ELO_K, ELO_HFA, ELO_CARRY = 1500.0, 4.0, 24.0, 0.7


def elo_expected(elo_home, elo_away):
    """Expected home-win probability from pre-game Elo ratings."""
    return 1.0 / (1.0 + 10.0 ** (-((elo_home - elo_away + ELO_HFA) / 400.0)))


def build_elo(games):
    """One chronological Elo pass over every game. Returns
    (per-game pre-ratings keyed by GamePk, per-team post-game history).

    This is the winner model's answer to April: win%/run-diff/form all
    reset to NaN each season, but true team strength carries over the
    winter. Elo starts each season at 1500 + ELO_CARRY * (last - 1500)."""
    g = games.copy()
    g["AwayR"] = pd.to_numeric(g["AwayScore"], errors="coerce")
    g["HomeR"] = pd.to_numeric(g["HomeScore"], errors="coerce")
    g = g.dropna(subset=["AwayR", "HomeR"]).sort_values(["Date", "GamePk"])
    state = {}  # team -> (season, elo after last game)
    pre, hist = [], []

    def rating(team, season):
        s_prev, e = state.get(team, (None, ELO_BASE))
        if s_prev is not None and s_prev != season:
            e = ELO_BASE + ELO_CARRY * (e - ELO_BASE)
        return e

    cur_date, day_start = None, {}
    for r in g.itertuples():
        # feature ratings are DAY-START: inference (Stores.team_elo) can only
        # see games strictly before the date, so game 1 of a doubleheader
        # must not leak into game 2's pre-game features
        if r.Date != cur_date:
            cur_date, day_start = r.Date, {}
        for team in (r.AwayTeam, r.HomeTeam):
            day_start.setdefault(team, rating(team, r.Season))
        ea_f, eh_f = day_start[r.AwayTeam], day_start[r.HomeTeam]
        pre.append((r.GamePk, ea_f, eh_f, elo_expected(eh_f, ea_f)))
        # the update chain itself stays game-by-game (running ratings)
        ea = rating(r.AwayTeam, r.Season)
        eh = rating(r.HomeTeam, r.Season)
        exp_h = elo_expected(eh, ea)
        out = 1.0 if r.HomeR > r.AwayR else 0.0
        delta = ELO_K * (out - exp_h)
        state[r.HomeTeam] = (r.Season, eh + delta)
        state[r.AwayTeam] = (r.Season, ea - delta)
        hist.append((r.AwayTeam, r.Season, r.Date, ea - delta))
        hist.append((r.HomeTeam, r.Season, r.Date, eh + delta))
    pre = pd.DataFrame(pre, columns=["GamePk", "away_elo", "home_elo",
                                     "elo_prob_home"])
    hist = pd.DataFrame(hist, columns=["Team", "Season", "Date", "elo_post"])
    return pre, hist


FORM_N, FORM_MIN = 20, 5  # recent form: last 20 games, need at least 5


def _team_form_table(games):
    """Per team per game: win% and run diff over the previous FORM_N games
    (as-of). Season records alone are stale by August; recent form is the
    standard complement for winner models."""
    form = _team_results_rows(games)
    form["rd"] = form["RF"] - form["RA"]
    grp = form.groupby(["Team", "Season"], sort=False)
    form["w20"] = grp["W"].transform(
        lambda x: x.shift(1).rolling(FORM_N, min_periods=FORM_MIN).mean())
    form["rd20"] = grp["rd"].transform(
        lambda x: x.shift(1).rolling(FORM_N, min_periods=FORM_MIN).mean())
    form[["w20", "rd20"]] = _snap_to_day_start(form, ["Team"], ["w20", "rd20"])
    return form[["GamePk", "Team", "w20", "rd20"]]


PYTH_EXP = 1.83  # standard pythagorean exponent for MLB


def _record_feats(cum_w, cum_rf, cum_ra, cum_n):
    rf_p = cum_rf ** PYTH_EXP
    ra_p = cum_ra ** PYTH_EXP
    return {"win_pct": cum_w / cum_n, "rd_pg": (cum_rf - cum_ra) / cum_n,
            "ra_pg": cum_ra / cum_n, "pyth": rf_p / (rf_p + ra_p)}


# Fly-ball carry gain per foot of elevation (~+26 ft at Coors' 5,200 ft):
# HR distances are normalized to sea level so a thin-air 440 doesn't read
# as more raw power than a sea-level 415.
ELEV_DIST_FT = 0.005


def _hr_quality_table(hr, parks):
    """Batter's HR exit velo / launch angle / elevation-adjusted distance
    profile as-of (from the HR log)."""
    h = hr.dropna(subset=["BatterId"]).copy()
    h["BatterId"] = h["BatterId"].astype("int64")
    elev = parks.set_index("Ballpark")["Elevation_ft"]
    h["dist_adj"] = h["Distance"] - ELEV_DIST_FT * h["Ballpark"].map(elev).fillna(0)
    h = h.sort_values(["BatterId", "Date"])
    g = h.groupby("BatterId", sort=False)
    h["cum_n"] = g.cumcount() + 1
    for c, name in [("Exit Velo", "ev"), ("dist_adj", "dist"),
                    ("Angle", "angle")]:
        h[f"cum_{name}"] = g[c].cumsum()
    h["cum_dist_max"] = g["dist_adj"].cummax()
    day = h.groupby(["BatterId", "Date"], as_index=False).last()
    return day[["BatterId", "Date", "cum_n", "cum_ev", "cum_dist",
                "cum_dist_max", "cum_angle"]]


# Statcast contact quality: league priors + stabilization Ks (in batted
# balls) for EB shrinkage of the per-player means/shares. Fixed constants
# (like SHRINK) so training and inference compute identically.
#   name -> (prior, K, numerator day-sum, denominator day-sum)
BIP_SHRINK = {
    "ev":    (88.3, 40, "ev_sum", "ev_n"),    # avg exit velo (mph)
    "la":    (12.7, 60, "la_sum", "la_n"),    # avg launch angle (deg)
    "hh":    (0.39, 60, "hh_n", "ev_n"),      # hard-hit share (EV >= 95)
    "brl":   (0.08, 80, "brl_n", "ev_n"),     # barrel share (Savant LSA 6)
    "xba":   (0.323, 60, "xba_sum", "xba_n"),  # xBA on contact
    "xwoba": (0.371, 60, "xw_sum", "xw_n"),   # xwOBA on contact
    "gb":    (0.43, 60, "gb_n", "n"),         # ground-ball share
    "pull":  (0.436, 60, "pull_n", "hc_n"),   # pull-side share (spray)
    "pullair": (0.166, 80, "pullair_n", "hc_n"),  # pulled fly/line share
    # 2026-07-12: the two BBType shares the gb/pullair pair misses — line
    # drives (~.650 BABIP, THE hit-tool contact outcome) and popups
    # (near-automatic outs). Priors measured on the 2015-2026 BIP file.
    "ld":    (0.246, 60, "ld_n", "n"),        # line-drive share
    "pu":    (0.071, 80, "pu_n", "n"),        # popup share
    # mean sea-level-adjusted FLY-BALL distance: the uncensored power
    # measure (the HR log only records each batter's best contact); the
    # pitcher mirror = how far he gets hit in the air. Prior measured.
    "flyd":  (315.0, 40, "fld_sum", "fld_n"),
    # audit wave (rank 13): the oppo-side spray bins (mirror of pull;
    # a stable independent skill the one-sided encoding missed) and the
    # 105+ EV tail share. Priors measured on the full BIP file.
    "oppo":    (0.282, 60, "oppo_n", "hc_n"),      # oppo-side share
    "oppoair": (0.161, 80, "oppoair_n", "hc_n"),   # oppo fly/line share
    "ev105":   (0.084, 80, "ev105_n", "ev_n"),     # crushed share (EV>=105)
}
# 90-day decayed too; xba joined 07-12 for the starter BABIP-luck residual
BIP_DECAYED = ("ev", "brl", "xwoba", "gb", "pullair", "ld", "xba", "flyd",
               "oppoair", "ev105")

# pitch-level daily aggregates (scrape_pitches.py): swing-and-miss and
# plate discipline. Same (prior, K, numerator, denominator) convention;
# priors are league rates measured on the 2020-2026 file.
PD_SHRINK = {
    "swstr": (0.112, 250, "wh_n", "n"),      # swinging strikes per pitch
    "csw":   (0.275, 250, "csw_n", "n"),     # called strikes + whiffs
    "wsw":   (0.235, 120, "wh_n", "sw_n"),   # whiffs per swing
    "chase": (0.286, 150, "oz_sw", "oz_n"),  # out-of-zone swing share
    "zone":  (0.488, 250, "z_n", "n"),       # in-zone pitch share
    "fbv":   (93.9, 50, "fb_v", "fb_n"),     # avg fastball velo (FF+SI)
    # zone-split whiffs + elite-velo buckets (2026-07-12 scrape schema):
    # in-zone whiff per swing (1 - zone contact, the most stable hit-tool
    # skill) and whiff per swing against 95+ mph fastballs (the velocity
    # weakness tonight's starter either can or cannot exploit)
    # v3-v5 priors below MEASURED on the full re-aggregated 2015-2026 file
    # (7.86M pitches, calibrate_priors.py, 2026-07-12 ~20:45)
    "zwsw":  (0.154, 120, "z_wh", "z_sw"),   # in-zone whiffs per swing
    "fb95wh": (0.195, 80, "fb95_wh", "fb95_sw"),  # whiff/swing vs 95+ fb
    # v3 scrape schema (2026-07-12): whiff splits by pitch class (fastball
    # bucket = everything not breaking/offspeed), usage shares, shadow-band
    # location share (command proxy), first-pitch strike/swing tendencies
    "brkwh": (0.321, 100, "brk_wh", "brk_sw"),  # whiff/swing vs breaking
    "offwh": (0.299, 80, "off_wh", "off_sw"),   # whiff/swing vs offspeed
    "fbwh":  (0.171, 120, "fbk_wh", "fbk_sw"),  # whiff/swing vs fastballs
    "brk":   (0.286, 150, "brk_n", "n"),        # breaking usage share
    "off":   (0.127, 150, "off_n", "n"),        # offspeed usage share
    "edge":  (0.405, 200, "edge_n", "n"),       # shadow-band pitch share
    "fps":   (0.609, 60, "fp_s", "fp_n"),       # first-pitch strike share
    "fpsw":  (0.301, 100, "fp_sw", "fp_n"),     # first-pitch swing share
    # v4 graded velocity bands on FF/SI (<92 / 92-95 / 95+), user-pulled
    # forward 07-12 eve: whiff splits per band (the graded version of
    # fb95wh) and the pitcher's usage share per band
    "fblowh": (0.134, 80, "fblo_wh", "fblo_sw"),   # whiff/swing vs <92
    "fbmidwh": (0.161, 80, "fbmid_wh", "fbmid_sw"),  # whiff/swing 92-95
    "fblou": (0.146, 150, "fblo_n", "n"),       # <92 fastball usage
    "fbmidu": (0.217, 150, "fbmid_n", "n"),     # 92-95 fastball usage
    "fb95u": (0.149, 150, "fb95_n", "n"),       # 95+ fastball usage
    # v5 count-leverage splits (user 07-12 eve): two-strike put-away /
    # survival, and the 3-2 payoff pitch — a ball there IS a walk, so
    # f32b = walk conversion (batter) / walks gifted (pitcher), f32z =
    # challenges-or-gives-in
    "tswh": (0.225, 100, "ts_wh", "ts_sw"),     # two-strike whiff/swing
    "f32z": (0.575, 60, "f32_z", "f32_n"),      # 3-2 zone share
    "f32b": (0.224, 60, "f32_b", "f32_n"),      # 3-2 ball (=walk) share
    # v6 sequencing / count-state / movement (2026-07-14 #25-27; priors
    # MEASURED on the full re-aggregated 2015-2026 pitcher file):
    # 0-2 waste rate (share of located 0-2 pitches beyond the shadow band —
    # a putaway plan vs nibbling), ahead/behind spin (brk+off) usage —
    # each side shrinks to ITS OWN league prior so no-history lands on the
    # league's usage shift, not zero — same-class back-to-back share
    # (predictability), fastball->breaking transition share (the tunneling
    # pair), and FF induced vertical break in inches (ride).
    "w02":  (0.540, 80, "c02_w", "c02_n"),      # 0-2 waste rate
    "ahsp": (0.497, 150, "_ah_sp", "ah_n"),     # spin share when ahead
    "bhsp": (0.329, 150, "_bh_sp", "bh_n"),     # spin share when behind
    "trsame": (0.504, 150, "tr_same", "tr_n"),  # same-class repeat share
    "trfbbrk": (0.156, 150, "tr_fbbrk", "tr_n"),  # fb->breaking share
    "ivb":  (15.96, 100, "ivb_sum", "ivb_n"),   # FF induced vert break (in)
    # v7 audit wave (2026-07-14): the passivity axis (zone-swing +
    # two-strike swing — called-K/walk path; priors measured on the full
    # batter file), the two-strike x breaking putaway cell, perceived-velo
    # premium (effective minus release on FF/SI; league prior fits the
    # Hawk-Eye era, drift absorbed by the trees), and breaking-ball
    # movement magnitude (cause axis; ivb idiom, shrink handles thin n)
    "zsw":  (0.672, 120, "z_sw", "z_n"),        # in-zone swing share
    "tssw": (0.604, 100, "ts_sw", "ts_n"),      # two-strike swing share
    "tsbrkwh": (0.30, 80, "ts_brk_wh", "ts_brk_sw"),  # 2K whiff vs breaking
    "effprem": (0.15, 200, "fbe_sum", "fbe_n"),  # perceived-velo premium
    "brkmov": (10.7, 100, "brkmov_sum", "brkmov_n"),  # breaking break (in)
    # v8 damage-on-contact wave (2026-07-15): xwOBA per BBE by velo band /
    # pitch class — the contact half of the velo-band axis (the whiff half
    # shipped v4). K = 60 BBE matching BIP_SHRINK's xwoba; a median batter
    # season has only ~18 BBE vs 95+, so the DECAYED reads are shrink-
    # dominated by design — the career reads carry the collision features.
    # Priors MEASURED on the re-aggregated 2015-2026 batter file
    # (2026-07-15; damage falls as velo rises, lowest on offspeed).
    "fbloxw": (0.389, 60, "fblo_xw", "fblo_bip"),   # xwOBA/BBE vs <92
    "fbmidxw": (0.385, 60, "fbmid_xw", "fbmid_bip"),  # xwOBA/BBE 92-95
    "fb95xw": (0.372, 60, "fb95_xw", "fb95_bip"),   # xwOBA/BBE vs 95+
    "brkxw": (0.355, 60, "brk_xw", "brk_bip"),      # xwOBA/BBE vs breaking
    "offxw": (0.343, 60, "off_xw", "off_bip"),      # xwOBA/BBE vs offspeed
    "fbkxw": (0.381, 60, "fbk_xw", "fbk_bip"),      # xwOBA/BBE vs fastballs
    "tsfb95wh": (0.187, 80, "ts_fb95_wh", "ts_fb95_sw"),  # 2K whiff vs 95+
}

# velo-dispersion / release-scatter gates: effective sample below which
# the sd reads are noise, not skill
FBSD_MIN_N = 30.0
RELSD_MIN_N = 100.0
# in-game velo-fade slope (v6 #26): effective fastballs of per-start-slope
# weight below which the decayed mean slope is noise
FADE_MIN_W = 30.0


def fbv_fade_from_sums(w, num):
    """Decayed weighted-mean per-start velo-fade slope from (weight, slope x
    weight) sums — shared by both paths; NaN under FADE_MIN_W."""
    w = np.asarray(w, dtype="float64")
    num = np.asarray(num, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = num / w
    return np.where(w >= FADE_MIN_W, slope, np.nan)


def velo_sd_from_sums(n, v, v2):
    """Within-pitcher fastball velo SPREAD from (count, sum, sum-of-sq)
    cumulative or decayed sums — the consistency/fatigue signal the mean
    (fbv) misses. Shared by both paths; NaN under FBSD_MIN_N effective
    fastballs."""
    n = np.asarray(n, dtype="float64")
    v = np.asarray(v, dtype="float64")
    v2 = np.asarray(v2, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        var = v2 / n - (v / n) ** 2
        sd = np.sqrt(np.clip(var, 0.0, None))
    return np.where(n >= FBSD_MIN_N, sd, np.nan)


def release_scatter_from_sums(n, x, x2, z, z2):
    """Release-point scatter sqrt(var_x + var_z) from cumulative/decayed
    sums — mechanical repeatability as a command/injury proxy. Shared by
    both paths; NaN under RELSD_MIN_N pitches with coordinates."""
    n = np.asarray(n, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        vx = np.asarray(x2, dtype="float64") / n \
            - (np.asarray(x, dtype="float64") / n) ** 2
        vz = np.asarray(z2, dtype="float64") / n \
            - (np.asarray(z, dtype="float64") / n) ** 2
        sd = np.sqrt(np.clip(vx, 0.0, None) + np.clip(vz, 0.0, None))
    return np.where(n >= RELSD_MIN_N, sd, np.nan)


# v7 audit wave (2026-07-14): base-out conditional velo + per-class
# release geometry, from the v7 scrape sums. Both are shared-by-both-paths
# helpers like velo_sd_from_sums.
STRETCH_K = 300.0            # shrink of the stretch-windup delta toward 0
RELSEP_MIN_N = 30.0          # per-class pitches before centroids mean much
RELSEP_PRIOR, RELSEP_K = 0.165, 100.0   # league-mean separation (ft)


def stretch_vdelta_from_sums(fb_n, fb_v, fbstr_n, fbstr_v):
    """FF/SI velo pitching from the STRETCH (runners on) minus from the
    windup, shrunk toward 0 by stretch-side support — conditional stuff on
    run-scoring states (base-out appears nowhere else in the pipeline).
    Sign = stretch minus windup (league mean ~ +0.08 mph). NaN when either
    split is empty."""
    sn = np.asarray(fbstr_n, dtype="float64")
    sv = np.asarray(fbstr_v, dtype="float64")
    wn = np.asarray(fb_n, dtype="float64") - sn
    wv = np.asarray(fb_v, dtype="float64") - sv
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = sv / sn - wv / wn
    delta = np.where((sn > 0) & (wn > 0), delta, np.nan)
    return delta * sn / (sn + STRETCH_K)


def release_sep_from_sums(fn, fx, fz, bn, bx, bz):
    """Between-class release-centroid separation (fastball-remainder vs
    breaking classes), K-shrunk toward the league-mean separation — the
    arm-slot deception trait the pooled rp_ scatter confounds (r=0.34 with
    it). NaN unless BOTH classes clear RELSEP_MIN_N (centroid gate, lower
    than RELSD_MIN_N since means need less n than SDs)."""
    fn = np.asarray(fn, dtype="float64")
    bn = np.asarray(bn, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        sep = np.hypot(
            np.asarray(bx, dtype="float64") / bn
            - np.asarray(fx, dtype="float64") / fn,
            np.asarray(bz, dtype="float64") / bn
            - np.asarray(fz, dtype="float64") / fn)
    w = np.minimum(fn, bn)
    shr = (sep * w + RELSEP_K * RELSEP_PRIOR) / (w + RELSEP_K)
    return np.where((fn >= RELSEP_MIN_N) & (bn >= RELSEP_MIN_N),
                    shr, np.nan)

# the batter-side plate-discipline set (career + decay both built) and the
# pitcher-side decayed set — shared by the vectorized frame builders and
# the serving Stores so the two paths can never drift apart
PD_BATTER = ("wsw", "chase", "zwsw", "fb95wh", "brkwh", "offwh", "fbwh",
             "fpsw", "fblowh", "fbmidwh", "tswh", "f32b",
             # v7 audit wave: passivity axis + 2K x breaking putaway cell
             "zsw", "tssw", "tsbrkwh",
             # v8 damage-on-contact: xwOBA/BBE by band + class, 2K x 95+
             "fbloxw", "fbmidxw", "fb95xw", "brkxw", "offxw", "fbkxw",
             "tsfb95wh")
PD_PITCHER_D = ("swstr", "csw", "wsw", "chase", "zone", "fbv", "zwsw",
                "brkwh", "offwh", "fbwh", "brk", "off", "edge", "fps",
                "fb95wh", "fblowh", "fbmidwh", "fblou", "fbmidu", "fb95u",
                "tswh", "f32z", "f32b",
                # v6 (2026-07-14): 0-2 waste, ahead/behind spin usage (the
                # shift is derived in add_starter_derived), transition
                # shares, FF ride
                "w02", "ahsp", "bhsp", "trsame", "trfbbrk", "ivb",
                # v7 audit wave: perceived-velo premium + breaking movement
                "effprem", "brkmov",
                # v8 damage-on-contact: xwOBA/BBE ALLOWED by band + class
                "fbloxw", "fbmidxw", "fb95xw", "brkxw", "offxw", "fbkxw")

# spray-angle pull cutoff: hit coordinates -> signed degrees off center
# (negative = LF); a RHB pull = LF side, LHB = RF. Verified empirically:
# RHB ground balls average -14deg, LHB +19deg.
PULL_DEG = 15.0


def _shrunk_rates(get, pre, spec, names=None):
    """Shrunk rates from cumulative (or decayed) sums. `get(day_sum_name)`
    returns a Series (vectorized) or scalar (inference); the same shrinkage
    applies to both because decayed counts are effective sample sizes. NaN
    sums (no history) propagate to NaN."""
    out = {}
    for name in (names or spec):
        prior, k, num, den = spec[name]
        out[f"{pre}_{name}"] = (get(num) + k * prior) / (get(den) + k)
    return out


def bip_feats(get, pre, names=None):
    return _shrunk_rates(get, pre, BIP_SHRINK, names)


def pd_feats(get, pre, names=None):
    return _shrunk_rates(get, pre, PD_SHRINK, names)


def _spray_flags(b):
    """(pull, pull-air, oppo, oppo-air) booleans per batted ball from hit
    coordinates and batter side. Shared by both paths so the cutoff math
    is identical. Oppo (audit wave rank 13) = the mirror bin: beyond
    PULL_DEG toward the opposite field — a stable independent skill
    (YoY 0.51, corr -0.46 with pullair) the pull-only encoding missed."""
    ang = np.degrees(np.arctan2(b["HcX"] - 125.42, 198.27 - b["HcY"]))
    pull = np.where(b["Stand"] == "R", ang < -PULL_DEG,
                    np.where(b["Stand"] == "L", ang > PULL_DEG, False))
    pull = pull & b["HcX"].notna().to_numpy()
    oppo = np.where(b["Stand"] == "R", ang > PULL_DEG,
                    np.where(b["Stand"] == "L", ang < -PULL_DEG, False))
    oppo = oppo & b["HcX"].notna().to_numpy()
    air = b["BBType"].isin(("fly_ball", "line_drive")).to_numpy()
    return pull, pull & air, oppo, oppo & air


def _bip_day_sums(bip, id_col):
    """Per (id, day) sums of contact-quality numerators/denominators. Each
    metric carries its own count: EV/angle/x-stats are missing on a small
    share of batted balls (tracking gaps, especially 2020)."""
    b = bip.dropna(subset=[id_col]).copy()
    b[id_col] = b[id_col].astype("int64")
    ev, la = b["ExitVelo"], b["LaunchAngle"]
    pull, pullair, oppo, oppoair = _spray_flags(b)
    day = pd.DataFrame({
        id_col: b[id_col], "Date": b["Date"],
        "n": 1.0,
        "ev_n": ev.notna().astype(float), "ev_sum": ev.fillna(0.0),
        "hh_n": (ev >= 95).astype(float),
        "brl_n": (b["LSA"] == 6).astype(float),
        "la_n": la.notna().astype(float), "la_sum": la.fillna(0.0),
        "xba_n": b["xBA"].notna().astype(float),
        "xba_sum": b["xBA"].fillna(0.0),
        "xw_n": b["xwOBA"].notna().astype(float),
        "xw_sum": b["xwOBA"].fillna(0.0),
        "gb_n": (b["BBType"] == "ground_ball").astype(float),
        "ld_n": (b["BBType"] == "line_drive").astype(float),
        "pu_n": (b["BBType"] == "popup").astype(float),
        "fld_n": ((b["BBType"] == "fly_ball")
                  & b["DistAdj"].notna()).astype(float),
        "fld_sum": b["DistAdj"].where(b["BBType"] == "fly_ball").fillna(0.0),
        "hc_n": b["HcX"].notna().astype(float),
        "pull_n": pull.astype(float),
        "pullair_n": pullair.astype(float),
        # audit wave (rank 13): oppo spray bin + the 105+ EV tail (a
        # distinct order statistic — hh saturates at 37.5% of contact,
        # brl is LA-window-gated, the mean is mishit-contaminated)
        "oppo_n": oppo.astype(float),
        "oppoair_n": oppoair.astype(float),
        "ev105_n": (ev >= 105).astype(float),
    })
    return day.groupby([id_col, "Date"], as_index=False).sum()


def _cum_decay_table(day, id_col):
    """Per (id, day): inclusive career cumsums (cum_*) and exp-decay cumsums
    (dk_*, stored as exp(lam*t)-weighted sums) of day-level sums. Consumers
    merge_asof with allow_exact_matches=False and must discount dk_* to the
    consuming row's date (multiply by exp(-lam * row date)) before using
    them in shrinkage — the +K*prior terms need real batted-ball units."""
    day = day.sort_values([id_col, "Date"])
    g = day.groupby(id_col, sort=False)
    t = (day["Date"] - DECAY_EPOCH).dt.days.to_numpy(dtype="float64")
    wup = np.exp(DECAY_LAM * t)
    out = day[[id_col, "Date"]].copy()
    cols = [c for c in day.columns if c not in (id_col, "Date")]
    for c in cols:
        out[f"cum_{c}"] = g[c].cumsum()
        xw = day[c].to_numpy(dtype="float64") * wup
        out[f"dk_{c}"] = pd.Series(xw, index=day.index).groupby(
            day[id_col]).cumsum()
    return out


def _bip_table(bip, id_col):
    return _cum_decay_table(_bip_day_sums(bip, id_col), id_col)


# Batter-vs-pitcher direct history (BvP): the batter's prior CONTACT outcomes
# against THIS specific starter, as-of. Statcast BIP is contact-only (no K/BB),
# so BvP speaks to contact props (hr/hit/tb2/single/double/hrr). It is encoded
# as a RESIDUAL off the batter's own as-of contact baseline (bip_xwoba), shrunk
# by pairwise sample size — carrying only the pitcher-specific effect NOT already
# in handedness/arsenal, and sitting at neutral 0 until enough pairwise history
# accrues. Priors/K are set by convention like BIP_SHRINK (NOT swept); the
# residual form is insensitive to K over a broad range. Serving/inference path +
# parity selftest deferred to ship (dev serving loads models.joblib, unaffected).
BVP_K_XW = 30.0        # xwOBA-on-contact residual: effective-sample weight
BVP_K_HR = 50.0        # HR-per-contact residual: effective-sample weight
BVP_HR_PRIOR = 0.046   # league HR per contacted ball (mlb_statcast_bip)


def _bvp_table(bip):
    """Per (BatterId, PitcherId, day): inclusive career cumsums of the pair's
    contact outcomes. Consumers merge_asof with allow_exact_matches=False so a
    game on date D sees pairwise history through D-1 (same-game PAs never leak),
    exactly like every other BIP table."""
    b = bip.dropna(subset=["BatterId", "PitcherId"]).copy()
    b["BatterId"] = b["BatterId"].astype("int64")
    b["PitcherId"] = b["PitcherId"].astype("int64")
    day = pd.DataFrame({
        "BatterId": b["BatterId"], "PitcherId": b["PitcherId"], "Date": b["Date"],
        "n": 1.0,
        "xw_n": b["xwOBA"].notna().astype(float),
        "xw_sum": b["xwOBA"].fillna(0.0),
        "hr_n": (b["Events"] == "home_run").astype(float),
    }).groupby(["BatterId", "PitcherId", "Date"], as_index=False).sum()
    day = day.sort_values(["BatterId", "PitcherId", "Date"])
    g = day.groupby(["BatterId", "PitcherId"], sort=False)
    out = day[["BatterId", "PitcherId", "Date"]].copy()
    for c in ("n", "xw_n", "xw_sum", "hr_n"):
        out[f"bvp_cum_{c}"] = g[c].cumsum()
    return out


# Times-through-the-order decay (2026-07-12): how much worse a starter's
# contact quality allowed gets the 3rd time through vs the 1st. The BIP file
# has no K/BB, so the TTO position is approximated by the rank of the
# pitcher's contact-PAs within the game (AtBat is the game-wide at-bat
# index; ~70% of PAs end in contact, so ~6 contact-PAs per time through).
# Both buckets shrink to the same league prior, so the difference sits at 0
# until real history accrues — the same neutral-until-proven idiom as BvP.
TTO_CONTACT_PER_ORDER = 6.0
TTO_K = 60.0
TTO_XW_PRIOR = 0.371     # league xwOBA on contact (BIP_SHRINK)


def _tto_day_sums(bip):
    """Per (PitcherId, day): xwOBA-on-contact sums split by 1st vs 3rd+ time
    through the order (approximated by contact-PA rank within the game)."""
    b = bip.dropna(subset=["PitcherId"]).copy()
    b["PitcherId"] = b["PitcherId"].astype("int64")
    rank = (b.groupby(["PitcherId", "GamePk"])["AtBat"]
            .rank(method="dense"))
    tto = np.ceil(rank / TTO_CONTACT_PER_ORDER)
    first, third = (tto <= 1).to_numpy(), (tto >= 3).to_numpy()
    xw_ok = b["xwOBA"].notna().to_numpy()
    xw = b["xwOBA"].fillna(0.0).to_numpy()
    day = pd.DataFrame({
        "PitcherId": b["PitcherId"], "Date": b["Date"],
        "xw1_n": (first & xw_ok).astype(float),
        "xw1_sum": np.where(first, xw, 0.0),
        "xw3_n": (third & xw_ok).astype(float),
        "xw3_sum": np.where(third, xw, 0.0),
    })
    return day.groupby(["PitcherId", "Date"], as_index=False).sum()


def _tto_table(bip):
    """Inclusive career cumsums of the TTO day sums (tto_cum_*). Consumers
    merge_asof with allow_exact_matches=False, like every BIP table."""
    day = _tto_day_sums(bip).sort_values(["PitcherId", "Date"])
    g = day.groupby("PitcherId", sort=False)
    out = day[["PitcherId", "Date"]].copy()
    for c in ("xw1_n", "xw1_sum", "xw3_n", "xw3_sum"):
        out[f"tto_cum_{c}"] = g[c].cumsum()
    return out


def tto_decay_from_sums(df):
    """p_tto_decay from the tto_cum_* columns — shared by the frame builders
    and the serving rows (parity). Positive = the starter degrades more than
    league the deeper he goes; 0 with no history."""
    x1 = ((df["tto_cum_xw1_sum"] + TTO_K * TTO_XW_PRIOR)
          / (df["tto_cum_xw1_n"] + TTO_K))
    x3 = ((df["tto_cum_xw3_sum"] + TTO_K * TTO_XW_PRIOR)
          / (df["tto_cum_xw3_n"] + TTO_K))
    return (x3 - x1).fillna(0.0)


def _bip_team_tables(bip, gb, gp):
    """Team-level contact quality, season-cumulative as-of (like every other
    team table): OFFENSE — the team's own batters' xwOBA-on-contact and
    barrel share (season scoring runs through contact quality before it
    shows up in R/PA) — and BULLPEN contact allowed (relief appearances
    only). The BIP file carries no team column, so rows are mapped through
    the game logs on (GamePk, PlayerId)."""
    def day_rows(side_df, id_col, log, log_flag=None):
        d = side_df.dropna(subset=[id_col])
        d = d[["GamePk", id_col, "Date", "xwOBA", "LSA", "ExitVelo"]].copy()
        d[id_col] = d[id_col].astype("int64")
        m = log if log_flag is None else log[log_flag]
        m = m[["GamePk", "PlayerId", "Team", "Season"]].rename(
            columns={"PlayerId": id_col})
        d = d.merge(m, on=["GamePk", id_col], how="inner")
        return pd.DataFrame({
            "Team": d["Team"], "Season": d["Season"], "Date": d["Date"],
            "xw_n": d["xwOBA"].notna().astype(float),
            "xw_sum": d["xwOBA"].fillna(0.0),
            "brl_n": (d["LSA"] == 6).astype(float),
            "ev_n": d["ExitVelo"].notna().astype(float),
        })
    stats = ["xw_n", "xw_sum", "brl_n", "ev_n"]
    off = _daily_cum(day_rows(bip, "BatterId", gb), ["Team", "Season"], stats)
    pen = _daily_cum(day_rows(bip, "PitcherId", gp, gp["GS"] == 0),
                     ["Team", "Season"], stats)
    return off, pen


# Hand-split contact quality: the same xwOBA-on-contact / barrel-share
# shrinkage as BIP_SHRINK, keyed by the OTHER side's hand. This is the
# process-stat platoon split — the outcome-based vsh_* rates need hundreds
# of PA to stabilize; contact quality vs a hand gets there in a fraction.
BVH_METRICS = ("xwoba", "brl")
BVH_SUMS = ("n", "ev_n", "brl_n", "xw_n", "xw_sum")


def _bip_hand_table(bip, id_col, hand_col):
    """Per (id, hand, day): inclusive cumulative sums for the hand-split
    shrinkage — batter contact vs pitcher hand (hand_col='PThrows') or
    pitcher contact allowed by batter side (hand_col='Stand'). Consumers
    merge_asof with allow_exact_matches=False, like every BIP table."""
    b = bip.dropna(subset=[id_col])
    b = b[b[hand_col].isin(("L", "R"))].copy()
    b[id_col] = b[id_col].astype("int64")
    day = pd.DataFrame({
        id_col: b[id_col], hand_col: b[hand_col], "Date": b["Date"],
        "n": 1.0,
        "ev_n": b["ExitVelo"].notna().astype(float),
        "brl_n": (b["LSA"] == 6).astype(float),
        "xw_n": b["xwOBA"].notna().astype(float),
        "xw_sum": b["xwOBA"].fillna(0.0),
    }).groupby([id_col, hand_col, "Date"], as_index=False).sum()
    day = day.sort_values([id_col, hand_col, "Date"])
    g = day.groupby([id_col, hand_col], sort=False)
    out = day[[id_col, hand_col, "Date"]].copy()
    for c in BVH_SUMS:
        out[f"cum_{c}"] = g[c].cumsum()
    return out


def _pitcher_name_map(gp):
    """(Name, Team) -> PlayerId from the pitching log, with a name-only
    fallback for unambiguous names. The HR log identifies the pitcher only
    by name+team; this recovers the id for the pitcher-side quality table."""
    by_team = (gp.drop_duplicates(["Name", "Team"])
               .set_index(["Name", "Team"])["PlayerId"])
    uniq = gp.groupby("Name")["PlayerId"].nunique()
    by_name = (gp[gp["Name"].isin(uniq[uniq == 1].index)]
               .drop_duplicates("Name").set_index("Name")["PlayerId"])
    return by_team, by_name


def _pitcher_hr_allowed_table(hr, gp, parks):
    """HR quality ALLOWED per pitcher as-of: how many, how hard, how far
    (elevation-adjusted). pc_hr_bf says how often a starter is taken deep;
    this says whether those homers are wall-scrapers or moonshots."""
    by_team, by_name = _pitcher_name_map(gp)
    h = hr.copy()
    key = pd.MultiIndex.from_frame(h[["Pitcher", "PitcherTeam"]])
    h["PitcherId"] = by_team.reindex(key).to_numpy()
    miss = h["PitcherId"].isna()
    h.loc[miss, "PitcherId"] = by_name.reindex(h.loc[miss, "Pitcher"]).to_numpy()
    h = h.dropna(subset=["PitcherId"])
    h["PitcherId"] = h["PitcherId"].astype("int64")
    elev = parks.set_index("Ballpark")["Elevation_ft"]
    h["dist_adj"] = h["Distance"] - ELEV_DIST_FT * h["Ballpark"].map(elev).fillna(0)
    h = h.sort_values(["PitcherId", "Date"])
    g = h.groupby("PitcherId", sort=False)
    h["cum_n"] = g.cumcount() + 1
    h["cum_ev"] = g["Exit Velo"].cumsum()
    h["cum_dist"] = g["dist_adj"].cumsum()
    day = h.groupby(["PitcherId", "Date"], as_index=False).last()
    return day[["PitcherId", "Date", "cum_n", "cum_ev", "cum_dist"]]


def _hr_pitch_counts(hr):
    """Cumulative career HR count per (BatterId, Pitch name, Date), for the
    HR-by-pitch-type matchup score. Rows are day-end totals; consumers join
    with allow_exact_matches=False so a game on D sees counts through D-1."""
    h = hr.dropna(subset=["BatterId"]).copy()
    h["BatterId"] = h["BatterId"].astype("int64")
    h = h.sort_values(["BatterId", "Pitch", "Date"])
    h["cnt"] = h.groupby(["BatterId", "Pitch"], sort=False).cumcount() + 1
    return h.groupby(["BatterId", "Pitch", "Date"], as_index=False)["cnt"].max()


def hrpt_from_counts(counts, usage, total_hr):
    """HR-by-pitch-type matchup score from a batter's per-pitch HR counts, a
    pitcher's usage fractions {pitch name: fraction}, and the batter's total
    career HR. Shared by the vectorized and inference paths: the share of
    the batter's homers that came on the pitches this starter actually
    throws, weighted by how often he throws them."""
    if not total_hr or pd.isna(total_hr) or not usage:
        return np.nan
    num = sum(frac * counts.get(pitch, 0) for pitch, frac in usage.items())
    return num / total_hr


def _hrpt_scores(df, hr, ars_p):
    """Vectorized hrpt: one score per batter-frame row (df needs PlayerId,
    StarterId, Season, Date). Prior-season usage; as-of HR counts."""
    u = ars_p.dropna(subset=["%"])[["PlayerId", "Year", "Pitch", "%"]].rename(
        columns={"PlayerId": "StarterId"})
    u["StarterId"] = u["StarterId"].astype(df["StarterId"].dtype)
    u["Season"] = u["Year"] + 1
    u["frac"] = u["%"] / 100.0
    pairs = df[["PlayerId", "StarterId", "Season", "Date"]].reset_index()
    m = pairs.merge(u[["StarterId", "Season", "Pitch", "frac"]],
                    on=["StarterId", "Season"], how="inner")
    m = m.rename(columns={"PlayerId": "BatterId"})
    cnts = _hr_pitch_counts(hr)
    m = pd.merge_asof(m.sort_values("Date"), cnts.sort_values("Date"),
                      on="Date", by=["BatterId", "Pitch"],
                      direction="backward", allow_exact_matches=False)
    m["w"] = m["frac"] * m["cnt"].fillna(0)
    return m.groupby("index")["w"].sum()


def _asof_merge(left, right, by, date_col="Date"):
    left = left.sort_values(date_col)
    right = right.sort_values(date_col)
    out = pd.merge_asof(left, right, on=date_col, by=by,
                        direction="backward", allow_exact_matches=False)
    return out


# ------------------------------------------------------- matchup features

ARS_B_METRICS = {"xSLG": "m_xslg", "xwOBA": "m_xwoba",
                 "Whiff %": "m_whiff", "Hard Hit %": "m_hh",
                 # audit wave (rank 19): the K dimension of the batter-vs-
                 # arsenal join — K%-per-PA is not whiff-per-swing (called
                 # strikes, foul-offs, count management) and PutAway% is
                 # type-grain two-strike conversion (ARS_K_METRICS treats
                 # them as additive pitcher-side already)
                 "K%": "m_kpct", "Put Away %": "m_paway"}
ARS_P_METRICS = {"xSLG": "pars_xslg", "Whiff %": "pars_whiff",
                 "Hard Hit %": "pars_hh", "RV/100": "pars_rv100"}
# K-model view of a starter's arsenal: the strikeout-predictive metrics
ARS_K_METRICS = {"Whiff %": "pars_whiff", "K%": "pars_kpct",
                 "Put Away %": "pars_paway", "RV/100": "pars_rv100"}
# two-year decay blend: last season dominates, the season before stabilizes
# small samples; either year alone is used when the other is missing
ARS_W1, ARS_W2 = 0.7, 0.3


def _blend_years(out1, out2, cols):
    """ARS_W1*y-1 + ARS_W2*y-2 per column, falling back to whichever year
    exists when the other is NaN. Inputs share an index."""
    out = out1[cols].copy()
    for c in cols:
        b = ARS_W1 * out1[c] + ARS_W2 * out2[c]
        out[c] = b.fillna(out1[c]).fillna(out2[c])
    return out


def _matchup_one(pairs, ars_p, ars_b, lag):
    """Batter-vs-arsenal scores against the season `lag` years back."""
    pairs = pairs.copy()
    pairs["ArsYear"] = pairs["Season"] - lag

    p = ars_p.rename(columns={"PlayerId": "PitcherId", "Year": "ArsYear"})
    p = p[["PitcherId", "ArsYear", "PitchType", "%",
           *ARS_P_METRICS.keys()]].rename(columns={"%": "usage"})
    b = ars_b.rename(columns={"PlayerId": "BatterId", "Year": "ArsYear"})
    b = b[["BatterId", "ArsYear", "PitchType", *ARS_B_METRICS.keys()]]
    b = b.rename(columns={k: v for k, v in ARS_B_METRICS.items()})

    m = pairs.merge(p, on=["PitcherId", "ArsYear"], how="left")
    m = m.merge(b, on=["BatterId", "ArsYear", "PitchType"], how="left")

    keys = ["BatterId", "PitcherId", "Season"]
    out = pairs[keys].copy().set_index(keys)
    usage = m["usage"].fillna(0)

    for src, dst in ARS_P_METRICS.items():
        w = usage * m[src]
        grp = m.assign(_w=w, _u=usage.where(m[src].notna()))
        agg = grp.groupby(keys)[["_w", "_u"]].sum(min_count=1)
        out[dst] = agg["_w"] / agg["_u"]
    for dst in ARS_B_METRICS.values():
        w = usage * m[dst]
        grp = m.assign(_w=w, _u=usage.where(m[dst].notna()))
        agg = grp.groupby(keys)[["_w", "_u"]].sum(min_count=1)
        out[dst] = agg["_w"] / agg["_u"]
    cov = m.assign(_c=usage.where(m["m_xslg"].notna(), 0)).groupby(keys)["_c"].sum()
    out["m_coverage"] = cov / 100.0
    return out


def matchup_features(pairs, ars_p, ars_b):
    """Batter-vs-starter-arsenal scores from prior-season Statcast data,
    decay-blended over the last two seasons (ARS_W1/ARS_W2).

    pairs: DataFrame[BatterId, PitcherId, Season]. For each pair, weight the
    batter's per-pitch-type results by how often the pitcher threw each
    pitch; also aggregate the pitcher's own arsenal quality.
    """
    pairs = pairs.drop_duplicates().copy()
    cols = [*ARS_P_METRICS.values(), *ARS_B_METRICS.values(), "m_coverage"]
    out1 = _matchup_one(pairs, ars_p, ars_b, 1)
    out2 = _matchup_one(pairs, ars_p, ars_b, 2)
    return _blend_years(out1, out2, cols).reset_index()


def pitcher_arsenal_feats(pitchers, ars_p):
    """Usage-weighted arsenal quality per (PitcherId, Season) for the K
    model — whiff, K%, put-away (two-strike conversion), run value — with
    the same two-year decay blend as matchup_features. pars_cov is the
    usage share the blend actually covered."""
    pitchers = pitchers.drop_duplicates().copy()

    def one(lag):
        pit = pitchers.copy()
        pit["ArsYear"] = pit["Season"] - lag
        p = ars_p.rename(columns={"PlayerId": "PitcherId", "Year": "ArsYear"})
        p = p[["PitcherId", "ArsYear", "PitchType", "%",
               *ARS_K_METRICS.keys()]].rename(columns={"%": "usage"})
        m = pit.merge(p, on=["PitcherId", "ArsYear"], how="left")
        keys = ["PitcherId", "Season"]
        out = pit[keys].copy().set_index(keys)
        usage = m["usage"].fillna(0)
        for src, dst in ARS_K_METRICS.items():
            w = usage * m[src]
            grp = m.assign(_w=w, _u=usage.where(m[src].notna()))
            agg = grp.groupby(keys)[["_w", "_u"]].sum(min_count=1)
            out[dst] = agg["_w"] / agg["_u"]
        cov = m.assign(_c=usage.where(m["Whiff %"].notna(), 0)).groupby(keys)["_c"].sum()
        out["pars_cov"] = cov / 100.0
        return out

    cols = [*ARS_K_METRICS.values(), "pars_cov"]
    return _blend_years(one(1), one(2), cols).reset_index()


ARS_DYN_COLS = ["pars_entropy", "pars_rv_trend", "pars_whiff_trend",
                "pars_new_usage"]


def _arsenal_dynamics(ars_p):
    """Per (PitcherId, serve Season): arsenal breadth + trajectory that the
    0.7/0.3 LEVEL blend hides (audit wave rank 22). Leakage-safe — a serve
    season S reads only arsenal Years <= S-1. Shared by the frame builders
    and the serving Store so the two paths compute identically.
      pars_entropy      Shannon entropy of the renormalized pitch-mix
                        (breadth/unpredictability), 0.7/0.3 blend S-1/S-2
      pars_rv_trend     usage-weighted RV/100 change S-1 vs S-2 (stuff
                        up/down; delta clipped +/-10 vs tiny-sample types)
      pars_whiff_trend  same for Whiff % (clip +/-25)
      pars_new_usage    share of S-1 usage on pitch types absent in S-2
                        (new-pitch adoption; NaN if no S-2 arsenal at all)
    """
    a = ars_p.dropna(subset=["PlayerId", "Year", "PitchType"]).copy()
    a["PlayerId"] = a["PlayerId"].astype("int64")
    a["Year"] = a["Year"].astype(int)
    for c in ("%", "RV/100", "Whiff %", "Pitches"):
        a[c] = pd.to_numeric(a[c], errors="coerce")
    # entropy per (PlayerId, Year), renormalizing usage (Statcast omits
    # rarely-thrown types so the raw shares under-sum on ~5% of rows)
    g = a.groupby(["PlayerId", "Year"])
    usum = g["%"].transform("sum")
    p = a["%"] / usum.where(usum > 0)
    a["_plnp"] = -(p * np.log(p.where(p > 0)))
    ent = g.agg(_H=("_plnp", "sum"), _usum=("%", "sum")).reset_index()
    ent["H"] = ent["_H"].where(ent["_usum"] >= 50)   # thin-mix rows -> NaN
    e1 = ent.assign(Season=ent["Year"] + 1).rename(columns={"H": "H1"})
    e2 = ent.assign(Season=ent["Year"] + 2).rename(columns={"H": "H2"})
    ent_s = e1[["PlayerId", "Season", "H1"]].merge(
        e2[["PlayerId", "Season", "H2"]], on=["PlayerId", "Season"],
        how="outer")
    b = ARS_W1 * ent_s["H1"] + ARS_W2 * ent_s["H2"]
    ent_s["pars_entropy"] = b.fillna(ent_s["H1"]).fillna(ent_s["H2"])
    ent_s = ent_s[["PlayerId", "Season", "pars_entropy"]]
    # year-over-year dynamics: shared pitch types, >=50 pitches both years
    a1 = a[a["Pitches"] >= 50][["PlayerId", "Year", "PitchType", "%",
                                "RV/100", "Whiff %"]]
    y1 = a1.rename(columns={"%": "u1", "RV/100": "rv1", "Whiff %": "wh1"})
    y1 = y1.assign(Season=y1["Year"] + 1)
    y2 = a1.rename(columns={"%": "u2", "RV/100": "rv2", "Whiff %": "wh2"})
    y2 = y2.assign(Season=y2["Year"] + 2)
    j = y1.merge(y2[["PlayerId", "Season", "PitchType", "u2", "rv2", "wh2"]],
                 on=["PlayerId", "Season", "PitchType"], how="left")
    sh = j[j["u2"].notna()].copy()
    sh["_rvw"] = sh["u1"] * (sh["rv1"] - sh["rv2"]).clip(-10, 10)
    sh["_whw"] = sh["u1"] * (sh["wh1"] - sh["wh2"]).clip(-25, 25)
    tr = sh.groupby(["PlayerId", "Season"]).agg(
        _rvw=("_rvw", "sum"), _whw=("_whw", "sum"),
        _u=("u1", "sum")).reset_index()
    tr["pars_rv_trend"] = tr["_rvw"] / tr["_u"].where(tr["_u"] > 0)
    tr["pars_whiff_trend"] = tr["_whw"] / tr["_u"].where(tr["_u"] > 0)
    # new-pitch usage: needs a S-2 arsenal to define "new" (else NaN)
    has2 = (y2.groupby(["PlayerId", "Season"]).size()
            .rename("_n2").reset_index())
    newu = (j[j["u2"].isna()].groupby(["PlayerId", "Season"])["u1"].sum()
            .rename("_newu").reset_index())
    nu = has2.merge(newu, on=["PlayerId", "Season"], how="left")
    nu["pars_new_usage"] = nu["_newu"].fillna(0.0) / 100.0
    out = ent_s.merge(
        tr[["PlayerId", "Season", "pars_rv_trend", "pars_whiff_trend"]],
        on=["PlayerId", "Season"], how="outer")
    out = out.merge(nu[["PlayerId", "Season", "pars_new_usage"]],
                    on=["PlayerId", "Season"], how="outer")
    return out


# ------------------------------------------------------------ rate helpers

# Empirical-Bayes shrinkage of batter rate stats toward league priors.
# shrunk = (successes + K*prior) / (trials + K); K ~ the stat's stabilization
# sample size, so a small-sample rate is pulled to league average and a
# large-sample rate barely moves. Denoises the most important features
# (early-season and rolling windows) without leaking. Constants are fixed
# league averages so the training and inference paths compute identically.
# KNOWN LIMITATION (2026-07-14 audit): these priors (and the BIP/PD ones
# below) were measured on the pooled files, so the future league level
# faintly informs past features. Not target leakage, and shared identically
# by both arms of every paired read (relative verdicts unaffected), but a
# rigorous absolute backtest would fit priors as-of. Kept frozen: parity
# between the two code paths requires fixed constants.
#   name -> (prior, K, numerator stat, denominator stat)
SHRINK = {
    "hr_pa":  (0.032, 170, "HR", "PA"),
    "tb_ab":  (0.410, 320, "TB", "AB"),
    "h_ab":   (0.245, 460, "H",  "AB"),
    "k_pct":  (0.225, 60,  "SO", "PA"),
    "bb_pct": (0.085, 120, "BB", "PA"),
    "iso":    (0.165, 160, None, "AB"),   # numerator = TB - H
    "sb_pa":  (0.016, 200, "SB", "PA"),   # steals are player-idiosyncratic
    "r_pa":   (0.125, 250, "R",  "PA"),   # runs scored (lineup context)
    "rbi_pa": (0.115, 250, "RBI", "PA"),
    # productive/unproductive outs (2026-07-09, train._PRODOUT -> run/rbi):
    # GIDP is a rally-killer that erases baserunners and the batter (suppresses
    # both his run and the runners' RBI); SF is a productive out that cashes a
    # run (an RBI without a hit). Both rare + noisy -> strong prior K; league
    # rates measured off the full game-batting log (GIDP/PA 0.018, SF/PA 0.007).
    "gidp_pa": (0.018,  200, "GIDP", "PA"),
    "sf_pa":   (0.0067, 250, "SF",   "PA"),
    # hit-type rates (audit wave 2026-07-14): the triple head's own history
    # — 3B/AB is a stable skill (y2y r=0.45) trees can't rebuild from the
    # fused iso/xbh/tb blends. Priors measured on the full game-batting log
    # (1B/AB 0.160, 2B/AB 0.049, 3B/AB 0.0046); Ks by convention, strong
    # for the thin 3B rate. 1B numerator = H-2B-3B-HR (iso precedent).
    "1b_ab":  (0.160,  460, None, "AB"),
    "2b_ab":  (0.049,  300, "2B", "AB"),
    "3b_ab":  (0.0046, 500, "3B", "AB"),
    # RBI conversion (audit wave, rank 14): realized cash-in rate per
    # driving-in opportunity — RBI/(RBI+LOB), the only column separating
    # realized opportunity from conversion (LOB had zero consumers).
    # Prior = league RBI/(RBI+LOB) measured on the game-batting log.
    "rbi_conv": (0.232, 40, "RBI", None),   # denominator = RBI + LOB
}
SHRINK_ROLL = ("hr_pa", "tb_ab", "k_pct", "sb_pa",
               "r_pa", "rbi_pa")  # rolling: PA denominator

# H+R+RBI per-game threshold shares (own-target history, hrr props only):
# prior = league share of batter-games clearing the line (full game-batting
# log, pinch-hit games included — the same population the histories sum
# over), K in games. hrr2/hrr3 were the only props with no joint target
# history; the marginals (c_r_pa_sh etc.) miss how a batter's H/R/RBI
# cluster within games (a cleanup hitter's H and RBI arrive together).
# Benched 2026-07-08 (hrr2_ece 1.5x band, rest flat), RE-ACCEPTED
# 2026-07-09 (queue Tier B6, keep-leaning bar) with an hrr2 ROUTE-AROUND:
# the six columns route via train._HRR_HIST to hrr3 + xhrr only; hrr2
# (the ece casualty) never sees them. Coverage 100%, parity 1.7e-16,
# target corr ~+0.11.
HRR_SHRINK = {"hrr2_g": (0.395, 40.0), "hrr3_g": (0.250, 40.0)}
# audit wave 2026-07-14: deeper thresholds for the heads that shipped
# without their own-target history (rbi2/run2/hrr4). Priors measured on
# the <=2024 game-batting log (hrr4 0.139, rbi2 0.085, run2 0.064), K=40
# by convention like the hrr pair. THRESH_HIST maps cum/decay sum keys to
# the output stems the shared helper emits.
THRESH_SHRINK = {**HRR_SHRINK, "hrr4_g": (0.139, 40.0),
                 "rbi2_g": (0.085, 40.0), "run2_g": (0.064, 40.0)}
THRESH_HIST = {"HRR2": "hrr2_g", "HRR3": "hrr3_g", "HRR4": "hrr4_g",
               "RBI2": "rbi2_g", "RUN2": "run2_g"}

# Home-plate umpire zone tendency: his career K% and BB% per batter-faced
# over PRIOR games (both teams' pitching lines), EB-shrunk toward the league
# rate. A tight zone inflates walks and suppresses strikeouts and vice
# versa; this is the only zone-authority signal, routed to the K/BB props.
# Priors are the measured league K/BF and BB/BF; K (~5 games of BF) only
# bites for an ump's first handful of games — they accrue ~75 BF/game and
# hundreds of games across the dataset, so the estimate is otherwise firm.
UMP_K_PRIOR, UMP_BB_PRIOR, UMP_K = 0.226, 0.085, 400.0
# ... and his run environment (2026-07-14 #21): shrunk as-of total runs per
# game with this HP ump — the totals/earned-run view of the same authority
# signal the K/BB split can't carry. Prior = measured league R/G (both
# teams, 2015-2026 games file); K in games.
UMP_RG_PRIOR, UMP_RG_K = 9.0, 40.0

# Starter venue splits (2026-07-14 #23): as-of home/road K / HR / ERA for a
# starter, EB-shrunk like the batter vloc_* family. Priors are league
# starter rates per BF (ERA per out); Ks set by convention (BIP_SHRINK
# precedent, not swept).
PVLOC_K_PRIOR, PVLOC_K_K = 0.222, 200.0      # K per BF
PVLOC_HR_PRIOR, PVLOC_HR_K = 0.030, 300.0    # HR per BF
PVLOC_ER_PRIOR, PVLOC_ER_K = 4.30 / 27.0, 150.0   # ER per out, K in outs
PVLOC_STATS = ["BF", "HR", "SO", "ER", "Outs"]

# audit wave (rank 21): the chaos channel — wild pitches + balks (free
# advancement invisible to the bb/h/era decomposition) and hit batsmen
# (a baserunner channel outside BB). ~1.8% of BF combined, so shrunk per
# the pvloc convention (plain pc_* rates stay unshrunk); priors measured
# on the 2015-2026 starter game logs.
CHAOS_PRIOR, CHAOS_K = 0.0082, 200.0      # (WP+BK) per BF
HBP_PRIOR, HBP_K = 0.0097, 200.0          # HBP per BF

# audit wave (rank 25): pitch-economy budget — league NP per out (stable
# 5.34-5.60 across years), K-shrunk so early-career pc_Outs=0 lands on the
# constant. p_exp_outs = recent budget / this ratio.
NP_PER_OUT_PRIOR, NP_PER_OUT_K = 5.48, 100.0

# Outing-shape gates (2026-07-14 #16): starts needed before the outing-
# length SD / short-start share mean anything
OUTS_SD_MIN_STARTS = 5
SHORT_START_OUTS = 12        # <= 12 outs = a short start (opener/quick hook)
SHORT_SHARE_WIN, SHORT_SHARE_MIN = 10, 3

# Layoff / ramp regime cutoffs (2026-07-14 #17)
GAP_DAYS = 15                # 15+ days = IL return / call-up regime
RAMP_NP = 60                 # last start under 60 pitches = ramping
SHORT_REST_DAYS, HEAVY_NP = 4, 100   # short rest after a heavy start

# BaseRuns cluster-luck residual (2026-07-14 #20): trailing-30-day team
# runs minus the BaseRuns expectation from components — sequencing luck
# that regresses. NaN under BSR_MIN_G games in the window (early April).
BSR_MIN_G = 10

# schedule/travel (2026-07-14 #18): tz approximation — degrees of
# longitude per hour; good enough for a fatigue feature and computable
# from the ballparks file alone at serve time
_TZ_DEG_PER_HR = 15.0
EARTH_R_KM = 6371.0


def _dn1(v):
    """Normalize a DayNight value ('night', 'Day', 'N', ...) to 'D'/'N'/''.
    Scalar; the vectorized table applies the same rule via .str ops."""
    s = str(v).strip().title()[:1] if v is not None and pd.notna(v) else ""
    return s if s in ("D", "N") else ""


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km; NaN inputs propagate. Vectorized."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def sched_from_prev(prev_date, prev_venue, prev_dn, date, dn, venue, coords):
    """Schedule-context features off a team's PREVIOUS game (strictly
    earlier DATE — day-start convention): day-after-night flag, great-circle
    travel km between venues, and the approximate timezone shift in hours
    (signed, east positive). One scalar implementation shared by the
    serving path; the vectorized table computes the same formulas.
    `coords` maps venue -> (lat, lon)."""
    if prev_date is None or pd.isna(prev_date):
        return {"day_after_night": np.nan, "travel_km": np.nan,
                "tz_delta": np.nan}
    dan = np.nan
    if prev_dn in ("D", "N") and dn in ("D", "N"):
        dan = float(prev_dn == "N" and dn == "D"
                    and (pd.Timestamp(date) - pd.Timestamp(prev_date)).days == 1)
    la1, lo1 = coords.get(prev_venue, (np.nan, np.nan))
    la2, lo2 = coords.get(venue, (np.nan, np.nan))
    return {"day_after_night": dan,
            "travel_km": float(haversine_km(la1, lo1, la2, lo2)),
            "tz_delta": float((lo2 - lo1) / _TZ_DEG_PER_HR)}


def pvloc_from_sums(bf, hr, so, er, outs):
    """Shrunk starter venue-split rates from raw sums — the single
    definition both paths call (batter vloc_* idiom)."""
    return {"pvloc_k_bf": (so + PVLOC_K_K * PVLOC_K_PRIOR) / (bf + PVLOC_K_K),
            "pvloc_hr_bf": (hr + PVLOC_HR_K * PVLOC_HR_PRIOR)
            / (bf + PVLOC_HR_K),
            "pvloc_era": (er + PVLOC_ER_K * PVLOC_ER_PRIOR)
            / (outs + PVLOC_ER_K) * 27.0}


def outs_sd_from_sums(n, s, s2):
    """As-of SD of a starter's outing lengths from (starts, sum, sum-of-
    squares); NaN under OUTS_SD_MIN_STARTS. Shared by both paths."""
    n = np.asarray(n, dtype="float64")
    s = np.asarray(s, dtype="float64")
    s2 = np.asarray(s2, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        var = s2 / n - (s / n) ** 2
        sd = np.sqrt(np.clip(var, 0.0, None))
    return np.where(n >= OUTS_SD_MIN_STARTS, sd, np.nan)

# Statcast bat tracking (scrape_bat_tracking.py, 2023+): raw CSV column ->
# feature name. Prior-season, routed to the power props (train._BAT). BANKED
# BUT INERT until the training window covers a bat-tracking season (~2027);
# wired now so it self-activates at the rollover with no code change.
BAT_TRACK_REN = {"BatSpeed": "bt_speed", "SwingLength": "bt_swlen",
                 "HardSwingRate": "bt_hardsw", "BlastPerSwing": "bt_blast"}
BAT_TRACK_COLS = list(BAT_TRACK_REN.values())

# MiLB level-translated priors (Model/milb_priors.py artifact): pooled
# AAA/AA/A+ line translated to MLB-equivalent PA-class rates + evidence
# mass (log1p of the decayed PA sum). Serve rows are keyed by the season
# they SERVE and only use MiLB seasons <= Season-1, so the join is
# EXACT-season (not _merge_prior_season). OUT is dropped (rates sum to 1).
MILB_REN = {"t_K": "k", "t_BB": "bb", "t_HBP": "hbp", "t_1B": "b1",
            "t_2B": "b2", "t_3B": "b3", "t_HR": "hr"}
# audit wave 2026-07-14 (rank 28): fit-free v1 extras already carrying
# their final feature names in the serve tables — bat steal prior (rookie
# speed axis; every MLB speed column is NaN exactly at debut) and pit
# workload pedigree (starter-role outs/start + GS share; all workload
# mains are MLB-history-only).
MILB_BAT_EXTRA = ["milb_att", "milb_sb_succ"]
MILB_PIT_EXTRA = ["pmilb_outs_ps", "pmilb_gs_share"]
MILB_BAT_COLS = ([f"milb_{v}" for v in MILB_REN.values()] + ["milb_n"]
                 + MILB_BAT_EXTRA)
MILB_PIT_COLS = ([f"pmilb_{v}" for v in MILB_REN.values()] + ["pmilb_n"]
                 + MILB_PIT_EXTRA)


def _milb_cols(serve, prefix):
    """Feature-named copy of a milb_priors serve table."""
    t = serve.rename(columns={k: f"{prefix}{v}" for k, v in MILB_REN.items()})
    t[f"{prefix}n"] = np.log1p(t["n_eff"])
    cols = [f"{prefix}{v}" for v in MILB_REN.values()] + [f"{prefix}n"]
    extras = [c for c in MILB_BAT_EXTRA + MILB_PIT_EXTRA if c in t.columns]
    return t[["PlayerId", "Season", *cols, *extras]]

# stolen-base success rate: prior ~ league SB% ; small K (fast to stabilize)
SB_SUCC_PRIOR, SB_SUCC_K = 0.75, 20.0
TSB_STOP_PRIOR = 1.0 - SB_SUCC_PRIOR  # league share of attempts cut down
PSB_MIN_ATT = 5  # attempts needed before a pitcher's stop-rate means much
# audit wave (rank 4): the sb opportunity chain — reach-first rate and
# attempt-per-opportunity, both K-shrunk. Priors measured on the full
# game-batting log (ob1 0.238; attempts 0.09 per the POST-2023 rules
# regime — league jumped 0.067 -> 0.104 at the rule change). ST_REACH is
# the centering constant for the starter-traffic product (league starter
# H/BF + BB/BF); CTX_BSLG_MEAN centers the protection product (rank 31,
# frame mean of ctx_behind_slg_d).
OB1_PRIOR, OB1_K = 0.24, 100.0
SB_ATT_PRIOR, SB_ATT_K = 0.09, 30.0
ST_REACH_PRIOR = 0.31
CTX_BSLG_MEAN = 0.41
SHRINK_COLS = ([f"c_{n}_sh" for n in SHRINK] + [f"s_{n}_sh" for n in SHRINK]
               + [f"r{w}_{n}_sh" for w in ROLL_WINDOWS for n in SHRINK_ROLL])

# rolling + decayed own R/RBI rates: computed in BOTH paths (parity
# 5.6e-17). Benched 2026-07-08 (run_ece +.0063 past band, rbi mixed),
# RE-ACCEPTED 2026-07-09 (queue Tier B5, keep-leaning bar) with a run
# ROUTE-AROUND — routed with train._RUNRBI_FORM to rbi/hrr2/hrr3/xhrr;
# run (the ece casualty) never sees them.
RUNRBI_FORM_COLS = ([f"r{w}_{n}_pa_sh" for w in ROLL_WINDOWS
                     for n in ("r", "rbi")]
                    + ["d_r_pa_sh", "d_rbi_pa_sh"])


def _shrink(numer, denom, name):
    prior, k = SHRINK[name][0], SHRINK[name][1]
    return (numer + k * prior) / (denom + k)


def shrunk_from_sums(sums, pre, roll=False):
    """Shrunk rates from a mapping of stat sums (scalar path, for inference).
    `roll` uses PA as the denominator (rolling windows track no AB)."""
    out = {}
    names = SHRINK_ROLL if roll else SHRINK
    for name in names:
        _, _, num_s, den_s = SHRINK[name]
        if name == "iso":
            numer = sums["TB"] - sums["H"]
        elif name == "1b_ab":   # composite numerator (iso precedent)
            numer = sums["H"] - sums["2B"] - sums["3B"] - sums["HR"]
        else:
            numer = sums[num_s]
        if roll:
            denom = sums["PA"]
        elif name == "rbi_conv":   # opportunity denominator = RBI + LOB
            denom = sums["RBI"] + sums["LOB"]
        else:
            denom = sums[den_s]
        out[f"{pre}_{name}_sh"] = _shrink(numer, denom, name)
    return out


def decayed_feats(sums):
    """Decay-weighted shrunk rates from {stat: decayed as-of sum}. Shared by
    the vectorized (Series) and inference (scalar) paths so both compute
    identically. d_PA is the effective (decayed) sample size."""
    out = {"d_PA": sums["PA"]}
    for name in DECAY_SHRINK:
        _, _, num_s, _ = SHRINK[name]
        out[f"d_{name}_sh"] = _shrink(sums[num_s], sums["PA"], name)
    return out


def hrr_hist_feats(sums, den, pre):
    """Shrunk per-game threshold shares (hrr2/3/4, rbi2, run2) from the
    THRESH_HIST sum keys and a game count. Shared by the vectorized
    (Series) and inference (scalar) paths so both compute identically;
    zero sums land exactly on the league prior."""
    out = {}
    for key, stem in THRESH_HIST.items():
        prior, k = THRESH_SHRINK[stem]
        out[f"{pre}_{stem}_sh"] = (sums[key] + k * prior) / (den + k)
    return out


def _ump_game_totals(gp):
    """Per-game strikeout / walk / batters-faced / runs totals (both teams'
    pitching lines) — the raw material for a home-plate ump's zone
    tendency and run environment. Shared by the vectorized frame and the
    inference store."""
    g = gp[["GamePk", "SO", "BB", "BF", "R"]].copy()
    for c in ("SO", "BB", "BF", "R"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    return g.groupby("GamePk").agg(g_SO=("SO", "sum"), g_BB=("BB", "sum"),
                                   g_BF=("BF", "sum"),
                                   g_R=("R", "sum")).reset_index()


def _ump_shrink(so, bb, bf):
    """Shrunk (K%, BB%) per batter-faced from as-of ump totals — the single
    definition both paths call so training and serving agree exactly."""
    return ((so + UMP_K * UMP_K_PRIOR) / (bf + UMP_K),
            (bb + UMP_K * UMP_BB_PRIOR) / (bf + UMP_K))


def _ump_rg_shrink(runs, games):
    """Shrunk runs-per-game with this HP ump (2026-07-14 #21) — shared by
    both paths; zero history lands on the league prior."""
    return (runs + UMP_RG_K * UMP_RG_PRIOR) / (games + UMP_RG_K)


def _ump_asof(umps, gp):
    """Per-GamePk home-plate-umpire tendency, as-of (his K%/BB% over PRIOR
    games only, leakage-free). An umpire never works the plate twice in one
    day, so a strictly-prior cumsum is already date-clean — no doubleheader
    snap needed."""
    tot = _ump_game_totals(gp)
    u = umps.merge(tot, on="GamePk", how="left")
    u["HpUmpId"] = pd.to_numeric(u["HpUmpId"], errors="coerce")
    u = u.dropna(subset=["HpUmpId"]).copy()
    u["HpUmpId"] = u["HpUmpId"].astype("int64")
    for s in ("g_SO", "g_BB", "g_BF", "g_R"):
        u[s] = pd.to_numeric(u[s], errors="coerce").fillna(0.0)
    u = u.sort_values(["HpUmpId", "Date", "GamePk"]).reset_index(drop=True)
    g = u.groupby("HpUmpId", sort=False)
    cso = g["g_SO"].cumsum() - u["g_SO"]
    cbb = g["g_BB"].cumsum() - u["g_BB"]
    cbf = g["g_BF"].cumsum() - u["g_BF"]
    u["ump_k_pct"], u["ump_bb_pct"] = _ump_shrink(cso, cbb, cbf)
    # run environment (#21): shrunk as-of R/G over his prior games
    crun = g["g_R"].cumsum() - u["g_R"]
    cn = g.cumcount().astype(float)
    u["ump_r_g"] = _ump_rg_shrink(crun, cn)
    return u[["GamePk", "ump_k_pct", "ump_bb_pct", "ump_r_g"]]


def _merge_ump(df, raw):
    """Attach ump_k_pct/ump_bb_pct by GamePk (NaN when the umpire file is
    absent — old cache / not yet scraped — so the models impute harmlessly)."""
    if raw.get("umps") is not None:
        return df.merge(_ump_asof(raw["umps"], raw["gp"]), on="GamePk",
                        how="left")
    df["ump_k_pct"] = np.nan
    df["ump_bb_pct"] = np.nan
    df["ump_r_g"] = np.nan
    return df


def add_bat_trends(d):
    """Form/trend deltas from existing shrunk rates: rolling-15 vs season
    (hot/cold streak) and season vs career (breakout/decline year). The trees
    could build these from the levels, but handing them the direction of
    change directly is denoised signal. Works on a DataFrame (training) or a
    plain row dict (inference); NaN inputs propagate to NaN."""
    d["tr15_hr"] = d["r15_hr_pa_sh"] - d["s_hr_pa_sh"]
    d["tr15_tb"] = d["r15_tb_ab_sh"] - d["s_tb_ab_sh"]
    d["tr15_k"] = d["r15_k_pct_sh"] - d["s_k_pct_sh"]
    d["dev_hr"] = d["s_hr_pa_sh"] - d["c_hr_pa_sh"]
    d["dev_tb"] = d["s_tb_ab_sh"] - d["c_tb_ab_sh"]
    d["dev_k"] = d["s_k_pct_sh"] - d["c_k_pct_sh"]
    return d


def add_pit_trends(d):
    """Starter form deltas: last-5-starts K/HR rates vs season, season ERA vs
    career — in-season improvement/decline the flat rates hide. Same
    DataFrame-or-dict duality as add_bat_trends."""
    d["p5_k_trend"] = d["p5_k_bf"] - d["ps_k_bf"]
    d["p5_hr_trend"] = d["p5_hr_bf"] - d["ps_hr_bf"]
    d["p_era_trend"] = d["ps_era"] - d["pc_era"]
    return d


def _bat_rates(d, pre):
    """Turn cumulative sums with prefix `pre` into rate features."""
    pa = d[f"{pre}_PA"]
    ab = d[f"{pre}_AB"]
    return pd.DataFrame({
        f"{pre}_hr_pa": d[f"{pre}_HR"] / pa,
        f"{pre}_tb_ab": d[f"{pre}_TB"] / ab,
        f"{pre}_h_ab": d[f"{pre}_H"] / ab,
        f"{pre}_k_pct": d[f"{pre}_SO"] / pa,
        f"{pre}_bb_pct": d[f"{pre}_BB"] / pa,
        f"{pre}_iso": (d[f"{pre}_TB"] - d[f"{pre}_H"]) / ab,
    })


def _bat_rates_shrunk(d, pre):
    """Vectorized empirical-Bayes shrunk rates (career/season prefixes)."""
    out = {}
    for name in SHRINK:
        _, _, num_s, den_s = SHRINK[name]
        if name == "iso":
            numer = d[f"{pre}_TB"] - d[f"{pre}_H"]
        elif name == "1b_ab":   # composite numerator (iso precedent)
            numer = (d[f"{pre}_H"] - d[f"{pre}_2B"]
                     - d[f"{pre}_3B"] - d[f"{pre}_HR"])
        else:
            numer = d[f"{pre}_{num_s}"]
        if name == "rbi_conv":   # opportunity denominator = RBI + LOB
            denom = d[f"{pre}_RBI"] + d[f"{pre}_LOB"]
        else:
            denom = d[f"{pre}_{den_s}"]
        out[f"{pre}_{name}_sh"] = _shrink(numer, denom, name)
    return pd.DataFrame(out)


def _pit_rates(d, pre):
    bf = d[f"{pre}_BF"]
    outs = d[f"{pre}_Outs"]
    return pd.DataFrame({
        f"{pre}_hr_bf": d[f"{pre}_HR"] / bf,
        f"{pre}_k_bf": d[f"{pre}_SO"] / bf,
        f"{pre}_bb_bf": d[f"{pre}_BB"] / bf,
        f"{pre}_h_bf": d[f"{pre}_H"] / bf,
        f"{pre}_era": d[f"{pre}_ER"] * 27 / outs,
        f"{pre}_strike_pct": d[f"{pre}_Strikes"] / d[f"{pre}_NP"],
        # audit wave (rank 21): shrunk chaos + HBP channels
        f"{pre}_chaos_bf": ((d[f"{pre}_WP"] + d[f"{pre}_BK"]
                             + CHAOS_K * CHAOS_PRIOR) / (bf + CHAOS_K)),
        f"{pre}_hbp_bf": ((d[f"{pre}_HBP"] + HBP_K * HBP_PRIOR)
                          / (bf + HBP_K)),
    })


# ------------------------------------------- prior-season lookup tables


def _batter_season_table(bat_season):
    """Per (PlayerId, Year): GO/AO, PA-weighted across team stints. The
    season files are the only GO/AO source (fly-ball tendency)."""
    t = bat_season.dropna(subset=["PlayerId"]).copy()
    t["_w"] = t["PA"].fillna(0) * t["GO/AO"].notna()
    t["_wg"] = t["GO/AO"] * t["_w"]
    g = t.groupby(["PlayerId", "Year"])[["_w", "_wg"]].sum(min_count=1)
    out = pd.DataFrame({"bat_goao": g["_wg"] / g["_w"]})
    return out.reset_index()


def _pitcher_season_table(pit_season):
    """Per (PlayerId, Year): GO/AO (TBF-weighted) plus stolen-base control —
    SB allowed per 27 outs and the stop rate (CS+PK)/attempts, NaN under
    PSB_MIN_ATT attempts. Feeds the SB prop's 'can you run on him' side."""
    t = pit_season.dropna(subset=["PlayerId"]).copy()
    t["_w"] = t["TBF"].fillna(0) * t["GO/AO"].notna()
    t["_wg"] = t["GO/AO"] * t["_w"]
    g = t.groupby(["PlayerId", "Year"]).agg(
        _w=("_w", "sum"), _wg=("_wg", "sum"), SB=("SB", "sum"),
        CS=("CS", "sum"), PK=("PK", "sum"), Outs=("Outs", "sum"))
    out = pd.DataFrame(index=g.index)
    out["pit_goao"] = g["_wg"] / g["_w"].replace(0, np.nan)
    out["psb_sb27"] = g["SB"] * 27 / g["Outs"].replace(0, np.nan)
    att = g["SB"] + g["CS"] + g["PK"]
    out["psb_stop"] = np.where(att >= PSB_MIN_ATT,
                               (g["CS"] + g["PK"]) / att, np.nan)
    return out.reset_index()


def _league_sb27_table(pit_season):
    """Per serve-Season: PRIOR-season league SB allowed per 27 outs — the
    regime-aware centering constant for the steal-permissiveness products
    (the 2023 rules moved league SB/27 from ~0.52 to ~0.73; a fixed
    constant would mis-center every era). Audit wave ranks 4/24."""
    t = pit_season.dropna(subset=["Year"]).groupby("Year").agg(
        SB=("SB", "sum"), Outs=("Outs", "sum"))
    out = pd.DataFrame({"lg_sb27_prior": t["SB"] * 27.0
                        / t["Outs"].replace(0, np.nan)}).reset_index()
    out["Season"] = out["Year"].astype(int) + 1
    return out[["Season", "lg_sb27_prior"]]


def _merge_prior_season(df, tab, id_col, cols):
    """Attach prior-season values by (id, Season-1), falling back to
    Season-2 when the player has no line the season before (injury year,
    rookie call-up mid-history). Shared join semantics for both paths."""
    for lag in (1, 2):
        t = tab.rename(columns={"PlayerId": id_col})
        t = t.assign(Season=t["Year"] + lag)[[id_col, "Season", *cols]]
        t[id_col] = t[id_col].astype(df[id_col].dtype)  # ids go float on left-merge
        t = t.rename(columns={c: f"{c}__l{lag}" for c in cols})
        df = df.merge(t, on=[id_col, "Season"], how="left")
    for c in cols:
        df[c] = df.pop(f"{c}__l1").fillna(df.pop(f"{c}__l2"))
    return df


# IL stints (2026-07-15): return-from-injury features off the paired
# transactions file. Beyond IL_RET_MAX days the "recent return" framing is
# meaningless — those rows go NaN (GBM imputes). ret21/szn default to 0.0
# for a KNOWN player with no recent stint (that absence is real pregame
# information, unlike an unknown starter, which stays NaN).
IL_RET_MAX = 365.0


def _il_stint_table(il):
    """Completed stints sorted by activation, with season-cumulative IL
    days — the shared as-of source for both paths."""
    s = il.dropna(subset=["ActDate"]).copy()
    s["StintDays"] = pd.to_numeric(s["StintDays"], errors="coerce")
    s = s.sort_values(["PlayerId", "ActDate"], kind="mergesort")
    s["_szn"] = s["ActDate"].dt.year
    s["_szn_days"] = s.groupby(["PlayerId", "_szn"])["StintDays"].cumsum()
    return s


def il_feats_from_stint(date, act_date, stint_days, rehab, szn_days,
                        pre=""):
    """Shared feature math from the last stint activated on/before `date`
    (NaT act_date = no stint). allow_exact_matches is deliberate: roster
    moves are announced before the game, so a player activated this
    morning plays tonight with ret_days = 0."""
    gap = ((date - act_date).days
           if pd.notna(act_date) else np.nan)
    ok = pd.notna(act_date) and gap <= IL_RET_MAX
    return {
        f"{pre}il_ret_days": float(gap) if ok else np.nan,
        f"{pre}il_last_len": float(stint_days) if ok else np.nan,
        f"{pre}il_ret21": (1.0 if (pd.notna(act_date) and gap <= 21)
                           else 0.0),
        f"{pre}il_szn_days": (float(szn_days)
                              if pd.notna(act_date)
                              and act_date.year == date.year else 0.0),
        f"{pre}il_rehab": float(rehab) if ok else np.nan,
    }


IL_FEATS = ("il_ret_days", "il_last_len", "il_ret21", "il_szn_days",
            "il_rehab")


def _il_asof(df, il, key, pre):
    """Vectorized twin of il_feats_from_stint: as-of last-activation merge
    per `key` (PlayerId / StarterId). Rows with a NaN key (unknown
    starter) stay NaN across the board."""
    cols = [f"{pre}{c}" for c in IL_FEATS]
    if il is None:
        for c in cols:
            df[c] = np.nan
        return df
    s = _il_stint_table(il)
    t = (s[["PlayerId", "ActDate", "StintDays", "Rehab", "_szn_days"]]
         .rename(columns={"PlayerId": key})
         .sort_values("ActDate", kind="mergesort"))
    d = df[[key, "Date"]].reset_index()
    dm = d[d[key].notna()].copy()
    t[key] = t[key].astype(dm[key].dtype)
    dm = dm.sort_values("Date", kind="mergesort")
    m = pd.merge_asof(dm, t, left_on="Date", right_on="ActDate", by=key,
                      direction="backward", allow_exact_matches=True)
    gap = (m["Date"] - m["ActDate"]).dt.days.astype("float64")
    ok = gap <= IL_RET_MAX                       # False where gap is NaN
    res = pd.DataFrame({
        cols[0]: gap.where(ok),
        cols[1]: pd.to_numeric(m["StintDays"], errors="coerce").where(ok),
        cols[2]: (gap <= 21).astype(float),
        cols[3]: m["_szn_days"].astype("float64").where(
            m["ActDate"].dt.year == m["Date"].dt.year, 0.0),
        cols[4]: pd.to_numeric(m["Rehab"], errors="coerce").where(ok),
    }).set_index(m["index"])
    for c in cols:
        df[c] = res[c].reindex(df.index)
    return df


# --------------------------------------------------------- frame assembly


def _attach_context(rows, raw, team_tab, pen_tab, park_tab,
                    pen_hl_tab=None, pen_fat_tab=None):
    """Merge game weather/park + as-of team offense, opp bullpen (overall,
    high-leverage, trailing fatigue), park factor."""
    games = raw["games"]
    rows = rows.merge(
        games[["GamePk", "Venue", "DayNight", "Temp", "Condition",
               "WindSpeed", "WindDir", "ShortGame"]], on="GamePk", how="left")
    if raw.get("weather") is not None:
        rows = rows.merge(raw["weather"][["GamePk", "Humidity", "Pressure",
                                          "Precip"]],
                          on="GamePk", how="left")
    else:
        rows["Humidity"] = np.nan
        rows["Pressure"] = np.nan
        rows["Precip"] = np.nan
    parks = raw["parks"].rename(columns={"Ballpark": "Venue"})
    rows = rows.merge(parks[["Venue", "LF", "CF", "RF", "Elevation_ft"]],
                      on="Venue", how="left")

    team = team_tab.rename(columns={c: f"toff_{c}" for c in team_tab.columns
                                    if c.startswith("cum")})
    rows = _asof_merge(rows, team, by=["Team", "Season"])
    opp_pen = pen_tab.rename(columns={"Team": "Opponent"})
    opp_pen = opp_pen.rename(columns={c: f"pen_{c}" for c in opp_pen.columns
                                      if c.startswith("cum")})
    rows = _asof_merge(rows, opp_pen, by=["Opponent", "Season"])
    if pen_hl_tab is not None:
        hl = pen_hl_tab.rename(columns={"Team": "Opponent"})
        hl = hl.rename(columns={c: f"penhl_{c}" for c in hl.columns
                                if c.startswith("cum")})
        rows = _asof_merge(rows, hl, by=["Opponent", "Season"])
        rows["pen_hl_era"] = rows["penhl_cum_ER"] * 27 / rows["penhl_cum_Outs"]
        rows["pen_hl_k_bf"] = rows["penhl_cum_SO"] / rows["penhl_cum_BF"]
    if pen_fat_tab is not None:
        fat = pen_fat_tab.rename(columns={"Team": "Opponent"})
        rows = rows.merge(fat, on=["Opponent", "Date"], how="left")
    park = park_tab.rename(columns={c: f"park_{c}" for c in park_tab.columns
                                    if c.startswith("cum")})
    rows = _asof_merge(rows, park, by=["Venue"])

    rows["toff_hr_pa"] = rows["toff_cum_HR"] / rows["toff_cum_PA"]
    rows["toff_r_pg"] = rows["toff_cum_R"] / rows["toff_cum_n"]
    rows["toff_k_pct"] = rows["toff_cum_SO"] / rows["toff_cum_PA"]
    rows["toff_bb_pct"] = rows["toff_cum_BB"] / rows["toff_cum_PA"]
    rows["pen_hr_bf"] = rows["pen_cum_HR"] / rows["pen_cum_BF"]
    rows["pen_k_bf"] = rows["pen_cum_SO"] / rows["pen_cum_BF"]
    rows["pen_h_bf"] = rows["pen_cum_H"] / rows["pen_cum_BF"]
    # audit wave (rank 1): the pen's walk rate — raw cum ratio like its
    # siblings; feeds xpen_bb (late-PA walk exposure)
    rows["pen_bb_bf"] = rows["pen_cum_BB"] / rows["pen_cum_BF"]
    rows["pen_era"] = rows["pen_cum_ER"] * 27 / rows["pen_cum_Outs"]
    # per-game venue rates over all PRIOR games, gated at 30 games so a new or
    # renamed park (Rate/Daikin) falls back to NaN. park_hr_pg reaches every
    # batter prop (legacy); R/H/2B/TB route to the offensive batter props
    # (train._PARK_OFF), the starter run-environment heads (outs/pha/per; the
    # K/walk heads drop them), and the team-runs model (2026-07-09).
    ok = rows["park_cum_n"] >= 30
    for stat, col in (("HR", "park_hr_pg"), ("R", "park_r_pg"),
                      ("H", "park_h_pg"), ("2B", "park_2b_pg"),
                      ("3B", "park_3b_pg"), ("TB", "park_tb_pg")):
        rows[col] = np.where(ok, rows[f"park_cum_{stat}"] / rows["park_cum_n"],
                             np.nan)
    return rows


def _attach_bio(rows, rosters, player_col, prefix):
    r = rosters[["PlayerId", "B", "T", "height_in", "Wt", "DOB"]].rename(
        columns={"PlayerId": player_col})
    rows = rows.merge(r, on=player_col, how="left")
    rows[f"{prefix}_height"] = rows.pop("height_in")
    rows[f"{prefix}_weight"] = pd.to_numeric(rows.pop("Wt"), errors="coerce")
    rows[f"{prefix}_age"] = (rows["Date"] - rows.pop("DOB")).dt.days / 365.25
    return rows


def _platoon(rows):
    """Effective batter hand vs pitcher throws (switch hitters bat opposite)."""
    b, t = rows["bat_hand"], rows["pit_throws"]
    eff = np.where(b == "S", np.where(t == "L", "R", "L"), b)
    same = np.where(b.isna() | t.isna(), np.nan,
                    (pd.Series(eff, index=rows.index) == t).astype(float))
    rows["same_hand"] = same
    # eff hand unknown if batter hand unknown, or switch hitter vs unknown arm
    rows["eff_hand"] = pd.Series(eff, index=rows.index).where(
        b.notna() & ~((b == "S") & t.isna()))
    return rows


def _lineup_brr_table(gb, baserun):
    """Per (GamePk, Team): mean prior-season baserunning run value of the
    posted lineup (total + extra-base advancement rate), mirroring
    _lineup_oaa_table's prior-season / NaN-skipping semantics so
    Stores.lineup_brr can compute the identical number from a bare lineup.
    Team-runs view of the batter frame's bat_brr/bat_brr_xb."""
    slot = pd.to_numeric(gb["BattingOrder"], errors="coerce")
    s = gb.loc[slot % 100 == 0,
               ["GamePk", "Team", "Season", "PlayerId"]].copy()
    br = baserun.copy()
    br["bat_brr"] = pd.to_numeric(br["RunnerRuns"], errors="coerce")
    br["bat_brr_xb"] = (pd.to_numeric(br["RunnerRunsXB"], errors="coerce")
                        / pd.to_numeric(br["Opportunities"], errors="coerce"))
    s = _merge_prior_season(
        s, br[["PlayerId", "Year", "bat_brr", "bat_brr_xb"]],
        "PlayerId", ["bat_brr", "bat_brr_xb"])
    return (s.groupby(["GamePk", "Team"], as_index=False)
            .agg(lu_brr=("bat_brr", "mean"),
                 lu_brr_xb=("bat_brr_xb", "mean")))


def _slot_pa_table(gb):
    """Per (slot, Date): as-of league PA per game at that lineup slot —
    strictly before the date (a day's own games never inform it). Built
    from the RAW game logs by both the training merge and the Stores
    lookup, so the two paths share one population by construction.
    cum_pa/cum_n are INCLUSIVE cumsums (through the row's own day);
    xpa_slot subtracts the own day out. First-ever day per slot is NaN."""
    slot = pd.to_numeric(gb["BattingOrder"], errors="coerce")
    starters = pd.DataFrame({
        "slot": (slot // 100), "Date": gb["Date"],
        "PA": pd.to_numeric(gb["PA"], errors="coerce").fillna(0.0),
    })[(slot % 100 == 0)]
    starters["slot"] = starters["slot"].astype(int)
    day = (starters.groupby(["slot", "Date"], sort=True)["PA"]
           .agg(day_pa="sum", day_n="size").reset_index())
    g = day.groupby("slot", sort=False)
    day["cum_pa"] = g["day_pa"].cumsum()
    day["cum_n"] = g["day_n"].cumsum()
    day["xpa_slot"] = ((day["cum_pa"] - day["day_pa"])
                       / (day["cum_n"] - day["day_n"]))
    return day


# positions for the lineup-defense splits (primary position per
# mlb_oaa_players.csv; catchers have no range-OAA, DHs mostly no row)
DEF_IF_POS = ("1B", "2B", "3B", "SS")
DEF_OF_POS = ("LF", "CF", "RF")


def _lineup_oaa_table(gb, oaa_players):
    """Per (GamePk, Team): mean PRIOR-SEASON player OAA of that game's
    starting lineup — overall plus infield/outfield splits (classified by
    the fielder's primary position in the OAA file, NOT tonight's fielding
    slot, so serving can compute the identical number from a bare lineup).
    The lineup is known pregame and the OAA is prior-season: leakage-free.
    This sharpens the team-season opp_oaa: it sees who is actually playing
    tonight instead of blending bench players into the everyday number."""
    slot = pd.to_numeric(gb["BattingOrder"], errors="coerce")
    s = gb.loc[slot % 100 == 0,
               ["GamePk", "Team", "Season", "PlayerId"]].copy()
    op = oaa_players.rename(columns={"OAA": "p_oaa", "Pos": "p_pos"})
    s = _merge_prior_season(s, op[["PlayerId", "Year", "p_oaa", "p_pos"]],
                            "PlayerId", ["p_oaa", "p_pos"])
    s["p_if"] = s["p_oaa"].where(s["p_pos"].isin(DEF_IF_POS))
    s["p_of"] = s["p_oaa"].where(s["p_pos"].isin(DEF_OF_POS))
    return (s.groupby(["GamePk", "Team"])
            .agg(def_p_oaa=("p_oaa", "mean"), def_p_if=("p_if", "mean"),
                 def_p_of=("p_of", "mean")).reset_index())


# roof-closed games: humidity from the outdoor weather feed is wrong indoors;
# climate control sits near 50% RH. Pressure stays ambient (a roof does not
# pressurize the building), Temp is already reported as the indoor value.
DOME_CONDITIONS = ("Dome", "Roof Closed")
INDOOR_HUMIDITY = 50.0
AIR_RHO0 = 1.165   # league-mean air density (kg/m3) — centers air_dens
                   # interactions so "thin air" is positive, heavy negative


def add_weather_derived(df):
    """Humidity/pressure-derived weather features, computed IDENTICALLY by
    every frame builder and the serving rows (parity): hum_eff (indoor-
    corrected relative humidity) and air_dens (physical air density from
    temp + station pressure + humidity — the carry variable: dense air
    knocks fly balls down; Coors sits ~0.98 kg/m3, a cold sea-level night
    ~1.25). NaN inputs propagate."""
    cond = df["Condition"].astype(str)
    hum = pd.to_numeric(df["Humidity"], errors="coerce")
    df["hum_eff"] = np.where(cond.isin(DOME_CONDITIONS), INDOOR_HUMIDITY, hum)
    t_c = (pd.to_numeric(df["Temp"], errors="coerce") - 32.0) * 5.0 / 9.0
    t_k = t_c + 273.15
    p_pa = pd.to_numeric(df["Pressure"], errors="coerce") * 100.0
    psat = 610.78 * np.exp(17.27 * t_c / (t_c + 237.3))   # Tetens, Pa
    pv = df["hum_eff"] / 100.0 * psat
    df["air_dens"] = (p_pa - pv) / (287.05 * t_k) + pv / (461.495 * t_k)
    return df


BATTER_FEATURES = None  # populated below


# ---- wind-carry features (physical HR drivers) --------------------------
# WindDir in the frames is field-relative and title-cased (games loader
# .str.title()): "Out To Cf/Lf/Rf", "In From Cf/Lf/Rf", "L To R", "R To L",
# "Varies", "Calm". Map each to the field it acts on and an out(+)/in(-) sign;
# crosswinds / calm / varies are neutral for carry over the fence.
_WIND_FIELD = {"Out To Lf": "L", "In From Lf": "L", "Out To Cf": "C",
               "In From Cf": "C", "Out To Rf": "R", "In From Rf": "R"}
_WIND_SIGN = {"Out To Lf": 1.0, "In From Lf": -1.0, "Out To Cf": 1.0,
              "In From Cf": -1.0, "Out To Rf": 1.0, "In From Rf": -1.0}
CARRY_CF_W = 0.7   # a center-field wind helps both pull sides at this weight


def add_wind_carry(df, pull=False):
    """Add out/in carry-wind features from the field-relative WindDir + WindSpeed.
    `wind_carry` = out(+)/in(-) sign x mph (general, helps any fly ball). With
    `pull=True` (batter frame, needs eff_hand) also `bat_wind_pull` = that carry
    projected onto the batter's PULL field (LHB pulls RF, RHB pulls LF; a CF wind
    helps both at CARRY_CF_W). Vectorized over the frame."""
    wd = df["WindDir"].astype(str)
    spd = pd.to_numeric(df["WindSpeed"], errors="coerce").fillna(0.0).to_numpy()
    sign = wd.map(_WIND_SIGN).fillna(0.0).to_numpy()
    df["wind_carry"] = sign * spd
    if pull:
        field = wd.map(_WIND_FIELD).to_numpy()          # "L"/"C"/"R"/NaN
        eff = df["eff_hand"].to_numpy()
        pull_field = np.where(eff == "L", "R", np.where(eff == "R", "L", ""))
        w = np.where(field == pull_field, 1.0,
                     np.where(field == "C", CARRY_CF_W, 0.0))
        df["bat_wind_pull"] = sign * spd * w
    return df


def add_batter_derived(df):
    """Row-wise derived batter features + interactions (2026-07-10 batches),
    computed IDENTICALLY by the vectorized frame and the serving path — predict.py
    calls this on the assembled single-row frame, so every derived feature is
    parity-safe by construction. All inputs are produced upstream: the as-of
    contact/plate/ump columns, _dk_* decayed sums, opp_oaa, park geometry (RF/LF),
    game weather (Temp/WindDir/WindSpeed/Elevation_ft), and the two as-of JOINS
    this function consumes but does not perform — park-hand (phh_*) and
    batter-vs-pitcher (bvp_cum_*). NaN inputs propagate (LightGBM tolerates NaN)."""
    # pull-porch geometry: the fence on the batter's pull side + HR clearance
    df["pull_fence"] = np.where(df["eff_hand"] == "L", df["RF"],
                                np.where(df["eff_hand"] == "R", df["LF"], np.nan))
    df["porch_margin"] = df["hrq_dist_avg"] - df["pull_fence"]
    # realized handed park-HR edge: venue HR/PA for the batter's EFFECTIVE hand
    # minus the other hand (short-RF parks help LHB) — the handed asymmetry that
    # park_hr_pg + fence distance miss. Gated 500 PA/hand; switch/unknown -> NaN.
    _lrate = np.where(df["phh_L_PA"] >= 500, df["phh_L_HR"] / df["phh_L_PA"], np.nan)
    _rrate = np.where(df["phh_R_PA"] >= 500, df["phh_R_HR"] / df["phh_R_PA"], np.nan)
    df["park_hand_hr_edge"] = np.where(
        df["eff_hand"] == "L", _lrate - _rrate,
        np.where(df["eff_hand"] == "R", _rrate - _lrate, np.nan))
    # wind carry onto the pull field + general carry; wind x short porch
    add_wind_carry(df, pull=True)                     # bat_wind_pull, wind_carry
    df["bat_wind_porch"] = df["bat_wind_pull"] * (330.0 - df["pull_fence"])
    # hot + high air both add carry (product of centered Temp / Elevation)
    df["carry_air"] = ((pd.to_numeric(df["Temp"], errors="coerce") - 70.0)
                       * (pd.to_numeric(df["Elevation_ft"], errors="coerce")
                          / 1000.0))
    # batted-ball profile x opponent defense (opp_oaa = defense faced)
    df["bip_gb_def"] = df["bip_gb"] * df["opp_oaa"]
    df["bip_air_def"] = df["bip_pullair"] * df["opp_oaa"]
    # BABIP / luck-regression: recent decayed BA-on-contact minus expected xBA
    _contact = df["_dk_AB"] - df["_dk_SO"]
    df["hit_luck"] = (np.where(_contact > 0, df["_dk_H"] / _contact, np.nan)
                      - df["bip_xba"])
    # umpire zone tendency x pitcher/batter matchup
    df["ump_k_x_pk"] = df["ump_k_pct"] * df["pc_k_bf"]
    df["ump_k_x_bk"] = df["ump_k_pct"] * df["c_k_pct"]
    df["ump_bb_x_pbb"] = df["ump_bb_pct"] * df["pc_bb_bf"]
    # air-density carry (humidity + station pressure; 2026-07-12 batch)
    add_weather_derived(df)
    # thin air x short pull porch, and thin air x the batter's pulled-air
    # tendency — the carry only cashes for hitters who put the ball in the
    # air toward a reachable fence (AIR_RHO0 = league-mean density, so both
    # are signed: heavy air turns them negative)
    df["air_porch"] = ((AIR_RHO0 - df["air_dens"])
                       * (330.0 - df["pull_fence"]))
    df["air_fly"] = (AIR_RHO0 - df["air_dens"]) * df["bip_pullair"]
    # velocity matchup: the batter's whiff-per-swing vs 95+ fastballs x how
    # far above league this starter's decayed fastball velo sits — the
    # weakness only materializes when tonight's arm can exploit it
    df["bat_velo_matchup"] = (df["bd_fb95wh_d"]
                              * (df["p_fbv_d"] - PD_SHRINK["fbv"][0]))
    # run conversion: his own extra-base advancement skill x the slugging
    # of the two hitters behind him — scoring once aboard needs both
    df["ctx_run_conv"] = df["bat_brr_xb"] * df["ctx_behind_slg"]
    # starter BABIP-luck regression: recent hits per contacted PA minus his
    # decayed expected BA on contact allowed — sequencing luck due to
    # regress (the pitcher sibling of the batter-side hit_luck)
    _pcon = 1.0 - df["p5_k_bf"] - df["p5_bb_bf"]
    df["p_hit_luck"] = (np.where(_pcon > 0, df["p5_h_bf"] / _pcon, np.nan)
                        - df["pbipd_xba"])
    # legs x ground balls: fast grounder hitters beat out infield hits
    # (sprint centered at the ~27 ft/s league average)
    df["bat_leg_hits"] = (df["bat_sprint"] - 27.0) * df["bip_gb"]
    # in-zone whiff skill x a zone-pounding starter (both centered): contact
    # hitters feast on strike-throwers, whiffers get buried by them
    df["zone_whiff_matchup"] = ((df["bd_zwsw_d"] - PD_SHRINK["zwsw"][0])
                                * (df["p_zone_d"] - PD_SHRINK["zone"][0]))
    # style-collision products (2026-07-12): batted-ball outcomes are
    # MULTIPLICATIVE in the two sides' tendencies — a flyball hitter vs a
    # flyball pitcher compounds air-ball probability in a way additive
    # tree splits on the mains can't express (log5 idea, batted-ball form)
    df["mix_air"] = (1.0 - df["bip_gb"]) * (1.0 - df["pbip_gb"])
    df["mix_brl"] = df["bipd_brl"] * df["pbipd_brl"]
    df["mix_xwcon"] = df["bipd_xwoba"] * df["pbipd_xwoba"]
    # chaser vs chase-hunter (both centered): a disciplined batter starves
    # a chase-dependent starter into the zone; a chaser feeds him
    df["chase_matchup"] = ((df["bd_chase_d"] - PD_SHRINK["chase"][0])
                           * (df["p_chase_d"] - PD_SHRINK["chase"][0]))
    # pitch-class arsenal collision (v3 schema): expected whiff-per-swing
    # vs THIS starter's actual mix — his usage shares weighting the
    # batter's whiff splits by class (fastball bucket = remainder). A
    # 3-share x 3-split log5 sum no tree can assemble from the mains.
    _fb_share = 1.0 - df["p_brk_d"] - df["p_off_d"]
    df["arsenal_whiff"] = (_fb_share * df["bd_fbwh_d"]
                           + df["p_brk_d"] * df["bd_brkwh_d"]
                           + df["p_off_d"] * df["bd_offwh_d"])
    # centered class deviations: breaking-vulnerable batter x breaking-
    # heavy pitcher (and the offspeed twin)
    df["brk_matchup"] = ((df["bd_brkwh_d"] - PD_SHRINK["brkwh"][0])
                         * (df["p_brk_d"] - PD_SHRINK["brk"][0]))
    df["off_matchup"] = ((df["bd_offwh_d"] - PD_SHRINK["offwh"][0])
                         * (df["p_off_d"] - PD_SHRINK["off"][0]))
    # first-pitch collision: an aggressive 0-0 swinger vs a first-pitch
    # strike-thrower settles the AB early — fewer walks and fewer deep
    # counts, in a way neither main expresses alone
    df["fp_matchup"] = ((df["bd_fpsw_d"] - PD_SHRINK["fpsw"][0])
                        * (df["p_fps_d"] - PD_SHRINK["fps"][0]))
    # graded velocity-band collision (v4): the batter's whiff splits by
    # fastball band weighted by THIS starter's banded usage — the shaped
    # version of bat_velo_matchup (velocity effects on whiff aren't
    # linear; a 92-95 sinkerballer and a 97 flamethrower attack the same
    # weakness very differently)
    df["velo_band_whiff"] = (df["p_fblou_d"] * df["bd_fblowh_d"]
                             + df["p_fbmidu_d"] * df["bd_fbmidwh_d"]
                             + df["p_fb95u_d"] * df["bd_fb95wh_d"])
    # two-strike collision (v5): a batter who folds with two strikes vs
    # a pitcher who finishes — the put-away endgame both K props and the
    # contact props feel (centered, like the other style products)
    df["ts_matchup"] = ((df["bd_tswh_d"] - PD_SHRINK["tswh"][0])
                        * (df["p_tswh_d"] - PD_SHRINK["tswh"][0]))
    # starter times-through-order decay (shrunk 3rd-vs-1st contact quality)
    df["p_tto_decay"] = tto_decay_from_sums(df)
    # batted-ball profile x the ACTUAL lineup defense behind tonight's
    # starter (player-level prior-season OAA, IF/OF split) — sharper than
    # the team-season opp_oaa interactions above
    df["bip_gb_def_if"] = df["bip_gb"] * df["opp_def_p_if"]
    df["bip_air_def_of"] = df["bip_pullair"] * df["opp_def_p_of"]
    # BvP shrunk residuals off the batter's own as-of contact baseline (bip_xwoba)
    _own = df["bip_xwoba"]
    _bn = df["bvp_cum_n"]
    df["bvp_n"] = np.log1p(_bn.fillna(0.0))
    df["bvp_xwoba_resid"] = ((df["bvp_cum_xw_sum"] - df["bvp_cum_xw_n"] * _own)
                             / (df["bvp_cum_xw_n"] + BVP_K_XW)).fillna(0.0)
    df["bvp_hr_resid"] = ((df["bvp_cum_hr_n"] - _bn * BVP_HR_PRIOR)
                          / (_bn + BVP_K_HR)).fillna(0.0)
    # ---- exposure products (2026-07-12 closing sweep): every prop is
    # ~ 1-(1-p)^PA — per-PA skill TIMES plate appearances. Both terms are in
    # the frame but trees can't multiply them; these are each head's natural
    # Poisson mean ("expected HRs tonight"), the most direct signal the
    # model never saw.
    df["xpa_x_hr"] = df["xpa_slot"] * df["c_hr_pa_sh"]
    df["xpa_x_hit"] = df["xpa_slot"] * df["c_h_ab_sh"]
    df["xpa_x_tb"] = df["xpa_slot"] * df["c_tb_ab_sh"]
    df["xpa_x_k"] = df["xpa_slot"] * df["c_k_pct_sh"]
    df["xpa_x_rbi"] = df["xpa_slot"] * df["c_rbi_pa_sh"]
    df["xpa_x_r"] = df["xpa_slot"] * df["c_r_pa_sh"]
    # ---- box-score rate collisions (log5): shrunk batter rate x the
    # starter's allowed rate — the outcome-level siblings of the contact
    # style products (mix_air/mix_brl/mix_xwcon)
    df["mix_k"] = df["c_k_pct_sh"] * df["pc_k_bf"]
    df["mix_bb"] = df["c_bb_pct_sh"] * df["pc_bb_bf"]
    df["mix_hr"] = df["c_hr_pa_sh"] * df["pc_hr_bf"]
    df["mix_hit"] = df["c_h_ab_sh"] * df["pc_h_bf"]
    df["mix_gb"] = df["bip_gb"] * df["pbip_gb"]       # rally-killer/GIDP side
    df["mix_ld"] = df["bipd_ld"] * df["pbipd_ld"]     # line-drive collision
    # audit wave (rank 5): the pull-air collision — pulled air runs 18.4%
    # HR per air ball vs 5.1% other; both sides' tendencies compound
    df["mix_pullair"] = df["bip_pullair"] * df["pbip_pullair"]
    # ---- conversion chains: run production is a product of sequential
    # events, not a sum
    df["rbi_conv"] = df["rbi_opp_obp"] * df["c_tb_ab_sh"]  # runners on x drive
    df["run_opp"] = df["c_obp"] * df["ctx_behind_slg"]     # get on x get driven
    # park HR environment x the batter's own HR skill
    df["park_x_hr"] = df["park_hr_pg"] * df["c_hr_pa_sh"]
    # starter K-BB rate: the classic single-number pitcher skill — trees
    # can't subtract any more than they can multiply
    df["p_kbb"] = df["pc_k_bf"] - df["pc_bb_bf"]
    # ---- bullpen exposure (2026-07-14 #15): every batter prop is secretly
    # a two-pitcher problem. The starter faces ~BF_st batters (his season
    # BF per start); the batter at slot s takes his k-th PA at team-PA
    # number s + 9(k-1), so his expected PAs against the PEN are whatever
    # xpa_slot exceeds the count that fits under BF_st. Collision columns
    # price WHO those late PAs meet: pen-minus-starter K/HR/H rate deltas.
    _bf_st = df["ps_BF"] / df["p_starts_season"]
    _pa_vs_st = np.clip(np.floor((_bf_st - df["slot"]) / 9.0) + 1.0,
                        0.0, None)
    df["xpa_pen"] = np.clip(df["xpa_slot"] - _pa_vs_st, 0.0, None)
    df["xpen_k"] = df["xpa_pen"] * (df["pen_k_bf"] - df["ps_k_bf"])
    df["xpen_hr"] = df["xpa_pen"] * (df["pen_hr_bf"] - df["ps_hr_bf"])
    df["xpen_h"] = df["xpa_pen"] * (df["pen_h_bf"] - df["ps_h_bf"])
    # ---- form-weighted exposure (2026-07-14 #31): the xpa_x_* products
    # use CAREER rates; the decayed variants price current form x
    # opportunity. Credit-splitting with the career versions is expected —
    # the selection vote keeps whichever carries each head.
    df["xpa_x_hr_d"] = df["xpa_slot"] * df["d_hr_pa_sh"]
    df["xpa_x_tb_d"] = df["xpa_slot"] * df["d_tb_ab_sh"]
    df["xpa_x_k_d"] = df["xpa_slot"] * df["d_k_pct_sh"]
    # ---- ride collisions (2026-07-14 #27): high-IVB fastballs live at the
    # top of the zone — whiffs for fastball-vulnerable swings, air balls
    # for flyball swings. Centered at the league IVB prior.
    _ivb_c = df["p_ivb_d"] - PD_SHRINK["ivb"][0]
    df["ride_matchup"] = _ivb_c * (df["bd_fbwh_d"] - PD_SHRINK["fbwh"][0])
    df["ride_air"] = _ivb_c * (1.0 - df["bip_gb"])
    # ---- audit wave (2026-07-14, Model/AUDIT_BUILD_SPECS_0714.md) ----
    # rank 1: exposure completion — the #1 feature family (xpa_x_*) had no
    # bb/sb member and the new xbb head shipped with zero exposure
    # coverage; xpen_bb prices late-PA walk exposure against the pen
    df["xpa_x_bb"] = df["xpa_slot"] * df["c_bb_pct_sh"]
    df["xpa_x_sb"] = df["xpa_slot"] * df["c_sb_pa_sh"]
    df["xpa_x_bb_d"] = df["xpa_slot"] * df["d_bb_pct_sh"]
    df["xpa_x_sb_d"] = df["xpa_slot"] * df["d_sb_pa_sh"]
    df["xpen_bb"] = df["xpa_pen"] * (df["pen_bb_bf"] - df["ps_bb_bf"])
    # rank 3: park hit-type products — park_x_hr's proven template for the
    # two hit types that never got it (the triple head's park geometry)
    df["park_x_2b"] = df["park_2b_pg"] * df["c_2b_ab_sh"]
    df["park_x_3b"] = df["park_3b_pg"] * df["c_3b_ab_sh"]
    # rank 4: sb opportunity chain — exposure x reach-first x intent, and
    # the chain against tonight's starter's steal permissiveness (centered
    # on the PRIOR-season league SB/27 so the 2023 rules regime shift
    # doesn't flip the sign era-wide); plus the traffic cross for run heads
    df["sb_chain"] = df["xpa_slot"] * df["ob1_sh"] * df["sb_att1"]
    df["sb_chain_env"] = df["sb_chain"] * (df["psb_sb27"]
                                           - df["lg_sb27_prior"])
    df["ob1_x_st"] = ((df["ob1_sh"] - OB1_PRIOR)
                      * (df["ps_h_bf"] + df["ps_bb_bf"] - ST_REACH_PRIOR))
    # rank 7: passivity collisions — a two-strike taker facing an edge-
    # painter is the called-strike-three geometry; a taker under a
    # K-leaning ump compounds
    df["called_k_matchup"] = ((PD_SHRINK["tssw"][0] - df["bd_tssw_d"])
                              * (df["p_edge_d"] - PD_SHRINK["edge"][0]))
    df["ump_k_x_take"] = df["ump_k_pct"] * (1.0 - df["bd_tssw_d"])
    # rank 13: oppo-side spray geometry — the fence the pull-only encoding
    # cannot see (13.2% of HRs are oppo)
    df["oppo_fence"] = np.where(df["eff_hand"] == "L", df["LF"],
                                np.where(df["eff_hand"] == "R", df["RF"],
                                         np.nan))
    df["oppo_air_porch"] = (330.0 - df["oppo_fence"]) * df["bip_oppoair"]
    # rank 14: realized RBI opportunity/conversion — LOB finally consumed.
    # Decayed conversion minus career = running hot/cold cashing runners
    # (hit_luck sibling); d_rob_pa = realized traffic per PA.
    _p0, _k0 = SHRINK["rbi_conv"][0], SHRINK["rbi_conv"][1]
    _dconv = ((df["_dk_RBI"] + _k0 * _p0)
              / (df["_dk_RBI"] + df["_dk_LOB"] + _k0))
    df["rbi_conv_gap"] = _dconv - df["c_rbi_conv_sh"]
    df["d_rob_pa"] = ((df["_dk_RBI"] + df["_dk_LOB"])
                      / df["_dk_PA"].where(df["_dk_PA"] > 0))
    # rank 23: ump x batter walk cell — completes the ump-matchup 2x2
    # (shrunk rate, matching mix_bb; the K-cell's unshrunk choice is the
    # noisier one)
    df["ump_bb_x_bb"] = df["ump_bb_pct"] * df["c_bb_pct_sh"]
    # rank 31: protection — measured POSITIVE interaction (protection
    # modulates walks mostly for NON-sluggers); deliberately unclipped
    df["pitch_around"] = ((df["c_iso_sh"] - SHRINK["iso"][0])
                          * (df["ctx_behind_slg_d"] - CTX_BSLG_MEAN))
    # rank 27: the two-strike x breaking putaway cell crossed with the
    # starter's breaking usage — the joint cell sits at its own level
    # (0.302 vs marginals 0.250/0.329), unreachable from the marginals
    df["ts_brk_matchup"] = ((df["bd_tsbrkwh_d"] - PD_SHRINK["tsbrkwh"][0])
                            * (df["p_brk_d"] - PD_SHRINK["brk"][0]))
    # ---- v8 damage-on-contact collisions (2026-07-15): the damage
    # siblings of bat_velo_matchup / velo_band_whiff / arsenal_whiff —
    # whiff says he misses 95+; these price what happens when he doesn't.
    # Batter side rides the CAREER reads (bd_*xw_c): a median season has
    # ~18 BBE vs 95+, so the decayed band rates are shrink-dominated and
    # the collision would collapse toward prior x velo (see PD_SHRINK).
    df["bat_velo_damage"] = (df["bd_fb95xw_c"]
                             * (df["p_fbv_d"] - PD_SHRINK["fbv"][0]))
    df["velo_band_damage"] = (df["p_fblou_d"] * df["bd_fbloxw_c"]
                              + df["p_fbmidu_d"] * df["bd_fbmidxw_c"]
                              + df["p_fb95u_d"] * df["bd_fb95xw_c"])
    df["arsenal_damage"] = (_fb_share * df["bd_fbkxw_c"]
                            + df["p_brk_d"] * df["bd_brkxw_c"]
                            + df["p_off_d"] * df["bd_offxw_c"])
    # elite-velo damage cell, mix_ (log5) form: his contact quality vs 95+
    # x how hard the starter's 95+ gets hit when it's touched
    df["mix_fb95xw"] = df["bd_fb95xw_c"] * df["p_fb95xw_d"]
    # 2K x elite-velo putaway collision (ts_brk_matchup's velocity mirror)
    df["ts_fb95_matchup"] = ((df["bd_tsfb95wh_d"]
                              - PD_SHRINK["tsfb95wh"][0])
                             * (df["p_fb95u_d"] - PD_SHRINK["fb95u"][0]))
    # ---- battery collisions (2026-07-15): the opposing catchers' framing
    # x a two-strike taker (called-strike-three geometry — ump_k_x_take's
    # battery sibling), and the SB opportunity chain against the battery's
    # ERA-CENTERED pop time (slower than that season's league =
    # permissive; see the PopC note in load_raw) — the catcher half
    # sb_chain_env's pitcher half never had
    df["frame_x_take"] = df["opp_cat_frame"] * (1.0 - df["bd_tssw_d"])
    df["sb_cat_env"] = df["sb_chain"] * df["opp_cat_popc"]
    # rank 33: the batter's platoon edges vs his own career baseline, both
    # TB/PA-shrunk (denominator-clean) — lineup_aggregates averages these
    # per posted lineup into the team frame's off_lu_vsh_edge/kedge
    df["_vsh_tb_edge"] = (df["vsh_tb_ab_sh"]
                          - _shrink(df["c_TB"], df["c_PA"], "tb_ab"))
    df["_vsh_k_edge"] = df["vsh_k_pct_sh"] - df["c_k_pct_sh"]
    return df


def build_batter_frame(raw):
    """Training frame: one row per starting batter per game, with targets."""
    gb, gp = raw["gb"], raw["gp"]
    df = _batter_asof(gb)

    # starters only (matches what the GUI knows before a game)
    df = df[df["BattingOrder"].notna()].copy()
    df["slot"] = pd.to_numeric(df["BattingOrder"], errors="coerce")
    df = df[(df["slot"] % 100 == 0)].copy()
    df["slot"] = (df["slot"] // 100).astype(int)

    # expected exposure at tonight's slot: as-of league PA/G at that
    # lineup slot (see XPA_PRIOR note; the shared table guarantees the
    # inference lookup computes the identical number)
    df = df.merge(_slot_pa_table(gb)[["slot", "Date", "xpa_slot"]],
                  on=["slot", "Date"], how="left")

    # opposing starter
    starters = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]].rename(
        columns={"Team": "Opponent", "PlayerId": "StarterId"})
    df = df.merge(starters, on=["GamePk", "Opponent"], how="left")

    st = _starter_asof(gp)
    st_feats = st[["GamePk", "PlayerId", "p_starts_career", "p_starts_season",
                   "p_days_rest", "p_np_last", "p_np_l3",
                   # the opposing starter's venue split for TONIGHT's
                   # context (2026-07-14 #23; his Home = 1 - batter's)
                   "pvloc_era", "pvloc_k_bf", "pvloc_hr_bf",
                   # audit wave (rank 11): outing-length dispersion + short-
                   # start share — xpa_pen is a convex clip of starter
                   # length, so dispersion shifts pen exposure at fixed mean
                   "p_outs_sd", "p_short_share",
                   *[f"pc_{s}" for s in PIT_STATS], *[f"ps_{s}" for s in PIT_STATS],
                   *[f"p5_{s}" for s in ["BF", "HR", "SO", "BB", "H"]]]].rename(
        columns={"PlayerId": "StarterId"})
    df = df.merge(st_feats, on=["GamePk", "StarterId"], how="left")

    # audit wave (rank 17): the opposing starter's MiLB-translated priors —
    # for debut/thin-history starters every pc_/ps_/p5_/pd_/pars_ main is
    # NaN by construction; this is the only pitcher description available
    # exactly there (89% coverage on debut starts). Exact-season join.
    if raw.get("milb") is not None:
        df = df.merge(_milb_cols(raw["milb"]["pit"], "pmilb_").rename(
            columns={"PlayerId": "StarterId"}),
            on=["StarterId", "Season"], how="left")
    else:
        for c in MILB_PIT_COLS:
            df[c] = np.nan

    # schedule context (2026-07-14 #18/#24): the batter's TEAM's previous-
    # game travel/day-night fatigue + tonight's doubleheader flags
    df = df.merge(_team_sched_table(raw["games"], raw["parks"]),
                  on=["GamePk", "Team"], how="left")

    team_tab = _team_offense_table(gb)
    pen_tab = _bullpen_table(gp)
    park_tab = _park_table(gb, raw["games"])
    df = _attach_context(df, raw, team_tab, pen_tab, park_tab,
                         pen_hl_tab=_bullpen_hl_table(gp),
                         pen_fat_tab=_pen_fatigue_table(gp))
    df = df.merge(_league_env_table(gb), on="Date", how="left")
    # audit wave (rank 15): own team's trailing-30-day BaseRuns sequencing-
    # luck residual — the regression-candidate discount term fused inside
    # toff_r_pg (shipped at team+winner grain, never reached this frame)
    df = df.merge(_team_bsr_table(gb).rename(
        columns={"bsr_luck": "toff_bsr_luck"}), on=["Team", "Date"],
        how="left")

    # own team's offense in TODAY's venue context (home or road split)
    loc = _team_offense_loc_table(gb)
    loc = loc.rename(columns={c: f"tloc_{c}" for c in loc.columns
                              if c.startswith("cum")})
    df = _asof_merge(df, loc, by=["Team", "Season", "Home"])
    df["toff_loc_hr_pa"] = df["tloc_cum_HR"] / df["tloc_cum_PA"]
    df["toff_loc_r_pg"] = df["tloc_cum_R"] / df["tloc_cum_n"]

    hrq = _hr_quality_table(raw["hr"], raw["parks"]).rename(
        columns={"BatterId": "PlayerId"})
    hrq = hrq.rename(columns={"cum_n": "hrq_n", "cum_ev": "hrq_ev",
                              "cum_dist": "hrq_dist", "cum_dist_max": "hrq_dist_max",
                              "cum_angle": "hrq_angle"})
    df = _asof_merge(df, hrq, by=["PlayerId"])
    df["hrq_ev_avg"] = df["hrq_ev"] / df["hrq_n"]
    df["hrq_dist_avg"] = df["hrq_dist"] / df["hrq_n"]
    df["hrq_angle_avg"] = df["hrq_angle"] / df["hrq_n"]

    # opposing starter's HR quality ALLOWED (how hard/far he gets hit)
    phrq = _pitcher_hr_allowed_table(raw["hr"], gp, raw["parks"]).rename(
        columns={"PitcherId": "StarterId", "cum_n": "phrq_n",
                 "cum_ev": "phrq_ev", "cum_dist": "phrq_dist"})
    phrq["StarterId"] = phrq["StarterId"].astype(df["StarterId"].dtype)
    df = _asof_merge(df, phrq, by=["StarterId"])
    df["phrq_ev_avg"] = df["phrq_ev"] / df["phrq_n"]

    # HR-by-pitch-type matchup: share of the batter's career homers that
    # came on the pitches this starter actually throws (usage-weighted)
    df = df.reset_index(drop=True)
    num = _hrpt_scores(df, raw["hr"], raw["ars_p"]).reindex(df.index)
    df["hrpt_score"] = np.where(df["hrq_n"] > 0, num / df["hrq_n"], np.nan)

    # Statcast contact quality (every batted ball, not just homers): the
    # batter's career + 90-day-decayed profile, and the opposing starter's
    # contact quality ALLOWED. Frames build without the file (all NaN).
    if raw.get("bip") is not None:
        bt = _bip_table(raw["bip"], "BatterId").rename(
            columns={"BatterId": "PlayerId"})
        bt = bt.rename(columns={c: f"bb_{c}" for c in bt.columns
                                if c.startswith(("cum_", "dk_"))})
        df = _asof_merge(df, bt, by=["PlayerId"])
        pt = _bip_table(raw["bip"], "PitcherId").rename(
            columns={"PitcherId": "StarterId"})
        pt["StarterId"] = pt["StarterId"].astype(df["StarterId"].dtype)
        pt = pt.rename(columns={c: f"pb_{c}" for c in pt.columns
                                if c.startswith(("cum_", "dk_"))})
        df = _asof_merge(df, pt, by=["StarterId"])
        # dk_* sums are exp(lam*t)-scaled to each TABLE date; discount them
        # to the game row's date so the shrink K operates in real
        # batted-ball units (rates alone would cancel the scale, but the
        # +K*prior terms would not)
        wdn_row = np.exp(-DECAY_LAM * (df["Date"] - DECAY_EPOCH)
                         .dt.days.to_numpy(dtype="float64"))
        for side, tag in (("bb", "bip"), ("pb", "pbip")):
            feats = bip_feats(lambda c, s=side: df[f"{s}_cum_{c}"], tag)
            feats.update(bip_feats(
                lambda c, s=side: df[f"{s}_dk_{c}"] * wdn_row,
                f"{tag}d", BIP_DECAYED))
            for k, v in feats.items():
                df[k] = v
            df[f"{tag}_n"] = df[f"{side}_cum_n"]
            df[f"{tag}d_n"] = df[f"{side}_dk_n"] * wdn_row
    else:
        for tag in ("bip", "pbip"):
            for name in BIP_SHRINK:
                df[f"{tag}_{name}"] = np.nan
            for name in BIP_DECAYED:
                df[f"{tag}d_{name}"] = np.nan
            df[f"{tag}_n"] = np.nan
            df[f"{tag}d_n"] = np.nan

    # batter-vs-pitcher direct history (BvP): as-of pairwise contact sums
    # (bvp_cum_*). The shrunk residuals off the batter's own baseline (bip_xwoba)
    # are derived in add_batter_derived (shared with serving). NaN cols when
    # there is no BIP file so the shared function still resolves.
    if raw.get("bip") is not None:
        bvt = _bvp_table(raw["bip"]).rename(
            columns={"BatterId": "PlayerId", "PitcherId": "StarterId"})
        bvt["StarterId"] = bvt["StarterId"].astype(df["StarterId"].dtype)
        df = _asof_merge(df, bvt, by=["PlayerId", "StarterId"])
    else:
        for c in ("bvp_cum_n", "bvp_cum_xw_n", "bvp_cum_xw_sum", "bvp_cum_hr_n"):
            df[c] = np.nan

    # opposing starter's times-through-order decay (tto_cum_*): the shrunk
    # 3rd-vs-1st difference is derived in add_batter_derived (shared with
    # serving). NaN cols when there is no BIP file.
    if raw.get("bip") is not None:
        tt = _tto_table(raw["bip"]).rename(columns={"PitcherId": "StarterId"})
        tt["StarterId"] = tt["StarterId"].astype(df["StarterId"].dtype)
        df = _asof_merge(df, tt, by=["StarterId"])
    else:
        for c in ("tto_cum_xw1_n", "tto_cum_xw1_sum",
                  "tto_cum_xw3_n", "tto_cum_xw3_sum"):
            df[c] = np.nan

    # plate discipline (pitch-level dailies): the batter's whiff-per-swing
    # and chase rates (career + 90-day decay) and the opposing starter's
    # decayed swinging-strike rate — swing decisions are the fastest-
    # stabilizing skills in the sport and none of them are in box scores
    if raw.get("pdb") is not None:
        bt2 = _cum_decay_table(raw["pdb"], "PlayerId")
        bt2 = bt2.rename(columns={c: f"bd_{c}" for c in bt2.columns
                                  if c.startswith(("cum_", "dk_"))})
        df = _asof_merge(df, bt2, by=["PlayerId"])
    if raw.get("pdp") is not None:
        po = _cum_decay_table(raw["pdp"], "PlayerId")[
            ["PlayerId", "Date", "dk_wh_n", "dk_n", "dk_fb_v", "dk_fb_n",
             "dk_z_n", "dk_oz_sw", "dk_oz_n",
             "dk_brk_n", "dk_off_n", "dk_edge_n", "dk_fp_n", "dk_fp_s",
             "dk_fblo_n", "dk_fbmid_n", "dk_fb95_n",
             "dk_ts_wh", "dk_ts_sw", "dk_f32_b", "dk_f32_n", "dk_fb_v2",
             "dk_rp_n", "dk_rp_x", "dk_rp_x2", "dk_rp_z", "dk_rp_z2",
             "dk_ivb_n", "dk_ivb_sum",
             "dk_fbstr_n", "dk_fbstr_v", "dk_fbe_n", "dk_fbe_sum",
             "dk_rpf_n", "dk_rpf_x", "dk_rpf_z",
             "dk_rpb_n", "dk_rpb_x", "dk_rpb_z",
             "dk_brkmov_n", "dk_brkmov_sum",
             "dk_fblo_bip", "dk_fblo_xw", "dk_fbmid_bip", "dk_fbmid_xw",
             "dk_fb95_bip", "dk_fb95_xw", "dk_brk_bip", "dk_brk_xw",
             "dk_off_bip", "dk_off_xw", "dk_fbk_bip", "dk_fbk_xw"]
        ].rename(
            columns={"PlayerId": "StarterId", "dk_wh_n": "pdo_dk_wh",
                     "dk_n": "pdo_dk_n", "dk_fb_v": "pdo_dk_fbv",
                     "dk_fb_n": "pdo_dk_fbn", "dk_z_n": "pdo_dk_zn",
                     "dk_oz_sw": "pdo_dk_ozsw", "dk_oz_n": "pdo_dk_ozn",
                     "dk_brk_n": "pdo_dk_brkn", "dk_off_n": "pdo_dk_offn",
                     "dk_edge_n": "pdo_dk_edgen", "dk_fp_n": "pdo_dk_fpn",
                     "dk_fp_s": "pdo_dk_fps",
                     "dk_fblo_n": "pdo_dk_fblon",
                     "dk_fbmid_n": "pdo_dk_fbmidn",
                     "dk_fb95_n": "pdo_dk_fb95n",
                     "dk_ts_wh": "pdo_dk_tswh", "dk_ts_sw": "pdo_dk_tssw",
                     "dk_f32_b": "pdo_dk_f32b", "dk_f32_n": "pdo_dk_f32n",
                     "dk_fb_v2": "pdo_dk_fbv2",
                     "dk_rp_n": "pdo_dk_rpn", "dk_rp_x": "pdo_dk_rpx",
                     "dk_rp_x2": "pdo_dk_rpx2", "dk_rp_z": "pdo_dk_rpz",
                     "dk_rp_z2": "pdo_dk_rpz2",
                     "dk_ivb_n": "pdo_dk_ivbn",
                     "dk_ivb_sum": "pdo_dk_ivbsum",
                     "dk_fbstr_n": "pdo_dk_fbstrn",
                     "dk_fbstr_v": "pdo_dk_fbstrv",
                     "dk_fbe_n": "pdo_dk_fben",
                     "dk_fbe_sum": "pdo_dk_fbesum",
                     "dk_rpf_n": "pdo_dk_rpfn", "dk_rpf_x": "pdo_dk_rpfx",
                     "dk_rpf_z": "pdo_dk_rpfz",
                     "dk_rpb_n": "pdo_dk_rpbn", "dk_rpb_x": "pdo_dk_rpbx",
                     "dk_rpb_z": "pdo_dk_rpbz",
                     "dk_brkmov_n": "pdo_dk_brkmovn",
                     "dk_brkmov_sum": "pdo_dk_brkmovsum",
                     "dk_fblo_bip": "pdo_dk_flobip",
                     "dk_fblo_xw": "pdo_dk_floxw",
                     "dk_fbmid_bip": "pdo_dk_fmidbip",
                     "dk_fbmid_xw": "pdo_dk_fmidxw",
                     "dk_fb95_bip": "pdo_dk_f95bip",
                     "dk_fb95_xw": "pdo_dk_f95xw",
                     "dk_brk_bip": "pdo_dk_brkbip",
                     "dk_brk_xw": "pdo_dk_brkxw",
                     "dk_off_bip": "pdo_dk_offbip",
                     "dk_off_xw": "pdo_dk_offxw",
                     "dk_fbk_bip": "pdo_dk_fbkbip",
                     "dk_fbk_xw": "pdo_dk_fbkxw"})
        po["StarterId"] = po["StarterId"].astype(df["StarterId"].dtype)
        df = _asof_merge(df, po, by=["StarterId"])
    wdn2 = np.exp(-DECAY_LAM * (df["Date"] - DECAY_EPOCH)
                  .dt.days.to_numpy(dtype="float64"))
    if raw.get("pdb") is not None:
        # zwsw/fb95wh/v3 splits (2026-07-12): NaN day-sums under a pre-
        # backfill file propagate to NaN features — same gating as every
        # optional source
        for name in PD_BATTER:
            prior, k, num, den = PD_SHRINK[name]
            df[f"bd_{name}_c"] = ((df[f"bd_cum_{num}"] + k * prior)
                                  / (df[f"bd_cum_{den}"] + k))
            df[f"bd_{name}_d"] = ((df[f"bd_dk_{num}"] * wdn2 + k * prior)
                                  / (df[f"bd_dk_{den}"] * wdn2 + k))
    else:
        for name in PD_BATTER:
            df[f"bd_{name}_c"] = np.nan
            df[f"bd_{name}_d"] = np.nan
    if raw.get("pdp") is not None:
        prior, k, _, _ = PD_SHRINK["swstr"]
        df["p_swstr_d"] = ((df["pdo_dk_wh"] * wdn2 + k * prior)
                           / (df["pdo_dk_n"] * wdn2 + k))
        # opposing starter's decayed fastball velo (velocity-matchup input;
        # same shrink as the starts frame's pd_fbv_d)
        prior, k, _, _ = PD_SHRINK["fbv"]
        df["p_fbv_d"] = ((df["pdo_dk_fbv"] * wdn2 + k * prior)
                         / (df["pdo_dk_fbn"] * wdn2 + k))
        # ... and his decayed zone share (zone-pounder vs zone-avoider),
        # the other half of the zone_whiff_matchup interaction
        prior, k, _, _ = PD_SHRINK["zone"]
        df["p_zone_d"] = ((df["pdo_dk_zn"] * wdn2 + k * prior)
                          / (df["pdo_dk_n"] * wdn2 + k))
        # ... and his decayed chase-INDUCED rate (the other half of the
        # chase_matchup style product)
        prior, k, _, _ = PD_SHRINK["chase"]
        df["p_chase_d"] = ((df["pdo_dk_ozsw"] * wdn2 + k * prior)
                           / (df["pdo_dk_ozn"] * wdn2 + k))
        # ... and his pitch-class usage mix, shadow-band share (command
        # proxy) and first-pitch strike tendency (v3 schema; NaN until the
        # backfilled file carries the counts)
        for nm, num_col in (("brk", "pdo_dk_brkn"), ("off", "pdo_dk_offn"),
                            ("edge", "pdo_dk_edgen"),
                            ("fblou", "pdo_dk_fblon"),
                            ("fbmidu", "pdo_dk_fbmidn"),
                            ("fb95u", "pdo_dk_fb95n")):
            prior, k, _, _ = PD_SHRINK[nm]
            df[f"p_{nm}_d"] = ((df[num_col] * wdn2 + k * prior)
                               / (df["pdo_dk_n"] * wdn2 + k))
        prior, k, _, _ = PD_SHRINK["fps"]
        df["p_fps_d"] = ((df["pdo_dk_fps"] * wdn2 + k * prior)
                         / (df["pdo_dk_fpn"] * wdn2 + k))
        # v5: put-away whiff, 3-2 walks gifted, and the dispersion reads
        prior, k, _, _ = PD_SHRINK["tswh"]
        df["p_tswh_d"] = ((df["pdo_dk_tswh"] * wdn2 + k * prior)
                          / (df["pdo_dk_tssw"] * wdn2 + k))
        prior, k, _, _ = PD_SHRINK["f32b"]
        df["p_f32b_d"] = ((df["pdo_dk_f32b"] * wdn2 + k * prior)
                          / (df["pdo_dk_f32n"] * wdn2 + k))
        df["p_fbv_sd"] = velo_sd_from_sums(
            df["pdo_dk_fbn"] * wdn2, df["pdo_dk_fbv"] * wdn2,
            df["pdo_dk_fbv2"] * wdn2)
        df["p_rel_sd"] = release_scatter_from_sums(
            df["pdo_dk_rpn"] * wdn2, df["pdo_dk_rpx"] * wdn2,
            df["pdo_dk_rpx2"] * wdn2, df["pdo_dk_rpz"] * wdn2,
            df["pdo_dk_rpz2"] * wdn2)
        # v6: the opposing starter's decayed FF ride (feeds the ride
        # collisions in add_batter_derived)
        prior, k, _, _ = PD_SHRINK["ivb"]
        df["p_ivb_d"] = ((df["pdo_dk_ivbsum"] * wdn2 + k * prior)
                         / (df["pdo_dk_ivbn"] * wdn2 + k))
        # v7 audit wave: the opposing starter's perceived-velo premium,
        # breaking-movement magnitude, base-out stretch delta and
        # per-class release separation (cause axes; NaN pre-v7 rows)
        prior, k, _, _ = PD_SHRINK["effprem"]
        df["p_effprem_d"] = ((df["pdo_dk_fbesum"] * wdn2 + k * prior)
                             / (df["pdo_dk_fben"] * wdn2 + k))
        prior, k, _, _ = PD_SHRINK["brkmov"]
        df["p_brkmov_d"] = ((df["pdo_dk_brkmovsum"] * wdn2 + k * prior)
                            / (df["pdo_dk_brkmovn"] * wdn2 + k))
        df["p_stretch_d"] = stretch_vdelta_from_sums(
            df["pdo_dk_fbn"] * wdn2, df["pdo_dk_fbv"] * wdn2,
            df["pdo_dk_fbstrn"] * wdn2, df["pdo_dk_fbstrv"] * wdn2)
        df["p_relsep_d"] = release_sep_from_sums(
            df["pdo_dk_rpfn"] * wdn2, df["pdo_dk_rpfx"] * wdn2,
            df["pdo_dk_rpfz"] * wdn2, df["pdo_dk_rpbn"] * wdn2,
            df["pdo_dk_rpbx"] * wdn2, df["pdo_dk_rpbz"] * wdn2)
        # v8: his damage-ALLOWED splits (xwOBA per BBE by velo band /
        # pitch class) — does the stuff hold up when it gets touched
        for nm, num_col, den_col in (
                ("fbloxw", "pdo_dk_floxw", "pdo_dk_flobip"),
                ("fbmidxw", "pdo_dk_fmidxw", "pdo_dk_fmidbip"),
                ("fb95xw", "pdo_dk_f95xw", "pdo_dk_f95bip"),
                ("brkxw", "pdo_dk_brkxw", "pdo_dk_brkbip"),
                ("offxw", "pdo_dk_offxw", "pdo_dk_offbip"),
                ("fbkxw", "pdo_dk_fbkxw", "pdo_dk_fbkbip")):
            prior, k, _, _ = PD_SHRINK[nm]
            df[f"p_{nm}_d"] = ((df[num_col] * wdn2 + k * prior)
                               / (df[den_col] * wdn2 + k))
    else:
        df["p_swstr_d"] = np.nan
        df["p_fbv_d"] = np.nan
        df["p_zone_d"] = np.nan
        df["p_chase_d"] = np.nan
        for c in ("p_brk_d", "p_off_d", "p_edge_d", "p_fps_d",
                  "p_fblou_d", "p_fbmidu_d", "p_fb95u_d",
                  "p_tswh_d", "p_f32b_d", "p_fbv_sd", "p_rel_sd",
                  "p_ivb_d",
                  "p_effprem_d", "p_brkmov_d", "p_stretch_d",
                  "p_relsep_d",
                  "p_fbloxw_d", "p_fbmidxw_d", "p_fb95xw_d",
                  "p_brkxw_d", "p_offxw_d", "p_fbkxw_d"):
            df[c] = np.nan

    # prior-season sprint speed (raw footspeed for the SB/run props) and
    # the OPPONENT's prior-season team defense (outs above average) —
    # leakage-free like GO/AO: a 2026 game sees 2025 measurements
    if raw.get("sprint") is not None:
        sp = raw["sprint"].rename(columns={"SprintSpeed": "bat_sprint",
                                           "HPto1B": "bat_hp1b"})
        df = _merge_prior_season(
            df, sp[["PlayerId", "Year", "bat_sprint", "bat_hp1b"]],
            "PlayerId", ["bat_sprint", "bat_hp1b"])
    else:
        df["bat_sprint"] = np.nan
        df["bat_hp1b"] = np.nan
    # prior-season Statcast bat tracking (bat speed / swing) — power signal
    # for the HR / total-base props. COVERAGE: 2023+ only, so under the
    # current selection suite (train <=2023) every training row is NaN and
    # the feature is INERT (unlearnable) — it is wired now so it activates
    # automatically once the season rollover puts a covered year into the
    # training window (~2027); see BAT_TRACK_COLS / train._BAT.
    if raw.get("bat_track") is not None:
        bt = raw["bat_track"].rename(columns=BAT_TRACK_REN)
        df = _merge_prior_season(df, bt[["PlayerId", "Year", *BAT_TRACK_COLS]],
                                 "PlayerId", BAT_TRACK_COLS)
    else:
        for c in BAT_TRACK_COLS:
            df[c] = np.nan
    if raw.get("oaa") is not None:
        oa = raw["oaa"].rename(columns={"Team": "PlayerId",
                                        "OAA_per162": "opp_oaa"})
        df = _merge_prior_season(df, oa[["PlayerId", "Year", "opp_oaa"]],
                                 "Opponent", ["opp_oaa"])
    else:
        df["opp_oaa"] = np.nan
    # the OPPONENT's battery, prior season (2026-07-15): framing runs per
    # 2000 called pitches (more called strikes AGAINST this batter), and
    # the running-game wall his SB chain must beat (CS-above-average per
    # attempt + pop time) — team playing-time-weighted, the same
    # serving-safe dodge as team OAA
    if raw.get("cat") is not None:
        ct = raw["cat"].rename(columns={
            "Team": "PlayerId", "FrameRV_pt": "opp_cat_frame",
            "CSAA_att": "opp_cat_csaa", "PopTime": "opp_cat_pop",
            "PopC": "opp_cat_popc"})
        df = _merge_prior_season(
            df, ct[["PlayerId", "Year", "opp_cat_frame", "opp_cat_csaa",
                    "opp_cat_pop", "opp_cat_popc"]],
            "Opponent", ["opp_cat_frame", "opp_cat_csaa", "opp_cat_pop",
                         "opp_cat_popc"])
    else:
        for c in ("opp_cat_frame", "opp_cat_csaa", "opp_cat_pop",
                  "opp_cat_popc"):
            df[c] = np.nan
    # IL return context (2026-07-15): the batter's own last stint and the
    # opposing starter's (a fresh-off-the-IL starter pitches on a leash —
    # complements xpa_pen and the ramp flags)
    df = _il_asof(df, raw.get("il"), "PlayerId", "")
    df = _il_asof(df, raw.get("il"), "StarterId", "p_")
    # the ACTUAL defense behind tonight's opposing starter: mean prior-season
    # player OAA of the OPPONENT's posted lineup (overall + IF/OF splits) —
    # the lineup is known pregame, so this is serving-computable and
    # leakage-free (2026-07-12)
    if raw.get("oaa_players") is not None:
        ld = _lineup_oaa_table(gb, raw["oaa_players"]).rename(
            columns={"Team": "Opponent", "def_p_oaa": "opp_def_p_oaa",
                     "def_p_if": "opp_def_p_if", "def_p_of": "opp_def_p_of"})
        df = df.merge(ld, on=["GamePk", "Opponent"], how="left")
    else:
        for c in ("opp_def_p_oaa", "opp_def_p_if", "opp_def_p_of"):
            df[c] = np.nan
    # prior-season baserunning run value (Savant): total runner runs and the
    # extra-base advancement rate per opportunity — the skill behind scoring
    # runs that raw sprint speed only proxies (2026-07-12)
    if raw.get("baserun") is not None:
        br = raw["baserun"].copy()
        br["bat_brr"] = pd.to_numeric(br["RunnerRuns"], errors="coerce")
        br["bat_brr_xb"] = (pd.to_numeric(br["RunnerRunsXB"], errors="coerce")
                            / pd.to_numeric(br["Opportunities"],
                                            errors="coerce"))
        df = _merge_prior_season(df, br[["PlayerId", "Year", "bat_brr",
                                         "bat_brr_xb"]],
                                 "PlayerId", ["bat_brr", "bat_brr_xb"])
    else:
        df["bat_brr"] = np.nan
        df["bat_brr_xb"] = np.nan
    # MiLB translated prior (2026-07-13, Phase-3 rider): the batter's
    # level-translated minors line — genuinely new information for
    # thin-MLB-history rows; NaN for everyone without kept-level minors
    # in the 3-season window (GBM imputes). Exact-season join by design.
    if raw.get("milb") is not None:
        df = df.merge(_milb_cols(raw["milb"]["bat"], "milb_"),
                      on=["PlayerId", "Season"], how="left")
    else:
        for c in MILB_BAT_COLS:
            df[c] = np.nan

    # prior-season GO/AO (fly-ball tendency) and pitcher SB control
    df = _merge_prior_season(df, _batter_season_table(raw["bat_season"]),
                             "PlayerId", ["bat_goao"])
    df = _merge_prior_season(df, _pitcher_season_table(raw["pit_season"]),
                             "StarterId", ["pit_goao", "psb_sb27", "psb_stop"])
    # audit wave (ranks 4/24): prior-season league SB/27 — regime-aware
    # centering for the steal-permissiveness products
    df = df.merge(_league_sb27_table(raw["pit_season"]), on="Season",
                  how="left")

    # opponent battery: SB allowed per game and the shrunk caught-stealing
    # rate, season-to-date (catcher-inclusive, unlike the psb_* priors)
    tsb = _team_sb_allowed_table(gb).rename(columns={"Team": "Opponent"})
    tsb = tsb.rename(columns={c: f"tsb_{c}" for c in tsb.columns
                              if c.startswith("cum_")})
    df = _asof_merge(df, tsb, by=["Opponent", "Season"])
    df["tsb_sb_g"] = df["tsb_cum_SB"] / df["tsb_cum_G"]
    df["tsb_stop"] = ((df["tsb_cum_CS"] + SB_SUCC_K * TSB_STOP_PRIOR)
                      / (df["tsb_cum_SB"] + df["tsb_cum_CS"] + SB_SUCC_K))

    # backlog #22 (audit-wave rebuild): the OPPONENT's high-leverage arm
    # availability — pairs with xpa_pen: exposure says how many late PAs,
    # this says whether the good arms are down tonight
    df = df.merge(_pen_unavail_table(gp).rename(
        columns={"Team": "Opponent"}), on=["Opponent", "Date"], how="left")

    pairs = df[["PlayerId", "StarterId", "Season"]].dropna().rename(
        columns={"PlayerId": "BatterId", "StarterId": "PitcherId"})
    mu = matchup_features(pairs, raw["ars_p"], raw["ars_b"])
    df = df.merge(mu, left_on=["PlayerId", "StarterId", "Season"],
                  right_on=["BatterId", "PitcherId", "Season"], how="left")
    # audit wave (rank 22): the opposing starter's arsenal breadth/
    # trajectory (batter-independent; merged by StarterId)
    df = df.merge(_arsenal_dynamics(raw["ars_p"]).rename(
        columns={"PlayerId": "StarterId"}),
        on=["StarterId", "Season"], how="left")

    df = _attach_bio(df, raw["rosters"], "PlayerId", "bat")
    df = df.rename(columns={"B": "bat_hand"}).drop(columns=["T"], errors="ignore")
    ros_p = raw["rosters"][["PlayerId", "T", "DOB"]].rename(
        columns={"PlayerId": "StarterId", "T": "pit_throws"})
    df = df.merge(ros_p, on="StarterId", how="left")
    # audit wave (rank 20): the opposing starter's age (bio prior for young
    # arms where the pc_/ps_/pd_ history is thin)
    df["p_age"] = (df["Date"] - df.pop("DOB")).dt.days / 365.25
    # statsapi handedness covers ALL players (rosters only current ones)
    bats = raw["hands"].set_index("PlayerId")["Bats"].replace("", np.nan)
    throws = raw["hands"].set_index("PlayerId")["Throws"].replace("", np.nan)
    df["bat_hand"] = df["PlayerId"].map(bats).fillna(df["bat_hand"])
    df["pit_throws"] = df["StarterId"].map(throws).fillna(df["pit_throws"])
    df = _platoon(df)

    # platoon-split features vs today's starter hand (shrunk rates)
    for s in VSH_STATS:
        df[f"vsh_{s}"] = np.where(
            df["opp_hand"] == "L", df[f"_vsL_{s}"],
            np.where(df["opp_hand"] == "R", df[f"_vsR_{s}"], np.nan))
    df["vsh_hr_pa_sh"] = _shrink(df["vsh_HR"], df["vsh_PA"], "hr_pa")
    df["vsh_tb_ab_sh"] = _shrink(df["vsh_TB"], df["vsh_PA"], "tb_ab")
    df["vsh_k_pct_sh"] = _shrink(df["vsh_SO"], df["vsh_PA"], "k_pct")

    # hand-split contact quality (process-stat platoon): the batter's career
    # contact vs pitchers of TODAY's starter's hand, and the starter's
    # contact allowed to batters of THIS batter's effective side. Unknown
    # hands merge through a never-matching sentinel -> NaN features.
    if raw.get("bip") is not None:
        df["_oh"] = df["opp_hand"].fillna("?")
        df["_eh"] = df["eff_hand"].fillna("?")
        bvh = _bip_hand_table(raw["bip"], "BatterId", "PThrows").rename(
            columns={"BatterId": "PlayerId", "PThrows": "_oh"})
        bvh = bvh.rename(columns={c: f"bvh_{c}" for c in bvh.columns
                                  if c.startswith("cum_")})
        df = _asof_merge(df, bvh, by=["PlayerId", "_oh"])
        pvh = _bip_hand_table(raw["bip"], "PitcherId", "Stand").rename(
            columns={"PitcherId": "StarterId", "Stand": "_eh"})
        pvh["StarterId"] = pvh["StarterId"].astype(df["StarterId"].dtype)
        pvh = pvh.rename(columns={c: f"pvh_{c}" for c in pvh.columns
                                  if c.startswith("cum_")})
        df = _asof_merge(df, pvh, by=["StarterId", "_eh"])
        df = df.drop(columns=["_oh", "_eh"])
        for tag in ("bvh", "pvh"):
            feats = bip_feats(lambda c, t=tag: df[f"{t}_cum_{c}"], tag,
                              BVH_METRICS)
            for k, v in feats.items():
                df[k] = v
            df[f"{tag}_n"] = df[f"{tag}_cum_n"]
    else:
        for tag in ("bvh", "pvh"):
            for name in BVH_METRICS:
                df[f"{tag}_{name}"] = np.nan
            df[f"{tag}_n"] = np.nan

    # park handed-HR: as-of venue HR/PA split by batter hand (phh_*). The edge
    # off eff_hand + pull-porch geometry are derived in add_batter_derived
    # (shared with the serving path, so both compute them identically).
    phh = _park_hand_hr_table(raw["gb"], raw["games"], raw["hands"])
    df = _asof_merge(df, phh, by=["Venue"])

    for pre in ["c", "s"]:
        df = pd.concat([df, _bat_rates(df, pre), _bat_rates_shrunk(df, pre)], axis=1)
    for w in ROLL_WINDOWS:
        df[f"r{w}_hr_pa"] = df[f"r{w}_HR"] / df[f"r{w}_PA"]
        df[f"r{w}_tb_ab"] = df[f"r{w}_TB"] / df[f"r{w}_PA"]
        df[f"r{w}_k_pct"] = df[f"r{w}_SO"] / df[f"r{w}_PA"]
        # shrunk rolling rates (PA denominator)
        df[f"r{w}_hr_pa_sh"] = _shrink(df[f"r{w}_HR"], df[f"r{w}_PA"], "hr_pa")
        df[f"r{w}_tb_ab_sh"] = _shrink(df[f"r{w}_TB"], df[f"r{w}_PA"], "tb_ab")
        df[f"r{w}_k_pct_sh"] = _shrink(df[f"r{w}_SO"], df[f"r{w}_PA"], "k_pct")
        df[f"r{w}_sb_pa_sh"] = _shrink(df[f"r{w}_SB"], df[f"r{w}_PA"], "sb_pa")
        df[f"r{w}_r_pa_sh"] = _shrink(df[f"r{w}_R"], df[f"r{w}_PA"], "r_pa")
        df[f"r{w}_rbi_pa_sh"] = _shrink(df[f"r{w}_RBI"], df[f"r{w}_PA"],
                                        "rbi_pa")
    # decay-weighted current-form rates + trend deltas (shared helpers)
    for k, v in decayed_feats({s: df[f"_dk_{s}"] for s in DECAY_STATS}).items():
        df[k] = v
    add_bat_trends(df)
    # batter's own decayed PA per game (exposure: pinch-hit/platoon-removal
    # risk), shrunk toward the league PA/G; a debut player sits at the prior
    df["xpa_bat"] = ((df["_dk_PA"] + XPA_K * XPA_PRIOR)
                     / (df["_dk_G"] + XPA_K))
    # H+R+RBI joint-threshold history (hrr props only — train.py _HRR_HIST):
    # career / season / 90-day-decayed share of games with 2+/3+ H+R+RBI
    for pre, src, den in (("c", "c", df["g_career"]),
                          ("s", "s", df["g_season"]),
                          ("d", "_dk", df["_dk_G"])):
        sums = {key: df[f"{src}_{key}"] for key in THRESH_HIST}
        for k, v in hrr_hist_feats(sums, den, pre).items():
            df[k] = v
    # stolen-base success rate (career, shrunk toward the league rate)
    df["c_sb_succ"] = ((df["c_SB"] + SB_SUCC_K * SB_SUCC_PRIOR)
                       / (df["c_SB"] + df["c_CS"] + SB_SUCC_K))
    # audit wave (rank 4): reach-first rate and attempt-per-opportunity —
    # the intent decomposition the fused SB/PA rate can't express (it is
    # (sb_pa/sb_succ)/ob1, a triple ratio trees cannot carve)
    _ob1 = (df["c_H"] - df["c_2B"] - df["c_3B"] - df["c_HR"]
            + df["c_BB"] + df["c_HBP"])
    df["ob1_sh"] = (_ob1 + OB1_K * OB1_PRIOR) / (df["c_PA"] + OB1_K)
    df["sb_att1"] = ((df["c_SB"] + df["c_CS"] + SB_ATT_K * SB_ATT_PRIOR)
                     / (_ob1 + SB_ATT_K))
    # extra-base-hit rate, full OBP (incl. HBP), feared-hitter IBB rate
    for pre in ["c", "s"]:
        df[f"{pre}_xbh_ab"] = ((df[f"{pre}_2B"] + df[f"{pre}_3B"]
                                + df[f"{pre}_HR"]) / df[f"{pre}_AB"])
        df[f"{pre}_obp"] = ((df[f"{pre}_H"] + df[f"{pre}_BB"]
                             + df[f"{pre}_HBP"]) / df[f"{pre}_PA"])
    df["c_ibb_pa"] = df["c_IBB"] / df["c_PA"]
    # position wear: career share of games caught / DH'd
    df["pos_c_share"] = df["_posC_n"] / df["g_career"]
    df["pos_dh_share"] = df["_posDH_n"] / df["g_career"]
    # career splits in TODAY's venue context (home or road), shrunk
    for s in LOC_STATS:
        df[f"vloc_{s}"] = np.where(df["Home"] == 1, df[f"_loc1_{s}"],
                                   df[f"_loc0_{s}"])
    df["vloc_hr_pa_sh"] = _shrink(df["vloc_HR"], df["vloc_PA"], "hr_pa")
    df["vloc_h_pa_sh"] = _shrink(df["vloc_H"], df["vloc_PA"], "h_ab")
    df["vloc_tb_ab_sh"] = _shrink(df["vloc_TB"], df["vloc_PA"], "tb_ab")
    df["vloc_k_pct_sh"] = _shrink(df["vloc_SO"], df["vloc_PA"], "k_pct")
    for pre in ["pc", "ps"]:
        df = pd.concat([df, _pit_rates(df, pre)], axis=1)
    df["p5_hr_bf"] = df["p5_HR"] / df["p5_BF"]
    df["p5_k_bf"] = df["p5_SO"] / df["p5_BF"]
    df["p5_h_bf"] = df["p5_H"] / df["p5_BF"]
    df["p5_bb_bf"] = df["p5_BB"] / df["p5_BF"]   # p_hit_luck denominator
    df["p_ip_per_start"] = df["ps_Outs"] / 3 / df["p_starts_season"]
    add_pit_trends(df)

    # the opposing DEFENSE's unearned-run rate (errors extend innings and
    # put extra runners on): the error-proneness OAA's range measure misses
    od = _team_defense_table(gp).rename(columns={"Team": "Opponent"})
    od = od.rename(columns={c: f"od_{c}" for c in od.columns
                            if c.startswith("cum")})
    df = _asof_merge(df, od, by=["Opponent", "Season"])
    df["opp_def_uer"] = ((df["od_cum_R"] - df["od_cum_ER"]) * 27
                         / df["od_cum_Outs"])

    # teammate context: career on-base of the two hitters AHEAD (they load
    # the bases for RBI) and slugging of the two BEHIND (they drive you in),
    # wrapping around the order. Uses each teammate's as-of career rates.
    df["_obpp"] = (df["c_H"] + df["c_BB"] + df["c_HBP"]) / df["c_PA"]
    df["_slgp"] = df["c_TB"] / df["c_AB"]
    # decayed variants (90-day half-life): the neighbor's CURRENT form —
    # the career rates miss a mid-season acquisition or a surging/slumping
    # neighbor entirely
    df["_obpp_d"] = (df["_dk_H"] + df["_dk_BB"] + df["_dk_HBP"]) / df["_dk_PA"]
    df["_slgp_d"] = df["_dk_TB"] / df["_dk_AB"]
    lt = df[["GamePk", "Team", "slot", "_obpp", "_slgp", "_obpp_d",
             "_slgp_d"]].drop_duplicates(["GamePk", "Team", "slot"])
    for off in (-2, -1, 1, 2):
        nb = lt.rename(columns={"slot": "_nslot", "_obpp": f"_obpp{off}",
                                "_slgp": f"_slgp{off}",
                                "_obpp_d": f"_obpp_d{off}",
                                "_slgp_d": f"_slgp_d{off}"})
        df["_nslot"] = ((df["slot"] + off - 1) % 9) + 1
        df = df.merge(nb, on=["GamePk", "Team", "_nslot"], how="left")
    df = df.drop(columns=["_nslot"])
    df["ctx_ahead_obp"] = df[["_obpp-2", "_obpp-1"]].mean(axis=1)
    df["ctx_behind_slg"] = df[["_slgp1", "_slgp2"]].mean(axis=1)
    df["ctx_ahead_obp_d"] = df[["_obpp_d-2", "_obpp_d-1"]].mean(axis=1)
    df["ctx_behind_slg_d"] = df[["_slgp_d1", "_slgp_d2"]].mean(axis=1)
    # audit wave (rank 12): the missing directions of the neighbor matrix —
    # ahead-SLG (WHICH base the supply runners occupy: a runner scores from
    # 2nd ~2x as often) and behind-OBP (chain continuation), + decayed
    df["ctx_ahead_slg"] = df[["_slgp-2", "_slgp-1"]].mean(axis=1)
    df["ctx_ahead_slg_d"] = df[["_slgp_d-2", "_slgp_d-1"]].mean(axis=1)
    df["ctx_behind_obp"] = df[["_obpp1", "_obpp2"]].mean(axis=1)
    df["ctx_behind_obp_d"] = df[["_obpp_d1", "_obpp_d2"]].mean(axis=1)
    # runners-ahead advancement (2026-07-12): mean prior-season extra-base
    # advancement rate of the two hitters ahead — whether the runners this
    # batter drives in can actually take the extra base. RBI-specific
    # sibling of ctx_ahead_obp (the batter's OWN bat_brr_xb speaks to his
    # Run prop; his neighbors' speaks to his RBI chances).
    brr_lt = df[["GamePk", "Team", "slot", "bat_brr_xb"]].drop_duplicates(
        ["GamePk", "Team", "slot"])
    for off in (-2, -1):
        nb = brr_lt.rename(columns={"slot": "_nslot",
                                    "bat_brr_xb": f"_brrp{off}"})
        df["_nslot"] = ((df["slot"] + off - 1) % 9) + 1
        df = df.merge(nb, on=["GamePk", "Team", "_nslot"], how="left")
    df = df.drop(columns=["_nslot"])
    df["ctx_ahead_brr"] = df[["_brrp-2", "_brrp-1"]].mean(axis=1)
    # audit wave (rank 12): the rally-kill axis — GIDP-proneness of the two
    # hitters BEHIND (they bat with this batter on first; a double-play
    # machine behind him erases his run chances). brr_lt shift-merge idiom.
    gidp_lt = df[["GamePk", "Team", "slot",
                  "c_gidp_pa_sh"]].drop_duplicates(["GamePk", "Team", "slot"])
    for off in (1, 2):
        nb = gidp_lt.rename(columns={"slot": "_nslot",
                                     "c_gidp_pa_sh": f"_gidpp{off}"})
        df["_nslot"] = ((df["slot"] + off - 1) % 9) + 1
        df = df.merge(nb, on=["GamePk", "Team", "_nslot"], how="left")
    df = df.drop(columns=["_nslot"])
    df["ctx_behind_gidp"] = df[["_gidpp1", "_gidpp2"]].mean(axis=1)
    # rbi opportunity: full-order proximity-decayed OBP of the hitters ahead
    # (RBI_OPP_AHEAD) — expected runners on base the 2-slot ctx can't see.
    obp_lt = df[["GamePk", "Team", "slot", "_obpp"]].drop_duplicates(
        ["GamePk", "Team", "slot"])
    num = np.zeros(len(df))
    den = np.zeros(len(df))
    for off, w in RBI_OPP_AHEAD:
        nb = obp_lt.rename(columns={"slot": "_nslot", "_obpp": "_ahd_obp"})
        df["_nslot"] = ((df["slot"] + off - 1) % 9) + 1
        df = df.merge(nb, on=["GamePk", "Team", "_nslot"], how="left")
        v = df["_ahd_obp"].to_numpy()
        m = ~np.isnan(v)
        num[m] += w * v[m]
        den[m] += w
        df = df.drop(columns=["_nslot", "_ahd_obp"])
    df["rbi_opp_obp"] = np.where(den > 0, num / den, np.nan)

    # home-plate umpire zone tendency (as-of), merged by game
    df = _merge_ump(df, raw)

    df["month"] = df["Date"].dt.month
    df["y_hr"] = (df["HR"] >= 1).astype(int)
    df["y_hit"] = (df["H"] >= 1).astype(int)
    df["y_run"] = (df["R"] >= 1).astype(int)
    df["y_rbi"] = (df["RBI"] >= 1).astype(int)
    df["y_hits2"] = (df["H"] >= 2).astype(int)
    df["y_tb2"] = (df["TB"] >= 2).astype(int)
    df["y_bb"] = (df["BB"] >= 1).astype(int)
    df["y_sb"] = (df["SB"] >= 1).astype(int)
    df["hr_count"] = df["HR"]
    # count-market targets: singles/doubles (real O/U markets), batter
    # strikeouts (0.5/1.5 lines) and H+R+RBI (1.5/2.5 lines). H+R+RBI is
    # modeled DIRECTLY (never derived from the H/R/RBI marginals: the three
    # are strongly positively correlated — a solo HR is 3 by itself — so
    # independent marginals understate the tails).
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    hrr = df["H"] + df["R"] + df["RBI"]
    df["y_1b"] = (singles >= 1).astype(int)
    df["y_2b"] = (df["2B"] >= 1).astype(int)
    df["y_bk1"] = (df["SO"] >= 1).astype(int)
    df["y_bk2"] = (df["SO"] >= 2).astype(int)
    df["y_hrr2"] = (hrr >= 2).astype(int)
    df["y_hrr3"] = (hrr >= 3).astype(int)
    # 2026-07-14 finish batch — new heads: the four deep binaries (H1,
    # count-vs-binary verdict: these lines must be BINARY-priced), the
    # triple (H3, completes the hit-type family; 1.21% base rate — Platt
    # is load-bearing), and the 2+ RBI / 2+ runs thresholds (H4)
    df["y_bk3"] = (df["SO"] >= 3).astype(int)
    df["y_tb3"] = (df["TB"] >= 3).astype(int)
    df["y_tb4"] = (df["TB"] >= 4).astype(int)
    df["y_hrr4"] = (hrr >= 4).astype(int)
    df["y_3b"] = (df["3B"] >= 1).astype(int)
    df["y_rbi2"] = (df["RBI"] >= 2).astype(int)
    df["y_run2"] = (df["R"] >= 2).astype(int)
    df["bk_count"] = df["SO"]
    df["hrr_count"] = hrr
    df["tb_count"] = df["TB"]   # total bases -> expected-TB head (xTB)
    # H6 count-head targets: the rest of the expected-stat-line (means
    # only; their banked line calibrators never ship — binaries own the
    # batter lines per the 07-13 shoot-out)
    df["h_count"] = df["H"]
    df["run_count"] = df["R"]
    df["rbi_count"] = df["RBI"]
    df["bb_count"] = df["BB"]

    # ---- row-wise derived features + interactions: one shared function that the
    # serving path also calls, so the two compute them identically (parity). The
    # as-of phh_* / bvp_cum_* joins above feed it; drop those intermediates. ----
    df = add_batter_derived(df)
    df = df.drop(columns=[c for c in df.columns
                          if c.startswith(("phh_", "bvp_cum_"))])
    return df


def batter_feature_cols():
    cols = ["slot", "Home", "Season", "month", "days_rest",
            "g_career", "g_season",
            "c_PA", "c_hr_pa", "c_tb_ab", "c_h_ab", "c_k_pct", "c_bb_pct", "c_iso",
            "s_PA", "s_hr_pa", "s_tb_ab", "s_h_ab", "s_k_pct", "s_bb_pct", "s_iso"]
    for w in ROLL_WINDOWS:
        cols += [f"r{w}_PA", f"r{w}_hr_pa", f"r{w}_tb_ab", f"r{w}_k_pct"]
    cols += ["p_starts_career", "p_starts_season", "p_days_rest",
             "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_h_bf", "pc_era",
             "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_h_bf", "ps_era",
             "p5_hr_bf", "p5_k_bf", "p5_h_bf", "p_ip_per_start",
             "toff_hr_pa", "toff_r_pg", "toff_k_pct", "toff_bb_pct",
             "toff_loc_hr_pa", "toff_loc_r_pg",
             "pen_hr_bf", "pen_k_bf", "pen_h_bf", "pen_era",
             "pen_hl_era", "pen_hl_k_bf", "pen_np_l3",
             "park_hr_pg", "park_r_pg", "park_h_pg", "park_2b_pg",
             "park_3b_pg", "park_tb_pg",
             "LF", "CF", "RF", "Elevation_ft",
             "Temp", "WindSpeed",
             "hrq_n", "hrq_ev_avg", "hrq_dist_avg", "hrq_dist_max",
             "hrq_angle_avg", "hrpt_score", "phrq_n", "phrq_ev_avg",
             "bat_goao", "pit_goao",
             *ARS_P_METRICS.values(), *ARS_B_METRICS.values(), "m_coverage",
             *ARS_DYN_COLS,   # audit wave rank 22 (opposing-starter arsenal)
             "bat_height", "bat_weight", "bat_age", "same_hand",
             "vsh_PA", "vsh_hr_pa_sh", "vsh_tb_ab_sh", "vsh_k_pct_sh",
             "vloc_PA", "vloc_hr_pa_sh", "vloc_h_pa_sh", "vloc_tb_ab_sh",
             "vloc_k_pct_sh",
             "c_sb_succ", "psb_sb27", "psb_stop", "tsb_sb_g", "tsb_stop",
             "c_xbh_ab", "s_xbh_ab", "c_obp", "s_obp", "c_ibb_pa",
             "pos_c_share", "pos_dh_share",
             "ctx_ahead_obp", "ctx_behind_slg",
             # decayed teammate ctx: benched 2026-07-08 (0/0/76 within
             # noise), RE-ACCEPTED under the keep-leaning bar 2026-07-09
             # (queue Tier A2) — routed with _CTX in train.py
             "ctx_ahead_obp_d", "ctx_behind_slg_d",
             # rbi_opp_obp + exposure (xpa_*) UNBENCHED 2026-07-10 for the
             # NUCLEAR PROBE (user: unbench everything, no routing) — both
             # were benched out of the superset 07-07/07-09; serving emits
             # all three (predict.py builds them per row).
             "rbi_opp_obp", "xpa_bat", "xpa_slot",
             "d_PA", "d_hr_pa_sh", "d_tb_ab_sh", "d_k_pct_sh",
             "d_bb_pct_sh", "d_sb_pa_sh",
             # decayed r/rbi form: re-accepted 2026-07-09 with the rolling
             # variants (RUNRBI_FORM_COLS note; run routed around)
             "d_r_pa_sh", "d_rbi_pa_sh",
             # own H+R+RBI threshold-share history: re-accepted 2026-07-09
             # (HRR_SHRINK note; hrr3/xhrr only, hrr2 routed around)
             "c_hrr2_g_sh", "s_hrr2_g_sh", "d_hrr2_g_sh",
             "c_hrr3_g_sh", "s_hrr3_g_sh", "d_hrr3_g_sh",
             "tr15_hr", "tr15_tb", "tr15_k", "dev_hr", "dev_tb", "dev_k",
             "p5_k_trend", "p5_hr_trend", "p_era_trend",
             # league 30-day environment + opposing-starter fatigue:
             # benched iteration 4, RE-ACCEPTED 2026-07-09 (queue Tier C)
             # — routed via train._ENV/_PNP (off the xhrr/xtb mean-heads)
             "lg_hr_pa", "lg_r_pa", "lg_k_pa", "lg_bb_pa", "lg_sb_pa",
             "p_np_last", "p_np_l3",
             "bip_n", "bip_ev", "bip_la", "bip_hh", "bip_brl", "bip_xba",
             "bip_xwoba", "bip_gb", "bip_pull", "bip_pullair",
             "bipd_n", "bipd_ev", "bipd_brl", "bipd_xwoba", "bipd_gb",
             "bipd_pullair",
             "pbip_n", "pbip_ev", "pbip_la", "pbip_hh", "pbip_brl",
             "pbip_xba", "pbip_xwoba", "pbip_gb",
             "pbipd_n", "pbipd_ev", "pbipd_brl", "pbipd_xwoba", "pbipd_gb",
             # hand-split contact quality: benched 2026-07-07 (tb2 ECE),
             # RE-ACCEPTED 2026-07-09 (queue Tier B4) with a tb2/xtb
             # route-around — see train._VHB_PWR/_VHB_CON
             "bvh_xwoba", "bvh_brl", "bvh_n",
             "pvh_xwoba", "pvh_brl", "pvh_n",
             "bd_wsw_c", "bd_wsw_d", "bd_chase_c", "bd_chase_d",
             "p_swstr_d", "bat_sprint", "bat_hp1b", "opp_oaa",
             # HP-umpire zone tendency — routed to the K/BB props only
             # (train.py _UMP); other batter props exclude it
             "ump_k_pct", "ump_bb_pct",
             # Statcast bat tracking (power) — routed to hr/tb2/xtb only
             # (train.py _BAT); INERT until ~2027 (2023+ coverage vs the
             # <=2023 training window), banked+wired to self-activate
             *BAT_TRACK_COLS,
             # HR-physics + matchup interactions (2026-07-10 dev batch): pull
             # geometry UNBENCHED (pull_fence/porch_margin), wind projected onto
             # the pull field + general carry, hot+high air carry, batted-ball x
             # opponent defense. Superset-dev exposure; selection decides at ship.
             "pull_fence", "porch_margin", "bat_wind_pull", "bat_wind_porch",
             "wind_carry", "carry_air", "bip_gb_def", "bip_air_def",
             # fatigue (games in last 7/14 days), luck-regression (recent actual
             # BA-on-contact vs expected xBA), and ump-zone x matchup interactions
             "g_l7d", "g_l14d", "hit_luck",
             "ump_k_x_pk", "ump_k_x_bk", "ump_bb_x_pbb",
             # batter-vs-pitcher direct history (2026-07-10): contact-quality +
             # HR residual off the batter's own baseline vs THIS starter, shrunk
             # by pairwise sample size (bvp_n). Contact-only (BIP has no K/BB);
             # selection decides per head, default off the x-heads per policy.
             "bvp_n", "bvp_xwoba_resid", "bvp_hr_resid",
             # realized handed park-HR edge (2026-07-10): venue HR/PA for the
             # batter's effective hand minus the other hand, as-of — the handed
             # asymmetry park_hr_pg + pull_fence both miss. Targets hr/tb2/hrr.
             "park_hand_hr_edge",
             # 2026-07-12 data batch — new sources, selection decides per head:
             # air density (humidity+pressure scrape; carry physics beyond
             # Temp x Elevation), zone-contact + elite-velo whiff (pitch-scrape
             # schema; the stable hit-tool skills), starter velo + the velo
             # matchup, starter TTO decay, the ACTUAL lineup defense faced
             # (player-level OAA, IF/OF x batted-ball profile), and prior-
             # season baserunning run value (advancement skill for run/sb)
             "hum_eff", "air_dens", "air_porch", "air_fly",
             "bd_zwsw_c", "bd_zwsw_d", "bd_fb95wh_c", "bd_fb95wh_d",
             "p_fbv_d", "bat_velo_matchup", "p_tto_decay",
             "opp_def_p_oaa", "opp_def_p_if", "opp_def_p_of",
             "bip_gb_def_if", "bip_air_def_of",
             "bat_brr", "bat_brr_xb", "ctx_ahead_brr", "ctx_run_conv",
             # BBType shares the gb/pullair pair missed (2026-07-12): line
             # drives + popups, batter and starter-allowed sides
             "bip_ld", "bipd_ld", "bip_pu", "pbip_ld", "pbipd_ld", "pbip_pu",
             # third wave (2026-07-12): uncensored fly-ball power (mean
             # sea-level-adjusted fly distance, both sides), starter
             # BABIP-luck regression, legs x grounders, error-prone defense
             # faced, and the zone-pounder x zone-contact matchup
             "bip_flyd", "bipd_flyd", "pbip_flyd", "pbipd_flyd",
             "p_hit_luck", "bat_leg_hits", "opp_def_uer",
             "p_zone_d", "zone_whiff_matchup",
             # style-collision products (log5 form): both sides' batted-ball
             # tendencies multiply; + the chase-style matchup and its
             # starter-side main
             "mix_air", "mix_brl", "mix_xwcon",
             "p_chase_d", "chase_matchup",
             # closing sweep (2026-07-12): structures trees can't build —
             # exposure products (per-PA skill x expected PA = each head's
             # Poisson mean), outcome-level log5 collisions, the run/RBI
             # conversion chains, park x power, starter K-BB arithmetic
             "xpa_x_hr", "xpa_x_hit", "xpa_x_tb", "xpa_x_k",
             "xpa_x_rbi", "xpa_x_r",
             "mix_k", "mix_bb", "mix_hr", "mix_hit", "mix_gb", "mix_ld",
             "rbi_conv", "run_opp", "park_x_hr", "p_kbb",
             # MiLB translated priors (2026-07-13 Phase-3 rider): the
             # batter's minors line translated to MLB-equivalent rates +
             # evidence mass — new info for thin-history rows; selection
             # votes per head as usual
             *MILB_BAT_COLS,
             # v3 scrape schema (2026-07-12): whiff splits by pitch class +
             # first-pitch aggression (batter), usage mix / shadow-band
             # command / first-pitch strike (opposing starter), and the
             # arsenal/class/first-pitch collisions
             "bd_brkwh_c", "bd_brkwh_d", "bd_offwh_c", "bd_offwh_d",
             "bd_fbwh_c", "bd_fbwh_d", "bd_fpsw_c", "bd_fpsw_d",
             "p_brk_d", "p_off_d", "p_edge_d", "p_fps_d",
             "arsenal_whiff", "brk_matchup", "off_matchup", "fp_matchup",
             # v4 graded velocity bands: whiff splits <92 / 92-95 (95+
             # above), the starter's banded usage, and the shaped collision
             "bd_fblowh_c", "bd_fblowh_d", "bd_fbmidwh_c", "bd_fbmidwh_d",
             "p_fblou_d", "p_fbmidu_d", "p_fb95u_d", "velo_band_whiff",
             # v5 count leverage + dispersion: two-strike survival/put-away
             # + collision, 3-2 walk conversion both sides, starter velo
             # spread and release-point scatter
             "bd_tswh_c", "bd_tswh_d", "bd_f32b_c", "bd_f32b_d",
             "p_tswh_d", "p_f32b_d", "ts_matchup",
             "p_fbv_sd", "p_rel_sd",
             # 2026-07-14 finish batch (backlog #15/#18/#23/#24/#27/#31):
             # bullpen exposure + pen-minus-starter collisions (the
             # second-pitcher problem), schedule/travel fatigue +
             # doubleheader flags, the opposing starter's venue split,
             # FF ride + its whiff/air collisions, and form-weighted
             # exposure (decayed siblings of the xpa_x_* products)
             "xpa_pen", "xpen_k", "xpen_hr", "xpen_h",
             "day_after_night", "travel_km", "tz_delta", "is_dh", "dh_game2",
             "pvloc_era", "pvloc_k_bf", "pvloc_hr_bf",
             "p_ivb_d", "ride_matchup", "ride_air",
             "xpa_x_hr_d", "xpa_x_tb_d", "xpa_x_k_d",
             # NOTE: pull_fence/porch_margin and batter-side fatigue were
             # tried (iteration 3) and hurt the batter props on the holdout;
             # the league-environment lg_* columns (iteration 4) were flat
             # to slightly negative for every prop; hand-split contact
             # quality (bvh_*/pvh_*, benched 2026-07-07 on tb2 ECE) was
             # re-accepted 2026-07-09 with a tb2/xtb route-around (cols
             # above); exposure features (xpa_bat/xpa_slot,
             # 2026-07-07 eve) likewise — rbi/single/bk ECE past band, no
             # AUC/edge movement (see the XPA_PRIOR note); own H+R+RBI
             # threshold-share history (c/s/d_hrr{2,3}_g_sh, 2026-07-08)
             # likewise — hrr2_ece 1.5x past band, everything else flat
             # (see the HRR_SHRINK note). All stay in the
             # frames but out of the batter models. Fatigue (p_np_*) and
             # lg_* were re-accepted into the batter models 2026-07-09
             # (cols above); they also live in starts cols for the K model.
             # This is the SUPERSET; what each head actually trains on is
             # decided solely by stability selection (feature_keep.json) —
             # the hand-curated PROP_EXCLUDE routing was deleted 2026-07-15.
             # ---- audit wave (2026-07-14, AUDIT_BUILD_SPECS_0714.md) ----
             # rank 1: bb/sb exposure completion + pen walk exposure
             "xpa_x_bb", "xpa_x_sb", "xpa_x_bb_d", "xpa_x_sb_d",
             "xpen_bb", "pen_bb_bf",
             # rank 2: deeper threshold histories for the 07-14 heads
             "c_hrr4_g_sh", "s_hrr4_g_sh", "d_hrr4_g_sh",
             "c_rbi2_g_sh", "s_rbi2_g_sh", "d_rbi2_g_sh",
             "c_run2_g_sh", "s_run2_g_sh", "d_run2_g_sh",
             # rank 3: park x hit-type products (rates ride SHRINK_COLS)
             "park_x_2b", "park_x_3b",
             # rank 4: sb opportunity chain
             "ob1_sh", "sb_att1", "sb_chain", "sb_chain_env", "ob1_x_st",
             # rank 7: passivity axis + collisions
             "bd_zsw_c", "bd_zsw_d", "bd_tssw_c", "bd_tssw_d",
             "called_k_matchup", "ump_k_x_take",
             # rank 5: pitcher spray exposure + pull-air collision
             "pbip_pull", "pbip_pullair", "pbipd_pullair", "mix_pullair",
             # rank 11: starter outing shape carried to the batter grain
             "p_outs_sd", "p_short_share",
             # rank 12: neighbor-matrix completion
             "ctx_ahead_slg", "ctx_ahead_slg_d",
             "ctx_behind_obp", "ctx_behind_obp_d", "ctx_behind_gidp",
             # rank 13: oppo spray + EV tail + oppo park geometry
             "bip_oppo", "bip_oppoair", "bipd_oppoair",
             "pbip_oppo", "pbip_oppoair", "pbipd_oppoair",
             "bip_ev105", "bipd_ev105", "pbip_ev105", "pbipd_ev105",
             "oppo_fence", "oppo_air_porch",
             # rank 14: realized RBI opportunity/conversion (LOB consumed;
             # c_/s_rbi_conv_sh ride SHRINK_COLS)
             "rbi_conv_gap", "d_rob_pa",
             # rank 15: own team's BaseRuns sequencing-luck residual
             "toff_bsr_luck",
             # rank 17: the opposing starter's MiLB-translated priors
             *MILB_PIT_COLS,
             # rank 21: chaos channel (career + season, both starter rates)
             "pc_chaos_bf", "pc_hbp_bf", "ps_chaos_bf", "ps_hbp_bf",
             # ranks 23/31: ump x walk cell, protection product
             "ump_bb_x_bb", "pitch_around",
             # rank 27: v7 cause axes of the opposing starter + the 2K x
             # breaking putaway cell
             "p_effprem_d", "p_brkmov_d", "p_stretch_d", "p_relsep_d",
             "bd_tsbrkwh_c", "bd_tsbrkwh_d", "ts_brk_matchup",
             # rank 29/30: ump run environment + precip (exposure-only)
             "ump_r_g", "Precip",
             # backlog #22 (rebuild): opponent high-leverage arm
             # availability — pairs with xpa_pen (how many late PAs vs
             # whether the good arms are down)
             "pen_hl_unavail",
             # rank 20: opposing starter age
             "p_age",
             # v8 damage-on-contact wave (2026-07-15): xwOBA-per-BBE
             # splits by velo band / pitch class — batter side (career +
             # decay), the opposing starter's damage-ALLOWED splits, the
             # damage collisions (contact half of the velo-band axis; the
             # whiff half shipped v4), and the 2K x elite-velo cell
             "bd_fbloxw_c", "bd_fbloxw_d", "bd_fbmidxw_c", "bd_fbmidxw_d",
             "bd_fb95xw_c", "bd_fb95xw_d", "bd_brkxw_c", "bd_brkxw_d",
             "bd_offxw_c", "bd_offxw_d", "bd_fbkxw_c", "bd_fbkxw_d",
             "p_fbloxw_d", "p_fbmidxw_d", "p_fb95xw_d",
             "p_brkxw_d", "p_offxw_d", "p_fbkxw_d",
             "bat_velo_damage", "velo_band_damage", "arsenal_damage",
             "mix_fb95xw",
             "bd_tsfb95wh_c", "bd_tsfb95wh_d", "ts_fb95_matchup",
             # battery + IL wave (2026-07-15): the opponent's playing-time-
             # weighted catcher quality (framing / CS value / pop time) +
             # the framing-take and SB-chain collisions; IL return context
             # for the batter and the opposing starter (the layoff CAUSE
             # the days-rest gaps can't see)
             "opp_cat_frame", "opp_cat_csaa", "opp_cat_pop",
             "frame_x_take", "sb_cat_env",
             "il_ret_days", "il_last_len", "il_ret21", "il_szn_days",
             "il_rehab",
             "p_il_ret_days", "p_il_last_len", "p_il_ret21",
             "p_il_szn_days", "p_il_rehab",
             *CAT_COLS]
    # r{w}_{r,rbi}_pa_sh re-accepted 2026-07-09 (RUNRBI_FORM_COLS note)
    cols += SHRINK_COLS
    return cols


def add_starter_derived(df):
    """Row-wise derived starter features, computed IDENTICALLY by
    build_starts_frame and the serving rows (parity): air-density weather,
    times-through-order decay, the K-BB composite, and the lineup-collision
    products — the lineup's whiff/K/chase tendencies MULTIPLIED by his
    stuff (log5 at lineup level; trees can't build the product from the
    mains). lu_* may be absent (frame built without the batter frame);
    those products go NaN."""
    add_weather_derived(df)
    df["p_tto_decay"] = tto_decay_from_sums(df)
    df["pc_kbb"] = df["pc_k_bf"] - df["pc_bb_bf"]
    for c in ("lu_wsw", "lu_k_sh", "lu_chase",
              "lu_brkwh", "lu_offwh", "lu_fbwh",
              # v8 banded whiff / damage lineup views
              "lu_fblowh", "lu_fbmidwh", "lu_fb95wh",
              "lu_fbloxw", "lu_fbmidxw", "lu_fb95xw",
              "lu_brkxw", "lu_offxw", "lu_fbkxw"):
        if c not in df.columns:
            df[c] = np.nan
    df["lu_mix_whiff"] = df["lu_wsw"] * df["pd_wsw_d"]
    df["lu_mix_k"] = df["lu_k_sh"] * df["pc_k_bf"]
    df["lu_mix_chase"] = df["lu_chase"] * df["pd_chase_d"]
    # usage-weighted lineup whiff vs HIS pitch classes (v3 schema): the
    # lineup's whiff splits weighted by his actual mix — the arsenal
    # collision at lineup level (fastball bucket = remainder share)
    _fb_share = 1.0 - df["pd_brk_d"] - df["pd_off_d"]
    df["lu_ars_whiff"] = (_fb_share * df["lu_fbwh"]
                          + df["pd_brk_d"] * df["lu_brkwh"]
                          + df["pd_off_d"] * df["lu_offwh"])
    # v8 (2026-07-15): the velocity-graded lineup collision (lu_ars_whiff's
    # velo-band sibling — the lineup's whiff-by-band weighted by HIS banded
    # usage) and the damage twins — what the lineup does when it CONNECTS
    # with each band/class of his stuff (the pha/per side of the collision)
    df["lu_velo_k"] = (df["pd_fblou_d"] * df["lu_fblowh"]
                       + df["pd_fbmidu_d"] * df["lu_fbmidwh"]
                       + df["pd_fb95u_d"] * df["lu_fb95wh"])
    df["lu_velo_dmg"] = (df["pd_fblou_d"] * df["lu_fbloxw"]
                         + df["pd_fbmidu_d"] * df["lu_fbmidxw"]
                         + df["pd_fb95u_d"] * df["lu_fb95xw"])
    df["lu_ars_dmg"] = (_fb_share * df["lu_fbkxw"]
                        + df["pd_brk_d"] * df["lu_brkxw"]
                        + df["pd_off_d"] * df["lu_offxw"])
    # layoff / ramp regime flags (2026-07-14 #17): thin-support interactions
    # of rest x pitch count handed over explicitly; NaN inputs propagate
    _rest, _npl = df["p_days_rest"], df["p_np_last"]
    df["p_gap15"] = (_rest >= GAP_DAYS).astype(float).where(_rest.notna())
    df["p_ramp60"] = (_npl < RAMP_NP).astype(float).where(_npl.notna())
    df["p_short_heavy"] = ((_rest <= SHORT_REST_DAYS)
                           & (_npl >= HEAVY_NP)).astype(float).where(
        _rest.notna() & _npl.notna())
    # earned-run conversion chain (2026-07-14 #28): baserunners allowed x
    # damage-on-contact allowed — the rbi_conv/run_opp template pointed at
    # the starter frame (career + recent sibling)
    for c in ("pbip_xwoba", "pbipd_xwoba", "p5_bb_bf"):
        if c not in df.columns:
            df[c] = np.nan
    df["p_conv_c"] = (df["pc_h_bf"] + df["pc_bb_bf"]) * df["pbip_xwoba"]
    df["p_conv_r"] = (df["p5_h_bf"] + df["p5_bb_bf"]) * df["pbipd_xwoba"]
    # ump x the actual lineup's K-proneness (2026-07-14 #29): completes the
    # proven ump_k_x_pk pattern with the lineup side
    df["ump_k_x_lu"] = df["ump_k_pct"] * df["lu_k_sh"]
    # v6 (#25): ahead-vs-behind spin-usage shift (does his mix collapse
    # when behind) — both sides shrink to their OWN league prior, so
    # no-history lands on the league's usage shift, not zero. The velo-fade
    # slope (#26) is computed from the decayed fade sums by each path
    # separately (pd_fbv_sd idiom).
    df["pd_absh_d"] = df["pd_ahsp_d"] - df["pd_bhsp_d"]
    # ---- audit wave (2026-07-14, AUDIT_BUILD_SPECS_0714.md) ----
    for c in ("lu_obp", "lu_slg", "lu_xwcon", "lu_brl", "lu_fpsw",
              "lu_gidp", "lu_sb", "lu_sprint", "vs_obp", "vs_slg",
              "pbip_gb", "pbipd_gb", "pbipd_xba", "air_dens",
              "lg_sb_pa", "lg_sb27_prior", "psb_sb27",
              "p5_h_bf", "p5_k_bf", "p5_bb_bf", "pc_NP", "pc_Outs"):
        if c not in df.columns:
            df[c] = np.nan
    # rank 16: signed wind carry -> HR/XBH allowed -> ER (the physics the
    # pull-only categorical can't be carved into; accepted twice elsewhere)
    add_wind_carry(df, pull=False)
    # rank 18: posted-lineup quality gap (B-lineup detector) + first-pitch
    # economy collision (aggressive lineup vs a strike-thrower)
    df["lu_obp_gap"] = df["lu_obp"] - df["vs_obp"]
    df["lu_slg_gap"] = df["lu_slg"] - df["vs_slg"]
    df["fp_lu_x"] = ((df["lu_fpsw"] - PD_SHRINK["fpsw"][0])
                     * (df["pd_fps_d"] - PD_SHRINK["fps"][0]))
    # rank 5c: GB-allowed x thin air (a grounder pitcher is insulated where
    # carry helps) + GB x the lineup's double-play proneness
    df["p_gb_air"] = ((AIR_RHO0 - df["air_dens"])
                      * (df["pbipd_gb"] - BIP_SHRINK["gb"][0]))
    df["gb_dp_x"] = ((df["pbipd_gb"] - BIP_SHRINK["gb"][0])
                     * (df["lu_gidp"] - SHRINK["gidp_pa"][0]))
    # rank 10: starter BABIP-luck residual (verbatim batter twin) — the
    # starts frame had no realized hits-allowed rate at all
    _pcon = 1.0 - df["p5_k_bf"] - df["p5_bb_bf"]
    df["p_hit_luck"] = (np.where(_pcon > 0, df["p5_h_bf"] / _pcon, np.nan)
                        - df["pbipd_xba"])
    # rank 24: steal-vulnerable pitcher x running lineup -> same contact
    # line, more runs; both centered on the as-of / prior-season league
    df["run_game_x"] = ((df["lu_sb"] - df["lg_sb_pa"])
                        * (df["psb_sb27"] - df["lg_sb27_prior"]))
    # battery collisions (2026-07-15): framing cashes at the shadow zone —
    # his battery's framing x his edge share (centered); and the running
    # game vs his battery: the lineup's SB intent x his catchers' stop
    # value (run_game_x's catcher half)
    for c in ("own_cat_frame", "own_cat_csaa", "own_cat_pop"):
        if c not in df.columns:
            df[c] = np.nan
    df["frame_x_edge"] = (df["own_cat_frame"]
                          * (df["pd_edge_d"] - PD_SHRINK["edge"][0]))
    df["run_cat_x"] = df["lu_sb"] * df["own_cat_csaa"]
    # rank 25: pitch-economy budget x efficiency (trees can't divide
    # pc_NP by pc_Outs); p_exp_outs = recent budget / this ratio
    df["p_np_per_out"] = ((df["pc_NP"] + NP_PER_OUT_K * NP_PER_OUT_PRIOR)
                          / (df["pc_Outs"] + NP_PER_OUT_K))
    df["p_exp_outs"] = df["p_np_l3"] / df["p_np_per_out"]
    return df


def build_starts_frame(raw, batter_frame=None):
    """Training frame for starter strikeouts: one row per start."""
    gp, gb = raw["gp"], raw["gb"]
    st = _starter_asof(gp)
    team_tab = _team_offense_table(gb)
    pen_tab = _bullpen_table(gp)
    park_tab = _park_table(gb, raw["games"])
    # opposing team offense: rename for join
    st = _attach_context(st, raw, team_tab, pen_tab, park_tab)
    # NOTE: for a pitcher, "toff_*" above is HIS team's offense; what matters
    # is the OPPONENT's. Swap: recompute against Opponent.
    opp = team_tab.rename(columns={"Team": "Opponent"})
    opp = opp.rename(columns={c: f"vs_{c}" for c in opp.columns if c.startswith("cum")})
    st = _asof_merge(st, opp, by=["Opponent", "Season"])
    st["vs_k_pct"] = st["vs_cum_SO"] / st["vs_cum_PA"]
    st["vs_bb_pct"] = st["vs_cum_BB"] / st["vs_cum_PA"]
    st["vs_hr_pa"] = st["vs_cum_HR"] / st["vs_cum_PA"]
    st["vs_r_pg"] = st["vs_cum_R"] / st["vs_cum_n"]
    # audit wave (rank 18): the OPPONENT team's season OBP/SLG norm — the
    # baseline the posted-lineup quality gap is measured against (B-lineup
    # detector, ported from the game frame's #19)
    st["vs_obp"] = ((st["vs_cum_H"] + st["vs_cum_BB"] + st["vs_cum_HBP"])
                    / st["vs_cum_PA"])
    st["vs_slg"] = st["vs_cum_TB"] / st["vs_cum_AB"]

    for pre in ["pc", "ps"]:
        st = pd.concat([st, _pit_rates(st, pre)], axis=1)
    st["p5_hr_bf"] = st["p5_HR"] / st["p5_BF"]
    st["p5_k_bf"] = st["p5_SO"] / st["p5_BF"]
    st["p5_h_bf"] = st["p5_H"] / st["p5_BF"]
    st["p5_bb_bf"] = st["p5_BB"] / st["p5_BF"]   # p_conv_r input (#28)
    st["p_ip_per_start"] = st["ps_Outs"] / 3 / st["p_starts_season"]
    add_pit_trends(st)
    st = st.merge(_league_env_table(gb), on="Date", how="left")

    # the manager's leash (2026-07-14 #16): his TEAM's as-of average
    # starter outs per game — organizational policy, not his own history
    lea = _team_starter_outs_table(gp)
    lea = lea.rename(columns={"cum_Outs": "lea_cum_Outs",
                              "cum_n": "lea_cum_n"})
    st = _asof_merge(st, lea, by=["Team", "Season"])
    st["team_st_outs_pg"] = st["lea_cum_Outs"] / st["lea_cum_n"]

    # audit wave (rank 6): the starter's OWN bullpen state — the manager's
    # pull decision runs on his own pen's fatigue/quality (team_st_outs_pg
    # is static policy; this is the dynamic day-of input). Re-key the same
    # leakage-vetted pen tables to Team (distinct openp_/openhl_ prefixes
    # avoid the opponent-keyed pen_cum_* intermediates from _attach_context)
    own = pen_tab.rename(columns={c: f"openp_{c}" for c in pen_tab.columns
                                  if c.startswith("cum")})
    st = _asof_merge(st, own, by=["Team", "Season"])
    st["own_pen_era"] = st["openp_cum_ER"] * 27 / st["openp_cum_Outs"]
    ohl = _bullpen_hl_table(gp).rename(
        columns={c: f"openhl_{c}" for c in _bullpen_hl_table(gp).columns
                 if c.startswith("cum")})
    st = _asof_merge(st, ohl, by=["Team", "Season"])
    st["own_pen_hl_era"] = st["openhl_cum_ER"] * 27 / st["openhl_cum_Outs"]
    ofat = _pen_fatigue_table(gp).rename(columns={"pen_np_l3": "own_pen_np_l3"})
    st = st.merge(ofat, on=["Team", "Date"], how="left")

    # audit wave (rank 26): the OWN defense's unearned-run rate — per is
    # the only head whose label is defined by the earned/unearned split and
    # it has zero error signal today (own_def_p_oaa is range, not errors)
    od = _team_defense_table(gp).rename(
        columns={c: f"od_{c}" for c in _team_defense_table(gp).columns
                 if c.startswith("cum")})
    st = _asof_merge(st, od, by=["Team", "Season"])
    st["own_def_uer"] = (st["od_cum_R"] - st["od_cum_ER"]) * 27 / st["od_cum_Outs"]

    # audit wave (rank 24): the running game vs THIS starter — his own
    # prior-season SB-allowed permissiveness (nothing on the starts frame
    # sees the running game today) + the regime-aware league centering
    st = _merge_prior_season(st, _pitcher_season_table(raw["pit_season"]),
                             "PlayerId", ["psb_sb27", "psb_stop"])
    st = st.merge(_league_sb27_table(raw["pit_season"]), on="Season",
                  how="left")

    # schedule context (2026-07-14 #18/#24): his team's travel/day-night
    # fatigue + tonight's doubleheader flags (game 2 = spot starters,
    # tired pens — outs/exposure policy the frames couldn't see)
    st = st.merge(_team_sched_table(raw["games"], raw["parks"]),
                  on=["GamePk", "Team"], how="left")

    # his own contact quality ALLOWED (career + decayed xwOBA-on-contact)
    # — the damage half of the earned-run conversion chain (#28); same
    # tables + shrink as the batter frame's pbip_* so serving
    # (Stores.bip_pitcher) matches by construction
    if raw.get("bip") is not None:
        pt0 = _bip_table(raw["bip"], "PitcherId").rename(
            columns={"PitcherId": "PlayerId"})
        pt0 = pt0.rename(columns={c: f"pb_{c}" for c in pt0.columns
                                  if c.startswith(("cum_", "dk_"))})
        st = _asof_merge(st, pt0, by=["PlayerId"])
        wdn_bip = np.exp(-DECAY_LAM * (st["Date"] - DECAY_EPOCH)
                         .dt.days.to_numpy(dtype="float64"))
        # #28 xwoba + audit-wave rank 5 contact-shape widening (the starts
        # frame's only contact-allowed view was one xwOBA scalar; batter +
        # team frames both carry the GB axis) + rank 10 xba (the pitcher-
        # side hit-luck residual's expected side). Tags pbip/pbipd produce
        # the exact names Stores.bip_pitcher already emits -> serving parity.
        for k, v in bip_feats(lambda c: st[f"pb_cum_{c}"], "pbip",
                              ("xwoba", "ev", "brl", "gb", "flyd",
                               "pullair")).items():
            st[k] = v
        for k, v in bip_feats(lambda c: st[f"pb_dk_{c}"] * wdn_bip, "pbipd",
                              ("xwoba", "gb", "brl", "xba")).items():
            st[k] = v
    else:
        for c in ("pbip_xwoba", "pbip_ev", "pbip_brl", "pbip_gb",
                  "pbip_flyd", "pbip_pullair",
                  "pbipd_xwoba", "pbipd_gb", "pbipd_brl", "pbipd_xba"):
            st[c] = np.nan

    # the starter's own arsenal, K-model view (whiff/K%/put-away, blended
    # over the last two Statcast seasons)
    pa = pitcher_arsenal_feats(
        st[["PlayerId", "Season"]].rename(columns={"PlayerId": "PitcherId"}),
        raw["ars_p"]).rename(columns={"PitcherId": "PlayerId"})
    st = st.merge(pa, on=["PlayerId", "Season"], how="left")
    # audit wave (rank 22): arsenal breadth + trajectory (the 0.7/0.3 level
    # blend hides both); annual, leakage-safe (serve season reads Y<=S-1)
    st = st.merge(_arsenal_dynamics(raw["ars_p"]),
                  on=["PlayerId", "Season"], how="left")

    # pitch-level dailies: as-of swinging-strike / CSW / whiff-per-swing /
    # chase-induced / zone rates (90-day decay + career) and the fastball
    # velocity trend (decayed minus career — the classic decline signal).
    # Unlike the prior-season arsenal blend, these move DURING the season.
    PD_C = ("swstr", "fbv")
    PD_D = PD_PITCHER_D
    if raw.get("pdp") is not None:
        pt2 = _cum_decay_table(raw["pdp"], "PlayerId")
        pt2 = pt2.rename(columns={c: f"pd_{c}" for c in pt2.columns
                                  if c.startswith(("cum_", "dk_"))})
        st = _asof_merge(st, pt2, by=["PlayerId"])
        wdn_pd = np.exp(-DECAY_LAM * (st["Date"] - DECAY_EPOCH)
                        .dt.days.to_numpy(dtype="float64"))
        for name in PD_C:
            prior, k, num, den = PD_SHRINK[name]
            st[f"pd_{name}_c"] = ((st[f"pd_cum_{num}"] + k * prior)
                                  / (st[f"pd_cum_{den}"] + k))
        for name in PD_D:
            prior, k, num, den = PD_SHRINK[name]
            st[f"pd_{name}_d"] = ((st[f"pd_dk_{num}"] * wdn_pd + k * prior)
                                  / (st[f"pd_dk_{den}"] * wdn_pd + k))
        st["pd_fbv_tr"] = st["pd_fbv_d"] - st["pd_fbv_c"]
        # v5 dispersion reads (decayed; the discount cancels in the
        # ratios, so it only sets the effective-sample gates)
        st["pd_fbv_sd"] = velo_sd_from_sums(
            st["pd_dk_fb_n"] * wdn_pd, st["pd_dk_fb_v"] * wdn_pd,
            st["pd_dk_fb_v2"] * wdn_pd)
        st["pd_rel_sd"] = release_scatter_from_sums(
            st["pd_dk_rp_n"] * wdn_pd, st["pd_dk_rp_x"] * wdn_pd,
            st["pd_dk_rp_x2"] * wdn_pd, st["pd_dk_rp_z"] * wdn_pd,
            st["pd_dk_rp_z2"] * wdn_pd)
        # v6 (#26): decayed weighted-mean per-start velo-fade slope
        st["pd_fbv_fade"] = fbv_fade_from_sums(
            st["pd_dk_fade_w"] * wdn_pd, st["pd_dk_fade_num"] * wdn_pd)
        # v7 audit wave (rank 27): base-out stretch delta + per-class
        # release separation (the shrink-ratio siblings pd_effprem_d /
        # pd_brkmov_d ride the PD_D loop above; these two need the helper)
        st["pd_stretch_vdelta_d"] = stretch_vdelta_from_sums(
            st["pd_dk_fb_n"] * wdn_pd, st["pd_dk_fb_v"] * wdn_pd,
            st["pd_dk_fbstr_n"] * wdn_pd, st["pd_dk_fbstr_v"] * wdn_pd)
        st["pd_relsep_d"] = release_sep_from_sums(
            st["pd_dk_rpf_n"] * wdn_pd, st["pd_dk_rpf_x"] * wdn_pd,
            st["pd_dk_rpf_z"] * wdn_pd, st["pd_dk_rpb_n"] * wdn_pd,
            st["pd_dk_rpb_x"] * wdn_pd, st["pd_dk_rpb_z"] * wdn_pd)
    else:
        for name in PD_C:
            st[f"pd_{name}_c"] = np.nan
        for name in PD_D:
            st[f"pd_{name}_d"] = np.nan
        st["pd_fbv_tr"] = np.nan
        st["pd_fbv_sd"] = np.nan
        st["pd_rel_sd"] = np.nan
        st["pd_fbv_fade"] = np.nan
        st["pd_stretch_vdelta_d"] = np.nan
        st["pd_relsep_d"] = np.nan

    # the ACTUAL opposing lineup (not just team-season rates): mean as-of
    # shrunk K%/BB%, whiff vs this starter's arsenal, and K% vs his hand,
    # aggregated over the nine batters he faces. Needs the built batter
    # frame; columns stay NaN without it (old cache compatibility).
    if batter_frame is not None:
        lu = (batter_frame.groupby(["GamePk", "Team"])
              .agg(lu_k_sh=("s_k_pct_sh", "mean"),
                   lu_bb_sh=("s_bb_pct_sh", "mean"),
                   lu_whiff=("m_whiff", "mean"),
                   lu_vsh_k=("vsh_k_pct_sh", "mean"),
                   lu_wsw=("bd_wsw_d", "mean"),
                   lu_chase=("bd_chase_d", "mean"),
                   lu_brkwh=("bd_brkwh_d", "mean"),
                   lu_offwh=("bd_offwh_d", "mean"),
                   lu_fbwh=("bd_fbwh_d", "mean"),
                   # audit wave rank 18 (damage/OBP the whiff-only lu_ view
                   # missed) + rank 5c (rally-kill) + rank 24 (running game)
                   lu_obp=("_obpp", "mean"), lu_slg=("_slgp", "mean"),
                   lu_xwcon=("bipd_xwoba", "mean"),
                   lu_brl=("bipd_brl", "mean"),
                   lu_fpsw=("bd_fpsw_d", "mean"),
                   lu_gidp=("c_gidp_pa_sh", "mean"),
                   lu_sb=("d_sb_pa_sh", "mean"),
                   lu_sprint=("bat_sprint", "mean"),
                   # v8 (2026-07-15): the lineup's whiff-by-velo-band and
                   # damage-by-band/class views (career damage reads —
                   # see the PD_SHRINK v8 note on thin per-band BBE)
                   lu_fblowh=("bd_fblowh_d", "mean"),
                   lu_fbmidwh=("bd_fbmidwh_d", "mean"),
                   lu_fb95wh=("bd_fb95wh_d", "mean"),
                   lu_fbloxw=("bd_fbloxw_c", "mean"),
                   lu_fbmidxw=("bd_fbmidxw_c", "mean"),
                   lu_fb95xw=("bd_fb95xw_c", "mean"),
                   lu_brkxw=("bd_brkxw_c", "mean"),
                   lu_offxw=("bd_offxw_c", "mean"),
                   lu_fbkxw=("bd_fbkxw_c", "mean"))
              .reset_index().rename(columns={"Team": "Opponent"}))
        st = st.merge(lu, on=["GamePk", "Opponent"], how="left")

    # home-plate umpire zone tendency (as-of), merged by game
    st = _merge_ump(st, raw)

    # the starter's own times-through-order decay (2026-07-12): how much his
    # contact quality allowed slips the 3rd time through — speaks directly
    # to the outs/hits/earned-run heads
    if raw.get("bip") is not None:
        tt = _tto_table(raw["bip"]).rename(columns={"PitcherId": "PlayerId"})
        tt["PlayerId"] = tt["PlayerId"].astype(st["PlayerId"].dtype)
        st = _asof_merge(st, tt, by=["PlayerId"])
    else:
        for c in ("tto_cum_xw1_n", "tto_cum_xw1_sum",
                  "tto_cum_xw3_n", "tto_cum_xw3_sum"):
            st[c] = np.nan

    # the ACTUAL defense playing behind him tonight (player-level
    # prior-season OAA of his own team's posted lineup, 2026-07-12)
    if raw.get("oaa_players") is not None:
        ld = _lineup_oaa_table(gb, raw["oaa_players"]).rename(
            columns={"def_p_oaa": "own_def_p_oaa",
                     "def_p_if": "own_def_p_if", "def_p_of": "own_def_p_of"})
        st = st.merge(ld, on=["GamePk", "Team"], how="left")
    else:
        for c in ("own_def_p_oaa", "own_def_p_if", "own_def_p_of"):
            st[c] = np.nan

    # his OWN battery, prior season (2026-07-15): framing (strikes FOR
    # him), CS value and pop time (his catchers' running-game wall) —
    # team playing-time-weighted, serving-safe
    if raw.get("cat") is not None:
        ct = raw["cat"].rename(columns={
            "Team": "PlayerId", "FrameRV_pt": "own_cat_frame",
            "CSAA_att": "own_cat_csaa", "PopTime": "own_cat_pop"})
        st = _merge_prior_season(
            st, ct[["PlayerId", "Year", "own_cat_frame", "own_cat_csaa",
                    "own_cat_pop"]],
            "Team", ["own_cat_frame", "own_cat_csaa", "own_cat_pop"])
    else:
        for c in ("own_cat_frame", "own_cat_csaa", "own_cat_pop"):
            st[c] = np.nan
    # his own IL return context (2026-07-15): pairs with the #17 ramp
    # flags — a 15-day gap off the IL is a different animal than a
    # skipped start, and the gap features alone can't tell them apart
    st = _il_asof(st, raw.get("il"), "PlayerId", "p_")

    # MiLB translated prior, pitcher-allowed side (2026-07-13 rider) —
    # same semantics as the batter block (exact-season join, NaN without
    # kept-level minors in window)
    if raw.get("milb") is not None:
        st = st.merge(_milb_cols(raw["milb"]["pit"], "pmilb_"),
                      on=["PlayerId", "Season"], how="left")
    else:
        for c in MILB_PIT_COLS:
            st[c] = np.nan

    # audit wave (rank 20): pitcher bio — bat bio is exposed and selection-
    # kept in 17 heads; the pitcher side had nothing but throws. Age is
    # distinct geometry from experience (p_starts_career) and realized
    # decline (pd_fbv_tr, noise on the thin histories where age matters).
    st = _attach_bio(st, raw["rosters"], "PlayerId", "pit")
    st = st.drop(columns=["B", "T"], errors="ignore")

    # shared derived features (weather, TTO decay, K-BB, lineup collisions)
    # — the serving path calls the same function (parity)
    st = add_starter_derived(st)

    st["month"] = st["Date"].dt.month
    st["y_so"] = st["SO"]
    # count-market targets for the starter prop heads
    st["y_outs"] = st["Outs"]
    st["y_pbb"] = st["BB"]
    st["y_pha"] = st["H"]
    st["y_per"] = st["ER"]  # earned runs allowed -> expected-ER head (xER)
    return st


def starts_feature_cols():
    return ["Season", "month", "Home", "p_starts_career", "p_starts_season",
            "p_days_rest", "p_np_last", "p_np_l3",
            "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_era",
            "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_era",
            "pc_strike_pct", "ps_strike_pct",
            "p5_hr_bf", "p5_k_bf", "p_ip_per_start",
            "p5_k_trend", "p_era_trend",
            "pars_whiff", "pars_kpct", "pars_paway", "pars_rv100", "pars_cov",
            *ARS_DYN_COLS,   # audit wave rank 22 (own arsenal dynamics)
            "pd_swstr_c", "pd_swstr_d", "pd_csw_d", "pd_wsw_d", "pd_chase_d",
            "pd_zone_d", "pd_fbv_c", "pd_fbv_d", "pd_fbv_tr",
            "lu_k_sh", "lu_bb_sh", "lu_whiff", "lu_vsh_k",
            "lu_wsw", "lu_chase",
            "vs_k_pct", "vs_bb_pct", "vs_hr_pa", "vs_r_pg",
            "park_hr_pg", "Elevation_ft", "Temp", "WindSpeed",
            # multi-dim park factors (as-of venue R/H/2B/TB per game): the
            # run/hit environment for the outs/hits/earned-run heads; the K
            # and walk heads drop them (train.py k_cols / st_exclude),
            # mirroring the batter-side _PARK_OFF routing
            "park_r_pg", "park_h_pg", "park_2b_pg", "park_tb_pg",
            "lg_k_pa", "lg_r_pa", "lg_hr_pa",
            # HP-umpire zone tendency: the K model uses both; the count
            # heads that don't speak to the zone (outs/pha/per) drop them
            # via train.py COUNT_HEADS st_exclude
            "ump_k_pct", "ump_bb_pct",
            # 2026-07-12 data batch: in-zone whiff (pure stuff), air density
            # (run environment for outs/pha/per), TTO decay, and the actual
            # lineup defense behind him (player-level prior-season OAA)
            "pd_zwsw_d", "hum_eff", "air_dens", "p_tto_decay",
            "own_def_p_oaa", "own_def_p_if", "own_def_p_of",
            # closing sweep: K-BB arithmetic + lineup-collision products
            # (the lineup's whiff/K/chase x his stuff — log5, lineup level)
            "pc_kbb", "lu_mix_whiff", "lu_mix_k", "lu_mix_chase",
            # v3 scrape schema (2026-07-12): whiff induced by pitch class,
            # usage mix, shadow-band command, first-pitch strike%, the
            # lineup's class-whiff splits and the usage-weighted collision
            "pd_brkwh_d", "pd_offwh_d", "pd_fbwh_d",
            "pd_brk_d", "pd_off_d", "pd_edge_d", "pd_fps_d",
            "lu_brkwh", "lu_offwh", "lu_fbwh", "lu_ars_whiff",
            # v4 graded velocity bands: usage mix + whiff induced per band
            "pd_fblou_d", "pd_fbmidu_d", "pd_fb95u_d",
            "pd_fblowh_d", "pd_fbmidwh_d", "pd_fb95wh_d",
            # v5 count leverage + dispersion: put-away whiff, 3-2 zone/
            # ball behavior, velo spread, release-point scatter
            "pd_tswh_d", "pd_f32z_d", "pd_f32b_d",
            "pd_fbv_sd", "pd_rel_sd",
            # MiLB translated prior, allowed side (2026-07-13 rider)
            *MILB_PIT_COLS,
            # 2026-07-14 finish batch (backlog #16/#17/#21/#23/#25-28/#29 +
            # #18/#24 sched): outing shape + the manager's leash, layoff/
            # ramp flags, ump run environment, own venue split, 0-2 waste /
            # ahead-behind mix shift / transition shares / FF ride / velo
            # fade, contact allowed + the earned-run conversion chain,
            # ump x lineup K, schedule/travel + doubleheader flags
            "p_outs_sd", "p_short_share", "team_st_outs_pg",
            "p_gap15", "p_ramp60", "p_short_heavy",
            "ump_r_g",
            "pvloc_era", "pvloc_k_bf", "pvloc_hr_bf",
            "pd_w02_d", "pd_absh_d", "pd_trsame_d", "pd_trfbbrk_d",
            "pd_ivb_d", "pd_fbv_fade",
            "pbip_xwoba", "pbipd_xwoba", "p5_bb_bf",
            "p_conv_c", "p_conv_r", "ump_k_x_lu",
            "day_after_night", "travel_km", "tz_delta", "is_dh", "dh_game2",
            "Precip",
            # ---- audit wave (2026-07-14, AUDIT_BUILD_SPECS_0714.md) ----
            # rank 5: contact-shape allowed (the frame's only view was one
            # xwOBA scalar) + the air/DP collisions
            "pbip_ev", "pbip_brl", "pbip_gb", "pbip_flyd", "pbip_pullair",
            "pbipd_gb", "pbipd_brl", "p_gb_air", "gb_dp_x", "lu_gidp",
            # rank 6: own bullpen state (the manager's day-of leash input)
            "own_pen_era", "own_pen_hl_era", "own_pen_np_l3",
            # rank 10: starter BABIP-luck residual (no realized H-allowed
            # rate existed on the frame)
            "pbipd_xba", "p_hit_luck",
            # rank 16: signed wind carry
            "wind_carry",
            # rank 18: posted-lineup damage/OBP view + gaps + fp economy
            "lu_obp", "lu_slg", "lu_xwcon", "lu_brl", "lu_fpsw",
            "lu_obp_gap", "lu_slg_gap", "fp_lu_x",
            # rank 20: pitcher bio
            "pit_age", "pit_height", "pit_weight",
            # rank 24: the running game vs this starter
            "psb_sb27", "psb_stop", "lu_sb", "lu_sprint", "run_game_x",
            # rank 25: pitch-economy budget x efficiency
            "p_np_per_out", "p_exp_outs",
            # rank 26: own defense unearned-run rate (per's error axis)
            "own_def_uer",
            # rank 27: v7 cause axes (perceived velo, breaking movement,
            # base-out stretch delta, per-class release separation)
            "pd_effprem_d", "pd_brkmov_d", "pd_stretch_vdelta_d",
            "pd_relsep_d",
            # v8 damage-on-contact wave (2026-07-15): his own xwOBA-allowed
            # splits by velo band / pitch class, the lineup's banded
            # whiff/damage views, and the velocity/arsenal collisions at
            # lineup grain (lu_velo_k = lu_ars_whiff's velo-band sibling;
            # the _dmg twins are the pha/per side)
            "pd_fbloxw_d", "pd_fbmidxw_d", "pd_fb95xw_d",
            "pd_brkxw_d", "pd_offxw_d", "pd_fbkxw_d",
            "lu_fblowh", "lu_fbmidwh", "lu_fb95wh",
            "lu_fbloxw", "lu_fbmidxw", "lu_fb95xw",
            "lu_brkxw", "lu_offxw", "lu_fbkxw",
            "lu_velo_k", "lu_velo_dmg", "lu_ars_dmg",
            # battery + IL wave (2026-07-15): his own catchers' framing /
            # CS value / pop time + the edge-framing and running-game
            # collisions; his own IL return context (the layoff cause the
            # #17 gap flags can't see)
            "own_cat_frame", "own_cat_csaa", "own_cat_pop",
            "frame_x_edge", "run_cat_x",
            "p_il_ret_days", "p_il_last_len", "p_il_ret21",
            "p_il_szn_days", "p_il_rehab",
            "DayNight", "Condition", "WindDir"]


def lineup_aggregates(bf):
    """Per (GamePk, Team): posted-lineup means the GAME frame consumes
    (2026-07-14 #19/#30/#32) — as-of career OBP/SLG (the B-lineup gap's
    lineup side), pulled-air share (the air-density collision's lineup
    side), and the mean usage-weighted arsenal whiff vs tonight's opposing
    starter (the cross-grain collision, already computed per batter row as
    arsenal_whiff). NaN-skipping means, exactly what the serving path
    computes from its own batter rows."""
    # rank 33 platoon edges (_vsh_tb_edge / _vsh_k_edge) are precomputed
    # per batter row in add_batter_derived so serving and training share
    # one definition; here we just average them over the posted lineup.
    return (bf.groupby(["GamePk", "Team"])
            .agg(lu_obp=("_obpp", "mean"), lu_slg=("_slgp", "mean"),
                 lu_pullair=("bip_pullair", "mean"),
                 lu_arswh=("arsenal_whiff", "mean"),
                 lu_vsh_edge=("_vsh_tb_edge", "mean"),
                 lu_vsh_kedge=("_vsh_k_edge", "mean"))
            .reset_index())


def add_team_game_derived(d):
    """Row-wise derived team-game features, shared by build_team_game_frame
    and the serving _team_rows (parity): thin air x how fly-ball-heavy this
    lineup is (2026-07-14 #30 — game-grain sibling of air_fly)."""
    d["air_lu_fly"] = (AIR_RHO0 - d["air_dens"]) * d["off_lu_pullair"]
    # audit wave rank 8: patience x wildness (a walk-drawing offense vs a
    # wild staff compounds; centered on the league walk rate, the mix_bb
    # idiom at team grain)
    for c in ("off_bb_pct", "opp_ps_bb_bf", "lg_bb_pa", "opp_ps_len",
              "opp_pen_era", "opp_ps_era"):
        if c not in d.columns:
            d[c] = np.nan
    d["patience_wild"] = ((d["off_bb_pct"] - d["lg_bb_pa"])
                          * (d["opp_ps_bb_bf"] - d["lg_bb_pa"]))
    # rank 9: expected extra runs from the starter->pen handoff — how many
    # of the 9 innings the pen must cover x how much worse it is (both
    # ERAs are per-9, so the product is in runs). Guard the early-season
    # 0-outs inf ERAs -> NaN (GBM imputes) rather than +/-inf.
    _gap = (d["opp_pen_era"] - d["opp_ps_era"]).replace(
        [np.inf, -np.inf], np.nan)
    d["xpen_r_gap"] = np.clip(9.0 - d["opp_ps_len"], 0.0, 9.0) * _gap / 9.0
    return d


def build_game_frame(raw, batter_frame=None):
    """Training frame for game totals and the winner model: one row per game.
    batter_frame (the built batter frame) supplies the posted-lineup
    aggregates (lineup_aggregates); without it those columns are NaN."""
    games, gb, gp = raw["games"], raw["gb"], raw["gp"]
    team_tab = _team_offense_table(gb)
    loc_tab = _team_offense_loc_table(gb)
    pen_tab = _bullpen_table(gp)
    pen_hl_tab = _bullpen_hl_table(gp)
    pen_fat_tab = _pen_fatigue_table(gp)
    def_tab = _team_defense_table(gp)
    lea_tab = _team_starter_outs_table(gp)   # rank 9: the manager's leash
    park_tab = _park_table(gb, games)
    res_tab = _team_results_table(games)
    form_tab = _team_form_table(games)
    st = gp.loc[gp["GS"] == 1, ["GamePk", "Team", "PlayerId"]]
    stf = _starter_asof(gp)
    stf = pd.concat([stf, _pit_rates(stf, "ps"), _pit_rates(stf, "pc")], axis=1)
    stf["p5_k_bf"] = stf["p5_SO"] / stf["p5_BF"]
    # audit wave (rank 9): starter length — how tonight's 9 innings divide
    # between the rotation and the pen (a 4.2-IP opener team and a 6.2-IP
    # workhorse look identical to the current rate mains)
    stf["p_ip_per_start"] = stf["ps_Outs"] / 3 / stf["p_starts_season"]

    # starter contact quality ALLOWED (Statcast): career + 90-day-decayed
    # xwOBA-on-contact, decayed barrel and ground-ball shares. Same shrink
    # constants as the batter frame's pbip_* (bip_feats), so the inference
    # path (Stores.bip_pitcher) matches by construction.
    ST_BIP = ["ps_xwcon", "ps_xwcon_d", "ps_brl_d", "ps_gb_d"]
    if raw.get("bip") is not None:
        pt = _bip_table(raw["bip"], "PitcherId").rename(
            columns={"PitcherId": "PlayerId"})
        pt = pt.rename(columns={c: f"pb_{c}" for c in pt.columns
                                if c.startswith(("cum_", "dk_"))})
        stf = _asof_merge(stf, pt, by=["PlayerId"])
        stf["ps_xwcon"] = bip_feats(
            lambda c: stf[f"pb_cum_{c}"], "t", ("xwoba",))["t_xwoba"]
        # discount dk_* to each start's date (see build_batter_frame note)
        wdn_st = np.exp(-DECAY_LAM * (stf["Date"] - DECAY_EPOCH)
                        .dt.days.to_numpy(dtype="float64"))
        dk = bip_feats(lambda c: stf[f"pb_dk_{c}"] * wdn_st, "t",
                       ("xwoba", "brl", "gb"))
        stf["ps_xwcon_d"] = dk["t_xwoba"]
        stf["ps_brl_d"] = dk["t_brl"]
        stf["ps_gb_d"] = dk["t_gb"]
    else:
        for c in ST_BIP:
            stf[c] = np.nan

    # opposing-starter TTO decay for the run-total environment (2026-07-12
    # full-surface pass): same table + shared helper as the batter/starts
    # frames; no-history and no-bip both land at 0.0 (fillna inside)
    if raw.get("bip") is not None:
        tt = _tto_table(raw["bip"]).rename(columns={"PitcherId": "PlayerId"})
        tt["PlayerId"] = tt["PlayerId"].astype(stf["PlayerId"].dtype)
        stf = _asof_merge(stf, tt, by=["PlayerId"])
    else:
        for c in ("tto_cum_xw1_n", "tto_cum_xw1_sum",
                  "tto_cum_xw3_n", "tto_cum_xw3_sum"):
            stf[c] = np.nan
    stf["ps_tto_decay"] = tto_decay_from_sums(stf)

    # team-level contact quality: own offense + bullpen allowed
    if raw.get("bip") is not None:
        bip_off_tab, bip_pen_tab = _bip_team_tables(raw["bip"], gb, gp)
    else:
        bip_off_tab = bip_pen_tab = None

    # posted-lineup tables (2026-07-12 full-surface pass): player-level
    # prior-season defense and baserunning of tonight's actual nine —
    # merged per side below, consumed by the team-runs frame
    ldef_tab = (_lineup_oaa_table(gb, raw["oaa_players"])
                if raw.get("oaa_players") is not None else None)
    lbrr_tab = (_lineup_brr_table(gb, raw["baserun"])
                if raw.get("baserun") is not None else None)
    # posted-lineup quality/style aggregates from the built batter frame
    # (2026-07-14 #19/#30/#32) + schedule context and the trailing-30-day
    # BaseRuns luck residual (#18/#20/#24)
    lagg_tab = (lineup_aggregates(batter_frame)
                if batter_frame is not None else None)
    sched_tab = _team_sched_table(games, raw["parks"])
    bsr_tab = _team_bsr_table(gb)

    g = games[~games["ShortGame"]].copy()
    away_sc = pd.to_numeric(g["AwayScore"], errors="coerce")
    home_sc = pd.to_numeric(g["HomeScore"], errors="coerce")
    g["total_runs"] = away_sc + home_sc
    g["y_home_win"] = np.where(away_sc.notna() & home_sc.notna(),
                               (home_sc > away_sc).astype(float), np.nan)

    ST_COLS = ["ps_era", "ps_k_bf", "ps_hr_bf", "ps_bb_bf", "ps_h_bf",
               "pc_era", "pc_hr_bf", "p_days_rest", "p5_k_bf",
               "ps_tto_decay", "p_ip_per_start", *ST_BIP]   # rank 9 length
    for side, team_col in [("away", "AwayTeam"), ("home", "HomeTeam")]:
        t = team_tab.rename(columns={"Team": team_col})
        t = t.rename(columns={c: f"{side}_{c}" for c in t.columns if c.startswith("cum")})
        g = _asof_merge(g, t, by=[team_col, "Season"])
        g[f"{side}_hr_pa"] = g[f"{side}_cum_HR"] / g[f"{side}_cum_PA"]
        g[f"{side}_r_pg"] = g[f"{side}_cum_R"] / g[f"{side}_cum_n"]
        g[f"{side}_k_pct"] = g[f"{side}_cum_SO"] / g[f"{side}_cum_PA"]
        # audit wave (rank 8): the walk channel — no walk encoding existed
        # on either side of the 64-col team frame (the largest run-scoring
        # input it couldn't see)
        g[f"{side}_bb_pct"] = g[f"{side}_cum_BB"] / g[f"{side}_cum_PA"]
        # season OBP/SLG norms — the B-lineup gap's team side (#19)
        g[f"{side}_obp"] = ((g[f"{side}_cum_H"] + g[f"{side}_cum_BB"]
                             + g[f"{side}_cum_HBP"]) / g[f"{side}_cum_PA"])
        g[f"{side}_slg"] = g[f"{side}_cum_TB"] / g[f"{side}_cum_AB"]
        # offense split for the venue context this side plays in tonight
        lo = loc_tab[loc_tab["Home"] == (1 if side == "home" else 0)]
        lo = lo.drop(columns=["Home"]).rename(columns={"Team": team_col})
        lo = lo.rename(columns={c: f"{side}_loc_{c}" for c in lo.columns
                                if c.startswith("cum")})
        g = _asof_merge(g, lo, by=[team_col, "Season"])
        g[f"{side}_loc_hr_pa"] = g[f"{side}_loc_cum_HR"] / g[f"{side}_loc_cum_PA"]
        g[f"{side}_loc_r_pg"] = g[f"{side}_loc_cum_R"] / g[f"{side}_loc_cum_n"]
        p = pen_tab.rename(columns={"Team": team_col})
        p = p.rename(columns={c: f"{side}_pen_{c}" for c in p.columns if c.startswith("cum")})
        g = _asof_merge(g, p, by=[team_col, "Season"])
        g[f"{side}_pen_era"] = g[f"{side}_pen_cum_ER"] * 27 / g[f"{side}_pen_cum_Outs"]
        if bip_off_tab is not None:
            bo = bip_off_tab.rename(columns={"Team": team_col})
            bo = bo.rename(columns={c: f"{side}_bo_{c}" for c in bo.columns
                                    if c.startswith("cum")})
            g = _asof_merge(g, bo, by=[team_col, "Season"])
            g[f"{side}_xwcon"] = (g[f"{side}_bo_cum_xw_sum"]
                                  / g[f"{side}_bo_cum_xw_n"])
            g[f"{side}_brl_con"] = (g[f"{side}_bo_cum_brl_n"]
                                    / g[f"{side}_bo_cum_ev_n"])
            bp = bip_pen_tab.rename(columns={"Team": team_col})
            bp = bp.rename(columns={c: f"{side}_bp_{c}" for c in bp.columns
                                    if c.startswith("cum")})
            g = _asof_merge(g, bp, by=[team_col, "Season"])
            g[f"{side}_pen_xwcon"] = (g[f"{side}_bp_cum_xw_sum"]
                                      / g[f"{side}_bp_cum_xw_n"])
        else:
            for c in (f"{side}_xwcon", f"{side}_brl_con",
                      f"{side}_pen_xwcon"):
                g[c] = np.nan
        if raw.get("oaa") is not None:
            oa = raw["oaa"].rename(columns={"Team": "PlayerId",
                                            "OAA_per162": f"{side}_oaa"})
            g = _merge_prior_season(
                g, oa[["PlayerId", "Year", f"{side}_oaa"]], team_col,
                [f"{side}_oaa"])
        else:
            g[f"{side}_oaa"] = np.nan
        hl = pen_hl_tab.rename(columns={"Team": team_col})
        hl = hl.rename(columns={c: f"{side}_penhl_{c}" for c in hl.columns
                                if c.startswith("cum")})
        g = _asof_merge(g, hl, by=[team_col, "Season"])
        g[f"{side}_pen_hl_era"] = (g[f"{side}_penhl_cum_ER"] * 27
                                   / g[f"{side}_penhl_cum_Outs"])
        # audit wave (opp_pen_bridge gray -> share-only rider): the share
        # of pen outs thrown by high-leverage arms — the mixture weight the
        # two pen-ERA mains can't recover (bridge ERA itself was 95%
        # redundant; the user kept only this)
        g[f"{side}_penhl_share"] = (g[f"{side}_penhl_cum_Outs"]
                                    / g[f"{side}_pen_cum_Outs"])
        fat = pen_fat_tab.rename(columns={"Team": team_col,
                                          "pen_np_l3": f"{side}_pen_np_l3"})
        g = g.merge(fat, on=[team_col, "Date"], how="left")
        # audit wave (rank 9): the team's as-of average starter outs (the
        # manager's organizational leash — the static policy paired with the
        # dynamic p_ip_per_start length)
        le = lea_tab.rename(columns={"Team": team_col})
        le = le.rename(columns={c: f"{side}_st_{c}" for c in le.columns
                                if c.startswith("cum")})
        g = _asof_merge(g, le, by=[team_col, "Season"])
        g[f"{side}_st_outs_pg"] = g[f"{side}_st_cum_Outs"] / g[f"{side}_st_cum_n"]
        d = def_tab.rename(columns={"Team": team_col})
        d = d.rename(columns={c: f"{side}_def_{c}" for c in d.columns
                              if c.startswith("cum")})
        g = _asof_merge(g, d, by=[team_col, "Season"])
        g[f"{side}_def_uer"] = ((g[f"{side}_def_cum_R"] - g[f"{side}_def_cum_ER"])
                                * 27 / g[f"{side}_def_cum_Outs"])
        r = res_tab.rename(columns={"Team": team_col})
        r = r.rename(columns={c: f"{side}_res_{c}" for c in r.columns if c.startswith("cum")})
        g = _asof_merge(g, r, by=[team_col, "Season"])
        rec = _record_feats(g[f"{side}_res_cum_W"], g[f"{side}_res_cum_RF"],
                            g[f"{side}_res_cum_RA"], g[f"{side}_res_cum_n"])
        for k, v in rec.items():
            g[f"{side}_{k}"] = v
        fm = form_tab.rename(columns={"Team": team_col, "w20": f"{side}_w20",
                                      "rd20": f"{side}_rd20"})
        g = g.merge(fm, on=["GamePk", team_col], how="left")
        sm = st.rename(columns={"Team": team_col, "PlayerId": f"{side}_starter"})
        g = g.merge(sm, on=["GamePk", team_col], how="left")
        sf = stf[["GamePk", "PlayerId", *ST_COLS]].rename(
            columns={"PlayerId": f"{side}_starter"})
        sf = sf.rename(columns={c: f"{side}_{c}" for c in ST_COLS})
        g = g.merge(sf, on=["GamePk", f"{side}_starter"], how="left")
        # this side's POSTED lineup: player-level prior-season defense and
        # baserunning (the team frame reads defense from the OPPOSING side)
        if ldef_tab is not None:
            ld = ldef_tab.rename(columns={
                "Team": team_col, "def_p_oaa": f"{side}_ldef_oaa",
                "def_p_if": f"{side}_ldef_if", "def_p_of": f"{side}_ldef_of"})
            g = g.merge(ld, on=["GamePk", team_col], how="left")
        else:
            for c in (f"{side}_ldef_oaa", f"{side}_ldef_if",
                      f"{side}_ldef_of"):
                g[c] = np.nan
        if lbrr_tab is not None:
            lb = lbrr_tab.rename(columns={
                "Team": team_col, "lu_brr": f"{side}_lu_brr",
                "lu_brr_xb": f"{side}_lu_brr_xb"})
            g = g.merge(lb, on=["GamePk", team_col], how="left")
        else:
            g[f"{side}_lu_brr"] = np.nan
            g[f"{side}_lu_brr_xb"] = np.nan
        # posted-lineup quality/style aggregates (2026-07-14 #19/#30/#32):
        # the B-lineup gap = tonight's lineup vs the team's own season norm
        if lagg_tab is not None:
            la = lagg_tab.rename(columns={
                "Team": team_col, "lu_obp": f"{side}_lu_obp",
                "lu_slg": f"{side}_lu_slg",
                "lu_pullair": f"{side}_lu_pullair",
                "lu_arswh": f"{side}_lu_arswh",
                # audit wave rank 33: the lineup's platoon edge tonight
                "lu_vsh_edge": f"{side}_lu_vsh_edge",
                "lu_vsh_kedge": f"{side}_lu_vsh_kedge"})
            g = g.merge(la, on=["GamePk", team_col], how="left")
        else:
            for c in (f"{side}_lu_obp", f"{side}_lu_slg",
                      f"{side}_lu_pullair", f"{side}_lu_arswh",
                      f"{side}_lu_vsh_edge", f"{side}_lu_vsh_kedge"):
                g[c] = np.nan
        g[f"{side}_lu_obp_gap"] = g[f"{side}_lu_obp"] - g[f"{side}_obp"]
        g[f"{side}_lu_slg_gap"] = g[f"{side}_lu_slg"] - g[f"{side}_slg"]
        # schedule context + trailing BaseRuns luck (2026-07-14
        # #18/#20/#24); the DH flags are game-level (identical both sides)
        sc = sched_tab.rename(columns={
            "Team": team_col, "day_after_night": f"{side}_dan",
            "travel_km": f"{side}_travel_km", "tz_delta": f"{side}_tz_delta",
            "is_dh": "is_dh", "dh_game2": "dh_game2"})
        if side == "away":      # keep one game-level copy of the DH flags
            g = g.merge(sc, on=["GamePk", team_col], how="left")
        else:
            g = g.merge(sc.drop(columns=["is_dh", "dh_game2"]),
                        on=["GamePk", team_col], how="left")
        bs = bsr_tab.rename(columns={"Team": team_col,
                                     "bsr_luck": f"{side}_bsr_luck"})
        g = g.merge(bs, on=[team_col, "Date"], how="left")

    elo_pre, _ = build_elo(games)
    g = g.merge(elo_pre, on="GamePk", how="left")
    g["d_elo"] = g["home_elo"] - g["away_elo"]

    # winner-model differentials (home minus away)
    for f in ["win_pct", "rd_pg", "pyth", "ra_pg", "r_pg", "w20", "rd20",
              "ps_era", "pc_era", "pen_era", "ps_k_bf"]:
        g[f"d_{f}"] = g[f"home_{f}"] - g[f"away_{f}"]
    g["d_rest"] = g["home_p_days_rest"] - g["away_p_days_rest"]
    # starter K-BB rate diff: the single most predictive starter skill
    g["d_ps_kbb"] = ((g["home_ps_k_bf"] - g["home_ps_bb_bf"])
                     - (g["away_ps_k_bf"] - g["away_ps_bb_bf"]))
    # starter contact-quality-allowed diff (decayed xwOBA on contact)
    g["d_ps_xwcon_d"] = g["home_ps_xwcon_d"] - g["away_ps_xwcon_d"]
    # 2026-07-14 finish batch: winner-facing diffs — sequencing luck due to
    # regress (#20), the B-lineup quality gap (#19), and the cross-grain
    # arsenal collision (#32); the small-n paired-CI caveat governs
    g["d_bsr_luck"] = g["home_bsr_luck"] - g["away_bsr_luck"]
    g["d_lu_obp_gap"] = g["home_lu_obp_gap"] - g["away_lu_obp_gap"]
    g["d_lu_arswh"] = g["home_lu_arswh"] - g["away_lu_arswh"]

    park = park_tab.rename(columns={c: f"park_{c}" for c in park_tab.columns
                                    if c.startswith("cum")})
    g = _asof_merge(g, park, by=["Venue"])
    ok = g["park_cum_n"] >= 30
    for stat, col in (("HR", "park_hr_pg"), ("R", "park_r_pg"),
                      ("H", "park_h_pg"), ("2B", "park_2b_pg"),
                      ("3B", "park_3b_pg"), ("TB", "park_tb_pg")):
        g[col] = np.where(ok, g[f"park_cum_{stat}"] / g["park_cum_n"], np.nan)
    # audit wave (rank 32): the venue run-DISPERSION (variance-to-mean
    # ratio, EB-shrunk toward the league VMR) — the second moment of runs,
    # for the totals-tail pricing layer. Column only; the NB pricing
    # substitution is gated on the 2025-only exponent fit (predict.py).
    g["park_vmr"] = park_vmr_from_sums(g["park_cum_n"], g["park_cum_R"],
                                       g["park_cum_R2"])
    parks = raw["parks"].rename(columns={"Ballpark": "Venue"})
    g = g.merge(parks[["Venue", "LF", "CF", "RF", "Elevation_ft"]], on="Venue", how="left")
    g = g.merge(_league_env_table(gb), on="Date", how="left")
    # air-density weather (humidity/pressure scrape, 2026-07-12) for the
    # run-total environment; Precip wired 2026-07-14 (#11 close-out — the
    # rain-shortening signal for the outs/total heads)
    if raw.get("weather") is not None:
        g = g.merge(raw["weather"][["GamePk", "Humidity", "Pressure",
                                    "Precip"]],
                    on="GamePk", how="left")
    else:
        g["Humidity"] = np.nan
        g["Pressure"] = np.nan
        g["Precip"] = np.nan
    add_weather_derived(g)
    # HP-ump run environment (2026-07-14 #21) for the totals surface
    g = _merge_ump(g, raw)
    g["month"] = g["Date"].dt.month
    return g


def game_feature_cols():
    cols = ["Season", "month", "Temp", "WindSpeed", "hum_eff", "air_dens",
            "park_hr_pg",
            "LF", "CF", "RF", "Elevation_ft", "DayNight", "Condition", "WindDir"]
    for side in ["away", "home"]:
        cols += [f"{side}_hr_pa", f"{side}_r_pg", f"{side}_k_pct", f"{side}_pen_era",
                 f"{side}_ps_era", f"{side}_ps_k_bf", f"{side}_ps_hr_bf",
                 f"{side}_pc_era", f"{side}_pc_hr_bf"]
    return cols


def win_feature_cols():
    """Features for the dedicated home-win classifier: team strength (win%,
    run diff, pythag, run prevention), recent form (last 20 games), both
    starters, both bullpens, rest — and explicit home-minus-away
    differentials, which is what actually decides a winner. Deliberately
    compact (~45 columns, no weather/park/categoricals): the game frame has
    ~10k training rows, and the v1 winner model showed that batter-scale
    capacity + wide features just overfits (early-stopped at 26 trees,
    test AUC 0.52)."""
    cols = ["Season", "month"]
    for side in ["away", "home"]:
        cols += [f"{side}_win_pct", f"{side}_rd_pg", f"{side}_pyth",
                 f"{side}_ra_pg", f"{side}_r_pg", f"{side}_w20",
                 f"{side}_rd20", f"{side}_pen_era",
                 f"{side}_ps_era", f"{side}_ps_k_bf", f"{side}_ps_hr_bf",
                 f"{side}_ps_bb_bf", f"{side}_pc_era", f"{side}_pc_hr_bf",
                 f"{side}_p_days_rest", f"{side}_p5_k_bf"]
    # NOTE: the starter contact-allowed diff (d_ps_xwcon_d) was tried here
    # (2026-07 Statcast batch) and pushed the winner's logloss BELOW the
    # always-home base rate — the 10k-row winner overfits when widened, same
    # as v1. It stays in the frames and the runs model; the winner sees the
    # Statcast signal only through the runs-model Poisson blend.
    cols += ["d_win_pct", "d_rd_pg", "d_pyth", "d_ra_pg", "d_r_pg",
             "d_w20", "d_rd20", "d_ps_era", "d_pc_era", "d_pen_era",
             "d_ps_k_bf", "d_ps_kbb", "d_rest",
             "away_elo", "home_elo", "d_elo", "elo_prob_home",
             # unbenched to the winner 2026-07-10 (superset dev — selection +
             # 2026 confirm decide at ship; the 10k-row overfit caveat stands).
             "d_ps_xwcon_d",
             # 2026-07-14 finish batch (#19/#20/#32 winner-facing diffs):
             # sequencing luck, B-lineup gap, arsenal collision. Small-n
             # caveat — the paired-CI read, not the keep-vote, arbitrates.
             "d_bsr_luck", "d_lu_obp_gap", "d_lu_arswh"]
    return cols


def build_team_game_frame(gf):
    """Two rows per game (one per team): own offense vs opposing pitching.
    Target = runs that team scored. One symmetric model serves both sides;
    the Home flag carries home-field advantage. Total game runs and win
    probability are derived from the two per-team predictions, so all game
    outputs stay coherent."""
    frames = []
    for side, opp, home, score in (("away", "home", 0, "AwayScore"),
                                   ("home", "away", 1, "HomeScore")):
        d = pd.DataFrame({
            "GamePk": gf["GamePk"], "Season": gf["Season"], "Date": gf["Date"],
            "month": gf["month"], "Home": home,
            "off_hr_pa": gf[f"{side}_hr_pa"], "off_r_pg": gf[f"{side}_r_pg"],
            "off_k_pct": gf[f"{side}_k_pct"],
            "off_loc_hr_pa": gf[f"{side}_loc_hr_pa"],
            "off_loc_r_pg": gf[f"{side}_loc_r_pg"],
            "off_xwcon": gf[f"{side}_xwcon"],
            "off_brl_con": gf[f"{side}_brl_con"],
            "opp_pen_era": gf[f"{opp}_pen_era"],
            "opp_pen_hl_era": gf[f"{opp}_pen_hl_era"],
            "opp_pen_np_l3": gf[f"{opp}_pen_np_l3"],
            "opp_def_uer": gf[f"{opp}_def_uer"],
            "opp_def_oaa": gf[f"{opp}_oaa"],
            "opp_pen_xwcon": gf[f"{opp}_pen_xwcon"],
            "opp_ps_era": gf[f"{opp}_ps_era"],
            "opp_ps_k_bf": gf[f"{opp}_ps_k_bf"],
            "opp_ps_hr_bf": gf[f"{opp}_ps_hr_bf"],
            "opp_ps_h_bf": gf[f"{opp}_ps_h_bf"],
            "opp_ps_xwcon": gf[f"{opp}_ps_xwcon"],
            "opp_ps_xwcon_d": gf[f"{opp}_ps_xwcon_d"],
            "opp_ps_brl_d": gf[f"{opp}_ps_brl_d"],
            "opp_ps_gb_d": gf[f"{opp}_ps_gb_d"],
            "opp_pc_era": gf[f"{opp}_pc_era"],
            "opp_pc_hr_bf": gf[f"{opp}_pc_hr_bf"],
            # 2026-07-12 full-surface pass: the ACTUAL defense this offense
            # faces (opponent's posted lineup, prior-season player OAA),
            # its own lineup's baserunning value, and the opposing
            # starter's TTO decay — the batter-frame signals at team grain
            "opp_def_p_oaa": gf[f"{opp}_ldef_oaa"],
            "opp_def_p_if": gf[f"{opp}_ldef_if"],
            "opp_def_p_of": gf[f"{opp}_ldef_of"],
            "off_lu_brr": gf[f"{side}_lu_brr"],
            "off_lu_brr_xb": gf[f"{side}_lu_brr_xb"],
            "opp_ps_tto_decay": gf[f"{opp}_ps_tto_decay"],
            # 2026-07-14 finish batch: B-lineup gap (#19), lineup air
            # profile (#30 input), cross-grain arsenal collision (#32),
            # BaseRuns luck (#20), schedule/travel + DH flags (#18/#24),
            # ump run environment (#21), precip (#11 close-out)
            "off_lu_obp_gap": gf[f"{side}_lu_obp_gap"],
            "off_lu_slg_gap": gf[f"{side}_lu_slg_gap"],
            "off_lu_pullair": gf[f"{side}_lu_pullair"],
            "off_lu_arswh": gf[f"{side}_lu_arswh"],
            "off_bsr_luck": gf[f"{side}_bsr_luck"],
            "off_dan": gf[f"{side}_dan"],
            "off_travel_km": gf[f"{side}_travel_km"],
            "off_tz_delta": gf[f"{side}_tz_delta"],
            # ---- audit wave (2026-07-14) ----
            # rank 8: the walk channel + patience_wild collision carrier
            "off_bb_pct": gf[f"{side}_bb_pct"],
            "opp_ps_bb_bf": gf[f"{opp}_ps_bb_bf"],
            "lg_bb_pa": gf["lg_bb_pa"],
            # rank 9: starter-length/pen innings split + leash + the handoff
            # run gap; gray rider: the opposing pen's HL-outs mixture weight
            "opp_ps_len": gf[f"{opp}_p_ip_per_start"],
            "opp_st_outs_pg": gf[f"{opp}_st_outs_pg"],
            "opp_penhl_share": gf[f"{opp}_penhl_share"],
            # rank 32: venue run-dispersion (totals-tail pricing input)
            "park_vmr": gf["park_vmr"],
            # rank 33: the lineup's platoon edge tonight
            "off_lu_vsh_edge": gf[f"{side}_lu_vsh_edge"],
            "off_lu_vsh_kedge": gf[f"{side}_lu_vsh_kedge"],
            "is_dh": gf["is_dh"], "dh_game2": gf["dh_game2"],
            "ump_r_g": gf["ump_r_g"], "Precip": gf["Precip"],
            "park_hr_pg": gf["park_hr_pg"], "park_r_pg": gf["park_r_pg"],
            "park_h_pg": gf["park_h_pg"], "park_2b_pg": gf["park_2b_pg"],
            "park_tb_pg": gf["park_tb_pg"],
            "LF": gf["LF"], "CF": gf["CF"],
            "RF": gf["RF"], "Elevation_ft": gf["Elevation_ft"],
            "Temp": gf["Temp"], "WindSpeed": gf["WindSpeed"],
            "hum_eff": gf["hum_eff"], "air_dens": gf["air_dens"],
            "lg_r_pa": gf["lg_r_pa"], "lg_hr_pa": gf["lg_hr_pa"],
            "DayNight": gf["DayNight"], "Condition": gf["Condition"],
            "WindDir": gf["WindDir"],
            "y_runs": pd.to_numeric(gf[score], errors="coerce"),
        })
        add_wind_carry(d)      # general out/in carry wind (wind_carry)
        add_team_game_derived(d)   # air x lineup air profile (#30)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def team_game_feature_cols():
    # NOTE: the lg_* environment columns were benched here (iteration 4, small
    # MAE cost); UNBENCHED 2026-07-10 (superset dev — selection + 2026 confirm
    # decide at ship). wind_carry = general out/in carry wind, new this batch.
    return ["Season", "month", "Home", "off_hr_pa", "off_r_pg", "off_k_pct",
            "off_loc_hr_pa", "off_loc_r_pg", "off_xwcon", "off_brl_con",
            "opp_pen_era", "opp_pen_hl_era", "opp_pen_np_l3", "opp_def_uer",
            "opp_def_oaa", "opp_pen_xwcon",
            "opp_ps_era", "opp_ps_k_bf", "opp_ps_hr_bf", "opp_ps_h_bf",
            "opp_ps_xwcon", "opp_ps_xwcon_d", "opp_ps_brl_d", "opp_ps_gb_d",
            "opp_pc_era", "opp_pc_hr_bf", "park_hr_pg",
            # multi-dim park factors: the as-of run environment of the venue,
            # which the lone HR factor + static dimensions can't carry
            "park_r_pg", "park_h_pg", "park_2b_pg", "park_tb_pg",
            "LF", "CF", "RF",
            "Elevation_ft", "Temp", "WindSpeed", "DayNight", "Condition",
            "WindDir",
            # air density (humidity+pressure scrape, 2026-07-12): the carry
            # physics of the run environment beyond Temp x Elevation
            "hum_eff", "air_dens",
            # full-surface pass (2026-07-12): posted-lineup defense faced,
            # own lineup baserunning, opposing starter's TTO decay
            "opp_def_p_oaa", "opp_def_p_if", "opp_def_p_of",
            "off_lu_brr", "off_lu_brr_xb", "opp_ps_tto_decay",
            "wind_carry", "lg_r_pa", "lg_hr_pa",
            # 2026-07-14 finish batch: B-lineup quality gap (#19), air x
            # lineup air profile (#30), cross-grain arsenal collision
            # (#32), BaseRuns cluster luck (#20), ump run environment
            # (#21), schedule/travel + doubleheader flags (#18/#24),
            # precip (#11 close-out)
            "off_lu_obp_gap", "off_lu_slg_gap",
            "off_lu_pullair", "air_lu_fly", "off_lu_arswh",
            "off_bsr_luck", "ump_r_g",
            "off_dan", "off_travel_km", "off_tz_delta",
            "is_dh", "dh_game2", "Precip",
            # ---- audit wave (2026-07-14, AUDIT_BUILD_SPECS_0714.md) ----
            # rank 8: the walk channel + patience x wildness collision
            "off_bb_pct", "opp_ps_bb_bf", "patience_wild",
            # rank 9: starter length/leash + the handoff run gap; gray
            # rider: the opposing pen's HL-outs mixture weight
            "opp_ps_len", "opp_st_outs_pg", "xpen_r_gap", "opp_penhl_share",
            # rank 32: venue run-dispersion (also fed to the totals-tail
            # NB pricing; near-zero for the mean-objective GBM, that's fine)
            "park_vmr",
            # rank 33: the lineup's platoon edge (weak measured value —
            # selection dropping it is informative, per the accept policy)
            "off_lu_vsh_edge", "off_lu_vsh_kedge"]


# --------------------------------------------------------------- inference


class _LazyGroups:
    """Lazy per-player history lookup; avoids materializing thousands of
    per-player DataFrames at startup."""

    def __init__(self, df, key):
        # Date alone under-sorts doubleheaders: the default (unstable) sort
        # left same-day rows in arbitrary order, so rolling windows could
        # disagree with the training frames' canonical
        # (PlayerId, Date, GamePk) order — latent until the 2015 backfill
        # resized the array and flipped a tie (selftest, 2026-07-09).
        by = ["Date", "GamePk"] if "GamePk" in df.columns else ["Date"]
        self._g = df.sort_values(by, kind="stable").groupby(key, sort=False)

    def get(self, key):
        try:
            return self._g.get_group(key)
        except KeyError:
            return None


class Stores:
    """Pre-indexed history for fast as-of feature computation at predict time."""

    def __init__(self, raw=None, progress=None):
        tick = progress or (lambda msg: None)
        tick("reading CSVs...")
        self.raw = raw or load_raw()
        r = self.raw
        tick("indexing player history...")
        self.gb_by_player = _LazyGroups(r["gb"], "PlayerId")
        self.starts_by_player = _LazyGroups(r["gp"][r["gp"]["GS"] == 1], "PlayerId")
        hrb = r["hr"].dropna(subset=["BatterId"]).copy()
        hrb["BatterId"] = hrb["BatterId"].astype("int64")
        self.hr_by_batter = _LazyGroups(hrb, "BatterId")
        self.bip_by_batter = self.bip_by_pitcher = None
        self.bip_off_tab = self.bip_pen_tab = None
        if r.get("bip") is not None:
            bip = r["bip"]
            bb = bip.dropna(subset=["BatterId"]).copy()
            bb["BatterId"] = bb["BatterId"].astype("int64")
            self.bip_by_batter = _LazyGroups(bb, "BatterId")
            pb = bip.dropna(subset=["PitcherId"]).copy()
            pb["PitcherId"] = pb["PitcherId"].astype("int64")
            self.bip_by_pitcher = _LazyGroups(pb, "PitcherId")
            self.bip_off_tab, self.bip_pen_tab = _bip_team_tables(
                bip, r["gb"], r["gp"])
        self.pd_pitcher_hist = (_LazyGroups(r["pdp"], "PlayerId")
                                if r.get("pdp") is not None else None)
        self.pd_batter_hist = (_LazyGroups(r["pdb"], "PlayerId")
                               if r.get("pdb") is not None else None)
        self.sprint_prior = (
            r["sprint"].drop_duplicates(["PlayerId", "Year"], keep="last")
            .set_index(["PlayerId", "Year"])
            if r.get("sprint") is not None else None)
        # MiLB translated priors: serve rows are keyed by the season they
        # serve (exact-season lookup, not _prior_val's Year+lag)
        self.milb = ({k: _milb_cols(r["milb"][k],
                                    "milb_" if k == "bat" else "pmilb_")
                      .set_index(["PlayerId", "Season"])
                      for k in ("bat", "pit")}
                     if r.get("milb") is not None else None)
        self.bat_track_prior = (
            r["bat_track"].drop_duplicates(["PlayerId", "Year"], keep="last")
            .set_index(["PlayerId", "Year"])
            if r.get("bat_track") is not None else None)
        self.oaa_prior = (
            r["oaa"].drop_duplicates(["Team", "Year"], keep="last")
            .set_index(["Team", "Year"])
            if r.get("oaa") is not None else None)
        # team battery quality (2026-07-15), prior-season like team OAA
        self.cat_prior = (
            r["cat"].drop_duplicates(["Team", "Year"], keep="last")
            .set_index(["Team", "Year"])
            if r.get("cat") is not None else None)
        # paired IL stints per player, activation-sorted (2026-07-15)
        self.il_by_player = (
            {pid: g for pid, g in _il_stint_table(r["il"]).groupby(
                "PlayerId", sort=False)}
            if r.get("il") is not None else None)
        self.oaa_p_prior = (
            r["oaa_players"].drop_duplicates(["PlayerId", "Year"], keep="last")
            .set_index(["PlayerId", "Year"])
            if r.get("oaa_players") is not None else None)
        self.brr_prior = None
        if r.get("baserun") is not None:
            br = r["baserun"].copy()
            br["bat_brr"] = pd.to_numeric(br["RunnerRuns"], errors="coerce")
            br["bat_brr_xb"] = (
                pd.to_numeric(br["RunnerRunsXB"], errors="coerce")
                / pd.to_numeric(br["Opportunities"], errors="coerce"))
            self.brr_prior = (br.drop_duplicates(["PlayerId", "Year"],
                                                 keep="last")
                              .set_index(["PlayerId", "Year"]))
        # HP-umpire history: per-game K/BB/BF totals grouped by ump for
        # as-of tendency lookups (Stores.ump_feats)
        self.ump_hist = None
        if r.get("umps") is not None:
            uh = r["umps"].merge(_ump_game_totals(r["gp"]), on="GamePk",
                                 how="left")
            uh["HpUmpId"] = pd.to_numeric(uh["HpUmpId"], errors="coerce")
            uh = uh.dropna(subset=["HpUmpId"]).copy()
            uh["HpUmpId"] = uh["HpUmpId"].astype("int64")
            for s in ("g_SO", "g_BB", "g_BF", "g_R"):
                uh[s] = pd.to_numeric(uh[s], errors="coerce").fillna(0.0)
            self.ump_hist = _LazyGroups(uh, "HpUmpId")
        tick("building team/park tables...")
        self.team_tab = _team_offense_table(r["gb"])
        self.team_loc_tab = _team_offense_loc_table(r["gb"])
        self.pen_tab = _bullpen_table(r["gp"])
        self.pen_hl_tab = _bullpen_hl_table(r["gp"])
        self._pen_fat = _pen_fatigue_table(r["gp"]).set_index(
            ["Team", "Date"])["pen_np_l3"]
        self._pen_fat_max = self._pen_fat.index.get_level_values("Date").max()
        self.def_tab = _team_defense_table(r["gp"])
        # 2026-07-14 finish batch: manager's-leash table (#16), trailing
        # BaseRuns luck (#20), per-team schedule history + park coords for
        # travel/day-night (#18), all mirroring their vectorized tables
        self.lea_tab = _team_starter_outs_table(r["gp"])
        self._bsr = _team_bsr_table(r["gb"]).set_index(
            ["Team", "Date"])["bsr_luck"]
        self._bsr_max = self._bsr.index.get_level_values("Date").max()
        # audit wave: opponent HL-arm availability (#22, share unavailable
        # tonight), the prior-season league SB/27 regime centering (ranks
        # 4/24), and per-(pitcher, season) arsenal breadth/trajectory (#22)
        self._pen_unavail = _pen_unavail_table(r["gp"]).set_index(
            ["Team", "Date"])["pen_hl_unavail"]
        self._pen_unavail_max = (self._pen_unavail.index
                                 .get_level_values("Date").max())
        self.lg_sb27 = _league_sb27_table(r["pit_season"]).set_index(
            "Season")["lg_sb27_prior"]
        self.ars_dyn = _arsenal_dynamics(r["ars_p"]).set_index(
            ["PlayerId", "Season"])
        rows = []
        for tc in ("AwayTeam", "HomeTeam"):
            rows.append(r["games"][["GamePk", "Date", "Venue", "DayNight"]]
                        .assign(Team=r["games"][tc]))
        self.sched_hist = _LazyGroups(
            pd.concat(rows, ignore_index=True), "Team")
        self.park_coords = {v: (row["Lat"], row["Lon"])
                            for v, row in r["parks"].set_index("Ballpark")
                            [["Lat", "Lon"]].iterrows()}
        self.tsb_tab = _team_sb_allowed_table(r["gb"])
        self.park_tab = _park_table(r["gb"], r["games"])
        self.phh_tab = _park_hand_hr_table(r["gb"], r["games"], r["hands"])
        self.env_tab = _league_env_table(r["gb"])
        self.slot_pa_tab = _slot_pa_table(r["gb"])
        self.res_rows = _team_results_rows(r["games"])
        self.res_tab = _daily_cum(self.res_rows.drop(columns=["GamePk"]),
                                  ["Team", "Season"], ["W", "RF", "RA"])
        _, self.elo_hist = build_elo(r["games"])
        self.hrq = _hr_quality_table(r["hr"], r["parks"])
        self.phrq = _pitcher_hr_allowed_table(r["hr"], r["gp"], r["parks"])
        self.bat_prior = _batter_season_table(r["bat_season"]).set_index(
            ["PlayerId", "Year"])
        self.pit_prior = _pitcher_season_table(r["pit_season"]).set_index(
            ["PlayerId", "Year"])
        self.rosters = r["rosters"].set_index("PlayerId")
        self.parks = r["parks"].set_index("Ballpark")
        h = r["hands"].replace("", np.nan)
        self.bats = h.set_index("PlayerId")["Bats"].to_dict()
        self.throws = h.set_index("PlayerId")["Throws"].to_dict()

    # -- entity helpers -----------------------------------------------

    def _cum(self, table, keys, date):
        m = table
        for k, v in keys.items():
            m = m[m[k] == v]
        m = m[m["Date"] < date]
        return m.iloc[-1] if len(m) else None

    def _vsh(self, h, opp_hand):
        """Platoon-split features vs today's starter hand (matches the
        vectorized _vsL/_vsR-then-select computation exactly)."""
        if opp_hand not in ("L", "R"):
            return {"vsh_PA": np.nan, "vsh_hr_pa_sh": np.nan,
                    "vsh_tb_ab_sh": np.nan, "vsh_k_pct_sh": np.nan}
        hh = h[h["opp_hand"] == opp_hand] if h is not None and len(h) else None
        s = (hh[VSH_STATS].sum() if hh is not None
             else {k: 0 for k in VSH_STATS})
        return {"vsh_PA": s["PA"],
                "vsh_hr_pa_sh": _shrink(s["HR"], s["PA"], "hr_pa"),
                "vsh_tb_ab_sh": _shrink(s["TB"], s["PA"], "tb_ab"),
                "vsh_k_pct_sh": _shrink(s["SO"], s["PA"], "k_pct")}

    def _vloc(self, h, home):
        """Career splits in today's venue context (home/road), matching the
        vectorized _loc0/_loc1-then-select computation exactly."""
        if home not in (0, 1):
            return {"vloc_PA": np.nan, "vloc_hr_pa_sh": np.nan,
                    "vloc_h_pa_sh": np.nan, "vloc_tb_ab_sh": np.nan,
                    "vloc_k_pct_sh": np.nan}
        hh = h[h["Home"] == home] if h is not None and len(h) else None
        s = (hh[LOC_STATS].sum() if hh is not None
             else {k: 0 for k in LOC_STATS})
        return {"vloc_PA": s["PA"],
                "vloc_hr_pa_sh": _shrink(s["HR"], s["PA"], "hr_pa"),
                "vloc_h_pa_sh": _shrink(s["H"], s["PA"], "h_ab"),
                "vloc_tb_ab_sh": _shrink(s["TB"], s["PA"], "tb_ab"),
                "vloc_k_pct_sh": _shrink(s["SO"], s["PA"], "k_pct")}

    def xpa_slot(self, slot, date):
        """As-of league PA per game at a lineup slot: the shared
        _slot_pa_table's inclusive cumsums through the last game-day
        strictly before `date` — identical to the training merge, where a
        row's value excludes its own day."""
        t = self.slot_pa_tab
        m = t[(t["slot"] == slot) & (t["Date"] < date)]
        if not len(m):
            return np.nan
        last = m.iloc[-1]
        return float(last["cum_pa"] / last["cum_n"])

    def _prior_val(self, tab, pid, season, col):
        """Prior-season lookup with the same y-1-then-y-2 fallback as
        _merge_prior_season."""
        for lag in (1, 2):
            try:
                v = tab.loc[(pid, season - lag), col]
            except KeyError:
                continue
            if pd.notna(v):
                return float(v)
        return np.nan

    def batter_feats(self, pid, date, season, opp_hand=None, home=None):
        hist = self.gb_by_player.get(pid)
        out = {}
        if hist is not None:
            h = hist[hist["Date"] < date]
        else:
            h = None
        out.update(self._vsh(h, opp_hand))
        out.update(self._vloc(h, home))
        zero = {k: 0 for k in BAT_STATS}
        if h is None or h.empty:
            for pre in ["c", "s"]:
                for k in ["PA", "TB", "hr_pa", "tb_ab", "h_ab", "k_pct",
                          "bb_pct", "iso", "xbh_ab", "obp"]:
                    out[f"{pre}_{k}"] = np.nan
                out.update(shrunk_from_sums(zero, pre))  # 0 sums -> priors
            out.update(g_career=0, g_season=0, days_rest=np.nan,
                       c_ibb_pa=np.nan, pos_c_share=np.nan,
                       pos_dh_share=np.nan, _obpp=np.nan, _slgp=np.nan,
                       _obpp_d=np.nan, _slgp_d=np.nan)
            out["c_sb_succ"] = SB_SUCC_PRIOR  # shrink of zero sums
            # audit wave rank 4: reach-first / attempt priors on zero sums
            out["ob1_sh"] = OB1_PRIOR
            out["sb_att1"] = SB_ATT_PRIOR
            for w in ROLL_WINDOWS:
                for k in ["PA", "hr_pa", "tb_ab", "k_pct"]:
                    out[f"r{w}_{k}"] = np.nan
                out.update(shrunk_from_sums(zero, f"r{w}", roll=True))
            out.update(decayed_feats({s: 0.0 for s in DECAY_STATS}))
            out["_dk_AB"] = out["_dk_H"] = out["_dk_SO"] = 0.0   # hit_luck inputs
            out["_dk_RBI"] = out["_dk_LOB"] = out["_dk_PA"] = 0.0   # d_rob_pa
            out["xpa_bat"] = XPA_PRIOR      # zero decayed sums -> the prior
            _zero_thr = {k: 0.0 for k in THRESH_HIST}
            for pre in ("c", "s", "d"):     # zero sums -> the league priors
                out.update(hrr_hist_feats(_zero_thr, 0.0, pre))
        else:
            hs = h[h["Season"] == season]
            for pre, frame in [("c", h), ("s", hs)]:
                sums = frame[BAT_STATS].sum()
                out[f"{pre}_PA"] = sums["PA"]
                out[f"{pre}_TB"] = sums["TB"]   # rank 33 vsh-edge baseline
                out[f"{pre}_hr_pa"] = sums["HR"] / sums["PA"] if sums["PA"] else np.nan
                out[f"{pre}_tb_ab"] = sums["TB"] / sums["AB"] if sums["AB"] else np.nan
                out[f"{pre}_h_ab"] = sums["H"] / sums["AB"] if sums["AB"] else np.nan
                out[f"{pre}_k_pct"] = sums["SO"] / sums["PA"] if sums["PA"] else np.nan
                out[f"{pre}_bb_pct"] = sums["BB"] / sums["PA"] if sums["PA"] else np.nan
                out[f"{pre}_iso"] = ((sums["TB"] - sums["H"]) / sums["AB"]
                                     if sums["AB"] else np.nan)
                out[f"{pre}_xbh_ab"] = ((sums["2B"] + sums["3B"] + sums["HR"])
                                        / sums["AB"] if sums["AB"] else np.nan)
                out[f"{pre}_obp"] = ((sums["H"] + sums["BB"] + sums["HBP"])
                                     / sums["PA"] if sums["PA"] else np.nan)
                out.update(shrunk_from_sums(sums, pre))
                out.update(hrr_hist_feats(sums, len(frame), pre))
                if pre == "c":
                    out["c_ibb_pa"] = (sums["IBB"] / sums["PA"]
                                       if sums["PA"] else np.nan)
                    out["c_sb_succ"] = ((sums["SB"] + SB_SUCC_K * SB_SUCC_PRIOR)
                                        / (sums["SB"] + sums["CS"] + SB_SUCC_K))
                    # audit wave rank 4: reach-first rate + attempt/opp
                    _ob1 = (sums["H"] - sums["2B"] - sums["3B"] - sums["HR"]
                            + sums["BB"] + sums["HBP"])
                    out["ob1_sh"] = ((_ob1 + OB1_K * OB1_PRIOR)
                                     / (sums["PA"] + OB1_K))
                    out["sb_att1"] = ((sums["SB"] + sums["CS"]
                                       + SB_ATT_K * SB_ATT_PRIOR)
                                      / (_ob1 + SB_ATT_K))
                    out["_obpp"] = ((sums["H"] + sums["BB"] + sums["HBP"])
                                    / sums["PA"] if sums["PA"] else np.nan)
                    out["_slgp"] = (sums["TB"] / sums["AB"]
                                    if sums["AB"] else np.nan)
            out["g_career"] = len(h)
            out["g_season"] = len(hs)
            out["days_rest"] = (date - h["Date"].iloc[-1]).days
            out["pos_c_share"] = float((h["Position"] == "C").sum()) / len(h)
            out["pos_dh_share"] = float((h["Position"] == "DH").sum()) / len(h)
            for w in ROLL_WINDOWS:
                tail = h.tail(w)[["PA", "HR", "TB", "SO", "SB",
                                  "R", "RBI"]].sum()
                out[f"r{w}_PA"] = tail["PA"]
                out[f"r{w}_hr_pa"] = tail["HR"] / tail["PA"] if tail["PA"] else np.nan
                out[f"r{w}_tb_ab"] = tail["TB"] / tail["PA"] if tail["PA"] else np.nan
                out[f"r{w}_k_pct"] = tail["SO"] / tail["PA"] if tail["PA"] else np.nan
                out.update(shrunk_from_sums(tail, f"r{w}", roll=True))
            # decay-weighted sums: exp(-lam * days_ago), matching the
            # vectorized exp-cumsum-then-discount computation exactly
            wd = np.exp(-DECAY_LAM
                        * (date - h["Date"]).dt.days.to_numpy(dtype="float64"))
            dks = {s: float((h[s].to_numpy(dtype="float64") * wd).sum())
                   for s in DECAY_STATS}
            out.update(decayed_feats(dks))
            out["_dk_AB"], out["_dk_H"], out["_dk_SO"] = (   # hit_luck inputs
                dks["AB"], dks["H"], dks["SO"])
            out["_dk_RBI"], out["_dk_LOB"], out["_dk_PA"] = (   # d_rob_pa
                dks["RBI"], dks["LOB"], dks["PA"])
            # decayed PA per game (exposure), same shrink as the frame
            out["xpa_bat"] = ((dks["PA"] + XPA_K * XPA_PRIOR)
                              / (float(wd.sum()) + XPA_K))
            # decayed H+R+RBI threshold shares (denominator = decayed game
            # count, matching the frame's _dk_G)
            out.update(hrr_hist_feats(dks, float(wd.sum()), "d"))
            # decayed own OBP/SLG: teammate-context inputs (predict.py
            # assembles ctx_*_d from the lineup's values)
            out["_obpp_d"] = ((dks["H"] + dks["BB"] + dks["HBP"]) / dks["PA"]
                              if dks["PA"] else np.nan)
            out["_slgp_d"] = dks["TB"] / dks["AB"] if dks["AB"] else np.nan
        out["bat_goao"] = self._prior_val(self.bat_prior, pid, season, "bat_goao")
        out["bat_sprint"] = (
            self._prior_val(self.sprint_prior, pid, season, "SprintSpeed")
            if self.sprint_prior is not None else np.nan)
        out["bat_hp1b"] = (
            self._prior_val(self.sprint_prior, pid, season, "HPto1B")
            if self.sprint_prior is not None else np.nan)
        # prior-season bat tracking (BatSpeed etc.) -> bt_* (parity with the
        # frame's BAT_TRACK_REN); NaN when unavailable / uncovered season
        for raw_c, feat in BAT_TRACK_REN.items():
            out[feat] = (self._prior_val(self.bat_track_prior, pid, season,
                                         raw_c)
                         if self.bat_track_prior is not None else np.nan)
        # MiLB translated prior (exact-season serve-table lookup)
        out.update(self.milb_feats(pid, season, "bat"))

        q = self.hrq[(self.hrq["BatterId"] == pid) & (self.hrq["Date"] < date)]
        if len(q):
            q = q.iloc[-1]
            out["hrq_n"] = q["cum_n"]
            out["hrq_ev_avg"] = q["cum_ev"] / q["cum_n"]
            out["hrq_dist_avg"] = q["cum_dist"] / q["cum_n"]
            out["hrq_dist_max"] = q["cum_dist_max"]
            out["hrq_angle_avg"] = q["cum_angle"] / q["cum_n"]
        else:
            out.update(hrq_n=np.nan, hrq_ev_avg=np.nan, hrq_dist_avg=np.nan,
                       hrq_dist_max=np.nan, hrq_angle_avg=np.nan)
        return out

    def _bip_entity(self, groups, pid, date, tag):
        """Contact-quality features for one batter or pitcher, as-of `date`
        (career shrunk rates + 90-day-decayed versions), mirroring the
        vectorized _bip_table math exactly."""
        nan = {f"{tag}_{n}": np.nan for n in BIP_SHRINK}
        nan.update({f"{tag}d_{n}": np.nan for n in BIP_DECAYED})
        nan.update({f"{tag}_n": np.nan, f"{tag}d_n": np.nan})
        if groups is None:
            return nan
        hist = groups.get(pid)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return nan
        ev, la = h["ExitVelo"], h["LaunchAngle"]
        pull, pullair, oppo, oppoair = _spray_flags(h)
        parts = {
            "n": np.ones(len(h)),
            "ev_n": ev.notna().to_numpy(dtype="float64"),
            "ev_sum": ev.fillna(0.0).to_numpy(dtype="float64"),
            "hh_n": (ev >= 95).to_numpy(dtype="float64"),
            "brl_n": (h["LSA"] == 6).to_numpy(dtype="float64"),
            "la_n": la.notna().to_numpy(dtype="float64"),
            "la_sum": la.fillna(0.0).to_numpy(dtype="float64"),
            "xba_n": h["xBA"].notna().to_numpy(dtype="float64"),
            "xba_sum": h["xBA"].fillna(0.0).to_numpy(dtype="float64"),
            "xw_n": h["xwOBA"].notna().to_numpy(dtype="float64"),
            "xw_sum": h["xwOBA"].fillna(0.0).to_numpy(dtype="float64"),
            "gb_n": (h["BBType"] == "ground_ball").to_numpy(dtype="float64"),
            "ld_n": (h["BBType"] == "line_drive").to_numpy(dtype="float64"),
            "pu_n": (h["BBType"] == "popup").to_numpy(dtype="float64"),
            "fld_n": ((h["BBType"] == "fly_ball")
                      & h["DistAdj"].notna()).to_numpy(dtype="float64"),
            "fld_sum": (h["DistAdj"].where(h["BBType"] == "fly_ball")
                        .fillna(0.0).to_numpy(dtype="float64")),
            "hc_n": h["HcX"].notna().to_numpy(dtype="float64"),
            "pull_n": pull.astype(float),
            "pullair_n": pullair.astype(float),
            # audit wave (rank 13): oppo spray bins + the 105+ EV tail
            "oppo_n": oppo.astype(float),
            "oppoair_n": oppoair.astype(float),
            "ev105_n": (ev >= 105).to_numpy(dtype="float64"),
        }
        cs = {k: float(v.sum()) for k, v in parts.items()}
        wd = np.exp(-DECAY_LAM
                    * (date - h["Date"]).dt.days.to_numpy(dtype="float64"))
        dk = {k: float((v * wd).sum()) for k, v in parts.items()}
        out = bip_feats(lambda c: cs[c], tag)
        out.update(bip_feats(lambda c: dk[c], f"{tag}d", BIP_DECAYED))
        out[f"{tag}_n"] = cs["n"]
        out[f"{tag}d_n"] = dk["n"]
        return out

    def bip_batter(self, pid, date):
        return self._bip_entity(self.bip_by_batter, pid, date, "bip")

    def bip_pitcher(self, pid, date):
        return self._bip_entity(self.bip_by_pitcher, pid, date, "pbip")

    def fatigue(self, pid, date):
        """Games in the trailing 7 / 14 calendar days strictly before `date`
        (day-start convention), mirroring the vectorized _prior_games: h is
        already Date<date, so day-diff is >=1 (same-day excluded) and <=win."""
        hist = self.gb_by_player.get(pid)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return {"g_l7d": 0.0, "g_l14d": 0.0}
        dd = (date - h["Date"]).dt.days
        return {"g_l7d": float((dd <= 7).sum()),
                "g_l14d": float((dd <= 14).sum())}

    def bvp(self, pid, pitcher_id, date):
        """Batter-vs-pitcher as-of pairwise contact sums (bvp_cum_*), mirroring
        _bvp_table: the batter's BIP filtered to this PitcherId, strictly before
        `date`. add_batter_derived turns these into the shrunk residuals."""
        if self.bip_by_batter is None or pitcher_id is None or pd.isna(pitcher_id):
            return {"bvp_cum_n": np.nan, "bvp_cum_xw_n": np.nan,
                    "bvp_cum_xw_sum": np.nan, "bvp_cum_hr_n": np.nan}
        hist = self.bip_by_batter.get(pid)
        h = (hist[(hist["Date"] < date) & (hist["PitcherId"] == pitcher_id)]
             if hist is not None else None)
        if h is None or h.empty:
            return {"bvp_cum_n": 0.0, "bvp_cum_xw_n": 0.0,
                    "bvp_cum_xw_sum": 0.0, "bvp_cum_hr_n": 0.0}
        return {"bvp_cum_n": float(len(h)),
                "bvp_cum_xw_n": float(h["xwOBA"].notna().sum()),
                "bvp_cum_xw_sum": float(h["xwOBA"].fillna(0.0).sum()),
                "bvp_cum_hr_n": float((h["Events"] == "home_run").sum())}

    def tto(self, pid, date):
        """Starter times-through-order contact splits as-of (tto_cum_*),
        mirroring _tto_table: ranks of his contact-PAs within each prior
        game. The shared tto_decay_from_sums turns these into p_tto_decay
        (0 with no history, matching the vectorized fillna)."""
        cols = ("tto_cum_xw1_n", "tto_cum_xw1_sum",
                "tto_cum_xw3_n", "tto_cum_xw3_sum")
        if self.bip_by_pitcher is None or pid is None or pd.isna(pid):
            return {c: np.nan for c in cols}
        hist = self.bip_by_pitcher.get(pid)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return {c: 0.0 for c in cols}
        rank = h.groupby("GamePk")["AtBat"].rank(method="dense")
        tto = np.ceil(rank / TTO_CONTACT_PER_ORDER)
        first, third = (tto <= 1).to_numpy(), (tto >= 3).to_numpy()
        xw_ok = h["xwOBA"].notna().to_numpy()
        xw = h["xwOBA"].fillna(0.0).to_numpy()
        return {"tto_cum_xw1_n": float((first & xw_ok).sum()),
                "tto_cum_xw1_sum": float(xw[first].sum()),
                "tto_cum_xw3_n": float((third & xw_ok).sum()),
                "tto_cum_xw3_sum": float(xw[third].sum())}

    def lineup_oaa(self, pids, season, prefix="opp"):
        """Mean prior-season player OAA of a posted lineup (overall + IF/OF
        splits), mirroring _lineup_oaa_table: y-1 with y-2 fallback per
        player, position classified by the OAA file's primary position,
        NaN-skipping means. Players without a row (DH-only, rookies,
        catchers) simply drop out, exactly like the vectorized merge."""
        nan = {f"{prefix}_def_p_oaa": np.nan, f"{prefix}_def_p_if": np.nan,
               f"{prefix}_def_p_of": np.nan}
        if self.oaa_p_prior is None or not pids:
            return nan
        alls, ifs, ofs = [], [], []
        for pid in pids:
            for lag in (1, 2):
                try:
                    row = self.oaa_p_prior.loc[(pid, season - lag)]
                except KeyError:
                    continue
                v = float(row["OAA"])
                alls.append(v)
                if row["Pos"] in DEF_IF_POS:
                    ifs.append(v)
                elif row["Pos"] in DEF_OF_POS:
                    ofs.append(v)
                break

        def mean(v):
            return float(np.mean(v)) if v else np.nan

        return {f"{prefix}_def_p_oaa": mean(alls),
                f"{prefix}_def_p_if": mean(ifs),
                f"{prefix}_def_p_of": mean(ofs)}

    def batter_brr(self, pid, season):
        """Prior-season baserunning run value (total + extra-base rate),
        mirroring the vectorized _merge_prior_season of mlb_baserunning.csv."""
        if self.brr_prior is None:
            return {"bat_brr": np.nan, "bat_brr_xb": np.nan}
        return {"bat_brr": self._prior_val(self.brr_prior, pid, season,
                                           "bat_brr"),
                "bat_brr_xb": self._prior_val(self.brr_prior, pid, season,
                                              "bat_brr_xb")}

    def lineup_brr(self, pids, season):
        """Mean prior-season baserunning of a posted lineup, mirroring
        _lineup_brr_table (NaN-skipping mean; players without a prior-season
        row simply drop out, exactly like the vectorized merge)."""
        if self.brr_prior is None or not pids:
            return {"off_lu_brr": np.nan, "off_lu_brr_xb": np.nan}
        vals = [self.batter_brr(p, season) for p in pids]
        a = [v["bat_brr"] for v in vals if not pd.isna(v["bat_brr"])]
        b = [v["bat_brr_xb"] for v in vals if not pd.isna(v["bat_brr_xb"])]
        return {"off_lu_brr": float(np.mean(a)) if a else np.nan,
                "off_lu_brr_xb": float(np.mean(b)) if b else np.nan}

    def park_hand_hr(self, venue, date):
        """As-of venue HR/PA split by batter hand (phh_*), mirroring the
        vectorized _park_hand_hr_table + _asof_merge by Venue (exclusive).
        add_batter_derived turns these into park_hand_hr_edge off eff_hand."""
        nan = {"phh_L_HR": np.nan, "phh_L_PA": np.nan,
               "phh_R_HR": np.nan, "phh_R_PA": np.nan}
        t = self.phh_tab
        m = t[(t["Venue"] == venue) & (t["Date"] < date)]
        if not len(m):
            return nan
        last = m.iloc[-1]
        return {k: float(last[k]) for k in
                ("phh_L_HR", "phh_L_PA", "phh_R_HR", "phh_R_PA")}

    def _bip_hand_entity(self, groups, pid, date, hand, hand_col, tag):
        """Hand-split contact quality vs one hand (career shrunk xwOBA-on-
        contact and barrel share), mirroring _bip_hand_table exactly."""
        nan = {f"{tag}_{n}": np.nan for n in BVH_METRICS}
        nan[f"{tag}_n"] = np.nan
        if groups is None or hand not in ("L", "R"):
            return nan
        hist = groups.get(pid)
        h = (hist[(hist["Date"] < date) & (hist[hand_col] == hand)]
             if hist is not None else None)
        if h is None or h.empty:
            return nan
        cs = {"n": float(len(h)),
              "ev_n": float(h["ExitVelo"].notna().sum()),
              "brl_n": float((h["LSA"] == 6).sum()),
              "xw_n": float(h["xwOBA"].notna().sum()),
              "xw_sum": float(h["xwOBA"].fillna(0.0).sum())}
        out = bip_feats(lambda c: cs[c], tag, BVH_METRICS)
        out[f"{tag}_n"] = cs["n"]
        return out

    def bip_batter_vs_hand(self, pid, date, opp_hand):
        """Batter's career contact quality vs pitchers of `opp_hand`."""
        return self._bip_hand_entity(self.bip_by_batter, pid, date,
                                     opp_hand, "PThrows", "bvh")

    def bip_pitcher_vs_hand(self, pid, date, bat_side):
        """Pitcher's career contact allowed to batters of `bat_side`."""
        return self._bip_hand_entity(self.bip_by_pitcher, pid, date,
                                     bat_side, "Stand", "pvh")

    def team_sb_allowed(self, team, season, date):
        """Battery SB control: steals allowed per game and shrunk
        caught-stealing rate by `team`, season-to-date as-of."""
        row = self._cum(self.tsb_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {"tsb_sb_g": np.nan, "tsb_stop": np.nan}
        return {"tsb_sb_g": row["cum_SB"] / row["cum_G"],
                "tsb_stop": ((row["cum_CS"] + SB_SUCC_K * TSB_STOP_PRIOR)
                             / (row["cum_SB"] + row["cum_CS"] + SB_SUCC_K))}

    def team_bip_offense(self, team, season, date):
        """Team offense contact quality (xwOBA-on-contact, barrel share),
        season-cumulative as-of."""
        row = (None if self.bip_off_tab is None else
               self._cum(self.bip_off_tab, {"Team": team, "Season": season},
                         date))
        if row is None:
            return {"off_xwcon": np.nan, "off_brl_con": np.nan}
        return {"off_xwcon": (row["cum_xw_sum"] / row["cum_xw_n"]
                              if row["cum_xw_n"] else np.nan),
                "off_brl_con": (row["cum_brl_n"] / row["cum_ev_n"]
                                if row["cum_ev_n"] else np.nan)}

    def team_bip_pen(self, team, season, date):
        """Bullpen contact quality ALLOWED (xwOBA-on-contact), as-of."""
        row = (None if self.bip_pen_tab is None else
               self._cum(self.bip_pen_tab, {"Team": team, "Season": season},
                         date))
        if row is None:
            return {"pen_xwcon": np.nan}
        return {"pen_xwcon": (row["cum_xw_sum"] / row["cum_xw_n"]
                              if row["cum_xw_n"] else np.nan)}

    def team_oaa(self, team, season):
        """Opponent team defense: prior-season outs above average per 162."""
        if self.oaa_prior is None:
            return np.nan
        return self._prior_val(self.oaa_prior, team, season, "OAA_per162")

    def catcher_feats(self, team, season, prefix):
        """Prior-season team battery quality (2026-07-15): framing runs
        per 2000 called pitches, CS-above-average per attempt, pop time.
        prefix = "opp" (batter rows) or "own" (starter rows)."""
        src = {"FrameRV_pt": f"{prefix}_cat_frame",
               "CSAA_att": f"{prefix}_cat_csaa",
               "PopTime": f"{prefix}_cat_pop",
               "PopC": f"{prefix}_cat_popc"}
        if self.cat_prior is None:
            return {v: np.nan for v in src.values()}
        return {v: self._prior_val(self.cat_prior, team, season, k)
                for k, v in src.items()}

    def il_feats(self, pid, date, pre=""):
        """IL return context (2026-07-15) — mirror of _il_asof. An unknown
        starter (sentinel/None) stays NaN across the board; a known player
        with no completed stint gets the informative 0.0 flags."""
        cols = [f"{pre}{c}" for c in IL_FEATS]
        if self.il_by_player is None or not pid or pid == -1:
            return {c: np.nan for c in cols}
        g = self.il_by_player.get(pid)
        h = g[g["ActDate"] <= date] if g is not None else None
        if h is None or h.empty:
            return il_feats_from_stint(date, pd.NaT, np.nan, np.nan,
                                       np.nan, pre)
        last = h.iloc[-1]
        return il_feats_from_stint(date, last["ActDate"],
                                   last["StintDays"], last["Rehab"],
                                   last["_szn_days"], pre)

    def milb_feats(self, pid, season, kind):
        """MiLB translated-prior features for one batter ("bat") or
        starter ("pit"), exact-season lookup — parity with the frames'
        exact-season merge."""
        cols = MILB_BAT_COLS if kind == "bat" else MILB_PIT_COLS
        if self.milb is not None:
            t = self.milb[kind]
            try:
                row = t.loc[(int(pid), int(season))]
                return {c: float(row[c]) for c in cols}
            except KeyError:
                pass
        return {c: np.nan for c in cols}

    def ump_feats(self, ump_id, date):
        """HP-umpire zone tendency (K%/BB% per batter faced) and run
        environment (shrunk R/G, #21) over his games strictly before
        `date`. Unknown ump / no file -> the league priors (neutral),
        matching the vectorized _ump_asof exactly."""
        so = bb = bf = runs = n = 0.0
        if ump_id is not None and self.ump_hist is not None:
            h = self.ump_hist.get(int(ump_id))
            if h is not None:
                h = h[h["Date"] < date]
                so = float(h["g_SO"].sum())
                bb = float(h["g_BB"].sum())
                bf = float(h["g_BF"].sum())
                runs = float(h["g_R"].sum())
                n = float(len(h))
        k, w = _ump_shrink(so, bb, bf)
        return {"ump_k_pct": k, "ump_bb_pct": w,
                "ump_r_g": _ump_rg_shrink(runs, n)}

    def _pd_sums(self, groups, pid, date):
        """(career sums, decayed sums) of the pitch-daily rows before
        `date`, or (None, None) without history. Matches _cum_decay_table
        plus the row-date discount exactly."""
        if groups is None:
            return None, None
        hist = groups.get(pid)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return None, None
        cols = [c for c in h.columns if c not in ("PlayerId", "Date")]
        cs = {c: float(h[c].sum()) for c in cols}
        wd = np.exp(-DECAY_LAM
                    * (date - h["Date"]).dt.days.to_numpy(dtype="float64"))
        dk = {c: float((h[c].to_numpy(dtype="float64") * wd).sum())
              for c in cols}
        return cs, dk

    def pd_pitcher_feats(self, pid, date):
        """Starter swing-and-miss form from the pitch-level dailies."""
        names_c, names_d = ("swstr", "fbv"), PD_PITCHER_D
        nan = {f"pd_{n}_c": np.nan for n in names_c}
        nan.update({f"pd_{n}_d": np.nan for n in names_d})
        nan["pd_fbv_tr"] = np.nan
        nan["pd_fbv_sd"] = np.nan
        nan["pd_rel_sd"] = np.nan
        nan["pd_fbv_fade"] = np.nan
        nan["pd_stretch_vdelta_d"] = np.nan
        nan["pd_relsep_d"] = np.nan
        cs, dk = self._pd_sums(self.pd_pitcher_hist, pid, date)
        if cs is None:
            return nan
        out = {}
        for name in names_c:
            prior, k, num, den = PD_SHRINK[name]
            out[f"pd_{name}_c"] = (cs[num] + k * prior) / (cs[den] + k)
        for name in names_d:
            prior, k, num, den = PD_SHRINK[name]
            out[f"pd_{name}_d"] = (dk[num] + k * prior) / (dk[den] + k)
        out["pd_fbv_tr"] = out["pd_fbv_d"] - out["pd_fbv_c"]
        # v5 dispersion reads from the same decayed sums (shared helpers)
        out["pd_fbv_sd"] = float(velo_sd_from_sums(
            dk["fb_n"], dk["fb_v"], dk["fb_v2"]))
        out["pd_rel_sd"] = float(release_scatter_from_sums(
            dk["rp_n"], dk["rp_x"], dk["rp_x2"], dk["rp_z"], dk["rp_z2"]))
        # v6 (#26): decayed weighted-mean per-start velo-fade slope
        out["pd_fbv_fade"] = float(fbv_fade_from_sums(
            dk["fade_w"], dk["fade_num"]))
        # v7 audit wave (rank 27): base-out stretch delta + per-class
        # release separation (effprem/brkmov ride the names_d loop)
        out["pd_stretch_vdelta_d"] = float(stretch_vdelta_from_sums(
            dk["fb_n"], dk["fb_v"], dk["fbstr_n"], dk["fbstr_v"]))
        out["pd_relsep_d"] = float(release_sep_from_sums(
            dk["rpf_n"], dk["rpf_x"], dk["rpf_z"],
            dk["rpb_n"], dk["rpb_x"], dk["rpb_z"]))
        return out

    def pd_batter_feats(self, pid, date):
        """Batter plate discipline (whiff per swing, chase, zone contact,
        elite-velo whiff, pitch-class whiff splits, first-pitch swing),
        career + decay."""
        names = PD_BATTER
        nan = {}
        for name in names:
            nan[f"bd_{name}_c"] = np.nan
            nan[f"bd_{name}_d"] = np.nan
        cs, dk = self._pd_sums(self.pd_batter_hist, pid, date)
        if cs is None:
            return nan
        out = {}
        for name in names:
            prior, k, num, den = PD_SHRINK[name]
            out[f"bd_{name}_c"] = (cs[num] + k * prior) / (cs[den] + k)
            out[f"bd_{name}_d"] = (dk[num] + k * prior) / (dk[den] + k)
        return out

    def hrpt(self, batter_id, pitcher_id, season, date):
        """HR-by-pitch-type matchup score (see hrpt_from_counts)."""
        hist = self.hr_by_batter.get(batter_id)
        h = hist[hist["Date"] < date] if hist is not None else None
        total = len(h) if h is not None else 0
        a = self.raw["ars_p"]
        u = a[(a["PlayerId"] == pitcher_id) & (a["Year"] == season - 1)]
        u = u.dropna(subset=["%"])
        usage = dict(zip(u["Pitch"], u["%"] / 100.0))
        counts = h["Pitch"].value_counts().to_dict() if total else {}
        return hrpt_from_counts(counts, usage, total)

    def pitcher_hr_quality(self, pid, date):
        """HR quality allowed by this pitcher, as-of (from the HR log)."""
        q = self.phrq[(self.phrq["PitcherId"] == pid)
                      & (self.phrq["Date"] < date)]
        if not len(q):
            return {"phrq_n": np.nan, "phrq_ev_avg": np.nan}
        q = q.iloc[-1]
        return {"phrq_n": q["cum_n"], "phrq_ev_avg": q["cum_ev"] / q["cum_n"]}

    def pitcher_prior(self, pid, season):
        """Prior-season GO/AO and stolen-base control for a pitcher."""
        return {c: self._prior_val(self.pit_prior, pid, season, c)
                for c in ["pit_goao", "psb_sb27", "psb_stop"]}

    def starter_feats(self, pid, date, season, prefix="", home=None):
        """`home` (0/1) = the starter's OWN venue context tonight — feeds
        the pvloc_* venue split (#23); None -> shrunk-to-prior at zero
        sums, matching a no-history merge."""
        hist = self.starts_by_player.get(pid)
        out = {}
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            keys = ["p_starts_career", "p_starts_season", "p_days_rest",
                    "p_np_last", "p_np_l3",
                    "pc_BF", "pc_hr_bf", "pc_k_bf", "pc_bb_bf", "pc_h_bf",
                    "pc_era", "pc_strike_pct", "pc_chaos_bf", "pc_hbp_bf",
                    "ps_BF", "ps_hr_bf", "ps_k_bf", "ps_bb_bf", "ps_h_bf",
                    "ps_era", "ps_strike_pct", "ps_chaos_bf", "ps_hbp_bf",
                    "p5_hr_bf", "p5_k_bf", "p5_h_bf", "p5_bb_bf",
                    "p_ip_per_start", "p_outs_sd", "p_short_share",
                    "pc_NP", "pc_Outs"]   # rank 25 economy inputs
            nan = {k: np.nan for k in keys}
            # zero venue-split sums -> the shrinkage priors (vectorized
            # rows with no history land there too)
            nan.update(pvloc_from_sums(0.0, 0.0, 0.0, 0.0, 0.0))
            return nan
        hs = h[h["Season"] == season]
        out["p_starts_career"] = len(h)
        out["p_starts_season"] = len(hs)
        out["p_days_rest"] = (date - h["Date"].iloc[-1]).days
        for pre, frame in [("pc", h), ("ps", hs)]:
            s = frame[PIT_STATS].sum()
            out[f"{pre}_BF"] = s["BF"]
            out[f"{pre}_hr_bf"] = s["HR"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_k_bf"] = s["SO"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_bb_bf"] = s["BB"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_h_bf"] = s["H"] / s["BF"] if s["BF"] else np.nan
            out[f"{pre}_era"] = s["ER"] * 27 / s["Outs"] if s["Outs"] else np.nan
            out[f"{pre}_strike_pct"] = (s["Strikes"] / s["NP"]
                                        if s["NP"] else np.nan)
            # audit wave rank 21: shrunk chaos + HBP channels
            out[f"{pre}_chaos_bf"] = ((s["WP"] + s["BK"]
                                       + CHAOS_K * CHAOS_PRIOR)
                                      / (s["BF"] + CHAOS_K))
            out[f"{pre}_hbp_bf"] = ((s["HBP"] + HBP_K * HBP_PRIOR)
                                    / (s["BF"] + HBP_K))
            if pre == "pc":   # rank 25 pitch-economy inputs (raw sums)
                out["pc_NP"] = s["NP"]
                out["pc_Outs"] = s["Outs"]
        t5 = h.tail(5)[["BF", "HR", "SO", "H", "BB"]].sum()
        out["p5_hr_bf"] = t5["HR"] / t5["BF"] if t5["BF"] else np.nan
        out["p5_k_bf"] = t5["SO"] / t5["BF"] if t5["BF"] else np.nan
        out["p5_h_bf"] = t5["H"] / t5["BF"] if t5["BF"] else np.nan
        out["p5_bb_bf"] = t5["BB"] / t5["BF"] if t5["BF"] else np.nan
        out["p_ip_per_start"] = (hs["Outs"].sum() / 3 / len(hs)) if len(hs) else np.nan
        out["p_np_last"] = h["NP"].iloc[-1]
        out["p_np_l3"] = h.tail(3)["NP"].mean()
        # outing shape (#16): career outing-length SD + short-start share
        # of the last SHORT_SHARE_WIN starts, mirroring _starter_asof
        outs = h["Outs"].to_numpy(dtype="float64")
        out["p_outs_sd"] = float(outs_sd_from_sums(
            len(h), outs.sum(), (outs ** 2).sum()))
        t10 = h.tail(SHORT_SHARE_WIN)["Outs"]
        out["p_short_share"] = (float((t10 <= SHORT_START_OUTS).mean())
                                if len(t10) >= SHORT_SHARE_MIN else np.nan)
        # venue split (#23) for tonight's context
        if home in (0, 1):
            hv = h[h["Home"] == home]
            sv = hv[PVLOC_STATS].sum()
            out.update(pvloc_from_sums(sv["BF"], sv["HR"], sv["SO"],
                                       sv["ER"], sv["Outs"]))
        else:
            out.update(pvloc_from_sums(0.0, 0.0, 0.0, 0.0, 0.0))
        return out

    def team_offense(self, team, season, date, prefix="toff"):
        row = self._cum(self.team_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {f"{prefix}_hr_pa": np.nan, f"{prefix}_r_pg": np.nan,
                    f"{prefix}_k_pct": np.nan, f"{prefix}_bb_pct": np.nan,
                    f"{prefix}_obp": np.nan, f"{prefix}_slg": np.nan}
        return {f"{prefix}_hr_pa": row["cum_HR"] / row["cum_PA"],
                f"{prefix}_r_pg": row["cum_R"] / row["cum_n"],
                f"{prefix}_k_pct": row["cum_SO"] / row["cum_PA"],
                f"{prefix}_bb_pct": row["cum_BB"] / row["cum_PA"],
                # season OBP/SLG norms — the B-lineup gap's team side (#19)
                f"{prefix}_obp": ((row["cum_H"] + row["cum_BB"]
                                   + row["cum_HBP"]) / row["cum_PA"]),
                f"{prefix}_slg": row["cum_TB"] / row["cum_AB"]}

    def team_sched(self, team, date, venue, day_night):
        """Schedule context off the team's previous game (2026-07-14 #18):
        day_after_night / travel_km / tz_delta via the shared
        sched_from_prev, from the games file's per-team history."""
        hist = self.sched_hist.get(team)
        h = hist[hist["Date"] < date] if hist is not None else None
        if h is None or h.empty:
            return sched_from_prev(None, None, None, date, None, venue,
                                   self.park_coords)
        prev = h.iloc[-1]
        return sched_from_prev(prev["Date"], prev["Venue"],
                               _dn1(prev["DayNight"]), date,
                               _dn1(day_night), venue, self.park_coords)

    def team_bsr(self, team, date):
        """Trailing-30-day BaseRuns luck residual (#20). Table rows exist a
        week past the data horizon; missing (thin window / far future) ->
        NaN, matching the vectorized left-merge."""
        try:
            return float(self._bsr.loc[(team, date)])
        except KeyError:
            return np.nan

    def team_st_outs(self, team, season, date):
        """The manager's leash (#16): team's as-of season starter outs/game."""
        row = self._cum(self.lea_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return np.nan
        return row["cum_Outs"] / row["cum_n"]

    def bullpen(self, team, season, date, prefix="pen"):
        row = self._cum(self.pen_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {f"{prefix}_hr_bf": np.nan, f"{prefix}_k_bf": np.nan,
                    f"{prefix}_h_bf": np.nan, f"{prefix}_bb_bf": np.nan,
                    f"{prefix}_era": np.nan}
        return {f"{prefix}_hr_bf": row["cum_HR"] / row["cum_BF"],
                f"{prefix}_k_bf": row["cum_SO"] / row["cum_BF"],
                f"{prefix}_h_bf": row["cum_H"] / row["cum_BF"],
                # audit wave rank 1: the pen's walk rate (xpen_bb input)
                f"{prefix}_bb_bf": row["cum_BB"] / row["cum_BF"],
                f"{prefix}_era": row["cum_ER"] * 27 / row["cum_Outs"]}

    def bullpen_hl(self, team, season, date, prefix="pen_hl"):
        """High-leverage bullpen quality (save/hold/game-finishing arms)."""
        row = self._cum(self.pen_hl_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {f"{prefix}_era": np.nan, f"{prefix}_k_bf": np.nan}
        return {f"{prefix}_era": row["cum_ER"] * 27 / row["cum_Outs"],
                f"{prefix}_k_bf": row["cum_SO"] / row["cum_BF"]}

    def penhl_share(self, team, season, date):
        """Share of the pen's outs thrown by HL arms (opp_pen_bridge gray
        rider) — the mixture weight the two pen-ERA mains can't recover."""
        hl = self._cum(self.pen_hl_tab, {"Team": team, "Season": season}, date)
        pen = self._cum(self.pen_tab, {"Team": team, "Season": season}, date)
        if hl is None or pen is None or not pen["cum_Outs"]:
            return np.nan
        return hl["cum_Outs"] / pen["cum_Outs"]

    def pen_fatigue(self, team, date):
        """Bullpen pitches thrown over the trailing PEN_FATIGUE_DAYS days.
        Uses the same table as training; dates past the table's horizon
        (predicting more than a week out) mean a fully rested pen -> 0."""
        try:
            return float(self._pen_fat.loc[(team, date)])
        except KeyError:
            return 0.0 if date > self._pen_fat_max else np.nan

    def pen_unavail(self, team, date):
        """Backlog #22: share of the team's HL relief capacity likely
        unavailable tonight (arms up both previous days). Past the table
        horizon -> 0 (fully rested); pre-first-save -> NaN like training."""
        try:
            return float(self._pen_unavail.loc[(team, date)])
        except KeyError:
            return 0.0 if date > self._pen_unavail_max else np.nan

    def lg_sb27_prior(self, season):
        """Prior-season league SB per 27 outs (ranks 4/24 centering)."""
        try:
            return float(self.lg_sb27.loc[int(season)])
        except KeyError:
            return np.nan

    def arsenal_dynamics(self, pid, season):
        """Arsenal breadth/trajectory for one pitcher (#22), exact-season
        lookup mirroring the frames' _arsenal_dynamics merge."""
        try:
            row = self.ars_dyn.loc[(int(pid), int(season))]
            return {c: float(row[c]) for c in ARS_DYN_COLS}
        except KeyError:
            return {c: np.nan for c in ARS_DYN_COLS}

    def team_offense_loc(self, team, season, home, date, prefix="toff_loc"):
        """Team offense in one venue context (home=1 / road=0), as-of."""
        row = self._cum(self.team_loc_tab,
                        {"Team": team, "Season": season, "Home": home}, date)
        if row is None:
            return {f"{prefix}_hr_pa": np.nan, f"{prefix}_r_pg": np.nan}
        return {f"{prefix}_hr_pa": row["cum_HR"] / row["cum_PA"],
                f"{prefix}_r_pg": row["cum_R"] / row["cum_n"]}

    def team_defense(self, team, season, date):
        """Unearned-run rate allowed (defense proxy), as-of."""
        row = self._cum(self.def_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return np.nan
        return (row["cum_R"] - row["cum_ER"]) * 27 / row["cum_Outs"]

    def league_env(self, date):
        """Trailing-30-day league rates for `date`. Exact row for dates the
        data covers (train/serve parity); the latest row for future dates."""
        t = self.env_tab
        m = t[t["Date"] <= date]
        if m.empty:
            return {c: np.nan for c in ENV_COLS}
        return {c: m.iloc[-1][c] for c in ENV_COLS}

    def team_record(self, team, season, date):
        """As-of season W%, run differential, runs allowed, pythag."""
        row = self._cum(self.res_tab, {"Team": team, "Season": season}, date)
        if row is None:
            return {"win_pct": np.nan, "rd_pg": np.nan, "ra_pg": np.nan,
                    "pyth": np.nan}
        return _record_feats(row["cum_W"], row["cum_RF"], row["cum_RA"],
                             row["cum_n"])

    def team_form(self, team, season, date):
        """Win% and run diff over the previous FORM_N games (as-of)."""
        h = self.res_rows
        m = h[(h["Team"] == team) & (h["Season"] == season)
              & (h["Date"] < date)].tail(FORM_N)
        if len(m) < FORM_MIN:
            return {"w20": np.nan, "rd20": np.nan}
        return {"w20": m["W"].mean(), "rd20": (m["RF"] - m["RA"]).mean()}

    def team_elo(self, team, season, date):
        """As-of Elo rating, with the winter carryover regression applied
        when the team's last game was in an earlier season."""
        h = self.elo_hist
        m = h[(h["Team"] == team) & (h["Date"] < date)]
        if m.empty:
            return ELO_BASE
        last = m.iloc[-1]
        e = last["elo_post"]
        if last["Season"] != season:
            e = ELO_BASE + ELO_CARRY * (e - ELO_BASE)
        return e

    def park(self, venue, date):
        out = {"LF": np.nan, "CF": np.nan, "RF": np.nan, "Elevation_ft": np.nan,
               "park_hr_pg": np.nan, "park_r_pg": np.nan, "park_h_pg": np.nan,
               "park_2b_pg": np.nan, "park_3b_pg": np.nan,
               "park_tb_pg": np.nan, "park_vmr": np.nan}
        if venue in self.parks.index:
            p = self.parks.loc[venue]
            out.update(LF=p["LF"], CF=p["CF"], RF=p["RF"],
                       Elevation_ft=p["Elevation_ft"])
        row = self._cum(self.park_tab, {"Venue": venue}, date)
        if row is not None and row["cum_n"] >= 30:
            n = row["cum_n"]
            for stat, col in (("HR", "park_hr_pg"), ("R", "park_r_pg"),
                              ("H", "park_h_pg"), ("2B", "park_2b_pg"),
                              ("3B", "park_3b_pg"), ("TB", "park_tb_pg")):
                out[col] = row[f"cum_{stat}"] / n
            # audit wave rank 32: venue run-dispersion (shared helper)
            out["park_vmr"] = float(park_vmr_from_sums(
                n, row["cum_R"], row["cum_R2"]))
        return out

    def bio(self, pid):
        r = self.rosters.loc[pid] if pid in self.rosters.index else None
        bat = self.bats.get(pid)
        thr = self.throws.get(pid)
        if pd.isna(bat) or bat is None:
            bat = r["B"] if r is not None else np.nan
        if pd.isna(thr) or thr is None:
            thr = r["T"] if r is not None else np.nan
        if r is not None:
            return {"bat_hand": bat, "pit_throws": thr,
                    "height": r["height_in"],
                    "weight": pd.to_numeric(r["Wt"], errors="coerce"),
                    "dob": r["DOB"]}
        return {"bat_hand": bat, "pit_throws": thr, "height": np.nan,
                "weight": np.nan, "dob": pd.NaT}
