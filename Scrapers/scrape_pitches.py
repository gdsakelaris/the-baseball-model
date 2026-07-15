"""Scrape pitch-level Statcast data, stored as DAILY PER-PLAYER AGGREGATES.

The batted-ball file (scrape_statcast.py) covers only balls put in play —
no whiffs, no called strikes, no chases. This pulls EVERY pitch 2020-2026
from the same statcast_search CSV endpoint, but ~700k pitches per season is
too much to keep raw, so each chunk is aggregated on arrival into one row
per player per day of sufficient statistics:

  both files:   pitches, swings, whiffs (total + out-of-zone), called
                strikes, zone/out-of-zone counts, graded velo-band buckets
                on FF/SI (fblo_ <92 / fbmid_ 92-95 / fb95_ 95+, each
                n/sw/wh), breaking + offspeed buckets (brk_/off_ n/sw/wh),
                edge-of-zone count (edge_n, shadow band 0.67-1.33 of
                scaled zone), first-pitch counts (fp_n seen, fp_sw swung,
                fp_s strike)
  pitcher file: + fastball velo sum/count (FF+SI), and the v6 sequencing/
                count-state/movement sums (2026-07-14): 0-2 waste
                (c02_n/c02_w — located 0-2 pitches and the share thrown
                beyond the shadow band), ahead/behind pitch-class usage
                (ah_/bh_ n/brk/off — does the mix collapse when behind),
                back-to-back pitch-class transitions within an at-bat
                (tr_n pairs, tr_same repeats, tr_fbbrk fastball->breaking
                — a tunneling/predictability proxy), per-start OLS slope
                of FF/SI velo vs the pitcher's own pitch index (fade_w
                weight, fade_num = slope x weight; in-game stamina), and
                FF induced vertical break (ivb_n/ivb_sum, inches — ride)
                + the v7 audit-wave sums (2026-07-14): FF/SI velo with
                runners on (fbstr_n/fbstr_v — stretch-vs-windup split),
                perceived-velo premium (fbe_n/fbe_sum = effective minus
                release speed on FF/SI; extension), per-class release
                centroids (rpf_/rpb_ n/x/z + x2/z2 — fastball-remainder
                vs breaking arm-slot separation; sumsqs staged for a
                within-class scatter refinement), and breaking-ball
                movement magnitude (brkmov_n/brkmov_sum, inches,
                12*hypot(pfx_x, pfx_z))
  both (v7):    two-strike x breaking-class cell (ts_brk_n/sw/wh — the
                putaway cell; batter side is the consumer)
  both (v8):    damage-on-contact sums (2026-07-15): balls in play with a
                Savant xwOBA estimate + the xwOBA sum, total (con_n/
                con_xw) and per bucket — velo bands (fblo_/fbmid_/fb95_
                bip + xw) and pitch classes (brk_/off_ bip + xw; the
                fastball-remainder class is derived downstream as
                con - brk - off, the fbk_sw/fbk_wh idiom) — the damage
                sibling of every whiff cell, and the two-strike x elite-
                velo cell (ts_fb95_n/sw/wh, the ts_brk mirror)

From these, features can rebuild any rate (swinging-strike%, CSW%, whiff
per swing, chase%, zone%) as career or decay-weighted as-of values —
swing-and-miss and plate discipline are the fastest-stabilizing skills in
baseball, and fastball-velocity trend is the classic in-season decline
signal. oz_wh splits whiffs by zone, so IN-ZONE contact rate (the most
stable hit-tool skill: z_contact = 1 - (wh_n-oz_wh)/(sw_n-oz_sw)) is
derivable; the fb95 buckets give a batter's performance against elite
velocity, to cross with tonight's starter's fastball velo; the brk/off
buckets give whiff-vs-pitch-class splits to collide with the opposing
pitcher's usage mix; edge_n/n is a command proxy; fp_s/fp_n is
first-pitch-strike% (pitcher) and fp_sw/fp_n first-pitch aggression
(batter). None of this exists in the box scores or the batted-ball file.

--backfill also archives every raw pitch to Data/raw_pitches/
pitches_{year}.parquet (~all Savant detail columns, zstd), so future
schema changes re-aggregate from disk via --from-raw instead of paying
another 6-hour Savant download.

Relational keys: PlayerId is the MLBAM id used everywhere; one row per
(PlayerId, Date).

Default run is incremental — the output CSVs double as the cache: stored
seasons are reused; only the newest REFETCH_DAYS (Statcast back-corrects)
plus any missing season are downloaded. Seconds in the daily job.
--backfill re-downloads all seasons (~25-35 minutes).

Usage:
    python scrape_pitches.py [--outdir DIR] [--backfill] [--from-raw]
"""

import argparse
import io
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from seasons import YEARS

DATA_DIR = Path(__file__).resolve().parents[1] / "Data"
RAW_DIR = DATA_DIR / "raw_pitches"
OUT_PITCHERS = "mlb_pitch_daily_pitchers.csv"
OUT_BATTERS = "mlb_pitch_daily_batters.csv"

API_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}

# description buckets (verified against live data 2026-07)
SWINGS = {"foul", "foul_tip", "hit_into_play", "swinging_strike",
          "swinging_strike_blocked", "foul_bunt", "missed_bunt",
          "bunt_foul_tip"}
WHIFFS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FASTBALLS = {"FF", "SI"}         # velo trend tracked on four-seam + sinker
ELITE_VELO = 95.0                # fb95 buckets: fastballs at/above this mph
BAND_LO = 92.0                   # graded velo bands: <92 / 92-95 / 95+
# Savant pitch-class grouping; cutters/others fall in the fastball remainder
BREAKING = {"SL", "ST", "SV", "CU", "KC", "CS", "SC", "KN"}
OFFSPEED = {"CH", "FS", "FO", "EP"}
# shadow band of the scaled zone (1.0 = edge; x incl. ball radius = 0.83 ft)
EDGE_X_HALF = 0.83
EDGE_LO, EDGE_HI = 0.67, 1.33

CHUNK_DAYS = 4                   # ~18k pitches in-season; cap guard splits
CAP_ROWS = 24000
SLEEP = 2.0
REFETCH_DAYS = 2
FADE_MIN_FB = 8                  # fastballs needed for a per-start velo slope


def fetch_range(d0, d1, tries=3):
    """All pitches in [d0, d1] (regular season). Splits on the result cap."""
    params = {
        "all": "true", "type": "details", "player_type": "batter",
        "hfGT": "R|", "minors": "false",
        "game_date_gt": str(d0), "game_date_lt": str(d1),
    }
    for attempt in range(tries):
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS,
                             timeout=240)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
            break
        except Exception as e:                      # noqa: BLE001
            if attempt == tries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    retry {d0}..{d1} in {wait}s ({e})", flush=True)
            time.sleep(wait)
    if len(df) >= CAP_ROWS and d0 != d1:
        mid = d0 + (d1 - d0) / 2
        print(f"    {d0}..{d1}: {len(df):,} rows (cap?) -> splitting",
              flush=True)
        time.sleep(SLEEP)
        left = fetch_range(d0, mid)
        time.sleep(SLEEP)
        right = fetch_range(mid + timedelta(days=1), d1)
        return pd.concat([left, right], ignore_index=True)
    return df


def aggregate(raw):
    """One Savant chunk -> (pitcher day rows, batter day rows)."""
    if raw.empty:
        return None, None
    # canonical in-game pitch order (game, at-bat, pitch) so the v6
    # sequencing pairs and the per-start velo-fade index are well-defined;
    # mergesort keeps ties stable. Games never span a chunk (one day each).
    raw = raw.copy()
    raw["_gpk"] = pd.to_numeric(raw["game_pk"], errors="coerce")
    raw["_abn"] = pd.to_numeric(raw["at_bat_number"], errors="coerce")
    raw["_pnum"] = pd.to_numeric(raw["pitch_number"], errors="coerce")
    raw = raw.sort_values(["_gpk", "_abn", "_pnum"],
                          kind="mergesort").reset_index(drop=True)
    desc = raw["description"].astype(str)
    zone = pd.to_numeric(raw["zone"], errors="coerce")
    swing = desc.isin(SWINGS)
    whiff = desc.isin(WHIFFS)
    in_zone = zone.between(1, 9)
    out_zone = zone >= 11
    is_fb = raw["pitch_type"].isin(FASTBALLS)
    is_brk = raw["pitch_type"].isin(BREAKING)
    is_off = raw["pitch_type"].isin(OFFSPEED)
    velo = pd.to_numeric(raw["release_speed"], errors="coerce")
    is_fb95 = is_fb & (velo >= ELITE_VELO)
    is_fbmid = is_fb & (velo >= BAND_LO) & (velo < ELITE_VELO)
    is_fblo = is_fb & (velo < BAND_LO)
    balls = pd.to_numeric(raw["balls"], errors="coerce")
    strikes = pd.to_numeric(raw["strikes"], errors="coerce")
    first = (balls == 0) & (strikes == 0)
    two_k = strikes == 2
    full = (balls == 3) & (strikes == 2)
    is_ball = desc.isin(("ball", "blocked_ball"))
    # release-point coords (mechanical repeatability -> sum + sum-of-
    # squares so consumers can rebuild the scatter from cumulative sums)
    rx = pd.to_numeric(raw["release_pos_x"], errors="coerce")
    rz = pd.to_numeric(raw["release_pos_z"], errors="coerce")
    rp_ok = rx.notna() & rz.notna()
    # shadow-band location: scale so 1.0 = zone edge on each axis
    px = pd.to_numeric(raw["plate_x"], errors="coerce")
    pz = pd.to_numeric(raw["plate_z"], errors="coerce")
    top = pd.to_numeric(raw["sz_top"], errors="coerce")
    bot = pd.to_numeric(raw["sz_bot"], errors="coerce")
    x_sc = px.abs() / EDGE_X_HALF
    z_sc = (pz - (top + bot) / 2).abs() / ((top - bot) / 2).clip(lower=0.1)
    loc = pd.concat([x_sc, z_sc], axis=1).max(axis=1)
    edge = (loc > EDGE_LO) & (loc <= EDGE_HI)
    # ---- v6 sequencing / count-state / movement (2026-07-14) ----
    # 0-2 waste: of located 0-2 pitches, the share thrown beyond the shadow
    # band (loc > EDGE_HI = non-competitive by design)
    is02 = (balls == 0) & (strikes == 2)
    loc_ok = loc.notna()
    # ahead/behind count states (0-0 and even counts sit in neither)
    ahead = strikes > balls
    behind = balls > strikes
    # back-to-back pitch-class transitions within one at-bat, assigned to
    # the SECOND pitch of the pair: same-class repeats (predictability) and
    # fastball->breaking (the classic tunneling pair)
    cls = pd.Series(np.where(is_brk, 1, np.where(is_off, 2, 0)),
                    index=raw.index)
    known = raw["pitch_type"].notna()
    same_ab = ((raw["_gpk"] == raw["_gpk"].shift())
               & (raw["_abn"] == raw["_abn"].shift()))
    tr_ok = same_ab & known & known.shift(fill_value=False)
    tr_same = tr_ok & (cls == cls.shift())
    tr_fbbrk = tr_ok & (cls.shift() == 0) & (cls == 1)
    # FF induced vertical break (ride), inches — pfx_z x 12 on four-seamers
    pfz = pd.to_numeric(raw["pfx_z"], errors="coerce")
    ivb_ok = (raw["pitch_type"] == "FF") & pfz.notna()
    # ---- v7 audit wave (2026-07-14) ----
    # stretch split: FF/SI velo with any runner on (pitching from the
    # stretch) vs the windup complement (rebuilt downstream from fb_n/fb_v)
    runners = (raw["on_1b"].notna() | raw["on_2b"].notna()
               | raw["on_3b"].notna())
    fbv_ok = is_fb & velo.notna()
    # perceived-velo premium: effective (extension-adjusted) minus release
    # speed on FF/SI — league mean drifts by tracking era (-0.56 in 2015 ->
    # +0.30 in 2024, Hawk-Eye era ~+0.15), so downstream priors fit the
    # recent era and the GBM absorbs the slow level shift
    eff = pd.to_numeric(raw["effective_speed"], errors="coerce")
    fbe_ok = fbv_ok & eff.notna()
    # per-class release centroids: fastball-remainder class (everything not
    # breaking/offspeed, incl. cutters — the cls==0 remainder) vs the
    # breaking set. Between-class arm-slot separation is a deception trait
    # the pooled rp_ scatter confounds; sumsqs are staged so a within-class
    # scatter can be rebuilt later without another re-agg.
    is_fbrem = known & ~is_brk & ~is_off
    rpf_ok = is_fbrem & rp_ok
    rpb_ok = is_brk & rp_ok
    # breaking-ball movement magnitude (cause axis; whiff outcome already
    # covered by brk_wh) — total break in inches from pfx components
    pfx = pd.to_numeric(raw["pfx_x"], errors="coerce")
    brkmov_ok = is_brk & pfx.notna() & pfz.notna()
    # ---- v8 damage-on-contact wave (2026-07-15) ----
    # Savant's estimated wOBA (EV+LA) on balls in play, per velo band and
    # pitch class — the damage sibling of the whiff cells (whiff says he
    # misses 95+; this says what happens when he doesn't). Counts gate on
    # a present estimate, mirroring the BIP file's xw_n convention.
    xw = pd.to_numeric(raw["estimated_woba_using_speedangle"],
                       errors="coerce")
    inplay = (desc == "hit_into_play") & xw.notna()
    is_tsfb95 = two_k & is_fb95
    base = pd.DataFrame({
        "Date": pd.to_datetime(raw["game_date"]).dt.date,
        "PitcherId": pd.to_numeric(raw["pitcher"], errors="coerce"),
        "BatterId": pd.to_numeric(raw["batter"], errors="coerce"),
        "n": 1.0,
        "sw_n": swing.astype(float),
        "wh_n": whiff.astype(float),
        "cs_n": (desc == "called_strike").astype(float),
        "z_n": in_zone.astype(float),
        "oz_n": out_zone.astype(float),
        "oz_sw": (out_zone & swing).astype(float),
        "oz_wh": (out_zone & whiff).astype(float),
        "fb95_n": is_fb95.astype(float),
        "fb95_sw": (is_fb95 & swing).astype(float),
        "fb95_wh": (is_fb95 & whiff).astype(float),
        "fbmid_n": is_fbmid.astype(float),
        "fbmid_sw": (is_fbmid & swing).astype(float),
        "fbmid_wh": (is_fbmid & whiff).astype(float),
        "fblo_n": is_fblo.astype(float),
        "fblo_sw": (is_fblo & swing).astype(float),
        "fblo_wh": (is_fblo & whiff).astype(float),
        "brk_n": is_brk.astype(float),
        "brk_sw": (is_brk & swing).astype(float),
        "brk_wh": (is_brk & whiff).astype(float),
        "off_n": is_off.astype(float),
        "off_sw": (is_off & swing).astype(float),
        "off_wh": (is_off & whiff).astype(float),
        "edge_n": edge.astype(float),
        "fp_n": first.astype(float),
        "fp_sw": (first & swing).astype(float),
        "fp_s": (first & (swing | (desc == "called_strike"))).astype(float),
        "ts_n": two_k.astype(float),
        "ts_sw": (two_k & swing).astype(float),
        "ts_wh": (two_k & whiff).astype(float),
        "f32_n": full.astype(float),
        "f32_z": (full & in_zone).astype(float),
        "f32_b": (full & is_ball).astype(float),
        "f32_sw": (full & swing).astype(float),
        "f32_wh": (full & whiff).astype(float),
        "fb_n": (is_fb & velo.notna()).astype(float),
        "fb_v": velo.where(is_fb).fillna(0.0),
        "fb_v2": (velo ** 2).where(is_fb).fillna(0.0),
        "rp_n": rp_ok.astype(float),
        "rp_x": rx.where(rp_ok).fillna(0.0),
        "rp_x2": (rx ** 2).where(rp_ok).fillna(0.0),
        "rp_z": rz.where(rp_ok).fillna(0.0),
        "rp_z2": (rz ** 2).where(rp_ok).fillna(0.0),
        "c02_n": (is02 & loc_ok).astype(float),
        "c02_w": (is02 & (loc > EDGE_HI)).astype(float),
        "ah_n": (ahead & known).astype(float),
        "ah_brk": (ahead & is_brk).astype(float),
        "ah_off": (ahead & is_off).astype(float),
        "bh_n": (behind & known).astype(float),
        "bh_brk": (behind & is_brk).astype(float),
        "bh_off": (behind & is_off).astype(float),
        "tr_n": tr_ok.astype(float),
        "tr_same": tr_same.astype(float),
        "tr_fbbrk": tr_fbbrk.astype(float),
        "ivb_n": ivb_ok.astype(float),
        "ivb_sum": (pfz * 12.0).where(ivb_ok).fillna(0.0),
        "ts_brk_n": (two_k & is_brk).astype(float),
        "ts_brk_sw": (two_k & is_brk & swing).astype(float),
        "ts_brk_wh": (two_k & is_brk & whiff).astype(float),
        "fbstr_n": (fbv_ok & runners).astype(float),
        "fbstr_v": velo.where(fbv_ok & runners).fillna(0.0),
        "fbe_n": fbe_ok.astype(float),
        "fbe_sum": (eff - velo).where(fbe_ok).fillna(0.0),
        "rpf_n": rpf_ok.astype(float),
        "rpf_x": rx.where(rpf_ok).fillna(0.0),
        "rpf_z": rz.where(rpf_ok).fillna(0.0),
        "rpf_x2": (rx ** 2).where(rpf_ok).fillna(0.0),
        "rpf_z2": (rz ** 2).where(rpf_ok).fillna(0.0),
        "rpb_n": rpb_ok.astype(float),
        "rpb_x": rx.where(rpb_ok).fillna(0.0),
        "rpb_z": rz.where(rpb_ok).fillna(0.0),
        "rpb_x2": (rx ** 2).where(rpb_ok).fillna(0.0),
        "rpb_z2": (rz ** 2).where(rpb_ok).fillna(0.0),
        "brkmov_n": brkmov_ok.astype(float),
        "brkmov_sum": (12.0 * np.hypot(pfx, pfz)).where(brkmov_ok)
                      .fillna(0.0),
        "con_n": inplay.astype(float),
        "con_xw": xw.where(inplay).fillna(0.0),
        "fblo_bip": (is_fblo & inplay).astype(float),
        "fblo_xw": xw.where(is_fblo & inplay).fillna(0.0),
        "fbmid_bip": (is_fbmid & inplay).astype(float),
        "fbmid_xw": xw.where(is_fbmid & inplay).fillna(0.0),
        "fb95_bip": (is_fb95 & inplay).astype(float),
        "fb95_xw": xw.where(is_fb95 & inplay).fillna(0.0),
        "brk_bip": (is_brk & inplay).astype(float),
        "brk_xw": xw.where(is_brk & inplay).fillna(0.0),
        "off_bip": (is_off & inplay).astype(float),
        "off_xw": xw.where(is_off & inplay).fillna(0.0),
        "ts_fb95_n": is_tsfb95.astype(float),
        "ts_fb95_sw": (is_tsfb95 & swing).astype(float),
        "ts_fb95_wh": (is_tsfb95 & whiff).astype(float),
    })
    shared = ["n", "sw_n", "wh_n", "cs_n", "z_n", "oz_n", "oz_sw", "oz_wh",
              "fb95_n", "fb95_sw", "fb95_wh",
              "fbmid_n", "fbmid_sw", "fbmid_wh",
              "fblo_n", "fblo_sw", "fblo_wh",
              "brk_n", "brk_sw", "brk_wh", "off_n", "off_sw", "off_wh",
              "edge_n", "fp_n", "fp_sw", "fp_s", "ts_n", "ts_sw", "ts_wh",
              "f32_n", "f32_z", "f32_b", "f32_sw", "f32_wh",
              "ts_brk_n", "ts_brk_sw", "ts_brk_wh",
              # v8 damage-on-contact sums + 2K x elite-velo cell
              "con_n", "con_xw",
              "fblo_bip", "fblo_xw", "fbmid_bip", "fbmid_xw",
              "fb95_bip", "fb95_xw",
              "brk_bip", "brk_xw", "off_bip", "off_xw",
              "ts_fb95_n", "ts_fb95_sw", "ts_fb95_wh"]
    pit_stats = shared + ["fb_n", "fb_v", "fb_v2",
                          "rp_n", "rp_x", "rp_x2", "rp_z", "rp_z2",
                          "c02_n", "c02_w", "ah_n", "ah_brk", "ah_off",
                          "bh_n", "bh_brk", "bh_off",
                          "tr_n", "tr_same", "tr_fbbrk",
                          "ivb_n", "ivb_sum",
                          "fbstr_n", "fbstr_v", "fbe_n", "fbe_sum",
                          "rpf_n", "rpf_x", "rpf_z", "rpf_x2", "rpf_z2",
                          "rpb_n", "rpb_x", "rpb_z", "rpb_x2", "rpb_z2",
                          "brkmov_n", "brkmov_sum"]
    bat_stats = shared
    pit = (base.dropna(subset=["PitcherId"])
           .astype({"PitcherId": "int64"})
           .groupby(["PitcherId", "Date"], as_index=False)[pit_stats].sum()
           .rename(columns={"PitcherId": "PlayerId"}))
    # per-start velo-fade slope (v6): OLS of FF/SI velo against the
    # pitcher's own pitch index within the game (all pitches count toward
    # the index; only fastballs enter the regression). Stored as a
    # weight (fade_w = fastballs, gated at FADE_MIN_FB) and slope x weight
    # (fade_num), so decayed sums rebuild a weighted mean slope — a pooled
    # cross-game regression would conflate game intercepts, so the slope is
    # computed per start HERE and only averaged downstream.
    pidx = raw.groupby(["_gpk", "pitcher"]).cumcount().astype(float)
    fb_ok = is_fb & velo.notna() & raw["_gpk"].notna()
    fr = pd.DataFrame({
        "PlayerId": pd.to_numeric(raw["pitcher"], errors="coerce"),
        "Date": pd.to_datetime(raw["game_date"]).dt.date,
        "g": raw["_gpk"], "x": pidx, "y": velo,
        "xy": pidx * velo, "xx": pidx * pidx})[fb_ok.to_numpy()]
    if len(fr):
        g = fr.dropna(subset=["PlayerId"]).astype({"PlayerId": "int64"}) \
              .groupby(["PlayerId", "Date", "g"])
        s = g.agg(n=("y", "size"), sx=("x", "sum"), sy=("y", "sum"),
                  sxy=("xy", "sum"), sxx=("xx", "sum")).reset_index()
        den = s["n"] * s["sxx"] - s["sx"] ** 2
        slope = (s["n"] * s["sxy"] - s["sx"] * s["sy"]) / den.where(den > 0)
        ok = (s["n"] >= FADE_MIN_FB) & slope.notna()
        s["fade_w"] = s["n"].where(ok, 0.0)
        s["fade_num"] = (slope * s["n"]).where(ok, 0.0)
        fade = s.groupby(["PlayerId", "Date"], as_index=False)[
            ["fade_w", "fade_num"]].sum()
        pit = pit.merge(fade, on=["PlayerId", "Date"], how="left")
    for c in ("fade_w", "fade_num"):
        if c not in pit.columns:
            pit[c] = 0.0
        pit[c] = pit[c].fillna(0.0)
    bat = (base.dropna(subset=["BatterId"])
           .astype({"BatterId": "int64"})
           .groupby(["BatterId", "Date"], as_index=False)[bat_stats].sum()
           .rename(columns={"BatterId": "PlayerId"}))
    return pit, bat


def season_windows(year, start=None):
    d0 = date(year, 3, 1) if start is None else max(start, date(year, 3, 1))
    end = min(date(year, 11, 30), date.today())
    windows = []
    while d0 <= end:
        d1 = min(d0 + timedelta(days=CHUNK_DAYS - 1), end)
        windows.append((d0, d1))
        d0 = d1 + timedelta(days=1)
    return windows


def write_raw(year, frames):
    """Archive one season of raw pitches so schema changes never need a
    re-download. Object columns are stringified (mixed-type chunks)."""
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).where(df[c].notna())
    path = RAW_DIR / f"pitches_{year}.parquet"
    try:
        df.to_parquet(path, index=False, compression="zstd")
    except Exception as e:                              # noqa: BLE001
        path = RAW_DIR / f"pitches_{year}.csv.gz"
        print(f"    parquet failed ({e}); falling back to csv.gz", flush=True)
        df.to_csv(path, index=False, compression="gzip")
    print(f"    raw archive: {len(df):,} pitches -> {path.name}", flush=True)


def load_existing(path, backfill):
    if backfill or not path.exists():
        return None, None, set()
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    have = set(pd.to_datetime(df["Date"]).map(lambda d: d.year))
    newest = df["Date"].max()
    start = newest - timedelta(days=REFETCH_DAYS)
    return df[df["Date"] < start], start, have


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(DATA_DIR))
    ap.add_argument("--backfill", action="store_true",
                    help="re-download all seasons (default is incremental)")
    ap.add_argument("--from-raw", action="store_true", dest="from_raw",
                    help="re-aggregate seasons from Data/raw_pitches archives "
                         "where present (implies --backfill for those years)")
    args = ap.parse_args()
    if args.from_raw:
        args.backfill = True
    outdir = Path(args.outdir)
    p_path, b_path = outdir / OUT_PITCHERS, outdir / OUT_BATTERS

    # the pitcher file drives incrementality; both files are written together
    kept_p, start, have = load_existing(p_path, args.backfill)
    kept_b, _, _ = load_existing(b_path, args.backfill)
    if start is not None:
        print(f"incremental: {len(kept_p):,} pitcher-day rows kept, "
              f"refetching from {start}", flush=True)

    pit_frames, bat_frames = [], []
    for year in YEARS:
        if start is not None and year < start.year and year in have:
            continue
        clip = start if start is not None and year >= start.year else None
        raw_path = RAW_DIR / f"pitches_{year}.parquet"
        if args.from_raw and clip is None and raw_path.exists():
            pit, bat = aggregate(pd.read_parquet(raw_path))
            if pit is not None:
                pit_frames.append(pit)
                bat_frames.append(bat)
            print(f"{year}: {int(pit['n'].sum()) if pit is not None else 0:,}"
                  f" pitches (raw archive)", flush=True)
            continue
        got = 0
        raw_chunks = [] if args.backfill else None
        for d0, d1 in season_windows(year, clip):
            raw = fetch_range(d0, d1)
            if raw_chunks is not None and not raw.empty:
                raw_chunks.append(raw)
            pit, bat = aggregate(raw)
            if pit is not None:
                got += int(pit["n"].sum())
                pit_frames.append(pit)
                bat_frames.append(bat)
            time.sleep(SLEEP)
        if raw_chunks is not None:
            write_raw(year, raw_chunks)
            raw_chunks.clear()
        print(f"{year}: {got:,} pitches", flush=True)

    def finish(kept, frames, path, key=("PlayerId", "Date")):
        new = pd.concat(frames, ignore_index=True) if frames else None
        if new is not None and kept is not None:
            new = pd.concat([kept, new], ignore_index=True)
        elif new is None:
            new = kept
        new = (new.drop_duplicates(list(key), keep="last")
               .sort_values(list(key)))
        path.parent.mkdir(exist_ok=True)
        new.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {len(new):,} rows -> {path}", flush=True)

    finish(kept_p, pit_frames, p_path)
    finish(kept_b, bat_frames, b_path)


if __name__ == "__main__":
    main()
