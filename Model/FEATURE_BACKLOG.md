# Feature Backlog — MLB model

Living list of (1) features that are **benched / out of the superset** and should be
unbenched or recreated, and (2) **new feature ideas** to build and expose to feature
selection.

**The process, given selection-as-curation (2026-07-10):** the job of this list is only
to get candidate columns *into the superset*. Whether any head keeps a column is decided
by `feature_select.py` (SHAP stability vote, PI≥0.75 both suites) + the paired-CI eval —
not by hand. So "expose it and let selection vote" is the default disposition for
everything here. Old manual benches are treated as hypotheses to re-test, not verdicts.

**Sequencing:** Part 1 (unbench/recreate) executes **after** the current selection train +
eval verdict lands and we re-baseline — not before (mid-experiment). Part 2 items get built
opportunistically and added to the superset for the *next* selection run.

---

## Part 1 — Unbench / recreate inventory

Everything currently computed-but-excluded, or removed entirely, as of 2026-07-10. Action =
add the column(s) back to the relevant `*_feature_cols()` list (unbench) or rebuild the
column first (recreate). After adding, re-run `feature_select.py --stat shap --write` so the
keep-list gets a vote on them.

| # | Feature | Frame / model | State | Old bench reason (regime it was judged under) | Action |
|---|---------|---------------|-------|-----------------------------------------------|--------|
| 1 | `pull_fence` | batter | computed [features.py:1706](features.py#L1706), not in `batter_feature_cols` | iteration-3: hurt batter props on holdout (manual regime) | **unbench** — add to cols list |
| 2 | `porch_margin` | batter | computed [features.py:1709](features.py#L1709), not in cols | iteration-3, same as above | **unbench** — add to cols list |
| 3 | batter-side fatigue | batter | **not computed anymore** (only pitcher/pen fatigue exists; batter has only `days_rest`) | iteration-3, grouped with pull_fence | **recreate** — build trailing games-in-N-days density / PA load / day-after-night flag, then add |
| 4 | `d_ps_xwcon_d` (starter contact-allowed diff) | winner | in frame + runs model; excluded from `win_feature_cols` [features.py:2254](features.py#L2254) | overfits the ~10k-row winner when widened (capacity, not signal) | **unbench to winner** — let selection's winner keep-list vote (see caveat) |
| 5 | `lg_*` env cols (`lg_r_pa`, `lg_hr_pa`) | team-runs | in team frame [features.py:2306](features.py#L2306); excluded from `team_game_feature_cols` [features.py:2316](features.py#L2316) | iteration-4: cost the runs model a little MAE | **unbench to runs** — let selection vote |

**Caveat on #4/#5 (small-n frames):** the winner (~10k games) and runs models overfit when
widened — that's why these were benched. SHAP importance on cal-year rows can look fine while
the column still hurts *generalization* on a small model, so selection's guardrail is weaker
here than on the ~190k-row batter frame. Expose them, but weight the paired-CI read (not the
keep-vote) as the real arbiter for these two.

**Not benched — do not touch (recorded so they aren't re-swept):**
- `BAT_TRACK_COLS` — already in the batter superset, all-NaN/**inert until ~2027** (bat-tracking
  coverage vs the training window). Time-gated, not benched.
- `MONOTONE = {"hr": HR_MONOTONE}` — **active** (re-accepted queue Tier A1). Not a feature.
- `STACK_DONORS` (thin-prop stacking), seed-bagging — modeling machinery behind levers, not
  superset features. Out of scope for "unbench features."

---

## Part 2 — New feature ideas (maintained backlog)

Spirit: **new information geometry / interactions the tree can't currently carve for itself**
— not re-encodings of signal it already has (recency, rolling rates, dev-from-baseline are
all already in). Status: `BUILD` = spec'd, ready; `VERIFY` = check it isn't already covered
before building; `SCRAPE` = needs data we don't have.

### Tier 1 — buildable from data already in the frame (no scrape)

1. **Effective pull-field carry wind** — `BUILD` — *headline.*
   Inputs: `WindDir` (already field-relative: `Out To CF/RF/LF`, `In From ...`, `L To R`,
   `R To L`, `Varies`, `Calm`), `WindSpeed`, `eff_hand`. Build a signed scalar = carry
   component toward the batter's pull field (LHB→RF, RHB→LF): Out-to-pull = +, In-from = −,
   crosswind/Calm/Varies ≈ 0, × `WindSpeed`. Hands the model the physics of a 3-way
   interaction it can't carve from a high-cardinality categorical × two numerics over a rare
   event. Targets: hr, tb2, xtb, single/double, and (handedness-free version) team-runs/total.
   Note: only "on" for the ~12k out/in games (crosswinds ≈ 7.3k, Varies/Calm ≈ 1.3k), so
   tempered magnitude. No new scrape — all inputs in the frame.

2. **General out/in carry wind** — `BUILD` — handedness-free sibling of #1 (Out=+, In=−,
   cross=0) × `WindSpeed`. For team-runs / total / the HR *environment* (park-level carry),
   where pull side doesn't apply. Cheap; pairs with #1.

3. **Wind × pull_fence interaction** — `BUILD` (after #1 + unbench of `pull_fence`) —
   short porch + wind blowing out compounds. Carry-wind (#2) × fence proximity on the pull
   side. Only meaningful once `pull_fence` is back in the superset.

4. **Temp × Elevation carry index** — `BUILD` — hot + high air both raise carry (Coors,
   hot Arlington). `Temp` and `Elevation_ft` are in the frame as separate main effects;
   give the model the product / a simple air-density carry index. Targets: hr/tb2/total.

5. **Batted-ball profile × opponent defense** — `BUILD` — `bip_gb` / `bip_pullair`
   (GB-vs-air tendency) × `opp_oaa`. A groundball hitter vs an elite infield loses BABIP; a
   flyball hitter's BABIP is less defense-sensitive. Targets: hit, single, double.

6. **Recent BABIP-regression signal** — `BUILD` — batter's recent *actual* wOBA/BA minus
   `bip_xwoba` (xwOBA-on-contact): running hot/cold on balls in play, due to regress.
   Distinct from raw rolling form (which is outcome-only). Targets: hit, single.

7. **Umpire × matchup interaction** — `VERIFY` then `BUILD` — `ump_k_pct` × pitcher
   called-strike rate, or × batter chase (`bd_chase_*`). Ump zone currently enters as a main
   effect only; the amplification is matchup-specific. Targets: k, bb, bk, pbb.

8. **Park handed-HR factor × batter hand** — `VERIFY` (may overlap `pull_fence`/`park_hr_pg`)
   — realized park HR rate split by batter handedness (short-RF parks help LHB specifically),
   crossed with `eff_hand`. Build only if it adds beyond fence distance. Targets: hr, tb2.

9. **Pitcher-usage × batter-pitch-type weakness** — `VERIFY` (check `matchup_features`
   doesn't already do this) then `BUILD` — `mlb_pitch_arsenals` (pitcher mix %) crossed with
   `mlb_pitch_arsenals_batters` (batter performance by pitch type): how heavily this pitcher
   throws the pitch types this batter is *worst* against. A true matchup score beyond
   arsenal-quality blend. Targets: hr, hit, k, tb2.

10. **Platoon-split magnitude** — `VERIFY` (may be covered by `bvh_*` hand-split contact) —
    the batter's own L/R split *size* (his platoon vulnerability) × matchup handedness.
    Targets: hit, tb2, hr.

### Tier 2 — needs a scrape / new data source

11. **Weather beyond wind** — `SCRAPE` — humidity + barometric pressure (denser air suppresses
    carry; a real, physical HR driver not captured by Temp alone). Needs a weather source
    keyed by game/venue/time.

12. **Batter-vs-pitcher direct history (BvP)** — **IMPLEMENTED 2026-07-10** (vectorized path;
    serving+parity deferred to ship). No scrape needed — `mlb_statcast_bip.csv` already carries
    `BatterId × PitcherId × Date × Events × xwOBA`. Built `_bvp_table` (pairwise as-of cumsums,
    same leakage-safe idiom as `_bip_table`/`_hrpt_scores`) + 3 cols in `batter_feature_cols`:
    `bvp_n` (log1p pairwise contact count), `bvp_xwoba_resid` (shrunk xwOBA-on-contact vs this
    starter, residual off the batter's OWN `bip_xwoba` baseline, K=30), `bvp_hr_resid` (HR/contact
    residual vs league prior, K=50). Contact-only (BIP has no K/BB → contact props only). Priors/K
    set by convention like BIP_SHRINK, NOT swept. Sparse: mean 2.68 shared contacts/pair, only
    3.4% of pairs ≥10 → shrinkage keeps most rows near neutral; a selection-gated experiment
    (default off x-heads). Was "mostly re-derives hand+arsenal" — the residual-off-own-baseline
    encoding is designed to strip exactly that redundancy, leaving the pitcher-specific effect.

13. **Catcher framing** — **SHELVED** — needs catcher-level serving input the user can't
    scrape at predict time. Not actionable; kept here only so it isn't re-proposed.

---

## Log
- 2026-07-10: created. Part 1 inventory verified against tree at commit `3123308`. Part 1
  execution deferred to after the selection-as-curation train verdict + re-baseline.
- 2026-07-10 (dev batch): IMPLEMENTED in the vectorized path + feature-col lists (serving
  path deferred to ship). Part 1 all done — `pull_fence`/`porch_margin` unbenched (batter),
  `lg_r_pa`/`lg_hr_pa` (runs), `d_ps_xwcon_d` (winner), batter-side fatigue recreated as
  `g_l7d`/`g_l14d` (games in last 7/14 days). Part 2 Tier-1 #1-6 built: `bat_wind_pull`,
  `wind_carry`, `bat_wind_porch`, `carry_air`, `bip_gb_def`/`bip_air_def`, `hit_luck`
  (recent decayed BA-on-contact minus expected xBA). Verify-first: #7 ump×matchup built
  (`ump_k_x_pk`/`ump_k_x_bk`/`ump_bb_x_pbb`); #9 ALREADY EXISTS (matchup_features weights
  batter pitch-type results by pitcher usage → m_xwoba/m_xslg/m_whiff/m_hh); #10 covered by
  vsh_*/same_hand; #8 geometry covered by pull_fence, realized handed factor deferred.
  Batter frame 251→265 cols. Retrained via `--rebuild --select` + set as 2025 dev baseline
  (Option A per user — binding verdict deferred to ship: selection + families + 2026 confirm).
- 2026-07-10 (BvP): Tier-2 #12 batter-vs-pitcher direct history IMPLEMENTED (vectorized path;
  serving+parity deferred to ship). `_bvp_table` + `bvp_n`/`bvp_xwoba_resid`/`bvp_hr_resid`.
  Batter frame 265→268 cols. Rides into the next full `--rebuild --select` retrain alongside
  the param-sweep winners (one retrain, not two — a 3-col sparse addition doesn't move the
  coarse per-head regularization the sweep tunes). See MEMORY [[feature-backlog]].
