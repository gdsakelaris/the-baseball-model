# Feature Backlog — MLB model

Living list of (1) features that are **benched / out of the superset** and should be
unbenched or recreated, and (2) **new feature ideas** to build and expose to feature
selection.

**The process, given selection-as-curation (2026-07-10):** the job of this list is only
to get candidate columns *into the superset*. Whether any head keeps a column is decided
by `feature_select.py` (SHAP stability vote, PI≥0.75 both suites) + the paired-CI eval —
not by hand. So "expose it and let selection vote" is the default disposition for
everything here. Old manual benches are treated as hypotheses to re-test, not verdicts.

**Single-train rider flow (2026-07-13, user question -> adopted for future batches):**
small rider batches do NOT need the two-train superset recipe. Train ONCE on
keep-list ∪ new-candidates (temporarily union the new cols into feature_keep.json),
run `feature_select` to vote on the new columns in that shipping-adjacent context,
and prune only on failure (a column that fails the stability vote is barely used —
the trained model is effectively identical without it, so the same train ships).
Reserve the full two-train superset regeneration for wholesale re-litigations
(era prunes, unbench-everything probes, regime changes like new families/window).

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

11. **Weather beyond wind** — **IMPLEMENTED 2026-07-12** (data-gap batch). Open-Meteo
    (keyless): `Scrapers/scrape_weather.py` → `mlb_weather.csv` (Humidity/Pressure/Precip
    per GamePk, 2015+ archive backfill + daily incremental; former/special venues have
    coords in-script; ballparks CSV now carries Lat/Lon/Roof). Features: `hum_eff`
    (indoor-corrected RH; Dome/Roof Closed → 50%) + `air_dens` (physical air density from
    Temp+Pressure+Humidity — Mexico City 0.91, Coors 0.99, marine parks 1.21) via shared
    `add_weather_derived` (batter + starts + team frames + serving). Serving:
    `1_get_todays_games.py` scrapes the forecast at each park's start hour into
    `todays_games.json`; GUI has editable Humidity/Pressure fields. Precip is in the file
    but NOT a feature (no serving-side forecast wired for it).

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

13. **Catcher framing** — **PARTIALLY UNSHELVED 2026-07-15 → see #35.** The original
    shelf reason (catcher-level serving input the user can't provide) still holds for
    the PLAYER grain, but the team playing-time-weighted aggregate has no serving
    problem — the same dodge team OAA uses — and shipped as `opp/own_cat_frame` in
    the #35 battery wave. Player-grain framing stays shelved (mlb_catchers.csv keeps
    the per-catcher rows if starting-catcher input ever becomes available).

14. **MiLB level-translated priors** — **DATA + TRANSLATION DONE 2026-07-13; main-model
    cols STAGED for the Phase-3 batch** (features.py untouched until then — code-fingerprint
    guard). Scrape: `Scrapers/scrape_milb.py` → `Data/milb_batting.csv`/`milb_pitching.csv`
    (2010+, levels AAA…Rk, Age+League, full counting stats). Translation:
    `Model/milb_priors.py` → `artifacts/milb_priors.joblib` — per-level/class logit offsets
    fit on mover pairs (MLB seasons ≤2024 only), levels kept by paired-weight gate
    (AAA/AA/A+ both sides), serving table = one row per (PlayerId, serve-season) with
    8 translated rates `t_K…t_OUT` + `n_eff` (PA-decayed, ≤ Y−1 seasons ONLY — season
    aggregates leak in-season; date-grain scrape is the designed upgrade). Already consumed
    by the PA model (EB career prior + `b_milb_n`/`p_milb_n`). **Phase-3 rider spec:** join
    `build_all()["bat"]["serve"]` on (PlayerId, Season) into the BATTER frame → 9 cols
    (`milb_t_*` ×8 + `milb_n`), same from `["pit"]["serve"]` into the STARTS frame for the
    pitcher; NaN where no kept-level MiLB in window (GBM imputes); serving path joins the
    same artifact. Targets: everything (thin-history players sit in every head); selection
    votes per head as usual.

### Tier 1 additions — 2026-07-14 review pass (no scrape; all inputs already in `Data/`)

15. **Bullpen-exposure share + quality delta** — `BUILD` — *headline.* Every batter prop is
    secretly a two-pitcher problem: the model prices the batter against the STARTER, but a
    slot-4 hitter gets 1–2 PAs against the pen, and more when the starter is a 4.2-IP guy.
    Spec: `xpa_pen` = expected share of tonight's PAs coming after the starter exits
    (slot × `p_ip_per_start` via league PA-per-inning geometry), then collision columns
    `xpa_pen × (pen_k_bf − ps_k_bf)`, `× (pen_hr_bf − ps_hr_bf)`, `× (pen_h_bf − ps_h_bf)`.
    Starter quality and pen quality are separate mains today; *who the batter actually faces
    in PA 3–5* is a three-way interaction the tree can't carve. Targets: bk/bk2/xbk, hr, bb,
    tb2, hrr2/3, and the atlas says exposure products are already the #1 feature family on
    13/24 heads — this is the same geometry pointed at the late game.

16. **Manager's leash + outing shape** — `BUILD` — the outs head's top features are all
    workload MEANS (`p_np_l3`, `p_ip_per_start`); give it shape and policy: `p_outs_sd`
    (as-of SD of his outing lengths), `p_short_share` (share of last 10 starts ≤ 12 outs —
    opener/bulk/quick-hook detection; outs is bimodal for these pitchers and a mean feature
    splits the difference), `team_st_outs_pg` (the TEAM's as-of average starter outs — the
    manager's leash, distinct from the pitcher's own history). Targets: outs, per, pha, k.
    These are the GBM-side siblings of the queued hazard-v2 work — cheap to ride ahead of it.

17. **Layoff / ramp regime flags** — `BUILD` — `p_days_rest` and `p_np_last` are mains; the
    regimes are thin-support interactions worth handing over explicitly: first start after a
    15+ day gap (IL return / callup — hard pitch cap), `p_np_last < 60` (ramping, short leash
    today), short rest (≤ 4 days) after a 100+ pitch start. Three indicator/product columns.
    Targets: outs, pha, per, k.

18. **Day-after-night + travel legs** — `BUILD` — Part 1 #3's original spec included a
    day-after-night flag; only `g_l7d`/`g_l14d` shipped. Build `day_after_night` (yesterday
    a night game, today a day game — games file has DayNight + dates), `travel_km`
    (yesterday's venue → today's venue great-circle from the ballparks Lat/Lon added in the
    weather batch) and `tz_delta`. Team-grain, merged into the batter frame like park cols.
    Targets: hit/tb2/run and the count means (fatigue-sensitive), total.

19. **Posted-lineup quality gap (B-lineup detector)** — `BUILD` — the model reads each
    batter individually but never asks "is this the A lineup?": posted lineup's mean as-of
    career OBP/SLG minus the team's own `toff_` season norm. Getaway-day and September
    lineups depress totals and flip winners, and the `lu_`/`off_lu_` infrastructure already
    aggregates posted lineups (K-model precedent), so this is a small delta. Targets: total,
    winner — with the Part 1 #4/#5 small-n caveat: the paired-CI read, not the keep-vote, is
    the arbiter on those frames.

20. **Team cluster-luck regression (BaseRuns residual)** — `BUILD` — trailing-30-day team
    runs minus the BaseRuns expectation from components (H/BB/TB/HR, all in the game logs):
    sequencing luck that regresses. `toff_r_pg` carries the luck fused with the skill; the
    residual separates them — the team-grain sibling of `hit_luck`/`p_hit_luck`. Targets:
    total, winner.

21. **Ump run environment** — `BUILD` — selection (correctly) routed `ump_k/bb` to the K/BB
    heads only, so the totals model never sees the ump at all. One column: shrunk as-of
    runs-per-game with this HP ump (same `_ump_shrink` idiom). Targets: total, per.

22. **High-leverage arm availability** — `BUILD` — `pen_np_l3` is aggregate fatigue;
    availability is arm-specific. From the per-reliever game logs: how many of the
    opponent's top save/hold arms pitched BOTH of the last two days (unavailable tonight, by
    bullpen convention), quality-weighted. Pairs with #15: exposure says how many late PAs,
    this says against whom. Targets: total, winner, run/rbi/hrr2/3.
    **AUDIT FINDING 2026-07-14: never actually built** — the finish batch skipped it (zero
    availability columns in features.py; FINISH_PLAN's Phase-1 build list omits it) despite
    the "all #15–32 built" record. USER 2026-07-14: rebuild in the audit wave alongside the
    own-pen leash item (spec in Model/AUDIT_BUILD_SPECS_0714.md).

23. **Starter venue split** — `BUILD` — batters got `vloc_`; starters never did. As-of
    home/road ERA / K / HR splits for the opposing starter (`pvloc_`), same shrunk idiom.
    Targets: starter heads, hr/tb2 (road-vulnerable starters), total.

24. **Doubleheader flags** — `BUILD` — `is_dh` + `dh_game2`: two GamePks, same teams+date
    in the games file. Game 2 means backup catchers, spot starters, tired pens — exposure
    and quality dilution the frames currently can't see. Serving: the slate already handles
    per-game times/weather (doubleheader-handling design), so the flag is knowable. Targets:
    exposure-sensitive batter props, outs, total.

### Raw-pitch archive additions — 2026-07-14 (free re-agg via `--from-raw`, ~4 min)

25. **Pitch sequencing + count-state splits, pitcher side** — `BUILD` — promoted from the
    07-12 log's "remaining ideas" line so they stop living only in a log entry: (a) 0-2
    waste rate and ahead/behind usage shift (does he have a putaway plan, does his mix
    collapse when behind); (b) back-to-back pitch-class transition shares (FF→SL rate — a
    tunneling proxy) as `pd_` columns. The batter two-strike side (`bd_tswh`) already
    exists; this is the pitcher half. Targets: k, bk/bk2/xbk, per.

26. **In-game velo fade slope** — `BUILD` — per-start OLS slope of FF/SI velo against pitch
    number, decayed across starts (`pd_fbv_fade`). Stamina signature distinct from
    `pd_fbv_sd` (spread, not trajectory) — predicts late-outing collapse. Targets: outs,
    per, pha.

27. **Movement / IVB stuff axes** — `BUILD` — **archive VERIFIED 2026-07-14**: the raw
    parquets carry `pfx_x`/`pfx_z`, `api_break_z_with_gravity`/`_x_arm`, `spin_axis`,
    `release_spin_rate`, `release_extension`, `arm_angle` — movement is FREE via
    `--from-raw`, no scrape. Build `pd_ivb_d` (FF induced vertical break, decayed) + a
    ride-vs-flyball-swing collision with `bd_fbwh`/`bip_air`. Velocity is in the frame
    everywhere; movement has never been. (Same check found pitch-level `bat_speed`/
    `swing_length` in the archive from the bat-tracking era, and `n_thruorder_pitcher` —
    noted for the ~2027 bat-tracking unshelf, not for now.)

33. **Damage-on-contact splits by velo band / pitch class (v8 wave)** — **BUILT
    2026-07-15** (user ask, from the "batter velocity-band splits" review item; wired
    same-day, awaiting the next superset retrain). CORRECTION recorded: the *whiff*
    half of the velo-band axis already shipped in v4 (`bd_fb95wh` + `bat_velo_matchup`
    kept on 14 heads; `velo_band_whiff` on 4) — what was actually missing was the
    *contact* half: the band/class cells carried only n/sw/wh, no damage. v8 re-agg
    (`--from-raw`, verified byte-identical on every pre-v8 column) added xwOBA-per-BBE
    sums (`con_/fblo_/fbmid_/fb95_/brk_/off_` bip+xw; fbk = remainder) + the 2K×95+
    cell (`ts_fb95_*`). New features: batter `bd_{band,class}xw_c/_d` (7 PD names,
    K=60 BBE, priors measured 07-15: .389/.385/.372 bands, .381/.355/.343 classes,
    .187 tsfb95wh), starter `pd_*xw_d` allowed-side mains, opponent `p_*xw_d` on the
    batter frame, collisions `bat_velo_damage`/`velo_band_damage`/`arsenal_damage`/
    `mix_fb95xw`/`ts_fb95_matchup` (damage collisions ride the CAREER reads — a
    median batter season has ~18 BBE vs 95+, decayed reads are shrink-dominated),
    and the lineup grain: `lu_{fblo,fbmid,fb95}wh` + `lu_*xw` means with
    `lu_velo_k`/`lu_velo_dmg`/`lu_ars_dmg` (the lu_ars_whiff velocity/damage
    siblings — the banded lineup collision the K/pha/per heads never had).
    Deliberately NOT built (recorded so they aren't re-proposed blind): hard-hit-per-
    band (xwOBA subsumes EV+LA), zone×band whiff cells (sparsity), per-band foul-share
    (now derivable free from sw−wh−bip if ever wanted), band-damage slope main
    (collisions carry it), game-grain lu damage collision (needs #32's batter→game
    plumbing first — natural rider WHEN #32 builds).

### Composite cluster (2026-07-14, user-confirmed)

Rationale: hand-built collisions are this model's best feature class (`xpa_x_*` #1 on
13/24 heads, `lu_mix_k` #1 on k); these five extend the proven templates to frames and
heads that never received them. Per the accept policy: **no pre-declared targets — every
head has room to improve**, every column enters the superset, and selection votes per
head. The "Targets" notes below only name where each column physically lives, not who is
allowed to benefit.

28. **`per` conversion chain** — `BUILD` — earned runs are sequential: baserunners
    allowed × damage-on-contact allowed, e.g. `(pc_h_bf + pc_bb_bf) × pbipd_xwoba`
    (+ decayed sibling). Same template as `rbi_conv`/`run_opp`, which both shipped —
    nobody ever pointed it at the starter frame. Targets: per, pha, total.

29. **Ump × lineup for the K head** — `BUILD` — `ump_k_x_pk` (ump × pitcher) exists;
    `ump_k_pct × lu_k_sh` (ump × the actual lineup's K-proneness) doesn't. Completes a
    proven pattern. Targets: k, xbk.

30. **Air density × lineup air profile at game grain** — `BUILD` — `air_fly` exists only
    at batter grain; the totals model gets `air_dens` as a main effect but can't cross it
    with how fly-ball-heavy tonight's lineups are (lineup mean `bip_pullair`/1−`bip_gb`,
    posted-lineup infra). Game-grain sibling of a shipped feature. Targets: total, winner.

31. **Form-weighted exposure** — `BUILD` — the `xpa_x_*` products use CAREER rates; decayed
    variants (`xpa_slot × d_hr_pa_sh`, `× d_tb_ab_sh`, `× d_k_pct_sh`) price current form ×
    opportunity. One calculated extension of the single most successful template. CAVEAT:
    most exposed to credit-splitting — expect selection to keep either the decayed or the
    career version per head, not both; that's the vote working, not a failure.

32. **Cross-grain arsenal collision (lineup class-whiff × opposing starter usage)** —
    `BUILD` — promoted from the 07-12 full-surface log's deferred item (a). The
    highest-value unbuilt composite on the board: the posted lineup's aggregate
    class-whiff (`lu_brkwh`/`lu_offwh`/`lu_fbwh` exist) crossed with the OPPOSING
    starter's actual usage mix — at TEAM grain, for the game heads. Needs the
    batter-frame → game-frame plumbing that deferred it; that plumbing is the cost, the
    signal design is proven (`lu_mix_k` is the k head's #1 feature). Targets: total,
    winner (small-n caveat applies).

**Spirit check on #15–32:** odds stay grading-only (phase rule). (Historical note: this
line originally kept catcher framing/pop-time SHELVED per #13's serving-input problem;
2026-07-15 the TEAM-grain version shipped as #35 — the player-grain shelf stands.)

### Battery + IL wave — 2026-07-15 (user ask; new scrapes; BUILT same-day)

34. **IL transaction stints** — **BUILT 2026-07-15** — `Scrapers/scrape_transactions.py`
    (statsapi transactions, sportId=1, "injured list" + pre-2019 "disabled list") →
    `mlb_il_events.csv` (raw, incremental) + `mlb_il.csv` (paired stints: PlaceDate,
    ActDate, StintDays, IL60, Rehab; 7,642 stints 2015–2026, ~500–980/yr). Features
    (shared `_il_asof` / `il_feats_from_stint`, allow_exact_matches — roster moves are
    announced pregame): `il_ret_days`/`il_last_len`/`il_ret21`/`il_szn_days`/`il_rehab`
    on the batter frame (own) + `p_il_*` for the opposing starter, and `p_il_*` on the
    starts frame (own) — the layoff CAUSE the #17 gap flags can't see (IL stint vs
    skipped start vs All-Star break), NaN-gated at IL_RET_MAX=365d. Targets: everything
    (ramp/regression is head-agnostic); selection votes per head.

35. **Running-game defense + team-grain battery (catchers)** — **BUILT 2026-07-15** —
    `Scrapers/scrape_catchers.py` (Savant catcher-framing 2015+, catcher-throwing
    2016+, poptime 2015+; Savant labels historic teams with CURRENT franchise abbrevs —
    mapped current-abbrev → franchise id → season abbrev) → `mlb_catchers.csv`
    (player grain, shelved-for-now consumer) + `mlb_catchers_team.csv` (playing-time-
    weighted battery per Year+Team: FrameRV_pt = framing runs/2000 called pitches,
    CSAA_att, PopTime; CSAA NaN 2015 by design). Prior-season serving like team OAA.
    Features: batter frame `opp_cat_frame/csaa/pop` + `frame_x_take` (framing × 2K
    taker, ump_k_x_take's battery sibling) + `sb_cat_env` (sb_chain × pop centered on
    the PER-YEAR league mean (PopC, computed in load_raw) — pop drifted ~0.05s
    2015→2025, comparable to the team spread, so a global center would let era drift
    flip the sign; same regime-aware centering as sb_chain_env's lg_sb27_prior —
    the catcher half sb_chain_env never had); starts frame
    `own_cat_frame/csaa/pop` + `frame_x_edge` (framing × edge share) + `run_cat_x`
    (lu_sb × catcher stop — run_game_x's catcher half). PA-sim steal-layer battery
    modulation: **BUILT 2026-07-15 AM (user moved it ahead of the Phase-3 re-sweep —
    the "engine change mid-forward-record" concern was moot with serving guard-down
    pre-chain, and the post-chain re-sweep re-fits SIM_BLEND weights regardless).**
    `pa_sim.build_battery_tables` (prior-season, Season=Year+1, TEAM_RENAMES-aliased):
    attempt rate × opposing starter's SB/27 EB-shrunk (BATT_K_UNITS=10 27-out units)
    ratioed to lg SB/27; success rate − opposing team's CSAA_att (2015 NaN imputed
    via the build-time-fitted CSAA~PopC OLS slope, −0.80/s), clips 0.6–1.7 / ±0.12.
    Shared `battery_context`/`battery_adjust` applied identically in pa_backtest +
    pa_serve (engine contract untouched); `STEAL_BATTERY=False` = exact runner-only
    revert; unit-tested (TestBattery); banked into pa_sim_tables.joblib. Graded at
    the Phase-3 steal re-sweep.

---

## Part 3 — New prediction HEADS (not features)

Distinct from Parts 1–2: these add *outputs*, not columns. The single-train rider flow
above does **not** obviously apply (see the open question in #H1) — settle that at
execution time, not now.

### H1 — Four deep batter binary heads: `bk3`, `tb3`, `tb4`, `hrr4` — `READY, BATCHED`

**What.** Dedicated binary heads for 3+ K, 3+ TB, 4+ TB, 4+ H+R+RBI — the four batter
thresholds the board does not currently sell. Targets are one-liners next to their
siblings in [features.py:2605-2627](features.py#L2605-L2627):
`y_bk3 = (SO >= 3)`, `y_tb3 = (TB >= 3)`, `y_tb4 = (TB >= 4)`, `y_hrr4 = (hrr >= 4)`,
plus four `PROPS` entries in [train.py:429](train.py#L429).

**Why binary and not the free path.** The `xbk`/`xtb`/`xhrr` count heads ALREADY price
these four lines — `fit_line_cals` banks a calibrator for every line in `COUNT_HEADS`,
and `predict.BAT_COUNT_COLS` reads none of them. Shipping those banked calibrators is a
~20-minute display change with no retrain. **Rejected 2026-07-13 (user):** it would put
a count-priced column (`H+R+RBI 4+`) directly beside binary-priced siblings
(`H+R+RBI 2+/3+`) in the same market — two methodologies in one family — and
`Model/count_vs_binary.py` had *just* shown the count-calibrator method is the WORSE of
the two at every threshold where both exist. Free is not a reason to ship the method
that loses.

**The acceptance bar is already measured** (`count_vs_binary.py`, held-out, both suites).
Each new binary head must beat its banked count calibrator:

| new head | line | logloss to beat (2025 / 2026) | AUC to beat | base rate |
|---|---|---|---|---|
| `bk3`  | `xbk`>2.5  | .17436 / .17793 | .694 / .689 | 4.5% |
| `tb3`  | `xtb`>2.5  | .50063 / .49840 | .599 / .597 | 20.7% |
| `tb4`  | `xtb`>3.5  | .40730 / .40333 | .609 / .613 | 14.4% |
| `hrr4` | `xhrr`>3.5 | .45355 / .45459 | .602 / .595 | 17.4% |

**Expect a MIXED verdict, and that's fine.** The binary-beats-count result was measured
at base rates 22–62%; these sit at 4.5–21%. Thin positives are exactly the regime where
pooling into a count mean starts to win (it's why starters price their own lines). If
`bk3` (4.5%) loses to its calibrator, ship 3+ K count-priced — a mixed board for a
*measured* reason is defensible; the accidental one was not.

**Open question to settle at execution:** a new head has no `feature_keep.json` entry, so
`_apply_keep` leaves it unrestricted — it trains on all 366 cols on the first train. Does
the rider-flow argument ("a column that fails the stability vote is barely used, so the
model is effectively identical without it") extend to a whole new head, letting one train
ship? Or do new heads need the second train to actually apply their keep-list? Decide
before starting; it is the difference between ~91 min and ~3 h.

**Trap.** `feature_select.py --write` has no per-head flag — it regenerates and clobbers
`feature_keep.json` for ALL heads ([feature_select.py:295-297](feature_select.py#L295-L297)).
Running it naively re-selects the 24 existing heads against current data, smuggling a
large uncontrolled change in with a small one. Selection for H1 must be **merged**, not
written: take only the 4 new heads' keep-lists and leave the other 24 byte-identical.

**Downstream wiring once the heads exist** (all small, all known):
`predict.PROP_COLS` / `BAT_HEADERS` / `BAT_ORDER` / `PCT_COLS` / `GLOSSARY`;
`4_grade_results.BAT_EVENTS` (`hit_rate_report.py` imports it, so it follows for free);
`prop_rankings.BIN_NAMES` **and** `BINARY_OWNED_LINES` — note the x-rows currently grade
*only* these four deep lines ([prop_rankings.py:127](../Tools/prop_rankings.py#L127)), so
promoting them to binary heads leaves `BATTER_X` with no uniquely-priced line to grade;
decide what those rows measure then. Plus `Props.txt` (55 → 59 props).

### H2 — First-5-innings (F5) outputs from the PA-sim — `IDEA, NEEDS GRADING DATA`

The sim already walks games PA-by-PA with the starter-hazard hook; F5 total runs and F5
lead/ML are readable off the SAME trajectories with zero new training — pure `pa_serve`
surface, not a GBM head. Books post F5 ML and F5 total lines, and F5 is where the sim's
starter modeling is strongest (no bullpen chain to get wrong). Blockers: (1) grading needs
per-inning linescores, which `mlb_games` doesn't carry — small scrape; (2) the sim's blend
weights are a week old and the steal layer shipped at w=0 — let the game-head blend earn
trust through a couple of weekly reads before selling a sim-only market. Parked, not spec'd.

### H3 — `triple` binary head (1+ triple) — `READY` (2026-07-14, user ask)

Completes the hit-type family: `single` and `double` heads exist; triple doesn't. Target
is a one-liner next to `y_1b`/`y_2b` in features.py (`y_3b = (3B >= 1)` — the game logs
carry a `3B` column, verified), one `PROPS` entry ("1+ triple"), and the same downstream
wiring list as H1 (predict cols/headers/glossary, `4_grade_results.BAT_EVENTS`,
`prop_rankings`, Props.txt → 60 together with H1's four). `PLATT_CAL` covers it
automatically via `set(PROPS)` — which matters here: measured base rate is **1.21% per
batter-game** (2025), the thinnest binary on the board (double 14.2%, bk3 4.5%), exactly
the regime where isotonic memorizes the tail. No banked count-calibrator bar exists (no
count head prices triples), so acceptance = the standing gates: CI-clear logloss edge over
base rate, honest ECE, no harm elsewhere. Its signal lives in speed + park geometry
(`bat_sprint`, `bat_hp1b`, `bat_leg_hits`, fence/CF distances — all already in the
superset); add a `park_3b_pg` as-of park factor rider (`park_2b_pg` idiom) so venue
triple-friendliness (deep gaps, quirky walls) is explicit. Expect selection's MIN_KEEP
floor to bind (thin positives → weak votes) — designed behavior, not failure.

### H4 — `rbi2` + `run2` binary heads (2+ RBI, 2+ runs) — `READY` (2026-07-14, user ask)

Deeper thresholds for two existing props: `rbi` (1+) and `run` (1+) heads exist; the 2+
lines the books post don't. Targets are one-liners next to their siblings
(`y_rbi2 = (RBI >= 2)`, `y_run2 = (R >= 2)`), two `PROPS` entries. Measured base rates
(2025): **rbi2 9.31%, run2 7.23%** — thicker than bk3 (4.5%) and sb (6.5%), healthy
targets. The hrr2/hrr3 precedent says 2+ composite thresholds train fine on these frames;
the run-production context groups (`ctx_`, `rbi_conv`, `run_opp`, exposure products) are
exactly their signal and already in the superset. No banked count-calibrator bar exists
(no count head prices RBI or R alone), so acceptance = the standing gates. `PLATT_CAL`
automatic via `set(PROPS)`. Same downstream wiring list as H1/H3.

### H5 — `team_total` head (per-team total runs) — `READY` (2026-07-14, user ask)

A new OUTPUT head — the underlying GBM already exists and is already evaluated:
`team_runs_model` predicts per-team
means at serving ([predict.py:748](predict.py#L748), `mu_away`/`mu_home`), the PA-sim
blend already blends them per side (SIM_BLEND `score` .35 → `x_away`/`x_home`), and
metrics.json tracks its MAE (2.52 vs 2.60 baseline, 2026). Today those means are only
summed into the game total and fed to the winner blend — trained but sold nowhere (the
orphan-count-lines lesson, again). Work to make it a head: (1) measure a TEAM-level
NB dispersion on the calibration year (game-total recipe — the 2.28 game dispersion does
NOT transfer, team variance is its own number); (2) NB P(over) for team-total lines
(2.5–5.5 typical); (3) display the blended per-team means in the workbook/GUI; (4) grade
from final scores (`mlb_games` AwayScore/HomeScore — no new data); (5)
`prop_rankings`/`Props.txt` entries; (6) evaluate_deep gains a team-total line read next
to the game-total one.

### H6 — Four new batter count heads: `xh`, `xrun`, `xrbi`, `xbb` — `READY` (2026-07-14, user ask)

Completes the expected-stat-line: the existing x-heads (`xbk`/`xtb`/`xhrr`) are exactly
the batter stats with heavy count mass above 1; these four finish the set (xH / xR /
xRBI / xBB / xTB / xSO / xHRR per batter). Measured 2025: H P(2+)=19.0% var/mean 0.95;
R 7.2% / 1.02; RBI 9.3% / **1.61**; BB 4.0% / 1.02. So: `xrbi` gets the Tweedie 1.3
objective (xhrr pattern); `xh`/`xrun`/`xbb` stay Poisson. Four `COUNT_HEADS` entries
(frame="bat"), targets are the raw per-game stats.

**Means only — lines stay binary-priced** (07-13 shoot-out verdict): `fit_line_cals`
banks per-line calibrators for free (xh 0.5/1.5/2.5, xrun 0.5/1.5, xrbi 0.5/1.5/2.5,
xbb 0.5/1.5) but NONE ship as prices — banked for future binary-vs-count shoot-outs,
the H1 lesson institutionalized. Acceptance = standing count-head gates (MAE vs naive
baseline, honest dispersion). Bonus diagnostic: Σ lineup `xrun` vs the `team_total`
mean — a batter-grain vs team-grain coherence read evaluate_deep has never had.

**Declined + recorded** (so they aren't re-proposed): `xhr`, `xsb`, and hit-type counts
(x1b/x2b/x3b) — rare events (P(2+) ≤ 0.7%) where E[X] ≈ P(X≥1): the binary already IS
the mean; a count head would re-learn it with extra variance.

---

## Log
- 2026-07-15 (adjudications, latest): user picked (1) steal-layer battery modulation →
  Phase-3 rider (spec at FINISH_PLAN Phase-3 step 2; do NOT build before the blend
  re-sweep) and (2) franchise-rename alias fix NOW → `TEAM_RENAMES`/`_alias_renamed_teams`
  in features.load_raw duplicates pre-rename rows (OAK→ATH) for the two team-keyed
  prior-season files (mlb_oaa + mlb_catchers_team, PopC computed pre-alias so league
  means don't double-count) — heals the ~3.3% A's-2025 NaN for `opp_cat_*` AND the
  pre-existing `opp_oaa` gap; training-frames-only change (2026 serving already found
  ATH-2025 rows), full franchise-id refactor declined as disproportionate.
- 2026-07-15 (battery + IL wave, later): #34 + #35 BUILT same-day (user ask; #13
  partially unshelved at team grain). Two NEW scrapers + four NEW Data files wired
  into update_all JOB_FILES + validate_data SPECS (events fresh_days=10 — IL moves
  aren't game-guaranteed). Scrape bugs caught in verification: pre-2019 stints say
  "disabled list" not "injured list" (2015–18 were empty until the regex fix), and
  the team rollup's plain sum turned 2015's all-NaN CSAA into fake 0.0 (min_count=1).
  Frames rebuilt; selection NOT regenerated (columns train at the next superset
  retrain, same as #33).
- 2026-07-15 (v8 damage-on-contact wave): #33 BUILT same-day (user ask). Scraper v8
  sums + `--from-raw` re-agg (~2 min; every pre-v8 column verified numerically
  identical vs backup, so the 07-14 superset's serving inputs are untouched; archive
  and CSVs both ended 07-11 — All-Star break — so nothing was in flight). features.py
  + predict.py wired both paths (PD lists drive train/serve shared code; `_LU_COLS`
  mirror extended). frames.joblib REBUILT with the new columns (pre-v8 cache kept as
  `artifacts/frames.pre_v8_0715.joblib`); selection NOT regenerated — the new columns
  enter the superset at the user's next retrain (user directive: features now,
  retrain later). NOTE: tree now differs from baseline_code_fp → expect the 06:00
  daily to go scrape-only until the next baseline set.
- 2026-07-14 (feature-gap audit, late eve): full coverage audit — 11 finder lenses
  (6 data-source × 5 head-family) + 49 adversarial verifications over every Data file
  × all 36 heads. Verdict: no first-order axis missing; **33 build items approved by
  the user (ALL 33 + #22 rebuild + park_vmr column with 2025-only gated pricing fit +
  opp_penhl_share share-only rider)** — full line-anchored specs in
  Model/AUDIT_BUILD_SPECS_0714.md (prune post-batch). Notable findings: the xpa_x_*
  family had NO bb/sb members (new xbb head shipped with zero exposure coverage);
  rbi2/run2/hrr4 shipped without the threshold histories their hrr2/3 siblings have;
  the triple head lacked realized-3B rates; the STARTS frame was systematically
  shortchanged (no lineup damage/OBP view, no own-pen state, no hit-luck mirror, no
  wind carry, no bio, no unearned-run axis, no pitch-economy ratio); spray encoding
  one-sided (no oppo-air); team frame missing the entire walk channel; #22 found
  never-built (corrected above). NEGATIVE result worth keeping: mlb_linescores is
  EMPTY as a GBM feature source — all six inning-grain traits fail persistence
  (e.g. starter 1st-inning bleed y2y r=−0.001) — it stays a grading/sim-validation
  asset. ~20 further dead-ends killed with measured numbers (see the specs doc) so
  they are not re-proposed. 2 rejects → Decline ledger #5–6; 1 partial → #7.
- 2026-07-14 (evening, later): Part 3 #H6 added (user picked all four: xh/xrun/xrbi/xbb —
  completes the expected-stat-line; xrbi Tweedie per measured var/mean 1.61; means only,
  banked-not-shipped line cals; xhr/xsb/hit-type counts declined with the rare-event
  rationale recorded). Board: 24 → 36 output heads (35 trained GBMs + team_total).
- 2026-07-14 (evening): Part 3 #H4 (`rbi2`/`run2` — base rates measured 9.31%/7.23%,
  standing gates, no banked bars) and #H5 (`team_total` head — the team-runs GBM is
  already trained/evaluated/sim-blended; head-ification = team-level NB dispersion +
  lines + display + grading, user directed it counts as a HEAD) added per user ask.
  Board: 24 → 32 output heads (31 trained GBM heads; team_total shares the team-runs
  GBM with `total`).
- 2026-07-14 (later still): Part 3 #H3 `triple` head added (user ask — completes the
  single/double/triple family; base rate measured 1.21% per batter-game 2025, thinnest
  board binary; `park_3b_pg` rider spec'd). Composite-cluster rationale reworded: the
  "headroom heads only" framing REMOVED per the accept policy (user correction — all
  heads have room to improve; no pre-declared targets; selection votes per head).
- 2026-07-14 (later, composites + archive verify): #28–32 composite cluster added
  (user-confirmed): per conversion chain, ump × lineup K, air × lineup
  air profile, form-weighted exposure, cross-grain arsenal collision. #27 flipped
  VERIFY→BUILD: raw parquet archive confirmed to carry pfx/api_break/spin/arm_angle —
  movement features are free re-agg, no scrape. Archive also carries pitch-level
  bat_speed/swing_length (for the ~2027 unshelf) and n_thruorder_pitcher. Checked
  mlb_games.csv: final scores only, NO linescore — #H2's grading blocker confirmed real.
- 2026-07-14 (Claude review pass): Part 2 extended with #15–27 + Part 3 #H2. Themes:
  (a) the **second-pitcher problem** — bullpen exposure share (#15) and arm-specific
  availability (#22): the model prices batters against the starter but late PAs come
  against the pen; (b) **outs-head shape/policy** (#16, #17, #26) — outing-length
  dispersion, opener detection, manager leash, layoff/ramp regimes, in-game velo fade:
  GBM-side siblings of the queued hazard-v2; (c) **schedule/lineup context the frames
  never see** (#18, #19, #24) — day-after-night + travel, B-lineup detection, doubleheader
  flags; (d) **regression + environment signals at team grain** (#20, #21, #23) — BaseRuns
  cluster-luck residual, ump run environment for totals, starter venue splits; (e)
  **raw-pitch archive round 2** (#25–27) — pitcher-side sequencing/count states, velo fade,
  movement axes (verify archive schema before treating as free). All are
  expose-and-let-selection-vote per the 2026-07-10 process; winner/runs targets inherit the
  Part 1 #4/#5 small-n caveat. Catcher framing stays shelved; odds stay grading-only.
- 2026-07-13 (deep batter heads): Part 3 #H1 added. `Model/count_vs_binary.py` (new,
  read-only) adjudicated count-vs-binary pricing: binary heads WIN at all 5 overlapping
  thresholds, the logit blend fails same-sign, and a straight flip to count pricing is
  CI-clear harm. Verdict = keep the asymmetry. Fallout: the 4 banked-but-unshipped count
  lines are real (all beat base rate both years, coherence-checked) but must ship as
  binary heads, not calibrators. User BANKED them pending a batched retrain.
- 2026-07-13 (MiLB priors): Tier-2 #14 added — scrape + translation layer built and
  live in the PA model; main-model columns staged as Phase-3 riders (spec in #14).
- 2026-07-12 (data-gap batch): new external data + features, all through the shared
  train/serve helpers (parity by construction):
  * **Weather** (#11 above) — humidity/pressure scrape + `hum_eff`/`air_dens`.
  * **Zone-split whiffs** — `oz_wh` added to scrape_pitches.py (both files) →
    `bd_zwsw_c/d` (batter in-zone whiff per swing = 1 − zone contact) and `pd_zwsw_d`
    (starter in-zone stuff). Full pitch re-backfill 2015+.
  * **Elite-velo buckets** — `fb95_n/sw/wh` (batter file) → `bd_fb95wh_c/d`, plus
    `p_fbv_d` (opposing starter decayed FF/SI velo in the batter frame) and the
    interaction `bat_velo_matchup` = bd_fb95wh_d × (p_fbv_d − league).
  * **TTO** (old backlog want, no scrape needed) — `_tto_table` from statcast_bip AtBat
    ranks → `p_tto_decay` (shrunk 3rd-vs-1st xwOBA-on-contact allowed), batter + starts
    frames + serving `Stores.tto`.
  * **Player-level OAA** — scrape_oaa.py now also writes `mlb_oaa_players.csv` →
    `_lineup_oaa_table`: the ACTUAL posted lineup's mean prior-season OAA + IF/OF splits
    (`opp_def_p_oaa/if/of` batter side, `own_def_p_oaa/if/of` starter side) + profile
    interactions `bip_gb_def_if`/`bip_air_def_of`.
  * **Baserunning run value** — new `scrape_baserunning.py` → `bat_brr` (total) +
    `bat_brr_xb` (extra-base rate/opportunity), prior-season.
  Second wave (same batch, user asked to maximize the batter six —
  hit/hits2/rbi/run/tb2/hr — before the one training chain):
  * `ctx_ahead_brr` — mean prior-season XB-advancement of the two hitters AHEAD
    (whether the runners he drives in can take the extra base; RBI sibling of
    ctx_ahead_obp) + `ctx_run_conv` = own bat_brr_xb × ctx_behind_slg (run
    conversion once aboard).
  * `bip_ld`/`bipd_ld`/`bip_pu` (+ `pbip_*` starter-allowed) — line-drive and
    popup shares from the BIP file (BBType was only carved into gb/pullair
    before); priors measured 0.246/0.071, ld also 90-day decayed.
  * `air_porch` = (1.165 − air_dens) × (330 − pull_fence), `air_fly` =
    (1.165 − air_dens) × bip_pullair — thin-air carry pointed at the batters
    it actually helps.
  Third wave (same batch, "squeeze everything"):
  * `bip_flyd`/`bipd_flyd` + `pbip_*` — mean sea-level-adjusted FLY-BALL distance
    (`DistAdj` col on the BIP frame at load; prior 315 ft K40): the UNCENSORED power
    measure — hrq_* only sees the HR log (each batter's best contact).
  * `p_hit_luck` — starter's last-5 hits per contacted PA (p5_h_bf/(1−k−bb); p5_bb_bf
    added both paths) minus `pbipd_xba` (xba joined BIP_DECAYED): starter BABIP
    sequencing luck, due to regress. Batter-side hit_luck's pitcher sibling.
  * `bat_leg_hits` = (bat_sprint − 27) × bip_gb — legs beat out grounders.
  * `opp_def_uer` in the BATTER frame (was game-frame-only): error-proneness the
    range-based OAA misses.
  * `p_zone_d` (starter decayed zone share, from pdo dk_z_n — works pre-backfill) +
    `zone_whiff_matchup` = centered bd_zwsw_d × centered p_zone_d.
  All in the supersets; `feature_select --write` + paired CI decide keeps per head.
  **Post-backfill auto-populating:** bd_zwsw/pd_zwsw, bd_fb95wh, bat_velo_matchup,
  zone_whiff_matchup.
- 2026-07-12 (later same day): **scrape-schema v3 IMPLEMENTED** — user chose to
  restart the backfill rather than defer. Both pitch dailies now carry fb95 (both
  sides), brk/off pitch-class buckets (n/sw/wh), edge_n (shadow band), fp_n/fp_sw/
  fp_s (first pitch). Features both paths: bd_{brkwh,offwh,fbwh,fpsw}_{c,d},
  p_{brk,off,edge,fps}_d (opposing starter), arsenal_whiff (usage-weighted class
  collision), brk/off/fp_matchup (centered products); starts pd_ mirrors +
  lu_{brkwh,offwh,fbwh} + lu_ars_whiff. `--backfill` now archives raw pitches to
  Data/raw_pitches/*.parquet and `--from-raw` re-aggregates from disk — future
  schema changes are FREE (no more 6-h downloads). Remaining raw-pitch ideas when
  wanted: per-count leverage splits, pitch-sequencing (tunneling) pairs, velo
  distribution shape (p95-p5), release-point consistency.
- 2026-07-12 full-surface pass: batch signals carried to TEAM grain
  (opp_def_p_oaa/if/of, off_lu_brr/_xb, opp_ps_tto_decay — both paths).
  Deferred: (a) offense-side arsenal collision at team grain (lineup class-whiff
  x opp starter usage — needs batter_frame -> game_frame dependency, real
  plumbing); (b) WINNER widening experiment (win_feature_cols documents real
  overfit harm from past widenings on the ~10k-row frame — only as a deliberate
  paired-CI experiment, maybe with heavier regularization).
- 2026-07-12 later eve: **v4+v5 IMPLEMENTED same day** (user pulled them forward
  pre-retrain). v4 graded velocity bands (<92 / 92-95 / 95+ FF/SI: bd whiff
  splits, p/pd banded usage, velo_band_whiff collision). v5 count leverage +
  dispersion (user picked 3 + added 3-2): two-strike ts_* (bd_tswh, pd_tswh,
  ts_matchup), full-count f32_* (bd_f32b walk conversion, pd_f32z/f32b), fb_v2
  → pd/p_fbv_sd (velo spread = fatigue/consistency), rp_* sums → pd/p_rel_sd
  (release scatter = command/injury proxy). All via ONE `--from-raw` re-agg
  (~4 min, 12 seasons) — the parquet archive already paid for itself. Remaining
  raw-pitch ideas: pitch-sequencing (tunneling) pairs, other count states
  (0-2 waste, ahead/behind splits).
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

## Decline ledger (pending user re-evaluation)

POLICY (user directive 2026-07-14): nothing gets declined unilaterally. Every
would-be decline is surfaced to the user with a recommendation + reasoning
before it is treated as settled, and every past decline lives here until the
user re-adjudicates it. Items below were auto-declined by Claude during the
finish batch and are OWED a proper decision.

1. **Selection: correlation pre-cut** (declined 2026-07-14, 1F design).
   Auto-drop one column of each |rho|>threshold pair before training/voting.
   Reasoning at decline: irreversible pre-filtering can remove columns the
   three families use differently; the co-failure REPORT covers the same
   redundancy concern reversibly. Claude rec: keep declined — report + user
   adjudication dominates it.
2. **Selection: time-block subsampling stability votes** (declined 2026-07-14,
   1F design). Re-run selection on temporal subsamples; keep only columns
   stable across blocks. Reasoning: multiplies selection cost several-fold;
   shadow-calibrated eps already gives a measured noise floor. Claude rec:
   revisit as a one-off batter-frame experiment IF keep-lists churn heavily
   across future chains; otherwise keep declined. 07-14 update: the revisit
   trigger is now OPERATIONALIZED — feature_select prints and persists a
   per-head keep-diff vs the newest feature_keep*.bak
   (artifacts/selection_report.json § keep_diff), so the churn evidence
   exists after every regen; Phase-5 step 4 reads it.
3. **Selection: LGBM-only voting** (declined 2026-07-14, 1F design). Use only
   LightGBM SHAP for votes. Reasoning: the shipped ensemble is 3-family;
   XGB/CatBoost keep genuinely different columns and deserve votes. Claude
   rec: keep declined.
4. **Shrinkage-prior leakage fixes** (declined 2026-07-14; caveat comment at
   features.SHRINK instead). Three rigorous options: (a) training-era-only
   priors, (b) sequential as-of priors, (c) external pre-period priors.
   Reasoning: priors are second-order centering constants; recomputing them
   mid-batch changes every historical feature and breaks comparability with
   the Phase-0 baseline. Claude rec: post-batch dedicated experiment, option
   (b) sequential as-of is the principled one — expected effect small, but it
   would close the README's known-limitation note.
5. **rbi_traffic_vsp** (audit reject 2026-07-14, surfaced to user in the audit
   report). Teammate-supply × tonight's-starter-leak products: the residualized
   interaction term measured flat zero (±0.001, within noise, U-shaped
   quintiles) on every claimed head — ~99% linearly reconstructable from mains
   the trees already see. The SAME probe validated three shipped composites
   (rbi_conv +0.0070, run_opp +0.0257, xpa_x_rbi +0.0093 partial corr), so the
   method is trusted. Claude rec: keep declined.
6. **xpa_team_turnover** (audit reject 2026-07-14, surfaced to user in the
   audit report). Team PA/G exposure adjustment: premise overstated the spread
   ~2.5× (measured ±1.6% SD); 72–81% already carried by toff_r_pg (residual
   below selection's noise floor, xpa_bat already embeds realized turnover);
   the home/away slot pooling bias is a uniform ~0.17-PA additive offset —
   additively separable, the easiest GBM carve. Salvage crumb if re-opened:
   expose plain team_pa_pg as a single main. Claude rec: keep declined.
7. **opp_pen_br_era** (USER-adjudicated 2026-07-14, audit gray item). Bridge-pen
   ERA measured R²=0.95 linearly recoverable from the two exposed mains
   (pen_era + pen_hl_era) with the HL-outs share nearly constant (sd 0.042).
   User decision: build ONLY the opp_penhl_share mixture weight (rider on the
   starter-length/pen-split item); the bridge-ERA column stays dropped.
8. **SIM_BLEND total weight = 0.20 is a grandfathered 2026-informed tune**
   (LEDGERED 2026-07-15, audit fix #7; user decision: keep + refit at chain).
   The 07-13 half-weight call was made after seeing 2026 flat — it predates
   the "2026 may veto, never TUNE" rule but violates it in substance. The
   raw 2025-only fit was w=0.50 (`artifacts/sim_blend_2025to2026.csv`);
   score/winner shipped weights were also user-moderated below their raw
   fits (0.60→0.35, 0.45→0.30). OWED: when the finish chain reruns
   pa_blend, re-decide ALL THREE weights from 2025-only evidence (2026 may
   veto the package, never set a weight). predict.SIM_BLEND carries the
   matching comment.
