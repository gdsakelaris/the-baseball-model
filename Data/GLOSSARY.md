# MLB Data Glossary

Eighteen CSV files, all UTF-8 (Excel-safe), scraped by the scripts in
`Scrapers/`. Sources: MLB.com rosters & stats, OnlyHomers.com (home run log),
Baseball Savant (pitch-level Statcast, batted balls, sprint speed, OAA,
baserunning), MLB Stats API (per-game boxscores), USGS/OpenTopoData
(elevations), Open-Meteo (per-game weather: humidity/pressure).

Filenames are stable across seasons; every multi-season file covers 2020
through the current season (`Scrapers/seasons.py` decides what "current"
means, so the files simply grow at each annual rollover).

| File | Built by |
|---|---|
| mlb_rosters.csv | scrape_rosters.py |
| mlb_batting_stats.csv | scrape_batting_stats.py |
| mlb_pitching_stats.csv | scrape_pitching_stats.py |
| mlb_ballparks.csv | build_ballparks.py |
| mlb_homeruns.csv | scrape_homeruns.py |
| mlb_pitch_arsenals.csv | scrape_pitch_arsenals_2F.py |
| mlb_pitch_arsenals_batters.csv | scrape_pitch_arsenals_2F.py --type batter |
| mlb_games.csv | scrape_gamelogs_3F.py (writes all three game files in one run) |
| mlb_game_batting.csv | scrape_gamelogs_3F.py |
| mlb_game_pitching.csv | scrape_gamelogs_3F.py |
| mlb_statcast_bip.csv | scrape_statcast.py (`--backfill` once; daily runs are incremental) |
| mlb_pitch_daily_pitchers.csv | scrape_pitches.py (writes both pitch-daily files; `--backfill` once) |
| mlb_pitch_daily_batters.csv | scrape_pitches.py |
| mlb_sprint_speed.csv | scrape_sprint_speed.py |
| mlb_oaa.csv | scrape_oaa.py |
| mlb_oaa_players.csv | scrape_oaa.py (same run writes both) |
| mlb_baserunning.csv | scrape_baserunning.py |
| mlb_weather.csv | scrape_weather.py (`--backfill` once; daily runs are incremental) |

All scrapers write into `Data/` by default. `Scrapers/update_all.py` runs every
scraper (everything except build_ballparks.py) for a one-command daily
refresh; add `--retrain` to also retrain the models afterward. Each scraped
file is schema-validated (`Scrapers/validate_data.py`) against the previous
copy in `Data/backups/` before the pipeline accepts it; a file that fails
validation is restored from backup and the retrain is skipped.

## How the files connect

| Key | Meaning | Appears in |
|---|---|---|
| `PlayerId` / `BatterId` | MLB's permanent player ID (same ID everywhere) | rosters, batting, pitching, both arsenals, game_batting, game_pitching, homeruns (`BatterId`) |
| `Team` (abbrev, e.g. PHI) | MLB team abbreviation | batting, pitching, both arsenals, game files, homeruns (`Team`, `PitcherTeam`) |
| `TeamName` / roster `Team` / ballparks `Team` (full, e.g. Philadelphia Phillies) | Full team name | rosters, batting, pitching, ballparks |
| `Ballpark` / `Venue` | Stadium name | ballparks, homeruns, games |
| `Year` / `Season` | Season | batting, pitching, both arsenals, homeruns, game files |
| `GamePk` | MLB's unique game ID | games, game_batting, game_pitching |
| `PitchType` / `Pitch` | Statcast pitch code / display name (FF = 4-Seam Fastball…) | both arsenals; homeruns has `Pitch` (display name only) |

Typical joins:

- Who hit a HR → their season stats: `homeruns.BatterId` = `batting.PlayerId` and `homeruns.Year` = `batting.Year`
- HR → park traits: `homeruns.Ballpark` = `ballparks.Ballpark` (current parks; old seasons include former/special venues with no ballparks row)
- Batter vs pitcher matchup: `arsenals_batters` × `arsenals` (pitcher file) on `PitchType` + `Year` — how a batter fares against each pitch type, weighted by how often a given pitcher throws it
- Roster bio → stats: `rosters.PlayerId` = `*.PlayerId`
- Stats abbrev → full name: `batting/pitching.TeamName` = `rosters.Team` = `ballparks.Team`
- `homeruns.Pitcher` is a display name only (no ID); join to pitching stats by name+year if needed (imperfect for duplicate names)
- Per-game label for modeling: `game_batting.HR > 0` = player homered that game; join game context (park, weather, opposing starter) via `GamePk` to games and game_pitching (`GS = 1`)

Park names are canonical across all files (one name per physical park, all
years): Daikin Park (ex-Minute Maid), Rate Field (ex-Guaranteed Rate),
Oriole Park at Camden Yards, loanDepot Park (ex-Marlins Park), Dodger
Stadium (2026 sponsor prefix stripped). The scrapers normalize stale and
sponsor-variant names at scrape time.

Format notes: roster `DOB` is `MM/DD/YYYY`; homerun `Date` is `YYYY-MM-DD`.
Roster `Ht` is text like `6' 2"`. Rate stats are strings like `.337` /
`0.337` depending on source. 2021 batting includes 713 pitchers (last
pre-universal-DH year). 2020 was a 60-game season (all counting stats are
~37% of normal scale); the homerun file's 2020 slice also includes the 66
postseason HRs. Homeruns 2024 includes 2 Futures Game rows (teams NLF/ALF).

---

## mlb_rosters.csv — current depth-chart rosters
One row per player per team (first depth-chart listing kept).

| Column | Meaning |
|---|---|
| PlayerId | MLB player ID (join key) |
| Name | Player name as shown on MLB.com |
| Team | Full team name |
| Position | Depth-chart group: Rotation, Bullpen, Catcher, First Base, Second Base, Third Base, Shortstop, Left Field, Center Field, Right Field, Designated Hitter |
| B | Bats: L, R, or S (switch) |
| T | Throws: L, R (S exists for rare ambidextrous) |
| Ht | Height, e.g. `6' 2"` |
| Wt | Weight (lb) |
| DOB | Date of birth, MM/DD/YYYY |

## mlb_batting_stats.csv — season batting lines
One row per player per season (traded players aggregated; `Team` = most recent).

| Column | Meaning |
|---|---|
| Year, PlayerId, Name, Pos | Season, MLB ID, name, primary position |
| Team / TeamName | Abbreviation / full name |
| G | Games played |
| AB | At-bats (PA minus walks, HBP, sacrifices, interference) |
| R | Runs scored |
| H | Hits |
| 2B / 3B / HR | Doubles / triples / home runs |
| RBI | Runs batted in |
| BB | Walks |
| SO | Strikeouts |
| SB / CS | Stolen bases / caught stealing |
| AVG | Batting average, H/AB |
| OBP | On-base %, (H+BB+HBP)/(AB+BB+HBP+SF) |
| SLG | Slugging, total bases per AB |
| OPS | OBP + SLG |
| PA | Plate appearances |
| HBP | Hit by pitch |
| SAC / SF | Sacrifice bunts / sacrifice flies |
| GIDP | Grounded into double play |
| GO/AO | Groundout-to-airout ratio |
| XBH | Extra-base hits (2B+3B+HR) |
| TB | Total bases |
| IBB | Intentional walks |
| BABIP | Batting avg on balls in play, (H−HR)/(AB−K−HR+SF) |
| ISO | Isolated power, SLG − AVG |
| AB/HR | At-bats per home run |
| BB/K | Walk-to-strikeout ratio |
| BB% / K% | Walks / strikeouts per plate appearance |

## mlb_pitching_stats.csv — season pitching lines
One row per pitcher per season (traded players aggregated).

| Column | Meaning |
|---|---|
| Year, PlayerId, Name, Pos, Team, TeamName | As above |
| W / L | Wins / losses |
| ERA | Earned runs per 9 innings |
| G / GS | Games pitched / started |
| CG / SHO | Complete games / shutouts |
| SV / SVO | Saves / save opportunities |
| IP | Innings pitched (.1 = one out, .2 = two outs) |
| H / R / ER | Hits / runs / earned runs allowed |
| HR | Home runs allowed |
| HB | Hit batsmen |
| BB / SO | Walks / strikeouts |
| WHIP | (BB+H)/IP |
| AVG | Opponent batting average |
| TBF | Total batters faced |
| NP | Number of pitches |
| P/IP | Pitches per inning |
| QS | Quality starts (6+ IP, ≤3 ER) |
| GF | Games finished (last pitcher, non-starter) |
| HLD | Holds |
| IBB | Intentional walks issued |
| WP / BK | Wild pitches / balks |
| GDP | Double plays induced |
| GO/AO | Groundout-to-airout ratio |
| SO/9 / BB/9 | Strikeouts / walks per 9 IP |
| K/BB | Strikeout-to-walk ratio |
| BABIP | Opponent BABIP |
| SB / CS | Stolen bases allowed / runners caught |
| PK | Pickoffs |

## mlb_ballparks.csv — stadium traits
One row per current MLB park, under its canonical 2026 name (Tropicana
Field is the Rays' park again after their 2025 season at Steinbrenner
Field). Former and special-event venues appearing in the game/homerun files
(Oakland Coliseum, Sahlen Field, Las Vegas Ballpark, London Stadium, …)
intentionally have no row here.

| Column | Meaning |
|---|---|
| Ballpark | Stadium name |
| Team | Full team name of home club |
| LF / CF / RF | Fence distance (ft) down left field / center / right field lines |
| Elevation_ft | Ground elevation above sea level (ft), from USGS/OpenTopoData |
| Lat / Lon | Stadium coordinates (the weather scrapers key Open-Meteo on these) |
| Roof | open / retractable / dome (fixed) |

## mlb_homeruns.csv — every home run since 2020
One row per home run. Source: OnlyHomers database. Rows run chronologically
from 2020's first homer to 2026's latest (the site's per-season sequence; a
couple dozen rows sit slightly out of date order where the site inserted
late corrections).

| Column | Meaning |
|---|---|
| Year | Season |
| Running Total | Sequential number across the whole file, 1 = first 2020 homer |
| Total | Season-running HR count across MLB (site's counter) |
| Team | Batter's team (abbrev) |
| BatterId / Batter | MLB ID / name of the hitter |
| HR | That batter's season HR number (1st, 2nd, …) |
| ROB | Runners on base when hit (0–3; 3 = grand slam; source nulls in older seasons are written as 0) |
| Inning | Inning hit |
| Outs | Outs at the time (0–2) |
| Angle | Launch angle (degrees) |
| Exit Velo | Exit velocity (mph) |
| Distance | Projected distance (ft) |
| Pitch | Pitch type hit (display name, e.g. 4-Seam Fastball) |
| Pitcher | Pitcher's name (no ID in source) |
| PitcherTeam | Pitcher's team (abbrev) |
| Ballpark | Stadium where hit |
| Date | Game date, YYYY-MM-DD |

## mlb_pitch_arsenals.csv — pitcher arsenal results (Statcast)
One row per pitcher, per pitch type, per season. Results are what opposing
batters did against that pitch.

| Column | Meaning |
|---|---|
| Year, PlayerId, Player, Team | Season, MLB ID, "Last, First", abbrev |
| PitchType | Statcast code: FF 4-seam, SI sinker, FC cutter, SL slider, ST sweeper, CU curve, KC knuckle-curve, CH changeup, FS splitter, SV slurve, KN knuckleball, EP eephus, FO forkball, SC screwball |
| Pitch | Pitch display name |
| RV/100 | Run value per 100 pitches (positive = good for the pitcher in this file) |
| Run Value | Total run value for the season on that pitch |
| Pitches | Times thrown |
| % | Usage: share of the pitcher's pitches |
| PA | Plate appearances ending on that pitch |
| BA / SLG / wOBA | Opponent results on PAs ending with it |
| Whiff % | Swings that missed / total swings |
| K% | Strikeout rate of those PAs |
| Put Away % | 2-strike pitches converted to strikeouts |
| xBA / xSLG / xwOBA | Expected stats from exit velo + launch angle (quality of contact) |
| Hard Hit % | Batted balls ≥ 95 mph |

## mlb_pitch_arsenals_batters.csv — batter vs pitch type (Statcast)
Same columns as above, but one row per **batter**, per pitch type faced, per
season. `%` = share of pitches the batter saw of that type; positive run
values favor the **batter**. Join to the pitcher file on `PitchType` + `Year`
to build batter-vs-arsenal matchups.

## mlb_games.csv — every regular-season game
One row per final regular-season game (MLB Stats API). Doubleheaders are two
rows (distinct `GamePk`), suspended games appear once on their official date.

| Column | Meaning |
|---|---|
| GamePk | MLB's unique game ID (join key to the two files below) |
| Season | Year |
| Date | Official game date, YYYY-MM-DD |
| DayNight | `day` or `night` |
| AwayTeam / HomeTeam | Team abbreviations (season-correct: OAK ≤2024, ATH ≥2025) |
| AwayScore / HomeScore | Final score |
| Venue | Stadium (matches `Ballpark` in mlb_ballparks.csv for current parks; older seasons include former/special venues) |
| Temp | Game-time temperature, °F |
| Condition | Sky/roof condition text (Clear, Cloudy, Dome, Roof Closed, …) |
| WindSpeed | Wind speed, mph |
| WindDir | Wind direction text (Out To CF, In From LF, L To R, Calm, …) |

## mlb_game_batting.csv — per-game batting lines
One row per batter per game appearance (includes pinch-hitters/runners).
This is the per-game label source for modeling: `HR > 0` means the player
homered that game.

| Column | Meaning |
|---|---|
| GamePk, Season, Date | Game keys |
| PlayerId / Name | MLB player ID / name |
| Team / Opponent | Abbreviations |
| Home | 1 = home team, 0 = away |
| BattingOrder | MLB slot code: 100 = leadoff starter, 400 = cleanup starter, 401 = first sub into slot 4, … (blank = no slot, e.g. some pitchers). Starters end in 00. |
| Position | Fielding position abbreviation (DH, C, 1B, …, PH, PR) |
| PA / AB | Plate appearances / at-bats |
| R / H / 2B / 3B / HR / RBI | Counting stats that game |
| BB / IBB / SO / HBP | Walks / intentional / strikeouts / hit-by-pitch |
| SB / CS | Stolen bases / caught stealing |
| SAC / SF | Sacrifice bunts / flies |
| GIDP | Double plays grounded into |
| TB | Total bases |
| LOB | Runners left on base |

## mlb_game_pitching.csv — per-game pitching lines
One row per pitcher per game appearance.

| Column | Meaning |
|---|---|
| GamePk, Season, Date, PlayerId, Name, Team, Opponent, Home | As above |
| GS | 1 = started the game |
| GF | 1 = finished the game (last pitcher, non-starter) |
| IP | Innings pitched that game (.1/.2 = one/two outs) |
| BF | Batters faced |
| NP / Strikes | Pitches thrown / strikes |
| H / R / ER / HR | Hits, runs, earned runs, homers allowed |
| BB / IBB / SO / HBP | Walks / intentional / strikeouts / hit batsmen |
| WP / BK | Wild pitches / balks |
| W / L / SV / HLD | Decision flags for that game (1/0) |

## mlb_statcast_bip.csv — every tracked ball in play (Statcast)
One row per batted ball (regular season). The HR log covers only homers — a
sample censored to each batter's best contact; this covers ALL contact, so
the model can see contact quality ("process" stats that stabilize far faster
than outcomes) for both batters and pitchers.

| Column | Meaning |
|---|---|
| GamePk, Season, Date | Game keys (GamePk matches the game files) |
| BatterId / PitcherId | MLB player IDs (match PlayerId everywhere) |
| Stand / PThrows | Batter side / pitcher hand for this matchup |
| Events | Outcome (single, home_run, field_out, …) |
| BBType | fly_ball / ground_ball / line_drive / popup |
| ExitVelo | Exit velocity, mph (blank when untracked) |
| LaunchAngle | Launch angle, degrees |
| LSA | Savant launch-speed-angle code 1–6; 6 = barrel |
| xBA / xwOBA | Expected BA / wOBA of this batted ball from EV+angle |
| HitDistance | Projected distance, ft |
| AtBat / PitchNum | At-bat number in game / pitch number in at-bat; (GamePk, AtBat, PitchNum) is unique |

## mlb_pitch_daily_pitchers.csv / mlb_pitch_daily_batters.csv — pitch-level daily aggregates
One row per player per day, aggregated at scrape time from EVERY pitch
(~700k/season). Swing-and-miss and plate discipline are the fastest-
stabilizing skills in baseball; none of this is in box scores or the
batted-ball file. `--backfill` also archives the raw pitches to
`Data/raw_pitches/pitches_{year}.parquet` (~117 MB/season, every Savant
detail column) so future schema changes re-aggregate from disk
(`--from-raw`) instead of re-downloading.

| Column | Meaning |
|---|---|
| PlayerId, Date | MLB player ID / game day |
| n | Pitches thrown (pitcher file) or seen (batter file) |
| sw_n | Swings |
| wh_n | Whiffs (swinging strikes, incl. blocked) |
| cs_n | Called strikes |
| z_n / oz_n | Pitches in / out of the strike zone |
| oz_sw | Out-of-zone swings (chases) |
| oz_wh | Out-of-zone whiffs — with wh_n this splits whiffs by zone, so in-zone contact rate (the most stable hit-tool skill) is derivable |
| fb95_n / fb95_sw / fb95_wh | Pitches / swings / whiffs vs 95+ mph fastballs (both files, v3) — batter: performance against elite velocity; pitcher: elite-velo usage |
| fbmid_* / fblo_* | The graded bands below fb95 (v4): 92–95 and <92 mph FF/SI, same n/sw/wh trios — whiff splits and usage by velocity band |
| brk_n / brk_sw / brk_wh | Breaking balls (SL/ST/SV/CU/KC/CS/SC/KN) seen or thrown / swings / whiffs (v3) — with off_* and the fastball remainder, whiff splits by pitch class |
| off_n / off_sw / off_wh | Offspeed (CH/FS/FO/EP) counterparts (v3) |
| edge_n | Pitches in the shadow band (0.67–1.33 of the scaled zone, plate_x/plate_z vs per-pitch sz_top/sz_bot) — edge_n/n is a command proxy (v3) |
| fp_n / fp_sw / fp_s | 0-0-count pitches / swings at them / first-pitch strikes (v3) — fp_s/fp_n = F-strike% (pitcher), fp_sw/fp_n = first-pitch aggression (batter) |
| ts_n / ts_sw / ts_wh | Two-strike pitches / swings / whiffs (v5) — pitcher put-away ability, batter two-strike survival |
| f32_n / f32_z / f32_b / f32_sw / f32_wh | Full-count (3-2) pitches: total / in zone / called+blocked balls (= walks) / swings / whiffs (v5) — payoff-pitch behavior for the walk heads |
| fb_n / fb_v / fb_v2 | Fastballs (FF+SI) with tracked velo / velo sum / velo sum-of-squares (pitcher file only; v2 enables within-pitcher velo SD, v5) |
| rp_n / rp_x / rp_x2 / rp_z / rp_z2 | Release-point coords: count / x,z sums / sums-of-squares (pitcher file only, v5) — rebuild release scatter (mechanical repeatability) from cumulative sums |

## mlb_sprint_speed.csv — Statcast sprint speed
One row per (Year, PlayerId), min 5 competitive runs. Consumed as
PRIOR-season values (leakage-free): a 2026 game sees the 2025 measurement.

| Column | Meaning |
|---|---|
| Year, PlayerId, Name, Team | Keys |
| CompetitiveRuns | Sample size (qualifying runs) |
| SprintSpeed | ft/s over the fastest one-second window |
| HPto1B | Home-to-first time, seconds |

## mlb_oaa.csv — team Outs Above Average (defense)
One row per (Year, Team). Consumed as PRIOR-season values.

| Column | Meaning |
|---|---|
| Year, Team, TeamId, TeamName | Keys (Team = that season's abbreviation, rename-aware) |
| OAA | Raw season outs above average |
| OAA_per162 | Scaled to 162 games (2020 was a 60-game season) |

## mlb_oaa_players.csv — per-fielder Outs Above Average
One row per (Year, PlayerId), 2016+ (Statcast fielding era). Lets the model
aggregate the ACTUAL lineup's defense instead of the team-season blend.
Consumed as PRIOR-season values.

| Column | Meaning |
|---|---|
| Year, PlayerId, Name | Keys |
| Pos | Primary position that season (SS, CF, 1B, …) — used for infield/outfield splits |
| OAA | Season outs above average |
| FRP | Fielding runs prevented (run-value version) |

## mlb_baserunning.csv — Statcast baserunning run value
One row per (Year, PlayerId), 2016+, qualified runners only (~190/season;
absent = treat as league-average). Consumed as PRIOR-season values.

| Column | Meaning |
|---|---|
| Year, PlayerId, Name | Keys |
| RunnerRuns | Total baserunning run value |
| RunnerRunsXB | Extra-base advancement component (1st-to-3rd, scoring from 2nd, tag-ups) |
| RunnerRunsSB | Basestealing component |
| Opportunities | Times on base with an advancement opportunity |

## mlb_weather.csv — per-game weather (air-density inputs)
One row per GamePk (Open-Meteo archive/forecast at each park's coordinates,
sampled at the approximate local start hour: day 13:00, night 19:00). The
boxscores already carry Temp/Wind; this adds what they lack. Former and
special-event venues have coordinates in scrape_weather.py.

| Column | Meaning |
|---|---|
| GamePk, Date, Venue | Game keys |
| Humidity | Relative humidity (%) at the start hour |
| Pressure | Surface (station-level) pressure, hPa — embeds elevation, so it is the air-density input directly |
| Precip | Total precipitation (mm) over the first three game hours |
