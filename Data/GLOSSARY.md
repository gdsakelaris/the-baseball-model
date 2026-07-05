# MLB Data Glossary

Ten CSV files, all UTF-8 (Excel-safe), scraped by the scripts in `Scripts/`.
Sources: MLB.com rosters & stats, OnlyHomers.com (home run log), Baseball
Savant (pitch-level Statcast), MLB Stats API (per-game boxscores),
USGS/OpenTopoData (elevations).

| File | Built by |
|---|---|
| mlb_rosters_2026.csv | scrape_rosters.py |
| mlb_batting_stats_2020_2026.csv | scrape_batting_stats.py |
| mlb_pitching_stats_2020_2026.csv | scrape_pitching_stats.py |
| mlb_ballparks.csv | build_ballparks.py |
| mlb_homeruns_2020_2026.csv | scrape_homeruns.py |
| mlb_pitch_arsenals_2020_2026.csv | scrape_pitch_arsenals_2F.py |
| mlb_pitch_arsenals_batters_2020_2026.csv | scrape_pitch_arsenals_2F.py --type batter |
| mlb_games_2020_2026.csv | scrape_gamelogs_3F.py (writes all three game files in one run) |
| mlb_game_batting_2020_2026.csv | scrape_gamelogs_3F.py |
| mlb_game_pitching_2020_2026.csv | scrape_gamelogs_3F.py |

All scrapers write into `Data/` by default. `Scripts/update_all.py` runs every
scraper (everything except build_ballparks.py) for a one-command daily
refresh; add `--retrain` to also retrain the models afterward.

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

## mlb_rosters_2026.csv — current 2026 depth-chart rosters
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

## mlb_batting_stats_2020_2026.csv — season batting lines
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

## mlb_pitching_stats_2020_2026.csv — season pitching lines
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

## mlb_homeruns_2020_2026.csv — every home run, 2020–2026
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

## mlb_pitch_arsenals_2020_2026.csv — pitcher arsenal results (Statcast)
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

## mlb_pitch_arsenals_batters_2020_2026.csv — batter vs pitch type (Statcast)
Same columns as above, but one row per **batter**, per pitch type faced, per
season. `%` = share of pitches the batter saw of that type; positive run
values favor the **batter**. Join to the pitcher file on `PitchType` + `Year`
to build batter-vs-arsenal matchups.

## mlb_games_2020_2026.csv — every regular-season game
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

## mlb_game_batting_2020_2026.csv — per-game batting lines
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

## mlb_game_pitching_2020_2026.csv — per-game pitching lines
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
