"""Sportsbook-odds utilities: the bridge between the model's fair probabilities
and the prices a book actually posts.

Everything the evaluation measured before this file compares the model to a
naive base rate. That answers "does the model know more than nothing?" — not
"does the edge survive the vig?" These helpers convert American prices to
implied probabilities, strip the book's hold (de-vig), and turn a model
probability + a real price into expected value and settled profit, so
evaluate_deep can finally ask the second question.

It also defines the canonical odds-store schema shared by the scraper
(Tools/2_scrape_odds.py, which writes it) and the market evaluation
(evaluate_deep Section 9, which reads it). Nothing here imports the model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------- store schema

# One row per (player, market, line) offering. Batter props are all "N+ of
# something", i.e. an over/under at a half-point line where YES == over LINE:
# the model's P(prop) is directly P(over LINE), so it lines up with OverPrice.
ODDS_COLUMNS = ["Date", "GamePk", "Team", "PlayerId", "PlayerName",
                "Market", "Line", "OverPrice", "UnderPrice", "Book",
                "CapturedAt"]

# model prop key -> the market that prices it. `api` is The Odds API market
# key the scraper requests; `line` is the over/under number that equals the
# prop (over 0.5 = "1+", over 1.5 = "2+"); `label` is for display.
PROP_MARKET = {
    "hr":     dict(api="batter_home_runs",     line=0.5, label="1+ HR"),
    "hit":    dict(api="batter_hits",          line=0.5, label="1+ hit"),
    "hits2":  dict(api="batter_hits",          line=1.5, label="2+ hits"),
    "tb2":    dict(api="batter_total_bases",   line=1.5, label="2+ total bases"),
    "run":    dict(api="batter_runs_scored",   line=0.5, label="run scored"),
    "rbi":    dict(api="batter_rbis",          line=0.5, label="1+ RBI"),
    "bb":     dict(api="batter_walks",         line=0.5, label="1+ walk"),
    "sb":     dict(api="batter_stolen_bases",  line=0.5, label="stolen base"),
    "single": dict(api="batter_singles",       line=0.5, label="1+ single"),
    "double": dict(api="batter_doubles",       line=0.5, label="1+ double"),
    "bk":     dict(api="batter_strikeouts",    line=0.5, label="1+ batter K"),
    "bk2":    dict(api="batter_strikeouts",    line=1.5, label="2+ batter K"),
    "hrr2":   dict(api="batter_hits_runs_rbis", line=1.5, label="2+ H+R+RBI"),
    "hrr3":   dict(api="batter_hits_runs_rbis", line=2.5, label="3+ H+R+RBI"),
}

# Pitcher/starter count props the model predicts as multi-line count heads
# (predict.py xK, xOuts, xHits, xBB, xER). Kept SEPARATE from PROP_MARKET
# because Section 9 grades single-line binary props keyed to `results`; these
# have many lines and no such binary key, so the scraper captures every line
# the book posts and Section 7b evaluates the count heads. `api` is the Odds
# API market key; `label` is for display. All five confirmed live 2026-07-08.
STARTER_MARKET = {
    "pk":    dict(api="pitcher_strikeouts",   label="pitcher strikeouts"),
    "pouts": dict(api="pitcher_outs",          label="pitcher outs"),
    "phits": dict(api="pitcher_hits_allowed",  label="pitcher hits allowed"),
    "pbb":   dict(api="pitcher_walks",         label="pitcher walks"),
    "per":   dict(api="pitcher_earned_runs",   label="pitcher earned runs"),
}

# reverse lookups used by the scraper and Section 9:
#   (api market, line) -> our prop key   (join market rows back to a prop)
#   api market -> default line           (Yes/No HR markets omit the point)
MARKET_LINE_TO_PROP = {(m["api"], m["line"]): p for p, m in PROP_MARKET.items()}
API_DEFAULT_LINE = {}
for _m in PROP_MARKET.values():
    API_DEFAULT_LINE[_m["api"]] = min(API_DEFAULT_LINE.get(_m["api"], _m["line"]),
                                      _m["line"])


# --------------------------------------------------- price <-> probability


def _isnum(x):
    return x is not None and not (isinstance(x, float) and np.isnan(x))


def american_to_prob(american):
    """Implied probability (WITH vig) of an American price. +150 -> 0.40,
    -200 -> 0.667. Returns NaN for a missing price."""
    if not _isnum(american):
        return float("nan")
    a = float(american)
    return 100.0 / (a + 100.0) if a >= 0 else (-a) / (-a + 100.0)


def american_profit(american, stake=1.0):
    """Profit (payout minus stake) on a winning bet at an American price:
    +150 pays 1.5u on 1u, -200 pays 0.5u on 1u."""
    a = float(american)
    return stake * (a / 100.0 if a >= 0 else 100.0 / (-a))


def fair_american(p):
    """Fair (no-vig) American price string for a probability, for display."""
    if not (0 < p < 1):
        return ""
    return f"-{round(100 * p / (1 - p)):d}" if p >= 0.5 else \
           f"+{round(100 * (1 - p) / p):d}"


# ------------------------------------------------------------------ de-vig


def devig(implied):
    """Proportional (multiplicative) de-vig: rescale same-market implied
    probabilities to sum to 1, removing the book's hold. Standard for two-way
    and multiway markets. Returns a list aligned to the input."""
    vals = [american_to_prob(x) if not isinstance(x, float) or not np.isnan(x)
            else float("nan") for x in implied]
    s = np.nansum(vals)
    if s <= 0:
        return [float("nan")] * len(implied)
    return [v / s for v in vals]


# The sharp book: lowest hold, prices widely treated as the market's fair
# number. When it quotes a line, its de-vigged prob IS the reference; the
# median across soft books is the fallback. The scraper's DEFAULT_BOOKS keeps
# pinnacle in every capture (Tools/2_scrape_odds.py, added 2026-07-09).
SHARP_BOOK = "pinnacle"


def sharp_fair(g, book_col="Book"):
    """Consensus fair P(over) for ONE group of same-market/line rows that were
    already de-vigged into a 'fair' column: the sharp book's quote when it
    posts the line, else the median across books. Shared by every consensus
    builder (5_prop_rankings MktEdge%, evaluate_deep Section 9, predict Bets
    sheet) so they all grade against the same reference."""
    s = g.loc[g[book_col].astype(str).str.lower() == SHARP_BOOK, "fair"].dropna()
    return float(s.median()) if len(s) else float(g["fair"].median())


def devig_two_way(over_price, under_price):
    """Return (fair P(over), hold) for a two-sided market. `hold` is the book's
    margin (implied over + implied under - 1). If only one side is present the
    vig can't be stripped, so the raw implied prob is returned with hold=NaN."""
    io, iu = american_to_prob(over_price), american_to_prob(under_price)
    if np.isnan(io) and np.isnan(iu):
        return float("nan"), float("nan")
    if np.isnan(io) or np.isnan(iu):        # one-sided: no vig to remove
        return (io if not np.isnan(io) else 1.0 - iu), float("nan")
    hold = io + iu - 1.0
    return io / (io + iu), hold


# ------------------------------------------------------------- EV & settle


def ev_per_unit(p_true, american):
    """Expected profit per 1u staked at `american` if the true win prob is
    p_true. Positive => +EV. NaN price -> NaN."""
    if not _isnum(american):
        return float("nan")
    win = american_profit(american, 1.0)
    return p_true * win - (1.0 - p_true) * 1.0


def settle(american, won, stake=1.0):
    """Realized profit of a settled bet: +profit if it won, -stake if it lost."""
    return american_profit(american, stake) if won else -stake


def pick_side(p_over, over_price, under_price):
    """Given the model's P(over) and both prices, return the side to bet
    ('over'/'under'/None) and its EV per unit. We bet the side whose EV against
    the ACTUAL offered price (vig included) is positive and larger; None when
    neither side clears the vig."""
    ev_o = ev_per_unit(p_over, over_price)
    ev_u = ev_per_unit(1.0 - p_over, under_price)
    ev_o = ev_o if _isnum(over_price) else float("-inf")
    ev_u = ev_u if _isnum(under_price) else float("-inf")
    best = max(ev_o, ev_u)
    if not np.isfinite(best) or best <= 0:
        return None, 0.0
    return ("over", ev_o) if ev_o >= ev_u else ("under", ev_u)


# ------------------------------------------------------------ store loader


def load_odds(path, year=None):
    """Load the odds store as a DataFrame in canonical schema. A missing OR
    empty file -> empty frame with the right columns (Section 9 degrades
    gracefully — e.g. after the store is cleared to re-scrape). Coerces Date to
    a date, prices to numeric, and optionally filters to one season."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=ODDS_COLUMNS)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:      # file exists but has no header/rows
        return pd.DataFrame(columns=ODDS_COLUMNS)
    for c in ODDS_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for c in ("Line", "OverPrice", "UnderPrice"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["PlayerId"] = pd.to_numeric(df["PlayerId"], errors="coerce").astype("Int64")
    if year is not None:
        df = df[pd.to_datetime(df["Date"]).dt.year == year]
    return df[ODDS_COLUMNS].reset_index(drop=True)


DEFAULT_STORE = Path(__file__).resolve().parents[1] / "Data" / "mlb_odds.csv"
