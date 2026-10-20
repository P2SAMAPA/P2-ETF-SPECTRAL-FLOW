"""
config.py  —  Configuration for P2 BASELINE-BOOK

A monetization study, not a new signal. Across the last several engines the
only ranking information that held up on real data sat in the plain price
baseline (momentum, volatility, beta): rank-IC around 0.03-0.05 — yet the
concentrated "top 3 vs the universe" product lost money after costs every
time. This repo asks the narrower question: can THAT baseline be turned into
a book that makes money after costs, and if so, which kind of book, and is it
alpha or just market exposure?
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-baseline-book-results"
LEDGER_FILE = "forward_ledger.json"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

# Hand-labeled, static asset-class tags: used to build class-neutral books and
# for dashboard coloring. Not a model input.
ASSET_CLASS_MAP = {
    "TLT": "Fixed Income", "VCIT": "Fixed Income", "LQD": "Fixed Income", "HYG": "Fixed Income",
    "VNQ": "Real Estate", "XLRE": "Real Estate",
    "GLD": "Commodity", "SLV": "Commodity", "GDX": "Commodity", "XME": "Commodity", "URA": "Commodity",
    "SPY": "Equity-Broad", "QQQ": "Equity-Broad", "IWM": "Equity-Broad", "IWD": "Equity-Broad",
    "IWO": "Equity-Broad", "IWF": "Equity-Broad", "IWR": "Equity-Broad", "VO": "Equity-Broad",
    "VB": "Equity-Broad", "VIG": "Equity-Broad", "VEA": "Equity-Broad", "QUAL": "Equity-Broad",
    "SPYG": "Equity-Broad",
    "XLK": "Equity-Sector", "XLF": "Equity-Sector", "XLE": "Equity-Sector", "XLV": "Equity-Sector",
    "XLI": "Equity-Sector", "VGT": "Equity-Sector", "VDE": "Equity-Sector", "XLC": "Equity-Sector",
    "IBB": "Equity-Sector", "XLY": "Equity-Sector", "XLP": "Equity-Sector", "XLU": "Equity-Sector",
    "XSD": "Equity-Sector", "SOXX": "Equity-Sector", "SMH": "Equity-Sector", "XBI": "Equity-Sector",
    "XLB": "Equity-Sector", "VUG": "Equity-Sector", "VTV": "Equity-Sector",
}
# Classes smaller than this are de-meaned / ranked against the universe-wide
# pool rather than against themselves (a singleton class de-meaned against
# itself is a constant zero). Carried over from the spectral-flow engine.
MIN_CLASS_SIZE = 3

# ---------------------------------------------------------------------------
# The score: a ridge regression of the cross-sectionally standardized forward
# H-day return on FOUR price features only (no spectral, graphon, hyperbolic
# or any other engine's features), refit at each rebalance on the trailing
# WINDOW days:  mom5, mom21, vol21, beta (rolling beta to the equal-weight
# universe return).
# ---------------------------------------------------------------------------
BETA_WINDOW = 126
HORIZONS = [1, 5, 10, 20]
WINDOWS = [504, 1008]
RIDGE_LAMBDA = 0.01
MIN_TRAIN_DAYS = 60
# Whether the score model is fit on class-neutral features/target (True) or
# plain cross-sectional ones (False). Both are run.
MODEL_NEUTRAL_OPTIONS = [False, True]

# ---------------------------------------------------------------------------
# The books built from the SAME score. Long-only books are measured ACTIVE vs
# the equal-weight universe (i.e. "own the top N instead of owning
# everything"); long/short books are dollar-neutral. The *_cn books rank
# WITHIN each asset class and are class-neutral by construction (each class's
# longs and shorts offset), so their raw P&L is a tradable class-neutral
# P&L — not a metric scored against adjusted returns.
# ---------------------------------------------------------------------------
CONSTRUCTIONS = ["top3_ew", "top5_ew", "topq_ew", "ls_quintile", "rank_ls", "ls_quintile_cn", "rank_ls_cn"]
LONG_ONLY = {"top3_ew", "top5_ew", "topq_ew"}
CONSTRUCTION_LABELS = {
    "top3_ew": "Top 3, equal weight (long-only vs universe)",
    "top5_ew": "Top 5, equal weight (long-only vs universe)",
    "topq_ew": "Top quintile, equal weight (long-only vs universe)",
    "ls_quintile": "Long top quintile / short bottom quintile",
    "rank_ls": "Rank-weighted long/short",
    "ls_quintile_cn": "Quintile long/short within asset class",
    "rank_ls_cn": "Rank-weighted long/short within asset class",
}
QUANTILE = 0.20

# Costs. TRADING_COST_BPS is a ROUND-TRIP cost per unit of notional replaced
# (the convention of every sibling engine), i.e. TRADING_COST_BPS/2 per unit
# of traded notional on each side; a book's per-period cost is
# sum|w_t - w_{t-1}| * TRADING_COST_BPS/2. Long/short books also pay a
# short-borrow/financing charge on short notional (an ASSUMPTION — liquid
# index ETFs are cheap to short, thematic ones are not).
TRADING_COST_BPS = 15
SHORT_BORROW_BPS_ANNUAL = 50

SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
MIN_SELECTION_N = 50
MIN_SELECTION_N_FRACTION = 0.15

# Verdict on the SELECTED book's holdout (chosen on the selection segment
# only, so the holdout is a single honest test):
#   Supported     : gate passed, n >= MIN_HOLDOUT_PERIODS, net mean > 0,
#                   net t >= SUPPORTED_T AND market-adjusted alpha t >= SUPPORTED_T
#   Suggestive    : gate passed, net mean > 0, net t >= SUGGESTIVE_T, alpha t > 0
#   Not supported : anything else
SUPPORTED_T = 1.65
SUGGESTIVE_T = 1.0

# Forward ledger: a PRE-REGISTERED set of books recorded every run with no
# selection and no later change, scored only after their holding period has
# elapsed. This is the only evidence here that is out-of-sample with respect
# to the analyst's own choices (everything in the backtest grid was designed
# after earlier engines already showed the baseline looking good on these
# same dates).
LEDGER_BOOKS = {
    "constructions": ["top3_ew", "top5_ew", "ls_quintile", "rank_ls", "rank_ls_cn"],
    "horizons": [5, 20],
    "window": 756,
    "model_neutral": False,
}

SERIES_STEP = 1
