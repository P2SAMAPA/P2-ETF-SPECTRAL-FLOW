"""
config.py  —  Configuration for P2 SPECTRAL-FLOW
"""

import os

HF_TOKEN = os.environ.get("HF_TOKEN")
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-etf-spectral-flow-results"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"],
    "COMBINED": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "VUG", "VTV", "SPYG", "QUAL", "IWR", "VO", "VB", "VIG", "VEA", "VGT", "VDE", "XLC", "IBB", "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA", "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE"]
}

TOP_N = 3

# Hand-labeled, static asset-class tags — used only by the asset-class-
# neutral ranking mode (see NEUTRALIZE_OPTIONS) and the dashboard coloring;
# just what each ETF IS, not a model input in the non-neutral mode.
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

# Classes with fewer members than this are NOT de-meaned within themselves in
# class-neutral mode (a singleton class de-meaned within itself becomes
# exactly zero on every day and then fills a top-3 slot as a meaningless 0.0%
# "pick" — seen on the sibling graphon engine's FI_COMMODITIES output, where
# VNQ is the only real-estate ETF). Their members are de-meaned against the
# universe-wide mean instead.
MIN_CLASS_SIZE = 3

# ---------------------------------------------------------------------------
# Spectral dynamics.
#
# Each day t, take the trailing CORR_WINDOW days of returns, form the
# correlation matrix C(t) (N x N), and eigendecompose it:
#     lambda_1(t) >= ... >= lambda_N(t),   eigenvectors u_k(t)
# (the eigenvalues of a correlation matrix sum to N). The engine models how
# the SPECTRUM MOVES — d(lambda_k)/dt over FLOW_LOOKBACK days — not returns:
#   * market-mode flow : relative change in lambda_1 (correlation building /
#                        dissolving across the whole universe)
#   * rotation flow    : relative change in the mean of lambda_2..lambda_K
#   * bulk flow        : relative change in lambda_{K+1}, the top edge of
#                        the remaining "bulk"
# and four market-level STRUCTURE statistics (diagnostics, tested against
# large aggregate moves — see LEAD_PARAMS):
#   * absorption ratio (Kritzman, Li, Page & Rigobon 2011, "Principal
#     Components as a Measure of Systemic Risk"): share of total variance in
#     the top ~N/5 eigenvalues
#   * relative spectral gap (lambda_1 - lambda_2) / lambda_1  ("gap collapse")
#   * bulk edge lambda_{K+1}                                  ("bulk expansion")
#   * mean consecutive-level-spacing ratio <r> (Oganesyan & Huse 2007; Atas
#     et al. 2013): ~0.53 for GOE-like level REPULSION, ~0.39 for Poisson-like
#     CONDENSATION/clustering; needs enough bulk eigenvalues, so only
#     computed when N >= MIN_N_SPACING.
#
# A market-level scalar (the same for every ETF on a given day) cannot rank
# ETFs by itself — cross-sectional z-scoring removes it. So the dynamics
# enter the ranking model only as INTERACTIONS with each ETF's own exposure
# to the moving mode (its squared eigenvector loading): "this mode is
# strengthening, and this ETF loads heavily on it." See spectral_model.py.
# ---------------------------------------------------------------------------
CORR_WINDOWS = [63, 126, 252]
FLOW_LOOKBACK = 10
TOP_MODES = 3
TS_Z_WINDOW = 252          # trailing window for time-series z-scoring of market-level flow scalars
TS_Z_MIN = 63
MIN_N_SPACING = 20

# Grid axis: de-mean features AND the forward-return target (and score the
# IC/spread against the de-meaned returns, not raw ones) within each asset
# class. Carried over from the sibling graphon engine, where neutralizing
# erased a COMBINED-universe "High" result that turned out to be a
# commodity-momentum call — and where scoring neutral rankings against RAW
# returns was found (by simulation) to understate within-class skill by up to
# ~half; that evaluation flaw is fixed here from the start. Spectral
# exposures are especially exposed to this confound, since the top
# eigenvectors ARE largely asset-class modes.
NEUTRALIZE_OPTIONS = [False, True]

# ---------------------------------------------------------------------------
# Prediction: pooled ridge regression of the cross-sectionally standardized
# forward H-day return on standardized features, refit at each evaluation
# date on the trailing WINDOW days. THREE nested models share the machinery:
#   price-only : mom5, mom21, vol21, beta
#   baseline   : price-only + STATIC spectral exposure (squared loadings on
#                the market mode and rotation modes) — i.e. ordinary
#                PCA/RMT-style structure with no dynamics
#   spectral   : baseline + the DYNAMICS (loading changes, and exposure x
#                eigenvalue-flow interactions)
# The gate and confidence compare `spectral` against `baseline`, so the
# dynamics must beat static spectral structure to earn credit; the
# price-only comparison is reported for context only.
# ---------------------------------------------------------------------------
HORIZONS = [1, 5, 20]
WINDOWS = [252, 504, 756, 1008]
RIDGE_LAMBDA = 0.01
MIN_TRAIN_DAYS = 60

SELECTION_FRACTION = 0.70
MIN_HOLDOUT_PERIODS = 30
TRADING_COST_BPS = 15
IC_T_HIGH = 1.65

# Selection eligibility requires an absolute AND relative selection-segment
# period-count floor, and confidence is capped at Low whenever the selection
# gate didn't pass — both carried over from the sibling engines after
# real-data runs there showed what goes wrong without them.
MIN_SELECTION_N = 50
MIN_SELECTION_N_FRACTION = 0.15

# Optional: pin the live configuration, e.g.
# {"corr_window": 126, "horizon": 5, "window": 756, "neutralize": True}
PINNED_CONFIG = None

# Structure-transition lead diagnostics: does a move in a market-level
# spectral statistic tend to sit before a large aggregate (equal-weight
# universe) move? Each statistic is converted to a trigger the same way
# (Kritzman et al.'s absorption-ratio "shift"): z = (mean of the last
# SHORT_AVG days - mean of the last LONG_AVG days) / std over LONG_AVG days;
# an onset is the CROSSING of +/- Z_THRESHOLD in the stated direction.
# Triggers, windows and thresholds are fixed here in advance, not tuned.
# NOTE: several triggers x several windows x several universes are tested, so
# a few "significant" p-values are expected by chance alone — the dashboard
# states the count.
LEAD_PARAMS = {
    "short_avg": 15,
    "long_avg": 252,
    "z_threshold": 1.5,
    "event_window": 10,
    "event_vol_k": 2.0,
    "dedup_days": 21,
    "lead_window": 20,
    "n_shifts": 300,
}
LEAD_TRIGGERS = {
    # name: (statistic, direction of crossing that counts as an onset)
    "absorption_up": ("absorption_ratio", "up"),
    "gap_collapse": ("rel_gap", "down"),
    "bulk_expansion": ("bulk_edge", "up"),
    "condensation": ("spacing_ratio", "down"),
    "repulsion": ("spacing_ratio", "up"),
}

SERIES_STEP = 3
