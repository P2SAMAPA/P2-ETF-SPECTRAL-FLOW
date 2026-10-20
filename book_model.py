"""
book_model.py  —  Turning the plain price baseline into books, and costing them

One score, many books
---------------------
The score is a ridge regression on four price features only (mom5, mom21,
vol21, beta), refit at each rebalance on a trailing window and used to rank the
ETFs of a universe. Seven different books are then built from the SAME ranking
so that the only thing varying is how the ranking is turned into positions:

  top3_ew / top5_ew / topq_ew   long-only equal-weight top N / top quintile,
                                measured ACTIVE vs the equal-weight universe
  ls_quintile                   long top quintile, short bottom quintile
  rank_ls                       weights proportional to centered rank,
                                dollar-neutral (long sum +1, short sum -1)
  ls_quintile_cn / rank_ls_cn   the same two long/short books built WITHIN each
                                asset class and scaled by class size, so each
                                class's longs and shorts offset — class-neutral
                                by construction, hence tradable as a
                                class-neutral book on RAW returns

Every book is evaluated on non-overlapping holding periods of `horizon` days
with costs charged on actual turnover: sum|w_t - w_{t-1}| * (round-trip
cost)/2 — for the top-3 long-only book this reproduces the sibling engines'
"fraction of picks replaced x round-trip cost" exactly (checked) — plus a
short-borrow charge on short notional for the long/short books.

"Is it alpha or just market exposure?"
--------------------------------------
Each book's net return series is regressed on the equal-weight universe's
forward return over the same periods: slope = market beta, intercept = alpha
per period, with an alpha t-statistic. A long/short book that is net long
high-beta names makes money in a rising market without any ranking skill; the
verdict therefore requires the ALPHA t-statistic, not just the raw P&L's.

"What is the signal?"
---------------------
`factor_attribution` reports each of the four raw features' own cross-sectional
rank-IC against forward returns (raw and class-neutral) so a blended score's
IC can be traced to momentum, a volatility tilt, or a beta tilt.

Honest scope
------------
* The ridge weights, the grid and the whole study were designed AFTER earlier
  engines had already shown this baseline looking good on these same dates, so
  the backtest here cannot be fully out-of-sample no matter how the split is
  drawn. The forward ledger (ledger.py) is the only evidence free of that.
* Long-only "active vs equal-weight" books are tradable only as a relative
  position (own the top N instead of the whole universe, or hedge the
  universe); the hedge's own cost is not modeled.
* Shorting costs are an assumption (config.SHORT_BORROW_BPS_ANNUAL); taxes,
  market impact, and borrow availability are not modeled.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple

import config

PRICE_FEATURES = ["mom5", "mom21", "vol21", "beta"]
N_FEAT = len(PRICE_FEATURES)


# ---------------------------------------------------------------------------
# Rolling helpers and forward returns
# ---------------------------------------------------------------------------
def rolling_sum(a: np.ndarray, w: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, axis=0)], axis=0)
    lo = np.maximum(np.arange(a.shape[0]) - w + 1, 0)
    return c[1:] - c[lo]


def trailing_sum(a: np.ndarray, w: int) -> np.ndarray:
    out = rolling_sum(a, w)
    out[: w - 1] = np.nan
    return out


def compute_forward_returns(returns: np.ndarray, horizon: int) -> np.ndarray:
    """fwd[i] = sum of returns over days i+1 .. i+horizon (NaN where not fully observed)."""
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    end = n - horizon
    if end > 0:
        fwd[:end] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


def build_price_features(returns: np.ndarray, beta_window: int) -> np.ndarray:
    """Raw (n,T,4) features: mom5, mom21, vol21 (log), beta to the equal-weight universe return."""
    mom5 = trailing_sum(returns, 5)
    mom21 = trailing_sum(returns, 21)
    s1, s2 = trailing_sum(returns, 21), trailing_sum(returns ** 2, 21)
    vol21 = 0.5 * np.log(np.maximum((s2 - s1 ** 2 / 21) / 20, 1e-10))
    mkt = np.nanmean(returns, axis=1, keepdims=True)
    sm_, smm = trailing_sum(returns * mkt, beta_window), trailing_sum(mkt ** 2, beta_window)
    r1, m1 = trailing_sum(returns, beta_window), trailing_sum(mkt, beta_window)
    cov = sm_ / beta_window - (r1 / beta_window) * (m1 / beta_window)
    varm = np.maximum(smm / beta_window - (m1 / beta_window) ** 2, 1e-14)
    beta = cov / varm
    return np.stack([mom5, mom21, vol21, beta], axis=2)


# ---------------------------------------------------------------------------
# Standardization (class-neutral option), carried over from the sibling engines
# ---------------------------------------------------------------------------
def group_ids_from_map(tickers: List[str], class_map: Dict[str, str], min_size: int) -> np.ndarray:
    labels = [class_map.get(t) for t in tickers]
    counts = Counter(l for l in labels if l)
    uniq = sorted(l for l, c in counts.items() if c >= min_size)
    idx = {l: i for i, l in enumerate(uniq)}
    return np.array([idx.get(l, -1) if l else -1 for l in labels])


def demean_by_group(a: np.ndarray, gids: np.ndarray) -> np.ndarray:
    """Subtract each ticker's class mean per day; group -1 (small/unlabeled classes) uses the universe-wide mean."""
    out = a.astype(float).copy()
    with np.errstate(invalid="ignore"):
        gmean = np.nanmean(a, axis=1, keepdims=True)
        for g in np.unique(gids):
            m = gids == g
            if g < 0:
                out[:, m] = a[:, m] - gmean if a.ndim == 2 else a[:, m, :] - gmean
            else:
                cm = np.nanmean(a[:, m] if a.ndim == 2 else a[:, m, :], axis=1, keepdims=True)
                if a.ndim == 2:
                    out[:, m] = a[:, m] - cm
                else:
                    out[:, m, :] = a[:, m, :] - cm
    out[~np.isfinite(a)] = np.nan
    return out


def cs_zscore(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(a, axis=1, keepdims=True)
        sd = np.nanstd(a, axis=1, keepdims=True)
        out = np.where(sd > 1e-9, (a - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    out[~np.isfinite(a)] = np.nan
    return out


def standardize(a: np.ndarray, gids: Optional[np.ndarray]) -> np.ndarray:
    return cs_zscore(a if gids is None else demean_by_group(a, gids))


# ---------------------------------------------------------------------------
# Ridge scores
# ---------------------------------------------------------------------------
def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    ok = feat_ok & np.isfinite(Yz).all(axis=1)
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {"cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
            "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
            "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]), "feat_ok": feat_ok}


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    lam = lam_rel * np.trace(XtX) / p + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(p), Xty)


def walk_forward_scores(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int) -> Optional[Dict]:
    """
    Scores at non-overlapping rebalance rows spaced `horizon` days apart. The
    fit at row i uses only rows <= i - horizon, whose forward returns have
    fully realized by day i — no look-ahead.
    """
    n = Fz.shape[0]
    rows, S, R = [], [], []
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < config.MIN_TRAIN_DAYS:
            i += horizon
            continue
        r = fwd[i]
        if not np.isfinite(r).all():
            i += horizon
            continue
        try:
            beta = ridge_solve(cp["cXtX"][b + 1] - cp["cXtX"][a], cp["cXty"][b + 1] - cp["cXty"][a], config.RIDGE_LAMBDA)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        rows.append(i)
        S.append(Fz[i] @ beta)
        R.append(r)
        i += horizon
    if len(rows) < 10:
        return None
    return {"rows": np.array(rows), "S": np.array(S), "R": np.array(R)}


def live_score(Fz: np.ndarray, cp: Dict, horizon: int, window: int) -> Optional[np.ndarray]:
    n = Fz.shape[0]
    i = n - 1
    a, b = max(0, i - window), i - horizon
    if b < a or not cp["feat_ok"][i] or cp["cN"][b + 1] - cp["cN"][a] < config.MIN_TRAIN_DAYS:
        return None
    beta = ridge_solve(cp["cXtX"][b + 1] - cp["cXtX"][a], cp["cXty"][b + 1] - cp["cXty"][a], config.RIDGE_LAMBDA)
    return Fz[i] @ beta


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------
def _top_n(score: np.ndarray, n: int) -> np.ndarray:
    T = len(score)
    n = int(max(1, min(n, T)))
    w = np.zeros(T)
    w[np.argsort(-score)[:n]] = 1.0 / n
    return w


def _quintile_ls(score: np.ndarray, quantile: float) -> np.ndarray:
    T = len(score)
    w = np.zeros(T)
    if T < 2:
        return w
    k = int(min(max(1, round(quantile * T)), T // 2))
    order = np.argsort(-score)
    w[order[:k]] = 1.0 / k
    w[order[-k:]] = -1.0 / k
    return w


def _rank_ls(score: np.ndarray) -> np.ndarray:
    T = len(score)
    w = np.zeros(T)
    if T < 2:
        return w
    ranks = np.argsort(np.argsort(score)).astype(float)       # 0 = lowest score
    c = ranks - (T - 1) / 2.0
    pos = c[c > 0].sum()
    return c / pos if pos > 0 else w


def _within_groups(score: np.ndarray, gids: np.ndarray, fn) -> np.ndarray:
    """Apply a dollar-neutral book builder inside each group, scaled by group size so total long = +1 and short = -1."""
    T = len(score)
    w = np.zeros(T)
    usable = 0
    parts = []
    for g in np.unique(gids):
        m = np.where(gids == g)[0]
        if len(m) < 2:
            continue
        parts.append((m, fn(score[m])))
        usable += len(m)
    for m, wg in parts:
        w[m] = wg * (len(m) / usable)
    return w


def book_weights(kind: str, score: np.ndarray, gids: np.ndarray) -> np.ndarray:
    T = len(score)
    if kind == "top3_ew":
        return _top_n(score, 3)
    if kind == "top5_ew":
        return _top_n(score, 5)
    if kind == "topq_ew":
        return _top_n(score, round(config.QUANTILE * T))
    if kind == "ls_quintile":
        return _quintile_ls(score, config.QUANTILE)
    if kind == "rank_ls":
        return _rank_ls(score)
    if kind == "ls_quintile_cn":
        return _within_groups(score, gids, lambda s: _quintile_ls(s, config.QUANTILE))
    if kind == "rank_ls_cn":
        return _within_groups(score, gids, _rank_ls)
    raise ValueError("unknown construction: " + kind)


def turnover_cost(w: np.ndarray, prev: Optional[np.ndarray], horizon: int, long_only: bool,
                  cost_bps: float, borrow_bps_annual: float) -> Tuple[float, float]:
    """(turnover, cost): cost = turnover * cost_bps/2 per unit traded, plus short-borrow on short notional."""
    to = 0.0 if prev is None else float(np.abs(w - prev).sum())
    cost = to * cost_bps / 2e4
    if not long_only:
        cost += float(np.abs(w[w < 0]).sum()) * borrow_bps_annual / 1e4 * horizon / 252.0
    return to, cost


def evaluate_book(kind: str, S: np.ndarray, R: np.ndarray, gids: np.ndarray, horizon: int,
                  cost_bps: float = None, borrow_bps_annual: float = None) -> Dict[str, np.ndarray]:
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    borrow = config.SHORT_BORROW_BPS_ANNUAL if borrow_bps_annual is None else borrow_bps_annual
    long_only = kind in config.LONG_ONLY
    m = S.shape[0]
    gross, cost, turn = np.empty(m), np.empty(m), np.empty(m)
    ew = R.mean(axis=1)
    prev = None
    for t in range(m):
        w = book_weights(kind, S[t], gids)
        gross[t] = w @ R[t] - (ew[t] if long_only else 0.0)
        turn[t], cost[t] = turnover_cost(w, prev, horizon, long_only, cost_bps, borrow)
        prev = w
    return {"gross": gross, "net": gross - cost, "turnover": turn, "ew": ew}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def market_regression(net: np.ndarray, ew: np.ndarray) -> Dict[str, float]:
    """OLS of net returns on the equal-weight universe return: beta, per-period alpha, alpha t-statistic."""
    n = len(net)
    if n < 5:
        return {"beta": None, "alpha": None, "alpha_t": None}
    x = ew - ew.mean()
    sxx = float((x ** 2).sum())
    if sxx <= 0:
        return {"beta": 0.0, "alpha": float(net.mean()), "alpha_t": tstat(net)}
    beta = float((x * (net - net.mean())).sum() / sxx)
    alpha = float(net.mean() - beta * ew.mean())
    resid = net - alpha - beta * ew
    s2 = float((resid ** 2).sum() / max(n - 2, 1))
    se = np.sqrt(s2 * (1.0 / n + ew.mean() ** 2 / sxx))
    return {"beta": beta, "alpha": alpha, "alpha_t": float(alpha / se) if se > 0 else 0.0}


def summarize(ev: Dict[str, np.ndarray], lo: int, hi: int, horizon: int) -> Dict:
    sl = slice(lo, hi)
    net, gross, turn, ew = ev["net"][sl], ev["gross"][sl], ev["turnover"][sl], ev["ew"][sl]
    n = len(net)
    if n < 3:
        return {"n": int(n)}
    ann = np.sqrt(252.0 / horizon)
    cum = np.cumsum(net)
    out = {
        "n": int(n),
        "gross_mean": float(gross.mean()), "net_mean": float(net.mean()), "net_t": tstat(net),
        "net_sharpe": float(net.mean() / (net.std() + 1e-12) * ann),
        "hit_rate": float((net > 0).mean()), "turnover_mean": float(turn.mean()),
        "cost_mean": float(gross.mean() - net.mean()),
        "max_drawdown": float((cum - np.maximum.accumulate(cum)).min()),
    }
    out.update(market_regression(net, ew))
    return out


def factor_attribution(F_raw: np.ndarray, fwd: np.ndarray, gids: np.ndarray, horizon: int,
                       split_fraction: float) -> Dict:
    """Each raw feature's own cross-sectional rank-IC at non-overlapping dates, raw and class-neutral, full and holdout."""
    n, T, p = F_raw.shape
    Fn = demean_by_group(F_raw, gids)
    fwdn = demean_by_group(fwd, gids)
    rows = [i for i in range(0, n - horizon, horizon)
            if np.isfinite(F_raw[i]).all() and np.isfinite(fwd[i]).all()]
    if len(rows) < 30:
        return {}
    split = int(len(rows) * split_fraction)
    out = {}
    for j, nm in enumerate(PRICE_FEATURES):
        res = {}
        for mode, F_, Y_ in [("raw", F_raw, fwd), ("neutral", Fn, fwdn)]:
            ics = np.array([spearman(F_[i, :, j], Y_[i]) for i in rows])
            res[mode] = {"ic": float(ics.mean()), "t": tstat(ics), "n": int(len(ics)),
                         "ic_holdout": float(ics[split:].mean()), "t_holdout": tstat(ics[split:])}
        out[nm] = res
    return out
