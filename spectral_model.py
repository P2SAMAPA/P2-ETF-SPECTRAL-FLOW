"""
spectral_model.py  —  Eigenvalue dynamics of the rolling correlation matrix

Idea
----
Ordinary RMT filtering / spectral graph methods use the spectrum of a
correlation matrix as a STATIC object: keep the eigenmodes above the
Marchenko-Pastur edge, discard the rest, build a graph from what is left.
This engine asks a different question — where is the spectrum MOVING?

Each day, take the trailing window of returns, form the N x N correlation
matrix C(t), and eigendecompose it: lambda_1(t) >= ... >= lambda_N(t) (they
sum to N). Then track the flow d(lambda_k)/dt over a short lookback:

  * market-mode flow   : lambda_1 rising = correlations building across the
                         whole universe; falling = dissolving
  * rotation flow      : the next few modes (typically asset-class / sector
                         modes) strengthening or weakening
  * bulk flow          : the top edge of the remaining bulk moving
  * structure statistics (market-level): absorption ratio, relative spectral
    gap, bulk edge, and the consecutive-level-spacing ratio <r> — which
    separates eigenvalue REPULSION (GOE-like, <r> ~ 0.53) from CONDENSATION /
    clustering (Poisson-like, <r> ~ 0.39) without needing to "unfold" the
    spectrum (Oganesyan & Huse 2007; Atas, Bogomolny, Giraud & Roux 2013).

Two uses, kept separate on purpose
----------------------------------
1. Ranking ETFs. A market-level scalar is identical for every ETF on a given
   day, so cross-sectional standardization erases it — "the gap collapsed
   today" cannot by itself say which ETF to prefer. The dynamics therefore
   enter ranking only as INTERACTIONS with each ETF's own exposure to the
   moving mode (its squared eigenvector loading): flow_k(t) * exposure_k,i(t).
   The interaction column is deliberately NOT re-standardized across ETFs
   (that would strip the flow's magnitude and keep only its sign — checked
   directly in testing; see README). Three nested models are compared: price
   features only; price + STATIC spectral exposure (the ordinary-PCA/RMT-style
   baseline); and baseline + the DYNAMICS. Only the last increment is the
   thesis, so the gate and confidence compare against the static-exposure
   baseline, not just against price features.
2. Detecting structure transitions. The market-level statistics are tested
   directly: does condensation / repulsion / gap collapse / bulk expansion /
   an absorption-ratio shift tend to sit before a large aggregate move, with a
   circular-shift permutation chance level and p-value?

Honest scope
------------
* Rank-ordered eigenvalue tracking blurs mode identity at avoided crossings
  (when lambda_2 and lambda_3 swap, "the 2nd mode" is a different eigenvector
  before and after). That is exactly the eigenvalue-repulsion phenomenon, not
  a bug, but it means the flow of the k-th eigenvalue is not always the flow
  of one persistent eigenmode. Only the market mode's loading change is
  tracked mode-by-mode (via absolute loadings, which are sign-safe).
* The correlation window and N are both small relative to what RMT theory
  assumes (q = N/W is 0.17-0.7 here), so level-spacing statistics are noisy
  per day; the trigger uses a 15-day average against its own year of history.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List, Optional, Tuple

PRICE_FEATURES = ["mom5", "mom21", "vol21", "beta"]
STATIC_FEATURES = ["e_mkt", "e_rot"]
DYN_FEATURES = ["mkt_load_chg", "idio_chg", "mkt_flow", "rot_flow", "bulk_flow"]
FEATURE_NAMES = PRICE_FEATURES + STATIC_FEATURES + DYN_FEATURES
N_PRICE = len(PRICE_FEATURES)
N_BASE = N_PRICE + len(STATIC_FEATURES)
N_FEAT = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Rolling helpers
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
    n, T = returns.shape
    cum = np.concatenate([np.zeros((1, T)), np.cumsum(returns, axis=0)], axis=0)
    fwd = np.full((n, T), np.nan)
    end = n - horizon
    if end > 0:
        fwd[:end] = cum[horizon + 1: n + 1] - cum[1: n - horizon + 1]
    return fwd


# ---------------------------------------------------------------------------
# Rolling correlation matrices and their eigendecomposition
# ---------------------------------------------------------------------------
def rolling_corr_matrices(returns: np.ndarray, window: int) -> np.ndarray:
    """(n,T,T) correlation matrix of the trailing `window` days; NaN rows before `window` days exist."""
    n, T = returns.shape
    s1 = trailing_sum(returns, window)
    s2 = trailing_sum(returns ** 2, window)
    cross = np.einsum("ti,tj->tij", returns, returns)
    s12 = trailing_sum(cross.reshape(n, T * T), window).reshape(n, T, T)
    mean1 = s1 / window
    var1 = np.maximum(s2 / window - mean1 ** 2, 1e-14)
    cov = s12 / window - mean1[:, :, None] * mean1[:, None, :]
    sd = np.sqrt(var1)
    corr = np.clip(cov / np.maximum(sd[:, :, None] * sd[:, None, :], 1e-14), -1.0, 1.0)
    idx = np.arange(T)
    corr[:, idx, idx] = 1.0
    return corr


def eigendecompose(C: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched symmetric eigendecomposition of the valid (finite) days.
    Returns lam (n,T) in DESCENDING order (NaN on invalid days) and u2 (n,T,K):
    squared eigenvector loadings of the top K modes (each mode's u2 sums to 1
    over ETFs).
    """
    n, T, _ = C.shape
    valid = np.isfinite(C).all(axis=(1, 2))
    lam = np.full((n, T), np.nan)
    u2 = np.full((n, T, top_k), np.nan)
    if valid.any():
        Cv = 0.5 * (C[valid] + np.swapaxes(C[valid], 1, 2))
        w, v = np.linalg.eigh(Cv)                     # ascending
        lam[valid] = w[:, ::-1]
        top = v[:, :, ::-1][:, :, :top_k]             # columns = top eigenvectors
        u2[valid] = top ** 2
    return lam, u2, valid


def spacing_ratio_mean(levels: np.ndarray) -> np.ndarray:
    """
    Mean consecutive-level-spacing ratio  r~_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    per row of `levels` (m, M), spacings taken between adjacent sorted
    levels. ~0.531 for GOE-like repulsion, 2*ln2-1 ~ 0.386 for independent
    (Poisson) levels. Rows with <3 levels, or only degenerate spacings, are NaN.
    """
    lv = np.sort(levels, axis=1)
    s = np.diff(lv, axis=1)
    if s.shape[1] < 2:
        return np.full(levels.shape[0], np.nan)
    a, b = s[:, :-1], s[:, 1:]
    mx = np.maximum(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(mx > 1e-12, np.minimum(a, b) / mx, np.nan)
    return np.nanmean(r, axis=1)


def structure_statistics(lam: np.ndarray, top_k: int, min_n_spacing: int) -> Dict[str, np.ndarray]:
    """Market-level spectral statistics per day (each (n,))."""
    n, T = lam.shape
    k_ar = max(1, int(round(T / 5.0)))
    absorption = np.nansum(lam[:, :k_ar], axis=1) / T
    absorption[~np.isfinite(lam[:, 0])] = np.nan
    rel_gap = (lam[:, 0] - lam[:, 1]) / lam[:, 0]
    bulk_edge = lam[:, top_k] if top_k < T else np.full(n, np.nan)
    spacing = np.full(n, np.nan)
    if T >= min_n_spacing:
        ok = np.isfinite(lam).all(axis=1)
        if ok.any():
            spacing[ok] = spacing_ratio_mean(lam[ok][:, top_k:])
    return {"absorption_ratio": absorption, "rel_gap": rel_gap, "bulk_edge": bulk_edge,
            "spacing_ratio": spacing}


def ts_zscore(x: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return ((s - mu) / sd.replace(0, np.nan)).values


def lagged(a: np.ndarray, k: int) -> np.ndarray:
    out = np.roll(a, k, axis=0)
    out[:k] = np.nan
    return out


# ---------------------------------------------------------------------------
# Class-neutral standardization (carried over from the graphon engine, with
# the singleton-class fix)
# ---------------------------------------------------------------------------
def group_ids_from_map(tickers: List[str], class_map: Dict[str, str], min_size: int) -> np.ndarray:
    labels = [class_map.get(t) for t in tickers]
    counts = Counter(l for l in labels if l)
    uniq = sorted(l for l, c in counts.items() if c >= min_size)
    idx = {l: i for i, l in enumerate(uniq)}
    return np.array([idx.get(l, -1) if l else -1 for l in labels])


def demean_by_group(a: np.ndarray, gids: np.ndarray) -> np.ndarray:
    """
    Subtract each ticker's own class mean (nanmean across the class's members,
    per day). Tickers in group -1 (unlabeled or in a class smaller than
    MIN_CLASS_SIZE) get the universe-wide mean subtracted instead, so a
    singleton class is never de-meaned against itself into a constant zero.
    """
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
    """Cross-sectional z-score; class-neutral (de-mean within class first) when gids is given."""
    return cs_zscore(a if gids is None else demean_by_group(a, gids))


# ---------------------------------------------------------------------------
# The spectrum and the feature tensor
# ---------------------------------------------------------------------------
def build_spectrum(returns: np.ndarray, window: int, top_modes: int, flow_lookback: int,
                   ts_z_window: int, ts_z_min: int, min_n_spacing: int) -> Dict:
    n, T = returns.shape
    K = int(min(top_modes, max(T - 2, 1)))
    C = rolling_corr_matrices(returns, window)
    lam, u2, valid = eigendecompose(C, K)
    struct = structure_statistics(lam, K, min_n_spacing)

    with np.errstate(invalid="ignore", divide="ignore"):
        lam1 = lam[:, 0]
        lam_rot = np.nanmean(lam[:, 1:K], axis=1) if K > 1 else np.full(n, np.nan)
        lam_bulk = lam[:, K] if K < T else np.full(n, np.nan)
        s_mkt = np.log(lam1 / lagged(lam1[:, None], flow_lookback)[:, 0])
        s_rot = np.log(lam_rot / lagged(lam_rot[:, None], flow_lookback)[:, 0])
        s_bulk = np.log(lam_bulk / lagged(lam_bulk[:, None], flow_lookback)[:, 0])
    # Winsorized at +/-3: a crisis day's log-ratio of lambda_1 is a many-sigma
    # outlier, and multiplied by a standardized exposure it would otherwise
    # create interaction values of +/-20 that dominate a pooled ridge fit.
    z_mkt = np.clip(ts_zscore(s_mkt, ts_z_window, ts_z_min), -3.0, 3.0)
    z_rot = np.clip(ts_zscore(s_rot, ts_z_window, ts_z_min), -3.0, 3.0)
    z_bulk = np.clip(ts_zscore(s_bulk, ts_z_window, ts_z_min), -3.0, 3.0)

    e_mkt = u2[:, :, 0]
    e_rot = u2[:, :, 1:K].sum(axis=2) if K > 1 else np.full((n, T), np.nan)
    e_idio = 1.0 - u2.sum(axis=2)
    a_mkt = np.sqrt(u2[:, :, 0])
    mkt_load_chg = a_mkt - lagged(a_mkt, flow_lookback)
    idio_chg = e_idio - lagged(e_idio, flow_lookback)

    return {"K": K, "lam": lam, "u2": u2, "valid": valid, "struct": struct,
            "flow": {"s_mkt": s_mkt, "s_rot": s_rot, "s_bulk": s_bulk,
                     "z_mkt": z_mkt, "z_rot": z_rot, "z_bulk": z_bulk},
            "expo": {"e_mkt": e_mkt, "e_rot": e_rot, "e_idio": e_idio,
                     "mkt_load_chg": mkt_load_chg, "idio_chg": idio_chg}}


def build_features(returns: np.ndarray, spec: Dict, window: int, gids: Optional[np.ndarray]) -> np.ndarray:
    """
    (n,T,N_FEAT) standardized feature tensor. Price features, static spectral
    exposures and the two ETF-level dynamics (loading changes) are
    cross-sectionally standardized (class-neutral when `gids` is given). The
    three exposure x flow INTERACTIONS are standardized exposure times the
    time-series-z-scored market-level flow scalar, and are left as products —
    NOT re-standardized across ETFs, which would erase the flow's magnitude.
    """
    n, T = returns.shape
    mom5 = trailing_sum(returns, 5)
    mom21 = trailing_sum(returns, 21)
    s1, s2 = trailing_sum(returns, 21), trailing_sum(returns ** 2, 21)
    vol21 = 0.5 * np.log(np.maximum((s2 - s1 ** 2 / 21) / 20, 1e-10))
    mkt = np.nanmean(returns, axis=1, keepdims=True)
    sm, smm = trailing_sum(returns * mkt, window), trailing_sum(mkt ** 2, window)
    r1, m1 = trailing_sum(returns, window), trailing_sum(mkt, window)
    cov = sm / window - (r1 / window) * (m1 / window)
    varm = np.maximum(smm / window - (m1 / window) ** 2, 1e-14)
    beta = cov / varm

    ex, fl = spec["expo"], spec["flow"]
    cols = [mom5, mom21, vol21, beta, ex["e_mkt"], ex["e_rot"], ex["mkt_load_chg"], ex["idio_chg"]]
    std_cols = [standardize(c, gids) for c in cols]
    f_mkt = standardize(ex["e_mkt"], gids) * fl["z_mkt"][:, None]
    f_rot = standardize(ex["e_rot"], gids) * fl["z_rot"][:, None]
    f_bulk = standardize(ex["e_idio"], gids) * fl["z_bulk"][:, None]

    feats = np.full((n, T, N_FEAT), np.nan)
    for j, c in enumerate(std_cols):
        feats[:, :, j] = c
    feats[:, :, 8] = f_mkt
    feats[:, :, 9] = f_rot
    feats[:, :, 10] = f_bulk
    return feats


# ---------------------------------------------------------------------------
# Cross-sectional regression machinery
# ---------------------------------------------------------------------------
def build_cross_products(Fz: np.ndarray, Yz: np.ndarray) -> Dict:
    n, T, p = Fz.shape
    feat_ok = np.isfinite(Fz).all(axis=(1, 2))
    y_ok = np.isfinite(Yz).all(axis=1)
    ok = feat_ok & y_ok
    F0 = np.where(ok[:, None, None], np.nan_to_num(Fz), 0.0)
    Y0 = np.where(ok[:, None], np.nan_to_num(Yz), 0.0)
    XtX = np.einsum("dtp,dtq->dpq", F0, F0)
    Xty = np.einsum("dtp,dt->dp", F0, Y0)
    return {
        "cXtX": np.concatenate([np.zeros((1, p, p)), np.cumsum(XtX, axis=0)], axis=0),
        "cXty": np.concatenate([np.zeros((1, p)), np.cumsum(Xty, axis=0)], axis=0),
        "cN": np.concatenate([[0], np.cumsum(ok.astype(int))]),
        "feat_ok": feat_ok,
    }


def ridge_solve(XtX: np.ndarray, Xty: np.ndarray, lam_rel: float) -> np.ndarray:
    p = XtX.shape[0]
    lam = lam_rel * np.trace(XtX) / p + 1e-12
    return np.linalg.solve(XtX + lam * np.eye(p), Xty)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return 0.0
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def walk_forward(Fz: np.ndarray, cp: Dict, fwd_eval: np.ndarray, horizon: int, window: int,
                 top_n: int, cfg: Dict) -> Optional[Dict]:
    """
    Three nested models on identical dates: spectral (all features), baseline
    (price + static spectral exposure), price-only. `fwd_eval` is what ICs and
    spreads are SCORED against — raw forward returns normally, class-de-meaned
    forward returns in class-neutral mode (scoring neutral rankings against
    raw returns leaves the class-level moves in as pure noise and understates
    within-class skill, measured by simulation at up to ~half).
    """
    n, T, p = Fz.shape
    base_idx = list(range(N_BASE))
    price_idx = list(range(N_PRICE))
    lam = cfg["ridge_lambda"]
    min_train = cfg["min_train_days"]
    cost = cfg["cost_bps"] / 1e4

    rows, ic_m, ic_b, ic_p, sg_m, sg_b, sn_m, sn_b = [], [], [], [], [], [], [], []
    prev_m, prev_b = set(), set()
    i = 0
    while i <= n - 1 - horizon:
        a, b = max(0, i - window), i - horizon
        if b < a or not cp["feat_ok"][i]:
            i += 1 if b < a else horizon
            continue
        if cp["cN"][b + 1] - cp["cN"][a] < min_train:
            i += horizon
            continue
        XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
        Xty = cp["cXty"][b + 1] - cp["cXty"][a]
        try:
            beta_m = ridge_solve(XtX, Xty, lam)
            beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], lam)
            beta_p = ridge_solve(XtX[np.ix_(price_idx, price_idx)], Xty[price_idx], lam)
        except np.linalg.LinAlgError:
            i += horizon
            continue
        f = Fz[i]
        s_m, s_b, s_p = f @ beta_m, f[:, base_idx] @ beta_b, f[:, price_idx] @ beta_p
        r = fwd_eval[i]
        if not np.isfinite(r).all():
            i += horizon
            continue
        top_m = set(np.argsort(-s_m)[:top_n].tolist())
        top_b = set(np.argsort(-s_b)[:top_n].tolist())
        g_m = r[list(top_m)].mean() - r.mean()
        g_b = r[list(top_b)].mean() - r.mean()
        c_m = (top_n - len(top_m & prev_m)) / top_n * cost
        c_b = (top_n - len(top_b & prev_b)) / top_n * cost
        rows.append(i)
        ic_m.append(spearman(s_m, r))
        ic_b.append(spearman(s_b, r))
        ic_p.append(spearman(s_p, r))
        sg_m.append(g_m)
        sg_b.append(g_b)
        sn_m.append(g_m - c_m)
        sn_b.append(g_b - c_b)
        prev_m, prev_b = top_m, top_b
        i += horizon

    if len(rows) < 10:
        return None
    return {k: np.array(v) for k, v in dict(rows=rows, ic_m=ic_m, ic_b=ic_b, ic_p=ic_p, sg_m=sg_m,
                                              sg_b=sg_b, sn_m=sn_m, sn_b=sn_b).items()}


def summarize(wf: Dict, lo: int, hi: int, horizon: int) -> Dict:
    sl = slice(lo, hi)
    ic_m, ic_b, ic_p = wf["ic_m"][sl], wf["ic_b"][sl], wf["ic_p"][sl]
    sn_m, sn_b, sg_m = wf["sn_m"][sl], wf["sn_b"][sl], wf["sg_m"][sl]
    n = len(ic_m)
    if n < 3:
        return {"n": n}
    ann = np.sqrt(252.0 / horizon)
    return {
        "n": int(n),
        "ic_mean": float(ic_m.mean()), "ic_t": tstat(ic_m),
        "ic_base_mean": float(ic_b.mean()), "ic_base_t": tstat(ic_b),
        "ic_price_mean": float(ic_p.mean()),
        "static_gain_ic": float((ic_b - ic_p).mean()), "static_gain_t": tstat(ic_b - ic_p),
        "net_spread_mean": float(sn_m.mean()), "net_spread_t": tstat(sn_m),
        "gross_spread_mean": float(sg_m.mean()),
        "base_net_spread_mean": float(sn_b.mean()),
        "hit_rate": float((sn_m > 0).mean()),
        "sharpe_net": float(sn_m.mean() / (sn_m.std() + 1e-12) * ann),
        "sharpe_gross": float(sg_m.mean() / (sg_m.std() + 1e-12) * ann),
        "d_ic_mean": float((ic_m - ic_b).mean()), "d_ic_t": tstat(ic_m - ic_b),
        "d_spread_mean": float((sn_m - sn_b).mean()), "d_spread_t": tstat(sn_m - sn_b),
    }


def live_scores(Fz: np.ndarray, cp: Dict, fwd: np.ndarray, horizon: int, window: int,
                cfg: Dict) -> Optional[Dict]:
    n = Fz.shape[0]
    i = n - 1
    a, b = max(0, i - window), i - horizon
    if b < a or not cp["feat_ok"][i] or cp["cN"][b + 1] - cp["cN"][a] < cfg["min_train_days"]:
        return None
    XtX = cp["cXtX"][b + 1] - cp["cXtX"][a]
    Xty = cp["cXty"][b + 1] - cp["cXty"][a]
    base_idx = list(range(N_BASE))
    beta_m = ridge_solve(XtX, Xty, cfg["ridge_lambda"])
    beta_b = ridge_solve(XtX[np.ix_(base_idx, base_idx)], Xty[base_idx], cfg["ridge_lambda"])
    f = Fz[i]
    disp = np.nanmean(np.nanstd(fwd[a:b + 1], axis=1))
    return {"score_m": f @ beta_m, "score_b": f[:, base_idx] @ beta_b, "beta_m": beta_m,
            "dispersion": float(disp) if np.isfinite(disp) else 0.0}


# ---------------------------------------------------------------------------
# Structure-transition lead diagnostics
# ---------------------------------------------------------------------------
def _onsets(flag: np.ndarray, dedup: int) -> np.ndarray:
    out, last = [], -10 ** 9
    for t in np.where(flag)[0]:
        if t - last > dedup:
            out.append(t)
            last = t   # advance only on a RECORDED onset (advancing on every flagged day breaks deduplication)
    return np.array(out, dtype=int)


def shift_trigger_z(series: np.ndarray, short_avg: int, long_avg: int) -> np.ndarray:
    """Kritzman et al.-style 'shift': (short-window mean - long-window mean) / long-window std."""
    s = pd.Series(series)
    short = s.rolling(short_avg, min_periods=short_avg).mean()
    lmean = s.rolling(long_avg, min_periods=long_avg // 2).mean()
    lstd = s.rolling(long_avg, min_periods=long_avg // 2).std()
    return ((short - lmean) / lstd.replace(0, np.nan)).values


def _count_hits(on: np.ndarray, ev: np.ndarray, LW: int) -> Tuple[int, int]:
    """(# events preceded by an onset within LW days, # onsets NOT followed by an event within LW days)."""
    if len(ev) == 0 or len(on) == 0:
        return 0, len(on)
    on_s, ev_s = np.sort(on), np.sort(ev)
    hits = int(np.sum(np.searchsorted(on_s, ev_s, side="left") - np.searchsorted(on_s, ev_s - LW, side="left") > 0))
    foll = np.searchsorted(ev_s, on_s + LW, side="right") - np.searchsorted(ev_s, on_s, side="right")
    return hits, int(np.sum(foll == 0))


def lead_diagnostic(z: np.ndarray, direction: str, agg_returns: np.ndarray, agg_vol21: np.ndarray,
                    lp: Dict, seed: int = 0) -> Dict:
    """
    Does the trigger's onset (z crossing +threshold for 'up', -threshold for
    'down'; the crossing moment, not the whole stretch) tend to sit within
    `lead_window` days before a large AGGREGATE move? Chance level and p-value
    come from circular-shifting the deduplicated onset times against the fixed
    event times (preserves the onsets' spacing pattern; an analytic
    uniform-placement chance formula was found biased on independent data in
    the sibling hyperbolic-flows engine). p-value = P(shifted hit count >=
    observed).
    """
    n = len(z)
    W = lp["event_window"]
    fwd_cum = np.full(n, np.nan)
    if n > W:
        cum = np.concatenate([[0.0], np.cumsum(agg_returns)])
        fwd_cum[: n - W] = cum[W + 1: n + 1] - cum[1: n - W + 1]
    event_flag = np.abs(fwd_cum) > lp["event_vol_k"] * agg_vol21 * np.sqrt(W)

    thr = lp["z_threshold"]
    beyond = (np.nan_to_num(z, nan=0.0) > thr) if direction == "up" else (np.nan_to_num(z, nan=0.0) < -thr)
    prev_not = np.roll(~beyond, 1)
    prev_not[0] = False
    onset_flag = beyond & prev_not

    ok = np.isfinite(z) & np.isfinite(fwd_cum)
    if not ok.any():
        return {"available": False}
    start = int(np.argmax(ok))
    dedup, LW = lp["dedup_days"], lp["lead_window"]
    ev = _onsets(event_flag & ok, dedup)
    on = _onsets(onset_flag & ok, dedup)
    on, ev = on[on >= start], ev[ev >= start]
    if len(ev) == 0 or len(on) == 0:
        return {"available": True, "n_onsets": int(len(on)), "n_large_move_events": int(len(ev)),
                "hit_rate": None, "chance_hit_rate": None, "p_value": None, "median_lead_days": None,
                "false_alarm_rate": None, "chance_false_alarm_rate": None}

    hits, false_alarms = _count_hits(on, ev, LW)
    leads = []
    for d in ev:
        prior = on[(on >= d - LW) & (on < d)]
        if len(prior):
            leads.append(int(d - prior.min()))

    rng = np.random.default_rng(seed)
    span = n - start
    shifts = rng.integers(0, span, size=lp["n_shifts"])
    hc = np.empty(len(shifts))
    fc = np.empty(len(shifts))
    for k, sh in enumerate(shifts):
        on_mod = start + np.mod(on + sh - start, span)
        hc[k], fc[k] = _count_hits(on_mod, ev, LW)
    return {
        "available": True,
        "n_onsets": int(len(on)), "n_large_move_events": int(len(ev)),
        "hit_rate": float(hits / len(ev)), "chance_hit_rate": float(hc.mean() / len(ev)),
        "p_value": float((1 + np.sum(hc >= hits)) / (1 + len(hc))),
        "median_lead_days": float(np.median(leads)) if leads else None,
        "false_alarm_rate": float(false_alarms / len(on)), "chance_false_alarm_rate": float(fc.mean() / len(on)),
    }
