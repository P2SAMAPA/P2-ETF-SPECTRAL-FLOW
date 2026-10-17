"""
trainer.py  —  P2 SPECTRAL-FLOW trainer

Tracks the eigenvalue spectrum of each universe's rolling correlation matrix
and asks two separate questions:

  1. RANKING. Does where the spectrum is MOVING (market-mode, rotation-mode
     and bulk flow, interacted with each ETF's own eigenvector exposure,
     plus loading changes) help rank ETFs by forward return relative to the
     universe, over and above (a) price features and (b) STATIC spectral
     exposure — i.e. ordinary PCA/RMT-style structure with no dynamics?
  2. TRANSITIONS. Do market-level structure statistics — absorption-ratio
     shift, spectral gap collapse, bulk expansion, level condensation, level
     repulsion — tend to sit before a large aggregate move, against a
     circular-shift permutation chance level with a p-value?

Design choices, each carried over from a sibling engine for a stated reason:
  * Cross-sectional standardization; non-overlapping test points one horizon
    apart; Sharpe annualized by sqrt(252/horizon); cost on the fraction of
    top-N replaced.
  * Baseline ablation — here a THREE-level one (price-only / + static spectral
    exposure / + dynamics), with the gate and confidence comparing against the
    static-exposure baseline so only the dynamics increment earns credit.
  * Selection/holdout split; absolute AND relative selection-period floors;
    confidence capped at Low whenever the selection gate did not pass.
  * Asset-class-neutral mode as a grid axis, with IC/spread SCORED against
    class-neutral returns in that mode (scoring against raw returns understates
    within-class skill) and small classes handled so no ticker is zeroed out.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import spectral_model as sm
from data_manager import load_master_data, validate_data

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _clean(d: Dict) -> Dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer,)) else v))
            for k, v in d.items()}


def confidence_label(hold: Dict, gate_passed: bool = True) -> str:
    """
    Graded on the HOLDOUT of the selected configuration and capped at Low
    whenever the selection gate did not pass. The comparison is against the
    STATIC-spectral-exposure baseline, so only the dynamics increment counts.
      High   : holdout rank-IC t >= IC_T_HIGH, net top-N spread > 0, and the
               spectral model beats the baseline on both IC and net spread.
      Medium : IC > 0, net spread > 0, and beats the baseline on both.
      Low    : anything else, fewer than MIN_HOLDOUT_PERIODS periods, or gate failed.
    """
    if not gate_passed:
        return "Low"
    if hold.get("n", 0) < config.MIN_HOLDOUT_PERIODS:
        return "Low"
    beats = hold["d_ic_mean"] > 0 and hold["d_spread_mean"] > 0
    if hold["ic_t"] >= config.IC_T_HIGH and hold["net_spread_mean"] > 0 and beats:
        return "High"
    if hold["ic_mean"] > 0 and hold["net_spread_mean"] > 0 and beats:
        return "Medium"
    return "Low"


def select_config(grid: List[Dict], corr_window: Optional[int] = None):
    """
    SELECTION segment only: eligible if rank-IC and net top-N spread are both
    positive there AND the selection-period count clears both the absolute
    (MIN_SELECTION_N) and relative (MIN_SELECTION_N_FRACTION of the pool's
    largest) floors; among eligible, rank-sum of the two t-stats. If none are
    eligible, fall back to the size-filtered pool with gate_passed=False.
    """
    pool = [g for g in grid if (corr_window is None or g["corr_window"] == corr_window)
            and g["selection"].get("n", 0) >= config.MIN_SELECTION_N]
    if not pool:
        return None, False
    max_n = max(g["selection"]["n"] for g in pool)
    pool = [g for g in pool if g["selection"]["n"] >= config.MIN_SELECTION_N_FRACTION * max_n]
    if not pool:
        return None, False
    elig = [g for g in pool if g["selection"]["ic_mean"] > 0 and g["selection"]["net_spread_mean"] > 0]
    gate = bool(elig)
    cand = elig if gate else pool
    ic_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["ic_t"]))}
    sp_order = {id(g): r for r, g in enumerate(sorted(cand, key=lambda g: g["selection"]["net_spread_t"]))}
    best = max(cand, key=lambda g: (ic_order[id(g)] + sp_order[id(g)], g["selection"]["ic_t"]))
    return best, gate


def make_picks(scores: np.ndarray, tickers: List[str], dispersion: float, horizon: int,
               confidence: str, top_n: int) -> List[Dict]:
    order = np.argsort(-scores)[:top_n]
    return [{"ticker": tickers[j], "expected_return": round(float(scores[j] * dispersion * 100), 3),
             "score_z": round(float(scores[j]), 4), "horizon_days": int(horizon), "confidence": confidence}
            for j in order]


def pattern_diagnostics(spec: Dict, tickers: List[str], dates: List[str], agg_returns: np.ndarray,
                        agg_vol21: np.ndarray) -> Dict:
    """Spectrum snapshot/series, structure state today, ETF exposures, and the transition lead tests."""
    n, T = spec["lam"].shape
    last = n - 1
    st, fl, ex, lam = spec["struct"], spec["flow"], spec["expo"], spec["lam"]
    lp = config.LEAD_PARAMS
    out = {}

    L = config.FLOW_LOOKBACK
    if np.isfinite(lam[last]).all():
        out["spectrum_today"] = [round(float(v), 4) for v in lam[last]]
        if last - L >= 0 and np.isfinite(lam[last - L]).all():
            out["spectrum_lag"] = [round(float(v), 4) for v in lam[last - L]]
        out["flow_lookback"] = L
        out["top_modes"] = spec["K"]
        out["flow_today"] = {k: _f(fl[k][last]) for k in ["z_mkt", "z_rot", "z_bulk"]}

        e_std = sm.standardize(ex["e_mkt"][last:last + 1], None)[0]
        snap = []
        for j in range(T):
            snap.append({"ticker": tickers[j], "e_mkt": _f(ex["e_mkt"][last, j]), "e_rot": _f(ex["e_rot"][last, j]),
                         "e_idio": _f(ex["e_idio"][last, j]), "mkt_load_chg": _f(ex["mkt_load_chg"][last, j]),
                         "idio_chg": _f(ex["idio_chg"][last, j]),
                         "mkt_flow": _f(e_std[j] * (fl["z_mkt"][last] if np.isfinite(fl["z_mkt"][last]) else np.nan)),
                         "asset_class": config.ASSET_CLASS_MAP.get(tickers[j])})
        out["snapshot"] = snap

    idx = np.arange(0, n, config.SERIES_STEP)
    idx = idx[np.isfinite(lam[idx, 0])]
    out["series"] = {
        "dates": [dates[i] for i in idx],
        "lam1": [_f(lam[i, 0]) for i in idx], "lam2": [_f(lam[i, 1]) for i in idx], "lam3": [_f(lam[i, 2]) for i in idx],
        "bulk_edge": [_f(st["bulk_edge"][i]) for i in idx],
        "absorption_ratio": [_f(st["absorption_ratio"][i]) for i in idx],
        "rel_gap": [_f(st["rel_gap"][i]) for i in idx],
        "spacing_ratio": [_f(st["spacing_ratio"][i]) for i in idx],
    }

    z_by_stat, state, leads = {}, {}, {}
    for stat in ["absorption_ratio", "rel_gap", "bulk_edge", "spacing_ratio"]:
        series = st[stat]
        if not np.isfinite(series).any():
            continue
        z = sm.shift_trigger_z(series, lp["short_avg"], lp["long_avg"])
        z_by_stat[stat] = z
        state[stat] = {"value": _f(series[last]), "shift_z": _f(z[last])}
    out["structure_state"] = state

    for seed, (name, (stat, direction)) in enumerate(config.LEAD_TRIGGERS.items()):
        if stat not in z_by_stat:
            continue
        leads[name] = sm.lead_diagnostic(z_by_stat[stat], direction, agg_returns, agg_vol21, lp, seed=seed)
        leads[name]["statistic"] = stat
        leads[name]["direction"] = direction
    out["leads"] = leads
    return out


def analyze_universe(name: str, tickers: List[str], prices_df: pd.DataFrame) -> Optional[Dict]:
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return None
    px_df = prices_df[available]
    px_df = px_df[~px_df.isna().any(axis=1)]
    if len(px_df) < 600:
        logger.warning(f"Not enough data for {name}")
        return None

    values = px_df.values
    returns = np.diff(np.log(values), axis=0)
    dates = [d.strftime("%Y-%m-%d") for d in px_df.index[1:]]
    n = len(dates)

    cfg = dict(ridge_lambda=config.RIDGE_LAMBDA, min_train_days=config.MIN_TRAIN_DAYS, cost_bps=config.TRADING_COST_BPS)
    gids = sm.group_ids_from_map(available, config.ASSET_CLASS_MAP, config.MIN_CLASS_SIZE)
    fwd_by_h = {h: sm.compute_forward_returns(returns, h) for h in config.HORIZONS}
    yz, fwd_eval = {}, {}
    for h in config.HORIZONS:
        for neut in config.NEUTRALIZE_OPTIONS:
            yz[(h, neut)] = sm.standardize(fwd_by_h[h], gids if neut else None)
            fwd_eval[(h, neut)] = sm.demean_by_group(fwd_by_h[h], gids) if neut else fwd_by_h[h]

    agg_returns = np.nanmean(returns, axis=1)
    s1 = sm.trailing_sum(agg_returns.reshape(-1, 1), 21)[:, 0]
    s2 = sm.trailing_sum((agg_returns ** 2).reshape(-1, 1), 21)[:, 0]
    agg_vol21 = np.sqrt(np.maximum((s2 - s1 ** 2 / 21) / 20, 1e-10))

    grid, wf_store, live_ctx, patterns = [], {}, {}, {}
    for cw in config.CORR_WINDOWS:
        logger.info(f"  [{name}] corr_window={cw}: eigendecomposing rolling correlation matrices...")
        spec = sm.build_spectrum(returns, cw, config.TOP_MODES, config.FLOW_LOOKBACK, config.TS_Z_WINDOW,
                                 config.TS_Z_MIN, config.MIN_N_SPACING)
        patterns[cw] = pattern_diagnostics(spec, available, dates, agg_returns, agg_vol21)

        for neut in config.NEUTRALIZE_OPTIONS:
            Fz = sm.build_features(returns, spec, cw, gids if neut else None)
            for h in config.HORIZONS:
                cp = sm.build_cross_products(Fz, yz[(h, neut)])
                for window in config.WINDOWS:
                    wf = sm.walk_forward(Fz, cp, fwd_eval[(h, neut)], h, window, config.TOP_N, cfg)
                    key = (cw, h, window, neut)
                    live_ctx[key] = (Fz, cp)
                    if wf is None:
                        continue
                    m = len(wf["rows"])
                    split = int(m * config.SELECTION_FRACTION)
                    grid.append({
                        "corr_window": cw, "horizon": h, "window": window, "neutralize": neut, "n_periods": m,
                        "selection": _clean(sm.summarize(wf, 0, split, h)),
                        "holdout": _clean(sm.summarize(wf, split, m, h)),
                        "full": _clean(sm.summarize(wf, 0, m, h)),
                    })
                    wf_store[key] = (wf, split)
        logger.info(f"  [{name}] corr_window={cw}: done ({len([g for g in grid if g['corr_window'] == cw])} configs)")

    if not grid:
        return None

    def live_for(g):
        Fz, cp = live_ctx[(g["corr_window"], g["horizon"], g["window"], g["neutralize"])]
        return sm.live_scores(Fz, cp, fwd_by_h[g["horizon"]], g["horizon"], g["window"], cfg)

    selected, gate = select_config(grid)
    pinned = False
    pin = config.PINNED_CONFIG
    if pin:
        match = [g for g in grid if g["corr_window"] == pin["corr_window"] and g["horizon"] == pin["horizon"]
                and g["window"] == pin["window"] and g["neutralize"] == pin.get("neutralize", False)]
        if match:
            selected, pinned, gate = match[0], True, True
    if selected is None:
        return None

    ls = live_for(selected)
    conf = confidence_label(selected["holdout"], gate)
    picks = base_picks = []
    if ls is not None:
        picks = make_picks(ls["score_m"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)
        base_picks = make_picks(ls["score_b"], available, ls["dispersion"], selected["horizon"], conf, config.TOP_N)

    cw_picks = {}
    for cw in patterns:
        g, sgate = select_config(grid, cw)
        if g is None:
            continue
        lst = live_for(g)
        sconf = confidence_label(g["holdout"], sgate)
        cw_picks[str(cw)] = {
            "config": {"corr_window": cw, "horizon": g["horizon"], "window": g["window"], "neutralize": g["neutralize"]},
            "gate_passed": sgate, "holdout": g["holdout"], "confidence": sconf,
            "picks": make_picks(lst["score_m"], available, lst["dispersion"], g["horizon"], sconf, config.TOP_N)
            if lst is not None else [],
        }

    wf, split = wf_store[(selected["corr_window"], selected["horizon"], selected["window"], selected["neutralize"])]
    curves = {"dates": [dates[i] for i in wf["rows"]],
              "spectral_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_m"])],
              "baseline_cum": [round(float(v) * 100, 4) for v in np.cumsum(wf["sn_b"])],
              "split_index": int(split)}

    top = {"corr_window": selected["corr_window"], "horizon": selected["horizon"], "window": selected["window"],
           "neutralize": selected["neutralize"], "gate_passed": gate, "pinned": pinned, "confidence": conf,
           "selection": selected["selection"], "holdout": selected["holdout"], "full": selected["full"],
           "n_periods": selected["n_periods"]}
    feature_weights = None
    if ls is not None:
        feature_weights = {nm: round(float(w), 5) for nm, w in zip(sm.FEATURE_NAMES, ls["beta_m"])}

    return {"tickers": available, "n_days": n, "last_date": dates[-1], "selected": top, "picks": picks,
            "baseline_picks": base_picks, "grid": grid, "cw_picks": cw_picks, "curves": curves,
            "patterns": {str(k): v for k, v in patterns.items()}, "feature_weights": feature_weights,
            "class_groups": {t: int(g) for t, g in zip(available, gids)}}


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    results = {
        "run_date": run_date, "algorithm": "SPECTRAL-FLOW",
        "top_picks": {}, "selected": {}, "grid": {}, "cw_picks": {}, "baseline_picks": {}, "curves": {},
        "patterns": {}, "feature_weights": {}, "universe_meta": {},
        "settings": {"trading_cost_bps": config.TRADING_COST_BPS, "top_n": config.TOP_N,
                     "selection_fraction": config.SELECTION_FRACTION, "flow_lookback": config.FLOW_LOOKBACK,
                     "top_modes": config.TOP_MODES, "lead_params": config.LEAD_PARAMS,
                     "n_lead_tests": sum(len(p.get("leads", {})) for p in []),
                     "feature_names": sm.FEATURE_NAMES},
    }
    n_lead_tests = 0
    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        for key, dst in [("picks", "top_picks"), ("selected", "selected"), ("grid", "grid"), ("cw_picks", "cw_picks"),
                         ("baseline_picks", "baseline_picks"), ("curves", "curves"), ("patterns", "patterns"),
                         ("feature_weights", "feature_weights")]:
            results[dst][name] = out[key]
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"],
                                          "last_date": out["last_date"], "class_groups": out["class_groups"]}
        n_lead_tests += sum(len(p.get("leads", {})) for p in out["patterns"].values())

        s = out["selected"]
        h = s["holdout"]
        logger.info(f"  ✅ Selected: corr_window={s['corr_window']} / {s['horizon']}d / window {s['window']} / "
                    f"neutralize={s['neutralize']} (selection gate {'passed' if s['gate_passed'] else 'NOT passed'}) | "
                    f"holdout n={h.get('n')} IC={h.get('ic_mean', 0):+.4f} (t={h.get('ic_t', 0):+.2f}) vs static baseline "
                    f"IC={h.get('ic_base_mean', 0):+.4f}, price-only IC={h.get('ic_price_mean', 0):+.4f} | net top-"
                    f"{config.TOP_N} excess {h.get('net_spread_mean', 0) * 1e4:+.1f}bps/period | confidence {s['confidence']}")
        for p in out["picks"]:
            logger.info(f"     {p['ticker']}: {p['expected_return']:+.3f}% over {p['horizon_days']}d ({p['confidence']})")
    results["settings"]["n_lead_tests"] = n_lead_tests

    output_path = f"spectral_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")
    try:
        from push_results import upload_results
        upload_results(output_path, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
