"""
trainer.py  —  P2 BASELINE-BOOK

A monetization study. For each universe it asks whether the plain price
baseline (mom5, mom21, vol21, beta -> ridge score) can be turned into a book
that makes money AFTER costs, and which kind of book:

  * a grid of 112 cells per universe: 2 score models (plain / class-neutral
    fit) x 2 training windows x 4 horizons x 7 book constructions
  * the book is CHOSEN on the first 70% of the non-overlapping periods only,
    then judged once on the untouched last 30% — so the holdout is a single
    honest test, not a best-of-112
  * the verdict requires both a positive net t-statistic and a positive
    MARKET-ADJUSTED alpha t-statistic: a long/short book that is quietly long
    high-beta names earns money in a rising market with no ranking skill
  * grid-wide robustness is reported next to the pick (share of cells that make
    money, by construction and horizon), plus each raw feature's own IC so the
    blended score can be traced to momentum, a volatility tilt, or a beta tilt
  * a pre-registered set of books is appended to a forward ledger every run and
    scored only after its holding period elapses — the one piece of evidence
    that is out-of-sample with respect to how this study was designed
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
import book_model as bm
import ledger as lg
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


def select_cell(grid: List[Dict]):
    """
    SELECTION segment only. Eligible: net mean > 0 and net t > 0 there, with the
    selection-period count clearing both the absolute (MIN_SELECTION_N) and
    relative (MIN_SELECTION_N_FRACTION of the pool's largest) floors — so a
    thin-sample long-horizon cell cannot win on variance. Among eligible, the
    highest selection net t. If nothing is eligible, fall back to the best
    size-filtered cell and report gate_passed=False.
    """
    pool = [g for g in grid if g["selection"].get("n", 0) >= config.MIN_SELECTION_N]
    if not pool:
        return None, False
    max_n = max(g["selection"]["n"] for g in pool)
    pool = [g for g in pool if g["selection"]["n"] >= config.MIN_SELECTION_N_FRACTION * max_n]
    if not pool:
        return None, False
    elig = [g for g in pool if g["selection"]["net_mean"] > 0 and g["selection"]["net_t"] > 0]
    gate = bool(elig)
    cand = elig if gate else pool
    return max(cand, key=lambda g: (g["selection"]["net_t"], g["selection"]["n"])), gate


def verdict_for(hold: Dict, gate_passed: bool) -> str:
    """
    Supported     : gate passed, enough holdout periods, net mean > 0, net t and
                    market-adjusted alpha t both >= SUPPORTED_T
    Suggestive    : gate passed, net mean > 0, net t >= SUGGESTIVE_T, alpha t > 0
    Not supported : anything else
    """
    if not gate_passed or hold.get("n", 0) < config.MIN_HOLDOUT_PERIODS:
        return "Not supported"
    nm, nt, at = hold["net_mean"], hold["net_t"], hold.get("alpha_t")
    if at is None:
        return "Not supported"
    if nm > 0 and nt >= config.SUPPORTED_T and at >= config.SUPPORTED_T:
        return "Supported"
    if nm > 0 and nt >= config.SUGGESTIVE_T and at > 0:
        return "Suggestive"
    return "Not supported"


def robustness_table(grid: List[Dict]) -> Dict:
    """Holdout results aggregated over models and windows, by construction and horizon."""
    out = {}
    for cons in config.CONSTRUCTIONS:
        out[cons] = {}
        for h in config.HORIZONS:
            cells = [g for g in grid if g["construction"] == cons and g["horizon"] == h
                     and g["holdout"].get("n", 0) >= 20]
            if not cells:
                continue
            nm = np.array([g["holdout"]["net_mean"] for g in cells])
            out[cons][str(h)] = {
                "n_cells": len(cells), "mean_net_bps": float(nm.mean() * 1e4),
                "mean_gross_bps": float(np.mean([g["holdout"]["gross_mean"] for g in cells]) * 1e4),
                "share_positive": float((nm > 0).mean()),
                "mean_net_t": float(np.mean([g["holdout"]["net_t"] for g in cells])),
                "mean_alpha_t": float(np.mean([g["holdout"]["alpha_t"] for g in cells if g["holdout"].get("alpha_t") is not None])),
                "mean_beta": float(np.mean([g["holdout"]["beta"] for g in cells if g["holdout"].get("beta") is not None])),
                "mean_turnover": float(np.mean([g["holdout"]["turnover_mean"] for g in cells])),
            }
    return out


def book_list(weights: np.ndarray, scores: Optional[np.ndarray], tickers: List[str]) -> List[Dict]:
    rows = []
    for j, t in enumerate(tickers):
        if weights[j] != 0:
            rows.append({"ticker": t, "weight": round(float(weights[j]), 5),
                         "score": round(float(scores[j]), 4) if scores is not None else None,
                         "asset_class": config.ASSET_CLASS_MAP.get(t)})
    return sorted(rows, key=lambda r: -r["weight"])


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
    gids = bm.group_ids_from_map(available, config.ASSET_CLASS_MAP, config.MIN_CLASS_SIZE)
    F_raw = bm.build_price_features(returns, config.BETA_WINDOW)
    fwd_by_h = {h: bm.compute_forward_returns(returns, h) for h in config.HORIZONS}

    grid, scores_store, ctx = [], {}, {}
    for mn in config.MODEL_NEUTRAL_OPTIONS:
        g_arg = gids if mn else None
        Fz = np.stack([bm.standardize(F_raw[:, :, j], g_arg) for j in range(bm.N_FEAT)], axis=2)
        for h in config.HORIZONS:
            cp = bm.build_cross_products(Fz, bm.standardize(fwd_by_h[h], g_arg))
            ctx[(mn, h)] = (Fz, cp)
            for window in config.WINDOWS:
                wf = bm.walk_forward_scores(Fz, cp, fwd_by_h[h], h, window)
                if wf is None:
                    continue
                scores_store[(mn, h, window)] = wf
                m = len(wf["rows"])
                split = int(m * config.SELECTION_FRACTION)
                for cons in config.CONSTRUCTIONS:
                    ev = bm.evaluate_book(cons, wf["S"], wf["R"], gids, h)
                    grid.append({
                        "model_neutral": mn, "window": window, "horizon": h, "construction": cons, "n_periods": m,
                        "selection": _clean(bm.summarize(ev, 0, split, h)),
                        "holdout": _clean(bm.summarize(ev, split, m, h)),
                        "full": _clean(bm.summarize(ev, 0, m, h)),
                    })
        logger.info(f"  [{name}] score model {'class-neutral' if mn else 'plain'}: {len(grid)} cells so far")
    if not grid:
        return None

    selected, gate = select_cell(grid)
    if selected is None:
        return None
    hold = selected["holdout"]
    verdict = verdict_for(hold, gate)

    # curves for the selected book
    mn, h, window, cons = selected["model_neutral"], selected["horizon"], selected["window"], selected["construction"]
    wf = scores_store[(mn, h, window)]
    ev = bm.evaluate_book(cons, wf["S"], wf["R"], gids, h)
    split = int(len(wf["rows"]) * config.SELECTION_FRACTION)
    curves = {"dates": [dates[i] for i in wf["rows"]],
              "net_cum": [round(float(v) * 100, 4) for v in np.cumsum(ev["net"])],
              "gross_cum": [round(float(v) * 100, 4) for v in np.cumsum(ev["gross"])],
              "ew_cum": [round(float(v) * 100, 4) for v in np.cumsum(ev["ew"])],
              "split_index": int(split)}

    # today's target book for the selected configuration
    Fz, cp = ctx[(mn, h)]
    sc = bm.live_score(Fz, cp, h, window)
    selected_book = book_list(bm.book_weights(cons, sc, gids), sc, available) if sc is not None else []

    # pre-registered ledger books (fixed set; never re-selected)
    lb = config.LEDGER_BOOKS
    ledger_weights, ledger_books_today = {}, {}
    for lh in lb["horizons"]:
        Fz2, cp2 = ctx[(lb["model_neutral"], lh)]
        sc2 = bm.live_score(Fz2, cp2, lh, lb["window"])
        if sc2 is None:
            continue
        for lc in lb["constructions"]:
            w = bm.book_weights(lc, sc2, gids)
            ledger_weights[(lc, lh)] = {available[j]: float(w[j]) for j in range(len(available))}
            ledger_books_today["%s|h%d" % (lc, lh)] = book_list(w, sc2, available)

    attribution = {str(hh): bm.factor_attribution(F_raw, fwd_by_h[hh], gids, hh, config.SELECTION_FRACTION)
                   for hh in config.HORIZONS}

    top_hold = sorted([g for g in grid if g["holdout"].get("n", 0) >= config.MIN_HOLDOUT_PERIODS],
                      key=lambda g: -g["holdout"]["net_t"])[:3]
    all_hold = [g for g in grid if g["holdout"].get("n", 0) >= 20]
    summary = {
        "n_cells": len(grid), "n_cells_holdout": len(all_hold),
        "share_cells_net_positive": float(np.mean([g["holdout"]["net_mean"] > 0 for g in all_hold])) if all_hold else None,
        "mean_net_bps": float(np.mean([g["holdout"]["net_mean"] for g in all_hold]) * 1e4) if all_hold else None,
        "mean_gross_bps": float(np.mean([g["holdout"]["gross_mean"] for g in all_hold]) * 1e4) if all_hold else None,
        "best_holdout_cells": [{"model_neutral": g["model_neutral"], "window": g["window"], "horizon": g["horizon"],
                                "construction": g["construction"], "net_t": g["holdout"]["net_t"],
                                "net_mean": g["holdout"]["net_mean"], "n": g["holdout"]["n"]} for g in top_hold],
    }

    return {
        "tickers": available, "n_days": n, "last_date": dates[-1],
        "selected": {"model_neutral": mn, "window": window, "horizon": h, "construction": cons,
                     "gate_passed": gate, "verdict": verdict, "n_periods": selected["n_periods"],
                     "selection": selected["selection"], "holdout": hold, "full": selected["full"]},
        "selected_book": selected_book, "grid": grid, "curves": curves, "robustness": robustness_table(grid),
        "summary": summary, "factor_ic": attribution, "ledger_books_today": ledger_books_today,
        "class_groups": {t: int(g) for t, g in zip(available, gids)},
        # not serialized into the results JSON; used by run_trainer for the ledger
        "_ledger_weights": ledger_weights, "_dates": dates, "_returns": returns,
    }


def run_trainer() -> Dict:
    logger.info("🔄 Loading data...")
    try:
        prices_df, macro_df = load_master_data()
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return {}

    run_date = datetime.now().strftime("%Y-%m-%d")
    ledger, ledger_status = lg.load_ledger(config.LEDGER_FILE, config.RESULTS_REPO, config.HF_TOKEN)
    logger.info(f"Forward ledger: {ledger_status} ({len(ledger['books'])} books)")
    ledger_writable = ledger_status != "hf_error"

    results = {
        "run_date": run_date, "algorithm": "BASELINE-BOOK", "ledger_status": ledger_status,
        "selected": {}, "grid": {}, "curves": {}, "robustness": {}, "summary": {}, "factor_ic": {},
        "selected_book": {}, "ledger_books_today": {}, "universe_meta": {},
        "settings": {"trading_cost_bps": config.TRADING_COST_BPS, "short_borrow_bps_annual": config.SHORT_BORROW_BPS_ANNUAL,
                     "selection_fraction": config.SELECTION_FRACTION, "constructions": config.CONSTRUCTIONS,
                     "construction_labels": config.CONSTRUCTION_LABELS, "long_only": sorted(config.LONG_ONLY),
                     "horizons": config.HORIZONS, "ledger_books": config.LEDGER_BOOKS,
                     "features": bm.PRICE_FEATURES},
    }

    for name, tickers in config.UNIVERSES.items():
        logger.info(f"\n📊 Analyzing {name}...")
        out = analyze_universe(name, tickers, prices_df)
        if out is None:
            continue
        for key in ["selected", "grid", "curves", "robustness", "summary", "factor_ic", "selected_book",
                    "ledger_books_today"]:
            results[key][name] = out[key]
        results["universe_meta"][name] = {"tickers": out["tickers"], "n_days": out["n_days"],
                                          "last_date": out["last_date"], "class_groups": out["class_groups"]}

        if ledger_writable:
            for (lc, lh), wdict in out["_ledger_weights"].items():
                key = lg.key_for(name, lc, lh)
                lg.maybe_append(ledger, key, lc, lh, out["_dates"][-1], out["_dates"], wdict, out["tickers"])
                lg.score_entries(ledger, key, out["_dates"], out["_returns"], out["tickers"])

        s = out["selected"]
        h = s["holdout"]
        logger.info(f"  ✅ Selected: {config.CONSTRUCTION_LABELS[s['construction']]} | {s['horizon']}d | window {s['window']} | "
                    f"{'class-neutral' if s['model_neutral'] else 'plain'} score (selection gate "
                    f"{'passed' if s['gate_passed'] else 'NOT passed'})")
        logger.info(f"     holdout n={h.get('n')} gross {h.get('gross_mean', 0) * 1e4:+.1f} / net {h.get('net_mean', 0) * 1e4:+.1f} "
                    f"bps/period (t={h.get('net_t', 0):+.2f}), market beta {h.get('beta') or 0:+.2f}, alpha t="
                    f"{h.get('alpha_t') or 0:+.2f} | verdict: {s['verdict']}")
        sm_ = out["summary"]
        logger.info(f"     grid: {sm_['share_cells_net_positive'] * 100:.0f}% of {sm_['n_cells_holdout']} holdout cells net-positive, "
                    f"mean net {sm_['mean_net_bps']:+.1f} bps")

    results["forward_record"] = lg.summarize_ledger(ledger) if ledger_writable else {}
    output_path = f"book_results_{run_date}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, separators=(",", ":"), default=str)
    logger.info(f"\n💾 Saved: {output_path}")

    files = [output_path]
    if ledger_writable:
        files.append(lg.save_ledger(ledger, config.LEDGER_FILE))
    else:
        logger.warning("Ledger download failed; NOT saving or uploading a ledger this run so stored history is not overwritten.")
    try:
        from push_results import upload_results
        upload_results(files[0], files[1] if len(files) > 1 else None, hf_token=config.HF_TOKEN)
    except Exception as e:
        logger.warning(f"Could not upload results: {e}")
    return results


if __name__ == "__main__":
    run_trainer()
