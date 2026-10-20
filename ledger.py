"""
ledger.py  —  A forward record that cannot be fit to the past

Everything in the backtest grid was designed AFTER earlier engines had already
shown the price baseline looking good on these same dates, so no split of that
history is fully out-of-sample with respect to the analyst. The ledger is the
fix: every run it records the TARGET WEIGHTS of a small, fixed, pre-registered
set of books (config.LEDGER_BOOKS — chosen once, never re-selected from
results, never changed after seeing them) and scores each one only after its
holding period has fully elapsed, using the same cost model as the backtest.

Mechanics
---------
* One ledger file (forward_ledger.json) lives in the results dataset and is
  carried from run to run.
* A book is recorded at most once per `horizon` trading days (non-overlapping
  holding periods, like the backtest), keyed "<universe>|<construction>|h<h>".
* An entry is SCORED once price data exists for all `horizon` days after its
  signal date: realized gross = sum_j w_j * (sum of log returns over the
  holding period) minus, for long-only books, the equal-weight universe's
  return over the same period; costs use turnover versus the previous entry of
  the same key.
* Scoring never rewrites a scored entry, and never backfills history.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

import config


def empty_ledger() -> Dict:
    return {"version": 1, "books": {}}


def key_for(universe: str, construction: str, horizon: int) -> str:
    return "%s|%s|h%d" % (universe, construction, horizon)


def load_ledger(local_path: str = None, repo_id: str = None, token: str = None):
    """
    Returns (ledger, status). status:
      "hf"       loaded the copy stored in the results dataset (the normal scheduled-run path)
      "new_hf"   the dataset or file does not exist yet — start a fresh ledger there
      "local"    no HF configured; loaded a local file
      "new"      no HF configured and no local file — fresh ledger
      "hf_error" HF is configured but the download FAILED for another reason. The caller must
                 NOT save or upload a ledger in this case: an empty ledger would overwrite the
                 stored history.
    """
    local_path = local_path or config.LEDGER_FILE
    if repo_id and token:
        try:
            from huggingface_hub import hf_hub_download
            try:
                from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
                missing = (EntryNotFoundError, RepositoryNotFoundError)
            except Exception:
                missing = tuple()
            try:
                path = hf_hub_download(repo_id=repo_id, filename=os.path.basename(local_path), repo_type="dataset",
                                       token=token, force_download=True)
            except missing:
                return empty_ledger(), "new_hf"
            with open(path, "r") as f:
                led = json.load(f)
            if isinstance(led, dict) and "books" in led:
                return led, "hf"
            return empty_ledger(), "hf_error"
        except Exception:
            return empty_ledger(), "hf_error"
    if os.path.exists(local_path):
        try:
            with open(local_path, "r") as f:
                led = json.load(f)
            if isinstance(led, dict) and "books" in led:
                return led, "local"
        except Exception:
            pass
    return empty_ledger(), "new"


def save_ledger(ledger: Dict, path: str = None) -> str:
    path = path or config.LEDGER_FILE
    with open(path, "w") as f:
        json.dump(ledger, f, separators=(",", ":"))
    return path


def maybe_append(ledger: Dict, key: str, construction: str, horizon: int, date: str, dates: List[str],
                 weights: Dict[str, float], universe_tickers: List[str]) -> bool:
    """Record today's target weights if at least `horizon` trading days have passed since this key's last entry."""
    book = ledger["books"].setdefault(key, {"construction": construction, "horizon": int(horizon), "entries": []})
    pos = {d: i for i, d in enumerate(dates)}
    if book["entries"]:
        last = book["entries"][-1]["date"]
        if last == date:
            return False
        if last in pos and date in pos and pos[date] - pos[last] < horizon:
            return False
    book["entries"].append({"date": date, "weights": {t: round(float(w), 6) for t, w in weights.items() if w != 0.0},
                            "universe": list(universe_tickers), "scored": False})
    return True


def score_entries(ledger: Dict, key: str, dates: List[str], returns: np.ndarray, tickers: List[str],
                  cost_bps: float = None, borrow_bps_annual: float = None) -> int:
    """Score every entry whose holding period has fully elapsed. Returns the number newly scored."""
    cost_bps = config.TRADING_COST_BPS if cost_bps is None else cost_bps
    borrow = config.SHORT_BORROW_BPS_ANNUAL if borrow_bps_annual is None else borrow_bps_annual
    book = ledger["books"].get(key)
    if not book:
        return 0
    h = int(book["horizon"])
    long_only = book["construction"] in config.LONG_ONLY
    pos = {d: i for i, d in enumerate(dates)}
    tix = {t: j for j, t in enumerate(tickers)}
    n = len(dates)
    newly = 0
    entries = book["entries"]
    for k, e in enumerate(entries):
        if e.get("scored"):
            continue
        p = pos.get(e["date"])
        if p is None or p + h > n - 1:
            continue
        r = returns[p + 1: p + h + 1].sum(axis=0)
        gross = 0.0
        for t, wt in e["weights"].items():
            if t in tix:
                gross += wt * r[tix[t]]
        uni = [tix[t] for t in e["universe"] if t in tix]
        ew = float(np.mean(r[uni])) if uni else 0.0
        if long_only:
            gross -= ew
        prev_w = entries[k - 1]["weights"] if k > 0 else None
        if prev_w is None:
            turnover = 0.0
        else:
            names = set(e["weights"]) | set(prev_w)
            turnover = float(sum(abs(e["weights"].get(t, 0.0) - prev_w.get(t, 0.0)) for t in names))
        cost = turnover * cost_bps / 2e4
        if not long_only:
            cost += sum(-wt for wt in e["weights"].values() if wt < 0) * borrow / 1e4 * h / 252.0
        e.update({"scored": True, "gross": float(gross), "net": float(gross - cost), "turnover": turnover,
                  "ew": ew, "exit_date": dates[p + h]})
        newly += 1
    return newly


def summarize_ledger(ledger: Dict) -> Dict:
    """Per-key realized record, for the dashboard and the results JSON."""
    out = {}
    for key, book in ledger.get("books", {}).items():
        sc = [e for e in book["entries"] if e.get("scored")]
        net = np.array([e["net"] for e in sc], dtype=float)
        n = len(net)
        t = 0.0
        if n >= 3 and net.std(ddof=1) > 0:
            t = float(net.mean() / (net.std(ddof=1) / np.sqrt(n)))
        out[key] = {
            "construction": book["construction"], "horizon": book["horizon"],
            "n_recorded": len(book["entries"]), "n_scored": int(n),
            "n_pending": len(book["entries"]) - int(n),
            "first_date": book["entries"][0]["date"] if book["entries"] else None,
            "net_mean": float(net.mean()) if n else None, "net_t": t if n >= 3 else None,
            "gross_mean": float(np.mean([e["gross"] for e in sc])) if n else None,
            "cum_net": float(net.sum()) if n else None,
            "hit_rate": float((net > 0).mean()) if n else None,
            "recent": [{"date": e["date"], "exit_date": e["exit_date"], "net": e["net"]} for e in sc[-5:]],
        }
    return out
