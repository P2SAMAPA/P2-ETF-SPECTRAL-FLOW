# P2-ETF-BASELINE-BOOK

A **monetization study**, not a new signal.

Across the last several engines, the only ranking information that held up on
real data sat in the plain price baseline — momentum, volatility, beta —
with rank-IC around 0.03–0.05. Yet the concentrated "top 3 vs the universe"
product lost money after costs every time, and every novel feature set added
nothing on top. This repo asks the narrower question that follows:

> Can that baseline be turned into a book that makes money **after costs**?
> Which kind of book? And is the P&L alpha, or just market exposure?

## What it does

**One score.** A ridge regression on four price features only — `mom5`,
`mom21`, `vol21`, `beta` (rolling beta to the equal-weight universe) — refit at
each rebalance on a trailing window. Nothing from any other engine.

**Seven books from the same ranking**, so the only thing varying is how a
ranking becomes positions:

| book | construction |
|---|---|
| `top3_ew`, `top5_ew`, `topq_ew` | long-only equal-weight top 3 / top 5 / top quintile, measured **active vs the equal-weight universe** |
| `ls_quintile` | long top quintile, short bottom quintile |
| `rank_ls` | weights ∝ centered rank, dollar-neutral |
| `ls_quintile_cn`, `rank_ls_cn` | the same two long/short books built **within each asset class** and scaled by class size, so each class's longs and shorts offset — class-neutral *by construction*, so the raw P&L is a tradable class-neutral P&L |

**A grid per universe:** 2 score models (plain / class-neutral fit) × 2 training
windows × 4 horizons (1, 5, 10, 20 days) × 7 books = 112 cells, all on
non-overlapping holding periods.

**Honest selection.** The book is chosen on the first 70% of periods only
(eligible = positive net mean and t on that segment, with an absolute *and*
relative period-count floor; best selection net t wins), then judged **once** on
the untouched last 30%. The holdout is a single test, not a best-of-112.

**Verdict**, on the selected book's holdout:

* **Supported** — gate passed, ≥30 periods, net mean > 0, and **both** the net
  t-statistic and the market-adjusted **alpha** t-statistic ≥ 1.65
* **Suggestive** — net mean > 0, net t ≥ 1.0, alpha t > 0
* **Not supported** — anything else

**Is it alpha or beta?** Each book's net returns are regressed on the
equal-weight universe's return over the same periods: slope = market beta,
intercept = alpha, with an alpha t-statistic. A long/short book that is quietly
net long high-beta names earns money in a rising market with no ranking skill;
requiring the *alpha* t-statistic is what catches that.

**What is the signal?** Each raw feature's own cross-sectional rank-IC (raw and
class-neutral, full sample and holdout) is reported, so a blended score's IC can
be traced to momentum, a volatility tilt, or a beta tilt — and raw IC far above
class-neutral IC flags a feature that only separates asset classes.

**Robustness next to the pick.** Heatmaps of mean holdout net return and mean
alpha t by book × horizon (averaged over models and windows), plus the share of
all cells that made money. A book that is green at only one horizon is a warning,
not a finding.

## Costs

`TRADING_COST_BPS = 15` is a **round-trip** cost per unit of notional replaced
(the convention of every sibling engine). A book's per-period cost is
`sum|w_t − w_{t−1}| × 15/2` bps — for the top-3 long-only book this reproduces the
sibling engines' "fraction of picks replaced × round-trip cost" **exactly**
(one of three replaced → 5.000 bps; checked). Long/short books also pay
`SHORT_BORROW_BPS_ANNUAL = 50` on short notional, prorated to the holding
period — an **assumption**: liquid index ETFs are cheap to short, thematic ones
are not. Taxes, market impact and borrow availability are not modeled.

## The forward ledger (the part that cannot be fit to the past)

Everything in the backtest grid was designed **after** earlier engines had
already shown this baseline looking good on these same dates, so no split of that
history is fully out-of-sample with respect to the analyst. The ledger is the fix.

Every run records the **target weights** of a small, fixed, pre-registered set of
books (`config.LEDGER_BOOKS`: `top3_ew`, `top5_ew`, `ls_quintile`, `rank_ls`,
`rank_ls_cn` × horizons 5 and 20 × a 756-day window × plain score, for each of the
three universes = 30 books) — **no selection, never changed after seeing results**
— and scores each one only after its holding period has fully elapsed, with the
same cost model. A book is recorded at most once per holding period, scoring never
rewrites a scored entry, and nothing is backfilled.

`forward_ledger.json` lives in the results dataset and is carried from run to run.
If the download fails for any reason other than "file does not exist yet", the run
**does not save or upload a ledger** — an empty one would overwrite the stored
history. It starts empty and accrues one 5-day period per week and one 20-day
period per month, so expect a long wait before it says anything. Thirty books are
tracked, so judge the whole set, not its best member.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.

* **Book weights:** every long/short book sums to 0 with long = +1 and short = −1;
  in the class-neutral books each class's net weight is exactly 0.
* **Costs:** one of three top-3 names replaced → 5.000 bps (= 1/3 × 15); no change →
  0; a long/short book held 20 days with zero turnover → 3.968 bps borrow
  (= 50 × 20/252).
* **Alpha/beta regression:** recovered beta 0.64 for a true 0.60; a pure 0.7-beta
  book with zero alpha gave alpha t = −0.36.
* **Detects a real signal, and costs alone sink random ones:** with scores that
  genuinely predict returns all seven books were profitable net (+48 to +93 bps per
  5-day period, t from 9.7 to 15.0); with random scores all seven lost 13–23 bps per
  period from costs alone.
* **Factor attribution:** with forward returns made to depend on `mom21` only, `mom21`
  showed IC +0.340 (t = 37.6), `mom5` +0.157 (correlated with it), `vol21` and
  `beta` ≈ 0; on the unplanted data all four were within noise.
* **Ledger:** replaying 170 daily runs online and recomputing offline from the same
  schedule gave identical results (≤ 7e-8, weight rounding); entries exactly one
  horizon apart; re-running the same day changes nothing; six load cases (download
  ok / file not there yet / network failure / corrupt file / no HF / local file) all
  behave as specified.
* **Full run** (three universes, 112 cells each) takes about 10 seconds; with the
  ledger, two consecutive daily runs, then a repeat on identical data, left the
  ledger unchanged.
* **Verdict calibration on pure noise** (30 independent datasets, 36-ETF universe):
  29 "Not supported", 1 "Suggestive", 0 "Supported"; the selected book's holdout net
  t averaged −0.71 (costs), with 3% ≥ 1.0 and none ≥ 1.65.
* **Verdict on planted momentum:** a weak but real effect (`mom21` IC +0.022, t = 2.2)
  was **not** monetizable — gross +8.8 bps, net −0.3 bps per 5-day period — while a
  stronger one (`mom21` IC +0.057, t = 5.8) was **Supported**: top-5, 5 days, net
  +19.7 bps (t = 1.91, alpha t = 1.84, beta +0.12), 100% of cells net-positive. The
  gap between those two is the point: a statistically significant IC of ~0.02 does not
  survive 15 bps round trip. (The thresholds are specific to the simulated ~1% daily
  volatility, 36 names and these costs — not a universal rule.)

## Limits to keep in mind

* The study design post-dates the observation that motivated it (see the ledger).
* Long-only "active vs equal-weight" books are tradable only as a relative
  position (own the top N instead of the whole universe, or hedge the universe);
  the hedge's own cost is not modeled.
* FI_COMMODITIES has 7 ETFs: "top 3" is 43% of the universe and a quintile is a
  single name, so its books are blunt by construction.
* 112 cells per universe and three universes: the verdict rests on the single
  selected holdout, but the heatmaps and the "best holdout cells" line are
  best-of-N views and will look good by chance.
* The score is refit with a trailing window and uses only data available at each
  rebalance (the fit at row *i* uses rows ≤ *i − horizon*, whose forward returns
  have fully realized), so there is no look-ahead — but survivorship in the ETF list
  itself is not corrected.

Nothing here says the baseline is monetizable on real markets. Run it, read the
verdict and the robustness tables together, and treat the forward ledger as the
real test. This is not financial advice.

## Repo structure

```
config.py          universes, asset-class tags, features/horizons/windows, the seven
                   books, costs, verdict thresholds, pre-registered ledger set
data_manager.py     master parquet from HF (same as sibling repos)
book_model.py       price features, ridge scores, class-neutral standardization, the
                   seven book builders, turnover/borrow costing, alpha/beta
                   regression, factor attribution
ledger.py           forward ledger: append (once per holding period), score matured
                   entries, summarize, safe load
trainer.py          grid, selection on the first 70%, holdout verdict, robustness
                   tables, live books, ledger update, JSON + ledger upload
push_results.py     HfApi.upload_file (results JSON and ledger together)
us_calendar.py       trading-day helpers
streamlit_app.py     Verdict / Robustness & Backtest / Forward Record tabs
.github/workflows/daily_run.yml   weekday cron after the US close
```

## Running

```bash
pip install -r requirements.txt
python trainer.py          # writes book_results_YYYY-MM-DD.json and forward_ledger.json, pushes both to HF
streamlit run streamlit_app.py
```

Set `HF_TOKEN`; create the dataset repo `P2SAMAPA/p2-etf-baseline-book-results`
or change `config.RESULTS_REPO`. For the ledger to persist between scheduled runs
the workflow must have `HF_TOKEN` set — without it each run starts a fresh local
ledger.
