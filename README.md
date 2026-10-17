# P2-ETF-SPECTRAL-FLOW

Eigenvalue dynamics of the rolling correlation matrix: **where is the
covariance spectrum moving?** — not what the returns did. Distinct from the
existing RMT-filtering and spectral-GSP repos, which treat the spectrum as a
static object (keep modes above the Marchenko-Pastur edge, build a graph from
what is left).

## The idea

Each day, form the correlation matrix `C(t)` of the trailing window of
returns and eigendecompose it: `λ1(t) ≥ … ≥ λN(t)` (they sum to N). Then track
the **flow** over a short lookback:

* **market-mode flow** — `log λ1(t)/λ1(t-L)`: correlation building or
  dissolving across the whole universe
* **rotation flow** — the same for the mean of `λ2..λK` (typically asset-class
  / sector modes)
* **bulk flow** — the same for `λ(K+1)`, the top edge of the remaining bulk

and four market-level **structure statistics**:

| statistic | reads as |
|---|---|
| absorption ratio (share of variance in the top ~N/5 modes; Kritzman, Li, Page & Rigobon 2011, *Principal Components as a Measure of Systemic Risk*) | correlations tightening / loosening |
| relative spectral gap `(λ1-λ2)/λ1` | gap **collapse** = the market mode losing dominance |
| bulk edge `λ(K+1)` | bulk **expansion** / contraction |
| consecutive-level-spacing ratio `<r>` (Oganesyan & Huse 2007; Atas, Bogomolny, Giraud & Roux 2013) | ~0.53 = eigenvalue **repulsion** (GOE-like); ~0.39 = **condensation** / clustering (Poisson-like); needs no spectral "unfolding"; computed only when N ≥ 20 |

## Two uses, kept separate on purpose

### 1. Ranking ETFs — and why dynamics only enter as interactions

A market-level scalar ("the gap collapsed today") is the same for every ETF on
a given day, so cross-sectional standardization erases it. It can only help
rank ETFs when combined with each ETF's **own exposure to the mode that is
moving** (its squared eigenvector loading): `flow_k(t) × exposure_k,i(t)`.

Those interaction columns are deliberately **not re-standardized across ETFs**
— doing so strips the flow's magnitude and keeps only its sign (checked
directly: with flow scales differing 15x between days, the product has feature
scale 3.6 on high-flow days vs 0.5 on low-flow days as built, but exactly 1.0
on both once re-standardized). Flow z-scores are
winsorized at ±3 so a crisis-day outlier cannot create ±20 feature values.

**Three nested models, same dates:**

1. *price-only* — mom5, mom21, vol21, beta
2. *baseline* — price-only + **static** spectral exposure (squared loadings on
   the market mode and rotation modes): ordinary PCA/RMT-style structure, no
   dynamics
3. *spectral* — baseline + the **dynamics** (market-mode loading change,
   idiosyncratic-share change, and the three exposure × flow interactions)

The gate and confidence compare model 3 against model 2, because the thesis is
that *dynamics* add something over static spectral structure. The price-only
comparison is shown for context only.

### 2. Detecting structure transitions

`lead_diagnostic` tests whether each statistic's trigger tends to sit within
20 days before a large **aggregate** move (equal-weight universe, ≥ 2× 21-day
realized vol over 10 days). Every statistic becomes a trigger the same way
(Kritzman et al.'s "shift"): `z = (15-day mean − 252-day mean) / 252-day std`,
with an onset defined as the **crossing** of ±1.5. Triggers, windows and
thresholds are fixed in advance. The chance level and **p-value** come from a
circular-shift permutation of the onset times against the fixed event times
(carried over from the hyperbolic-flows engine, where an analytic
uniform-placement formula was found to be biased on independent data).

**Multiple comparisons, stated plainly:** five triggers × three windows ×
three universes are tested (the dashboard prints the exact count), so a few
small p-values are expected by luck alone.

## Asset-class-neutral mode (carried over, with two fixes)

`NEUTRALIZE_OPTIONS = [False, True]` runs both modes in the same grid. In
neutral mode features **and** the forward-return target are de-meaned within
hand-labeled asset class before ranking. This matters here more than usual:
the top eigenvectors *are* largely asset-class modes, so unneutralized
spectral exposure can look like skill while only separating bonds from
equities. Two defects found on the sibling graphon engine are fixed from the
start:

* **Evaluation:** neutral rankings are scored against **class-neutral**
  returns. Scoring them against raw returns leaves the class-level moves in as
  pure noise and understates within-class skill (simulated: the measured IC kept
  only 78% / 59% / 51% of the true IC when class-level dispersion was 1× / 2× /
  3× the within-class dispersion).
* **Small classes:** a class with fewer than `MIN_CLASS_SIZE` (3) members
  (e.g. VNQ, the lone real-estate ETF in FI_COMMODITIES) is de-meaned against
  the universe-wide mean, not itself. De-meaning a singleton against itself
  made its score exactly zero every day, and it then filled a top-3 slot as a
  0.00% "pick".

## How the result is kept honest

Same checks as every sibling engine, applied from the start: cross-sectional
target; non-overlapping test points one horizon apart (Sharpe annualized by
`sqrt(252/horizon)`); selection/holdout split; an absolute **and** relative
selection-period floor; confidence **capped at Low whenever the selection
gate did not pass**; the product that is tested is the top-3 minus universe
average after cost, plus cross-sectional rank-IC.

## Validation status (please read)

Tested on synthetic data only; this sandbox cannot reach `huggingface.co`.

* **Level-spacing ratio, known answers:** GOE matrices gave `<r>` = 0.535
  (reference ≈ 0.531), independent (Poisson) levels 0.385 (reference
  `2ln2−1` = 0.386), and the correlation matrix of i.i.d. noise (N=40, W=252)
  0.526 — the same universality class as GOE, as RMT predicts.
* **Eigendecomposition:** vectorized rolling correlation matched
  `np.corrcoef` to 7e-16; eigenvalues summed to N to 5e-15; squared loadings
  summed to 1 to 3e-15; eigenvalues descending.
* **Ablation, planted signal (target only):** with the forecast target made to
  depend on the market-mode interaction feature (features untouched), the
  spectral model beat the static baseline with dIC t = +6.6; on the unplanted
  null it showed dIC t = −2.0 (chance). Class-neutral mode recovered the same
  planted signal (dIC t = +6.9 scored against neutral returns).
* **Lead test, positive and negative controls:** a trigger engineered to lead
  large moves by ~12 days gave hit rate 0.37 vs chance 0.13 (p = 0.003, false-
  alarm rate 0.05 vs chance 0.66); an unrelated trigger on the same events gave
  nothing.
* **Lead test, null calibration** (full pipeline on noise): hit rate minus
  chance averaged +0.002 over the 31 testable triggers; none reached p < 0.05
  (mean p = 0.60, slightly conservative because ties count against).
* **The flow statistics respond to a real structural change:** when a common
  factor's loading was raised 5× at day 800, λ1 went 3.2 → 16.8, the
  absorption ratio 0.43 → 0.73, the relative gap 0.29 → 0.92, the market-mode
  flow z peaked at day 803, and the absorption-shift z reached 3.6.
* **Full three-universe run** (2,500 days, up to 43 ETFs, 72 grid cells per
  universe) takes about 45 seconds; no iterative optimization anywhere.

**Limits to keep in mind:**

* Rank-ordered eigenvalue tracking blurs mode identity at avoided crossings
  (when λ2 and λ3 swap, "the 2nd mode" is a different eigenvector before and
  after) — the very eigenvalue-repulsion phenomenon being studied, not a bug,
  but it means the k-th eigenvalue's flow is not always one persistent mode.
  Only the market mode's loading change is tracked mode-by-mode (absolute
  loadings, sign-safe).
* N and the window are small relative to RMT's asymptotics (q = N/W is
  0.17–0.7), so daily level-spacing statistics are noisy; with W=63 some
  spacing triggers produce no onsets at all (shown as n/a).
* FI_COMMODITIES has only 7 ETFs, so the spacing statistics are skipped and
  "top 3" is 43% of the universe — the same structural limit seen on the
  sibling engines.
* The correlation spectrum is a statistic of the same returns being ranked;
  anything it adds has to beat price features and static exposure on the
  holdout, which is exactly what the three-level comparison is for.

Nothing here says it works on real markets. The first real run is the actual
test; read the holdout numbers and both baseline comparisons before the picks.
Realistic rank-IC for a daily cross-sectional signal is small (0.01–0.05), and
with 72 configurations per universe a single strong-looking cell is expected
by chance. This is not financial advice.

## Repo structure

```
config.py          universes, asset-class tags, spectral/flow parameters, grid,
                   LEAD_PARAMS and LEAD_TRIGGERS
data_manager.py     master parquet from HF (same as sibling repos)
spectral_model.py   rolling correlation matrices, batched eigendecomposition,
                   structure statistics, flow features, class-neutral
                   standardization, O(1) window ridge fits, three-model walk-
                   forward, permutation lead diagnostic
trainer.py          per-universe orchestration, selection/holdout, live picks, JSON
push_results.py     HfApi.upload_file to the results dataset
us_calendar.py       trading-day helpers
streamlit_app.py     Tab 1 live picks, spectrum today vs lookback, structure
                   state, ETF exposure map; Tab 2 backtest grid, curves,
                   eigenvalue paths, transition lead tests with p-values
.github/workflows/daily_run.yml   weekday cron after the US close
```

## Running

```bash
pip install -r requirements.txt
python trainer.py                 # writes spectral_results_YYYY-MM-DD.json, pushes to HF
streamlit run streamlit_app.py
```

Set `HF_TOKEN`; create the dataset repo `P2SAMAPA/p2-etf-spectral-flow-results`
or change `config.RESULTS_REPO`. To stop the live configuration re-selecting
daily, set `config.PINNED_CONFIG = {"corr_window": 126, "horizon": 5,
"window": 756, "neutralize": True}`.
