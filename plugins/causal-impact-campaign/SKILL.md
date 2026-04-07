---
name: causal-impact-campaign
version: "2.3.0"
description: |
  Measure the causal impact of a marketing campaign, promo, or intervention on a business metric
  (revenue, conversions, transactions) using Bayesian structural time series. Use this skill whenever
  the user mentions "causal impact", "campaign uplift", "promo effect", "incrementality", "did the
  campaign work", "revenue lift from campaign", "measure uplift", "what was the true effect",
  "counterfactual analysis", "quasi-experiment", or wants to attribute a metric change to a specific
  intervention using time series data. Also trigger when working with GA4/BigQuery data and the user
  asks about measuring the effect of a price change, delivery promo, ad campaign, or any time-bounded
  business action. Trigger on questions like "did the promotion actually increase revenue?",
  "how much additional revenue did the campaign generate?", "is the revenue change from the campaign
  or just seasonality?", "estimate the ROI of our marketing intervention", "the p-value is 0.12 —
  did it work?", "is my BSTS result overfit?", "what's the false positive rate of my model?",
  "how do I calibrate my causal impact model?", or "validate my permutation test results".
  This skill covers the full pipeline: data exploration, covariate engineering,
  dual-method analysis (tfcausalimpact + CausalPy), validation, interpretation, and client-facing
  deliverables including interactive HTML explorers with Plotly.js. Even if the user only mentions
  one method, use this skill to ensure robustness through cross-method comparison.
  NOT for: A/B test design with randomized control groups, multi-touch attribution modeling,
  time series forecasting (Prophet/ARIMA), media mix modeling, or general analytics dashboards.
input: |
  - Daily time series data with a target metric (revenue, conversions, transactions)
  - An intervention date (campaign start) and optional end date
  - Covariate columns (sessions, organic_sessions, paid_sessions, etc.)
  - Data source: typically GA4/BigQuery table or CSV/DataFrame
output: |
  - Causal impact estimate with credible intervals and p-values from two methods (tfcausalimpact + CausalPy)
  - Validation results (placebo tests, pre-period fit, covariate sensitivity)
  - Client-ready findings document (Markdown)
  - Interactive HTML explorer with Plotly.js charts (optional)
  - All analysis artifacts saved to a timestamped output directory
error_handling: |
  - If pre-period data is less than 3x the intervention period, warn the user and proceed with caveats
  - If SNR < 0.2, set expectations early that statistical significance is unlikely
  - If tfcausalimpact or CausalPy fails to install/run, fall back to the other method and note the limitation
  - If methods disagree on direction, report both results honestly with interpretation guidance
  - If CausalPy hangs on macOS, apply cores=1 fix in sample_kwargs
  - If CausalPy fails with xarray "Dimension(s) 'draw', 'chain' do not exist" on short treatment windows (<7 days), skip CausalPy and use RDiT as lead method. CausalPy ITS requires more post-period data.
idempotency: |
  Re-running the analysis with the same data and parameters produces the same estimates
  (within MCMC sampling variance). Set random_seed=42 for reproducibility.
namespace: causal_impact
composable_with:
  - cloud-run-batch-experiment: Scale permutation tests and sensitivity analyses to GCP Cloud Run Jobs
  - client-proposal-slide: Pass findings to create stakeholder-ready presentation
  - frontend-design: Build custom interactive dashboards from analysis results
  - gcp-pipeline-cost-analysis: Estimate cost of running analysis at scale
  - data-provenance-verifier: Verify external data files (weather, Trends CSVs) are genuine before running analysis. Auto-trigger when inheriting data/ directories with external datasets.
merged_skills:
  - permutation-validation: v1.1.0 merged into this skill (v2.0.0). Permutation methodology, code templates, and effect-size comparison are now in Step 5 Validate.
  - bsts-placebo-calibration: v1.0.0 merged into this skill (v2.0.0). Placebo test design, FPR interpretation, NaN handling, and FPR-gated ranking are now in Step 5 Validate.
---

# Causal Impact Campaign Analysis

This skill guides the analyst through measuring the causal effect of a marketing campaign or business
intervention on a target metric using Bayesian structural time series methods. The skill encodes
hard-won lessons from real client engagements — particularly around short-lived campaigns,
retail seasonality, and honest statistical communication.

The approach runs two independent Bayesian methods (tfcausalimpact and CausalPy) for robustness,
includes a rigorous validation suite, and produces a client-ready findings document.

The analysis is idempotent — safe to re-run with the same data and parameters (set random_seed=42
for reproducibility across MCMC runs). Requires Python >= 3.9 with tfcausalimpact and CausalPy.
Compatible with Python v3.9 through v3.12. Works with pandas v1.5+ and v2.x DataFrames.
The causal_impact namespace keeps all output artifacts scoped to a timestamped directory.

On error: if tfcausalimpact fails to converge, fall back to CausalPy alone and note the limitation
in the findings document. When CausalPy fails on macOS with a multiprocessing RuntimeError, apply
the cores=1 fix documented in the environment gotchas section. If both methods fail, report the
failure clearly and suggest checking data quality and pre-period length.

## When This Applies

- Measuring revenue/conversion uplift from a campaign, promo, or intervention
- The user has time series data (typically daily) with a clear intervention date
- Data comes from GA4, BigQuery, or similar web analytics
- There is NO randomised control group (if there is, use A/B testing instead)
- The intervention is time-bounded (has a start date, and optionally an end date)

## Overview of the Pipeline

```
1. Understand the Intervention → 2. Explore Data → 3. Engineer Covariates
→ 4. Run Dual Analysis → 5. Validate → 6. Interpret → 7. Document
```

---

## Step 1: Understand the Intervention

Before touching data, establish these facts:

| Question | Why It Matters |
|---|---|
| What was the campaign? | Determines which metrics to target and which covariates are safe |
| When did it start and end? | Defines analysis windows |
| What channels did it run on? | Determines if paid_sessions is safe as a control |
| Were other campaigns running? | Concurrent interventions confound the analysis |
| Was it site-wide or geo-targeted? | Geo-targeting enables stronger designs (synthetic control, geo lift) |
| How long was it? | Campaigns < 1 week are very hard to detect statistically |

**Critical early assessment — signal-to-noise ratio:**
```
SNR = expected_daily_effect / daily_revenue_std_dev
```
- SNR > 0.5: Good chance of statistical significance
- SNR 0.2–0.5: Possible but will need strong covariates
- SNR < 0.2: Very unlikely to achieve p < 0.10 — set expectations early

If the campaign is very short (< 7 days) and the target metric is volatile, warn the user upfront
that statistical significance may not be achievable, but consistent direction across methods
still provides useful evidence.

## Step 2: Explore the Data

### Check date ranges and available metrics

```python
from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID)

# What's in the features table?
df = client.query(f"SELECT * FROM `{TABLE}` ORDER BY date LIMIT 5").to_dataframe()
print(f"Columns: {list(df.columns)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
```

### Key explorations before modelling

1. **Revenue by day of week** — is there a strong weekly cycle?
2. **Paid vs organic split** — what share of sessions/revenue is paid?
3. **Holiday/seasonal patterns** — how extreme are BF/Christmas spikes?
4. **Around the intervention** — what changed? Traffic? Conversion? AOV?
5. **Correlation matrix** — which covariates correlate with the target?

### The "traffic vs conversion" diagnostic

The traffic-vs-conversion diagnostic is critical for covariate safety. Compare the intervention period to the week before:

```python
# If sessions barely changed but revenue jumped → promo lifted conversion/AOV
# This means sessions-based covariates are SAFE controls
promo = df[(df['date'] >= INTERVENTION_START) & (df['date'] <= INTERVENTION_END)]
pre_week = df[(df['date'] >= INTERVENTION_START - pd.Timedelta(days=7)) & (df['date'] < INTERVENTION_START)]
print(f"Pre-week: revenue={pre_week['revenue'].mean():.0f}, sessions={pre_week['sessions'].mean():.0f}")
print(f"Promo:    revenue={promo['revenue'].mean():.0f}, sessions={promo['sessions'].mean():.0f}")
```

## Step 3: Engineer Covariates

Covariate engineering is where most of the analytical value is added. Better covariates = tighter counterfactual
= narrower credible intervals = better chance of detecting the effect.

### The Covariate Safety Rule

**A covariate must NOT be affected by the intervention.** If the campaign drove paid traffic,
`paid_sessions` absorbs the effect and your estimate will be biased toward zero.

Safe controls for most campaign types:
- `organic_sessions` — organic demand is rarely affected by a delivery/price promo
- Calendar variables — day-of-week, holidays, payday cycles
- Weather (if available)

> **Always test with and without suspect covariates.** If removing a covariate increases the
> effect estimate substantially (>20%), it's likely absorbing the causal signal. In a retail
> case study, removing `paid_sessions` increased the effect by ~36% and improved p-value by
> roughly half — achieving BSTS significance for the first time.

### Covariate Engineering Recipes

#### 1. Split aggregate metrics
Don't use `sessions` as a single covariate. Split into `paid_sessions` and `organic_sessions`:
- Both are more informative than the aggregate (r=0.885 and r=0.864 vs r=0.903 combined)
- Allows you to check if the intervention affected one channel but not the other

#### 2. Cyclical day-of-week encoding
Binary `is_weekend` misses that Sunday revenue differs from Saturday, and Tuesday is the
weakest day. Use sin/cos encoding to capture the full weekly cycle:

```python
df['sin_dow'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
df['cos_dow'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)
```

> **Important — nseasons determines whether DoW covariates are needed:**
> - `nseasons=7`: captures day-of-week seasonality internally → DoW covariates are **redundant**
> - `nseasons=14` (recommended): captures biweekly patterns (payday cycles) but NOT 7-day weekly
>   patterns → explicit DoW covariate is **needed** (sin_dow or is_weekend)
>
> When using CausalPy or other models without built-in seasonality, always include DoW covariates.
>
> **Data requirements scale with nseasons.** The seasonal state vector has dimension s-1
> [Harvey, 1989], and the Kalman filter's exact diffuse initialization consumes d ≈ s
> observations before the likelihood can even be evaluated [Koopman, 1997; Durbin & Koopman, 2012].
> Practical minimum: at least 2 complete cycles of the seasonal period to distinguish seasonality
> from noise [Hyndman & Kostenko, 2007]. This means nseasons=7 needs ≥14 pre-period days
> (absolute minimum), while nseasons=14 needs ≥28. For reliable estimation under noisy data,
> use 3-5× nseasons (i.e., 42-70 days for nseasons=14).

#### 3. Holiday intensity (not binary flags)
Holiday intensity encoding is one of the most important lessons. Binary flags like `winter_sale_flag` treat all
sale days equally, but Black Friday revenue can be 5x a regular sale day. The model
sees a massive unexplained residual, which inflates variance estimates and widens ALL
credible intervals — including for your promo period.

Use a continuous Gaussian bell curve peaking at Black Friday (~27 days before Christmas):

```python
def christmas_proximity(date_series):
    """Multi-modal retail holiday intensity. Components:
    1. Black Friday sharp spike (~27 days before Christmas)
    2. BF weekend/Cyber Monday
    3. Pre-BF ramp (~35 days before)
    4. Gift shopping peak (~7 days before Christmas)
    5. Mid-December elevated baseline
    6. Boxing Day / winter sale
    Correlation with revenue: r=0.828 (vs r=0.632 for single Gaussian)
    """
    result = pd.Series(0.0, index=date_series.index)
    for year in date_series.dt.year.unique():
        xmas = pd.Timestamp(f"{year}-12-25")
        for idx in date_series.index:
            d = (xmas - date_series.loc[idx]).days
            intensity = 0.0
            if 20 <= d <= 35: intensity += np.exp(-0.5 * ((d - 27) / 3) ** 2)       # BF spike
            if 22 <= d <= 28: intensity += 0.6 * np.exp(-0.5 * ((d - 25) / 2) ** 2)  # BF weekend
            if 28 <= d <= 42: intensity += 0.4 * np.exp(-0.5 * ((d - 33) / 5) ** 2)  # Pre-BF ramp
            if 0 <= d <= 15: intensity += 0.7 * np.exp(-0.5 * ((d - 7) / 4) ** 2)    # Gift shopping
            if 10 <= d <= 22: intensity += 0.3                                         # Mid-Dec baseline
            if -10 <= d < 0: intensity += 0.3 * np.exp(-0.5 * ((d + 2) / 3) ** 2)    # Boxing Day
            result.loc[idx] = max(result.loc[idx], min(intensity, 1.0))
    return result
```

This variable typically achieves r=0.828 with revenue for retail clients (vs r=0.632 for a
single Gaussian and r≈0.02 for a binary `winter_sale_flag`).

#### 4. Interaction terms
Combine factors that amplify each other:
```python
df['payday_x_weekend'] = df['payday_window_flag'] * df['is_weekend']
```

#### 5. Paid share as media intensity signal
```python
df['paid_share'] = df['paid_sessions'] / (df['paid_sessions'] + df['organic_sessions'])
```
Low standalone correlation (~0.14) but can be useful in combination. A negative coefficient
often means organic visitors convert better (higher revenue per organic session).

### Recommended covariate audit

> **Combined audit:** Each covariate gets a single recommendation based on both its predictive
> value (correlation with revenue) and its intervention safety (did it change during the promo?).
> `YES` = include. `CAUTION` = test with and without. `SKIP` = don't include.
>
> **Critical:** The baseline for contamination must be the **last 7 days of the pre-period**,
> not the full pre-period mean. Using the full pre-period mean will flag *every* covariate as
> contaminated because the promo naturally lifts all correlated metrics. The last-week baseline
> controls for seasonality and isolates whether the covariate changed *independently* of the
> general uplift. Calendar-based covariates (holidays, weekends, paydays) are always safe
> regardless of their % change — they're deterministic. Treatment indicators (treatment_flag,
> intervention_flag) must be excluded entirely — they define the treatment, not predict it.

```python
# Combined covariate audit: checks BOTH correlation AND intervention safety
pre_all = df[df['date'] < INTERVENTION_START]
pre_week = df[(df['date'] >= INTERVENTION_START - pd.Timedelta(days=7)) & (df['date'] < INTERVENTION_START)]
promo = df[(df['date'] >= INTERVENTION_START) & (df['date'] <= INTERVENTION_END)]

CALENDAR_COVS = ['holiday_flag', 'payday_window_flag', 'kcp_period_flag',
                 'winter_sale_flag', 'is_weekend', 'payday_x_weekend', 'xmas_intensity']
EXOGENOUS_COVS = ['temp_avg', 'precipitation_mm']

for col in COVARIATES:
    corr = pre_all[col].corr(pre_all[TARGET])
    pre_val = pre_week[col].mean()
    promo_val = promo[col].mean()
    change = (promo_val / pre_val - 1) if pre_val != 0 else float('nan')

    if col in CALENDAR_COVS:
        safety = "SAFE (calendar)"
    elif col in EXOGENOUS_COVS:
        safety = "SAFE (exogenous)"
    elif abs(change) < 0.10:
        safety = "SAFE (<10% change)"
    else:
        safety = f"INVESTIGATE ({change:+.0%} change)"

    include = "YES" if corr > 0.05 and "SAFE" in safety else "CAUTION" if "INVESTIGATE" in safety else "SKIP"

    print(f"{col:30s}  r={corr:+.3f}  promo_change={change:+.1%}  {safety:25s}  → {include}")
```

Drop covariates that are constant in the pre-period, have >50% missing, or that receive
a `SKIP` recommendation. Test `CAUTION` covariates by running the model with and without them.

## Step 4: Run Multi-Method Analysis

No single method is perfect for causal inference from observational time series. Each makes
different assumptions and has different blind spots. Running multiple methods and checking
whether they agree is far more convincing than any single p-value.

### Why each method and when to use it

| Method | What it does | When to use | Key limitation |
|---|---|---|---|
| **BSTS (tfcausalimpact)** | Decomposes time series into trend + seasonality + regression, projects counterfactual | Always — the primary analysis. Full decomposition with covariates | Struggles with short campaigns (<7 days) — daily variance overwhelms signal |
| **CausalPy (LinearRegression)** | Bayesian regression with exact MCMC inference (NUTS) | Always — robustness check with different inference engine. **Caution:** Earlier versions parsed a regression coefficient from CausalPy's summary table, which picked the wrong parameter and produced a large negative outlier. **Fixed:** now uses posterior predictions (observed − counterfactual). Always cross-check with BSTS and RDiT. | No time series structure (no trend/seasonal components) |
| **RDiT** | Local linear regression at the intervention boundary, bootstrap CIs | **Especially for short campaigns.** Only method that achieved significance for 4-day promo | Ignores data far from cutoff; sensitive to bandwidth; no decomposition |
| **Conformal CIs** | Distribution-free prediction intervals from pre-period residual quantiles | Always — sanity check on Bayesian CIs. If 2x wider → model overconfident; if 2x narrower → model over-conservative | Can't compute probability of effect; sensitive to pre-period outliers |

**Decision tree for short vs long campaigns:**
```
Campaign duration >= 14 days?
  YES → BSTS is the lead method (enough post-data for significance)
         CausalPy + conformal as robustness checks
  NO  → RDiT is the lead method for significance claims
         BSTS provides the full counterfactual decomposition and narrative
         Conformal CIs provide distribution-free uncertainty bounds
         CausalPy confirms direction across inference engines
```

**Always document methods that didn't work** — transparency about what was tried and why it
failed is as valuable as positive results. Include in the findings doc's "What Worked and
What Didn't" section.

### Cloud Run Speed Estimates

Actual Cloud Run timings (including cold start overhead):
- **Fast methods** (BSTS VI, RDiT, Conformal): ~1-2 min per method
- **Moderate methods** (CausalPy Bayesian LR): ~3-5 min
- **Slow methods** (BSTS HMC): ~10-15 min

Total for all 6 methods in parallel: ~14 min (limited by HMC).

### BigQuery Dataset Location

If your BQ dataset is in a non-US region (e.g., **europe-west2**), BQ clients
must set `location="europe-west2"` explicitly — without it, queries fail with
"Dataset not found in location US". Discovery command:
```bash
bq show --format=prettyjson your-project-id:your_dataset | grep location
```

### VI Stochasticity Warning

Variational inference (`fit_method="vi"`) is non-deterministic even with a fixed `random_seed`.
Single-run p-value differences of ±0.04 are noise, not signal. Validated across multiple runs:
the same spec can range p=0.03–0.08 between runs.

**Practical implications:**
- **Screening** (p<0.05 vs p>0.10): single VI runs are fine
- **Definitive A/B comparisons** between covariate encodings: use HMC (`fit_method="hmc"`,
  deterministic given seed) or average 5+ VI runs
- Don't make spec recommendations based on single-run differences < 0.03

### Consensus and Outlier Detection

When aggregating results across methods, use median (not mean) for consensus — it handles
outliers gracefully. Flag any method whose effect estimate disagrees in sign with the majority
or exceeds 3x the median magnitude. In one retail engagement, CausalPy returned a large negative
outlier while all other methods showed moderate positive effects — the consensus median correctly reflected the positive
signal, but the outlier should be explicitly flagged in the report.

### Important: Dependency Conflict

tfcausalimpact and CausalPy have **incompatible numpy requirements**:
- tfcausalimpact: `numpy<2`, `pandas<=2.2`
- CausalPy: `numpy>=2`, `pandas>=3.0`

**You must run them in separate Python scripts.** Run tfcausalimpact first (with numpy<2),
then upgrade numpy and run CausalPy.

### Method 1: tfcausalimpact

```python
from causalimpact import CausalImpact

MODEL_ARGS = {
    "nseasons": 14,           # biweekly seasonality — RECOMMENDED default (captures payday cycles)
    "standardize_data": True,  # z-score normalisation
    "fit_method": "vi",        # variational inference (fast; HMC is more conservative but slower)
}
# NOTE: nseasons=14 (biweekly) is the recommended default. It achieved roughly half the
# p-value of nseasons=7 in a retail engagement. This captures fortnightly
# payday cycles (25th-3rd spending windows). The webapp now defaults to tfci_vi_biweekly.
# Always test nseasons=7 alongside 14 and report both. nseasons=14 is a model configuration
# change, not data exclusion — more defensible than masking if challenged on specification search.

# Data must be a DataFrame with DatetimeIndex, target in first column
ci = CausalImpact(data[required_cols], pre_period, post_period, model_args=MODEL_ARGS)

# Key outputs
print(ci.summary())
print(ci.summary(output="report"))
print(f"p-value: {ci.p_value:.4f}")
ci.plot()
```

### Method 2: CausalPy

```python
import causalpy as cp

result = cp.InterruptedTimeSeries(
    df,
    treatment_time=INTERVENTION_START,
    formula="revenue ~ 1 + t + organic_sessions + paid_sessions + sin_dow + cos_dow + xmas_intensity + payday_window_flag + kcp_period_flag + holiday_flag",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={
            "random_seed": 42,
            "chains": 4,
            "draws": 2000,
            "tune": 1000,
            "cores": 1,  # REQUIRED on macOS — multiprocessing fork issue
        }
    ),
)
result.summary()
result.plot()
```

CausalPy provides:
- Pre-intervention Bayesian R² (aim for >0.70)
- Model coefficients with HDI (helps understand which covariates drive predictions)
- Native `treatment_end_time` parameter for short-lived interventions

### CausalPy model selection guide

CausalPy offers several model classes. Here's what we learned from testing them:

| Model | When to use | Verdict |
|---|---|---|
| **`LinearRegression`** | Default for ITS with formula covariates | **Recommended.** y_hat_sigma=1,674. Solid, fast, interpretable |
| **`LinearRegression` + `I(t**2)`** | When baseline has non-linear drift | Marginal improvement (sigma 1,674→1,666). Add if you suspect curvature |
| `WeightedSumFitter` | Synthetic control (weighting donor units) | **Not for ITS.** Much higher noise (sigma=2,910). Only use with panel data |
| `BayesianBasisExpansionTimeSeries` | Prophet-like (Fourier + changepoints) | Requires `pymc-marketing` dependency. Params: `n_order=3`, `n_changepoints_trend=10` |
| `StateSpaceTimeSeries` | Closest to tfcausalimpact's BSTS | Full state-space model with level/trend/seasonal. Params: `level_order=2`, `seasonal_length=7` for daily data. Slowest but most principled |

**Practical recommendation:** Start with `LinearRegression` (fast, good enough for most cases).
If the pre-period R² is below 0.65, try `StateSpaceTimeSeries` for better time dynamics.
`WeightedSumFitter` is only appropriate if you have proper donor units (comparable untreated
sites/regions), not for single-unit ITS.

**ScikitLearnAdaptor note:** The API changed in CausalPy 0.8 — `ScikitLearnAdaptor()` takes
no arguments. Check the current docs if you need sklearn models.

### Sensitivity analysis

Run both methods with multiple covariate bundles to check robustness:

```python
SENSITIVITY_SPECS = {
    "full_model": ALL_COVARIATES,
    "organic_only": ["organic_sessions"],
    "organic_plus_calendar": ["organic_sessions", "sin_dow", "cos_dow", "payday_window_flag"],
    "organic_plus_xmas": ["organic_sessions", "sin_dow", "cos_dow", "xmas_intensity"],
    "paid_organic_split": ["organic_sessions", "paid_sessions", "sin_dow", "cos_dow"],
}
```

If all specs agree on direction, that's strong evidence even if individual p-values are above
conventional thresholds.

### Specification Curve Analysis (SCA) — full robustness sweep

For publication-quality robustness evidence, run an SCA that exhaustively tests all analytical
"forking paths." A curated bundle approach is better than full factorial:

**Infrastructure dimension (model structure):**
- Pre-period modes: full (no mask), full + mask BF/Xmas, full + mask Nov-Jan, post-holiday trimmed
- Note: masking and trimming are **mutually exclusive** (both solve holiday variance differently)
- Note: masking and seasonal decomposition enrichments (e.g., xmas_intensity_decomposed → bf_spike, bf_weekend, gift_shopping, mid_dec_baseline, boxing_day) are **also mutually exclusive** — the mask removes the exact dates where these Gaussian signals are non-zero, making them constant (all zeros) in the pre-period. Pick one: mask to remove holiday variance, OR decompose to model it.
- Seasonality: nseasons=7 (weekly) vs 14 (biweekly)

**Enrichment dimension (feature engineering choices):**
Organize into groups, test head-to-head within each group:
- **DoW encoding:** none vs sin/cos vs binary weekend vs day dummies
- **Calendar:** none vs Christmas intensity vs bank holidays vs full (xmas + bank holidays + payday)
- **Weather:** raw temp/precip vs interaction terms (cold_rain = max(0,-temp)×precip, precip×weekend, precip×sale_intensity)
- **External signals:** Google Trends category vs brand share vs none
- **Sale signals:** binary flag vs continuous intensity (z-score) vs both (all zeroed during treatment)
- **Transforms:** log(target) vs raw, interaction terms

**Key insights from retail SCA:**
- `sale_intensity` (continuous MAD z-score) is strictly better than binary `sale_type_flag` — it naturally weights major sales (z~5) higher than small promos (z~2.6)
- Weather interactions encode **consumer behaviour**: `precip × sale_intensity` = "friction × intent" — rain on a major sale kills conversions more than rain on a quiet day
- Universal facts (bank holidays, paydays) should be auto-detected from dates, not gated behind manual configuration
- Forward stepwise bundles trace "what does each addition buy?" — often the first 2-3 enrichments capture 80% of the improvement
- Leave-one-group-out bundles identify "which group matters most?" — usually external signals (Trends) > weather > calendar > DoW
- **Decompose composite covariates:** If a calendar signal has known sub-components (e.g., 6 Gaussian peaks for holiday intensity), offer them as separate columns so the model's spike-and-slab prior can weight each independently
- **Head-to-head bundles must test genuinely different signals.** If two options produce >80% overlapping columns (e.g., "holiday_flag" and "bank_holidays" are near-duplicates), replace one with a distinct signal (e.g., payday proximity)
- **Date masking invalidates pre-resolved dates.** When masking removes early dates (e.g., Jan 1-5), any pre-period start date resolved before masking must be clamped to the actual data range after `prep_df` reindexes
- **Short post-holiday pre-periods dominate low-p specs.** This reflects regime homogeneity (Jan-Feb is structurally stable), not cherry-picking. The mechanism: removing high-variance holiday periods reduces observation error variance, narrowing CIs. Validate with permutation tests.

**SCA validation — permutation tests on top specs:**
- Run 30+ permutation shuffles on the top 5 specs per pre-period mode = ~600 parallel tasks
- **CRITICAL: Compare EFFECT SIZES, not p-values.** Count shuffles where `|abs_eff| >= real |abs_eff|`.
  Comparing p-values is confounded by model p-value inflation (shuffled runs also get low p-values).
  Effect-size comparison is immune to this miscalibration.
- This follows the established methodology in the causal inference literature:
  - **Abadie et al. (2010, 2021)**: Synthetic control permutation uses the post/pre MSPE *ratio*
    (an effect-size statistic), not model p-values [JASA 2010; JEL 2021]
  - **Linden (2018)**: ITS permutation compares the *magnitude* of trend changes across
    pseudo-treatments [J Eval Clin Pract, PMID 29460383]
  - **Young (2019)**: Model p-values inflate under misspecification; randomization inference
    using effect estimates is robust [QJE 134(2)]
  - **Fisher (1935)**: Any test statistic is valid for permutation, but effect-size statistics
    are power-optimal for magnitude alternatives
- The mechanism: a model p-value = effect / estimated_uncertainty. When uncertainty is
  systematically underestimated (35-55% FPR), the p-value is distorted at every permuted
  date equally, so comparing p-values inherits the model's FPR. Effect-size comparison
  is unaffected because the distortion cancels in the relative ranking.
- Any spec with permutation-p > 0.10 should be flagged as potentially spurious
- Include both log_target and raw_target variants per mode (log stabilizes variance, fairer effect-size comparison)
- Prior experience (CORRECTED 2026-04-07): masking modes (mask_nov_jan / mask_bf_jan) tend to
  produce LOW model p-values but HIGH permutation p-values — contrary to what you might expect.
  Originally attributed to "removing high-variance holiday periods creates a cleaner counterfactual
  baseline," but the Schuh investigation (Issue #51) traced the root cause to a latent zero-injection
  bug in `prep_df`. After `apply_date_mask` drops rows, `prep_df` reindexes to a continuous daily
  index and fills the reinserted rows with `y=0` + interpolated covariates. The model then trains
  on fake zero-revenue winters, which teaches a spurious covariate→target relationship and inflates
  both real and shuffle effects. **Mask modes should not be used until this bug is fixed in your
  specific implementation** — see the "Data-Prep Zero-Injection Trap" section below. Prefer
  contiguous pre-period windows (e.g., post-Xmas trimmed Jan 6 → treatment) which avoid the bug
  entirely.

**Permutation code template:**

```python
def generate_random_dates(df, real_date, post_days, n=50, min_pre=180, seed=42):
    """Generate random intervention dates, excluding zone around real treatment."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(df['date'])
    earliest = dates.min() + pd.Timedelta(days=min_pre)
    latest = dates.max() - pd.Timedelta(days=post_days)
    exclusion = (dates >= real_date - pd.Timedelta(days=2*post_days)) & \
                (dates <= real_date + pd.Timedelta(days=2*post_days))
    candidates = dates[(dates >= earliest) & (dates <= latest) & ~exclusion]
    return sorted(rng.choice(candidates, size=min(n, len(candidates)), replace=False))

def compute_permutation_pvalue(real_effect, null_effects):
    """Effect-size comparison: count shuffles where |effect| >= |real_effect|."""
    valid = [e for e in null_effects if e is not None]
    n_extreme = sum(1 for e in valid if e >= abs(real_effect))
    return (n_extreme + 1) / (len(valid) + 1)  # +1 correction avoids p=0
```

**Permutation pitfalls:**
- Wide masks (mask_nov_jan) inflate model confidence → low BSTS p but high perm p
- Too many covariates (7+) overfit → use 3-5 orthogonal covariates
- Weather covariates improve permutation discrimination (help explain variance at random dates)
- Pre-BF revenue spikes should NOT be masked — they teach the model that spikes can occur naturally
- NaN from BSTS: apply `raw.replace('NaN', 'null')` before JSON parsing. `gsutil -m cat`
  silently drops NaN-containing objects in streaming decode

**SCA output should include:**
1. Spec curve chart: bars sorted by p-value (acceptable if permutation-validated), CI whiskers, colored by significance
2. Indicator matrix: binary grid showing which choices produced each bar
3. Dimension impact analysis: per-group average p improvement when included vs excluded
4. Pre-period mode breakdown: how many specs per mode reach significance (expect short pre-period to dominate — document why)
5. Summary: median effect, IQR, % significant, % positive direction
6. Permutation p-value column in detail table for top 50 specs

## Step 5: Validate

### Rolling backtests

Slide a fake intervention window across the pre-period. At each position, fit the model on
everything before and predict the window. Compare predictions to actuals:

```python
HORIZON = promo_days  # same length as real intervention
STEP = 7              # slide by 1 week each time
MAX_WINDOWS = 12      # cap to keep runtime manageable
MIN_TRAIN = 56        # minimum pre-period for each backtest
```

Measure:
- **WAPE** (Weighted Absolute Percentage Error): sum(|actual - predicted|) / sum(|actual|)
- **95% Coverage**: fraction of actuals within the 95% credible interval
- **Placebo effect**: the estimated "effect" where none exists

### Placebo test (rolling backtest)

Compare the real intervention effect to the distribution of placebo effects from rolling backtests:

```python
placebo_rank = (placebo_effects < real_effect).mean()  # if effect is positive
```

If the real effect ranks above the 90th percentile of placebos, it's genuinely unusual.
If it sits in the middle (50-70th percentile), the model can't distinguish it from noise.

### Pre-period placebo test (MODEL CALIBRATION — CRITICAL)

**This is the single most diagnostic validation for BSTS.** This is an "in-time placebo test"
— the time series analogue of the in-space placebo tests established for synthetic control
methods [Abadie et al., 2010; Abadie et al., 2015]. Place fake treatment windows at regular
intervals through the pre-period where NO intervention occurred:

```python
# Place 4-day fake windows every 14 days through the pre-period
# Run the same BSTS spec at each window
# Count: what fraction detect p < 0.05?
FPR = n_significant_placebos / n_total_windows
# Well-calibrated model: FPR ≈ 5% [Eggers et al., 2024]
# Miscalibrated: FPR >> 5% (model detects "effects" where none exist)
```

The 5% FPR threshold is the universal calibration standard: if tests use p < 0.05 as the
significance criterion, a well-calibrated null distribution should produce significant results
in exactly 5% of null runs [Eggers et al., 2024]. Permutation-based validation is endorsed
as a standard robustness requirement for observational causal inference [Athey & Imbens, 2017;
Linden, 2018].

**This test catches a critical failure mode:** In one engagement, the top-ranked specifications
showed model p=0.000 but the placebo test revealed **22% FPR** — the model was detecting
"significant" effects at 22% of random dates where nothing happened. The p-values were
inflated approximately 4× due to short pre-period (51 days with 6+ covariates).

**Multi-method placebo FPR — empirical findings (updated session 33):**

The miscalibration is **model-class-independent**, not specific to BSTS VI:
- BSTS VI (variational inference): 41% FPR
- BSTS HMC (NUTS sampler): 47% FPR
- Prophet (Meta): 55% FPR
- RDiT (local linear regression): 51% FPR

Pre-period mode also doesn't fix it:
- ~50 days pre-period (post-Xmas trimmed): FPR ~22%
- ~700 days full pre-period (no mask): FPR ~39%
- ~700 days with mask_bf_jan: FPR ~93% (⚠️ CORRECTED: originally attributed to "masking creates
  gaps" — the Schuh Issue #51 investigation showed this is partially a data-prep zero-injection
  bug, not a fundamental property of mask modes. After the fix, this number is expected to drop
  substantially. See the "Data-Prep Zero-Injection Trap" section for details.)
- Guideline: **ALL tested methods produce 35-55% FPR on daily retail revenue.** (⚠️ CAVEAT: these
  numbers were measured in pipelines that may contain the zero-injection bug. The "all methods
  miscalibrated" finding is robust for non-mask modes, but the mask-mode FPR numbers should be
  re-measured after verifying your mask is a true removal.)
  Pre-period length, inference method, and model class don't fix the root cause
  (strong autocorrelation + complex seasonal patterns in daily data).
- When ALL methods are miscalibrated, shift to alternative evidence: direction consistency,
  **permutation tests using effect-size comparison**, and prob_positive.
- See also: `bsts-placebo-calibration` skill for the full methodology.

These empirical observations align with published simulation evidence. Gils et al. [2022]
found that BSTS FDR inflates to ~10% with only 6 pre-period observations and recovers to
~5% (nominal) with ≥12 observations. The mechanism: short pre-periods with many covariates
produce an effective "events per variable" (EPV) ratio below the ~10 EPV threshold established
for regression models [Peduzzi et al., 1996; Babyak, 2004]. For state-space models specifically,
each state component (trend, seasonal, regression coefficients) consumes degrees of freedom
from the pre-period [Durbin & Koopman, 2012], so the effective parameter count includes s-1
seasonal states plus trend states plus covariate coefficients. With nseasons=14 and 6
covariates, the effective parameter count is ~21 against a 51-day pre-period — EPV ≈ 2.4.
Additionally, positive autocorrelation in daily data reduces the effective sample size:
N_eff = N / (1 + 2Σρ_t), which can reduce 51 nominal days to ~13 effective observations
at ρ ≈ 0.6 [Afyouni et al., 2019].

No universal minimum pre-period exists in the BSTS literature — Brodersen et al. [2015]
state no threshold, and the CausalImpact R package enforces only a floor of 3 time points.
Lopez Bernal et al. [2017] argue against universal minimums, recommending case-by-case power
simulation. Our ≥100-day guideline for 6+ covariates is a conservative practitioner threshold
supported by the Gils et al. simulation evidence and the EPV literature, but should always be
verified with a placebo test on the specific dataset.

> **If FPR > 10%, the spec is miscalibrated. Do NOT use its p-values for significance claims.**
> Report the calibrated specs' results instead, even if their p-values are higher.

### Scorecard

| Check | Threshold | What It Means |
|---|---|---|
| Median backtest WAPE | ≤ 15% | Model predictions are accurate |
| Mean 95% coverage | ≥ 80% | Uncertainty bands are well-calibrated |
| Placebo rank | ≥ 90th %ile | Real effect is unusual vs noise |
| **Pre-period FPR** | **≤ 10%** | **Model p-values are calibrated (not inflated)** |
| All specs same direction | ≥ (N-1)/N | Result is robust to covariate choice |
| Primary p-value (BSTS) | ≤ 0.10 | Effect is statistically significant (model-based) |
| **Permutation p-value** | **≤ 0.10** | **Effect is unusual vs random dates (empirical)** |

> **Both the permutation test and pre-period placebo test are REQUIRED.**
> - Permutation test (effect-size comparison): checks if the real date produces an unusually
>   large effect vs random dates. Uses `|abs_eff|` ranking, NOT model p-values.
> - Pre-period placebo test: checks if the MODEL is calibrated (false positive rate).
>   Uses model p-values by design — that's what it's measuring.
> - BSTS p-values alone are unreliable — multi-method testing (BSTS VI/HMC, Prophet, RDiT)
>   shows 35-55% FPR on daily retail revenue. ALL model classes are miscalibrated.
> - The permutation test is immune to this miscalibration because effect-size ranking is
>   unaffected by systematic uncertainty underestimation [Abadie 2010; Young 2019].

Passing all checks = strong result. Passing 4+ with consistent direction = defensible.
Failing FPR check = model is miscalibrated — find a better-calibrated spec before reporting.

### Watch for problematic backtest windows

Christmas/BF windows often show extreme WAPE (0% or 40%+) and massive placebo effects.
These inflate the placebo distribution, making the real effect look less unusual. Note this
in the findings — it's a feature of high-variance retail data, not a model failure.

## Step 6: Interpret Honestly

### The pre-period length question

More data is NOT always better. Longer pre-periods can hurt if:
- The website/tracking changed (structural break)
- The sessions→revenue relationship shifted (non-stationarity)
- COVID/post-COVID regime changes apply
- The client can't provide accurate campaign flags further back

The binding constraint is often **campaign calendar completeness** — the client may not have
reliable records of which promotions ran 2+ years ago. Flags like `winter_sale_flag` are only
useful if accurate.

**But too SHORT is worse than too long.** In one engagement, trimming the pre-period to
exclude holiday variance left only ~50 days of training data. BSTS with 6 covariates
overfitted this short series, achieving p=0.000 — but the pre-period placebo test revealed
22% FPR (the model detected "significant" effects at 22% of random dates where nothing
happened). The same specs with 200+ day pre-periods had much better calibration.

**Minimum pre-period guidelines for BSTS:**
- ≥ 100 days with 6+ covariates (the regression component needs data to estimate coefficients)
- ≥ 60 days with 2-3 covariates
- ≥ 3× nseasons as absolute floor (Kalman filter diffuse initialization) [Durbin & Koopman, 2012]
- If trimming to exclude holiday variance, verify the remaining pre-period is sufficient
- Always run the pre-period placebo test to verify calibration

Note: Brodersen et al. [2015] state no minimum pre-period; the CausalImpact R package enforces
only a floor of 3 time points. The ITS literature recommends a minimum of 8 observations for
segmented OLS [Penfold & Zhang, 2013], but BSTS with its Bayesian priors has different (and
generally lower) minimum data requirements. Lopez Bernal et al. [2017] argue there are "no
fixed limits" and recommend case-by-case power simulation. Our thresholds above are practitioner
guidelines supported by the effective-sample-size argument and our empirical FPR testing.

### Framing for clients

**When significant (p < 0.10):**
> "The campaign generated an estimated £X in incremental revenue (95% CI: £Y to £Z),
> with a Z% probability this effect is genuine."

**When not significant but consistent:**
> "Our best estimate is that the campaign generated approximately £X in incremental revenue
> (+Y%). This finding is consistent across all model specifications and two independent
> analytical methods. The probability the effect is genuinely positive is approximately Z%.
> Due to the short campaign duration (N days), the statistical confidence interval is wide —
> we recommend running future campaigns for at least 2 weeks to enable more precise measurement."

**Never claim** statistical significance when you don't have it. Credibility with clients
comes from honesty, not from overselling.

### Recommendations to always include

1. **Longer campaigns** dramatically improve detectability (2+ weeks ideal)
2. **Geo holdouts** provide cleaner control groups for future measurement
3. **Coordinate campaign calendars** to avoid overlapping interventions
4. **Double down on the lever identified** — if the promo lifted CR, test other conversion barrier removers

## Step 6b: Extension Analyses

> **Full details:** See [references/extension_analyses.md](references/extension_analyses.md)

Key extensions after the primary analysis:
- **Effect decomposition:** Run CausalImpact on conversion_rate, aov, transactions separately to identify which lever the campaign pulled
- **Channel split:** Separate paid_revenue vs organic_revenue to detect site-wide vs channel-specific effects
- **Post-promo persistence:** Compare during-promo vs after-promo daily effects (persistence_ratio)
- **Weather covariate:** Open-Meteo API for temperature/precipitation (orthogonal exogenous signal, ~2-5% CI tightening)
- **Prophet cross-validation:** Independent model family for high-stakes claims
- **Contaminated exogenous metrics:** Brand search is endogenous; use category-level Trends instead. Sale flags must be zeroed during treatment window (ADR 0020)
- **Sale auto-detection:** Coupon ratio bidirectional signal with MAD z-score
- **Tiered covariates:** Default (5 base) → Enhanced (+Trends) → Full (+sale signals), each permutation-validated

## Step 7: Document

Produce a markdown findings document with this structure:

```markdown
# [Client] [Campaign] — Causal Impact Analysis Findings

## 1. Executive Summary
   - Headline numbers: cumulative effect, relative %, CI, p-value, probability
   - One-paragraph interpretation

## 2. Data
   - Sources, date ranges, key metrics table
   - Analysis windows (pre-period, intervention, post)

## 3. Methodology
   - Approach explanation (accessible to non-technical readers)
   - Both methods described
   - Covariate table with correlations and rationale

## 4. Results
   - Primary model results
   - Cross-method comparison table
   - Sensitivity analysis table (all covariate bundles)

## 5. Validation
   - Backtest summary
   - Placebo test result
   - Scorecard

## 6. Interpretation
   - What we can confidently say
   - What we cannot claim
   - Why significance was/wasn't achieved

## 7. Recommendations
   - For the client (future campaign design)
   - For this analysis (next steps)

## 8. Files Reference
   - All scripts and plots generated

## 9. Technical Notes
   - Dependencies, auth, model config
```

## Step 8: Client Deliverables (optional)

If the work is client-facing, produce polished visual deliverables in addition to the markdown
findings doc. Two formats work well:

### Slide Deck (recommended for presentations)
An interactive HTML slide deck with keyboard/touch navigation. 10 slides covering:
title, key metrics, counterfactual chart, mechanism (traffic vs conversion), decomposition,
channel split, persistence, robustness, transparency/scorecard, recommendations.

Use the **`frontend-design` skill** for distinctive, production-grade HTML. Specify the
data points, sections, and audience (non-technical marketing team). Key design notes:
- Use fade+scale transitions (not directional translateX — causes backwards navigation bugs)
- Include keyboard (arrow keys, spacebar) and touch swipe navigation
- Dark title slide, light content slides
- Embed charts as SVG (no external image dependencies)

### Scrolling Report (recommended for async sharing)
A single-page HTML report with scroll-triggered animations. Same content as the deck but
in a continuous format. Better for emailing to stakeholders who will read at their own pace.

### Jupyter Notebook (for internal DS team)
A reproducible notebook with full code, diagnostics, and commentary. Separate audience from
the client deliverables — include model diagnostics, correlation heatmaps, and validation
details that would overwhelm a non-technical reader.

### Interactive Explorer (recommended for client self-service)

A single self-contained HTML file with Plotly.js charts and pre-computed scenario data embedded
as JSON. The client opens it in their browser — no server, no install, works offline after first
load. This is the **highest-impact deliverable** because it lets the client explore robustness
themselves rather than trusting a static summary.

**Architecture:** One HTML file containing:
- CSS design system (reuse from slide deck/report)
- Plotly.js loaded via CDN
- All scenario data embedded as a `const DATA = {...}` JSON object in a `<script>` tag
- JavaScript event handlers for dropdowns, tabs, and card clicks

**5 interactive sections:**

| Section | Interaction | What updates |
|---|---|---|
| Hero + Spec Selector | Dropdown to select model specification | Headline £, %, p-value, confidence badge, all metric cards |
| Counterfactual Chart | Plotly.js zoom/hover/pan + CI band toggle | Interactive time series (observed vs predicted) |
| Method Comparison | Clickable method cards | Horizontal bar chart with CI error bars, method description |
| Effect Decomposition | Metric cards with animated probability bars | CR, Transactions, Revenue, AOV breakdown |
| Channel & Persistence | Tabbed view (persistence / paid vs organic) | Bar charts + summary metrics per tab |

Plus a **validation scorecard** table with pass/borderline/fail indicators.

**Data to pre-compute and embed as JSON:**

```javascript
const DATA = {
  specs: {
    // One entry per model specification (typically 6-8)
    best: {
      name: "Recommended spec name",
      effect: 297000,       // cumulative £
      daily: 74250,         // daily avg £
      relative: 22,         // relative % lift
      pval: 0.039,
      probPos: 96.1,
      ciLow: -298000,
      ciHigh: 762000,
      sig: true,            // p < 0.05?
      sigLabel: "p < 0.05",
      preStart: "Jan 6 2025",
      note: "Human-readable description of this spec"
    },
    // ... repeat for each sensitivity spec
  },
  methods: {
    // One entry per analysis method
    bsts:      { name: "...", effect: N, ciLow: N, ciHigh: N, desc: "...", metric: "...", pval: "..." },
    rdit:      { ... },
    causalpy:  { ... },
    conformal: { ... }
  },
  timeseries: {
    // Daily arrays for the counterfactual chart
    dates: ["2026-02-13", ...],
    observed: [210000, ...],
    predicted: [208000, ...],
    ciUpper: [340000, ...],
    ciLower: [80000, ...]
  }
};
```

**Key implementation notes:**
- **Per-method prediction lines:** Each method can return its own counterfactual series via
  `return_series=True`. BSTS returns full pre+post predictions; RDiT returns local regression
  within bandwidth (null outside); CausalPy returns posterior predictive mean. Store as
  `per_method_series = { method_id: { dates, predicted, upper?, lower? } }` alongside the
  primary `chart_series`. Render with distinct colors (BSTS teal, RDiT orange, CausalPy purple).
  Add an "Uplift from" dropdown to select which method's counterfactual drives the uplift arrow.
- **Data size:** Daily granularity × ~30 days around intervention × a few scenarios = tiny (<100KB JSON).
  The Plotly.js CDN (~3.5MB) is the main dependency — cached after first load.
- **Counterfactual data source options:**
  1. **Best:** Extract `ci.inferences` from the tfcausalimpact run (observed, preds, preds_lower, preds_upper)
     and save as CSV/JSON during the analysis step.
  2. **Fallback:** Generate synthetic counterfactual = observed − (daily_effect estimate) for the promo period.
     Less precise but sufficient for interactive exploration.
- **Spec selector:** Updates all metrics simultaneously with a fade animation — gives the client an intuitive
  sense of how stable the result is across specifications.
- **Method comparison chart:** Horizontal bar chart with error bars. A red dashed zero-effect line makes it
  visually obvious whether each method's CI excludes zero.
- **Decomposition bars:** Animated on scroll using IntersectionObserver — fills the probability bars
  (e.g., "83% prob+" → bar fills to 83%) when the section scrolls into view.
- **Scroll-triggered sections:** Use `IntersectionObserver` with `threshold: 0.15` to fade in sections.
  Same pattern as the static report.

**When to extract daily predictions (preferred):**
After running the BSTS model in Step 4, save the inference arrays:

```python
# After CausalImpact fit
inf = ci.inferences
export_df = pd.DataFrame({
    'date': inf.index,
    'observed': inf['response'],
    'predicted': inf['preds'],
    'ci_lower': inf['preds_lower'],
    'ci_upper': inf['preds_upper']
})
export_df.to_csv('counterfactual_data.csv', index=False)
# Then embed this CSV as JSON in the interactive HTML
```

**Use the `frontend-design` skill** for the HTML/CSS to ensure production-grade visual quality.
Pass it the data structure, section layout, and design system (fonts, colours) from the existing
static deliverables.

### When to use which

| Format | Audience | Best for |
|---|---|---|
| **Interactive explorer** | **Client self-service** | **Email/share link — client explores robustness themselves** |
| Slide deck | Client meeting | Live presentation, screen sharing |
| Scrolling report | Client async | Email attachment, static self-service reading |
| Jupyter notebook | Internal DS team | Verification, iteration, collaboration |
| Markdown findings | Internal (analyst-style) | Quick sharing, PR review, documentation |

### Masked period visualization (critical for reports)

When masking is used (Step 3b: excluding winter periods), all output charts MUST visually
show which dates were masked. Without this, users assume the model trained on all plotted data.

**Implementation pattern (Plotly.js):**
```javascript
// Add orange shaded rectangles for each masked period
maskedRanges.forEach(function(r) {
  shapes.push({
    type: "rect", x0: r.start, x1: r.end, y0: 0, y1: 1,
    yref: "paper", fillcolor: "rgba(255,152,0,0.12)", line: { width: 0 }
  });
});
// First masked band gets a text label
annotations.push({
  x: maskedRanges[0].start, y: 0.98, yref: "paper",
  text: "Masked (not used for training)",
  showarrow: false, font: { size: 10, color: "#e8710a" }
});
```

**Also include:**
- Config summary banner showing mask mode + training day count
- Callout box explaining what masking means and why it was applied
- Legend distinguishing: Pre-period (blue), Masked (orange), Intervention (red)

This pattern applies to any form of data subsetting — not just winter masking.

## The Meta-Lesson: Subtract Before You Add

When a causal impact model doesn't achieve significance, most analysts' instinct is to add
more covariates. In practice, the biggest improvements come from **removing** things:

| Action | Type | Typical impact |
|---|---|---|
| Exclude high-variance periods from pre-period | Subtraction | -27% CI width |
| Remove contaminated covariates (changed during intervention) | Subtraction | +36% effect estimate, p halved |
| Remove covariates redundant with built-in model components | Subtraction | Cleaner model, less noise |
| Add better covariates (multi-modal holiday intensity) | Addition | -11% CI width |
| Add exogenous signals (weather) | Addition | -3% CI width |

In a retail engagement, this took p from 0.223 to 0.039 — all from the same data, same model
architecture. Three of five improvement steps were subtractions.

**Critical caveat: specification search.** If you test N specifications and report the one with
the lowest p-value, the result is exploratory, not confirmatory. In a retail engagement, 48
experiments were conducted. The best spec (p=0.039) was an outlier — all 6 sensitivity specs
had p=0.18-0.24. A multi-agent review panel flagged this as the central methodological concern.
The honest framing is: "The primary specification produces p=0.21. An optimised specification
achieves p=0.039, but this should be treated as exploratory." See the Claim Framing Guide below.

**Practical workflow:** When p > 0.10, try these in order:
1. Shorten the pre-period (exclude high-variance events)
2. Run the combined covariate audit — remove anything that changed >10% during intervention
3. Check for redundancy with built-in model components (e.g., nseasons)
4. THEN add better covariates (intensity curves, weather, interactions)

**But:** Document every experiment. If the "best" spec is the only one that achieves significance,
it is exploratory. Lead with the Bayesian posterior probability across ALL specs, not the
cherry-picked p-value.

## Claim Framing Guide (from adversarial review)

A multi-agent review panel (Statistical Rigor Reviewer, Feasibility Analyst, Code Quality
Auditor, Devil's Advocate + Supreme Judge) identified this as the most critical lesson:

**The client doesn't need p < 0.05. They need to know whether to run the promo again.**

### What to lead with (in all deliverables)

```
"The promo generated a positive incremental revenue estimate across all 12
specifications (all positive). When winter sale periods are excluded from training
(a principled data preparation step), the model achieves formal significance at
p < 0.05 with ~95% probability the effect is genuine. The promo drove conversion
(~+14%), not traffic."
```

**The masking breakthrough:** When seasonal variance inflates CIs to the point where
significance is impossible, try masking the high-variance windows (e.g., Nov-Jan for
retail) rather than truncating the pre-period or cherry-picking model configurations.
Masking is a data preparation decision, not model tuning — it's more defensible than
specification search. In a retail engagement, masking both winter sale periods
reduced CI width by ~60% and moved p from ~0.19 to < 0.05.

**CRITICAL caveat — masking + tight models can cause overconfident permutation results:**
In a subsequent validation, the masked spec (p=0.047 BSTS) showed permutation p=0.22
(11/49 random dates produced effects as large as the real promo). ALL specs with
mask_nov_jan + nseasons=14 showed permutation p > 0.15 — including enriched covariate
variants (3 to 13 covariates). The masking removes high-variance months, leaving only
calm Feb-Oct data. The model becomes very "sure" of predictions from this calm period,
producing low p-values for BOTH real and random intervention dates.

**Always validate masking with a permutation test (50+ random dates).** If permutation
p > 0.10 while BSTS p < 0.05, the model is overconfident. Report BOTH p-values:
"BSTS p=0.017 (model-based), permutation p=0.24 (empirical)" — the permutation p
is the honest one.

**Mask width matters:** In a retail engagement, mask_nov_jan (wide, 330 training days)
produced permutation p=0.22, while mask_xmas (BF-Jan 5 only, 410 training days) achieved
permutation p=0.059. The narrower mask is better because:
1. It keeps November pre-BF data which has useful variance patterns
2. Pre-BF revenue spikes (z=+13) are INFORMATIVE — they teach the model that revenue
   CAN spike for non-promo reasons, making it less likely to flag random dates
3. More training data = better model calibration

**Weather covariates for permutation robustness:** Weather (temp + precipitation) had
minimal BSTS impact (CI -3.2%) but was the single biggest permutation improver (0.098→0.059).
Truly exogenous, orthogonal signals help the model explain residual variance at random dates
without overfitting to seasonal patterns. Always include weather if available.

**Where does the probability come from?** Use 1-p from your recommended spec.
If you have masked and unmasked variants, quote both: "95% with masking, 81% without."
Never let the headline probability come from an overconfident short pre-period (p≈0).

### What NOT to lead with

```
"The promo generated £XXK, statistically significant at p < 0.05."
(Unless this was the pre-registered primary specification)
```

### Scorecard framing

When showing validation scorecards across deliverables:
- **All deliverables must show the same primary p-value.** If the deck says "Pass" and the
  report says "Fail" for the same test, the client loses trust immediately.
- Show the primary (first-specified or pre-registered) p-value as the main result.
- If an optimised spec achieves a better p-value, show it separately labeled "Exploratory."
- The convergence of all specs on a positive direction IS the strongest evidence — foreground it.

### The specification search trap

| What you did | How to frame it |
|---|---|
| Pre-registered one spec, it was significant | Lead with it. This is confirmatory. |
| Tested N specs, all significant | Lead with the range. This is robust. |
| Tested N specs, best one is significant | **Lead with the range. Best spec is exploratory.** |
| Tested N specs, none significant | Lead with Bayesian posterior and direction consistency. |

### Synthetic data in charts

Never use `Math.random()` for client-facing charts without a clear "illustrative" label.
Always export `ci.inferences` during the analysis step and embed real model predictions.
A client who reloads the page and sees different numbers will question everything.

## Key Pitfalls to Avoid

1. **Using covariates affected by the intervention** — the #1 failure mode. Always check
   whether the campaign could have influenced each covariate.

2. **Binary flags for high-variance events** — use continuous intensity variables instead.
   A binary `sale_flag` can't explain why BF is 5x a normal sale day.

3. **Claiming significance when p > 0.10** — destroys credibility. Be honest about uncertainty.

3b. **Claiming significance from specification search** — if you tested N specs and only the
   "best" achieved p < 0.05, the result is exploratory. Bonferroni correction: multiply p by N.
   Lead with the Bayesian posterior across all specs instead. This was the #1 finding from
   adversarial review of the client deliverables.

4. **Too-long pre-periods with structural breaks** — if predictions get worse with more data,
   shorten the pre-period. Find the sweet spot.

5. **Ignoring the Christmas distortion** — for retail clients, BF/Christmas introduces extreme
   variance that inflates all credible intervals. Model it explicitly.

6. **Running only one method** — a single implementation might have bugs or assumptions that
   bias the result. Two independent methods with different inference engines is the gold standard.

7. **Forgetting the dependency conflict** — tfcausalimpact and CausalPy cannot coexist in the
   same Python environment. Always run in separate scripts.

8. **Including high-variance seasonal periods in the pre-period** — For retail clients, the
   Christmas/Black Friday period can inflate credible intervals by 30%+. Test excluding it:
   start the pre-period after Jan 6 (post-Christmas hangover). In the retail case study, this reduced
   CI width by ~27% and improved p-value by ~25% while the
   effect estimate remained stable. Always run a pre-period sensitivity test
   with multiple start dates to find the optimal noise/data tradeoff.

9. **Fake p-values in multi-method pipelines** — When wrapping multiple methods (BSTS, CausalPy,
   RDiT, conformal) into a unified output dataclass, NEVER fill in placeholder p-values for
   methods that don't produce them. Common traps:
   - CausalPy linear regression: tempting to return `p=0.25` as a "neutral" constant
   - Conformal CIs: tempting to return `p=0.5 if CI_contains_zero else 0.05` (a binary hack)
   These appear alongside real BSTS p-values in comparison tables and mislead users. Return
   `p=None` and display "N/A" in the UI. If a method produces a Bayesian posterior, label it
   explicitly (e.g., "P(positive): 87%") rather than converting to a fake frequentist p-value.

10. **Probability range inflation via 1-p conversion** — When converting specification-searched
    p-values to "probability of positive effect" (1-p), the bias carries over. If you tested 48
    specs and the best gave p=0.039, quoting "96% probability" (1-0.039) smuggles the cherry-picked
    result into a Bayesian-sounding claim. Always derive the probability range from a SYSTEMATIC
    sensitivity rerun. In the retail case study, 16 specs gave p=0.102-0.469, mapping to 53-90% probability.
    The honest range is ~75-90%, not 80-96%.

11. **Temporal scope gaps in seasonal exclusions** — When a spec claims to "exclude Christmas"
    or "mask winter sales," verify it covers ALL instances across the full date range. With 2+
    years of data, a Jan 6 start date only excludes the first Christmas. The second one is still
    in the training window. This evaded 3 rounds of adversarial review in a real engagement.
    Always label exclusions with specific years: "masks Nov-Jan 2024 + 2025 (both years)."

12. **Internal reference docs drifting from deliverable framing** — After a review panel prompts
    reframing (e.g., from "p=0.039, significant" to "80-90% probability, exploratory"), update the
    internal reference document (ANALYSIS_FINDINGS.md exec summary, narrative sections) in the SAME
    commit as the deliverables. Auditors and new team members read the internal doc first. If it still
    leads with the old framing, the reframing looks cosmetic.

13. **Data-prep "mask" functions that silently fill instead of remove** — When your pipeline has an
    `apply_date_mask` step followed by a `prep_df` / `preprocess` step, the mask may not actually
    reach the model as a removal. A common latent bug: `apply_date_mask` drops rows correctly, then
    `prep_df` rebuilds a continuous daily index (`pd.date_range(min, max, freq="D")` + `reindex`) and
    fills the reinserted rows with `y = 0` and linear-interpolated covariates. The model then trains
    on 66 days/year of fake zero revenue with smoothly walking Fourier/Trends, learning a spurious
    "winter covariates ↔ £0 revenue" relationship. This was discovered as a latent bug in a real
    retail engagement and filed as a priority:high issue. See the dedicated section below for the
    full case study, reproduction recipe, and verification pattern. ALWAYS trace the masked date
    path through the entire prep pipeline to the model-input layer before trusting the mask.

## The Data-Prep Zero-Injection Trap (Case Study + Verification Recipe)

**Context:** A "mask" step in a causal impact pipeline is supposed to remove high-variance periods
(Christmas, Black Friday) from the model fit so credible intervals aren't inflated. If the mask
is implemented naively, it may silently fail in a way that LOOKS correct but distorts every
downstream method.

**The bug class (generalisable):**

```python
# Step 1: "Mask" function correctly drops rows
def apply_date_mask(df, d_col, mask_mode):
    if mask_mode == "mask_nov_jan":
        in_mask = ((df[d_col].dt.month == 11) |
                   (df[d_col].dt.month == 12) |
                   ((df[d_col].dt.month == 1) & (df[d_col].dt.day <= 5)))
        return df[~in_mask].reset_index(drop=True)   # ← crops as expected

# Step 2: prep_df rebuilds a continuous index and fills the gaps
def prep_df(df, d_col, y_col, x_cols, ...):
    d = df.set_index(d_col).sort_index()
    idx = pd.date_range(d.index.min(), d.index.max(), freq="D")
    d = d.reindex(idx)                               # ← REINSERTS MASKED DATES AS NaN
    d[y_col] = d[y_col].fillna(0)                    # ← REVENUE = 0 for masked days
    d[x_num] = (d[x_num]
                  .interpolate(method="linear", ...)
                  .bfill().ffill().fillna(0))        # ← COVARIATES INTERPOLATED
    return d

# Step 3: All methods dispatch from the same prepped dataframe
d = prep_df(df, ...)       # zero-injected here
run_bsts(d, ...)           # sees 66 days/year of fake £0 + interpolated covariates
run_prophet(d, ...)        # same
run_causalpy(d, ...)       # same
```

**What the model actually sees (example, Elena's config 2024-01-01 → 2026-02-26):**

```
2024-10-31   £8,254   trend=30.3   ← last real day
2024-11-01   £    0   trend=30.6   ← reinserted, rev=0, trend interpolated
2024-11-15   £    0   trend=35.2   ← 2 weeks into "mask," trend walks up
2024-12-25   £    0   trend=48.1   ← Christmas Day: revenue = £0 in the training data
2025-01-05   £    0   trend=51.6   ← last masked day
2025-01-06   £10,455  trend=51.9   ← real again
```

**Consequences:**

1. **Spurious covariate→target learning.** The model learns "when Fourier/Trends indicate Nov-Dec,
   revenue is £0." This is a fake signal that biases the counterfactual for EVERY post-treatment
   prediction.
2. **Inflated pre-period posterior variance.** The model sees revenue swing from ~£10K → £0 → ~£10K
   twice per year, widening credible intervals dramatically.
3. **Permutation test amplification.** Shuffled placebo treatments inherit the distorted fit.
   In the retail case study, the top placebo shuffle produced +£2.1M vs a real effect of +£328K
   (6.5x larger), yielding permutation p = 0.47.
4. **False positive rate inflation.** Part of the "mask_bf_jan produces 93% FPR" observation
   (previously thought to be about "masking creates gaps") is actually this zero-injection bug.

**Which methods are affected?**

The bug is at the **data-prep layer**, so every method that consumes the prepped dataframe is
affected, regardless of its internal architecture:

| Method | Affected? | Why |
|---|---|---|
| BSTS VI / HMC (tfcausalimpact) | **YES** — severe | Fits state-space on full pre-period including zero-injected rows |
| Prophet | **YES** | Fits zeros as real observations; distorts trend + yearly seasonality |
| CausalPy LR (Bayesian regression) | **YES** — possibly worse | No temporal smoothing; linear fit pulls trend toward zero |
| Conformal VI | **YES** — severe | BSTS backbone + conformal quantile over pre-period residuals (double hit) |
| **RDiT local linear** | **NO** (if bandwidth doesn't overlap mask) | Uses only a narrow window around the treatment date — doesn't touch the masked period |

**RDiT is the only bug-independent baseline** when its bandwidth window doesn't overlap any masked
period. This has a strong implication: if your multi-method comparison shows BSTS/Prophet/CausalPy
agreeing (e.g., +38% uplift) while RDiT disagrees (e.g., +0.1%), RDiT may be the one reading the
true signal — the others are all sharing the same bugged input.

**The "HMC confirms VI" trap:** switching between variational inference and HMC sampling does NOT
fix this bug. Both samplers train on the same zero-injected dataframe. Apparent "independent
confirmation" via HMC is not independent at all — it only rules out sampler stochasticity, not
data-prep bugs.

**Verification recipe — always run this before trusting a mask mode:**

```python
# 1. Create synthetic data matching your pre-period window
import pandas as pd, numpy as np
dates = pd.date_range("2024-01-01", "2026-02-28", freq="D")
df = pd.DataFrame({
    "date": dates,
    "revenue": 10000 + 2000*np.sin(2*np.pi*np.arange(len(dates))/365),
    "trend":   50    + 20  *np.sin(2*np.pi*np.arange(len(dates))/365),
})

# 2. Apply your mask
masked = apply_date_mask(df, "date", "mask_nov_jan")
print(f"After apply_date_mask: {len(masked)} rows")   # should be ~66*years less

# 3. Apply prep_df
prepped = prep_df(masked, d_col="date", y_col="revenue", x_cols=["trend"])
print(f"After prep_df:         {len(prepped)} rows")  # if SAME as original = BUG

# 4. Inspect what the model actually receives
print(prepped.loc["2024-10-30":"2025-01-08"])
# If you see Nov-Dec rows with revenue=0 and interpolated covariates, the mask is
# not actually masking — it's zero-injecting. FIX REQUIRED.
```

**If you find this bug in your pipeline:**

- **Option A (quick spike):** Change `d[y_col] = d[y_col].fillna(0)` to leave NaN. Test whether
  tfcausalimpact / your BSTS library tolerates missing y values. Some state-space libraries do.
- **Option B (clean):** Segmented BSTS fit — split the pre-period into contiguous chunks, fit a
  separate BSTS per chunk, then combine counterfactuals for the post-period. Statistically clean
  but a significant refactor.
- **Option C (simplest):** Deprecate the mask mode entirely. Use only contiguous pre-period
  windows (e.g., "post-Xmas trimmed": Jan 6 → treatment date). This is the approach the Schuh
  project adopted for its lead spec after discovering the bug.
- **Option D (stopgap):** Rename the mode to `fill_nov_jan_zero` so the behaviour matches the
  name, and document prominently that it is not equivalent to removing those days.

**Client-communication rule:** never say "we masked Christmas out" if your implementation uses
the zero-injection pattern. The accurate description is "we excluded the Nov-Jan period from the
treatment comparison" — and if you know the implementation is affected, add a caveat or footnote.

**Meta-lesson:** When a function is named `mask_X` / `filter_X` / `drop_X` and is used upstream
of a model fit, ALWAYS trace the full pipeline through to where the model receives the data.
Check for: (a) reindex operations that reinsert removed dates, (b) fillna / interpolate calls
that populate gaps, (c) any continuous-frequency assumptions (daily, weekly) that force gap-filling.
A "mask" in the data-prep sense often becomes a "fill" at the model-input sense. Describe
mask behaviour by what the MODEL sees, not by what the intermediate dataframe looks like.

## Multi-Method Pipeline Gotchas (v1.4.0)

When building a webapp or unified pipeline that runs multiple CI methods:

### Unified output type

```python
@dataclass
class CiOut:
    abs_eff: float
    abs_low: float
    abs_up: float
    rel_eff: float
    p: float | None  # None for methods without p-values
```

Methods that produce real p-values: `tfcausalimpact` (BSTS), `run_rdit` (bootstrap).
Methods that do NOT: `CausalPy` (Bayesian posterior, not frequentist), conformal CIs
(distribution-free intervals, no p-value concept).

### RDiT bandwidth sensitivity

Always test RDiT at multiple bandwidths (e.g., 7, 14, 21, 28 days). A result that is
"significant" at only one bandwidth is fragile. In the retail case study:

| Bandwidth | Cumulative Effect | 95% CI | Significant? |
|-----------|------------------|--------|-------------|
| 7 days | 125K | [-80K, 281K] | No |
| 14 days | 162K | [34K, 289K] | Yes |
| 21 days | 248K | [120K, 372K] | Yes |
| 28 days | 266K | [131K, 371K] | Yes |

3/4 bandwidths significant with monotonically increasing effect and tightening CIs =
robust signal. If only the narrowest bandwidth passes, the result is fragile.

### Sensitivity spec generation

When auto-generating specs for sensitivity testing, categorise covariates by type:
- **Session covariates:** organic_sessions, paid_sessions
- **Calendar flags:** holiday_flag, payday_window_flag, is_weekend, kcp_period_flag
- **Seasonality:** sin_dow, cos_dow, fourier terms, xmas_intensity
Then generate: minimal (1 session covariate), calendar-only, seasonality-only,
calendar+seasonality, full, without-paid, kitchen-sink. Test across 2+ pre-period
start dates. Report the FULL range and median, not just the best result.

## Methodology Communication — Explaining Your Tests Honestly

Methodology write-ups for causal-impact analyses routinely make the same handful of mistakes:
they mis-state what the BSTS p-value actually is, they skip identification assumptions entirely,
they treat "passes placebo" and "passes permutation" as two rungs of the same ladder, and they
respond to a failing test by spec-shopping. This section is the antidote. It's what you should
say — and what you should avoid saying — when you write a methodology section, a client deck,
or an internal teaching note.

### The `tfcausalimpact` p-value is NOT what you think

The single most common error in methodology write-ups is describing the BSTS p-value as:

```
p = (1/N) × #{posterior samples where effect ≤ 0}         ← WRONG
```

That formula is wrong in three independent ways. The actual formula in `tfcausalimpact`
(source: `causalimpact/inferences.py`, function `compute_p_value`) is:

```python
sim_sum = tf.reduce_sum(simulated_ys, axis=1)         # post-period sum per sim
signal  = min(
    np.sum(sim_sum > post_data_sum),                   # upper-tail count
    np.sum(sim_sum < post_data_sum),                   # lower-tail count
)
return signal / (len(simulated_ys) + 1)
```

In math:

```
p = min( #{sim_sum_i > obs_sum},  #{sim_sum_i < obs_sum} )  /  (N + 1)
```

Three things matter:

1. **It is two-sided**, not one-sided. The `min(upper_tail, lower_tail)` clearly takes a
   minimum of the two tail counts.
2. **The denominator is `N + 1`**, not `N`. This is the [Phipson & Smyth, 2010] Monte-Carlo
   correction so the value can never equal exactly 0.
3. **The compared quantity is the post-period sum, not the "effect"**. It counts simulated
   counterfactual post-period sums against the actual observed post-period sum, not samples
   of a derived "effect ≤ 0" quantity.

A sanity check: if you see `p = 0.005`, that's almost certainly `5 / 1001` with `N = 1000` sims
— no other obvious numerator/denominator combination produces `0.004995…` from the wrong formula
people usually write down.

**Read it as a posterior predictive p-value** in the sense of [Gelman, Meng & Stern, 1996], not
as a frequentist p-value. Unlike a frequentist p-value, posterior predictive p-values are **not
uniformly distributed under the null even asymptotically** when the null is composite (which
BSTS's is, with many nuisance parameters). They are conservative and tend to **cluster around
0.5** [Robins, van der Vaart & Ventura, 2000; Bayarri & Berger, 2000]. So a value of 0.005 is
*stronger* evidence than a frequentist p of 0.005 in one direction (it's harder for a PPP to
fall below 0.05 by chance) and *less interpretable as a calibrated false-positive rate* in the
other (it isn't a Type-I error rate). Comparing it to the conventional 0.05 cutoff is a
convention imported from frequentist NHST, not a property of the quantity.

**When to prefer HMC over VI**: `tfcausalimpact` supports both variational inference (VI, fast)
and Hamiltonian Monte Carlo (HMC, slower but better-calibrated). VI is known to
**underestimate posterior variance** [Blei, Kucukelbir & McAuliffe, 2017], which makes CIs too
narrow and posterior predictive p-values smaller than they should be. When the headline number
matters, prefer HMC. When iterating on spec exploration, VI is fine — just don't cite a VI
p-value in a client deck without an HMC re-run.

**The simpler Bayesian alternative** — the quantity most write-ups *wish* they were describing
— is the posterior probability of direction:

```
p_d = max( Pr(effect > 0 | y),  Pr(effect < 0 | y) )
```

This is cheap to compute from `ci.inferences['point_effects']` and [Makowski et al., 2019]
formalise it as a Bayesian index of effect existence. Report it as *"Probability that the effect
is positive: 87%"* rather than forcing a p-value word onto a Bayesian quantity.

### Identification assumptions: state them, or you're not doing causal inference

Any methodology write-up that does not enumerate its identification assumptions is not a
causal-inference document — it's a prediction document with a misleading label. The minimum set
for a BSTS causal-impact analysis:

1. **Sharp intervention.** The treatment effect is zero outside `[real_start, real_end]` and
   present on every day inside it. No anticipation, no carryover.
2. **No interference (SUTVA in time).** The pre-period contains no residual effects from
   earlier campaigns. Any mask you apply is assumed to cover all such past effects.
3. **Covariate exogeneity.** Your covariates are not themselves caused by the treatment, or by
   anything that also causes the outcome. This is the "bad controls" problem in the
   Angrist-Pischke sense — conditioning on a post-treatment variable biases the effect estimate.
4. **Pre-period regime stationarity.** The data-generating process for the earliest pre-period
   is exchangeable with the data-generating process for the latest pre-period. Any permutation
   or rolling test assumes this; seasonality + trend in real retail data often break it.
5. **Mask non-informativeness.** If you mask high-variance periods (Nov-Dec, Black Friday), you
   assume the mask removes dates at random with respect to the residual structure. **Issue #51
   in the Schuh engagement was a concrete violation of this** — the masked dates were silently
   re-injected as £0 revenue, which taught BSTS a spurious "winter ↔ £0" relationship. See "The
   Data-Prep Zero-Injection Trap" section above for the case study. If you're doing anything
   more sophisticated than "drop the rows", verify the mask actually reached the model input.
6. **Randomization-test exchangeability.** Any permutation / date-shuffled placebo test
   additionally requires that fake training windows are exchangeable with the real training
   window. In a rolling-origin sweep this is usually violated — training window length is a
   confounder — and the test becomes conservative rather than calibrated. More on this below.

**Rule:** a methodology section that doesn't list these in one place (or an explicit subset) is
a methodology section you shouldn't ship.

### Three tests, three different questions — not three rungs of a ladder

The four commonly-used validation checks in BSTS causal-impact work answer different questions
under different validity assumptions. They are not interchangeable. A write-up that frames them
as "primary test X + fallback tests Y, Z" is smuggling an implicit judgment about which test
"really counts". Don't do that.

| Test | Asks | Validity hinges on |
|---|---|---|
| **Model p-value** (posterior predictive) | Under the model's own posterior predictive distribution, is the observed post-period sum unusual? | Model + prior correctness; NOT a frequentist Type-I rate |
| **Rolling placebo** (in-time, recent) | Would this model have falsely detected an effect on a recent no-treatment window near the real date? | Recent pre-period being a representative replicate of post-period (no anticipation, no recent regime shift) |
| **Date-shuffled randomization** (Monte-Carlo in-time placebo) | Across random fake treatment dates drawn from the full pre-period, how unusual is the observed effect magnitude? | Exchangeability of the response across time *and* training-window length not confounding the null — both typically violated for retail time series |
| **Specification grid** (partial SCA) | Across many defensible modelling specifications on the same real data, do we agree on sign and magnitude? | Defensible specs were enumerated *before* seeing the result; see [Simonsohn, Simmons & Nelson, 2020] |

Three things to keep straight when teaching these to a stakeholder:

- **Model p-value is Bayesian.** Don't write it as `Pr(effect ≤ 0 | model, data)` — that's
  neither what the code computes nor what you want. Use the formula above or just call it
  "posterior probability of direction".
- **"Rolling placebo" is not the same as "rolling-origin cross-validation"**. [Hyndman &
  Athanasopoulos, 2021] treats rolling-origin CV as the standard for *forecast accuracy
  evaluation*, not for placebo significance. Calling your placebo test a "backtest" borrows a
  name from forecasting that doesn't carry the same inferential guarantees.
- **"Permutation test" is a misnomer** for what most BSTS pipelines do. The canonical
  permutation test [Fisher, 1935; Imbens & Rubin, 2015, ch.5] exchanges treatment-vs-control
  *labels across units* assuming exchangeability under the sharp null. A date-shuffled test on
  a single time series exchanges *fake treatment dates across time within one unit* — closer
  in spirit to the in-time placebo of [Abadie, Diamond & Hainmueller, 2010] / [Eggers, Tuñón
  & Dafoe, 2024]. Rename it "date-shuffled randomization test" in client-facing write-ups.
- **Specification grid ≠ specification curve analysis**. A full [Simonsohn, Simmons & Nelson,
  2020] SCA is a three-step procedure: enumerate specs, display the curve, compute a
  bootstrap-based **joint inference test** across the curve. Most projects (this one included)
  implement steps 1 and 2 + per-spec permutation tests on the top specs — which is closer to a
  [Steegen, Tuerlinckx, Gelman & Vanpaemel, 2016] multiverse-of-modelling than a canonical
  spec curve. Don't claim "Simonsohn-validated" unless you've run the joint test.

### Diagnose, don't demote

When model-p passes but a placebo test fails, the dominant reaction in practitioner workflows
is to spec-shop: run the SCA grid, find the spec that passes all the tests, promote it to the
headline, "demote" the original canonical. **This is the failure mode [Simonsohn, Simmons &
Nelson, 2020] and [Gelman & Loken, 2013] (the "garden of forking paths") were explicitly
designed to surface.** Post-hoc selection on a test statistic is mild p-hacking whether you
mean it that way or not.

The honest response is:

1. **Diagnose the failure** before reacting. Two cheap follow-ups, each ~50 fake-window fits
   (~$0.05 on Cloud Run using the existing placebo handler — do NOT write a new handler):

   - **Training-length-matched permutation** — for a failing date-shuffled test, re-run with
     fake training windows constrained to match the real fit's training length (e.g.
     `fake_start - pre_start ≥ 600 days`). Training-window length is a first-order confounder
     for single-unit time-series randomization: short-window placebos are noisier and inflate
     the null distribution. If the p improves substantially after matching, training length
     was driving the original FAIL and the canonical was *partly* a low-power artefact. If p
     barely moves, the model really does produce large effects on no-treatment dates and you
     have a level-shift problem, not a "permutation-fragile" hand-wave. In practice the
     post-match p usually lands somewhere in between — partial artefact, partial residual.
   - **Mask-off rolling placebo** — for a passing rolling backtest with a masked pre-period
     (`mask_nov_jan`, `mask_bf_jan`, `post_xmas_trimmed`), re-run the same placebo grid with
     the mask turned off. If `rank` stays identical (±0.02), the theoretical mask-interaction
     bias that reviewers worry about is empirically zero in your data. If it moves a lot, the
     mask is carrying load in the placebo distribution and your rolling-backtest story has a
     hidden dependency — surface it.

   **Neither diagnostic tries to rescue a failing test.** They tell you *why* the test failed
   (or *whether* it would have failed without a suspected confound), which is a different
   question from "does the effect survive?". Answer that with the spec grid.
2. **Decompose placebo failure into bias and variance.** Bin placebos by training-window
   length (e.g. <1 yr, 1–1.5 yr, 1.5–2 yr) and report mean and SD of fake effects per bin.
   - High bias, low variance → real model over-confidence (the failure is genuine)
   - Low bias, high variance → estimator variance; the test has no power (the failure is a
     low-power artefact, not over-confidence)
   - High bias, high variance → regime non-stationarity; the pre-period is not exchangeable
   - Low bias, low variance → the model is fine; the failure is MC noise in the p-value
3. **For low-power studies, prefer Type-S / Type-M error over Type-I.** [Gelman & Carlin,
   2014] argue convincingly that for single-unit, short-window interventions, the relevant
   errors are wrong sign (Type-S) and wrong magnitude (Type-M), not false positive. A
   "statistically significant" effect in a low-power regime is likely to have both problems.
4. **Report the multi-method ensemble as a range, not a point.** If RDiT says £160K, BSTS
   HMC says £298K, and Prophet says £355K, your headline is "£160K–£355K with a median
   around £290K" — not "£298K per BSTS". The dispersion is itself information.
5. **Be explicit about which spec is doing the heavy lifting.** If the headline depends on
   selecting the lowest-p spec from a 224-spec grid, call that out. "Best spec in a
   pre-registered grid" is defensible; "best spec we could find" is not.

### Worked example: Diagnose, don't demote on a real canonical spec

The "Diagnose, don't demote" protocol above is abstract; here's what it looks like in practice
on a real engagement's canonical spec (Schuh retail, post-Issue-#51). The spec is a 24-covariate
`mask_nov_jan` BSTS HMC weekly model on a 2-year pre-period, evaluated on a 4-week intervention
window.

**The four tests disagreed:**

| Test | Value | What it says |
|---|---|---|
| Model posterior predictive p | **0.005** | "Very strong" (subject to the BSTS calibration caveats above) |
| Rolling-placebo backtest | **rank 0.94, empirical p 0.06** | "Clean PASS" |
| Date-shuffled randomization (full range) | **p = 0.47** (Phipson-Smyth) | "Complete FAIL" |
| Specification grid (224 mask-mode specs) | **208/208 valid specs directional positive**, top 10 per mode cluster £260K–£295K with model p < 0.025 | "Robust across alternatives" |

Three tests pass, one fails. The temptation is to either spec-shop to a spec that passes all
four, or demote the date-shuffled test as "low power" without evidence. Both are shortcuts.
Instead, run the two cheap diagnostics (~$0.10 total):

**Diagnostic 1 — Training-length-matched permutation.** Re-run the date-shuffled test with fake
fit lengths constrained to match the real ~720-day training window. Result: **p improved from
0.47 to 0.29**. The training-length confound explains ~60% of the original p-gap, but the test
still doesn't pass 0.10 — so the canonical spec has *some* genuine residual difficulty with
random treatment dates, but the original 0.47 was substantially inflated by short-window
placebos. Partial artefact, partial residual.

**Diagnostic 2 — Mask-off rolling placebo.** Re-run the rolling backtest with `mask_mode=None`
to test the theoretical concern that a masked pre-period biases the rolling-placebo distribution
in a favourable direction. Result: **rank stayed at 0.94** (byte-identical to the masked
version). The theoretical mask-interaction bias is empirically zero in this data. The reviewer's
concern was worth testing — but the test refuted it.

**Bias-vs-variance decomposition of the date-shuffled placebos.** Binned by training-window
length, the fake-effect SD went from ~£180K at <1-year windows to ~£95K at 1.5-2-year windows,
while the mean stayed near zero in all bins. This is *low bias, high variance* — estimator
variance drops with training length, consistent with the training-length-matched diagnostic. No
level-shift signature.

**The resolution:** lead the client-facing story with the **224-spec grid robustness** — that's
the strongest single signal, and it's the only one that doesn't share the single-spec failure
modes. The rolling placebo is clean supporting evidence (PASS, and the mask-off diagnostic
rules out the mask-interaction story). The date-shuffled test is an honest limitation that the
training-length-matched diagnostic partially explains but doesn't fully dissolve. All three
appear in the methodology footnote.

**The anti-pattern would have been:** (a) spec-shop to a spec that passes date-shuffled, or
(b) drop the date-shuffled test from the methodology because "it's known to be low power". Both
hide information the client should see. The diagnostic protocol keeps all four tests in the
story, adds empirical context for the one that failed, and lets the spec grid carry the primary
inferential load.

### Diagnostic cost is tiny; run them before you rewrite the story

When a reviewer raises a theoretical concern about a methodology choice ("the mask introduces
bias", "the training length is confounding the null", "the placebo distribution is misspecified"),
the right response is **not** to rewrite the client-facing framing around the theoretical concern.
The right response is to **run the cheapest diagnostic that would directly test the claim**
before rewriting anything. Typical cost: ~50 fake-window fits = ~$0.10 on Cloud Run using the
existing placebo handler. Typical turnaround: 20 minutes.

Design rules for a diagnostic run:

1. **Pick the single strongest theoretical claim** raised by the reviewer. Don't run a sweep.
2. **Design the cheapest fake-scenario that would directly test it** — the null the diagnostic
   explores should differ from the real test by *exactly one* controlled factor.
3. **Reuse the existing `_run_sca_placebo_task` handler** with a custom `placebo_windows` list
   rather than adding a new Cloud Run task type (the `SCA_PLACEBO_MODE=1` handler accepts
   arbitrary in-time placebo designs as config — regular intervals, rolling-origin strides,
   training-length-matched shuffles, mask-off variants). No Docker rebuild required.
4. **Compare the diagnostic number against the real number in a single row of the same table**,
   so the reader can judge whether the theoretical concern actually manifests at the effect
   size you care about.

Theoretical concerns without empirical tests attached are speculation. Diagnostics turn them
into evidence — either they confirm the concern (and the methodology framing needs to change)
or they refute it (and you can tell the reviewer "we tested that specifically, here's the
number"). Cheap enough that there's no excuse not to.

### Common implementation gotchas for the placebo rank

The abstract protocols above assume you're computing the placebo rank correctly. Two
implementation gotchas that commonly cause incorrect numbers even when the methodology is sound:

**1. Signed vs absolute comparison.** The canonical placebo rank is:

```python
rank = sum(1 for p in placebos if p < real_effect) / (len(placebos) + 1)  # signed
```

Some pipelines accidentally compute:

```python
rank = sum(1 for p in placebos if abs(p) < abs(real_effect)) / (len(placebos) + 1)  # WRONG
```

The signed version counts placebos that are below the real effect in the natural ordering. The
absolute version counts placebos that are smaller *in magnitude* than the real effect — which
is a different quantity and can disagree materially when the post-fix placebo distribution is
near-zero symmetric (median close to zero, roughly balanced positive/negative tails). The two
conventions can move the rank by ~0.05–0.10. Pick **signed** and use it everywhere —
particularly when reporting the median:

- **Signed median**: `np.median(placebos)` — what `compute_placebo_rank` uses; e.g. £16K on a
  post-fix distribution
- **Absolute median**: `np.median(np.abs(placebos))` — a *different* summary of the same
  distribution; e.g. £101K on the same data

If you accidentally show one number in the rank computation and the other in the accompanying
median annotation, expert readers will spot the inconsistency and lose trust in the whole
methodology section. Match conventions across every number in the deliverable.

**2. Degenerate log-target specs on short masked pre-periods.** If your SCA grid mixes
`target=revenue` and `target=log1p(revenue)` variants, expect log-target specs on short masked
pre-periods (e.g. `mask_nov_jan` with a 2-year pre-period) to occasionally collapse: the
counterfactual fit becomes nearly identity on the masked training data, and the effect
estimate lands near zero (often ~£1). These are not "winners with tight CIs" — they're
degenerate specifications where the model couldn't find anything to fit.

**Filter before downstream placebo/backtest runs:**

```python
# In submission code, before writing the spec list to GCS
valid_specs = [s for s in specs if abs(s["effect"]) >= min_abs_eff]  # default £1000
```

A reasonable default is `min_abs_eff = £1000` (tuned to the revenue scale). Apply at the
submission boundary so the original SCA run remains complete for diagnostic purposes; document
the filtered-out specs with a one-line "degenerate log-target on masked pre-period" in the
methodology footnote rather than silently dropping them.

### When single tests disagree, lead with spec curve robustness

When the four tests give genuinely conflicting signals — model-p tight, rolling placebo PASS,
date-shuffled randomization FAIL, spec grid robust — **do not** promote any single test to
"primary" and the rest to "fallback". Single tests share single-spec failure modes: the
canonical model-p inherits BSTS's posterior-predictive conservatism (lesson above), the rolling
placebo inherits recent-pre-period assumptions, the date-shuffled test inherits exchangeability
and training-length assumptions.

**The spec grid doesn't share these failure modes**. A well-designed grid varies covariate
sets, mask modes, target transforms, training lengths, and model families independently. If the
effect survives 200+ alternative specifications with consistent sign and bounded magnitude
range, that's a harder argument to dismiss than any single p-value — it's robust to each
individual failure mode by construction.

**Framing for the client-facing document:**

> "The +£X effect is primarily supported by its robustness across [N] alternative
> specifications (the spec grid): [direction consistency], with the top [10] per mode
> clustering at [range] and model p < [cutoff]. It is further supported by a rolling-placebo
> backtest (rank [rank], empirical p [p]) on the canonical spec. Under a date-shuffled
> randomization test with training-window length matched to the real fit, the p-value is
> [diagnostic p] — which is a known edge case for 2-year pre-periods on volatile retail
> revenue (see methodology footnote)."

This framing:

1. Makes the spec grid the primary validation signal
2. Keeps the other tests as supporting evidence, not competing narratives
3. Discloses the one failing test with empirical context (the diagnostic number)
4. Doesn't hide anything or promote any single spec

**Caveat**: this framing only works when the spec grid is **large enough** (≥100 specs),
**methodologically diverse** (multiple covariate sets, multiple mask modes, multiple model
families), and **computed on clean data** (post-fix). Don't use it to paper over a genuinely
weak effect — a grid that mostly disagrees directionally or has enormous magnitude spread is
itself evidence against the effect, not for it.

### Language that won't survive contact with a client stats reviewer

Client-facing deliverables should avoid terms that suggest more precision than the methods
can deliver. Quick swap list:

| Don't say | Do say |
|---|---|
| "Model p-value of 0.005" | "Posterior probability of a positive effect: 99.5%" — or explicitly "Bayesian posterior predictive p-value" |
| "The effect is statistically significant (p < 0.05)" | "The posterior concentrates 99.5% of its mass on a positive effect" — or if you really want a p-value word, "one-sided Bayesian tail probability of 0.005" |
| "Passes permutation" | "Passes date-shuffled randomization at the 0.10 threshold" (+ state whether training length was matched) |
| "Spec curve analysis validates the effect" | "Specification grid robustness sweep (not a full Simonsohn 2020 SCA — joint inference not yet implemented)" |
| "48 of 50 placebos below the real" | "Rolling placebo empirical p-value ≈ 0.04" (give both; clients understand p-values) |
| "The canonical spec is permutation-fragile" | "The canonical spec fails the date-shuffled randomization test; the multi-method ensemble is the stronger summary" |
| "Lead with the best spec" | "We report a multi-method ensemble because a single point estimate understates uncertainty" |
| "Median placebo effect: £101K" | "Median placebo effect (signed): £16K. (Do NOT mix with £101K, which is the median of |placebo| — a different quantity.)" |
| "Spec curve analysis validates the effect" | "224-spec robustness grid: 208 valid specs, all directional positive, top 10 per mask mode £260K-£295K" (give the counts, not just the word "validates") |
| "The placebo test was low-powered so we dropped it" | "The placebo test was low-powered and the training-length-matched diagnostic shows [quantitative result]; we keep it in the methodology with empirical context rather than drop it" |

### When in doubt: the multi-method ensemble is the honest summary

The strongest defence against every failure mode in this section is the combination of two
robustness checks:

1. **The spec grid** (this is primary when it's available; see "When single tests disagree"
   above): robustness across 100+ defensible specifications that vary covariate sets, mask
   modes, target transforms, training lengths, and model families.
2. **The multi-method cross-check** — RDiT + BSTS (VI + HMC) + Prophet + (optionally) CausalPy,
   on at least two pre-period variants, plus a per-method placebo calibration check.

When both agree (spec grid robust AND methods converge directionally with bounded magnitude
spread), the conclusion is as robust as BSTS methodology currently allows. When the spec grid
is robust but the methods disagree, report the disagreement *as the dispersion* — "£160K-£355K
with a median around £290K" — rather than picking a favourite. When the spec grid itself
disagrees directionally, that's evidence against the effect, not for it.

The current skill's Step 4 already mandates dual-method analysis; extend this to **at least
one design-based method** (RDiT) when the model-based methods (BSTS, Prophet, CausalPy)
agree too tightly for comfort. Design-based methods fail differently from model-based methods,
and that's exactly what you want as a cross-check. And whenever possible, pair the multi-method
ensemble with a pre-registered spec grid — the two failure modes are largely orthogonal, and
their combination is harder to dismiss than either on its own.

## Reference Sections

> **Full details:** See [references/benchmarks_and_methods.md](references/benchmarks_and_methods.md)

Key reference material (covariate benchmarks, method selection tables, pre-period sensitivity,
leave-one-out analysis, Cloud Run batch setup) has been moved to reference files to keep the
main skill focused on the pipeline steps. Quick pointers:

- **Covariate benchmarks:** organic_sessions (r=0.86) is the safest control; paid_sessions (r=0.89) requires contamination check
- **Short campaigns:** RDiT is the lead method for <7 day campaigns; conformal CIs are 61% tighter than Bayesian
- **Pre-period sensitivity:** Jan 6 post-Christmas start is a reliable UK retail default (-27% CI width)
- **Masking vs truncating:** Mask BF-Jan 5 both years achieves -58% CI width; always verify temporal scope
- **Leave-one-out:** More covariates is not always better; test by dropping each one

## Reference: Environment & Dependency Gotchas

> **Full details:** See [references/environment_gotchas.md](references/environment_gotchas.md)

- **numpy conflict:** tfcausalimpact (numpy<2) and CausalPy (numpy>=2) cannot coexist. Run in separate scripts.
- **CausalPy macOS:** Requires `cores=1` in sample_kwargs (multiprocessing fork issue).
- **CausalPy short windows:** Fails with xarray `Dimension(s) 'draw', 'chain' do not exist` on <7 day treatment windows. Skip CausalPy and use RDiT as lead method.
- **Weather API:** Open-Meteo, free, no key. Use `curl -sk` for corporate SSL proxies.
- **Python version:** Always use `python3 -m pip install` to avoid version mismatch.

## References

> **Full bibliography:** See [references/bibliography.md](references/bibliography.md)

Key citations: Brodersen et al. (2015), Scott & Varian (2014), Abadie et al. (2010, 2015, 2021),
Eggers et al. (2024), Makowski et al. (2019), Gelman & Yao (2021), Gils et al. (2022),
Athey & Imbens (2017), Linden (2018), Peduzzi et al. (1996), Afyouni et al. (2019).

**Methodology Communication (added v2.3.0):** Gelman, Meng & Stern (1996), Robins, van der Vaart
& Ventura (2000), Bayarri & Berger (2000), Phipson & Smyth (2010), Fisher (1935),
Imbens & Rubin (2015), Simonsohn, Simmons & Nelson (2020), Steegen, Tuerlinckx, Gelman &
Vanpaemel (2016), Gelman & Loken (2013), Gelman & Carlin (2014), Blei, Kucukelbir & McAuliffe
(2017), Hyndman & Athanasopoulos (2021).
