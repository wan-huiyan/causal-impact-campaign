---
name: causal-impact-campaign
version: "1.5.0"
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
  or just seasonality?", "estimate the ROI of our marketing intervention", or "the p-value is 0.12 —
  did it work?". This skill covers the full pipeline: data exploration, covariate engineering,
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
idempotency: |
  Re-running the analysis with the same data and parameters produces the same estimates
  (within MCMC sampling variance). Set random_seed=42 for reproducibility.
namespace: causal_impact
composable_with:
  - permutation-validation: Validate model p-values with empirical permutation tests (REQUIRED before presenting results)
  - cloud-run-batch-experiment: Scale permutation tests and sensitivity analyses to GCP Cloud Run Jobs
  - client-proposal-slide: Pass findings to create stakeholder-ready presentation
  - frontend-design: Build custom interactive dashboards from analysis results
  - gcp-pipeline-cost-analysis: Estimate cost of running analysis at scale
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

> **Important:** If using tfcausalimpact with `nseasons=7`, the model already captures day-of-week
> seasonality internally. Adding explicit DoW covariates (sin/cos, dummies, or is_weekend) is
> redundant and can add noise. Only add DoW covariates when using CausalPy or other models
> without built-in seasonality.

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

### Placebo test

Compare the real intervention effect to the distribution of placebo effects:

```python
placebo_rank = (placebo_effects < real_effect).mean()  # if effect is positive
```

If the real effect ranks above the 90th percentile of placebos, it's genuinely unusual.
If it sits in the middle (50-70th percentile), the model can't distinguish it from noise.

### Scorecard

| Check | Threshold | What It Means |
|---|---|---|
| Median backtest WAPE | ≤ 15% | Model predictions are accurate |
| Mean 95% coverage | ≥ 80% | Uncertainty bands are well-calibrated |
| Placebo rank | ≥ 90th %ile | Real effect is unusual vs noise |
| All specs same direction | ≥ (N-1)/N | Result is robust to covariate choice |
| Primary p-value (BSTS) | ≤ 0.10 | Effect is statistically significant (model-based) |
| **Permutation p-value** | **≤ 0.10** | **Effect is unusual vs random dates (empirical)** |

> **The permutation test is now REQUIRED, not optional.** BSTS p-values can be overconfident
> when masking or tight model configurations (e.g., nseasons=14) are used. In a retail engagement,
> BSTS p=0.017 corresponded to permutation p=0.24 (11/49 random dates matched the real effect).
> The permutation p is the honest measure of whether the effect is genuinely distinguishable.
> Run 50+ random intervention dates with the same model config. Use 1-model-per-Cloud-Run-job
> for maximum parallelism (~10 min wall time for 50 permutations).

Passing all 5 = strong result. Passing 3+ with consistent direction = defensible.
Failing most = honest finding — report it as such.

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

After the primary analysis, run these extensions to deepen the insight:

### Effect Decomposition

Run separate CausalImpact on `conversion_rate`, `aov`, and `transactions` as targets. This reveals
**which lever the campaign pulled** — was it conversion, basket size, or traffic?

In one retail case study: conversion rate showed the strongest signal (~+14%, ~83% prob positive)
while AOV barely moved. This told us the delivery promo removed a conversion barrier — people
already browsing decided to buy because delivery was free. They didn't spend more per order.

This is often the most valuable insight for the client — it informs future offer design.

### Channel Split

Run CausalImpact on `paid_revenue` and `organic_revenue` separately to see if the campaign
affected all channels or just one. Use `organic_sessions` as control for both (don't use
`paid_sessions` as control for paid revenue — endogeneity risk).

If both channels lift proportionally, it's a site-wide conversion effect. If only paid lifts,
the campaign may be driving traffic rather than conversion.

### Post-Promo Persistence

Run CausalImpact with the full post-period (intervention start → data end) instead of just the
promo window. Compare average daily effect during promo vs after promo:

```
persistence_ratio = post_promo_avg_daily_effect / during_promo_avg_daily_effect
```

- Ratio > 50%: Significant persistence — report total impact including post-period
- Ratio 10-50%: Partial persistence — mention as additional upside
- Ratio < 10%: Effect dissipated — report promo-period only

**Warning:** Persistence analysis is unreliable for short campaigns. In a retail case study, persistence
ratios ranged from 55% to 188% across specifications (median ~97%). An extended post-period test (17 days)
showed implausibly large cumulative effects — clearly a model artefact, not genuine persistence. BSTS continues to
underpredict after the intervention ends because the structural break shifts the level.
Frame persistence as "inconclusive" unless multiple specs agree and the extended post-period is plausible.

### Weather Covariate

For retail/ecommerce clients, add daily temperature and precipitation as covariates.
Source: [Open-Meteo API](https://open-meteo.com/) — free, no API key needed.

```python
# Fetch via curl (bypass corporate SSL proxies) or requests
import requests
resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": 51.5074, "longitude": -0.1278,  # London
    "start_date": "2024-10-01", "end_date": "2026-03-15",
    "daily": "temperature_2m_mean,precipitation_sum",
    "timezone": "Europe/London",
})
```

Weather typically has low standalone correlation with revenue (r ≈ -0.05 to +0.03) but provides
an orthogonal exogenous signal that can tighten credible intervals by 2-5%. Worth including when
available, but not transformative.

**Why weather matters for retail:** Rain/cold drives online purchasing (people stay home). For
retail specifically, seasonal patterns (boots in autumn, sandals in spring) correlate with
temperature.

**SSL note:** Corporate proxies may block the Open-Meteo API. Use `curl -sk` to bypass, or
`requests.get(..., verify=False)`.

### Prophet Cross-Validation

For high-stakes claims, run Facebook Prophet as an independent cross-validation method. Prophet
uses additive decomposition (Fourier seasonality + changepoint trend) — a fundamentally different
model family from BSTS (structural time series + state space). Agreement on both direction AND
magnitude across model families is much more convincing than multiple specs within the same model.

```python
from prophet import Prophet
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
m.add_regressor("organic_sessions")  # add key exogenous regressors
m.fit(train_df)  # pre-period only, columns: ds, y, organic_sessions
forecast = m.predict(future_df)  # post-period dates + regressor values
# Effect = actual - yhat; CI from yhat_lower/yhat_upper
```

Prophet does not produce a frequentist p-value. Report as "CI excludes zero" if
`actual_sum - yhat_upper_sum > 0`. In one retail case study, Prophet showed a moderate positive effect with
CI excluding zero — consistent with BSTS and RDiT.

### Contaminated Exogenous Metrics

**Never use Google Trends brand search or sale detection flags as covariates in promo analysis.**
These metrics are endogenous — the promo itself drives search interest. Using them as covariates absorbs
part of the treatment effect, exactly like contaminated paid_sessions (lesson 2).

Validated with real Google Trends data: adding a brand search index as a covariate worsened p by ~3x
and dropped the effect estimate by ~30%. Similarly, `sale_type_flag` worsened p by ~2x when the
intervention IS a sale (the flag captures the campaign itself).

**Safe exogenous alternatives:**
- `trend_brand_share` (brand / brand+competitors) — relative metric, partially cancels campaign effect
- `trend_category` (e.g., "buy shoes online") — market-level demand, exogenous to specific brand
- `trend_competitor_N` — competitor search, exogenous to your campaign
- Weather (temp_avg, precipitation_mm) — always exogenous, validated improvement

**The contamination test:** did the metric change *because of* the promo? If yes, it's endogenous.

**Data provenance requirement:** NEVER use fabricated/synthetic data for experiments. Always fetch real
data (browser export for Google Trends, API for weather). Add a `.provenance.md` companion file.

### Sale Period Auto-Detection (Coupon Ratio)

If the data contains `transactions_with_coupon` alongside `transactions`, you can auto-detect sale
periods without client-supplied calendars. The coupon redemption ratio is a **bidirectional** signal:

- **Coupon-based sales** (ratio spikes UP to 0.30–0.42): Promo-code-driven sales (January clearance,
  spring sale, September sale). Customers redeem codes → ratio rises.
- **Site-wide sales** (ratio drops DOWN to 0.11–0.17 + volume surge): Blanket discounts (Black Friday,
  summer sale). Discounts are automatic → fewer code redemptions → ratio falls.

**Detection algorithm:**
```python
coupon_ratio = transactions_with_coupon / transactions
trailing_median = coupon_ratio.rolling(28).median()
trailing_mad = coupon_ratio.rolling(28).apply(lambda x: np.median(np.abs(x - np.median(x))))
z = (coupon_ratio - trailing_median) / (1.4826 * trailing_mad)

# Classification
sale_type = "coupon_sale"   if z > +2.5
sale_type = "sitewide_sale" if z < -2.5 AND transactions > P25(28-day)
sale_type = "normal"        otherwise
```

**Key design choices:**
- MAD over standard deviation: robust to the very outliers being detected
- 28-day trailing window: smooths weekly cycles, adapts to seasonal baseline shifts
- 1.4826 consistency constant normalises MAD to SD-equivalent for normal distributions
- Volume floor for sitewide detection prevents flagging quiet days with noisy ratios

**Integration:** The `sale_period_detection` enrichment in the webapp produces `sale_type_flag` (binary
covariate) plus `coupon_ratio_zscore` (continuous). Detected sale periods appear as orange (coupon) /
purple (sitewide) bands on the validate chart, with warnings when they overlap the treatment window.

**CONTAMINATION WARNING:** When the intervention being tested IS a sale/promotion, `sale_type_flag`
is contaminated — it absorbs part of the causal effect (validated: p worsened ~2x, effect dropped ~20%).
Auto-exclude `sale_type_flag` when it overlaps the intervention window.
Only use it as a covariate for non-sale interventions (e.g., website redesign, pricing change).

**Caution:** Thresholds (z=±2.5) were calibrated on retail data. For other retailers,
inspect the coupon ratio distribution before trusting default thresholds — if the retailer never uses
promo codes, the signal won't exist.

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

## Reference: Covariate Correlation Benchmarks

From a retail engagement (retail, daily revenue):

| Covariate | r with Revenue | Notes |
|---|---|---|
| paid_sessions | +0.885 | Strong but check intervention safety |
| organic_sessions | +0.863 | Usually the safest control |
| xmas_intensity | +0.828 | Multi-modal v2 (BF spike + gift shopping + Boxing Day) |
| kcp_period_flag | +0.523 | Key consumption period |
| payday_window_flag | +0.213 | 25th–3rd spending window |
| holiday_flag | +0.153 | Named public holidays |
| paid_share | +0.140 | Media intensity ratio |
| payday_x_weekend | +0.123 | Interaction term |
| sin_dow / cos_dow | -0.078 / +0.004 | Low standalone but captures weekly cycle in model |
| winter_sale_flag | -0.024 | Near-zero — binary flag is inadequate for retail peaks |

These are benchmarks, not universals — always compute correlations for the specific client.

## Reference: Method Selection for Short Campaigns

From a retail engagement — key lessons about which methods work for campaigns under 1 week:

| Method | Result | Key Insight |
|---|---|---|
| **BSTS (tfcausalimpact)** | +22%, p=0.21, not significant | Global time series model — daily variance drowns out short effects |
| **CausalPy (PyMC)** | Consistent, R²=0.72 | Confirms direction but same significance challenge |
| **RDiT** | **~+18%, CI excludes zero — significant** | Local boundary comparison avoids global variance problem |
| **Conformal CI** | Moderate positive effect, CI 61% tighter than Bayesian | Distribution-free — doesn't depend on model specification |

**Key strategic insight:** For short campaigns (< 7 days), **RDiT should be the lead method**, not BSTS.
BSTS is powerful for long interventions where the full time series structure matters, but for short
campaigns the global variance dominates. RDiT focuses only on the local discontinuity at the boundary,
sidestepping the noise problem entirely. Use BSTS as a supporting method for the full counterfactual
decomposition, and RDiT for the significance claim.

**Conformal intervals** should always be run alongside Bayesian CIs. They were 61% tighter in a retail
case — a dramatic improvement. Use the pre-period residual quantile approach: `np.quantile(np.abs(residuals), 0.95)`.

**Fourier seasonality (k=1..4):** Did NOT help with ~17 months of data (+0.9% CI width). Requires 2+ full
annual cycles to learn meaningful patterns. Don't add Fourier terms unless the pre-period spans 2+ years.

## Reference: Pre-period Start Date Sensitivity

From a retail engagement (retail):

| Start Date | Description | Days | CI Width Impact | p-value |
|---|---|---|---|---|
| Oct start | Full data (with Christmas) | ~500 | Baseline | ~0.22 |
| Jan 6 start | Post-Christmas (recommended) | ~420 | -27% | ~0.16 |
| Feb start | Post-winter-sale | ~390 | -30% | ~0.18 |
| Mar start | Spring onward | ~360 | -31% | ~0.16 |

The sweet spot is usually just after the major seasonal peak — enough data to learn patterns,
but excluding the period that dominates the variance. For UK retail, Jan 6 (post-Christmas
hangover) is a reliable default.

### Advanced: Masking high-variance periods instead of truncating

Truncating the pre-period loses data. An alternative: **mask out** the high-variance windows
while keeping the rest. This preserves the full annual cycle (spring-summer-autumn) while
removing the Christmas noise.

From a retail engagement:

| Approach | Days | Std Dev | CI Width | p-value | Prob+ |
|---|---|---|---|---|---|
| Full (no mask) | 514 | High | Baseline (wide) | ~0.24 | ~76% |
| Jan 6 start (truncate) | 417 | Medium | -29% | ~0.09 | ~92% |
| **Mask BF-Jan 5 both years** | **410** | **Low** | **-58%** | **~0.06** | **~94%** |
| **Mask Nov-Jan both years** | **330** | **Low** | **-64%** | **~0.05** | **~95%** |
| Very short start (too short) | 52 | Very low | -81% | ~0.00 | 100% |

Masking BF-Jan 5 keeps 410 days and drops std by ~65% — better than truncation
(which keeps 417 days but retains high std because it includes the second Christmas).

**Warning:** Very short pre-periods (< 60 days) produce overconfident results. If p≈0 and
CI is 3x tighter than other specs, the model is underestimating uncertainty — not finding a
stronger signal. The Jan 6 2026 spec (52 days) should always carry a caveat.

**Temporal scope verification (critical):** When masking or excluding seasonal periods, count
how many instances exist in your date range and verify ALL are handled. Common mistake: "Jan 6
pre-period start" was labeled "excludes Christmas" but only excluded Christmas 2024 — Christmas
2025 was still in the 14-month training window. Three rounds of adversarial review (12 reviewers)
missed this; the analyst caught it 6 days later. Always label specs precisely: "excludes Christmas
2024 only" vs "masks Christmas 2024 + 2025 (both years)."

```python
# Verify temporal scope: count instances before masking
import pandas as pd
for event_name, month_start, month_end in [('Christmas/winter', 11, 1)]:
    instances = []
    for year in df.index.year.unique():
        mask = (df.index >= f'{year}-{month_start:02d}-01') & (df.index <= f'{year+1}-{month_end:02d}-31')
        if mask.any():
            instances.append(year)
    print(f"{event_name}: {len(instances)} instances in data ({instances})")
    # Ensure your masking covers ALL instances, not just the first
```

**Implementation:** Drop masked dates from the DataFrame before passing to CausalImpact:
```python
for start, end in [('2024-11-01', '2025-01-31'), ('2025-11-01', '2026-01-31')]:
    df = df.loc[~((df.index >= start) & (df.index <= end))]
```

## Reference: Leave-One-Out Covariate Sensitivity

After finding the best covariate bundle, run leave-one-out to identify noise contributors:

```python
base_covs = ['organic_sessions', 'sin_dow', 'cos_dow', 'xmas_intensity', 'payday_window_flag']
for drop in base_covs:
    subset = [c for c in base_covs if c != drop]
    # Run CausalImpact with subset, record p-value and effect
```

From a retail engagement (5-covariate enhanced bundle, mask_nov_jan):

| Dropped | p-value | Effect | Verdict |
|---|---|---|---|
| None (full model) | 0.07 | Baseline | Baseline |
| cos_dow | **0.03** | +19% | **Noise -- model improves without it** |
| payday_window_flag | 0.04 | +46% | Marginal -- can be dropped |
| xmas_intensity | 0.055 | — | Helpful but not critical |
| sin_dow | 0.056 | — | Helpful but not critical |
| organic_sessions | **0.148** | — | **Critical — model collapses without it** |

**Key insight:** More covariates is not always better. cos_dow was adding noise because nseasons=7
already captures weekly seasonality internally (see Lesson 7). The pruned model (4 covariates)
outperformed the full model (5 covariates).

## Reference: Cloud Run for Batch BSTS Runs

Running 12+ BSTS specs locally can take 60+ min. Use Cloud Run Jobs for parallel execution:

1. **Containerize** the single-spec runner (tfcausalimpact + pandas + google-cloud-storage)
2. **Upload data** to GCS (features CSV + weather CSV)
3. **Launch all specs** with `--async` — each runs independently on 2 vCPU + 4GB
4. **Collect results** from GCS when all complete

Key settings:
- `--task-timeout=1800s` (NOT 600s — longer pre-periods take 15-20 min for 6 BSTS models)
- `--memory=4Gi --cpu=2` for tfcausalimpact
- Pass spec key via `SPEC_KEY` env var, data location via `GCS_BUCKET`

Typical cost: ~$0.50 for 12 parallel specs. Completes in ~15 min vs 60+ min sequential.

## Reference: Environment & Dependency Gotchas

### numpy version conflict (tfcausalimpact vs CausalPy)

These two packages **cannot coexist** in the same Python environment:

| Package | numpy | pandas | Notes |
|---|---|---|---|
| `tfcausalimpact` | < 2.0 | <= 2.2 | TensorFlow 2.16 needs numpy 1.x |
| `CausalPy` | >= 2.0 | >= 3.0 | PyMC/PyTensor needs numpy 2.x |

**Workflow:** Run tfcausalimpact first (numpy<2), then `pip install "numpy>=2"`, then run
CausalPy in a separate script. Never import both in the same process.

**CausalPy on macOS:** Requires `cores=1` in `sample_kwargs` — the default multiprocessing
fork causes `RuntimeError: An attempt has been made to start a new process before the current
process has finished its bootstrapping phase`. Fix:

```python
model=cp.pymc_models.LinearRegression(
    sample_kwargs={"random_seed": 42, "chains": 4, "draws": 2000, "tune": 1000, "cores": 1}
)
```

### Weather data: Open-Meteo API

Best free source for daily weather covariates (temperature, precipitation). No API key needed.

```bash
# Corporate SSL proxies may block Python requests — use curl -sk to bypass
curl -sk "https://archive-api.open-meteo.com/v1/archive?latitude=51.5074&longitude=-0.1278&start_date=2024-10-01&end_date=2026-03-15&daily=temperature_2m_mean,precipitation_sum&timezone=Europe/London"
```

For recent days not yet in the archive, backfill from the forecast API:
`https://api.open-meteo.com/v1/forecast` (same parameters).

### Python version mismatch

`pip3 install` may install to a different Python version's site-packages. Always use:
```bash
python3 -m pip install <package>  # installs to the correct python3's site-packages
```
Verify with `python3 -m pip show <package> | grep Location`.
