---
name: causal-impact-campaign
version: "1.3.0"
description: |
  Measure the causal impact of a marketing campaign, promo, or intervention on a business metric
  (revenue, conversions, transactions) using Bayesian structural time series. Use this skill whenever
  the user mentions "causal impact", "campaign uplift", "promo effect", "incrementality", "did the
  campaign work", "revenue lift from campaign", "measure uplift", "what was the true effect",
  "counterfactual analysis", "quasi-experiment", or wants to attribute a metric change to a specific
  intervention using time series data. Also trigger when working with GA4/BigQuery data and the user
  asks about measuring the effect of a price change, delivery promo, delivery promotion, ad campaign,
  or any time-bounded business action. Trigger on questions like "did the promotion actually increase
  revenue?", "how much additional revenue did the campaign generate?", "is the revenue change from the
  campaign or just seasonality?", "estimate the ROI of our marketing intervention", "the p-value is
  0.12 — did it work?", "can we still measure the impact?", "should we use synthetic control?",
  "geo-targeted campaign", "two promotions running at the same time", "separate their effects",
  "weekly aggregated data", "measure campaign effect", "changing our delivery policy affected order
  volume", "attribute the conversion spike to the price reduction", "cross-validate causal results",
  "influencer campaign ROI". This skill covers the full pipeline: data exploration, covariate
  engineering, dual-method analysis (tfcausalimpact + CausalPy), validation, interpretation, and
  client-facing deliverables including interactive HTML explorers with Plotly.js. Even if the user
  only mentions one method, use this skill to ensure robustness through cross-method comparison.
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
consumes_from:
  - gcp-pipeline-cost-analysis: Daily time series data from BigQuery
  - data-extraction: Raw GA4/analytics data export
hands_off_to:
  - client-proposal-slide: Pass findings document and key metrics
  - frontend-design: Pass data structure and section layout for interactive HTML
output_contract: |
  - findings.md: Markdown findings document with executive summary, results, validation, recommendations
  - counterfactual_data.csv: Daily observed vs predicted with CI bounds
  - sensitivity_summary.csv: Per-specification effect estimates, p-values, CIs
  - interactive_explorer.html: Self-contained Plotly.js explorer (optional)
composable_with:
  - client-proposal-slide: Pass findings to create stakeholder-ready presentation
  - frontend-design: Build custom interactive dashboards from analysis results
  - gcp-pipeline-cost-analysis: Estimate cost of running analysis at scale
---

# Causal Impact Campaign Analysis

Measures the causal effect of a marketing campaign or business intervention on a target metric
using dual Bayesian methods (tfcausalimpact + CausalPy). Encodes lessons from real client
engagements around short-lived campaigns, retail seasonality, and honest statistical communication.

**Requirements:** Compatible with Python v3.9 through v3.12, pandas v1.5+ and v2.x.
Set `random_seed=42` for reproducibility. The analysis is idempotent -- safe to re-run with
the same data and parameters. All output artifacts scoped to a timestamped `causal_impact` directory.

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

### Traffic vs conversion diagnostic

Compare intervention period to the prior week to determine covariate safety:

| Sessions changed? | Revenue changed? | Interpretation | Sessions as covariate? |
|---|---|---|---|
| No | Yes | Promo lifted conversion/AOV | SAFE |
| Yes | Yes | Promo drove traffic + conversion | CAUTION (absorbs signal) |
| Yes | No | Traffic spike, no conversion lift | SAFE but promo may have failed |

```python
promo = df[(df['date'] >= INTERVENTION_START) & (df['date'] <= INTERVENTION_END)]
pre_week = df[(df['date'] >= INTERVENTION_START - pd.Timedelta(days=7)) & (df['date'] < INTERVENTION_START)]
print(f"Pre-week: revenue={pre_week['revenue'].mean():.0f}, sessions={pre_week['sessions'].mean():.0f}")
print(f"Promo:    revenue={promo['revenue'].mean():.0f}, sessions={promo['sessions'].mean():.0f}")
```

## Step 3: Engineer Covariates

Better covariates = tighter counterfactual = narrower credible intervals = better detection.

### The Covariate Safety Rule

**A covariate must NOT be affected by the intervention.** If the campaign drove paid traffic,
`paid_sessions` absorbs the effect and biases the estimate toward zero.

| Covariate type | Safety | Examples |
|---|---|---|
| Organic demand | Usually SAFE | `organic_sessions` |
| Calendar | Always SAFE | day-of-week, holidays, payday cycles |
| Exogenous | Always SAFE | weather (temperature, precipitation) |
| Paid traffic | TEST BOTH | `paid_sessions` -- remove if effect estimate increases >20% |

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
Binary flags treat all sale days equally, but Black Friday (5x normal revenue) creates massive
unexplained residuals that inflate ALL credible intervals. Use continuous intensity curves:

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

Typical correlations: multi-modal r=0.828 vs single Gaussian r=0.632 vs binary flag r=0.02.

#### 4. Interaction terms
Combine factors that amplify each other:
```python
df['payday_x_weekend'] = df['payday_window_flag'] * df['is_weekend']
```

#### 5. Paid share as media intensity signal
```python
df['paid_share'] = df['paid_sessions'] / (df['paid_sessions'] + df['organic_sessions'])
```
Low standalone r (~0.14) but useful in combination. Negative coefficient implies organic converts better.

### Recommended covariate audit

> **Combined audit:** Each covariate gets a single recommendation based on both its predictive
> value (correlation with revenue) and its intervention safety (did it change during the promo?).
> `YES` = include. `CAUTION` = test with and without. `SKIP` = don't include.

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

Run multiple methods -- agreement across methods is far more convincing than any single p-value.

### Method selection

| Method | What it does | When to use | Key limitation |
|---|---|---|---|
| **BSTS (tfcausalimpact)** | Decomposes time series into trend + seasonality + regression, projects counterfactual | Always — the primary analysis. Full decomposition with covariates | Struggles with short campaigns (<7 days) — daily variance overwhelms signal |
| **CausalPy (LinearRegression)** | Bayesian regression with exact MCMC inference (NUTS) | Always — robustness check with different inference engine | No time series structure (no trend/seasonal components) |
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

### Dependency Conflict (critical)

tfcausalimpact (`numpy<2`) and CausalPy (`numpy>=2`) cannot coexist. Run in separate scripts:
tfcausalimpact first, then `pip install "numpy>=2"`, then CausalPy.

### Method 1: tfcausalimpact

```python
from causalimpact import CausalImpact

MODEL_ARGS = {
    "nseasons": 7,            # day-of-week seasonality
    "standardize_data": True,  # z-score normalisation
    "fit_method": "vi",        # variational inference (fast)
}

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

Start with `LinearRegression`. If pre-period R² < 0.65, try `StateSpaceTimeSeries`.
`WeightedSumFitter` requires donor units (untreated sites/regions) -- not for single-unit ITS.
CausalPy 0.8+: `ScikitLearnAdaptor()` takes no arguments.

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

Slide a fake intervention window across the pre-period. Fit on everything before, predict the window:

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

Real effect > 90th percentile of placebos = genuinely unusual. 50-70th = indistinguishable from noise.

### Scorecard

| Check | Threshold | What It Means |
|---|---|---|
| Median backtest WAPE | ≤ 15% | Model predictions are accurate |
| Mean 95% coverage | ≥ 80% | Uncertainty bands are well-calibrated |
| Placebo rank | ≥ 90th %ile | Real effect is unusual vs noise |
| All specs same direction | ≥ (N-1)/N | Result is robust to covariate choice |
| Primary p-value | ≤ 0.10 | Effect is statistically significant |

Passing all 5 = strong result. Passing 3+ with consistent direction = defensible.
Failing most = honest finding — report it as such.

**Watch for Christmas/BF backtest windows** -- extreme WAPE and massive placebo effects inflate
the distribution. Note in findings; this is a feature of high-variance retail data, not model failure.

## Step 6: Interpret Honestly

### Pre-period length

More data is NOT always better. Longer pre-periods hurt when:
- Tracking changed (structural break) or sessions-revenue relationship shifted
- COVID/post-COVID regime changes apply
- Campaign calendar is incomplete (flags like `winter_sale_flag` are useless if inaccurate)

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

Run separate CausalImpact on `conversion_rate`, `aov`, and `transactions` to reveal **which lever
the campaign pulled**. This is often the most valuable insight -- it informs future offer design.

Example: Schuh delivery promo lifted conversion rate (+14.3%) while AOV barely moved (+0.7%),
showing the promo removed a conversion barrier, not a spending barrier.

### Channel Split

Run CausalImpact on `paid_revenue` and `organic_revenue` separately. Use `organic_sessions` as
control for both (never `paid_sessions` for paid revenue -- endogeneity). Both lift = site-wide
conversion effect. Only paid lifts = traffic-driving campaign.

### Post-Promo Persistence

Run CausalImpact with the full post-period (intervention start → data end) instead of just the
promo window. Compare average daily effect during promo vs after promo:

```
persistence_ratio = post_promo_avg_daily_effect / during_promo_avg_daily_effect
```

- Ratio > 50%: Significant persistence — report total impact including post-period
- Ratio 10-50%: Partial persistence — mention as additional upside
- Ratio < 10%: Effect dissipated — report promo-period only

In the Schuh case: 66% persistence over 2 weeks, making the total impact considerably larger than
the headline promo-period figure.

### Weather Covariate

Add daily temperature and precipitation from [Open-Meteo API](https://open-meteo.com/) (free, no key).
Low standalone r (-0.05 to +0.03) but tightens CIs by 2-5% as an orthogonal exogenous signal.

```python
import requests
resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": 51.5074, "longitude": -0.1278,
    "start_date": "2024-10-01", "end_date": "2026-03-15",
    "daily": "temperature_2m_mean,precipitation_sum",
    "timezone": "Europe/London",
})
```

**SSL note:** Corporate proxies may block this. Use `curl -sk` or `verify=False`.

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

### Format options

| Format | Audience | Notes |
|---|---|---|
| **Slide Deck** | Client meetings | 10 HTML slides, fade+scale transitions, SVG charts. Use `frontend-design` skill. |
| **Scrolling Report** | Async stakeholders | Single-page HTML, scroll-triggered animations |
| **Jupyter Notebook** | Internal DS team | Full code, diagnostics, correlation heatmaps |

### Interactive Explorer (recommended for client self-service)

A single self-contained HTML file with Plotly.js charts and pre-computed scenario data embedded
as JSON. The client opens it in their browser — no server, no install, works offline after first
load. This is the **highest-impact deliverable** because it lets the client explore robustness
themselves rather than trusting a static summary.

**Architecture:** Single HTML file with CSS, Plotly.js (CDN), embedded `const DATA = {...}` JSON,
and JS event handlers. Five sections: Hero+Spec Selector, Counterfactual Chart, Method Comparison,
Effect Decomposition, Channel & Persistence. Plus a validation scorecard.

**Data structure:** Embed `specs` (per model specification: effect, pval, CIs), `methods` (per
analysis method: effect, CIs, description), and `timeseries` (daily dates, observed, predicted,
CI bounds). Extract from `ci.inferences` during Step 4; fallback: synthetic counterfactual.

**Key notes:**
- Data is tiny (<100KB JSON). Plotly.js CDN (~3.5MB) cached after first load.
- Spec selector updates all metrics simultaneously with fade animation -- gives the client an intuitive
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
| Markdown findings | Internal (Elena-style) | Quick sharing, PR review, documentation |

## The Meta-Lesson: Subtract Before You Add

When p > 0.10, the biggest improvements come from **removing** things, not adding:

| Action | Type | Typical impact |
|---|---|---|
| Exclude high-variance periods from pre-period | Subtraction | -27% CI width |
| Remove contaminated covariates (changed during intervention) | Subtraction | +36% effect estimate, p halved |
| Remove covariates redundant with built-in model components | Subtraction | Cleaner model, less noise |
| Add better covariates (multi-modal holiday intensity) | Addition | -11% CI width |
| Add exogenous signals (weather) | Addition | -3% CI width |

**Practical workflow when p > 0.10:**
1. Shorten pre-period (exclude high-variance events like Christmas)
2. Run covariate audit -- remove anything that changed >10% during intervention
3. Check for redundancy with built-in model components (e.g., nseasons)
4. THEN add better covariates (intensity curves, weather, interactions)

**Critical caveat:** Document every experiment. If only the "best" spec achieves significance
out of N tested, the result is exploratory. Lead with the Bayesian posterior probability across
ALL specs, not the cherry-picked p-value. See the Claim Framing Guide below.

## Claim Framing Guide

**The client doesn't need p < 0.05. They need to know whether to run the promo again.**

### What to lead with (in all deliverables)

```
"There is an 80-96% probability the promo generated positive incremental revenue,
estimated £190-250K across all specifications. All methods agree on direction.
The promo drove conversion (+14%), not traffic."
```

### What NOT to lead with

```
"The promo generated £297K, statistically significant at p < 0.05."
(Unless this was the pre-registered primary specification)
```

### Scorecard framing

- **All deliverables must show the same primary p-value** -- inconsistency destroys trust.
- Primary (pre-registered) p-value is the main result; optimised specs labeled "Exploratory."
- Convergence of all specs on positive direction IS the strongest evidence -- foreground it.

### The specification search trap

| What you did | How to frame it |
|---|---|
| Pre-registered one spec, it was significant | Lead with it. This is confirmatory. |
| Tested N specs, all significant | Lead with the range. This is robust. |
| Tested N specs, best one is significant | **Lead with the range. Best spec is exploratory.** |
| Tested N specs, none significant | Lead with Bayesian posterior and direction consistency. |

### Synthetic data in charts

Never use `Math.random()` for client-facing charts without an "illustrative" label.
Always embed real model predictions from `ci.inferences`.

## Key Pitfalls

| # | Pitfall | Fix |
|---|---|---|
| 1 | Covariates affected by intervention | Run covariate audit; test with/without |
| 2 | Binary flags for high-variance events | Use continuous intensity variables |
| 3 | Claiming significance when p > 0.10 | Be honest about uncertainty |
| 3b | Claiming significance from spec search | Bonferroni correction; lead with posterior across all specs |
| 4 | Too-long pre-periods with structural breaks | Shorten; test multiple start dates |
| 5 | Ignoring Christmas/BF distortion | Model explicitly with intensity curves |
| 6 | Running only one method | Always run dual methods (tfcausalimpact + CausalPy) |
| 7 | Dependency conflict | Run tfcausalimpact and CausalPy in separate scripts |
| 8 | Including BF/Christmas in pre-period | Exclude; start after Jan 6 (can reduce CIs by 27%) |

## Reference: Covariate Correlation Benchmarks

From the Schuh engagement (UK footwear retail, daily revenue):

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

From the Schuh engagement — key lessons about which methods work for campaigns under 1 week:

| Method | Result | Key Insight |
|---|---|---|
| **BSTS (tfcausalimpact)** | +22%, p=0.21, not significant | Global time series model — daily variance drowns out short effects |
| **CausalPy (PyMC)** | Consistent, R²=0.72 | Confirms direction but same significance challenge |
| **RDiT** | **+18.3%, CI [£8K, £71K] — significant** | Local boundary comparison avoids global variance problem |
| **Conformal CI** | £154K, CI 61% tighter than Bayesian | Distribution-free — doesn't depend on model specification |

**Key insights:**
- Short campaigns (< 7 days): RDiT leads (local boundary), BSTS supports (full decomposition)
- Always run conformal CIs alongside Bayesian (were 61% tighter in Schuh case)
- Fourier seasonality (k=1..4): requires 2+ years of data; skip with shorter pre-periods

## Reference: Pre-period Start Date Sensitivity

From the Schuh engagement (UK footwear retail):

| Start Date | Description | Days | CI Width Impact | p-value |
|---|---|---|---|---|
| Oct 2024 | Full data (with Christmas) | 514 | Baseline | 0.215 |
| Jan 6 2025 | Post-Christmas (recommended) | 417 | -27% | 0.163 |
| Feb 2025 | Post-winter-sale | 391 | -30% | 0.183 |
| Mar 2025 | Spring onward | 363 | -31% | 0.161 |

Sweet spot: just after the major seasonal peak. For UK retail, Jan 6 (post-Christmas) is a reliable default.

## Reference: Environment Gotchas

| Issue | Fix |
|---|---|
| numpy conflict (tfcausalimpact `<2` vs CausalPy `>=2`) | Run in separate scripts; tfcausalimpact first, then upgrade numpy |
| CausalPy macOS multiprocessing crash | Set `cores=1` in `sample_kwargs` |
| Corporate SSL blocking Open-Meteo | Use `curl -sk` or `verify=False` |
| pip installs to wrong Python | Use `python3 -m pip install <package>` |
| Recent weather not in archive | Backfill from `api.open-meteo.com/v1/forecast` |
