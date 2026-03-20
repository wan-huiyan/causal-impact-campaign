---
name: causal-impact-campaign
description: |
  Measure the causal impact of a campaign, policy change, or intervention on a target metric
  (revenue, accidents, conversions, transactions) using Bayesian structural time series. Use this
  skill whenever the user mentions "causal impact", "campaign uplift", "policy effect",
  "incrementality", "did the intervention work", "metric lift from campaign", or wants to attribute
  a metric change to a specific intervention using time series data. Also trigger when working with
  time series data and the user asks about measuring the effect of a policy change, campaign, ad
  spend shift, or any time-bounded action. This skill covers the full pipeline: data exploration,
  covariate engineering, dual-method analysis (tfcausalimpact + CausalPy), validation,
  interpretation, and deliverables. Even if the user only mentions one method, use this skill to
  ensure robustness through cross-method comparison.
---

# Causal Impact Campaign Analysis

This skill guides you through measuring the causal effect of a campaign, policy change, or
intervention on a target metric using Bayesian structural time series methods. It encodes
hard-won lessons from real analyses — particularly around short-lived interventions,
seasonal distortions, and honest statistical communication.

The approach runs two independent Bayesian methods (tfcausalimpact and CausalPy) for robustness,
includes a rigorous validation suite, and produces a client-ready findings document.

## When This Applies

- Measuring metric uplift/reduction from a campaign, policy change, or intervention
- The user has time series data (typically daily) with a clear intervention date
- Data comes from BigQuery, a database, CSV, or similar structured source
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
| What was the intervention? | Determines which metrics to target and which covariates are safe |
| When did it start and end? | Defines analysis windows |
| What was the scope? | Determines which covariates are safe as controls |
| Were other interventions running? | Concurrent interventions confound the analysis |
| Was it region-wide or targeted? | Targeting enables stronger designs (synthetic control, geo lift) |
| How long was it? | Interventions < 1 week are very hard to detect statistically |

**Critical early assessment — signal-to-noise ratio:**
```
SNR = expected_daily_effect / daily_metric_std_dev
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

1. **Metric by day of week** — is there a strong weekly cycle?
2. **Component splits** — can you decompose the metric into meaningful sub-components?
3. **Holiday/seasonal patterns** — how extreme are seasonal spikes or troughs?
4. **Around the intervention** — what changed? Which sub-metrics moved?
5. **Correlation matrix** — which covariates correlate with the target?

### The "what changed" diagnostic

This is critical for covariate safety. Compare the intervention period to the week before:

```python
# If covariates barely changed but the target metric shifted → intervention drove the change
# This means those covariates are SAFE controls
post = df[(df['date'] >= INTERVENTION_START) & (df['date'] <= INTERVENTION_END)]
pre_week = df[(df['date'] >= INTERVENTION_START - pd.Timedelta(days=7)) & (df['date'] < INTERVENTION_START)]
print(f"Pre-week: target={pre_week[TARGET].mean():.1f}, covariate={pre_week[COVARIATE].mean():.1f}")
print(f"Post:     target={post[TARGET].mean():.1f}, covariate={post[COVARIATE].mean():.1f}")
```

## Step 3: Engineer Covariates

This is where most of the analytical value is added. Better covariates = tighter counterfactual
= narrower credible intervals = better chance of detecting the effect.

### The Covariate Safety Rule

**A covariate must NOT be affected by the intervention.** If the policy change also triggered
road construction changes, `construction_zones` absorbs the effect and your estimate will be biased toward zero.

Safe controls for most intervention types:
- Exogenous variables — weather, daylight hours, calendar effects
- Calendar variables — day-of-week, holidays, school terms
- Unaffected infrastructure metrics (if available)

> **Always test with and without suspect covariates.** If removing a covariate increases the
> effect estimate substantially (>20%), it's likely absorbing the causal signal. In the example
> public safety case, removing `construction_zones` increased the effect by 8 accidents/day (+36%)
> and improved p-value from 0.142 to 0.060 — achieving BSTS significance for the first time.

### Covariate Engineering Recipes

#### 1. Split aggregate metrics
Don't use a single aggregate covariate when sub-components are available. For example, split
`total_traffic_volume` into `residential_traffic` and `arterial_traffic`:
- Both are more informative than the aggregate
- Allows you to check if the intervention affected one sub-component but not the other

#### 2. Cyclical day-of-week encoding
Binary `is_weekend` misses that Sunday accident counts differ from Saturday, and Tuesday is the
weakest day. Use sin/cos encoding to capture the full weekly cycle:

```python
df['sin_dow'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
df['cos_dow'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)
```

> **Important:** If using tfcausalimpact with `nseasons=7`, the model already captures day-of-week
> seasonality internally. Adding explicit DoW covariates (sin/cos, dummies, or is_weekend) is
> redundant and can add noise. Only add DoW covariates when using CausalPy or other models
> without built-in seasonality.

#### 3. Seasonal intensity (not binary flags)
This is one of the most important lessons. Binary flags like `school_holiday_flag` treat all
holiday days equally, but half-term (elevated accidents due to children near roads) differs
from summer holidays (reduced commuter traffic). The model sees unexplained residuals, which
inflate variance estimates and widen ALL credible intervals — including for your intervention period.

Use a continuous intensity curve that captures the shape of seasonal disruption:

```python
def school_holiday_intensity(date_series):
    """Multi-modal school holiday intensity for traffic analysis. Components:
    1. Half-term spikes (Feb, May, Oct)
    2. Summer holiday plateau (Jul-Aug)
    3. Christmas/New Year break
    4. Easter break
    5. Bank holiday weekends
    6. End-of-term transition days
    Correlation with accident count: r=0.791 (vs r=0.432 for single binary flag)
    """
    result = pd.Series(0.0, index=date_series.index)
    for idx in date_series.index:
        d = date_series.loc[idx]
        intensity = 0.0
        # Summer holiday plateau (late Jul - Aug)
        if d.month == 8 or (d.month == 7 and d.day >= 20):
            intensity += 0.8
        # Half-term spikes (approximate)
        if d.month == 2 and 10 <= d.day <= 21: intensity += 0.6      # Feb half-term
        if d.month == 5 and 24 <= d.day <= 31: intensity += 0.6      # May half-term
        if d.month == 10 and 21 <= d.day <= 31: intensity += 0.6     # Oct half-term
        # Christmas break
        if d.month == 12 and d.day >= 20: intensity += 0.7
        if d.month == 1 and d.day <= 6: intensity += 0.5
        # Easter (approximate)
        if d.month == 4 and 1 <= d.day <= 18: intensity += 0.5
        result.loc[idx] = min(intensity, 1.0)
    return result
```

This variable typically achieves r=0.791 with accident counts for traffic analysis (vs r=0.432
for a single binary flag and r~0.02 for a simple `is_holiday` indicator).

#### 4. Interaction terms
Combine factors that amplify each other:
```python
df['payday_x_weekend'] = df['payday_window_flag'] * df['is_weekend']
```

#### 5. Derived ratio signals
```python
df['rain_intensity'] = df['precipitation_mm'] / (df['precipitation_mm'].rolling(30).mean() + 0.01)
```
Low standalone correlation (~0.14) but can be useful in combination. A ratio captures whether
today's conditions are unusual relative to recent history.

### Recommended covariate audit

> **Combined audit:** Each covariate gets a single recommendation based on both its predictive
> value (correlation with target metric) and its intervention safety (did it change during the intervention?).
> `YES` = include. `CAUTION` = test with and without. `SKIP` = don't include.

```python
# Combined covariate audit: checks BOTH correlation AND intervention safety
pre_all = df[df['date'] < INTERVENTION_START]
pre_week = df[(df['date'] >= INTERVENTION_START - pd.Timedelta(days=7)) & (df['date'] < INTERVENTION_START)]
post = df[(df['date'] >= INTERVENTION_START) & (df['date'] <= INTERVENTION_END)]

CALENDAR_COVS = ['bank_holiday_flag', 'school_holiday_intensity', 'is_weekend',
                 'weekend_x_rainfall']
EXOGENOUS_COVS = ['temperature_avg', 'precipitation_mm', 'daylight_hours']

for col in COVARIATES:
    corr = pre_all[col].corr(pre_all[TARGET])
    pre_val = pre_week[col].mean()
    post_val = post[col].mean()
    change = (post_val / pre_val - 1) if pre_val != 0 else float('nan')

    if col in CALENDAR_COVS:
        safety = "SAFE (calendar)"
    elif col in EXOGENOUS_COVS:
        safety = "SAFE (exogenous)"
    elif abs(change) < 0.10:
        safety = "SAFE (<10% change)"
    else:
        safety = f"INVESTIGATE ({change:+.0%} change)"

    include = "YES" if corr > 0.05 and "SAFE" in safety else "CAUTION" if "INVESTIGATE" in safety else "SKIP"

    print(f"{col:30s}  r={corr:+.3f}  post_change={change:+.1%}  {safety:25s}  → {include}")
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
| **CausalPy (LinearRegression)** | Bayesian regression with exact MCMC inference (NUTS) | Always — robustness check with different inference engine | No time series structure (no trend/seasonal components) |
| **RDiT** | Local linear regression at the intervention boundary, bootstrap CIs | **Especially for short interventions.** Only method that achieved significance for short policy changes | Ignores data far from cutoff; sensitive to bandwidth; no decomposition |
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
    formula="accident_count ~ 1 + t + rainfall_mm + daylight_hours + sin_dow + cos_dow + school_holiday_intensity + temperature_avg + bank_holiday_flag",
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
    "weather_only": ["rainfall_mm", "daylight_hours"],
    "weather_plus_calendar": ["rainfall_mm", "daylight_hours", "sin_dow", "cos_dow", "bank_holiday_flag"],
    "weather_plus_school": ["rainfall_mm", "daylight_hours", "sin_dow", "cos_dow", "school_holiday_intensity"],
    "weather_plus_construction": ["rainfall_mm", "daylight_hours", "construction_zones", "sin_dow", "cos_dow"],
}
```

If all specs agree on direction, that's strong evidence even if individual p-values are above
conventional thresholds.

## Step 5: Validate

### Rolling backtests

Slide a fake intervention window across the pre-period. At each position, fit the model on
everything before and predict the window. Compare predictions to actuals:

```python
HORIZON = intervention_days  # same length as real intervention
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
| Primary p-value | ≤ 0.10 | Effect is statistically significant |

Passing all 5 = strong result. Passing 3+ with consistent direction = defensible.
Failing most = honest finding — report it as such.

### Watch for problematic backtest windows

School holiday windows often show extreme WAPE (0% or 40%+) and massive placebo effects.
These inflate the placebo distribution, making the real effect look less unusual. Note this
in the findings — it's a feature of high-variance seasonal data, not a model failure.

## Step 6: Interpret Honestly

### The pre-period length question

More data is NOT always better. Longer pre-periods can hurt if:
- The website/tracking changed (structural break)
- The covariate→metric relationship shifted (non-stationarity)
- COVID/post-COVID regime changes apply
- The data provider can't supply accurate intervention flags further back

The binding constraint is often **intervention calendar completeness** — the team may not have
reliable records of which policy changes occurred 2+ years ago. Flags like `school_holiday_flag`
are only useful if accurate.

### Framing for clients

**When significant (p < 0.10):**
> "The intervention produced an estimated X unit change in the target metric (95% CI: Y to Z),
> with a W% probability this effect is genuine."

**When not significant but consistent:**
> "Our best estimate is that the intervention produced approximately X unit change in the
> target metric (+/−Y%). This finding is consistent across all model specifications and two
> independent analytical methods. The probability the effect is genuine is approximately Z%.
> Due to the short intervention duration (N days), the statistical confidence interval is wide —
> we recommend longer intervention periods for more precise measurement."

**Never claim** statistical significance when you don't have it. Credibility with clients
comes from honesty, not from overselling.

### Recommendations to always include

1. **Longer intervention periods** dramatically improve detectability (2+ weeks ideal)
2. **Control regions** provide cleaner control groups for future measurement
3. **Coordinate intervention calendars** to avoid overlapping policy changes
4. **Double down on the lever identified** — if the speed limit reduced accidents, test other traffic calming measures

## Step 6b: Extension Analyses

After the primary analysis, run these extensions to deepen the insight:

### Effect Decomposition

Run separate CausalImpact on `serious_accidents`, `minor_accidents`, and `pedestrian_incidents` as targets.
This reveals **which lever the intervention pulled** — was it severity, frequency, or a specific road user group?

In the example public safety case: serious accidents showed the strongest signal (−31%, p=0.041, 96% prob)
while minor incidents barely moved (−4%). This told us the speed limit reduction primarily prevented
high-severity collisions — lower speeds meant survivable impacts rather than fewer total incidents.

This is often the most valuable insight for stakeholders — it informs future policy design.

### Zone Split

Run CausalImpact on `residential_accidents` and `arterial_accidents` separately to see if the
intervention affected all road types or just targeted zones. Use `weather` covariates as controls
for both (don't use `traffic_volume` as control for residential roads — endogeneity risk).

If both zones improve proportionally, it's a city-wide behavioural effect. If only residential
zones improve, the policy is working as targeted.

### Post-Intervention Persistence

Run CausalImpact with the full post-period (intervention start → data end) instead of just the
intervention window. Compare average daily effect during intervention vs after:

```
persistence_ratio = post_intervention_avg_daily_effect / during_intervention_avg_daily_effect
```

- Ratio > 50%: Significant persistence — report total impact including post-period
- Ratio 10-50%: Partial persistence — mention as additional benefit
- Ratio < 10%: Effect dissipated — report intervention-period only

In the example public safety case: 72% persistence over 2 weeks, suggesting driver behaviour adapted
and the safety benefit persisted beyond the initial enforcement period.

### Weather Covariate

For traffic safety and outdoor activity analyses, add daily temperature and precipitation as covariates.
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

Weather typically has moderate correlation with accident counts (r ~ +0.15 to +0.30 for precipitation)
and provides an orthogonal exogenous signal that can tighten credible intervals by 5-12%. Worth
including when available, and often more impactful than in other domains.

**Why weather matters for traffic safety:** Rain/ice increases accident rates directly (reduced
visibility, longer stopping distances). Temperature extremes affect road surface conditions.
Daylight hours affect visibility — shorter days correlate with higher accident rates.

**SSL note:** Corporate proxies may block the Open-Meteo API. Use `curl -sk` to bypass, or
`requests.get(..., verify=False)`.

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

### When to use which

| Format | Audience | Best for |
|---|---|---|
| Slide deck | Client meeting | Live presentation, screen sharing |
| Scrolling report | Client async | Email attachment, self-service reading |
| Jupyter notebook | Internal DS team | Verification, iteration, collaboration |
| Markdown findings | Internal (Elena-style) | Quick sharing, PR review, documentation |

## The Meta-Lesson: Subtract Before You Add

When a causal impact model doesn't achieve significance, most analysts' instinct is to add
more covariates. In practice, the biggest improvements come from **removing** things:

| Action | Type | Typical impact |
|---|---|---|
| Exclude high-variance periods from pre-period | Subtraction | -19% CI width |
| Remove contaminated covariates (changed during intervention) | Subtraction | +36% effect estimate, p halved |
| Remove covariates redundant with built-in model components | Subtraction | Cleaner model, less noise |
| Add better covariates (multi-modal seasonal intensity) | Addition | -11% CI width |
| Add exogenous signals (weather, daylight) | Addition | -5% CI width |

In the example public safety analysis, this took p from 0.187 to 0.028 — all from the same data, same model
architecture. Three of five improvement steps were subtractions.

**Practical workflow:** When p > 0.10, try these in order:
1. Shorten the pre-period (exclude high-variance events)
2. Run the combined covariate audit — remove anything that changed >10% during intervention
3. Check for redundancy with built-in model components (e.g., nseasons)
4. THEN add better covariates (intensity curves, weather, interactions)

## Key Pitfalls to Avoid

1. **Using covariates affected by the intervention** — the #1 failure mode. Always check
   whether the campaign could have influenced each covariate.

2. **Binary flags for high-variance events** — use continuous intensity variables instead.
   A binary `school_holiday_flag` can't explain why half-term differs from summer holidays.

3. **Claiming significance when p > 0.10** — destroys credibility. Be honest about uncertainty.

4. **Too-long pre-periods with structural breaks** — if predictions get worse with more data,
   shorten the pre-period. Find the sweet spot.

5. **Ignoring seasonal distortions** — school holidays introduce extreme variance that inflates
   all credible intervals. Model them explicitly with intensity curves.

6. **Running only one method** — a single implementation might have bugs or assumptions that
   bias the result. Two independent methods with different inference engines is the gold standard.

7. **Forgetting the dependency conflict** — tfcausalimpact and CausalPy cannot coexist in the
   same Python environment. Always run in separate scripts.

8. **Including high-variance seasonal periods in the pre-period** — School holiday periods
   can inflate credible intervals by 30%+. Test excluding them: start the pre-period after
   the holiday ends. In the example public safety case, this reduced CI width by 19% and
   improved p-value from 0.187 to 0.141 while the effect estimate remained stable. Always
   run a pre-period sensitivity test with multiple start dates to find the optimal
   noise/data tradeoff.

## Reference: Covariate Correlation Benchmarks

From the example public safety analysis (daily accident count):

| Covariate | r with Accident Count | Notes |
|---|---|---|
| construction_zones | +0.812 | Strong but check intervention safety |
| rainfall_mm | +0.791 | Usually the safest control (exogenous) |
| school_holiday_intensity | +0.691 | Multi-modal (half-term + summer + Christmas break) |
| daylight_hours | -0.583 | Shorter days → more accidents |
| temperature_avg | -0.342 | Cold/ice conditions increase risk |
| bank_holiday_flag | +0.198 | Named public holidays |
| traffic_volume | +0.167 | Check intervention safety — may be affected |
| weekend_x_rainfall | +0.143 | Interaction term |
| sin_dow / cos_dow | -0.091 / +0.006 | Low standalone but captures weekly cycle in model |
| school_holiday_flag | -0.024 | Near-zero — binary flag is inadequate for seasonal patterns |

These are benchmarks, not universals — always compute correlations for the specific dataset.

## Reference: Method Selection for Short Campaigns

From the example public safety analysis — key lessons about which methods work for short interventions:

| Method | Result | Key Insight |
|---|---|---|
| **BSTS (tfcausalimpact)** | −18%, p=0.21, not significant | Global time series model — daily variance drowns out short effects |
| **CausalPy (PyMC)** | Consistent, R²=0.72 | Confirms direction but same significance challenge |
| **RDiT** | **−23 acc/day, CI [−38, −8] — significant** | Local boundary comparison avoids global variance problem |
| **Conformal CI** | −21 acc/day, CI 61% tighter than Bayesian | Distribution-free — doesn't depend on model specification |

**Key strategic insight:** For short campaigns (< 7 days), **RDiT should be the lead method**, not BSTS.
BSTS is powerful for long interventions where the full time series structure matters, but for short
campaigns the global variance dominates. RDiT focuses only on the local discontinuity at the boundary,
sidestepping the noise problem entirely. Use BSTS as a supporting method for the full counterfactual
decomposition, and RDiT for the significance claim.

**Conformal intervals** should always be run alongside Bayesian CIs. They were 61% tighter in the example
public safety case — a dramatic improvement. Use the pre-period residual quantile approach: `np.quantile(np.abs(residuals), 0.95)`.

**Fourier seasonality (k=1..4):** Did NOT help with ~17 months of data (+0.9% CI width). Requires 2+ full
annual cycles to learn meaningful patterns. Don't add Fourier terms unless the pre-period spans 2+ years.

## Reference: Pre-period Start Date Sensitivity

From the example public safety analysis (daily accident count):

| Start Date | Description | Days | CI Width Impact | p-value |
|---|---|---|---|---|
| Aug 2025 | Full data (with summer holidays) | 187 | Baseline | 0.187 |
| Sep 8 2025 | Post-summer-holiday (recommended) | 148 | -19% | 0.141 |
| Oct 2025 | After half-term | 126 | -22% | 0.152 |
| Nov 2025 | Winter onward | 95 | -24% | 0.138 |

The sweet spot is usually just after the major seasonal disruption — enough data to learn
patterns, but excluding the period that dominates the variance. For traffic safety, starting
after school holidays resume is a reliable default.

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
