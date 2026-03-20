# Code Templates for Causal Impact Campaign Analysis

## Table of Contents
1. [Data Loading (BigQuery + GA4)](#1-data-loading)
2. [Covariate Engineering](#2-covariate-engineering)
3. [tfcausalimpact Analysis](#3-tfcausalimpact)
4. [CausalPy Analysis](#4-causalpy)
5. [Validation Suite](#5-validation-suite)
6. [Sensitivity Analysis](#6-sensitivity-analysis)
7. [Scorecard](#7-scorecard)
8. [Findings Document Template](#8-findings-template)

---

## 1. Data Loading

```python
from google.cloud import bigquery
import pandas as pd
import numpy as np

PROJECT_ID = "your-project"
client = bigquery.Client(project=PROJECT_ID)

# Load features + calendar
query = """
SELECT t.*, c.* EXCEPT(date)
FROM `{project}.{dataset}.features` t
LEFT JOIN `{project}.{dataset}.dim_calendar` c ON t.date = c.date
ORDER BY t.date
"""
df = client.query(query.format(project=PROJECT_ID, dataset=DATASET)).to_dataframe()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
```

### Calendar Table SQL (BigQuery)

```sql
CREATE OR REPLACE TABLE `project.dataset.dim_calendar` AS
WITH date_spine AS (
  SELECT day AS date
  FROM UNNEST(GENERATE_DATE_ARRAY(DATE '2024-01-01', DATE '2026-12-31')) AS day
),
holiday_logic AS (
  SELECT date,
    EXTRACT(YEAR FROM date) AS year,
    -- Easter Sunday (anonymous Gregorian algorithm)
    DATE_ADD(DATE(EXTRACT(YEAR FROM date), 3, 22),
      INTERVAL (MOD(19 * MOD(EXTRACT(YEAR FROM date), 19) + 24, 30)
        + MOD(2 * MOD(EXTRACT(YEAR FROM date), 4) + 4 * MOD(EXTRACT(YEAR FROM date), 7)
        + 6 * MOD(19 * MOD(EXTRACT(YEAR FROM date), 19) + 24, 30) + 5, 7)) DAY
    ) AS easter_sunday,
    -- Black Friday
    DATE_ADD(DATE(EXTRACT(YEAR FROM date), 11, 1), INTERVAL (
      CASE WHEN EXTRACT(DAYOFWEEK FROM DATE(EXTRACT(YEAR FROM date), 11, 1)) <= 5
        THEN 5 - EXTRACT(DAYOFWEEK FROM DATE(EXTRACT(YEAR FROM date), 11, 1)) + 21
        ELSE 5 - EXTRACT(DAYOFWEEK FROM DATE(EXTRACT(YEAR FROM date), 11, 1)) + 28
      END) DAY) AS black_friday
  FROM date_spine
)
SELECT
  date,
  EXTRACT(DAYOFWEEK FROM date) AS day_of_week,
  CASE WHEN EXTRACT(DAYOFWEEK FROM date) IN (1, 7) THEN 1 ELSE 0 END AS is_weekend,
  CASE WHEN EXTRACT(DAY FROM date) IN (25,26,27,28,29,30,31,1,2,3) THEN 1 ELSE 0 END AS payday_window_flag,
  -- Add holiday_name, kcp_period_flag, etc. per client
FROM holiday_logic
```

### GA4 Features Table SQL

```sql
CREATE OR REPLACE TABLE `project.dataset.features` AS
WITH base AS (
  SELECT
    PARSE_DATE('%Y%m%d', event_date) AS date,
    event_name,
    CONCAT(user_pseudo_id,
      CAST((SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS STRING)
    ) AS session_id,
    ecommerce.transaction_id,
    ecommerce.purchase_revenue,
    -- Session traffic source (last click)
    LOWER(IF(session_traffic_source_last_click.google_ads_campaign.account_name IS NOT NULL,
      'google', session_traffic_source_last_click.manual_campaign.source)) AS traffic_source,
    LOWER(IF(session_traffic_source_last_click.google_ads_campaign.account_name IS NOT NULL,
      'cpc', session_traffic_source_last_click.manual_campaign.medium)) AS traffic_medium
  FROM `project.analytics_XXXXXXX.events_*`
  WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
    AND privacy_info.analytics_storage = 'Yes'
    AND event_name IN ('purchase', 'session_start')
),
enriched AS (
  SELECT *,
    CASE
      WHEN REGEXP_CONTAINS(LOWER(COALESCE(traffic_medium, '')), 'paid|cpc|ppc|display') THEN 'paid'
      ELSE 'organic'
    END AS channel_group
  FROM base
)
SELECT
  date,
  ROUND(SUM(CASE WHEN event_name = 'purchase' THEN COALESCE(purchase_revenue, 0) ELSE 0 END), 2) AS revenue,
  COUNT(DISTINCT CASE WHEN event_name = 'purchase' THEN transaction_id END) AS transactions,
  COUNT(DISTINCT session_id) AS sessions,
  COUNT(DISTINCT CASE WHEN channel_group = 'paid' THEN session_id END) AS paid_sessions,
  COUNT(DISTINCT CASE WHEN channel_group = 'organic' THEN session_id END) AS organic_sessions,
  SUM(CASE WHEN event_name = 'purchase' AND channel_group = 'paid' THEN COALESCE(purchase_revenue, 0) ELSE 0 END) AS paid_revenue,
  SUM(CASE WHEN event_name = 'purchase' AND channel_group = 'organic' THEN COALESCE(purchase_revenue, 0) ELSE 0 END) AS organic_revenue
FROM enriched
GROUP BY 1
```

---

## 2. Covariate Engineering

```python
# Cyclical day-of-week
df['dow'] = df['date'].dt.dayofweek
df['sin_dow'] = np.sin(2 * np.pi * df['dow'] / 7)
df['cos_dow'] = np.cos(2 * np.pi * df['dow'] / 7)

# Christmas intensity (continuous Gaussian bell)
def christmas_proximity(dates):
    result = pd.Series(0.0, index=dates if isinstance(dates, pd.DatetimeIndex) else dates.index)
    bf_peak = 27
    for year in (dates.year if isinstance(dates, pd.DatetimeIndex) else dates.dt.year).unique():
        xmas = pd.Timestamp(f"{year}-12-25")
        for idx in (dates if isinstance(dates, pd.DatetimeIndex) else dates):
            d = (xmas - idx).days
            if 0 <= d <= 45:
                sigma = 12 if d >= bf_peak else 8
                result.loc[idx] = max(result.loc[idx], np.exp(-0.5 * ((d - bf_peak) / sigma) ** 2))
            elif -10 <= d < 0:
                result.loc[idx] = max(result.loc[idx], np.exp(-0.5 * ((-d) / 4) ** 2) * 0.5)
    return result

df['xmas_intensity'] = christmas_proximity(df['date'])

# Holiday flag
df['holiday_flag'] = (df['holiday_name'].notna()).astype(int)

# Paid share ratio
df['paid_share'] = df['paid_sessions'] / (df['paid_sessions'] + df['organic_sessions'])

# Interaction
df['payday_x_weekend'] = df['payday_window_flag'] * df['is_weekend']
```

### Data preparation function

```python
def prepare_model_data(df, date_col, target_col, covariates, binary_covariates=None):
    binary_covariates = binary_covariates or []
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).set_index(date_col)
    keep = [target_col] + [c for c in covariates if c in data.columns]
    data = data[keep].apply(pd.to_numeric, errors="coerce")
    full_idx = pd.date_range(data.index.min(), data.index.max(), freq="D")
    data = data.reindex(full_idx)
    data[target_col] = data[target_col].fillna(0)
    continuous = [c for c in covariates if c not in binary_covariates and c in data.columns]
    binary = [c for c in covariates if c in binary_covariates and c in data.columns]
    if continuous:
        data[continuous] = data[continuous].interpolate(method="linear", limit_direction="both").bfill().ffill().fillna(0)
    if binary:
        data[binary] = data[binary].fillna(0).clip(0, 1).round().astype(int)
    return data
```

---

## 3. tfcausalimpact

```python
# Requires: pip install tfcausalimpact (numpy<2)
import tensorflow as tf
np.random.seed(42)
tf.random.set_seed(42)
from causalimpact import CausalImpact

MODEL_ARGS = {"nseasons": 7, "standardize_data": True, "fit_method": "vi"}

ci = CausalImpact(data[required_cols], pre_period, promo_window, model_args=MODEL_ARGS)

# Extract key metrics
s = ci.summary_data
print(f"Cumulative effect: £{s.loc['abs_effect', 'cumulative']:,.0f}")
print(f"95% CI: [£{s.loc['abs_effect_lower', 'cumulative']:,.0f}, £{s.loc['abs_effect_upper', 'cumulative']:,.0f}]")
print(f"Relative effect: {s.loc['rel_effect', 'average']:.1%}")
print(f"p-value: {ci.p_value:.4f}")
```

---

## 4. CausalPy

```python
# Requires: pip install causalpy (numpy>=2)
# MUST run in a separate script from tfcausalimpact
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
            "cores": 1,  # REQUIRED on macOS
        }
    ),
)
result.summary()
fig, axes = result.plot()
```

---

## 5. Validation Suite

```python
def extract_metrics(ci, label):
    s = ci.summary_data
    return {
        "label": label,
        "abs_effect_cum": float(s.loc["abs_effect", "cumulative"]),
        "rel_effect": float(s.loc["rel_effect", "average"]),
        "p_value": float(ci.p_value),
    }

def backtest_metrics(ci, actual_post):
    inf = ci.inferences.loc[actual_post.index]
    pred = inf["post_preds_means"].astype(float)
    lower, upper = inf["post_preds_lower"].astype(float), inf["post_preds_upper"].astype(float)
    err = actual_post.astype(float) - pred
    return {
        "wape": float(np.abs(err).sum() / actual_post.abs().sum()) if actual_post.abs().sum() else np.nan,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "coverage_95": float(((actual_post >= lower) & (actual_post <= upper)).mean()),
    }

# Rolling backtests
HORIZON = promo_days
candidates = []
ps = pre_dates[-1] - pd.Timedelta(days=HORIZON)
while ps >= pre_dates[0] + pd.Timedelta(days=MIN_TRAIN) and len(candidates) < MAX_WINDOWS:
    candidates.append(ps)
    ps -= pd.Timedelta(days=STEP)

for ps in candidates:
    pe = ps + pd.Timedelta(days=HORIZON - 1)
    bt_pre = [data.index.min().strftime("%Y-%m-%d"), (ps - pd.Timedelta(days=1)).strftime("%Y-%m-%d")]
    bt_post = [ps.strftime("%Y-%m-%d"), pe.strftime("%Y-%m-%d")]
    ci = CausalImpact(data[cols], bt_pre, bt_post, model_args=MODEL_ARGS)
    # ... extract metrics ...

# Placebo test
placebo_rank = (placebo_effects < real_effect).mean()
```

---

## 6. Sensitivity Analysis

```python
SENSITIVITY_SPECS = {
    "full_model": ALL_COVARIATES,
    "organic_only": ["organic_sessions"],
    "organic_plus_calendar": ["organic_sessions", "sin_dow", "cos_dow", "payday_window_flag", "holiday_flag"],
    "organic_plus_xmas": ["organic_sessions", "sin_dow", "cos_dow", "xmas_intensity", "kcp_period_flag"],
    "paid_organic_split": ["organic_sessions", "paid_sessions", "sin_dow", "cos_dow"],
    "kitchen_sink": ALL_COVARIATES + ["paid_share"],
}

for name, covars in SENSITIVITY_SPECS.items():
    ci = CausalImpact(data[[TARGET] + covars], pre_period, promo_window, model_args=MODEL_ARGS)
    row = extract_metrics(ci, name)
```

---

## 7. Scorecard

```python
scorecard = {
    "Median backtest WAPE <= 15%": (bt_ok["wape"].median() <= 0.15, f"{bt_ok['wape'].median():.2%}"),
    "Mean 95% coverage >= 80%": (bt_ok["coverage_95"].mean() >= 0.80, f"{bt_ok['coverage_95'].mean():.2%}"),
    "Placebo rank >= 90th %ile": (placebo_rank >= 0.90, f"{placebo_rank:.0%}"),
    "All specs same direction": (pos_share >= (n_ok-1)/n_ok, f"{pos_share:.0%}"),
    "p-value <= 0.10": (ci.p_value <= 0.10, f"{ci.p_value:.3f}"),
}
```

---

## 8. Findings Template

See the SKILL.md Step 7 for the document structure. The findings markdown should include:
- Executive summary with headline table
- Data sources and windows
- Methodology (accessible to non-technical)
- Results from both methods
- Validation scorecard
- Honest interpretation
- Client narrative paragraph
- Recommendations
- File reference
- Technical notes (deps, auth, config)
