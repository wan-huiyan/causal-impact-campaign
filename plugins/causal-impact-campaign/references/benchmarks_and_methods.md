# Reference: Benchmarks, Method Selection, and Sensitivity

## Covariate Correlation Benchmarks

From a retail engagement (daily revenue):

| Covariate | r with Revenue | Notes |
|---|---|---|
| paid_sessions | +0.885 | Strong but check intervention safety |
| organic_sessions | +0.863 | Usually the safest control |
| xmas_intensity | +0.828 | Multi-modal v2 (BF spike + gift shopping + Boxing Day) |
| kcp_period_flag | +0.523 | Key consumption period |
| payday_window_flag | +0.213 | 25th-3rd spending window |
| holiday_flag | +0.153 | Named public holidays |
| paid_share | +0.140 | Media intensity ratio |
| payday_x_weekend | +0.123 | Interaction term |
| sin_dow / cos_dow | -0.078 / +0.004 | Low standalone but captures weekly cycle |
| winter_sale_flag | -0.024 | Near-zero — binary flag inadequate for retail peaks |

These are benchmarks, not universals — always compute correlations for the specific client.

## Method Selection for Short Campaigns

| Method | Result | Key Insight |
|---|---|---|
| **BSTS (tfcausalimpact)** | +22%, p=0.21, not significant | Daily variance drowns out short effects |
| **CausalPy (PyMC)** | Consistent, R2=0.72 | Confirms direction but same significance challenge |
| **RDiT** | **~+18%, CI excludes zero** | Local boundary comparison avoids global variance |
| **Conformal CI** | Moderate positive, CI 61% tighter | Distribution-free, no model specification |

**Key insight:** For short campaigns (< 7 days), RDiT should be the lead method.

**Conformal intervals** should always be run alongside Bayesian CIs. Use:
`np.quantile(np.abs(residuals), 0.95)`.

**Fourier seasonality (k=1..4):** Requires 2+ full annual cycles. Don't add unless pre-period
spans 2+ years.

## Pre-period Start Date Sensitivity

| Start Date | Description | Days | CI Width Impact | p-value |
|---|---|---|---|---|
| Oct start | Full data (with Christmas) | ~500 | Baseline | ~0.22 |
| Jan 6 start | Post-Christmas (recommended) | ~420 | -27% | ~0.16 |
| Feb start | Post-winter-sale | ~390 | -30% | ~0.18 |
| Mar start | Spring onward | ~360 | -31% | ~0.16 |

### Advanced: Masking high-variance periods

| Approach | Days | CI Width | p-value | Prob+ |
|---|---|---|---|---|
| Full (no mask) | 514 | Baseline | ~0.24 | ~76% |
| Jan 6 start (truncate) | 417 | -29% | ~0.09 | ~92% |
| **Mask BF-Jan 5 both years** | **410** | **-58%** | **~0.06** | **~94%** |
| **Mask Nov-Jan both years** | **330** | **-64%** | **~0.05** | **~95%** |
| Very short (too short) | 52 | -81% | ~0.00 | 100% |

**Warning:** Very short pre-periods (< 60 days) produce overconfident results.

**Temporal scope verification:** Always count how many seasonal instances exist and verify
ALL are handled. Label specs precisely: "masks Christmas 2024 + 2025 (both years)."

```python
# Verify temporal scope
for event_name, month_start, month_end in [('Christmas/winter', 11, 1)]:
    instances = []
    for year in df.index.year.unique():
        mask = (df.index >= f'{year}-{month_start:02d}-01') & (df.index <= f'{year+1}-{month_end:02d}-31')
        if mask.any():
            instances.append(year)
    print(f"{event_name}: {len(instances)} instances in data ({instances})")
```

## Leave-One-Out Covariate Sensitivity

| Dropped | p-value | Effect | Verdict |
|---|---|---|---|
| None (full model) | 0.07 | Baseline | Baseline |
| cos_dow | **0.03** | +19% | **Noise — model improves without it** |
| payday_window_flag | 0.04 | +46% | Marginal |
| xmas_intensity | 0.055 | — | Helpful but not critical |
| sin_dow | 0.056 | — | Helpful but not critical |
| organic_sessions | **0.148** | — | **Critical — model collapses without it** |

**Key insight:** More covariates is not always better. cos_dow was noise because nseasons=7
already captures weekly seasonality internally.

## Cloud Run for Batch BSTS Runs

Running 12+ specs locally takes 60+ min. Use Cloud Run Jobs:
- `--task-timeout=1800s`, `--memory=4Gi --cpu=2`
- Pass spec key via `SPEC_KEY` env var, data location via `GCS_BUCKET`
- Typical cost: ~$0.50 for 12 parallel specs, completes in ~15 min
