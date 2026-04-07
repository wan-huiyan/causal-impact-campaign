# Step 6b: Extension Analyses

After the primary analysis, run these extensions to deepen the insight.

## Effect Decomposition

Run separate CausalImpact on `conversion_rate`, `aov`, and `transactions` as targets. This reveals
**which lever the campaign pulled** — was it conversion, basket size, or traffic?

In one retail engagement: conversion rate showed the strongest signal (+17%, 87% probability
positive) while AOV barely moved (+0.6%). Transactions and revenue rose proportionally (+27-28%).
This "conversion barrier removal" pattern told us the promo made hesitant shoppers buy —
it didn't attract new visitors or increase basket size. This insight directly informed
the client's next promotional strategy.

**Client-friendly framing:** Use `prob_positive = 1 - p` (as a percentage) instead of raw
p-values. "87% probability the effect is positive" resonates far better with business
stakeholders than "p=0.133". Show this as a stat card and table column alongside p-values.

This metric is formally called the "Probability of Direction" (pd) in the Bayesian literature
[Makowski et al., 2019] and is exactly the "Posterior prob. of a causal effect" that CausalImpact
reports in its summary output [Brodersen et al., 2015]. Under uniform priors, the one-sided
frequentist p-value equals the posterior probability mass below zero [Marsman & Wagenmakers, 2016],
so 1-p is a mathematically grounded Bayesian metric, not an informal conversion. The clinical
trials literature consistently recommends posterior probabilities over p-values for non-technical
stakeholders [Muehlemann et al., 2023; Ruberg, 2021].

**Caution with flat priors:** Gelman & Yao [2021] warn that Pr(effect > 0) can overstate certainty
when priors are flat/uninformative. CausalImpact uses spike-and-slab priors (regularizing), making
this metric more defensible than it would be under flat priors. Still, always derive probability
ranges from systematic sensitivity reruns, not cherry-picked specs (see Pitfall 10).

This is often the most valuable insight for the client — it informs future offer design.

## Channel Split

Run CausalImpact on `paid_revenue` and `organic_revenue` separately to see if the campaign
affected all channels or just one. Use `organic_sessions` as control for both (don't use
`paid_sessions` as control for paid revenue — endogeneity risk).

If both channels lift proportionally, it's a site-wide conversion effect. If only paid lifts,
the campaign may be driving traffic rather than conversion.

## Post-Promo Persistence

Run CausalImpact with the full post-period (intervention start -> data end) instead of just the
promo window. Compare average daily effect during promo vs after promo:

```
persistence_ratio = post_promo_avg_daily_effect / during_promo_avg_daily_effect
```

- Ratio > 50%: Significant persistence — report total impact including post-period
- Ratio 10-50%: Partial persistence — mention as additional upside
- Ratio < 10%: Effect dissipated — report promo-period only

**Warning:** Persistence analysis is unreliable for short campaigns. Persistence estimates are
highly sensitive to covariate choice. Use clean covariates and interpret conservatively.
Frame persistence as "inconclusive" unless multiple specs agree and the extended post-period is plausible.

## Weather Covariate

For retail/ecommerce clients, add daily temperature and precipitation as covariates.
Source: [Open-Meteo API](https://open-meteo.com/) — free, no API key needed.

```python
import requests
resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": 51.5074, "longitude": -0.1278,
    "start_date": "2024-10-01", "end_date": "2026-03-15",
    "daily": "temperature_2m_mean,precipitation_sum",
    "timezone": "Europe/London",
})
```

Weather typically has low standalone correlation with revenue (r ~ -0.05 to +0.03) but provides
an orthogonal exogenous signal that can tighten credible intervals by 2-5%.

**SSL note:** Corporate proxies may block the Open-Meteo API. Use `curl -sk` to bypass, or
`requests.get(..., verify=False)`.

## Prophet Cross-Validation

For high-stakes claims, run Facebook Prophet as an independent cross-validation method. Prophet
uses additive decomposition (Fourier seasonality + changepoint trend) — a fundamentally different
model family from BSTS (structural time series + state space).

```python
from prophet import Prophet
m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
m.add_regressor("organic_sessions")
m.fit(train_df)  # pre-period only, columns: ds, y, organic_sessions
forecast = m.predict(future_df)
# Effect = actual - yhat; CI from yhat_lower/yhat_upper
```

Prophet does not produce a frequentist p-value. Report as "CI excludes zero" if
`actual_sum - yhat_upper_sum > 0`.

## Contaminated Exogenous Metrics

**Never use Google Trends brand search as a covariate in promo analysis.**
Brand search is endogenous — the promo drives search interest. Validated: brand search
worsened p by ~3x and dropped the effect estimate by ~30%.

**Sale detection flags CAN be used — but must be zeroed during the treatment window.**
Implementation: `prep_df(treatment_s=..., treatment_e=..., zero_treatment_cols=["sale_type_flag", "sale_intensity"])`.

**The best exogenous search signal: category-level Google Trends at daily resolution.**
- Use a generic category term (e.g., "shoes" for a shoe retailer) — NOT the brand name
- Include BOTH daily AND weekly as dual-frequency signal
- Good BSTS covariates need: low revenue correlation + high daily variation + exogeneity

**The Correlation Paradox:** Competitor brand search has HIGH revenue correlation (~0.6-0.7)
but HURTS the model — it's redundant with BSTS components. Category search has NEAR-ZERO
correlation (~0.01) but is the BEST covariate.

## Sale Period Auto-Detection (Coupon Ratio)

```python
coupon_ratio = transactions_with_coupon / transactions
trailing_median = coupon_ratio.rolling(28).median()
trailing_mad = coupon_ratio.rolling(28).apply(lambda x: np.median(np.abs(x - np.median(x))))
z = (coupon_ratio - trailing_median) / (1.4826 * trailing_mad)
# sale_type = "coupon_sale" if z > +2.5, "sitewide_sale" if z < -2.5 AND volume > P25
```

**Treatment-window zeroing (ADR 0020):** Both sale covariates automatically zeroed during
treatment window via `prep_df()`. Validated: unzeroed p=0.074, zeroed combined p=0.033.

## Tiered Covariate Recommendation

| Tier | Covariates | Permutation | When to Use |
|---|---|---|---|
| **Default** | ~5 base (sessions, DoW, holiday, weather) | perm p ~0.06 | Always |
| **Enhanced** | Default + category search (daily + weekly) | perm p ~0.04 | When Trends available |
| **Full** | Enhanced + sale signals (zeroed) | perm p ~0.08 | When coupon data available |

**No hard covariate cap.** Use permutation p < 0.10 as gatekeeper, not arbitrary covariate budget.
