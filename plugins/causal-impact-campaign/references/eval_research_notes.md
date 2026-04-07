# Eval Suite v2.0.0 — Research Notes

Paper-to-assertion mapping documenting which academic sources support each eval assertion.

## Assertion → Source Mapping

### Methodology (5 assertions)

| ID | Assertion | Primary Source | Key Claim |
|---|---|---|---|
| `assert-perm-effect-size` | Compare effect sizes, not p-values | Abadie et al. (2010, JASA); Young (2019, QJE) | SCM placebo uses MSPE ratio (effect-size statistic). Model p-values inflate under misspecification (13-22% fewer significant results with randomization tests). |
| `assert-perm-plus-one` | +1 correction in p-value formula | Phipson & Smyth (2010); North et al. (2002) | (n_extreme + 1)/(n_total + 1) ensures exact uniformity under the null and avoids p=0. |
| `assert-perm-exclusion-zone` | Exclude dates near real intervention | Abadie (2021, JEL) | Spillover from proximate dates invalidates the sharp null hypothesis. |
| `assert-dual-method` | Run ≥2 independent methods | Skill best practice | Single-method results may have implementation bugs or violated assumptions. Cross-method agreement is more convincing. |

### Validation (4 assertions)

| ID | Assertion | Primary Source | Key Claim |
|---|---|---|---|
| `assert-fpr-threshold` | FPR > 10% = miscalibrated | Eggers et al. (2024, AJPS) | "FPR = 5% defines well-calibrated" — a test at α=0.05 should reject exactly 5% of true nulls. |
| `assert-fpr-all-methods` | FPR inflation is model-class-independent | Skill empirical (session 33); Gils et al. (2022) | BSTS VI 41%, HMC 47%, Prophet 55%, RDiT 51%. Root cause: autocorrelation + seasonal complexity, not inference method. |
| `assert-pre-period-minimum` | Short pre-period + many covariates = overfit | Peduzzi et al. (1996); Afyouni et al. (2019); Gils et al. (2022) | EPV < 10 threshold. N_eff = N/(1+2Σρ_t) reduces effective sample size. BSTS FDR inflates to ~10% with 6 pre-period obs. |
| `assert-scorecard-required` | Both permutation AND placebo required | Athey & Imbens (2017); Linden (2018) | Placebo analyses are "standard robustness requirement" for observational causal inference. |
| `assert-masking-perm-validation` | Masking must be validated with permutation | Skill empirical finding | mask_nov_jan: BSTS p=0.047 but perm p=0.22. Overconfident because calm data produces low p at random dates too. |

### Covariates (3 assertions)

| ID | Assertion | Primary Source | Key Claim |
|---|---|---|---|
| `assert-covariate-contamination` | Covariates affected by treatment must be flagged | Scott & Varian (2014); Pearl (2009) | Covariates must be "predictive, not causal" — unaffected by treatment. Post-treatment variables are "bad controls." |
| `assert-covariate-sensitivity-test` | Test with/without suspect covariates | Skill covariate safety rule | >20% effect change when removing covariate indicates signal absorption (mediator/collider bias). |
| `assert-brand-search-endogenous` | Brand search is endogenous | Skill empirical finding | Brand search worsened p by ~3x and dropped effect by ~30%. Category search (r≈0.01 with revenue) was the best covariate. |
| `assert-sale-covariate-zeroing` | Sale covariates must be zeroed during treatment | Skill ADR 0020 | Unzeroed p=0.074, zeroed combined p=0.033. Pre-period signal is valuable but treatment-period signal is contaminated. |

### Interpretation (3 assertions)

| ID | Assertion | Primary Source | Key Claim |
|---|---|---|---|
| `assert-direction-consistency` | Unanimous direction is strong evidence | Skill claim framing guide | "The convergence of all specs on a positive direction IS the strongest evidence." |
| `assert-method-disagreement` | Report both results when methods disagree | Skill methodology | Use median for consensus. Flag 3x outliers. Never cherry-pick the favorable result. |
| `assert-vi-stochasticity` | VI p-value differences <0.03 are noise | Skill empirical finding | Same spec ranges p=0.03-0.08 between VI runs. Use HMC or average 5+ runs for definitive comparisons. |

### Client Framing (3 assertions)

| ID | Assertion | Primary Source | Key Claim |
|---|---|---|---|
| `assert-prob-positive-framing` | Use prob_positive for non-technical audiences | Makowski et al. (2019); Muehlemann et al. (2023) | "Probability of Direction" (pd) formally defined. Clinical trials literature recommends posterior probabilities over p-values. |
| `assert-prob-positive-cherry-pick` | Don't convert cherry-picked p to probability | Gelman & Yao (2021); Skill Pitfall 10 | Pr(effect > 0) overstates certainty with flat priors. 1-p from spec search smuggles bias into Bayesian-sounding claim. |
| `assert-honest-nonsignificant` | Don't claim significance when p > 0.10 | Skill Pitfall 3 | "Claiming significance when p > 0.10 destroys credibility." |

### Edge Cases (3 assertions)

| ID | Assertion | Primary Source | Key Claim |
|---|---|---|---|
| `assert-short-campaign-rdit` | Short campaigns → RDiT as lead | Skill reference table | RDiT achieved significance for 4-day promo where BSTS did not. Local boundary focus avoids global variance. |
| `assert-nan-handling` | NaN from BSTS needs JSON handling | Skill permutation pitfalls | `raw.replace('NaN', 'null')` before JSON parsing. gsutil streaming silently drops NaN objects. |
| `assert-causalpy-macos-cores` | CausalPy macOS requires cores=1 | Skill environment gotchas | macOS multiprocessing fork issue causes RuntimeError. |

## New Trigger Tests (5 added for merged skills)

| ID | Trigger Phrase | Source Skill |
|---|---|---|
| `pos-16` | "placebo test to validate BSTS" | bsts-placebo-calibration |
| `pos-17` | "validate p-value with permutation" | permutation-validation |
| `pos-18` | "is my BSTS result overfit" | bsts-placebo-calibration |
| `pos-19` | "false positive rate of my model" | bsts-placebo-calibration |
| `pos-20` | "how many permutation shuffles" | permutation-validation |

## Summary Statistics

- **Assertions**: 22 (methodology: 4, validation: 5, covariates: 4, interpretation: 3, client_framing: 3, edge_cases: 3)
- **Trigger tests**: 40 (20 positive, 13 negative, 1 new negative, 8 edge = 42 total triggers)
- **Test cases**: 6 (unchanged from v1)
- **Edge cases**: 5 (unchanged from v1)
- **Academic sources cited**: 15 papers + 3 industry tools
