# Case Study: The Path to a Validated Result

Every causal analysis follows the same pattern: a promising-but-not-significant first result, then systematic improvements that either find the real signal or confirm there isn't one. Here's what that looks like on a real engagement.

## The Journey: p=0.22 to Permutation-Validated

| Step | p-value | Key Insight |
|---|---|---|
| Original specification | 0.223 | Correct direction, wide credible intervals |
| Exclude high-variance pre-period | 0.163 | -27% CI width. Seasonal variance was the dominant noise source |
| Multi-modal holiday intensity (v2) | 0.140 | 6-component Gaussian curve (r=0.828) replaces binary flag (r=0.02) |
| Remove contaminated covariate | ~0.06 | Biggest single improvement — covariate was absorbing the treatment effect |
| Mask winter sale periods (both years) | 0.047 | Masking Nov-Jan: CV drops from 58% to 24%, 12/12 specs positive |
| **Permutation validation (50 shuffles)** | **perm p=0.032** | **Effect-size comparison confirms the result is unusual vs random dates** |

**Important:** The model-based p=0.047 alone was not sufficient. Multi-method placebo testing revealed ALL methods (BSTS VI, HMC, Prophet, RDiT) show 35-55% false positive rates on this dataset. The permutation test (which compares effect sizes, not model p-values) provided the honest validation.

## The Meta-Lesson: Subtract Before You Add

When a causal impact model doesn't achieve significance, most analysts add more covariates. In practice, the biggest improvements come from **removing** things:

| Action | Type | Typical Impact |
|---|---|---|
| Exclude high-variance periods from pre-period | Subtraction | -27% CI width |
| Remove contaminated covariates | Subtraction | +36% effect estimate, p halved |
| Remove covariates redundant with model components | Subtraction | Cleaner model, less noise |
| Add better covariates (multi-modal holiday intensity) | Addition | -11% CI width |
| Add exogenous signals (weather) | Addition | -3% CI width |

Three of five improvement steps were subtractions.

## The Specification Search Caveat

If you test N specifications and report the one with the lowest p-value, the result is exploratory, not confirmatory. In this engagement, 48 experiments were conducted. The best single-spec p=0.039 was an outlier — all 6 sensitivity specs had p=0.18-0.24.

The honest framing: "The primary specification produces p=0.21. An optimised specification achieves p=0.039, but this should be treated as exploratory." Lead with the Bayesian posterior probability across ALL specs, not the cherry-picked p-value.

## Key Lessons Encoded in the Skill

**Strategic** (apply to any causal analysis):
- Subtract before you add — removing contaminated covariates and high-variance pre-periods beats adding more features
- Mask, don't truncate — masking high-variance windows preserves the annual cycle while removing noise
- Contaminated covariates silently absorb causal effects — always run a safety audit
- Two methods > one — cross-method agreement is stronger evidence than any single p-value
- Honest uncertainty builds client trust — never claim statistical significance you don't have
- Run many specs, report all of them — direction consistency across 12/12 specs is stronger than 1/1 significant

**Tactical** (specific techniques):
- Binary flags can't capture magnitude — use multi-modal intensity curves for seasonal peaks
- RDiT beats BSTS for short interventions (< 7 days) — local boundary comparison achieves significance where global BSTS can't
- Conformal CIs are 61% tighter than Bayesian — distribution-free uncertainty as a sanity check
- nseasons=7 makes DoW covariates redundant, but nseasons=14 needs them
- Continuous sale intensity > binary flag — MAD z-scores naturally weight major sales higher
- Weather interactions encode consumer behaviour — `precip x sale_intensity` captures "friction x intent"
