# Case Study: The Path to a Validated Result

Every causal analysis follows the same pattern: a promising-but-not-significant first result, then systematic improvements that either find the real signal or confirm there isn't one. Here's what that looks like on a real engagement.

## The Journey: from p=0.22 to a spec-curve-validated effect

| Step | Headline signal | Key Insight |
|---|---|---|
| Original specification | model p=0.223 | Correct direction, wide credible intervals |
| Exclude high-variance pre-period | model p=0.163 | -27% CI width. Seasonal variance was the dominant noise source |
| Multi-modal holiday intensity (v2) | model p=0.140 | 6-component Gaussian curve (r=0.828) replaces binary flag (r=0.02) |
| Remove contaminated covariate | model p≈0.06 | Biggest single improvement — covariate was absorbing the treatment effect |
| Mask winter sale periods (Nov-Jan) | model p=0.047 | Masking Nov-Jan: CV drops from 58% to 24%, 12/12 specs positive |
| **Fix the mask zero-injection bug (Issue #51)** | model p=**0.005** | Mask-aware index filter: CI narrows 26%, p tightens 4×. The pre-fix p=0.047 was computed on a silently corrupted pre-period (masked days reinjected as $0 revenue). See "Data-Prep Zero-Injection Trap" in SKILL.md. |
| Rolling-placebo backtest (post-fix, HMC weekly) | **rank 0.94**, empirical p=0.06 | Clean PASS. Mask-off diagnostic confirms the rank is not mask-interaction biased (byte-identical 0.94 without the mask). |
| Date-shuffled randomization (post-fix, full range) | p=**0.47** | Honest FAIL on the single spec — but see diagnostic + spec curve below |
| Date-shuffled randomization (training-length-matched) | p=**0.29** | Training-window length was a first-order confounder: matching pre-period length explains ~60% of the pre-fix p gap but doesn't fully dissolve it |
| **224-spec robustness grid** (post-fix) | **208/208 valid specs directional positive**, top 10 per mask mode $199K–$226K with model p < 0.025 | The strongest single signal: the effect survives all 208 defensible alternative specifications. Spec grid becomes the primary validation for the client headline. |

**Important:** The model p=0.005 alone is not sufficient — tfcausalimpact's posterior predictive p-value is a Bayesian model-criticism diagnostic, not a frequentist Type-I rate. Multi-method placebo testing on this dataset shows ALL model-based methods (BSTS VI, HMC, Prophet, RDiT) have 35-55% FPR on naive pre-periods. The honest validation comes from **three complementary signals**: (a) 224-spec grid robustness, (b) rolling-placebo backtest rank 0.94, (c) date-shuffled randomization test with training-length-matched diagnostic at p=0.29. When the three disagree, the grid carries primary inferential load — it doesn't share single-spec failure modes.

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

If you test N specifications and report only the one with the lowest p-value, the result is exploratory, not confirmatory. This engagement ran two distinct types of specification search, and they answer different questions:

- **Single-spec optimisation (exploratory)**. Early in the engagement, 48 ad-hoc specifications were tried to reach a defensible canonical. The "best single-spec p=0.039" was an outlier — sensitivity specs clustered at p=0.18–0.24. This path is pure exploration; lead with the Bayesian posterior probability across ALL specs tried, not the cherry-picked p-value.
- **Pre-registered 224-spec grid (confirmatory)**. A defensible grid of 224 mask-mode specifications (55 covariate bundles × 4 mask modes × 2 seasonalities, restricted to the mask-mode subset) was run *after* the canonical was fixed. The grid's primary claim is about **robustness**, not about the lowest p-value: 208/208 valid specs directional positive, top 10 per mask mode $199K–$226K with model p < 0.025. The 16 degenerate log-target specs on short masked pre-periods (effect collapses to ~$1) were filtered before aggregation — see the "Common implementation gotchas for the placebo rank" section in SKILL.md.

The honest framing for the client: **"The +$214K effect is primarily supported by its robustness across 224 alternative specifications; the rolling-placebo backtest is a clean supporting signal; the date-shuffled randomization test at p=0.29 (training-length-matched) is an honest limitation that the spec curve robustness check partly answers."** No single spec carries the headline — the grid does.

## Key Lessons Encoded in the Skill

**Strategic** (apply to any causal analysis):
- Subtract before you add — removing contaminated covariates and high-variance pre-periods beats adding more features
- Mask, don't truncate — masking high-variance windows preserves the annual cycle while removing noise (but verify the mask actually reached the model input — see "Data-Prep Zero-Injection Trap" in SKILL.md)
- Contaminated covariates silently absorb causal effects — always run a safety audit
- Two methods > one — cross-method agreement is stronger evidence than any single p-value
- Honest uncertainty builds client trust — never claim statistical significance you don't have
- Run many specs, report all of them — direction consistency across 208/208 valid specs is stronger than 1/1 significant
- **Lead with spec curve robustness when single tests disagree** — when model-p passes but placebo/randomization tests conflict, the grid is the strongest single signal because it doesn't share single-spec failure modes
- **Diagnose, don't demote** — when a placebo test fails, run the cheap training-length-matched + mask-off diagnostics (~$0.10 on Cloud Run) before rewriting the client framing. Theoretical concerns need empirical tests.

**Tactical** (specific techniques):
- Binary flags can't capture magnitude — use multi-modal intensity curves for seasonal peaks
- RDiT beats BSTS for short interventions (< 7 days) — local boundary comparison achieves significance where global BSTS can't
- Conformal CIs are 61% tighter than Bayesian — distribution-free uncertainty as a sanity check
- nseasons=7 makes DoW covariates redundant, but nseasons=14 needs them
- Continuous sale intensity > binary flag — MAD z-scores naturally weight major sales higher
- Weather interactions encode consumer behaviour — `precip x sale_intensity` captures "friction x intent"
- **Placebo rank must use signed comparison** (`p < real_effect`), not `abs(p) < abs(real_effect)` — the two disagree materially on near-zero-symmetric post-fix distributions (e.g. signed median $12K vs absolute median $77K on the same data)
- **Filter degenerate log-target specs** on short masked pre-periods via `--min-abs-eff` before downstream permutation/backtest — they collapse to ~$1 and crowd out valid specs
