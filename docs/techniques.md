# Key Techniques Explored

## What Worked

| Technique | Impact | How |
|---|---|---|
| **Combined covariate safety audit** | Removed significant bias | Checks both correlation AND intervention safety per covariate |
| **Masked pre-periods (post-Issue-#51 fix)** | CV 58% to 24%; CI narrows 26%, model p 0.047 → 0.005 | Mask winter sale windows via a mask-aware index filter that drops the days entirely (not zero-injects them) — preserves annual cycle. See "Data-Prep Zero-Injection Trap" in SKILL.md for why the pre-fix behaviour silently corrupted the pre-period. |
| **Pre-period exclusion of high-variance events** | -27% CI width | Start after seasonal peaks to avoid variance inflation |
| **Multi-modal holiday intensity (v2)** | r: -0.024 to 0.828 | 6-component curve: main peak, secondary peak, ramp, shoulder, baseline, post-event |
| **RDiT (Regression Discontinuity in Time)** | Achieved significance | Local boundary comparison — best method for interventions < 7 days |
| **Conformal prediction intervals** | 61% tighter CIs | Distribution-free intervals from pre-period residual quantiles |
| **Effect decomposition** | Identified primary lever | Separate CausalImpact on sub-metrics reveals which lever moved |
| **Weather covariates** | -3.2% CI width | Open-Meteo API — exogenous, always safe |
| **Rolling-placebo backtest (HMC weekly)** | Clean PASS: rank 0.94, empirical p 0.06 | In-time placebo with recent pre-period windows; mask-off diagnostic confirms rank is not mask-interaction biased |
| **224-spec robustness grid** | 208/208 valid specs directional positive, top 10 $199K–$226K | The strongest single validation signal when single tests disagree — doesn't share single-spec failure modes |
| **Training-length-matched date-shuffled test** | p 0.47 → 0.29 (confound is real but partial) | Constrain fake training windows to match real pre-period length; disentangles training-length confound from genuine model behaviour |

## What Didn't Work (and Why)

| Technique | Hypothesis | Result | Lesson |
|---|---|---|---|
| Fourier annual seasonality (k=1..4) | Capture yearly patterns | +0.9% CI (worse) | Needs 2+ annual cycles; insufficient data |
| sin/cos day-of-week encoding | Capture full weekly cycle | Redundant with nseasons=7 | tfcausalimpact already models DoW internally |
| CausalPy WeightedSumFitter | SC-style weighted combination | Sigma doubled | Wrong model class for single-unit ITS |
| Holiday intensity v3 | Better post-peak fit | r dropped 0.828 to 0.795 | Post-peak is tiny fraction of data |
| Google Trends brand search | Exogenous demand signal | p worsened 3x | Endogenous — the campaign drives brand search |
| Date-shuffled randomization on canonical spec | Substitute for model p-value | Full-range p=0.47; training-length-matched p=0.29; **0/20 top SCA specs pass** | **Structural issue, not model defect.** Random fake treatment dates on a 2-year pre-period with strong seasonality allow 6.5× fluctuations, which demands an impossibly strong effect to reject. Backtest + spec curve are the correct validation tools for mask-mode SCA specs on retail data. |
| P-value comparison in permutation tests | Simpler than effect-size comparison | Confounded by model FPR | All methods have inflated FPR; comparing p-values inherits the inflation |
| Full pre-period (700 days, no mask) | More data is better | FPR ~39% | Structural breaks and regime changes contaminate long pre-periods |
| Mask BF-Jan with full data (pre-Issue-#51) | Remove only BF period | FPR ~93% | The high FPR was dominated by the mask zero-injection bug — the mask silently reinjected masked days as $0 revenue, teaching the model a spurious "winter ↔ $0" relationship. Fixed in PR #52; see "Data-Prep Zero-Injection Trap" section in SKILL.md. |
| Absolute-value comparison in placebo rank | Symmetric treatment of positive/negative placebos | Disagrees materially with signed on near-zero-symmetric distributions (e.g. signed median $12K vs abs median $77K on the same data) | **Use signed comparison** `p < real_effect` — it matches the canonical Abadie/Eggers convention and is what tfcausalimpact's downstream consumers expect |
| Log-target on short masked pre-periods (no filter) | Capture multiplicative effects | 16 of 20 top SCA specs collapsed to effect ≈ $1 | `log1p(y)` on a masked pre-period leaves so little variation that the model converges to a near-identity counterfactual. Filter via `--min-abs-eff` (default $1000) before downstream permutation/backtest runs. |
| Rewriting client framing based on theoretical reviewer concerns | "The mask biases the rolling placebo", "Training length inflates the null" | Two ~$0.05 Cloud Run diagnostics showed mask-interaction bias is empirically ZERO (rank 0.94 unchanged), training-length confound explains ~60% of the p-gap but not all (p 0.47 → 0.29) | **Run the cheap empirical diagnostic before rewriting the story.** Reuse `_run_sca_placebo_task` with custom `placebo_windows` rather than writing a new Cloud Run handler. Theoretical concerns need empirical tests. |
