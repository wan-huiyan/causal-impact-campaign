# Key Techniques Explored

## What Worked

| Technique | Impact | How |
|---|---|---|
| **Combined covariate safety audit** | Removed significant bias | Checks both correlation AND intervention safety per covariate |
| **Masked pre-periods** | CV 58% to 24%, p<0.05 | Mask winter sale windows instead of truncating — preserves annual cycle |
| **Pre-period exclusion of high-variance events** | -27% CI width | Start after seasonal peaks to avoid variance inflation |
| **Multi-modal holiday intensity (v2)** | r: -0.024 to 0.828 | 6-component curve: main peak, secondary peak, ramp, shoulder, baseline, post-event |
| **RDiT (Regression Discontinuity in Time)** | Achieved significance | Local boundary comparison — best method for interventions < 7 days |
| **Conformal prediction intervals** | 61% tighter CIs | Distribution-free intervals from pre-period residual quantiles |
| **Effect decomposition** | Identified primary lever | Separate CausalImpact on sub-metrics reveals which lever moved |
| **Weather covariates** | -3.2% CI width | Open-Meteo API — exogenous, always safe |
| **Permutation validation** | Honest significance | Effect-size comparison immune to model FPR inflation |

## What Didn't Work (and Why)

| Technique | Hypothesis | Result | Lesson |
|---|---|---|---|
| Fourier annual seasonality (k=1..4) | Capture yearly patterns | +0.9% CI (worse) | Needs 2+ annual cycles; insufficient data |
| sin/cos day-of-week encoding | Capture full weekly cycle | Redundant with nseasons=7 | tfcausalimpact already models DoW internally |
| CausalPy WeightedSumFitter | SC-style weighted combination | Sigma doubled | Wrong model class for single-unit ITS |
| Holiday intensity v3 | Better post-peak fit | r dropped 0.828 to 0.795 | Post-peak is tiny fraction of data |
| Google Trends brand search | Exogenous demand signal | p worsened 3x | Endogenous — the campaign drives brand search |
| mask_nov_jan without permutation | Remove holiday variance | BSTS p=0.047 but perm p=0.22 | Masking makes model overconfident at random dates too |
| P-value comparison in permutation tests | Simpler than effect-size comparison | Confounded by model FPR | All methods have inflated FPR; comparing p-values inherits the inflation |
| Full pre-period (700 days, no mask) | More data is better | FPR ~39% | Structural breaks and regime changes contaminate long pre-periods |
| Mask BF-Jan with full data | Remove only BF period | FPR ~93% | Masking creates gaps that destabilize the model |
