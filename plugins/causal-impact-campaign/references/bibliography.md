# References

## Core BSTS / CausalImpact

- [Brodersen et al., 2015] Brodersen, K.H., Gallusser, F., Koehler, J., Remy, N., Scott, S.L. "Inferring causal impact using Bayesian structural time-series models." *Annals of Applied Statistics*, 9(1), 247-274. DOI: 10.1214/14-AOAS788
- [Scott & Varian, 2014] Scott, S.L., Varian, H.R. "Predicting the present with Bayesian structural time series." *International Journal of Mathematical Modelling and Numerical Optimisation*, 5(1/2), 4-23. DOI: 10.1504/IJMMNO.2014.059942

## State-Space Methods & Seasonal Estimation

- [Harvey, 1989] Harvey, A.C. *Forecasting, Structural Time Series Models and the Kalman Filter.* Cambridge University Press.
- [Durbin & Koopman, 2012] Durbin, J., Koopman, S.J. *Time Series Analysis by State Space Methods.* 2nd ed., Oxford University Press.
- [Koopman, 1997] Koopman, S.J. "Exact initial Kalman filtering and smoothing for nonstationary time series models." *JASA*, 92(440), 1630-1638. DOI: 10.1080/01621459.1997.10473685
- [Hyndman & Kostenko, 2007] Hyndman, R.J., Kostenko, A.V. "Minimum sample size requirements for seasonal forecasting models." *Foresight*, Issue 6, 12-15.

## FPR Calibration & Sample Size

- [Gils et al., 2022] Gils, T. et al. "Evaluating the power of the causal impact method in observational studies of HCV treatment as prevention." *BMC Infectious Diseases* (PMC9204771). — BSTS FDR inflates to ~10% with 6 pre-period observations; ~5% with 12+.
- [Peduzzi et al., 1996] Peduzzi, P. et al. "A simulation study of the number of events per variable in logistic regression analysis." *J. Clinical Epidemiology*, 49(12), 1373-1379. — EPV < 10 causes unreliable significance tests.
- [Babyak, 2004] Babyak, M.A. "What you see may not be what you get: a brief, nontechnical introduction to overfitting in regression-type models." *Psychosomatic Medicine*, 66(3), 411-421.
- [Afyouni et al., 2019] Afyouni, S., Smith, S.M., Nichols, T.E. "Effective degrees of freedom of the Pearson's correlation coefficient under autocorrelation." *NeuroImage*, 199, 609-625. — N_eff = N / (1 + 2*sum(rho_t)) formula.
- [Oelrich et al., 2020] Oelrich, O. et al. "When are Bayesian model probabilities overconfident?" arXiv:2003.04026. — Bayesian posteriors overconfident when models are misspecified with large degrees of freedom.

## Placebo Tests & Causal Inference Validation

- [Abadie et al., 2010] Abadie, A., Diamond, A., Hainmueller, J. "Synthetic control methods for comparative case studies." *JASA*, 105(490), 493-505. — Established in-space placebo test protocol.
- [Abadie et al., 2015] Abadie, A., Diamond, A., Hainmueller, J. "Comparative politics and the synthetic control method." *AJPS*, 59(2), 495-510. — Introduced in-time placebo test.
- [Abadie, 2021] Abadie, A. "Using synthetic controls: feasibility, data requirements, and methodological aspects." *J. Economic Literature*, 59(2), 391-425. — Canonical definitions; more pre-period = lower bias.
- [Eggers et al., 2024] Eggers, A.C., Tunon, G., Dafoe, A. "Placebo tests for causal inference." *AJPS*, 68(3), 1106-1121. — FPR = 5% defines well-calibrated; identified null-hacking threat.
- [Athey & Imbens, 2017] Athey, S., Imbens, G. "The state of applied econometrics: causality and policy evaluation." *J. Economic Perspectives*, 31(2), 3-32. — Endorses placebo analyses as standard robustness requirement.
- [Linden, 2018] Linden, A. "Using permutation tests to enhance causal inference in interrupted time series analysis." *J. Evaluation in Clinical Practice*, 24(3), 496-501. PMID: 29460383.

## Interrupted Time Series

- [Lopez Bernal et al., 2017] Lopez Bernal, J.A. et al. "Interrupted time series regression for the evaluation of public health interventions: a tutorial." *International Journal of Epidemiology*, 46(1), 348-355. — "No fixed limits" on minimum observations.
- [Penfold & Zhang, 2013] Penfold, R.B., Zhang, F. "Use of interrupted time series analysis in evaluating health care quality improvements." *Academic Pediatrics*, 13(6 Suppl), S38-S44. — Minimum 8 observations for segmented OLS (not BSTS).

## Posterior Probability Communication

- [Makowski et al., 2019] Makowski, D. et al. "Indices of effect existence and significance in the Bayesian framework." *Frontiers in Psychology*, 10, Article 2767. — Formalized "Probability of Direction" (pd) metric.
- [Marsman & Wagenmakers, 2016] Marsman, M., Wagenmakers, E.-J. "Three insights from a Bayesian interpretation of the one-sided p value." *Educational and Psychological Measurement*, 77(3), 529-539. — Under uniform priors, one-sided p = posterior mass below zero.
- [Gelman & Yao, 2021] Gelman, A., Yao, Y. "Holes in Bayesian statistics." *Journal of Physics G*, 48(1). arXiv:2002.06467. — Pr(effect > 0) overstates certainty with flat priors.
- [Muehlemann et al., 2023] Muehlemann, N. et al. "A tutorial on modern Bayesian methods in clinical trials." *Therapeutic Innovation & Regulatory Science* (PMC10117244). — Recommends posterior probabilities over p-values for non-technical audiences.
- [Ruberg, 2021] Ruberg, S.J. "Detente: a practical understanding of p values and Bayesian posterior probabilities." *Clinical Pharmacology & Therapeutics*, 109(6), 1489-1498 (PMC8246739).
