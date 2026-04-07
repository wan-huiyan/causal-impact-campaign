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

## Posterior Predictive p-values (Bayesian Model Criticism)

- [Gelman, Meng & Stern, 1996] Gelman, A., Meng, X.-L., Stern, H. "Posterior predictive assessment of model fitness via realized discrepancies." *Statistica Sinica*, 6, 733–807. — The canonical framework for posterior predictive checks. The `tfcausalimpact` `compute_p_value` function in `causalimpact/inferences.py` is an instance of this framework (the package documentation does not say so).
- [Robins, van der Vaart & Ventura, 2000] Robins, J.M., van der Vaart, A., Ventura, V. "Asymptotic distribution of p values in composite null models." *JASA*, 95(452), 1143–1156. DOI: 10.1080/01621459.2000.10474310. — Foundational result that posterior predictive p-values are conservative under composite nulls (cluster around 0.5, not uniformly distributed). Cite this when a methodology write-up treats a BSTS p-value as a frequentist Type-I error rate.
- [Bayarri & Berger, 2000] Bayarri, M.J., Berger, J.O. "P-values for composite null models." *JASA*, 95(452), 1127–1142. — Companion paper proposing partial-posterior-predictive and conditional-predictive p-values that ARE asymptotically uniform under the null. `tfcausalimpact` does not implement these.
- [Phipson & Smyth, 2010] Phipson, B., Smyth, G.K. "Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn." *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39. PubMed: 21044043. arXiv: 1603.05766. — Justification for the `(k+1)/(N+1)` Monte-Carlo correction used by `tfcausalimpact` internally and by most permutation aggregators.

## Randomization Inference & Fisher's Sharp Null

- [Fisher, 1935] Fisher, R.A. *The Design of Experiments.* Edinburgh: Oliver & Boyd. — Historical origin of randomization inference.
- [Imbens & Rubin, 2015] Imbens, G.W., Rubin, D.B. *Causal Inference for Statistics, Social, and Biomedical Sciences.* Cambridge University Press. DOI: 10.1017/CBO9781139025751. — Chapter 5, "Fisher's Exact P-Values for Sharp Null Hypotheses", is the textbook treatment of randomization inference under the sharp null `H_0: τ_t = 0 for all t in post`. The right framework for thinking about any date-shuffled randomization test.

## Specification Curve & Multiverse Analysis

- [Simonsohn, Simmons & Nelson, 2020] Simonsohn, U., Simmons, J.P., Nelson, L.D. "Specification curve analysis." *Nature Human Behaviour*, 4(11), 1208–1214. DOI: 10.1038/s41562-020-0912-z. — The canonical SCA paper. Three-step procedure: (1) enumerate defensible specs, (2) display the curve sorted by effect size with a descriptor matrix, (3) conduct a bootstrap-based joint inference test. Most projects implement 1-2 only; claim "Simonsohn-validated" only if joint inference is run.
- [Steegen, Tuerlinckx, Gelman & Vanpaemel, 2016] Steegen, S., Tuerlinckx, F., Gelman, A., Vanpaemel, W. "Increasing transparency through a multiverse analysis." *Perspectives on Psychological Science*, 11(5), 702–712. DOI: 10.1177/1745691616658637. — The data-processing-decision counterpart to SCA. Varying data-processing choices (exclusions, recoding, transformations) rather than modelling choices.
- [Gelman & Loken, 2013] Gelman, A., Loken, E. "The garden of forking paths: Why multiple comparisons can be a problem, even when there is no 'fishing expedition' or 'p-hacking' and the research hypothesis was posited ahead of time." Working paper, Columbia Department of Statistics. [Columbia PDF](https://sites.stat.columbia.edu/gelman/research/unpublished/p_hacking.pdf). — Conceptual umbrella: even an honest single analysis is implicitly multiple comparisons because of the data-contingent decisions a researcher *would have* made on a different sample. Especially relevant to "find a better spec via the spec curve" failure modes.

## Bayesian Inference Theory & Type S/M Errors

- [Gelman & Carlin, 2014] Gelman, A., Carlin, J. "Beyond power calculations: Assessing Type S (sign) and Type M (magnitude) errors." *Perspectives on Psychological Science*, 9(6), 641–651. DOI: 10.1177/1745691614551642. — Argues that for low-power studies, Type-S (sign) and Type-M (magnitude) errors matter more than Type-I. Highly relevant for short-window single-unit causal-impact analyses.
- [Blei, Kucukelbir & McAuliffe, 2017] Blei, D.M., Kucukelbir, A., McAuliffe, J.D. "Variational Inference: A Review for Statisticians." *JASA*, 112(518), 859–877. DOI: 10.1080/01621459.2017.1285773. — Canonical VI review. Establishes that variational inference underestimates posterior variance, which makes CIs too narrow and posterior predictive p-values too small. Use as the citation when recommending HMC over VI for headline results.

## Forecasting & Time Series Cross-Validation

- [Hyndman & Athanasopoulos, 2021] Hyndman, R.J., Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. OTexts. Chapter 5.10, "Time series cross-validation". [https://otexts.com/fpp3/tscv.html](https://otexts.com/fpp3/tscv.html). — The standard reference for rolling-origin cross-validation. Important caveat: H&A treats rolling-origin CV as the standard for *forecast accuracy evaluation*, not as a placebo significance test. When a methodology write-up cites H&A to justify a "backtest as primary placebo" framing, it's a category error.
