# Graph Report - .  (2026-07-08)

## Corpus Check
- 22 files · ~53,975 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 128 nodes · 165 edges · 15 communities (12 shown, 3 thin omitted)
- Extraction: 80% EXTRACTED · 19% INFERRED · 1% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Causal Methods & Explorer|Causal Methods & Explorer]]
- [[_COMMUNITY_Demo Case 20mph Policy Report|Demo Case: 20mph Policy Report]]
- [[_COMMUNITY_Pitfalls & Case Study|Pitfalls & Case Study]]
- [[_COMMUNITY_Validation Tests & Citations|Validation Tests & Citations]]
- [[_COMMUNITY_Manifest Consistency Tests|Manifest Consistency Tests]]
- [[_COMMUNITY_Skill Core & Methodology|Skill Core & Methodology]]
- [[_COMMUNITY_Robustness (SCAPlacebo) & Framing|Robustness (SCA/Placebo) & Framing]]
- [[_COMMUNITY_Prob-of-Direction & p-value Theory|Prob-of-Direction & p-value Theory]]
- [[_COMMUNITY_package.json manifest|package.json manifest]]
- [[_COMMUNITY_Marketplace manifest|Marketplace manifest]]
- [[_COMMUNITY_Eval-Suite Integrity Test|Eval-Suite Integrity Test]]
- [[_COMMUNITY_Trigger Classification Test|Trigger Classification Test]]
- [[_COMMUNITY_Signal-to-Noise Ratio Assessment|Signal-to-Noise Ratio Assessment]]
- [[_COMMUNITY_Abadie (2021)|Abadie (2021)]]
- [[_COMMUNITY_Post-Promo Persistence|Post-Promo Persistence]]

## God Nodes (most connected - your core abstractions)
1. `BSTS (tfcausalimpact)` - 10 edges
2. `Causal Impact Campaign Skill` - 9 edges
3. `Eval Suite (Assertion->Source Mapping)` - 9 edges
4. `Covariate Safety Audit` - 8 edges
5. `Specification Curve Analysis (SCA)` - 8 edges
6. `Pre-Period Placebo Test / FPR Calibration` - 8 edges
7. `Interactive Causal Impact Report (Retail Delivery Promo)` - 8 edges
8. `Policy Impact Report Card (20 mph Speed Limit Rollout)` - 8 edges
9. `Multi-Method Analysis` - 7 edges
10. `CausalPy` - 6 edges

## Surprising Connections (you probably didn't know these)
- `CI Test Workflow (npm test on Node 20/22)` --references--> `Causal Impact Campaign Skill`  [AMBIGUOUS]
  .github/workflows/test.yml → plugins/causal-impact-campaign/SKILL.md
- `224-Spec Robustness Grid` --conceptually_related_to--> `Specification Curve Analysis (SCA)`  [INFERRED]
  docs/case-study.md → plugins/causal-impact-campaign/SKILL.md
- `Demo Output: 20mph Speed Limit Policy Impact Card` --semantically_similar_to--> `Interactive HTML Explorer`  [INFERRED] [semantically similar]
  docs/demo-output.html → plugins/causal-impact-campaign/SKILL.md
- `Companion & Related Skills` --references--> `Causal Impact Campaign Skill`  [EXTRACTED]
  README.md → plugins/causal-impact-campaign/SKILL.md
- `Comparison vs Alternatives` --conceptually_related_to--> `Multi-Method Analysis`  [INFERRED]
  README.md → plugins/causal-impact-campaign/SKILL.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Multi-Method Robustness Ensemble** — causal_impact_campaign_skill_multi_method_analysis, causal_impact_campaign_skill_bsts_tfcausalimpact, causal_impact_campaign_skill_causalpy, causal_impact_campaign_skill_rdit, causal_impact_campaign_skill_conformal_ci, references_extension_analyses_prophet_cv [EXTRACTED 1.00]
- **Three Tests, Three Different Questions (Validation Battery)** — causal_impact_campaign_skill_tfci_pvalue_formula, causal_impact_campaign_skill_rolling_backtest_rank, causal_impact_campaign_skill_date_shuffled_randomization, causal_impact_campaign_skill_specification_curve_analysis, causal_impact_campaign_skill_pre_period_placebo_fpr [EXTRACTED 1.00]
- **Step 6b Extension Analyses** — references_extension_analyses_effect_decomposition, references_extension_analyses_channel_split, references_extension_analyses_post_promo_persistence, references_extension_analyses_weather_covariate, references_extension_analyses_prophet_cv, references_extension_analyses_sale_autodetection [EXTRACTED 1.00]
- **Dual-Method Agreement (BSTS + RDiT cross-validation)** — docs_demo_output_dual_method_agreement, docs_demo_output_tfcausalimpact_bsts, docs_demo_output_causalpy_rdit [EXTRACTED 1.00]
- **Exclude Contaminating Covariates/Periods to Sharpen the Causal Estimate** — docs_demo_interactive_paid_sessions, docs_demo_interactive_enhanced_model_spec, docs_demo_output_construction_zones, docs_demo_output_modelling_decisions [INFERRED 0.85]
- **Retail Delivery Promo Causal Finding** — docs_demo_interactive_delivery_promo, docs_demo_interactive_incremental_revenue, docs_demo_interactive_bsts, docs_demo_interactive_counterfactual [EXTRACTED 1.00]

## Communities (15 total, 3 thin omitted)

### Community 0 - "Causal Methods & Explorer"
Cohesion: 0.16
Nodes (18): BSTS (tfcausalimpact), CausalPy, Conformal Prediction Intervals, Fake p-values in Multi-Method Pipelines, Interactive HTML Explorer, Multi-Method Analysis, RDiT (Regression Discontinuity in Time), VI Stochasticity Warning (+10 more)

### Community 1 - "Demo Case: 20mph Policy Report"
Cohesion: 0.26
Nodes (16): Interactive Causal Impact Report (Retail Delivery Promo), BSTS Significance (p=0.039), Counterfactual Analysis (What would have happened without the promo), Retail Delivery Promo Campaign, Enhanced Model Spec (excl. Xmas & paid_sessions), £XXXk Incremental Revenue Estimate, Model Specification Selector, Contaminated paid_sessions Covariate (excluded) (+8 more)

### Community 2 - "Pitfalls & Case Study"
Cohesion: 0.15
Nodes (13): Identification Assumptions, Masking High-Variance Pre-Periods, Multi-Modal Holiday Intensity (xmas_intensity), Data-Prep Zero-Injection Trap, Issue #51 Mask Zero-Injection Fix, Case Study: Path to a Validated Result, 224-Spec Robustness Grid, Techniques That Didn't Work (and Why) (+5 more)

### Community 3 - "Validation Tests & Citations"
Cohesion: 0.19
Nodes (13): Pre-Period Placebo Test / FPR Calibration, Rolling-Backtest Empirical-Null Rank, Abadie et al. (2010), Abadie et al. (2015), Athey & Imbens (2017), Eggers et al. (2024), Gils et al. (2022), Hyndman & Athanasopoulos (2021) (+5 more)

### Community 4 - "Manifest Consistency Tests"
Cohesion: 0.15
Nodes (9): __dirname, evalSuitePath, files, marketplaceJsonPath, packageJsonPath, pluginJsonPath, pluginsRootForSkills, ROOT (+1 more)

### Community 5 - "Skill Core & Methodology"
Cohesion: 0.18
Nodes (12): Covariate Safety Audit, Causal Impact Campaign Skill, Subtract Before You Add (Meta-Lesson), The Journey: p~0.22 to p<0.05, Companion & Related Skills, Covariate Correlation Benchmarks, Leave-One-Out Covariate Sensitivity, Scott & Varian (2014) (+4 more)

### Community 6 - "Robustness (SCA/Placebo) & Framing"
Cohesion: 0.31
Nodes (9): Claim Framing Guide, Date-Shuffled Randomization Test, Diagnose, Don't Demote, Specification Curve Analysis (SCA), Fisher (1935), Gelman & Carlin (2014), Simonsohn, Simmons & Nelson (2020), Open-Meteo Weather API (+1 more)

### Community 7 - "Prob-of-Direction & p-value Theory"
Cohesion: 0.25
Nodes (8): Probability of Direction (prob_positive), tfcausalimpact p-value Formula, Blei, Kucukelbir & McAuliffe (2017), Gelman, Meng & Stern (1996), Lakens (2017), Makowski et al. (2019), Marsman & Wagenmakers (2016), Phipson & Smyth (2010)

### Community 8 - "package.json manifest"
Cohesion: 0.25
Nodes (7): description, name, private, scripts, test, type, version

### Community 9 - "Marketplace manifest"
Cohesion: 0.29
Nodes (6): description, name, owner, name, plugins, $schema

### Community 10 - "Eval-Suite Integrity Test"
Cohesion: 0.50
Nodes (3): __dirname, evalSuitePath, ROOT

### Community 11 - "Trigger Classification Test"
Cohesion: 0.50
Nodes (3): __dirname, evalSuitePath, ROOT

## Ambiguous Edges - Review These
- `Causal Impact Campaign Skill` → `CI Test Workflow (npm test on Node 20/22)`  [AMBIGUOUS]
  .github/workflows/test.yml · relation: references

## Knowledge Gaps
- **57 isolated node(s):** `$schema`, `name`, `description`, `name`, `plugins` (+52 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Causal Impact Campaign Skill` and `CI Test Workflow (npm test on Node 20/22)`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `Causal Impact Campaign Skill` connect `Skill Core & Methodology` to `Causal Methods & Explorer`, `Validation Tests & Citations`, `Robustness (SCA/Placebo) & Framing`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `Specification Curve Analysis (SCA)` connect `Robustness (SCA/Placebo) & Framing` to `Pitfalls & Case Study`, `Validation Tests & Citations`, `Skill Core & Methodology`?**
  _High betweenness centrality (0.081) - this node is a cross-community bridge._
- **Why does `Multi-Method Analysis` connect `Causal Methods & Explorer` to `Skill Core & Methodology`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `BSTS (tfcausalimpact)` (e.g. with `numpy Version Conflict (tfcausalimpact vs CausalPy)` and `Prophet Cross-Validation`) actually correct?**
  _`BSTS (tfcausalimpact)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `Covariate Safety Audit` (e.g. with `Covariate Correlation Benchmarks` and `Leave-One-Out Covariate Sensitivity`) actually correct?**
  _`Covariate Safety Audit` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `$schema`, `name`, `description` to the rest of the system?**
  _57 weakly-connected nodes found - possible documentation gaps or missing edges._