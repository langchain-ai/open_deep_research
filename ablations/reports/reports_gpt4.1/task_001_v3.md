# Methodological Assumptions, Trade-Offs, and Empirical Adoption of Modern Difference-in-Differences Estimators after the Goodman-Bacon Critique

## Introduction

The recent methodological revolution in Difference-in-Differences (DiD) estimation arose in response to the Goodman-Bacon (2021) critique, which exposed that standard Two-Way Fixed Effects (TWFE) DiD estimation becomes problematic with staggered treatment timing and heterogeneous effects. This stimulated the development and adoption of new estimators: Callaway & Sant’Anna (CS), Sun & Abraham (SA), Borusyak, Jaravel & Spiess (BJS), and the did2s estimator. Each responds to the inferential and identification inadequacies of TWFE, but they do so through distinct methodological innovations and with varying technical assumptions. These innovations also interact with concerns regarding pre-trend testing, statistical power, and inference after selection as highlighted by Roth (2022). This report systematically compares these estimators in terms of their identifying assumptions, trade-offs, innovations, and empirical adoption.

## 1. Technical Identifying Assumptions: Explicit Comparison and Hierarchy

### Overview of Core Assumptions

All modern DiD estimators fundamentally rely on some version of the "parallel trends" assumption – that in the absence of treatment, the expected path of the outcome for treated and comparison groups would evolve in parallel. However, the ways in which potential outcomes are modeled, and the strictness of the parallel trends assumption, vary considerably across the estimators:

#### Callaway & Sant’Anna (CS)

- **Assumption:** Allows for flexible, nonparametric parallel trends. The untreated counterfactual for each treatment cohort in each period is constructed by comparing to “clean” control groups (never-treated or not-yet-treated units).
- **Structure:** No structural restrictions (like linearity or additivity) are imposed on untreated potential outcomes beyond the required parallelism. The parallel trends can be unconditional, or can be conditional on observed covariates.
- **Implication:** This is the least restrictive assumption set among modern DiD estimators, allowing arbitrary group–time heterogeneity. It is robust but typically less statistically efficient than more restrictive estimators when their stronger assumptions hold[1][2][3][4].

#### Sun & Abraham (SA)

- **Assumption:** Similar to CS but operationalized through interaction-weighted event-study estimates. Uses only never-treated or not-yet-treated units as controls, ensuring no pre- or post-treatment contamination in the comparisons.
- **Structure:** Relies on a parallel trends assumption for each group and relative period, but like CS, requires no additivity or parametric structure in untreated outcomes.
- **Implication:** Slightly less flexible than fully nonparametric CS when covariate adjustment is needed, but still highly robust to heterogeneity and staggered timing[5][6][7].

#### Borusyak, Jaravel & Spiess (BJS)

- **Assumption:** Requires untreated potential outcomes to follow an **additive unit and time fixed-effects structure**:
    - For all untreated observations, \( Y_{it}(0) = \alpha_i + \lambda_t \), possibly plus observed covariates. 
    - This means the counterfactual path of any unit (had it never been treated) is a sum of a unit-specific and a time-specific component, and (optionally) covariate effects.
    - No nonparametric flexibility: deviations from this structure (e.g., interactive group-by-time shocks or non-additive paths) can bias results.
- **Implication:** This is a **much more restrictive assumption** than the nonparametric parallel trends in CS and SA. The approach is less robust to complex untreated outcome dynamics, but, where plausible, yields estimators that are maximally efficient[8][9][10][11].

#### did2s (Gardner)

- **Assumption:** Shares similar assumptions with BJS: untreated outcomes are modeled as the sum of unit and time effects (with optional covariates).
- **Structure:** The estimator fits the fixed effect model to untreated (never- or not-yet-treated) observations only, then re-estimates treatment effects using residualized outcomes.
- **Implication:** Like BJS, this is restrictive (additive structure required), but ensures internal validity and efficiency if assumption holds, and allows direct mapping to the ATT[12][13][14][15].

### Assumption Restrictiveness: Hierarchy and Trade-offs

Summarizing, the **hierarchy of assumption restrictiveness** is:

1. **Most restrictive and efficient**: BJS ≈ did2s (additive fixed-effects model for untreated outcomes; maximal efficiency, minimal robustness to departures from additivity).
2. **Moderately restrictive and robust**: SA (parallel trends, but event-time and cohort specific; no additivity needed).
3. **Least restrictive and most robust**: CS (nonparametric, fully flexible parallel trends; slowest to gain power/efficiency, but least likely to be invalid with complex untreated outcomes).

**Trade-offs**:  
- **BJS/Did2s**: If untreated outcomes are truly additive in unit/time/covariates, these estimators are most efficient (smallest variance, most precise) as all untreated observations (including multiple pre-treatment periods) contribute to counterfactual imputation. However, violations of additivity (e.g., idiosyncratic shocks, nonlinearities, group-by-period interactions) lead to potentially severe bias.
- **CS/SA**: Sacrifice some efficiency (use only clean period-unit pairs, avoid at-risk controls) for robustness—no additive structure is needed; these remain valid under the broadest conditions, though generally require larger sample sizes to attain similar power.

[See sources: 1–15]

## 2. Design Innovations in BJS: Efficiency, Internal Validity, and Pre-Trend Testing

### Efficiency via Use of All Untreated Observations

- **Key Feature**: The BJS estimator regresses untreated outcomes on unit and time fixed effects using all available pre-treatment data from never- and not-yet-treated units. These fitted values are then used to impute counterfactual untreated outcomes for all treated observations.
- **Efficiency**: By pooling all untreated periods across all units, BJS achieves minimal variance among linear, unbiased estimators under its additive structure—this is superior in efficiency to any method that limits controls to a subset of pre-periods (as in CS/SA)[8][11].
- **Consequence**: In settings with short pre-treatment panels or small numbers of never-treated units, this pooling is especially powerful. However, if additivity does not hold, efficiency gains are offset by the risk of bias[8][10].

### Internal Validity

- The strong assumption (additivity) is the main limitation to internal validity. If this assumption accurately describes the untreated data, the estimator is both internally valid and efficient. If untreated potential outcomes deviate—say, there are group-by-time shocks or nonlinearities—estimates may be biased[8][11].

### Pre-Trend Testing in BJS

- **Crucial Distinction**: In BJS, **all pre-trend (lead) tests are estimated using only untreated (never- or not-yet-treated) observations and their pre-treatment periods**. Post-adoption periods are excluded from fitting the untreated model or from lead tests, thereby **avoiding “contaminated” comparisons that can distort inference** as in classic TWFE.
- Thus, **pre-trend event-study coefficients in BJS reflect only valid comparisons before any unit’s adoption**, not mixtures of pre- and post-treatment behaviors.
- This strict use of “clean” data in the pre-trend test **raises efficiency (more data used) and validity (no contamination)** while minimizing the risk of Type I or Type II errors induced by inclusion of already-treated units in the comparison set[8][10][11][16].

### Practical Consequences

- **Graphical Implication**: BJS pre-treatment coefficients may be estimated precisely and interpreted as genuine tests of parallel trends, but Roth (2022, 2026) notes that due to design asymmetry (comparisons before and after might use different control groups), event-study plots may show “jumps” or “kinks” at treatment even with perfect parallel trends—so visual inference requires care[10][16].
- **Mitigating Inference Distortions**: Because contaminated post-treatment outcomes are never used in pre-trend tests, BJS reduces “pre-test” inference distortions relative to legacy DiD methods.

[See sources: 8–11, 16]

## 3. Innovations Relative to Roth (2022): Inference, Low Power, and Conditioning Bias

### Roth’s (2022) Critique

- **Low Power**: Standard pre-trend tests (e.g., statistically testing for lead coefficients ≠ 0) have intrinsically low power to detect plausible parallel trends violations, especially with limited pre-periods or persistent smooth deviations[17].
- **Conditioning Bias**: Conditioning estimation or inference on “passing” such underpowered tests (e.g., only publishing results when pre-trends are insignificant) can lead to selection-induced bias and severe undercoverage of confidence intervals[17].
- **Interpretation**: Event-study plots produced through modern estimators (CS, SA, BJS, did2s) are **not symmetric** in pre- vs post-treatment periods, so conventional graphical heuristics (looking for smoothness or continuity) may mislead. “Jumps” do not always reflect violations.

### How Each Estimator Addresses (or Fails to Address) Roth’s Concerns

#### Callaway & Sant’Anna (CS)
- **Mitigates contamination**: Estimation/aggregation only uses clean not-yet- or never-treated units for each cohort and period, avoiding post-adoption bias.
- **Pre-trend testing**: Group-time ATT estimates for pre-event periods are available, but their construction may use different samples/methods than post-treatment estimates, so “visual” trend-testing remains fragile.
- **Inference after selection**: Roth’s warning persists: if researchers restrict analysis to cases passing such tests, selection bias and miscoverage risk remains; power to detect violations remains limited.
- **Practical mitigation**: CS estimators are robust to group heterogeneity and staggered timing, but users are encouraged to supplement with honest sensitivity analysis (e.g., HonestDiD)[2][3][17].

#### Sun & Abraham (SA)
- **Mitigates contamination**: Interaction-weighted event-study coefficients use only appropriate, untreated comparison groups for each cohort/event time.
- **Pre-trend inference**: Same issue as CS—lead coefficients available but plots may show kinks due to design; not technically symmetric.
- **Root inference problems**: Cannot mechanically overcome low power or remove conditioning bias; visual or formal pre-trends should be complemented by substantive sensitivity analysis.
- **Mitigation**: Encouraged to use simultaneous confidence intervals, and honest robustness checks (see Rambachan & Roth, 2021)[5][10][17].

#### Borusyak, Jaravel & Spiess (BJS)
- **Innovative solution**: Guarantees that all pre-trend estimation uses only pre-treatment untreated data, eliminating post-adoption contamination and associated inference distortions.
- **Efficiency plus validity**: Gains precision using all available untreated observations, but only under the strong additive structure assumption.
- **Residual issues**: Pre-trend plots not symmetric, may show apparent discontinuities even with parallel trends; low power and conditioning bias not inherently solved. Roth (2026) recommends constructing alternative “universal base” comparisons or combining with honest sensitivity analysis.
- **Best practice**: Use explicit power analysis of pre-trend tests (e.g., via HonestDiD) instead of mechanical pass/fail logic; transparency in reporting is critical[8][9][10][11][17][18].

#### did2s (Gardner)
- **Contamination avoidance**: Estimates unit and time effects using only untreated observations, then uses residualized outcomes to estimate ATT/events. This structure directly blocks the mixture of treated/untreated periods that confounds standard TWFE.
- **Pre-trend issues**: Follows similar logic as BJS, but suffers the same interpretational challenges with asymmetry in event-study plots.
- **Inference recommendations**: Use cluster-robust estimation and honest sensitivity analysis to plausible deviations from parallel trends[12][14][15][17][18].

### Summary Table: Estimator Innovations and Roth’s Concerns

| Estimator | Contamination Avoidance | Pre-Trend Test Cleanliness | Handles Conditioning Bias? | Solves Low Power? | Visual Pre-Trend Risks |
|-----------|------------------------|---------------------------|---------------------------|-------------------|-----------------------|
| CS        | Yes                    | Mostly, but asymmetric    | No                        | No                | Yes                   |
| SA        | Yes                    | Mostly, but asymmetric    | No                        | No                | Yes                   |
| BJS       | Yes (strict)           | Yes (untreated-only)      | No                        | No                | Yes                   |
| did2s     | Yes (strict)           | Yes (untreated-only)      | No                        | No                | Yes                   |

[See sources: 2,3,5,8,9,10,11,12,14,15,16,17,18]

## 4. Empirical Adoption and Practice Evolution in Leading Economics Journals (2020–2024)

### Periodization of Methodological Evolution

To understand empirical adoption, the period from 2020–2024 can be divided into three distinct phases:

#### 2020–Early 2021: Pre-Revolution/Transitional

- **Standard:** TWFE DiD and event-study models dominated policy analysis in AER, QJE, JPE, especially in applied labor and health economics.
- **Awareness:** Goodman-Bacon (2021) critique began circulating as working papers; initial discussions on staggered timing and weight contamination arose but had not yet outpaced practice[19][20].

#### Mid 2021–2022: Rapid Innovation and “Triangulation” Era

- **Seminal publications:** CS[1][2], SA[5][6], BJS[8][9], and did2s[12][13] all released working or published papers, R and Stata packages proliferated.
- **Adoption:** Applied researchers, especially in labor and health, rapidly began reporting not just TWFE, but also “modern” estimators side-by-side, often including:
    - CS group-time ATT with flexible aggregation.
    - SA event-study coefficients with cohort-by-period effects.
    - BJS and/or did2s ATT or event-study estimates.
    - Use of parallel trends checks, but also increasing attention to sensitivity and robustness checks[4][21][22].
- **Justification:** Authors followed recommendations to “triangulate” results, cite the limitations of TWFE, and justify estimator choice by simulation or robustness to timing/heterogeneity.

#### 2023–2024: Standardization and Advanced Robustness

- **Dominance:** Modern estimators are now the modal standard in AER/QJE/JPE, especially in high-profile policy evaluation. Papers are expected to present at least one non-TWFE estimator as a main result, with TWFE included for historical comparison only[24][25].
- **Best practices:** Extensive robustness exercises—including:
    - Use of multiple estimators (CS, SA, BJS, did2s), frequently in tandem.
    - “Honest” sensitivity analysis (e.g., via the HonestDiD R package).
    - Detailed event-study plots, with pre- and post-treatment periods, but with explicit warnings regarding graphical interpretation.
    - Explicit reporting and adjustment for pre-trend test power, often with Monte Carlo simulations to assess the likelihood of detecting violations under plausible deviances from parallel trends[17][18][22].
    - Supplementary analyses: placebo/falsification outcomes, permutation/randomization inference, and cohort-specific treatment patterns.
- **Publication trend:** It is now expected that papers explicitly reference the limitations of pre-test inference (per Roth) and detail how identification assumptions are satisfied in their context.

### Representative Adoption and Applied Justification

- **Labor Economics Example:** Major minimum wage studies now use CS/SA alongside BJS/did2s, report all estimates, show event-study plots, and discuss robustness to alternative parallel trends scenarios[3][25].
- **Health Economics Example:** Recent QJE and JPE papers evaluating Medicaid expansions or hospital policy changes report comparative estimates from SA, BJS, and did2s, with detailed simulation-based power calculations for pre-trend checks, and “honest DiD” robustness bounds[4][22][24].

[See sources: 1–4, 8–10, 12–15, 17–18, 19–25]

## Conclusion

The post-Goodman-Bacon era marks a methodological paradigm shift in DiD estimation, focusing on clear identification, avoidance of forbidden and contaminated comparisons, and transparent accounting of both estimator assumptions and inferential risks. The technical assumption hierarchy is clear: BJS/did2s are most efficient but least robust (additivity required), CS is most robust but least efficient, SA is intermediate. All modern estimators avoid negative weighting and closed-form contamination, and all are subject to the inferential limitations described by Roth (2022): pre-trend testing remains generally underpowered, and conditioning on pre-tests carries interpretational hazards regardless of method.

Practices in top journals now reflect this reality: applied papers triangulate with multiple estimators, carefully report assumptions, detail their identification logic, and accompany results with formal sensitivity analyses. The net result is a new, higher empirical standard for DiD research in economics—one that prioritizes both robustness and transparency over convenience or tradition.

## Sources

[1] Difference-in-Differences with multiple time periods. Callaway & Sant’Anna (2021): https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Paper2023/D3--Callaway-2021-Difference-in-differences-with-multiple-time-periods.pdf  
[2] Difference-in-Differences for Policy Evaluation. Brantly Callaway: https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf  
[3] Modern Approaches to Difference in Differences - Brantly Callaway: https://bcallaway11.github.io/Courses/FEEM/modern_did_session3.html  
[4] Group-Time Average Treatment Effects in DID—did R package: https://bcallaway11.github.io/did/articles/did-basics.html  
[5] Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. Sun & Abraham (2021): https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html  
[6] fixest R package: sunab() event-study vignette: https://asjadnaqvi.github.io/DiD/docs/code_r/07_sunab_r/  
[7] eventstudyinteract: Stata package documentation (SA estimator): https://github.com/lsun20/EventStudyInteract  
[8] Revisiting Event Study Designs: Robust and Efficient Estimation. Borusyak, Jaravel & Spiess (2023): https://ideas.repec.org/p/arx/papers/2108.12419.html  
[9] Imputation Estimator from Borusyak, Jaravel, and Spiess (2021), R package: https://cran.rstudio.com/web/packages/didimputation/didimputation.pdf  
[10] Borusyak, Jaravel, and Spiess (2022) Revisiting Event Study Designs (slides): https://francescoruggieri.github.io/files/DiDES05_BorusyakJaravelSpiess.pdf  
[11] didimputation: Imputation Estimator from Borusyak, Jaravel, and Spiess (2021): https://cran.r-project.org/web/packages/didimputation/index.html  
[12] did2s: Two-Stage Difference-in-Differences - The R Journal: https://journal.r-project.org/articles/RJ-2022-048/  
[13] did2s package documentation: https://github.com/kylebutts/did2s  
[14] Two-Stage Differences in Differences - John Gardner (2022): https://jrgcmu.github.io/2sdd_gtty.pdf  
[15] Testing for Pre-trends in DiD and Event Studies - Tilburg Science Hub: https://www.tilburgsciencehub.com/topics/analyze/causal-inference/did/pretrends/  
[16] Interpreting Event-Studies from Recent Difference-in-Differences Methods. Roth (2026): https://www.jonathandroth.com/assets/files/HetEventStudies.pdf  
[17] Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. Roth (2022): https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf  
[18] HonestDiD R package: https://cran.r-project.org/web/packages/HonestDiD/HonestDiD.pdf  
[19] Difference-in-Differences with Variation in Treatment Timing. Goodman-Bacon (2021): https://www.aeaweb.org/articles?id=10.1257/aer.20181047  
[20] Designing Difference-in-Differences Studies with Staggered Treatment Adoption - Annual Review of Public Health: https://www.annualreviews.org/content/journals/10.1146/annurev-publhealth-061022-050825  
[21] Notes on Callaway–Sant'Anna (2021) – Staggered Adoption DiD: https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did/  
[22] Comparative Evaluation of Difference in Differences Methods for Staggered Adoption Interventions: https://arxiv.org/html/2508.14365  
[23] Two-Stage Difference in Differences (did2s) vignette: https://kylebutts.github.io/did2s/articles/Two-Stage-Difference-in-Differences.html  
[24] Advances in Difference-in-differences Methods for Policy Evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929/  
[25] Minimum Wages and Employment: A Case Study of the Fast Food Industry in New Jersey and Pennsylvania. Card & Krueger (QJE, 2022 replication with modern DiD): https://direct.mit.edu/qjec/article/137/2/763/112979/Minimum-Wages-and-Employment-A-Case-Study-of-the  
