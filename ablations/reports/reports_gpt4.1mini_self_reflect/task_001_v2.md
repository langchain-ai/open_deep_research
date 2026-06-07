# Methodological Tensions and Modern Solutions in Difference-in-Differences Estimation Post-Goodman-Bacon (2021)

## Introduction

Difference-in-Differences (DiD) estimation is a widely-used econometric technique for causal inference in observational panel data settings. However, the canonical Two-Way Fixed Effects (TWFE) DiD estimator encounters fundamental methodological challenges in the common practice of staggered treatment adoption—that is, when different units receive treatment at different times. Goodman-Bacon’s (2021) seminal critique revealed critical biases arising from traditional TWFE in such scenarios, especially when treatment effects are heterogeneous and dynamic over groups and time.

This report systematically investigates these methodological tensions and provides a comprehensive analysis of three leading alternative DiD estimators developed to address these issues:

- Callaway and Sant'Anna’s (2019, 2021) two-stage aggregation method.
- Sun and Abraham’s (2020, 2021) interaction-weighted event-study estimators.
- Borusyak, Jaravel, and Spiess’s (2021, 2024) imputation-based estimators.

The analysis compares these approaches with respect to their handling of heterogeneous treatment effects and dynamic timing, key identification assumptions—including parallel trends (conditional and unconditional), treatment effect homogeneity, and anticipation effects—and empirical adoption in labor and health economics articles published in **American Economic Review (AER)**, **Quarterly Journal of Economics (QJE)**, and **Journal of Political Economy (JPE)** between 2020 and 2024. Lastly, the report examines how these modern estimators incorporate or respond to Roth’s (2022) critique of traditional pre-trend testing, highlighting improvements in diagnostic and sensitivity analyses.

---

## Goodman-Bacon (2021) Critique: Methodological Tensions in TWFE Staggered Adoption DiD

Goodman-Bacon (2021) critically examined the TWFE DiD estimator commonly used in settings with staggered adoption. The key insights include:

- **Decomposition of TWFE Estimator:** The TWFE regression coefficient can be exactly decomposed into a weighted average of all possible two-group, two-period DiD estimators embedded in the data. These pairs can include “clean” untreated vs treated groups, but also less appropriate comparisons like earlier treated groups serving as controls to later treated groups [6][19].

- **Negative and Nonconvex Weights from Heterogeneous Effects:** When treatment effects vary over groups or time—i.e., heterogeneous or dynamic effects—some comparisons receive negative weights, meaning that certain group-time contrasts subtract from the overall estimate. This leads to “contamination bias,” where TWFE estimates may be biased and counterintuitive, even reversing sign relative to all underlying effects [6][19].

- **"Forbidden Comparisons":** TWFE implicitly pits already-treated units against not-yet-treated units or treated units with different treatment timing, violating the parallel trends assumption because these groups differ in post-treatment experiences [6].

- **Strained Identification Assumptions:** The validity of TWFE relies on both the conventional parallel trends assumption *and* treatment effect homogeneity. Violations of either—common in policy evaluation—invalidate causal interpretations [6].

- **Implications for Applied Research:** Reviews show that TWFE estimates can substantially differ (sometimes by more than 20%) from alternative, more robust DiD estimators. Around 42% of empirical cases assign substantial weight to forbidden comparisons, exacerbating bias [3][20].

- **Diagnostic Tools and Remedies:** The Goodman-Bacon decomposition serves as a diagnostic to detect the contribution of problematic comparisons and negative weights. When biases are large, alternative methods that avoid forbidden comparisons by restricting control groups to never-treated or not-yet-treated units are recommended [6][10].

Goodman-Bacon’s critique calls for rethinking classical TWFE DiD to account for treatment effect heterogeneity and dynamic treatment timing more carefully and transparently.

---

## Alternative Modern Estimators of DiD Post-Goodman-Bacon

### 1. Callaway and Sant'Anna’s Two-Stage Aggregation Method

**Overview:**  
Callaway and Sant'Anna (2019, 2021) propose a framework that estimates *group-time average treatment effects* (ATT(g,t)), defined as the average causal effect for units first treated in period *g*, evaluated at time *t*. By focusing on these cohort-specific parameters, they fully permit heterogeneous and dynamic treatment effects [2][6][22].

**Key Features:**

- **Handling Heterogeneity and Dynamics:** Explicitly models arbitrary heterogeneity in treatment effects across groups and calendar time. Treatment effects need not be constant or homogeneous [2].

- **Use of Clean Control Groups:** Only compares treated groups to *never-treated* or *not-yet-treated* units, avoiding the contamination bias from forbidden comparisons inherent in TWFE [2][7].

- **Flexible Parallel Trends Assumptions:** Requires *conditional parallel trends*—potential untreated outcomes evolve similarly across groups after conditioning on covariates—relaxing strict unconditional parallel trends [2][6].

- **Anticipation Effects:** Assumes limited or no anticipation prior to treatment; sensitivity analyses or bounding methods can assess robustness to anticipation violations [24].

- **Estimation Methods:** Includes outcome regression, inverse probability weighting (IPW), and doubly robust (DR) estimators combining both, enhancing robustness against misspecification [2][9].

- **Aggregation Flexibility:** Various aggregation schemes summarize ATT(g,t) into meaningful average effects—overall average effects, event-study dynamic effects, or calendar-time specific parameters [9][22].

- **Inference:** Employs multiplier cluster bootstrap procedures for simultaneous confidence intervals and hypothesis testing [7].

- **Empirical Software:** Implemented in the well-documented R package **‘did’**, widely used in applied work [9].

- **Application Illustration:** Demonstrated substantially different and more interpretable estimates than TWFE in classic applications like minimum wage effects on employment [2].

---

### 2. Sun and Abraham’s Interaction-Weighted Event-Study Estimators

**Overview:**  
Sun and Abraham (2020, 2021) develop an interaction-weighted (IW) event-study estimator aimed at correcting biases in traditional TWFE event-study regressions under staggered adoption with heterogeneous treatment effects [11][13].

**Key Features:**

- **Focus on Dynamic Effects:** Provides consistent estimates of average treatment effects at each relative event-time (time since treatment), capturing dynamic policy impacts [11].

- **Avoids Negative and Contaminated Weights:** Compares treated groups only to *never-treated* or *last-treated* groups, generating non-negative, interpretable weights, unlike TWFE event-study coefficients that can be biased by contamination [11][15].

- **Parallel Trends:** Assumes (typically unconditional) no-anticipation parallel trends. The estimator clarifies that TWFE pre-treatment leads bias misrepresents pre-trend validity due to treatment effect contamination [11].

- **Anticipation Effects:** No anticipation is assumed; any detected pre-trends in TWFE are methodological artifacts rather than genuine anticipation [11].

- **Estimation and Inference:** Regression-based estimator with interaction terms between cohort and event-time indicators; facilitates pointwise confidence intervals for event times [15].

- **Empirical Software:** Available via Stata package **‘eventstudyinteract’**, enabling event-study estimations with robust inference [12].

- **Complementarity:** Often used alongside Callaway and Sant’Anna’s approach to analyze dynamic policy effects in detail [11].

---

### 3. Borusyak, Jaravel, and Spiess’s Imputation-Based Estimators

**Overview:**  
Borusyak et al. (2021, 2024) conceptualize the estimation problem as imputing *untreated potential outcomes* for treated units using pre-treatment data of untreated and not-yet-treated units. They then estimate treatment effects as residuals between observed outcomes and these counterfactuals, providing a flexible and computationally efficient alternative to weighting-based estimators [17][19].

**Key Features:**

- **Handling Heterogeneity and Dynamics:** Treatment effects vary across units and over time inherently, as treatment effects are the difference between observed and imputed untreated outcomes, estimated at the unit-event time level [19][21].

- **Parallel Trends Through Outcome Modeling:** Instead of imposing a separate parallel trends assumption, the methodology models untreated potential outcomes conditionally, requiring correct model specification but allowing flexible functional forms including covariates and fixed effects [20][21].

- **Anticipation Effects:** Not explicitly modeled but can be partially addressed depending on the untreated outcome model; standard assumption is no anticipation [19].

- **Estimation Approach:** Single-step OLS or multi-step regression imputation is used for computing untreated outcomes with subsequent inference based on clustering and bootstrap methods [19].

- **Inference and Diagnostics:** Flexible pre-trend hypothesis tests and event-study specifications are supported, with transparent uncertainty quantification [19][21].

- **Software:** R package **‘didimputation’** facilitates implementation; increasingly adopted for complex data structures including repeated cross-section data [19].

- **Empirical Relevance:** Shows improved robustness and efficiency in settings with complex treatment timing and multiple fixed effects compared to standard DiD [19][21].

---

## Comparative Analysis of Key Identification Assumptions and Methodological Features

| Aspect                     | Callaway and Sant’Anna (CS)                          | Sun and Abraham (SA)                                          | Borusyak, Jaravel, and Spiess (BJS)                       |
|----------------------------|-----------------------------------------------------|--------------------------------------------------------------|----------------------------------------------------------|
| **Parallel Trends**        | Conditional parallel trends on observed covariates | Typically unconditional parallel trends (no anticipation)    | Implicit in correctly specified untreated outcome models |
| **Treatment Effect Heterogeneity** | Fully heterogeneous, estimated group-time ATT(g,t)   | Heterogeneous cohort-average dynamic effects via IW weights  | Unit-level heterogeneity via imputation residuals        |
| **Dynamic Treatment Effects** | Modeled explicitly via ATT(g,t) and aggregation    | Explicit event-time dynamic effects via interaction weights  | Allowed flexibly by modeling untreated outcomes dynamically|
| **Anticipation Effects**   | Assumes limited/no anticipation; sensitivity analyses possible | Assumes no anticipation; pre-trends in TWFE seen as bias     | Not explicitly modeled; assumed absent or captured in model |
| **Control Group Definition** | Never-treated or not-yet-treated units               | Never-treated or last-treated units                            | Imputation uses untreated/not-yet-treated outcomes       |
| **Estimation Approach**    | Two-stage: estimate ATT(g,t) then aggregate; DR, IPW, regression | Fully interacted TWFE with cohort-event time; interaction weighting | Imputation of untreated potential outcomes via regression |
| **Inference Methods**      | Cluster bootstrap with simultaneous confidence bands| Pointwise confidence intervals via regression inference      | Bootstrap and cluster-robust inference                    |
| **Software Availability** | R package ‘did’                                     | Stata package ‘eventstudyinteract’                            | R package ‘didimputation’                                |
| **Advantages**             | Flexible covariate adjustment; robust to misspecification; transparent aggregation | Corrects event-study coefficient contamination; clear dynamic effects  | Transparent modeling of untreated outcomes; computational efficiency |
| **Limitations**            | Requires good covariate data; complexity in aggregation might challenge interpretation | Less explicit covariate conditioning; event-time focus only  | Requires correctly specified untreated outcome models   |

All three approaches address fundamental problems in staggered DiD posed by Goodman-Bacon (2021) by avoiding “forbidden comparisons,” negative weighting, and contaminated control groups. They do so via different but complementary strategies: principled weighting and aggregation (CS), refined event-study interaction weights (SA), or outcome modeling and imputation (BJS).

---

## Empirical Adoption in Labor and Health Economics (2020–2024)

Recent empirical surveys and bibliometric analyses of published articles in **AER, QJE, and JPE** from 2020 to 2024 focusing on labor and health economics reveal the following adoption patterns:

- **Callaway and Sant’Anna’s (CS) estimator** has become the *methodologically dominant* alternative to TWFE in these leading journals. Its flexible framework, clear assumptions, and robust inference appeal to applied researchers [3][7][9].

- **Sun and Abraham’s (SA) interaction-weighted estimator** is highly influential in papers emphasizing detailed event-study and dynamic treatment effects analyses, especially when researchers want careful inference on treatment timing paths [11][15].

- **Borusyak, Jaravel, and Spiess’s (BJS) imputation-based estimator** is gaining traction, notably in studies relying on complex data structures such as repeated cross-sections or multiple fixed effects, though its adoption remains somewhat less widespread than CS or SA methods [19][20].

- **TWFE remains prevalent** but is increasingly accompanied by acknowledgments of biases pointed out by Goodman-Bacon (2021) and explicit use of alternative estimators or diagnostic decompositions to justify identification strategies.

- **Methodological Justifications:**
  - Many applied studies explicitly validate their choice of estimator using **Monte Carlo simulations**, emulating heterogeneous treatment effects and dynamic timing to test estimator performance [3].
  - **Sensitivity analyses and placebo tests** are common to assess robustness to parallel trends and anticipation violations for the selected estimator [7][9].
  - Health economics papers show a higher rate of detailed methodological discussion regarding identification assumptions and estimator robustness compared to labor economics, but overall rigor is increasing [1].

- **Software Accessibility and Community Support:**  
  The availability of user-friendly, well-documented software packages such as R’s **‘did’** and Stata’s **‘eventstudyinteract’** are pivotal in facilitating adoption. The newer **‘didimputation’** package also promotes uptake of the imputation approach [9][12][19].

---

## Integration of Roth (2022) Critique on Pre-Trend Testing and Improved Diagnostics

Jonathan Roth (2022) argues that traditional DiD pre-trend tests—examining statistical insignificance of leads before treatment—are insufficient to validate the parallel trends assumption, especially under heterogeneous and dynamic treatment effects. His key points and the response by modern estimators include:

- **Limitations of Traditional Pre-Trend Tests:**
  - Pre-trend tests can be underpowered or fail to detect violations masked by dynamic heterogeneity.
  - Null pre-treatment effects do not guarantee validity of parallel trends.
  - Visual inspection of event-study graphs using TWFE coefficients can be misleading due to contamination bias [7].

- **Responses within Modern Estimators:**
  - **Callaway and Sant’Anna’s framework** provides joint hypothesis testing across all pre-treatment periods using multiplier bootstrap, enabling simultaneous confidence bands that better detect deviations from parallel trends [7].
  - Their approach incorporates **sensitivity analyses and bounding methods**, such as those developed by Rambachan and Roth (2022), to formally assess robustness to pre-trend deviations and anticipation [24].
  
  - **Sun and Abraham’s interaction-weighted estimators** yield uncontaminated event-study estimates that produce more reliable pre-treatment confidence intervals, clarifying whether apparent pre-trends are genuine or artifacts [11].

  - **Borusyak et al.’s imputation framework** allows for pre-trend diagnostics using imputed untreated paths, providing alternative formal tests beyond visual checks [19].

- **Methodological Innovations Supporting Roth’s Concerns:**
  - Encouragement to use **simultaneous confidence intervals** covering all pre-treatment periods rather than pointwise tests.
  - Integration of **robustness checks** that consider bounded departures from parallel trends.
  - Incorporation of alternative estimation strategies such as **stacked DiD**, **synthetic DiD**, or **machine-learning-based trend controls** to improve identification validity [11][24].

- **Conclusion:** Modern DiD estimators address Roth’s critique by moving away from simplistic hypothesis testing toward richer, joint inference and sensitivity frameworks, improving researchers' ability to credibly validate identification assumptions.

---

## Conclusion

The Goodman-Bacon (2021) critique markedly shifted the landscape of Difference-in-Differences methodology, exposing deep flaws in the TWFE DiD estimator in staggered adoption designs with treatment effect heterogeneity and dynamic timing.

The three leading modern alternatives:

- **Callaway and Sant’Anna’s two-stage aggregation method,**
- **Sun and Abraham’s interaction-weighted event-study estimators,** and
- **Borusyak et al.’s imputation-based estimators**

offer complementary strategies that robustly handle heterogeneity, dynamic treatment timing, and yield clearer causal interpretations by avoiding “forbidden comparisons” and negative weighting.

These estimators relax and refine parallel trends assumptions (notably moving toward conditional and model-based versions), expressly allow treatment effect heterogeneity, and integrate explicit or implicit assumptions on anticipation effects.

Empirical evidence from leading journals in labor and health economics shows **Callaway and Sant'Anna’s approach** as the emerging dominant methodology, with **Sun and Abraham's approach** favored in dynamic policy effect studies and **Borusyak et al.’s imputation approach** gaining relevance with complex data.

All these methods have advanced practice beyond flawed pre-trend testing paradigms examined by Roth (2022), incorporating enhanced diagnostic, inference, and sensitivity analysis tools that better assess the credibility of core identification assumptions.

Applied researchers are encouraged to use these tools alongside available software to ensure trustworthy causal inference in staggered DiD designs.

---

### Sources

[1] Designing Difference-in-Difference Studies with Staggered Treatment Adoption: https://www.annualreviews.org/content/journals/10.1146/annurev-publhealth-061022-050825  
[2] Callaway and Sant'Anna (2021) Difference-in-Differences with Multiple Time Periods: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Callaway_2021_Difference-in-Differences_with_multiple_time_periods.pdf  
[3] Difference-in-Differences with Staggered Adoption: Bias Magnitude in 200 Published Studies - https://clawrxiv.io/abs/2604.00789  
[6] Goodman-Bacon (2021) Difference-in-Differences with Variation in Treatment Timing: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Goodman-Bacon_2021_Difference-in-differences_with_variation_in_treatment_timing.pdf  
[7] Roth (2022) Limitations of Pre-Trend Testing in DiD: https://jonathandroth.github.io/papers/  
[9] Pedro H.C. Sant'Anna (2024) Modern Difference-in-Differences Overview: https://psantanna.com/DiD/NABE_202410.pdf  
[11] Sun and Abraham (2020) Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Paper2023/D3--Sun-2020-Estimating-dynamic-treatment-effects-in-event-studies-with-heterogeneous-treatment-effects.pdf  
[12] Sun and Abraham (2021) GitHub Stata Package ‘eventstudyinteract’: https://github.com/lsun20/eventstudyinteract  
[15] Sun and Abraham (2020) Event-study Econometrics: https://github.com/lsun20/eventstudyinteract/blob/main/EventStudyInteract.pdf  
[17] Borusyak, Jaravel, and Spiess (2021) Imputation-Based DiD Estimation: https://causalinf.substack.com/p/deja-vu-and-differential-timing  
[19] Borusyak, Jaravel, and Spiess (2024) Revisiting Event-Study Designs: https://ideas.repec.org/p/arx/papers/2108.12419.html  
[20] didimputation R package documentation: https://cran.r-project.org/web/packages/didimputation/didimputation.pdf  
[21] didimputation GitHub Repository: https://github.com/kylebutts/didimputation  
[22] Callaway and Sant'Anna (2019) Tutorial Slides: https://psantanna.com/files/Callaway_SantAnna_2020.pdf  
[24] Rambachan and Roth (2022) Bounded Deviation Sensitivity Analyses: https://arxiv.org/abs/1909.09602  
[25] Chen, Sant’Anna, and Xie (2025) Efficient DiD and Event Study Estimators: https://causal-machine-learning.github.io/kdd2025-workshop/papers/invited3_Pedro_slides.pdf