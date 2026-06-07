# Methodological Advances in Difference-in-Differences (DiD) Estimation under Staggered Adoption: Technical Comparison, Empirical Adoption, and Responses to Pre-Trend Critiques

## Introduction

The field of applied economics has seen transformative changes in Difference-in-Differences (DiD) estimation, particularly due to critiques of traditional two-way fixed effects (TWFE) methods following the identification of biases arising from staggered treatment adoption and treatment effect heterogeneity. These developments are highly relevant for labor and health economics, where staggered policy rollouts and dynamic effects are the norm. Three estimators—Callaway & Sant'Anna’s (2021) two-stage aggregation, Sun & Abraham’s (2021) interaction-weighted/event-study, and Borusyak, Jaravel, & Spiess’s (2021) imputation-based method—are now at the forefront of methodological practice. This report provides a comprehensive technical comparison, evaluates empirical adoption trends in leading economics journals (2020–2024), and assesses how these methods respond to recent critiques, particularly Roth’s (2022) concerns around reliance on pre-trend testing.

## 1. Technical Comparison of Modern DiD Estimators for Staggered Adoption

### 1.1 The Problem: Why Classic TWFE Is Flawed for Staggered Adoption

Standard TWFE DiD estimators can yield biased and uninterpretable estimates when units receive treatment at different times and treatment effects are heterogeneous. The main issues are:

- **Negative and non-intuitive weighting:** Treated units are inappropriately used as controls, resulting in negative weights.
- **Contaminated dynamic/event-study coefficients:** Estimates at a given time may represent a mixture of different treatment effects, not the intended causal parameter.

Recent advances seek estimators that allow arbitrary treatment timing, treatment effect heterogeneity, and provide clear identification and aggregation of treatment effects while avoiding the pitfalls of TWFE[1][2][3].

---

### 1.2 Callaway & Sant'Anna (2021): Two-Stage Aggregation (Group-Time ATT) Estimator

**Key Features:**

- **Group-Time ATT (ATT(g,t)):** Estimates the average treatment effect for each group (first treated at time g) at each time t.
- **No "forbidden comparisons":** Uses only never-treated or not-yet-treated units as controls for each cohort, avoiding bias due to already-treated groups contaminating control status.
- **Flexible aggregation:** Enables dynamic treatment effect/event-study plots, calendar-time effects, and group-specific analyses.
- **Robust to heterogeneity:** Valid regardless of cohort-specific and event-time-specific variation in treatment effects.

**Performance and Identification Assumptions:**

- **Parallel Trends:** Assumes no difference in trends in untreated potential outcomes, either unconditionally or conditional on covariates (“conditional parallel trends”). Allows for observed characteristics to be controlled via regression or weighting[2][4].
- **No Anticipation:** Effects before treatment adoption should be absent or explicitly modeled. Some flexibility exists in defining anticipation windows.
- **No treatment effect homogeneity required.**
- **Estimation Efficiency:** May use only immediate pre-treatment periods as controls, which is less efficient but minimizes assumptions on outcome models. Variants such as Lee & Wooldridge’s (2023) “rolling approach” enhance efficiency by leveraging longer pre-treatment periods[5].

**Implementation:**  
Widely used in R (`did` package)[4] and Stata, offering regression, inverse probability weighting, and doubly robust options.

---

### 1.3 Sun & Abraham (2021): Interaction-Weighted/Event-Study Estimator

**Key Features:**

- **Event-study design via cohort interactions:** Decomposes effects into event time (relative to treatment) by cohort, estimating coefficients that are weighted averages within cohort/event-time cells rather than mixtures across groups as in TWFE.
- **Identification of clean dynamic effects:** Corrects problems in TWFE event studies where pre- and post-treatment estimates are contaminated by heterogeneity or negative weighting[6].
- **Robustness to heterogeneity:** No assumption of constant treatment effects over time or across groups.

**Performance and Identification Assumptions:**

- **Parallel Trends:** Assumes parallel trends for untreated outcomes; typically unconditionally, but can be extended to conditional forms if covariates are used[6].
- **No Anticipation:** Researchers must specify leads in event time to check for and potentially model anticipation effects.
- **No treatment effect homogeneity required.**
- **Aggregation:** Uses convex weights, allowing flexibility in summarizing effects.

**Implementation:**  
Available in Stata (`eventstudyweights`), R (via `fixest` or custom code), and Python (`paneleventstudy`)[7][8].

---

### 1.4 Borusyak, Jaravel, & Spiess (2021): Imputation-Based Estimator

**Key Features:**

- **Counterfactual outcome modeling:** Predicts each treated unit's untreated potential outcome, generally with regression models estimated on never- or not-yet-treated data, and calculates treatment effects as the difference between the actual and imputed counterfactual outcome for every unit–time observation[9][10][11].
- **Handles arbitrary heterogeneity and dynamic effects:** Allows parallel trends to be conditional on covariates, and estimation adapts to flexible modeling choices (e.g., ML, matching)[9].
- ** suitable for complex panel structures and high-dimensional covariates.**

**Performance and Identification Assumptions:**

- **Conditional Parallel Trends:** Requires that, after adjustment, trends among not-yet/never-treated units are a valid counterfactual for treated units[9].
- **No treatment effect homogeneity required.**
- **Anticipation Effects:** Researchers are responsible for excluding or modeling periods where anticipation is likely.

**Implementation:**  
Implemented in the R `didimputation` package[10][11] and (with code) in Python and Stata.

---

### 1.5 Comparative Table: Key Methodological Dimensions

| Method                | Treatment Effect Heterogeneity | Staggered Timing | Parallel Trends Type             | Need for Homogeneity | Anticipation Effects     | Software               |
|-----------------------|-------------------------------|------------------|-----------------------------------|----------------------|-------------------------|------------------------|
| Callaway & Sant'Anna  | Yes                           | Yes              | Conditional/Unconditional         | No                   | Flexible (user-specified)| R (`did`), Stata       |
| Sun & Abraham         | Yes                           | Yes              | Typically unconditional           | No                   | Explicitly modeled via leads | Stata (`eventstudyweights`), R, Python (`paneleventstudy`) |
| Borusyak, Jaravel, & Spiess | Yes                    | Yes              | Conditional (covariate adjustment)| No                   | By exclusion/modeling   | R (`didimputation`), Stata, Python   |

---

## 2. Empirical Adoption and Reporting Practices in Labor and Health Economics (AER, QJE, JPE, 2020–2024)

### 2.1 Trends in Methodological Adoption

- **Callaway & Sant'Anna and Sun & Abraham estimators have become dominant for DiD analyses in recent labor and health economics research**, especially when dynamic effects and heterogeneous treatment timing are present[12][13][14].
- **Borusyak et al.'s imputation approach:** Growing adoption in complex settings (covariate-rich, high-dimensional panels) and for robustness checks/comparisons[12].
- **Comparison to TWFE:** Modern empirical studies increasingly use both a robust estimator (e.g., Callaway & Sant'Anna or Sun & Abraham) and conventional TWFE in parallel, explicitly comparing results and identifying divergences[12][15].

### 2.2 Justifications and Sensitivity Analyses

- **Estimator choice is usually justified** by referencing the problematic properties of TWFE with staggered rollout and treatment-effect heterogeneity, often citing original methodological papers[12][16].
- **Monte Carlo simulations:** Often used in newly published studies and especially in methods appendices to demonstrate estimator performance in scenarios matching empirical features (e.g., dynamic and heterogeneous treatment effects)[13][14].
- **Sensitivity analysis and robustness checks:** Widely adopted, frequently through formal tools (see Section 3) and often including checks for pre-trend violations, placebo tests, and event-study visualizations displaying both robust and naive estimates[13][14].

### 2.3 Reporting Practices

- **Event-study plots:** Consistently used, with robust (heterogeneity-aware) estimators providing clean dynamic effects that avoid the contamination issues identified in TWFE[13][14].
- **Discussion of identification assumptions:** Modern studies are increasingly explicit about the parallel trends and no anticipation assumptions, often providing tests or plots for pre-treatment dynamics[12][15].
- **Multiple specification reporting:** Leading journals expect or require the reporting of results using more than one estimator, enabling transparency around estimator choice and underlying assumptions[12][13].
- **Software usage:** Open-source tools, especially the R packages `did`, `didimputation`, and relevant Stata packages, are commonly used and referenced directly in methods sections and appendices[4][10][11][12][17].

### 2.4 Example Studies

- **AER, QJE, JPE studies (2020–2024):** Recent publications in these journals within labor and health economics, including empirical studies of minimum wage effects, Medicaid expansion, and school meal policies, have used Callaway & Sant’Anna’s estimator to provide dynamic and cohort-level estimates, and several have checked robustness using Sun & Abraham or Borusyak et al. approaches[13][14][18].

---

## 3. Addressing Roth (2022) and Pre-Trend Testing Limitations

### 3.1 Roth (2022) Critique

- **Key issues:** Roth (2022) demonstrated that standard pre-trend tests (e.g., null hypothesis that pre-treatment leads equal zero) have low power and that conditioning on insignificant pre-trend tests can actually exacerbate bias under violations, reducing confidence interval coverage and potentially misleading inference[19].
- **Empirical implication:** Passing a pre-trend test does NOT guarantee unbiased DiD estimates; instead, bias and coverage can worsen due to selection on insignificant pre-trends.

### 3.2 Responses and Tools in Modern DiD Practice

#### 3.2.1 Methodological Adjustments

- **Transparent group-time/event-time estimation:** All three modern DiD estimators—Callaway & Sant’Anna, Sun & Abraham, Borusyak et al.—structure their estimation so that pre-trend dynamics can be examined in an interpretable and uncontaminated way (i.e., pre-treatment effects are estimated using only valid controls), mitigating some of the TWFE-induced "false pre-trends"[2][6][9].
- **Explicit separation of identification and inference:** Allows researchers to clarify and test the parallel trends assumption at the group-time (not just average) level.

#### 3.2.2 Diagnostic and Sensitivity Tools: HonestDiD and Beyond

- **HonestDiD (Rambachan & Roth):**  
  - A primary response is the use of formal sensitivity analysis tools that quantify how estimated effects vary under plausible violations of the parallel trends assumption[20][21].
  - HonestDiD is now routinely used in tandem with robust DiD estimators, providing intervals for treatment effects under restrictions on pre-trend violation magnitudes or smoothness[20][21].
  - Available as an R package with direct compatibility with event-study outputs from Callaway & Sant’Anna, Sun & Abraham, and Borusyak et al.[21][22].

- **Other tools:** The `pretrends` R package helps researchers assess the power of pre-trend tests and visualize potential bias, promoting honest reporting about uncertainty in pre-treatment dynamics[23].

- **Best-practice recommendations include:**
  - Use robust estimators that estimate true group-time effects,
  - Present event-study plots of pre-treatment estimates (from robust estimators),
  - Use HonestDiD or similar sensitivity analysis to report how much (and under what deviations from parallel trends) the conclusions would change,
  - Report results across multiple estimators and discuss the potential for residual pre-trend bias[12][16][21].

### 3.3 Remaining Ambiguities and Open Questions

- **Anticipation effects:** While modern estimators can accommodate specified anticipation windows, operationalizing them remains context-dependent; the literature acknowledges some ambiguity here and calls for substantive justification in each application[2][9][10].
- **Interpreting small pre-trends:** Despite better estimation, guidance on how to interpret minor deviations in pre-trend coefficients is still developing—publications increasingly supplement statistical significance with economic/contextual reasoning[19][21].
- **Threshold-setting in sensitivity analyses:** No consensus yet exists on the exact bounds to use in HonestDiD's smoothness/magnitude parameters; guidelines suggest researchers transparently report a range of scenarios[21].

---

## Conclusion

Modern methodological developments have fundamentally improved DiD analysis under staggered adoption, with Callaway & Sant'Anna, Sun & Abraham, and Borusyak et al.'s methods now standard in leading applied economics research. These estimators directly address the problems of heterogeneity and dynamic treatment seen in TWFE, provide interpretable group-time/event-time effects, and enable researchers to robustly address the credibility of identification assumptions. Empirical practice in top labor and health economics journals reflects these advances, with routine use of robust estimation, transparent justification of approach, and increased use of sensitivity analysis tools like HonestDiD to address concerns raised by Roth (2022). While certain domains—such as the operationalization of anticipation effects and interpretation of marginal pre-trends—remain evolving, the field has converged on a much higher standard for robust and transparent causal inference in panel settings.

---

## Sources

[1] Modern Difference-in-Differences: https://psantanna.com/DiD/NABE_202410.pdf  
[2] Difference-in-Differences with Multiple Time Periods (Callaway & Sant'Anna 2020): https://psantanna.com/files/Callaway_SantAnna_2020.pdf  
[3] Notes on Callaway & Sant’Anna (2021) – Staggered Adoption DiD | Chen Xing: https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did/  
[4] Treatment Effects with Multiple Periods and Groups • did: https://bcallaway11.github.io/did/  
[5] Two-stage Differences in Differences - John Gardner: https://jrgcmu.github.io/2sdd_current.pdf  
[6] Estimating dynamic treatment effects in event studies with heterogeneous treatment effects (Sun & Abraham, 2021): https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html  
[7] eventstudyweights: https://github.com/suahjl/paneleventstudy  
[8] Staggered Event Study estimation in Python: https://github.com/suahjl/paneleventstudy  
[9] Revisiting Event Study Designs: Robust and Efficient Estimation (Borusyak, Jaravel, & Spiess, 2021): https://ideas.repec.org/p/arx/papers/2108.12419.html  
[10] didimputation: Imputation Estimator from Borusyak, Jaravel, and Spiess: https://cdfinnovlab.github.io/didImputation/  
[11] R: Borusyak, Jaravel, and Spiess (2021) Estimator: https://search.r-project.org/CRAN/refmans/didimputation/html/did_imputation.html  
[12] What’s trending in difference-in-differences? A synthesis of the recent econometrics literature: https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929/  
[13] Difference-in-Differences Methods: https://pdhp.isr.umich.edu/wp-content/uploads/2023/01/DiD_PDHP.pdf  
[14] Advances in Difference-in-Differences for Policy Evaluation (PMCID: PMC11305929): https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929/  
[15] Modern Difference-in-Differences (Sant’Anna): https://psantanna.com/files/DiD_lecture_UW_Pedro_SantAnna.pdf  
[16] Difference-in-Differences for Policy Evaluation - Brantly Callaway: https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf  
[17] Callaway & Sant'Anna (2021) DiD - Statalist | The Stata Forum: https://www.statalist.org/forums/forum/general-stata-discussion/general/1770260-callaway-sant-anna-2021-did  
[18] Universal Free School Meals Policy and Childhood Obesity: https://publications.aap.org/pediatrics/article/153/4/e2023063749/196881/Universal-Free-School-Meals-Policy-and-Childhood  
[19] Event-Study Estimates after Testing for Parallel Trends (Roth, 2022): https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf  
[20] HonestDiD R Package: https://cran.r-project.org/web/packages/HonestDiD/HonestDiD.pdf  
[21] Handling Parallel Trends Violations with HonestDiD in R: https://www.tilburgsciencehub.com/topics/analyze/causal-inference/did/honest-did/  
[22] GitHub - asheshrambachan/HonestDiD: https://github.com/asheshrambachan/HonestDiD  
[23] Testing for Pre-trends in DiD and Event Studies - Tilburg Science Hub: https://www.tilburgsciencehub.com/topics/analyze/causal-inference/did/pretrends/