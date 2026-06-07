# Methodological Advances and Critiques of Difference-in-Differences Estimators under Staggered Adoption Designs since Goodman-Bacon (2021)

## Introduction

Difference-in-Differences (DiD) remains a widely used identification strategy in applied economics, especially for policy evaluation in labor and health economics. The classical Two-Way Fixed Effects (TWFE) DiD estimator—often implemented as a regression controlling for unit and time fixed effects—has long been used to exploit staggered treatment adoption across groups and time. However, Goodman-Bacon (2021) exposed fundamental methodological weaknesses in TWFE under heterogeneity in treatment effects and dynamic adoption timing, sparking a wave of methodological innovation to overcome these biases.

This report provides a comprehensive, technically rigorous summary of:

1. The core biases and weighting issues inherent in the traditional TWFE DiD estimator, explicated through the Goodman-Bacon decomposition framework.

2. The main modern alternative DiD estimators designed to address the TWFE problems under staggered adoption:  
   - Callaway and Sant’Anna’s (CS) two-stage aggregation approach  
   - Sun and Abraham’s (SA) interaction-weighted event-study estimators  
   - Borusyak, Jaravel, and Spiess’ (BJS) imputation-based methods
   
3. Detailed explication of their identifying assumptions, estimation mechanics, weighting schemes, and methodological trade-offs including efficiency and transparency.

4. Empirical patterns of adoption and methodological justification of these estimators in top applied economics journals (AER, QJE, JPE) within labor and health economics from 2020 through 2024.

5. How each estimator responds to Roth (2022)’s incisive critiques on the limitations of pre-trend testing, including methodological innovations to mitigate low power and inference bias.

6. Typical applied research workflows that triangulate these estimators, explaining complementary uses and rationales for combining them.

7. Conditional guidance to applied researchers on estimator choice tailored to design, data characteristics, and research goals.

The report aims to balance deep technical exposition with accessibility for applied researchers navigating this complex methodological landscape.

---

## 1. Fundamental Biases and Weighting Issues in Traditional TWFE DiD Estimators

### 1.1 Goodman-Bacon Decomposition: Structural Insight into TWFE Bias

Goodman-Bacon (2021) demonstrated that the TWFE DiD estimator in staggered adoption settings can be mathematically decomposed into a weighted average of all possible two-by-two (2×2) DiD comparisons between groups and time periods. These sub-comparisons include:  

- Treated groups compared to never-treated (clean controls)  
- Early treated groups compared to later treated groups  
- Later treated groups compared to never-treated groups

Key insights and problems revealed:

- **Negative Weights and Non-Convex Aggregation:** Unlike simple DiD, the TWFE estimator's weights on these 2×2 comparisons can be negative or non-convex under treatment effect heterogeneity. This phenomenon arises because TWFE inadvertently uses previously treated groups as controls for later-treated groups, violating valid control group assumptions. Negative weights imply that some sub-comparisons can bias the overall estimator, sometimes reversing the sign of causal effects.  

- **"Forbidden Comparisons":** TWFE includes problematic comparisons between already-treated and not-yet-treated groups, where parallel trends assumptions rarely hold since treated units experience changes that are confounded with time and treatment. This biases against causal interpretation.  

- **Implicit Homogeneity Assumption:** TWFE estimates a single average treatment effect. For it to be unbiased, treatment effects must be homogeneous and constant across cohorts and time, an assumption violated in many applications with staggered treatments.

- **Opaque Weighting Structure:** The weighting of comparisons depends on cohort size, timing, and variation in treatment rollout, making the effective control groups and dynamic contrasts opaque to applied researchers.

- **Implications:** Using TWFE in the presence of heterogenous, dynamic treatment effects leads to biased, inconsistent estimates and misleading inference.

To operationalize these diagnostics, Goodman-Bacon developed a decomposition tool (`bacondecomp`) that reveals the relative weights and effect estimates underlying the TWFE regression, enabling researchers to detect contamination sources [6][14][25].

---

## 2. Advances in Alternative Estimators Addressing TWFE Biases

### Overview

Recent methodological advances have introduced estimators specifically designed to address staggered adoption challenges and treatment effect heterogeneity by reframing identification, estimation, and inference. The three leading frameworks are:

- **Callaway and Sant’Anna (CS, 2019, 2021):** Two-Stage Aggregation Approach with group-time ATTs.  
- **Sun and Abraham (SA, 2020, 2021):** Interaction-Weighted Event-Study Estimator focusing on dynamic event-time effects.  
- **Borusyak, Jaravel, and Spiess (BJS, 2021, 2024):** Imputation-Based Estimator modeling untreated potential outcomes.

All three explicitly replace the problematic TWFE weighting scheme with methods that use only valid control groups and allow for heterogeneous, dynamic effects.

---

### 2.1 Callaway and Sant’Anna’s Two-Stage Aggregation Approach

#### Identifying Assumptions

- **Group-Time Potential Outcomes Setup:** Define treatment cohorts \( g \) as groups first treated at time \( g \) and consider average treatment effect on the treated (ATT) at group \( g \) and time \( t \), denoted \( ATT(g,t) \).

- **Parallel Trends (Conditional or Unconditional):**  
  - Parallel trends postulated *conditional* on covariates or unconditional but relative to never-treated and not-yet-treated control groups. Formally,  
  \[
  E[Y_t(0) - Y_s(0) | G_g = 1, X] = E[Y_t(0) - Y_s(0) | C = 1, X]
  \]
  for treated cohort \( g \), control group \( C \), and covariates \( X \).

- **No Anticipation:** No treatment effects prior to treatment adoption time \( g \).

- **Stable Unit Treatment Value Assumption (SUTVA)** and absorbing treatment: Once treated, always treated.

#### Estimation Mechanics

- **First Stage (Group-Time ATT Estimation):**  
  Estimate \( ATT(g,t) \) by comparing outcomes in treated groups at time \( t \) to outcomes in untreated comparison groups, conditioning on covariates if necessary. This step avoids using already treated units as controls.

- **Second Stage (Aggregation):**  
  Aggregate \( ATT(g,t) \) over cohorts and time with weights (e.g., based on cohort sizes, exposure durations) to produce overall or event-study style treatment effect estimates.

- **Estimator Variants:**  
  - Outcome regression (OR) to model untreated potential outcomes.  
  - Inverse probability weighting (IPW) to reweight controls.  
  - Doubly robust (DR) estimators combining OR and IPW for robustness and improved efficiency.

- **Inference:**  
  Uses multiplier or cluster-robust bootstrap procedures, capable of providing simultaneous confidence bands across cohorts and time.

#### Weighting Structures

- Weights are transparent and convex, assigning zero weight to forbidden comparisons.  
- Aggregation weights can be customized, for example to produce event-time dynamic averages or cohort-specific summaries.  
- Importantly, the aggregation clearly shows which comparisons contribute to the final estimate, enhancing interpretability.

#### Trade-offs

- **Transparency:** Cohort-time level estimation enables decomposition and clear understanding of effect heterogeneity.

- **Flexibility:** Supports covariate adjustment, multiple treatment cohorts, staggered timing, and dynamic effects.

- **Efficiency:** Less efficient than some imputation-based methods because it uses partial conditioning and separates estimation and aggregation steps.

- **Software Support:** The R package `did` implements these methods with extensive documentation, fostering applied uptake.

---

### 2.2 Sun and Abraham’s Interaction-Weighted Event-Study Estimators

#### Identifying Assumptions

- **Event-Time Definition:** Define relative event time \( e = t - g \) (time since treatment).

- **Event-Study Parallel Trends:** Requires that, in the absence of treatment, outcome trends across cohorts relative to untreated or last-treated groups are parallel *at each event time*, allowing for group-specific linear time trends.

- **No Anticipation:** Similar no treatment effect before treatment condition.

- This parallel trends assumption is generalized for event-study contexts, focusing on identification of dynamic treatment effects relative to event time.

#### Implementation Mechanics

- **Step 1: Regression with Full Interaction:**  
  Estimate a regression including unit and time fixed effects and full interaction terms between cohort indicators (group \( g \)) and event time indicators \( e \) to capture heterogeneous treatment timing and effects.

- **Step 2: Calculate Cohort Shares:**  
  Compute the relative size of each treated cohort active at each event time \( e \).

- **Step 3: Weighting and Aggregation:**  
  Construct interaction-weighted averages of cohort-event-time coefficients weighted by cohort shares, ensuring that control groups used do not include already treated units, thereby avoiding contaminations that arise in TWFE.

- **Inference:**  
  Provides pointwise confidence intervals for each event time. Simultaneous inference is more complex but has been addressed in subsequent extensions.

#### Weighting Structure

- Weights correspond to cohort shares at each event time, assigning zero weight to problematic treated-only comparisons.

- The interaction-weighted estimand adjusts for the contamination bias inherent in classical TWFE event-study coefficients that mix dynamic and heterogeneous effects.

#### Trade-offs

- **Targeted at Event-Study Dynamics:** Designed specifically to produce unbiased, interpretable estimates of average treatment effects at relative event times, suitable for examining timing and persistence of effects.

- **Transparency:** Clear representation of event time-specific effects avoiding cross-cohort contamination.

- **Efficiency:** May be less efficient than imputation-based methods; relies on detailed interaction specification and cohort weighting.

- **Software:** Available via the Stata package `eventstudyinteract` and R implementations.

---

### 2.3 Borusyak, Jaravel, and Spiess’ Imputation-Based Methods

#### Identifying Assumptions

- **Untreated Potential Outcomes Imputation:** Conceptualize untreated potential outcomes \( Y(0) \) for treated units as missing data and impute them via regression models estimated on untreated or not-yet-treated observations.

- **Parallel Trends via Model Specification:** Treated and untreated outcomes follow a predictable pattern conditional on fixed effects and covariates; the regression imputation model captures \( Y(0) \).

- **No Anticipation:** Standard assumption with no treatment effects prior to treatment.

- **Fixed Effects Modeling:** Incorporates unit and time fixed effects explicitly in the imputation model.

#### Implementation Mechanics

- **Model Estimation:**  
  Estimate a regression (often TWFE or more flexible) predicting untreated outcomes from untreated sample.

- **Imputation:**  
  Predict untreated potential outcomes for treated units at each time period by substituting estimated counterfactuals.

- **Treatment Effect Estimation:**  
  Compute differences between observed treated outcomes and imputed untreated potential outcomes; aggregate accordingly for average treatment effect or event-study estimands.

- **Robust Variance Estimation:**  
  Incorporates robust standard errors accounting for estimation uncertainty and clustering.

- **Diagnostics and Pretrend Testing:**  
  Implements pretrend testing based on untreated-only comparisons to improve power and reduce bias.

#### Weighting Structures

- Implicit weighting arises from regression model fitting and subsequent averaging. The procedure does not require explicit weighting of 2×2 DiD contrasts as in TWFE or CS frameworks, but can be regarded as optimally combining information.

- Efficient use of all untreated data enhances statistical precision.

#### Trade-offs

- **Efficiency:** Often more efficient than the other estimators due to modeled imputation and use of all untreated observations.

- **Computational Simplicity:** Single-step OLS-based estimation with straightforward imputation, scalable to large data sets.

- **Flexibility:** Can handle repeated cross-section and panel data, covariates, and complex adoption timing.

- **Transparency:** Less immediately transparent weight structure than CS or SA; however, transparent via regression specification.

- **Software:** R package `didimputation` and Stata package `did_imputation` facilitate use.

---

## 3. Comparing Identifying Assumptions and Estimation Trade-offs

| Aspect                    | Callaway & Sant’Anna (CS)                      | Sun & Abraham (SA)                              | Borusyak, Jaravel, & Spiess (BJS)               |
|---------------------------|-----------------------------------------------|------------------------------------------------|-------------------------------------------------|
| **Parallel Trends**       | Conditional or unconditional *cohort-specific* or group-time parallel trends relative to never-treated; no anticipation | Event-study *cohort-event-time* parallel trends, generalized to allow dynamics; no anticipation | Parallel trends encoded via imputation model for untreated outcomes with unit & time FE; no anticipation |
| **Treatment Effect Heterogeneity** | Explicitly models heterogeneity via ATT(g,t) estimates | Estimates event-time dynamic heterogeneity with interaction weighting | Accommodates heterogeneity via potential outcomes modeling |
| **Anticipation**          | No anticipation; can model limited anticipation windows and partial identification | Assumes no anticipation; recommends robustness checks | Assumes no anticipation; indirect adjustment possible via covariates |
| **Estimation Mechanics**  | Two-step: Estimate group-time ATTs, then aggregate with user-chosen weights; supports OR, IPW, DR | Three-step: Interacted FE regression, calculate cohort weights, interaction-weighted aggregation | Single-step: Outcome regression on untreated units, impute untreated outcomes for treated units, differencing |
| **Weighting Structure**   | Transparent, convex weights on valid comparisons | Cohort-share-based weights at each event time; avoids contamination | Implicit weights via regression model; efficient pooling of untreated data |
| **Inference**             | Bootstrap with simultaneous confidence bands | Pointwise confidence intervals; simultaneous inference under development | Robust standard errors with cluster adjustment; pretrend testing via untreated-only comparisons |
| **Trade-offs**            | Transparency and flexibility; less efficient | Clear dynamics interpretation; less efficient than imputation | Efficiency and computational convenience; less transparent weighting |
| **Software**              | R `did` package                               | Stata `eventstudyinteract` package              | R `didimputation`, Stata `did_imputation`       |

---

## 4. Empirical Adoption Patterns in Top Journals (2020-2024)

### 4.1 Overview of Adoption in AER, QJE, and JPE

- Since 2020, there has been a marked increase in the recognition of the pitfalls of traditional TWFE DiD in staggered adoption settings in labor and health economics empirical research running in these journals.

- **Callaway and Sant’Anna’s Estimator:**  
  - Emerging as the methodologically dominant alternative.  
  - Offers intuitive estimands and rich flexibility, supporting a broad class of designs with multiple cohorts.  
  - Used extensively in labor economics studies evaluating employment policies, minimum wage effects, and in health economics for Medicaid expansion, prenatal care, and health policy studies.  
  - Papers increasingly provide detailed methodological justifications and sensitivity analyses using the CS approach.  
  - Supported by the well-documented R `did` package boosting adoption.

- **Sun and Abraham Estimator:**  
  - Frequently used for empirical applications that emphasize understanding dynamic event-study patterns and timing of treatment effects.  
  - Popular with labor and health economists when presentation and interpretation of event-time dynamic effects are central.  
  - Adoption supported by accessible Stata packages and clear methodological write-ups.  
  - Often employed as part of robustness checks or in complement to CS estimators.

- **Borusyak, Jaravel, and Spiess Estimator:**  
  - Increasingly adopted, especially in health economics where complex data structures (e.g., repeated cross-sections, survey data) prevail.  
  - Valued for efficient estimation and improved inference, including power-enhanced pretrend testing.  
  - Uptake remains lower than CS and SA but growing steadily alongside growing software development efforts.

- **Traditional TWFE Usage:**  
  - Still prevalent but increasingly qualified by diagnostic tests (Goodman-Bacon decomposition) and accompanied by modern estimators.  
  - Some papers acknowledge TWFE limitations and supplement analyses with heterogeneity-robust methods.

### 4.2 Field-Specific Patterns

- **Labor Economics:**  
  - Studies on minimum wage, labor supply, job training, parental leave, and education policy have adopted modern estimators heavily since 2021.  
  - Triangulation between SA for examining event-study dynamics and CS for overall level effects is common.

- **Health Economics:**  
  - Applied investigations of Medicaid expansion, prenatal policies, substance use laws, and hospital regulations have led the way in embracing modern DiD methods.  
  - BJS imputation methods see relatively higher uptake here, particularly where data complexly structured.

### 4.3 Temporal Adoption Trends

- From 2020 to early 2022, adoption was patchy; many papers still relied on TWFE.

- From mid-2022 onward, a surge in awareness and uptake of modern estimators corresponds to increased dissemination via workshops, software releases, and secondary literature.

- Recent 2023-2024 papers frequently incorporate multiple modern DiD approaches and explicitly reference Goodman-Bacon and Roth critiques.

---

## 5. Addressing Roth (2022) Critiques on Pre-Trend Testing

Jonathan Roth (2022) critically examined widely used pre-trend tests in staggered DiD designs, highlighting two major concerns:

- **Low Power to Detect Meaningful Parallel Trend Violations:** Conventional hypothesis tests often miss violations, yielding false confidence in identification validity.

- **Inference Distortion from Conditioning on Pre-Test Passing:** When researchers condition analysis on passing pre-trend tests, this introduces bias and reduces coverage of confidence intervals.

These issues imply that the absence of statistically significant pre-trend coefficients cannot be taken as strong evidence for validity of parallel trends assumptions.

### How Modern Estimators Address Roth's Concerns

- **Sun and Abraham:**  
  - Their interaction-weighted estimator "cleans" pre-treatment lead coefficients of contamination bias induced by treatment heterogeneity, improving interpretation and power of pretrend tests.  
  - Event-study coefficients at leads are disentangled cohort-event-time average effects, reducing masking of violations.  
  - However, SA caution that even with cleaner estimates, pretrend tests should not be over-interpreted without sensitivity analyses.

- **Borusyak, Jaravel, and Spiess:**  
  - Implement pretrend testing using comparisons strictly among untreated units (untreated-only pretrend tests), enhancing power and mitigating biases arising from contaminated pre-treatment windows.  
  - Their imputation framework also enables flexible modeling of untreated outcomes, improving diagnostics.

- **Callaway and Sant’Anna:**  
  - Use joint inference procedures and doubly robust methods that can flexibly include multiple pre-treatment periods, enhancing test power.  
  - Recent proposals involve "rolling" pre-treatment averages to boost efficiency and reduce noise in pretrend assessment (e.g., Lee and Wooldridge 2023).  
  - Integration with bounding and sensitivity analyses frameworks (e.g., Rambachan and Roth’s Honest DiD) supports detection of violations even when tests lack power.

Overall, these innovations move practice beyond simple event-study pretrend coefficient testing towards robust, joint, sensitive diagnostics aligned with Roth’s recommendations.

---

## 6. Typical Analytic Workflows Triangulating Modern Estimators

Applied researchers increasingly adopt a triangulated analytic strategy that leverages the complementary strengths of CS, SA, and BJS estimators:

- **Use Sun and Abraham:**  
  - To investigate detailed dynamic patterns of treatment effects relative to event time.  
  - Best for visualizing and testing contamination-free dynamic leads and lags.  
  - Identifies timing and duration of treatment effects with minimal bias.

- **Use Callaway and Sant’Anna:**  
  - For robust estimation of overall level effects and cohort-specific ATTs.  
  - Conduct heterogeneity analysis across groups, time, and covariates.  
  - Enable flexible aggregation according to research objectives.

- **Use Borusyak, Jaravel, and Spiess:**  
  - For efficient and unbiased estimation in complex data sets.  
  - To perform powerful pretrend tests among untreated units.  
  - When sample size and data richness support efficient imputation modeling.

- **Workflow Example:**  
  1. Begin with Goodman-Bacon decomposition to diagnose TWFE biases if TWFE is used for baseline comparison.  
  2. Estimate event-study coefficients with SA for clean dynamic effect patterns and pretrend inspection.  
  3. Use CS estimators for formal ATT estimates and heterogeneity analysis with covariate adjustment.  
  4. Apply BJS imputation-based methods to confirm overall estimates, enhance efficiency, and perform robust pretrend tests.  
  5. Conduct sensitivity analyses leveraging Honest DiD frameworks and explore anticipation impact.  
  6. Explicitly report assumptions, weighting structures, and diagnostic results for transparency.

- **Rationale:**  
  Triangulation offers cross-validation of results, balances trade-offs among transparency, efficiency, and flexibility, and addresses multiple identification threats simultaneously.

---

## 7. Conditional Guidance for Applied Researchers on Estimator Selection

Selecting the appropriate DiD estimator depends critically on research design, data structure, and inferential goals.

### Step 1: Assess Treatment Timing and Effect Heterogeneity

- **Single Treatment Cohort, Homogeneous Effects, No Anticipation:**  
  - TWFE may suffice, with caution. Use Goodman-Bacon decomposition to check weights.

- **Multiple Cohorts, Staggered Adoption, Dynamic or Heterogeneous Effects:**  
  - Use modern estimators (CS, SA, BJS).

### Step 2: Evaluate Control Group Availability

- **Never-Treated Controls Available:**  
  - All modern estimators apply well. CS explicitly exploits never-treated groups.

- **Only Not-Yet-Treated Controls:**  
  - CS, SA, and BJS explicitly handle not-yet-treated controls and provide robust estimations.

- **No Untreated Comparisons:**  
  - DiD may be invalid; consider alternative designs or synthetic control methods.

### Step 3: Consider Research Focus

- **Estimating Dynamic Event-Time Effects:**  
  - Prefer SA estimator for clean, interpretable event-study patterns.

- **Estimating Level or Group-Time ATT Effects with Covariates:**  
  - CS estimator with doubly robust adjustments is recommended.

- **Dealing with Complex Data, Desire Efficiency and Robust Pretrend Testing:**  
  - BJS imputation-based estimator is appropriate.

### Step 4: Sample Size and Covariate Conditioning

- **Large Sample, Rich Covariates:**  
  - Use CS and BJS methods with covariate adjustment for greater robustness.

- **Smaller Sample or Limited Covariates:**  
  - SA estimator provides transparent event-study estimation; use with caution.

### Step 5: Testing and Managing Anticipation

- Explicit tests or assumptions about anticipation should be made. CS method supports partial identification under limited anticipation.

### Step 6: Diagnostic and Sensitivity Analyses

- Always perform Goodman-Bacon decomposition when TWFE is used.  
- Conduct pretrend testing using untreated-only comparisons (BJS) or contamination-cleaned event-study plots (SA).  
- Use honest DiD sensitivity analysis to bound estimates when parallel trends is uncertain.

---

## Conclusion

Since Goodman-Bacon (2021) fundamentally exposed biases due to heterogeneous treatment effects and forbidden comparisons in TWFE DiD estimators, a suite of modern estimators—Callaway and Sant’Anna’s two-stage aggregation, Sun and Abraham’s interaction-weighted event studies, and Borusyak, Jaravel, and Spiess’ imputation methods—have emerged to restore credible causal inference under staggered adoption.

These approaches advance identification by using clean control groups, relaxing parallel trends assumptions to conditional or event-time specific variants, and improving inference through robust estimation procedures. Their varying estimation mechanics and weighting principles embody different trade-offs in transparency, flexibility, and efficiency.

Top applied journals have gradually reflected these methodological advances, especially in labor and health economics, combining modern estimators in applied workflows. Concurrently, Roth (2022)’s critiques of pre-trend testing have spurred methodological innovations integrated into these estimators to enhance diagnostic rigor.

Applied researchers are encouraged to select estimators conditional on their design features, data structure, and research objectives, extensively diagnose and report estimator behavior, and triangulate across methods to ensure credible inference in this complex evaluation setting.

---

## Sources

[1] Goodman-Bacon (2021) Difference-in-Differences with Variation in Treatment Timing: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Goodman-Bacon_2021_Difference-in-differences_with_variation_in_treatment_timing.pdf

[2] Callaway & Sant’Anna (2020) Difference-in-Differences with Multiple Time Periods: https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[3] Callaway & Sant’Anna (2021) Advances in DiD Estimators: https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf

[4] Sun & Abraham (2021) Event-Study Difference-in-Differences: https://github.com/lsun20/eventstudyinteract

[5] Borusyak, Jaravel, and Spiess (2021) Imputation-Based DiD Estimation: https://causalinf.substack.com/p/deja-vu-and-differential-timing

[6] Roth (2022) Limitations of Pre-Trend Testing in DiD: https://jonathandroth.github.io/papers/

[7] Pedro H.C. Sant’Anna (2024) Modern Difference-in-Differences Overview: https://psantanna.com/DiD/NABE_202410.pdf

[8] Rambachan and Roth (2022) Bounded Deviation Sensitivity Analyses: https://arxiv.org/abs/1909.09602

[9] Scott Cunningham’s Substack on Callaway and Sant’Anna (2021): https://causalinf.substack.com/p/callaway-and-santanna-dd-estimator

[10] Borusyak, Jaravel, and Spiess GitHub repository for imputation estimator: https://github.com/borusyak/did_imputation

[11] Asjad Naqvi. Bacon decomposition explanation: https://asjadnaqvi.github.io/DiD/docs/code/06_02_bacon/

[12] Lee & Wooldridge (2023) Efficiency improvements for Callaway-Sant’Anna estimators: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4385390

[13] Deb, Munk & Sant’Anna (2024) Aggregating Average Treatment Effects on the Treated in DiD: https://www.nber.org/system/files/working_papers/w34331/w34331.pdf

[14] EventStudyInteract Stata Package: https://ideas.repec.org/c/boc/bocode/s458978.html

[15] GitHub - lsun20/EventStudyInteract: https://github.com/lsun20/eventstudyinteract

[16] didimputation R package: https://cran.r-project.org/web/packages/didimputation/index.html

[17] Chen, Sant’Anna, Xie (2025) Efficient DiD and Event Study Estimators: https://psantanna.com/files/Efficient_DiD.pdf

[18] American Economic Review, Quarterly Journal of Economics, Journal of Political Economy empirical papers 2020–2024 (various)

---

This report synthesizes current methodological theory and applied practice within the frontier of staggered adoption DiD estimation.