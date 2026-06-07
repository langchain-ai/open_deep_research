# Technical Report on Modern Difference-in-Differences Estimators Addressing Staggered Adoption Critiques: Callaway & Sant’Anna, Sun & Abraham, and Borusyak, Jaravel & Spiess

---

## Introduction

Recent advances in Difference-in-Differences (DiD) methodology have substantially enhanced the robustness and interpretability of causal inference under **staggered adoption designs**, where treatment timing varies across units/groups. Traditional Two-Way Fixed Effects (TWFE) DiD regressions suffer from well-documented biases under treatment effect heterogeneity and dynamic adoption [Goodman-Bacon (2021)], motivating the development of alternative estimators.

This report provides a comprehensive, technical exposition of three leading modern DiD estimators explicitly designed to address these challenges:

- **Callaway & Sant’Anna (2019, 2021)** two-stage aggregation estimator  
- **Sun & Abraham (2020, 2021)** interaction-weighted event-study estimator  
- **Borusyak, Jaravel & Spiess (2021, 2024)** imputation-based estimator  

Each section systematically covers:

- **Identification assumptions**, including modeling of parallel trends, allowance for treatment effect heterogeneity, and anticipation assumptions  
- **Estimation mechanics**, detailing fixed effects structures, 'clean control' construction, weighting schemes, and inference procedures  
- **Treatment effect heterogeneity scope and flexibility**  
- **Relations and equivalences across estimators**  
- **Best-practice recommendations for pre-trend testing**, informed by Roth (2022) and subsequent refinements  
- **Implications for applied researchers in labor and health economics**

Methodological insights primarily draw on original source papers, ensuring precise and authoritative exposition.

---

## 1. Callaway & Sant’Anna Two-Stage Aggregation Estimator

### 1.1 Identification Assumptions

The Callaway & Sant’Anna (CS) estimator targets **group-time average treatment effects on the treated**:

\[
ATT(g,t) = E[Y_t(g) - Y_t(\infty) \mid G_g=1]
\]

where:

- \(g\) denotes the **group or cohort** first treated in period \(g\)  
- \(t\) indexes calendar time  
- \(Y_t(g)\) is the potential outcome at time \(t\) if treated from \(g\) onward  
- \(Y_t(\infty)\) is untreated potential outcome  
- \(G_g=1\) indicates membership in cohort \(g\)

Key assumptions include:

- **Conditional Parallel Trends (CPT):** Potential untreated outcomes follow parallel trends over time across groups **conditional on covariates \(X\)**:

\[
E[Y_t(0) - Y_s(0) \mid G_g=1, X] = E[Y_t(0) - Y_s(0) \mid C=1, X]
\]

Here, \(C\) denotes the control group (never-treated or not-yet-treated units). This assumption is **cohort- and time-specific**, allowing flexible heterogeneity in untreated outcome trends—a relaxation over unconditional PT in classic DiD.

- **No Anticipation:** No treatment effect manifests before actual adoption: 

\[
Y_t(g) = Y_t(\infty) \quad \text{for } t < g
\]

- **Overlap:** Sufficient representation of control units at each group-time combination, along with common support in covariates.

These assumptions explicitly exclude the problematic "forbidden comparisons" inherent to TWFE methods (such as already-treated units serving as controls for later-treated units), enabling unbiased group-time ATT identification [2][3].

### 1.2 Treatment Effect Heterogeneity Scope

CS estimator **allows fully unrestricted heterogeneity** in treatment effects across cohorts \(g\) and calendar times \(t\):

- Each \(ATT(g,t)\) is separately identified, reflecting genuine treatment effect **dynamics** and **cohort-specific effects**
- No restrictions on effect homogeneity or constancy over time or groups
- Aggregation into summaries or event-study parameters occurs *after* separate estimation of fundamental \(ATT(g,t)\) building blocks, preserving heterogeneity information

This contrasts traditional TWFE which assumes homogeneous or constant treatment effects to identify a unique parameter, often leading to biased aggregates [2].

### 1.3 Estimation Mechanics: Two-Stage Procedure

The CS estimator employs the following **two-stage approach**:

#### Stage 1: Estimation of Group-Time ATT \(ATT(g,t)\)

- **Select 'clean controls'**: For each treated group \(g\) at time \(t\), the control group comprises units that are **never-treated or not-yet-treated** as of \(t\), ensuring no contamination from already-treated units.  
- **Estimation strategies**:

  - **Outcome Regression (OR):** Model untreated potential outcomes conditional on covariates with fixed effects, typically via regression:  
  \[
  Y_{it} = \alpha_i + \lambda_t + f(X_{it}) + \epsilon_{it}
  \]
  using untreated and not-yet-treated units to fit the model, then predict \(E[Y_t(0) \mid G_g=1, X]\).

  - **Inverse Probability Weighting (IPW):** Estimate propensity scores for treatment timing, reweight control units to match treated cohort covariate distribution.

  - **Doubly Robust (DR):** Combine OR and IPW to improve robustness and efficiency.

- **Fixed effects structure:**  
  Fixed effects for units and time (and possibly interactions or higher-order terms) can be incorporated **within regression adjustments or propensity score models**, absorbing unit- and time-specific heterogeneity in untreated outcomes.

This stage produces consistent and **unbiased estimates of each \(ATT(g,t)\) parameter**, circumventing the negative-weighting and contamination biases identified by Goodman-Bacon (2021).

#### Stage 2: Aggregation and Inference

- **Flexible aggregation:** Apply user-specified convex weights \(w(g,t)\) to form overall treatment effect estimands, e.g.,

\[
\theta = \sum_{g,t} w(g,t) \hat{ATT}(g,t)
\]

Weights are interpretable and respect the treatment adoption structure (e.g., cohort size, exposure length) and economic relevance.

- **Robust inference:** Typically uses **multiplier bootstrap procedures** that:

  - Accommodate clustering (e.g., at unit level)
  - Provide **simultaneous confidence bands** over the \(ATT(g,t)\) surface or event-study patterns
  - Allow joint inference on multiple group-time effects

Separating estimation from aggregation **ensures transparency** of weighting and clean decomposition of heterogeneous effects.

### 1.4 Fixed Effects and Clean Controls

- The CS estimator **does not fit a single TWFE regression** with group and time fixed effects and a treatment dummy. Instead:

  - Fixed effects enter **first stage covariate adjustment models** flexibly.
  - The approach **explicitly excludes** already treated units at time \(t\) from control groups, thus defining "clean" untreated units per group-time contrast, enhancing validity.
  - This construction respects the causal estimand \(ATT(g,t)\) avoiding flawed forbidden comparisons endemic to TWFE designs.

### 1.5 Weighting Schemes

- Aggregation weights \(w(g,t)\) are explicitly defined and convex, never negative, unlike TWFE’s implicit and sometimes negative weights.
- Weights can reflect:

  - Proportional cohort sizes in the population
  - Exposure durations (difference \(t-g\))
  - Analytical priorities, such as weighting later periods more heavily to capture dynamic effects
- The package implementation (`did` in R) allows flexibility to customize summary parameters tailored to researcher objectives [3][9].

### 1.6 Relation to Imputation-Based Estimators

- The CS two-stage estimator, which estimates group-time ATT contrasts using clean untreated controls and aggregates, corresponds in spirit to a **partially imputation-based approach** where the untreated potential outcomes are implicitly modeled via regression or weighting.

- While imputation-based estimators directly **predict untreated potential outcomes for treated units**, CS’s approach estimates contrasts group-wise, offering greater transparency by estimating each ATT separately before aggregation.

- Both avoid the negative weight biases present in TWFE by leveraging untreated units only as controls.

- Formal connections exist between CS’s doubly robust estimators and imputation frameworks but CS emphasizes separation of identification steps to enhance interpretability and transparency [2][3][9].

---

## 2. Sun & Abraham Interaction-Weighted Event-Study Estimator

### 2.1 Identification Assumptions

Sun & Abraham (SA) extend event-study DiD frameworks to explicitly account for:

- **Cohort-specific event time effects**: Defining relative time \(e = t - g\)  
- **Event-time specific parallel trends:** For each relative time \(e\), control and treated cohorts' potential untreated outcomes evolve in parallel conditional on fixed effects:

\[
E[Y_{i,t}(0) \mid G_g=1] - E[Y_{i,s}(0) \mid G_g=1] = E[Y_{i,t}(0) \mid C=1] - E[Y_{i,s}(0) \mid C=1]
\]

for \(t,s\) such that \(t-s = e\).

- **No Anticipation:** No treatment effect before \(e=0\).

- The assumption is stronger than in CS, requiring parallel trends at each **event time** (relative to treatment) rather than only calendar time or grouped time-specific.

- The method uses **clean controls** defined as:

  - **Never-treated units**, or  
  - **Last-treated units** (units treated at the latest time)

  These avoid contamination typical of not-yet-treated controls who themselves become treated later [3][6][8].

### 2.2 Treatment Effect Heterogeneity

- SA estimator permits **fully unrestricted heterogeneity** in treatment effects across cohorts and event times.

- By constructing **cohort-by-event-time interactions**, it estimates cohort-specific event-time treatment effects \(CATT(g,e)\).

- Aggregation is via **interaction-weighted averages** of these cohort-event-time effects, weighted by **cohort shares at event time \(e\)**, guaranteeing convex and non-negative weights.

- This avoids the TWFE problem where event-study coefficients conflate dynamic effects across cohorts with varied treatment timing, potentially producing nonsensical or non-convex estimands.

### 2.3 Estimation Mechanics

The SA estimator involves:

- **Step 1: Saturated Regression**

  Regress the outcome on:

  \[
  Y_{it} = \alpha_i + \lambda_t + \sum_{g} \sum_{e \neq -1} \beta_{g,e} \cdot 1\{G_g=1\} \times 1\{t - g = e\} + \epsilon_{it}
  \]

  where:

  - \(\alpha_i\) are unit fixed effects,  
  - \(\lambda_t\) are time fixed effects,  
  - \(\beta_{g,e}\) capture the event-time effects for cohort \(g\), excluding a baseline event time (e.g., \(e = -1\)).

  This **full saturation of cohort × event-time interactions** enables nonparametric identification of heterogeneous effects.

- **Step 2: Calculate Cohort Weights**

  Determine cohort shares \(\pi_{g,e} = N_{g,e} / \sum_g N_{g,e}\), normalized shares of units in cohort \(g\) at event time \(e\).

- **Step 3: Interaction-Weighted Aggregation**

  Compute aggregate event-study estimates:

  \[
  \hat{\beta}_e = \sum_g \pi_{g,e} \hat{\beta}_{g,e}
  \]

  ensuring estimates represent a **weighted average of cohort-specific treatment effects at event time \(e\)** using population proportions.

- **Fixed Effects Structure:** By including unit and time fixed effects, the model controls for level unit heterogeneity and common time shocks, isolating dynamic treatment effects.

- This approach **avoids contaminating pre-treatment leads** with post-treatment effects from other cohorts, a problem in TWFE event-study designs.

- **Inference:** Standard cluster-robust errors support **pointwise confidence intervals**. While simultaneous inference was initially less developed, extensions exist [3][9].

### 2.4 Weighting Scheme and Clean Controls

- SA estimator uses **cohort shares as weights** at each event time, which are non-negative and sum to unity.

- The use of last-treated or never-treated units as controls at each event time produces **clean control groups**, avoiding the TWFE issue where not-yet-treated units may already be affected or have heterogeneous anticipatory effects [3][6].

### 2.5 Relation to Goodman-Bacon and Other Estimators 

- Unlike TWFE which mixes cohorts and event times and combines them with implicit and sometimes negative weights, SA produces **explicit, interpretable, convex weights** avoiding contamination.

- The estimator yields **cohort-event-time specific causal effects**, clear dynamic effects trajectories free from crossover biases.

- It complements, rather than replaces, CS estimators: SA excels at **dynamic event-time visualization and testing**, while CS is optimal for estimating group-time ATT and aggregated parameters with covariate adjustment.

---

## 3. Borusyak, Jaravel & Spiess Imputation-Based Estimator

### 3.1 Identification Assumptions

Borusyak, Jaravel & Spiess (BJS) model DiD as a **missing data imputation problem**, where untreated potential outcomes for treated units are unobserved and imputed using a regression model fit on untreated units:

\[
Y_{it}(0) \approx \hat{Y}_{it}(0) = \hat{\alpha}_i + \hat{\lambda}_t + \text{(optional covariates)} + \hat{\eta}_{it}
\]

Key assumptions:

- **Parallel Trends for Untreated Potential Outcomes:**  
Untreated outcomes evolve smoothly and obey the model estimated from untreated units (never-treated or not-yet-treated), implying conditional parallel trends with fixed effects and covariates.

- **No Anticipation:** No systemic difference in untreated potential outcomes before treatment adoption—assumed for identification, but also testable via pre-treatment lead terms in the model.

- **Stable Unit Treatment Value Assumption (SUTVA):** No interference or dynamic spillovers.

- Crucially, this approach allows for **unrestricted heterogeneity** in treatment effects over units and time by recovering individual treatment effects as residuals from imputed untreated outcomes.

### 3.2 Treatment Effect Heterogeneity and Flexibility

- The imputation method produces **unit-time level treatment effect estimates**:

\[
\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0), \quad \text{for treated }i,t
\]

- These effects can then be aggregated flexibly to cohort- or event-time contrasts analogous to \(ATT(g,t)\) or event-study parameters without homogeneity restrictions.

- By relying on a rich regression model for untreated outcomes, the approach naturally captures heterogeneity in both observed and unobserved confounders within the fixed effects and modeling structure.

### 3.3 Estimation Mechanics

- **Step 1: Estimate Model of Untreated Outcomes**

  - Fit a linear model on untreated or not-yet-treated observations, including:

    - Unit fixed effects: \(\alpha_i\),  
    - Time fixed effects: \(\lambda_t\),  
    - Optional covariates or flexible terms.

  - This is typically a TWFE regression restricted to untreated observations.

- **Step 2: Impute Untreated Potential Outcomes**

  - Predict counterfactual untreated outcomes \(\hat{Y}_{it}(0)\) for treated units based on the model.

- **Step 3: Compute Treatment Effects**

  - For each treated unit-period, calculate:

    \[
    \hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)
    \]

  - Aggregate \(\hat{\tau}_{it}\) across units and times to obtain average treatment effects or event-study parameters.

- **Pretrend Testing**

  - Tests for parallel trends are implemented by examining whether predicted untreated outcomes differ systematically pre-treatment.

  - Tests based on untreated-only comparisons improve power over TWFE lead-based pretrend tests by avoiding contamination.

- **Inference**

  - Uses heteroskedasticity-robust and cluster-robust standard errors accounting for imputation uncertainty.

- **Extensions**

  - The 2024 paper by Borusyak et al. extends the framework to explicitly model **anticipation** and **treatment misclassification**, providing bias-corrected estimators and valid inference in presence of these violations [26].

### 3.4 Fixed Effects and Control Construction

- The unit and time fixed effects remove all time-invariant unit heterogeneity and common shocks, isolating systematic untreated outcome trends.

- Clean controls, in practice, are the untreated or not-yet-treated units used to estimate the regression.

- Unlike TWFE approaches, no treated units serve as controls, eliminating contamination.

### 3.5 Weighting Structure

- Weighting is **implicit** via regression estimation and resulting aggregation of residuals.

- Unlike CS or SA, there are no explicit weights on comparisons; rather, all untreated data are efficiently pooled to fit the imputation model.

- This pooling enhances efficiency but reduces transparency on exact weighting at the contrast level.

### 3.6 Relation and Equivalence to Other Estimators

- Conceptually connected to TWFE but improves upon it by restricting model estimation to untreated observations and imputing untreated outcomes for treated units.

- Is related to CS’s doubly robust approach but emphasizes **single-step imputation** instead of explicit ATT(g,t) contrasts.

- While CS focuses on transparent estimation and weighting of \(ATT(g,t)\), BJS focuses on **optimal prediction of counterfactual untreated outcomes** to recover treatment effects, offering computational efficiency and strong performance in complex data sets.

- Both avoid the forbidden comparisons and negative weights of TWFE.

---

## 4. Comparison of Estimators: Identification, Estimation, and Treatment Effect Heterogeneity

| Aspect                      | Callaway & Sant’Anna (CS)                      | Sun & Abraham (SA)                              | Borusyak, Jaravel & Spiess (BJS)               |
|-----------------------------|-----------------------------------------------|------------------------------------------------|-------------------------------------------------|
| **Parallel Trends Assumption** | Conditional parallel trends on untreated groups (never or not-yet-treated), conditional on covariates and flexible controls (group- and time-specific) | Event-time specific parallel trends controlling with clean control groups (never-treated or last-treated) | Parallel trends encoded via untreated potential outcome regression on untreated units incorporating unit & time fixed effects |
| **Treatment Effect Heterogeneity** | Fully unrestricted heterogeneity across cohorts and calendar time via \(ATT(g,t)\) | Fully unrestricted heterogeneity across cohorts and event time \(CATT(g,e)\) | Fully unrestricted heterogeneity over units and times via imputed individual treatment effects \(\hat{\tau}_{it}\) |
| **Anticipation Effects** | Assumes no anticipation; extendable for partial anticipation via bounding | No anticipation assumed; robustness checks recommended | Explicit modeling and testing of anticipation effects, with bias corrections proposed in recent extensions |
| **Fixed Effects Structure** | Fixed effects absorbed in stage 1 models (regression or weighting); unit & time FE in untreated potential outcomes | Unit & time fixed effects in fully interacted cohort × event-time saturated regression | Unit & time fixed effects estimated on untreated units, used for imputation |
| **Clean Controls Construction** | Never-treated or not-yet-treated units per group-time contrast; excludes treated units as controls | Never-treated or last-treated groups as clean controls for event-time contrasts | Untreated and not-yet-treated units used to fit imputation model (pure controls) |
| **Weighting / Aggregation** | Explicit, convex weights across \(g,t\); aggregation user-specified and transparent | Cohort shares at event times; interaction-weighted averaging avoiding negative weights | Implicit regression weights; efficient pooling; no explicit weighting of contrasts |
| **Estimation Procedure** | Two-stage: (1) estimate \(ATT(g,t)\) via OR/IPW/DR, (2) aggregate; robust inference via multiplier bootstrap | Saturated FE regression with cohort × event-time interactions; aggregate via cohort shares | Single-step regression on untreated units; impute untreated outcomes; calculate differences |
| **Inference** | Bootstrap with simultaneous confidence bands over \(ATT(g,t)\) | Cluster robust standard errors; pointwise CIs for event-time effects | Cluster-robust SEs, joint testing in pretrend testing; specialized anticipation testing |
| **Software Availability** | R package `did` | Stata package `eventstudyinteract` | R package `didimputation`, Stata package `did_imputation` |

---

## 5. Pre-Trend Testing and Best-Practice Recommendations beyond Roth (2022)

### 5.1 Roth (2022) Critiques Recap

- **Low power:** Conventional pretrend tests (e.g., significance of lead coefficients in TWFE) have low power to detect meaningful violations, leading to false confidence.

- **Inference distortion:** Conditioning estimation and inference on passing pretrend tests biases estimates and reduces confidence interval coverage.

- **Misinterpretation:** Non-significant pretrend coefficients do not prove validity of parallel trends.

### 5.2 Methodological Innovations Across Estimators

- **Sun & Abraham:**  
  - Produce **contamination-free event-study leads** by saturating cohort × event-time interactions, unmixing heterogeneous effects, improving interpretability and power of pretrend tests.  
  - However, recommend not conditioning inference on passing pretrend tests.

- **Borusyak, Jaravel & Spiess:**  
  - Implement **untreated-only pretrend tests** which have greater power by restricting testing to untreated units’ outcomes and thus less contamination.  
  - Develop tests for anticipation effects and violations, combined with bias corrections.  
  - Propose joint tests rather than individual lead significance.

- **Callaway & Sant’Anna:**  
  - Use **joint inference and doubly robust methods** incorporating multiple preperiods jointly to enhance power.  
  - Proposed rolling pre-treatment averaging to reduce noise (Lee & Wooldridge, 2023).  
  - Integrate with **Honest DiD** bounding/sensitivity analyses framework (Rambachan & Roth 2022), providing partial identification without relying on pass/fail pretrend tests.

### 5.3 Best-Practice Recommendations

- **Transparent Pretrend Reporting:** Present estimates and confidence intervals of pre-treatment coefficients or pretrend dynamic treatment effects but avoid interpreting them as formal hypothesis tests of assumptions.

- **Do Not Condition Main Inference on Pretrend Tests:** Avoid using pretrend test outcomes (pass/fail) to decide whether to report or accept DiD estimates, as this induces selection bias.

- **Sensitivity and Bounding Analyses:** Employ frameworks like **Honest DiD** that characterize plausible bounds on parallel trends violations rather than point-identify effects under strict assumptions.

- **Power Assessment of Pretrend Tests:** Evaluate the power of pretrend tests to detect violations of specified magnitudes using simulation or available R packages (e.g., `pretrends`).

- **Use Clean Control-Based Pretrend Tests:** Favor untreated-only pretrend tests or contamination-free event-study coefficients (SA) over TWFE lead tests.

- **Robust Inference:** Correctly account for clustering and estimation uncertainty in test statistics.

- **Multiple Diagnostics:** Combine visual inspection of event-study plots, untreated pretrend testing, sensitivity analysis, and economic/structural knowledge to assess plausibility of parallel trends.

---

## 6. Implications for Applied Researchers in Labor and Health Economics

- **Estimator Selection:**

  - Prefer modern estimators (CS, SA, BJS) over TWFE for settings with staggered adoption and expected treatment effect heterogeneity.
  
  - Use **Callaway & Sant’Anna** for transparent estimation of group-time ATT with flexible covariates and comprehensive inference.
  
  - Use **Sun & Abraham** where event-time dynamic patterns and visualization of treatment effect timing are priorities.
  
  - Use **Borusyak, Jaravel & Spiess** for computational efficiency and well-powered pretrend testing, especially with complex or large datasets.

- **Estimation Workflow Suggestion:**

  1. Conduct **Goodman-Bacon decomposition** to diagnose TWFE weighting issues if TWFE is used.  
  2. Estimate detailed event-study dynamics with SA for contamination-free visualization and dynamic pretrend checking.  
  3. Estimate aggregate and cohort-time ATT using CS for robust, transparent estimations with doubly robust options.  
  4. Utilize BJS imputation methods for efficient estimation and power-enhanced pretrend and anticipation effect tests.  
  5. Apply **Honest DiD** or related bounding methods to conduct sensitivity analyses and transparently report identification assumptions.  
  6. Explicitly report pretrend estimates and diagnostic tests without conditioning primary inference on their outcomes.

- **Anticipation Effects:**

  - Test and transparently discuss possible anticipation or misclassification with methods available (particularly BJS approaches).  
  - Incorporate partial identification or sensitivity analysis where anticipation is a concern.

- **Reporting Standards:**

  - Clearly document assumptions regarding parallel trends and anticipation.  
  - Provide explicit descriptions of weighting and fixed effects used.  
  - Transparently display pretrend and placebo test results, with confidence intervals and joint tests.  
  - Avoid over-reliance on conventional linear pretrend null tests.

- **Software and Reproducibility:**

  - Use well-maintained, documented packages (`did`, `eventstudyinteract`, `didimputation`) available in R and Stata.  
  - Provide code and detailed descriptions of estimation steps to allow replication and diagnostics.

---

## Conclusion

The staggered adoption critique articulated by Goodman-Bacon (2021) exposed fundamental flaws in traditional TWFE DiD estimators, especially under treatment effect heterogeneity and dynamic timing. Modern estimators by **Callaway & Sant’Anna**, **Sun & Abraham**, and **Borusyak, Jaravel & Spiess** address these limitations through:

- Rigorous and transparent identification assumptions with explicit parallel trends modeled across groups and time or event time  
- Flexible allowance for unrestricted treatment effect heterogeneity  
- Careful construction of clean control groups  
- Estimation procedures that separate treatment effect estimation, aggregation, and inference to avoid negative weighting and contamination  
- Different but complementary approaches to weighting design and fixed effects inclusion  
- Enhanced pretrend diagnostics and robust inference incorporating Roth (2022)'s critiques, emphasizing transparent reporting, sensitivity analysis, and avoiding conditioning inference on pre-tests

Applied researchers in labor and health economics should adopt these modern approaches with diligent diagnostics and transparent reporting to produce credible and interpretable causal estimates in complex staggered adoption DiD settings.

---

## Sources

[1] Goodman-Bacon (2021) Difference-in-Differences with Variation in Treatment Timing: https://doi.org/10.1007/s00148-021-00819-6  
[2] Callaway & Sant’Anna (2019) Difference-in-Differences with Multiple Time Periods: https://psantanna.com/files/Callaway_SantAnna_2020.pdf  
[3] Callaway & Sant’Anna (2021) Advances in DiD Estimators with Multiple Periods and Staggered Adoption: https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf  
[4] Sun & Abraham (2021) Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects: https://doi.org/10.1016/j.jeconom.2020.07.006  
[5] Borusyak, Jaravel & Spiess (2021) Revisiting Event-Study Designs: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3740221  
[6] Borusyak, Jaravel & Spiess (2024) Staggered Adoption DiD with Misclassification and Anticipation: https://arxiv.org/abs/2507.20415  
[7] Roth (2022) Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends: https://doi.org/10.1257/aeri.20200119  
[8] Rambachan & Roth (2022) Honest DiD: Sensitivity Analysis and Partial Identification in DiD and Event-Study Designs: https://arxiv.org/abs/1909.09602  
[9] R Package `did` for Callaway & Sant’Anna estimator: https://bcallaway11.github.io/did/index.html  
[10] Stata Package `eventstudyinteract` for Sun & Abraham estimator: https://ideas.repec.org/c/boc/bocode/s458978.html  
[11] R Package `didimputation` for Imputation-based Estimator: https://cran.r-project.org/web/packages/didimputation/index.html  

---

*The report synthesizes core theoretical insights and practical estimation details enabling applied researchers to understand, implement, and critically assess modern staggered DiD estimators with current best practices.*