# Advanced Difference-in-Differences (DiD) Methods after the Staggered Adoption Critique: Technical Mechanics, Empirical Practice, and Pre-Trend Test Critique

## Introduction

Following Goodman-Bacon (2021), a major methodological shift has occurred in Difference-in-Differences (DiD) estimation, especially to address biases from treatment effect heterogeneity and staggered adoption across units. New estimator families—Callaway & Sant’Anna (CS), Sun & Abraham (SA), Borusyak, Jaravel & Spiess (BJS), and did2s—have emerged as leading alternatives to the Two-Way Fixed Effects (TWFE) model. This report provides a comprehensive technical deep-dive into the mechanics, assumptions, decomposition structures, and estimation procedures of each estimator. It also traces empirical adoption in leading economics journals (AER, QJE, JPE; labor and health economics, 2020–2024), modal triangulation practices, and critically examines how modern DiD estimators address the Roth (2022) critique of pre-trend testing, emphasizing inference after pretests.

---

## 1. Two-Way Fixed Effects (TWFE) and Its Decomposition in Staggered Adoption

### 1.1 Mechanics and Estimation Logic

The canonical TWFE DiD estimator regresses the outcome on group (unit) and period fixed effects, and a treatment indicator:

\[
Y_{it} = \alpha_i + \lambda_t + \delta \cdot D_{it} + \epsilon_{it}
\]

- \( Y_{it} \): outcome for unit \( i \) at time \( t \)
- \( \alpha_i \): unit fixed effect
- \( \lambda_t \): time fixed effect
- \( D_{it} \): indicator for treatment (1 if treated, 0 else)
- \( \delta \): DiD estimator for policy effect

With **staggered adoption**, units are treated at different times. The TWFE model can be extended to event-study regressions:

\[
Y_{it} = \alpha_i + \lambda_t + \sum_{l \neq -1} \beta_l \cdot \mathbb{1}[event\_time = l] + \epsilon_{it}
\]

where \( event\_time = t - G_i \) (the number of periods relative to the group’s first treatment date).

**Key Issue:** When treatment effects are dynamic/heterogeneous, TWFE averages across all possible two-group, two-period DiD contrasts, mixing never-treated, earlier-, and later-treated units as controls—even after their own treatment. This can create:
- **Negative weights** on some group-time combinations
- Bias toward non-zero pre-trend estimates
- Estimands that do not map to interpretable causal parameters

### 1.2 Goodman-Bacon Decomposition

Goodman-Bacon (2021) [1,2]:
- Shows TWFE equals a **weighted average of all possible 2×2 DiD comparisons**, including "forbidden" ones (e.g., late-treated units acting as controls after their own treatment).
- Decomposition:

\[
TWFE = \sum_{g,a} w_{g,a} \cdot ATT_{g,a}
\]

where \( ATT_{g,a} \) are group (g)-by-adoption (a) average treatment effects, and \( w_{g,a} \) may be **negative**.

- **Implication:** In staggered designs with treatment effect heterogeneity, TWFE can be invalid—even sign-reversed [1,3].

### 1.3 Identification Assumptions

- **Parallel Trends:** \( E[Y_{it}(0) - Y_{i,t-1}(0) | G_{i}=g] = E[Y_{it}(0) - Y_{i,t-1}(0) | G_{i}=a] \) for any two groups.
- **Homogeneous Treatment Effects:** Assumed implicitly; needed for unbiasedness.
- **No Anticipation**: Units do not respond prior to treatment.
- **No Interference (SUTVA):** Each unit’s outcome is unaffected by others’ treatment.

### 1.4 Handling Heterogeneity, Staggered Adoption, and Dynamics

- **Does not accommodate heterogeneity or dynamic adoption robustly:** Can yield misleading results when these are present [2].

### 1.5 Empirical Adoption

- **Pre-2021 modal estimator.** Rapidly being replaced in high-profile work; now most often reported for comparison/benchmarking only.

---

## 2. Callaway & Sant’Anna (CS) Group-Time ATT Estimator

### 2.1 Mechanics and Identification

CS (2021) [4,5]:
- **Parameter:** Group-time average treatment effect, \( ATT(g, t) \): effect in period \( t \) for cohort \( g \) (first treated in period \( g \)).
- **Identification:** For each (g, t), compare units first treated at \( g \) (and not before) with controls: never-treated or "clean" not-yet-treated units (treated after \( t \)).
- **Equation:**

\[
ATT(g, t) = E[Y_{it}(1) - Y_{it}(0) | G_i = g]
\]

- **Parallel Trends (Conditional/Unconditional):**

\[
E[Y_{it}(0) - Y_{i, t-1}(0) | G_i = g, X_i] = E[Y_{it}(0) - Y_{i, t-1}(0) | C_i, X_i]
\]

where \( X_i \) are covariates, \( C_i \) denotes control group.

- **No Anticipation:** Can allow for explicit anticipation window \( \delta \).

### 2.2 Decomposition and Aggregation

- **Estimation consists of:**
  1. Compute all valid \( ATT(g, t) \) contrasts, using only clean controls for each (exclude any units already treated).
  2. **Aggregate** using interpretable weights:
      - *Simple average* over all ATTs
      - *Event-study style*: by time since treatment
      - *Group-specific* or *calendar time*-specific
      - User can define weights matching their estimand (e.g., average for first-year post-treatment)

- **Flexibility:** No homogeneity requirement; full heterogeneity and dynamic effects accommodated.

### 2.3 Estimation and Software

- Open source implementations:
    - R: `did` package [`att_gt()`, `aggte()`] [6,7]
    - Stata: `csdid` [8]
- **Estimation Steps:**
  - Specify outcome, time, cohort, and control group.
  - Estimate ATT(g,t) using **doubly robust/ IPW/ regression** estimator.
  - Use multiplier bootstrap for simultaneous confidence intervals.
  - Aggregate and visualize (e.g., event-study plot with `ggdid()`).

### 2.4 Requirements for Comparison Groups

- **Cohort-specific:** Controls must be either never-treated or units not treated yet as of \( t \) (not previously treated).
- **Anticipation:** Tune the anticipation window (periods prior to treatment may be excluded if anticipation is plausible).

### 2.5 Handling Heterogeneity, Dynamics, Parallel Trends, Anticipation

- Fully heterogeneity- and dynamic-adoption robust.
- Parallel trends can be conditional/unconditional, depending on design.
- Allows explicit anticipation window.
- **Homogeneity not required**: target parameter is a (possibly non-linear) average of cohort-period effects.

---

## 3. Sun & Abraham (SA) Interaction-Weighted (Event-Study) Estimator

### 3.1 Mechanics and Identification

Sun & Abraham (2021) [9,10]:
- **Motivation:** TWFE event-study regression coefficients are contaminated by treatment effect heterogeneity and staggered adoption timing.
- **Estimator:** Constructs **cohort-by-event-time** effects and aggregates them with cohort shares.

#### Equation:

\[
Y_{it} = \alpha_i + \lambda_t + \sum_{g~\in~G}\sum_{l~\in~L} \beta_{g,l} \cdot D_{g,l,it} + \epsilon_{it}
\]

- \( D_{g,l,it} \): indicator for cohort \( g \) at event time \( l \) (relative to own adoption).
- For each event time \( l \), aggregate as:

\[
\beta_l^{IW} = \sum_{g} w_g(l) \cdot \beta_{g,l}
\]

where \( w_g(l) \) is the share of cohort \( g \) in the sample at event time \( l \).

- **Parallel Trends:** Assumed for untreated (never/not-yet treated) at each analysis time.
- **No Anticipation**, **No Interference**.

### 3.2 Estimation Steps and Implementation

- **Step-by-step:**
  1. For every cohort and event time, define indicators.
  2. Run regression with these interactions alongside standard fixed effects.
  3. Compute cohort/event-time-specific effects and aggregate by event time using respective weights.

- **Software:**
  - Stata: `eventstudyinteract` [11]
  - R: `fixest::sunab()` [12]

### 3.3 Handling Heterogeneity, Dynamics, Anticipation

- **Handles full heterogeneity**: event-time effects can vary by cohort.
- Weights are **never negative**; estimator is a convex combination of group-time effects.
- Allows flexible exploration of dynamic patterns and modeling of anticipation as “leads”.
- **No homogeneity assumption required**.

### 3.4 Comparison vs. TWFE

- Event-study plots constructed using IW estimator have **direct causal interpretation** for given event time—unlike TWFE, which can mislead due to contamination [9].

---

## 4. Imputation-Based Approaches: Borusyak, Jaravel & Spiess (BJS) and did2s

### 4.1 BJS Imputation-Based Estimator

#### Mechanics

Borusyak, Jaravel & Spiess (BJS, 2021) [13,14]:
- **Two-step logic:**
  1. **Estimate untreated outcome model** (\( Y_{it}(0) \)) from never- and not-yet-treated units. Model can be regression, machine learning, or matching.
  2. **Impute counterfactuals**: Predict each treated unit/time's \( \hat{Y}_{it}(0) \).
  3. **Estimate treatment effect**: \( \tau_{it} = Y_{it} - \hat{Y}_{it}(0) \)
  4. **Aggregate**: Compute ATT (static or dynamic/event-study) with chosen weights.

- Allows arbitrary covariates and flexible functional-form assumptions.

#### Equations

\[
ATT_{w} = \sum_{i,t \in \Omega_1} w_{it} [ Y_{it} - \hat{Y}_{it}(0) ]
\]

where \( w_{it} \) are pre-specified weights over treated units/time.

#### Assumptions

- **(Conditional) Parallel Trends:** Validity hinges on model for untreated evolution.
- **No Anticipation:** Responses before adoption cannot be attributed to treatment.

- **Identification flexible to conditioning on observed variables.**

#### Software

- R package: `didimputation` [15]

#### Implementation Steps

- Choose untreated periods and units for model fitting.
- Fit model for \( Y_{it}(0) \).
- Impute counterfactuals, compute differences, aggregate.
- Supports event-study and static ATT.

#### Heterogeneity, Dynamics, Anticipation

- Full heterogeneity and dynamic adoption robust; design accommodates dynamic effects.
- Functional-form/modeling flexibility is both a strength (customizability) and a responsibility (risk of misspecification).

### 4.2 did2s (Two-Stage DiD, Gardner 2021/2022)

#### Mechanics

- **Step 1:** Estimate unit and time fixed effects (sometimes with covariates) **using only untreated data** (never- and not-yet-treated).
- **Step 2:** Use these to flexibly "differences out" unit/time trends, compute \( \hat{Y}_{it}(0) \), then regress residualized outcomes on treatment. Static or dynamic effects as desired [16,17].
- Returns estimates robust to functional-form misspecification (relative to TWFE) and reliable standard errors.

#### Software

- R package: `did2s` [18]
- Stata: `did2s_stata` [19]

#### Event-Study and Workflow

- Event-study: residualize outcome in untreated units, estimate event-time profile on differences.
- Supports standard methods for inference (bootstrapping, clustering, etc.).

#### Strengths and Weaknesses

- Same identification assumptions as above.
- Can be efficiently implemented on large panels.
- Interpretation is ATT on treated, can be weighted as desired.

---

## 5. Comparative Summary: Handling Key Methodological Issues

| Estimator         | Hetero. Effects | Dynamic Adoption | Parallel Trends         | Homogeneity Needed | Anticipation | Explicit Equations/Steps |
|-------------------|------------------|------------------|------------------------|--------------------|--------------|-------------------------|
| TWFE              | No               | No               | Unconditional (often)  | Yes                | No           | Yes; see (1.1-1.2)      |
| Callaway–Sant’Anna| Yes              | Yes              | Cond./Uncond.          | No                 | Yes (delta)  | Yes; (2.1-2.4)          |
| Sun–Abraham       | Yes              | Yes              | Unconditional          | No                 | Yes (leads)  | Yes; (3.1-3.3)          |
| Borusyak et al., did2s | Yes         | Yes              | Cond./Uncond.          | No                 | Yes          | Yes; (4.1-4.2)          |

**All modern estimators avoid negative weights, disentangle dynamic/differential effects, and report interpretable causal targets**.

---

## 6. Empirical Adoption and Workflow in Top Economics Journals (2020–2024)

### 6.1 Adoption Patterns

- **Pre-2021:** Most papers relied on TWFE models for DiD.
- **Post-2021 (following Goodman-Bacon):** Rapid, widespread adoption of CS, SA, and imputation variants (BJS, did2s) in AER/QJE/JPE, especially in labor and health economics.
- **Typical workflow (2022–2024):**
    - Replication of results with **multiple estimators**: TWFE, CS, SA, BJS/did2s [20,21].
    - Event-study plots and pre-trend diagnostics (using robust estimators, not just TWFE).
    - Explicit **reporting of group-period ATTs** (cohort dynamics).
    - Placebo/falsification/lead-lag checks to explore anticipation and robustness [22].
    - **Conditional estimator choice:** More complex designs (many groups, strong covariate trends, few never-treated) favor BJS/did2s; cohort-ATT and dynamic patterns: CS/SA.

### 6.2 Modal Triangulation Practices

- Use at least two modern estimators (CS, SA, BJS/did2s) and compare to TWFE.
- Report both static and dynamic/event-study effects.
- Visualize event-time profiles, explore sensitivity to grouping/weighting.
- Justify estimator choice by simulation, diagnostic plots, and robustness checks [21,23].

---

## 7. Connections to Roth (2022) and Inference after Pre-Trend Testing

### 7.1 The Problem with Conventional Pre-Trend Tests

Roth (2022) [24,25]:
- Conventional pre-trend tests (test pre-treatment event study leads ≠ 0) have low power.
- Conditioning on passing such tests can magnify post-treatment estimation bias, notably under violations of parallel trends (e.g., smooth drifts), especially in event-study setups and under staggered adoption.
- **Key result:** Even if pre-trend coefficients are "insignificant," post-treatment estimates may remain severely biased.

### 7.2 How Modern DiD Estimators Respond

- **Design avoids contaminated comparisons:** Modern estimators explicitly use only never-/not-yet-treated units as controls, so pre-period "treatment" coefficients reflect only valid comparisons [10,11,13].
- **Explicit separation of identification and inference:** E.g., CS & SA methods make clear what is being compared in each estimate.
- **Incorporation with HonestDiD:** Sensitivity analysis tools such as HonestDiD [26,27] quantify robustness of findings to plausible pre-trends. Researchers can:
    - Impose smoothness or size restrictions on post/pre–treatment trend violations.
    - Construct "honest" confidence intervals accounting for possible deviations.
    - Report "breakdown points" where parallel trends violations would overturn findings.
- **Empirical workflow:** Routine use of event-study plots with simultaneous confidence bands, reporting both pre- and post-period effects, sensitivity to modelling choices, and leverage of HonestDiD.

---

## 8. Conditional Guidance on Estimator Choice

- **If** treatment effects are expected to be homogeneous, TWFE/TWFE-event study may suffice—but trend is to report CS/SA/BJS/did2s as main estimates for transparency.
- **With** substantial heterogeneity, staggered timing, or concern for dynamic patterns:
    - **For ATT(g,t), aggregation/change by cohort/event-time, or exploration of anticipation:** Use CS or SA (e.g., labor economics, staggered program rollouts).
    - **With many covariates, complex panel structures, or limited never-treated units:** BJS/did2s provides maximum flexibility (e.g., health policy evaluations with rich covariate data).
    - **If anticipating reviewer expectations (for recent AER/QJE/JPE):** Report all, justify via simulation/diagnostics, and conduct robust sensitivity checks with HonestDiD.

---

## 9. Conclusion

The post–Goodman-Bacon recognition of TWFE flaws in staggered DiD settings has led to a paradigm shift in empirical economics, especially within labor and health policy research. Callaway & Sant’Anna, Sun & Abraham, and imputation-based estimators (BJS, did2s), underpinned by clear mechanics, identification assumptions, and robust software implementations, have become dominant. Modern workflows triangulate across estimator families, use diagnostic and sensitivity tools (notably HonestDiD), and anchor inference in credible, interpretable causal structures. The field now emphasizes both identification discipline and transparent reporting of robustness to parallel trends violations, setting a new standard for DiD-based causal inference.

---

## Sources

1. Goodman-Bacon, A. (2021). Difference-in-Differences with Variation in Treatment Timing. [https://www.aeaweb.org/articles?id=10.1257/aer.20181047](https://www.aeaweb.org/articles?id=10.1257/aer.20181047)
2. Two-Way Fixed Effects and Differences-in-Differences with Heterogeneous Treatment Effects, de Chaisemartin & D’Haultfœuille. [https://www.aeaweb.org/conference/2022/preliminary/paper/s38GffaD](https://www.aeaweb.org/conference/2022/preliminary/paper/s38GffaD)
3. The Problems of TWFE with Staggered Treatment Adoption, Sant’Anna. [https://psantanna.com/DiD/11_Staggered_problems.pdf](https://psantanna.com/DiD/11_Staggered_problems.pdf)
4. Callaway, B., & Sant’Anna, P. H. C. (2021). Difference-in-Differences with Multiple Time Periods. [https://econpapers.repec.org/RePEc:eee:econom:v:225:y:2021:i:2:p:200-230](https://econpapers.repec.org/RePEc:eee:econom:v:225:y:2021:i:2:p:200-230)
5. Introduction to DiD with Multiple Time Periods • did. [https://bcallaway11.github.io/did/articles/multi-period-did.html](https://bcallaway11.github.io/did/articles/multi-period-did.html)
6. Getting Started with the did Package • did (R documentation). [https://bcallaway11.github.io/did/articles/did-basics.html](https://bcallaway11.github.io/did/articles/did-basics.html)
7. Group-Time Average Treatment Effects — att_gt • did (R function docs). [https://bcallaway11.github.io/did/reference/att_gt.html](https://bcallaway11.github.io/did/reference/att_gt.html)
8. csdid: Difference-in-Differences with Multiple Time Periods in Stata. [https://www.stata.com/meeting/us21/slides/US21_SantAnna.pdf](https://www.stata.com/meeting/us21/slides/US21_SantAnna.pdf)
9. Sun, L., & Abraham, S. (2021). Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects. [https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html](https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html)
10. Replicating the Sun & Abraham event study fix by hand. [https://apithymaxim.wordpress.com/2025/02/19/replicating-the-sun-abraham-event-study-fix-by-hand/](https://apithymaxim.wordpress.com/2025/02/19/replicating-the-sun-abraham-event-study-fix-by-hand/)
11. eventstudyinteract: Stata package documentation. [https://github.com/lsun20/EventStudyInteract](https://github.com/lsun20/EventStudyInteract)
12. fixest R package: sunab() event-study vignette. [https://asjadnaqvi.github.io/DiD/docs/code_r/07_sunab_r/](https://asjadnaqvi.github.io/DiD/docs/code_r/07_sunab_r/)
13. Borusyak, K., Jaravel, X., & Spiess, J. (2023). Revisiting Event Study Designs: Robust and Efficient Estimation. [https://ideas.repec.org/p/arx/papers/2108.12419.html](https://ideas.repec.org/p/arx/papers/2108.12419.html)
14. Revisiting Event Study Designs: Robust and Efficient Estimation (Slides). [https://francescoruggieri.github.io/files/DiDES05_BorusyakJaravelSpiess.pdf](https://francescoruggieri.github.io/files/DiDES05_BorusyakJaravelSpiess.pdf)
15. R package didimputation. [https://cran.rstudio.com/web/packages/didimputation/didimputation.pdf](https://cran.rstudio.com/web/packages/didimputation/didimputation.pdf)
16. Gardner, J. (2021). Two-stage difference-in-differences. [https://pdhp.isr.umich.edu/wp-content/uploads/2023/01/DiD_PDHP.pdf](https://pdhp.isr.umich.edu/wp-content/uploads/2023/01/DiD_PDHP.pdf)
17. Two-Stage Difference in Differences (did2s) vignette. [https://kylebutts.github.io/did2s/articles/Two-Stage-Difference-in-Differences.html](https://kylebutts.github.io/did2s/articles/Two-Stage-Difference-in-Differences.html)
18. did2s: Calculate two-stage difference-in-differences (R) [https://cran.r-project.org/web/packages/did2s/did2s.pdf](https://cran.r-project.org/web/packages/did2s/did2s.pdf)
19. did2s Stata Package GitHub. [https://github.com/kylebutts/did2s_stata](https://github.com/kylebutts/did2s_stata)
20. Designing Difference-in-Difference Studies with Staggered Treatment Adoption: Key Concepts and Practical Guidelines. [https://www.annualreviews.org/content/journals/10.1146/annurev-publhealth-061022-050825](https://www.annualreviews.org/content/journals/10.1146/annurev-publhealth-061022-050825)
21. Chen Xing, "Notes on Callaway–Sant'Anna (2021) – Staggered Adoption DiD." [https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did/](https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did/)
22. Designing Difference in Difference Studies With Staggered ... - NBER Working Paper. [https://www.nber.org/system/files/working_papers/w31842/w31842.pdf](https://www.nber.org/system/files/working_papers/w31842/w31842.pdf)
23. Comparative Evaluation of Difference in Differences Methods for Staggered Adoption Interventions. [https://arxiv.org/html/2508.14365](https://arxiv.org/html/2508.14365)
24. Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. [https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf](https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf)
25. Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends – AEA. [https://www.aeaweb.org/articles?id=10.1257/aeri.20210236](https://www.aeaweb.org/articles?id=10.1257/aeri.20210236)
26. HonestDiD R package. [https://cran.r-project.org/web/packages/HonestDiD/HonestDiD.pdf](https://cran.r-project.org/web/packages/HonestDiD/HonestDiD.pdf)
27. HonestDiD GitHub. [https://github.com/asheshrambachan/HonestDiD](https://github.com/asheshrambachan/HonestDiD)