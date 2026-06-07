# Advances in Difference-in-Differences (DiD) Methods for Staggered Adoption: Technical Comparison, Empirical Adoption, and Response to Pre-Trend Critiques

## Introduction

Recent methodological advances in Difference-in-Differences (DiD) estimation have profoundly impacted applied economics, especially in light of the “staggered adoption” critique, which highlighted biases in traditional two-way fixed effects (TWFE) approaches when treatment is adopted at different times across units and treatment effects are heterogeneous. This report comprehensively analyzes three leading solutions—Callaway & Sant'Anna’s two-stage aggregation, Sun & Abraham’s interaction-weighted estimator, and Borusyak, Jaravel, and Spiess’s imputation-based method—focusing on how they handle treatment effect heterogeneity, dynamic/staggered timing, and performance assumptions. It compares their adoption in top economics journals (AER, QJE, JPE) for labor and health economics, summarizes empirical practice regarding estimator justification, and examines how these estimators address Roth’s (2022) critique of standard pre-trend tests.

## 1. Technical Comparison of Leading Staggered DiD Estimators

### 1.1 The Staggered Adoption Problem: Why Robust Alternatives Are Needed

The classic TWFE DiD estimator can produce severely biased or even sign-reversed estimates when treatments occur in different time periods and effects vary across units or time. This is due to "forbidden comparisons," where already-treated units are inappropriately used as controls, contaminating both static and dynamic effect estimates—even when the parallel trends assumption holds. The result is negative weighting and uninterpretable average treatment effect on the treated (ATT) estimates[^1,^2].

Robust DiD estimators for staggered adoption have been developed to:
- Allow for heterogeneous treatment effects,
- Correctly handle variation in treatment timing,
- Avoid negative/compositional weights,
- Clearly target and estimate interpretable causal parameters.

### 1.2 Callaway & Sant'Anna (2021): Two-Stage Aggregation (Group-Time ATT) Method

**Core Features:**  
- Decomposes treatment effect estimation by group (cohort first treated in period g) and time (period t), estimating ATT(g, t) for each group-time pair.
- Avoids “forbidden comparisons” by using only never-treated or not-yet-treated units as controls.
- Allows for arbitrary heterogeneity: estimates are valid with time-varying, group-specific, or event-time-specific effects.

**Assumptions:**
- **Parallel Trends:** Assumes parallel trends in untreated potential outcomes, either unconditional or conditional on observed covariates[^3].
- **Treatment Irreversibility:** Once treated, unit remains treated.
- **Anticipation Effects:** Limited or known; methods allow users to model anticipation windows.

**Treatment Effect Heterogeneity & Timing:**  
Handles both cleanly. Researchers can estimate and aggregate treatment effects by cohort, by event time, or overall. Allows flexibility in aggregation to match the empirical question[^3,^4].

**Estimation & Implementation:**  
Open-source software (e.g., R package `did`) implements regression, IPW, and doubly robust estimators, with robust variance estimation and uniform confidence bands[^3,^4].

### 1.3 Sun & Abraham (2021): Interaction-Weighted (Event-Study) Estimator

**Core Features:**  
- Constructs an event-study design by interacting treatment leads/lags with cohort dummies, generating event-time-specific effects by cohort[^5].
- Corrects for dynamic and heterogeneous effects generating misleading inference in TWFE event studies—no more automatic zero pre-trends or identically scaled post-trends.

**Assumptions:**
- **Parallel Trends:** Standard assumption, using never-treated or not-yet-treated as control groups[^5].
- **No Need for Homogeneity:** Explicitly allows for heterogeneity across cohorts and event times.
- **Anticipation Effects:** Researchers must specify leads to capture anticipation; assumptions about anticipation can be tested.

**Treatment Effect Heterogeneity & Timing:**  
Event-study plots are robust—coefficients for each event time correspond to effects for specific cohorts, not mishmashes contaminated by other groups or event times[^5]. Yields interpretable plots and estimates for dynamic treatment effects.

**Estimation & Implementation:**  
User-friendly implementations in R and Stata are available (e.g., `fixest` and custom code), facilitating routine adoption[^5,^6].

### 1.4 Borusyak, Jaravel, and Spiess: Imputation-Based Estimator

**Core Features:**  
- Models untreated (“counterfactual”) outcomes for each treated unit and time, using information from never-treated and not-yet-treated units[^7].
- The difference between actual and imputed untreated outcome for each unit-time constitutes the estimated effect.
- Approaches estimation flexibly: can use regression, matching, or modern ML to specify untreated outcome models.

**Assumptions:**
- **Conditional Parallel Trends:** Requires modeled and matched untreated outcomes demonstrate parallel trends[^7,^8].
- **No Homogeneity Required:** Estimator is robust to treatment effect heterogeneity.
- **Anticipation Effects:** Requires careful attention; can be flexibly incorporated into modeling.

**Treatment Effect Heterogeneity & Timing:**  
Handles both static and dynamic effects, including arbitrary treatment timing, with flexibility as to how control group and timing structure are specified at the modeling stage[^7,^8].

**Estimation & Implementation:**  
Implemented in R (package `didimputation`). Suitable for complex covariate structures and feasible with large datasets[^7,^8].

### 1.5 Summary Table: Handling of Core Methodological Issues

| Method                | Heterogeneous Effects | Flexible Timing | Parallel Trends | Homogeneity Needed? | Anticipation Effects |
|-----------------------|----------------------|----------------|-----------------|--------------------|---------------------|
| Callaway & Sant'Anna  | Yes                  | Yes            | Conditional/Unconditional | No                 | Modelable/Limited   |
| Sun & Abraham         | Yes                  | Yes            | Standard         | No                 | Modelable/Explicit  |
| Borusyak et al.       | Yes                  | Yes            | Conditional      | No                 | Modelable/Explicit  |

Each method avoids the negative weights problem and enables interpretable estimation in staggered designs[^1,^3,^5,^7].

## 2. Empirical Adoption in Top Economics Journals (2020–2024)

### 2.1 Evidence from Reviews and Applied Papers

- Major survey and synthesis papers confirm that **Callaway & Sant'Anna (2021)** and **Sun & Abraham (2021)** have effectively become the “default” approaches among new empirical studies using DiD in top economics journals when staggered adoption is present, particularly in labor and health economics[^9,^10,^11].
- Borusyak et al. (imputation) estimators are gaining traction, especially in studies with more complex covariate or outcome modeling needs[^11,^12].
- While no meta-analysis systematically tallies empirical adoption in AER, QJE, and JPE, recent review articles highlight rapid take-up of these robust methods in published applications, often replacing or running in parallel to TWFE for comparison[^9,^10,^11].

### 2.2 Justification Practices: Simulations, Sensitivity Analysis, Diagnostics

- Leading empirical papers—and reviewer checklists—since 2021 frequently require:
    - Event-study analysis with robust (heterogeneity-aware) estimators,
    - Sensitivity analyses (e.g., checking robustness to parallel trends violations using HonestDiD or similar[^13]),
    - Monte Carlo simulations, especially for high-profile causal studies[^11,^14],
    - Explicit diagnostic plots for treatment timing and group-by-time effects[^10].

- There is a strong move away from simply reporting TWFE estimates without further diagnostics or robustness checks[^9,^10]. Authors are now expected to:
    - Clearly identify the estimand (e.g., ATT(g, t), dynamic effects),
    - Report both static/aggregrate and dynamic/event time estimates,
    - Test alternative aggregation/weighting schemes,
    - Run placebo and lead–lag pre-trend tests (but see Section 3 for limitations).

### 2.3 Methodological Dominance and Trends

- **Callaway & Sant’Anna** and **Sun & Abraham** methods can be considered methodologically dominant for labor and health economics DiD publications in the 2020–2024 period, as supported by evidence from practitioner reviews, empirical adoption, and software tools[^9,^10,^11].
- The Borusyak et al. imputation approach is popular for complex designs and gaining further ground[^11,^12].
- Most empirical papers now explain estimator choice and frequently demonstrate robustness—at minimum, juxtaposing TWFE with one or more robust estimator, running diagnostics, and discussing parallel trends via event-study or placebo analysis[^9,^10,^14].

## 3. Addressing Roth (2022) and Pre-Trend Testing in Modern DiD

### 3.1 Roth (2022) Critique of Pre-Trend Testing

- Roth’s critique: Standard pre-trend tests (e.g., testing for insignificant pre-treatment leads in an event study) have low power and may create false security. Conditioning on passing a pre-trend test can exacerbate bias; under realistic violations, post-treatment estimates can remain substantially biased even when pre-tests are “passed”[^15,^16].
- Pre-trends are informative but should not be relied on mechanically; context and economic reasoning should guide interpretation.

### 3.2 Response of Modern DiD Estimators

#### Callaway & Sant’Anna

- Provide pre-testing tools for the parallel trends assumption at the group-time level, both graphically and via statistical tests[^3,^4].
- Crucially, separation of identification and inference steps clarifies where pre-trend information is being used.
- Directly compatible with sensitivity analysis tools (e.g., HonestDiD by Rambachan & Roth), allowing researchers to quantify how much findings depend on assumed trend restrictions. This provides partial identification and uniform confidence intervals accounting for plausible violations[^13,^17,^18].

#### Sun & Abraham

- Demonstrate that event-study coefficients are only meaningful if comparisons are uncontaminated; otherwise, pre-trends may arise spuriously from heterogeneity in treatment effects[^5].
- Emphasize robust construction of event-study plots, avoiding contaminated comparisons (i.e., only never-treated or not-yet-treated as controls).
- Recommend augmentation by HonestDiD tools for quantifying identification robustness to pre-trend violations[^13,^17].

#### Borusyak et al.

- Acknowledge Roth’s critique and provide guidance for robustly testing for violations, including pre- and post-treatment periods.
- Event-study and placebo estimates within their framework are constructed to avoid contamination and facilitate integration with sensitivity/robustness analysis[^7,^12].

### 3.3 HonestDiD and Sensitivity Analysis: New Standard

- HonestDiD framework (Rambachan & Roth) is increasingly standard, as it:
    - Does not assume exact parallel trends,
    - Allows researchers to impose smoothness or size restrictions on possible trend violations,
    - Provides formal inference accounting for both sampling and identification uncertainty,
    - Is directly implementable with robust DiD estimators from Callaway & Sant’Anna, Sun & Abraham, and Borusyak et al. (R packages available)[^13,^17,^18].
- Best practice: Use robust DiD estimators, complemented by formal sensitivity analysis (not just blanket pre-trend tests) to inform both estimation and possibility of bias[^13,^17].

### 3.4 Practical Guidance

- Researchers are encouraged to:
    - Define the target causal parameter and ensure estimator aligns with it,
    - Plot and examine event-study dynamics with cohort-specific (robust) estimators,
    - Use placebo (pre-treatment) and sensitivity analyses to assess reliance on parallel trend assumptions[^10,^13,^14],
    - Report estimates using multiple robust estimators when possible.

## 4. Conclusion

Methodological advances following the staggered adoption critique have reshaped empirical practice in applied economics. Robust DiD estimators—especially those of Callaway & Sant'Anna, Sun & Abraham, and Borusyak et al.—now dominate empirical work in top journals for labor and health economics, superseding the flawed TWFE approach in settings with heterogeneous and dynamic treatment effects. These methods directly address concerns regarding dynamic timing, treatment effect heterogeneity, and aggregation, and enable researchers to transparently separate identification, estimation, and aggregation steps.

In response to critiques like Roth (2022), these estimators are now routinely coupled with formal sensitivity analyses such as HonestDiD, ensuring that inference reflects plausible uncertainties in the parallel trends assumption, rather than relying solely on underpowered pre-trend statistical tests.

As a result, empirical DiD work today is both more robust and more transparent, setting new standards for methodological rigor in the estimation of causal effects in staggered adoption settings.

---

## Sources

[1] The Difference-in-Difference Revolution - LearnEconomicsOnline: https://learneconomicsonline.com/blog/archives/1598  
[2] Two-Way Fixed Effects and Differences-in-Differences with Heterogeneous Treatment Effects: A Survey (de Chaisemartin & D’Haultfœuille): https://www.aeaweb.org/conference/2022/preliminary/paper/s38GffaD  
[3] Difference-in-Differences with multiple time periods (Callaway & Sant’Anna 2021): https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Callaway_2021_Difference-in-Differences_with_multiple_time_periods.pdf  
[4] “Difference-in-Differences with multiple time periods” Callaway (webinar slides): https://francescoruggieri.github.io/files/DiDES04_CallawaySantAnna.pdf  
[5] Estimating dynamic treatment effects in event studies with heterogeneous treatment effects (Sun & Abraham 2021): https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html  
[6] R packages | DiD: https://asjadnaqvi.github.io/DiD/docs/02_R  
[7] Revisiting Event Study Designs: Robust and Efficient Estimation (Borusyak, Jaravel, Spiess): https://ideas.repec.org/p/arx/papers/2108.12419.html  
[8] didimputation: Imputation Estimator from Borusyak, Jaravel, and Spiess (2021): https://rdrr.io/cran/didimputation/  
[9] Advances in Difference-in-differences Methods for Policy Evaluation: https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929/  
[10] Modern Difference-in-Differences (Sant’Anna): https://psantanna.com/DiD/NABE_202410.pdf  
[11] What’s trending in difference-in-differences? A synthesis of the recent econometrics literature: https://www.sciencedirect.com/science/article/abs/pii/S0304407623001318  
[12] GitHub - borusyak/did_imputation: https://github.com/borusyak/did_imputation  
[13] HonestParallelTrends_Main.pdf (HonestDiD by Rambachan & Roth): https://www.jonathandroth.com/assets/files/HonestParallelTrends_Main.pdf  
[14] Handling Parallel Trends Violations with HonestDiD in R: https://www.tilburgsciencehub.com/topics/analyze/causal-inference/did/honest-did/  
[15] Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends (Roth, 2022): https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf  
[16] Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends – AEA: https://www.aeaweb.org/articles?id=10.1257/aeri.20210236  
[17] HonestDiD R Package: https://cran.r-project.org/web/packages/HonestDiD/HonestDiD.pdf  
[18] Pre-Testing in a DiD Setup using the did Package • did: https://bcallaway11.github.io/did/articles/pre-testing.html