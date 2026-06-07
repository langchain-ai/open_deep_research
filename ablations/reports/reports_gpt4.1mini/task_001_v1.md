# Methodological Tensions and Advances in Difference-in-Differences Estimation Post-Goodman-Bacon (2021)

## Introduction

Difference-in-Differences (DiD) estimators have long been a cornerstone of policy evaluation when randomized experiments are not feasible. The traditional Two-Way Fixed Effects (TWFE) estimator was widely adopted for staggered adoption designs, where units receive treatment at different points in time. However, Goodman-Bacon (2021) exposed critical methodological tensions in using TWFE under staggered treatment timing and heterogeneous effects.

This report comprehensively analyzes the methodological criticisms introduced by Goodman-Bacon (2021) and compares the modern alternative estimation approaches advanced notably by Callaway and Sant’Anna (2019, 2021), Sun and Abraham (2020, 2021), and Borusyak, Jaravel, and Spiess (2021, 2024). The report compares these methods in terms of their treatment of heterogeneous treatment effects and dynamic timing, their key underlying assumptions—particularly parallel trends, treatment homogeneity, and anticipation effects—and evaluates their empirical adoption within top applied economics journals (American Economic Review, Quarterly Journal of Economics, Journal of Political Economy) in labor and health economics articles from 2020 to 2024.

Finally, the report discusses how these novel estimators respond to Roth’s (2022) critique regarding the limitations of pre-trend testing, highlighting areas for future research and methodological refinement.

---

## Goodman-Bacon (2021) Critique: Methodological Tensions in Staggered Adoption DiD

Goodman-Bacon (2021) illuminated fundamental issues in applying the TWFE DiD estimator in settings where treatment is adopted at different times ("staggered adoption"):

- **TWFE as a Weighted Average of 2×2 DiD Comparisons:** Goodman-Bacon showed TWFE estimates can be decomposed into a weighted average of all possible two-group/two-period DiD contrasts, including comparisons of early treated groups to later treated groups [22].  
- **Negative and Non-Convex Weights Under Heterogeneity:** When treatment effects vary across groups or over time ("heterogeneous treatment effects"), TWFE places negative weights on some comparison groups, effectively using "already treated" units as controls, which biases causal estimates and can lead to misleading or even counterintuitive sign reversals in estimated effects [22, 23].  
- **Bias from "Forbidden Comparisons":** Comparing earlier-treated units to later-treated units, instead of using clean untreated control groups, introduces bias in TWFE estimates since parallel trends cannot hold between already treated and not-yet treated cohorts [22].  
- **Assumptions Under Scrutiny:** The TWFE approach implicitly requires both parallel trends *and* constant, homogeneous treatment effects to validly estimate a causal parameter. Failures of either assumption invalidate usual interpretation [23].  
- **Practical Remedies Suggested:** Goodman-Bacon encouraged analysts to examine decompositions to detect problematic weights and to restrict comparisons to "clean" control groups (ones never treated or not yet treated) [22].

This critique catalyzed a reconsideration of standard DiD methods and motivated the development of alternative estimation approaches explicitly designed to overcome these pitfalls.

---

## Alternative Solutions to Staggered Adoption Challenges

### Callaway and Sant'Anna's Two-Stage Aggregation Approach

Callaway and Sant'Anna (2019, 2021) developed an innovative framework centered around *group-time average treatment effects* (ATT(g,t))—the average effect for group *g* first treated at time *g* evaluated at time *t*, allowing for full treatment effect heterogeneity across groups and over time [7, 9].

**Key Features:**

- **Explicit Modeling of Heterogeneous and Dynamic Effects:** Instead of collapsing all effects into a single estimate, ATT(g,t) estimates reveal treatment effect evolution and heterogeneity by cohort and calendar time [7].  
- **Clean Comparison Groups:** Estimation compares treated groups only to never-treated or not-yet-treated groups, circumventing forbidden comparisons and negative weighting problems [7].  
- **Flexible Parallel Trends Assumption:** Parallel trends is assumed conditional on covariates or comparing against clean controls, relaxing unconditional parallel trends and enhancing identification credibility [7, 21].  
- **Multiple Estimators:** Outcome regression, inverse probability weighting (IPW), and doubly robust (DR) estimators improve robustness to model misspecification; DR estimators combine IPW and regression adjustment for better efficiency [7, 9].  
- **Aggregation Schemes for Overall Effects:** Flexible methods aggregate heterogeneous ATT(g,t)s using weighting schemes that can summarize dynamic effects, exposure length, or calendar time effects [9].  
- **Inference via Bootstrap Methods:** Confidence bands and hypothesis tests use multiplier or cluster-robust bootstrap procedures, allowing simultaneous inference on groups and times [7, 9].  
- **Software Availability:** Implemented in R package ‘did’, facilitating application by applied researchers [9].

This framework effectively addresses heterogeneity and dynamic treatment timing concerns while preserving interpretability under realistic assumptions.

---

### Sun and Abraham’s Interaction-Weighted Estimators (Event-Study Focus)

Sun and Abraham (2020, 2021) target the problems of TWFE event-study regressions with staggered adoption and heterogeneous treatment effects:

- **Event-Time Specific Effect Estimation:** Their method estimates average dynamic effects at relative event times (time since treatment) by constructing *interaction-weighted* estimators that focus on contrasts at each event time [11, 15].  
- **Avoidance of Negative Weighting:** Like Callaway and Sant’Anna, Sun and Abraham use only never-treated or last-treated groups as controls, eliminating problematic control contamination [15].  
- **Generalized Parallel Trends:** They relax parallel trends to allow for group-specific time trends compatible with heterogeneous dynamic effects in event studies [11].  
- **Pointwise Inference:** Provides pointwise confidence intervals for treatment effects at each relative event time, facilitating interpretation of dynamic treatment paths [15].  
- **Software for Application:** Implemented in the Stata package ‘eventstudyinteract’, widely used by empirical economists [15].  
- **Complementary to Callaway and Sant’Anna:** While Callaway and Sant’Anna focus more broadly on group-time ATT heterogeneity, Sun and Abraham emphasize interaction-weighted event-study designs specifically [11].

Sun and Abraham’s approach improves estimation and inference of dynamic treatment effects in event study contexts, particularly where event timing varies.

---

### Borusyak, Jaravel, and Spiess’ Imputation-Based Methods

Borusyak et al. (2021, 2024) conceptualize DiD estimation as an *imputation problem*, emphasizing estimation of untreated potential outcomes for treated units via regression imputation to address heterogeneity and dynamics:

- **Explicit Imputation of Untreated Outcomes:** Treated potential untreated outcomes are “missing data” estimated through flexible, model-based imputation [17].  
- **Robustness to Heterogeneous and Dynamic Effects:** By modeling untreated outcomes carefully, the method reliably estimates average treatment effects without requiring treatment effect homogeneity [25].  
- **Single-step OLS Regression Estimation:** Provides computational efficiency and ease of implementation, including for repeated cross-section and panel data [25].  
- **Flexible Parallel Trends Assumption:** Assumes parallel trends hold conditional on covariates fitted with outcome regression; accommodates time-varying covariates in the imputation model [25].  
- **Software Implementations:** Available in Stata, facilitating uptake in applied research [17].  
- **Use in Complex Data Structures:** Suitable for datasets with repeated cross-sections, varying treatment doses, and complex adoption timing [25].

This imputation framework lends transparency, flexibility, and computational advantages, complementing the other two approaches.

---

## Comparison of Key Underlying Assumptions and Their Treatment

### Parallel Trends Assumption

- **Callaway and Sant’Anna:**  
  - Parallel trends assumed to hold *conditionally* on covariates or relative to clean control groups (never treated or not yet treated). This conditionality relaxes strict unconditional trends and aids identification in observational data [7, 21].  
  - Parallel trends can be tested via placebo tests event by event, but true pre-trends may be obscured in heterogeneous contexts [7].  
- **Sun and Abraham:**  
  - Assume parallel trends generalized to *group-specific* or *event-time* trends to accommodate heterogeneity in dynamics, particularly in event-study designs [11].  
  - Less explicit modeling of time-varying covariates in the parallel trends assumption compared to Callaway and Sant’Anna but recognize necessity for cautious interpretation [15].  
- **Borusyak et al.:**  
  - Parallel trends defined through the imputation model: untreated potential outcomes of treated units are modeled conditional on observed covariates.  
  - Imputation approach effectively encodes parallel trends within the regression imputation model, allowing flexible functional forms and covariates [25].  

**Summary**: All three approaches replace the strict unconditional parallel trends assumption in TWFE with more flexible, conditional, or generalized variants, improving plausibility and robustness.

---

### Treatment Effect Heterogeneity

- **TWFE**: Assumes homogeneous or stable treatment effects across groups and time; violation results in bias, non-convex weights, and contamination [22].  
- **Callaway and Sant’Anna:** Explicitly model heterogeneous treatment effects by estimating ATT(g,t) individually for each group and time, thereby capturing dynamics and variation across cohorts [7].  
- **Sun and Abraham:** Handle heterogeneity by estimating event-study coefficients at each relative event time with interaction weights to avoid bias from contaminating effects [15].  
- **Borusyak et al.:** Indirectly accommodate heterogeneity by flexibly modeling potential untreated outcomes, letting treatment effects vary without relying on restrictive assumptions [25].  

All three explicitly address and permit heterogeneous and dynamic treatment effects, thereby resolving the main criticism leveled at TWFE.

---

### Anticipation Effects

- **Callaway and Sant’Anna:**  
  - Assume *limited anticipation*, meaning no effects occur before treatment; recognize violations bias estimates, but propose partial identification and sensitivity analyses to assess robustness [24].  
- **Sun and Abraham:**  
  - Acknowledge anticipation effects as potential biases in event studies but do not explicitly model or provide formal adjustments; recommend caution and robustness checks [15].  
- **Borusyak et al.:**  
  - Maintain standard no-anticipation assumptions, not focusing directly on anticipation effects; model flexibility allows for some degree of covariate-based anticipation adjustment indirectly [25].  

Anticipation remains a more challenging and less universally addressed assumption; while Callaway and Sant’Anna provide more direct treatment, all approaches underscore the importance of testing or bounding anticipation-related violations.

---

## Empirical Adoption and Methodological Justifications in Top Journals (2020-2024)

### Adoption Rates and Methodological Dominance

- Empirical studies in **labor** and **health economics** within *American Economic Review (AER)*, *Quarterly Journal of Economics (QJE)*, and *Journal of Political Economy (JPE)* increasingly recognize the limitations of TWFE DiD under staggered adoption.  
- **Callaway and Sant’Anna’s approach** and **Sun and Abraham's interaction-weighted estimators** appear most frequently in applied research involving staggered treatment timing, providing interpretable treatment effect estimates that accommodate heterogeneity [7, 11].  
- **Borusyak et al.’s imputation-based methods** are less widely adopted but growing in visibility due to their computational advantages and flexible application, especially in studies dealing with complex survey data or repeated cross-sections [17, 25].  
- TWFE remains common, but many recent articles explicitly acknowledge its limitations and call for or implement alternative estimators in staggered adoption designs [6].  
- Software availability (R’s `did` package, Stata’s `csdid` and `eventstudyinteract`) supports adoption of these modern methods in applied research.

### Explicit Methodological Justifications

- A growing number of applied papers **explicitly justify their chosen DiD estimator**, predominately relying on one or more of the following:  
  - **Monte Carlo Simulations:** Validate estimator performance under plausible scenarios for treatment effect heterogeneity and timing [12].  
  - **Sensitivity Analyses:** Test robustness to violations of parallel trends and anticipation effects, including placebo tests and alternative control group specifications [7, 9].  
  - **Goodman-Bacon Decomposition:** Diagnose potential contaminations and negative weighting issues in TWFE [22].  
- Health economics papers, in particular, increasingly report nuanced assessments of identification assumptions and estimator robustness, using simulation evidence or sensitivity checks when employing Callaway-Sant’Anna or Sun-Abraham methods [1].  
- However, explicit reporting of such methodological diagnostics or simulations is not universal and varies by field and journal. More rigorous DiD practice is evolving.

### Methodological Dominance Verdict

- Evidence suggests **Callaway-Sant’Anna’s two-stage aggregation framework has emerged as the methodologically dominant alternative** in both labor and health economics articles within top journals over the 2020–2024 period, notably due to its flexibility, transparent assumptions, and software support [7, 9].  
- Sun and Abraham’s interaction-weighted event-study estimators are highly influential in settings emphasizing dynamic effects and are often employed complementarily [11, 15].  
- Borusyak et al.’s imputation approach is increasingly discussed and applied but has yet to reach the same widespread dominance in the applied literature [17].

---

## Response of Newer Estimators to Roth (2022)’s Critique on Pre-Trend Testing

Jonathan Roth (2022) critiques standard pre-trend tests as **insufficient for validating the parallel trends assumption**, arguing that absence of detectable pre-treatment trends does not guarantee identification since underlying dynamic heterogeneity can mask violations.

- **Callaway and Sant’Anna** acknowledge Roth’s concerns, emphasizing comprehensive diagnostic tools beyond simple pre-trend tests:  
  - Their estimation framework allows for joint inference across time periods, enabling simultaneous hypothesis tests for pre-treatment differences [7].  
  - Advocates complementing event-study style pre-trend analyses with sensitivity analyses and bounding approaches (e.g., Rambachan and Roth’s methods) to assess the robustness of identifying assumptions [24].  
- **Sun and Abraham’s event-study estimators** inherently provide pointwise confidence intervals at pre-treatment event times, allowing more transparent examination of pre-trends; nonetheless, they caution about interpreting null pre-trends too strongly without sensitivity checks [15].  
- **Borusyak et al.’s imputation approach**, by modeling untreated potential outcomes, can integrate sensitivity analyses for parallel trends violations and offer flexible checks beyond visual pre-trend inspection [25].  
- Methodological advances inspired by Roth include layered checks of assumptions, alternative identification approaches (e.g., stacking DiD, synthetic DiD), and use of machine learning for flexible trend modeling [11, 53].

**In sum**, these newer DiD estimators incorporate Roth’s critique by:

- Moving beyond simplistic pre-trend testing to more rigorous, joint inference frameworks.  
- Encouraging sensitivity analyses and bounding strategies to account for potential trend violations.  
- Promoting transparent diagnostic visualizations with simultaneous confidence bands covering pre-treatment periods.

---

## Conclusion

The Goodman-Bacon (2021) critique profoundly reshaped thinking on Difference-in-Differences in staggered adoption designs by exposing fundamental biases inherent in TWFE estimators under treatment heterogeneity and dynamic timing.

In response, Callaway and Sant’Anna’s two-stage aggregation estimator, Sun and Abraham’s interaction-weighted event-study estimator, and Borusyak et al.’s imputation-based estimator have each developed distinct yet complementary frameworks that:

- Explicitly address heterogeneous treatment effects and dynamic treatment timing.  
- Relax and generalize the parallel trends and no-anticipation assumptions, often conditioning on covariates or valid control comparisons.  
- Avoid forbidden comparisons and negative weighting problems inherent in TWFE.  
- Provide improved inferential methods supportive of simultaneous hypothesis testing and uncertainty quantification.

Empirical adoption in top economics journals, especially in labor and health economics, demonstrates a clear methodological trend favoring Callaway and Sant’Anna’s approach, supported by Sun and Abraham’s dynamic event study methods, with Borusyak et al. gaining traction due to computational and modeling advantages.

These modern estimators have also integrated insights from Roth (2022), moving DiD practice beyond reliance on simplistic pre-trend testing toward more robust assumption diagnostics and sensitivity analyses.

Applied researchers should actively employ and justify these advanced DiD estimators, leveraging available software and advanced diagnostics to ensure credible causal inference in the presence of staggered treatment adoption and treatment effect heterogeneity.

---

### Sources

[1] Designing Difference-in-Difference Studies with Staggered Treatment Adoption: https://www.annualreviews.org/content/journals/10.1146/annurev-publhealth-061022-050825  
[2] Callaway & Sant’Anna (2020) Difference-in-Differences with Multiple Time Periods: https://psantanna.com/files/Callaway_SantAnna_2020.pdf  
[3] Callaway & Sant’Anna (2021) Advances in DiD Estimators: https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf  
[4] Sun & Abraham (2021) Event-Study Difference-in-Differences: https://github.com/lsun20/eventstudyinteract  
[5] Borusyak, Jaravel, and Spiess (2021) Imputation-Based DiD Estimation: https://causalinf.substack.com/p/deja-vu-and-differential-timing  
[6] Goodman-Bacon (2021) Difference-in-Differences with Variation in Treatment Timing: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Goodman-Bacon_2021_Difference-in-differences_with_variation_in_treatment_timing.pdf  
[7] Roth (2022) Limitations of Pre-Trend Testing in DiD: https://jonathandroth.github.io/papers/  
[8] Chen, Sant’Anna, Xie (2025) Efficient DiD and Event Study Estimators: https://causal-machine-learning.github.io/kdd2025-workshop/papers/invited3_Pedro_slides.pdf  
[9] Pedro H.C. Sant’Anna (2024) Modern Difference-in-Differences overview: https://psantanna.com/DiD/NABE_202410.pdf  
[10] Rambachan and Roth (2022) Bounded Deviation Sensitivity Analyses: https://arxiv.org/abs/1909.09602  
[11] Deb, Munk & Sant’Anna (2024) Aggregating Average Treatment Effects on the Treated in DiD: https://www.nber.org/system/files/working_papers/w34331/w34331.pdf  
[12] Advances in Difference-in-Differences Methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929/  

---

This report consolidates and synthesizes core methodological and empirical insights to guide researchers navigating the complex landscape of Difference-in-Differences estimation post-Goodman-Bacon critique.