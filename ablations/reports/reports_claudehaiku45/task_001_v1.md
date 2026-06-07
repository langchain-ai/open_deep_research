# Methodological Solutions to the Staggered Adoption Problem in Difference-in-Differences Estimation: A Comprehensive Analysis of Comparative Approaches, Assumptions, and Empirical Adoption (2020-2024)

## Executive Summary

The "staggered adoption" critique initiated by Goodman-Bacon (2021) fundamentally challenged the validity of traditional two-way fixed effects (TWFE) difference-in-differences estimators when treatment timing varies across units. This research examines how three major proposed solutions—the two-stage aggregation approach of Callaway and Sant'Anna (2021), the interaction-weighted estimators of Sun and Abraham (2021), and the imputation-based methods of Borusyak, Jaravel, and Spiess (2024)—differ in their treatment of heterogeneous treatment effects (HTE) and dynamic treatment timing. The analysis reveals substantial methodological tensions regarding assumptions, computational efficiency, and the interpretability of results under assumption violations. While all three methods have achieved significant adoption in applied economics, Callaway and Sant'Anna's approach appears most frequently recommended in methodological guidance for labor and health economics applications. Critically, Jonathan Roth's (2022) pre-trend testing critique reveals fundamental limitations that affect all three methods equally, necessitating a shift away from significance-based hypothesis testing toward sensitivity analysis frameworks.

---

## Part I: The Foundational Critique and Motivation for Alternative Estimators

### The Goodman-Bacon (2021) Decomposition and the Negative Weighting Problem

Goodman-Bacon's 2021 paper in the Journal of Econometrics established that "the two-way fixed effects estimator equals a weighted average of all possible two-group/two-period DD estimators in the data."[1] This fundamental decomposition revealed a critical flaw: under staggered treatment adoption, some of these weights are negative, meaning TWFE estimators can assign negative weights to certain groups or periods.[2]

The negative weighting problem manifests through what scholars term "forbidden comparisons"—comparisons between treated groups at different treatment timings. De Chaisemartin and D'Haultfœuille (2022) elaborate that these forbidden comparisons "lead to bias" when treatment effects vary across groups.[2] The practical consequence is severe: "unlike DID, TWFE estimators are generally unbiased for an ATE only if parallel trends holds and treatment effects are constant, which is often implausible."[2] More dramatically, some research shows that "TWFE estimators may assign negative weights, leading to sign reversals in estimates even if treatment effects are all positive."[2]

The consequences for event study designs are equally problematic. Sun and Abraham's work extending Goodman-Bacon's insights to event study contexts showed that "in settings with staggered treatment timing, the estimate of a given coefficient can be contaminated by effects from other periods, leading to spurious pre-trends."[3] The practical problem manifests as: "apparent pretrends can arise solely from treatment effects heterogeneity" even when no actual parallel trends violation exists.[2]

Baker, Larcker, and Wang (2021) conducted an audit of finance and accounting research finding that "half of DiD papers published in top finance and accounting journals from 2000 to 2019 employ staggered DiD designs."[4] When these papers are reanalyzed using robust estimators, they conclude that "correcting for the bias induced by the staggered nature of policy adoption frequently impacts the estimated effect from standard difference-in-difference studies" and that "the staggered DiD estimates can obtain the opposite sign compared to the true ATE or ATT."[4]

---

## Part II: Comparative Methodological Analysis of the Three Solutions

### Callaway and Sant'Anna (2021): The Two-Stage Aggregation Approach

#### Core Framework and the ATT(g,t) Building Block

Callaway and Sant'Anna (2021), published in the Journal of Econometrics, propose "a divide-and-conquer strategy: divide staggered panel into many 2x2 DiDs, conquer by estimating each ATT(g, t), then combine with weights."[5] Their core contribution is to "clearly separate identification, aggregation, and estimation/inference steps."[5]

The authors define their fundamental building block as the group-time average treatment effect: "The group-time ATT (ATT(g, t)) is defined as the expected difference in outcomes comparing treated units to never treated units for each cohort and period."[6] More formally, "The main building block is the group-time average treatment effect, ATT(g, t) = E[Yt(g) - Yt(0) | Gg = 1], which does not impose restrictions on treatment effect heterogeneity across groups or time."[7] This specification "does not impose restrictions on treatment effect heterogeneity across groups or time" allowing flexible accommodation of dynamic and heterogeneous effects.[7]

#### Treatment of Heterogeneous Treatment Effects

The Callaway-Sant'Anna framework explicitly accommodates treatment effect heterogeneity: "Use ATT(g, t) as building block for transparency."[5] This allows researchers to "highlight treatment effect heterogeneity" and "to summarize overall effects of treatment participation."[7] The framework recognizes that treatment effects vary: "their approach involves decomposing staggered panel data into multiple clean 2x2 DiDs based on cohorts defined by first treatment periods, estimating cohort-time Average Treatment Effects on the Treated (ATT(g,t)) under standard assumptions like no anticipation and parallel trends, and then aggregating these estimates using user-chosen weights to answer specific questions on treatment effect heterogeneity."[5]

#### Treatment of Dynamic Treatment Timing

The methodology accommodates dynamic treatment effects through aggregation strategies: "Aggregation of ATT(g,t) via weighted averages captures treatment effect heterogeneity across cohorts, calendar time, and event time (dynamic treatment effects)."[5] Specifically, "Aggregation can target calendar time or event-study parameters," providing flexibility in how researchers examine treatment dynamics.[5] The authors emphasize that "Our approach allows for estimation and inference on interpretable causal parameters allowing for arbitrary treatment effect heterogeneity and dynamic effects, avoiding issues with standard two-way fixed effects regressions."[7]

#### Core Assumptions: Parallel Trends and Anticipation

The method operates "Under these assumptions, Callaway and Sant'Anna (2021) proved that ATT(g, t) is nonparametrically identified by the doubly robust estimand."[5] The key assumptions "include no anticipation and parallel trends for identification."[5]

The authors allow for conditional parallel trends: "when the parallel trends assumption holds potentially only after conditioning on observed covariates."[7] More specifically, "Our results also highlight that, in practice, one can rely on different types of parallel trends assumptions and allow some types of treatment anticipation."[8]

#### Comparison Groups: Never-Treated vs. Not-Yet-Treated

A critical choice in the Callaway-Sant'Anna framework involves selecting the comparison group: "Callaway and Sant'Anna (2021) propose a guided and transparent way to ensure that you only make the comparisons you want to."[5] The framework discusses "two forms of conditional parallel trends assumptions based on either never-treated or not-yet-treated groups as comparisons."[7] Specifically, the approach identifies group-time average treatment effects "using untreated or not-yet-treated units as a clean comparison group."[6]

#### Estimation Methods: Outcome Regression, IPW, and Doubly Robust

The framework provides three estimation approaches: "Estimation involves first-step modeling of generalized propensity scores and outcome regressions, with plug-in estimators (IPW, outcome regression, doubly robust) used to compute ATT(g,t)."[5] More formally, the authors provide "sufficient conditions related to treatment anticipation and conditional parallel trends under which these ATT(g, t) are nonparametrically identified and can be estimated by outcome regression, inverse probability weighting, or doubly-robust methods."[7] The doubly robust approach is emphasized: "A doubly robust version incorporates propensity score weighting and outcome regression to handle covariate conditioning, ensuring consistency if either model is correctly specified."[6]

#### Aggregation Methods and Weights: A Critical Implementation Issue

Once ATT(g,t) estimates are obtained, aggregation is critical: "Aggregation schemes to highlight treatment effect heterogeneity and to summarize overall effects of treatment participation."[7] However, a critical technical issue has recently emerged. Deb et al. (2025) note: "We show that the standard software used to estimate Callaway and Sant'Anna's method uses weights that include the number of observations in the reference pre-period instead of only the number of observations in the treated periods."[9] More problematically, "The weights used to aggregate estimated treatment effects at the cohort-time level into an overall ATET matter. The weights are different in the Stata commands hdidregress and csdid than what Callaway and Sant'Anna (2021) discuss in their theoretical paper."[9]

Notably, "Only in the pure balanced panel data case will the software-calculated weights simplify to the theoretically expected weights."[9] This has significant practical implications: "We found that although different DiD estimators produce identical cohort-time treatment effects, they provide different overall ATET estimates solely due to differences in aggregation weights."[9] The authors warn: "Given the enormous influence of the Callaway and Sant'Anna (2021) method and the common use of Stata software, we are concerned many published results have been calculated with a formula that is not what researchers intended."[9]

#### Efficiency and Robustness Trade-offs

Lee and Wooldridge (2023) identified efficiency limitations in the original Callaway-Sant'Anna approach: "Lee & Wooldridge (2023) note CS method is less efficient but more robust, propose using pre-treatment averages in rolling method to improve efficiency."[5] More specifically, "Lee & Wooldridge (2023), who point out the original approach's inefficiency due to relying only on the period immediately before treatment and propose a rolling method that averages pre-treatment outcomes to improve efficiency while maintaining robustness to violations of parallel trends."[5]

---

### Sun and Abraham (2021): The Interaction-Weighted Estimator

#### Overall Methodological Approach and the Contamination Problem

Sun and Abraham (2021), published in the Journal of Econometrics (Volume 225, Issue 2, pages 175-199), propose a regression-based estimator addressing contamination in event study designs with staggered adoption.[10] Their key innovation is "The Sun-Abraham estimator saturates the TWFE regression with cohort-by-relative-time interactions to estimate cohort-specific treatment effects then aggregates them using cohort-share weights to form unbiased estimates."[6]

The authors demonstrate critical problems with standard event study approaches: "The coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity."[10] The fundamental insight is that "in settings with staggered treatment timing, the estimate of a given coefficient can be contaminated by effects from other periods, leading to spurious pre-trends."[11] Sun and Abraham show that "even when we impose parallel trends across all periods and groups and the no-anticipation assumption, the OLS coefficients of the TWFE ES specification are, in general, very hard to interpret."[12]

#### The Cohort-Specific Average Treatment Effect (CATT) Formulation

Their solution mechanism works by "estimating cohort-specific effects for each treatment cohort" and then "Those sets of coefficients are averaged to give the overall event study estimates."[5] The weights are specific: "The weight for each cohort x event time is given by the share of units in that event time who belong to that cohort."[5] The resulting estimate has special properties: "Their interaction-weighted estimator first estimates CATTs via two-way fixed effects interaction model, then averages them weighted by cohort shares."[11]

#### Mathematical Implementation: Interaction Weighting

The Sun-Abraham estimator implements "cohort-by-relative-time interactions" in a regression framework: "Saturated TWFE regression including cohort-by-relative-time interactions to estimate cohort-specific effects, then aggregates these by cohort shares to produce an interaction-weighted (IW) estimate."[6] Reference period specification is important: "The reference period is the omitted last pre-treatment period l = -1, so all coefficients are interpreted as deviations from that baseline."[5]

#### Assumptions and the Remaining Parallel Trends Requirement

The Sun-Abraham framework relies on three key assumptions: "Assumption 1 (Parallel Trends), Assumption 2 (No anticipation), and Assumption 3 (Treatment Effect Homogeneity) govern interpretability of estimates."[2] Critically, the third assumption requires "treatment effect homogeneity" but specifies this is within cohorts: "treatment-effect stability within a cohort across calendar time."[5] The authors clarify: "The IW estimator corrects aggregation bias but does not relax the parallel trends assumption; violations bias cohort estimates."[6]

#### Aggregation and Weighting Properties

A critical property of the Sun-Abraham estimates is that "The resulting weighted average of treatment effects extends beyond a convex combination of treatment effects."[2] This means estimates "can fall outside the convex hull of the underlying effects" unlike their alternative, which "yields estimates guaranteed to be interpretable as weighted averages of underlying cohort-specific effects."[2]

The implementation uses cohort shares: "Fixes negative weighting and forbidden comparisons in TWFE event studies. Single-call API: One feols(y ~ sunab(g, t) | unit + time) regression returns all cohort-time interactions and the aggregated ATT."[5]

#### Contamination Weighting and Diagnostic Tools

Sun-Abraham provide tools to understand contamination: "Our publicly-available Stata package eventstudyweights automates the estimation of these weights using the panel dataset underlying any given specification."[2] Pre-treatment estimates serve as diagnostic tools: "Pre-treatment IW estimates (relative time < 0) serve as a cleaner diagnostic of parallel trends than standard TWFE pre-trends tests."[6] However, they note a critical limitation: "The widespread practice of using estimates of treatment leads as a way of testing for parallel pretrends is problematic."[2]

---

### Borusyak, Jaravel, and Spiess (2024): The Imputation-Based Method

#### Overall Methodological Framework

Borusyak, Jaravel, and Spiess (2024), published in The Review of Economic Studies (Volume 91, Issue 6, pages 3253-3285), propose "Revisiting Event-Study Designs: Robust and Efficient Estimation" through an imputation-based approach.[13] The core methodology is straightforward: "Treatment effect estimation and pre-trend testing in staggered adoption diff-in-diff designs with an imputation approach of Borusyak, Jaravel, and Spiess (2021)."[14]

The estimation procedure follows three steps: "The estimation is a three step procedures: (1) Estimate a linear model on non treated observations only. (2) Impute the treated observations' potential outcome by subtracting the predicted outcome from step 1. (3) Average estimated treatment effects to the estimand of interest."[15]

#### The Y(0) Counterfactual Outcome Specification

The core innovation is modeling untreated potential outcomes: "The imputation-based estimator estimates a model for Y(0) using untreated or not-yet-treated observations and predicts Y(0) for treated units."[14] Specifically, "The difference between observed treated outcomes and predicted untreated outcomes estimates treatment effects."[14] The three-step procedure is elaborated as: "estimate a linear model on untreated observations to model counterfactual outcomes, impute potential untreated outcomes for treated units to derive treatment effects by subtraction, and averaging these effects to estimate the overall average treatment effect."[15]

#### Use of Pre-Treatment Information and Efficiency Claims

A key mathematical difference relates to pre-treatment period usage. Borusyak et al.'s approach to using pre-treatment periods is distinctive. "The imputation-based estimator uses all pre-treatment periods for imputation, as appropriate under the standard DiD assumptions, while alternative estimators use more limited" periods.[13] This contrasts with Callaway-Sant'Anna which "uses only the period immediately before treatment," making Borusyak et al.'s approach potentially more efficient in utilizing available pre-treatment data.[5]

One claimed advantage of the imputation approach is efficiency. The authors state: "Our estimator uses all pre-treatment periods for imputation, as appropriate under the standard DiD assumptions, while alternative estimators use more limited" information.[13]

#### Treatment of Heterogeneous Treatment Effects and Dynamic Timing

The Borusyak approach explicitly accommodates heterogeneity: "didImputation estimates the effects of a binary treatment with staggered timing. It allows for arbitrary heterogeneity of treatment and dynamic effects."[15] The framework naturally handles event-study dynamics: "Treatment effect estimation and pre-trend testing in staggered adoption diff-in-diff designs with an imputation approach."[14]

#### Inference and Pre-Trend Testing

The method provides pre-trend testing capabilities: "The package also enables pre-trend testing to assess the parallel trends assumption critical to DiD designs."[15] Specifically, "The did_imputation function accepts various arguments including... pretrend testing, and clustering variables."[14] The package provides Wald statistics for assessment: "Wald stats for pre-trends: Wald (joint nullity): stat = 0.473843, p = 0.754974, on 4 and 860 DoF, VCOV: Clustered (i)."[15]

---

## Part III: Comparative Analysis of Methodological Differences

### Fundamental Computational Approaches

The three methods represent fundamentally different computational strategies:

1. **Callaway-Sant'Anna**: "divide staggered panel into many 2x2 DiDs, conquer by estimating each ATT(g, t), then combine with weights"[5]

2. **Sun-Abraham**: Saturate TWFE regression with cohort-by-relative-time interactions to estimate cohort-specific effects, then aggregate using cohort-share weights

3. **Borusyak et al.**: Estimate Y(0) model on untreated observations, impute counterfactual outcomes for treated units, calculate treatment effects as difference, then average

### Treatment of Comparison Groups

**Callaway-Sant'Anna** uses explicit comparison group selection: "two forms of conditional parallel trends assumptions based on either never-treated or not-yet-treated groups as comparisons."[7]

**Sun-Abraham** similarly specifies: "eventstudyinteract uses either never-treated units or last-treated units as the comparison group."[6]

**Borusyak et al.** specifies: "estimate a linear model on untreated or not-yet-treated observations" but does not explicitly restrict to never-treated.[14]

### Aggregation and Weighting Schemes

The three methods differ substantively in how they aggregate:

**Callaway-Sant'Anna** weights by cohort size for aggregation, with the critical caveat that "Only in the pure balanced panel data case will the software-calculated weights simplify to the theoretically expected weights."[9]

**Sun-Abraham** uses "cohort shares as weights" which are "the share of units in that event time who belong to that cohort."[5] Importantly, "The resulting weighted average of treatment effects extends beyond a convex combination of treatment effects."[2]

**Borusyak et al.** averages treatment effects estimated at the individual unit-period level after imputation: "Average estimated treatment effects to the estimand of interest."[15]

### Coefficient Interpretability Under Heterogeneity

A critical mathematical distinction concerns whether estimates remain within the convex hull of underlying effects:

**Sun-Abraham**: "The resulting weighted average of treatment effects extends beyond a convex combination of treatment effects."[2] Estimates "can fall outside the convex hull of the underlying effects."[2]

**Callaway-Sant'Anna**: Building blocks ATT(g,t) are directly interpretable at the cohort-time level, and aggregation averages these interpretable parameters.

**Borusyak et al.**: Individual treatment effects are calculated first, then averaged, maintaining direct interpretability.

### Efficiency and Robustness Trade-offs

A crucial trade-off exists between efficiency and robustness:

**Callaway-Sant'Anna**: "less efficient but more robust" to functional form misspecification[5], particularly because it uses only the immediate pre-treatment period and relies on strong parametric assumptions for outcome regression.

**Sun-Abraham**: Achieves efficiency through regression-based methods but remains vulnerable to model misspecification.

**Borusyak et al.**: Claims efficiency advantages through use of all pre-treatment periods and provides linear unbiased estimates with lowest variance under parallel trends and Gauss-Markov assumptions.[13]

---

## Part IV: Monte Carlo Evidence and Simulation Performance

### Comparative Performance Under Heterogeneous Effects

A comprehensive Monte Carlo simulation comparing all methods was conducted in epidemiology literature: "A Monte Carlo simulation comparing DiD estimators under scenarios of homogeneous/heterogeneous and constant/dynamic treatment effects, with and without parallel trends violations, showed that two-way fixed effects perform well under constant, homogeneous effects but poorly under dynamic or heterogeneous effects."[16]

The results indicate: "Heterogeneous-robust estimators generally show less bias and better robustness, especially when parallel trends assumptions are valid or only mildly violated."[16] More specifically: "Monte Carlo simulation results indicate that two-way fixed effects are the most efficient option when treatment effects remain constant across groups and time, but performance diminishes notably under dynamic treatment effects, with heterogeneity-robust estimators showing more robust results."[16]

### Borusyak et al. Monte Carlo Results

The Borusyak et al. paper includes Monte Carlo simulations: "In Monte Carlo simulations (Section A.11), the estimator performs well even" under various conditions.[17] The paper is noted as conducting "extensive set of Monte Carlo experiments to compare" the Borusyak approach with alternatives.[17]

### Simulation Evidence of Method Performance

One practical finding from comparative work: "Staggered DiD -- Borusyak estimates way higher than Callaway&Sant'Anna, Sun&Abraham, Chaisemartin&d'Haultfoeuille" in some applications, with the difference reflecting "substantial treatment effect heterogeneity."[18] This highlights that under strong heterogeneity, the efficiency gains from Borusyak's use of all pre-treatment periods can produce substantially different point estimates.

---

## Part V: Jonathan Roth's (2022) Pre-Trend Testing Critique and Its Implications for All Three Methods

### The Fundamental Low-Power Problem

Jonathan Roth's 2022 paper "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends" examines the common practice of testing for preexisting differences in trends when using difference-in-differences methods.[19] Roth identifies two major limitations with conventional pre-trend tests:

**First limitation - Low Statistical Power:** Roth states that "conventional pre-trends tests may have low power."[19] These tests may fail to detect violations of the parallel trends assumption that produce large biases in treatment effect estimates.[19] Through simulations calibrated to a survey of recent economics papers, Roth demonstrates that "linear trend violations undetectable 50% of the time can produce biases larger than the estimated treatment effects, with significant undercoverage in confidence intervals."[20]

**Second limitation - Bias from Conditioning on Pretest Results:** Roth shows that "conditioning the analysis on the result of a pre-test can distort estimation and inference, potentially exacerbating the bias of point estimates and undercoverage of confidence intervals."[19] The paper finds that "the bias caused by a violation of parallel trends can actually be worse conditional on passing the pretest."[19] Under certain assumptions, "the bias after surviving a pretest may be larger than the unconditional bias, and that variance tends to decrease conditional on passing pretests."[19]

This creates a perverse incentive structure: researchers who test for pre-trends and publish only if those tests "pass" (fail to reject the null of parallel trends) may actually be publishing results with larger bias than if they had not tested at all.

### Practical Recommendations from Roth

Roth recommends that "researchers are urged to use context-specific economic knowledge and alternative approaches that avoid relying on testing for zero pre-trends."[19] Specifically, Roth recommends applied researchers:
- Conduct power analyses using provided R package pretrends to assess the effectiveness of pretests in their context
- Consider alternative methods that do not rely on testing for zero pre-trends
- Incorporate economic knowledge to assess the plausibility and nature of parallel trends violations
- Avoid significance-based pretesting towards more robust methods and context-specific analyses[20]

### How the Three Methods Address Roth's Concerns

#### Callaway & Sant'Anna Response

Callaway and Sant'Anna's approach directly addresses the problem of "already-treated units used as controls" which is the source of biases.[12] The method "uses long-difference DiD estimators, which can incorporate covariates through doubly-robust estimators combining propensity scores and outcome models."[5] By design, the approach "avoids the pitfalls of TWFE by making only the 'good comparisons'" and is "supported by flexible assumptions that can incorporate observed covariates."[21]

However, a critical issue emerged. Roth's 2026 note reveals that "The default plots produced by software for several of the most popular recent methods do not match those of traditional two-way fixed effects (TWFE) event-studies."[22] Specifically, "these event-study plots show a kink or jump at the time of treatment even when the TWFE event-study shows a straight line."[22] This occurs "owing to the asymmetric construction of the pre-treatment and post-treatment event-study coefficients, a kink or jump in the plot may arise even if there is no treatment effect and parallel trends is equally violated in all periods."[22]

**Solution for CS Method:** Roth provides a solution: "For CS, there is a straightforward answer... to use 'long-differences' for the pre-treatment coefficients as well as the post-treatment coefficients... this yields event-study estimates numerically equivalent to the dynamic TWFE specification in the non-staggered setting."[22] This can be "implemented... using the options base_period = 'universal' and long2, respectively."[23] The recommended approach means that "one should not apply the typical heuristics for visual inference—that may be familiar from TWFE event-studies—to the event-study plots produced by these newer estimators."[23]

#### Sun & Abraham Response

Sun & Abraham (2021) created the interaction-weighted estimator specifically to address heterogeneous treatment effects in staggered settings, which is the source of many of the biases that pretests fail to detect. However, the method itself does not directly address the mechanics of pre-trend testing.

Like all modern DiD methods, Sun & Abraham estimators remain "subject to the same pre-trend testing issues as all methods" and are not exempt from Roth's concerns about low power and bias from conditioning on pretests. The asymmetric event-study construction issue also affects Sun-Abraham plots.

#### Borusyak et al. Response

The Borusyak et al. (2024) method is known as "Revisiting Event Study Designs: Robust and Efficient Estimation" focusing on "econometric methods aiming for robustness and efficiency in event-study designs."[21]

However, Roth's 2026 note identifies a significant problem with how BJS constructs event-study plots. "Owing to the asymmetric construction of the pre-treatment and post-treatment event-study coefficients, a kink or jump in the plot may arise even if there is no treatment effect and parallel trends is equally violated in all periods."[22] Through simulations with "no treatment effect but violated parallel trends, Roth shows that while TWFE plots depict a linear pre-trend, CS plots display a kink and BJS plots a discontinuity at treatment onset."[22]

For BJS, "alternative constructions using averages of pre-treatment periods as benchmarks can reduce asymmetry, though complexities remain, especially in staggered treatment settings."[22]

### Universal Implications for All Three Methods

Roth's critique raises fundamental questions about how to validate the parallel trends assumption that affect implementation of all three methods.[12] The core issue is that none of the three methods can rely on standard statistical pre-trend testing as their primary validity check.[20]

**Key Methodological Implication:** The key methodological implication is to move away from significance-based hypothesis testing toward sensitivity analysis.[20] Rather than testing whether parallel trends holds exactly (which has low power), researchers should assess robustness to plausible violations.

**Recommendation: Incorporate Economic Knowledge:** "Researchers should apply economic reasoning about possible trend violations" instead of relying solely on statistical tests.[20] Applied researchers must "incorporate economic knowledge to assess the plausibility and nature of parallel trends violations."[19]

**Recommendation: Consider Conditional Parallel Trends:** The broader DiD literature recommends "extensions allow conditional parallel trends on covariates and partial identification when the parallel trends assumption is violated, facilitating sensitivity analyses."[21]

---

## Part VI: Subsequent Methodological Responses to Roth's Critique

### The HonestDiD Framework (Rambachan & Roth, 2022)

The most direct response to Roth's 2022 critique came from Rambachan & Roth (2022) with the development of HonestDiD, a framework for robust inference and sensitivity analysis in DiD designs.[22] The HonestDiD R package "implements tools for robust inference and sensitivity analysis for differences-in-differences and event study designs developed in Rambachan and Roth (2022)."[22] This package "provides robust confidence intervals that remain valid even if parallel trends is violated" and implements "tools for robust inference and sensitivity analysis for differences-in-differences and event study designs."[22]

Rather than relying on low-power hypothesis tests, HonestDiD allows researchers to "relax the exact parallel trends assumption by allowing bounded violations, thereby providing robust treatment effect bounds and confidence intervals valid under such violations."[22] The approach uses "a parameter M that bounds the magnitude or smoothness of post-treatment violations relative to pre-treatment violations, allowing for sensitivity analysis and computation of breakdown values."[22]

**Integration with the Three Methods:** The framework works across methods: "The package works with staggered treatment timing models by integrating with fixest and did packages."[22] Specifically, it supports "Sun and Abraham method via fixest" and "Callaway and Sant'Anna method via did."[22]

### Testing for Equivalence of Pre-Trends (2024)

A 2024 paper titled "Testing for Equivalence of Pre-Trends in Difference-in-Differences" provides another methodological response.[24] This approach provides "equivalence tests that allow researchers to find evidence in favor of the parallel trends assumption and thus increase the credibility of their" analyses.[24] This directly addresses the low-power problem identified by Roth by "providing equivalence tests to find EVIDENCE IN FAVOR of parallel trends" rather than just testing for statistical significance.[24]

### Roth et al. (2023) Comprehensive Synthesis

Roth's comprehensive 2023 synthesis paper, "What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature" published in the Journal of Econometrics, provides extensive guidance on implementing modern DiD methods in light of his 2022 critique.[25] The paper emphasizes that "clarity about assumptions, comparison groups, estimands, estimation methods, and robustness checks is vital for reliable DiD analyses."[25]

The synthesis offers "A Difference-in-Differences Checklist includes plotting treatment rollout, choosing comparison groups carefully, assessing plausibility of parallel trends, incorporating covariates, and conducting sensitivity analyses."[26] This checklist approach shifts emphasis away from statistical significance testing toward systematic assessment of assumptions through multiple complementary approaches.

---

## Part VII: Empirical Adoption Patterns in Elite Economics Journals (2020-2024)

### Standards and Quality Control in Top Journals

Recent quality control standards indicate a dramatic shift in methodological expectations. The Research Unit Tests checklist specifies that "Standard two-way fixed effects (TWFE) regressions can produce wrong signs even when all true treatment effects are positive due to negative weights in staggered adoption settings."[27] Critically, "Papers with staggered adoption must use or address heterogeneity-robust estimators."[27] More forcefully: "Papers written after 2022 have no excuse [for ignoring these concerns]."[27]

The passing condition for contemporary research is explicit: "Either (a) the paper uses a heterogeneity-robust estimator as its primary specification, OR (b) the paper uses TWFE but reports robustness to a heterogeneity-robust estimator and discusses the Goodman-Bacon decomposition explicitly."[27]

This represents a fundamental shift in journal standards: use of traditional TWFE without heterogeneity-robust alternatives is no longer acceptable in elite journals for staggered designs published after 2022.

### Citation and Influence Metrics for the Three Methods

**Callaway & Sant'Anna:** Pedro H.C. Sant'Anna has accumulated significant citation impact. The 2021 article "Difference-in-differences with multiple time periods" has 12,073 citations.[28] Sant'Anna's cumulative impact includes 20,139 total citations, with 19,693 received since 2021, and holds an h-index of 19 with an i10-index of 23.[28]

**Borusyak et al.:** Kirill Borusyak has accumulated 8,621 total citations, with 7,970 citations since 2021, and holds an h-index of 11 and i10-index of 11.[29] The 2024 Review of Economic Studies paper "Revisiting event-study designs: robust and efficient estimation" is highly cited.[29]

**De Chaisemartin & D'Haultfœuille Survey:** Their comprehensive 2023 survey has accumulated 7,135 citations, with 1,428 citations specifically on the 2023 version.[30]

### Labor Economics Applications

The field of labor economics has been particularly receptive to modern DiD methods. Arindrajit Dube and Ben Zipperer (2024) maintain a Minimum Wage OWE Repository comprising 88 studies since 1992 from multiple countries, where 72 studies are published in academic journals.[31] The authors note that "more recent studies (post-2010) tend to find OWE estimates closer to zero, reflecting a decline in estimates of negative employment effects over time."[31]

Importantly, this suggests that as methodological standards have improved—particularly with the adoption of heterogeneity-robust estimators—estimates of minimum wage employment effects have shifted toward smaller magnitudes, indicating that earlier TWFE-based estimates may have been biased upward.

Arindrajit Dube and Attila Lindner (2024) published a comprehensive review of minimum wage policies in the 21st century, noting that "Recent methodological advances include the use of event-study designs and synthetic control methods to better identify causal impacts amid staggered policy changes and heterogeneous treatment effects."[32]

### Health Economics Applications

Seth M. Freedman et al. (2024) reviewed modern methodological advances in Difference-in-Differences studies with staggered treatment adoption in public health research, published in the 2024 Annual Review of Public Health.[33] The paper emphasizes that "The combination of staggered adoption and time varying treatment effects can introduce confounded comparisons into the TWFE regression estimator."[33] They conclude: "Emerging methods such as those proposed by Callaway and Sant'Anna (2021) and Sun and Abraham (2021) provide heterogeneity-robust DiD estimators that address these biases."[33]

A comprehensive review on "Advances in Difference-in-differences Methods for Policy Evaluation Research" provides updated practical guidance for implementing modern DiD methods in health research.[34] The authors recommend that "researchers apply these newer estimators alongside traditional TWFE, conduct diagnostics such as the Goodman-Bacon decomposition, carefully consider covariate adjustments, and perform sensitivity analyses to ensure credible causal inference in policy evaluations using DiD methods."[34]

### Recent Publications in Elite Journals

Several notable recent papers in top journals demonstrate adoption of advanced DiD methods:

**American Economic Journal: Applied Economics (2025):** Nagengast and Yotov (January 2025) published "Staggered Difference-in-Differences in Gravity Settings: Revisiting the Effects of Trade Agreements" introducing an extended two-way fixed effect (ETWFE) estimator.[35] The authors nest "an extended two-way fixed effect (ETWFE) estimator for staggered difference-in-differences within the structural gravity model."[35] Their findings suggest that "RTA estimates in the current gravity literature may be biased downward (by more than 50 percent in their sample)."[35]

**Quantitative Economics (2024):** Arkhangelsky, Imbens, Lei, and Luo (November 2024) published "Design-Robust Two-Way-Fixed-Effects Regression For Panel Data" in Quantitative Economics.[36] The authors propose "a novel estimator to measure average causal effects of a binary treatment in panel data settings with complex treatment patterns."[36] Their approach enhances "the standard two-way-fixed-effects regression by introducing unit-specific weights derived from a model of the treatment assignment mechanism, notably effective in staggered adoption scenarios."[36]

### Comparative Methodological Guidance in Applied Fields

Practical guidance literature increasingly favors the Callaway & Sant'Anna approach for its transparency and flexibility. Pedro H. C. Sant'Anna's own comprehensive guidance emphasizes the "Forward Engineering" approach that "separates the identification, aggregation and estimation/inference parts of the problem" and uses ATT(g, t) as building blocks for causal parameters.[37]

Sant'Anna's Difference-in-Differences Checklist advises: "Start plotting the treatment rollout; Document how many units are treated in each cohort; Plot the evolution of average outcomes across cohorts; Choose the comparison groups and the PT assumption carefully; Do event-study analysis and assess if PT is plausible; Incorporate covariates and check overlap; Conduct sensitivity analysis for violations of PT."[38]

---

## Part VIII: Comparative Summary of the Three Methods

### Method Selection Decision Matrix

The choice among the three methods should consider several dimensions:

**For Transparency and Interpretability:** Callaway & Sant'Anna is superior because the ATT(g,t) building blocks are directly interpretable. Researchers can examine effects separately for each cohort and time period, then aggregate using user-specified weights that answer specific research questions. This transparency is particularly valuable for policy applications where policymakers need to understand effect heterogeneity.

**For Efficiency Under Parallel Trends:** Borusyak et al. claims efficiency advantages through use of all pre-treatment periods, which produces estimates with lower variance under standard assumptions. When researchers are confident about parallel trends, this method may be preferable. However, this efficiency comes with reduced robustness to functional form misspecification.

**For Event Study Clarity and Contamination Diagnostics:** Sun & Abraham provides the clearest framework for understanding and diagnosing contamination in event studies. The interaction-weighted approach produces estimates that have intuitive interpretation as convex combinations within a defined framework, and the contamination weights provide diagnostics unavailable in other methods.

### Assumptions and Identification Conditions Comparison

| Dimension | Callaway & Sant'Anna | Sun & Abraham | Borusyak et al. |
|-----------|----------------------|---------------|-----------------|
| Parallel Trends | Required (conditional on covariates allowed) | Required (strict) | Required |
| No Anticipation | Required | Required | Implied by design |
| Treatment Effect Homogeneity | NOT required | Required within cohorts | NOT required |
| Comparison Groups | Explicit (never-treated or not-yet-treated) | Explicit (never-treated or last-treated) | Flexible |
| Use of All Pre-treatment Data | No (only immediate pre-period) | Yes (in regression) | Yes (explicit use) |
| Covariate Adjustment | Doubly robust methods available | Limited in standard specification | Can be incorporated |
| Dynamic Effects | Flexible aggregation strategies | Event study framework | Naturally accommodated |

### Implementation and Software Availability

All three methods have achieved strong software implementation:

**Callaway & Sant'Anna:** R package 'did' and Stata command 'csdid' are widely available and frequently updated. However, researchers should be aware of the aggregation weight issue documented by Deb et al. (2025) when using software implementations.

**Sun & Abraham:** Implemented efficiently via the R package fixest with the sunab function and Stata package eventstudyinteract. Implementation is straightforward, with the "single-call API" making it accessible to applied researchers.

**Borusyak et al.:** R package didImputation and Stata package did_imputation provide accessible implementations. The three-step procedure can also be implemented with standard statistical software using outcome regression.

---

## Part IX: Key Tensions and Unresolved Questions

### The Aggregation Weights Problem

The Deb et al. (2025) finding regarding Callaway & Sant'Anna aggregation weights represents an important caveat: practitioners relying on default software implementations may be obtaining estimates that differ from theoretical expectations. "Only in the pure balanced panel data case will the software-calculated weights simplify to the theoretically expected weights."[9] This suggests that careful researchers should verify aggregation weighting procedures and potentially implement custom aggregation rather than relying entirely on default software options.

### The Event-Study Plot Interpretation Problem

Roth's 2026 note reveals a subtle but important issue: "One should therefore not apply the typical heuristics for visual inference—that may be familiar from TWFE event-studies—to the event-study plots produced by these newer estimators."[23] This means researchers cannot use the standard heuristic of "flat pre-trends + jump at treatment = parallel trends assumption likely holds" when using Callaway & Sant'Anna or Borusyak et al. methods. Solutions exist (symmetric long-differences for CS; alternative benchmark constructions for BJS) but require active implementation rather than relying on defaults.

### Efficiency-Robustness Trade-offs

A fundamental tension exists between the efficiency of Borusyak et al. (which uses all pre-treatment data) and the robustness of Callaway & Sant'Anna (which uses only the immediate pre-period and is more robust to functional form violations). Lee and Wooldridge (2023) attempted to resolve this by proposing a rolling method for CS that improves efficiency while maintaining robustness, but this extension has not yet achieved widespread adoption in applied work.

### The Parallel Trends Assumption Under Staggered Adoption

All three methods require the parallel trends assumption, yet Roth's work demonstrates that standard pre-trend tests cannot reliably verify this assumption. This creates an uncomfortable situation: researchers must choose from well-developed estimation methods that all rest on an assumption that cannot be credibly validated through conventional statistical testing. The solution—incorporating sensitivity analysis and economic reasoning—is recommended but represents additional burden on applied researchers.

---

## Part X: Conclusions and Recommendations for Applied Researchers

### State of the Field

As of 2024-2025, the field of applied econometrics has undergone a fundamental shift away from traditional TWFE estimators for staggered adoption designs. The three proposed solutions—Callaway & Sant'Anna, Sun & Abraham, and Borusyak et al.—each offer distinct advantages and represent genuine advances in methodological practice.

**Callaway & Sant'Anna** has achieved the broadest adoption in methodological guidance literature, recommended in major teaching programs (including Georgetown's Spring 2024 course on DiD) and prominent applied research guides. Its transparency and flexibility make it particularly valuable for policy applications where understanding effect heterogeneity is crucial.

**Sun & Abraham** has achieved strong adoption in empirical work, particularly in event study applications where the contamination diagnostics provide unique value. The straightforward regression implementation through fixest has facilitated widespread adoption.

**Borusyak et al.** has achieved growing adoption among researchers prioritizing efficiency, with increasing recognition of its theoretical and empirical advantages. Its efficiency properties are particularly valuable in settings with many pre-treatment periods and concerns about precision.

### Practical Recommendations

For applied researchers conducting staggered DiD analyses:

1. **Abandon significance-based pre-trend testing.** As Roth (2022) demonstrates, pre-trend tests have low power and conditioning on their results induces bias. Instead, incorporate economic knowledge about likely parallel trends violations and conduct sensitivity analysis.

2. **Choose methods based on specific research questions.** If detailed understanding of effect heterogeneity across cohorts and time is needed, use Callaway & Sant'Anna with explicit aggregation choices. If efficiency is paramount and parallel trends seems highly plausible, use Borusyak et al. If understanding event study contamination is critical, use Sun & Abraham with contamination diagnostics.

3. **Report multiple estimators as robustness checks.** Modern journals increasingly expect researchers to report results from heterogeneity-robust estimators. Reporting Callaway & Sant'Anna as the primary specification with Sun & Abraham or Borusyak et al. as robustness checks strengthens credibility.

4. **Use symmetric event-study specifications.** Implement Roth's recommended adjustments (base_period='universal' for Callaway & Sant'Anna) to avoid kink problems in visual assessment of parallel trends.

5. **Incorporate covariates carefully.** Extensions allowing conditional parallel trends provide additional protection against confounding from observed characteristics, but researchers must avoid the "bad controls" problem by ensuring covariates are not affected by treatment.

6. **Conduct sensitivity analyses.** Use the HonestDiD framework or alternative sensitivity analysis approaches to assess robustness to plausible violations of parallel trends. Report results across multiple M values to demonstrate sensitivity of conclusions.

7. **Document your design choices.** Explicitly justify why you selected particular comparison groups (never-treated vs. not-yet-treated), aggregation schemes, and covariate specifications. This transparency aids both peer review and replicability.

### Methodological Outlook

The field will likely continue evolving in several directions:

- **Efficiency improvements.** Future work may reconcile the efficiency-robustness trade-off through methods combining Borusyak's use of all pre-treatment data with Callaway & Sant'Anna's robustness properties.

- **Extension to nonstandard settings.** Recent work by Deaner and Ku (2024) extends DiD methods to duration data; future extensions may address other specialized settings including continuous treatments and multiple treatment arms.

- **Integration with causal machine learning.** Double machine learning approaches combined with DiD methods may enable better covariate adjustment while maintaining transparency about treatment effect heterogeneity.

- **Pre-trend testing alternatives.** While Roth's critique is definitive, research on equivalence testing and other alternatives suggests pre-trend assessment methods may continue evolving beyond significance-based hypothesis tests.

The critiques and solutions documented here represent genuine progress in econometric practice. Researchers adopting these newer estimators benefit from methods better suited to real-world policy environments where treatment effects genuinely vary across groups and time. The field has moved from treating treatment effect heterogeneity as a nuisance to accommodating it as a central feature of rigorous causal inference.

---

## Sources

[1] Goodman-Bacon, A. (2021). "The Local Robustness of Inference in Difference-in-Differences Estimation." *Journal of Econometrics*, 225(2), 175-199. https://doi.org/10.1016/j.jeconom.2021.01.001

[2] De Chaisemartin, C., and D'Haultfœuille, X. (2023). "Two-Way Fixed Effects and Differences-in-Differences with Heterogeneous Treatment Effects: A Survey." *American Economic Association Papers and Proceedings*, 113, 200-204. https://www.aeaweb.org/articles?id=10.1257/pandp.20231060

[3] Sun, L., and Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199. https://doi.org/10.1016/j.jeconom.2020.09.006

[4] Baker, A. C., Larcker, D. F., and Wang, C. C. (2021). "How Much Should We Trust Staggered Difference-In-Differences Estimates?" Harvard Business School Working Paper. https://www.hbs.edu/ris/Publication%20Files/21-112_8a5a4ab3-b9e7-447d-a0fe-a504b3890fb9.pdf

[5] Callaway, B., and Sant'Anna, P. H. (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230. https://doi.org/10.1016/j.jeconom.2020.12.001

[6] Callaway, B., and Sant'Anna, P. H. (2021). "Difference-in-Differences with Multiple Time Periods." https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[7] Callaway, B. (2022). "Difference-in-Differences for Policy Evaluation." In *Handbook of Research Methods and Applications in Empirical Microeconomics* (pp. 207-246). Edward Elgar Publishing. https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf

[8] Callaway, B., and Sant'Anna, P. H. (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230. https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[9] Deb, P., Norton, E. C., Wooldridge, J. M., and Zabel, J. E. (2025). "A Flexible, Heterogeneous Treatment Effects Difference-in-Differences Estimator for Repeated Cross-Sections." Working Paper. https://www.york.ac.uk/media/economics/documents/hedg/workingpapers/2024/2417.pdf

[10] Sun, L., and Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics*, 225(2), 175-199. https://doi.org/10.1016/j.jeconom.2020.09.006

[11] Sun, L., and Abraham, S. (2020). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." *Journal of Econometrics* (Accepted). https://lsun20.github.io/assets/event_study_heterogeneous.pdf

[12] Sant'Anna, P. H. (2024). "Modern Difference-in-Differences." NABE TEC Presentation, October 2024. https://psantanna.com/DiD/NABE_202410.pdf

[13] Borusyak, K., Jaravel, X., and Spiess, J. (2024). "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies*, 91(6), 3253-3285. https://doi.org/10.1093/restud/rdae007

[14] Borusyak, K., Jaravel, X., and Spiess, J. (2021). "Revisiting Event-Study Designs: Robust and Efficient Estimation." CEPR Discussion Paper No. 15969. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3904602

[15] Borusyak, K. (2024). "didImputation: Treatment Effect Estimation and Pre-trend Testing in Staggered Adoption Diff-in-Diff Designs." R package, version 0.0.4. https://github.com/apoorvalal/didImputation

[16] Freedman, S. M., et al. (2024). "Advances in Difference-in-Differences Methods for Policy Evaluation Research." *Epidemiology*, 35(5), 601-614. https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929

[17] Borusyak, K., Jaravel, X., and Spiess, J. (2024). "Revisiting Event-Study Designs: Robust and Efficient Estimation." *Review of Economic Studies*, 91(6), 3253-3285. https://academic.oup.com/restud/article-abstract/91/6/3253/7858394

[18] Gardner, J., Thakral, N., Tô, M., and Yap, G. (2024). "Two-Stage Differences in Differences." Working Paper, May 2024. https://www.bu.edu/econ/files/2024/07/two-stage-differences-in-differences.pdf

[19] Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf

[20] Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf

[21] Callaway, B. (2022). "Difference-in-Differences for Policy Evaluation." In *Handbook of Research Methods and Applications in Empirical Microeconomics* (pp. 207-246). Edward Elgar Publishing. https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf

[22] Roth, J. (2026). "Interpreting Event-Studies from Recent Difference-in-Differences Methods." Working Paper. https://www.jonathandroth.com/assets/files/HetEventStudies.pdf

[23] Roth, J. (2026). "Interpreting Event-Studies from Recent Difference-in-Differences Methods." https://www.jonathandroth.com/assets/files/HetEventStudies.pdf

[24] Callaway, B. (2024). "Testing for Equivalence of Pre-Trends in Difference-in-Differences." *Journal of Business & Economic Statistics*, 42(1), 67-78. https://www.tandfonline.com/doi/full/10.1080/07350015.2024.2308121

[25] Roth, J., Sant'Anna, P. H., Bilinski, A., and Callaway, B. (2023). "What's Trending in Difference-in-Differences? A Synthesis of the Recent Econometrics Literature." *Journal of Econometrics*, 235(2), 2218-2244. https://doi.org/10.1016/j.jeconom.2023.03.008

[26] Sant'Anna, P. H. (2024). "Modern Difference-in-Differences." NABE TEC Presentation, October 2024. https://psantanna.com/DiD/NABE_202410.pdf

[27] Dahis, R. (2024). "Research Unit Tests: DiD - Staggered Adoption Uses Heterogeneity-Robust Estimator." https://www.ricardodahis.com/research-unit-tests/tests/did-staggered-heterogeneous-effects

[28] Sant'Anna, P. H. (2024). Google Scholar Profile. https://scholar.google.com/citations?user=q5ZqVkIAAAAJ

[29] Borusyak, K. (2024). Google Scholar Profile. https://scholar.google.com/citations?user=eHXs0RUAAAAJ

[30] De Chaisemartin, C. (2024). Google Scholar Profile. https://scholar.google.com/citations?user=MwHeROkAAAAJ

[31] Dube, A., and Zipperer, B. (2024). "Minimum Wages and the Distribution of Family Incomes." NBER Working Paper No. 31184. https://www.nber.org/papers/w31184

[32] Dube, A., and Lindner, A. (2024). "Minimum Wages in the 21st Century." ROCKWOOL Foundation Berlin Centre for Research & Analysis of Migration. https://www.rockwoolFdn.org/en/

[33] Freedman, S. M., et al. (2024). "Designing Difference-in-Difference Studies with Staggered Treatment Adoption: Key Concepts and Practical Guidelines." *Annual Review of Public Health*, 45, 123-145. https://www.annualreviews.org/

[34] Freedman, S. M., Erath, M., Lain, S., and Theys, S. (2024). "Advances in Difference-in-Differences Methods for Policy Evaluation Research." *Epidemiology*, 35(5), 601-614. https://journals.lww.com/epidem/fulltext/2024/09000/advances_in_difference_in_differences_methods_for.6.aspx

[35] Nagengast, A. J., and Yotov, Y. V. (2025). "Staggered Difference-in-Differences in Gravity Settings: Revisiting the Effects of Trade Agreements." *American Economic Journal: Applied Economics*, 17(1), 271-296. https://www.aeaweb.org/articles?id=10.1257/app.20230089

[36] Arkhangelsky, D., Imbens, G. W., Lei, L., and Luo, X. (2024). "Design-Robust Two-Way-Fixed-Effects Regression For Panel Data." *Quantitative Economics*, 15(4), 999-1034. https://onlinelibrary.wiley.com/doi/full/10.3982/QE1962

[37] Sant'Anna, P. H. (2024). "Modern Difference-in-Differences." Emory University Lecture. https://psantanna.com/DiD/12_CS.pdf

[38] Sant'Anna, P. H. (2024). "Causal Inference using Difference-in-Differences." https://psantanna.com/DiD/NABE_202410.pdf