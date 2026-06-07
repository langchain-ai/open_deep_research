# Methodological Tensions in Staggered Difference-in-Differences Estimation: A Comprehensive Analysis of Modern Solutions

## 1. Introduction

The past five years have witnessed a fundamental rethinking of difference-in-differences (DiD) methodology, driven by the recognition that traditional two-way fixed effects (TWFE) estimators can produce severely misleading results when treatment adoption is staggered and treatment effects are heterogeneous. The critique, crystallized in Goodman-Bacon's (2021) decomposition theorem, revealed that the TWFE estimator is a weighted average of many 2×2 DiD comparisons, some of which use already-treated units as controls—creating "forbidden comparisons" that can generate negative weights and bias estimates of any sign.

This crisis has spawned a rich ecosystem of alternative estimators. This report provides a comprehensive analysis of four leading approaches: **Callaway and Sant'Anna (2021)**, **Sun and Abraham (2021)**, **Borusyak, Jaravel, and Spiess (2024)**, and the diagnostic framework of **Goodman-Bacon (2021)**. We examine how each handles heterogeneous treatment effects and dynamic treatment timing, compare their underlying assumptions, evaluate their adoption rates in top economics journals, and analyze how they address Roth's (2022) concerns about pre-trend testing.

---

## 2. How the Proposed Solutions Handle Heterogeneous Treatment Effects and Dynamic Treatment Timing

### 2.1 Goodman-Bacon (2021): The Decomposition Theorem as Diagnostic

**Publication:** Andrew Goodman-Bacon, "Difference-in-differences with variation in treatment timing," *Journal of Econometrics*, Volume 225, Issue 2, December 2021, Pages 254-277. [Working paper: NBER Working Paper No. 25018, September 2018] [1][2][3]

Goodman-Bacon (2021) does not propose a new estimator but rather provides a diagnostic framework for understanding the TWFE estimator's components. The paper proves that the TWFE DiD estimator equals a weighted average of all possible two-group, two-period (2×2) DD estimators in the data, with weights depending on group sizes and the variance of treatment timing.

**Key insight on heterogeneity:** The paper shows that "When treatment effects do not change over time, TWFEDD yields a variance-weighted average of cross-group treatment effects and all weights are positive." Critically, "Negative weights only arise when average treatment effects vary over time." This means that even under valid parallel trends, TWFE can be misleading when effects evolve dynamically.

The decomposition identifies four types of 2×2 comparisons:
- **Comparisons between treated and never-treated units** (unproblematic)
- **Early-treated vs. late-treated (before late units are treated)** — uses not-yet-treated as controls (unproblematic)
- **Late-treated vs. early-treated (after late units are treated)** — uses already-treated as controls (problematic when effects vary over time)
- **Different timing groups where both serve as treatment and control at different points**

**How it handles timing:** The paper shows that units treated near the middle of the panel tend to have higher influence as treated groups, while those at the edges often act more as controls. The method enables researchers to decompose estimates into their 2×2 components and assess which comparisons drive results.

**Software:** Stata/R package `bacondecomp` [4]

### 2.2 Callaway and Sant'Anna (2021): Two-Stage Aggregation with Doubly-Robust Estimation

**Publication:** Brantly Callaway and Pedro H.C. Sant'Anna, "Difference-in-Differences with multiple time periods," *Journal of Econometrics*, Volume 225, Issue 2, December 2021, Pages 200-230. [Working paper: arXiv:1803.09015, March 2018] [5][6][7]

Callaway and Sant'Anna (2021) propose a "divide-and-conquer" strategy centered on **group-time average treatment effects on the treated**, denoted ATT(g,t)—the average treatment effect for units first treated in period g at time t. This building block parameter is fully flexible, allowing unrestricted heterogeneity across cohorts and over time.

**How heterogeneity is handled:** The approach separately estimates ATT(g,t) for each combination of cohort g and time period t, imposing no restrictions on how effects vary across groups or time. These can then be aggregated in multiple ways:
- **Event-study aggregation:** Average effects by length of exposure to treatment
- **Calendar time aggregation:** Effects across groups at each time period
- **Group-specific effects:** Average effect for each cohort
- **Overall average treatment effect:** Weighted average across all groups and post-treatment periods

**How timing is handled:** Units are grouped into cohorts by their first treatment period. For each cohort, ATT(g,t) is estimated separately, using either never-treated or not-yet-treated units as the comparison group. This avoids the "forbidden comparisons" that plague TWFE because already-treated units are never used as controls.

**Doubly-robust estimation:** The method offers three identification approaches—outcome regression (OR), inverse probability weighting (IPW), and doubly-robust (DR) methods that combine both. The DR estimator is consistent if either the outcome regression or propensity score model is correctly specified, providing robustness against model misspecification.

**Software:** R package `did` (available on CRAN and GitHub at bcallaway11.github.io/did); Stata packages `csdid` and `xthdidregress` (Stata 18) [5][6][8]

### 2.3 Sun and Abraham (2021): Interaction-Weighted Estimator

**Publication:** Liyang Sun and Sarah Abraham, "Estimating dynamic treatment effects in event studies with heterogeneous treatment effects," *Journal of Econometrics*, Volume 225, Issue 2, December 2021, Pages 175-199. [Working paper: arXiv:1804.05785, April 2018] [9][10][11]

Sun and Abraham (2021) propose the **interaction-weighted (IW) estimator**, which corrects the contamination problems in TWFE event-study regressions. The paper proves that "the coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity."

**How heterogeneity is handled:** The estimator works in two stages:
1. **First stage:** For each cohort e, estimate cohort-specific dynamic treatment effects (CATT(e,l)) by running regressions that interact cohort indicators with event time indicators, using only never-treated or not-yet-treated units as the comparison group.
2. **Second stage:** Aggregate the cohort-specific estimates into an average dynamic treatment effect at each event time l using sample-size weights: δ̂_l = Σ_e [Pr(E=e | E∈[l_min,l_max]) · CATT̂(e,l)]

**How timing is handled:** The estimator computes separate CATT(e,l) for each cohort at each event time, then aggregates with interpretable cohort-share weights. This ensures that the event-study coefficients are "free of contamination" and represent weighted averages of the underlying effects.

**Comparison group:** The estimator uses either never-treated units or last-treated units as the comparison group. It does NOT use already-treated units as comparisons, avoiding the negative weighting bias.

**Software:** Stata package `eventstudyinteract` (available via GitHub); R implementation via `fixest::feols` with the `sunab` interaction term [9][12][13]

### 2.4 Borusyak, Jaravel, and Spiess (2024): Imputation-Based Estimation

**Publication:** Kirill Borusyak, Xavier Jaravel, and Jann Spiess, "Revisiting Event-Study Designs: Robust and Efficient Estimation," *The Review of Economic Studies*, Volume 91, Issue 6, 2024, Pages 3253-3285. [Working paper: arXiv:2108.12419, August 2021] [14][15][16]

Borusyak et al. (2024) develop an **imputation-based estimator** that separates identification assumptions from the estimation target. The approach is intuitive: estimate a model for untreated potential outcomes Y(0) using only untreated observations, then impute counterfactual outcomes for treated observations.

**How heterogeneity is handled:** The method proceeds in three stages:
1. **Model estimation:** Estimate a model for Y(0) using only never-treated and not-yet-treated observations (typically unit and time fixed effects, possibly with covariates)
2. **Imputation:** For treated observations, impute the counterfactual Ŷ_it(0) using the estimated model parameters
3. **Treatment effect estimation:** Calculate τ̂_it = Y_it − Ŷ_it(0) for each treated observation, then average these individual effects with researcher-chosen weights

The method "allows for unrestricted treatment-effect heterogeneity" and "consistently estimates pre-specified weighted averages of causal effects" without requiring homogeneity restrictions. This flexibility extends to the aggregation step, where researchers can specify any weighting scheme.

**How timing is handled:** The first-stage model is estimated using all untreated observations across all time periods—including not-yet-treated units before their treatment starts. This means the estimator uses all available pre-treatment information to construct the counterfactual, potentially improving efficiency.

**Efficiency advantage:** The paper demonstrates that "our imputation estimator exhibits higher efficiency than other robust estimators with typically 30-360% reduction in standard deviation in simulations and application." This efficiency gain is particularly valuable in settings with many pre-treatment periods.

**Software:** Stata package `did_imputation` (available via SSC and GitHub); R package `didimputation` (available on CRAN) [14][15][17]

### 2.5 Comparative Summary of Heterogeneity Handling

| Feature | Goodman-Bacon (2021) | Callaway & Sant'Anna (2021) | Sun & Abraham (2021) | Borusyak et al. (2024) |
|---|---|---|---|---|
| **Core technique** | Decomposition theorem | Group-time ATT with doubly-robust estimation | Cohort-specific interacted regressions | Imputation of counterfactual Y(0) |
| **Target parameter** | VWATT (variance-weighted ATT) | ATT(g,t) for each group/time | CATT(e,l) aggregated to δ_l | Individual τ_it averaged to ATT |
| **Heterogeneity flexibility** | Diagnoses bias from heterogeneity | Fully flexible: separate ATT(g,t) for each group-time | Cohort-specific dynamic effects, then aggregated | Unrestricted; imputes individual effects before averaging |
| **Avoids forbidden comparisons?** | No (diagnoses their bias) | Yes | Yes | Yes |
| **Efficiency** | N/A (diagnostic only) | Moderate (requires many parameters) | Moderate | High (uses all pre-treatment periods) |

---

## 3. Comparison of Performance Assumptions

### 3.1 Parallel Trends Assumptions

#### 3.1.1 Goodman-Bacon (2021)

Goodman-Bacon relies on an **unconditional** parallel trends assumption. The paper states: "A causal interpretation of two-way fixed effects DD estimates requires both a parallel trends assumption and treatment effects that are constant over time." The paper introduces a "variance-weighted common trends" condition that accounts for how differential trends across timing groups bias the estimate. The identifying assumption generalizes to variance-weighted common trends across timing groups, showing how differential trends bias the estimate depending on whether groups act more as treatments or controls. [1][2]

**Comparison groups:** All possible 2×2 comparisons enter the decomposition—never-treated groups, not-yet-treated groups, and already-treated groups. Timing comparisons using already-treated units as controls are the source of potential negative weighting.

#### 3.1.2 Callaway and Sant'Anna (2021)

Callaway and Sant'Anna offer the most flexible parallel trends framework, providing **two distinct conditional parallel trends assumptions** that researchers can choose between:

**Assumption 4 (Conditional Parallel Trends based on a "Never-Treated" Group):** For each g ∈ 𝒢 and t ∈ {2,...,T} such that t ≥ g−δ: E[Y_t(0)−Y_{t-1}(0) | X, G_g=1] = E[Y_t(0)−Y_{t-1}(0) | X, C=1] a.s. This states that conditional on covariates, average outcomes for the treated group and never-treated group would have followed parallel paths. [5][6]

**Assumption 5 (Conditional Parallel Trends based on "Not-Yet-Treated" Groups):** For each g ∈ 𝒢 and (s,t): E[Y_t(0)−Y_{t-1}(0) | X, G_g=1] = E[Y_t(0)−Y_{t-1}(0) | X, D_s=0, G_g=0] a.s. This imposes conditional parallel trends between group g and groups not yet treated by a certain time. [5][6]

**Trade-offs between assumptions:** As the paper notes, "practitioners may favor Assumption 4 with respect to Assumption 5 when there is a sizable group of units that do not participate in the treatment in any period." However, "when a 'never-treated' group of units is not available or 'too small,' researchers may favor Assumption 5." A key difference is that "Assumption 4 does not restrict observed pre-treatment trends across groups, whereas Assumption 5 does." [6]

**Conditional on covariates:** Both assumptions allow the parallel trends assumption to hold after conditioning on observed covariates (X). "This can be important in many applications in economics particularly in cases where there are covariate specific trends in outcomes over time and when the distribution of covariates is different across groups." [6]

#### 3.1.3 Sun and Abraham (2021)

Sun and Abraham (2021) rely on an **unconditional** parallel trends assumption across cohorts. As stated in the paper, "Identification rests on parallel trends across cohorts, no anticipation, and treatment-effect stability within a cohort across calendar time." [9][10]

**Comparison groups:** The interaction-weighted estimator uses either never-treated units or last-treated units as the comparison group. This differs from Callaway and Sant'Anna, which also allows not-yet-treated units more broadly. [9][12]

#### 3.1.4 Borusyak, Jaravel, and Spiess (2024)

The imputation estimator relies on an **unconditional** parallel trends assumption in its baseline version. The first-stage model Y_it = α_i + α_t + ε_it (estimated on never-treated and not-yet-treated observations) assumes that untreated potential outcomes follow a linear additive model with unit and time fixed effects. [14][15]

**Comparison groups:** The estimator uses "ALL untreated observations (never-treated + not-yet-treated) to estimate the counterfactual model." This means the comparison group includes both units that are never treated and units that will be treated in the future but have not yet been treated. [14][17]

**Key quote:** "Second, the imputation approach is intuitive and transparently links the parallel trends and no-anticipation assumptions to the estimator." [14]

### 3.2 Treatment Effect Homogeneity Assumptions

#### 3.2.1 Goodman-Bacon (2021)

Goodman-Bacon is fundamentally about the consequences of treatment effect heterogeneity. The paper's central finding is that TWFE requires **both** parallel trends **and** treatment effects constant over time for a causal interpretation. When effects vary over time, some 2×2 components receive negative weights. The paper does not propose a new estimator but provides diagnostic tools to assess the severity of this problem. [1][2]

#### 3.2.2 Callaway and Sant'Anna (2021)

The method is "explicitly designed to allow for arbitrary treatment effect heterogeneity and dynamic effects." ATT(g,t) is defined separately for each group and time period, imposing no restrictions on how effects vary. The paper emphasizes that "the ATT(g,t) does not impose any restriction on treatment effect heterogeneity across groups or across time." This is a complete departure from the homogeneity assumptions required by TWFE. [5][6]

#### 3.2.3 Sun and Abraham (2021)

The interaction-weighted estimator allows for **fully heterogeneous treatment effects across cohorts and over time** in the sense that CATT(e,l) is separately estimated for each cohort and each event time. The aggregation then uses cohort-share weights to produce interpretable dynamic treatment effect estimates. The key advance is showing that standard TWFE event studies are contaminated by heterogeneity, and the IW estimator resolves this. [9][10]

#### 3.2.4 Borusyak, Jaravel, and Spiess (2024)

The imputation estimator accommodates **heterogeneous treatment effects** in its aggregation step. After imputing counterfactual outcomes, individual treatment effects τ̂_it = Y_it − Ŷ_it(0) can be averaged over any researcher-specified set of weights. The paper notes that "conventional regression-based estimators fail to provide unbiased estimates of relevant estimands absent strong restrictions on treatment-effect homogeneity." [14][15]

### 3.3 Anticipation Effects

#### 3.3.1 Callaway and Sant'Anna (2021): The Most Explicit Handling

Callaway and Sant'Anna provide the **most explicit and flexible handling of anticipation effects** through **Assumption 3 (Limited Treatment Anticipation):**

"There is a known δ ≥ 0 such that E[Y_t(g) | X, G_g = 1] = E[Y_t(0) | X, G_g = 1] a.s. for all g ∈ 𝒢, t ∈ {1,...,T} such that t < g − δ."

**Key features:**
- When δ = 0, this imposes a "no-anticipation" assumption—appropriate "when the treatment path is not a priori known and/or when units are not the ones who 'choose' treatment status."
- When δ > 0, the method accommodates anticipated treatment. For instance, "if units anticipate treatment by one period, Assumption 3 would hold with δ = 1."
- **Reference period adjustment:** Under anticipation, "we can use the time period t = g − δ − 1 as an appropriate reference time period." The more anticipation is allowed (higher δ), "the further back in time one needs to go."
- **Implication for parallel trends:** "The parallel trends assumptions become stronger as one increases δ" because they must hold over a longer pre-treatment window. [5][6]

#### 3.3.2 Sun and Abraham (2021)

Sun and Abraham incorporate the no-anticipation assumption as a core identifying condition. Pre-treatment leads in the event-study specification serve as placebo tests for the parallel trends assumption. The paper cautions that under no anticipation, contaminated TWFE pre-treatment coefficients can appear non-zero due to heterogeneity contamination. The IW estimator does **not** have an explicit parameter analogous to Callaway and Sant'Anna's δ; instead, it relies on the standard event-study framework with pre-treatment leads. [9][10]

#### 3.3.3 Borusyak, Jaravel, and Spiess (2024)

The imputation estimator handles anticipation by allowing the user to specify anticipation periods. When anticipation is specified, those periods are treated as contaminated and excluded from the estimation of the counterfactual model. The paper notes that "fully dynamic specifications are under-identified when there are no never-treated units due to anticipation effects." The `did_imputation` package accepts an `anticipation` parameter that affects which observations are considered "untreated" for estimation purposes. [14][15][17]

#### 3.3.4 Goodman-Bacon (2021)

Goodman-Bacon does not explicitly discuss anticipation effects in the core assumptions of the decomposition. The paper's identifying assumption comprises constant treatment effects over time plus parallel trends, with no formal no-anticipation condition. [1][2]

### 3.4 Summary of Assumptions Comparison

| Dimension | Goodman-Bacon (2021) | Callaway & Sant'Anna (2021) | Sun & Abraham (2021) | Borusyak et al. (2024) |
|---|---|---|---|---|
| **Parallel trends type** | Unconditional only | **Conditional (on X) or unconditional** | Unconditional only | Unconditional only (baseline) |
| **Comparison group options** | All comparisons (diagnostic) | Never-treated OR not-yet-treated | Never-treated OR last-treated | All untreated observations |
| **Covariates** | Yes (time-varying controls) | Yes (conditional PT) | Yes (via reghdfe) | Yes (time-varying controls) |
| **Treatment effect homogeneity** | Required for TWFE validity | **Not required** (fully flexible) | **Not required** (fully flexible) | **Not required** (fully flexible) |
| **Anticipation handling** | Not explicitly addressed | **Most explicit**: δ parameter, reference period adjustment | Standard no-anticipation; pre-treatment leads as tests | User-specified anticipation periods excluded from estimation |
| **Anticipation flexibility** | None | δ ≥ 0 can be any integer | Binary (anticipate or not) | User-specified periods |

---

## 4. Adoption Rates in Top Economics Journals (2020-2024)

### 4.1 Methodological Papers Published in Top Journals

The foundational methodological papers themselves have been published in top outlets:
- de Chaisemartin and D'Haultfœuille (2020): **"Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects,"** *American Economic Review*, 110(9), 2964-2996 [18]
- Roth and Sant'Anna (2023): **"Efficient Estimation for Staggered Rollout Designs,"** *Journal of Political Economy: Microeconomics*, 1(4), 669-709 [19]
- Borusyak, Jaravel, and Spiess (2024): **"Revisiting Event-Study Designs: Robust and Efficient Estimation,"** *Review of Economic Studies*, 91(6), 3253-3285 [14]
- Roth (2022): **"Pretest with Caution: Event-study Estimates After Testing for Parallel Trends,"** *American Economic Review: Insights*, 4(3), 305-322 [20]
- Rambachan and Roth (2023): **"A More Credible Approach to Parallel Trends,"** *Review of Economic Studies*, 90(5), 2555-2591 [21]

### 4.2 Systematic Evidence on Adoption Patterns

#### Citation Impact as a Proxy for Adoption

The citation counts from Google Scholar provide strong evidence of widespread engagement:
- **Callaway and Sant'Anna (2021):** **12,073 citations** [22]
- **Borusyak, Jaravel, and Spiess (2024):** **5,118 citations** [23]
- **Sun and Abraham (2021):** Estimated in the thousands (exact count varies)
- **Goodman-Bacon (2021):** Estimated in the thousands
- **Roth, Sant'Anna, Bilinski, and Poe (2023):** **3,177 citations** for "What's trending in difference-in-differences?" [24]

The **Callaway and Sant'Anna (2021)** paper is the single most cited methodological paper in this literature, suggesting it is the most widely used approach.

#### Evidence from Published Application Papers

**Labor Economics:**
- **Cengiz, Dube, Lindner, and Zipperer (2019)** on minimum wage effects (*Quarterly Journal of Economics*) uses a stacked DiD estimator that anticipates the heterogeneity-robust literature and is frequently cited alongside the newer methods. Their data is used as an empirical reference in Callaway and Sant'Anna's software documentation. [25]
- A paper on "Long-term employment effects of the minimum wage in Germany" (published in *Journal of Comparative Economics*, 2024) explicitly states: "We use the estimators developed by Sun and Abraham (2021), Callaway and Sant'Anna (2021), Borusyak et al. (2021), and De Chaisemartin and d'Haultfoeuille (2022a)" — demonstrating the practice of reporting **all four estimators** as robustness checks. [26]
- A Stack Overflow discussion documents an applied researcher using all four estimators in a staggered DiD setup, finding that "the results of 3/4 are very similar, however, the Borusyak estimates are substantially higher than any of those (between 50% and 100%)." The user notes pre-trends are similar across all estimators, suggesting the differences arise from data structure rather than assumption violations. [27]

**Health Economics:**
- A study titled **"Labor and Coverage Effects of Medicaid Expansion: A Callaway–Sant'Anna Approach to Staggered Adoption"** "employs both traditional two-way fixed effects and the modern Callaway and Sant'Anna (2021) estimator that addresses staggered adoption and heterogeneous treatment effects." Findings show "Medicaid expansion led to substantial gains in public insurance coverage" while "labor market outcomes...remained largely unchanged." [28]
- A study revisiting the ACA Medicaid expansion on interstate migration "employing difference-in-differences (DiD) and advanced staggered DiD methodologies to account for varying state expansion timing" found "evidence of increased migration from non-expansion-to-expansion states." [29]
- The article **"Advances in Difference-in-differences Methods for Policy Evaluation Research"** (PMC, 2024) provides a comprehensive review stating: "Recent economics literature has revealed that DiD estimators may exhibit bias when heterogeneous treatment effects, a common consequence of staggered policy implementation, are present." The review covers Callaway and Sant'Anna, Borusyak et al., Sun and Abraham, de Chaisemartin and D'Haultfœuille, Cengiz et al., and Wooldridge. [30]

#### Evidence from Systematic Audits

The **Baker, Larcker, and Wang (2022)** paper ("How Much Should We Trust Staggered Difference-In-Differences Estimates?," *Journal of Financial Economics*, 144(2), 370-395) is particularly influential. It documents that **staggered DiD designs comprise nearly half of DiD studies in top finance and accounting journals from 2000-2019**. When they re-analyze three major published studies using the new robust estimators, they find that "original findings often become statistically insignificant or change in magnitude and direction." [31]

The **Research Unit Tests** guide (dahis.com/research-unit-tests) codifies best practices for journal referees: "Papers with staggered adoption must use or address heterogeneity-robust estimators. Either: (a) the paper uses a heterogeneity-robust estimator as its primary specification, OR (b) the paper uses TWFE but reports robustness to a heterogeneity-robust estimator and discusses the Goodman-Bacon decomposition explicitly. Papers written after 2022 have no excuse." [32]

### 4.3 Which Estimator Has Achieved "Methodological Dominance"?

Based on the available evidence, **no single estimator has achieved unambiguous dominance** across all dimensions. Instead, the literature reveals a nuanced pattern:

1. **Callaway and Sant'Anna (2021)** is the most cited (12,073 citations) and is often used as the default heterogeneity-robust estimator, particularly in health economics applications. Its ability to handle covariates conditionally and its flexible anticipation parameter make it attractive for applications with rich covariate data.

2. **Borusyak et al. (2024)** , despite being published later (2024 vs. 2021), has accumulated **5,118 citations** and is noted for superior efficiency (30-360% reduction in standard deviations). The imputation approach is described as "easier to use" by some practitioners and is increasingly common.

3. **Sun and Abraham (2021)** remains the standard for event-study applications that want a "drop-in" replacement for TWFE event-study regressions. The `sunab()` function in R's `fixest` package makes it the easiest to implement as a simple regression modification.

4. **Best practice increasingly involves reporting multiple estimators** as robustness checks. The Germany minimum wage paper using all four estimators exemplifies the emerging norm.

5. The **forthcoming "Practitioner's Guide"** by Baker, Sant'Anna, et al. in the *Journal of Economic Literature* suggests the literature is consolidating around these methods. [33]

### 4.4 Software Availability and Implementation in Applied Work

| Package | Language | Estimator | Key Features |
|---|---|---|---|
| `did` | R | Callaway & Sant'Anna | Group-time ATTs, doubly-robust, pre-testing |
| `csdid` | Stata | Callaway & Sant'Anna | Native Stata implementation |
| `xthdidregress` | Stata 18 | Callaway & Sant'Anna | Built-in Stata 18 command |
| `eventstudyinteract` | Stata | Sun & Abraham | Interaction-weighted estimator |
| `sunab()` via `fixest` | R | Sun & Abraham | Single regression with `feols` |
| `did_imputation` | Stata | Borusyak et al. | Imputation approach |
| `didimputation` | R | Borusyak et al. | Imputation with pre-trend testing |
| `did_multiplegt` | Stata | de Chaisemartin & D'Haultfœuille | Handles non-binary, non-absorbing treatments |
| `DIDmultiplegt` | R | de Chaisemartin & D'Haultfœuille | R version |
| `bacondecomp` | Stata/R | Goodman-Bacon | Decomposition diagnostic |
| `drdid` | Stata | Sant'Anna & Zhao | Doubly-robust DiD |

---

## 5. How Newer Estimators Address Roth's (2022) Concerns About Pre-Trend Testing

### 5.1 Roth's Core Argument

Jonathan Roth's (2022) paper "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends" (AER: Insights) identifies two fundamental problems with the common practice of testing for pre-trends:

**1. Low statistical power:** "Conventional pre-trends tests may have low power, meaning that preexisting trends that produce meaningful bias in the treatment effects estimates may not be detected with substantial probability." Roth surveys 12 published papers and finds many tests fail to detect violations that cause biases comparable to or larger than the estimated treatment effects.

**2. Pretest bias:** "Conditioning the analysis on the result of a pretest induces distortions to estimation and inference from pretesting...the bias caused by a violation of parallel trends can actually be worse conditional on passing the pretest." Roth shows that under homoskedasticity, "the bias will always be larger after surviving the pretest whenever the difference in trends is monotonically increasing over time."

The paper demonstrates that in simulations calibrated to published papers, bias from undetected trends can exceed the treatment effect, with nominal 95% confidence intervals containing the true effect less than 25% of the time. [20][34]

### 5.2 The Pretrends Sensitivity Analysis Proposed by Roth

Roth proposes two complementary tools that have been widely adopted:

#### The `pretrends` Package (Roth, 2022)

The `pretrends` package provides "tools for power calculations and visualization for pre-trends tests." Its purpose is to quantify how likely a pre-trends test can detect violations of the parallel trends assumption—analogous to minimal detectable effect sizes in RCTs. [35][36]

**Key functions:**
- **`slope_for_power()`**: Calculates the slope of a linear violation of parallel trends that a pre-trends test would detect a specified fraction of the time (e.g., 80% power). This provides the **minimum detectable violation (MDV)** — "the smallest violation magnitude discernible at a chosen power level."
- **`pretrends()`**: Enables power analyses and visualization given event-study results and a user-hypothesized violation. It computes statistics such as power, Bayes factors, and likelihood ratios.

The package "can accommodate event-study results from any asymptotically normal estimator, **including Callaway and Sant'Anna (2020) and Sun and Abraham (2020)** ." [35] This is crucial: it means Roth's sensitivity analysis is designed to work with the heterogeneity-robust estimators, not just TWFE.

**Software:** Available as:
- **R package:** `pretrends` on GitHub (github.com/jonathandroth/pretrends) [35]
- **Stata package:** `pretrends` on SSC (boc:bocode:s459205) [36]
- **Python module:** Via `diff-diff` package [37]
- **Shiny app:** Interactive web application for non-R users [35]

#### The HonestDiD Package (Rambachan and Roth, 2023)

The `HonestDiD` package implements "robust inference and sensitivity analysis for differences-in-differences and event study designs." Instead of asking "Do parallel trends hold?" it asks "How large would violations of parallel trends need to be before my conclusion changes?" The answer is called the **breakdown value** — "a single number that tells the reader exactly how robust the result is." [38][39]

**Key features:**
- **Bounds on relative magnitudes (M̄):** Post-treatment violations cannot be much larger than the maximum pre-treatment violation
- **Smoothness restrictions (M):** Limits how much the slope of difference in trends changes between periods
- **Robust confidence intervals:** Guarantee coverage if restrictions hold
- **Integration:** "Supports integration with staggered DiD methods like Sun and Abraham (fixest) and Callaway and Sant'Anna (did)"

**Software:**
- **R package:** `HonestDiD` on GitHub (github.com/asheshrambachan/HonestDiD) [38]
- **Stata package:** `honestdid` on SSC [39]
- **Shiny app:** Interactive web application [38]

### 5.3 How Each Estimator's Assumptions Interact with Pre-Trend Testing

#### 5.3.1 Callaway and Sant'Anna (2021)

Callaway and Sant'Anna provide **dedicated pre-testing functionality** in their `did` R package that directly addresses Roth's concerns:

**Key contributions to pre-trend testing:**
- **Valid pre-tests even with selective treatment timing:** "Contrary to event study regressions, pre-tests based on group-time average treatment effects are still valid even in the presence of selective treatment timing." This is because, unlike TWFE event studies, ATT(g,t) for pre-treatment periods are not contaminated by treatment effects from other periods. [6][40]
- **Conditional parallel trends pre-testing:** The `conditional_did_pretest` function tests the conditional parallel trends assumption when covariates are included. This addresses the concern that "pre-tests based only on group-time average treatment effects can fail to detect some violations" when covariates matter. [40]
- **Simultaneous confidence bands:** The package reports "simultaneous confidence bands in plots of group-time average treatment effects with multiple time periods—confidence bands robust to multiple hypothesis testing." This addresses the multiple-testing problem inherent in examining multiple pre-treatment periods. [40]
- **Explicit reference to Roth:** The vignette states "see Roth (2022) for limitations in parallel trends tests," acknowledging that pretests only provide evidence, not proof. [40]

**How conditional parallel trends interacts with pretesting:** Because Callaway and Sant'Anna allow parallel trends to hold conditionally on covariates, the pre-trend test becomes a test of **conditional** parallel trends—a weaker and often more plausible assumption. This means that even if unconditional pre-trends show violations, the conditional specification might pass pre-trends tests, improving the credibility of the design.

#### 5.3.2 Sun and Abraham (2021)

Sun and Abraham's interaction-weighted estimator addresses pre-trend testing by **eliminating the contamination of pre-treatment coefficients** that plagues TWFE event studies:

**Key contributions:**
- **Contamination-free pre-treatment coefficients:** The paper demonstrates that "the coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity." The IW estimator resolves this by estimating cohort-specific coefficients. [9][10]
- **Clean placebo tests:** Pre-treatment coefficients in the IW estimator are not contaminated by treatment effects, making them valid placebo tests for parallel trends. This directly addresses the concern that TWFE pre-trend tests could reject due to heterogeneity rather than actual violations.
- **Weight transparency:** The estimator reports weights corresponding to cohort shares, ensuring that pre-treatment estimates have clear interpretation.

**Interaction with pretesting:** The `pretrends` package explicitly supports input from Sun and Abraham (2020) estimates. The package vignette states that it can accommodate "event-study results from any asymptotically normal estimator, including Callaway and Sant'Anna (2020) and Sun and Abraham (2020)." [35]

#### 5.3.3 Borusyak, Jaravel, and Spiess (2024)

Borusyak et al.'s imputation estimator offers a distinctive approach to pre-trend testing:

**Key contributions:**
- **Transparent link between assumptions and estimator:** "The imputation approach is intuitive and transparently links the parallel trends and no-anticipation assumptions to the estimator." This transparency makes it easier to understand what violations would affect the estimates. [14][15]
- **Uses all pre-treatment periods for imputation:** "Our estimator uses all pre-treatment periods for imputation, as appropriate under the standard DiD assumptions, while alternative estimators use more limited" pre-treatment data. This can increase efficiency but also means the estimator is sensitive to violations in any pre-treatment period. [14]
- **Pre-trend testing implemented:** The R function `did_imputation` explicitly supports "treatment effect estimation and pre-trend testing in staggered adoption diff-in-diff designs with an imputation approach." It can conduct event-study analyses with pre-treatment coefficients as placebo tests. [17]
- **Fully dynamic specification under-identification:** A critical finding is that "fully dynamic specifications are under-identified when there are no never-treated units due to anticipation effects." This means researchers must be careful about which pre-treatment periods are used for identification. [14]

**Interaction with pretesting:** Because the imputation estimator uses all pre-treatment observations to estimate the counterfactual model, a violation of parallel trends in any pre-treatment period can bias the estimated fixed effects. This makes pre-trend testing **more critical** for the imputation estimator than for approaches that only use a single pre-treatment period as the baseline. At the same time, the estimator's efficiency advantage means that pre-trend tests may have higher power.

#### 5.3.4 Comparison of Pre-Trend Testing Across Estimators

| Dimension | Callaway & Sant'Anna (2021) | Sun & Abraham (2021) | Borusyak et al. (2024) |
|---|---|---|---|
| **Pre-treatment period used** | Single reference period (g-1 or g-δ-1) | Single omitted reference period (l=-1) | **All pre-treatment periods** for imputation |
| **Pre-test contamination?** | No (separate ATT(g,t) for pre-periods) | No (cohort-specific CATT) | No (but all periods affect counterfactual model) |
| **Conditional pre-test available?** | **Yes** (conditional_did_pretest) | No (unconditional only) | Limited (via covariates in first stage) |
| **Multiple testing adjustment** | **Simultaneous confidence bands** | Pointwise confidence intervals | Standard multiple testing tools |
| **Compatibility with pretrends package** | Yes | Yes | Yes |
| **Compatibility with HonestDiD** | Yes | Yes | Yes |

### 5.4 How Applied Researchers Have Responded to Roth's Critique

The applied literature shows several clear responses to Roth's (2022) concerns:

**1. Widespread adoption of heterogeneity-robust estimators:** The Baker, Larcker, and Wang (2022) paper was influential in documenting that "standard DiD regression estimates with staggered treatment timing often do not provide valid estimates of the causal estimands...even under random assignment of treatment." Papers after 2022 are increasingly expected to use robust estimators. [31]

**2. Pre-test reporting with HonestDiD sensitivity analysis:** A growing number of applied papers now include HonestDiD sensitivity analysis as standard practice. As one tutorial concludes: "The breakdown value is the single most informative number to report alongside any DiD estimate. It tells the reader exactly how much they need to doubt parallel trends before the result breaks down." [39]

**3. Use of pretrends for power diagnostics:** Applied researchers are increasingly using the `pretrends` package to compute the power of their pre-trend tests, replacing the traditional binary "parallel trends holds/does not hold" conclusion with a quantitative robustness measure.

**4. Reporting multiple comparison groups:** Best practice now involves showing results with both never-treated and not-yet-treated comparison groups to assess sensitivity.

**5. Incorporation of context-specific economic knowledge:** As Roth recommends, "researchers should draw on context-specific theory to provide evidence as to why treatment and comparison groups would be expected to trend in parallel in the absence of intervention."

**6. The Roth, Sant'Anna, Bilinski, and Poe (2023) practitioner checklist** (3,177 citations) provides a framework for applied researchers, including guidance on treatment timing, parallel trends validity, and inference methods. [24][41]

---

## 6. Practical Guidance for Applied Researchers

### 6.1 Recommended Workflow Based on the Literature

Based on the synthesis of methodological guidance from the papers and the practitioner literature, the following workflow represents current best practice:

1. **Plot the treatment rollout** (e.g., using `panelView` R package) to understand the timing structure [42]
2. **Document how many units are treated in each cohort** and the share of never-treated units
3. **Plot the evolution of average outcomes across cohorts** to visually assess parallel trends
4. **Choose comparison group and parallel trends assumption** carefully:
   - If a large never-treated group exists: Use Callaway & Sant'Anna with Assumption 4
   - If few never-treated units: Use not-yet-treated assumption (Assumption 5)
5. **Use an event-study analysis without covariates** and assess if parallel trends is plausible
6. **If unconditional parallel trends is not plausible, incorporate covariates** (Callaway & Sant'Anna's doubly-robust estimator is designed for this)
7. **When using covariates, check for overlap** (positivity condition)
8. **Report results from multiple heterogeneity-robust estimators** (at least two of: Callaway & Sant'Anna, Sun & Abraham, Borusyak et al.)
9. **Conduct power analysis** using the `pretrends` package to assess the minimum detectable violation of parallel trends
10. **Conduct sensitivity analysis** using `HonestDiD` to compute breakdown values
11. **If conditional parallel trends is not plausible, consider alternative methods** (synthetic control, instrumental variables)

### 6.2 Choosing Between Estimators

| Setting | Recommended Estimator | Rationale |
|---|---|---|
| Strong theoretical basis for parallel trends; few covariates | Sun & Abraham (2021) | Simple "drop-in" replacement for TWFE event study |
| Rich covariate data; conditional PT more plausible | Callaway & Sant'Anna (2021) | Doubly-robust with covariates; conditional PT tests |
| Many pre-treatment periods; efficiency matters | Borusyak et al. (2024) | Uses all pre-period data; 30-360% efficiency gain |
| Non-binary or non-absorbing treatments | de Chaisemartin & D'Haultfœuille (2020) | Handles multi-valued and reversible treatments |
| Need to diagnose TWFE bias | Goodman-Bacon (2021) | Decomposition diagnostic |

---

## 7. Conclusion

The staggered DiD literature has undergone a remarkable transformation since Goodman-Bacon's (2021) decomposition revealed the vulnerabilities of TWFE estimators. Three main approaches have emerged—Callaway and Sant'Anna's (2021) group-time ATTs with doubly-robust estimation, Sun and Abraham's (2021) interaction-weighted estimator, and Borusyak et al.'s (2024) imputation-based estimator—each offering distinct advantages in handling heterogeneous treatment effects and dynamic treatment timing.

The **assumptions** underlying these estimators differ in important ways. Callaway and Sant'Anna offer the most flexible framework, allowing conditional parallel trends with covariates and a formal parameter for anticipation effects. Sun and Abraham and Borusyak et al. rely on unconditional parallel trends in their baseline versions, though both can accommodate covariates. The choice between never-treated and not-yet-treated comparison groups is a critical design decision that affects which parallel trends assumption is invoked.

Regarding **adoption rates**, the Callaway and Sant'Anna (2021) paper is the most cited (12,073 citations), suggesting it is the most widely used approach. However, Borusyak et al. (2024) has accumulated 5,118 citations in just two years and is noted for superior efficiency. Best practice increasingly involves reporting multiple estimators as robustness checks, with the four-estimator approach (Sun & Abraham, Callaway & Sant'Anna, Borusyak et al., and de Chaisemartin & D'Haultfœuille) becoming common in methodologically rigorous applications.

Finally, **Roth's (2022) critique of pre-trend testing** has been directly addressed by the newer estimators in several ways: (1) eliminating the contamination of pre-treatment coefficients that plagued TWFE event studies, (2) providing dedicated pre-testing functions with simultaneous inference (Callaway & Sant'Anna's `did` package), (3) enabling transparent linking of assumptions to estimators (Borusyak et al.'s imputation approach), and (4) being compatible with Roth's own `pretrends` and `HonestDiD` sensitivity analysis packages. The consensus emerging from this literature is that pre-trend tests should be supplemented with power analyses and formal sensitivity analyses, rather than being used as binary "validity checks."

The *Journal of Economic Literature*'s forthcoming "Practitioner's Guide" (Baker, Sant'Anna, et al.) signals that these methods have matured enough for codification as best practice. For applied researchers in labor and health economics, the message is clear: papers with staggered adoption should use at least one heterogeneity-robust estimator as their primary specification, or (at minimum) report robustness checks using such estimators alongside the Goodman-Bacon decomposition. Papers written after 2022 are increasingly held to this standard.

---

## Sources

[1] Goodman-Bacon (2021) - Journal of Econometrics: https://www.sciencedirect.com/science/article/abs/pii/S0304407621001445

[2] Goodman-Bacon (2021) - Full text PDF: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Goodman-Bacon_2021_Difference-in-differences_with_variation_in_treatment_timing.pdf

[3] Goodman-Bacon (2018) - NBER Working Paper: https://www.nber.org/papers/w25018

[4] bacondecomp Stata/R package: https://github.com/evanjflack/bacondecomp

[5] Callaway & Sant'Anna (2021) - Journal of Econometrics: https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948

[6] Callaway & Sant'Anna (2021) - Full text PDF: https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[7] Callaway & Sant'Anna (2018/2020) - arXiv working paper: https://arxiv.org/abs/1803.09015

[8] did R package documentation: https://bcallaway11.github.io/did/articles/multi-period-did.html

[9] Sun & Abraham (2021) - Journal of Econometrics: https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html

[10] Sun & Abraham (2021) - Full text PDF: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Paper2023/D3--Sun-2020-Estimating-dynamic-treatment-effects-in-event-studies-with-heterogeneous-treatment-effects.pdf

[11] Sun & Abraham (2018/2020) - arXiv working paper: https://arxiv.org/abs/1804.05785

[12] MetricGate - Sun-Abraham estimator: https://metricgate.com/docs/event-study-sun-abraham

[13] eventstudyinteract Stata package (GitHub): https://github.com/lsun20/EventStudyInteract

[14] Borusyak, Jaravel & Spiess (2024) - Review of Economic Studies: https://academic.oup.com/restud/article/91/6/3253/7601390

[15] Borusyak, Jaravel & Spiess (2024) - arXiv working paper: https://arxiv.org/abs/2108.12419

[16] Borusyak, Jaravel & Spiess - LSE Research Online: https://researchonline.lse.ac.uk/id/eprint/123781/1/Jaravel_revisiting-even-study-designs--published.pdf

[17] did_imputation Stata package (GitHub): https://github.com/borusyak/did_imputation

[18] de Chaisemartin & D'Haultfœuille (2020) - American Economic Review: https://econpapers.repec.org/RePEc:aea:aecrev:v:110:y:2020:i:9:p:2964-96

[19] Roth & Sant'Anna (2023) - Journal of Political Economy: Microeconomics: https://www.journals.uchicago.edu/doi/10.1086/726596

[20] Roth (2022) - American Economic Review: Insights: https://www.aeaweb.org/articles?id=10.1257/aeri.20210236

[21] Rambachan & Roth (2023) - Review of Economic Studies: https://academic.oup.com/restud/article/90/5/2555/6950347

[22] Callaway Google Scholar: https://scholar.google.com/citations?user=BVWmSY4AAAAJ&hl=en

[23] Borusyak Google Scholar: https://scholar.google.com/citations?user=dix7FRQAAAAJ&hl=en

[24] Roth, Sant'Anna, Bilinski & Poe (2023) - "What's trending in difference-in-differences?": https://www.jonathandroth.com/assets/files/DiD_Review_Paper.pdf

[25] Cengiz, Dube, Lindner & Zipperer (2019) - QJE: https://academic.oup.com/qje/article/134/3/1405/5420451

[26] Minimum wage Germany paper - Journal of Comparative Economics: https://www.sciencedirect.com/science/article/pii/S0927537124001441

[27] Stack Overflow: Staggered DiD estimator comparison: https://stackoverflow.com/questions/76101056/staggered-did-borusyak-estimates-way-higher-than-callawaysantanna-sunabra

[28] Medicaid Expansion - Callaway-Sant'Anna approach: https://arctic.gsu.edu/2025/07/22/labor-and-coverage-effects-of-medicaid-expansion-a-callaway-santanna-approach-to-staggered-adoption

[29] ACA Medicaid expansion migration study: https://www.strath.ac.uk/media/1newwebsite/departmentsubject/economics/research/researchdiscussionpapers/22-08_Revisiting_The_Effect_of_the_Affordable_Care_Act.pdf

[30] Advances in DiD Methods - PMC (2024): https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929

[31] Baker, Larcker & Wang (2022) - Journal of Financial Economics: https://www.ecgi.global/sites/default/files/working_papers/documents/bakerlarckerwangfinal_0.pdf

[32] Research Unit Tests - Staggered DiD guidelines: https://www.ricardodahis.com/research-unit-tests/tests/did-staggered-heterogeneous-effects

[33] Sant'Anna Research Page: https://pedrohcgs.github.io/research

[34] Roth (2022) - Full text PDF: https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf

[35] pretrends R package (GitHub): https://github.com/jonathandroth/pretrends

[36] pretrends Stata package (SSC): https://ideas.repec.org/c/boc/bocode/s459205.html

[37] diff-diff Python documentation - Pre-Trends Power: https://diff-diff.readthedocs.io/en/stable/api/pretrends.html

[38] HonestDiD R package (GitHub): https://github.com/asheshrambachan/HonestDiD

[39] honestdid Stata package (SSC): https://ideas.repec.org/c/boc/bocode/s459138.html

[40] Pre-Testing in the did R package: https://bcallaway11.github.io/did/articles/pre-testing.html

[41] Roth et al. (2023) - DiD synthesis: https://www.jonathandroth.com/assets/files/DiD_Review_Paper.pdf

[42] Sant'Anna - DiD Resources and Checklist: https://psantanna.com/did-resources