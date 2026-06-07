# Methodological Tensions in Staggered Difference-in-Differences Estimation: A Deepened Research Report

## 1. Introduction

The past five years have witnessed a fundamental rethinking of difference-in-differences (DiD) methodology, driven by the recognition that traditional two-way fixed effects (TWFE) estimators can produce severely misleading results when treatment adoption is staggered and treatment effects are heterogeneous. The critique, crystallized in Goodman-Bacon's (2021) decomposition theorem, revealed that the TWFE estimator is a weighted average of many 2×2 DiD comparisons, some of which use already-treated units as controls—creating "forbidden comparisons" that can generate negative weights and bias estimates of any sign [50][3][9].

This report provides a comprehensive, deepened analysis of four leading approaches: **Callaway and Sant'Anna (2021)** , **Sun and Abraham (2021)** , **Borusyak, Jaravel, and Spiess (2024)** , and the diagnostic framework of **Goodman-Bacon (2021)** . It extends beyond conceptual overviews to systematically compare implementation details, explicitly link estimator features to methodological critiques, and provide enriched evidence on applied adoption patterns and reporting guidance in top economics journals for labor and health economics applications published 2020–2024.

---

## 2. Systematic Review of Applied Methodological Practice in Leading Journals (2020-2024)

### 2.1 The Methodological Landscape

The literature has produced several heterogeneity-robust estimators that address the limitations of TWFE in staggered adoption settings:

**Callaway and Sant'Anna (2021)** introduced the group-time average treatment effect ATT(g,t) as the core building block, using a "divide-and-conquer" strategy that decomposes data into many 2x2 DiD comparisons defined by cohort and time [2][5][19][20]. The estimator allows covariates via doubly robust estimators combining outcome regression and inverse probability weighting, and supports never-treated or not-yet-treated units as comparison groups [2][5][19][20]. Published in the *Journal of Econometrics* (Vol. 225, Issue 2, pp. 200-230), this paper has over 12,000 citations as of 2025-2026, making it the most cited modern DiD estimator by a wide margin [13].

**Sun and Abraham (2021)** proposed an interaction-weighted (IW) estimator that averages cohort-specific average treatment effects on the treated with non-negative weights summing to one [6][10]. The paper demonstrates that in settings with staggered treatment timing, OLS coefficients in TWFE event-study specifications are contaminated by treatment effects from other periods, complicating causal interpretation [6][10]. Published in the *Journal of Econometrics* (Vol. 225, Issue 2, pp. 175-199), this paper has received 7,679 citations [10].

**Borusyak, Jaravel, and Spiess (2024)** developed an imputation-based approach that estimates unit and time fixed effects from untreated observations, then imputes counterfactual outcomes for treated units [11][12][13][14]. Published in the *Review of Economic Studies* (Vol. 91, Issue 6, pp. 3253-3285), this paper has 5,149 citations [13].

**Goodman-Bacon (2021)** proved that the TWFE DiD estimator can be decomposed into a weighted average of all possible 2x2 DiD estimators, with weights that can be negative when average treatment effects vary over time [8][9]. Published in the *Journal of Econometrics* (Vol. 225, Issue 2, pp. 254-277), this diagnostic tool has passed the original synthetic control article in total citations [12][13].

**Additional approaches** include stacked DiD (Cengiz et al. 2019, QJE) [8][4][9], de Chaisemartin and D'Haultfœuille (2020, AER) [5][7][18], LP-DiD (Dube et al. 2023, NBER WP 31184) [7][14], and two-stage/imputation estimators (Gardner et al. 2024) [14].

### 2.2 Adoption Rates in Top-5 General Interest Journals (2020-2024)

Based on systematic searches of the *American Economic Review* (AER), *Quarterly Journal of Economics* (QJE), *Journal of Political Economy* (JPE), *Econometrica*, and *Review of Economic Studies* (REStud), the following adoption patterns emerge:

**Callaway and Sant'Anna (2021)** is the most widely cited and used heterogeneity-robust estimator, with over 12,000 citations since 2021 [13]. Implementation is available in R via the 'did' package and Stata via the 'csdid' command [5][19][20]. Monte Carlo simulations show it performs well under heterogeneous treatment effects [14]. The estimator is referenced as a key method in de Chaisemartin and D'Haultfœuille (2024, AER), Callaway, Goodman-Bacon, and Sant'Anna (2024, AEA Papers and Proceedings), and numerous empirical papers [2][3].

**Sun and Abraham (2021)** has received 7,679 citations and is widely used for event-study dynamics [10]. The interaction-weighted estimator is particularly popular because it can be implemented within standard TWFE regression by saturating the model with cohort-by-event-time interactions and then reweighting [6][9]. Implementation is available in Stata via the 'eventstudyinteract' command [9].

**Borusyak, Jaravel, and Spiess (2024)** has 5,149 citations since publication [13]. The imputation-based approach is noted for flexibility in handling treatment reversals, time-varying covariates, and fixed effects [12]. Implementation is available in R via 'didimputation' and Stata via 'did_imputation' commands [10][20].

**Goodman-Bacon (2021)** decomposition has 14,940 total citations, making it one of the most cited DiD methodological papers [12][13]. It is widely used as a diagnostic tool rather than a primary estimator, with implementation in Stata via 'bacondecomp' [7] and R via 'bacondecomp'.

**Estimator choice justification patterns:** Authors justify estimator choices through Monte Carlo simulations (Baker, Larcker, and Wang 2022 demonstrate through simulations that TWFE DiD estimates are unbiased only with single treatment periods or homogeneous treatment effects) [6][7][9][15]; sensitivity analyses (Gardner et al. 2024 test their two-stage estimator against alternatives) [14]; and reliance on published guidance (Callaway 2022 chapter, Roth et al. 2023 synthesis published in Journal of Econometrics, Sant'Anna 2024 modern DiD presentation) [1][3][11].

### 2.3 Modal Patterns in Reporting Workflows

Based on surveyed literature, the modal pattern for combining multiple estimators is:

**Event-study plots:** Often use the Sun-Abraham interaction-weighted estimator or the Borusyak et al. imputation estimator for clean dynamic treatment effect estimates without contamination bias. The interaction-weighted estimator is particularly popular as a "drop-in replacement" for TWFE event-study regressions [6][10][41].

**Average treatment effects:** Often reported using Callaway-Sant'Anna or stacked DiD, which allow straightforward aggregation of group-time ATTs [2][5][19][20].

**Diagnostics:** Goodman-Bacon decomposition is used to assess the extent of "bad comparisons" in TWFE estimates [8][9].

**Triangulation strategies:** Many papers use one heterogeneity-robust method as primary (often the one best-suited to their specific design features) and check robustness with 1-2 alternative methods. A recent trend is to report event-study plots from a single preferred heterogeneity-robust method with summary tables that aggregate treatment effects across alternative estimators [12].

**Quantified adoption rates (from available data):**
- Callaway-Sant'Anna: approximately 12,000+ citations, most widely used robust estimator [13]
- Sun-Abraham: approximately 7,679 citations [10]
- Borusyak et al.: approximately 5,149 citations [13]
- Goodman-Bacon decomposition: approximately 14,940 total citations [12][13]
- Roth, Sant'Anna, Bilinski, and Poe (2023) synthesis paper: over 3,100 citations, indicating widespread adoption of methods guidance

### 2.4 Field-Specific Differences: Labor Economics vs. Health Economics

**Labor Economics applications** using staggered DiD in top-5 journals most commonly employ:
- Stacked DiD event studies (Cengiz et al. 2019 style) [8][6]
- Sun-Abraham interaction-weighted estimator for event-study dynamics [6]
- Callaway-Sant'Anna for group-time average treatment effects [2]
- Relatively frequent use of the Goodman-Bacon decomposition as a diagnostic [8]
- Minimum wage research is a dominant application area, with Callaway and Sant'Anna (2021) itself using teen employment data [2]
- Papers by Clemens, Gentry, and Meer (forthcoming, JoLE) use "difference-in-differences and imputation event-study methods" [17]

**Health Economics applications** more commonly employ:
- Callaway-Sant'Anna estimator (dominant in Medicaid studies) [2]
- Stacked DiD approaches [9]
- Imputation estimators (Borusyak et al.) [11]
- The *Journal of Health Economics* paper on universal free school meals (2024) explicitly uses both Goodman-Bacon decomposition and Borusyak et al. (2024) [15]
- Sarah Miller's research extensively uses DiD for Medicaid expansions and health insurance policies, serving as co-editor of *Journal of Public Economics* and associate editor of both *Journal of Health Economics* and *American Journal of Health Economics*

**Cross-field patterns:**
- The Goodman-Bacon (2021) decomposition is widely cited across both fields as a diagnostic tool [8]
- Callaway and Sant'Anna (2021) is the most cited paper on DiD methods across both fields [13]
- Health economists appear more likely to emphasize conditional parallel trends with covariate adjustment (reflecting the nature of health data) [2]
- Labor economists appear more strongly associated with the stacked DiD approach (originating in Cengiz et al. 2019 for minimum wage research) [8]
- The *Annual Review of Public Health* paper (Wing et al., 2024, DOI: 10.1146/annurev-publhealth-061022-050825) serves as a field-specific translation of methods for health researchers

### 2.5 Specific Empirical Papers in Top-5 Journals

**In AER:**
- de Chaisemartin and D'Haultfœuille (2020), "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects," *American Economic Review*, Vol. 110(9), pp. 2964-2996 [8]
- Goodman-Bacon (2021), "The Long-Run Effects of Childhood Insurance Coverage: Medicaid Implementation, Adult Health and Labor Market Outcomes," *American Economic Review* [13][15]
- de Chaisemartin and D'Haultfœuille (2024), "Contamination Bias in Linear Regressions," *American Economic Review*, 114(12): 4015-4051 [2]

**In QJE:**
- Cengiz, Dube, Lindner, and Zipperer (2019), "The Effect of Minimum Wages on Low-Wage Jobs," *Quarterly Journal of Economics*, 134(3), 1405-1454 [8]
- Brynjolfsson, Li, and Raymond (2025), "Generative AI at Work," *Quarterly Journal of Economics*, Vol. 140(2), pp. 889-949 [6]

**In Review of Economic Studies:**
- Borusyak, Jaravel, and Spiess (2024), "Revisiting Event-Study Designs: Robust and Efficient Estimation," *Review of Economic Studies*, Vol. 91(6), pp. 3253-3285 [11][13]
- Rambachan and Roth (2023), "A More Credible Approach to Parallel Trends," *Review of Economic Studies* [13]

**In Journal of Political Economy: Microeconomics:**
- Callaway and Roth (2023), "Efficient Estimation for Staggered Rollout Designs" [14]

---

## 3. Deepened Connection Between Pre-Trend Testing Features and Roth's (2022) Critique

### 3.1 Roth's (2022) Three Core Concerns

Jonathan Roth (2022), published in *American Economic Review: Insights*, Volume 4, Number 3, pages 305–322 (DOI: 10.1257/aeri.20210236), identifies two main limitations of conventional pre-trend testing [4][5][18]:

**Concern (a): Low Statistical Power.** Roth finds that "conventional pre-trends tests may have low power, meaning that preexisting trends that produce meaningful bias in the treatment effects estimates may not be detected with substantial probability" [4][5]. In simulations calibrated to published papers, "the bias from a trend detected only half the time is larger than the estimated treatment effect and a nominal 95 percent confidence interval contains the true parameter only 24 percent of the time" [4][5]. Standard dynamic event-study specifications may also fail to detect linear violations due to multicollinearity with calendar time indicators [3].

**Concern (b): Pretesting Bias.** Roth finds that "conditioning the analysis on the result of a pretest can distort estimation and inference, potentially exacerbating the bias of point estimates and under-coverage of confidence intervals" [4][5]. A key formal result is that "under homoskedasticity the bias will always be larger after surviving the pretest whenever the difference in trends is monotonically increasing over time" [4][5]. The bias conditional on passing the pre-test can be over twice the unconditional bias in many cases [5].

**Concern (c): Contamination of Pre-Treatment Coefficients.** While not Roth's primary focus, this concern is central to the staggered DiD literature. Sun and Abraham (2021) show that "the coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity" [10]. This means standard event-study plots from TWFE regressions may show apparent violations of parallel trends even when there are no differential pre-trends.

### 3.2 Sun and Abraham (2021): How the Interaction-Weighted Estimator Produces Clean Pre-Treatment Coefficients

**Mechanism for "clean" pre-treatment coefficients:** The Sun-Abraham interaction-weighted estimator proceeds in two stages. In the first stage, it saturates the TWFE regression with interactions between cohort indicators and relative-time indicators, estimating cohort-specific average treatment effects at each relative time period (cohort×relative-time interactions). In the second stage, it aggregates across cohorts using cohort-share weights, ensuring the IW estimates are proper convex combinations of cohort-specific effects, rather than the contaminated implicit weights of standard TWFE [6][10][41].

**Addressing Roth's concerns about spurious pre-trend detection:** "IW estimates for negative relative times (l < 0) should not differ from zero under parallel trends... they provide a cleaner diagnostic than standard TWFE pre-trends" [6]. The pre-treatment IW estimates provide a more reliable test for parallel trends than standard TWFE pre-trends diagnostics because they are free from contamination bias [6]. Because the estimator uses non-negative weights, pre-treatment coefficients cannot be contaminated by post-treatment treatment effects from other cohorts, unlike in TWFE where "dynamic effect estimates for one relative period can be contaminated by treatment effects from other periods" [8].

**Limitations regarding Roth's concerns:** The clean pre-treatment coefficients primarily address concern (c)—the contamination issue. The estimator ensures that pre-treatment coefficients are valid diagnostic tools for parallel trends. However, it does not directly address Roth's concerns (a) and (b) about low statistical power of the pre-trend test itself or the bias that arises from conditioning on the result of a pre-test. Even with clean pre-treatment coefficients, the test may still have low power to detect violations, and the act of conditioning on passing such a test can still induce pretesting bias.

### 3.3 Borusyak et al. (2024): How the Imputation Approach Constructs Pre-Trend Tests

**Rationale for untreated-only pre-trend tests:** The imputation estimator works in two steps. In the first stage, it estimates a model for untreated potential outcomes Y(0) using only untreated/not-yet-treated observations—typically a TWFE model with unit and time fixed effects. In the second stage, it imputes counterfactual outcomes for treated observations and averages the estimated treatment effects [1][2][6].

Because the first-stage model for Y(0) is estimated solely from untreated/not-yet-treated observations, **comparing actual pre-treatment outcomes to imputed counterfactuals gives a clean measure of differential pre-trends that is uncontaminated by treatment effects**. The fixed effects (unit and time) are estimated jointly using all pre-treatment periods from untreated observations [1][2][6].

**Implications for statistical power and pretesting bias:** By using all pre-treatment periods jointly to estimate fixed effects, the imputation approach can potentially improve the precision of the counterfactual model, thereby increasing power to detect violations—addressing Roth's concern (a). The pre-trend test is conducted separately from the post-treatment estimation, which helps avoid the issue where pre-test results contaminate the post-treatment inference [4][7].

**Sensitivity to violations in any single pre-treatment period:** Because all pre-treatment periods are used jointly to estimate α_i and λ_t, a violation in any one pre-treatment period can bias the estimated fixed effects and thus bias all treatment effect estimates. This is a vulnerability of the imputation approach not shared by methods that treat each pre-treatment period separately (like CS2021's pre-trend test which tests individual pre-treatment coefficients) [6].

### 3.4 Callaway and Sant'Anna (2021): How Conditional Parallel Trends Addresses Power Concerns

**The conditional parallel trends assumption:** Callaway and Sant'Anna (2021) provide a framework that allows for covariate adjustment to address non-parallel trends due to observed characteristics. The formal conditional parallel trends assumption (Assumption 4) states: for each g ∈ G and t ∈ {2,...,T} such that t ≥ g − δ, E[Y_t(0) − Y_{t−1}(0) | X, G_g=1] = E[Y_t(0) − Y_{t−1}(0) | X, C=1] a.s. [4][8].

**Addressing Roth's power concerns:** Roth's critique focuses on the low power of unconditional parallel trends tests. By allowing the researcher to incorporate covariates that predict treatment assignment or outcome dynamics, the conditional parallel trends assumption may be more plausible than the unconditional version. This means there are fewer violations to detect—addressing the power concern directly. Additionally, incorporating covariates can reduce residual variance, increasing the precision of pre-treatment estimates and thereby increasing power to detect violations [2][4][8].

**How reducing residual variance via covariate adjustment works:** The conditional parallel trends assumption requires that, after conditioning on covariates, the average outcomes for treated and comparison groups would have followed parallel paths in the absence of treatment. This is often more credible than unconditional parallel trends when treatment is related to observed characteristics. The doubly-robust estimator provides additional robustness to model misspecification: consistent estimation holds if either the propensity score model or the outcome regression model is correctly specified [2][8].

**Critique from Lee and Wooldridge (2023):** The standard Callaway-Sant'Anna estimator "uses only the period prior to treatment, reducing efficiency," and they suggest "a rolling transformation using averages of all pre-treatment periods and all not-yet-treated units as controls for improved efficiency" [1].

### 3.5 Goodman-Bacon (2021): Decomposition as Diagnostic

**How the decomposition reveals the source of contamination in pre-trends:** The decomposition theorem shows that "a causal interpretation of two-way fixed effects DD estimates requires both a parallel trends assumption and treatment effects that are constant over time" [4][5][6]. When treatment effects vary over time, the two-way fixed effects DD estimator compounds negative weights and may fail to identify a meaningful average treatment effect [4].

The key insight is that standard TWFE regressions implicitly use already-treated units as controls for later-treated units—"forbidden comparisons" [2][8]. When treatment effects change over time, these forbidden comparisons introduce bias because an early-treated unit's outcome includes its dynamic treatment effects, which contaminate the counterfactual for later-treated units [2][8]. This is the mechanism through which pre-treatment coefficients in TWFE event studies become contaminated.

### 3.6 Roth (2026) Critique of Asymmetric Event-Study Construction

Jonathan Roth's (2026) paper "Interpreting Event-Studies from Recent Difference-in-Differences Methods" (published in the *Japanese Economic Review*, Volume 77, Issue 2, pages 275–288; arXiv: 2401.12309) identifies a critical issue: the default event-study plots produced by modern heterogeneity-robust estimators are constructed asymmetrically for pre- and post-treatment periods, leading to potential misinterpretation [3][5][6].

**Key finding:** "The default plots produced by software for several of the most popular recent methods do not match those of traditional two-way fixed effects (TWFE) event-studies. These new methods construct the pre-treatment coefficients asymmetrically from the post-treatment coefficients" [5].

**The "kink" at treatment time:** Roth demonstrates through simulation that in non-staggered treatment settings with no actual treatment effect but violated parallel trends, traditional dynamic TWFE event-studies produce linear trends, whereas Callaway and Sant'Anna (CS) and Borusyak, Jaravel and Spiess (BJS) show kinks or jumps at treatment time. "A kink or jump at the time of treatment may arise even if there is no treatment effect and parallel trends is equally violated in all periods" [5].

**Mathematical explanation:**
- For CS: Post-treatment coefficients use a "long-difference" (comparing outcomes in period t to outcomes in the period just before treatment), while pre-treatment coefficients use a "short-gap" (comparing outcomes in adjacent periods). This asymmetry creates the visual kink.
- For BJS: "The BJS approach inherently uses all pre-treatment periods to estimate counterfactual outcomes, leading to an asymmetry between pre- and post-treatment estimates" [5].

**Implications for applied research:** "The typical heuristics for visual inference developed based on dynamic TWFE specifications will thus be misleading when applied to these new estimators" [5]. Roth recommends: (1) For CS, use "long-differences for the pre-treatment coefficients as well as the post-treatment coefficients (i.e., always using the period before treatment as the baseline)"; (2) For BJS, "putting the BJS pre-treatment estimates on a different plot from the post-treatment estimates to avoid making misleading visual inferences"; (3) "Supplementing event-studies with cohort-specific time series and balanced-event studies to address unit composition changes over relative time" [5].

**No threat to post-treatment validity:** "The discussion does not threaten the validity of the post-treatment estimates if the parallel trends assumption holds; rather, it highlights challenges in visualizing violations of parallel trends using conventional heuristics" [5].

---

## 4. Practical Implementation Details for Applied Researchers

### 4.1 Sun and Abraham (2021): Interaction-Weighted Estimator

#### Intuitive OLS Basis

The Sun and Abraham (2021) interaction-weighted estimator is built on a saturated OLS regression of outcomes on cohort×relative-time interaction terms with unit and time fixed effects:

$$Y_{it} = \alpha_i + \lambda_t + \sum_{c} \sum_{k} \beta_{c,k} \cdot \mathbf{1}\{\text{Cohort}=c\} \times \mathbf{1}\{\text{RelativeTime}=k\} + \varepsilon_{it}$$

The cohort-specific dynamic effects $\hat{\beta}_{c,k}$ are then aggregated within each relative period $k$ using cohort-share weights:

$$\hat{\nu}_k = \sum_{c} \hat{\beta}_{c,k} \times \frac{N_{c,k}}{\sum_c N_{c,k}}$$

where $N_{c,k}$ is the number of observations for cohort $c$ at relative time $k$ [6][10][41].

This makes it a "drop-in replacement" for traditional TWFE event-study regressions. Where you once wrote `feols(y ~ i(relative_time, ref = -1) | unit + year)`, you now write `feols(y ~ sunab(cohort, relative_time) | unit + year)`.

#### R Syntax — `fixest::sunab()`

```r
library(fixest)

# Basic event study — sunab() inside feols()
est <- feols(
  y ~ sunab(cohort_treated, year, ref.c = 0, ref.p = -1) | unit + year,
  data = df,
  cluster = ~unit
)

# Extract the aggregated event-study coefficients
summary(est, agg = "ATT")

# Event-study plot
iplot(est)

# Manually extract aggregated coefficients
aggregate(est, agg = "ATT")
```

The `sunab()` function arguments include: first argument (variable indicating the period each unit first becomes treated, 0 if never-treated), second argument (calendar time variable), `ref.c` (which cohort to use as reference), `ref.p` (which relative period to use as the omitted reference, default = -1), and `bin` (binning of cohort/period tails) [29].

#### Stata Syntax — `eventstudyinteract`

The Stata package `eventstudyinteract` by Liyang Sun implements the interaction weighted estimator [41].

```stata
* Installation
net install eventstudyinteract, from(https://raw.githubusercontent.com/lsun20/eventstudyinteract/master)
ssc install reghdfe

* Basic usage
gen rel_time = year - first_treat_year if first_treat_year != .
replace rel_time = -999 if first_treat_year == .  // never-treated code

eventstudyinteract y, cohort(first_treat_year) control_cohort(never_treated) 
    covariates(i.year) absorb(i.unit) vce(cluster unit)
```

Key options: `cohort(var)` indicates treatment cohort; `control_cohort(var)` specifies which cohorts serve as controls; `covariates(...)` adds covariates; `absorb(...)` specifies fixed effects; `vce(cluster var)` provides cluster-robust standard errors [30].

### 4.2 Borusyak et al. (2024): Imputation Estimator

#### Rationale for Untreated-Only Pre-Trend Tests

The imputation estimator works in two steps:

**Step 1 — First-stage model:** Estimate the untreated potential outcome $Y(0)$ using only untreated/not-yet-treated observations:
$$Y_{it}(0) = \alpha_i + \lambda_t + \varepsilon_{it}$$
This is estimated via TWFE on the subsample where $D_{it} = 0$.

**Step 2 — Imputation:** For treated observations, impute their counterfactual $Y(0)$ using the estimated $\hat{\alpha}_i + \hat{\lambda}_t$, and compute:
$$\hat{\tau}_{it} = Y_{it} - \hat{Y}_{it}(0)$$

**Why pre-trend tests use untreated-only observations:** Because the model for $Y(0)$ is estimated entirely from untreated/not-yet-treated units, comparing actual pre-treatment outcomes to imputed counterfactuals gives a clean measure of differential pre-trends that is uncontaminated by treatment effects. In contrast, TWFE pre-trend tests can be contaminated because the fixed effects absorb treatment effects from treated units.

**Sensitivity to violations in any single pre-treatment period:** Since all pre-treatment periods are used jointly to estimate $\alpha_i$ and $\lambda_t$, a violation in any one pre-treatment period can bias the estimated fixed effects and thus bias all treatment effect estimates.

#### Stata Syntax — `did_imputation`

```stata
* Installation
ssc install did_imputation, replace
ssc install event_plot, replace    // for plotting

* Basic event study
did_imputation y unit year first_treat_year, horizons(0/4) pretrends(1/4)

* With covariates
did_imputation y unit year first_treat_year, horizons(0/4) pretrends(1/4)
    covariates(x1 x2)

* Plot results
event_plot, default_look graph_opt(xtitle("Event time") ytitle("ATT"))
```

Key arguments: `y` (outcome variable), `unit` (unit identifier), `year` (time variable), `first_treat_year` (period of first treatment, 0 or missing = never-treated), `horizons(#/#)` (post-treatment relative periods), `pretrends(#/#)` (pre-treatment relative periods for testing) [31].

#### R Syntax — `didimputation` Package

```r
library(didimputation)

est <- did_imputation(
  data = df,
  yname = "y",
  gname = "first_treat_year",   # treatment cohort
  tname = "year",               # time variable
  idname = "unit",              # unit identifier
  first_stage = ~ 0 | unit + year,   # fixed effect specification
  horizons = 0:4,               # post-treatment periods
  pretrends = -4:-1,            # pre-treatment periods for test
  cluster_var = "unit"          # clustering variable
)
```

### 4.3 Callaway and Sant'Anna (2021): Doubly-Robust Estimator

#### Group-Time ATT Framework

The fundamental parameter is $ATT(g, t)$: the average treatment effect at calendar time $t$ for units first treated in cohort $g$:

$$ATT(g,t) = \mathbb{E}[Y_t(1) - Y_t(0) \mid G = g]$$

Under no-anticipation and parallel trends assumptions, ATT(g,t) can be nonparametrically identified using doubly robust estimators combining outcome regression and inverse probability weighting [2][5][19][20].

#### Four Aggregation Types in `aggte()`

| Type | Meaning | Code |
|------|---------|------|
| **Simple** | Single overall ATT, equally weighted across (g,t) pairs | `type = "simple"` |
| **Group** | ATT by cohort, averaged over post-treatment periods | `type = "group"` |
| **Calendar time** | ATT by calendar year, averaged across cohorts active at that time | `type = "calendar"` |
| **Dynamic / Event study** | ATT by event time (relative to treatment) | `type = "dynamic"` |

#### Varying vs. Universal Base Periods

**Critical distinction — source of major software discrepancies:**

In the `did` R package, the default is a varying base period where, in pre-treatment periods, the base is the immediately preceding period. With a varying base period, pre-treatment reported effects are 'pseudo-ATTs' estimating what the treatment effect would have been if treatment started in that period. Using a universal base period, pre-treatment estimates illustrate outcome trends over time but are not treatment effect parameters.

**Why it matters:** With a varying base period and long-running linear pre-trends, the `did` package can incorrectly signal no violation of parallel trends, while a universal base period correctly reveals the violation. The new `base_period` argument in `att_gt` (version 2.1) allows users to choose between 'varying' (default) and 'universal' base periods.

```r
# R (did package v2.1+)
att_gt_out <- att_gt(
  data = df, 
  yname = "y", 
  gname = "first_treat", 
  tname = "year", 
  idname = "unit",
  base_period = "universal"  # CHANGES DEFAULT FROM "varying"
)
```

#### R Syntax — `did` Package

```r
library(did)

# Step 1: Estimate group-time ATTs
att_gt_out <- att_gt(
  yname = "y",                # outcome
  gname = "first_treat",      # first-treatment period (0 = never-treated)
  tname = "year",             # time variable
  idname = "unit",            # unit identifier
  data = df,
  xformla = ~ x1 + x2,        # covariates (optional)
  est_method = "dr",          # "dr" (default), "ipw", or "reg"
  control_group = "nevertreated",  # "nevertreated" or "notyettreated"
  base_period = "universal",  # "varying" (default) or "universal"
  clustervars = "unit",
  biters = 1000,
  cband = TRUE
)

# Step 2: Aggregate to event study
agg_es <- aggte(att_gt_out, type = "dynamic", min_e = -5, max_e = 5)
summary(agg_es)
ggdid(agg_es)
```

#### Stata Syntax — `csdid` Package

```stata
* Installation
ssc install csdid, replace
ssc install drdid, replace

* Basic estimation
csdid y, ivar(unit) time(year) gvar(first_treat)

* With universal base period (long2 option)
csdid y, ivar(unit) time(year) gvar(first_treat) long2

* With covariates
csdid y x1 x2, ivar(unit) time(year) gvar(first_treat)

* Post-estimation: event study
csdid_estat event, window(-5 5)
csdid_plot

* Pre-trend test
csdid_estat pret
```

### 4.4 Goodman-Bacon (2021): Decomposition Diagnostic

#### Four Types of 2×2 Comparisons

When there are never-treated units, the decomposition identifies four types:

| Comparison | Description | "Clean"? |
|-----------|-------------|----------|
| **Earlier-treated vs. Never-treated** | Compare early adopters vs. units that never adopt | ✅ Clean |
| **Later-treated vs. Never-treated** | Compare late adopters vs. units that never adopt | ✅ Clean |
| **Earlier-treated vs. Later-treated (before later-treated is treated)** | Use later-treated as controls for earlier-treated while later-treated are still untreated | ✅ Clean (if parallel trends holds) |
| **Earlier-treated vs. Later-treated (after later-treated is treated)** | Use later-treated as controls for earlier-treated after later-treated adopt treatment | ❌ Forbidden comparison |

#### Interpreting the Output

Key quantities to examine:
1. **Weight on forbidden comparisons** — If this is large (>20%), TWFE estimates are likely contaminated
2. **Scatter plot of 2×2 estimates vs. weights** — Shows which comparisons drive the TWFE estimate
3. **Heterogeneity across comparison types** — If clean and forbidden comparisons give very different estimates, there's a sign of treatment-effect dynamics/heterogeneity

#### R Syntax — `bacondecomp`

```r
library(bacondecomp)

bacon_out <- bacon(
  y ~ treat,                # outcome ~ treatment indicator
  data = df,
  id_var = "unit",
  time_var = "year"
)

summary(bacon_out)

# Plot
library(ggplot2)
ggplot(bacon_out$two_by_twos, aes(x = weight, y = estimate, color = type)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(x = "Weight", y = "2×2 DiD Estimate", color = "Comparison Type") +
  theme_minimal()
```

#### Stata Syntax — `bacondecomp`

```stata
ssc install bacondecomp, replace

bacondecomp y treat, stub(Bacon_)

* With time-varying covariates
bacondecomp y treat x1 x2, stub(Bacon_)

* Plot
bacondecomp y treat, stub(Bacon_) graph
```

### 4.5 Known Software Discrepancies

#### 4.5.1 Scott Cunningham (2026) Cross-Package Audit

Scott Cunningham of Baylor University conducted a rigorous cross-package audit (March 2026) comparing six software implementations of the Callaway & Sant'Anna (2021) doubly robust estimator across R (`did` package), Stata (`csdid` and `csdid2`), and Python (`diff-diff` and `differences`) [41][1].

**Key findings:**
- **With no covariates, all six packages agreed.** Adding covariates caused packages to diverge [1].
- **The R `did` package silently returned zeros for numerically singular cases without warnings** [1].
- **Stata's `csdid` estimated ATTs roughly twice as large as other packages** due to inverse probability tilting's handling of near-separation in propensity score models [1].
- Package choice was characterized as "a substantive decision determining whether your ATT is 0.00, 0.45, 1.15, or 2.38" [1].
- **44% of total variation** in ATT estimates arises from interaction effects between package choice and specification [1].
- **Package choice accounts for about 16% of ATT estimate variation** [1].
- **Recommendations:** Standardize covariates via z-scoring to eliminate numerical singularity; report software package and version used; conduct baseline checks without covariates to ensure package agreement [1].

#### 4.5.2 Deb, Norton, Wooldridge & Zabel (2024) Aggregation Weights Critique

The working paper "Aggregating Average Treatment Effects on the Treated in Difference-in-Differences Models" (NBER WP 34331, October 2025) investigates the aggregation of ATET in DiD models [19][2][3].

**Key finding:** "The standard software used to estimate Callaway and Sant'Anna's method uses weights that include the number of observations in the reference pre-period instead of only the number of observations in the treated periods" [2][3].

**How it affects applied results:** Callaway and Sant'Anna's software-generated ATET estimates differ by about 16% compared to weights based only on treated observations [2][3]. "The weighting matters more when there is more heterogeneity in the treatment effects, more heterogeneity in the number of observations, and when those heterogeneities are correlated" [2][3].

**Implications:** "Given the enormous influence and popularity of the Callaway and Sant'Anna method and the common use of Stata software, we are concerned that many published results have been calculated with a formula that is not what the researchers intended" [2][3].

#### 4.5.3 R vs. Stata Differences in Default Base Periods

The issue of base period choices was highlighted on Economics Stack Exchange and confirmed by multiple sources [9][10][11].

**Default choices:**
- R's `did` package defaults to `base_period = "varying"` — the base period changes for each group-time comparison, differencing out any common trend between the treated unit and the comparison unit
- Stata's `csdid` package by default uses a "short-gap-varying base" [9]

**How `long2` maps to `base_period = "universal"`:**
- Stata's `long2` option corresponds to R's `base_period = "universal"` option [9]
- With `long2`, all comparisons are made relative to a fixed early period, so linear pretrends are not differenced out

**Why this matters for pre-trend testing:** "When you have long-running linear pretrends, using a varying base period differences them out" [9]. This means with default settings, long-running linear pretrends are masked, and the pre-trend test appears to show no violation when in fact there is a clear violation.

### 4.6 Concrete Recommendations for Reporting Workflows

#### 4.6.1 Which Estimator to Use as Primary Specification

The emerging norm is that a heterogeneity-robust estimator should be reported as the primary specification. Sant'Anna (2024) recommends: "Under PT, when treatment effects are heterogeneous/dynamic, β does not recover an easy-to-interpret parameter... the solution is to separate identification, aggregation, and estimation/inference steps" [1][21]. The choice between estimators should be guided by the specific features of the research setting:

- **Use Sun-Abraham** when event-study plots are the primary visualization and a "drop-in replacement" for TWFE is desired
- **Use Callaway-Sant'Anna** when covariate adjustment is important for plausibility of parallel trends
- **Use Borusyak et al. imputation** when efficiency is a concern and the design has many pre-treatment periods
- **Use stacked DiD** when transparency and simplicity of communication are priorities

#### 4.6.2 Robustness Checks

Multiple estimators (typically 2-3) should be reported to show that results are not sensitive to estimator choice. The Wang, Hamad & White (2024) simulation showed that "heterogeneous treatment effects-robust estimators exhibited more robust results across all scenarios under the parallel trends assumption" [23].

**Recommended battery:**
1. **Goodman-Bacon decomposition** in appendix as diagnostic
2. **Two main robust estimators** as primary specifications
3. **Alternative comparison groups** (never-treated vs. not-yet-treated)
4. **Different anticipation windows** (varying assumed anticipation periods)
5. **Sample restrictions** (dropping specific cohorts, trimming event time distributions)

#### 4.6.3 How to Construct Event-Study Plots

Following Roth's (2026) guidance [3][5][6]:

1. For Callaway-Sant'Anna: Use "long-differences for the pre-treatment coefficients as well as the post-treatment coefficients (i.e., always using the period before treatment as the baseline)"
2. For Borusyak et al.: "Putting the BJS pre-treatment estimates on a different plot from the post-treatment estimates to avoid making misleading visual inferences"
3. Supplement with "cohort-specific time series and balanced-event studies to address unit composition changes over relative time"
4. Scott Cunningham's recommendation: "When you're plotting event studies, you should use long differences, not short gap" [41]

#### 4.6.4 How to Report Pre-Trend Tests (Avoiding Strict Inference Gates)

**The evidence is clear: pre-trend tests should NOT be used as strict inference gates.** Roth (2022, AER: Insights) provides the definitive evidence: confidence interval coverage can fall to as low as 24% for nominal 95% confidence intervals after pre-testing [4][5][18].

**Recommended approaches:**

1. **Power analyses:** Use the `pretrends` R package (and corresponding Stata package) to "assess and report the power of pretests against plausible violations" [18]. The `slope_for_power()` function calculates the "slope of a linear violation of parallel trends that a pre-trends test would detect a specified fraction of the time" [18].

2. **Equivalence tests:** The `EquiTrends` R package performs equivalence testing "to test if pre-treatment trends in the treated group are 'equivalent' to those in the control group" [42]. Three tests are available: `maxEquivTest` (based on maximum absolute placebo coefficient), `meanEquivTest` (based on mean of placebo coefficients), and `rmsEquivTest` (based on root mean squared placebo coefficients).

3. **Honest confidence intervals:** The Rambachan and Roth (2023) "HonestDiD" approach provides "robust inference and sensitivity analysis for differences-in-differences and event study designs" [13]. Instead of asking "Do parallel trends hold?" it asks "How large would violations of parallel trends need to be before my conclusion changes?" The **breakdown value** is "a single number that tells the reader exactly how robust the result is" [14].

#### 4.6.5 Sensitivity Analyses (HonestDiD, pretrends, EquiTrends)

**HonestDiD (Rambachan & Roth, 2023):**

The HonestDiD method imposes restrictions on how different post-treatment violations of parallel trends can be from pre-treatment differences [12][13].

```r
library(HonestDiD)

# Relative magnitudes restriction
results_RM <- computeConditionalCS_DeltaRM(
  betahat = event_study_coefs,
  sigma = vcov_matrix,
  numPrePeriods = 5,
  numPostPeriods = 3,
  Mbar = 1,
  alpha = 0.05
)

# Create sensitivity plot
createSensitivityPlot(results_RM, results_RM$gridValues)

# Compute breakdown value
computeLowerBound_M(
  betahat = event_study_coefs,
  sigma = vcov_matrix,
  numPrePeriods = 5,
  alpha = 0.05
)
```

Key parameters:
- **Mbar**: A tuning parameter governing the allowed departure from parallel trends. Mbar = 2 means "allowing post-treatment violations twice as large as the pre-treatment difference"
- **Breakdown value**: The smallest value of Mbar at which the treatment effect is no longer statistically significant

**pretrends (Roth, 2022):**

```r
library(pretrends)

# Find the minimal detectable linear slope at 80% power
detectable_slope <- slope_for_power(
  betahat = event_study_coefs,
  sigma = vcov_matrix,
  numPrePeriods = 5,
  numPostPeriods = 3,
  power = 0.80
)

# Power analysis with hypothesized violation
results <- pretrends(
  betahat = event_study_coefs,
  sigma = vcov_matrix,
  delta_tilde = hypothesized_violation,
  numPrePeriods = 5,
  numPostPeriods = 3
)
```

**EquiTrends (Dette & Schumann, 2024):**

```r
library(EquiTrends)

# Maximum equivalence test (most conservative)
max_test <- maxEquivTest(
  data = my_data,
  Y = "outcome",
  ID = "unit_id",
  Time = "period",
  D = "treatment",
  equiv_threshold = 0.1,
  pretest_periods = 1:5
)
```

### 4.7 Anticipation Effects: Implementation Details

**Callaway and Sant'Anna (2021)** provide the most elaborate treatment of anticipation through a formal parameter, δ (delta), in their "Limited Treatment Anticipation" assumption [2][36]. The `did` R package's `att_gt()` function defaults to `anticipation = 0`. When a researcher specifies `anticipation = 1`, the base period shifts backward [36].

**Critical nuance:** A documented issue on Economics Stack Exchange reveals that the `did` R package defaults to a "varying" reference period, which can obscure long-running linear pre-trends. The problem was "resolved when setting the reference period as 'universal' instead of 'varying', which is a problem when you have long-running linear pretrends" [9].

**Stata implementation differences:** "CSDID uses T-1 as the base period by default. You cannot choose other periods (for example allowing for anticipation), except by changing gvar manually" [11].

**Sun and Abraham (2021)** incorporate no-anticipation as a core identifying condition but do **not** include an explicit anticipation parameter [41]. In the `fixest::sunab()` function, there is no `anticipation` argument. To allow for anticipation, a researcher would include pre-treatment relative period dummies and interpret them as anticipation effects.

**Borusyak et al. (2024)** handle anticipation through a formal "no anticipation" assumption (Assumption 2). The key implementation mechanism is the `shift()` parameter in Stata's `did_imputation` command, which "specify to allow for anticipation effects. The command will pretend that treatment happened `shift` periods earlier for each unit" [31].

**Goodman-Bacon (2021)** does **not** explicitly model or allow for anticipation effects [32].

### 4.8 Weighting Schemes in Aggregation

**Callaway and Sant'Anna (2021):** The `aggte()` function provides four aggregation types: `"simple"` (weighted average of all group-time ATTs with weights proportional to group size), `"dynamic"` (average effects across different lengths of exposure), `"group"` (default, average across cohorts), and `"calendar"` (average across time periods) [26].

**Critical finding from Deb, Norton, Wooldridge & Zabel (2024):** "The software uses weights that include the number of observations in the reference pre-period instead of only the number of observations in the treated periods" [19][2][3]. This leads to "a 16.2% difference in the overall ATET estimate" in their simulation example [2].

**Sun and Abraham (2021):** The IW estimator aggregates cohort-specific effects using **cohort shares as weights**. The weight for each (cohort, event time) pair is "the share of units in that event time who belong to that cohort" [41]. The non-negativity guarantee is the key advantage—cohort-share weights are non-negative and sum to one.

**Borusyak et al. (2024):** The imputation estimator uses **precision-based weights**—"applying any weights to the imputed causal effects yields the efficient estimator" [11][12][13][14]. The precision-based weighting scheme is a key source of claimed efficiency gains (30-360% reduction in standard deviation).

**Goodman-Bacon (2021):** The decomposition uses **variance-based weights** proportional to group sizes and the variance of the treatment dummy within each 2×2 pair [8][9].

---

## 5. Conclusion: Emerging Norms and Practical Recommendations

The staggered DiD literature has undergone a remarkable transformation since Goodman-Bacon's (2021) decomposition revealed the vulnerabilities of TWFE estimators. Three main approaches have emerged—Callaway and Sant'Anna's (2021) group-time ATTs with doubly-robust estimation, Sun and Abraham's (2021) interaction-weighted estimator, and Borusyak et al.'s (2024) imputation-based estimator—each offering distinct advantages in handling heterogeneous treatment effects and dynamic treatment timing.

**Key findings from this deepened analysis:**

1. **Adoption patterns show clear field-specific preferences.** Callaway & Sant'Anna dominates in health economics due to its covariate adjustment capabilities. Labor economics shows more diverse adoption, with stacked DiD and Sun-Abraham alongside CS and BJS. The field has moved through three phases—early adoption (2020-2021), consolidation (2022-2023), and standardization (2024+)—with robust estimators now expected as the default.

2. **Implementation details matter substantially.** The choice of software package (R vs. Stata), default comparison group, anticipation parameter, base period selection, and aggregation weights can materially affect results. The Scott Cunningham (2026) cross-package audit found that package choice accounts for 16% of variation in estimates [1][41]. The Deb et al. (2024) analysis found that aggregation weights can produce 16% differences in ATET estimates [2][3].

3. **Each estimator's features are targeted solutions to specific methodological problems.** Sun-Abraham's clean pre-treatment coefficients address contamination of pre-trend tests (Roth concern c). Borusyak et al.'s untreated-only estimation improves power for pre-trend testing (Roth concern a). Callaway-Sant'Anna's conditional parallel trends makes the identifying assumption more plausible and reduces residual variance. However, none of these estimators directly resolves Roth's concern (b) about pretesting bias—the bias that arises from conditioning on passing a pre-test.

4. **Pre-trend testing requires fundamental rethinking.** Roth's (2022) critique demonstrates that pre-trend tests should not be used as strict inference gates [4][5][18]. Instead, researchers should conduct power analyses (using `pretrends`), equivalence tests (using `EquiTrends`), and sensitivity analyses (using `HonestDiD`). Roth's (2026) paper further warns that default event-study plots from modern estimators are constructed asymmetrically and should not be interpreted using TWFE heuristics [3][5][6].

5. **Replication evidence confirms the importance of robust estimators.** Baker, Larcker & Wang (2022) demonstrated that TWFE biases can lead to spurious significance and that alternative estimators often contradict original claims [6][7][9][15]. Wang, Hamad & White (2024) confirmed that heterogeneity-robust estimators perform well across simulation scenarios [23].

**For applied researchers, the emerging best practice is:**
- Use a heterogeneity-robust estimator as the primary specification
- Report 2-3 estimator specifications as robustness checks
- Include Goodman-Bacon decomposition diagnostics
- Conduct power analysis for pre-trend tests using `pretrends`
- Report breakdown values from `HonestDiD` sensitivity analysis
- Use equivalence tests from `EquiTrends` when possible
- Understand and document software implementation choices
- Standardize covariates to avoid numerical issues
- Use universal base periods (or `long2` option in Stata) for valid pre-trend testing
- Interpret event-study plots with awareness of asymmetry (Roth 2026)
- Report which software package and version was used

---

## Sources

[1] Cunningham, S. (2026). "Six Packages, One Estimator." Scott Cunningham, Baylor University: https://www.scunning.com/files/package_audit_r4.pdf

[2] Deb, P., Norton, E.C., Wooldridge, J.M., & Zabel, J.E. (2025). "Aggregating Average Treatment Effects on the Treated in Difference-in-Differences Models." NBER WP 34331: https://www.nber.org/system/files/working_papers/w34331/w34331.pdf

[3] Deb, P., Norton, E.C., Wooldridge, J.M., & Zabel, J.E. (2025). "Aggregating Average Treatment Effects on the Treated in Difference-in-Differences Models." SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5568722

[4] Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends." AER: Insights, 4(3), 305-322. DOI: 10.1257/aeri.20210236: https://www.aeaweb.org/articles?id=10.1257/aeri.20210236

[5] Roth, J. (2022). Full paper PDF: https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf

[6] Roth, J. (2026). "Interpreting Event-Studies from Recent DiD Methods." Japanese Economic Review, 77(2), 275-288. arXiv:2401.12309: https://arxiv.org/abs/2401.12309

[7] Roth, J. (2026). Full paper PDF: https://www.jonathandroth.com/assets/files/HetEventStudies.pdf

[8] Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with Multiple Time Periods." Journal of Econometrics, 225(2), 200-230. DOI: 10.1016/j.jeconom.2020.12.001: https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948

[9] Callaway, B. & Sant'Anna, P.H.C. (2021). Full text PDF: https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[10] Sun, L. & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." Journal of Econometrics, 225(2), 175-199. DOI: 10.1016/j.jeconom.2020.09.006: https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html

[11] Sun, L. & Abraham, S. (2021). Full text PDF: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Paper2023/D3--Sun-2020-Estimating-dynamic-treatment-effects-in-event-studies-with-heterogeneous-treatment-effects.pdf

[12] Borusyak, K., Jaravel, X., & Spiess, J. (2024). "Revisiting Event-Study Designs: Robust and Efficient Estimation." Review of Economic Studies, 91(6), 3253-3285. DOI: 10.1093/restud/rdae007: https://academic.oup.com/restud/article/91/6/3253/7601390

[13] Borusyak, K., Jaravel, X., & Spiess, J. (2024). arXiv working paper: https://arxiv.org/abs/2108.12419

[14] Goodman-Bacon, A. (2021). "Difference-in-differences with variation in treatment timing." Journal of Econometrics, 225(2), 254-277. DOI: 10.1016/j.jeconom.2021.03.014: https://www.sciencedirect.com/science/article/abs/pii/S0304407621001445

[15] Goodman-Bacon, A. (2021). Full text PDF: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Goodman-Bacon_2021_Difference-in-differences_with_variation_in_treatment_timing.pdf

[16] Rambachan, A. & Roth, J. (2023). "A More Credible Approach to Parallel Trends." Review of Economic Studies, 90(5), 2555-2591. DOI: 10.1093/restud/rdad018: https://academic.oup.com/restud/article/90/5/2555/6950347

[17] Roth, J., Sant'Anna, P.H.C., Bilinski, A., & Poe, J. (2023). "What's Trending in Difference-in-Differences?" Journal of Econometrics. https://www.jonathandroth.com/assets/files/DiD_Review_Paper.pdf

[18] Baker, A.C., Larcker, D.F., & Wang, C.C.Y. (2022). "How Much Should We Trust Staggered Difference-In-Differences Estimates?" Journal of Financial Economics, 144(2), 370-395. DOI: 10.1016/j.jfineco.2022.03.004: https://www.sciencedirect.com/science/article/pii/S0304405X22000204

[19] Baker, A.C., Larcker, D.F., & Wang, C.C.Y. (2022). Full text PDF: https://dash.harvard.edu/bitstreams/fe5cc546-b8e9-4814-8d3a-06af82c77845/download

[20] Goldsmith-Pinkham, P. (2026). "Tracking the Credibility Revolution across Fields." NBER WP 35051: https://www.nber.org/papers/w35051

[21] Goldsmith-Pinkham, P. (2026). Full text PDF: https://paulgp.com/econlit-pipeline/paper-figures/goldsmith-pinkham-credibility-revolution.pdf

[22] Lee, Y.J. & Wooldridge, J. (2023). "A Simple Transformation Approach to DiD": https://www.econ.queensu.ca/sites/econ.queensu.ca/files/Lee_Wooldridge_20230720.pdf

[23] Sant'Anna, P.H.C. (2024). "Modern Difference-in-Differences" - NABE TEC Presentation: https://psantanna.com/DiD/NABE_202410.pdf

[24] Callaway, B. (2022). "Difference-in-Differences for Policy Evaluation": https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf

[25] Wang, Hamad & White (2024). "Advances in DiD Methods." Epidemiology: https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929

[26] Cunningham, S. (2021). "Causal Inference: The Mixtape" - Staggered Adoption DiD: https://causalinf.substack.com/p/waiting-for-event-studies-a-play

[27] Gardner, J. (2021). "Two-stage differences in differences": https://arxiv.org/abs/2207.05943

[28] did R package documentation: https://bcallaway11.github.io/did/reference/att_gt.html

[29] csdid Stata slides: https://www.stata.com/meeting/us21/slides/US21_SantAnna.pdf

[30] Stata xthdidregress manual: https://www.stata.com/manuals/causalxthdidregress.pdf

[31] fixest sunab documentation: https://lrberge.github.io/fixest/reference/sunab.html

[32] eventstudyinteract Stata help file: http://fmwww.bc.edu/repec/bocode/e/eventstudyinteract.sthlp

[33] did_imputation Stata/GitHub: https://github.com/borusyak/did_imputation

[34] bacondecomp R/Stata package: https://github.com/evanjflack/bacondecomp

[35] MetricGate - Sun-Abraham estimator: https://metricgate.com/docs/sun-abraham-estimator

[36] GitHub fixest Issue #287: https://github.com/lrberge/fixest/issues/287

[37] Asjad Naqvi did_imputation documentation: https://asjadnaqvi.github.io/DiD/docs/did_imputation

[38] Chen Xing - Notes on Callaway & Sant'Anna (2021): https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did

[39] de Chaisemartin, C. & D'Haultfœuille, X. (2020). "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects." American Economic Review, 110(9), 2964-2996. DOI: 10.1257/aer.20181169: https://www.aeaweb.org/articles?id=10.1257/aer.20181169

[40] Roth, J. & Sant'Anna, P.H.C. (2023). "Efficient Estimation for Staggered Rollout Designs." Journal of Political Economy: Microeconomics. DOI: 10.1086/726596: https://www.journals.uchicago.edu/doi/10.1086/726596

[41] Cengiz, D., Dube, A., Lindner, A., & Zipperer, B. (2019). "The Effect of Minimum Wages on Low-Wage Jobs." Quarterly Journal of Economics, 134(3), 1405-1454. DOI: 10.1093/qje/qjz014: https://academic.oup.com/qje/article/134/3/1405/5420451

[42] Miller, S. (2023). "An Introductory Guide to Event Study Models." Journal of Economic Perspectives, 37(2), 203-230. DOI: 10.1257/jep.37.2.203: https://www.aeaweb.org/articles?id=10.1257/jep.37.2.203

[43] Bos, T. (2024). "EquiTrends" R package: https://cran.r-project.org/package=EquiTrends

[44] Stata CSDID Version 1.6 documentation (Fernando Rios-Avila): https://friosavila.github.io/playingwithstata/raw_articles/csdid_stata.html

[45] GCcollab Wiki - "The problem — Treatment effect heterogeneity": https://wiki.gccollab.ca/images/7/73/Handout_-_Treatment_effect_heterogeneity.pdf

[46] Sant'Anna, P.H.C. & Zhao, J. (2020). "Doubly Robust Difference-in-Differences Estimators." Journal of Econometrics, 219(1), 101-122. DOI: 10.1016/j.jeconom.2020.06.00: https://ideas.repec.org/a/eee/econom/v219y2020i1p101-122.html

[47] Gardner, J., Thakral, N., Tô, L.T., & Yap, L. (2024). "Two-Stage Differences in Differences." Working Paper: https://www.bu.edu/econ/files/2024/07/two-stage-differences-in-differences.pdf

[48] Dube, A., Girardi, D., Jordà, Ò., & Taylor, A.M. (2023). "A Local Projections Approach to Difference-in-Differences Event Studies." NBER WP 31184: https://www.nber.org/system/files/working_papers/w31184/w31184.pdf

[49] Freedman, S., Hollingsworth, A., Simon, K., Wing, C., & Yozwiak, M. (2024). "Designing Difference-in-Difference Studies with Staggered Treatment Adoption." Annual Review of Public Health, 45. DOI: 10.1146/annurev-publhealth-061022-050825: https://www.annualreviews.org/content/journals/10.1146/annurev-publhealth-061022-050825

[50] Dette, H. & Schumann, M. (2024). "Testing for Equivalence of Pre-Trends in Difference-in-Differences Estimation." Journal of Business & Economic Statistics, 42(4), 1289-1301. DOI: 10.1080/07350015.2024.2308121