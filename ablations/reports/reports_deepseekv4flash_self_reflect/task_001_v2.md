# Methodological Tensions in Staggered Difference-in-Differences Estimation: A Comprehensive Revised Analysis

## 1. Introduction

This report provides a substantially revised and expanded analysis of methodological tensions in staggered Difference-in-Differences (DiD) estimation, building on recent developments through May 2026. The report addresses nine specific dimensions identified in the research brief: (1) deeper engagement with the Rambachan and Roth (2023) HonestDiD approach and its interaction with each major estimator; (2) the practical significance of conditional vs. unconditional parallel trends; (3) systematic quantitative evidence on adoption patterns in top-5 economics journals; (4) critical evaluation of efficiency comparisons; (5) anticipation effects across applied domains; (6) the role of the Goodman-Bacon decomposition in current practice; (7) practical implementation challenges; (8) the Borusyak et al. efficiency advantage controversy; and (9) updated practitioner recommendations.

The analysis draws extensively on primary sources including journal publications, package documentation, replication archives, and systematic surveys of empirical practice. The report prioritizes evidence from the *American Economic Review*, *Quarterly Journal of Economics*, *Journal of Political Economy*, *Econometrica*, *Review of Economic Studies*, *Journal of Econometrics*, and the *Journal of Economic Literature*.

---

## 2. HonestDiD Sensitivity Analysis and Its Interaction with Staggered DiD Estimators

### 2.1 The Rambachan and Roth (2023) Framework

Rambachan and Roth (2023), published in the *Review of Economic Studies* (Volume 90, Issue 5, pages 2555–2591), propose a formal sensitivity analysis framework for DiD and event-study designs [1]. Rather than asking "Do parallel trends hold?" the framework asks "How large would violations of parallel trends need to be before my conclusion changes?"

The framework introduces two key families of restrictions on the violation of parallel trends, parameterized by M and M̄:

**Smoothness Restrictions (ΔSD) — The M Parameter:** The M parameter bounds the curvature of the trend deviation by placing limits on the magnitude of second differences—how much the "acceleration" of the differential trend can change between consecutive periods [1][2]. When M = 0, this enforces that violations follow a linear trend (linear extrapolation of pre-trends). As M increases, the trend can become more non-linear post-treatment, and confidence intervals widen accordingly.

**Relative Magnitude Restrictions (ΔRM) — The M̄ Parameter:** The M̄ parameter caps post-treatment trend deviations as a multiple of the maximum observed pre-treatment deviation [1][3]. When M̄ = 0, this enforces exact parallel trends post-treatment. M̄ = 1 means post-treatment violations can be at most as large as the worst pre-treatment violation. M̄ = 2 allows violations up to twice as large as the worst pre-treatment deviation.

**The Breakdown Value:** The single most informative statistic produced by HonestDiD is the "breakdown value"—the smallest M (or M̄) at which the robust confidence interval first includes zero [2][3]. A larger breakdown value indicates greater robustness to violations of parallel trends.

### 2.2 Integration of HonestDiD with Each Major Estimator

The HonestDiD framework operates on top of staggered DiD estimators, not as a replacement [1][4]. The workflow is: (1) estimate event-study coefficients using a staggered DiD estimator; (2) extract these coefficients and their variance-covariance matrix; (3) apply HonestDiD sensitivity analysis. The framework officially supports integration with Callaway and Sant'Anna (via the `did` R package), Sun and Abraham (via the `fixest` R package), and the imputation estimator (via Python `diff-diff` package) [4][5].

#### 2.2.1 HonestDiD Applied to Callaway and Sant'Anna (2021)

The Callaway and Sant'Anna (CS) estimator provides the cleanest baseline for HonestDiD sensitivity analysis among all estimators [6][7]. Because CS avoids "forbidden comparisons" by using only never-treated or not-yet-treated units as controls, its point estimates are not contaminated by negative weighting from time-varying treatment effects [8]. When HonestDiD is applied to CS estimates, the resulting breakdown values reflect genuine sensitivity to parallel trends violations, not contamination from estimator misspecification.

In the principal published application—Medicaid expansion under the ACA using American Community Survey data—CS estimates combined with HonestDiD produced breakdown values of approximately 1.5–2 under relative magnitude restrictions [2][9]. This means the result that Medicaid expansion increased insurance coverage holds unless post-treatment violations of parallel trends are 1.5–2 times larger than the maximum pre-treatment violation. Under smoothness restrictions (ΔSD), the breakdown values were around 0.015–0.02, indicating the results are robust to modest non-linear departures from parallel trends [9].

For CS, the M and M̄ parameters bound deviations from *conditional* parallel trends when covariates are included. This is particularly valuable because CS allows parallel trends to hold conditional on observed covariates through its doubly-robust framework [8].

#### 2.2.2 HonestDiD Applied to Sun and Abraham (2021)

HonestDiD integrates directly with the Sun and Abraham (SA) interaction-weighted estimator [4][5]. Because SA similarly avoids "bad comparisons" by estimating cohort-specific effects and aggregating with non-negative cohort-share weights, its sensitivity characteristics are generally similar to CS [10]. The HonestDiD package accepts event-study coefficients from SA estimation (via `fixest::sunab`) and performs sensitivity analysis on these estimates.

An important finding from the HonestDiD Interactive Lab confirms that "TWFE (full panel) and csdid (staggered) agree. Both produce breakdown ~1.5–2" when applied to the same data with a single treatment cohort [2]. This suggests SA and CS produce similar HonestDiD breakdown values when applied to the same data, as both avoid the contamination problems of TWFE.

#### 2.2.3 HonestDiD Applied to Borusyak, Jaravel, and Spiess (2024)

The Borusyak et al. (BJS) imputation estimator presents unique considerations for HonestDiD sensitivity analysis. The BJS estimator is "most efficient under homogeneity" but "assumes treatment effects are homogeneous across cohorts and time" in its baseline specification [11]. This matters for HonestDiD in several ways:

When treatment effects are truly homogeneous, BJS yields more precise estimates with narrower standard errors [11]. When HonestDiD is applied, the narrower baseline confidence intervals produce higher breakdown values (more robustness) because larger violations are needed to push the tighter interval to include zero.

However, when treatment effects are heterogeneous (the more realistic case in many applications), BJS estimates can be biased [12]. The "Advances in Difference-in-differences Methods" review found that among heterogeneity-robust estimators, "Callaway and Sant'Anna and Sun and Abraham estimators outperformed Borusyak et al and Wooldridge" in simulation settings with treatment effect heterogeneity [12]. This bias can manifest as different point estimates that, when fed into HonestDiD, produce different breakdown values.

A Stack Overflow discussion reported that "the Borusyak estimates are substantially higher than any of those [CS, SA, de Chaisemartin] (between 50% and 100%)" in data with "substantial treatment effect heterogeneity" [13]. This implies HonestDiD applied to BJS estimates may produce misleadingly high or low breakdown values depending on the direction of bias from effect heterogeneity.

#### 2.2.4 HonestDiD and the Goodman-Bacon Decomposition

Goodman-Bacon (2021) is primarily a diagnostic decomposition tool rather than an estimator that directly feeds into HonestDiD [14]. However, the decomposition reveals why HonestDiD applied to TWFE estimates may be misleading. Because TWFE estimates are weighted averages of multiple 2×2 comparisons—including "forbidden comparisons" that can produce negatively weighted estimates—the baseline estimate can itself be contaminated [14][15]. As Baker, Larcker, and Wang (2022) demonstrated, "TWFE down-weights some of the earlier- vs. later-treated comparisons and up-weights some of the later- vs. earlier-treated comparisons, thereby increasing the influence of the potentially problematic 2x2s" [15]. This highlights the importance of applying HonestDiD to heterogeneity-robust estimators (CS, SA, BJS) rather than to TWFE estimates directly.

### 2.3 Differential Sensitivity Patterns Across Estimators: Summary

| Estimator | Key Assumption | HonestDiD M/M̄ Interpretation | Sensitivity Pattern |
|-----------|---------------|-------------------------------|-------------------|
| Callaway & Sant'Anna (2021) | Conditional parallel trends, no anticipation | M and M̄ bound deviations from conditional parallel trends for each ATT(g,t) | Clean identification; breakdown reflects robustness of clean comparisons |
| Sun & Abraham (2021) | Parallel trends for each cohort | Same interpretation as CS; applied to cohort-specific event-study coefficients | Similar to CS; avoids contamination across relative time periods |
| Borusyak et al. (2024) | Homogeneous treatment effects across cohorts and time (for efficiency) | M and M̄ bound deviations under stronger parametric assumptions | May produce inflated breakdown values under heterogeneity due to biased point estimates |
| Goodman-Bacon (2021) | Diagnostic only; not an estimator | HonestDiD applied to TWFE estimates that GB decomposes may be misleading | TWFE estimates can be sign-reversed; HonestDiD breakdown on TWFE may be uninformative |

### 2.4 Critiques and Limitations of HonestDiD

Several important limitations of the HonestDiD approach should be noted. First, the choice of M and M̄ requires researcher judgment about what constitutes a "plausible" deviation from parallel trends [1][16]. The authors acknowledge this: the researcher must specify what magnitude of violation is plausible, informed by economic knowledge and context [1].

Second, the method requires multiple pre-treatment periods—at least three for meaningful implementation [1][2]. With only one pre-period, the relative magnitude restriction becomes less demanding and can inflate breakdown values.

Third, HonestDiD confidence intervals are wider than those from conventional methods; "the gap quantifies the price of robustness" [2]. At large M values, confidence intervals may be so wide as to be uninformative for policy.

Fourth, the framework implicitly assumes that pre-treatment trends are informative about post-treatment violations [1]. The relative magnitudes restriction assumes the maximum post-treatment violation is bounded by a multiple of the maximum observed pre-treatment violation—an assumption that is credible in many settings but not all.

Finally, the method faces "challenges with unbalanced data designs" and relies on balanced panels [1][16]. The R package documentation notes that some functions have limitations with unbalanced panels.

---

## 3. Conditional vs. Unconditional Parallel Trends

### 3.1 Theoretical Distinction and Practical Significance

The distinction between conditional and unconditional parallel trends is fundamental to applied DiD practice. **Unconditional parallel trends** assumes that, in the absence of treatment, the average outcomes for treated and untreated groups would have followed the same path: E[Y_t(0) - Y_{t-1}(0) | G=1] = E[Y_t(0) - Y_{t-1}(0) | G=0]. **Conditional parallel trends** allows this assumption to hold after conditioning on observed covariates: E[Y_t(0) - Y_{t-1}(0) | G=1, X] = E[Y_t(0) - Y_{t-1}(0) | G=0, X] [8][17].

The practical significance is substantial. When treatment is non-random and related to observable characteristics that also affect outcome trends, the unconditional assumption is unlikely to hold [17]. In such settings, researchers must condition on covariates to make the parallel trends assumption plausible. However, this introduces several critical challenges that Caetano and Callaway (2024) term "hidden linearity bias" [17].

### 3.2 How Each Estimator Handles Covariates

#### 3.2.1 Callaway and Sant'Anna (2021): The Uniquely Flexible Framework

Callaway and Sant'Anna (2021) offer the most comprehensive framework for conditional parallel trends among all major estimators [8][18]. Their methodology provides three estimation approaches:

- **Outcome Regression (OR):** Directly modeling the conditional mean of the outcome given covariates
- **Inverse Probability Weighting (IPW):** Weighting by the inverse of the propensity score for treatment conditional on covariates
- **Doubly Robust (DR):** Combining OR and IPW such that the estimator is consistent if either model is correctly specified

The doubly-robust property is particularly valuable. As Sant'Anna and Zhao (2020) show, the estimator "remains consistent even if one (but not both) nuisance parameters are misspecified" and "achieves the semiparametric efficiency bound when both pscore and OR are correctly specified" [18].

The CS framework provides two distinct conditional parallel trends assumptions that researchers can choose between [8]:

**Assumption 4 (Conditional on Never-Treated Group):** For each g ∈ 𝒢 and t such that t ≥ g−δ: E[Y_t(0)−Y_{t-1}(0) | X, G_g=1] = E[Y_t(0)−Y_{t-1}(0) | X, C=1]. This states that conditional on covariates, outcomes for the treated group and never-treated group would have followed parallel paths.

**Assumption 5 (Conditional on Not-Yet-Treated Groups):** For each g ∈ 𝒢 and (s,t): E[Y_t(0)−Y_{t-1}(0) | X, G_g=1] = E[Y_t(0)−Y_{t-1}(0) | X, D_s=0, G_g=0]. This imposes conditional parallel trends between group g and groups not yet treated by a certain time.

Importantly, "Assumption 4 does not restrict observed pre-treatment trends across groups, whereas Assumption 5 does" [8]. This is because Assumption 5 imposes parallel trends for multiple pre-treatment periods, while Assumption 4 only requires it relative to the base period. The CS framework uniquely provides a dedicated pre-test for the conditional parallel trends assumption via the `conditional_did_pretest` function in the `did` R package [19].

#### 3.2.2 Sun and Abraham (2021): Limited Covariate Handling

The Sun and Abraham (2021) interaction-weighted estimator relies on an unconditional parallel trends assumption in its baseline version [10][20]. Covariates can be included in the regression specification, but the estimator does NOT have a dedicated doubly-robust framework for conditional parallel trends. As noted in the MetricGate documentation, the estimator "assumes binary treatment and may lack flexibility with covariates compared to alternatives like Callaway-Sant'Anna" [20].

The SA estimator's primary innovation is its use of cohort-specific interaction terms to avoid contamination across relative time periods [10]. While covariates can be added as control variables in the regression, the identification still relies on unconditional parallel trends within each cohort, conditional only on the additive structure of the regression specification [20][21].

#### 3.2.3 Borusyak, Jaravel, and Spiess (2024): First-Stage Regression Covariates

The Borusyak et al. (2024) imputation estimator handles covariates through the first-stage OLS regression that estimates unit and time fixed effects [11][22]. The first-stage model Y_it = α_i + α_t + βX_it + ε_it is estimated on all untreated observations (never-treated + not-yet-treated), and covariates enter linearly. The imputation approach does NOT have a built-in doubly-robust framework for conditional parallel trends [11].

This means the BJS estimator's covariate handling is fundamentally different from CS. While CS allows the parallel trends assumption to hold conditional on covariates through either outcome regression or propensity score weighting, BJS simply includes covariates as linear controls in the counterfactual model. The difference matters when the relationship between covariates and outcomes is non-linear or when covariate distributions differ substantially between treated and untreated groups.

#### 3.3.4 Goodman-Bacon (2021): Decomposition Diagnostic Only

Goodman-Bacon (2021) provides a "new analysis of models that include time-varying controls" [14] but is primarily a diagnostic decomposition tool, not an estimator for conditional parallel trends. The paper shows how the TWFE estimator's components can be understood as weighted averages of 2×2 DiD comparisons, including when covariates are included [14][23].

### 3.4 How Conditional Parallel Trends Uniquely Positions Callaway and Sant'Anna

The conditional parallel trends capability is perhaps the single most important differentiator among the four major estimators [24]. Callaway and Sant'Anna is the only estimator that provides:

1. **Formal conditional parallel trends assumptions** with two distinct options (Assumption 4 vs. Assumption 5)
2. **Doubly-robust estimation** that remains consistent if either the outcome model or propensity score model is correctly specified
3. **A dedicated pre-test for conditional parallel trends** via `conditional_did_pretest`
4. **Flexibility for time-varying covariates** through the doubly-robust framework

The practical implications are substantial. In applied settings where unconditional parallel trends is not plausible—which is the case in many observational studies—CS provides a principled way to restore identification by conditioning on covariates. This is particularly valuable when:

- Treatment assignment is correlated with observable characteristics that also affect outcome trends
- Covariate distributions differ between treated and untreated groups
- Rich covariate data is available (e.g., demographic controls, pre-treatment outcomes)

### 3.5 Applied Patterns of Covariate Use in Top Journals

While systematic data on covariate use in applied papers is limited, several patterns emerge from the literature [25][26]:

**Health economics applications** frequently condition on detailed health status measures, demographic characteristics, and pre-treatment health trajectories. The Caetano, Callaway, Payne, and Sant'Anna (2024) framework is directly applicable to these settings [27].

**Labor economics applications** commonly include demographic controls (age, education, race/ethnicity), geographic fixed effects (state, county), and pre-treatment outcomes. In the minimum wage application from Callaway and Sant'Anna (2021), "log county population" is included as a covariate [8].

**Settings with time-varying confounders** represent a particularly important domain for conditional parallel trends. The recent working paper by Caetano, Callaway, Payne, and Sant'Anna (2024) on "Difference in Differences with Time-Varying Covariates" directly addresses the problem of time-varying covariates being affected by treatment [27].

---

## 4. Adoption Patterns with Quantitative Evidence

### 4.1 Systematic Evidence from the Literature

Goldsmith-Pinkham (2025) provides the most comprehensive quantitative evidence on DiD adoption across economics subfields, tracking mentions across approximately 44,000 papers including 31,500 NBER working papers (1982–2025) and 12,300 articles from top economics and finance journals (2011–2024) [28]. Key findings include:

- **As of 2024, 63% of applied micro papers mention experimental or quasi-experimental methods**, compared to 47% in finance and 39% in macro/other
- "The credibility revolution outside applied micro has been—to a first approximation—a difference-in-differences revolution"
- DiD accounts for a **55% increase in method mentions in finance** compared to 30% in applied micro
- The rapid diffusion of methodological refinements suggests that practitioners are adopting improved estimators alongside the research design itself [28]

Roth, Sant'Anna, Bilinski, and Poe (2023), published in the *Journal of Econometrics*, has accumulated 3,165 citations as of May 2026 [29]. The paper provides a practical checklist guiding researchers on treatment timing, parallel trends assumptions, and inference approaches. Table 1 of the paper distills insights into a practical checklist for DiD practitioners [29].

Baker, Larcker, and Wang (2022) found that staggering DiD is employed in nearly half of all DiD studies in top finance and accounting journals (49% of DiD studies) [15]. When they re-analyze three major published studies, they find that "original findings often become statistically insignificant or change in magnitude and direction" [15].

The Dahis Research Unit Tests checklist explicitly states: "Papers with staggered adoption must use or address heterogeneity-robust estimators" [30]. Pass conditions require either: (a) the paper uses a heterogeneity-robust estimator as primary specification, OR (b) the paper uses TWFE but reports robustness to a heterogeneity-robust estimator AND discusses the Goodman-Bacon decomposition explicitly. The guidance notes: "Papers written after 2022 have no excuse" [30].

### 4.2 Time Trends in Adoption

The evidence suggests a clear discontinuity in adoption patterns:

**2020 and earlier:** Most papers used traditional TWFE only. Key methodological papers (Goodman-Bacon working paper, Callaway & Sant'Anna working paper, Sun & Abraham working paper) were just appearing or in working paper form [31].

**2021–2022:** The *Journal of Econometrics* published Goodman-Bacon (2021), Callaway and Sant'Anna (2021), and Sun and Abraham (2021) in a concentrated period. Awareness of the issues spread rapidly. The Roth et al. (2023) paper (arXiv January 2022) synthesized findings and provided a checklist [29].

**2022–2024:** Adoption of heterogeneity-robust estimators accelerated. The Dahis checklist states: "Papers written after 2022 have no excuse" [30]. A consensus had crystallized that heterogeneity-robust estimators should be standard practice.

**2024–2025:** The Borusyak, Jaravel, and Spiess (2024) paper was published in the *Review of Economic Studies*, further solidifying availability of these methods in a top-5 journal outlet [22]. Baker, Callaway, Cunningham, Goodman-Bacon, and Sant'Anna (2026) provide a comprehensive practitioner's guide forthcoming in the *Journal of Economic Literature* [31].

### 4.3 Justification Practices

The Roth et al. (2023) synthesis recommends that researchers discuss: (1) what parallel trends assumption they are relying on, (2) which comparison group is used (never-treated or not-yet-treated), (3) how they are handling staggered treatment timing, and (4) what robustness checks they perform [29].

Common approaches to justification observed in the literature include:
- **Goodman-Bacon decomposition** as a diagnostic to identify the extent of "bad comparisons"
- **Event-study plots** with leads and lags to assess pre-trends and dynamic treatment effects
- **Reporting multiple estimators** as robustness checks (TWFE + CS + SA + BJS + stacked DiD)
- **Citing the methodological literature** (Roth et al. 2023, Goodman-Bacon 2021, Callaway & Sant'Anna 2021, Sun & Abraham 2021, Borusyak et al. 2024)
- **Monte Carlo simulations** (less common but present in some methodologically-oriented papers)
- **Sensitivity analyses** such as Rambachan and Roth (2023) HonestDiD [1][30]

### 4.4 Differences Between Labor and Health Economics

While both subfields have rapidly adopted heterogeneity-robust methods, important differences exist [28]:

**Labor economics** was an earlier and faster adopter, in part because several key methodological advances were developed by labor economists. Callaway's minimum wage research, Dube's minimum wage work, and the Cengiz et al. (2019) stacked DiD approach were all developed specifically for labor economics applications [8][32].

**Health economics** has actively disseminated these methods through review articles and tutorials (Freedman et al. 2023, Wang et al. 2024, Annual Reviews 2024) [12][33]. The 2024 PMC article on "Advances in Difference-in-differences Methods for Policy Evaluation Research" includes a Monte Carlo simulation comparing traditional and modern estimators specifically for health policy contexts [12].

Goldsmith-Pinkham (2025) finds that "the credibility revolution outside applied micro has been—to a first approximation—a difference-in-differences revolution," and that growth in finance (which includes some health finance) is largely driven by DiD methods [28]. However, applied micro (which includes labor economics) leads adoption overall at 63% of papers mentioning experimental or quasi-experimental methods.

---

## 5. Efficiency Comparisons in Finite Samples

### 5.1 The Borusyak et al. Efficiency Claims

Borusyak, Jaravel, and Spiess (2024) claim that their imputation estimator achieves "30-360% reduction in standard deviation" compared to alternative heterogeneity-robust estimators [22]. The published version states: "In the MPC application, we find large gains of our imputation estimator: the confidence interval is about 50% longer for each week relative to the rebate for de Chaisemartin and D'Haultfœuille" [22].

The primary source of these efficiency gains lies in how the imputation estimator uses data. The estimator "uses ALL untreated observations (never-treated + not-yet-treated periods of eventually-treated units) to estimate the counterfactual model" [11]. This means:
- **All pre-treatment periods** are used for estimation of unit and time fixed effects
- **All untreated observations across all time periods** contribute to the counterfactual model
- **No observations are discarded** as "contaminated" (unlike CS, which uses only a single pre-treatment period as the base period)

The `diff-diff` Python package documentation states that the imputation estimator produces "~50% shorter confidence intervals than Callaway-Sant'Anna and 2-3.5 times shorter than Sun-Abraham" [11].

### 5.2 Critical Evaluation of Efficiency Claims

The efficiency gains of BJS are genuine but come with important caveats:

**First, the gains depend on treatment effect homogeneity.** The BJS estimator in its baseline specification imposes "potentially homogeneous treatment effects across cohorts and time" [11]. When treatment effects are actually homogeneous, this restriction increases efficiency because it pools information across all treated cohorts. When effects are heterogeneous, the restriction can introduce bias [12].

**Second, the gains require a large pool of untreated observations.** The efficiency advantage of BJS comes from using ALL untreated observations to estimate the counterfactual model. When there are few never-treated units, this advantage diminishes because the estimator relies more heavily on not-yet-treated units, which may have different characteristics [22][34].

**Third, the "Advances in Difference-in-differences" PMC article found** that "among heterogeneous treatment effects-robust estimators, Callaway and Sant'Anna and Sun and Abraham estimators outperformed Borusyak et al and Wooldridge" in simulation settings with treatment effect heterogeneity [12]. This suggests that when treatment effects are heterogeneous, the efficiency gains of BJS may come at the cost of increased bias.

**Fourth, Chen, Sant'Anna, and Xie (2024)** show that semiparametric efficient estimators can "optimally explore different pre-treatment periods and comparison groups to obtain the tightest (asymptotic) confidence intervals." Their simulations demonstrate "substantial precision gains, often exceeding 40%" compared to popular existing methods—but these gains are achieved through optimal weighting rather than homogeneity restrictions [35].

### 5.3 Effect of Comparison Group Choice on Efficiency

The choice of comparison group (never-treated vs. not-yet-treated) has important efficiency implications for all estimators [8][36]:

**When a large never-treated group exists:** All estimators are most efficient when there is a large pool of never-treated units. CS with Assumption 4 (never-treated comparison) is well-suited here. BJS also benefits from a large never-treated pool because its first-stage model is estimated on all untreated observations.

**When there are few never-treated units:** Efficiency declines for all estimators. CS must rely on not-yet-treated comparisons (Assumption 5), which reduces the available control observations. BJS continues to use not-yet-treated units but may have reduced efficiency because these units eventually become treated, limiting the number of pre-treatment periods available for estimating the counterfactual.

**When there are many cohorts:** Efficiency declines for CS and SA because these estimators estimate separate parameters for each cohort, reducing the effective sample size per parameter. BJS, by pooling information across cohorts, is more efficient in this setting—but only if the homogeneity assumption is valid [22][34].

### 5.4 Replication Evidence

The CLLX (2026) reanalysis, published in the *American Political Science Review*, evaluated 49 studies from leading political science journals (2017-2023) using six HTE-robust estimators [36]. Their key findings on efficiency include:

- "HTE-robust estimators yield qualitatively similar but highly variable results compared to TWFE"
- "Many studies are underpowered when accounting for HTE and potential PT violations"
- "The main threats to causal inference with panel data are PT violations and insufficient power, more so than the weighting issue under HTE" [36]

This suggests that efficiency considerations are practically important—many published studies may lack power when using heterogeneity-robust estimators, and the choice of estimator can meaningfully affect whether results reach statistical significance.

---

## 6. Anticipation Effects Across Domains

### 6.1 Theoretical Framework

The "no-anticipation assumption" in DiD states that treatment has no effect on outcomes before its implementation [8][29]. Roth, Sant'Anna, Bilinski, and Poe (2023) explain that this means "units do not act on the knowledge of their future treatment date before treatment starts" [29].

A recent paper (arXiv:2507.12891) clarifies that confusion arises from the ambiguity between the treatment *implementation* and the *plan or decision* to implement the treatment. The authors propose an expanded causal model introducing a decision variable (P), which deterministically causes treatment implementation (A), clarifying that anticipation effects relate to the influence of plans (not implementations) on prior outcomes [37].

### 6.2 Applied Domains Where Anticipation Effects Are Most Serious

**Labor Economics:** Minimum wage increases are among the most studied policies where anticipation effects are a serious concern. The Callaway and Sant'Anna (2021) minimum wage application shows that "while unconditional DiD shows negative employment effects, these results may be affected by anticipation" [8]. The German minimum wage experience provides clear evidence: ignoring existing wage distribution trends prior to 2015 "biases minimum wage effect estimates upwards" [38]. The classic "Ashenfelter dip" in training program evaluations is another canonical example where labor market outcomes dip before program participation due to anticipation [39].

**Health Economics:** Health policies often involve announcement lags between legislative passage and implementation. Medicaid expansion under the ACA involved state-level decisions that were publicly debated and announced in advance, creating windows for anticipation effects [12]. Malani and Reif (2015) provide the most direct evidence, finding that "Interpreting pre-trends as evidence of anticipation increases the estimated effect of these reforms by a factor of two compared to a model that ignores anticipation" [40].

**Public Finance/Tax Policy:** Anticipation effects are extremely well-documented in tax policy. Mertens and Ravn find that "about half of tax changes are anticipated with a median lag of six quarters" and that "anticipated tax liability tax cuts give rise to contractions in output, investment and hours worked prior to their implementation" [41].

**Tort Reform:** Malani and Reif (2015) extensively document anticipation effects in tort reform settings. Their paper titled "Interpreting pre-trends as anticipation: Impact on estimated treatment effects from tort reform" shows that "accounting for anticipation effects doubles the estimated effect of tort reform on physician supply" [40].

### 6.3 How Each Estimator Handles Anticipation

#### 6.3.1 Callaway and Sant'Anna (2021): The Most Explicit Handling

Callaway and Sant'Anna provide the most explicit and flexible handling of anticipation effects through their δ (delta) parameter [8][42]. The Assumption 3 (Limited Treatment Anticipation) states:

"There is a known δ ≥ 0 such that E[Y_t(g) | X, G_g = 1] = E[Y_t(0) | X, G_g = 1] for all g ∈ 𝒢, t ∈ {1,...,T} such that t < g − δ."

Key features:
- When δ = 0, this imposes a "no-anticipation" assumption
- When δ > 0, the method accommodates anticipated treatment
- The reference period adjusts to t = g - δ - 1; the more anticipation is allowed (higher δ), the further back the reference period moves
- The parallel trends assumptions become stronger as δ increases, because they must hold over a longer pre-treatment window

The `did` R package implements this through the `anticipation` argument in `att_gt()`, with a default of 0 [42]. In the `csdid` Stata package, users can set the `anticipation` parameter [43].

A dedicated vignette titled "Writing Extensions to the did Package" demonstrates the complete workflow: simulated data exhibiting pre-treatment dips due to anticipation; the standard `att_gt()` with `anticipation=0` produces biased estimates and rejected parallel trends; setting the appropriate anticipation period recovers the true effects [42].

#### 6.3.2 Sun and Abraham (2021): No Explicit Mechanism

Sun and Abraham (2021) do not provide an explicit mechanism for handling anticipation effects. Instead, they rely on researchers to drop pre-treatment periods from the estimation window if anticipation is suspected [10]. If anticipation is present, the SA estimator's pre-treatment coefficients will be biased, and post-treatment event-study coefficients may also be contaminated because the "clean controls" approach requires that comparison group outcomes in pre-periods are valid counterfactuals [20].

#### 6.3.3 Borusyak, Jaravel, and Spiess (2024): User-Specified Anticipation Periods

The BJS imputation estimator handles anticipation by allowing the user to specify anticipation periods. When anticipation is specified, those periods are treated as contaminated and excluded from the estimation of the counterfactual model [22][44]. The `did_imputation` package accepts an `anticipation` parameter that affects which observations are considered "untreated" for estimation purposes.

#### 6.3.4 Goodman-Bacon (2021): Not Addressed

Goodman-Bacon (2021) does not explicitly discuss anticipation effects in the core assumptions of the decomposition [14][23]. The paper's identifying assumption comprises constant treatment effects over time plus parallel trends, with no formal no-anticipation condition.

### 6.4 Consequences of Misspecifying the Anticipation Parameter

The consequences of incorrect anticipation parameterization are substantial and well-documented [42]:

**When δ=0 is assumed but anticipation exists:**
- Biased estimates of ATT(g,t)
- Rejected parallel trends pre-tests (which serves as a diagnostic signal)
- Event-study plots showing significant pre-treatment coefficients
- Potential misinterpretation of anticipation effects as parallel trends violations

The Malani and Reif (2015) finding is striking: ignoring anticipation can change estimated effects by a factor of two [40].

**When δ is set too large (over-correcting):**
- Loss of efficiency and fewer usable pre-treatment periods
- Potential bias from using more distant base periods (parallel trends is more demanding over longer horizons)
- Potential mistaking pre-existing trends for anticipation

**When δ is set too small (under-correcting):**
- Partial correction for anticipation; remaining bias
- Attenuated but possibly still significant pre-treatment coefficients

### 6.5 Current Applied Practice

The Roth et al. (2023) synthesis notes that most DiD applications assume no anticipation without explicitly testing it [29]. The standard approach in many applied papers is to:

1. Estimate a TWFE or event-study specification
2. Check for pre-trends visually or via a pre-trends test
3. If pre-trends are found, drop the affected periods, add group-specific linear time trends, or switch to a different comparison group
4. Rarely is anticipation explicitly modeled as a behavioral response

The Chen, Denteh, Kédagni, and Bilinski (2024) working paper proposes a novel bounding approach for anticipation effects, providing partial identification of ATT under bounded anticipation assumptions [45]. This approach is still in the working paper stage and has not yet been implemented in standard software packages.

---

## 7. The Role of the Goodman-Bacon Decomposition in Applied Practice

### 7.1 What the Decomposition Reveals

Goodman-Bacon (2021) proves that the TWFE DiD estimator equals a weighted average of all possible two-group, two-period DiD estimators in the data [14][23]. The decomposition identifies four types of 2×2 comparisons:
- **Treated vs. never-treated** (unproblematic)
- **Early-treated vs. late-treated (before late units are treated)** — uses not-yet-treated as controls (unproblematic)
- **Late-treated vs. early-treated (after late units are treated)** — uses already-treated as controls (problematic when effects vary over time)
- **Different timing groups where both serve as treatment and control at different points**

The paper demonstrates that "a causal interpretation of two-way fixed effects DD estimates requires both a parallel trends assumption and treatment effects that are constant over time" [14]. "When treatment effects do not change over time, TWFEDD yields a variance-weighted average of cross-group treatment effects and all weights are positive. Negative weights only arise when average treatment effects vary over time" [14].

### 7.2 Usage Patterns in Applied Work

The `bacondecomp` Stata module and R package have been widely adopted [46][47]. The Stata module "generates a scatterplot of 2x2 difference-in-difference estimates and their associated weights" [47]. The module has been cited in 39 research papers spanning economics and social sciences.

Baker, Larcker, and Wang (2022) find that "staggered DiD is employed in nearly half of DiD studies in top finance and accounting journals" (55% of 744 DiD papers manually reviewed) [15]. They recommend "decomposing the static TWFE DiD estimator (e.g., the Goodman-Bacon, 2021, decomposition) when possible" [15].

### 7.3 Editorial Expectations: Is Bacon Decomposition Sufficient?

The consensus has evolved substantially since 2021 [29][30][31]:

**2020–2021:** The Bacon decomposition, combined with event-study plots, was considered a reasonable diagnostic approach for assessing the extent of "bad comparisons" in TWFE estimates.

**2022–2023:** The Dahis checklist states: "Papers with staggered adoption must use or address heterogeneity-robust estimators" [30]. Pass conditions require either primary use of heterogeneity-robust estimators OR reporting robustness checks using such estimators alongside Bacon decomposition. "Papers written after 2022 have no excuse" [30].

**2024–2025:** The emerging standard is that Bacon decomposition alone is insufficient. Editors and referees at top-5 journals now expect authors to use heterogeneity-robust estimators (CS, SA, BJS, or de Chaisemartin & D'Haultfœuille) as primary specifications, with Bacon decomposition as a complementary diagnostic [29][31].

The Roth et al. (2023) synthesis explicitly recommends: "Overall, the literature emphasizes careful assumption articulation, transparent estimands and control group selection, and robust inference strategies to improve the credibility and interpretability of DiD analyses" [29].

### 7.4 Settings Where Bacon Decomposition Reveals Unique Insights

While heterogeneity-robust estimators solve the negative weighting problem, the Bacon decomposition reveals important information that these estimators may obscure [14][15][23]:

**Identifying Which 2×2 Comparisons Drive Results:** The Bacon decomposition explicitly shows the weight and contribution of each type of comparison. This enables researchers to see whether their TWFE estimate is being driven by clean comparisons (treated vs. never-treated) or contaminated comparisons (treated vs. already-treated). This information is lost when using heterogeneity-robust estimators that simply exclude the problematic comparisons.

**Diagnosing Why Estimates Differ:** The Bacon decomposition serves as "a quick check on negative weights and explains differences in estimates" between TWFE and heterogeneity-robust methods [48]. It shows why the newer estimators produce different estimates by revealing which comparisons are dropped.

**Understanding Treatment Effect Dynamics:** The decomposition reveals whether time-varying treatment effects are biasing the static TWFE estimate. As Goodman-Bacon (2021) shows, "a simple DD underestimates the effect at about -3 suicides per million women, whereas an event-study specification suggests a stronger effect closer to -5" [14].

**Mapping to CS ATT(g,t) Framework:** The timing groups in the Bacon decomposition map directly onto the CS framework. Early-treated vs. never-treated comparisons correspond to CS ATT(g,t) with never-treated comparison. The "forbidden" comparisons (late-treated vs. early-treated) are those that CS explicitly avoids [8][48].

---

## 8. Practical Implementation Challenges

### 8.1 Cross-Validation: Do Software Implementations Agree?

A critical implementation concern is whether different software implementations of the same estimator produce consistent results [49][50][51].

**Callaway and Sant'Anna: R `did` vs. Stata `csdid`:** A Statalist discussion reports "Differences between results from 'csdid' command and 'did' package in R" [49]. The R package (version 2.3.0, released 2025-12-13) provides group-time ATTs with bootstrap standard errors and uniform confidence bands. The Stata package offers outcome regression, IPW, and doubly-robust estimation [50]. A new version called `csdid2` has been fully rewritten in Mata for enhanced speed [51].

**Sun and Abraham: R `fixest::sunab` vs. Stata `eventstudyinteract`:** A critical discrepancy has been documented in GitHub Issue #287 on the `fixest` repository [52]. The issue reports that "`sunab` not matching Stata Sun and Abraham implementation" produces "wonky results that don't look like other estimators." The finding is that "Sun and Abraham with a never-treated group, the correct reference periods being dropped, and no covariates is numerically equivalent to Callaway and Sant'Anna." The R `did` package matches the Stata implementation, implying the `sunab` code in `fixest` may have errors [52]. Users should verify results against the Stata implementation or use alternative specifications.

**Borusyak et al.: R `didimputation` vs. Stata `did_imputation`:** Both implementations of the imputation estimator are available, with the Stata package maintained by Borusyak and the R package by Kyle Butts [53][54]. Known issues include the Stata command producing error r(123) with large datasets, and errors when covariates are included [55][56].

**De Chaisemartin & D'Haultfœuille: `did_multiplegt` vs. `did_multiplegt_dyn`:** The newer `did_multiplegt_dyn` command uses analytic variance formulas and is "much faster than did_multiplegt" [57]. The R implementation supports parallelization and is "faster than Stata" [58].

### 8.2 Common User Errors

Several recurring implementation errors are documented across user forums [49][55][56][59]:

**Incorrect group variable specification:** The group variable must indicate the period when a unit first becomes treated. For never-treated units, this should be coded as 0 or Inf (depending on the package) [59].

**Mishandling of panel structure:** The `csdid` command requires "proper panel data setup, with only one observation per ID per period" [50].

**Misunderstanding comparison groups:** Users frequently confuse "never-treated" and "not-yet-treated" comparison groups, which can produce meaningfully different results [8][59].

**Dropped observations and missing data:** The `did` R package has `allow_unbalanced_panel = TRUE` option. Without it, observations with missing data are dropped, potentially causing errors [60].

**Incorrect aggregation of group-time ATTs:** Users must understand that different aggregation types ("simple," "dynamic," "group," "calendar") answer different causal questions [59].

**The csdid "Stopped by user" error:** Documented in GitHub Issue #21 on `friosavila/csdid_drdid`, this error occurs when the sample does not include an observation for every treatment in every time period [61].

### 8.3 Handling of Missing Data, Unbalanced Panels, and Time-Varying Covariates

**Missing data and unbalanced panels:**
- The `did` R package has `allow_unbalanced_panel = TRUE` [60]
- The `csdid` Stata package "accommodates both panel data and repeated cross-section data" [50]
- `did_multiplegt_dyn` "handles unbalanced panels" explicitly [57]
- The `sunab()` function in `fixest` excludes always-treated units via NA values [62]

**Time-varying covariates:**
A blog post by Brantly Callaway issues several critical warnings about time-varying covariates in TWFE: "TWFE regressions won't work well if the time-varying covariates are affected by the treatment" (the "bad control" problem) [63]. The CS doubly-robust framework is designed to handle time-varying covariates properly [8][27]. The Caetano, Callaway, Payne, and Sant'Anna (2024) working paper introduces novel identification strategies for cases where covariates may be affected by treatment, proposing doubly-robust estimation methods and regression-based imputation approaches [27].

### 8.4 Computational Cost for Large Datasets

Computational requirements vary substantially across packages and data sizes [50][58][62][64]:

**`fixest` R package (Sun & Abraham via `sunab()`):** Consistently identified as the fastest option. Based on optimized parallel C++ code. "Estimation is also very fast—you will likely find it to be the fastest option among all of the specialist DiD libraries" [62]. Version 0.14.1 was published May 4, 2026. For very large datasets (150M+ observations), variables should be numeric rather than factors to avoid memory explosions [64].

**`did` R package (Callaway & Sant'Anna):** Designed for efficient computation with parallel processing options, but does not make the same speed claims as `fixest` [59].

**`csdid` Stata package:** Can encounter "too many variables specified" error with large datasets (200K+ observations) because the command internally creates many variables for each group-time combination [50]. The `csdid2` (Mata) version provides memory requirement estimates: (nobs × periods × groups × 16 / 1024^3) GB [51].

**`didimputation` R package:** Depends on `data.table` and `fixest` for efficient computation [53].

**`DIDmultiplegt` / `DIDmultiplegtDYN`:** The original `did_multiplegt()` is "the slowest of the available DiD estimators due to bootstrap replications." The newer `did_multiplegt_dyn` is much faster [57][58].

**Julia implementation:** `EventStudyInteracts.jl` is "3.69 times faster than Stata" for Sun & Abraham estimation with 2M observations, and supports GPU acceleration [65].

---

## 9. The Borusyak et al. Efficiency Advantage Controversy

### 9.1 Documented Divergence in Applied Work

Several documented cases show Borusyak et al. estimates diverging substantially from other estimators [13][36][66]:

A Stack Overflow question (ID 76101056) reports: "The results of 3/4 are very similar, however, the Borusyak estimates are substantially higher than any of those (between 50% and 100%)." Pre-trends look fine and are very similar for all methods. The respondent notes that "in the data there is very likely substantial treatment effect heterogeneity" [13].

The CLLX (2026) reanalysis found that "HTE-robust estimators yield qualitatively similar but highly variable results" across 49 studies [36]. While they did not single out BJS specifically, the finding that estimates vary across estimators is broadly consistent with the divergence documented elsewhere.

### 9.2 What Data Structures Produce Divergence

Based on the evidence gathered, divergence between BJS and CS/SA estimates is most likely when [11][22][34][66]:

1. **Treatment effects are heterogeneous across cohorts:** If early-treated cohorts have systematically different effects from late-treated cohorts, the different weighting schemes across estimators produce different aggregate estimates.

2. **Treatment effects are dynamic:** If effects evolve over time post-treatment (phase-in or fade-out), the different ways estimators handle dynamics produce divergence.

3. **There are few never-treated units:** When almost all units eventually become treated, the imputation estimator relies heavily on not-yet-treated units, and performance differences emerge.

4. **There are few pre-treatment periods:** With limited pre-treatment data, the precision advantage of BJS diminishes.

5. **Covariate distributions differ across groups:** BJS includes covariates linearly in the first-stage model; CS allows doubly-robust conditioning through propensity score weighting.

### 9.3 Is the Imputation Estimator More Sensitive to Parallel Trends Violations?

The BJS paper explicitly claims: "Moreover, these gains [in efficiency] do not come at a cost of systematically higher sensitivity to parallel-trend violations" [22]. The `diff-diff` documentation notes that both estimators "should give similar results when: Treatment effects are relatively homogeneous across cohorts; Parallel trends holds" [11]. This implies that when parallel trends does NOT hold, estimators can diverge.

The sensitivity mechanisms differ across estimators:

1. **BJS uses ALL untreated observations** for the counterfactual model. If never-treated and not-yet-treated units have different trends, pooling them can induce bias if parallel trends holds only for specific subsets.

2. **CS and SA allow explicit specification** of which comparison group is used (never-treated, not-yet-treated, or both), giving more control over potential violations.

3. **BJS's use of all pre-treatment periods** for imputation means a violation in any pre-treatment period can bias the estimated fixed effects. CS uses only the base period (g-1 or g-δ-1) as reference, so violations in earlier pre-treatment periods do not directly affect the counterfactual.

### 9.4 Should Researchers Be Concerned When BJS Differs?

Based on the available evidence, researchers should treat divergence between BJS and other estimators as a diagnostic signal worth investigating [11][29][31][66]:

**First, check whether the parallel trends assumption is plausible** for all comparison groups simultaneously.

**Second, investigate whether treatment effects are heterogeneous across cohorts.** The `diff-diff` documentation explicitly recommends running both estimators and comparing as a robustness check, with the understanding that agreement suggests homogeneous effects and valid parallel trends [11].

**Third, use the BJS homogeneity assumption as a benchmark.** The `five_estimators_example.do` file in the BJS GitHub repository demonstrates comparing five estimators on the same data to assess robustness [66].

**Fourth, prefer CS or SA as primary when treatment effect heterogeneity is suspected.** The PMC article found that "among heterogeneous treatment effects-robust estimators, Callaway and Sant'Anna and Sun and Abraham estimators outperformed Borusyak et al and Wooldridge" under heterogeneity [12].

**Fifth, conduct HonestDiD sensitivity analysis on all estimators** to assess whether the divergence is driven by parallel trends violations or by the homogeneity assumption [1][4].

The CLLX (2026) overall recommendation is that "the credibility of identifying assumptions is more important than the choice of estimator" [36]. When estimates diverge, researchers should probe the sources of divergence rather than simply choosing the estimator that produces the most favorable result.

---

## 10. Updated Practitioner Recommendations

### 10.1 Which Estimator(s) to Use as Primary in Different Data Environments

Based on the comprehensive analysis above, the following recommendations emerge for applied researchers in labor and health economics:

| Data Environment | Primary Estimator | Rationale |
|---|---|---|
| Strong theoretical basis for parallel trends; few covariates | Sun & Abraham (2021) via `fixest::sunab` | Simple "drop-in" replacement for TWFE event study; fast computation |
| Rich covariate data; conditional PT more plausible | Callaway & Sant'Anna (2021) via `did` or `csdid` | Doubly-robust with covariates; conditional PT tests; flexibility for time-varying covariates |
| Many pre-treatment periods; homogeneity plausible | Borusyak et al. (2024) via `did_imputation` or `didimputation` | Uses all pre-period data; potential efficiency gains |
| Non-binary or non-absorbing treatments | De Chaisemartin & D'Haultfœuille via `did_multiplegt_dyn` | Handles multi-valued and reversible treatments |
| Need to diagnose TWFE bias | Goodman-Bacon (2021) via `bacondecomp` | Decomposition diagnostic; use alongside a heterogeneity-robust estimator |
| Few never-treated units | Callaway & Sant'Anna with not-yet-treated comparison | Avoids reliance on few never-treated units; clean identification |

### 10.2 How to Report Results and Justify Estimator Choice

Following the Roth et al. (2023) checklist and Baker et al. (2026) practitioner guide, researchers should [29][31]:

1. **Clearly articulate the parallel trends assumption** used (unconditional or conditional, and which covariates are conditioned on)
2. **Specify the comparison group** (never-treated or not-yet-treated) and justify the choice
3. **Define the target estimand** (overall ATT, event-study parameters, group-specific effects)
4. **Report results from multiple heterogeneity-robust estimators** (at least two of: CS, SA, BJS)
5. **Conduct and report HonestDiD sensitivity analysis** with breakdown values
6. **If using conditional parallel trends, report the pre-test results**
7. **If using BJS, explicitly discuss the homogeneity assumption** and test for heterogeneity using CS or SA as a robustness check
8. **Report the Goodman-Bacon decomposition** as a complementary diagnostic

### 10.3 Checklist of Robustness Checks Top Journal Referees Now Expect

Based on the consensus across surveys and editorial guidelines [29][30][31]:

- [ ] Uses at least one heterogeneity-robust estimator as primary specification (CS, SA, BJS, de Chaisemartin & D'Haultfœuille)
- [ ] Reports results from at least two heterogeneity-robust estimators as robustness
- [ ] Reports Goodman-Bacon decomposition to assess "bad comparisons" in TWFE
- [ ] Includes event-study plots with pre-treatment coefficients as placebo tests
- [ ] Reports HonestDiD breakdown values (both M and M̄)
- [ ] Explicitly states which parallel trends assumption is used (conditional or unconditional, with which covariates)
- [ ] States which comparison group is used (never-treated or not-yet-treated) and why
- [ ] If conditional parallel trends is used, reports the `conditional_did_pretest` results
- [ ] Discusses anticipation effects and, if relevant, addresses them using the δ parameter
- [ ] If BJS is used, discusses the homogeneity assumption and shows robustness to CS/SA
- [ ] Reports sensitivity to comparison group choice
- [ ] If few treated or few never-treated units, discusses finite-sample implications

### 10.4 Handling Specific Cases: Few Treated Units, Few Never-Treated Units, or Both

**Few treated units:** When there are few treated units (e.g., a single treated state), all estimators will have large standard errors. In this setting:
- Consider using CS with not-yet-treated comparison to maximize the control pool
- Use BJS if the homogeneity assumption is plausible (pooling information across the few treated units may increase precision)
- Report HonestDiD sensitivity analysis with careful attention to the breakdown value (it may be low due to wide confidence intervals)
- Consider synthetic control methods as an alternative

**Few never-treated units:** When almost all units eventually become treated, the never-treated comparison group is small or absent [8][36]:
- Use CS with not-yet-treated comparison group (Assumption 5)
- The BJS estimator also relies on not-yet-treated units in this setting
- Avoid SA if last-treated units are needed as the comparison group (they may be few)
- Baker, Larcker, and Wang (2022) find that "the prevalence of staggered DiD reflects a common pattern" with few never-treated units [15]

**Both few treated and few never-treated units:** This is the most challenging case:
- Consider using stacked DiD (Cengiz et al. 2019 approach) which creates clean 2×2 comparisons [32]
- Use CS with not-yet-treated comparison
- The BJS homogeneity assumption may be more credible with few treated units (less opportunity for heterogeneity across cohorts)
- Report permutation-based inference or Fisher randomization tests
- Consider whether the study has adequate power at all before proceeding
- The CLLX (2026) finding is directly relevant: "many studies are underpowered when accounting for HTE and potential PT violations" [36]

---

## 11. Conclusion

The staggered DiD literature has undergone a remarkable transformation since Goodman-Bacon's (2021) decomposition revealed the vulnerabilities of TWFE estimators. Four main approaches have emerged—Callaway and Sant'Anna's (2021) group-time ATTs with doubly-robust estimation, Sun and Abraham's (2021) interaction-weighted estimator, Borusyak et al.'s (2024) imputation-based estimator, and de Chaisemartin and D'Haultfœuille's (2020) estimator for non-binary treatments—each offering distinct advantages in handling heterogeneous treatment effects and dynamic treatment timing.

The assumptions underlying these estimators differ in important ways. Callaway and Sant'Anna offer the most flexible framework, allowing conditional parallel trends with covariates and a formal parameter for anticipation effects. Sun and Abraham and Borusyak et al. rely on unconditional parallel trends in their baseline versions, though both can accommodate covariates to varying degrees. The choice between never-treated and not-yet-treated comparison groups is a critical design decision.

Rambachan and Roth's (2023) HonestDiD framework has emerged as the standard approach for sensitivity analysis, replacing problematic binary pre-trend tests with continuous robustness measures. The framework integrates with all major estimators, but the interpretation of sensitivity analysis depends on the estimator's assumptions. CS and SA provide the cleanest baseline for HonestDiD because they are robust to treatment effect heterogeneity.

The evidence on adoption patterns in top-5 economics journals indicates a clear discontinuity around 2021-2022, with heterogeneity-robust estimators becoming standard practice. The Baker et al. (2026) practitioner's guide forthcoming in the *Journal of Economic Literature* signals that these methods have matured enough for codification as best practice.

For applied researchers, the message is clear: papers with staggered adoption should use at least one heterogeneity-robust estimator as their primary specification, report results from multiple estimators as robustness checks, conduct HonestDiD sensitivity analysis, and carefully articulate their identifying assumptions. When estimates diverge across estimators, researchers should investigate the sources rather than choosing the most favorable result.

---

## Sources

[1] Rambachan, A. & Roth, J. (2023). "A More Credible Approach to Parallel Trends." Review of Economic Studies, 90(5), 2555-2591. https://academic.oup.com/restud/article/90/5/2555/6950347

[2] HonestDiD Interactive Lab. https://metricgate.com/docs/honest-did-breakdown-value

[3] HonestDiD R Package (GitHub). https://github.com/asheshrambachan/HonestDiD

[4] HonestDiD Stata Package (SSC). https://ideas.repec.org/c/boc/bocode/s459138.html

[5] diff-diff Python Documentation - HonestDiD Integration. https://diff-diff.readthedocs.io/en/latest/api/honestdid.html

[6] Callaway, B. & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with Multiple Time Periods." Journal of Econometrics, 225(2), 200-230. https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948

[7] Sant'Anna, P.H.C. "Causal Inference using Difference-in-Differences" Lecture Slides, January 2025. https://psantanna.com/DiD/12_CS.pdf

[8] Callaway, B. & Sant'Anna, P.H.C. (2021). Full text PDF. https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[9] HonestDiD Applied: Medicaid Expansion Analysis. https://carlos-mendez.org/post/2023-11-15-honestdid-stata-tutorial/

[10] Sun, L. & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects." Journal of Econometrics, 225(2), 175-199. https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Paper2023/D3--Sun-2020-Estimating-dynamic-treatment-effects-in-event-studies-with-heterogeneous-treatment-effects.pdf

[11] diff-diff Documentation - Imputation DiD. https://diff-diff.readthedocs.io/en/latest/api/imputation.html

[12] Wang, G., Hamad, R., & White, J.S. (2024). "Advances in Difference-in-differences Methods for Policy Evaluation Research." PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929

[13] Stack Overflow Discussion - Borusyak Estimates Higher Than Other Estimators. https://stackoverflow.com/questions/76101056/staggered-did-borusyak-estimates-way-higher-than-callawaysantanna-sunabra

[14] Goodman-Bacon, A. (2021). "Difference-in-differences with variation in treatment timing." Journal of Econometrics, 225(2), 254-277. https://www.sciencedirect.com/science/article/abs/pii/S0304407621001445

[15] Baker, A.C., Larcker, D.F., & Wang, C.C.Y. (2022). "How much should we trust staggered difference-in-differences estimates?" Journal of Financial Economics, 144(2), 370-395. https://www.sciencedirect.com/science/article/pii/S0304405X22000204

[16] HonestDiD CRAN Documentation. https://cran.r-project.org/web/packages/HonestDiD/index.html

[17] Caetano, C. & Callaway, B. (2024). "Difference-in-Differences when Parallel Trends Holds Conditional on Covariates." arXiv:2406.15288. https://bcallaway11.github.io/files/DID-Covariates/Caetano_Callaway_2024.pdf

[18] Sant'Anna, P.H.C. & Zhao, J. (2020). "Doubly Robust Difference-in-Differences Estimators." Slides. https://psantanna.com/files/SantAnna_Zhao_2020_slides.pdf

[19] did R Package - Pre-Testing Vignette. https://bcallaway11.github.io/did/articles/pre-testing.html

[20] MetricGate - Sun-Abraham Estimator. https://metricgate.com/docs/event-study-sun-abraham

[21] Statalist Discussion - Wooldridge ETWFE vs Sun & Abraham. https://www.statalist.org/forums/forum/general-stata-discussion/general/1770049-difference-between-wooldridge-extended-twfe-sun-abraham

[22] Borusyak, K., Jaravel, X., & Spiess, J. (2024). "Revisiting Event-Study Designs: Robust and Efficient Estimation." Review of Economic Studies, 91(6), 3253-3285. https://academic.oup.com/restud/article/91/6/3253/7601390

[23] Goodman-Bacon, A. (2021). Full text PDF. https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Spring2024/Goodman-Bacon-2021-JoE-DID.pdf

[24] CSDID Documentation - What CS Can Do. https://d2cml-ai.github.io/csdid/examples/csdid_basic.html

[25] Chen Xing. "Notes on Callaway & Sant'Anna (2021) – Staggered Adoption DiD." https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did

[26] RPubs - Controls in DiD. https://rpubs.com/NickCHK/1031567

[27] Caetano, C., Callaway, B., Payne, S., & Sant'Anna, P.H.C. (2024). "Difference in Differences with Time-Varying Covariates." Working Paper. https://hsantanna.org/workingpapers/badcontrols

[28] Goldsmith-Pinkham, P. (2025). "Tracking the Credibility Revolution across Fields." https://paulgp.com/econlit-pipeline/paper.html

[29] Roth, J., Sant'Anna, P.H.C., Bilinski, A., & Poe, J. (2023). "What's trending in difference-in-differences? A synthesis of the recent econometrics literature." Journal of Econometrics, 235(2). https://www.jonathandroth.com/assets/files/DiD_Review_Paper.pdf

[30] Dahis, R. "DiD: Staggered adoption uses heterogeneity-robust estimator." Research Methods Checklist. https://www.ricardodahis.com/research-unit-tests/tests/did-staggered-heterogeneous-effects

[31] Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P.H.C. (2026). "Difference-in-Differences Designs: A Practitioner's Guide." Journal of Economic Literature, 64(2). https://www.aeaweb.org/articles?id=10.1257%2Fjel.20251650

[32] Cengiz, D., Dube, A., Lindner, A., & Zipperer, B. (2019). "The effect of minimum wages on low-wage jobs." Quarterly Journal of Economics, 134(3), 1405-1454. https://academic.oup.com/qje/article/134/3/1405/5420451

[33] Freedman, S.M. et al. (2023). "Designing Difference-in-Difference Studies With Staggered Treatment Adoption." NBER Working Paper 31842. https://www.nber.org/system/files/working_papers/w31842/w31842.pdf

[34] Lee, J. & Wooldridge, J.M. (2023). "Rolling Methods for Staggered DiD." Referenced in Chen Xing blog notes.

[35] Chen, J., Sant'Anna, P.H.C., & Xie, Z. (2024). "Efficient Difference-in-Differences and Event Study Estimators." Working Paper. https://psantanna.com/files/Efficient_DiD.pdf

[36] CLLX (2026). "Causal Panel Analysis under Parallel Trends: Lessons from a Large Reanalysis Study." American Political Science Review, 120(1). https://www.cambridge.org/core/journals/american-political-science-review/article/causal-panel-analysis-under-parallel-trends-lessons-from-a-large-reanalysis-study/219275E0CE901F099F2CFFBA07079243

[37] arXiv:2507.12891. "Refining the Notion of No Anticipation in Difference-in-Differences Studies." https://arxiv.org/abs/2507.12891

[38] IZA Discussion Paper (2022). "The German Minimum Wage: Anticipation and Pre-existing Trends." IZA.

[39] DiTraglia Lecture Notes - Anticipation Effects. https://ditraglia.com/lectures/anticipation.html

[40] Malani, A. & Reif, J. (2015). "Interpreting pre-trends as anticipation: Impact on estimated treatment effects from tort reform." Journal of Public Economics, 124, 1-17. NBER Working Paper w16593.

[41] Mertens, K. & Ravn, M.O. (2009, 2012). "Anticipated and Unanticipated Tax Shocks." Multiple publications.

[42] Callaway, B. & Sant'Anna, P.H.C. (2025). "Writing Extensions to the did Package" Vignette. https://bcallaway11.github.io/did/articles/extensions.html

[43] Rios-Avila, F., Callaway, B., & Sant'Anna, P.H.C. "csdid: Difference-in-Differences with Multiple Time Periods in Stata." https://www.stata.com/meeting/us21/slides/US21_SantAnna.pdf

[44] Borusyak, K., Jaravel, X., & Spiess, J. "did_imputation Stata Package." GitHub. https://github.com/borusyak/did_imputation

[45] Chen, J., Denteh, B., Kédagni, D., & Bilinski, A. (2024). "Anticipation Effects in Difference-in-Differences Models." Working Paper.

[46] bacondecomp R Package (CRAN). https://cran.r-project.org/web/packages/bacondecomp/bacondecomp.pdf

[47] BACONDECOMP Stata Module. https://econpapers.repec.org/RePEc:boc:bocode:s458676

[48] XJMR Discussion - Callaway-Sant'Anna Estimators. https://www.econjobrumors.com/topic/callaway-santanna-estimators

[49] Statalist Discussion - Differences Between csdid and did R Package. https://www.statalist.org/forums/forum/general-stata-discussion/general/1759587-differences-between-results-from-csdid-command-and-did-package-in-r

[50] Rios-Avila, F. "CSDID Version 1.6 Documentation." https://friosavila.github.io/playingwithstata/raw_articles/csdid_stata.html

[51] friosavila/csdid2 GitHub Repository. https://github.com/friosavila/csdid2

[52] GitHub Issue #287 - fixest: sunab Not Matching Stata Implementation. https://github.com/lrberge/fixest/issues/287

[53] didimputation R Package (CRAN). https://cran.r-project.org/web/packages/didimputation/didimputation.pdf

[54] borusyak/did_imputation GitHub. https://github.com/borusyak/did_imputation

[55] Stack Overflow - Error r(123) in did_imputation. https://stackoverflow.com/questions/error-with-did-imputation

[56] Statalist Discussion - Error did_imputation Borusyak (2021). https://www.statalist.org/forums/forum/general-stata-discussion/general/1670248-error-did-imputation-borusyak-2021

[57] did_multiplegt_dyn Stata Help File. https://fmwww.bc.edu/repec/bocode/d/did_multiplegt_dyn.sthlp

[58] DIDmultiplegtDYN R Package (CRAN). https://cran.r-project.org/web/packages/DIDmultiplegtDYN/index.html

[59] Tilburg Science Hub - Staggered DiD Guide. https://tilburgsciencehub.com/topics/causal-methods/difference-in-differences/staggered-did

[60] did R Package Documentation - att_gt Function. https://bcallaway11.github.io/did/reference/att_gt.html

[61] GitHub Issue #21 - friosavila/csdid_drdid: "Stopped by user" Bug. https://github.com/friosavila/csdid_drdid/issues/21

[62] sunab R Documentation - fixest Package. https://lrberge.github.io/fixest/reference/sunab.html

[63] Callaway, B. "Five Minute Summary: DiD with time-varying covariates." https://bcallaway11.github.io

[64] Stack Overflow - R fixest with 150M Observations. https://stackoverflow.com/questions/fixest-large-dataset

[65] EventStudyInteracts.jl Julia Package. https://juliapackages.com/p/eventstudyinteracts

[66] BJS GitHub - five_estimators_example.do. https://github.com/borusyak/did_imputation/blob/main/five_estimators_example.do