# Methodological Tensions in Staggered Difference-in-Differences Estimation: A Deepened Analysis of Implementation, Critiques, and Applied Adoption

## 1. Introduction

The past five years have witnessed a fundamental rethinking of difference-in-differences (DiD) methodology, driven by the recognition that traditional two-way fixed effects (TWFE) estimators can produce severely misleading results when treatment adoption is staggered and treatment effects are heterogeneous. The critique, crystallized in Goodman-Bacon's (2021) decomposition theorem, revealed that the TWFE estimator is a weighted average of many 2×2 DiD comparisons, some of which use already-treated units as controls—creating "forbidden comparisons" that can generate negative weights and bias estimates of any sign.

This report provides a comprehensive, deepened analysis of four leading approaches: **Callaway and Sant'Anna (2021)** , **Sun and Abraham (2021)** , **Borusyak, Jaravel, and Spiess (2024)** , and the diagnostic framework of **Goodman-Bacon (2021)** . It extends beyond conceptual overviews to systematically compare implementation details, explicitly link estimator features to methodological critiques, and provide enriched evidence on applied adoption patterns and reporting guidance in top economics journals for labor and health economics applications published 2020–2024. The report retains all valid content from prior analyses on heterogeneity handling, assumption comparisons, methodological dominance, and pre-trend testing, now integrated with three deeply revised areas of focus.

---

## 2. Implementation Details and Technical Distinctions Across Estimators

### 2.1 Anticipation Effects: Formal Parameters, Reference Period Selection, and Exclusion of Contaminated Observations

Anticipation effects—where units adjust their behavior before formal treatment begins—represent a critical threat to identification that each estimator addresses with distinct formal machinery.

#### 2.1.1 Callaway and Sant'Anna (2021): The Most Explicit Framework

Callaway and Sant'Anna provide the most elaborate treatment of anticipation through a formal parameter, δ (delta), in their "Limited Treatment Anticipation" assumption (Assumption 3). The assumption states: there exists a known δ ≥ 0 such that for all groups g and all time periods t occurring before g − δ, the expected untreated potential outcomes for treated units equal their actual outcomes. Mathematically: E[Y_t(g) | X, G_g = 1] = E[Y_t(0) | X, G_g = 1] for all t < g − δ.

**Default behavior and user specification:** The `did` R package's `att_gt()` function defaults to `anticipation = 0`, meaning no anticipation is allowed—the standard assumption that units cannot foresee and react to treatment before it occurs. The package documentation states that when δ = 0, the base period is the last untreated period (t = g − 1). When a researcher specifies `anticipation = 1`, the method allows anticipation exactly one period before treatment, and the base period shifts backward by that many periods to t = g − δ − 1 = g − 2 [Source: CRAN did package reference].

**Critical nuance with varying vs. universal base periods:** A documented issue on Economics Stack Exchange reveals that the `did` R package defaults to a "varying" reference period, which can obscure long-running linear pre-trends. The problem was "resolved when setting the reference period as 'universal' instead of 'varying', which is a problem when you have long-running linear pretrends" [Source: Economics Stack Exchange]. This distinction is crucial for applied researchers: the varying base period compares each cohort to its own immediate pre-treatment period, while the universal base period uses a common pre-treatment period across all cohorts. The choice can materially affect whether pre-trend violations are detected.

**Stata implementation differences:** In the Stata `csdid` package, the base period is handled differently. A Statalist post documents that "CSDID uses T-1 as the base period by default. You cannot choose other periods (for example allowing for anticipation), except by changing gvar manually" [Source: Statalist]. This means that researchers using Stata's `csdid` must manually manipulate the group variable to allow anticipation—a less user-friendly approach than the R implementation. The `csdid` help file confirms that for each treated group and period, csdid estimates ATTGT using "the last untreated period as the base and current period as post-treatment period" [Source: csdid Stata help file]. The `long2` option in Stata allows estimation using a longer pre-treatment gap, though this is not the default.

#### 2.1.2 Sun and Abraham (2021): No Explicit Anticipation Parameter

Sun and Abraham's interaction-weighted (IW) estimator incorporates the no-anticipation assumption as a core identifying condition but does **not** include an explicit anticipation parameter analogous to Callaway and Sant'Anna's δ. The MetricGate documentation states that "Identification rests on parallel trends across cohorts, no anticipation, and treatment-effect stability within a cohort across calendar time" [Source: MetricGate Sun-Abraham documentation].

In the `fixest::sunab()` function, which implements the Sun-Abraham estimator in R, there is no `anticipation` argument. Instead, the user controls which relative periods are included in the regression specification. To allow for anticipation, a researcher would include pre-treatment relative period dummies and interpret them as anticipation effects. However, the formal identification assumption remains that anticipation is zero—meaning Y_it(0) = Y_it for all pre-treatment periods.

The Stata `eventstudyinteract` command similarly has no explicit anticipation parameter [Source: eventstudyinteract Stata help file]. Users must manually include leads as controls if they wish to test for anticipation effects. This places a greater burden on the researcher to specify the anticipation structure correctly and may lead to inconsistent results if anticipation is present but not modeled.

#### 2.1.3 Borusyak, Jaravel, and Spiess (2024): Treatment Timing Exclusion Window

Borusyak, Jaravel, and Spiess handle anticipation through a formal "no anticipation" assumption (Assumption 2 in their paper): "No anticipation: Y_it = Y_it(0), ∀ it ∈ Ω_0" [Source: Borusyak, Jaravel & Spiess (2024), arXiv version 2108.12419v5]. The key implementation mechanism is the `shift()` parameter in the Stata `did_imputation` command, which "specify to allow for anticipation effects. The command will pretend that treatment happened `shift` periods earlier for each unit" [Source: did_imputation Stata help file].

For example, `shift(1)` means units are treated 1 period earlier for estimation purposes, effectively shifting the treatment timing window backward. This allows the researcher to exclude anticipation-affected periods from the counterfactual model. The `pretrend()` option specifies the number of pre-treatment periods to use for pre-trend testing—for example, `pretrend(10)` tests for pre-trends using 10 pre-treatment periods [Source: Asjad Naqvi did_imputation documentation].

The R `didimputation` package's `did_imputation()` function similarly includes options for `horizon` (event-study horizons) and `pretrends` (pre-trend periods), allowing specification of which periods to exclude from the estimation window [Source: R didimputation package documentation].

**Under-identification without never-treated units:** A critical finding from Borusyak et al. is that "fully dynamic specifications are under-identified when there are no never-treated units due to anticipation effects" [Source: Borusyak, Jaravel & Spiess (2024), Review of Economic Studies]. This means researchers using the imputation estimator must be careful about which pre-treatment periods are used for identification when no never-treated units exist.

#### 2.1.4 Goodman-Bacon (2021): No Anticipation Handling

The Goodman-Bacon decomposition does **not** explicitly model or allow for anticipation effects. The decomposition theorem takes the TWFE estimator as given and decomposes it into 2×2 DD components without any anticipation parameters [Source: Goodman-Bacon (2021), Journal of Econometrics]. The `bacondecomp` Stata and R commands implement the decomposition without any anticipation parameters; users must ensure the no-anticipation assumption holds in their setting before applying the decomposition [Source: bacondecomp documentation]. If anticipation effects exist, the TWFE estimate that Bacon decomposes would be contaminated, and the decomposition itself would not isolate anticipation bias from other sources of bias.

### 2.2 Weighting Schemes in Aggregation

The aggregation of heterogeneous treatment effects into summary parameters is a crucial design choice that differs substantially across estimators, with important implications for the interpretation of results.

#### 2.2.1 Callaway and Sant'Anna (2021): The `aggte()` Function

The `aggte()` function in the `did` R package provides four aggregation types, each with distinct weighting schemes:

1. **`"simple"`**: Computes a weighted average of all group-time average treatment effects (ATT(g,t)) with weights proportional to group size. The formula is θ̂simple = (1/κ) Σ_g Σ_t 1{g ≤ t} ATT(g,t) P(G=g | C ≠ 1) [Source: Callaway & Sant'Anna (2021) slides; NBER WP 34331].

2. **`"dynamic"`** (event-study type): Computes average effects across different lengths of exposure to treatment. The dynamic effect for exposure length e is: θ̂dynamic(e) = Σ_g 1{g+e ≤ T} ATT(g,g+e) P(G=g | G+e ≤ T, C ≠ 1) [Source: csdid Stata slides; `aggte` documentation].

3. **`"group"`** (default option): Computes average treatment effects across different groups (cohorts), with weights proportional to cohort size.

4. **`"calendar"`**: Computes average treatment effects across different time periods.

**Critical finding from Deb, Norton, Wooldridge & Zabel (2024):** NBER Working Paper No. 34331 documents a significant issue with how aggregation weights are implemented in standard software. The paper finds that "the software uses weights that include the number of observations in the reference pre-period instead of only the number of observations in the treated periods" [Source: Deb, Norton, Wooldridge & Zabel (2024), NBER WP 34331]. This leads to "a 16.2% difference in the overall ATET estimate" in their simulated repeated cross-sectional example, though "there is no difference when the data are a balanced panel" [Source: Deb, Norton, Wooldridge & Zabel (2024)].

The paper warns: "Given the enormous influence and popularity of the Callaway and Sant'Anna (2021) method... we are concerned that many published results have been calculated with a formula that is not what the researchers intended" [Source: Deb, Norton, Wooldridge & Zabel (2024)]. This represents a significant concern for the interpretation of published results using the CS estimator, particularly in studies with unbalanced panels.

**User-supplied weights:** The `att_gt()` function supports sampling weights via a `weights` argument, with the documentation stating that the function "supports time-varying sampling weights and different strategies for fixing weights across periods using the `fix_weights` argument" [Source: `att_gt` documentation]. The `balance_e` parameter in `aggte()` "balances the sample with respect to event time" by computing each dynamic effect using only groups that have been exposed for at least e periods [Source: `aggte` documentation].

#### 2.2.2 Sun and Abraham (2021): Cohort Share Weights

The Sun-Abraham interaction-weighted estimator proceeds in two stages: (1) estimate cohort-specific dynamic treatment effects (CATT(e,l)) from a saturated TWFE regression with cohort × relative time interactions, then (2) aggregate these effects using **cohort shares as weights** [Source: Sun & Abraham (2021); MetricGate Sun-Abraham documentation].

The weight for each (cohort, event time) pair is "the share of units in that event time who belong to that cohort" [Source: apithymaxim.wordpress.com replication]. The IW estimate for each relative time period r is:

ν̂_r = Σ_c β̂_c,r · ŝ_c,r

where ŝ_c,r is the sample share of cohort c among units observed at relative time r (i.e., the proportion of units at event time r that belong to cohort c) [Source: Sun & Abraham (2021), Section 3; MetricGate Sun-Abraham documentation].

The `fixest::sunab()` function automatically aggregates the cohort × relative period interaction coefficients to obtain the ATT for each relative period. The weights are based on the cohort share in the sample, which the fixest documentation notes is "considered to be a perfect measure for the cohort share in the population, contrary to the SA article" [Source: fixest sunab documentation]. The Stata `eventstudyinteract` command calculates "cohort shares underlying each relative time" and "take the weighted average of estimates with weights set to cohort shares" [Source: eventstudyinteract Stata help file].

**Non-negativity guarantee:** The key advantage of cohort-share weights is that they are non-negative and sum to one, ensuring the aggregated estimator is a convex combination of cohort-specific effects. This eliminates the negative weighting problem that plagues TWFE estimators.

#### 2.2.3 Borusyak, Jaravel, and Spiess (2024): Precision-Based Weights

The Borusyak et al. imputation estimator is described as efficient among linear, unbiased estimators. "Applying any weights to the imputed causal effects yields the efficient estimator"—the estimator uses weights that are proportional to the precision of the estimated treatment effects [Source: Borusyak, Jaravel & Spiess (2024), Review of Economic Studies].

The estimator proceeds in three stages: (1) estimate a model for non-treated potential outcomes using non-treated observations (typically a two-way fixed effects model); (2) impute non-treated potential outcomes for treated observations; (3) take averages of estimated treatment effects for the estimand of interest [Source: did_imputation Stata help file]. The `horizons()` option in Stata's `did_imputation` specifies the event time windows, returning coefficients (tau0 to tau10 for post-treatment, pre1 to pre10 for pre-treatment) [Source: Asjad Naqvi did_imputation documentation].

**User-supplied weights:** The Stata `did_imputation` command supports "estimation weights" via the standard `weights` option, and the command "allows saving of individual treatment effects and weights for further analysis, efficiency improvements via saved weights" [Source: did_imputation Stata help file]. The R `didimputation` package includes an `est_weights` parameter for specification of estimation weights [Source: R didimputation documentation].

**Efficiency advantage:** The precision-based weighting scheme is a key source of the estimator's claimed efficiency gains. The paper demonstrates "typically 30-360% reduction in standard deviation in simulations and application" [Source: Borusyak, Jaravel & Spiess (2024)].

#### 2.2.4 Goodman-Bacon (2021): Variance-Based Decomposition Weights

The Goodman-Bacon decomposition theorem states that "the two-way fixed effects estimator equals a weighted average of all possible two-group/two-period DD estimators in the data" [Source: Goodman-Bacon (2021), Abstract]. The weights are proportional to group sizes and the variance of the treatment dummy within each 2×2 pair. Critically, "weights depend on group sizes and timing relative to the panel and do not rely on outcome data" [Source: Stata bacondecomp slides].

The decomposition identifies four types of 2×2 comparisons: (1) early treated vs. untreated (clean), (2) late treated vs. untreated (clean), (3) early treated vs. late control (where late-treated units serve as controls for early-treated units before their treatment), and (4) late treated vs. early control (where early-treated units serve as controls for late-treated units—the "forbidden comparison" that can introduce negative weights) [Source: Goodman-Bacon (2021), Section 3; Asjad Naqvi bacon-decomp documentation].

**Negative weight condition:** "Negative weights only arise when average treatment effects vary over time" [Source: Goodman-Bacon (2021)]. The decomposition reveals which comparisons drive the TWFE estimate and whether negative weights are a concern.

### 2.3 Equivalence to Alternative Estimation Procedures

Understanding the relationships between these estimators and simpler alternatives is crucial for applied researchers seeking to implement methodologically sound analyses.

#### 2.3.1 Callaway and Sant'Anna (2021) and Stacked DiD

The Callaway and Sant'Anna estimator is conceptually equivalent to a **stacked DiD** approach where, for each treatment cohort g, one creates a sub-dataset containing: (i) units first treated at time g, and (ii) a comparison group (never-treated or not-yet-treated units). Each sub-dataset covers time periods relative to cohort g, and cohort-by-time fixed effects are included [Source: csdid Stata documentation].

However, the formal mathematical equivalence is more nuanced. Callaway and Sant'Anna do not use a single stacked regression. Instead, they compute each ATT(g,t) separately using long-difference estimands:

ATT_unc^nev(g,t) = E[Y_t − Y_{g−1} | G_g = 1] − E[Y_t − Y_{g−1} | C = 1] (using never-treated)

ATT_unc^ny(g,t) = E[Y_t − Y_{g−1} | G_g = 1] − E[Y_t − Y_{g−1} | D_t = 0, G_g = 0] (using not-yet-treated)

[Source: csdid Stata slides, pp. 11-12]

The equivalence to stacked DiD with cohort-by-time fixed effects holds **only when covariates are absent and the never-treated group is used**. When covariates are included, the doubly robust estimator diverges from simple stacked regressions because the propensity score weighting and outcome regression adjustment are not captured by a single stacked regression specification.

#### 2.3.2 Borusyak, Jaravel, and Spiess (2024) Imputation and Gardner (2021) Two-Stage DiD

The Borusyak et al. imputation estimator is mathematically equivalent to the **two-stage estimator of Gardner (2021)** [Source: Gardner (2021), "Two-stage differences in differences"]. Gardner's framework proceeds as:

**Stage 1**: Estimate unit FE (α_i) and time FE (β_t) using only untreated/not-yet-treated observations: Y_it = α_i + β_t + ε_it, for it ∈ Ω_0

**Stage 2**: For treated observations, compute the treatment effect as: τ̂_it = Y_it − α̂_i − β̂_t. Then aggregate these τ̂_it to obtain the desired summary measure.

This is exactly the imputation approach: "The imputation-based estimator... estimates a model for Y(0) using untreated/not-yet-treated observations and predicts Y(0) for the treated observations ĥat(Y_it(0)); the difference Y_it(1) − ĥat(Y_it(0)) serves as an estimate for the treatment effect for unit i in period t" [Source: R didimputation documentation].

The `did2s` Stata package (Kyle Butts) implements Gardner's two-stage procedure explicitly, while the `did_imputation` approach differs slightly in how it handles the first-stage estimation (e.g., use of treatment effect parameterization in BJS rather than explicit two-stage). However, the fundamental principle is identical: both approaches estimate the untreated counterfactual model from untreated observations only, then impute counterfactuals for treated units.

#### 2.3.3 Sun and Abraham (2021) and Cohort × Relative Time Interactions

The Sun and Abraham interaction-weighted estimator is equivalent to: (1) estimating a saturated regression that interacts **cohort indicators** with **relative time dummies** (while including unit and time fixed effects), and then (2) taking a weighted average of the cohort-specific coefficients using cohort shares as weights [Source: Sun & Abraham (2021); MetricGate Sun-Abraham documentation].

Let D_{i,t}^{c,r} be an indicator for unit i belonging to cohort c and being at relative time r = t − g_c. The saturated TWFE regression is:

Y_it = α_i + λ_t + Σ_c Σ_{r ≠ −1} β_{c,r} · D_{i,t}^{c,r} + ε_it

The IW estimator for relative period r is then: ν̂_r = Σ_c β̂_{c,r} · ŝ_{c,r}, where ŝ_{c,r} = N_{c,r} / Σ_{c'} N_{c',r} and N_{c,r} is the number of units in cohort c observed at relative time r [Source: Sun & Abraham (2021), Section 3.2].

**Equivalence to Callaway and Sant'Anna (when no covariates):** The `fixest` GitHub issue #287 documents: "Sun and Abraham with a never-treated group, the correct reference periods being dropped, and no covariates is numerically equivalent to Callaway and Sant'Anna" [Source: GitHub fixest Issue #287]. The same issue notes that "did matches with the Stata Sun and Abraham implementation." However, when covariates are included, the two estimators diverge because Callaway and Sant'Anna uses doubly robust/propensity score methods while Sun and Abraham uses regression-based adjustment [Source: GitHub fixest Issue #287].

### 2.4 Software Implementation Choices That Affect Results

The specific software implementation details—including default comparison groups, bandwidth/trimming parameters, and covariate handling—can materially affect results in ways that applied researchers must understand.

#### 2.4.1 Default Comparison Groups Across Packages

| Package/Language | Default Comparison Group | Alternative Options |
|-----------------|------------------------|-------------------|
| `did` (R) | `control_group = "nevertreated"` | `"notyettreated"` |
| `csdid` (Stata) | Never-treated | `notyet` option |
| `xthdidregress` (Stata 18) | `control(nevertreated)` | `control(notyettreated)` |
| `fixest::sunab()` (R) | Never-treated cohorts as reference | `ref.c` and `ref.p` arguments |
| `eventstudyinteract` (Stata) | Never-treated or last-treated | `control_cohort()` option |
| `did_imputation` (Stata/R) | All untreated observations (never-treated + not-yet-treated) | No alternative (inherent to method) |

**Source:** [att_gt documentation](https://bcallaway11.github.io/did/reference/att_gt.html), [csdid Stata slides](https://www.stata.com/meeting/us21/slides/US21_SantAnna.pdf), [Stata xthdidregress manual](https://www.stata.com/manuals/causalxthdidregress.pdf), [sunab documentation](https://lrberge.github.io/fixest/reference/sunab.html), [eventstudyinteract Stata help file](http://fmwww.bc.edu/repec/bocode/e/eventstudyinteract.sthlp), [did_imputation Stata help file](https://github.com/borusyak/did_imputation)

#### 2.4.2 Stata vs. R Discrepancies: The Scott Cunningham (2026) Audit

Scott Cunningham conducted a comprehensive cross-package audit of six implementations of the Callaway & Sant'Anna (2021) doubly robust (DR) estimator across R, Stata, and Python [Source: Scott Cunningham, "Six Packages, One Estimator" (2026)]. Key findings:

- **Despite identical specifications, the six software packages yielded notably divergent ATT estimates when covariates were included, sometimes differing by up to 2.5 times.**
- **44% of total variation** in estimates arises from the interaction between package and specification choice.
- **Package choice accounts for 16% of variation** in estimates across all specifications.
- The divergence primarily arises from numerical implementation issues—especially handling near-singular matrices during the inversion of design matrices.
- Covariate scaling issues (e.g., variables on the order of 10 billion) cause numerical instability.
- The `did` package may handle near-separation in logistic regressions differently than other packages.
- Some packages silently return zero or NaN estimates; others revert to fallback methods without warnings.
- **Recommendation:** Standardize covariates via z-scoring to eliminate numerical singularity.

This audit has profound implications for applied researchers: the choice of software package is not a neutral implementation decision but can substantively affect results. The audit recommends that researchers standardize covariates and report results from multiple software packages to assess sensitivity.

#### 2.4.3 Known Discrepancies Between R `did` and Stata `csdid`

Beyond the Cunningham audit, several specific discrepancies between the R and Stata implementations of the Callaway-Sant'Anna estimator are documented:

1. **Default base period**: "By default, Stata uses a 'short-gap-varying base'. You can re-estimate using the long2 option in the csdid command." The R package can use either universal or varying base periods [Source: Economics Stack Exchange].

2. **Default estimation method**: The R `did` package default is `"dr"` (doubly robust). The Stata `csdid` default is `"drimp"` (improved doubly robust) [Source: csdid Stata slides, p. 16; att_gt documentation]. Both use the same underlying DRDID methodology but differ in implementation details.

3. **Variance calculation**: CSDID Version 1.6 "replaced 'nlcom' for variance calculations, which was slow" with "matrix multiplications and the sandwich formula for improved efficiency" [Source: CSDID Version 1.6, Fernando Rios-Avila].

4. **`fixest::sunab()` vs. Stata `eventstudyinteract`**: GitHub Issue #287 documents that the fixest implementation can produce results that differ from the canonical Sun and Abraham implementation. The `did` package "matches with the Stata Sun and Abraham implementation" [Source: GitHub fixest Issue #287].

5. **`xthdidregress` vs. `csdid`**: The official Stata 18 command `xthdidregress` includes an extended TWFE estimator (Wooldridge, 2021) that is not in `csdid`. Both implement the Callaway-Sant'Anna methodology, but `xthdidregress` benefits from official support and integration with Stata's ecosystem [Source: Statalist discussion].

#### 2.4.4 Covariate Handling Across Packages

**`did` R Package:** The `est_method` parameter accepts `"dr"` (doubly robust—default), `"ipw"` (inverse probability weighting), or `"reg"` (outcome regression). When `est_method = "dr"`, covariates enter via a doubly robust formula that combines propensity score weighting and outcome regression adjustment. The doubly robust estimator is consistent if either the propensity score model or the outcome regression model is correctly specified. Covariates are assumed to be pre-treatment and time-constant [Source: csdid Stata slides, pp. 12-13; csdid Stata help file].

**`csdid` (Stata):** The `method()` option accepts `"drimp"` (default, improved doubly robust), `"dripw"` (doubly robust IPW), `"reg"` (outcome regression), `"ipw"` (stabilized IPW). "The underlying assumption is that all covariates are time constant" [Source: csdid Stata help file].

**`xthdidregress` (Stata 18):** Offers four estimators: TWFE (extended two-way fixed effects), RA (regression adjustment), IPW (inverse-probability weighting), and AIPW (augmented inverse-probability weighting, which has double-robustness property). "The AIPW models both treatment and outcome. If at least one of the models is correctly specified, it provides consistent estimates" [Source: Stata xthdidregress manual].

**`fixest::sunab()` (R):** Covariates can be added directly to the `feols()` formula (e.g., `feols(y ~ sunab(cohort, period) + x1 + x2 | id + period)`). They enter linearly in the regression. No propensity score or IPW adjustment for covariates [Source: sunab documentation].

**`did_imputation` (Stata):** "Supports inclusion of fixed effects, continuous time-varying and interacted controls" [Source: did_imputation Stata help file]. The first stage uses a TWFE regression of the outcome on unit and time fixed effects, possibly including covariates.

#### 2.4.5 Bandwidth/Trimming Parameters and Bootstrapping Defaults

**`did` R Package:**
- `bw`: Used in bootstrap standard error computation (multiplicity correction for uniform confidence bands) [Source: `aggte` documentation]
- `biters`: Number of bootstrap iterations (default = 1000) [Source: `att_gt` documentation]
- `cband`: Boolean for whether to compute uniform confidence bands (TRUE) or pointwise (FALSE) [Source: `att_gt` documentation]
- `bstrap`: Boolean for whether to compute standard errors using the multiplier bootstrap (default = FALSE in `att_gt`, meaning analytical standard errors are used by default) [Source: `att_gt` documentation]
- `min_e` and `max_e`: Minimum and maximum event times in `aggte()` [Source: `aggte` documentation]

**`did_imputation` (Stata):**
- `minn(0)`: Minimum number of observations per cohort [Source: Asjad Naqvi did_imputation documentation]
- `horizons(0/10)`: Set event study horizons [Source: Asjad Naqvi did_imputation documentation]
- `pretrend(10)`: Number of pre-treatment periods for pre-trend testing [Source: Asjad Naqvi did_imputation documentation]
- `truncate`: Option for automatic sample trimming [Source: did_imputation Stata help file]

**`fixest::sunab()`:**
- `bin`, `bin.c`, `bin.p`, `bin.rel`: Binning/grouping of cohorts and periods [Source: sunab documentation]
- `ref.c` and `ref.p`: Reference cohort and reference period [Source: sunab documentation]

**`xthdidregress` (Stata):**
- `base`: "adaptive" (default) or "common" base time. "Adaptive base time chooses the earliest pre-treatment period for each cohort" while "common base time uses a single pre-treatment period for all cohorts" [Source: Stata xthdidregress manual]
- `hettype()`: Specifies heterogeneity type—"over cohort only, time only, or both" (to reduce model complexity) [Source: Stata xthdidregress manual]

---

## 3. Explicit Links Between Estimator Features and Methodological Critiques

### 3.1 Clean Control Selection and the "Forbidden Comparisons" Problem

The "forbidden comparisons" problem, identified by Goodman-Bacon (2021), arises because the TWFE estimator implicitly uses already-treated units as controls for later-treated units. When treatment effects vary over time, these comparisons produce negative weights that can reverse the sign of estimated effects.

**How each estimator addresses this:**

**Callaway and Sant'Anna (2021):** The estimator explicitly restricts comparison units to either never-treated or not-yet-treated groups. By construction, each ATT(g,t) compares the outcome evolution of units first treated in period g at time t to units that are either never-treated or not-yet-treated at time t. This ensures that already-treated units are never used as comparisons, so their dynamic treatment effects cannot contaminate the counterfactual [Source: Callaway & Sant'Anna (2021), Section 4; csdid Stata documentation]. The formal condition is that the identification strategy "does not restrict observed pre-treatment trends across groups" under Assumption 4 (never-treated) but uses only "clean" comparisons throughout [Source: Callaway & Sant'Anna (2021)].

**Sun and Abraham (2021):** The IW estimator estimates cohort-specific effects separately for each adoption cohort, comparing each cohort only to never-treated or last-treated units. The estimator "requires an identifiable clean comparison group consisting of never-treated or last-treated units" [Source: MetricGate Sun-Abraham documentation]. This ensures that the counterfactual for a given cohort at a given relative time is constructed only from units that are not yet treated at that point.

**Borusyak, Jaravel, and Spiess (2024):** The imputation estimator estimates a model for untreated potential outcomes Y(0) using only untreated/not-yet-treated observations. For treated units, the counterfactual is imputed from this model (which is identified solely from untreated observations), so the estimated treatment effect Y_it(1) − ĥat(Y_it(0)) reflects only the difference between the actual treated outcome and the imputed untreated counterfactual. No already-treated observations enter the counterfactual estimation [Source: Borusyak, Jaravel & Spiess (2024), Section 3; did_imputation Stata help file].

**Goodman-Bacon (2021):** The decomposition does not itself avoid forbidden comparisons but rather diagnoses their presence. The decomposition reveals "how much of the identifying variation comes from comparisons between groups that do not change treatment status" versus "comparisons between groups changing treatment status" [Source: Goodman-Bacon (2021), Section 3]. This diagnostic allows researchers to assess whether the TWFE estimate is likely biased by forbidden comparisons.

### 3.2 Untreated-Only Estimation of the Counterfactual and Contamination Bias

Contamination bias occurs when the counterfactual for treated units is constructed using observations from other treated units—causing dynamic treatment effects to be subtracted from the DiD estimate.

**TWFE contamination mechanism:** Goodman-Bacon (2021) shows that "when already-treated units act as controls, changes in their treatment effects over time get subtracted from the DID estimate, typically biasing estimates away from the sign of the true treatment effect" [Source: Goodman-Bacon (2021); GCcollab Wiki]. The combination of staggered treatment timing and dynamic treatment effects accentuates the presence of "bad comparisons," causing potentially severe biases in TWFE DiD estimates.

**How each estimator avoids contamination:**

**Callaway and Sant'Anna (2021):** By computing each ATT(g,t) from a clean 2×2 comparison that never uses already-treated units, contamination bias is eliminated by construction. The only comparisons used are between treated units and never-treated or not-yet-treated units, ensuring the counterfactual reflects untreated potential outcomes uncontaminated by treatment effects from other cohorts [Source: Callaway & Sant'Anna (2021), Section 4].

**Sun and Abraham (2021):** The IW estimator estimates cohort-specific effects separately, then aggregates them with non-negative cohort-share weights. Because the counterfactual for each cohort at each relative time is constructed only from units that are not yet treated at that point, dynamic treatment effects from other cohorts cannot contaminate the estimates [Source: Sun & Abraham (2021), Section 3].

**Borusyak et al. (2024):** The imputation estimator avoids contamination by separating the estimation of the untreated counterfactual model (using only untreated observations) from the calculation of treatment effects (comparing actual outcomes to imputed counterfactuals). The first-stage model is estimated on observations where D_it = 0, ensuring that no treatment effects enter the counterfactual estimation [Source: Borusyak, Jaravel & Spiess (2024), Section 3].

### 3.3 Cohort-Specific Estimation and the Negative-Weighting Problem

The negative-weighting problem, identified by de Chaisemartin and D'Haultfœuille (2020) and Goodman-Bacon (2021), occurs because the TWFE estimator implicitly assigns weights to different 2×2 comparisons in ways that can be negative—meaning a positive treatment effect can appear negative in the aggregate.

**De Chaisemartin and D'Haultfœuille (2020):** "Equation (2.3) implies that some of the weights W_g,t may be negative, so the estimated coefficient could be positive while the minimum wage's effect on employment is negative both in Santa Clara and in Wayne county" [Source: de Chaisemartin & D'Haultfœuille (2020), American Economic Review].

**How each estimator addresses negative weights:**

**Callaway and Sant'Anna (2021):** The group-time ATT approach computes each ATT(g,t) separately from clean 2×2 comparisons. Each ATT(g,t) is a well-defined causal parameter estimated from a valid comparison, and there is no weighting across different (g,t) cells at the estimation stage. Aggregation of group-time ATTs applies non-negative weights to the group-time ATTs, ensuring interpretable summary measures [Source: Callaway & Sant'Anna (2021), Section 5].

**Sun and Abraham (2021):** The IW estimator directly targets the negative weighting problem. Standard TWFE event-study regressions implicitly assign weights that can be negative when treatment effects are heterogeneous across adoption cohorts. "The Sun-Abraham estimator reconstructs the TWFE estimator as a sum of cohort-by-time effects with non-negative cohort-share weights, eliminating the sign-reversal and bias due to treatment effect heterogeneity" [Source: Sun & Abraham (2021); MetricGate Sun-Abraham documentation]. Since cohort shares are proportions (non-negative and sum to one), the aggregated estimator is guaranteed to be a convex combination of cohort-specific effects.

**Borusyak et al. (2024):** The imputation approach side-steps the weighting problem entirely. Instead of comparing treated and control units in a regression framework that implicitly weights different comparisons, the imputation estimator first estimates the untreated outcome model from untreated observations only, then imputes counterfactuals for treated units. "Negative weighting arising in static TWFE regressions implies that long-run treatment effects may enter estimators with negative weights, biasing estimates. The imputation approach avoids this because it does not rely on any regression-based comparison between treated and control units; instead it directly constructs the counterfactual for each treated unit" [Source: Borusyak, Jaravel & Spiess (2024), Section 3].

### 3.4 Pre-Trend Testing Contamination and Roth (2022)

Roth (2022) identifies two fundamental problems with conventional pre-trend testing: (1) low power—"conventional pre-trends tests may have low power, meaning that preexisting trends that produce meaningful bias in the treatment effects estimates may not be detected with substantial probability" [Source: Roth (2022), American Economic Review: Insights]; and (2) pretesting bias—"conditioning the analysis on the result of a pretest induces distortions to estimation and inference from pretesting... the bias caused by a violation of parallel trends can actually be worse conditional on passing the pretest" [Source: Roth (2022)].

**The contamination problem in TWFE event studies:** Sun and Abraham (2021) show that "the coefficient on a given lead or lag can be contaminated by effects from other periods, and apparent pretrends can arise solely from treatment effects heterogeneity" [Source: Sun & Abraham (2021), Journal of Econometrics]. This means standard event-study plots from TWFE regressions may show apparent violations of parallel trends even when there are no differential pre-trends, simply because of treatment effect heterogeneity across cohorts contaminating the pre-treatment coefficients.

**How each heterogeneity-robust estimator produces "clean" pre-treatment coefficients:**

**Sun and Abraham (2021):** The IW estimates for negative relative time periods (pre-treatment) serve as a cleaner parallel trends diagnostic compared to standard TWFE models, "free from contamination bias" [Source: Sun & Abraham (2021)]. The IW estimator estimates each pre-treatment coefficient separately using only not-yet-treated units as comparisons, so there is no possibility of post-treatment contamination of pre-treatment coefficients.

**Callaway and Sant'Anna (2021):** For pre-treatment periods, the estimator compares the outcome change for a given group to the outcome change for never-treated or not-yet-treated units over the same period. Since the comparison units are untreated at that time, the pre-treatment coefficients reflect only differential pre-trends, not contamination from treatment effects. The `did` R package provides dedicated pre-testing functionality through the `conditional_did_pretest` function, which tests the conditional parallel trends assumption when covariates are included [Source: did R package vignette].

**Borusyak et al. (2024):** The imputation approach estimates the untreated potential outcomes model from all untreated observations and tests for pre-trends by examining whether pre-treatment differences between actual outcomes and imputed counterfactuals are systematically different from zero. Because the imputation model is estimated only from untreated data, these pre-treatment differences are clean measures of differential pre-trends, not contaminated by treatment effects from other cohorts [Source: Borusyak, Jaravel & Spiess (2024), Section 3].

### 3.5 Conditional Parallel Trends and Statistical Power

Callaway and Sant'Anna (2021) provide a framework that allows for covariate adjustment to address non-parallel trends due to observed characteristics. The formal conditional parallel trends assumption (Assumption 4) states: for each g ∈ G and t ∈ {2,...,T} such that t ≥ g − δ, E[Y_t(0) − Y_{t−1}(0) | X, G_g=1] = E[Y_t(0) − Y_{t−1}(0) | X, C=1] a.s. This states that, conditional on covariates, average outcomes for the treated group and never-treated group would have followed parallel paths [Source: Callaway & Sant'Anna (2021), Section 3].

**Why conditioning on covariates addresses Roth's (2022) concerns about low power:** Roth's critique focuses on the low power of unconditional parallel trends tests. By allowing the researcher to incorporate covariates that predict treatment assignment or outcome dynamics, the conditional parallel trends assumption may be more plausible than the unconditional version. This means there are fewer violations to detect—addressing the power concern directly. Additionally, incorporating covariates can reduce residual variance, increasing the precision of pre-treatment estimates and thereby increasing power to detect violations [Source: Callaway & Sant'Anna (2021), Section 5; Roth (2022)].

The doubly-robust estimator provides additional robustness to model misspecification: consistent estimation holds if either the propensity score model or the outcome regression model is correctly specified. This means the researcher is protected against misspecification of the conditional parallel trends assumption in a way that unconditional approaches are not.

### 3.6 Imputation Approach Sensitivity to Multi-Period Violations

The Borusyak et al. (2024) imputation estimator uses ALL pre-treatment periods simultaneously to estimate unit and time fixed effects. The first stage estimates a model for untreated potential outcomes that includes unit fixed effects and time fixed effects, identified from all person-time observations where the unit is not yet treated [Source: Borusyak, Jaravel & Spiess (2024), Section 3].

**Sensitivity to violations in any single pre-treatment period:** Because the imputation approach uses all pre-treatment periods jointly to estimate the fixed effects, a parallel trends violation in any single pre-treatment period can affect the estimated fixed effects for all units, and thus potentially contaminate the imputed counterfactuals for ALL post-treatment periods. Specifically, if there is a deviation from parallel trends in pre-treatment period t*, this deviation will be absorbed into the estimated unit fixed effects and/or time fixed effects. Since these fixed effects are then used to impute the counterfactual for all post-treatment periods, the violation in a single pre-treatment period can affect treatment effect estimates at all horizons [Source: Borusyak, Jaravel & Spiess (2024)].

**Comparison to event-study approaches:** In contrast, event-study approaches (such as Sun & Abraham 2021 or Callaway & Sant'Anna 2021) estimate each pre-treatment coefficient separately. If there is a parallel trends violation in pre-treatment period t*, this will be reflected in the coefficient for that specific pre-treatment period, but it will not directly contaminate the estimated coefficients for other pre-treatment periods or post-treatment periods [Source: Sun & Abraham (2021); Callaway & Sant'Anna (2021)]. The event-study approach effectively allows each pre-treatment period to have its own deviation from the counterfactual, so violations in one period do not mechanically propagate to other periods. However, this comes at the cost of estimating many parameters separately, which can reduce precision.

### 3.7 The Roth (2026) Critique of Asymmetric Event-Study Construction

Jonathan Roth's (2026) paper "Interpreting Event-Studies from Recent Difference-in-Differences Methods" identifies a critical and previously underappreciated issue: the default event-study plots produced by modern heterogeneity-robust estimators are constructed asymmetrically for pre- and post-treatment periods, leading to potential misinterpretation [Source: Roth (2026), Japanese Economic Review].

**Key finding:** "The default plots produced by software for several of the most popular recent methods do not match those of traditional two-way fixed effects (TWFE) event-studies. These new methods construct the pre-treatment coefficients asymmetrically from the post-treatment coefficients" [Source: Roth (2026)].

**The "kink" or "jump" at treatment time:** Roth demonstrates that in a simple non-staggered treatment timing simulation with no treatment effect but violated parallel trends, event-study plots from Callaway-Sant'Anna, de Chaisemartin-D'Haultfœuille, and Borusyak et al. all show kinks or jumps at the treatment time, unlike the TWFE event-study which shows a smooth linear trend. "A kink or jump at the time of treatment may arise even if there is no treatment effect and parallel trends is equally violated in all periods" [Source: Roth (2026)].

**Mathematical explanation:** In the CS method, post-treatment coefficients use a "long-difference" (comparing outcomes in period t to outcomes in the period just before treatment), while pre-treatment coefficients use a "short-gap" (comparing outcomes in adjacent periods). This asymmetry creates the visual kink. For BJS, "the BJS approach inherently uses all pre-treatment periods to estimate counterfactual outcomes, leading to an asymmetry between pre- and post-treatment estimates" [Source: Roth (2026)].

**Implication for applied research:** "The typical heuristics for visual inference developed based on dynamic TWFE specifications will thus be misleading when applied to these new estimators" [Source: Roth (2026)]. Roth recommends: (1) for CS, use long-differences for pre-treatment coefficients as well as post-treatment coefficients; (2) for BJS, put pre-treatment estimates on a separate plot from post-treatment estimates; (3) supplement event-studies with cohort-specific time series and balanced event studies.

---

## 4. Enriched Synthesis of Applied Adoption Patterns and Reporting Guidance

### 4.1 Systematic Evidence on Adoption Rates in Top Economics Journals (2020-2024)

**Note:** No single published audit tracks the exact share of labor and health economics papers in the top-five journals (AER, QJE, JPE, Econometrica, REStud) using each specific robust estimator for the 2020-2024 period. What is available is field-level evidence from several large-scale studies and citation-based proxies for estimator adoption.

#### 4.1.1 Goldsmith-Pinkham (2026): The Credibility Revolution Across Fields

Paul Goldsmith-Pinkham's (2026) NBER Working Paper No. 35051 provides the most comprehensive evidence on the diffusion of quasi-experimental methods. Analyzing approximately 44,000 papers—including 31,500 NBER working papers (1982-2025) and 12,300 articles from top economics and finance journals (2011-2024)—the paper finds:

- **As of 2024, 63% of applied micro papers** mention experimental or quasi-experimental methods, compared to 47% in finance and 39% in macro/other fields [Source: Goldsmith-Pinkham (2026), NBER WP 35051].
- Finance and macro/other fields are at levels comparable to where applied micro stood in 2008-2010, suggesting significant room for further adoption.
- **"The credibility revolution outside applied micro has been—to a first approximation—a difference-in-differences revolution"** [Source: Goldsmith-Pinkham (2026)].
- Including DiD raises the share of finance papers mentioning any experimental or quasi-experimental method by roughly 55 percent, versus 30 percent for applied micro.
- Published articles from top journals show trends that "closely mirror the NBER data, with slightly higher rates of credibility revolution methods—consistent with a publication selection effect favoring methodologically rigorous papers" [Source: Goldsmith-Pinkham (2026)].
- "The rare instances where econometric theory and applied literatures intersect have been extraordinarily productive, changing how researchers implement common methods in under five years" [Source: Goldsmith-Pinkham (2026)].

**Important caveat:** The Goldsmith-Pinkham paper uses broad keyword matching for "difference-in-differences" and "event study"—it does **not** provide a breakdown of adoption by specific robust DiD estimator. It confirms the overall dominance of DiD methods but not the distribution across estimators.

#### 4.1.2 Citation-Based Evidence for Specific Estimator Adoption

Citation counts (from Google Scholar as of 2025/2026) provide the best available proxy for relative adoption rates:

- **Callaway & Sant'Anna (2021)** – "Difference-in-Differences with Multiple Time Periods" – ~12,112 citations [Source: Callaway Google Scholar]. This is the most cited modern DiD estimator by a wide margin, suggesting it is the most widely used approach.
- **Borusyak, Jaravel & Spiess (2024)** – "Revisiting Event-Study Designs" – Published in Review of Economic Studies, Vol. 91, Issue 6, November 2024 – ~5,149 citations [Source: Borusyak Google Scholar]. This is the most rapidly gaining recent estimator.
- **Goodman-Bacon (2021)** – "Difference-in-differences with variation in treatment timing" – Journal of Econometrics. Scott Cunningham (2025) reports that this paper has "passed the original synthetic control article in total citations" [Source: Scott Cunningham, Mixtape Substack].
- **Sun & Abraham (2021)** – "Estimating Dynamic Treatment Effects in Event Studies" – Journal of Econometrics. Widely cited as the key reference for why TWFE event-study coefficients are contaminated by treatment effect heterogeneity.
- **Baker, Larcker & Wang (2022)** – "How Much Should We Trust Staggered Difference-In-Differences Estimates?" – Journal of Financial Economics – ~1,800+ citations. The key replication/audit paper demonstrating the magnitude of TWFE bias.
- **Roth, Sant'Anna, Bilinski & Poe (2023)** – "What's trending in difference-in-differences?" – Journal of Econometrics – ~3,177 citations [Source: Roth Google Scholar].

#### 4.1.3 Evidence from Published Application Papers

**Labor Economics:**
- Cengiz, Dube, Lindner & Zipperer (2019, QJE) used a "stacked DiD" design examining wage bins rather than aggregate employment, finding significant employment effects. This influential paper anticipated the heterogeneity-robust literature and is frequently cited alongside the newer methods. Their data is used as an empirical reference in Callaway and Sant'Anna's software documentation.
- A paper on "Long-term employment effects of the minimum wage in Germany" (Journal of Comparative Economics, 2024) explicitly states: "We use the estimators developed by Sun and Abraham (2021), Callaway and Sant'Anna (2021), Borusyak et al. (2021), and De Chaisemartin and d'Haultfoeuille (2022a)"—demonstrating the practice of reporting **all four estimators** as robustness checks.
- Dube, Girardi, Jordà & Taylor (2023) propose "LP-DiD"—a local projections approach to DiD event studies that "addresses the bias of conventional fixed effects estimators, leading to potentially different results" [Source: Dube et al. (2023)].

**Health Economics:**
- A study titled "Labor and Coverage Effects of Medicaid Expansion: A Callaway–Sant'Anna Approach to Staggered Adoption" finds "Medicaid expansion led to substantial gains in public insurance coverage" while "labor market outcomes...remained largely unchanged" [Source: Khosravi (2025)].
- A study revisiting the ACA Medicaid expansion on interstate migration "employing difference-in-differences (DiD) and advanced staggered DiD methodologies to account for varying state expansion timing" found "evidence of increased migration from non-expansion-to-expansion states" [Source: ACA Medicaid expansion migration study].
- Wang, Hamad & White (2024), "Advances in Difference-in-differences Methods for Policy Evaluation Research" (Epidemiology), provides a comprehensive review stating: "Two-way fixed effects DiD designs may generate biased DiD estimators under heterogeneous treatment effects, that is, when effects vary across groups or time" [Source: Wang, Hamad & White (2024)].

### 4.2 Field-Specific Adoption Rates: Labor vs. Health Economics

#### 4.2.1 Labor Economics

**Dominant patterns:** Labor economics shows the most diverse set of estimator choices. While Callaway-Sant'Anna and Sun-Abraham are widely used, several other approaches are also common:

- **Stacked DiD** remains popular for its transparency and simplicity, especially in settings with many cohorts (e.g., minimum wage studies).
- **LP-DiD (local projections)** has been proposed by Dube et al. (2023) specifically for labor applications with dynamic heterogeneous treatment effects.
- **Event-study plots** have been standard in labor economics for decades, with Miller (2023, JEP) noting that "one of its most appealing features is that it creates a built-in graphical summary of results" [Source: Miller (2023), Journal of Economic Perspectives].
- **The minimum wage literature** has been a frontier for methodological innovation, with Cengiz et al. (2019, QJE) using clean-controls approaches before the formal methods literature.

**Adoption drivers:** The labor field's long history of using event-study plots means that Sun-Abraham's "drop-in replacement" for TWFE event studies has been particularly attractive. However, the field also values covariate adjustment, driving use of Callaway-Sant'Anna in some applications.

#### 4.2.2 Health Economics

**Dominant patterns:** Callaway & Sant'Anna is the dominant robust estimator in health economics. The covariate adjustment feature is particularly valued because health policy evaluations often have rich covariate data:

- "Callaway and Sant'Anna (2021) discussed three different strategies to account for covariates: regression adjustment, inverse-probability weighting, doubly robust" [Source: Callaway & Sant'Anna (2021)].
- "Adding covariates linearly into TWFE will not give you the ATT(g, t)'s, alternative methods such as doubly robust procedures should be used" [Source: Chen Xing, Notes on Callaway & Sant'Anna (2021)].
- **ACA Medicaid expansion studies** are the most common application of robust DiD in health economics. A PDHP presentation (2023) states: "Example: Effect of ACA Medicaid Expansion on Health Insurance rate... In what follows, I will focus on Callaway and Sant'Anna (2021)'s approach" [Source: PDHP Presentation (2023)].
- Riddell & Goin (2023) in Epidemiology provide a "Guide for comparing estimators of policy change effects on health" recommending "Group-Time ATT, Cohort ATT, and Target Trial estimators" as alternatives to TWFE [Source: Riddell & Goin (2023)].

**Adoption drivers:** Health economists place a premium on covariate adjustment because health outcomes are often affected by many observable confounders. The doubly-robust property of Callaway-Sant'Anna—consistent if either the propensity score or outcome regression is correctly specified—is particularly attractive. Health economics was slower to adopt event-study plots than labor economics, so the transition to robust estimators has been more direct.

### 4.3 Historical Phases in Estimator Adoption

The literature reveals three distinct phases in the adoption of heterogeneity-robust DiD estimators:

#### Phase 1: Early Adopter Phase (2020-2021)

**Key publications:**
- **2020**: Callaway & Sant'Anna (2021) working paper circulated; published in Journal of Econometrics, December 2020. Sant'Anna & Zhao (2020) "Doubly Robust Difference-in-Differences Estimators" published in Journal of Econometrics.
- **2021**: Goodman-Bacon "Difference-in-differences with variation in treatment timing" published in Journal of Econometrics. Sun & Abraham "Estimating Dynamic Treatment Effects in Event Studies" published in Journal of Econometrics. Borusyak, Jaravel & Spiess working paper circulated (published 2024 in REStud). Baker, Larcker & Wang working paper circulated (published 2022 in JFE).

**Early adopting applications:**
- Cengiz, Dube, Lindner & Zipperer (2019, QJE) used stacked DiD design before the formal methods literature, representing an early "clean controls" approach.
- The ACA Medicaid expansion literature quickly adopted the Callaway-Sant'Anna (2021) estimator for staggered state adoption.
- The Goodman-Bacon decomposition was rapidly adopted as a diagnostic tool, becoming standard practice in DiD papers by 2021-2022.

**Characteristics:** Early adopters were typically methodologically sophisticated researchers in labor and health economics, often those involved in the methods development or close collaborators. The approach was to report TWFE as the main specification with robust estimators as robustness checks.

#### Phase 2: Consolidation Phase (2022-2023)

**Key developments:**
- **Roth, Sant'Anna, Bilinski & Poe (2023)** published "What's trending in difference-in-differences?" in Journal of Econometrics, synthesizing recent advances and providing concrete recommendations for practitioners.
- **Roth (2022)** published "Pretest with Caution" in AER: Insights, critically warning about pre-trend testing.
- **Baker, Larcker & Wang (2022)** published in Journal of Financial Economics, demonstrating that "TWFE biases can lead to spurious significance and misleading conclusions."
- **Software maturation**: Stata 18 released with built-in `xthdidregress` command, CSDID Version 1.6 released, `fixest::sunab()` stabilized.

**Characteristics:** Journals increasingly expected robustness checks with multiple estimators. The "four estimator" check (Sun & Abraham, Callaway & Sant'Anna, Borusyak et al., de Chaisemartin & D'Haultfœuille) became common in methodologically rigorous applications. The approach shifted from TWFE as main with robust as robustness, to robust estimators as the primary specification.

#### Phase 3: Standardization Phase (2024+)

**Key developments:**
- **Borusyak, Jaravel & Spiess (2024)** published in Review of Economic Studies, with ~5,149 citations.
- **Gardner et al. (2024)** "Two-Stage Differences in Differences" proposes a regression-based alternative claiming superior finite-sample performance.
- **Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna** — "Difference-in-Differences Designs: A Practitioner's Guide" — forthcoming in Journal of Economic Literature.
- **Gerber (May 2026)** introduces the `diff-diff` Python package implementing design-based variance for fifteen modern DiD estimators.
- **Roth (2026)** published "Interpreting Event-Studies from Recent Difference-in-Differences Methods" in Japanese Economic Review.

**Characteristics:** The emerging norm is that robust estimators are the default, with TWFE relegated to a robustness check or appendix. Multiple robustness estimators (usually 2-3) are expected in the same paper. The "practitioner's guide" signals that these methods have matured enough for codification as best practice.

### 4.4 Best-Practice Reporting Strategies

#### 4.4.1 How to Display Pre-Trend Estimates

**Event-study plots** are the primary graphical tool. Miller (2023, JEP) notes that "one of its most appealing features is that it creates a built-in graphical summary of results" [Source: Miller (2023)].

**Critical caveat from Roth (2026):** The default event-studies from CS, dCDH, and BJS "should not be interpreted in the same way as traditional dynamic TWFE event-study plots" [Source: Roth (2026)]. The new methods "may show a kink or jump at the time of treatment even when the TWFE event-study shows a straight line... a kink or jump in the plot may arise even if there is no treatment effect and parallel trends is equally violated in all periods."

**Roth's (2026) recommendations for researchers:**
1. **For Callaway & Sant'Anna:** Use "long-differences for the pre-treatment coefficients as well as the post-treatment coefficients (i.e., always using the period before treatment as the baseline)." This option is available in the `did` package.
2. **For Borusyak et al.:** "Putting the BJS pre-treatment estimates on a different plot from the post-treatment estimates to avoid making misleading visual inferences."
3. **Supplement with additional diagnostics:** "Supplementing event-studies with cohort-specific time series and balanced-event studies to address unit composition changes over relative time."
4. **No threat to post-treatment validity:** "The discussion does not threaten the validity of the post-treatment estimates if the parallel trends assumption holds; rather, it highlights challenges in visualizing violations of parallel trends using conventional heuristics" [Source: Roth (2026)].

**Scott Cunningham's recommendation:** "When you're plotting event studies, you should use long differences, not short gap" [Source: Scott Cunningham (2024), Mixtape Substack].

#### 4.4.2 Whether to Use Pre-Trend Tests as Strict Inference Gates

**The evidence is clear: pre-trend tests should NOT be used as strict inference gates.**

Roth (2022, AER: Insights) provides the definitive evidence:

- "Conventional pre-trends tests may have low power and can exacerbate bias when conditioning estimation and inference on passing the test."
- "Linear violations of parallel trends detectable only 50% of the time may produce biases larger than treatment effects and cause severe under-coverage of confidence intervals."
- "Under homoskedasticity and monotonic trend violations, bias after pretesting is always larger than unconditional bias."
- Confidence interval coverage can fall to as low as **24%** for nominal 95% confidence intervals after pre-testing.

**Recommended alternatives:**

1. **Power analyses:** "Researchers should assess and report the power of pretests against plausible violations" using the `pretrends` R package (and corresponding Stata package) [Source: Roth (2022)].

2. **Equivalence tests:** The `EquiTrends` R package (Bos, 2024) performs equivalence testing "to test if pre-treatment trends in the treated group are 'equivalent' to those in the control group" [Source: Bos (2024)]. Equivalence testing reverses the null hypothesis: the null is that the pre-trend difference exceeds a threshold, and rejecting this null allows the researcher to conclude that trends are "equivalent."

3. **Honest confidence intervals:** The Rambachan and Roth (2023) "HonestDiD" approach provides "robust inference and sensitivity analysis for differences-in-differences and event study designs." Instead of asking "Do parallel trends hold?" it asks "How large would violations of parallel trends need to be before my conclusion changes?" The answer is called the **breakdown value**—"a single number that tells the reader exactly how robust the result is" [Source: Rambachan & Roth (2023), Review of Economic Studies].

4. **Sensitivity analyses:** Roth and Sant'Anna (2023) "falsification tests for the null that parallel trends is insensitive to functional form" [Source: Roth & Sant'Anna (2023), Econometrica].

#### 4.4.3 How to Report Multiple Estimators (Main vs. Robustness)

Based on the accumulated evidence, the following reporting strategy is recommended:

**Main specification:** Increasingly, a heterogeneity-robust estimator (Callaway-Sant'Anna, Sun-Abraham, or BJS imputation) is reported as the primary specification. Sant'Anna (2024) recommends: "Under PT, when treatment effects are heterogeneous/dynamic, β does not recover an easy-to-interpret parameter... the solution is to separate identification, aggregation, and estimation/inference steps" [Source: Sant'Anna (2024), NABE TEC presentation].

**Robustness appendix:** Multiple estimators (typically 2-3) should be reported to show that results are not sensitive to estimator choice. The Wang, Hamad & White (2024) simulation showed that "heterogeneous treatment effects-robust estimators exhibited more robust results across all scenarios under the parallel trends assumption" [Source: Wang, Hamad & White (2024)].

**Goodman-Bacon decomposition:** Should be reported in the appendix as a diagnostic to demonstrate the extent of "forbidden comparisons" and assess whether the TWFE estimator is likely biased [Source: Goodman-Bacon (2021)].

**The Roth et al. (2023) checklist recommends:** "transparent assumption discussion, appropriate choice of estimands, and robustness checks to improve validity and interpretability in DiD studies" [Source: Roth, Sant'Anna, Bilinski & Poe (2023)].

#### 4.4.4 Concrete Guidance on Sensitivity Analyses

The evidence points to the following standard battery of sensitivity analyses:

1. **Different comparison groups:** Using never-treated vs. not-yet-treated as the control group (Callaway & Sant'Anna, 2021 offer both options).
2. **Different anticipation windows:** Testing no-anticipation assumptions by varying the number of periods before treatment assumed to be unaffected.
3. **Different covariates:** Using regression adjustment, inverse probability weighting, and doubly robust approaches for covariate inclusion/exclusion.
4. **Sample restrictions:** Dropping specific cohorts, trimming the tails of event time distributions.
5. **Functional form sensitivity:** Roth & Sant'Anna (2023) address sensitivity to transformations of the outcome variable.
6. **Honest confidence intervals:** Rambachan & Roth (2022) propose relaxing strict parallel trends and using sensitivity parameters.
7. **Pre-trend test power:** Use the `pretrends` package to compute the minimal detectable violation (MDV) and assess whether the test has adequate power.
8. **Breakdown values:** Use `HonestDiD` to compute the magnitude of parallel trends violation needed to overturn results.

**The Sant'Anna DiD Checklist** (NABE TEC 2024) recommends: start by plotting treatment rollout, document cohorts, assess parallel trends assumptions, incorporate covariates if needed, conduct sensitivity analyses, use ATT(g,t) as building blocks, and report multiple aggregation schemes [Source: Sant'Anna (2024)].

### 4.5 Evidence from Systematic Audits and Replications

#### 4.5.1 Baker, Larcker & Wang (2022) — Key Quantitative Findings

Published in the Journal of Financial Economics, this is the most influential replication study in the staggered DiD literature:

- **~49% of DiD studies in top finance and accounting journals from 2000-2019 employ staggered DiD designs** [Source: Baker, Larcker & Wang (2022)].
- **Simulation results**: "TWFE estimates are unbiased with a single treatment period or with homogeneous treatment effects, but biased when combined with staggered timing and heterogeneous effects."
- **Biases can reverse signs**: "Static staggered DiD treatment effect estimates can obtain the opposite sign of the true ATT, even under random assignment."
- **Event-study bias**: "Biases in static staggered DiD estimates are not resolved by event-study estimators; dynamic effects are also problematic due to contamination across periods."
- **Alternative estimators**: "Alternative estimators modify the set of effective comparison units to avoid comparing treated units with previously treated units, improving causal effect recovery."

**Specific replication findings:**
- **Beck, Levine, and Levkov (2010)** on bank deregulation: "The published staggered DiD estimates are susceptible to biases from treatment effect heterogeneity and alternative estimators often do not support the papers' original claims."
- **Fauver, Hung, Li, and Taboada (2017)** on global board governance reform: "Applying alternative DiD estimators to prior key studies often reverses original findings, rendering them insignificant."
- **Wang, Yin & Yu (2021)** replication: "Previously reported significant negative effects of stock repurchase legalization on firm investment disappear" when using robust DiD alternatives.

#### 4.5.2 Wang, Hamad & White (2024) — Monte Carlo Simulation for Health Economics

Published in Epidemiology, this simulation study specifically targets health policy researchers:

- "Two-way fixed effects performed well only under constant homogeneous effects; under dynamic or heterogeneous effects, heterogeneity-robust estimators reduce bias significantly."
- "The two-way fixed effects estimator was shown to be biased and/or estimate an unintuitive target parameter when treatment effects were dynamic or heterogeneous."
- Recommends: "employ heterogeneous treatment effects-robust methods alongside diagnostics like the Goodman-Bacon test and sensitivity analyses to ensure more reliable causal inference in health policy evaluations" [Source: Wang, Hamad & White (2024)].

#### 4.5.3 Deb, Norton, Wooldridge & Zabel (2024) — Aggregation Weights in CS Software

This paper reveals that software implementations of Callaway and Sant'Anna's method use aggregation weights that differ from what the literature describes:

- "The software uses weights that include the number of observations in the reference pre-period instead of only the number of observations in the treated periods."
- "The weighting matters more when there is more heterogeneity in the treatment effects, when there is more heterogeneity in the number of observations, and when those heterogeneities are correlated."
- "For many simple difference-in-differences model specifications, the choice of estimator yields identical point estimates of the group-time treatment effects... However, for the Callaway and Sant'Anna method, the Stata commands use weights based in part on the number of observations in the period just prior to the start of treatment, yielding different ATET estimates whenever the data set has an unbalanced number of observations."
- "We are concerned that many published results have been calculated with a formula that is not what the researchers intended."

The paper demonstrates that overall ATETs can differ by over 16% in their simulation example due to this aggregation weighting issue [Source: Deb, Norton, Wooldridge & Zabel (2024), NBER WP 34331].

#### 4.5.4 Lee & Wooldridge (2023) — Efficiency Critique of Callaway & Sant'Anna

Lee and Wooldridge (2023) critique the efficiency of the Callaway & Sant'Anna estimator, proposing a "rolling transformation" approach:

- "Callaway & Sant'Anna's (2021) method is less efficient" due to "only using the period before treatment for control comparison."
- "Long differencing methods, such as those proposed by Callaway and Sant'Anna (2021), can be considerably less efficient."
- Their rolling method uses the **average of all pre-treatment outcomes** to form control groups, improving efficiency while maintaining robustness.
- "Monte Carlo simulations demonstrate improved bias and precision of the proposed method versus competing approaches including Callaway and Sant'Anna (2021)" [Source: Lee & Wooldridge (2023)].

This critique suggests that researchers should consider whether the efficiency loss from using only one pre-treatment period (as in CS) is acceptable, or whether a rolling/averaging approach might yield more precise estimates.

---

## 5. Conclusion: Emerging Norms and Practical Recommendations

The staggered DiD literature has undergone a remarkable transformation since Goodman-Bacon's (2021) decomposition revealed the vulnerabilities of TWFE estimators. Three main approaches have emerged—Callaway and Sant'Anna's (2021) group-time ATTs with doubly-robust estimation, Sun and Abraham's (2021) interaction-weighted estimator, and Borusyak et al.'s (2024) imputation-based estimator—each offering distinct advantages in handling heterogeneous treatment effects and dynamic treatment timing.

**Key findings from this deepened analysis:**

1. **Implementation details matter substantially.** The choice of software package (R vs. Stata), default comparison group, anticipation parameter, base period selection, and aggregation weights can materially affect results. The Scott Cunningham (2026) cross-package audit found that package choice accounts for 16% of variation in estimates, and the Deb et al. (2024) analysis found that aggregation weights can produce 16% differences in ATET estimates. Researchers should understand these implementation nuances and, ideally, report results from multiple software packages.

2. **Each estimator's features are targeted solutions to specific methodological problems.** The clean control selection (never-treated vs. not-yet-treated vs. last-treated) directly addresses the forbidden comparisons problem. Untreated-only counterfactual estimation eliminates contamination bias. Cohort-specific estimation prevents negative weighting. Clean pre-treatment coefficients enable valid pre-trend testing. Conditional parallel trends addresses low power in unconditional tests. The imputation approach's use of all pre-treatment periods increases efficiency but makes estimates sensitive to violations in any single period.

3. **Adoption patterns show clear field-specific preferences.** Callaway & Sant'Anna dominates in health economics due to its covariate adjustment capabilities. Labor economics shows more diverse adoption, with stacked DiD and LP-DiD alongside CS and SA. The field has moved through three phases—early adoption (2020-2021), consolidation (2022-2023), and standardization (2024+)—with robust estimators now expected as the default.

4. **Pre-trend testing requires fundamental rethinking.** Roth's (2022) critique demonstrates that pre-trend tests should not be used as strict inference gates. Instead, researchers should conduct power analyses (using `pretrends`), equivalence tests (using `EquiTrends`), and sensitivity analyses (using `HonestDiD`). Roth's (2026) paper further warns that default event-study plots from modern estimators are constructed asymmetrically and should not be interpreted using TWFE heuristics.

5. **Replication evidence confirms the importance of robust estimators.** Baker, Larcker & Wang (2022) demonstrated that TWFE biases can lead to spurious significance and that alternative estimators often contradict original claims. Wang, Hamad & White (2024) confirmed that heterogeneity-robust estimators perform well across simulation scenarios.

**For applied researchers, the emerging best practice is:**
- Use a heterogeneity-robust estimator as the primary specification
- Report 2-3 estimator specifications as robustness checks
- Include Goodman-Bacon decomposition diagnostics
- Conduct power analysis for pre-trend tests
- Report breakdown values from HonestDiD sensitivity analysis
- Understand and document software implementation choices
- Standardize covariates to avoid numerical issues
- Interpret event-study plots with awareness of asymmetry

The forthcoming "Practitioner's Guide" by Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna in the Journal of Economic Literature signals that these methods have matured enough for codification as best practice. For applied researchers in labor and health economics, the message is clear: heterogeneity-robust estimators are no longer optional but are the expected standard for causal inference with staggered treatment timing.

---

## Sources

[1] Callaway & Sant'Anna (2021) - Journal of Econometrics: https://www.sciencedirect.com/science/article/abs/pii/S0304407620303948

[2] Callaway & Sant'Anna (2021) - Full text PDF: https://psantanna.com/files/Callaway_SantAnna_2020.pdf

[3] Sun & Abraham (2021) - Journal of Econometrics: https://ideas.repec.org/a/eee/econom/v225y2021i2p175-199.html

[4] Sun & Abraham (2021) - Full text PDF: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/Paper2023/D3--Sun-2020-Estimating-dynamic-treatment-effects-in-event-studies-with-heterogeneous-treatment-effects.pdf

[5] Borusyak, Jaravel & Spiess (2024) - Review of Economic Studies: https://academic.oup.com/restud/article/91/6/3253/7601390

[6] Borusyak, Jaravel & Spiess (2024) - arXiv working paper: https://arxiv.org/abs/2108.12419

[7] Goodman-Bacon (2021) - Journal of Econometrics: https://www.sciencedirect.com/science/article/abs/pii/S0304407621001445

[8] Goodman-Bacon (2021) - Full text PDF: https://file-lianxh.oss-cn-shenzhen.aliyuncs.com/Refs/2025-08-Yang/Goodman-Bacon_2021_Difference-in-differences_with_variation_in_treatment_timing.pdf

[9] Roth (2022) - American Economic Review: Insights: https://www.aeaweb.org/articles?id=10.1257/aeri.20210236

[10] Roth (2022) - Full text PDF: https://www.jonathandroth.com/assets/files/roth_pretrends_testing.pdf

[11] Roth (2026) - "Interpreting Event-Studies from Recent DiD Methods" - Japanese Economic Review: https://arxiv.org/abs/2401.12309

[12] Roth (2026) - Full text PDF: https://www.jonathandroth.com/assets/files/HetEventStudies.pdf

[13] Rambachan & Roth (2023) - Review of Economic Studies: https://academic.oup.com/restud/article/90/5/2555/6950347

[14] Roth, Sant'Anna, Bilinski & Poe (2023) - Journal of Econometrics: https://www.jonathandroth.com/assets/files/DiD_Review_Paper.pdf

[15] Baker, Larcker & Wang (2022) - Journal of Financial Economics: https://www.sciencedirect.com/science/article/pii/S0304405X22000204

[16] Baker, Larcker & Wang (2022) - Full text PDF: https://dash.harvard.edu/bitstreams/fe5cc546-b8e9-4814-8d3a-06af82c77845/download

[17] Goldsmith-Pinkham (2026) - "Tracking the Credibility Revolution across Fields" - NBER WP 35051: https://www.nber.org/papers/w35051

[18] Goldsmith-Pinkham (2026) - Full text PDF: https://paulgp.com/econlit-pipeline/paper-figures/goldsmith-pinkham-credibility-revolution.pdf

[19] Deb, Norton, Wooldridge & Zabel (2024) - NBER WP 34331: https://www.nber.org/system/files/working_papers/w34331/w34331.pdf

[20] Lee & Wooldridge (2023) - "A Simple Transformation Approach to DiD": https://www.econ.queensu.ca/sites/econ.queensu.ca/files/Lee_Wooldridge_20230720.pdf

[21] Sant'Anna (2024) - "Modern Difference-in-Differences" - NABE TEC Presentation: https://psantanna.com/DiD/NABE_202410.pdf

[22] Callaway (2022) - "Difference-in-Differences for Policy Evaluation": https://bcallaway11.github.io/files/Callaway-Chapter-2022/main.pdf

[23] Wang, Hamad & White (2024) - "Advances in DiD Methods" - Epidemiology: https://pmc.ncbi.nlm.nih.gov/articles/PMC11305929

[24] Cunningham (2021) - "Causal Inference: The Mixtape" - Staggered Adoption DiD: https://causalinf.substack.com/p/waiting-for-event-studies-a-play

[25] Gardner (2021) - "Two-stage differences in differences": https://arxiv.org/abs/2207.05943

[26] did R package documentation: https://bcallaway11.github.io/did/reference/att_gt.html

[27] csdid Stata slides: https://www.stata.com/meeting/us21/slides/US21_SantAnna.pdf

[28] Stata xthdidregress manual: https://www.stata.com/manuals/causalxthdidregress.pdf

[29] fixest sunab documentation: https://lrberge.github.io/fixest/reference/sunab.html

[30] eventstudyinteract Stata help file: http://fmwww.bc.edu/repec/bocode/e/eventstudyinteract.sthlp

[31] did_imputation Stata/GitHub: https://github.com/borusyak/did_imputation

[32] bacondecomp R/Stata package: https://github.com/evanjflack/bacondecomp

[33] MetricGate - Sun-Abraham estimator: https://metricgate.com/docs/sun-abraham-estimator

[34] GitHub fixest Issue #287: https://github.com/lrberge/fixest/issues/287

[35] Asjad Naqvi did_imputation documentation: https://asjadnaqvi.github.io/DiD/docs/did_imputation

[36] Chen Xing - Notes on Callaway & Sant'Anna (2021): https://chenxing.space/blog/notes-on-callaway-sant-anna-2021-staggered-adoption-did

[37] de Chaisemartin & D'Haultfœuille (2020) - American Economic Review: https://www.aeaweb.org/articles?id=10.1257/aer.20181169

[38] Roth & Sant'Anna (2023) - Journal of Political Economy: Microeconomics: https://www.journals.uchicago.edu/doi/10.1086/726596

[39] Cengiz, Dube, Lindner & Zipperer (2019) - Quarterly Journal of Economics: https://academic.oup.com/qje/article/134/3/1405/5420451

[40] Miller (2023) - "An Introductory Guide to Event Study Models" - Journal of Economic Perspectives: https://www.aeaweb.org/articles?id=10.1257/jep.37.2.203

[41] Scott Cunningham (2026) - "Six Packages, One Estimator" - Package Audit: https://causalinf.substack.com/

[42] Bos (2024) - "EquiTrends" R package: https://cran.r-project.org/package=EquiTrends

[43] Stata CSDID Version 1.6 documentation (Fernando Rios-Avila): https://friosavila.github.io/playingwithstata/raw_articles/csdid_stata.html

[44] GCcollab Wiki - "The problem — Treatment effect heterogeneity": https://wiki.gccollab.ca/images/7/73/Handout_-_Treatment_effect_heterogeneity.pdf

[45] Sant'Anna & Zhao (2020) - "Doubly Robust Difference-in-Differences Estimators" - Journal of Econometrics: https://ideas.repec.org/a/eee/econom/v219y2020i1p101-122.html