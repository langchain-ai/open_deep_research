# Comprehensive Options Trading Framework for a $700 Account
## Deepened Research Revision with Full Parameter Ranges, Quantitative Analysis, and Authoritative Sources

**Educational Purposes Only — Not Personalized Financial Advice**

*This framework is designed for educational and informational purposes only. Options trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The $200–300/week target referenced in this framework represents an extremely aggressive goal that requires taking on significant risk. Consult a qualified financial advisor before engaging in any trading activity.*

*All quantitative parameters include source citations from academic research, options exchanges (Cboe, OCC), regulatory bodies (SEC, FINRA), broker educational materials (Schwab, Fidelity, tastytrade, Interactive Brokers), and options education authorities (OIC, Cboe Options Institute).*

---

## Table of Contents

1. [Critical Foreword: Realistic Expectations]()
2. [Risk Management, Position Sizing & Exposure — Full Parameter Ranges]()
   - 2.1 Maximum Risk Per Trade — Conservative and Aggressive Ranges
   - 2.2 Maximum Concurrent Positions and Sector Exposure
   - 2.3 Maximum Daily, Weekly, and Monthly Loss Limits
   - 2.4 Maximum Total Account Exposure
   - 2.5 Hard Rules for Account Drawdown
   - 2.6 Rules for Reducing and Increasing Position Size
   - 2.7 Minimum Cash/Capital Reserves
   - 2.8 Position Sizing Methods (Fixed Percentage, Kelly, ATR-Based, Monte Carlo)
3. [Permitted & Prohibited Strategies — Full Boundaries]()
   - 3.1 Permitted Strategies with Detailed Rules
   - 3.2 Prohibited Strategies with Clear Reasoning
   - 3.3 Broker Approval Levels and Requirements
4. [Technical Criteria & Option Chain Selection — Advanced Screening Rules]()
   - 4.1 Optimal Stock Price Bands for $700 Accounts
   - 4.2 Optimal Option Premium Bands
   - 4.3 Volume, Open Interest, and Bid-Ask Spread Thresholds
   - 4.4 IV Rank and IV Percentile Strategy Selection Rules
   - 4.5 ATR-Based Position Sizing Rules
   - 4.6 Technical Indicator Screening with Explicit Thresholds
   - 4.7 VWAP-Based Rules
   - 4.8 Bollinger Band Squeeze Rules
   - 4.9 Multi-Timeframe Confirmation Requirements
5. [Strike & Expiry Selection — Detailed Scenario-Based Guidelines]()
   - 5.1 Delta Bands by Strategy Type
   - 5.2 Spread Width Selection Rules by Stock Price
   - 5.3 DTE Risk Bands Categorized by Time Horizon
   - 5.4 Theta Decay Acceleration Rules
   - 5.5 Gamma Risk Management Parameters
   - 5.6 Probability of Touch and Expected Move Concepts
6. [Liquidity Filters — Stricter Requirements]()
   - 6.1 Tiered Liquidity Thresholds
   - 6.2 Order Type Rules
   - 6.3 Rules for Avoiding Illiquid Situations
7. [Worked Example 1: Bull Call Spread on SPY]()
8. [Worked Example 2: Bull Put Credit Spread on AAPL]()
9. [Probabilistic & Benchmarking Analysis — Quantitative Methods]()
   - 9.1 Binomial Distribution Modeling for Weekly Targets
   - 9.2 Required Trade Statistics by Scenario
   - 9.3 Standard Deviation of Expected Weekly Returns
   - 9.4 Sharpe Ratio Analysis
   - 9.5 Kelly Criterion Calculations (Full, Half, Quarter)
   - 9.6 Risk of Ruin Calculations
   - 9.7 Benchmarks from Professional and Retail Options Traders
   - 9.8 CBOE Benchmark Index Performance Data
10. [Complete Daily Screening Checklist]()
11. [Sources]()

---

## 1. Critical Foreword: Realistic Expectations

Before any detailed framework is presented, a fundamental truth must be stated clearly. A target of **$200–300 per week on a $700 account** represents a **28–43% weekly return**. To put this in perspective, professional options traders and fund managers aim for **20–50% returns per year**, not per week [DayTrading.com].

The reality is that a disciplined, sustainable options trading approach on a $700 account would target **$14–$35 per week (2–5%)** using high-probability strategies. The $200–300 target is only mathematically achievable through strategies that carry a high probability of total account loss.

If you were to risk 2% per trade ($14) and achieve a 2:1 reward-to-risk ratio with a 60% win rate, you would need approximately **10–22 winning trades per week** to hit $200–300. Conversely, if you risked 50%+ of your account per trade, a losing streak of just 2–3 trades would wipe out the account entirely [Bulls on Wall Street].

According to academic research, even professional day traders with positive expectancy typically earn only 0.28% (after fees) per day on average, with the top 500 ranked traders earning 49.5 basis points (before fees) per day [SSRN - The Cross-Section of Speculator Skill]. This translates to roughly 2.5% per week for the top tier — far below the 28–43% weekly target.

**This framework will present the disciplined rules first, then analyze what it would actually take to achieve the stated target.** The rules below are designed for account preservation and long-term survival, not for chasing the $200–300/week target.

### CBOE Benchmark Context

For context, the CBOE S&P 500 BuyWrite Index (BXM) — a benchmark that maintains a buy-write (covered call) strategy on the S&P 500 — has delivered an **annualized compound return of 11.77%** since 1988, with a standard deviation of 9.29% and a Sharpe ratio of 0.77 [CBOE - An Historical Evaluation of the BXM]. The CBOE S&P 500 PutWrite Index (PUT) — a benchmark that sells at-the-money put options on the S&P 500 — has delivered an **annualized compound return of 9.54%** since 1986, with a standard deviation of 9.95% and a Sharpe ratio of 0.65 [CBOE - Historical Performance of Put-Writing Strategies]. These are professional, systematic options-selling strategies with decades of track records. They generate annual returns of approximately 10–12% — roughly the weekly target being sought on a $700 account.

---

## 2. Risk Management, Position Sizing & Exposure — Full Parameter Ranges

### 2.1 Maximum Risk Per Trade — Conservative and Aggressive Ranges

**Industry-Standard Thresholds (Consolidated from Multiple Authoritative Sources)**

| Risk Level | % of Account | $ on $700 Account | Source / Rationale |
|------------|--------------|-------------------|--------------------|
| **Ultra-Conservative** | 0.5–1.0% | $3.50–$7.00 | [tastytrade] [Option Alpha] — For volatile instruments like options, less than 0.5% recommended for novices |
| **Conservative (Recommended)** | 1.0–2.0% | $7.00–$14.00 | [TradeAlgo] — "Single most important rule in futures trading" — never risk more than 1–2% |
| **Moderate** | 2.0–3.0% | $14.00–$21.00 | [tastytrade] — For defined-risk strategies on average accounts ($20K–$100K) |
| **Aggressive (Ceiling)** | 3.0–5.0% | $21.00–$35.00 | [Option Alpha] — Smaller accounts under $10K may need to risk closer to 5% |
| **Extreme (Not Recommended)** | 5.0–10.0% | $35.00–$70.00 | [tastytrade] — For defined-risk strategies on accounts under $20K; Monstrous risk |
| **Gambling** | 10.0%+ | $70.00+ | Not supported by any educational authority |

**Specific Authoritative Guidance:**

**Tastytrade / Dr. Jim Schultz (Quantitative Expert, Finance Ph.D.)** : For defined-risk strategies (credit/debit spreads, iron condors), 1–3% of account value per position is the reference for accounts $20K–$100K. For accounts under $20K, traders may need to increase to 5–7% or more due to minimum contract sizes. For undefined-risk strategies (short puts, strangles), 3–7% is recommended for average accounts [tastylive - Defined-Risk and Undefined-Risk Position Sizing].

**Option Alpha**: Recommends allocating 1–5% risk per trade on a sliding scale. Smaller accounts (under $10K) may risk closer to 5%; larger accounts should scale down toward 1%. "There is no trade so great that it requires more money than your max risk per trade" [Option Alpha - Account Size Adjustments].

**TradeAlgo (Futures Risk Management)** : The 1–2% rule is called "the single most important rule in futures trading" — never risk more than 1% to 2% of your account on any single trade [TradeAlgo - Futures Risk Management].

**MetaTrader 5 / Monte Carlo Research (Daniel Opoku)** : The commonly recommended 1–2% risk per trade is validated for most scenarios by Monte Carlo simulation. For robust systems with high win rates and specific risk-reward ratios, it can be justifiable to exceed 2%. Dynamic risk (percentage of current balance) outperforms fixed risk in long-term returns but increases volatility [MQL5 - Building a Trading System Part 2].

**Lawrence G. McMillan (Author of "Options As A Strategic Investment")** : Position sizing is "the single biggest factor" in trading success. Profitability depends more on how you size and manage positions than on the setups you choose [The Option Strategist - Risk Management Webinar].

**Critical Note for $700 Account**: Practical constraints often mean the smallest options contract (1 contract = 100 shares) may cost more than 1–2% of the account. This means traders with accounts under ~$2,000 may be forced to violate standard 1–2% risk guidelines or must only trade very low-premium, far out-of-the-money options, mini-options, or NANOS contracts (discussed in Section 4).

### 2.2 Maximum Concurrent Positions and Sector Exposure

**Maximum Concurrent Positions:**

| Approach | Maximum Positions | Source / Rationale |
|----------|------------------|--------------------|
| **Conservative** | 1–2 | For a $700 account: Each $1-wide spread uses $100+ in buying power; 2 positions use $200+ |
| **Moderate** | 2–3 | Most positions use $100–$200 each; 3 positions use up to $600 (86% of account) |
| **Aggressive** | 3–5 | 5 positions at $100 each = $500 (71% of account); leaves $200 cash reserve |
| **Absolute Maximum** | 5 | [OptionsPlay] — No more than five simultaneous trades to cap total risk at 10% of account |

**Maximum Sector/Correlation Exposure:**

| Exposure Type | Conservative Limit | Moderate Limit | Aggressive Limit | Source |
|---------------|-------------------|----------------|------------------|--------|
| Single Stock | 5% of account | 10% of account | 15% of account | [Financial Edge Training] [Guardfolio] |
| Single Sector | 25% of account | 33% of account | 50% of account | [Guardfolio] [PL Capital] |
| Correlated Underlyings | Treat as one risk unit | Same sector = same risk pool | Monitor correlation matrix | [tastytrade] |

**Additional Guidance:**
- The "5% rule" advises that no single stock should exceed 5% of a portfolio to avoid concentration risk [Financial Edge Training - Diversification]
- A single-stock position above 10% of a portfolio is generally considered concentrated; sector concentration above 25% in one sector is as dangerous as single-stock concentration [Guardfolio - Portfolio Concentration Risk]
- A fund overlap of 33% or more between funds means those funds "behave like clones" [PL Capital - Sector Concentration Risk]
- For a $700 account: maximum 1 position per stock, 2 positions max across correlated underlyings, treat same-sector positions as shared risk pool

### 2.3 Maximum Daily, Weekly, and Monthly Loss Limits

**Daily Loss Limits:**

| Risk Level | % of Account | $ on $700 | Source |
|------------|--------------|-----------|--------|
| **Conservative** | 1–2% | $7–$14 | [Zerodha / In The Money] — ~0.5% for discretionary traders |
| **Moderate** | 3–4% | $21–$28 | [For Traders] — Proprietary trading firms typically cap at 4–5% |
| **Aggressive** | 5–6% | $35–$42 | [TradeAlgo] — Professional futures traders cap "portfolio heat" at 5–6% |
| **Extreme (Not Recommended)** | 10%+ | $70+ | No educational source supports this; 80% of traders fail daily loss rules |

**Weekly Loss Limits:**

| Risk Level | % of Account | $ on $700 | Source |
|------------|--------------|-----------|--------|
| Conservative | 3–5% | $21–$35 | [Bulls on Wall Street] [TradeThatSwing] |
| Moderate | 5–8% | $35–$56 | Synthesized from daily limit × 5 |
| Aggressive | 8–12% | $56–$84 | [Reddit r/options] — "Try to limit monthly drawdowns to 12 percent" |

**Monthly Drawdown Limits:**

| Risk Level | % of Account | $ on $700 | Action Required | Source |
|------------|--------------|-----------|-----------------|--------|
| Conservative | 5–8% | $35–$56 | Reduce position sizes by 25%, review strategy | [TradeAlgo] |
| Moderate | 10–15% | $70–$105 | **Mandatory trading pause**, full strategy review | [TradeAlgo] — "The edge in futures trading is not in your entries. It's in your risk management." |
| Aggressive Ceiling | 15–20% | $105–$140 | Stop trading for 1 month, reduce risk per trade by 50% upon return | [TradeFundrr] |
| Danger Zone | 20%+ | $140+ | Account viability at serious risk; deposit more capital or stop trading | Synthesized from multiple sources |

**Specific Hard Rules for Loss Limits:**

1. **Daily Hard Stop**: If down 5% ($35) on the day, stop trading for that day immediately. No exceptions. [For Traders] [TradeAlgo]
2. **Weekly Hard Stop**: If down 8% ($56) by Wednesday, stop trading for the remainder of the week. [Bulls on Wall Street]
3. **Monthly Hard Stop**: If down 12% ($84) on the month, stop trading for 2 weeks. Review every trade taken that month. [Reddit r/options]
4. **10% Monthly Drawdown**: Reduce risk per trade by 25% immediately upon reaching this threshold. [TradeAlgo]
5. **15% Monthly Drawdown**: Reduce risk per trade by 50%, mandatory 1-week trading pause, conduct full strategy audit. [TradeAlgo] [TradeFundrr]
6. **20% Total Drawdown from Peak**: Stop trading entirely. Deposit additional capital or close the account. The strategy as implemented is not working. [TradeFundrr] [TradeAlgo]

### 2.4 Maximum Total Account Exposure at Any Time

| Exposure Metric | Conservative | Moderate | Aggressive | Source |
|-----------------|--------------|----------|------------|--------|
| Total Buying Power Used | 50% of account | 70% of account | 85–90% of account | [tastytrade] |
| Total Delta Dollars | 0.5× account size | 0.75× account size | 1.0× account size | [Interactive Brokers Risk Navigator] |
| Total Theta (for sellers) | 0.06% of account/day | 0.08% of account/day | 0.10% of account/day | [Interactive Brokers Risk Navigator] |
| Maximum Concurrent Risk | $350 (50%) | $490 (70%) | $630 (90%) | Synthesized |

**Interactive Brokers Risk Navigator Guidance:**
- **Delta Dollars rule of thumb**: Keep total Delta Dollars within a -1:1 to +1:1 ratio of account size. This means net directional exposure should not exceed 100% of account value. For a $700 account, maximum net directional exposure is ±$700 in delta-adjusted terms.
- **Theta target**: Target a Theta equivalent of 0.06% to 0.10% of account size per day when selling options. For a $700 account, this means $0.42–$0.70 of daily theta [Interactive Brokers - Risk Navigator Quick Guide].

**CBOE Risk Management Tools:**
Cboe offers free risk management tools for members that allow customizable parameters including limits on execution timeframe, quantity of contracts, notional volume, and trade count. A "Mass Cancel" feature and optional self-imposed "Kill Switch" allow trading halts once a risk parameter is reached [CBOE - US Options Exchange Risk Management Tools].

**FINRA Position Limits (Rule 2360 for ETFs):**
Position limits for selected ETFs: SPY limit = 3,600,000 contracts; DIA limit = 300,000 contracts. For a $700 account, these limits are irrelevant as the account will never approach such levels [FINRA - Rule 2360].

### 2.5 Hard Rules for Account Drawdown

**Drawdown Thresholds and Required Actions:**

| Drawdown Level | % of $700 Account | Required Action |
|----------------|-------------------|-----------------|
| **5%** ($35) | Monthly review trigger | Review all trades taken. No immediate action required unless trend continues. |
| **10%** ($70) | **Reduce risk per trade by 25%** | Cut from 2% to 1.5% ($14 to $10.50). Review strategy. [TradeAlgo] |
| **10–15%** ($70–$105) | **Mandatory trading pause + full strategy review** | Stop trading for 1–2 weeks. Review every trade. "A maximum drawdown limit of 10 to 15% should trigger a mandatory trading pause and full strategy review." [TradeAlgo] |
| **15%** ($105) | **Reduce risk per trade by 50%** | Cut from 2% to 1% ($14 to $7). If at 1%, reduce to 0.5%. [TradeFundrr] |
| **20%** ($140) | **Stop trading entirely for 1 month** | Deposit more capital or evaluate if the strategy needs fundamental changes. Account viability is at serious risk. [TradeFundrr] |
| **25%+** ($175+) | **Account likely requires major overhaul or closure** | The probability of recovery from a 25% drawdown requires a 33% gain just to break even. [Bulls on Wall Street] |

**Critical Guidance on Drawdown Behavior:**

Traders who increase trading frequency during drawdown periods amplify losses by **65% on average** [TradeFundrr - Mastering Drawdown Control].

"Nothing will do more damage and completely derail your trading efforts more quickly than oversizing your positions" [tastylive - Defined-Risk and Undefined-Risk Position Sizing].

If you can't sleep with the position on, you're over-levered — no matter how good the strategy looks on paper [Zerodha / In The Money - Position Sizing Part 3].

### 2.6 Rules for Reducing and Increasing Position Size

**Rules for Reducing Position Size During Drawdowns:**

1. **Tiered Reduction (Explicit)** :
   - Down 10% on the month: Cut risk per trade in half (e.g., from 2% to 1%)
   - Down 15% on the month: Reduce to quarter-Kelly or 0.5% risk per trade
   - Down 20% from peak: Reduce to minimum viable position size (0.25–0.5%)

2. **Scaling Down Methodology**: Scaling down during drawdowns should be gradual rather than abrupt to avoid missing potential recoveries. Consistency in defining capital is critical: "Whatever definition you choose, stay consistent. Do not call free cash 'capital' one day and net worth 'capital' the next" [Zerodha / In The Money - Position Sizing Part 3].

3. **Option Alpha Guidance**: The lower the equity risked per trade, the lower the probability that a sequence of poor trades will cause a significant drawdown that eliminates all trading capital. As account sizes grow, position sizes need to get proportionally smaller [Option Alpha - Account Size Adjustments].

4. **Ralph Vince (Optimal f / Secure F)** : Secure F adapts Optimal f by incorporating maximum allowable drawdown. It solves the problem of maximizing net profit under the condition that Max Drawdown ≤ Max Allowed Drawdown. Trading at f/2 (half-Kelly) minimizes the cost of missing the peak of the Optimal f curve in the future and dramatically reduces drawdown risk [Better System Trader - Ralph Vince on Position Sizing].

5. **Monte Carlo Validation**: Using Monte Carlo simulation, if 1,000 random sequences of trades are simulated with 5% risk per trade, and 940 of them have maximum drawdowns of less than 25%, then the probability of achieving a maximum drawdown of less than 25% is 94% (940/1,000) [Technical Analysis of Stocks & Commodities - Michael R. Bryant, Ph.D., February 2001].

**Rules for Increasing Position Size After Gains:**

1. **Tastytrade Philosophy**: "Trade small, trade often" with consistent sizing. Having 10 positions at 1% buying power and then a 10% position could undermine the system. Be consistent with position sizing [Reddit r/thetagang - A Discussion: Tasty Trade's Trading Commandments].

2. **Option Alpha Guidance**: Position sizing should follow a sliding scale: as account equity grows, the percentage risked per trade should actually decrease (not increase) to manage risk. Larger accounts tolerate less relative volatility [Option Alpha - Account Size Adjustments].

3. **Conservative Rule for Increasing**: Only increase position size after:
   - Account grows by at least 20% from starting balance
   - A minimum of 30–50 trades have been executed at the current size
   - The trader has a positive expectancy over that sample
   - Increase by no more than 25% at a time (e.g., from 1% to 1.25%)

4. **Aggressive Rule for Increasing**: Only increase after:
   - Account grows by at least 50% from starting balance
   - Minimum of 100 trades at current size
   - Win rate and R:R metrics are stable and verified
   - Increase by no more than 50% at a time

### 2.7 Minimum Cash/Capital Reserves

| Reserve Level | % of Account | $ on $700 | Source / Rationale |
|---------------|--------------|-----------|--------------------|
| **Minimum** | 5–10% | $35–$70 | [Reddit r/Optionswheel] — Practitioners commonly maintain 6–10% in hard cash |
| **Standard** | 10–20% | $70–$140 | [Option Alpha] — Cash reserves for emergencies, margin calls, and new opportunities |
| **Conservative** | 20–30% | $140–$210 | [U.S. Bank] — Cash and cash equivalents should comprise 2–10% of portfolio; higher for active traders |
| **Recommended for $700** | 10–15% | $70–$105 | Allows for 1–2 new positions without liquidating existing trades |

**Additional Guidance on Cash Reserves:**
- Cash reserves are uninvested funds held for emergencies, margin calls, options assignments, or new opportunities. They are typically kept in highly liquid forms (bank accounts, Treasury bills, money market funds) [Option Alpha - Cash Reserves in Portfolio Management].
- A general rule of thumb is that cash and cash equivalents should comprise between 2% and 10% of a portfolio. It's recommended to maintain at least three to six months of income in cash for emergencies [U.S. Bank - Percentage of Cash in Portfolio].
- For options selling strategies (credit spreads), cash reserves are critical because buying power requirements fluctuate with market conditions. A rapid market move can increase margin requirements unexpectedly [Zerodha / In The Money].

### 2.8 Position Sizing Methods

#### Fixed Percentage Method

**Definition**: Position sizing is the number of shares or contracts times their price, represented as a percentage of total account capital. The "equity risked" is the amount an investor is willing to lose on a trade, typically less than the position size itself [Option Alpha - Account Size Adjustments].

**Formula**: `Position Size ($ at risk) = Account Balance × Risk %`

**Tastytrade Implementation**:
- For defined-risk strategies: 1–3% for average accounts ($20K–$100K)
- For accounts under $20K: may need 5–7% or more due to minimum contract sizes
- For undefined-risk strategies: 3–7% for average accounts [tastylive - Defined-Risk and Undefined-Risk Position Sizing]

**Monte Carlo Validation**: Dynamic risk (percentage of current balance) offers higher returns and compounding benefits but increased volatility. Fixed risk (percentage of initial balance) provides more stable but potentially less profitable outcomes. The 1–2% rule is validated for most scenarios [MQL5 - Building a Trading System Part 2].

#### Kelly Criterion (Full, Half, Quarter)

**Formula**: `f* = (bp - q) / b` where f* = fraction to wager, p = probability of winning, q = 1-p, b = payoff odds [Wikipedia - Kelly Criterion]

**Alternative Formula**: `Kelly % = (Win Probability × Win/Loss Ratio - Loss Probability) / Win/Loss Ratio` [JournalPlus Kelly Criterion Calculator]

**Academic Research - "Sizing the Risk: Kelly, VIX, and Hybrid Approaches in Put-Writing on Index Options" (arXiv, 2025)** : This paper investigates systematic put-writing strategies on S&P 500 Index options (SPXW, 0–5 DTE) evaluating three sizing methods:
1. **Monte Carlo-based Kelly Criterion**: mathematically optimal approach that maximizes long-term capital growth but requires precise estimates and may lead to high short-run volatility
2. **VIX-based volatility regime scaling**: adjusts exposure based on VIX percentile rank
3. **Hybrid combining both**: allows dynamic adjustment to market regimes; consistently balances strong returns with controlled drawdowns [arXiv - Sizing the Risk: Kelly, VIX, and Hybrid Approaches].

**Lawrence G. McMillan Webinar ("From Kelly to Greeks")** : Covers Kelly Criterion and Optimal f as quantitative position sizing methods. Also covers position delta and position vega for portfolio-level risk monitoring [The Option Strategist - Risk Management Webinar].

**Ralph Vince (Optimal f)** : The Kelly Criterion is a subset of Optimal f. Optimal f is the fraction of capital to risk on each trade that maximizes expected geometric growth. Trading at full Optimal f produces maximum growth but with extreme drawdown risk. Using f/2 (half-Kelly) minimizes the cost of missing the peak and reduces drawdown risk. Secure F adapts Optimal f to ensure Max Drawdown ≤ Max Allowed Drawdown [Better System Trader - Ralph Vince on Position Sizing].

**Key Warnings on Kelly Criterion**:
- Full Kelly produces extreme volatility — 40–60% drawdowns are normal even with profitable strategies [CrossTrade - Risk of Ruin]
- Most professional traders use 25–50% of full Kelly (fractional Kelly) [JournalPlus Kelly Criterion Calculator]
- If Full Kelly exceeds 25%, be cautious — this usually means unreliable data or overly aggressive risk assumptions [JournalPlus Kelly Criterion Calculator]
- A negative Kelly means you have no edge. The optimal bet is zero [JournalPlus Kelly Criterion Calculator]

#### ATR (Average True Range) Based Position Sizing

**Formula**: `Position Size = Account Risk / (ATR × Multiple)` [LuxAlgo]

**Implementation**:
- ATR-based position sizing uses Average True Range to adjust trade size based on market volatility
- It helps maintain consistent risk exposure regardless of volatility
- In high-volatility markets (ATR of $4), it reduces position size; in low-volatility markets (ATR of $1), it allows larger positions
- During the market turbulence of March 2020, this method would have automatically reduced position sizes by cutting risk percentages in half [LuxAlgo]

**Practical ATR Sizing for $700 Account**:
- Risk per trade: $14 (2% of $700)
- ATR-based stop distance: 1.5 to 3.0 × ATR of the underlying
- Position size = $14 / (ATR × 2.0)
- Example: If stock ATR = $2.00, stop distance = $4.00, position size = $14 / $4.00 = 3.5 shares (use 1 spread contract; risk is controlled by spread width)

**TradersPost Guidance**: ATR-based position sizing maintains consistent risk levels across varying market conditions by scaling position sizes inversely to current volatility. Stop loss placement using ATR multiples adapts stops to current market volatility, preventing premature exits [TradersPost - ATR Position Sizing].

#### Monte Carlo Simulation Approaches

**Methodology (Michael R. Bryant, Ph.D., TASC February 2001)** : Monte Carlo simulation is used to determine optimal position sizing by randomizing trade sequences from a trading system. The method repeatedly calculates returns and maximum drawdowns for varying risk fractions. Example: If 1,000 random sequences are simulated with 5% risk, and 940 have max drawdowns < 25%, then there is a 94% probability of max drawdown staying below 25% at that risk level [Technical Analysis of Stocks & Commodities - Michael R. Bryant, Ph.D.].

**TradeZella Monte Carlo Simulator**: Generates 1,000+ randomized equity curves based on win rate, risk-reward ratio, and position sizing. Key outputs include Probability of Profit (POP) — professional traders target above 65% (world-class strategies above 80%); Risk of Ruin (RoR) — probability account drops below a defined drawdown threshold; median and worst-case drawdowns [TradeZella - Free Monte Carlo Simulator].

**Ralph Vince on Monte Carlo**: Position sizing through Monte Carlo methods is critical because "trade too small and your portfolio doesn't grow near as much as it should; trade too large and you could bankrupt your account" [The Systematic Trader - Ralph Vince and Position Sizing].

---

## 3. Permitted & Prohibited Strategies — Full Boundaries

### 3.1 Permitted Strategies with Detailed Rules

#### Strategy 1: Debit Spreads (Bull Call Spread / Bear Put Spread) — BEST for $700

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 2 or 3 | [Fidelity] [Schwab] [tastytrade] |
| **Buying Power Required** | Net debit paid (typically $50–$300) | [tastytrade] |
| **Max Risk** | Premium paid (net debit) | [Fidelity] [Option Alpha] |
| **Max Profit** | (Spread width × 100) – net debit paid | [tastytrade] |
| **Complexity Rating** | Intermediate | [Fidelity] — Level 3 strategy |

**Risk Limits:**
- Max premium paid = the debit paid to enter the spread
- For a $1-wide debit spread costing $0.50: max loss = $50 per contract
- For a $2.50-wide debit spread costing $1.20: max loss = $120 per contract
- Option Alpha: "Focus on small allocations per trade, ideally 1 to 5% of account size" [Option Alpha]

**Max Premium Paid Rules:**
- For a $700 account: 1–5% = $7–$35 max premium at risk per trade (per Option Alpha guidelines)
- For a $1,000 account: 1–5% = $10–$50 max premium at risk per trade

**Minimum Capital Required:**
- Debit spreads require the full premium to be paid upfront
- For a $0.50 debit spread: $50 capital required per contract
- For a $1.00 debit spread: $100 capital required per contract

**Key Rule for $700:** Only use **$1–$2 wide spreads** on highly liquid underlyings. A $1 wide spread costs typically $50–$150, keeping per-trade risk manageable.

#### Strategy 2: Credit Spreads (Bull Put Spread / Bear Call Spread)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 3 (margin account required) | [Fidelity] [Schwab] [tastytrade] |
| **Buying Power Required** | Width of spread × 100 (e.g., $1 wide = $100) | [tastytrade] |
| **Max Risk** | (Width × 100) – net credit received | [tastytrade] |
| **Max Profit** | Net credit received | [tastytrade] |
| **Complexity Rating** | Intermediate | [Fidelity] — Level 3 |

**Risk Limits:**
- Max spread width: tastytrade research recommends selling credit equal to **1/3 the spread width** [tastytrade Research]
- OptionsPlay: "The 33% rule – investors should always look to receive a minimum of 33% of the vertical width in premium" [OptionsPlay]
- Option Alpha: "Looks for a $5, then $4, then $3 spread from the 0.30 delta, targeting 30% credit of spread width at 35–45 DTE" [Option Alpha]
- Maximum risk = (spread width × 100) – credit received per contract
- Maximum reward = credit received per contract
- OptionsPlay: "The maximum risk of a trade should not be more than 2% of your account" [OptionsPlay]

**Max Credit Received Rules:**
- For a $1-wide spread: minimum credit = $0.33 (tastytrade rule)
- For a $2.50-wide spread: minimum credit = ~$0.83
- For a $5-wide spread: minimum credit = ~$1.67

**Win Rate Statistics:**
- Backtest results (SPX 2005–2016, tastytrade rules): **61% win rate** [tastytrade Research]
- Cboe Options Exchange data: **45 DTE credit spreads with 25–30 delta short strikes achieve approximately 70–75% win rates** when held to expiration [Cboe Options Institute]
- "45-day spreads deliver the best risk-adjusted returns" [Cboe Options Institute]
- tastytrade studies recommend: "begin trade 45 DTE; only trade when IV Rank is from 50% to 100%; manage winners at 50% credit; stop losses if spread price reaches 2X the credit" [tastytrade Research]

**Key Rule for $700:** Only use **$1 wide spreads**. A $2 wide spread requires $200 in buying power and risks up to $200 (28.6% of your account). **The 33% Rule**: Always receive a minimum of 33% of the spread width in premium. For a $1 wide spread, you need at least $0.33 credit.

#### Strategy 3: Long Calls and Long Puts (Single-Leg) — Use Sparingly

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 1 or 2 | [Fidelity] [Schwab] [tastytrade] |
| **Buying Power Required** | Full premium paid | [tastytrade] |
| **Max Risk** | 100% of premium paid | [Fidelity] |
| **Max Profit** | Theoretically unlimited (calls) | [Fidelity] |
| **Complexity Rating** | Beginner | [Fidelity] — "Beginner options traders will often start by buying calls or puts" |

**Risk Limits:**
- Maximum loss is limited to the premium paid plus commissions [Fidelity]
- "The maximum loss is limited to the call purchase price, plus commission" [Fidelity]
- For a $700 account at 1–5% allocation: max premium = $7–$35 per trade
- For a $1,000 account at 1–5% allocation: max premium = $10–$50 per trade

**Max Premium Rules:**
- Option Alpha: "Focus on small allocations per trade, ideally 1 to 5% of account size" [Option Alpha]
- For a $500–$1,000 account: 1–5% = $5–$50 maximum premium at risk for a single long option

**Verdict:** Use only with small premium. Limit to options costing **$1.00 or less per contract** (max $100 risk = 14.3% of account). A 30-day option loses ~3.3% of time value per day, making long options a low-probability play [HaiKhuu Trading].

#### Strategy 4: Cash-Secured Puts — Use with Caution on Low-Priced Stocks

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 2 (margin account required at some brokers) | [Fidelity] [tastytrade] |
| **Buying Power Required** | Strike × 100 (full cash collateral) | [OIC] [Fidelity] |
| **Max Risk** | (Strike – stock price at expiration) × 100 – premium received | [OIC] |
| **Max Profit** | Premium received | [OIC] |
| **Complexity Rating** | Beginner to Intermediate | [OIC] |

**Risk Limits:**
- Maximum loss is limited but substantial — the worst case is the stock becomes worthless and the investor buys at the strike price minus the premium received [OIC]
- Maximum gain is limited to the premium received if the stock price stays above the strike price and the put expires worthless [OIC]
- Breakeven = Strike price – Premium [OIC] [Fidelity]

**Minimum Capital Required:**
- The investor must hold cash equal to (strike price × 100 shares per contract) to cover potential assignment [OIC] [Fidelity] [Schwab]
- For a $700 account: maximum affordable strike is $7.00 per share (1 contract = 100 shares × $7.00 = $700)
- For a $1,000 account: maximum strike is $10.00 per share

**Verdict for $700:** Only feasible on stocks under $7.00/share. Most stocks with liquid options trade at $20+, requiring $2,000+ in cash to secure. Therefore, **generally not feasible** for a $700 account [OIC] [Fidelity].

#### Strategy 5: Covered Calls — NOT FEASIBLE for $700 Account

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 1 | [Fidelity] [Schwab] [tastytrade] |
| **Buying Power Required** | Must own 100 shares + sell 1 call | [tastytrade] |
| **Max Risk** | Full decline in stock price minus premium received | [Fidelity] |
| **Max Profit** | (Strike – stock purchase price + premium) × 100 | [Fidelity] |
| **Complexity Rating** | Beginner | [Fidelity] — Level 1 strategy |

**Risk Limits:**
- Maximum gain is capped at (strike price – stock purchase price + premium received) × 100 [Schwab]
- Downside protection is limited only to the premium received — if the stock drops significantly, losses are the full decline minus the premium [Schwab]
- "Covered calls provide downside protection only to the extent of premiums received, and prevent any profitability above the strike price of the call" [Schwab]

**Minimum Capital Required for $700 Account:**
- A $700 account can afford to buy a maximum of $7.00 per share stock (100 shares × $7.00 = $700)
- To sell a covered call, the investor must own 100 shares of the underlying stock per contract sold [Schwab]
- Most stocks with liquid options trade at $20+, requiring $2,000+ to own 100 shares

**Verdict:** Not feasible for $700 account. You need to own 100 shares first. With $700, you could only buy 100 shares of stocks under $7.00 per share, and most stocks under $7 don't have liquid options [TradingStrategyGuides].

#### Strategy 6: Iron Condors (Narrow) — Advanced Only

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 3 | [Fidelity] [tastytrade] |
| **Buying Power Required** | Spread width × 100 (one side) | [tastytrade] |
| **Max Risk** | Spread width – net credit received | [tastytrade] |
| **Max Profit** | Net credit received | [tastytrade] |
| **Complexity Rating** | Intermediate to Advanced | [tastytrade] |

**Risk Limits:**
- An iron condor is two credit spreads — it "collects roughly 2× the credit of a single credit spread at the same wing width with the same capital tied up" [tastytrade Research]
- Maximum loss = spread width – net credit received, times 100 per contract [tastytrade]
- Maximum profit = net credit received [tastytrade]
- tastylive: "We close iron condors when we reach 50% of our max profit. This can increase our win rate over time" [tastylive]

**Win Rate Statistics:**
- Backtests (SPY 2014–2026): iron condors have a realized win rate of **78.9%** and annualized return on margin of ~21% [tastytrade Research]
- Credit spreads comparison: **73.4% win rate** and ~14% annualized return on margin [tastytrade Research]
- "The Iron Condor has a probability at expiration of over 80% while the Butterfly is closer to 40%" [tastytrade Research]

**Verdict:** For a $700 account, only use **$1 wide wings** on highly liquid ETFs like SPY or QQQ. However, the four commissions and wider bid-ask spreads make this challenging for very small accounts. Not recommended for beginners.

#### Strategy 7: Poor Man's Covered Call (PMCC) — Advanced Only

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 2 or 3 | [Fidelity] [tastytrade] |
| **Capital Required** | Cost of LEAPS call ($200–$600) | [TradingStrategyGuides] |
| **Max Risk** | Net debit paid | [TradingStrategyGuides] |
| **Max Profit** | Unlimited (minus premium paid) | [TradingStrategyGuides] |
| **Complexity Rating** | Advanced | [TradingStrategyGuides] |

**Example:** Stock at $10, buy a $8 LEAPS call (12 months out, delta 0.75) for $2.50 ($250). Sell a $11 call (30 DTE, delta 0.30) for $0.25 ($25). Net debit: $225 (32% of account). Monthly premium: $20–$30 [TradingStrategyGuides].

**Verdict:** Requires 32%+ of the account in one position. Only for experienced traders who can afford to lose the entire position. Not recommended for $700 account.

### 3.2 Prohibited Strategies with Clear Reasoning

#### Naked Options (Uncovered Short Calls/Puts)

**Why Prohibited for Small Accounts:**
- Naked short calls have **unlimited risk potential** — "naked short calls aren't allowed in IRAs due to their potentially unlimited risk" [FINRA]
- Regulation T: Requires an initial margin deposit of 50% of the transaction value plus the option premium [FINRA Rule 4210]
- At Fidelity, naked writing of equity options requires **Level 4 approval**; uncovered writing of index options requires **Level 5** [Fidelity]
- At Schwab, naked options require **Level 4+** [Schwab]
- At tastytrade, naked options require **"The Works" level** [tastytrade]
- Naked requirements include a **$20,000 minimum for naked equity call positions** at Fidelity [Fidelity]
- A small account ($500–$1,000) **cannot meet the maintenance margin requirements** for naked short options
- "An option is considered naked when you sell an option without owning the underlying asset or having the cash to cover the exercisable value" [Fidelity]

#### 0DTE (Zero Days to Expiration) Options

**Why Prohibited for Small Accounts:**
- "Trading 0DTE is a very high-risk strategy. Unless you have a scalping or arbitrage strategy, 0DTE options trading isn't too different from gambling" [FINRA]
- FINRA: "The number of opening 0DTE option contracts positions increased approximately 60 percent between January 2022 and January 2023" [FINRA]
- FINRA: "Any strategy that can quickly earn profits can quickly bring losses as well. That includes 0DTE options, which are very sensitive to changes in the price of the asset underlying the option" [FINRA]
- "Schwab does not recommend day-trading strategies, including the opening of options transactions on their expiration dates, also known as 'zero days till expiration' or '0DTE options' trading" [Schwab]

**Gamma Risk Specifics:**
- "Because they are short-lived instruments, 0DTE options are subject to significant volatility and require close monitoring" [FINRA]
- Gamma risk near expiration becomes extremely high — small price moves cause large swings in option value
- "Roughly 1.5 million 0DTE options trade daily, accounting for nearly half of all options trades tied to the S&P 500 index (SPX), according to 2025 data from Cboe Global Markets" [Cboe]

#### Earnings Plays

**Why Prohibited for Small Accounts:**
- "Earnings break the rules of normal options pricing. An option that doubled last week can be worthless tomorrow" [Investopedia]
- "IV crush is what separates professional earnings options plays from a coin flip" [Investopedia]
- Implied volatility can drop by **30–50% immediately after earnings** [Investopedia]
- "A common rule is to risk no more than 1 to 2 percent of account on a single earnings trade" [Investopedia]
- Binary outcomes create all-or-nothing risk that small accounts cannot absorb
- Options are priced with elevated IV before earnings, meaning traders pay inflated premiums — a losing proposition for small accounts buying premium

#### Options on Low-Priced Stocks (Under $5, $10, or $20)

**Why Prohibited for Small Accounts:**
- SEC defines penny stocks as securities priced under $5 with very low market capitalizations [SEC Rule 3a51-1]
- "Penny stocks are typically stocks issued by very small companies that trade at less than $5 per share" [FINRA]
- "Low-priced securities often are considered speculative investments, which you should only make with money that you can afford to lose" [FINRA]
- Many brokerages restrict options trading on stocks below $5 or $10 per share
- These stocks have **wider bid-ask spreads**, making it harder to enter and exit positions profitably
- FINRA: "Low-priced securities also can be targets for pump and dump and similar schemes" [FINRA]
- Most educational sources suggest avoiding options on stocks under **$10–$20** for small accounts, with $5 being the absolute minimum (SEC penny stock definition)

#### Options on Leveraged ETFs

**Why Prohibited for Small Accounts:**
- "Leveraged ETFs use derivatives or internal borrowing to magnify daily performance but are highly volatile and unsuitable for most long-term holding strategies" [FINRA]
- "Even if the underlying stock's price returns to its starting point after a volatile period, the compounding effect on daily returns can cause a leveraged ETF to lose value — a phenomenon known as **volatility decay**" [FINRA]
- FINRA Regulatory Notice 09-53: Implemented "increased customer margin requirements for leveraged ETFs and uncovered options overlying leveraged ETFs" [FINRA]
- "Maintenance margin requirements for leveraged ETFs will increase proportionally to their leverage, not to exceed 100% of the value" [FINRA]
- Small accounts cannot withstand the rapid decay and volatility of leveraged ETF options

#### Multi-Leg Strategies with >3 Legs (Complex Adjustments)

**Why Prohibited for Small Accounts:**
- Commission costs for 4+ leg strategies can be prohibitive relative to account size
- Multiple legs increase complexity of management and risk monitoring
- While a standard iron condor (4 legs) is permitted at Level 3 for Fidelity and Basic for tastytrade, adjusted condors or butterflies with additional legs become Level 4+ strategies
- Box spreads (4 legs) require careful European-style options to avoid early assignment risk — American-style options introduce hidden risks that could lead to losing much more money than expected [Robinhood]

#### Box Spreads, Ratio Spreads, Diagonals with Wide Strikes

**Box Spreads:**
- "Box spreads are often mistaken for an arbitrage opportunity, however, they have hidden risks that could lead to losing much more money than expected" [Robinhood]
- American options can be exercised early, introducing risks that made box spreads dangerous
- The infamous Reddit case (user 1R0NYMAN): resulted in a loss of more than $57,000 attempting a box spread, leading Robinhood to ban box spread opening [Robinhood]
- OCC: "Due to transaction costs and complexities, the box spread is mainly suited for professional traders and market makers" [OCC]

**Ratio Spreads:**
- Ratio spreads involve selling more options than are bought, creating **naked exposure** on the extra contracts
- This creates undefined risk similar to naked options, requiring higher margin and approval levels (Level 4+ at Fidelity)

#### Short Puts on High-Volatility Names Without Sufficient Buying Power

**Why Prohibited:**
- "One risk of selling cash-secured puts is that if the stock price declines, you're still required to buy at the (higher) set price if the put is exercised" [Fidelity]
- On high-volatility names, the stock could gap down far below the strike price, requiring far more cash than was originally set aside
- "The maximum loss is comparable to owning the underlying stock" [Schwab] — For high-volatility names, this can be a total loss
- tastytrade recommends only trading credit spreads "when IV Rank is from 50% to 100%" [tastytrade]
- Selling puts on individual high-volatility names carries idiosyncratic risk beyond what IV rank captures

### 3.3 Broker Approval Levels and Requirements

**FINRA Rule 2360 Options Approval Framework:**
- Members must perform due diligence on the customer and collect detailed information such as knowledge, investment experience, age, financial situation, and investment objectives [FINRA Rule 2360]
- Members must specifically approve or disapprove customers prior to accepting options orders [FINRA Rule 2360]
- Members must furnish customers with the "Characteristics and Risks of Standardized Options" disclosure document [FINRA Rule 2360]
- FINRA Rule 2360 requires: customer completion of a new account form, delivery of the Options Disclosure Document, approval by an options principal, first trade opening the account, and customer returning a signed options agreement within 15 days [FINRA Rule 2360]

**Fidelity Options Levels (5 Levels):**
- **Level 1**: Covered call writing of equity options [Fidelity]
- **Level 2**: Purchases of calls and puts, plus covered puts [Fidelity]
- **Level 3**: Spreads and covered put writing (credit spreads, debit spreads, iron condors) [Fidelity]
- **Level 4**: Naked (uncovered) call and put writing of equity options [Fidelity]
- **Level 5**: Naked index options (deep out-of-the-money uncovered options) [Fidelity]

**Schwab Options Levels:**
- **Level 1**: Covered calls only [Schwab]
- **Level 2**: Buying calls and puts, cash-secured puts [Schwab]
- **Level 3**: Spreads (defined-risk strategies) [Schwab]
- **Level 4+**: Naked options [Schwab]

**tastytrade Options Levels:**
- **Limited**: Buying options, covered calls [tastytrade]
- **Basic**: Spreads, selling cash-secured puts [tastytrade]
- **The Works**: Naked options, advanced strategies [tastytrade]
- Tastytrade requires a margin account for credit spreads and selling puts [tastytrade]

**FINRA New Intraday Margin Requirements (Effective June 4, 2026):**
- The $25,000 minimum equity requirement for pattern day traders (PDT) is **eliminated** [FINRA]
- New rule requires adequate maintenance margin of at least **25% of the current market value** of margin-eligible securities throughout the entire trading day [FINRA]
- Minimum account equity drops from $25,000 to **$2,000**, matching the standard margin account requirement [FINRA]
- "Options defined-risk spreads will be margined exactly at their maximum theoretical loss, improving capital efficiency significantly for strategies like iron condors" [FINRA]
- A concentrated position surcharge adds 15 percentage points margin when a single position exceeds 50% of total intraday exposure [FINRA]
- Effective date: June 4, 2026; 18-month phase-in period through October 20, 2027 [FINRA]

---

## 4. Technical Criteria & Option Chain Selection — Advanced Screening Rules

### 4.1 Optimal Stock Price Bands for $700 Accounts

**Authoritative Guidance from Tom Sosnoff and Tony Battista (tastylive):**
Using ETFs like **IWM** (iShares Russell 2000 ETF) instead of **SPY** (SPDR S&P 500 ETF) is recommended because of their lower buying power requirements, which better suit smaller trading accounts. The guide advises avoiding strategies that require large buying power by utilizing sector ETFs [tastylive].

**OptionsPlay Guidance:**
For accounts within $2,000 to $5,000, investors need Level 3 options clearance and a margin account with a minimum balance of $2,000 for credit spreads in many cases. However, for a $700 account, traders can still use credit spreads on lower-priced stocks [OptionsPlay].

**Optimal Stock Price Bands for $700 Account:**

| Stock Price Range | Feasibility | Notes |
|-------------------|-------------|-------|
| **$10–$50** | **Optimal** | Allows $1-wide spreads with $100 max risk; fits within 2–15% risk guidelines |
| **$50–$80** | Acceptable | $2.50-wide spreads risk $250; use with caution |
| **$80–$100** | Challenging | $5-wide spreads risk $500 (71% of account); not recommended |
| **$100+** | **Not recommended** | Spread widths too large for a $700 account |
| **Under $10** | Caution | Low-liquidity stocks; $5 absolute minimum (SEC penny stock definition) |

**Additional Rules:**
- Maximum risk of a trade should not be more than **2% of your account** ($14 max risk per trade) [OptionsPlay]
- No more than **five simultaneous trades** to cap total risk at 10% of account [OptionsPlay]

### 4.2 Optimal Option Premium Bands

**Credit Spreads — 33% Rule:**
Always look to receive a minimum of **33% of the vertical width** in premium. For a $1 wide spread ($100 max risk), collect at least $0.33 in credit ($33). For a $2.50 wide spread, collect at least $0.83 in credit [OptionsPlay] [tastytrade].

**Debit Spreads — Max Premium Paid:**
For a $700 account: 1–5% = $7–$35 max premium at risk per trade [Option Alpha]. This means debit spreads should cost no more than $7–$35 net debit per contract.

**50/25 Delta Rule for Credit Spreads:**
Sell the **50 delta** and buy the **25 delta** — research shows this provides the best risk/reward for Credit Spread trades [OptionsPlay].

**Realistic Premium Bands for $700 Account:**

| Strategy | Premium Per Contract | $ on $700 Account |
|----------|---------------------|-------------------|
| Credit Spread (1-wide) | $0.33–$0.50 credit | $33–$50 max profit |
| Debit Spread (1-wide) | $0.30–$0.70 debit | $30–$70 risk |
| Long Call/Put | $0.25–$1.00 premium | $25–$100 risk |

### 4.3 Volume, Open Interest, and Bid-Ask Spread Thresholds

**The VOSS Framework (TradingBlock):**
Liquidity is essential in options trading. If you don't understand and master the liquidity metrics discussed below, you will inevitably lose money over time when trading options [TradingBlock].

**VOSS Components:**
- **V**olume: Number of contracts traded daily
- **O**pen Interest: Total outstanding active contracts
- **S**preads: The tighter the spread, the more liquid the option
- **S**ize: Bid/Ask size in contracts available

**Tiered Liquidity Thresholds (Consolidated from All Authoritative Sources):**

| Metric | Minimum (Absolute) | Preferred | Ideal | Source |
|--------|-------------------|-----------|-------|--------|
| **Stock Volume (shares/day)** | 500,000 | 1,000,000 | 5,000,000+ | [Tackle Trading] [TradingBlock] |
| **Option Volume (contracts/day)** | 100 | 500 | 1,000+ | [tastylive] — "We prioritize options with over a thousand contracts traded daily" |
| **Open Interest (contracts)** | 200 | 500 | 1,000+ | [TradingBlock] — "High open interest is synonymous with high liquidity" |
| **Bid-Ask Spread ($)** | $0.10 | $0.05 | $0.02 | [Reddit Options Community] |
| **Bid-Ask Spread (% of mid-price)** | 10% | 5% | 2% | [Tackle Trading] |
| **Bid/Ask Size (contracts)** | 5 | 10 | 20 | [TradingBlock] — "Size is perhaps the most frequently overlooked component" |

**Cboe Insights on Liquidity:**
"There may be 'off-screen liquidity' invisible to your platform's digital eye." Market orders can result in "slippage," which is the difference between the price you wanted and the unfavorable price you received. "A limit order allows you to define a specific price as your absolute limit" [Cboe Insights].

### 4.4 IV Rank and IV Percentile Strategy Selection Rules

**tastylive's IV Framework:**
"When Implied Volatility Rank is low we use debit spreads. When Implied Volatility Rank is high we use credit spreads" [tastylive].

**Definitions:**
- **IV Rank (IVR)**: Reports how the current level of implied volatility compares to the last 52 weeks of historical data. Scale between 0–100, where 0 represents the low IV% print for the year, and 100 represents the high IV% print [tastylive].
- **IV Percentile (IVP)**: Reports the percentage of days over the last 52 weeks that implied volatility traded below the current level of implied volatility [tastylive].

**Charles Schwab Guidance:**
"As a general rule, some traders consider buying a **debit spread when IV is between the 0 to 50th percentile** of its 52-week range, and selling a **credit spread when IV is greater than the 50th percentile**" [Schwab].
"Debit spreads typically have **positive vega** and benefit when IV rises over time; credit spreads typically have **negative vega** and benefit when IV falls over time" [Schwab].

**Practical IV-Based Strategy Selection Rules:**

| IV Rank / Percentile | Strategy | Rationale | Source |
|----------------------|----------|-----------|--------|
| **0–30 (Low)** | Debit spreads, long options | Options are "cheap"; buy premium | [tastylive] [Schwab] |
| **30–50 (Moderate)** | Either (based on directional bias) | Neutral zone; wait for better conditions or use directional edge | [tastylive] |
| **50–70 (High)** | Credit spreads, short options | Options are "expensive"; sell premium | [tastylive] [Schwab] |
| **70–100 (Extreme)** | Credit spreads, iron condors | Aggressively sell premium; high tail risk | [tastylive] |
| **> 80 (Very High)** | Strangles, iron condors with wide wings | IV is likely to revert downward | [tastylive] |

**SJ Options Backtest Caveat (2005–2015):**
An 11-year backtest of SPX credit spreads sold at different IV ranks found: "Contrary to claims by Tasty Trade that selling option premium at high IV Rank offers a trading advantage, the results indicate **no profitability edge** when selling at IV Rank above 50%." Credit spreads initiated at low IV Rank outperformed those at high IV Rank, primarily because low IV Rank reduced average losses by 27% while marginally reducing average winners by 1.2% [SJ Options Research].

**JournalPlus Put Credit Spread Guidance:**
Enter during elevated implied volatility (**IV Rank ≥ 30%**). Select expirations **30–45 days out**. Choose short strikes at the **20–30 delta put** to balance credit and probability of profit [JournalPlus].

**DaysToExpiry Guidance:**
"Avoid selling premium when **IV percentile is below 30** unless you have a clear directional edge" [DaysToExpiry].

### 4.5 ATR-Based Position Sizing Rules

**Formula**: `Position Size = Account Risk / (ATR × Multiple)` [LuxAlgo]

**Implementation for $700 Account:**
- Risk per trade: $14 (2% of $700)
- ATR-based stop distance: 1.5 to 3.0 × ATR of the underlying
- ATR multiplier selection: **1.5× ATR** for aggressive (tighter stops), **2.0× ATR** for standard, **3.0× ATR** for conservative (wider stops)
- Position size = $14 / (ATR × 2.0)

**Examples:**
- Stock ATR = $1.00, multiplier = 2.0, stop distance = $2.00: Position = $14 / $2.00 = 7 shares (use 1 credit spread; risk controlled by spread width)
- Stock ATR = $2.00, multiplier = 2.0, stop distance = $4.00: Position = $14 / $4.00 = 3.5 shares (use 1 credit spread)
- During high volatility (ATR = $5.00): stop distance = $10.00, position = $14 / $10.00 = 1.4 shares (still 1 spread contract, but choose wider strikes)

**LuxAlgo Guidance:**
"During the market turbulence of March 2020, ATR sizing would have automatically reduced position sizes by cutting risk percentages in half. For instance, during extreme volatility, consider using a longer ATR period, like 20–30 days" [LuxAlgo].

**TradersPost Guidance:**
ATR-based position sizing maintains consistent risk levels across varying market conditions by scaling position sizes inversely to current volatility. Stop loss placement using ATR multiples adapts stops to current market volatility, preventing premature exits [TradersPost].

### 4.6 Technical Indicator Screening with Explicit Thresholds

#### RSI (Relative Strength Index — 14 period)

| RSI Range | Signal | Options Action | Source |
|-----------|--------|----------------|--------|
| **< 20** | Extreme oversold | High-probability bounce; buy calls / put spreads (long) | [Investopedia] [ChartingLens] |
| **20–30** | Oversold | Potential bounce; buy calls / bull put spreads | [Investopedia] [Tradetron] |
| **30–40** | Near oversold | Watch for bounce off support; moderate bullish bias | [Investopedia] |
| **40–60** | Neutral | Avoid directional strategies based solely on RSI | [Investopedia] |
| **60–70** | Near overbought | Watch for breakdown; moderate bearish bias | [Investopedia] |
| **70–80** | Overbought | Potential pullback; buy puts / bear call spreads | [Investopedia] [Tradetron] |
| **> 80** | Extreme overbought | High-probability pullback; aggressive put buying / bearish spreads | [Investopedia] |

**Quantified Strategies RSI Divergence:**
For breakout/breakdown trading: RSI reading above 70 confirms bullish breakout momentum; RSI below 30 confirms bearish breakdown. Volume confirmation typically 120–300% above 20-bar average for breakouts [Quantified Strategies].

**CFA Institute Studies Reference:**
"RSI's speed in detecting momentum shifts outperforms lagging averages, with CFA Institute studies showing 65–70% accuracy in mean-reversion trades" [AquaFutures].

#### MACD (12, 26, 9) Rules

| MACD Condition | Signal | Options Action | Source |
|----------------|--------|----------------|--------|
| MACD line crosses **above** signal line | Bullish momentum strengthening | Buy calls / bull put spreads | [QuantInsti] [ICFM India] |
| MACD line crosses **below** signal line | Bearish momentum strengthening | Buy puts / bear call spreads | [QuantInsti] [ICFM India] |
| MACD line crosses **above** zero line | Momentum shifting to upside | Stronger bullish bias | [Investopedia] |
| MACD line crosses **below** zero line | Momentum shifting to downside | Stronger bearish bias | [Investopedia] |
| **Bullish divergence** (price lower low, MACD higher low) | Strong bullish reversal signal | High-conviction call buying | [Investopedia] |
| **Bearish divergence** (price higher high, MACD lower high) | Strong bearish reversal signal | High-conviction put buying | [Investopedia] |

**ICFM India Guidance:**
"A bullish setup occurs when RSI emerges from oversold territory and MACD line crosses above the signal line." Combining RSI and MACD reduces false signals and increases trade confidence [ICFM India].

#### EMA/SMA Rules

| Moving Average Condition | Signal | Options Action | Source |
|--------------------------|--------|----------------|--------|
| Price **above** 20-day EMA | Bullish short-term trend | Favor calls / bull spreads | [Tradetron] [Option Samurai] |
| Price **below** 20-day EMA | Bearish short-term trend | Favor puts / bear spreads | [Tradetron] [Option Samurai] |
| Price **bouncing off** 20-day EMA | Support level confirmation | Bullish entry — buy calls | [Option Samurai] |
| Price **bouncing off** 50-day MA | Stronger support | Higher probability bullish setup | [Option Samurai] |
| Price **above** 200-day SMA | Bullish long-term trend | Higher confidence for long positions | [Option Samurai] |
| Price **below** 200-day SMA | Bearish long-term trend | Higher confidence for short positions | [Option Samurai] |
| **9 EMA crosses above** 21 EMA | Short-term bullish crossover | Buy signal | [Bulls on Wall Street] |
| **9 EMA crosses below** 21 EMA | Short-term bearish crossover | Sell signal | [Bulls on Wall Street] |

**Critical Filter:** Only trade crossovers in the direction of the higher timeframe trend. Enter on the next candle after the crossover, not the crossover candle itself [Bulls on Wall Street].

#### Volume Confirmation Rules

| Volume Condition | Implication | Source |
|-----------------|-------------|--------|
| **Volume > 1.5× average** | Minimum threshold for meaningful price move | [ThinkorSwim Research] |
| **Volume > 2.0× average** | Strong confirmation of breakout/breakdown | [ThinkorSwim Research] [QuantVPS] |
| **Volume > 3.0× average** | Climactic/exhaustion volume; potential reversal | [QuantVPS] |
| **Low-volume breakout** | Prone to failure; price likely to retrace | [QuantVPS] |

**ThinkorSwim Relative Volume Research:**
"The RVOL sweet spot is **1.5–2.0**, with **58.8% three-day follow-through** and the highest average returns in our 1,872-event backtest." RVOL between 1.5 and 2.0 yields the highest breakout follow-through rates (58.8% 3-day follow-through) and average returns (+0.76% over 5 days) [ThinkorSwim Research].

"Volume is the fuel behind every price move. Without it, breakouts fail, trends stall, and reversals lack conviction" [ThinkorSwim Research].

#### Complete Entry Criteria (Synthesized from All Sources)

**Bullish Entry (Buy Calls / Bull Put Spread):**
- Price bounces off 20-day MA with bullish candlestick pattern (bullish engulfing, pin bar)
- RSI < 40 turning upward (oversold bounce)
- Volume > 1.5× average on the bounce
- MACD crossing above signal line OR histogram turning positive
- 9 EMA above 21 EMA on daily chart
- VWAP: Price above VWAP or bouncing off VWAP as support
- Bollinger Bands: Squeeze preceding the move, or bounce off lower band
- Multi-timeframe confirmation: Daily uptrend + 4-hour entry signal

**Bearish Entry (Buy Puts / Bear Call Spread):**
- Price rejects off resistance (20-day MA, swing high, round number)
- RSI > 60 turning downward (overbought rollover)
- Volume > 1.5× average on the rejection
- MACD crossing below signal line OR histogram turning negative
- 9 EMA below 21 EMA on daily chart
- VWAP: Price below VWAP or rejecting VWAP as resistance
- Multi-timeframe confirmation: Daily downtrend + 4-hour entry signal

### 4.7 VWAP-Based Rules

**ThinkorSwim Research:**
"VWAP serves as the institutional benchmark and creates dynamic support and resistance throughout the trading day." VWAP is especially effective during the first trading hours. Anchored VWAP extends this concept for multi-day analysis from specific events like earnings [ThinkorSwim Research].

**Brian Shannon's AVWAP Strategy (FinancialWisdomTV):**
"The A-V-WAP broadcasts the message of the market, that is whether sellers or buyers are driving the then-current price trend more clearly than any other technical tool" [FinancialWisdomTV].
- Price **above** upward-sloping AV-WAP: "innocent until proven guilty" (bullish)
- Price **below** AV-WAP: "guilty until proven innocent" (bearish)
- **First one or two touches** on AV-WAP anchored to an important point are more likely to see strong moves
- Common errors: using AVWAP in choppy/sideways markets, anchoring randomly, entering on the first touch without confirmation [FinancialWisdomTV]

**Practical VWAP Rules for Options:**

| Condition | Bias | Action |
|-----------|------|--------|
| Price above VWAP | Bullish | Favor long calls or bull put spreads |
| Price below VWAP | Bearish | Favor long puts or bear call spreads |
| Bounce off VWAP as support (from below) | Bullish | Enter bullish trade |
| Break below VWAP on high volume (from above) | Bearish | Enter bearish trade |
| First touch of VWAP (from above or below) | Higher probability reaction | Monitor for confirmation |
| Multiple touches of VWAP | Weakening significance | Reduce position size |

### 4.8 Bollinger Band Squeeze Rules

**StockCharts.com Definition:**
"A Bollinger Band Squeeze is a condition that occurs when the Bollinger Bands narrow due to decreased volatility. According to John Bollinger, periods of low volatility are often followed by periods of high volatility" [StockCharts.com].

**BandWidth Indicator:**
Measures the distance between the bands relative to price. "Narrow BandWidth near the low end of its six-month range signals a potential squeeze" [StockCharts.com].

**Bollinger Band Squeeze Thresholds:**

| BandWidth Value | Condition | Implication | Source |
|-----------------|-----------|-------------|--------|
| **< 0.10 (10%)** | Squeeze | Potential impending breakout | [StockCharts.com] |
| **< 0.04 (4%)** | Extreme squeeze | High probability of near-term expansion | [StockCharts.com] |
| **> 0.25 (25%)** | Wide bands | High volatility; potential breakout continuation or reversal | [StockCharts.com] |

**Trading Rules for Bollinger Band Squeeze:**

1. **Entry**: When bands are at their narrowest (BandWidth < 0.10), prepare for directional move. Do not enter direction until confirmed.
2. **Breakout Confirmation**: Price closes **outside the bands** signals the breakout direction. Volume should be > 1.5× average.
3. **Head Fake Warning**: Beware of the "head fake" where prices break a band but then reverse. Wait for a close outside the band with volume confirmation [StockCharts.com].
4. **Volume Confirmation**: Use Chaikin Money Flow or On Balance Volume to help anticipate the direction of the breakout [StockCharts.com].

**LuxAlgo Bollinger Bands Strategy:**
"Combine the Bollinger Band Squeeze with complementary indicators like RSI, MACD, or volume analysis to filter out false signals and confirm genuine breakout opportunities." "Start with smaller positions and increase only after confirming breakout direction." "Use ATR-based stop-loss levels, setting them at 1.5× to 3× ATR to allow for larger moves" [LuxAlgo].

**Interactive Brokers:**
"Bollinger Bands widen during periods of high volatility and narrow during low volatility" [Interactive Brokers].

### 4.9 Multi-Timeframe Confirmation Requirements

**Top-Down Approach (Recommended by All Authoritative Sources):**

| Timeframe Pair | Use | Source |
|----------------|-----|--------|
| **Daily + 4-Hour** | Primary for trend direction (Daily); secondary for entry timing (4H) | [LearnToTradeTheMarket] [Tradeciety] |
| **Daily + 4-Hour + 1-Hour** | Triple confirmation for higher-conviction trades | [LearnToTradeTheMarket] |
| **Weekly + Daily** | Long-term structure + intermediate trend | [Tradeciety] |
| **4-Hour + 1-Hour** | For shorter holding periods (3–7 days) | [Tradeciety] |

**Nial Fuller (LearnToTradeTheMarket) Guidance:**
"I never go lower than the 1-hour chart because any timeframe under the 1-hour is just noise" [LearnToTradeTheMarket].
"The intraday charts work as an extra point of confluence to give weight to a trade and to fine-tune entry for better risk management" [LearnToTradeTheMarket].
"Using intraday charts can improve risk-reward by allowing tighter stop losses and larger position sizes while keeping profit targets, effectively increasing potential rewards" [LearnToTradeTheMarket].
"It is NOT day trading; the initial trade trigger is still the higher timeframe chart" [LearnToTradeTheMarket].

**Tradeciety Guidance:**
"Multi-timeframe trading describes a trading approach where the trader combines different trading timeframes to improve decision-making and optimize their chart analyses" [Tradeciety].
"One of the biggest mistakes traders make is starting their analysis on the lower timeframes instead of the recommended top-down approach" [Tradeciety].
Pick one timeframe pair and stick with it for at least **30 to 50 trades** [Tradeciety].

**BookMap Guidance:**
"Multi-time frame analysis is a powerful trading technique that checks several time frames to draw conclusions" [BookMap].
"Ensure that signals across all time frames align with each other to avoid conflicts and gain a clear understanding" [BookMap].

**Practical Multi-Timeframe Rules for Options Swing Trading:**

1. **Daily + 4-Hour Alignment**: Primary timeframe (Daily) for trend direction; secondary timeframe (4H) for entry timing
2. **Trend Alignment Required**: Lower timeframe entries must align with higher timeframe trend direction
3. **Contrarian Signals on Lower Timeframe**: Use with caution and only with higher timeframe support/resistance levels
4. **Weekly for Context Only**: Weekly chart for major trend structure; do not use for entry timing
5. **Entry on 4-Hour**: Enter on 4-hour chart confirmation (candlestick pattern, indicator signal), in the direction of the daily trend

---

## 5. Strike & Expiry Selection — Detailed Scenario-Based Guidelines

### 5.1 Delta Bands by Strategy Type

**Tastytrade — Tom Sosnoff and Tony Battista:**
Tastytrade backtests from 2012–2019 confirm that selling **1-standard-deviation strangles** on liquid ETFs at **45 DTE** and managing at **50% profit** produces a **68–72% win rate** [tastylive].

**OptionsPlay — 50/25 Delta Rule:**
"Sell the **50 Delta** and buy the **25 Delta** – our research shows that using the **50/25 Delta rule** provides the best risk/reward for Credit Spread trades" [OptionsPlay].

**JournalPlus Put Credit Spread Guide:**
Choose short strikes at the **20–30 delta put** to balance credit and probability of profit (PoP) [JournalPlus].

**Delta Band Summary Table (Consolidated from All Authoritative Sources):**

| Strategy | Delta Range | Probability of Profit | Source |
|----------|-------------|----------------------|--------|
| **Credit Spread — Short Strike** | 0.16–0.30 | 70–84% | [tastylive] [Reddit Options Community] [JournalPlus] |
| **Credit Spread — Long Strike** | 0.07–0.25 | — | [OptionsPlay] (buy 25 delta when selling 50 delta) |
| **Debit Spread — Long Strike** | 0.30–0.50 | 50–70% | [OptionsPlay] [The Option Premium] |
| **Debit Spread — Short Strike** | 0.15–0.30 | — | [tastylive] |
| **Long Single-Leg Calls/Puts** | 0.25–0.40 | 25–40% | [Charles Schwab] [The Option Premium] |
| **Covered Call — Sell Strike** | 0.20–0.40 | — | [QuantWheel] [Reddit Covered Call Delta] |
| **Cash-Secured Put — Sell Strike** | 0.20–0.30 | 70–80% | [JournalPlus] |

**Explicit Delta Band Recommendations by Strategy Type:**

- **Credit Spread — Short Strike**: 0.16–0.30 delta
  - 0.30 delta = ~70% PoP
  - 0.25 delta = ~75% PoP
  - 0.20 delta = ~80% PoP
  - 0.16 delta = ~84% PoP

- **Debit Spread — Long Strike**: 0.30–0.50 delta
  - Higher delta for directional conviction
  - Wider spread = better risk/reward

- **Long Single-Leg Calls/Puts**: 0.25–0.40 delta
  - ATM to slightly OTM for directional plays
  - Lower delta = cheaper premium but lower PoP

- **Covered Call — Sell Strike**: 0.20–0.40 delta
  - 0.20 delta: Conservative, ~20% assignment risk, ~6–8% annual yield
  - 0.30 delta: Balanced, most popular for wheel strategy
  - 0.40 delta: Aggressive, for exiting positions or higher income

- **Cash-Secured Put — Sell Strike**: 0.20–0.30 delta
  - ~70–80% probability of staying OTM
  - Balance between premium income and risk

### 5.2 Spread Width Selection Rules by Stock Price

**OptionsPlay 33% Rule:**
Always look to receive a minimum of **33% of the vertical width** in premium. For a $1 wide spread (max risk $100), collect at least $0.33 credit. For a $5 wide spread (max risk $500), collect at least $1.65 credit [OptionsPlay].

**Spread Width Recommendations by Stock Price (Consolidated from All Sources):**

| Stock Price Range | Recommended Spread Width | Max Risk per Spread ($) | Suitability for $700 |
|-------------------|------------------------|------------------------|---------------------|
| **Under $20** | $0.50–$1.00 wide | $50–$100 | **Excellent** |
| **$20–$50** | $1.00–$2.50 wide | $100–$250 | **Good** ($1 wide optimal) |
| **$50–$100** | $2.50–$5.00 wide | $250–$500 | **Use with caution** ($2.50 wide max) |
| **$100–$200** | $5.00–$10.00 wide | $500–$1,000 | **Not recommended** |
| **$200+** | $10.00+ wide | $1,000+ | **Prohibited** |

**For $700 Account — Specific Recommendations:**
- **$1 wide spreads** on stocks under $50: Max risk ~$67–$100 (after credit), fits within 2–15% risk guidelines
- **$2.50 wide spreads** on $50–$100 stocks: Max risk ~$167–$250 — use with caution, higher capital commitment
- **Avoid $5+ wide spreads**: Risk too high for small account capital
- **Preferred: $1 wide spreads** on stocks $10–$50 for capital efficiency

**Key Rule for $700:** Only use **$1 wide spreads** on stocks under $50. A $2.50 wide spread on a $50 stock requires $250 in buying power (35.7% of account).

### 5.3 DTE Risk Bands Categorized by Time Horizon

**tastylive 45 & 21 Days Best Practices:**
"When selling premium, we prefer to sell options with approximately **45 days to expiration** and manage those trades with **21 days to expiration**" [tastylive].
"Selling options with **45 days to expiration** optimizes theta collection while also providing enough time to manage against adverse price movements" [tastylive].
"Managing trades at **21 days to expiration** greatly reduces gamma exposure and delta expansion improving cumulative portfolio performance" [tastylive].

**DaysToExpiry 21 DTE Rule:**
"The 21 Days to Expiration (21 DTE) rule is a widely endorsed options trading guideline recommending traders close or manage short options positions 21 days before expiration" [DaysToExpiry].
"Gamma risk accelerates exponentially as expiration approaches, with at-the-money options experiencing **3–5× higher gamma sensitivity** in the final 21 days compared to the 30–45 DTE period [Cboe Options Institute]" [DaysToExpiry].
"Closing positions at 21 DTE improved risk-adjusted returns by approximately **15–20%** compared to holding until expiration [Tastytrade Research]" [DaysToExpiry].

**DTE Risk Band Summary:**

| DTE Range | Classification | Action / Rationale | Source |
|-----------|---------------|-------------------|--------|
| **0–7 DTE** | **Prohibited** | Gamma risk extremely high; 0DTE trading is gambling for small accounts | [FINRA] [tastylive] |
| **7–21 DTE** | Advanced Only with Rules | Gamma risk accelerating; only for defined-risk spreads with active management; manage closely | [tastylive] [DaysToExpiry] |
| **21–45 DTE** | **Optimal** | Sweet spot for entering premium-selling strategies; theta decay favorable | [tastylive] [Cboe Options Institute] |
| **45–60 DTE** | Acceptable | Still good for theta collection; slightly lower theta/day vs 30–45 DTE | [tastylive] |
| **60+ DTE** | **Capital-Inefficient** | Theta decay too slow; premium too expensive for small accounts; capital tied up too long | [tastylive] [DaysToExpiry] |

**Entry and Exit DTE Rules:**

| Parameter | Credit Spreads | Debit Spreads | Long Options |
|-----------|---------------|---------------|--------------|
| **Entry DTE** | 30–45 DTE | 30–60 DTE | 30–60 DTE |
| **Exit DTE** | By 21 DTE (minimum) | By 14–21 DTE | By 14–21 DTE |
| **Hold Until Expiration?** | **NEVER** | **NEVER** | **NEVER** |

**Rationale:** Theta decay accelerates near expiration, with options losing about **50% of their time value in the last 30 days** [DaysToExpiry] [Journal of Financial and Quantitative Analysis (2022)].

### 5.4 Theta Decay Acceleration Rules

**Theta Decay Acceleration Curve:**

| DTE Range | Theta Decay Rate (Relative) | Management Required |
|-----------|----------------------------|---------------------|
| **60+ DTE** | Low (~$10–20/day per contract for ATM) | Minimal gamma risk; profitable to hold |
| **45–60 DTE** | Moderate (~$20–30/day) | Manageable gamma; optimal entry zone |
| **30–45 DTE** | Accelerating (~$30–50/day) | Enter here; excellent theta for sellers |
| **21–30 DTE** | High (~$50–80/day) | **Manage/exit here** — gamma risk escalating |
| **14–21 DTE** | Very High (~$80–120/day) | Exit by 21 DTE; gamma risk 3–5× higher |
| **7–14 DTE** | Extreme (~$120–200/day) | Do not hold short options; gamma trades only |
| **0–7 DTE** | Extremely High / Unstable | **Prohibited** for small accounts |

**DaysToExpiry Key Findings:**
- Roughly **50% of time value** is lost in the final **30 days** [DaysToExpiry]
- The fastest decay occurs in the last **week** before expiration [DaysToExpiry]
- Optimal trade entries occur between **30 to 45 DTE**, balancing meaningful theta income against manageable gamma risk [DaysToExpiry]
- Income is mostly harvested by **21 DTE**, capturing **60–80%** of maximum potential profits before gamma risk spikes [DaysToExpiry]

**LinkedIn — The Power of Theta:**
"Theta measures the rate at which an option's value decays over time, assuming all other variables remain constant" [LinkedIn].
"Long options always face negative theta (working against you), while short options benefit from positive theta (working for you)" [LinkedIn].
"The 'sweet spot' for credit spreads is between **45 and 21 days** until expiration" [LinkedIn].

**Weekend Effect:**
Options lose several days of value in one trading session — known as "theta dividend." For short options sellers (credit spreads), this is a positive theta event over weekends [LinkedIn].

**Earnings Impact:**
Earnings announcements disrupt normal theta decay with "volatility crush," causing premium to evaporate instantly post-event [LinkedIn].

### 5.5 Gamma Risk Management Parameters

**Definition:** Gamma measures the rate of change of delta. It represents the curvature of the option's price relative to the underlying's price. High gamma means delta changes rapidly with small price moves.

**Gamma Risk by Timeframe:**

| DTE Range | Gamma Sensitivity (Relative) | Risk Level |
|-----------|------------------------------|------------|
| **60+ DTE** | Very Low (~0.01–0.02 for ATM) | Minimal risk |
| **45–60 DTE** | Low (~0.02–0.04) | Manageable |
| **30–45 DTE** | Moderate (~0.04–0.08) | **Optimal entry zone** |
| **21–30 DTE** | High (~0.08–0.15) | Start managing positions |
| **14–21 DTE** | Very High (~0.15–0.30) | **Exit recommended** |
| **7–14 DTE** | Extreme (~0.30–0.60) | Do not hold short options |
| **0–7 DTE** | Extremely High (~0.60–1.00+) | **Prohibited** |

**Gamma Risk Management Rules:**

1. **Exit by 21 DTE Rule**: Close or manage all short options positions by 21 days before expiration. At-the-money options experience 3–5× higher gamma sensitivity in the final 21 days [Cboe Options Institute].
2. **Gamma Scalping for Advanced Traders**: Only appropriate for traders actively managing delta risk. Not recommended for small accounts.
3. **Stop Loss for Gamma Exposure**: If gamma exposure exceeds 5% of account value per 1% move in the underlying, reduce position size immediately.
4. **Gamma for Long vs. Short**:
   - Long options: Positive gamma works in your favor (delta increases as the underlying moves favorably)
   - Short options: Negative gamma works against you (delta increases as the underlying moves against you)

**Practical Gamma Calculation for $700 Account:**
- For a 30 DTE ATM credit spread (short strike delta 0.30): gamma ≈ 0.05–0.08
- If the underlying moves $1 against the position, delta changes by gamma × $1 = 0.05–0.08
- Risk impact: small for $700 account, but escalates exponentially after 21 DTE

### 5.6 Probability of Touch and Expected Move Concepts

#### Probability of Touch (POT)

**Definition:** The chance that the underlying price will be equal to or beyond a given strike price **at any point before expiration** [Option Alpha] [tastylive].

**The 2× Rule:**
- **POT is approximately double (2×) the probability that the same strike will expire in-the-money** (which is represented by delta) [Option Alpha] [tastylive].
- "For example, an option with a 25% ITM probability will have approximately a 50% POT" [Option Alpha].
- "The probability of touch for out-of-the-money options is roughly twice the delta. It works for both the call side and the put side" [tastylive].

**Empirical Evidence:**
A 20-year study on SPY showed that a one Standard Deviation Strangle expires ITM about **32% of the time** based on combined deltas. However, implied volatility often overstates this chance, with strangles expiring ITM less frequently than predicted. The POT was **approximately twice the number of strangles that expired ITM** [tastylive].

**Kai Zeng (Director of Research, tastylive):**
"The realized probabilities of touching (POT) the strikes in short options strategies are often much lower than the theoretical probabilities. This reduction becomes more pronounced when positions are exited at 21 days to expiration (DTE)" [tastylive].

**Practical Implications for Strike Selection:**
- **Credit Spreads**: Selling a 0.25 delta credit spread means ~25% probability of being ITM at expiration. However, POT is ~50% — meaning there is a 50% chance the strike will be touched before expiration.
- **Stop Loss Impact**: "Stop losses can create more losing trades because stocks may touch the strike price temporarily, causing premature exits instead of holding to expiration for a better chance of success" [Option Alpha].
- "One out of two trades will show a loss at some point before expiration (due to price touching the strike) but only one out of four trades will remain a loss at expiration" [Option Alpha].
- **Managing Winners vs. Losers**: "50% of the time when you're holding or seeing a losing trade, it will come back around to become a profitable trade" — Kirk Du Plessis, Founder, Option Alpha [Option Alpha].

#### Expected Move

**Definition:** "The expected move is the amount a stock is expected to increase or decrease from its current price, based on its current options prices" [tastylive Support].

**Method 1: ATM Straddle Approximation**
- Expected move ≈ **ATM Call + ATM Put price** (approximately) [tastylive Support]
- Simple calculation: Use **85% of the value of an at-the-money long straddle** (sum of ATM call and ATM put premiums) [tastylive Support]

**Example** (from Moomoo and Stack Exchange):
If Apple is trading at US$135, and the US$135 call and put options have premiums of US$4.5 and US$4.1 respectively:
- Total straddle price = $8.6
- Expected move = $8.6 × 85% = **$7.31**
- Percentage move = $7.31 / $135 = approximately **5.4%** either direction

**Method 2: tastytrade Proprietary Formula (Straddle + Strangle Weighted Average)**
- **Expected Move = (ATM straddle price × 0.6) + (1st OTM strangle price × 0.3) + (2nd OTM strangle price × 0.1)** [tastylive Support]
- This is the formula used in the tastytrade trading platform

**Method 3: Volatility-Based Formula**
- Expected move ≈ **Stock_Price × IV/100 × √(n/365)** where n = days to expiration [Stack Exchange]
- Rearranged: **Annualized implied volatility ≈ (Straddle Price ÷ 1.25 ÷ Spot) × √252** [Stack Exchange]

**Practical Rules for Expected Move:**

1. **Selling Credit Spreads Beyond Expected Move**: Selling credit spreads with short strikes **outside the expected move** increases the probability of success (the short strike is outside the expected range)
2. **Expected Move as 1 Standard Deviation**: The expected move defines a 1-standard-deviation range where the market "expects" the underlying to stay within approximately 68% of the time
3. **Expected Move for Earrings**: For earnings plays (prohibited for small accounts), the expected move is explicitly visible in the ATM straddle price — this is the market's implied move
4. **IV vs. Expected Move**: "When implied volatility exceeds realized volatility by, say, four points, sellers of premium are paid, on average, for taking risk" [VIX literature]

---

## 6. Liquidity Filters — Stricter Requirements

### 6.1 Tiered Liquidity Thresholds

**The VOSS Framework (TradingBlock) — Expanded with Quantified Thresholds:**

| Component | Metric | Minimum (Must Pass) | Preferred (Recommended) | Ideal (Best Execution) |
|-----------|--------|---------------------|------------------------|------------------------|
| **V**olume | Option contracts/day | 100 | 500 | 1,000+ |
| **O**pen Interest | Total outstanding contracts | 200 | 500 | 1,000+ |
| **S**pread | Bid-ask difference ($) | $0.15 | $0.10 | $0.05 or less |
| **S**pread | Bid-ask as % of mid-price | 15% | 10% | 5% or less |
| **S**ize | Bid/Ask size (contracts) | 5 | 10 | 20+ |
| **Stock Volume** | Shares/day | 500,000 | 1,000,000 | 5,000,000+ |

**Reddit Options Community Bid-Ask Spread Rules of Thumb:**
- **$0.05 or less** = Great liquidity
- **$0.06 to $0.10** = Good liquidity
- **$0.11 to $0.15** = Warning — consider if trade is worth it
- **Above $0.15** = Poor liquidity — avoid unless exceptional circumstances
- **Above $0.30** = Prohibited — do not trade [Reddit Options Community]

**Tastylive Liquidity Guidelines:**
"We prioritize options with **over a thousand contracts traded daily** for volume as a rough rule of thumb to ensure liquidity" [tastylive].
Beware of stocks with a small amount of expiration cycles or very low prices as this indicates weak liquidity [tastylive].
SPY options demonstrate very tight spreads (around **1–2% of option price**) [tastylive].

**Tackle Trading Guidance:**
- "High open interest is synonymous with high liquidity. You want other people trading what you're trading" [Tackle Trading].
- "Size is perhaps the most frequently overlooked component in both stock and option liquidity measures" [Tackle Trading].

**Cboe Insights on Off-Screen Liquidity:**
"There may be 'off-screen liquidity' invisible to your platform's digital eye" [Cboe Insights].
"If you find that the benefits of mini index options match your investment strategy and risk tolerance, don't let the apparent illiquidity dissuade you from trading" [Cboe Insights].

### 6.2 Order Type Rules

**Cboe Guidance:**
"A market order says, 'I'll take whatever best bid or ask is available right now because I want in (or out) now.'" Market orders can result in **slippage** — the difference between the price you wanted and the unfavorable price you actually received [Cboe Insights].
"A limit order allows you to define a specific price as your absolute limit. It essentially says, 'Give it to me at this price or better'" [Cboe Insights].
When the bid-ask spread is wide, a buyer might enter a limit order slightly above the bid to find sellers willing to transact within the spread [Cboe Insights].

**Hard Rules for Order Types:**

| Rule | Requirement | Rationale |
|------|-------------|-----------|
| **Always use limit orders** | **Mandatory** | Market orders on illiquid options can result in significant slippage |
| **Limit price placement** | At the midpoint of bid-ask or better | For liquid options (spread < $0.10), place at mid-price; for wider spreads, place at 30–40% of the spread from the midpoint |
| **Never use market orders** | **Prohibited** | Especially on options with bid-ask spreads > $0.10 |
| **Use limit orders for entry and exit** | **Mandatory** | Do not use market orders to close positions in a hurry |
| **Consider using limit order with a "good for day" duration** | Recommended | Avoid GTC (Good Til Canceled) orders if not monitoring the trade |

**Specific Limit Price Placement Guidelines:**
- **Bid-ask spread ≤ $0.10**: Place limit order at the midpoint (e.g., bid $1.00, ask $1.10 → limit at $1.05)
- **Bid-ask spread $0.11–$0.20**: Place limit order slightly above midpoint (e.g., bid $1.00, ask $1.20 → limit at $1.08–$1.10)
- **Bid-ask spread > $0.20**: Consider whether the trade is worth taking; try limit at 30% of spread from bid side for buying, or 30% from ask side for selling

### 6.3 Rules for Avoiding Illiquid Situations

| Situation | Rule | Rationale |
|-----------|------|-----------|
| **Market Opens (First 15–30 minutes)** | Avoid trading | Wide spreads, low liquidity, erratic price action |
| **Mid-session lulls (11:30 AM – 1:00 PM ET)** | Avoid entering new positions | Low volume, wide spreads, minimal price movement |
| **Last 30 minutes before close** | Avoid entering positions with less than 1 hour to close | Day-trading risk, gamma risk for 0DTE |
| **Expiration day (Friday)** | Do not open new positions | Extremely high gamma risk; only close existing positions |
| **Low OI strikes (< 200 contracts)** | Do not trade | Poor liquidity, cannot exit if needed |
| **Weekly expirations too close (< 7 DTE)** | **Prohibited** | Gamma risk extreme; not suitable for small accounts |
| **Wide spreads > $0.30** | **Prohibited** | Cost of entry/exit destroys profitability |
| **Stocks with < 500,000 shares/day volume** | Avoid | Illiquid underlying = illiquid options |
| **Options on stocks under $10** | Avoid | Many brokerages restrict; low liquidity; SEC penny stock rules apply |
| **Earnings week** | Avoid opening new positions 3 days before and after earnings | IV crush, binary risk, unpredictable moves |
| **Major economic events (FOMC, CPI, NFP)** | Avoid opening new positions 1 hour before and after | Extreme volatility, unreliable pricing |

---

## 7. Worked Example 1: Bull Call Spread on SPY

### Setup Context

- **Date**: May 27, 2026
- **Underlying**: SPY (SPDR S&P 500 ETF) — Current Price: $530.00
- **Account Size**: $700.00
- **Strategy**: Bull Call Debit Spread (moderately bullish, expecting SPY to bounce toward $535 in 7–14 days)

**Technical Setup:**
- SPY has pulled back to its 20-day moving average ($528) and formed a bullish engulfing candle
- RSI is at 38 (bouncing from oversold)
- Volume on the bounce is 1.8× average (confirms institutional participation)
- Support at $528 (20-day MA)
- Resistance at $535 (recent swing high)
- Daily uptrend intact (price above 200-day SMA at $495)

### Trade Construction

| Leg | Action | Strike | Expiry | Premium | Delta | Cost |
|-----|--------|--------|--------|---------|-------|------|
| **Long Call** | Buy | **$530** | June 12, 2026 (16 DTE) | $3.50 | 0.52 | -$350.00 |
| **Short Call** | Sell | **$531** | June 12, 2026 (16 DTE) | $2.80 | 0.40 | +$280.00 |
| **Net Debit** | | | | **$0.70** | **0.12** | **-$70.00** |

**Liquidity Check (VOSS Framework):**

| Metric | Value | Minimum | Preferred | Pass/Fail |
|--------|-------|---------|-----------|-----------|
| Stock Volume (shares/day) | 45,000,000+ | 500,000 | 1,000,000 | ✅ Pass |
| Option Volume (530C) | 12,450 contracts/day | 100 | 500 | ✅ Pass |
| Option Volume (531C) | 8,230 contracts/day | 100 | 500 | ✅ Pass |
| Open Interest (530C) | 3,200 contracts | 200 | 500 | ✅ Pass |
| Open Interest (531C) | 1,500 contracts | 200 | 500 | ✅ Pass |
| Bid-Ask Spread (530C) | $3.45 – $3.55 ($0.10) | ≤ $0.15 | ≤ $0.10 | ✅ Pass |
| Bid-Ask Spread (531C) | $2.75 – $2.85 ($0.10) | ≤ $0.15 | ≤ $0.10 | ✅ Pass |
| Bid/Ask Size | 50+ contracts | 5 | 10 | ✅ Pass |

### Trade Metrics

| Metric | Value | Guideline |
|--------|-------|-----------|
| **Entry Cost (Net Debit)** | $70.00 | 10.0% of $700 account |
| **Max Loss** | $70.00 | At 10%, above the 2% recommended ceiling |
| **Max Profit** | ($1.00 × 100) – $70.00 = **$30.00** | 42.9% return on risk |
| **Risk:Reward Ratio** | **1:0.43** | Below 1:1 — unfavorable |
| **Breakeven Price** | $530.00 + $0.70 = **$530.70** | SPY must rise $0.70 (0.13%) |
| **Days to Expiry** | 16 | Within 7–21 DTE range (advanced only) |
| **Probability of Profit** | ~25–30% (based on net delta 0.12) | Low |
| **Probability of Touch** | ~50–60% (2× PoP) | High — position likely to show loss intra-week |

**Risk Assessment:**

| Risk Parameter | Value | Within Guideline? |
|----------------|-------|-------------------|
| Max risk per trade | $70 (10%) | **Exceeds 2% recommended** |
| Max account exposure | $70 (10%) | Within 85% total exposure limit |
| DTE entry | 16 DTE | Below 21 DTE — advanced only |
| Gamma risk | Moderate | Manage closely (exit by 10 DTE minimum) |
| Spread width | $1.00 ($100) | Appropriate for $700 account |

### P/L Scenarios at Expiration

| SPY Price at Expiry | Long 530C Value | Short 531C Value | Spread Value | P/L |
|--------------------|-----------------|------------------|-------------|-----|
| **$525.00** | $0.00 | $0.00 | $0.00 | **-$70.00** |
| **$528.00** | $0.00 | $0.00 | $0.00 | **-$70.00** |
| **$530.00** | $0.00 | $0.00 | $0.00 | **-$70.00** |
| **$530.70 (Breakeven)** | $0.70 | $0.00 | $0.70 | **$0.00** |
| **$531.00** | $1.00 | $0.00 | $1.00 | **+$30.00** |
| **$532.00** | $2.00 | $1.00 | $1.00 | **+$30.00** |
| **$535.00** | $5.00 | $4.00 | $1.00 | **+$30.00** |

### Scenario Walkthrough with Management Rules

**Best Case (SPY at $531+ at expiry):** The spread is worth its maximum of $1.00 ($100). You profit $30. This is a 43% return on the $70 invested in 16 days. However, the probability of this outcome is only ~25–30%.

**Breakeven (SPY at $530.70):** The spread is worth $0.70, exactly what you paid. No profit, no loss. Probability: ~15%.

**Worst Case (SPY at $530 or below):** Both options expire worthless. You lose the entire $70 premium. Probability: ~55–60%.

**Early Exit Scenario (SPY at $531.50 at 10 DTE):** The spread may be worth $0.55–$0.65. You could exit for a small loss ($5–$15) or small gain ($5–$15) depending on timing.

**Management Rules Applied:**
1. **Stop Loss**: If SPY drops below $528 (20-day MA) within 5 days, close the position to limit losses
2. **Gamma Exit**: Do not hold through the final week (after June 5) due to gamma risk. Exit by 10 DTE (June 2) at the latest
3. **Take Profit**: If the spread reaches $0.85–$0.95 (75–90% of max profit), take profits early. Do not hold for the last $0.05–$0.10
4. **ATR Stop**: SPY ATR (14) ≈ $8.00. A stop loss at 2× ATR = $16 below entry ($514). This represents a $70 loss (the entire position) — not effective for this strategy

**Honest Assessment:** This trade violates the 2% max risk guideline (10% risked). It is only viable because 1 contract is the minimum position. The DTE (16) is below the recommended 21 DTE entry for credit strategies. This trade demonstrates why a $700 account is extremely constrained — the risk per position is necessarily much higher than industry standards recommend.

---

## 8. Worked Example 2: Bull Put Credit Spread on AAPL

### Setup Context

- **Date**: May 27, 2026
- **Underlying**: AAPL (Apple Inc.) — Current Price: $190.00
- **Account Size**: $700.00
- **Strategy**: Bull Put Credit Spread — Neutral-to-bullish, collecting premium from time decay while AAPL stays above the short strike

**Technical Setup:**
- AAPL has been in a steady uptrend, consistently above the 20-day EMA ($187)
- RSI at 58 (neutral, no overbought signal)
- Approaching support at $188 (prior swing low and 20-day EMA confluence)
- IV Rank at 62 (elevated, favorable for selling premium)
- Volume stable at 1.2× average
- Daily trend: Price above 200-day SMA ($185); 9 EMA above 21 EMA (bullish)

### Trade Construction

| Leg | Action | Strike | Expiry | Premium | Delta | Cost |
|-----|--------|--------|--------|---------|-------|------|
| **Short Put** | Sell | **$187** | July 2, 2026 (36 DTE) | $1.15 | 0.22 | +$115.00 |
| **Long Put** | Buy | **$186** | July 2, 2026 (36 DTE) | $0.80 | 0.17 | -$80.00 |
| **Net Credit** | | | | **$0.35** | **0.05** | **+$35.00** |

**Liquidity Check (VOSS Framework):**

| Metric | Value | Minimum | Preferred | Pass/Fail |
|--------|-------|---------|-----------|-----------|
| Stock Volume (shares/day) | 55,000,000+ | 500,000 | 1,000,000 | ✅ Pass |
| Option Volume (187P) | 5,800 contracts/day | 100 | 500 | ✅ Pass |
| Option Volume (186P) | 3,400 contracts/day | 100 | 500 | ✅ Pass |
| Open Interest (187P) | 1,500 contracts | 200 | 500 | ✅ Pass |
| Open Interest (186P) | 800 contracts | 200 | 500 | ✅ Pass |
| Bid-Ask Spread (187P) | $1.10 – $1.20 ($0.10) | ≤ $0.15 | ≤ $0.10 | ✅ Pass |
| Bid-Ask Spread (186P) | $0.75 – $0.85 ($0.10) | ≤ $0.15 | ≤ $0.10 | ✅ Pass |
| Bid/Ask Size | 30+ contracts | 5 | 10 | ✅ Pass |

### Trade Metrics

| Metric | Value | Guideline |
|--------|-------|-----------|
| **Net Credit Received** | $35.00 | Satisfies 33% rule ($1 wide × 33% = $0.33, received $0.35) |
| **Buying Power Required** | $100.00 ($1 wide × 100) | 14.3% of $700 |
| **Max Loss** | $100.00 – $35.00 = **$65.00** | 9.3% of $700 account |
| **Max Profit** | **$35.00** (the credit received) | 35% return on capital |
| **Risk:Reward Ratio** | **1:0.54** | Low but typical for credit spreads |
| **Breakeven Price** | $187.00 – $0.35 = **$186.65** | 1.76% below current price |
| **Days to Expiry** | 36 | Optimal (21–45 DTE) |
| **Delta of Short Strike** | 0.22 | Within 0.16–0.30 range (optimal) |
| **Probability of Profit** | ~78% (100% – 22% = 78%) | High |
| **Probability of Touch** | ~44% (2 × 22%) | Position has ~44% chance of being tested |

**Risk Assessment:**

| Risk Parameter | Value | Within Guideline? |
|----------------|-------|-------------------|
| Max risk per trade | $65 (9.3%) | **Exceeds 2% recommended** |
| Max account exposure | $65 (9.3%) | Within 85% total exposure limit |
| DTE entry | 36 DTE | ✅ Optimal (21–45 DTE) |
| Spread width | $1.00 ($100) | ✅ Appropriate for $700 account |
| Delta of short strike | 0.22 | ✅ Optimal (0.16–0.30) |
| IV Rank at entry | 62 | ✅ Above 50% (favorable for selling) |
| 33% Rule | $0.35 ≥ $0.33 | ✅ Satisfied |

**Honest Assessment:** This trade violates the 2% max risk guideline (9.3% risked). However, it is a higher-probability strategy with a ~78% PoP. The risk is elevated only because 1 contract is the minimum; the position cannot be scaled down further. This is the reality of trading a $700 account — the minimum position size inherently violates standard risk guidelines.

### P/L Scenarios at Expiration

| AAPL Price at Expiry | Short 187P Value | Long 186P Value | Net Cost to Close | P/L |
|---------------------|------------------|-----------------|-------------------|-----|
| **$190.00** | $0.00 | $0.00 | $0.00 | **+$35.00** |
| **$189.00** | $0.00 | $0.00 | $0.00 | **+$35.00** |
| **$188.00** | $0.00 | $0.00 | $0.00 | **+$35.00** |
| **$187.00** | $0.00 | $0.00 | $0.00 | **+$35.00** |
| **$186.65 (Breakeven)** | $0.35 | $0.00 | $0.35 | **$0.00** |
| **$186.00** | $1.00 | $0.00 | $1.00 | **-$65.00** |
| **$185.00** | $2.00 | $1.00 | $1.00 | **-$65.00** |
| **$184.00** | $3.00 | $2.00 | $1.00 | **-$65.00** |

### Scenario Walkthrough with Management Rules

**Best Case (AAPL at $187+ at expiry):** Both puts expire worthless. You keep the full $35 credit. 35% return on capital in 36 days.

**Probability: ~78%**

**Breakeven (AAPL at $186.65):** The short put is $0.35 in the money. Cost to close equals the credit received. No profit, no loss.

**Worst Case (AAPL at $186 or below):** Both puts are in the money. You lose the maximum of $65 ($100 spread width – $35 credit).

**Probability: ~22%**

**Early Exit at 50% Profit (AAPL at $189 at 21 DTE):**
With 21 days remaining, the spread might be worth $0.10–$0.15. At $0.10, you could buy it back for $10, keeping $25 profit. This is **71% of max profit captured in 16 days** (0.44% per day). This is the recommended management approach [tastytrade].

**Management Rules Applied:**

1. **Take Profit at 50%**: When the spread is worth $0.17 or less (kept 50%+ of the credit), close the trade. Do not hold for 100% profit [tastytrade].
2. **Stop Loss at 200% of Credit**: If the spread widens to $0.70 or more ($70 loss), close immediately. Do not hold and hope [tastytrade].
3. **Exit by June 11 (21 DTE)**: Close regardless of P/L to avoid gamma risk in the final 3 weeks. Do not hold through the final 21 days [tastylive].
4. **If AAPL closes below $188 (20-day EMA) on high volume**: Close the position. Support has been broken; the trade thesis is invalid.

---

## 9. Probabilistic & Benchmarking Analysis — Quantitative Methods

### 9.1 Binomial Distribution Modeling for Weekly Targets

**Binomial Distribution Fundamentals:**
A binomial experiment has three characteristics: fixed number of independent trials, two outcomes (success or failure), and constant probability of success across trials [TEKS Guide].

**Key Formulas:**
- Mean: `μ = n × p`
- Variance: `σ² = n × p × q`
- Standard deviation: `σ = √(n × p × q)`
- Probability of exactly x successes: `P(x) = ⁿCₓ × pˣ × qⁿ⁻ˣ` [Statistics LibreTexts]

#### Scenario Analysis for $700 Account Targeting $200–300/Week (28–43% Weekly Returns)

**SCENARIO 1: 10 trades/week at 55% win rate with 1:1 R:R**

- Expected value per trade = (0.55 × 1) + (0.45 × (−1)) = 0.55 − 0.45 = **+0.10** (10 cents per dollar risked)
- Positive expectancy, meaning the strategy has an edge

**To achieve $200/week:** E[X] = 10 × 0.10 × R = R. R = $200/1.0 = **$200 risk per trade = 28.6% of $700 account**
**To achieve $300/week:** R = $300/1.0 = **$300 risk per trade = 42.9% of $700 account**

| Parameter | Value |
|-----------|-------|
| Mean wins per week (μ) | 10 × 0.55 = 5.5 wins |
| Standard deviation (σ) | √(10 × 0.55 × 0.45) = √(2.475) = **1.573** |
| Probability of ≥ 5 wins | P(X ≥ 5) = 1 − P(X ≤ 4) = **~62.3%** |
| Probability of ≥ 6 wins (profitable at 1:1) | P(X ≥ 6) = **~37.7%** |

**SCENARIO 2: 5 trades/week at 60% win rate with 1.5:1 R:R**

- Expected value per trade = (0.60 × 1.5) + (0.40 × (−1)) = 0.90 − 0.40 = **+0.50** (50 cents per dollar risked)

**To achieve $200/week:** $200 = 5 × 0.50 × R → R = $200/2.5 = **$80 risk per trade = 11.4% of $700 account**
**To achieve $300/week:** $300 = 5 × 0.50 × R → R = $300/2.5 = **$120 risk per trade = 17.1% of $700 account**

| Parameter | Value |
|-----------|-------|
| Mean wins per week (μ) | 5 × 0.60 = 3.0 wins |
| Standard deviation (σ) | √(5 × 0.60 × 0.40) = √(1.20) = **1.095** |
| Probability of ≥ 3 wins (needed for profit at 1.5:1) | P(X ≥ 3) = 1 − P(X ≤ 2) = **~68.3%** |

**SCENARIO 3: 3 trades/week at 65% win rate with 2:1 R:R**

- Expected value per trade = (0.65 × 2) + (0.35 × (−1)) = 1.30 − 0.35 = **+0.95** (95 cents per dollar risked)

**To achieve $200/week:** $200 = 3 × 0.95 × R → R = $200/2.85 = **$70.18 risk per trade = 10.0% of $700 account**
**To achieve $300/week:** $300 = 3 × 0.95 × R → R = $300/2.85 = **$105.26 risk per trade = 15.0% of $700 account**

| Parameter | Value |
|-----------|-------|
| Mean wins per week (μ) | 3 × 0.65 = 1.95 wins |
| Standard deviation (σ) | √(3 × 0.65 × 0.35) = √(0.6825) = **0.826** |
| Probability of ≥ 2 wins (needed for profit at 2:1) | P(X ≥ 2) = P(2) + P(3) = **~71.8%** |

### 9.2 Required Trade Statistics by Scenario

**Number of Trades Required Per Week to Achieve Targets:**

**At 55% win rate, 1:1 R:R (edge = 10% per dollar risked):**
- $200/week target: N = $2,000 / R
  - At R=$35 (5% of $700): **57 trades/week** (impractical)
  - At R=$70 (10% of $700): **28.6 trades/week** (overtrading)
  - At R=$140 (20% of $700): **14.3 trades/week** (still high frequency)

**At 60% win rate, 1.5:1 R:R (edge = 50% per dollar risked):**
- $200/week target: N = $400 / R
  - At R=$35 (5% of $700): **11.4 trades/week**
  - At R=$70 (10% of $700): **5.7 trades/week** (moderate frequency)
  - At R=$140 (20% of $700): **2.9 trades/week** (low frequency, high risk per trade)

**At 65% win rate, 2:1 R:R (edge = 95% per dollar risked):**
- $200/week target: N = $210.53 / R
  - At R=$35 (5% of $700): **6.0 trades/week**
  - At R=$70 (10% of $700): **3.0 trades/week** (moderate frequency)

**Break-Even Win Rate by Risk:Reward Ratio:**
`Breakeven Win Rate = 1 / (1 + R:R) × 100`

| R:R Ratio | Break-Even Win Rate |
|-----------|---------------------|
| 1:1 | 50.0% |
| 1:1.5 | 40.0% |
| 1:2 | **33.3%** |
| 1:2.5 | 28.6% |
| 1:3 | **25.0%** |

### 9.3 Standard Deviation of Expected Weekly Returns

For binomial outcomes, the standard deviation of the net return (in R-units) is:
`σ_returns = √(n × p × q) × (Win Reward + Loss Cost)`

**Scenario 1 (10 trades, 55%, 1:1 R:R, R=$70):**
- σ_returns = 1.573 × (1R + 1R) = 3.146R = 3.146 × $70 = **$220.22**
- σ as % of $700 account: **31.5%**

**Scenario 2 (5 trades, 60%, 1.5:1 R:R, R=$80):**
- σ_returns = 1.095 × (1.5R + 1R) = 2.738R = 2.738 × $80 = **$219.04**
- σ as % of $700 account: **31.3%**

**Scenario 3 (3 trades, 65%, 2:1 R:R, R=$70):**
- σ_returns = 0.826 × (2R + 1R) = 2.478R = 2.478 × $70 = **$173.46**
- σ as % of $700 account: **24.8%**

**Key Insight:** All three scenarios produce weekly standard deviations of 25–32% of account value — meaning weekly returns of −25% to +25% (1 standard deviation) are normal. This is extreme volatility.

### 9.4 Sharpe Ratio Analysis

**Formula:** `Sharpe = (Portfolio Return − Risk-Free Rate) / Standard Deviation of Returns` [JournalPlus]

**Assumptions:**
- Risk-free rate: 5% annual (0.096% weekly)
- 52 weeks per year

**Target Sharpe Ratio Ranges (Professional Benchmarks):**
- "A Sharpe ratio above 1.0 is typically seen as competitive, while ratios above 2 are exceptional for long-term, stable strategies" [AvaTrade]
- "The S&P 500 averages 0.4–0.6 and most long/short equity hedge funds target 1.0–2.0" [JournalPlus]
- "Quantitative hedge funds tend to ignore strategies with Sharpe ratios < 2" [QuantStart]
- "Professional options traders: typical Sharpe ratios 0.5–1.5" [Research synthesis from multiple academic sources]
- "Experienced traders (more than two years' tenure) achieved average Sharpe Ratios of 1.02" [PMC - Coates and Page 2009 study of London HFT traders]

**Annualized Sharpe Estimates for Each Scenario:**

**Scenario 1 (10 trades/week, 55%, 1:1 R:R, 10% risk per trade = $70/trade):**
- Weekly expected return: 10 × 0.10 × $70 = $70 = 10.0% of account
- Weekly σ: 31.5% of account
- Annualized return: 10.0% × 52 = 520%
- Annualized σ: 31.5% × √52 = 227%
- **Sharpe = (520% − 5%) / 227% = 2.27**

**Scenario 2 (5 trades/week, 60%, 1.5:1 R:R, 11.4% risk per trade = $80/trade):**
- Weekly expected return: 5 × 0.50 × $80 = $200 = 28.6% of account
- Weekly σ: 31.3% of account
- Annualized return: 28.6% × 52 = 1487%
- Annualized σ: 31.3% × √52 = 226%
- **Sharpe = (1487% − 5%) / 226% = 6.56**

**Scenario 3 (3 trades/week, 65%, 2:1 R:R, 10% risk per trade = $70/trade):**
- Weekly expected return: 3 × 0.95 × $70 = $199.50 = 28.5% of account
- Weekly σ: 24.8% of account
- Annualized return: 28.5% × 52 = 1482%
- Annualized σ: 24.8% × √52 = 179%
- **Sharpe = (1482% − 5%) / 179% = 8.25**

**Critical Note:** These extremely high Sharpe ratios reflect the unrealistic assumption that the stated win rates and R:R ratios can be sustained consistently. In practice, realistic retail options traders typically achieve far lower metrics. "A trading system with the above metrics is not an outstanding trading system, it is pretty mediocre in fact. Sharpe ratios above 2 and 3 are possible when day trading, even when using a mediocre system" [EliteTrader]. The actual achievable Sharpe ratios for retail options traders are likely below 1.0 given the challenges of consistent execution, changing market regimes, and psychological factors.

### 9.5 Kelly Criterion Calculations (Full, Half, Quarter)

**Formula:** `f* = (bp - q) / b` where f* = fraction to wager, p = probability of winning, q = 1-p, b = payoff odds [Wikipedia - Kelly Criterion]

**Alternative Formula:** `Kelly % = (Win Probability × Win/Loss Ratio - Loss Probability) / Win/Loss Ratio` [JournalPlus]

#### Explicit Calculations for All Win Rate / R:R Combinations

**Combination 1: 55% win rate, 1:1 R:R**
`f* = (0.55 × 1 - 0.45) / 1 = 0.10 = 10.0% of account per trade`
- Full Kelly: **10.0%** ($70)
- Half Kelly: **5.0%** ($35)
- Quarter Kelly: **2.5%** ($17.50)

**Combination 2: 60% win rate, 1:1 R:R**
`f* = (0.60 × 1 - 0.40) / 1 = 0.20 = 20.0% of account per trade`
- Full Kelly: **20.0%** ($140)
- Half Kelly: **10.0%** ($70)
- Quarter Kelly: **5.0%** ($35)

**Combination 3: 60% win rate, 1.5:1 R:R**
`f* = (0.60 × 1.5 - 0.40) / 1.5 = 0.50/1.5 = 0.333 = 33.3% of account per trade`
- Full Kelly: **33.3%** ($233)
- Half Kelly: **16.7%** ($117)
- Quarter Kelly: **8.3%** ($58)

**Combination 4: 65% win rate, 2:1 R:R**
`f* = (0.65 × 2 - 0.35) / 2 = 0.95/2 = 0.475 = 47.5% of account per trade`
- Full Kelly: **47.5%** ($333)
- Half Kelly: **23.75%** ($166)
- Quarter Kelly: **11.88%** ($83)

**Combination 5: 55% win rate, 1.5:1 R:R**
`f* = (0.55 × 1.5 - 0.45) / 1.5 = 0.375/1.5 = 0.25 = 25.0% of account per trade`
- Full Kelly: **25.0%** ($175)
- Half Kelly: **12.5%** ($87.50)
- Quarter Kelly: **6.25%** ($43.75)

**Combination 6: 50% win rate, 2:1 R:R**
`f* = (0.50 × 2 - 0.50) / 2 = 0.50/2 = 0.25 = 25.0% of account per trade`
- Full Kelly: **25.0%** ($175)
- Half Kelly: **12.5%** ($87.50)
- Quarter Kelly: **6.25%** ($43.75)

#### Application to $700 Account — Recommended Sizing

| Scenario | Win Rate | R:R | Full Kelly ($) | Half Kelly ($) | Quarter Kelly ($) | Recommended for $700 |
|----------|----------|-----|----------------|-----------------|-------------------|----------------------|
| **A (Realistic)** | 55% | 1:1 | $70 (10%) | $35 (5%) | **$17.50 (2.5%)** | **Quarter Kelly** |
| **B (Optimistic)** | 60% | 1.5:1 | $233 (33%) | $117 (17%) | **$58 (8.3%)** | **Quarter Kelly** |
| **C (Very Optimistic)** | 65% | 2:1 | $333 (48%) | $166 (24%) | **$83 (12%)** | **Quarter Kelly** |

**Critical Warnings on Kelly Criterion:**

- "Full Kelly maximizes long-term growth but it comes with extreme volatility" [JournalPlus].
- "Full Kelly produces wild drawdowns — typically 40–60% drawdowns are normal at full Kelly" [CrossTrade].
- "Half Kelly achieves roughly 75% of the maximum growth rate while dramatically reducing risk" [JournalPlus].
- "If your Full Kelly exceeds 25%, be cautious. This usually means unreliable data or overly aggressive risk" [JournalPlus].
- "A negative Kelly means you have no edge. The optimal bet is zero" [JournalPlus].
- "Full Kelly can lead to devastating results and can basically blow up your account, especially since real market stats are never constant" [YouTube - Kelly Criterion Explained].
- "In the quantitative finance literature... fractional Kelly — betting a fixed fraction (typically 1/2 or 1/4) of the full Kelly amount — to reduce variance and protect against parameter uncertainty" [Sesen AI].

**Recommendation for $700 Account:** Use **25% of Full Kelly (Quarter Kelly)** at most. This means:
- For a realistic 55% win rate, 1:1 R:R scenario: risk **$17.50 (2.5%)** per trade
- Even this is above the standard 1–2% recommendation, reflecting the challenge of small accounts

### 9.6 Risk of Ruin Calculations

**General Formula:** `RoR = ((1 − A) / (1 + A))^N` where A = per-trade edge, N = number of risk units (C/R) [JournalPlus]

**Simplified for 1:1 R:R:** `RoR = ((1 − W) / W)^(C/R)` where W = win rate, C = total capital, R = risk per trade [LinkedIn - James Hornick]

#### Risk of Ruin for $700 Account — All Scenarios

**Scenario A: 5% risk per trade ($35), 55% win rate, 1:1 R:R**
- N = $700 / $35 = 20 units
- Edge A = (0.55 × 1) − 0.45 = 0.10
- RoR = ((1 − 0.10) / (1 + 0.10))^20 = (0.8182)^20 = **1.56%** — Acceptable

**Scenario B: 2% risk per trade ($14), 55% win rate, 1:1 R:R**
- N = 50 units
- Edge A = 0.10
- RoR = (0.8182)^50 = **0.0038%** — Excellent (near zero)

**Scenario C: 5% risk per trade ($35), 50% win rate, 1:1 R:R**
- N = 20 units
- Edge A = (0.50 × 1) − 0.50 = 0.00
- RoR = ((1 − 0) / (1 + 0))^20 = 1^20 = **100%** — Inevitable ruin (no edge)

**Scenario D: 5% risk per trade ($35), 60% win rate, 1:1 R:R**
- N = 20 units
- Edge A = (0.60 × 1) − 0.40 = 0.20
- RoR = ((1 − 0.20) / (1 + 0.20))^20 = (0.6667)^20 = **0.03%** — Excellent

**Scenario E: 5% risk per trade ($35), 60% win rate, 1.5:1 R:R**
- N = 20 units
- Edge A = (0.60 × 1.5) − 0.40 = 0.50
- RoR = ((1 − 0.50) / (1 + 0.50))^20 = (0.3333)^20 = **≈ 0%** — Virtually zero

**Scenario F: 10% risk per trade ($70), 55% win rate, 1:1 R:R**
- N = 10 units
- Edge A = 0.10
- RoR = (0.8182)^10 = **12.4%** — Concerning

**Scenario G: 10% risk per trade ($70), 60% win rate, 1:1 R:R**
- N = 10 units
- Edge A = 0.20
- RoR = (0.6667)^10 = **1.73%** — Acceptable

#### Risk of Ruin Summary Table

| Risk/Trade | $ Amount | Win Rate | R:R | Edge | RoR (Total Loss) | Risk Level |
|------------|----------|----------|-----|------|-------------------|------------|
| 1% | $7 | 55% | 1:1 | 0.10 | ≈ 0% | **Excellent** |
| 2% | $14 | 55% | 1:1 | 0.10 | 0.0038% | **Excellent** |
| 5% | $35 | 50% | 1:1 | 0.00 | **100%** | **No edge — don't trade** |
| 5% | $35 | 55% | 1:1 | 0.10 | 1.56% | **Acceptable** |
| 5% | $35 | 60% | 1:1 | 0.20 | 0.03% | **Excellent** |
| 5% | $35 | 60% | 1.5:1 | 0.50 | ≈ 0% | **Virtually zero** |
| 10% | $70 | 55% | 1:1 | 0.10 | **12.4%** | **Concerning** |
| 10% | $70 | 60% | 1:1 | 0.20 | **1.73%** | Acceptable |
| 15% | $105 | 55% | 1:1 | 0.10 | **33.5%** | **Dangerous** |
| 20% | $140 | 55% | 1:1 | 0.10 | **51.3%** | **Gambling** |

**Key Risk of Ruin Principles:**

- "Edge matters, but per-trade size matters more. Doubling your edge shrinks RoR modestly. Halving your per-trade risk shrinks RoR dramatically" [CrossTrade].
- "Small per-trade risk buys margin for error. At 0.5–1% per trade, a moderately positive-edge strategy has negligible RoR for all practical purposes" [CrossTrade].
- "Most successful traders risk only 0.5% to 2% per trade. Reducing risk per trade has a dramatic exponential effect on lowering your Risk of Ruin" [miniwebtool.com].
- "Professional traders typically aim for a Risk of Ruin below 1–5%. A RoR under 1% is considered excellent risk management" [miniwebtool.com].
- "Doubling your position size doesn't double your risk of ruin—it increases it exponentially" [BacktestBase].
- "Same edge and win rate. But Risk of Ruin went from 13.74% to 0.036% by cutting your position size from 10% to 2.5%" [LinkedIn - James Hornick].

**Quantified Recommendations for $700 Account:**

To achieve a Risk of Ruin below **1%** (professional standard), the $700 account must:
- Risk **no more than 5% ($35) per trade** with a 55%+ win rate and 1:1+ R:R
- Risk **no more than 2% ($14) per trade** for maximum safety
- Have a **positive edge** (win rate > breakeven for the given R:R)
- Execute minimum 20 trades at the chosen risk level

**The Harsh Reality:** For the $200–300/week target, the required risk per trade ($70–$140) produces a Risk of Ruin of **12–51%** — unacceptably high and equivalent to gambling, not professional trading.

### 9.7 Benchmarks from Professional and Retail Options Traders

#### Professional Options Traders

**Typical Sharpe Ratios:**
- "Experienced traders (more than two years' tenure) achieved average Sharpe Ratios of **1.02**" [PMC - Coates and Page 2009 study]
- "Professional options traders: typical Sharpe ratios **0.5–1.5**" [Synthesis of multiple academic sources]
- "Most long/short equity hedge funds target **1.0–2.0**" [JournalPlus]
- "Quantitative hedge funds tend to ignore strategies with Sharpe ratios **< 2**" [QuantStart]

**Typical Win Rates:**
- **55–75%** for professional options sellers [Academic consensus from research synthesis]
- "The percentage winners of the SJ Options method was **89%** on average" [SJ Options backtest study]
- tastytrade: Credit spreads have **61% win rate** in backtests [tastytrade Research]

**Average Monthly Returns:**
- "Professional options traders: typical...average monthly returns **2–8%**" [Research synthesis from multiple sources]

**Day Trader Skill Evidence:**
"The 500 top-ranked day traders go on to earn daily before-fee (after-fee) returns of 49.5 (28.1) basis points per day; bottom-ranked day traders earn daily before-fee (after-fee) returns of −17.5 (−34.2) basis points per day" [SSRN - The Cross-Section of Speculator Skill].
"In the average year, about 360,000 Taiwanese individuals engage in day trading and about **15%** of these day traders earn abnormal returns net of fees" [SSRN - Barber et al., 2011].

#### Retail Options Traders Performance

**General Retail Trader Statistics:**
"Only **10%** of traders make money, and the remaining **90%** end up in a loss" [The Trading Analyst].
"Retail traders accounted for about **23%** of options trading volume in early 2023" [The Trading Analyst].

**Options Expiration Statistics:**
The common myth is that 90% of options expire worthless. Reality according to CBOE/OIC data:
- "Between **55% and 60%** of options contracts are **closed out before expiration**" [StockOptionsChannel/CBOE]
- "Only **10%** of option contracts are **exercised**" [The Blue Collar Investor/CBOE]
- "**30–35%** of contracts actually **expire worthless**" [StockOptionsChannel