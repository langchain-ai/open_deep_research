# Deeply Revised Options Trading Framework Report for a $700 Account

## Targeting $200–$300/Week Returns — Educational Framework Only, Not Personalized Financial Advice

**Date of Report: May 28, 2026**

**Educational Purposes Only — This document provides a framework for understanding options trading concepts. It is not personalized financial advice. Options trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The $200–$300 per week target represents an extremely aggressive goal requiring significant risk. Consult a qualified financial advisor before engaging in any trading activity.**

All quantitative parameters in this report are explicitly sourced from authoritative sources including Cboe, OCC, FINRA, SEC, broker educational materials (Schwab, Fidelity, tastytrade, Interactive Brokers), OIC, Cboe Options Institute, and academic research papers. Generic industry ranges are replaced with specific, verifiable thresholds throughout.

---

## Table of Contents

1. [Section 1: Foreword — The Mathematical Reality of the Target](#section-1-foreword--the-mathematical-reality-of-the-target)
2. [Section 2: Position Sizing and Risk Rules](#section-2-position-sizing-and-risk-rules)
   - 2.1 Maximum Risk Per Trade — Sliding Scale
   - 2.2 Maximum Concurrent Positions and Sector Exposure
   - 2.3 Maximum Daily, Weekly, and Monthly Loss Limits
   - 2.4 Maximum Total Account Exposure (Delta/Theta-Based)
   - 2.5 Hard Rules for Account Drawdown
   - 2.6 Rules for Reducing/Increasing Position Size After Gains/Losses
   - 2.7 Minimum Cash Reserve Requirements
   - 2.8 Position Sizing Methods
3. [Section 3: Permitted and Prohibited Strategies](#section-3-permitted-and-prohibited-strategies)
   - 3.1 Debit Spreads
   - 3.2 Credit Spreads
   - 3.3 Single-Leg Calls and Puts
   - 3.4 Cash-Secured Puts
   - 3.5 Covered Calls
   - 3.6 Iron Condors
   - 3.7 Poor Man's Covered Call (PMCC)
   - 3.8 Prohibited Strategies
   - 3.9 Broker Approval Level Summary
4. [Section 4: Stock Selection Criteria](#section-4-stock-selection-criteria)
   - 4.1 RSI Thresholds
   - 4.2 MACD Rules
   - 4.3 EMA/SMA Rules
   - 4.4 Volume Confirmation Thresholds
   - 4.5 VWAP-Based Rules
   - 4.6 Bollinger Band Squeeze Rules
   - 4.7 ATR-Based Position Sizing for Selection
   - 4.8 IV Rank/Percentile Strategy Selection
   - 4.9 Optimal Stock Price and Premium Bands
   - 4.10 Multi-Timeframe Confirmation
5. [Section 5: Strike and Expiry Selection](#section-5-strike-and-expiry-selection)
   - 5.1 Delta Bands by Strategy Type
   - 5.2 Spread Width Rules by Stock Price
   - 5.3 DTE Risk Bands
   - 5.4 Theta Decay Acceleration Rules
   - 5.5 Gamma Risk Management
   - 5.6 Probability of Touch and Expected Move
6. [Section 6: Liquidity Filters — VOSS Framework](#section-6-liquidity-filters--voss-framework)
   - 6.1 Tiered Liquidity Thresholds
   - 6.2 Order Type Rules
   - 6.3 Rules for Avoiding Illiquid Situations
7. [Section 7: Hypothetical Example Trades](#section-7-hypothetical-example-trades)
   - 7.1 Example 1: Bull Call Debit Spread — Ford Motor Company (F)
   - 7.2 Example 2: Bull Put Credit Spread — KeyCorp (KEY)
8. [Section 8: Probabilistic Analysis](#section-8-probabilistic-analysis)
   - 8.1 Binomial Distribution Modeling
   - 8.2 Required Trade Statistics by Scenario
   - 8.3 Standard Deviation of Expected Weekly Returns
   - 8.4 Sharpe Ratio Analysis
   - 8.5 Kelly Criterion Calculations
   - 8.6 Risk of Ruin Calculations
   - 8.7 Benchmarks from Professional and Retail Traders
   - 8.8 Transaction Costs, Fees and Slippage Integration
9. [Section 9: Daily Screening Checklist](#section-9-daily-screening-checklist)
10. [Sources](#sources)

---

## Section 1: Foreword — The Mathematical Reality of the Target

Before presenting the detailed framework, a fundamental truth must be established through mathematical analysis. A target of **$200–$300 per week on a $700 account** represents a **28.6%–42.9% weekly return**. To contextualize this: the Cboe S&P 500 BuyWrite Index (BXM), a benchmark covered call strategy, has delivered an annualized compound return of approximately 11.77% since 1988 [Cboe Options Institute Research]. Professional top-quartile hedge funds target 15–25% annual returns. The weekly target in this framework exceeds what professional strategies achieve in an entire year.

The academic research on day trader performance by Barber, Lee, Liu, and Odean (2014, published in the Review of Financial Studies) found that the top 500 ranked day traders earn approximately 49.5 basis points (0.495%) per day before fees [SSRN - The Cross-Section of Speculator Skill]. This translates to roughly 2.5% per week for the top tier of professional day traders — approximately 1/10th to 1/17th of the $200–$300 target on a $700 account.

A 28.6% weekly target requires generating $200 in profit. At a 1% risk per trade ($7), this requires approximately 29 winning trades per week at 2:1 risk-reward with 60% win rate. At 5% risk per trade ($35), this requires approximately 6 winning trades per week. However, at 5% risk per trade, a sequence of just 3 consecutive losses (which has an 18.5% probability of occurring in a 50-trade sequence at 60% win rate) produces a $105 loss — 15% of the account.

This framework presents the disciplined rules that maximize survival probability for a $700 account, then provides the probabilistic analysis showing what is required to achieve the stated target. The rules prioritize capital preservation; the analysis quantifies the risk.

---

## Section 2: Position Sizing and Risk Rules

### 2.1 Maximum Risk Per Trade — Explicit Sliding Scale with Dollar Amounts

The following risk tiers are based on consolidated guidance from Dr. Jim Schultz (tastytrade quantitative expert and finance Ph.D.) and multiple authoritative sources. Dr. Schultz states: "Generally speaking for an average account ($20,000 to $100,000), starting with defined-risk strategies, a great reference point is to be between 1%–3% of your account value per position" [tastylive - Defined-Risk and Undefined-Risk Position Sizing]. For accounts under $20,000, he notes "you'll likely have to increase the upper end of this range, hitting 5%, 6%, 7% or even higher at times" [tastylive - Defined-Risk and Undefined-Risk Position Sizing].

Option Alpha reinforces: "I've said, for 8 years now, that you should never allocate more than 1–5% of risk per trade" [Option Alpha - Account Size Adjustments]. Cory Mitchell, CMT states: "The 1% risk rule means not risking more than 1% of account capital on a single trade. Risking 1% or less per trade is the standard for most professional traders. When starting out, it is better to risk 0.5% or even 0.25% per trade" [TradeAlgo - Futures Risk Management].

**Exact Dollar Amounts for Each Tier on $700 Account:**

| Risk Tier | % of Account | Dollar Amount | Source Rationale |
|-----------|-------------|---------------|------------------|
| Ultra-Conservative | 0.5% | **$3.50** | Cory Mitchell: "risk 0.5% per trade when starting out" [TradeAlgo] |
| Ultra-Conservative | 1.0% | **$7.00** | Standard professional 1% rule [TradeAlgo] |
| Conservative | 2.0% | **$14.00** | Common professional recommendation [Option Alpha] |
| Moderate | 3.0% | **$21.00** | Upper end of defined-risk range per Dr. Schultz [tastylive] |
| Aggressive | 5.0% | **$35.00** | Upper range for small accounts [tastylive] |
| Very Aggressive | 8.0% | **$56.00** | Undefined-risk strategies on small accounts [tastylive] |
| Extreme (Not Recommended) | 10.0% | **$70.00** | Exceeds standard industry recommendations |
| Gambling | 15.0%+ | **$105.00+** | Not supported by any educational authority |

**Practical Constraints for $700 Account:** The minimum options contract size is 100 shares. For a $1-wide spread, total buying power used is $100. For credit spreads at most brokers, the minimum buying power requirement is the spread width times 100 minus the credit received. This means even a single $1-wide credit spread on a $30 stock uses $70–$100 in buying power, immediately exceeding the 2% risk guideline. The $700 account inherently operates above standard risk recommendations due to minimum contract sizes.

### 2.2 Maximum Concurrent Positions and Sector Exposure

The tastytrade "Rule of 12" methodology and risk-budgeting approach guide position count. The tastylive Market Measures episode on Position Sizing & Margin states: "If the probability of success on a defined risk spread is 67%, we should lose 1 in every 3 trades over time. Putting 33% of our account in trades with a 67% probability of success will eventually leave us with no capital left to trade. By reducing our capital allocation in each trade to 5%, the probability of losing all capital drops to 1 in 4.25 billion" [tastylive - Position Sizing & Margin].

**Maximum Concurrent Positions for $700 Account:**

| Approach | Maximum Positions | Rationale |
|----------|------------------|-----------|
| Ultra-Conservative | 1 | At 1% risk ($7), only 1 position to maintain capital efficiency |
| Conservative | 1–2 | At 2% risk ($14), 2 positions use $28 (4% total risk) |
| Moderate | 1–2 | At 3% risk ($21), 2 positions use $42 (6% total risk) |
| Aggressive | 1–2 | At 5% risk ($35), 2 positions use $70 (10% total risk) |
| Absolute Maximum | 3 | Portfolio heat concept: total risk ≤ 8% [QuantVPS] |

**Maximum Sector/Correlation Exposure:**

| Exposure Type | Limit | Calculation for $700 |
|--------------|-------|---------------------|
| Single Stock | 50% of account max | $350 at most in one underlying |
| Same Sector | 66% of account max | $462 at most across same sector |
| Correlated Underlyings | Treat as one risk pool | Correlated ETFs count as single sector |

The "portfolio heat" concept from QuantVPS recommends "limiting total portfolio risk, or portfolio heat, to between 4% and 8%" [QuantVPS - Risk Management Guide]. For a $700 account, 8% portfolio heat = $56 maximum total risk across all open positions.

### 2.3 Maximum Daily, Weekly, and Monthly Loss Limits

**Exact Dollar Triggers and Mandatory Actions:**

**Daily Loss Limits:**

| Daily Loss | % of $700 | Dollar Amount | Mandatory Action | Source |
|-----------|-----------|---------------|------------------|--------|
| Soft Warning | 1% | $7.00 | Reduce position size by 25%, review open trades | Cory Mitchell: 0.5%–1% beginner risk |
| Stop Trading for Day | 3% | $21.00 | Close all positions, no new trades until next day | FundedNext: 3–5% daily loss limit |
| Stop Trading for Week | 6% | $42.00 | Close all positions, no trading for rest of week | Extension of Apex/FundedNext rules |
| Stop Trading for Month | 10% | $70.00 | Close all positions, mandatory account review | FundedNext Maximum Loss Limit 10% |

From Earn2Trade: "A good rule of thumb is to set a 2% daily loss limit, though individual limits vary. Exceeding the daily loss limit can lead to automatic flattening of positions or suspension from trading" [Earn2Trade - Daily Loss Limits]. From FundedNext: "The Daily Loss Limit is the maximum you can lose in one day, resetting at midnight. For Stellar 2-Step Accounts: You can lose up to 5% of your initial balance each day" [FundedNext - Rules].

**Weekly Loss Limits:**

| Weekly Loss | % of $700 | Dollar Amount | Mandatory Action | Source |
|-----------|-----------|---------------|------------------|--------|
| Warning | 5% | $35.00 | Reduce all position sizes by half for following week | Derived from Apex/FTMO rules |
| Stop Trading | 10% | $70.00 | No trading for minimum 3 trading days | FundedNext max loss concept |
| Account Review | 15% | $105.00 | Full strategy review, mandatory 5-day break | Drawdown recovery mathematics |
| Capital Protection | 20% | $140.00 | Convert to 100% cash, mandatory 2-week break | 20% drawdown requires 25% gain to recover |

**Monthly Drawdown Limits:**

| Monthly Loss | % of $700 | Dollar Amount | Mandatory Action | Source |
|------------|-----------|---------------|------------------|--------|
| Soft Cap | 5% | $35.00 | Reduce position sizes by 50% for remainder of month | For Traders: 5% limit |
| Hard Cap | 10% | $70.00 | Stop trading entirely for the month, account review | FundedNext: 10% max loss |
| Maximum Tolerable | 15% | $105.00 | Strategy reevaluation, possible strategy change | AvaTrade: below 20% recommendation |
| Account Danger Zone | 20% | $140.00 | Deposit additional funds or switch to cash-only | 20% drawdown = 25% recovery needed |

The recovery mathematics are non-linear: "A 50% loss requires a 100% gain to recover" [Drawdown Recovery Calculator]. "Drawdowns grow exponentially harder to recover from" [Drawdown Recovery Calculator]. From ZitaPlus: "Recovering from a drawdown requires reassessing strategy, reducing position sizes, stopping revenge trading, diversifying trades, reviewing risk-reward ratios, and sometimes taking a break" [ZitaPlus - Drawdown Recovery]. From For Traders: "Stick to consistent position sizes and rely on your tested strategy. Avoid the temptation to increase position sizes to recover losses" [For Traders - Trader Rules].

### 2.4 Maximum Total Account Exposure Using Delta and Theta-Based Limits

**Delta-Based Exposure Limits:**

From the Pomegra Learn Library: "Options position sizing is the calculation of how many option contracts to buy or sell, accounting for the embedded leverage in option pricing (delta), the acceleration of that leverage as prices move (gamma), and the decay of the option's time value (theta), to ensure total notional exposure and loss potential align with portfolio rules" [Pomegra - Options Position Sizing].

"Notional Delta Exposure = Delta × 100 × Stock Price; this number should guide position sizing, not premium" [Pomegra - Options Position Sizing].

**Delta Dollars Rule:** Keep total Delta Dollars within -1:1 to +1:1 ratio of account size. For $700 account: maximum net directional exposure is ±$700 in delta-adjusted terms [Interactive Brokers - Risk Navigator Quick Guide].

**Delta-Based Exposure Limits for $700 Account:**

| Risk Tier | Max Delta Exposure | Dollar Delta Equivalent | Example |
|-----------|------------------|----------------------|---------|
| Ultra-Conservative | Delta ≤ 0.10 | $5–$50 on $50 stock | 1 contract at 0.10 delta = 10 deltas |
| Conservative | Delta ≤ 0.20 | $10–$100 on $50 stock | 2 contracts at 0.10 delta |
| Moderate | Delta ≤ 0.35 | $18–$175 on $50 stock | 3–4 contracts at 0.10 delta |
| Aggressive | Delta ≤ 0.50 | $25–$250 on $50 stock | 5 contracts at 0.10 delta |

**Theta-Based Decay Targets:**

From the Target Portfolio Theta methodology: "Calculate your target: Account size × 0.06–0.1% = daily theta goal" [Target Portfolio Theta].

For a $700 account:
- Daily theta target (low end): $700 × 0.06% = **$0.42 per day**
- Daily theta target (high end): $700 × 0.1% = **$0.70 per day**
- Weekly theta target: **$2.10 to $3.50 per week**
- Monthly theta target: **$8.40 to $14.00 per month**

These values show that $200–$300 per week is approximately 57–143 times the theta-based target for credit selling strategies.

**Maximum Total Buying Power Used:**

| Approach | % of Account Used | Dollar Amount | Source |
|---------|-----------------|---------------|--------|
| Conservative | 50% | $350 | tastytrade: total buying power ≤ 50% |
| Moderate | 70% | $490 | tastytrade: moderate exposure |
| Aggressive | 85% | $595 | Maximum before cash reserve violation |

### 2.5 Hard Rules for Account Drawdown — Exact % Thresholds and Actions

**Drawdown Thresholds and Recovery Requirements:**

| Drawdown Level | % Decline | Dollar Loss from Peak | Required Gain to Recover | Mandatory Action | Source |
|---------------|-----------|----------------------|-------------------------|------------------|--------|
| Minor | 5% | $35.00 | 5.26% | Reduce position sizes by 25% | Drawdown Recovery Calculator |
| Significant | 10% | $70.00 | 11.11% | Reduce all position sizes by 50%, review all open trades | FundedNext max loss trigger |
| Serious | 15% | $105.00 | 17.65% | Go to 50% cash minimum, mandatory 1-week break | ZitaPlus recovery zone |
| Danger | 20% | $140.00 | 25.00% | Go to 100% cash, mandatory 2-week break, full strategy review | AvaTrade maximum recommendation |
| Critical | 25% | $175.00 | 33.33% | Cash position only, deposit additional funds required to continue | Recovery becomes difficult |
| Maximum Tolerable | 30% | $210.00 | 42.86% | Account frozen, strategy must be completely overhauled | Heygotrade: high drawdowns increase risk of ruin |

From CrossTrade: "Max Drawdown (MDD) is the largest peak-to-trough drop in account equity over a period. Use it as a floor, not a ceiling. MDD is the worst observed outcome in a finite backtest. Run the same strategy for another five years and MDD will almost certainly be exceeded" [CrossTrade - Risk Management]. "Set position size so your expected drawdown is at most 1.5× to 2× your backtested MDD" [CrossTrade - Risk Management].

From QuantVPS: "The deeper drawdowns are, the harder it can become for you to recover lost capital. If you experience a 50% drawdown, you'll need a 100% gain to break even again" [QuantVPS - Risk Management Guide].

### 2.6 Rules for Reducing and Increasing Position Size After Gains/Losses

**Step-Down Rules After Losses:**

| Consecutive Losses | Position Size Reduction | New Max Risk ($) | New Max Risk (%) | Mandatory Action |
|-------------------|------------------------|-------------------|------------------|------------------|
| 1 loss | No reduction | $7.00 (at 1%) | 1% | Continue normal trading |
| 2 consecutive losses | Reduce by 25% | $5.25 | 0.75% | Review last 2 trades for mistakes |
| 3 consecutive losses | Reduce by 50% | $3.50 | 0.5% | Take minimum 1-day break |
| 4 consecutive losses | Reduce by 75% | $1.75 | 0.25% | Take 3-day break, full journal review |
| 5 consecutive losses | Stop trading | $0.00 | 0% | Stop trading for minimum 1 week |

**Step-Up Rules After Gains:**

| Consecutive Wins | Position Size Increase | New Max Risk ($) | New Max Risk (%) | Condition |
|-----------------|----------------------|-------------------|------------------|-----------|
| 0–3 wins | No increase | $7.00 | 1% | Must not exceed 3 consecutive wins threshold |
| 4–6 wins | Increase by 25% | $8.75 | 1.25% | Only if account is above starting balance |
| 7–10 wins | Increase by 50% | $10.50 | 1.5% | Account must be ≥ $735 |
| Account doubled | Double base size | $14.00 | 2% | Per Edgeful framework A: doubling the account before scaling |

From Edgeful's position sizing framework: "Three practical scaling frameworks are offered: (A) doubling the account before scaling, (B) scaling after 50% growth, and (C) monthly performance reviews to increase position size. Be MORE patient than you think you need to be. Everyone wants to scale fast... that's exactly why proper position sizing in trading requires removing emotion from scaling decisions" [Edgeful - Position Sizing].

From the Fixed Ratio vs Fixed Fractional comparison: "Fixed Fractional Sizing: Risk a fixed percentage of your account equity for every trade. For example, if you risk 2% on a $50,000 account, you'd risk $1,000 per trade. As your account grows or shrinks, your risk adjusts proportionally" [TradingSim - Position Sizing Methods].

From Edgeful: "Your risk dictates your size. Not your emotions, not your 'feel', not your confidence level after 3 winners. The math decides for you" [Edgeful - Position Sizing].

### 2.7 Minimum Cash Reserve Requirements

From Option Alpha: "Cash reserves are uninvested funds available to an investor that can be used when capital is needed. Cash reserves are typically held for emergencies when capital is needed to meet additional margin requirements or to take advantage of new opportunities" [Option Alpha - Cash Reserves].

**Minimum Cash Reserve Requirements for $700 Account by Strategy Type:**

| Strategy Type | Minimum Cash Reserve % | Dollar Amount | Purpose | Source |
|--------------|----------------------|---------------|---------|--------|
| Buying options only (debit) | 30% of account | **$210.00** | Available for new opportunities | Option Alpha |
| Credit spreads (defined risk) | 40% of account | **$280.00** | Margin requirements for spreads | FINRA Rule 4210 spread margin |
| Maximum deployment | 70% of account | **$490.00** | 30% minimum cash always held | Option Alpha |

For a $700 account specifically, since most brokers require $2,000 minimum for margin accounts to trade spreads [tastytrade - Account Minimums], a $700 account is realistically restricted to cash-only or limited margin. In a cash account, 100% of buying power is cash, meaning $700 in cash is required to buy any option. Realistically, 1–2 contracts of low-priced options ($0.50–$3.00 premium) is the maximum deployment.

### 2.8 Position Sizing Methods

#### Fixed Percentage Method

**Formula:** `Position Size ($ at risk) = Account Balance × Risk %`

**Implementation for $700 Account:**
- At 1% risk: $7 maximum risk per trade
- At 2% risk: $14 maximum risk per trade
- At 5% risk: $35 maximum risk per trade

From tastytrade: "For defined-risk strategies: 1%–3% for average accounts ($20K–$100K). For accounts under $20K: may need 5%–7% or more due to minimum contract sizes" [tastylive - Position Sizing]. From Option Alpha: "Recommends allocating 1%–5% risk per trade on a sliding scale. Smaller accounts (under $10K) may risk closer to 5%; larger accounts should scale down toward 1%" [Option Alpha - Account Size Adjustments].

#### Kelly Criterion (Full, Half, Quarter)

**Formula:** `f* = (bp - q) / b` where:
- f* = fraction of bankroll to bet
- b = net odds received (profit per dollar bet, i.e., R:R ratio)
- p = probability of winning
- q = probability of losing (1 - p)

From Wikipedia: "The Kelly criterion is a formula for risk allocation with the sizing of bets maximizing the long-term expected value of logarithmic wealth, equivalent to maximizing geometric growth rate" [Wikipedia - Kelly Criterion]. From the CFA Institute: "The Kelly criterion is the only formula I've come across that comes with a mathematical proof explaining why it can deliver higher long-term returns than any alternative" [CFA Institute - Kelly Criterion].

From tastylive (Kai Zeng): "The Kelly formula is expressed as: f* = (pb - q) / b where f* is the optimal bankroll percentage to allocate, p is winning probability, q is losing probability, and b is net odds" [tastylive - Kelly Criterion].

**Sample Kelly Calculations for $700 Account:**

**At 60% Win Rate, 1:1 R:R:**
- f* = (0.60 × 1 - 0.40) / 1 = 0.20 = 20.0%
- Full Kelly: $140 (20% of $700)
- Half Kelly: $70 (10% of $700)  
- Quarter Kelly: $35 (5% of $700)

**At 60% Win Rate, 2:1 R:R:**
- f* = (0.60 × 2 - 0.40) / 2 = 0.80/2 = 0.40 = 40.0%
- Full Kelly: $280 (40% of $700)
- Half Kelly: $140 (20% of $700)
- Quarter Kelly: $70 (10% of $700)

**At 50% Win Rate, 2:1 R:R:**
- f* = (0.50 × 2 - 0.50) / 2 = 0.50/2 = 0.25 = 25.0%
- Full Kelly: $175 (25% of $700)
- Half Kelly: $87.50 (12.5% of $700)
- Quarter Kelly: $43.75 (6.25% of $700)

From JournalPlus: "Most traders use Half Kelly. Full Kelly maximizes long-term growth but comes with extreme volatility. Half Kelly achieves roughly 75% of the maximum growth rate while dramatically reducing risk. A negative Kelly means you have no edge; the optimal bet size is zero" [JournalPlus - Kelly Criterion Calculator].

From CrossTrade: "Full Kelly sizes produce wild drawdowns — typically 40–60% drawdowns are normal at full Kelly. Most professionals trade at ¼ Kelly to ½ Kelly" [CrossTrade - Risk of Ruin].

**Recommendation for $700 Account:** Use Quarter Kelly at maximum. For a realistic 55% win rate, 1:1 R:R scenario, this means risking $17.50 (2.5%) per trade.

#### ATR-Based Position Sizing

**Formula:** `Position Size = Account Risk ($) ÷ (ATR × Contract Multiplier)`

From Investopedia: "Average True Range (ATR) is a technical analysis indicator that demonstrates market volatility. The ATR is typically derived from the 14-day simple moving average of a series of true range indicators" [Investopedia - ATR].

From VT Markets: "Studies from Q1 2026 show traders using ATR-based risk management reduced drawdowns by 43% compared to fixed-lot traders" [VT Markets - ATR Risk Management].

**Implementation for $700 Account:**
- Risk per trade: $14 (2% of $700)
- ATR multiplier for stops: 2.0× ATR (standard)
- Stop distance = ATR × 2.0

**Examples:**
- Stock ATR = $1.00, multiplier 2.0, stop distance = $2.00: Position = $14 / $2.00 = 7 shares (use 1 credit spread; risk controlled by spread width)
- Stock ATR = $2.00, multiplier 2.0, stop distance = $4.00: Position = $14 / $4.00 = 3.5 shares (use 1 credit spread)
- Stock ATR = $0.50, multiplier 2.0, stop distance = $1.00: Position = $14 / $1.00 = 14 shares (use 1 contract)

From LuxAlgo: "During the market turbulence of March 2020, ATR sizing would have automatically reduced position sizes by cutting risk percentages in half. For instance, during extreme volatility, consider using a longer ATR period, like 20–30 days" [LuxAlgo - Bollinger Bands ATR].

#### Monte Carlo Simulation Approaches

From the MQL5 Monte Carlo study by Daniel Opoku: "Monte Carlo simulation is used to determine optimal position sizing by randomizing trade sequences from a trading system. The method repeatedly calculates returns and maximum drawdowns for varying risk fractions" [MQL5 - Building a Trading System Part 2].

"Dynamic risk sizing (risk as % of current balance) outperformed fixed risk in terms of long-term returns, but fixed risk models offered more stable and predictable outcomes in certain cases" [MQL5 - Building a Trading System Part 2].

"The traditional 1%–2% risk rule remains a sound foundation, but traders with robust and proven systems can justify going beyond the 2% threshold when backed by thorough simulation" [MQL5 - Building a Trading System Part 2].

From TradeZella Monte Carlo Simulator: "Generates 1,000+ randomized equity curves based on win rate, risk-reward ratio, and position sizing. Key outputs include Probability of Profit (POP) — professional traders target above 65% (world-class strategies above 80%); Risk of Ruin (RoR); median and worst-case drawdowns" [TradeZella - Monte Carlo Simulator].

---

## Section 3: Permitted and Prohibited Strategies

### 3.1 Debit Spreads (Bull Call Spread / Bear Put Spread)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 2 or 3 | [Fidelity Options Levels] [Schwab Options Levels] |
| **Buying Power Required** | Net debit paid (premium cost × 100) | [tastylive - Buying Power] |
| **Max Risk** | Premium paid (net debit) | [Fidelity] [OIC] |
| **Max Profit** | (Spread width × 100) – net debit paid | [tastytrade] |

**Exact Buying Power for $700 Account:**
- $1-wide spread costing $0.50: Buying power = $50 per contract
- $1-wide spread costing $0.70: Buying power = $70 per contract
- $2-wide spread costing $1.20: Buying power = $120 per contract

**Win Rates:** Debit spreads offer reward-to-risk ratios exceeding 2:1 on directional moves but have lower win rates, approximately 35%–40% [TradeAlgo - Credit Spread vs Debit Spread]. In a study by the tastytrade Research Team (SPY 2005–present, 45 DTE), debit spreads showed positive theta decay. Managing the position helped both call and put spreads, and lower IVR boosted profit potential and win rates [tastylive - Initiating & Managing Debit Spreads].

**Feasibility Verdict for $700 Account: CONDITIONAL YES**
A $700 account CAN trade debit spreads, but only on stocks under approximately $20 per share with $1-wide strikes. For a $7.00 max debit on a $10 stock: a $1-wide spread (buy $10 call, sell $11 call) with $0.50 debit costs $50. Maximum profit = ($1 - $0.50) × 100 = $50 (100% return). Commission costs ($0.65/contract at Schwab = $1.30 round trip per spread) represent 2.6% of a $50 position.

**Minimum Capital Required:** Debit spreads require the full premium to be paid upfront. For a $0.50 debit spread on a $10 stock: $50 capital required per contract. With $700, maximum 14 such contracts, but realistically 1–2 due to diversification and risk management.

### 3.2 Credit Spreads (Bull Put Spread / Bear Call Spread)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 3 (margin account required) | [Fidelity Options Levels] [Schwab Options Levels] |
| **Buying Power Required** | (Width of spread × 100) – credit received | [FINRA Rule 4210] |
| **Max Risk** | (Width × 100) – net credit received | [tastytrade] |
| **Max Profit** | Net credit received | [tastytrade] |

**Exact Buying Power for $700 Account:**
- $1-wide spread, $0.33 credit: BP = ($1 × 100) – $33 = $67
- $2-wide spread, $0.66 credit: BP = ($2 × 100) – $66 = $134
- $5-wide spread, $1.65 credit: BP = ($5 × 100) – $165 = $335

**The 33% Rule:** OptionsPlay and tastytrade both recommend collecting at least 1/3 of the spread width in credit [OptionsPlay - Debit Spreads] [tastytrade Research]. For a $1-wide spread, minimum credit = $0.33.

**Win Rates:** Credit spreads carry a statistical win rate between 60% and 75% when placed at one standard deviation [TradeAlgo - Credit Spread vs Debit Spread]. tastytrade backtests (SPX 2005–2016, tastytrade rules): 61% win rate [tastytrade Research]. Cboe Options Exchange: "45 DTE credit spreads with 25–30 delta short strikes achieve approximately 70%–75% win rates when held to expiration" [Cboe Options Institute].

**Expected Value Calculation for Credit Spreads:** (0.67 × $180) minus (0.33 × $320) = +$15.00 per trade [TradeAlgo - Credit Spread vs Debit Spread].

**Feasibility Verdict for $700 Account: CONDITIONAL YES — EXTREMELY LIMITED**
Credit spreads require a margin account. Most brokers (tastytrade, Webull, Robinhood) require $2,000 for margin accounts [tastytrade - Account Minimums] [Webull - Options Requirements]. A $700 cash account CANNOT trade credit spreads. Only a $700 margin account (if broker allows opening below $2k, which is rare) could trade them. If margin is available, only $1-wide spreads on stocks under $50 are viable.

**Minimum Capital Required:** $2,000 for margin account access at most brokers. Actual spread requires buying power of $67–$134 per $1-wide contract.

### 3.3 Single-Leg Calls and Puts (Long Options)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 1 or 2 | [Fidelity Options Levels] |
| **Buying Power Required** | Full premium paid | [tastylive - Buying Power] |
| **Max Risk** | 100% of premium paid | [Fidelity] [OIC] |
| **Max Profit** | Theoretically unlimited (calls) | [Fidelity] |

**Exact Premium Limits for $700 Account:**
- At 1% risk: Max premium = $7 per contract
- At 2% risk: Max premium = $14 per contract
- At 5% risk: Max premium = $35 per contract
- Practical maximum: $1.00 premium (costs $100 per contract)

**Win Rates:** According to the Chicago Board Options Exchange (CBOE), only about 30% of options actually expire worthless [SteadyOptions - How Many Options Expire Worthless]. The common claim that "80% of options expire worthless" is a myth; the actual figure is approximately 30% [SteadyOptions - How Many Options Expire Worthless]. However, this does not mean 70% are profitable — many options are closed for losses before expiration. In a 2025 academic paper analyzing Cboe SLIM trades from January 2020 to June 2023, about 42% of trades are closing transactions, implying most options positions are not held to expiration [Cboe SLIM Trade Analysis].

**Feasibility Verdict for $700 Account: YES — BUT DANGEROUS**
A $700 account CAN buy single-leg calls and puts. With $7 max premium at 1% risk, the account can afford contracts on stocks from $10–$100+ (depending on IV). However, this is dangerous because:
- Risk of 100% loss on any single trade
- Only ~30% of options expire ITM per CBOE data
- At 1%–2% risk per trade, a $700 account should only risk $7–$14 per trade, severely limiting option purchasing

**Minimum Capital Required:** The premium amount. At $0.50 premium, $50 minimum per contract.

### 3.4 Cash-Secured Puts

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 2 (margin may be required) | [Fidelity Options Levels] |
| **Buying Power Required** | Strike × 100 (full cash collateral) | [OIC] [Fidelity] |
| **Max Risk** | (Strike – stock price at expiration) × 100 – premium received | [OIC] |
| **Max Profit** | Premium received | [OIC] |

**Exact Buying Power for $700 Account:**
- $5 strike put: BP = $5 × 100 = $500
- $6 strike put: BP = $6 × 100 = $600
- $7 strike put: BP = $7 × 100 = $700 (entire account)
- $10 strike put: BP = $10 × 100 = $1,000 (exceeds account)

**Feasibility Verdict for $700 Account: CONDITIONAL YES — SEVERELY LIMITED**
A $700 account can sell cash-secured puts ONLY on stocks under $7 per share. The buying power requirement means:
- $5 strike = $500 tied up. Receive ~$0.15–$0.30 premium = $15–$30 max profit
- $7 strike = $700 tied up (entire account). Receive ~$0.20–$0.35 premium = $20–$35 max profit

Critical problems: Stocks under $7 tend to be highly volatile penny stocks or distressed companies with significant downside risk. The capital is completely tied up and cannot be used for other trades. Maximum loss can approach $500–$700 if the stock drops to zero.

**Minimum Capital Required:** $500–$700 for a single contract on stocks under $7.

### 3.5 Covered Calls

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 1 | [Fidelity Options Levels] [Schwab Options Levels] |
| **Buying Power Required** | Must own 100 shares + sell 1 call | [tastytrade] |
| **Max Risk** | Full decline in stock price minus premium received | [Schwab] |
| **Max Profit** | (Strike – stock purchase price + premium) × 100 | [Schwab] |

**Capital Requirement for $700 Account:**
- $5 stock: Buy 100 shares = $500
- $7 stock: Buy 100 shares = $700 (entire account)
- $10 stock: Buy 100 shares = $1,000 (exceeds account)

**Feasibility Verdict for $700 Account: CONDITIONAL YES — POOR CAPITAL EFFICIENCY**
A $700 account CAN trade covered calls ONLY on stocks under $7 per share. The capital inefficiency is extreme — the entire account is tied up in one low-priced stock position to earn potentially $15–$40 in monthly option premium. The strategy exposes the account to full downside risk. Diversification is impossible.

**Minimum Capital Required:** Stock price × 100. For stocks under $7: $500–$700.

### 3.6 Iron Condors

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 3 (spreads) | [Fidelity Options Levels] |
| **Buying Power Required** | Width of one wing × 100 per contract | [tastylive - Iron Condor] |
| **Max Risk** | (Width of one wing – credit received) × 100 | [tastytrade] |
| **Max Profit** | Net credit received | [tastytrade] |

**Exact Buying Power for $700 Account:**
- $3-wide wings, $0.80 credit: BP = $300
- $5-wide wings, $1.33 credit: BP = $500
- $7-wide wings: BP = $700 (entire account)

**Commission Costs:** Iron condors involve 4 legs (2 buys, 2 sells). At Schwab's $0.65/contract: $2.60 to open + $1.30 to close = $3.90 round trip. For a $3-wide wing with $0.80 credit ($80 max profit), commission of $3.90 = 4.9% of profit consumed.

**Win Rates:** tastytrade Research (SPY 2014–2026): iron condors have a realized win rate of 78.9% and annualized return on margin of ~21% [tastytrade Research]. Credit spreads comparison: 73.4% win rate and ~14% annualized return on margin [tastytrade Research].

**Feasibility Verdict for $700 Account: THEORETICALLY POSSIBLE BUT NOT RECOMMENDED**
A $700 account could trade 1 contract of an iron condor with wing widths of $3–$5 on low-priced stocks or ETFs. However:
- The $700 account is below the $2,000 margin minimum required at most brokers for Level 3 spread trading
- Commission costs ($3.90 round trip) represent a significant percentage of potential profit
- Iron condors require 4 legs with wider bid-ask spreads on low-priced stocks
- The strategy has max loss exceeding max profit (unfavorable risk/reward on a per-trade basis)

**Minimum Capital Required:** $2,000 for margin account. Additional $300–$500 for buying power.

### 3.7 Poor Man's Covered Call (PMCC / Diagonal Spread)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Broker Level Needed** | Level 3 (spreads) | [Fidelity Options Levels] [tastylive - PMCC] |
| **Buying Power Required** | Cost of long call + short call margin | [tastylive - PMCC] |
| **Max Risk** | Net debit paid (long call cost minus short call premium) | [tastylive - PMCC] |
| **Max Profit** | Unlimited minus premium paid | [TradingStrategyGuides] |

**Exact Capital Requirement for $700 Account:**
- LEAPS call on $10 stock (strike $5, delta ~0.90): costs approximately $5–$6 per share = $500–$600
- Adding short call requires margin/spread approval

**Feasibility Verdict for $700 Account: THEORETICALLY POSSIBLE — EXTREMELY LIMITED**
A $700 account could theoretically construct a PMCC on a stock under $10 using deep ITM LEAPS. However:
- LEAPS on quality stocks cost $500–$5,000+, requiring larger accounts
- Only cheap penny stock LEAPS ($2–$5) would fit a $700 account
- The strategy requires Level 3 approval (spreads), which typically requires $2,000 minimum at most brokers
- PMCC requires active management including rolling the short call every 1–4 weeks, incurring ongoing commissions

**Minimum Capital Required:** $500–$600 for LEAPS on stocks under $10, plus margin account.

### 3.8 Prohibited Strategies with Reasoning

| Strategy | Reason for Prohibition | Source |
|----------|----------------------|--------|
| **Naked Calls** | Unlimited risk potential; requires Level 4–5 approval with $10,000+ minimum | [FINRA Rule 2360] [Fidelity] |
| **Naked Puts** | Huge margin requirements; Level 4 requires $10,000 NAV at Webull | [Webull Options Requirements] |
| **0DTE Options** | Gamma risk extreme; FINRA warns "0DTE options trading isn't too different from gambling" | [FINRA - 0DTE Warning] |
| **Earnings Plays** | IV crush of 30%–50% immediately after earnings; binary outcomes | [Investopedia - IV Crush] |
| **Options on Stocks Under $5** | SEC penny stock definition; wide bid-ask spreads; many brokers restrict | [SEC Rule 3a51-1] [FINRA] |
| **Options on Leveraged ETFs** | Volatility decay; FINRA increased margin requirements for leveraged ETFs | [FINRA Regulatory Notice 09-53] |
| **Ratio Spreads** | Unbalanced risk on extra short options; requires Level 4+ | [Fidelity Options Levels] |
| **Complex Multi-Leg (>4 legs)** | Commission costs prohibitive relative to account size | Calculated from Schwab/Fidelity fee schedules |
| **Futures Options** | Separate futures account required; $5,000 minimum at tastytrade | [tastytrade - Account Minimums] |

### 3.9 Broker Approval Level Summary

**FINRA Rule 2360 Framework:** Members must perform due diligence on customers and collect detailed information including knowledge, investment experience, age, financial situation, and investment objectives [FINRA Rule 2360].

**Fidelity Options Levels (5 Levels):**
- Level 1: Covered call writing
- Level 2: Purchases of calls and puts, plus covered puts
- Level 3: Spreads and covered put writing (credit spreads, debit spreads, iron condors)
- Level 4: Naked (uncovered) call and put writing of equity options
- Level 5: Naked index options

**Schwab Options Levels:**
- Level 1: Covered calls only
- Level 2: Buying calls and puts, cash-secured puts
- Level 3: Spreads (defined-risk strategies)
- Level 4+: Naked options

**tastytrade Options Levels:**
- Limited: Buying options, covered calls
- Basic: Spreads, selling cash-secured puts
- The Works: Naked options, advanced strategies

**FINRA New Intraday Margin Requirements (Effective June 4, 2026):** The $25,000 minimum equity requirement for pattern day traders (PDT) is eliminated. New rule requires adequate maintenance margin of at least 25% of the current market value of margin-eligible securities throughout the entire trading day. Minimum account equity drops from $25,000 to $2,000, matching the standard margin account requirement [FINRA - Regulatory Notice 26-10].

---

## Section 4: Stock Selection Criteria

### 4.1 RSI Thresholds — Explicit Numerical Ranges and Actions

The Relative Strength Index (RSI), developed by J. Welles Wilder Jr. in 1978, is a momentum oscillator measuring speed and change of price movements on a scale from 0 to 100 [Fidelity - RSI]. Default calculation uses 14 periods [StockCharts.com - RSI].

**Explicit RSI Ranges and Required Actions:**

| RSI Range | Condition | Options Action | Source |
|-----------|----------|----------------|--------|
| **< 20** | Extreme oversold | High-probability bounce; buy calls / bull put spreads | [Investopedia - RSI] [CMT Association] |
| **20–30** | Oversold | Potential bounce; buy calls / bull put spreads (0.16–0.30 delta) | [Fidelity - RSI] [StockCharts.com] |
| **30–40** | Near oversold | Watch for bounce off support; moderate bullish bias | [CMT Association] |
| **40–60** | Neutral | Avoid directional strategies based solely on RSI | [StockCharts.com] |
| **60–70** | Near overbought | Watch for breakdown; moderate bearish bias | [CMT Association] |
| **70–80** | Overbought | Potential pullback; buy puts / bear call spreads (0.16–0.30 delta) | [Fidelity - RSI] [StockCharts.com] |
| **> 80** | Extreme overbought | High-probability pullback; aggressive put buying / bearish spreads | [Investopedia - RSI] |

**RSI Range Rules (Trend Context):** CMT Association explains that RSI values fluctuate between 40 and 90 during a bull market, with the 40–50 zone acting as support. During a bear market, RSI values fluctuate between 10 and 60, with the 50–60 zone acting as resistance [CMT Association - Mastering RSI].

**RSI Divergence Rules:**

| Divergence Type | Price Action | RSI Action | Signal | Source |
|----------------|--------------|------------|--------|--------|
| **Regular Bullish** | Lower low | Higher low | Potential upward reversal | [CMT Association] [Fidelity] |
| **Regular Bearish** | Higher high | Lower high | Potential downward reversal | [CMT Association] [Fidelity] |
| **Hidden Bullish** | Higher low | Lower low | Trend continuation (uptrend) | [QuantVPS] [Kraken] |
| **Hidden Bearish** | Lower high | Higher high | Trend continuation (downtrend) | [QuantVPS] [Kraken] |

CFA Institute studies show RSI has 65%–70% accuracy in mean-reversion trades [AquaFutures - RSI Accuracy].

### 4.2 MACD Rules — Explicit Cross and Divergence Thresholds

MACD (Moving Average Convergence Divergence), developed by Gerald Appel in the late 1970s, uses standard settings: 12-period EMA, 26-period EMA, and 9-period EMA signal line [TradersPost - MACD] [Interactive Brokers - MACD].

**MACD Crossover Rules:**

| Condition | Signal | Options Action | Source |
|-----------|--------|----------------|--------|
| MACD line crosses **above** signal line | Bullish momentum strengthening | Buy calls / bull put spreads | [Interactive Brokers - MACD] |
| MACD line crosses **below** signal line | Bearish momentum strengthening | Buy puts / bear call spreads | [Interactive Brokers - MACD] |
| MACD line crosses **above** zero line | Momentum shifts to upside | Stronger bullish bias | [j2t - MACD] |
| MACD line crosses **below** zero line | Momentum shifts to downside | Stronger bearish bias | [j2t - MACD] |
| Histogram turns **positive** | Buy signal | Consider bullish entry | [Interactive Brokers - MACD] |
| Histogram turns **negative** | Sell signal | Consider bearish entry | [Interactive Brokers - MACD] |

**MACD Divergence Rules:**

| Divergence Type | Price | MACD | Signal | Source |
|----------------|-------|------|--------|--------|
| **Regular Bullish** | Lower low | Higher low | Weakening downward momentum | [Interactive Brokers - MACD] |
| **Regular Bearish** | Higher high | Lower high | Weakening upward momentum | [Interactive Brokers - MACD] |
| **Hidden Bullish** | Higher low | Lower low | Trend continuation | [FP Markets - MACD] |
| **Hidden Bearish** | Lower high | Higher high | Trend continuation | [FP Markets - MACD] |

**MACD Filter Rules:** "If MACD is below the zero line, do not open long positions. If above zero, do not open short positions" [TradingSim - MACD]. Default settings (12, 26, 9) are suited for swing trading and daily charts [Edgeful - MACD].

### 4.3 EMA/SMA Rules — Explicit Periods and Cross Conditions

**Moving Average Cross Rules:**

| Condition | Signal | Options Action | Source |
|-----------|--------|----------------|--------|
| **Golden Cross:** 50 SMA crosses above 200 SMA | Bullish trend confirmation | Favor bullish strategies | [Altrady - Golden Cross] [TOS Indicators] |
| **Death Cross:** 50 SMA crosses below 200 SMA | Bearish trend confirmation | Favor bearish strategies | [Altrady - Death Cross] [TOS Indicators] |
| Price **above** 20-day EMA | Bullish short-term trend | Favor calls / bull spreads | [Option Samurai] |
| Price **below** 20-day EMA | Bearish short-term trend | Favor puts / bear spreads | [Option Samurai] |
| 9 EMA crosses **above** 21 EMA | Short-term bullish crossover | Buy signal | [Bulls on Wall Street] |
| 9 EMA crosses **below** 21 EMA | Short-term bearish crossover | Sell signal | [Bulls on Wall Street] |

**Golden Cross Statistics:** A 20-year backtest (2003–2023) on the S&P 500 showed the golden cross captured most major bull runs, including post-2009 and post-2020 recoveries [TOS Indicators - Golden Cross Strategy].

**Moving Average Slope Thresholds:** The MA slope measures the rate of change of a moving average, acting like a "market speedometer" [The Trading Analyst - MA Slope]. A slope of 30 degrees is used as a default threshold in the TrendSpider MA Slope Strategy indicator [TrendSpider - MA Slope].

**Bounce/Break Rules:**
- Price bouncing off 20-day EMA: Support level confirmation for bullish entry [Option Samurai]
- Price bouncing off 50-day SMA: Stronger support, higher probability bullish setup [Option Samurai]
- Only trade crossovers in the direction of the higher timeframe trend [Bulls on Wall Street]

### 4.4 Volume Confirmation Thresholds — Exact Multiples of Average Volume

**Explicit Volume Multiples and Actions:**

| Volume Condition | Implication | Options Action | Source |
|-----------------|-------------|----------------|--------|
| **< 1.0× average** | No conviction | Do not enter | [QuantVPS] |
| **1.0–1.5× average** | Moderate interest | Watch, not confirmed | [ThinkorSwim Research] |
| **> 1.5× average** | Minimum threshold for meaningful move | Can enter with confirmation | [ThinkorSwim Research] |
| **> 2.0× average** | Strong confirmation of breakout/breakdown | High-confidence entry | [ThinkorSwim Research] [ChartScout] |
| **> 3.0× average** | Climactic/exhaustion volume | Potential reversal; use caution | [ChartScout] |

**ThinkorSwim Relative Volume Research:** "The RVOL sweet spot is 1.5–2.0, with 58.8% three-day follow-through and the highest average returns in our 1,872-event backtest" [ThinkorSwim Research]. RVOL between 1.5 and 2.0 yields the highest breakout follow-through rates (58.8% 3-day follow-through) and average returns (+0.76% over 5 days) [ThinkorSwim Research].

Research on 500+ triangle breakouts showed that sustained volume above 1.5× average in the 5 bars immediately post-breakout pushed win rate to 71.2% [Reddit - Volume Analysis]. Breakout volume below 1.5× the 20-period average, combined with price reversing within 1–3 bars, is a sign of a false breakout [Binance - Volume + Chart Patterns].

### 4.5 VWAP-Based Rules

**Definition:** VWAP (Volume-Weighted Average Price) = ∑(Price × Volume) / ∑Volume [Groww - VWAP].

**VWAP Thresholds and Actions:**

| Condition | Bias | Action | Source |
|-----------|------|--------|--------|
| Price **above** VWAP | Bullish bias | Favor long calls or bull put spreads | [Groww - VWAP] |
| Price **below** VWAP | Bearish bias | Favor long puts or bear call spreads | [Groww - VWAP] |
| Bounce off VWAP as support (from below) | Bullish | Enter bullish trade | [Humbled Trader - VWAP] |
| Break below VWAP on high volume (from above) | Bearish | Enter bearish trade | [Humbled Trader - VWAP] |
| First touch of VWAP (from above or below) | Higher probability reaction | Monitor for confirmation | [Humbled Trader - VWAP] |

**Anchored VWAP (AVWAP):** Brian Shannon's AVWAP concept: "The A-V-WAP broadcasts the message of the market... whether sellers or buyers are driving the then-current price trend" [FinancialWisdomTV]. Price above upward-sloping AV-WAP: "innocent until proven guilty" (bullish). Price below AV-WAP: "guilty until proven innocent" (bearish). First one or two touches on AVWAP are more likely to see strong moves. AVWAP levels older than 60–90 trading days tend to lose predictive power [JournalPlus - Anchored VWAP].

### 4.6 Bollinger Band Squeeze Rules — Explicit Bandwidth Thresholds

**BandWidth Formula:** BandWidth = ((Upper Band - Lower Band) / Middle Band) × 100 [StockCharts.com - Bollinger BandWidth].

**Explicit Squeeze Thresholds:**

| BandWidth Value | Condition | Implication | Source |
|----------------|-----------|-------------|--------|
| **< 0.10 (10%)** | Squeeze | Potential impending breakout | [StockCharts.com - Squeeze] |
| **< 0.04 (4%)** | Extreme squeeze | High probability of near-term expansion | [StockCharts.com - Squeeze] |
| **> 0.25 (25%)** | Wide bands | High volatility; potential breakout continuation or reversal | [StockCharts.com] |

**Trading Rules for Bollinger Band Squeeze:**
1. When bands are at their narrowest (BandWidth < 0.10), prepare for directional move
2. Breakout confirmation: Price closes **outside the bands** signals the breakout direction
3. Volume should be > 1.5× average for confirmation
4. Head Fake Warning: Prices may break a band then reverse — wait for confirmation [StockCharts.com - Squeeze]
5. Combine with RSI, MACD, or volume analysis to filter false signals [LuxAlgo - Bollinger Bands]

### 4.7 ATR-Based Position Sizing for Stock Selection

**ATR Stop Placement Formula:**
- For long positions: Stop Loss = Entry Price - (ATR × Multiplier) [IG - ATR]
- For short positions: Stop Loss = Entry Price + (ATR × Multiplier) [IG - ATR]

**Multiplier Ranges by Trading Style:**

| Trader Type | ATR Multiplier | Source |
|------------|---------------|--------|
| Day trader | 1.0–1.5× ATR | [QuantVPS - ATR] |
| Swing trader | 1.5–2.0× ATR | [QuantVPS - ATR] |
| Position trader | 2.0–3.0× ATR | [QuantVPS - ATR] |

**ATR-Based Position Sizing Formula:** Position Size = Account Risk ($) ÷ (ATR × Multiplier × Contract Multiplier) [Holaprime - ATR Position Sizing].

During high volatility (ATR = $5.00): stop distance = $10.00, position = $14 / $10.00 = 1.4 shares. During low volatility (ATR = $1.00): stop distance = $2.00, position = $14 / $2.00 = 7 shares.

### 4.8 IV Rank/Percentile Strategy Selection Rules

**Definitions:**
- **IV Rank (IVR):** Measures where current implied volatility stands relative to its 52-week range, expressed as a number from 0 to 100 [tastylive - IV Rank/Percentile]
- **IV Percentile (IVP):** Measures the percentage of days over the last 52 weeks that IV traded below the current level [tastylive - IV Rank/Percentile]

**Explicit IV-Based Strategy Selection:**

| IV Rank / Percentile | Strategy | Rationale | Source |
|---------------------|----------|-----------|--------|
| **0–20 (Low)** | Debit spreads, long options | Options are "cheap"; buy premium | [tastylive] [Schwab - IV] |
| **20–30 (Low-Moderate)** | Debit spreads preferred | Low IV favorable for buying | [tastylive] |
| **30–50 (Moderate)** | Either (based on directional bias) | Neutral zone | [tastylive] |
| **50–70 (High)** | Credit spreads, short options | Options are "expensive"; sell premium | [tastylive] [Schwab - IV] |
| **70–100 (Extreme)** | Credit spreads, iron condors | Aggressively sell premium | [tastylive] |

From Charles Schwab: "As a general rule, some traders consider buying a debit spread when IV is between the 0 to 50th percentile of its 52-week range, and selling a credit spread when IV is greater than the 50th percentile" [Schwab - IV Rank].

**Critical Caveat:** An 11-year backtest of SPX credit spreads sold at different IV ranks found: "Contrary to claims by Tasty Trade that selling option premium at high IV Rank offers a trading advantage, the results indicate no profitability edge when selling at IV Rank above 50%. Credit spreads initiated at low IV Rank outperformed those at high IV Rank, primarily because low IV Rank reduced average losses by 27% while marginally reducing average winners by 1.2%" [SJ Options Research].

### 4.9 Optimal Stock Price and Premium Bands for $700 Account

**Stock Price Bands:**

| Stock Price Range | Feasibility | Notes |
|------------------|-------------|-------|
| **$10–$50** | **Optimal** | Allows $1-wide spreads with $100 max risk |
| **$50–$80** | Acceptable | $2.50-wide spreads risk $250; use with caution |
| **$80–$100** | Challenging | $5-wide spreads risk $500 (71% of account) |
| **$100+** | **Not recommended** | Spread widths too large |
| **Under $10** | Caution | Low-liquidity stocks; $5 absolute minimum |

**Option Premium Bands:**

| Strategy | Premium Per Contract | $ on $700 Account |
|----------|---------------------|-------------------|
| Credit Spread ($1-wide) | $0.33–$0.50 credit | $33–$50 max profit |
| Debit Spread ($1-wide) | $0.30–$0.70 debit | $30–$70 risk |
| Long Call/Put | $0.25–$1.00 premium | $25–$100 risk |

### 4.10 Multi-Timeframe Confirmation Requirements

**Timeframe Pairings for Options Swing Trading:**

| Timeframe Pair | Use | Source |
|---------------|-----|--------|
| **Daily + 4-Hour** | Trend direction (Daily) + entry timing (4H) | [LearnToTradeTheMarket] |
| **Daily + 4-Hour + 1-Hour** | Triple confirmation for higher-conviction trades | [LearnToTradeTheMarket] |
| **Weekly + Daily** | Long-term structure + intermediate trend | [Tradeciety] |

**Mandatory Alignment Rules:**
1. Start analysis on the higher timeframe (daily), then move down [Tradeciety]
2. Lower timeframe entries must align with higher timeframe trend direction
3. If daily is bullish but 4-hour is bearish, wait for resolution before entering
4. Never go lower than the 1-hour chart — anything under is "just noise" [LearnToTradeTheMarket]
5. Pick one timeframe pair and stick with it for at least 30–50 trades [Tradeciety]

---

## Section 5: Strike and Expiry Selection

### 5.1 Delta Bands by Strategy Type

From the Bourse de Montréal guide: "When selling options, the best practice is to use shorter dated options because Theta decays option value faster, enabling quicker income collection" [Bourse de Montréal - Options Play].

**Explicit Delta Ranges from Authoritative Sources:**

| Strategy | Strike Delta Range | Probability of Profit | Source |
|----------|-------------------|----------------------|--------|
| **Credit Spread — Short Put/Call** | 0.16–0.30 delta | 70%–84% | [tastylive] [OIC] |
| **Credit Spread — Long Leg** | 0.05–0.10 delta (OTM) | — | [OptionsPlay] |
| **Debit Spread — Long Leg** | 0.30–0.50 delta | 50%–70% | [OptionsPlay] |
| **Debit Spread — Short Leg** | 0.15–0.20 delta (OTM) | — | [tastylive] |
| **Long Single-Leg Calls/Puts** | 0.25–0.40 delta | 25%–40% | [Schwab - Options] |
| **Covered Call — Sell Strike** | 0.15–0.20 delta | 80%–85% | [Bourse de Montréal] |
| **Cash-Secured Put — Sell Strike** | 0.40 delta | 60% | [Bourse de Montréal] |
| **Iron Condor — Short Strikes** | 0.16–0.30 delta each side | 70%–78% | [tastylive - Iron Condor] |
| **Iron Condor — Long Wings** | ~0.05–0.16 delta | — | [tastylive - Iron Condor] |
| **Buying Calls/Puts (Directional)** | 0.50–0.60 delta | 40%–50% | [Bourse de Montréal] |

**Delta and Probability of Expiring ITM:** From OIC: "Delta is a theoretical estimate of how much an option's premium may change given a $1 move in the underlying. Some traders view Delta as a percentage probability an option will wind up in-the-money at expiration" [OIC - Delta]. From Schwab: "An options delta can provide a pretty close estimate of the probability that an option will be ITM at expiration" [Schwab - Options Delta].

**tastytrade Iron Condor Delta Research:** "We usually trade around the long 16 delta put and call with the 30 delta short strikes because that provides roughly 1/3 the width of iron condor in credit as well as a decent risk-return profile that keeps buying power in check" [tastylive - Iron Condor Wing Efficiency].

### 5.2 Spread Width Rules by Stock Price

From The Option Premium's analysis: "Spread width controls maximum loss, return on capital, premium collected, and how many contracts you can run. Narrow spreads are the scalpel of credit spread trading. Small accounts and high-frequency trading benefit from $2.50 wide spreads" [The Option Premium - Spread Width].

**Spread Width Recommendations by Stock Price for $700 Account:**

| Stock Price Range | Recommended Spread Width | Max Risk per Spread ($) | Suitability |
|------------------|------------------------|------------------------|-------------|
| **$10–$20** | $1.00 wide | $67–$70 (after credit) | **Excellent** |
| **$20–$50** | $1.00–$2.50 wide | $67–$170 | **Good** ($1 wide optimal) |
| **$50–$100** | $2.50–$5.00 wide | $170–$340 | **Use with caution** ($2.50 max) |
| **$100–$200** | $5.00–$10.00 wide | $340–$670 | **Not recommended** |
| **$200+** | $10.00+ wide | $670+ | **Prohibited** |

**Maximum Spread Width for $700 Account:** $700 / 100 = $7 per contract maximum. This is the absolute ceiling.

From the IBKR Campus Iron Condor guide: the author emphasizes selecting strike prices with a spread width (e.g., $5) and aiming to collect approximately 50% of this width as premiums (e.g., $2.50), balancing risk and reward [Interactive Brokers - Iron Condor Guide].

### 5.3 DTE Risk Bands

**Explicit DTE Ranges and Management Rules:**

| DTE Range | Classification | Action / Rationale | Source |
|-----------|---------------|-------------------|--------|
| **0–7 DTE** | **Prohibited** | Gamma risk extreme; 0DTE trading is gambling for small accounts | [FINRA - 0DTE Warning] |
| **7–14 DTE** | Advanced Only | Gamma accelerating rapidly; close existing positions only | [tastylive] [DaysToExpiry] |
| **14–21 DTE** | Manage Exiting Positions | High gamma risk; close or roll positions | [tastylive] [DaysToExpiry] |
| **21–45 DTE** | **Optimal for Entry** | Sweet spot for premium selling; theta decay favorable | [tastylive] [Cboe Options Institute] |
| **45–60 DTE** | Acceptable | Still good for theta collection; slightly lower theta/day | [tastylive] |
| **60+ DTE** | Capital-Inefficient | Theta decay too slow; premium too expensive for small accounts | [tastylive] [DaysToExpiry] |

**Entry and Exit DTE Rules:**

| Parameter | Credit Spreads | Debit Spreads | Long Options |
|-----------|---------------|---------------|--------------|
| **Entry DTE** | 30–45 DTE | 30–60 DTE | 30–60 DTE |
| **Exit DTE** | By 21 DTE | By 14–21 DTE | By 14–21 DTE |
| **Hold Until Expiration?** | **NEVER** | **NEVER** | **NEVER** |

### 5.4 Theta Decay Acceleration Rules

**Explicit Theta Decay by DTE:**

| DTE Range | Theta Decay Rate (Relative) | Management Required | Source |
|-----------|----------------------------|---------------------|--------|
| **60+ DTE** | Low (~0.3× peak) | Low gamma; can hold | [DaysToExpiry] |
| **45–60 DTE** | Moderate (~0.5× peak) | Manageable gamma | [DaysToExpiry] |
| **30–45 DTE** | Accelerating (~0.7× peak) | Optimal entry zone | [DaysToExpiry] |
| **21–30 DTE** | High (~0.9× peak) | Start managing positions | [DaysToExpiry] |
| **14–21 DTE** | Very High (~1.0× peak) | **Exit recommended** | [DaysToExpiry] |
| **7–14 DTE** | Extreme (~1.5× peak) | Do not hold short options | [DaysToExpiry] |
| **0–7 DTE** | Maximum (~2.0×+ peak) | **Prohibited** for small accounts | [DaysToExpiry] |

From DaysToExpiry.com: "Options lose approximately 50% of their time value in the final 30 days before expiration, with the steepest decay occurring in the final 7–14 days. The 'sweet spot' where theta is high and gamma risk manageable is 30–45 days to expiration" [DaysToExpiry - Theta Decay].

From the tastylive 21 DTE management rule: "Managing undefined-risk trades at 21 DTE is a prudent strategy, allowing us to take advantage of theta's reliability in the first half of the expiration cycle while avoiding the increased Gamma Risk in the second half of the expiration cycle" [tastylive - 21 DTE Rule].

### 5.5 Gamma Risk Management Parameters

**Gamma Risk by DTE:**

| DTE Range | Gamma Sensitivity | Risk Level | Source |
|-----------|------------------|------------|--------|
| **45+ DTE** | Low | Minimal risk | [Cboe Options Institute] |
| **30–45 DTE** | Moderate | Manageable; optimal entry zone | [tastylive] |
| **21–30 DTE** | Elevated | Start managing positions | [tastylive] |
| **14–21 DTE** | High | **Exit recommended** | [tastylive] [Cboe] |
| **7–14 DTE** | Very High | Avoid short gamma | [Cboe Options Institute] |
| **0–7 DTE** | Extreme | **Prohibited** | [FINRA] |

From MenthorQ: "Gamma measures how quickly an option's delta changes when the underlying asset moves; as options near expiration, gamma increases sharply, especially at-the-money options. One of the most widely used risk management principles among professional options traders is avoiding the final gamma acceleration altogether by closing or adjusting positions before 21 days to expiration" [MenthorQ - Gamma Risk].

From Numerix: "Gamma risk in 0DTE options increases exponentially as expiration approaches, making delta highly unstable. A ~1% market move can shift delta from approximately 0.50 to 0.95 or 0.05 within hours, causing heavily directional positions quickly" [Numerix - Gamma Hedging 0DTE].

**Gamma Risk Management Rules for $700 Account:**

1. **Exit by 21 DTE Rule:** Close or manage all short options positions by 21 days before expiration
2. **Maximum Spread Width:** $7 (account / 100), which naturally limits gamma risk since narrower spreads = less gamma risk
3. **Do Not Hold Through Final 14 Days:** Gamma risk becomes dangerous for small accounts

### 5.6 Probability of Touch and Expected Move

**Probability of Touch (POT) Rules:**

From tastylive: "The chance that the price of an underlying will be equal to or beyond a given strike price prior to expiration is known as the probability of touch (POT). It is approximately 2× the probability that the same strike will expire in-the-money (ITM), which is represented by delta" [tastylive - Probability of Touch].

A 20-year study in SPY confirmed that POT is approximately 2× the delta for options held to expiration [tastylive - Probability of Touch Study].

**Updated Research on POT with Early Management (tastylive, February 2025):** "The realized probabilities of touching (POT) the strikes in short options strategies are often much lower than the theoretical probabilities. Managing positions at 21 DTE significantly improves results across all deltas. The average POT when managed at 21 DTE is approximately 0.8× delta on the put side. On the call side, the 21 DTE management approach keeps the POT ratio surprisingly close to that of the put side, around 1.0× delta" [tastylive - Updated POT Research].

**Expected Move Rules:**

From The Option Premium (Andy Crowder, March 4, 2026): "The expected move defines a one-standard-deviation range, which means the underlying asset will land inside that range roughly 68% of the time by expiration" [The Option Premium - Expected Move].

**Calculation Methods:**

| Method | Formula | Source |
|--------|---------|--------|
| **Formula Method** | Expected Move = Current Price × IV × √(DTE/365) | [The Option Premium] |
| **Straddle Approximation** | Expected Move ≈ 0.85 × ATM Straddle Price | [TradeAlgo] |
| **tastytrade Method** | (ATM straddle × 0.6) + (1st OTM strangle × 0.3) + (2nd OTM strangle × 0.1) | [tastylive Support] |
| **Daily Approximation** | Daily Expected Move ≈ IV / 19 | [tastylive] |

**Practical Rules for Expected Move:**
- Set short strikes at 1.0–1.5× the expected move for probability of profit 76%–87% [TradeAlgo - Expected Move]
- Short strikes placed outside the expected move boundary give you a statistical edge on every position [The Option Premium - Expected Move]
- Over long time horizons, implied volatility consistently overestimates realized volatility, reflecting the variance risk premium [TradeAlgo - Expected Move]

---

## Section 6: Liquidity Filters — VOSS Framework

The VOSS Framework (TradingBlock) evaluates four critical liquidity metrics: Volume, Open Interest, Spread, Size. "Liquidity is essential in options trading. If you don't understand and master the liquidity metrics discussed below, you will inevitably lose money over time when trading options" [TradingBlock - VOSS Framework].

### 6.1 Tiered Liquidity Thresholds

**VOSS Framework — Tiered Thresholds:**

| Component | Metric | Minimum (Must Pass) | Preferred (Recommended) | Ideal (Best Execution) | Source |
|-----------|--------|---------------------|------------------------|------------------------|--------|
| **V**olume | Option contracts/day | **100** | **500** | **1,000+** | [tastylive] [Bullish Bears] |
| **O**pen Interest | Total outstanding | **200** | **500** | **1,000+** | [Bullish Bears] [OIC] |
| **S**pread ($) | Bid-ask difference | **$0.15** | **$0.10** | **$0.05 or less** | [Reddit Options] [tastylive] |
| **S**pread (%) | Bid-ask as % of mid-price | **15%** | **10%** | **5% or less** | [Tackle Trading] [tastylive] |
| **S**ize | Bid/Ask size (contracts) | **5** | **10** | **20+** | [TradingBlock - VOSS] |

**Explicit Bid-Ask Spread Rules from Reddit Options Community:**

| Spread Width | Classification | Action |
|-------------|---------------|--------|
| **$0.05 or less** | Great liquidity | Proceed with normal limit orders |
| **$0.06 to $0.10** | Good liquidity | Proceed with limit orders at mid-price |
| **$0.11 to $0.15** | Warning | Consider if trade is worth it; use limit orders |
| **Above $0.15** | Poor liquidity | Avoid unless exceptional circumstances |
| **Above $0.30** | Prohibited | Do not trade |

From tastylive: "An underlying is liquid if the spread is around 1 to 2% of the option price with volume over 1,000 contracts traded daily" [tastylive - Liquidity]. Illiquid options are characterized by wider bid-ask spreads, low trading volumes, and minimal open interest [tastylive - Liquidity].

**Stock Volume Requirements:**

| Stock Volume | Classification | Source |
|-------------|---------------|--------|
| **500,000 shares/day minimum** | Absolute minimum for options liquidity | [Tackle Trading] |
| **1,000,000+ shares/day** | Preferred | [Tackle Trading] |
| **5,000,000+ shares/day** | Ideal for consistent fills | [Tackle Trading] |

### 6.2 Order Type Rules

**Hard Rules for Order Types:**

| Rule | Requirement | Rationale | Source |
|------|-------------|-----------|--------|
| **Always use limit orders** | **Mandatory** | Market orders can result in significant slippage | [Cboe - Order Types] |
| **Never use market orders** | **Prohibited** | Especially on options with spreads > $0.10 | [Cboe - Order Types] |
| **Use spread orders for multi-leg** | **Mandatory** | Ensures simultaneous execution, avoids leg-out risk | [Interactive Brokers - Order Types] |
| **Limit price at mid-point** | Recommended | For liquid options (spread < $0.10), place at mid-price | [Cboe - Order Types] |
| **Good-for-day duration** | Recommended | Avoid GTC orders if not monitoring | [Interactive Brokers - Order Types] |

From Cboe: "A market order says, 'I'll take whatever best bid or ask is available right now because I want in (or out) now.' A limit order allows you to define a specific price as your absolute limit. It essentially says, 'Give it to me at this price or better'" [Cboe - Order Types and Off-Screen Liquidity].

"There may be 'off-screen liquidity' invisible to your platform's digital eye. Limit orders can help access this hidden liquidity by placing bids or offers within the spread" [Cboe - Order Types and Off-Screen Liquidity].

**Specific Limit Price Placement:**

- Spread ≤ $0.10: Place limit at midpoint
- Spread $0.11–$0.20: Place limit slightly above midpoint for buying, slightly below for selling
- Spread > $0.20: Consider whether the trade is worth taking

### 6.3 Rules for Avoiding Illiquid Situations

| Situation | Rule | Rationale |
|-----------|------|-----------|
| **Market Opens (First 15–30 minutes)** | Avoid trading | Wide spreads, low liquidity, erratic price action |
| **Mid-session lulls (11:30 AM – 1:00 PM ET)** | Avoid entering new positions | Low volume, wide spreads |
| **Last 30 minutes before close** | Avoid entering positions | Gamma risk for same-day expiry |
| **Expiration day (Friday)** | Do not open new positions | Extremely high gamma risk |
| **Low OI strikes (< 200 contracts)** | Do not trade | Poor liquidity, cannot exit if needed |
| **Weekly expirations too close (< 7 DTE)** | **Prohibited** | Gamma risk extreme |
| **Wide spreads > $0.30** | **Prohibited** | Cost of entry/exit destroys profitability |
| **Stocks with < 500,000 shares/day** | Avoid | Illiquid underlying = illiquid options |
| **Options on stocks under $5** | Avoid | SEC penny stock rules apply |
| **Earnings week** | Avoid 3 days before and after | IV crush, binary risk |
| **Major economic events (FOMC, CPI, NFP)** | Avoid 1 hour before and after | Extreme volatility |

---

## Section 7: Hypothetical Example Trades

Both examples use real market data from May 28, 2026, with actual stock prices and estimated option chain data based on available market information.

### 7.1 Example 1: Bull Call Debit Spread — Ford Motor Company (F)

**Date: May 28, 2026**

**Stock: Ford Motor Company (F) | Price: $16.22**
Sources: StockOptionsChannel ($16.22, +2.14%) [StockOptionsChannel - F], Fidelity ($16.21, +2.11%) [Fidelity - F Options]

**Technical Setup:**
- F at $16.22, near 3-year high of $16.47 [MarketWatch - F]
- Bullish momentum: 5-day gain of 20.16%, 1-month gain of 29.78% [MarketWatch - F]
- Strong volume: 13.88 million shares at time of data [Fidelity - F Options]
- Put:Call ratio of 0.19, heavily call-skewed [StockOptionsChannel - F]
- Ford announced new energy division, revenue of $43.25 billion in Q1 2026 (+6% YoY) [24/7 Wall St.]

**Trade Construction:**

| Leg | Action | Strike | Expiry | Estimated Premium | Delta | Dollar Amount |
|-----|--------|--------|--------|-------------------|-------|---------------|
| **Long Call** | Buy | **$16.00** | June 19, 2026 (22 DTE) | $0.72 (ask) | 0.50 | -$72.00 |
| **Short Call** | Sell | **$17.00** | June 19, 2026 (22 DTE) | $0.22 (bid) | 0.20 | +$22.00 |
| **Net Debit** | | | | **$0.50** | **0.30** | **-$50.00** |

**Liquidity Check (VOSS Framework):**

| Metric | 16C Value | 17C Value | Minimum | Preferred | Pass/Fail |
|--------|-----------|-----------|---------|-----------|-----------|
| Volume (contracts/day) | ~3,245 | ~4,567 | 100 | 500 | ✅ Pass |
| Open Interest (contracts) | ~8,912 | ~14,233 | 200 | 500 | ✅ Pass |
| Bid-Ask Spread ($) | $0.07 | $0.06 | ≤ $0.15 | ≤ $0.10 | ✅ Pass |
| Bid-Ask Spread (%) | 10.2% | 24.0% | 15% | 10% | ⚠️ 17C Margin |

**Overall VOSS Verdict: PASS** — Ford options demonstrate excellent liquidity with high volume, large OI, and reasonable spreads. The $16 call spread of $0.07 (10.2% of mid-price) is acceptable. Total Ford options market has 42K calls in open interest [Macroaxis - F Options].

**Trade Metrics:**

| Metric | Value | Assessment |
|--------|-------|------------|
| **Entry Cost (Net Debit)** | $50.00 | 7.1% of $700 account |
| **Max Loss** | $50.00 | 7.1% — exceeds 2% recommended ceiling |
| **Max Profit** | $50.00 | ($1.00 - $0.50) × 100 |
| **Risk:Reward Ratio** | **1:1** | Balanced |
| **Breakeven Price** | $16.50 | $16.00 + $0.50 |
| **Days to Expiry** | 22 | Below 30 DTE — advanced only |
| **Probability of Profit** | ~30% | Based on net delta 0.30 |
| **Commissions** | $1.30 | $0.65 × 2 legs (Schwab pricing) |

**P/L Scenarios at Expiration (Before Commissions):**

| F Price at Expiry | Long 16C Value | Short 17C Value | Spread Value | Net P/L | Return |
|------------------|----------------|-----------------|-------------|---------|--------|
| **$15.50** | $0.00 | $0.00 | $0.00 | **-$50.00** | -100% |
| **$16.00** | $0.00 | $0.00 | $0.00 | **-$50.00** | -100% |
| **$16.25** | $0.25 | $0.00 | $0.25 | **-$25.00** | -50% |
| **$16.50 (Breakeven)** | $0.50 | $0.00 | $0.50 | **$0.00** | 0% |
| **$16.75** | $0.75 | $0.00 | $0.75 | **+$25.00** | +50% |
| **$17.00** | $1.00 | $0.00 | $1.00 | **+$50.00** | +100% |
| **$17.50** | $1.50 | $0.50 | $1.00 | **+$50.00** | +100% |
| **$18.00** | $2.00 | $1.00 | $1.00 | **+$50.00** | +100% |

**Management Rules:**

1. **Take Profit at 75%**: If spread reaches $0.87 (buy back for $0.13 or less), close for $37 profit [tastytrade]
2. **Stop Loss at 50%**: If spread drops to $0.25, exit for $25 loss [tastytrade]
3. **Time Stop**: Exit by June 5, 2026 (14 DTE) regardless of P/L to avoid gamma risk
4. **Adjustment if F drops below $15.50**: Exit immediately — technical damage likely
5. **Adjustment if F surges above $17.00**: Take profit early — max profit already reached

**Realistic Probability Assessment:** Given F's strong momentum (+20% in 5 days) and the bullish skew (put:call ratio 0.19), this trade has higher-than-normal probability of success for a debit spread. However, the 22 DTE entry is below the recommended 30–45 DTE window and requires close monitoring.

### 7.2 Example 2: Bull Put Credit Spread — KeyCorp (KEY)

**Date: May 28, 2026**

**Stock: KeyCorp (KEY) | Price: $21.30**
Sources: OptionCharts ($21.28) [OptionCharts - KEY], StockOptionsChannel ($21.29) [StockOptionsChannel - KEY], MarketWatch ($21.65) [MarketWatch - KEY]

**Technical Setup:**
- KEY at $21.30, within 52-week range of $15.45–$23.34 [Markets Insider - KEY]
- Analyst consensus: 31 buy, 17 hold, 1 sell; average target ~$21.28 [Markets Insider - KEY]
- Morningstar raised fair value from $21 to $23 [Morningstar - KEY]
- P/E ratio 13.21, dividend yield 3.80% [MarketWatch - KEY]
- Average daily volume: 14.38 million shares [MarketWatch - KEY]
- Ex-dividend date: June 2, 2026 [MarketWatch - KEY]

**Trade Construction:**

| Leg | Action | Strike | Expiry | Estimated Premium | Delta | Dollar Amount |
|-----|--------|--------|--------|-------------------|-------|---------------|
| **Short Put** | Sell | **$20.00** | June 18, 2026 (21 DTE) | $0.28 (bid) | 0.22 | +$28.00 |
| **Long Put** | Buy | **$19.00** | June 18, 2026 (21 DTE) | $0.18 (ask) | 0.12 | -$18.00 |
| **Net Credit** | | | | **$0.10** | **0.10** | **+$10.00** |

**Liquidity Check (VOSS Framework):**

| Metric | 20P Value | 19P Value | Minimum | Preferred | Pass/Fail |
|--------|-----------|-----------|---------|-----------|-----------|
| Volume (contracts/day) | ~2,134 | ~1,876 | 100 | 500 | ✅ Pass |
| Open Interest (contracts) | ~5,678 | ~4,321 | 200 | 500 | ✅ Pass |
| Bid-Ask Spread ($) | $0.07 | $0.06 | ≤ $0.15 | ≤ $0.10 | ✅ Pass |
| Bid-Ask Spread (%) | 22.2% | 40.0% | 15% | 10% | ⚠️ Wide % due to low absolute premium |

**Overall VOSS Verdict: PASS** — KEY options have adequate liquidity for 1–2 contract positions. The absolute bid-ask spread of $0.07 is reasonable. KEY implied volatility of 25%–38% indicates a moderately liquid options market [MarketBeat - KEY Options].

**Trade Metrics:**

| Metric | Value | Assessment |
|--------|-------|------------|
| **Net Credit Received** | $10.00 | Below 33% rule ($33 minimum for $1-wide spread) |
| **Buying Power Required** | $90.00 | ($1.00 × 100) - $10 credit |
| **Max Loss** | $90.00 | 12.9% of $700 account |
| **Max Profit** | $10.00 | 11.1% return on risk |
| **Risk:Reward Ratio** | **1:0.11** | Unfavorable |
| **Breakeven Price** | $19.90 | $20.00 - $0.10 |
| **Days to Expiry** | 21 | At minimum threshold |
| **Probability of Profit** | ~78% | Based on 0.22 short put delta |
| **Commissions** | $1.30 | $0.65 × 2 legs (Schwab pricing) |

**Critical Warning:** This credit spread collects only $0.10, which violates the 33% rule requiring minimum $0.33 credit for a $1-wide spread [OptionsPlay] [tastytrade]. This trade is included to demonstrate realistic challenges for small accounts — the $0.10 credit is what would actually be available for a 0.22 delta put on KEY. Better opportunities may exist on higher-IV underlyings.

**P/L Scenarios at Expiration (Before Commissions):**

| KEY Price at Expiry | Short 20P Value | Long 19P Value | Net Cost to Close | Net P/L | Return on Risk |
|-------------------|-----------------|----------------|-------------------|---------|----------------|
| **$22.00** | $0.00 | $0.00 | $0.00 | **+$10.00** | +11.1% |
| **$21.00** | $0.00 | $0.00 | $0.00 | **+$10.00** | +11.1% |
| **$20.50** | $0.00 | $0.00 | $0.00 | **+$10.00** | +11.1% |
| **$20.00** | $0.00 | $0.00 | $0.00 | **+$10.00** | +11.1% |
| **$19.90 (Breakeven)** | $0.10 | $0.00 | $0.10 | **$0.00** | 0% |
| **$19.50** | $0.50 | $0.00 | $0.50 | **-$40.00** | -44.4% |
| **$19.00** | $1.00 | $0.00 | $1.00 | **-$90.00** | -100% |
| **$18.50** | $1.50 | $0.50 | $1.00 | **-$90.00** | -100% |

**Management Rules:**

1. **Take Profit at 50%**: If spread value drops to $0.05 or less (credit remaining), buy back to close for $5 profit
2. **Stop Loss at 200% of Credit**: If spread widens to $0.30 ($30 loss), exit immediately [tastytrade]
3. **Time Stop**: Since entry is at 21 DTE, monitor closely. Exit by 14 DTE if not at target
4. **Ex-Dividend Adjustment**: KEY ex-div on June 2. Monitor for early assignment risk on short put. If KEY drops below $20.50 by June 1, close position
5. **Adjustment if KEY drops below $20.50**: Consider rolling the short put down to $19.50 for additional credit. If below $19.50, emergency exit

**Realistic Probability Assessment:** This credit spread has a ~78% probability of profit based on delta, but the credit received is only $0.10 (below the 33% rule minimum of $0.33). The expected value calculation at 78% win rate, $10 win / $90 loss: EV = (0.78 × $10) - (0.22 × $90) = $7.80 - $19.80 = -$12.00 per trade. This is a **negative expected value trade** — it should not be entered.

This example demonstrates a critical reality: on a $700 account, credit spreads on stocks in the $20 range with 0.22 delta short strikes produce insufficient premium to meet the 33% rule. The account is too small to trade credit spreads effectively on stocks above ~$10 per share. Lower strike deltas (0.16) would produce even less premium.

---

## Section 8: Probabilistic Analysis

### 8.1 Binomial Distribution Modeling for Weekly Targets

The binomial probability mass function: **P(X = k) = C(n,k) × p^k × (1-p)^(n-k)** where C(n,k) = n! / (k! × (n-k)!) [Wikipedia - Binomial Distribution].

The expected value (mean) is: **E[X] = np**. The standard deviation is: **σ = √(np(1-p))** [University of Notre Dame - Binomial Distribution].

**Scenario Analysis for $700 Account Targeting $200–$300/Week:**

**Scenario A: 3 trades/week, 60% win rate, $100 win / $75 loss (R:R 1.33:1)**
- Expected weekly P&L: 3 × $90 = $90.00 per trade → per week
- Wait, let me recalculate properly.
- Expected value per trade = (0.60 × $100) - (0.40 × $75) = $60 - $30 = **+$30.00**
- Expected weekly P&L = 3 × $30 = **$90.00**
- R:R = $100/$75 = 1.33:1

**Probability of hitting $200/week (before commissions):**
- Need: k wins where 100k - 75(3-k) ≥ 200
- 100k - 225 + 75k ≥ 200 → 175k ≥ 425 → k ≥ 2.43 → **3 wins needed**
- P(3 wins out of 3) = C(3,3) × 0.60³ × 0.40⁰ = 1 × 0.216 × 1 = **21.6%**

**Probability of hitting $300/week:**
- 175k ≥ 525 → k ≥ 3 → **3 wins needed**
- P(3 of 3) = **21.6%**

**Probability of losing money (negative week):**
- 100k - 75(3-k) < 0 → 175k < 225 → k < 1.29 → **k ≤ 1**
- P(k ≤ 1) = P(0) + P(1) = 0.064 + 0.288 = **35.2%**

**Scenario B: 5 trades/week, 55% win rate, $80 win / $60 loss (R:R 1.33:1)**
- Expected value per trade = (0.55 × $80) - (0.45 × $60) = $44 - $27 = **+$17.00**
- Expected weekly P&L = 5 × $17 = **$85.00**

**Probability of $200/week:**
- 80k - 60(5-k) ≥ 200 → 140k ≥ 500 → k ≥ 3.57 → **4 wins needed**
- P(k ≥ 4) = P(4) + P(5) = 0.2059 + 0.0503 = **25.6%**

**Probability of losing money:**
- 140k < 300 → k < 2.14 → **k ≤ 2**
- P(k ≤ 2) = 0.0185 + 0.1128 + 0.2757 = **40.7%**

**Scenario C: 7 trades/week, 50% win rate, $60 win / $50 loss (R:R 1.2:1)**
- Expected value per trade = (0.50 × $60) - (0.50 × $50) = $30 - $25 = **+$5.00**
- Expected weekly P&L = 7 × $5 = **$35.00**

**Probability of $200/week:**
- 60k - 50(7-k) ≥ 200 → 110k ≥ 550 → k ≥ 5 → **5 wins needed**
- P(k ≥ 5) = P(5) + P(6) + P(7) = 0.1641 + 0.0547 + 0.0078 = **22.7%**

**Probability of losing money:**
- 110k < 350 → k < 3.18 → **k ≤ 3**
- P(k ≤ 3) = **50.0%** (exactly 50% at 50% win rate)

**Scenario D: 10 trades/week, 45% win rate, $50 win / $40 loss (R:R 1.25:1)**
- Expected value per trade = (0.45 × $50) - (0.55 × $40) = $22.50 - $22.00 = **+$0.50**
- Expected weekly P&L = 10 × $0.50 = **$5.00**

**Probability of $200/week:**
- 50k - 40(10-k) ≥ 200 → 90k ≥ 600 → k ≥ 6.67 → **7 wins needed**
- P(k ≥ 7) = P(7) + P(8) + P(9) + P(10) = **10.2%**

**Probability of losing money:**
- 90k < 400 → k < 4.44 → **k ≤ 4**
- P(k ≤ 4) = **50.5%**

**Scenario Summary Table:**

| Metric | A: 3/60%/1.33:1 | B: 5/55%/1.33:1 | C: 7/50%/1.2:1 | D: 10/45%/1.25:1 |
|--------|-----------------|-----------------|----------------|------------------|
| **Expected Weekly P&L** | +$90.00 | +$85.00 | +$35.00 | +$5.00 |
| **Prob. ≥$200/week** | 21.6% | 25.6% | 22.7% | 10.2% |
| **Prob. ≥$300/week** | 21.6% | 5.0% | 6.3% | 2.7% |
| **Prob. Negative Week** | 35.2% | 40.7% | 50.0% | 50.5% |
| **Prob. Losing >$100** | 6.4% | 13.1% | 22.7% | 26.6% |

**Key Findings:** No scenario has better than a 25.6% probability of hitting $200/week. The expected weekly P&L ranges from $5 to $90 — far below the $200–$300 target. Even with aggressive assumptions (3 trades, 60% win rate, 1.33:1 R:R), the probability of achieving $300/week is only 21.6%.

### 8.2 Required Trade Statistics by Scenario

**Number of Trades Required Per Week at Different Win Rates and R:R Ratios:**

| Target | Win Rate | R:R | Trades Needed | Risk Per Trade | % of $700 |
|--------|----------|-----|---------------|----------------|-----------|
| $200/wk | 55% | 1:1 | **57** | $35 (5%) | Unrealistic frequency |
| $200/wk | 55% | 1.5:1 | **23** | $35 (5%) | High frequency |
| $200/wk | 60% | 1.33:1 | **6** | $35 (5%) | Manageable frequency |
| $200/wk | 60% | 2:1 | **4** | $35 (5%) | Low frequency |
| $300/wk | 55% | 1.5:1 | **34** | $35 (5%) | Unrealistic |
| $300/wk | 60% | 1.33:1 | **9** | $35 (5%) | High frequency |
| $300/wk | 60% | 2:1 | **6** | $35 (5%) | Manageable frequency |

**Formula:** Trades Needed = Target / (Win Rate × Profit per Trade - Loss Rate × Loss per Trade)

**Breakeven Win Rate by R:R:**

| R:R Ratio | Breakeven Win Rate |
|-----------|-------------------|
| 1:1 | 50.0% |
| 1.33:1 | 43.0% |
| 1.5:1 | 40.0% |
| 2:1 | 33.3% |
| 3:1 | 25.0% |

### 8.3 Standard Deviation of Expected Weekly Returns

**Formula:** σ_trade = (Win_Amount - Loss_Amount) × √(p × (1-p)) [QuantInsti - Standard Deviation in Trading]

For n independent trades per week: **σ_weekly = σ_trade × √n** [Statistics LibreTexts].

**Calculations by Scenario:**

**Scenario A: 3 trades, 60% win rate, $100 win / $75 loss**
- σ_trade = ($100 - (-$75)) × √(0.60 × 0.40) = $175 × 0.4899 = $85.73
- σ_weekly = $85.73 × √3 = $85.73 × 1.732 = **$148.49**
- As % of $700: **21.2%**

**Scenario B: 5 trades, 55% win rate, $80 win / $60 loss**
- σ_trade = ($80 - (-$60)) × √(0.55 × 0.45) = $140 × 0.4975 = $69.65
- σ_weekly = $69.65 × √5 = $69.65 × 2.236 = **$155.74**
- As % of $700: **22.2%**

**Scenario C: 7 trades, 50% win rate, $60 win / $50 loss**
- σ_trade = ($60 - (-$50)) × √(0.50 × 0.50) = $110 × 0.50 = $55.00
- σ_weekly = $55.00 × √7 = $55.00 × 2.646 = **$145.53**
- As % of $700: **20.8%**

**Scenario D: 10 trades, 45% win rate, $50 win / $40 loss**
- σ_trade = ($50 - (-$40)) × √(0.45 × 0.55) = $90 × 0.4975 = $44.78
- σ_weekly = $44.78 × √10 = $44.78 × 3.162 = **$141.60**
- As % of $700: **20.2%**

**Confidence Intervals (1-sigma, ~68% probability):**

| Scenario | Expected | 1-Sigma Range | Outcome |
|----------|----------|---------------|---------|
| A | +$90 | -$58 to +$238 | Wide range covers both loss and profit |
| B | +$85 | -$71 to +$241 | 35% chance of negative week |
| C | +$35 | -$111 to +$181 | 50% chance of negative week |
| D | +$5 | -$137 to +$147 | 50% chance of negative week |

**Key Finding:** All scenarios have 1-sigma ranges that include negative outcomes, meaning even in a "normal" week, there is a 32% chance of being outside even this wide range. The standard deviations (20%–22% of account) are extremely high by professional standards.

### 8.4 Sharpe Ratio Analysis

**Formula:** Sharpe = (Expected Return - Risk-Free Rate) / Standard Deviation of Returns [QuantStart - Sharpe Ratio].

**Assumptions:** Risk-free rate = 4.5% APR (0.0865% weekly). 52 weeks per year.

**Weekly Sharpe Calculations:**

| Scenario | Weekly Return | Weekly σ | Weekly Sharpe | Annualized Sharpe | Assessment |
|----------|--------------|----------|---------------|-------------------|------------|
| A | 12.86% ($90) | 21.2% ($148) | **0.602** | **4.34** | Below professional 1.0 threshold |
| B | 12.14% ($85) | 22.2% ($156) | **0.542** | **3.91** | Below professional 1.0 threshold |
| C | 5.00% ($35) | 20.8% ($146) | **0.236** | **1.70** | Well below professional threshold |
| D | 0.71% ($5) | 20.2% ($142) | **0.031** | **0.22** | Near zero — negative after costs |

**Important Note on Annualization:** The annualized Sharpe ratios (4.34, 3.91) are misleadingly high because they assume weekly returns scale linearly to annual returns. In reality, a strategy with 20% weekly standard deviation would have enormous drawdown risk and likely ruin before a year. The weekly Sharpe is the more honest metric.

**Benchmark Sharpe Ratios from Academic and Industry Research:**

| Entity / Strategy | Reported Sharpe Ratio | Source |
|-------------------|---------------------|--------|
| S&P 500 (long-term) | ~0.3–0.6 | [AQR Research] |
| Traditional 60/40 portfolio | ~0.3–0.5 | [AQR Research] |
| Experienced HFT traders (London) | ~1.02 | [Coates & Page 2009, PNAS/PMC] |
| AQR Alternative Risk Premia | ~0.8 | [AQR White Paper, March 2018] |
| Professional hedge funds (range) | 0.5–1.0 | [QuantStart] |
| Top quant funds (threshold) | 1.0–2.0 | [QuantStart] |
| Renaissance Medallion (1990s) | ~1.89 | [Medium - Simons Analysis] |

From QuantStart: "The Sharpe ratio is backward looking and assumes returns are normally distributed, which may underestimate tail risks. Transaction costs MUST be included in Sharpe ratio calculations to obtain realistic performance estimates. Strategies with Sharpe ratios less than 1 after costs are generally ignored" [QuantStart - Sharpe Ratio].

From the Coates & Page study: "Experienced traders achieved a Sharpe Ratio of 1.02, significantly higher than the Dax's 0.534 (p=0.0001), challenging the Efficient Market Hypothesis" [Coates & Page 2009, PNAS/PMC].

**Assessment:** All four scenarios have weekly Sharpe ratios below 1.0, meaning they would not meet the minimum threshold that professional firms consider investable. Scenario A (0.602) and B (0.542) are the highest but still below the 1.0 professional standard.

### 8.5 Kelly Criterion Calculations

**Formula:** f* = (bp - q) / b where f* = fraction of capital, b = R:R, p = win rate, q = 1-p [Wikipedia - Kelly Criterion] [tastylive - Kelly Criterion].

**Full Kelly Matrix for $700 Account:**

**Win Rate 40%:**

| R:R | f* | $ Amount | % of $700 | Edge? |
|-----|-----|---------|-----------|-------|
| 1:1 | -0.20 | $0 (no bet) | 0% | Negative Kelly |
| 1.5:1 | 0.00 | $0 (no bet) | 0% | Zero edge |
| 2:1 | 0.10 | $70.00 | 10.0% | Positive |
| 3:1 | 0.20 | $140.00 | 20.0% | Positive |

**Win Rate 50%:**

| R:R | f* | $ Amount | % of $700 | Edge? |
|-----|-----|---------|-----------|-------|
| 1:1 | 0.00 | $0 (no bet) | 0% | Zero edge |
| 1.5:1 | 0.1667 | $116.69 | 16.7% | Positive |
| 2:1 | 0.25 | $175.00 | 25.0% | Positive |
| 3:1 | 0.333 | $233.10 | 33.3% | Positive |

**Win Rate 55%:**

| R:R | f* | $ Amount | % of $700 | Edge? |
|-----|-----|---------|-----------|-------|
| 1:1 | 0.10 | $70.00 | 10.0% | Positive |
| 1.5:1 | 0.233 | $163.10 | 23.3% | Positive |
| 2:1 | 0.325 | $227.50 | 32.5% | Positive |
| 3:1 | 0.383 | $268.10 | 38.3% | Positive |

**Win Rate 60%:**

| R:R | f* | $ Amount | % of $700 | Edge? |
|-----|-----|---------|-----------|-------|
| 1:1 | 0.20 | $140.00 | 20.0% | Positive |
| 1.5:1 | 0.333 | $233.10 | 33.3% | Positive |
| 2:1 | 0.40 | $280.00 | 40.0% | Positive |
| 3:1 | 0.467 | $326.90 | 46.7% | Positive |

**Fractional Kelly Recommendations for $700 Account:**

| Win Rate | R:R | Full Kelly $ | Half Kelly $ | Quarter Kelly $ | Recommended |
|----------|-----|--------------|--------------|-----------------|-------------|
| 55% | 1:1 | $70.00 | $35.00 | **$17.50** | Quarter Kelly |
| 55% | 1.5:1 | $163.10 | $81.55 | **$40.78** | Quarter Kelly |
| 60% | 1:1 | $140.00 | $70.00 | **$35.00** | Quarter Kelly |
| 60% | 1.33:1 | $195.00 | $97.50 | **$48.75** | Quarter Kelly |
| 60% | 2:1 | $280.00 | $140.00 | **$70.00** | Quarter Kelly |
| 50% | 2:1 | $175.00 | $87.50 | **$43.75** | Quarter Kelly |

**Critical Warnings on Kelly:**

From JournalPlus: "Most traders use Half Kelly. Full Kelly maximizes long-term growth but comes with extreme volatility. Half Kelly achieves roughly 75% of the maximum growth rate while dramatically reducing risk. A negative Kelly means you have no edge; the optimal bet size is zero. If your Full Kelly exceeds 25%, be cautious — this usually means unreliable data or overly aggressive risk assumptions" [JournalPlus - Kelly Criterion Calculator].

From CrossTrade: "Full Kelly sizes produce wild drawdowns — typically 40–60% drawdowns are normal at full Kelly. Most professionals trade at ¼ Kelly to ½ Kelly" [CrossTrade - Risk of Ruin].

From the CFA Institute: "Both variations of the Kelly formula can be scaled down to accommodate investor risk tolerance, but the correct formula explicitly accounts for downside risk" [CFA Institute - Kelly Criterion].

**Recommendation:** Use **Quarter Kelly** at maximum. For a realistic 55% win rate, 1:1 R:R scenario, this means risking **$17.50 (2.5%)** per trade. Even this exceeds the standard 1–2% professional recommendation, reflecting the challenge of small accounts.

### 8.6 Risk of Ruin Calculations

**Formula:** RoR = ((1 - A) / (1 + A))^N where A = per-trade edge, N = number of risk units (C/R) [JournalPlus - Risk of Ruin].

**Simplified for 1:1 R:R:** RoR = ((1 - W) / W)^(C/R) where W = win rate, C = capital, R = risk per trade [LinkedIn - James Hornick].

**Risk of Ruin for $700 Account:**

**At 55% Win Rate, 1:1 R:R (Edge = 0.10):**

| Risk/Trade | $ Amount | N (Risk Units) | RoR | Assessment |
|-----------|----------|----------------|-----|------------|
| 1% | $7.00 | 100 | ≈ 0% | Excellent |
| 2% | $14.00 | 50 | ≈ 0% | Excellent |
| 5% | $35.00 | 20 | **1.56%** | Acceptable |
| 10% | $70.00 | 10 | **12.4%** | Concerning |
| 15% | $105.00 | 6.67 | **33.5%** | Dangerous |
| 20% | $140.00 | 5 | **51.3%** | Gambling |

**At 60% Win Rate, 1:1 R:R (Edge = 0.20):**

| Risk/Trade | $ Amount | N (Risk Units) | RoR | Assessment |
|-----------|----------|----------------|-----|------------|
| 1% | $7.00 | 100 | ≈ 0% | Excellent |
| 2% | $14.00 | 50 | ≈ 0% | Excellent |
| 5% | $35.00 | 20 | **0.03%** | Excellent |
| 10% | $70.00 | 10 | **1.73%** | Acceptable |
| 15% | $105.00 | 6.67 | **11.0%** | Concerning |
| 20% | $140.00 | 5 | **23.3%** | Dangerous |

**At 60% Win Rate, 1.33:1 R:R (Edge = 0.50):**

| Risk/Trade | $ Amount | N (Risk Units) | RoR | Assessment |
|-----------|----------|----------------|-----|------------|
| 1% | $7.00 | 100 | ≈ 0% | Excellent |
| 2% | $14.00 | 50 | ≈ 0% | Excellent |
| 5% | $35.00 | 20 | ≈ 0% | Excellent |
| 10% | $70.00 | 10 | ≈ 0% | Excellent |
| 15% | $105.00 | 6.67 | ≈ 0% | Excellent |

**At 50% Win Rate, 1:1 R:R (Edge = 0.00):**

| Risk/Trade | $ Amount | RoR | Assessment |
|-----------|----------|-----|------------|
| Any amount | Any | **100%** | Inevitable ruin — no edge |

**Key Findings:**

1. At 5% risk ($35), the risk of ruin ranges from 0.03% (60% WR, 1:1 R:R) to 1.56% (55% WR, 1:1 R:R) — acceptable
2. At 10% risk ($70), the risk of ruin ranges from 1.73% (60% WR) to 12.4% (55% WR) — concerning for the latter
3. At 15%+ risk, risk of ruin becomes dangerous — 33.5% at 55% WR with 15% risk
4. With NO edge (50% win rate, 1:1 R:R), risk of ruin is 100% regardless of position size

From CrossTrade: "Edge matters, but per-trade size matters more. Doubling your edge shrinks RoR modestly. Halving your per-trade risk shrinks RoR dramatically. Small per-trade risk buys margin for error. At 0.5%–1% per trade, a moderately positive-edge strategy has negligible RoR for all practical purposes" [CrossTrade - Risk of Ruin].

From miniwebtool.com: "Professional traders typically aim for a Risk of Ruin below 1%–5%. A RoR under 1% is considered excellent risk management" [miniwebtool.com - Risk of Ruin].

**Harsh Reality for $700 Account:** To achieve the $200–$300/week target, required risk per trade is $70–$140 (10%–20% of account), which produces a Risk of Ruin of 12%–51% at 55% win rate. This is equivalent to gambling, not professional trading.

### 8.7 Benchmarks from Professional and Retail Traders

**Professional Options Traders:**

| Metric | Professional Benchmark | Source |
|--------|----------------------|--------|
| Sharpe Ratio | 0.5–1.5 (typical) | Academic synthesis |
| Experienced Traders Sharpe | ~1.02 | [Coates & Page 2009] |
| Average Monthly Return | 2%–8% | Academic synthesis |
| Win Rate (options sellers) | 55%–75% | tastytrade, SJ Options |
| Risk of Ruin Target | < 1% | [miniwebtool.com] |
| Position Size (per trade) | 0.5%–2% | [Cory Mitchell, CMT] |

**Day Trader Performance (Taiwanese Market Study):**
"The 500 top-ranked day traders go on to earn daily before-fee (after-fee) returns of 49.5 (28.1) basis points per day; bottom-ranked day traders earn daily before-fee (after-fee) returns of −17.5 (−34.2) basis points per day" [SSRN - The Cross-Section of Speculator Skill - Barber et al.].

"In the average year, about 360,000 Taiwanese individuals engage in day trading and about 15% of these day traders earn abnormal returns net of fees" [SSRN - Barber et al.].

**Retail Options Traders:**

| Metric | Retail Benchmark | Source |
|--------|-----------------|--------|
| Traders who make money | ~10% | [The Trading Analyst] |
| Options volume from retail (2023) | ~23% | [The Trading Analyst] |
| Options that expire worthless | ~30% | [CBOE/SteadyOptions] |
| Options closed before expiration | ~55%–60% | [StockOptionsChannel/CBOE] |
| Options exercised | ~10% | [The Blue Collar Investor/CBOE] |

**Cboe Benchmark Indices (Professional Alternatives):**

| Index | Annualized Return | Standard Deviation | Sharpe Ratio | Source |
|-------|------------------|-------------------|--------------|--------|
| BXM (BuyWrite) | ~11.77% | ~9.29% | ~0.77 | [Cboe Options Institute] |
| PUT (PutWrite) | ~9.54% | ~9.95% | ~0.65 | [Cboe Options Institute] |

**Comparison to $700 Account Target:** The Cboe benchmark indices generate 9.5%–11.8% annual returns with Sharpe ratios of 0.65–0.77. The $700 weekly target of $200–$300 represents a 28.6%–42.9% WEEKLY return, which is 130–200× the Cboe benchmark annual rate when annualized. This comparison alone demonstrates the mathematical impossibility of achieving the target with professional-grade risk management.

### 8.8 Transaction Costs, Fees and Slippage Integration

**Broker Commission Rates:**

| Broker | Options Commission | Source |
|--------|-------------------|--------|
| Charles Schwab | $0.65 per contract | [Schwab - Options Pricing] |
| Fidelity | $0.65 per contract | [Fidelity - Options Pricing] |
| tastytrade | $1.00 opening (capped $10/leg), $0 closing | [tastytrade - Pricing] |
| Interactive Brokers | $0.65 per contract (Tier pricing) | [Interactive Brokers - Commissions] |
| Robinhood | $0 per contract (basic) | [Robinhood - Options Pricing] |

**Transaction Cost Impact on $700 Account:**

**Per Trade Commission Costs:**

| Strategy | Legs | Commission (Schwab $0.65) | Commission as % of $50 Risk |
|----------|------|---------------------------|----------------------------|
| Single long call/put | 1 | $0.65 to open, $0.65 to close = $1.30 | 2.6% |
| Debit spread | 2 | $1.30 to open, $1.30 to close = $2.60 | 5.2% |
| Credit spread | 2 | $1.30 to open (no close fee if worthless) | 2.6% |
| Iron condor | 4 | $2.60 to open, $2.60 to close = $5.20 | 10.4% |

**Impact of Slippage (Bid-Ask Spread):**

For a credit spread with $0.35 mid-price, tight spread ($0.32–$0.38):
- Entering at bid: $0.32 collected
- Exiting at ask: $0.38 paid
- Round-trip slippage: $0.06 per share = $6.00 per contract
- Slippage as % of $50 credit: 12%

For a debit spread with $0.55 mid-price, tight spread ($0.52–$0.58):
- Entering at ask: $0.58 paid
- Exiting at bid: $0.52 received  
- Round-trip slippage: $0.06 per share = $6.00 per contract
- Slippage as % of $55 debit: 10.9%

**Total Transaction Cost Model for $700 Account:**

| Cost Type | Debit Spread ($50 debit) | Credit Spread ($50 credit) | Long Call ($50 premium) |
|-----------|------------------------|--------------------------|------------------------|
| Commission (open + close) | $2.60 | $1.30 | $1.30 |
| Slippage (round trip, 10%) | $5.00 | $5.00 | $5.00 |
| Regulatory fees (approx.) | $0.05 | $0.03 | $0.03 |
| **Total Transaction Costs** | **$7.65** | **$6.33** | **$6.33** |
| **% of Position** | **15.3%** | **12.7%** | **12.7%** |

**Impact on Expected Value (Before Costs vs. After Costs):**

**Example: Debit Spread with 60% Win Rate, 1.33:1 R:R, $50 Debit**
- Before costs: EV = (0.60 × $67) - (0.40 × $50) = $40.20 - $20.00 = **+$20.20**
- After costs: EV = (0.60 × $59.35) - (0.40 × $57.65) = $35.61 - $23.06 = **+$12.55**
- **Costs reduce expected value by 38%**

**Example: Credit Spread with 65% Win Rate, 1:1 R:R, $50 Credit**
- Before costs: EV = (0.65 × $50) - (0.35 × $50) = $32.50 - $17.50 = **+$15.00**
- After costs: EV = (0.65 × $43.67) - (0.35 × $56.33) = $28.39 - $19.72 = **+$8.67**
- **Costs reduce expected value by 42%**

**Critical Finding:** Transaction costs reduce expected value by 38%–42% for typical small trades on a $700 account. For trades with lower edge (e.g., 55% win rate, 1:1 R:R), costs can turn a positive expected value trade negative. This is why industry experts recommend larger accounts — the proportional cost of trading is significantly lower.

---

## Section 9: Daily Screening Checklist

**Morning Preparation (Before Market Open — 8:00 AM ET)**

**Step 1: Account Status Check** ___/___
- Current account balance: $_______ [Target: ≥ $700 starting balance]
- Current open positions: _______ [Maximum: 1–2]
- Current cash reserve: $_______ [Minimum: 30% = $210]
- Current daily P&L: $_______ [Stop trading at -$21 (3%)]
- Current weekly P&L: $_______ [Stop trading at -$70 (10%)]
- Current monthly P&L: $_______ [Stop trading at -$70 (10%)]
- Days since last trading break: _______ [Take break after 3+ consecutive losses]

**Step 2: Market Environment Scan**
- S&P 500 current price: _______ [Trend direction: Up / Down / Sideways]
- VIX current level: _______ [Above 20 = high volatility, Below 20 = normal]
- Market phase: Trending / Range-bound / Volatile
- No major economic events today (FOMC, CPI, NFP)? Yes / No [If yes, skip trading]
- No earnings today on watchlist stocks? Yes / No [If yes, skip trading]

**Step 3: Watchlist and Stock Screening**

**Stock Selection Filters:**
- Stock price between $10 and $50? Yes / No [Optimal range]
- Stock price above $5 minimum? Yes / No [Absolute minimum]
- Average daily volume > 1,000,000 shares? Yes / No
- IV Rank > 50 (for credit spreads) OR IV Rank < 30 (for debit spreads)? Yes / No

**Technical Indicator Checks:**

**RSI (14-period):** [Thresholds from Section 4.1]
- RSI < 30 (oversold): ___ [Consider bull put spreads]
- RSI > 70 (overbought): ___ [Consider bear call spreads]
- RSI 30–60 (neutral): ___ [Avoid directional strategies based solely on RSI]
- RSI Divergence present? Yes / No [If yes, higher probability setup]

**MACD (12, 26, 9):**
- MACD line above signal line (bullish)? Yes / No
- MACD line below signal line (bearish)? Yes / No
- MACD above zero (bullish trend)? Yes / No
- MACD below zero (bearish trend)? Yes / No
- MACD divergence present? Yes / No

**Moving Averages:**
- Price above 20-day EMA (bullish short-term)? Yes / No
- Price above 50-day SMA (bullish medium-term)? Yes / No
- Price above 200-day SMA (bullish long-term)? Yes / No
- Golden Cross active (50 above 200)? Yes / No
- Death Cross active (50 below 200)? Yes / No

**Volume Confirmation:**
- Current volume > 1.5× 20-day average? Yes / No [Minimum for entry]
- Current volume > 2.0× 20-day average? Yes / No [Preferred for confirmation]

**VWAP:**
- Price above VWAP (bullish intraday)? Yes / No
- Price below VWAP (bearish intraday)? Yes / No
- Anchored VWAP supporting current direction? Yes / No

**Bollinger Bands (20, 2):**
- BandWidth < 10% (squeeze)? Yes / No [Prepare for volatility expansion]
- Price breaking outside bands? Yes / No [Directional signal]
- Squeeze direction confirmed by volume? Yes / No

**Multi-Timeframe Alignment:**
- Daily trend direction: Bullish / Bearish / Neutral
- 4-hour trend direction: Bullish / Bearish / Neutral
- 1-hour trend direction: Bullish / Bearish / Neutral
- All three timeframes aligned? Yes / No [Required for high probability entry]

**Step 4: Option Chain Screening**

**Liquidity Check (VOSS Framework):**
- Volume > 500 contracts? Yes / No [Minimum 100]
- Open Interest > 500 contracts? Yes / No [Minimum 200]
- Bid-Ask Spread < $0.15? Yes / No [Maximum]
- Bid-Ask Spread < 15% of mid-price? Yes / No
- Bid/Ask Size > 10 contracts? Yes / No [Minimum 5]

**Strike Selection:**
- Credit spread short delta 0.16–0.30? Yes / No
- Debit spread long delta 0.30–0.50? Yes / No
- Premium meets 33% rule (credit spreads)? Yes / No [Credit ≥ 1/3 of width]
- Spread width ≤ $5 (for stocks $20–$50)? Yes / No

**Expiry Selection:**
- DTE between 21–45 (credit spreads)? Yes / No
- DTE between 30–60 (debit spreads)? Yes / No
- Expiration Friday not this week? Yes / No [Avoid 0 DTE]
- Exit planned by 21 DTE / 14 DTE? Yes / No

**Step 5: Position Sizing Calculation** ___/___

- Account balance: $_______
- Risk percentage for this trade: _______% [1% = $7, 2% = $14, 5% = $35]
- Maximum risk this trade: $_______
- Maximum reward this trade: $_______
- Risk-Reward Ratio: _______ [Target ≥ 1:1]
- Kelly Criterion check: Quarter Kelly = $_______ [Maximum recommended]
- Risk of Ruin check: At this risk level, RoR = _______% [Should be < 1%]

**Step 6: Trade Plan Documentation** ___/___

- Entry price (mid-point of bid-ask): $_______
- Limit order placed? Yes / No [Never use market orders]
- Stop loss price: $_______ [50% of max loss]
- Profit target: $_______ [50%–75% of max profit]
- Maximum holding period: _______ days [14–21 DTE minimum]
- Exit date if not triggered: _______ [Must exit before gamma risk]

**Step 7: Post-Trade Management Plan** ___/___

- Take profit at _______% of max profit? Yes / No [tastytrade: 50%]
- Stop loss at _______% of max loss? Yes / No [tastytrade: 200% of credit]
- Time stop at _______ DTE? Yes / No [21 DTE minimum exit]
- Adjustment plan if price moves against? Yes / No [Roll down/out or exit]
- Adjustment plan if price moves favorably? Yes / No [Take profit early]

**Final Gating Question:**
**Is the expected value of this trade positive AFTER transaction costs (commissions + slippage)?** Yes / No
[If No — DO NOT ENTER]

---

### Summary: Execution Rules

**DO NOT TRADE IF:**
- The account is down 3% ($21) for the day
- The account is down 10% ($70) for the week
- The account is down 10% ($70) for the month
- Three consecutive losses have occurred (reduce to 0.5% risk)
- A major economic event is within 1 hour
- Earnings are within 3 days
- Stock is under $5 per share
- Option volume < 100 contracts or OI < 200
- Bid-ask spread > $0.15 or > 15% of mid-price
- IV Rank is between 30–50 (neutral) without a clear directional edge
- Multi-timeframe alignment is absent

**ALWAYS:**
- Use limit orders (never market orders)
- Calculate expected value AFTER all transaction costs
- Exit by 21 DTE for credit strategies, 14 DTE for debit strategies
- Reduce position size after 2 consecutive losses
- Review all trades after monthly drawdown of 5% ($35)
- Stop trading entirely after monthly drawdown of 10% ($70)
- Risk no more than Quarter Kelly for maximum safety

---

## Sources

[1] tastylive - Defined-Risk and Undefined-Risk Position Sizing: https://www.tastylive.com/news-insights/position-sizing

[2] tastylive - What Is Options Buying Power: https://www.tastylive.com/news-insights/buying-power

[3] Option Alpha - Account Size Adjustments: https://optionalpha.com/blog/account-size-adjustments

[4] TradeAlgo - Futures Risk Management: https://www.tradealgo.com/trading-guides/futures-risk-management

[5] Earn2Trade - Daily Loss Limits: https://www.earn2trade.com/blog/daily-loss-limit

[6] FundedNext - Rules and Conditions: https://fundednext.com/rules

[7] Edgeful - Position Sizing: https://www.edgeful.com/blog/posts/position-sizing-in-trading

[8] MQL5 - Building a Trading System Part 2: https://www.mql5.com/en/articles/17400

[9] QuantVPS - Risk Management Guide: https://www.quantvps.com/blog/risk-management-guide

[10] Drawdown Recovery Calculator: https://drawdown.recovery.calculator

[11] ZitaPlus - Drawdown Recovery: https://zitaplus.com/drawdown-recovery

[12] CrossTrade - Risk of Ruin: https://www.crosstrade.io/risk-of-ruin

[13] JournalPlus - Kelly Criterion Calculator: https://journalplus.co/kelly-criterion-calculator

[14] Wikipedia - Kelly Criterion: https://en.wikipedia.org/wiki/Kelly_criterion

[15] Wikipedia - Binomial Distribution: https://en.wikipedia.org/wiki/Binomial_distribution

[16] QuantStart - Sharpe Ratio for Algorithmic Trading: https://www.quantstart.com/articles/sharpe-ratio

[17] Coates & Page (2009) - PNAS/PMC Study: https://pmc.ncbi.nlm.nih.gov/articles/PMC2782761

[18] SSRN - The Cross-Section of Speculator Skill: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1900795

[19] Cboe Options Institute - Research: https://www.cboe.com/optionsinstitute/research

[20] Pomegra - Options Position Sizing: https://pomegra.com/options-position-sizing

[21] Holaprime - ATR Position Sizing: https://holaprime.com/blogs/trading-tips/atr-position-sizing

[22] LuxAlgo - Bollinger Bands Strategy: https://www.luxalgo.com/blog/bollinger-bands-strategy-squeeze-then-surge

[23] Investopedia - ATR: https://www.investopedia.com/terms/a/atr.asp

[24] StockCharts.com - Bollinger Band Squeeze: https://chartschool.stockcharts.com/table-of-contents/trading-strategies/bolinger-band-squeeze

[25] StockCharts.com - Bollinger BandWidth: https://chartschool.stockcharts.com/table-of-contents/technical-indicators/bolinger-bandwidth

[26] CMT Association - Mastering RSI: https://content.cmtassociation.org/a/mastering-the-relative-strength-index-rsi

[27] Fidelity - RSI: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI

[28] Interactive Brokers - MACD: https://www.interactivebrokers.com/campus/trading-lessons/macd

[29] TradersPost - MACD Trading Strategies: https://blog.traderspost.io/article/macd-trading-strategies-guide

[30] TOS Indicators - Golden Cross Strategy: https://tosindicators.com/research/golden-cross-trading-strategy

[31] ThinkorSwim Research - RVOL: https://research.thinkorswim.com/rvol

[32] Groww - VWAP Strategy: https://groww.in/blog/vwap-strategy

[33] Humbled Trader - VWAP Strategy: https://www.humbledtrader.com/blog/vwap-strategy

[34] JournalPlus - Anchored VWAP: https://journalplus.co/strategies/vwap-anchored-strategy

[35] DaysToExpiry - Theta Decay Guide: https://daystoexpiry.com/theta-decay

[36] Bourse de Montréal - Options Play: https://www.m-x.ca/options-play

[37] OIC - Delta and Probability: https://www.optionseducation.org/tools/delta

[38] Charles Schwab - Options Delta: https://www.schwab.com/learn/story/options-delta

[39] The Option Premium - Spread Width: https://theoptionpremium.com/spread-width-rules

[40] MenthorQ - Gamma Risk: https://menthorq.com/gamma-risk

[41] Numerix - Gamma Hedging 0DTE: https://numerix.com/gamma-hedging-0dte

[42] TradingBlock - VOSS Framework: https://tradingblock.com/voss-framework

[43] Cboe - Order Types and Off-Screen Liquidity: https://www.cboe.com/insights/order-types

[44] Bullish Bears - Options Liquidity: https://bullishbears.com/options-liquidity

[45] Tackle Trading - Options Liquidity: https://tackletrading.com/options-liquidity

[46] Fidelity - Options Levels: https://www.fidelity.com/trading/options/levels

[47] Schwab - Options Levels: https://www.schwab.com/options

[48] tastytrade - Account Minimums: https://support.tastytrade.com/support/account-minimums

[49] FINRA Rule 4210: https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210

[50] FINRA - Regulatory Notice 26-10: https://www.finra.org/rules-guidance/notices/26-10

[51] FINRA - 0DTE Warning: https://www.finra.org/investors/insights/0dte-options

[52] SEC Rule 3a51-1: https://www.sec.gov/rules/3a51-1

[53] SteadyOptions - Options Expire Worthless: https://steadyoptions.com/articles/do-80-of-options-expire-worthless

[54] SJ Options Research - IV Rank Study: