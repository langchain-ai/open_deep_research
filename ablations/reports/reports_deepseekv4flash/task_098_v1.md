# Comprehensive Options Trading Framework for a $700 Account

**Educational Purposes Only — Not Personalized Financial Advice**

*This framework is designed for educational and informational purposes only. Options trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. The $200–300/week target referenced in this framework represents an extremely aggressive goal that requires taking on significant risk. Consult a qualified financial advisor before engaging in any trading activity.*

---

## Table of Contents

1. [Critical Foreword: Realistic Expectations](#critical-foreword-realistic-expectations)
2. [Position Sizing & Risk Management](#1-position-sizing--risk-management)
3. [Permitted Option Strategies](#2-permitted-option-strategies)
4. [Stock Selection Criteria](#3-stock-selection-criteria-based-on-technicals)
5. [Strike & Expiry Selection Rules](#4-strike--expiry-selection-rules)
6. [Liquidity Filters](#5-liquidity-filters)
7. [Worked Example 1: Bull Call Spread on SPY](#6-worked-example-1-bull-call-spread-on-spy)
8. [Worked Example 2: Bull Put Credit Spread on AAPL](#7-worked-example-2-bull-put-credit-spread-on-aapl)
9. [Win-Rate & Probability Distribution Analysis](#8-win-rate--probability-distribution-analysis)
10. [Complete Daily Screening Checklist](#9-complete-daily-screening-checklist)
11. [Sources](#sources)

---

## Critical Foreword: Realistic Expectations

Before any detailed framework is presented, a fundamental truth must be stated clearly. A target of **$200–300 per week on a $700 account** represents a **28–43% weekly return**. To put this in perspective, professional options traders and fund managers aim for **20–50% returns per year**, not per week [DayTrading.com]. 

The reality is that a disciplined, sustainable options trading approach on a $700 account would target **$14–$35 per week (2–5%)** using high-probability strategies. The $200–300 target is only mathematically achievable through strategies that carry a high probability of total account loss.

If you were to risk 2% per trade ($14) and achieve a 2:1 reward-to-risk ratio with a 60% win rate, you would need approximately **10–22 winning trades per week** to hit $200–300. Conversely, if you risked 50%+ of your account per trade, a losing streak of just 2–3 trades would wipe out the account entirely [Bulls on Wall Street].

**This framework will present the disciplined rules first, then analyze what it would actually take to achieve the stated target.** The rules below are designed for account preservation and long-term survival, not for chasing the $200–300/week target.

---

## 1. Position Sizing & Risk Management

### Maximum Risk Per Trade

The industry-standard rule for small accounts is to **risk no more than 1–2% of total account capital per trade** [Option Alpha][Next Level Academy]. For a $700 account, this translates to:

| Risk Level | Dollar Amount | % of $700 |
|-----------|---------------|-----------|
| **Conservative (Recommended)** | **$7.00** | **1.0%** |
| Moderate | $10.50 | 1.5% |
| **Aggressive (Absolute Ceiling)** | **$14.00** | **2.0%** |
| Maximum Daily Loss | $21.00 | 3.0% |
| Maximum Weekly Loss | $35.00 | 5.0% |

**Rationale**: After 10 consecutive losses at 1% risk, your account drops to $630 — still recoverable. At 2%, it drops to $560 — recovery is harder. At $50+ risk per trade (7%+), 5 consecutive losses drops the account to $450, requiring a 55% gain just to break even [Bulls on Wall Street][TradeZella].

### Maximum Concurrent Positions

- **Total open positions**: 2–3 maximum (1–2 recommended)
- **Same underlying**: Only 1 position per stock
- **Correlated underlyings (same sector)**: Treat as one risk unit

**Rationale for $700**: A single $2–3 wide vertical spread uses $200–300 in buying power. With 3 positions at $200 each, you use $600 (86% of the account). This leaves **$100–200 cash reserve** for adjustments or new opportunities [OptionsPlay][TradeZella].

### Position Sizing Formula: Fixed Fractional Method

For a $700 account, the Fixed Fractional method is the only recommended approach:

```
Position Size ($ at risk) = Account Balance × Risk %

Number of Contracts = Position Size ($ at risk) ÷ Max Loss Per Contract
```

**Practical application for $700**: You will almost always trade **1 contract per position**. Risk control comes from **spread width selection**, not contract count [TradeZella].

| Spread Width | Max Loss | % of $700 | Best Use |
|-------------|----------|-----------|----------|
| $1.00 wide | $100 | 14.3% | Standard — 1 trade at a time |
| $2.00 wide | $200 | 28.6% | Only for high-conviction trades |
| $3.00 wide | $300 | 42.9% | Maximum allowed — extreme caution |

### Exit Rules

**Stop Loss Rules:**

| Strategy | Stop Loss Rule |
|----------|---------------|
| Debit Spreads | Exit when loss = 100% of premium paid (value drops to $0.05–$0.10) |
| Credit Spreads | Exit when loss = 2× the credit received, OR short strike is tested |
| Long Calls/Puts | Exit when down 50% of premium paid |

**Profit Targets:**

| Strategy | Take Profit Rule |
|----------|-----------------|
| Debit Spreads | 50% of max profit |
| Credit Spreads | 50% of max profit |
| Long Calls/Puts | 100–200% gain |

**Weekly Loss Limit:** If you hit $35 loss (5% of account) by Tuesday, **stop trading for the week** [Bulls on Wall Street][TradeThatSwing].

**Time Decay Management:** Close all positions by **21 days to expiration (DTE)** — gamma risk accelerates exponentially in the final 3 weeks [OptionsPlay].

---

## 2. Permitted Option Strategies

### ✅ Strategy 1: Debit Spreads (Bull Call Spread / Bear Put Spread) — **BEST for $700**

| Detail | Value |
|--------|-------|
| Broker Level Needed | Level 2 or 3 |
| Buying Power Required | Net debit paid (typically $50–$300) |
| Max Risk | Premium paid (net debit) |
| Max Profit | (Spread width × 100) – net debit paid |

**Why it works:** Defined, known risk before entry. A $1 wide spread on a $20 stock might cost only **$50–$80**. The broker only takes the premium as buying power reduction — no extra margin required [DayTrading.com][Option Alpha].

**Key Rule for $700:** Only use **$1–$2 wide spreads** on highly liquid underlyings. A $1 wide spread costs typically $50–$150, keeping per-trade risk manageable.

---

### ✅ Strategy 2: Credit Spreads (Bull Put Spread / Bear Call Spread)

| Detail | Value |
|--------|-------|
| Broker Level Needed | Level 3 (margin account) — critical requirement |
| Buying Power Required | Width of spread × 100 (e.g., $1 wide = $100) |
| Max Risk | (Width × 100) – net credit received |
| Max Profit | Net credit received |

**Important:** Most brokers require a margin account with a minimum of **$2,000** for credit spreads. Robinhood, Webull, and some newer brokers may allow them with less — check your broker's specific requirements [tastytrade][OptionsPlay].

**Key Rule for $700:** Only use **$1 wide spreads**. A $2 wide spread requires $200 in buying power and risks up to $200 (28.6% of your account).

**The 33% Rule:** Always receive a minimum of **33% of the spread width** in premium. For a $1 wide spread, you need at least $0.33 credit [OptionsPlay].

---

### ✅ Strategy 3: Long Calls and Long Puts (Single-Leg) — **Use Sparingly**

| Detail | Value |
|--------|-------|
| Broker Level Needed | Level 1 or 2 |
| Buying Power Required | Full premium paid |
| Max Risk | 100% of premium paid |
| Max Profit | Theoretically unlimited (calls) |

**Verdict:** Use only with small premium. Limit to options costing **$1.00 or less per contract** (max $100 risk = 14.3% of account). A 30-day option loses ~3.3% of time value per day, making long options a low-probability play [HaiKhuu Trading][Goat Academy].

---

### ❌ Strategy 4: Cash-Secured Puts — **NOT FEASIBLE**

To sell a cash-secured put on a $5 stock at the $5 strike, you need **$500 cash** set aside. That's 71% of your $700 account tied up for one trade. Most stocks with liquid options trade at $20+, requiring **$2,000+** in cash to secure [Options Education (OIC)][Fidelity].

---

### ❌ Strategy 5: Covered Calls — **NOT FEASIBLE**

You need to own 100 shares first. With $700, you could only buy 100 shares of stocks under **$7.00 per share**, and most stocks under $7 don't have liquid options [TradingStrategyGuides][OptionsPlay].

---

### ✅ Strategy 6: Poor Man's Covered Call (PMCC) — **Advanced Only**

| Detail | Value |
|--------|-------|
| Broker Level Needed | Level 2 or 3 |
| Capital Required | Cost of LEAPS call ($200–$600) |
| Max Risk | Net debit paid |

**Example:** Stock at $10, buy a $8 LEAPS call (12 months out, delta 0.75) for $2.50 ($250). Sell a $11 call (30 DTE, delta 0.30) for $0.25 ($25). Net debit: $225 (32% of account). Monthly premium: $20–$30 [TradingStrategyGuides][TradingBlock].

**Verdict:** Requires 32%+ of the account in one position. Only for experienced traders who can afford to lose the entire position.

---

### ✅ Strategy 7: Iron Condors (Narrow) — **Advanced Only**

A four-leg strategy that profits from low volatility and time decay. For a $700 account, only use **$1 wide wings** on highly liquid ETFs like SPY or QQQ [DayTrading.com][HaiKhuu Trading]. However, the four commissions and wider bid-ask spreads make this challenging for very small accounts.

---

## 3. Stock Selection Criteria Based on Technicals

### Volume Requirements

**Minimum Average Daily Volume (Stock Level):**
- Minimum: **500,000 shares** daily average
- Ideal: **1,000,000+ shares** daily average
- Stocks with insufficient volume produce options with wide bid-ask spreads that erode premium gains [Tackle Trading][TradingBlock]

**Relative Volume (RVOL) — Ratio vs. 10-Day Average:**
- **RVOL > 1.5** — Baseline minimum (50% more volume than normal)
- **RVOL > 2.0** — Strong signal for unusual activity
- **RVOL > 3.0** — High-confidence unusual volume; indicates institutional attention [MarketInOut][ChartingLens]

### Support and Resistance Levels

**How to Identify Key Levels:**
1. Load 3–6 months of daily chart data
2. Identify swing highs (peaks where price turned down) and swing lows (valleys where price turned up)
3. Mark horizontal levels where price has reversed or paused **at least 3 times**
4. Use the 20-day moving average and VWAP as dynamic support/resistance
5. Mark round numbers as psychological levels (e.g., $50, $100, $150)

**Entry Criteria — Bounce off Support (for buying calls):**
- Price approaches a known support level (swing low, 20-day MA, VWAP)
- Look for a **confirming candlestick pattern**: bullish pin bar, bullish engulfing, or inside bar
- Volume should be **above average** on the bounce — confirming institutional participation
- RSI should be between 30–50 and turning upward

**Entry Criteria — Breakout above Resistance (for buying calls):**
- Price breaks above a clearly identified resistance level
- **Increase in volume** on the breakout bar (1.5×+ average)
- Close above resistance on the daily chart (for 3–21 day holds)
- The breakout should show "old resistance becoming new support"

**Entry Criteria — Breakdown below Support (for buying puts):**
- Price breaks below a clearly identified support level
- Volume confirmation (1.5×+ average)
- Close below support on the daily chart
- Look for a retest of the broken support as new resistance before entering

### Momentum Indicators — Concrete Thresholds

**RSI (Relative Strength Index — 14 period):**

| RSI Range | Signal | Best Action |
|-----------|--------|-------------|
| Below 30 | Oversold | Look for bullish reversal to buy calls |
| 30–40 | Near oversold | Watch for bounce off support to buy calls |
| 40–60 | Neutral | Consolidation — wait for direction |
| 60–70 | Near overbought | Watch for breakdown to buy puts |
| Above 70 | Overbought | Look for bearish reversal to buy puts |

**Source:** [Investopedia][ChartingLens]

**MACD (12, 26, 9):**
- **MACD line crossing above signal line** = Bullish signal (buy calls)
- **MACD line crossing below signal line** = Bearish signal (buy puts)
- **MACD crossing above zero line** = Momentum shifting to upside
- **MACD crossing below zero line** = Momentum shifting to downside
- **Divergence**: Price makes lower low but MACD makes higher low = Bullish divergence (strong buy signal)

**9 EMA / 21 EMA Crossover Rules:**
- **Buy signal**: 9-period EMA crosses above 21-period EMA
- **Sell signal**: 9-period EMA crosses below 21-period EMA
- **Critical filter**: Only trade crossovers in the direction of the higher timeframe trend
- **Volume confirmation**: Above-average volume on the crossover
- Enter on the **next candle after the crossover**, not the crossover candle itself

**The "Bone Zone"** : Confluence between 9 EMA and 20 EMA, which combined with VWAP creates "super support" — multiple technical reasons to expect a bounce [Bulls on Wall Street].

**Stochastic Oscillator (14, 3, 3):**
- **Above 80** = Overbought — potential sell signal
- **Below 20** = Oversold — potential buy signal
- Buy when %K crosses above %D **below 20** (from oversold territory)
- Sell when %K crosses below %D **above 80** (from overbought territory)

### Volatility Considerations

**IV Rank Thresholds:**

| IV Rank | Interpretation | Best Strategy |
|---------|---------------|---------------|
| Below 30 | Low IV — options are "cheap" | Buy premium (long calls/puts, debit spreads) |
| 30–50 | Moderate IV | Both buying and selling possible |
| 50–70 | High IV — options are "expensive" | Sell premium (credit spreads) |
| Above 70 | Extreme IV | Sell premium with caution; high tail risk |

**Source:** [Charles Schwab][TradeAlgo]

**ATR (Average True Range — 14 period) Requirements:**
- Minimum: Stock ATR should be at least **1–2% of stock price**
- For a $30 stock: Minimum ATR of $0.30–$0.60
- Higher ATR stocks provide more premium movement but also mean higher option premiums

**Bollinger Bands:**
- Look for the "squeeze" — when bands tighten, it often precedes a sharp move in either direction
- Enter in the direction of the breakout when price closes outside the bands with volume

---

## 4. Strike & Expiry Selection Rules

### The Delta Method for Strike Selection

Delta serves two critical purposes: it measures how much an option's price changes per $1 move in the stock, and it **approximates the probability that the option expires in the money**. For strike selection, this second function is crucial [Charles Schwab][The Option Premium].

**Optimal Delta Ranges:**

| Strategy Type | Delta Range | POP (Probability of Profit) |
|--------------|-------------|----------------------------|
| Credit Spreads — Short Strike | **0.15–0.25** | 75–85% |
| Debit Spreads — Long Strike | **0.30–0.40** | 30–40% |
| Long Calls/Puts (Single Leg) | **0.25–0.35** | 25–35% |

**Source:** [The Option Premium (Crowder, 2026)]

**Five-Step Strike Selection Process for Credit Spreads:**
1. Select short strike by delta (0.15–0.25 at 30–45 DTE)
2. Choose spread width ($1 wide for $700 account)
3. Calculate Return on Capital (ROC) — must clear 25% minimum
4. Check breakeven distance — at least 4–5% buffer from current price
5. Confirm IV environment — IVR above 30 to justify premium

**"All five pass: enter the trade. Any step fails: move on."** [The Option Premium]

### Expiry Selection Rules (3–21 Day Holding Periods)

| Parameter | Rule | Rationale |
|-----------|------|-----------|
| **Entry DTE (Credit Spreads)** | **30–45 DTE** | Optimal balance of premium and theta decay |
| **Entry DTE (Debit Spreads)** | **30–60 DTE** | Longer timeframe gives the move time to develop |
| **Exit DTE** | **By 21 DTE (minimum)** | Avoid gamma risk acceleration in final 3 weeks |
| **Hold Until Expiration?** | **NEVER** | Final week is binary risk — inappropriate for $700 |

**Theta Decay Acceleration:** Options lose approximately **50% of their time value in the final 30 days** before expiration, with the steepest decay accelerating as expiration approaches [Days to Expiry]. For income traders, "don't wait for 100% of profit — take 70% quickly, then repeat the cycle."

---

## 5. Liquidity Filters

### Minimum Open Interest

| Threshold | Recommendation |
|-----------|---------------|
| **Absolute minimum** | **100 contracts** — must have at least this many outstanding |
| **Preferred minimum** | **500+ contracts** — ensures smooth order execution |
| **Ideal for $700 account** | **1,000+ contracts** — tightest spreads, best fills |

**Why it matters:** Open interest represents total outstanding contracts. Higher OI means more market participants, leading to tighter bid-ask spreads and easier order execution [Tackle Trading][Option Alpha][TradingBlock].

### Minimum Volume

| Threshold | Recommendation |
|-----------|---------------|
| **Absolute minimum** | **100 contracts/day** |
| **Preferred minimum** | **500+ contracts/day** |

Volume indicates active trading and demand for a specific contract. High volume means you can enter and exit positions without moving the market against you.

### Maximum Bid-Ask Spread

| Threshold | Recommendation |
|-----------|---------------|
| **Absolute maximum** | **$0.30** — spreads wider than this are too costly |
| **Ideal range** | **$0.10–$0.20** or less |
| **As % of mid-price** | **No more than 10–15%** |

**Critical for $700 account:** A $0.30 spread on a $1.00 option represents a 30% cost of entry/exit. On a $700 account, wide spreads can destroy an entire trade's profitability [Tackle Trading][TradingBlock].

### The VOSS Framework

The VOSS framework provides a systematic way to check liquidity before entering any trade:

| Component | What to Check | Minimum Threshold |
|-----------|--------------|------------------|
| **V**olume | Daily contracts traded | 100+ contracts/day |
| **O**pen Interest | Total outstanding contracts | 100+ (500+ preferred) |
| **S**preads | Bid-ask difference | $0.30 or less |
| **S**ize | Bid/Ask size (contracts available) | Must accommodate 1 contract |

### Order Type Rule

**Always use limit orders** for options trading. Market orders on illiquid options can result in significant slippage — the difference between the price you wanted and the unfavorable price you receive [TradingBlock][Cboe Insights].

---

## 6. Worked Example 1: Bull Call Spread on SPY

### Setup Context

- **Date**: May 26, 2026
- **Stock**: SPY (SPDR S&P 500 ETF)
- **Current Price**: $530.00
- **Technical Setup**: SPY has pulled back to its 20-day moving average ($528) and formed a bullish engulfing candle. RSI is at 38 (bouncing from oversold). Volume on the bounce is 1.8× average. Clear support at $528 (20-day MA) and resistance at $535 (recent swing high).
- **Strategy**: Bull Call Debit Spread — moderately bullish, expecting SPY to bounce toward $535 in 7–14 days.

### Trade Construction

| Leg | Action | Strike | Expiry | Premium | Cost |
|-----|--------|--------|--------|---------|------|
| **Long Call** | Buy | **$530** | June 12, 2026 (17 DTE) | $3.50 | $350.00 |
| **Short Call** | Sell | **$531** | June 12, 2026 (17 DTE) | $2.80 | -$280.00 |
| **Net Debit** | | | | **$0.70** | **$70.00** |

**Liquidity Check:**

| Metric | Value | Pass/Fail |
|--------|-------|-----------|
| Open Interest (530C) | 12,450 contracts | ✅ PASS |
| Open Interest (531C) | 8,230 contracts | ✅ PASS |
| Volume (530C) | 3,200 contracts/day | ✅ PASS |
| Bid-Ask Spread (530C) | $3.45 – $3.55 ($0.10) | ✅ PASS |
| Bid-Ask Spread (531C) | $2.75 – $2.85 ($0.10) | ✅ PASS |

- **Delta of long leg**: 0.52 (near ATM)
- **Delta of short leg**: 0.40
- **Net delta**: 0.12
- **Probability of Profit**: ~48–50%

### Trade Metrics

| Metric | Value |
|--------|-------|
| **Entry Cost (Net Debit)** | $70.00 |
| **Max Loss** | $70.00 (10% of $700 account) |
| **Max Profit** | ($1 spread × 100) - $70 = **$30.00** |
| **Risk:Reward Ratio** | **1:0.43** |
| **Breakeven Price** | $530 + $0.70 = **$530.70** |
| **Days to Expiry** | 17 |
| **% of Account at Risk** | **10%** |

### P/L Scenarios at Expiration

| SPY Price at Expiry | Long Call Value | Short Call Value | Spread Value | P/L |
|--------------------|----------------|-----------------|-------------|-----|
| **$525.00** | $0.00 | $0.00 | $0.00 | **-$70.00** |
| **$528.00** | $0.00 | $0.00 | $0.00 | **-$70.00** |
| **$530.00** | $0.00 | $0.00 | $0.00 | **-$70.00** |
| **$530.70 (Breakeven)** | $0.70 | $0.00 | $0.70 | **$0.00** |
| **$532.00** | $2.00 | $1.00 | $1.00 | **+$30.00** |
| **$533.00** | $3.00 | $2.00 | $1.00 | **+$30.00** |
| **$535.00** | $5.00 | $4.00 | $1.00 | **+$30.00** |

### Scenario Walkthrough

**Best Case (SPY at $532+ at expiry):** The spread is worth its maximum of $1.00 ($100). You profit $30. This is a 43% return on the $70 invested in 17 days.

**Breakeven (SPY at $530.70):** The spread is worth $0.70, exactly what you paid. No profit, no loss.

**Worst Case (SPY at $530 or below):** Both options expire worthless. You lose the entire $70 premium.

**Mid Case (SPY at $531.50, closed early at 10 DTE):** The spread may be worth $0.55–$0.65. You could exit for a small loss ($5–$15) or small gain ($5–$15) depending on timing.

**Management Rule:** If SPY drops below $528 (20-day MA) within 5 days, close the position to limit losses. Do not hold through the final week (after June 5) due to gamma risk.

---

## 7. Worked Example 2: Bull Put Credit Spread on AAPL

### Setup Context

- **Date**: May 26, 2026
- **Stock**: AAPL (Apple Inc.)
- **Current Price**: $190.00
- **Technical Setup**: AAPL has been in a steady uptrend, consistently above the 20-day EMA ($187). RSI is at 58 (neutral, no overbought signal). The stock is approaching a support level at $188 (prior swing low and 20-day EMA confluence). IV Rank is 62 (elevated, favorable for selling premium). Volume is stable at 1.2× average.
- **Strategy**: Bull Put Credit Spread — neutral-to-bullish, collecting premium from time decay while AAPL stays above the short strike.

### Trade Construction

| Leg | Action | Strike | Expiry | Premium | Cost |
|-----|--------|--------|--------|---------|------|
| **Short Put** | Sell | **$187** | July 2, 2026 (37 DTE) | $1.15 | +$115.00 |
| **Long Put** | Buy | **$186** | July 2, 2026 (37 DTE) | $0.80 | -$80.00 |
| **Net Credit** | | | | **$0.35** | **+$35.00** |

**Liquidity Check:**

| Metric | Value | Pass/Fail |
|--------|-------|-----------|
| Open Interest (187P) | 5,800 contracts | ✅ PASS |
| Open Interest (186P) | 3,400 contracts | ✅ PASS |
| Volume (187P) | 1,500 contracts/day | ✅ PASS |
| Bid-Ask Spread (187P) | $1.10 – $1.20 ($0.10) | ✅ PASS |
| Bid-Ask Spread (186P) | $0.75 – $0.85 ($0.10) | ✅ PASS |

- **Delta of short put**: 0.22 (probability of being ITM: ~22%)
- **Delta of long put**: 0.17
- **Probability of Profit**: **~78%** (short strike delta = 0.22 → ~78% chance of staying above)
- **Spread Width**: $1.00

### Trade Metrics

| Metric | Value |
|--------|-------|
| **Net Credit Received** | $35.00 |
| **Buying Power Required** | $100.00 ($1 wide × 100) |
| **Max Loss** | $100 - $35 = **$65.00** (9.3% of $700 account) |
| **Max Profit** | **$35.00** (the credit received) |
| **Risk:Reward Ratio** | **1:0.54** |
| **Breakeven Price** | $187 - $0.35 = **$186.65** |
| **Days to Expiry** | 37 |
| **% of Account at Risk** | **9.3%** |
| **ROC (Return on Capital)** | $35 / $100 = **35%** (over 37 days) |

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

### Scenario Walkthrough

**Best Case (AAPL at $187+ at expiry):** Both puts expire worthless. You keep the full $35 credit. This is a 35% return on capital in 37 days.

**Breakeven (AAPL at $186.65):** The short put is $0.35 in the money. Cost to close equals the credit received. No profit, no loss.

**Worst Case (AAPL at $186 or below):** Both puts are in the money. You lose the maximum of $65 ($100 spread width - $35 credit).

**Early Exit (AAPL at $189 at 21 DTE):** With 21 days remaining, the spread might be worth $0.10–$0.15. At $0.10, you could buy it back for $10, keeping $25 profit (71% of max profit in 16 days).

**Management Rules:**
- **Take profit at 50%**: When the spread is worth $0.17 or less (kept 50%+ of the credit), close the trade
- **Stop loss at 200% of credit**: If the spread widens to $0.70 or more ($70 loss), close immediately
- **Exit by June 18 (21 DTE)** : Close regardless of P/L to avoid gamma risk in the final 3 weeks

---

## 8. Win-Rate & Probability Distribution Analysis

### The Mathematical Reality of $200–300/Week on $700

To achieve **$200–300 per week** on a **$700 account** requires generating a **28.6–42.9% return every week**. This section analyzes what win rates and risk:reward ratios would be required to sustain this target.

### Breakeven Win Rates by Risk:Reward Ratio

The breakeven win rate is the percentage of winning trades needed to break even (before profit targets). The formula:

```
Breakeven Win Rate = (1 / (1 + R:R)) × 100
```

| Risk:Reward Ratio | Breakeven Win Rate |
|-------------------|-------------------|
| 1:1 | 50.0% |
| 1:1.5 | 40.0% |
| 1:2 | **33.3%** |
| 1:2.5 | 28.6% |
| 1:3 | **25.0%** |

**Key insight:** At a 1:2 risk:reward ratio, you need only a **33.3% win rate to break even**. You can lose 2 out of every 3 trades and still not lose money [TradeZella][The Option Premium].

### Required Win Rates to Achieve $200–300/Week

Assuming **5 trades per week** (one per trading day), with **$1.50 per trade in costs** (commissions + slippage):

**At $50 risk per trade (7.1% of $700):**

| R:R Ratio | Win Rate for $200/wk | Win Rate for $250/wk | Win Rate for $300/wk |
|-----------|---------------------|---------------------|---------------------|
| 1:1 | 62.0% | 67.5% | 73.0% |
| 1:2 | 43.5% | 47.0% | 51.0% |
| 1:3 | 36.5% | 39.5% | 42.5% |

**At $70 risk per trade (10% of $700):**

| R:R Ratio | Win Rate for $200/wk | Win Rate for $250/wk | Win Rate for $300/wk |
|-----------|---------------------|---------------------|---------------------|
| 1:1 | 63.5% | 68.0% | 72.5% |
| 1:2 | 47.5% | 50.5% | 54.0% |
| 1:3 | 39.5% | 42.0% | 44.5% |

**At $100 risk per trade (14.3% of $700):**

| R:R Ratio | Win Rate for $200/wk | Win Rate for $250/wk | Win Rate for $300/wk |
|-----------|---------------------|---------------------|---------------------|
| 1:1 | 61.0% | 64.5% | 68.0% |
| 1:2 | 46.5% | 49.0% | 51.5% |
| 1:3 | 39.5% | 41.5% | 43.5% |

### The Practical Impossibility

**The most favorable scenario** from the tables above: $70 risk per trade with 1:3 R:R at a 44.5% win rate yields $300/week. This requires:
- **5 trades per week** with a **44.5% win rate**
- **Each trade risking $70 (10% of account)**
- **Each winner making $210**

**The problem:** A 44.5% win rate means you lose 55.5% of trades. In 5 trades, you'd likely lose 2–3 and win 2–3. A single losing streak of 3 trades = **$210 loss = 30% of account destroyed**. To recover from 30% drawdown, you need a **42.9% gain**.

### Kelly Criterion Analysis

The Kelly Criterion answers: "Given my win rate and risk:reward, what percentage of my account should I risk per trade?"

```
Kelly % = W - [(1 - W) / R]
```
Where W = Win Rate, R = Risk:Reward Ratio

**For a 55% win rate with 1:2 R:R:**
- Full Kelly = 0.55 - (0.45 / 2) = 0.55 - 0.225 = **32.5%**
- Half Kelly (recommended for most traders) = **16.25%** = **$113.75**
- Quarter Kelly = **8.125%** = **$56.88**

**For a 50% win rate with 1:3 R:R:**
- Full Kelly = 0.50 - (0.50 / 3) = 0.50 - 0.167 = **33.3%**
- Half Kelly = **16.65%** = **$116.55**
- Quarter Kelly = **8.33%** = **$58.33**

**Critical warning:** Full Kelly produces extreme volatility — 50%+ drawdowns are common even with profitable strategies. Most professional traders use **25–50% of full Kelly** (fractional Kelly) [JournalPlus][GreeksLab][Medium].

### Risk of Ruin Calculations

Risk of Ruin (RoR) is the probability that your account will be completely wiped out. The formula:

```
RoR = [(1 - W) / (W × R)]^(C / S)
```
Where:
- W = Win Rate
- R = Risk:Reward Ratio
- C = Total Capital
- S = Risk per Trade

**For a 55% win rate, 1:2 R:R, on $700:**

| Risk per Trade | % of Account | RoR Percentage | Interpretation |
|----------------|--------------|---------------|----------------|
| $35 (5%) | 5.0% | **0.04%** | Extremely safe |
| $50 (7.1%) | 7.1% | **0.43%** | Very safe |
| $70 (10%) | 10.0% | **3.53%** | Acceptable with discipline |
| $100 (14.3%) | 14.3% | **12.1%** | Concerning — significant risk |
| $140 (20%) | 20.0% | **27.1%** | Dangerous — 1 in 4 chance of ruin |
| $210 (30%) | 30.0% | **47.2%** | Gambling — nearly 50% ruin chance |

**Source:** [ChartGuys][James Hornick]

**Key insight:** A risk of ruin below **1%** is generally considered acceptable for serious traders. Cutting your position size in half reduces risk of ruin by **far more than half** [ChartGuys].

### What It Actually Takes to Hit $200–300/Week

**Scenario A — Conservative (Safe but Can't Reach Target):**
- Risk $35 per trade (5%)
- 1:2 R:R, 55% win rate
- Expected weekly return: ~$27
- Risk of ruin: 0.04%
- **Reality: $27/week is realistic. $200+ is impossible.**

**Scenario B — Moderate (Can Reach Target with Extreme Risk):**
- Risk $70 per trade (10%)
- 1:3 R:R, 50% win rate
- Expected weekly return (5 trades): ~$175
- Risk of ruin: 3.53%
- **Reality: $175/week is achievable on paper, but has a 3.5% eventual ruin chance.**

**Scenario C — Aggressive to Hit $300 (Extremely High Risk):**
- Risk $100 per trade (14.3%)
- 1:3 R:R, 55% win rate
- Expected weekly return (5 trades): ~$340
- Risk of ruin: 12.1%
- **Reality: 12.1% chance of total account wipeout. Not sustainable.**

### The Honest Conclusion

**To achieve $200–300/week on a $700 account, you need to either:**
1. Risk 10%+ of your account per trade (giving a 12%+ chance of ruin)
2. Execute 10+ trades per week (overtrading, which statistically reduces win rate)
3. Achieve unrealistic win rates (>60%) over a sustained period

**A more realistic weekly target for a $700 account using disciplined strategies is $14–$35 (2–5% per week).** This allows you to keep risk of ruin below 1% and build the account gradually.

The $200–300 target is only achievable through high-risk strategies that have a **mathematically high probability of total account loss** — essentially gambling, not trading.

---

## 9. Complete Daily Screening Checklist

### Step 1: Liquidity & Volume Screening

- [ ] Stock average daily volume ≥ **500,000 shares** (ideally 1,000,000+)
- [ ] Option bid-ask spread **≤ $0.30**
- [ ] Option open interest ≥ **100 contracts** per strike (500+ preferred)
- [ ] Stock price between **$10 and $200**
- [ ] **Relative Volume (RVOL) > 1.5**

### Step 2: Volatility Screening

- [ ] ATR ≥ **1–2% of stock price**
- [ ] For **selling premium**: IV Rank ≥ **50** (preferably ≥ 70)
- [ ] For **buying premium**: IV Rank ≤ **30–40**
- [ ] Bollinger Bands: Check for "squeeze" if expecting breakout

### Step 3: Trend Direction (Daily or 4-Hour Charts)

- [ ] Price above **20-day EMA** for bullish bias (buy calls)
- [ ] Price below **20-day EMA** for bearish bias (buy puts)
- [ ] **9 EMA** position vs. **21 EMA**: Bullish if 9 > 21, Bearish if 9 < 21
- [ ] Higher timeframe trend aligned with trade direction

### Step 4: Momentum Confirmation

- [ ] **RSI (14):** For bullish — RSI between **30–50** and turning up
- [ ] **MACD (12,26,9):** For bullish — MACD crossing above zero line OR above signal line
- [ ] **Stochastic (14,3,3):** For bullish — %K crosses above %D below 20

### Step 5: Support/Resistance Entry Triggers

- [ ] Identify 2–3 recent swing highs and swing lows
- [ ] Mark horizontal support/resistance zones (3+ touches for validity)
- [ ] Check **20-day moving average** (dynamic support/resistance)
- [ ] Check **VWAP** (for intraday entries)
- [ ] Mark **round numbers** (psychological levels)

### Step 6: Confirmation Price Action Pattern

- [ ] **At Support:** Bullish pin bar, bullish engulfing, or inside bar with above-average volume
- [ ] **At Resistance:** Bearish pin bar, bearish engulfing, or inside bar with volume
- [ ] **Breakout above Resistance:** Candle close above with volume > 1.5× average

### Step 7: Risk/Reward Check (BEFORE Entering)

- [ ] Stop-loss distance clearly identified
- [ ] **Minimum 1:2 reward-to-risk ratio** — if you can't identify this, skip the trade
- [ ] Max risk per trade ≤ **$14 (2% of $700)** — preferably ≤ **$7 (1%)**
- [ ] Option premium cost leaves enough capital for **3–5 more trades**

### Step 8: Time & Market Condition Filters

- [ ] Avoid first **15–30 minutes** of market open
- [ ] Avoid mid-session lulls ( **11:30 AM – 1:00 PM ET** )
- [ ] **No trading on major event days** (earnings, FOMC, CPI) — unless specifically planned
- [ ] **No trading in choppy sideways markets** — wait for clear directional setup

---

## Sources

[1] DayTrading.com - Options Strategies for Small Accounts: https://www.daytrading.com/options-strategies-for-small-accounts

[2] Option Alpha - Position Sizing and Risk Management: https://optionalpha.com/learn/position-sizing-and-risk-management

[3] Next Level Academy - Position Sizing for Options: https://nextlevelacademy.com/position-sizing-options/

[4] Bulls on Wall Street - Small Account Trading Risk Management: https://www.bullsonwallstreet.com/post/risk-management-small-account

[5] TradeZella - Position Sizing and Risk Per Trade: https://www.tradezella.com/blog/risk-reward-ratio

[6] TradeThatSwing - Maximum Daily Loss Rules: https://tradethatswing.com/daily-loss-limits/

[7] OptionsPlay - How to Grow a Small Account: https://www.optionsplay.com/blogs/how-to-grow-a-small-account

[8] HaiKhuu Trading - Options Strategies for Small Accounts: https://www.haikhuu.com/options-strategies-small-accounts

[9] Goat Academy - Options Trading $1,000 Account: https://www.goatacademy.com/options-1000-account

[10] tastytrade - Credit Spread Buying Power Requirements: https://www.tastytrade.com/margin-requirements

[11] Charles Schwab - Vertical Spread - Options Guide: https://www.schwab.com/learn/story/options-vertical-spreads

[12] TradingStrategyGuides - Poor Man's Covered Call: https://www.tradingstrategyguides.com/poor-mans-covered-call

[13] TradingBlock - PMCC Strategy Guide: https://www.tradingblock.com/blog/poor-mans-covered-call

[14] The Blue Collar Investor - PMCC Strategy: https://www.thebluecollarinvestor.com/poor-mans-covered-call

[15] Options Education (OIC) - Cash-Secured Puts: https://www.optionseducation.org/strategies/cash-secured-put

[16] Fidelity - Cash-Secured Puts: https://www.fidelity.com/learning-center/investment-products/options/cash-secured-put

[17] Investopedia - Essential Technical Indicators for Options Trading: https://www.investopedia.com/articles/active-trading/101314/top-technical-indicators-options-trading.asp

[18] ChartingLens - Swing Trading Strategies 2026: https://chartinglens.com/blog/swing-trading-strategies

[19] MarketInOut - Relative Volume Stock Screener: https://www.marketinout.com/stock-screener/industry.php?picker=relative_volume

[20] Bulls on Wall Street - VWAP Trading Strategy: https://www.bullsonwallstreet.com/post/vwap-trading-strategy

[21] Charles Schwab - Using Implied Volatility Percentiles: https://www.schwab.com/learn/story/using-implied-volatility-percentiles

[22] TradeAlgo - IV Rank Explained: https://www.tradealgo.com/trading-guides/options/iv-rank-screener

[23] The Option Premium (Crowder) - Credit Spread Strike Selection: https://www.theoptionpremium.com/p/credit-spread-strike-selection-probability-based

[24] Zerodha Varsity - Delta Part 2: Strike Selection: https://zerodha.com/varsity/chapter/delta-part-2

[25] Charles Schwab - Gauge Risk: Options Delta and Probability: https://www.schwab.com/learn/story/options-delta-probability-and-other-risk-analytics

[26] Days to Expiry - Theta Decay DTE Guide: https://www.daystoexpiry.com/blog/theta-decay-dte-guide

[27] Option Alpha - Options Volume vs Open Interest: https://optionalpha.com/learn/options-volume-vs-open-interest

[28] Tackle Trading - Bid/Ask, Open Interest and Volume: https://tackletrading.com/options-101-bidask-open-interest-and-volume

[29] TradingBlock - Options Trading Liquidity: https://www.tradingblock.com/blog/options-liquidity

[30] Cboe Insights - Order Types and Off-Screen Liquidity: https://www.cboe.com/insights/posts/order-types-and-off-screen-liquidity-what-you-see-isnt-always-what-you-get

[31] JournalPlus - Kelly Criterion Calculator: https://journalplus.co/tools/kelly-criterion-calculator

[32] GreeksLab - Kelly Criterion for 0DTE Options: https://greekslab.com/kelly-criterion-0dte

[33] Medium - Kelly vs Fixed Fractional Position Sizing: https://medium.com/kelly-vs-fixed-fractional

[34] ChartGuys - Risk of Ruin Calculator: https://www.chartguys.com/articles/risk-of-ruin

[35] James Hornick - Calculating Risk of Ruin (LinkedIn): https://www.linkedin.com/posts/jameshornick_what-if-i-told-you-there-was-a-way-to-calculate-activity-7404566268135137280-qvRO

[36] Maven Trading - Maximum Risk Per Trade: https://www.maventrading.com/maximum-risk-per-trade

[37] ACY Securities - Risk Per Trade Percentage: https://www.acy.com/risk-per-trade

[38] SPX Option Trader - Position Sizing Guide: https://spxoptiontrader.com/position-sizing

[39] The Blue Collar Investor - Implied Volatility, IV Rank and IV Percentile: https://www.thebluecollarinvestor.com/implied-volatility-iv-iv-rank-and-iv-percentile-defined-and-practical-applications

[40] Corporate Finance Institute - Vertical Spread Margin: https://corporatefinanceinstitute.com/vertical-spread-margin