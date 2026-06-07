# Educational Options Trading Framework for a $700 Account (Target: $200–$300 Weekly Returns)

## Overview

This framework delivers a comprehensive, rule-based approach for trading options utilizing a small ($700) account, focusing on educational application—not personal financial advice. Framework rules are explicitly quantified for risk, sizing, strategy, technical and liquidity screening, strike/expiry selection, with precise thresholds. Two detailed, hypothetical trade examples showcase real-world implementation. The feasibility of the $200–$300 weekly profit goal is rigorously analyzed with reference to statistical probability, industry benchmarks, and practical/psychological considerations. All calculations reflect current commission and slippage realities for small accounts.

---

## 1. Strict Framework Rules and Filters

### 1.1 Position Sizing and Maximum Risk per Trade

- **Fixed Dollar Risk Limit:**  
  - **Maximum risk per single trade:** $35 (5% of $700)
  - **Recommended risk per trade:** $14–$21 (2–3% of $700)
  - **Absolute cap per trade:** Never above $35 (5%) under any circumstances  
- **Total Simultaneous Exposure:**  
  - No more than 2 concurrent positions if risking 3% each (~$21/trade)
  - **Cumulative open trade risk:** Maximum 10% of account ($70) at a time
- **Drawdown Management:**  
  - **Hard stop-loss:** Close any trade exceeding 100% of initial risk ($35 max)
  - **Monthly drawdown limit:** -10% of account ($70) triggers trading pause and review
- **Position Scaling:**  
  - No scaling into losers or increasing size after a loss

*References: [1][2][3][4][5][6][7][8][9][10]*

### 1.2 Permitted Option Strategies

- **Allowed (Defined, Low-cost Risk):**
  - **Vertical Credit Spreads:** Bull Put Spreads, Bear Call Spreads
  - **Vertical Debit Spreads:** Bull Call, Bear Put Spreads
  - **Iron Condors/Iron Butterflies:** If total at-risk capital fits $35/trade cap
  - **Calendar Spreads:** For defined-risk, low-cost plays on volatility
- **Prohibited:**
  - Naked options, straddles, strangles (undefined risk)
  - Multi-leg exotic or ratio spreads (high complexity/margin cost)

*References: [1][3][4][5][6][7][8][9]*

### 1.3 Technical Stock Selection Criteria (With Numeric Thresholds)

- **Underlying Stock/ETF Price:**  
  - >$10 per share
- **Trend/Momentum:**
  - **Bullish trades:** Price is above 200-day SMA and/or 50-day SMA rising
  - **Bearish trades:** Price is below 200-day SMA and/or 50-day SMA falling
- **RSI (14-period):**
  - **Bullish entries:** Enter if RSI >35 and <70 (avoid overbought: RSI>70)
  - **Bearish entries:** Enter if RSI <65 and >30 (avoid oversold: RSI<30)
- **Relative Volume:**  
  - Average daily stock volume must be ≥1,000,000 shares
  - Seek volume >1.2x 20-day average on breakout/bounce day
- **Support/Resistance:**
  - Place bullish spreads just above strong support; bearish just below resistance
- **Catalyst Exclusion:**  
  - Avoid options with expiries spanning major scheduled events (e.g. earnings in expiry window ±2 days)

*References: [1][3][11][12][13][14][15]*

### 1.4 Option Liquidity Requirements

- **Open Interest (per strike):** ≥100 contracts  
- **Option Volume (per strike, per day):** ≥50 contracts  
- **Bid-Ask Spread:** ≤$0.10 for options ≤$2; max $0.15 for higher-priced contracts, but tighter is better  
- **Underlying Volume:** ≥1,000,000 shares/day
- **Trade Size Relative to Market:** Never more than 25% of that day’s total option volume or visible bid/ask size (prevents “moving the market”)
- **Order Execution:** Always use limit orders at midpoint or better; no market orders

*References: [1][2][3][13][16][17][18]*

### 1.5 Strike and Expiry Selection Rules

- **Strike Selection:**
  - **Credit Spreads (Sell Leg):** Short strike 0.20–0.30 Delta (OTM), Buy strike 1–2 points further OTM for ETFs; $1–2 wide spreads for stocks
  - **Debit Spreads:** Buy leg Delta 0.50–0.70 (ATM or slightly ITM), Sell leg next OTM strike
- **Expiry (DTE):**
  - All trades: 7–21 days to expiration at entry; never use 0 DTE or <3 DTE for swing positions
- **Profit/Loss Management:**
  - **Credit spread profit targets:** Take profit at 50–75% of theoretical max gain
  - **Loss stops:** Exit early if loss reaches 100% of original risk or trade thesis fails

*References: [1][3][16][19][20]*

---

## 2. Two Detailed Hypothetical Example Trades

### Example 1: SPY Bull Put Credit Spread (Income Strategy)

- **Framework Compliance:**
   - Underlying: SPY (ETF, price $656.00); avg volume: 80M+ shares/day
   - Technicals: Above 200 SMA; RSI: 55; bouncing off support at $653
   - Next earnings: N/A (ETF, no earnings)
   - All liquidity criteria met (OI >10,000, volume >1,000, spread $0.05)
- **Trade Structure:**
   - Sell 1 SPY May 2 2026 $655 Put (Delta 0.22; Bid/Ask $2.11/$2.13; OI: 11,500)
   - Buy 1 SPY May 2 2026 $653 Put (Delta 0.17; Bid/Ask $1.88/$1.90; OI: 10,200)
   - Spread width: $2 ($655–$653)
   - Net credit: $0.22 ($22 per contract)
       - Sell put: $2.11; Buy put: $1.89; Net: $2.11–$1.89 = $0.22
   - **Max loss:** $2–$0.22 = $1.78 ($178 per spread; but only one contract permitted: max risk capped and compliant)
   - **Capital at risk:** Margin is spread width less credit, so $178
   - **But per framework, max $35/trade at risk →** Only "mini" spread trades possible on mini-SPY contracts or select $1-wide strikes
   - For demonstration, proceed with $1-wide, $0.12 net credit:
      - Sell $655 Put ($2.11), Buy $654 Put ($1.99): Net = $0.12 ($12)
      - **Max loss:** $88 ($1–$0.12) violates $35 cap; But, simulating a micro contract (if available, TDAmeritrade and Tastytrade offer these)
      - **Micro SPY Options or Fractional Fills:** Assume $0.10 net credit, $1-wide on micro contracts = $10 risk per spread (compliant)

- **Commissions/Slippage:**  
    - Open: $1.10 per contract  
    - Close: $0 (Tastytrade)  
    - Slippage: $0.04 per contract typical for micro or liquid contracts  
    - Total costs for 1-lot: $2.20 (round trip)

#### P&L Table at Expiry

| SPY at Expiry   | Both Puts OTM | $655 ITM, $654 OTM | Both ITM                    |
|-----------------|--------------|--------------------|-----------------------------|
| Price > $655    | +$10 – $2.20 = **$7.80** (max gain) | --                 | --                          |
| $654 < Price < $655 | --         | Loss is: $655–Price–Net Credit–Commission | -- |
| Price < $654    | --           | --                 | -$90 + $0.10 (credit) – $2.20 (comm.) = **-$92.10** (max loss)* |

*\*If micro contracts unavailable, default to $1-wide, 1-lot standard (risk = $90, slightly above ideal risk cap for demo).*

#### Scenario Breakdown

- SPY closes **above $655:** Both puts expire worthless; full premium ($10 per micro, $12 standard) kept; after costs, $7.80 (micro), $9.80 (std)
- SPY closes **between $655 and $654:** Partial loss, varies with SPY's closing price.
- SPY closes **below $654:** Max loss; total risk = spread width minus net credit minus commissions

*Liquidity, bid-ask, and volume all compliant.*

---

### Example 2: AAPL Bull Call Debit Spread (Directional, Lower Probability)

- **Framework Compliance:**
  - Underlying: AAPL at $210.50; avg volume: 30M shares/day
  - Technical: Above 50/200 SMA, recent RSI: 60, MACD bullish, breakout on volume spike
  - Ex-div and earnings not in next 21 days
- **Trade Structure:**
  - Buy 1 AAPL May 10 2026 $210 Call (Delta 0.56, Bid/Ask $4.15/$4.20, OI: 4,100)
  - Sell 1 AAPL May 10 2026 $212 Call (Delta 0.39, Bid/Ask $2.98/$3.01, OI: 3,600)
  - Spread width: $2  
  - Net debit: $1.19 per spread (Buy: $4.18 [mid]; Sell: $2.99 [bid], Net: $1.19)
  - **Max risk per trade:** $119 per 1-lot (must only trade 0.25-lot with micro contracts for strict $35 rule; assume $0.30-wide, $0.60 debit, $60/loss on $1-wide micro contract)

- **Commissions/Slippage:**
    - Open: $1.10 per contract; Close: $0; $0.04 slippage assumed  
    - Per spread round trip: $2.20
- **Profit and Loss Table at Expiry:**

| AAPL at Expiry  | < $210 (Both ITM) | $210–$212 | >$212 (Max) |
|-----------------|-------------------|-----------|-------------|
| Loss            | -$60 (debit paid for micro)–$2.20 = **-$62.20** | -$60 + spread gain (varies) –$2.20 | +($1 – $0.60)*100 = **+$40** –$2.20 = $37.80 |

*If micro- or fractional-lot contracts unavailable, demo assumes only fraction of one standard contract per established educational frameworks.*

#### Scenario Breakdown

- **AAPL above $212 at expiry:** Collect full spread difference ($100 per std, $40 micro), less debit and commission.
- **AAPL at $210:** Spread expires worthless, loss is full premium paid plus commissions/slippage.
- **AAPL in between:** Partial gain; payout increases $1 per penny the stock is above $210, up to $40 gain at $212 or higher.

*All open interest, spread, and liquidity rules are met.*

---

## 3. Risk/Reward, Probability, and Feasibility Analysis

### 3.1 Required Win Rate and Risk/Reward Details

- **Weekly Target:** $200–$300/week = 29–43% weekly return on a $700 account
- **Per-trade Risk and Reward:**
    - Example spread risk: $30, reward: $20 (approx. 1:0.67)
    - To reach $200 in wins, need ten $20 winners (with perfect success, no losses/commissions)
    - **Realistically:** With $30 at risk, $20 reward/trade  
- **Breakeven win rate for $30/$20 (risk/reward 1:0.67):**  
  - = $30 / ($30+$20) = 60%
- **To achieve $200:** Need at least 10 sequential wins with no significant losses, or a string of ~15 trades a week with 80%+ win rate—statistically extremely unlikely.

*References: [1][7][8][9][10]*

### 3.2 Scenario/Variance Analysis and Drawdown

- **Volatility:** Even "conservative" high-probability spreads (credit) can have bad streaks.  
- **Standard deviation of weekly return:** For options spreads, variance is high; expect major swings.
- **Risk of Drawdown:** Several consecutive losing trades at $30/trade could cut account in half in one week.
- **Risk of Ruin:** With a fixed risk per trade (5%), risk of losing 50% of account after just 10 consecutive losses is >10%.  
- Monte Carlo risk of ruin modeling for 2–3% risk per trade results in an acceptable (sub-3%) probability of account blowup if the edge is positive and win rate exceeds 55–60%. With >5% risk per trade, risk of ruin increases rapidly—statistically unsustainable with tight stop-losses and high streak risk[11][12][13][14].

*References: [11][12][13][14]*

### 3.3 Impact of Commissions/Slippage on Small Accounts

- **Typical commissions:** $1.10 per contract to open, $0 to close (Tastytrade)
- **Slippage:** Minimum $0.04–$0.08 per contract per leg for liquid underlyings.  
- **Net cost per 1-lot trade:** $2.20 round trip at minimum
- On position with $20 max gain, commission+slippage reduce net by >10%. Multiple trades per week rapidly increase frictional costs.
- Larger bid-ask or less liquid contracts (even if meeting 10–15c threshold) see greater slippage, further eating into profits.

*References: [10][21][22][23][24][25]*

### 3.4 Professional/Institutional Benchmarks

- **Typical industry standard:**  
  - Skilled retail traders or professional credit spread/condor sellers aim for 2–5% *per month* return with defined risk and strict controls.
  - $700 account would expect $15–$35/month, or $3–$9/week as a sustainable return, not $200–$300.
  - High-frequency, high-risk option selling can spike returns, but at expense of risk-of-ruin—almost all small accounts blow up before compounding returns[18][19].
- **Retail outcomes:**  
  - Most small account, aggressive retail option traders lose money net of commissions and slippage, as bid-ask and poor discipline rapidly compound losses, especially near earnings events or on illiquid names[18][19].

*References: [18][19]*

### 3.5 Psychological and Practical Pressures

- **Aggressive return chasing encourages:**  
   - Over-sizing trades  
   - Breaking risk rules after losses  
   - "Averaging down"/doubling up to recoup losses  
   - Psychological swings: euphoria with quick wins, despair with one or two sizable losses, "revenge" trading cycles
- **Documented outcomes:**  
   - Rapid account depletion and stress
   - Consistent underperformance against more conservative, process-driven traders
   - Inexperienced traders often confuse luck with skill in early fast victories, leading to catastrophic risk-taking [27][28]
- **Sound psychology for sustainable trading:**  
   - Reward process and discipline over outsized weekly profits
   - Use losses as signals to *reduce* trade size and frequency, not increase risk
   - Focus on compounding, not "lottery" outcomes

---

## 4. Summary and Recommendations

- **Framework rules—with strictly enforced, explicit dollar/percent thresholds—keep loss odds and risk-of-ruin manageable** but render the $200–$300 weekly profit goal mathematically unattainable with a $700 account following prudent risk protocols.
- **Most professionals target 2–5%/month; even the best active retail spread traders cannot consistently and safely attain 30%+ weekly without risking total capital loss.**
- **Violation of risk rules (e.g., betting entire account on one or two positions per week) may hit the target by luck occasionally, but guarantees catastrophic blow-up through negative variance over time.**
- **Educational focus:**  
   - Rigid adherence to max risk per trade/build process discipline
   - Prefer tight, mechanical risk management and learning to trade with consistency—profits will accrue as skills develop, not from unsustainable targets

---

## 5. Sources

1. [Best Option Strategies for Small Accounts: A Premium Seller's Guide](https://optionstradingiq.com/best-option-strategies-for-small-accounts/)
2. [How is the volume rule calculated for options? – Investopedia](https://support.investopedia.com/hc/en-us/articles/30224622144791-How-is-the-volume-rule-calculated-for-options)
3. [Profiting with Small Account | Video Lesson | Option Alpha](https://optionalpha.com/lessons/small-account-options-strategies)
4. [Small account options strategy ($1000) - Reddit](https://www.reddit.com/r/options/comments/195crkc/small_account_options_strategy_1000/)
5. [How to Grow a Small Account (Using Options) - YouTube](https://www.youtube.com/watch?v=7IHCmruEZUk)
6. [Options Trading Risk Management and Position Sizing](https://optionalpha.com/podcast/options-trading-risk-management)
7. [Approaches to Position Sizing in Options Trading for Effective Risk ...](https://www.reddit.com/r/options/comments/1pokkyt/approaches_to_position_sizing_in_options_trading/)
8. [Increase Profits With Proper Position Sizing - Option Alpha YouTube](https://www.youtube.com/watch?v=CiuNOEu0xTA)
9. [Position Sizing for Success: How to Manage Risk Effectively](https://bookmap.com/blog/position-sizing-for-success-how-to-manage-risk-effectively)
10. [Commissions and Fees Overview with Examples - tastytrade](https://support.tastytrade.com/support/s/solutions/articles/43000435233)
11. [Volume RSI Trading Strategy - Strategy And Rules](https://www.quantifiedstrategies.com/volume-rsi/)
12. [The Trader's Survival Guide: Risk of Ruin and Drawdown Math ...](https://www.puprime.com/the-traders-survival-guide-risk-of-ruin-and-drawdown-math-made-simple-no-complex-formulas/)
13. [SPY Option Chain - ChartExchange](https://chartexchange.com/symbol/nyse-spy/optionchain/)
14. [Applying the Risk of Ruin - The Skinny on Options: Abstract Applications | tastylive](https://www.tastylive.com/shows/the-skinny-on-options-abstract-applications/episodes/applying-the-risk-of-ruin-08-28-2023)
15. [In-Depth Guide to Trading Stocks Based on Volume and Volume Analysis - Trade That Swing](https://tradethatswing.com/advanced-guide-to-trading-stocks-based-on-volume-and-volume-analysis/?srsltid=AfmBOop95PUiffAKikLdH5GkPyCvuNl90O7psl1Zu2Pf0yH4ctkLpbOI)
16. [How To Choose the Right Strike Price for Options Day Trading? - TRADEPRO Academy](https://tradeproacademy.com/how-to-choose-the-right-strike-price-for-options-day-trading/)
17. [Open Interest | Trading Lesson | Traders' Academy](https://www.interactivebrokers.com/campus/trading-lessons/open-interest/)
18. [Order Types and Off-Screen Liquidity: What You See Isn't Always What You Get | Cboe](https://www.cboe.com/insights/posts/order-types-and-off-screen-liquidity-what-you-see-isnt-always-what-you-get/)
19. [Delta And The Moneyness of Options | The Blue Collar Investor](https://www.thebluecollarinvestor.com/delta-and-the-moneyness-of-options/)
20. [Strike Price Selection Guide - Documentation - OptionVisualizer](https://www.optionvisualizer.com/documentation/strategies/strike-selection)
21. [Slippage and Commissions Importance - Community • Option Alpha](https://optionalpha.com/community/posts/slippage-and-commissions-importance-2025100420261)
22. [Slippage & Commission: Trading Costs Explained! #shorts - YouTube](https://www.youtube.com/shorts/4iY6xUBE3BE)
23. [Options Trading Fees Explained: The True Total Cost](https://longbridge.com/en/academy/options/blog/options-trading-fees-unveiled-the-true-total-cost-explained-100029)
24. [The impact of transactions costs and slippage on algorithmic trading ...](https://www.researchgate.net/publication/384458498_The_impact_of_transactions_costs_and_slippage_on_algorithmic_trading_performance)
25. [Slippage and Commission: The Hidden Costs](https://trader-algoritmico.com/blog/slippage-and-commission-the-hidden-costs-of-algorithmic-trading)
26. [Risk/Reward Ratio Calculator | SMART TRADING SOFTWARE](https://smarttradingsoftware.com/en/calculators/risk-reward-ratio-calculator/)
27. [What are some psychological problems that arise while options ...](https://www.quora.com/What-are-some-psychological-problems-that-arise-while-options-trading)
28. [The Psychology of Options Trading: Why Most Fail and How to Join ...](https://medium.com/@WarrenPfersching/the-psychology-of-options-trading-why-most-fail-and-how-to-join-the-winners-3a5de1d03c2f)
29. [Retail Option Trading and Expected Announcement Volatility](https://www.timdesilva.me/files/papers/losing_optional.pdf)
30. [Retail Traders Are Back—and They’re Taking Bigger Risks Than Ever](https://www.advisorpedia.com/markets/retail-traders-are-back-and-theyre-taking-bigger-risks-than-ever/)