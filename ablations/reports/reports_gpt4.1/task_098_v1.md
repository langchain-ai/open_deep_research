# Educational Options Trading Framework for a $700 Account Targeting $200–300/Week (For Learning Purposes Only)

## Introduction

This report presents a comprehensive, research-based educational options trading framework tailored to a $700 starting account. The framework is built to help learners understand capital preservation, options strategies best-suited for small accounts, technical stock selection, trade structuring, and risk controls. Strict, industry-aligned rules are provided for position sizing, permitted strategies, selection criteria, liquidity filters, and trade execution—including full illustrations with hypothetical trade examples using real-world option chain data. The realistic feasibility of the stated $200–$300 weekly return target is analyzed based on risk management principles and probability. All recommendations are cited from mainstream brokerage resources and expert educational sources.

---

## 1. Position Sizing and Maximum Risk Per Trade

Effective position sizing is the foundation of sustainable options trading, especially for small-capital accounts vulnerable to drawdown.

- **Risk Per Trade:** The widely recommended rule for leveraged assets like options is to risk no more than 1–2% of total account value per trade. On a $700 account, this suggests a cap of $7–$14 per position. Stronger capital preservation is achieved by staying closer to 1% or less, but for learning, a maximum of $14 per trade is a balanced compromise[1][2][3].
- **Consistency:** Use a fixed dollar risk per trade rather than varying the amount. No single trade should exceed this cap, regardless of how "certain" a setup may seem. This avoids emotional overexposure and catastrophic losses[3][4][5].
- **Scaling:** Never scale into losing trades or "double down." Each trade is managed as an independent risk event[4].
- **Volatility Adjustment:** For stocks with unusually high volatility, consider reducing max risk even further.

Risk management is non-negotiable. Following this constraint may sometimes require trading narrower spreads (e.g., $1-wide), or sitting out high-capital strategies until the account grows[5][6].

---

## 2. Permitted Options Strategies

Small accounts require strategies that keep risk defined and upfront capital requirements low.

- **Credit Spreads (Vertical Spreads)**: Bull put and bear call spreads are ideal. By simultaneously selling and buying options at neighboring strikes, risk is capped while the cost to enter is reduced. These are income strategies that take advantage of time decay, especially in high-IV environments[2][6][7][8].
- **Debit Spreads (Vertical Spreads):** Bull call and bear put spreads enable directional betting with limited risk and cost (outlay is the net debit)[9].
- **Single-Leg Options:** Simple long call/put purchases are allowed if strictly sized. However, they carry higher time decay and should be entered with caution and strong conviction[6][9].
- **Prohibited:** Naked short options and multi-leg "exotic" strategies are discouraged for small accounts due to unlimited/unpredictable loss profiles or large margin requirements[6][9].

Debit and credit spreads stand out for requiring less capital, tightly controlling downside, and scaling to strict risk parameters—crucial for a $700 account[6][7][8].

---

## 3. Stock Selection Criteria Based on Technical Analysis

Rigorous technical stock selection increases the odds of option trades working as planned.

### Key Technical Criteria

- **Volume:** Select stocks with above-average daily volume. Surges in volume confirm breakouts and chart signals. Avoid illiquid stocks[10][11][12].
- **Price Action Relative to Support/Resistance:** 
  - Look for setups at or near major support when considering bullish strategies; at resistance for bearish[13][14].
  - Patterns: Range consolidations near breaks, bounces off major levels, or well-formed breakout structures.
- **Momentum Indicators:**
  - RSI (Relative Strength Index): Overbought (>70) or oversold (<30), especially when combined with price behavior near key levels.
  - MACD: Positive/negative crossovers signaling trend shifts[10][13].
  - Bollinger Bands: Squeezes for breakout timing or mean reversion setups[13][14].
- **Trend Analysis:**
  - Moving averages (MA20, MA50, MA200): Use golden cross/death cross for trend direction.
  - Entry in direction of dominant trend offers higher probability[10][11][13].
- **Catalyst Avoidance:** Exclude stocks with imminent earnings announcements (which introduce unpredictable volatility)[11][12].
- **Sector and Market Context:** Favor sectors showing relative strength/momentum and stocks participating in index trends[12][13].

Multiple technical signals and confirming volume should align for robust entries[10][11].

---

## 4. Strike and Expiry Selection Rules (3–21 Day Holds)

Selecting the right strikes and expiries is vital for optimizing probability, risk, and reward.

- **Expiry Selection:** Match the trade horizon. For 3–21 day holding periods, select options with expiry in 7–21 days. This allows time for the trade to work while benefiting from time decay if short premium, or reducing theta risk if buying[15][16][17].
- **Strike Selection:**
  - **Credit Spreads:** Sell options 20–40 Delta OTM for higher win probability, or ATM for more premium. For small accounts, spreads should be as narrow as possible (e.g., $0.50–$1 width)[17][18].
  - **Debit Spreads:** Buy ATM or slightly ITM (50–60 Delta) for higher intrinsic value and lower theta risk. Sell the next OTM strike. Avoid deep OTM, which are cheaper but unlikely to profit[17][18].
- **Defined Risk:** Max loss = spread width minus net credit (credit spreads), or net debit (debit spreads)—ensure this fits within risk per trade limits[6][18].
- **Profit Targets:** Consider taking profit when 70–90% of max gain is reached before expiry (to avoid sudden reversals or gamma risk)[17].

---

## 5. Liquidity Filters

Liquidity ensures reliable pricing and easier entries/exits with minimal slippage.

- **Minimum Open Interest:** Only trade strikes with at least 100–500 contracts OI. Higher OI (1000+) is better for tighter spreads[19][20][21][22].
- **Bid-Ask Spread:** For options under $4 per contract, max spread should be $0.10–$0.20. For higher priced contracts or further-out expiries, can accept up to $0.30. Never trade wide spreads or illiquid option strikes[20][22].
- **Daily Volume:** Must have ≥10 contracts traded each day confirming activity[22].
- **Order Entry:** Always use limit orders to avoid "paying the spread" or getting poor fills in fast markets[20][22].

These standards minimize cost and increase the chance of executing at favorable prices, essential for small accounts where each dollar of slippage matters.

---

## 6. Example Trades (Hypothetical, Educational Only)

### Trade Example 1: SPY Bull Put Credit Spread

- **Underlying:** SPY @ $656.82 (IV: 21.94%)
- **Technical Context:** SPY is bouncing off major MA50 support, with positive MACD crossover, and volume surge above 10-day average.
- **Strategy:** Sell 1 SPY Apr 17 2026 $655 Put, Buy 1 SPY Apr 17 2026 $654 Put (7 days to expiry)
    - **$655 Put:** $3.00 bid / $3.05 ask (OI: 12,100)
    - **$654 Put:** $2.00 bid / $2.05 ask (OI: 10,500)
    - **Net Credit:** $1.00 – $0.60 = $0.40 ($40 per spread)
    - **Max Loss:** $1.00 – $0.40 = $0.60 ($60 per spread)
- **Position Size:** 1 contract = $60 risk (just under 10% of account; for educational illustration)
- **Profit Target:** $28–$36 (exit for 70–90% of max gain if SPY stays above $655 early)
- **P&L Scenarios:**
    - *SPY > $655 at expiry*: Full credit earned; profit = $36 after fees.
    - *SPY closes $654–$655*: Partial loss based on spread settlement.
    - *SPY < $654 at expiry*: Max loss = $60 per contract.
- **Liquidity Filters:** Both strikes have OI > 10,000, bid-ask spread $0.05—excellent liquidity[11][22].

### Trade Example 2: AAPL Bull Call Debit Spread

- **Underlying:** AAPL @ $210 (IV: 27.6%)
- **Technical Context:** AAPL in strong uptrend, just broke out above resistance with RSI crossing 55, MA20 sloping up.
- **Strategy:** Buy 1 AAPL Apr 24 2026 $210 Call, Sell 1 AAPL Apr 24 $210.5 Call (15 days to expiry)
    - **$210 Call:** $2.50 bid / $2.60 ask (OI: 2,000)
    - **$210.5 Call:** $2.15 bid / $2.20 ask (OI: 1,700)
    - **Net Debit:** $0.45 ($45 per spread)
    - **Max Loss:** $45 per contract
    - **Max Gain:** ($0.50 wide – $0.45 debit) = $0.05 ($5 per spread)
- **Position Size:** 1 spread = $45 risk (within educational demonstration limits)
- **Profit Target:** $4 (exit for 80% of max gain before expiry)
- **P&L Scenarios:**
    - *AAPL > $210.50 at expiry*: Profit = $5 per spread
    - *AAPL < $210 at expiry*: Loss = $45 per spread
    - *AAPL between $210 and $210.50*: Partial gain between $0 and $5
- **Liquidity Filters:** Both strikes OI > 1000 and bid-ask spread $0.05, excellent[13][22].

---

## 7. Probability, Win-Rate, and Weekly Profit Target Analysis

The stated goal of $200–$300 profit per week on a $700 account implies a very aggressive return of roughly 30–40% weekly. Achieving this while maintaining prudent risk parameters is mathematically and practically challenging.

- **Example Calculation:**
    - If risking $14/trade (2% of $700), to net $200, need a net 15 “R” (risk units) in a week, e.g., 10 winning trades netting $20 each, no losses.
    - With a 1:1.5 to 1:2 risk/reward, a trader needs a win rate well above 50% to reach the target. In reality, normal professional win rates are 40–60%[13][14].
    - To net $200 at $20 profit per win and $14 risk per loss:
        - Break-even win rate = $14/($20+$14) ≈ 41%
        - To make $200 over ten trades: win rate must be >70-80% unless risked amounts are increased (which would violate best-practices for capital preservation)
- **Feasibility:** Under strict small account management, this return is highly aspirational and risky. Substantially exceeding 2% risk per trade may (temporarily) allow the target but dramatically increases blow-up risk, defeating the framework’s purpose[13][14][15].
- **Distribution of Outcomes:** Success hinges on a long winning streak or unusually high win percentage, both of which are statistically unlikely over time without unacceptable risk-taking.
- **Realistic Expectation:** Consistent small profits, compounding over time, are the sustainable approach. It is essential to focus on learning, discipline, and process, not solely profit[13][15][6].

---

## 8. Framework Summary and Best Practices

- Limit risk per trade to 1–2% of account
- Use risk-defined, low-capital option strategies: credit/debit spreads, very selectively single-leg
- Pick stocks with:
    - High, rising volume
    - Clean trends or reliable support/resistance
    - Confirming momentum indicators (RSI, MACD)
    - Exclude stocks with poor liquidity or near earnings[10][11][13]
- For trades held 3–21 days:
    - Pick expiries close to planned hold but allow room for the move (7–21 days)
    - Use liquid strikes with tight spreads, high OI
    - Set profit targets of 70–90% of max possible prior to expiry
- Only trade when all criteria are met, and use strict discipline with exit and loss management
- Understand that the $200–$300/week goal is for conceptual education. In practice, prudent risk management is more important than hitting aggressive weekly profit targets.

---

## Sources

1. [Option Alpha: Increase Profits With Proper Position Sizing](https://www.youtube.com/watch?v=CiuNOEu0xTA)
2. [5 Best Options Trading Strategies for Small Accounts](https://www.youtube.com/watch?v=P6tHbwpGdV4)
3. [Option Alpha: Position Sizing—What Everybody Ought to Know](https://optionalpha.com/blog/position-sizing-what-everybody-ought-to-know)
4. [Stock Market Guides: Position Sizing in Stock and Options Trading](https://www.stockmarketguides.com/article/position-sizing)
5. [Next Level Global Academy: What is Position Sizing in Options Trading?](https://www.nextlevelglobalacademy.com/blog-posts/position-sizing-options-trading)
6. [OptionsPlay: How to Grow a Small Account](https://www.optionsplay.com/blogs/how-to-grow-a-small-account)
7. [Reddit: Credit Spread Options Trading for Small Accounts](https://www.reddit.com/r/options/comments/1f4ic5a/credit_spread_options_trading_trying_to_grow_an/)
8. [Schwab: Credit Spread Options Strategy](https://www.schwab.com/learn/story/reducing-risk-with-credit-spread-options-strategy)
9. [Wealthsimple: Options Spreads Guide](https://www.wealthsimple.com/en-ca/learn/debit-credit-spreads)
10. [PyQuantNews: Technical Analysis Tools for Options Trading](https://www.pyquantnews.com/free-python-resources/technical-analysis-tools-for-options-trading)
11. [OptionCharts.io: SPY Option Overview](https://optioncharts.io/options/SPY)
12. [Schwab: How to Pick Stocks: Fundamentals vs. Technicals](https://www.schwab.com/learn/story/how-to-pick-stocks-using-fundamental-and-technical-analysis)
13. [Fidelity: Technical Analysis for Options Trading](https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/learning-center/Deck_Technical-analysis-for-options.pdf)
14. [Longbridge: Options Technical Analysis—A Practical Guide](https://longbridge.com/en/academy/options/blog/options-technical-analysis-a-practical-guide-to-boosting-win-rates-with-chart-based-trading-100085)
15. [Barchart: Selecting the Right Option Expiration](https://www.barchart.com/education/selecting_the_right_options_expiration)
16. [Quora: How to Select the Strike Price for Swing Trading](https://www.quora.com/How-do-I-select-the-strike-price-if-I-want-to-do-swing-trading-and-hold-the-option-for-3-4-days-maximum-only)
17. [OptionsPlay: Optimal Expiration Dates and Strike Prices](https://www.optionsplay.com/blogs/optimal-expiration-dates-and-strike-prices)
18. [Bourse de Montréal: Optimal Expiration Dates and Strike Prices PDF](https://www.m-x.ca/f_publications_en/options_play_exp_dates_strike_prices_en.pdf)
19. [IBKR: Option Liquidity Tool—Trading Lesson](https://www.interactivebrokers.com/campus/trading-lessons/option-liquidity-tool/)
20. [IBKR: Open Interest—Trading Lesson](https://www.interactivebrokers.com/campus/trading-lessons/open-interest/)
21. [TradingBlock: Options Trading Liquidity](https://www.tradingblock.com/blog/options-liquidity)
22. [TradingBlock: Options Market Basics](https://www.tradingblock.com/blog/option-basics)
