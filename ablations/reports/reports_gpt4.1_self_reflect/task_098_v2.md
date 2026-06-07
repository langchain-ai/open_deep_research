# Educational Options-Trading Framework for a $700 Account Targeting $200–$300 Weekly Returns  
*For Research and Learning Purposes Only — Not Financial Advice*

---

## 1. Position Sizing Rules and Maximum Risk Per Trade

Effective position sizing is critical for small accounts to control risk and avoid catastrophic losses. Industry standards for options trading are strict on risk per trade:

- **Recommended Risk Per Trade:**
  - **Conservative:** 1–2% of account per trade ($7–$14 for a $700 account)
  - **Moderate:** 3–5% ($21–$35)
  - **Aggressive (not recommended for long-term sustainability):** Max 10% ($70) — discouraged except for rare, high conviction cases, as this exposes the account to unacceptable drawdown risk[1][2][3].

- **Calculation Example:**  
  For a $700 account at 5% risk per trade ($35 max loss per trade), one could open up to 2 simultaneous trades if only risking $35 in each (totaling $70/10% portfolio "heat"). With stricter 2% risk ($14/trade), up to 5 simultaneous trades are possible before reaching a 10% maximum portfolio risk.

- **Portfolio Heat:**  
  Total aggregate open trade risk exposure (“portfolio heat”) should not exceed 10% of account value at any time (i.e., $70 on $700)[4][5].

- **Number of Simultaneous Trades:**
  - If risking 2% ($14): Up to 5 trades (5 × $14 = $70)
  - If risking 5% ($35): 2 trades ($35 × 2 = $70)
  - Always round down contract size to stay below these caps, especially when contract risk is not a round number.

- **Consistency Requirement:**  
  Always size trades based on maximum allowed risk per trade, not exposure to premium or margin requirement[3][6].

This approach not only prevents single-trade blow-ups but also minimizes the impact of recurring losing streaks, which are statistically inevitable[3][7].

---

## 2. Permitted Option Strategies (and Rationales)

Given the size and risk profile of a $700 account, strategies must keep both risk and capital requirement tightly defined.

### **Permitted Strategies**

**a. Vertical Credit Spreads (Bull Put and Bear Call Spreads)**
- **Why Permitted:** Defined risk (max loss = spread width minus net credit), comparatively low margin requirement, positive time decay (theta), ability to profit from direction or sideways movement[8][9].
- **Application:** Sell-at or out-of-the-money (OTM) option; buy further OTM (insured); both legs same expiry.

**b. Vertical Debit Spreads (Bull Call and Bear Put Spreads)**
- **Why Permitted:** Defined risk (max loss = net debit), low upfront cost compared to naked long options, suitable for directional moves[8][10].
- **Application:** Buy-at or slightly ITM option; sell OTM option, same expiry.

**c. Long Calls or Long Puts (Single-Leg Options)**
- **Why Limited/Permitted Cautiously:** Limited to trades where the premium risk is comfortably within 2–5% of account ($7–$35). These carry higher risk of total premium loss and higher theta decay, so should be used sparingly and with strong conviction and technical confirmation[11].

**d. Iron Condor & Iron Butterfly**
- **Why Permitted for Advanced Learners:** Both are defined-risk and can provide income in range-bound markets, but require careful sizing and high liquidity in all four legs. Suitable only if $1-wide spreads can be opened with single contracts and costs kept <5% of account[12][13][14].

**e. Poor Man’s Covered Call (Deep ITM LEAP + Short OTM Call)**
- **Conditional:** Allowed if the LEAP and short call setup can be constructed with risk under 5% of account, and only on highly liquid, lower-priced underlyings[15].

### **Not Permitted**
- **Naked Short Options (Calls or Puts):** Unlimited risk, potentially massive losses exceeding account size[8].
- **Strangles/Naked Premium Selling:** Undefined risk not suitable below $25k account size given regulatory and margin limitations[9].
- **Multi-Leg Exotic or Complex Spreads (Ratio, Calendar, etc.):** Capital-inefficient and/or unlimited risk in some scenarios.

The strategies above are all capital-efficient and have loss maximums known in advance, which aligns with the strict position-sizing required for small, aggressive accounts[8][15].

---

## 3. Stock Selection Criteria Using Technical Analysis

Robust, objective stock selection improves trade probability and minimizes slippage.

### **Liquidity Requirements**
- **Underlying Price:** $20/share or higher (preferably major ETFs: SPY, QQQ, IWM)[16].
- **Stock Avg Daily Volume:** ≥ 1 million shares[16][17].
- **Options Volume (Selected Strike):** ≥ 100 contracts traded per day[17].
- **Open Interest:** ≥ 100 contracts on the strike; ≥ 500 is ideal for tight spreads[17].

### **Technical Indicators and Parameters**
- **Moving Averages (MA20, MA50, MA200):**  
  - *Trend identification:* Price above rising MA20/50 for bullish; below falling MA20/50 for bearish[18].
- **RSI (14-period):**
  - *Oversold (<30):* Consider for bullish entries, especially if price at support[19].
  - *Overbought (>70):* Consider for bearish entries, especially at resistance.
- **MACD (12,26,9):**
  - Bullish entry: MACD line crosses above signal line, preferably with price above MA20.
  - Bearish entry: MACD line crosses below[20].
- **Bollinger Bands (20-period, 2 SD):**
  - Entry signals: Price closes outside band, then reverts, or bands squeeze/contraction signals upcoming breakout.
- **Support/Resistance:**  
  - *Bullish trades:* Price tests/bounces off clear support (prior low, moving average, or congestion zone) with increasing volume/confirmation[21][22].
  - *Bearish trades:* Price rejection or reversal at known resistance areas, often confirmed by overbought RSI and momentum loss.
- **Volume:**  
  - Above 10-day average volume on entry, confirming breakout or momentum[17][18].
- **Catalyst Avoidance:**  
  - No trades on stocks with earnings, split, or major news due within contract expiry; avoids unpredictable moves[17].

### **Entry Rule Synthesis**
- Commit only when at least two technical signals align (e.g., price above MA and MACD bullish, or RSI oversold at support with volume surge).
- Example: “Enter a bull put spread on SPY if price bounces off MA20 with an RSI crossover from below 30 and MACD bullish; all option liquidity criteria must be satisfied.”

---

## 4. Strike and Expiry Selection Rules for 3–21 Day Holds

Proper strike/expiry selection balances premium, risk, and probability of success.

**a. Expiry Selection**
- For 3–21 day intended holds: Trade options expiring in 7–21 calendar days, ensuring sufficient time for the move but not so much as to dilute theta/time decay edge[23][24].

**b. Strike Selection**
- **Credit Spreads:**  
  - Sell leg: 20–40 Delta OTM for higher win probability (e.g., 20–30 Delta is common; select a strike just OTM)[24].
  - Buy leg: One strike further out. Use $1 or $0.50-wide spreads to keep risk defined and within account sizing limits.
- **Debit Spreads:**
  - Buy leg: At-the-money or slightly ITM (Delta 50–60) to maximize intrinsic value and reduce theta risk.
  - Sell leg: Next OTM strike; maintains low entry cost and quantifiable max gain/loss.
- **Single-Leg Options:**  
  - Prefer contracts with Delta 40–60 for directional; deeper ITM to balance premium cost and increase win probability.

**c. Premium Guidance**
- Maximum risk per spread = spread width ($1) × 100 – net premium (for credit spreads), or net debit paid (for debit spreads). Ensure max risk per trade fits $14–$35 cap.
- For small accounts: Target 30–50% ROI on risk per spread, i.e., risk $30 to make $10–$15[23].

---

## 5. Option Liquidity Filters

Ensuring tradeable, fair fills and managing exit risk:

- **Open Interest on Strike:** ≥ 100 (preferably ≥ 500)[17][25].
- **Option Contract Daily Volume:** ≥ 10–20 contracts; 100+ ideal[17].
- **Bid-Ask Spread:**  
  - < $0.10 (10¢) for options priced <$2;  
  - < $0.20 for options up to $5.  
  - Accept no more than 10% of the option’s price as the spread.
- **Underlying Asset:** Must meet underlying volume standards above.
- **Only enter/exit trades with limit orders. Never use market orders for small accounts[26].**

---

## 6. Illustrative Example Trades

**Note:** Option prices are for educational purposes only and should be verified live before actual trading.

---

### **Example 1: Bull Put Credit Spread on SPY**

- **Underlying:** SPY @ $656.82 (volatility 21.94%)
- **Trade Thesis:** SPY bounced off MA20 support, positive MACD crossover, RSI went from 28 to 35, volume above 10-day average—bullish, no earnings within next 3 weeks.
- **Trade Structure:**
  - Sell 1 SPY 2026-05-01 $655 put (Delta ≈ 25; bid $3.10 / ask $3.20; OI: 12,500)
  - Buy 1 SPY 2026-05-01 $654 put (bid $2.25 / ask $2.30; OI: 10,100)
  - **Net Credit:** $0.85 (midpoint) = $85 per spread.
  - **Max Loss:** Spread width ($1) × 100 – $85 = $15 per trade.
  - **Position Size:** 1 contract = $15 risk (just over 2% of $700).
- **Profit Target:** $8–$12 (60–80% of max gain) if SPY spends 1–2 days stable above $655.
- **P/L at Expiry (05/01/2026):**
  - SPY CLOSES ≥ $655: Keep full credit, profit $85.
  - SPY CLOSES ≤ $654: Max loss: lose $15.
  - SPY between $655–$654: Variable, partial loss.
- **Liquidity:** OI > 10,000, daily volume >1,000, bid-ask spread $0.10—excellent[25].
- **Order Entry:** Use limit order at midpoint; never take the market offer.

---

### **Example 2: Long Call Debit Spread on AAPL**

- **Underlying:** AAPL at $210.50 (volatility 27.6%)
- **Trade Thesis:** Strong uptrend, breakout above resistance, MACD bullish crossover, RSI rising from 40 to 55.
- **Trade Structure:**
  - Buy 1 AAPL 2026-05-10 $210 call (Delta ≈ 55, bid $2.70 / ask $2.75; OI: 3,100)
  - Sell 1 AAPL 2026-05-10 $212 call (Delta ≈ 41, bid $1.70 / ask $1.75; OI: 2,200)
  - **Net Debit:** $1.05 (midpoint) = $105 per spread.
  - **Max Loss:** $105 (premium paid)
  - **Max Gain:** ($2 × 100) – $105 = $95
  - **Position Size:** 1 spread; this is 15% of account (for illustration), but a smaller strike width or single contract with narrower spread ($1 or $0.50) is preferred to comply with position-sizing rules.
- **Profit Target:** Sell for $1.80–$1.90 (70–90% of max gain) if AAPL moves quickly above $212.
- **P/L at Expiry (05/10/2026):**
  - AAPL ≤ $210: Lose $105 (max risk)
  - AAPL ≥ $212: Profit $95
  - AAPL between $210–$212: Partial gain $0–$95.
- **Liquidity:** OI high, volume strong, bid-ask $0.05—top tier.
- **Order Entry:** Limit order; never hit market.

---

## 7. Win-Rate and Result Analysis for Target Profit

The **target of $200–$300 per week (28–43%/wk on $700)** is significantly above even aggressive professional benchmarks. Analysis with the above risk/reward parameters:

- **Position Sizing:** At $15 risk per trade, even a 100% win rate on 5 trades nets only $75/week.
- **To achieve $200/week:**
  - At $15 risk/trade, ROI of 100% ($15 profit per trade): need 14 winning trades/week—impossible within sizing and trade entry limitations.
  - **If max risk raised to $35 (5%/trade):** Winning 6 trades with $35 profit/trade = $210/week; but this requires accepting >25% probability of significant drawdown if losses cluster.
- **Required Win Rate and Distribution:**
  - With credit spreads returning 30–50% of risk ($15 trade returns $5–$7.50), achieving $200–$300/week would demand an unsustainably high win rate (above 90%) or risk stacking, which sharply increases risk of ruin[27].
- **Probability of Blow-up:**  
  - With 5%/trade risk, a losing streak of 5–10 trades (not rare) can halve or wipe out the account[1][7].
- **Professional Benchmarks:** Most pro options traders target ~2–7%/month; weekly goals above 5%/week are not sustainable for long periods[2][17].

---

## 8. Feasibility Discussion and Realistic Learning Goals

- **Aggressive weekly profit targets for micro-accounts are mathematically extremely unlikely to be sustained while adhering to credible risk management.**
  - Professional-grade risk principles (max 2–5% per trade, 10% total risk) make $200–$300/week on $700 nearly unattainable long term, unless extreme risk-taking is used—which will result in eventual account wipeout[1][2][4][17].
- **Critical Risks:**  
  - Increasing trade risk to chase profit targets (10%+ per trade) invites high drawdowns, psychological stress, emotional trading, and inevitable large losses[7].
  - Options are complex derivatives; beginners should prioritize learning risk management, trade construction, technical analysis, and strict discipline over chasing outsized returns[7][27].
- **Educational and Sustainable Focus:**  
  - Use paper trading or tiny real trades to build process discipline.
  - Accept realistic growth targets: Consistently achieving even 5–10% compounded monthly is a significant win in options trading[2].
  - Focus on capital preservation, learning, and steady compounding.  
  - Avoid adjusting position size upward after a winning streak to chase goals ("revenge trading").

---

## 9. Framework Summary and Best Practices

- **Risk no more than 2–5% per trade;** never more than 10% portfolio heat.
- **Trade only risk-defined options strategies:** vertical spreads, cautious long calls/puts, and carefully sized iron condors/butterflies.
- **Choose stocks/options with strong liquidity:** high daily volume, high OI, and tight spreads.
- **Enter trades only on clear technical signals, never on hope or rumor.**
- **Strictly adhere to risk, profit target, and exit rules; always use limit orders.**
- **Do not chase unrealistic weekly profit targets at the expense of prudent risk management.**

---

### Sources

[1] 3-5-7 Rule in Trading: Everything Traders Should Know: https://www.metrotrade.com/3-5-7-rule-in-trading/  
[2] How To Grow a Small Account | OptionsPlay Blog: https://www.optionsplay.com/blogs/how-to-grow-a-small-account  
[3] Risk Management - OptionStrat AI: https://mintlify.com/EconomiaUNMSM/OptionStrat-AI/guides/risk-management  
[4] Position Sizing in Options Trading | Optionstrading.org: https://www.optionstrading.org/blog/position-sizing-in-options-trading/  
[5] Options Position Sizing: How Much to Risk Per Trade | Ainvest Options Pilot: https://optionpilot.ainvest.com/blog/position-sizing-options-guide  
[6] Options: How to choose your trade size (Fidelity): https://www.fidelity.com/viewpoints/active-investor/options-trade-size  
[7] Best Option Strategies for Small Accounts: A Premium Seller's Guide: https://optionstradingiq.com/best-option-strategies-for-small-accounts/  
[8] Charles Schwab: Credit Spread Options Strategy: https://www.schwab.com/learn/story/reducing-risk-with-credit-spread-options-strategy  
[9] Tastytrade: Short Vertical (Credit) Spread: https://support.tastytrade.com/support/s/solutions/articles/43000435260  
[10] Wealthsimple: Options Spreads Guide: https://www.wealthsimple.com/en-ca/learn/debit-credit-spreads  
[11] Timothy Sykes: A Comprehensive Guide to Trading Options with a Small Account: https://www.timothysykes.com/blog/small-account-trading/  
[12] TradeStation: Iron Condors & Butterflies Explained: https://www.tradestation.com/learn/options-education-center/iron-condors-butterflies/  
[13] Alpaca.Markets: Iron Condor vs Iron Butterfly: https://alpaca.markets/learn/iron-condor-vs-iron-butterfly  
[14] Option Alpha: Iron condor vs. Iron Butterfly: https://optionalpha.com/learn/iron-condor-vs-iron-butterfly  
[15] Options Trading IQ: Poor Man’s Covered Call for Small Accounts: https://optionstradingiq.com/poor-mans-covered-call/  
[16] TradingBlock: Options Trading Liquidity: https://www.tradingblock.com/blog/options-liquidity  
[17] Interactive Brokers: Option Liquidity Tool: https://www.interactivebrokers.com/campus/trading-lessons/option-liquidity-tool/  
[18] Groww: Best Indicators for Option Trading: https://groww.in/blog/best-indicators-for-option-trading  
[19] Best Indicators for Option Trading | Tradetron Blog: https://tradetron.tech/blog/best-indicators-for-options-trading  
[20] Investopedia: Essential Technical Indicators for Successful Options Trading: https://www.investopedia.com/articles/active-trading/101314/top-technical-indicators-options-trading.asp  
[21] TradingView: Support and Resistance — Indicators and Strategies: https://www.tradingview.com/scripts/supportandresistance/  
[22] Zerodha: Support and resistance in stock market: https://zerodha.com/varsity/chapter/support-resistance/  
[23] Option Alpha: Optimal Expiration Dates and Strike Prices: https://www.optionsplay.com/blogs/optimal-expiration-dates-and-strike-prices  
[24] Barchart: Selecting the Right Option Expiration: https://www.barchart.com/education/selecting_the_right_options_expiration  
[25] TradingBlock: Options Market Basics: https://www.tradingblock.com/blog/option-basics  
[26] IBKR: Open Interest—Trading Lesson: https://www.interactivebrokers.com/campus/trading-lessons/open-interest/  
[27] DayTrading.com: Options Strategies for Small Accounts: https://www.daytrading.com/options-strategies-small-accounts  

---

**Disclaimer:** This framework is for research and educational purposes only and does not constitute individualized investment advice. Options trading involves significant risk and is not suitable for all investors. Always consult with a registered financial advisor and review the OCC’s [Characteristics and Risks of Standardized Options](https://www.theocc.com/getcontentasset/a151a9ae-d784-4a15-bdeb-23a029f50b70/dfc3d011-8f63-43f6-9ed8-4b444333a1d0/riskstoc.pdf) before trading options.