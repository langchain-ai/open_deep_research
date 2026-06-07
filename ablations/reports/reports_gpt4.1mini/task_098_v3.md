# Comprehensive, Explicit Options Trading Framework for a $700 Retail Account Targeting $200–300 Weekly Returns  
*Educational Purposes Only – Not Personalized Financial Advice*

---

## 1. Explicit Position-Sizing Rules and Maximum Risk Thresholds with Tiered Risk Levels

### 1.1 Capital and Risk Allocation

To protect capital in a small $700 account while aiming for weekly target returns of $200–300 (corresponding to ~28.5–42.9% weekly return), position sizing must be rigorously controlled due to high leverage and transaction costs. Institutional-grade guidelines recommend:

- **Conservative Risk Tier:**  
  - **Maximum Risk per Trade:** 1% of account = **$7**  
  - Focus: Capital preservation, limits drawdowns  
  - Position sizes determined so max loss per trade ≤ $7  
- **Moderate Risk Tier:**  
  - **Maximum Risk per Trade:** 2% of account = **$14**  
  - Enables moderate gains but increases drawdown risk  
- **Aggressive Risk Tier:**  
  - **Maximum Risk per Trade:** 5% of account = **$35** (maximum allowed, but generally discouraged for small accounts)  
  - Suitable only for well-established, experienced traders willing to accept large volatility

### 1.2 Position Size Calculation

Position sizing is derived by:

\[
\text{Number of Contracts} = \left\lfloor \frac{\text{Max Risk per Trade}}{\text{Max Risk per Contract}} \right\rfloor
\]

- *Max Risk per Contract* = *(Spread Width in $)* − *(Net Credit Received or Net Debit Paid)*  
- For debit spreads: risk = net debit paid  
- For credit spreads: risk = spread width − credit received

Example:  
- If risking $14 per trade, and spread max risk is $2 per contract, then position size = 7 contracts.

### 1.3 Capital Deployment and Concurrent Positions

- **Maximum Concurrent Exposure:** Do not allocate more than **20% of total capital** ($140) at once to open positions, to mitigate risk from simultaneous adverse moves.  
- **Total Risk Limit:** Total account risk across all trades at a time ≤ **10% of account** ($70). This ensures that multiple losing trades do not cause catastrophic drawdowns.

### 1.4 Stop-Loss and Drawdown Controls

- **Trade Stop-Loss:** Exit a trade if loss reaches 50-75% of max allowed risk per trade (e.g., $3.50–$5.25 for $7 max risk). This risk-limiting practice reduces big losses that can quickly deplete capital.  
- **Weekly Loss Threshold:** Cease trading for the week if cumulative losses ≥ **5%** of account ($35) conservatively, or up to **10%** ($70) if willing to accept higher drawdown.  
- **Monthly Max Drawdown:** Recommended max drawdown at **15%** ($105) to pause and reevaluate strategy.

---

## 2. Permitted Option Strategies with Explicit Rationale for Exclusions

### 2.1 Allowed Strategies

Focus on *defined risk* strategies suitable for a small capital base to protect against unlimited loss and excessive margin calls:

- **Vertical Credit Spreads:**  
  - Bull Put Spread: sell OTM put, buy further OTM put  
  - Bear Call Spread: sell OTM call, buy further OTM call  
  Provides risk-defined maximum loss and benefits from time decay (Theta).  
- **Vertical Debit Spreads:**  
  - Bull Call Spread: buy call, sell higher strike call  
  - Bear Put Spread: buy put, sell lower strike put  
  Offers directional exposure with limited max loss (net debit).  
- **Iron Condors / Iron Butterflies:**  
  - Multi-leg combinations of credit spreads for neutral market bias  
  Suitable only if enough capital and risk management discipline are present.  
- **Calendar Spreads (Limited Use):**  
  - Sell short-term options while buying longer-term options to benefit from time decay differences.

### 2.2 Explicitly Excluded Strategies

Due to disproportionately high risk, margin requirements, or capital limitations, exclude:

- **Naked Option Selling:**  
  - Naked calls or puts carry unlimited or undefined risk, inappropriate for small retail accounts.  
- **Unhedged Straddles or Strangles:**  
  - High Vega and high risk exposure without defined risk limits.  
- **Leveraged Multi-leg Spreads with Undefined Risk:**  
  - Complex, margin intensive, and capital-prohibitive for $700 accounts.  
- **Speculative Single-leg Long Calls/Puts Without Clear Exit:**  
  - Exposed to full premium decay and require substantial capital or luck to profit.  
- **Portfolio Margin or Margin Beyond Reg T:**  
  - Generally unavailable or too risky for smaller retail accounts.

---

## 3. Precise Technical Stock and ETF Selection Criteria

To ensure adequate liquidity, controlled risk, and technical edge, all underlying stocks/ETFs must meet these strict, quantifiable filters:

### 3.1 Liquidity Filters

- **Minimum Average Daily Volume (ADV):** At least **250,000 shares traded daily**, preferably >500,000 for tighter fills and lower slippage.  
- **Option Open Interest per Strike:** Minimum **100 contracts** to ensure tight spreads and ease of entry/exit.  
- **Bid-Ask Spread Maximum:** ≤ **$0.30** on options, ideally ≤ $0.20 for spreads to reduce frictional costs.

### 3.2 Price and Volatility Filters

- **Stock Price:** Prefer stocks priced **> $20**, avoiding nano or penny-like stocks prone to erratic price swings and illiquidity.  
- **Implied Volatility (IV):** Between **20% and 40%** annualized.  
  - Lower IV (<20%) reduces option premiums (less income potential).  
  - Higher IV (>40%) increases premium but signals higher risk of sharp moves.  
- **Average True Range (ATR):** Between **2% and 5%** of stock price over last 14 days to ensure meaningful movement vs spread width (e.g., for $50 stock, ATR $1–$2.50).

### 3.3 Technical Indicator Thresholds (All Must Apply)

| Indicator                | Precise Threshold            | Purpose                              |
|--------------------------|-----------------------------|------------------------------------|
| RSI (14 period)           | Between **40 and 60**        | Neutral to mild momentum; avoid oversold/overbought extremes |
| MACD Histogram           | ≥ **0**, ideally increasing | Confirms bullish momentum/confirmation for selects |
| Moving Averages Cross    | 10-day EMA above 50-day EMA | Indicates established uptrend or trend strength |
| Support/Resistance Levels | ≥ 3 distinct historical touches| Confirms validity for trade entry and stop-loss placement |
| Relative Volume (RVOL)   | ≥ **1.2** current vs avg     | Shows increased trader interest for reliable liquidity |

### 3.4 ETF Criteria

- Select highly liquid ETFs tracking major indices or sectors (e.g., SPY, QQQ, IWM)  
- ETF volume ≥ **1,000,000 shares per day**  
- Options open interest ≥ **300 contracts** per strike  
- Bid-ask spread ≤ **$0.10** for liquid ETFs

---

## 4. Strike Price and Expiry Selection Rules for 3 to 21 Day Holdings

### 4.1 Expiry Selection

- Choose expirations between **3 and 21 calendar days**.  
- For **credit spreads**, prefer expirations **14–21 days out at initiation** to balance time decay and gamma risk.  
- For **debit spreads**, expirations closer to **21 days** to reduce time decay impact; avoid < 7 days unless very confident.  
- Gradually manage or close positions before the last 3 days to avoid heightened gamma and volatility risk.

### 4.2 Strike Price Delta Ranges

| Strategy Type       | Short Option Delta Range | Long Option Delta Range | Explanation                                     |
|---------------------|-------------------------|------------------------|------------------------------------------------|
| Credit Spreads      | 0.15 to 0.40            | 0.05 to 0.25           | Short options with higher chance to expire worthless, long options as protective hedge |
| Debit Spreads       | 0.30 to 0.55            | 0.50 to 0.65           | Long options near ATM or slightly ITM to maximize directional payoff, short option offsets cost |

### 4.3 Spread Widths

- **Minimum spread width:** $3 per contract (sufficient reward for risk)  
- **Maximum spread width:** $8 per contract (to limit max loss and margin)  
- Preference for tighter spreads ($3–5) in small accounts to allow meaningful contracts without exceeding risk limits.

### 4.4 Risk/Reward Ratios

- **Credit Spreads (selling premium):**  
  - Target credit collected ≥ 33% of spread width (min 0.33:1 reward/risk)  
  - Aim to close trades after achieving **50–75% of max profit** to protect gains.  
- **Debit Spreads (buying premium):**  
  - Aim for risk/reward ≥ 1:1 (potential reward at least equal to debit paid)  
  - Profit targets at **50–70% of max profit**; consider stop-loss at max debit paid.

---

## 5. Incorporation of Transaction Costs, Fees, and Realistic Slippage Into All Calculations

### 5.1 Transaction Cost Components

- **Commissions:** Usually $0 with major brokers, but some small fees may apply (e.g., $0.65–$1 per contract roundtrip). Use $1 per contract roundtrip as conservative estimate.  
- **Bid-Ask Spread Cost:** If spread = $0.30, half spread = $0.15 (cost per side), so expect to lose up to $0.30 premium roundtrip spread friction per contract.  
- **Slippage:** Realistic slippage estimated at **0.5% to 1.5% of trade size**, higher in volatile or illiquid markets.

### 5.2 Impact on Position Sizing and Returns

- All profit/loss and risk calculations must factor in:

\[
\text{Total Cost per Contract} = \text{Commissions} + \text{Bid-Ask Spread} + \text{Slippage}
\]

E.g., for a credit spread collecting $1 premium with $0.30 spread and $1 commission: actual net credit might effectively be closer to $0.60.

- Realistic P&L scenarios and risk should assume **5-10% reduction in theoretical profits** due to trading friction.  
- Position sizing should be based on *net* achievable profit/risk after transaction costs.

---

## 6. Scenario-Based Probability Analyses Including Weekly Outcome Distributions, Win Rate Ranges, Account Survival, and Drawdown Risks

### 6.1 Weekly Return Modeling with Position Sizing Constraints

- To generate target $200–$300 weekly returns on $700 (28.5–42.9%), with max $14 risk per trade:

Assuming:

- Average net profit per winning trade (after costs): $9  
- Max loss per losing trade: $14  
- Number of trades per week: \(n\)

Expected weekly profit equation:

\[
\text{Weekly Profit} = n \times [p \times 9 - (1-p) \times 14]
\]

Solving for win rate \(p\) at \(n=25\) trades/week and target $200:

\[
200 = 25 \times (9p - 14 + 14p) = 25 \times (23p - 14)
\]
\[
8 = 23p - 14 \Rightarrow p = \frac{22}{23} \approx 95.7\%
\]

- **Required win rate ~95.7% at 25 weekly trades**—extremely high and practically unachievable over time.

For fewer trades (e.g., 10/week), required win rate exceeds 99%.

### 6.2 Probability Distributions and Risk of Ruin

- High required win rates stem from low reward-to-risk ratios constrained by small account size and commissions/slippage.  
- Risk of ruin is high if any single loss approaches 2%+ of account given limited capital and leverage.  
- Realistic win rates for defined-risk credit spreads are 70-80% but with reward/risk ~0.5:1, limiting expected returns. Debit spreads have lower win rates (~40-60%) but higher reward/risk.  
- Sustaining consistent 30-40% weekly returns is statistically improbable for retail traders under these conditions.

### 6.3 Risk Management Emphasis

- Prioritize survival: ensure max losses per trade remain below 2%–3% of account.  
- Employ strict stop-loss discipline and avoid “double down” or over-sizing after losses.  
- Use scenario Monte Carlo or binomial models to simulate trade sequences and estimate probability of account drawdowns ≥ 10% within 4-week intervals.

---

## 7. Benchmarking Feasibility and Statistical Requirements Against Professional Trader Metrics

- Professional option traders typically target **1–3% weekly returns** on significantly larger capital, using sophisticated risk models, hedging techniques, and premium data access [36].  
- Typical professional win rates hover between 50–70%, with positive expectancy derived by scaling and small edge strategies, not seeking 30-40% weekly gains consistently.  
- Retail studies show many small account traders suffer losses or small returns especially due to commissions, slippage, and psychological factors influencing overtrading [39][40].  
- Small accounts face disproportionate friction costs and margin constraints that professional accounts mitigate via scale and infrastructure.  
- The high weekly target ($200–300 on $700) equates to very aggressive risk-taking often incompatible with sustainable growth according to academic and industry data.  
- Realistic goals for retail traders: 5–10% weekly returns with strict risk discipline and defined-risk strategies, accumulating gains over months—not weeks [1][32].

---

# Summary

This comprehensive framework underscores an explicit, tightly controlled approach to options trading on a $700 account aiming for $200–300 weekly returns. It defines:

- **Strict position sizing (1–2% max risk per trade),** with clear formulas and tiered risk levels.  
- **Permitted option strategies limited to defined risk vertical spreads, iron condors, and select debit spreads.**  
- **Explicit technical criteria requiring liquidity, momentum, and volatility thresholds with exact numeric cutoffs.**  
- **Precise delta ranges and expiry windows for strike pricing and lifespan, balancing time decay, risk, and reward.**  
- **Inclusion of commissions, bid-ask spreads, and slippage realistically reducing achievable profits, directly impacting risk/reward and trade sizing.**  
- **Scenario analyses showing impractical required win rates (>90%) and severe drawdown risks for sustaining such high weekly returns on a small base.**  
- **Benchmarking that highlights the extreme difficulty of meeting this target consistently, contrasting professional trader norms and retail trading realities.**

Given these findings, this framework is invaluable for educational purposes and simulation, illustrating the quantitative constraints and risks inherent in ambitious small account options trading. It also serves as a rigorous guideline for disciplined planning, trade selection, and risk management.

---

### Sources

[1] Options Trading with a Small Account — https://us.amazon.com/Options-Trading-Small-Account-Strategies-ebook/dp/B0GMJN6M5D  
[2] The Best Option Strategies for Small Accounts — https://www.viperreport.com/option-strategies-for-small-accounts/  
[3] Profiting with Small Account | Option Alpha — https://optionalpha.com/lessons/small-account-options-strategies  
[6] What Everybody Needs to Know About Proper Position Sizing — https://optionalpha.com/blog/position-sizing-what-everybody-ought-to-know  
[7] Position Sizing in Stock and Options Trading — https://www.stockmarketguides.com/article/position-sizing  
[9] What is Position Sizing in Options Trading? & Why It's Key — https://www.nextlevelglobalacademy.com/blog-posts/position-sizing-options-trading  
[10] Position Sizing Guide — https://www.tradingsim.com/blog/position-sizing-guide  
[13] Options Trading Strategies — Charles Schwab — https://www.schwab.com/options/options-trading-strategies  
[16] Essential Technical Indicators for Successful Options Trading — https://www.investopedia.com/articles/active-trading/101314/top-technical-indicators-options-trading.asp  
[20] How to Choose an ETF — https://us.etrade.com/knowledge/library/etfs/how-to-choose-etfs  
[21] How to Maximize Profits with Strike and Expiration — https://www.insiderfinance.io/resources/how-to-maximize-profits-with-strike-and-expiration  
[22] Options Selection Strategies: Complete Guide to Choosing Strike Prices and Expiry Dates — https://longbridge.com/en/academy/options/blog/options-trading-strategies-a-comprehensive-guide-to-strike-price-and-expiry-selection-100092  
[23] OPTIMAL EXPIRATION DATES AND STRIKE PRICES (PDF) — https://www.m-x.ca/f_publications_en/options_play_exp_dates_strike_prices_en.pdf  
[24] Strike Price Selection: The Complete Guide to Maximizing Returns — https://www.barchart.com/education/strike_price_selection  
[25] Selecting a Strike Price and Expiration Date — https://www.fidelity.com/learning-center/investment-products/options/selecting-strike-price-expiration-date  
[26] Slippage and Commissions Importance — https://optionalpha.com/community/posts/slippage-and-commissions-importance-2025100420261  
[27] What Is Slippage And How Does It Impact Our Trading Success? — https://www.thebluecollarinvestor.com/what-is-slippage-and-how-does-it-impact-our-trading-success/  
[28] Absent fees, retail traders do better — https://newsroom.haas.berkeley.edu/research/absent-fees-retail-traders-do-better/  
[29] Commission Trading: What Is Realistic To Pay? — https://www.quantifiedstrategies.com/commission-trading/  
[30] Slippage and Commission: The Hidden Costs of Algorithmic Trading — https://trader-algoritmico.com/blog/slippage-and-commission-the-hidden-costs-of-algorithmic-trading  
[32] OptionsPlay Client Guide on Growing Small Account with Credit Spreads — https://www.optionsplay.com/blogs/how-to-grow-a-small-account  
[35] Scaling Up in Small Accounts — https://www.tastylive.com/news-insights/scaling-up-in-small-accounts  
[36] Professional Traders vs. Retail Traders - The Key Differences — https://www.daytrading.com/professional-vs-retail-traders  
[39] An Anatomy of Retail Option Trading (LSU Paper) — https://www.lsu.edu/business/files/event-files/2025-finance-mardi-gras/retail_option_trading_v2.pdf  
[40] An Anatomy of Retail Option Trading (PDF) — https://www.brettonwoodsskiconference.com/uploads/b/f9bfc8b0-0251-11ed-a646-3dea17112d2f/An%20Anatomy%20of%20Retail%20Option%20Trading.pdf