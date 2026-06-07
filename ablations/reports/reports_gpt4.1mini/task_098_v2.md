# Comprehensive Options-Trading Framework for a $700 Retail Account Targeting $200–300 Weekly Returns  
*Educational only; not personalized financial advice*

---

## 1. Position Sizing and Risk Management Rules

Effective position sizing is critical for capital preservation and achieving ambitious weekly return goals in a small $700 account. The following quantitative rules define conservative and aggressive tiers, capital deployment, concurrent positions, and risk management controls.

### 1.1 Position Sizing Tiers

- **Conservative Risk Tier:**  
  - Risk per trade: **1% of account equity** = **$7** per trade  
  - Capital deployed per trade: Adjusted to keep max loss ≤ $7  
  - Intended for steady, low-risk growth and minimal drawdowns

- **Aggressive Risk Tier:**  
  - Risk per trade: Up to **5% of account equity** = **$35** per trade  
  - Capital deployed per trade: Allowed to risk up to $35 loss if stop-loss triggered  
  - Intended for traders tolerating higher volatility and drawdown risk aiming for faster growth

### 1.2 Maximum Capital Deployment per Trade and Concurrent Positions

- **Per Trade Capital Deployment:**  
  Trade size must reflect the capped max loss amount. For example, a vertical spread with a spread width of $5 and net credit/debit of $1 requires position sizing such that:

  \[
  \text{Position Size} = \left\lfloor \frac{\text{Max Risk Dollar Amount}}{\text{Max Risk Per Contract}} \right\rfloor
  \]
  
  For a $14 risk per trade, with $100 max risk per contract, position size = 0 contracts → thus only 1 contract at most unless smaller spread widths or cheaper strikes are found.

- **Concurrent Positions Limit:**  
  Total max exposure at any time should not exceed **10-20% of total equity** ($70–$140). This limits overexposure and risk of large simultaneous losses.

### 1.3 Weekly Loss Threshold and Stop-Trading Triggers

- **Weekly Loss Threshold:**  
  Set at **5% of account value** = **$35** for conservative, up to **10%** = **$70** for aggressive. If reached in realized/unrealized losses, stop all trading for the remainder of the week to avoid forced blowups.

- **Stop-Loss Rules:**  
  - Individual trade stop loss at **50-75% of max allowable loss per trade** (e.g., $3.50–$5.25 loss when risking $7)  
  - Stops can be triggered by option premium decay or underlying price moving beyond predefined technical invalidation levels.

- **Drawdown/Max Loss Triggers:**  
  - **Max drawdown per month: 15%** (approx. $105) recommended  
  - Breaching monthly max drawdown should trigger reevaluation and trading pause.

---

## 2. Permitted Option Strategies: Eligibility and Restrictions

The framework mandates strictly controlled, defined-risk option strategies tailored to small accounts to prevent excessive risk, margin use, or undefined losses.

### 2.1 Allowed Strategies

- **Vertical Credit Spreads** (Bull Put and Bear Call)  
  Defined risk, margin-efficient, benefit from time decay. Examples:  
  - Bull Put Spread: Sell OTM put, buy further OTM put.  
  - Bear Call Spread: Sell OTM call, buy further OTM call.

- **Vertical Debit Spreads** (Bull Call and Bear Put)  
  Directional, limited risk to net debit paid, reduced premium cost from long options alone.

- **Iron Condors** and **Iron Butterflies** (If capital allows)  
  Multi-leg defined-risk spreads offering neutral market bias.

- **Cash-Secured Puts** (if willing to hold stock)  
  Limited risk, margin-efficient for income generation.

### 2.2 Explicitly Prohibited Strategies

- **Naked option selling** (calls or puts) without coverage  
- **Unhedged straddles, strangles, or spreads with undefined risk**  
- **Portfolio margin or margin-intensive trades beyond Reg T**  
- **Leveraged multi-leg spreads with unknown maximum loss**  
- **Highly speculative single-leg long calls/puts without clear exit strategy**  
- **Complex debit strategies with large initial premium outlays exceeding risk limits**

---

## 3. Quantitative Strike Selection and Expiry Rules

Clear numerical thresholds serve to optimize probability of success, risk/reward, and transaction cost efficiency.

### 3.1 Strike Delta Ranges

- **Credit Spreads:**  
  - Short option delta: **0.20 to 0.40** (20%-40%) for higher probability of expiring worthless  
  - Long option delta: **0.10 to 0.25**, farther OTM for hedge

- **Debit Spreads:**  
  - Long option delta: **0.40 to 0.60** (near ATM or slightly ITM) to balance premium cost and payoff  
  - Short option delta: Approximately **0.30 to 0.45**

- **Single-leg directional options (if used):**  
  Delta range **0.40 to 0.60**, preferring ITM or ATM options to reduce time decay risk

### 3.2 Spread Width Parameters

- Minimum spread width: **$3 to $5** per contract to ensure meaningful max profit  
- Maximum spread width: **$8 to $10** per contract to limit max loss and margin

### 3.3 Expiration Durations

- Allowed expirations: **3–21 calendar days** from trade entry  
- Rationale:  
  - Short enough to maximize theta decay and capital turnover  
  - Long enough to reduce gamma risk and allow time for directional move

- Prefer expirations closer to 14–21 days when initiating credit spreads, gradually managing or closing down to 3–7 days out for optimal theta capture.

### 3.4 Risk/Reward Ratio Thresholds

- Target **minimum reward-to-risk ratio of 0.5:1** for credit spreads (e.g., collect at least 33% of spread width as premium)  
- Target **minimum 1:1 risk/reward** for debit spreads (e.g., potential reward at least equal to debit paid)  
- Set profit targets at **50-75% of max gain** to optimize win rate and reduce holding time

---

## 4. Underlying Stock / ETF Selection Criteria Using Technical Analysis

Stringent, quantitative technical criteria ensure selection of underlying assets that support liquidity, risk controls, and technical edge.

### 4.1 Liquidity and Price Filters

- **Average Daily Volume (ADV):** Minimum **1 million shares**; preferably above 3 million  
- **Option Open Interest (per strike):** Minimum **100 contracts**  
- **Bid-Ask Spread:** ≤ **$0.30** preferred  
- **Stock Price:** Prefer stocks priced **above $20**, avoiding penny-like volatility and illiquidity  
- **Relative Volume (RVOL):** Ideally **≥ 1.2** to indicate increased current trading vs typical volume  

### 4.2 Volatility and Price Movement Metrics

- **Average True Range (ATR):** Favor stocks with ATR between **2% to 5% of stock price** (e.g., $1.20–$3.00 on $60 stock) to define sensible spread widths and stop levels  
- **Implied Volatility (IV):** Moderate IV (e.g., 20% to 40%) preferred to balance premium received vs price movement risk

### 4.3 Technical Indicator Thresholds

| Indicator       | Minimum Range | Preferred Range                | Notes                                        |
|-----------------|---------------|-------------------------------|----------------------------------------------|
| RSI             | 40            | 45 - 60                      | Neutral to mild momentum, avoid extremes      |
| MACD Histogram  | ≥ 0           | Positive and increasing      | Confirms bullish momentum / trend strength    |
| Moving Averages | 10-day EMA > 50-day EMA | Confirmed crossover    | Indicates sustained trend direction           |
| Support Levels  | Validated by ≥ 3 historical touches | Clear horizontal support | Entry near support for bullish strategies |
| Resistance Levels| Validated by ≥ 3 touches | Clear horizontal resistance| Entry near resistance for bearish strategies |

### 4.4 Summary of Stock Criteria

Select liquid stocks or ETFs trading above $20, with strong volume and moderate volatility, clear and validated technical patterns (MA crossovers, confirmed support/resistance, neutral RSI), and expanding or steady momentum signals (MACD). Avoid stocks with erratic volume, shallow order books, or extreme RSI values above 70 or below 30.

---

## 5. Detailed Example Trades with Profit/Loss Scenarios

### Example Trade 1: Bull Put Credit Spread (Conservative, Income Strategy)

| Parameter             | Specification                                                             |
|-----------------------|----------------------------------------------------------------------------|
| Underlying            | Stock XYZ at $100                                                          |
| Technical Rationale   | Supports at $97 confirmed by 5 prior lows; RSI 52; 10-day EMA > 50-day EMA; RVOL 1.3 |
| Option Selection      | Sell 1 XYZ 3-week $97 put at $1.50 (delta 0.30); Buy 1 XYZ 3-week $94 put at $0.50 (delta 0.15) |
| Net Credit            | $1.00 per share × 100 shares = $100 gross credit                          |
| Spread Width          | $3.00 per share ($97 - $94)                                               |
| Max Risk              | $3.00 spread - $1.00 credit = $2.00 per share × 100 = $200 total          |
| Position Sizing       | Trader caps max risk at $14 → max contracts = 0.07 → rounded to 1 contract (educational example) |
| Target Profit         | 50-75% max credit = $50 - $75                                              |
| Expiry                | 15 calendar days                                                          |
| Transaction Costs     | Estimated $1.00 roundtrip (per contract)                                 |

**Profit/Loss Scenarios at Expiry (Price & P/L):**

| Stock Price at Expiry | Profit/Loss per Contract     | Explanation                                         |
|----------------------|-----------------------------|----------------------------------------------------|
| ≥ $97                | +$100 (max profit)           | Both puts expire worthless; keep full premium      |
| $96                  | +$66                       | Partial credit retained; short put expires worthless, long put worthless |
| $94                   | -$100 (max loss)            | Short put fully in-the-money; long put hedges loss |
| < $94                  | -$100 (max loss)            | Maximum loss capped by spread width minus premium   |

**Percentage Returns:**

- Max Profit: 0.5% of $700 account  
- Max Loss: 0.29% of $700 account

---

### Example Trade 2: Bull Call Debit Spread (Aggressive, Directional)

| Parameter             | Specification                                                              |
|-----------------------|------------------------------------------------------------------------------|
| Underlying            | Stock ABC at $50                                                            |
| Technical Rationale   | Breakout above resistance at $49; RSI 58; MACD histogram positive & rising; RVOL 1.5 |
| Option Selection      | Buy 1 ABC 3-week $51 call at $1.80 (delta 0.55); Sell 1 ABC 3-week $54 call at $0.80 (delta 0.35) |
| Net Debit             | $1.00 per share × 100 shares = $100 net debit                               |
| Spread Width          | $3.00 per share ($54 - $51)                                                 |
| Max Profit            | $3.00 - $1.00 = $2.00 per share × 100 = $200                               |
| Position Sizing       | Max risk $14 per trade → max contracts = 0.14 → rounded to 1 contract (illustrative) |
| Target Profit         | 50-70% max profit = $100 to $140                                            |
| Expiry                | 21 calendar days                                                            |
| Transaction Costs     | Estimated $1.00 roundtrip                                                    |

**Profit/Loss Scenarios at Expiry (Price & P/L):**

| Stock Price at Expiry | Profit/Loss per Contract   | Explanation                                         |
|----------------------|---------------------------|----------------------------------------------------|
| ≥ $54                | +$200 (max profit)         | Spread fully in the money                           |
| $52                  | +$100                     | Partial intrinsic value, after recovering net debit|
| ≤ $51                | -$100 (max loss)           | Spread expires worthless, loss capped at debit     |

**Percentage Returns:**

- Max profit: 28.6% of $700 account  
- Max loss: 14.3% of $700 account

---

## 6. Quantitative Analysis: Required Win Rates, Probability Distributions, Risk Metrics

### 6.1 Required Win Rate to Achieve $200–300 Weekly Returns

- Target weekly returns: $200–300 ⇒ approx. **28.5% to 42.9% return** on $700  
- Realistic expectation requires trading multiple contracts/trades with sufficient win probability and expected value (EV) per trade.

Assuming conservative trades with:

- Risk per trade: $7–$14  
- Average reward: 50-70% of max profit (e.g., $7–$10 per trade)  
- Trades per week: ≥25 (e.g., 5 trades a day over 5 trading days)  

Then:

\[
\text{Weekly Profit} = n \times [p \times \text{average win} - (1-p) \times \text{max loss}]
\]

Solving for win rate \(p\), to reach $200 with:

- \(n=25\), average win = $9, max loss = $7:  
\[
200 = 25 \times (9p - 7(1-p)) \Rightarrow 200 = 25 \times (16p -7) \Rightarrow 8 = 16p - 7 \Rightarrow p = 0.9375 = 93.75\%
\]

Even at this optimistic scenario, a **win rate exceeding 90%** is required, which is extraordinarily high and unrealistic for most retail traders.

### 6.2 Expected Value (EV) and Risk of Ruin (RoR)

- EV per trade must remain positive after factoring transaction costs (estimated 1–2% per trade) and slippage (typically 1–3% intraday price impact for small volume)

- Risk of Ruin increases steeply if risk per trade exceeds **1-2%**, which is often unavoidable to hit $200+ weekly

### 6.3 Probability Distributions and Outcome Modeling

- High-probability credit spreads usually exhibit 70–80% win rates but low reward-to-risk ratios (~0.5:1)

- Debit spreads have lower win rates (40–60%) but higher payoffs (≥2:1)

- Combining multiple position types and averaging outcomes is necessary, yet volatility in returns remains high

- A large number of trades is needed to smooth outcomes (Law of Large Numbers), impractical in short weekly horizons

---

## 7. Framework Contextualization: Realism, Psychological Considerations, and Alternative Targets

### 7.1 Realistic Benchmarking

- Professional traders typically target **1–3% weekly returns** (~5–15% monthly), reflecting risk-adjusted, sustainable performance.

- Retail traders often face high transaction costs and emotional pitfalls that reduce net returns well below professional benchmarks.

- Achieving **28–43% weekly return consistently on $700** is extraordinarily aggressive, with historical data indicating elevated risk of losing the entire account rapidly.

### 7.2 Psychological Risks

- Target-driven trading induces stress, emotional decision-making, overtrading, and position sizing violations.

- Drawdowns will be inevitable; managing psychology around these is critical yet difficult under high pressure.

- Risk of Ruin is heightened: aggressive sizing or chasing losses can lead to quick depletion of capital.

### 7.3 Alternative, More Achievable Framework

- **Target Weekly Returns:** Adjust to **5-10%** per week ($35–$70 on $700)

- **Position Sizing:** Risk no more than **1-2% of capital per trade** ($7–$14)

- **Strategy Focus:** Primarily defined-risk credit spreads and cash-secured puts with tight risk controls

- **Trade Frequency:** Moderate (5–10 trades/week) with strict stop-loss discipline

- **Win Rate Expectation:** 60-70% realistic with a risk-reward ratio near 0.5:1

- Long-term capital compounding over months focused on steady growth, capital preservation, and learning

---

## 8. Fundamental Payoff Mechanics for Debit and Credit Spreads (Preserved Clarification)

### 8.1 Debit Spreads (Bull Call, Bear Put)

- **Initial Outflow:** Net debit (premium paid) at trade initiation

- **Maximum Risk:** Limited strictly to net debit paid per contract

- **Maximum Profit:** Difference between strikes (spread width) minus debit paid

- **Breakeven:** Strike price of long option plus (calls) or minus (puts) net debit paid

- **Risk/Reward Balance:** Requires sufficient directional move within expiration window

### 8.2 Credit Spreads (Bull Put, Bear Call)

- **Initial Inflow:** Net credit received (premium collected)

- **Maximum Profit:** Limited to credit received if options expire worthless

- **Maximum Risk:** Strike width minus credit received, the worst-case assignment scenario

- **Breakeven Price:** Strike of short option minus or plus net credit received

- **Profit Drivers:** Time decay, implied volatility contraction, underlying price staying outside short strike

---

## Conclusion

This comprehensive, quantitatively precise framework enables retail traders with a $700 account to understand and potentially execute defined-risk options trading strategies aimed at ambitious weekly returns of $200–300. However, rigorous quantitative analysis shows such targets require exceedingly high success rates, aggressive sizing, and multiple weekly trades — all with significant psychological and financial risks.

Adherence to strict position sizing, strategy eligibility, strike selection, and technical stock filters is essential. Educational examples illustrate the application of the framework in real-world contexts with risk/reward scenarios.

Ultimately, the framework underscores the disproportionate risk in targeting near 30–40% weekly gains on small accounts, encouraging reconsideration towards more attainable goals emphasizing capital preservation, measured growth, and continuous learning within defined-risk option strategies.

---

### Sources

[1] Reddit: Level 2 trading strategies for small accounts targeting $200-$800: https://www.reddit.com/r/options/comments/e5mk16/level_2_trading_strategies_for_small_accounts/  
[2] Next Level Global Academy: Position Sizing Guide in Options Trading: https://www.nextlevelglobalacademy.com/blog-posts/position-sizing-options-trading  
[3] OptionAlpha: Position sizing what everybody ought to know: https://optionalpha.com/blog/position-sizing-what-everybody-ought-to-know  
[4] Investopedia: How to Reduce Risk With Optimal Position Size: https://www.investopedia.com/articles/trading/09/determine-position-size.asp  
[5] DayTrading.com: Options Strategies for Small Accounts: https://www.daytrading.com/options-strategies-small-accounts  
[6] ViperReport: Best Option Strategies for Small Accounts: https://www.viperreport.com/option-strategies-for-small-accounts/  
[7] Fidelity Vertical Spreads PDF: https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/learning-center/Deck_Vertical-spreads.pdf  
[8] Barchart.com: Strike Price Selection: https://www.barchart.com/education/strike_price_selection  
[9] Fidelity: Hitting the Right Strike Price: https://www.fidelity.com/viewpoints/active-investor/hitting-the-right-strike-price  
[10] OptionAlpha: How Many Trades to be Successful: https://optionalpha.com/blog/probability-theory-how-many-trades-to-be-successful  
[11] Cabot Wealth Network: High Probability Options Strategy: https://www.cabotwealth.com/daily/options-trading/high-probability-options-strategy-87-win-rate  
[12] Bulls on Wall Street: How to Manage Risk Trading a Small Account: https://www.bullsonwallstreet.com/post/how-to-manage-risk-trading-a-small-account  
[13] Optionstradingin21days.com: Small Account Options Trading - Realistic Strategies: https://www.optionstradingin21days.com/blog/small-account-options-trading-realistic-strategies  
[14] Tastylive Debit Spreads Guide: https://www.tastylive.com/concepts-strategies/debit-spreads  
[15] Fidelity: Technical Analysis for Options Trading (PDF): https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/learning-center/Deck_Technical-analysis-for-options.pdf