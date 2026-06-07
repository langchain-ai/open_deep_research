# Detailed Options-Trading Framework for a $700 Account Targeting $200–300 Weekly Returns

This report provides a comprehensive, well-structured options-trading framework tailored for a small retail trading account of $700, aiming to generate consistent weekly returns between $200 and $300. The framework covers critical elements including position sizing, permitted trading strategies, stock selection criteria via technical analysis, option strike and expiry selection for short- to medium-term holds, liquidity filters to ensure trade feasibility, detailed hypothetical trade examples, and analysis of required win rates and probability distributions.

---

## 1. Position-Sizing Rules

Position sizing is the foundation for managing risk and sustaining growth in a small trading account.

- **Maximum Risk per Trade**: To protect capital and reduce the risk of ruin, it is widely recommended to risk no more than **1-3%** of the total account equity per trade. For a $700 account, this translates to a maximum loss of approximately **$14 (2% of $700)** per trade. This conservative approach balances growth potential with capital preservation.

- **Risk Consistency**: Maintain a consistent maximum risk per trade to avoid large drawdowns. Avoid increasing trade size impulsively as profits grow, instead scale manually and gradually.

- **Strategy Impact on Position Size**:  
  - For long option positions (buying calls or puts), total risk equals the premium paid, capped by the position size.
  - For defined-risk spreads (credit or debit), maximum loss is based on the spread width minus premiums received or paid. Position size must adjust accordingly to keep risk under the 2% guideline.

- **Portfolio Delta Exposure**: Keep total portfolio directional (delta) exposure under 50% of the account value to avoid excessive directional risk.

- **Use of Stop Losses**: Apply stop-loss rules based on maximum tolerable loss or technical triggers.

**Summary**: For a $700 account, limiting risk to about $10–$14 per trade is essential. This constraint often requires trading fewer contracts or smaller-sized spreads.

---

## 2. Permitted Option Strategies and Their Rationale

Selecting option strategies that fit small accounts and the risk framework is critical. The focus is on strategies with defined risk and high probability of profit.

### a) Credit Spreads (Bull Put and Bear Call Spreads)

- **Definition**: Selling an OTM option and buying a further OTM option of the same type to limit risk.
- **Rationale**: Known maximum loss, margin-efficient, and profit from time decay (theta). They combine high probability outcomes with controlled risk.
- **Typical Use**: Bull put spreads for bullish or neutral bias; bear call spreads for bearish/neutral bias.
- **Advantages**:  
  - Limited risk per trade, defined upfront.  
  - Profit from time decay and volatility contraction.  
  - Generally, win rates of 70–80% in appropriately selected strikes.

### b) Debit Spreads (Bull Call, Bear Put Spreads)

- **Definition**: Buying one option and selling another closer to or out-of-the-money to reduce net premium paid.
- **Rationale**: Directional plays with limited risk equal to the net debit.  
- **Advantages**: Limited max loss (debit paid), reduced capital outlay compared to naked options.  
- **Considerations**: More reliant on correct directional move; time decay negatively affects debit spreads, so strike and expiry choice is key.

### c) Single-Leg Calls or Puts (Directional)

- **Rationale**: Simple to understand and trade with limited risk—loss limited to premium paid.
- **Drawbacks**: Vulnerable to time decay and volatility. Generally require a strong directional conviction and favorable entry.

### d) Optional but Recommended (If Capital Permits):

- **Iron Condors**: Combination of bull put and bear call spreads, market neutral, high probability of profit but often requires more capital.
- **Cash-Secured Puts and Covered Calls**: For those willing to own or sell stock, these are conservative income-generation strategies but may require more capital.

**Summary**: Credit and debit spreads are prioritized due to their defined risk profile and appropriateness for a small account. Single-leg options can be used sparingly for directional plays with proper risk management.

---

## 3. Criteria for Selecting Underlying Stocks Based on Technical Analysis

Successful option trades depend heavily on selecting suitable underlying assets with favorable technical setups.

### a) Volume and Liquidity

- Choose stocks with **high daily average volume** (ideally millions of shares traded daily).
- High volume confirms strong interest and reduces slippage risk.
- Volume supports validity of support and resistance levels.

### b) Support and Resistance Levels

- Identify clear **support** (price floors) and **resistance** (price ceilings) by examining past price action and patterns.
- Use technical tools such as:
  - Horizontal price levels (previous lows/highs).
  - Moving averages (e.g., 20, 50 SMA) as dynamic support/resistance.
  - Fibonacci retracement levels for price reaction zones.

### c) Momentum Indicators

- Use measurements of momentum, including:
  - Relative Strength Index (RSI): Look for oversold (RSI <30) or overbought (>70) conditions.
  - Moving Average Convergence Divergence (MACD): Crossovers to confirm trend momentum.
  - On-Balance Volume (OBV): Confirms buying or selling pressure.
- Momentum helps time entry and exit points.

### d) Breakouts and Trend Confirmation

- Identify breakouts above resistance or below support with volume confirmation.
- Use confirmed breakouts to employ directional option strategies like debit spreads or single-leg options.

**Summary**: Select liquid, well-traded stocks showing clear technical signals of support/resistance and momentum to time trades. These criteria help reduce risk and improve probability of success.

---

## 4. Option Strikes and Expiry Selection Rules (Holding Period 3–21 Days)

### a) Strike Selection

- **Directional Strategies (Debit Spreads and Single Legs):**  
  - Select strikes near the money (ATM or slightly ITM), corresponding roughly to deltas of **0.50 to 0.60**. This balances probability and premium cost while reducing time decay impact.
- **Credit Spreads (Premium Selling):**  
  - Select strikes further OTM (sell ~0.30 to 0.40 delta options) and hedge with further OTM options (buy ~0.15 to 0.25 delta).  
  - This increases probability of selling options that expire worthless, capturing premium.

### b) Expiration Selection

- For retail traders with small accounts seeking weekly returns, expirations between **3 and 21 days** are ideal as this window maximizes theta decay while limiting risk from gamma (price sensitivity) and vega (volatility).
- **Credit spreads:** Typically opened 30–45 days out and closed with 3–21 days remaining to optimize risk/reward.
- **Debit spreads:** Use expirations on the shorter side (3–21 days) to keep capital locked for less time while maintaining time premium.

### c) Holding Period Management

- Hold trades typically from **3 to 21 calendar days**, aiming to either close at target profit or cutoff loss before expiration.
- Avoid holding through major earnings or news events to limit volatility risk.

**Summary**: For small accounts, choose strikes and expiry that maximize high probability of small-to-moderate gains within 3 to 21-day windows, balancing time decay and directional bias to avoid sudden large losses.

---

## 5. Liquidity Filters to Ensure Trade Feasibility

Adequate liquidity is necessary to minimize slippage, reduce transaction fees, and ensure easy entry/exit.

- **Minimum Open Interest:**  
  - Select option contracts with at least **100 open contracts** to ensure active market participation.
- **Bid-Ask Spread:**  
  - Favor options with **bid-ask spreads less than $0.30**, ideally tighter. Wide spreads increase cost and reduce profit potential.
- **Volume:**  
  - Prefer strikes with daily trading volume above 300 contracts, where possible.
- **Underlying Liquidity:**  
  - Trade options on underlying stocks or ETFs known for liquid option markets such as SPY, QQQ, AAPL, MSFT.
- **Order Execution:**  
  - Use limit orders to control fill price and avoid market slippage.
- **Evaluate Volatility:**  
  - Avoid illiquid or low implied volatility strikes where spreads widen significantly.

**Summary**: Enforce liquidity filters to ensure feasible and cost-effective trades, critical for small accounts where transaction costs disproportionately impact returns.

---

## 6. Hypothetical Example Trades Illustrating the Framework

### Example Trade 1: Bull Put Credit Spread (Defined Risk, Credit Strategy)

| Parameter           | Details                                             |
|---------------------|-----------------------------------------------------|
| Underlying          | Stock XYZ trading at $100                            |
| Technical Rationale | Near support at $98, RSI oversold, volume 1M+       |
| Option Chain        | Sell 1 XYZ 3-week $97 put at $1.50; Buy 1 XYZ $95 put at $0.50 |
| Net Credit          | $1.00 per share = $100 total                         |
| Position Size       | 1 contract (to respect max risk ~2%)                 |
| Max Risk            | $2.00 spread - $1.00 credit = $1.00 x 100 = $100    |
| Scaled Position     | To keep max risk under $14, trade 0.14 contracts (rounded to 1 for example educational purposes with capital caution) |
| Target Profit       | 50–75% of max credit = $50 to $75                    |
| Holding Period      | 3 to 21 days                                        |

**Profit / Loss Scenarios**

- Stock price > $97 at expiry: Max profit $100 (the credit received)  
- Stock price < $95 at expiry: Max loss $100 (spread width less premium)  
- Stock price between $95 and $97: Partial credit retained, partial loss (linear)  

**Notes**: Position size should be adjusted down or margin used accordingly to meet $14 max loss per trade limit.

---

### Example Trade 2: Bull Call Debit Spread (Directional, Limited Risk)

| Parameter           | Details                                             |
|---------------------|-----------------------------------------------------|
| Underlying          | Stock ABC trading at $50                             |
| Technical Rationale | Breakout above resistance $49, RSI ~55, strong volume |
| Option Chain        | Buy 1 ABC 3-week $51 call at $1.80; Sell 1 ABC 3-week $54 call at $0.80 |
| Net Debit           | $1.00 per share = $100 total                         |
| Position Size       | 1 contract (consistent with $14 max loss)           |
| Max Loss            | $100 (net debit)                                    |
| Max Profit          | ($54 - $51) = $3.00 - $1.00 = $2.00 x 100 = $200   |
| Target Profit       | 50–70% max profit = $100 to $140                     |
| Holding Period      | 3 to 21 days                                        |

**Profit / Loss Scenarios**

- Stock price > $54 at expiry: Max profit $200  
- Stock price ≤ $51 at expiry: Max loss $100  
- Price between $51 and $54: Partial profit or loss

---

## 7. Win-Rate and Probability Distribution Required for Target Weekly Returns

### Analytical Considerations

- **Account Size**: $700  
- **Weekly Target**: $200–300 (~28–43% weekly gain) — ambitious for consistent success, requiring aggressive and disciplined trading.

### Position Sizing & Expected Returns

- With 2% max risk per trade ($14), target profit per trade should be 50-75% of risk ($7-$10).  
- To hit $200 weekly, approximately **20-30 winning trades** at $7-$10 gain or fewer with larger wins needed weekly.  
- Given typical weekly trading frequency, multiple trades per week are required.

### Required Win Rate

- High-probability strategies such as credit spreads offer **70-80% win rates** on individual trades.  
- A win rate of **~75% with an average reward-to-risk ratio of about 0.7 to 1** supports steady gains.  
- Losing trades capped at 1-2% risk limit ensure drawdowns remain manageable.

### Probability Distribution

- The majority of trades result in small profits (~50-75% of max credit).  
- Occasional losses occur due to adverse price movement, requiring strict adherence to stop-loss and position sizing rules.  
- Use of multiple small, consistent gains over several trades a week compounds returns towards target.  

### Realistic Expectations

- Aggressive targets require disciplined execution, patience, and acceptance of losing streaks.  
- Small account growth is non-linear and requires compounding gains over months more than weeks.  
- Theoretical models like the Law of Large Numbers support success given sufficient trades and consistent edge.

---

## Conclusion

Managing a $700 trading account with a weekly return target of $200–300 requires disciplined position sizing, selection of high-probability, defined-risk option strategies (primarily credit and debit spreads), meticulous underlying selection based on technicals, and adherence to strict liquidity filters. Position sizes must be kept small to respect risk limits, often involving just 1 contract or fractional equivalents for education.

By focusing on high-liquidity options on technically sound stocks and/or ETFs, choosing strikes and expirations aligned with a 3-21 day holding framework, and executing trades with tight risk controls, a retail trader can aim for consistent returns while managing downside risk.

Realistic achievement of these aggressive weekly returns depends on maintaining a win rate of approximately 70-80%, leveraging high-probability spreads, and trading multiple contracts/times per week under strict risk controls. While challenging, this framework emphasizes educational best practices for retail options traders looking to grow small accounts responsibly.

---

## Sources

[1] Best Option Strategies for Small Accounts: A Premium Seller's Guide: https://optionstradingiq.com/best-option-strategies-for-small-accounts/  
[2] How To Trade Options In Small Accounts - INO.com Trader's Blog: https://www.ino.com/blog/2019/11/how-to-trade-options-in-small-accounts/  
[3] What Everybody Needs to Know About Proper Position Sizing - OptionAlpha: https://optionalpha.com/blog/position-sizing-what-everybody-ought-to-know  
[4] What is Position Sizing in Options Trading? & Why It's Key - Next Level Global Academy: https://www.nextlevelglobalacademy.com/blog-posts/position-sizing-options-trading  
[5] Position Sizing – Part 3 | Sizing Option Strategies the Right Way - IntheMoney by Zerodha: https://inthemoneybyzerodha.substack.com/p/position-sizing-part-3-sizing-option  
[6] How to Grow a Small Account (Using Options) - YouTube: https://www.youtube.com/watch?v=7IHCmruEZUk  
[7] OptionsPlay Client First - Growing Small Accounts: https://www.optionsplay.com/blogs/how-to-grow-a-small-account  
[8] OptionsPlay Client First - Strike & Expiration Selection: https://www.optionsplay.com/blogs/optimal-expiration-dates-and-strike-prices  
[9] Selecting a Strike Price and Expiration Date - Fidelity: https://www.fidelity.com/learning-center/investment-products/options/selecting-strike-price-expiration-date  
[10] How to Pick the Right Strike Price in Options Trading | tastylive: https://www.tastylive.com/concepts-strategies/how-to-pick-the-right-strike-price  
[11] Options Liquidity: A Complete Guide for Traders | tastylive: https://www.tastylive.com/concepts-strategies/options-liquidity  
[12] Options Trading Liquidity: Volume, Open Interest, Size & More | TradingBlock: https://www.tradingblock.com/blog/options-liquidity  
[13] Options 101: Bid/Ask, Open Interest and Volume | Tackle Trading: https://tackletrading.com/options-101-bidask-open-interest-and-volume/  
[14] The bid/ask spread and how it affects options buyers and sellers - Unusual Whales: https://unusualwhales.com/information/the-bid-ask-spread-and-how-it-affects-options-buyers-and-sellers  
[15] Technical Analysis for Options Trading - Fidelity Investments (PDF): https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/learning-center/Deck_Technical-analysis-for-options.pdf  
[16] Technical Analysis 101: Understanding Support and Resistance - Financial Modeling Prep: https://site.financialmodelingprep.com/market-news/technical-analysis--understanding-support-and-resistance  
[17] Trading Volume Analysis: A Guide to Market Momentum - Trade with the Pros: https://tradewiththepros.com/trading-volume-analysis/  
[18] Options Expiration: A Complete Guide for Traders - Trade with the Pros: https://tradewiththepros.com/options-expiration/  
[19] The High-Probability Options Strategy With an 80.4% Win Rate - Cabot Wealth Network: https://www.cabotwealth.com/daily/options-trading/high-probability-options-strategy-87-win-rate  
[20] How Many Trades Does it Take to be Successful? - Option Alpha: https://optionalpha.com/blog/probability-theory-how-many-trades-to-be-successful  

---

This framework supports retail traders in developing structured, risk-controlled options trading tailored to small accounts and ambitious targets, emphasizing education and discipline.