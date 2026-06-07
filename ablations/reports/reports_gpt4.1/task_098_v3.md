# Comprehensive, Tiered, Scenario-Driven Educational Framework for Options Trading with a $700 Account

## Introduction

This robust educational framework addresses options trading with a $700 account, explicitly targeting the aspirational (but mathematically untenable) goal of $200–$300 in weekly returns. All core requirements are addressed:
- Position sizing and risk management
- Permitted/prohibited strategies (precisely mapped to account/margin limits)
- Stock and options selection based on technical and liquidity filters
- Scenario-driven, parameterized, and tiered rulesets (conservative to aggressive)
- Hypothetical, detailed trade examples with outcome modeling and probability assessment
- Full win-rate, expected value, and feasibility analysis (using probability distributions)
- Contextualization against industry benchmarks, with realistic alternative targets and their quantitative rationale

Rules are explicitly parameterized and scenario-dependent, prioritizing discipline and risk control. Where preferences are not stated, the framework documents reasonable, industry-standard choices. This is a guide for educational purposes only and is not personal financial advice.

---

## Account Constraints and Performance Goals

- **Account size:** $700 cash
- **Stated target:** $200–$300/week (29–43%/week, >1000% annualized)
- **Industry-recommended target:** 0.5–1%/week ($3.50–$7.00) or 2–5%/month ($14–$35), per leading broker and research data
- **Implication:** Safely attaining $200–$300/week is impossible without breaking prudent risk, margin, and probability rules. The framework details why, and how to approach sustainable risk/reward instead.

---

## Tiered Position Sizing and Risk Management Rules

**Overview:** Position sizing is the foundation of risk control, especially in small accounts where even a single outsized loss can create catastrophic drawdowns.

### Parameterized, Tiered Sizing and Risk Limits

| Tier           | Maximum Risk per Trade | % of Account | Typical Position Size | Total Open Risk |
|----------------|-----------------------|--------------|----------------------|-----------------|
| Conservative   | $7–$14                | 1–2%         | 1 contract, $1-wide  | Max 2 positions (≤5%)   |
| Moderate       | $14–$21               | 2–3%         | 1 contract, $1-$2-wide| Max 3 positions (≤10%)  |
| Aggressive     | $28–$35               | 4–5%         | 1 contract, up to narrow $3-wide if allowed by margin, OCC, and broker| Max 14% (not recommended for small accounts)   |

- **Stop-loss (tight):**
    - Hard-exit at 100% of risk (credit spreads: max loss spread width minus premium)
    - Optionally, exit at 50% loss for conservative capital preservation
- **Drawdown control:**
    - Cease trading if account drops by 10% from peak ($70), review rules and methods before resuming
    - Never increase size after a loss or average down under any tier

_Sources: [1][2][3][4][5]_

---

## Permitted and Prohibited Strategies (Account and Margin Mapping)

### Explicitly Permitted Strategies (based on risk containment and account size)

| Strategy           | Allowed?         | Tier         | Rationale |
|--------------------|------------------|--------------|-----------|
| Vertical credit spreads (bull put, bear call) | Yes       | All tiers | Defined risk, lowest capital use; credit to margin balances |
| Vertical debit spreads (bull call, bear put)  | Yes       | All       | Defined risk, small outlay, directional bets      |
| Iron condors (narrow)                      | Conditional| Conservative/Moderate | Only if max loss <$35; rarely practical unless on ultra-liquid ETFs with $1-wide wings|
| Micro/mini contract spreads                | Yes (if broker offers) | All   | For strict risk targeting below $20              |

### Prohibited Strategies

| Strategy                   | Excluded | Rationale                                         |
|-----------------------------|---------|----------------------------------------------------|
| Naked/single-leg puts/calls | Yes     | Unlimited or large losses possible                 |
| Straddles/Strangles         | Yes     | High undefined risk, margin requirements exceed account |
| Ratio/Calendar/Diagonal spreads | Yes | Complex risk/margin profiles, premium, and execution failures possible       |
| Spreads with width > allowed per risk | Yes| Can exceed entire account value                   |
| Covered calls, naked puts, wheel | Yes | Requires >$2000 collateral for 100 shares of liquid stocks or ETF |

**Key: The only strategies which align strictly with account size and risk are defined-risk vertical (credit and debit) spreads in ultra-liquid, high-volume stocks or ETFs.**

_Sources: [4][5][6][7][8]_

---

## Technical Stock Selection Criteria (Explicit and Parameterized)

### Stock/ETF Technical Screens

| Parameter                  | Conservative        | Moderate/Aggressive         | Rationale and Ranges                    |
|----------------------------|--------------------|-----------------------------|-----------------------------------------|
| Price per share            | >$20               | >$10                        | Higher liquidity in higher-priced stocks|
| Underlying Volume          | >1,000,000 shares/day| >500,000 shares/day        | Lower slippage, easier fills            |
| OI per strike              | >1,000             | >200                        | More volume = less execution risk       |
| Option Volume              | >100 contracts/day | >50 contracts/day           | Ensures fills at or near mark           |
| Bid/Ask spread             | ≤$0.05 (ultra-liquid)| ≤$0.10                    | Tightest possible minimizes cost        |
| 20/50/200 day SMA          | Trending (e.g. price above both on bullish trade, below on bearish)| Mild trend ok           | Confirms direction, technical confirmation  |
| RSI (14)                   | 35–65 (avoid extreme levels) | 25–75 (if comfortable with mean reversion) | Avoids overbought/oversold extremes     |
| MACD/Volume confirmation   | Preferred          | Optional                    | Added confluence for higher-probability moves|
| Earnings/Catalyst window   | None in next 30 days| None within expiry period   | Avoid huge unexpected moves             |

_Both tiers strongly favor mega-cap stocks and core index ETFs (SPY, QQQ, IWM, XLF) for liquidity, cost, and low assignment risk._

_Sources: [6][9][10][11][12][13]_

---

## Options Liquidity and Execution Rules

| Filter               | Conservative Tier         | Moderate/Aggressive    |
|----------------------|--------------------------|------------------------|
| Minimum OI           | 1,000+ (SPY, QQQ, IWM)   | 200+                   |
| Option volume        | 100+ contracts per day   | 50+                    |
| Bid/Ask spread       | ≤$0.05 (target)          | ≤$0.10                 |
| Limit Orders         | Mandatory                | Mandatory              |
| Underlying volume    | >1,000,000/day           | >500,000/day           |
| % of daily series traded | <5% of visible size    | <15%                  |

- Avoid any illiquid or wide-spread option, as slippage can instantly erase profits or tilt actual risk above planned limits.

_Sources: [14][15][16][17][18]_

---

## Strike and Expiry Selection (Parameterized, Scenario-Dependent)

### Tiered Strike/Expiry Rules

| Tier           | Short Strike Delta | Spread Width | DTE at Entry | Close/Exit Rule | Profit Target Rule |
|----------------|-------------------|-------------|--------------|-----------------|-------------------|
| Conservative   | 0.15–0.20         | $1          | 30–45        | Close at 21 DTE or 50% max profit | 50% of max gain, whichever hit first |
| Moderate       | 0.20–0.25         | $1–$2       | 21–30        | At 75% profit/21 DTE | 50–75% max gain  |
| Aggressive     | 0.25–0.30         | $2–$3 (If margin allows, up to $35 risk) | 14–21 | At 90% profit/10 DTE | Try for max, use stop-loss discipline |

- Always collect at least 1/3 of spread width (e.g., $0.33 on $1-wide) for new positions (risk/reward >= 2:1).
- For debit spreads: Buy ATM or just ITM (delta 0.50–0.70), sell leg next OTM.
- Do not trade any contract with less than 7 days to expiry.

_Sources: [19][20][21][22][23][24][25]_

---

## Complete, Scenario-Driven Hypothetical Example Trades

### Example 1: SPY Bull Put Credit Spread (Conservative)

- **Underlying:** SPY at $500.00. Meets all volume/liquidity/technical criteria.
- **Trade Structure:**  
    - Sell 1x SPY $495 Put (delta 0.15)
    - Buy 1x SPY $494 Put
    - Expiry: 30 days out
    - Net credit: $0.33 ($33 per contract, 1/3 rule)
    - Spread width: $1
    - **Max loss:** $1 - $0.33 = $0.67 ($67 per contract; this is just below 10% of account, so only one open trade at a time for true $700 tier)
    - **Breakeven:** $495 - $0.33 = $494.67
    - **Position size:** 1 contract
- **Entry:** $33 credit received, $67 at risk

#### Multi-Scenario Outcomes

| SPY at Expiry           | Outcome                      | Net P/L |
|------------------------ |-----------------------------|---------|
| >$495                   | Full profit                  | +$33    |
| $494.67–$495            | Partial loss                 | $33 - (difference between expiry price and $495)  |
| <$494                   | Max loss                     | -$67    |

- **Close early**: If price rises and spread falls to $0.16, close for $16 profit (lock in 50% of potential).
- **If SPY declines below $495,** position can be managed/closed to avoid max loss, but only if liquidity and fill quality allow.

### Example 2: QQQ Bear Call Credit Spread (Moderate/Aggressive)

- **Underlying:** QQQ at $430, technicals neutral/rolling over, heavy resistance at $435, high OI/volume.
- **Trade Structure:**  
    - Sell 1x QQQ $435 Call (delta 0.21)
    - Buy 1x QQQ $437 Call
    - Expiry: 21 days
    - Net credit: $0.55 ($55 per contract, for $2-wide; slightly rich for increased risk)
    - **Max loss:** $2 - $0.55 = $1.45 ($145 at risk; this is over the ideal for $700 account, so size down or seek $1-wide equivalent)
    - **Position size:** 1 contract ($1 wide would be $0.28 credit/$0.72 risk; better matched to account limits)
- **Breakeven:** $435 + $0.55 = $435.55

#### Multi-Scenario Outcomes

| QQQ at Expiry           | Outcome                  | Net P/L |
|------------------------ |-------------------------|---------|
| <$435                   | Full profit               | +$28 ($1-wide)      |
| $435–$436               | Partial loss             | $28 - (difference between expiry price and $435)  |
| >$436                   | Max loss                 | -$72    |

- Take profit at 50–75% gain, or close at 50% loss if price reversal looms.

---

## Probability Calculations and Required Win Rates

### Expected Value & Probability Modeling

#### Conservative Spread Example

- **Risk per trade:** $67, **potential gain:** $33, **PoP** (Probability of Profit): ~70%
- **Expected Value (EV) per trade:** (0.70 x $33) - (0.30 x $67) = $23.10 - $20.10 = **$3.00**
- **Break-even win rate:** $67 / ($33 + $67) = **67%** minimum needed just to remain flat (before costs).
- **To earn $200/week:**  
    - Need ~7 full-win trades per week, risking nearly all capital multiple times
    - With conservative sizing, would require >20 trades/potential max open risk of $1,500+ (which is impossible for $700)

#### Binomial Model: Probability of Achieving $200 in 10 Trades

- With 70% win rate and +$33 per win/-$67 per loss:
    - Outcomes after 10 trades:
        - Probability of 7+ wins: ~25%
        - But, even with 7 wins/3 losses: (7 x $33) - (3 x $67) = $231 - $201 = $30 total, massively under target.
- **Conclusion:** $200/week cannot be reached probabilistically; even the best possible runs will fall short without violating all risk/position sizing discipline.

_Sources: [3][4][8][26][27][28][29]_

---

## Contextualizing Against Industry Benchmarks

- **Average sustainable return for active defined-risk spread traders:** 20–40% annualized (~0.5–1%/week, or $3.50–$7.00 per week for $700)
- **Most successful small account traders target <$7/week as realistic ceiling for compounded growth** [4][5][6][25][30].
- **Professional or leveraged trading for higher returns:** Relies on breaking risk controls or access to larger margin; accounts commonly “blow up” before ever doubling [7][8][30].
- **Empirical result:** Any attempt to reach $200–$300 per week in a $700 account is virtually guaranteed to fail due to position-sizing, risk, and win-rate math, compounded by fees and slippage.

---

## Alternative, Realistic Targets and Adjusted Rules

### Realistic Target

| Account Size | Sustainable Weekly Target | Annualized Return    |
|--------------|--------------------------|---------------------|
| $700         | $3.50–$7.00              | 26–52%              |
| $7,000       | $35–$70                  | 26–52%              |

- Requires strict risk protocols, mechanical trade process, loss discipline, and focus on compounding over time.
- Emphasize education, skill-building, and psychological resilience rather than “lotto ticket” mindset [3][4][8][30][31].

---

## Rule Documentation and Trade Process Discipline

- **Document every trade:** Entry/exit reason, risk, target, stop, outcome, and lessons learned
- **Never break max risk rule:** $35/trade absolute cap (preferably $14–$21)
- **Use a checklist:** For each trade, confirm all liquidity, technical, expiry, strike, and catalyst filters are met
- **Continuous review:** Pause trading after 10% monthly drawdown, revise failures/weaknesses

---

## Conclusion

A $700 account cannot reliably and safely yield $200–$300 per week with defined-risk options strategies; this is validated by industry benchmarks, probability models, and real-world constraints. However, by following a strict, tiered, scenario-driven ruleset as documented above, small account traders can maximize their learning, maintain capital, and potentially compound realistic returns over time. Prioritizing education and capital preservation will put new traders in the minority that persists—and eventually grows.

---

### Sources

[1] How To Trade Options In Small Accounts - INO.com Trader's Blog: https://www.ino.com/blog/2019/11/how-to-trade-options-in-small-accounts/  
[2] Option Alpha - Position Sizing Guide: https://optionalpha.com/blog/position-sizing-what-everybody-ought-to-know  
[3] Interactive Brokers - Generating Income with Credit Vertical Spreads: https://www.interactivebrokers.com/campus/traders-insight/securities/options/generating-income-with-credit-vertical-spreads/  
[4] Optionstradingiq - Best Option Strategies for Small Accounts: https://optionstradingiq.com/best-option-strategies-for-small-accounts/  
[5] PurePowerPicks - Options Trading Risk Management: https://purepowerpicks.com/options-trading-risk-management-tips/  
[6] Option Alpha - Trading Small Accounts (Video): https://optionalpha.com/videos/how-i-would-start-trading-options-with-3-000  
[7] Tastytrade - Trading Account Levels and Allowable Strategies: https://support.tastytrade.com/support/s/solutions/articles/43000435222  
[8] Option Alpha - Probability Theory: https://optionalpha.com/blog/probability-theory-how-many-trades-to-be-successful  
[9] TradeVision - Effective Option Trading Using Technical Analysis: https://tradevision.io/blog/effective-option-trading-using-technical-analysis/  
[10] Timothy Sykes - Guide to Trading Options with a Small Account: https://www.timothysykes.com/blog/small-account-trading/  
[11] TradeVision - Options Trading Strategy: https://tradevision.io/blog/options-trading-strategy-boost-with-technical-analysis/  
[12] Simpler Trading - Asset Liquidity: https://my.simplertrading.com/news/asset-liquidity-in-trading  
[13] Option Alpha - SPY Put Credit Spread Backtest: https://optionalpha.com/blog/spy-put-credit-spread-backtest  
[14] TradingBlock - Options Liquidity: https://www.tradingblock.com/blog/options-liquidity  
[15] Tastylive - Options Liquidity: https://www.tastylive.com/concepts-strategies/options-liquidity  
[16] HeyGoTrade - Options Liquidity Explained: https://www.heygotrade.com/en/blog/options-liquidity-explained/  
[17] Interactive Brokers - Open Interest: https://www.interactivebrokers.com/campus/trading-lessons/open-interest/  
[18] AdvancedAutoTrades - Credit Spread Examples: https://advancedautotrades.com/credit-spread-examples/  
[19] Option Visualizer - Strike Price Selection Guide: https://www.optionvisualizer.com/documentation/strategies/strike-selection  
[20] Fidelity - Selecting a Strike Price and Expiration Date: https://www.fidelity.com/learning-center/investment-products/options/selecting-strike-price-expiration-date  
[21] Barchart - Strike Price Selection: https://www.barchart.com/education/strike_price_selection  
[22] TheOptionPremium - Credit Spread Calculator: https://www.theoptionpremium.com/p/credit-spread-calculator-returns-risk-reward  
[23] Option Alpha - SPY Put Credit Spread Backtest: https://optionalpha.com/blog/spy-put-credit-spread-backtest  
[24] Tradeoptionswithme - IWM Credit Spread Example: https://tradeoptionswithme.com/live-trade-examplescase-studies/iwm-credit-spread/  
[25] Reddit - Strike and Expiry Criteria: https://www.reddit.com/r/options/comments/1go0ejo/a_simple_quantitative_method_for_choosing_strike/  
[26] Reddit - Performance Benchmark for Options: https://www.reddit.com/r/options/comments/iftp5c/is_there_a_wellaccepted_performance_benchmark_for/  
[27] Option Alpha - How Many Trades To Be Successful: https://optionalpha.com/blog/probability-theory-how-many-trades-to-be-successful  
[28] InsiderFinance - Maximize Profits with Strike and Expiration: https://www.insiderfinance.io/resources/how-to-maximize-profits-with-strike-and-expiration  
[29] Option Alpha - Probability Theory: https://optionalpha.com/blog/probability-theory-how-many-trades-to-be-successful  
[30] Yahoo Finance - Trade Options for Income with a Small Account: https://finance.yahoo.com/news/trade-options-income-small-account-142108675.html  
[31] Option Alpha - Trading Psychology and Process Returns: https://optionalpha.com/blog/options-trading-psychology-process-returns