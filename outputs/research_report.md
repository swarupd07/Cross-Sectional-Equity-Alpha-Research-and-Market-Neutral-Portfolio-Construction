# Cross-Sectional Equity Alpha — Research Report

Universe: real NSE (India) daily OHLCV data. 21-day holding period / forward return horizon throughout.

## 1. Signal-by-Signal Research

| Signal | Mean IC | ICIR | Bottom Q return | Top Q return | Monotonic? |
|---|---:|---:|---:|---:|---|
| log_price_slope | 0.0274 | 0.113 | 1.6869% | 1.9368% | no |
| Volatility | 0.0301 | 0.152 | 1.3160% | 2.2402% | no |
| Low Liquidity | 0.0646 | 0.381 | 1.0123% | 2.6142% | no |

_Best standalone signal by |ICIR|: **Low Liquidity**. See `ic_timeseries.png`._

## 2. Alpha Combination: Linear vs ML, Net of Costs

| Method | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Avg Turnover | Avg Gross |
|---|---:|---:|---:|---:|---:|---:|
| Linear combo (fixed weights) | 8.78% | 8.08% | 1.08 | -11.76% | 0.50 | 2.00 |
| Ridge regression | 1.67% | 7.87% | 0.25 | -17.25% | 0.83 | 1.99 |
| XGBoost | 0.50% | 6.97% | 0.11 | -15.79% | 1.15 | 1.99 |

_Best method by net Sharpe: **Linear combo (fixed weights)**._

## 3. Factor Exposure

Regressing **Linear combo (fixed weights)**'s net 21-day returns on market, momentum, volatility, and Low liquidity factors:

- Annualized regression intercept (return unexplained by included factors): **6.78%**
- Beta to market: **0.020**
- Beta to momentum: **0.181**
- Beta to volatility: **-0.011**
- Beta to Low liquidity: **0.066**
- R²: **0.196**

The intercept represents return not explained by the included market and style factors; it should be interpreted as factor-adjusted alpha rather than proof of unique alpha.

## 4. Effect of Sector Neutralization

| Variant | Ann. Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Without neutralization | 11.08% | 1.06 | -10.54% |
| With sector neutralization | 8.78% | 1.08 | -11.76% |

## 5. Notes on Methodology

- All features are point-in-time correct (computed only from data known as of date t).
- ML models are fit on a rolling historical window using no information after the rebalance date, with a purge gap equal to the holding period.
- Alpha is neutralized against sector exposure via cross-sectional regression before portfolio construction.
- Portfolio weights are solved via constrained optimization (dollar-neutral, per-name position cap, per-sector exposure cap, covariance-based portfolio risk and quadratic turnover penalties).
- Reported returns are net of a linear transaction-cost model (brokerage + half bid-ask spread, in bps per unit turnover).
- Gross exposure is capped at 2.0 by the optimizer; realized gross exposure may be lower and is reported as the average \(\sum_i|w_i|\) across rebalance dates.
