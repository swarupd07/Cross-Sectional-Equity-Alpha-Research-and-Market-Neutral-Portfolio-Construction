# Cross-Sectional Equity Alpha Research & Market-Neutral Portfolio Construction

A compact research project on whether simple cross-sectional equity signals can produce market-neutral long-short returns after realistic portfolio constraints and transaction costs.

The project uses **real NSE daily OHLCV data**, a **21-trading-day rebalance / forward-return horizon**, walk-forward ML evaluation, sector neutralization, covariance-aware portfolio optimization, and transaction costs.

## Research Question

> Can weak cross-sectional signals be combined into a stable market-neutral equity strategy, and do ML-based combinations improve on a simple fixed signal blend?

## Workflow

```text
NSE OHLCV data
      ↓
Point-in-time signals
Momentum + Volatility + Low Liquidity
      ↓
Signal validation
IC / ICIR / quantile returns
      ↓
Alpha combination
Fixed linear blend / Ridge / XGBoost
      ↓
Sector neutralization
      ↓
252-day covariance estimate
      ↓
Constrained long-short optimization
      ↓
21-day walk-forward backtest
      ↓
Transaction costs + factor attribution
```

## Signal Research

| Signal | Mean IC | ICIR | Bottom Q | Top Q |
|---|---:|---:|---:|---:|
| 12M log-price slope | 0.0274 | 0.113 | 1.6869% | 1.9368% |
| Volatility | 0.0301 | 0.152 | 1.3160% | 2.2402% |
| Low Liquidity | **0.0646** | **0.381** | 1.0123% | 2.6142% |

None of the signals produced perfectly monotonic quintile returns, but all showed positive average cross-sectional information at the 21-day horizon. **Low Liquidity was the strongest standalone signal.**

A 42-day reversal signal produced **IC 0.0369** and **ICIR 0.175**, but ablation testing showed that including it reduced portfolio performance. Signal correlations were low, so the weak incremental contribution was not mainly due to redundancy. The final production signals were therefore kept to **Momentum, Volatility, and Low Liquidity**.

## Portfolio Results

All returns are **net of transaction costs**.

| Method | Ann. Return | Ann. Vol | Sharpe | Max DD | Avg Turnover | Avg Gross |
|---|---:|---:|---:|---:|---:|---:|
| Fixed linear blend | **8.78%** | 8.08% | **1.08** | -11.76% | 0.50 | 2.00 |
| Ridge | 1.67% | 7.87% | 0.25 | -17.25% | 0.83 | 1.99 |
| XGBoost | 0.50% | 6.97% | 0.11 | -15.79% | 1.15 | 1.99 |

The fixed blend generalized much better than the ML combinations. Ridge and XGBoost produced higher turnover and did not extract enough stable incremental signal to justify their added flexibility.

## Covariance-Risk Experiment

The original optimizer used the concentration proxy

**wᵀw**

and was upgraded to portfolio variance

**wᵀΣw**

using a **252-day trailing sample covariance matrix**.

For the fixed linear strategy:

- Sharpe improved from **0.69 → 1.08**
- Annualized return improved from **5.26% → 8.78%**
- Annualized volatility stayed near **8%**
- Max drawdown remained near **-12%**

Risk-aversion sensitivity:

| Risk Aversion | Return | Vol | Sharpe |
|---:|---:|---:|---:|
| 1 | 8.78% | 8.10% | 1.083 |
| 5 | 8.78% | 8.08% | 1.084 |
| 10 | 8.78% | 8.04% | 1.090 |
| 25 | 7.82% | 7.78% | 1.009 |
| 50 | 6.54% | 7.37% | 0.898 |

I retain **risk_aversion = 5** because values 5 and 10 perform almost identically, avoiding tuning to a marginal Sharpe improvement.

## Factor Exposure

For the best fixed-blend strategy:

- Factor-adjusted annualized intercept: **6.78%**
- Market beta: **0.020**
- Momentum beta: **0.181**
- Volatility beta: **-0.011**
- Low-liquidity beta: **0.066**
- R²: **0.196**

The near-zero market beta is consistent with the intended market-neutral design. The strategy retains some momentum exposure, while most return variation is not explained by the included factors.

## Sector Neutralization

| Variant | Ann. Return | Sharpe | Max DD |
|---|---:|---:|---:|
| Without sector neutralization | 11.08% | 1.06 | -10.54% |
| With sector neutralization | 8.78% | **1.08** | -11.76% |

Sector neutralization reduces raw return but leaves risk-adjusted performance nearly unchanged while making the alpha interpretation cleaner by limiting sector bets.

## Design Decisions

- **21-day horizon only:** all signals, labels, factors, and backtests use the same 21-trading-day horizon.
- **Walk-forward ML:** models are retrained using only historical information available by the rebalance date.
- **Purged labels:** training excludes labels that would use information after the rebalance date.
- **Sector-neutral alpha:** sector effects are removed before portfolio construction.
- **Market-neutral portfolio:** weights satisfy approximately `sum(w) = 0`.
- **Gross exposure cap:** `sum(abs(w)) ≤ 2`.
- **Per-name cap:** absolute position size is limited to 5%.
- **Sector exposure cap:** sector net exposures are constrained.
- **Risk model:** 252-day trailing sample covariance.
- **Costs:** 2 bps brokerage + 5 bps half bid-ask spread per unit turnover.
- **Sharpe risk-free rate:** 0, consistent with an approximately self-financing long-short strategy.

## Main Conclusion

The experiment suggests that **simple, weak signals can become useful when combined with disciplined portfolio construction**.

> A stable fixed signal blend + sector controls + covariance-aware optimization outperformed Ridge and XGBoost out of sample.

The result supports the idea that, in noisy financial data, lower-variance modeling choices can generalize better than more flexible estimators.

## Assumptions & Limitations

- The universe uses a **fixed modern NSE constituent set**, so historical results may contain **survivorship bias**.
- The data is not a full point-in-time constituent database.
- Costs include brokerage and half-spread only; **slippage, market impact, borrow fees, and short availability** are not modeled.
- The covariance model is a simple trailing sample covariance rather than a full institutional factor-risk model.
- Turnover compares target weights across rebalance dates and does not explicitly model weight drift between rebalances.
- SLSQP occasionally reaches its iteration limit because of the non-smooth gross-exposure constraint.
- Results come from one equity universe and historical period and should not be interpreted as deployable alpha without further validation.

## Further Extensions

1. **Add more signal features** — quality, value, earnings revisions, residual momentum, volume/flow, and alternative liquidity measures.
2. **Use a point-in-time historical universe** to reduce survivorship bias and test robustness across changing constituents.
3. **Improve execution and risk modeling** with borrow costs, market impact, drift-aware turnover, covariance shrinkage/factor risk models, and an untouched final holdout period.

## Project Structure

```text
Cross-Sectional-Equity-Alpha-Research/
│
├── data.py
├── features.py
├── analysis.py
├── model.py
├── portfolio.py
├── backtest.py
├── run_research.py
├── test_signal_testing.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── nifty50_real.csv
    ├── research_report.md
    ├── equity_curve.png
    ├── drawdown.png
    └── ic_timeseries.png
```

- `data.py` — NSE OHLCV download, loading, universe and sector mapping
- `features.py` — momentum, volatility, reversal and low-liquidity signals
- `analysis.py` — IC, ICIR, quantile returns, turnover and factor attribution
- `model.py` — Ridge and XGBoost alpha models
- `portfolio.py` — sector neutralization, covariance-aware optimization and trading costs
- `backtest.py` — 21-day walk-forward backtest engine
- `run_research.py` — end-to-end experiment and report generation
- `test_signal_testing.py` — sandbox for testing new candidate signals
- `outputs/` — data, report and generated figures

## AI Assistance

Used AI as a **assistant** for implementation support, debugging, and documentation. Research questions, experiment design choices, interpretation of results, and final methodological decisions were made by me.
