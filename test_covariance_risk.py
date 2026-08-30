import pandas as pd

from analysis import turnover_timeseries
from backtest import BacktestConfig, run_backtest, summary_stats
from data import load_ohlcv_from_csv


panel = load_ohlcv_from_csv(r"outputs\nifty50_real.csv")

for risk_aversion in [1, 5, 10, 25, 50]:
    result = run_backtest(panel, BacktestConfig(alpha_mode="linear_combo", risk_aversion=risk_aversion))
    stats = summary_stats(result.to_frame()["net_return"])
    turnover = turnover_timeseries(result.weights_history).mean()
    gross = pd.Series({d: w.abs().sum() for d, w in result.weights_history.items()}).mean()

    print(
        f"risk_aversion={risk_aversion:<2} | Return={stats['annualized_return']:.2%} | "
        f"Vol={stats['annualized_vol']:.2%} | Sharpe={stats['sharpe']:.3f} | "
        f"MDD={stats['max_drawdown']:.2%} | Turnover={turnover:.3f} | Gross={gross:.3f}"
    )
