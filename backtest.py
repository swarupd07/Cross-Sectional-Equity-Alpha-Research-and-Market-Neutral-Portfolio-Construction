from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import to_wide_close
from features import log_price_slope, compute_volatility, compute_low_liquidity
from model import build_dataset, LinearAlphaModel, XGBoostAlphaModel
from portfolio import neutralize_cross_section, compute_trade_costs, optimize_weights


PERIODS_PER_YEAR = 252 / 21


@dataclass
class BacktestConfig:
    holding_period: int = 21
    train_years: float = 2.0
    min_train_days: int = 252

    alpha_mode: str = "linear_combo"
    combo_weights: dict = field(default_factory=lambda: {
        "momentum": 0.5, "volatility": 0.25, "Low liquidity": 0.25
    })

    max_position: float = 0.05
    sector_tol: float = 0.10
    gross_target: float = 2.0
    risk_aversion: float = 5.0
    cost_aversion: float = 2.0
    cov_lookback: int = 252

    brokerage_bps: float = 2.0
    half_spread_bps: float = 5.0
    neutralize: bool = True


class BacktestResult:
    def __init__(self):
        self.returns, self.gross_returns, self.costs = {}, {}, {}
        self.weights_history, self.alpha_history, self.neutral_alpha_history = {}, {}, {}

    def to_frame(self) -> pd.DataFrame:
        idx = sorted(self.returns)
        return pd.DataFrame({
            "net_return": pd.Series(self.returns).reindex(idx),
            "gross_return": pd.Series(self.gross_returns).reindex(idx),
            "cost": pd.Series(self.costs).reindex(idx),
        })


def _build_features(panel: pd.DataFrame) -> dict:
    close = to_wide_close(panel)
    volume = panel["volume"].unstack("ticker")
    return {
        "momentum": log_price_slope(close),
        "volatility": compute_volatility(close),
        "Low liquidity": compute_low_liquidity(close, volume),
    }


def run_backtest(panel: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    close = to_wide_close(panel)
    daily_return = close.pct_change()
    sector = panel["sector"].unstack("ticker").ffill()
    features = _build_features(panel)
    forward_return = close.shift(-config.holding_period) / close - 1.0

    all_dates = close.index
    rebalance_dates = all_dates[config.min_train_days::config.holding_period]
    result, prev_weights = BacktestResult(), pd.Series(dtype=float)

    for date in rebalance_dates:
        loc = all_dates.get_loc(date)
        if loc + config.holding_period >= len(all_dates):
            break

        if config.alpha_mode == "linear_combo":
            today = pd.DataFrame({name: df.loc[date] for name, df in features.items()})
            z = (today - today.mean()) / today.std()
            alpha_today = (z * pd.Series(config.combo_weights)).sum(axis=1, skipna=True)
        else:
            train_end_loc = loc - config.holding_period
            train_start_date = date - pd.Timedelta(days=int(config.train_years * 365))
            train_dates = all_dates[
                (all_dates >= train_start_date) &
                (all_dates <= all_dates[max(train_end_loc, 0)])
            ]
            if len(train_dates) < config.min_train_days:
                continue

            train_features = {name: df.loc[train_dates] for name, df in features.items()}
            X_train, y_train = build_dataset(train_features, forward_return.loc[train_dates])
            if X_train.empty:
                continue

            model = LinearAlphaModel() if config.alpha_mode == "ridge" else XGBoostAlphaModel()
            model.fit(X_train, y_train)

            X_today = pd.DataFrame({name: df.loc[date] for name, df in features.items()}).dropna()
            if X_today.empty:
                continue
            alpha_today = model.predict(X_today)

        result.alpha_history[date] = alpha_today
        neutral_today = neutralize_cross_section(alpha_today, sector.loc[date]) if config.neutralize else alpha_today
        result.neutral_alpha_history[date] = neutral_today

        cov_start = max(0, loc - config.cov_lookback + 1)
        covariance = daily_return.iloc[cov_start:loc + 1].cov() * 252

        weights_today = optimize_weights(
            neutral_today, sector.loc[date], covariance,
            prev_weights=prev_weights, max_position=config.max_position,
            sector_tol=config.sector_tol, gross_target=config.gross_target,
            risk_aversion=config.risk_aversion, cost_aversion=config.cost_aversion,
        )
        result.weights_history[date] = weights_today

        fwd = forward_return.loc[date].reindex(weights_today.index).fillna(0.0)
        gross_ret = float((weights_today * fwd).sum())
        cost = compute_trade_costs(
            weights_today, prev_weights,
            brokerage_bps=config.brokerage_bps, half_spread_bps=config.half_spread_bps,
        )
        result.gross_returns[date] = gross_ret
        result.costs[date] = cost
        result.returns[date] = gross_ret - cost
        prev_weights = weights_today

    return result


def sharpe_ratio(returns, periods_per_year=PERIODS_PER_YEAR, risk_free=0.0):
    excess = returns - risk_free / periods_per_year
    return 0.0 if excess.dropna().empty or excess.std() == 0 else float(
        np.sqrt(periods_per_year) * excess.mean() / excess.std()
    )


def annualized_return(returns, periods_per_year=PERIODS_PER_YEAR):
    returns = returns.dropna()
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() ** (periods_per_year / len(returns)) - 1)


def annualized_vol(returns, periods_per_year=PERIODS_PER_YEAR):
    return float(returns.dropna().std() * np.sqrt(periods_per_year))


def cumulative_curve(returns):
    return (1 + returns.fillna(0)).cumprod()


def max_drawdown(returns):
    curve = cumulative_curve(returns)
    return float((curve / curve.cummax() - 1).min())


def drawdown_series(returns):
    curve = cumulative_curve(returns)
    return curve / curve.cummax() - 1


def summary_stats(returns, periods_per_year=PERIODS_PER_YEAR):
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_vol": annualized_vol(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "n_periods": int(returns.dropna().shape[0]),
    }
