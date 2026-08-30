import numpy as np
import pandas as pd


def log_price_slope(close: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    log_price = np.log(close)
    x = np.arange(lookback)
    return log_price.rolling(lookback).apply(
        lambda y: np.polyfit(x, y, 1)[0] if not np.isnan(y).any() else np.nan, raw=True
    )


def compute_reversal(close: pd.DataFrame, lookback: int = 42) -> pd.DataFrame:
    log_price = np.log(close)
    x = np.arange(lookback)
    return -log_price.rolling(lookback).apply(
        lambda y: np.polyfit(x, y, 1)[0] if not np.isnan(y).any() else np.nan, raw=True
    )


def compute_volatility(close: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    return close.pct_change().rolling(lookback).std() * np.sqrt(252)


def compute_low_liquidity(close: pd.DataFrame, volume: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    return -(close * volume).rolling(lookback).mean()
