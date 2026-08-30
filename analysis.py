import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PERIODS_PER_YEAR = 252 / 21


def ic_timeseries(alpha: pd.DataFrame, forward_return: pd.DataFrame) -> pd.Series:
    out = {}
    for date in alpha.index.intersection(forward_return.index):
        joined = pd.concat([alpha.loc[date], forward_return.loc[date]], axis=1).dropna()
        out[date] = spearmanr(joined.iloc[:, 0], joined.iloc[:, 1])[0]
    return pd.Series(out).sort_index()


def icir(ic_series: pd.Series) -> float:
    clean = ic_series.dropna()
    return 0.0 if clean.empty or clean.std() == 0 else float(clean.mean() / clean.std())


def quantile_returns(alpha: pd.DataFrame, forward_return: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    rows = []
    for date in alpha.index.intersection(forward_return.index):
        joined = pd.concat([alpha.loc[date], forward_return.loc[date]], axis=1).dropna()
        joined.columns = ["alpha", "ret"]
        joined["bucket"] = pd.qcut(joined["alpha"], n_quantiles, labels=False, duplicates="drop") + 1
        rows.append(joined.groupby("bucket")["ret"].mean())
    return pd.DataFrame(rows).mean(axis=0).to_frame("mean_forward_return")


def compute_market_factor(wide_close: pd.DataFrame) -> pd.Series:
    return (wide_close.shift(-21) / wide_close - 1).mean(axis=1)


def compute_style_factor(signal_scores: pd.DataFrame, forward_return: pd.DataFrame) -> pd.Series:
    out = {}
    for date in signal_scores.index.intersection(forward_return.index):
        joined = pd.concat([signal_scores.loc[date], forward_return.loc[date]], axis=1).dropna()
        joined.columns = ["score", "ret"]
        if len(joined) < 10:
            continue
        joined["bucket"] = pd.qcut(joined["score"], 5, labels=False, duplicates="drop")
        top = joined.loc[joined["bucket"] == joined["bucket"].max(), "ret"].mean()
        bottom = joined.loc[joined["bucket"] == joined["bucket"].min(), "ret"].mean()
        out[date] = top - bottom
    return pd.Series(out).sort_index()


def factor_regression(strategy_returns: pd.Series, factors: dict) -> dict:
    joined = pd.concat([strategy_returns.rename("strategy"), pd.DataFrame(factors)], axis=1).dropna()
    y = joined["strategy"].values
    X = joined.drop(columns="strategy")
    X_design = np.column_stack([np.ones(len(y)), X.values])

    coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    pred = X_design @ coef
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    return {
        "alpha": coef[0] * PERIODS_PER_YEAR,
        "betas": dict(zip(X.columns, coef[1:])),
        "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
    }


def turnover_timeseries(weights_history: dict) -> pd.Series:
    out, prev = {}, pd.Series(dtype=float)
    for date in sorted(weights_history):
        w = weights_history[date]
        tickers = w.index.union(prev.index)
        out[date] = (w.reindex(tickers).fillna(0) - prev.reindex(tickers).fillna(0)).abs().sum()
        prev = w
    return pd.Series(out).sort_index()
