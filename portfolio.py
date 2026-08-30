from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def optimize_weights(
    alpha: pd.Series,
    sector: pd.Series,
    covariance: pd.DataFrame,
    prev_weights: Optional[pd.Series] = None,
    max_position: float = 0.05,
    sector_tol: float = 0.10,
    gross_target: float = 2.0,
    risk_aversion: float = 5.0,
    cost_aversion: float = 2.0,
) -> pd.Series:
    tickers = alpha.dropna().index
    a = alpha.loc[tickers].values.astype(float)
    a = (a - a.mean()) / (a.std() + 1e-9)

    sec = sector.loc[tickers]
    sigma = covariance.reindex(index=tickers, columns=tickers).fillna(0.0).values
    sigma = 0.5 * (sigma + sigma.T)

    n = len(tickers)
    w_prev = prev_weights.reindex(tickers).fillna(0.0).values if prev_weights is not None else np.zeros(n)
    sector_masks = [np.array(sec.values == s, dtype=float) for s in sec.unique()]

    def objective(w):
        alpha_term = a @ w
        risk_term = risk_aversion * (w @ sigma @ w)
        cost_term = cost_aversion * np.sum((w - w_prev) ** 2)
        return -(alpha_term - risk_term - cost_term)

    def objective_grad(w):
        return -(a - 2 * risk_aversion * (sigma @ w) - 2 * cost_aversion * (w - w_prev))

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w)},
        {"type": "ineq", "fun": lambda w: gross_target - np.sum(np.abs(w))},
    ]
    for mask in sector_masks:
        constraints.append({"type": "ineq", "fun": lambda w, m=mask: sector_tol - m @ w})
        constraints.append({"type": "ineq", "fun": lambda w, m=mask: sector_tol + m @ w})

    bounds = [(-max_position, max_position)] * n
    w0 = np.clip(a / (np.abs(a).sum() + 1e-9), -max_position, max_position)

    result = minimize(
        objective, w0, jac=objective_grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-7},
    )
    if not result.success:
        print("Optimizer warning:", result.message)

    return pd.Series(result.x, index=tickers)


def neutralize_cross_section(alpha_row: pd.Series, sector_row: pd.Series) -> pd.Series:
    valid = alpha_row.dropna().index.intersection(sector_row.dropna().index)
    y = alpha_row.loc[valid].values.astype(float)
    dummies = pd.get_dummies(sector_row.loc[valid], drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(valid)), dummies.values])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    out = pd.Series(np.nan, index=alpha_row.index)
    out.loc[valid] = y - X @ coef
    return out


def neutralize_panel(alpha: pd.DataFrame, sector: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        date: neutralize_cross_section(alpha.loc[date], sector.loc[date])
        for date in alpha.index
    }).T


def compute_trade_costs(
    weights_today: pd.Series,
    weights_prev: pd.Series,
    brokerage_bps: float = 2.0,
    half_spread_bps: float = 5.0,
) -> float:
    tickers = weights_today.index.union(weights_prev.index)
    wt = weights_today.reindex(tickers).fillna(0.0)
    wp = weights_prev.reindex(tickers).fillna(0.0)
    turnover = (wt - wp).abs().sum()
    return turnover * (brokerage_bps + half_spread_bps) / 10_000.0
