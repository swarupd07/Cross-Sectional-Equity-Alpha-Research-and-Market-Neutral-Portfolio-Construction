import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import (
    compute_market_factor, compute_style_factor, factor_regression,
    ic_timeseries, icir, quantile_returns, turnover_timeseries,
)
from backtest import BacktestConfig, cumulative_curve, drawdown_series, run_backtest, summary_stats
from data import load_ohlcv_from_csv, to_wide_close
from features import log_price_slope, compute_volatility, compute_low_liquidity


OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
DATA_PATH = os.path.join(OUT_DIR, "nifty50_real.csv")


def section_signal_research(panel: pd.DataFrame) -> str:
    close = to_wide_close(panel)
    volume = panel["volume"].unstack("ticker")
    forward_return = close.shift(-21) / close - 1.0
    signals = {
        "log_price_slope": log_price_slope(close),
        "Volatility": compute_volatility(close),
        "Low Liquidity": compute_low_liquidity(close, volume),
    }

    lines = [
        "## 1. Signal-by-Signal Research\n",
        "| Signal | Mean IC | ICIR | Bottom Q return | Top Q return | Monotonic? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    ic_by_signal = {}

    for name, raw in signals.items():
        ic_s = ic_timeseries(raw, forward_return)
        q = quantile_returns(raw, forward_return, 5)
        ic_by_signal[name] = ic_s

        if q.empty:
            bottom, top, monotonic = float("nan"), float("nan"), "n/a"
        else:
            bottom, top = q.iloc[0, 0], q.iloc[-1, 0]
            monotonic = "yes" if q["mean_forward_return"].is_monotonic_increasing else "no"

        lines.append(
            f"| {name} | {ic_s.mean():.4f} | {icir(ic_s):.3f} | "
            f"{bottom:.4%} | {top:.4%} | {monotonic} |"
        )

    best_signal = max(ic_by_signal, key=lambda name: abs(icir(ic_by_signal[name])))
    plt.figure(figsize=(9, 4))
    ic_by_signal[best_signal].rolling(20).mean().plot()
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title(f"Rolling IC: {best_signal}")
    plt.ylabel("Spearman IC")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ic_timeseries.png"), dpi=120)
    plt.close()

    lines.append(f"\n_Best standalone signal by |ICIR|: **{best_signal}**. See `ic_timeseries.png`._\n")
    return "\n".join(lines)


def section_backtest_comparison(panel: pd.DataFrame):
    lines = [
        "## 2. Alpha Combination: Linear vs ML, Net of Costs\n",
        "| Method | Ann. Return | Ann. Vol | Sharpe | Max Drawdown | Avg Turnover | Avg Gross |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    configs = {
        "Linear combo (fixed weights)": BacktestConfig(alpha_mode="linear_combo"),
        "Ridge regression": BacktestConfig(alpha_mode="ridge"),
        "XGBoost": BacktestConfig(alpha_mode="xgboost"),
    }

    best_name, best_result, best_sharpe = None, None, -float("inf")
    all_stats = {}

    for name, cfg in configs.items():
        result = run_backtest(panel, cfg)
        stats = summary_stats(result.to_frame()["net_return"])
        turnover = turnover_timeseries(result.weights_history)
        gross = pd.Series({d: w.abs().sum() for d, w in result.weights_history.items()})
        avg_turnover = turnover.mean() if not turnover.empty else float("nan")
        avg_gross = gross.mean() if not gross.empty else float("nan")

        all_stats[name] = {"result": result, "stats": stats, "turnover": avg_turnover, "gross": avg_gross}
        lines.append(
            f"| {name} | {stats['annualized_return']:.2%} | {stats['annualized_vol']:.2%} | "
            f"{stats['sharpe']:.2f} | {stats['max_drawdown']:.2%} | {avg_turnover:.2f} | {avg_gross:.2f} |"
        )

        if stats["sharpe"] > best_sharpe:
            best_name, best_result, best_sharpe = name, result, stats["sharpe"]

    lines.append(f"\n_Best method by net Sharpe: **{best_name}**._\n")
    best_df = best_result.to_frame()

    plt.figure(figsize=(9, 4))
    cumulative_curve(best_df["net_return"]).plot(label="Net of costs")
    cumulative_curve(best_df["gross_return"]).plot(label="Gross of costs", linestyle="--")
    plt.legend()
    plt.title(f"Equity Curve: {best_name}")
    plt.ylabel("Growth of 1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_curve.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(9, 3.5))
    drawdown_series(best_df["net_return"]).plot(color="firebrick")
    plt.title(f"Drawdown: {best_name}")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "drawdown.png"), dpi=120)
    plt.close()

    return "\n".join(lines), best_df, {"name": best_name, "result": best_result, "all_stats": all_stats}


def section_factor_exposure(panel: pd.DataFrame, best_df: pd.DataFrame, best_name: str) -> str:
    close = to_wide_close(panel)
    volume = panel["volume"].unstack("ticker")
    forward_return = close.shift(-21) / close - 1.0

    factors = {
        "market": compute_market_factor(close),
        "momentum": compute_style_factor(log_price_slope(close), forward_return),
        "volatility": compute_style_factor(compute_volatility(close), forward_return),
        "Low liquidity": compute_style_factor(compute_low_liquidity(close, volume), forward_return),
    }
    reg = factor_regression(best_df["net_return"], factors)

    lines = [
        "## 3. Factor Exposure\n",
        f"Regressing **{best_name}**'s net 21-day returns on market, momentum, volatility, and Low liquidity factors:\n",
        f"- Annualized regression intercept (return unexplained by included factors): **{reg['alpha']:.2%}**",
    ]
    lines.extend(f"- Beta to {factor}: **{beta:.3f}**" for factor, beta in reg["betas"].items())
    lines += [
        f"- R²: **{reg['r_squared']:.3f}**\n",
        "The intercept represents return not explained by the included market and style factors; it should be "
        "interpreted as factor-adjusted alpha rather than proof of unique alpha.\n",
    ]
    return "\n".join(lines)


def section_neutralization_effect(panel: pd.DataFrame) -> str:
    res_raw = run_backtest(panel, BacktestConfig(alpha_mode="linear_combo", neutralize=False))
    res_neutral = run_backtest(panel, BacktestConfig(alpha_mode="linear_combo", neutralize=True))
    s_raw = summary_stats(res_raw.to_frame()["net_return"])
    s_neutral = summary_stats(res_neutral.to_frame()["net_return"])

    return "\n".join([
        "## 4. Effect of Sector Neutralization\n",
        "| Variant | Ann. Return | Sharpe | Max Drawdown |",
        "|---|---:|---:|---:|",
        f"| Without neutralization | {s_raw['annualized_return']:.2%} | {s_raw['sharpe']:.2f} | {s_raw['max_drawdown']:.2%} |",
        f"| With sector neutralization | {s_neutral['annualized_return']:.2%} | {s_neutral['sharpe']:.2f} | {s_neutral['max_drawdown']:.2%} |",
        "",
    ])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading real data...")
    panel = load_ohlcv_from_csv(DATA_PATH)

    print("Running signal research...")
    signal_section = section_signal_research(panel)

    print("Running walk-forward backtests...")
    backtest_section, best_df, best_info = section_backtest_comparison(panel)

    print("Running factor exposure analysis...")
    factor_section = section_factor_exposure(panel, best_df, best_info["name"])

    print("Testing neutralization effect...")
    neutral_section = section_neutralization_effect(panel)

    report = "\n".join([
        "# Cross-Sectional Equity Alpha — Research Report", "",
        "Universe: real NSE (India) daily OHLCV data. 21-day holding period / forward return horizon throughout.", "",
        signal_section, backtest_section, factor_section, neutral_section,
        "## 5. Notes on Methodology", "",
        "- All features are point-in-time correct (computed only from data known as of date t).",
        "- ML models are fit on a rolling historical window using no information after the rebalance date, "
        "with a purge gap equal to the holding period.",
        "- Alpha is neutralized against sector exposure via cross-sectional regression before portfolio construction.",
        "- Portfolio weights are solved via constrained optimization (dollar-neutral, per-name position cap, "
        "per-sector exposure cap, covariance-based portfolio risk and quadratic turnover penalties).",
        "- Reported returns are net of a linear transaction-cost model "
        "(brokerage + half bid-ask spread, in bps per unit turnover).",
        "- Gross exposure is capped at 2.0 by the optimizer; realized gross exposure may be lower and is reported "
        "as the average \\(\\sum_i|w_i|\\) across rebalance dates.", "",
    ])

    report_path = os.path.join(OUT_DIR, "research_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nDone. Report written to {report_path}")
    print(f"Charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
