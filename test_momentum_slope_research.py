import pandas as pd

from analysis import ic_timeseries, icir, quantile_returns
from data import load_ohlcv_from_csv, to_wide_close
from features import log_price_slope


panel = load_ohlcv_from_csv(r"outputs\nifty50_real.csv")
close = to_wide_close(panel)
forward_return = close.shift(-21) / close - 1

signals = {
    "Log-Price Slope 12M": log_price_slope(close, 252),
    "Log-Price Slope 24M": log_price_slope(close, 504),
}

results = []
for name, signal in signals.items():
    ic = ic_timeseries(signal, forward_return)
    q = quantile_returns(signal, forward_return, 5)
    results.append({
        "Signal": name, "Mean IC": ic.mean(), "ICIR": icir(ic),
        "Bottom Q": q.iloc[0, 0], "Top Q": q.iloc[-1, 0],
        "Monotonic": q["mean_forward_return"].is_monotonic_increasing,
    })

print(pd.DataFrame(results).to_string(index=False))
