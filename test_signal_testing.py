import numpy as np
import pandas as pd

from data import load_ohlcv_from_csv, to_wide_close
from analysis import ic_timeseries, icir, quantile_returns


panel = load_ohlcv_from_csv(r"outputs\nifty50_real.csv")
close = to_wide_close(panel)
volume = panel["volume"].unstack("ticker")

forward_return = close.shift(-21) / close - 1


def test_signal(name, signal):
    ic = ic_timeseries(signal, forward_return)
    q = quantile_returns(signal, forward_return, n_quantiles=5)

    print(f"\n{name}")
    print(f"Mean IC: {ic.mean():.4f}")
    print(f"ICIR: {icir(ic):.3f}")
    print("Quantile returns:")
    print(q)

    if not q.empty:
        print("Monotonic:", q["mean_forward_return"].is_monotonic_increasing)


# =========================================================
# NEW SIGNAL EXPERIMENT
# =========================================================

'''
def new_signal(close, volume):
    ............
    ..........
    ........
    signal = ..
    return signal
    
'''

# test_signal = new_signal(close, volume)
# test_signal("New Signal", test_signal)