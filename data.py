from __future__ import annotations

import argparse
import pandas as pd
import yfinance as yf


SECTOR_MAP = {
    "ADANIENT.NS": "Diversified",
    "ADANIPORTS.NS": "Infrastructure",
    "APOLLOHOSP.NS": "Healthcare",
    "ASIANPAINT.NS": "Consumer",
    "AXISBANK.NS": "Banking",
    "BAJAJ-AUTO.NS": "Auto",
    "BAJFINANCE.NS": "Financial Services",
    "BAJAJFINSV.NS": "Financial Services",
    "BEL.NS": "Industrials",
    "BHARTIARTL.NS": "Telecom",
    "CIPLA.NS": "Pharma",
    "COALINDIA.NS": "Metals & Mining",
    "DRREDDY.NS": "Pharma",
    "EICHERMOT.NS": "Auto",
    "ETERNAL.NS": "Consumer Services",
    "GRASIM.NS": "Materials",
    "HCLTECH.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "HDFCLIFE.NS": "Financial Services",
    "HEROMOTOCO.NS": "Auto",
    "HINDALCO.NS": "Metals & Mining",
    "HINDUNILVR.NS": "FMCG",
    "ICICIBANK.NS": "Banking",
    "INDUSINDBK.NS": "Banking",
    "INFY.NS": "IT",
    "ITC.NS": "FMCG",
    "JIOFIN.NS": "Financial Services",
    "JSWSTEEL.NS": "Metals & Mining",
    "KOTAKBANK.NS": "Banking",
    "LT.NS": "Infrastructure",
    "M&M.NS": "Auto",
    "MARUTI.NS": "Auto",
    "MAXHEALTH.NS": "Healthcare",
    "NESTLEIND.NS": "FMCG",
    "NTPC.NS": "Power",
    "ONGC.NS": "Energy",
    "POWERGRID.NS": "Power",
    "RELIANCE.NS": "Energy",
    "SBILIFE.NS": "Financial Services",
    "SBIN.NS": "Banking",
    "SHRIRAMFIN.NS": "Financial Services",
    "SUNPHARMA.NS": "Pharma",
    "TATACONSUM.NS": "FMCG",
    "TATASTEEL.NS": "Metals & Mining",
    "TCS.NS": "IT",
    "TECHM.NS": "IT",
    "TITAN.NS": "Consumer",
    "TRENT.NS": "Consumer",
    "ULTRACEMCO.NS": "Materials",
    "WIPRO.NS": "IT",
}

TICKERS = list(SECTOR_MAP)


def fetch_ohlcv(start, end):
    print(f"Using {len(TICKERS)} stocks")

    end_download = (
        pd.Timestamp(end) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    raw = yf.download(
        TICKERS,
        start=start,
        end=end_download,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=True,
    )

    frames = []

    for ticker in TICKERS:
        
        df = raw[ticker].copy().dropna(how="all")
        df.columns = df.columns.str.lower()

        df = df[["open", "high", "low", "close", "volume"]]
        df["ticker"] = ticker
        df["sector"] = SECTOR_MAP[ticker]

        frames.append(
            df.reset_index().rename(columns={"Date": "date"})
        )

    return pd.concat(frames, ignore_index=True)

def load_ohlcv_from_csv(path: str) -> pd.DataFrame:
   
    df = pd.read_csv(path, parse_dates=["date"])
    
    df = (
        df.sort_values(["date", "ticker"])
        .set_index(["date", "ticker"])
    )

    return df[
        ["open", "high", "low", "close", "volume", "sector"]
    ]



def to_wide_close(panel: pd.DataFrame) -> pd.DataFrame:
    return panel["close"].unstack("ticker")



def main():

    # Terminal input to independently download data 
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--out", default="outputs\\nifty50_real.csv")
    args = parser.parse_args()

    df = fetch_ohlcv(args.start, args.end)

    df = df[
        ["date", "ticker", "open", "high", "low", "close", "volume", "sector"]
    ].sort_values(["date", "ticker"])

    df.to_csv(args.out, index=False)

    print(f"\nDownloaded: {df['ticker'].nunique()} stocks")
    print(f"Rows: {len(df):,}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()



# Output: date × ticker × OHLCV × sector (87321 x 8) "2019-01-01" to "2026-03-31"