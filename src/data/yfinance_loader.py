"""Download OHLCV via yfinance."""

from __future__ import annotations

import pandas as pd


def load_tickers(
    tickers: list[str],
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    import yfinance as yf

    if not tickers:
        raise ValueError("At least one ticker required")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        raise ValueError(f"No data returned for {tickers}")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"]
        else:
            prices = raw.xs("Close", axis=1, level=0)
    else:
        prices = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all").sort_index()
    prices.columns = [str(c).upper() for c in prices.columns]
    return prices
