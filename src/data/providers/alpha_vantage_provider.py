"""Alpha Vantage daily prices provider."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import ProviderError
from src.data.providers.http_client import get_json


@dataclass
class AlphaVantageProvider:
    name: str = "alphavantage"
    api_key_env: str = "ALPHAVANTAGE_API_KEY"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        key = getenv(self.api_key_env)
        if not key:
            raise ProviderError(f"missing {self.api_key_env}")
        frames: list[pd.Series] = []
        for ticker in tickers:
            payload = get_json(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": "full",
                    "apikey": key,
                },
            )
            ts = payload.get("Time Series (Daily)", {})
            if not ts:
                raise ProviderError(f"no Alpha Vantage series for {ticker}")
            df = pd.DataFrame.from_dict(ts, orient="index")
            col = "5. adjusted close" if "5. adjusted close" in df.columns else "4. close"
            s = pd.to_numeric(df[col], errors="coerce")
            s.index = pd.to_datetime(s.index, errors="coerce")
            s = s.sort_index().dropna()
            if s.empty:
                raise ProviderError(f"empty Alpha Vantage prices for {ticker}")
            frames.append(s.rename(ticker))
        out = pd.concat(frames, axis=1).dropna(how="all")
        if start:
            out = out[out.index >= pd.Timestamp(start)]
        if end:
            out = out[out.index <= pd.Timestamp(end)]
        return out

