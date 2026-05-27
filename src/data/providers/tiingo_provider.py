"""Tiingo daily prices provider."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import ProviderError
from src.data.providers.http_client import get_json


@dataclass
class TiingoProvider:
    name: str = "tiingo"
    api_key_env: str = "TIINGO_API_KEY"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        key = getenv(self.api_key_env)
        if not key:
            raise ProviderError(f"missing {self.api_key_env}")
        rows: list[pd.Series] = []
        for ticker in tickers:
            payload = get_json(
                f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                params={"startDate": start, "endDate": end or "", "token": key},
            )
            if not isinstance(payload, list) or not payload:
                raise ProviderError(f"no Tiingo prices for {ticker}")
            df = pd.DataFrame(payload)
            if "adjClose" not in df.columns:
                raise ProviderError(f"Tiingo payload missing adjClose for {ticker}")
            s = pd.Series(pd.to_numeric(df["adjClose"], errors="coerce").values, index=pd.to_datetime(df["date"]))
            s = s.sort_index().dropna()
            rows.append(s.rename(ticker))
        return pd.concat(rows, axis=1).dropna(how="all")

