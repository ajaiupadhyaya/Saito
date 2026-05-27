"""Polygon aggregates prices provider."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import ProviderError
from src.data.providers.http_client import get_json


@dataclass
class PolygonProvider:
    name: str = "polygon"
    api_key_env: str = "POLYGON_API_KEY"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        key = getenv(self.api_key_env)
        if not key:
            raise ProviderError(f"missing {self.api_key_env}")
        if end is None:
            end = start
        frames: list[pd.Series] = []
        for ticker in tickers:
            payload = get_json(
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}",
                params={"adjusted": "true", "sort": "asc", "limit": "50000", "apiKey": key},
            )
            bars = payload.get("results", [])
            if not bars:
                raise ProviderError(f"no Polygon bars for {ticker}")
            df = pd.DataFrame(bars)
            s = pd.Series(pd.to_numeric(df["c"], errors="coerce").values, index=pd.to_datetime(df["t"], unit="ms"))
            s = s.sort_index().dropna()
            frames.append(s.rename(ticker))
        return pd.concat(frames, axis=1).dropna(how="all")

