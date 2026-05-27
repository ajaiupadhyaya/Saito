"""Finnhub candle prices provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import ProviderError
from src.data.providers.http_client import get_json


def _to_unix(s: str) -> int:
    return int(pd.Timestamp(s).timestamp())


@dataclass
class FinnhubProvider:
    name: str = "finnhub"
    api_key_env: str = "FINNHUB_API_KEY"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        key = getenv(self.api_key_env)
        if not key:
            raise ProviderError(f"missing {self.api_key_env}")
        to_ts = _to_unix(end or datetime.utcnow().strftime("%Y-%m-%d"))
        from_ts = _to_unix(start)
        frames: list[pd.Series] = []
        for ticker in tickers:
            payload = get_json(
                "https://finnhub.io/api/v1/stock/candle",
                params={
                    "symbol": ticker,
                    "resolution": "D",
                    "from": str(from_ts),
                    "to": str(to_ts),
                    "token": key,
                },
            )
            if payload.get("s") != "ok" or not payload.get("c"):
                raise ProviderError(f"no Finnhub candles for {ticker}")
            idx = pd.to_datetime(payload["t"], unit="s")
            s = pd.Series(pd.to_numeric(payload["c"], errors="coerce"), index=idx).dropna()
            frames.append(s.rename(ticker))
        return pd.concat(frames, axis=1).dropna(how="all")

