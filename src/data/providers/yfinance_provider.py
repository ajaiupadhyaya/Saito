"""yfinance-backed market provider."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.hub_service import ProviderError
from src.data.yfinance_loader import load_tickers


@dataclass
class YFinanceProvider:
    name: str = "yfinance"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        try:
            return load_tickers(tickers, start=start, end=end)
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def get_option_expiries(self, ticker: str) -> list[str]:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            expiries = list(t.options)
            if not expiries:
                raise ProviderError(f"no listed option expiries for {ticker}")
            return expiries
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

    def get_option_chain(self, ticker: str, expiry: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf

        try:
            expiries = self.get_option_expiries(ticker)
            t = yf.Ticker(ticker)
            target = expiry or expiries[0]
            chain = t.option_chain(target)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["type"] = "call"
            puts["type"] = "put"
            calls["expiry"] = target
            puts["expiry"] = target
            return calls, puts
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

