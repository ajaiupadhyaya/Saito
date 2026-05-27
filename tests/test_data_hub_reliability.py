"""Reliability tests for market data hub routing behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.hub_service import MarketDataHub, ProviderError


class _FlakyThenWorks:
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            raise ProviderError("transient timeout")
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        return pd.DataFrame({t: np.linspace(100, 103, len(idx)) for t in tickers}, index=idx)


class _AlwaysFail:
    name = "always_fail"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        raise ProviderError("hard fail")


def test_hub_retries_transient_failure_and_succeeds():
    flaky = _FlakyThenWorks()
    hub = MarketDataHub(price_providers=[flaky], retries=1, retry_backoff_seconds=0.0, cooldown_seconds=0.0)
    prices, meta = hub.get_prices(["AAPL"], start="2024-01-01", end="2024-01-31")
    assert not prices.empty
    assert flaky.calls == 2
    assert len(meta["attempts"]) == 2
    assert meta["attempts"][0]["ok"] is False
    assert meta["attempts"][1]["ok"] is True


def test_hub_cooldown_records_and_blocks_immediate_retry():
    hub = MarketDataHub(price_providers=[_AlwaysFail()], retries=0, cooldown_seconds=10.0)
    try:
        hub.get_prices(["AAPL"], start="2024-01-01", end="2024-01-31")
    except ProviderError:
        pass
    try:
        hub.get_prices(["AAPL"], start="2024-01-01", end="2024-01-31")
    except ProviderError:
        pass
    assert "always_fail" in hub.last_errors
    assert "cooldown active" in hub.last_errors["always_fail"]

