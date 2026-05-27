"""Tests for API-backed research hub helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.hub_service import MarketDataHub, ProviderError
from src.metrics import research


class _FailingProvider:
    name = "failing"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        raise ProviderError("boom")


class _WorkingProvider:
    name = "working"

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        data = {t: np.linspace(100.0, 104.0, len(idx)) for t in tickers}
        return pd.DataFrame(data, index=idx)


def test_market_data_hub_falls_back_to_next_provider():
    hub = MarketDataHub(price_providers=[_FailingProvider(), _WorkingProvider()])
    prices, meta = hub.get_prices(["AAPL", "MSFT"], start="2024-01-01", end="2024-01-31")
    assert list(prices.columns) == ["AAPL", "MSFT"]
    assert meta["provider"] == "working"


def test_var_cvar_outputs_expected_ordering():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.02, 1500))
    out = research.var_cvar(returns, alpha=0.95)
    assert out["cvar"] >= out["var"]
    assert out["tail_count"] > 0


def test_signal_backtest_runs_and_returns_stats():
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    prices = pd.Series(np.linspace(100, 130, len(idx)), index=idx)
    signal = pd.Series(1.0, index=idx)
    result = research.run_signal_backtest(prices, signal, fee_bps=2.0)
    assert "sharpe" in result.metrics
    assert result.equity_curve.iloc[-1] > 0

