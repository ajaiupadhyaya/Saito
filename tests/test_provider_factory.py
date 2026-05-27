"""Tests for provider routing policies."""

from __future__ import annotations

from src.data.providers.factory import build_price_hub


def test_free_only_policy_uses_yfinance(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    hub = build_price_hub(policy="free-only")
    assert [p.name for p in hub.price_providers] == ["yfinance"]


def test_premium_first_includes_premium_before_yfinance(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "x")
    monkeypatch.setenv("FINNHUB_API_KEY", "x")
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    hub = build_price_hub(policy="premium-first")
    names = [p.name for p in hub.price_providers]
    assert names[-1] == "yfinance"
    assert "polygon" in names[:-1]
    assert "finnhub" in names[:-1]

