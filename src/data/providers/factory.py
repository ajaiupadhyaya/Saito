"""Provider construction and routing policies."""

from __future__ import annotations

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import MarketDataHub
from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.finnhub_provider import FinnhubProvider
from src.data.providers.polygon_provider import PolygonProvider
from src.data.providers.tiingo_provider import TiingoProvider
from src.data.providers.yfinance_provider import YFinanceProvider

POLICIES = ("auto", "premium-first", "free-only")


def build_price_hub(policy: str = "auto") -> MarketDataHub:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    y = YFinanceProvider()
    premium = []
    if getenv("POLYGON_API_KEY"):
        premium.append(PolygonProvider())
    if getenv("TIINGO_API_KEY"):
        premium.append(TiingoProvider())
    if getenv("ALPHAVANTAGE_API_KEY"):
        premium.append(AlphaVantageProvider())
    if getenv("FINNHUB_API_KEY"):
        premium.append(FinnhubProvider())

    if policy == "free-only":
        return MarketDataHub(price_providers=[y], policy=policy, retries=0)
    if policy == "premium-first":
        providers = premium + [y]
    else:  # auto
        providers = premium + [y] if premium else [y]
    return MarketDataHub(price_providers=providers, policy=policy, retries=1)


def provider_capability_matrix() -> pd.DataFrame:
    """Describe configured providers and core capabilities for UI diagnostics."""
    rows = [
        {
            "provider": "polygon",
            "enabled": bool(getenv("POLYGON_API_KEY")),
            "prices": True,
            "options": True,
            "macro": False,
            "sentiment": False,
            "tier": "premium",
        },
        {
            "provider": "tiingo",
            "enabled": bool(getenv("TIINGO_API_KEY")),
            "prices": True,
            "options": False,
            "macro": False,
            "sentiment": False,
            "tier": "premium",
        },
        {
            "provider": "alphavantage",
            "enabled": bool(getenv("ALPHAVANTAGE_API_KEY")),
            "prices": True,
            "options": False,
            "macro": False,
            "sentiment": False,
            "tier": "premium",
        },
        {
            "provider": "finnhub",
            "enabled": bool(getenv("FINNHUB_API_KEY")),
            "prices": True,
            "options": False,
            "macro": False,
            "sentiment": False,
            "tier": "premium",
        },
        {
            "provider": "yfinance",
            "enabled": True,
            "prices": True,
            "options": True,
            "macro": False,
            "sentiment": False,
            "tier": "free",
        },
    ]
    return pd.DataFrame(rows)

