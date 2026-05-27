"""Market and macro data provider adapters."""

from src.data.providers.alpha_vantage_provider import AlphaVantageProvider
from src.data.providers.factory import POLICIES, build_price_hub, provider_capability_matrix
from src.data.providers.finnhub_provider import FinnhubProvider
from src.data.providers.fred_provider import FredProvider
from src.data.providers.newsapi_provider import NewsApiProvider
from src.data.providers.polygon_provider import PolygonProvider
from src.data.providers.tiingo_provider import TiingoProvider
from src.data.providers.yfinance_provider import YFinanceProvider

__all__ = [
    "AlphaVantageProvider",
    "POLICIES",
    "FinnhubProvider",
    "FredProvider",
    "NewsApiProvider",
    "PolygonProvider",
    "TiingoProvider",
    "YFinanceProvider",
    "build_price_hub",
    "provider_capability_matrix",
]

