"""NewsAPI sentiment proxy provider."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import ProviderError
from src.data.providers.http_client import get_json

POS = {"beat", "growth", "upgrade", "surge", "record", "bullish", "optimistic", "strong"}
NEG = {"miss", "downgrade", "cut", "fall", "lawsuit", "bearish", "risk", "weak"}


def _score(text: str) -> int:
    lower = text.lower()
    pos = sum(1 for w in POS if w in lower)
    neg = sum(1 for w in NEG if w in lower)
    return pos - neg


@dataclass
class NewsApiProvider:
    name: str = "newsapi"
    api_key_env: str = "NEWSAPI_KEY"

    def get_headlines(self, query: str, page_size: int = 25) -> pd.DataFrame:
        key = getenv(self.api_key_env)
        if not key:
            raise ProviderError(f"missing {self.api_key_env}")
        payload = get_json(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": str(max(5, min(page_size, 100))),
                "apiKey": key,
            },
        )
        arts = payload.get("articles", [])
        if not arts:
            raise ProviderError(f"no news articles for query={query}")
        df = pd.DataFrame(
            {
                "publishedAt": [a.get("publishedAt") for a in arts],
                "source": [a.get("source", {}).get("name", "") for a in arts],
                "title": [a.get("title", "") for a in arts],
                "description": [a.get("description", "") for a in arts],
                "url": [a.get("url", "") for a in arts],
            }
        )
        df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
        df["sentiment_score"] = (df["title"].fillna("") + " " + df["description"].fillna("")).map(_score)
        return df.sort_values("publishedAt", ascending=False).reset_index(drop=True)

