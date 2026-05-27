"""FRED macro series provider via public CSV endpoint."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.env import getenv
from src.data.hub_service import ProviderError
from src.data.providers.http_client import get_json


@dataclass
class FredProvider:
    name: str = "fred"
    base_url: str = "https://api.stlouisfed.org/fred/series/observations"
    fallback_csv_url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
    api_key_env: str = "FRED_API_KEY"

    def get_series(self, series_id: str, start: str | None = None, end: str | None = None) -> pd.Series:
        key = getenv(self.api_key_env)
        try:
            if key:
                payload = get_json(
                    self.base_url,
                    params={
                        "series_id": series_id,
                        "api_key": key,
                        "file_type": "json",
                        "observation_start": start or "",
                        "observation_end": end or "",
                    },
                )
                obs = payload.get("observations", [])
                if not obs:
                    raise ProviderError(f"no FRED values for {series_id}")
                out = pd.DataFrame(obs)
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out["value"] = pd.to_numeric(out["value"], errors="coerce")
                out = out.dropna().set_index("date").sort_index()["value"]
            else:
                url = f"{self.fallback_csv_url}{series_id}"
                df = pd.read_csv(url)
                if "DATE" not in df.columns or series_id not in df.columns:
                    raise ProviderError(f"unexpected FRED response for {series_id}")
                out = df.rename(columns={"DATE": "date", series_id: "value"})
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out["value"] = pd.to_numeric(out["value"], errors="coerce")
                out = out.dropna().set_index("date").sort_index()["value"]
            if start:
                out = out[out.index >= pd.Timestamp(start)]
            if end:
                out = out[out.index <= pd.Timestamp(end)]
            if out.empty:
                raise ProviderError(f"no FRED values for {series_id}")
            return out
        except Exception as exc:
            raise ProviderError(str(exc)) from exc

