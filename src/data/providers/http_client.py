"""Small HTTP JSON client with standard library only."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.data.hub_service import ProviderError


def get_json(url: str, params: dict[str, str] | None = None, headers: dict[str, str] | None = None) -> dict:
    if params:
        qs = urlencode(params)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{qs}"
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=20) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ProviderError(f"http {exc.code}: {body[:200]}") from exc
    except URLError as exc:
        raise ProviderError(f"network error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ProviderError(f"invalid json payload from {url}") from exc

