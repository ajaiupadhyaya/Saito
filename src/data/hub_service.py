"""Unified API-first data service for research pages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter, sleep
from typing import Any, Protocol

import pandas as pd


class ProviderError(RuntimeError):
    """Raised when an upstream provider fails."""


class PriceProvider(Protocol):
    name: str

    def get_prices(self, tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame: ...


@dataclass
class ProviderAttempt:
    provider: str
    ok: bool
    latency_ms: int
    error: str = ""


@dataclass
class MarketDataHub:
    """Provider orchestrator with graceful fallback and provenance."""

    price_providers: list[PriceProvider]
    policy: str = "auto"
    retries: int = 1
    retry_backoff_seconds: float = 0.2
    cooldown_seconds: float = 30.0
    _last_errors: dict[str, str] = field(default_factory=dict)
    _cooldown_until: dict[str, float] = field(default_factory=dict)
    _last_attempts: list[ProviderAttempt] = field(default_factory=list)

    def get_prices(
        self,
        tickers: list[str],
        *,
        start: str,
        end: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if not tickers:
            raise ValueError("tickers must be non-empty")
        self._last_errors.clear()
        self._last_attempts.clear()
        now = perf_counter()
        for provider in self.price_providers:
            cooldown_deadline = self._cooldown_until.get(provider.name, 0.0)
            if cooldown_deadline > now:
                wait_s = max(0.0, cooldown_deadline - now)
                self._last_errors[provider.name] = f"cooldown active ({wait_s:.1f}s left)"
                self._last_attempts.append(
                    ProviderAttempt(
                        provider=provider.name,
                        ok=False,
                        latency_ms=0,
                        error=self._last_errors[provider.name],
                    )
                )
                continue
            for i in range(max(1, self.retries + 1)):
                t0 = perf_counter()
                try:
                    df = provider.get_prices(tickers, start=start, end=end)
                    if df is None or df.empty:
                        raise ProviderError("empty dataset")
                    out = df.sort_index().dropna(how="all")
                    if out.empty:
                        raise ProviderError("all rows were NaN")
                    latency_ms = int((perf_counter() - t0) * 1000)
                    self._last_attempts.append(ProviderAttempt(provider=provider.name, ok=True, latency_ms=latency_ms))
                    meta = {
                        "provider": provider.name,
                        "policy": self.policy,
                        "n_assets": int(out.shape[1]),
                        "n_obs": int(out.shape[0]),
                        "start": str(out.index.min().date()) if len(out.index) else None,
                        "end": str(out.index.max().date()) if len(out.index) else None,
                        "latency_ms": latency_ms,
                        "provider_count": len(self.price_providers),
                        "attempts": [asdict(x) for x in self._last_attempts],
                    }
                    return out, meta
                except Exception as exc:  # pragma: no cover - aggregated error path
                    latency_ms = int((perf_counter() - t0) * 1000)
                    msg = str(exc)
                    self._last_attempts.append(
                        ProviderAttempt(provider=provider.name, ok=False, latency_ms=latency_ms, error=msg)
                    )
                    self._last_errors[provider.name] = msg
                    if i < self.retries:
                        sleep(self.retry_backoff_seconds * (i + 1))
                    else:
                        self._cooldown_until[provider.name] = perf_counter() + self.cooldown_seconds
        raise ProviderError(f"all providers failed: {self._last_errors}")

    @property
    def last_errors(self) -> dict[str, str]:
        return dict(self._last_errors)

    @property
    def last_attempts(self) -> list[dict[str, Any]]:
        return [asdict(x) for x in self._last_attempts]

