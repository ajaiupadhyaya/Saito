"""Tiny in-process TTL cache for provider responses."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(Generic[T]):
    def __init__(self, ttl_seconds: int = 600):
        self.ttl_seconds = int(ttl_seconds)
        self._items: dict[str, _Entry[T]] = {}

    def get(self, key: str) -> T | None:
        now = time.time()
        item = self._items.get(key)
        if item is None:
            return None
        if item.expires_at < now:
            self._items.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: T) -> None:
        self._items[key] = _Entry(value=value, expires_at=time.time() + self.ttl_seconds)

