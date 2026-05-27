"""Environment variable helpers for API-backed providers."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_dotenv_if_present() -> None:
    """Best-effort .env loader without external dependencies."""
    global _LOADED
    if _LOADED:
        return
    path = Path(__file__).resolve().parents[2] / ".env"
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    _LOADED = True


def getenv(name: str) -> str | None:
    load_dotenv_if_present()
    val = os.getenv(name)
    return val.strip() if val else None

