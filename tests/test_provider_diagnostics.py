"""Tests for provider diagnostics matrix."""

from __future__ import annotations

from src.data.providers.factory import provider_capability_matrix


def test_provider_capability_matrix_has_expected_columns():
    df = provider_capability_matrix()
    needed = {"provider", "enabled", "prices", "options", "macro", "sentiment", "tier"}
    assert needed.issubset(set(df.columns))
    assert "yfinance" in set(df["provider"])

