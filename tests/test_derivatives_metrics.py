"""Tests for derivatives and volatility analytics."""

from __future__ import annotations

import pandas as pd

from src.metrics import derivatives


def _sample_chain(expiry: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "expiry": [expiry] * 6,
            "strike": [90, 95, 100, 105, 110, 115],
            "impliedVolatility": [0.35, 0.31, 0.28, 0.27, 0.29, 0.33],
            "openInterest": [120, 140, 200, 180, 150, 100],
            "bid": [12, 9, 7, 5, 3, 2],
            "ask": [12.5, 9.5, 7.5, 5.5, 3.5, 2.5],
            "type": ["call"] * 6,
        }
    )


def test_option_chain_quality_in_bounds():
    q = derivatives.option_chain_quality(_sample_chain("2026-12-19"))
    assert 0 <= q["quality_score"] <= 1


def test_iv_term_structure_outputs_sorted_rows():
    chains = {
        "2026-12-19": _sample_chain("2026-12-19"),
        "2026-09-19": _sample_chain("2026-09-19"),
    }
    ts = derivatives.iv_term_structure(chains)
    assert len(ts) == 2
    assert ts["expiry"].iloc[0] <= ts["expiry"].iloc[1]


def test_black_scholes_call_greeks_positive_gamma():
    g = derivatives.black_scholes_call_greeks(spot=100, strike=100, ttm_years=0.5, rate=0.03, vol=0.25)
    assert g.gamma > 0
    assert g.price > 0

