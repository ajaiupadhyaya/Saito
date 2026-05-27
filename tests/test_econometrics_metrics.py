"""Tests for econometrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import econometrics


def test_ar1_fit_outputs_parameters():
    rng = np.random.default_rng(42)
    n = 300
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.6 * x[i - 1] + rng.normal(0, 0.1)
    s = pd.Series(x)
    out = econometrics.ar1_fit(s)
    assert "phi" in out and "sigma_resid" in out
    assert abs(out["phi"]) <= 1.2


def test_ljung_box_returns_valid_probability():
    s = pd.Series(np.random.default_rng(1).normal(0, 1, 350))
    p = econometrics.ljung_box_pvalue(s, lags=10)
    assert 0 <= p <= 1


def test_volatility_regime_flags_are_binary():
    s = pd.Series(np.random.default_rng(3).normal(0, 0.02, 260))
    vol = econometrics.rolling_realized_vol(s, window=21)
    flags = econometrics.volatility_regime_flags(vol, high_quantile=0.8)
    assert set(flags.unique()).issubset({0, 1})

