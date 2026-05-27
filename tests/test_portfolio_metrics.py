"""Tests for portfolio analytics module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.synthetic import generate_preset
from src.metrics import portfolio


def test_frontier_returns_points():
    prices = generate_preset("correlated_normal", n_obs=300)
    rets = prices.pct_change().dropna()
    mu, cov = portfolio.annualized_mu_cov(rets)
    pts = portfolio.efficient_frontier_constrained(mu, cov, n_points=8)
    assert len(pts) >= 3
    assert all(p.volatility >= 0 for p in pts)


def test_risk_parity_weights_sum_to_one():
    prices = generate_preset("correlated_normal", n_obs=250)
    rets = prices.pct_change().dropna()
    _mu, cov = portfolio.annualized_mu_cov(rets)
    w = portfolio.risk_parity_weights(cov)
    assert np.isclose(np.sum(w), 1.0, atol=1e-8)
    assert np.all(w > 0)


def test_hrp_weights_sum_to_one():
    prices = generate_preset("correlated_normal", n_obs=250)
    rets = prices.pct_change().dropna()
    w = portfolio.hrp_weights(rets)
    assert np.isclose(float(w.sum()), 1.0, atol=1e-8)


def test_factor_exposure_runs():
    idx = pd.date_range("2024-01-01", periods=180, freq="B")
    f1 = pd.Series(np.random.default_rng(1).normal(0, 0.01, len(idx)), index=idx)
    f2 = pd.Series(np.random.default_rng(2).normal(0, 0.01, len(idx)), index=idx)
    target = 0.4 * f1 + 0.2 * f2 + pd.Series(np.random.default_rng(3).normal(0, 0.01, len(idx)), index=idx)
    exp, r2 = portfolio.factor_exposure(target, pd.DataFrame({"f1": f1, "f2": f2}))
    assert set(exp.index) == {"f1", "f2"}
    assert 0 <= r2 <= 1

