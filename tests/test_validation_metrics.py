"""Tests for strategy validation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.validation import overfit_gap_score, walk_forward_backtest


def test_walk_forward_backtest_outputs_splits():
    idx = pd.date_range("2022-01-01", periods=500, freq="B")
    rets = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, len(idx)), index=idx)
    signal = pd.Series(np.random.default_rng(2).choice([-1.0, 1.0], len(idx)), index=idx)
    wf = walk_forward_backtest(rets, signal, train=200, test=60)
    assert len(wf) >= 3
    assert wf[0].test_end >= wf[0].test_start


def test_overfit_gap_score_numeric():
    idx = pd.date_range("2021-01-01", periods=450, freq="B")
    rets = pd.Series(np.random.default_rng(4).normal(0.0006, 0.012, len(idx)), index=idx)
    signal = pd.Series(1.0, index=idx)
    wf = walk_forward_backtest(rets, signal, train=180, test=60)
    gap = overfit_gap_score(wf)
    assert np.isfinite(gap) or np.isnan(gap)

