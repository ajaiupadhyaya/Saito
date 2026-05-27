"""Tests for macro event-study utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import event_study


def test_macro_shock_dates_detects_outlier():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    s = pd.Series(np.random.default_rng(1).normal(0, 1, len(idx)).cumsum(), index=idx)
    s.iloc[120] += 8.0
    shocks = event_study.macro_shock_dates(s, z_threshold=2.0)
    assert len(shocks) > 0


def test_event_window_car_returns_panel():
    idx = pd.date_range("2021-01-01", periods=120, freq="B")
    asset = pd.Series(np.random.default_rng(2).normal(0, 0.01, len(idx)), index=idx)
    bench = pd.Series(np.random.default_rng(3).normal(0, 0.01, len(idx)), index=idx)
    ar = event_study.abnormal_returns(asset, bench)
    panel = event_study.event_window_car(ar, pd.DatetimeIndex([idx[50], idx[80]]), pre=2, post=3)
    assert not panel.empty
    assert {"event_date", "t", "ar", "car"}.issubset(panel.columns)

