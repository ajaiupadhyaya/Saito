"""Macro and event-study utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def macro_shock_dates(series: pd.Series, z_threshold: float = 2.0) -> pd.DatetimeIndex:
    s = series.dropna().astype(float)
    if s.shape[0] < 30:
        raise ValueError("need at least 30 observations to detect shocks")
    diff = s.diff().dropna()
    z = (diff - diff.mean()) / (diff.std(ddof=1) + 1e-12)
    return z.index[np.abs(z) >= z_threshold]


def abnormal_returns(asset_rets: pd.Series, benchmark_rets: pd.Series) -> pd.Series:
    panel = pd.concat([asset_rets.rename("asset"), benchmark_rets.rename("bench")], axis=1).dropna()
    if panel.shape[0] < 30:
        raise ValueError("need at least 30 overlapping observations")
    x = panel["bench"].values
    y = panel["asset"].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ar = y - pred
    return pd.Series(ar, index=panel.index, name="abnormal_return")


def event_window_car(ar: pd.Series, event_dates: pd.DatetimeIndex, pre: int = 3, post: int = 5) -> pd.DataFrame:
    idx = ar.index
    rows = []
    for d in event_dates:
        if d not in idx:
            pos = idx.searchsorted(d)
            if pos <= 0 or pos >= len(idx):
                continue
            d = idx[pos]
        i = idx.get_loc(d)
        lo = max(0, i - pre)
        hi = min(len(idx), i + post + 1)
        win = ar.iloc[lo:hi]
        rel = np.arange(lo - i, lo - i + len(win))
        rows.extend({"event_date": d, "t": int(t), "ar": float(v)} for t, v in zip(rel, win.values))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["car"] = out.groupby("event_date")["ar"].cumsum()
    return out

