"""Strategy validation helpers: walk-forward and overfit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_sharpe: float
    test_sharpe: float
    test_return: float


def _annualized_stats(r: pd.Series) -> tuple[float, float]:
    ann_ret = float(r.mean() * 252)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    return ann_ret, float(sharpe)


def walk_forward_splits(
    series: pd.Series,
    train: int = 252,
    test: int = 63,
    step: int = 63,
) -> list[tuple[slice, slice]]:
    n = len(series)
    out: list[tuple[slice, slice]] = []
    i = train
    while i + test <= n:
        out.append((slice(i - train, i), slice(i, i + test)))
        i += step
    return out


def walk_forward_backtest(
    returns: pd.Series,
    signal: pd.Series,
    train: int = 252,
    test: int = 63,
) -> list[WalkForwardResult]:
    r = returns.dropna().astype(float)
    s = signal.reindex(r.index).fillna(0.0)
    strat = s.shift(1).fillna(0.0) * r
    splits = walk_forward_splits(strat, train=train, test=test, step=test)
    rows: list[WalkForwardResult] = []
    for k, (tr, te) in enumerate(splits, start=1):
        tr_s = strat.iloc[tr]
        te_s = strat.iloc[te]
        tr_ret, tr_sh = _annualized_stats(tr_s)
        te_ret, te_sh = _annualized_stats(te_s)
        rows.append(
            WalkForwardResult(
                split_id=k,
                train_start=tr_s.index[0],
                train_end=tr_s.index[-1],
                test_start=te_s.index[0],
                test_end=te_s.index[-1],
                train_sharpe=tr_sh,
                test_sharpe=te_sh,
                test_return=te_ret,
            )
        )
    return rows


def overfit_gap_score(wf: list[WalkForwardResult]) -> float:
    if not wf:
        return np.nan
    train = np.array([x.train_sharpe for x in wf], dtype=float)
    test = np.array([x.test_sharpe for x in wf], dtype=float)
    with np.errstate(invalid="ignore"):
        gap = np.nanmean(train - test)
    return float(gap)

