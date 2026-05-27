"""Research metrics for risk, factor diagnostics, and strategy backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    if window < 5:
        raise ValueError("window must be >= 5")
    panel = pd.concat([asset.rename("asset"), benchmark.rename("bench")], axis=1).dropna()
    cov = panel["asset"].rolling(window).cov(panel["bench"])
    var = panel["bench"].rolling(window).var(ddof=1)
    beta = cov / var.replace(0.0, np.nan)
    return beta.dropna()


def var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if r.empty:
        raise ValueError("returns must be non-empty")
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be in (0.5, 1.0)")
    losses = -r
    q = float(losses.quantile(alpha))
    tail = losses[losses >= q]
    cvar = float(tail.mean()) if not tail.empty else q
    return {"var": q, "cvar": cvar, "tail_count": int(tail.shape[0])}


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    metrics: dict[str, float]


def run_signal_backtest(prices: pd.Series, signal: pd.Series, fee_bps: float = 2.0) -> BacktestResult:
    p = prices.dropna().astype(float)
    s = signal.reindex(p.index).fillna(0.0).clip(-1.0, 1.0)
    rets = p.pct_change().fillna(0.0)
    pos = s.shift(1).fillna(0.0)
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    costs = turnover * (fee_bps / 10000.0)
    strat = pos * rets - costs
    equity = (1.0 + strat).cumprod()
    ann_mean = float(strat.mean() * 252)
    ann_vol = float(strat.std(ddof=1) * np.sqrt(252))
    sharpe = ann_mean / ann_vol if ann_vol > 1e-12 else np.nan
    dd = equity / equity.cummax() - 1.0
    metrics = {
        "ann_return": ann_mean,
        "ann_vol": ann_vol,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_drawdown": float(dd.min()),
        "turnover": float(turnover.mean() * 252),
    }
    return BacktestResult(equity_curve=equity, returns=strat, metrics=metrics)

