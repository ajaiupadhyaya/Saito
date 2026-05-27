"""Portfolio and risk analytics for research workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize


@dataclass
class FrontierPoint:
    target_return: float
    volatility: float
    weights: np.ndarray


def annualized_mu_cov(returns: pd.DataFrame, periods: int = 252) -> tuple[np.ndarray, np.ndarray]:
    r = returns.dropna(how="any").astype(float)
    if r.empty:
        raise ValueError("returns must have complete rows")
    mu = r.mean().values * periods
    cov = r.cov().values * periods
    return mu, cov


def efficient_frontier_constrained(mu: np.ndarray, cov: np.ndarray, n_points: int = 20) -> list[FrontierPoint]:
    n = len(mu)
    targets = np.linspace(float(np.min(mu)), float(np.max(mu)), n_points)
    bounds = [(0.0, 1.0)] * n
    pts: list[FrontierPoint] = []

    def var_obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu - t)},
        ]
        res = minimize(var_obj, np.ones(n) / n, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w = np.asarray(res.x, dtype=float)
            pts.append(
                FrontierPoint(
                    target_return=float(target),
                    volatility=float(np.sqrt(max(var_obj(w), 0.0))),
                    weights=w,
                )
            )
    if not pts:
        raise ValueError("no feasible frontier points")
    return pts


def risk_parity_weights(cov: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n, dtype=float) / n
    target = 1.0 / n
    for _ in range(max_iter):
        mrc = cov @ w
        port_var = float(w @ mrc)
        if port_var <= 0:
            break
        rc = w * mrc / port_var
        err = rc - target
        if np.max(np.abs(err)) < tol:
            break
        w = np.clip(w * (target / np.clip(rc, 1e-10, None)), 1e-8, None)
        w /= w.sum()
    return w


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    r = returns.dropna(how="any").astype(float)
    if r.shape[1] < 2:
        raise ValueError("need at least two assets for HRP")
    corr = r.corr().values
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    tri = dist[np.triu_indices_from(dist, k=1)]
    link = linkage(tri, method="single")
    order = leaves_list(link)
    cols = list(r.columns[order])
    cov = r[cols].cov().values
    ivp = 1.0 / np.clip(np.diag(cov), 1e-12, None)
    w = ivp / np.sum(ivp)
    out = pd.Series(w, index=cols)
    return out.reindex(r.columns).fillna(0.0)


def factor_exposure(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> tuple[pd.Series, float]:
    panel = pd.concat([asset_returns.rename("asset"), factor_returns], axis=1).dropna()
    if panel.shape[0] < 20:
        raise ValueError("need at least 20 observations for factor regression")
    y = panel["asset"].values
    Xf = panel.drop(columns=["asset"]).values
    X = np.column_stack([np.ones(len(panel)), Xf])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    exposures = pd.Series(beta[1:], index=panel.drop(columns=["asset"]).columns, name="exposure")
    return exposures, float(r2)


def rolling_stress_var(returns: pd.Series, window: int = 63, alpha: float = 0.95) -> pd.Series:
    r = returns.dropna().astype(float)
    if window < 20:
        raise ValueError("window must be >= 20")
    out = []
    idx = []
    for i in range(window, len(r) + 1):
        sub = r.iloc[i - window : i]
        var = float((-sub).quantile(alpha))
        out.append(var)
        idx.append(sub.index[-1])
    return pd.Series(out, index=idx, name=f"var_{int(alpha*100)}")

