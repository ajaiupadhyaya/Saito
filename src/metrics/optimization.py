"""SGD/Adam toy optimizers and constrained Markowitz demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _as_psd_cov(cov: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    s = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    vals, vecs = np.linalg.eigh(s)
    vals = np.clip(vals, eps, None)
    return vecs @ np.diag(vals) @ vecs.T


def _safe_solve(mat: np.ndarray, rhs: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    m = np.asarray(mat, dtype=float)
    b = np.asarray(rhs, dtype=float)
    try:
        return np.linalg.solve(m, b)
    except np.linalg.LinAlgError:
        n = m.shape[0]
        return np.linalg.solve(m + ridge * np.eye(n), b)


def quadratic_loss(w: np.ndarray) -> float:
    return float(w @ w)


def rosenbrock(xy: np.ndarray) -> float:
    x, y = xy[0], xy[1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def linear_mse_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    residuals = y - X @ w
    return float(np.mean(residuals**2))


def quadratic_grad(w: np.ndarray) -> np.ndarray:
    return 2.0 * w


def rosenbrock_grad(xy: np.ndarray) -> np.ndarray:
    x, y = float(xy[0]), float(xy[1])
    # ∇[(1−x)² + 100(y − x²)²]
    g0 = -2 * (1 - x) + 400 * x * (x**2 - y)
    g1 = 200 * (y - x**2)
    return np.array([g0, g1], dtype=float)


def linear_mse_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = X @ w - y
    return (2.0 / X.shape[0]) * (X.T @ r)


def ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return beta.astype(float)


def _numerical_gradient(loss_fn, x: np.ndarray, grad_eps: float) -> np.ndarray:
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += grad_eps
        xm[i] -= grad_eps
        grad[i] = (loss_fn(xp) - loss_fn(xm)) / (2 * grad_eps)
    return grad


def sgd_minimize(
    loss_fn,
    x0: np.ndarray,
    lr: float = 0.01,
    steps: int = 200,
    grad_fn=None,
    grad_eps: float = 1e-5,
) -> tuple[list[float], np.ndarray]:
    x = x0.astype(float).copy()
    history = []
    for _ in range(steps):
        history.append(float(loss_fn(x)))
        if grad_fn is not None:
            grad = np.asarray(grad_fn(x), dtype=float)
        else:
            grad = _numerical_gradient(loss_fn, x, grad_eps)
        x -= lr * grad
    return history, x


def adam_minimize(
    loss_fn,
    x0: np.ndarray,
    lr: float = 0.05,
    steps: int = 200,
    beta1: float = 0.9,
    beta2: float = 0.999,
    grad_fn=None,
    grad_eps: float = 1e-5,
) -> tuple[list[float], np.ndarray]:
    x = x0.astype(float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    history = []
    for t in range(1, steps + 1):
        history.append(float(loss_fn(x)))
        if grad_fn is not None:
            grad = np.asarray(grad_fn(x), dtype=float)
        else:
            grad = _numerical_gradient(loss_fn, x, grad_eps)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return history, x


@dataclass
class FrontierPoint:
    target_return: float
    weights: np.ndarray
    volatility: float


def markowitz_frontier(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 25,
    long_only: bool = True,
) -> list[FrontierPoint]:
    n = len(mu)
    if cov.shape != (n, n):
        raise ValueError("cov must have shape (n_assets, n_assets)")
    cov = _as_psd_cov(cov)
    targets = np.linspace(mu.min(), mu.max(), n_points)
    points = []

    def port_var(w):
        return float(w @ cov @ w)

    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(0.0, 1.0)] * n if long_only else None
        w0 = np.ones(n) / n
        res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            vol = np.sqrt(port_var(res.x))
            points.append(FrontierPoint(float(target), res.x, vol))
    if not points:
        raise ValueError("frontier optimization failed for all target returns")
    return points


def min_variance_portfolio(mu: np.ndarray, cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    n = len(mu)
    if cov.shape != (n, n):
        raise ValueError("cov must have shape (n_assets, n_assets)")
    cov = _as_psd_cov(cov)

    def port_var(w):
        return float(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else None
    w0 = np.ones(n) / n
    res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    return w0


def global_minimum_variance_weights_analytic(cov: np.ndarray) -> np.ndarray:
    """Closed form GMV with Σw=1 (short sales allowed): w ∝ Σ⁻¹𝟙."""
    one = np.ones(cov.shape[0], dtype=float)
    Sigma = _as_psd_cov(cov)
    z = _safe_solve(Sigma, one)
    s = float(z.sum())
    if abs(s) < 1e-14:
        return one / len(one)
    return z / s


def markowitz_min_variance_target_analytic(mu: np.ndarray, cov: np.ndarray, target_return: float) -> np.ndarray:
    """Lagrange multiplier (KKT) solution for min wᵀΣw s.t. 𝟙ᵀw=1, μᵀw=r with Σ SPD (shorts OK).

    w = Σ⁻¹(λ𝟙 + νμ) where [λ,ν] solves the 2×2 normal equations from constraints.
    """
    Sigma = _as_psd_cov(cov)
    m = np.asarray(mu, dtype=float).ravel()
    one = np.ones_like(m)
    inv1 = _safe_solve(Sigma, one)
    invm = _safe_solve(Sigma, m)
    A = np.array([[float(one @ inv1), float(one @ invm)], [float(m @ inv1), float(m @ invm)]], dtype=float)
    b = np.array([1.0, float(target_return)], dtype=float)
    lam = _safe_solve(A, b, ridge=1e-10)
    w = lam[0] * inv1 + lam[1] * invm
    return w


def tangency_portfolio_weights_uncstr(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> np.ndarray:
    """Max-Sharpe (tangency) with 𝟙ᵀw=1 and shorts OK: w ∝ Σ⁻¹(μ − rf·𝟙), then sum-normalize."""
    m = np.asarray(mu, dtype=float).ravel()
    Sigma = _as_psd_cov(cov)
    one = np.ones_like(m)
    excess = m - float(rf) * one
    z = _safe_solve(Sigma, excess)
    s = float(z.sum())
    if abs(s) < 1e-14:
        return one / len(one)
    return z / s


def portfolio_sharpe(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> float:
    ex = float(w @ mu) - float(rf)
    vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
    return ex / vol
