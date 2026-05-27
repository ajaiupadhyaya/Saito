"""Distribution fitting and CLT / joint probability demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _as_1d_float(x: np.ndarray, *, min_len: int = 2, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < min_len:
        raise ValueError(f"{name} must contain at least {min_len} finite observations")
    return arr


@dataclass
class GaussianBivariateMoments:
    mean_x: float
    mean_y: float
    std_x: float
    std_y: float
    rho: float
    beta_y_on_x: float
    conditional_var_y_given_x: float


def gaussian_bivariate_moments(x: np.ndarray, y: np.ndarray) -> GaussianBivariateMoments:
    """Method-of-moment Gaussian pair using sample moments (ddof=1)."""
    xv = _as_1d_float(x, min_len=2, name="x")
    yv = _as_1d_float(y, min_len=2, name="y")
    n = min(len(xv), len(yv))
    if n < 2:
        raise ValueError("x and y must have at least 2 aligned observations")
    xv, yv = xv[:n], yv[:n]
    mx, my = float(np.mean(xv)), float(np.mean(yv))
    cx = xv - mx
    cy = yv - my
    denom = max(n - 1, 1)
    var_x = float(np.sum(cx**2) / denom)
    var_y = float(np.sum(cy**2) / denom)
    cov_xy = float(np.sum(cx * cy) / denom)
    sx = np.sqrt(max(var_x, 1e-18))
    sy = np.sqrt(max(var_y, 1e-18))
    rho = float(np.clip(cov_xy / (sx * sy), -1.0, 1.0))
    beta = rho * sy / sx if sx > 1e-18 else 0.0
    cond_var = max((1 - rho * rho), 0.0) * sy**2
    return GaussianBivariateMoments(mx, my, sx, sy, rho, beta, float(cond_var))


def conditional_gaussian_mean_var_y_given_x(
    x_val: float, params: GaussianBivariateMoments
) -> tuple[float, float]:
    """If (X,Y) is bivariate normal with given moments, return E[Y|X=x], Var(Y|X=x)."""
    mu_y_x = params.mean_y + params.rho * (params.std_y / max(params.std_x, 1e-18)) * (x_val - params.mean_x)
    return mu_y_x, params.conditional_var_y_given_x


@dataclass
class FitResult:
    name: str
    params: dict
    aic: float
    bic: float
    loglik: float


def fit_normal(x: np.ndarray) -> FitResult:
    x = _as_1d_float(x, min_len=2, name="x")
    mu, sigma = stats.norm.fit(x)
    sigma = max(float(sigma), 1e-12)
    ll = np.sum(stats.norm.logpdf(x, mu, sigma))
    k = 2
    n = len(x)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return FitResult("normal", {"mu": mu, "sigma": sigma}, aic, bic, ll)


def fit_student_t(x: np.ndarray) -> FitResult:
    x = _as_1d_float(x, min_len=3, name="x")
    df, loc, scale = stats.t.fit(x)
    scale = max(float(scale), 1e-12)
    ll = np.sum(stats.t.logpdf(x, df, loc, scale))
    k = 3
    n = len(x)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return FitResult("student_t", {"df": df, "loc": loc, "scale": scale}, aic, bic, ll)


def fit_binomial(counts: np.ndarray, n_trials: int) -> FitResult:
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    counts = _as_1d_float(counts, min_len=2, name="counts")
    if np.any((counts < 0) | (counts > n_trials)):
        raise ValueError("counts must lie in [0, n_trials]")
    p = counts.mean() / n_trials
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll = np.sum(stats.binom.logpmf(counts, n_trials, p))
    k = 1
    n = len(counts)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return FitResult("binomial", {"n": n_trials, "p": p}, aic, bic, ll)


def fit_poisson(counts: np.ndarray) -> FitResult:
    counts = _as_1d_float(counts, min_len=2, name="counts")
    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")
    lam = counts.mean()
    lam = max(lam, 1e-6)
    ll = np.sum(stats.poisson.logpmf(counts.astype(int), lam))
    k = 1
    n = len(counts)
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return FitResult("poisson", {"lambda": lam}, aic, bic, ll)


def clt_sample_means(x: np.ndarray, sample_sizes: list[int], n_replicates: int = 500) -> dict[int, np.ndarray]:
    x = _as_1d_float(x, min_len=2, name="x")
    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    rng = np.random.default_rng(42)
    out = {}
    for n in sample_sizes:
        if n <= 0:
            raise ValueError("sample_sizes entries must be positive")
        means = [rng.choice(x, size=n, replace=True).mean() for _ in range(n_replicates)]
        out[n] = np.array(means)
    return out


def clt_standardized_sample_means(x: np.ndarray, n: int, n_replicates: int = 2000) -> np.ndarray:
    r"""Standardized bootstrap means: \(\sqrt{n}(\bar{X}-\mu)/\sigma\) ≈ \(\mathcal{N}(0,1)\)."""
    x = _as_1d_float(x, min_len=2, name="x")
    if n <= 0 or n_replicates <= 0:
        raise ValueError("n and n_replicates must be positive")
    rng = np.random.default_rng(42)
    mu, sigma = x.mean(), x.std(ddof=1)
    sigma = max(sigma, 1e-12)
    draws = rng.choice(x, size=(n_replicates, n), replace=True)
    bars = draws.mean(axis=1)
    return np.sqrt(n) * (bars - mu) / sigma


def qq_normal_coordinates(x: np.ndarray, mu: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    x = _as_1d_float(x, min_len=2, name="x")
    sigma = max(sigma, 1e-12)
    x_sorted = np.sort(x)
    n = len(x_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theory = mu + sigma * stats.norm.ppf(probs)
    return theory, x_sorted


def qq_student_t_coordinates(x: np.ndarray, df: float, loc: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    x_sorted = np.sort(_as_1d_float(x, min_len=2, name="x"))
    n = len(x_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theory = stats.t.ppf(probs, df, loc, scale)
    return theory, x_sorted


def joint_binning(x: np.ndarray, y: np.ndarray, bins: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    x = _as_1d_float(x, min_len=2, name="x")
    y = _as_1d_float(y, min_len=2, name="y")
    n = min(len(x), len(y))
    if n < 2:
        raise ValueError("x and y must have at least 2 aligned observations")
    x = x[:n]
    y = y[:n]
    H, xe, ye = np.histogram2d(x, y, bins=bins, density=True)
    return H, xe, ye


def conditional_hist(
    x: np.ndarray, y: np.ndarray, x_value: float, bandwidth: float
) -> tuple[np.ndarray, np.ndarray]:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    x = _as_1d_float(x, min_len=2, name="x")
    y = _as_1d_float(y, min_len=2, name="y")
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    mask = np.abs(x - x_value) <= bandwidth
    if mask.sum() < 5:
        return np.array([]), np.array([])
    subset = y[mask]
    counts, edges = np.histogram(subset, bins=15, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def trade_win_loss_counts(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """Wins in rolling window as binomial count data."""
    if window <= 0:
        raise ValueError("window must be positive")
    returns = _as_1d_float(returns, min_len=window, name="returns")
    wins = (returns > 0).astype(int)
    counts = pd.Series(wins).rolling(window).sum().dropna().values
    return counts
