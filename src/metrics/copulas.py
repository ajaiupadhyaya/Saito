"""Gaussian and Student-t copula fits via rank transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class CopulaResult:
    family: str
    correlation: np.ndarray
    df: float | None
    samples: np.ndarray
    parameter: float | None = None


@dataclass
class TailDependence:
    lambda_lower: float
    lambda_upper: float


def pseudo_observations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    u = stats.rankdata(x) / (n + 1)
    v = stats.rankdata(y) / (n + 1)
    return u, v


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    u, v = pseudo_observations(x, y)
    res = stats.kendalltau(u, v)
    return float(res.correlation if res.correlation is not None else 0.0)


def empirical_copula_grid(x: np.ndarray, y: np.ndarray, grid: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = pseudo_observations(x, y)
    edges = np.linspace(0, 1, grid + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    H = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            H[i, j] = np.mean((u <= edges[i + 1]) & (u > edges[i]) & (v <= edges[j + 1]) & (v > edges[j]))
    return H, centers, centers


def fit_gaussian_copula(x: np.ndarray, y: np.ndarray, n_samples: int = 500) -> CopulaResult:
    u, v = pseudo_observations(x, y)
    z1 = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
    z2 = stats.norm.ppf(np.clip(v, 1e-6, 1 - 1e-6))
    rho = np.corrcoef(z1, z2)[0, 1]
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(42)
    z = rng.multivariate_normal([0, 0], cov, size=n_samples)
    samples = stats.norm.cdf(z)
    return CopulaResult("gaussian", cov, None, samples, parameter=float(rho))


def fit_student_t_copula(
    x: np.ndarray, y: np.ndarray, df: float = 5.0, n_samples: int = 500
) -> CopulaResult:
    u, v = pseudo_observations(x, y)
    z1 = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df)
    z2 = stats.t.ppf(np.clip(v, 1e-6, 1 - 1e-6), df)
    rho = np.corrcoef(z1, z2)[0, 1]
    cov = np.array([[1.0, rho], [rho, 1.0]])
    rng = np.random.default_rng(42)
    z = rng.multivariate_normal([0, 0], cov, size=n_samples)
    s = rng.chisquare(df, n_samples) / df
    t = z / np.sqrt(s[:, None])
    samples = stats.t.cdf(t, df)
    return CopulaResult("student_t", cov, df, samples, parameter=float(rho))


def _clayton_theta_from_tau(tau: float) -> float:
    return max(2 * tau / max(1 - tau, 1e-6), 1e-6)


def _gumbel_theta_from_tau(tau: float) -> float:
    return max(1 / (1 - tau), 1.01)


def _frank_theta_from_tau(tau: float) -> float:
    if abs(tau) < 1e-6:
        return 1e-6

    def tau_from_theta(th: float) -> float:
        if abs(th) < 1e-8:
            return 0.0
        et = np.exp(th)
        return 1 - 4 / th + 12 / (th * (et - 1))

    def sq_err(th: float) -> float:
        d = tau_from_theta(th) - tau
        return float(d * d)

    if tau > 0:
        bracket = (1e-4, 50.0)
    else:
        bracket = (-50.0, -1e-4)
    try:
        return float(optimize.brentq(lambda th: tau_from_theta(th) - tau, *bracket))
    except ValueError:
        res = optimize.minimize_scalar(sq_err, bounds=bracket, method="bounded")
        return float(res.x)


def fit_clayton_copula(x: np.ndarray, y: np.ndarray, n_samples: int = 500) -> CopulaResult:
    tau = kendall_tau(x, y)
    theta = _clayton_theta_from_tau(tau)
    samples = archimedean_samples("clayton", theta, n_samples)
    return CopulaResult("clayton", np.eye(2), None, samples, parameter=theta)


def fit_gumbel_copula(x: np.ndarray, y: np.ndarray, n_samples: int = 500) -> CopulaResult:
    tau = kendall_tau(x, y)
    theta = _gumbel_theta_from_tau(tau)
    samples = archimedean_samples("gumbel", theta, n_samples)
    return CopulaResult("gumbel", np.eye(2), None, samples, parameter=theta)


def fit_frank_copula(x: np.ndarray, y: np.ndarray, n_samples: int = 500) -> CopulaResult:
    tau = kendall_tau(x, y)
    theta = _frank_theta_from_tau(tau)
    samples = archimedean_samples("frank", theta, n_samples)
    return CopulaResult("frank", np.eye(2), None, samples, parameter=theta)


def archimedean_samples(family: str, theta: float, n: int = 500, seed: int = 42) -> np.ndarray:
    family = family.lower()
    if family == "clayton":
        return clayton_samples(theta, n, seed)
    rng = np.random.default_rng(seed)
    if family == "gumbel":
        v = rng.gamma(1.0 / theta, 1.0, n)
        e1 = rng.exponential(1.0, n)
        e2 = rng.exponential(1.0, n)
        u1 = np.exp(-((e1 / v) ** (1.0 / theta)))
        u2 = np.exp(-((e2 / v) ** (1.0 / theta)))
    elif family == "frank":
        u1 = rng.uniform(1e-6, 1 - 1e-6, n)
        w = rng.uniform(1e-6, 1 - 1e-6, n)
        num = w * (1 - np.exp(-theta))
        den = (1 - w) * (1 - np.exp(-theta * u1))
        ratio = np.clip(1 + num / np.maximum(den, 1e-12), 1e-12, None)
        u2 = -np.log(ratio) / theta
    else:
        raise ValueError(f"Unknown Archimedean family: {family}")
    u1 = np.clip(u1 if family != "frank" else u1, 1e-6, 1 - 1e-6)
    u2 = np.clip(u2, 1e-6, 1 - 1e-6)
    return np.column_stack([u1, u2])


def clayton_samples(theta: float = 2.0, n: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = max(float(theta), 1e-6)
    v = rng.gamma(1 / theta, 1, n)
    u1 = rng.uniform(1e-6, 1 - 1e-6, n)
    inner = 1 + np.clip(u1 ** (-theta) * (v ** (-1) - 1), 1e-12, 1e12)
    u2 = inner ** (-1 / theta)
    u2 = np.clip(u2, 1e-6, 1 - 1e-6)
    return np.column_stack([u1, u2])


def tail_dependence_coefficients(
    family: str,
    *,
    rho: float | None = None,
    df: float | None = None,
    theta: float | None = None,
) -> TailDependence:
    family = family.lower()
    if family == "gaussian":
        return TailDependence(0.0, 0.0)
    if family == "student_t" and df is not None and rho is not None:
        tq = stats.t.ppf(0.999, df)
        lam = 2 * stats.t.cdf(-tq * np.sqrt((df + 1) / (1 - rho)), df + 1)
        return TailDependence(float(lam), float(lam))
    if family == "clayton" and theta is not None:
        return TailDependence(float(2 ** (-1 / theta)), 0.0)
    if family == "gumbel" and theta is not None:
        return TailDependence(0.0, float(2 - 2 ** (1 / theta)))
    if family == "frank":
        return TailDependence(0.0, 0.0)
    return TailDependence(0.0, 0.0)


def conditional_sample_y_given_x(
    u_val: float,
    samples: np.ndarray,
    bandwidth: float = 0.05,
) -> np.ndarray:
    """Empirical conditional copula samples v | u ≈ u_val."""
    u_val = float(np.clip(u_val, 0.01, 0.99))
    mask = np.abs(samples[:, 0] - u_val) <= bandwidth
    if mask.sum() < 5:
        return samples[:, 1]
    return samples[mask, 1]
