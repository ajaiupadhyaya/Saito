"""Brownian motion, GBM, OU, Itô lemma demos."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats


def _require_positive_int(value: int, name: str) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_series(series: pd.Series, *, min_len: int = 2, name: str = "series") -> np.ndarray:
    if series is None:
        raise ValueError(f"{name} must not be None")
    x = series.dropna().astype(float).values
    if x.size < min_len:
        raise ValueError(f"{name} must contain at least {min_len} observations")
    return x


def simulate_brownian(
    n_steps: int,
    n_paths: int = 5,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    _require_positive_int(n_steps, "n_steps")
    _require_positive_int(n_paths, "n_paths")
    if dt <= 0:
        raise ValueError("dt must be positive")
    rng = np.random.default_rng(seed)
    z = rng.normal(0, np.sqrt(dt), size=(n_steps, n_paths))
    return np.cumsum(z, axis=0)


def brownian_increment_stats(paths: np.ndarray, dt: float = 1 / 252) -> dict[str, float]:
    """Diagnostics on BM increments: mean≈0, var≈dt, quadratic variation≈T."""
    diffs = np.diff(paths, axis=0)
    flat = diffs.ravel()
    n_increments = diffs.shape[0]
    t_horizon = n_increments * dt
    # QV per path, then average (do not sum across paths)
    qv_per_path = np.sum(diffs**2, axis=0)
    qv = float(np.mean(qv_per_path))
    return {
        "increment_mean": float(np.mean(flat)),
        "increment_var": float(np.var(flat, ddof=1)),
        "expected_var": float(dt),
        "quadratic_variation": qv,
        "expected_qv": float(t_horizon),
        "qv_ratio": qv / max(t_horizon, 1e-12),
    }


def euler_maruyama(
    drift_fn: Callable[[float, float], float],
    diff_fn: Callable[[float, float], float],
    x0: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    """Scalar Euler–Maruyama: X_{t+1} = X_t + a(t,X)dt + b(t,X)√dt Z."""
    _require_positive_int(n_steps, "n_steps")
    if dt <= 0:
        raise ValueError("dt must be positive")
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps, dtype=float)
    x[0] = x0
    t = 0.0
    for i in range(1, n_steps):
        z = rng.normal()
        xi = x[i - 1]
        x[i] = xi + drift_fn(t, xi) * dt + diff_fn(t, xi) * np.sqrt(dt) * z
        t += dt
    return x


def calibrate_gbm(prices: pd.Series) -> tuple[float, float]:
    if prices is None:
        raise ValueError("prices must not be None")
    px = prices.dropna().astype(float)
    if px.shape[0] < 3:
        raise ValueError("need at least 3 prices to calibrate GBM")
    if np.any(px.values <= 0):
        raise ValueError("GBM calibration requires strictly positive prices")
    log_ret = np.log(px / px.shift(1)).dropna()
    mu = log_ret.mean() * 252
    sigma = log_ret.std(ddof=1) * np.sqrt(252)
    return float(mu), float(sigma)


def gbm_log_moments(mu: float, sigma: float, t: float) -> tuple[float, float]:
    """E[ln S_t] and Var[ln S_t] under GBM starting at ln S_0 = 0."""
    mean_log = (mu - 0.5 * sigma**2) * t
    var_log = sigma**2 * t
    return float(mean_log), float(var_log)


def gbm_terminal_quantiles(
    s0: float,
    mu: float,
    sigma: float,
    t: float,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, float]:
    """Analytic log-normal quantiles of S_t."""
    m, v = gbm_log_moments(mu, sigma, t)
    log_s0 = np.log(s0)
    out = {}
    for q in quantiles:
        z = stats.norm.ppf(q)
        out[f"p{int(q * 100)}"] = float(np.exp(log_s0 + m + np.sqrt(v) * z))
    return out


def simulate_gbm(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    n_paths: int = 20,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    _require_positive_int(n_steps, "n_steps")
    _require_positive_int(n_paths, "n_paths")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if s0 <= 0:
        raise ValueError("s0 must be positive")
    rng = np.random.default_rng(seed)
    drift = (mu - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    z = rng.normal(drift, vol, size=(n_steps, n_paths))
    log_s = np.log(s0) + np.cumsum(z, axis=0)
    return np.exp(log_s)


def simulate_gbm_euler(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    """GBM via generic Euler–Maruyama (drift μS, diffusion σS)."""
    if s0 <= 0:
        raise ValueError("s0 must be positive")

    def drift(_t: float, s: float) -> float:
        return mu * s

    def diff(_t: float, s: float) -> float:
        return sigma * s

    return euler_maruyama(drift, diff, s0, n_steps, dt=dt, seed=seed)


def ou_max_likelihood(series: pd.Series, dt: float = 1 / 252) -> dict[str, float]:
    """Discrete-time Gaussian MLE for OU under exact transition density."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    x = _validate_series(series, min_len=10, name="series")
    x_lag = x[:-1]
    x_now = x[1:]
    x_lag_bar = float(np.mean(x_lag))
    x_now_bar = float(np.mean(x_now))
    denom = float(np.sum((x_lag - x_lag_bar) ** 2))
    if denom <= 1e-14:
        raise ValueError("OU fit failed: near-constant lag series")
    b = float(np.sum((x_lag - x_lag_bar) * (x_now - x_now_bar)) / denom)
    if not (0 < b < 1):
        raise ValueError("OU fit invalid: estimated AR(1) coefficient not in (0,1)")
    a = x_now_bar - b * x_lag_bar
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    resid = x_now - (a + b * x_lag)
    n = resid.size
    sigma_eps2 = float(np.sum(resid**2) / n)
    one_minus_b2 = max(1 - b * b, 1e-12)
    sigma = np.sqrt(max(sigma_eps2 * (2 * theta) / one_minus_b2, 0.0))
    half_life = np.log(2) / theta if theta > 0 else np.inf
    return {
        "theta": float(theta),
        "mu": float(mu),
        "sigma": float(sigma),
        "half_life": float(half_life),
        "estimate_method": "exact_discrete_mle",
    }


def fit_ou(series: pd.Series, dt: float = 1 / 252) -> dict[str, float]:
    """Alias for strict OU estimator used across pages/tests."""
    return ou_max_likelihood(series, dt=dt)


def ou_stationary_var(theta: float, sigma: float) -> float:
    """Long-run variance σ²/(2θ) of OU with dX = θ(μ−X)dt + σ dW."""
    if theta <= 0:
        return float("inf")
    return float(sigma**2 / (2 * theta))


def simulate_ou(
    x0: float,
    theta: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    _require_positive_int(n_steps, "n_steps")
    if dt <= 0:
        raise ValueError("dt must be positive")
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps)
    x[0] = x0
    for t in range(1, n_steps):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    return x


def ou_sim_vs_analytic(
    theta: float,
    mu: float,
    sigma: float,
    n_steps: int = 2000,
    dt: float = 1 / 252,
    seed: int = 42,
) -> dict[str, float]:
    """Compare simulated half-life and variance to OU theory."""
    sim = simulate_ou(mu, theta, mu, sigma, n_steps, dt=dt, seed=seed)
    fitted = fit_ou(pd.Series(sim), dt=dt)
    theory_hl = np.log(2) / theta if theta > 0 else float("inf")
    theory_var = ou_stationary_var(theta, sigma)
    sim_var = float(np.var(sim))
    return {
        "theta_true": float(theta),
        "theta_hat": fitted["theta"],
        "half_life_true": float(theory_hl),
        "half_life_hat": fitted["half_life"],
        "var_true": float(theory_var),
        "var_sim": sim_var,
    }


def ito_log_demo(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compare naive d(log S) vs Itô-correct log dynamics under GBM."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n_steps)
    s[0] = s0
    log_s_ito = np.zeros(n_steps)
    log_s_ito[0] = np.log(s0)
    log_s_naive = np.zeros(n_steps)
    log_s_naive[0] = np.log(s0)
    for t in range(1, n_steps):
        z = rng.normal()
        ds = mu * s[t - 1] * dt + sigma * s[t - 1] * np.sqrt(dt) * z
        s[t] = max(s[t - 1] + ds, 1e-8)
        log_s_ito[t] = log_s_ito[t - 1] + (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        log_s_naive[t] = log_s_naive[t - 1] + ds / s[t - 1]
    return {"prices": s, "log_ito": log_s_ito, "log_naive": log_s_naive}


def apply_ito_lemma_gbm_log(mu: float, sigma: float, t: float) -> dict[str, float]:
    """Itô drift correction for ln S under GBM: cumulative −½σ²t term."""
    correction = -0.5 * sigma**2 * t
    naive_drift = mu * t
    ito_drift = (mu - 0.5 * sigma**2) * t
    return {
        "naive_log_drift": float(naive_drift),
        "ito_log_drift": float(ito_drift),
        "correction": float(correction),
        "horizon": float(t),
    }


def ito_power_demo(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt: float = 1 / 252,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Itô on f(S)=S²: d(S²) has extra +σ²S² dt vs naive chain rule."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n_steps)
    f_ito = np.zeros(n_steps)
    f_naive = np.zeros(n_steps)
    s[0] = s0
    f_ito[0] = s0**2
    f_naive[0] = s0**2
    for t in range(1, n_steps):
        z = rng.normal()
        st_1 = s[t - 1]
        ds = mu * st_1 * dt + sigma * st_1 * np.sqrt(dt) * z
        s[t] = max(st_1 + ds, 1e-8)
        # Itô: d(S²) = 2S dS + σ²S² dt
        f_ito[t] = f_ito[t - 1] + 2 * st_1 * ds + (sigma * st_1) ** 2 * dt
        # Naive: df ≈ 2S dS only
        f_naive[t] = f_naive[t - 1] + 2 * st_1 * ds
    return {"prices": s, "f_ito": f_ito, "f_naive": f_naive, "f_actual": s**2}
