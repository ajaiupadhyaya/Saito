"""Extreme value theory — peaks over threshold / GPD."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def _as_losses(x: np.ndarray, *, min_len: int = 30) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < min_len:
        raise ValueError(f"need at least {min_len} finite observations for EVT fit")
    return arr


def _pot_var_level(
    *,
    threshold: float,
    shape: float,
    scale: float,
    n_obs: int,
    n_exc: int,
    confidence: float,
) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be in (0,1)")
    if n_exc <= 0 or n_obs <= 0 or n_exc > n_obs:
        raise ValueError("invalid exceedance counts for POT")
    if scale <= 0:
        raise ValueError("GPD scale must be positive")
    tail_prob = 1.0 - confidence
    ratio = (n_obs / n_exc) * tail_prob
    ratio = max(ratio, 1e-14)
    if abs(shape) < 1e-10:
        exc = scale * np.log(1.0 / ratio)
    else:
        exc = (scale / shape) * (ratio ** (-shape) - 1.0)
    return float(threshold + exc)


def _pot_es_level(var_level: float, *, threshold: float, shape: float, scale: float) -> float:
    if shape >= 1.0:
        return float("nan")
    return float((var_level + scale - shape * threshold) / (1.0 - shape))


@dataclass
class GPDResult:
    threshold: float
    shape: float
    scale: float
    n_exceedances: int
    var_99: float
    es_99: float


def fit_gpd_pot(x: np.ndarray, threshold: float | None = None, quantile: float = 0.95) -> GPDResult:
    x = _as_losses(x)
    if threshold is None:
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0,1)")
        threshold = float(np.quantile(x, quantile))
    excess = x[x > threshold] - threshold
    if len(excess) < 10:
        raise ValueError("Too few exceedances; lower threshold or use more data")
    shape, loc, scale = stats.genpareto.fit(excess, floc=0)
    n, nu = len(x), len(excess)
    var_99 = _pot_var_level(
        threshold=float(threshold),
        shape=float(shape),
        scale=float(scale),
        n_obs=n,
        n_exc=nu,
        confidence=0.99,
    )
    es_99 = _pot_es_level(var_99, threshold=float(threshold), shape=float(shape), scale=float(scale))
    return GPDResult(threshold, shape, scale, len(excess), float(var_99), float(es_99))


def mean_excess_plot(x: np.ndarray, n_thresholds: int = 40) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(_as_losses(x))
    qs = np.linspace(0.85, 0.99, n_thresholds)
    thresholds = np.quantile(x, qs)
    means = []
    for u in thresholds:
        exc = x[x > u] - u
        means.append(exc.mean() if len(exc) > 0 else np.nan)
    return thresholds, np.array(means)


def mean_excess_stability(x: np.ndarray, n_thresholds: int = 30) -> tuple[float, np.ndarray, np.ndarray]:
    """Pick threshold u in upper quantile band where mean-excess slope is most stable."""
    thr, means = mean_excess_plot(x, n_thresholds=n_thresholds)
    valid = np.isfinite(means)
    thr, means = thr[valid], means[valid]
    if len(thr) < 5:
        u = float(np.quantile(_as_losses(x), 0.95))
        return u, thr, means
    slopes = np.abs(np.diff(means) / np.diff(thr))
    mid = len(slopes) // 3
    window = slopes[mid : mid + max(3, len(slopes) // 3)]
    idx = mid + int(np.argmin(window))
    u = float(thr[idx])
    return u, thr, means


def hill_tail_index(x: np.ndarray, k: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Hill estimator ξ = (1/k) Σ ln(X_{(i)}/X_{(k+1)}) on sorted losses (descending)."""
    xs = np.sort(_as_losses(x))[::-1]
    n = len(xs)
    if k is None:
        k = max(10, n // 20)
    k = min(max(k, 5), n - 2)
    ks = np.arange(5, min(n - 1, max(k * 2, 50)))
    x_k1 = xs[ks]
    logs = np.log(xs[: ks[-1] + 1][:, None] / x_k1[None, :])
    hill = np.mean(logs[: ks[-1], :], axis=0)
    xi = float(hill[ks.tolist().index(k)] if k in ks else hill[len(hill) // 2])
    return xi, ks, hill


def gpd_qq_coordinates(excess: np.ndarray, shape: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
    excess = np.sort(_as_losses(excess, min_len=2))
    n = len(excess)
    probs = (np.arange(1, n + 1) - 0.5) / n
    if abs(shape) < 1e-8:
        theory = scale * stats.expon.ppf(probs)
    else:
        theory = stats.genpareto.ppf(probs, shape, scale=scale)
    return theory, excess


def gpd_return_level(
    x: np.ndarray,
    threshold: float,
    shape: float,
    scale: float,
    probs: tuple[float, ...] = (0.95, 0.99, 0.995, 0.999),
) -> pd.DataFrame:
    """Return levels (quantiles of loss distribution) from POT/GPD."""
    x = _as_losses(x)
    n = len(x)
    nu = int(np.sum(x > threshold))
    rows = []
    for p in probs:
        rl = _pot_var_level(
            threshold=float(threshold),
            shape=float(shape),
            scale=float(scale),
            n_obs=n,
            n_exc=nu,
            confidence=float(p),
        )
        rows.append({"prob": p, "return_level": rl})
    return pd.DataFrame(rows)


def gpd_var_es_at_confidence(
    x: np.ndarray,
    threshold: float,
    shape: float,
    scale: float,
    n_exceedances: int,
    confidence: float,
) -> tuple[float, float]:
    """POT/GPD VaR and ES at arbitrary confidence level p (losses, right tail)."""
    x = _as_losses(x)
    n = len(x)
    nu = int(n_exceedances)
    var_level = _pot_var_level(
        threshold=float(threshold),
        shape=float(shape),
        scale=float(scale),
        n_obs=n,
        n_exc=nu,
        confidence=float(confidence),
    )
    es_level = _pot_es_level(
        var_level,
        threshold=float(threshold),
        shape=float(shape),
        scale=float(scale),
    )
    return float(var_level), float(es_level)


def var_es_comparison(
    x: np.ndarray,
    confidence: float = 0.99,
    gpd_threshold: float | None = None,
) -> pd.DataFrame:
    """Compare VaR/ES at confidence level: empirical, Normal, Student-t, GPD."""
    x = _as_losses(x)
    emp_var = float(np.quantile(x, confidence))
    emp_es = float(x[x >= emp_var].mean()) if np.any(x >= emp_var) else emp_var

    mu, sigma = stats.norm.fit(x)
    norm_var = float(stats.norm.ppf(confidence, mu, sigma))
    tail = x[x >= norm_var]
    norm_es = float(tail.mean()) if len(tail) else norm_var

    df, loc, sc = stats.t.fit(x)
    t_var = float(stats.t.ppf(confidence, df, loc, sc))
    tail_t = x[x >= t_var]
    t_es = float(tail_t.mean()) if len(tail_t) else t_var

    gpd = fit_gpd_pot(x, threshold=gpd_threshold)
    gpd_var, gpd_es = gpd_var_es_at_confidence(
        x, gpd.threshold, gpd.shape, gpd.scale, gpd.n_exceedances, confidence
    )

    return pd.DataFrame(
        {
            "method": ["empirical", "normal", "student_t", "gpd"],
            "VaR": [emp_var, norm_var, t_var, gpd_var],
            "ES": [emp_es, norm_es, t_es, gpd_es],
        }
    )
