"""Random matrix theory — Marčenko–Pastur bounds and denoising."""

from __future__ import annotations

import numpy as np
import pandas as pd


def marchenko_pastur_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """q = n_assets / n_obs (must be < 1 for MP bulk)."""
    if not (0 < q < 1):
        raise ValueError("q must be in (0, 1) for Marcenko-Pastur bounds")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive")
    lam_minus = sigma2 * (1 - np.sqrt(q)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    return float(lam_minus), float(lam_plus)


def mp_density(lam: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    lam_minus, lam_plus = marchenko_pastur_bounds(q, sigma2)
    rho = np.zeros_like(lam, dtype=float)
    mask = (lam >= lam_minus) & (lam <= lam_plus)
    rho[mask] = (
        np.sqrt((lam_plus - lam[mask]) * (lam[mask] - lam_minus))
        / (2 * np.pi * sigma2 * q * lam[mask])
    )
    return rho


def correlation_eigenvalues(returns: pd.DataFrame) -> np.ndarray:
    if returns is None or returns.empty:
        raise ValueError("returns must be non-empty")
    corr = returns.dropna().corr().values
    vals = np.linalg.eigvalsh(corr)
    return np.sort(vals)[::-1]


def effective_rank(eigenvalues: np.ndarray) -> float:
    """Entropy-based effective rank: exp(H) where H = −Σ p_i log p_i, p_i = λ_i/Σλ."""
    lam = np.asarray(eigenvalues, dtype=float)
    lam = np.clip(lam, 1e-12, None)
    p = lam / lam.sum()
    h = -np.sum(p * np.log(p))
    return float(np.exp(h))


def _estimate_mp_sigma2(vals: np.ndarray, q: float) -> float:
    # Scale MP bulk to empirical spectrum by matching central tendency.
    median_raw = float(np.median(vals))
    mp_median_unit = float(np.median(np.linspace(*marchenko_pastur_bounds(q, sigma2=1.0), num=200)))
    return max(median_raw / max(mp_median_unit, 1e-12), 1e-8)


def _nearest_psd_correlation(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 1e-10, None)
    psd = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
    corr = psd / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return corr


def _clean_eigenvalues(vals: np.ndarray, q: float, method: str) -> np.ndarray:
    sigma2_hat = _estimate_mp_sigma2(vals, q)
    lam_minus, lam_plus = marchenko_pastur_bounds(q, sigma2=sigma2_hat)
    vals_clean = vals.copy()
    noise_mask = (vals >= lam_minus) & (vals <= lam_plus)
    if not noise_mask.any():
        return vals_clean
    bulk_mean = float(np.mean(vals[noise_mask]))
    if method == "constant":
        vals_clean[noise_mask] = bulk_mean
        # Preserve trace (sum eigenvalues = matrix dimension for correlation)
        target = float(np.sum(vals))
        current = float(np.sum(vals_clean))
        if current > 1e-12:
            vals_clean *= target / current
    else:
        vals_clean[noise_mask] = bulk_mean
    return vals_clean


def denoise_correlation(
    returns: pd.DataFrame,
    q: float | None = None,
    method: str = "mean",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Keep eigenvectors outside MP bulk; shrink noise eigenvalues (mean or constant)."""
    if returns is None or returns.empty:
        raise ValueError("returns must be non-empty")
    r = returns.dropna()
    n_obs, n_assets = r.shape
    if n_obs <= n_assets:
        raise ValueError("need n_obs > n_assets for stable RMT denoising")
    if q is None:
        q = n_assets / n_obs
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1)")
    corr = r.corr().values
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    vals_clean = _clean_eigenvalues(vals, q, method)
    corr_clean = vecs @ np.diag(vals_clean) @ vecs.T
    corr_clean = _nearest_psd_correlation(corr_clean)
    cols = list(returns.columns)
    return pd.DataFrame(corr_clean, index=cols, columns=cols), vals, vals_clean


def constant_residual_eigenvalue_method(
    returns: pd.DataFrame,
    q: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """López de Prado style: replace MP-bulk eigenvalues with their mean."""
    return denoise_correlation(returns, q=q, method="constant")


def eigenvalue_comparison_table(
    raw_evals: np.ndarray,
    clean_evals: np.ndarray,
    q: float,
) -> pd.DataFrame:
    lam_minus, lam_plus = marchenko_pastur_bounds(q)
    idx = np.arange(1, len(raw_evals) + 1)
    in_bulk = (raw_evals >= lam_minus) & (raw_evals <= lam_plus)
    return pd.DataFrame(
        {
            "index": idx,
            "lambda_raw": raw_evals,
            "lambda_clean": clean_evals,
            "in_mp_bulk": in_bulk,
        }
    )


def portfolio_variance(weights: np.ndarray, corr: np.ndarray, vols: np.ndarray) -> float:
    """Portfolio variance wᵀ Σ w with Σ = D corr D, D = diag(vols)."""
    w = np.asarray(weights, dtype=float).ravel()
    d = np.diag(vols)
    cov = d @ corr @ d
    return float(w @ cov @ w)


def equal_weight_variance(corr: pd.DataFrame, vols: pd.Series | None = None) -> float:
    n = corr.shape[0]
    w = np.ones(n) / n
    if vols is None:
        vols = pd.Series(np.ones(n), index=corr.index)
    v = vols.reindex(corr.index).values.astype(float)
    return portfolio_variance(w, corr.values, v)
