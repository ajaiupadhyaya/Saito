"""PCA, SVD, eigen decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAResult:
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    components: np.ndarray
    scores: np.ndarray
    column_names: list[str]
    explained_variance: np.ndarray


def pca_analysis(
    returns: pd.DataFrame,
    n_components: int | None = None,
    *,
    standardize: bool = True,
) -> PCAResult:
    df = returns.dropna()
    X_raw = df.values
    names = list(df.columns)
    if standardize:
        X = StandardScaler().fit_transform(X_raw)
    else:
        X = X_raw.astype(float)

    max_k = max(1, min(X.shape) - 1)
    if n_components is None:
        n_comp = max_k
    else:
        n_comp = max(1, min(n_components, max_k))

    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    return PCAResult(
        pca.explained_variance_ratio_,
        cum,
        pca.components_,
        scores,
        names,
        pca.explained_variance_,
    )


def biplot_arrow_scale(pca_scores: np.ndarray, loadings_scaled: np.ndarray) -> tuple[float, float]:
    """Return axis ranges that fit both scores and loading arrows."""
    xs = np.r_[pca_scores[:, 0], 0.0, loadings_scaled[:, 0]]
    ys = np.r_[pca_scores[:, 1], 0.0, loadings_scaled[:, 1]]
    pad_x = max(1e-6, 0.1 * np.ptp(xs))
    pad_y = max(1e-6, 0.1 * np.ptp(ys))
    return float(xs.min() - pad_x), float(xs.max() + pad_x), float(ys.min() - pad_y), float(ys.max() + pad_y)


def scaled_loadings_for_biplot(components: np.ndarray, explained_variance: np.ndarray) -> np.ndarray:
    """PC loadings weighted by √(explained_variance) — standard biplot scaling."""
    s = np.sqrt(np.maximum(explained_variance[:2], 1e-12))
    vec = components[:2].T * s[np.newaxis, :]
    return vec


@dataclass
class SVDResult:
    singular_values: np.ndarray
    U: np.ndarray
    Vt: np.ndarray
    reconstruction_error: float


def svd_analysis(returns: pd.DataFrame, rank: int | None = None) -> SVDResult:
    X = returns.dropna().values.astype(float)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = rank or min(5, len(s))
    X_hat = U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
    err = np.linalg.norm(X - X_hat, ord="fro") / np.linalg.norm(X, ord="fro")
    return SVDResult(s, U, Vt, float(err))


def svd_relative_error_curve(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Singular values and relative Frobenius error for rank-r truncations (r = 1..len(s)-1)."""
    X = returns.dropna().values.astype(float)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    denom = max(float(np.linalg.norm(X, ord="fro")), 1e-12)
    tail_sq = np.flip(np.cumsum(np.flip(s**2)))
    rel = np.sqrt(tail_sq[1:]) / denom
    return s, rel


def eigen_decomposition(cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov.values)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def explained_variance_ratios_from_singular_values(values_s: np.ndarray) -> np.ndarray:
    """For a mean-centered design matrix X = UΣVᵀ, λᵢ ∝ σᵢ² ⇒ shareᵢ = σᵢ² / Σⱼσⱼ² (finite samples)."""
    s2 = np.asarray(values_s, dtype=float) ** 2
    denom = np.sum(s2)
    if denom <= 0:
        return np.zeros_like(s2)
    return s2 / denom


def stable_rank(singular_values: np.ndarray) -> float:
    """Effective rank via ‖X‖_F² / ‖X‖₂² = (Σσᵢ²) / σ_max² (≥1, equals rank for equal σ's)."""
    s = np.asarray(singular_values, dtype=float)
    if s.size == 0:
        return 0.0
    smax = float(np.max(s))
    if smax <= 0:
        return 0.0
    return float(np.sum(s**2) / (smax**2))


def pca_svd_agreement(
    returns: pd.DataFrame,
    *,
    standardize: bool = True,
    n_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compare PCA explained-variance ratios to σ² shares from SVD of the same matrix sklearn uses.

    Returns (pca_ratios, svd_ratios, max_abs_diff). Should match to ~1e-12 when settings align.
    """
    df = returns.dropna()
    X_raw = df.values.astype(float)
    if standardize:
        X = StandardScaler().fit_transform(X_raw)
    else:
        X = X_raw
    # sklearn centers columns before SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    max_k = max(1, min(Xc.shape) - 1)
    k = max_k if n_components is None else max(1, min(n_components, max_k))
    pca = PCA(n_components=k)
    pca.fit(Xc)
    ratios_pca = np.asarray(pca.explained_variance_ratio_, dtype=float)
    _u, s, _vt = np.linalg.svd(Xc, full_matrices=False)
    ratios_svd_full = explained_variance_ratios_from_singular_values(s)
    ratios_svd = ratios_svd_full[:k]
    d = float(np.max(np.abs(ratios_pca - ratios_svd)))
    return ratios_pca, ratios_svd, d
