"""Descriptive statistics on return panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def _validate_panel(returns: pd.DataFrame, *, min_rows: int = 2, min_cols: int = 1) -> pd.DataFrame:
    if returns is None or returns.empty:
        raise ValueError("returns must be a non-empty DataFrame")
    r = returns.dropna(how="any")
    if r.shape[0] < min_rows:
        raise ValueError(f"need at least {min_rows} complete rows after dropna")
    if r.shape[1] < min_cols:
        raise ValueError(f"need at least {min_cols} asset columns")
    return r.astype(float)


def summary_table(returns: pd.DataFrame, *, ddof_var: int = 1) -> pd.DataFrame:
    """Moments pooled over time per column.

    Uses sample moments by default (ddof=1) to match standard finance reporting.
    """
    if ddof_var not in (0, 1):
        raise ValueError("ddof_var must be 0 (population) or 1 (sample)")
    r = _validate_panel(returns, min_rows=max(2, ddof_var + 1))
    rows = {
        "mean": r.mean(),
        "std": r.std(ddof=ddof_var),
        "variance": r.var(ddof=ddof_var),
        "skewness": r.skew(),
        "excess_kurtosis": r.kurtosis(),
    }
    return pd.DataFrame(rows).T


def rolling_moments(returns: pd.Series, window: int) -> pd.DataFrame:
    if window < 2:
        raise ValueError("window must be >= 2")
    if returns is None or returns.dropna().empty:
        raise ValueError("returns series must be non-empty")
    return pd.DataFrame(
        {
            "mean": returns.rolling(window).mean(),
            "std": returns.rolling(window).std(ddof=1),
            "skew": returns.rolling(window).skew(),
            "kurtosis": returns.rolling(window).kurt(),
        }
    ).dropna()


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    r = _validate_panel(returns, min_rows=2, min_cols=1)
    return r.cov(ddof=1)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    r = _validate_panel(returns, min_rows=2, min_cols=1)
    std = r.std(ddof=1)
    bad = std[std <= 0].index.tolist()
    if bad:
        raise ValueError(f"cannot compute correlation with zero-variance columns: {bad}")
    return r.corr()


def ledoit_wolf_covariance(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shrinkage covariance (Ledoit–Wolf) and implied correlation matrix."""
    df = _validate_panel(returns, min_rows=3, min_cols=2)
    X = df.values.astype(float)
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    cov_df = pd.DataFrame(cov, index=df.columns, columns=df.columns)
    corr_df = pd.DataFrame(corr, index=df.columns, columns=df.columns)
    return cov_df, corr_df


def blended_correlation(returns: pd.DataFrame, shrink: float) -> pd.DataFrame:
    """Convex blend corr → I for simple shrinkage (0 = sample, 1 = identity diag)."""
    shrink = float(np.clip(shrink, 0.0, 1.0))
    c = correlation_matrix(returns)
    ident = pd.DataFrame(np.eye(len(c)), index=c.index, columns=c.columns)
    return (1 - shrink) * c + shrink * ident


def eigen_spectrum(corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if corr is None or corr.empty:
        raise ValueError("corr must be non-empty")
    vals, vecs = np.linalg.eigh(corr.values)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def regression_design_constant_plus_features(
    returns: pd.DataFrame,
    *,
    target: str,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """X = [1, feats], y as arrays after dropna."""
    cols = [target] + list(features)
    if len(features) == 0:
        raise ValueError("features list must be non-empty")
    missing = [c for c in cols if c not in returns.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    df = returns[cols].dropna().astype(float)
    if df.empty:
        raise ValueError("no complete rows after dropna for requested target/features")
    y = df[target].values
    feats = df[features].values
    X = np.column_stack([np.ones(len(df)), feats])
    return X, y
