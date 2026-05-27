"""Econometrics and regime diagnostics for advanced research workflows."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg


def ar1_fit(series: pd.Series) -> dict[str, float]:
    s = series.dropna().astype(float)
    if len(s) < 40:
        raise ValueError("need at least 40 observations for AR(1)")
    model = AutoReg(s, lags=1, old_names=False).fit()
    phi = float(model.params.get("y.L1", list(model.params)[1]))
    sigma = float(np.std(model.resid, ddof=1))
    return {"phi": phi, "intercept": float(model.params.iloc[0]), "sigma_resid": sigma}


def ljung_box_pvalue(series: pd.Series, lags: int = 10) -> float:
    s = series.dropna().astype(float)
    if len(s) < max(30, lags + 5):
        raise ValueError("insufficient observations for Ljung-Box test")
    out = acorr_ljungbox(s, lags=[lags], return_df=True)
    return float(out["lb_pvalue"].iloc[0])


def rolling_realized_vol(series: pd.Series, window: int = 21) -> pd.Series:
    s = series.dropna().astype(float)
    if window < 5:
        raise ValueError("window must be >= 5")
    vol = s.rolling(window).std(ddof=1) * np.sqrt(252)
    return vol.dropna().rename("realized_vol")


def volatility_regime_flags(vol: pd.Series, high_quantile: float = 0.8) -> pd.Series:
    v = vol.dropna().astype(float)
    thr = float(v.quantile(high_quantile))
    flag = (v >= thr).astype(int)
    return flag.rename("high_vol_regime")

