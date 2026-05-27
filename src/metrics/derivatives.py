"""Derivatives and volatility analytics helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def option_chain_quality(chain: pd.DataFrame) -> dict[str, float]:
    if chain.empty:
        raise ValueError("option chain is empty")
    n = len(chain)
    missing_iv = float(chain["impliedVolatility"].isna().mean()) if "impliedVolatility" in chain.columns else 1.0
    zero_oi = float((chain.get("openInterest", 0) == 0).mean()) if "openInterest" in chain.columns else 1.0
    wide_spread = 0.0
    if {"bid", "ask"}.issubset(chain.columns):
        mid = (chain["bid"] + chain["ask"]) / 2.0
        spread = chain["ask"] - chain["bid"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = spread / mid.replace(0, np.nan)
        wide_spread = float((rel > 0.15).mean())
    score = float(np.clip(1.0 - (0.4 * missing_iv + 0.3 * zero_oi + 0.3 * wide_spread), 0.0, 1.0))
    return {
        "n_contracts": float(n),
        "missing_iv_ratio": missing_iv,
        "zero_oi_ratio": zero_oi,
        "wide_spread_ratio": wide_spread,
        "quality_score": score,
    }


def iv_skew(chain: pd.DataFrame) -> float:
    c = chain.dropna(subset=["strike", "impliedVolatility"]).copy()
    if c.shape[0] < 8:
        return float("nan")
    low = float(c["strike"].quantile(0.25))
    high = float(c["strike"].quantile(0.75))
    iv_low = float(c.loc[c["strike"] <= low, "impliedVolatility"].median())
    iv_high = float(c.loc[c["strike"] >= high, "impliedVolatility"].median())
    return iv_high - iv_low


def iv_term_structure(chains: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for expiry, chain in chains.items():
        if chain.empty or "impliedVolatility" not in chain.columns:
            continue
        iv = pd.to_numeric(chain["impliedVolatility"], errors="coerce").dropna()
        if iv.empty:
            continue
        rows.append({"expiry": expiry, "iv_median": float(iv.median()), "iv_mean": float(iv.mean())})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    return out.sort_values("expiry")


@dataclass
class GreeksResult:
    delta: float
    gamma: float
    vega: float
    theta: float
    price: float


def black_scholes_call_greeks(spot: float, strike: float, ttm_years: float, rate: float, vol: float) -> GreeksResult:
    if spot <= 0 or strike <= 0 or ttm_years <= 0 or vol <= 0:
        raise ValueError("invalid BS inputs")
    sqrt_t = np.sqrt(ttm_years)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * ttm_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    nd1 = norm.pdf(d1)
    delta = norm.cdf(d1)
    gamma = nd1 / (spot * vol * sqrt_t)
    vega = spot * nd1 * sqrt_t
    theta = -(spot * nd1 * vol) / (2 * sqrt_t) - rate * strike * np.exp(-rate * ttm_years) * norm.cdf(d2)
    price = spot * norm.cdf(d1) - strike * np.exp(-rate * ttm_years) * norm.cdf(d2)
    return GreeksResult(float(delta), float(gamma), float(vega), float(theta), float(price))


def scenario_grid(
    spot: float,
    strike: float,
    base_ttm_years: float,
    rate: float,
    base_vol: float,
    spot_shocks: list[float],
    vol_shocks: list[float],
) -> pd.DataFrame:
    rows = []
    base = black_scholes_call_greeks(spot, strike, base_ttm_years, rate, base_vol).price
    for ds in spot_shocks:
        for dv in vol_shocks:
            s = spot * (1 + ds)
            v = max(0.01, base_vol + dv)
            g = black_scholes_call_greeks(s, strike, base_ttm_years, rate, v)
            rows.append({"spot_shock": ds, "vol_shock": dv, "price": g.price, "pnl_vs_base": g.price - base})
    return pd.DataFrame(rows)

