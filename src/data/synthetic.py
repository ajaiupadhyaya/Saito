"""Synthetic datasets for demos and bundled fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRESETS = {
    "iid_normal": "IID normal daily returns (5 assets)",
    "correlated_normal": "Correlated multivariate normal returns",
    "fat_t_student": "Fat-tailed Student-t returns",
    "gbm_prices": "GBM simulated price paths",
    "ou_spread": "Ornstein-Uhlenbeck mean-reverting spread",
    "crash_correlation": "High correlation spike (RMT/copula demo)",
}


def generate_preset(name: str, n_obs: int = 756, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_obs)

    if name == "iid_normal":
        r = rng.normal(0.0003, 0.012, size=(n_obs, 5))
        cols = [f"ASSET_{i}" for i in range(5)]
        prices = 100 * np.exp(np.cumsum(r, axis=0))
        return pd.DataFrame(prices, index=dates, columns=cols)

    if name == "correlated_normal":
        n = 8
        rho = 0.4
        cov = np.full((n, n), rho)
        np.fill_diagonal(cov, 1.0)
        cov *= 0.000144
        r = rng.multivariate_normal(np.zeros(n) + 0.0002, cov, size=n_obs)
        cols = [f"SYN_{i}" for i in range(n)]
        prices = 100 * np.exp(np.cumsum(r, axis=0))
        return pd.DataFrame(prices, index=dates, columns=cols)

    if name == "fat_t_student":
        r = rng.standard_t(df=4, size=(n_obs, 4)) * 0.012
        cols = [f"FAT_{i}" for i in range(4)]
        prices = 100 * np.exp(np.cumsum(r, axis=0))
        return pd.DataFrame(prices, index=dates, columns=cols)

    if name == "gbm_prices":
        mu, sigma, s0 = 0.08 / 252, 0.2 / np.sqrt(252), 100.0
        z = rng.normal((mu - 0.5 * sigma**2), sigma, n_obs)
        log_s = np.log(s0) + np.cumsum(z)
        return pd.DataFrame({"GBM": np.exp(log_s)}, index=dates)

    if name == "ou_spread":
        theta, mu, sigma = 2.0, 0.0, 0.02
        dt = 1 / 252
        x = np.zeros(n_obs)
        x[0] = mu + 0.5
        for t in range(1, n_obs):
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
        leg_a = 100 + np.cumsum(rng.normal(0.0002, 0.01, n_obs))
        leg_b = leg_a - x
        return pd.DataFrame({"LEG_A": leg_a, "LEG_B": leg_b, "SPREAD": x}, index=dates)

    if name == "crash_correlation":
        n = 50
        base = rng.normal(0, 0.01, size=(n_obs, n))
        crash_idx = n_obs // 3
        shock = rng.normal(-0.08, 0.02, size=n)
        base[crash_idx : crash_idx + 5] += shock
        cols = [f"RMT_{i:02d}" for i in range(n)]
        prices = 100 * np.exp(np.cumsum(base, axis=0))
        return pd.DataFrame(prices, index=dates, columns=cols)

    raise ValueError(f"Unknown preset: {name}")


def load_bundled(name: str) -> pd.DataFrame:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "datasets"
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()
