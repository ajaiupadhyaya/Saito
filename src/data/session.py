"""Streamlit session state contract for loaded market data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

SESSION_KEY = "saito_data"


@dataclass
class DatasetProvenance:
    provider: str = ""
    policy: str = "auto"
    requested_tickers: list[str] = field(default_factory=list)
    requested_start: str = ""
    requested_end: str = ""
    loaded_at_utc: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)
    transforms: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataContext:
    """Active dataset shared across metric pages."""

    prices: pd.DataFrame | None = None
    returns: pd.DataFrame | None = None
    source: str = ""
    tickers: list[str] = field(default_factory=list)
    return_type: str = "log"
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.returns is not None and not self.returns.empty

    @property
    def n_assets(self) -> int:
        if self.returns is None:
            return 0
        return self.returns.shape[1]

    @property
    def n_obs(self) -> int:
        if self.returns is None:
            return 0
        return len(self.returns)


def ensure_session() -> None:
    import streamlit as st

    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = DataContext()


def get_context() -> DataContext:
    import streamlit as st

    ensure_session()
    return st.session_state[SESSION_KEY]


def set_context(ctx: DataContext) -> None:
    import streamlit as st

    st.session_state[SESSION_KEY] = ctx


def compute_returns(prices: pd.DataFrame, return_type: str = "log") -> pd.DataFrame:
    prices = prices.sort_index().astype(float)
    if return_type == "log":
        return np_log_returns(prices)
    return prices.pct_change().dropna(how="all")


def np_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    import numpy as np

    log_p = np.log(prices.replace(0, np.nan))
    return log_p.diff().dropna(how="all")


__all__ = [
    "DataContext",
    "DatasetProvenance",
    "ensure_session",
    "get_context",
    "set_context",
    "compute_returns",
]
