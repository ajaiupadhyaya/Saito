"""Showcase / lab UX helpers for Streamlit metric pages.

Goal: make pages feel like applied case studies (one-click scenarios, insight KPIs, sensitivity),
not just a worksheet of formulas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st

from src.data.session import DataContext, compute_returns, get_context, set_context
from src.data.synthetic import generate_preset


@dataclass(frozen=True)
class Scenario:
    id: str
    label: str
    description: str
    generator: Callable[[], pd.DataFrame]  # returns prices-like DataFrame


def _gen(name: str, *, n_obs: int) -> Callable[[], pd.DataFrame]:
    return lambda: generate_preset(name, n_obs=n_obs)


SCENARIOS_BY_TOPIC: dict[str, list[Scenario]] = {
    "descriptive": [
        Scenario(
            "desc_calm",
            "Calm regime (correlated)",
            "Clean multi-asset panel; moments behave smoothly.",
            _gen("correlated_normal", n_obs=756),
        ),
        Scenario(
            "desc_shock",
            "Shock week (crash correlation)",
            "Injects a correlated selloff window; skew/kurt spikes pop.",
            _gen("crash_correlation", n_obs=756),
        ),
    ],
    "covariance": [
        Scenario(
            "cov_calm",
            "Calm correlations",
            "Moderate correlations; useful for PCA baseline.",
            _gen("correlated_normal", n_obs=756),
        ),
        Scenario(
            "cov_crisis",
            "Crisis correlations",
            "Correlation spike window (good for shrinkage + eigen spectrum).",
            _gen("crash_correlation", n_obs=756),
        ),
    ],
    "pca": [
        Scenario(
            "pca_factors",
            "Factor discovery (moderate corr)",
            "PCA biplot shows latent directions; components are interpretable.",
            _gen("correlated_normal", n_obs=756),
        ),
        Scenario(
            "pca_noise",
            "Large-N correlation cleaning (RMT-style)",
            "50 assets with a shock — good for PCA/SVD/eigen demos.",
            _gen("crash_correlation", n_obs=900),
        ),
    ],
    "optimization": [
        Scenario(
            "opt_regression",
            "Regression demo panel",
            "Multi-asset panel; use one return to predict another.",
            _gen("correlated_normal", n_obs=900),
        ),
    ],
    "distributions": [
        Scenario(
            "prob_gauss",
            "Near-Gaussian returns",
            "Good CLT/QQ baseline.",
            _gen("iid_normal", n_obs=900),
        ),
        Scenario(
            "prob_fat",
            "Fat tails (Student-t)",
            "QQ clearly deviates from Normal; tail fit improves.",
            _gen("fat_t_student", n_obs=1200),
        ),
    ],
    "stochastic": [
        Scenario(
            "stoch_gbm",
            "GBM single-asset path",
            "Great for GBM/Itô intuition (simulate + compare).",
            _gen("gbm_prices", n_obs=756),
        ),
        Scenario(
            "stoch_ou",
            "OU spread (pairs)",
            "Mean reversion + half-life is obvious; fitting works well.",
            _gen("ou_spread", n_obs=900),
        ),
    ],
    "evt": [
        Scenario(
            "evt_fat",
            "Tail risk showcase",
            "Fat-t returns — POT threshold sweep is informative.",
            _gen("fat_t_student", n_obs=1500),
        ),
        Scenario(
            "evt_crash",
            "Crisis losses",
            "Correlation shock panel — tail measures jump.",
            _gen("crash_correlation", n_obs=1200),
        ),
    ],
    "copulas": [
        Scenario(
            "cop_dep",
            "Dependence showcase (crisis panel)",
            "Use two series during a shock window; tail dependence visible.",
            _gen("crash_correlation", n_obs=900),
        ),
    ],
    "rmt": [
        Scenario(
            "rmt_clean",
            "50×50 correlation panel (RMT)",
            "Same structure as corr_noise_50x50.csv — MP bulk vs signal eigenvalues.",
            _gen("crash_correlation", n_obs=500),
        ),
    ],
}


def load_scenario_into_session(scen: Scenario) -> DataContext:
    """Materialize a scenario and make it the active shared dataset."""
    prices = scen.generator()
    ctx = DataContext(
        prices=prices,
        returns=compute_returns(prices, get_context().return_type),
        source=f"showcase:{scen.id}",
        tickers=list(prices.columns),
        return_type=get_context().return_type,
        meta={"scenario": scen.id},
    )
    set_context(ctx)
    return ctx


def load_default_topic_scenario(topic_key: str) -> tuple[bool, str]:
    """Load first showcase scenario for a topic. Returns success + label/message."""
    scenarios = SCENARIOS_BY_TOPIC.get(topic_key, [])
    if not scenarios:
        return False, "No built-in scenario available for this topic."
    scen = scenarios[0]
    ctx = load_scenario_into_session(scen)
    return True, f"{scen.label} ({ctx.n_assets} series, {ctx.n_obs} rows)"


def render_showcase(topic_key: str) -> None:
    """Top-of-page UX: scenario selector + one-click load into session."""
    scenarios = SCENARIOS_BY_TOPIC.get(topic_key, [])
    if not scenarios:
        return

    with st.expander("Showcase (one click)", expanded=not get_context().ready):
        st.markdown(
            "Use a prebuilt scenario to make this page immediately demonstrative. "
            "It overwrites the current session dataset (you can always reload on **01_Data_Hub**)."
        )
        labels = [s.label for s in scenarios]
        choice = st.selectbox("Scenario", labels, index=0)
        scen = scenarios[labels.index(choice)]
        st.caption(scen.description)
        if st.button("Load scenario into session", type="primary"):
            ctx = load_scenario_into_session(scen)
            st.success(f"Loaded showcase dataset: {scen.label} ({ctx.n_assets} series, {ctx.n_obs} rows).")


def insight_row(items: list[tuple[str, str]]) -> None:
    """Render a small KPI row: list of (label, value_str)."""
    cols = st.columns(len(items))
    for c, (k, v) in zip(cols, items):
        c.metric(k, v)


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def num(x: float) -> str:
    if np.isnan(x):
        return "—"
    if abs(x) >= 1e6:
        return f"{x:,.0f}"
    if abs(x) >= 1:
        return f"{x:,.3f}"
    return f"{x:,.4f}"

