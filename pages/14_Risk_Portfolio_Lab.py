"""Portfolio construction and risk diagnostics lab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.session import get_context
from src.metrics import portfolio
from src.ui.lab import num, pct
from src.ui.page_helpers import render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Risk & Portfolio Lab | Saito", layout="wide")
render_metric_page_shell(
    title="Risk & Portfolio Lab",
    explain_key="optimization_kkt",
    showcase_key="optimization",
)

if not require_data(min_assets=3, min_obs=120):
    st.stop()

ctx = get_context()
rets = ctx.returns.dropna(how="any")
mu, cov = portfolio.annualized_mu_cov(rets)
assets = list(rets.columns)

tab1, tab2, tab3 = st.tabs(["Efficient Frontier", "Risk Parity / HRP", "Factor & Stress"])

with tab1:
    st.subheader("Constrained Efficient Frontier")
    n_points = st.slider("Frontier points", 10, 40, 20)
    pts = portfolio.efficient_frontier_constrained(mu, cov, n_points=n_points)
    df = pd.DataFrame(
        {
            "target_return": [p.target_return for p in pts],
            "volatility": [p.volatility for p in pts],
            "sharpe": [(p.target_return / p.volatility) if p.volatility > 1e-10 else np.nan for p in pts],
        }
    )
    fig = px.line(df, x="volatility", y="target_return", title="Long-only constrained frontier")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
    best = df.loc[df["sharpe"].idxmax()]
    k1, k2, k3 = st.columns(3)
    k1.metric("Best Sharpe", num(float(best["sharpe"])))
    k2.metric("Return @ best", pct(float(best["target_return"])))
    k3.metric("Vol @ best", pct(float(best["volatility"])))

with tab2:
    st.subheader("Allocation Diagnostics")
    rp = portfolio.risk_parity_weights(cov)
    hrp = portfolio.hrp_weights(rets).values
    w_df = pd.DataFrame({"asset": assets, "risk_parity": rp, "hrp": hrp})
    st.dataframe(w_df, use_container_width=True)
    w_melt = w_df.melt(id_vars=["asset"], var_name="method", value_name="weight")
    figw = px.bar(w_melt, x="asset", y="weight", color="method")
    apply_theme(figw, "Allocation comparison")
    st.plotly_chart(figw, use_container_width=True)

with tab3:
    st.subheader("Factor Exposure & Rolling Stress")
    target = st.selectbox("Target asset", assets, index=0)
    factor_opts = [a for a in assets if a != target]
    default_f = factor_opts[: min(3, len(factor_opts))]
    factors = st.multiselect("Factor proxies", factor_opts, default=default_f)
    if factors:
        exp, r2 = portfolio.factor_exposure(rets[target], rets[factors])
        e_df = exp.rename_axis("factor").reset_index()
        st.metric("Factor model R²", num(r2))
        fig_e = px.bar(e_df, x="factor", y="exposure", title=f"OLS exposures for {target}")
        apply_theme(fig_e)
        st.plotly_chart(fig_e, use_container_width=True)
    stress = portfolio.rolling_stress_var(rets[target], window=63, alpha=0.95)
    fig_s = px.line(stress, title=f"Rolling VaR95 stress ({target})")
    apply_theme(fig_s)
    st.plotly_chart(fig_s, use_container_width=True)

