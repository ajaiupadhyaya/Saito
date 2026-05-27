"""Econometrics and regime diagnostics lab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.session import get_context
from src.metrics import econometrics
from src.ui.lab import num, pct
from src.ui.page_helpers import render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Econometrics & Regime Lab | Saito", layout="wide")
render_metric_page_shell(
    title="Econometrics & Regime Lab",
    explain_key="distributions",
    showcase_key="distributions",
)

if not require_data(min_assets=2, min_obs=180):
    st.stop()

ctx = get_context()
asset = st.selectbox("Asset", list(ctx.returns.columns), index=0)
r = ctx.returns[asset].dropna()

tab1, tab2 = st.tabs(["AR(1) and Residual Diagnostics", "Volatility Regimes"])

with tab1:
    st.subheader("AR(1) Fit and Serial Correlation Check")
    ar = econometrics.ar1_fit(r)
    pval = econometrics.ljung_box_pvalue(r, lags=10)
    k1, k2, k3 = st.columns(3)
    k1.metric("AR(1) phi", num(ar["phi"]))
    k2.metric("Residual sigma", pct(ar["sigma_resid"]))
    k3.metric("Ljung-Box p-value (lag 10)", num(pval))
    st.caption("Low p-value suggests residual autocorrelation and potential model misspecification.")

with tab2:
    st.subheader("Rolling Volatility Regime Detector")
    win = st.slider("Vol window", 10, 63, 21)
    vol = econometrics.rolling_realized_vol(r, window=win)
    flags = econometrics.volatility_regime_flags(vol, high_quantile=0.8)
    df = pd.concat([vol, flags], axis=1).dropna()

    high_ratio = float(df["high_vol_regime"].mean())
    st.metric("High-vol regime share", pct(high_ratio))

    fig = px.line(df, y="realized_vol", title=f"Rolling realized volatility ({asset})")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.area(df, y="high_vol_regime", title="High-vol regime indicator")
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

