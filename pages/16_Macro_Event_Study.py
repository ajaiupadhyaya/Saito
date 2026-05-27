"""Macro-event study workflow with CAR diagnostics."""

from __future__ import annotations

from datetime import date, timedelta

import plotly.express as px
import streamlit as st

from src.data.providers import FredProvider, build_price_hub
from src.metrics import event_study
from src.ui.lab import num
from src.ui.page_helpers import render_metric_page_shell
from src.viz.plots import apply_theme

st.set_page_config(page_title="Macro Event Study | Saito", layout="wide")
render_metric_page_shell(title="Macro Event Study", explain_key="evt", showcase_key=None)

col1, col2, col3 = st.columns(3)
with col1:
    asset = st.text_input("Asset", value="SPY").strip().upper()
with col2:
    benchmark = st.text_input("Benchmark", value="QQQ").strip().upper()
with col3:
    macro_sid = st.selectbox("Macro series", ["DGS10", "DGS2", "CPIAUCSL", "UNRATE", "FEDFUNDS"], index=0)

start = st.date_input("Start", value=date.today() - timedelta(days=365 * 5))
end = st.date_input("End", value=date.today())
zthr = st.slider("Shock z-threshold", 1.0, 4.0, 2.0, 0.1)

fred = FredProvider()
hub = build_price_hub(policy="auto")

try:
    prices, _meta = hub.get_prices([asset, benchmark], start=str(start), end=str(end))
    rets = prices.pct_change().dropna()
    macro = fred.get_series(macro_sid, start=str(start), end=str(end)).rename(macro_sid)
except Exception as exc:
    st.error(f"Load failed: {exc}")
    st.stop()

shocks = event_study.macro_shock_dates(macro, z_threshold=float(zthr))
ar = event_study.abnormal_returns(rets[asset], rets[benchmark])
panel = event_study.event_window_car(ar, shocks, pre=3, post=5)

k1, k2, k3 = st.columns(3)
k1.metric("Detected macro shock events", f"{len(shocks)}")
k2.metric("AR sample size", f"{len(ar)}")
k3.metric("Mean AR", num(float(ar.mean())))

fig_m = px.line(macro, title=f"{macro_sid} macro series")
apply_theme(fig_m)
st.plotly_chart(fig_m, use_container_width=True)

if panel.empty:
    st.warning("No event windows found in selected range.")
else:
    avg = panel.groupby("t")["car"].mean().reset_index()
    fig_car = px.line(avg, x="t", y="car", title="Average CAR around macro shock events")
    apply_theme(fig_car)
    st.plotly_chart(fig_car, use_container_width=True)
    st.dataframe(panel.head(200), use_container_width=True)

