"""Advanced derivatives and volatility analytics page."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.hub_service import ProviderError
from src.data.providers import YFinanceProvider
from src.metrics import derivatives
from src.ui.lab import num, pct
from src.ui.page_helpers import render_metric_page_shell
from src.viz.plots import apply_theme

st.set_page_config(page_title="Derivatives & Vol Lab | Saito", layout="wide")
render_metric_page_shell(
    title="Derivatives & Volatility Lab",
    explain_key="optimization_kkt",
    showcase_key=None,
)

provider = YFinanceProvider()
ticker = st.text_input("Ticker", value="AAPL").strip().upper()
if not ticker:
    st.warning("Enter a ticker")
    st.stop()

try:
    expiries = provider.get_option_expiries(ticker)
except ProviderError as exc:
    st.error(f"Could not load expiries: {exc}")
    st.stop()

pick_n = st.slider("Number of expiries to analyze", 1, min(8, len(expiries)), min(4, len(expiries)))
selected = expiries[:pick_n]

chains: dict[str, pd.DataFrame] = {}
all_rows: list[pd.DataFrame] = []
for ex in selected:
    try:
        calls, puts = provider.get_option_chain(ticker, ex)
        c = pd.concat([calls, puts], ignore_index=True)
        chains[ex] = c
        all_rows.append(c)
    except ProviderError:
        continue

if not all_rows:
    st.warning("No option chains loaded.")
    st.stop()

full = pd.concat(all_rows, ignore_index=True)
q = derivatives.option_chain_quality(full)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Contracts", f"{int(q['n_contracts'])}")
k2.metric("Quality score", pct(q["quality_score"]))
k3.metric("Missing IV", pct(q["missing_iv_ratio"]))
k4.metric("Zero OI", pct(q["zero_oi_ratio"]))

term = derivatives.iv_term_structure(chains)
if not term.empty:
    fig_t = px.line(term, x="expiry", y="iv_median", title="IV term structure (median)")
    apply_theme(fig_t)
    st.plotly_chart(fig_t, use_container_width=True)

pick_exp = st.selectbox("Expiry for skew and scenario", list(chains.keys()), index=0)
chain = chains[pick_exp]
skew = derivatives.iv_skew(chain)
st.metric("IV skew (Q75 - Q25)", num(skew))

if {"strike", "impliedVolatility", "type"}.issubset(chain.columns):
    fig_s = px.scatter(chain, x="strike", y="impliedVolatility", color="type", title=f"Smile snapshot ({pick_exp})")
    apply_theme(fig_s)
    st.plotly_chart(fig_s, use_container_width=True)

if "lastPrice" in chain.columns:
    spot_guess = float(chain.get("lastPrice", pd.Series([100.0])).dropna().median())
else:
    spot_guess = 100.0
strike_guess = float(chain.get("strike", pd.Series([spot_guess])).dropna().median())
ttm = max(1 / 365, (pd.Timestamp(pick_exp) - pd.Timestamp(datetime.utcnow())).days / 365)
iv_guess = float(pd.to_numeric(chain.get("impliedVolatility", pd.Series([0.3])), errors="coerce").dropna().median())
if not np.isfinite(iv_guess) or iv_guess <= 0:
    iv_guess = 0.3

spot_shocks = [-0.1, -0.05, 0.0, 0.05, 0.1]
vol_shocks = [-0.1, -0.05, 0.0, 0.05, 0.1]
grid = derivatives.scenario_grid(
    spot=spot_guess,
    strike=strike_guess,
    base_ttm_years=ttm,
    rate=0.04,
    base_vol=iv_guess,
    spot_shocks=spot_shocks,
    vol_shocks=vol_shocks,
)
fig_h = px.density_heatmap(
    grid,
    x="spot_shock",
    y="vol_shock",
    z="pnl_vs_base",
    title="Scenario PnL surface vs base call",
)
apply_theme(fig_h)
st.plotly_chart(fig_h, use_container_width=True)

