"""Data Hub — load tickers, CSV, or synthetic presets."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.csv_loader import load_csv_bytes
from src.data.providers import POLICIES, build_price_hub, provider_capability_matrix
from src.data.session import DataContext, DatasetProvenance, compute_returns, ensure_session, get_context, set_context
from src.data.synthetic import PRESETS, generate_preset, load_bundled
from src.viz.plots import apply_theme

st.set_page_config(page_title="Data Hub | Saito", layout="wide")
ensure_session()

st.title("Data Hub")
st.markdown(
    "Configure the active dataset used across all metric pages. "
    "You can also skip this page: model pages auto-load built-in demo datasets."
)

source = st.radio(
    "Data source",
    ["API router", "CSV upload", "Synthetic preset", "Bundled dataset"],
    horizontal=True,
)

existing = get_context()
rt_options = ["log", "simple"]
rt_index = rt_options.index(existing.return_type) if existing.return_type in rt_options else 0
return_type = st.selectbox("Return type", rt_options, index=rt_index)

if existing.ready and existing.prices is not None and existing.return_type != return_type:
    existing.return_type = return_type
    existing.returns = compute_returns(existing.prices, return_type)
    set_context(existing)


@st.cache_data(ttl=3600)
def cached_api_prices(policy: str, tickers_key: tuple[str, ...], start_s: str, end_s: str | None):
    """Disk-backed cache keyed by policy + ticker set + dates."""
    hub = build_price_hub(policy=policy)
    return hub.get_prices(list(tickers_key), start=start_s, end=end_s)


ctx = DataContext(return_type=return_type)

try:
    if source == "API router":
        policy = st.selectbox(
            "Provider policy",
            list(POLICIES),
            index=0,
            help=(
                "auto: premium if available then fallback, premium-first: force premium attempts first, "
                "free-only: yfinance only"
            ),
        )
        tickers_raw = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA")
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Start", value=date.today() - timedelta(days=365 * 3))
        with col2:
            end = st.date_input("End", value=date.today())
        if st.button("Load from API router", type="primary"):
            with st.spinner("Downloading…"):
                prices, meta = cached_api_prices(policy, tuple(sorted(tickers)), str(start), str(end))
            prov = DatasetProvenance(
                provider=meta.get("provider", ""),
                policy=policy,
                requested_tickers=tickers,
                requested_start=str(start),
                requested_end=str(end),
                loaded_at_utc=pd.Timestamp.utcnow().isoformat(),
                attempts=meta.get("attempts", []),
                transforms=[f"returns:{return_type}"],
            )
            ctx.prices = prices
            ctx.returns = compute_returns(prices, return_type)
            ctx.source = f"api:{meta['provider']}"
            ctx.tickers = list(prices.columns)
            ctx.meta = {"start": str(start), "end": str(end), **meta, "provenance": prov.to_meta()}
            set_context(ctx)
            st.success(
                f"Loaded {ctx.n_assets} assets, {ctx.n_obs} observations "
                f"from {meta['provider']} ({meta.get('latency_ms', 0)} ms)."
            )
        with st.expander("Provider diagnostics"):
            st.caption("Capabilities and env-based availability for configured data providers.")
            st.dataframe(provider_capability_matrix(), use_container_width=True)

    elif source == "CSV upload":
        uploaded = st.file_uploader("Upload CSV (rows=dates, cols=prices)", type=["csv"])
        if uploaded and st.button("Load CSV", type="primary"):
            prices = load_csv_bytes(uploaded.getvalue())
            ctx.prices = prices
            ctx.returns = compute_returns(prices, return_type)
            ctx.source = "csv"
            ctx.tickers = list(prices.columns)
            set_context(ctx)
            st.success(f"Loaded {ctx.n_assets} assets, {ctx.n_obs} observations.")

    elif source == "Synthetic preset":
        preset = st.selectbox("Preset", list(PRESETS.keys()), format_func=lambda k: PRESETS[k])
        n_obs = st.slider("Observations", 252, 2000, 756)
        if st.button("Generate synthetic data", type="primary"):
            prices = generate_preset(preset, n_obs=n_obs)
            ctx.prices = prices
            ctx.returns = compute_returns(prices, return_type)
            ctx.source = f"synthetic:{preset}"
            ctx.tickers = list(prices.columns)
            set_context(ctx)
            st.success(f"Generated {ctx.n_assets} assets, {ctx.n_obs} observations.")

    else:
        bundled = st.selectbox(
            "Bundled file",
            ["corr_noise_50x50.csv", "pair_spread_ou.csv", "tail_events.csv"],
        )
        if st.button("Load bundled dataset", type="primary"):
            prices = load_bundled(bundled)
            ctx.prices = prices
            ctx.returns = compute_returns(prices, return_type)
            ctx.source = f"bundled:{bundled}"
            ctx.tickers = list(prices.columns)
            set_context(ctx)
            st.success(f"Loaded {ctx.n_assets} assets, {ctx.n_obs} observations.")

except Exception as e:
    st.error(f"Load failed: {e}")

ctx = get_context()
if ctx and ctx.ready:
    st.subheader("Preview")
    if isinstance(ctx.meta, dict) and ctx.meta.get("provenance"):
        with st.expander("Dataset provenance"):
            st.json(ctx.meta["provenance"])
    tab1, tab2 = st.tabs(["Prices", "Returns"])
    with tab1:
        st.dataframe(ctx.prices.tail(10), use_container_width=True)
    with tab2:
        st.dataframe(ctx.returns.tail(10), use_container_width=True)
        fig = go.Figure()
        for col in ctx.returns.columns[:10]:
            fig.add_scatter(y=ctx.returns[col].cumsum(), mode="lines", name=str(col))
        apply_theme(fig, "Cumulative returns preview")
        st.plotly_chart(fig, use_container_width=True)
