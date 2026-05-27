"""API-first research studio: factor-risk, options, macro, strategy, and sentiment."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.hub_service import ProviderError
from src.data.providers import POLICIES, FredProvider, NewsApiProvider, YFinanceProvider, build_price_hub
from src.data.session import DataContext, DatasetProvenance, compute_returns, set_context
from src.metrics.research import rolling_beta, run_signal_backtest, var_cvar
from src.metrics.validation import overfit_gap_score, walk_forward_backtest
from src.ui.insights import strategy_narrative
from src.ui.lab import num, pct
from src.viz.plots import apply_theme

st.set_page_config(page_title="Research Studio | Saito", layout="wide")
st.title("Research Studio (API-First)")
st.caption("Factor-risk, options-vol, macro engine, and strategy studio on a shared data contract.")

macro_provider = FredProvider()
news_provider = NewsApiProvider()


@st.cache_data(ttl=900)
def load_prices(policy: str, tickers: tuple[str, ...], start_s: str, end_s: str):
    return build_price_hub(policy=policy).get_prices(list(tickers), start=start_s, end=end_s)


@st.cache_data(ttl=900)
def load_macro_series(series_id: str, start_s: str, end_s: str):
    s = macro_provider.get_series(series_id, start=start_s, end=end_s)
    return s.rename(series_id)


@st.cache_data(ttl=900)
def load_chain(ticker: str, expiry: str | None):
    return YFinanceProvider().get_option_chain(ticker=ticker, expiry=expiry)


col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    tickers_raw = st.text_input("Universe tickers", "SPY,QQQ,IWM,TLT,GLD")
with col_b:
    start = st.date_input("Start", value=date.today() - timedelta(days=365 * 3))
with col_c:
    end = st.date_input("End", value=date.today())
policy = st.selectbox("Provider policy", list(POLICIES), index=0)

tickers = tuple(sorted({x.strip().upper() for x in tickers_raw.split(",") if x.strip()}))
if not tickers:
    st.warning("Add at least one ticker to run the studio.")
    st.stop()

try:
    prices, meta = load_prices(policy, tickers, str(start), str(end))
except Exception as exc:
    st.error(f"Price load failed: {exc}")
    st.stop()

returns = compute_returns(prices, return_type="log")
st.info(
    f"Source `{meta['provider']}` via policy `{meta.get('policy', policy)}` | "
    f"{meta['n_assets']} assets | {meta['n_obs']} observations "
    f"({meta['start']} → {meta['end']}) | latency {meta.get('latency_ms', 0)} ms"
)

if st.button("Use this dataset across other pages"):
    prov = DatasetProvenance(
        provider=meta.get("provider", ""),
        policy=policy,
        requested_tickers=list(tickers),
        requested_start=str(start),
        requested_end=str(end),
        loaded_at_utc=pd.Timestamp.utcnow().isoformat(),
        attempts=meta.get("attempts", []),
        transforms=["returns:log"],
    )
    ctx = DataContext(
        prices=prices,
        returns=returns,
        source=f"research-studio:{meta['provider']}",
        tickers=list(prices.columns),
        return_type="log",
        meta={"provider": meta["provider"], "loaded_by": "research_studio", "provenance": prov.to_meta()},
    )
    set_context(ctx)
    st.success("Session dataset updated. Other pages now use this panel.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Factor & Risk", "Options & Vol", "Macro Engine", "Strategy Studio", "Alt Data & Sentiment"]
)

with tab1:
    st.subheader("Factor & Risk Lab")
    cols = list(returns.columns)
    if len(cols) < 2:
        st.warning("Need at least 2 assets for beta and relative risk diagnostics.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            asset = st.selectbox("Asset", cols, index=0, key="risk_asset")
        with c2:
            bench = st.selectbox("Benchmark", cols, index=min(1, len(cols) - 1), key="risk_bench")
        with c3:
            beta_window = st.slider("Beta window", 20, 180, 60)

        panel = pd.concat([returns[asset], returns[bench]], axis=1).dropna()
        beta_s = rolling_beta(panel[asset], panel[bench], window=beta_window)
        vc = var_cvar(panel[asset], alpha=0.95)
        ann_ret = float(panel[asset].mean() * 252)
        ann_vol = float(panel[asset].std(ddof=1) * (252**0.5))
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else float("nan")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Ann. return", pct(ann_ret))
        k2.metric("Ann. vol", pct(ann_vol))
        k3.metric("Sharpe", num(sharpe))
        k4.metric("Rolling beta (latest)", num(float(beta_s.iloc[-1])) if not beta_s.empty else "—")
        st.caption(f"Historical VaR(95): {pct(vc['var'])} | CVaR(95): {pct(vc['cvar'])}")

        fig_beta = px.line(beta_s, title=f"Rolling beta: {asset} vs {bench}")
        apply_theme(fig_beta)
        st.plotly_chart(fig_beta, use_container_width=True)

        corr = returns.corr()
        fig_corr = px.imshow(
            corr,
            title="Cross-asset correlation regime map",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        apply_theme(fig_corr)
        st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    st.subheader("Derivatives & Volatility Lab")
    opt_ticker = st.text_input("Option ticker", value="AAPL", key="opt_ticker").upper().strip()
    expiry = st.text_input("Expiry (YYYY-MM-DD, optional)", value="", key="opt_expiry").strip() or None
    if st.button("Load option chain", key="load_chain"):
        try:
            calls, puts = load_chain(opt_ticker, expiry)
            chain = pd.concat([calls, puts], ignore_index=True)
            keep = [
                c
                for c in [
                    "type",
                    "strike",
                    "lastPrice",
                    "bid",
                    "ask",
                    "volume",
                    "openInterest",
                    "impliedVolatility",
                ]
                if c in chain.columns
            ]
            st.dataframe(chain[keep].sort_values(["type", "strike"]).head(300), use_container_width=True)
            if {"strike", "impliedVolatility", "type"}.issubset(chain.columns):
                fig_iv = px.scatter(
                    chain,
                    x="strike",
                    y="impliedVolatility",
                    color="type",
                    title=f"IV smile snapshot ({opt_ticker})",
                    hover_data=[c for c in ["openInterest", "volume", "lastPrice"] if c in chain.columns],
                )
                apply_theme(fig_iv)
                st.plotly_chart(fig_iv, use_container_width=True)
        except ProviderError as exc:
            st.error(f"Option chain unavailable: {exc}")

with tab3:
    st.subheader("Macro Engine")
    st.caption("FRED series via public endpoint. Pick any valid series id.")
    series_opts = ["DGS10", "DGS2", "CPIAUCSL", "UNRATE", "FEDFUNDS"]
    s1, s2 = st.columns(2)
    with s1:
        sid_a = st.selectbox("Series A", series_opts, index=0)
    with s2:
        sid_b = st.selectbox("Series B", series_opts, index=1)

    try:
        a = load_macro_series(sid_a, str(start), str(end))
        b = load_macro_series(sid_b, str(start), str(end))
        macro_df = pd.concat([a, b], axis=1).dropna()
        macro_df["spread"] = macro_df.iloc[:, 0] - macro_df.iloc[:, 1]
        fig_macro = px.line(macro_df[[sid_a, sid_b]], title=f"{sid_a} vs {sid_b}")
        apply_theme(fig_macro)
        st.plotly_chart(fig_macro, use_container_width=True)
        fig_spread = px.line(macro_df["spread"], title=f"Macro spread: {sid_a} - {sid_b}")
        apply_theme(fig_spread)
        st.plotly_chart(fig_spread, use_container_width=True)
    except ProviderError as exc:
        st.error(f"Macro load failed: {exc}")

with tab4:
    st.subheader("Strategy Studio")
    c1, c2, c3 = st.columns(3)
    with c1:
        s_asset = st.selectbox("Backtest asset", list(prices.columns), key="strat_asset")
    with c2:
        lookback = st.slider("Signal lookback", 10, 120, 40)
    with c3:
        fee_bps = st.slider("Fee/slippage (bps)", 0.0, 25.0, 2.5, 0.5)

    p = prices[s_asset].dropna()
    mom = p.pct_change(lookback)
    signal = (mom > 0).astype(float) - (mom <= 0).astype(float)
    bt = run_signal_backtest(p, signal, fee_bps=float(fee_bps))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ann. return", pct(bt.metrics["ann_return"]))
    m2.metric("Ann. vol", pct(bt.metrics["ann_vol"]))
    m3.metric("Sharpe", num(bt.metrics["sharpe"]))
    m4.metric("Max drawdown", pct(bt.metrics["max_drawdown"]))

    fig_bt = px.line(bt.equity_curve, title=f"Momentum signal backtest equity ({s_asset})")
    apply_theme(fig_bt)
    st.plotly_chart(fig_bt, use_container_width=True)

    wf = walk_forward_backtest(bt.returns, signal.reindex(bt.returns.index), train=252, test=63)
    if wf:
        wf_df = pd.DataFrame(
            {
                "split": [x.split_id for x in wf],
                "train_sharpe": [x.train_sharpe for x in wf],
                "test_sharpe": [x.test_sharpe for x in wf],
                "test_return": [x.test_return for x in wf],
            }
        )
        gap = overfit_gap_score(wf)
        st.metric("Overfit gap (train - test Sharpe)", num(gap))
        fig_wf = px.line(
            wf_df.melt(id_vars=["split"], value_vars=["train_sharpe", "test_sharpe"]),
            x="split",
            y="value",
            color="variable",
            title="Walk-forward Sharpe by split",
        )
        apply_theme(fig_wf)
        st.plotly_chart(fig_wf, use_container_width=True)
        st.info(strategy_narrative(bt.metrics["ann_return"], bt.metrics["sharpe"], bt.metrics["max_drawdown"], gap))

with tab5:
    st.subheader("Alt Data & Sentiment Lab")
    query = st.text_input("News query", value="federal reserve OR inflation OR earnings", key="news_query")
    if st.button("Load news sentiment", key="load_news_sent"):
        try:
            news = news_provider.get_headlines(query, page_size=30)
            avg_sent = float(news["sentiment_score"].mean())
            st.metric("Average headline sentiment score", num(avg_sent))
            daily = news.set_index("publishedAt").resample("D")["sentiment_score"].mean().dropna()
            fig_sent = px.bar(daily, title="Daily average sentiment score")
            apply_theme(fig_sent)
            st.plotly_chart(fig_sent, use_container_width=True)
            st.dataframe(news[["publishedAt", "source", "title", "sentiment_score", "url"]], use_container_width=True)
        except ProviderError as exc:
            st.error(f"News sentiment unavailable: {exc}")

