import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data.session import get_context
from src.metrics import stochastic
from src.ui.lab import insight_row, num, pct
from src.ui.page_helpers import render_metric_page_shell
from src.viz.plots import apply_theme

st.set_page_config(page_title="Brownian, GBM & OU | Saito", layout="wide")
render_metric_page_shell(
    title="Brownian Motion, GBM, and OU",
    explain_key="stochastic",
    showcase_key="stochastic",
)

st.caption("Tip: load **ou_spread** preset or `pair_spread_ou.csv` for OU demos.")
seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1)

ctx = get_context()
tab_bm, tab_gbm, tab_ou = st.tabs(["Brownian motion", "GBM", "OU mean reversion"])

with tab_bm:
    n_steps = st.slider("Steps", 100, 1000, 500, key="bm_steps")
    n_paths = st.slider("Paths", 3, 20, 8, key="bm_paths")
    paths = stochastic.simulate_brownian(n_steps, n_paths, seed=int(seed))
    stats = stochastic.brownian_increment_stats(paths)
    st.subheader("Insight")
    insight_row(
        [
            ("ΔW mean", num(stats["increment_mean"])),
            ("ΔW var", num(stats["increment_var"])),
            ("QV / T", num(stats["qv_ratio"])),
            ("paths", str(n_paths)),
        ]
    )
    fig = go.Figure()
    for i in range(paths.shape[1]):
        fig.add_scatter(y=paths[:, i], mode="lines", name=f"path {i + 1}")
    apply_theme(fig, "Brownian motion paths")
    st.plotly_chart(fig, use_container_width=True)
    qv_path = np.cumsum(np.diff(paths[:, 0]) ** 2)
    t_axis = np.arange(1, len(qv_path) + 1) / 252
    fig_qv = go.Figure()
    fig_qv.add_scatter(x=t_axis, y=qv_path, mode="lines", name="QV estimate")
    fig_qv.add_scatter(x=t_axis, y=t_axis, mode="lines", name="theory (t)", line=dict(dash="dash"))
    apply_theme(fig_qv, "Quadratic variation — path 1")
    st.plotly_chart(fig_qv, use_container_width=True)

with tab_gbm:
    use_data = ctx.ready and st.checkbox("Calibrate from loaded prices", value=ctx.ready, key="gbm_cal")
    if use_data:
        asset = st.selectbox("Asset", list(ctx.returns.columns), key="gbm_asset")
        mu, sigma = stochastic.calibrate_gbm(ctx.prices[asset].dropna())
        s0 = float(ctx.prices[asset].dropna().iloc[-1])
        st.caption(f"Calibrated μ={mu:.2%}, σ={sigma:.2%} (annualized)")
    else:
        mu = st.number_input("Drift μ (annual)", 0.08, key="gbm_mu")
        sigma = st.number_input("Vol σ (annual)", 0.2, key="gbm_sig")
        s0 = st.number_input("S0", 100.0, key="gbm_s0")
    n_steps = st.slider("Sim steps", 50, 500, 252, key="gbm_steps")
    paths = stochastic.simulate_gbm(s0, mu, sigma, n_steps, n_paths=15, seed=int(seed))
    dt = 1 / 252
    t_axis = np.arange(1, n_steps + 1) * dt
    p10_line = np.array(
        [stochastic.gbm_terminal_quantiles(s0, mu, sigma, t)["p10"] for t in t_axis]
    )
    p90_line = np.array(
        [stochastic.gbm_terminal_quantiles(s0, mu, sigma, t)["p90"] for t in t_axis]
    )
    t_horizon = n_steps / 252
    bands = stochastic.gbm_terminal_quantiles(s0, mu, sigma, t_horizon)
    st.subheader("Insight")
    insight_row(
        [
            ("μ (ann.)", pct(mu)),
            ("σ (ann.)", pct(sigma)),
            ("analytic p10", num(bands["p10"])),
            ("analytic p90", num(bands["p90"])),
        ]
    )
    fig = go.Figure()
    fig.add_scatter(x=t_axis, y=p10_line, mode="lines", name="analytic p10 band", line=dict(dash="dot"))
    fig.add_scatter(x=t_axis, y=p90_line, mode="lines", name="analytic p90 band", line=dict(dash="dot"))
    for i in range(paths.shape[1]):
        fig.add_scatter(x=t_axis, y=paths[:, i], mode="lines", opacity=0.7, showlegend=False)
    apply_theme(fig, "GBM simulated prices with analytic confidence bands")
    st.plotly_chart(fig, use_container_width=True)
    p10 = np.percentile(paths[-1, :], 10)
    p90 = np.percentile(paths[-1, :], 90)
    st.caption(
        f"Terminal simulated p10/p90: {p10:.2f} / {p90:.2f} "
        f"(analytic: {bands['p10']:.2f} / {bands['p90']:.2f})"
    )

with tab_ou:
    series = None
    true_theta = None
    if ctx.ready:
        cols = list(ctx.prices.columns)
        spread_cols = [c for c in cols if c == "SPREAD"] + [c for c in cols if c != "SPREAD"]
        series_name = st.selectbox("Series to fit", spread_cols, key="ou_series")
        if series_name in ctx.prices.columns:
            series = ctx.prices[series_name]
        else:
            series = ctx.returns[series_name].cumsum()
        if "SPREAD" in ctx.prices.columns and series_name == "SPREAD":
            true_theta = 2.0
            st.caption("Synthetic ou_spread preset: true θ ≈ 2.0")
    if series is not None and len(series.dropna()) >= 30:
        params = stochastic.fit_ou(series.dropna())
        st.subheader("Insight")
        rows = [
            ("θ̂", num(params["theta"])),
            ("μ̂", num(params["mu"])),
            ("σ̂", num(params["sigma"])),
            ("half-life", f"{num(params['half_life'])} d"),
        ]
        if true_theta is not None:
            rows.append(("θ true", num(true_theta)))
        insight_row(rows)
        diag = stochastic.ou_sim_vs_analytic(params["theta"], params["mu"], params["sigma"])
        st.caption(f"Theory vs sim variance ratio: {diag['var_sim'] / max(diag['var_true'], 1e-12):.3f}")
        ou_kw = {k: params[k] for k in ("theta", "mu", "sigma")}
        sim = stochastic.simulate_ou(float(series.iloc[-1]), n_steps=252, seed=int(seed), **ou_kw)
        fig = go.Figure()
        fig.add_scatter(y=series.dropna().values[-252:], mode="lines", name="history")
        fig.add_scatter(y=sim, mode="lines", name="simulated OU")
        apply_theme(fig, "OU spread")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load **ou_spread** preset or `pair_spread_ou.csv`. Using default parameters.")
        sim = stochastic.simulate_ou(0.5, 2.0, 0.0, 0.02, 252, seed=int(seed))
        fig = go.Figure(go.Scatter(y=sim, mode="lines"))
        apply_theme(fig, "Synthetic OU")
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Methodology"):
    st.markdown(
        r"""
- **Brownian motion:** increments are i.i.d. \( \mathcal{N}(0,\Delta t) \);
  quadratic variation should track elapsed time.
- **GBM:** exact log-increment simulation uses \( (\mu-\frac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z \).
- **OU:** fit uses exact discrete-time MLE under AR(1) mapping
  \(X_{t+\Delta t}=a+bX_t+\varepsilon_t\), with \(b=e^{-\theta\Delta t}\).
"""
    )
