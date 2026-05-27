import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.session import get_context
from src.metrics import optimization
from src.ui.lab import insight_row, num
from src.ui.page_helpers import render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Constrained Optimization | Saito", layout="wide")
render_metric_page_shell(
    title="Constrained Optimization (Markowitz and KKT)",
    explain_key="optimization_kkt",
    showcase_key="optimization",
)

if not require_data(min_assets=2, min_obs=60):
    st.stop()

ctx = get_context()
with st.form("constrained_opt_controls"):
    long_only = st.checkbox("Long-only (w ≥ 0)", value=True)
    rf = st.slider(
        "Risk-free rate (annualized, tangency only when shorts allowed)",
        -0.05,
        0.12,
        0.0,
        0.005,
    )
    submitted = st.form_submit_button("Apply controls")
if not submitted:
    st.info("Adjust controls and click **Apply controls** to recompute frontier.")
    st.stop()

mu = ctx.returns.mean().values * 252
cov = ctx.returns.cov().values * 252

@st.cache_data(show_spinner=False)
def _compute_frontier(mu_arr: np.ndarray, cov_arr: np.ndarray, long_only_flag: bool):
    return optimization.markowitz_frontier(mu_arr, cov_arr, n_points=30, long_only=long_only_flag)

points = _compute_frontier(mu, cov, long_only)

rets = [p.target_return for p in points]
vols = [p.volatility for p in points]

fig = go.Figure(go.Scatter(x=vols, y=rets, mode="lines+markers", name="frontier (numerical)"))
st.markdown("### Visuals")

analytic_err = None
if not long_only:
    targets_line = np.linspace(float(np.min(mu)), float(np.max(mu)), 30)
    avols, arets = [], []
    for t in targets_line:
        w_a = optimization.markowitz_min_variance_target_analytic(mu, cov, float(t))
        avols.append(float(np.sqrt(max(w_a @ cov @ w_a, 0.0))))
        arets.append(float(w_a @ mu))
    fig.add_scatter(
        x=avols,
        y=arets,
        mode="lines",
        name="frontier (Lagrange, shorts OK)",
        line=dict(dash="dash", width=3),
    )
    diffs = []
    for p in points:
        w_a = optimization.markowitz_min_variance_target_analytic(mu, cov, float(p.target_return))
        v_a = float(np.sqrt(max(w_a @ cov @ w_a, 0.0)))
        diffs.append(abs(v_a - p.volatility))
    analytic_err = float(np.max(diffs)) if diffs else None

    w_t = optimization.tangency_portfolio_weights_uncstr(mu, cov, rf=rf)
    ret_t = float(w_t @ mu)
    vol_t = float(np.sqrt(max(w_t @ cov @ w_t, 0.0)))
    sp = optimization.portfolio_sharpe(w_t, mu, cov, rf=rf)
    fig.add_scatter(
        x=[vol_t],
        y=[ret_t],
        mode="markers",
        name=f"tangency (Sharpe {sp:.2f})",
        marker=dict(size=18, symbol="star"),
    )

w_min = optimization.min_variance_portfolio(mu, cov, long_only=long_only)
vol_min = float((w_min @ cov @ w_min) ** 0.5)
ret_min = float(w_min @ mu)
fig.add_scatter(x=[vol_min], y=[ret_min], mode="markers", name="min variance", marker_size=14)

if not long_only:
    w_gmv = optimization.global_minimum_variance_weights_analytic(cov)
    vol_gmv = float(np.sqrt(max(w_gmv @ cov @ w_gmv, 0.0)))
    ret_gmv = float(w_gmv @ mu)
    fig.add_scatter(
        x=[vol_gmv],
        y=[ret_gmv],
        mode="markers",
        name="GMV (closed form)",
        marker=dict(size=12, symbol="x"),
    )

apply_theme(fig, "Efficient frontier (annualized)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Insights")
insight_row(
    [
        ("min-var vol", num(vol_min)),
        ("min-var return", num(ret_min)),
        ("assets", str(len(mu))),
        ("long-only", "yes" if long_only else "no"),
    ]
)

if not long_only and analytic_err is not None:
    st.subheader("Diagnostics")
    insight_row(
        [
            ("max |σ_num − σ_Lagrange|", num(analytic_err)),
            ("GMV closed vs SLSQP Δσ", num(abs(vol_gmv - vol_min))),
            ("tangency Sharpe", num(sp)),
        ]
    )

st.markdown("### Results")
st.subheader("Minimum-variance portfolio")
tickers = list(ctx.returns.columns)
weights = pd.DataFrame({"ticker": tickers, "weight": w_min.astype(float)}).sort_values(
    "weight", ascending=False
)
st.dataframe(weights, use_container_width=True, hide_index=True)
binding = sorted([tickers[i] for i in range(len(w_min)) if w_min[i] < 5e-4])
with st.expander("KKT / active constraints"):
    st.markdown(
        r"""
- **Budget** \( \mathbf 1^{\mathsf T} w = 1 \): equality ⇒ Lagrange multiplier.
- **Mean target** \( \mu^{\mathsf T} w = r \): second equality ⇒ second multiplier ⇒ two‑fund
  theorem — portfolio is Σ⁻𝟙 and Σ⁻μ linear (see analytic frontier).
- With **shorts permitted**, \(\arg\min w^{\mathsf T}\Sigma w\) with both equalities uses the \(2\times 2\)
  system in **`markowitz_min_variance_target_analytic`**.
- **Tangency** (max Sharpe vs. cash \(r_f\)): shorts OK ⇒
  \( w \propto \Sigma^{-1}(\mu - r_f \mathbf 1) \),
  \(\mathbf 1^{\mathsf T}w = 1\) after normalization.
- **Long-only \(w\ge 0\)**: complementary slackness ⇒ zeros on the boundary; SLSQP clips.
"""
    )
    st.write({"near_zero_weights": binding or "none"})
