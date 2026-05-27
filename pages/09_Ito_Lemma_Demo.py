import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.metrics import stochastic
from src.ui.lab import insight_row, num
from src.ui.page_helpers import render_metric_page_shell
from src.viz.plots import apply_theme

st.set_page_config(page_title="Itô Lemma | Saito", layout="wide")
render_metric_page_shell(title="Itô Lemma Demonstration", explain_key="ito")

mu = st.number_input("μ (annual)", 0.08)
sigma = st.number_input("σ (annual)", 0.25)
s0 = st.number_input("S0", 100.0)
n_steps = st.slider("Steps", 100, 500, 252)
t_horizon = n_steps / 252

ito_info = stochastic.apply_ito_lemma_gbm_log(mu, sigma, t_horizon)
st.markdown("### Insights")
st.subheader("Insight")
insight_row(
    [
        ("Itô log drift", num(ito_info["ito_log_drift"])),
        ("naive drift", num(ito_info["naive_log_drift"])),
        ("−½σ²t correction", num(ito_info["correction"])),
        ("horizon (y)", num(t_horizon)),
    ]
)

demo = st.radio("Demo", ["log(S) drift", "f(S)=S² Itô term"], horizontal=True)
st.markdown("### Visuals")

if demo.startswith("log"):
    out = stochastic.ito_log_demo(s0, mu, sigma, n_steps)
    bias = float(np.mean(out["log_naive"] - out["log_ito"]))
    st.caption(f"Mean log-path bias (naive − Itô): {bias:.6f}")

    tab_log, tab_price = st.tabs(["Log paths", "Price paths"])
    with tab_log:
        fig = go.Figure()
        fig.add_scatter(y=out["log_ito"], name="d(log S) Itô-correct")
        fig.add_scatter(y=out["log_naive"], name="d(log S) naive (wrong drift)")
        apply_theme(fig, "Log-price dynamics under GBM")
        st.plotly_chart(fig, use_container_width=True)
    with tab_price:
        fig2 = go.Figure(go.Scatter(y=out["prices"], mode="lines", name="S_t"))
        apply_theme(fig2, "Simulated price path")
        st.plotly_chart(fig2, use_container_width=True)
else:
    out = stochastic.ito_power_demo(s0, mu, sigma, n_steps)
    fig = go.Figure()
    fig.add_scatter(y=out["f_ito"], name="Itô f(S)=S²")
    fig.add_scatter(y=out["f_naive"], name="naive chain rule")
    fig.add_scatter(y=out["f_actual"], name="actual S²", line=dict(dash="dot"))
    apply_theme(fig, "Itô on f(S)=S² — extra σ²S² dt term")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Methodology"):
    st.markdown(
        r"""
**GBM:** \(dS_t = \mu S_t\,dt + \sigma S_t\,dW_t\).

**Itô on \(\ln S\):** \(d\ln S = (\mu - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dW\) — the \(-\tfrac{1}{2}\sigma^2\) term
is the convexity correction (not present in ordinary calculus).

**Itô on \(S^2\):** \(d(S^2) = 2S\,dS + \sigma^2 S^2\,dt\) — the \(\sigma^2 S^2\,dt\) term is the same correction idea.

Euler–Maruyama integrates the SDE directly; Itô's lemma tells you the correct dynamics for functions of \(S\).
"""
    )
