import numpy as np
import plotly.express as px
import streamlit as st

from src.data.session import get_context
from src.metrics import descriptive
from src.ui.lab import insight_row, num
from src.ui.page_helpers import download_df, render_metric_page_shell, require_data
from src.viz.plots import apply_theme, eigenvalue_bar

st.set_page_config(page_title="Covariance & Correlation | Saito", layout="wide")
render_metric_page_shell(title="Covariance and Correlation", explain_key="covariance", showcase_key="covariance")

if not require_data(min_assets=2):
    st.stop()

ctx = get_context()
r = ctx.returns.dropna()

corr_mode = st.radio(
    "Estimator",
    ["Sample correlation", "Ledoit–Wolf shrinkage", "Simple blend toward identity"],
    horizontal=True,
)

if corr_mode == "Simple blend toward identity":
    shrink_slider = st.slider(
        r"Diagonal blend \(\lambda\cdot I + (1-\lambda)\cdot R\)",
        min_value=0.0,
        max_value=0.95,
        value=0.15,
        step=0.05,
    )
else:
    shrink_slider = 0.0

if corr_mode == "Ledoit–Wolf shrinkage":
    _cov_hat, corr = descriptive.ledoit_wolf_covariance(r)
elif corr_mode == "Simple blend toward identity" and shrink_slider > 0:
    corr = descriptive.blended_correlation(r, shrink_slider)
else:
    corr = descriptive.correlation_matrix(r)

vals, _vecs = descriptive.eigen_spectrum(corr)

st.markdown("### Insights")
st.subheader("Insight")
top_share = float(vals[0] / np.sum(vals)) if len(vals) else float("nan")
insight_row(
    [
        ("Top eigen share", num(top_share)),
        ("N assets", str(r.shape[1])),
        ("T obs", str(r.shape[0])),
        ("Estimator", corr_mode),
    ]
)

tab1, tab2 = st.tabs(["Correlation heatmap", "Eigenvalue spectrum"])
st.markdown("### Visuals")
with tab1:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    apply_theme(fig, corr_mode)
    st.plotly_chart(fig, use_container_width=True)
    download_df("Download correlation CSV", corr, "correlation.csv")

with tab2:
    fig2 = eigenvalue_bar(vals, "Eigenvalues of correlation matrix")
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Large leading eigenvalues often resemble a latent **market factor**. "
        f"Rough MP noise band width scales as O(1/√n)≈ {1/np.sqrt(r.shape[0]):.4f}; use RMT page for formal bulk."
    )

with st.expander("Methodology"):
    st.markdown(
        "Sample correlation uses pairwise-complete observations. "
        "Ledoit-Wolf builds a shrinkage covariance first, then converts to implied correlation. "
        "Identity blending is a fast pedagogical shrinkage baseline."
    )
