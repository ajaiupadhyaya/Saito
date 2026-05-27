import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.session import get_context
from src.metrics import descriptive, linear_algebra
from src.ui.lab import insight_row, num
from src.ui.page_helpers import download_df, render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="PCA, SVD & Eigen | Saito", layout="wide")
render_metric_page_shell(title="PCA, SVD, and Eigen Decomposition", explain_key="pca", showcase_key="pca")

if not require_data(min_assets=3, min_obs=60):
    st.stop()

ctx = get_context()
r = ctx.returns.dropna()


@st.cache_data(show_spinner=False)
def _compute_pca_agreement(df: pd.DataFrame, standardize: bool, n_components: int):
    return linear_algebra.pca_svd_agreement(df, standardize=standardize, n_components=n_components)


@st.cache_data(show_spinner=False)
def _compute_pca(df: pd.DataFrame, n_components: int, standardize: bool):
    return linear_algebra.pca_analysis(df, n_components=n_components, standardize=standardize)


@st.cache_data(show_spinner=False)
def _compute_svd(df: pd.DataFrame, rank: int):
    return linear_algebra.svd_analysis(df, rank=rank), linear_algebra.svd_relative_error_curve(df)

col_cf1, col_cf2 = st.columns(2)
with col_cf1:
    k_pca = st.slider("PCA components (k)", 1, min(ctx.n_assets, 15), min(5, ctx.n_assets))
with col_cf2:
    max_rank = min(ctx.n_assets, ctx.n_obs) - 1
    max_rank = max(1, max_rank)
    rank_svd = st.slider("SVD truncation rank", 1, max_rank, min(5, max_rank))

standardize = st.toggle("Standardize columns before PCA (corr-style factors)", value=True)

rat_pca, rat_svd, diff_pc = _compute_pca_agreement(r, standardize=standardize, n_components=k_pca)
pca = _compute_pca(r, n_components=k_pca, standardize=standardize)

tab_pca, tab_svd, tab_eigen = st.tabs(["PCA / biplot", "SVD approximation", "Eigen decomposition"])

st.markdown("### Insights")
st.subheader("Insight")
target_var = 0.80
k80 = int(np.searchsorted(pca.cumulative_variance, target_var) + 1)
insight_row(
    [
        ("k (chosen)", str(k_pca)),
        ("cum var@k", num(float(pca.cumulative_variance[k_pca - 1]))),
        ("k for 80%", str(k80)),
        ("standardized", "yes" if standardize else "no"),
        ("PCA−SVD Δ (ratios)", num(diff_pc)),
    ]
)

st.markdown("### Visuals")
with tab_pca:
    fig = go.Figure()
    fig.add_scatter(y=pca.explained_variance_ratio, mode="lines+markers", name="Variance share")
    fig.add_scatter(y=pca.cumulative_variance, mode="lines", name="Cumulative")
    apply_theme(fig, "Explained variance (PCA)")
    st.plotly_chart(fig, use_container_width=True)

    if pca.scores.shape[1] >= 2:
        arrows = linear_algebra.scaled_loadings_for_biplot(pca.components, pca.explained_variance)
        arrow_scale_ui = st.slider("Biplot arrow scale", 0.2, 3.0, 1.0, 0.1)
        scaled_arrows = arrows * arrow_scale_ui
        fig3 = go.Figure()
        fig3.add_scatter(
            x=pca.scores[:, 0],
            y=pca.scores[:, 1],
            mode="markers",
            marker=dict(size=6, opacity=0.45, color="#74b9ff"),
            name="scores",
            hovertemplate="PC1 %{x:.3f}<br>PC2 %{y:.3f}<extra></extra>",
        )
        for j, name in enumerate(pca.column_names):
            ax, ay = float(scaled_arrows[j, 0]), float(scaled_arrows[j, 1])
            fig3.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=ax,
                y1=ay,
                line=dict(color="#ff9f43", width=3),
            )
            fig3.add_annotation(
                x=ax,
                y=ay,
                text=name,
                showarrow=False,
                font=dict(size=11, color="#feca57"),
                yshift=8,
            )
        xmin, xmax, ymin, ymax = linear_algebra.biplot_arrow_scale(pca.scores[:, :2], scaled_arrows)
        fig3.update_xaxes(range=[xmin, xmax])
        fig3.update_yaxes(range=[ymin, ymax])
        apply_theme(fig3, "PCA biplot — scores + loadings (scaled by √(λ))")
        st.plotly_chart(fig3, use_container_width=True)

    loadings_tbl = pd.DataFrame(
        pca.components[: min(5, pca.components.shape[0])],
        columns=pca.column_names,
        index=[f"PC{i + 1}" for i in range(min(5, pca.components.shape[0]))],
    ).T
    download_df("Download PCA loadings (CSV)", loadings_tbl, "pca_loadings.csv")

with tab_svd:
    svd, svd_curve = _compute_svd(r, rank_svd)
    sigmas, err_curve = svd_curve
    sr = linear_algebra.stable_rank(sigmas)
    ranks_vec = np.arange(1, len(err_curve) + 1)
    fig2 = go.Figure(go.Bar(x=np.arange(1, len(sigmas) + 1), y=sigmas, name="σᵢ"))
    apply_theme(fig2, f"SVD spectrum @ chosen rank ({rank_svd}) — Fro error {svd.reconstruction_error:.3f}")
    st.plotly_chart(fig2, use_container_width=True)
    fig_err = go.Figure(go.Scatter(x=ranks_vec, y=err_curve, mode="lines+markers", name="error"))
    apply_theme(fig_err, "Truncation residual vs rank")
    st.plotly_chart(fig_err, use_container_width=True)
    st.caption(
        f"SVD is on the raw T×N return matrix (no standardization). Stable rank ≈ {sr:.2f} "
        "(Frobenius energy / squared top singular value)."
    )

with tab_eigen:
    corr_mat = descriptive.correlation_matrix(r)
    evals, evecs = linear_algebra.eigen_decomposition(corr_mat)
    fig4 = go.Figure(go.Bar(x=list(range(1, len(evals) + 1)), y=evals, name="λᵢ"))
    apply_theme(fig4, "Correlation eigenvalues")
    st.plotly_chart(fig4, use_container_width=True)
    fig5 = go.Figure(go.Bar(x=list(r.columns), y=evecs[:, 0]))
    apply_theme(fig5, "Leading eigenvector (market beta proxy)")
    st.plotly_chart(fig5, use_container_width=True)

with st.expander("Linear algebra cheatsheet"):
    st.markdown(
        r"""
- **PCA** ↔ **SVD**: for column-centered \(X\), the PC variance shares match \(\sigma_i^2 / \sum_j \sigma_j^2\)
  from the compact SVD \(X = U\Sigma V^{\mathsf T}\) (see *Insight* Δ).
- **Correlation eigenpairs** factorize structural co-movement vs idiosyncratic directions.
- **Eigenvectors** describe **risk directions** — the leading vector often aligns with a latent market factor."""
    )
