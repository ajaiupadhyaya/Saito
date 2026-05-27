import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data.session import get_context
from src.metrics import rmt
from src.ui.lab import insight_row, num
from src.ui.page_helpers import download_df, render_metric_page_shell, require_data
from src.viz.plots import apply_theme, correlation_heatmap

st.set_page_config(page_title="RMT | Saito", layout="wide")
render_metric_page_shell(
    title="Random Matrix Theory",
    explain_key="rmt",
    showcase_key="rmt",
)

st.caption("Tip: load **`corr_noise_50x50.csv`** or **crash_correlation** preset (N≥5, T≥100).")

if not require_data(min_assets=5, min_obs=100):
    st.stop()

ctx = get_context()
r = ctx.returns.dropna()
q = r.shape[1] / r.shape[0]
lam_minus, lam_plus = rmt.marchenko_pastur_bounds(q)

denoise_method = st.radio(
    "Denoise method",
    ["Mean bulk shrink", "Constant residual (López de Prado)"],
    horizontal=True,
)
method_key = "constant" if "Constant" in denoise_method else "mean"

@st.cache_data(show_spinner=False)
def _denoise(df, method):
    return rmt.denoise_correlation(df, method=method)

try:
    corr_clean, raw_evals, clean_evals = _denoise(r, method_key)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

evals = rmt.correlation_eigenvalues(r)
lam_grid = np.linspace(max(1e-6, evals.min() * 0.5), evals.max() * 1.1, 200)
density = rmt.mp_density(lam_grid, q)
outside = int(np.sum((raw_evals < lam_minus) | (raw_evals > lam_plus)))
eff_rank = rmt.effective_rank(raw_evals)

vols = r.std()
var_raw = rmt.equal_weight_variance(r.corr(), vols)
var_clean = rmt.equal_weight_variance(corr_clean, vols)

st.subheader("Insight")
insight_row(
    [
        ("q=N/T", num(float(q))),
        ("MP bulk", f"[{lam_minus:.3f}, {lam_plus:.3f}]"),
        ("signal λ's", str(outside)),
        ("Δ eq-wt vol", num(float(np.sqrt(var_clean) - np.sqrt(var_raw)))),
    ]
)

tab_spec, tab_corr, tab_port = st.tabs(["Eigen spectrum", "Correlation matrices", "Portfolio impact"])

with tab_spec:
    fig = go.Figure()
    fig.add_histogram(x=evals, nbinsx=30, name="empirical eigenvalues", opacity=0.6)
    fig.add_scatter(x=lam_grid, y=density, name="MP density", mode="lines")
    fig.add_vline(x=lam_minus, line_dash="dash", annotation_text="λ−")
    fig.add_vline(x=lam_plus, line_dash="dash", annotation_text="λ+")
    apply_theme(fig, "Eigenvalue spectrum vs Marčenko–Pastur")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Effective rank (entropy): {eff_rank:.2f}")
    eig_tbl = rmt.eigenvalue_comparison_table(raw_evals, clean_evals, q)
    st.dataframe(eig_tbl.head(15), use_container_width=True, hide_index=True)
    download_df("Download eigenvalue table CSV", eig_tbl, "rmt_eigenvalues.csv")

with tab_corr:
    col1, col2 = st.columns(2)
    with col1:
        fig2 = correlation_heatmap(r.corr(), "Raw correlation")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = correlation_heatmap(corr_clean, "Denoised correlation")
        st.plotly_chart(fig3, use_container_width=True)
    download_df("Download denoised correlation CSV", corr_clean, "correlation_denoised.csv")

with tab_port:
    st.dataframe(
        {
            "metric": ["Equal-weight variance", "Equal-weight vol"],
            "raw": [var_raw, np.sqrt(var_raw)],
            "denoised": [var_clean, np.sqrt(var_clean)],
        },
        use_container_width=True,
        hide_index=True,
    )
    fig_p = go.Figure()
    fig_p.add_bar(x=["raw", "denoised"], y=[np.sqrt(var_raw), np.sqrt(var_clean)], name="eq-wt vol")
    apply_theme(fig_p, "Portfolio vol impact")
    st.plotly_chart(fig_p, use_container_width=True)

st.write(f"Eigenvalues outside MP bulk: **{outside}** / {len(raw_evals)}")

with st.expander("Methodology"):
    st.markdown(
        r"""
RMT compares empirical correlation eigenvalues against the Marcenko-Pastur bulk under \(q=N/T\).
Bulk eigenvalues are shrunk while preserving dominant eigenvectors, then projected back to a PSD
correlation matrix with unit diagonal for downstream portfolio use.
"""
    )
