import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.data.session import get_context
from src.metrics import distributions, evt
from src.ui.lab import insight_row, num
from src.ui.page_helpers import download_df, download_json, render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Fat Tails & EVT | Saito", layout="wide")
render_metric_page_shell(title="Fat Tails and EVT", explain_key="evt", showcase_key="evt")

st.caption("Tip: load **fat_t_student** preset or `tail_events.csv` for heavy-tail demos.")

if not require_data(min_obs=200):
    st.stop()

ctx = get_context()
asset = st.selectbox("Asset", list(ctx.returns.columns))
x = ctx.returns[asset].dropna().values
losses = -x


@st.cache_data(show_spinner=False)
def _fit_gpd(loss_arr: np.ndarray, threshold: float):
    return evt.fit_gpd_pot(loss_arr, threshold=threshold)


@st.cache_data(show_spinner=False)
def _var_es_compare(loss_arr: np.ndarray, confidence: float, threshold: float):
    return evt.var_es_comparison(loss_arr, confidence=confidence, gpd_threshold=threshold)

tab_fit, tab_pot, tab_levels, tab_compare = st.tabs(
    ["Tail fit", "POT / GPD", "Return levels", "VaR/ES compare"]
)

with tab_fit:
    norm = distributions.fit_normal(x)
    stud = distributions.fit_student_t(x)
    st.subheader("Insight")
    insight_row(
        [
            ("Better AIC", "Student‑t" if stud.aic < norm.aic else "Normal"),
            ("t df", num(float(stud.params["df"]))),
            ("ΔAIC", num(float(stud.aic - norm.aic))),
            ("n", str(len(x))),
        ]
    )
    xs = np.linspace(x.min(), x.max(), 200)
    fig = go.Figure()
    fig.add_histogram(x=x, nbinsx=50, name="returns", opacity=0.5)
    fig.add_scatter(x=xs, y=stats.norm.pdf(xs, norm.params["mu"], norm.params["sigma"]), name="normal")
    fig.add_scatter(
        x=xs,
        y=stats.t.pdf(xs, stud.params["df"], stud.params["loc"], stud.params["scale"]),
        name="student-t",
    )
    apply_theme(fig, "Fat tails: Normal vs Student-t")
    st.plotly_chart(fig, use_container_width=True)

with tab_pot:
    auto_u, thr_auto, means_auto = evt.mean_excess_stability(losses)
    q = st.slider("POT threshold quantile", 0.9, 0.99, 0.95)
    threshold = float(np.quantile(losses, q))
    use_auto = st.checkbox("Use stability-picked threshold", value=False)
    if use_auto:
        threshold = auto_u
    try:
        gpd = _fit_gpd(losses, threshold)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    excess = losses[losses > threshold] - threshold
    theory, emp = evt.gpd_qq_coordinates(excess, gpd.shape, gpd.scale)
    st.subheader("Insight")
    insight_row(
        [
            ("u", num(float(gpd.threshold))),
            ("ξ shape", num(float(gpd.shape))),
            ("exceedances", str(gpd.n_exceedances)),
            ("VaR 99", num(float(gpd.var_99))),
        ]
    )
    if np.isnan(gpd.es_99):
        st.warning(
            "GPD ES is undefined for shape ξ ≥ 1; "
            "this indicates an extremely heavy (infinite-mean) tail regime."
        )
    st.json(
        {
            "threshold": gpd.threshold,
            "shape": gpd.shape,
            "scale": gpd.scale,
            "n_exceedances": gpd.n_exceedances,
            "VaR_99": gpd.var_99,
            "ES_99": gpd.es_99,
        }
    )
    download_json("Download EVT JSON", gpd.__dict__, "evt_gpd.json")

    col1, col2 = st.columns(2)
    with col1:
        fig_m = go.Figure(go.Scatter(x=thr_auto, y=means_auto, mode="lines+markers"))
        fig_m.add_vline(x=threshold, line_dash="dash", annotation_text="u")
        apply_theme(fig_m, "Mean excess plot")
        st.plotly_chart(fig_m, use_container_width=True)
    with col2:
        lims = [float(min(theory.min(), emp.min())), float(max(theory.max(), emp.max()))]
        fig_q = go.Figure()
        fig_q.add_scatter(x=theory, y=emp, mode="markers", name="exceedances")
        fig_q.add_scatter(x=lims, y=lims, mode="lines", name="y=x", line=dict(dash="dash"))
        apply_theme(fig_q, "GPD QQ — exceedances")
        st.plotly_chart(fig_q, use_container_width=True)

    xi_hill, ks, hill_curve = evt.hill_tail_index(losses)
    fig_h = go.Figure(go.Scatter(x=ks, y=hill_curve, mode="lines"))
    apply_theme(fig_h, f"Hill tail index (ξ ≈ {xi_hill:.3f})")
    st.plotly_chart(fig_h, use_container_width=True)

with tab_levels:
    q_lvl = st.slider("Threshold quantile (levels)", 0.9, 0.99, 0.95, key="rl_q")
    u_lvl = float(np.quantile(losses, q_lvl))
    gpd_lvl = _fit_gpd(losses, u_lvl)
    levels = evt.gpd_return_level(losses, u_lvl, gpd_lvl.shape, gpd_lvl.scale)
    fig_rl = go.Figure()
    fig_rl.add_scatter(x=levels["prob"], y=levels["return_level"], mode="lines+markers")
    apply_theme(fig_rl, "GPD return levels")
    st.plotly_chart(fig_rl, use_container_width=True)
    st.dataframe(levels, use_container_width=True, hide_index=True)
    download_df("Download return levels CSV", levels, "gpd_return_levels.csv")

with tab_compare:
    conf = st.slider("Confidence", 0.95, 0.999, 0.99, format="%.3f")
    u_cmp = float(np.quantile(losses, 0.95))
    cmp_tbl = _var_es_compare(losses, confidence=conf, threshold=u_cmp)
    emp_var = float(cmp_tbl.loc[cmp_tbl["method"] == "empirical", "VaR"].iloc[0])
    gpd_var = float(cmp_tbl.loc[cmp_tbl["method"] == "gpd", "VaR"].iloc[0])
    st.subheader("Insight")
    insight_row(
        [
            ("confidence", num(conf)),
            ("empirical VaR", num(emp_var)),
            ("GPD VaR", num(gpd_var)),
            ("Δ (GPD−emp)", num(gpd_var - emp_var)),
        ]
    )
    st.dataframe(cmp_tbl, use_container_width=True, hide_index=True)
    fig_c = go.Figure()
    fig_c.add_bar(x=cmp_tbl["method"], y=cmp_tbl["VaR"], name="VaR")
    fig_c.add_bar(x=cmp_tbl["method"], y=cmp_tbl["ES"], name="ES", opacity=0.6)
    apply_theme(fig_c, "VaR / ES by method")
    st.plotly_chart(fig_c, use_container_width=True)
