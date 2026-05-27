import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src.data.session import get_context
from src.metrics import distributions
from src.ui.lab import insight_row, num
from src.ui.page_helpers import download_json, render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Distributions & CLT | Saito", layout="wide")
render_metric_page_shell(title="Distributions and CLT", explain_key="distributions", showcase_key="distributions")

ctx = get_context()
mode = st.radio(
    "Mode",
    ["Return distribution fits", "CLT (standardized)", "Joint / conditional", "Count models"],
    horizontal=True,
)

if mode == "Return distribution fits":
    if not require_data():
        st.stop()
    asset = st.selectbox("Asset", list(ctx.returns.columns))
    x = ctx.returns[asset].dropna().values
    norm = distributions.fit_normal(x)
    stud = distributions.fit_student_t(x)
    st.markdown("### Insights")
    st.subheader("Insight")
    insight_row(
        [
            ("Better AIC", "Student‑t" if stud.aic < norm.aic else "Normal"),
            ("ΔAIC (t−N)", num(float(stud.aic - norm.aic))),
            ("t df", num(float(stud.params["df"]))),
            ("N σ", num(float(norm.params["sigma"]))),
        ]
    )
    st.table(
        {
            "model": ["normal", "student_t"],
            "AIC": [norm.aic, stud.aic],
            "BIC": [norm.bic, stud.bic],
        }
    )
    tabs = st.tabs(["Histogram + PDF", "QQ overlays"])
    st.markdown("### Visuals")
    with tabs[0]:
        fig = go.Figure()
        fig.add_histogram(x=x, nbinsx=50, name="empirical", opacity=0.55)
        xs = np.linspace(x.min(), x.max(), 240)
        fig.add_scatter(x=xs, y=stats.norm.pdf(xs, **{k: norm.params[k] for k in ("mu", "sigma")}), name="normal")
        fig.add_scatter(
            x=xs,
            y=stats.t.pdf(xs, stud.params["df"], stud.params["loc"], stud.params["scale"]),
            name="student-t",
        )
        apply_theme(fig, "Fitted densities")
        st.plotly_chart(fig, use_container_width=True)
        download_json(
            "Download fits JSON",
            {"normal": norm.params, "student_t": stud.params},
            "fits.json",
        )
    with tabs[1]:
        tn, en = distributions.qq_normal_coordinates(x, norm.params["mu"], norm.params["sigma"])
        tt, et = distributions.qq_student_t_coordinates(
            x, stud.params["df"], stud.params["loc"], stud.params["scale"]
        )
        lims = [float(min(tn.min(), en.min())), float(max(tn.max(), en.max()))]
        fig_q = go.Figure()
        fig_q.add_scatter(x=tn, y=en, mode="markers", name="vs Normal", opacity=0.7)
        fig_q.add_scatter(x=lims, y=lims, mode="lines", name="y=x", line=dict(dash="dash"))
        apply_theme(fig_q, "QQ — empirical vs Normal quantiles")
        st.plotly_chart(fig_q, use_container_width=True)
        fig_q2 = go.Figure()
        fig_q2.add_scatter(x=tt, y=et, mode="markers", name="vs Student-t", opacity=0.7)
        lims2 = [float(min(tt.min(), et.min())), float(max(tt.max(), et.max()))]
        fig_q2.add_scatter(x=lims2, y=lims2, mode="lines", name="y=x", line=dict(dash="dash"))
        apply_theme(fig_q2, "QQ — empirical vs Student-t quantiles")
        st.plotly_chart(fig_q2, use_container_width=True)

elif mode == "CLT (standardized)":
    if not require_data(min_obs=120):
        st.stop()
    asset = st.selectbox("Asset", list(ctx.returns.columns))
    x = ctx.returns[asset].dropna().values
    n_clt = st.slider("Sample size n", 5, 200, 40)
    z = distributions.clt_standardized_sample_means(x, n_clt, n_replicates=2500)
    st.markdown("### Visuals")
    grid = np.linspace(z.min(), z.max(), 300)
    fig = go.Figure()
    fig.add_histogram(x=z, nbinsx=45, name="√n(̄X−μ)/σ", opacity=0.55, histnorm="probability density")
    fig.add_scatter(x=grid, y=stats.norm.pdf(grid, 0, 1), name="N(0,1)", line=dict(width=3))
    apply_theme(fig, "CLT — standardized sample mean")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Bootstrap resamples rows with replacement; curve should track the standard normal as n grows.")

elif mode == "Joint / conditional":
    if not require_data(min_assets=2):
        st.stop()
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Asset X", list(ctx.returns.columns), index=0)
    with c2:
        b = st.selectbox("Asset Y", list(ctx.returns.columns), index=1)
    xv, yv = ctx.returns[a].dropna().values, ctx.returns[b].dropna().values
    n = min(len(xv), len(yv))
    xv, yv = xv[-n:], yv[-n:]
    gm = distributions.gaussian_bivariate_moments(xv, yv)
    st.markdown("### Insights")
    insight_row(
        [
            ("rho", num(float(gm.rho))),
            ("beta(y|x)", num(float(gm.beta_y_on_x))),
            ("sigma_x", num(float(gm.std_x))),
            ("sigma_y", num(float(gm.std_y))),
        ]
    )
    st.markdown("### Visuals")
    grid_x = np.linspace(float(np.min(xv)), float(np.max(xv)), 120)
    cond_mean = gm.mean_y + gm.beta_y_on_x * (grid_x - gm.mean_x)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=xv, y=yv, mode="markers", opacity=0.45, marker=dict(size=6), name="observed")
    )
    fig.add_trace(
        go.Scatter(
            x=grid_x,
            y=cond_mean,
            mode="lines",
            name="E[Y|X] Gaussian",
            line=dict(width=3),
        )
    )
    apply_theme(fig, f"Joint scatter — {a} vs {b}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Gaussian ρ̂≈ {gm.rho:.3f}; "
        "Var(Y | X) from bivariate‑normal approximation (constant in X for jointly Gaussian pairs)."
    )
    fig_h = px.density_heatmap(x=xv, y=yv, nbinsx=35, nbinsy=35, color_continuous_scale="Viridis")
    apply_theme(fig_h, "Empirical joint density (binned)")
    st.plotly_chart(fig_h, use_container_width=True)
    x0 = st.slider("Condition on X near", float(xv.min()), float(xv.max()), float(np.median(xv)))
    muc, _vc = distributions.conditional_gaussian_mean_var_y_given_x(x0, gm)
    st.write({"Gaussian_E_Y_given_x": round(muc, 6)})
    bw = st.slider("Bandwidth", 0.0005, 0.02, 0.005, format="%.4f")
    centers, counts = distributions.conditional_hist(xv, yv, x0, bw)
    if len(centers):
        fig2 = go.Figure(go.Bar(x=centers, y=counts))
        apply_theme(fig2, f"Conditional slice P(Y | X ≈ {x0:.4f})")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.caption("Count models from trade win/loss windows or synthetic counts.")
    if require_data(min_obs=50):
        asset = st.selectbox("Asset (for win/loss)", list(ctx.returns.columns))
        window = st.slider("Rolling window (trials)", 3, 20, 5)
        counts = distributions.trade_win_loss_counts(ctx.returns[asset].values, window)
        n_trials = window
        bfit = distributions.fit_binomial(counts.astype(int), n_trials)
        pfit = distributions.fit_poisson(counts.astype(int))
        st.json({"binomial": bfit.params, "poisson": pfit.params})
    synth_n = st.slider("Synthetic Poisson λ", 1.0, 20.0, 5.0)
    synth = np.random.default_rng(0).poisson(synth_n, 500)
    pfit2 = distributions.fit_poisson(synth)
    st.write(f"Synthetic Poisson fit: λ̂ = {pfit2.params['lambda']:.2f}")
