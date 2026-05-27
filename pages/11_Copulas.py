import plotly.graph_objects as go
import streamlit as st

from src.data.session import get_context
from src.metrics import copulas
from src.ui.lab import insight_row, num
from src.ui.page_helpers import render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Copulas | Saito", layout="wide")
render_metric_page_shell(title="Copulas", explain_key="copulas", showcase_key="copulas")

st.caption("Tip: load **crash_correlation** preset for joint tail stress.")

family = st.selectbox(
    "Copula",
    ["Gaussian", "Student-t", "Clayton", "Gumbel", "Frank"],
)

if not require_data(min_assets=2):
    st.stop()

ctx = get_context()
c1, c2 = st.columns(2)
with c1:
    a = st.selectbox("Asset 1", list(ctx.returns.columns))
with c2:
    b = st.selectbox("Asset 2", list(ctx.returns.columns), index=min(1, ctx.n_assets - 1))

crash_only = st.checkbox("Crash window only (5 days around mid-sample shock)", value=False)
crash_window = st.slider("Crash window length", 20, 120, 40, 5, disabled=not crash_only)
x = ctx.returns[a].dropna()
y = ctx.returns[b].dropna()
n = min(len(x), len(y))
x, y = x[-n:].values, y[-n:].values
if crash_only and n > crash_window:
    mid = n // 3
    x, y = x[mid : mid + crash_window], y[mid : mid + crash_window]
    if len(x) < 30:
        st.warning("Crash window has very few points; copula fits may be unstable. Use full sample for robust fits.")

if len(x) < 10:
    st.error("Need at least 10 aligned observations for copula fit.")
    st.stop()

tau = copulas.kendall_tau(x, y)
df_t = 5.0
if family == "Gaussian":
    res = copulas.fit_gaussian_copula(x, y)
    tail = copulas.tail_dependence_coefficients("gaussian", rho=res.parameter)
elif family == "Student-t":
    df_t = st.slider("t copula df", 3.0, 15.0, 5.0)
    res = copulas.fit_student_t_copula(x, y, df=df_t)
    tail = copulas.tail_dependence_coefficients("student_t", rho=res.parameter, df=df_t)
elif family == "Clayton":
    res = copulas.fit_clayton_copula(x, y)
    tail = copulas.tail_dependence_coefficients("clayton", theta=res.parameter)
elif family == "Gumbel":
    res = copulas.fit_gumbel_copula(x, y)
    tail = copulas.tail_dependence_coefficients("gumbel", theta=res.parameter)
else:
    res = copulas.fit_frank_copula(x, y)
    tail = copulas.tail_dependence_coefficients("frank", theta=res.parameter)

samples = res.samples
emp_grid, gx, gy = copulas.empirical_copula_grid(x, y, grid=15)

st.subheader("Insight")
param_label = "ρ" if family in ("Gaussian", "Student-t") else "θ"
insight_row(
    [
        ("Kendall τ", num(tau)),
        (param_label, num(float(res.parameter or res.correlation[0, 1]))),
        ("λ_L", num(tail.lambda_lower)),
        ("λ_U", num(tail.lambda_upper)),
    ]
)
st.caption(f"Sample size used for fit: {len(x)} observations.")

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=samples[:, 0], y=samples[:, 1], mode="markers", opacity=0.45, name="fitted samples")
)
apply_theme(fig, f"{family} copula samples (uniform margins)")
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure(
    data=go.Contour(
        z=emp_grid.T,
        x=gx,
        y=gy,
        contours=dict(coloring="lines"),
        name="empirical",
    )
)
fig2.add_trace(
    go.Histogram2dContour(
        x=samples[:, 0],
        y=samples[:, 1],
        ncontours=12,
        line=dict(color="orange"),
        name="fitted",
    )
)
apply_theme(fig2, "Empirical vs fitted copula contours")
st.plotly_chart(fig2, use_container_width=True)

u_stress = st.slider("Stress u (conditional slice)", 0.05, 0.95, 0.1, 0.05)
cond = copulas.conditional_sample_y_given_x(u_stress, samples)
fig3 = go.Figure(go.Histogram(x=cond, nbinsx=25))
apply_theme(fig3, f"Conditional v | u ≈ {u_stress:.2f}")
st.plotly_chart(fig3, use_container_width=True)

with st.expander("Methodology"):
    st.markdown(
        r"""
**Sklar:** joint CDF = copula(margins). Archimedean families (Clayton/Gumbel/Frank) are fit via Kendall τ → parameter.

**Tail dependence:** Gaussian copula has λ_L = λ_U = 0; Clayton loads lower-tail co-movement; Gumbel upper-tail;
Student-t has symmetric tail dependence increasing as df ↓.

Crash-window toggle isolates the shock segment in the **crash_correlation** synthetic preset.
"""
    )
