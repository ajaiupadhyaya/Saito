import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data.session import get_context
from src.metrics import descriptive, optimization
from src.ui.lab import insight_row, num
from src.ui.page_helpers import render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="SGD & Adam | Saito", layout="wide")
render_metric_page_shell(
    title="Optimization: SGD and Adam",
    explain_key="optimization_sgd",
    showcase_key="optimization",
)

mode = st.radio("Demo", ["Toy losses", "Linear regression MSE on returns"], horizontal=True)

lr = st.slider("Learning rate", 0.001, 0.2, 0.02, format="%.3f")
steps = st.slider("Steps", 50, 800, 250)
beta1 = st.slider("Adam β₁", 0.80, 0.99, 0.9, 0.01)
beta2 = st.slider("Adam β₂", 0.90, 0.9999, 0.999, 0.0001)

if mode == "Toy losses":
    loss_name = st.selectbox("Loss", ["Quadratic (2D)", "Rosenbrock (2D)"])
    if loss_name.startswith("Quadratic"):
        loss_fn = optimization.quadratic_loss
        x0 = st.text_input("Start w (comma)", "2.5, -1.0")
        w0 = [float(x.strip()) for x in x0.split(",")]
    else:
        loss_fn = optimization.rosenbrock
        w0 = [-1.0, 1.0]
    w0_arr = np.array(w0, dtype=float)
else:
    if not require_data(min_assets=2, min_obs=150):
        st.stop()
    ctx = get_context()
    cols = list(ctx.returns.columns)
    target = st.selectbox("Target return", cols)
    default_features = [c for c in cols if c != target][: min(2, len(cols) - 1)]
    features = st.multiselect(
        "Feature returns",
        [c for c in cols if c != target],
        default=default_features,
    )
    if not features:
        st.error("Pick at least one feature column.")
        st.stop()
    X, y = descriptive.regression_design_constant_plus_features(
        ctx.returns,
        target=target,
        features=features,
    )
    ols = optimization.ols_closed_form(X, y)
    st.write("OLS (closed form) weights:", dict(zip(["intercept"] + features, ols)))

    def loss_fn(w: np.ndarray) -> float:
        return optimization.linear_mse_loss(w, X, y)

    def grad_fn(w: np.ndarray) -> np.ndarray:
        return optimization.linear_mse_grad(w, X, y)

    w0_arr = np.zeros(X.shape[1])

grad_fn_eff = None
if mode == "Toy losses":
    if loss_name.startswith("Quadratic"):
        grad_fn_eff = optimization.quadratic_grad
    else:
        grad_fn_eff = optimization.rosenbrock_grad
else:
    grad_fn_eff = grad_fn

hist_sgd, final_sgd = optimization.sgd_minimize(
    loss_fn, w0_arr, lr=lr, steps=steps, grad_fn=grad_fn_eff
)
hist_adam, final_adam = optimization.adam_minimize(
    loss_fn,
    w0_arr,
    lr=lr,
    steps=steps,
    beta1=beta1,
    beta2=beta2,
    grad_fn=grad_fn_eff,
)

st.markdown("### Insights")
st.subheader("Insight")
insight_row(
    [
        ("SGD final", num(float(loss_fn(final_sgd)))),
        ("Adam final", num(float(loss_fn(final_adam)))),
        ("Steps", str(steps)),
        ("LR", num(float(lr))),
    ]
)

st.markdown("### Visuals")
fig = go.Figure()
fig.add_scatter(y=hist_sgd, name="SGD loss")
fig.add_scatter(y=hist_adam, name="Adam loss")
if mode != "Toy losses":
    fig.add_hline(y=optimization.linear_mse_loss(ols, X, y), line_dash="dot", name="OLS MSE")
apply_theme(fig, "Loss vs iteration")
st.plotly_chart(fig, use_container_width=True)
if mode != "Toy losses":
    st.write("SGD / Adam vs OLS (vector):", final_adam, ols)

with st.expander("Methodology"):
    st.markdown(
        "Toy losses and linear regression MSE use **analytic gradients** (exact ∂L/∂w). "
        "Pass `grad_fn=None` elsewhere to fall back on symmetric finite differences. "
        "Production stacks rely on reverse-mode autograd (JAX/Torch); momentum + adaptive curvature "
        "in Adam follows the same idea."
    )
