import plotly.express as px
import streamlit as st

from src.data.session import get_context
from src.metrics import descriptive
from src.ui.lab import insight_row, num, pct
from src.ui.page_helpers import download_df, render_metric_page_shell, require_data
from src.viz.plots import apply_theme

st.set_page_config(page_title="Descriptive Stats | Saito", layout="wide")
render_metric_page_shell(title="Descriptive Statistics", explain_key="descriptive", showcase_key="descriptive")

if not require_data():
    st.stop()

ctx = get_context()
asset = st.selectbox("Asset", list(ctx.returns.columns))
window = st.slider("Rolling window (days)", 20, 120, 60)
ddof_var = int(
    st.radio(
        "Variance / std normalization (ddof)",
        options=[("Sample (divides by N−1)", 1), ("Population (divides by N)", 0)],
        format_func=lambda x: x[0],
        horizontal=True,
    )[1]
)

summary = descriptive.summary_table(ctx.returns.dropna(how="all"), ddof_var=ddof_var)
st.markdown("### Results")
st.subheader("Cross-sectional summary")
st.dataframe(summary, use_container_width=True)
download_df("Download summary CSV", summary, "descriptive_summary.csv")

series = ctx.returns[asset].dropna()
roll = descriptive.rolling_moments(series, window)

st.markdown("### Insights")
st.subheader("Insight")
mean_ann = float(series.mean() * 252)
vol_ann = float(series.std() * (252**0.5))
insight_row(
    [
        ("μ (ann.)", pct(mean_ann)),
        ("σ (ann.)", pct(vol_ann)),
        ("skew", num(float(series.skew()))),
        ("excess kurt", num(float(series.kurtosis()))),
    ]
)

tab_r, tab_h = st.tabs(["Rolling traces", "Return histogram"])
st.markdown("### Visuals")
with tab_r:
    fig = px.line(roll, title=f"Rolling moments — {asset}")
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)
with tab_h:
    fig2 = px.histogram(series, nbins=50, marginal="rug", title=f"Return distribution — {asset}")
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("Methodology"):
    st.markdown(
        "Rolling moments highlight **non-stationarity**. "
        "Cross‑sectional rows pool all dates per asset. "
        "Skew/excess kurtosis follow pandas defaults; variance switches with **ddof**."
    )
