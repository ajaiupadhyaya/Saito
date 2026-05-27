"""Saito — Section 1 Mathematical & Statistical Foundations."""

import streamlit as st

from src.data.session import ensure_session, get_context
from src.ui.lab import load_default_topic_scenario
from src.ui.page_helpers import TOPICS

st.set_page_config(
    page_title="Saito Quant Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session()
ctx = get_context()

st.title("Saito Quant Terminal")
st.caption("Section 1: Mathematical & Statistical Foundations — interactive, explainable metrics")

st.markdown(
    """
1. Open **`00_Topic_Guide`** for the recommended path (mirrors *list.md* groupings).
2. Optional: load your own data in **`01_Data_Hub`** (yfinance/CSV/synthetic/bundled).
3. Each topic page shows **Explain → controls → charts** and repeats dataset status in the sidebar.
"""
)

if st.button("Start in Demo Mode (no upload needed)", type="primary"):
    ok, info = load_default_topic_scenario("descriptive")
    if ok:
        st.success(f"Demo dataset ready: {info}. Open any topic page to start.")
    else:
        st.error(info)

status = "Ready" if ctx.ready else "No data loaded"
if ctx.ready:
    st.info(f"Data status: **{status}** — {ctx.n_assets} assets, {ctx.n_obs} obs ({ctx.source})")
else:
    st.info(f"Data status: **{status}**")

st.subheader("Metric map (by topic)")
for title, rows in TOPICS:
    st.markdown(f"**{title}**")
    for _path, label in rows:
        st.markdown(f"- {label}")

EXTRA = [
    ("Data Hub", "Load prices / returns driving every widget"),
]

st.subheader("Platform")
for name, desc in EXTRA:
    st.markdown(f"- **{name}** — {desc}")

st.divider()
st.caption("Development workflow: see README for `uv` commands. Not investment advice.")
