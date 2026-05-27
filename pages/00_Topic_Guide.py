"""Learning path grouped like `list.md` Section 1 (basics → advanced)."""

import streamlit as st

from src.data.session import ensure_session
from src.ui.page_helpers import TOPICS

st.set_page_config(page_title="Topic Guide | Saito", layout="wide", initial_sidebar_state="expanded")
ensure_session()

st.title("Topic guide — Section 1")
st.markdown(
    """
Use this page as the syllabus. Sidebar pages are prefixed so the **reading order matches how quants stack topics**:
descriptive intuition → correlation structure → **linear algebra (PCA/SVD)** → **optimization**
→ probability & CLT → stochastic calculus → tails & dependence.

1. **`01_Data_Hub`** — load tickers, CSV, bundled CSVs, or synthetic paths (nothing else works without data here).
"""
)

for title, rows in TOPICS:
    st.subheader(title)
    for _path, label in rows:
        st.markdown(f"- **{label}**")

st.divider()
st.markdown(
    """
**Recommended flow — foundations**

| Step | Why |
|------|-----|
| Data Hub → synthetic `correlated_normal` | Instantly validates multi‑asset PCA & correlation |
| `05_*` PCA / SVD | See factors before touching portfolio weights |
| `06_*` SGD vs Adam → `07_*` constrained frontier | Optimization story arc |
| `03_*` returns + QQ + standardized CLT | Probability foundations on the same strip you trade |

**Recommended flow — advanced (list.md §1 PhD level)**

| Step | Preset / CSV | Why |
|------|--------------|-----|
| `08_*` Brownian / GBM / OU | `ou_spread` or `pair_spread_ou.csv` | SDEs + mean reversion on a spread |
| `09_*` Itô lemma | (no data required) | Convexity correction on log S and S² |
| `10_*` EVT / GPD | `fat_t_student` or `tail_events.csv` | Tail VaR, Hill index, return levels |
| `11_*` Copulas | `crash_correlation` | Tail dependence vs Gaussian ρ |
| `12_*` RMT | `corr_noise_50x50.csv` | MP bulk, denoised corr, portfolio vol impact |
"""
)
st.caption("Each metric page repeats dataset status directly in the sidebar.")
