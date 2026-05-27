"""Reusable Streamlit patterns for metric pages."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.data.session import get_context

TOPICS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "API-first research modules",
        [
            ("pages/13_Research_Studio.py", "Research Studio (factor risk, options vol, macro, strategy)"),
            ("pages/14_Risk_Portfolio_Lab.py", "Risk & Portfolio Lab (frontier, HRP, stress)"),
            ("pages/15_Derivatives_Vol_Lab.py", "Derivatives & Vol Lab (term structure, skew, scenario)"),
            ("pages/16_Macro_Event_Study.py", "Macro Event Study (shock detection and CAR)"),
            ("pages/17_Econometrics_Regime_Lab.py", "Econometrics & Regime Lab (AR, Ljung-Box, vol regimes)"),
        ],
    ),
    (
        "Foundations (descriptive)",
        [
            ("pages/02_Descriptive_Stats.py", "Descriptive statistics"),
            ("pages/04_Covariance_and_Correlation.py", "Covariance & correlation"),
        ],
    ),
    (
        "Linear algebra — SVD, PCA & eigen",
        [
            ("pages/05_PCA_SVD_Eigen.py", "PCA, SVD & eigendecomposition"),
        ],
    ),
    (
        "Calculus & optimization — SGD, Adam & constraints",
        [
            ("pages/06_Optimization_SGD_Adam.py", "SGD & Adam"),
            ("pages/07_Constrained_Optimization.py", "Constrained optimization (Markowitz/KKT)"),
        ],
    ),
    (
        "Probability — distributions & limits",
        [
            ("pages/03_Distributions_and_CLT.py", "Distributions & CLT"),
        ],
    ),
    (
        "Stochastic calculus — diffusion & Itô",
        [
            ("pages/08_Brownian_GBM_OU.py", "Brownian motion, GBM & OU"),
            ("pages/09_Ito_Lemma_Demo.py", "Itô lemma demo"),
        ],
    ),
    (
        "Tail risk & dependence — EVT, copulas & RMT",
        [
            ("pages/10_Fat_Tails_EVT.py", "Fat tails & EVT"),
            ("pages/11_Copulas.py", "Copulas"),
            ("pages/12_Random_Matrix_Theory.py", "Random matrix theory"),
        ],
    ),
]

EXPLAIN: dict[str, dict[str, str]] = {
    "descriptive": {
        "what": "Mean, variance, skewness, and kurtosis summarize the center, dispersion, and tail shape of returns.",
        "why": "Regime detection, risk budgeting, and sanity checks before factor models or optimization.",
        "formula": (
            r"\mu = \mathbb{E}[R],\ \sigma^2 = \mathrm{Var}(R),\ "
            r"\gamma_1 = \mathbb{E}\left[\left(\frac{R-\mu}{\sigma}\right)^3\right]"
        ),
    },
    "distributions": {
        "what": "Fit parametric laws (Normal, Student-t, Binomial, Poisson) and demonstrate the Central Limit Theorem.",
        "why": "Risk models that assume normality often underestimate tail risk; count models apply to trade outcomes.",
        "formula": r"\bar{X}_n \xrightarrow{d} \mathcal{N}(\mu,\sigma^2/n)\ \text{as } n\to\infty",
    },
    "covariance": {
        "what": "Covariance and correlation describe linear co-movement across assets.",
        "why": "Portfolio risk, hedging ratios, and eigenstructure preview for PCA/RMT.",
        "formula": r"\rho_{ij} = \frac{\mathrm{Cov}(R_i,R_j)}{\sigma_i\sigma_j}",
    },
    "pca": {
        "what": "PCA projects correlated returns onto orthogonal components maximizing variance.",
        "why": "Factor reduction, collinearity removal, and latent structure in cross-sections.",
        "formula": r"\Sigma = V \Lambda V^\top,\ \text{scores} = X V_k",
    },
    "optimization_sgd": {
        "what": (
            "SGD and Adam are iterative first-order optimizers for non-convex and convex objectives."
        ),
        "why": "Core for model fitting when closed-form solutions are unavailable or too expensive.",
        "formula": r"m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\ v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2",
    },
    "optimization_kkt": {
        "what": (
            "Markowitz optimization solves constrained risk minimization "
            "using Lagrange multipliers / KKT conditions."
        ),
        "why": "Portfolio construction depends on budget, target-return, and long-only feasibility constraints.",
        "formula": r"\min_w\ w^\top\Sigma w\ \ \text{s.t.}\ \mathbf{1}^\top w=1,\ \mu^\top w=r",
    },
    "stochastic": {
        "what": "Brownian motion, GBM, and OU SDEs model diffusion, prices, and mean-reverting spreads.",
        "why": "Option pricing, pairs trading, and rate/spread modeling; Euler–Maruyama integrates generic SDEs.",
        "formula": r"dS_t = \mu S_t\,dt + \sigma S_t\,dW_t;\quad dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t",
    },
    "ito": {
        "what": "Itô's lemma gives correct drift for functions of diffusions (log S, S², etc.).",
        "why": "Naive calculus omits convexity terms; wrong drift biases simulated log returns and payoffs.",
        "formula": r"d\ln S = \left(\mu - \tfrac{1}{2}\sigma^2\right)dt + \sigma\,dW",
    },
    "evt": {
        "what": "Peaks-over-threshold fits tail exceedances with the Generalized Pareto Distribution (GPD).",
        "why": "VaR/ES for rare losses; Hill estimator and return levels quantify tail index and extremal quantiles.",
        "formula": r"F(x) = 1 - \left(1 + \xi\frac{x-u}{\beta}\right)^{-1/\xi}",
    },
    "copulas": {
        "what": "Copulas (Gaussian, Student-t, Archimedean) separate margins from dependence.",
        "why": "Tail dependence λ_L, λ_U captures crash co-movement that linear ρ misses.",
        "formula": r"C(u,v) = \Phi_\rho(\Phi^{-1}(u),\Phi^{-1}(v));\ \lambda_L = \lim_{q\to 0} P(U\leq q \mid V\leq q)",
    },
    "rmt": {
        "what": "Marčenko–Pastur bulk separates noise eigenvalues; denoising shrinks the bulk before portfolio use.",
        "why": "Large N/T correlation matrices are mostly noise; cleaning reduces spurious diversification.",
        "formula": r"\rho(\lambda) = \frac{1}{2\pi\sigma^2 q\lambda}\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}",
    },
}


def render_explain(key: str) -> None:
    block = EXPLAIN.get(key, {})
    st.subheader("Explain")
    if block.get("what"):
        st.markdown(f"**What:** {block['what']}")
    if block.get("why"):
        st.markdown(f"**Why in finance:** {block['why']}")
    if block.get("formula"):
        st.latex(block["formula"])
    st.caption("Maps to repo `list.md` — Section 1. Use **00_Topic_Guide** for the learning path.")


def render_metric_page_shell(*, title: str, explain_key: str, showcase_key: str | None = None) -> None:
    """Consistent page shell for metric demos."""
    st.session_state["saito_active_topic"] = showcase_key
    if showcase_key and not get_context().ready:
        from src.ui.lab import load_default_topic_scenario

        ok, info = load_default_topic_scenario(showcase_key)
        if ok:
            st.info(f"Loaded built-in demo dataset for this page: {info}")

    render_data_sidebar()
    st.title(title)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.page_link("pages/00_Topic_Guide.py", label="Topic Guide", icon="🧭")
    with c2:
        st.page_link("pages/01_Data_Hub.py", label="Data Hub", icon="🗂️")
    render_explain(explain_key)
    if showcase_key:
        from src.ui.lab import render_showcase

        render_showcase(showcase_key)
    st.markdown("### Controls")


def render_data_sidebar() -> None:
    """Compact dataset context on every metric page (sidebar)."""
    ctx = get_context()
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Dataset**")
        if ctx.ready:
            st.caption(f"{ctx.n_assets} series · {ctx.n_obs} rows · {ctx.return_type} returns")
            src = str(ctx.source)[:48] + ("…" if len(str(ctx.source)) > 48 else "")
            st.caption(f"Source: `{src}`")
            st.caption("Change data in **01_Data_Hub**.")
        else:
            st.warning("No data loaded.")
            st.caption("Open **01_Data_Hub** in the sidebar.")
        st.markdown("---")


def require_data(min_assets: int = 1, min_obs: int = 30) -> bool:
    ctx = get_context()
    topic = st.session_state.get("saito_active_topic")

    if not ctx.ready and topic:
        from src.ui.lab import load_default_topic_scenario

        ok, _info = load_default_topic_scenario(topic)
        if ok:
            ctx = get_context()

    if not ctx.ready:
        st.warning("Load data on **Data Hub** first (sidebar or Data Hub page).")
        return False
    if ctx.n_obs < min_obs:
        st.warning(f"Need at least {min_obs} observations; currently {ctx.n_obs}.")
        if topic:
            if st.button("Load recommended built-in dataset for this page", key=f"reload_obs_{topic}"):
                from src.ui.lab import load_default_topic_scenario

                ok, info = load_default_topic_scenario(topic)
                if ok:
                    st.success(f"Loaded built-in dataset: {info}")
                    st.rerun()
        return False
    if ctx.n_assets < min_assets:
        st.warning(f"Need at least {min_assets} asset(s); currently {ctx.n_assets}.")
        if topic:
            if st.button("Load recommended built-in dataset for this page", key=f"reload_assets_{topic}"):
                from src.ui.lab import load_default_topic_scenario

                ok, info = load_default_topic_scenario(topic)
                if ok:
                    st.success(f"Loaded built-in dataset: {info}")
                    st.rerun()
        return False
    return True


def download_df(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        df.to_csv().encode(),
        file_name=filename,
        mime="text/csv",
    )


def download_json(label: str, obj: dict, filename: str) -> None:
    st.download_button(
        label,
        json.dumps(obj, indent=2, default=str).encode(),
        file_name=filename,
        mime="application/json",
    )
