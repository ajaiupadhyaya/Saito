# Saito

Home for strategies, models, algorithms, and tools — open source and free for all.

## Section 1 — Mathematical & statistical foundations (Streamlit)

Interactive lab: **topic guide → data hub → explain + controls + Plotly**.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (recommended: `curl -LsSf https://astral.sh/uv/install.sh | sh`).

### Bootstrap with uv

From the repo root:

```bash
cd /path/to/Saito
uv sync --group dev
uv run python scripts/generate_datasets.py
uv run streamlit run app.py
```

Daily commands:

```bash
uv sync --group dev      # deps + dev tools (pytest, ruff)
uv run pytest
uv run ruff check src tests pages app.py
```

The project installs as an editable wheel so `src.*` imports work under `uv run`.

### API-first research studio

New page: **`13_Research_Studio.py`**.

- Unified API hub with provider fallback and provenance metadata
- Provider policies: `auto`, `premium-first`, `free-only`
- Factor + risk diagnostics (rolling beta, VaR/CVaR, regime correlation)
- Options/vol snapshot from live option chains
- Macro engine using FRED public series
- Strategy studio with fast signal backtest and risk stats
- Alt-data sentiment panel from NewsAPI

Premium adapters (env-driven): `polygon`, `alphavantage`, `tiingo`, `finnhub`, `newsapi`, plus `fred`.
See `docs/resources_integration_plan.md` for the broader roadmap aligned to `resources.md`.

Additional advanced labs:

- `14_Risk_Portfolio_Lab.py` — constrained frontier, risk parity, HRP, factor exposure, rolling stress VaR.
- `15_Derivatives_Vol_Lab.py` — IV term structure, smile/skew diagnostics, option-chain quality, scenario Greeks/PnL surface.
- `16_Macro_Event_Study.py` — macro shock detection and CAR-style event windows.
- `17_Econometrics_Regime_Lab.py` — AR diagnostics, Ljung-Box, and volatility regime tagging.

Documentation bundle:

- `docs/architecture.md` — system architecture and data contracts.
- `docs/methodology/` — methodology notes by research pillar.
- `docs/demos/` — recruiter, technical, and deep-dive demo scripts.

### Topics (mirror `list.md` §1)

| Track | Pages |
| --- | --- |
| **Foundations** | Descriptive stats, covariance / correlation (+ Ledoit–Wolf & blend shrinkage) |
| **Linear algebra** | PCA (standardized optional), biplot, SVD + fast error curve, eigen decomposition |
| **Optimization** | Toy + **linear regression MSE on your panel**, constrained Markowitz / KKT hints |
| **Probability** | QQ plots, standardized CLT, joint density, count models |
| **Stochastic calc** | Brownian / GBM / OU, Itô vs naive drift |
| **Tails & dependence** | EVT, copulas, RMT |

See **`pages/00_Topic_Guide.py`** inside the app for the narrative syllabus.

### Data sources

Load in **Data Hub**:

- Yahoo Finance (`yfinance`, cached ~1 h per ticker set + date range via Streamlit `@st.cache_data`)
- CSV upload
- Synthetic presets (corr, OU, fat tails, crash correlation…)
- Bundled CSVs (`datasets/`)

### Disclaimer

Education and research only — **not investment advice**.
