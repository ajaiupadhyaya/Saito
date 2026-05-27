# Resources Integration Plan

This document maps priority items from `resources.md` into the Saito roadmap so the project evolves beyond a static dashboard.

## Phase 1 (implemented now)

- API-first routing with premium and free market providers:
  - `polygon`, `alphavantage`, `tiingo`, `finnhub`, `yfinance`
- Macro provider with FRED API support.
- Alt-data sentiment ingestion using NewsAPI.
- Shared provenance metadata in the data hub (`provider`, `policy`, `latency`, coverage).
- Research Studio modules:
  - Factor and risk diagnostics
  - Options and volatility snapshot
  - Macro spread monitor
  - Strategy sandbox
  - News sentiment panel

## Phase 2 (implemented in this cycle)

- Portfolio optimization and risk stack:
  - Efficient frontier / constraints / HRP / risk parity
  - VaR and CVaR stress testing
  - Rolling regime segmentation
- Derivatives depth:
  - IV term structure and skew history
  - scenario Greeks and surface diagnostics
- Factor research:
  - cross-sectional factors and exposure decomposition
  - factor IC / turnover / crowding diagnostics

## Phase 3 (implemented in this cycle)

- Time-series modeling and forecasting:
  - ARCH/GARCH and forecast comparison workflows
  - feature extraction / stationarity diagnostics
- Event and macro research:
  - event-study and abnormal return windows
  - economic release overlays and regime shifts
- Strategy validation:
  - walk-forward and overfitting checks
  - performance attribution and cost sensitivity

## Resources Coverage Matrix

- Numerical/data stack: covered (`numpy`, `pandas`, `scipy`) and extensible.
- Market data sources: covered with `yfinance` + premium API routing.
- Portfolio/risk analytics: partially covered, expanded in Phase 2.
- Time-series / econometrics: partially covered, expanded in Phase 3.
- Alternative data / sentiment: started with NewsAPI, extendable to additional providers.
- Backtesting and strategy research: started, to be expanded with robust validation loops.

## Sprint Coverage Gate (required per sprint)

| Sprint | Focus | Resources buckets mapped | Status |
| --- | --- | --- | --- |
| Sprint 1 | Data/API infrastructure | Market data APIs, macro APIs, alt sentiment APIs | Completed |
| Sprint 2 | Risk/portfolio + derivatives | Portfolio optimization, risk analytics, derivatives/vol models | Completed |
| Sprint 3 | Macro events + strategy validation | Event studies, econometrics workflows, backtesting validation | Completed |
| Sprint 4+ | Advanced expansion | Time series ML, additional alt-data, execution/microstructure | Planned |

## Additional APIs worth adding later

- IEX / Tiingo news / Polygon options references for richer options and fundamentals.
- Institutional-grade macro calendars and filings feeds for event studies.
- Exchange calendars (`exchange_calendars`) for robust market-session aware modeling.
