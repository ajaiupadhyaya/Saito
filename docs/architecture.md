# Saito Quant Terminal Architecture

## Purpose

This document defines how data flows through Saito from external APIs to user-facing quantitative workflows, and how reliability, reproducibility, and explainability are enforced.

## Core Layers

1. **Provider Layer** (`src/data/providers`)
   - External API adapters (Polygon, AlphaVantage, Tiingo, Finnhub, yfinance, FRED, NewsAPI).
   - Handles source-specific response parsing and normalization.
2. **Routing Layer** (`src/data/providers/factory.py`, `src/data/hub_service.py`)
   - Selects providers according to policy (`auto`, `premium-first`, `free-only`).
   - Runs retries, fallback, cooldown, and attempt logging.
3. **Session + Provenance Layer** (`src/data/session.py`)
   - Stores active dataset and provenance metadata:
     - request inputs,
     - provider attempt chain,
     - transform lineage,
     - load timestamps.
4. **Quant Engine Layer** (`src/metrics`)
   - Pure analytics modules:
     - portfolio/risk,
     - derivatives/volatility,
     - event studies,
     - strategy validation.
5. **Product/UX Layer** (`pages`, `src/ui`)
   - Labs and studio pages with narrative cards, diagnostics, and export-oriented presentation.

## Reliability Contract

- Provider calls must be deterministic in schema even if source quality differs.
- Every load captures:
  - selected policy,
  - attempted providers and latency,
  - chosen provider and coverage stats.
- Failing providers enter short cooldown to prevent repeated quota/timeouts from blocking flow.

## Reproducibility Contract

Every research run should be reconstructible from:

- dataset provenance object,
- parameter controls visible in page state,
- metric outputs and exported tables.

## Module Contracts

- `portfolio.py`: optimization and risk decomposition outputs as arrays/series/dataframes.
- `derivatives.py`: option quality/term/skew and scenario surface tables.
- `event_study.py`: event date sets, AR series, CAR panel.
- `validation.py`: walk-forward split outputs and overfit gap diagnostics.

## Testing Strategy

- Provider contract tests: parse/normalization/fallback behavior.
- Metric unit tests: numerical sanity, shape, bounds.
- Integration tests: provider -> returns -> metrics -> UI-ready frames.
- Regression fixtures: known synthetic datasets for stable expected behaviors.

## Extension Path

- Add econometrics and forecasting modules in `src/metrics` without changing provider/session contracts.
- Add new APIs by implementing provider adapters and registering in factory capability matrix.
