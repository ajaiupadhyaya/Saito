# Deep-Dive Quant Demo

## Goal
Run a full research narrative from data ingestion to decision-quality diagnostics.

## Flow

1. Use `API router` with selected policy and load a multi-asset panel.
2. Validate data source quality and provenance.
3. Perform allocation diagnostics:
   - constrained frontier,
   - HRP and risk parity comparison.
4. Run derivatives diagnostics on representative ticker:
   - chain quality,
   - IV structure,
   - scenario surface.
5. Run macro event study:
   - shock extraction,
   - CAR path around events.
6. Evaluate strategy robustness:
   - transaction-cost sensitivity,
   - walk-forward splits,
   - overfit gap narrative.

## Deliverables to Show

- Exported tables and JSON provenance.
- Methodology docs from `docs/methodology`.
- Updated resources coverage map.
