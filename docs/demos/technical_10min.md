# 10-Minute Technical Demo

## Goal
Demonstrate architecture, quantitative rigor, and validation workflow.

## Sequence

1. **Architecture pass (2 min)**  
   Walk through `docs/architecture.md` and provider routing policy.
2. **Risk/Portfolio pass (3 min)**  
   Open `14_Risk_Portfolio_Lab`:
   - frontier,
   - HRP vs risk parity,
   - factor and stress diagnostics.
3. **Derivatives pass (2 min)**  
   Open `15_Derivatives_Vol_Lab`:
   - term structure,
   - skew,
   - scenario surface.
4. **Macro + validation pass (3 min)**  
   Open `16_Macro_Event_Study` and `13_Research_Studio` strategy tab:
   - CAR around shocks,
   - walk-forward split metrics.

## Talking Points

- Separation of data routing, quant metrics, and UI layers.
- Provenance-first design for reproducibility.
- Tests and linting as non-optional quality gates.
