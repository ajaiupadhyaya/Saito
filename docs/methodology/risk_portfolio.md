# Risk and Portfolio Methodology

## Objective
Translate return panels into robust allocation and risk diagnostics with transparent assumptions.

## Methods

- **Constrained efficient frontier**  
  Long-only optimization with budget and target-return constraints.
- **Risk parity weights**  
  Iterative scaling toward equal risk contribution across assets.
- **HRP proxy allocation**  
  Hierarchical clustering of correlation structure and inverse-variance weighting baseline.
- **Factor exposure decomposition**  
  OLS regression of target asset returns on selected factor proxy returns.
- **Rolling stress VaR**  
  Historical rolling-window VaR to capture regime changes in tail risk.

## Inputs

- Synchronized return panel from active session dataset.
- User-selected target asset and optional factor proxy set.
- Window controls for rolling and frontier granularity.

## Key Assumptions

- Returns are sufficiently stationary within local rolling windows.
- Factor proxies are not perfect structural factors; they are practical tradable approximations.
- Long-only frontier reflects implementation-focused constraints.

## Limitations

- No transaction-cost-aware rebalancing in frontier/HRP outputs yet.
- Factor decomposition uses linear model and may miss nonlinear exposures.
- VaR is historical and should be interpreted with EVT/copula diagnostics for severe tails.
