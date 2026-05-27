# Derivatives and Volatility Methodology

## Objective
Move from static option snapshots to interpretable volatility structure and scenario analytics.

## Methods

- **Option chain quality scoring**
  - missing IV ratio,
  - zero open-interest ratio,
  - wide relative spread ratio.
- **IV skew estimator**
  Difference between upper and lower strike quantile IV medians.
- **IV term structure**
  Median/mean implied volatility across expiries.
- **Scenario Greeks/PnL surface**
  Black-Scholes call analytics under spot/volatility shock grid.

## Inputs

- Live option chains (currently yfinance path in advanced lab).
- Expiry selection and scenario assumptions (spot, vol, rate, maturity).

## Key Assumptions

- Black-Scholes used as baseline diagnostic model, not full market microstructure model.
- Spot and implied volatility shocks are independent knobs for scenario exploration.
- Chain quality metrics proxy tradeability risk, not execution certainty.

## Limitations

- Surface interpolation is deliberately simple in current phase.
- Some providers may have stale/partial chain fields.
- No full local/stochastic volatility calibration loop in this stage.
