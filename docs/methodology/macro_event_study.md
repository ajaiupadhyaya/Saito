# Macro Event Study Methodology

## Objective
Quantify how abnormal asset behavior aligns with macro shocks and event windows.

## Methods

- **Shock detection**  
  Z-score thresholding on first differences of macro series.
- **Abnormal return model**  
  Linear market model residuals (asset vs benchmark).
- **CAR windows**  
  Cumulative abnormal return around event dates with configurable pre/post horizon.

## Inputs

- Macro time series (FRED endpoint).
- Asset and benchmark return series from routed market providers.
- Shock threshold and event window controls.

## Key Assumptions

- Event relevance is proxied by macro series shock events.
- Benchmark captures broad market movement for abnormal return isolation.
- Event windows are short-horizon explanatory diagnostics, not full causal proof.

## Limitations

- No scheduled-calendar semantic labeling yet.
- Overlapping events can blend attribution.
- Model is linear and can underfit nonlinear macro-asset transmission.
