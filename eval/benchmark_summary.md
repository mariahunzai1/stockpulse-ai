# StockPulse AI — Walmart M5 Backtesting Benchmark

**Evaluation Window:** `2016-04-25` to `2016-05-22` (28 days × 30 store-SKU series = 840 points)

## 1. Demand Forecasting Benchmark (28-Day Autoregressive Rollout)

| Model | WAPE | MAE | RMSE | Pinball P10 | Pinball P50 | Pinball P90 | 80% PI Coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7-Day Seasonal Naive | 0.3875 | 4.283 | 7.894 | 0.8846 | 2.1417 | 1.0480 | 72.38% |
| **LightGBM Quantile (P10/P50/P90)** | **0.3222** | **3.561** | **6.411** | **0.6889** | **1.7805** | **0.9803** | **80.60%** |

- **WAPE Improvement vs. 7-Day Seasonal Naive:** `+16.86%`

## 2. Inventory Policy Benchmark (HiGHS MILP vs. 14-Day Rule-of-Thumb)

| Scenario | MILP Cost ($) | Baseline Cost ($) | Savings ($) | Cost Reduction | MILP Service | Baseline Service | Baseline Cap Violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Nominal | $46.92 | $284.00 | $237.08 | +83.48% | 100.00% | 95.77% | No |
| +30% Demand Surge | $62.37 | $538.23 | $475.86 | +88.41% | 99.99% | 91.51% | No |
| +50% Lead-Time Delay | $67.08 | $470.21 | $403.13 | +85.73% | 99.99% | 92.61% | No |
| Tight Working-Capital Cap | $274.23 | $284.00 | $9.77 | +3.44% | 90.34% | 95.77% | Yes |
