"""Reproducible Walmart M5 forecasting & MILP inventory backtesting benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.features import build_temporal_features, split_chronological
from core.forecaster import QuantileForecaster
from core.optimizer import InventoryOptimizer
from core.schemas import ForecastPoint, OptimizationConstraints
from core.warehouse import StockPulseWarehouse


def _build_seasonal_naive_predictions(
    history_raw: pd.DataFrame,
    future_raw: pd.DataFrame,
    horizon_days: int = 28,
) -> list[ForecastPoint]:
    """Generate 7-day seasonal naive forecasts with empirical 28-day quantile spread."""
    points: list[ForecastPoint] = []
    for (store_id, item_id), fut_grp in future_raw.groupby(["store_id", "item_id"], sort=True):
        hist_grp = history_raw[
            (history_raw["store_id"] == store_id) & (history_raw["item_id"] == item_id)
        ].sort_values("d_num")
        tail_28 = hist_grp["sales"].to_numpy(dtype=np.float64)[-28:]
        tail_7 = tail_28[-7:]
        std_7 = float(np.std(tail_28, ddof=1)) if len(tail_28) > 1 else 0.0

        fut_sorted = fut_grp.sort_values("d_num").iloc[:horizon_days]
        for step, (_, row) in enumerate(fut_sorted.iterrows()):
            p50 = max(0.0, float(tail_7[step % 7]))
            p10 = max(0.0, p50 - 1.2816 * std_7)
            p90 = max(p50, p50 + 1.2816 * std_7)
            points.append(
                ForecastPoint(
                    date=pd.Timestamp(row["date"]).date(),
                    store_id=str(store_id),
                    item_id=str(item_id),
                    actual=float(row["sales"]),
                    p10=round(p10, 4),
                    p50=round(p50, 4),
                    p90=round(p90, 4),
                )
            )
    return points


def _category_wape_breakdown(
    points: list[ForecastPoint],
    item_to_cat: dict[str, str],
) -> dict[str, float]:
    buckets: dict[str, list[ForecastPoint]] = {}
    for pt in points:
        cat = item_to_cat[pt.item_id]
        buckets.setdefault(cat, []).append(pt)
    return {
        cat: round(QuantileForecaster.evaluate_predictions(pts).wape, 6)
        for cat, pts in sorted(buckets.items())
    }


def run_benchmark(output_dir: Path | None = None) -> dict[str, Any]:
    """Execute full forecasting and inventory optimization benchmark and save artifacts."""
    settings = get_settings()
    out_path = output_dir or (PROJECT_ROOT / "eval")
    out_path.mkdir(parents=True, exist_ok=True)

    with StockPulseWarehouse() as wh:
        wh.ensure_initialized()
        raw_sales = wh.load_sales_dataframe()
        policies = wh.list_sku_policies()

    item_to_cat = {p.item_id: p.cat_id for p in policies}
    featured = build_temporal_features(raw_sales, drop_na_lags=True)
    split = split_chronological(featured, horizon_days=settings.forecast_horizon_days)

    val_start = pd.Timestamp(split.summary.val_start)
    history_raw = raw_sales[pd.to_datetime(raw_sales["date"]) < val_start].copy()
    future_raw = raw_sales[pd.to_datetime(raw_sales["date"]) >= val_start].copy()

    forecaster = QuantileForecaster.load_or_train(train_df=split.train_df)
    lgbm_points = forecaster.predict_horizon(
        history_df=history_raw,
        future_calendar_df=future_raw,
        horizon_days=settings.forecast_horizon_days,
    )
    naive_points = _build_seasonal_naive_predictions(
        history_raw=history_raw,
        future_raw=future_raw,
        horizon_days=settings.forecast_horizon_days,
    )

    lgbm_metrics = QuantileForecaster.evaluate_predictions(lgbm_points)
    naive_metrics = QuantileForecaster.evaluate_predictions(naive_points)
    wape_improvement_pct = round(
        ((naive_metrics.wape - lgbm_metrics.wape) / max(naive_metrics.wape, 1e-9)) * 100.0,
        2,
    )

    trailing_means = (
        history_raw.sort_values("d_num")
        .groupby(["store_id", "item_id"])["sales"]
        .apply(lambda s: float(s.tail(28).mean()))
        .to_dict()
    )
    trailing_map = {(str(k[0]), str(k[1])): float(v) for k, v in trailing_means.items()}

    optimizer = InventoryOptimizer()
    scenarios = [
        (
            "Nominal",
            OptimizationConstraints(max_total_budget=50_000.0, max_total_units=20_000),
        ),
        (
            "+30% Demand Surge",
            OptimizationConstraints(
                max_total_budget=50_000.0,
                max_total_units=20_000,
                demand_shock_pct=30.0,
            ),
        ),
        (
            "+50% Lead-Time Delay",
            OptimizationConstraints(
                max_total_budget=50_000.0,
                max_total_units=20_000,
                lead_time_multiplier=1.5,
            ),
        ),
        (
            "Tight Working-Capital Cap",
            OptimizationConstraints(max_total_budget=2_000.0, max_total_units=900),
        ),
    ]

    inventory_scenarios: list[dict[str, Any]] = []
    for name, constraints in scenarios:
        res = optimizer.optimize(
            policies=policies,
            forecast_points=lgbm_points,
            constraints=constraints,
            trailing_28d_means=trailing_map,
        )
        cost_reduction_pct = round(
            (res.total_cost_savings_vs_baseline / max(res.baseline_total_cost, 1e-9)) * 100.0,
            2,
        )
        inventory_scenarios.append(
            {
                "scenario": name,
                "constraints": constraints.model_dump(),
                "solver_status": res.solver_status,
                "milp_order_cost": res.total_order_cost,
                "milp_units_ordered": res.total_units_ordered,
                "milp_expected_cost": res.total_expected_cost,
                "baseline_expected_cost": res.baseline_total_cost,
                "cost_savings": res.total_cost_savings_vs_baseline,
                "cost_reduction_pct": cost_reduction_pct,
                "milp_service_level": res.expected_service_level,
                "baseline_service_level": res.baseline_service_level,
                "baseline_budget_violated": res.baseline_budget_violated,
                "baseline_capacity_violated": res.baseline_capacity_violated,
            }
        )

    results = {
        "dataset": {
            "stores": list(settings.selected_stores),
            "sku_series": 30,
            "train_rows": split.summary.train_rows,
            "val_rows": split.summary.val_rows,
            "val_window": f"{split.summary.val_start} to {split.summary.val_end}",
        },
        "forecasting": {
            "seasonal_naive_7d": naive_metrics.model_dump(),
            "lightgbm_quantile": lgbm_metrics.model_dump(),
            "wape_improvement_pct": wape_improvement_pct,
            "category_wape_lightgbm": _category_wape_breakdown(lgbm_points, item_to_cat),
            "category_wape_naive": _category_wape_breakdown(naive_points, item_to_cat),
        },
        "inventory_optimization": inventory_scenarios,
    }

    json_path = out_path / "benchmark_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# StockPulse AI — Walmart M5 Backtesting Benchmark",
        "",
        f"**Evaluation Window:** `{split.summary.val_start}` to `{split.summary.val_end}` (28 days × 30 store-SKU series = 840 points)",
        "",
        "## 1. Demand Forecasting Benchmark (28-Day Autoregressive Rollout)",
        "",
        "| Model | WAPE | MAE | RMSE | Pinball P10 | Pinball P50 | Pinball P90 | 80% PI Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| 7-Day Seasonal Naive | {naive_metrics.wape:.4f} | {naive_metrics.mae:.3f} | "
            f"{naive_metrics.rmse:.3f} | {naive_metrics.pinball_p10:.4f} | "
            f"{naive_metrics.pinball_p50:.4f} | {naive_metrics.pinball_p90:.4f} | "
            f"{naive_metrics.interval_coverage_80:.2%} |"
        ),
        (
            f"| **LightGBM Quantile (P10/P50/P90)** | **{lgbm_metrics.wape:.4f}** | "
            f"**{lgbm_metrics.mae:.3f}** | **{lgbm_metrics.rmse:.3f}** | "
            f"**{lgbm_metrics.pinball_p10:.4f}** | **{lgbm_metrics.pinball_p50:.4f}** | "
            f"**{lgbm_metrics.pinball_p90:.4f}** | **{lgbm_metrics.interval_coverage_80:.2%}** |"
        ),
        "",
        f"- **WAPE Improvement vs. 7-Day Seasonal Naive:** `{wape_improvement_pct:+.2f}%`",
        "",
        "## 2. Inventory Policy Benchmark (HiGHS MILP vs. 14-Day Rule-of-Thumb)",
        "",
        "| Scenario | MILP Cost ($) | Baseline Cost ($) | Savings ($) | Cost Reduction | MILP Service | Baseline Service | Baseline Cap Violation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in inventory_scenarios:
        violated = "Yes" if (row["baseline_budget_violated"] or row["baseline_capacity_violated"]) else "No"
        md_lines.append(
            f"| {row['scenario']} | ${row['milp_expected_cost']:,.2f} | "
            f"${row['baseline_expected_cost']:,.2f} | ${row['cost_savings']:,.2f} | "
            f"{row['cost_reduction_pct']:+.2f}% | {row['milp_service_level']:.2%} | "
            f"{row['baseline_service_level']:.2%} | {violated} |"
        )

    md_path = out_path / "benchmark_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    summary = run_benchmark()
    print(json.dumps(summary["forecasting"], indent=2))
    print(f"Saved benchmark artifacts to {PROJECT_ROOT / 'eval'}")
