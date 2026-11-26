"""Offline tests for Phase 2 forecaster, anomaly/drift detector, SHAP, and MILP optimizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.anomaly import AnomalyAndDriftDetector
from core.config import get_settings
from core.explainer import DemandExplainer
from core.features import FEATURE_COLUMNS, build_temporal_features, split_chronological
from core.forecaster import QuantileForecaster
from core.optimizer import InventoryOptimizer
from core.schemas import OptimizationConstraints
from core.warehouse import StockPulseWarehouse


@pytest.fixture(scope="module")
def phase2_context(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build deterministic warehouse, chronological split, and trained forecaster once per module."""
    tmp_dir: Path = tmp_path_factory.mktemp("phase2")
    settings = get_settings()

    with StockPulseWarehouse(db_path=tmp_dir / "phase2.duckdb") as wh:
        raw_sales = wh.load_sales_dataframe()
        policies = wh.list_sku_policies()

    featured = build_temporal_features(raw_sales, drop_na_lags=True)
    split = split_chronological(featured, horizon_days=settings.forecast_horizon_days)

    val_start = pd.Timestamp(split.summary.val_start)
    history_raw = raw_sales[pd.to_datetime(raw_sales["date"]) < val_start].copy()
    future_raw = raw_sales[pd.to_datetime(raw_sales["date"]) >= val_start].copy()

    bundle_path = tmp_dir / "lgbm_quantile_bundle.joblib"
    forecaster = QuantileForecaster.load_or_train(
        train_df=split.train_df,
        bundle_path=bundle_path,
    )
    val_points = forecaster.predict_horizon(
        history_df=history_raw,
        future_calendar_df=future_raw,
        horizon_days=settings.forecast_horizon_days,
    )

    trailing_means = (
        history_raw.sort_values("d_num")
        .groupby(["store_id", "item_id"])["sales"]
        .apply(lambda s: float(s.tail(28).mean()))
        .to_dict()
    )

    return {
        "tmp_dir": tmp_dir,
        "bundle_path": bundle_path,
        "raw_sales": raw_sales,
        "policies": policies,
        "featured": featured,
        "split": split,
        "history_raw": history_raw,
        "future_raw": future_raw,
        "forecaster": forecaster,
        "val_points": val_points,
        "trailing_means": trailing_means,
    }


def test_quantile_forecaster_recursive_horizon_and_persistence(
    phase2_context: dict[str, object],
) -> None:
    forecaster: QuantileForecaster = phase2_context["forecaster"]  # type: ignore[assignment]
    bundle_path: Path = phase2_context["bundle_path"]  # type: ignore[assignment]
    val_points = phase2_context["val_points"]
    split = phase2_context["split"]

    assert bundle_path.exists()
    assert len(val_points) == 30 * 28

    for pt in val_points:
        assert 0.0 <= pt.p10 <= pt.p50 <= pt.p90
        assert pt.actual is not None

    metrics = QuantileForecaster.evaluate_predictions(val_points)
    assert metrics.wape >= 0.0
    assert metrics.mae >= 0.0
    assert metrics.rmse >= metrics.mae
    assert 0.40 <= metrics.interval_coverage_80 <= 0.99

    # Reload via load_or_train and confirm bit-for-bit identical predictions
    reloaded = QuantileForecaster.load_or_train(bundle_path=bundle_path)
    p1 = forecaster.predict_quantiles(split.val_df.head(50))
    p2 = reloaded.predict_quantiles(split.val_df.head(50))
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def test_anomaly_detector_and_psi_drift(phase2_context: dict[str, object]) -> None:
    forecaster: QuantileForecaster = phase2_context["forecaster"]  # type: ignore[assignment]
    split = phase2_context["split"]

    train_p50 = forecaster.predict_quantiles(split.train_df)[:, 1]
    detector = AnomalyAndDriftDetector(contamination=0.04, random_state=42).fit(
        split.train_df, train_p50
    )

    # Inject synthetic SURGE and DROP into a copy of validation rows
    val_probe = split.val_df.copy()
    val_p50 = forecaster.predict_quantiles(val_probe)[:, 1]

    # Row 0: high-velocity SKU or any SKU -> inject extreme 6x + 40 surge
    val_probe.loc[val_probe.index[0], "sales"] = int(max(50, round(val_p50[0] * 6 + 40)))
    # Pick a row with high expected P50 (> 5 units) and collapse actual sales to 0
    high_idx_candidates = np.where(val_p50 > 5.0)[0]
    assert len(high_idx_candidates) > 0
    drop_pos = int(high_idx_candidates[0])
    if drop_pos == 0 and len(high_idx_candidates) > 1:
        drop_pos = int(high_idx_candidates[1])
    val_probe.loc[val_probe.index[drop_pos], "sales"] = 0
    val_probe.loc[val_probe.index[drop_pos], "rolling_mean_28"] = max(
        float(val_probe.loc[val_probe.index[drop_pos], "rolling_mean_28"]), 25.0
    )
    val_p50[drop_pos] = max(val_p50[drop_pos], 25.0)

    records = detector.detect(val_probe, val_p50)
    assert len(records) == len(val_probe)
    assert records[0].is_anomaly is True
    assert records[0].shock_type == "SURGE"
    assert "surge" in records[0].reason.lower()

    assert records[drop_pos].is_anomaly is True
    assert records[drop_pos].shock_type == "DROP"
    assert "drop" in records[drop_pos].reason.lower()

    # PSI & KS Drift checks
    stable_report = AnomalyAndDriftDetector.compute_drift_report(split.train_df, split.train_df)
    assert stable_report.status == "STABLE"
    assert stable_report.psi_score == pytest.approx(0.0, abs=1e-6)
    assert stable_report.ks_pvalue == pytest.approx(1.0, abs=1e-6)

    shifted_df = split.val_df.copy()
    for col in ("sales", "sell_price", "rolling_mean_7", "rolling_mean_28"):
        shifted_df[col] = shifted_df[col] * 5.0 + 10.0
    shifted_df["price_momentum_7d"] = 0.75
    shifted_df["rolling_zero_rate_28"] = 0.99

    drift_report = AnomalyAndDriftDetector.compute_drift_report(split.train_df, shifted_df)
    assert drift_report.status == "SIGNIFICANT_DRIFT"
    assert drift_report.psi_score >= 0.25
    assert drift_report.ks_pvalue < 0.01


def test_shap_explainer_additivity_and_global_importance(
    phase2_context: dict[str, object],
) -> None:
    forecaster: QuantileForecaster = phase2_context["forecaster"]  # type: ignore[assignment]
    split = phase2_context["split"]

    explainer = DemandExplainer(forecaster)
    sample_row = split.val_df.iloc[0]
    explanation = explainer.explain_prediction(
        featured_df=split.val_df,
        store_id=str(sample_row["store_id"]),
        item_id=str(sample_row["item_id"]),
        target_date=pd.Timestamp(sample_row["date"]).date(),
    )

    assert len(explanation.contributions) == len(FEATURE_COLUMNS)
    reconstructed = explanation.base_value + sum(
        c.shap_value for c in explanation.contributions
    )
    assert abs(reconstructed - explanation.predicted_p50) < 1e-4

    # Contributions must be ordered by descending |shap_value|
    abs_vals = [abs(c.shap_value) for c in explanation.contributions]
    assert abs_vals == sorted(abs_vals, reverse=True)

    importance = explainer.global_feature_importance(split.val_df)
    assert set(importance.keys()) == set(FEATURE_COLUMNS)
    imp_vals = list(importance.values())
    assert all(v >= 0.0 for v in imp_vals)
    assert imp_vals == sorted(imp_vals, reverse=True)


def test_milp_optimizer_constraints_whatif_and_baseline_savings(
    phase2_context: dict[str, object],
) -> None:
    policies = phase2_context["policies"]
    val_points = phase2_context["val_points"]
    trailing_means = phase2_context["trailing_means"]
    optimizer = InventoryOptimizer()

    # 1) Generous caps: MILP should outperform or match rule-of-thumb baseline & honor MOQ
    loose_constraints = OptimizationConstraints(
        max_total_budget=50_000.0,
        max_total_units=20_000,
    )
    res_loose = optimizer.optimize(
        policies=policies,  # type: ignore[arg-type]
        forecast_points=val_points,  # type: ignore[arg-type]
        constraints=loose_constraints,
        trailing_28d_means=trailing_means,  # type: ignore[arg-type]
    )
    assert res_loose.solver_status == "OPTIMAL"
    assert len(res_loose.decisions) == 30
    assert res_loose.total_order_cost <= loose_constraints.max_total_budget + 1e-6
    assert res_loose.total_units_ordered <= loose_constraints.max_total_units
    assert res_loose.total_expected_cost <= res_loose.baseline_total_cost + 1e-6
    assert res_loose.expected_service_level >= 0.85

    pol_map = {(p.store_id, p.item_id): p for p in policies}  # type: ignore[attr-defined]
    for dec in res_loose.decisions:
        pol = pol_map[(dec.store_id, dec.item_id)]
        assert isinstance(dec.recommended_order_qty, int)
        if dec.order_triggered:
            assert pol.moq <= dec.recommended_order_qty <= pol.max_order_qty
        else:
            assert dec.recommended_order_qty == 0

    # 2) Tight budget/capacity caps: MILP strictly respects caps while flagging baseline violations
    tight_constraints = OptimizationConstraints(
        max_total_budget=1_500.0,
        max_total_units=600,
    )
    res_tight = optimizer.optimize(
        policies=policies,  # type: ignore[arg-type]
        forecast_points=val_points,  # type: ignore[arg-type]
        constraints=tight_constraints,
        trailing_28d_means=trailing_means,  # type: ignore[arg-type]
    )
    assert res_tight.solver_status == "OPTIMAL"
    assert res_tight.total_order_cost <= tight_constraints.max_total_budget + 1e-6
    assert res_tight.total_units_ordered <= tight_constraints.max_total_units
    assert res_tight.baseline_budget_violated or res_tight.baseline_capacity_violated

    # 3) What-If demand surge + longer lead time increases horizon demand & safety stock
    shock_constraints = OptimizationConstraints(
        max_total_budget=50_000.0,
        max_total_units=20_000,
        lead_time_multiplier=1.5,
        demand_shock_pct=35.0,
        holding_cost_multiplier=1.2,
    )
    res_shock = optimizer.optimize(
        policies=policies,  # type: ignore[arg-type]
        forecast_points=val_points,  # type: ignore[arg-type]
        constraints=shock_constraints,
        trailing_28d_means=trailing_means,  # type: ignore[arg-type]
    )
    base_horizon_sum = sum(d.forecast_demand_horizon for d in res_loose.decisions)
    shock_horizon_sum = sum(d.forecast_demand_horizon for d in res_shock.decisions)
    assert shock_horizon_sum > base_horizon_sum
