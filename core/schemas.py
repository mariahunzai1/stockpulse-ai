"""Pydantic v2 domain contracts for StockPulse AI."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model enforcing strict schema boundaries across layers."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class DailySalesRecord(StrictModel):
    """Single store-SKU daily observation from the curated Walmart M5 dataset."""

    id: str
    item_id: str
    dept_id: str
    cat_id: str
    store_id: str
    state_id: str
    d: str
    d_num: int = Field(ge=1)
    date: dt.date
    wm_yr_wk: int = Field(ge=10000)
    weekday: str
    wday: int = Field(ge=1, le=7)
    month: int = Field(ge=1, le=12)
    year: int = Field(ge=2011, le=2030)
    event_name_1: str
    event_type_1: str
    snap: int = Field(ge=0, le=1)
    sell_price: float = Field(gt=0.0)
    sales: int = Field(ge=0)


class SkuPolicyRecord(StrictModel):
    """Operational inventory and cost parameters per (store_id, item_id)."""

    store_id: str
    item_id: str
    dept_id: str
    cat_id: str
    state_id: str
    avg_sell_price: float = Field(gt=0.0)
    unit_cost: float = Field(gt=0.0)
    daily_holding_cost: float = Field(gt=0.0)
    stockout_penalty: float = Field(gt=0.0)
    lead_time_days: int = Field(ge=1)
    current_on_hand: int = Field(ge=0)
    moq: int = Field(ge=1)
    max_order_qty: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_economics(self) -> SkuPolicyRecord:
        if self.unit_cost >= self.avg_sell_price:
            raise ValueError("unit_cost must be strictly less than avg_sell_price")
        if self.max_order_qty < self.moq:
            raise ValueError("max_order_qty must be >= moq")
        return self


class StoreCategorySummary(StrictModel):
    """Analytical rollup by store and product category from DuckDB."""

    store_id: str
    state_id: str
    cat_id: str
    sku_count: int = Field(ge=1)
    row_count: int = Field(ge=1)
    total_units: int = Field(ge=0)
    avg_daily_units: float = Field(ge=0.0)
    zero_sale_rate: float = Field(ge=0.0, le=1.0)
    avg_sell_price: float = Field(gt=0.0)
    total_revenue_proxy: float = Field(ge=0.0)


class PriceDemandPoint(StrictModel):
    """Historical price and unit demand observation for elasticity views."""

    date: dt.date
    wm_yr_wk: int = Field(ge=10000)
    sell_price: float = Field(gt=0.0)
    sales: int = Field(ge=0)
    snap: int = Field(ge=0, le=1)
    event_name_1: str


class ChronologicalSplitSummary(StrictModel):
    """Summary metadata for a strict chronological train/validation split."""

    train_rows: int = Field(ge=1)
    val_rows: int = Field(ge=1)
    train_start: dt.date
    train_end: dt.date
    val_start: dt.date
    val_end: dt.date
    horizon_days: int = Field(ge=1)
    sku_series_count: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> ChronologicalSplitSummary:
        if self.train_end >= self.val_start:
            raise ValueError("Chronological split violation: train_end must precede val_start")
        return self


# --- Phase 2-4 Forecast, Explainability, Anomaly & Optimization Contracts ---


class ForecastPoint(StrictModel):
    """Probabilistic P10/P50/P90 daily demand forecast point."""

    date: dt.date
    store_id: str
    item_id: str
    actual: float | None = Field(default=None, ge=0.0)
    p10: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p90: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_quantile_order(self) -> ForecastPoint:
        if not (self.p10 <= self.p50 <= self.p90):
            raise ValueError("Quantiles must satisfy p10 <= p50 <= p90")
        return self


class ForecastMetrics(StrictModel):
    """Aggregate forecast accuracy and quantile calibration metrics."""

    wape: float = Field(ge=0.0)
    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    pinball_p10: float = Field(ge=0.0)
    pinball_p50: float = Field(ge=0.0)
    pinball_p90: float = Field(ge=0.0)
    interval_coverage_80: float = Field(ge=0.0, le=1.0)


class SkuForecastResponse(StrictModel):
    """Forecast response payload for a single store-SKU over a horizon."""

    store_id: str
    item_id: str
    horizon_days: int = Field(ge=1)
    metrics: ForecastMetrics | None = None
    points: list[ForecastPoint]


class AnomalyRecord(StrictModel):
    """Point-in-time demand anomaly detection output."""

    date: dt.date
    store_id: str
    item_id: str
    actual_sales: float = Field(ge=0.0)
    expected_p50: float = Field(ge=0.0)
    anomaly_score: float
    is_anomaly: bool
    shock_type: Literal["SURGE", "DROP", "NORMAL"]
    reason: str


class DriftReport(StrictModel):
    """Population Stability Index (PSI) and Kolmogorov-Smirnov distribution drift report."""

    psi_score: float = Field(ge=0.0)
    ks_statistic: float = Field(ge=0.0, le=1.0)
    ks_pvalue: float = Field(ge=0.0, le=1.0)
    status: Literal["STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT"]
    feature_psis: dict[str, float]


class ShapContribution(StrictModel):
    """Single feature attribution toward a P50 demand forecast."""

    feature: str
    feature_value: float
    shap_value: float


class ShapExplanation(StrictModel):
    """Additive local TreeSHAP explanation for one SKU prediction."""

    store_id: str
    item_id: str
    date: dt.date
    base_value: float
    predicted_p50: float
    contributions: list[ShapContribution]


class OptimizationConstraints(StrictModel):
    """Global budget, warehouse capacity, and What-If sensitivity constraints."""

    max_total_budget: float = Field(gt=0.0)
    max_total_units: int = Field(ge=0)
    lead_time_multiplier: float = Field(default=1.0, ge=0.5, le=3.0)
    demand_shock_pct: float = Field(default=0.0, ge=-50.0, le=150.0)
    holding_cost_multiplier: float = Field(default=1.0, ge=0.25, le=5.0)


class SkuOrderDecision(StrictModel):
    """Per-SKU optimal order recommendation vs. rule-of-thumb baseline."""

    store_id: str
    item_id: str
    dept_id: str
    current_on_hand: int = Field(ge=0)
    forecast_demand_horizon: float = Field(ge=0.0)
    safety_stock_p90: float = Field(ge=0.0)
    recommended_order_qty: int = Field(ge=0)
    order_triggered: bool
    unit_cost: float = Field(gt=0.0)
    order_cost: float = Field(ge=0.0)
    expected_holding_cost: float = Field(ge=0.0)
    expected_stockout_cost: float = Field(ge=0.0)
    baseline_order_qty: int = Field(ge=0)
    cost_savings_vs_baseline: float


class OptimizationResult(StrictModel):
    """Portfolio-wide MILP optimization summary and SKU-level order decisions."""

    constraints: OptimizationConstraints
    solver_status: str
    total_order_cost: float = Field(ge=0.0)
    total_units_ordered: int = Field(ge=0)
    total_expected_holding_cost: float = Field(ge=0.0)
    total_expected_stockout_cost: float = Field(ge=0.0)
    total_expected_cost: float = Field(ge=0.0)
    baseline_total_cost: float = Field(ge=0.0)
    total_cost_savings_vs_baseline: float
    expected_service_level: float = Field(ge=0.0, le=1.0)
    baseline_service_level: float = Field(ge=0.0, le=1.0)
    baseline_budget_violated: bool
    baseline_capacity_violated: bool
    decisions: list[SkuOrderDecision]


# --- Phase 3 Serving, Briefing & XAI Studio Contracts ---


class HealthResponse(StrictModel):
    """FastAPI service health and dataset initialization status."""

    status: Literal["ok"]
    duckdb_sales_rows: int = Field(ge=0)
    store_count: int = Field(ge=0)
    sku_count: int = Field(ge=0)
    llm_mode: Literal["llm", "deterministic_fallback"]


class SkuSummary(StrictModel):
    """Store-SKU summary combining latest price and operational inventory policy."""

    store_id: str
    state_id: str
    item_id: str
    dept_id: str
    cat_id: str
    sell_price: float = Field(gt=0.0)
    unit_cost: float = Field(gt=0.0)
    current_on_hand: int = Field(ge=0)
    lead_time_days: int = Field(ge=1)
    moq: int = Field(ge=1)
    max_order_qty: int = Field(ge=1)


class ForecastBundleResponse(StrictModel):
    """Combined trailing M5 actuals, 28-day quantile forecast, and SKU metrics."""

    store_id: str
    item_id: str
    history: list[PriceDemandPoint]
    forecast: list[ForecastPoint]
    metrics: ForecastMetrics


class ExplainResponse(StrictModel):
    """Combined local TreeSHAP waterfall and global validation feature importance."""

    local_explanation: ShapExplanation
    global_importance: list[ShapContribution]


class AnomalyAndDriftResponse(StrictModel):
    """Detected demand anomalies and population distribution drift summary."""

    anomalies: list[AnomalyRecord]
    drift_report: DriftReport


class BriefingRequest(StrictModel):
    """Input scenario for generating a grounded executive procurement briefing."""

    constraints: OptimizationConstraints
    focus_store_id: str | None = None


class ExecutiveBriefingResponse(StrictModel):
    """Grounded procurement and inventory executive briefing payload."""

    headline: str
    executive_summary: str
    priority_actions: list[str]
    risk_alerts: list[str]
    generation_mode: Literal["llm", "deterministic_fallback"]
    grounded_metrics: dict[str, Any]


class SpreadDriver(StrictModel):
    """Feature attribution comparing P90 vs. P10 tail-risk uncertainty spread."""

    feature: str
    feature_value: float
    shap_p10: float
    shap_p50: float
    shap_p90: float
    spread_impact: float


class QuantileSpreadExplanation(StrictModel):
    """Multi-quantile TreeSHAP attribution across P10, P50, and P90 models."""

    store_id: str
    item_id: str
    date: dt.date
    base_p10: float
    base_p50: float
    base_p90: float
    predicted_p10: float
    predicted_p50: float
    predicted_p90: float
    p10_contributions: list[ShapContribution]
    p50_contributions: list[ShapContribution]
    p90_contributions: list[ShapContribution]
    uncertainty_spread_drivers: list[SpreadDriver]


class IceCurve(StrictModel):
    """Individual Conditional Expectation (ICE) curve for a single store-SKU."""

    store_id: str
    item_id: str
    values: list[float]


class PdpIceResponse(StrictModel):
    """Partial Dependence and sampled ICE curves over a feature grid."""

    feature: str
    grid_values: list[float]
    average_pdp: list[float]
    ice_curves: list[IceCurve]


class CounterfactualRequest(StrictModel):
    """Request for actionable price/lead-time counterfactual search."""

    store_id: str
    item_id: str
    date: dt.date | None = None
    target_demand_delta_pct: float = Field(default=-15.0, ge=-80.0, le=100.0)
    eliminate_stockout: bool = False
    current_on_hand_override: int | None = Field(default=None, ge=0)


class CounterfactualResponse(StrictModel):
    """Minimal actionable intervention achieving or approaching target demand/stockout goal."""

    store_id: str
    item_id: str
    date: dt.date
    lever: str
    original_value: float
    counterfactual_value: float
    pct_change: float
    original_p50: float = Field(ge=0.0)
    achieved_p50: float = Field(ge=0.0)
    target_p50: float = Field(ge=0.0)
    target_achieved: bool
    stockout_eliminated: bool
    lead_time_days_original: int = Field(ge=1)
    lead_time_days_counterfactual: int = Field(ge=1)
    action_summary: str


class DecisionAuditItem(StrictModel):
    """Economic & constraint audit explaining a single SKU's MILP order decision."""

    store_id: str
    item_id: str
    dept_id: str
    recommended_order_qty: int = Field(ge=0)
    order_triggered: bool
    net_economic_benefit: float
    stockout_risk_ratio: float = Field(ge=0.0, le=1.0)
    moq_binding: bool
    budget_or_capacity_constrained: bool
    decision_rationale: str


class DecisionAuditResponse(StrictModel):
    """Portfolio-wide MILP constraint and economic decision audit."""

    solver_status: str
    total_net_economic_benefit: float
    moq_binding_count: int = Field(ge=0)
    constrained_sku_count: int = Field(ge=0)
    items: list[DecisionAuditItem]
