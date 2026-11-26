"""FastAPI backend exposing StockPulse AI forecasting, XAI, anomaly, optimization & briefing."""

from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from core.anomaly import AnomalyAndDriftDetector
from core.briefing import ExecutiveBriefingGenerator
from core.config import get_settings
from core.explainer import DemandExplainer
from core.features import build_temporal_features, split_chronological
from core.forecaster import QuantileForecaster
from core.optimizer import InventoryOptimizer
from core.schemas import (
    AnomalyAndDriftResponse,
    AnomalyRecord,
    BriefingRequest,
    CounterfactualRequest,
    CounterfactualResponse,
    DecisionAuditResponse,
    DriftReport,
    ExecutiveBriefingResponse,
    ExplainResponse,
    ForecastBundleResponse,
    ForecastMetrics,
    ForecastPoint,
    HealthResponse,
    OptimizationConstraints,
    OptimizationResult,
    PdpIceResponse,
    PriceDemandPoint,
    QuantileSpreadExplanation,
    ShapContribution,
    SkuPolicyRecord,
    SkuSummary,
)
from core.warehouse import StockPulseWarehouse
from core.xai_studio import ExplainableAIStudio


@dataclass
class ServiceContext:
    """In-memory runtime state initialized once during FastAPI lifespan."""

    warehouse: StockPulseWarehouse
    sales_rows_count: int
    raw_sales: pd.DataFrame
    history_raw: pd.DataFrame
    featured_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    policies: list[SkuPolicyRecord]
    policy_map: dict[tuple[str, str], SkuPolicyRecord]
    latest_prices: dict[tuple[str, str], float]
    trailing_28d_means: dict[tuple[str, str], float]
    forecaster: QuantileForecaster
    val_points: list[ForecastPoint]
    val_metrics: ForecastMetrics
    detector: AnomalyAndDriftDetector
    anomalies: list[AnomalyRecord]
    drift_report: DriftReport
    explainer: DemandExplainer
    global_importance: list[ShapContribution]
    xai_studio: ExplainableAIStudio
    optimizer: InventoryOptimizer
    briefing_gen: ExecutiveBriefingGenerator


def _build_service_context() -> ServiceContext:
    settings = get_settings()
    wh = StockPulseWarehouse()
    sales_count, _ = wh.ensure_initialized()

    raw_sales = wh.load_sales_dataframe()
    policies = wh.list_sku_policies()
    policy_map = {(p.store_id, p.item_id): p for p in policies}

    latest_prices_series = (
        raw_sales.sort_values("d_num")
        .groupby(["store_id", "item_id"])["sell_price"]
        .last()
    )
    latest_prices = {
        (str(k[0]), str(k[1])): float(v) for k, v in latest_prices_series.items()
    }

    featured_df = build_temporal_features(raw_sales, drop_na_lags=True)
    split = split_chronological(featured_df, horizon_days=settings.forecast_horizon_days)

    val_start = pd.Timestamp(split.summary.val_start)
    history_raw = raw_sales[pd.to_datetime(raw_sales["date"]) < val_start].copy()
    future_raw = raw_sales[pd.to_datetime(raw_sales["date"]) >= val_start].copy()

    trailing_means_series = (
        history_raw.sort_values("d_num")
        .groupby(["store_id", "item_id"])["sales"]
        .apply(lambda s: float(s.tail(28).mean()))
    )
    trailing_28d_means = {
        (str(k[0]), str(k[1])): float(v) for k, v in trailing_means_series.items()
    }

    forecaster = QuantileForecaster.load_or_train(train_df=split.train_df)
    val_points = forecaster.predict_horizon(
        history_df=history_raw,
        future_calendar_df=future_raw,
        horizon_days=settings.forecast_horizon_days,
    )
    val_metrics = QuantileForecaster.evaluate_predictions(val_points)

    train_p50 = forecaster.predict_quantiles(split.train_df)[:, 1]
    val_p50 = forecaster.predict_quantiles(split.val_df)[:, 1]
    detector = AnomalyAndDriftDetector(
        contamination=0.04, random_state=settings.random_seed
    ).fit(split.train_df, train_p50)
    anomalies = detector.detect(split.val_df, val_p50)
    drift_report = AnomalyAndDriftDetector.compute_drift_report(
        split.train_df, split.val_df
    )

    explainer = DemandExplainer(forecaster)
    global_imp_dict = explainer.global_feature_importance(split.val_df)
    val_feature_means = split.val_df[list(forecaster.feature_cols)].mean().to_dict()
    global_importance = [
        ShapContribution(
            feature=feat,
            feature_value=round(float(val_feature_means.get(feat, 0.0)), 6),
            shap_value=score,
        )
        for feat, score in global_imp_dict.items()
    ]

    return ServiceContext(
        warehouse=wh,
        sales_rows_count=sales_count,
        raw_sales=raw_sales,
        history_raw=history_raw,
        featured_df=featured_df,
        train_df=split.train_df,
        val_df=split.val_df,
        policies=policies,
        policy_map=policy_map,
        latest_prices=latest_prices,
        trailing_28d_means=trailing_28d_means,
        forecaster=forecaster,
        val_points=val_points,
        val_metrics=val_metrics,
        detector=detector,
        anomalies=anomalies,
        drift_report=drift_report,
        explainer=explainer,
        global_importance=global_importance,
        xai_studio=ExplainableAIStudio(forecaster),
        optimizer=InventoryOptimizer(),
        briefing_gen=ExecutiveBriefingGenerator(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize warehouse, ML bundles, and XAI studio on startup and clean up on exit."""
    ctx = _build_service_context()
    app.state.ctx = ctx
    try:
        yield
    finally:
        ctx.warehouse.close()


app = FastAPI(
    title="StockPulse AI API",
    version="0.3.0",
    description=(
        "Probabilistic Walmart M5 Demand Forecasting, XAI Studio, Anomaly/Drift "
        "Monitoring & HiGHS MILP Inventory Optimization Service."
    ),
    lifespan=lifespan,
)


def _get_ctx() -> ServiceContext:
    ctx: ServiceContext | None = getattr(app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=503, detail="Service context is not initialized")
    return ctx


def _validate_sku(ctx: ServiceContext, store_id: str, item_id: str) -> SkuPolicyRecord:
    key = (store_id, item_id)
    if key not in ctx.policy_map:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown store/SKU pair: store_id={store_id}, item_id={item_id}",
        )
    return ctx.policy_map[key]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service readiness, DuckDB row count, store/SKU counts, and LLM mode."""
    ctx = _get_ctx()
    llm_mode = "llm" if ctx.briefing_gen.is_llm_configured else "deterministic_fallback"
    stores = {p.store_id for p in ctx.policies}
    items = {p.item_id for p in ctx.policies}
    return HealthResponse(
        status="ok",
        duckdb_sales_rows=ctx.sales_rows_count,
        store_count=len(stores),
        sku_count=len(items),
        llm_mode=llm_mode,
    )


@app.get("/skus", response_model=list[SkuSummary])
def list_skus() -> list[SkuSummary]:
    """Return all 30 curated store-SKU policies with latest observed M5 selling prices."""
    ctx = _get_ctx()
    return [
        SkuSummary(
            store_id=p.store_id,
            state_id=p.state_id,
            item_id=p.item_id,
            dept_id=p.dept_id,
            cat_id=p.cat_id,
            sell_price=round(ctx.latest_prices.get((p.store_id, p.item_id), p.avg_sell_price), 4),
            unit_cost=p.unit_cost,
            current_on_hand=p.current_on_hand,
            lead_time_days=p.lead_time_days,
            moq=p.moq,
            max_order_qty=p.max_order_qty,
        )
        for p in ctx.policies
    ]


@app.get("/forecast/{store_id}/{item_id}", response_model=ForecastBundleResponse)
def get_forecast(store_id: str, item_id: str) -> ForecastBundleResponse:
    """Return trailing 60-day actual M5 history and 28-day P10/P50/P90 forecast vs actuals."""
    ctx = _get_ctx()
    _validate_sku(ctx, store_id, item_id)

    hist_df = (
        ctx.history_raw[
            (ctx.history_raw["store_id"] == store_id)
            & (ctx.history_raw["item_id"] == item_id)
        ]
        .sort_values("d_num")
        .tail(60)
    )
    history_points = [
        PriceDemandPoint(
            date=pd.Timestamp(row["date"]).date(),
            wm_yr_wk=int(row["wm_yr_wk"]),
            sell_price=float(row["sell_price"]),
            sales=int(row["sales"]),
            snap=int(row["snap"]),
            event_name_1=str(row["event_name_1"]),
        )
        for row in hist_df.to_dict(orient="records")
    ]

    sku_forecast = [
        pt for pt in ctx.val_points if pt.store_id == store_id and pt.item_id == item_id
    ]
    sku_metrics = QuantileForecaster.evaluate_predictions(sku_forecast)

    return ForecastBundleResponse(
        store_id=store_id,
        item_id=item_id,
        history=history_points,
        forecast=sku_forecast,
        metrics=sku_metrics,
    )


@app.get("/explain/{store_id}/{item_id}", response_model=ExplainResponse)
def explain_sku(
    store_id: str,
    item_id: str,
    date: dt.date | None = Query(default=None, description="Validation date YYYY-MM-DD"),
) -> ExplainResponse:
    """Return local TreeSHAP waterfall attributions and global validation SHAP importance."""
    ctx = _get_ctx()
    _validate_sku(ctx, store_id, item_id)

    target_date = date or pd.Timestamp(ctx.val_df["date"].max()).date()
    try:
        local_exp = ctx.explainer.explain_prediction(
            ctx.val_df, store_id=store_id, item_id=item_id, target_date=target_date
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ExplainResponse(
        local_explanation=local_exp,
        global_importance=ctx.global_importance,
    )


@app.get("/anomalies", response_model=AnomalyAndDriftResponse)
def get_anomalies(
    include_normal: bool = Query(default=False, description="Include non-anomalous rows"),
) -> AnomalyAndDriftResponse:
    """Return IsolationForest demand anomalies and validation-vs-train PSI/KS drift report."""
    ctx = _get_ctx()
    filtered = (
        ctx.anomalies
        if include_normal
        else [a for a in ctx.anomalies if a.is_anomaly]
    )
    ordered = sorted(filtered, key=lambda a: a.anomaly_score, reverse=True)
    return AnomalyAndDriftResponse(anomalies=ordered, drift_report=ctx.drift_report)


@app.post("/optimize", response_model=OptimizationResult)
def run_optimization(constraints: OptimizationConstraints) -> OptimizationResult:
    """Solve budget- and capacity-constrained MILP replenishment across all 30 store-SKUs."""
    ctx = _get_ctx()
    return ctx.optimizer.optimize(
        policies=ctx.policies,
        forecast_points=ctx.val_points,
        constraints=constraints,
        trailing_28d_means=ctx.trailing_28d_means,
    )


@app.post("/briefing", response_model=ExecutiveBriefingResponse)
def generate_briefing(request: BriefingRequest) -> ExecutiveBriefingResponse:
    """Run constrained optimization and return a grounded executive procurement briefing."""
    ctx = _get_ctx()
    if request.focus_store_id is not None:
        valid_stores = {p.store_id for p in ctx.policies}
        if request.focus_store_id not in valid_stores:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown focus_store_id={request.focus_store_id}",
            )

    opt_result = ctx.optimizer.optimize(
        policies=ctx.policies,
        forecast_points=ctx.val_points,
        constraints=request.constraints,
        trailing_28d_means=ctx.trailing_28d_means,
    )
    return ctx.briefing_gen.generate(
        forecast_metrics=ctx.val_metrics,
        drift_report=ctx.drift_report,
        anomalies=ctx.anomalies,
        optimization_result=opt_result,
        focus_store_id=request.focus_store_id,
    )


# --- Explainable AI (XAI) Studio Endpoints ---


@app.get("/xai/tail-risk/{store_id}/{item_id}", response_model=QuantileSpreadExplanation)
def xai_tail_risk(
    store_id: str,
    item_id: str,
    date: dt.date | None = Query(default=None, description="Validation date YYYY-MM-DD"),
) -> QuantileSpreadExplanation:
    """Return multi-quantile P10/P50/P90 TreeSHAP attributions and uncertainty spread drivers."""
    ctx = _get_ctx()
    _validate_sku(ctx, store_id, item_id)
    try:
        return ctx.xai_studio.explain_quantile_spread(
            ctx.val_df, store_id=store_id, item_id=item_id, target_date=date
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/xai/pdp/{feature_name}", response_model=PdpIceResponse)
def xai_pdp_ice(feature_name: str) -> PdpIceResponse:
    """Return Partial Dependence and sampled SKU ICE curves for a supported feature."""
    ctx = _get_ctx()
    try:
        return ctx.xai_studio.compute_pdp_ice(ctx.val_df, feature_name=feature_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/xai/counterfactual", response_model=CounterfactualResponse)
def xai_counterfactual(request: CounterfactualRequest) -> CounterfactualResponse:
    """Search minimal bounded price/lead-time intervention for a target demand or stockout goal."""
    ctx = _get_ctx()
    _validate_sku(ctx, request.store_id, request.item_id)
    try:
        return ctx.xai_studio.generate_counterfactual(
            ctx.val_df, ctx.policies, request=request
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/xai/decision-audit", response_model=DecisionAuditResponse)
def xai_decision_audit(constraints: OptimizationConstraints) -> DecisionAuditResponse:
    """Solve MILP under `constraints` and return structured per-SKU economic & constraint audit."""
    ctx = _get_ctx()
    opt_result = ctx.optimizer.optimize(
        policies=ctx.policies,
        forecast_points=ctx.val_points,
        constraints=constraints,
        trailing_28d_means=ctx.trailing_28d_means,
    )
    return ctx.xai_studio.explain_optimization_decisions(opt_result, ctx.policies)
