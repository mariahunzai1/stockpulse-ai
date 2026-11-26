"""Offline tests for FastAPI endpoints and ExecutiveBriefingGenerator (LLM + fallback)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.briefing import ExecutiveBriefingGenerator
from core.schemas import (
    AnomalyRecord,
    DriftReport,
    ForecastMetrics,
    OptimizationConstraints,
    OptimizationResult,
    SkuOrderDecision,
)


class _FakeCompletions:
    def __init__(self, content: str | None = None, should_fail: bool = False) -> None:
        self._content = content
        self._should_fail = should_fail

    def create(self, **_: Any) -> Any:
        if self._should_fail:
            raise RuntimeError("Simulated LLM endpoint outage")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))]
        )


class _FakeOpenAIClient:
    def __init__(self, content: str | None = None, should_fail: bool = False) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(content=content, should_fail=should_fail)
        )


def _sample_briefing_inputs() -> tuple[
    ForecastMetrics, DriftReport, list[AnomalyRecord], OptimizationResult
]:
    metrics = ForecastMetrics(
        wape=0.32,
        mae=3.5,
        rmse=6.4,
        pinball_p10=0.68,
        pinball_p50=1.75,
        pinball_p90=0.98,
        interval_coverage_80=0.81,
    )
    drift = DriftReport(
        psi_score=0.14,
        ks_statistic=0.04,
        ks_pvalue=0.25,
        status="MODERATE_DRIFT",
        feature_psis={"sales": 0.01, "sell_price": 0.27},
    )
    anomalies = [
        AnomalyRecord(
            date="2016-05-14",
            store_id="CA_1",
            item_id="FOODS_3_090",
            actual_sales=95.0,
            expected_p50=38.0,
            anomaly_score=0.12,
            is_anomaly=True,
            shock_type="SURGE",
            reason="Weekend SNAP surge (+3.9σ).",
        )
    ]
    constraints = OptimizationConstraints(max_total_budget=5000.0, max_total_units=2500)
    decisions = [
        SkuOrderDecision(
            store_id="CA_1",
            item_id="FOODS_3_090",
            dept_id="FOODS_3",
            current_on_hand=120,
            forecast_demand_horizon=350.0,
            safety_stock_p90=60.0,
            recommended_order_qty=280,
            order_triggered=True,
            unit_cost=0.98,
            order_cost=274.4,
            expected_holding_cost=1.2,
            expected_stockout_cost=2.1,
            baseline_order_qty=400,
            cost_savings_vs_baseline=14.5,
        )
    ]
    opt = OptimizationResult(
        constraints=constraints,
        solver_status="OPTIMAL",
        total_order_cost=274.4,
        total_units_ordered=280,
        total_expected_holding_cost=1.2,
        total_expected_stockout_cost=2.1,
        total_expected_cost=3.3,
        baseline_total_cost=17.8,
        total_cost_savings_vs_baseline=14.5,
        expected_service_level=0.99,
        baseline_service_level=0.95,
        baseline_budget_violated=False,
        baseline_capacity_violated=False,
        decisions=decisions,
    )
    return metrics, drift, anomalies, opt


def test_briefing_generator_fallback_and_mocked_llm() -> None:
    metrics, drift, anomalies, opt = _sample_briefing_inputs()

    # 1) Deterministic fallback when API key is blank/placeholder
    gen_fallback = ExecutiveBriefingGenerator(api_key="", base_url="https://api.openai.com/v1")
    res_fb = gen_fallback.generate(metrics, drift, anomalies, opt, focus_store_id="CA_1")
    assert res_fb.generation_mode == "deterministic_fallback"
    assert "CA_1" in res_fb.headline
    assert len(res_fb.priority_actions) >= 1
    assert len(res_fb.risk_alerts) >= 1
    assert res_fb.grounded_metrics["total_cost_savings_vs_baseline"] == pytest.approx(14.5)

    # 2) Mocked OpenAI-compatible client returning strict JSON -> llm mode
    fake_json = (
        '{"headline": "CA_1 Replenishment Saves $14.50",'
        '"executive_summary": "MILP achieves 99.00% service at $3.30 expected cost.",'
        '"priority_actions": ["Order 280 units of FOODS_3_090 at CA_1."],'
        '"risk_alerts": ["MODERATE_DRIFT detected (PSI=0.1400)."]}'
    )
    gen_llm = ExecutiveBriefingGenerator(
        api_key="test-key",
        client=_FakeOpenAIClient(content=fake_json),
    )
    res_llm = gen_llm.generate(metrics, drift, anomalies, opt, focus_store_id="CA_1")
    assert res_llm.generation_mode == "llm"
    assert res_llm.headline == "CA_1 Replenishment Saves $14.50"

    # 3) Mocked client error -> graceful deterministic fallback
    gen_err = ExecutiveBriefingGenerator(
        api_key="test-key",
        client=_FakeOpenAIClient(should_fail=True),
    )
    res_err = gen_err.generate(metrics, drift, anomalies, opt)
    assert res_err.generation_mode == "deterministic_fallback"


def test_fastapi_endpoints_end_to_end() -> None:
    with TestClient(app) as client:
        # /health
        r_health = client.get("/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "ok"
        assert health_data["duckdb_sales_rows"] == 21900
        assert health_data["store_count"] == 3
        assert health_data["sku_count"] == 10

        # /skus
        r_skus = client.get("/skus")
        assert r_skus.status_code == 200
        skus = r_skus.json()
        assert len(skus) == 30
        assert all(s["sell_price"] > s["unit_cost"] > 0 for s in skus)

        # /forecast valid + 404
        r_fc = client.get("/forecast/CA_1/FOODS_3_090")
        assert r_fc.status_code == 200
        fc = r_fc.json()
        assert len(fc["history"]) == 60
        assert len(fc["forecast"]) == 28
        assert fc["metrics"]["wape"] >= 0.0
        for pt in fc["forecast"]:
            assert 0.0 <= pt["p10"] <= pt["p50"] <= pt["p90"]

        assert client.get("/forecast/BAD_STORE/BAD_ITEM").status_code == 404

        # /explain default latest date + explicit date + 404
        r_exp = client.get("/explain/CA_1/FOODS_3_090?date=2016-05-10")
        assert r_exp.status_code == 200
        exp = r_exp.json()
        local = exp["local_explanation"]
        reconstructed = local["base_value"] + sum(
            c["shap_value"] for c in local["contributions"]
        )
        assert abs(reconstructed - local["predicted_p50"]) < 1e-4
        assert len(exp["global_importance"]) == 23

        assert client.get("/explain/BAD_STORE/FOODS_3_090").status_code == 404
        assert client.get("/explain/CA_1/FOODS_3_090?date=1999-01-01").status_code == 404

        # /anomalies
        r_anom = client.get("/anomalies")
        assert r_anom.status_code == 200
        anom_data = r_anom.json()
        assert anom_data["drift_report"]["psi_score"] >= 0.0
        assert all(a["is_anomaly"] is True for a in anom_data["anomalies"])

        # /optimize with tight What-If constraints
        payload = {
            "max_total_budget": 1500.0,
            "max_total_units": 600,
            "lead_time_multiplier": 1.25,
            "demand_shock_pct": 25.0,
            "holding_cost_multiplier": 1.1,
        }
        r_opt = client.post("/optimize", json=payload)
        assert r_opt.status_code == 200
        opt_data = r_opt.json()
        assert opt_data["solver_status"] == "OPTIMAL"
        assert opt_data["total_order_cost"] <= 1500.0 + 1e-6
        assert opt_data["total_units_ordered"] <= 600
        assert len(opt_data["decisions"]) == 30

        # /briefing
        r_brief = client.post(
            "/briefing",
            json={"constraints": payload, "focus_store_id": "CA_1"},
        )
        assert r_brief.status_code == 200
        brief_data = r_brief.json()
        assert brief_data["generation_mode"] in {"llm", "deterministic_fallback"}
        assert len(brief_data["priority_actions"]) >= 1
        assert brief_data["grounded_metrics"]["focus_store_id"] == "CA_1"
