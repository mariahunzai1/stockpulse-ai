"""UI process-isolation AST checks, syntax compilation, and HTTP client/chart smoke tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from api.main import app
from ui.api_client import StockPulseApiClient
from ui.charts import (
    build_forecast_fan_chart,
    build_milp_comparison_charts,
    build_pdp_ice_chart,
    build_psi_drift_chart,
    build_quantile_spread_chart,
    build_shap_waterfall_bar,
    build_store_category_bar,
)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def test_ui_has_zero_core_or_api_imports() -> None:
    """Enforce strict architectural boundary: ui/ must never import core/ or api/."""
    py_files = sorted(UI_DIR.glob("*.py"))
    assert len(py_files) >= 3

    forbidden_roots = {"core", "api"}
    for path in py_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        compile(tree, filename=str(path), mode="exec")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    assert root_pkg not in forbidden_roots, (
                        f"Forbidden import '{alias.name}' found in {path.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_pkg = node.module.split(".")[0]
                assert root_pkg not in forbidden_roots, (
                    f"Forbidden import from '{node.module}' found in {path.name}"
                )


class _TestClientTransport(httpx.BaseTransport):
    """Bridge httpx.Client calls inside StockPulseApiClient to FastAPI TestClient."""

    def __init__(self, tc: TestClient) -> None:
        self._tc = tc

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.raw_path.decode("ascii")
        resp = self._tc.request(
            method=request.method,
            url=path,
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(
            status_code=resp.status_code,
            headers=resp.headers,
            content=resp.content,
            request=request,
        )


def test_api_client_and_plotly_charts_end_to_end() -> None:
    """Verify StockPulseApiClient and all Plotly chart builders against live FastAPI payloads."""
    with TestClient(app) as tc:
        client = StockPulseApiClient(
            base_url="http://testserver",
            transport=_TestClientTransport(tc),
        )
        assert client.ensure_backend_running() is True

        health = client.get_health()
        assert health["status"] == "ok"

        skus = client.get_skus()
        assert len(skus) == 30
        assert build_store_category_bar(skus).data

        forecast = client.get_forecast("CA_1", "FOODS_3_090")
        assert len(forecast["forecast"]) == 28
        assert build_forecast_fan_chart(forecast).data

        explain = client.get_explain("CA_1", "FOODS_3_090", "2016-05-10")
        assert build_shap_waterfall_bar(
            explain["local_explanation"]["contributions"], "Local SHAP"
        ).data

        tail_risk = client.get_xai_tail_risk("CA_1", "FOODS_3_090", "2016-05-10")
        assert build_quantile_spread_chart(tail_risk).data

        pdp = client.get_xai_pdp("sell_price")
        assert build_pdp_ice_chart(pdp).data

        cf = client.post_xai_counterfactual(
            {
                "store_id": "CA_1",
                "item_id": "FOODS_3_090",
                "date": "2016-05-10",
                "target_demand_delta_pct": -15.0,
                "eliminate_stockout": False,
            }
        )
        assert "action_summary" in cf

        anomalies = client.get_anomalies()
        assert build_psi_drift_chart(anomalies["drift_report"]).data

        constraints = {
            "max_total_budget": 15000.0,
            "max_total_units": 10000,
            "lead_time_multiplier": 1.0,
            "demand_shock_pct": 0.0,
            "holding_cost_multiplier": 1.0,
        }
        opt = client.post_optimize(constraints)
        cost_fig, gauge_fig = build_milp_comparison_charts(opt)
        assert cost_fig.data and gauge_fig.data

        audit = client.post_xai_decision_audit(constraints)
        assert len(audit["items"]) == 30

        brief = client.post_briefing({"constraints": constraints, "focus_store_id": None})
        assert json.loads(json.dumps(brief))["headline"]
