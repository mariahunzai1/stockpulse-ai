"""Offline tests for ExplainableAIStudio and FastAPI /xai/* endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_xai_studio_endpoints_end_to_end() -> None:
    with TestClient(app) as client:
        # 1) Multi-quantile tail-risk TreeSHAP attribution & exact additivity across P10/P50/P90
        r_tail = client.get("/xai/tail-risk/CA_1/FOODS_3_090?date=2016-05-10")
        assert r_tail.status_code == 200
        tail = r_tail.json()

        p10_sum = tail["base_p10"] + sum(c["shap_value"] for c in tail["p10_contributions"])
        p50_sum = tail["base_p50"] + sum(c["shap_value"] for c in tail["p50_contributions"])
        p90_sum = tail["base_p90"] + sum(c["shap_value"] for c in tail["p90_contributions"])
        assert abs(p10_sum - tail["predicted_p10"]) < 1e-4
        assert abs(p50_sum - tail["predicted_p50"]) < 1e-4
        assert abs(p90_sum - tail["predicted_p90"]) < 1e-4

        drivers = tail["uncertainty_spread_drivers"]
        assert len(drivers) == 23
        for drv in drivers:
            assert abs((drv["shap_p90"] - drv["shap_p10"]) - drv["spread_impact"]) < 1e-9
        abs_spreads = [abs(d["spread_impact"]) for d in drivers]
        assert abs_spreads == sorted(abs_spreads, reverse=True)

        # 2) Partial Dependence & ICE curves (+ 400 on invalid feature)
        r_pdp = client.get("/xai/pdp/sell_price")
        assert r_pdp.status_code == 200
        pdp = r_pdp.json()
        assert pdp["feature"] == "sell_price"
        assert len(pdp["grid_values"]) >= 2
        assert len(pdp["average_pdp"]) == len(pdp["grid_values"])
        assert len(pdp["ice_curves"]) == 12
        assert all(len(c["values"]) == len(pdp["grid_values"]) for c in pdp["ice_curves"])

        assert client.get("/xai/pdp/not_a_valid_feature").status_code == 400

        # 3) Actionable Counterfactual Search (demand reduction + stockout elimination)
        r_cf_down = client.post(
            "/xai/counterfactual",
            json={
                "store_id": "CA_1",
                "item_id": "FOODS_3_090",
                "date": "2016-05-10",
                "target_demand_delta_pct": -15.0,
                "eliminate_stockout": False,
            },
        )
        assert r_cf_down.status_code == 200
        cf_down = r_cf_down.json()
        assert -25.0 <= cf_down["pct_change"] <= 25.0
        assert cf_down["achieved_p50"] <= cf_down["original_p50"] + 1e-6
        assert len(cf_down["action_summary"]) > 10

        r_cf_stockout = client.post(
            "/xai/counterfactual",
            json={
                "store_id": "TX_1",
                "item_id": "HOUSEHOLD_1_110",
                "date": "2016-05-10",
                "eliminate_stockout": True,
                "current_on_hand_override": 4,
            },
        )
        assert r_cf_stockout.status_code == 200
        cf_so = r_cf_stockout.json()
        assert cf_so["lead_time_days_counterfactual"] <= cf_so["lead_time_days_original"]

        # 4) MILP Decision & Constraint Attribution Audit under tight budget cap
        r_audit = client.post(
            "/xai/decision-audit",
            json={
                "max_total_budget": 1500.0,
                "max_total_units": 600,
                "lead_time_multiplier": 1.0,
                "demand_shock_pct": 0.0,
                "holding_cost_multiplier": 1.0,
            },
        )
        assert r_audit.status_code == 200
        audit = r_audit.json()
        assert audit["solver_status"] == "OPTIMAL"
        assert len(audit["items"]) == 30
        assert audit["constrained_sku_count"] >= 1
        assert all(len(item["decision_rationale"]) > 10 for item in audit["items"])
