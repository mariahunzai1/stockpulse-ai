"""Grounded Executive Briefing Generator with OpenAI-compatible LLM & offline fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from core.config import get_settings
from core.schemas import (
    AnomalyRecord,
    DriftReport,
    ExecutiveBriefingResponse,
    ForecastMetrics,
    OptimizationResult,
)

PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {
        "",
        "none",
        "null",
        "placeholder",
        "changeme",
        "your-api-key",
        "your_api_key",
        "your_google_ai_studio_key",
        "sk-placeholder",
    }
)

GROUNDING_SYSTEM_PROMPT = (
    "You are StockPulse AI's Senior Supply Chain Executive Strategist. "
    "You MUST strictly ground every claim and number in the provided JSON `grounded_metrics` payload. "
    "Do NOT invent, extrapolate, or alter any currency figures, unit quantities, percentages, store IDs, or SKU IDs. "
    "Return ONLY valid JSON with exact keys: "
    "`headline` (string), `executive_summary` (string, 2-4 concise sentences), "
    "`priority_actions` (list of 3-4 actionable strings), and `risk_alerts` (list of 2-3 strings)."
)


class ExecutiveBriefingGenerator:
    """Generates executive inventory briefings via LLM or deterministic fallback."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = settings.llm_api_key if api_key is None else api_key
        self.base_url = settings.llm_base_url if base_url is None else base_url
        self.model = settings.llm_model if model is None else model
        self._injected_client = client

    @property
    def is_llm_configured(self) -> bool:
        """Return True when an injected client or non-placeholder API key is available."""
        if self._injected_client is not None:
            return True
        cleaned = self.api_key.strip().lower()
        return cleaned not in PLACEHOLDER_KEYS

    @staticmethod
    def build_grounded_metrics(
        forecast_metrics: ForecastMetrics,
        drift_report: DriftReport,
        anomalies: list[AnomalyRecord],
        optimization_result: OptimizationResult,
        focus_store_id: str | None = None,
    ) -> dict[str, Any]:
        """Assemble strictly computed metrics dictionary for grounding & auditability."""
        active_anomalies = [a for a in anomalies if a.is_anomaly]
        if focus_store_id is not None:
            scoped_anomalies = [a for a in active_anomalies if a.store_id == focus_store_id]
            scoped_decisions = [
                d for d in optimization_result.decisions if d.store_id == focus_store_id
            ]
        else:
            scoped_anomalies = active_anomalies
            scoped_decisions = optimization_result.decisions

        top_anomalies = sorted(
            scoped_anomalies, key=lambda a: a.anomaly_score, reverse=True
        )[:5]
        top_orders = sorted(
            [d for d in scoped_decisions if d.order_triggered],
            key=lambda d: d.order_cost,
            reverse=True,
        )[:5]

        return {
            "focus_store_id": focus_store_id or "ALL_STORES",
            "forecast_wape_pct": round(forecast_metrics.wape * 100.0, 2),
            "forecast_mae": round(forecast_metrics.mae, 2),
            "forecast_rmse": round(forecast_metrics.rmse, 2),
            "interval_coverage_80_pct": round(forecast_metrics.interval_coverage_80 * 100.0, 2),
            "drift_status": drift_report.status,
            "drift_psi_score": round(drift_report.psi_score, 4),
            "ks_pvalue": round(drift_report.ks_pvalue, 4),
            "anomaly_count": len(scoped_anomalies),
            "top_anomalies": [
                {
                    "date": str(a.date),
                    "store_id": a.store_id,
                    "item_id": a.item_id,
                    "shock_type": a.shock_type,
                    "actual_sales": round(a.actual_sales, 1),
                    "expected_p50": round(a.expected_p50, 1),
                }
                for a in top_anomalies
            ],
            "total_order_cost": round(optimization_result.total_order_cost, 2),
            "total_units_ordered": optimization_result.total_units_ordered,
            "total_expected_cost": round(optimization_result.total_expected_cost, 2),
            "baseline_total_cost": round(optimization_result.baseline_total_cost, 2),
            "total_cost_savings_vs_baseline": round(
                optimization_result.total_cost_savings_vs_baseline, 2
            ),
            "expected_service_level_pct": round(
                optimization_result.expected_service_level * 100.0, 2
            ),
            "baseline_service_level_pct": round(
                optimization_result.baseline_service_level * 100.0, 2
            ),
            "budget_cap": round(optimization_result.constraints.max_total_budget, 2),
            "unit_cap": optimization_result.constraints.max_total_units,
            "demand_shock_pct": round(optimization_result.constraints.demand_shock_pct, 1),
            "lead_time_multiplier": round(
                optimization_result.constraints.lead_time_multiplier, 2
            ),
            "baseline_budget_violated": optimization_result.baseline_budget_violated,
            "baseline_capacity_violated": optimization_result.baseline_capacity_violated,
            "top_orders": [
                {
                    "store_id": d.store_id,
                    "item_id": d.item_id,
                    "recommended_order_qty": d.recommended_order_qty,
                    "order_cost": round(d.order_cost, 2),
                    "cost_savings_vs_baseline": round(d.cost_savings_vs_baseline, 2),
                }
                for d in top_orders
            ],
        }

    @staticmethod
    def _parse_llm_json(raw_text: str) -> dict[str, Any]:
        cleaned = raw_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("LLM response JSON must be an object")
        for req_key in ("headline", "executive_summary", "priority_actions", "risk_alerts"):
            if req_key not in data:
                raise ValueError(f"LLM response missing required key: {req_key}")
        return data

    @staticmethod
    def _build_deterministic_fallback(
        metrics: dict[str, Any],
    ) -> ExecutiveBriefingResponse:
        scope = metrics["focus_store_id"]
        savings = metrics["total_cost_savings_vs_baseline"]
        svc_pct = metrics["expected_service_level_pct"]
        base_svc_pct = metrics["baseline_service_level_pct"]

        headline = (
            f"[{scope}] HiGHS MILP Achieves {svc_pct:.2f}% Service Level "
            f"with ${savings:,.2f} Cost Savings vs. Rule-of-Thumb"
        )
        executive_summary = (
            f"Across scope {scope}, StockPulse AI's LightGBM quantile forecaster recorded "
            f"{metrics['forecast_wape_pct']:.2f}% WAPE and {metrics['interval_coverage_80_pct']:.2f}% "
            f"80% prediction-interval coverage. Under a ${metrics['budget_cap']:,.2f} budget and "
            f"{metrics['unit_cap']:,}-unit capacity cap, the MILP optimizer allocated "
            f"{metrics['total_units_ordered']:,} units (${metrics['total_order_cost']:,.2f} procurement), "
            f"reducing expected holding + stockout cost from ${metrics['baseline_total_cost']:,.2f} "
            f"to ${metrics['total_expected_cost']:,.2f} (${savings:,.2f} savings) while lifting service "
            f"level from {base_svc_pct:.2f}% to {svc_pct:.2f}%."
        )

        priority_actions: list[str] = []
        for order in metrics["top_orders"][:3]:
            priority_actions.append(
                f"Release PO for {order['store_id']} / {order['item_id']}: "
                f"{order['recommended_order_qty']:,} units (${order['order_cost']:,.2f}, "
                f"saving ${order['cost_savings_vs_baseline']:,.2f} vs. baseline)."
            )
        if not priority_actions:
            priority_actions.append(
                "Maintain current on-hand stock levels; zero replenishment orders triggered under active constraints."
            )
        if metrics["baseline_budget_violated"] or metrics["baseline_capacity_violated"]:
            priority_actions.append(
                "Enforce MILP allocation strictly: the 14-day rule-of-thumb heuristic violated global budget or warehouse caps."
            )

        risk_alerts: list[str] = [
            (
                f"Distribution monitoring reports {metrics['drift_status']} "
                f"(PSI={metrics['drift_psi_score']:.4f}, KS p-value={metrics['ks_pvalue']:.4f})."
            )
        ]
        if metrics["top_anomalies"]:
            top_a = metrics["top_anomalies"][0]
            risk_alerts.append(
                f"Detected {metrics['anomaly_count']} demand anomalies; largest {top_a['shock_type']} at "
                f"{top_a['store_id']} / {top_a['item_id']} on {top_a['date']} "
                f"(actual={top_a['actual_sales']}, P50={top_a['expected_p50']})."
            )
        else:
            risk_alerts.append("Zero demand anomalies detected in the active validation scope.")

        return ExecutiveBriefingResponse(
            headline=headline,
            executive_summary=executive_summary,
            priority_actions=priority_actions,
            risk_alerts=risk_alerts,
            generation_mode="deterministic_fallback",
            grounded_metrics=metrics,
        )

    def generate(
        self,
        forecast_metrics: ForecastMetrics,
        drift_report: DriftReport,
        anomalies: list[AnomalyRecord],
        optimization_result: OptimizationResult,
        focus_store_id: str | None = None,
    ) -> ExecutiveBriefingResponse:
        """Generate grounded executive briefing using LLM if available, else fallback."""
        grounded = self.build_grounded_metrics(
            forecast_metrics=forecast_metrics,
            drift_report=drift_report,
            anomalies=anomalies,
            optimization_result=optimization_result,
            focus_store_id=focus_store_id,
        )

        if not self.is_llm_configured:
            return self._build_deterministic_fallback(grounded)

        try:
            client = self._injected_client or OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=15.0,
            )
            completion = client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Synthesize an executive procurement briefing from these exact "
                            f"grounded metrics:\n{json.dumps(grounded, indent=2)}"
                        ),
                    },
                ],
            )
            raw_content = completion.choices[0].message.content or ""
            parsed = self._parse_llm_json(raw_content)
            return ExecutiveBriefingResponse(
                headline=str(parsed["headline"]),
                executive_summary=str(parsed["executive_summary"]),
                priority_actions=[str(x) for x in parsed["priority_actions"]],
                risk_alerts=[str(x) for x in parsed["risk_alerts"]],
                generation_mode="llm",
                grounded_metrics=grounded,
            )
        except Exception:
            return self._build_deterministic_fallback(grounded)
