"""Dark-themed Plotly figure builders for the StockPulse AI Command Center."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

EMERALD = "#10B981"
CYAN = "#06B6D4"
AMBER = "#F59E0B"
ROSE = "#EF4444"
SLATE_TEXT = "#F9FAFB"
MUTED_LINE = "rgba(148, 163, 184, 0.25)"
PLOT_BG = "rgba(17, 24, 39, 0.65)"


def _apply_dark_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=SLATE_TEXT)),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PLOT_BG,
        font=dict(color=SLATE_TEXT, size=12),
        margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(gridcolor=MUTED_LINE, zerolinecolor=MUTED_LINE),
        yaxis=dict(gridcolor=MUTED_LINE, zerolinecolor=MUTED_LINE),
    )
    return fig


def build_store_category_bar(skus: list[dict[str, Any]]) -> go.Figure:
    """Plot current on-hand stock value across M5 regional stores and categories."""
    df = pd.DataFrame(skus)
    df["inventory_value"] = df["current_on_hand"] * df["unit_cost"]
    agg = df.groupby(["store_id", "cat_id"], as_index=False)["inventory_value"].sum()

    colors = {"FOODS": EMERALD, "HOBBIES": CYAN, "HOUSEHOLD": AMBER}
    fig = go.Figure()
    for cat in ("FOODS", "HOBBIES", "HOUSEHOLD"):
        sub = agg[agg["cat_id"] == cat]
        fig.add_trace(
            go.Bar(
                name=cat,
                x=sub["store_id"],
                y=sub["inventory_value"],
                marker_color=colors.get(cat, EMERALD),
                text=[f"${v:,.0f}" for v in sub["inventory_value"]],
                textposition="auto",
            )
        )
    fig.update_layout(barmode="group", xaxis_title="Store", yaxis_title="On-Hand Cost Value ($)")
    return _apply_dark_layout(fig, "On-Hand Inventory Valuation by Store & Category", height=360)


def build_forecast_fan_chart(bundle: dict[str, Any]) -> go.Figure:
    """Plot trailing 60-day M5 actuals alongside 28-day P10/P50/P90 quantile fan & actuals."""
    hist_df = pd.DataFrame(bundle["history"])
    fc_df = pd.DataFrame(bundle["forecast"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Trailing 60d Actuals",
            x=hist_df["date"],
            y=hist_df["sales"],
            mode="lines",
            line=dict(color="#94A3B8", width=2),
        )
    )

    # P10-P90 shaded confidence band
    fig.add_trace(
        go.Scatter(
            name="P90 Upper Bound",
            x=fc_df["date"],
            y=fc_df["p90"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            name="80% Interval (P10-P90)",
            x=fc_df["date"],
            y=fc_df["p10"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.22)",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="P50 Median Forecast",
            x=fc_df["date"],
            y=fc_df["p50"],
            mode="lines+markers",
            line=dict(color=EMERALD, width=2.5),
            marker=dict(size=5),
        )
    )
    if "actual" in fc_df.columns and fc_df["actual"].notna().any():
        fig.add_trace(
            go.Scatter(
                name="28d Validation Actuals",
                x=fc_df["date"],
                y=fc_df["actual"],
                mode="lines+markers",
                line=dict(color=AMBER, width=2, dash="dot"),
                marker=dict(size=5),
            )
        )

    title = f"Demand Forecast Fan — {bundle['store_id']} / {bundle['item_id']}"
    fig.update_layout(xaxis_title="Date", yaxis_title="Daily Units Sold")
    return _apply_dark_layout(fig, title, height=400)


def build_shap_waterfall_bar(
    contributions: list[dict[str, Any]],
    title: str,
    top_n: int = 12,
    use_abs: bool = False,
) -> go.Figure:
    """Plot horizontal TreeSHAP feature attribution bars."""
    top = list(contributions)[:top_n][::-1]
    labels = [
        f"{c['feature']} ({c['feature_value']:.2f})" if not use_abs else c["feature"]
        for c in top
    ]
    vals = [abs(c["shap_value"]) if use_abs else c["shap_value"] for c in top]
    colors = [EMERALD if v >= 0 else ROSE for v in vals]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" if not use_abs else f"{v:.3f}" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis_title="Mean |SHAP| Impact" if use_abs else "SHAP Contribution (Units)",
        yaxis_title="",
    )
    return _apply_dark_layout(fig, title, height=380)


def build_quantile_spread_chart(tail_risk: dict[str, Any], top_n: int = 8) -> go.Figure:
    """Plot multi-quantile P10/P50/P90 SHAP drivers and P90-P10 spread impact."""
    drivers = list(tail_risk["uncertainty_spread_drivers"])[:top_n][::-1]
    feats = [d["feature"] for d in drivers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="P10 SHAP", y=feats, x=[d["shap_p10"] for d in drivers], orientation="h", marker_color=CYAN)
    )
    fig.add_trace(
        go.Bar(name="P50 SHAP", y=feats, x=[d["shap_p50"] for d in drivers], orientation="h", marker_color=EMERALD)
    )
    fig.add_trace(
        go.Bar(name="P90 SHAP", y=feats, x=[d["shap_p90"] for d in drivers], orientation="h", marker_color=AMBER)
    )
    fig.add_trace(
        go.Bar(
            name="Spread (P90 - P10)",
            y=feats,
            x=[d["spread_impact"] for d in drivers],
            orientation="h",
            marker_color=ROSE,
        )
    )
    fig.update_layout(barmode="group", xaxis_title="SHAP Value (Units)")
    return _apply_dark_layout(
        fig,
        f"Multi-Quantile Tail-Risk Drivers — {tail_risk['store_id']} / {tail_risk['item_id']}",
        height=420,
    )


def build_pdp_ice_chart(pdp_data: dict[str, Any]) -> go.Figure:
    """Plot bold global Partial Dependence curve overlaid on sampled SKU ICE trajectories."""
    grid = pdp_data["grid_values"]
    fig = go.Figure()

    for idx, curve in enumerate(pdp_data["ice_curves"]):
        fig.add_trace(
            go.Scatter(
                name=f"{curve['store_id']}/{curve['item_id']}",
                x=grid,
                y=curve["values"],
                mode="lines",
                line=dict(color="rgba(6, 182, 212, 0.35)", width=1.5),
                showlegend=(idx < 4),
            )
        )

    fig.add_trace(
        go.Scatter(
            name="Global Average PDP",
            x=grid,
            y=pdp_data["average_pdp"],
            mode="lines+markers",
            line=dict(color=EMERALD, width=3.5),
            marker=dict(size=6),
        )
    )
    fig.update_layout(xaxis_title=pdp_data["feature"], yaxis_title="Predicted P50 Demand (Units)")
    return _apply_dark_layout(
        fig, f"Partial Dependence & Sampled ICE — {pdp_data['feature']}", height=390
    )


def build_psi_drift_chart(drift_report: dict[str, Any]) -> go.Figure:
    """Plot per-feature Population Stability Index bars with 0.10 / 0.25 thresholds."""
    items = sorted(drift_report["feature_psis"].items(), key=lambda kv: kv[1])
    features = [k for k, _ in items]
    psis = [v for _, v in items]
    colors = [
        ROSE if v >= 0.25 else (AMBER if v >= 0.10 else EMERALD)
        for v in psis
    ]

    fig = go.Figure(
        go.Bar(
            x=psis,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in psis],
            textposition="auto",
        )
    )
    fig.add_vline(x=0.10, line_dash="dash", line_color=AMBER, annotation_text="Moderate (0.10)")
    fig.add_vline(x=0.25, line_dash="dash", line_color=ROSE, annotation_text="Significant (0.25)")
    fig.update_layout(xaxis_title="Population Stability Index (PSI)", yaxis_title="")
    return _apply_dark_layout(fig, "Feature Distribution Drift (28d Validation vs. Train)", height=360)


def build_milp_comparison_charts(opt_result: dict[str, Any]) -> tuple[go.Figure, go.Figure]:
    """Build (cost_service_fig, budget_gauge_fig) comparing HiGHS MILP vs 14-day baseline."""
    comp_fig = go.Figure()
    comp_fig.add_trace(
        go.Bar(
            name="HiGHS MILP Optimizer",
            x=["Expected Total Cost ($)", "Holding Cost ($)", "Stockout Cost ($)"],
            y=[
                opt_result["total_expected_cost"],
                opt_result["total_expected_holding_cost"],
                opt_result["total_expected_stockout_cost"],
            ],
            marker_color=EMERALD,
            text=[
                f"${opt_result['total_expected_cost']:,.2f}",
                f"${opt_result['total_expected_holding_cost']:,.2f}",
                f"${opt_result['total_expected_stockout_cost']:,.2f}",
            ],
            textposition="auto",
        )
    )
    comp_fig.add_trace(
        go.Bar(
            name="14-Day Rule-of-Thumb Baseline",
            x=["Expected Total Cost ($)", "Holding Cost ($)", "Stockout Cost ($)"],
            y=[opt_result["baseline_total_cost"], 0.0, 0.0],
            marker_color=AMBER,
            text=[f"${opt_result['baseline_total_cost']:,.2f}", "—", "—"],
            textposition="auto",
        )
    )
    comp_fig.update_layout(barmode="group", yaxis_title="USD ($)")
    _apply_dark_layout(comp_fig, "Expected Operational Cost: HiGHS MILP vs. Baseline", height=350)

    budget_cap = max(float(opt_result["constraints"]["max_total_budget"]), 1.0)
    spent = float(opt_result["total_order_cost"])
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=spent,
            number={"prefix": "$", "valueformat": ",.0f"},
            delta={"reference": budget_cap, "prefix": "$", "valueformat": ",.0f"},
            title={"text": "Working-Capital Budget Utilization"},
            gauge={
                "axis": {"range": [0, budget_cap]},
                "bar": {"color": EMERALD},
                "bgcolor": "#1F2937",
                "steps": [
                    {"range": [0, 0.75 * budget_cap], "color": "rgba(16,185,129,0.18)"},
                    {"range": [0.75 * budget_cap, budget_cap], "color": "rgba(245,158,11,0.22)"},
                ],
                "threshold": {"line": {"color": ROSE, "width": 3}, "value": budget_cap},
            },
        )
    )
    gauge_fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=SLATE_TEXT),
        margin=dict(l=24, r=24, t=56, b=20),
    )
    return comp_fig, gauge_fig
