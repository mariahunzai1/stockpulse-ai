"""Dark-themed Plotly figure builders with plain-English retail copy for StockPulse AI."""

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

FEATURE_LABELS: dict[str, str] = {
    "rolling_mean_7": "7-Day Average Sales",
    "lag_1": "Yesterday's Sales",
    "lag_7": "Sales 1 Week Ago",
    "lag_14": "Sales 2 Weeks Ago",
    "lag_28": "Sales 4 Weeks Ago",
    "rolling_std_7": "7-Day Sales Swing",
    "rolling_mean_28": "28-Day Average Sales",
    "rolling_zero_rate_28": "Zero-Sale Days (Past 28d)",
    "sell_price": "Item Selling Price ($)",
    "price_momentum_7d": "7-Day Price Change (%)",
    "price_vs_item_avg": "Price vs. Historical Average",
    "has_event": "Holiday / Special Event",
    "event_type_code": "Event Category",
    "snap": "SNAP / Assistance Day",
    "is_weekend": "Weekend Shopping Day",
    "dow_sin": "Day-of-Week Cycle (A)",
    "dow_cos": "Day-of-Week Cycle (B)",
    "month_sin": "Seasonal Month Cycle (A)",
    "month_cos": "Seasonal Month Cycle (B)",
    "store_code": "Store Location",
    "cat_code": "Product Category",
    "dept_code": "Store Department",
    "item_code": "Product ID",
}


def friendly_feature_name(raw_name: str) -> str:
    """Translate internal model feature names into plain-English business labels."""
    return FEATURE_LABELS.get(raw_name, raw_name.replace("_", " ").title())


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
    """Plot current on-hand stock cost across regional stores and product categories."""
    df = pd.DataFrame(skus)
    df["inventory_value"] = df["current_on_hand"] * df["unit_cost"]
    agg = df.groupby(["store_id", "cat_id"], as_index=False)["inventory_value"].sum()

    colors = {"FOODS": EMERALD, "HOBBIES": CYAN, "HOUSEHOLD": AMBER}
    fig = go.Figure()
    for cat in ("FOODS", "HOBBIES", "HOUSEHOLD"):
        sub = agg[agg["cat_id"] == cat]
        fig.add_trace(
            go.Bar(
                name=cat.title(),
                x=sub["store_id"],
                y=sub["inventory_value"],
                marker_color=colors.get(cat, EMERALD),
                text=[f"${v:,.0f}" for v in sub["inventory_value"]],
                textposition="auto",
            )
        )
    fig.update_layout(
        barmode="group",
        xaxis_title="Store Location",
        yaxis_title="Current Stock Value at Cost ($)",
    )
    return _apply_dark_layout(fig, "Current Stock Value Across Stores & Categories", height=360)


def build_forecast_fan_chart(bundle: dict[str, Any]) -> go.Figure:
    """Plot trailing 60-day real sales alongside 28-day expected range & actual comparison."""
    hist_df = pd.DataFrame(bundle["history"])
    fc_df = pd.DataFrame(bundle["forecast"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name="Past 60 Days (Real Sales)",
            x=hist_df["date"],
            y=hist_df["sales"],
            mode="lines",
            line=dict(color="#94A3B8", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Busy-Day Peak (P90)",
            x=fc_df["date"],
            y=fc_df["p90"],
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.35)", width=1, dash="dash"),
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Expected Sales Range (covers 81% of real days)",
            x=fc_df["date"],
            y=fc_df["p10"],
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.35)", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(16, 185, 129, 0.22)",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Expected Sales (P50)",
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
                name="Actual Sales (28-Day Test)",
                x=fc_df["date"],
                y=fc_df["actual"],
                mode="lines+markers",
                line=dict(color=AMBER, width=2, dash="dot"),
                marker=dict(size=5),
            )
        )

    title = f"28-Day Sales Forecast vs. Recent History — {bundle['store_id']} / {bundle['item_id']}"
    fig.update_layout(xaxis_title="Date", yaxis_title="Units Sold per Day")
    return _apply_dark_layout(fig, title, height=400)


def build_shap_waterfall_bar(
    contributions: list[dict[str, Any]],
    title: str,
    top_n: int = 12,
    use_abs: bool = False,
) -> go.Figure:
    """Plot horizontal plain-English sales driver attribution bars."""
    top = list(contributions)[:top_n][::-1]
    labels = [
        (
            f"{friendly_feature_name(c['feature'])} ({c['feature_value']:.2f})"
            if not use_abs
            else friendly_feature_name(c["feature"])
        )
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
            text=[f"{v:+.2f} units" if not use_abs else f"{v:.2f} impact" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        xaxis_title=(
            "Average Impact on Daily Sales" if use_abs else "Impact on Expected Sales (Units)"
        ),
        yaxis_title="",
    )
    return _apply_dark_layout(fig, title, height=390)


def build_quantile_spread_chart(tail_risk: dict[str, Any], top_n: int = 8) -> go.Figure:
    """Plot what drives Low Estimate, Expected Sales, Busy-Day Peak, and Day-to-Day Uncertainty."""
    drivers = list(tail_risk["uncertainty_spread_drivers"])[:top_n][::-1]
    feats = [friendly_feature_name(d["feature"]) for d in drivers]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Low Estimate Driver (P10)",
            y=feats,
            x=[d["shap_p10"] for d in drivers],
            orientation="h",
            marker_color=CYAN,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Expected Sales Driver (P50)",
            y=feats,
            x=[d["shap_p50"] for d in drivers],
            orientation="h",
            marker_color=EMERALD,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Busy-Day Peak Driver (P90)",
            y=feats,
            x=[d["shap_p90"] for d in drivers],
            orientation="h",
            marker_color=AMBER,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Spike Uncertainty Gap (Peak − Low)",
            y=feats,
            x=[d["spread_impact"] for d in drivers],
            orientation="h",
            marker_color=ROSE,
        )
    )
    fig.update_layout(barmode="group", xaxis_title="Impact on Daily Units Sold")
    return _apply_dark_layout(
        fig,
        f"What Causes Busy-Day Spikes? — {tail_risk['store_id']} / {tail_risk['item_id']}",
        height=430,
    )


def build_pdp_ice_chart(pdp_data: dict[str, Any]) -> go.Figure:
    """Plot average sales response curve overlaid on individual store-product curves."""
    grid = pdp_data["grid_values"]
    feat_label = friendly_feature_name(pdp_data["feature"])
    fig = go.Figure()

    for idx, curve in enumerate(pdp_data["ice_curves"]):
        fig.add_trace(
            go.Scatter(
                name=f"{curve['store_id']} / {curve['item_id']}",
                x=grid,
                y=curve["values"],
                mode="lines",
                line=dict(color="rgba(6, 182, 212, 0.35)", width=1.5),
                showlegend=(idx < 4),
            )
        )

    fig.add_trace(
        go.Scatter(
            name="Average Across All Stores",
            x=grid,
            y=pdp_data["average_pdp"],
            mode="lines+markers",
            line=dict(color=EMERALD, width=3.5),
            marker=dict(size=6),
        )
    )
    fig.update_layout(xaxis_title=feat_label, yaxis_title="Expected Daily Sales (Units)")
    return _apply_dark_layout(
        fig, f"How '{feat_label}' Changes Expected Daily Sales", height=390
    )


def build_psi_drift_chart(drift_report: dict[str, Any]) -> go.Figure:
    """Plot buying-habit shift score per signal with normal/warning threshold lines."""
    items = sorted(drift_report["feature_psis"].items(), key=lambda kv: kv[1])
    features = [friendly_feature_name(k) for k, _ in items]
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
            text=[f"{v:.3f}" for v in psis],
            textposition="auto",
        )
    )
    fig.add_vline(x=0.10, line_dash="dash", line_color=AMBER, annotation_text="Slight Shift (0.10)")
    fig.add_vline(x=0.25, line_dash="dash", line_color=ROSE, annotation_text="Major Shift (0.25)")
    fig.update_layout(xaxis_title="Buying-Habit Shift Score (PSI — Lower Is Steadier)", yaxis_title="")
    return _apply_dark_layout(fig, "Which Buying Signals Shifted Most Recently?", height=360)


def build_milp_comparison_charts(opt_result: dict[str, Any]) -> tuple[go.Figure, go.Figure]:
    """Build (cost_comparison_fig, cash_budget_gauge_fig) for Smart Restock vs. Simple Rule."""
    comp_fig = go.Figure()
    comp_fig.add_trace(
        go.Bar(
            name="Smart Restock Optimizer",
            x=["Total Risk & Storage Cost ($)", "Storage Cost ($)", "Lost-Sale Penalty ($)"],
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
            name="Simple 14-Day Rule-of-Thumb",
            x=["Total Risk & Storage Cost ($)", "Storage Cost ($)", "Lost-Sale Penalty ($)"],
            y=[opt_result["baseline_total_cost"], 0.0, 0.0],
            marker_color=AMBER,
            text=[f"${opt_result['baseline_total_cost']:,.2f}", "—", "—"],
            textposition="auto",
        )
    )
    comp_fig.update_layout(barmode="group", yaxis_title="Cost ($)")
    _apply_dark_layout(
        comp_fig, "Expected Storage & Stockout Cost: Smart Restock vs. Simple Rule", height=350
    )

    budget_cap = max(float(opt_result["constraints"]["max_total_budget"]), 1.0)
    spent = float(opt_result["total_order_cost"])
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=spent,
            number={"prefix": "$", "valueformat": ",.0f"},
            delta={"reference": budget_cap, "prefix": "$", "valueformat": ",.0f"},
            title={"text": "Cash Budget Used vs. Limit"},
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
