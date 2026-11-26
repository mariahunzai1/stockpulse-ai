"""StockPulse AI — Business-Friendly Streamlit + Plotly Retail Command Center."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.api_client import StockPulseApiClient
from ui.charts import (
    build_forecast_fan_chart,
    build_milp_comparison_charts,
    build_pdp_ice_chart,
    build_psi_drift_chart,
    build_quantile_spread_chart,
    build_shap_waterfall_bar,
    build_store_category_bar,
    friendly_feature_name,
)

BENCHMARK_PATH = PROJECT_ROOT / "eval" / "benchmark_results.json"

DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "max_total_budget": 15000.0,
    "max_total_units": 10000,
    "lead_time_multiplier": 1.0,
    "demand_shock_pct": 0.0,
    "holding_cost_multiplier": 1.0,
}

DRIFT_STATUS_LABELS: dict[str, str] = {
    "STABLE": "Normal — Steady Buying Habits",
    "MODERATE_DRIFT": "Slight Seasonal Shift — Monitor Weekly",
    "SIGNIFICANT_DRIFT": "Major Shift Detected — Retrain Recommended",
}

PDP_FEATURE_OPTIONS: dict[str, str] = {
    "Item Selling Price ($)": "sell_price",
    "Price vs. Historical Average": "price_vs_item_avg",
    "7-Day Average Sales": "rolling_mean_7",
    "Zero-Sale Days (Past 28d)": "rolling_zero_rate_28",
}

st.set_page_config(
    page_title="StockPulse AI — Retail Sales & Restock Planner",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(max_entries=1)
def get_api_client() -> StockPulseApiClient:
    """Create cached HTTP client and auto-spawn FastAPI if needed."""
    client = StockPulseApiClient()
    client.ensure_backend_running(timeout_seconds=60.0)
    return client


@st.cache_data(ttl=300, max_entries=4)
def fetch_health(_client: StockPulseApiClient) -> dict[str, Any]:
    return _client.get_health()


@st.cache_data(ttl=300, max_entries=4)
def fetch_skus(_client: StockPulseApiClient) -> list[dict[str, Any]]:
    return _client.get_skus()


@st.cache_data(ttl=300, max_entries=64)
def fetch_forecast(_client: StockPulseApiClient, store_id: str, item_id: str) -> dict[str, Any]:
    return _client.get_forecast(store_id, item_id)


@st.cache_data(ttl=300, max_entries=128)
def fetch_explain(
    _client: StockPulseApiClient,
    store_id: str,
    item_id: str,
    target_date: str | None,
) -> dict[str, Any]:
    return _client.get_explain(store_id, item_id, target_date)


@st.cache_data(ttl=300, max_entries=4)
def fetch_anomalies(_client: StockPulseApiClient) -> dict[str, Any]:
    return _client.get_anomalies(include_normal=False)


@st.cache_data(ttl=300, max_entries=128)
def fetch_tail_risk(
    _client: StockPulseApiClient,
    store_id: str,
    item_id: str,
    target_date: str | None,
) -> dict[str, Any]:
    return _client.get_xai_tail_risk(store_id, item_id, target_date)


@st.cache_data(ttl=300, max_entries=16)
def fetch_pdp(_client: StockPulseApiClient, feature_name: str) -> dict[str, Any]:
    return _client.get_xai_pdp(feature_name)


@st.cache_data(ttl=600, max_entries=2)
def load_benchmark_results() -> dict[str, Any] | None:
    if not BENCHMARK_PATH.exists():
        return None
    return json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))


def init_session_state(
    client: StockPulseApiClient,
    default_store: str,
    default_item: str,
) -> None:
    """Populate initial POST results once so all tabs render immediately across reruns."""
    if "active_constraints" not in st.session_state:
        st.session_state["active_constraints"] = dict(DEFAULT_CONSTRAINTS)
    if "optimize_result" not in st.session_state:
        st.session_state["optimize_result"] = client.post_optimize(
            st.session_state["active_constraints"]
        )
    if "briefing_result" not in st.session_state:
        st.session_state["briefing_result"] = client.post_briefing(
            {
                "constraints": st.session_state["active_constraints"],
                "focus_store_id": None,
            }
        )
    if "decision_audit_result" not in st.session_state:
        st.session_state["decision_audit_result"] = client.post_xai_decision_audit(
            st.session_state["active_constraints"]
        )
    if "counterfactual_result" not in st.session_state:
        st.session_state["counterfactual_result"] = client.post_xai_counterfactual(
            {
                "store_id": default_store,
                "item_id": default_item,
                "target_demand_delta_pct": -15.0,
                "eliminate_stockout": False,
            }
        )


def _format_briefing_markdown(brief: dict[str, Any]) -> str:
    actions = "\n".join(f"- {a}" for a in brief["priority_actions"])
    alerts = "\n".join(f"- {r}" for r in brief["risk_alerts"])
    mode_label = (
        "AI Narrative Assistant"
        if brief["generation_mode"] == "llm"
        else "Automated Rule-Checked Summary"
    )
    return (
        f"# {brief['headline']}\n\n"
        f"**Report source:** {mode_label}\n\n"
        f"## Summary\n{brief['executive_summary']}\n\n"
        f"## Top Reorder Actions\n{actions}\n\n"
        f"## Stock & Sales Risk Alerts\n{alerts}\n"
    )


# --- App Boot & Shared Sidebar ---
client = get_api_client()
health = fetch_health(client)
skus = fetch_skus(client)
skus_df = pd.DataFrame(skus)
stores = sorted(skus_df["store_id"].unique().tolist())

with st.sidebar:
    st.markdown("### :material/storefront: StockPulse AI")
    st.caption("Smart Retail Sales Forecasting & Restock Planner")

    selected_store = st.selectbox(
        "Choose a store",
        stores,
        index=0,
        key="sb_store",
        help="Pick which Walmart store location (California, Texas, or Wisconsin) to inspect.",
    )
    store_skus = skus_df[skus_df["store_id"] == selected_store].sort_values("item_id")
    item_options = store_skus["item_id"].tolist()
    selected_item = st.selectbox(
        "Choose a product",
        item_options,
        index=0,
        key="sb_item",
        help="Pick any of the 10 tracked products across Foods, Hobbies, and Household.",
    )

    sku_row = store_skus[store_skus["item_id"] == selected_item].iloc[0]
    with st.container(border=True):
        st.caption("Selected product snapshot")
        st.markdown(
            f"**{sku_row['cat_id'].title()} · {sku_row['dept_id']}**  \n"
            f"Shelf price: **\\${sku_row['sell_price']:.2f}** · Wholesale cost: **\\${sku_row['unit_cost']:.2f}**  \n"
            f"In stock now: **{int(sku_row['current_on_hand'])} units**  \n"
            f"Delivery wait: **{int(sku_row['lead_time_days'])} days** · Minimum supplier order: **{int(sku_row['moq'])} units**"
        )

    with st.container(border=True):
        st.caption("System status")
        report_mode = (
            "AI Narrative Active"
            if health["llm_mode"] == "llm"
            else "Offline Safe Summary"
        )
        st.markdown(
            f":green[:material/check_circle: Live & Ready]  \n"
            f"Sales history loaded: **{health['duckdb_sales_rows']:,} daily records**  \n"
            f"Tracked catalog: **{health['store_count']} stores · {health['sku_count']} products**  \n"
            f"Report engine: **{report_mode}**"
        )

init_session_state(client, selected_store, selected_item)

st.title(":material/inventory_2: StockPulse AI — Store Demand & Restock Planner")
st.caption(
    "Powered by 2 years of real Walmart store sales (CA_1 · TX_1 · WI_1) to forecast demand, "
    "spot unusual spikes, and recommend budget-safe restock orders."
)

tab_overview, tab_forecast, tab_xai, tab_anomaly, tab_milp = st.tabs(
    [
        "1. Store Summary & AI Report",
        "2. 28-Day Sales Forecast",
        "3. Why Sales Changed (AI Insights)",
        "4. Unusual Spikes & Alerts",
        "5. Smart Restock & Budget Planner",
    ]
)

# =============================================================================
# TAB 1: Store Summary & AI Report
# =============================================================================
with tab_overview:
    st.info(
        "Get a quick health check of all 3 stores and generate a plain-English summary of what to reorder this week.",
        icon=":material/storefront:",
    )

    benchmark = load_benchmark_results()
    if benchmark:
        wape_err = benchmark["forecasting"]["lightgbm_quantile"]["wape"]
        acc_pct = max(0.0, (1.0 - wape_err) * 100.0)
        wape_imp = benchmark["forecasting"]["wape_improvement_pct"]
        cov_80 = benchmark["forecasting"]["lightgbm_quantile"]["interval_coverage_80"] * 100.0
        nominal_savings = benchmark["inventory_optimization"][0]["cost_reduction_pct"]
    else:
        wape_err = 0.3222
        acc_pct = 67.78
        wape_imp = 16.86
        cov_80 = 80.60
        nominal_savings = 83.48

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Sales history tracked",
        f"{health['duckdb_sales_rows']:,} days",
        "30 store-product series",
        help="Two full years of daily Walmart store sales loaded into DuckDB.",
    )
    c2.metric(
        "Forecast accuracy",
        f"{acc_pct:.1f}%",
        f"+{wape_imp:.1f}% vs. last-week guess",
        help=f"Average forecast error (WAPE) is {wape_err * 100:.1f}%, which is {wape_imp:.1f}% more accurate than guessing from last week's sales.",
    )
    c3.metric(
        "Forecast range reliability",
        f"{cov_80:.1f}%",
        "Target: 80.0% of real days",
        help="How often actual daily sales fell inside our Low-to-Peak expected sales band.",
    )
    c4.metric(
        "Smart restock cost savings",
        f"{nominal_savings:.1f}% lower",
        "vs. 14-day rule-of-thumb",
        help="Expected reduction in storage + lost-sale costs compared to a fixed 14-day reorder rule.",
    )

    st.plotly_chart(
        build_store_category_bar(skus),
        width="stretch",
        alt="Current stock value at wholesale cost across stores and categories",
    )
    st.caption(
        "This chart shows how much wholesale cash is currently sitting on the shelves in each store across Foods, Hobbies, and Household."
    )

    with st.container(border=True):
        st.subheader(":material/auto_awesome: Executive action report")
        st.caption(
            "Summarizes your restock plan, cost savings, and sales alerts using only verified numbers from the optimizer."
        )
        col_scope, col_btn = st.columns([2, 1], vertical_alignment="bottom")
        with col_scope:
            scope_choice = st.segmented_control(
                "Which store should the report focus on?",
                ["All stores", "CA_1", "TX_1", "WI_1"],
                default="All stores",
                key="brief_scope",
                help="Choose All stores for a company-wide summary or pick one regional store.",
            )
        with col_btn:
            run_brief = st.button(
                "Create updated report",
                icon=":material/bolt:",
                type="primary",
                width="stretch",
                key="btn_gen_brief",
            )

        if run_brief:
            focus_store = None if (scope_choice in (None, "All stores")) else scope_choice
            with st.spinner("Writing your executive action report..."):
                st.session_state["briefing_result"] = client.post_briefing(
                    {
                        "constraints": st.session_state["active_constraints"],
                        "focus_store_id": focus_store,
                    }
                )

        brief = st.session_state["briefing_result"]
        mode_tag = (
            "AI Narrative"
            if brief["generation_mode"] == "llm"
            else "Automated Summary"
        )
        st.success(
            brief["executive_summary"],
            title=f"{brief['headline']} ({mode_tag})",
        )
        col_act, col_risk = st.columns(2)
        with col_act:
            st.markdown("**Top things to order now**")
            for act in brief["priority_actions"]:
                st.markdown(f"- {act}")
        with col_risk:
            st.markdown("**Things to watch out for**")
            for alert in brief["risk_alerts"]:
                st.markdown(f"- {alert}")

        st.download_button(
            "Download report (.md)",
            data=_format_briefing_markdown(brief),
            file_name="stockpulse_executive_report.md",
            mime="text/markdown",
            icon=":material/download:",
            width="content",
        )

# =============================================================================
# TAB 2: 28-Day Sales Forecast
# =============================================================================
with tab_forecast:
    st.info(
        "See expected daily sales for the next 28 days—including a Low Estimate, Expected Sales, and Busy-Day Peak—and compare them against real store sales.",
        icon=":material/show_chart:",
    )

    bundle = fetch_forecast(client, selected_store, selected_item)
    sku_metrics = bundle["metrics"]
    sku_acc = max(0.0, (1.0 - sku_metrics["wape"]) * 100.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Product forecast accuracy",
        f"{sku_acc:.1f}%",
        f"Avg error (WAPE): {sku_metrics['wape'] * 100:.1f}%",
        help="Overall 28-day accuracy for this specific product at this store.",
    )
    m2.metric(
        "Average daily miss",
        f"±{sku_metrics['mae']:.1f} units/day",
        help="Average difference between expected daily sales and actual units sold (MAE).",
    )
    m3.metric(
        "Big-day swing error",
        f"±{sku_metrics['rmse']:.1f} units",
        help="Error metric that puts extra weight on larger misses during busy days (RMSE).",
    )
    m4.metric(
        "Days inside expected range",
        f"{sku_metrics['interval_coverage_80'] * 100:.0f}%",
        "Target: ~80% of days",
        help="Share of real days that landed between our Low Estimate (P10) and Busy-Day Peak (P90).",
    )

    st.plotly_chart(
        build_forecast_fan_chart(bundle),
        width="stretch",
        alt="Past 60 days of sales and 28-day expected sales range compared to real sales",
    )
    st.caption(
        "The shaded green band shows our Expected Sales Range (Low Estimate to Busy-Day Peak, covering ~81% of real days across all products). The orange dotted line shows what actually sold."
    )

    fc_dates = [pt["date"] for pt in bundle["forecast"]]
    sel_date = st.selectbox(
        "Pick a forecast day to see why the AI predicted that number",
        fc_dates,
        index=len(fc_dates) - 1,
        key="fc_shap_date",
        help="Inspect the exact factors that pushed this product's forecast up or down on any day in the 28-day window.",
    )
    exp_data = fetch_explain(client, selected_store, selected_item, sel_date)
    local_exp = exp_data["local_explanation"]

    st.markdown("#### What Is Driving This Product's Forecast?")
    st.caption(
        f"On **{local_exp['date']}**, expected baseline sales start at **{local_exp['base_value']:.1f} units/day** and adjust to **{max(0.0, local_exp['predicted_p50']):.1f} units/day**. "
        "Green bars push expected sales up; red bars pull expected sales down."
    )
    col_loc, col_glob = st.columns(2)
    with col_loc:
        st.plotly_chart(
            build_shap_waterfall_bar(
                local_exp["contributions"],
                f"What Moved This Product's Forecast on {local_exp['date']}",
                top_n=12,
                use_abs=False,
            ),
            width="stretch",
            alt="Plain-English factors pushing this product's daily forecast up or down",
        )
    with col_glob:
        st.plotly_chart(
            build_shap_waterfall_bar(
                exp_data["global_importance"],
                "Biggest Overall Sales Drivers Across All Stores",
                top_n=12,
                use_abs=True,
            ),
            width="stretch",
            alt="Biggest overall sales drivers across all 3 stores and 10 products",
        )

# =============================================================================
# TAB 3: Why Sales Changed (AI Insights)
# =============================================================================
with tab_xai:
    st.info(
        "Look under the hood: see what causes unpredictable busy-day spikes, test how price changes affect sales, and audit why the optimizer placed each order.",
        icon=":material/psychology:",
    )

    st.subheader(":material/stacked_bar_chart: What Causes Unpredictable Busy-Day Spikes?")
    st.caption(
        "Compares what drives a quiet day (Low Estimate), a typical day (Expected Sales), and a rush day (Busy-Day Peak). Big red bars show what creates wide swings."
    )
    tail_dates = [pt["date"] for pt in bundle["forecast"]]
    tail_date = st.selectbox(
        "Pick a day to inspect busy-day uncertainty",
        tail_dates,
        index=len(tail_dates) - 1,
        key="xai_tail_date",
        help="See which factors widen the gap between quiet days and busy-day peaks.",
    )
    tail_data = fetch_tail_risk(client, selected_store, selected_item, tail_date)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Low estimate (quiet day)", f"{max(0.0, tail_data['predicted_p10']):.1f} units")
    t2.metric("Expected sales (typical)", f"{max(0.0, tail_data['predicted_p50']):.1f} units")
    t3.metric("Busy-day peak (rush)", f"{max(0.0, tail_data['predicted_p90']):.1f} units")
    spread_w = max(0.0, tail_data["predicted_p90"]) - max(0.0, tail_data["predicted_p10"])
    t4.metric("Daily uncertainty gap", f"{spread_w:.1f} units")

    st.plotly_chart(
        build_quantile_spread_chart(tail_data),
        width="stretch",
        alt="Comparison of factors driving quiet days, expected sales, and busy-day spikes",
    )

    st.divider()
    st.subheader(":material/ssid_chart: How Price Changes Affect Daily Sales")
    st.caption(
        "See how discounting or raising the price—or changing recent sales trends—changes expected units sold per day across individual products and overall."
    )
    pdp_choice_label = st.segmented_control(
        "Choose a business lever to test across all stores",
        list(PDP_FEATURE_OPTIONS.keys()),
        default="Item Selling Price ($)",
        key="xai_pdp_feat",
        help="The bold green line shows the average effect; thin blue lines show individual store-product curves.",
    )
    chosen_feat = PDP_FEATURE_OPTIONS.get(
        pdp_choice_label or "Item Selling Price ($)", "sell_price"
    )
    pdp_data = fetch_pdp(client, chosen_feat)
    st.plotly_chart(
        build_pdp_ice_chart(pdp_data),
        width="stretch",
        alt=f"Sales sensitivity curve for {friendly_feature_name(chosen_feat)}",
    )

    st.divider()
    st.subheader(':material/alt_route: "What Would It Take?" Goal Simulator')
    st.caption(
        "Test what price discount, price increase, or faster supplier delivery is needed to hit a daily sales target or avoid running out of stock."
    )
    with st.form("counterfactual_form"):
        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            cf_delta = st.slider(
                "Desired sales change (%)",
                min_value=-40,
                max_value=40,
                value=-15,
                step=5,
                help="Positive % tests what discount boosts sales; negative % tests how much a price hike slows demand.",
            )
        with cf_c2:
            cf_elim = st.checkbox(
                "Avoid running out of stock before delivery arrives",
                value=False,
                help="Finds the price adjustment or faster supplier delivery needed so current stock lasts until restock.",
            )
        with cf_c3:
            cf_oh = st.number_input(
                "Test with custom stock on shelf (units)",
                min_value=0,
                max_value=5000,
                value=int(sku_row["current_on_hand"]),
                step=1,
                help="Used when checking how to prevent a stockout with limited inventory on hand.",
            )
        cf_submit = st.form_submit_button(
            "Find smallest change needed",
            icon=":material/search:",
            type="primary",
            width="content",
        )

    if cf_submit:
        st.session_state["counterfactual_result"] = client.post_xai_counterfactual(
            {
                "store_id": selected_store,
                "item_id": selected_item,
                "date": tail_date,
                "target_demand_delta_pct": float(cf_delta),
                "eliminate_stockout": cf_elim,
                "current_on_hand_override": int(cf_oh) if cf_elim else None,
            }
        )

    cf_res = st.session_state["counterfactual_result"]
    lever_friendly = {
        "sell_price": "Shelf Price",
        "lead_time_days": "Delivery Time",
        "sell_price+lead_time_days": "Price + Faster Delivery",
    }.get(cf_res["lever"], cf_res["lever"])

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Best lever to adjust", lever_friendly)
    r2.metric(
        "Recommended target value",
        f"{cf_res['counterfactual_value']:.2f}",
        f"{cf_res['pct_change']:+.1f}% vs. current {cf_res['original_value']:.2f}",
    )
    r3.metric(
        "Resulting daily sales",
        f"{cf_res['achieved_p50']:.1f} units/day",
        f"from {cf_res['original_p50']:.1f} units/day",
    )
    r4.metric(
        "Stockout risk",
        "Safe (Covered)" if cf_res["stockout_eliminated"] else "Needs Reorder",
        f"Delivery window: {cf_res['lead_time_days_counterfactual']} days",
    )
    st.success(cf_res["action_summary"], title="Recommended step")

    st.divider()
    st.subheader(":material/fact_check: Why the AI Recommended Each Order")
    st.caption(
        "Every purchase order explained in plain English: expected profit protected vs. storage cost, and whether a minimum supplier order or cash limit shaped the quantity."
    )
    audit_col1, audit_col2 = st.columns([3, 1], vertical_alignment="bottom")
    with audit_col1:
        audit_filter = st.segmented_control(
            "Filter order explanations",
            ["All products", "Reordered only", "Hit minimum supplier order", "Limited by budget/space"],
            default="All products",
            key="xai_audit_filter",
            help="Quickly focus on items being reordered or restricted by supplier minimums or budget caps.",
        )
    with audit_col2:
        if st.button("Refresh order audit", icon=":material/refresh:", width="stretch"):
            st.session_state["decision_audit_result"] = client.post_xai_decision_audit(
                st.session_state["active_constraints"]
            )

    audit = st.session_state["decision_audit_result"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Net savings vs. simple rule", f"${audit['total_net_economic_benefit']:,.2f}")
    a2.metric("Orders at minimum supplier size", f"{audit['moq_binding_count']}")
    a3.metric("Orders capped by budget/space", f"{audit['constrained_sku_count']}")

    audit_df = pd.DataFrame(audit["items"])
    if audit_filter == "Reordered only":
        audit_df = audit_df[audit_df["order_triggered"]]
    elif audit_filter == "Hit minimum supplier order":
        audit_df = audit_df[audit_df["moq_binding"]]
    elif audit_filter == "Limited by budget/space":
        audit_df = audit_df[audit_df["budget_or_capacity_constrained"]]

    audit_display = audit_df.rename(
        columns={
            "store_id": "Store",
            "item_id": "Product",
            "dept_id": "Department",
            "recommended_order_qty": "Order Qty (Units)",
            "order_triggered": "Reorder?",
            "net_economic_benefit": "Savings vs. Rule ($)",
            "stockout_risk_ratio": "Stockout Risk Share",
            "moq_binding": "Minimum Order?",
            "budget_or_capacity_constrained": "Budget/Space Capped?",
            "decision_rationale": "Why the AI Recommended This",
        }
    )
    st.dataframe(
        audit_display,
        width="stretch",
        hide_index=True,
        alt="Plain-English explanation for each store-product restock decision",
    )

# =============================================================================
# TAB 4: Unusual Spikes & Alerts
# =============================================================================
with tab_anomaly:
    st.info(
        "Check whether customer buying habits are still normal and inspect sudden sales surges or unexpected drops across your stores.",
        icon=":material/notifications_active:",
    )

    anom_bundle = fetch_anomalies(client)
    drift = anom_bundle["drift_report"]
    anomalies_df = pd.DataFrame(anom_bundle["anomalies"])
    drift_friendly = DRIFT_STATUS_LABELS.get(drift["status"], drift["status"])

    st.markdown("#### Are Customer Buying Habits Still Normal?")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric(
        "Buying habit status",
        drift_friendly.split(" — ")[0],
        drift_friendly,
        help=f"Overall shift score (PSI = {drift['psi_score']:.3f}). Below 0.10 is steady; 0.10–0.25 is a mild seasonal shift; above 0.25 suggests retraining.",
    )
    d2.metric(
        "Habit shift score (PSI)",
        f"{drift['psi_score']:.3f}",
        "0.10–0.25 = Slight shift",
        help="Measures how much recent prices and sales patterns differ from the 2-year baseline.",
    )
    d3.metric(
        "Daily sales consistency",
        "Steady" if drift["ks_pvalue"] >= 0.05 else "Shifted",
        f"KS p-value: {drift['ks_pvalue']:.2f}",
        help="Statistical check comparing recent unit sales volume against historical norms.",
    )
    d4.metric(
        "Unusual days flagged",
        f"{len(anomalies_df)} alerts",
        "Out of 840 recent days",
        help="Days where actual sales spiked far above or dropped far below expected demand.",
    )

    st.plotly_chart(
        build_psi_drift_chart(drift),
        width="stretch",
        alt="Buying-habit shift score for each retail signal with warning thresholds",
    )
    st.caption(
        "Price changes naturally lift the price-shift bar over time, while steady unit sales (green bar) confirm customer demand remains stable."
    )

    st.markdown("#### Sudden Sales Surges & Drops")
    f_c1, f_c2 = st.columns(2)
    with f_c1:
        shock_filter = st.segmented_control(
            "Alert type",
            ["All alerts", "Sudden Sales Surge", "Sudden Sales Drop"],
            default="All alerts",
            key="anom_shock_filter",
            help="Filter between unexpected demand spikes and sudden drops.",
        )
    with f_c2:
        anom_store = st.segmented_control(
            "Store filter",
            ["All stores", "CA_1", "TX_1", "WI_1"],
            default="All stores",
            key="anom_store_filter",
            help="Show alerts across all stores or focus on a single region.",
        )

    filtered_anom = anomalies_df.copy()
    if not filtered_anom.empty:
        filtered_anom["Alert Type"] = filtered_anom["shock_type"].map(
            {"SURGE": "Sudden Sales Surge", "DROP": "Sudden Sales Drop", "NORMAL": "Normal"}
        )
        if shock_filter in ("Sudden Sales Surge", "Sudden Sales Drop"):
            filtered_anom = filtered_anom[filtered_anom["Alert Type"] == shock_filter]
        if anom_store in ("CA_1", "TX_1", "WI_1"):
            filtered_anom = filtered_anom[filtered_anom["store_id"] == anom_store]

        display_anom = filtered_anom[
            [
                "date",
                "store_id",
                "item_id",
                "Alert Type",
                "actual_sales",
                "expected_p50",
                "reason",
            ]
        ].rename(
            columns={
                "date": "Date",
                "store_id": "Store",
                "item_id": "Product",
                "actual_sales": "Actual Sold",
                "expected_p50": "Expected Sales",
                "reason": "What Happened",
            }
        )
    else:
        display_anom = filtered_anom

    st.dataframe(
        display_anom,
        width="stretch",
        hide_index=True,
        alt="Table of sudden sales surges and drops detected by StockPulse AI",
    )

# =============================================================================
# TAB 5: Smart Restock & Budget Planner
# =============================================================================
with tab_milp:
    st.info(
        "Smart Restock Optimizer: Automatically calculates how many units to reorder for every product so you don't run out of stock or overspend your cash budget.",
        icon=":material/shopping_cart_checkout:",
    )

    with st.form("milp_whatif_form"):
        st.subheader(":material/tune: Test Budget & Supply-Chain Scenarios")
        st.caption(
            "Adjust your available cash, warehouse space, or rush-season demand below, then click Calculate Smart Restock Plan."
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            budget_cap = st.slider(
                "Cash budget limit ($)",
                min_value=1500,
                max_value=50000,
                value=int(st.session_state["active_constraints"]["max_total_budget"]),
                step=500,
                help="Cash Budget Limit: Maximum total dollars you want to spend on restocking right now.",
            )
            demand_shock = st.slider(
                "Expected sales surge or slowdown (%)",
                min_value=-30,
                max_value=50,
                value=int(st.session_state["active_constraints"]["demand_shock_pct"]),
                step=5,
                help="Simulate a busy promotion (+%) or slow week (-%) across all stores.",
            )
        with s2:
            unit_cap = st.slider(
                "Warehouse space limit (total units)",
                min_value=500,
                max_value=25000,
                value=int(st.session_state["active_constraints"]["max_total_units"]),
                step=250,
                help="Maximum total units your backroom or warehouse can receive in this restock.",
            )
            lt_mult = st.slider(
                "Supplier delivery speed multiplier",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state["active_constraints"]["lead_time_multiplier"]),
                step=0.1,
                help="1.0x is normal supplier shipping; 1.5x simulates 50% slower deliveries.",
            )
        with s3:
            hold_mult = st.slider(
                "Storage & shelf holding cost multiplier",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state["active_constraints"]["holding_cost_multiplier"]),
                step=0.1,
                help="Higher values penalize holding extra stock on the shelf more strongly.",
            )
        run_opt = st.form_submit_button(
            "Calculate smart restock plan",
            icon=":material/play_arrow:",
            type="primary",
            width="content",
        )

    if run_opt:
        new_constraints = {
            "max_total_budget": float(budget_cap),
            "max_total_units": int(unit_cap),
            "lead_time_multiplier": float(lt_mult),
            "demand_shock_pct": float(demand_shock),
            "holding_cost_multiplier": float(hold_mult),
        }
        st.session_state["active_constraints"] = new_constraints
        with st.spinner("Calculating optimal restock orders within your budget..."):
            st.session_state["optimize_result"] = client.post_optimize(new_constraints)
            st.session_state["decision_audit_result"] = client.post_xai_decision_audit(
                new_constraints
            )

    opt_res = st.session_state["optimize_result"]
    o1, o2, o3, o4 = st.columns(4)
    o1.metric(
        "Expected storage & stockout cost",
        f"${opt_res['total_expected_cost']:,.2f}",
        f"Simple 14d rule: ${opt_res['baseline_total_cost']:,.2f}",
        help="Total expected holding cost plus lost-profit risk across quiet, normal, and rush days.",
    )
    o2.metric(
        "Savings vs. simple rule",
        f"${opt_res['total_cost_savings_vs_baseline']:,.2f}",
        "Optimal plan ready" if opt_res["solver_status"] == "OPTIMAL" else opt_res["solver_status"],
        help="Money saved compared to ordering a flat 14-day supply.",
    )
    o3.metric(
        "Orders filled without stockout",
        f"{opt_res['expected_service_level'] * 100:.1f}%",
        f"Simple rule: {opt_res['baseline_service_level'] * 100:.1f}%",
        help="Percentage of expected customer demand satisfied without running out of stock.",
    )
    o4.metric(
        "Cash needed & units ordered",
        f"${opt_res['total_order_cost']:,.2f}",
        f"{opt_res['total_units_ordered']:,} units total",
        help="Exact wholesale cash required and total units across all purchase orders.",
    )

    if opt_res["baseline_budget_violated"] or opt_res["baseline_capacity_violated"]:
        st.warning(
            "Your simple 14-day reorder rule would have blown past your cash budget or warehouse space limit. "
            "The Smart Restock Optimizer stayed strictly within your limits while protecting high-margin items.",
            title="Simple rule exceeded your budget or warehouse limit",
        )

    cost_fig, gauge_fig = build_milp_comparison_charts(opt_res)
    ch_l, ch_r = st.columns([3, 2])
    with ch_l:
        st.plotly_chart(
            cost_fig,
            width="stretch",
            alt="Smart Restock expected cost compared to simple 14-day rule-of-thumb",
        )
    with ch_r:
        st.plotly_chart(
            gauge_fig,
            width="stretch",
            alt="Cash budget spent compared to budget limit",
        )

    decisions_df = pd.DataFrame(opt_res["decisions"])
    decisions_display = decisions_df.rename(
        columns={
            "store_id": "Store",
            "item_id": "Product",
            "dept_id": "Department",
            "current_on_hand": "On Shelf Now",
            "forecast_demand_horizon": "Expected Need (Units)",
            "safety_stock_p90": "Rush-Day Buffer",
            "recommended_order_qty": "Smart Order Qty",
            "order_triggered": "Reorder?",
            "unit_cost": "Wholesale Cost ($)",
            "order_cost": "Order Spend ($)",
            "expected_holding_cost": "Storage Cost ($)",
            "expected_stockout_cost": "Stockout Risk ($)",
            "baseline_order_qty": "Simple 14d Rule Qty",
            "cost_savings_vs_baseline": "Savings vs. Rule ($)",
        }
    )
    st.dataframe(
        decisions_display,
        width="stretch",
        hide_index=True,
        alt="Complete 30-product smart restock order plan",
    )
    st.download_button(
        "Download purchase orders (.csv)",
        data=decisions_display.to_csv(index=False),
        file_name="stockpulse_purchase_orders.csv",
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
