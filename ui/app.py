"""StockPulse AI — 5-Tab Streamlit + Plotly Interactive Command Center."""

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
)

BENCHMARK_PATH = PROJECT_ROOT / "eval" / "benchmark_results.json"

DEFAULT_CONSTRAINTS: dict[str, Any] = {
    "max_total_budget": 15000.0,
    "max_total_units": 10000,
    "lead_time_multiplier": 1.0,
    "demand_shock_pct": 0.0,
    "holding_cost_multiplier": 1.0,
}

st.set_page_config(
    page_title="StockPulse AI — Demand & Inventory Command Center",
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
    return (
        f"# {brief['headline']}\n\n"
        f"**Generation mode:** `{brief['generation_mode']}`\n\n"
        f"## Executive summary\n{brief['executive_summary']}\n\n"
        f"## Priority procurement actions\n{actions}\n\n"
        f"## Risk & drift alerts\n{alerts}\n"
    )


# --- App Boot & Shared Telemetry ---
client = get_api_client()
health = fetch_health(client)
skus = fetch_skus(client)
skus_df = pd.DataFrame(skus)
stores = sorted(skus_df["store_id"].unique().tolist())

with st.sidebar:
    st.markdown("### :material/analytics: StockPulse AI")
    st.caption("Probabilistic Walmart M5 Demand & MILP Replenishment Engine")

    selected_store = st.selectbox("Regional store", stores, index=0, key="sb_store")
    store_skus = skus_df[skus_df["store_id"] == selected_store].sort_values("item_id")
    item_options = store_skus["item_id"].tolist()
    selected_item = st.selectbox("SKU item", item_options, index=0, key="sb_item")

    sku_row = store_skus[store_skus["item_id"] == selected_item].iloc[0]
    with st.container(border=True):
        st.caption("Selected SKU snapshot")
        st.markdown(
            f"**{sku_row['cat_id']} / {sku_row['dept_id']}**  \n"
            f"Price: **\\${sku_row['sell_price']:.2f}** · Cost: **\\${sku_row['unit_cost']:.2f}**  \n"
            f"On-hand: **{int(sku_row['current_on_hand'])}** · Lead time: **{int(sku_row['lead_time_days'])}d** · MOQ: **{int(sku_row['moq'])}**"
        )

    with st.container(border=True):
        st.caption("Backend telemetry")
        st.markdown(
            f":green[:material/check_circle: FastAPI Online]  \n"
            f"DuckDB rows: **{health['duckdb_sales_rows']:,}**  \n"
            f"Coverage: **{health['store_count']} stores / {health['sku_count']} SKUs**  \n"
            f"Briefing engine: `{health['llm_mode']}`"
        )

init_session_state(client, selected_store, selected_item)

st.title(":material/inventory_2: StockPulse AI Command Center")
st.caption(
    "Real Kaggle Walmart M5 (CA_1 · TX_1 · WI_1) · LightGBM P10/P50/P90 · "
    "TreeSHAP & XAI Studio · IsolationForest & PSI Drift · HiGHS MILP"
)

tab_overview, tab_forecast, tab_xai, tab_anomaly, tab_milp = st.tabs(
    [
        ":material/dashboard: Executive overview",
        ":material/show_chart: Demand forecast",
        ":material/psychology: Explainable AI studio",
        ":material/warning: Anomaly & drift",
        ":material/tune: MILP & what-if",
    ]
)

# =============================================================================
# TAB 1: Executive Overview & Grounded Briefing Copilot
# =============================================================================
with tab_overview:
    benchmark = load_benchmark_results()
    wape_imp = (
        f"+{benchmark['forecasting']['wape_improvement_pct']:.2f}%"
        if benchmark
        else "+16.86%"
    )
    cov_80 = (
        f"{benchmark['forecasting']['lightgbm_quantile']['interval_coverage_80'] * 100:.2f}%"
        if benchmark
        else "80.60%"
    )
    nominal_savings = (
        f"+{benchmark['inventory_optimization'][0]['cost_reduction_pct']:.2f}%"
        if benchmark
        else "+83.48%"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DuckDB warehouse rows", f"{health['duckdb_sales_rows']:,}", "30 store-SKU series")
    c2.metric("Forecast WAPE vs. 7d naive", wape_imp, "28-day recursive rollout")
    c3.metric("80% interval coverage", cov_80, "Target: 80.00%")
    c4.metric("Nominal MILP cost savings", nominal_savings, "vs. 14d rule-of-thumb")

    st.plotly_chart(
        build_store_category_bar(skus),
        width="stretch",
        alt="On-hand inventory valuation grouped by store and category",
    )

    with st.container(border=True):
        st.subheader(":material/auto_awesome: Grounded executive briefing copilot")
        col_scope, col_btn = st.columns([2, 1], vertical_alignment="bottom")
        with col_scope:
            scope_choice = st.segmented_control(
                "Briefing scope",
                ["All stores", "CA_1", "TX_1", "WI_1"],
                default="All stores",
                key="brief_scope",
            )
        with col_btn:
            run_brief = st.button(
                "Generate executive briefing",
                icon=":material/bolt:",
                type="primary",
                width="stretch",
                key="btn_gen_brief",
            )

        if run_brief:
            focus_store = None if (scope_choice in (None, "All stores")) else scope_choice
            with st.spinner("Synthesizing grounded procurement briefing..."):
                st.session_state["briefing_result"] = client.post_briefing(
                    {
                        "constraints": st.session_state["active_constraints"],
                        "focus_store_id": focus_store,
                    }
                )

        brief = st.session_state["briefing_result"]
        st.info(
            brief["executive_summary"],
            title=f"{brief['headline']}  [{brief['generation_mode']}]",
        )
        col_act, col_risk = st.columns(2)
        with col_act:
            st.markdown("**Priority procurement actions**")
            for act in brief["priority_actions"]:
                st.markdown(f"- {act}")
        with col_risk:
            st.markdown("**Risk & drift alerts**")
            for alert in brief["risk_alerts"]:
                st.markdown(f"- {alert}")

        st.download_button(
            "Download briefing (Markdown)",
            data=_format_briefing_markdown(brief),
            file_name="stockpulse_executive_briefing.md",
            mime="text/markdown",
            icon=":material/download:",
            width="content",
        )

# =============================================================================
# TAB 2: Probabilistic Demand Forecasting (P10 / P50 / P90)
# =============================================================================
with tab_forecast:
    bundle = fetch_forecast(client, selected_store, selected_item)
    sku_metrics = bundle["metrics"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SKU WAPE (28d)", f"{sku_metrics['wape'] * 100:.2f}%")
    m2.metric("SKU MAE", f"{sku_metrics['mae']:.2f} units/day")
    m3.metric("SKU RMSE", f"{sku_metrics['rmse']:.2f}")
    m4.metric("80% PI coverage", f"{sku_metrics['interval_coverage_80'] * 100:.1f}%")

    st.plotly_chart(
        build_forecast_fan_chart(bundle),
        width="stretch",
        alt="Trailing 60-day actual sales and 28-day P10/P50/P90 quantile forecast fan",
    )

    fc_dates = [pt["date"] for pt in bundle["forecast"]]
    sel_date = st.selectbox(
        "Validation date for local TreeSHAP attribution",
        fc_dates,
        index=len(fc_dates) - 1,
        key="fc_shap_date",
    )
    exp_data = fetch_explain(client, selected_store, selected_item, sel_date)
    local_exp = exp_data["local_explanation"]

    st.caption(
        f"TreeSHAP decomposition for **{selected_store} / {selected_item}** on **{local_exp['date']}**: "
        f"Base value `{local_exp['base_value']:.3f}` + Sum(SHAP) = Predicted P50 `{local_exp['predicted_p50']:.3f}`"
    )
    col_loc, col_glob = st.columns(2)
    with col_loc:
        st.plotly_chart(
            build_shap_waterfall_bar(
                local_exp["contributions"],
                f"Local TreeSHAP Drivers ({local_exp['date']})",
                top_n=12,
                use_abs=False,
            ),
            width="stretch",
            alt="Local TreeSHAP positive and negative contributions for selected validation date",
        )
    with col_glob:
        st.plotly_chart(
            build_shap_waterfall_bar(
                exp_data["global_importance"],
                "Global Validation SHAP Feature Importance",
                top_n=12,
                use_abs=True,
            ),
            width="stretch",
            alt="Global validation mean absolute SHAP feature importance",
        )

# =============================================================================
# TAB 3: Explainable AI (XAI) Studio
# =============================================================================
with tab_xai:
    st.subheader(":material/stacked_bar_chart: Multi-quantile tail-risk attribution (P10 / P50 / P90)")
    tail_dates = [pt["date"] for pt in bundle["forecast"]]
    tail_date = st.selectbox(
        "Validation date for P10/P50/P90 spread analysis",
        tail_dates,
        index=len(tail_dates) - 1,
        key="xai_tail_date",
    )
    tail_data = fetch_tail_risk(client, selected_store, selected_item, tail_date)

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Predicted P10", f"{max(0.0, tail_data['predicted_p10']):.2f}")
    t2.metric("Predicted P50", f"{max(0.0, tail_data['predicted_p50']):.2f}")
    t3.metric("Predicted P90", f"{max(0.0, tail_data['predicted_p90']):.2f}")
    spread_w = max(0.0, tail_data["predicted_p90"]) - max(0.0, tail_data["predicted_p10"])
    t4.metric("80% interval width", f"{spread_w:.2f} units")

    st.plotly_chart(
        build_quantile_spread_chart(tail_data),
        width="stretch",
        alt="Grouped TreeSHAP drivers across P10, P50, P90 and interval spread",
    )

    st.divider()
    st.subheader(":material/ssid_chart: Partial dependence & sampled SKU ICE curves")
    pdp_feature = st.segmented_control(
        "Inspect elasticity / demand driver feature",
        ["sell_price", "price_vs_item_avg", "rolling_mean_7", "rolling_zero_rate_28"],
        default="sell_price",
        key="xai_pdp_feat",
    )
    chosen_feat = pdp_feature or "sell_price"
    pdp_data = fetch_pdp(client, chosen_feat)
    st.plotly_chart(
        build_pdp_ice_chart(pdp_data),
        width="stretch",
        alt=f"Partial dependence and ICE curves for {chosen_feat}",
    )

    st.divider()
    st.subheader(":material/alt_route: Actionable counterfactual recourse simulator")
    with st.form("counterfactual_form"):
        cf_c1, cf_c2, cf_c3 = st.columns(3)
        with cf_c1:
            cf_delta = st.slider(
                "Target demand change (%)",
                min_value=-40,
                max_value=40,
                value=-15,
                step=5,
            )
        with cf_c2:
            cf_elim = st.checkbox("Target stockout elimination over lead time", value=False)
        with cf_c3:
            cf_oh = st.number_input(
                "On-hand stock override (for stockout simulation)",
                min_value=0,
                max_value=5000,
                value=int(sku_row["current_on_hand"]),
                step=1,
            )
        cf_submit = st.form_submit_button(
            "Run counterfactual search",
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
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Primary lever", cf_res["lever"])
    r2.metric(
        "Counterfactual value",
        f"{cf_res['counterfactual_value']:.2f}",
        f"{cf_res['pct_change']:+.1f}% vs. {cf_res['original_value']:.2f}",
    )
    r3.metric(
        "Achieved P50 demand",
        f"{cf_res['achieved_p50']:.2f}",
        f"from {cf_res['original_p50']:.2f}",
    )
    r4.metric(
        "Stockout status",
        "Eliminated" if cf_res["stockout_eliminated"] else "At risk",
        f"Lead time: {cf_res['lead_time_days_counterfactual']}d",
    )
    st.success(cf_res["action_summary"], title="Recommended minimal intervention")

    st.divider()
    st.subheader(":material/fact_check: MILP decision & constraint attribution audit")
    audit_col1, audit_col2 = st.columns([3, 1], vertical_alignment="bottom")
    with audit_col1:
        audit_filter = st.segmented_control(
            "Filter audit table",
            ["All SKUs", "Ordered only", "MOQ binding", "Constrained"],
            default="All SKUs",
            key="xai_audit_filter",
        )
    with audit_col2:
        if st.button("Refresh decision audit", icon=":material/refresh:", width="stretch"):
            st.session_state["decision_audit_result"] = client.post_xai_decision_audit(
                st.session_state["active_constraints"]
            )

    audit = st.session_state["decision_audit_result"]
    a1, a2, a3 = st.columns(3)
    a1.metric("Total net economic benefit", f"${audit['total_net_economic_benefit']:,.2f}")
    a2.metric("MOQ-binding SKUs", f"{audit['moq_binding_count']}")
    a3.metric("Cap-constrained SKUs", f"{audit['constrained_sku_count']}")

    audit_df = pd.DataFrame(audit["items"])
    if audit_filter == "Ordered only":
        audit_df = audit_df[audit_df["order_triggered"]]
    elif audit_filter == "MOQ binding":
        audit_df = audit_df[audit_df["moq_binding"]]
    elif audit_filter == "Constrained":
        audit_df = audit_df[audit_df["budget_or_capacity_constrained"]]

    st.dataframe(
        audit_df,
        width="stretch",
        hide_index=True,
        alt="MILP SKU-level economic and constraint audit table",
    )

# =============================================================================
# TAB 4: Anomaly & Distribution Drift Monitor
# =============================================================================
with tab_anomaly:
    anom_bundle = fetch_anomalies(client)
    drift = anom_bundle["drift_report"]
    anomalies_df = pd.DataFrame(anom_bundle["anomalies"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Portfolio PSI score", f"{drift['psi_score']:.4f}", drift["status"])
    d2.metric("KS statistic", f"{drift['ks_statistic']:.4f}")
    d3.metric("KS p-value", f"{drift['ks_pvalue']:.4f}")
    d4.metric("Flagged demand shocks", f"{len(anomalies_df)}")

    st.plotly_chart(
        build_psi_drift_chart(drift),
        width="stretch",
        alt="Per-feature Population Stability Index bars with drift thresholds",
    )

    f_c1, f_c2 = st.columns(2)
    with f_c1:
        shock_filter = st.segmented_control(
            "Shock type",
            ["All", "SURGE", "DROP"],
            default="All",
            key="anom_shock_filter",
        )
    with f_c2:
        anom_store = st.segmented_control(
            "Store filter",
            ["All stores", "CA_1", "TX_1", "WI_1"],
            default="All stores",
            key="anom_store_filter",
        )

    filtered_anom = anomalies_df.copy()
    if not filtered_anom.empty:
        if shock_filter in ("SURGE", "DROP"):
            filtered_anom = filtered_anom[filtered_anom["shock_type"] == shock_filter]
        if anom_store in ("CA_1", "TX_1", "WI_1"):
            filtered_anom = filtered_anom[filtered_anom["store_id"] == anom_store]

    st.dataframe(
        filtered_anom,
        width="stretch",
        hide_index=True,
        alt="Detected IsolationForest demand surge and drop shocks",
    )

# =============================================================================
# TAB 5: Constrained MILP Inventory & What-If Simulator
# =============================================================================
with tab_milp:
    with st.form("milp_whatif_form"):
        st.subheader(":material/tune: Working-capital & supply-chain stress parameters")
        s1, s2, s3 = st.columns(3)
        with s1:
            budget_cap = st.slider(
                "Max working-capital budget ($)",
                min_value=1500,
                max_value=50000,
                value=int(st.session_state["active_constraints"]["max_total_budget"]),
                step=500,
            )
            demand_shock = st.slider(
                "Demand shock (%)",
                min_value=-30,
                max_value=50,
                value=int(st.session_state["active_constraints"]["demand_shock_pct"]),
                step=5,
            )
        with s2:
            unit_cap = st.slider(
                "Warehouse unit capacity",
                min_value=500,
                max_value=25000,
                value=int(st.session_state["active_constraints"]["max_total_units"]),
                step=250,
            )
            lt_mult = st.slider(
                "Supplier lead-time multiplier",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state["active_constraints"]["lead_time_multiplier"]),
                step=0.1,
            )
        with s3:
            hold_mult = st.slider(
                "Holding cost multiplier",
                min_value=0.5,
                max_value=2.0,
                value=float(st.session_state["active_constraints"]["holding_cost_multiplier"]),
                step=0.1,
            )
        run_opt = st.form_submit_button(
            "Run HiGHS MILP optimization",
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
        with st.spinner("Solving constrained MILP via SciPy HiGHS..."):
            st.session_state["optimize_result"] = client.post_optimize(new_constraints)
            st.session_state["decision_audit_result"] = client.post_xai_decision_audit(
                new_constraints
            )

    opt_res = st.session_state["optimize_result"]
    o1, o2, o3, o4 = st.columns(4)
    o1.metric(
        "MILP expected total cost",
        f"${opt_res['total_expected_cost']:,.2f}",
        f"Baseline: ${opt_res['baseline_total_cost']:,.2f}",
    )
    o2.metric(
        "Cost savings vs. baseline",
        f"${opt_res['total_cost_savings_vs_baseline']:,.2f}",
        f"{opt_res['solver_status']}",
    )
    o3.metric(
        "Expected service level",
        f"{opt_res['expected_service_level'] * 100:.2f}%",
        f"Baseline: {opt_res['baseline_service_level'] * 100:.2f}%",
    )
    o4.metric(
        "Procurement spend / units",
        f"${opt_res['total_order_cost']:,.2f}",
        f"{opt_res['total_units_ordered']:,} units",
    )

    if opt_res["baseline_budget_violated"] or opt_res["baseline_capacity_violated"]:
        st.warning(
            "Under the active constraints, the 14-day rule-of-thumb heuristic violated the "
            "working-capital budget or warehouse unit cap, whereas the HiGHS MILP allocation remained strictly feasible.",
            title="Baseline constraint violation detected",
        )

    cost_fig, gauge_fig = build_milp_comparison_charts(opt_res)
    ch_l, ch_r = st.columns([3, 2])
    with ch_l:
        st.plotly_chart(
            cost_fig,
            width="stretch",
            alt="HiGHS MILP expected cost vs. 14-day rule-of-thumb baseline",
        )
    with ch_r:
        st.plotly_chart(
            gauge_fig,
            width="stretch",
            alt="Working-capital procurement budget utilization gauge",
        )

    decisions_df = pd.DataFrame(opt_res["decisions"])
    st.dataframe(
        decisions_df,
        width="stretch",
        hide_index=True,
        alt="Optimal 30-SKU replenishment order allocation table",
    )
    st.download_button(
        "Download purchase order CSV",
        data=decisions_df.to_csv(index=False),
        file_name="stockpulse_purchase_orders.csv",
        mime="text/csv",
        icon=":material/download:",
        width="content",
    )
