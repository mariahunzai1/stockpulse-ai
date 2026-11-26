"""Expanded Explainable AI Studio built atop QuantileForecaster, DemandExplainer & MILP."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence

from core.forecaster import QUANTILE_ALPHAS, QuantileForecaster
from core.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    DecisionAuditItem,
    DecisionAuditResponse,
    IceCurve,
    OptimizationResult,
    PdpIceResponse,
    QuantileSpreadExplanation,
    ShapContribution,
    SkuPolicyRecord,
    SpreadDriver,
)

SUPPORTED_PDP_FEATURES: tuple[str, ...] = (
    "sell_price",
    "price_vs_item_avg",
    "rolling_mean_7",
    "rolling_zero_rate_28",
)


class ExplainableAIStudio:
    """Multi-quantile SHAP, PDP/ICE, Counterfactual, and MILP Decision Audit engine."""

    def __init__(self, forecaster: QuantileForecaster) -> None:
        if len(forecaster.models) != len(QUANTILE_ALPHAS):
            raise RuntimeError("QuantileForecaster must have P10, P50, and P90 models fitted")
        self.forecaster = forecaster
        self.feature_cols = list(forecaster.feature_cols)
        self._explainers: dict[float, shap.TreeExplainer] = {
            alpha: shap.TreeExplainer(forecaster.models[alpha])
            for alpha in QUANTILE_ALPHAS
        }

    @staticmethod
    def _extract_expected_value(expected_value: object) -> float:
        return float(np.asarray(expected_value, dtype=np.float64).ravel()[0])

    @staticmethod
    def _select_row(
        featured_df: pd.DataFrame,
        store_id: str,
        item_id: str,
        target_date: dt.date | str | None,
    ) -> tuple[pd.DataFrame, dt.date]:
        work = featured_df.copy()
        work["_date_only"] = pd.to_datetime(work["date"]).dt.date
        sku_rows = work[(work["store_id"] == store_id) & (work["item_id"] == item_id)]
        if sku_rows.empty:
            raise KeyError(f"No feature rows found for store={store_id}, item={item_id}")

        if target_date is None:
            resolved_date = max(sku_rows["_date_only"].tolist())
        else:
            resolved_date = pd.Timestamp(target_date).date()

        match = sku_rows[sku_rows["_date_only"] == resolved_date]
        if match.empty:
            raise KeyError(
                f"No feature row found for store={store_id}, item={item_id}, date={resolved_date}"
            )
        return match.iloc[[0]], resolved_date

    def explain_quantile_spread(
        self,
        featured_df: pd.DataFrame,
        store_id: str,
        item_id: str,
        target_date: dt.date | str | None = None,
    ) -> QuantileSpreadExplanation:
        """Compute exact TreeSHAP attributions across P10, P50, and P90 + spread drivers."""
        matched_row, resolved_date = self._select_row(
            featured_df, store_id, item_id, target_date
        )
        x_row = matched_row[self.feature_cols]
        row_values = x_row.iloc[0].to_numpy(dtype=np.float64)

        bases: dict[float, float] = {}
        preds: dict[float, float] = {}
        shap_arrays: dict[float, np.ndarray] = {}
        contribs_by_alpha: dict[float, list[ShapContribution]] = {}

        for alpha in QUANTILE_ALPHAS:
            explainer = self._explainers[alpha]
            svals = np.asarray(explainer.shap_values(x_row), dtype=np.float64).reshape(-1)
            bases[alpha] = self._extract_expected_value(explainer.expected_value)
            preds[alpha] = float(self.forecaster.models[alpha].predict(x_row)[0])
            shap_arrays[alpha] = svals

            contribs = [
                ShapContribution(
                    feature=feat,
                    feature_value=round(float(val), 6),
                    shap_value=float(sv),
                )
                for feat, val, sv in zip(self.feature_cols, row_values, svals)
            ]
            contribs.sort(key=lambda c: abs(c.shap_value), reverse=True)
            contribs_by_alpha[alpha] = contribs

        spread_drivers: list[SpreadDriver] = []
        for idx, feat in enumerate(self.feature_cols):
            sp10 = float(shap_arrays[0.10][idx])
            sp50 = float(shap_arrays[0.50][idx])
            sp90 = float(shap_arrays[0.90][idx])
            spread_drivers.append(
                SpreadDriver(
                    feature=feat,
                    feature_value=round(float(row_values[idx]), 6),
                    shap_p10=sp10,
                    shap_p50=sp50,
                    shap_p90=sp90,
                    spread_impact=sp90 - sp10,
                )
            )
        spread_drivers.sort(key=lambda d: abs(d.spread_impact), reverse=True)

        return QuantileSpreadExplanation(
            store_id=store_id,
            item_id=item_id,
            date=resolved_date,
            base_p10=bases[0.10],
            base_p50=bases[0.50],
            base_p90=bases[0.90],
            predicted_p10=preds[0.10],
            predicted_p50=preds[0.50],
            predicted_p90=preds[0.90],
            p10_contributions=contribs_by_alpha[0.10],
            p50_contributions=contribs_by_alpha[0.50],
            p90_contributions=contribs_by_alpha[0.90],
            uncertainty_spread_drivers=spread_drivers,
        )

    def compute_pdp_ice(
        self,
        val_df: pd.DataFrame,
        feature_name: str,
        max_ice_curves: int = 12,
        grid_resolution: int = 20,
    ) -> PdpIceResponse:
        """Compute Partial Dependence & sampled ICE curves across the validation matrix."""
        if feature_name not in SUPPORTED_PDP_FEATURES:
            raise ValueError(
                f"Unsupported PDP feature '{feature_name}'. Choose from {SUPPORTED_PDP_FEATURES}"
            )

        work = val_df.sort_values(["store_id", "item_id", "date"], kind="mergesort").reset_index(
            drop=True
        )
        x_val = work[self.feature_cols]

        pdp_res = partial_dependence(
            self.forecaster.models[0.50],
            x_val,
            features=[feature_name],
            kind="both",
            grid_resolution=grid_resolution,
            method="brute",
        )
        grid_values = [round(float(v), 6) for v in pdp_res["grid_values"][0].tolist()]
        avg_pdp = [round(float(max(0.0, v)), 6) for v in pdp_res["average"][0].tolist()]

        # Select one latest validation observation per unique (store_id, item_id) up to max_ice_curves
        latest_indices = (
            work.groupby(["store_id", "item_id"], sort=True)
            .tail(1)
            .index.tolist()[:max_ice_curves]
        )
        individual = pdp_res["individual"][0]
        ice_curves: list[IceCurve] = []
        for idx in latest_indices:
            row = work.iloc[idx]
            curve_vals = [round(float(max(0.0, v)), 6) for v in individual[idx].tolist()]
            ice_curves.append(
                IceCurve(
                    store_id=str(row["store_id"]),
                    item_id=str(row["item_id"]),
                    values=curve_vals,
                )
            )

        return PdpIceResponse(
            feature=feature_name,
            grid_values=grid_values,
            average_pdp=avg_pdp,
            ice_curves=ice_curves,
        )

    def generate_counterfactual(
        self,
        featured_df: pd.DataFrame,
        policies: list[SkuPolicyRecord],
        request: CounterfactualRequest,
    ) -> CounterfactualResponse:
        """Search bounded sell_price (-25%..+25%) & lead_time (-50%..0%) interventions."""
        matched_row, resolved_date = self._select_row(
            featured_df, request.store_id, request.item_id, request.date
        )
        pol_map = {(p.store_id, p.item_id): p for p in policies}
        key = (request.store_id, request.item_id)
        if key not in pol_map:
            raise KeyError(f"Policy not found for {key}")
        policy = pol_map[key]

        base_features = matched_row[self.feature_cols].iloc[0].to_dict()
        orig_price = float(base_features["sell_price"])
        orig_price_avg_ratio = float(base_features["price_vs_item_avg"])
        orig_p50 = float(
            self.forecaster.predict_quantiles(matched_row[self.feature_cols])[0, 1]
        )

        on_hand = (
            request.current_on_hand_override
            if request.current_on_hand_override is not None
            else policy.current_on_hand
        )
        orig_lt = policy.lead_time_days
        if request.eliminate_stockout:
            stock_daily_cap = float(on_hand) / max(orig_lt, 1)
            target_p50 = min(orig_p50, stock_daily_cap)
        else:
            target_p50 = max(0.0, orig_p50 * (1.0 + request.target_demand_delta_pct / 100.0))

        # Vectorized evaluation of sell_price perturbations in [-25%, +25%]
        deltas = np.linspace(-0.25, 0.25, 51)
        cand_rows: list[dict[str, Any]] = []
        implied_hist_avg = orig_price / max(orig_price_avg_ratio, 1e-6)
        orig_mom = float(base_features["price_momentum_7d"])
        price_7d_ago = orig_price / max(1.0 + orig_mom, 1e-6)

        for d in deltas:
            new_p = max(0.05, orig_price * (1.0 + float(d)))
            cand = dict(base_features)
            cand["sell_price"] = new_p
            cand["price_vs_item_avg"] = new_p / max(implied_hist_avg, 1e-6)
            cand["price_momentum_7d"] = (new_p - price_7d_ago) / max(price_7d_ago, 1e-6)
            cand_rows.append(cand)

        cand_df = pd.DataFrame(cand_rows, columns=self.feature_cols)
        cand_p50s = self.forecaster.predict_quantiles(cand_df)[:, 1]

        # Select candidate minimizing distance to target_p50, breaking ties by minimal |delta|
        if request.eliminate_stockout or request.target_demand_delta_pct <= 0.0:
            feasible_idx = np.where(cand_p50s <= target_p50 + 1e-6)[0]
        else:
            feasible_idx = np.where(cand_p50s >= target_p50 - 1e-6)[0]

        if len(feasible_idx) > 0:
            best_idx = int(min(feasible_idx, key=lambda idx: abs(float(deltas[idx]))))
            target_achieved = True
        else:
            best_idx = int(
                min(
                    range(len(deltas)),
                    key=lambda idx: (
                        abs(float(cand_p50s[idx]) - target_p50),
                        abs(float(deltas[idx])),
                    ),
                )
            )
            target_achieved = bool(abs(float(cand_p50s[best_idx]) - target_p50) <= 0.25)

        best_delta = float(deltas[best_idx])
        cf_price = round(max(0.05, orig_price * (1.0 + best_delta)), 4)
        achieved_p50 = round(float(cand_p50s[best_idx]), 4)

        # Evaluate lead_time compression (-50% to 0%) if eliminate_stockout is active
        cf_lt = orig_lt
        stockout_eliminated = (achieved_p50 * cf_lt) <= float(on_hand)
        if request.eliminate_stockout and not stockout_eliminated:
            min_lt = max(1, int(round(orig_lt * 0.5)))
            for lt_cand in range(orig_lt - 1, min_lt - 1, -1):
                cf_lt = lt_cand
                if (achieved_p50 * cf_lt) <= float(on_hand):
                    stockout_eliminated = True
                    target_achieved = True
                    break

        if cf_lt != orig_lt and abs(best_delta) < 1e-6:
            lever = "lead_time_days"
            orig_val = float(orig_lt)
            cf_val = float(cf_lt)
            pct_chg = round(((cf_lt - orig_lt) / max(orig_lt, 1)) * 100.0, 2)
        elif cf_lt != orig_lt:
            lever = "sell_price+lead_time_days"
            orig_val = round(orig_price, 4)
            cf_val = cf_price
            pct_chg = round(best_delta * 100.0, 2)
        else:
            lever = "sell_price"
            orig_val = round(orig_price, 4)
            cf_val = cf_price
            pct_chg = round(best_delta * 100.0, 2)

        action_summary = (
            f"Adjust {lever} from {orig_val:.2f} to {cf_val:.2f} ({pct_chg:+.1f}%)"
            + (
                f" and compress lead time {orig_lt}d->{cf_lt}d"
                if cf_lt != orig_lt and lever != "lead_time_days"
                else ""
            )
            + f": shifts P50 demand from {orig_p50:.2f} to {achieved_p50:.2f} units/day "
            f"(target {target_p50:.2f})."
        )

        return CounterfactualResponse(
            store_id=request.store_id,
            item_id=request.item_id,
            date=resolved_date,
            lever=lever,
            original_value=orig_val,
            counterfactual_value=cf_val,
            pct_change=pct_chg,
            original_p50=round(orig_p50, 4),
            achieved_p50=achieved_p50,
            target_p50=round(target_p50, 4),
            target_achieved=target_achieved,
            stockout_eliminated=stockout_eliminated,
            lead_time_days_original=orig_lt,
            lead_time_days_counterfactual=cf_lt,
            action_summary=action_summary,
        )

    @staticmethod
    def explain_optimization_decisions(
        optimization_result: OptimizationResult,
        policies: list[SkuPolicyRecord],
    ) -> DecisionAuditResponse:
        """Audit every SKU MILP decision with economic, MOQ, and cap attribution."""
        pol_map = {(p.store_id, p.item_id): p for p in policies}
        constraints = optimization_result.constraints
        budget_tight = (
            optimization_result.total_order_cost >= 0.98 * constraints.max_total_budget
        )
        capacity_tight = (
            optimization_result.total_units_ordered >= 0.98 * constraints.max_total_units
        )
        global_cap_binding = budget_tight or capacity_tight

        audit_items: list[DecisionAuditItem] = []
        moq_count = 0
        constrained_count = 0
        total_benefit = 0.0

        for dec in optimization_result.decisions:
            pol = pol_map[(dec.store_id, dec.item_id)]
            denom_cost = max(dec.expected_holding_cost + dec.expected_stockout_cost, 1e-6)
            risk_ratio = float(np.clip(dec.expected_stockout_cost / denom_cost, 0.0, 1.0))

            moq_binding = bool(dec.order_triggered and dec.recommended_order_qty == pol.moq)
            if moq_binding:
                moq_count += 1

            sku_constrained = bool(
                global_cap_binding
                and dec.recommended_order_qty < pol.max_order_qty
                and dec.expected_stockout_cost > 0.01
            )
            if sku_constrained:
                constrained_count += 1

            total_benefit += dec.cost_savings_vs_baseline

            if not dec.order_triggered:
                rationale = (
                    f"No order triggered (on-hand {dec.current_on_hand} covers "
                    f"{dec.forecast_demand_horizon:.1f} P50 horizon demand; MOQ={pol.moq} "
                    f"exceeds marginal stockout savings)."
                )
            elif sku_constrained:
                rationale = (
                    f"Ordered {dec.recommended_order_qty} units under active global budget/capacity "
                    f"cap (stockout risk share={risk_ratio:.0%}, net savings=${dec.cost_savings_vs_baseline:.2f})."
                )
            elif moq_binding:
                rationale = (
                    f"Ordered exact MOQ={pol.moq} units to protect against P90 tail risk "
                    f"(+{dec.safety_stock_p90:.1f} buffer), saving ${dec.cost_savings_vs_baseline:.2f} vs. baseline."
                )
            else:
                rationale = (
                    f"Ordered {dec.recommended_order_qty} units (vs. baseline {dec.baseline_order_qty}) "
                    f"balancing ${dec.expected_holding_cost:.2f} holding vs. ${dec.expected_stockout_cost:.2f} "
                    f"stockout exposure (+${dec.cost_savings_vs_baseline:.2f} net benefit)."
                )

            audit_items.append(
                DecisionAuditItem(
                    store_id=dec.store_id,
                    item_id=dec.item_id,
                    dept_id=dec.dept_id,
                    recommended_order_qty=dec.recommended_order_qty,
                    order_triggered=dec.order_triggered,
                    net_economic_benefit=round(dec.cost_savings_vs_baseline, 4),
                    stockout_risk_ratio=round(risk_ratio, 4),
                    moq_binding=moq_binding,
                    budget_or_capacity_constrained=sku_constrained,
                    decision_rationale=rationale,
                )
            )

        return DecisionAuditResponse(
            solver_status=optimization_result.solver_status,
            total_net_economic_benefit=round(total_benefit, 4),
            moq_binding_count=moq_count,
            constrained_sku_count=constrained_count,
            items=audit_items,
        )
