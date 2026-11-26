"""Constrained MILP inventory optimization engine powered by SciPy HiGHS."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from core.schemas import (
    ForecastPoint,
    OptimizationConstraints,
    OptimizationResult,
    SkuOrderDecision,
    SkuPolicyRecord,
)

SCENARIO_WEIGHTS: tuple[float, float, float] = (0.20, 0.60, 0.20)
REVIEW_PERIOD_DAYS = 7


class InventoryOptimizer:
    """Multi-SKU constrained MILP replenishment optimizer over P10/P50/P90 demand scenarios."""

    @staticmethod
    def _horizon_scenario_demands(
        points: list[ForecastPoint],
        horizon_days: int,
        demand_shock_pct: float,
    ) -> tuple[float, float, float]:
        """Sum P10/P50/P90 daily forecasts over `horizon_days` with shock scaling."""
        ordered = sorted(points, key=lambda p: p.date)
        n_avail = len(ordered)
        if n_avail == 0:
            return 0.0, 0.0, 0.0

        take = min(horizon_days, n_avail)
        p10_sum = sum(p.p10 for p in ordered[:take])
        p50_sum = sum(p.p50 for p in ordered[:take])
        p90_sum = sum(p.p90 for p in ordered[:take])

        if horizon_days > n_avail:
            scale_extrap = horizon_days / n_avail
            p10_sum *= scale_extrap
            p50_sum *= scale_extrap
            p90_sum *= scale_extrap

        shock_mult = 1.0 + (demand_shock_pct / 100.0)
        return (
            max(0.0, p10_sum * shock_mult),
            max(0.0, p50_sum * shock_mult),
            max(0.0, p90_sum * shock_mult),
        )

    @staticmethod
    def _evaluate_order_economics(
        order_qty: int,
        on_hand: int,
        scenario_demands: tuple[float, float, float],
        holding_cost_per_unit_horizon: float,
        stockout_penalty: float,
    ) -> tuple[float, float, float]:
        """Return `(expected_holding_cost, expected_stockout_cost, expected_stockout_units)`."""
        stock_pos = float(on_hand + order_qty)
        exp_hold = 0.0
        exp_stockout_cost = 0.0
        exp_stockout_units = 0.0
        for w, demand in zip(SCENARIO_WEIGHTS, scenario_demands):
            excess = max(0.0, stock_pos - demand)
            shortfall = max(0.0, demand - stock_pos)
            exp_hold += w * holding_cost_per_unit_horizon * excess
            exp_stockout_cost += w * stockout_penalty * shortfall
            exp_stockout_units += w * shortfall
        return exp_hold, exp_stockout_cost, exp_stockout_units

    @staticmethod
    def _rule_of_thumb_order_qty(
        policy: SkuPolicyRecord,
        trailing_daily_mean: float,
    ) -> int:
        """Compute 14-day trailing-velocity rule-of-thumb order quantity snapped to MOQ bounds."""
        raw_need = int(math.ceil(14.0 * max(0.0, trailing_daily_mean))) - policy.current_on_hand
        if raw_need <= 0:
            return 0
        return int(min(policy.max_order_qty, max(policy.moq, raw_need)))

    def optimize(
        self,
        policies: list[SkuPolicyRecord],
        forecast_points: list[ForecastPoint],
        constraints: OptimizationConstraints,
        trailing_28d_means: dict[tuple[str, str], float] | None = None,
    ) -> OptimizationResult:
        """Solve global budget- and capacity-constrained MILP across all SKU policies."""
        if not policies:
            raise ValueError("policies list cannot be empty")

        grouped_fc: dict[tuple[str, str], list[ForecastPoint]] = defaultdict(list)
        for pt in forecast_points:
            grouped_fc[(pt.store_id, pt.item_id)].append(pt)

        n_skus = len(policies)
        stride = 8  # [Q_i, y_i, e_0, e_1, e_2, u_0, u_1, u_2]
        n_vars = n_skus * stride

        c_obj = np.zeros(n_vars, dtype=np.float64)
        lb = np.zeros(n_vars, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)
        integrality = np.zeros(n_vars, dtype=np.int32)

        # 8 local constraints per SKU + 2 global constraints (budget + capacity)
        n_rows = (8 * n_skus) + 2
        a_mat = np.zeros((n_rows, n_vars), dtype=np.float64)
        con_lb = np.full(n_rows, -np.inf, dtype=np.float64)
        con_ub = np.full(n_rows, np.inf, dtype=np.float64)

        sku_metadata: list[tuple[SkuPolicyRecord, int, tuple[float, float, float], float]] = []

        for i, pol in enumerate(policies):
            key = (pol.store_id, pol.item_id)
            pts = grouped_fc.get(key, [])
            if not pts:
                raise KeyError(f"Missing forecast points for policy {key}")

            horizon_days = max(
                1,
                int(round(pol.lead_time_days * constraints.lead_time_multiplier))
                + REVIEW_PERIOD_DAYS,
            )
            scen_demands = self._horizon_scenario_demands(
                pts, horizon_days, constraints.demand_shock_pct
            )
            hold_per_unit = (
                pol.daily_holding_cost * constraints.holding_cost_multiplier * horizon_days
            )
            sku_metadata.append((pol, horizon_days, scen_demands, hold_per_unit))

            base = i * stride
            # Q_i integer in [0, max_order_qty], y_i binary in {0, 1}
            ub[base + 0] = float(pol.max_order_qty)
            ub[base + 1] = 1.0
            integrality[base + 0] = 1
            integrality[base + 1] = 1

            for s in range(3):
                c_obj[base + 2 + s] = SCENARIO_WEIGHTS[s] * hold_per_unit
                c_obj[base + 5 + s] = SCENARIO_WEIGHTS[s] * pol.stockout_penalty

            r0 = i * 8
            # 1) Q_i - moq_i * y_i >= 0
            a_mat[r0, base + 0] = 1.0
            a_mat[r0, base + 1] = -float(pol.moq)
            con_lb[r0] = 0.0

            # 2) Q_i - max_order_qty_i * y_i <= 0
            a_mat[r0 + 1, base + 0] = 1.0
            a_mat[r0 + 1, base + 1] = -float(pol.max_order_qty)
            con_ub[r0 + 1] = 0.0

            # 3..8) Scenario inventory balance: e_{i,s} - Q_i >= on_hand - D_{i,s}; u_{i,s} + Q_i >= D_{i,s} - on_hand
            for s in range(3):
                row_e = r0 + 2 + s
                a_mat[row_e, base + 2 + s] = 1.0
                a_mat[row_e, base + 0] = -1.0
                con_lb[row_e] = float(pol.current_on_hand) - scen_demands[s]

                row_u = r0 + 5 + s
                a_mat[row_u, base + 5 + s] = 1.0
                a_mat[row_u, base + 0] = 1.0
                con_lb[row_u] = scen_demands[s] - float(pol.current_on_hand)

            # Global budget & unit capacity rows
            budget_row = 8 * n_skus
            units_row = budget_row + 1
            a_mat[budget_row, base + 0] = float(pol.unit_cost)
            a_mat[units_row, base + 0] = 1.0

        con_ub[8 * n_skus] = float(constraints.max_total_budget)
        con_ub[8 * n_skus + 1] = float(constraints.max_total_units)

        res = milp(
            c=c_obj,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            constraints=LinearConstraint(a_mat, con_lb, con_ub),
        )
        if not res.success or res.x is None:
            raise RuntimeError(f"MILP solver failed (status={res.status}): {res.message}")

        decisions: list[SkuOrderDecision] = []
        total_order_cost = 0.0
        total_units_ordered = 0
        total_exp_hold = 0.0
        total_exp_stockout = 0.0
        total_exp_stockout_units = 0.0

        baseline_order_cost = 0.0
        baseline_units = 0
        baseline_total_cost = 0.0
        baseline_stockout_units = 0.0
        total_p50_demand = 0.0

        for i, (pol, horizon_days, scen_demands, hold_per_unit) in enumerate(sku_metadata):
            base = i * stride
            q_opt = int(round(float(res.x[base + 0])))
            triggered = q_opt > 0

            opt_hold, opt_stock, opt_short_units = self._evaluate_order_economics(
                order_qty=q_opt,
                on_hand=pol.current_on_hand,
                scenario_demands=scen_demands,
                holding_cost_per_unit_horizon=hold_per_unit,
                stockout_penalty=pol.stockout_penalty,
            )

            key = (pol.store_id, pol.item_id)
            if trailing_28d_means is not None and key in trailing_28d_means:
                daily_mean = trailing_28d_means[key]
            else:
                daily_mean = scen_demands[1] / max(horizon_days, 1)

            q_base = self._rule_of_thumb_order_qty(pol, daily_mean)
            base_hold, base_stock, base_short_units = self._evaluate_order_economics(
                order_qty=q_base,
                on_hand=pol.current_on_hand,
                scenario_demands=scen_demands,
                holding_cost_per_unit_horizon=hold_per_unit,
                stockout_penalty=pol.stockout_penalty,
            )

            opt_total_sku = opt_hold + opt_stock
            base_total_sku = base_hold + base_stock
            order_cost = q_opt * pol.unit_cost

            total_order_cost += order_cost
            total_units_ordered += q_opt
            total_exp_hold += opt_hold
            total_exp_stockout += opt_stock
            total_exp_stockout_units += opt_short_units

            baseline_order_cost += q_base * pol.unit_cost
            baseline_units += q_base
            baseline_total_cost += base_total_sku
            baseline_stockout_units += base_short_units
            total_p50_demand += scen_demands[1]

            decisions.append(
                SkuOrderDecision(
                    store_id=pol.store_id,
                    item_id=pol.item_id,
                    dept_id=pol.dept_id,
                    current_on_hand=pol.current_on_hand,
                    forecast_demand_horizon=round(scen_demands[1], 4),
                    safety_stock_p90=round(max(0.0, scen_demands[2] - scen_demands[1]), 4),
                    recommended_order_qty=q_opt,
                    order_triggered=triggered,
                    unit_cost=pol.unit_cost,
                    order_cost=round(order_cost, 4),
                    expected_holding_cost=round(opt_hold, 4),
                    expected_stockout_cost=round(opt_stock, 4),
                    baseline_order_qty=q_base,
                    cost_savings_vs_baseline=round(base_total_sku - opt_total_sku, 4),
                )
            )

        denom_demand = max(total_p50_demand, 1e-9)
        exp_service = float(np.clip(1.0 - (total_exp_stockout_units / denom_demand), 0.0, 1.0))
        base_service = float(np.clip(1.0 - (baseline_stockout_units / denom_demand), 0.0, 1.0))
        total_expected_cost = total_exp_hold + total_exp_stockout

        return OptimizationResult(
            constraints=constraints,
            solver_status="OPTIMAL",
            total_order_cost=round(total_order_cost, 4),
            total_units_ordered=total_units_ordered,
            total_expected_holding_cost=round(total_exp_hold, 4),
            total_expected_stockout_cost=round(total_exp_stockout, 4),
            total_expected_cost=round(total_expected_cost, 4),
            baseline_total_cost=round(baseline_total_cost, 4),
            total_cost_savings_vs_baseline=round(baseline_total_cost - total_expected_cost, 4),
            expected_service_level=round(exp_service, 6),
            baseline_service_level=round(base_service, 6),
            baseline_budget_violated=bool(baseline_order_cost > constraints.max_total_budget + 1e-6),
            baseline_capacity_violated=bool(baseline_units > constraints.max_total_units),
            decisions=decisions,
        )
