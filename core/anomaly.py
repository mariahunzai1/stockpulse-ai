"""Demand anomaly detection (IsolationForest) and PSI/KS distribution drift monitoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.schemas import AnomalyRecord, DriftReport

DRIFT_FEATURES: tuple[str, ...] = (
    "sales",
    "sell_price",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_zero_rate_28",
    "price_momentum_7d",
)


def _compute_feature_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index between reference and current 1D arrays."""
    ref_clean = reference[np.isfinite(reference)]
    cur_clean = current[np.isfinite(current)]
    if len(ref_clean) < 2 or len(cur_clean) < 2:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref_clean, quantiles))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    cur_counts, _ = np.histogram(cur_clean, bins=edges)

    ref_pct = np.clip(ref_counts / max(len(ref_clean), 1), eps, None)
    cur_pct = np.clip(cur_counts / max(len(cur_clean), 1), eps, None)
    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


class AnomalyAndDriftDetector:
    """IsolationForest demand shock detector and PSI/KS distribution drift monitor."""

    def __init__(self, contamination: float = 0.04, random_state: int = 42) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=150,
            random_state=random_state,
            n_jobs=1,
        )
        self._is_fitted = False

    @staticmethod
    def _build_anomaly_features(
        df: pd.DataFrame,
        expected_p50: Sequence[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct scaled feature matrix and diagnostic ratios for anomaly scoring."""
        actual = df["sales"].to_numpy(dtype=np.float64)
        p50 = np.asarray(expected_p50, dtype=np.float64)
        if len(actual) != len(p50):
            raise ValueError(
                f"Length mismatch between df ({len(actual)}) and expected_p50 ({len(p50)})"
            )

        rolling_std = (
            df["rolling_std_7"].to_numpy(dtype=np.float64)
            if "rolling_std_7" in df.columns
            else np.ones(len(df), dtype=np.float64)
        )
        rolling_mean = (
            df["rolling_mean_28"].to_numpy(dtype=np.float64)
            if "rolling_mean_28" in df.columns
            else np.maximum(p50, 0.5)
        )
        price_mom = (
            df["price_momentum_7d"].to_numpy(dtype=np.float64)
            if "price_momentum_7d" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )

        residual = actual - p50
        z_residual = residual / np.maximum(rolling_std, 1.0)
        volatility_ratio = actual / np.maximum(rolling_mean, 0.5)

        matrix = np.column_stack([residual, z_residual, volatility_ratio, price_mom])
        return matrix, z_residual, volatility_ratio

    def fit(
        self,
        train_df: pd.DataFrame,
        expected_p50: Sequence[float] | np.ndarray,
    ) -> AnomalyAndDriftDetector:
        """Fit IsolationForest on historical demand residuals, volatility, and price momentum."""
        matrix, _, _ = self._build_anomaly_features(train_df, expected_p50)
        scaled = self._scaler.fit_transform(matrix)
        self._iso_forest.fit(scaled)
        self._is_fitted = True
        return self

    def detect(
        self,
        df: pd.DataFrame,
        expected_p50: Sequence[float] | np.ndarray,
    ) -> list[AnomalyRecord]:
        """Score observations for demand surges/drops and attach human-readable explanations."""
        if not self._is_fitted:
            raise RuntimeError("AnomalyAndDriftDetector must be fitted before calling detect()")

        matrix, z_residual, vol_ratio = self._build_anomaly_features(df, expected_p50)
        scaled = self._scaler.transform(matrix)
        preds = self._iso_forest.predict(scaled)
        raw_scores = -self._iso_forest.decision_function(scaled)

        actual = df["sales"].to_numpy(dtype=np.float64)
        p50 = np.asarray(expected_p50, dtype=np.float64)
        dates = pd.to_datetime(df["date"]).dt.date.tolist()
        stores = df["store_id"].astype(str).tolist()
        items = df["item_id"].astype(str).tolist()
        events = (
            df["event_name_1"].astype(str).tolist()
            if "event_name_1" in df.columns
            else ["None"] * len(df)
        )
        snaps = df["snap"].astype(int).tolist() if "snap" in df.columns else [0] * len(df)

        records: list[AnomalyRecord] = []
        for idx in range(len(df)):
            is_anom = bool(preds[idx] == -1)
            act_val = float(actual[idx])
            exp_val = float(p50[idx])
            z_val = float(z_residual[idx])
            vr = float(vol_ratio[idx])

            if is_anom and act_val > exp_val:
                shock_type = "SURGE"
                drivers: list[str] = []
                if events[idx] != "None":
                    drivers.append(f"event={events[idx]}")
                if snaps[idx] == 1:
                    drivers.append("SNAP active")
                driver_note = f" ({', '.join(drivers)})" if drivers else ""
                reason = (
                    f"Demand surge: actual {act_val:.0f} vs P50 {exp_val:.1f} "
                    f"({z_val:+.2f}σ, {vr:.2f}x 28d avg){driver_note}."
                )
            elif is_anom and act_val < exp_val:
                shock_type = "DROP"
                reason = (
                    f"Demand drop: actual {act_val:.0f} vs P50 {exp_val:.1f} "
                    f"({z_val:+.2f}σ, {vr:.2f}x 28d avg)."
                )
            else:
                shock_type = "NORMAL"
                reason = f"Within expected demand regime (actual {act_val:.0f}, P50 {exp_val:.1f})."

            records.append(
                AnomalyRecord(
                    date=dates[idx],
                    store_id=stores[idx],
                    item_id=items[idx],
                    actual_sales=act_val,
                    expected_p50=exp_val,
                    anomaly_score=round(float(raw_scores[idx]), 6),
                    is_anomaly=is_anom,
                    shock_type=shock_type,
                    reason=reason,
                )
            )
        return records

    @staticmethod
    def compute_drift_report(
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Sequence[str] | None = None,
    ) -> DriftReport:
        """Compare recent/validation distribution against historical baseline via PSI & KS test."""
        candidate_cols = list(feature_cols) if feature_cols is not None else list(DRIFT_FEATURES)
        shared_cols = [
            col for col in candidate_cols if col in reference_df.columns and col in current_df.columns
        ]
        if not shared_cols:
            raise ValueError("No shared numeric columns found between reference_df and current_df")

        feature_psis: dict[str, float] = {}
        for col in shared_cols:
            ref_vals = reference_df[col].to_numpy(dtype=np.float64)
            cur_vals = current_df[col].to_numpy(dtype=np.float64)
            feature_psis[col] = round(_compute_feature_psi(ref_vals, cur_vals), 6)

        psi_score = round(float(np.mean(list(feature_psis.values()))), 6)
        ks_col = "sales" if "sales" in shared_cols else shared_cols[0]
        ks_res = ks_2samp(
            reference_df[ks_col].to_numpy(dtype=np.float64),
            current_df[ks_col].to_numpy(dtype=np.float64),
        )

        if psi_score < 0.10:
            status = "STABLE"
        elif psi_score < 0.25:
            status = "MODERATE_DRIFT"
        else:
            status = "SIGNIFICANT_DRIFT"

        return DriftReport(
            psi_score=psi_score,
            ks_statistic=round(float(ks_res.statistic), 6),
            ks_pvalue=round(float(ks_res.pvalue), 6),
            status=status,
            feature_psis=feature_psis,
        )
