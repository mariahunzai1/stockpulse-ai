"""SHAP TreeExplainer engine for P50 demand forecast explainability."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import shap

from core.forecaster import QuantileForecaster
from core.schemas import ShapContribution, ShapExplanation


class DemandExplainer:
    """Exact additive TreeSHAP attribution engine over the P50 LightGBM model."""

    def __init__(self, forecaster: QuantileForecaster) -> None:
        if 0.50 not in forecaster.models:
            raise RuntimeError("QuantileForecaster must contain a fitted 0.50 model")
        self.forecaster = forecaster
        self.feature_cols = list(forecaster.feature_cols)
        self._explainer = shap.TreeExplainer(forecaster.models[0.50])

    @staticmethod
    def _extract_expected_value(expected_value: object) -> float:
        arr = np.asarray(expected_value, dtype=np.float64).ravel()
        return float(arr[0])

    def explain_prediction(
        self,
        featured_df: pd.DataFrame,
        store_id: str,
        item_id: str,
        target_date: dt.date | str,
    ) -> ShapExplanation:
        """Explain a single (store_id, item_id, date) P50 prediction with exact SHAP additivity."""
        parsed_date = pd.Timestamp(target_date).date()
        work = featured_df.copy()
        work["_date_only"] = pd.to_datetime(work["date"]).dt.date
        match = work[
            (work["store_id"] == store_id)
            & (work["item_id"] == item_id)
            & (work["_date_only"] == parsed_date)
        ]
        if match.empty:
            raise KeyError(
                f"No feature row found for store={store_id}, item={item_id}, date={parsed_date}"
            )

        x_row = match.iloc[[0]][self.feature_cols]
        shap_vals = np.asarray(self._explainer.shap_values(x_row), dtype=np.float64).reshape(-1)
        base_value = self._extract_expected_value(self._explainer.expected_value)
        raw_p50 = float(self.forecaster.models[0.50].predict(x_row)[0])

        row_values = x_row.iloc[0].to_numpy(dtype=np.float64)
        contributions = [
            ShapContribution(
                feature=feat,
                feature_value=round(float(val), 6),
                shap_value=float(sval),
            )
            for feat, val, sval in zip(self.feature_cols, row_values, shap_vals)
        ]
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

        return ShapExplanation(
            store_id=store_id,
            item_id=item_id,
            date=parsed_date,
            base_value=base_value,
            predicted_p50=raw_p50,
            contributions=contributions,
        )

    def global_feature_importance(self, featured_df: pd.DataFrame) -> dict[str, float]:
        """Compute mean absolute TreeSHAP importance across `featured_df`, sorted descending."""
        x = featured_df[self.feature_cols]
        shap_matrix = np.asarray(self._explainer.shap_values(x), dtype=np.float64)
        mean_abs = np.mean(np.abs(shap_matrix), axis=0)
        paired = sorted(
            zip(self.feature_cols, mean_abs.tolist()),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return {feat: round(float(score), 6) for feat, score in paired}
