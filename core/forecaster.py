"""LightGBM P10/P50/P90 quantile demand forecaster with recursive multi-step rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from core.config import get_settings
from core.features import (
    EVENT_TYPE_MAP,
    FEATURE_COLUMNS,
    build_temporal_features,
    split_chronological,
)
from core.schemas import ForecastMetrics, ForecastPoint
from core.warehouse import StockPulseWarehouse

QUANTILE_ALPHAS: tuple[float, float, float] = (0.10, 0.50, 0.90)


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Compute mean pinball (quantile) loss for quantile level `alpha`."""
    diff = y_true - y_pred
    loss = np.where(diff >= 0.0, alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


class QuantileForecaster:
    """Three-model LightGBM quantile forecaster enforcing non-crossing P10/P50/P90 bounds."""

    def __init__(self, random_seed: int | None = None) -> None:
        settings = get_settings()
        self.random_seed = settings.random_seed if random_seed is None else random_seed
        self.feature_cols: tuple[str, ...] = FEATURE_COLUMNS
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.entity_mappings: dict[str, dict[str, int]] = {}

    def _build_entity_mappings(self, df: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Capture deterministic sorted category-to-int mappings used during training."""
        mappings: dict[str, dict[str, int]] = {}
        for col in ("store_id", "cat_id", "dept_id", "item_id"):
            cats = sorted(df[col].astype(str).unique().tolist())
            mappings[col] = {val: idx for idx, val in enumerate(cats)}
        return mappings

    def fit(self, train_df: pd.DataFrame) -> QuantileForecaster:
        """Fit independent LightGBM quantile regressors for alpha = 0.10, 0.50, 0.90."""
        missing = set(self.feature_cols) - set(train_df.columns)
        if missing:
            raise ValueError(f"Training frame missing feature columns: {sorted(missing)}")

        self.entity_mappings = self._build_entity_mappings(train_df)
        x_train = train_df[list(self.feature_cols)]
        y_train = train_df["sales"].to_numpy(dtype=np.float64)

        self.models = {}
        for alpha in QUANTILE_ALPHAS:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.85,
                random_state=self.random_seed,
                n_jobs=1,
                verbose=-1,
            )
            model.fit(x_train, y_train)
            self.models[alpha] = model

        return self

    def _ensure_fitted(self) -> None:
        if len(self.models) != len(QUANTILE_ALPHAS):
            raise RuntimeError("QuantileForecaster must be fitted or loaded before prediction")

    def predict_quantiles(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict non-negative, non-crossing (p10, p50, p90) array of shape (N, 3)."""
        self._ensure_fitted()
        x = features_df[list(self.feature_cols)]
        preds = np.column_stack(
            [self.models[alpha].predict(x) for alpha in QUANTILE_ALPHAS]
        )
        preds = np.clip(preds, 0.0, None)
        preds = np.sort(preds, axis=1)
        return preds

    def predict_horizon(
        self,
        history_df: pd.DataFrame,
        future_calendar_df: pd.DataFrame,
        horizon_days: int = 28,
    ) -> list[ForecastPoint]:
        """Run strictly autoregressive multi-step P10/P50/P90 forecasts over `horizon_days`.

        At each future day `h`, predicted `p50` values update past lag and rolling window
        buffers for subsequent days `h+1..horizon_days-1` with zero future target leakage.
        """
        self._ensure_fitted()
        if horizon_days < 1:
            raise ValueError("horizon_days must be >= 1")

        hist = history_df.copy()
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values(["store_id", "item_id", "d_num"], kind="mergesort")

        fut = future_calendar_df.copy()
        fut["date"] = pd.to_datetime(fut["date"])
        fut = fut.sort_values(["store_id", "item_id", "d_num"], kind="mergesort")

        # Initialize per-series causal rolling state strictly from history_df
        states: dict[tuple[str, str], dict[str, Any]] = {}
        series_groups: list[tuple[tuple[str, str], pd.DataFrame]] = []

        for key, fut_grp in fut.groupby(["store_id", "item_id"], sort=True):
            store_id, item_id = str(key[0]), str(key[1])
            hist_grp = hist[(hist["store_id"] == store_id) & (hist["item_id"] == item_id)]
            if len(hist_grp) < 28:
                raise ValueError(
                    f"Series ({store_id}, {item_id}) requires >= 28 history days, got {len(hist_grp)}"
                )

            sales_vals = hist_grp["sales"].to_numpy(dtype=np.float64).tolist()
            price_vals = hist_grp["sell_price"].to_numpy(dtype=np.float64).tolist()
            states[(store_id, item_id)] = {
                "sales_hist": sales_vals[-28:],
                "price_hist": price_vals[-7:],
                "price_sum": float(np.sum(price_vals)),
                "price_count": len(price_vals),
            }
            series_groups.append(((store_id, item_id), fut_grp.iloc[:horizon_days].reset_index(drop=True)))

        max_steps = max(len(grp) for _, grp in series_groups) if series_groups else 0
        collected: list[dict[str, Any]] = []

        for step in range(max_steps):
            step_rows: list[dict[str, Any]] = []
            step_meta: list[tuple[str, str, Any, float | None]] = []

            for (store_id, item_id), fut_grp in series_groups:
                if step >= len(fut_grp):
                    continue
                row = fut_grp.iloc[step]
                st = states[(store_id, item_id)]
                s_hist: list[float] = st["sales_hist"]
                p_hist: list[float] = st["price_hist"]

                sell_price = float(row["sell_price"])
                st["price_sum"] += sell_price
                st["price_count"] += 1
                avg_price = st["price_sum"] / st["price_count"]

                price_7_ago = p_hist[-7] if len(p_hist) >= 7 else p_hist[0]
                price_momentum_7d = (
                    (sell_price - price_7_ago) / price_7_ago if price_7_ago > 0.0 else 0.0
                )

                past_7 = s_hist[-7:]
                past_28 = s_hist[-28:]
                rolling_std_7 = float(np.std(past_7, ddof=1)) if len(past_7) >= 2 else 0.0
                zero_rate_28 = float(np.mean([1.0 if v < 0.5 else 0.0 for v in past_28]))

                wday = int(row["wday"])
                month = int(row["month"])
                dow_angle = 2.0 * np.pi * (wday - 1) / 7.0
                month_angle = 2.0 * np.pi * (month - 1) / 12.0

                event_name = str(row.get("event_name_1", "None") or "None")
                event_type = str(row.get("event_type_1", "None") or "None")
                cat_id = str(row["cat_id"])
                dept_id = str(row["dept_id"])

                feat_row = {
                    "lag_1": float(s_hist[-1]),
                    "lag_7": float(s_hist[-7]),
                    "lag_14": float(s_hist[-14]),
                    "lag_28": float(s_hist[-28]),
                    "rolling_mean_7": float(np.mean(past_7)),
                    "rolling_std_7": rolling_std_7,
                    "rolling_mean_28": float(np.mean(past_28)),
                    "rolling_zero_rate_28": zero_rate_28,
                    "sell_price": sell_price,
                    "price_momentum_7d": float(price_momentum_7d),
                    "price_vs_item_avg": float(sell_price / avg_price) if avg_price > 0.0 else 1.0,
                    "has_event": int(event_name != "None"),
                    "event_type_code": int(EVENT_TYPE_MAP.get(event_type, 0)),
                    "snap": int(row["snap"]),
                    "is_weekend": int(wday in (1, 2)),
                    "dow_sin": float(np.sin(dow_angle)),
                    "dow_cos": float(np.cos(dow_angle)),
                    "month_sin": float(np.sin(month_angle)),
                    "month_cos": float(np.cos(month_angle)),
                    "store_code": self.entity_mappings["store_id"][store_id],
                    "cat_code": self.entity_mappings["cat_id"][cat_id],
                    "dept_code": self.entity_mappings["dept_id"][dept_id],
                    "item_code": self.entity_mappings["item_id"][item_id],
                }
                step_rows.append(feat_row)
                actual_val = float(row["sales"]) if "sales" in row and pd.notna(row["sales"]) else None
                step_meta.append((store_id, item_id, pd.Timestamp(row["date"]).date(), actual_val))

            if not step_rows:
                continue

            step_df = pd.DataFrame(step_rows, columns=list(self.feature_cols))
            preds = self.predict_quantiles(step_df)

            for idx, (store_id, item_id, pred_date, actual_val) in enumerate(step_meta):
                p10, p50, p90 = float(preds[idx, 0]), float(preds[idx, 1]), float(preds[idx, 2])
                st = states[(store_id, item_id)]
                st["sales_hist"] = (st["sales_hist"] + [p50])[-28:]
                st["price_hist"] = (st["price_hist"] + [float(step_rows[idx]["sell_price"])])[-7:]

                collected.append(
                    {
                        "date": pred_date,
                        "store_id": store_id,
                        "item_id": item_id,
                        "actual": actual_val,
                        "p10": round(p10, 4),
                        "p50": round(p50, 4),
                        "p90": round(p90, 4),
                    }
                )

        collected.sort(key=lambda r: (r["store_id"], r["item_id"], r["date"]))
        return [ForecastPoint(**r) for r in collected]

    @staticmethod
    def evaluate_predictions(points: list[ForecastPoint]) -> ForecastMetrics:
        """Compute WAPE, MAE, RMSE, quantile pinball losses, and 80% PI coverage."""
        scored = [p for p in points if p.actual is not None]
        if not scored:
            raise ValueError("Cannot evaluate ForecastMetrics without actual values")

        y_true = np.array([p.actual for p in scored], dtype=np.float64)
        p10 = np.array([p.p10 for p in scored], dtype=np.float64)
        p50 = np.array([p.p50 for p in scored], dtype=np.float64)
        p90 = np.array([p.p90 for p in scored], dtype=np.float64)

        abs_err = np.abs(y_true - p50)
        denom = max(float(np.sum(np.abs(y_true))), 1e-9)
        wape = float(np.sum(abs_err) / denom)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean((y_true - p50) ** 2)))
        coverage = float(np.mean((y_true >= p10) & (y_true <= p90)))

        return ForecastMetrics(
            wape=round(wape, 6),
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            pinball_p10=round(_pinball_loss(y_true, p10, 0.10), 6),
            pinball_p50=round(_pinball_loss(y_true, p50, 0.50), 6),
            pinball_p90=round(_pinball_loss(y_true, p90, 0.90), 6),
            interval_coverage_80=round(coverage, 6),
        )

    def save(self, path: Path | str | None = None) -> Path:
        """Persist fitted quantile model bundle via joblib."""
        self._ensure_fitted()
        out_path = Path(path) if path is not None else get_settings().model_bundle_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, out_path, compress=3)
        return out_path

    @classmethod
    def load(cls, path: Path | str | None = None) -> QuantileForecaster:
        """Load a persisted QuantileForecaster bundle from disk."""
        in_path = Path(path) if path is not None else get_settings().model_bundle_path
        bundle = joblib.load(in_path)
        if not isinstance(bundle, cls):
            raise TypeError(f"Expected QuantileForecaster in {in_path}, got {type(bundle)}")
        return bundle

    @classmethod
    def load_or_train(
        cls,
        train_df: pd.DataFrame | None = None,
        bundle_path: Path | str | None = None,
        force_retrain: bool = False,
    ) -> QuantileForecaster:
        """Load cached model bundle if present; otherwise train on M5 history and save."""
        settings = get_settings()
        target_path = Path(bundle_path) if bundle_path is not None else settings.model_bundle_path
        if target_path.exists() and not force_retrain:
            return cls.load(target_path)

        if train_df is None:
            with StockPulseWarehouse() as wh:
                raw_sales = wh.load_sales_dataframe()
            featured = build_temporal_features(raw_sales, drop_na_lags=True)
            split = split_chronological(featured, horizon_days=settings.forecast_horizon_days)
            train_df = split.train_df

        forecaster = cls(random_seed=settings.random_seed).fit(train_df)
        forecaster.save(target_path)
        return forecaster
