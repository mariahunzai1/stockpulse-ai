"""Leak-free temporal feature engineering and chronological splitting for M5 demand."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.schemas import ChronologicalSplitSummary

EVENT_TYPE_MAP: dict[str, int] = {
    "None": 0,
    "Sporting": 1,
    "Cultural": 2,
    "National": 3,
    "Religious": 4,
}

FEATURE_COLUMNS: tuple[str, ...] = (
    # Past-only demand lags
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    # Past-only rolling demand statistics
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_zero_rate_28",
    # Causal price dynamics
    "sell_price",
    "price_momentum_7d",
    "price_vs_item_avg",
    # Calendar, holiday, SNAP & cyclical encodings
    "has_event",
    "event_type_code",
    "snap",
    "is_weekend",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    # Deterministic entity encodings
    "store_code",
    "cat_code",
    "dept_code",
    "item_code",
)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Train/validation split holding out the final forecast horizon strictly in time."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    feature_cols: tuple[str, ...]
    target_col: str
    summary: ChronologicalSplitSummary


def _encode_deterministic(series: pd.Series) -> pd.Series:
    """Map sorted unique string categories to stable integer codes."""
    categories = sorted(series.astype(str).unique().tolist())
    mapping = {val: idx for idx, val in enumerate(categories)}
    return series.astype(str).map(mapping).astype(np.int32)


def build_temporal_features(
    df: pd.DataFrame,
    drop_na_lags: bool = True,
) -> pd.DataFrame:
    """Engineer strictly causal (zero-leakage) temporal, price, and calendar features.

    All lag and rolling demand statistics are computed after shifting sales by at least
    1 day (`shift(1)`) within each `(store_id, item_id)` group so row `t` never accesses
    target values at `>= t`. Price relative to historical average uses an expanding mean
    up to day `t`, avoiding lookahead from future price changes.
    """
    required = {
        "store_id",
        "item_id",
        "dept_id",
        "cat_id",
        "d_num",
        "date",
        "wday",
        "month",
        "event_name_1",
        "event_type_1",
        "snap",
        "sell_price",
        "sales",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["store_id", "item_id", "d_num"], kind="mergesort").reset_index(
        drop=True
    )

    grp = out.groupby(["store_id", "item_id"], sort=False)

    # Strictly past-only sales lags
    for lag in (1, 7, 14, 28):
        out[f"lag_{lag}"] = grp["sales"].shift(lag)

    # Strictly past-only rolling statistics (shift(1) before rolling window)
    out["_past_sales_1"] = grp["sales"].shift(1)
    out["_past_zero_1"] = (out["_past_sales_1"] == 0).astype(np.float64)
    out.loc[out["_past_sales_1"].isna(), "_past_zero_1"] = np.nan

    past_grp = out.groupby(["store_id", "item_id"], sort=False)
    out["rolling_mean_7"] = (
        past_grp["_past_sales_1"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    out["rolling_std_7"] = (
        past_grp["_past_sales_1"]
        .rolling(window=7, min_periods=2)
        .std()
        .fillna(0.0)
        .reset_index(level=[0, 1], drop=True)
    )
    out["rolling_mean_28"] = (
        past_grp["_past_sales_1"]
        .rolling(window=28, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    out["rolling_zero_rate_28"] = (
        past_grp["_past_zero_1"]
        .rolling(window=28, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Causal price dynamics
    price_lag_7 = grp["sell_price"].shift(7)
    denom = price_lag_7.replace(0.0, np.nan)
    out["price_momentum_7d"] = ((out["sell_price"] - price_lag_7) / denom).fillna(0.0)

    expanding_price_mean = (
        grp["sell_price"]
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    out["price_vs_item_avg"] = out["sell_price"] / expanding_price_mean.replace(0.0, np.nan)

    # Calendar, holiday & SNAP features (M5 wday: 1=Saturday, 2=Sunday)
    out["has_event"] = (out["event_name_1"].fillna("None") != "None").astype(np.int8)
    out["event_type_code"] = (
        out["event_type_1"].fillna("None").map(EVENT_TYPE_MAP).fillna(0).astype(np.int8)
    )
    out["snap"] = out["snap"].astype(np.int8)
    out["is_weekend"] = out["wday"].isin([1, 2]).astype(np.int8)

    dow_angle = 2.0 * np.pi * (out["wday"].astype(np.float64) - 1.0) / 7.0
    out["dow_sin"] = np.sin(dow_angle)
    out["dow_cos"] = np.cos(dow_angle)

    month_angle = 2.0 * np.pi * (out["month"].astype(np.float64) - 1.0) / 12.0
    out["month_sin"] = np.sin(month_angle)
    out["month_cos"] = np.cos(month_angle)

    # Deterministic categorical entity codes
    out["store_code"] = _encode_deterministic(out["store_id"])
    out["cat_code"] = _encode_deterministic(out["cat_id"])
    out["dept_code"] = _encode_deterministic(out["dept_id"])
    out["item_code"] = _encode_deterministic(out["item_id"])

    out = out.drop(columns=["_past_sales_1", "_past_zero_1"])

    if drop_na_lags:
        out = out.dropna(subset=["lag_28"]).reset_index(drop=True)

    return out


def split_chronological(
    df: pd.DataFrame,
    horizon_days: int = 28,
) -> ChronologicalSplit:
    """Split a featured DataFrame chronologically, holding out the final `horizon_days` dates."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    unique_dates = np.sort(work["date"].unique())
    if len(unique_dates) <= horizon_days:
        raise ValueError(
            f"Not enough unique dates ({len(unique_dates)}) for horizon_days={horizon_days}"
        )

    val_start_ts = pd.Timestamp(unique_dates[-horizon_days])
    train_df = (
        work[work["date"] < val_start_ts]
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )
    val_df = (
        work[work["date"] >= val_start_ts]
        .sort_values(["store_id", "item_id", "date"])
        .reset_index(drop=True)
    )

    sku_series_count = int(work.groupby(["store_id", "item_id"]).ngroups)
    summary = ChronologicalSplitSummary(
        train_rows=len(train_df),
        val_rows=len(val_df),
        train_start=train_df["date"].min().date(),
        train_end=train_df["date"].max().date(),
        val_start=val_df["date"].min().date(),
        val_end=val_df["date"].max().date(),
        horizon_days=horizon_days,
        sku_series_count=sku_series_count,
    )
    return ChronologicalSplit(
        train_df=train_df,
        val_df=val_df,
        feature_cols=FEATURE_COLUMNS,
        target_col="sales",
        summary=summary,
    )
