"""Tests for zero-leakage temporal feature engineering and chronological splitting."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from core.config import get_settings
from core.features import FEATURE_COLUMNS, build_temporal_features, split_chronological
from core.warehouse import StockPulseWarehouse


def _make_synthetic_series(n_days: int = 80) -> pd.DataFrame:
    """Create a deterministic 2-SKU synthetic daily panel for leakage and intermittency tests."""
    start = dt.date(2025, 1, 1)
    rows: list[dict[str, object]] = []
    specs = [
        ("CA_1", "CA", "FOODS_3_090", "FOODS_3", "FOODS"),
        ("TX_1", "TX", "HOBBIES_1_048", "HOBBIES_1", "HOBBIES"),
    ]
    for store_id, state_id, item_id, dept_id, cat_id in specs:
        for i in range(n_days):
            cur_date = start + dt.timedelta(days=i)
            # Item 1 has steady positive sales; Item 2 has a 30-day zero-sale streak in the middle
            if item_id == "HOBBIES_1_048" and 20 <= i < 55:
                sales = 0
            else:
                sales = int((i % 7) + (2 if store_id == "CA_1" else 1))
            rows.append(
                {
                    "id": f"{item_id}_{store_id}_evaluation",
                    "item_id": item_id,
                    "dept_id": dept_id,
                    "cat_id": cat_id,
                    "store_id": store_id,
                    "state_id": state_id,
                    "d": f"d_{1000 + i}",
                    "d_num": 1000 + i,
                    "date": pd.Timestamp(cur_date),
                    "wm_yr_wk": 12501 + (i // 7),
                    "weekday": cur_date.strftime("%A"),
                    "wday": ((cur_date.weekday() + 2) % 7) + 1,
                    "month": cur_date.month,
                    "year": cur_date.year,
                    "event_name_1": "SuperBowl" if i == 35 else "None",
                    "event_type_1": "Sporting" if i == 35 else "None",
                    "snap": int(i % 10 == 0),
                    "sell_price": 2.50 if i < 40 else 3.00,
                    "sales": sales,
                }
            )
    return pd.DataFrame(rows)


def test_zero_target_and_future_leakage_perturbation() -> None:
    """Mutating sales[t] and future sales/prices[t+1:] must never alter features at <= t."""
    base_df = _make_synthetic_series(n_days=70)
    perturbed_df = base_df.copy()

    # Mutate sales on day index 45 and all future days (>= 45) for CA_1/FOODS_3_090
    cutoff_d_num = 1045
    mask_target_and_future = (perturbed_df["store_id"] == "CA_1") & (
        perturbed_df["d_num"] >= cutoff_d_num
    )
    perturbed_df.loc[mask_target_and_future, "sales"] = 999

    # Mutate future prices strictly AFTER cutoff_d_num (> 1045)
    mask_strict_future = (perturbed_df["store_id"] == "CA_1") & (
        perturbed_df["d_num"] > cutoff_d_num
    )
    perturbed_df.loc[mask_strict_future, "sell_price"] = 99.99

    feat_base = build_temporal_features(base_df, drop_na_lags=True)
    feat_pert = build_temporal_features(perturbed_df, drop_na_lags=True)

    check_cols = list(FEATURE_COLUMNS)
    base_le_cutoff = feat_base[
        (feat_base["store_id"] == "CA_1") & (feat_base["d_num"] <= cutoff_d_num)
    ][check_cols].reset_index(drop=True)
    pert_le_cutoff = feat_pert[
        (feat_pert["store_id"] == "CA_1") & (feat_pert["d_num"] <= cutoff_d_num)
    ][check_cols].reset_index(drop=True)

    pd.testing.assert_frame_equal(base_le_cutoff, pert_le_cutoff)


def test_intermittent_zero_rate_and_rolling_windows() -> None:
    """Verify rolling_zero_rate_28 captures intermittent zero-sale regimes accurately."""
    df = _make_synthetic_series(n_days=80)
    featured = build_temporal_features(df, drop_na_lags=True)

    hobbies = featured[featured["item_id"] == "HOBBIES_1_048"].sort_values("d_num")
    # Days 20..54 are all zero -> on day 50 (after 30 consecutive zeros), past 28 days are all 0
    day_50 = hobbies.loc[hobbies["d_num"] == 1050].iloc[0]
    assert day_50["rolling_zero_rate_28"] == pytest.approx(1.0)
    assert day_50["rolling_mean_28"] == pytest.approx(0.0)
    assert day_50["rolling_std_7"] == pytest.approx(0.0)


def test_chronological_split_on_curated_m5_data(tmp_path) -> None:
    """Verify 28-day chronological holdout on real curated M5 data has zero overlap."""
    settings = get_settings()
    with StockPulseWarehouse(db_path=tmp_path / "split_test.duckdb") as wh:
        sales_df = wh.load_sales_dataframe()

    featured = build_temporal_features(sales_df, drop_na_lags=True)
    assert not featured[list(FEATURE_COLUMNS)].isna().any().any()

    # 730 days - 28 warmup days = 702 days * 30 series = 21,060 rows
    assert len(featured) == (settings.history_days - 28) * 30

    split = split_chronological(featured, horizon_days=settings.forecast_horizon_days)
    assert split.summary.train_end < split.summary.val_start
    assert split.summary.horizon_days == 28
    assert split.summary.sku_series_count == 30
    assert split.summary.val_rows == 30 * 28
    assert split.summary.train_rows == len(featured) - 30 * 28

    # Ensure zero (store_id, item_id, date) key overlap between train and validation
    train_keys = set(
        zip(split.train_df["store_id"], split.train_df["item_id"], split.train_df["date"])
    )
    val_keys = set(zip(split.val_df["store_id"], split.val_df["item_id"], split.val_df["date"]))
    assert train_keys.isdisjoint(val_keys)

    # Confirm cyclical bounds and weekend indicator integrity
    assert np.all(np.abs(featured["dow_sin"]) <= 1.0 + 1e-9)
    assert np.all(np.abs(featured["month_cos"]) <= 1.0 + 1e-9)
    assert set(featured["is_weekend"].unique()) == {0, 1}
