"""Tests for curated M5 Parquet contracts and DuckDB warehouse idempotency."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from core.config import get_settings
from core.schemas import DailySalesRecord, SkuPolicyRecord
from core.warehouse import StockPulseWarehouse


@pytest.fixture(scope="module")
def curated_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    settings = get_settings()
    assert settings.curated_sales_path.exists(), "Run scripts/ingest_m5.py before running tests"
    assert settings.sku_policies_path.exists(), "Run scripts/ingest_m5.py before running tests"
    sales_df = pd.read_parquet(settings.curated_sales_path)
    policies_df = pd.read_parquet(settings.sku_policies_path)
    return sales_df, policies_df


def test_curated_parquet_dimensions_and_schemas(
    curated_frames: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    sales_df, policies_df = curated_frames
    settings = get_settings()

    expected_series = len(settings.selected_stores) * len(settings.selected_items)
    assert expected_series == 30
    assert len(sales_df) == expected_series * settings.history_days
    assert len(policies_df) == expected_series

    assert set(sales_df["store_id"].unique()) == set(settings.selected_stores)
    assert set(sales_df["item_id"].unique()) == set(settings.selected_items)
    assert set(sales_df["cat_id"].unique()) == {"FOODS", "HOBBIES", "HOUSEHOLD"}
    assert sales_df["sell_price"].isna().sum() == 0
    assert (sales_df["sales"] >= 0).all()

    # Validate every SKU policy row against Pydantic v2 contract
    for row in policies_df.to_dict(orient="records"):
        policy = SkuPolicyRecord(**row)
        assert policy.unit_cost < policy.avg_sell_price
        assert policy.max_order_qty >= policy.moq

    # Validate stratified sample of daily sales rows against Pydantic v2 contract
    sample_indices = [0, 729, 730, len(sales_df) // 2, len(sales_df) - 1]
    for idx in sample_indices:
        rec = sales_df.iloc[idx].to_dict()
        rec["date"] = pd.Timestamp(rec["date"]).date()
        DailySalesRecord(**rec)


def test_warehouse_idempotent_initialization_and_queries(tmp_path: Path) -> None:
    db_file = tmp_path / "test_stockpulse.duckdb"
    with StockPulseWarehouse(db_path=db_file) as wh:
        count_1 = wh.ensure_initialized()
        count_2 = wh.ensure_initialized()
        count_3 = wh.ensure_initialized(force_reload=True)

        assert count_1 == (21900, 30)
        assert count_2 == (21900, 30)
        assert count_3 == (21900, 30)

        summaries = wh.get_store_category_summary()
        assert len(summaries) == 9  # 3 stores x 3 categories
        assert sum(s.sku_count for s in summaries) == 30
        assert sum(s.row_count for s in summaries) == 21900

        policies = wh.list_sku_policies(store_id="CA_1", cat_id="FOODS")
        assert len(policies) == 4
        assert all(p.store_id == "CA_1" and p.cat_id == "FOODS" for p in policies)

        policy = wh.get_sku_policy("TX_1", "HOBBIES_1_225")
        assert policy.state_id == "TX"
        assert policy.cat_id == "HOBBIES"

        series = wh.get_sku_series(
            "WI_1",
            "HOUSEHOLD_1_334",
            start_date=dt.date(2016, 4, 25),
            end_date=dt.date(2016, 5, 22),
        )
        assert len(series) == 28
        assert [r.d_num for r in series] == sorted(r.d_num for r in series)

        price_hist = wh.get_price_demand_history("CA_1", "FOODS_3_090")
        assert len(price_hist) == 730
