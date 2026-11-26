"""Walmart M5 raw CSV ETL and curated Parquet builder for StockPulse AI."""

from __future__ import annotations

import argparse
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running directly as `python scripts/ingest_m5.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from core.schemas import DailySalesRecord, SkuPolicyRecord
from core.warehouse import StockPulseWarehouse

META_COLUMNS: tuple[str, ...] = (
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
)

CATEGORY_COST_RATIO: dict[str, float] = {
    "FOODS": 0.68,
    "HOUSEHOLD": 0.62,
    "HOBBIES": 0.58,
}

CATEGORY_BASE_LEAD_TIME: dict[str, int] = {
    "FOODS": 2,
    "HOUSEHOLD": 4,
    "HOBBIES": 5,
}

STATE_LEAD_OFFSET: dict[str, int] = {
    "CA": 0,
    "TX": 1,
    "WI": 2,
}


def _resolve_sales_source(raw_dir: Path) -> Path:
    """Return sales_train_evaluation.csv, falling back to sales_train_validation.csv."""
    eval_path = raw_dir / "sales_train_evaluation.csv"
    if eval_path.exists():
        return eval_path
    val_path = raw_dir / "sales_train_validation.csv"
    if val_path.exists():
        return val_path
    raise FileNotFoundError(
        f"Neither sales_train_evaluation.csv nor sales_train_validation.csv found in {raw_dir}"
    )


def _deterministic_cover_days(store_id: str, item_id: str) -> int:
    """Return a stable pseudo-random days-of-cover value in [4, 9] per store-SKU."""
    key = f"{store_id}:{item_id}".encode("utf-8")
    return 4 + (zlib.crc32(key) % 6)


def _select_moq(avg_daily_sales: float) -> int:
    """Assign pack-aligned minimum order quantity based on item velocity."""
    if avg_daily_sales < 1.0:
        return 6
    if avg_daily_sales < 4.0:
        return 12
    if avg_daily_sales < 12.0:
        return 24
    return 48


def _load_filtered_prices(
    prices_path: Path,
    stores: tuple[str, ...],
    items: tuple[str, ...],
    weeks: set[int],
) -> pd.DataFrame:
    """Stream sell_prices.csv in chunks to filter strictly to selected stores/items/weeks."""
    store_set = set(stores)
    item_set = set(items)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        prices_path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        dtype={
            "store_id": "string",
            "item_id": "string",
            "wm_yr_wk": "int32",
            "sell_price": "float64",
        },
        chunksize=500_000,
    ):
        mask = (
            chunk["store_id"].isin(store_set)
            & chunk["item_id"].isin(item_set)
            & chunk["wm_yr_wk"].isin(weeks)
        )
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        raise ValueError("No matching price records found in sell_prices.csv")

    prices = pd.concat(chunks, ignore_index=True)
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)
    return prices


def _build_sku_policies(curated_sales: pd.DataFrame) -> pd.DataFrame:
    """Derive operational cost, lead-time, on-hand stock, and MOQ policies per series."""
    policies: list[dict[str, object]] = []
    max_d_num = int(curated_sales["d_num"].max())
    recent_cutoff = max_d_num - 28

    for (store_id, item_id), grp in curated_sales.groupby(["store_id", "item_id"], sort=True):
        grp_sorted = grp.sort_values("d_num")
        dept_id = str(grp_sorted["dept_id"].iloc[0])
        cat_id = str(grp_sorted["cat_id"].iloc[0])
        state_id = str(grp_sorted["state_id"].iloc[0])

        avg_sell_price = float(grp_sorted["sell_price"].mean())
        cost_ratio = CATEGORY_COST_RATIO[cat_id]
        unit_cost = round(avg_sell_price * cost_ratio, 4)
        daily_holding_cost = round(max(0.002, (unit_cost * 0.25) / 365.0), 4)
        stockout_penalty = round(1.25 * (avg_sell_price - unit_cost), 4)

        lead_time_days = CATEGORY_BASE_LEAD_TIME[cat_id] + STATE_LEAD_OFFSET.get(state_id, 1)
        avg_daily_sales = float(grp_sorted["sales"].mean())
        trailing_28_mean = float(
            grp_sorted.loc[grp_sorted["d_num"] > recent_cutoff, "sales"].mean()
        )
        cover_days = _deterministic_cover_days(str(store_id), str(item_id))
        current_on_hand = max(0, int(round(trailing_28_mean * cover_days)))
        moq = _select_moq(avg_daily_sales)
        max_order_qty = max(moq * 4, int(math.ceil(avg_daily_sales * 60.0)))

        row = {
            "store_id": str(store_id),
            "item_id": str(item_id),
            "dept_id": dept_id,
            "cat_id": cat_id,
            "state_id": state_id,
            "avg_sell_price": round(avg_sell_price, 4),
            "unit_cost": unit_cost,
            "daily_holding_cost": daily_holding_cost,
            "stockout_penalty": stockout_penalty,
            "lead_time_days": int(lead_time_days),
            "current_on_hand": current_on_hand,
            "moq": int(moq),
            "max_order_qty": int(max_order_qty),
        }
        SkuPolicyRecord(**row)
        policies.append(row)

    return pd.DataFrame(policies).sort_values(["store_id", "item_id"]).reset_index(drop=True)


def build_curated_m5_datasets(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    history_days: int | None = None,
    init_warehouse: bool = True,
) -> tuple[Path, Path]:
    """Ingest raw M5 CSVs, curate 30 store-SKU series, and write Parquet artifacts."""
    settings = get_settings()
    raw_path = raw_dir or settings.raw_data_dir
    processed_path = out_dir or settings.processed_data_dir
    days = history_days or settings.history_days
    stores = settings.selected_stores
    items = settings.selected_items

    sales_csv = _resolve_sales_source(raw_path)
    calendar_csv = raw_path / "calendar.csv"
    prices_csv = raw_path / "sell_prices.csv"
    for req_file in (calendar_csv, prices_csv):
        if not req_file.exists():
            raise FileNotFoundError(f"Required M5 file missing: {req_file}")

    # Inspect header to select strictly the final `days` columns
    all_cols = pd.read_csv(sales_csv, nrows=0).columns.tolist()
    d_cols = [c for c in all_cols if c.startswith("d_")]
    if len(d_cols) < days:
        raise ValueError(f"Requested {days} days, but {sales_csv.name} only has {len(d_cols)}")
    selected_d_cols = d_cols[-days:]

    sales_wide = pd.read_csv(sales_csv, usecols=list(META_COLUMNS) + selected_d_cols)
    sales_wide = sales_wide[
        sales_wide["store_id"].isin(stores) & sales_wide["item_id"].isin(items)
    ].copy()

    expected_series = len(stores) * len(items)
    if len(sales_wide) != expected_series:
        raise ValueError(
            f"Expected {expected_series} store-SKU rows after filtering, got {len(sales_wide)}"
        )

    # Unpivot wide daily columns -> long format
    sales_long = sales_wide.melt(
        id_vars=list(META_COLUMNS),
        value_vars=selected_d_cols,
        var_name="d",
        value_name="sales",
    )
    sales_long["d_num"] = sales_long["d"].str.slice(2).astype(np.int32)
    sales_long["sales"] = sales_long["sales"].astype(np.int32)

    # Join calendar metadata and state-matched SNAP flag
    calendar = pd.read_csv(
        calendar_csv,
        usecols=[
            "d",
            "date",
            "wm_yr_wk",
            "weekday",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ],
    )
    calendar = calendar[calendar["d"].isin(selected_d_cols)].copy()
    calendar["event_name_1"] = calendar["event_name_1"].fillna("None").astype(str)
    calendar["event_type_1"] = calendar["event_type_1"].fillna("None").astype(str)

    merged = sales_long.merge(calendar, on="d", how="left", validate="many_to_one")
    if merged["date"].isna().any():
        raise ValueError("Calendar join produced missing dates")

    snap_lookup = {
        "CA": merged["snap_CA"].to_numpy(),
        "TX": merged["snap_TX"].to_numpy(),
        "WI": merged["snap_WI"].to_numpy(),
    }
    state_arr = merged["state_id"].to_numpy()
    snap_values = np.select(
        [state_arr == "CA", state_arr == "TX", state_arr == "WI"],
        [snap_lookup["CA"], snap_lookup["TX"], snap_lookup["WI"]],
        default=0,
    ).astype(np.int8)
    merged["snap"] = snap_values
    merged = merged.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    # Join weekly sell prices and forward/back-fill per series
    target_weeks = set(merged["wm_yr_wk"].unique().tolist())
    prices = _load_filtered_prices(prices_csv, stores, items, target_weeks)
    merged = merged.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.sort_values(["store_id", "item_id", "d_num"], kind="mergesort").reset_index(
        drop=True
    )
    merged["sell_price"] = (
        merged.groupby(["store_id", "item_id"], sort=False)["sell_price"]
        .transform(lambda s: s.ffill().bfill())
        .astype(np.float64)
    )
    if merged["sell_price"].isna().any():
        raise ValueError("Missing sell_price values remain after forward/back-fill")

    merged["date"] = pd.to_datetime(merged["date"])
    merged["wm_yr_wk"] = merged["wm_yr_wk"].astype(np.int32)
    merged["wday"] = merged["wday"].astype(np.int8)
    merged["month"] = merged["month"].astype(np.int8)
    merged["year"] = merged["year"].astype(np.int16)

    ordered_cols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "d_num",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "event_name_1",
        "event_type_1",
        "snap",
        "sell_price",
        "sales",
    ]
    curated_sales = merged[ordered_cols].copy()

    # Validate sample & boundary records against Pydantic v2 DailySalesRecord
    for idx in (0, len(curated_sales) // 2, len(curated_sales) - 1):
        rec = curated_sales.iloc[idx].to_dict()
        rec["date"] = rec["date"].date()
        DailySalesRecord(**rec)

    sku_policies = _build_sku_policies(curated_sales)

    processed_path.mkdir(parents=True, exist_ok=True)
    sales_out = processed_path / "m5_curated_sales.parquet"
    policies_out = processed_path / "m5_sku_policies.parquet"

    curated_sales.to_parquet(sales_out, index=False, compression="snappy")
    sku_policies.to_parquet(policies_out, index=False, compression="snappy")

    if init_warehouse:
        with StockPulseWarehouse(
            db_path=settings.duckdb_path,
            curated_sales_path=sales_out,
            sku_policies_path=policies_out,
        ) as wh:
            wh.ensure_initialized(force_reload=True)

    return sales_out, policies_out


def main() -> None:
    """CLI entrypoint for Walmart M5 curated Parquet generation."""
    parser = argparse.ArgumentParser(description="Ingest raw Walmart M5 CSVs to curated Parquet.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Directory with raw M5 CSVs")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--days", type=int, default=None, help="Trailing days of daily history to curate"
    )
    parser.add_argument(
        "--no-warehouse",
        action="store_true",
        help="Skip initializing DuckDB after writing Parquet files",
    )
    args = parser.parse_args()

    sales_out, policies_out = build_curated_m5_datasets(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        history_days=args.days,
        init_warehouse=not args.no_warehouse,
    )
    print(f"Wrote curated sales: {sales_out}")
    print(f"Wrote SKU policies:  {policies_out}")


if __name__ == "__main__":
    main()
