"""DuckDB analytical warehouse backed by curated Walmart M5 Parquet tables."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from core.config import get_settings
from core.schemas import (
    DailySalesRecord,
    PriceDemandPoint,
    SkuPolicyRecord,
    StoreCategorySummary,
)


class StockPulseWarehouse:
    """Idempotent DuckDB warehouse over curated Walmart M5 Parquet datasets."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        curated_sales_path: Path | str | None = None,
        sku_policies_path: Path | str | None = None,
    ) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else settings.duckdb_path
        self.curated_sales_path = (
            Path(curated_sales_path)
            if curated_sales_path is not None
            else settings.curated_sales_path
        )
        self.sku_policies_path = (
            Path(sku_policies_path)
            if sku_policies_path is not None
            else settings.sku_policies_path
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(self.db_path))

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> StockPulseWarehouse:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _table_exists(self, table_name: str) -> bool:
        row = self._conn.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            """,
            [table_name],
        ).fetchone()
        return bool(row and row[0] > 0)

    def ensure_initialized(self, force_reload: bool = False) -> tuple[int, int]:
        """Load curated Parquet files into DuckDB idempotently.

        Uses CREATE OR REPLACE TABLE so repeated initializations never duplicate rows.
        Returns `(curated_sales_count, sku_policies_count)`.
        """
        if not self.curated_sales_path.exists():
            raise FileNotFoundError(
                f"Curated sales Parquet not found: {self.curated_sales_path}. "
                "Run `python scripts/ingest_m5.py` first."
            )
        if not self.sku_policies_path.exists():
            raise FileNotFoundError(
                f"SKU policies Parquet not found: {self.sku_policies_path}. "
                "Run `python scripts/ingest_m5.py` first."
            )

        if force_reload or not self._table_exists("curated_sales"):
            self._conn.execute(
                """
                CREATE OR REPLACE TABLE curated_sales AS
                SELECT
                    id,
                    item_id,
                    dept_id,
                    cat_id,
                    store_id,
                    state_id,
                    d,
                    CAST(d_num AS INTEGER) AS d_num,
                    CAST(date AS DATE) AS date,
                    CAST(wm_yr_wk AS INTEGER) AS wm_yr_wk,
                    weekday,
                    CAST(wday AS INTEGER) AS wday,
                    CAST(month AS INTEGER) AS month,
                    CAST(year AS INTEGER) AS year,
                    event_name_1,
                    event_type_1,
                    CAST(snap AS INTEGER) AS snap,
                    CAST(sell_price AS DOUBLE) AS sell_price,
                    CAST(sales AS INTEGER) AS sales
                FROM read_parquet(?)
                """,
                [str(self.curated_sales_path)],
            )

        if force_reload or not self._table_exists("sku_policies"):
            self._conn.execute(
                """
                CREATE OR REPLACE TABLE sku_policies AS
                SELECT
                    store_id,
                    item_id,
                    dept_id,
                    cat_id,
                    state_id,
                    CAST(avg_sell_price AS DOUBLE) AS avg_sell_price,
                    CAST(unit_cost AS DOUBLE) AS unit_cost,
                    CAST(daily_holding_cost AS DOUBLE) AS daily_holding_cost,
                    CAST(stockout_penalty AS DOUBLE) AS stockout_penalty,
                    CAST(lead_time_days AS INTEGER) AS lead_time_days,
                    CAST(current_on_hand AS INTEGER) AS current_on_hand,
                    CAST(moq AS INTEGER) AS moq,
                    CAST(max_order_qty AS INTEGER) AS max_order_qty
                FROM read_parquet(?)
                """,
                [str(self.sku_policies_path)],
            )

        sales_count = self._conn.execute("SELECT COUNT(*) FROM curated_sales").fetchone()[0]
        policies_count = self._conn.execute("SELECT COUNT(*) FROM sku_policies").fetchone()[0]
        return int(sales_count), int(policies_count)

    def _fetch_dicts(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        cursor = self._conn.execute(query, params or [])
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_store_category_summary(self) -> list[StoreCategorySummary]:
        """Aggregate unit volume, zero-sale intermittency, and revenue proxy by store and category."""
        self.ensure_initialized()
        rows = self._fetch_dicts(
            """
            SELECT
                store_id,
                state_id,
                cat_id,
                COUNT(DISTINCT item_id) AS sku_count,
                COUNT(*) AS row_count,
                CAST(SUM(sales) AS BIGINT) AS total_units,
                ROUND(AVG(sales), 4) AS avg_daily_units,
                ROUND(AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END), 4) AS zero_sale_rate,
                ROUND(AVG(sell_price), 4) AS avg_sell_price,
                ROUND(SUM(sales * sell_price), 2) AS total_revenue_proxy
            FROM curated_sales
            GROUP BY store_id, state_id, cat_id
            ORDER BY store_id, cat_id
            """
        )
        return [StoreCategorySummary(**row) for row in rows]

    def get_sku_series(
        self,
        store_id: str,
        item_id: str,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> list[DailySalesRecord]:
        """Return chronological daily sales history for a single (store_id, item_id)."""
        self.ensure_initialized()
        clauses = ["store_id = ?", "item_id = ?"]
        params: list[Any] = [store_id, item_id]
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(end_date)

        where_sql = " AND ".join(clauses)
        rows = self._fetch_dicts(
            f"""
            SELECT
                id, item_id, dept_id, cat_id, store_id, state_id,
                d, d_num, date, wm_yr_wk, weekday, wday, month, year,
                event_name_1, event_type_1, snap, sell_price, sales
            FROM curated_sales
            WHERE {where_sql}
            ORDER BY d_num ASC
            """,
            params,
        )
        return [DailySalesRecord(**row) for row in rows]

    def get_sku_policy(self, store_id: str, item_id: str) -> SkuPolicyRecord:
        """Fetch operational cost and inventory policy for one store-SKU."""
        self.ensure_initialized()
        rows = self._fetch_dicts(
            """
            SELECT
                store_id, item_id, dept_id, cat_id, state_id,
                avg_sell_price, unit_cost, daily_holding_cost, stockout_penalty,
                lead_time_days, current_on_hand, moq, max_order_qty
            FROM sku_policies
            WHERE store_id = ? AND item_id = ?
            """,
            [store_id, item_id],
        )
        if not rows:
            raise KeyError(f"SKU policy not found for ({store_id}, {item_id})")
        return SkuPolicyRecord(**rows[0])

    def list_sku_policies(
        self,
        store_id: str | None = None,
        cat_id: str | None = None,
    ) -> list[SkuPolicyRecord]:
        """List SKU operational policies with optional store and category filters."""
        self.ensure_initialized()
        clauses: list[str] = []
        params: list[Any] = []
        if store_id is not None:
            clauses.append("store_id = ?")
            params.append(store_id)
        if cat_id is not None:
            clauses.append("cat_id = ?")
            params.append(cat_id)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._fetch_dicts(
            f"""
            SELECT
                store_id, item_id, dept_id, cat_id, state_id,
                avg_sell_price, unit_cost, daily_holding_cost, stockout_penalty,
                lead_time_days, current_on_hand, moq, max_order_qty
            FROM sku_policies
            {where_sql}
            ORDER BY store_id, item_id
            """,
            params,
        )
        return [SkuPolicyRecord(**row) for row in rows]

    def get_price_demand_history(self, store_id: str, item_id: str) -> list[PriceDemandPoint]:
        """Return chronological price and unit demand observations for a store-SKU."""
        self.ensure_initialized()
        rows = self._fetch_dicts(
            """
            SELECT date, wm_yr_wk, sell_price, sales, snap, event_name_1
            FROM curated_sales
            WHERE store_id = ? AND item_id = ?
            ORDER BY d_num ASC
            """,
            [store_id, item_id],
        )
        return [PriceDemandPoint(**row) for row in rows]

    def load_sales_dataframe(self) -> pd.DataFrame:
        """Return full curated sales table ordered deterministically for feature engineering."""
        self.ensure_initialized()
        df = self._conn.execute(
            """
            SELECT
                id, item_id, dept_id, cat_id, store_id, state_id,
                d, d_num, date, wm_yr_wk, weekday, wday, month, year,
                event_name_1, event_type_1, snap, sell_price, sales
            FROM curated_sales
            ORDER BY store_id ASC, item_id ASC, d_num ASC
            """
        ).df()
        df["date"] = pd.to_datetime(df["date"])
        return df
