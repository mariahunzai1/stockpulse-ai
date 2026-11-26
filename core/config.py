"""Central configuration for StockPulse AI using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_STORES: tuple[str, ...] = ("CA_1", "TX_1", "WI_1")
SELECTED_ITEMS: tuple[str, ...] = (
    # FOODS (4 SKUs spanning FOODS_1, FOODS_2, FOODS_3)
    "FOODS_3_090",
    "FOODS_3_586",
    "FOODS_2_052",
    "FOODS_1_180",
    # HOBBIES (3 SKUs spanning HOBBIES_1, HOBBIES_2)
    "HOBBIES_1_048",
    "HOBBIES_1_225",
    "HOBBIES_2_033",
    # HOUSEHOLD (3 SKUs spanning HOUSEHOLD_1, HOUSEHOLD_2)
    "HOUSEHOLD_1_334",
    "HOUSEHOLD_1_110",
    "HOUSEHOLD_2_037",
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="STOCKPULSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    env: str = "development"
    raw_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    duckdb_path: Path = Field(default=PROJECT_ROOT / "data" / "stockpulse.duckdb")
    artifacts_dir: Path = Field(default=PROJECT_ROOT / "artifacts")

    history_days: int = Field(default=730, ge=90)
    forecast_horizon_days: int = Field(default=28, ge=1)
    random_seed: int = 42

    selected_stores: tuple[str, ...] = SELECTED_STORES
    selected_items: tuple[str, ...] = SELECTED_ITEMS

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_base_url: str = "http://127.0.0.1:8000"

    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices(
            "LLM_API_KEY",
            "STOCKPULSE_LLM_API_KEY",
            "STOCKPULSE_OPENAI_API_KEY",
        ),
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices(
            "LLM_BASE_URL",
            "STOCKPULSE_LLM_BASE_URL",
            "STOCKPULSE_OPENAI_BASE_URL",
        ),
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        validation_alias=AliasChoices(
            "LLM_MODEL",
            "STOCKPULSE_LLM_MODEL",
            "STOCKPULSE_OPENAI_MODEL",
        ),
    )

    @property
    def curated_sales_path(self) -> Path:
        """Path to the curated Walmart M5 daily sales Parquet dataset."""
        return self.processed_data_dir / "m5_curated_sales.parquet"

    @property
    def sku_policies_path(self) -> Path:
        """Path to the curated SKU inventory policy Parquet dataset."""
        return self.processed_data_dir / "m5_sku_policies.parquet"

    @property
    def model_bundle_path(self) -> Path:
        """Path to the persisted LightGBM P10/P50/P90 quantile model bundle."""
        return self.artifacts_dir / "models" / "lgbm_quantile_bundle.joblib"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
