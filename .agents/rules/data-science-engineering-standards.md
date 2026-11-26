# StockPulse AI — Data Science & Engineering Standards

## 1. Strict Layered Architecture

All code in this repository MUST adhere to strict separation of concerns across layers:

- `scripts/` — Offline ETL scripts and curated dataset builders (e.g., `ingest_m5.py`). Must not contain serving routes or UI logic.
- `core/` — Pure domain logic, analytical DuckDB warehouse (`warehouse.py`), leak-free temporal feature engineering (`features.py`), LightGBM P10/P50/P90 quantile forecaster, IsolationForest anomaly + PSI drift detector, SHAP TreeExplainer, SciPy constrained MILP inventory optimizer, and grounded LLM executive briefing generator. Must not import `fastapi` or `streamlit`.
- `api/` — FastAPI HTTP layer exposing typed endpoints backed by `core/` and Pydantic v2 schemas.
- `ui/` — Streamlit + Plotly interactive command center. **MUST communicate strictly via HTTP (`httpx`) to FastAPI** and **MUST NEVER import `core/` directly**.
- `eval/` — Offline backtesting benchmark comparing LightGBM P10/P50/P90 + SciPy MILP optimizer against seasonal naive and rule-of-thumb reorder baselines.
- `tests/` — 100% offline `pytest` verification suite using deterministic seeds and synthetic/local curated fixtures (zero external network or paid LLM calls).

## 2. Typed Contracts (Pydantic v2)

- Enforce Pydantic v2 (`pydantic>=2.7`) models across all forecast payloads, optimization constraints, what-if scenario inputs, SHAP attributions, warehouse query records, and FastAPI request/response payloads.
- Reject undeclared fields (`extra="forbid"`) on API/domain contracts where appropriate, and validate domain bounds (non-negative sales, positive prices, valid quantile ordering `p10 <= p50 <= p90`).

## 3. Zero Target Leakage & Temporal Integrity

- All lag (`lag_1`, `lag_7`, `lag_14`, `lag_28`) and rolling statistics MUST shift strictly past-only (`shift(>=1)`) before any rolling window aggregation.
- Price normalization features (such as `price_vs_item_avg`) MUST use strictly causal expanding windows up to time `t`, never full-horizon future means.
- Train/validation splits MUST be strictly chronological (no random shuffling), holding out the official 28-day horizon (`train_max_date < val_min_date`).

## 4. Reproducibility & Repository Hygiene

- CPU-first \$0 stack: no GPU-only dependencies or paid external databases required to clone, test, or deploy.
- Raw Kaggle M5 CSVs (`data/raw/*`) and generated DuckDB files (`*.duckdb`) MUST remain gitignored.
- Curated Parquet datasets (`data/processed/*.parquet`) MUST remain tracked in Git for zero-setup cloning and Streamlit Community Cloud deployment.
