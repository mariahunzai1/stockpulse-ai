"""HTTP-only client for StockPulse AI UI with zero-config local Uvicorn auto-boot."""

from __future__ import annotations

import atexit
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class StockPulseApiClient:
    """Thin HTTPX wrapper communicating strictly with the FastAPI backend."""

    def __init__(
        self,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ) -> None:
        resolved_url = (
            base_url
            or os.getenv("API_BASE_URL")
            or os.getenv("STOCKPULSE_API_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.base_url = resolved_url
        self._transport = transport
        self._timeout = timeout
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._client = httpx.Client(
            base_url=self.base_url,
            transport=transport,
            timeout=timeout,
        )

    def close(self) -> None:
        """Close the HTTP client and terminate any auto-spawned Uvicorn subprocess."""
        self._client.close()
        if self._server_proc is not None and self._server_proc.poll() is None:
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_proc.kill()
            self._server_proc = None

    def _is_healthy(self) -> bool:
        try:
            resp = self._client.get("/health", timeout=2.0)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception:
            return False

    def ensure_backend_running(self, timeout_seconds: float = 45.0) -> bool:
        """Return True once FastAPI /health responds, spawning local Uvicorn if needed."""
        if self._is_healthy():
            return True

        if self._transport is not None:
            raise RuntimeError("Mock/ASGI transport backend did not respond to /health")

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "127.0.0.1"
        if host not in {"127.0.0.1", "localhost"}:
            raise RuntimeError(f"Remote FastAPI backend unreachable at {self.base_url}")

        port = parsed.port or 8000
        if self._server_proc is None or self._server_proc.poll() is not None:
            self._server_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "api.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            atexit.register(self.close)

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self._is_healthy():
                return True
            if self._server_proc.poll() is not None:
                stderr_out = ""
                if self._server_proc.stderr is not None:
                    stderr_out = self._server_proc.stderr.read().decode("utf-8", errors="ignore")
                raise RuntimeError(
                    f"Auto-started Uvicorn exited with code {self._server_proc.returncode}: "
                    f"{stderr_out[-600:]}"
                )
            time.sleep(0.5)

        raise TimeoutError(
            f"FastAPI backend did not pass /health within {timeout_seconds:.0f}s at {self.base_url}"
        )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        resp = self._client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()

    def get_health(self) -> dict[str, Any]:
        return self._get("/health")

    def get_skus(self) -> list[dict[str, Any]]:
        return self._get("/skus")

    def get_forecast(self, store_id: str, item_id: str) -> dict[str, Any]:
        return self._get(f"/forecast/{store_id}/{item_id}")

    def get_explain(
        self,
        store_id: str,
        item_id: str,
        target_date: str | None = None,
    ) -> dict[str, Any]:
        params = {"date": target_date} if target_date else None
        return self._get(f"/explain/{store_id}/{item_id}", params=params)

    def get_anomalies(self, include_normal: bool = False) -> dict[str, Any]:
        return self._get("/anomalies", params={"include_normal": include_normal})

    def post_optimize(self, constraints: dict[str, Any]) -> dict[str, Any]:
        return self._post("/optimize", constraints)

    def post_briefing(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/briefing", payload)

    def get_xai_tail_risk(
        self,
        store_id: str,
        item_id: str,
        target_date: str | None = None,
    ) -> dict[str, Any]:
        params = {"date": target_date} if target_date else None
        return self._get(f"/xai/tail-risk/{store_id}/{item_id}", params=params)

    def get_xai_pdp(self, feature_name: str) -> dict[str, Any]:
        return self._get(f"/xai/pdp/{feature_name}")

    def post_xai_counterfactual(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post("/xai/counterfactual", payload)

    def post_xai_decision_audit(self, constraints: dict[str, Any]) -> dict[str, Any]:
        return self._post("/xai/decision-audit", constraints)
