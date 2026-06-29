"""Tests for the admin-only monitoring metrics endpoints.

Mirrors ``test_admin.py``: verifies the ``current_admin`` gating on
``/api/admin/metrics`` and ``/api/admin/metrics/history`` (401 / 403 /
200) and the response contract the dashboard relies on.

The underlying SQL is PostgreSQL-specific, so against the SQLite test
database ``get_metrics`` falls back to its zero-valued response. These
tests therefore assert the auth behaviour and the response *shape*, not
computed aggregation values (those are covered, against real Postgres,
in ``test_monitoring.py``).
"""

import pytest
from httpx import AsyncClient

_METRICS = "/api/admin/metrics"
_HISTORY = "/api/admin/metrics/history"

_METRIC_KEYS = {
    "total_requests",
    "error_count",
    "error_rate",
    "avg_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
    "avg_confidence",
    "avg_entropy",
    "total_predictions",
    "letter_counts",
    "error_breakdown",
}


class TestAdminMetricsAuth:
    async def test_metrics_no_token(self, client: AsyncClient):
        """No token — 401."""
        resp = await client.get(_METRICS)
        assert resp.status_code == 401

    async def test_metrics_regular_user(self, client: AsyncClient, user_token: str):
        """Regular user — 403."""
        resp = await client.get(
            _METRICS, headers={"Authorization": f"Bearer {user_token}"}
        )
        assert resp.status_code == 403

    async def test_metrics_superuser(self, client: AsyncClient, admin_token: str):
        """Superuser — 200."""
        resp = await client.get(
            _METRICS, headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert resp.status_code == 200

    async def test_history_no_token(self, client: AsyncClient):
        """No token — 401."""
        resp = await client.get(_HISTORY)
        assert resp.status_code == 401

    async def test_history_regular_user(self, client: AsyncClient, user_token: str):
        """Regular user — 403."""
        resp = await client.get(
            _HISTORY, headers={"Authorization": f"Bearer {user_token}"}
        )
        assert resp.status_code == 403

    async def test_history_superuser(self, client: AsyncClient, admin_token: str):
        """Superuser — 200."""
        resp = await client.get(
            _HISTORY, headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert resp.status_code == 200


class TestAdminMetricsShape:
    async def test_metrics_has_all_keys(self, client: AsyncClient, admin_token: str):
        """Metrics response exposes the full documented key set."""
        resp = await client.get(
            _METRICS, headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert set(resp.json().keys()) == _METRIC_KEYS

    async def test_metrics_container_types(self, client: AsyncClient, admin_token: str):
        """Metrics values use the expected container types."""
        body = (
            await client.get(
                _METRICS, headers={"Authorization": f"Bearer {admin_token}"}
            )
        ).json()
        assert isinstance(body["total_requests"], int)
        assert isinstance(body["letter_counts"], dict)
        assert isinstance(body["error_breakdown"], list)

    async def test_history_returns_list(self, client: AsyncClient, admin_token: str):
        """History endpoint returns a JSON list."""
        body = (
            await client.get(
                _HISTORY, headers={"Authorization": f"Bearer {admin_token}"}
            )
        ).json()
        assert isinstance(body, list)
