"""Unit tests for the operational monitoring module.

Covers the dialect-independent logic of ``monitoring.py`` (endpoint
mapping, error-swallowing, fallback values, middleware behaviour) using
the same mock-based approach as the rest of the API test suite.

The real SQL aggregation (``percentile_cont``, time bucketing) is
PostgreSQL-specific and cannot run on the SQLite test database, so those
assertions live in ``TestRealPostgresAggregation`` and only run when a
disposable Postgres test database is provided via the
``MONITORING_TEST_PG_URL`` environment variable. They skip otherwise, so
the suite stays green in CI where no Postgres instance is available.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import sign_language.api.monitoring as monitoring

# Keys every get_metrics() response must expose — the contract the
# frontend dashboard depends on.
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


# ============================================================================
# track_prediction — endpoint-to-type mapping and argument forwarding
# ============================================================================


class TestTrackPrediction:
    async def test_http_endpoint_maps_to_http_type(self):
        """http_predict source is recorded with endpoint_type='http'."""
        with patch.object(monitoring, "_insert_event", new=AsyncMock()) as mock_insert:
            await monitoring.track_prediction("A", 0.9, "http_predict", entropy=0.5)

        mock_insert.assert_awaited_once_with(
            path="/prediction",
            method="INFERENCE",
            status_code=200,
            latency_ms=0.0,
            predicted_letter="A",
            confidence=0.9,
            endpoint_type="http",
            entropy=0.5,
        )

    async def test_ws_endpoint_maps_to_ws_type(self):
        """ws_predict source is recorded with endpoint_type='ws'."""
        with patch.object(monitoring, "_insert_event", new=AsyncMock()) as mock_insert:
            await monitoring.track_prediction("B", 0.7, "ws_predict", entropy=0.1)

        assert mock_insert.await_args.kwargs["endpoint_type"] == "ws"

    async def test_writes_inference_method_and_path(self):
        """Predictions are tagged method='INFERENCE', path='/prediction'."""
        with patch.object(monitoring, "_insert_event", new=AsyncMock()) as mock_insert:
            await monitoring.track_prediction("C", 0.6, "http_predict")

        kwargs = mock_insert.await_args.kwargs
        assert kwargs["method"] == "INFERENCE"
        assert kwargs["path"] == "/prediction"

    async def test_none_letter_is_forwarded(self):
        """A missing prediction (no hand) forwards letter=None."""
        with patch.object(monitoring, "_insert_event", new=AsyncMock()) as mock_insert:
            await monitoring.track_prediction(None, 0.0, "ws_predict")

        assert mock_insert.await_args.kwargs["predicted_letter"] is None


# ============================================================================
# _insert_event — database errors must never propagate
# ============================================================================


class TestInsertEventErrorHandling:
    async def test_db_error_is_swallowed(self):
        """A failure opening the session is logged, not raised."""
        with patch.object(
            monitoring, "AsyncSessionLocal", side_effect=Exception("db down")
        ):
            # Must not raise.
            await monitoring._insert_event(
                path="/x", method="GET", status_code=200, latency_ms=1.0
            )


# ============================================================================
# get_metrics — fallback contract on database failure
# ============================================================================


class TestGetMetricsFallback:
    async def test_returns_zero_dict_on_db_error(self):
        """When the DB query fails, a zero-valued dict is returned."""
        with patch.object(
            monitoring, "AsyncSessionLocal", side_effect=Exception("db down")
        ):
            result = await monitoring.get_metrics(range_str="all")

        assert result["total_requests"] == 0
        assert result["error_rate"] == 0.0
        assert result["letter_counts"] == {}
        assert result["error_breakdown"] == []

    async def test_fallback_has_all_keys(self):
        """The fallback dict exposes the full documented contract."""
        with patch.object(
            monitoring, "AsyncSessionLocal", side_effect=Exception("db down")
        ):
            result = await monitoring.get_metrics(range_str="1h")

        assert set(result.keys()) == _METRIC_KEYS


# ============================================================================
# get_metrics_history — fallback and custom-date parsing
# ============================================================================


class TestGetMetricsHistoryFallback:
    async def test_returns_empty_list_on_db_error(self):
        """A DB failure yields an empty history list, not an exception."""
        with patch.object(
            monitoring, "AsyncSessionLocal", side_effect=Exception("db down")
        ):
            result = await monitoring.get_metrics_history(range_str="1h")

        assert result == []

    async def test_bad_custom_date_does_not_raise(self):
        """Malformed custom dates fall back gracefully instead of crashing."""
        with patch.object(
            monitoring, "AsyncSessionLocal", side_effect=Exception("db down")
        ):
            # Invalid dates → parse fallback → DB error → [] (no raise).
            result = await monitoring.get_metrics_history(
                date_from="not-a-date", date_to="also-bad"
            )

        assert result == []


# ============================================================================
# MonitoringMiddleware — latency/status recording and 500 handling
# ============================================================================


class TestMonitoringMiddleware:
    async def test_records_event_on_success(self):
        """A successful request is recorded with its path, method, status."""
        app = FastAPI()
        app.add_middleware(monitoring.MonitoringMiddleware)

        @app.get("/ping")
        async def ping() -> dict:
            return {"ok": True}

        with patch.object(monitoring, "_insert_event", new=AsyncMock()) as mock_insert:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as c:
                resp = await c.get("/ping")

        assert resp.status_code == 200
        mock_insert.assert_awaited_once()
        kwargs = mock_insert.await_args.kwargs
        assert kwargs["path"] == "/ping"
        assert kwargs["method"] == "GET"
        assert kwargs["status_code"] == 200

    async def test_records_500_on_handler_error(self):
        """An unhandled handler error is recorded as a 500 event."""
        app = FastAPI()
        app.add_middleware(monitoring.MonitoringMiddleware)

        @app.get("/boom")
        async def boom() -> dict:
            raise RuntimeError("kaboom")

        with patch.object(monitoring, "_insert_event", new=AsyncMock()) as mock_insert:
            transport = ASGITransport(app=app, raise_app_exceptions=False)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                await c.get("/boom")

        mock_insert.assert_awaited_once()
        assert mock_insert.await_args.kwargs["status_code"] == 500


# ============================================================================
# Real PostgreSQL aggregation — opt-in, skips without a test DB
# ============================================================================

PG_TEST_URL = os.environ.get("MONITORING_TEST_PG_URL")

_MARKER = "/__mon_test__"  # tags rows this test owns, for safe cleanup


@pytest_asyncio.fixture
async def pg_metrics_db():
    """Seed a disposable Postgres DB with known rows and point monitoring at it.

    Requires ``MONITORING_TEST_PG_URL`` to reference an EMPTY, dedicated
    test database (e.g. ``postgresql+asyncpg://signlang:signlang@localhost
    :5432/signlang_test``). Aggregations run over every row in the table,
    so a non-empty database will make the assertions fail.

    Inserted rows are tagged with a marker path and deleted on teardown;
    the test never truncates or drops the table.
    """
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from sign_language.auth.models import MonitoringEvent
    from sign_language.db.engine import Base

    engine = create_async_engine(PG_TEST_URL)
    sessionmaker = async_sessionmaker(
        engine, expire_on_commit=False, class_=AsyncSession
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    now = datetime.now(timezone.utc)

    def _row(**kw) -> "MonitoringEvent":
        base = dict(id=uuid.uuid4(), timestamp=now, path=_MARKER)
        base.update(kw)
        return MonitoringEvent(**base)

    rows = [
        # Request rows: latencies 100/200/300/400, one is an error (500).
        _row(method="GET", status_code=200, latency_ms=100.0),
        _row(method="GET", status_code=200, latency_ms=200.0),
        _row(method="GET", status_code=200, latency_ms=300.0),
        _row(method="GET", status_code=500, latency_ms=400.0),
        # Inference rows: two "A", one "B".
        _row(
            method="INFERENCE",
            status_code=200,
            latency_ms=0.0,
            predicted_letter="A",
            confidence=0.9,
            endpoint_type="http",
            entropy=0.2,
        ),
        _row(
            method="INFERENCE",
            status_code=200,
            latency_ms=0.0,
            predicted_letter="A",
            confidence=0.8,
            endpoint_type="http",
            entropy=0.4,
        ),
        _row(
            method="INFERENCE",
            status_code=200,
            latency_ms=0.0,
            predicted_letter="B",
            confidence=0.6,
            endpoint_type="ws",
            entropy=0.6,
        ),
    ]

    async with sessionmaker() as session:
        session.add_all(rows)
        await session.commit()

    with patch.object(monitoring, "AsyncSessionLocal", sessionmaker):
        yield

    async with sessionmaker() as session:
        await session.execute(
            delete(MonitoringEvent).where(MonitoringEvent.path == _MARKER)
        )
        await session.commit()
    await engine.dispose()


@pytest.mark.skipif(
    not PG_TEST_URL,
    reason="Set MONITORING_TEST_PG_URL to a disposable Postgres test DB to "
    "verify real SQL aggregation (percentiles, counts, entropy).",
)
class TestRealPostgresAggregation:
    async def test_request_metrics(self, pg_metrics_db):
        """Request counts, error rate, and latency percentiles are correct."""
        result = await monitoring.get_metrics(range_str="all")

        assert result["total_requests"] == 4
        assert result["error_count"] == 1
        assert result["error_rate"] == 0.25
        assert result["avg_latency_ms"] == 250.0
        assert result["p50_latency_ms"] == 250.0
        assert result["p95_latency_ms"] == 385.0

    async def test_prediction_metrics(self, pg_metrics_db):
        """Prediction counts, confidence, entropy, and letter counts match."""
        result = await monitoring.get_metrics(range_str="all")

        assert result["total_predictions"] == 3
        assert result["avg_confidence"] == round((0.9 + 0.8 + 0.6) / 3, 4)
        assert result["avg_entropy"] == 0.4
        assert result["letter_counts"] == {"A": 2, "B": 1}

    async def test_error_breakdown(self, pg_metrics_db):
        """The single 500 request appears in the error breakdown."""
        result = await monitoring.get_metrics(range_str="all")

        assert result["error_breakdown"] == [
            {"path": _MARKER, "status_code": 500, "count": 1}
        ]
