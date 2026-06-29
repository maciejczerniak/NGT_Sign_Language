"""
Operational monitoring for the Sign Language API.

Persists the following events to the ``monitoring_events`` PostgreSQL
table:

- Every HTTP request — path, method, status code, latency in ms.
- Every inference result — predicted letter, confidence score,
  endpoint type (``"http"`` or ``"ws"``), Shannon entropy over top-3.

The ``/api/admin/metrics`` endpoint in ``routes.py`` queries this table
and returns aggregated stats to the frontend admin dashboard.

Usage
-----
1. Add ``MonitoringMiddleware`` to the FastAPI app in ``app.py``.
2. Call ``track_prediction()`` from ``routes.py`` and ``ws.py`` after
   each inference call.
3. Read aggregated stats via the ``get_metrics()`` helper.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

import sqlalchemy as sa
from fastapi import Request, Response
from sqlalchemy import func, select
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from sign_language.auth.models import MonitoringEvent
from sign_language.db.engine import AsyncSessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _insert_event(
    path: str,
    method: str,
    status_code: int,
    latency_ms: float,
    predicted_letter: str | None = None,
    confidence: float | None = None,
    endpoint_type: str | None = None,
    entropy: float | None = None,
) -> None:
    """Write one monitoring event row to the database.

    Opens its own session rather than relying on dependency injection
    because this function is called from middleware and from the
    prediction handlers, where FastAPI's DI is not available.

    The write is awaited inline by its callers, but is intentionally
    best-effort: any exception is logged and swallowed so that a
    monitoring failure never propagates to the caller or affects the
    user-facing response.

    Args:
        path: Request path, e.g. ``/api/predict``.
        method: HTTP method, e.g. ``POST``. Use ``"INFERENCE"`` for
            prediction-only rows that are not tied to an HTTP request.
        status_code: HTTP response status code.
        latency_ms: Wall-clock request duration in milliseconds.
        predicted_letter: Top-1 predicted letter, or ``None`` when no
            hand was detected or the row is a plain HTTP event.
        confidence: Model confidence score in the range ``[0, 1]``,
            or ``None`` for non-inference rows.
        endpoint_type: ``"http"`` for REST predictions, ``"ws"`` for
            WebSocket predictions, or ``None`` for non-inference rows.
        entropy: Shannon entropy calculated over the normalised top-3
            class probabilities, or ``None`` when not available.
    """
    event = MonitoringEvent(
        id=uuid.uuid4(),
        timestamp=datetime.now(timezone.utc),
        path=path,
        method=method,
        status_code=status_code,
        latency_ms=latency_ms,
        predicted_letter=predicted_letter,
        confidence=confidence,
        endpoint_type=endpoint_type,
        entropy=entropy,
    )
    try:
        async with AsyncSessionLocal() as session:
            session.add(event)
            await session.commit()
    except Exception:
        logger.exception("Failed to persist monitoring event — skipping.")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def track_prediction(
    letter: str | None,
    confidence: float,
    endpoint: str,
    entropy: float | None = None,
) -> None:
    """Record a single inference result in the monitoring database.

    Should be called from ``routes.py`` and ``ws.py`` immediately after
    each successful inference call so that confidence and entropy trends
    can be tracked over time. These trends serve as unsupervised proxies
    for data and concept drift — they do not replace labelled accuracy
    measurement.

    The row is written with ``method="INFERENCE"`` and
    ``path="/prediction"`` so that it can be distinguished from ordinary
    HTTP request rows in all downstream queries.

    Args:
        letter: Top-1 predicted letter, or ``None`` if no hand was
            detected in the frame.
        confidence: Model confidence score for the top-1 prediction,
            in the range ``[0, 1]``.
        endpoint: Source of the prediction — either ``"http_predict"``
            for REST requests or ``"ws_predict"`` for WebSocket frames.
        entropy: Shannon entropy calculated over the normalised top-3
            class probabilities at inference time. Computed as
            ``-sum(p * log2(p) for p in normalised_top3)``, where the
            top-3 probabilities are normalised to sum to 1 before
            applying the formula. ``None`` if not available.
    """
    endpoint_type = "http" if endpoint == "http_predict" else "ws"
    await _insert_event(
        path="/prediction",
        method="INFERENCE",
        status_code=200,
        latency_ms=0.0,
        predicted_letter=letter,
        confidence=confidence,
        endpoint_type=endpoint_type,
        entropy=entropy,
    )
    logger.debug(
        "Prediction tracked: letter=%s confidence=%.4f entropy=%s endpoint=%s",
        letter,
        confidence,
        f"{entropy:.4f}" if entropy is not None else "None",
        endpoint,
    )


async def get_metrics(
    range_str: str = "all",
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict:
    """Query the database and return aggregated operational metrics.

    Returns a single snapshot of all key metrics for the selected time
    window, suitable for populating the KPI cards on the admin dashboard.
    When ``range_str`` is ``"all"``, no time filter is applied and the
    query covers the full history of recorded events.

    Supports the same two time-window modes as :func:`get_metrics_history`:
    a preset ``range_str`` (e.g. ``"7d"``), or a custom range given by
    ``date_from`` and ``date_to``. When both custom dates are provided
    they take priority and ``range_str`` is ignored, so the KPI cards stay
    consistent with the history charts.

    Metrics computed from non-inference rows (HTTP requests):

    - Total request count.
    - Error count and error rate (status code >= 400).
    - Average, p50, and p95 response latency in milliseconds.

    Metrics computed from inference rows:

    - Total prediction count.
    - Average model confidence score.
    - Average Shannon entropy over top-3 class probabilities.
    - Per-letter prediction counts.
    - Error breakdown grouped by endpoint path and status code.

    Args:
        range_str: Time window for the query. Accepted values are
            ``"1h"``, ``"6h"``, ``"1d"``, ``"7d"``, ``"30d"``, or
            ``"all"`` (no time filter). Ignored when ``date_from`` and
            ``date_to`` are both provided.
        date_from: Start of a custom date range in ISO format, e.g.
            ``"2026-05-01"``. Optional.
        date_to: End of a custom date range in ISO format, e.g.
            ``"2026-05-30"``. Optional. Must be provided together with
            ``date_from``. Falls back to the preset range on parse error.

    Returns:
        Dictionary ready to be serialised as a JSON response. Keys:
        ``total_requests``, ``error_count``, ``error_rate``,
        ``avg_latency_ms``, ``p50_latency_ms``, ``p95_latency_ms``,
        ``avg_confidence``, ``avg_entropy``, ``total_predictions``,
        ``letter_counts``, ``error_breakdown``. On any database error,
        a zero-valued fallback dictionary is returned instead of raising.
    """
    try:
        async with AsyncSessionLocal() as session:
            # Build optional time filter. A custom date range takes
            # priority over the preset range, mirroring get_metrics_history
            # so the KPI cards and the history charts cover the same window.
            range_map = {
                "1h": 3600,
                "6h": 21600,
                "1d": 86400,
                "7d": 604800,
                "30d": 2592000,
            }

            def _preset_filter() -> list:
                if range_str in range_map:
                    seconds = range_map[range_str]
                    return [
                        MonitoringEvent.timestamp
                        >= sa.func.now() - sa.text(f"interval '{seconds} seconds'")
                    ]
                return []

            time_filter: list = []
            if date_from and date_to:
                try:
                    dt_from = datetime.fromisoformat(date_from).replace(
                        tzinfo=timezone.utc
                    )
                    dt_to = datetime.fromisoformat(date_to).replace(
                        hour=23, minute=59, second=59, tzinfo=timezone.utc
                    )
                    time_filter = [
                        MonitoringEvent.timestamp >= dt_from,
                        MonitoringEvent.timestamp <= dt_to,
                    ]
                except ValueError:
                    logger.warning(
                        "Invalid date_from/date_to format — "
                        "falling back to range '%s'.",
                        range_str,
                    )
                    time_filter = _preset_filter()
            else:
                time_filter = _preset_filter()

            # ── Request metrics ──────────────────────────────────────
            request_rows = await session.execute(
                select(
                    func.count(MonitoringEvent.id).label("total"),
                    func.sum(
                        sa.case(
                            (MonitoringEvent.status_code >= 400, 1),
                            else_=0,
                        )
                    ).label("errors"),
                    func.avg(MonitoringEvent.latency_ms).label("avg_latency"),
                    func.percentile_cont(0.5)
                    .within_group(MonitoringEvent.latency_ms)
                    .label("p50_latency"),
                    func.percentile_cont(0.95)
                    .within_group(MonitoringEvent.latency_ms)
                    .label("p95_latency"),
                ).where(MonitoringEvent.method != "INFERENCE", *time_filter)
            )
            req = request_rows.one()

            total: int = int(req.total or 0)
            errors: int = int(req.errors or 0)
            avg_latency: float = round(float(req.avg_latency or 0.0), 2)
            p50_latency: float = round(float(req.p50_latency or 0.0), 2)
            p95_latency: float = round(float(req.p95_latency or 0.0), 2)
            error_rate: float = round(errors / total, 4) if total > 0 else 0.0

            # ── Prediction metrics ───────────────────────────────────
            pred_rows = await session.execute(
                select(
                    func.count(MonitoringEvent.id).label("total_predictions"),
                    func.avg(MonitoringEvent.confidence).label("avg_confidence"),
                    func.avg(MonitoringEvent.entropy).label("avg_entropy"),
                ).where(MonitoringEvent.method == "INFERENCE", *time_filter)
            )
            pred = pred_rows.one()

            total_predictions: int = int(pred.total_predictions or 0)
            avg_confidence: float = round(float(pred.avg_confidence or 0.0), 4)
            avg_entropy: float = round(float(pred.avg_entropy or 0.0), 4)

            # ── Per-letter counts ────────────────────────────────────
            letter_rows = await session.execute(
                select(
                    MonitoringEvent.predicted_letter,
                    func.count(MonitoringEvent.id).label("count"),
                )
                .where(
                    MonitoringEvent.method == "INFERENCE",
                    MonitoringEvent.predicted_letter.is_not(None),
                    *time_filter,
                )
                .group_by(MonitoringEvent.predicted_letter)
            )
            letter_counts = {row.predicted_letter: row.count for row in letter_rows}

            # ── Error breakdown by endpoint ──────────────────────────
            error_rows = await session.execute(
                select(
                    MonitoringEvent.path,
                    MonitoringEvent.status_code,
                    func.count(MonitoringEvent.id).label("count"),
                )
                .where(
                    MonitoringEvent.method != "INFERENCE",
                    MonitoringEvent.status_code >= 400,
                    *time_filter,
                )
                .group_by(MonitoringEvent.path, MonitoringEvent.status_code)
                .order_by(func.count(MonitoringEvent.id).desc())
            )
            error_breakdown = [
                {
                    "path": row.path,
                    "status_code": row.status_code,
                    "count": row.count,
                }
                for row in error_rows
            ]

    except Exception:
        logger.exception("Failed to query monitoring metrics.")
        return {
            "total_requests": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "avg_confidence": 0.0,
            "avg_entropy": 0.0,
            "total_predictions": 0,
            "letter_counts": {},
            "error_breakdown": [],
        }

    return {
        "total_requests": total,
        "error_count": errors,
        "error_rate": error_rate,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "avg_confidence": avg_confidence,
        "avg_entropy": avg_entropy,
        "total_predictions": total_predictions,
        "letter_counts": letter_counts,
        "error_breakdown": error_breakdown,
    }


async def get_metrics_history(
    range_str: str = "1h",
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict]:
    """Return time-bucketed request, latency, and entropy history.

    Divides the selected time window into approximately 60 equal buckets
    and returns per-bucket aggregates, suitable for rendering stock-style
    time-series charts on the admin dashboard.

    Entropy is stored on ``INFERENCE`` rows rather than request rows, so
    it is queried in a separate subquery and joined to the request buckets
    by timestamp. Buckets with no inference activity receive an entropy
    value of ``0.0``.

    Supports two time-window modes:

    - **Preset range** — pass a ``range_str`` value such as ``"1h"`` or
      ``"7d"``. Bucket size is chosen automatically to produce roughly
      60 data points.
    - **Custom range** — pass both ``date_from`` and ``date_to`` as ISO
      date strings (e.g. ``"2026-05-01"``). When both are provided,
      ``range_str`` is ignored. The end date is automatically extended
      to 23:59:59 UTC so the full day is included. Bucket size scales
      with the span of the custom range.

    Args:
        range_str: Preset time window. Accepted values are ``"1h"``,
            ``"6h"``, ``"1d"``, ``"7d"``, ``"30d"``, or ``"all"``.
            Ignored when ``date_from`` and ``date_to`` are both provided.
        date_from: Start of a custom date range in ISO format, e.g.
            ``"2026-05-01"``. Optional.
        date_to: End of a custom date range in ISO format, e.g.
            ``"2026-05-30"``. Optional. Must be provided together with
            ``date_from``. Falls back to the preset range on parse error.

    Returns:
        List of dicts, one per time bucket, ordered chronologically.
        Each dict contains: ``timestamp`` (ISO string), ``request_count``,
        ``error_count``, ``avg_latency_ms``, ``p50_latency_ms``,
        ``p95_latency_ms``, ``avg_entropy``. Returns an empty list on
        any database error.
    """
    range_map = {
        "1h": (3600, 60),
        "6h": (21600, 360),
        "1d": (86400, 1440),
        "7d": (604800, 10080),
        "30d": (2592000, 43200),
        "all": (None, 43200),
    }

    dt_from: datetime | None = None
    dt_to: datetime | None = None

    # Custom date range takes priority over preset range
    if date_from and date_to:
        try:
            dt_from = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
            dt_to = datetime.fromisoformat(date_to).replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            delta_seconds = int((dt_to - dt_from).total_seconds())
            bucket_seconds = max(60, delta_seconds // 60)
            total_seconds = None
        except ValueError:
            logger.warning(
                "Invalid date_from/date_to format — falling back to 1h range."
            )
            dt_from = None
            dt_to = None
            total_seconds, bucket_seconds = range_map.get(range_str, (3600, 60))
    else:
        total_seconds, bucket_seconds = range_map.get(range_str, (3600, 60))

    try:
        async with AsyncSessionLocal() as session:
            bucket_expr = sa.func.to_timestamp(
                sa.func.floor(
                    sa.func.extract("epoch", MonitoringEvent.timestamp) / bucket_seconds
                )
                * bucket_seconds
            )

            # Build time conditions used by both queries
            if dt_from and dt_to:
                time_conditions = [
                    MonitoringEvent.timestamp >= dt_from,
                    MonitoringEvent.timestamp <= dt_to,
                ]
            elif total_seconds is not None:
                time_conditions = [
                    MonitoringEvent.timestamp
                    >= sa.func.now() - sa.text(f"interval '{total_seconds} seconds'"),
                ]
            else:
                # "all" range — no time filter
                time_conditions = []

            # ── Entropy subquery (INFERENCE rows only) ───────────────
            # Entropy lives on INFERENCE rows, not on request rows, so
            # we query them separately and join by bucket.
            entropy_rows = await session.execute(
                select(
                    bucket_expr.label("bucket"),
                    func.avg(MonitoringEvent.entropy).label("avg_entropy"),
                )
                .where(
                    MonitoringEvent.method == "INFERENCE",
                    *time_conditions,
                )
                .group_by("bucket")
                .order_by("bucket")
            )
            entropy_by_bucket = {
                row.bucket: round(float(row.avg_entropy or 0.0), 4)
                for row in entropy_rows
            }

            # ── Request/latency history (non-INFERENCE rows) ─────────
            rows = await session.execute(
                select(
                    bucket_expr.label("bucket"),
                    func.count(MonitoringEvent.id).label("request_count"),
                    func.sum(
                        sa.case(
                            (MonitoringEvent.status_code >= 400, 1),
                            else_=0,
                        )
                    ).label("error_count"),
                    func.avg(MonitoringEvent.latency_ms).label("avg_latency"),
                    func.percentile_cont(0.5)
                    .within_group(MonitoringEvent.latency_ms)
                    .label("p50_latency"),
                    func.percentile_cont(0.95)
                    .within_group(MonitoringEvent.latency_ms)
                    .label("p95_latency"),
                )
                .where(
                    MonitoringEvent.method != "INFERENCE",
                    *time_conditions,
                )
                .group_by("bucket")
                .order_by("bucket")
            )

            return [
                {
                    "timestamp": row.bucket.isoformat(),
                    "request_count": int(row.request_count or 0),
                    "error_count": int(row.error_count or 0),
                    "avg_latency_ms": round(float(row.avg_latency or 0.0), 2),
                    "p50_latency_ms": round(float(row.p50_latency or 0.0), 2),
                    "p95_latency_ms": round(float(row.p95_latency or 0.0), 2),
                    "avg_entropy": entropy_by_bucket.get(row.bucket, 0.0),
                }
                for row in rows
            ]

    except Exception:
        logger.exception("Failed to query metrics history.")
        return []


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class MonitoringMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that records latency and status for every HTTP request.

    Attach to the FastAPI application via
    ``app.add_middleware(MonitoringMiddleware)`` in ``app.py``.

    For each incoming request the middleware:

    1. Records the wall-clock start time before passing the request to
       the next handler.
    2. Calls the next handler and captures the response status code and
       elapsed time.
    3. Awaits a write of a ``MonitoringEvent`` row to PostgreSQL before
       returning the response.
    4. Logs an ``ERROR``-level message for any 4xx or 5xx response, and
       a ``DEBUG``-level message for successful responses.

    If the downstream handler raises an unhandled exception, a 500 event
    is recorded before the exception is re-raised so that the error is
    still visible in the monitoring dashboard.

    The event write is awaited inline but is best-effort: a write failure
    is logged but never propagates to the caller or affects the HTTP
    response.
    """

    #: Request paths excluded from monitoring. The admin dashboard polls
    #: the ``/api/admin`` endpoints on a timer; recording those requests
    #: would inflate the very request-volume, latency, and error metrics
    #: the dashboard displays.
    _EXCLUDED_PREFIXES: tuple[str, ...] = ("/api/admin",)

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Measure latency, call the next handler, and persist the event.

        Requests to excluded paths (see :attr:`_EXCLUDED_PREFIXES`) are
        passed straight through without recording a monitoring event.

        Args:
            request: The incoming HTTP request.
            call_next: Callable that passes the request to the next
                middleware or route handler and returns the response.

        Returns:
            The unmodified response from the downstream handler.

        Raises:
            Exception: Any unhandled exception raised by the downstream
                handler is re-raised after the monitoring event is written.
        """
        if request.url.path.startswith(self._EXCLUDED_PREFIXES):
            return await call_next(request)

        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Unhandled exception on %s %s after %.1f ms",
                request.method,
                request.url.path,
                elapsed_ms,
            )
            await _insert_event(
                path=request.url.path,
                method=request.method,
                status_code=500,
                latency_ms=round(elapsed_ms, 2),
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        status = response.status_code

        await _insert_event(
            path=request.url.path,
            method=request.method,
            status_code=status,
            latency_ms=round(elapsed_ms, 2),
        )

        if status >= 400:
            logger.error(
                "HTTP %s  %s %s  latency=%.1f ms",
                status,
                request.method,
                request.url.path,
                elapsed_ms,
            )
        else:
            logger.debug(
                "HTTP %s  %s %s  latency=%.1f ms",
                status,
                request.method,
                request.url.path,
                elapsed_ms,
            )

        return response
