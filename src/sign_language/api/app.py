"""FastAPI application factory for the Sign Language API.

This module defines two entry points:

- :func:`lifespan` — an async context manager that loads ML models on
  startup and disposes of the database engine on shutdown.
- :func:`create_app` — the application factory that wires together
  middleware, authentication routers, API routers, and the static
  frontend file server.

The factory pattern (rather than a module-level ``app`` instance) allows
the application to be instantiated multiple times in tests with different
configurations, and is compatible with ``uvicorn``'s ``--factory`` flag.

Database schema management is intentionally left to Alembic. The
``create_all`` call is omitted here so that the schema has exactly one
source of truth. Run ``alembic upgrade head`` before starting the server.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sign_language.core.settings import settings
from sign_language.db.engine import engine
from sign_language.models.loader import load_all

from sign_language.auth.schemas import UserCreate, UserRead, UserUpdate
from sign_language.auth.users import auth_backend, fastapi_users
from .monitoring import MonitoringMiddleware
from .routes import router
from .shared_routes import shared_router
from .state import AppState
from .ws import ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    On startup, loads all ML models via :func:`load_all` and attaches the
    resulting :class:`AppState` instance to ``app.state.app_state`` so
    that it is accessible from every request handler via
    ``request.app.state.app_state``.

    On shutdown (after the ``yield``), disposes of the SQLAlchemy async
    engine to cleanly close all database connections in the connection
    pool.

    Database schema management is handled exclusively by Alembic.
    ``create_all`` is not called here — run ``alembic upgrade head``
    before starting the server to ensure the schema is up to date.

    Args:
        app: The FastAPI application instance. Passed automatically by
            FastAPI when the lifespan context manager is registered.

    Yields:
        Nothing. Control returns to FastAPI between the startup and
        shutdown phases.
    """

    logger.info("Loading models …")
    models = load_all()
    app.state.app_state = AppState(models=models)
    logger.info("Models ready.")

    yield

    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Registers all middleware, authentication routers, API routers, and
    the static frontend file server in the correct order. Should be
    called once per process, typically via ``uvicorn`` with the
    ``--factory`` flag:

    .. code-block:: bash

        uvicorn sign_language.api:create_app --factory --reload

    **Middleware (applied in reverse registration order):**

    - :class:`~fastapi.middleware.cors.CORSMiddleware` — handles
      cross-origin requests. When ``settings.cors_origins`` is set,
      only the listed origins are allowed with credentials. When unset,
      all origins are permitted without credentials.
    - :class:`.MonitoringMiddleware` — records per-request latency and
      status codes to the ``monitoring_events`` PostgreSQL table.

    **Router registration order:**

    Auth routers are registered before the static file mount so that
    ``/api/auth/*`` and ``/api/users/*`` paths are matched by their
    handlers rather than falling through to the frontend.

    The WebSocket router is registered without a prefix; the prediction
    endpoint lives at ``/ws/predict``.

    If the frontend build directory configured in settings does not
    exist, the application starts in API-only mode and logs a warning.

    Returns:
        A fully configured :class:`fastapi.FastAPI` instance ready to
        be served by an ASGI server.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
    )

    configured_origins = settings.cors_origins
    origins = configured_origins or ["*"]
    allow_credentials = bool(configured_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(MonitoringMiddleware)

    # ── Auth routers (before the catch-all static mount) ────────────
    app.include_router(
        fastapi_users.get_auth_router(auth_backend),
        prefix="/api/auth/jwt",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/api/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_reset_password_router(),
        prefix="/api/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_verify_router(UserRead),
        prefix="/api/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate),
        prefix="/api/users",
        tags=["users"],
    )

    # ── App routers ─────────────────────────────────────────────────
    app.include_router(router, prefix="/api")
    app.include_router(shared_router, prefix="/api")
    app.include_router(ws_router)  # /ws/predict — no prefix

    if settings.frontend_build_dir.exists():
        app.mount(
            "/",
            StaticFiles(directory=settings.frontend_build_dir, html=True),
            name="frontend",
        )
    else:
        logger.info(
            "Frontend build dir not found (%s) — API-only mode.",
            settings.frontend_build_dir,
        )

    return app
