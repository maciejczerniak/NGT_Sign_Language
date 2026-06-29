"""Shared fixtures for auth tests.

Uses SQLite in-memory — no live Postgres needed.
AppState is set manually to bypass lifespan (ASGITransport doesn't trigger it).
All user-creating fixtures are session-scoped to match the session-scoped DB.
"""

from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from sign_language.auth.models import User
from sign_language.api.state import AppState
from sign_language.db.engine import Base, get_async_session

# ── In-memory SQLite ─────────────────────────────────────────────────
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(
    TEST_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestSessionLocal = async_sessionmaker(
    test_engine, expire_on_commit=False, class_=AsyncSession
)


async def override_get_async_session() -> AsyncIterator[AsyncSession]:
    async with TestSessionLocal() as session:
        yield session


def _mock_loaded_models() -> MagicMock:
    m = MagicMock()
    m.device = "cpu"
    m.class_names = ["A", "B", "C"]
    m.model = MagicMock()
    m.landmark_model = None
    m.lm_class_names = []
    m.hands_detector = None
    return m


# ── App — session-scoped, AppState set manually ──────────────────────
@pytest_asyncio.fixture(scope="session")
async def app():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    from sign_language.api.app import create_app

    _app = create_app()
    _app.dependency_overrides[get_async_session] = override_get_async_session

    # Set AppState directly — ASGITransport never runs lifespan
    _app.state.app_state = AppState(models=_mock_loaded_models())

    yield _app

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


# ── HTTP client — session-scoped to match app ────────────────────────
@pytest_asyncio.fixture(scope="session")
async def client(app) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


# ── Users — session-scoped, registered once per session ──────────────
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "Str0ngPassword!"
ADMIN_EMAIL = "admin@example.com"


@pytest_asyncio.fixture(scope="session")
async def registered_user(client: AsyncClient) -> dict:
    resp = await client.post(
        "/api/auth/register",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest_asyncio.fixture(scope="session")
async def user_token(client: AsyncClient, registered_user) -> str:
    resp = await client.post(
        "/api/auth/jwt/login",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest_asyncio.fixture(scope="session")
async def admin_token(client: AsyncClient) -> str:
    """Register a separate admin user and promote it — avoids role
    collision with the regular test user."""
    # Register
    resp = await client.post(
        "/api/auth/register",
        json={"email": ADMIN_EMAIL, "password": TEST_PASSWORD},
    )
    assert resp.status_code == 201, resp.text

    # Promote
    async with TestSessionLocal() as session:
        await session.execute(
            update(User).where(User.email == ADMIN_EMAIL).values(is_superuser=True)
        )
        await session.commit()

    # Login
    resp = await client.post(
        "/api/auth/jwt/login",
        data={"username": ADMIN_EMAIL, "password": TEST_PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# ── Sync client — for WebSocket tests ────────────────────────────────
@pytest.fixture(scope="session")
def sync_client(app):
    from unittest.mock import patch
    from starlette.testclient import TestClient

    with patch(
        "sign_language.api.app.load_all",
        return_value=_mock_loaded_models(),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c
