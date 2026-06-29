"""Tests for admin-only endpoints."""

import pytest
from httpx import AsyncClient


async def test_admin_whoami_no_token(client: AsyncClient):
    resp = await client.get("/api/admin/whoami")
    assert resp.status_code == 401


async def test_admin_whoami_regular_user(client: AsyncClient, user_token: str):
    resp = await client.get(
        "/api/admin/whoami",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 403


async def test_admin_whoami_superuser(client: AsyncClient, admin_token: str):
    resp = await client.get(
        "/api/admin/whoami",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_admin"] is True
    assert "email" in body
