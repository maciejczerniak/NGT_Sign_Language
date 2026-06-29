"""Tests for POST /api/auth/jwt/login and /logout."""

import pytest
from httpx import AsyncClient


async def test_login_success(client: AsyncClient, registered_user):
    resp = await client.post(
        "/api/auth/jwt/login",
        data={"username": "test@example.com", "password": "Str0ngPassword!"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert body["token_type"] == "bearer"


async def test_login_wrong_password(client: AsyncClient, registered_user):
    resp = await client.post(
        "/api/auth/jwt/login",
        data={"username": "test@example.com", "password": "wrongpassword"},
    )
    assert resp.status_code == 400


async def test_login_unknown_email(client: AsyncClient):
    resp = await client.post(
        "/api/auth/jwt/login",
        data={"username": "ghost@example.com", "password": "Str0ngPassword!"},
    )
    assert resp.status_code == 400


async def test_me_authenticated(client: AsyncClient, user_token: str):
    resp = await client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "test@example.com"


async def test_me_unauthenticated(client: AsyncClient):
    resp = await client.get("/api/users/me")
    assert resp.status_code == 401


async def test_me_invalid_token(client: AsyncClient):
    resp = await client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer this.is.garbage"},
    )
    assert resp.status_code == 401
