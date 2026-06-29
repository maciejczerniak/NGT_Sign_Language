"""Tests for POST /api/auth/register."""

import pytest
from httpx import AsyncClient


async def test_register_success(client: AsyncClient):
    resp = await client.post(
        "/api/auth/register",
        json={"email": "new@example.com", "password": "Str0ngPassword!"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "new@example.com"
    assert body["is_active"] is True
    assert body["is_superuser"] is False
    assert "id" in body
    # hashed_password must never be exposed
    assert "hashed_password" not in body
    assert "password" not in body


async def test_register_duplicate_email(client: AsyncClient, registered_user):
    resp = await client.post(
        "/api/auth/register",
        json={"email": "test@example.com", "password": "Str0ngPassword!"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "REGISTER_USER_ALREADY_EXISTS"


async def test_register_invalid_email(client: AsyncClient):
    resp = await client.post(
        "/api/auth/register",
        json={"email": "not-an-email", "password": "Str0ngPassword!"},
    )
    assert resp.status_code == 422


async def test_register_missing_password(client: AsyncClient):
    resp = await client.post(
        "/api/auth/register",
        json={"email": "nopw@example.com"},
    )
    assert resp.status_code == 422


async def test_register_empty_payload(client: AsyncClient):
    resp = await client.post("/api/auth/register", json={})
    assert resp.status_code == 422
