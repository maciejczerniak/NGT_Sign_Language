"""Tests for the training trigger FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI

from sign_language_training.trigger_api.app import create_app


def test_create_app_returns_fastapi_instance() -> None:
    app = create_app()

    assert isinstance(app, FastAPI)


def test_app_title_is_training_trigger_api() -> None:
    app = create_app()

    assert app.title == "Sign Language Training Trigger API"


def test_health_route_is_registered(client) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "training-trigger",
    }
