"""Fixtures for the training trigger FastAPI tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sign_language_training.trigger_api.app import create_app
from sign_language_training.trigger_api.settings import settings


@pytest.fixture
def trigger_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Configure a test API key."""
    key = "test-trigger-key"
    monkeypatch.setattr(settings, "training_trigger_api_key", key)
    return key


@pytest.fixture
def client(trigger_api_key: str) -> TestClient:
    """Create a test client for the trigger API."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
