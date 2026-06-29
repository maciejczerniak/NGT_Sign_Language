"""API-level fixtures: a TestClient whose lifespan is fully mocked."""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from sign_language.api.app import create_app


# patch target — must match the name used *inside* app.py
_LOAD_ALL = "sign_language.api.app.load_all"


@pytest.fixture
def client(mock_models) -> TestClient:
    """
    A function-scoped TestClient.

    Each test gets a fresh FastAPI app with a fresh AppState, so smoother
    and sequence state never bleeds between tests.
    """
    with patch(_LOAD_ALL, return_value=mock_models):
        app = create_app()
        with TestClient(app) as c:
            yield c
