"""Tests that /predict works with and without auth (optional auth)."""

import pytest
from unittest.mock import patch, MagicMock
from httpx import AsyncClient

import torch

DUMMY_IMAGE = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="  # minimal base64

# Mock returns for inference pipeline
_MOCK_PREPROCESS = (True, torch.zeros(1, 3, 224, 224), None)
_MOCK_INFERENCE = ("A", 0.95, [{"letter": "A", "confidence": 0.95}])


@pytest.fixture(autouse=True)
def mock_inference():
    """Patch heavy ML calls for all tests in this module."""
    with (
        patch(
            "sign_language.api.routes.preprocess_image",
            return_value=_MOCK_PREPROCESS,
        ),
        patch(
            "sign_language.api.routes.run_inference",
            return_value=_MOCK_INFERENCE,
        ),
    ):
        yield


async def test_predict_anonymous(client: AsyncClient):
    """No token — should still return 200."""
    resp = await client.post("/api/predict", json={"image": DUMMY_IMAGE})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_letter"] == "A"
    assert body["hand_detected"] is True


async def test_predict_with_valid_token(client: AsyncClient, user_token: str):
    """Valid token — user attributed, still 200."""
    resp = await client.post(
        "/api/predict",
        json={"image": DUMMY_IMAGE},
        headers={"Authorization": f"Bearer {user_token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["predicted_letter"] == "A"


async def test_predict_with_invalid_token(client: AsyncClient):
    """Garbage token — optional auth, so still 200 (not 401)."""
    resp = await client.post(
        "/api/predict",
        json={"image": DUMMY_IMAGE},
        headers={"Authorization": "Bearer garbage.token.here"},
    )
    assert resp.status_code == 200


async def test_predict_missing_image(client: AsyncClient):
    resp = await client.post("/api/predict", json={})
    assert resp.status_code == 422
