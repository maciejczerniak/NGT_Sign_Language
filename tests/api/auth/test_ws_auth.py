"""WebSocket auth tests — uses starlette TestClient (sync WS support)."""

import json
from unittest.mock import patch

import pytest
from PIL import Image
from starlette.testclient import TestClient

import torch

from sign_language.core.preprocessing import HandDetection

_MOCK_HAND = HandDetection(
    label="Right",
    crop=Image.new("RGB", (224, 224)),
    landmarks=[{"x": 0.5, "y": 0.5, "z": 0.0}] * 21,
    wrist_x=0.5,
    wrist_y=0.5,
    tensor=torch.zeros(1, 3, 224, 224),
)
_MOCK_PREPROCESS = [_MOCK_HAND]
_MOCK_INFERENCE = ("A", 0.95, [{"letter": "A", "confidence": 0.95}])


@pytest.fixture(autouse=True)
def mock_ws_inference():
    with (
        patch(
            "sign_language.api.ws.preprocess_all_hands",
            return_value=_MOCK_PREPROCESS,
        ),
        patch(
            "sign_language.api.ws.run_inference",
            return_value=_MOCK_INFERENCE,
        ),
    ):
        yield


def test_ws_anonymous(sync_client: TestClient):
    """No token — connection accepted, predictions work."""
    with sync_client.websocket_connect("/ws/predict") as ws:
        ws.send_text(json.dumps({"image": "data:image/jpeg;base64,abc"}))
        data = ws.receive_json()
    assert "hands" in data
    assert data["hands"][0]["predicted_letter"] == "A"
    assert data["hands"][0]["confidence"] == 0.95


def test_ws_invalid_token_still_connects(sync_client: TestClient):
    """Garbage token — treated as anonymous, not rejected."""
    with sync_client.websocket_connect("/ws/predict?token=garbage.token") as ws:
        ws.send_text(json.dumps({"image": "data:image/jpeg;base64,abc"}))
        data = ws.receive_json()
    assert "hands" in data


def test_ws_reset_action(sync_client: TestClient):
    with sync_client.websocket_connect("/ws/predict") as ws:
        ws.send_text(json.dumps({"action": "reset"}))
        data = ws.receive_json()
    assert data == {"ok": True}


def test_ws_invalid_json(sync_client: TestClient):
    with sync_client.websocket_connect("/ws/predict") as ws:
        ws.send_text("this is not json")
        data = ws.receive_json()
    assert "error" in data
    assert data["error"] == "Invalid JSON"


def test_ws_missing_image_field(sync_client: TestClient):
    with sync_client.websocket_connect("/ws/predict") as ws:
        ws.send_text(json.dumps({"foo": "bar"}))
        data = ws.receive_json()
    assert data["error"] == "Missing 'image' field"


def test_ws_token_db_error_treated_as_anonymous(sync_client: TestClient):
    """If DB explodes during token decode, connection still accepted."""
    with patch(
        "sign_language.auth.ws_auth.SQLAlchemyUserDatabase",
        side_effect=Exception("DB connection lost"),
    ):
        with sync_client.websocket_connect("/ws/predict?token=anytoken") as ws:
            ws.send_text(json.dumps({"action": "reset"}))
            data = ws.receive_json()
    assert data == {"ok": True}
