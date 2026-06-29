"""Integration tests for HTTP routes."""

import pytest
import torch
from unittest.mock import patch, MagicMock, AsyncMock

from sign_language.core.settings import settings
from sign_language.utils.smoothing import PredictionSmoother

# ---- patch targets (must match the names *imported into* routes.py) --------
_PREPROCESS = "sign_language.api.routes.preprocess_image"
_INFERENCE = "sign_language.api.routes.run_inference"

# ---- URL constants ----------------------------------------------------------
_HEALTH = "/api/health"
_INFO = "/api/info"
_PREDICT = "/api/predict"
_RESET = "/api/reset"

# ---- Reusable defaults ------------------------------------------------------
_TENSOR = torch.zeros(1, 3, 224, 224)

_TOP3_RAW = [
    {"letter": "A", "confidence": 0.95},
    {"letter": "B", "confidence": 0.03},
    {"letter": "C", "confidence": 0.02},
]


def _preprocess_ok(hand_detected=True, landmarks=None):
    return (hand_detected, _TENSOR, landmarks)


def _inference_ok(letter="A", conf=0.95, top3=None):
    return (letter, conf, top3 or _TOP3_RAW)


# ---- Hermetic monitoring ----------------------------------------------------
@pytest.fixture(autouse=True)
def _stub_monitoring():
    """Keep route tests hermetic — never write a real monitoring event.

    The /predict endpoint awaits track_prediction, which opens its own
    database session. Without this stub the tests would attempt (and, if
    the dev DB is reachable, succeed at) writing rows to monitoring_events
    on every request. The monitoring logic itself is tested in
    test_monitoring.py.
    """
    with patch("sign_language.api.routes.track_prediction", new=AsyncMock()):
        yield


# ============================================================================
# /api/health
# ============================================================================


class TestHealth:
    def test_returns_200(self, client):
        """Verify health returns 200."""
        r = client.get(_HEALTH)
        assert r.status_code == 200

    def test_returns_status_ok(self, client):
        """Verify health returns status ok."""
        assert client.get(_HEALTH).json() == {"status": "ok"}


# ============================================================================
# /api/info
# ============================================================================


class TestInfo:
    def test_returns_200(self, client):
        """Verify info returns 200."""
        assert client.get(_INFO).status_code == 200

    def test_class_names_match_mock(self, client):
        """Verify info class names match mock."""
        data = client.get(_INFO).json()
        assert data["class_names"] == ["A", "B", "C"]
        assert data["num_classes"] == 3

    def test_no_landmark_model(self, client, mock_models):
        """Verify info no landmark model."""
        mock_models.landmark_model = None
        mock_models.hands_detector = None
        data = client.get(_INFO).json()
        assert data["landmark_model_available"] is False
        assert data["hand_detector_available"] is False

    def test_with_landmark_model_and_detector(self, client, mock_models):
        """Verify info with landmark model and detector."""
        mock_models.landmark_model = MagicMock()
        mock_models.hands_detector = MagicMock()
        data = client.get(_INFO).json()
        assert data["landmark_model_available"] is True
        assert data["hand_detector_available"] is True

    def test_device_is_string(self, client):
        """Verify info device is string."""
        data = client.get(_INFO).json()
        assert isinstance(data["device"], str)

    def test_version_and_app_name_from_settings(self, client):
        """Verify info version and app name from settings."""
        data = client.get(_INFO).json()
        assert data["app_name"]
        assert data["version"]


# ============================================================================
# /api/predict
# ============================================================================


class TestPredict:
    # ---- Happy paths --------------------------------------------------------

    def test_returns_200_on_valid_frame(self, client, dummy_image_b64):
        """Verify predict returns 200 on valid frame."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok()),
        ):
            r = client.post(_PREDICT, json={"image": dummy_image_b64})
        assert r.status_code == 200

    def test_response_shape(self, client, dummy_image_b64):
        """Verify predict response shape."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok()),
        ):
            data = client.post(_PREDICT, json={"image": dummy_image_b64}).json()

        assert "hand_detected" in data
        assert "predicted_letter" in data
        assert "confidence" in data
        assert "top_3" in data
        assert "stable_letter" in data
        assert "stable_confidence" in data
        assert "current_word" in data
        assert "sentence" in data
        assert "committed_letter" in data

    def test_predicted_letter_and_confidence(self, client, dummy_image_b64):
        """Verify predict predicted letter and confidence."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("B", 0.88)),
        ):
            data = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        assert data["predicted_letter"] == "B"
        assert data["confidence"] == pytest.approx(0.88)

    def test_hand_detected_true(self, client, dummy_image_b64):
        """Verify predict hand detected true."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok(hand_detected=True)),
            patch(_INFERENCE, return_value=_inference_ok()),
        ):
            data = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        assert data["hand_detected"] is True

    def test_hand_not_detected(self, client, dummy_image_b64):
        """Verify predict hand not detected."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok(hand_detected=False)),
            patch(_INFERENCE, return_value=_inference_ok()),
        ):
            data = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        assert data["hand_detected"] is False

    def test_top_3_items_have_letter_and_confidence(self, client, dummy_image_b64):
        """Verify predict top 3 items have letter and confidence."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok()),
        ):
            data = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        assert len(data["top_3"]) == 3
        for item in data["top_3"]:
            assert "letter" in item
            assert "confidence" in item

    def test_accepts_data_url_prefix(self, client, dummy_image_b64_dataurl):
        """Verify predict accepts data url prefix."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok()),
        ):
            r = client.post(_PREDICT, json={"image": dummy_image_b64_dataurl})
        assert r.status_code == 200

    # ---- Smoother integration -----------------------------------------------

    def test_stable_letter_is_none_initially(self, client, dummy_image_b64):
        """Verify predict stable letter is none initially."""
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("A", 0.99)),
        ):
            data = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        # Only 1 frame — window not full yet
        assert data["stable_letter"] is None

    def test_stable_letter_appears_after_full_window(self, client, dummy_image_b64):
        """Verify predict stable letter appears after full window."""
        n = settings.smoother_window_size
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("C", 0.99)),
        ):
            for _ in range(n):
                resp = client.post(_PREDICT, json={"image": dummy_image_b64})
        data = resp.json()
        assert data["stable_letter"] == "C"
        assert data["stable_confidence"] > 0

    def test_low_confidence_prevents_stable_letter(self, client, dummy_image_b64):
        """Verify predict low confidence prevents stable letter."""
        n = settings.smoother_window_size
        low_conf = settings.smoother_min_confidence - 0.1  # below threshold
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("A", low_conf)),
        ):
            for _ in range(n):
                resp = client.post(_PREDICT, json={"image": dummy_image_b64})
        assert resp.json()["stable_letter"] is None

    # ---- Error paths --------------------------------------------------------

    def test_invalid_base64_returns_400(self, client):
        """Verify predict invalid base64 returns 400."""
        with patch(
            _PREPROCESS, side_effect=ValueError("Failed to decode base64 image")
        ):
            r = client.post(_PREDICT, json={"image": "!!!notbase64!!!"})
        assert r.status_code == 400
        assert "Failed to decode" in r.json()["detail"]

    def test_missing_image_field_returns_422(self, client):
        """Verify predict missing image field returns 422."""
        r = client.post(_PREDICT, json={})
        assert r.status_code == 422

    def test_non_json_body_returns_422(self, client):
        """Verify predict non json body returns 422."""
        r = client.post(
            _PREDICT, content=b"not-json", headers={"Content-Type": "application/json"}
        )
        assert r.status_code == 422

    def test_preprocess_value_error_detail_propagated(self, client, dummy_image_b64):
        """Verify predict preprocess value error detail propagated."""
        msg = "Specific decode failure"
        with patch(_PREPROCESS, side_effect=ValueError(msg)):
            r = client.post(_PREDICT, json={"image": dummy_image_b64})
        assert r.json()["detail"] == msg


# ============================================================================
# /api/reset
# ============================================================================


class TestReset:
    def test_returns_200(self, client):
        """Verify reset returns 200."""
        assert client.post(_RESET).status_code == 200

    def test_returns_ok_true(self, client):
        """Verify reset returns ok true."""
        assert client.post(_RESET).json()["ok"] is True

    def test_reset_clears_smoother(self, client, dummy_image_b64):
        """After filling the smoother and resetting, the next single frame
        must not produce a stable letter."""
        n = settings.smoother_window_size
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("A", 0.99)),
        ):
            for _ in range(n):
                client.post(_PREDICT, json={"image": dummy_image_b64})

        # Confirm stable letter was set
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("A", 0.99)),
        ):
            before = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        assert before["stable_letter"] == "A"

        client.post(_RESET)

        # One frame after reset — window not full, stable_letter must be None
        with (
            patch(_PREPROCESS, return_value=_preprocess_ok()),
            patch(_INFERENCE, return_value=_inference_ok("A", 0.99)),
        ):
            after = client.post(_PREDICT, json={"image": dummy_image_b64}).json()
        assert after["stable_letter"] is None

    def test_reset_can_be_called_multiple_times(self, client):
        """Verify reset reset can be called multiple times."""
        for _ in range(3):
            assert client.post(_RESET).status_code == 200
