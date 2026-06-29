"""
Tests for the WebSocket prediction endpoint (/ws/predict).

Uses FastAPI's TestClient WebSocket support — no real network, fully
synchronous test code.

Patch targets must match the names *imported inside ws.py*:
    sign_language.api.ws.preprocess_all_hands
    sign_language.api.ws.run_inference
"""

from unittest.mock import patch, AsyncMock

import pytest
import torch

from sign_language.core.settings import settings
from sign_language.core.preprocessing import HandDetection
from PIL import Image

_PREPROCESS = "sign_language.api.ws.preprocess_all_hands"
_INFERENCE = "sign_language.api.ws.run_inference"
_LOAD_ALL = "sign_language.api.app.load_all"

_WS_URL = "/ws/predict"
_TENSOR = torch.zeros(1, 3, 224, 224)

_TOP3_RAW = [
    {"letter": "A", "confidence": 0.95},
    {"letter": "B", "confidence": 0.03},
    {"letter": "C", "confidence": 0.02},
]

_DUMMY_CROP = Image.new("RGB", (64, 64), color=(120, 80, 200))


def _make_detection(
    label: str = "Right",
    wrist_x: float = 0.5,
    wrist_y: float = 0.5,
    landmarks: list[dict] | None = None,
) -> HandDetection:
    """Build a HandDetection with a dummy crop and tensor."""
    lms = landmarks or [{"x": 0.5, "y": 0.5, "z": 0.0}] * 21
    return HandDetection(
        label=label,
        crop=_DUMMY_CROP,
        landmarks=lms,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        tensor=_TENSOR,
    )


def _ok_detections(n: int = 1) -> list[HandDetection]:
    """Return a list of n detections at distinct positions."""
    dets = []
    for i in range(n):
        dets.append(
            _make_detection(
                label="Right" if i % 2 == 0 else "Left",
                wrist_x=0.3 + i * 0.3,
                wrist_y=0.5,
            )
        )
    return dets


def _ok_inference(letter: str = "A", conf: float = 0.95, top3: list | None = None):
    return (letter, conf, top3 or _TOP3_RAW)


def _first_hand(data: dict) -> dict:
    """Extract the first hand from a multi-hand response."""
    assert "hands" in data, f"Expected 'hands' key, got: {data.keys()}"
    assert len(data["hands"]) > 0, "Expected at least one hand"
    return data["hands"][0]


# ---- Hermetic monitoring ----------------------------------------------------
@pytest.fixture(autouse=True)
def _stub_monitoring():
    """Keep WebSocket tests hermetic — never write a real monitoring event.

    The frame handler awaits track_prediction once per detected hand,
    which opens its own database session. Without this stub the tests
    would attempt (and, if the dev DB is reachable, succeed at) writing
    rows to monitoring_events on every frame. The monitoring logic itself
    is tested in test_monitoring.py.
    """
    with patch("sign_language.api.ws.track_prediction", new=AsyncMock()):
        yield


# ============================================================================
# Connection lifecycle
# ============================================================================


class TestConnection:
    def test_connection_is_accepted(self, client):
        """Verify connection is accepted."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": "dGVzdA=="})
                ws.receive_json()  # must not raise

    def test_connection_closes_cleanly(self, client):
        """Exiting the context manager must not raise."""
        with client.websocket_connect(_WS_URL):
            pass  # just open and close


# ============================================================================
# Happy-path frame processing
# ============================================================================


class TestFrameProcessing:
    def test_response_contains_hands_key(self, client, dummy_image_b64):
        """Response must have a 'hands' list."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        assert "hands" in data
        assert isinstance(data["hands"], list)

    def test_hand_entry_contains_all_fields(self, client, dummy_image_b64):
        """Each hand entry must have the expected fields."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        hand = _first_hand(data)
        expected_keys = {
            "hand_id",
            "label",
            "predicted_letter",
            "confidence",
            "top_3",
            "stable_letter",
            "stable_confidence",
            "current_word",
            "sentence",
            "committed_letter",
            "landmarks",
        }
        assert expected_keys.issubset(hand.keys())

    def test_predicted_letter_and_confidence(self, client, dummy_image_b64):
        """Verify predicted letter and confidence from inference."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference("B", 0.88)),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        hand = _first_hand(data)
        assert hand["predicted_letter"] == "B"
        assert pytest.approx(hand["confidence"], abs=1e-4) == 0.88

    def test_one_hand_detected(self, client, dummy_image_b64):
        """One detection produces one hand entry."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        assert len(data["hands"]) == 1

    def test_no_hands_detected(self, client, dummy_image_b64):
        """No detections produces an empty hands list."""
        with (
            patch(_PREPROCESS, return_value=[]),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        assert data["hands"] == []

    def test_two_hands_detected(self, client, dummy_image_b64):
        """Two detections produce two hand entries with distinct IDs."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(2)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        assert len(data["hands"]) == 2
        ids = [h["hand_id"] for h in data["hands"]]
        assert ids[0] != ids[1]

    def test_top_3_shape(self, client, dummy_image_b64):
        """Each hand entry has a top_3 list with letter and confidence."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        hand = _first_hand(data)
        assert len(hand["top_3"]) == 3
        for item in hand["top_3"]:
            assert "letter" in item and "confidence" in item

    def test_multiple_frames_in_one_connection(self, client, dummy_image_b64):
        """The connection must handle multiple sequential frames."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                for _ in range(5):
                    ws.send_json({"image": dummy_image_b64})
                    data = ws.receive_json()
                    assert len(data["hands"]) == 1

    def test_data_url_prefix_accepted(self, client, dummy_image_b64_dataurl):
        """Data URL prefix should be accepted without error."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64_dataurl})
                data = ws.receive_json()

        assert "error" not in data

    def test_hand_label_is_passed_through(self, client, dummy_image_b64):
        """The MediaPipe handedness label should appear in the response."""
        det = _make_detection(label="Left")
        with (
            patch(_PREPROCESS, return_value=[det]),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        assert _first_hand(data)["label"] == "Left"


# ============================================================================
# Error frames — connection must survive every bad message
# ============================================================================


class TestErrorFrames:
    def test_invalid_json_returns_error(self, client):
        """Verify invalid JSON returns error."""
        with client.websocket_connect(_WS_URL) as ws:
            ws.send_text("this is not json")
            data = ws.receive_json()
        assert data == {"error": "Invalid JSON"}

    def test_missing_image_field_returns_error(self, client):
        """Verify missing image field returns error."""
        with client.websocket_connect(_WS_URL) as ws:
            ws.send_json({"not_image": "x"})
            data = ws.receive_json()
        assert "error" in data
        assert "image" in data["error"].lower()

    def test_bad_base64_returns_error(self, client):
        """Verify bad base64 returns error."""
        with patch(
            _PREPROCESS, side_effect=ValueError("Failed to decode base64 image")
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": "!!!notbase64!!!"})
                data = ws.receive_json()
        assert "error" in data
        assert "decode" in data["error"].lower()

    def test_connection_survives_bad_frame(self, client, dummy_image_b64):
        """After an error frame the connection must still process valid frames."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                # bad frame first
                ws.send_text("garbage")
                error = ws.receive_json()
                assert "error" in error

                # valid frame second — must still work
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()
                assert "hands" in data

    def test_error_detail_is_propagated(self, client, dummy_image_b64):
        """Verify error detail is propagated."""
        detail = "Specific decode failure xyz"
        with patch(_PREPROCESS, side_effect=ValueError(detail)):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()
        assert data["error"] == detail


# ============================================================================
# Reset action
# ============================================================================


class TestResetAction:
    def test_reset_returns_ok(self, client):
        """Verify reset returns ok."""
        with client.websocket_connect(_WS_URL) as ws:
            ws.send_json({"action": "reset"})
            data = ws.receive_json()
        assert data == {"ok": True}

    def test_reset_clears_tracker(self, client, dummy_image_b64):
        """Fill the smoother to produce a stable letter, reset, then confirm
        the very next frame has no stable letter."""
        n = settings.smoother_window_size

        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference("A", 0.99)),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                # fill window
                for _ in range(n):
                    ws.send_json({"image": dummy_image_b64})
                    ws.receive_json()

                # confirm stable
                ws.send_json({"image": dummy_image_b64})
                before = ws.receive_json()
                assert _first_hand(before)["stable_letter"] == "A"

                # reset
                ws.send_json({"action": "reset"})
                ws.receive_json()

                # one frame after reset — window not full
                ws.send_json({"image": dummy_image_b64})
                after = ws.receive_json()

        assert _first_hand(after)["stable_letter"] is None

    def test_reset_can_be_sent_multiple_times(self, client):
        """Verify reset can be sent multiple times."""
        with client.websocket_connect(_WS_URL) as ws:
            for _ in range(3):
                ws.send_json({"action": "reset"})
                data = ws.receive_json()
                assert data == {"ok": True}


# ============================================================================
# Smoother integration (per-hand state)
# ============================================================================


class TestSmootherIntegration:
    def test_stable_letter_none_after_single_frame(self, client, dummy_image_b64):
        """Stable letter should be None after a single frame."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference("A", 0.99)),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        assert _first_hand(data)["stable_letter"] is None

    def test_stable_letter_appears_after_full_window(self, client, dummy_image_b64):
        """Stable letter should appear after smoother window is filled."""
        n = settings.smoother_window_size
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference("C", 0.99)),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                for _ in range(n):
                    ws.send_json({"image": dummy_image_b64})
                    data = ws.receive_json()

        hand = _first_hand(data)
        assert hand["stable_letter"] == "C"
        assert hand["stable_confidence"] > 0

    def test_low_confidence_prevents_stable_letter(self, client, dummy_image_b64):
        """Low confidence should prevent a stable letter."""
        n = settings.smoother_window_size
        low_conf = settings.smoother_min_confidence - 0.1
        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference("A", low_conf)),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                for _ in range(n):
                    ws.send_json({"image": dummy_image_b64})
                    data = ws.receive_json()

        assert _first_hand(data)["stable_letter"] is None


# ============================================================================
# Per-connection state isolation
# ============================================================================


class TestConnectionIsolation:
    def test_two_connections_have_independent_trackers(self, client, dummy_image_b64):
        """Filling the smoother on connection A must not affect connection B."""
        n = settings.smoother_window_size

        with (
            patch(_PREPROCESS, return_value=_ok_detections(1)),
            patch(_INFERENCE, return_value=_ok_inference("A", 0.99)),
        ):
            # Fill smoother on connection A
            with client.websocket_connect(_WS_URL) as ws_a:
                for _ in range(n):
                    ws_a.send_json({"image": dummy_image_b64})
                    data_a = ws_a.receive_json()
            assert _first_hand(data_a)["stable_letter"] == "A"

            # Connection B starts fresh — single frame, no stable letter
            with client.websocket_connect(_WS_URL) as ws_b:
                ws_b.send_json({"image": dummy_image_b64})
                data_b = ws_b.receive_json()

        assert _first_hand(data_b)["stable_letter"] is None


# ============================================================================
# Hand tracking across frames
# ============================================================================


class TestHandTracking:
    def test_hand_id_is_stable_across_frames(self, client, dummy_image_b64):
        """The same hand at the same position should keep its hand_id."""
        det = _make_detection(wrist_x=0.5, wrist_y=0.5)
        with (
            patch(_PREPROCESS, return_value=[det]),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data1 = ws.receive_json()

                ws.send_json({"image": dummy_image_b64})
                data2 = ws.receive_json()

        id1 = _first_hand(data1)["hand_id"]
        id2 = _first_hand(data2)["hand_id"]
        assert id1 == id2

    def test_two_hands_get_different_ids(self, client, dummy_image_b64):
        """Two hands at different positions must get different IDs."""
        with (
            patch(_PREPROCESS, return_value=_ok_detections(2)),
            patch(_INFERENCE, return_value=_ok_inference()),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                ws.send_json({"image": dummy_image_b64})
                data = ws.receive_json()

        ids = [h["hand_id"] for h in data["hands"]]
        assert len(set(ids)) == 2

    def test_each_hand_has_independent_smoother(self, client, dummy_image_b64):
        """Two hands at distinct positions should each build their own
        stable prediction independently."""
        n = settings.smoother_window_size

        det_right = _make_detection(label="Right", wrist_x=0.3, wrist_y=0.5)
        det_left = _make_detection(label="Left", wrist_x=0.7, wrist_y=0.5)

        with (
            patch(_PREPROCESS, return_value=[det_right, det_left]),
            patch(_INFERENCE, return_value=_ok_inference("A", 0.99)),
        ):
            with client.websocket_connect(_WS_URL) as ws:
                for _ in range(n):
                    ws.send_json({"image": dummy_image_b64})
                    data = ws.receive_json()

        assert len(data["hands"]) == 2
        for hand in data["hands"]:
            assert hand["stable_letter"] == "A"


# ============================================================================
# HandTracker unit tests (direct, no WebSocket)
# ============================================================================


class TestHandTrackerDirect:
    """Test HandTracker.match() directly without going through the endpoint."""

    def test_same_position_keeps_same_slot(self):
        """A hand at the same position across two frames keeps its slot."""
        from sign_language.api.ws import HandTracker

        tracker = HandTracker()
        det = _make_detection(wrist_x=0.5, wrist_y=0.5)

        pairs1 = tracker.match([det])
        hand_id_1 = pairs1[0][1].hand_id

        pairs2 = tracker.match([det])
        hand_id_2 = pairs2[0][1].hand_id

        assert hand_id_1 == hand_id_2

    def test_small_movement_keeps_same_slot(self):
        """A hand that moves slightly between frames keeps its slot."""
        from sign_language.api.ws import HandTracker

        tracker = HandTracker()

        det1 = _make_detection(wrist_x=0.50, wrist_y=0.50)
        det2 = _make_detection(wrist_x=0.52, wrist_y=0.51)

        pairs1 = tracker.match([det1])
        hand_id_1 = pairs1[0][1].hand_id

        pairs2 = tracker.match([det2])
        hand_id_2 = pairs2[0][1].hand_id

        assert hand_id_1 == hand_id_2

    def test_far_position_creates_new_slot(self):
        """A hand far from all existing slots gets a new slot."""
        from sign_language.api.ws import HandTracker, MATCH_DISTANCE_THRESHOLD

        tracker = HandTracker()

        det1 = _make_detection(wrist_x=0.1, wrist_y=0.1)
        pairs1 = tracker.match([det1])
        hand_id_1 = pairs1[0][1].hand_id

        # Place second detection well beyond the threshold
        det2 = _make_detection(wrist_x=0.9, wrist_y=0.9)
        pairs2 = tracker.match([det2])
        hand_id_2 = pairs2[0][1].hand_id

        assert hand_id_1 != hand_id_2

    def test_two_hands_matched_correctly(self):
        """Two hands at distinct positions should match to their own slots."""
        from sign_language.api.ws import HandTracker

        tracker = HandTracker()

        det_left = _make_detection(wrist_x=0.2, wrist_y=0.5)
        det_right = _make_detection(wrist_x=0.8, wrist_y=0.5)

        pairs1 = tracker.match([det_left, det_right])
        id_left_1 = pairs1[0][1].hand_id
        id_right_1 = pairs1[1][1].hand_id

        # Second frame — same positions
        pairs2 = tracker.match([det_left, det_right])
        id_left_2 = pairs2[0][1].hand_id
        id_right_2 = pairs2[1][1].hand_id

        assert id_left_1 == id_left_2
        assert id_right_1 == id_right_2
        assert id_left_1 != id_right_1

    def test_slot_expires_after_threshold(self):
        """A slot that goes unmatched for SLOT_EXPIRY_FRAMES is removed."""
        from sign_language.api.ws import HandTracker, SLOT_EXPIRY_FRAMES

        tracker = HandTracker()

        det = _make_detection(wrist_x=0.5, wrist_y=0.5)
        pairs = tracker.match([det])
        hand_id = pairs[0][1].hand_id

        assert hand_id in tracker.slots

        # Send empty detections for exactly SLOT_EXPIRY_FRAMES frames
        for _ in range(SLOT_EXPIRY_FRAMES):
            tracker.match([])

        # Slot should still exist (expiry is > threshold, not >=)
        # One more frame to push it past
        tracker.match([])

        assert hand_id not in tracker.slots

    def test_slot_survives_below_expiry(self):
        """A slot that reappears before expiry keeps its state."""
        from sign_language.api.ws import HandTracker, SLOT_EXPIRY_FRAMES

        tracker = HandTracker()

        det = _make_detection(wrist_x=0.5, wrist_y=0.5)
        pairs1 = tracker.match([det])
        hand_id = pairs1[0][1].hand_id

        # Miss for fewer frames than the expiry threshold
        for _ in range(SLOT_EXPIRY_FRAMES - 5):
            tracker.match([])

        # Hand reappears
        pairs2 = tracker.match([det])
        hand_id_after = pairs2[0][1].hand_id

        assert hand_id == hand_id_after

    def test_clear_removes_all_slots(self):
        """HandTracker.clear() should remove all tracked slots."""
        from sign_language.api.ws import HandTracker

        tracker = HandTracker()

        tracker.match([_make_detection(wrist_x=0.3, wrist_y=0.5)])
        tracker.match([_make_detection(wrist_x=0.7, wrist_y=0.5)])

        assert len(tracker.slots) > 0

        tracker.clear()

        assert len(tracker.slots) == 0

    def test_each_slot_has_own_smoother_and_sequence(self):
        """Each slot should have independent smoother and sequence builder."""
        from sign_language.api.ws import HandTracker

        tracker = HandTracker()

        det1 = _make_detection(wrist_x=0.2, wrist_y=0.5)
        det2 = _make_detection(wrist_x=0.8, wrist_y=0.5)

        pairs = tracker.match([det1, det2])

        slot_a = pairs[0][1]
        slot_b = pairs[1][1]

        assert slot_a.smoother is not slot_b.smoother
        assert slot_a.sequence is not slot_b.sequence
