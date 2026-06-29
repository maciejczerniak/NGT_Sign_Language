"""Tests for the image preprocessing pipeline."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
import torch
from PIL import Image

from sign_language.core.preprocessing import (
    CroppedHandDetection,
    HandDetection,
    _crop_single_hand,
    decode_base64_image,
    detect_and_crop_all_hands,
    detect_and_crop_hand,
    preprocess_all_hands,
    preprocess_image,
    IMAGENET_TRANSFORM,
)
from sign_language.core.settings import settings


def _make_base64_image(width: int = 100, height: int = 100, color: str = "red") -> str:
    """Create a base64-encoded JPEG image string for testing."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64


def _make_base64_with_prefix(width: int = 100, height: int = 100) -> str:
    """Create a base64 string with the data URI prefix."""
    raw = _make_base64_image(width, height)
    return f"data:image/jpeg;base64,{raw}"


def _landmarks(x: float = 0.5, y: float = 0.5) -> list[SimpleNamespace]:
    """Create a MediaPipe-like landmark list."""
    return [SimpleNamespace(x=x, y=y, z=0.0) for _ in range(21)]


class TestDecodeBase64Image:
    """Tests for decode_base64_image."""

    def test_decodes_raw_base64(self):
        """Should decode a plain base64 string into an RGB PIL image."""
        b64 = _make_base64_image()
        img = decode_base64_image(b64)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_decodes_with_data_uri_prefix(self):
        """Should strip the data URI prefix and decode correctly."""
        b64 = _make_base64_with_prefix()
        img = decode_base64_image(b64)

        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_output_dimensions(self):
        """Decoded image should match the original dimensions."""
        b64 = _make_base64_image(width=200, height=150)
        img = decode_base64_image(b64)

        assert img.size == (200, 150)

    def test_invalid_base64_raises_value_error(self):
        """Garbage input should raise ValueError."""
        with pytest.raises(ValueError, match="Failed to decode"):
            decode_base64_image("not_valid_base64!!!")


class TestDetectAndCropHand:
    """Tests for detect_and_crop_hand."""

    def test_returns_none_when_detector_is_none(self):
        """With no detector, should return (None, None)."""
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop, landmarks = detect_and_crop_hand(frame, hands_detector=None)

        assert crop is None
        assert landmarks is None

    def test_handles_empty_frame(self):
        """A zero-size frame should not crash."""
        frame = np.zeros((0, 0, 3), dtype=np.uint8)
        crop, landmarks = detect_and_crop_hand(frame, hands_detector=None)

        assert crop is None
        assert landmarks is None

    def test_successful_detection_with_mock(self):
        """Should return a cropped image and landmarks when hand is found."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128

        # Build a mock detector that returns one hand
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        mock_result = MagicMock()
        mock_result.hand_landmarks = [[mock_landmark] * 21]

        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result

        with patch("sign_language.core.preprocessing.mp") as mock_mp:
            mock_mp.Image.return_value = MagicMock()
            mock_mp.ImageFormat.SRGB = 0
            mock_detector.detect.return_value = mock_result

            crop, landmarks = detect_and_crop_hand(frame, mock_detector)

        assert crop is not None
        assert isinstance(crop, Image.Image)
        assert landmarks is not None
        assert len(landmarks) == 21

    def test_no_landmarks_returns_none(self):
        """Should return (None, None) when detector finds no hands."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128

        mock_result = MagicMock()
        mock_result.hand_landmarks = []

        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result

        with patch("sign_language.core.preprocessing.mp") as mock_mp:
            mock_mp.Image.return_value = MagicMock()
            mock_mp.ImageFormat.SRGB = 0

            crop, landmarks = detect_and_crop_hand(frame, mock_detector)

        assert crop is None
        assert landmarks is None

    def test_detector_exception_returns_none(self):
        """Should return (None, None) if detector raises."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("detector crashed")

        crop, landmarks = detect_and_crop_hand(frame, mock_detector)

        assert crop is None
        assert landmarks is None

    def test_empty_crop_returns_none(self):
        """A bounding box outside the frame should return an empty crop result."""
        frame = np.ones((100, 100, 3), dtype=np.uint8)

        crop, landmarks = _crop_single_hand(frame, _landmarks(x=1.5, y=1.5), 0)

        assert crop is None
        assert landmarks == []

    def test_empty_crop_from_detection_returns_none(self):
        """Single-hand detection should ignore an empty crop."""
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.hand_landmarks = [_landmarks(x=1.5, y=1.5)]
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result

        with patch("sign_language.core.preprocessing.mp") as mock_mp:
            mock_mp.Image.return_value = MagicMock()
            mock_mp.ImageFormat.SRGB = 0

            crop, landmarks = detect_and_crop_hand(frame, mock_detector, padding=0)

        assert crop is None
        assert landmarks is None


class TestDetectAndCropAllHands:
    """Tests for multi-hand detection."""

    def test_returns_empty_list_without_detector(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        assert detect_and_crop_all_hands(frame, hands_detector=None) == []

    def test_returns_empty_list_when_no_landmarks(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.hand_landmarks = []
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result

        with patch("sign_language.core.preprocessing.mp") as mock_mp:
            mock_mp.Image.return_value = MagicMock()
            mock_mp.ImageFormat.SRGB = 0

            detections = detect_and_crop_all_hands(frame, mock_detector)

        assert detections == []

    def test_returns_all_valid_hand_detections_with_labels(self):
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
        mock_result = MagicMock()
        mock_result.hand_landmarks = [_landmarks(0.3, 0.4), _landmarks(0.7, 0.6)]
        mock_result.handedness = [
            [SimpleNamespace(display_name="Right")],
            [],
        ]
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result

        with patch("sign_language.core.preprocessing.mp") as mock_mp:
            mock_mp.Image.return_value = MagicMock()
            mock_mp.ImageFormat.SRGB = 0

            detections = detect_and_crop_all_hands(frame, mock_detector, padding=5)

        assert [detection.label for detection in detections] == ["Right", "Hand_1"]
        assert all(isinstance(detection.crop, Image.Image) for detection in detections)
        assert detections[0].wrist_x == 0.3
        assert detections[0].wrist_y == 0.4
        assert len(detections[0].landmarks) == 21

    def test_skips_empty_crops(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        mock_result = MagicMock()
        mock_result.hand_landmarks = [_landmarks(1.5, 1.5), _landmarks(0.5, 0.5)]
        mock_result.handedness = [[], []]
        mock_detector = MagicMock()
        mock_detector.detect.return_value = mock_result

        with patch("sign_language.core.preprocessing.mp") as mock_mp:
            mock_mp.Image.return_value = MagicMock()
            mock_mp.ImageFormat.SRGB = 0

            detections = detect_and_crop_all_hands(frame, mock_detector, padding=5)

        assert len(detections) == 1
        assert detections[0].label == "Hand_1"

    def test_detector_exception_returns_empty_list(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("detector crashed")

        assert detect_and_crop_all_hands(frame, mock_detector) == []


class TestImagenetTransform:
    """Tests for the IMAGENET_TRANSFORM pipeline."""

    def test_output_shape(self):
        """Transform should produce a (3, IMG_SIZE, IMG_SIZE) tensor."""
        img = Image.new("RGB", (300, 400), color="blue")
        tensor = IMAGENET_TRANSFORM(img)

        assert tensor.shape == (3, settings.img_size, settings.img_size)

    def test_output_is_float(self):
        """Transform output should be a float tensor."""
        img = Image.new("RGB", (100, 100), color="green")
        tensor = IMAGENET_TRANSFORM(img)

        assert tensor.dtype == torch.float32

    def test_output_is_normalized(self):
        """Values should not be in raw 0-255 range after normalisation."""
        img = Image.new("RGB", (100, 100), color="white")
        tensor = IMAGENET_TRANSFORM(img)

        # White pixels (255) after ToTensor become 1.0, after normalisation
        # they shift — so values should not all be between 0 and 1
        assert tensor.min() < 0.0 or tensor.max() > 1.0


class TestPreprocessImage:
    """Tests for the full preprocess_image pipeline."""

    def test_without_hand_detector(self):
        """With no detector, should still return a valid tensor."""
        b64 = _make_base64_image()
        device = torch.device("cpu")

        hand_detected, tensor, landmarks = preprocess_image(
            b64, hands_detector=None, device=device
        )

        assert hand_detected is False
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, settings.img_size, settings.img_size)
        assert tensor.device.type == "cpu"
        assert landmarks is None

    def test_with_data_uri_prefix(self):
        """Should handle the data URI prefix gracefully."""
        b64 = _make_base64_with_prefix()
        device = torch.device("cpu")

        hand_detected, tensor, landmarks = preprocess_image(
            b64, hands_detector=None, device=device
        )

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, settings.img_size, settings.img_size)

    def test_different_image_sizes(self):
        """Output tensor shape should be consistent regardless of input size."""
        device = torch.device("cpu")

        for w, h in [(50, 50), (640, 480), (1920, 1080)]:
            b64 = _make_base64_image(width=w, height=h)
            _, tensor, _ = preprocess_image(b64, hands_detector=None, device=device)
            assert tensor.shape == (1, 3, settings.img_size, settings.img_size)

    def test_invalid_image_raises(self):
        """Bad base64 data should raise ValueError."""
        device = torch.device("cpu")
        with pytest.raises(ValueError):
            preprocess_image("garbage_data!!!", hands_detector=None, device=device)


class TestPreprocessAllHands:
    """Tests for the all-hands preprocessing pipeline."""

    def test_populates_tensor_for_each_detection(self):
        b64 = _make_base64_image()
        device = torch.device("cpu")
        detections = [
            CroppedHandDetection(
                label="Right",
                crop=Image.new("RGB", (20, 20)),
                landmarks=[{"x": 0.5, "y": 0.5, "z": 0.0}],
                wrist_x=0.5,
                wrist_y=0.5,
            )
        ]

        with (
            patch(
                "sign_language.core.preprocessing.detect_and_crop_all_hands",
                return_value=detections,
            ),
            patch(
                "sign_language.core.preprocessing.IMAGENET_TRANSFORM",
                return_value=torch.zeros(3, settings.img_size, settings.img_size),
            ),
        ):
            result = preprocess_all_hands(b64, hands_detector=object(), device=device)

        assert result[0].label == detections[0].label
        assert result[0].crop == detections[0].crop
        assert result[0].landmarks == detections[0].landmarks
        assert result[0].tensor.shape == (1, 3, settings.img_size, settings.img_size)

    def test_returns_empty_list_when_no_hands_detected(self):
        b64 = _make_base64_image()

        detections = preprocess_all_hands(
            b64,
            hands_detector=None,
            device=torch.device("cpu"),
        )

        assert detections == []
