"""Image preprocessing pipeline.

Decodes base64-encoded camera frames, detects and crops hands via
MediaPipe, and applies the ImageNet normalisation transform expected
by EfficientNet-B0.
"""

import base64
import binascii
import io
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import mediapipe as mp  # type: ignore[import-untyped]
from PIL import Image, UnidentifiedImageError

from sign_language.core.image_transforms import create_imagenet_transform
from sign_language.core.settings import settings

logger = logging.getLogger(__name__)

IMAGENET_TRANSFORM = create_imagenet_transform(settings.img_size)


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode a base64-encoded image string into an RGB PIL Image.

    :param image_data: Base64 string, optionally with a
        ``data:image/...;base64,`` prefix which is stripped automatically.
    :returns: An RGB :class:`~PIL.Image.Image`.
    :raises ValueError: If the base64 data cannot be decoded or the image
        format is not recognised.
    """
    if "," in image_data:
        image_data = image_data.split(",")[1]

    try:
        img_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc


@dataclass(frozen=True)
class CroppedHandDetection:
    """Detected hand crop before model preprocessing.

    :param label: Handedness label from MediaPipe, either ``"Left"`` or
        ``"Right"``, or ``"Hand_<i>"`` if handedness is unavailable.
    :param crop: Cropped :class:`~PIL.Image.Image` of the hand region.
    :param landmarks: List of 21 landmark dicts each with ``x``, ``y``,
        and ``z`` keys in normalised [0, 1] coordinates.
    :param wrist_x: Normalised x coordinate of landmark 0 (wrist), used
        for cross-frame hand tracking.
    :param wrist_y: Normalised y coordinate of landmark 0 (wrist), used
        for cross-frame hand tracking.
    """

    label: str
    crop: Image.Image
    landmarks: list[dict]
    wrist_x: float
    wrist_y: float


@dataclass(frozen=True)
class HandDetection(CroppedHandDetection):
    """Fully preprocessed hand detection ready for model inference.

    Extends :class:`CroppedHandDetection` with a preprocessed image tensor
    that has been resized, normalised, and moved to the target device.

    :param tensor: Preprocessed image tensor of shape ``(1, 3, H, W)``
        on the target device, ready for EfficientNet-B0 inference.
    """

    tensor: torch.Tensor


def _crop_single_hand(
    frame: np.ndarray,
    lms: Any,
    padding: int,
) -> tuple[Optional[Image.Image], list[dict]]:
    """Crop a single hand from a frame given its MediaPipe landmarks.

    :param frame: RGB numpy array of shape ``(H, W, 3)``.
    :param lms: MediaPipe landmark list for one hand (21 landmarks).
    :param padding: Pixels to add around the bounding box on each side.
    :returns: A tuple of ``(cropped_PIL_image, landmarks_dicts)``. The image
        is ``None`` and the list is empty if the computed crop has zero size.
    """
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]

    x1 = int(max(0, min(xs) - padding))
    x2 = int(min(w, max(xs) + padding))
    y1 = int(max(0, min(ys) - padding))
    y2 = int(min(h, max(ys) + padding))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, []

    landmarks_data = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in lms]
    return Image.fromarray(crop), landmarks_data


def detect_and_crop_hand(
    frame: np.ndarray,
    hands_detector: Any,
    padding: int = 30,
) -> tuple[Optional[Image.Image], Optional[list[dict]]]:
    """Detect a hand in a frame and return a cropped PIL image plus landmarks.

    Legacy single-hand interface — returns only the first detected hand.
    Prefer :func:`detect_and_crop_all_hands` for multi-hand support.

    :param frame: RGB numpy array of shape ``(H, W, 3)``.
    :param hands_detector: A MediaPipe ``HandLandmarker`` instance, or
        ``None`` to skip detection.
    :param padding: Pixels to add around the detected bounding box.
    :returns: A tuple of ``(cropped_image, landmarks)`` for the first
        detected hand, or ``(None, None)`` if no hand is detected or
        ``hands_detector`` is ``None``.
    """
    if hands_detector is None:
        return None, None

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = hands_detector.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        crop_pil, landmarks_data = _crop_single_hand(
            frame, result.hand_landmarks[0], padding
        )
        if crop_pil is None:
            return None, None
        return crop_pil, landmarks_data

    except Exception as exc:
        logger.error("MediaPipe hand detection error: %s", exc)
        return None, None


def detect_and_crop_all_hands(
    frame: np.ndarray,
    hands_detector: Any,
    padding: int = 30,
) -> list[CroppedHandDetection]:
    """Detect all hands in a frame and return a list of cropped results.

    Each result includes the handedness label from MediaPipe (``"Left"``
    or ``"Right"``), the cropped image, normalised landmarks, and the
    wrist position for cross-frame tracking.

    :param frame: RGB numpy array of shape ``(H, W, 3)``.
    :param hands_detector: A MediaPipe ``HandLandmarker`` instance, or
        ``None`` to skip detection.
    :param padding: Pixels to add around each detected bounding box.
    :returns: A list of :class:`CroppedHandDetection` instances, one per
        detected hand. Returns an empty list if no hands are detected,
        ``hands_detector`` is ``None``, or an error occurs.
    """
    if hands_detector is None:
        return []

    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = hands_detector.detect(mp_image)

        if not result.hand_landmarks:
            return []

        detections: list[CroppedHandDetection] = []

        for i, lms in enumerate(result.hand_landmarks):
            if i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].display_name
            else:
                label = f"Hand_{i}"

            crop_pil, landmarks_data = _crop_single_hand(frame, lms, padding)
            if crop_pil is None:
                continue

            wrist = lms[0]

            detections.append(
                CroppedHandDetection(
                    label=label,
                    crop=crop_pil,
                    landmarks=landmarks_data,
                    wrist_x=wrist.x,
                    wrist_y=wrist.y,
                )
            )

        return detections

    except Exception as exc:
        logger.error("MediaPipe hand detection error: %s", exc)
        return []


def preprocess_image(
    image_data: str,
    hands_detector: Any,
    device: torch.device,
) -> tuple[bool, torch.Tensor, Optional[list[dict]]]:
    """Full preprocessing pipeline: decode → detect hand → crop → transform.

    Legacy single-hand interface kept for backward compatibility with the
    HTTP route and CLI. The WebSocket endpoint uses :func:`preprocess_all_hands`.

    Falls back to the full uncropped image if no hand is detected, so
    inference always receives a valid tensor.

    :param image_data: Base64-encoded image from the client.
    :param hands_detector: MediaPipe ``HandLandmarker`` instance, or ``None``.
    :param device: Torch device for the output tensor.
    :returns: A three-tuple of ``(hand_detected, tensor, landmarks)`` where
        ``hand_detected`` indicates whether a hand crop was used, ``tensor``
        is the preprocessed image tensor of shape ``(1, 3, H, W)``, and
        ``landmarks`` is the list of 21 landmark dicts or ``None``.
    """
    pil_img = decode_base64_image(image_data)
    frame = np.array(pil_img)

    crop_pil, landmarks_data = detect_and_crop_hand(frame, hands_detector)

    hand_detected = crop_pil is not None
    source = crop_pil if hand_detected else pil_img

    tensor = IMAGENET_TRANSFORM(source).unsqueeze(0).to(device)
    return hand_detected, tensor, landmarks_data


def preprocess_all_hands(
    image_data: str,
    hands_detector: Any,
    device: torch.device,
) -> list[HandDetection]:
    """Full preprocessing pipeline for all detected hands.

    Decodes the base64 image, runs MediaPipe detection on all hands,
    crops each hand region, and applies the ImageNet transform to produce
    a tensor per hand.

    :param image_data: Base64-encoded image from the client.
    :param hands_detector: MediaPipe ``HandLandmarker`` instance, or ``None``.
    :param device: Torch device for the output tensors.
    :returns: A list of :class:`HandDetection` instances, one per detected
        hand, each containing the label, crop, landmarks, wrist position,
        and preprocessed tensor. Returns an empty list when no hands are
        found.
    """
    pil_img = decode_base64_image(image_data)
    frame = np.array(pil_img)

    cropped_detections = detect_and_crop_all_hands(frame, hands_detector)

    return [
        HandDetection(
            label=det.label,
            crop=det.crop,
            landmarks=det.landmarks,
            wrist_x=det.wrist_x,
            wrist_y=det.wrist_y,
            tensor=IMAGENET_TRANSFORM(det.crop).unsqueeze(0).to(device),
        )
        for det in cropped_detections
    ]
