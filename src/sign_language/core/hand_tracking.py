"""Shared hand tracking helpers for WebSocket prediction streams."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence, TypeVar

from sign_language.core.preprocessing import CroppedHandDetection
from sign_language.utils.sequence import SequenceBuilder
from sign_language.utils.smoothing import PredictionSmoother

logger = logging.getLogger(__name__)

MATCH_DISTANCE_THRESHOLD = 0.20
SLOT_EXPIRY_FRAMES = 30

DetectionT = TypeVar("DetectionT", bound=CroppedHandDetection)


@dataclass
class HandSlot:
    """Persistent state for one tracked hand across consecutive frames.

    Args:
        hand_id: Unique string identifier for this slot.
        smoother: Per-hand prediction smoother.
        sequence: Per-hand sequence builder.
        last_wrist_x: Last normalized wrist x coordinate.
        last_wrist_y: Last normalized wrist y coordinate.
        frames_since_seen: Consecutive frames without a matching detection.
    """

    hand_id: str
    smoother: PredictionSmoother = field(default_factory=PredictionSmoother)
    sequence: SequenceBuilder = field(default_factory=SequenceBuilder)
    last_wrist_x: float = 0.0
    last_wrist_y: float = 0.0
    frames_since_seen: int = 0


class HandTracker:
    """Match detected hands to persistent slots using wrist proximity."""

    def __init__(self) -> None:
        """Initialize the tracker with no active hand slots."""
        self.slots: dict[str, HandSlot] = {}
        self._next_id = 0

    def _new_id(self) -> str:
        """Generate a unique hand slot identifier.

        Returns:
            Identifier in the form ``hand_<n>``.
        """
        hand_id = f"hand_{self._next_id}"
        self._next_id += 1
        return hand_id

    @staticmethod
    def _distance(det: CroppedHandDetection, slot: HandSlot) -> float:
        """Compute normalized wrist distance between a detection and slot.

        Args:
            det: Incoming hand detection.
            slot: Existing tracked hand slot.

        Returns:
            Euclidean distance in normalized image coordinates.
        """
        dx = det.wrist_x - slot.last_wrist_x
        dy = det.wrist_y - slot.last_wrist_y
        return math.sqrt(dx * dx + dy * dy)

    def match(
        self,
        detections: Sequence[DetectionT],
    ) -> list[tuple[DetectionT, HandSlot]]:
        """Match detections to existing slots and create new slots as needed.

        Args:
            detections: Hand detections from the current frame.

        Returns:
            Matched ``(detection, slot)`` pairs for the current frame.
        """
        for slot in self.slots.values():
            slot.frames_since_seen += 1

        matched_pairs: list[tuple[DetectionT, HandSlot]] = []
        used_slots: set[str] = set()

        candidates: list[tuple[float, DetectionT, HandSlot]] = []
        for det in detections:
            for slot in self.slots.values():
                candidates.append((self._distance(det, slot), det, slot))
        candidates.sort(key=lambda candidate: candidate[0])

        matched_det_ids: set[int] = set()
        for dist, det, slot in candidates:
            if id(det) in matched_det_ids or slot.hand_id in used_slots:
                continue
            if dist <= MATCH_DISTANCE_THRESHOLD:
                slot.last_wrist_x = det.wrist_x
                slot.last_wrist_y = det.wrist_y
                slot.frames_since_seen = 0
                matched_pairs.append((det, slot))
                matched_det_ids.add(id(det))
                used_slots.add(slot.hand_id)

        for det in detections:
            if id(det) not in matched_det_ids:
                slot = HandSlot(
                    hand_id=self._new_id(),
                    last_wrist_x=det.wrist_x,
                    last_wrist_y=det.wrist_y,
                )
                self.slots[slot.hand_id] = slot
                matched_pairs.append((det, slot))

        expired = [
            slot_id
            for slot_id, slot in self.slots.items()
            if slot.frames_since_seen > SLOT_EXPIRY_FRAMES
        ]
        for slot_id in expired:
            logger.debug(
                "Expiring hand slot %s (unseen for %d frames)",
                slot_id,
                self.slots[slot_id].frames_since_seen,
            )
            del self.slots[slot_id]

        return matched_pairs

    def clear(self) -> None:
        """Remove all tracked hand slots."""
        self.slots.clear()


def serialize_landmarks(landmarks_data: Any) -> list[dict] | None:
    """Convert MediaPipe landmarks to JSON-serializable dictionaries.

    Args:
        landmarks_data: MediaPipe landmark list or existing list of dictionaries.

    Returns:
        Landmarks as ``[{x, y, z}, ...]``, or ``None`` when unavailable.
    """
    if landmarks_data is None:
        return None
    if hasattr(landmarks_data, "landmark"):
        return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks_data.landmark]
    if isinstance(landmarks_data, list):
        return landmarks_data
    return None
