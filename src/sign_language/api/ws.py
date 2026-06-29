"""WebSocket endpoint for real-time sign language prediction."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Any, Optional, cast

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from sign_language.api.monitoring import track_prediction
from sign_language.api.state import AppState
from sign_language.auth.ws_auth import get_user_from_ws_token
from sign_language.core.hand_tracking import (
    MATCH_DISTANCE_THRESHOLD,
    SLOT_EXPIRY_FRAMES,
    HandTracker,
    serialize_landmarks,
)
from sign_language.core.inference import run_inference
from sign_language.core.preprocessing import preprocess_all_hands

logger = logging.getLogger(__name__)
ws_router = APIRouter()

__all__ = [
    "MATCH_DISTANCE_THRESHOLD",
    "SLOT_EXPIRY_FRAMES",
    "HandTracker",
    "ws_router",
]


def _app_state(websocket: WebSocket) -> AppState:
    """Extract the application state from a WebSocket connection.

    Args:
        websocket: Active WebSocket connection.

    Returns:
        Application state containing loaded models and runtime helpers.
    """
    return cast(AppState, websocket.app.state.app_state)


def _process_frame_multi(
    image_data: str,
    app_state: AppState,
    tracker: HandTracker,
) -> dict[str, Any]:
    """Run the on-prem multi-hand prediction pipeline.

    Args:
        image_data: Base64-encoded camera frame.
        app_state: Shared application state with loaded models.
        tracker: Per-connection hand tracker.

    Returns:
        Frontend-compatible ``{"hands": [...]}`` response.
    """
    models = app_state.models
    detections = preprocess_all_hands(
        image_data,
        models.hands_detector,
        models.device,
    )
    pairs = tracker.match(detections)

    hands_output: list[dict[str, Any]] = []
    for detection, slot in pairs:
        letter, confidence, top_3_raw = run_inference(
            tensor=detection.tensor,
            model=models.model,
            class_names=models.class_names,
            device=models.device,
            landmarks_data=detection.landmarks,
            landmark_model=models.landmark_model,
            lm_class_names=models.lm_class_names,
        )

        slot.smoother.update(letter, confidence)
        sequence = slot.sequence.update(slot.smoother.stable_letter, True)

        hands_output.append(
            {
                "hand_id": slot.hand_id,
                "label": detection.label,
                "predicted_letter": letter,
                "confidence": confidence,
                "top_3": top_3_raw,
                "stable_letter": slot.smoother.stable_letter,
                "stable_confidence": slot.smoother.stable_confidence,
                "current_word": sequence["current_word"],
                "sentence": sequence["sentence"],
                "committed_letter": sequence["committed_letter"],
                "landmarks": serialize_landmarks(detection.landmarks),
            }
        )

    seen_ids = {slot.hand_id for _, slot in pairs}
    for slot_id, slot in tracker.slots.items():
        if slot_id not in seen_ids:
            slot.sequence.update(None, False)

    return {"hands": hands_output}


@ws_router.websocket("/ws/predict")
async def ws_predict(
    websocket: WebSocket,
    token: Optional[str] = Query(
        None,
        description="Optional JWT for user attribution.",
    ),
) -> None:
    """Stream sign language predictions over a WebSocket connection.

    Args:
        websocket: Incoming browser WebSocket connection.
        token: Optional JWT bearer token for user attribution.
    """
    app_state = _app_state(websocket)
    tracker = HandTracker()
    user = await get_user_from_ws_token(token)

    await websocket.accept()
    logger.info(
        "WebSocket connection accepted from %s (user=%s)",
        websocket.client,
        user.id if user else "anonymous",
    )

    loop = asyncio.get_running_loop()
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            if message.get("action") == "reset":
                tracker.clear()
                await websocket.send_json({"ok": True})
                continue

            image_data = message.get("image")
            if not image_data:
                await websocket.send_json({"error": "Missing 'image' field"})
                continue

            try:
                result = await loop.run_in_executor(
                    None,
                    _process_frame_multi,
                    str(image_data),
                    app_state,
                    tracker,
                )
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            for hand in result.get("hands", []):
                probabilities = [item["confidence"] for item in hand.get("top_3", [])]
                total = sum(probabilities)
                normalized = [
                    probability / total for probability in probabilities if total > 0
                ]
                entropy = -sum(
                    probability * math.log2(probability)
                    for probability in normalized
                    if probability > 0
                )

                await track_prediction(
                    hand.get("predicted_letter"),
                    hand.get("confidence", 0.0),
                    "ws_predict",
                    entropy=round(entropy, 6),
                )

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(
            "WebSocket connection closed by client (user=%s)",
            user.id if user else "anonymous",
        )
