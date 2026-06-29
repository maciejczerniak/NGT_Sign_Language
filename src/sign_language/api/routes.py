"""HTTP endpoints for the Sign Language API.

Exposes the following route groups:

- **General** — ``/health``, ``/info``: liveness check and model metadata.
- **Inference** — ``/predict``: single-frame sign language classification.
- **Session** — ``/reset``: clear per-connection smoother and sequence state.
- **Admin** — ``/admin/*``: operational metrics and monitoring history,
  restricted to superuser accounts.

Authentication is handled via FastAPI-Users JWT bearer tokens. Most
endpoints are publicly accessible; admin endpoints require a valid token
belonging to a superuser account (``is_superuser=True``).
"""

import math
import logging
import base64
import binascii
import uuid
from pathlib import Path
from typing import Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Request

from sign_language.core.inference import run_inference
from sign_language.core.preprocessing import preprocess_image
from sign_language.core.settings import settings
from sqlalchemy.ext.asyncio import AsyncSession
from sign_language.db.engine import get_async_session

from sign_language.auth.models import (
    CollectedSample,
    User,
)
from sign_language.auth.users import (
    current_active_user_optional,
)
from .monitoring import track_prediction
from .schemas import (
    CollectRequest,
    CollectResponse,
    InfoResponse,
    PredictRequest,
    PredictResponse,
    ResetResponse,
    TopKItem,
)
from .state import AppState

logger = logging.getLogger(__name__)
router = APIRouter()


def _state(request: Request) -> AppState:
    """Extract the shared :class:`AppState` from the FastAPI application state.

    Args:
        request: The incoming HTTP request whose ``app`` attribute holds
            the FastAPI application instance.

    Returns:
        The shared ``AppState`` instance containing loaded models, device
        configuration, smoother, and sequence builder.
    """
    return cast(AppState, request.app.state.app_state)


@router.get("/health")
def health() -> dict[str, str]:
    """Return a simple liveness check response.

    Used by container orchestrators and load balancers to verify that
    the API process is running and accepting connections.

    Returns:
        ``{"status": "ok"}`` when the server is healthy.
    """
    return {"status": "ok"}


@router.get("/info", response_model=InfoResponse)
def info(request: Request) -> InfoResponse:
    """Return metadata about the loaded models and runtime configuration.

    Provides information useful for client-side feature detection, such
    as whether the landmark model and hand detector are available, the
    number of classes the model was trained on, and the torch device in
    use.

    Args:
        request: The incoming HTTP request, used to access shared state.

    Returns:
        :class:`InfoResponse` containing app name, version, device,
        class count, class names, and availability flags for optional
        model components.
    """
    s = _state(request)
    return InfoResponse(
        app_name=settings.app_name,
        version=settings.version,
        device=str(s.models.device),
        num_classes=len(s.models.class_names),
        class_names=s.models.class_names,
        landmark_model_available=s.models.landmark_model is not None,
        hand_detector_available=s.models.hands_detector is not None,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    request: Request,
    user: Optional[User] = Depends(current_active_user_optional),
) -> PredictResponse:
    """Run sign language inference on a single base64-encoded image frame.

    Authentication is optional. Anonymous callers receive full inference
    functionality. When a valid bearer token is present, the ``user``
    parameter is populated and the request can be attributed to that
    user for future stats and progress tracking.

    The endpoint preprocesses the image, runs the EfficientNet classifier
    (with optional landmark MLP fallback), updates the per-connection
    smoother and sequence builder, and records a monitoring event
    including the Shannon entropy of the top-3 prediction distribution.

    Shannon entropy is computed over the normalised top-3 probabilities
    as ``-sum(p * log2(p))`` and stored alongside confidence as a proxy
    for model uncertainty over time.

    Args:
        payload: Request body containing the base64-encoded JPEG image.
        request: The incoming HTTP request, used to access shared state.
        user: Authenticated user, or ``None`` for anonymous requests.

    Returns:
        :class:`PredictResponse` containing the predicted letter,
        confidence, top-3 alternatives, stable letter, current word,
        sentence, committed letter, and hand detection flag.

    Raises:
        HTTPException: 400 if the image cannot be decoded or no valid
            hand region can be extracted from the frame.
    """
    s = _state(request)
    m = s.models

    if user is not None:
        logger.debug("Predict request from user=%s", user.id)

    try:
        hand_detected, tensor, landmarks_data = preprocess_image(
            payload.image, m.hands_detector, m.device
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    letter, conf, top_3_raw = run_inference(
        tensor=tensor,
        model=m.model,
        class_names=m.class_names,
        device=m.device,
        landmarks_data=landmarks_data,
        landmark_model=m.landmark_model,
        lm_class_names=m.lm_class_names,
    )

    top_3 = [
        TopKItem(letter=item["letter"], confidence=item["confidence"])
        for item in top_3_raw
    ]

    with s.lock:
        if hand_detected:
            s.smoother.update(letter, conf)
            stable_letter = s.smoother.stable_letter
            stable_conf = s.smoother.stable_confidence
        else:
            stable_letter = None
            stable_conf = None
        seq = s.sequence.update(stable_letter, hand_detected)

    _probs = [item["confidence"] for item in top_3_raw]
    _total = sum(_probs)
    _norm = [p / _total for p in _probs if _total > 0]
    _entropy = -sum(p * math.log2(p) for p in _norm if p > 0)

    await track_prediction(letter, conf, "http_predict", entropy=round(_entropy, 6))
    return PredictResponse(
        hand_detected=hand_detected,
        predicted_letter=letter,
        confidence=conf,
        top_3=top_3,
        stable_letter=stable_letter,
        stable_confidence=stable_conf,
        current_word=seq["current_word"],
        sentence=seq["sentence"],
        committed_letter=seq["committed_letter"],
    )


@router.post("/reset", response_model=ResetResponse)
def reset(request: Request) -> ResetResponse:
    """Reset the per-connection smoother and sequence builder state.

    Clears accumulated letter history so that the next prediction starts
    fresh. Useful when a user wants to begin a new signing session without
    reloading the page.

    Args:
        request: The incoming HTTP request, used to access shared state.

    Returns:
        :class:`ResetResponse` confirming the reset was applied.
    """
    _state(request).reset()
    return ResetResponse()


# ── Collect mode endpoint ────────────────────────────────────────────
_ALLOWED_SOURCES = {"camera", "upload", "auto"}


@router.post("/collect", response_model=CollectResponse)
async def collect_sample(
    payload: CollectRequest,
    user: Optional[User] = Depends(current_active_user_optional),
    session: AsyncSession = Depends(get_async_session),
) -> CollectResponse:
    """Store one labelled fingerspelling sample contributed via Collect mode.

    Authentication is optional — anonymous/guest callers may contribute
    (``user_id`` is left NULL); a valid bearer token attributes the sample to
    the user. The image bytes are decoded and written to a file under
    ``settings.collect_storage_dir``; a :class:`~sign_language.auth.models.CollectedSample`
    row records the file path and metadata. A future deployment can swap the
    local file write for Azure Blob without changing this contract.

    :param payload: The sample — base64 image, letter label, source, language.
    :param user: The authenticated user if a token was provided, else ``None``.
    :param session: The active async SQLAlchemy session.
    :returns: A :class:`~sign_language.api.schemas.CollectResponse` with the
        stored sample's id.
    :raises HTTPException: ``400`` if ``source`` is invalid or the image cannot
        be decoded; ``500`` if the file cannot be written.
    """
    if payload.source not in _ALLOWED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"source must be one of {sorted(_ALLOWED_SOURCES)}",
        )

    # Strip a data-URL prefix if present ("data:image/jpeg;base64,....").
    raw_b64 = payload.image
    if raw_b64.startswith("data:"):
        _, _, raw_b64 = raw_b64.partition(",")

    try:
        image_bytes = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image data."
        ) from exc

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image data.")

    sample_id = uuid.uuid4()
    storage_dir = Path(settings.collect_storage_dir)
    try:
        storage_dir.mkdir(parents=True, exist_ok=True)
        image_path = storage_dir / f"{sample_id}.jpg"
        image_path.write_bytes(image_bytes)
    except OSError as exc:
        logger.exception("Failed to write collected sample to disk")
        raise HTTPException(
            status_code=500, detail="Could not store the sample."
        ) from exc

    sample = CollectedSample(
        id=sample_id,
        user_id=user.id if user is not None else None,
        letter=payload.letter,
        image_path=str(image_path),
        source=payload.source,
        language=payload.language,
    )
    session.add(sample)
    await session.commit()

    logger.info(
        "Stored collected sample %s (letter=%s, source=%s, user=%s)",
        sample_id,
        payload.letter,
        payload.source,
        user.id if user else "anonymous",
    )
    return CollectResponse(id=str(sample_id), letter=payload.letter)
