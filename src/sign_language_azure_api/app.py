"""Standalone FastAPI app for Azure ML endpoint inference."""

from __future__ import annotations

import base64
import asyncio
import io
import json
import logging
import math
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from sign_language.api.shared_routes import shared_router
from sign_language.api.monitoring import MonitoringMiddleware, track_prediction
from sign_language.auth.schemas import UserCreate, UserRead, UserUpdate
from sign_language.auth.users import auth_backend, fastapi_users
from sign_language.db.engine import engine
from sign_language.core.hand_tracking import HandTracker, serialize_landmarks
from sign_language.core.preprocessing import (
    decode_base64_image,
    detect_and_crop_all_hands,
)
from sign_language.core.settings import settings as core_settings
from sign_language.models.loader import init_hand_detector
from sign_language_azure_api.client import (
    AzureEndpointError,
    AzureMLEndpointClient,
)
from sign_language_azure_api.collection import (
    CollectionStorageError,
    CollectionValidationError,
    decode_collection_image,
    store_pending_sample,
    validate_collection_metadata,
)
from sign_language_azure_api.schemas import (
    AzureApiInfoResponse,
    CollectRequest,
    CollectResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TopKItem,
)
from sign_language_azure_api.settings import settings

logger = logging.getLogger(__name__)
_HAND_DETECTOR: object | None = None
_HAND_DETECTOR_INITIALIZED = False


def _client() -> AzureMLEndpointClient:
    """Create an Azure ML endpoint client from application settings.

    Returns:
        Configured synchronous Azure ML endpoint client.

    Raises:
        ValueError: If the endpoint URL or key is not configured.
    """
    return AzureMLEndpointClient(
        endpoint_url=settings.azure_api_online_endpoint_url,
        endpoint_key=settings.azure_api_online_endpoint_key,
        timeout_seconds=settings.azure_api_online_timeout_seconds,
    )


def _deployment_name(requested_deployment: str | None = None) -> str | None:
    """Resolve the Azure ML deployment name for a prediction request.

    Args:
        requested_deployment: Optional deployment name supplied by the caller.

    Returns:
        The requested deployment, configured default deployment, or ``None``.
    """
    return requested_deployment or settings.azure_api_default_deployment or None


def _parse_model_reference(model: object) -> tuple[str | None, str | None]:
    """Extract model name/version from an Azure ML deployment model reference."""
    name = getattr(model, "name", None)
    version = getattr(model, "version", None)

    if name or version:
        return (
            str(name) if name is not None else None,
            str(version) if version is not None else None,
        )

    if isinstance(model, str):
        # Common forms:
        # azureml:ngt-sign-language:3
        # azureml://.../models/ngt-sign-language/versions/3
        if model.startswith("azureml:"):
            parts = model.split(":")
            if len(parts) >= 3:
                return parts[-2], parts[-1]

        if "/models/" in model and "/versions/" in model:
            try:
                model_name = model.split("/models/", 1)[1].split("/", 1)[0]
                model_version = model.split("/versions/", 1)[1].split("/", 1)[0]
                return model_name, model_version
            except IndexError:
                return None, None

    return None, None


def _select_deployment_name(
    traffic: dict[str, int],
    deployment_names: list[str],
) -> str | None:
    """Choose the deployment to report in /info."""
    configured = settings.azure_api_default_deployment
    if configured:
        return configured

    if traffic:
        return max(traffic.items(), key=lambda item: item[1])[0]

    if len(deployment_names) == 1:
        return deployment_names[0]

    return None


def _fetch_azure_endpoint_metadata() -> dict[str, Any]:
    """Fetch Azure ML online endpoint/deployment metadata for /info.

    This is best-effort. If identity/permissions/network are not available,
    the API still works and /info reports the metadata error.
    """
    required = [
        settings.azure_api_ml_subscription_id,
        settings.azure_api_ml_resource_group,
        settings.azure_api_ml_workspace,
        settings.azure_api_online_endpoint_name,
    ]
    if not all(required):
        return {
            "metadata_available": False,
            "metadata_error": "Azure ML workspace or endpoint metadata settings are incomplete.",
        }

    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        ml_client = MLClient(
            credential=credential,
            subscription_id=settings.azure_api_ml_subscription_id,
            resource_group_name=settings.azure_api_ml_resource_group,
            workspace_name=settings.azure_api_ml_workspace,
        )

        endpoint = ml_client.online_endpoints.get(
            name=settings.azure_api_online_endpoint_name
        )
        deployments = list(
            ml_client.online_deployments.list(
                endpoint_name=settings.azure_api_online_endpoint_name
            )
        )

        traffic = getattr(endpoint, "traffic", {}) or {}
        traffic = {str(key): int(value) for key, value in traffic.items()}

        deployment_names = [str(deployment.name) for deployment in deployments]
        selected_name = _select_deployment_name(traffic, deployment_names)

        selected_deployment = None
        for deployment in deployments:
            if deployment.name == selected_name:
                selected_deployment = deployment
                break

        model_name = None
        model_version = None
        provisioning_state = None

        if selected_deployment is not None:
            model_name, model_version = _parse_model_reference(
                getattr(selected_deployment, "model", None)
            )
            provisioning_state = getattr(
                selected_deployment,
                "provisioning_state",
                None,
            )

        return {
            "metadata_available": True,
            "metadata_error": None,
            "endpoint_name": settings.azure_api_online_endpoint_name,
            "selected_deployment": selected_name,
            "endpoint_traffic": traffic,
            "model_name": model_name,
            "model_version": model_version,
            "deployment_provisioning_state": (
                str(provisioning_state) if provisioning_state is not None else None
            ),
        }

    except Exception as exc:
        logger.warning("Could not fetch Azure ML endpoint metadata: %s", exc)
        return {
            "metadata_available": False,
            "metadata_error": str(exc),
            "endpoint_name": settings.azure_api_online_endpoint_name,
        }


def _hand_detector() -> object | None:
    """Return a lazily initialized MediaPipe hand detector.

    Returns:
        MediaPipe hand detector, or ``None`` if initialization fails.
    """
    global _HAND_DETECTOR, _HAND_DETECTOR_INITIALIZED
    if not _HAND_DETECTOR_INITIALIZED:
        _HAND_DETECTOR = init_hand_detector(core_settings.hand_landmarker_path)
        _HAND_DETECTOR_INITIALIZED = True
    return _HAND_DETECTOR


def _encode_crop(image: Any) -> str:
    """Encode a cropped hand image for Azure ML scoring.

    Args:
        image: PIL image crop.

    Returns:
        Base64 data URL suitable for the Azure ML endpoint.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _detect_hands(image_data: str) -> list[Any]:
    """Detect all hands in a frontend camera frame.

    Args:
        image_data: Base64-encoded camera frame from the frontend.

    Returns:
        Detected hand crops with landmarks and wrist coordinates.
    """
    detector = _hand_detector()
    if detector is None:
        return []
    try:
        frame = decode_base64_image(image_data)
        return detect_and_crop_all_hands(
            frame=np.array(frame),
            hands_detector=detector,
        )
    except ValueError as exc:
        logger.debug("Could not decode frame for landmark detection: %s", exc)
        return []


def _process_frame_multi(
    image_data: str,
    tracker: HandTracker,
) -> dict[str, Any]:
    """Run the Azure multi-hand prediction pipeline.

    Args:
        image_data: Base64-encoded camera frame.
        tracker: Per-connection hand tracker.

    Returns:
        Frontend-compatible ``{"hands": [...]}`` response.
    """
    detections = _detect_hands(image_data)
    pairs = tracker.match(detections)
    client = _client()

    hands_output: list[dict[str, Any]] = []
    for detection, slot in pairs:
        prediction = client.predict(
            _encode_crop(detection.crop),
            deployment_name=_deployment_name(),
        )

        letter = prediction.predicted_letter or ""
        confidence = prediction.confidence
        slot.smoother.update(letter, confidence)
        sequence = slot.sequence.update(slot.smoother.stable_letter, True)

        hands_output.append(
            {
                "hand_id": slot.hand_id,
                "label": detection.label,
                "predicted_letter": prediction.predicted_letter,
                "confidence": confidence,
                "top_3": prediction.top_3,
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


def _prediction_entropy(top_3: list[dict[str, Any]]) -> float:
    """Calculate Shannon entropy over normalized top-three confidences.

    Args:
        top_3: Azure endpoint top-three prediction entries.

    Returns:
        Shannon entropy rounded to six decimal places.
    """
    probabilities = [float(item.get("confidence", 0.0)) for item in top_3]
    total = sum(probabilities)
    normalized = [probability / total for probability in probabilities if total > 0]
    entropy = -sum(
        probability * math.log2(probability)
        for probability in normalized
        if probability > 0
    )
    return round(entropy, 6)


def create_app() -> FastAPI:
    """Create the standalone Azure endpoint proxy API.

    Returns:
        Configured FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Dispose the async DB engine on shutdown.

        Database schema management is handled exclusively by Alembic; run
        ``alembic upgrade head`` before starting the server.

        Args:
            app: The FastAPI application instance.

        Yields:
            Nothing. Control returns to FastAPI between startup and shutdown.
        """
        yield
        await engine.dispose()

    app = FastAPI(title=settings.azure_api_app_name, lifespan=lifespan)

    configured_origins = core_settings.cors_origins
    origins = configured_origins or ["*"]
    allow_credentials = bool(configured_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(MonitoringMiddleware)

    # ── Auth routers ────────────────────────────────────────────────
    app.include_router(
        fastapi_users.get_auth_router(auth_backend),
        prefix="/api/auth/jwt",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/api/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_reset_password_router(),
        prefix="/api/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_verify_router(UserRead),
        prefix="/api/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate),
        prefix="/api/users",
        tags=["users"],
    )

    # ── Shared DB-backed routes (stats, progress, admin metrics) ────
    app.include_router(shared_router, prefix="/api")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return the API process health status.

        Returns:
            Static healthy response.
        """
        return HealthResponse()

    @app.get("/info", response_model=AzureApiInfoResponse)
    def info() -> AzureApiInfoResponse:
        """Return non-secret Azure endpoint configuration information.

        Returns:
            API name, endpoint configuration state, deployment routing metadata,
            and model metadata when Azure ML access is available.
        """
        metadata = _fetch_azure_endpoint_metadata()

        return AzureApiInfoResponse(
            app_name=settings.azure_api_app_name,
            endpoint_configured=bool(settings.azure_api_online_endpoint_url),
            model_version=(
                metadata.get("model_version")
                or settings.azure_api_online_model_version
                or None
            ),
            model_name=metadata.get("model_name"),
            endpoint_name=metadata.get("endpoint_name")
            or settings.azure_api_online_endpoint_name
            or None,
            selected_deployment=metadata.get("selected_deployment"),
            default_deployment=settings.azure_api_default_deployment,
            endpoint_traffic=metadata.get("endpoint_traffic", {}),
            deployment_provisioning_state=metadata.get("deployment_provisioning_state"),
            metadata_available=bool(metadata.get("metadata_available")),
            metadata_error=metadata.get("metadata_error"),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(payload: PredictRequest) -> PredictResponse:
        """Forward one base64 image prediction request to Azure ML.

        Args:
            payload: Image request and optional target deployment.

        Returns:
            Normalized prediction response.

        Raises:
            HTTPException: If Azure ML cannot return a valid prediction.
        """
        deployment_name = _deployment_name(payload.deployment_name)
        try:
            prediction = _client().predict(
                payload.image,
                deployment_name=deployment_name,
            )
        except (AzureEndpointError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        await track_prediction(
            prediction.predicted_letter,
            prediction.confidence,
            "http_predict",
            entropy=_prediction_entropy(prediction.top_3),
        )

        return PredictResponse(
            predicted_letter=prediction.predicted_letter,
            confidence=prediction.confidence,
            top_3=[
                TopKItem(
                    letter=str(item["letter"]),
                    confidence=float(item["confidence"]),
                )
                for item in prediction.top_3
            ],
            model_name=prediction.model_name,
            model_version=prediction.model_version,
        )

    @app.post("/api/collect", response_model=CollectResponse)
    def collect(payload: CollectRequest) -> CollectResponse:
        """Store one contributed sample in pending Azure Blob storage.

        Args:
            payload: Labelled collection image and source metadata.

        Returns:
            Stored sample identifier and pending blob path.

        Raises:
            HTTPException: If the sample is invalid or storage is unavailable.
        """
        try:
            letter, source = validate_collection_metadata(
                payload.letter,
                payload.source,
            )
            image_bytes = decode_collection_image(
                payload.image,
                settings.azure_api_collect_max_bytes,
            )
            sample_id, blob_path = store_pending_sample(
                image_bytes=image_bytes,
                letter=letter,
                source=source,
                language=payload.language.strip().upper(),
                settings=settings,
            )
        except CollectionValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CollectionStorageError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return CollectResponse(
            id=sample_id,
            letter=letter,
            blob_path=blob_path,
        )

    @app.websocket("/ws/predict")
    async def ws_predict(websocket: WebSocket) -> None:
        """Stream frontend camera frames to the Azure ML endpoint.

        Args:
            websocket: Browser WebSocket connection opened by the frontend.
        """
        await websocket.accept()
        tracker = HandTracker()
        loop = asyncio.get_running_loop()

        try:
            while True:
                raw_message = await websocket.receive_text()

                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
                    continue

                if message.get("action") == "reset":
                    tracker.clear()
                    await websocket.send_json({"ok": True})
                    continue

                image = message.get("image")
                if not image:
                    await websocket.send_json({"error": "Missing 'image' field"})
                    continue

                try:
                    result = await loop.run_in_executor(
                        None,
                        _process_frame_multi,
                        str(image),
                        tracker,
                    )
                except (AzureEndpointError, ValueError) as exc:
                    await websocket.send_json({"error": str(exc)})
                    continue

                for hand in result.get("hands", []):
                    await track_prediction(
                        hand.get("predicted_letter"),
                        hand.get("confidence", 0.0),
                        "ws_predict",
                        entropy=_prediction_entropy(hand.get("top_3", [])),
                    )

                await websocket.send_json(result)
        except WebSocketDisconnect:
            return

    return app
