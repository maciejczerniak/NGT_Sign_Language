"""FastAPI app for manual, scheduled, and data-change training triggers.

Exposes two endpoints:

- ``GET /health`` — liveness check.
- ``POST /train`` — evaluate the retraining trigger policy and optionally
  submit an Azure ML preprocessing and training pipeline job.

The ``/train`` endpoint is protected by an API key passed in the
``X-API-Key`` header.
"""

from __future__ import annotations

import logging
from fastapi import FastAPI, Depends, HTTPException, status
from sign_language_training.azure_config import (
    raw_data_asset_reference,
    settings as azure_settings,
)
from sign_language_training.orchestration.trigger_policy import (
    TriggerPolicyConfig,
    evaluate_and_maybe_submit,
)
from sign_language_training.trigger_api.schemas import (
    HealthResponse,
    TrainTriggerRequest,
    TrainTriggerResponse,
)
from sign_language_training.trigger_api.security import require_api_key
from sign_language_training.trigger_api.settings import (
    get_settings,
    resolve_project_path,
)

logger = logging.getLogger(__name__)


def _azure_data_reference() -> tuple[str, str]:
    """Resolve Azure data asset settings for request handling.

    Returns:
        Raw data asset reference and configured asset version.
    """
    return (
        raw_data_asset_reference(),
        str(azure_settings.azure_raw_data_asset_version),
    )


def _build_policy_config(payload: TrainTriggerRequest) -> TriggerPolicyConfig:
    """Build trigger policy configuration from a request.

    Args:
        payload: Training-trigger request overrides.

    Returns:
        Resolved trigger policy configuration.
    """
    settings = get_settings()
    data_dir = (
        resolve_project_path(payload.data_dir)
        if payload.data_dir is not None
        else resolve_project_path(settings.training_trigger_data_dir)
    )

    state_path = (
        resolve_project_path(payload.state_path)
        if payload.state_path is not None
        else resolve_project_path(settings.training_trigger_state_path)
    )
    raw_data_asset, raw_data_version = _azure_data_reference()

    return TriggerPolicyConfig(
        data_dir=data_dir,
        state_path=state_path,
        raw_data_asset=raw_data_asset,
        raw_data_version=raw_data_version,
        min_new_images=(
            payload.min_new_images
            if payload.min_new_images is not None
            else settings.training_trigger_min_new_images
        ),
        interval_days=(
            payload.interval_days
            if payload.interval_days is not None
            else settings.training_trigger_interval_days
        ),
        experiment_name=(
            payload.experiment_name
            if payload.experiment_name is not None
            else settings.training_trigger_experiment_name
        ),
    )


def create_app() -> FastAPI:
    """Create the training-trigger FastAPI application.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="Sign Language Training Trigger API",
        version="0.1.0",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Return the trigger API health status.

        Returns:
            Healthy service response.
        """
        return HealthResponse()

    @app.post(
        "/train",
        response_model=TrainTriggerResponse,
        dependencies=[Depends(require_api_key)],
        status_code=status.HTTP_200_OK,
    )
    def train(payload: TrainTriggerRequest) -> TrainTriggerResponse:
        """Evaluate retraining policy and submit Azure ML when needed.

        Args:
            payload: Trigger reason and optional policy overrides.

        Returns:
            Policy decision and submitted job metadata.

        Raises:
            HTTPException: If request values are invalid or submission fails.
        """
        try:
            config = _build_policy_config(payload)
            decision = evaluate_and_maybe_submit(
                reason=payload.reason,
                config=config,
                force=payload.force,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception("Training trigger failed.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Training trigger failed.",
            ) from exc

        return TrainTriggerResponse(
            submitted=decision.should_submit,
            reason=decision.reason,
            message=decision.message,
            current_image_count=decision.current_image_count,
            new_image_count=decision.new_image_count,
            job_name=decision.submitted_job_name,
            studio_url=decision.studio_url,
        )

    return app


# ASGI entrypoint for uvicorn: sign_language_training.trigger_api.app:app.
# Tests and embedded callers can still use create_app() when they need a fresh app.
app = create_app()
