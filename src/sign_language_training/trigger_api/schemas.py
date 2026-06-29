"""Request and response schemas for the training trigger API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

TriggerReason = Literal["manual", "data_change", "scheduled"]


class TrainTriggerRequest(BaseModel):
    """Request body for ``POST /train``.

    Args:
        reason: Trigger reason controlling which policy is evaluated.
        force: If ``True``, bypasses all policy checks and submits
            retraining unconditionally.
        data_dir: Optional override for the local ImageFolder dataset
            root used for change detection. Defaults to the trigger API setting.
        state_path: Optional override for the local trigger state JSON
            file path. Defaults to the trigger API setting.
        min_new_images: Minimum number of new images required to trigger
            ``data_change`` retraining. Defaults to the trigger API setting.
        interval_days: Minimum days since the last submission
            required to trigger ``scheduled`` retraining. Defaults to the
            trigger API setting.
        experiment_name: Azure ML experiment name for submitted
            retraining jobs. Defaults to the trigger API setting.
    """

    reason: TriggerReason = Field(
        default="manual",
        description="Trigger reason: manual, data_change, or scheduled.",
    )
    force: bool = Field(
        default=False,
        description="Force retraining regardless of policy checks.",
    )
    data_dir: Path | None = Field(
        default=None,
        description="Optional local ImageFolder root for change detection.",
    )
    state_path: Path | None = Field(
        default=None,
        description="Optional path to the local trigger state JSON file.",
    )
    min_new_images: int | None = Field(
        default=None,
        ge=1,
        description="Minimum number of new images required for data-change retraining.",
    )
    interval_days: int | None = Field(
        default=None,
        ge=1,
        description="Minimum interval between scheduled fallback retraining runs.",
    )
    experiment_name: str | None = Field(
        default=None,
        description="Azure ML experiment name used for submitted retraining jobs.",
    )


class TrainTriggerResponse(BaseModel):
    """Response body returned by ``POST /train``.

    Args:
        submitted: Whether a retraining pipeline was submitted.
        reason: The trigger reason that was evaluated.
        message: Human-readable summary of the policy decision.
        current_image_count: Total number of images in the dataset at
            evaluation time.
        new_image_count: Number of images added since the last recorded
            submission state.
        job_name: The Azure ML job name if a pipeline was submitted,
            otherwise ``None``.
        studio_url: The Azure ML Studio URL for the submitted job, or
            ``None`` if no job was submitted.
    """

    submitted: bool
    reason: TriggerReason
    message: str
    current_image_count: int
    new_image_count: int
    job_name: str | None = None
    studio_url: str | None = None


class HealthResponse(BaseModel):
    """Health check response for the trigger API.

    Args:
        status: Service status string, always ``"ok"`` when the service
            is running normally.
        service: Service identifier string.
    """

    status: str = "ok"
    service: str = "training-trigger"
