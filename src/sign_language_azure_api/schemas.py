"""Schemas for the separate Azure ML endpoint API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class AzureApiInfoResponse(BaseModel):
    app_name: str
    endpoint_configured: bool

    model_version: str | None = None
    model_name: str | None = None

    endpoint_name: str | None = None
    selected_deployment: str | None = None
    default_deployment: str | None = None
    endpoint_traffic: dict[str, int] = Field(default_factory=dict)
    deployment_provisioning_state: str | None = None

    metadata_available: bool = False
    metadata_error: str | None = None


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded frame, raw or data URL.")
    deployment_name: str | None = Field(
        default=None,
        description="Optional Azure ML deployment name to route to directly.",
    )


class TopKItem(BaseModel):
    letter: str
    confidence: float


class PredictResponse(BaseModel):
    predicted_letter: str | None
    confidence: float
    top_3: list[TopKItem]
    model_name: str | None = None
    model_version: str | None = None


class CollectRequest(BaseModel):
    """One labelled sample submitted from the frontend collection page."""

    image: str = Field(..., description="Base64-encoded image, raw or data URL.")
    letter: str = Field(..., min_length=1, max_length=1)
    source: str = Field(..., description='"camera", "upload", or "auto".')
    language: str = Field(default="NGT", min_length=1, max_length=10)


class CollectResponse(BaseModel):
    """Confirmation that one sample was stored in pending Azure Blob storage."""

    id: str
    letter: str
    stored: bool = True
    blob_path: str
