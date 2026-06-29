"""Client for Azure ML managed online endpoint inference."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AzureEndpointPrediction:
    """Normalized prediction returned by the Azure ML online endpoint."""

    predicted_letter: str | None
    confidence: float
    top_3: list[dict[str, float | str]]
    model_name: str | None = None
    model_version: str | None = None


class AzureEndpointError(RuntimeError):
    """Raised when the Azure ML endpoint cannot return a valid prediction."""


class AzureMLEndpointClient:
    """Small synchronous HTTP client for Azure ML online endpoint calls."""

    def __init__(self, endpoint_url: str, endpoint_key: str, timeout_seconds: float):
        """Initialize the Azure ML online endpoint client.

        Args:
            endpoint_url: Azure ML scoring URI.
            endpoint_key: Endpoint authentication key.
            timeout_seconds: Maximum duration for one endpoint request.

        Raises:
            ValueError: If the endpoint URL or key is empty.
        """
        if not endpoint_url.strip():
            raise ValueError("Azure ML endpoint URL cannot be empty.")
        if not endpoint_key.strip():
            raise ValueError("Azure ML endpoint key cannot be empty.")
        self.endpoint_url = endpoint_url.strip()
        self.endpoint_key = endpoint_key.strip()
        self.timeout_seconds = timeout_seconds

    def predict(
        self, image_data: str, deployment_name: str | None = None
    ) -> AzureEndpointPrediction:
        """Send one base64 image to the Azure ML online endpoint.

        Args:
            image_data: Base64-encoded image content.
            deployment_name: Optional deployment targeted by the request.

        Returns:
            Normalized Azure endpoint prediction.

        Raises:
            AzureEndpointError: If the request fails or Azure returns invalid data.
        """
        payload = json.dumps({"image": image_data}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.endpoint_key}",
        }
        if deployment_name:
            headers["azureml-model-deployment"] = deployment_name

        request = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise AzureEndpointError(
                f"Azure ML endpoint returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AzureEndpointError(
                f"Azure ML endpoint request failed: {exc}"
            ) from exc

        try:
            parsed: Any = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise AzureEndpointError(
                f"Azure ML endpoint returned invalid JSON: {raw_body[:200]}"
            ) from exc

        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError as exc:
                raise AzureEndpointError(
                    f"Azure ML endpoint returned invalid JSON string: {parsed[:200]}"
                ) from exc

        if not isinstance(parsed, dict):
            raise AzureEndpointError(
                "Azure ML endpoint response must be a JSON object."
            )

        if parsed.get("error"):
            raise AzureEndpointError(str(parsed["error"]))

        return _normalize_prediction(parsed)


def _normalize_prediction(payload: dict[str, Any]) -> AzureEndpointPrediction:
    """Normalize endpoint JSON into a stable API shape.

    Args:
        payload: Parsed Azure ML scoring response.

    Returns:
        Normalized prediction object.
    """
    top_3_raw = payload.get("top_3") or []
    top_3: list[dict[str, float | str]] = []
    if isinstance(top_3_raw, list):
        for item in top_3_raw:
            if not isinstance(item, dict):
                continue
            top_3.append(
                {
                    "letter": str(item.get("letter", "")),
                    "confidence": float(item.get("confidence", 0.0)),
                }
            )

    predicted_letter = payload.get("predicted_letter")
    return AzureEndpointPrediction(
        predicted_letter=(
            str(predicted_letter) if predicted_letter is not None else None
        ),
        confidence=float(payload.get("confidence", 0.0)),
        top_3=top_3,
        model_name=(
            str(payload["model_name"])
            if payload.get("model_name") is not None
            else None
        ),
        model_version=(
            str(payload["model_version"])
            if payload.get("model_version") is not None
            else None
        ),
    )
