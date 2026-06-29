"""Azure ML managed online endpoint scoring script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.score_common import ScoringRuntime  # noqa: E402


runtime: ScoringRuntime | None = None


def init() -> None:
    """Load the registered model when the endpoint container starts.

    Azure ML calls this function once during worker initialization. The loaded
    runtime is retained globally for subsequent scoring requests.
    """
    global runtime
    runtime = ScoringRuntime.from_model_dir()


def _parse_payload(raw_data: Any) -> dict[str, Any]:
    """Normalize an Azure inference request body into a JSON object.

    Args:
        raw_data: Request body supplied by the Azure ML inference server.

    Returns:
        Parsed request payload.

    Raises:
        ValueError: If the request body is not bytes, JSON text, or a dictionary.
        json.JSONDecodeError: If a string request body is not valid JSON.
    """
    if isinstance(raw_data, bytes):
        raw_data = raw_data.decode("utf-8")
    if isinstance(raw_data, str):
        parsed = json.loads(raw_data)
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object.")
        return parsed
    if isinstance(raw_data, dict):
        return raw_data
    raise ValueError("Request body must be a JSON object.")


def run(raw_data: Any) -> dict[str, Any]:
    """Score one image request through the initialized model runtime.

    Args:
        raw_data: Request body containing ``{"image": "<base64 image>"}``.

    Returns:
        Prediction metadata or an error object.

    Raises:
        RuntimeError: If Azure calls the scoring function before initialization.
    """
    if runtime is None:
        raise RuntimeError("Scoring runtime has not been initialized.")

    try:
        payload = _parse_payload(raw_data)
        image_data = payload.get("image")
        if not image_data:
            return {"error": "Missing required field: image"}
        return runtime.predict_base64(str(image_data))
    except Exception as exc:
        return {"error": str(exc)}
