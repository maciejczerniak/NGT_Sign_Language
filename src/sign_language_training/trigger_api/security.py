"""Security helpers for the training trigger API.

Provides API key authentication for the ``POST /train`` endpoint using
the ``X-API-Key`` request header. The API fails closed if
``TRAINING_TRIGGER_API_KEY`` is not configured, since submitting an
unconfigured trigger could initiate paid Azure ML jobs.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from sign_language_training.trigger_api.settings import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(api_key_header)) -> None:
    """Validate the ``X-API-Key`` header against the configured trigger API key.

    Used as a FastAPI dependency on the ``POST /train`` endpoint. The check
    uses :func:`secrets.compare_digest` to prevent timing attacks.

    The endpoint fails with ``503 Service Unavailable`` rather than ``401``
    when the key is not configured, to clearly distinguish a misconfigured
    deployment from an invalid client key.

    Args:
        api_key: The API key extracted from the ``X-API-Key`` request
            header by the :class:`~fastapi.security.APIKeyHeader` dependency,
            or ``None`` if the header is absent.

    Raises:
        HTTPException: With status ``503`` if ``TRAINING_TRIGGER_API_KEY``
            is not set in the trigger API settings.
        HTTPException: With status ``401`` if the provided key is missing
            or does not match the configured key.
    """
    expected_key = get_settings().training_trigger_api_key

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training trigger API key is not configured.",
        )

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
