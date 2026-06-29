"""Tests for the standalone Azure API CLI and settings."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sign_language_azure_api import main as azure_main
from sign_language_azure_api.settings import AzureApiSettings


def test_serve_uses_settings_defaults() -> None:
    with (
        patch.object(azure_main.settings, "azure_api_host", "127.0.0.1"),
        patch.object(azure_main.settings, "azure_api_port", 8123),
        patch("sign_language_azure_api.main.uvicorn.run") as run,
    ):
        azure_main.serve(host=None, port=None, reload=False)

    run.assert_called_once_with(
        "sign_language_azure_api.app:create_app",
        factory=True,
        host="127.0.0.1",
        port=8123,
        reload=False,
    )


def test_serve_accepts_cli_overrides() -> None:
    with patch("sign_language_azure_api.main.uvicorn.run") as run:
        azure_main.serve(host="0.0.0.0", port=9000, reload=True)

    run.assert_called_once_with(
        "sign_language_azure_api.app:create_app",
        factory=True,
        host="0.0.0.0",
        port=9000,
        reload=True,
    )


def test_main_runs_typer_app() -> None:
    with patch.object(azure_main, "app") as app:
        azure_main.main()

    app.assert_called_once()


def test_azure_api_settings_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        AzureApiSettings(azure_api_online_timeout_seconds=0)


def test_azure_api_settings_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="port must be between"):
        AzureApiSettings(azure_api_port=70000)
