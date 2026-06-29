"""Tests for the optional live Azure endpoint smoke-test script."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import typer


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "test_online_endpoint.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "test_online_endpoint_script", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_endpoint_script_skips_when_config_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module()
    monkeypatch.setattr(module.azure_api_settings, "azure_api_online_endpoint_url", "")
    monkeypatch.setattr(module.azure_api_settings, "azure_api_online_endpoint_key", "")

    with pytest.raises(typer.Exit) as exc_info:
        module.main(endpoint_url=None, endpoint_key=None, deployment_name=None)

    assert exc_info.value.exit_code == 0
    assert "Skipping live Azure ML endpoint checks" in capsys.readouterr().out


def test_live_endpoint_script_uses_configured_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module()
    calls: list[tuple[str, str, dict, str | None]] = []

    def fake_post(
        endpoint_url: str,
        endpoint_key: str,
        payload: dict,
        deployment_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> tuple[int, dict, float]:
        calls.append((endpoint_url, endpoint_key, payload, deployment_name))
        if payload.get("image") == "not-base64" or "image" not in payload:
            return 400, {"error": "bad request"}, 0.01
        return (
            200,
            {"predicted_letter": "A", "confidence": 0.9, "top_3": []},
            0.01,
        )

    monkeypatch.setattr(
        module.azure_api_settings,
        "azure_api_online_endpoint_url",
        "https://example.test/score",
    )
    monkeypatch.setattr(
        module.azure_api_settings, "azure_api_online_endpoint_key", "key"
    )
    monkeypatch.setattr(
        module.azure_api_settings, "azure_api_default_deployment", "blue"
    )
    monkeypatch.setattr(module, "_post", fake_post)

    module.main(
        endpoint_url=None,
        endpoint_key=None,
        deployment_name=None,
        max_latency_seconds=1.0,
    )

    assert len(calls) == 3
    assert calls[0][0] == "https://example.test/score"
    assert calls[0][1] == "key"
    assert calls[0][3] == "blue"
