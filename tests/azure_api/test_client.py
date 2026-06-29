"""Tests for the standalone Azure ML endpoint client."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from sign_language_azure_api.client import AzureEndpointError, AzureMLEndpointClient


class _FakeResponse:
    def __init__(self, payload: dict | str):
        self.status = 200
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


class _RawResponse:
    def __init__(self, body: str):
        self.status = 200
        self._body = body

    def __enter__(self) -> "_RawResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body.encode("utf-8")


def test_client_rejects_missing_endpoint_url() -> None:
    with pytest.raises(ValueError, match="URL cannot be empty"):
        AzureMLEndpointClient(" ", "key", 3.0)


def test_client_rejects_missing_endpoint_key() -> None:
    with pytest.raises(ValueError, match="key cannot be empty"):
        AzureMLEndpointClient("https://example.test/score", " ", 3.0)


def test_predict_normalizes_endpoint_response() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)
    payload = {
        "predicted_letter": "A",
        "confidence": 0.95,
        "top_3": [{"letter": "A", "confidence": 0.95}],
        "model_name": "ngt-sign-language",
        "model_version": "7",
    }

    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        result = client.predict("abc")

    assert result.predicted_letter == "A"
    assert result.confidence == pytest.approx(0.95)
    assert result.top_3 == [{"letter": "A", "confidence": 0.95}]
    assert result.model_version == "7"


def test_predict_can_route_to_deployment() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)

    with patch("urllib.request.urlopen", return_value=_FakeResponse({})) as opened:
        client.predict("abc", deployment_name="green")

    request = opened.call_args.args[0]
    assert request.headers["Azureml-model-deployment"] == "green"


def test_predict_raises_on_endpoint_error_payload() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)

    with (
        patch("urllib.request.urlopen", return_value=_FakeResponse({"error": "bad"})),
        pytest.raises(AzureEndpointError, match="bad"),
    ):
        client.predict("abc")


def test_predict_raises_on_http_error() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)
    error = urllib.error.HTTPError(
        url="https://example.test/score",
        code=500,
        msg="server error",
        hdrs=MagicMock(),
        fp=io.BytesIO(b"failed"),
    )

    with (
        patch("urllib.request.urlopen", side_effect=error),
        pytest.raises(AzureEndpointError, match="HTTP 500"),
    ):
        client.predict("abc")


def test_predict_raises_on_url_error() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)

    with (
        patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("network down"),
        ),
        pytest.raises(AzureEndpointError, match="request failed"),
    ):
        client.predict("abc")


def test_predict_raises_on_invalid_json() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)

    with (
        patch("urllib.request.urlopen", return_value=_RawResponse("not-json")),
        pytest.raises(AzureEndpointError, match="invalid JSON"),
    ):
        client.predict("abc")


def test_predict_raises_on_invalid_json_string_payload() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)

    with (
        patch(
            "urllib.request.urlopen",
            return_value=_RawResponse(json.dumps("not-json")),
        ),
        pytest.raises(AzureEndpointError, match="invalid JSON string"),
    ):
        client.predict("abc")


def test_predict_raises_on_non_object_json() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)

    with (
        patch("urllib.request.urlopen", return_value=_RawResponse(json.dumps([]))),
        pytest.raises(AzureEndpointError, match="must be a JSON object"),
    ):
        client.predict("abc")


def test_predict_normalizes_sparse_and_mixed_top3_payload() -> None:
    client = AzureMLEndpointClient("https://example.test/score", "key", 3.0)
    payload = {
        "predicted_letter": None,
        "confidence": "0.5",
        "top_3": [
            {"letter": "B", "confidence": "0.5"},
            "invalid-item",
            {"letter": None},
        ],
    }

    with patch("urllib.request.urlopen", return_value=_FakeResponse(payload)):
        result = client.predict("abc")

    assert result.predicted_letter is None
    assert result.confidence == pytest.approx(0.5)
    assert result.top_3 == [
        {"letter": "B", "confidence": 0.5},
        {"letter": "None", "confidence": 0.0},
    ]
    assert result.model_name is None
    assert result.model_version is None


def test_predict_builds_expected_authorization_request() -> None:
    client = AzureMLEndpointClient(" https://example.test/score ", " key ", 3.0)

    with patch("urllib.request.urlopen", return_value=_FakeResponse({})) as opened:
        client.predict("abc")

    request: urllib.request.Request = opened.call_args.args[0]
    assert request.full_url == "https://example.test/score"
    assert request.headers["Authorization"] == "Bearer key"
    assert request.data == b'{"image": "abc"}'
