"""Tests for the standalone Azure ML endpoint FastAPI app."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from sign_language.core.preprocessing import CroppedHandDetection
from sign_language_azure_api import app as azure_app
from sign_language_azure_api.app import create_app
from sign_language_azure_api.client import AzureEndpointError, AzureEndpointPrediction
from sign_language_azure_api.collection import CollectionStorageError
from sign_language_azure_api.settings import settings


@pytest.fixture(autouse=True)
def _stub_monitoring():
    """Keep Azure API tests from writing to either deployment database."""
    with (
        patch("sign_language.api.monitoring._insert_event", new=AsyncMock()),
        patch("sign_language_azure_api.app.track_prediction", new=AsyncMock()),
    ):
        yield


def test_health_returns_ok() -> None:
    client = TestClient(create_app())
    assert client.get("/health").json() == {"status": "ok"}


def test_health_records_azure_request_metrics() -> None:
    with patch(
        "sign_language.api.monitoring._insert_event",
        new=AsyncMock(),
    ) as insert_event:
        client = TestClient(create_app())
        response = client.get("/health")

    assert response.status_code == 200
    insert_event.assert_awaited_once()
    assert insert_event.call_args.kwargs["path"] == "/health"
    assert insert_event.call_args.kwargs["method"] == "GET"
    assert insert_event.call_args.kwargs["status_code"] == 200


def test_info_reports_endpoint_configuration() -> None:
    with (
        patch.object(settings, "azure_api_online_endpoint_url", "https://example.test"),
        patch.object(settings, "azure_api_online_model_version", "9"),
        patch(
            "sign_language_azure_api.app._fetch_azure_endpoint_metadata",
            return_value={"metadata_available": False},
        ),
    ):
        client = TestClient(create_app())
        body = client.get("/info").json()

    assert body["endpoint_configured"] is True
    assert body["model_version"] == "9"


def test_model_reference_and_deployment_selection_helpers() -> None:
    assert azure_app._parse_model_reference(
        SimpleNamespace(name="model", version=3)
    ) == (
        "model",
        "3",
    )
    assert azure_app._parse_model_reference("azureml:model:4") == ("model", "4")
    assert azure_app._parse_model_reference(
        "azureml://workspaces/ws/models/model/versions/5"
    ) == ("model", "5")
    assert azure_app._parse_model_reference(object()) == (None, None)

    with patch.object(settings, "azure_api_default_deployment", "green"):
        assert azure_app._select_deployment_name({"blue": 100}, ["blue"]) == "green"
    with patch.object(settings, "azure_api_default_deployment", None):
        assert (
            azure_app._select_deployment_name({"blue": 10, "green": 90}, []) == "green"
        )
        assert azure_app._select_deployment_name({}, ["blue"]) == "blue"
        assert azure_app._select_deployment_name({}, ["blue", "green"]) is None


def test_fetch_endpoint_metadata_reports_incomplete_configuration() -> None:
    with patch.object(settings, "azure_api_ml_subscription_id", ""):
        metadata = azure_app._fetch_azure_endpoint_metadata()

    assert metadata["metadata_available"] is False
    assert "incomplete" in metadata["metadata_error"]


def test_fetch_endpoint_metadata_returns_active_deployment() -> None:
    endpoint = SimpleNamespace(traffic={"blue": 10, "green": 90})
    deployments = [
        SimpleNamespace(
            name="blue",
            model="azureml:model:1",
            provisioning_state="Succeeded",
        ),
        SimpleNamespace(
            name="green",
            model="azureml:model:2",
            provisioning_state="Succeeded",
        ),
    ]
    client = SimpleNamespace(
        online_endpoints=SimpleNamespace(get=MagicMock(return_value=endpoint)),
        online_deployments=SimpleNamespace(list=MagicMock(return_value=deployments)),
    )

    with (
        patch.object(settings, "azure_api_ml_subscription_id", "sub"),
        patch.object(settings, "azure_api_ml_resource_group", "rg"),
        patch.object(settings, "azure_api_ml_workspace", "workspace"),
        patch.object(settings, "azure_api_online_endpoint_name", "endpoint"),
        patch.object(settings, "azure_api_default_deployment", None),
        patch("azure.identity.DefaultAzureCredential"),
        patch("azure.ai.ml.MLClient", return_value=client),
    ):
        metadata = azure_app._fetch_azure_endpoint_metadata()

    assert metadata["metadata_available"] is True
    assert metadata["selected_deployment"] == "green"
    assert metadata["model_name"] == "model"
    assert metadata["model_version"] == "2"


def test_fetch_endpoint_metadata_handles_sdk_failure() -> None:
    with (
        patch.object(settings, "azure_api_ml_subscription_id", "sub"),
        patch.object(settings, "azure_api_ml_resource_group", "rg"),
        patch.object(settings, "azure_api_ml_workspace", "workspace"),
        patch.object(settings, "azure_api_online_endpoint_name", "endpoint"),
        patch(
            "azure.identity.DefaultAzureCredential", side_effect=RuntimeError("offline")
        ),
    ):
        metadata = azure_app._fetch_azure_endpoint_metadata()

    assert metadata["metadata_available"] is False
    assert metadata["metadata_error"] == "offline"


def test_detect_hands_handles_missing_detector_and_bad_image() -> None:
    with patch("sign_language_azure_api.app._hand_detector", return_value=None):
        assert azure_app._detect_hands("image") == []

    with (
        patch("sign_language_azure_api.app._hand_detector", return_value=object()),
        patch(
            "sign_language_azure_api.app.decode_base64_image",
            side_effect=ValueError("bad image"),
        ),
    ):
        assert azure_app._detect_hands("image") == []


def test_prediction_entropy_normalizes_endpoint_confidences() -> None:
    entropy = azure_app._prediction_entropy(
        [
            {"letter": "A", "confidence": 0.5},
            {"letter": "B", "confidence": 0.25},
            {"letter": "C", "confidence": 0.25},
        ]
    )

    assert entropy == 1.5


def test_predict_calls_azure_endpoint_client(dummy_image_b64: str) -> None:
    prediction = AzureEndpointPrediction(
        predicted_letter="A",
        confidence=0.91,
        top_3=[{"letter": "A", "confidence": 0.91}],
        model_name="ngt-sign-language",
        model_version="9",
    )

    with (
        patch.object(settings, "azure_api_online_endpoint_url", "https://example.test"),
        patch.object(settings, "azure_api_online_endpoint_key", "key"),
        patch.object(settings, "azure_api_default_deployment", None),
        patch("sign_language_azure_api.app.AzureMLEndpointClient") as mock_client,
        patch(
            "sign_language_azure_api.app.track_prediction",
            new=AsyncMock(),
        ) as track_prediction,
    ):
        mock_client.return_value.predict.return_value = prediction
        client = TestClient(create_app())
        response = client.post("/predict", json={"image": dummy_image_b64})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_letter"] == "A"
    assert body["model_version"] == "9"
    mock_client.return_value.predict.assert_called_once_with(
        dummy_image_b64,
        deployment_name=None,
    )
    track_prediction.assert_awaited_once_with(
        "A",
        0.91,
        "http_predict",
        entropy=0.0,
    )


def test_predict_returns_502_when_endpoint_client_fails(dummy_image_b64: str) -> None:
    with (
        patch.object(settings, "azure_api_online_endpoint_url", "https://example.test"),
        patch.object(settings, "azure_api_online_endpoint_key", "key"),
        patch.object(settings, "azure_api_default_deployment", "blue"),
        patch("sign_language_azure_api.app.AzureMLEndpointClient") as mock_client,
        patch(
            "sign_language_azure_api.app.track_prediction",
            new=AsyncMock(),
        ) as track_prediction,
    ):
        mock_client.return_value.predict.side_effect = AzureEndpointError("offline")
        client = TestClient(create_app())
        response = client.post("/predict", json={"image": dummy_image_b64})

    assert response.status_code == 502
    assert response.json()["detail"] == "offline"
    mock_client.return_value.predict.assert_called_once_with(
        dummy_image_b64,
        deployment_name="blue",
    )
    track_prediction.assert_not_awaited()


def test_collect_stores_pending_sample(dummy_image_b64: str) -> None:
    with patch(
        "sign_language_azure_api.app.store_pending_sample",
        return_value=("sample-id", "pending/A/sample-id.jpg"),
    ) as store:
        client = TestClient(create_app())
        response = client.post(
            "/api/collect",
            json={
                "image": dummy_image_b64,
                "letter": "a",
                "source": "camera",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "sample-id",
        "letter": "A",
        "stored": True,
        "blob_path": "pending/A/sample-id.jpg",
    }
    assert store.call_args.kwargs["letter"] == "A"
    assert store.call_args.kwargs["source"] == "camera"
    assert store.call_args.kwargs["language"] == "NGT"


def test_collect_rejects_invalid_metadata(dummy_image_b64: str) -> None:
    client = TestClient(create_app())
    response = client.post(
        "/api/collect",
        json={
            "image": dummy_image_b64,
            "letter": "1",
            "source": "camera",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Letter must be one character from A to Z."


def test_collect_returns_503_when_storage_fails(dummy_image_b64: str) -> None:
    with patch(
        "sign_language_azure_api.app.store_pending_sample",
        side_effect=CollectionStorageError("Could not store sample in Azure Blob."),
    ):
        client = TestClient(create_app())
        response = client.post(
            "/api/collect",
            json={
                "image": dummy_image_b64,
                "letter": "A",
                "source": "upload",
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Could not store sample in Azure Blob."


def test_ws_predict_bridges_to_azure_endpoint(dummy_image_b64: str) -> None:
    prediction = AzureEndpointPrediction(
        predicted_letter="B",
        confidence=0.84,
        top_3=[{"letter": "B", "confidence": 0.84}],
        model_name="ngt-sign-language",
        model_version="9",
    )

    with (
        patch.object(settings, "azure_api_online_endpoint_url", "https://example.test"),
        patch.object(settings, "azure_api_online_endpoint_key", "key"),
        patch.object(settings, "azure_api_default_deployment", "blue"),
        patch(
            "sign_language_azure_api.app._detect_hands",
            return_value=[
                CroppedHandDetection(
                    label="Left",
                    crop=Image.new("RGB", (16, 16), color=(255, 255, 255)),
                    landmarks=[{"x": 0.1, "y": 0.2, "z": 0.0}],
                    wrist_x=0.1,
                    wrist_y=0.2,
                )
            ],
        ),
        patch("sign_language_azure_api.app.AzureMLEndpointClient") as mock_client,
        patch(
            "sign_language_azure_api.app.track_prediction",
            new=AsyncMock(),
        ) as track_prediction,
    ):
        mock_client.return_value.predict.return_value = prediction
        client = TestClient(create_app())
        with client.websocket_connect("/ws/predict") as websocket:
            websocket.send_json({"image": dummy_image_b64})
            body = websocket.receive_json()

    assert body["hands"][0]["predicted_letter"] == "B"
    assert body["hands"][0]["stable_letter"] is None
    assert body["hands"][0]["label"] == "Left"
    assert body["hands"][0]["landmarks"] == [{"x": 0.1, "y": 0.2, "z": 0.0}]
    call_args = mock_client.return_value.predict.call_args
    assert call_args.kwargs == {"deployment_name": "blue"}
    assert call_args.args[0].startswith("data:image/jpeg;base64,")
    track_prediction.assert_awaited_once_with(
        "B",
        0.84,
        "ws_predict",
        entropy=0.0,
    )


def test_ws_predict_reports_endpoint_errors(dummy_image_b64: str) -> None:
    with (
        patch.object(settings, "azure_api_online_endpoint_url", "https://example.test"),
        patch.object(settings, "azure_api_online_endpoint_key", "key"),
        patch.object(settings, "azure_api_default_deployment", None),
        patch(
            "sign_language_azure_api.app._detect_hands",
            return_value=[
                CroppedHandDetection(
                    label="Left",
                    crop=Image.new("RGB", (16, 16), color=(255, 255, 255)),
                    landmarks=[{"x": 0.1, "y": 0.2, "z": 0.0}],
                    wrist_x=0.1,
                    wrist_y=0.2,
                )
            ],
        ),
        patch("sign_language_azure_api.app.AzureMLEndpointClient") as mock_client,
        patch(
            "sign_language_azure_api.app.track_prediction",
            new=AsyncMock(),
        ) as track_prediction,
    ):
        mock_client.return_value.predict.side_effect = AzureEndpointError("offline")
        client = TestClient(create_app())
        with client.websocket_connect("/ws/predict") as websocket:
            websocket.send_json({"image": dummy_image_b64})
            body = websocket.receive_json()

    assert body == {"error": "offline"}
    track_prediction.assert_not_awaited()


def test_ws_reset_clears_tracker() -> None:
    tracker = MagicMock()
    with patch("sign_language_azure_api.app.HandTracker", return_value=tracker):
        client = TestClient(create_app())
        with client.websocket_connect("/ws/predict") as websocket:
            websocket.send_json({"action": "reset"})
            assert websocket.receive_json() == {"ok": True}

    tracker.clear.assert_called_once_with()
