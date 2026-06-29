"""Route tests for the training trigger API."""

from __future__ import annotations

from pathlib import Path

import pytest

from sign_language_training.orchestration.trigger_policy import TriggerDecision
from sign_language_training.trigger_api import app as trigger_app_module
from sign_language_training.trigger_api.settings import settings

_TRAIN = "/train"


def _decision(
    *,
    submitted: bool = False,
    reason: str = "manual",
    message: str = "Skipped.",
    current_image_count: int = 50,
    new_image_count: int = 0,
    job_name: str | None = None,
    studio_url: str | None = None,
) -> TriggerDecision:
    return TriggerDecision(
        should_submit=submitted,
        reason=reason,  # type: ignore[arg-type]
        message=message,
        current_image_count=current_image_count,
        new_image_count=new_image_count,
        submitted_job_name=job_name,
        studio_url=studio_url,
    )


class TestSecurity:
    def test_train_without_api_key_returns_401(self, client) -> None:
        response = client.post(_TRAIN, json={"reason": "manual"})

        assert response.status_code == 401

    def test_train_with_wrong_api_key_returns_401(self, client) -> None:
        response = client.post(
            _TRAIN,
            headers={"X-API-Key": "wrong"},
            json={"reason": "manual"},
        )

        assert response.status_code == 401

    def test_train_returns_503_when_api_key_not_configured(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi.testclient import TestClient
        from sign_language_training.trigger_api.app import create_app

        monkeypatch.setattr(settings, "training_trigger_api_key", None)

        app = create_app()
        with TestClient(app) as c:
            response = c.post(
                _TRAIN,
                headers={"X-API-Key": "anything"},
                json={"reason": "manual"},
            )

        assert response.status_code == 503


class TestTrainRoute:
    def test_manual_force_submits_training(
        self,
        client,
        trigger_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict = {}

        def fake_evaluate_and_maybe_submit(*, reason, config, force):
            captured["reason"] = reason
            captured["config"] = config
            captured["force"] = force
            return _decision(
                submitted=True,
                reason=reason,
                message="Submitted Azure ML retraining pipeline.",
                current_image_count=120,
                new_image_count=120,
                job_name="test-job",
                studio_url="https://studio.example/job",
            )

        monkeypatch.setattr(
            trigger_app_module,
            "evaluate_and_maybe_submit",
            fake_evaluate_and_maybe_submit,
        )
        monkeypatch.setattr(
            trigger_app_module,
            "raw_data_asset_reference",
            lambda: "azureml:ngt-raw:1",
        )
        monkeypatch.setattr(
            trigger_app_module.azure_settings,
            "azure_raw_data_asset_version",
            "1",
        )

        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={"reason": "manual", "force": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["submitted"] is True
        assert data["reason"] == "manual"
        assert data["job_name"] == "test-job"
        assert data["studio_url"] == "https://studio.example/job"
        assert captured["reason"] == "manual"
        assert captured["force"] is True

    def test_data_change_skip_response(
        self,
        client,
        trigger_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_evaluate_and_maybe_submit(*, reason, config, force):
            return _decision(
                submitted=False,
                reason=reason,
                message="Skipped retraining: 10 new images < threshold 100.",
                current_image_count=80,
                new_image_count=10,
            )

        monkeypatch.setattr(
            trigger_app_module,
            "evaluate_and_maybe_submit",
            fake_evaluate_and_maybe_submit,
        )
        monkeypatch.setattr(
            trigger_app_module,
            "raw_data_asset_reference",
            lambda: "azureml:ngt-raw:1",
        )

        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={"reason": "data_change"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["submitted"] is False
        assert data["reason"] == "data_change"
        assert data["current_image_count"] == 80
        assert data["new_image_count"] == 10
        assert data["job_name"] is None

    def test_scheduled_submit_response(
        self,
        client,
        trigger_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_evaluate_and_maybe_submit(*, reason, config, force):
            return _decision(
                submitted=True,
                reason=reason,
                message="Submitted Azure ML retraining pipeline.",
                current_image_count=120,
                new_image_count=0,
                job_name="scheduled-job",
                studio_url="https://studio.example/scheduled-job",
            )

        monkeypatch.setattr(
            trigger_app_module,
            "evaluate_and_maybe_submit",
            fake_evaluate_and_maybe_submit,
        )
        monkeypatch.setattr(
            trigger_app_module,
            "raw_data_asset_reference",
            lambda: "azureml:ngt-raw:1",
        )

        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={"reason": "scheduled"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["submitted"] is True
        assert data["reason"] == "scheduled"
        assert data["job_name"] == "scheduled-job"
        assert data["studio_url"] == "https://studio.example/scheduled-job"

    def test_builds_policy_config_from_payload_overrides(
        self,
        client,
        trigger_api_key: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict = {}

        data_dir = tmp_path / "data"
        state_path = tmp_path / "state.json"

        def fake_evaluate_and_maybe_submit(*, reason, config, force):
            captured["config"] = config
            return _decision(reason=reason)

        monkeypatch.setattr(
            trigger_app_module,
            "evaluate_and_maybe_submit",
            fake_evaluate_and_maybe_submit,
        )
        monkeypatch.setattr(
            trigger_app_module,
            "raw_data_asset_reference",
            lambda: "azureml:ngt-raw:5",
        )
        monkeypatch.setattr(
            trigger_app_module.azure_settings,
            "azure_raw_data_asset_version",
            "5",
        )

        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={
                "reason": "scheduled",
                "force": False,
                "data_dir": str(data_dir),
                "state_path": str(state_path),
                "min_new_images": 25,
                "interval_days": 14,
                "experiment_name": "test-exp",
            },
        )

        assert response.status_code == 200
        config = captured["config"]
        assert config.data_dir == data_dir
        assert config.state_path == state_path
        assert config.raw_data_asset == "azureml:ngt-raw:5"
        assert config.raw_data_version == "5"
        assert config.min_new_images == 25
        assert config.interval_days == 14
        assert config.experiment_name == "test-exp"

    def test_invalid_reason_returns_422(
        self,
        client,
        trigger_api_key: str,
    ) -> None:
        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={"reason": "bad_reason"},
        )

        assert response.status_code == 422

    def test_policy_file_not_found_returns_400(
        self,
        client,
        trigger_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_evaluate_and_maybe_submit(*, reason, config, force):
            raise FileNotFoundError("Dataset directory not found")

        monkeypatch.setattr(
            trigger_app_module,
            "evaluate_and_maybe_submit",
            fake_evaluate_and_maybe_submit,
        )
        monkeypatch.setattr(
            trigger_app_module,
            "raw_data_asset_reference",
            lambda: "azureml:ngt-raw:1",
        )

        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={"reason": "manual", "force": True},
        )

        assert response.status_code == 400
        assert "Dataset directory not found" in response.json()["detail"]

    def test_unexpected_policy_error_returns_500(
        self,
        client,
        trigger_api_key: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_evaluate_and_maybe_submit(*, reason, config, force):
            raise RuntimeError("Azure failed")

        monkeypatch.setattr(
            trigger_app_module,
            "evaluate_and_maybe_submit",
            fake_evaluate_and_maybe_submit,
        )
        monkeypatch.setattr(
            trigger_app_module,
            "raw_data_asset_reference",
            lambda: "azureml:ngt-raw:1",
        )

        response = client.post(
            _TRAIN,
            headers={"X-API-Key": trigger_api_key},
            json={"reason": "manual", "force": True},
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Training trigger failed."
