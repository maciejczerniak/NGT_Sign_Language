"""Tests for Azure ML configuration and resolver helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sign_language_training import azure_config


def test_require_setting_strips_value_and_rejects_empty() -> None:
    assert azure_config.require_setting("workspace", " team-R2-2026 ") == "team-R2-2026"

    with pytest.raises(ValueError, match="WORKSPACE must be set"):
        azure_config.require_setting("workspace", " ")


@pytest.mark.parametrize(
    ("mode", "credential_name"),
    [
        ("default", "AzureCliCredential"),
        ("interactive", "InteractiveBrowserCredential"),
        ("managed_identity", "ManagedIdentityCredential"),
    ],
)
def test_get_credential_selects_auth_mode(mode: str, credential_name: str) -> None:
    with (
        patch.object(azure_config.settings, "azure_auth_mode", mode),
        patch.object(azure_config, credential_name) as credential,
    ):
        assert azure_config.get_credential() is credential.return_value


def test_get_credential_rejects_unknown_mode() -> None:
    with (
        patch.object(azure_config.settings, "azure_auth_mode", "password"),
        pytest.raises(ValueError, match="Invalid AZURE_AUTH_MODE"),
    ):
        azure_config.get_credential()


def test_get_client_uses_required_workspace_settings() -> None:
    credential = object()
    with (
        patch.object(azure_config.settings, "azure_subscription_id", "sub"),
        patch.object(azure_config.settings, "azure_resource_group", "rg"),
        patch.object(azure_config.settings, "azure_workspace", "workspace"),
        patch.object(azure_config, "get_credential", return_value=credential),
        patch.object(azure_config, "MLClient") as ml_client,
    ):
        result = azure_config.get_client()

    assert result is ml_client.return_value
    ml_client.assert_called_once_with(
        credential=credential,
        subscription_id="sub",
        resource_group_name="rg",
        workspace_name="workspace",
    )


def test_get_mlflow_tracking_uri_uses_client_workspace_name() -> None:
    workspace = SimpleNamespace(mlflow_tracking_uri="azureml://tracking")
    client = SimpleNamespace(
        workspace_name="workspace",
        workspaces=SimpleNamespace(get=MagicMock(return_value=workspace)),
    )

    assert azure_config.get_mlflow_tracking_uri(client) == "azureml://tracking"
    client.workspaces.get.assert_called_once_with("workspace")


def test_get_mlflow_tracking_uri_rejects_missing_uri() -> None:
    client = SimpleNamespace(
        workspace_name="workspace",
        workspaces=SimpleNamespace(
            get=MagicMock(return_value=SimpleNamespace(mlflow_tracking_uri=None))
        ),
    )

    with pytest.raises(RuntimeError, match="Could not resolve"):
        azure_config.get_mlflow_tracking_uri(client)


def test_configure_mlflow_tracking_prefers_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.test")

    with (
        patch.object(azure_config.mlflow, "set_tracking_uri") as set_uri,
        patch.object(azure_config.mlflow, "set_experiment") as set_experiment,
    ):
        result = azure_config.configure_mlflow_tracking("experiment")

    assert result == "https://tracking.test"
    set_uri.assert_called_once_with("https://tracking.test")
    set_experiment.assert_called_once_with("experiment")


def test_compute_helpers_list_print_and_resolve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = [
        SimpleNamespace(
            name="pending", type="kubernetes", provisioning_state="Creating"
        ),
        SimpleNamespace(
            name="ready", type="kubernetes", provisioning_state="Succeeded"
        ),
    ]
    client = SimpleNamespace(compute=SimpleNamespace(list=lambda: targets))

    with patch.object(azure_config.settings, "azure_compute_target", None):
        assert azure_config.list_compute_targets(client) == targets
        assert azure_config.resolve_compute_target(client) == "ready"
        azure_config.print_compute_targets(client)

    assert "ready: kubernetes, provisioning_state=Succeeded" in capsys.readouterr().out


def test_resolve_compute_target_handles_override_fallback_and_empty() -> None:
    with patch.object(azure_config.settings, "azure_compute_target", "lambda-2"):
        assert azure_config.resolve_compute_target() == "lambda-2"

    client = SimpleNamespace(
        compute=SimpleNamespace(
            list=lambda: [SimpleNamespace(name="first", provisioning_state="Creating")]
        )
    )
    with patch.object(azure_config.settings, "azure_compute_target", None):
        assert azure_config.resolve_compute_target(client) == "first"

    empty = SimpleNamespace(compute=SimpleNamespace(list=lambda: []))
    with (
        patch.object(azure_config.settings, "azure_compute_target", None),
        pytest.raises(ValueError, match="No Azure ML compute targets"),
    ):
        azure_config.resolve_compute_target(empty)


def test_environment_helpers_list_print_and_resolve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    environments = [SimpleNamespace(name="training", version="3")]
    client = SimpleNamespace(environments=SimpleNamespace(list=lambda: environments))

    with (
        patch.object(azure_config.settings, "azure_environment_name", None),
        patch.object(azure_config.settings, "azure_environment_version", None),
    ):
        assert azure_config.list_environments(client) == environments
        assert azure_config.resolve_environment(client) == "azureml:training:3"
        azure_config.print_environments(client)

    assert "training: version=3" in capsys.readouterr().out
    assert azure_config.environment_reference("training") == "azureml:training@latest"


def test_resolve_environment_handles_override_and_empty() -> None:
    with (
        patch.object(azure_config.settings, "azure_environment_name", "training"),
        patch.object(azure_config.settings, "azure_environment_version", "4"),
    ):
        assert azure_config.resolve_environment() == "azureml:training:4"

    empty = SimpleNamespace(environments=SimpleNamespace(list=lambda: []))
    with (
        patch.object(azure_config.settings, "azure_environment_name", None),
        pytest.raises(ValueError, match="No Azure ML environments"),
    ):
        azure_config.resolve_environment(empty)


def test_asset_checkpoint_and_instance_references() -> None:
    with (
        patch.object(azure_config.settings, "azure_raw_data_asset_name", "raw"),
        patch.object(azure_config.settings, "azure_raw_data_asset_version", "6"),
    ):
        assert azure_config.raw_data_asset_reference() == "azureml:raw:6"

    with (
        patch.object(
            azure_config.settings,
            "azure_pretrained_checkpoint_asset_name",
            "checkpoint",
        ),
        patch.object(
            azure_config.settings,
            "azure_pretrained_checkpoint_asset_version",
            "2",
        ),
    ):
        assert (
            azure_config.pretrained_checkpoint_reference_or_path()
            == "azureml:checkpoint:2"
        )

    with (
        patch.object(
            azure_config.settings, "azure_pretrained_checkpoint_asset_name", None
        ),
        patch.object(
            azure_config.settings, "azure_pretrained_checkpoint_asset_version", None
        ),
        patch.object(azure_config.settings, "model_path", "models/base.pth"),
    ):
        assert (
            azure_config.pretrained_checkpoint_reference_or_path() == "models/base.pth"
        )

    with (
        patch.object(azure_config.settings, "azure_instance_type", None),
        patch.object(azure_config.settings, "azure_prefer_gpu", True),
    ):
        assert azure_config.resolve_instance_type() == "gpu"

    with (
        patch.object(azure_config.settings, "azure_instance_type", None),
        patch.object(azure_config.settings, "azure_prefer_gpu", False),
    ):
        assert azure_config.resolve_instance_type() == "cpu-xl"
