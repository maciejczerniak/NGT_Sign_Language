"""Tests for Azure ML online endpoint canary rollout helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import typer
from azure.ai.ml.entities import KubernetesOnlineDeployment, KubernetesOnlineEndpoint
from azure.core.exceptions import ResourceNotFoundError


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from deploy_online_endpoint import (  # noqa: E402
    build_traffic_allocation,
    deploy_online_endpoint,
    get_or_create_endpoint,
)
from endpoint_common import resolve_endpoint_instance_type  # noqa: E402
from rollout_canary import main, parse_canary_steps  # noqa: E402


def test_build_traffic_allocation_preserves_stable_for_zero_percent() -> None:
    assert build_traffic_allocation({"stable": 100}, "canary", 0) == {
        "stable": 100,
        "canary": 0,
    }


def test_build_traffic_allocation_switches_all_traffic_at_100_percent() -> None:
    assert build_traffic_allocation({"stable": 100, "canary": 0}, "canary", 100) == {
        "stable": 0,
        "canary": 100,
    }


def test_kubernetes_endpoint_instance_type_defaults_to_gpu() -> None:
    with patch("endpoint_common.azure_settings.azure_instance_type", None):
        assert resolve_endpoint_instance_type() == "gpu"


def test_kubernetes_endpoint_rejects_managed_vm_sku() -> None:
    with pytest.raises(ValueError, match="Kubernetes online endpoints"):
        resolve_endpoint_instance_type("Standard_DS3_v2")


def test_get_or_create_endpoint_does_not_update_existing_endpoint() -> None:
    endpoint = SimpleNamespace(
        traffic={"stable": 100},
        compute=(
            "/subscriptions/sub/resourceGroups/rg/providers/"
            "Microsoft.MachineLearningServices/workspaces/ws/computes/lambda-0"
        ),
    )
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint

    result = get_or_create_endpoint(
        ml_client=ml_client,
        endpoint_name="ngt-sign-language-canary",
        auth_mode="key",
        compute="lambda-0",
        tags={"rollout": "online"},
    )

    assert result is endpoint
    ml_client.online_endpoints.begin_create_or_update.assert_not_called()


def test_get_or_create_endpoint_creates_missing_endpoint() -> None:
    ml_client = Mock()
    ml_client.online_endpoints.get.side_effect = ResourceNotFoundError("missing")
    created_endpoint = SimpleNamespace(traffic={})
    operation = Mock()
    operation.result.return_value = created_endpoint
    ml_client.online_endpoints.begin_create_or_update.return_value = operation

    result = get_or_create_endpoint(
        ml_client=ml_client,
        endpoint_name="ngt-sign-language-canary",
        auth_mode="key",
        compute="lambda-0",
        tags={"rollout": "online"},
    )

    assert result is created_endpoint
    ml_client.online_endpoints.begin_create_or_update.assert_called_once()
    created_request = ml_client.online_endpoints.begin_create_or_update.call_args.args[
        0
    ]
    assert isinstance(created_request, KubernetesOnlineEndpoint)
    assert created_request.compute == "lambda-0"


def test_deploy_online_endpoint_uses_kubernetes_deployment() -> None:
    endpoint = SimpleNamespace(traffic={})
    created_deployment = SimpleNamespace(name="blue")
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint
    ml_client.online_endpoints.begin_create_or_update.return_value.result.return_value = (
        endpoint
    )
    ml_client.online_deployments.begin_create_or_update.return_value.result.return_value = (
        created_deployment
    )

    with (
        patch("deploy_online_endpoint.get_ml_client", return_value=ml_client),
        patch("deploy_online_endpoint.resolve_model_version", return_value="3"),
        patch(
            "deploy_online_endpoint.resolve_endpoint_environment",
            return_value="azureml:inference:1",
        ),
        patch(
            "deploy_online_endpoint.resolve_endpoint_instance_type",
            return_value="defaultinstancetype",
        ),
        patch(
            "deploy_online_endpoint.resolve_endpoint_compute", return_value="lambda-0"
        ),
        patch("deploy_online_endpoint.print_mapping"),
    ):
        result = deploy_online_endpoint(
            endpoint_name="ngt-sign-language-online",
            deployment_name="blue",
            model_name="ngt-sign-language",
            model_version="3",
            promoted=False,
            latest=False,
            traffic_percent=100,
            instance_type=None,
            instance_count=1,
            auth_mode="key",
        )

    deployment = ml_client.online_deployments.begin_create_or_update.call_args.args[0]
    assert result is created_deployment
    assert isinstance(deployment, KubernetesOnlineDeployment)
    assert deployment.endpoint_name == "ngt-sign-language-online"
    assert deployment.instance_type == "defaultinstancetype"


@pytest.mark.parametrize("steps", ["", "10,5", "0,10", "10,101", "ten,20"])
def test_parse_canary_steps_rejects_invalid_steps(steps: str) -> None:
    with pytest.raises(typer.BadParameter):
        parse_canary_steps(steps)


def test_canary_rollout_deploys_candidate_and_shifts_traffic() -> None:
    endpoint = SimpleNamespace(traffic={"stable": 100, "canary": 0})
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint
    traffic_updates: list[dict[str, int]] = []

    def record_update(updated_endpoint: object) -> Mock:
        traffic_updates.append(dict(updated_endpoint.traffic))
        operation = Mock()
        operation.result.return_value = updated_endpoint
        return operation

    ml_client.online_endpoints.begin_create_or_update.side_effect = record_update

    with (
        patch("rollout_canary.get_ml_client", return_value=ml_client),
        patch("rollout_canary.deploy_online_endpoint") as deploy_candidate,
        patch("rollout_canary.print_mapping"),
    ):
        main(
            endpoint_name="ngt-sign-language-online",
            stable_deployment="stable",
            candidate_deployment="canary",
            model_name="ngt-sign-language",
            model_version="3",
            promoted=False,
            latest=False,
            instance_type="Standard_DS3_v2",
            instance_count=1,
            steps="10,50,100",
            wait_seconds=0,
            rollback=False,
        )

    deploy_candidate.assert_called_once_with(
        endpoint_name="ngt-sign-language-online",
        deployment_name="canary",
        model_name="ngt-sign-language",
        model_version="3",
        promoted=False,
        latest=False,
        traffic_percent=0,
        instance_type="Standard_DS3_v2",
        instance_count=1,
        auth_mode="key",
    )
    assert traffic_updates == [
        {"stable": 100, "canary": 0},
        {"stable": 90, "canary": 10},
        {"stable": 50, "canary": 50},
        {"stable": 0, "canary": 100},
    ]


def test_canary_rollback_returns_all_traffic_to_stable() -> None:
    endpoint = SimpleNamespace(traffic={"stable": 50, "canary": 50})
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint

    with (
        patch("rollout_canary.get_ml_client", return_value=ml_client),
        patch("rollout_canary.deploy_online_endpoint") as deploy_candidate,
        patch("rollout_canary.print_mapping"),
    ):
        main(
            endpoint_name="ngt-sign-language-online",
            stable_deployment="stable",
            candidate_deployment="canary",
            model_name="ngt-sign-language",
            model_version=None,
            promoted=False,
            latest=False,
            instance_type=None,
            instance_count=1,
            steps="10,25,50,100",
            wait_seconds=0,
            rollback=True,
        )

    deploy_candidate.assert_not_called()
    assert endpoint.traffic == {"stable": 100, "canary": 0}
