"""Tests for Azure ML Kubernetes endpoint blue/green rollouts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import typer
from azure.core.exceptions import ResourceNotFoundError


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rollout_blue_green import (  # noqa: E402
    _can_reuse_running_deployment,
    _matches_desired_state,
    main,
)


def test_initial_rollout_creates_blue_kubernetes_deployment() -> None:
    ml_client = Mock()
    ml_client.online_endpoints.get.side_effect = ResourceNotFoundError("missing")

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        patch("rollout_blue_green.deploy_online_endpoint") as deploy,
        patch("rollout_blue_green.print_mapping"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version="3",
            promoted=False,
            latest=False,
            instance_type="defaultinstancetype",
            instance_count=1,
            rollback=False,
            stage_only=False,
            activate_staged=False,
            reuse_running=False,
            source_revision=None,
        )

    deploy.assert_called_once_with(
        endpoint_name="endpoint",
        deployment_name="blue",
        model_name="model",
        model_version="3",
        promoted=False,
        latest=False,
        traffic_percent=100,
        instance_type="defaultinstancetype",
        instance_count=1,
        auth_mode="key",
        source_revision=None,
    )


def test_rollout_deploys_inactive_color_then_switches_traffic() -> None:
    initial = SimpleNamespace(traffic={"blue": 100})
    refreshed = SimpleNamespace(traffic={"blue": 100, "green": 0})
    ml_client = Mock()
    ml_client.online_endpoints.get.side_effect = [initial, refreshed]
    ml_client.online_deployments.get.return_value = SimpleNamespace(tags={})

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        patch("rollout_blue_green.deploy_online_endpoint") as deploy,
        patch("rollout_blue_green.print_mapping"),
        patch("rollout_blue_green.resolve_model_version", return_value="4"),
        patch(
            "rollout_blue_green.resolve_endpoint_environment",
            return_value="azureml:env:2",
        ),
        patch("rollout_blue_green.resolve_endpoint_instance_type", return_value="gpu"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version="4",
            promoted=False,
            latest=False,
            instance_type=None,
            instance_count=1,
            rollback=False,
            stage_only=False,
            activate_staged=False,
            reuse_running=False,
            source_revision=None,
        )

    deploy.assert_called_once()
    assert deploy.call_args.kwargs["deployment_name"] == "green"
    assert deploy.call_args.kwargs["traffic_percent"] == 0
    assert refreshed.traffic == {"green": 100, "blue": 0}
    ml_client.online_endpoints.begin_create_or_update.assert_called_once_with(refreshed)


def test_rollout_skips_active_deployment_matching_desired_state() -> None:
    """Avoid redeploying when CI already deployed the requested state."""
    endpoint = SimpleNamespace(traffic={"blue": 100, "green": 0})
    active_deployment = SimpleNamespace(
        tags={
            "model_name": "model",
            "model_version": "4",
            "source_revision": "abc123",
            "environment": "azureml:env:2",
            "instance_type": "gpu",
            "instance_count": "1",
        }
    )
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint
    ml_client.online_deployments.get.return_value = active_deployment

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        patch("rollout_blue_green.deploy_online_endpoint") as deploy,
        patch("rollout_blue_green.print_mapping") as print_result,
        patch("rollout_blue_green.resolve_model_version", return_value="4"),
        patch(
            "rollout_blue_green.resolve_endpoint_environment",
            return_value="azureml:env:2",
        ),
        patch("rollout_blue_green.resolve_endpoint_instance_type", return_value="gpu"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version="4",
            promoted=False,
            latest=False,
            instance_type=None,
            instance_count=1,
            rollback=False,
            stage_only=False,
            activate_staged=False,
            reuse_running=False,
            source_revision="abc123",
        )

    deploy.assert_not_called()
    ml_client.online_endpoints.begin_create_or_update.assert_not_called()
    print_result.assert_called_once()


def test_matching_desired_state_requires_source_revision() -> None:
    """Local rollouts without a source revision should continue to deploy."""
    deployment = SimpleNamespace(tags={})

    assert not _matches_desired_state(
        deployment,
        model_name="model",
        model_version="4",
        source_revision=None,
        environment="azureml:env:2",
        instance_type="gpu",
        instance_count=1,
    )


def test_running_deployment_with_matching_instance_type_is_reusable() -> None:
    deployment = SimpleNamespace(
        provisioning_state="Succeeded",
        instance_type="gpu",
        tags={},
    )

    assert _can_reuse_running_deployment(deployment, instance_type="gpu")
    assert not _can_reuse_running_deployment(deployment, instance_type="cpu-xl")


def test_reuse_running_skips_model_rollout() -> None:
    endpoint = SimpleNamespace(traffic={"blue": 100, "green": 0})
    active_deployment = SimpleNamespace(
        provisioning_state="Succeeded",
        instance_type="gpu",
        tags={},
    )
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint
    ml_client.online_deployments.get.return_value = active_deployment

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        patch("rollout_blue_green.deploy_online_endpoint") as deploy,
        patch("rollout_blue_green.print_mapping") as print_result,
        patch("rollout_blue_green.resolve_model_version", return_value="5"),
        patch(
            "rollout_blue_green.resolve_endpoint_environment",
            return_value="azureml:env:2",
        ),
        patch("rollout_blue_green.resolve_endpoint_instance_type", return_value="gpu"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version="5",
            promoted=False,
            latest=False,
            instance_type="gpu",
            instance_count=1,
            rollback=False,
            stage_only=False,
            activate_staged=False,
            reuse_running=True,
            source_revision="new-revision",
        )

    deploy.assert_not_called()
    ml_client.online_endpoints.begin_create_or_update.assert_not_called()
    print_result.assert_called_once()


def test_missing_endpoint_cannot_be_rolled_back() -> None:
    ml_client = Mock()
    ml_client.online_endpoints.get.side_effect = ResourceNotFoundError("missing")

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        pytest.raises(typer.BadParameter, match="Cannot change traffic on missing"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version=None,
            promoted=False,
            latest=False,
            instance_type=None,
            instance_count=1,
            rollback=True,
            stage_only=False,
            activate_staged=False,
            reuse_running=False,
            source_revision=None,
        )


def test_stage_only_deploys_inactive_color_without_switching_traffic() -> None:
    endpoint = SimpleNamespace(traffic={"blue": 100})
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint
    ml_client.online_deployments.get.return_value = SimpleNamespace(tags={})

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        patch("rollout_blue_green.deploy_online_endpoint") as deploy,
        patch("rollout_blue_green.print_mapping"),
        patch("rollout_blue_green.resolve_model_version", return_value="4"),
        patch(
            "rollout_blue_green.resolve_endpoint_environment",
            return_value="azureml:env:2",
        ),
        patch("rollout_blue_green.resolve_endpoint_instance_type", return_value="gpu"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version="4",
            promoted=False,
            latest=False,
            instance_type=None,
            instance_count=1,
            rollback=False,
            stage_only=True,
            activate_staged=False,
            reuse_running=False,
            source_revision="abc123",
        )

    assert deploy.call_args.kwargs["deployment_name"] == "green"
    assert deploy.call_args.kwargs["traffic_percent"] == 0
    ml_client.online_endpoints.begin_create_or_update.assert_not_called()


def test_activate_staged_switches_traffic_without_deploying() -> None:
    endpoint = SimpleNamespace(traffic={"blue": 100, "green": 0})
    ml_client = Mock()
    ml_client.online_endpoints.get.return_value = endpoint
    ml_client.online_deployments.get.return_value = SimpleNamespace(name="green")

    with (
        patch("rollout_blue_green.get_ml_client", return_value=ml_client),
        patch("rollout_blue_green.deploy_online_endpoint") as deploy,
        patch("rollout_blue_green.print_mapping"),
    ):
        main(
            endpoint_name="endpoint",
            model_name="model",
            model_version=None,
            promoted=False,
            latest=False,
            instance_type=None,
            instance_count=1,
            rollback=False,
            stage_only=False,
            activate_staged=True,
            reuse_running=False,
            source_revision=None,
        )

    deploy.assert_not_called()
    assert endpoint.traffic == {"green": 100, "blue": 0}
    ml_client.online_endpoints.begin_create_or_update.assert_called_once_with(endpoint)
