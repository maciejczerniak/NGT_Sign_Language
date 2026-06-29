"""Blue/green rollout helper for an Azure ML Kubernetes online endpoint."""

from __future__ import annotations

from typing import Any

import typer
from azure.core.exceptions import ResourceNotFoundError

from deploy_online_endpoint import deploy_online_endpoint
from endpoint_common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_ONLINE_ENDPOINT,
    get_ml_client,
    print_mapping,
    resolve_endpoint_environment,
    resolve_endpoint_instance_type,
    resolve_model_version,
)


app = typer.Typer(add_completion=False)


def _inactive(active: str) -> str:
    """Return the inactive blue/green deployment name.

    Args:
        active: Currently active deployment name.

    Returns:
        ``green`` when blue is active, otherwise ``blue``.
    """
    return "green" if active == "blue" else "blue"


def _active_deployment(endpoint: object) -> str:
    """Resolve the deployment receiving the largest traffic percentage.

    Args:
        endpoint: Azure ML online endpoint entity.

    Returns:
        Active deployment name, defaulting to ``blue`` when traffic is empty.
    """
    traffic = dict(getattr(endpoint, "traffic", {}) or {})
    if not traffic:
        return "blue"
    return str(max(traffic, key=lambda name: traffic[name]))


def _switch_traffic(ml_client: Any, endpoint: Any, active: str, target: str) -> None:
    """Switch all endpoint traffic from the active deployment to the target.

    Args:
        ml_client: Authenticated Azure ML client.
        endpoint: Azure ML online endpoint entity.
        active: Deployment currently receiving production traffic.
        target: Deployment that should receive production traffic.
    """
    endpoint.traffic = {target: 100, active: 0}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def _matches_desired_state(
    deployment: object,
    *,
    model_name: str,
    model_version: str,
    source_revision: str | None,
    environment: str,
    instance_type: str,
    instance_count: int,
) -> bool:
    """Return whether an active deployment already matches the requested state.

    Args:
        deployment: Existing Azure ML online deployment.
        model_name: Requested registered model name.
        model_version: Requested registered model version.
        source_revision: Requested source commit.
        environment: Requested Azure ML environment reference.
        instance_type: Requested Kubernetes instance type.
        instance_count: Requested serving instance count.

    Returns:
        ``True`` when all deployment identity tags match.
    """
    if not source_revision:
        return False
    tags = dict(getattr(deployment, "tags", {}) or {})
    expected = {
        "model_name": model_name,
        "model_version": model_version,
        "source_revision": source_revision,
        "environment": environment,
        "instance_type": instance_type,
        "instance_count": str(instance_count),
    }
    return all(tags.get(name) == value for name, value in expected.items())


def _can_reuse_running_deployment(
    deployment: object,
    *,
    instance_type: str,
) -> bool:
    """Return whether a running deployment can be reused without a rollout.

    Args:
        deployment: Active Azure ML online deployment.
        instance_type: Requested Kubernetes instance type.

    Returns:
        ``True`` when the deployment is successfully provisioned and uses the
        requested instance type.
    """
    provisioning_state = (
        str(getattr(deployment, "provisioning_state", "")).strip().lower()
    )
    deployed_instance_type = getattr(deployment, "instance_type", None)
    if not deployed_instance_type:
        deployed_instance_type = dict(getattr(deployment, "tags", {}) or {}).get(
            "instance_type"
        )
    return (
        provisioning_state == "succeeded"
        and str(deployed_instance_type) == instance_type
    )


@app.command()
def main(
    endpoint_name: str = typer.Option(DEFAULT_ONLINE_ENDPOINT, "--endpoint-name"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, "--model-name"),
    model_version: str | None = typer.Option(None, "--model-version"),
    promoted: bool = typer.Option(False, "--promoted"),
    latest: bool = typer.Option(False, "--latest"),
    instance_type: str | None = typer.Option(None, "--instance-type"),
    instance_count: int = typer.Option(1, "--instance-count", min=1),
    rollback: bool = typer.Option(False, "--rollback"),
    stage_only: bool = typer.Option(
        False,
        "--stage-only",
        help="Deploy the inactive color at 0% traffic without activating it.",
    ),
    activate_staged: bool = typer.Option(
        False,
        "--activate-staged",
        help="Switch traffic to the already staged inactive color.",
    ),
    reuse_running: bool = typer.Option(
        False,
        "--reuse-running",
        help=(
            "Reuse a successfully provisioned active deployment when its "
            "Kubernetes instance type matches."
        ),
    ),
    source_revision: str | None = typer.Option(None, "--source-revision"),
) -> None:
    """Deploy the inactive color and switch all endpoint traffic.

    The first rollout initializes a missing endpoint with the ``blue``
    deployment. Later rollouts alternate between ``blue`` and ``green``.

    Args:
        endpoint_name: Kubernetes online endpoint name.
        model_name: Registered model name.
        model_version: Optional explicit model version.
        promoted: Select the promoted model version.
        latest: Select the latest model version.
        instance_type: Optional serving instance type override.
        instance_count: Number of serving instances.
        rollback: Switch traffic back to the inactive color.
        stage_only: Deploy the inactive color without switching traffic.
        activate_staged: Switch traffic to the already staged inactive color.
        reuse_running: Skip rollout when the active deployment is healthy and
            uses the requested Kubernetes instance type.
        source_revision: Optional source commit used for idempotent CI rollouts.
    """
    selected_actions = sum((rollback, stage_only, activate_staged))
    if selected_actions > 1:
        raise typer.BadParameter(
            "Choose at most one of --rollback, --stage-only, or --activate-staged."
        )

    ml_client = get_ml_client()
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
    except ResourceNotFoundError:
        if rollback or activate_staged:
            raise typer.BadParameter(
                f"Cannot change traffic on missing endpoint '{endpoint_name}'."
            ) from None
        deploy_online_endpoint(
            endpoint_name=endpoint_name,
            deployment_name="blue",
            model_name=model_name,
            model_version=model_version,
            promoted=promoted,
            latest=latest,
            traffic_percent=100,
            instance_type=instance_type,
            instance_count=instance_count,
            auth_mode="key",
            source_revision=source_revision,
        )
        print_mapping(
            "Blue/green endpoint initialized:",
            {"endpoint": endpoint_name, "active": "blue"},
        )
        return

    active = _active_deployment(endpoint)
    target = _inactive(active)

    if rollback:
        _switch_traffic(ml_client, endpoint, active, target)
        print_mapping(
            "Blue/green rollback complete:",
            {"endpoint": endpoint_name, "active": target, "previous": active},
        )
        return

    if activate_staged:
        try:
            ml_client.online_deployments.get(
                name=target,
                endpoint_name=endpoint_name,
            )
        except ResourceNotFoundError:
            raise typer.BadParameter(
                f"Cannot activate missing staged deployment '{target}'."
            ) from None
        _switch_traffic(ml_client, endpoint, active, target)
        print_mapping(
            "Blue/green staged deployment activated:",
            {"endpoint": endpoint_name, "active": target, "previous": active},
        )
        return

    resolved_version = resolve_model_version(
        ml_client=ml_client,
        model_name=model_name,
        model_version=model_version,
        use_promoted=promoted,
        use_latest=latest,
    )
    resolved_environment = resolve_endpoint_environment()
    resolved_instance_type = resolve_endpoint_instance_type(instance_type)
    active_deployment = ml_client.online_deployments.get(
        name=active,
        endpoint_name=endpoint_name,
    )
    if reuse_running and _can_reuse_running_deployment(
        active_deployment,
        instance_type=resolved_instance_type,
    ):
        print_mapping(
            "Blue/green rollout skipped; running endpoint is reusable:",
            {
                "endpoint": endpoint_name,
                "active": active,
                "instance_type": resolved_instance_type,
            },
        )
        return

    if _matches_desired_state(
        active_deployment,
        model_name=model_name,
        model_version=resolved_version,
        source_revision=source_revision,
        environment=resolved_environment,
        instance_type=resolved_instance_type,
        instance_count=instance_count,
    ):
        print_mapping(
            "Blue/green rollout skipped; active deployment is current:",
            {"endpoint": endpoint_name, "active": active},
        )
        return

    deploy_online_endpoint(
        endpoint_name=endpoint_name,
        deployment_name=target,
        model_name=model_name,
        model_version=resolved_version,
        promoted=False,
        latest=False,
        traffic_percent=0,
        instance_type=instance_type,
        instance_count=instance_count,
        auth_mode="key",
        source_revision=source_revision,
    )

    if stage_only:
        print_mapping(
            "Blue/green deployment staged:",
            {"endpoint": endpoint_name, "staged": target, "active": active},
        )
        return

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    _switch_traffic(ml_client, endpoint, active, target)

    print_mapping(
        "Blue/green rollout complete:",
        {"endpoint": endpoint_name, "active": target, "previous": active},
    )


if __name__ == "__main__":
    app()
