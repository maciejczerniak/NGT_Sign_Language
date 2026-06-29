"""Deploy the registered NGT model as an Azure ML Kubernetes online endpoint."""

from __future__ import annotations

from typing import Any

import typer
from azure.ai.ml.entities import (
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
    OnlineRequestSettings,
)
from azure.core.exceptions import ResourceNotFoundError

from endpoint_common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_ONLINE_ENDPOINT,
    REPO_ROOT,
    deployment_tags,
    get_ml_client,
    model_reference,
    print_mapping,
    resolve_endpoint_environment,
    resolve_endpoint_instance_type,
    resolve_endpoint_compute,
    resolve_model_version,
)


app = typer.Typer(add_completion=False)


def build_traffic_allocation(
    existing_traffic: dict[str, int],
    deployment_name: str,
    traffic_percent: int,
) -> dict[str, int]:
    """Build a traffic map while preserving existing live deployments.

    Args:
        existing_traffic: Current deployment traffic percentages.
        deployment_name: Deployment receiving the requested traffic.
        traffic_percent: Percentage assigned to the deployment.

    Returns:
        Updated deployment traffic mapping.

    Raises:
        ValueError: If the requested allocation would exceed 100 percent.
    """
    if traffic_percent == 100:
        traffic = {name: 0 for name in existing_traffic if name != deployment_name}
        traffic[deployment_name] = 100
        return traffic

    other_traffic = sum(
        percent for name, percent in existing_traffic.items() if name != deployment_name
    )
    if other_traffic + traffic_percent > 100:
        raise ValueError(
            f"Cannot assign {traffic_percent}% to '{deployment_name}': "
            f"other deployments already receive {other_traffic}%."
        )

    return {**existing_traffic, deployment_name: traffic_percent}


def get_or_create_endpoint(
    ml_client: Any,
    endpoint_name: str,
    auth_mode: str,
    compute: str,
    tags: dict[str, str],
) -> object:
    """Get an existing endpoint or create it without disturbing traffic.

    Args:
        ml_client: Azure ML workspace client.
        endpoint_name: Kubernetes online endpoint name.
        auth_mode: Endpoint authentication mode.
        compute: Attached Kubernetes compute target name.
        tags: Tags applied when creating the endpoint.

    Returns:
        Existing or newly created endpoint object.
    """
    try:
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        endpoint_compute = getattr(endpoint, "compute", None)
        endpoint_compute_name = (
            str(endpoint_compute).rstrip("/").rsplit("/", maxsplit=1)[-1]
            if endpoint_compute
            else None
        )
        if endpoint_compute_name and endpoint_compute_name != compute:
            raise ValueError(
                f"Existing endpoint '{endpoint_name}' uses compute "
                f"'{endpoint_compute}', expected '{compute}'."
            )
        if endpoint.__class__.__name__ == "ManagedOnlineEndpoint":
            raise ValueError(
                f"Existing endpoint '{endpoint_name}' is managed, not Kubernetes. "
                "Use a new endpoint name or delete and recreate it."
            )
        return endpoint
    except ResourceNotFoundError:
        endpoint = KubernetesOnlineEndpoint(
            name=endpoint_name,
            auth_mode=auth_mode,
            compute=compute,
            description="NGT sign-language online inference endpoint.",
            tags=tags,
        )
        return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def deploy_online_endpoint(
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str | None,
    promoted: bool,
    latest: bool,
    traffic_percent: int,
    instance_type: str | None,
    instance_count: int,
    auth_mode: str,
    source_revision: str | None = None,
) -> object:
    """Create or update an Azure ML Kubernetes online deployment.

    Args:
        endpoint_name: Kubernetes online endpoint name.
        deployment_name: Deployment name within the endpoint.
        model_name: Registered Azure ML model name.
        model_version: Optional explicit registered model version.
        promoted: Select the promoted model version.
        latest: Select the latest model version.
        traffic_percent: Traffic percentage assigned after deployment.
        instance_type: Optional serving instance type override.
        instance_count: Number of serving instances.
        auth_mode: Endpoint authentication mode.
        source_revision: Optional source commit included in deployment tags.

    Returns:
        Created or updated Azure ML deployment object.
    """
    ml_client = get_ml_client()
    resolved_version = resolve_model_version(
        ml_client=ml_client,
        model_name=model_name,
        model_version=model_version,
        use_promoted=promoted,
        use_latest=latest,
    )
    environment = resolve_endpoint_environment()
    resolved_instance_type = resolve_endpoint_instance_type(instance_type)
    compute = resolve_endpoint_compute()
    desired_tags = deployment_tags(
        model_name,
        resolved_version,
        deployment_name,
        source_revision=source_revision,
        environment=environment,
        instance_type=resolved_instance_type,
        instance_count=instance_count,
    )

    get_or_create_endpoint(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        auth_mode=auth_mode,
        compute=compute,
        tags=deployment_tags(model_name, resolved_version, "online"),
    )

    deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_reference(model_name, resolved_version),
        environment=environment,
        code_path=str(REPO_ROOT),
        scoring_script="deployment/online/score.py",
        instance_type=resolved_instance_type,
        instance_count=instance_count,
        app_insights_enabled=True,
        request_settings=OnlineRequestSettings(request_timeout_ms=10000),
        tags=desired_tags,
        environment_variables={
            "AZUREML_MODEL_NAME": model_name,
            "AZUREML_MODEL_VERSION": resolved_version,
        },
    )
    created_deployment = ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    existing_traffic = dict(getattr(endpoint, "traffic", {}) or {})
    endpoint.traffic = build_traffic_allocation(
        existing_traffic=existing_traffic,
        deployment_name=deployment_name,
        traffic_percent=traffic_percent,
    )

    updated_endpoint = ml_client.online_endpoints.begin_create_or_update(
        endpoint
    ).result()

    print_mapping(
        "Online endpoint deployed:",
        {
            "endpoint": endpoint_name,
            "deployment": deployment_name,
            "model": model_name,
            "model_version": resolved_version,
            "environment": environment,
            "compute": compute,
            "instance_type": resolved_instance_type,
            "instance_count": instance_count,
            "traffic": f"{traffic_percent}%",
            "scoring_uri": getattr(updated_endpoint, "scoring_uri", ""),
        },
    )
    return created_deployment


@app.command()
def main(
    endpoint_name: str = typer.Option(DEFAULT_ONLINE_ENDPOINT, "--endpoint-name"),
    deployment_name: str = typer.Option("blue", "--deployment-name"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, "--model-name"),
    model_version: str | None = typer.Option(None, "--model-version"),
    promoted: bool = typer.Option(False, "--promoted"),
    latest: bool = typer.Option(False, "--latest"),
    traffic_percent: int = typer.Option(100, "--traffic-percent", min=0, max=100),
    instance_type: str | None = typer.Option(None, "--instance-type"),
    instance_count: int = typer.Option(1, "--instance-count", min=1),
    auth_mode: str = typer.Option("key", "--auth-mode"),
    source_revision: str | None = typer.Option(None, "--source-revision"),
) -> None:
    """Deploy or update an Azure ML online endpoint from CLI options.

    Args:
        endpoint_name: Kubernetes online endpoint name.
        deployment_name: Deployment name within the endpoint.
        model_name: Registered Azure ML model name.
        model_version: Optional explicit model version.
        promoted: Select the promoted model version.
        latest: Select the latest model version.
        traffic_percent: Traffic percentage assigned after deployment.
        instance_type: Optional serving instance type override.
        instance_count: Number of serving instances.
        auth_mode: Endpoint authentication mode.
        source_revision: Optional source commit included in deployment tags.
    """
    deploy_online_endpoint(
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_name=model_name,
        model_version=model_version,
        promoted=promoted,
        latest=latest,
        traffic_percent=traffic_percent,
        instance_type=instance_type,
        instance_count=instance_count,
        auth_mode=auth_mode,
        source_revision=source_revision,
    )


if __name__ == "__main__":
    app()
