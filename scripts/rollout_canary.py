"""Canary rollout helper for the Azure ML online endpoint."""

from __future__ import annotations

import time
from typing import Any

import typer

from deploy_online_endpoint import deploy_online_endpoint
from endpoint_common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_ONLINE_ENDPOINT,
    get_ml_client,
    print_mapping,
)


app = typer.Typer(add_completion=False)


def parse_canary_steps(steps: str) -> list[int]:
    """Parse and validate candidate traffic percentages.

    Args:
        steps: Comma-separated traffic percentages.

    Returns:
        Strictly increasing percentages between 1 and 100.

    Raises:
        typer.BadParameter: If steps are empty, invalid, or not increasing.
    """
    try:
        parsed = [int(step.strip()) for step in steps.split(",") if step.strip()]
    except ValueError as exc:
        raise typer.BadParameter(
            "Canary steps must be comma-separated whole numbers."
        ) from exc

    if not parsed:
        raise typer.BadParameter("At least one canary step is required.")
    if any(not 1 <= step <= 100 for step in parsed):
        raise typer.BadParameter("Canary steps must be between 1 and 100.")
    if any(current <= previous for previous, current in zip(parsed, parsed[1:])):
        raise typer.BadParameter("Canary steps must be strictly increasing.")
    return parsed


def set_traffic(
    ml_client: Any,
    endpoint_name: str,
    stable_deployment: str,
    candidate_deployment: str,
    candidate_percent: int,
) -> None:
    """Set stable and candidate traffic without retaining stale routes.

    Args:
        ml_client: Azure ML workspace client.
        endpoint_name: Kubernetes online endpoint name.
        stable_deployment: Existing stable deployment name.
        candidate_deployment: Candidate deployment name.
        candidate_percent: Traffic percentage assigned to the candidate.
    """
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {
        stable_deployment: 100 - candidate_percent,
        candidate_deployment: candidate_percent,
    }
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def validate_stable_deployment(
    endpoint: object,
    stable_deployment: str,
    candidate_deployment: str,
) -> None:
    """Validate that a canary starts from a live stable deployment.

    Args:
        endpoint: Azure ML endpoint object.
        stable_deployment: Existing stable deployment name.
        candidate_deployment: Candidate deployment name.

    Raises:
        typer.BadParameter: If names collide or the stable deployment is not live.
    """
    if stable_deployment == candidate_deployment:
        raise typer.BadParameter(
            "Stable and candidate deployment names must be different."
        )

    traffic = dict(getattr(endpoint, "traffic", {}) or {})
    if stable_deployment not in traffic or traffic[stable_deployment] <= 0:
        raise typer.BadParameter(
            f"Stable deployment '{stable_deployment}' is not receiving live traffic. "
            f"Current traffic: {traffic}"
        )


@app.command()
def main(
    endpoint_name: str = typer.Option(DEFAULT_ONLINE_ENDPOINT, "--endpoint-name"),
    stable_deployment: str = typer.Option("stable", "--stable-deployment"),
    candidate_deployment: str = typer.Option("canary", "--candidate-deployment"),
    model_name: str = typer.Option(DEFAULT_MODEL_NAME, "--model-name"),
    model_version: str | None = typer.Option(None, "--model-version"),
    promoted: bool = typer.Option(False, "--promoted"),
    latest: bool = typer.Option(False, "--latest"),
    instance_type: str | None = typer.Option(None, "--instance-type"),
    instance_count: int = typer.Option(1, "--instance-count", min=1),
    steps: str = typer.Option("10,25,50,100", "--steps"),
    wait_seconds: int = typer.Option(60, "--wait-seconds", min=0),
    rollback: bool = typer.Option(False, "--rollback"),
) -> None:
    """Gradually shift endpoint traffic from stable to candidate.

    Args:
        endpoint_name: Kubernetes online endpoint name.
        stable_deployment: Existing stable deployment name.
        candidate_deployment: Candidate deployment name.
        model_name: Registered model name for the candidate.
        model_version: Optional explicit model version.
        promoted: Select the promoted model version.
        latest: Select the latest model version.
        instance_type: Optional serving instance type override.
        instance_count: Number of candidate serving instances.
        steps: Comma-separated candidate traffic percentages.
        wait_seconds: Delay between traffic shifts.
        rollback: Restore all traffic to the stable deployment.
    """
    ml_client = get_ml_client()
    endpoint = ml_client.online_endpoints.get(endpoint_name)

    if stable_deployment == candidate_deployment:
        raise typer.BadParameter(
            "Stable and candidate deployment names must be different."
        )

    if rollback:
        set_traffic(
            ml_client=ml_client,
            endpoint_name=endpoint_name,
            stable_deployment=stable_deployment,
            candidate_deployment=candidate_deployment,
            candidate_percent=0,
        )
        print_mapping(
            "Canary rollback complete:",
            {
                "endpoint": endpoint_name,
                "stable": stable_deployment,
                "candidate": candidate_deployment,
                "traffic": f"{stable_deployment}=100%, {candidate_deployment}=0%",
            },
        )
        return

    parsed_steps = parse_canary_steps(steps)
    validate_stable_deployment(
        endpoint=endpoint,
        stable_deployment=stable_deployment,
        candidate_deployment=candidate_deployment,
    )

    deploy_online_endpoint(
        endpoint_name=endpoint_name,
        deployment_name=candidate_deployment,
        model_name=model_name,
        model_version=model_version,
        promoted=promoted,
        latest=latest,
        traffic_percent=0,
        instance_type=instance_type,
        instance_count=instance_count,
        auth_mode="key",
    )

    set_traffic(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        stable_deployment=stable_deployment,
        candidate_deployment=candidate_deployment,
        candidate_percent=0,
    )

    for candidate_percent in parsed_steps:
        set_traffic(
            ml_client=ml_client,
            endpoint_name=endpoint_name,
            stable_deployment=stable_deployment,
            candidate_deployment=candidate_deployment,
            candidate_percent=candidate_percent,
        )
        print_mapping(
            "Canary traffic updated:",
            {
                "endpoint": endpoint_name,
                stable_deployment: f"{100 - candidate_percent}%",
                candidate_deployment: f"{candidate_percent}%",
            },
        )
        if candidate_percent < parsed_steps[-1] and wait_seconds:
            time.sleep(wait_seconds)


if __name__ == "__main__":
    app()
