"""Shared helpers for Azure ML endpoint deployment scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from azure.ai.ml import MLClient


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    environment_reference,
    get_client,
    require_setting,
    settings as azure_settings,
)


DEFAULT_MODEL_NAME = "ngt-sign-language"
DEFAULT_ONLINE_ENDPOINT = "ngt-sign-language-blue-green"
DEFAULT_ENVIRONMENT_NAME = "sign-language-training-env-gpu"


def get_ml_client() -> MLClient:
    """Return the configured Azure ML client.

    Returns:
        Authenticated client for the configured workspace.
    """
    return get_client()


def resolve_model_version(
    ml_client: MLClient,
    model_name: str,
    model_version: str | None,
    use_promoted: bool,
    use_latest: bool,
) -> str:
    """Resolve the model version requested by a deployment command.

    Args:
        ml_client: Azure ML workspace client.
        model_name: Registered model name.
        model_version: Optional explicit model version.
        use_promoted: Select the highest-quality promoted version.
        use_latest: Select the version carrying Azure ML's latest label.

    Returns:
        Resolved registered model version.

    Raises:
        ValueError: If selection flags are invalid or no matching model exists.
    """
    selected = [bool(model_version), use_promoted, use_latest]
    if sum(selected) != 1:
        raise ValueError(
            "Choose exactly one of --model-version, --promoted, or --latest."
        )

    if model_version:
        ml_client.models.get(name=model_name, version=model_version)
        return model_version

    versions = list(ml_client.models.list(name=model_name))
    if not versions:
        raise ValueError(f"No registered versions found for model '{model_name}'.")

    if use_promoted:
        promoted = [v for v in versions if (v.tags or {}).get("promoted") == "true"]
        if not promoted:
            raise ValueError(
                f"No promoted model found for '{model_name}'. "
                "Run scripts/promote_model.py first or pass --model-version."
            )
        best = max(
            promoted,
            key=lambda v: (
                float((v.tags or {}).get("f1_macro", 0)),
                float((v.tags or {}).get("accuracy", 0)),
            ),
        )
        return str(best.version)

    latest = ml_client.models.get(name=model_name, label="latest")
    return str(latest.version)


def model_reference(model_name: str, model_version: str) -> str:
    """Build an Azure ML model asset reference.

    Args:
        model_name: Registered model name.
        model_version: Registered model version.

    Returns:
        Versioned Azure ML model reference.
    """
    return f"azureml:{model_name}:{model_version}"


def resolve_endpoint_environment() -> str:
    """Resolve the Azure ML environment used for endpoint serving.

    Returns:
        Configured inference or training environment reference.
    """
    env_name = (
        azure_settings.azure_inference_environment_name
        or azure_settings.azure_environment_name
        or DEFAULT_ENVIRONMENT_NAME
    )
    env_version = (
        azure_settings.azure_inference_environment_version
        or azure_settings.azure_environment_version
    )
    return environment_reference(env_name, env_version)


def resolve_endpoint_instance_type(override: str | None = None) -> str:
    """Resolve the instance type used by online endpoint deployments.

    Kubernetes online deployments use instance profiles defined on the
    attached compute, such as ``gpu`` or ``cpu-xl``. Managed endpoint VM SKUs
    such as ``Standard_DS3_v2`` are not valid Kubernetes instance types.

    Args:
        override: Optional Kubernetes instance profile supplied by the caller.

    Returns:
        Configured Kubernetes instance profile, defaulting to ``gpu``.

    Raises:
        ValueError: If a managed endpoint VM SKU is supplied.
    """
    instance_type = override or azure_settings.azure_instance_type or "gpu"
    if instance_type.lower().startswith("standard_"):
        raise ValueError(
            f"'{instance_type}' is a managed online endpoint VM SKU. "
            "Kubernetes online endpoints require an attached-compute instance "
            "profile such as 'gpu' or 'cpu-xl'."
        )
    return instance_type


def resolve_endpoint_compute() -> str:
    """Resolve the attached Kubernetes compute target for online endpoints.

    Returns:
        Configured Kubernetes compute target name, for example ``"lambda-0"``.

    Raises:
        ValueError: If ``AZURE_COMPUTE_TARGET`` is not configured.
    """
    return require_setting("azure_compute_target", azure_settings.azure_compute_target)


def deployment_tags(
    model_name: str,
    model_version: str,
    rollout: str,
    *,
    source_revision: str | None = None,
    environment: str | None = None,
    instance_type: str | None = None,
    instance_count: int | None = None,
) -> dict[str, str]:
    """Build consistent tags for an Azure ML endpoint deployment.

    Args:
        model_name: Registered model name.
        model_version: Registered model version.
        rollout: Rollout or deployment identifier.
        source_revision: Optional source commit deployed with the model.
        environment: Optional Azure ML environment reference.
        instance_type: Optional Kubernetes serving instance type.
        instance_count: Optional number of serving instances.

    Returns:
        Deployment tag mapping.
    """
    tags = {
        "model_name": model_name,
        "model_version": model_version,
        "source": "epic1-cloud-deployment",
        "rollout": rollout,
    }
    optional_tags = {
        "source_revision": source_revision,
        "environment": environment,
        "instance_type": instance_type,
        "instance_count": str(instance_count) if instance_count is not None else None,
    }
    tags.update({name: value for name, value in optional_tags.items() if value})
    return tags


def print_mapping(title: str, values: dict[str, Any]) -> None:
    """Print a readable mapping for CLI output.

    Args:
        title: Heading printed before the mapping.
        values: Key-value pairs to display.
    """
    print(title)
    for key, value in values.items():
        print(f"  {key:<18}: {value}")
