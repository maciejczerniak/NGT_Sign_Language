"""Shared Azure ML SDK v2 helpers.

This module is used by project scripts and training orchestration code that
register assets, register environments, submit jobs, and fetch the Azure ML
MLflow tracking URI.

IMPORTANT: This file must NEVER import ``sign_language.core.settings`` because
that module imports torch at the top level, which can crash on Windows with a
CPU-only virtual environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from azure.ai.ml import MLClient
from azure.identity import (
    AzureCliCredential,
    InteractiveBrowserCredential,
    ManagedIdentityCredential,
)
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV = REPO_ROOT / ".env"


class AzureSettings(BaseSettings):
    """Minimal Azure ML and MLflow settings loaded from the project .env file.

    This class intentionally excludes any sign_language package imports to
    avoid pulling in torch on CPU-only environments. All fields map directly
    to environment variables defined in .env.
    """

    model_config = SettingsConfigDict(
        env_file=str(DOTENV),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    azure_subscription_id: str | None = Field(default=None)
    azure_resource_group: str | None = Field(default=None)
    azure_workspace: str | None = Field(default=None)
    azure_compute_target: str | None = Field(default=None)
    azure_environment_name: str | None = Field(default=None)
    azure_environment_version: str | None = Field(default=None)
    azure_inference_environment_name: str | None = Field(default=None)
    azure_inference_environment_version: str | None = Field(default=None)
    azure_raw_data_asset_name: str = Field(default="ngt-raw")
    azure_raw_data_asset_version: str = Field(default="1")
    azure_pretrained_checkpoint_asset_name: str | None = Field(default=None)
    azure_pretrained_checkpoint_asset_version: str | None = Field(default=None)
    azure_instance_type: str | None = Field(default=None)
    azure_prefer_gpu: bool = Field(default=False)
    azure_auth_mode: str = Field(default="default")
    mlflow_enabled: bool = Field(default=False)
    mlflow_tracking_uri: str | None = Field(default=None)
    mlflow_experiment_name: str = Field(default="sign-language")
    training_local_data_dir: str = "data/raw"
    model_path: str | None = Field(default=None)


settings = AzureSettings()


def require_setting(name: str, value: str | None) -> str:
    """Return a normalized required setting.

    Args:
        name: Setting name used in configuration errors.
        value: Candidate setting value.

    Returns:
        Stripped non-empty setting value.

    Raises:
        ValueError: If the setting is empty or unset.
    """
    if value and str(value).strip():
        return str(value).strip()

    env_name = name.upper()
    raise ValueError(
        f"{env_name} must be set. Add it to .env or export it before running this script."
    )


def get_credential() -> (
    AzureCliCredential | InteractiveBrowserCredential | ManagedIdentityCredential
):
    """Return an Azure credential based on ``AZURE_AUTH_MODE``.

    Returns:
        Azure CLI credential for ``default`` mode, browser credential for
        ``interactive`` mode, or managed identity credential for
        ``managed_identity`` mode.

    Raises:
        ValueError: If ``AZURE_AUTH_MODE`` is unsupported.
    """
    auth_mode = settings.azure_auth_mode.strip().lower()

    if auth_mode == "default":
        return AzureCliCredential()

    if auth_mode == "interactive":
        return InteractiveBrowserCredential()

    if auth_mode == "managed_identity":
        return ManagedIdentityCredential()

    raise ValueError(
        "Invalid AZURE_AUTH_MODE. Use 'default' (az login), "
        "'interactive' (browser), or 'managed_identity' (Azure ML jobs)."
    )


def get_client() -> MLClient:
    """Create and return an Azure ML SDK v2 client from project settings.

    Reads subscription ID, resource group, and workspace name from the
    :class:`AzureSettings` instance and authenticates using :func:`get_credential`.

    Returns:
        A configured :class:`~azure.ai.ml.MLClient` instance.

    Raises:
        ValueError: If any required Azure setting is missing.
    """
    return MLClient(
        credential=get_credential(),
        subscription_id=require_setting(
            "azure_subscription_id",
            settings.azure_subscription_id,
        ),
        resource_group_name=require_setting(
            "azure_resource_group",
            settings.azure_resource_group,
        ),
        workspace_name=require_setting(
            "azure_workspace",
            settings.azure_workspace,
        ),
    )


def get_mlflow_tracking_uri(ml_client: MLClient | None = None) -> str:
    """Return the MLflow tracking URI for the configured Azure ML workspace.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.

    Returns:
        The MLflow tracking URI string for the Azure ML workspace.

    Raises:
        RuntimeError: If the tracking URI cannot be resolved from the workspace.
    """
    client = ml_client or get_client()
    workspace_name = (
        str(getattr(client, "workspace_name"))
        if getattr(client, "workspace_name", None)
        else require_setting("azure_workspace", settings.azure_workspace)
    )
    workspace = client.workspaces.get(workspace_name)

    tracking_uri = getattr(workspace, "mlflow_tracking_uri", None)
    if not tracking_uri:
        raise RuntimeError(
            "Could not resolve Azure MLflow tracking URI from the workspace."
        )

    return str(tracking_uri)


def configure_mlflow_tracking(
    experiment_name: str | None = None,
    ml_client: MLClient | None = None,
) -> str:
    """Configure the MLflow tracking URI and active experiment.

    Resolves the tracking URI in the following priority order:

    1. ``MLFLOW_TRACKING_URI`` environment variable
    2. ``mlflow_tracking_uri`` from project settings
    3. Azure ML workspace URI via :func:`get_mlflow_tracking_uri`

    Args:
        experiment_name: The MLflow experiment name to activate. Defaults to
            ``mlflow_experiment_name`` from project settings.
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance used
            when falling back to Azure ML URI resolution. If not provided, a new
            client is created as needed.

    Returns:
        The resolved MLflow tracking URI that was set.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or settings.mlflow_tracking_uri

    if not tracking_uri:
        tracking_uri = get_mlflow_tracking_uri(ml_client)

    mlflow.set_tracking_uri(tracking_uri)

    resolved_experiment_name = experiment_name or settings.mlflow_experiment_name
    mlflow.set_experiment(resolved_experiment_name)

    return tracking_uri


def list_compute_targets(ml_client: MLClient | None = None) -> list[Any]:
    """Return all compute targets available in the Azure ML workspace.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.

    Returns:
        A list of compute target objects from the workspace.
    """
    client = ml_client or get_client()
    return list(client.compute.list())


def print_compute_targets(ml_client: MLClient | None = None) -> None:
    """Print the name, type, and provisioning state of each workspace compute target.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.
    """
    for compute_target in list_compute_targets(ml_client):
        name = getattr(compute_target, "name", "<unknown>")
        target_type = getattr(compute_target, "type", "<unknown>")
        state = getattr(compute_target, "provisioning_state", "<unknown>")
        print(f"{name}: {target_type}, provisioning_state={state}")


def resolve_compute_target(ml_client: MLClient | None = None) -> str:
    """Return the compute target name to use for job submission.

    Resolution order:

    1. ``azure_compute_target`` from project settings if set.
    2. First compute target in the workspace with a ``succeeded`` or
       ``provisioned`` provisioning state.
    3. First compute target in the workspace regardless of state.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.

    Returns:
        The resolved compute target name as a string.

    Raises:
        ValueError: If no compute targets exist in the workspace.
    """
    if settings.azure_compute_target:
        return settings.azure_compute_target

    compute_targets = list_compute_targets(ml_client)
    if not compute_targets:
        raise ValueError("No Azure ML compute targets found in the workspace.")

    for compute_target in compute_targets:
        state = str(getattr(compute_target, "provisioning_state", "")).lower()
        if state in {"succeeded", "provisioned"}:
            return str(compute_target.name)

    return str(compute_targets[0].name)


def list_environments(ml_client: MLClient | None = None) -> list[Any]:
    """Return all Azure ML environments available in the workspace.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.

    Returns:
        A list of environment objects from the workspace.
    """
    client = ml_client or get_client()
    return list(client.environments.list())


def print_environments(ml_client: MLClient | None = None) -> None:
    """Print the name and version of each Azure ML environment in the workspace.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.
    """
    for environment in list_environments(ml_client):
        name = getattr(environment, "name", "<unknown>")
        version = getattr(environment, "version", "<unknown>")
        print(f"{name}: version={version}")


def environment_reference(name: str, version: str | None = None) -> str:
    """Build an Azure ML environment reference string for use in command jobs.

    Args:
        name: The registered environment name.
        version: The environment version. If omitted, resolves to ``@latest``.

    Returns:
        A reference string in the format ``azureml:<name>:<version>``
            or ``azureml:<name>@latest``.
    """
    if version:
        return f"azureml:{name}:{version}"
    return f"azureml:{name}@latest"


def resolve_environment(ml_client: MLClient | None = None) -> str:
    """Return the Azure ML environment reference to use for job submission.

    Resolution order:

    1. ``azure_environment_name`` from project settings if set.
    2. First environment registered in the workspace.

    Args:
        ml_client: An existing :class:`~azure.ai.ml.MLClient` instance.
            If not provided, a new client is created via :func:`get_client`.

    Returns:
        An environment reference string suitable for command job configuration.

    Raises:
        ValueError: If no environments are registered in the workspace.
    """
    if settings.azure_environment_name:
        return environment_reference(
            settings.azure_environment_name,
            settings.azure_environment_version,
        )

    environments = list_environments(ml_client)
    if not environments:
        raise ValueError(
            "No Azure ML environments found. Run scripts/register_env.py first."
        )

    environment = environments[0]
    return environment_reference(
        str(environment.name),
        getattr(environment, "version", None),
    )


def raw_data_asset_reference() -> str:
    """Return the Azure ML asset reference string for the configured raw dataset.

    Returns:
        A reference string in the format ``azureml:<name>:<version>``
            using ``azure_raw_data_asset_name`` and ``azure_raw_data_asset_version``
            from project settings.
    """
    return (
        f"azureml:{settings.azure_raw_data_asset_name}:"
        f"{settings.azure_raw_data_asset_version}"
    )


def pretrained_checkpoint_reference_or_path() -> str:
    """Return the pretrained checkpoint reference or local path for job submission.

    Resolution order:

    1. Azure ML asset reference if both ``azure_pretrained_checkpoint_asset_name``
       and ``azure_pretrained_checkpoint_asset_version`` are configured.
    2. Local file path from ``model_path`` in project settings.

    Returns:
        Either an ``azureml:<name>:<version>`` reference string or a
            local file path string.

    Raises:
        ValueError: If neither an asset reference nor a local model path
            is configured.
    """
    if (
        settings.azure_pretrained_checkpoint_asset_name
        and settings.azure_pretrained_checkpoint_asset_version
    ):
        return (
            f"azureml:{settings.azure_pretrained_checkpoint_asset_name}:"
            f"{settings.azure_pretrained_checkpoint_asset_version}"
        )

    if settings.model_path:
        return settings.model_path

    raise ValueError(
        "No pretrained checkpoint configured. Set AZURE_PRETRAINED_CHECKPOINT_ASSET_NAME "
        "and AZURE_PRETRAINED_CHECKPOINT_ASSET_VERSION in .env."
    )


def resolve_instance_type() -> str:
    """Resolve the Azure ML Kubernetes instance type for job submission.

    Resolution order:

    1. ``azure_instance_type`` from project settings if set.
    2. ``gpu`` if ``azure_prefer_gpu`` is enabled.
    3. ``cpu-xl`` as the default fallback.

    Returns:
        The instance type string to use for the Azure ML job.
    """
    if settings.azure_instance_type:
        return settings.azure_instance_type

    if settings.azure_prefer_gpu:
        return "gpu"

    return "cpu-xl"
