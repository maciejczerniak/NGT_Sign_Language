"""Register the Azure ML training environment.

This uses an existing conda YAML file from the repository to build and
register a versioned Azure ML environment asset. Both CPU and GPU base
images are supported.

Usage
-----
CPU environment (default)::

    poetry run python scripts/register_env.py

GPU environment::

    poetry run python scripts/register_env.py \\
        --env-name sign-language-training-env-gpu \\
        --env-version 7 \\
        --conda-file src/sign_language_training/environments/train-env-gpu.yml \\
        --gpu
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from azure.ai.ml.entities import Environment

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import get_client, settings  # noqa: E402

DEFAULT_ENV_NAME = "sign-language-training-env"
DEFAULT_ENV_VERSION = "1"
DEFAULT_CONDA_FILE = REPO_ROOT / "conda.yaml"

CPU_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
GPU_BASE_IMAGE = (
    "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest"
)


def main(
    env_name: Optional[str] = typer.Option(
        None,
        "--env-name",
        help=(
            "Azure ML environment name. "
            "Defaults to AZURE_ENVIRONMENT_NAME or sign-language-training-env."
        ),
    ),
    env_version: Optional[str] = typer.Option(
        None,
        "--env-version",
        help="Azure ML environment version. Defaults to AZURE_ENVIRONMENT_VERSION or 1.",
    ),
    conda_file: Optional[Path] = typer.Option(
        None,
        "--conda-file",
        help="Path to the conda YAML file used to build the Azure ML environment.",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu/--cpu",
        help="Use CUDA 11.8 GPU base image instead of CPU base image.",
    ),
) -> None:
    """
    Register or update an Azure ML environment asset from a conda YAML file.

    Resolves the environment name, version, conda file, and base image from
    CLI options with fallback to project settings and built-in defaults.
    Prints a warning to stderr if no explicit version is provided, since the
    default version will be created or overwritten silently.

    Resolution order for each option:

    - ``env_name``: ``--env-name`` → ``AZURE_ENVIRONMENT_NAME`` → ``sign-language-training-env``
    - ``env_version``: ``--env-version`` → ``AZURE_ENVIRONMENT_VERSION`` → ``1``
    - ``conda_file``: ``--conda-file`` → ``conda.yaml`` in the repository root
    - base image: ``--gpu`` flag → CUDA 11.8 GPU image or CPU image

    Args:
        env_name: Azure ML environment name to register or update. Falls back
            to ``AZURE_ENVIRONMENT_NAME`` from project settings, then the default name.
        env_version: Version string to assign to the environment asset. Falls
            back to ``AZURE_ENVIRONMENT_VERSION`` from project settings, then ``"1"``.
        conda_file: Path to the conda YAML file defining the environment
            dependencies. Falls back to ``conda.yaml`` in the repository root.
        gpu: If ``True``, uses the CUDA 11.8 + cuDNN 8 GPU base image.
            If ``False`` (default), uses the CPU-only OpenMPI base image.

    Raises:
        typer.BadParameter: If the resolved conda file does not exist on disk.
    """
    ml_client = get_client()

    resolved_env_name = env_name or settings.azure_environment_name or DEFAULT_ENV_NAME
    resolved_env_version = (
        env_version or settings.azure_environment_version or DEFAULT_ENV_VERSION
    )
    resolved_conda_file = conda_file or DEFAULT_CONDA_FILE
    base_image = GPU_BASE_IMAGE if gpu else CPU_BASE_IMAGE

    if not resolved_conda_file.exists():
        raise typer.BadParameter(
            f"Conda environment file not found: {resolved_conda_file}. "
            "Create the file first or pass --conda-file <path>."
        )

    typer.echo(
        f"Registering environment '{resolved_env_name}' version {resolved_env_version}"
    )
    typer.echo(f"  conda file : {resolved_conda_file}")
    typer.echo(f"  base image : {base_image}")
    typer.echo(f"  gpu        : {gpu}")
    if env_version is None and settings.azure_environment_version is None:
        typer.echo(
            "Warning: no --env-version/AZURE_ENVIRONMENT_VERSION was provided; "
            f"version {DEFAULT_ENV_VERSION} will be created or overwritten.",
            err=True,
        )

    environment = Environment(
        name=resolved_env_name,
        version=resolved_env_version,
        description="Training environment for the NGT sign-language classifier.",
        conda_file=str(resolved_conda_file),
        image=base_image,
        tags={
            "project": "sign-language",
            "framework": "pytorch",
            "gpu": str(gpu),
        },
    )

    registered = ml_client.environments.create_or_update(environment)

    typer.echo("Registered Azure ML environment:")
    typer.echo(f"  name: {registered.name}")
    typer.echo(f"  version: {registered.version}")
    typer.echo(f"  image: {registered.image}")


if __name__ == "__main__":
    typer.run(main)
