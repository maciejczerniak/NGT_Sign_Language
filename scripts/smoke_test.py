"""Submit a small Azure ML smoke test job.

This validates that:

- Azure ML client connection works
- compute/environment resolution works
- selected instance type is accepted
- Python dependencies import correctly
- the project package installs correctly
- the sign-language-training CLI is available
- MLflow can start and log a small run

Typical usage::

    poetry run python scripts/smoke_test.py
    poetry run python scripts/smoke_test.py --instance-type cpu-small
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import typer
from azure.ai.ml import command

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    get_client,
    print_compute_targets,
    print_environments,
    resolve_compute_target,
    resolve_environment,
    resolve_instance_type,
)

app = typer.Typer(
    name="smoke-test",
    help="Submit a small Azure ML smoke test job.",
    add_completion=False,
)


@app.command()
def main(
    instance_type: Optional[str] = typer.Option(
        None,
        "--instance-type",
        help="Azure ML Kubernetes instance type, e.g. cpu-small, cpu-xl, gpu.",
    ),
) -> None:
    """Submit a smoke test job to Azure ML to validate the end-to-end setup.

    Resolves the compute target, environment, and instance type from project
    settings, then submits a single-instance Azure ML command job that:

    - installs the project package in the Azure ML environment
    - imports ``torch``, ``mlflow``, and ``sign_language`` and prints versions
    - logs a test param and metric via MLflow
    - runs ``sign-language --help`` and ``sign-language train --help`` to
      confirm the CLI is available and functional

    Prints available compute targets and environments to stdout before
    submission, then prints the Azure ML Studio URL for monitoring the job.

    Args:
        instance_type: Azure ML Kubernetes instance type override, e.g.
            ``cpu-small``, ``cpu-xl``, or ``gpu``. If not provided, falls back to
            :func:`~azure_config.resolve_instance_type` which checks project
            settings and the ``AZURE_PREFER_GPU`` flag.

    Raises:
        ValueError: If required Azure settings are missing from ``.env``.
    """
    ml_client = get_client()
    compute_target = resolve_compute_target(ml_client)
    environment = resolve_environment(ml_client)
    resolved_instance_type = instance_type or resolve_instance_type()

    typer.echo("Available compute targets:")
    print_compute_targets(ml_client)
    typer.echo(f"Using compute target: {compute_target}")

    typer.echo("Available environments:")
    print_environments(ml_client)
    typer.echo(f"Using environment: {environment}")
    typer.echo(f"Using instance type: {resolved_instance_type}")

    smoke_command = """
set -e
python -m pip install -e src/sign_language_training/ --no-deps
python -c "
import torch
import mlflow
import sign_language_training

print('Environment imports are stable.')
print(f'torch version: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'cuda device count: {torch.cuda.device_count()}')

mlflow.set_experiment('azure-smoke-test')
with mlflow.start_run(run_name='smoke-test'):
    mlflow.log_param('smoke_test', True)
    mlflow.log_metric('status', 1.0)

print('MLflow logging works.')
"
sign-language-training --help
python -m sign_language_training.train --help
"""

    job = command(
        code=str(REPO_ROOT),
        command=smoke_command,
        environment=environment,
        display_name="smoke-test-validation",
        experiment_name="azure-smoke-test",
        compute=compute_target,
        resources={
            "instance_type": resolved_instance_type,
            "instance_count": 1,
        },
    )

    returned_job = ml_client.jobs.create_or_update(job)

    typer.echo("Smoke test job submitted.")
    typer.echo(f"Monitor progress here: {returned_job.studio_url}")


if __name__ == "__main__":
    app()
