"""Submit the NGT training workflow as an Azure ML command job.

This job consumes:

- one raw ImageFolder data asset  (azureml:ngt-raw:1)
- one pretrained checkpoint asset  (azureml:ngt-pretrained-checkpoint:1)
- CLI-overridable training hyperparameters

The package is installed inside the job before calling the training module.

Typical usage::

    poetry run python scripts/submit_training_job.py
    poetry run python scripts/submit_training_job.py \\
        --instance-type cpu-small \\
        --epochs 1 \\
        --batch-size 8 \\
        --num-workers 0 \\
        --mlflow-enabled
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration

# sign_language_training.settings is safe on Windows — no torch at module level.
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from sign_language_training.azure_config import (  # noqa: E402
    get_client,
    pretrained_checkpoint_reference_or_path,
    raw_data_asset_reference,
    resolve_compute_target,
    resolve_environment,
    resolve_instance_type,
)
from sign_language_training.settings import settings as training_settings  # noqa: E402

app = typer.Typer(
    name="submit-training-job",
    help="Submit the NGT training workflow as an Azure ML command job.",
    add_completion=False,
)


@app.command()
def main(
    job_name: Optional[str] = typer.Option(
        None,
        "--job-name",
        help="Optional Azure ML job name. Leave empty to let Azure generate one.",
    ),
    experiment_name: str = typer.Option(
        "sign-language-training",
        "--experiment-name",
        help="Azure ML experiment name.",
    ),
    display_name: str = typer.Option(
        "NGT sign-language training",
        "--display-name",
        help="Display name shown in Azure ML Studio.",
    ),
    description: str = typer.Option(
        "Train NGT classifier on Azure ML.",
        "--description",
        help="Azure ML job description.",
    ),
    data_asset: str | None = typer.Option(
        None,
        "--data-asset",
        help="Azure ML raw data asset reference, e.g. azureml:ngt-raw:1.",
    ),
    pretrained_checkpoint: str | None = typer.Option(
        None,
        "--pretrained-checkpoint",
        help="Pretrained checkpoint asset reference, e.g. azureml:ngt-pretrained-checkpoint:1",
    ),
    batch_size: int = typer.Option(
        training_settings.training_batch_size,
        "--batch-size",
        min=1,
    ),
    epochs: int = typer.Option(
        training_settings.training_epochs,
        "--epochs",
        min=1,
    ),
    learning_rate: float = typer.Option(
        training_settings.training_learning_rate,
        "--learning-rate",
        min=0.0,
    ),
    img_size: int = typer.Option(
        training_settings.training_img_size,
        "--img-size",
        min=1,
    ),
    val_split: float = typer.Option(
        training_settings.training_val_split,
        "--val-split",
        min=0.01,
        max=0.99,
    ),
    seed: int = typer.Option(
        training_settings.training_seed,
        "--seed",
    ),
    patience: int = typer.Option(
        training_settings.training_patience,
        "--patience",
        min=1,
    ),
    target_accuracy: float = typer.Option(
        training_settings.training_target_accuracy,
        "--target-accuracy",
        min=0.0,
        max=1.0,
    ),
    expected_num_classes: int = typer.Option(
        training_settings.training_expected_num_classes,
        "--expected-num-classes",
        min=1,
    ),
    num_workers: int = typer.Option(
        training_settings.training_num_workers,
        "--num-workers",
        min=0,
    ),
    f1_threshold: float = typer.Option(
        training_settings.training_f1_threshold,
        "--f1-threshold",
        min=0.0,
        max=1.0,
    ),
    mlflow_enabled: bool = typer.Option(
        True,
        "--mlflow-enabled/--no-mlflow-enabled",
        help="Enable MLflow tracking inside the Azure ML job.",
    ),
    mlflow_experiment_name: str = typer.Option(
        training_settings.mlflow_experiment_name,
        "--mlflow-experiment-name",
    ),
    mlflow_run_name: Optional[str] = typer.Option(
        None,
        "--mlflow-run-name",
        help="Optional MLflow run name.",
    ),
    mlflow_autolog: bool = typer.Option(
        training_settings.mlflow_autolog,
        "--mlflow-autolog/--no-mlflow-autolog",
    ),
    mlflow_log_artifacts: bool = typer.Option(
        training_settings.mlflow_log_artifacts,
        "--mlflow-log-artifacts/--no-mlflow-log-artifacts",
    ),
) -> None:
    """Submit one Azure ML command job for NGT sign-language model training.

    Resolves the data asset reference, pretrained checkpoint, compute target,
    environment, and instance type from CLI options with fallback to project
    settings. Builds an Azure ML command job that installs the training package
    inside the job environment and runs the training module with the configured
    hyperparameters.

    MLflow settings are injected as environment variables rather than CLI flags
    because ``sign_language_training.settings`` reads them automatically via
    pydantic-settings. Checkpoints and evaluation results are written to Azure
    ML output mounts.

    Prints the submitted job name, experiment, compute target, environment,
    instance type, and Azure ML Studio URL.

    Args:
        job_name: Optional Azure ML job name. If omitted, Azure generates
            a unique name automatically.
        experiment_name: Azure ML experiment name to group the job under.
        display_name: Human-readable display name shown in Azure ML Studio.
        description: Description text for the Azure ML job.
        data_asset: Raw data asset reference in ``azureml:<name>:<version>``
            format. Defaults to :func:`~azure_config.raw_data_asset_reference`.
        pretrained_checkpoint: Pretrained checkpoint asset reference or
            local path. Defaults to
            :func:`~azure_config.pretrained_checkpoint_reference_or_path`.
        batch_size: Number of samples per training batch.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimiser.
        img_size: Image size in pixels used for resizing during training
            (applied to both height and width).
        val_split: Fraction of the dataset to use for validation.
        seed: Random seed for deterministic splitting and training.
        patience: Number of epochs without validation improvement before
            early stopping is triggered.
        target_accuracy: Minimum validation accuracy required to pass the
            post-training gate check.
        expected_num_classes: Expected number of output classes used to
            validate the dataset before training begins.
        num_workers: Number of DataLoader worker processes inside the job.
        f1_threshold: Minimum macro F1 score required to pass the gate check.
        mlflow_enabled: If ``True``, sets ``MLFLOW_ENABLED=true`` in the
            job environment to enable MLflow tracking.
        mlflow_experiment_name: MLflow experiment name passed to the job
            via the ``MLFLOW_EXPERIMENT_NAME`` environment variable.
        mlflow_run_name: Optional MLflow run name passed via
            ``MLFLOW_RUN_NAME``. If omitted, MLflow generates a name automatically.
        mlflow_autolog: Whether to enable MLflow autologging in the job,
            passed via the ``MLFLOW_AUTOLOG`` environment variable.
        mlflow_log_artifacts: Whether to log training artifacts to MLflow,
            passed via the ``MLFLOW_LOG_ARTIFACTS`` environment variable.

    Raises:
        ValueError: If required Azure or checkpoint settings are missing
            from ``.env``.
    """
    resolved_data_asset = data_asset or raw_data_asset_reference()
    resolved_pretrained_checkpoint = (
        pretrained_checkpoint or pretrained_checkpoint_reference_or_path()
    )
    ml_client = get_client()
    compute_target = resolve_compute_target(ml_client)
    environment = resolve_environment(ml_client)
    instance_type = resolve_instance_type()

    # ── Job command ────────────────────────────────────────────────────────
    # Installs the standalone training package staged under src/sign_language_training/.
    # The root pyproject.toml is excluded by .amlignore in Azure ML jobs.
    job_command = (
        "pip install -e src/sign_language_training/ && "
        "python -m sign_language_training.train "
        "--data-dir ${{inputs.data}} "
        "--pretrained-checkpoint ${{inputs.pretrained_checkpoint}} "
        "--checkpoints-root ${{outputs.checkpoints}} "
        "--results-root ${{outputs.results}} "
        "--batch-size ${{inputs.batch_size}} "
        "--epochs ${{inputs.epochs}} "
        "--learning-rate ${{inputs.learning_rate}} "
        "--img-size ${{inputs.img_size}} "
        "--val-split ${{inputs.val_split}} "
        "--seed ${{inputs.seed}} "
        "--patience ${{inputs.patience}} "
        "--target-accuracy ${{inputs.target_accuracy}} "
        "--expected-num-classes ${{inputs.expected_num_classes}} "
        "--num-workers ${{inputs.num_workers}} "
        "--f1-threshold ${{inputs.f1_threshold}}"
    )

    job_inputs: dict[str, object] = {
        "data": Input(
            type=AssetTypes.URI_FOLDER,
            path=resolved_data_asset,
            mode=InputOutputModes.RO_MOUNT,
        ),
        "pretrained_checkpoint": Input(
            type=AssetTypes.URI_FILE,
            path=resolved_pretrained_checkpoint,
            mode=InputOutputModes.RO_MOUNT,
        ),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "img_size": img_size,
        "val_split": val_split,
        "seed": seed,
        "patience": patience,
        "target_accuracy": target_accuracy,
        "expected_num_classes": expected_num_classes,
        "num_workers": num_workers,
        "f1_threshold": f1_threshold,
    }

    job_env_vars: dict[str, str] = {
        "MLFLOW_ENABLED": str(mlflow_enabled).lower(),
        "MLFLOW_EXPERIMENT_NAME": mlflow_experiment_name,
        "MLFLOW_AUTOLOG": str(mlflow_autolog).lower(),
        "MLFLOW_LOG_ARTIFACTS": str(mlflow_log_artifacts).lower(),
    }
    if mlflow_run_name:
        job_env_vars["MLFLOW_RUN_NAME"] = mlflow_run_name

    azure_job = command(
        name=job_name,
        display_name=display_name,
        description=description,
        experiment_name=experiment_name,
        code=str(REPO_ROOT),
        command=job_command,
        inputs=job_inputs,
        environment_variables=job_env_vars,
        outputs={
            "checkpoints": {
                "type": AssetTypes.URI_FOLDER,
                "mode": "rw_mount",
            },
            "results": {
                "type": AssetTypes.URI_FOLDER,
                "mode": "rw_mount",
            },
        },
        environment=environment,
        compute=compute_target,
        resources=JobResourceConfiguration(instance_type=instance_type),
    )

    created_job = ml_client.jobs.create_or_update(azure_job)

    typer.echo("Submitted Azure ML training job:")
    typer.echo(f"  name        : {created_job.name}")
    typer.echo(f"  experiment  : {experiment_name}")
    typer.echo(f"  compute     : {compute_target}")
    typer.echo(f"  environment : {environment}")
    typer.echo(f"  instance    : {instance_type}")
    typer.echo(f"  studio_url  : {created_job.studio_url}")


if __name__ == "__main__":
    app()
