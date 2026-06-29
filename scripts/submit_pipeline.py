"""Submit the NGT preprocessing + training pipeline to Azure ML.

This script is now only a CLI wrapper. The reusable submission logic lives in:

    sign_language_training.orchestration.pipeline_submitter

That keeps the Azure ML pipeline submission callable from scripts, tests, and
future FastAPI trigger endpoints.

Typical usage::

    poetry run python scripts/submit_pipeline.py
    poetry run python scripts/submit_pipeline.py --epochs 10 --batch-size 16
    poetry run python scripts/submit_pipeline.py --force-preprocess
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    pretrained_checkpoint_reference_or_path,
    raw_data_asset_reference,
    settings as azure_settings,
)
from sign_language_training.orchestration.pipeline_submitter import (  # noqa: E402
    submit_retraining_pipeline,
)
from sign_language_training.settings import settings as training_settings  # noqa: E402

app = typer.Typer(
    name="submit-pipeline",
    help="Submit the NGT preprocessing + training pipeline to Azure ML.",
    add_completion=False,
)


@app.command()
def main(
    experiment_name: str = typer.Option(
        "sign-language-training",
        "--experiment-name",
        help="Azure ML experiment name.",
    ),
    display_name: str = typer.Option(
        "NGT preprocess + train",
        "--display-name",
        help="Display name shown in Azure ML Studio.",
    ),
    data_asset: str | None = typer.Option(
        None,
        "--data-asset",
        help="Azure ML raw data asset reference, e.g. azureml:ngt-raw:1.",
    ),
    ngt_raw_version: str = typer.Option(
        azure_settings.azure_raw_data_asset_version,
        "--ngt-raw-version",
        help="Version of the ngt-raw asset used for augmentation cache lookup.",
    ),
    pretrained_checkpoint: str | None = typer.Option(
        None,
        "--pretrained-checkpoint",
        help="Azure ML pretrained checkpoint asset reference.",
    ),
    augmented_asset_name: str = typer.Option(
        "ngt-augmented-train",
        "--augmented-asset-name",
        help="Azure ML data asset name for cached augmented train data.",
    ),
    augment_copies: int = typer.Option(
        4,
        "--augment-copies",
        help="Number of augmented copies per original train image.",
        min=1,
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
        help="Enable MLflow tracking inside the Azure ML training step.",
    ),
    force_preprocess: bool = typer.Option(
        False,
        "--force-preprocess",
        help="Re-run preprocessing even if a cached augmented asset exists.",
    ),
) -> None:
    """Submit the two-step NGT preprocessing and training pipeline to Azure ML.

    Resolves the raw data asset reference and pretrained checkpoint from CLI
    options with fallback to project settings, then delegates to
    :func:`~sign_language_training.orchestration.pipeline_submitter.submit_retraining_pipeline`.

    The submitted pipeline runs two steps in Azure ML:

    1. **Preprocessing**: stratified split and offline augmentation of the raw
       ImageFolder dataset. Results are optionally cached as a versioned
       ``ngt-augmented-train`` data asset.
    2. **Training**: EfficientNet-B0 fine-tuning, evaluation, gate check, and
       MLflow logging.

    Prints the submitted job name, experiment name, and Azure ML Studio URL.

    Args:
        experiment_name: Azure ML experiment name under which the pipeline
            job is grouped.
        display_name: Human-readable display name shown in Azure ML Studio.
        data_asset: Raw data asset reference in the format
            ``azureml:<name>:<version>``. Defaults to the reference built from
            project settings via :func:`~azure_config.raw_data_asset_reference`.
        ngt_raw_version: Version string of the ``ngt-raw`` asset, used for
            augmentation cache lookup to avoid redundant preprocessing runs.
        pretrained_checkpoint: Pretrained checkpoint asset reference or
            local path. Defaults to the value from
            :func:`~azure_config.pretrained_checkpoint_reference_or_path`.
        augmented_asset_name: Azure ML data asset name used to cache the
            augmented training split between pipeline runs.
        augment_copies: Number of augmented copies to generate per original
            training image during the preprocessing step.
        batch_size: Number of samples per training batch.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimiser.
        img_size: Image size in pixels for resizing during preprocessing
            and training (applied to both height and width).
        seed: Random seed for deterministic splitting and augmentation.
        patience: Number of epochs without validation improvement before
            early stopping is triggered.
        target_accuracy: Minimum validation accuracy required to pass the
            post-training gate check.
        expected_num_classes: Expected number of output classes used to
            validate the dataset before training begins.
        num_workers: Number of DataLoader worker processes.
        f1_threshold: Minimum macro F1 score required to pass the gate check.
        mlflow_enabled: If ``True``, enables MLflow metric and artifact
            logging inside the Azure ML training step.
        force_preprocess: If ``True``, re-runs the preprocessing step even
            if a cached augmented asset already exists for this raw data version.

    Raises:
        ValueError: If required Azure or checkpoint settings are missing
            from ``.env``.
    """
    resolved_data_asset = data_asset or raw_data_asset_reference()
    resolved_pretrained_checkpoint = (
        pretrained_checkpoint or pretrained_checkpoint_reference_or_path()
    )

    submitted = submit_retraining_pipeline(
        experiment_name=experiment_name,
        display_name=display_name,
        data_asset=resolved_data_asset,
        ngt_raw_version=ngt_raw_version,
        pretrained_checkpoint=resolved_pretrained_checkpoint,
        augmented_asset_name=augmented_asset_name,
        augment_copies=augment_copies,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        img_size=img_size,
        seed=seed,
        patience=patience,
        target_accuracy=target_accuracy,
        expected_num_classes=expected_num_classes,
        num_workers=num_workers,
        f1_threshold=f1_threshold,
        mlflow_enabled=mlflow_enabled,
        force_preprocess=force_preprocess,
    )

    typer.echo("Submitted Azure ML pipeline:")
    typer.echo(f"  name        : {submitted.name}")
    typer.echo(f"  experiment  : {submitted.experiment_name}")
    typer.echo(f"  studio_url  : {submitted.studio_url}")


if __name__ == "__main__":
    app()
