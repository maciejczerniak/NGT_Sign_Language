"""Azure ML sweep submission helpers for automated retraining triggers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
import math
import shlex

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import JobResourceConfiguration
from azure.ai.ml.sweep import Choice, LogUniform

from sign_language_training.azure_config import (
    get_client,
    pretrained_checkpoint_reference_or_path,
    resolve_compute_target,
    resolve_environment,
    resolve_instance_type,
)
from sign_language_training.settings import settings as training_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_TIMEOUT_SECONDS = 28800


def build_sweep_trial_prefix(
    *,
    instance_type: str,
    experiment_name: str,
    model_name: str,
) -> str:
    """Build shell setup that survives Azure nested sweep submission.

    Azure HyperDrive does not reliably preserve custom command-job environment
    variables when the sweep is nested inside a pipeline. Export critical
    training settings in the command itself and fail immediately when a GPU
    instance type does not provide CUDA.

    Args:
        instance_type: Kubernetes instance type requested by the trial.
        experiment_name: MLflow experiment name used by each trial.
        model_name: Azure ML model registry name used by the model gate.

    Returns:
        Shell command prefix ending with ``&&``.
    """
    commands = [
        "set -e",
        "export MLFLOW_ENABLED=true",
        f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(experiment_name)}",
        f"export MLFLOW_AUTOLOG={str(training_settings.mlflow_autolog).lower()}",
        (
            "export MLFLOW_LOG_ARTIFACTS="
            f"{str(training_settings.mlflow_log_artifacts).lower()}"
        ),
        f"export MODEL_REGISTRY_NAME={shlex.quote(model_name)}",
    ]
    if "gpu" in instance_type.lower():
        commands.append(
            'python -c "import torch; assert torch.cuda.is_available(), '
            "'GPU sweep trial received no CUDA device. Check the sweep node "
            "instance_type resource configuration.'\""
        )
    return " && ".join(commands) + " && "


@dataclass(frozen=True)
class SubmittedSweep:
    """Stable response object returned after a successful Azure ML sweep submission.

    Args:
        name: Azure ML sweep job name.
        experiment_name: Azure ML experiment name.
        studio_url: Azure ML Studio URL for monitoring the sweep, or ``None``.
    """

    name: str
    experiment_name: str
    studio_url: str | None


def submit_retraining_sweep(
    *,
    experiment_name: str = "sign-language-training",
    display_name: str = "NGT sign-language hyperparameter sweep",
    data_asset: str,
    ngt_raw_version: str,
    pretrained_checkpoint: str | None = None,
    epochs: int = training_settings.training_epochs,
    img_size: int = training_settings.training_img_size,
    seed: int = training_settings.training_seed,
    target_accuracy: float = training_settings.training_target_accuracy,
    expected_num_classes: int = training_settings.training_expected_num_classes,
    num_workers: int = training_settings.training_num_workers,
    f1_threshold: float = training_settings.training_f1_threshold,
    learning_rate_min: float = 1e-5,
    learning_rate_max: float = 3e-4,
    batch_sizes: list[int] | None = None,
    patience_values: list[int] | None = None,
    max_total_trials: int = 12,
    max_concurrent_trials: int = 2,
    timeout: int = DEFAULT_SWEEP_TIMEOUT_SECONDS,
    trial_timeout: int = 7200,
    trigger_reason: str | None = None,
    trigger_image_count: int | None = None,
    raw_data_manifest_hash: str | None = None,
) -> SubmittedSweep:
    """Submit the Azure ML sweep used by automatic retraining.

    Args:
        experiment_name: Azure ML experiment name for the sweep.
        display_name: Human-readable sweep display name.
        data_asset: Raw data asset reference, e.g. ``azureml:ngt-raw:6``.
        ngt_raw_version: Raw data asset version recorded in job tags.
        pretrained_checkpoint: Pretrained checkpoint asset reference or path.
        epochs: Maximum epochs per trial.
        img_size: Image size in pixels.
        seed: Random seed.
        target_accuracy: Accuracy threshold used by the model gate.
        expected_num_classes: Expected output class count.
        num_workers: DataLoader worker count.
        f1_threshold: Macro F1 threshold used by the model gate.
        learning_rate_min: Lower log-uniform learning-rate bound.
        learning_rate_max: Upper log-uniform learning-rate bound.
        batch_sizes: Batch-size choices. Defaults to ``[8, 16, 32]``.
        patience_values: Early-stopping patience choices. Defaults to
            ``[5, 7, 10]``.
        max_total_trials: Maximum trial count.
        max_concurrent_trials: Maximum concurrent trial count.
        timeout: Maximum total sweep duration in seconds.
        trial_timeout: Maximum duration per trial in seconds.
        trigger_reason: Trigger reason recorded in sweep tags.
        trigger_image_count: Raw image count recorded in sweep tags.
        raw_data_manifest_hash: Raw dataset manifest hash recorded in tags.

    Returns:
        Submitted sweep metadata.
    """
    resolved_pretrained_checkpoint = (
        pretrained_checkpoint or pretrained_checkpoint_reference_or_path()
    )
    resolved_batch_sizes = batch_sizes or [8, 16, 32]
    resolved_patience_values = patience_values or [5, 7, 10]

    ml_client = get_client()
    compute_target = resolve_compute_target(ml_client)
    environment = resolve_environment(ml_client)
    instance_type = resolve_instance_type()

    preprocess_cmd = command(
        name="preprocess",
        display_name="NGT offline augmentation",
        code=str(REPO_ROOT),
        command=(
            "pip install -e src/sign_language_training/ && "
            "python -m sign_language_training.augmentation "
            "--input-dir ${{inputs.raw_data}} "
            "--output-train-dir ${{outputs.augmented_train}} "
            "--output-val-dir ${{outputs.val_data}} "
            "--output-test-dir ${{outputs.test_data}} "
            "--copies ${{inputs.augment_copies}} "
            "--img-size ${{inputs.img_size}} "
            "--seed ${{inputs.seed}} "
            "--augmented-asset-name ngt-augmented-train "
            f"--ngt-raw-version {ngt_raw_version}"
        ),
        inputs={
            "raw_data": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "augment_copies": 4,
            "img_size": img_size,
            "seed": seed,
        },
        outputs={
            "augmented_train": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
            "val_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
            "test_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
        },
        environment=environment,
        compute=compute_target,
        resources=JobResourceConfiguration(instance_type=instance_type),
    )

    trial_command = (
        build_sweep_trial_prefix(
            instance_type=instance_type,
            experiment_name=experiment_name,
            model_name=training_settings.model_registry_name,
        )
        + "pip install -e src/sign_language_training/ && "
        "python -m sign_language_training.train "
        "--data-dir ${{inputs.data}} "
        "--val-dir ${{inputs.val_data}} "
        "--pretrained-checkpoint ${{inputs.pretrained_checkpoint}} "
        "--checkpoint-dir ${{outputs.checkpoints}} "
        "--results-dir ${{outputs.results}} "
        "--batch-size ${{inputs.batch_size}} "
        "--epochs ${{inputs.epochs}} "
        "--learning-rate ${{inputs.learning_rate}} "
        "--img-size ${{inputs.img_size}} "
        "--seed ${{inputs.seed}} "
        "--patience ${{inputs.patience}} "
        "--target-accuracy ${{inputs.target_accuracy}} "
        "--expected-num-classes ${{inputs.expected_num_classes}} "
        "--f1-threshold ${{inputs.f1_threshold}} "
        "--num-workers ${{inputs.num_workers}}"
    )

    trial_job = command(
        display_name="NGT sign-language sweep trial",
        description="One trial for automatic NGT sign-language retraining sweep.",
        experiment_name=experiment_name,
        code=str(REPO_ROOT),
        command=trial_command,
        inputs={
            "data": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "val_data": Input(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "pretrained_checkpoint": Input(
                type=AssetTypes.URI_FILE,
                path=resolved_pretrained_checkpoint,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "learning_rate": LogUniform(
                min_value=math.log(learning_rate_min),
                max_value=math.log(learning_rate_max),
            ),
            "batch_size": Choice(values=cast(Any, resolved_batch_sizes)),
            "patience": Choice(values=cast(Any, resolved_patience_values)),
            "epochs": epochs,
            "img_size": img_size,
            "seed": seed,
            "target_accuracy": target_accuracy,
            "expected_num_classes": expected_num_classes,
            "f1_threshold": f1_threshold,
            "num_workers": num_workers,
        },
        outputs={
            "checkpoints": {"type": AssetTypes.URI_FOLDER, "mode": "rw_mount"},
            "results": {"type": AssetTypes.URI_FOLDER, "mode": "rw_mount"},
        },
        environment=environment,
        compute=compute_target,
        resources=JobResourceConfiguration(instance_type=instance_type),
        environment_variables={
            "MLFLOW_ENABLED": "true",
            "MLFLOW_EXPERIMENT_NAME": experiment_name,
            "MLFLOW_AUTOLOG": str(training_settings.mlflow_autolog).lower(),
            "MLFLOW_LOG_ARTIFACTS": str(training_settings.mlflow_log_artifacts).lower(),
            "MODEL_REGISTRY_NAME": training_settings.model_registry_name,
        },
    )

    @pipeline(  # type: ignore[call-overload, untyped-decorator]
        display_name=display_name,
        experiment_name=experiment_name,
    )
    def preprocess_then_sweep(raw_data):  # type: ignore[no-untyped-def]
        """Define a pipeline that preprocesses once before sweep trials.

        Args:
            raw_data: Azure ML raw ImageFolder input.
        """
        preprocess_step: Any = preprocess_cmd(raw_data=raw_data)
        sweep_node: Any = trial_job.sweep(
            sampling_algorithm="random",
            primary_metric="best_val_accuracy",
            goal="Maximize",
            max_total_trials=max_total_trials,
            max_concurrent_trials=max_concurrent_trials,
            timeout=timeout,
            trial_timeout=trial_timeout,
        )
        sweep_node.resources = JobResourceConfiguration(
            instance_type=instance_type,
            instance_count=1,
        )
        sweep_node.inputs.data = preprocess_step.outputs.augmented_train
        sweep_node.inputs.val_data = preprocess_step.outputs.val_data

    pipeline_job = preprocess_then_sweep(
        raw_data=Input(
            type=AssetTypes.URI_FOLDER,
            path=data_asset,
            mode=InputOutputModes.RO_MOUNT,
        )
    )

    tags = {
        "project": "sign-language",
        "purpose": "retraining-sweep",
        "finalization_status": "pending",
        "raw_data_asset": data_asset,
        "raw_data_version": str(ngt_raw_version),
    }
    if trigger_reason:
        tags["trigger_reason"] = trigger_reason
    if trigger_image_count is not None:
        tags["trigger_image_count"] = str(trigger_image_count)
    if raw_data_manifest_hash:
        tags["manifest_hash"] = raw_data_manifest_hash
    if isinstance(pipeline_job, dict):
        pipeline_job["tags"] = tags
    else:
        pipeline_job.tags = tags

    created = ml_client.jobs.create_or_update(pipeline_job)
    return SubmittedSweep(
        name=str(created.name),
        experiment_name=experiment_name,
        studio_url=getattr(created, "studio_url", None),
    )
