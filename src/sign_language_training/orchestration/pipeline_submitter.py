"""Reusable Azure ML pipeline submission for NGT retraining.

This module contains the actual Azure ML SDK v2 pipeline construction and
submission logic. CLI scripts and future FastAPI trigger endpoints should call
this module instead of duplicating pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import JobResourceConfiguration
from sign_language_training.azure_config import (
    get_client,
    pretrained_checkpoint_reference_or_path,
    raw_data_asset_reference,
    resolve_compute_target,
    resolve_environment,
    resolve_instance_type,
)
from sign_language_training.settings import settings as training_settings

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SubmittedPipeline:
    """Stable response object returned after a successful Azure ML pipeline submission.

    Args:
        name: The Azure ML job name assigned to the submitted pipeline.
        experiment_name: The experiment name the pipeline job was grouped under.
        studio_url: The Azure ML Studio URL for monitoring the pipeline job,
            or ``None`` if the URL could not be resolved.
    """

    name: str
    experiment_name: str
    studio_url: str | None


def find_cached_augmented_asset(
    ml_client: object,
    asset_name: str,
    ngt_raw_version: str,
) -> Optional[str]:
    """Return a cached augmented data asset reference matching the given raw version.

    The augmentation pipeline step registers the augmented training split as
    an Azure ML data asset tagged with ``ngt_raw_version``. This function
    searches all registered versions of ``asset_name`` for a version whose
    tag matches ``ngt_raw_version`` and returns its asset reference string.

    Args:
        ml_client: An authenticated Azure ML client instance with a
            ``data.list`` method.
        asset_name: The Azure ML data asset name to search, e.g.
            ``"ngt-augmented-train"``.
        ngt_raw_version: The raw data version string to match against the
            ``ngt_raw_version`` tag on registered asset versions.

    Returns:
        An asset reference string in the format
            ``azureml:<name>:<version>`` if a matching version is found,
            otherwise ``None``.
    """
    try:
        versions = list(ml_client.data.list(name=asset_name))  # type: ignore[attr-defined]
    except Exception:
        return None

    for version in versions:
        tags = getattr(version, "tags", {}) or {}
        if tags.get("ngt_raw_version") == ngt_raw_version:
            return f"azureml:{asset_name}:{version.version}"

    return None


def submit_retraining_pipeline(
    *,
    experiment_name: str = "sign-language-training",
    display_name: str = "NGT preprocess + train",
    data_asset: str | None = None,
    ngt_raw_version: str = "1",
    pretrained_checkpoint: str | None = None,
    augmented_asset_name: str = "ngt-augmented-train",
    augment_copies: int = 4,
    batch_size: int = training_settings.training_batch_size,
    epochs: int = training_settings.training_epochs,
    learning_rate: float = training_settings.training_learning_rate,
    img_size: int = training_settings.training_img_size,
    seed: int = training_settings.training_seed,
    patience: int = training_settings.training_patience,
    target_accuracy: float = training_settings.training_target_accuracy,
    expected_num_classes: int = training_settings.training_expected_num_classes,
    num_workers: int = training_settings.training_num_workers,
    f1_threshold: float = training_settings.training_f1_threshold,
    mlflow_enabled: bool = True,
    force_preprocess: bool = False,
    trigger_reason: str | None = None,
    trigger_image_count: int | None = None,
    raw_data_manifest_hash: str | None = None,
) -> SubmittedPipeline:
    """Submit the Azure ML preprocessing and training pipeline for NGT retraining.

    Constructs and submits either a full two-step pipeline (preprocess → train)
    or a training-only pipeline if a cached augmented asset exists for the
    requested raw data version. Cache reuse is currently disabled pending
    registration of both train and validation splits together.

    The full pipeline executes:

    1. **Preprocess step**: stratified split and offline augmentation of the
       raw ImageFolder dataset. Outputs augmented train, validation, and test
       splits.
    2. **Train step**: EfficientNet-B0 fine-tuning on the augmented train
       split, evaluated on the validation split, with optional MLflow logging
       and model gate check.

    Args:
        experiment_name: Azure ML experiment name to group the pipeline job.
        display_name: Human-readable display name shown in Azure ML Studio.
        data_asset: Raw data asset reference in ``azureml:<name>:<version>``
            format. Defaults to :func:`~azure_config.raw_data_asset_reference`.
        ngt_raw_version: Version string of the raw data asset, used for
            augmentation cache lookup.
        pretrained_checkpoint: Pretrained checkpoint asset reference or
            local path. Defaults to
            :func:`~azure_config.pretrained_checkpoint_reference_or_path`.
        augmented_asset_name: Azure ML data asset name for the cached
            augmented training split.
        augment_copies: Number of augmented copies per original training
            image in the preprocessing step.
        batch_size: Number of samples per training batch.
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimiser.
        img_size: Image size in pixels for preprocessing and training.
        seed: Random seed for deterministic splitting and augmentation.
        patience: Early stopping patience in epochs.
        target_accuracy: Minimum validation accuracy for the gate check.
        expected_num_classes: Expected number of output classes.
        num_workers: Number of DataLoader worker processes.
        f1_threshold: Minimum macro F1 score for the gate check.
        mlflow_enabled: If ``True``, sets ``MLFLOW_ENABLED=true`` in the
            training step environment.
        force_preprocess: If ``True``, skips cache lookup and always runs
            the full preprocessing step. Currently unused while cache reuse is
            disabled.
        trigger_reason: Trigger reason recorded as an Azure ML job tag for
            scheduled checker bookkeeping.
        trigger_image_count: Raw image count recorded as an Azure ML job tag
            for future data-change checks.
        raw_data_manifest_hash: Raw dataset manifest hash recorded as an
            Azure ML job tag when available.

    Returns:
        A :class:`SubmittedPipeline` containing the job name, experiment
            name, and Azure ML Studio URL.

    Raises:
        ValueError: If required Azure or checkpoint settings are missing.
    """
    resolved_data_asset = data_asset or raw_data_asset_reference()
    resolved_pretrained_checkpoint = (
        pretrained_checkpoint or pretrained_checkpoint_reference_or_path()
    )

    ml_client = get_client()
    compute_target = resolve_compute_target(ml_client)
    environment = resolve_environment(ml_client)
    instance_type = resolve_instance_type()

    # Cache reuse is temporarily disabled.
    # Reason: the current cache stores only the augmented train asset.
    # The validation split is not cached/registered yet, so cached mode would
    # train on cached augmented train data but validate on the full raw dataset.
    cached_asset: Optional[str] = None

    def _make_train_command(data_input_name: str = "data") -> Any:
        """Build the Azure ML training command component.

        Constructs a :func:`~azure.ai.ml.command` job that installs the
        training package and runs ``sign_language_training.train`` with
        all configured hyperparameters. The data input name is parameterised
        so the same command can be wired to either a cached asset or the
        output of the preprocessing step.

        Args:
            data_input_name: Name of the training data input binding,
                e.g. ``"data"`` for cached mode or ``"data"`` wired from the
                preprocess step output.

        Returns:
            A configured Azure ML command component ready to be
                included in a pipeline.
        """
        return command(
            name="train",
            display_name="NGT EfficientNet-B0 fine-tuning",
            code=str(REPO_ROOT),
            command=(
                "pip install -e src/sign_language_training/ && "
                "python -m sign_language_training.train "
                f"--data-dir ${{{{inputs.{data_input_name}}}}} "
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
                "--num-workers ${{inputs.num_workers}} "
                "--f1-threshold ${{inputs.f1_threshold}}"
            ),
            inputs={
                data_input_name: Input(
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
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "img_size": img_size,
                "seed": seed,
                "patience": patience,
                "target_accuracy": target_accuracy,
                "expected_num_classes": expected_num_classes,
                "num_workers": num_workers,
                "f1_threshold": f1_threshold,
            },
            outputs={
                "checkpoints": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
                "results": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount"),
            },
            environment=environment,
            compute=compute_target,
            resources=JobResourceConfiguration(instance_type=instance_type),
            environment_variables={
                "MLFLOW_ENABLED": str(mlflow_enabled).lower(),
                "MLFLOW_EXPERIMENT_NAME": experiment_name,
                "MLFLOW_AUTOLOG": str(training_settings.mlflow_autolog).lower(),
                "MLFLOW_LOG_ARTIFACTS": str(
                    training_settings.mlflow_log_artifacts
                ).lower(),
            },
        )

    if cached_asset:
        train_cmd = _make_train_command("data")

        @pipeline(  # type: ignore[call-overload, untyped-decorator]
            display_name=f"{display_name} (train only)",
            experiment_name=experiment_name,
        )
        def training_only_pipeline(raw_val: Any) -> None:
            """Define a pipeline that trains from a cached augmented asset.

            Args:
                raw_val: Azure ML input containing clean validation images.
            """
            train_cmd(
                data=Input(
                    type=AssetTypes.URI_FOLDER,
                    path=cached_asset,
                    mode=InputOutputModes.RO_MOUNT,
                ),
                val_data=raw_val,
            )

        pipeline_job = training_only_pipeline(
            raw_val=Input(
                type=AssetTypes.URI_FOLDER,
                path=resolved_data_asset,
                mode=InputOutputModes.RO_MOUNT,
            ),
        )

    else:
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
                f"--augmented-asset-name {augmented_asset_name} "
                f"--ngt-raw-version {ngt_raw_version}"
            ),
            inputs={
                "raw_data": Input(
                    type=AssetTypes.URI_FOLDER,
                    mode=InputOutputModes.RO_MOUNT,
                ),
                "augment_copies": augment_copies,
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

        train_cmd = _make_train_command("data")

        @pipeline(  # type: ignore[call-overload, untyped-decorator]
            display_name=display_name,
            experiment_name=experiment_name,
        )
        def full_pipeline(raw_data: Any) -> None:
            """Define a pipeline that preprocesses raw data before training.

            Args:
                raw_data: Azure ML input containing the raw ImageFolder dataset.
            """
            preprocess_step: Any = preprocess_cmd(raw_data=raw_data)
            train_cmd(
                data=preprocess_step.outputs.augmented_train,
                val_data=preprocess_step.outputs.val_data,
            )

        pipeline_job = full_pipeline(
            raw_data=Input(
                type=AssetTypes.URI_FOLDER,
                path=resolved_data_asset,
                mode=InputOutputModes.RO_MOUNT,
            ),
        )

    existing_tags = (
        pipeline_job.get("tags", {})
        if isinstance(pipeline_job, dict)
        else getattr(pipeline_job, "tags", {})
    )
    tags = dict(existing_tags or {})
    tags.update(
        {
            "project": "sign-language",
            "purpose": "retraining",
            "raw_data_asset": resolved_data_asset,
            "raw_data_version": str(ngt_raw_version),
        }
    )
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

    return SubmittedPipeline(
        name=str(created.name),
        experiment_name=experiment_name,
        studio_url=getattr(created, "studio_url", None),
    )
