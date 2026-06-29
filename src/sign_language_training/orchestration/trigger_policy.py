"""Retraining trigger policy.

Decides whether an Azure ML retraining pipeline should be submitted based
on the configured trigger reason and the current dataset state. Supports
three trigger reasons:

- ``manual``: submits only when ``force=True``.
- ``data_change``: submits when new or removed image count meets the
  configured threshold.
- ``scheduled``: submits when the last submission is older than the
  configured interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any, Literal

from sign_language_training.azure_config import get_client
from sign_language_training.orchestration.dataset_inventory import (
    DatasetInventory,
    build_dataset_inventory,
    count_new_images,
    count_removed_images,
)
from sign_language_training.orchestration.pipeline_submitter import (
    SubmittedPipeline,
    submit_retraining_pipeline,
)
from sign_language_training.orchestration.training_state import (
    LastTrainingState,
    TrainingTriggerState,
    load_state,
    save_state,
    utc_now_iso,
)

TriggerReason = Literal["manual", "data_change", "scheduled"]
ACTIVE_JOB_STATUSES = {
    "notstarted",
    "not started",
    "preparing",
    "queued",
    "starting",
    "running",
    "finalizing",
    "provisioning",
}
COMPLETED_JOB_STATUSES = {"completed"}
RETRAINING_JOB_TAG_PURPOSE = "retraining"
RETRAINING_SWEEP_TAG_PURPOSE = "retraining-sweep"
RETRAINING_JOB_PURPOSES = {RETRAINING_JOB_TAG_PURPOSE, RETRAINING_SWEEP_TAG_PURPOSE}
RAW_DATA_IMAGE_COUNT_TAG = "image_count"
RAW_DATA_MANIFEST_HASH_TAG = "manifest_hash"
JOB_RAW_DATA_ASSET_TAG = "raw_data_asset"
JOB_RAW_DATA_VERSION_TAG = "raw_data_version"
JOB_TRIGGER_IMAGE_COUNT_TAG = "trigger_image_count"
JOB_TRIGGER_REASON_TAG = "trigger_reason"
JOB_PURPOSE_TAG = "purpose"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriggerDecision:
    """Result produced by the trigger policy evaluation.

    Args:
        should_submit: Whether a retraining pipeline was submitted.
        reason: The trigger reason that was evaluated.
        message: Human-readable summary of the policy decision.
        current_image_count: Total number of images in the dataset at
            evaluation time.
        new_image_count: Number of images added since the last recorded
            submission state.
        submitted_job_name: The Azure ML job name if a pipeline was
            submitted, otherwise ``None``.
        studio_url: The Azure ML Studio URL for the submitted job, or
            ``None`` if no job was submitted.
    """

    should_submit: bool
    reason: TriggerReason
    message: str
    current_image_count: int
    new_image_count: int
    submitted_job_name: str | None = None
    studio_url: str | None = None


@dataclass(frozen=True)
class TriggerPolicyConfig:
    """Configuration for retraining trigger policy decisions.

    Args:
        data_dir: Local ImageFolder dataset root used to build the
            current dataset inventory.
        state_path: Local path persisting the last recorded trigger state.
        raw_data_asset: Azure ML raw data asset reference string, e.g.
            ``"azureml:ngt-raw:1"``.
        raw_data_version: Version string of the raw data asset, used for
            augmentation cache lookup.
        min_new_images: Minimum number of new or removed images required
            to trigger ``data_change`` retraining.
        interval_days: Minimum days since the last submission
            required to trigger ``scheduled`` retraining.
        experiment_name: Azure ML experiment name used when submitting
            the pipeline job.
    """

    data_dir: Path
    state_path: Path
    raw_data_asset: str
    raw_data_version: str
    min_new_images: int = 100
    interval_days: int = 7
    experiment_name: str = "sign-language-training"


@dataclass(frozen=True)
class RawDataAssetSnapshot:
    """Metadata snapshot for the latest registered raw data asset.

    Args:
        name: Azure ML data asset name.
        version: Azure ML data asset version.
        reference: Asset reference string in ``azureml:<name>:<version>`` format.
        image_count: Total raw image count recorded in the asset tags, or
            ``None`` when the tag is missing.
        manifest_hash: Dataset manifest hash recorded in the asset tags, or
            ``None`` when the tag is missing.
    """

    name: str
    version: str
    reference: str
    image_count: int | None
    manifest_hash: str | None


@dataclass(frozen=True)
class RetrainingJobSnapshot:
    """Metadata snapshot for a previous Azure ML retraining job.

    Args:
        name: Azure ML job name.
        status: Azure ML job status.
        created_at: Job creation timestamp, or ``None`` when unavailable.
        raw_data_asset: Raw data asset reference recorded on the job.
        raw_data_version: Raw data asset version recorded on the job.
        trigger_image_count: Raw image count recorded at job submission time.
        studio_url: Azure ML Studio URL for the job, or ``None``.
    """

    name: str
    status: str
    created_at: datetime | None
    raw_data_asset: str | None
    raw_data_version: str | None
    trigger_image_count: int | None
    studio_url: str | None


def _parse_optional_int(value: object) -> int | None:
    """Parse an optional integer metadata value.

    Args:
        value: Metadata value from Azure ML tags or properties.

    Returns:
        Parsed integer, or ``None`` when the value is empty or invalid.
    """
    if value is None:
        return None

    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _job_created_at(job: Any) -> datetime | None:
    """Return the creation timestamp from an Azure ML SDK job object.

    Args:
        job: Azure ML SDK job object.

    Returns:
        Timezone-aware creation timestamp, or ``None`` when unavailable.
    """
    creation_context = getattr(job, "creation_context", None)
    created_at = getattr(creation_context, "created_at", None)
    if created_at is None:
        created_at = getattr(job, "creation_time", None)
    if created_at is None:
        return None
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    return created_at


def _numeric_version_key(version: object) -> tuple[int, int | str]:
    """Return a stable sort key for Azure ML asset versions.

    Args:
        version: Version value from Azure ML.

    Returns:
        Sort key that places numeric versions after non-numeric versions and
        sorts numeric versions by integer value.
    """
    version_text = str(version)
    if version_text.isdigit():
        return (1, int(version_text))
    return (0, version_text)


def raw_data_asset_snapshot(asset: Any) -> RawDataAssetSnapshot:
    """Build a raw data snapshot from an Azure ML data asset object.

    Args:
        asset: Azure ML SDK data asset object.

    Returns:
        Normalized raw data asset metadata used by the scheduled checker.
    """
    tags = getattr(asset, "tags", {}) or {}
    name = str(getattr(asset, "name"))
    version = str(getattr(asset, "version"))
    return RawDataAssetSnapshot(
        name=name,
        version=version,
        reference=f"azureml:{name}:{version}",
        image_count=_parse_optional_int(tags.get(RAW_DATA_IMAGE_COUNT_TAG)),
        manifest_hash=(
            str(tags[RAW_DATA_MANIFEST_HASH_TAG])
            if tags.get(RAW_DATA_MANIFEST_HASH_TAG)
            else None
        ),
    )


def get_latest_raw_data_snapshot(
    ml_client: Any,
    asset_name: str,
) -> RawDataAssetSnapshot:
    """Return the latest registered version of a raw data asset.

    Args:
        ml_client: Authenticated Azure ML client.
        asset_name: Azure ML data asset name, e.g. ``"ngt-raw"``.

    Returns:
        Snapshot of the latest registered raw data asset.

    Raises:
        ValueError: If no versions exist for ``asset_name``.
    """
    versions = list(ml_client.data.list(name=asset_name))
    if not versions:
        raise ValueError(f"No Azure ML data asset versions found for {asset_name!r}.")

    latest = max(versions, key=lambda asset: _numeric_version_key(asset.version))
    return raw_data_asset_snapshot(latest)


def retraining_job_snapshot(job: Any) -> RetrainingJobSnapshot:
    """Build a retraining job snapshot from an Azure ML job object.

    Args:
        job: Azure ML SDK job object.

    Returns:
        Normalized retraining job metadata used by the scheduled checker.
    """
    tags = getattr(job, "tags", {}) or {}
    return RetrainingJobSnapshot(
        name=str(getattr(job, "name", "<unknown>")),
        status=str(getattr(job, "status", "")),
        created_at=_job_created_at(job),
        raw_data_asset=(
            str(tags[JOB_RAW_DATA_ASSET_TAG])
            if tags.get(JOB_RAW_DATA_ASSET_TAG)
            else None
        ),
        raw_data_version=(
            str(tags[JOB_RAW_DATA_VERSION_TAG])
            if tags.get(JOB_RAW_DATA_VERSION_TAG)
            else None
        ),
        trigger_image_count=_parse_optional_int(tags.get(JOB_TRIGGER_IMAGE_COUNT_TAG)),
        studio_url=(
            str(getattr(job, "studio_url"))
            if getattr(job, "studio_url", None)
            else None
        ),
    )


def _is_retraining_job(job: Any, experiment_name: str) -> bool:
    """Return whether an Azure ML job belongs to this retraining flow.

    Args:
        job: Azure ML SDK job object.
        experiment_name: Expected Azure ML experiment name.

    Returns:
        ``True`` when the job belongs to the experiment and is tagged as a
        retraining job.
    """
    if getattr(job, "experiment_name", None) != experiment_name:
        return False

    tags = getattr(job, "tags", {}) or {}
    return tags.get(JOB_PURPOSE_TAG) in RETRAINING_JOB_PURPOSES


def find_latest_completed_retraining_job(
    ml_client: Any,
    experiment_name: str,
) -> RetrainingJobSnapshot | None:
    """Return the latest completed retraining job recorded in Azure ML.

    Args:
        ml_client: Authenticated Azure ML client.
        experiment_name: Azure ML experiment to inspect.

    Returns:
        Latest completed retraining job snapshot, or ``None`` if none exists.
    """
    candidates: list[RetrainingJobSnapshot] = []
    for job in ml_client.jobs.list():
        if not _is_retraining_job(job, experiment_name):
            continue

        status = str(getattr(job, "status", "")).strip().lower()
        if status not in COMPLETED_JOB_STATUSES:
            continue
        candidates.append(retraining_job_snapshot(job))

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item.created_at or datetime.min.replace(tzinfo=timezone.utc),
    )


def find_active_retraining_job(experiment_name: str) -> Any | None:
    """Return an active Azure ML retraining job for an experiment.

    Args:
        experiment_name: Azure ML experiment to inspect.

    Returns:
        First active job found, or ``None`` when none are active or Azure ML
        cannot be queried.
    """
    try:
        ml_client = get_client()
        for job in ml_client.jobs.list():
            if not _is_retraining_job(job, experiment_name):
                continue

            status = str(getattr(job, "status", "")).strip().lower()
            if status in ACTIVE_JOB_STATUSES:
                return job
    except Exception as exc:
        logger.warning("Could not query Azure ML jobs before retraining: %s", exc)
        return None

    return None


def _last_training_is_recent(
    state: TrainingTriggerState,
    interval_days: int,
) -> bool:
    """Return ``True`` if the last submitted training is within the given interval.

    Args:
        state: The current :class:`~sign_language_training.orchestration.training_state.TrainingTriggerState`.
        interval_days: Number of days defining the recency window.

    Returns:
        ``True`` if a previous submission exists and occurred less than
            ``interval_days`` ago, ``False`` otherwise.
    """
    last = state.last_submitted_training
    if last is None:
        return False

    submitted_at = datetime.fromisoformat(last.submitted_at)
    if submitted_at.tzinfo is None:
        submitted_at = submitted_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)

    return now - submitted_at < timedelta(days=interval_days)


def _submit_and_update_state(
    *,
    config: TriggerPolicyConfig,
    reason: TriggerReason,
    inventory: DatasetInventory,
    force_preprocess: bool,
    new_image_count: int,
    skip_active_job_check: bool = False,
) -> TriggerDecision:
    """Submit the retraining pipeline and persist the updated trigger state.

    On successful submission, saves the new state to disk including the job
    name, submission timestamp, dataset inventory snapshot, and trigger reason.
    On failure, returns a :class:`TriggerDecision` with ``should_submit=False``
    and the error message.

    Args:
        config: The :class:`TriggerPolicyConfig` providing data paths,
            asset references, and experiment name.
        reason: The trigger reason to record in the persisted state.
        inventory: The current :class:`~sign_language_training.orchestration.dataset_inventory.DatasetInventory`
            snapshot to persist alongside the submission.
        force_preprocess: If ``True``, passes ``force_preprocess=True`` to
            :func:`~sign_language_training.orchestration.pipeline_submitter.submit_retraining_pipeline`
            to skip cache lookup.
        new_image_count: Number of new images detected since the last
            submission, included in the returned decision.

    Returns:
        A :class:`TriggerDecision` with ``should_submit=True`` and job
            details on success, or ``should_submit=False`` with the error message
            on failure.
    """
    active_job = (
        None
        if skip_active_job_check
        else find_active_retraining_job(config.experiment_name)
    )
    if active_job is not None:
        active_name = str(getattr(active_job, "name", "<unknown>"))
        active_status = str(getattr(active_job, "status", "<unknown>"))
        active_url = getattr(active_job, "studio_url", None)
        return TriggerDecision(
            should_submit=False,
            reason=reason,
            message=(
                "Skipped retraining: Azure ML job "
                f"'{active_name}' is already {active_status}."
            ),
            current_image_count=inventory.image_count,
            new_image_count=new_image_count,
            submitted_job_name=active_name,
            studio_url=str(active_url) if active_url else None,
        )

    try:
        submitted: SubmittedPipeline = submit_retraining_pipeline(
            experiment_name=config.experiment_name,
            data_asset=config.raw_data_asset,
            ngt_raw_version=config.raw_data_version,
            force_preprocess=force_preprocess,
            mlflow_enabled=True,
            trigger_reason=reason,
            trigger_image_count=inventory.image_count,
            raw_data_manifest_hash=inventory.manifest_hash,
        )
    except Exception as exc:
        return TriggerDecision(
            should_submit=False,
            reason=reason,
            message=f"Azure ML retraining submission failed: {exc}",
            current_image_count=inventory.image_count,
            new_image_count=new_image_count,
        )

    state = TrainingTriggerState(
        last_submitted_training=LastTrainingState(
            job_name=submitted.name,
            studio_url=submitted.studio_url,
            submitted_at=utc_now_iso(),
            reason=reason,
            raw_data_asset=config.raw_data_asset,
            raw_data_version=config.raw_data_version,
            image_count=inventory.image_count,
            manifest_hash=inventory.manifest_hash,
            files=inventory.files,
        )
    )
    save_state(config.state_path, state)

    return TriggerDecision(
        should_submit=True,
        reason=reason,
        message="Submitted Azure ML retraining pipeline.",
        current_image_count=inventory.image_count,
        new_image_count=new_image_count,
        submitted_job_name=submitted.name,
        studio_url=submitted.studio_url,
    )


def evaluate_and_maybe_submit(
    *,
    reason: TriggerReason,
    config: TriggerPolicyConfig,
    force: bool = False,
) -> TriggerDecision:
    """Evaluate the trigger policy and submit retraining if the policy allows it.

    Loads the current trigger state, builds the dataset inventory, and
    evaluates the policy for the given reason:

    - ``force=True``: submits unconditionally regardless of reason.
    - ``data_change``: submits if new or removed image count meets
      ``config.min_new_images``.
    - ``scheduled``: submits if the last submission is older than
      ``config.interval_days``.
    - ``manual`` with ``force=False``: never submits.

    Args:
        reason: The trigger reason to evaluate. One of ``"manual"``,
            ``"data_change"``, or ``"scheduled"``.
        config: The :class:`TriggerPolicyConfig` controlling thresholds,
            paths, and asset references.
        force: If ``True``, bypasses all policy checks and submits
            unconditionally.

    Returns:
        A :class:`TriggerDecision` describing the outcome, including
            whether a pipeline was submitted and the current dataset counts.
    """
    state = load_state(config.state_path)
    inventory = build_dataset_inventory(config.data_dir)

    previous_files = (
        state.last_submitted_training.files
        if state.last_submitted_training is not None
        else None
    )
    new_image_count = count_new_images(inventory, previous_files)
    removed_image_count = count_removed_images(inventory, previous_files)

    if force:
        return _submit_and_update_state(
            config=config,
            reason=reason,
            inventory=inventory,
            force_preprocess=True,
            new_image_count=new_image_count,
            skip_active_job_check=True,
        )

    if reason == "data_change":
        if (
            new_image_count >= config.min_new_images
            or removed_image_count >= config.min_new_images
        ):
            return _submit_and_update_state(
                config=config,
                reason=reason,
                inventory=inventory,
                force_preprocess=False,
                new_image_count=new_image_count,
            )

        return TriggerDecision(
            should_submit=False,
            reason=reason,
            message=(
                f"Skipped retraining: {new_image_count} new images and "
                f"{removed_image_count} removed images < threshold "
                f"{config.min_new_images}."
            ),
            current_image_count=inventory.image_count,
            new_image_count=new_image_count,
        )

    if reason == "scheduled":
        if not _last_training_is_recent(state, config.interval_days):
            return _submit_and_update_state(
                config=config,
                reason=reason,
                inventory=inventory,
                force_preprocess=False,
                new_image_count=new_image_count,
            )

        return TriggerDecision(
            should_submit=False,
            reason=reason,
            message=(
                f"Skipped scheduled retraining: last submission is within "
                f"{config.interval_days} days."
            ),
            current_image_count=inventory.image_count,
            new_image_count=new_image_count,
        )

    return TriggerDecision(
        should_submit=False,
        reason=reason,
        message="Manual trigger received with force=false; no policy condition selected.",
        current_image_count=inventory.image_count,
        new_image_count=new_image_count,
    )
