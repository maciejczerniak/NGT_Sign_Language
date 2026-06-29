"""On-prem / local retraining trigger policy backed by MLflow.

This module is the MLflow analogue of
:mod:`sign_language_training.orchestration.trigger_policy`, which targets the
Azure ML registry. The two mirror each other: both decide whether to retrain
using a *data-change first, scheduled-interval fallback* policy. They differ
only in the source of truth:

- Azure (:mod:`trigger_policy` / ``scripts/check_training_triggers.py``):
  the latest ``ngt-raw`` data asset ``image_count`` tag vs. the last completed
  retraining job, plus the job ``created_at`` for the interval.
- On-prem (this module): a local ImageFolder
  :class:`~sign_language_training.orchestration.dataset_inventory.DatasetInventory`
  for the current image count, the
  :class:`~sign_language_training.orchestration.training_state.TrainingTriggerState`
  JSON for the last triggered count, and the MLflow registry for the
  "last training time" used by the interval check.

Why MLflow drives the interval (not the local state JSON):
    A triggered run registers a new ``@candidate`` version and never touches
    ``@champion`` (promotion is a separate, manual step). The interval check
    therefore keys off the *most recent registered version* of the model
    (champion or candidate) — any version means "we trained recently", so the
    scheduled clock resets even before promotion. This survives loss of the
    local state file and matches the stateless spirit of the Azure checker.

The local state JSON is still used for the *data-change* count, because that
needs a per-file manifest diff (``count_new_images`` / ``count_removed_images``)
which MLflow does not store. The two sources are complementary, not redundant.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Literal, Protocol

from sign_language_training.orchestration.dataset_inventory import (
    DatasetInventory,
    build_dataset_inventory,
    count_new_images,
    count_removed_images,
)
from sign_language_training.orchestration.training_state import (
    LastTrainingState,
    TrainingTriggerState,
    load_state,
    save_state,
    utc_now_iso,
)

TriggerReason = Literal["manual", "data_change", "scheduled"]

logger = logging.getLogger(__name__)


class TrainingRunner(Protocol):
    """Callable that runs one local/on-prem training pipeline.

    Implementations must run preprocessing + training + the model gate and,
    on a passing gate, register the result as ``@candidate`` in MLflow. The
    return value is the new model version string when a candidate was
    registered, or ``None`` when the gate failed or MLflow was unavailable.
    """

    def __call__(self) -> str | None:
        """Run the pipeline and return the new candidate version, if any."""
        ...


@dataclass(frozen=True)
class MlflowTriggerConfig:
    """Configuration for on-prem MLflow-backed trigger decisions.

    Args:
        data_dir: Local ImageFolder dataset root used to build the current
            dataset inventory (the on-prem equivalent of the Azure raw data
            asset).
        state_path: Path to the JSON file persisting the last triggered
            training state (image count + per-file manifest).
        model_name: MLflow registered model name whose latest version
            timestamp drives the scheduled-interval check, e.g.
            ``"ngt-sign-language"``.
        min_new_images: Minimum number of new or removed images required to
            trigger ``data_change`` retraining.
        interval_days: Minimum days since the latest registered model version
            required to trigger ``scheduled`` retraining.
    """

    data_dir: Path
    state_path: Path
    model_name: str = "ngt-sign-language"
    min_new_images: int = 10
    interval_days: int = 7


@dataclass(frozen=True)
class MlflowTriggerDecision:
    """Result of an MLflow-backed trigger policy evaluation.

    Args:
        should_train: Whether a training run was started.
        reason: The trigger reason that fired, or ``None`` when skipped.
        message: Human-readable summary of the decision.
        current_image_count: Total images in the local dataset at evaluation.
        new_image_count: Images added since the last triggered state.
        removed_image_count: Images removed since the last triggered state.
        last_trained_at: Timestamp of the latest registered MLflow version, or
            ``None`` if the model has no versions yet.
        candidate_version: New ``@candidate`` version registered by the run, or
            ``None`` if no run started or the gate failed.
    """

    should_train: bool
    reason: TriggerReason | None
    message: str
    current_image_count: int
    new_image_count: int
    removed_image_count: int
    last_trained_at: datetime | None = None
    candidate_version: str | None = None


def latest_registered_version_timestamp(
    model_name: str,
    tracking_uri: str | None = None,
) -> datetime | None:
    """Return the creation time of the most recent registered model version.

    Considers *all* versions (champion, candidate, or unaliased), because any
    registered version means a training run completed recently — which is the
    signal the scheduled-interval check needs. A triggered run only sets
    ``@candidate``, so keying off ``@champion`` alone would never reset the
    interval clock until a human promoted.

    The ``mlflow`` import is intentionally inside the function so this module
    imports cleanly in environments without the optional ``training`` extras
    installed, and so tests can patch the import site.

    Args:
        model_name: MLflow registered model name.
        tracking_uri: MLflow tracking URI. Falls back to the
            ``MLFLOW_TRACKING_URI`` environment variable when ``None``.

    Returns:
        Timezone-aware creation timestamp of the newest version, or ``None``
        when MLflow is unavailable, the URI is unset, or the model has no
        versions.
    """
    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.warning(
            "MLflow not installed — cannot read last training time. "
            "Install with: poetry install --with training"
        )
        return None

    import os

    uri = (tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "")).strip()
    if not uri:
        logger.warning("MLFLOW_TRACKING_URI not set — cannot read last training time.")
        return None

    client = MlflowClient(tracking_uri=uri)
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception as exc:  # noqa: BLE001 - registry may not exist yet
        logger.warning("Could not query MLflow model versions: %s", exc)
        return None

    timestamps: list[int] = []
    for version in versions:
        raw_ts = getattr(version, "creation_timestamp", None)
        if raw_ts is not None:
            timestamps.append(int(raw_ts))
    if not timestamps:
        return None

    # MLflow stores creation_timestamp as epoch milliseconds.
    newest_ms = max(timestamps)
    return datetime.fromtimestamp(newest_ms / 1000.0, tz=timezone.utc)


def _interval_due(last_trained_at: datetime | None, interval_days: int) -> bool:
    """Return whether the scheduled-interval fallback is due.

    Args:
        last_trained_at: Timestamp of the latest registered version, or
            ``None`` when the model has never been trained.
        interval_days: Minimum days between scheduled runs.

    Returns:
        ``True`` if no version exists yet or the newest version is at least
        ``interval_days`` old.
    """
    if last_trained_at is None:
        return True

    return datetime.now(timezone.utc) - last_trained_at >= timedelta(days=interval_days)


def _persist_state(config: MlflowTriggerConfig, inventory: DatasetInventory) -> None:
    """Record the triggered dataset snapshot so future data-change diffs work.

    Mirrors the Azure path's state write: after a run is started we snapshot
    the current image manifest so the *next* evaluation can count only images
    added since this run. The MLflow-specific fields (job name / studio URL)
    are filled with on-prem-appropriate placeholders.

    Args:
        config: Active trigger configuration.
        inventory: The dataset inventory captured at trigger time.
    """
    state = TrainingTriggerState(
        last_submitted_training=LastTrainingState(
            job_name=f"local-pipeline-{utc_now_iso()}",
            studio_url=None,
            submitted_at=utc_now_iso(),
            reason="scheduled",
            raw_data_asset=str(config.data_dir),
            raw_data_version="local",
            image_count=inventory.image_count,
            manifest_hash=inventory.manifest_hash,
            files=inventory.files,
        )
    )
    save_state(config.state_path, state)


def build_decision(
    *,
    config: MlflowTriggerConfig,
    force: bool = False,
    tracking_uri: str | None = None,
) -> MlflowTriggerDecision:
    """Compute the trigger decision *without* running training.

    This is the pure policy core shared by every caller: the supercronic
    checker (via :func:`evaluate_and_maybe_train`), the Airflow ``decide``
    branch task, and any future scheduler. It builds the dataset inventory,
    diffs against the persisted manifest, reads the newest MLflow registered
    version timestamp, and applies the data-change-first / interval-fallback
    rule. It does not invoke a runner and does not persist state.

    Args:
        config: Thresholds, paths, and model name.
        force: Bypass all policy checks (reason becomes ``"manual"``).
        tracking_uri: Optional explicit MLflow tracking URI for the interval
            check; falls back to ``MLFLOW_TRACKING_URI``.

    Returns:
        A :class:`MlflowTriggerDecision`. ``should_train`` reflects the policy
        outcome; ``candidate_version`` is always ``None`` here (no run yet).
    """
    state = load_state(config.state_path)
    inventory = build_dataset_inventory(config.data_dir)

    previous_files = (
        state.last_submitted_training.files
        if state.last_submitted_training is not None
        else None
    )
    new_count = count_new_images(inventory, previous_files)
    removed_count = count_removed_images(inventory, previous_files)
    last_trained_at = latest_registered_version_timestamp(
        config.model_name, tracking_uri
    )

    reason: TriggerReason | None
    if force:
        reason = "manual"
    elif new_count >= config.min_new_images or removed_count >= config.min_new_images:
        reason = "data_change"
    elif _interval_due(last_trained_at, config.interval_days):
        reason = "scheduled"
    else:
        reason = None

    if reason is None:
        message = (
            f"Skipped retraining: {new_count} new / {removed_count} removed "
            f"images < threshold {config.min_new_images}, and last training "
            f"is within {config.interval_days} days."
        )
    else:
        message = (
            f"Trigger fired (reason={reason}): {new_count} new / "
            f"{removed_count} removed images; current total "
            f"{inventory.image_count}."
        )

    return MlflowTriggerDecision(
        should_train=reason is not None,
        reason=reason,
        message=message,
        current_image_count=inventory.image_count,
        new_image_count=new_count,
        removed_image_count=removed_count,
        last_trained_at=last_trained_at,
    )


def evaluate_and_maybe_train(
    *,
    config: MlflowTriggerConfig,
    runner: TrainingRunner,
    force: bool = False,
    tracking_uri: str | None = None,
) -> MlflowTriggerDecision:
    """Evaluate the on-prem trigger policy and run training if it allows it.

    Evaluation order mirrors the Azure scheduled checker exactly:

    1. ``force=True`` → train unconditionally (reason ``"manual"``).
    2. ``data_change`` → train if new-or-removed image count meets
       ``config.min_new_images``.
    3. ``scheduled`` → otherwise train if the newest MLflow version is older
       than ``config.interval_days`` (or no version exists yet).

    The decision is delegated to :func:`build_decision` so this path and the
    Airflow DAG share identical policy logic. On a decision to train,
    ``runner`` is invoked and the state file is updated with the current
    dataset snapshot *only after* the runner is invoked, so a runner crash does
    not silently advance the data-change baseline.

    Args:
        config: Thresholds, paths, and model name.
        runner: Callable that runs the local pipeline and registers a
            ``@candidate`` on a passing gate.
        force: Bypass all policy checks and train unconditionally.
        tracking_uri: Optional explicit MLflow tracking URI for the interval
            check; falls back to ``MLFLOW_TRACKING_URI``.

    Returns:
        A :class:`MlflowTriggerDecision` describing the outcome.
    """
    decision = build_decision(config=config, force=force, tracking_uri=tracking_uri)

    if not decision.should_train:
        return decision

    logger.info("Trigger fired (reason=%s) — starting local pipeline.", decision.reason)
    candidate_version = runner()
    inventory = build_dataset_inventory(config.data_dir)
    _persist_state(config, inventory)

    if candidate_version is None:
        message = (
            f"Ran local pipeline (reason={decision.reason}) but no @candidate "
            "was registered (gate failed or MLflow unavailable)."
        )
    else:
        message = (
            f"Ran local pipeline (reason={decision.reason}); registered "
            f"{config.model_name} v{candidate_version} as @candidate. "
            "Promote to @champion when ready."
        )

    return MlflowTriggerDecision(
        should_train=True,
        reason=decision.reason,
        message=message,
        current_image_count=inventory.image_count,
        new_image_count=decision.new_image_count,
        removed_image_count=decision.removed_image_count,
        last_trained_at=decision.last_trained_at,
        candidate_version=candidate_version,
    )
