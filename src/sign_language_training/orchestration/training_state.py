"""Persistent trigger state for retraining decisions.

Stores and loads the last submitted training job state as a JSON file.
This allows the trigger policy to compare the current dataset against
the state at the time of the last submission.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LastTrainingState:
    """State captured immediately after a retraining pipeline job is submitted.

    Args:
        job_name: The Azure ML job name assigned to the submitted pipeline.
        studio_url: The Azure ML Studio URL for the job, or ``None`` if
            unavailable.
        submitted_at: ISO 8601 UTC timestamp of the submission.
        reason: The trigger reason that caused the submission, e.g.
            ``"manual"``, ``"data_change"``, or ``"scheduled"``.
        raw_data_asset: The Azure ML raw data asset reference used, e.g.
            ``"azureml:ngt-raw:1"``.
        raw_data_version: The version string of the raw data asset.
        image_count: Total number of images in the dataset at submission
            time.
        manifest_hash: SHA-256 hex digest of the dataset manifest at
            submission time, used for change detection.
        files: Sorted list of relative image file paths present at
            submission time.
    """

    job_name: str
    studio_url: str | None
    submitted_at: str
    reason: str
    raw_data_asset: str
    raw_data_version: str
    image_count: int
    manifest_hash: str
    files: list[str]


@dataclass(frozen=True)
class TrainingTriggerState:
    """Full trigger state persisted to disk as a JSON file.

    Args:
        last_submitted_training: The state from the most recently
            submitted training job, or ``None`` if no job has been submitted yet.
    """

    last_submitted_training: LastTrainingState | None = None


def _state_from_json(raw: dict[str, Any]) -> TrainingTriggerState:
    """Deserialize a trigger state payload.

    Args:
        raw: Parsed JSON dictionary.

    Returns:
        Deserialized trigger state.
    """
    last_raw = raw.get("last_submitted_training")
    last = LastTrainingState(**last_raw) if last_raw else None
    return TrainingTriggerState(last_submitted_training=last)


def _state_to_json(state: TrainingTriggerState) -> str:
    """Serialize trigger state to formatted JSON.

    Args:
        state: Trigger state to serialize.

    Returns:
        JSON document string ending with a newline.
    """
    payload = {
        "last_submitted_training": (
            asdict(state.last_submitted_training)
            if state.last_submitted_training is not None
            else None
        )
    }
    return json.dumps(payload, indent=2) + "\n"


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO 8601 string.

    Returns:
        Current UTC datetime formatted as an ISO 8601 string with
            timezone offset, e.g. ``"2026-01-15T10:30:00+00:00"``.
    """
    return datetime.now(timezone.utc).isoformat()


def load_state(path: Path) -> TrainingTriggerState:
    """Load the trigger state from a JSON file on disk.

    Returns an empty :class:`TrainingTriggerState` if the file does not
    exist or cannot be parsed, so the caller always receives a valid object.

    Args:
        path: Path to the JSON trigger state file.

    Returns:
        The deserialized :class:`TrainingTriggerState`, or a fresh
            instance with ``last_submitted_training=None`` if the file is
            missing or invalid.
    """
    local_path = Path(path)
    if not local_path.exists():
        return TrainingTriggerState()

    try:
        with local_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        return _state_from_json(raw)
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        logger.warning("Ignoring invalid trigger state file %s: %s", path, exc)
        return TrainingTriggerState()


def save_state(path: Path, state: TrainingTriggerState) -> None:
    """Atomically save the trigger state to a JSON file on disk.

    Writes to a temporary ``.tmp`` file first, then replaces the target
    file atomically using :meth:`~pathlib.Path.replace` to avoid partial
    writes. Creates parent directories if they do not exist.

    Args:
        path: Destination path for the JSON trigger state file.
        state: The :class:`TrainingTriggerState` instance to persist.
    """
    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = local_path.with_suffix(f"{local_path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(_state_to_json(state))

    tmp_path.replace(local_path)
