"""Finalize completed Azure ML retraining sweeps.

This script is intended to run after automated retraining sweeps have had time
to finish. It finds completed sweep/pipeline jobs tagged as pending finalization,
promotes the best registered model version from that sweep, and marks the job as
finalized so it is not processed again.

Typical usage
-------------
Dry-run:

    poetry run python scripts/finalize_completed_sweeps.py --dry-run

Apply changes:

    poetry run python scripts/finalize_completed_sweeps.py --yes

The script expects retraining jobs to be tagged with:

    purpose=retraining-sweep
    finalization_status=pending

It promotes model versions by setting:

    promoted=true

on the selected version and setting:

    promoted=false

on any previously promoted versions of the same model.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import typer
from azure.ai.ml import MLClient

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import get_client  # noqa: E402
from sign_language_training.settings import settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


TERMINAL_SUCCESS_STATUSES = {"Completed"}
TERMINAL_FAILED_STATUSES = {"Failed", "Canceled", "Cancelled"}


@dataclass(frozen=True)
class CandidateModel:
    """Candidate model version registered by a completed retraining sweep."""

    version: Any
    version_id: str
    f1_macro: float
    accuracy: float
    run_id: str | None
    sweep_id: str | None


def _job_status(job: Any) -> str:
    """Return a stable string status for an Azure ML job."""
    status = getattr(job, "status", "")
    return str(status)


def _job_tags(job: Any) -> dict[str, str]:
    """Return Azure ML job tags as a mutable string dictionary."""
    raw_tags = getattr(job, "tags", None) or {}
    return {str(key): str(value) for key, value in raw_tags.items()}


def _model_tags(model_version: Any) -> dict[str, str]:
    """Return Azure ML model-version tags as a string dictionary."""
    raw_tags = getattr(model_version, "tags", None) or {}
    return {str(key): str(value) for key, value in raw_tags.items()}


def _to_float(value: str | None) -> float:
    """Parse metric tag values safely."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _list_jobs(ml_client: MLClient) -> Iterable[Any]:
    """List jobs in the workspace.

    Azure ML SDK versions differ slightly in accepted keyword arguments, so this
    wrapper intentionally uses the broadest compatible call.
    """
    return ml_client.jobs.list()


def _list_child_jobs(ml_client: MLClient, parent_job_name: str) -> list[Any]:
    """Best-effort listing of child jobs under a pipeline/sweep parent."""
    try:
        return list(ml_client.jobs.list(parent_job_name=parent_job_name))
    except TypeError:
        return []
    except Exception as exc:
        logger.warning("Could not list child jobs for %s: %s", parent_job_name, exc)
        return []


def _collect_possible_sweep_ids(ml_client: MLClient, parent_job: Any) -> set[str]:
    """Collect parent and child job names that model tags may reference."""
    parent_name = str(getattr(parent_job, "name", ""))
    possible_ids = {parent_name} if parent_name else set()

    # For nested pipeline -> sweep -> trial structures, the registered model may
    # tag the nested sweep job instead of the outer pipeline job.
    first_level_children = _list_child_jobs(ml_client, parent_name)
    for child in first_level_children:
        child_name = str(getattr(child, "name", ""))
        if child_name:
            possible_ids.add(child_name)
            for grandchild in _list_child_jobs(ml_client, child_name):
                grandchild_name = str(getattr(grandchild, "name", ""))
                if grandchild_name:
                    possible_ids.add(grandchild_name)

    return possible_ids


def _matches_sweep(tags: dict[str, str], possible_sweep_ids: set[str]) -> bool:
    """Return True if model tags indicate membership in the sweep."""
    sweep_id = tags.get("sweep_id")
    run_id = tags.get("run_id")
    parent_run_id = tags.get("parent_run_id")
    root_run_id = tags.get("root_run_id")
    azureml_root_run_id = tags.get("azureml_root_run_id")

    direct_values = {
        value
        for value in (
            sweep_id,
            parent_run_id,
            root_run_id,
            azureml_root_run_id,
        )
        if value
    }

    if direct_values.intersection(possible_sweep_ids):
        return True

    # Some training code saves a child trial run id. For Azure ML sweeps, trial
    # run ids often start with the parent sweep id.
    if run_id and any(run_id.startswith(sweep_id) for sweep_id in possible_sweep_ids):
        return True

    return False


def _find_best_model_from_sweep(
    ml_client: MLClient,
    *,
    model_name: str,
    possible_sweep_ids: set[str],
) -> CandidateModel | None:
    """Find the best registered model version produced by a sweep."""
    candidates: list[CandidateModel] = []

    for version in ml_client.models.list(name=model_name):
        tags = _model_tags(version)
        if not _matches_sweep(tags, possible_sweep_ids):
            continue

        candidates.append(
            CandidateModel(
                version=version,
                version_id=str(getattr(version, "version", "")),
                f1_macro=_to_float(tags.get("f1_macro")),
                accuracy=_to_float(tags.get("accuracy")),
                run_id=tags.get("run_id"),
                sweep_id=tags.get("sweep_id"),
            )
        )

    if not candidates:
        return None

    return max(candidates, key=lambda item: (item.accuracy, item.f1_macro))


def _promote_model_version(
    ml_client: MLClient,
    *,
    model_name: str,
    target_version: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Set promoted=true on target version and demote older promoted versions."""
    versions_to_update = []

    logger.info("Planned promotion changes for model '%s':", model_name)
    for version in ml_client.models.list(name=model_name):
        tags = _model_tags(version)
        version_id = str(getattr(version, "version", ""))
        currently_promoted = tags.get("promoted", "false")

        if version_id == target_version:
            logger.info(
                "  version %s: promoted %r -> 'true'",
                version_id,
                currently_promoted,
            )
            versions_to_update.append(version)
        elif currently_promoted == "true":
            logger.info("  version %s: promoted 'true' -> 'false'", version_id)
            versions_to_update.append(version)

    if dry_run:
        return 0, 0

    promoted_count = 0
    demoted_count = 0

    for version in versions_to_update:
        tags = _model_tags(version)
        version_id = str(getattr(version, "version", ""))

        if version_id == target_version:
            if tags.get("promoted") == "true":
                logger.info("Version %s is already promoted.", version_id)
                continue
            version.tags = {**tags, "promoted": "true"}
            ml_client.models.create_or_update(version)
            logger.info("Promoted version %s.", version_id)
            promoted_count += 1
        elif tags.get("promoted") == "true":
            version.tags = {**tags, "promoted": "false"}
            ml_client.models.create_or_update(version)
            logger.info("Demoted version %s.", version_id)
            demoted_count += 1

    return promoted_count, demoted_count


def _mark_job_finalized(
    ml_client: MLClient,
    job: Any,
    *,
    status: str,
    model_name: str | None = None,
    model_version: str | None = None,
    reason: str | None = None,
    dry_run: bool,
) -> None:
    """Update finalization tags on the processed retraining job."""
    tags = _job_tags(job)
    finalized_at = datetime.now(timezone.utc).isoformat()

    updated_tags = {
        **tags,
        "finalization_status": status,
        "finalized_at": finalized_at,
    }

    if model_name:
        updated_tags["promoted_model_name"] = model_name
    if model_version:
        updated_tags["promoted_model_version"] = model_version
    if reason:
        updated_tags["finalization_reason"] = reason

    logger.info(
        "Marking job %s finalization_status=%s",
        getattr(job, "name", "<unknown>"),
        status,
    )

    if dry_run:
        return

    job.tags = updated_tags
    ml_client.jobs.create_or_update(job)


def _pending_retraining_jobs(
    ml_client: MLClient,
    *,
    purpose_tag: str,
    pending_status: str,
    limit: int,
) -> list[Any]:
    """Return jobs tagged as pending retraining sweeps."""
    matches: list[Any] = []

    for job in _list_jobs(ml_client):
        tags = _job_tags(job)
        if tags.get("purpose") != purpose_tag:
            continue
        if tags.get("finalization_status") != pending_status:
            continue
        matches.append(job)
        if len(matches) >= limit:
            break

    return matches


@app.command()
def main(
    model_name: str = typer.Option(
        settings.model_registry_name,
        "--model-name",
        help="Azure ML registered model name to promote.",
    ),
    purpose_tag: str = typer.Option(
        "retraining-sweep",
        "--purpose-tag",
        help="Job tag value used to identify retraining sweeps.",
    ),
    pending_status: str = typer.Option(
        "pending",
        "--pending-status",
        help="finalization_status tag value that means the sweep is not finalized.",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        min=1,
        help="Maximum number of pending jobs to inspect in one run.",
    ),
    mark_failed: bool = typer.Option(
        True,
        "--mark-failed/--no-mark-failed",
        help="Mark failed/cancelled pending sweeps as finalized failures.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print planned changes without modifying Azure ML jobs or models.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Apply changes without interactive confirmation.",
    ),
) -> None:
    """Finalize completed pending retraining sweeps.

    This command is safe to run on a schedule. If no completed pending sweep is
    found, it exits successfully without changing anything.
    """
    ml_client = get_client()

    pending_jobs = _pending_retraining_jobs(
        ml_client,
        purpose_tag=purpose_tag,
        pending_status=pending_status,
        limit=limit,
    )

    if not pending_jobs:
        typer.echo("No pending retraining sweeps found. Nothing to finalize.")
        return

    typer.echo(f"Found {len(pending_jobs)} pending retraining sweep job(s).")

    planned_completed_jobs = []
    planned_failed_jobs = []

    for job in pending_jobs:
        job_name = str(getattr(job, "name", ""))
        status = _job_status(job)

        if status in TERMINAL_SUCCESS_STATUSES:
            planned_completed_jobs.append(job)
            typer.echo(f"  completed: {job_name}")
        elif status in TERMINAL_FAILED_STATUSES:
            planned_failed_jobs.append(job)
            typer.echo(f"  failed/cancelled: {job_name} ({status})")
        else:
            typer.echo(f"  still running or queued: {job_name} ({status})")

    if not planned_completed_jobs and not planned_failed_jobs:
        typer.echo("No completed or failed pending sweeps found. Nothing to finalize.")
        return

    if dry_run:
        typer.echo("Dry-run mode: no Azure ML changes will be applied.")

    if not dry_run and not yes:
        if not typer.confirm("Apply finalization and model promotion changes?"):
            raise typer.Exit(1)

    finalized_count = 0
    promoted_count_total = 0
    demoted_count_total = 0

    for job in planned_completed_jobs:
        job_name = str(getattr(job, "name", ""))
        possible_sweep_ids = _collect_possible_sweep_ids(ml_client, job)

        logger.info(
            "Looking for candidate model versions for job %s using ids: %s",
            job_name,
            sorted(possible_sweep_ids),
        )

        best = _find_best_model_from_sweep(
            ml_client,
            model_name=model_name,
            possible_sweep_ids=possible_sweep_ids,
        )

        if best is None:
            typer.echo(
                f"No registered candidate model found for completed sweep {job_name}. "
                "Leaving it pending."
            )
            continue

        typer.echo(
            "Best candidate for "
            f"{job_name}: model={model_name} version={best.version_id} "
            f"f1_macro={best.f1_macro} accuracy={best.accuracy} "
            f"run_id={best.run_id}"
        )

        promoted_count, demoted_count = _promote_model_version(
            ml_client,
            model_name=model_name,
            target_version=best.version_id,
            dry_run=dry_run,
        )
        promoted_count_total += promoted_count
        demoted_count_total += demoted_count

        _mark_job_finalized(
            ml_client,
            job,
            status="completed",
            model_name=model_name,
            model_version=best.version_id,
            dry_run=dry_run,
        )
        finalized_count += 1

    if mark_failed:
        for job in planned_failed_jobs:
            _mark_job_finalized(
                ml_client,
                job,
                status="failed",
                reason=f"sweep_status={_job_status(job)}",
                dry_run=dry_run,
            )
            finalized_count += 1

    typer.echo(
        "\nDone. "
        f"Finalized jobs: {finalized_count}. "
        f"Promoted models: {promoted_count_total}. "
        f"Demoted models: {demoted_count_total}."
    )


if __name__ == "__main__":
    app()
