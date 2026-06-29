"""Evaluate automatic Azure ML retraining triggers from Azure ML metadata.

The scheduled checker does not write trigger state and does not acquire blob
leases. It uses Azure ML as the source of truth:

1. Skip if another retraining job is already active.
2. Read the latest ``ngt-raw`` data asset version and its metadata tags.
3. Compare that snapshot with the latest completed retraining job tags.
4. Submit retraining when the data-change or interval condition passes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    get_client,
    settings as azure_settings,
)
from sign_language_training.orchestration.pipeline_submitter import (  # noqa: E402
    submit_retraining_pipeline,
)
from sign_language_training.orchestration.sweep_finalizer import (  # noqa: E402
    finalize_completed_sweeps,
)
from sign_language_training.orchestration.sweep_submitter import (  # noqa: E402
    submit_retraining_sweep,
)
from sign_language_training.orchestration.trigger_policy import (  # noqa: E402
    RawDataAssetSnapshot,
    RetrainingJobSnapshot,
    find_active_retraining_job,
    find_latest_completed_retraining_job,
    get_latest_raw_data_snapshot,
)

app = typer.Typer(
    name="check-training-triggers",
    help="Evaluate Azure ML data-change and interval retraining triggers.",
    add_completion=False,
)


def _print_snapshot(snapshot: RawDataAssetSnapshot) -> None:
    """Print the latest raw data asset snapshot.

    Args:
        snapshot: Raw data asset metadata discovered in Azure ML.
    """
    typer.echo("Latest raw data asset:")
    typer.echo(f"  reference     : {snapshot.reference}")
    typer.echo(f"  image_count   : {snapshot.image_count}")
    typer.echo(f"  manifest_hash : {snapshot.manifest_hash}")


def _print_last_job(job: RetrainingJobSnapshot | None) -> None:
    """Print the latest completed retraining job snapshot.

    Args:
        job: Latest completed retraining job metadata, or ``None``.
    """
    if job is None:
        typer.echo("Latest completed retraining job: none")
        return

    typer.echo("Latest completed retraining job:")
    typer.echo(f"  name                : {job.name}")
    typer.echo(f"  status              : {job.status}")
    typer.echo(f"  created_at          : {job.created_at}")
    typer.echo(f"  raw_data_asset      : {job.raw_data_asset}")
    typer.echo(f"  raw_data_version    : {job.raw_data_version}")
    typer.echo(f"  trigger_image_count : {job.trigger_image_count}")
    if job.studio_url:
        typer.echo(f"  studio_url          : {job.studio_url}")


def _is_interval_due(
    last_job: RetrainingJobSnapshot | None,
    interval_days: int,
) -> bool:
    """Return whether the interval fallback trigger is due.

    Args:
        last_job: Latest completed retraining job metadata, or ``None``.
        interval_days: Minimum number of days between fallback retraining runs.

    Returns:
        ``True`` if no completed retraining job exists or the latest completed
        job is older than ``interval_days``.
    """
    if last_job is None or last_job.created_at is None:
        return True

    return datetime.now(timezone.utc) - last_job.created_at >= timedelta(
        days=interval_days
    )


def _new_image_count(
    latest_raw: RawDataAssetSnapshot,
    last_job: RetrainingJobSnapshot | None,
) -> int | None:
    """Return new image count based on Azure ML metadata tags.

    Args:
        latest_raw: Latest raw data asset metadata.
        last_job: Latest completed retraining job metadata, or ``None``.

    Returns:
        Number of images added since the last completed retraining job, or
        ``None`` if the required metadata is missing.
    """
    if latest_raw.image_count is None:
        return None

    if last_job is None:
        return latest_raw.image_count

    if last_job.trigger_image_count is None:
        return None

    return max(0, latest_raw.image_count - last_job.trigger_image_count)


@app.command()
def main(
    asset_name: str = typer.Option(
        azure_settings.azure_raw_data_asset_name,
        "--asset-name",
        help="Azure ML raw data asset name to inspect.",
    ),
    min_new_images: int = typer.Option(
        10,
        "--min-new-images",
        help="Minimum new images required before retraining.",
        min=1,
    ),
    interval_days: int = typer.Option(
        7,
        "--interval-days",
        help="Minimum interval between fallback scheduled retraining runs.",
        min=1,
    ),
    experiment_name: str = typer.Option(
        "sign-language-training",
        "--experiment-name",
        help="Azure ML experiment name.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Submit retraining regardless of metadata trigger conditions.",
    ),
    submit_kind: str = typer.Option(
        "sweep",
        "--submit-kind",
        help="Submission type when trigger conditions pass: sweep or train.",
    ),
    model_name: str = typer.Option(
        "ngt-sign-language",
        "--model-name",
        help="Azure ML model name used for sweep finalization.",
    ),
    archive_non_best: bool = typer.Option(
        True,
        "--archive-non-best/--keep-non-best",
        help="Archive non-best models from finalized sweeps.",
    ),
) -> None:
    """Evaluate automatic Azure retraining triggers and submit when needed.

    Args:
        asset_name: Azure ML raw data asset name, usually ``ngt-raw``.
        min_new_images: Minimum new images required for data-change retraining.
        interval_days: Minimum days between interval fallback retraining runs.
        experiment_name: Azure ML experiment used by checker and submitted jobs.
        force: If ``True``, bypasses policy checks and submits retraining.
        submit_kind: ``"sweep"`` to submit a hyperparameter sweep, or
            ``"train"`` to submit the fixed preprocessing/training pipeline.
        model_name: Azure ML model name used when finalizing completed sweeps.
        archive_non_best: If ``True``, archive non-best models from a sweep
            after finalization.
    """
    if submit_kind not in {"sweep", "train"}:
        raise typer.BadParameter(
            "--submit-kind must be either 'sweep' or 'train'.",
            param_hint="--submit-kind",
        )

    ml_client = get_client()

    finalization_results = finalize_completed_sweeps(
        ml_client=ml_client,
        experiment_name=experiment_name,
        model_name=model_name,
        archive_non_best=archive_non_best,
    )
    for result in finalization_results:
        typer.echo(f"Finalized sweep {result.sweep_id}: {result.message}")
        if result.archived_versions:
            typer.echo(f"  archived_versions: {', '.join(result.archived_versions)}")

    active_job = find_active_retraining_job(experiment_name)
    if active_job is not None and not force:
        typer.echo(
            "Skipped retraining: Azure ML job "
            f"'{getattr(active_job, 'name', '<unknown>')}' is already "
            f"{getattr(active_job, 'status', '<unknown>')}."
        )
        return

    latest_raw = get_latest_raw_data_snapshot(ml_client, asset_name)
    last_job = find_latest_completed_retraining_job(ml_client, experiment_name)
    new_count = _new_image_count(latest_raw, last_job)
    interval_due = _is_interval_due(last_job, interval_days)

    _print_snapshot(latest_raw)
    _print_last_job(last_job)

    reason: str | None = None
    force_preprocess = False
    if force:
        reason = "manual"
        force_preprocess = True
    elif new_count is not None and new_count >= min_new_images:
        reason = "data_change"
    elif interval_due:
        reason = "scheduled"

    if reason is None:
        typer.echo("Skipped retraining:")
        if new_count is None:
            typer.echo(
                "  data_change : unavailable because image_count metadata is missing"
            )
        else:
            typer.echo(
                f"  data_change : {new_count} new images < threshold "
                f"{min_new_images}"
            )
        typer.echo(
            f"  scheduled   : latest completed retraining is within {interval_days} days"
        )
        return

    typer.echo(f"Submitting Azure ML retraining {submit_kind}. reason={reason}")
    if submit_kind == "sweep":
        submitted = submit_retraining_sweep(
            experiment_name=experiment_name,
            data_asset=latest_raw.reference,
            ngt_raw_version=latest_raw.version,
            trigger_reason=reason,
            trigger_image_count=latest_raw.image_count,
            raw_data_manifest_hash=latest_raw.manifest_hash,
        )
    else:
        submitted = submit_retraining_pipeline(
            experiment_name=experiment_name,
            data_asset=latest_raw.reference,
            ngt_raw_version=latest_raw.version,
            force_preprocess=force_preprocess,
            mlflow_enabled=True,
            trigger_reason=reason,
            trigger_image_count=latest_raw.image_count,
            raw_data_manifest_hash=latest_raw.manifest_hash,
        )

    typer.echo(f"Submitted Azure ML retraining {submit_kind}:")
    typer.echo(f"  job_name   : {submitted.name}")
    typer.echo(f"  studio_url : {submitted.studio_url}")


if __name__ == "__main__":
    app()
