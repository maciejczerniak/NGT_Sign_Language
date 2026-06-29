"""Create or update the Azure ML training trigger schedule.

The schedule runs a lightweight checker job on a cron interval. The checker
evaluates data-change first and scheduled fallback second; it only submits the
full preprocessing and training pipeline when policy conditions pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from azure.ai.ml import command
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CronTrigger, JobResourceConfiguration, JobSchedule

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    get_client,
    resolve_compute_target,
    resolve_environment,
    resolve_instance_type,
    settings,
)

app = typer.Typer(
    name="create-training-schedule",
    help="Create or update the Azure ML scheduled retraining checker.",
    add_completion=False,
)


@app.command()
def main(
    schedule_name: str = typer.Option(
        "sign-language-training-trigger-daily",
        "--schedule-name",
        help="Azure ML schedule name.",
    ),
    cron: str = typer.Option(
        "0 7 * * *",
        "--cron",
        help="Cron expression for the checker schedule.",
    ),
    time_zone: str = typer.Option(
        "UTC",
        "--time-zone",
        help="Time zone used by the Azure ML cron trigger.",
    ),
    data_asset: str | None = typer.Option(
        None,
        "--data-asset",
        help="Raw data asset name to inspect. Defaults to AZURE_RAW_DATA_ASSET_NAME.",
    ),
    min_new_images: int = typer.Option(
        10,
        "--min-new-images",
        min=1,
        help="Minimum new or removed images required before retraining.",
    ),
    interval_days: int = typer.Option(
        7,
        "--interval-days",
        min=1,
        help="Minimum interval between fallback scheduled retraining runs.",
    ),
    experiment_name: str = typer.Option(
        "sign-language-training",
        "--experiment-name",
        help="Azure ML experiment used by checker and submitted pipeline jobs.",
    ),
) -> None:
    """Create or update the daily Azure ML trigger-checker schedule.

    Args:
        schedule_name: Azure ML schedule name.
        cron: Cron expression for the checker schedule.
        time_zone: Time zone used by the cron trigger.
        data_asset: Raw data asset name to inspect. Defaults to project settings.
        min_new_images: Minimum changed image count for data-change retraining.
        interval_days: Minimum days between scheduled fallback retraining runs.
        experiment_name: Azure ML experiment name.
    """
    ml_client = get_client()
    compute_target = resolve_compute_target(ml_client)
    environment = resolve_environment(ml_client)
    instance_type = resolve_instance_type()
    resolved_asset_name = data_asset or settings.azure_raw_data_asset_name

    checker_cmd = command(
        name="check-training-triggers",
        display_name="Check NGT retraining triggers",
        code=str(REPO_ROOT),
        command=(
            "pip install -e src/sign_language_training/ && "
            "python scripts/check_training_triggers.py "
            f"--asset-name {resolved_asset_name} "
            f"--min-new-images {min_new_images} "
            f"--interval-days {interval_days} "
            f"--experiment-name {experiment_name} "
            "--submit-kind sweep"
        ),
        environment=environment,
        compute=compute_target,
        resources=JobResourceConfiguration(instance_type=instance_type),
        environment_variables={
            "AZURE_AUTH_MODE": "managed_identity",
            "AZURE_SUBSCRIPTION_ID": str(settings.azure_subscription_id or ""),
            "AZURE_RESOURCE_GROUP": str(settings.azure_resource_group or ""),
            "AZURE_WORKSPACE": str(settings.azure_workspace or ""),
            "AZURE_COMPUTE_TARGET": str(settings.azure_compute_target or ""),
            "AZURE_ENVIRONMENT_NAME": str(settings.azure_environment_name or ""),
            "AZURE_ENVIRONMENT_VERSION": str(settings.azure_environment_version or ""),
            "AZURE_INSTANCE_TYPE": str(settings.azure_instance_type or ""),
            "AZURE_RAW_DATA_ASSET_NAME": settings.azure_raw_data_asset_name,
            "AZURE_PRETRAINED_CHECKPOINT_ASSET_NAME": str(
                settings.azure_pretrained_checkpoint_asset_name or ""
            ),
            "AZURE_PRETRAINED_CHECKPOINT_ASSET_VERSION": str(
                settings.azure_pretrained_checkpoint_asset_version or ""
            ),
        },
    )

    @pipeline(  # type: ignore[call-overload, untyped-decorator]
        display_name="Check NGT retraining triggers",
        experiment_name=experiment_name,
    )
    def checker_pipeline():
        """Define the one-step scheduled trigger checker pipeline."""
        checker_cmd()

    checker_job = checker_pipeline()

    schedule = JobSchedule(
        name=schedule_name,
        display_name="Daily sign-language retraining trigger check",
        description=(
            "Runs the retraining trigger checker. The checker submits Azure ML "
            "training only when enough images changed or the fallback interval "
            "has elapsed."
        ),
        trigger=CronTrigger(expression=cron, time_zone=time_zone),
        create_job=checker_job,
        tags={
            "project": "sign-language",
            "purpose": "training-trigger",
            "raw_data_asset_name": resolved_asset_name,
        },
    )

    created = ml_client.schedules.begin_create_or_update(schedule).result()
    typer.echo("Created/updated Azure ML schedule:")
    typer.echo(f"  name       : {created.name}")
    typer.echo(f"  cron       : {cron}")
    typer.echo(f"  time_zone  : {time_zone}")
    typer.echo(f"  asset_name : {resolved_asset_name}")


if __name__ == "__main__":
    app()
