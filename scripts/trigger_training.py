"""Trigger Azure ML retraining through the shared trigger policy.

This script is the general-purpose retraining trigger supporting all three
policy reasons: ``manual``, ``data_change``, and ``scheduled``. It evaluates
the configured policy and submits an Azure ML preprocessing and training
pipeline job if the policy allows it.

For data-change-specific automation, see ``check_data_change_and_train.py``.

Typical usage::

    poetry run python scripts/trigger_training.py --reason manual --force
    poetry run python scripts/trigger_training.py --reason data_change
    poetry run python scripts/trigger_training.py --reason scheduled
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    raw_data_asset_reference,
    settings,
)
from sign_language_training.orchestration.trigger_policy import (  # noqa: E402
    TriggerPolicyConfig,
    evaluate_and_maybe_submit,
)

TriggerReason = Literal["manual", "data_change", "scheduled"]

app = typer.Typer(
    name="trigger-training",
    help="Evaluate retraining policy and optionally submit Azure ML pipeline.",
    add_completion=False,
)


@app.command()
def main(
    reason: TriggerReason = typer.Option(
        "manual",
        "--reason",
        help="Trigger reason: manual, data_change, or scheduled.",
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Force submission regardless of policy checks.",
    ),
    data_dir: Path = typer.Option(
        REPO_ROOT / "data" / "raw",
        "--data-dir",
        help="Local ImageFolder dataset root used for change detection.",
    ),
    state_path: Path = typer.Option(
        REPO_ROOT / "state" / "training_trigger_state.json",
        "--state-path",
        help="Path to trigger state JSON file.",
    ),
    min_new_images: int = typer.Option(
        100,
        "--min-new-images",
        help="Minimum number of new images required for data-change retraining.",
        min=1,
    ),
    interval_days: int = typer.Option(
        7,
        "--interval-days",
        help="Minimum interval between scheduled fallback retraining runs.",
        min=1,
    ),
) -> None:
    """Evaluate the retraining trigger policy and submit an Azure ML pipeline if needed.

    Builds a :class:`~sign_language_training.orchestration.trigger_policy.TriggerPolicyConfig`
    from the provided options and delegates to
    :func:`~sign_language_training.orchestration.trigger_policy.evaluate_and_maybe_submit`.

    Policy behaviour by reason:

    - ``manual``: submits only when ``--force`` is set.
    - ``data_change``: submits when the number of new images in ``data_dir``
      since the last recorded state meets or exceeds ``min_new_images``.
    - ``scheduled``: submits when the time since the last submission exceeds
      ``interval_days``.

    Prints the policy decision summary to stdout including the trigger reason,
    current and new image counts, submission status, and job details if a
    pipeline was submitted.

    Args:
        reason: The trigger reason controlling which policy is evaluated.
            One of ``manual``, ``data_change``, or ``scheduled``.
        force: If ``True``, bypasses all policy checks and submits
            retraining unconditionally regardless of the chosen reason.
        data_dir: Local ImageFolder dataset root used to count current
            images and detect growth since the last recorded state.
        state_path: Path to the JSON file persisting the last recorded
            training trigger state (image count, file manifest, submission time).
        min_new_images: Minimum number of new images that must be present
            since the last submission before ``data_change`` retraining is triggered.
        interval_days: Minimum number of days that must have elapsed
            since the last submission before ``scheduled`` retraining is triggered.
    """
    config = TriggerPolicyConfig(
        data_dir=data_dir,
        state_path=state_path,
        raw_data_asset=raw_data_asset_reference(),
        raw_data_version=settings.azure_raw_data_asset_version,
        min_new_images=min_new_images,
        interval_days=interval_days,
    )

    decision = evaluate_and_maybe_submit(
        reason=reason,
        config=config,
        force=force,
    )

    typer.echo(decision.message)
    typer.echo(f"  reason              : {decision.reason}")
    typer.echo(f"  current_image_count : {decision.current_image_count}")
    typer.echo(f"  new_image_count     : {decision.new_image_count}")
    typer.echo(f"  submitted           : {decision.should_submit}")

    if decision.submitted_job_name:
        typer.echo(f"  job_name            : {decision.submitted_job_name}")
    if decision.studio_url:
        typer.echo(f"  studio_url          : {decision.studio_url}")


if __name__ == "__main__":
    app()
