"""Check dataset growth and submit Azure ML retraining if needed.

This script evaluates whether the local dataset has grown enough to warrant
retraining. It compares the current image count against the last recorded
state and submits an Azure ML pipeline job if the configured threshold is met
or if retraining is forced manually.

Typical usage::

    poetry run python scripts/check_data_change_and_train.py --data-dir data/raw
    poetry run python scripts/check_data_change_and_train.py --force
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
    raw_data_asset_reference,
    settings,
)
from sign_language_training.orchestration.trigger_policy import (  # noqa: E402
    TriggerPolicyConfig,
    evaluate_and_maybe_submit,
)

app = typer.Typer(
    name="check-data-change-and-train",
    help="Submit retraining when enough new images are detected.",
    add_completion=False,
)


@app.command()
def main(
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
        help="Minimum number of new images required before retraining.",
        min=1,
    ),
    force: bool = typer.Option(
        False,
        "--force/--no-force",
        help="Force retraining even if the threshold is not reached.",
    ),
) -> None:
    """Check image growth against the trigger policy and submit retraining if needed.

    Builds a :class:`~sign_language_training.orchestration.trigger_policy.TriggerPolicyConfig`
    from the provided options, evaluates the ``data_change`` trigger policy, and
    prints a decision summary to stdout. If the policy allows submission (or
    ``--force`` is set), an Azure ML preprocessing and training pipeline job is
    submitted.

    Args:
        data_dir: Local ImageFolder dataset root directory used to count
            current images and detect growth since the last recorded state.
        state_path: Path to the JSON file that persists the last recorded
            training trigger state (image count, file manifest, submission time).
        min_new_images: Minimum number of new images that must be present
            since the last submission before retraining is triggered automatically.
        force: If ``True``, bypasses the threshold check and submits
            retraining unconditionally.
    """
    config = TriggerPolicyConfig(
        data_dir=data_dir,
        state_path=state_path,
        raw_data_asset=raw_data_asset_reference(),
        raw_data_version=settings.azure_raw_data_asset_version,
        min_new_images=min_new_images,
    )

    decision = evaluate_and_maybe_submit(
        reason="data_change",
        config=config,
        force=force,
    )

    typer.echo(decision.message)
    typer.echo(f"  current_image_count : {decision.current_image_count}")
    typer.echo(f"  new_image_count     : {decision.new_image_count}")
    typer.echo(f"  submitted           : {decision.should_submit}")

    if decision.submitted_job_name:
        typer.echo(f"  job_name            : {decision.submitted_job_name}")
    if decision.studio_url:
        typer.echo(f"  studio_url          : {decision.studio_url}")


if __name__ == "__main__":
    app()
