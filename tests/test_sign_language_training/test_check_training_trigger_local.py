"""Evaluate on-prem / local retraining triggers and run training when due.

This is the on-prem analogue of ``scripts/check_training_triggers.py`` (which
targets Azure ML). It is what the Portainer cron container runs on each tick:

1. Build the current dataset inventory from a local ImageFolder.
2. Read the latest MLflow registered version timestamp (the "last training").
3. Apply the data-change-first / scheduled-interval-fallback policy.
4. If due, run ``scripts/run_local_pipeline.py --register-as-candidate`` in a
   subprocess. A passing gate registers a new ``@candidate`` version; the live
   ``@champion`` is left untouched (promote separately with
   ``scripts/promote_mlflow_champion.py``).

The subprocess approach is deliberate: it reuses the existing, tested pipeline
CLI rather than re-implementing the workflow, isolates the heavy PyTorch /
MediaPipe run from this lightweight checker (which may live in a long-running
cron container), and surfaces a non-zero exit code on failure for the cron
container logs.

Typical usage (inside the cron container)::

    python scripts/check_training_triggers_local.py \\
        --data-dir /data \\
        --output-dir /outputs \\
        --min-new-images 10 \\
        --interval-days 7

Force a run regardless of policy (manual)::

    python scripts/check_training_triggers_local.py --force
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.orchestration.mlflow_trigger_policy import (  # noqa: E402
    MlflowTriggerConfig,
    evaluate_and_maybe_train,
    latest_registered_version_timestamp,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="check-training-triggers-local",
    help="Evaluate on-prem MLflow retraining triggers and run local training.",
    add_completion=False,
)


def _build_runner(
    *,
    data_dir: Path,
    output_dir: Path,
    model_name: str,
    epochs: int | None,
    batch_size: int | None,
    num_workers: int,
    extra_args: list[str],
):
    """Return a callable that runs the local pipeline as a subprocess.

    The returned callable invokes ``scripts/run_local_pipeline.py`` with
    ``--register-as-candidate`` and parses the new candidate version from the
    MLflow registry after the run (the pipeline logs it, but reading the
    registry is more robust than scraping stdout).

    Args:
        data_dir: Raw ImageFolder dataset root passed as ``--raw-data-dir``.
        output_dir: Pipeline output root passed as ``--output-dir``.
        model_name: MLflow registered model name, used to read back the new
            candidate version after the run.
        epochs: Optional ``--epochs`` override.
        batch_size: Optional ``--batch-size`` override.
        num_workers: ``--num-workers`` value (0 is safest in containers).
        extra_args: Additional raw arguments appended to the pipeline command.

    Returns:
        A zero-argument callable returning the new ``@candidate`` version
        string, or ``None`` if the gate failed / no new version appeared.
    """

    def _run() -> str | None:
        before = latest_registered_version_timestamp(model_name)

        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_local_pipeline.py"),
            "--raw-data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--register-as-candidate",
            "--mlflow",
            "--num-workers",
            str(num_workers),
            "--clean",
        ]
        if epochs is not None:
            command += ["--epochs", str(epochs)]
        if batch_size is not None:
            command += ["--batch-size", str(batch_size)]
        command += extra_args

        logger.info("Running local pipeline: %s", " ".join(command))
        # Stream child output straight to this process's stdout/stderr so it
        # lands in the cron container logs. check=True raises on non-zero exit.
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)

        after = latest_registered_version_timestamp(model_name)
        if after is None:
            return None
        if before is not None and after <= before:
            # No newer version registered — gate likely failed.
            return None

        # Read back the newest version number for reporting.
        try:
            from mlflow.tracking import MlflowClient
            import os

            client = MlflowClient(
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "").strip() or None
            )
            versions = client.search_model_versions(f"name='{model_name}'")
            newest = max(
                versions,
                key=lambda v: int(getattr(v, "creation_timestamp", 0)),
                default=None,
            )
            return str(newest.version) if newest is not None else None
        except Exception:  # noqa: BLE001 - reporting only, never fatal
            return "unknown"

    return _run


@app.command()
def main(
    data_dir: Path = typer.Option(
        Path("/data"),
        "--data-dir",
        help="Local ImageFolder dataset root used for change detection.",
    ),
    output_dir: Path = typer.Option(
        Path("/outputs"),
        "--output-dir",
        help="Pipeline output root for checkpoints, metrics, and gate results.",
    ),
    state_path: Path = typer.Option(
        REPO_ROOT / "state" / "training_trigger_state_local.json",
        "--state-path",
        help="Path to the local trigger state JSON (data-change baseline).",
    ),
    model_name: str = typer.Option(
        "ngt-sign-language",
        "--model-name",
        help="MLflow registered model name driving the interval check.",
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
        help="Minimum days since the last registered version before fallback.",
    ),
    epochs: int = typer.Option(
        None, "--epochs", help="Optional epochs override for the pipeline."
    ),
    batch_size: int = typer.Option(
        None, "--batch-size", help="Optional batch-size override for the pipeline."
    ),
    num_workers: int = typer.Option(
        0, "--num-workers", help="DataLoader workers (0 is safest in containers)."
    ),
    force: bool = typer.Option(
        False, "--force", help="Run training regardless of policy conditions."
    ),
) -> None:
    """Evaluate on-prem retraining triggers and run the local pipeline if due.

    Args:
        data_dir: Local ImageFolder dataset root.
        output_dir: Pipeline output root.
        state_path: Path to the local trigger state JSON used for the
            data-change baseline.
        model_name: MLflow registered model name driving the interval check.
        min_new_images: Minimum changed image count for data-change retraining.
        interval_days: Minimum days between scheduled fallback runs.
        epochs: Optional epochs override forwarded to the pipeline.
        batch_size: Optional batch-size override forwarded to the pipeline.
        num_workers: DataLoader worker count forwarded to the pipeline.
        force: If ``True``, bypass policy checks and train unconditionally.
    """
    config = MlflowTriggerConfig(
        data_dir=data_dir,
        state_path=state_path,
        model_name=model_name,
        min_new_images=min_new_images,
        interval_days=interval_days,
    )

    runner = _build_runner(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        extra_args=[],
    )

    decision = evaluate_and_maybe_train(config=config, runner=runner, force=force)

    typer.echo(decision.message)
    typer.echo(f"  reason              : {decision.reason}")
    typer.echo(f"  current_image_count : {decision.current_image_count}")
    typer.echo(f"  new_image_count     : {decision.new_image_count}")
    typer.echo(f"  removed_image_count : {decision.removed_image_count}")
    typer.echo(f"  last_trained_at     : {decision.last_trained_at}")
    typer.echo(f"  trained             : {decision.should_train}")
    if decision.candidate_version:
        typer.echo(f"  candidate_version   : {decision.candidate_version}")


if __name__ == "__main__":
    app()
