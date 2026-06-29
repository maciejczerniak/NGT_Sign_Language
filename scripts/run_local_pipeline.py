"""Run the full preprocess → train pipeline locally.

This script replicates the Azure ML pipeline graph on a local machine:

1. Stratified split + offline augmentation  (augmentation.py)
2. Training + evaluation + gate check       (train.py)

It calls the same functions that run inside Azure ML jobs, so metrics
and results are directly comparable.

Two integrations with the self-hosted MLflow stack (DEPLOY_TARGET=onprem):

- ``--pretrain-from-mlflow``: download the current ``@champion`` checkpoint
  from MLflow to use as the pretrained starting point. Removes the need to
  pass ``--pretrained-checkpoint`` when running inside the Portainer training
  stack.
- ``--register-as-candidate``: after training succeeds (gate passes), upload
  the new checkpoint to MLflow, register it as a new version of
  ``ngt-sign-language``, and set the ``@candidate`` alias. Someone (or
  scripts/promote_mlflow_champion.py) then promotes ``@candidate`` →
  ``@champion`` to make the backend pick it up.

Both integrations require ``MLFLOW_TRACKING_URI`` to be set in the
environment. They are independent — you can run the pipeline without either,
with just one, or with both.

Typical usage::

    poetry run python scripts/run_local_pipeline.py

Inside the Portainer training stack (full MLflow round-trip)::

    python scripts/run_local_pipeline.py \\
        --raw-data-dir /data \\
        --output-dir /outputs \\
        --pretrain-from-mlflow \\
        --register-as-candidate \\
        --mlflow \\
        --num-workers 0 --clean

Override defaults from a local clone::

    poetry run python scripts/run_local_pipeline.py \\
        --raw-data-dir data/raw \\
        --output-dir outputs/local_pipeline \\
        --pretrained-checkpoint src/sign_language/models/best_ngt_model_v2.pth \\
        --epochs 5 --batch-size 8 --num-workers 0
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer

# ---------------------------------------------------------------------------
# Make src/ importable when running from repo root
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.augmentation import (  # noqa: E402
    augment_dir,
    stratified_split,
)
from sign_language_training.configuration import (  # noqa: E402
    TrainingConfig,
    TrainingPaths,
)
from sign_language_training.train import run_training_workflow  # noqa: E402

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="local-pipeline",
    help="Run the full local preprocess → train pipeline.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# MLflow integration helpers
# ---------------------------------------------------------------------------


def _fetch_pretrained_from_mlflow(
    model_name: str,
    alias: str,
    cache_dir: Path,
) -> Path:
    """Download the aliased checkpoint from MLflow to use as a starting point.

    :param model_name: Registered model name in MLflow, e.g. ``ngt-sign-language``.
    :param alias: Registry alias to resolve. Typically ``"champion"``.
    :param cache_dir: Local directory to download the artifact into.
    :returns: Path to the downloaded ``.pth`` file.
    :raises typer.Exit: If MLflow is not installed, ``MLFLOW_TRACKING_URI`` is
        not set, the alias doesn't resolve, or no ``.pth`` file is found in
        the downloaded artifact directory.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.error(
            "MLflow is not installed but --pretrain-from-mlflow was requested. "
            "Install with: poetry install --with training"
        )
        raise typer.Exit(1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        logger.error("--pretrain-from-mlflow requires MLFLOW_TRACKING_URI to be set.")
        raise typer.Exit(1)

    logger.info(
        "Fetching pretrained checkpoint: models:/%s@%s from %s",
        model_name,
        alias,
        tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    except Exception as exc:
        logger.error(
            "Could not resolve alias '@%s' for model '%s': %s. "
            "If this is the first run, register an initial model with "
            "scripts/upload_to_mlflow.py first.",
            alias,
            model_name,
            exc,
        )
        raise typer.Exit(1) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    artifact_uri = f"models:/{model_name}@{alias}"
    mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri, dst_path=str(cache_dir)
    )

    pth_files = list(cache_dir.rglob("*.pth"))
    if not pth_files:
        logger.error(
            "No .pth file found after downloading '%s@%s' (v%s) into %s",
            model_name,
            alias,
            mv.version,
            cache_dir,
        )
        raise typer.Exit(1)

    logger.info("Pretrained checkpoint: v%s → %s", mv.version, pth_files[0])
    return pth_files[0]


def _register_as_candidate(
    checkpoint_path: Path,
    model_name: str,
    alias: str,
    val_acc: float,
    f1_macro: float,
    run_name: str,
) -> Optional[str]:
    """Upload the new checkpoint to MLflow, register it, set the alias.

    Mirrors the pattern in ``scripts/upload_to_mlflow.py``: low-level
    ``MlflowClient.create_model_version()`` because MLflow 3.x's high-level
    ``mlflow.register_model()`` requires a flavor-tagged ``logged_model``
    object that our raw ``.pth`` doesn't satisfy.

    :param checkpoint_path: Path to the trained ``.pth`` to upload.
    :param model_name: Registered model name in MLflow.
    :param alias: Registry alias to set. Typically ``"candidate"`` so it
        doesn't immediately become live — a human or quality gate promotes
        ``@candidate`` → ``@champion`` separately.
    :param val_acc: Best validation accuracy from training, used as a tag.
    :param f1_macro: Final macro-F1 score, used as a tag.
    :param run_name: Run name shown in the MLflow UI.
    :returns: The new model version string, or ``None`` if MLflow isn't
        configured (in which case a warning is logged and registration is
        skipped — same shape as ``--no-mlflow``).
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.warning("MLflow not installed — skipping candidate registration.")
        return None

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI not set — skipping candidate registration.")
        return None

    if not checkpoint_path.exists():
        logger.warning(
            "Trained checkpoint not found at %s — skipping candidate "
            "registration (training may have failed the gate).",
            checkpoint_path,
        )
        return None

    logger.info(
        "Uploading new checkpoint to MLflow registry as '%s@%s candidate' …",
        model_name,
        alias,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("ngt-training-runs")
    client = MlflowClient(tracking_uri=tracking_uri)

    # Ensure the registered model entry exists (idempotent).
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_metric("val_acc", val_acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_artifact(str(checkpoint_path), artifact_path="data")

    mv = client.create_model_version(
        name=model_name,
        source=f"runs:/{run_id}/data",
        run_id=run_id,
    )
    client.set_registered_model_alias(name=model_name, alias=alias, version=mv.version)
    client.set_model_version_tag(
        name=model_name, version=mv.version, key="val_acc", value=f"{val_acc:.4f}"
    )
    client.set_model_version_tag(
        name=model_name, version=mv.version, key="f1_macro", value=f"{f1_macro:.4f}"
    )
    client.set_model_version_tag(
        name=model_name, version=mv.version, key="source", value="local-pipeline"
    )

    logger.info(
        "Registered %s v%s with alias '@%s'. Promote to @champion when ready.",
        model_name,
        mv.version,
        alias,
    )
    return str(mv.version)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@app.command()
def main(
    raw_data_dir: Path = typer.Option(
        "data/raw",
        "--raw-data-dir",
        help="Path to the raw ImageFolder dataset.",
    ),
    output_dir: Path = typer.Option(
        "outputs/local_pipeline",
        "--output-dir",
        help="Root directory for all pipeline outputs.",
    ),
    pretrained_checkpoint: Optional[Path] = typer.Option(
        None,
        "--pretrained-checkpoint",
        help=(
            "Path to the pretrained .pth checkpoint. Either pass this or use "
            "--pretrain-from-mlflow to fetch from the registry."
        ),
    ),
    pretrain_from_mlflow: bool = typer.Option(
        False,
        "--pretrain-from-mlflow",
        help=(
            "Fetch the pretrained checkpoint from MLflow (models:/<name>@champion) "
            "instead of using a local file. Requires MLFLOW_TRACKING_URI."
        ),
    ),
    register_as_candidate: bool = typer.Option(
        False,
        "--register-as-candidate",
        help=(
            "After training succeeds, upload the trained checkpoint to MLflow "
            "as a new version aliased '@candidate'. Requires MLFLOW_TRACKING_URI. "
            "The backend only serves '@champion', so candidates don't go live "
            "automatically — use scripts/promote_mlflow_champion.py to promote."
        ),
    ),
    model_name: str = typer.Option(
        "ngt-sign-language",
        "--model-name",
        help="MLflow registered model name (used by --pretrain-from-mlflow and --register-as-candidate).",
    ),
    augment_copies: int = typer.Option(4, "--augment-copies", min=1),
    img_size: int = typer.Option(224, "--img-size", min=1),
    seed: int = typer.Option(42, "--seed"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    batch_size: int = typer.Option(16, "--batch-size", min=1),
    epochs: int = typer.Option(30, "--epochs", min=1),
    learning_rate: float = typer.Option(1e-4, "--learning-rate"),
    patience: int = typer.Option(7, "--patience", min=1),
    target_accuracy: float = typer.Option(0.85, "--target-accuracy"),
    expected_num_classes: int = typer.Option(22, "--expected-num-classes", min=1),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    f1_threshold: float = typer.Option(0.80, "--f1-threshold"),
    mlflow_enabled: bool = typer.Option(False, "--mlflow/--no-mlflow"),
    skip_preprocess: bool = typer.Option(
        False,
        "--skip-preprocess",
        help="Skip preprocessing if output dirs already exist from a previous run.",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Delete existing output directory before running.",
    ),
) -> None:
    """Run the full local retraining pipeline (preprocess + train + evaluate).

    Executes the same two-step pipeline that runs in Azure ML:

    **Step 1 — Preprocessing**: performs a deterministic stratified split of
    the raw ImageFolder dataset into train, validation, and test subsets, then
    applies offline augmentation to the training split.

    **Step 2 — Training**: fine-tunes the EfficientNet-B0 model from the
    pretrained checkpoint, evaluates on the validation set, and runs the gate
    check. Results and checkpoints are written under ``output_dir``.

    With ``--pretrain-from-mlflow`` the starting checkpoint is downloaded from
    the on-prem MLflow registry (``models:/<name>@champion``) instead of being
    read from a local path.

    With ``--register-as-candidate``, after the gate passes the trained
    checkpoint is uploaded back to MLflow and aliased ``@candidate``. The
    backend only loads ``@champion``, so candidates do not become live
    automatically — promote with ``scripts/promote_mlflow_champion.py``.

    Preprocessing can be skipped with ``--skip-preprocess`` if the output
    directories already exist from a prior run. Use ``--clean`` to wipe the
    output directory before starting.

    :raises typer.Exit: If neither ``--pretrained-checkpoint`` nor
        ``--pretrain-from-mlflow`` is provided, if the resolved checkpoint
        does not exist, if ``raw_data_dir`` does not exist, or if either
        MLflow integration fails.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ── Resolve paths ──────────────────────────────────────────────────────
    raw_data_dir = Path(raw_data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    train_dir = output_dir / "preprocessed" / "train"
    val_dir = output_dir / "preprocessed" / "val"
    test_dir = output_dir / "preprocessed" / "test"
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    pretrained_cache_dir = output_dir / "pretrained_cache"

    if not raw_data_dir.exists():
        logger.error("Raw data directory not found: %s", raw_data_dir)
        raise typer.Exit(code=1)

    # Clean output_dir BEFORE resolving pretrained, because the MLflow download
    # writes into output_dir/pretrained_cache/. If we cleaned after, we would
    # wipe the freshly-downloaded pretrained checkpoint.
    #
    # Delete the CONTENTS of output_dir rather than output_dir itself.
    # `shutil.rmtree(output_dir)` fails with "Device or resource busy" when
    # output_dir is a Docker volume mount point (the kernel won't let us
    # rmdir a mount). Removing children one by one works in both cases.
    if clean and output_dir.exists():
        logger.info("Cleaning output directory: %s", output_dir)
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    # Resolve pretrained source: explicit path OR MLflow.
    if pretrain_from_mlflow and pretrained_checkpoint is not None:
        logger.error(
            "Pass either --pretrained-checkpoint OR --pretrain-from-mlflow, "
            "not both."
        )
        raise typer.Exit(1)

    if pretrain_from_mlflow:
        pretrained_checkpoint = _fetch_pretrained_from_mlflow(
            model_name=model_name,
            alias="champion",
            cache_dir=pretrained_cache_dir,
        )
    elif pretrained_checkpoint is None:
        logger.error(
            "No pretrained checkpoint source. Pass --pretrained-checkpoint "
            "<path> or --pretrain-from-mlflow."
        )
        raise typer.Exit(1)
    else:
        pretrained_checkpoint = Path(pretrained_checkpoint).resolve()
        if not pretrained_checkpoint.exists():
            logger.error("Pretrained checkpoint not found: %s", pretrained_checkpoint)
            raise typer.Exit(code=1)

    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    preprocess_done = train_dir.exists() and val_dir.exists()

    if skip_preprocess and preprocess_done:
        logger.info("Skipping preprocessing — output dirs already exist")
    else:
        logger.info("=" * 60)
        logger.info("STEP 1/2 — Preprocessing (split + augmentation)")
        logger.info("=" * 60)
        logger.info("  raw_data_dir : %s", raw_data_dir)
        logger.info("  train_dir    : %s", train_dir)
        logger.info("  val_dir      : %s", val_dir)
        logger.info("  test_dir     : %s", test_dir)

        raw_train_tmp = output_dir / "preprocessed" / "_train_raw_tmp"

        logger.info(
            "Splitting dataset (%.0f/%.0f/%.0f)...",
            train_ratio * 100,
            val_ratio * 100,
            (1 - train_ratio - val_ratio) * 100,
        )

        stratified_split(
            input_dir=raw_data_dir,
            train_dir=raw_train_tmp,
            val_dir=val_dir,
            test_dir=test_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )

        logger.info("Augmenting train split (%d copies per image)...", augment_copies)
        augment_dir(
            source_dir=raw_train_tmp,
            output_dir=train_dir,
            copies=augment_copies,
            img_size=img_size,
            seed=seed,
        )

        shutil.rmtree(raw_train_tmp, ignore_errors=True)
        logger.info("Preprocessing complete.")

    # ── Step 2: Training ───────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2/2 — Training")
    logger.info("=" * 60)

    paths = TrainingPaths(
        data_dir=train_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )

    config = TrainingConfig.from_mapping(
        {
            "img_size": img_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "target_accuracy": target_accuracy,
            "seed": seed,
            "expected_num_classes": expected_num_classes,
            "num_workers": num_workers,
            "f1_threshold": f1_threshold,
            "use_mlflow": mlflow_enabled,
        }
    )

    training_result, evaluation_summary, gate_result = run_training_workflow(
        paths=paths,
        config=config,
        val_dir=val_dir,
    )

    # ── Step 3 (optional): Register as @candidate in MLflow ────────────────
    candidate_version: Optional[str] = None
    if register_as_candidate:
        if gate_result.passed:
            candidate_version = _register_as_candidate(
                checkpoint_path=paths.best_model_path,
                model_name=model_name,
                alias="candidate",
                val_acc=training_result.best_val_accuracy,
                f1_macro=evaluation_summary.f1_macro,
                run_name=f"local-pipeline-{output_dir.name}",
            )
        else:
            logger.warning(
                "Skipping candidate registration — gate did not pass "
                "(f1_macro=%.4f < threshold=%.4f).",
                evaluation_summary.f1_macro,
                f1_threshold,
            )

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("LOCAL PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("  best_val_accuracy  : %.4f", training_result.best_val_accuracy)
    logger.info("  epochs_trained     : %d", training_result.epochs_trained)
    logger.info("  final_accuracy     : %.4f", evaluation_summary.accuracy)
    logger.info("  final_f1_macro     : %.4f", evaluation_summary.f1_macro)
    logger.info("  gate_passed        : %s", gate_result.passed)
    logger.info(
        "  registered_version : %s",
        gate_result.registered_version or "N/A (local)",
    )
    if register_as_candidate:
        logger.info(
            "  mlflow_candidate   : %s",
            f"{model_name} v{candidate_version}" if candidate_version else "skipped",
        )
    logger.info("  checkpoint         : %s", paths.best_model_path)
    logger.info("  results            : %s", results_dir)


if __name__ == "__main__":
    app()
