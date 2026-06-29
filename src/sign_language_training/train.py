"""Main training workflow and CLI entrypoint.

Run as a module::

    python -m sign_language_training.train \\
        --data-dir /mnt/.../INPUT_data \\
        --pretrained-checkpoint /mnt/.../INPUT_pretrained_checkpoint/best_ngt_model_v2.pth \\
        --checkpoints-root ./outputs/checkpoints \\
        --results-root ./outputs/results \\
        --epochs 30 \\
        --batch-size 16

All hyperparameters not provided on the command line fall back to the
defaults in ``sign_language_training.settings``.
"""

from __future__ import annotations

from typing import Optional, Any
import typer
import json
import logging
import sys
from pathlib import Path

from sign_language_training.configuration import TrainingConfig, TrainingPaths
from sign_language_training.data_loading import (
    create_dataloaders,
    create_presplit_dataloaders,
    create_stratified_split,
    load_dataset,
)
from sign_language_training.logging_utils import log_training_summary
from sign_language_training.model_definitions import (
    build_model_from_pretrained,
    count_parameters,
)
from sign_language_training.model_evaluation import (
    EvaluationSummary,
    collect_predictions,
    load_best_model_state,
    save_evaluation_summary,
    summarize_predictions,
)
from sign_language_training.model_registration import (
    GateResult,
    run_model_gate_and_register,
)
from sign_language_training.model_training import TrainingResult, train_model
from sign_language_training.preprocessing import (
    create_train_transform,
    create_val_transform,
)
from sign_language_training.runtime import get_device, set_seed

logger = logging.getLogger(__name__)

TRAINING_ACCURACY_PLOT_NAME = "training_accuracy.png"
TRAINING_LOSS_PLOT_NAME = "training_loss.png"


def log_device_info() -> None:
    """Log full CUDA and device diagnostics so GPU usage is verifiable in job logs.

    Prints the PyTorch version, CUDA availability, CUDA version, GPU count,
    per-GPU name and VRAM, and the currently active GPU. Logs a warning if
    CUDA is not available to help diagnose misconfigured Azure ML job instances.
    """
    import torch

    logger.info("=" * 60)
    logger.info("DEVICE DIAGNOSTICS")
    logger.info("torch version      : %s", torch.__version__)
    logger.info("CUDA available     : %s", torch.cuda.is_available())
    logger.info("CUDA version       : %s", torch.version.cuda)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info("GPU count          : %d", device_count)
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                "GPU %d             : %s (%d MiB VRAM)",
                i,
                props.name,
                props.total_memory // (1024**2),
            )
        current = torch.cuda.current_device()
        logger.info(
            "Active GPU         : %d (%s)",
            current,
            torch.cuda.get_device_name(current),
        )
    else:
        logger.warning(
            "CUDA NOT available — training will run on CPU. "
            "Verify that the job is using the 'gpu' instance type on lambda-2 "
            "and that the environment has a CUDA-enabled PyTorch build."
        )
    logger.info("=" * 60)


class MLflowExperimentRun:
    """Adapter matching the training loop's :class:`~sign_language_training.model_training.ExperimentRun` protocol.

    Wraps the MLflow module to provide per-epoch metric logging and artifact
    upload via a minimal interface so the training loop is not coupled to
    MLflow directly.

    Args:
        mlflow_module: The imported ``mlflow`` module.
    """

    def __init__(self, mlflow_module: Any) -> None:
        """Create an experiment-run adapter.

        Args:
            mlflow_module: Imported MLflow module used for tracking calls.
        """
        self._mlflow = mlflow_module

    def log(self, data: dict[str, int | float]) -> None:
        """Log a dictionary of scalar metrics to the active MLflow run.

        Extracts ``epoch`` from ``data`` as the step value and logs all
        other numeric entries as MLflow metrics.

        Args:
            data: Dict mapping metric names to scalar values. The special
                ``"epoch"`` key is used as the MLflow step and excluded from
                the logged metrics.
        """
        step = int(data["epoch"]) if "epoch" in data else None
        metrics = {
            key: float(value)
            for key, value in data.items()
            if key != "epoch" and isinstance(value, (int, float))
        }
        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an MLflow artifact.

        Args:
            local_path: Local filesystem path to the file to upload.
            artifact_path: Optional subdirectory path within the MLflow
                artifact store.
        """
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def finish(self) -> None:
        """End the active MLflow run."""
        self._mlflow.end_run()


def _start_mlflow_run(config: TrainingConfig) -> MLflowExperimentRun | None:
    """Start an MLflow run if MLflow tracking is enabled in the config.

    When running inside an Azure ML job, resumes the injected run via
    ``MLFLOW_RUN_ID`` instead of creating a new one. Enables autologging
    if ``config.mlflow_autolog`` is set, and logs all hyperparameters as
    MLflow params.

    Args:
        config: :class:`~sign_language_training.configuration.TrainingConfig`
            controlling MLflow behaviour.

    Returns:
        An :class:`MLflowExperimentRun` adapter if MLflow is enabled,
            otherwise ``None``.
    """
    if not config.use_mlflow:
        return None

    import os
    import mlflow

    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    azure_run_id = os.environ.get("MLFLOW_RUN_ID")
    if azure_run_id:
        logger.info("Resuming Azure ML injected MLflow run: %s", azure_run_id)
        mlflow.start_run(run_id=azure_run_id)
    else:
        mlflow.set_experiment(config.mlflow_experiment_name)
        if mlflow.active_run() is not None:
            mlflow.end_run()
        mlflow.start_run(run_name=config.mlflow_run_name)

    if config.mlflow_autolog:
        mlflow.pytorch.autolog(log_models=False)

    mlflow.log_params(config.to_mlflow_params())
    return MLflowExperimentRun(mlflow)


def _save_training_accuracy_plot(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    """Save a training and validation accuracy plot to disk.

    Args:
        history: Training history dict containing ``train_acc`` and
            ``val_acc`` lists.
        output_path: Destination path for the saved PNG plot.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(output_path)
    plt.close(fig)


def _save_training_loss_plot(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    """Save a training and validation loss plot to disk.

    Args:
        history: Training history dict containing ``train_loss`` and
            ``val_loss`` lists.
        output_path: Destination path for the saved PNG plot.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(output_path)
    plt.close(fig)


def _save_class_names(class_names: list[str], path: str | Any) -> None:
    """Save the class names list to a JSON file.

    Args:
        class_names: Ordered list of class label strings to save.
        path: Destination path for the JSON file.
    """
    logger.info("Saving class names to %s", path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(class_names, handle, indent=2)


def run_training_workflow(
    paths: TrainingPaths,
    config: TrainingConfig,
    val_dir: Path | None = None,
) -> tuple[TrainingResult, EvaluationSummary, GateResult]:
    """Run the full NGT training workflow from data loading to model registration.

    Executes the following steps in order:

    1. Validates paths and config, seeds the RNG, and selects the device.
    2. Loads data — either from pre-split train/val directories (when
       ``val_dir`` is provided) or from a single raw directory with an
       internal stratified split.
    3. Builds the EfficientNet-B0 model from the pretrained checkpoint.
    4. Trains the model with early stopping, saving the best checkpoint.
    5. Evaluates on the validation or test split.
    6. Logs final metrics and optionally saves plots and artifacts to MLflow.
    7. Runs the model gate check and conditionally registers the model in
       Azure ML.

    Args:
        paths: :class:`~sign_language_training.configuration.TrainingPaths`
            providing all filesystem paths. When ``val_dir`` is provided,
            ``paths.data_dir`` must point to the training split only.
        config: :class:`~sign_language_training.configuration.TrainingConfig`
            providing all hyperparameters and MLflow settings.
        val_dir: Optional pre-split validation directory. When provided,
            the internal stratified split is skipped and this directory is used
            directly. A sibling ``test/`` directory is used for final evaluation
            if it exists.

    Returns:
        A three-tuple of ``(training_result, evaluation_summary,
            gate_result)``.

    Raises:
        FileNotFoundError: If required input paths do not exist.
        ValueError: If the dataset class count does not match
            ``config.expected_num_classes``, or if class names differ between
            train and validation directories.
    """
    config.validate()
    paths.validate_inputs()
    paths.ensure_directories()

    log_device_info()
    set_seed(config.seed)
    device = get_device()
    logger.info("Training device: %s", device)

    train_transform = create_train_transform(config.img_size)
    val_transform = create_val_transform(config.img_size)

    def _validate_class_count(class_names: list[str]) -> int:
        """Validate and return the number of discovered classes.

        Args:
            class_names: Class names discovered from the dataset.

        Returns:
            Number of discovered classes.

        Raises:
            ValueError: If the count differs from the configured expectation.
        """
        num_classes = len(class_names)
        if num_classes != config.expected_num_classes:
            raise ValueError(
                f"Expected {config.expected_num_classes} NGT classes, found {num_classes}."
            )
        return num_classes

    if val_dir is not None:
        logger.info("Pre-split mode: using separate train/val directories")
        logger.info("  train_dir : %s", paths.data_dir)
        logger.info("  val_dir   : %s", val_dir)
        train_loader, val_loader, class_names = create_presplit_dataloaders(
            train_dir=paths.data_dir,
            val_dir=val_dir,
            train_transform=train_transform,
            val_transform=val_transform,
            config=config,
        )
        full_dataset_size = len(train_loader.dataset) + len(val_loader.dataset)  # type: ignore[arg-type]
        evaluation_loader = val_loader
        test_dir = val_dir.parent / "test"
        if test_dir.exists():
            _, test_loader, test_class_names = create_presplit_dataloaders(
                train_dir=paths.data_dir,
                val_dir=test_dir,
                train_transform=train_transform,
                val_transform=val_transform,
                config=config,
            )
            if test_class_names != class_names:
                raise ValueError(
                    f"Class mismatch between train and test directories.\n"
                    f"  train: {class_names}\n"
                    f"  test : {test_class_names}"
                )
            evaluation_loader = test_loader
            full_dataset_size += len(test_loader.dataset)  # type: ignore[arg-type]
            logger.info("Final evaluation will use held-out test_dir: %s", test_dir)
    else:
        full_dataset, class_names, targets = load_dataset(paths.data_dir)
        full_dataset_size = len(full_dataset)

        logger.info("Training data directory  : %s", paths.data_dir)
        logger.info("Pretrained checkpoint    : %s", paths.pretrained_checkpoint)
        logger.info("Detected class folders   : %s", class_names)

        _validate_class_count(class_names)

        train_idx, val_idx = create_stratified_split(
            targets=targets,
            n_splits=config.n_splits,
            split_ratio=config.val_split,
            seed=config.seed,
        )
        train_loader, val_loader = create_dataloaders(
            data_dir=paths.data_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            train_transform=train_transform,
            val_transform=val_transform,
            config=config,
        )
        evaluation_loader = val_loader

    num_classes = _validate_class_count(class_names)

    model = build_model_from_pretrained(
        pretrained_checkpoint_path=paths.pretrained_checkpoint,
        device=device,
        num_ngt_classes=num_classes,
    )

    mlflow_run = _start_mlflow_run(config)
    gate_result: GateResult | None = None

    try:
        training_result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            checkpoint_path=paths.best_model_path,
            history_path=paths.history_path,
            class_names=class_names,
            experiment_run=mlflow_run,
        )

        load_best_model_state(model, paths.best_model_path, device)
        prediction_result = collect_predictions(model, evaluation_loader, device)
        evaluation_summary = summarize_predictions(prediction_result, class_names)

        if mlflow_run is not None:
            mlflow_run.log(
                {
                    "final/accuracy": evaluation_summary.accuracy,
                    "final/f1_macro": evaluation_summary.f1_macro,
                    "final/f1_weighted": evaluation_summary.f1_weighted,
                    "final/precision_macro": evaluation_summary.precision_macro,
                    "final/recall_macro": evaluation_summary.recall_macro,
                    "best_val_accuracy": training_result.best_val_accuracy,
                    "epochs_trained": training_result.epochs_trained,
                }
            )

        save_evaluation_summary(
            evaluation_summary=evaluation_summary,
            metrics_path=paths.metrics_path,
            report_path=paths.report_path,
        )
        _save_class_names(class_names, paths.class_names_path)

        if mlflow_run is not None and config.mlflow_log_artifacts:
            accuracy_plot_path = paths.results_dir / TRAINING_ACCURACY_PLOT_NAME
            loss_plot_path = paths.results_dir / TRAINING_LOSS_PLOT_NAME
            _save_training_accuracy_plot(training_result.history, accuracy_plot_path)
            _save_training_loss_plot(training_result.history, loss_plot_path)
            mlflow_run.log_artifact(str(paths.best_model_path), "checkpoints")
            mlflow_run.log_artifact(str(paths.history_path), "training")
            mlflow_run.log_artifact(str(accuracy_plot_path), "training")
            mlflow_run.log_artifact(str(loss_plot_path), "training")
            mlflow_run.log_artifact(str(paths.metrics_path), "evaluation")
            mlflow_run.log_artifact(str(paths.report_path), "evaluation")

        gate_result = run_model_gate_and_register(
            evaluation_summary=evaluation_summary,
            model_path=paths.best_model_path,
            model_name=config.model_name,
            class_names=class_names,
            accuracy_threshold=config.target_accuracy,
            f1_threshold=config.f1_threshold,
        )
        logger.info("%s", gate_result)

        if mlflow_run is not None:
            mlflow_run.log({"gate/passed": float(gate_result.passed)})
            if gate_result.registered_version is not None:
                logger.info(
                    "Model registered as '%s' version %s",
                    config.model_name,
                    gate_result.registered_version,
                )

    finally:
        if mlflow_run is not None:
            mlflow_run.finish()

    assert gate_result is not None

    total_parameters, trainable_parameters, _ = count_parameters(model)
    log_training_summary(
        paths=paths,
        config=config,
        dataset_size=full_dataset_size,
        num_classes=num_classes,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        training_result=training_result,
        evaluation_summary=evaluation_summary,
    )

    return training_result, evaluation_summary, gate_result


app = typer.Typer(
    name="sign-language-training",
    help="Train the NGT sign-language EfficientNet-B0 model.",
    add_completion=False,
)


def _configure_logging() -> None:
    """Configure basic console logging for CLI execution.

    Sets the root logger to ``INFO`` level with a timestamped format
    writing to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


@app.command()
def train(
    data_dir: Path = typer.Option(
        ...,
        "--data-dir",
        help="Path to the training ImageFolder root.",
        metavar="PATH",
    ),
    val_dir: Optional[Path] = typer.Option(
        None,
        "--val-dir",
        help=(
            "Optional validation ImageFolder root. "
            "When provided, --data-dir is treated as the training split "
            "and the internal stratified split is skipped."
        ),
        metavar="PATH",
    ),
    pretrained_checkpoint: Path = typer.Option(
        ...,
        "--pretrained-checkpoint",
        help="Path to the pretrained .pth checkpoint.",
        metavar="PATH",
    ),
    checkpoint_dir: Path = typer.Option(
        ...,
        "--checkpoint-dir",
        "--checkpoints-root",
        help=(
            "Root directory for run checkpoint subfolders. "
            "--checkpoints-root is kept as a backward-compatible alias."
        ),
        metavar="PATH",
    ),
    results_dir: Path = typer.Option(
        ...,
        "--results-dir",
        "--results-root",
        help=(
            "Root directory for run results subfolders. "
            "--results-root is kept as a backward-compatible alias."
        ),
        metavar="PATH",
    ),
    run_name: Optional[str] = typer.Option(
        None,
        "--run-name",
        help="Explicit run name. Defaults to auto-generated model_N.",
    ),
    img_size: Optional[int] = typer.Option(
        None,
        "--img-size",
        help="Input image size. Defaults to TRAINING_IMG_SIZE.",
        min=1,
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Training batch size. Defaults to TRAINING_BATCH_SIZE.",
        min=1,
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "--learning-rate",
        help="Optimizer learning rate. Defaults to TRAINING_LEARNING_RATE.",
        min=0.0,
    ),
    epochs: Optional[int] = typer.Option(
        None,
        "--epochs",
        help="Maximum training epochs. Defaults to TRAINING_EPOCHS.",
        min=1,
    ),
    patience: Optional[int] = typer.Option(
        None,
        "--patience",
        help="Early stopping patience. Defaults to TRAINING_PATIENCE.",
        min=1,
    ),
    val_split: Optional[float] = typer.Option(
        None,
        "--val-split",
        help="Validation split ratio. Defaults to TRAINING_VAL_SPLIT.",
        min=0.01,
        max=0.99,
    ),
    target_accuracy: Optional[float] = typer.Option(
        None,
        "--target-accuracy",
        help="Accuracy threshold used by the model gate.",
        min=0.0,
        max=1.0,
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed. Defaults to TRAINING_SEED.",
    ),
    num_workers: Optional[int] = typer.Option(
        None,
        "--num-workers",
        help="DataLoader worker count. Defaults to TRAINING_NUM_WORKERS.",
        min=0,
    ),
    expected_num_classes: Optional[int] = typer.Option(
        None,
        "--expected-num-classes",
        help="Expected number of class folders.",
        min=1,
    ),
    f1_threshold: Optional[float] = typer.Option(
        None,
        "--f1-threshold",
        help="F1 macro threshold used by the model gate.",
        min=0.0,
        max=1.0,
    ),
) -> None:
    """Run the full NGT EfficientNet-B0 training workflow from the CLI.

    Builds :class:`~sign_language_training.configuration.TrainingPaths` and
    :class:`~sign_language_training.configuration.TrainingConfig` from the
    provided CLI options (falling back to project settings for any ``None``
    values), then delegates to :func:`run_training_workflow`.

    Args:
        data_dir: Path to the training ImageFolder root, or the training
            split when ``--val-dir`` is also provided.
        val_dir: Optional pre-split validation ImageFolder root. Skips
            the internal stratified split when provided.
        pretrained_checkpoint: Path to the pretrained ``.pth`` checkpoint.
        checkpoint_dir: Root directory for per-run checkpoint
            subdirectories.
        results_dir: Root directory for per-run results subdirectories.
        run_name: Explicit run name override. Auto-generated if ``None``.
        img_size: Input image size in pixels override.
        batch_size: Training batch size override.
        learning_rate: Optimiser learning rate override.
        epochs: Maximum training epoch count override.
        patience: Early stopping patience override.
        val_split: Validation split ratio override.
        target_accuracy: Model gate accuracy threshold override.
        seed: Random seed override.
        num_workers: DataLoader worker count override.
        expected_num_classes: Expected class folder count override.
        f1_threshold: Model gate macro F1 threshold override.

    Raises:
        typer.Exit: With code ``2`` if a :exc:`FileNotFoundError` or
            :exc:`ValueError` is raised during the training workflow.
    """
    _configure_logging()

    paths = TrainingPaths.for_run(
        data_dir=data_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoints_root=checkpoint_dir,
        results_root=results_dir,
        run_name=run_name,
    )

    config_overrides = {
        "img_size": img_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "val_split": val_split,
        "target_accuracy": target_accuracy,
        "seed": seed,
        "num_workers": num_workers,
        "expected_num_classes": expected_num_classes,
        "f1_threshold": f1_threshold,
    }
    config = TrainingConfig.from_mapping(config_overrides)

    logger.info("Starting NGT training")
    logger.info("  data_dir              : %s", paths.data_dir)
    logger.info("  pretrained_checkpoint : %s", paths.pretrained_checkpoint)
    logger.info("  checkpoint_dir        : %s", paths.checkpoint_dir)
    logger.info("  results_dir           : %s", paths.results_dir)
    logger.info("  batch_size            : %d", config.batch_size)
    logger.info("  epochs                : %d", config.epochs)
    logger.info("  learning_rate         : %s", config.learning_rate)
    logger.info("  img_size              : %d", config.img_size)
    logger.info("  val_split             : %.2f", config.val_split)
    logger.info("  seed                  : %d", config.seed)
    logger.info("  mlflow_enabled        : %s", config.use_mlflow)
    if config.use_mlflow:
        logger.info("  mlflow_tracking_uri   : %s", config.mlflow_tracking_uri)
        logger.info("  mlflow_experiment     : %s", config.mlflow_experiment_name)

    try:
        training_result, evaluation_summary, gate_result = run_training_workflow(
            paths,
            config,
            val_dir=val_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Training failed: %s", exc)
        raise typer.Exit(code=2) from exc

    logger.info("Training complete.")
    logger.info("  best_val_accuracy : %.4f", training_result.best_val_accuracy)
    logger.info("  epochs_trained    : %d", training_result.epochs_trained)
    logger.info("  final_accuracy    : %.4f", evaluation_summary.accuracy)
    logger.info("  final_f1_macro    : %.4f", evaluation_summary.f1_macro)
    logger.info("  gate_passed       : %s", gate_result.passed)

    if gate_result.registered_version:
        logger.info("  registered_version: %s", gate_result.registered_version)


def main() -> None:
    """Package entrypoint registered in ``pyproject.toml``.

    Delegates directly to the Typer :data:`app`.
    """
    app()


if __name__ == "__main__":
    main()
