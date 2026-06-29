"""Command-line interface.

Provides four subcommands:

- ``predict``        — run inference on a single image file
- ``serve``          — start the FastAPI server with uvicorn
- ``train``          — run the NGT training workflow
- ``local-pipeline`` — run the full local retraining pipeline (preprocess → train)

Examples
--------
Run inference on an image::

    sign-language predict --image hand.jpg

Run inference with a custom model checkpoint::

    sign-language predict --image hand.jpg --model models/best_ngt_model_v2.pth

Start the API server with defaults::

    sign-language serve

Start the API server for production::

    sign-language serve --host 0.0.0.0 --port 8080 --workers 4

Start the API server in development mode::

    sign-language serve --reload

Run the full local retraining pipeline::

    sign-language local-pipeline \\
        --pretrained-checkpoint models/best_ngt_model_v2.pth

Run the local pipeline with overrides::

    sign-language local-pipeline \\
        --raw-data-dir data/sample \\
        --pretrained-checkpoint models/best_ngt_model_v2.pth \\
        --epochs 5 --batch-size 8
"""

import base64
import json
import logging
from pathlib import Path
from typing import Any, Optional

import typer
import uvicorn

from sign_language.core.inference import run_inference
from sign_language.core.logging import get_logger
from sign_language.core.preprocessing import preprocess_image
from sign_language.core.settings import settings
from sign_language.models.loader import load_all

logger = get_logger(__name__)

TrainingConfig: Any = None
TrainingPaths: Any = None
run_training_workflow: Any = None

app = typer.Typer(
    name="sign-language",
    help=(
        "Sign Language Recognition CLI - classify NGT hand signs "
        "from images or run the API server."
    ),
    add_completion=False,
)

DEFAULT_TRAINING_RESULTS_DIR = Path("logs/training")
DEFAULT_TRAINING_CHECKPOINT_DIR = DEFAULT_TRAINING_RESULTS_DIR


def _image_to_base64(image_path: Path) -> str:
    """Read an image file from disk and encode it as a base64 string.

    :param image_path: Path to a JPEG or PNG image file.
    :returns: Base64-encoded string of the raw image bytes.
    :raises typer.BadParameter: If the file does not exist or has an
        unsupported extension (``.jpg``, ``.jpeg``, or ``.png``).
    """
    if not image_path.exists():
        raise typer.BadParameter(f"Image file not found: {image_path}")

    supported = {".jpg", ".jpeg", ".png"}
    if image_path.suffix.lower() not in supported:
        raise typer.BadParameter(
            f"Unsupported file type '{image_path.suffix}'. "
            f"Supported formats: {', '.join(supported)}"
        )

    logger.debug("Reading image from %s", image_path)
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _configure_training_console_logging() -> None:
    """Attach a console handler to the ``sign_language_training`` logger.

    Ensures training progress is printed to stdout when the ``train`` or
    ``local-pipeline`` commands are run from the CLI. Skips attaching a
    second handler if one is already present to avoid duplicate output.
    """
    training_logger = logging.getLogger("sign_language_training")
    training_logger.setLevel(logging.INFO)

    if not any(
        isinstance(handler, logging.StreamHandler)
        for handler in training_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        training_logger.addHandler(handler)

    training_logger.propagate = True


def _load_training_dependencies() -> tuple[Any, Any, Any]:
    """Import and cache training package dependencies on first call.

    Defers importing ``sign_language_training`` until the ``train`` command
    actually runs to avoid pulling in torch and heavy training dependencies
    on every CLI invocation.

    :returns: A three-tuple of ``(TrainingConfig, TrainingPaths,
        run_training_workflow)`` from the ``sign_language_training`` package.
    """
    global TrainingConfig, TrainingPaths, run_training_workflow

    if TrainingConfig is None or TrainingPaths is None or run_training_workflow is None:
        from sign_language_training.configuration import (
            TrainingConfig as ImportedTrainingConfig,
            TrainingPaths as ImportedTrainingPaths,
        )
        from sign_language_training.train import (
            run_training_workflow as imported_run_training_workflow,
        )

        TrainingConfig = ImportedTrainingConfig
        TrainingPaths = ImportedTrainingPaths
        run_training_workflow = imported_run_training_workflow

    return TrainingConfig, TrainingPaths, run_training_workflow


@app.command()
def train(
    pretrained_checkpoint: Path = typer.Option(
        ...,
        "--pretrained-checkpoint",
        help="Path to the pretrained EfficientNet checkpoint used as the fine-tuning starting point.",
        metavar="PATH",
    ),
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Override training data directory. Defaults to settings.get_training_data_dir().",
        metavar="PATH",
    ),
    checkpoint_dir: Path = typer.Option(
        DEFAULT_TRAINING_CHECKPOINT_DIR,
        "--checkpoint-dir",
        help="Directory where per-run NGT checkpoints are written.",
        metavar="PATH",
    ),
    results_dir: Path = typer.Option(
        DEFAULT_TRAINING_RESULTS_DIR,
        "--results-dir",
        help="Directory where training history, metrics, report, and plots are written.",
        metavar="PATH",
    ),
    batch_size: int = typer.Option(
        settings.training_batch_size,
        "--batch-size",
        help="Training batch size.",
        min=1,
    ),
    epochs: int = typer.Option(
        settings.training_epochs,
        "--epochs",
        help="Maximum number of training epochs.",
        min=1,
    ),
    learning_rate: float = typer.Option(
        settings.training_learning_rate,
        "--learning-rate",
        help="Optimizer learning rate.",
        min=0.0,
    ),
    eta_min: float = typer.Option(
        settings.training_eta_min,
        "--eta-min",
        help="Minimum learning rate for cosine scheduler.",
        min=0.0,
    ),
    img_size: int = typer.Option(
        settings.training_img_size,
        "--img-size",
        help="Input image size for training transforms.",
        min=1,
    ),
    val_split: float = typer.Option(
        settings.training_val_split,
        "--val-split",
        help="Validation split ratio.",
        min=0.01,
        max=0.99,
    ),
    seed: int = typer.Option(
        settings.training_seed,
        "--seed",
        help="Random seed for deterministic split/training setup.",
    ),
    patience: int = typer.Option(
        settings.training_patience,
        "--patience",
        help="Early stopping patience.",
        min=1,
    ),
    target_accuracy: float = typer.Option(
        settings.training_target_accuracy,
        "--target-accuracy",
        help="Target validation accuracy.",
        min=0.0,
        max=1.0,
    ),
    expected_num_classes: int = typer.Option(
        settings.training_expected_num_classes,
        "--expected-num-classes",
        help="Expected number of class folders in the dataset.",
        min=1,
    ),
    n_splits: int = typer.Option(
        settings.training_n_splits,
        "--n-splits",
        help="Number of stratified splits. Current workflow supports only 1.",
        min=1,
    ),
    num_workers: int = typer.Option(
        settings.training_num_workers,
        "--num-workers",
        help="DataLoader worker process count.",
        min=0,
    ),
    pin_memory: bool = typer.Option(
        settings.training_pin_memory,
        "--pin-memory/--no-pin-memory",
        help="Whether DataLoader should use pinned memory.",
    ),
    mlflow_enabled: bool = typer.Option(
        settings.mlflow_enabled,
        "--mlflow-enabled/--no-mlflow-enabled",
        help="Enable MLflow tracking.",
    ),
    mlflow_tracking_uri: Optional[str] = typer.Option(
        settings.mlflow_tracking_uri,
        "--mlflow-tracking-uri",
        help="Optional MLflow tracking URI. Empty uses MLflow default/Azure configured tracking.",
    ),
    mlflow_experiment_name: str = typer.Option(
        settings.mlflow_experiment_name,
        "--mlflow-experiment-name",
        help="MLflow experiment name.",
    ),
    mlflow_run_name: Optional[str] = typer.Option(
        settings.mlflow_run_name,
        "--mlflow-run-name",
        help="Optional MLflow run name.",
    ),
    mlflow_autolog: bool = typer.Option(
        settings.mlflow_autolog,
        "--mlflow-autolog/--no-mlflow-autolog",
        help="Enable MLflow PyTorch autologging.",
    ),
    mlflow_log_artifacts: bool = typer.Option(
        settings.mlflow_log_artifacts,
        "--mlflow-log-artifacts/--no-mlflow-log-artifacts",
        help="Log checkpoints, metrics, reports, and plots as MLflow artifacts.",
    ),
) -> None:
    """Run the NGT sign-language training workflow.

    Loads training dependencies lazily, resolves the data directory from
    the CLI option or project settings, builds :class:`TrainingPaths` and
    :class:`TrainingConfig` from the provided options, and runs the full
    training workflow including evaluation and model gate check.

    Prints a summary of resolved paths and hyperparameters before training
    starts, and a results summary on completion.

    :param pretrained_checkpoint: Path to the pretrained EfficientNet
        ``.pth`` checkpoint used as the fine-tuning starting point.
    :param data_dir: Training data directory override. Defaults to
        ``settings.get_training_data_dir()``.
    :param checkpoint_dir: Directory for per-run model checkpoints.
    :param results_dir: Directory for training history, metrics, and plots.
    :param batch_size: Number of samples per training batch.
    :param epochs: Maximum number of training epochs.
    :param learning_rate: Initial optimiser learning rate.
    :param eta_min: Minimum learning rate for the cosine annealing scheduler.
    :param img_size: Input image size in pixels for training transforms.
    :param val_split: Fraction of the dataset to use for validation.
    :param seed: Random seed for reproducible splitting and training.
    :param patience: Early stopping patience in epochs.
    :param target_accuracy: Target validation accuracy for the gate check.
    :param expected_num_classes: Expected number of class folders in the
        dataset.
    :param n_splits: Number of stratified splits (currently must be 1).
    :param num_workers: Number of DataLoader worker processes.
    :param pin_memory: Whether to use pinned memory in the DataLoader.
    :param mlflow_enabled: Whether to enable MLflow experiment tracking.
    :param mlflow_tracking_uri: MLflow tracking URI override. Uses the
        MLflow default or Azure-configured URI if not set.
    :param mlflow_experiment_name: MLflow experiment name.
    :param mlflow_run_name: Optional MLflow run name.
    :param mlflow_autolog: Whether to enable MLflow PyTorch autologging.
    :param mlflow_log_artifacts: Whether to log training outputs as MLflow
        artifacts.
    :raises typer.Exit: With code ``1`` if a :exc:`FileNotFoundError` or
        :exc:`ValueError` is raised during training.
    """
    training_config_cls, training_paths_cls, run_workflow = (
        _load_training_dependencies()
    )

    resolved_data_dir = data_dir or settings.get_training_data_dir()

    paths = training_paths_cls.for_run(
        data_dir=resolved_data_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoints_root=checkpoint_dir,
        results_root=results_dir,
    )

    config = training_config_cls.from_mapping(
        {
            "img_size": img_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "val_split": val_split,
            "target_accuracy": target_accuracy,
            "seed": seed,
            "n_splits": n_splits,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "eta_min": eta_min,
            "expected_num_classes": expected_num_classes,
            "use_mlflow": mlflow_enabled,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
            "mlflow_autolog": mlflow_autolog,
            "mlflow_log_artifacts": mlflow_log_artifacts,
        }
    )

    typer.echo("Starting NGT training:")
    typer.echo(f"  data_dir: {paths.data_dir}")
    typer.echo(f"  pretrained_checkpoint: {paths.pretrained_checkpoint}")
    typer.echo(f"  checkpoint_dir: {paths.checkpoint_dir}")
    typer.echo(f"  results_dir: {paths.results_dir}")
    typer.echo(f"  batch_size: {config.batch_size}")
    typer.echo(f"  epochs: {config.epochs}")
    typer.echo(f"  learning_rate: {config.learning_rate}")
    typer.echo(f"  img_size: {config.img_size}")
    typer.echo(f"  val_split: {config.val_split}")
    typer.echo(f"  seed: {config.seed}")
    typer.echo(f"  mlflow_enabled: {config.use_mlflow}")
    typer.echo(
        f"  mlflow_tracking_uri: {config.mlflow_tracking_uri or '<default/azure/local>'}"
    )
    typer.echo(f"  mlflow_experiment_name: {config.mlflow_experiment_name}")

    _configure_training_console_logging()

    try:
        training_result, evaluation_summary, gate_result = run_workflow(paths, config)
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Training complete:")
    typer.echo(f"  best_val_accuracy: {training_result.best_val_accuracy:.4f}")
    typer.echo(f"  epochs_trained: {training_result.epochs_trained}")
    typer.echo(f"  final_accuracy: {evaluation_summary.accuracy:.4f}")
    typer.echo(f"  checkpoint: {paths.best_model_path}")
    typer.echo(f"  metrics: {paths.metrics_path}")
    typer.echo(f"  gate_passed: {gate_result.passed}")
    typer.echo(
        f"  registered_version: {gate_result.registered_version or 'not registered'}"
    )


@app.command(name="local-pipeline")
def local_pipeline(
    raw_data_dir: Path = typer.Option(
        "data/raw",
        "--raw-data-dir",
        help="Path to the raw ImageFolder dataset.",
        metavar="PATH",
    ),
    output_dir: Path = typer.Option(
        "outputs/local_pipeline",
        "--output-dir",
        help="Root directory for all pipeline outputs.",
        metavar="PATH",
    ),
    pretrained_checkpoint: Path = typer.Option(
        ...,
        "--pretrained-checkpoint",
        help="Path to the pretrained .pth checkpoint.",
        metavar="PATH",
    ),
    augment_copies: int = typer.Option(4, "--augment-copies", min=1),
    img_size: int = typer.Option(settings.training_img_size, "--img-size", min=1),
    seed: int = typer.Option(settings.training_seed, "--seed"),
    train_ratio: float = typer.Option(0.8, "--train-ratio"),
    val_ratio: float = typer.Option(0.1, "--val-ratio"),
    batch_size: int = typer.Option(settings.training_batch_size, "--batch-size", min=1),
    epochs: int = typer.Option(settings.training_epochs, "--epochs", min=1),
    learning_rate: float = typer.Option(
        settings.training_learning_rate, "--learning-rate"
    ),
    patience: int = typer.Option(settings.training_patience, "--patience", min=1),
    target_accuracy: float = typer.Option(
        settings.training_target_accuracy, "--target-accuracy"
    ),
    expected_num_classes: int = typer.Option(
        settings.training_expected_num_classes, "--expected-num-classes", min=1
    ),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    f1_threshold: float = typer.Option(0.80, "--f1-threshold"),
    mlflow_enabled: bool = typer.Option(False, "--mlflow/--no-mlflow"),
    skip_preprocess: bool = typer.Option(
        False,
        "--skip-preprocess",
        help="Skip preprocessing if output dirs already exist.",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Delete existing output directory before running.",
    ),
) -> None:
    """Run the full local retraining pipeline (preprocess → train).

    Replicates the Azure ML pipeline graph locally: stratified split,
    offline augmentation, training, evaluation, and model gate check.

    :param raw_data_dir: Path to the raw ImageFolder dataset root.
    :param output_dir: Root directory for all pipeline outputs.
    :param pretrained_checkpoint: Path to the pretrained ``.pth`` checkpoint.
    :param augment_copies: Number of augmented copies per training image.
    :param img_size: Image size in pixels for preprocessing and training.
    :param seed: Random seed for deterministic splitting and augmentation.
    :param train_ratio: Fraction of the dataset to use for training.
    :param val_ratio: Fraction of the dataset to use for validation.
    :param batch_size: Number of samples per training batch.
    :param epochs: Maximum number of training epochs.
    :param learning_rate: Initial optimiser learning rate.
    :param patience: Early stopping patience in epochs.
    :param target_accuracy: Target validation accuracy for the gate check.
    :param expected_num_classes: Expected number of output classes.
    :param num_workers: Number of DataLoader worker processes.
    :param f1_threshold: Minimum macro F1 score to pass the gate check.
    :param mlflow_enabled: Whether to enable MLflow tracking.
    :param skip_preprocess: Skip Step 1 if preprocessed directories already
        exist from a previous run.
    :param clean: Delete ``output_dir`` before running for a clean start.
    :raises typer.Exit: With code ``1`` if required paths do not exist or
        a training error occurs.
    """
    import shutil

    from sign_language_training.augmentation import augment_dir, stratified_split

    _configure_training_console_logging()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw_data_dir = Path(raw_data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    pretrained_checkpoint = Path(pretrained_checkpoint).resolve()

    train_dir = output_dir / "preprocessed" / "train"
    val_dir = output_dir / "preprocessed" / "val"
    test_dir = output_dir / "preprocessed" / "test"
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    if not raw_data_dir.exists():
        typer.echo(f"Error: Raw data directory not found: {raw_data_dir}", err=True)
        raise typer.Exit(code=1)

    if not pretrained_checkpoint.exists():
        typer.echo(
            f"Error: Pretrained checkpoint not found: {pretrained_checkpoint}",
            err=True,
        )
        raise typer.Exit(code=1)

    if clean and output_dir.exists():
        typer.echo(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    preprocess_done = train_dir.exists() and val_dir.exists()

    if skip_preprocess and preprocess_done:
        typer.echo("Skipping preprocessing — output dirs already exist")
    else:
        typer.echo("=" * 60)
        typer.echo("STEP 1/2 — Preprocessing (split + augmentation)")
        typer.echo("=" * 60)

        raw_train_tmp = output_dir / "preprocessed" / "_train_raw_tmp"

        typer.echo(
            f"Splitting dataset ({train_ratio*100:.0f}/"
            f"{val_ratio*100:.0f}/"
            f"{(1-train_ratio-val_ratio)*100:.0f})..."
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

        typer.echo(f"Augmenting train split ({augment_copies} copies per image)...")
        augment_dir(
            source_dir=raw_train_tmp,
            output_dir=train_dir,
            copies=augment_copies,
            img_size=img_size,
            seed=seed,
        )

        shutil.rmtree(raw_train_tmp, ignore_errors=True)
        typer.echo("Preprocessing complete.")

    typer.echo("=" * 60)
    typer.echo("STEP 2/2 — Training")
    typer.echo("=" * 60)

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

    try:
        training_result, evaluation_summary, gate_result = run_training_workflow(
            paths=paths,
            config=config,
            val_dir=val_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("=" * 60)
    typer.echo("LOCAL PIPELINE COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  best_val_accuracy  : {training_result.best_val_accuracy:.4f}")
    typer.echo(f"  epochs_trained     : {training_result.epochs_trained}")
    typer.echo(f"  final_accuracy     : {evaluation_summary.accuracy:.4f}")
    typer.echo(f"  final_f1_macro     : {evaluation_summary.f1_macro:.4f}")
    typer.echo(f"  gate_passed        : {gate_result.passed}")
    typer.echo(
        f"  registered_version : {gate_result.registered_version or 'N/A (local)'}"
    )
    typer.echo(f"  checkpoint         : {paths.best_model_path}")
    typer.echo(f"  results            : {results_dir}")


@app.command()
def predict(
    image: Path = typer.Option(
        ...,
        "--image",
        "-i",
        help="Path to the input image file (JPEG or PNG).",
        metavar="PATH",
    ),
    model: Optional[Path] = typer.Option(
        None,
        "--model",
        help=(
            "Path to EfficientNet checkpoint (.pth). "
            "Defaults to the path configured in settings."
        ),
        metavar="PATH",
    ),
    lm_model: Optional[Path] = typer.Option(
        None,
        "--lm-model",
        help="Path to Landmark MLP checkpoint (.pth). Optional fallback model.",
        metavar="PATH",
    ),
    landmarker: Optional[Path] = typer.Option(
        None,
        "--landmarker",
        help="Path to MediaPipe hand landmarker task file (.task). Optional.",
        metavar="PATH",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        "-k",
        help="Number of top predictions to include in the output (max 3).",
        min=1,
        max=3,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging output.",
    ),
) -> None:
    """Run inference on a single image and print the result as JSON.

    Loads the sign language models, preprocesses the input image,
    runs inference, and prints the predicted letter with confidence
    scores to stdout as JSON.

    All three model arguments are optional — if not provided, paths from
    project settings are used, allowing custom checkpoints to be tested
    without changing configuration.

    :param image: Path to the input JPEG or PNG image file.
    :param model: Path to the EfficientNet ``.pth`` checkpoint. Defaults
        to ``settings.model_path``.
    :param lm_model: Path to the Landmark MLP ``.pth`` checkpoint. Defaults
        to ``settings.lm_model_path``.
    :param landmarker: Path to the MediaPipe ``hand_landmarker.task`` file.
        Defaults to ``settings.hand_landmarker_path``.
    :param top_k: Number of top-k predictions to include in the JSON output.
    :param verbose: If ``True``, sets the root logger to ``DEBUG`` level.
    :raises typer.Exit: With code ``1`` if the image path is invalid or a
        model file is not found.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        image_b64 = _image_to_base64(image)
    except typer.BadParameter as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    model_path = model or settings.model_path
    lm_model_path = lm_model or settings.lm_model_path
    landmarker_path = landmarker or settings.hand_landmarker_path

    typer.echo("Loading models:")
    try:
        loaded = load_all(
            model_path=model_path,
            lm_model_path=lm_model_path,
            landmarker_path=landmarker_path,
        )
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Preprocessing image:")
    hand_detected, tensor, landmarks_data = preprocess_image(
        image_data=image_b64,
        hands_detector=loaded.hands_detector,
        device=loaded.device,
    )

    if not hand_detected:
        typer.echo(
            "Warning: no hand detected - running inference on full image.",
            err=True,
        )

    typer.echo("Running inference:")
    predicted_letter, confidence, top_3 = run_inference(
        tensor=tensor,
        model=loaded.model,
        class_names=loaded.class_names,
        device=loaded.device,
        landmarks_data=landmarks_data,
        landmark_model=loaded.landmark_model,
        lm_class_names=loaded.lm_class_names,
    )

    result = {
        "image": str(image),
        "hand_detected": hand_detected,
        "prediction": predicted_letter,
        "confidence": round(confidence, 4),
        "top_k": top_3[:top_k],
    }

    typer.echo(json.dumps(result, indent=2))


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help=(
            "Bind socket to this host. "
            "Use 0.0.0.0 to listen on all interfaces (required for Docker)."
        ),
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Bind socket to this port.",
        min=1,
        max=65535,
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        "-w",
        help=(
            "Number of worker processes for production. "
            "Cannot be used together with --reload."
        ),
        min=1,
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help=(
            "Enable auto-reload on code changes for development. "
            "Cannot be used with --workers > 1."
        ),
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Uvicorn log level: debug, info, warning, error, or critical.",
    ),
) -> None:
    """Start the Sign Language Recognition API server with Uvicorn.

    Launches the FastAPI application. Use ``--reload`` for local development
    (auto-restarts on file changes) and ``--workers`` for production
    deployments with multiple processes. These two options are mutually
    exclusive.

    :param host: Host address to bind. Use ``0.0.0.0`` for Docker or
        external access.
    :param port: TCP port to bind the server to.
    :param workers: Number of Uvicorn worker processes. Ignored when
        ``--reload`` is set (forced to 1).
    :param reload: Enable Uvicorn auto-reload for development.
    :param log_level: Uvicorn log level string.
    :raises typer.Exit: With code ``1`` if both ``--reload`` and
        ``--workers > 1`` are specified.
    """
    if reload and workers > 1:
        typer.echo(
            "Error: --reload cannot be used with --workers > 1.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Starting Sign Language API on http://{host}:{port} …")

    uvicorn.run(
        "sign_language.api:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level.lower(),
    )
