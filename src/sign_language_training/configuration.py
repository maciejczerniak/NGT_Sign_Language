"""Training configuration and path helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sign_language_training.settings import settings

logger = logging.getLogger(__name__)

MODEL_CHECKPOINT_FILENAME: str = settings.training_best_checkpoint_name
TRAINING_HISTORY_FILENAME: str = settings.training_history_filename
METRICS_FILENAME: str = settings.training_metrics_filename
CLASSIFICATION_REPORT_FILENAME: str = settings.training_report_filename
CLASS_NAMES_FILENAME: str = settings.training_class_names_filename


@dataclass(frozen=True)
class TrainingPaths:
    """Filesystem paths used by the training workflow.

    All path properties are derived from the four required directories and
    the filename constants, which default to the values from project settings.

    Args:
        data_dir: Root directory of the training ImageFolder dataset.
        pretrained_checkpoint: Path to the pretrained ``.pth`` checkpoint
            used as the fine-tuning starting point.
        checkpoint_dir: Directory where model checkpoints are saved during
            training.
        results_dir: Directory where training history, metrics, classification
            report, and class names are saved.
        best_checkpoint_name: Filename for the best model checkpoint.
        history_filename: Filename for the saved training history JSON.
        metrics_filename: Filename for the saved evaluation metrics JSON.
        report_filename: Filename for the saved classification report text.
        class_names_filename: Filename for the saved class names JSON.
    """

    data_dir: Path
    pretrained_checkpoint: Path
    checkpoint_dir: Path
    results_dir: Path
    best_checkpoint_name: str = settings.training_best_checkpoint_name
    history_filename: str = settings.training_history_filename
    metrics_filename: str = settings.training_metrics_filename
    report_filename: str = settings.training_report_filename
    class_names_filename: str = settings.training_class_names_filename

    @property
    def best_model_path(self) -> Path:
        """Return the full path to the best model checkpoint file.

        Returns:
            ``checkpoint_dir / best_checkpoint_name``.
        """
        return self.checkpoint_dir / self.best_checkpoint_name

    @property
    def history_path(self) -> Path:
        """Return the full path to the training history JSON file.

        Returns:
            ``results_dir / history_filename``.
        """
        return self.results_dir / self.history_filename

    @property
    def metrics_path(self) -> Path:
        """Return the full path to the evaluation metrics JSON file.

        Returns:
            ``results_dir / metrics_filename``.
        """
        return self.results_dir / self.metrics_filename

    @property
    def report_path(self) -> Path:
        """Return the full path to the classification report text file.

        Returns:
            ``results_dir / report_filename``.
        """
        return self.results_dir / self.report_filename

    @property
    def class_names_path(self) -> Path:
        """Return the full path to the class names JSON file.

        Returns:
            ``results_dir / class_names_filename``.
        """
        return self.results_dir / self.class_names_filename

    def ensure_directories(self) -> None:
        """Create the checkpoint and results directories if they do not exist."""
        logger.info("Ensuring checkpoint and results directories exist")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def validate_inputs(self) -> None:
        """Validate that required input paths exist on disk.

        Raises:
            FileNotFoundError: If ``data_dir`` or ``pretrained_checkpoint``
                does not exist.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not self.pretrained_checkpoint.exists():
            raise FileNotFoundError(
                f"Pretrained checkpoint not found: {self.pretrained_checkpoint}"
            )

    @staticmethod
    def for_run(
        data_dir: Path,
        pretrained_checkpoint: Path,
        checkpoints_root: Path,
        results_root: Path,
        run_name: str | None = None,
    ) -> "TrainingPaths":
        """Create a :class:`TrainingPaths` instance for a single training run.

        Generates a unique run name via
        :func:`~sign_language_training.run_naming.generate_run_name` if one
        is not provided, then constructs per-run subdirectories under the
        given root directories.

        Args:
            data_dir: Root directory of the training dataset.
            pretrained_checkpoint: Path to the pretrained checkpoint.
            checkpoints_root: Root directory under which per-run checkpoint
                subdirectories are created.
            results_root: Root directory under which per-run results
                subdirectories are created.
            run_name: Optional run name override. If ``None``, a unique name
                is generated automatically.

        Returns:
            A :class:`TrainingPaths` instance with per-run directories
                set to ``checkpoints_root / run_name`` and
                ``results_root / run_name``.
        """
        from sign_language_training.run_naming import generate_run_name

        if run_name is None:
            run_name = generate_run_name(checkpoints_root)

        checkpoint_dir = checkpoints_root / run_name
        results_dir = results_root / run_name

        logger.info("Checkpoint directory: %s", checkpoint_dir)
        logger.info("Results directory: %s", results_dir)

        return TrainingPaths(
            data_dir=data_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            best_checkpoint_name=settings.training_best_checkpoint_name,
            history_filename=settings.training_history_filename,
            metrics_filename=settings.training_metrics_filename,
            report_filename=settings.training_report_filename,
            class_names_filename=settings.training_class_names_filename,
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters and runtime settings for NGT EfficientNet-B0 fine-tuning.

    All fields default to the corresponding values from project settings and
    can be overridden at construction time or via :meth:`from_mapping`.
    Validation is run automatically after construction via ``__post_init__``.

    Args:
        img_size: Input image size in pixels.
        batch_size: Number of samples per training batch.
        learning_rate: Initial optimiser learning rate.
        epochs: Maximum number of training epochs.
        patience: Early stopping patience in epochs.
        val_split: Fraction of the dataset to use for validation.
        target_accuracy: Minimum validation accuracy for the gate check.
        seed: Random seed for reproducible training.
        n_splits: Number of stratified splits (must be 1).
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to use pinned memory in the DataLoader.
        eta_min: Minimum learning rate for the cosine annealing scheduler.
        expected_num_classes: Expected number of output classes.
        use_mlflow: Whether to enable MLflow experiment tracking.
        mlflow_tracking_uri: MLflow tracking URI override, or ``None`` for
            the default.
        mlflow_experiment_name: MLflow experiment name.
        mlflow_run_name: Optional MLflow run name.
        mlflow_autolog: Whether to enable MLflow PyTorch autologging.
        mlflow_log_artifacts: Whether to log training artifacts to MLflow.
        model_name: Azure ML model registry name used by the gate check.
        f1_threshold: Minimum macro F1 score required to pass the gate check.
    """

    img_size: int = settings.training_img_size
    batch_size: int = settings.training_batch_size
    learning_rate: float = settings.training_learning_rate
    epochs: int = settings.training_epochs
    patience: int = settings.training_patience
    val_split: float = settings.training_val_split
    target_accuracy: float = settings.training_target_accuracy
    seed: int = settings.training_seed
    n_splits: int = settings.training_n_splits
    num_workers: int = settings.training_num_workers
    pin_memory: bool = settings.training_pin_memory
    eta_min: float = settings.training_eta_min
    expected_num_classes: int = settings.training_expected_num_classes

    use_mlflow: bool = settings.mlflow_enabled
    mlflow_tracking_uri: str | None = settings.mlflow_tracking_uri
    mlflow_experiment_name: str = settings.mlflow_experiment_name
    mlflow_run_name: str | None = settings.mlflow_run_name
    mlflow_autolog: bool = settings.mlflow_autolog
    mlflow_log_artifacts: bool = settings.mlflow_log_artifacts
    model_name: str = settings.model_registry_name
    f1_threshold: float = settings.training_f1_threshold

    def __post_init__(self) -> None:
        """Run validation immediately after construction."""
        self.validate()

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingConfig":
        """Create a :class:`TrainingConfig` from a dictionary of values.

        Filters out unknown keys (logging a warning) and ``None`` values so
        that CLI and API callers can pass partial overrides without specifying
        every field.

        Args:
            values: Dictionary of field names to values. Keys not matching
                any dataclass field are ignored with a warning. ``None`` values
                are also ignored so the field default is used instead.

        Returns:
            A new :class:`TrainingConfig` instance with the provided
                values applied over the defaults.
        """
        valid_fields = set(cls.__dataclass_fields__.keys())
        unknown_fields = sorted(set(values) - valid_fields)
        if unknown_fields:
            logger.warning(
                "Ignoring unknown training config fields: %s",
                ", ".join(unknown_fields),
            )
        filtered = {
            key: value
            for key, value in values.items()
            if key in valid_fields and value is not None
        }
        return cls(**filtered)

    def validate(self) -> None:
        """Validate all hyperparameter constraints.

        Raises:
            ValueError: If any field violates its constraint, such as
                non-positive batch size, learning rate outside valid range,
                ``val_split`` outside (0, 1), ``n_splits`` not equal to 1, or
                missing MLflow experiment name when MLflow is enabled.
        """
        if self.img_size <= 0:
            raise ValueError("Image size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")
        if self.epochs <= 0:
            raise ValueError("Epoch count must be positive.")
        if self.patience <= 0:
            raise ValueError("Patience must be positive.")
        if not 0 < self.val_split < 1:
            raise ValueError("Validation split must be between 0 and 1.")
        if not 0 <= self.target_accuracy <= 1:
            raise ValueError("Target accuracy must be between 0 and 1.")
        if self.n_splits != 1:
            raise ValueError(
                "Number of splits must be exactly 1 for the current training workflow."
            )
        if self.num_workers < 0:
            raise ValueError("Number of workers cannot be negative.")
        if self.eta_min < 0:
            raise ValueError("eta_min cannot be negative.")
        if self.eta_min >= self.learning_rate:
            raise ValueError("eta_min must be smaller than learning_rate.")
        if self.expected_num_classes <= 0:
            raise ValueError("Expected number of NGT classes must be positive.")
        if self.use_mlflow and not self.mlflow_experiment_name.strip():
            raise ValueError(
                "MLflow experiment name must be set when MLflow tracking is enabled."
            )

    def to_mlflow_params(self) -> dict[str, int | float | str | bool | None]:
        """Return a flat dictionary of parameters suitable for MLflow logging.

        Returns:
            Dict of hyperparameter names to their values, covering all
                training and MLflow configuration fields that should be recorded
                as MLflow run parameters.
        """
        return {
            "img_size": self.img_size,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "patience": self.patience,
            "val_split": self.val_split,
            "target_accuracy": self.target_accuracy,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "eta_min": self.eta_min,
            "expected_num_classes": self.expected_num_classes,
            "use_mlflow": self.use_mlflow,
            "mlflow_experiment_name": self.mlflow_experiment_name,
            "mlflow_autolog": self.mlflow_autolog,
            "mlflow_log_artifacts": self.mlflow_log_artifacts,
        }
