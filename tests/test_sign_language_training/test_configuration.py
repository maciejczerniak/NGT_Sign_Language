from dataclasses import replace
from pathlib import Path
from collections.abc import Callable

import pytest

from sign_language_training.settings import settings
from sign_language_training.configuration import (
    CLASS_NAMES_FILENAME,
    CLASSIFICATION_REPORT_FILENAME,
    METRICS_FILENAME,
    MODEL_CHECKPOINT_FILENAME,
    TRAINING_HISTORY_FILENAME,
    TrainingConfig,
    TrainingPaths,
)


def test_training_paths_build_output_paths_and_directories(tmp_path: Path) -> None:
    """Verify training paths build output paths and directories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pretrained_checkpoint = tmp_path / "pretrained_checkpoint.pth"
    pretrained_checkpoint.write_bytes(b"checkpoint")

    paths = TrainingPaths(
        data_dir=data_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
    )

    paths.ensure_directories()

    assert paths.best_model_path == paths.checkpoint_dir / MODEL_CHECKPOINT_FILENAME
    assert paths.best_checkpoint_name == MODEL_CHECKPOINT_FILENAME
    assert paths.history_path == paths.results_dir / TRAINING_HISTORY_FILENAME
    assert paths.metrics_path == paths.results_dir / METRICS_FILENAME
    assert paths.report_path == paths.results_dir / CLASSIFICATION_REPORT_FILENAME
    assert paths.class_names_path == paths.results_dir / CLASS_NAMES_FILENAME
    assert paths.checkpoint_dir.is_dir()
    assert paths.results_dir.is_dir()


def test_training_paths_validate_inputs_raises_for_missing_directory(
    tmp_path: Path,
) -> None:
    """Verify training paths validate inputs raises for missing directory."""
    paths = TrainingPaths(
        data_dir=tmp_path / "missing_data",
        pretrained_checkpoint=tmp_path / "pretrained_checkpoint.pth",
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
    )

    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        paths.validate_inputs()


def test_training_paths_validate_inputs_raises_for_missing_checkpoint(
    tmp_path: Path,
) -> None:
    """Verify training paths validate inputs raises for missing checkpoint."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paths = TrainingPaths(
        data_dir=data_dir,
        pretrained_checkpoint=tmp_path / "missing_checkpoint.pth",
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
    )

    with pytest.raises(FileNotFoundError, match="Pretrained checkpoint not found"):
        paths.validate_inputs()


def test_training_paths_for_run_builds_next_run_directories(tmp_path: Path) -> None:
    """Verify for_run builds matching checkpoint and result run directories."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pretrained_checkpoint = tmp_path / "pretrained_checkpoint.pth"
    pretrained_checkpoint.write_bytes(b"checkpoint")
    checkpoints_root = tmp_path / "checkpoints"
    results_root = tmp_path / "results"
    (checkpoints_root / "model_1").mkdir(parents=True)

    paths = TrainingPaths.for_run(
        data_dir=data_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoints_root=checkpoints_root,
        results_root=results_root,
    )

    assert paths.checkpoint_dir == checkpoints_root / "model_2"
    assert paths.results_dir == results_root / "model_2"
    assert (
        paths.best_model_path
        == checkpoints_root / "model_2" / MODEL_CHECKPOINT_FILENAME
    )
    assert paths.history_path == results_root / "model_2" / TRAINING_HISTORY_FILENAME


def test_training_paths_for_run_uses_explicit_run_name(tmp_path: Path) -> None:
    """Verify for_run uses explicit run names without auto-generating."""
    paths = TrainingPaths.for_run(
        data_dir=tmp_path / "data",
        pretrained_checkpoint=tmp_path / "pretrained_checkpoint.pth",
        checkpoints_root=tmp_path / "checkpoints",
        results_root=tmp_path / "results",
        run_name="manual_run",
    )

    assert paths.checkpoint_dir == tmp_path / "checkpoints" / "manual_run"
    assert paths.results_dir == tmp_path / "results" / "manual_run"


def test_training_config_validate_accepts_defaults() -> None:
    """Verify training config validate accepts defaults."""
    config = TrainingConfig()
    assert config.img_size == settings.training_img_size
    assert config.batch_size == settings.training_batch_size
    assert config.seed == settings.training_seed
    assert config.expected_num_classes == settings.training_expected_num_classes
    assert config.use_mlflow == settings.mlflow_enabled


def test_training_config_builds_mlflow_params() -> None:
    """Verify training config builds MLflow parameters."""
    config = TrainingConfig(
        batch_size=8,
        learning_rate=0.001,
        epochs=2,
        use_mlflow=True,
        mlflow_tracking_uri="file:./mlruns",
        mlflow_experiment_name="example-project",
        mlflow_run_name="example-run",
    )

    metadata = config.to_mlflow_params()

    assert metadata["batch_size"] == 8
    assert metadata["learning_rate"] == 0.001
    assert metadata["epochs"] == 2


@pytest.mark.parametrize(
    ("build_invalid_config", "message"),
    [
        (lambda: replace(TrainingConfig(), img_size=0), "Image size must be positive."),
        (
            lambda: replace(TrainingConfig(), batch_size=0),
            "Batch size must be positive.",
        ),
        (
            lambda: replace(TrainingConfig(), learning_rate=0.0),
            "Learning rate must be positive.",
        ),
        (
            lambda: replace(TrainingConfig(), val_split=1.0),
            "Validation split must be between 0 and 1.",
        ),
        (
            lambda: replace(TrainingConfig(), target_accuracy=1.1),
            "Target accuracy must be between 0 and 1.",
        ),
        (
            lambda: replace(TrainingConfig(), n_splits=2),
            "Number of splits must be exactly 1 for the current training workflow.",
        ),
        (
            lambda: replace(TrainingConfig(), num_workers=-1),
            "Number of workers cannot be negative.",
        ),
        (
            lambda: replace(
                TrainingConfig(), use_mlflow=True, mlflow_experiment_name=""
            ),
            "MLflow experiment name must be set when MLflow tracking is enabled.",
        ),
    ],
)
def test_training_config_validate_rejects_invalid_values(
    build_invalid_config: Callable[[], TrainingConfig],
    message: str,
) -> None:
    """Verify training config validate rejects invalid values."""
    with pytest.raises(ValueError, match=message):
        build_invalid_config()
