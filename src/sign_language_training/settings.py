"""Standalone training settings.

This module intentionally avoids importing from ``sign_language`` so the
training job environment can stay minimal. It defines only the fields
needed by the training workflow itself; Azure ML submission concerns
(workspace, compute, instance type, data assets) live in the submission
script, not in the training job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent
DOTENV: Path = BASE_DIR.parent.parent / ".env"


class TrainingSettings(BaseSettings):
    """Training-job settings loaded from environment variables and ``.env``.

    All fields map directly to environment variables (case-insensitive).
    Unknown variables are silently ignored. Defaults are suitable for local
    development; production deployments should override values via ``.env``
    or job environment variables.

    **Output filenames**

    Args:
        training_best_checkpoint_name: Filename for the best model
            checkpoint saved during training.
        training_history_filename: Filename for the per-epoch training
            history JSON.
        training_metrics_filename: Filename for the evaluation metrics
            JSON.
        training_report_filename: Filename for the classification report
            text file.
        training_class_names_filename: Filename for the class names JSON.

    **Hyperparameters**

    Args:
        training_img_size: Input image size in pixels.
        training_batch_size: Number of samples per training batch.
        training_learning_rate: Initial optimiser learning rate.
        training_epochs: Maximum number of training epochs.
        training_patience: Early stopping patience in epochs.
        training_val_split: Fraction of the dataset to use for validation.
        training_target_accuracy: Minimum validation accuracy required to
            pass the model gate check.
        training_seed: Random seed for reproducible training.
        training_n_splits: Number of stratified splits (must be 1).
        training_num_workers: Number of DataLoader worker processes.
        training_pin_memory: Whether to use pinned memory in the
            DataLoader.
        training_eta_min: Minimum learning rate for the cosine annealing
            scheduler.
        training_expected_num_classes: Expected number of NGT output
            classes.

    **MLflow**

    Args:
        mlflow_enabled: Whether to enable MLflow experiment tracking.
        mlflow_tracking_uri: MLflow tracking URI override, or ``None``
            to use the default.
        mlflow_experiment_name: MLflow experiment name.
        mlflow_run_name: Optional MLflow run name.
        mlflow_autolog: Whether to enable MLflow PyTorch autologging.
        mlflow_log_artifacts: Whether to log training artifacts to
            MLflow.

    **Model registry**

    Args:
        model_registry_name: Azure ML model registry name used by the
            gate check and registration step.
        training_f1_threshold: Minimum macro F1 score required to pass
            the model gate check and trigger registration.
    """

    model_config = SettingsConfigDict(
        env_file=DOTENV,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    training_best_checkpoint_name: str = Field(default="model.pth")
    training_history_filename: str = Field(default="training_history.json")
    training_metrics_filename: str = Field(default="metrics.json")
    training_report_filename: str = Field(default="classification_report.txt")
    training_class_names_filename: str = Field(default="class_names.json")

    training_img_size: int = Field(default=224)
    training_batch_size: int = Field(default=16)
    training_learning_rate: float = Field(default=1e-4)
    training_epochs: int = Field(default=30)
    training_patience: int = Field(default=7)
    training_val_split: float = Field(default=0.2)
    training_target_accuracy: float = Field(default=0.85)
    training_seed: int = Field(default=42)
    training_n_splits: int = Field(default=1)
    training_num_workers: int = Field(default=4)
    training_pin_memory: bool = Field(default=True)
    training_eta_min: float = Field(default=1e-6)
    training_expected_num_classes: int = Field(default=22)

    mlflow_enabled: bool = Field(default=False)
    mlflow_tracking_uri: Optional[str] = Field(default=None)
    mlflow_experiment_name: str = Field(default="sign-language")
    mlflow_run_name: Optional[str] = Field(default=None)
    mlflow_autolog: bool = Field(default=True)
    mlflow_log_artifacts: bool = Field(default=True)

    model_registry_name: str = Field(default="ngt-sign-language")
    training_f1_threshold: float = Field(default=0.80)


settings = TrainingSettings()


if __name__ == "__main__":
    print(settings.model_dump())
