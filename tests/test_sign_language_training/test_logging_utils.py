import logging
from pathlib import Path

import pytest

from sign_language_training.configuration import TrainingConfig, TrainingPaths
from sign_language_training.logging_utils import log_training_summary
from sign_language_training.model_evaluation import EvaluationSummary
from sign_language_training.model_training import TrainingResult


@pytest.mark.parametrize(
    ("best_accuracy", "expected_status"),
    [
        (0.90, "TARGET MET"),
        (0.70, "BELOW TARGET"),
    ],
)
def test_log_training_summary_reports_training_outcome(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    best_accuracy: float,
    expected_status: str,
) -> None:
    """Verify log training summary reports training outcome."""
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
    config = TrainingConfig(target_accuracy=0.80, val_split=0.25, seed=123)
    training_result = TrainingResult(
        history={
            "train_loss": [0.4],
            "val_loss": [0.3],
            "train_acc": [0.8],
            "val_acc": [best_accuracy],
            "lr": [0.001],
        },
        best_val_accuracy=best_accuracy,
        epochs_trained=3,
    )
    evaluation_summary = EvaluationSummary(
        accuracy=0.88,
        f1_macro=0.87,
        f1_weighted=0.89,
        precision_macro=0.86,
        recall_macro=0.85,
        report="classification report",
    )

    with caplog.at_level(logging.INFO, logger="sign_language_training.logging_utils"):
        log_training_summary(
            paths=paths,
            config=config,
            dataset_size=40,
            num_classes=4,
            trainable_parameters=100,
            total_parameters=200,
            training_result=training_result,
            evaluation_summary=evaluation_summary,
        )

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "NGT fine-tuning summary" in messages
    assert "Pretrained checkpoint: pretrained_checkpoint.pth" in messages
    assert "Dataset size: 40 images across 4 classes" in messages
    assert "Validation split: 25% (seed=123)" in messages
    assert "Trainable parameters: 100 / 200" in messages
    assert (
        f"Best validation accuracy: {best_accuracy:.4f} ({expected_status})" in messages
    )
    assert "Accuracy: 0.8800" in messages
    assert f"Best checkpoint: {paths.best_model_path}" in messages
