import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sign_language_training.model_evaluation import (
    EvaluationSummary,
    PredictionResult,
    collect_predictions,
    load_best_model_state,
    save_evaluation_summary,
    summarize_predictions,
)


class IdentityModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs


def test_load_best_model_state_restores_parameters_and_sets_eval_mode(
    tmp_path: Path,
) -> None:
    """Verify load best model state restores parameters and sets eval mode."""
    model = nn.Linear(2, 2)
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "best_model.pth"
    torch.save(expected_state, checkpoint_path)

    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    model.train()

    load_best_model_state(model, checkpoint_path, torch.device("cpu"))

    for key, value in expected_state.items():
        assert torch.equal(model.state_dict()[key], value)
    assert model.training is False


def test_load_best_model_state_reads_structured_checkpoint_format(
    tmp_path: Path,
) -> None:
    """Verify load best model state reads structured checkpoint format."""
    model = nn.Linear(2, 2)
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "best_model.pth"
    torch.save(
        {
            "model_state": expected_state,
            "class_names": ["A", "B"],
            "epoch": 3,
            "val_acc": 0.9,
        },
        checkpoint_path,
    )

    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    model.train()

    load_best_model_state(model, checkpoint_path, torch.device("cpu"))

    for key, value in expected_state.items():
        assert torch.equal(model.state_dict()[key], value)
    assert model.training is False


def test_collect_predictions_returns_predictions_labels_and_probabilities() -> None:
    """Verify collect predictions returns predictions labels and probabilities."""
    logits = torch.tensor(
        [
            [4.0, 1.0],
            [0.5, 2.5],
            [3.0, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2, shuffle=False)

    result = collect_predictions(IdentityModel(), loader, torch.device("cpu"))

    assert np.array_equal(result.predictions, np.array([0, 1, 0], dtype=np.int_))
    assert np.array_equal(result.labels, np.array([0, 1, 0], dtype=np.int_))
    assert result.probabilities.shape == (3, 2)
    assert np.allclose(result.probabilities.sum(axis=1), np.ones(3))


def test_summarize_predictions_returns_expected_metrics() -> None:
    """Verify summarize predictions returns expected metrics."""
    prediction_result = PredictionResult(
        predictions=np.array([0, 1, 0, 1], dtype=np.int_),
        labels=np.array([0, 1, 0, 1], dtype=np.int_),
        probabilities=np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.8, 0.2],
                [0.2, 0.8],
            ],
            dtype=np.float64,
        ),
    )

    summary = summarize_predictions(prediction_result, ["A", "B"])

    assert summary.accuracy == 1.0
    assert summary.f1_macro == 1.0
    assert summary.f1_weighted == 1.0
    assert summary.precision_macro == 1.0
    assert summary.recall_macro == 1.0
    assert "A" in summary.report
    assert "B" in summary.report


def test_save_evaluation_summary_writes_metrics_and_report_files(
    tmp_path: Path,
) -> None:
    """Verify save evaluation summary writes metrics and report files."""
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.txt"
    summary = EvaluationSummary(
        accuracy=0.8,
        f1_macro=0.75,
        f1_weighted=0.78,
        precision_macro=0.74,
        recall_macro=0.76,
        report="classification report",
    )

    save_evaluation_summary(summary, metrics_path, report_path)

    assert report_path.read_text(encoding="utf-8") == "classification report"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == {
        "accuracy": 0.8,
        "f1_macro": 0.75,
        "f1_weighted": 0.78,
        "precision_macro": 0.74,
        "recall_macro": 0.76,
    }
