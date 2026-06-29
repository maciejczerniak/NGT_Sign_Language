"""Model evaluation — predictions, summary metrics, persistence."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    """Predictions, ground-truth labels, and class probabilities from a dataset pass.

    Args:
        predictions: Integer class predictions for each sample, shape ``(N,)``.
        labels: Ground-truth integer class labels for each sample, shape ``(N,)``.
        probabilities: Softmax class probabilities for each sample,
            shape ``(N, num_classes)``.
    """

    predictions: NDArray[np.int_]
    labels: NDArray[np.int_]
    probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class EvaluationSummary:
    """Validation metrics and printable classification report for one evaluation run.

    Args:
        accuracy: Overall fraction of correctly classified samples.
        f1_macro: Macro-averaged F1 score across all classes.
        f1_weighted: Weighted-averaged F1 score across all classes.
        precision_macro: Macro-averaged precision across all classes.
        recall_macro: Macro-averaged recall across all classes.
        report: Full per-class classification report string from
            :func:`~sklearn.metrics.classification_report`.
    """

    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    report: str


def load_best_model_state(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load a saved checkpoint into the model and switch it to eval mode.

    Accepts both raw state dicts and checkpoint dicts containing a
    ``"model_state"`` key.

    Args:
        model: The :class:`~torch.nn.Module` to load weights into.
            Modified in place.
        checkpoint_path: Path to the ``.pth`` checkpoint file.
        device: Torch device to map the checkpoint weights onto.
    """
    logger.info("Loading best model checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    model.load_state_dict(state_dict)
    model.eval()


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> PredictionResult:
    """Collect model predictions, labels, and softmax probabilities from a DataLoader.

    Runs the model in eval mode with ``torch.no_grad()`` over all batches in
    ``data_loader``.

    Args:
        model: The :class:`~torch.nn.Module` to evaluate. Set to eval mode
            before inference.
        data_loader: :class:`~torch.utils.data.DataLoader` yielding
            ``(images, labels)`` batches.
        device: Torch device to move image tensors onto before inference.

    Returns:
        A :class:`PredictionResult` containing predictions, labels, and
            probabilities as NumPy arrays.
    """
    all_predictions: list[int] = []
    all_labels: list[int] = []
    all_probabilities: list[list[float]] = []

    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            all_predictions.extend(outputs.argmax(1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return PredictionResult(
        predictions=np.array(all_predictions, dtype=np.int_),
        labels=np.array(all_labels, dtype=np.int_),
        probabilities=np.array(all_probabilities, dtype=np.float64),
    )


def summarize_predictions(
    prediction_result: PredictionResult,
    class_names: Sequence[str],
) -> EvaluationSummary:
    """Compute validation metrics and a classification report from a prediction result.

    Args:
        prediction_result: A :class:`PredictionResult` containing predictions
            and ground-truth labels.
        class_names: Ordered sequence of class label strings matching the
            integer indices in ``prediction_result``.

    Returns:
        An :class:`EvaluationSummary` containing accuracy, macro and
            weighted F1, macro precision, macro recall, and the full classification
            report string.
    """
    logger.info("Computing validation summary metrics")

    report = classification_report(
        prediction_result.labels,
        prediction_result.predictions,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    accuracy = float((prediction_result.predictions == prediction_result.labels).mean())
    f1_macro = float(
        f1_score(
            prediction_result.labels,
            prediction_result.predictions,
            average="macro",
        )
    )
    f1_weighted = float(
        f1_score(
            prediction_result.labels,
            prediction_result.predictions,
            average="weighted",
        )
    )
    precision_macro = float(
        precision_score(
            prediction_result.labels,
            prediction_result.predictions,
            average="macro",
            zero_division=0,
        )
    )
    recall_macro = float(
        recall_score(
            prediction_result.labels,
            prediction_result.predictions,
            average="macro",
            zero_division=0,
        )
    )

    return EvaluationSummary(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        report=report,
    )


def save_evaluation_summary(
    evaluation_summary: EvaluationSummary,
    metrics_path: str | Path,
    report_path: str | Path,
) -> None:
    """Save the evaluation metrics as JSON and the classification report as text.

    Args:
        evaluation_summary: The :class:`EvaluationSummary` to persist.
        metrics_path: Destination path for the JSON metrics file containing
            accuracy, F1, precision, and recall values.
        report_path: Destination path for the plain-text classification
            report file.
    """
    logger.info("Saving evaluation report to %s", report_path)
    with Path(report_path).open("w", encoding="utf-8") as handle:
        handle.write(evaluation_summary.report)

    payload = {
        "accuracy": evaluation_summary.accuracy,
        "f1_macro": evaluation_summary.f1_macro,
        "f1_weighted": evaluation_summary.f1_weighted,
        "precision_macro": evaluation_summary.precision_macro,
        "recall_macro": evaluation_summary.recall_macro,
    }
    logger.info("Saving evaluation metrics to %s", metrics_path)
    with Path(metrics_path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
