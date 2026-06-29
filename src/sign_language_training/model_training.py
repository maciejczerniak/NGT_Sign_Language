"""Training loop, checkpoint persistence, history JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sign_language_training.configuration import TrainingConfig

logger = logging.getLogger(__name__)

History = dict[str, list[float]]


class ExperimentRun(Protocol):
    """Minimal experiment tracker interface used by the training loop.

    Any object implementing this protocol can be passed to :func:`train_model`
    as the ``experiment_run`` argument. The training loop calls :meth:`log`
    once per epoch with a dict of scalar metrics.
    """

    def log(self, data: dict[str, int | float]) -> None:
        """Log a dictionary of scalar metrics for the current step.

        Args:
            data: Dict mapping metric names to scalar values.
        """
        ...


@dataclass(frozen=True)
class EpochMetrics:
    """Loss and accuracy measured at the end of a single training or validation epoch.

    Args:
        loss: Average cross-entropy loss over all samples in the epoch.
        accuracy: Fraction of correctly classified samples in the epoch.
    """

    loss: float
    accuracy: float


@dataclass(frozen=True)
class TrainingResult:
    """Summary values and per-epoch history returned by the training loop.

    Args:
        history: Dict with keys ``train_loss``, ``val_loss``,
            ``train_acc``, ``val_acc``, and ``lr``, each mapping to a list of
            float values — one entry per epoch trained.
        best_val_accuracy: Highest validation accuracy achieved across
            all epochs.
        epochs_trained: Number of epochs completed before early stopping
            or the configured maximum was reached.
    """

    history: History
    best_val_accuracy: float
    epochs_trained: int


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Train the model for one epoch over the entire training DataLoader.

    Sets the model to training mode, iterates over all batches, and
    accumulates loss and correct prediction counts.

    Args:
        model: The :class:`~torch.nn.Module` to train. Set to train mode.
        data_loader: :class:`~torch.utils.data.DataLoader` yielding
            ``(images, labels)`` batches.
        criterion: Loss function, typically
            :class:`~torch.nn.CrossEntropyLoss`.
        optimizer: Optimiser used to update model parameters.
        device: Torch device to move tensors onto before forward pass.

    Returns:
        An :class:`EpochMetrics` with the average loss and accuracy
            for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return EpochMetrics(loss=epoch_loss, accuracy=epoch_accuracy)


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate the model for one epoch over the entire validation DataLoader.

    Sets the model to eval mode and runs inference under ``torch.no_grad()``.

    Args:
        model: The :class:`~torch.nn.Module` to evaluate. Set to eval mode.
        data_loader: :class:`~torch.utils.data.DataLoader` yielding
            ``(images, labels)`` batches.
        criterion: Loss function used to compute validation loss.
        device: Torch device to move tensors onto before forward pass.

    Returns:
        An :class:`EpochMetrics` with the average loss and accuracy
            for this epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    return EpochMetrics(loss=epoch_loss, accuracy=epoch_accuracy)


def save_training_history(history: History, history_path: str | Path) -> None:
    """Save the training history dictionary to a JSON file.

    Args:
        history: Dict with per-epoch metric lists as produced by
            :func:`train_model`.
        history_path: Destination path for the JSON history file.
    """
    logger.info("Saving training history to %s", history_path)
    with Path(history_path).open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def save_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    class_names: Sequence[str],
    epoch: int,
    val_accuracy: float,
) -> None:
    """Save a model checkpoint in the runtime loader format.

    The checkpoint dict contains ``model_state``, ``class_names``,
    ``epoch``, and ``val_acc`` keys so it can be loaded by
    :func:`~sign_language_training.model_evaluation.load_best_model_state`
    and the inference loader.

    Args:
        model: The :class:`~torch.nn.Module` whose state dict is saved.
        checkpoint_path: Destination path for the ``.pth`` checkpoint file.
        class_names: Ordered list of class label strings saved alongside
            the model state.
        epoch: The epoch number at which this checkpoint was saved.
        val_accuracy: Validation accuracy at the time of saving.
    """
    payload = {
        "model_state": model.state_dict(),
        "class_names": list(class_names),
        "epoch": epoch,
        "val_acc": float(val_accuracy),
    }
    logger.info(
        "Saving runtime-compatible checkpoint to %s (epoch=%d, val_acc=%.4f)",
        checkpoint_path,
        epoch,
        val_accuracy,
    )
    torch.save(payload, str(checkpoint_path))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    checkpoint_path: str | Path,
    history_path: str | Path,
    class_names: Sequence[str],
    experiment_run: ExperimentRun | None = None,
) -> TrainingResult:
    """Train the model, save the best checkpoint, and persist training history.

    Uses :class:`~torch.optim.Adam` with
    :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` scheduling.
    Saves a checkpoint whenever validation accuracy improves. Applies early
    stopping when validation accuracy does not improve for ``config.patience``
    consecutive epochs.

    Args:
        model: The :class:`~torch.nn.Module` to train. Only parameters
            with ``requires_grad=True`` are passed to the optimiser.
        train_loader: Training :class:`~torch.utils.data.DataLoader`.
        val_loader: Validation :class:`~torch.utils.data.DataLoader`.
        device: Torch device to move tensors onto during training.
        config: :class:`~sign_language_training.configuration.TrainingConfig`
            providing all hyperparameters.
        checkpoint_path: Path where the best model checkpoint is saved.
        history_path: Path where the per-epoch training history JSON is
            saved after training completes.
        class_names: Ordered list of class label strings stored in the
            checkpoint.
        experiment_run: Optional :class:`ExperimentRun` instance receiving
            per-epoch metrics via :meth:`ExperimentRun.log`. Pass ``None`` to
            skip experiment tracking.

    Returns:
        A :class:`TrainingResult` containing the per-epoch history dict,
            best validation accuracy, and number of epochs trained.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.eta_min,
    )

    history: History = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_accuracy = -1.0
    patience_counter = 0
    epochs_trained = 0

    logger.info("Starting training for %d epochs", config.epochs)
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_metrics = validate_one_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(val_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_acc"].append(val_metrics.accuracy)
        history["lr"].append(current_lr)
        epochs_trained = epoch

        if experiment_run is not None:
            experiment_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics.loss,
                    "train_acc": train_metrics.accuracy,
                    "val_loss": val_metrics.loss,
                    "val_acc": val_metrics.accuracy,
                    "learning_rate": current_lr,
                }
            )

        if val_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = val_metrics.accuracy
            patience_counter = 0
            save_model_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                class_names=class_names,
                epoch=epoch,
                val_accuracy=val_metrics.accuracy,
            )
            logger.info(
                "Saved best model to %s with validation accuracy %.4f",
                checkpoint_path,
                val_metrics.accuracy,
            )
        else:
            patience_counter += 1

        logger.info(
            "Epoch %02d/%d | Train Loss: %.4f Acc: %.4f | "
            "Val Loss: %.4f Acc: %.4f | LR: %.6f",
            epoch,
            config.epochs,
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
            current_lr,
        )

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Best validation accuracy: %.4f", best_val_accuracy)
    if best_val_accuracy >= config.target_accuracy:
        logger.info("Target accuracy %.0f%% achieved", config.target_accuracy * 100)
    else:
        logger.warning(
            "Validation accuracy is below target %.0f%%",
            config.target_accuracy * 100,
        )

    save_training_history(history, history_path)
    return TrainingResult(
        history=history,
        best_val_accuracy=best_val_accuracy,
        epochs_trained=epochs_trained,
    )
