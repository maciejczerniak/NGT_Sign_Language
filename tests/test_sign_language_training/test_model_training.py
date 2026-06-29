import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sign_language.models.architectures import build_efficientnet
from sign_language.models.loader import load_efficientnet
from sign_language_training.configuration import TrainingConfig
from sign_language_training import model_training
from sign_language_training.model_training import EpochMetrics


class DummyScheduler:
    def __init__(
        self,
        optimizer: object,
        T_max: int,
        eta_min: float,
        **_: object,
    ) -> None:
        """Store scheduler parameters for training-loop assertions."""
        self.optimizer = optimizer
        self.t_max = T_max
        self.eta_min = eta_min
        self.steps = 0

    def step(self) -> None:
        """Count scheduler steps without changing optimizer state."""
        self.steps += 1


def create_toy_loader() -> DataLoader:
    """Create a small deterministic dataloader for model-training tests."""
    features = torch.tensor(
        [
            [2.0, -1.0],
            [-1.0, 2.0],
            [2.5, -1.5],
            [-1.5, 2.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=2, shuffle=False)


class DummyExperimentRun:
    def __init__(self) -> None:
        """Create an experiment-run stub that records logged metrics."""
        self.logged: list[dict[str, int | float]] = []

    def log(self, data: dict[str, int | float]) -> None:
        """Record a metrics payload sent by the training loop."""
        self.logged.append(data)


def test_train_one_epoch_returns_epoch_metrics() -> None:
    """Verify train one epoch returns epoch metrics."""
    loader = create_toy_loader()
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = model_training.train_one_epoch(
        model=model,
        data_loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert metrics.loss > 0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_validate_one_epoch_returns_epoch_metrics() -> None:
    """Verify validate one epoch returns epoch metrics."""
    loader = create_toy_loader()
    model = nn.Linear(2, 2)
    criterion = nn.CrossEntropyLoss()

    metrics = model_training.validate_one_epoch(
        model=model,
        data_loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert metrics.loss > 0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_save_training_history_writes_json_file(tmp_path: Path) -> None:
    """Verify save training history writes json file."""
    history_path = tmp_path / "history.json"
    history = {
        "train_loss": [0.5],
        "val_loss": [0.4],
        "train_acc": [0.75],
        "val_acc": [0.8],
        "lr": [0.001],
    }

    model_training.save_training_history(history, history_path)

    assert json.loads(history_path.read_text(encoding="utf-8")) == history


def test_train_model_saves_best_checkpoint_and_stops_early(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train model saves best checkpoint and stops early."""
    train_loader = create_toy_loader()
    val_loader = create_toy_loader()
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "best_model.pth"
    history_path = tmp_path / "history.json"
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=5,
        patience=2,
        target_accuracy=0.9,
        eta_min=1e-6,
    )

    train_metrics_sequence = iter(
        [
            EpochMetrics(loss=0.8, accuracy=0.55),
            EpochMetrics(loss=0.7, accuracy=0.60),
            EpochMetrics(loss=0.65, accuracy=0.62),
            EpochMetrics(loss=0.64, accuracy=0.63),
        ]
    )
    val_metrics_sequence = iter(
        [
            EpochMetrics(loss=0.75, accuracy=0.50),
            EpochMetrics(loss=0.65, accuracy=0.60),
            EpochMetrics(loss=0.66, accuracy=0.55),
            EpochMetrics(loss=0.67, accuracy=0.55),
        ]
    )

    monkeypatch.setattr(
        model_training,
        "train_one_epoch",
        lambda *args, **kwargs: next(train_metrics_sequence),
    )
    monkeypatch.setattr(
        model_training,
        "validate_one_epoch",
        lambda *args, **kwargs: next(val_metrics_sequence),
    )
    monkeypatch.setattr(model_training, "CosineAnnealingLR", DummyScheduler)

    result = model_training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        class_names=["A", "B"],
    )

    saved_history = json.loads(history_path.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    assert result.best_val_accuracy == pytest.approx(0.60)
    assert result.epochs_trained == 4
    assert result.history["val_acc"] == [0.5, 0.6, 0.55, 0.55]
    assert len(result.history["lr"]) == 4
    assert checkpoint_path.is_file()
    assert checkpoint_payload["class_names"] == ["A", "B"]
    assert checkpoint_payload["epoch"] == 2
    assert checkpoint_payload["val_acc"] == pytest.approx(0.60)
    assert "model_state" in checkpoint_payload
    assert saved_history["val_acc"] == [0.5, 0.6, 0.55, 0.55]


def test_train_model_logs_epoch_metrics_to_experiment_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train model logs epoch metrics to experiment run."""
    train_loader = create_toy_loader()
    val_loader = create_toy_loader()
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "best_model.pth"
    history_path = tmp_path / "history.json"
    experiment_run = DummyExperimentRun()
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=1,
        patience=1,
        eta_min=1e-6,
    )

    monkeypatch.setattr(
        model_training,
        "train_one_epoch",
        lambda *args, **kwargs: EpochMetrics(loss=0.8, accuracy=0.55),
    )
    monkeypatch.setattr(
        model_training,
        "validate_one_epoch",
        lambda *args, **kwargs: EpochMetrics(loss=0.7, accuracy=0.60),
    )
    monkeypatch.setattr(model_training, "CosineAnnealingLR", DummyScheduler)

    model_training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        class_names=["A", "B"],
        experiment_run=experiment_run,
    )

    assert experiment_run.logged == [
        {
            "epoch": 1,
            "train_loss": 0.8,
            "train_acc": 0.55,
            "val_loss": 0.7,
            "val_acc": 0.60,
            "learning_rate": 0.01,
        }
    ]


def test_train_model_saves_checkpoint_compatible_with_runtime_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train model saves checkpoint compatible with runtime loader."""
    train_loader = create_toy_loader()
    val_loader = create_toy_loader()
    model = build_efficientnet(2)
    checkpoint_path = tmp_path / "best_model_v2.pth"
    history_path = tmp_path / "history.json"
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=2,
        patience=2,
        target_accuracy=0.9,
        eta_min=1e-6,
    )

    monkeypatch.setattr(
        model_training,
        "train_one_epoch",
        lambda *args, **kwargs: EpochMetrics(loss=0.8, accuracy=0.55),
    )
    monkeypatch.setattr(
        model_training,
        "validate_one_epoch",
        lambda *args, **kwargs: EpochMetrics(loss=0.7, accuracy=0.60),
    )
    monkeypatch.setattr(model_training, "CosineAnnealingLR", DummyScheduler)

    model_training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        class_names=["A", "B"],
    )

    loaded_model, loaded_names, raw_checkpoint = load_efficientnet(
        checkpoint_path,
        torch.device("cpu"),
    )

    assert loaded_names == ["A", "B"]
    assert raw_checkpoint["class_names"] == ["A", "B"]
    assert raw_checkpoint["epoch"] == 1
    assert raw_checkpoint["val_acc"] == pytest.approx(0.60)
    assert isinstance(loaded_model, nn.Module)


def test_train_model_saves_first_checkpoint_when_accuracy_stays_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train model saves first checkpoint when accuracy stays zero."""
    train_loader = create_toy_loader()
    val_loader = create_toy_loader()
    model = nn.Linear(2, 2)
    checkpoint_path = tmp_path / "best_model_zero.pth"
    history_path = tmp_path / "history.json"
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=2,
        patience=2,
        target_accuracy=0.9,
        eta_min=1e-6,
    )

    monkeypatch.setattr(
        model_training,
        "train_one_epoch",
        lambda *args, **kwargs: EpochMetrics(loss=1.0, accuracy=0.0),
    )
    monkeypatch.setattr(
        model_training,
        "validate_one_epoch",
        lambda *args, **kwargs: EpochMetrics(loss=1.0, accuracy=0.0),
    )
    monkeypatch.setattr(model_training, "CosineAnnealingLR", DummyScheduler)

    result = model_training.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        config=config,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        class_names=["A", "B"],
    )

    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert checkpoint_path.is_file()
    assert result.best_val_accuracy == pytest.approx(0.0)
    assert checkpoint_payload["epoch"] == 1
    assert checkpoint_payload["val_acc"] == pytest.approx(0.0)
