from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sign_language_training import train as training_workflow
from sign_language_training.configuration import TrainingConfig, TrainingPaths
from sign_language_training.model_evaluation import (
    EvaluationSummary,
    PredictionResult,
)
from sign_language_training.model_training import TrainingResult
from sign_language_training.model_registration import GateResult


def create_training_paths(tmp_path: Path) -> TrainingPaths:
    """Create valid temporary paths for the training workflow tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pretrained_checkpoint = tmp_path / "pretrained_checkpoint.pth"
    pretrained_checkpoint.write_bytes(b"checkpoint")
    return TrainingPaths(
        data_dir=data_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
    )


def test_run_training_workflow_coordinates_training_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run training workflow coordinates training dependencies."""
    paths = create_training_paths(tmp_path)
    config = TrainingConfig(
        expected_num_classes=2,
        num_workers=0,
        pin_memory=False,
        use_mlflow=False,
    )
    training_result = TrainingResult(
        history={
            "train_loss": [0.4],
            "val_loss": [0.3],
            "train_acc": [0.8],
            "val_acc": [0.85],
            "lr": [0.001],
        },
        best_val_accuracy=0.85,
        epochs_trained=1,
    )
    prediction_result = PredictionResult(
        predictions=np.array([0, 1], dtype=np.int_),
        labels=np.array([0, 1], dtype=np.int_),
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64),
    )
    evaluation_summary = EvaluationSummary(
        accuracy=1.0,
        f1_macro=1.0,
        f1_weighted=1.0,
        precision_macro=1.0,
        recall_macro=1.0,
        report="classification report",
    )
    gate_result = GateResult(
        passed=True,
        accuracy=1.0,
        f1_macro=1.0,
        accuracy_threshold=0.85,
        f1_threshold=0.80,
        registered_version="3",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        training_workflow,
        "set_seed",
        lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        training_workflow,
        "get_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        training_workflow,
        "log_device_info",
        lambda: None,
    )
    monkeypatch.setattr(
        training_workflow,
        "load_dataset",
        lambda data_dir: (
            [object(), object(), object(), object()],
            ["A", "B"],
            np.array([0, 0, 1, 1]),
        ),
    )

    def fake_create_stratified_split(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        calls["split"] = kwargs
        return np.array([0, 2]), np.array([1, 3])

    def fake_create_train_transform(img_size: int) -> str:
        calls["train_transform_size"] = img_size
        return "train_transform"

    def fake_create_val_transform(img_size: int) -> str:
        calls["val_transform_size"] = img_size
        return "val_transform"

    def fake_create_dataloaders(**kwargs: object) -> tuple[str, str]:
        calls["dataloaders"] = kwargs
        return "train_loader", "val_loader"

    def fake_build_model_from_pretrained(**kwargs: object) -> str:
        calls["build_model"] = kwargs
        return "model"

    def fake_train_model(**kwargs: object) -> TrainingResult:
        calls["train_model"] = kwargs
        return training_result

    def fake_load_best_model_state(
        model: object, checkpoint_path: Path, device: torch.device
    ) -> None:
        calls["load_best_model_state"] = (model, checkpoint_path, device)

    def fake_collect_predictions(
        model: object, data_loader: object, device: torch.device
    ) -> PredictionResult:
        calls["collect_predictions"] = (model, data_loader, device)
        return prediction_result

    def fake_summarize_predictions(
        prediction_result_value: PredictionResult, class_names: list[str]
    ) -> EvaluationSummary:
        calls["summarize_predictions"] = (prediction_result_value, tuple(class_names))
        return evaluation_summary

    def fake_run_model_gate_and_register(**kwargs: object) -> GateResult:
        calls["gate_and_register"] = kwargs
        return gate_result

    monkeypatch.setattr(
        training_workflow, "create_stratified_split", fake_create_stratified_split
    )
    monkeypatch.setattr(
        training_workflow, "create_train_transform", fake_create_train_transform
    )
    monkeypatch.setattr(
        training_workflow, "create_val_transform", fake_create_val_transform
    )
    monkeypatch.setattr(
        training_workflow, "create_dataloaders", fake_create_dataloaders
    )
    monkeypatch.setattr(
        training_workflow,
        "build_model_from_pretrained",
        fake_build_model_from_pretrained,
    )
    monkeypatch.setattr(training_workflow, "train_model", fake_train_model)
    monkeypatch.setattr(
        training_workflow, "load_best_model_state", fake_load_best_model_state
    )
    monkeypatch.setattr(
        training_workflow, "collect_predictions", fake_collect_predictions
    )
    monkeypatch.setattr(
        training_workflow, "summarize_predictions", fake_summarize_predictions
    )
    monkeypatch.setattr(
        training_workflow,
        "run_model_gate_and_register",
        fake_run_model_gate_and_register,
    )
    monkeypatch.setattr(
        training_workflow,
        "save_evaluation_summary",
        lambda **kwargs: calls.setdefault("save_evaluation_summary", kwargs),
    )
    monkeypatch.setattr(
        training_workflow, "count_parameters", lambda model: (100, 20, 80)
    )
    monkeypatch.setattr(
        training_workflow,
        "log_training_summary",
        lambda **kwargs: calls.setdefault("log_training_summary", kwargs),
    )

    result_tr, result_es, result_gr = training_workflow.run_training_workflow(
        paths, config
    )

    assert result_tr == training_result
    assert result_es == evaluation_summary
    assert result_gr == gate_result
    assert calls["seed"] == config.seed
    assert calls["train_transform_size"] == config.img_size
    assert calls["val_transform_size"] == config.img_size
    split_kwargs = calls["split"]
    assert isinstance(split_kwargs, dict)
    assert np.array_equal(split_kwargs["targets"], np.array([0, 0, 1, 1]))
    assert split_kwargs["n_splits"] == config.n_splits
    assert split_kwargs["split_ratio"] == config.val_split
    assert split_kwargs["seed"] == config.seed
    dataloader_kwargs = calls["dataloaders"]
    assert isinstance(dataloader_kwargs, dict)
    assert dataloader_kwargs["data_dir"] == paths.data_dir
    assert np.array_equal(dataloader_kwargs["train_idx"], np.array([0, 2]))
    assert np.array_equal(dataloader_kwargs["val_idx"], np.array([1, 3]))
    assert dataloader_kwargs["train_transform"] == "train_transform"
    assert dataloader_kwargs["val_transform"] == "val_transform"
    assert dataloader_kwargs["config"] == config
    assert calls["build_model"] == {
        "pretrained_checkpoint_path": paths.pretrained_checkpoint,
        "device": torch.device("cpu"),
        "num_ngt_classes": 2,
    }
    train_model_kwargs = calls["train_model"]
    assert isinstance(train_model_kwargs, dict)
    assert train_model_kwargs["checkpoint_path"] == paths.best_model_path
    assert train_model_kwargs["history_path"] == paths.history_path
    assert train_model_kwargs["class_names"] == ["A", "B"]
    assert train_model_kwargs["experiment_run"] is None
    assert calls["save_evaluation_summary"] == {
        "evaluation_summary": evaluation_summary,
        "metrics_path": paths.metrics_path,
        "report_path": paths.report_path,
    }
    assert paths.checkpoint_dir.is_dir()
    assert paths.results_dir.is_dir()


def test_start_mlflow_run_configures_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify MLflow run setup follows the configured tracking settings."""
    calls: dict[str, object] = {}

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: calls.setdefault("tracking_uri", uri),
        set_experiment=lambda name: calls.setdefault("experiment_name", name),
        start_run=lambda run_name=None, run_id=None: calls.setdefault(
            "run_name", run_name
        ),
        active_run=lambda: None,
        end_run=lambda: calls.setdefault("ended", True),
        log_params=lambda params: calls.setdefault("params", params),
        log_metrics=lambda metrics, step=None: calls.setdefault(
            "metrics", (metrics, step)
        ),
        log_artifact=lambda local_path, artifact_path=None: calls.setdefault(
            "artifact", (local_path, artifact_path)
        ),
        pytorch=SimpleNamespace(
            autolog=lambda log_models=False: calls.setdefault(
                "autolog_log_models", log_models
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    config = TrainingConfig(
        use_mlflow=True,
        mlflow_tracking_uri="file:./mlruns",
        mlflow_experiment_name="sign-language-tests",
        mlflow_run_name="unit-run",
    )

    run = training_workflow._start_mlflow_run(config)
    assert run is not None
    run.log({"epoch": 3, "train_loss": 0.4, "val_acc": 0.8})
    run.log_artifact("metrics.json", "evaluation")
    run.finish()

    assert calls["tracking_uri"] == "file:./mlruns"
    assert calls["experiment_name"] == "sign-language-tests"
    assert calls["run_name"] == "unit-run"
    assert calls["autolog_log_models"] is False
    assert calls["params"]["batch_size"] == config.batch_size
    assert calls["metrics"] == ({"train_loss": 0.4, "val_acc": 0.8}, 3)
    assert calls["artifact"] == ("metrics.json", "evaluation")
    assert calls["ended"] is True


def test_save_training_plots_create_png_files(tmp_path: Path) -> None:
    """Verify training plot artifacts are saved as PNG images."""
    accuracy_plot_path = tmp_path / "training_accuracy.png"
    loss_plot_path = tmp_path / "training_loss.png"
    history = {
        "train_loss": [0.6, 0.4],
        "val_loss": [0.7, 0.5],
        "train_acc": [0.7, 0.8],
        "val_acc": [0.65, 0.75],
    }

    training_workflow._save_training_accuracy_plot(history, accuracy_plot_path)
    training_workflow._save_training_loss_plot(history, loss_plot_path)

    assert accuracy_plot_path.is_file()
    assert accuracy_plot_path.read_bytes().startswith(b"\x89PNG")
    assert loss_plot_path.is_file()
    assert loss_plot_path.read_bytes().startswith(b"\x89PNG")


def test_run_training_workflow_raises_for_unexpected_number_of_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run training workflow raises for unexpected number of classes."""
    paths = create_training_paths(tmp_path)
    config = TrainingConfig(expected_num_classes=3, use_mlflow=False)

    monkeypatch.setattr(training_workflow, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_workflow, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training_workflow, "log_device_info", lambda: None)
    monkeypatch.setattr(
        training_workflow,
        "load_dataset",
        lambda data_dir: ([object(), object()], ["A", "B"], np.array([0, 1])),
    )

    with pytest.raises(ValueError, match="Expected 3 NGT classes, found 2."):
        training_workflow.run_training_workflow(paths, config)
