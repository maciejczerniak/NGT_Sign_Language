"""Tests for sign_language_training.train.

Covers:
- log_device_info() CUDA and CPU branches
- _start_mlflow_run() Azure MLflow run handling
- run_training_workflow() MLflow logging/artifact/finally behavior
- Typer CLI entrypoint
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from sign_language_training import train as training_workflow
from sign_language_training.configuration import TrainingConfig, TrainingPaths
from sign_language_training.model_evaluation import EvaluationSummary, PredictionResult
from sign_language_training.model_registration import GateResult
from sign_language_training.model_training import TrainingResult


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Remove Rich/Typer ANSI styling from CLI output for stable assertions."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_training_paths(tmp_path: Path) -> TrainingPaths:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    ckpt = tmp_path / "pretrained.pth"
    ckpt.write_bytes(b"x")

    return TrainingPaths(
        data_dir=data_dir,
        pretrained_checkpoint=ckpt,
        checkpoint_dir=tmp_path / "checkpoints",
        results_dir=tmp_path / "results",
    )


def _make_base_cli_args(tmp_path: Path) -> list[str]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"x")

    return [
        "--data-dir",
        str(data_dir),
        "--pretrained-checkpoint",
        str(ckpt),
        "--checkpoint-dir",
        str(tmp_path / "checkpoints"),
        "--results-dir",
        str(tmp_path / "results"),
    ]


def _make_fake_results() -> tuple[TrainingResult, EvaluationSummary, GateResult]:
    training_result = TrainingResult(
        history={
            "train_loss": [0.4, 0.3],
            "val_loss": [0.5, 0.4],
            "train_acc": [0.75, 0.80],
            "val_acc": [0.70, 0.80],
            "lr": [0.001, 0.0009],
        },
        best_val_accuracy=0.80,
        epochs_trained=2,
    )

    evaluation_summary = EvaluationSummary(
        accuracy=0.92,
        f1_macro=0.91,
        f1_weighted=0.91,
        precision_macro=0.90,
        recall_macro=0.89,
        report="report",
    )

    gate_result = GateResult(
        passed=True,
        accuracy=0.92,
        f1_macro=0.91,
        accuracy_threshold=0.85,
        f1_threshold=0.80,
        registered_version="3",
    )

    return training_result, evaluation_summary, gate_result


def _patch_workflow_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch heavy dependencies inside run_training_workflow."""
    training_result, evaluation_summary, gate_result = _make_fake_results()

    monkeypatch.setattr(training_workflow, "set_seed", lambda seed: None)
    monkeypatch.setattr(training_workflow, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(training_workflow, "log_device_info", lambda: None)

    monkeypatch.setattr(
        training_workflow,
        "load_dataset",
        lambda data_dir: (list(range(4)), ["A", "B"], np.array([0, 0, 1, 1])),
    )
    monkeypatch.setattr(
        training_workflow,
        "create_stratified_split",
        lambda **kw: (np.array([0, 2]), np.array([1, 3])),
    )
    monkeypatch.setattr(
        training_workflow,
        "create_train_transform",
        lambda img_size: "train_tf",
    )
    monkeypatch.setattr(
        training_workflow,
        "create_val_transform",
        lambda img_size: "val_tf",
    )
    monkeypatch.setattr(
        training_workflow,
        "create_dataloaders",
        lambda **kw: ("train_loader", "val_loader"),
    )
    monkeypatch.setattr(
        training_workflow,
        "build_model_from_pretrained",
        lambda **kw: "model",
    )
    monkeypatch.setattr(training_workflow, "train_model", lambda **kw: training_result)
    monkeypatch.setattr(training_workflow, "load_best_model_state", lambda *a: None)
    monkeypatch.setattr(
        training_workflow,
        "collect_predictions",
        lambda *a: PredictionResult(
            predictions=np.array([0, 1]),
            labels=np.array([0, 1]),
            probabilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
        ),
    )
    monkeypatch.setattr(
        training_workflow,
        "summarize_predictions",
        lambda *a: evaluation_summary,
    )
    monkeypatch.setattr(
        training_workflow,
        "run_model_gate_and_register",
        lambda **kw: gate_result,
    )
    monkeypatch.setattr(training_workflow, "save_evaluation_summary", lambda **kw: None)
    monkeypatch.setattr(
        training_workflow, "count_parameters", lambda model: (200, 50, 150)
    )
    monkeypatch.setattr(training_workflow, "log_training_summary", lambda **kw: None)
    monkeypatch.setattr(training_workflow, "_save_class_names", lambda *a: None)


# ---------------------------------------------------------------------------
# log_device_info()
# ---------------------------------------------------------------------------


class TestLogDeviceInfo:
    def test_no_cuda_logs_cpu_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            patch("torch.cuda.is_available", return_value=False),
            caplog.at_level(logging.WARNING, logger="sign_language_training.train"),
        ):
            training_workflow.log_device_info()

        assert "CUDA NOT available" in caplog.text

    def test_cuda_available_logs_gpu_info(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_props = MagicMock()
        mock_props.name = "FakeGPU"
        mock_props.total_memory = 8 * 1024**3

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.current_device", return_value=0),
            patch("torch.cuda.get_device_name", return_value="FakeGPU"),
            caplog.at_level(logging.INFO, logger="sign_language_training.train"),
        ):
            training_workflow.log_device_info()

        assert "FakeGPU" in caplog.text
        assert "GPU count" in caplog.text


# ---------------------------------------------------------------------------
# _start_mlflow_run()
# ---------------------------------------------------------------------------


class TestStartMlflowRun:
    def test_resumes_azure_injected_run_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, str | None] = {}

        fake_mlflow = SimpleNamespace(
            set_tracking_uri=lambda uri: None,
            set_experiment=lambda name: None,
            start_run=lambda run_name=None, run_id=None: calls.setdefault(
                "run_id",
                run_id,
            ),
            active_run=lambda: None,
            end_run=lambda: None,
            log_params=lambda params: None,
            pytorch=SimpleNamespace(autolog=lambda log_models=False: None),
        )

        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.setenv("MLFLOW_RUN_ID", "azure-injected-run-abc123")

        config = TrainingConfig(
            use_mlflow=True,
            mlflow_experiment_name="test-exp",
            mlflow_autolog=False,
        )

        run = training_workflow._start_mlflow_run(config)

        assert run is not None
        assert calls["run_id"] == "azure-injected-run-abc123"

    def test_ends_active_run_before_starting_new_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: dict[str, bool] = {}
        fake_active_run = object()

        fake_mlflow = SimpleNamespace(
            set_tracking_uri=lambda uri: None,
            set_experiment=lambda name: None,
            start_run=lambda run_name=None, run_id=None: None,
            active_run=lambda: fake_active_run,
            end_run=lambda: calls.setdefault("end_run_called", True),
            log_params=lambda params: None,
            pytorch=SimpleNamespace(autolog=lambda log_models=False: None),
        )

        monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
        monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)

        config = TrainingConfig(
            use_mlflow=True,
            mlflow_experiment_name="test-exp",
            mlflow_autolog=False,
        )

        training_workflow._start_mlflow_run(config)

        assert calls.get("end_run_called") is True


# ---------------------------------------------------------------------------
# run_training_workflow() MLflow integration
# ---------------------------------------------------------------------------


class TestWorkflowMlflowIntegration:
    def _make_fake_mlflow_run(self) -> tuple[MagicMock, list]:
        logged: list = []

        mock_run = MagicMock()
        mock_run.log.side_effect = lambda data: logged.append(("log", data))
        mock_run.log_artifact.side_effect = lambda *a, **kw: logged.append(
            ("artifact", a, kw),
        )
        mock_run.finish.side_effect = lambda: logged.append(("finish",))

        return mock_run, logged

    def test_mlflow_run_logs_final_metrics_after_evaluation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _make_training_paths(tmp_path)
        config = TrainingConfig(
            expected_num_classes=2,
            num_workers=0,
            pin_memory=False,
            use_mlflow=True,
            mlflow_experiment_name="test",
            mlflow_log_artifacts=False,
        )

        mock_run, logged = self._make_fake_mlflow_run()
        _patch_workflow_internals(monkeypatch)
        monkeypatch.setattr(
            training_workflow, "_start_mlflow_run", lambda cfg: mock_run
        )

        training_workflow.run_training_workflow(paths, config)

        logged_keys = [list(entry[1].keys()) for entry in logged if entry[0] == "log"]
        assert any("final/accuracy" in keys for keys in logged_keys)
        assert any("best_val_accuracy" in keys for keys in logged_keys)

    def test_mlflow_run_logs_artifacts_when_enabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _make_training_paths(tmp_path)
        config = TrainingConfig(
            expected_num_classes=2,
            num_workers=0,
            pin_memory=False,
            use_mlflow=True,
            mlflow_experiment_name="test",
            mlflow_log_artifacts=True,
        )

        mock_run, logged = self._make_fake_mlflow_run()
        _patch_workflow_internals(monkeypatch)
        monkeypatch.setattr(
            training_workflow, "_start_mlflow_run", lambda cfg: mock_run
        )
        monkeypatch.setattr(
            training_workflow,
            "_save_training_accuracy_plot",
            lambda *a: None,
        )
        monkeypatch.setattr(
            training_workflow,
            "_save_training_loss_plot",
            lambda *a: None,
        )

        training_workflow.run_training_workflow(paths, config)

        artifact_calls = [entry for entry in logged if entry[0] == "artifact"]
        artifact_paths = [entry[1][0] for entry in artifact_calls]

        assert any("model.pth" in str(path) for path in artifact_paths)
        assert any("training_history.json" in str(path) for path in artifact_paths)

    def test_mlflow_run_logs_gate_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _make_training_paths(tmp_path)
        config = TrainingConfig(
            expected_num_classes=2,
            num_workers=0,
            pin_memory=False,
            use_mlflow=True,
            mlflow_experiment_name="test",
            mlflow_log_artifacts=False,
        )

        mock_run, logged = self._make_fake_mlflow_run()
        _patch_workflow_internals(monkeypatch)
        monkeypatch.setattr(
            training_workflow, "_start_mlflow_run", lambda cfg: mock_run
        )

        training_workflow.run_training_workflow(paths, config)

        gate_logs = [
            entry[1]
            for entry in logged
            if entry[0] == "log" and "gate/passed" in entry[1]
        ]

        assert len(gate_logs) == 1
        assert gate_logs[0]["gate/passed"] == 1.0

    def test_mlflow_run_finish_called_in_finally(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _make_training_paths(tmp_path)
        config = TrainingConfig(
            expected_num_classes=2,
            num_workers=0,
            pin_memory=False,
            use_mlflow=True,
            mlflow_experiment_name="test",
            mlflow_log_artifacts=False,
        )

        mock_run, logged = self._make_fake_mlflow_run()
        _patch_workflow_internals(monkeypatch)
        monkeypatch.setattr(
            training_workflow, "_start_mlflow_run", lambda cfg: mock_run
        )

        training_workflow.run_training_workflow(paths, config)

        assert ("finish",) in logged

    def test_mlflow_run_finish_called_even_when_workflow_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _make_training_paths(tmp_path)
        config = TrainingConfig(
            expected_num_classes=2,
            num_workers=0,
            pin_memory=False,
            use_mlflow=True,
            mlflow_experiment_name="test",
            mlflow_log_artifacts=False,
        )

        mock_run, logged = self._make_fake_mlflow_run()
        _patch_workflow_internals(monkeypatch)
        monkeypatch.setattr(
            training_workflow, "_start_mlflow_run", lambda cfg: mock_run
        )
        monkeypatch.setattr(
            training_workflow,
            "train_model",
            lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError, match="boom"):
            training_workflow.run_training_workflow(paths, config)

        assert ("finish",) in logged

    def test_expected_num_classes_mismatch_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        paths = _make_training_paths(tmp_path)
        config = TrainingConfig(
            expected_num_classes=5,
            num_workers=0,
            pin_memory=False,
            use_mlflow=False,
        )
        _patch_workflow_internals(monkeypatch)

        with pytest.raises(ValueError, match="Expected 5 NGT classes, found 2"):
            training_workflow.run_training_workflow(paths, config)


# ---------------------------------------------------------------------------
# _configure_logging()
# ---------------------------------------------------------------------------


def test_configure_logging_calls_basic_config_with_info_level() -> None:
    with patch("sign_language_training.train.logging.basicConfig") as mock_basic:
        training_workflow._configure_logging()

    mock_basic.assert_called_once()
    _, kwargs = mock_basic.call_args
    assert kwargs["level"] == logging.INFO


# ---------------------------------------------------------------------------
# Typer CLI
# ---------------------------------------------------------------------------


class TestTyperCli:
    def test_help_displays_training_options(self) -> None:
        result = runner.invoke(training_workflow.app, ["--help"])
        output = _strip_ansi(result.output)

        assert result.exit_code == 0
        assert (
            "Run the full NGT EfficientNet-B0 training workflow from the CLI" in output
        )
        assert "--data-dir" in output
        assert "--val-dir" in output
        assert "--help" in output

    def test_cli_accepts_required_long_options(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()
        captured: dict = {}

        def fake_workflow(paths, config, val_dir=None):
            captured["paths"] = paths
            return training_result, evaluation_summary, gate_result

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            _make_base_cli_args(tmp_path),
        )

        assert result.exit_code == 0
        assert captured["paths"].pretrained_checkpoint == tmp_path / "ckpt.pth"
        assert captured["paths"].checkpoint_dir.parent == tmp_path / "checkpoints"
        assert captured["paths"].results_dir.parent == tmp_path / "results"

    def test_cli_runs_workflow_successfully(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()
        captured: dict = {}

        def fake_workflow(paths, config, val_dir=None):
            captured["paths"] = paths
            captured["config"] = config
            captured["val_dir"] = val_dir
            return training_result, evaluation_summary, gate_result

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            _make_base_cli_args(tmp_path),
        )

        assert result.exit_code == 0
        assert captured["paths"].data_dir == tmp_path / "data"
        assert captured["paths"].pretrained_checkpoint == tmp_path / "ckpt.pth"
        assert captured["val_dir"] is None

    def test_cli_accepts_val_dir_for_presplit_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()
        captured: dict = {}

        val_dir = tmp_path / "val"
        val_dir.mkdir()

        def fake_workflow(paths, config, val_dir=None):
            captured["val_dir"] = val_dir
            return training_result, evaluation_summary, gate_result

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            _make_base_cli_args(tmp_path) + ["--val-dir", str(val_dir)],
        )

        assert result.exit_code == 0
        assert captured["val_dir"] == val_dir

    def test_cli_accepts_backward_compatible_output_aliases(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()
        captured: dict = {}

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        ckpt = tmp_path / "ckpt.pth"
        ckpt.write_bytes(b"x")

        def fake_workflow(paths, config, val_dir=None):
            captured["paths"] = paths
            return training_result, evaluation_summary, gate_result

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            [
                "--data-dir",
                str(data_dir),
                "--pretrained-checkpoint",
                str(ckpt),
                "--checkpoints-root",
                str(tmp_path / "ckpts"),
                "--results-root",
                str(tmp_path / "results"),
            ],
        )

        assert result.exit_code == 0
        assert captured["paths"].checkpoint_dir.parent == tmp_path / "ckpts"
        assert captured["paths"].results_dir.parent == tmp_path / "results"

    def test_cli_passes_hyperparameter_overrides(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()
        captured: dict = {}

        def fake_workflow(paths, config, val_dir=None):
            captured["config"] = config
            return training_result, evaluation_summary, gate_result

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            _make_base_cli_args(tmp_path)
            + [
                "--batch-size",
                "8",
                "--epochs",
                "5",
                "--learning-rate",
                "0.001",
                "--seed",
                "7",
                "--num-workers",
                "0",
                "--expected-num-classes",
                "2",
                "--f1-threshold",
                "0.75",
            ],
        )

        assert result.exit_code == 0
        assert captured["config"].batch_size == 8
        assert captured["config"].epochs == 5
        assert captured["config"].learning_rate == pytest.approx(0.001)
        assert captured["config"].seed == 7
        assert captured["config"].num_workers == 0
        assert captured["config"].expected_num_classes == 2
        assert captured["config"].f1_threshold == pytest.approx(0.75)

    def test_cli_exits_two_on_file_not_found_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_workflow(paths, config, val_dir=None):
            raise FileNotFoundError("missing")

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            _make_base_cli_args(tmp_path),
        )

        assert result.exit_code == 2
        assert (
            "Training failed: missing" in result.output or result.exception is not None
        )

    def test_cli_exits_two_on_value_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_workflow(paths, config, val_dir=None):
            raise ValueError("bad config")

        monkeypatch.setattr(training_workflow, "run_training_workflow", fake_workflow)

        result = runner.invoke(
            training_workflow.app,
            _make_base_cli_args(tmp_path),
        )

        assert result.exit_code == 2
        assert (
            "Training failed: bad config" in result.output
            or result.exception is not None
        )

    def test_cli_logs_registered_version_when_present(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()

        monkeypatch.setattr(
            training_workflow,
            "run_training_workflow",
            lambda paths, config, val_dir=None: (
                training_result,
                evaluation_summary,
                gate_result,
            ),
        )

        with caplog.at_level(logging.INFO, logger="sign_language_training.train"):
            result = runner.invoke(
                training_workflow.app,
                _make_base_cli_args(tmp_path),
            )

        assert result.exit_code == 0
        assert "registered_version" in caplog.text

    def test_cli_logs_mlflow_settings_when_enabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        training_result, evaluation_summary, gate_result = _make_fake_results()

        monkeypatch.setattr(
            training_workflow,
            "run_training_workflow",
            lambda paths, config, val_dir=None: (
                training_result,
                evaluation_summary,
                gate_result,
            ),
        )

        original_from_mapping = training_workflow.TrainingConfig.from_mapping

        def patched_from_mapping(overrides):
            cfg = original_from_mapping(overrides)
            from dataclasses import replace

            return replace(
                cfg,
                use_mlflow=True,
                mlflow_experiment_name="test-exp",
            )

        monkeypatch.setattr(
            training_workflow.TrainingConfig,
            "from_mapping",
            staticmethod(patched_from_mapping),
        )

        with caplog.at_level(logging.INFO, logger="sign_language_training.train"):
            result = runner.invoke(
                training_workflow.app,
                _make_base_cli_args(tmp_path),
            )

        assert result.exit_code == 0
        assert "mlflow_enabled" in caplog.text
        assert "mlflow_experiment" in caplog.text
