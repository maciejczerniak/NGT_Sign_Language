"""
Tests for sign_language/cli.py.

Uses Typer's CliRunner to invoke commands without starting a real
server or loading real models.
"""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner
from PIL import Image

from sign_language.cli import (
    DEFAULT_TRAINING_CHECKPOINT_DIR,
    DEFAULT_TRAINING_RESULTS_DIR,
    app,
    _load_training_dependencies,
    _image_to_base64,
    _configure_training_console_logging,
)

runner = CliRunner()

_LOAD_ALL = "sign_language.cli.load_all"
_PREPROCESS = "sign_language.cli.preprocess_image"
_INFERENCE = "sign_language.cli.run_inference"


def test_training_outputs_default_to_logs_training() -> None:
    """Verify training checkpoints and result files share the logs/training root."""
    expected_root = Path("logs/training")

    assert DEFAULT_TRAINING_CHECKPOINT_DIR == expected_root
    assert DEFAULT_TRAINING_RESULTS_DIR == expected_root


def test_configure_training_console_logging_adds_handler_once() -> None:
    """Verify training CLI logging is configured without duplicate handlers."""
    import logging

    training_logger = logging.getLogger("sign_language_training")
    original_handlers = list(training_logger.handlers)
    original_propagate = training_logger.propagate
    training_logger.handlers.clear()

    try:
        _configure_training_console_logging()
        _configure_training_console_logging()

        stream_handlers = [
            handler
            for handler in training_logger.handlers
            if isinstance(handler, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1
        assert training_logger.level == logging.INFO
        assert training_logger.propagate is True
    finally:
        training_logger.handlers[:] = original_handlers
        training_logger.propagate = original_propagate


def test_load_training_dependencies_imports_and_caches() -> None:
    """Verify lazy training dependencies are imported once and cached."""
    with (
        patch("sign_language.cli.TrainingConfig", None),
        patch("sign_language.cli.TrainingPaths", None),
        patch("sign_language.cli.run_training_workflow", None),
    ):
        first = _load_training_dependencies()
        second = _load_training_dependencies()

    assert first == second
    assert all(item is not None for item in first)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(tmp_path: Path, name: str = "hand.png") -> Path:
    """Write a tiny PNG to tmp_path and return its path."""
    img = Image.new("RGB", (8, 8), color=(100, 150, 200))
    p = tmp_path / name
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# _image_to_base64
# ---------------------------------------------------------------------------


class TestImageToBase64:
    def test_valid_png(self, tmp_path: Path) -> None:
        """Verify PNG files are encoded as base64 strings."""
        p = _make_png(tmp_path)
        result = _image_to_base64(p)
        assert isinstance(result, str)
        # Must be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_valid_jpg(self, tmp_path: Path) -> None:
        """Verify JPG files are accepted and encoded as base64 strings."""
        img = Image.new("RGB", (8, 8))
        p = tmp_path / "hand.jpg"
        img.save(p)
        result = _image_to_base64(p)
        assert isinstance(result, str)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Verify missing image paths raise a Typer parameter error."""
        import typer

        with pytest.raises(typer.BadParameter, match="not found"):
            _image_to_base64(tmp_path / "missing.png")

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """Verify unsupported image extensions are rejected."""
        import typer

        p = tmp_path / "hand.bmp"
        p.write_bytes(b"fake")
        with pytest.raises(typer.BadParameter, match="Unsupported"):
            _image_to_base64(p)


# ---------------------------------------------------------------------------
# predict command
# ---------------------------------------------------------------------------


class TestPredictCommand:
    def test_missing_image_option(self) -> None:
        """Verify the predict command fails when no image is provided."""
        result = runner.invoke(app, ["predict"])
        assert result.exit_code != 0

    def test_image_not_found(self, tmp_path: Path) -> None:
        """Verify the predict command reports a missing image path."""
        result = runner.invoke(
            app, ["predict", "--image", str(tmp_path / "missing.png")]
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_unsupported_format(self, tmp_path: Path) -> None:
        """Verify the predict command rejects unsupported image formats."""
        p = tmp_path / "hand.bmp"
        p.write_bytes(b"fake")
        result = runner.invoke(app, ["predict", "--image", str(p)])
        assert result.exit_code == 1
        assert "unsupported" in result.output.lower()

    def test_successful_prediction(
        self, tmp_path: Path, mock_models: MagicMock
    ) -> None:
        """Verify the predict command returns JSON for a successful prediction."""
        p = _make_png(tmp_path)
        import torch

        tensor = torch.zeros(1, 3, 224, 224)

        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch(_PREPROCESS, return_value=(True, tensor, None)),
            patch(
                _INFERENCE,
                return_value=("A", 0.95, [{"letter": "A", "confidence": 0.95}]),
            ),
        ):
            result = runner.invoke(app, ["predict", "--image", str(p)])

        assert result.exit_code == 0
        # Extract JSON part from output (after the status messages)
        json_str = result.output[result.output.index("{") :]
        output = json.loads(json_str)
        assert output["prediction"] == "A"
        assert output["confidence"] == 0.95
        assert output["hand_detected"] is True

    def test_verbose_prediction_enables_debug_logging(
        self,
        tmp_path: Path,
        mock_models: MagicMock,
    ) -> None:
        """Verify --verbose raises the root logger level to DEBUG."""
        p = _make_png(tmp_path)
        import torch

        tensor = torch.zeros(1, 3, 224, 224)

        with (
            patch("sign_language.cli.logging.getLogger") as mock_get_logger,
            patch(_LOAD_ALL, return_value=mock_models),
            patch(_PREPROCESS, return_value=(True, tensor, None)),
            patch(_INFERENCE, return_value=("A", 0.95, [])),
        ):
            result = runner.invoke(app, ["predict", "--image", str(p), "--verbose"])

        assert result.exit_code == 0
        mock_get_logger.return_value.setLevel.assert_called_once()

    def test_top_k_limits_results(self, tmp_path: Path, mock_models: MagicMock) -> None:
        """Verify the --top-k option limits the returned predictions."""
        p = _make_png(tmp_path)
        import torch

        tensor = torch.zeros(1, 3, 224, 224)
        top_3 = [
            {"letter": "A", "confidence": 0.9},
            {"letter": "B", "confidence": 0.07},
            {"letter": "C", "confidence": 0.03},
        ]

        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch(_PREPROCESS, return_value=(True, tensor, None)),
            patch(_INFERENCE, return_value=("A", 0.9, top_3)),
        ):
            result = runner.invoke(app, ["predict", "--image", str(p), "--top-k", "1"])

        assert result.exit_code == 0
        json_str = result.output[result.output.index("{") :]
        output = json.loads(json_str)
        assert len(output["top_k"]) == 1

    def test_no_hand_detected_warning(
        self, tmp_path: Path, mock_models: MagicMock
    ) -> None:
        """Verify the predict command warns when preprocessing finds no hand."""
        p = _make_png(tmp_path)
        import torch

        tensor = torch.zeros(1, 3, 224, 224)

        with (
            patch(_LOAD_ALL, return_value=mock_models),
            patch(_PREPROCESS, return_value=(False, tensor, None)),
            patch(
                _INFERENCE,
                return_value=("A", 0.6, [{"letter": "A", "confidence": 0.6}]),
            ),
        ):
            result = runner.invoke(app, ["predict", "--image", str(p)])

        assert "no hand detected" in result.output.lower()

    def test_model_not_found_error(self, tmp_path: Path) -> None:
        """Verify model loading errors are reported by the predict command."""
        p = _make_png(tmp_path)

        with patch(_LOAD_ALL, side_effect=FileNotFoundError("model not found")):
            result = runner.invoke(app, ["predict", "--image", str(p)])

        assert result.exit_code == 1
        assert "error" in result.output.lower()


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_reload_with_multiple_workers_fails(self) -> None:
        """Verify reload mode rejects multiple worker processes."""
        result = runner.invoke(app, ["serve", "--reload", "--workers", "2"])
        assert result.exit_code == 1
        assert "--reload" in result.output

    def test_serve_calls_uvicorn(self) -> None:
        """Verify the serve command starts uvicorn."""
        with patch("sign_language.cli.uvicorn.run") as mock_run:
            result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_serve_default_host_and_port(self) -> None:
        """Verify the serve command uses the default host and port."""
        with patch("sign_language.cli.uvicorn.run") as mock_run:
            runner.invoke(app, ["serve"])
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8000

    def test_serve_custom_host_and_port(self) -> None:
        """Verify custom host and port options are passed to uvicorn."""
        with patch("sign_language.cli.uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--host", "0.0.0.0", "--port", "8080"])
        _, kwargs = mock_run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 8080

    def test_serve_reload_mode(self) -> None:
        """Verify reload mode is passed to uvicorn with one worker."""
        with patch("sign_language.cli.uvicorn.run") as mock_run:
            runner.invoke(app, ["serve", "--reload"])
        _, kwargs = mock_run.call_args
        assert kwargs["reload"] is True
        assert kwargs["workers"] == 1


class TestTrainCommand:
    def test_train_runs_workflow_and_prints_summary(self, tmp_path: Path) -> None:
        """Verify train builds paths/config and prints workflow results."""
        checkpoint = tmp_path / "pretrained.pth"
        checkpoint.write_bytes(b"checkpoint")
        paths = SimpleNamespace(
            data_dir=tmp_path / "data",
            pretrained_checkpoint=checkpoint,
            checkpoint_dir=tmp_path / "checkpoints" / "run",
            results_dir=tmp_path / "results" / "run",
            best_model_path=tmp_path / "checkpoints" / "run" / "model.pth",
            metrics_path=tmp_path / "results" / "run" / "metrics.json",
        )
        config = SimpleNamespace(
            batch_size=4,
            epochs=2,
            learning_rate=0.001,
            img_size=128,
            val_split=0.2,
            seed=123,
            use_mlflow=True,
            mlflow_tracking_uri=None,
            mlflow_experiment_name="exp",
        )
        training_result = SimpleNamespace(best_val_accuracy=0.91, epochs_trained=2)
        evaluation_summary = SimpleNamespace(accuracy=0.89)
        gate_result = SimpleNamespace(passed=True, registered_version="7")

        fake_paths_cls = SimpleNamespace(for_run=MagicMock(return_value=paths))
        fake_config_cls = SimpleNamespace(from_mapping=MagicMock(return_value=config))
        mock_run = MagicMock(
            return_value=(training_result, evaluation_summary, gate_result)
        )

        with patch(
            "sign_language.cli._load_training_dependencies",
            return_value=(fake_config_cls, fake_paths_cls, mock_run),
        ):
            result = runner.invoke(
                app,
                [
                    "train",
                    "--pretrained-checkpoint",
                    str(checkpoint),
                    "--data-dir",
                    str(tmp_path / "data"),
                    "--checkpoint-dir",
                    str(tmp_path / "checkpoints"),
                    "--results-dir",
                    str(tmp_path / "results"),
                    "--batch-size",
                    "4",
                    "--epochs",
                    "2",
                    "--learning-rate",
                    "0.001",
                    "--img-size",
                    "128",
                    "--seed",
                    "123",
                    "--mlflow-enabled",
                    "--mlflow-experiment-name",
                    "exp",
                ],
            )

        assert result.exit_code == 0
        assert "Starting NGT training:" in result.output
        assert "Training complete:" in result.output
        assert "registered_version: 7" in result.output
        mock_run.assert_called_once_with(paths, config)

    def test_train_reports_workflow_configuration_errors(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify training workflow FileNotFoundError/ValueError becomes exit 1."""
        checkpoint = tmp_path / "pretrained.pth"
        checkpoint.write_bytes(b"checkpoint")
        paths = SimpleNamespace(
            data_dir=tmp_path / "data",
            pretrained_checkpoint=checkpoint,
            checkpoint_dir=tmp_path / "checkpoints" / "run",
            results_dir=tmp_path / "results" / "run",
            best_model_path=tmp_path / "checkpoints" / "run" / "model.pth",
            metrics_path=tmp_path / "results" / "run" / "metrics.json",
        )
        config = SimpleNamespace(
            batch_size=4,
            epochs=2,
            learning_rate=0.001,
            img_size=128,
            val_split=0.2,
            seed=123,
            use_mlflow=False,
            mlflow_tracking_uri=None,
            mlflow_experiment_name="exp",
        )

        fake_paths_cls = SimpleNamespace(for_run=MagicMock(return_value=paths))
        fake_config_cls = SimpleNamespace(from_mapping=MagicMock(return_value=config))
        mock_run = MagicMock(side_effect=ValueError("bad config"))

        with patch(
            "sign_language.cli._load_training_dependencies",
            return_value=(fake_config_cls, fake_paths_cls, mock_run),
        ):
            result = runner.invoke(
                app,
                ["train", "--pretrained-checkpoint", str(checkpoint)],
            )

        assert result.exit_code == 1
        assert "Error: bad config" in result.output


class TestLocalPipelineCommand:
    """Tests for the local Azure-like preprocessing and training pipeline."""

    def test_rejects_missing_raw_data(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")

        result = runner.invoke(
            app,
            [
                "local-pipeline",
                "--raw-data-dir",
                str(tmp_path / "missing"),
                "--pretrained-checkpoint",
                str(checkpoint),
            ],
        )

        assert result.exit_code == 1
        assert "Raw data directory not found" in result.output

    def test_rejects_missing_checkpoint(self, tmp_path: Path) -> None:
        raw_data = tmp_path / "raw"
        raw_data.mkdir()

        result = runner.invoke(
            app,
            [
                "local-pipeline",
                "--raw-data-dir",
                str(raw_data),
                "--pretrained-checkpoint",
                str(tmp_path / "missing.pth"),
            ],
        )

        assert result.exit_code == 1
        assert "Pretrained checkpoint not found" in result.output

    def test_skip_preprocess_runs_training_and_prints_summary(
        self,
        tmp_path: Path,
    ) -> None:
        raw_data = tmp_path / "raw"
        raw_data.mkdir()
        checkpoint = tmp_path / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        output = tmp_path / "output"
        (output / "preprocessed" / "train").mkdir(parents=True)
        (output / "preprocessed" / "val").mkdir(parents=True)

        paths = SimpleNamespace(
            best_model_path=output / "checkpoints" / "model.pth",
        )
        training_result = SimpleNamespace(best_val_accuracy=0.91, epochs_trained=2)
        evaluation = SimpleNamespace(accuracy=0.89, f1_macro=0.88)
        gate = SimpleNamespace(passed=True, registered_version=None)
        config_class = SimpleNamespace(
            from_mapping=MagicMock(return_value=SimpleNamespace())
        )

        with (
            patch("sign_language.cli.TrainingPaths", return_value=paths),
            patch("sign_language.cli.TrainingConfig", config_class),
            patch(
                "sign_language.cli.run_training_workflow",
                return_value=(training_result, evaluation, gate),
            ) as run_workflow,
            patch("sign_language_training.augmentation.stratified_split") as split,
            patch("sign_language_training.augmentation.augment_dir") as augment,
        ):
            result = runner.invoke(
                app,
                [
                    "local-pipeline",
                    "--raw-data-dir",
                    str(raw_data),
                    "--output-dir",
                    str(output),
                    "--pretrained-checkpoint",
                    str(checkpoint),
                    "--skip-preprocess",
                ],
            )

        assert result.exit_code == 0
        assert "Skipping preprocessing" in result.output
        assert "LOCAL PIPELINE COMPLETE" in result.output
        split.assert_not_called()
        augment.assert_not_called()
        run_workflow.assert_called_once()

    def test_clean_preprocesses_before_training(self, tmp_path: Path) -> None:
        raw_data = tmp_path / "raw"
        raw_data.mkdir()
        checkpoint = tmp_path / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        output = tmp_path / "output"
        output.mkdir()
        stale = output / "stale.txt"
        stale.write_text("stale", encoding="utf-8")

        paths = SimpleNamespace(best_model_path=output / "checkpoints" / "model.pth")
        summary = (
            SimpleNamespace(best_val_accuracy=0.9, epochs_trained=1),
            SimpleNamespace(accuracy=0.88, f1_macro=0.87),
            SimpleNamespace(passed=True, registered_version="7"),
        )
        config_class = SimpleNamespace(
            from_mapping=MagicMock(return_value=SimpleNamespace())
        )

        with (
            patch("sign_language.cli.TrainingPaths", return_value=paths),
            patch("sign_language.cli.TrainingConfig", config_class),
            patch("sign_language.cli.run_training_workflow", return_value=summary),
            patch("sign_language_training.augmentation.stratified_split") as split,
            patch("sign_language_training.augmentation.augment_dir") as augment,
        ):
            result = runner.invoke(
                app,
                [
                    "local-pipeline",
                    "--raw-data-dir",
                    str(raw_data),
                    "--output-dir",
                    str(output),
                    "--pretrained-checkpoint",
                    str(checkpoint),
                    "--clean",
                ],
            )

        assert result.exit_code == 0
        assert "Cleaning output directory" in result.output
        assert "Preprocessing complete" in result.output
        assert not stale.exists()
        split.assert_called_once()
        augment.assert_called_once()

    def test_reports_training_error(self, tmp_path: Path) -> None:
        raw_data = tmp_path / "raw"
        raw_data.mkdir()
        checkpoint = tmp_path / "checkpoint.pth"
        checkpoint.write_bytes(b"checkpoint")
        output = tmp_path / "output"
        (output / "preprocessed" / "train").mkdir(parents=True)
        (output / "preprocessed" / "val").mkdir(parents=True)
        config_class = SimpleNamespace(
            from_mapping=MagicMock(return_value=SimpleNamespace())
        )

        with (
            patch(
                "sign_language.cli.TrainingPaths",
                return_value=SimpleNamespace(),
            ),
            patch("sign_language.cli.TrainingConfig", config_class),
            patch(
                "sign_language.cli.run_training_workflow",
                side_effect=ValueError("bad training"),
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "local-pipeline",
                    "--raw-data-dir",
                    str(raw_data),
                    "--output-dir",
                    str(output),
                    "--pretrained-checkpoint",
                    str(checkpoint),
                    "--skip-preprocess",
                ],
            )

        assert result.exit_code == 1
        assert "Error: bad training" in result.output
