"""Tests for the local retraining pipeline and environment-aware registration."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sign_language_training.model_evaluation import EvaluationSummary
from sign_language_training.model_registration import (
    GateResult,
    _is_azure_ml_environment,
    evaluate_model_gate,
    run_model_gate_and_register,
)


# ---------------------------------------------------------------------------
# _is_azure_ml_environment
# ---------------------------------------------------------------------------


class TestIsAzureMLEnvironment:
    def test_returns_false_when_env_var_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
        assert _is_azure_ml_environment() is False

    def test_returns_false_when_env_var_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZUREML_RUN_ID", "")
        assert _is_azure_ml_environment() is False

    def test_returns_true_when_env_var_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZUREML_RUN_ID", "some_run_id_123")
        assert _is_azure_ml_environment() is True


# ---------------------------------------------------------------------------
# Gate + registration behaviour
# ---------------------------------------------------------------------------


def _make_summary(accuracy: float = 0.95, f1_macro: float = 0.90) -> EvaluationSummary:
    return EvaluationSummary(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=0.90,
        precision_macro=0.90,
        recall_macro=0.90,
        report="dummy report",
    )


class TestModelGateAndRegister:
    def test_gate_fails_returns_no_registration(self, tmp_path: Path) -> None:
        """When the gate fails, no registration should be attempted."""
        summary = _make_summary(accuracy=0.30, f1_macro=0.20)
        result = run_model_gate_and_register(
            evaluation_summary=summary,
            model_path=tmp_path / "model.pth",
            model_name="test-model",
            class_names=["A", "B", "C"],
            accuracy_threshold=0.85,
            f1_threshold=0.80,
        )
        assert result.passed is False
        assert result.registered_version is None

    def test_gate_passes_local_skips_registration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the gate passes locally, registration is skipped gracefully."""
        monkeypatch.delenv("AZUREML_RUN_ID", raising=False)

        summary = _make_summary(accuracy=0.95, f1_macro=0.90)
        result = run_model_gate_and_register(
            evaluation_summary=summary,
            model_path=tmp_path / "model.pth",
            model_name="test-model",
            class_names=["A", "B", "C"],
            accuracy_threshold=0.85,
            f1_threshold=0.80,
        )
        assert result.passed is True
        assert result.registered_version is None

    def test_gate_passes_azure_calls_registration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When running in Azure ML, registration should be called."""
        monkeypatch.setenv("AZUREML_RUN_ID", "test_run_123")
        monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "fake-sub")
        monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "fake-rg")
        monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "fake-ws")

        # Mock the Azure SDK registration call
        monkeypatch.setattr(
            "sign_language_training.model_registration.register_model_azure_sdk",
            lambda **kwargs: "42",
        )

        summary = _make_summary(accuracy=0.95, f1_macro=0.90)
        result = run_model_gate_and_register(
            evaluation_summary=summary,
            model_path=tmp_path / "model.pth",
            model_name="test-model",
            class_names=["A", "B", "C"],
            accuracy_threshold=0.85,
            f1_threshold=0.80,
        )
        assert result.passed is True
        assert result.registered_version == "42"


# ---------------------------------------------------------------------------
# Preprocessing functions (unit-level)
# ---------------------------------------------------------------------------


class TestStratifiedSplitLocal:
    """Verify the split + augment functions work with a tiny local dataset."""

    @pytest.fixture()
    def tiny_dataset(self, tmp_path: Path) -> Path:
        """Create a minimal ImageFolder with 3 classes, 30 images each."""
        from PIL import Image

        raw_dir = tmp_path / "raw"
        for class_name in ["A", "B", "C"]:
            class_dir = raw_dir / class_name
            class_dir.mkdir(parents=True)
            for i in range(30):
                img = Image.new("RGB", (32, 32), color=(i * 8, 0, 0))
                img.save(class_dir / f"img_{i:03d}.jpg")
        return raw_dir

    def test_split_creates_three_dirs(self, tiny_dataset: Path, tmp_path: Path) -> None:
        from sign_language_training.augmentation import stratified_split

        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        test_dir = tmp_path / "test"

        class_names = stratified_split(
            input_dir=tiny_dataset,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=42,
        )

        assert set(class_names) == {"A", "B", "C"}
        assert train_dir.exists()
        assert val_dir.exists()
        assert test_dir.exists()

        # Every class should have at least one image in train
        for cls in class_names:
            assert len(list((train_dir / cls).iterdir())) >= 1

    def test_augment_dir_creates_copies(
        self, tiny_dataset: Path, tmp_path: Path
    ) -> None:
        from sign_language_training.augmentation import augment_dir

        output = tmp_path / "augmented"
        augment_dir(
            source_dir=tiny_dataset,
            output_dir=output,
            copies=2,
            img_size=32,
            seed=42,
        )

        # Each class had 30 images → 30 originals + 60 augmented = 90
        for cls in ["A", "B", "C"]:
            files = list((output / cls).iterdir())
            assert len(files) == 90
