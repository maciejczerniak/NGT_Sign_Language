"""Tests for sign_language_training/model_registration.py.

Covers:
- GateResult.__str__          (lines 35-43)
- evaluate_model_gate          (lines 52-68)
- register_model_azure_sdk     (lines 78-115)
- run_model_gate_and_register  (lines 127-159)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sign_language_training.model_evaluation import EvaluationSummary
from sign_language_training.model_registration import (
    GateResult,
    evaluate_model_gate,
    register_model_azure_sdk,
    run_model_gate_and_register,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_summary(accuracy: float = 0.90, f1_macro: float = 0.88) -> EvaluationSummary:
    return EvaluationSummary(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=0.89,
        precision_macro=0.87,
        recall_macro=0.86,
        report="report",
    )


# ---------------------------------------------------------------------------
# GateResult.__str__
# ---------------------------------------------------------------------------


class TestGateResultStr:
    def test_passed_without_registered_version(self) -> None:
        result = GateResult(
            passed=True,
            accuracy=0.9123,
            f1_macro=0.8876,
            accuracy_threshold=0.85,
            f1_threshold=0.80,
            registered_version=None,
        )
        text = str(result)
        assert "PASSED" in text
        assert "0.9123" in text
        assert "0.8876" in text
        assert "registered version" not in text

    def test_passed_with_registered_version(self) -> None:
        result = GateResult(
            passed=True,
            accuracy=0.92,
            f1_macro=0.91,
            accuracy_threshold=0.85,
            f1_threshold=0.80,
            registered_version="7",
        )
        text = str(result)
        assert "PASSED" in text
        assert "registered version: 7" in text

    def test_failed_without_registered_version(self) -> None:
        result = GateResult(
            passed=False,
            accuracy=0.70,
            f1_macro=0.68,
            accuracy_threshold=0.85,
            f1_threshold=0.80,
            registered_version=None,
        )
        text = str(result)
        assert "FAILED" in text
        assert "registered version" not in text

    def test_contains_threshold_values(self) -> None:
        result = GateResult(
            passed=True,
            accuracy=0.91,
            f1_macro=0.89,
            accuracy_threshold=0.85,
            f1_threshold=0.80,
        )
        text = str(result)
        assert "threshold=0.8500" in text
        assert "threshold=0.8000" in text


# ---------------------------------------------------------------------------
# evaluate_model_gate
# ---------------------------------------------------------------------------


class TestEvaluateModelGate:
    def test_passes_when_both_thresholds_met(self) -> None:
        assert evaluate_model_gate(make_summary(0.90, 0.88), 0.85, 0.80) is True

    def test_fails_when_accuracy_below_threshold(self) -> None:
        assert evaluate_model_gate(make_summary(0.80, 0.88), 0.85, 0.80) is False

    def test_fails_when_f1_below_threshold(self) -> None:
        assert evaluate_model_gate(make_summary(0.90, 0.75), 0.85, 0.80) is False

    def test_fails_when_both_below_threshold(self) -> None:
        assert evaluate_model_gate(make_summary(0.70, 0.65), 0.85, 0.80) is False

    def test_passes_exactly_at_threshold(self) -> None:
        # boundary: >= not >
        assert evaluate_model_gate(make_summary(0.85, 0.80), 0.85, 0.80) is True


# ---------------------------------------------------------------------------
# register_model_azure_sdk
# ---------------------------------------------------------------------------


def _make_fake_azure(
    registered_version: str = "5",
) -> tuple[dict, MagicMock, SimpleNamespace, SimpleNamespace]:
    """Return fake Azure modules and clients for patching sys.modules."""
    mock_registered = MagicMock()
    mock_registered.version = registered_version

    mock_models_client = MagicMock()
    mock_models_client.create_or_update.return_value = mock_registered

    mock_ml_client = MagicMock()
    mock_ml_client.models = mock_models_client

    fake_azure_ml = SimpleNamespace(
        MLClient=MagicMock(return_value=mock_ml_client),
        constants=SimpleNamespace(
            AssetTypes=SimpleNamespace(CUSTOM_MODEL="custom_model")
        ),
        entities=SimpleNamespace(Model=MagicMock(return_value=MagicMock())),
    )
    fake_azure_identity = SimpleNamespace(
        ManagedIdentityCredential=MagicMock(return_value=MagicMock()),
        AzureCliCredential=MagicMock(return_value=MagicMock()),
    )
    return {}, mock_ml_client, fake_azure_ml, fake_azure_identity


class TestRegisterModelAzureSdk:
    def _patch_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AZUREML_ARM_SUBSCRIPTION", "sub-123")
        monkeypatch.setenv("AZUREML_ARM_RESOURCEGROUP", "rg-test")
        monkeypatch.setenv("AZUREML_ARM_WORKSPACE_NAME", "ws-test")

    def test_returns_registered_version_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_env(monkeypatch)
        _, mock_ml_client, fake_azure_ml, fake_azure_identity = _make_fake_azure("5")

        monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_azure_ml)
        monkeypatch.setitem(
            sys.modules, "azure.ai.ml.constants", fake_azure_ml.constants
        )
        monkeypatch.setitem(sys.modules, "azure.ai.ml.entities", fake_azure_ml.entities)
        monkeypatch.setitem(sys.modules, "azure.identity", fake_azure_identity)

        summary = make_summary(0.91, 0.89)
        version = register_model_azure_sdk(
            model_path=tmp_path / "model.pth",
            model_name="ngt-sign-language",
            evaluation_summary=summary,
            class_names=["A", "B"],
        )

        assert version == "5"
        mock_ml_client.models.create_or_update.assert_called_once()

    def test_falls_back_to_cli_credential_when_managed_identity_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_env(monkeypatch)
        _, _, fake_azure_ml, fake_azure_identity = _make_fake_azure("3")
        fake_azure_identity.ManagedIdentityCredential.side_effect = Exception("no MI")

        monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_azure_ml)
        monkeypatch.setitem(
            sys.modules, "azure.ai.ml.constants", fake_azure_ml.constants
        )
        monkeypatch.setitem(sys.modules, "azure.ai.ml.entities", fake_azure_ml.entities)
        monkeypatch.setitem(sys.modules, "azure.identity", fake_azure_identity)

        version = register_model_azure_sdk(
            model_path=tmp_path / "model.pth",
            model_name="ngt-sign-language",
            evaluation_summary=make_summary(),
            class_names=["A", "B"],
        )

        assert version == "3"
        fake_azure_identity.AzureCliCredential.assert_called_once()

    def test_model_tags_include_metrics_and_class_info(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_env(monkeypatch)
        _, _, fake_azure_ml, fake_azure_identity = _make_fake_azure("1")
        captured_model: list = []

        def capturing_model(**kwargs):
            captured_model.append(kwargs)
            return MagicMock()

        fake_azure_ml.entities.Model = capturing_model

        monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_azure_ml)
        monkeypatch.setitem(
            sys.modules, "azure.ai.ml.constants", fake_azure_ml.constants
        )
        monkeypatch.setitem(sys.modules, "azure.ai.ml.entities", fake_azure_ml.entities)
        monkeypatch.setitem(sys.modules, "azure.identity", fake_azure_identity)

        register_model_azure_sdk(
            model_path=tmp_path / "model.pth",
            model_name="ngt-sign-language",
            evaluation_summary=make_summary(0.92, 0.90),
            class_names=["A", "B", "C"],
        )

        assert len(captured_model) == 1
        tags = captured_model[0]["tags"]
        assert tags["accuracy"] == "0.92"
        assert tags["f1_macro"] == "0.9"
        assert tags["num_classes"] == "3"
        assert tags["class_names"] == "A,B,C"

    def test_missing_azure_environment_variables_raise_descriptive_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in (
            "AZUREML_ARM_SUBSCRIPTION",
            "AZUREML_ARM_RESOURCEGROUP",
            "AZUREML_ARM_WORKSPACE_NAME",
        ):
            monkeypatch.delenv(name, raising=False)

        _, _, fake_azure_ml, fake_azure_identity = _make_fake_azure("3")
        monkeypatch.setitem(sys.modules, "azure.ai.ml", fake_azure_ml)
        monkeypatch.setitem(
            sys.modules, "azure.ai.ml.constants", fake_azure_ml.constants
        )
        monkeypatch.setitem(sys.modules, "azure.ai.ml.entities", fake_azure_ml.entities)
        monkeypatch.setitem(sys.modules, "azure.identity", fake_azure_identity)

        with pytest.raises(RuntimeError, match="requires these Azure ML job"):
            register_model_azure_sdk(
                model_path=tmp_path / "model.pth",
                model_name="ngt-sign-language",
                evaluation_summary=make_summary(),
                class_names=["A", "B"],
            )


# ---------------------------------------------------------------------------
# run_model_gate_and_register
# ---------------------------------------------------------------------------


class TestRunModelGateAndRegister:
    def test_gate_failed_returns_failed_result_without_registering(
        self, tmp_path: Path
    ) -> None:
        summary = make_summary(accuracy=0.70, f1_macro=0.65)

        with patch(
            "sign_language_training.model_registration.register_model_azure_sdk"
        ) as mock_register:
            result = run_model_gate_and_register(
                evaluation_summary=summary,
                model_path=tmp_path / "model.pth",
                model_name="ngt-sign-language",
                class_names=["A", "B"],
                accuracy_threshold=0.85,
                f1_threshold=0.80,
            )

        assert result.passed is False
        assert result.registered_version is None
        mock_register.assert_not_called()

    def test_gate_passed_calls_registration_and_returns_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AZUREML_RUN_ID", "test_run_123")
        summary = make_summary(accuracy=0.92, f1_macro=0.90)

        with patch(
            "sign_language_training.model_registration.register_model_azure_sdk",
            return_value="4",
        ) as mock_register:
            result = run_model_gate_and_register(
                evaluation_summary=summary,
                model_path=tmp_path / "model.pth",
                model_name="ngt-sign-language",
                class_names=["A", "B"],
                accuracy_threshold=0.85,
                f1_threshold=0.80,
            )

        assert result.passed is True
        assert result.registered_version == "4"
        mock_register.assert_called_once_with(
            model_path=tmp_path / "model.pth",
            model_name="ngt-sign-language",
            evaluation_summary=summary,
            class_names=["A", "B"],
        )

    def test_gate_result_carries_threshold_values(self, tmp_path: Path) -> None:
        summary = make_summary(accuracy=0.92, f1_macro=0.90)

        with patch(
            "sign_language_training.model_registration.register_model_azure_sdk",
            return_value="2",
        ):
            result = run_model_gate_and_register(
                evaluation_summary=summary,
                model_path=tmp_path / "model.pth",
                model_name="ngt-sign-language",
                class_names=["A", "B"],
                accuracy_threshold=0.85,
                f1_threshold=0.80,
            )

        assert result.accuracy_threshold == 0.85
        assert result.f1_threshold == 0.80
        assert result.accuracy == 0.92
        assert result.f1_macro == 0.90
