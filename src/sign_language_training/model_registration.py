"""Model gate and Azure ML model registration for the NGT training workflow.

Registration uses the Azure ML SDK v2 directly (not MLflow) to avoid the
``mlflow``/``mlflow-skinny`` version conflict that produces 404 errors on
``/api/2.0/mlflow/logged-models/search``.

Inside Azure ML jobs, ``AZUREML_ARM_*`` environment variables are always
present and ``ManagedIdentityCredential`` succeeds without user interaction.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from sign_language_training.model_evaluation import EvaluationSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateResult:
    """Outcome of the model gate check and optional registration.

    Args:
        passed: Whether the model met both quality thresholds.
        accuracy: Validation accuracy achieved by the model.
        f1_macro: Macro-averaged F1 score achieved by the model.
        accuracy_threshold: Minimum accuracy required to pass the gate.
        f1_threshold: Minimum macro F1 score required to pass the gate.
        registered_version: The Azure ML model version string if the model
            was registered, otherwise ``None``.
    """

    passed: bool
    accuracy: float
    f1_macro: float
    accuracy_threshold: float
    f1_threshold: float
    registered_version: str | None = None

    def __str__(self) -> str:
        """Return a human-readable summary of the gate result.

        Returns:
            Multi-line string showing pass/fail status, metric values
                with thresholds, and the registered version if applicable.
        """
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Model gate: {status}",
            f"  accuracy : {self.accuracy:.4f} (threshold={self.accuracy_threshold:.4f})",
            f"  f1_macro : {self.f1_macro:.4f} (threshold={self.f1_threshold:.4f})",
        ]
        if self.registered_version:
            lines.append(f"  registered version: {self.registered_version}")
        return "\n".join(lines)


def evaluate_model_gate(
    evaluation_summary: EvaluationSummary,
    accuracy_threshold: float,
    f1_threshold: float,
) -> bool:
    """Return ``True`` when the model meets both quality thresholds.

    Args:
        evaluation_summary: The :class:`~sign_language_training.model_evaluation.EvaluationSummary`
            containing the model's validation metrics.
        accuracy_threshold: Minimum validation accuracy required to pass.
        f1_threshold: Minimum macro F1 score required to pass.

    Returns:
        ``True`` if both ``accuracy >= accuracy_threshold`` and
            ``f1_macro >= f1_threshold``, ``False`` otherwise.
    """
    accuracy_ok = evaluation_summary.accuracy >= accuracy_threshold
    f1_ok = evaluation_summary.f1_macro >= f1_threshold

    logger.info(
        "Gate check — accuracy: %.4f >= %.4f ? %s",
        evaluation_summary.accuracy,
        accuracy_threshold,
        "YES" if accuracy_ok else "NO",
    )
    logger.info(
        "Gate check — f1_macro: %.4f >= %.4f ? %s",
        evaluation_summary.f1_macro,
        f1_threshold,
        "YES" if f1_ok else "NO",
    )

    return accuracy_ok and f1_ok


def _version_from_run_id(run_id: str) -> str:
    """Derive a unique positive integer version string from a run ID.

    Uses the first 8 hex characters of the SHA-256 digest of ``run_id``,
    giving a value in [1, 4_294_967_296]. Collision probability across 4
    concurrent trials is approximately 4e-9 — effectively zero.

    Args:
        run_id: The Azure ML run ID string, e.g.
            ``"red_truck_8jccp8n0hs_0"``.

    Returns:
        A deterministic positive integer string derived from the
            SHA-256 hash of ``run_id``.
    """
    import hashlib

    digest = hashlib.sha256(run_id.encode()).hexdigest()[:8]
    return str(int(digest, 16) or 1)


def register_model_azure_sdk(
    model_path: str | Path,
    model_name: str,
    evaluation_summary: EvaluationSummary,
    class_names: list[str],
) -> str:
    """Register the model in Azure ML using the SDK v2 directly.

    Derives the model version from a SHA-256 hash of ``AZUREML_RUN_ID``,
    producing a deterministic positive integer that is unique per trial and
    safe for concurrent sweep runs. Attaches evaluation metrics, class names,
    and sweep hyperparameters as tags on the registered model version.

    Args:
        model_path: Path to the ``.pth`` checkpoint file to register.
        model_name: Azure ML model registry name, e.g.
            ``"ngt-sign-language"``.
        evaluation_summary: The :class:`~sign_language_training.model_evaluation.EvaluationSummary`
            providing accuracy and F1 metrics to tag the registered model with.
        class_names: List of class label strings stored as a tag.

    Returns:
        The registered model version string assigned by Azure ML.

    Raises:
        RuntimeError: If any of the required ``AZUREML_ARM_*``
            environment variables are missing.
    """
    from azure.ai.ml import MLClient
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from typing import Any

    from azure.identity import AzureCliCredential, ManagedIdentityCredential

    credential: Any
    try:
        credential = ManagedIdentityCredential()
    except Exception:
        credential = AzureCliCredential()

    required_env = {
        "AZUREML_ARM_SUBSCRIPTION": os.environ.get("AZUREML_ARM_SUBSCRIPTION"),
        "AZUREML_ARM_RESOURCEGROUP": os.environ.get("AZUREML_ARM_RESOURCEGROUP"),
        "AZUREML_ARM_WORKSPACE_NAME": os.environ.get("AZUREML_ARM_WORKSPACE_NAME"),
    }
    missing = [name for name, value in required_env.items() if not value]
    if missing:
        raise RuntimeError(
            "Azure ML model registration requires these Azure ML job environment "
            f"variables: {', '.join(missing)}"
        )

    ml_client = MLClient(
        credential=credential,
        subscription_id=required_env["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=required_env["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=required_env["AZUREML_ARM_WORKSPACE_NAME"],
    )

    run_id = os.environ.get("AZUREML_RUN_ID", "local")
    version = _version_from_run_id(run_id)

    logger.info(
        "Registering model '%s' version %s (derived from run_id='%s')",
        model_name,
        version,
        run_id,
    )

    model = Model(
        path=str(model_path),
        name=model_name,
        version=version,
        type=AssetTypes.CUSTOM_MODEL,
        description="NGT sign language EfficientNet-B0",
        tags={
            "accuracy": str(round(evaluation_summary.accuracy, 4)),
            "f1_macro": str(round(evaluation_summary.f1_macro, 4)),
            "num_classes": str(len(class_names)),
            "class_names": ",".join(class_names),
            "run_id": run_id,
            "learning_rate": os.environ.get("AZUREML_PARAMETER_learning_rate", ""),
            "batch_size": os.environ.get("AZUREML_PARAMETER_batch_size", ""),
            "patience": os.environ.get("AZUREML_PARAMETER_patience", ""),
            "sweep_id": os.environ.get("AZUREML_ROOT_RUN_ID", ""),
        },
    )

    registered = ml_client.models.create_or_update(model)
    logger.info(
        "Registered model '%s' version %s in Azure ML workspace",
        model_name,
        registered.version,
    )
    return str(registered.version)


def _is_azure_ml_environment() -> bool:
    """Return ``True`` when running inside an Azure ML job.

    Detected by the presence of the ``AZUREML_RUN_ID`` environment variable,
    which is always set by the Azure ML job runtime.

    Returns:
        ``True`` if ``AZUREML_RUN_ID`` is set and non-empty.
    """
    return bool(os.environ.get("AZUREML_RUN_ID"))


def run_model_gate_and_register(
    evaluation_summary: EvaluationSummary,
    model_path: str | Path,
    model_name: str,
    class_names: list[str],
    accuracy_threshold: float,
    f1_threshold: float,
) -> GateResult:
    """Run the model gate check and conditionally register the model in Azure ML.

    Registration only runs inside Azure ML jobs, detected via the
    ``AZUREML_RUN_ID`` environment variable. Local runs log a message and
    return ``registered_version=None``.

    Args:
        evaluation_summary: The :class:`~sign_language_training.model_evaluation.EvaluationSummary`
            containing the model's validation metrics.
        model_path: Path to the best ``.pth`` checkpoint to register if
            the gate passes.
        model_name: Azure ML model registry name.
        class_names: Ordered list of class label strings.
        accuracy_threshold: Minimum validation accuracy required to pass
            the gate and trigger registration.
        f1_threshold: Minimum macro F1 score required to pass the gate
            and trigger registration.

    Returns:
        A :class:`GateResult` describing whether the gate passed and,
            if inside an Azure ML job and the gate passed, the registered model
            version string.
    """
    passed = evaluate_model_gate(
        evaluation_summary=evaluation_summary,
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold,
    )

    if not passed:
        logger.warning(
            "Model gate FAILED — model will NOT be registered. "
            "accuracy=%.4f (threshold=%.4f), f1_macro=%.4f (threshold=%.4f)",
            evaluation_summary.accuracy,
            accuracy_threshold,
            evaluation_summary.f1_macro,
            f1_threshold,
        )
        return GateResult(
            passed=False,
            accuracy=evaluation_summary.accuracy,
            f1_macro=evaluation_summary.f1_macro,
            accuracy_threshold=accuracy_threshold,
            f1_threshold=f1_threshold,
            registered_version=None,
        )

    logger.info("Model gate PASSED — proceeding with registration")

    if _is_azure_ml_environment():
        version = register_model_azure_sdk(
            model_path=model_path,
            model_name=model_name,
            evaluation_summary=evaluation_summary,
            class_names=class_names,
        )
    else:
        logger.info(
            "Local run — skipping Azure ML model registration. "
            "Best checkpoint saved at: %s",
            model_path,
        )
        version = None

    return GateResult(
        passed=True,
        accuracy=evaluation_summary.accuracy,
        f1_macro=evaluation_summary.f1_macro,
        accuracy_threshold=accuracy_threshold,
        f1_threshold=f1_threshold,
        registered_version=version,
    )
