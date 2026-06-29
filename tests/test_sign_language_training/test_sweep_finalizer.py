"""Tests for Azure ML sweep finalization helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sign_language_training.orchestration import sweep_submitter
from sign_language_training.orchestration.sweep_finalizer import (
    finalize_completed_sweeps,
    get_current_promoted_model,
    is_better_than_current,
)


class FakeModels:
    """Minimal fake Azure ML model operations."""

    def __init__(self, models: list[SimpleNamespace]) -> None:
        self.models = models
        self.updated: list[SimpleNamespace] = []
        self.archived: list[tuple[str, str]] = []

    def list(self, name: str) -> list[SimpleNamespace]:
        """Return fake model versions."""
        return self.models

    def create_or_update(self, model: SimpleNamespace) -> SimpleNamespace:
        """Record fake model updates."""
        self.updated.append(model)
        return model

    def archive(self, name: str, version: str) -> None:
        """Record fake model archives."""
        self.archived.append((name, version))


class FakeJobs:
    """Minimal fake Azure ML job operations."""

    def __init__(self, jobs: list[SimpleNamespace]) -> None:
        self.jobs = jobs
        self.updated: list[SimpleNamespace] = []

    def list(self) -> list[SimpleNamespace]:
        """Return fake jobs."""
        return self.jobs

    def create_or_update(self, job: SimpleNamespace) -> SimpleNamespace:
        """Record fake job updates."""
        self.updated.append(job)
        return job


def test_is_better_than_current_uses_f1_then_accuracy() -> None:
    best = SimpleNamespace(tags={"f1_macro": "0.90", "accuracy": "0.80"})
    current = SimpleNamespace(tags={"f1_macro": "0.89", "accuracy": "0.99"})

    assert is_better_than_current(best, current) is True


def test_get_current_promoted_model_returns_highest_scoring_promoted() -> None:
    low = SimpleNamespace(version="1", tags={"promoted": "true", "f1_macro": "0.70"})
    high = SimpleNamespace(version="2", tags={"promoted": "true", "f1_macro": "0.90"})
    client = SimpleNamespace(models=FakeModels([low, high]))

    assert get_current_promoted_model(client, "ngt-sign-language") is high


def test_finalize_completed_sweep_promotes_best_and_archives_rest() -> None:
    sweep = SimpleNamespace(
        name="sweep-1",
        experiment_name="exp",
        status="Completed",
        tags={"purpose": "retraining-sweep", "finalization_status": "pending"},
    )
    current = SimpleNamespace(
        version="1",
        tags={"promoted": "true", "f1_macro": "0.80", "accuracy": "0.80"},
    )
    weak = SimpleNamespace(
        version="2",
        tags={"sweep_id": "sweep-1", "f1_macro": "0.81", "accuracy": "0.82"},
    )
    best = SimpleNamespace(
        version="3",
        tags={"sweep_id": "sweep-1", "f1_macro": "0.91", "accuracy": "0.85"},
    )
    models = FakeModels([current, weak, best])
    jobs = FakeJobs([sweep])
    client = SimpleNamespace(models=models, jobs=jobs)

    results = finalize_completed_sweeps(client, "exp", "ngt-sign-language")

    assert len(results) == 1
    assert results[0].promoted_version == "3"
    assert results[0].archived_versions == ["2"]
    assert best.tags["promoted"] == "true"
    assert current.tags["promoted"] == "false"
    assert models.archived == [("ngt-sign-language", "2")]
    assert sweep.tags["finalization_status"] == "promoted"


def test_finalize_completed_sweep_does_not_promote_weaker_best() -> None:
    sweep = SimpleNamespace(
        name="sweep-1",
        experiment_name="exp",
        status="Completed",
        tags={"purpose": "retraining-sweep", "finalization_status": "pending"},
    )
    current = SimpleNamespace(
        version="1",
        tags={"promoted": "true", "f1_macro": "0.95", "accuracy": "0.90"},
    )
    candidate = SimpleNamespace(
        version="2",
        tags={"sweep_id": "sweep-1", "f1_macro": "0.91", "accuracy": "0.85"},
    )
    client = SimpleNamespace(
        models=FakeModels([current, candidate]),
        jobs=FakeJobs([sweep]),
    )

    results = finalize_completed_sweeps(client, "exp", "ngt-sign-language")

    assert results[0].promoted_version is None
    assert current.tags["promoted"] == "true"
    assert sweep.tags["finalization_status"] == "not_better_than_current"


def test_submit_retraining_sweep_preprocesses_before_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[dict[str, object]] = []
    sweep_nodes: list[object] = []
    submitted_jobs: list[object] = []

    class FakeCommand:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def __call__(self, **kwargs: object) -> object:
            return SimpleNamespace(
                outputs=SimpleNamespace(
                    augmented_train="augmented-train",
                    val_data="val-data",
                    test_data="test-data",
                )
            )

        def sweep(self, **kwargs: object) -> object:
            node = SimpleNamespace(
                kwargs=kwargs,
                inputs=SimpleNamespace(data=None, val_data=None),
                resources=None,
            )
            sweep_nodes.append(node)
            return node

    class FakeJobs:
        def create_or_update(self, job: object) -> object:
            submitted_jobs.append(job)
            return SimpleNamespace(name="sweep-pipeline", studio_url="https://job")

    def fake_command(**kwargs: object) -> FakeCommand:
        commands.append(kwargs)
        return FakeCommand(**kwargs)

    def fake_pipeline(**kwargs: object):
        def decorator(func):
            def wrapper(**call_kwargs: object) -> dict[str, object]:
                func(**call_kwargs)
                return {"pipeline_inputs": call_kwargs}

            return wrapper

        return decorator

    monkeypatch.setattr(sweep_submitter, "command", fake_command)
    monkeypatch.setattr(sweep_submitter, "pipeline", fake_pipeline)
    monkeypatch.setattr(sweep_submitter, "Input", lambda **kwargs: {"input": kwargs})
    monkeypatch.setattr(sweep_submitter, "Output", lambda **kwargs: {"output": kwargs})
    monkeypatch.setattr(
        sweep_submitter,
        "JobResourceConfiguration",
        lambda **kwargs: {"resources": kwargs},
    )
    monkeypatch.setattr(sweep_submitter, "Choice", lambda **kwargs: {"choice": kwargs})
    monkeypatch.setattr(
        sweep_submitter, "LogUniform", lambda **kwargs: {"log_uniform": kwargs}
    )
    monkeypatch.setattr(
        sweep_submitter, "get_client", lambda: SimpleNamespace(jobs=FakeJobs())
    )
    monkeypatch.setattr(sweep_submitter, "resolve_compute_target", lambda client: "cpu")
    monkeypatch.setattr(sweep_submitter, "resolve_environment", lambda client: "env")
    monkeypatch.setattr(sweep_submitter, "resolve_instance_type", lambda: "gpu")
    monkeypatch.setattr(
        sweep_submitter,
        "pretrained_checkpoint_reference_or_path",
        lambda: "azureml:checkpoint:1",
    )

    submitted = sweep_submitter.submit_retraining_sweep(
        experiment_name="exp",
        data_asset="azureml:ngt-raw:6",
        ngt_raw_version="6",
        max_total_trials=2,
        max_concurrent_trials=1,
    )

    assert submitted.name == "sweep-pipeline"
    assert [command["display_name"] for command in commands] == [
        "NGT offline augmentation",
        "NGT sign-language sweep trial",
    ]
    assert "--val-dir ${{inputs.val_data}}" in str(commands[1]["command"])
    assert "--val-split" not in str(commands[1]["command"])
    assert "assert torch.cuda.is_available()" in str(commands[1]["command"])
    assert "export MLFLOW_ENABLED=true" in str(commands[1]["command"])
    assert sweep_nodes[0].resources == {
        "resources": {"instance_type": "gpu", "instance_count": 1}
    }
    assert submitted_jobs[0]["tags"]["purpose"] == "retraining-sweep"
    assert submitted_jobs[0]["tags"]["raw_data_version"] == "6"


def test_sweep_trial_prefix_does_not_require_cuda_for_cpu() -> None:
    prefix = sweep_submitter.build_sweep_trial_prefix(
        instance_type="cpu-xl",
        experiment_name="experiment",
        model_name="model",
    )

    assert "assert torch.cuda.is_available()" not in prefix
    assert "export MLFLOW_ENABLED=true" in prefix
    assert "export MLFLOW_EXPERIMENT_NAME=experiment" in prefix
    assert "export MODEL_REGISTRY_NAME=model" in prefix
