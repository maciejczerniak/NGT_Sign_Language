"""Tests for the on-prem MLflow-backed retraining trigger policy.

Mirrors ``test_orchestration.py``'s style: ``tmp_path`` for the dataset and
state file, an in-test ``_write_image`` helper, and ``monkeypatch.setattr`` at
the *import site* (the ``mlflow_trigger_policy`` module) so the optional MLflow
dependency is never touched. The injected ``runner`` stands in for the real
subprocess pipeline call.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sign_language_training.orchestration import mlflow_trigger_policy
from sign_language_training.orchestration.mlflow_trigger_policy import (
    MlflowTriggerConfig,
    build_decision,
    evaluate_and_maybe_train,
)
from sign_language_training.orchestration.training_state import load_state


def _write_image(path: Path) -> None:
    """Write a placeholder image file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def _config(tmp_path: Path, *, min_new_images: int = 2) -> MlflowTriggerConfig:
    """Build a trigger config rooted under ``tmp_path``."""
    return MlflowTriggerConfig(
        data_dir=tmp_path / "data",
        state_path=tmp_path / "state" / "trigger.json",
        model_name="ngt-sign-language",
        min_new_images=min_new_images,
        interval_days=7,
    )


def _patch_last_trained(
    monkeypatch: pytest.MonkeyPatch,
    when: datetime | None,
) -> None:
    """Patch the MLflow timestamp source at its import site."""
    monkeypatch.setattr(
        mlflow_trigger_policy,
        "latest_registered_version_timestamp",
        lambda *args, **kwargs: when,
    )


def _counting_runner(calls: dict[str, int], version: str | None = "7"):
    """Return a runner that records invocations and returns ``version``."""

    def _run() -> str | None:
        calls["n"] = calls.get("n", 0) + 1
        return version

    return _run


class TestMlflowTriggerPolicy:
    def test_first_run_treats_all_images_as_new_data_change(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "one.jpg")
        _write_image(config.data_dir / "A" / "two.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))
        calls: dict[str, int] = {}

        decision = evaluate_and_maybe_train(
            config=config, runner=_counting_runner(calls)
        )

        assert decision.should_train is True
        assert decision.reason == "data_change"
        assert decision.new_image_count == 2
        assert decision.candidate_version == "7"
        assert calls["n"] == 1

    def test_no_change_recent_training_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))
        calls: dict[str, int] = {}

        # Force the first run so the baseline state is persisted (one image is
        # below the data-change threshold and would not trigger on its own).
        evaluate_and_maybe_train(
            config=config, runner=_counting_runner(calls), force=True
        )
        # Second run: same files, recent training -> skip.
        decision = evaluate_and_maybe_train(
            config=config, runner=_counting_runner(calls)
        )

        assert decision.should_train is False
        assert decision.reason is None
        assert decision.new_image_count == 0

    def test_interval_elapsed_triggers_scheduled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=100)
        _write_image(config.data_dir / "A" / "one.jpg")
        # Seed baseline (forced) so no images count as new on the next run.
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))
        evaluate_and_maybe_train(config=config, runner=_counting_runner({}), force=True)

        _patch_last_trained(
            monkeypatch, datetime.now(timezone.utc) - timedelta(days=10)
        )
        calls: dict[str, int] = {}

        decision = evaluate_and_maybe_train(
            config=config, runner=_counting_runner(calls)
        )

        assert decision.should_train is True
        assert decision.reason == "scheduled"
        assert calls["n"] == 1

    def test_data_change_takes_precedence_over_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))
        # Force first run so a one-image baseline is persisted.
        evaluate_and_maybe_train(config=config, runner=_counting_runner({}), force=True)

        # Add enough new images to trip data_change; keep training "recent".
        _write_image(config.data_dir / "A" / "two.jpg")
        _write_image(config.data_dir / "A" / "three.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))

        decision = evaluate_and_maybe_train(config=config, runner=_counting_runner({}))

        assert decision.reason == "data_change"
        assert decision.new_image_count == 2

    def test_no_registered_version_triggers_scheduled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=100)
        _write_image(config.data_dir / "A" / "one.jpg")
        # Seed baseline (forced) so image count alone won't trigger data_change.
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))
        evaluate_and_maybe_train(config=config, runner=_counting_runner({}), force=True)

        # No registered version at all -> interval is due.
        _patch_last_trained(monkeypatch, None)
        decision = evaluate_and_maybe_train(config=config, runner=_counting_runner({}))

        assert decision.should_train is True
        assert decision.reason == "scheduled"

    def test_force_triggers_manual_regardless(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=100)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))
        calls: dict[str, int] = {}

        decision = evaluate_and_maybe_train(
            config=config, runner=_counting_runner(calls), force=True
        )

        assert decision.should_train is True
        assert decision.reason == "manual"
        assert calls["n"] == 1

    def test_state_persisted_after_trigger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=1)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))

        evaluate_and_maybe_train(config=config, runner=_counting_runner({}))

        saved = load_state(config.state_path).last_submitted_training
        assert saved is not None
        assert saved.files == ["A/one.jpg"]
        assert saved.image_count == 1

    def test_gate_failure_reports_no_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=1)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))

        # Runner returns None -> gate failed / no candidate registered.
        decision = evaluate_and_maybe_train(
            config=config, runner=_counting_runner({}, version=None)
        )

        assert decision.should_train is True
        assert decision.candidate_version is None
        assert "no @candidate" in decision.message


class TestBuildDecision:
    """The pure decision core shared by the checker and the Airflow DAG."""

    def test_build_decision_does_not_run_or_persist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "one.jpg")
        _write_image(config.data_dir / "A" / "two.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))

        decision = build_decision(config=config)

        assert decision.should_train is True
        assert decision.reason == "data_change"
        assert decision.candidate_version is None
        # No state file written — build_decision is side-effect free.
        assert not config.state_path.exists()

    def test_build_decision_skip_when_quiet(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=999999)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))

        decision = build_decision(config=config, force=False)

        assert decision.should_train is False
        assert decision.reason is None

    def test_build_decision_force(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = _config(tmp_path, min_new_images=999999)
        _write_image(config.data_dir / "A" / "one.jpg")
        _patch_last_trained(monkeypatch, datetime.now(timezone.utc))

        decision = build_decision(config=config, force=True)

        assert decision.should_train is True
        assert decision.reason == "manual"
