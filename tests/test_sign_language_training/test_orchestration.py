"""Tests for Azure ML retraining orchestration helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from sign_language_training.orchestration import pipeline_submitter, trigger_policy
from sign_language_training.orchestration.dataset_inventory import (
    DatasetInventory,
    build_dataset_inventory,
    count_new_images,
    count_removed_images,
)
from sign_language_training.orchestration.pipeline_submitter import (
    SubmittedPipeline,
    find_cached_augmented_asset,
    submit_retraining_pipeline,
)
from sign_language_training.orchestration.training_state import (
    LastTrainingState,
    TrainingTriggerState,
    load_state,
    save_state,
)
from sign_language_training.orchestration.trigger_policy import (
    TriggerPolicyConfig,
    evaluate_and_maybe_submit,
    find_active_retraining_job,
    find_latest_completed_retraining_job,
    get_latest_raw_data_snapshot,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")


def _config(tmp_path: Path, *, min_new_images: int = 2) -> TriggerPolicyConfig:
    return TriggerPolicyConfig(
        data_dir=tmp_path / "data",
        state_path=tmp_path / "state" / "trigger.json",
        raw_data_asset="azureml:ngt-raw:1",
        raw_data_version="1",
        min_new_images=min_new_images,
        interval_days=30,
        experiment_name="test-exp",
    )


def _state(
    *,
    submitted_at: str | None = None,
    files: list[str] | None = None,
) -> TrainingTriggerState:
    return TrainingTriggerState(
        last_submitted_training=LastTrainingState(
            job_name="previous-job",
            studio_url=None,
            submitted_at=submitted_at or datetime.now(timezone.utc).isoformat(),
            reason="scheduled",
            raw_data_asset="azureml:ngt-raw:1",
            raw_data_version="1",
            image_count=len(files or []),
            manifest_hash="hash",
            files=files or [],
        )
    )


class TestDatasetInventory:
    def test_build_dataset_inventory_filters_and_sorts_images(
        self,
        tmp_path: Path,
    ) -> None:
        data_dir = tmp_path / "raw"
        _write_image(data_dir / "B" / "two.PNG")
        _write_image(data_dir / "A" / "one.jpg")
        (data_dir / "A" / "notes.txt").write_text("ignore", encoding="utf-8")

        inventory = build_dataset_inventory(data_dir)
        repeated = build_dataset_inventory(data_dir)

        assert inventory.root == str(data_dir.resolve())
        assert inventory.image_count == 2
        assert inventory.files == ["A/one.jpg", "B/two.PNG"]
        assert inventory.manifest_hash == repeated.manifest_hash

        (data_dir / "A" / "one.jpg").write_bytes(b"different")
        changed = build_dataset_inventory(data_dir)
        assert changed.manifest_hash != inventory.manifest_hash

    def test_build_dataset_inventory_rejects_missing_or_file_path(
        self,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(FileNotFoundError):
            build_dataset_inventory(tmp_path / "missing")

        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory", encoding="utf-8")

        with pytest.raises(NotADirectoryError):
            build_dataset_inventory(file_path)

    def test_build_dataset_inventory_rejects_empty_dataset(
        self, tmp_path: Path
    ) -> None:
        data_dir = tmp_path / "raw"
        data_dir.mkdir()

        with pytest.raises(ValueError, match="contains no supported images"):
            build_dataset_inventory(data_dir)

    def test_count_new_images_handles_empty_and_previous_manifests(self) -> None:
        inventory = DatasetInventory(
            root="/data",
            image_count=3,
            files=["A/1.jpg", "A/2.jpg", "B/3.jpg"],
            manifest_hash="hash",
        )

        assert count_new_images(inventory, None) == 3
        assert count_new_images(inventory, []) == 3
        assert count_new_images(inventory, ["A/1.jpg", "B/3.jpg"]) == 1
        assert count_removed_images(inventory, ["A/1.jpg", "Z/old.jpg"]) == 1


class TestTrainingState:
    def test_load_state_returns_empty_state_for_missing_file(
        self,
        tmp_path: Path,
    ) -> None:
        assert load_state(tmp_path / "missing.json") == TrainingTriggerState()

    def test_save_state_creates_parent_and_round_trips(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "nested" / "state.json"
        state = _state(files=["A/1.jpg"])

        save_state(path, state)

        assert path.exists()
        assert load_state(path) == state

    def test_save_state_round_trips_empty_state(self, tmp_path: Path) -> None:
        path = tmp_path / "state.json"

        save_state(path, TrainingTriggerState())

        assert load_state(path) == TrainingTriggerState()

    def test_load_state_returns_empty_state_for_invalid_json(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "state.json"
        path.write_text("{not-json", encoding="utf-8")

        assert load_state(path) == TrainingTriggerState()


class TestAzureMetadataTriggerHelpers:
    def test_get_latest_raw_data_snapshot_uses_highest_numeric_version(self) -> None:
        ml_client = SimpleNamespace(
            data=SimpleNamespace(
                list=lambda name: [
                    SimpleNamespace(name=name, version="2", tags={"image_count": "20"}),
                    SimpleNamespace(
                        name=name,
                        version="10",
                        tags={"image_count": "35", "manifest_hash": "abc"},
                    ),
                ]
            )
        )

        snapshot = get_latest_raw_data_snapshot(ml_client, "ngt-raw")

        assert snapshot.reference == "azureml:ngt-raw:10"
        assert snapshot.image_count == 35
        assert snapshot.manifest_hash == "abc"

    def test_get_latest_raw_data_snapshot_rejects_missing_versions(self) -> None:
        ml_client = SimpleNamespace(data=SimpleNamespace(list=lambda name: []))

        with pytest.raises(ValueError, match="No Azure ML data asset versions"):
            get_latest_raw_data_snapshot(ml_client, "ngt-raw")

    def test_find_latest_completed_retraining_job_uses_tags_and_created_at(
        self,
    ) -> None:
        older = datetime(2026, 1, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 1, 2, tzinfo=timezone.utc)
        jobs = [
            SimpleNamespace(
                name="ignored-active",
                experiment_name="exp",
                status="Running",
                tags={"purpose": "retraining"},
                creation_context=SimpleNamespace(created_at=newer),
            ),
            SimpleNamespace(
                name="ignored-other-purpose",
                experiment_name="exp",
                status="Completed",
                tags={"purpose": "smoke"},
                creation_context=SimpleNamespace(created_at=newer),
            ),
            SimpleNamespace(
                name="old",
                experiment_name="exp",
                status="Completed",
                tags={"purpose": "retraining", "trigger_image_count": "10"},
                creation_context=SimpleNamespace(created_at=older),
            ),
            SimpleNamespace(
                name="new",
                experiment_name="exp",
                status="Completed",
                tags={
                    "purpose": "retraining-sweep",
                    "raw_data_asset": "azureml:ngt-raw:3",
                    "raw_data_version": "3",
                    "trigger_image_count": "22",
                },
                creation_context=SimpleNamespace(created_at=newer),
                studio_url="https://studio/job",
            ),
        ]
        ml_client = SimpleNamespace(jobs=SimpleNamespace(list=lambda: jobs))

        snapshot = find_latest_completed_retraining_job(ml_client, "exp")

        assert snapshot is not None
        assert snapshot.name == "new"
        assert snapshot.raw_data_asset == "azureml:ngt-raw:3"
        assert snapshot.raw_data_version == "3"
        assert snapshot.trigger_image_count == 22
        assert snapshot.studio_url == "https://studio/job"

    def test_find_active_retraining_job_ignores_scheduled_checker(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        jobs = [
            SimpleNamespace(
                name="daily-checker",
                experiment_name="exp",
                status="Running",
                tags={"purpose": "training-trigger"},
            ),
            SimpleNamespace(
                name="active-sweep",
                experiment_name="exp",
                status="Running",
                tags={"purpose": "retraining-sweep"},
            ),
        ]
        ml_client = SimpleNamespace(jobs=SimpleNamespace(list=lambda: jobs))
        monkeypatch.setattr(trigger_policy, "get_client", lambda: ml_client)

        active = find_active_retraining_job("exp")

        assert active is jobs[1]

    def test_find_active_retraining_job_returns_none_for_checker_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        checker = SimpleNamespace(
            name="daily-checker",
            experiment_name="exp",
            status="Running",
            tags={"purpose": "training-trigger"},
        )
        ml_client = SimpleNamespace(jobs=SimpleNamespace(list=lambda: [checker]))
        monkeypatch.setattr(trigger_policy, "get_client", lambda: ml_client)

        assert find_active_retraining_job("exp") is None

    def test_load_state_returns_empty_state_for_invalid_structure(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "state.json"
        path.write_text(
            '{"last_submitted_training": {"job_name": "incomplete"}}',
            encoding="utf-8",
        )

        assert load_state(path) == TrainingTriggerState()


class TestTriggerPolicy:
    @pytest.fixture(autouse=True)
    def _no_active_azure_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            trigger_policy,
            "find_active_retraining_job",
            lambda experiment_name: None,
        )

    def test_manual_without_force_does_not_submit(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _write_image(config.data_dir / "A" / "one.jpg")

        decision = evaluate_and_maybe_submit(reason="manual", config=config)

        assert decision.should_submit is False
        assert decision.reason == "manual"
        assert decision.current_image_count == 1
        assert decision.new_image_count == 1

    def test_force_submits_and_persists_state(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _config(tmp_path)
        _write_image(config.data_dir / "A" / "one.jpg")
        captured: dict[str, object] = {}

        def fake_submit_retraining_pipeline(**kwargs: object) -> SubmittedPipeline:
            captured.update(kwargs)
            return SubmittedPipeline(
                name="submitted-job",
                experiment_name="test-exp",
                studio_url="https://studio/job",
            )

        monkeypatch.setattr(
            trigger_policy,
            "submit_retraining_pipeline",
            fake_submit_retraining_pipeline,
        )

        decision = evaluate_and_maybe_submit(
            reason="manual",
            config=config,
            force=True,
        )

        assert decision.should_submit is True
        assert decision.submitted_job_name == "submitted-job"
        assert decision.studio_url == "https://studio/job"
        assert captured["force_preprocess"] is True
        assert captured["mlflow_enabled"] is True

        saved = load_state(config.state_path).last_submitted_training
        assert saved is not None
        assert saved.job_name == "submitted-job"
        assert saved.files == ["A/one.jpg"]

    def test_active_azure_job_skips_duplicate_submission(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _config(tmp_path, min_new_images=1)
        _write_image(config.data_dir / "A" / "one.jpg")

        monkeypatch.setattr(
            trigger_policy,
            "find_active_retraining_job",
            lambda experiment_name: SimpleNamespace(
                name="running-job",
                status="Running",
                studio_url="https://studio/job",
            ),
        )

        def fail_submit(**kwargs: object) -> SubmittedPipeline:
            raise AssertionError("Duplicate submission should be skipped.")

        monkeypatch.setattr(
            trigger_policy,
            "submit_retraining_pipeline",
            fail_submit,
        )

        decision = evaluate_and_maybe_submit(reason="data_change", config=config)

        assert decision.should_submit is False
        assert decision.submitted_job_name == "running-job"
        assert decision.studio_url == "https://studio/job"
        assert "already Running" in decision.message

    def test_data_change_below_threshold_skips(self, tmp_path: Path) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "old.jpg")
        _write_image(config.data_dir / "A" / "new.jpg")
        save_state(config.state_path, _state(files=["A/old.jpg"]))

        decision = evaluate_and_maybe_submit(reason="data_change", config=config)

        assert decision.should_submit is False
        assert decision.new_image_count == 1
        assert "threshold 2" in decision.message

    def test_data_change_at_threshold_submits_without_forcing_preprocess(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "old.jpg")
        _write_image(config.data_dir / "A" / "new-1.jpg")
        _write_image(config.data_dir / "A" / "new-2.jpg")
        save_state(config.state_path, _state(files=["A/old.jpg"]))
        captured: dict[str, object] = {}

        def fake_submit_retraining_pipeline(**kwargs: object) -> SubmittedPipeline:
            captured.update(kwargs)
            return SubmittedPipeline("data-job", "test-exp", None)

        monkeypatch.setattr(
            trigger_policy,
            "submit_retraining_pipeline",
            fake_submit_retraining_pipeline,
        )

        decision = evaluate_and_maybe_submit(reason="data_change", config=config)

        assert decision.should_submit is True
        assert decision.new_image_count == 2
        assert captured["force_preprocess"] is False

    def test_data_change_at_removed_threshold_submits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _config(tmp_path, min_new_images=2)
        _write_image(config.data_dir / "A" / "kept.jpg")
        save_state(
            config.state_path,
            _state(files=["A/kept.jpg", "A/removed-1.jpg", "A/removed-2.jpg"]),
        )

        monkeypatch.setattr(
            trigger_policy,
            "submit_retraining_pipeline",
            lambda **kwargs: SubmittedPipeline("removed-job", "test-exp", None),
        )

        decision = evaluate_and_maybe_submit(reason="data_change", config=config)

        assert decision.should_submit is True
        assert decision.submitted_job_name == "removed-job"

    def test_recent_scheduled_training_skips(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _write_image(config.data_dir / "A" / "one.jpg")
        save_state(config.state_path, _state(files=["A/one.jpg"]))

        decision = evaluate_and_maybe_submit(reason="scheduled", config=config)

        assert decision.should_submit is False
        assert "within 30 days" in decision.message

    def test_recent_scheduled_training_accepts_naive_timestamp(
        self, tmp_path: Path
    ) -> None:
        config = _config(tmp_path)
        _write_image(config.data_dir / "A" / "one.jpg")
        naive_timestamp = datetime.now().isoformat()
        save_state(
            config.state_path,
            _state(submitted_at=naive_timestamp, files=["A/one.jpg"]),
        )

        decision = evaluate_and_maybe_submit(reason="scheduled", config=config)

        assert decision.should_submit is False

    def test_old_scheduled_training_submits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = _config(tmp_path)
        _write_image(config.data_dir / "A" / "one.jpg")
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        save_state(config.state_path, _state(submitted_at=old_timestamp))

        monkeypatch.setattr(
            trigger_policy,
            "submit_retraining_pipeline",
            lambda **kwargs: SubmittedPipeline("scheduled-job", "test-exp", None),
        )

        decision = evaluate_and_maybe_submit(reason="scheduled", config=config)

        assert decision.should_submit is True
        assert decision.submitted_job_name == "scheduled-job"


class TestPipelineSubmitter:
    def test_find_cached_augmented_asset_returns_matching_version(self) -> None:
        ml_client = SimpleNamespace(
            data=SimpleNamespace(
                list=lambda name: [
                    SimpleNamespace(version="1", tags={"ngt_raw_version": "old"}),
                    SimpleNamespace(version="2", tags={"ngt_raw_version": "raw-2"}),
                ]
            )
        )

        assert (
            find_cached_augmented_asset(ml_client, "augmented", "raw-2")
            == "azureml:augmented:2"
        )

    def test_find_cached_augmented_asset_returns_none_on_error(self) -> None:
        class BrokenData:
            def list(self, name: str) -> list[object]:
                raise RuntimeError("Azure unavailable")

        ml_client = SimpleNamespace(data=BrokenData())

        assert find_cached_augmented_asset(ml_client, "augmented", "raw-2") is None

    def test_submit_retraining_pipeline_builds_and_submits_full_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        commands: list[dict[str, object]] = []
        command_calls: list[dict[str, object]] = []
        pipeline_options: dict[str, object] = {}
        submitted_jobs: list[object] = []

        class FakeCommand:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            def __call__(self, **kwargs: object) -> object:
                command_calls.append(kwargs)
                return SimpleNamespace(
                    outputs=SimpleNamespace(
                        augmented_train="augmented-output",
                        val_data="validation-output",
                    )
                )

        class FakeJobs:
            def create_or_update(self, pipeline_job: object) -> object:
                submitted_jobs.append(pipeline_job)
                return SimpleNamespace(name="pipeline-job", studio_url="https://job")

        def fake_command(**kwargs: object) -> FakeCommand:
            commands.append(kwargs)
            return FakeCommand(**kwargs)

        def fake_pipeline(**kwargs: object):
            pipeline_options.update(kwargs)

            def decorator(func):
                def wrapper(**call_kwargs: object) -> object:
                    func(**call_kwargs)
                    return {"pipeline_inputs": call_kwargs}

                return wrapper

            return decorator

        monkeypatch.setattr(pipeline_submitter, "command", fake_command)
        monkeypatch.setattr(
            pipeline_submitter,
            "Input",
            lambda **kwargs: {"input": kwargs},
        )
        monkeypatch.setattr(
            pipeline_submitter,
            "Output",
            lambda **kwargs: {"output": kwargs},
        )
        monkeypatch.setattr(
            pipeline_submitter,
            "JobResourceConfiguration",
            lambda **kwargs: {"resources": kwargs},
        )
        monkeypatch.setattr(pipeline_submitter, "pipeline", fake_pipeline)
        monkeypatch.setattr(
            pipeline_submitter, "get_client", lambda: SimpleNamespace(jobs=FakeJobs())
        )
        monkeypatch.setattr(
            pipeline_submitter, "resolve_compute_target", lambda client: "cpu"
        )
        monkeypatch.setattr(
            pipeline_submitter, "resolve_environment", lambda client: "env"
        )
        monkeypatch.setattr(
            pipeline_submitter, "resolve_instance_type", lambda: "Standard_DS3"
        )
        monkeypatch.setattr(
            pipeline_submitter,
            "raw_data_asset_reference",
            lambda: "azureml:default-data:1",
        )
        monkeypatch.setattr(
            pipeline_submitter,
            "pretrained_checkpoint_reference_or_path",
            lambda: "azureml:checkpoint:1",
        )

        submitted = submit_retraining_pipeline(
            experiment_name="exp",
            display_name="Display",
            data_asset=None,
            ngt_raw_version="7",
            pretrained_checkpoint=None,
            augmented_asset_name="augmented",
            augment_copies=3,
            batch_size=8,
            epochs=2,
            learning_rate=0.01,
            img_size=128,
            seed=123,
            patience=4,
            target_accuracy=0.9,
            expected_num_classes=22,
            num_workers=0,
            f1_threshold=0.7,
            mlflow_enabled=False,
        )

        assert submitted == SubmittedPipeline(
            name="pipeline-job",
            experiment_name="exp",
            studio_url="https://job",
        )
        assert [command["name"] for command in commands] == ["preprocess", "train"]
        assert pipeline_options == {"display_name": "Display", "experiment_name": "exp"}
        assert submitted_jobs == [
            {
                "pipeline_inputs": {
                    "raw_data": {
                        "input": {
                            "type": pipeline_submitter.AssetTypes.URI_FOLDER,
                            "path": "azureml:default-data:1",
                            "mode": pipeline_submitter.InputOutputModes.RO_MOUNT,
                        }
                    }
                },
                "tags": {
                    "project": "sign-language",
                    "purpose": "retraining",
                    "raw_data_asset": "azureml:default-data:1",
                    "raw_data_version": "7",
                },
            }
        ]

        preprocess_kwargs = commands[0]
        train_kwargs = commands[1]
        assert "--augmented-asset-name augmented" in str(preprocess_kwargs["command"])
        assert "--ngt-raw-version 7" in str(preprocess_kwargs["command"])
        assert train_kwargs["environment_variables"] == {
            "MLFLOW_ENABLED": "false",
            "MLFLOW_EXPERIMENT_NAME": "exp",
            "MLFLOW_AUTOLOG": "true",
            "MLFLOW_LOG_ARTIFACTS": "true",
        }
        assert command_calls[-1] == {
            "data": "augmented-output",
            "val_data": "validation-output",
        }
