"""Tests for the model loader module."""

from pathlib import Path
from contextlib import ExitStack, contextmanager
from unittest.mock import patch, MagicMock

import torch
import pytest

from sign_language.models.loader import (
    LoadedModels,
    load_efficientnet,
    load_landmark_mlp,
    init_hand_detector,
    load_all,
)


class TestLoadEfficientnet:
    """Tests for load_efficientnet."""

    def test_missing_file_raises(self, tmp_path):
        """Should raise FileNotFoundError if checkpoint doesn't exist."""
        fake_path = tmp_path / "nonexistent.pth"
        with pytest.raises(FileNotFoundError, match="not found"):
            load_efficientnet(fake_path, torch.device("cpu"))

    def test_loads_valid_checkpoint(self, tmp_path):
        """Should load a model from a valid checkpoint file."""
        from sign_language.models.architectures import build_efficientnet

        num_classes = 5
        class_names = ["A", "B", "C", "D", "E"]
        model = build_efficientnet(num_classes)

        ckpt = {
            "class_names": class_names,
            "model_state": model.state_dict(),
            "val_acc": 0.95,
            "epoch": 10,
        }
        ckpt_path = tmp_path / "test_model.pth"
        torch.save(ckpt, ckpt_path)

        loaded_model, loaded_names, raw_ckpt = load_efficientnet(
            ckpt_path, torch.device("cpu")
        )

        assert loaded_names == class_names
        assert not loaded_model.training
        assert raw_ckpt["val_acc"] == 0.95


class TestLoadLandmarkMlp:
    """Tests for load_landmark_mlp."""

    def test_missing_file_returns_none(self, tmp_path):
        """Should return (None, []) if checkpoint doesn't exist."""
        fake_path = tmp_path / "nonexistent.pth"
        model, names = load_landmark_mlp(fake_path, torch.device("cpu"))

        assert model is None
        assert names == []

    def test_loads_valid_checkpoint(self, tmp_path):
        """Should load a model from a valid checkpoint file."""
        from sign_language.models.architectures import build_landmark_mlp

        input_dim = 94
        num_classes = 5
        class_names = ["A", "B", "C", "D", "E"]
        model = build_landmark_mlp(input_dim, num_classes)

        ckpt = {
            "class_names": class_names,
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "val_acc": 0.88,
        }
        ckpt_path = tmp_path / "test_lm.pth"
        torch.save(ckpt, ckpt_path)

        loaded_model, loaded_names = load_landmark_mlp(ckpt_path, torch.device("cpu"))

        assert loaded_names == class_names
        assert loaded_model is not None
        assert not loaded_model.training


class TestInitHandDetector:
    """Tests for init_hand_detector."""

    def test_missing_file_returns_none(self, tmp_path):
        """Should return None if task file doesn't exist."""
        fake_path = tmp_path / "nonexistent.task"
        result = init_hand_detector(fake_path)
        assert result is None

    def test_mediapipe_import_failure_returns_none(self, tmp_path):
        """Should return None if MediaPipe fails to initialise."""
        task_file = tmp_path / "hand_landmarker.task"
        task_file.write_bytes(b"fake")

        with patch.dict(
            "sys.modules", {"mediapipe.tasks": None, "mediapipe.tasks.python": None}
        ):
            result = init_hand_detector(task_file)

        assert result is None


class TestLoadAll:
    """Tests for load_all."""

    def test_returns_loaded_models_dataclass(self, tmp_path):
        """Should return a LoadedModels instance."""
        from sign_language.models.architectures import (
            build_efficientnet,
            build_landmark_mlp,
        )

        # Create fake EfficientNet checkpoint
        class_names = ["A", "B", "C"]
        model = build_efficientnet(len(class_names))
        ckpt = {
            "class_names": class_names,
            "model_state": model.state_dict(),
            "val_acc": 0.90,
            "epoch": 5,
        }
        eff_path = tmp_path / "eff.pth"
        torch.save(ckpt, eff_path)

        # Create fake Landmark MLP checkpoint
        input_dim = 94
        lm_model = build_landmark_mlp(input_dim, len(class_names))
        lm_ckpt = {
            "class_names": class_names,
            "model_state": lm_model.state_dict(),
            "input_dim": input_dim,
            "val_acc": 0.85,
        }
        lm_path = tmp_path / "lm.pth"
        torch.save(lm_ckpt, lm_path)

        # Fake landmarker path (won't load but won't crash)
        fake_task = tmp_path / "hand.task"

        loaded = load_all(
            model_path=eff_path,
            lm_model_path=lm_path,
            landmarker_path=fake_task,
        )

        assert isinstance(loaded, LoadedModels)
        assert loaded.class_names == class_names
        assert loaded.model is not None
        assert loaded.hands_detector is None


class TestResolveEfficientnetPath:
    """Tests for resolve_efficientnet_path dispatch on deploy_target."""

    def test_returns_local_path_when_target_is_local(self, tmp_path):
        """deploy_target='local' returns settings.model_path unchanged."""
        from sign_language.models.loader import resolve_efficientnet_path
        from sign_language.core.settings import settings

        with patch.object(settings, "deploy_target", "local"):
            result = resolve_efficientnet_path()

        assert result == settings.model_path

    def test_calls_azure_download_when_target_is_azure(self, tmp_path):
        """deploy_target='azure' calls download_latest_model_azure."""
        from sign_language.models.loader import resolve_efficientnet_path
        from sign_language.core.settings import settings

        expected = tmp_path / "model.pth"

        with (
            patch.object(settings, "deploy_target", "azure"),
            patch.object(settings, "model_registry_name", "ngt-sign-language"),
            patch.object(settings, "model_cache_dir", tmp_path),
            patch(
                "sign_language.models.loader.download_latest_model_azure",
                return_value=expected,
            ) as mock_dl,
        ):
            result = resolve_efficientnet_path()

        mock_dl.assert_called_once_with(
            model_name="ngt-sign-language",
            download_dir=tmp_path,
        )
        assert result == expected

    def test_calls_mlflow_download_when_target_is_onprem(self, tmp_path):
        """deploy_target='onprem' calls download_latest_model_mlflow."""
        from sign_language.models.loader import resolve_efficientnet_path
        from sign_language.core.settings import settings

        expected = tmp_path / "model.pth"

        with (
            patch.object(settings, "deploy_target", "onprem"),
            patch.object(settings, "model_registry_name", "ngt-sign-language"),
            patch.object(settings, "model_cache_dir", tmp_path),
            patch(
                "sign_language.models.loader.download_latest_model_mlflow",
                return_value=expected,
            ) as mock_dl,
        ):
            result = resolve_efficientnet_path()

        mock_dl.assert_called_once_with(
            model_name="ngt-sign-language",
            download_dir=tmp_path,
        )
        assert result == expected


class TestDownloadLatestModelAzure:
    """Tests for download_latest_model_azure."""

    @contextmanager
    def _patch_azure_settings(self):
        """Patch required Azure workspace settings for registry-download tests."""
        from sign_language.core.settings import settings

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(settings, "azure_subscription_id", "sub-123")
            )
            stack.enter_context(
                patch.object(settings, "azure_resource_group", "rg-test")
            )
            stack.enter_context(patch.object(settings, "azure_workspace", "ws-test"))
            yield

    def _make_mock_client(self, version: str, pth_file: Path) -> MagicMock:
        """Return a mock MLClient whose models.get returns *version*."""
        mock_model_info = MagicMock()
        mock_model_info.version = version
        mock_model_info.tags = {}

        mock_models = MagicMock()
        mock_models.get.return_value = mock_model_info
        mock_models.list.return_value = []
        mock_models.download = MagicMock()

        mock_client = MagicMock()
        mock_client.models = mock_models
        return mock_client

    def test_uses_cache_when_version_dir_exists(self, tmp_path):
        """Should skip download and return cached .pth when version already downloaded."""
        from sign_language.models.loader import download_latest_model_azure

        # Cache key is now nested under "azure/" subdir per the new layout.
        version_dir = tmp_path / "azure" / "v3"
        version_dir.mkdir(parents=True)
        cached_pth = version_dir / "model.pth"
        cached_pth.write_bytes(b"fake")

        mock_client = self._make_mock_client("3", cached_pth)

        with (
            self._patch_azure_settings(),
            patch("sign_language.models.loader.MLClient", return_value=mock_client),
            patch("sign_language.models.loader.ManagedIdentityCredential"),
        ):
            result = download_latest_model_azure("ngt-sign-language", tmp_path)

        mock_client.models.download.assert_not_called()
        assert result == cached_pth

    def test_downloads_when_no_cache(self, tmp_path):
        """Should call models.download when no cached version exists."""
        from sign_language.models.loader import download_latest_model_azure

        def fake_download(name, version, download_path):
            pth = Path(download_path) / "model.pth"
            pth.write_bytes(b"fake")

        mock_client = self._make_mock_client(
            "4", tmp_path / "azure" / "v4" / "model.pth"
        )
        mock_client.models.download.side_effect = fake_download

        with (
            self._patch_azure_settings(),
            patch("sign_language.models.loader.MLClient", return_value=mock_client),
            patch("sign_language.models.loader.ManagedIdentityCredential"),
        ):
            result = download_latest_model_azure("ngt-sign-language", tmp_path)

        mock_client.models.download.assert_called_once()
        assert result.suffix == ".pth"
        assert result.exists()

    def test_raises_if_no_pth_after_download(self, tmp_path):
        """Should raise FileNotFoundError if download produces no .pth file."""
        from sign_language.models.loader import download_latest_model_azure

        mock_client = self._make_mock_client(
            "5", tmp_path / "azure" / "v5" / "model.pth"
        )
        mock_client.models.download = MagicMock()  # no side effect

        with (
            self._patch_azure_settings(),
            patch("sign_language.models.loader.MLClient", return_value=mock_client),
            patch("sign_language.models.loader.ManagedIdentityCredential"),
        ):
            with pytest.raises(FileNotFoundError, match="No .pth file found"):
                download_latest_model_azure("ngt-sign-language", tmp_path)

    def test_picks_highest_scoring_promoted_version(self, tmp_path):
        """When multiple versions have promoted=true, pick the one with highest
        f1_macro (tiebreak on accuracy)."""
        from sign_language.models.loader import download_latest_model_azure

        def fake_download(name, version, download_path):
            pth = Path(download_path) / "model.pth"
            pth.write_bytes(b"fake")

        v_low = MagicMock()
        v_low.version = "1"
        v_low.tags = {"promoted": "true", "f1_macro": "0.70", "accuracy": "0.80"}

        v_high = MagicMock()
        v_high.version = "2"
        v_high.tags = {"promoted": "true", "f1_macro": "0.85", "accuracy": "0.82"}

        mock_models = MagicMock()
        mock_models.list.return_value = [v_low, v_high]
        # download should be called with version="2" (the higher f1_macro)
        mock_models.download.side_effect = fake_download

        mock_client = MagicMock()
        mock_client.models = mock_models

        with (
            self._patch_azure_settings(),
            patch("sign_language.models.loader.MLClient", return_value=mock_client),
            patch("sign_language.models.loader.ManagedIdentityCredential"),
        ):
            result = download_latest_model_azure("ngt-sign-language", tmp_path)

        # download should be called with the higher-scoring version
        mock_models.download.assert_called_once()
        call_kwargs = mock_models.download.call_args.kwargs
        assert call_kwargs["version"] == "2"
        # models.get should NOT be called — we found a promoted version, no fallback
        mock_models.get.assert_not_called()
        assert result.exists()

    def test_falls_back_to_latest_when_no_promoted_version(self, tmp_path):
        """When no version is tagged promoted=true, fall back to label='latest'."""
        from sign_language.models.loader import download_latest_model_azure

        def fake_download(name, version, download_path):
            pth = Path(download_path) / "model.pth"
            pth.write_bytes(b"fake")

        # Mock client: list returns versions with no promoted=true tag,
        # so the function should fall back to models.get(label="latest").
        mock_model_info = MagicMock()
        mock_model_info.version = "9"
        mock_model_info.tags = {}  # no "promoted" tag

        # list() returns versions, but none are promoted.
        unpromoted_version = MagicMock()
        unpromoted_version.tags = {"some_other_tag": "value"}

        mock_models = MagicMock()
        mock_models.list.return_value = [unpromoted_version]
        mock_models.get.return_value = mock_model_info
        mock_models.download.side_effect = fake_download

        mock_client = MagicMock()
        mock_client.models = mock_models

        with (
            self._patch_azure_settings(),
            patch("sign_language.models.loader.MLClient", return_value=mock_client),
            patch("sign_language.models.loader.ManagedIdentityCredential"),
        ):
            result = download_latest_model_azure("ngt-sign-language", tmp_path)

        mock_models.get.assert_called_once_with(
            name="ngt-sign-language", label="latest"
        )
        assert result.exists()

    def test_uses_chained_credential_for_managed_identity_and_cli(self, tmp_path):
        """Should let Azure Identity fall back at token-request time."""
        from sign_language.models.loader import download_latest_model_azure

        def fake_download(name, version, download_path):
            pth = Path(download_path) / "model.pth"
            pth.write_bytes(b"fake")

        mock_client = self._make_mock_client(
            "6", tmp_path / "azure" / "v6" / "model.pth"
        )
        mock_client.models.download.side_effect = fake_download

        with (
            self._patch_azure_settings(),
            patch("sign_language.models.loader.ManagedIdentityCredential") as mock_mi,
            patch("sign_language.models.loader.AzureCliCredential") as mock_cli,
            patch("sign_language.models.loader.ChainedTokenCredential") as mock_chain,
            patch("sign_language.models.loader.MLClient", return_value=mock_client),
        ):
            download_latest_model_azure("ngt-sign-language", tmp_path)

        mock_mi.assert_called_once()
        mock_cli.assert_called_once()
        mock_chain.assert_called_once_with(mock_mi.return_value, mock_cli.return_value)

    def test_missing_azure_settings_raise_descriptive_error(self, tmp_path):
        """Should fail clearly when Azure registry settings are incomplete."""
        from sign_language.core.settings import settings
        from sign_language.models.loader import download_latest_model_azure

        with (
            patch.object(settings, "azure_subscription_id", None),
            patch.object(settings, "azure_resource_group", None),
            patch.object(settings, "azure_workspace", None),
            pytest.raises(RuntimeError, match="required settings are missing"),
        ):
            download_latest_model_azure("ngt-sign-language", tmp_path)


class TestDownloadLatestModelMlflow:
    """Tests for download_latest_model_mlflow."""

    def _make_mock_mlflow(self, version: str):
        """Return mock mlflow module + MlflowClient class."""
        mock_mv = MagicMock()
        mock_mv.version = version

        mock_client_instance = MagicMock()
        mock_client_instance.get_model_version_by_alias.return_value = mock_mv

        mock_client_cls = MagicMock(return_value=mock_client_instance)
        mock_mlflow = MagicMock()
        mock_mlflow.set_tracking_uri = MagicMock()
        mock_mlflow.artifacts.download_artifacts = MagicMock()
        return mock_mlflow, mock_client_cls, mock_client_instance

    def test_uses_cache_when_version_dir_exists(self, tmp_path):
        """Should skip download_artifacts and return cached .pth when version present."""
        from sign_language.core.settings import settings
        from sign_language.models.loader import download_latest_model_mlflow

        version_dir = tmp_path / "mlflow" / "v7"
        version_dir.mkdir(parents=True)
        cached_pth = version_dir / "model.pth"
        cached_pth.write_bytes(b"fake")

        mock_mlflow, mock_client_cls, mock_client_instance = self._make_mock_mlflow("7")

        with (
            patch.object(settings, "mlflow_tracking_uri", "http://mlflow:5000"),
            patch("sign_language.models.loader.mlflow", mock_mlflow),
            patch("sign_language.models.loader.MlflowClient", mock_client_cls),
        ):
            result = download_latest_model_mlflow("ngt-sign-language", tmp_path)

        mock_mlflow.artifacts.download_artifacts.assert_not_called()
        assert result == cached_pth

    def test_downloads_when_no_cache(self, tmp_path):
        """Should call mlflow.artifacts.download_artifacts when no cached version."""
        from sign_language.core.settings import settings
        from sign_language.models.loader import download_latest_model_mlflow

        mock_mlflow, mock_client_cls, _ = self._make_mock_mlflow("8")

        def fake_download(artifact_uri, dst_path):
            # Mimic mlflow placing files into dst_path
            pth = Path(dst_path) / "data" / "model.pth"
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_bytes(b"fake")
            return str(pth.parent)

        mock_mlflow.artifacts.download_artifacts.side_effect = fake_download

        with (
            patch.object(settings, "mlflow_tracking_uri", "http://mlflow:5000"),
            patch("sign_language.models.loader.mlflow", mock_mlflow),
            patch("sign_language.models.loader.MlflowClient", mock_client_cls),
        ):
            result = download_latest_model_mlflow("ngt-sign-language", tmp_path)

        mock_mlflow.artifacts.download_artifacts.assert_called_once()
        assert result.suffix == ".pth"
        assert result.exists()

    def test_raises_if_no_pth_after_download(self, tmp_path):
        """FileNotFoundError if download_artifacts leaves no .pth in the cache dir."""
        from sign_language.core.settings import settings
        from sign_language.models.loader import download_latest_model_mlflow

        mock_mlflow, mock_client_cls, _ = self._make_mock_mlflow("9")
        # download_artifacts returns but writes nothing
        mock_mlflow.artifacts.download_artifacts.return_value = str(
            tmp_path / "mlflow" / "v9"
        )

        with (
            patch.object(settings, "mlflow_tracking_uri", "http://mlflow:5000"),
            patch("sign_language.models.loader.mlflow", mock_mlflow),
            patch("sign_language.models.loader.MlflowClient", mock_client_cls),
        ):
            with pytest.raises(FileNotFoundError, match="No .pth file found"):
                download_latest_model_mlflow("ngt-sign-language", tmp_path)

    def test_missing_tracking_uri_raises(self, tmp_path):
        """Should fail clearly when MLFLOW_TRACKING_URI is empty."""
        from sign_language.core.settings import settings
        from sign_language.models.loader import download_latest_model_mlflow

        # Even with mlflow installed, an empty tracking URI is a configuration error
        mock_mlflow, mock_client_cls, _ = self._make_mock_mlflow("1")

        with (
            patch.object(settings, "mlflow_tracking_uri", ""),
            patch("sign_language.models.loader.mlflow", mock_mlflow),
            patch("sign_language.models.loader.MlflowClient", mock_client_cls),
            pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI is not configured"),
        ):
            download_latest_model_mlflow("ngt-sign-language", tmp_path)

    def test_resolves_custom_alias(self, tmp_path):
        """Should pass `alias` through to get_model_version_by_alias and the URI."""
        from sign_language.core.settings import settings
        from sign_language.models.loader import download_latest_model_mlflow

        mock_mlflow, mock_client_cls, mock_client_instance = self._make_mock_mlflow("2")

        def fake_download(artifact_uri, dst_path):
            pth = Path(dst_path) / "model.pth"
            pth.write_bytes(b"fake")
            return str(dst_path)

        mock_mlflow.artifacts.download_artifacts.side_effect = fake_download

        with (
            patch.object(settings, "mlflow_tracking_uri", "http://mlflow:5000"),
            patch("sign_language.models.loader.mlflow", mock_mlflow),
            patch("sign_language.models.loader.MlflowClient", mock_client_cls),
        ):
            download_latest_model_mlflow(
                "ngt-sign-language", tmp_path, alias="candidate"
            )

        mock_client_instance.get_model_version_by_alias.assert_called_once_with(
            name="ngt-sign-language", alias="candidate"
        )
        artifact_uri = mock_mlflow.artifacts.download_artifacts.call_args.kwargs[
            "artifact_uri"
        ]
        assert artifact_uri == "models:/ngt-sign-language@candidate"


class TestResolveLandmarkPath:
    """Tests for resolve_landmark_path dispatch on deploy_target."""

    def test_returns_local_path_when_target_is_local(self, tmp_path):
        """deploy_target='local' returns settings.lm_model_path unchanged."""
        from sign_language.models.loader import resolve_landmark_path
        from sign_language.core.settings import settings

        with patch.object(settings, "deploy_target", "local"):
            result = resolve_landmark_path()

        assert result == settings.lm_model_path

    def test_calls_azure_download_when_target_is_azure(self, tmp_path):
        """deploy_target='azure' calls download_latest_model_azure with the
        landmark registry name."""
        from sign_language.models.loader import resolve_landmark_path
        from sign_language.core.settings import settings

        expected = tmp_path / "mlp.pth"

        with (
            patch.object(settings, "deploy_target", "azure"),
            patch.object(settings, "lm_model_registry_name", "ngt-landmark-mlp"),
            patch.object(settings, "model_cache_dir", tmp_path),
            patch(
                "sign_language.models.loader.download_latest_model_azure",
                return_value=expected,
            ) as mock_dl,
        ):
            result = resolve_landmark_path()

        mock_dl.assert_called_once_with(
            model_name="ngt-landmark-mlp",
            download_dir=tmp_path,
        )
        assert result == expected

    def test_calls_mlflow_download_when_target_is_onprem(self, tmp_path):
        """deploy_target='onprem' calls download_latest_model_mlflow with the
        landmark registry name."""
        from sign_language.models.loader import resolve_landmark_path
        from sign_language.core.settings import settings

        expected = tmp_path / "mlp.pth"

        with (
            patch.object(settings, "deploy_target", "onprem"),
            patch.object(settings, "lm_model_registry_name", "ngt-landmark-mlp"),
            patch.object(settings, "model_cache_dir", tmp_path),
            patch(
                "sign_language.models.loader.download_latest_model_mlflow",
                return_value=expected,
            ) as mock_dl,
        ):
            result = resolve_landmark_path()

        mock_dl.assert_called_once_with(
            model_name="ngt-landmark-mlp",
            download_dir=tmp_path,
        )
        assert result == expected

    def test_returns_none_when_azure_download_fails(self, tmp_path):
        """Azure registry failures degrade gracefully — landmark MLP is a
        fallback model, not a hard dependency."""
        from sign_language.models.loader import resolve_landmark_path
        from sign_language.core.settings import settings

        with (
            patch.object(settings, "deploy_target", "azure"),
            patch.object(settings, "lm_model_registry_name", "ngt-landmark-mlp"),
            patch.object(settings, "model_cache_dir", tmp_path),
            patch(
                "sign_language.models.loader.download_latest_model_azure",
                side_effect=RuntimeError("model not registered"),
            ),
        ):
            result = resolve_landmark_path()

        assert result is None

    def test_returns_none_when_mlflow_download_fails(self, tmp_path):
        """MLflow registry failures degrade gracefully."""
        from sign_language.models.loader import resolve_landmark_path
        from sign_language.core.settings import settings

        with (
            patch.object(settings, "deploy_target", "onprem"),
            patch.object(settings, "lm_model_registry_name", "ngt-landmark-mlp"),
            patch.object(settings, "model_cache_dir", tmp_path),
            patch(
                "sign_language.models.loader.download_latest_model_mlflow",
                side_effect=RuntimeError("connection refused"),
            ),
        ):
            result = resolve_landmark_path()

        assert result is None
