"""Model loading and initialisation.

Loads EfficientNet and Landmark MLP checkpoints from disk, selects the
compute device, and initialises the MediaPipe hand landmarker.

The EfficientNet checkpoint source is selected by ``settings.deploy_target``:

- ``"local"``: read from ``settings.model_path`` on disk.
- ``"onprem"``: fetch the version aliased ``@champion`` from the self-hosted
  MLflow model registry at ``settings.mlflow_tracking_uri``.
- ``"azure"``: fetch the latest promoted version from the Azure ML model
  registry under ``settings.azure_subscription_id`` /
  ``settings.azure_resource_group`` / ``settings.azure_workspace``.

Both remote paths cache downloaded checkpoints under
``settings.model_cache_dir`` keyed by version, so repeated container
restarts do not re-download unchanged models.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Azure ML SDK (optional — only required when deploy_target == "azure")
try:
    from azure.ai.ml import MLClient
    from azure.identity import (
        AzureCliCredential,
        ChainedTokenCredential,
        ManagedIdentityCredential,
    )
except ImportError:
    MLClient = None  # type: ignore[assignment,misc]
    ManagedIdentityCredential = None  # type: ignore[assignment,misc]
    AzureCliCredential = None  # type: ignore[assignment,misc]
    ChainedTokenCredential = None  # type: ignore[assignment,misc]

# MLflow client (optional — only required when deploy_target == "onprem")
try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None  # type: ignore[assignment]
    MlflowClient = None  # type: ignore[assignment,misc]

from sign_language.core.settings import settings
from sign_language.models.architectures import (
    build_efficientnet,
    build_landmark_mlp,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Azure ML registry path
# =============================================================================


def download_latest_model_azure(
    model_name: str,
    download_dir: Path,
) -> Path:
    """Download the latest registered EfficientNet checkpoint from Azure ML.

    Prefers the version explicitly tagged ``promoted=true``, selecting the
    one with the highest ``f1_macro`` (with ``accuracy`` as tiebreaker).
    Falls back to ``label="latest"`` when no promoted version exists.

    Downloaded assets are cached under ``download_dir`` keyed by version
    number. On subsequent startups the cached file is reused if the version
    has not changed.

    :param model_name: Registry name, e.g. ``"ngt-sign-language"``.
    :param download_dir: Local directory to cache the downloaded checkpoint.
    :returns: :class:`~pathlib.Path` to the downloaded ``.pth`` file.
    :raises RuntimeError: If Azure ML dependencies are not installed, or if
        any required Azure settings (subscription ID, resource group,
        workspace) are missing.
    :raises FileNotFoundError: If no ``.pth`` file is found in the download
        directory after the download completes.
    """
    logger.info("Fetching promoted '%s' from Azure ML registry …", model_name)

    if (
        MLClient is None
        or ManagedIdentityCredential is None
        or AzureCliCredential is None
        or ChainedTokenCredential is None
    ):
        raise RuntimeError(
            "Azure ML registry loading requires azure-ai-ml and azure-identity. "
            "Install Azure dependencies or set DEPLOY_TARGET=local."
        )

    missing = [
        name
        for name, value in {
            "AZURE_SUBSCRIPTION_ID": settings.azure_subscription_id,
            "AZURE_RESOURCE_GROUP": settings.azure_resource_group,
            "AZURE_WORKSPACE": settings.azure_workspace,
        }.items()
        if not str(value or "").strip()
    ]
    if missing:
        raise RuntimeError(
            "DEPLOY_TARGET=azure but required settings are missing: "
            + ", ".join(missing)
        )

    credential = ChainedTokenCredential(
        ManagedIdentityCredential(),
        AzureCliCredential(),
    )

    ml_client = MLClient(
        credential=credential,
        subscription_id=settings.azure_subscription_id,
        resource_group_name=settings.azure_resource_group,
        workspace_name=settings.azure_workspace,
    )

    all_versions = list(ml_client.models.list(name=model_name))
    promoted = [v for v in all_versions if v.tags.get("promoted") == "true"]

    if promoted:
        model_info = max(
            promoted,
            key=lambda v: (
                float(v.tags.get("f1_macro", 0)),
                float(v.tags.get("accuracy", 0)),
            ),
        )
        logger.info(
            "Using promoted version %s (f1_macro=%s, accuracy=%s)",
            model_info.version,
            model_info.tags.get("f1_macro"),
            model_info.tags.get("accuracy"),
        )
    else:
        logger.warning(
            "No promoted version found for '%s' — falling back to label='latest'.",
            model_name,
        )
        model_info = ml_client.models.get(name=model_name, label="latest")
        logger.info("Fallback to latest version: %s", model_info.version)

    version_dir = download_dir / "azure" / f"v{model_info.version}"

    cached = list(version_dir.rglob("*.pth"))
    if cached:
        logger.info("Using cached model at %s", cached[0])
        return cached[0]

    version_dir.mkdir(parents=True, exist_ok=True)
    ml_client.models.download(
        name=model_name,
        version=str(model_info.version),
        download_path=str(version_dir),
    )

    pth_files = list(version_dir.rglob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(
            f"No .pth file found after downloading '{model_name}' "
            f"v{model_info.version} to {version_dir}"
        )

    logger.info("Downloaded model to %s", pth_files[0])
    return pth_files[0]


# =============================================================================
# Self-hosted MLflow registry path
# =============================================================================


def download_latest_model_mlflow(
    model_name: str,
    download_dir: Path,
    alias: str = "champion",
) -> Path:
    """Download the EfficientNet checkpoint aliased ``@<alias>`` from MLflow.

    Resolves ``models:/<model_name>@<alias>`` against the MLflow registry
    configured by ``settings.mlflow_tracking_uri``. Downloads the artifact
    directory to a version-keyed local cache; on subsequent startups the
    cached ``.pth`` file is reused if the aliased version has not changed.

    The MLflow server is expected to run with ``--serve-artifacts`` (the
    BUaS on-prem reference configuration), so the client only needs the
    tracking URI — no MinIO / S3 credentials.

    :param model_name: Registered model name, e.g. ``"ngt-sign-language"``.
    :param download_dir: Local directory to cache the downloaded checkpoint.
    :param alias: Registry alias to resolve. Defaults to ``"champion"``.
    :returns: :class:`~pathlib.Path` to the downloaded ``.pth`` file.
    :raises RuntimeError: If MLflow is not installed, or if
        ``MLFLOW_TRACKING_URI`` is not configured.
    :raises FileNotFoundError: If no ``.pth`` file is present in the
        downloaded artifact directory.
    """
    logger.info(
        "Fetching '%s@%s' from MLflow registry at %s …",
        model_name,
        alias,
        settings.mlflow_tracking_uri,
    )

    if mlflow is None or MlflowClient is None:
        raise RuntimeError(
            "MLflow registry loading requires the mlflow package. "
            "Install MLflow or set DEPLOY_TARGET=local."
        )

    tracking_uri = (settings.mlflow_tracking_uri or "").strip()
    if not tracking_uri:
        raise RuntimeError(
            "DEPLOY_TARGET=onprem but MLFLOW_TRACKING_URI is not configured."
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolve the alias to a concrete version for cache keying.
    mv = client.get_model_version_by_alias(name=model_name, alias=alias)
    version = mv.version
    logger.info("Alias '%s' resolves to %s v%s", alias, model_name, version)

    version_dir = download_dir / "mlflow" / f"v{version}"

    cached = list(version_dir.rglob("*.pth"))
    if cached:
        logger.info("Using cached model at %s", cached[0])
        return cached[0]

    version_dir.mkdir(parents=True, exist_ok=True)

    # Download the entire artifact directory for this version. With
    # --serve-artifacts the tracking server proxies the bytes from MinIO/S3,
    # so no object-store credentials are needed on this client.
    artifact_uri = f"models:/{model_name}@{alias}"
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri,
        dst_path=str(version_dir),
    )
    logger.info("MLflow artifacts downloaded to %s", local_path)

    pth_files = list(version_dir.rglob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(
            f"No .pth file found after downloading '{model_name}@{alias}' "
            f"(version {version}) to {version_dir}"
        )

    logger.info("Resolved checkpoint: %s", pth_files[0])
    return pth_files[0]


# =============================================================================
# Dispatch
# =============================================================================


def resolve_efficientnet_path() -> Path:
    """Return the EfficientNet checkpoint path to load at startup.

    Dispatches on :attr:`settings.deploy_target`:

    - ``"local"``: returns ``settings.model_path`` unchanged.
    - ``"onprem"``: downloads the version aliased ``@champion`` from the
      self-hosted MLflow registry via :func:`download_latest_model_mlflow`.
    - ``"azure"``: downloads the latest promoted version from the Azure ML
      registry via :func:`download_latest_model_azure`.

    :returns: :class:`~pathlib.Path` to the EfficientNet ``.pth`` checkpoint.
    :raises RuntimeError: If the selected target's required dependencies or
        settings are missing.
    :raises FileNotFoundError: If a remote download completes but no
        ``.pth`` file is found in the cache directory.
    :raises ValueError: If ``settings.deploy_target`` is set to an unknown
        value (should be unreachable due to the ``Literal`` constraint).
    """
    target = settings.deploy_target
    logger.info("Resolving EfficientNet checkpoint for DEPLOY_TARGET=%s", target)

    match target:
        case "azure":
            return download_latest_model_azure(
                model_name=settings.model_registry_name,
                download_dir=settings.model_cache_dir,
            )
        case "onprem":
            return download_latest_model_mlflow(
                model_name=settings.model_registry_name,
                download_dir=settings.model_cache_dir,
            )
        case "local":
            logger.info("Local mode — using on-disk path: %s", settings.model_path)
            return settings.model_path
        case _:  # pragma: no cover — guarded by Literal
            raise ValueError(f"Unknown DEPLOY_TARGET: {target!r}")


def resolve_landmark_path() -> Optional[Path]:
    """Return the Landmark MLP checkpoint path to load at startup, if available.

    Dispatches on :attr:`settings.deploy_target`, mirroring
    :func:`resolve_efficientnet_path`:

    - ``"local"``: returns ``settings.lm_model_path`` unchanged. If the file
      doesn't exist, ``load_landmark_mlp`` will skip it gracefully — same
      behaviour as before.
    - ``"onprem"`` / ``"azure"``: attempts to download the version aliased
      ``@champion`` from the corresponding registry. Returns ``None`` if the
      model isn't registered, the registry is unreachable, or the download
      fails — the landmark MLP is a fallback, not a hard requirement, so we
      degrade gracefully rather than blocking app startup.

    :returns: :class:`~pathlib.Path` to the Landmark MLP ``.pth`` file, or
        ``None`` if no checkpoint could be resolved.
    """
    target = settings.deploy_target
    logger.info("Resolving Landmark MLP checkpoint for DEPLOY_TARGET=%s", target)

    match target:
        case "azure":
            try:
                return download_latest_model_azure(
                    model_name=settings.lm_model_registry_name,
                    download_dir=settings.model_cache_dir,
                )
            except Exception as exc:
                logger.warning(
                    "Landmark MLP unavailable from Azure ML registry "
                    "('%s'): %s — continuing without fallback model.",
                    settings.lm_model_registry_name,
                    exc,
                )
                return None
        case "onprem":
            try:
                return download_latest_model_mlflow(
                    model_name=settings.lm_model_registry_name,
                    download_dir=settings.model_cache_dir,
                )
            except Exception as exc:
                logger.warning(
                    "Landmark MLP unavailable from MLflow registry "
                    "('%s'): %s — continuing without fallback model.",
                    settings.lm_model_registry_name,
                    exc,
                )
                return None
        case "local":
            logger.info(
                "Local mode — using on-disk landmark path: %s",
                settings.lm_model_path,
            )
            return settings.lm_model_path
        case _:  # pragma: no cover — guarded by Literal
            raise ValueError(f"Unknown DEPLOY_TARGET: {target!r}")


@dataclass
class LoadedModels:
    """Container holding all loaded models and associated metadata.

    Populated by :func:`load_all` at application startup and stored on
    ``app.state.app_state`` for access during inference.

    :param device: The torch device all models are loaded onto.
    :param model: The loaded EfficientNet-B0 classifier in eval mode.
    :param class_names: Ordered list of class label strings matching the
        EfficientNet output head.
    :param landmark_model: The loaded Landmark MLP in eval mode, or ``None``
        if the checkpoint was not found.
    :param lm_class_names: Ordered list of class label strings matching the
        MLP output head. Empty if the landmark model is not loaded.
    :param hands_detector: The initialised MediaPipe ``HandLandmarker``
        instance, or ``None`` if the task file was not found or MediaPipe
        is unavailable.
    """

    device: torch.device
    model: nn.Module
    class_names: list[str]
    landmark_model: Optional[nn.Module] = None
    lm_class_names: list[str] = field(default_factory=list)
    hands_detector: Optional[object] = None


def load_efficientnet(
    model_path: Path, device: torch.device
) -> tuple[nn.Module, list[str], dict]:
    """Load an EfficientNet-B0 checkpoint from disk.

    :param model_path: Path to the ``.pth`` checkpoint file containing
        ``model_state``, ``class_names``, ``val_acc``, and ``epoch`` keys.
    :param device: Torch device to map the weights onto.
    :returns: A three-tuple of ``(model, class_names, checkpoint_dict)``
        where ``model`` is in eval mode on ``device``.
    :raises FileNotFoundError: If ``model_path`` does not exist on disk.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"EfficientNet checkpoint not found: {model_path}")

    logger.info("Loading EfficientNet from %s …", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    class_names: list[str] = ckpt["class_names"]

    model = build_efficientnet(len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logger.info(
        "EfficientNet loaded — %d classes, val_acc=%.4f, epoch=%d",
        len(class_names),
        ckpt.get("val_acc", 0.0),
        ckpt.get("epoch", -1),
    )
    return model, class_names, ckpt


def load_landmark_mlp(
    model_path: Path, device: torch.device
) -> tuple[Optional[nn.Module], list[str]]:
    """Load the Landmark MLP checkpoint from disk.

    Returns ``(None, [])`` gracefully when the checkpoint file is missing,
    allowing the application to start without the landmark fallback model.

    :param model_path: Path to the ``.pth`` checkpoint file containing
        ``model_state``, ``class_names``, ``input_dim``, and ``val_acc`` keys.
    :param device: Torch device to map the weights onto.
    :returns: A two-tuple of ``(model, class_names)`` where ``model`` is in
        eval mode on ``device``, or ``(None, [])`` if the checkpoint is missing.
    """
    if not model_path.exists():
        logger.warning("Landmark MLP checkpoint not found: %s — skipping", model_path)
        return None, []

    logger.info("Loading Landmark MLP from %s …", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    lm_class_names: list[str] = ckpt["class_names"]
    input_dim: int = ckpt["input_dim"]

    model = build_landmark_mlp(input_dim, len(lm_class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    logger.info(
        "Landmark MLP loaded — %d classes, val_acc=%.4f",
        len(lm_class_names),
        ckpt.get("val_acc", 0.0),
    )
    return model, lm_class_names


def init_hand_detector(landmarker_path: Path) -> object | None:
    """Initialise the MediaPipe hand landmarker from a task file.

    Returns ``None`` gracefully if the task file is missing or if MediaPipe
    is unavailable, allowing the application to start without hand detection.

    :param landmarker_path: Path to the ``hand_landmarker.task`` file.
    :returns: A MediaPipe ``HandLandmarker`` instance configured for up to
        two hands, or ``None`` if the task file is missing or an error occurs.
    """
    if not landmarker_path.exists():
        logger.warning("Hand landmarker task file not found: %s", landmarker_path)
        return None

    try:
        from mediapipe.tasks import (  # type: ignore[import-untyped]
            python as mp_python,
        )
        from mediapipe.tasks.python import (  # type: ignore[import-untyped]
            vision,
        )

        base_options = mp_python.BaseOptions(model_asset_path=str(landmarker_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        logger.info("MediaPipe hand landmarker initialised")
        return detector  # type: ignore[no-any-return]
    except Exception as exc:
        logger.warning(
            "MediaPipe unavailable: %s — continuing without hand detection",
            exc,
        )
        return None


def load_all(
    model_path: Optional[Path] = None,
    lm_model_path: Optional[Path] = None,
    landmarker_path: Optional[Path] = None,
) -> LoadedModels:
    """Load every model and return them in a single :class:`LoadedModels` container.

    When ``model_path`` is ``None``, :func:`resolve_efficientnet_path` is
    called, which dispatches on ``settings.deploy_target`` to either return
    the local on-disk path, download from MLflow (``onprem``), or download
    from the Azure ML registry (``azure``).

    :param model_path: Path to the EfficientNet ``.pth`` checkpoint, or
        ``None`` to resolve automatically via :func:`resolve_efficientnet_path`.
    :param lm_model_path: Path to the Landmark MLP ``.pth`` checkpoint, or
        ``None`` to resolve automatically via :func:`resolve_landmark_path`.
    :param landmarker_path: Path to the MediaPipe ``hand_landmarker.task``
        file. Defaults to ``settings.hand_landmarker_path``.
    :returns: A :class:`LoadedModels` dataclass with all models loaded and
        ready for inference.
    """
    device = settings.get_device()

    resolved_model_path = model_path or resolve_efficientnet_path()
    resolved_lm_model_path = lm_model_path or resolve_landmark_path()
    resolved_landmarker_path = landmarker_path or settings.hand_landmarker_path
    model, class_names, _ = load_efficientnet(resolved_model_path, device)
    # resolve_landmark_path() may return None when running in onprem/azure
    # mode and no landmark MLP is registered yet. load_landmark_mlp expects
    # a Path, so guard with a sentinel that triggers its graceful skip path.
    landmark_model, lm_class_names = (
        load_landmark_mlp(resolved_lm_model_path, device)
        if resolved_lm_model_path is not None
        else (None, [])
    )
    hands_detector = init_hand_detector(resolved_landmarker_path)

    return LoadedModels(
        device=device,
        model=model,
        class_names=class_names,
        landmark_model=landmark_model,
        lm_class_names=lm_class_names,
        hands_detector=hands_detector,
    )
