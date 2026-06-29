"""Application settings loaded from environment variables and the ``.env`` file.

Provides a single :class:`Settings` instance (``settings``) used throughout
the application for configuration. All fields are overridable via environment
variables or the ``.env`` file at the repository root.

Torch is imported lazily via :class:`_LazyTorch` to avoid pulling it in on
import in environments where it is not needed (e.g. Windows CPU-only scripts).
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Literal, List, Optional, Dict, Any, Union

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _LazyTorch:
    """Proxy that imports :mod:`torch` only when a device helper first accesses it.

    Replaces the module-level ``torch`` name so that importing this settings
    module never triggers a torch import on its own.
    """

    def __getattr__(self, name: str) -> Any:
        import torch as torch_module

        globals()["torch"] = torch_module
        return getattr(torch_module, name)


torch: Any = _LazyTorch()

# ---------------------------------------------------------------------------
# Base directory — supports both normal run and bundled exe (PyInstaller)
# ---------------------------------------------------------------------------
BASE_DIR: Path = (
    Path(sys._MEIPASS)  # type: ignore[attr-defined]
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parent.parent.parent.parent
)

DOTENV: Path = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables and ``.env``.

    All fields map directly to environment variables (case-insensitive).
    Unknown variables in ``.env`` are silently ignored. Sensible defaults are
    provided for every field so the application starts without a ``.env`` file,
    but production deployments must override ``secret_key``, ``database_url``,
    and any Azure/MLflow fields they use.

    Sections:

    - **Application identity**: name, version, authors, environment status.
    - **Debug / logging**: debug flag, log level, log file path.
    - **CORS**: allowed origins for the FastAPI CORS middleware.
    - **Auth / database**: PostgreSQL URL, JWT secret, token lifetime.
    - **Training**: hyperparameters, file names, dataset paths.
    - **MLflow**: experiment tracking configuration.
    - **Azure ML**: workspace, compute, environment, and asset settings.
    - **Model paths**: local paths to model checkpoints and MediaPipe assets.
    - **Server**: Uvicorn host and port.
    - **Inference thresholds**: EfficientNet and landmark MLP confidence gates.
    - **Sequence builder**: timing parameters for letter commit and spacing.
    - **Prediction smoother**: majority-vote window and threshold settings.
    """

    model_config = SettingsConfigDict(
        env_file=DOTENV,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application identity
    # -------------------------------------------------------------------------
    app_name: str = Field(default="Sign Language", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    authors: List[str] = Field(
        default=[
            "Szymon Chirowski",
            "Maciej Czerniak",
            "Ana Farazică",
            "Raya Kichukova",
            "Kinga Marchlewska",
        ],
        description="List of authors",
    )
    authors_email: List[str] = Field(
        default=[
            "242621@buas.nl",
            "243552@buas.nl",
            "242845@buas.nl",
            "241329@buas.nl",
            "241290@buas.nl",
        ],
        description="Author emails",
    )
    status: Literal["DEVELOPMENT", "PRODUCTION"] = Field(
        default="DEVELOPMENT", description="Application environment"
    )

    # -------------------------------------------------------------------------
    # Deployment target — selects model resolution strategy at startup
    # -------------------------------------------------------------------------
    deploy_target: Literal["local", "onprem", "azure"] = Field(
        default="local",
        description=(
            "Deployment target. Selects how the EfficientNet checkpoint is "
            "resolved at startup: 'local' uses the on-disk model_path; "
            "'onprem' downloads from the self-hosted MLflow registry via "
            "MLFLOW_TRACKING_URI; 'azure' downloads from the Azure ML model "
            "registry. Set DEPLOY_TARGET=onprem|azure in deployed environments."
        ),
    )

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    debug: Optional[bool] = Field(default=None, description="Enable debug mode")

    # -------------------------------------------------------------------------
    # Logger
    # -------------------------------------------------------------------------
    log_level: Optional[str] = Field(default=None, description="Logging level")
    log_path: Optional[Path] = Field(
        default=Path("logs/app.log"),
        description="Path to log file; create parent directories during app startup when configuring logging",
    )

    # -------------------------------------------------------------------------
    # CORS
    # -------------------------------------------------------------------------
    cors_origins: Optional[Union[List[str], str]] = Field(
        default=None, description="Allowed CORS origins"
    )

    # -------------------------------------------------------------------------
    # Auth / Database
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://changeme:changeme@localhost:5432/signlang",
        description="Async PostgreSQL connection string for SQLAlchemy",
    )
    require_ssl: bool = Field(
        default=False, description="Require SSL to connect to DB."
    )
    secret_key: str = Field(
        default="change-me-in-production",
        description="JWT signing secret — override via SECRET_KEY env var",
    )
    jwt_lifetime_seconds: int = Field(
        default=3600,
        description="JWT token lifetime in seconds",
    )

    # Training defaults
    training_best_checkpoint_name: str = Field(
        default="model.pth",
        description="Filename for the best training checkpoint",
    )
    training_history_filename: str = Field(
        default="training_history.json",
        description="Filename for saved training history",
    )
    training_metrics_filename: str = Field(
        default="metrics.json",
        description="Filename for saved evaluation metrics",
    )
    training_report_filename: str = Field(
        default="classification_report.txt",
        description="Filename for saved classification report",
    )
    training_class_names_filename: str = Field(
        default="class_names.json",
        description="Filename for saved class names",
    )
    training_img_size: int = Field(
        default=224,
        description="Input image size used by training transforms",
    )
    training_batch_size: int = Field(default=16, description="Training batch size")
    training_learning_rate: float = Field(
        default=1e-4,
        description="Training optimizer learning rate",
    )
    training_epochs: int = Field(default=30, description="Maximum training epochs")
    training_patience: int = Field(default=7, description="Early stopping patience")
    training_val_split: float = Field(
        default=0.2,
        description="Validation split ratio",
    )
    training_target_accuracy: float = Field(
        default=0.85,
        description="Target validation accuracy",
    )
    training_seed: int = Field(default=42, description="Random seed for training")
    training_n_splits: int = Field(
        default=1,
        description="Number of stratified splits used by the training workflow",
    )
    training_num_workers: int = Field(
        default=4,
        description="DataLoader worker process count",
    )
    training_pin_memory: bool = Field(
        default=True,
        description="Whether DataLoader should pin memory",
    )
    training_eta_min: float = Field(
        default=1e-6,
        description="Minimum learning rate for the scheduler",
    )
    training_expected_num_classes: int = Field(
        default=22,
        description="Expected number of NGT classes in the training dataset",
    )
    training_local_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Local ImageFolder training dataset root",
    )

    # -------------------------------------------------------------------------
    # MLflow
    # -------------------------------------------------------------------------
    mlflow_enabled: bool = Field(
        default=False,
        description="Enable MLflow experiment tracking for training",
    )
    mlflow_tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking URI; leave empty for local ./mlruns tracking",
    )
    mlflow_experiment_name: str = Field(
        default="sign-language",
        description="MLflow experiment name",
    )
    mlflow_run_name: Optional[str] = Field(
        default=None,
        description="Optional MLflow run name",
    )
    mlflow_autolog: bool = Field(
        default=True,
        description="Enable MLflow PyTorch autologging before manual training starts",
    )
    mlflow_log_artifacts: bool = Field(
        default=True,
        description="Log training outputs and checkpoints as MLflow artifacts",
    )

    # -------------------------------------------------------------------------
    # Azure ML
    # -------------------------------------------------------------------------
    azure_subscription_id: Optional[str] = Field(
        default=None, description="Azure subscription ID"
    )
    azure_resource_group: Optional[str] = Field(
        default=None, description="Azure resource group name"
    )
    azure_workspace: Optional[str] = Field(
        default=None, description="Azure ML workspace name"
    )
    azure_compute_target: Optional[str] = Field(
        default=None, description="Target Azure ML compute cluster name"
    )
    azure_environment_name: Optional[str] = Field(
        default=None, description="Azure ML environment asset name"
    )
    azure_environment_version: Optional[str] = Field(
        default=None, description="Azure ML environment asset version"
    )
    azure_raw_data_asset_name: str = Field(
        default="ngt-raw",
        description="Azure ML raw ImageFolder data asset name",
    )
    azure_raw_data_asset_version: str = Field(
        default="1",
        description="Azure ML raw ImageFolder data asset version",
    )
    azure_pretrained_checkpoint_asset_name: Optional[str] = Field(
        default=None,
        description="Optional Azure ML data/model asset name for the pretrained checkpoint",
    )
    azure_pretrained_checkpoint_asset_version: Optional[str] = Field(
        default=None,
        description="Optional Azure ML asset version for the pretrained checkpoint",
    )
    azure_instance_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional Azure ML Kubernetes instance type. "
            "Examples: gpu, cpu-xl, cpu-large, cpu-med, cpu-small, defaultinstancetype."
        ),
    )
    azure_prefer_gpu: bool = Field(
        default=False,
        description=(
            "If true and AZURE_INSTANCE_TYPE is not set, prefer the 'gpu' instance type. "
            "Otherwise prefer the strongest CPU instance type."
        ),
    )
    model_registry_name: str = Field(
        default="ngt-sign-language", description="Name for the model registration"
    )
    lm_model_registry_name: str = Field(
        default="ngt-landmark-mlp",
        description=(
            "Registry name for the landmark MLP fallback model. Used when "
            "deploy_target is 'onprem' or 'azure'. Local mode reads "
            "lm_model_path from disk and ignores this field."
        ),
    )
    model_cache_dir: Path = Field(
        default_factory=lambda: BASE_DIR / "models" / "registry_cache",
        description=(
            "Local cache directory for models downloaded from a remote "
            "registry (Azure ML or self-hosted MLflow). Versioned subdirectories "
            "keyed by version number; on startup the cached file is reused if "
            "the latest version has not changed."
        ),
    )
    training_f1_threshold: float = Field(
        default=0.80, description="A threshold when the model will be registered"
    )

    # -------------------------------------------------------------------------
    # Model & asset paths
    # -------------------------------------------------------------------------
    img_size: int = Field(default=224, description="Input image size for the model")
    model_path: Path = Field(
        default_factory=lambda: BASE_DIR / "models" / "best_ngt_model_v2.pth",
        description="Path to the main NGT recognition model",
    )
    lm_model_path: Path = Field(
        default_factory=lambda: BASE_DIR / "models" / "best_landmark_mlp.pth",
        description="Path to the landmark MLP model",
    )
    hand_landmarker_path: Path = Field(
        default_factory=lambda: BASE_DIR / "models" / "hand_landmarker.task",
        description="Path to the MediaPipe hand landmarker task file",
    )
    frontend_build_dir: Path = Field(
        default_factory=lambda: BASE_DIR.parent / "frontend" / "dist",
        description="Path to the compiled frontend build directory",
    )
    recordings_dir: Path = Field(
        default_factory=lambda: BASE_DIR / "recordings",
        description="Directory where sign-language session recordings are stored",
    )
    collect_storage_dir: Path = Field(
        default_factory=lambda: BASE_DIR / "collected_samples",
        description=(
            "Directory where Collect-mode training samples are written. Local "
            "filesystem for now; a future deployment can point this elsewhere "
            "(or swap the storage layer for Azure Blob)."
        ),
    )

    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------
    server_host: str = Field(
        default="0.0.0.0",
        description="Host address for the uvicorn server",
    )
    server_port: int = Field(
        default=8000,
        description="Port for the uvicorn server",
    )

    # -------------------------------------------------------------------------
    # Inference thresholds
    # -------------------------------------------------------------------------
    efficientnet_confidence_threshold: float = Field(
        default=0.70,
        description=(
            "Minimum EfficientNet softmax confidence required before considering the "
            "landmark MLP fallback; predictions above this threshold skip the MLP entirely"
        ),
    )
    landmark_override_threshold: float = Field(
        default=0.90,
        description=(
            "Minimum landmark MLP confidence required to override the EfficientNet "
            "prediction when EfficientNet confidence is below the primary threshold"
        ),
    )

    # -------------------------------------------------------------------------
    # Sequence builder  (seconds)
    # -------------------------------------------------------------------------
    sequence_letter_hold_sec: float = Field(
        default=1.0,
        description="How long a stable prediction must be held before it is committed as a letter",
    )
    sequence_cooldown_sec: float = Field(
        default=1.0,
        description="Minimum gap required between two consecutive committed letters",
    )
    sequence_space_pause_sec: float = Field(
        default=1.5,
        description="Duration of hand-absent silence that triggers an automatic space insertion",
    )
    sequence_stable_grace_sec: float = Field(
        default=0.4,
        description=(
            "Maximum gap duration for which a briefly-missing stable prediction is "
            "treated as a noisy frame rather than a genuine loss of tracking; "
            "keeps the hold timer running across single-frame smoother blips"
        ),
    )

    # -------------------------------------------------------------------------
    # Prediction smoother
    # -------------------------------------------------------------------------
    smoother_window_size: int = Field(
        default=15,
        description="Number of recent frames kept in the majority-vote smoothing window",
    )
    smoother_acquire_threshold: int = Field(
        default=10,
        description=(
            "Minimum vote count within the window required to acquire stability "
            "(strict bar — 10/15 ≈ 66 %% with defaults)"
        ),
    )
    smoother_sticky_threshold: int = Field(
        default=7,
        description=(
            "Minimum vote count within the window required to keep an already-stable "
            "letter stable (lenient bar — 7/15 ≈ 46 %% with defaults)"
        ),
    )
    smoother_min_confidence: float = Field(
        default=0.55,
        description="Per-frame confidence floor; frames below this value are excluded from smoothing",
    )

    # =========================================================================
    # VALIDATORS
    # =========================================================================

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_input(cls, v: Optional[bool | str]) -> Optional[bool | str]:
        """Normalise string representations of the debug flag to booleans.

        Accepts common truthy/falsy string values including environment-style
        words such as ``"development"`` and ``"production"``.

        :param v: Raw input value from the environment or ``.env`` file.
        :returns: ``True``, ``False``, or the original value if no normalisation
            applies.
        """
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in {"release", "production", "prod", "false", "0", "off"}:
                return False
            if normalized in {"development", "dev", "true", "1", "on"}:
                return True
        return v

    @field_validator("debug")
    @classmethod
    def set_and_verify_debug_mode(cls, v: Optional[bool], info: ValidationInfo) -> bool:
        """Derive and validate the debug flag against the application status.

        If ``debug`` is ``None``, infers ``True`` for ``DEVELOPMENT`` and
        ``False`` for ``PRODUCTION``. Raises if debug is explicitly enabled
        in a production environment.

        :param v: The normalised debug value, or ``None`` if not set.
        :param info: Pydantic validation context providing access to already
            validated fields.
        :returns: The resolved boolean debug flag.
        :raises ValueError: If ``debug=True`` is set while ``status`` is
            ``"PRODUCTION"``.
        """
        status = info.data.get("status")
        if v is None:
            return status == "DEVELOPMENT"
        if status == "PRODUCTION" and v:
            raise ValueError("Debug mode cannot be enabled in PRODUCTION environment")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: Optional[str], info: ValidationInfo) -> str:
        """Validate and resolve the log level string.

        If ``None``, defaults to ``"DEBUG"`` in development and ``"INFO"``
        in production.

        :param v: The raw log level string, or ``None``.
        :param info: Pydantic validation context.
        :returns: A valid uppercase log level string.
        :raises ValueError: If the provided string is not a recognised
            Python logging level name.
        """
        if v is None:
            return "DEBUG" if info.data.get("debug", False) else "INFO"
        valid_levels = [
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARNING",
            "WARN",
            "INFO",
            "DEBUG",
            "NOTSET",
        ]
        if v not in valid_levels:
            raise ValueError(
                f"Invalid log level. Must be one of: {', '.join(valid_levels)}"
            )
        return v

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate that the version string follows semantic versioning.

        :param v: The raw version string.
        :returns: The version string unchanged if valid.
        :raises ValueError: If the string does not match the semver pattern.
        """
        if not re.match(r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$", v):
            raise ValueError(
                "Version must follow semantic versioning (e.g., 1.0.0 or 1.0.0-rc2)"
            )
        return v

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, authors: List[str]) -> List[str]:
        """Validate that at least one non-empty author name is provided.

        :param authors: List of author name strings.
        :returns: The list unchanged if valid.
        :raises ValueError: If the list is empty or any name is blank.
        """
        if not authors:
            raise ValueError("At least one author must be specified")
        for author in authors:
            if not author.strip():
                raise ValueError("Author names cannot be empty")
        return authors

    @field_validator("authors_email")
    @classmethod
    def validate_emails(cls, emails: List[str], info: ValidationInfo) -> List[str]:
        """Validate that email count matches author count and all emails are valid.

        :param emails: List of author email strings.
        :param info: Pydantic validation context providing access to ``authors``.
        :returns: The list unchanged if valid.
        :raises ValueError: If the email count does not match the author count,
            or if any email does not match the expected format.
        """
        authors_count = len(info.data.get("authors", []))
        if len(emails) != authors_count:
            raise ValueError(
                f"Number of emails ({len(emails)}) must match number of authors ({authors_count})"
            )
        email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        for email in emails:
            if not re.match(email_pattern, email):
                raise ValueError(f"Invalid email format: {email}")
        return emails

    @field_validator("log_path")
    @classmethod
    def validate_log_path(cls, path: Optional[Path]) -> Optional[Path]:
        """Pass through the log path unchanged.

        :param path: The raw log path value, or ``None``.
        :returns: The path unchanged.
        """
        if path is None:
            return None
        return path

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Optional[Union[str, List[str]]]) -> List[str]:
        """Parse CORS origins from a list, JSON array string, or comma-separated string.

        :param v: Raw input — a list, a JSON array string, a comma-separated
            string, or ``None``.
        :returns: A list of origin strings. Returns an empty list for ``None``.
        :raises ValueError: If the input type is not supported.
        """
        if v is None:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    pass
            return [origin.strip() for origin in stripped.split(",")]
        raise ValueError(
            "cors_origins must be a list of strings, a comma-separated string, or null"
        )

    @field_validator(
        "training_img_size",
        "training_batch_size",
        "training_epochs",
        "training_patience",
        "training_expected_num_classes",
    )
    @classmethod
    def validate_positive_training_ints(cls, v: int) -> int:
        """Validate that core training integer settings are positive.

        :param v: The integer value to validate.
        :returns: The value unchanged if positive.
        :raises ValueError: If the value is zero or negative.
        """
        if v <= 0:
            raise ValueError("Training integer settings must be positive")
        return v

    @field_validator("training_learning_rate")
    @classmethod
    def validate_positive_training_floats(cls, v: float) -> float:
        """Validate that the learning rate is strictly positive.

        :param v: The learning rate value.
        :returns: The value unchanged if positive.
        :raises ValueError: If the value is zero or negative.
        """
        if v <= 0:
            raise ValueError("Training float settings must be positive")
        return v

    @field_validator("training_eta_min")
    @classmethod
    def validate_training_eta_min(cls, v: float) -> float:
        """Validate that the minimum learning rate is non-negative.

        :param v: The eta_min value.
        :returns: The value unchanged if non-negative.
        :raises ValueError: If the value is negative.
        """
        if v < 0:
            raise ValueError("TRAINING_ETA_MIN cannot be negative")
        return v

    @field_validator("training_val_split", "training_target_accuracy")
    @classmethod
    def validate_training_ratios(cls, v: float) -> float:
        """Validate that training ratio settings are strictly between 0 and 1.

        :param v: The ratio value.
        :returns: The value unchanged if in (0, 1).
        :raises ValueError: If the value is outside the open interval (0, 1).
        """
        if not 0 < v < 1:
            raise ValueError("Training ratio settings must be between 0 and 1")
        return v

    @field_validator("training_n_splits")
    @classmethod
    def validate_training_n_splits(cls, v: int) -> int:
        """Validate that the number of training splits is exactly 1.

        :param v: The n_splits value.
        :returns: The value unchanged if equal to 1.
        :raises ValueError: If the value is not 1.
        """
        if v != 1:
            raise ValueError(
                "Number of splits must be exactly 1 for the current training workflow"
            )
        return v

    @field_validator("training_num_workers")
    @classmethod
    def validate_training_num_workers(cls, v: int) -> int:
        """Validate that the DataLoader worker count is non-negative.

        :param v: The num_workers value.
        :returns: The value unchanged if non-negative.
        :raises ValueError: If the value is negative.
        """
        if v < 0:
            raise ValueError("Number of workers cannot be negative")
        return v

    @field_validator("img_size")
    @classmethod
    def validate_img_size(cls, v: int) -> int:
        """Validate that the inference image size is a positive integer.

        :param v: The img_size value.
        :returns: The value unchanged if positive.
        :raises ValueError: If the value is zero or negative.
        """
        if v <= 0:
            raise ValueError("IMG_SIZE must be a positive integer")
        return v

    @field_validator("server_port")
    @classmethod
    def validate_server_port(cls, v: int) -> int:
        """Validate that the server port is in the valid TCP range.

        :param v: The port number.
        :returns: The value unchanged if in [1, 65535].
        :raises ValueError: If the port is outside the valid range.
        """
        if not 1 <= v <= 65535:
            raise ValueError("server_port must be between 1 and 65535")
        return v

    @field_validator("server_host")
    @classmethod
    def validate_server_host(cls, v: str) -> str:
        """Validate that the server host string is not blank.

        :param v: The host string.
        :returns: The value unchanged if non-empty.
        :raises ValueError: If the string is empty or whitespace only.
        """
        if not v.strip():
            raise ValueError("server_host cannot be empty")
        return v

    @field_validator(
        "efficientnet_confidence_threshold",
        "landmark_override_threshold",
        "smoother_min_confidence",
    )
    @classmethod
    def validate_unit_interval(cls, v: float) -> float:
        """Validate that confidence and threshold values are in the open interval (0, 1).

        :param v: The threshold value.
        :returns: The value unchanged if in (0, 1).
        :raises ValueError: If the value is outside the open interval.
        """
        if not 0.0 < v < 1.0:
            raise ValueError(
                "Confidence/threshold values must be between 0 and 1 (exclusive)"
            )
        return v

    @field_validator(
        "sequence_letter_hold_sec",
        "sequence_cooldown_sec",
        "sequence_space_pause_sec",
        "sequence_stable_grace_sec",
    )
    @classmethod
    def validate_positive_durations(cls, v: float) -> float:
        """Validate that sequence timing values are strictly positive.

        :param v: The duration in seconds.
        :returns: The value unchanged if positive.
        :raises ValueError: If the value is zero or negative.
        """
        if v <= 0:
            raise ValueError("Timing values must be positive")
        return v

    @field_validator(
        "smoother_window_size",
        "smoother_acquire_threshold",
        "smoother_sticky_threshold",
    )
    @classmethod
    def validate_positive_smoother_ints(cls, v: int) -> int:
        """Validate that smoother integer settings are at least 1.

        :param v: The smoother integer value.
        :returns: The value unchanged if >= 1.
        :raises ValueError: If the value is less than 1.
        """
        if v < 1:
            raise ValueError("Smoother integer settings must be at least 1")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate that the database URL uses the asyncpg scheme.

        :param v: The raw database URL string.
        :returns: The URL unchanged if valid.
        :raises ValueError: If the URL does not start with
            ``postgresql+asyncpg://``.
        """
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the postgresql+asyncpg:// scheme. " f"Got: {v!r}"
            )
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate the JWT secret key length and warn if using the default placeholder.

        Allows the default ``"change-me-in-production"`` placeholder through
        with a warning so development environments start without configuration.
        In all other cases, enforces a minimum length of 32 characters.

        :param v: The raw secret key string.
        :returns: The key unchanged if valid.
        :raises ValueError: If the key is not the default placeholder and is
            shorter than 32 characters.
        """
        import logging as _logging

        _log = _logging.getLogger(__name__)
        if v == "change-me-in-production":
            _log.warning(
                "SECRET_KEY is set to the default placeholder — "
                "override it in .env before deploying."
            )
            return v
        if len(v) < 32:
            raise ValueError(
                f"SECRET_KEY must be at least 32 characters, got {len(v)}."
            )
        return v

    @field_validator("jwt_lifetime_seconds")
    @classmethod
    def validate_jwt_lifetime(cls, v: int) -> int:
        """Validate that the JWT token lifetime is a positive integer.

        :param v: The lifetime in seconds.
        :returns: The value unchanged if positive.
        :raises ValueError: If the value is zero or negative.
        """
        if v <= 0:
            raise ValueError("JWT_LIFETIME_SECONDS must be a positive integer.")
        return v

    # =========================================================================
    # GETTERS — application
    # =========================================================================

    def get_debug_state(self) -> bool:
        """Return the debug flag as a boolean.

        :returns: ``True`` if debug mode is active, ``False`` otherwise.
        """
        return bool(self.debug)

    def get_log_level(self) -> int:
        """Convert the string log level to its corresponding integer value.

        :returns: The integer logging level constant from the :mod:`logging`
            module corresponding to ``self.log_level``.
        """
        level_map: Dict[str, int] = {
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        return level_map[self.log_level or "INFO"]

    def get_app_info(self) -> Dict[str, Any]:
        """Return a dictionary with basic application identity information.

        :returns: Dict containing ``name``, ``version``, ``environment``,
            and ``debug`` keys.
        """
        return {
            "name": self.app_name,
            "version": self.version,
            "environment": self.status,
            "debug": self.debug,
        }

    def get_authors_with_emails(self) -> List[Dict[str, str]]:
        """Return a list of authors paired with their email addresses.

        :returns: List of dicts each containing ``name`` and ``email`` keys,
            zipped from ``authors`` and ``authors_email``.
        """
        return [
            {"name": author, "email": email}
            for author, email in zip(self.authors, self.authors_email)
        ]

    def get_logging_config(self) -> Dict[str, Any]:
        """Return logging configuration as a dictionary.

        :returns: Dict containing ``level``, ``level_int``, ``path``,
            and ``directory`` keys.
        """
        return {
            "level": self.log_level,
            "level_int": self.get_log_level(),
            "path": str(self.log_path) if self.log_path is not None else None,
            "directory": (
                str(self.log_path.parent) if self.log_path is not None else None
            ),
        }

    def is_development(self) -> bool:
        """Return ``True`` if the application is running in development mode.

        :returns: ``True`` when ``status`` is ``"DEVELOPMENT"``.
        """
        return self.status == "DEVELOPMENT"

    def is_production(self) -> bool:
        """Return ``True`` if the application is running in production mode.

        :returns: ``True`` when ``status`` is ``"PRODUCTION"``.
        """
        return self.status == "PRODUCTION"

    def get_version_components(self) -> Dict[str, str]:
        """Parse and return the individual components of the semantic version string.

        :returns: Dict with ``major``, ``minor``, ``patch``, ``prerelease``,
            and ``build`` keys. All values are strings; ``prerelease`` and
            ``build`` are empty strings when not present.
        """
        match = re.match(
            r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$",
            self.version,
        )
        if match:
            major, minor, patch, prerelease, build = match.groups()
            return {
                "major": major,
                "minor": minor,
                "patch": patch,
                "prerelease": prerelease or "",
                "build": build or "",
            }
        return {"major": "", "minor": "", "patch": "", "prerelease": "", "build": ""}

    def as_dict(self) -> Dict[str, Any]:
        """Return core settings as a flat dictionary.

        :returns: Dict containing app identity, status, debug, log settings,
            training config, and model config.
        """
        return {
            "app_name": self.app_name,
            "version": self.version,
            "authors": self.authors,
            "authors_email": self.authors_email,
            "status": self.status,
            "debug": self.debug,
            "log_level": self.log_level,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "training": self.get_training_config(),
            "model": self.get_model_config(),
        }

    def get_environment_info(self) -> Dict[str, Any]:
        """Return runtime environment information.

        :returns: Dict containing ``status``, ``debug``, ``is_development``,
            ``is_production``, ``python_version``, and ``platform`` keys.
        """
        return {
            "status": self.status,
            "debug": self.debug,
            "is_development": self.is_development(),
            "is_production": self.is_production(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        }

    # =========================================================================
    # GETTERS — model / device
    # =========================================================================

    def get_device(self) -> Any:
        """Detect and return the best available torch device.

        Selection priority: CUDA > MPS > CPU.

        :returns: A :class:`torch.device` instance for the best available
            compute backend.
        """
        logger = logging.getLogger(__name__)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("Using device: %s", device)
        return device

    def get_training_data_dir(self) -> Path:
        """Return the default local training dataset root path.

        :returns: The :class:`~pathlib.Path` from ``training_local_data_dir``.
        """
        return self.training_local_data_dir

    def get_training_config(self) -> Dict[str, Any]:
        """Return all training-related settings as a dictionary.

        :returns: Dict containing hyperparameters, file name constants,
            dataset paths, and Azure ML asset references.
        """
        return {
            "best_checkpoint_name": self.training_best_checkpoint_name,
            "history_filename": self.training_history_filename,
            "metrics_filename": self.training_metrics_filename,
            "report_filename": self.training_report_filename,
            "class_names_filename": self.training_class_names_filename,
            "img_size": self.training_img_size,
            "batch_size": self.training_batch_size,
            "learning_rate": self.training_learning_rate,
            "epochs": self.training_epochs,
            "patience": self.training_patience,
            "val_split": self.training_val_split,
            "target_accuracy": self.training_target_accuracy,
            "seed": self.training_seed,
            "n_splits": self.training_n_splits,
            "num_workers": self.training_num_workers,
            "pin_memory": self.training_pin_memory,
            "eta_min": self.training_eta_min,
            "expected_num_classes": self.training_expected_num_classes,
            "data_dir": str(self.training_local_data_dir),
            "azure_raw_data_asset_name": self.azure_raw_data_asset_name,
            "azure_raw_data_asset_version": self.azure_raw_data_asset_version,
            "azure_pretrained_checkpoint_asset_name": self.azure_pretrained_checkpoint_asset_name,
            "azure_pretrained_checkpoint_asset_version": self.azure_pretrained_checkpoint_asset_version,
            "azure_instance_type": self.azure_instance_type,
            "azure_prefer_gpu": self.azure_prefer_gpu,
        }

    def get_mlflow_config(self) -> Dict[str, Any]:
        """Return MLflow experiment tracking settings as a dictionary.

        :returns: Dict containing ``enabled``, ``tracking_uri``,
            ``experiment_name``, ``run_name``, ``autolog``, and
            ``log_artifacts`` keys.
        """
        return {
            "enabled": self.mlflow_enabled,
            "tracking_uri": self.mlflow_tracking_uri,
            "experiment_name": self.mlflow_experiment_name,
            "run_name": self.mlflow_run_name,
            "autolog": self.mlflow_autolog,
            "log_artifacts": self.mlflow_log_artifacts,
        }

    def get_model_config(self) -> Dict[str, Any]:
        """Return all model-related paths and constants as a dictionary.

        :returns: Dict containing ``img_size``, ``model_path``,
            ``lm_model_path``, ``hand_landmarker_path``,
            ``frontend_build_dir``, and ``recordings_dir`` keys.
        """
        return {
            "img_size": self.img_size,
            "model_path": str(self.model_path),
            "lm_model_path": str(self.lm_model_path),
            "hand_landmarker_path": str(self.hand_landmarker_path),
            "frontend_build_dir": str(self.frontend_build_dir),
            "recordings_dir": str(self.recordings_dir),
        }

    def get_server_config(self) -> Dict[str, Any]:
        """Return Uvicorn server settings as a dictionary.

        :returns: Dict containing ``host``, ``port``, and ``reload`` keys.
            ``reload`` is ``True`` in development mode.
        """
        return {
            "host": self.server_host,
            "port": self.server_port,
            "reload": self.is_development(),
        }

    def get_inference_config(self) -> Dict[str, Any]:
        """Return inference threshold settings as a dictionary.

        :returns: Dict containing ``efficientnet_confidence_threshold`` and
            ``landmark_override_threshold`` keys.
        """
        return {
            "efficientnet_confidence_threshold": self.efficientnet_confidence_threshold,
            "landmark_override_threshold": self.landmark_override_threshold,
        }

    def get_sequence_config(self) -> Dict[str, Any]:
        """Return sequence-builder timing settings as a dictionary.

        :returns: Dict containing ``letter_hold_sec``, ``cooldown_sec``,
            ``space_pause_sec``, and ``stable_grace_sec`` keys.
        """
        return {
            "letter_hold_sec": self.sequence_letter_hold_sec,
            "cooldown_sec": self.sequence_cooldown_sec,
            "space_pause_sec": self.sequence_space_pause_sec,
            "stable_grace_sec": self.sequence_stable_grace_sec,
        }

    def get_smoother_config(self) -> Dict[str, Any]:
        """Return prediction-smoother settings as a dictionary.

        :returns: Dict containing ``window_size``, ``acquire_threshold``,
            ``sticky_threshold``, and ``min_confidence`` keys.
        """
        return {
            "window_size": self.smoother_window_size,
            "acquire_threshold": self.smoother_acquire_threshold,
            "sticky_threshold": self.smoother_sticky_threshold,
            "min_confidence": self.smoother_min_confidence,
        }


def get_settings(use_test_env: bool = False) -> Settings:
    """Create and return a :class:`Settings` instance.

    When ``use_test_env`` is ``True``, loads from ``.env.test`` in the same
    directory as the main ``.env`` file instead of the default ``.env``.

    :param use_test_env: If ``True``, load settings from ``.env.test``
        for use in the test suite.
    :returns: A fully validated :class:`Settings` instance.
    """
    if use_test_env:
        env_test_path = Path(DOTENV).resolve().parent / ".env.test"

        class TestSettings(Settings):
            model_config = SettingsConfigDict(
                **{**Settings.model_config, "env_file": str(env_test_path)}
            )

        return TestSettings()
    return Settings()


settings = get_settings()

if __name__ == "__main__":
    print(settings)
