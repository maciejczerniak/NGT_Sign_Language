"""Settings for the training trigger FastAPI service."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingTriggerApiSettings(BaseSettings):
    """Configuration for the training trigger FastAPI service.

    All fields are read from environment variables (case-insensitive).
    No ``.env`` file is loaded by default — settings are expected to come
    from the environment or be overridden in tests.

    Args:
        training_trigger_base_dir: Optional base directory used to
            resolve relative ``data_dir`` and ``state_path`` values. Defaults
            to the current working directory if not set.
        training_trigger_api_key: API key required in the ``X-API-Key``
            header for ``POST /train``. The endpoint returns ``503`` if this
            is not configured.
        training_trigger_data_dir: Relative or absolute path to the
            local ImageFolder dataset root used for change detection.
        training_trigger_state_path: Relative or absolute path to the
            JSON file persisting the last trigger state.
        training_trigger_min_new_images: Minimum number of new images
            required to trigger ``data_change`` retraining.
        training_trigger_interval_days: Minimum days since the
            last submission required to trigger ``scheduled`` retraining.
        training_trigger_experiment_name: Azure ML experiment name used
            when submitting retraining pipeline jobs.
    """

    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    training_trigger_base_dir: str | None = Field(default=None)
    training_trigger_api_key: str | None = Field(default=None)
    training_trigger_data_dir: str = Field(default="data/raw")
    training_trigger_state_path: str = Field(
        default="state/training_trigger_state.json"
    )
    training_trigger_min_new_images: int = Field(default=10, ge=1)
    training_trigger_interval_days: int = Field(
        default=7,
        ge=1,
    )
    training_trigger_experiment_name: str = Field(default="sign-language-training")


@lru_cache
def get_settings() -> TrainingTriggerApiSettings:
    """Return the cached trigger API settings instance.

    Uses :func:`functools.lru_cache` so settings are only instantiated
    once per process. The cache can be cleared in tests via
    ``get_settings.cache_clear()``.

    Returns:
        The singleton :class:`TrainingTriggerApiSettings` instance.
    """
    return TrainingTriggerApiSettings()


class _SettingsProxy:
    """Lazy proxy that forwards attribute access to the cached settings instance.

    Allows module-level ``settings`` usage without eagerly instantiating
    the settings object at import time.
    """

    def __getattr__(self, name: str) -> object:
        """Read a setting from the cached settings instance.

        Args:
            name: Setting attribute name.

        Returns:
            Configured setting value.
        """
        return getattr(get_settings(), name)

    def __setattr__(self, name: str, value: object) -> None:
        """Update a setting on the cached settings instance.

        Args:
            name: Setting attribute name.
            value: Replacement setting value.
        """
        setattr(get_settings(), name, value)


settings = _SettingsProxy()


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a path value against the configured base directory or cwd.

    Absolute paths are returned unchanged. Relative paths are resolved
    against ``training_trigger_base_dir`` from settings if set, otherwise
    against the current working directory.

    Args:
        path_value: A relative or absolute path string or
            :class:`~pathlib.Path`.

    Returns:
        A resolved absolute :class:`~pathlib.Path`.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    settings = get_settings()
    base_dir = (
        Path(settings.training_trigger_base_dir)
        if settings.training_trigger_base_dir
        else Path.cwd()
    )
    return (base_dir / path).resolve()
