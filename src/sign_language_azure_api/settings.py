"""Settings for the separate Azure ML endpoint API."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV = REPO_ROOT / ".env"


class AzureApiSettings(BaseSettings):
    """Configuration for the standalone Azure endpoint proxy API."""

    model_config = SettingsConfigDict(
        env_file=str(DOTENV),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    azure_api_app_name: str = Field(default="Sign Language Azure API")
    azure_api_host: str = Field(default="0.0.0.0")
    azure_api_port: int = Field(default=8010)

    azure_api_online_endpoint_url: str = Field(default="")
    azure_api_online_endpoint_key: str = Field(default="")
    azure_api_online_timeout_seconds: float = Field(default=10.0)
    azure_api_online_model_version: str = Field(default="")
    azure_api_default_deployment: str | None = Field(default=None)

    azure_api_collect_storage_account: str = Field(default="")
    azure_api_collect_container: str = Field(default="signlang-r2-collected")
    azure_api_collect_sas_token: str = Field(default="")
    azure_api_collect_prefix: str = Field(default="pending")
    azure_api_collect_max_bytes: int = Field(default=5_000_000)

    azure_api_ml_subscription_id: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AZURE_API_ML_SUBSCRIPTION_ID",
            "AZURE_SUBSCRIPTION_ID",
        ),
    )
    azure_api_ml_resource_group: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AZURE_API_ML_RESOURCE_GROUP",
            "AZURE_ML_RESOURCE_GROUP",
            "AZURE_RESOURCE_GROUP",
        ),
    )
    azure_api_ml_workspace: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AZURE_API_ML_WORKSPACE",
            "AZURE_WORKSPACE",
        ),
    )
    azure_api_online_endpoint_name: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AZURE_API_ONLINE_ENDPOINT_NAME",
            "AZURE_ONLINE_ENDPOINT_NAME",
        ),
    )

    @field_validator("azure_api_online_timeout_seconds")
    @classmethod
    def validate_timeout(cls, value: float) -> float:
        """Validate the Azure endpoint request timeout."""
        if value <= 0:
            raise ValueError("Azure API online timeout must be positive.")
        return value

    @field_validator("azure_api_port")
    @classmethod
    def validate_port(cls, value: int) -> int:
        """Validate the standalone API listening port."""
        if not 1 <= value <= 65535:
            raise ValueError("Azure API port must be between 1 and 65535.")
        return value

    @field_validator("azure_api_collect_max_bytes")
    @classmethod
    def validate_collect_max_bytes(cls, value: int) -> int:
        """Validate the maximum accepted collection image size."""
        if value <= 0:
            raise ValueError("Azure collection maximum image size must be positive.")
        return value


settings = AzureApiSettings()
