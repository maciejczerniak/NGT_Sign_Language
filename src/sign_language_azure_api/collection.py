"""Azure Blob storage helpers for pending frontend collection samples."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
import io
import uuid

from azure.storage.blob import BlobClient, ContentSettings
from PIL import Image, UnidentifiedImageError

from sign_language_azure_api.settings import AzureApiSettings


ALLOWED_SOURCES = {"camera", "upload", "auto"}


class CollectionStorageError(RuntimeError):
    """Raised when a collected sample cannot be stored."""


class CollectionValidationError(ValueError):
    """Raised when a collected sample is invalid."""


def decode_collection_image(image_data: str, max_bytes: int) -> bytes:
    """Decode and validate one collected JPEG-compatible image.

    Args:
        image_data: Raw base64 image or data URL.
        max_bytes: Maximum decoded image size.

    Returns:
        Validated image bytes.

    Raises:
        CollectionValidationError: If the image is invalid or exceeds the limit.
    """
    raw_base64 = (
        image_data.partition(",")[2] if image_data.startswith("data:") else image_data
    )
    try:
        image_bytes = base64.b64decode(raw_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CollectionValidationError("Invalid base64 image data.") from exc

    if not image_bytes:
        raise CollectionValidationError("Empty image data.")
    if len(image_bytes) > max_bytes:
        raise CollectionValidationError(
            f"Image exceeds the maximum size of {max_bytes} bytes."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            converted = image.convert("RGB")
            output = io.BytesIO()
            converted.save(output, format="JPEG", quality=95)
    except (UnidentifiedImageError, OSError) as exc:
        raise CollectionValidationError(
            "Decoded content is not a valid image."
        ) from exc
    return output.getvalue()


def validate_collection_metadata(letter: str, source: str) -> tuple[str, str]:
    """Normalize and validate a collection label and source.

    Args:
        letter: User-provided NGT fingerspelling label.
        source: Collection source.

    Returns:
        Normalized ``(letter, source)``.

    Raises:
        CollectionValidationError: If metadata is unsupported.
    """
    normalized_letter = letter.strip().upper()
    normalized_source = source.strip().lower()
    if (
        len(normalized_letter) != 1
        or normalized_letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ):
        raise CollectionValidationError("Letter must be one character from A to Z.")
    if normalized_source not in ALLOWED_SOURCES:
        raise CollectionValidationError(
            f"Source must be one of {sorted(ALLOWED_SOURCES)}."
        )
    return normalized_letter, normalized_source


def store_pending_sample(
    *,
    image_bytes: bytes,
    letter: str,
    source: str,
    language: str,
    settings: AzureApiSettings,
) -> tuple[str, str]:
    """Store one sample in the configured pending Azure Blob path.

    Args:
        image_bytes: Validated image bytes.
        letter: Normalized label.
        source: Normalized collection source.
        language: Sign language identifier.
        settings: Azure API collection storage settings.

    Returns:
        ``(sample_id, blob_path)``.

    Raises:
        CollectionStorageError: If collection storage is not configured or the
            upload fails.
    """
    account = settings.azure_api_collect_storage_account.strip()
    container = settings.azure_api_collect_container.strip()
    sas_token = settings.azure_api_collect_sas_token.strip().lstrip("?")
    prefix = settings.azure_api_collect_prefix.strip().strip("/") or "pending"
    if not account or not container or not sas_token:
        raise CollectionStorageError("Azure collection storage is not configured.")

    sample_id = str(uuid.uuid4())
    blob_path = f"{prefix}/{letter}/{sample_id}.jpg"
    blob_url = f"https://{account}.blob.core.windows.net/{container}/{blob_path}"
    blob = BlobClient.from_blob_url(f"{blob_url}?{sas_token}")
    metadata = {
        "letter": letter,
        "source": source,
        "language": language,
        "review_status": "pending",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        blob.upload_blob(
            image_bytes,
            overwrite=False,
            metadata=metadata,
            content_settings=ContentSettings(content_type="image/jpeg"),
        )
    except Exception as exc:
        raise CollectionStorageError("Could not store sample in Azure Blob.") from exc
    return sample_id, blob_path
