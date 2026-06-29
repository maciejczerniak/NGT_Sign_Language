"""Tests for Azure Blob storage of pending collection samples."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from sign_language_azure_api.collection import (
    CollectionStorageError,
    CollectionValidationError,
    decode_collection_image,
    store_pending_sample,
    validate_collection_metadata,
)
from sign_language_azure_api.settings import AzureApiSettings


def test_decode_collection_image_accepts_valid_image(dummy_image_b64: str) -> None:
    result = decode_collection_image(dummy_image_b64, max_bytes=10_000)

    with Image.open(io.BytesIO(result)) as image:
        assert image.format == "JPEG"


@pytest.mark.parametrize("image_data", ["", "not-base64"])
def test_decode_collection_image_rejects_invalid_data(image_data: str) -> None:
    with pytest.raises(CollectionValidationError):
        decode_collection_image(image_data, max_bytes=10_000)


def test_decode_collection_image_rejects_oversized_image(
    dummy_image_b64: str,
) -> None:
    with pytest.raises(CollectionValidationError, match="maximum size"):
        decode_collection_image(dummy_image_b64, max_bytes=1)


def test_validate_collection_metadata_normalizes_values() -> None:
    assert validate_collection_metadata(" a ", " CAMERA ") == ("A", "camera")


@pytest.mark.parametrize(
    ("letter", "source"),
    [("1", "camera"), ("A", "unknown")],
)
def test_validate_collection_metadata_rejects_invalid_values(
    letter: str,
    source: str,
) -> None:
    with pytest.raises(CollectionValidationError):
        validate_collection_metadata(letter, source)


def test_store_pending_sample_uploads_with_metadata() -> None:
    settings = AzureApiSettings(
        azure_api_collect_storage_account="storage",
        azure_api_collect_container="collected",
        azure_api_collect_sas_token="?sas-token",
        azure_api_collect_prefix="pending",
    )
    blob = MagicMock()

    with (
        patch(
            "sign_language_azure_api.collection.BlobClient.from_blob_url",
            return_value=blob,
        ) as from_url,
        patch(
            "sign_language_azure_api.collection.uuid.uuid4",
            return_value="sample-id",
        ),
    ):
        sample_id, blob_path = store_pending_sample(
            image_bytes=b"image",
            letter="A",
            source="camera",
            language="NGT",
            settings=settings,
        )

    assert sample_id == "sample-id"
    assert blob_path == "pending/A/sample-id.jpg"
    from_url.assert_called_once_with(
        "https://storage.blob.core.windows.net/collected/"
        "pending/A/sample-id.jpg?sas-token"
    )
    upload = blob.upload_blob.call_args
    assert upload.args == (b"image",)
    assert upload.kwargs["overwrite"] is False
    assert upload.kwargs["metadata"]["review_status"] == "pending"
    assert upload.kwargs["metadata"]["letter"] == "A"
    assert upload.kwargs["content_settings"].content_type == "image/jpeg"


def test_store_pending_sample_requires_configuration() -> None:
    with pytest.raises(CollectionStorageError, match="not configured"):
        store_pending_sample(
            image_bytes=b"image",
            letter="A",
            source="camera",
            language="NGT",
            settings=AzureApiSettings(),
        )


def test_store_pending_sample_wraps_upload_failure() -> None:
    settings = AzureApiSettings(
        azure_api_collect_storage_account="storage",
        azure_api_collect_container="collected",
        azure_api_collect_sas_token="sas-token",
    )
    blob = MagicMock()
    blob.upload_blob.side_effect = RuntimeError("offline")

    with patch(
        "sign_language_azure_api.collection.BlobClient.from_blob_url",
        return_value=blob,
    ):
        with pytest.raises(CollectionStorageError, match="Could not store"):
            store_pending_sample(
                image_bytes=b"image",
                letter="A",
                source="camera",
                language="NGT",
                settings=settings,
            )
