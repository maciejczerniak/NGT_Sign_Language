"""Dataset inventory helpers for trigger policy decisions.

Provides utilities to build a deterministic file manifest for an
ImageFolder dataset and compare it against a previously recorded state
to detect new or removed images.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetInventory:
    """Immutable snapshot of an ImageFolder dataset at a point in time.

    Args:
        root: Resolved absolute path to the dataset root directory.
        image_count: Total number of supported image files found.
        files: Sorted list of relative file paths (forward-slash
            separated) for all images under ``root``.
        manifest_hash: SHA-256 hex digest computed from file names,
            sizes, and contents, used for deterministic change detection.
        removed_image_count: Number of images present in the previous
            state but missing from this snapshot. Defaults to ``0``.
    """

    root: str
    image_count: int
    files: list[str]
    manifest_hash: str
    removed_image_count: int = 0


def build_dataset_inventory(data_dir: Path) -> DatasetInventory:
    """Build a deterministic file manifest for an ImageFolder dataset.

    Recursively finds all supported image files under ``data_dir``, sorts
    them for determinism, and computes a SHA-256 digest over their relative
    paths, sizes, and contents.

    Args:
        data_dir: Root directory of the ImageFolder dataset.

    Returns:
        A :class:`DatasetInventory` snapshot of the current dataset
            state.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
        NotADirectoryError: If ``data_dir`` is not a directory.
        ValueError: If no supported image files are found under
            ``data_dir``.
    """
    root = data_dir.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    files = [str(path.relative_to(root)).replace("\\", "/") for path in paths]

    if not files:
        raise ValueError(f"Dataset directory contains no supported images: {root}")

    digest = hashlib.sha256()
    for path, file_name in zip(paths, files):
        digest.update(file_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())

    return DatasetInventory(
        root=str(root),
        image_count=len(files),
        files=files,
        manifest_hash=digest.hexdigest(),
    )


def count_new_images(
    current: DatasetInventory,
    previous_files: list[str] | None,
) -> int:
    """Return the number of images present now but absent in the previous state.

    If no previous state exists, returns the full current image count since
    all images are considered new.

    Args:
        current: The :class:`DatasetInventory` snapshot of the current
            dataset state.
        previous_files: List of relative file paths from the previously
            recorded state, or ``None`` if no prior state exists.

    Returns:
        Count of image files in ``current`` that are not in
            ``previous_files``.
    """
    if not previous_files:
        return current.image_count

    previous = set(previous_files)
    current_files = set(current.files)

    return len(current_files - previous)


def count_removed_images(
    current: DatasetInventory,
    previous_files: list[str] | None,
) -> int:
    """Return the number of previously tracked images missing from the current state.

    If no previous state exists, returns ``0`` since there is nothing to
    compare against.

    Args:
        current: The :class:`DatasetInventory` snapshot of the current
            dataset state.
        previous_files: List of relative file paths from the previously
            recorded state, or ``None`` if no prior state exists.

    Returns:
        Count of image files in ``previous_files`` that are no longer
            present in ``current``.
    """
    if not previous_files:
        return 0

    previous = set(previous_files)
    current_files = set(current.files)

    return len(previous - current_files)
