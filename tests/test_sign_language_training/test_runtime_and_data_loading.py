"""Tests targeting runtime.py and data_loading.py coverage gaps.

runtime.py missing lines:
- 30   : torch.cuda.manual_seed_all branch inside set_seed
- 38-45: get_device() function

data_loading.py missing lines:
- 24: FileNotFoundError for missing directory
- 26: NotADirectoryError when path is a file
- 33: ValueError when ImageFolder has no classes
- 35: ValueError when ImageFolder has no images
- 54: ValueError for split_ratio out of range
- 56: ValueError for empty targets list
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sign_language_training import data_loading
from sign_language_training.data_loading import (
    create_stratified_split,
    load_dataset,
)
from sign_language_training.runtime import get_device, set_seed


# ---------------------------------------------------------------------------
# runtime.py — set_seed CUDA branch (line 30)
# ---------------------------------------------------------------------------


class TestSetSeedCudaBranch:
    def test_calls_cuda_manual_seed_all_when_cuda_available(self) -> None:
        # Also mock torch.manual_seed because newer PyTorch versions call
        # manual_seed_all internally inside manual_seed when CUDA is available,
        # which would pollute the call count we want to assert on.
        with (
            patch(
                "sign_language_training.runtime.torch.cuda.is_available",
                return_value=True,
            ),
            patch("sign_language_training.runtime.torch.manual_seed"),
            patch(
                "sign_language_training.runtime.torch.cuda.manual_seed_all"
            ) as mock_all,
        ):
            set_seed(99)

        mock_all.assert_called_once_with(99)

    def test_does_not_call_cuda_manual_seed_all_when_cuda_unavailable(self) -> None:
        with (
            patch(
                "sign_language_training.runtime.torch.cuda.is_available",
                return_value=False,
            ),
            patch("sign_language_training.runtime.torch.manual_seed"),
            patch(
                "sign_language_training.runtime.torch.cuda.manual_seed_all"
            ) as mock_all,
        ):
            set_seed(99)

        mock_all.assert_not_called()


# ---------------------------------------------------------------------------
# runtime.py — get_device (lines 38-45)
# ---------------------------------------------------------------------------


class TestGetDevice:
    def test_returns_cuda_when_available(self) -> None:
        with (
            patch(
                "sign_language_training.runtime.torch.cuda.is_available",
                return_value=True,
            ),
        ):
            device = get_device()

        assert device.type == "cuda"

    def test_returns_mps_when_cuda_unavailable_and_mps_available(self) -> None:
        with (
            patch(
                "sign_language_training.runtime.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "sign_language_training.runtime.torch.backends.mps.is_available",
                return_value=True,
            ),
        ):
            device = get_device()

        assert device.type == "mps"

    def test_returns_cpu_when_neither_cuda_nor_mps_available(self) -> None:
        with (
            patch(
                "sign_language_training.runtime.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "sign_language_training.runtime.torch.backends.mps.is_available",
                return_value=False,
            ),
        ):
            device = get_device()

        assert device.type == "cpu"


# ---------------------------------------------------------------------------
# data_loading.py — load_dataset error paths (lines 24, 26, 33, 35)
# ---------------------------------------------------------------------------


class TestLoadDatasetErrorPaths:
    def test_raises_file_not_found_for_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist"
        with pytest.raises(
            FileNotFoundError, match="Training data directory not found"
        ):
            load_dataset(missing)

    def test_raises_not_a_directory_error_for_file_path(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("hello")
        with pytest.raises(NotADirectoryError, match="not a directory"):
            load_dataset(file_path)

    def test_raises_value_error_when_no_class_folders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class EmptyClassFolder:
            def __init__(self, root, transform=None):
                self.classes = []
                self.targets = []

        monkeypatch.setattr(data_loading.datasets, "ImageFolder", EmptyClassFolder)
        with pytest.raises(ValueError, match="No class folders found"):
            load_dataset(tmp_path)

    def test_raises_value_error_when_no_images(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class NoImagesFolder:
            def __init__(self, root, transform=None):
                self.classes = ["A"]
                self.targets = []

        monkeypatch.setattr(data_loading.datasets, "ImageFolder", NoImagesFolder)
        with pytest.raises(ValueError, match="No images found"):
            load_dataset(tmp_path)


# ---------------------------------------------------------------------------
# data_loading.py — create_stratified_split error paths (lines 54, 56)
# ---------------------------------------------------------------------------


class TestCreateStratifiedSplitErrorPaths:
    def test_raises_value_error_for_split_ratio_of_zero(self) -> None:
        targets = np.array([0, 1, 0, 1], dtype=np.int_)
        with pytest.raises(ValueError, match="split_ratio must be between 0 and 1"):
            create_stratified_split(
                targets=targets, n_splits=1, split_ratio=0.0, seed=42
            )

    def test_raises_value_error_for_split_ratio_of_one(self) -> None:
        targets = np.array([0, 1, 0, 1], dtype=np.int_)
        with pytest.raises(ValueError, match="split_ratio must be between 0 and 1"):
            create_stratified_split(
                targets=targets, n_splits=1, split_ratio=1.0, seed=42
            )

    def test_raises_value_error_for_split_ratio_above_one(self) -> None:
        targets = np.array([0, 1, 0, 1], dtype=np.int_)
        with pytest.raises(ValueError, match="split_ratio must be between 0 and 1"):
            create_stratified_split(
                targets=targets, n_splits=1, split_ratio=1.5, seed=42
            )

    def test_raises_value_error_for_empty_targets_list(self) -> None:
        with pytest.raises(ValueError, match="Cannot split an empty target list"):
            create_stratified_split(targets=[], n_splits=1, split_ratio=0.2, seed=42)

    def test_raises_value_error_for_empty_numpy_array(self) -> None:
        with pytest.raises(ValueError, match="Cannot split an empty target list"):
            create_stratified_split(
                targets=np.array([], dtype=np.int_),
                n_splits=1,
                split_ratio=0.2,
                seed=42,
            )
