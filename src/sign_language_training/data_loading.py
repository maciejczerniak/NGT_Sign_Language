"""Dataset loading and deterministic train/validation split helpers.

Provides utilities to load ImageFolder datasets, create stratified
train/validation splits, and build PyTorch DataLoaders for both
single-directory and pre-split directory layouts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sign_language_training.configuration import TrainingConfig

logger = logging.getLogger(__name__)


def load_dataset(
    data_dir: str | Path,
) -> tuple[datasets.ImageFolder, list[str], list[int]]:
    """Load a raw ImageFolder dataset from disk.

    Args:
        data_dir: Root directory of the ImageFolder dataset. Each
            subdirectory is treated as one class.

    Returns:
        A three-tuple of ``(dataset, class_names, targets)`` where
            ``dataset`` is the loaded :class:`~torchvision.datasets.ImageFolder`,
            ``class_names`` is the sorted list of class folder names, and
            ``targets`` is the list of integer class indices for each sample.

    Raises:
        FileNotFoundError: If ``data_dir`` does not exist.
        NotADirectoryError: If ``data_dir`` is not a directory.
        ValueError: If no class folders or no images are found.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Training data directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Training data path is not a directory: {root}")

    dataset = datasets.ImageFolder(root=str(root))
    class_names = list(dataset.classes)
    targets = list(dataset.targets)

    if not class_names:
        raise ValueError(f"No class folders found in {root}")
    if len(targets) == 0:
        raise ValueError(f"No images found in {root}")

    logger.info("Loaded ImageFolder dataset from %s", root)
    logger.info("Dataset size: %d", len(dataset))
    logger.info("Classes: %s", class_names)

    return dataset, class_names, targets


def create_stratified_split(
    targets: Sequence[int],
    n_splits: int,
    split_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Create a deterministic stratified train/validation index split.

    Args:
        targets: Sequence of integer class labels, one per sample.
        n_splits: Number of splits — must be ``1`` for the current workflow.
        split_ratio: Fraction of samples to assign to the validation set.
            Must be strictly between 0 and 1.
        seed: Random seed for reproducible splitting.

    Returns:
        A two-tuple of ``(train_indices, val_indices)`` as plain Python
            lists of integer sample indices.

    Raises:
        ValueError: If ``n_splits`` is not 1, ``split_ratio`` is outside
            (0, 1), or ``targets`` is empty.
    """
    if n_splits != 1:
        raise ValueError("Only n_splits=1 is supported by the current workflow.")
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between 0 and 1.")
    if len(targets) == 0:
        raise ValueError("Cannot split an empty target list.")

    indices = list(range(len(targets)))

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=split_ratio,
        random_state=seed,
    )

    train_idx, val_idx = next(splitter.split(indices, targets))

    train_indices = train_idx.tolist()
    val_indices = val_idx.tolist()

    logger.info(
        "Created stratified split: train=%d, val=%d, val_split=%.2f, seed=%d",
        len(train_indices),
        len(val_indices),
        split_ratio,
        seed,
    )

    return train_indices, val_indices


def create_dataloaders(
    data_dir: str | Path,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from a single ImageFolder root.

    Loads the same ImageFolder twice — once with ``train_transform`` and once
    with ``val_transform`` — then wraps each in a :class:`~torch.utils.data.Subset`
    using the provided index lists.

    Args:
        data_dir: Root directory of the ImageFolder dataset.
        train_idx: Indices into the dataset selecting training samples.
        val_idx: Indices into the dataset selecting validation samples.
        train_transform: Transform pipeline applied to training images.
        val_transform: Transform pipeline applied to validation images.
        config: :class:`~sign_language_training.configuration.TrainingConfig`
            providing ``batch_size``, ``num_workers``, and ``pin_memory``.

    Returns:
        A two-tuple of ``(train_loader, val_loader)``.
    """
    root = Path(data_dir)

    train_dataset = datasets.ImageFolder(root=str(root), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(root), transform=val_transform)

    train_subset = Subset(train_dataset, list(train_idx))
    val_subset = Subset(val_dataset, list(val_idx))

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    logger.info(
        "Created dataloaders: train_batches=%d, val_batches=%d, batch_size=%d",
        len(train_loader),
        len(val_loader),
        config.batch_size,
    )

    return train_loader, val_loader


def create_presplit_dataloaders(
    train_dir: Path,
    val_dir: Path,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create DataLoaders from pre-split train and validation directories.

    Used when the preprocessing component has already split the dataset into
    separate directories. Validates that the class names in both directories
    are identical before building the loaders.

    Args:
        train_dir: Root of the augmented training ImageFolder directory.
        val_dir: Root of the clean validation ImageFolder directory.
        train_transform: Transform pipeline applied to training images.
        val_transform: Transform pipeline applied to validation images.
        config: :class:`~sign_language_training.configuration.TrainingConfig`
            providing ``batch_size``, ``num_workers``, and ``pin_memory``.

    Returns:
        A three-tuple of ``(train_loader, val_loader, class_names)``
            where ``class_names`` is the sorted list from the training directory.

    Raises:
        ValueError: If the sorted class names in ``train_dir`` and
            ``val_dir`` do not match.
    """
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

    train_classes = sorted(train_dataset.classes)
    val_classes = sorted(val_dataset.classes)
    if train_classes != val_classes:
        raise ValueError(
            f"Class mismatch between train and val directories.\n"
            f"  train: {train_classes}\n"
            f"  val  : {val_classes}"
        )

    class_names = train_dataset.classes

    logger.info(
        "Pre-split dataloaders — train=%d val=%d classes=%d",
        len(train_dataset),
        len(val_dataset),
        len(class_names),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, class_names
