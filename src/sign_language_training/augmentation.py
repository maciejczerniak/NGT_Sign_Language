"""Offline data augmentation component for the NGT training pipeline.

Performs three tasks in sequence:

1. Stratified 80/10/10 split of the raw ImageFolder dataset.
2. Augment the train split only (N copies per image).
3. Register the augmented train split as an Azure ML data asset
   tagged with the source ``ngt-raw`` version so downstream jobs can
   detect whether augmentation needs to be re-run.

Usage::

    python -m sign_language_training.augmentation \\
        --input-dir  /data/ngt-raw \\
        --output-train-dir  /data/aug/train \\
        --output-val-dir    /data/aug/val \\
        --output-test-dir   /data/aug/test \\
        --copies 4 --seed 42
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import typer
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)
app = typer.Typer(
    name="augmentation",
    help="Stratified split and offline augmentation for the NGT pipeline.",
    add_completion=False,
)


def _build_augment_fn(img_size: int) -> Any:
    """Build the torchvision augmentation transform pipeline.

    Imports torchvision lazily to avoid pulling it in at module level.
    Applies random resized crop, horizontal flip, colour jitter, and
    Gaussian blur.

    Args:
        img_size: Target output image size in pixels applied to both
            height and width by the random resized crop transform.

    Returns:
        A :class:`~torchvision.transforms.Compose` augmentation
            pipeline.
    """
    from torchvision import transforms  # noqa: PLC0415

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    )


def _collect_samples(
    input_dir: Path,
) -> tuple[list[Path], list[str], list[int]]:
    """Collect image paths, class names, and labels from an ImageFolder.

    Args:
        input_dir: Root directory containing one subdirectory per class.

    Returns:
        Image paths, sorted class names, and integer labels.

    Raises:
        ValueError: If no class subdirectories are present.
    """
    class_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in {input_dir}")

    class_names = [d.name for d in class_dirs]
    image_paths: list[Path] = []
    labels: list[int] = []

    for idx, class_dir in enumerate(class_dirs):
        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        image_paths.extend(images)
        labels.extend([idx] * len(images))

    return image_paths, class_names, labels


def _copy_split(
    image_paths: list[Path],
    labels: list[int],
    class_names: list[str],
    indices: list[int],
    output_dir: Path,
) -> None:
    """Copy a dataset subset while preserving its class directories.

    Args:
        image_paths: All source image paths.
        labels: Integer class label for each image.
        class_names: Class names indexed by label.
        indices: Indices of images to copy.
        output_dir: Destination ImageFolder root.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        (output_dir / class_name).mkdir(exist_ok=True)

    for i in indices:
        src = image_paths[i]
        class_name = class_names[labels[i]]
        shutil.copy2(src, output_dir / class_name / src.name)


def stratified_split(
    input_dir: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> list[str]:
    """Split an ImageFolder dataset into train, validation, and test sets.

    Uses two sequential :class:`~sklearn.model_selection.StratifiedShuffleSplit`
    operations to preserve class distribution across all three splits. The
    test ratio is computed as ``1 - train_ratio - val_ratio``.

    Args:
        input_dir: Root directory of the ImageFolder dataset to split.
        train_dir: Output directory for training images.
        val_dir: Output directory for validation images.
        test_dir: Output directory for test images.
        train_ratio: Fraction of the dataset to use for training.
        val_ratio: Fraction of the dataset to use for validation.
        seed: Random seed for reproducible splitting.

    Returns:
        Sorted list of class name strings from the dataset.

    Raises:
        ValueError: If no class subdirectories are found in ``input_dir``.
    """
    image_paths, class_names, labels = _collect_samples(input_dir)
    labels_arr = np.array(labels)
    indices = np.arange(len(labels_arr))

    test_size = 1.0 - train_ratio - val_ratio
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    trainval_idx, test_idx = next(sss_test.split(indices, labels_arr))

    val_size_relative = val_ratio / (train_ratio + val_ratio)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_relative, random_state=seed
    )
    train_idx, val_idx = next(sss_val.split(trainval_idx, labels_arr[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

    logger.info(
        "Split: total=%d train=%d val=%d test=%d",
        len(labels_arr),
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    _copy_split(image_paths, labels, class_names, train_idx.tolist(), train_dir)
    _copy_split(image_paths, labels, class_names, val_idx.tolist(), val_dir)
    _copy_split(image_paths, labels, class_names, test_idx.tolist(), test_dir)

    return class_names


def augment_dir(
    source_dir: Path,
    output_dir: Path,
    copies: int,
    img_size: int,
    seed: int,
) -> None:
    """Copy originals and generate augmented training images.

    Args:
        source_dir: Root of the unaugmented training split.
        output_dir: Destination root for originals and augmented copies.
        copies: Number of augmented copies to create per image.
        img_size: Square augmentation output size in pixels.
        seed: Random seed for reproducible augmentation.
    """
    random.seed(seed)
    np.random.seed(seed)
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    augment = _build_augment_fn(img_size)

    class_dirs = sorted(p for p in source_dir.iterdir() if p.is_dir())
    total_orig = 0
    total_aug = 0

    for class_dir in class_dirs:
        out_class = output_dir / class_dir.name
        out_class.mkdir(parents=True, exist_ok=True)

        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        for img_path in images:
            shutil.copy2(img_path, out_class / img_path.name)
            total_orig += 1

            img = Image.open(img_path).convert("RGB")
            for i in range(copies):
                aug = augment(img)
                aug.save(out_class / f"{img_path.stem}_aug{i:02d}{img_path.suffix}")
                total_aug += 1

        logger.info(
            "  %s: %d originals → %d total",
            class_dir.name,
            len(images),
            len(images) * (1 + copies),
        )

    logger.info(
        "Augmentation done — %d originals + %d augmented = %d total",
        total_orig,
        total_aug,
        total_orig + total_aug,
    )


def register_split_asset(
    split_dir: Path,
    asset_name: str,
    ngt_raw_version: str,
    split_name: str,
) -> None:
    """Register the split directory as an Azure ML ``URI_FOLDER`` data asset.

    Tags the registered asset with ``ngt_raw_version`` so the pipeline
    submission script can detect whether augmentation needs to be re-run
    when the source raw data version changes.

    Only executes when the ``AZUREML_ARM_SUBSCRIPTION`` environment variable
    is present, i.e. inside an Azure ML job. Silently skips in local
    environments.

    Args:
        split_dir: Path to the split directory to register as a data asset.
        asset_name: Azure ML data asset name to register under, e.g.
            ``"ngt-augmented-train"``.
        validation_asset_name: Azure ML data asset name for validation data.
        test_asset_name: Azure ML data asset name for test data.
        ngt_raw_version: Source raw data version string to store as the
            ``ngt_raw_version`` tag on the registered asset.
    """
    import os  # noqa: PLC0415

    subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    if not subscription_id:
        logger.info("Not in Azure ML job — skipping data asset registration.")
        return

    from azure.ai.ml import MLClient  # noqa: PLC0415
    from azure.ai.ml.constants import AssetTypes  # noqa: PLC0415
    from azure.ai.ml.entities import Data  # noqa: PLC0415
    from azure.identity import ManagedIdentityCredential  # noqa: PLC0415

    ml_client = MLClient(
        credential=ManagedIdentityCredential(),
        subscription_id=subscription_id,
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )

    data_asset = Data(
        name=asset_name,
        path=str(split_dir),
        type=AssetTypes.URI_FOLDER,
        description=(
            f"NGT {split_name} split " f"(source: ngt-raw v{ngt_raw_version})"
        ),
        tags={
            "ngt_raw_version": ngt_raw_version,
            "split": split_name,
        },
    )

    registered = ml_client.data.create_or_update(data_asset)
    logger.info(
        "Registered %s dataset '%s' version %s",
        registered.name,
        registered.version,
    )


@app.command()
def main(
    input_dir: Path = typer.Option(
        ...,
        "--input-dir",
        help="Raw ImageFolder dataset root.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    output_train_dir: Path = typer.Option(
        ...,
        "--output-train-dir",
        help="Output directory for augmented train images.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_val_dir: Path = typer.Option(
        ...,
        "--output-val-dir",
        help="Output directory for validation images.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_test_dir: Path = typer.Option(
        ...,
        "--output-test-dir",
        help="Output directory for test images.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    copies: int = typer.Option(4, "--copies", min=0),
    img_size: int = typer.Option(224, "--img-size", min=1),
    seed: int = typer.Option(42, "--seed"),
    train_ratio: float = typer.Option(0.8, "--train-ratio", min=0.0, max=1.0),
    val_ratio: float = typer.Option(0.1, "--val-ratio", min=0.0, max=1.0),
    augmented_asset_name: str = typer.Option(
        "ngt-augmented-train",
        "--augmented-asset-name",
        help="Azure ML data asset name to register the augmented train split under.",
    ),
    validation_asset_name: str = typer.Option(
        "ngt-val",
        "--validation-asset-name",
        help="Azure ML data asset name to register the validation split under.",
    ),
    test_asset_name: str = typer.Option(
        "ngt-test",
        "--test-asset-name",
        help="Azure ML data asset name to register the test split under.",
    ),
    ngt_raw_version: str = typer.Option(
        "1",
        "--ngt-raw-version",
        help="Version of the ngt-raw source asset used as registration tag.",
    ),
) -> None:
    """Split, augment, and register an ImageFolder dataset.

    Args:
        input_dir: Raw ImageFolder dataset root.
        output_train_dir: Destination for augmented training images.
        output_val_dir: Destination for clean validation images.
        output_test_dir: Destination for clean test images.
        copies: Number of augmented copies per training image.
        img_size: Square augmentation output size in pixels.
        seed: Random seed used for splitting and augmentation.
        train_ratio: Fraction of images assigned to training.
        val_ratio: Fraction of images assigned to validation.
        augmented_asset_name: Azure ML data asset name for augmented training data.
        ngt_raw_version: Raw data asset version stored in registration tags.

    Raises:
        typer.BadParameter: If train and validation ratios leave no test split.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if train_ratio + val_ratio >= 1.0:
        raise typer.BadParameter("train-ratio + val-ratio must be less than 1.")

    raw_train_dir = output_train_dir.parent / "_train_raw_tmp"

    logger.info(
        "Step 1/3 — Stratified split (%.0f/%.0f/%.0f)",
        train_ratio * 100,
        val_ratio * 100,
        (1 - train_ratio - val_ratio) * 100,
    )
    stratified_split(
        input_dir=input_dir,
        train_dir=raw_train_dir,
        val_dir=output_val_dir,
        test_dir=output_test_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    logger.info("Step 2/3 — Augmenting train split (%d copies per image)", copies)

    augment_dir(
        source_dir=raw_train_dir,
        output_dir=output_train_dir,
        copies=copies,
        img_size=img_size,
        seed=seed,
    )

    shutil.rmtree(raw_train_dir, ignore_errors=True)

    logger.info("Step 3/3 — Registering train, validation, and test assets")

    register_split_asset(
        split_dir=output_train_dir,
        asset_name=augmented_asset_name,
        ngt_raw_version=ngt_raw_version,
        split_name="train",
    )

    register_split_asset(
        split_dir=output_val_dir,
        asset_name=validation_asset_name,
        ngt_raw_version=ngt_raw_version,
        split_name="validation",
    )

    register_split_asset(
        split_dir=output_test_dir,
        asset_name=test_asset_name,
        ngt_raw_version=ngt_raw_version,
        split_name="test",
    )


if __name__ == "__main__":
    app()
