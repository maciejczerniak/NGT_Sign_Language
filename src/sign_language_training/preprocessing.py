"""Image preprocessing transforms for the training workflow.

ImageNet normalisation constants and the inference-compatible preprocessing
transform are inlined here so the training package has no dependency on
``sign_language.core.image_transforms``.

The values MUST stay aligned with the inference path
(``sign_language.core.image_transforms``) so that training-time
normalisation matches prediction-time normalisation. If those values
ever change in the inference package, mirror the change here.

Augmentation note
-----------------
Online augmentation (RandomCrop, ColorJitter, etc.) has been moved to
the offline preprocessing component (``augmentation.py``). When the
full Azure ML pipeline is used, the training component receives an
already-augmented dataset so ``create_train_transform`` only needs to
resize and normalise.

When running training locally against raw data (no preprocessing step),
pass ``augment=True`` to ``create_train_transform`` to re-enable the
online augmentation fallback.
"""

from __future__ import annotations

import logging

from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_train_transform(
    img_size: int,
    augment: bool = False,
) -> transforms.Compose:
    """Create the training image transform pipeline.

    When ``augment=False`` (default), applies only resize and ImageNet
    normalisation — suitable for the Azure ML pipeline where augmentation
    has already been applied offline by the preprocessing component.

    When ``augment=True``, prepends online augmentation transforms (random
    resized crop, horizontal flip, colour jitter, and Gaussian blur) before
    normalisation — suitable for local training directly against raw data
    without the offline preprocessing step.

    Args:
        img_size: Target image size in pixels applied to both height and
            width.
        augment: If ``True``, online augmentation transforms are included
            in the pipeline. Defaults to ``False`` for Azure ML pipeline usage.

    Returns:
        A :class:`~torchvision.transforms.Compose` pipeline ready for
            use as the training DataLoader transform.
    """
    logger.info(
        "Creating training transforms with image size %d (augment=%s)",
        img_size,
        augment,
    )

    steps: list[object] = []

    if augment:
        steps += [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        ]
    else:
        steps.append(transforms.Resize((img_size, img_size)))

    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return transforms.Compose(steps)


def create_val_transform(img_size: int) -> transforms.Compose:
    """Create the validation and inference-compatible image transform pipeline.

    Applies deterministic resize and ImageNet normalisation only — no
    augmentation. Must match the transform used during inference so that
    validation metrics are comparable to production prediction behaviour.

    Args:
        img_size: Target image size in pixels applied to both height and
            width.

    Returns:
        A :class:`~torchvision.transforms.Compose` pipeline consisting
            of resize, to-tensor, and normalise steps.
    """
    logger.info("Creating validation transforms with image size %d", img_size)

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
