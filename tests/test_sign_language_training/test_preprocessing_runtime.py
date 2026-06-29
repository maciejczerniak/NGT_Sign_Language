import random

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from sign_language_training.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    create_train_transform,
    create_val_transform,
)
from sign_language_training.runtime import set_seed


def test_create_train_transform_builds_expected_pipeline() -> None:
    """Verify create train transform builds expected pipeline."""
    transform = create_train_transform(img_size=24, augment=True)

    assert isinstance(transform, transforms.Compose)
    assert [type(step) for step in transform.transforms] == [
        transforms.RandomResizedCrop,
        transforms.RandomHorizontalFlip,
        transforms.ColorJitter,
        transforms.GaussianBlur,
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_create_train_transform_returns_tensor_with_target_shape() -> None:
    """Verify create train transform returns tensor with target shape."""
    transform = create_train_transform(img_size=24, augment=True)
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))

    tensor = transform(image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 24, 24)
    assert tensor.dtype == torch.float32


def test_create_val_transform_builds_expected_pipeline() -> None:
    """Verify create val transform builds expected pipeline."""
    transform = create_val_transform(img_size=24)

    assert isinstance(transform, transforms.Compose)
    assert [type(step) for step in transform.transforms] == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]


def test_create_val_transform_normalizes_with_imagenet_statistics() -> None:
    """Verify create val transform normalizes with imagenet statistics."""
    transform = create_val_transform(img_size=16)
    image = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))

    tensor = transform(image)

    expected_channel_values = torch.tensor(
        [(1.0 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        dtype=torch.float32,
    )
    assert tensor.shape == (3, 16, 16)
    assert torch.allclose(tensor[:, 0, 0], expected_channel_values, atol=1e-5)


def test_set_seed_makes_random_generators_reproducible() -> None:
    """Verify set seed makes random generators reproducible."""
    set_seed(123)
    first_python = random.random()
    first_numpy = np.random.rand(3)
    first_torch = torch.rand(3)

    set_seed(123)
    second_python = random.random()
    second_numpy = np.random.rand(3)
    second_torch = torch.rand(3)

    assert first_python == second_python
    assert np.allclose(first_numpy, second_numpy)
    assert torch.allclose(first_torch, second_torch)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
