import numpy as np
import pytest
import torch

from sign_language_training import data_loading
from sign_language_training.configuration import TrainingConfig
from sign_language_training.data_loading import (
    create_dataloaders,
    create_stratified_split,
    load_dataset,
)


class FakeImageFolder:
    instances: list["FakeImageFolder"] = []

    def __init__(self, root: str, transform=None) -> None:
        """Create a fake image folder with deterministic classes and targets."""
        self.root = root
        self.transform = transform
        self.classes = ["A", "B"]
        self.targets = [0, 0, 1, 1]
        self.__class__.instances.append(self)

    def __len__(self) -> int:
        """Return the number of fake dataset samples."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return a fake tensor sample and its target label."""
        image = torch.tensor([float(index)], dtype=torch.float32)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def test_load_dataset_returns_image_folder_metadata(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify load dataset returns image folder metadata."""
    monkeypatch.setattr(data_loading.datasets, "ImageFolder", FakeImageFolder)

    dataset, class_names, targets = load_dataset(tmp_path)

    assert isinstance(dataset, FakeImageFolder)
    assert dataset.root == str(tmp_path)
    assert class_names == ["A", "B"]
    assert np.array_equal(targets, np.array([0, 0, 1, 1]))


def test_create_stratified_split_rejects_multiple_splits() -> None:
    """Verify create stratified split rejects multiple splits."""
    targets = np.array([0, 0, 1, 1], dtype=np.int_)

    with pytest.raises(
        ValueError,
        match="Only n_splits=1 is supported by the current workflow.",
    ):
        create_stratified_split(
            targets=targets,
            n_splits=2,
            split_ratio=0.5,
            seed=42,
        )


def test_create_stratified_split_returns_reproducible_indices() -> None:
    """Verify create stratified split returns reproducible indices."""
    targets = np.array([0, 0, 0, 1, 1, 1], dtype=np.int_)

    first_train, first_val = create_stratified_split(
        targets=targets,
        n_splits=1,
        split_ratio=0.33,
        seed=42,
    )
    second_train, second_val = create_stratified_split(
        targets=targets,
        n_splits=1,
        split_ratio=0.33,
        seed=42,
    )

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_val, second_val)
    assert len(first_train) == 4
    assert len(first_val) == 2
    assert set(targets[first_val]) == {0, 1}


def test_create_dataloaders_builds_train_and_validation_loaders(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify create dataloaders builds train and validation loaders."""
    FakeImageFolder.instances = []
    monkeypatch.setattr(data_loading.datasets, "ImageFolder", FakeImageFolder)
    config = TrainingConfig(batch_size=2, num_workers=0, pin_memory=False)

    train_loader, val_loader = create_dataloaders(
        data_dir=tmp_path,
        train_idx=np.array([0, 2], dtype=np.int_),
        val_idx=np.array([1, 3], dtype=np.int_),
        train_transform=lambda tensor: tensor + 10,
        val_transform=lambda tensor: tensor + 20,
        config=config,
    )

    assert len(FakeImageFolder.instances) == 2
    assert all(instance.root == str(tmp_path) for instance in FakeImageFolder.instances)
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 2
    assert len(train_loader) == 1
    assert len(val_loader) == 1

    val_images, val_labels = next(iter(val_loader))
    assert torch.equal(val_images.flatten(), torch.tensor([21.0, 23.0]))
    assert torch.equal(val_labels, torch.tensor([0, 1]))
