from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sign_language_training import model_definitions


class DummyEfficientNet(nn.Module):
    def __init__(self, use_linear_classifier: bool = True) -> None:
        """Create a minimal EfficientNet-like model for definition tests."""
        super().__init__()
        self.features = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.ReLU()) for _ in range(9)]
        )
        classifier_tail: nn.Module
        if use_linear_classifier:
            classifier_tail = nn.Linear(4, 5)
        else:
            classifier_tail = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), classifier_tail
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inputs through the dummy classifier."""
        return self.classifier(inputs)


def create_pretrained_checkpoint_state(
    num_base_classes: int,
) -> dict[str, torch.Tensor]:
    """Create a checkpoint state with the requested pretrained classifier size."""
    checkpoint_model = DummyEfficientNet()
    checkpoint_model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(4, num_base_classes),
    )
    return checkpoint_model.state_dict()


def test_count_parameters_returns_total_trainable_and_frozen_counts() -> None:
    """Verify count parameters returns total trainable and frozen counts."""
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad = False

    total, trainable, frozen = model_definitions.count_parameters(model)

    assert total == 13
    assert trainable == 9
    assert frozen == 4


def test_build_model_from_checkpoint_requires_linear_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build model from pretrained replaces classifier and unfreezes expected blocks."""
    monkeypatch.setattr(
        model_definitions.models,
        "efficientnet_b0",
        lambda weights=None: DummyEfficientNet(use_linear_classifier=False),
    )
    monkeypatch.setattr(
        model_definitions.torch,
        "load",
        lambda *args, **kwargs: {
            "model_state": create_pretrained_checkpoint_state(num_base_classes=29),
            "epoch": 3,
            "val_acc": 0.9,
        },
    )

    with pytest.raises(
        TypeError,
        match="Expected EfficientNet classifier\\[1\\] to be nn.Linear.",
    ):
        model_definitions.build_model_from_pretrained(
            pretrained_checkpoint_path=Path("pretrained_checkpoint.pth"),
            device=torch.device("cpu"),
            num_ngt_classes=22,
        )


def test_build_model_from_checkpoint_adapts_classifier_and_freezes_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_definitions.models,
        "efficientnet_b0",
        lambda weights=None: DummyEfficientNet(use_linear_classifier=True),
    )
    monkeypatch.setattr(
        model_definitions.torch,
        "load",
        lambda *args, **kwargs: {
            "model_state": create_pretrained_checkpoint_state(num_base_classes=22),
            "epoch": 3,
            "val_acc": 0.9,
        },
    )

    model = model_definitions.build_model_from_pretrained(
        pretrained_checkpoint_path=Path("pretrained_checkpoint.pth"),
        device=torch.device("cpu"),
        num_ngt_classes=22,
    )

    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].out_features == 22
    assert all(
        not parameter.requires_grad for parameter in model.features[0].parameters()
    )
    assert all(parameter.requires_grad for parameter in model.features[6].parameters())
    assert all(parameter.requires_grad for parameter in model.features[7].parameters())
    assert all(parameter.requires_grad for parameter in model.features[8].parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
