"""EfficientNet-B0 model construction for NGT fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> tuple[int, int, int]:
    """Count the total, trainable, and frozen parameters in a model.

    Args:
        model: The :class:`~torch.nn.Module` to inspect.

    Returns:
        A three-tuple of ``(total, trainable, frozen)`` parameter counts.
    """
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen = total - trainable
    return total, trainable, frozen


def build_model_from_pretrained(
    pretrained_checkpoint_path: str | Path,
    device: torch.device,
    num_ngt_classes: int,
) -> nn.Module:
    """Load a pretrained checkpoint and adapt EfficientNet-B0 for NGT fine-tuning.

    Loads the checkpoint state dict into an EfficientNet-B0 backbone, then
    replaces the classifier head if the checkpoint class count differs from
    ``num_ngt_classes``. After loading, freezes all parameters except the
    last two feature blocks (indices 6 and 7), the final batch norm layer
    (index 8), and the classifier head.

    Args:
        pretrained_checkpoint_path: Path to the ``.pth`` checkpoint file.
            The checkpoint may be a raw state dict or a dict containing a
            ``"model_state"`` key.
        device: Torch device to map the model onto.
        num_ngt_classes: Number of NGT output classes for the new
            classifier head.

    Returns:
        The adapted EfficientNet-B0 :class:`~torch.nn.Module` on
            ``device``, with only the specified layers set to
            ``requires_grad=True``.

    Raises:
        TypeError: If ``model.classifier[1]`` is not a
            :class:`~torch.nn.Linear` layer.
        KeyError: If the checkpoint state dict does not contain
            ``"classifier.1.weight"``.
    """
    logger.info("Building EfficientNet-B0 for NGT fine-tuning")

    model = models.efficientnet_b0(weights=None)
    classifier_layer = model.classifier[1]
    if not isinstance(classifier_layer, nn.Linear):
        raise TypeError("Expected EfficientNet classifier[1] to be nn.Linear.")

    in_features = classifier_layer.in_features

    checkpoint: Any = torch.load(
        str(pretrained_checkpoint_path),
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        logger.info(
            "Loaded pretrained checkpoint: %s",
            Path(pretrained_checkpoint_path).name,
        )
        logger.info(
            "Pretrained checkpoint metadata: epoch=%s, val_acc=%s",
            checkpoint.get("epoch", "?"),
            checkpoint.get("val_acc", "?"),
        )

    classifier_weight = state_dict.get("classifier.1.weight")
    if classifier_weight is None:
        raise KeyError("Pretrained checkpoint is missing classifier.1.weight.")

    checkpoint_class_count = int(classifier_weight.shape[0])

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, checkpoint_class_count),
    )

    model.load_state_dict(state_dict)

    if checkpoint_class_count != num_ngt_classes:
        logger.info(
            "Replacing classifier head: %d -> %d classes",
            checkpoint_class_count,
            num_ngt_classes,
        )
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_ngt_classes),
        )
    else:
        logger.info(
            "Checkpoint already has %d classes — keeping classifier head",
            num_ngt_classes,
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block_idx in (6, 7):
        for parameter in model.features[block_idx].parameters():
            parameter.requires_grad = True

    for parameter in model.features[8].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    total, trainable, frozen = count_parameters(model)
    logger.info(
        "Model parameters: total=%d, trainable=%d, frozen=%d",
        total,
        trainable,
        frozen,
    )

    return cast(nn.Module, model.to(device))
