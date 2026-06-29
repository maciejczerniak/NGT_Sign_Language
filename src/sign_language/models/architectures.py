"""Neural network architecture definitions.

Contains the EfficientNet-B0 image classifier and the Landmark MLP
used as a fallback when image confidence is low.
"""

import torch.nn as nn
from torchvision import models


def build_efficientnet(num_classes: int) -> nn.Module:
    """Build an EfficientNet-B0 model with a custom classification head.

    Loads EfficientNet-B0 without pretrained weights and replaces the
    default classifier with a dropout + linear head sized to ``num_classes``.

    :param num_classes: Number of output classes (sign language letters).
    :returns: An :class:`~torch.nn.Module` EfficientNet-B0 with the replaced
        classifier head.
    """
    net = models.efficientnet_b0(weights=None)
    in_features = net.classifier[1].in_features
    net.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return net  # type: ignore[no-any-return]


def build_landmark_mlp(input_dim: int, num_classes: int) -> nn.Module:
    """Build a multi-layer perceptron for hand landmark feature classification.

    Architecture: Linear(256) → BN → ReLU → Dropout(0.3) →
    Linear(128) → BN → ReLU → Dropout(0.2) → Linear(64) → ReLU →
    Linear(num_classes).

    :param input_dim: Dimensionality of the input landmark feature vector,
        must match the output of
        :func:`~sign_language.features.landmarks.extract_landmark_features`.
    :param num_classes: Number of output classes.
    :returns: A :class:`~torch.nn.Sequential` MLP model.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )
