"""Runtime helpers — seeding and device selection.

``get_device`` replicates the helper that used to live in
``sign_language.core.settings`` so the training package has no
dependency on the inference package.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch to make training reproducible.

    Sets seeds for :mod:`random`, :mod:`numpy`, :mod:`torch`, and all CUDA
    devices. Also disables cuDNN benchmark mode and enables deterministic
    algorithms to ensure reproducible results across runs.

    Args:
        seed: The integer random seed to apply across all libraries.
    """
    logger.info("Setting random seed to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Detect and return the best available torch device.

    Selection priority: CUDA > MPS > CPU.

    Returns:
        A :class:`torch.device` instance for the best available
            compute backend on the current machine.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device
