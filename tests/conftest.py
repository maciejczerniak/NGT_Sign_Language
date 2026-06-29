"""
Shared test fixtures.

Provides reusable fake data (landmarks, predictions) so individual
test files don't have to build them from scratch, as well as shared
fixtures for the API test suite.
"""

import base64
import io
import sys
import types
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

from sign_language.models.loader import LoadedModels


def _install_torchvision_stub() -> None:
    """Install lightweight torchvision stand-ins when torchvision cannot import."""
    torchvision_module = types.ModuleType("torchvision")
    datasets_module = types.ModuleType("torchvision.datasets")
    models_module = types.ModuleType("torchvision.models")
    transforms_module = types.ModuleType("torchvision.transforms")

    class StubEfficientNet(nn.Module):
        def __init__(self) -> None:
            """Create a minimal EfficientNet-like model for loader tests."""
            super().__init__()
            self.features = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv2d(3, 3, kernel_size=1), nn.ReLU())
                    for _ in range(9)
                ]
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(3, 1000),
            )

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """Run the stub feature extractor and classifier."""
            outputs = inputs
            for block in self.features:
                outputs = block(outputs)
            outputs = self.avgpool(outputs).flatten(1)
            return self.classifier(outputs)

    class Compose:
        def __init__(self, transforms: list[object]) -> None:
            """Store transforms for sequential application."""
            self.transforms = transforms

        def __call__(self, image):
            """Apply each stored transform to an image."""
            for transform in self.transforms:
                image = transform(image)
            return image

    class RandomResizedCrop:
        def __init__(self, size: int, scale: tuple[float, float]) -> None:
            """Store crop size and scale for the resize-only stub."""
            self.size = size
            self.scale = scale

        def __call__(self, image: Image.Image) -> Image.Image:
            """Resize the image to the configured square size."""
            return image.resize((self.size, self.size))

    class RandomHorizontalFlip:
        def __init__(self, p: float = 0.5) -> None:
            """Store the flip probability without changing image behavior."""
            self.p = p

        def __call__(self, image: Image.Image) -> Image.Image:
            """Return the image unchanged for deterministic tests."""
            return image

    class ColorJitter:
        def __init__(self, **kwargs: float) -> None:
            """Store color jitter arguments without applying augmentation."""
            self.kwargs = kwargs

        def __call__(self, image: Image.Image) -> Image.Image:
            """Return the image unchanged for deterministic tests."""
            return image

    class GaussianBlur:
        def __init__(self, kernel_size: int, sigma: tuple[float, float]) -> None:
            """Store blur arguments without applying augmentation."""
            self.kernel_size = kernel_size
            self.sigma = sigma

        def __call__(self, image: Image.Image) -> Image.Image:
            """Return the image unchanged for deterministic tests."""
            return image

    class Resize:
        def __init__(self, size: tuple[int, int]) -> None:
            """Store the output size for the resize stub."""
            self.size = size

        def __call__(self, image: Image.Image) -> Image.Image:
            """Resize the image to the configured dimensions."""
            return image.resize(self.size)

    class ToTensor:
        def __call__(self, image: Image.Image) -> torch.Tensor:
            """Convert a PIL image into a channel-first float tensor."""
            array = np.asarray(image, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = array[:, :, None]
            return torch.from_numpy(array).permute(2, 0, 1)

    class Normalize:
        def __init__(self, mean: list[float], std: list[float]) -> None:
            """Store channel-wise mean and standard deviation tensors."""
            self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
            self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            """Normalize a tensor using the stored channel statistics."""
            return (tensor - self.mean) / self.std

    class ImageFolder:
        def __init__(self, root: str, transform=None) -> None:
            """Create a minimal ImageFolder-compatible dataset shell."""
            self.root = root
            self.transform = transform
            self.classes: list[str] = []
            self.targets: list[int] = []

        def __len__(self) -> int:
            """Return the number of target labels in the stub dataset."""
            return len(self.targets)

    transforms_module.Compose = Compose
    transforms_module.RandomResizedCrop = RandomResizedCrop
    transforms_module.RandomHorizontalFlip = RandomHorizontalFlip
    transforms_module.ColorJitter = ColorJitter
    transforms_module.GaussianBlur = GaussianBlur
    transforms_module.Resize = Resize
    transforms_module.ToTensor = ToTensor
    transforms_module.Normalize = Normalize

    datasets_module.ImageFolder = ImageFolder
    models_module.efficientnet_b0 = lambda weights=None: StubEfficientNet()

    torchvision_module.datasets = datasets_module
    torchvision_module.models = models_module
    torchvision_module.transforms = transforms_module

    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.datasets"] = datasets_module
    sys.modules["torchvision.models"] = models_module
    sys.modules["torchvision.transforms"] = transforms_module


try:
    import torchvision  # noqa: F401
except Exception:
    _install_torchvision_stub()


# ---------------------------------------------------------------------------
# Landmark fixtures (preprocessing / feature tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_landmarks() -> list[dict]:
    """21 fake hand landmark points with realistic-ish spread.

    All points are generated deterministically from a seeded random
    number generator so tests get stable, non-zero landmark values.
    """
    rng = np.random.RandomState(42)
    landmarks = []
    for _ in range(21):
        landmarks.append(
            {
                "x": float(rng.uniform(0.3, 0.7)),
                "y": float(rng.uniform(0.3, 0.7)),
                "z": float(rng.uniform(-0.05, 0.05)),
            }
        )
    return landmarks


@pytest.fixture
def zero_landmarks() -> list[dict]:
    """21 landmarks all at the origin — edge case for normalisation."""
    return [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]


# ---------------------------------------------------------------------------
# API fixtures (routes / app / state tests)
# ---------------------------------------------------------------------------


def _make_png_b64(width: int = 8, height: int = 8) -> str:
    """Create a tiny solid-colour PNG and return it as a base64 string."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(120, 80, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def dummy_image_b64() -> str:
    """A valid base64-encoded PNG the preprocessing pipeline can decode."""
    return _make_png_b64()


@pytest.fixture
def dummy_image_b64_dataurl() -> str:
    """Same image wrapped in a data-URL prefix."""
    return f"data:image/png;base64,{_make_png_b64()}"


@pytest.fixture
def mock_models() -> MagicMock:
    """
    Minimal stand-in for LoadedModels.

    Uses MagicMock so attribute access never raises; individual tests
    that care about specific values override them directly.
    """
    m = MagicMock(spec=LoadedModels)
    m.device = torch.device("cpu")
    m.class_names = ["A", "B", "C"]
    m.lm_class_names = ["A", "B", "C"]
    m.landmark_model = None
    m.hands_detector = None
    m.model = MagicMock()
    return m
