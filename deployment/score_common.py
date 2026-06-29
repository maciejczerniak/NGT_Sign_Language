"""Shared Azure ML scoring helpers for online endpoints."""

from __future__ import annotations

import base64
import binascii
import io
import os
import sys
from os import PathLike
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.model_definitions import (  # noqa: E402
    build_model_from_pretrained,
)
from sign_language_training.preprocessing import create_val_transform  # noqa: E402

MODEL_NAME_ENV = "AZUREML_MODEL_NAME"
MODEL_VERSION_ENV = "AZUREML_MODEL_VERSION"
DEFAULT_MODEL_NAME = "ngt-sign-language"


def run_image_inference(
    tensor: torch.Tensor,
    model: nn.Module,
    class_names: list[str],
) -> tuple[str, float, list[dict[str, float | str]]]:
    """Run EfficientNet inference and return the strongest predictions.

    Args:
        tensor: Preprocessed model input tensor.
        model: Loaded PyTorch classification model.
        class_names: Class names ordered by model output index.

    Returns:
        Predicted letter, confidence, and top-three prediction records.
    """
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top3_vals, top3_idxs = torch.topk(probs, k=min(3, len(class_names)))
    predicted_letter = class_names[int(top3_idxs[0].item())]
    confidence = float(top3_vals[0].item())
    top_3: list[dict[str, float | str]] = [
        {"letter": class_names[int(i.item())], "confidence": float(v.item())}
        for i, v in zip(top3_idxs, top3_vals)
    ]
    return predicted_letter, confidence, top_3


def model_metadata() -> dict[str, str]:
    """Return model metadata exposed in prediction responses.

    Returns:
        Model name and version read from deployment environment variables.
    """
    return {
        "model_name": os.getenv(MODEL_NAME_ENV, DEFAULT_MODEL_NAME),
        "model_version": os.getenv(MODEL_VERSION_ENV, ""),
    }


def find_model_file(model_dir: str | Path | None = None) -> Path:
    """Find the first PyTorch checkpoint in the Azure ML model directory.

    Args:
        model_dir: Explicit model file or directory. Defaults to
            ``AZUREML_MODEL_DIR``.

    Returns:
        Resolved checkpoint path.

    Raises:
        FileNotFoundError: If no ``.pth`` checkpoint exists under the model path.
    """
    model_root: str | Path | PathLike[str] = (
        model_dir if model_dir is not None else os.getenv("AZUREML_MODEL_DIR", ".")
    )
    root = Path(model_root).resolve()
    if root.is_file() and root.suffix == ".pth":
        return root

    candidates = sorted(root.rglob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth model file found under {root}")
    return candidates[0]


def load_model(model_path: str | Path, device: torch.device) -> tuple[Any, list[str]]:
    """Load the EfficientNet checkpoint used by the project.

    Args:
        model_path: Path to the registered model checkpoint.
        device: PyTorch device used for model weights and inference.

    Returns:
        Loaded evaluation model and ordered class names.

    Raises:
        KeyError: If the checkpoint does not contain ``class_names``.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = list(checkpoint["class_names"])
    model = build_model_from_pretrained(
        pretrained_checkpoint_path=model_path,
        device=device,
        num_ngt_classes=len(class_names),
    )
    model.eval()
    return model, class_names


def decode_base64_image(image_data: str) -> Image.Image:
    """Decode a raw base64 string or data URL into an RGB PIL image.

    Args:
        image_data: Base64 image content or a base64 data URL.

    Returns:
        Decoded RGB image.

    Raises:
        ValueError: If the input cannot be decoded as an image.
    """
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (binascii.Error, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Failed to decode base64 image: {exc}") from exc


class ScoringRuntime:
    """Small runtime wrapper around the loaded model and transform."""

    def __init__(
        self, model: Any, class_names: list[str], device: torch.device
    ) -> None:
        """Initialize a reusable scoring runtime.

        Args:
            model: Loaded PyTorch classification model.
            class_names: Class names ordered by model output index.
            device: Device used for inference.
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.transform = create_val_transform(224)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path | None = None) -> "ScoringRuntime":
        """Create a scoring runtime from an Azure ML model directory.

        Args:
            model_dir: Explicit model file or directory. Defaults to
                ``AZUREML_MODEL_DIR``.

        Returns:
            Initialized scoring runtime.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = find_model_file(model_dir)
        model, class_names = load_model(model_path, device)
        return cls(model=model, class_names=class_names, device=device)

    def predict_image(self, image: Image.Image) -> dict[str, Any]:
        """Run prediction on a PIL image.

        Args:
            image: RGB image to classify.

        Returns:
            Prediction, confidence, top-three classes, and model metadata.
        """
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        letter, confidence, top_3 = run_image_inference(
            tensor=tensor,
            model=self.model,
            class_names=self.class_names,
        )
        return {
            "predicted_letter": letter,
            "confidence": float(confidence),
            "top_3": [
                {
                    "letter": str(item["letter"]),
                    "confidence": float(item["confidence"]),
                }
                for item in top_3
            ],
            **model_metadata(),
        }

    def predict_base64(self, image_data: str) -> dict[str, Any]:
        """Decode and predict one base64 image.

        Args:
            image_data: Base64 image content or a base64 data URL.

        Returns:
            Prediction, confidence, top-three classes, and model metadata.
        """
        return self.predict_image(decode_base64_image(image_data))
