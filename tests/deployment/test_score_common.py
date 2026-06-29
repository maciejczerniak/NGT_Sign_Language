"""Tests for Azure ML scoring helpers."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image
import pytest
import torch
from torchvision import models

from deployment.score_common import (
    ScoringRuntime,
    decode_base64_image,
    find_model_file,
)
from sign_language_training.model_definitions import build_model_from_pretrained


def _image_b64() -> str:
    image = Image.new("RGB", (16, 16), color=(120, 80, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_decode_base64_image_accepts_raw_and_data_url() -> None:
    raw = _image_b64()
    assert decode_base64_image(raw).mode == "RGB"
    assert decode_base64_image(f"data:image/png;base64,{raw}").mode == "RGB"


def test_find_model_file_returns_nested_pth(tmp_path: Path) -> None:
    nested = tmp_path / "model" / "outputs"
    nested.mkdir(parents=True)
    expected = nested / "model.pth"
    expected.write_bytes(b"fake")

    assert find_model_file(tmp_path) == expected


def test_scoring_runtime_predicts_image(tmp_path: Path) -> None:
    class_names = ["A", "B", "C"]
    seed_checkpoint_path = tmp_path / "seed_model.pth"
    torch.save(
        {"model_state": _build_seed_state(len(class_names))},
        seed_checkpoint_path,
    )
    model = build_model_from_pretrained(
        pretrained_checkpoint_path=seed_checkpoint_path,
        device=torch.device("cpu"),
        num_ngt_classes=len(class_names),
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {
            "class_names": class_names,
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )

    runtime = ScoringRuntime.from_model_dir(tmp_path)
    result = runtime.predict_base64(_image_b64())

    assert result["predicted_letter"] in class_names
    assert result["confidence"] >= 0
    assert len(result["top_3"]) == 3


def test_decode_base64_image_rejects_invalid_data() -> None:
    with pytest.raises(ValueError, match="Failed to decode"):
        decode_base64_image("not-base64")


def _build_seed_state(num_classes: int) -> dict[str, torch.Tensor]:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features, num_classes),
    )
    return model.state_dict()
