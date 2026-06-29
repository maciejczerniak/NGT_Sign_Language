"""Inference engine.

Runs EfficientNet-B0 as the primary classifier and falls back to the
Landmark MLP when EfficientNet confidence drops below a threshold.

The confidence thresholds are controlled by
``efficientnet_confidence_threshold`` and ``landmark_override_threshold``
in project settings.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from sign_language.features.landmarks import extract_landmark_features
from sign_language.core.settings import settings

logger = logging.getLogger(__name__)


def run_inference(
    tensor: torch.Tensor,
    model: nn.Module,
    class_names: list[str],
    device: torch.device,
    landmarks_data: Optional[list[dict]] = None,
    landmark_model: Optional[nn.Module] = None,
    lm_class_names: Optional[list[str]] = None,
) -> tuple[str, float, list[dict]]:
    """Run inference on a preprocessed image tensor.

    Uses EfficientNet-B0 as the primary model. When its top-1 confidence
    falls below ``settings.efficientnet_confidence_threshold`` and landmark
    data is available, the Landmark MLP is consulted. If the MLP confidence
    exceeds ``settings.landmark_override_threshold``, its prediction and
    top-3 list replace the image model's output.

    :param tensor: Preprocessed image tensor of shape ``(1, 3, 224, 224)``
        already on the correct device.
    :param model: Loaded EfficientNet-B0 model in eval mode.
    :param class_names: Ordered list of class label strings matching the
        EfficientNet output head.
    :param device: Torch device used to move the landmark feature tensor.
    :param landmarks_data: Optional list of 21 landmark dicts with ``x``,
        ``y``, and ``z`` keys, as returned by MediaPipe. Required for
        landmark MLP fallback.
    :param landmark_model: Optional loaded Landmark MLP model in eval mode.
        If ``None``, the fallback is skipped entirely.
    :param lm_class_names: Ordered list of class label strings matching the
        MLP output head. Required for landmark MLP fallback.
    :returns: A three-tuple of ``(predicted_letter, confidence, top_3)``
        where ``predicted_letter`` is the top-1 class label, ``confidence``
        is its score in [0, 1], and ``top_3`` is a list of up to three dicts
        each containing ``letter`` and ``confidence`` keys.
    """
    with torch.no_grad():
        img_probs = torch.softmax(model(tensor), dim=1)[0]

    top3_vals, top3_idxs = torch.topk(img_probs, k=min(3, len(class_names)))
    predicted_letter = class_names[int(top3_idxs[0].item())]
    confidence = top3_vals[0].item()

    if (
        landmark_model is not None
        and landmarks_data is not None
        and lm_class_names
        and confidence < settings.efficientnet_confidence_threshold
    ):
        try:
            lm_tensor = extract_landmark_features(landmarks_data).to(device)
            with torch.no_grad():
                lm_probs = torch.softmax(landmark_model(lm_tensor), dim=1)[0]

            lm_top1 = lm_class_names[int(lm_probs.argmax().item())]
            lm_conf = lm_probs.max().item()

            if lm_conf > settings.landmark_override_threshold:
                logger.debug(
                    "Landmark MLP override: %s (%.2f) over %s (%.2f)",
                    lm_top1,
                    lm_conf,
                    predicted_letter,
                    confidence,
                )
                predicted_letter = lm_top1
                confidence = lm_conf
                lm_top_k = min(3, len(lm_class_names))
                lm_top3_vals, lm_top3_idxs = torch.topk(lm_probs, k=lm_top_k)
                top_3 = [
                    {
                        "letter": lm_class_names[int(i.item())],
                        "confidence": v.item(),
                    }
                    for i, v in zip(lm_top3_idxs, lm_top3_vals)
                ]
                return predicted_letter, confidence, top_3
        except Exception as exc:
            logger.error("Landmark inference error: %s", exc)

    top_3 = [
        {"letter": class_names[int(i.item())], "confidence": v.item()}
        for i, v in zip(top3_idxs, top3_vals)
    ]
    return predicted_letter, confidence, top_3
