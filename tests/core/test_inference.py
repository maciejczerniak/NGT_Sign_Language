"""Tests for the inference engine."""

import torch
from unittest.mock import patch

from sign_language.models.architectures import build_efficientnet, build_landmark_mlp
from sign_language.core.inference import run_inference


class TestRunInference:
    """Tests for run_inference with real (untrained) models."""

    NUM_CLASSES = 5
    CLASS_NAMES = ["A", "B", "C", "D", "E"]
    DEVICE = torch.device("cpu")

    def _make_efficientnet(self) -> torch.nn.Module:
        """Create an evaluation-mode EfficientNet for inference tests."""
        model = build_efficientnet(self.NUM_CLASSES)
        model.eval()
        return model

    def _make_landmark_mlp(self, input_dim: int = 94) -> torch.nn.Module:
        """Create an evaluation-mode landmark MLP for inference tests."""
        model = build_landmark_mlp(input_dim, self.NUM_CLASSES)
        model.eval()
        return model

    def _make_dummy_tensor(self) -> torch.Tensor:
        """Create a dummy image tensor with the expected model input shape."""
        return torch.randn(1, 3, 224, 224)

    def test_returns_three_values(self):
        """Should return (letter, confidence, top_3)."""
        model = self._make_efficientnet()
        tensor = self._make_dummy_tensor()

        result = run_inference(tensor, model, self.CLASS_NAMES, self.DEVICE)

        assert len(result) == 3

    def test_predicted_letter_is_valid(self):
        """Predicted letter should be one of the class names."""
        model = self._make_efficientnet()
        tensor = self._make_dummy_tensor()

        letter, _, _ = run_inference(tensor, model, self.CLASS_NAMES, self.DEVICE)

        assert letter in self.CLASS_NAMES

    def test_confidence_between_zero_and_one(self):
        """Confidence should be a probability."""
        model = self._make_efficientnet()
        tensor = self._make_dummy_tensor()

        _, confidence, _ = run_inference(tensor, model, self.CLASS_NAMES, self.DEVICE)

        assert 0.0 <= confidence <= 1.0

    def test_top_3_structure(self):
        """Top 3 should be a list of dicts with 'letter' and 'confidence'."""
        model = self._make_efficientnet()
        tensor = self._make_dummy_tensor()

        _, _, top_3 = run_inference(tensor, model, self.CLASS_NAMES, self.DEVICE)

        assert isinstance(top_3, list)
        assert len(top_3) <= 3
        for entry in top_3:
            assert "letter" in entry
            assert "confidence" in entry
            assert entry["letter"] in self.CLASS_NAMES
            assert 0.0 <= entry["confidence"] <= 1.0

    def test_top_3_sorted_descending(self):
        """Top 3 confidences should be in descending order."""
        model = self._make_efficientnet()
        tensor = self._make_dummy_tensor()

        _, _, top_3 = run_inference(tensor, model, self.CLASS_NAMES, self.DEVICE)

        confidences = [e["confidence"] for e in top_3]
        assert confidences == sorted(confidences, reverse=True)

    def test_works_without_landmark_model(self):
        """Should work fine when no landmark model is provided."""
        model = self._make_efficientnet()
        tensor = self._make_dummy_tensor()

        letter, confidence, top_3 = run_inference(
            tensor,
            model,
            self.CLASS_NAMES,
            self.DEVICE,
            landmarks_data=None,
            landmark_model=None,
            lm_class_names=None,
        )

        assert letter in self.CLASS_NAMES
        assert len(top_3) > 0

    def test_works_with_landmark_model(self):
        """Should not crash when landmark model is provided."""
        model = self._make_efficientnet()
        lm_model = self._make_landmark_mlp()
        tensor = self._make_dummy_tensor()

        # Fake landmarks (21 points)
        landmarks = [
            {"x": float(i) * 0.05, "y": float(i) * 0.03, "z": 0.0} for i in range(21)
        ]

        letter, confidence, top_3 = run_inference(
            tensor,
            model,
            self.CLASS_NAMES,
            self.DEVICE,
            landmarks_data=landmarks,
            landmark_model=lm_model,
            lm_class_names=self.CLASS_NAMES,
        )

        assert letter in self.CLASS_NAMES
        assert 0.0 <= confidence <= 1.0

    def test_fewer_classes_than_three(self):
        """Should handle models with fewer than 3 classes."""
        class_names = ["A", "B"]
        model = build_efficientnet(2)
        model.eval()
        tensor = self._make_dummy_tensor()

        _, _, top_3 = run_inference(tensor, model, class_names, self.DEVICE)

        assert len(top_3) <= 2

    def test_landmark_model_overrides_low_confidence_image_model(self):
        """A strong landmark prediction should replace a weak image prediction."""

        class FixedModel(torch.nn.Module):
            def __init__(self, logits):
                super().__init__()
                self.logits = torch.tensor([logits], dtype=torch.float32)

            def forward(self, tensor):
                return self.logits

        image_model = FixedModel([0.0, 0.0])
        landmark_model = FixedModel([0.0, 10.0])

        with (
            patch(
                "sign_language.core.inference.extract_landmark_features",
                return_value=torch.zeros(1, 2),
            ),
            patch(
                "sign_language.core.inference.settings.efficientnet_confidence_threshold",
                0.9,
            ),
            patch(
                "sign_language.core.inference.settings.landmark_override_threshold",
                0.8,
            ),
        ):
            letter, confidence, top_3 = run_inference(
                torch.zeros(1, 1),
                image_model,
                ["A", "B"],
                self.DEVICE,
                landmarks_data=[{"x": 0, "y": 0, "z": 0}],
                landmark_model=landmark_model,
                lm_class_names=["A", "B"],
            )

        assert letter == "B"
        assert confidence > 0.99
        assert top_3[0]["letter"] == "B"

    def test_landmark_failure_falls_back_to_image_prediction(self):
        """Landmark extraction failures should not fail image inference."""

        class FixedModel(torch.nn.Module):
            def forward(self, tensor):
                return torch.tensor([[2.0, 1.0]])

        with (
            patch(
                "sign_language.core.inference.extract_landmark_features",
                side_effect=ValueError("invalid landmarks"),
            ),
            patch(
                "sign_language.core.inference.settings.efficientnet_confidence_threshold",
                0.99,
            ),
        ):
            letter, _, top_3 = run_inference(
                torch.zeros(1, 1),
                FixedModel(),
                ["A", "B"],
                self.DEVICE,
                landmarks_data=[{"x": 0, "y": 0, "z": 0}],
                landmark_model=FixedModel(),
                lm_class_names=["A", "B"],
            )

        assert letter == "A"
        assert top_3[0]["letter"] == "A"
