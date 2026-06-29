"""Tests for neural network architecture builders."""

import torch

from sign_language.models.architectures import build_efficientnet, build_landmark_mlp


class TestBuildEfficientnet:
    """Tests for the EfficientNet-B0 builder."""

    def test_output_shape(self):
        """Output should match the requested number of classes."""
        num_classes = 26
        model = build_efficientnet(num_classes)
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        assert output.shape == (1, num_classes)

    def test_different_class_counts(self):
        """Should work for any positive number of classes."""
        for n in [5, 10, 26, 50]:
            model = build_efficientnet(n)
            model.eval()
            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(dummy)
            assert output.shape == (1, n)

    def test_model_is_in_eval_after_eval_call(self):
        """After .eval(), model should not be in training mode."""
        model = build_efficientnet(26)
        model.eval()
        assert not model.training


class TestBuildLandmarkMlp:
    """Tests for the Landmark MLP builder."""

    def test_output_shape(self):
        """Output should match the requested number of classes."""
        input_dim = 94
        num_classes = 26
        model = build_landmark_mlp(input_dim, num_classes)
        model.eval()

        dummy = torch.randn(1, input_dim)
        with torch.no_grad():
            output = model(dummy)

        assert output.shape == (1, num_classes)

    def test_different_input_dims(self):
        """Should handle various input dimensions."""
        for dim in [63, 94, 128]:
            model = build_landmark_mlp(dim, 26)
            model.eval()
            dummy = torch.randn(1, dim)
            with torch.no_grad():
                output = model(dummy)
            assert output.shape == (1, 26)

    def test_batch_input(self):
        """Should handle batch sizes larger than 1."""
        model = build_landmark_mlp(94, 26)
        model.eval()
        dummy = torch.randn(8, 94)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (8, 26)
