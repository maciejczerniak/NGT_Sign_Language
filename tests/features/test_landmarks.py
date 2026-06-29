"""Tests for landmark feature extraction."""

import math

import numpy as np
import torch

from sign_language.features.landmarks import (
    JOINT_TRIPLETS,
    FINGERTIP_PAIRS,
    normalize_landmarks,
    angle_between,
    extract_landmark_features,
)


class TestNormalizeLandmarks:
    """Tests for the normalize_landmarks function."""

    def test_wrist_becomes_origin(self, fake_landmarks):
        """After normalisation, point 0 (wrist) should be at the origin."""
        pts = normalize_landmarks(fake_landmarks)
        assert pts.shape == (21, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_output_shape(self, fake_landmarks):
        """Should return a (21, 3) float32 array."""
        pts = normalize_landmarks(fake_landmarks)
        assert pts.shape == (21, 3)
        assert pts.dtype == np.float32

    def test_scale_normalisation(self, fake_landmarks):
        """Point 9 (middle MCP) should have unit distance from origin after scaling."""
        pts = normalize_landmarks(fake_landmarks)
        dist_to_9 = np.linalg.norm(pts[9])
        assert abs(dist_to_9 - 1.0) < 1e-5

    def test_zero_landmarks_no_crash(self, zero_landmarks):
        """All-zero landmarks should not raise (scale guard handles it)."""
        pts = normalize_landmarks(zero_landmarks)
        assert pts.shape == (21, 3)


class TestAngleBetween:
    """Tests for the angle_between function."""

    def test_parallel_vectors(self):
        """Parallel vectors should have angle 0."""
        v = np.array([1.0, 0.0, 0.0])
        assert abs(angle_between(v, v)) < 1e-6

    def test_perpendicular_vectors(self):
        """Perpendicular vectors should have angle π/2."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert abs(angle_between(v1, v2) - math.pi / 2) < 1e-6

    def test_opposite_vectors(self):
        """Opposite vectors should have angle π."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert abs(angle_between(v1, v2) - math.pi) < 1e-6

    def test_zero_vector_returns_zero(self):
        """A zero-length vector should return 0.0 (not crash)."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert angle_between(v1, v2) == 0.0


class TestExtractLandmarkFeatures:
    """Tests for the full feature extraction pipeline."""

    def test_output_is_tensor(self, fake_landmarks):
        """Should return a torch Tensor."""
        result = extract_landmark_features(fake_landmarks)
        assert isinstance(result, torch.Tensor)

    def test_output_shape(self, fake_landmarks):
        """Should return shape (1, feature_dim)."""
        result = extract_landmark_features(fake_landmarks)
        assert result.dim() == 2
        assert result.shape[0] == 1

        expected_dim = 21 * 3 + len(JOINT_TRIPLETS) + len(FINGERTIP_PAIRS)
        assert result.shape[1] == expected_dim

    def test_output_is_float(self, fake_landmarks):
        """Feature tensor should be float32."""
        result = extract_landmark_features(fake_landmarks)
        assert result.dtype == torch.float32

    def test_deterministic(self, fake_landmarks):
        """Same input should produce identical output."""
        r1 = extract_landmark_features(fake_landmarks)
        r2 = extract_landmark_features(fake_landmarks)
        torch.testing.assert_close(r1, r2)
