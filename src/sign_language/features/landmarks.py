"""Hand landmark feature extraction.

Normalises raw MediaPipe hand landmarks and engineers geometric features
(joint angles, fingertip distances) to match the training pipeline used
by the Landmark MLP.
"""

import math
from itertools import combinations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Joint triplets used for angle computation (must match training exactly)
# ---------------------------------------------------------------------------
JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (0, 5, 6),
    (5, 6, 7),
    (6, 7, 8),
    (0, 9, 10),
    (9, 10, 11),
    (10, 11, 12),
    (0, 13, 14),
    (13, 14, 15),
    (14, 15, 16),
    (0, 17, 18),
    (17, 18, 19),
    (18, 19, 20),
    (5, 9, 13),
    (9, 13, 17),
    (0, 5, 17),
    (5, 9, 17),
    (9, 13, 17),
    (4, 8, 12),
]

FINGERTIP_PAIRS: list[tuple[int, int]] = list(combinations([4, 8, 12, 16, 20], 2))


def normalize_landmarks(landmarks: list[dict]) -> np.ndarray:
    """Centre landmarks on the wrist and scale by the wrist-to-middle-MCP distance.

    Translates all 21 landmarks so that landmark 0 (wrist) is at the origin,
    then divides by the Euclidean norm of landmark 9 (middle-finger MCP) to
    make the representation scale-invariant.

    :param landmarks: List of 21 dicts each with ``x``, ``y``, and ``z`` keys
        containing normalised MediaPipe coordinates.
    :returns: A ``(21, 3)`` float32 :class:`numpy.ndarray` of normalised
        landmark coordinates.
    """
    pts = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)
    pts -= pts[0].copy()
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale
    return pts


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angle in radians between two vectors.

    :param v1: First vector as a :class:`numpy.ndarray`.
    :param v2: Second vector as a :class:`numpy.ndarray`.
    :returns: Angle in radians in [0, π], or ``0.0`` if either vector has
        near-zero magnitude.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))


def extract_landmark_features(landmarks: list[dict]) -> torch.Tensor:
    """Build the full feature vector from raw MediaPipe landmarks.

    Concatenates three feature groups into a single 1D tensor:

    1. Flattened normalised landmark coordinates (``21 × 3 = 63`` values).
    2. Joint angles for each triplet in :data:`JOINT_TRIPLETS`.
    3. Euclidean distances between all fingertip pairs in :data:`FINGERTIP_PAIRS`.

    The resulting feature vector must match the input dimensionality expected
    by the Landmark MLP exactly.

    :param landmarks: List of 21 landmark dicts with ``x``, ``y``, and ``z``
        keys, as returned by MediaPipe or the preprocessing pipeline.
    :returns: A :class:`torch.Tensor` of shape ``(1, feature_dim)`` with
        dtype ``float32``, ready for direct input to the Landmark MLP.
    """
    pts = normalize_landmarks(landmarks)
    raw = pts.flatten()
    angles = [
        angle_between(pts[a] - pts[b], pts[c] - pts[b]) for a, b, c in JOINT_TRIPLETS
    ]
    dists = [float(np.linalg.norm(pts[i] - pts[j])) for i, j in FINGERTIP_PAIRS]
    features = np.concatenate([raw, angles, dists]).astype(np.float32)
    return torch.tensor(features).unsqueeze(0)
