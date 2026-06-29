"""Tests for the PredictionSmoother."""

from sign_language.utils.smoothing import PredictionSmoother


class TestPredictionSmoother:
    """Group all smoother tests together."""

    # Fixed values so tests are independent of whatever .env says.
    _WINDOW = 15
    _ACQUIRE = 10
    _STICKY = 7
    _MIN_CONF = 0.55

    @classmethod
    def _make_smoother(cls, **overrides) -> PredictionSmoother:
        """Return a PredictionSmoother with deterministic, settings-independent config."""
        return PredictionSmoother(
            window_size=overrides.get("window_size", cls._WINDOW),
            acquire_threshold=overrides.get("acquire_threshold", cls._ACQUIRE),
            sticky_threshold=overrides.get("sticky_threshold", cls._STICKY),
            min_confidence=overrides.get("min_confidence", cls._MIN_CONF),
        )

    def test_no_stable_letter_before_window_full(self):
        """Should not produce a stable letter until the buffer is full."""
        s = self._make_smoother()
        for _ in range(s.window_size - 1):
            s.update("A", 0.95)
        assert s.stable_letter is None

    def test_stable_letter_when_threshold_met(self):
        """Should produce a stable letter once enough frames agree."""
        s = self._make_smoother()
        for _ in range(s.window_size):
            s.update("B", 0.90)
        assert s.stable_letter == "B"
        assert s.stable_confidence >= 0.90

    def test_no_stable_letter_when_mixed(self):
        """Should not stabilise when votes are split."""
        s = self._make_smoother()
        letters = ["A", "B", "C"]
        for i in range(s.window_size):
            s.update(letters[i % 3], 0.90)
        assert s.stable_letter is None

    def test_no_stable_letter_when_confidence_low(self):
        """Should reject even unanimous votes if confidence is below threshold."""
        s = self._make_smoother()
        for _ in range(s.window_size):
            s.update("D", 0.30)
        assert s.stable_letter is None

    def test_stable_letter_changes(self):
        """Stable letter should update when a new letter dominates."""
        s = self._make_smoother()
        for _ in range(s.window_size):
            s.update("A", 0.95)
        assert s.stable_letter == "A"

        for _ in range(s.window_size):
            s.update("Z", 0.95)
        assert s.stable_letter == "Z"

    def test_clear_resets_everything(self):
        """After clear(), buffer and stable state should be empty."""
        s = self._make_smoother()
        for _ in range(s.window_size):
            s.update("X", 0.99)
        assert s.stable_letter == "X"

        s.clear()
        assert s.stable_letter is None
        assert s.stable_confidence == 0.0
        assert len(s.buffer) == 0

    def test_confidence_is_average_of_matching_frames(self):
        """Stable confidence should be the average of the winning letter's frames."""
        s = self._make_smoother()
        confidences = [
            0.80,
            0.90,
            0.85,
            0.88,
            0.92,
            0.80,
            0.90,
            0.85,
            0.88,
            0.92,
            0.80,
            0.90,
            0.85,
            0.88,
            0.92,
        ]
        for c in confidences:
            s.update("A", c)

        expected = sum(confidences) / len(confidences)
        assert abs(s.stable_confidence - expected) < 1e-6
