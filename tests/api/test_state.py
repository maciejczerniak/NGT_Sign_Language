"""Unit tests for AppState."""

import threading

import pytest

from sign_language.api.state import AppState
from sign_language.core.settings import settings
from sign_language.utils.smoothing import PredictionSmoother
from sign_language.utils.sequence import SequenceBuilder

# Smoother constants pulled from settings so tests stay in sync with configuration.
_WINDOW = settings.smoother_window_size
_THRESHOLD = settings.smoother_acquire_threshold
_MIN_CONF = settings.smoother_min_confidence


@pytest.fixture
def state(mock_models) -> AppState:
    return AppState(models=mock_models)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAppStateConstruction:
    def test_smoother_is_prediction_smoother(self, state):
        """Verify app state construction smoother is prediction smoother."""
        assert isinstance(state.smoother, PredictionSmoother)

    def test_sequence_is_sequence_builder(self, state):
        """Verify app state construction sequence is sequence builder."""
        assert isinstance(state.sequence, SequenceBuilder)

    def test_lock_is_threading_lock(self, state):
        """Verify app state construction lock is threading lock."""
        assert isinstance(state.lock, type(threading.Lock()))

    def test_models_are_stored(self, state, mock_models):
        """Verify app state construction models are stored."""
        assert state.models is mock_models

    def test_stable_letter_starts_none(self, state):
        """Verify app state construction stable letter starts none."""
        assert state.smoother.stable_letter is None

    def test_sequence_word_starts_empty(self, state):
        """Verify app state construction sequence word starts empty."""
        assert state.sequence.current_word == ""


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestAppStateReset:
    def _fill_smoother(self, state: AppState) -> None:
        """Push enough high-confidence frames to produce a stable letter."""
        for _ in range(_WINDOW):
            state.smoother.update("A", _MIN_CONF + 0.1)

    def test_reset_clears_stable_letter(self, state):
        """Verify app state reset reset clears stable letter."""
        self._fill_smoother(state)
        assert state.smoother.stable_letter == "A"

        state.reset()
        assert state.smoother.stable_letter is None

    def test_reset_clears_stable_confidence(self, state):
        """Verify app state reset reset clears stable confidence."""
        self._fill_smoother(state)
        state.reset()
        assert state.smoother.stable_confidence == 0.0

    def test_reset_clears_smoother_buffer(self, state):
        """Verify app state reset reset clears smoother buffer."""
        self._fill_smoother(state)
        state.reset()
        assert len(state.smoother.buffer) == 0

    def test_reset_clears_sequence_word(self, state):
        """Verify app state reset reset clears sequence word."""
        state.sequence.current_word = "HELLO"
        state.reset()
        assert state.sequence.current_word == ""

    def test_reset_clears_sequence_sentence(self, state):
        """Verify app state reset reset clears sequence sentence."""
        state.sequence.sentence = "WORLD "
        state.reset()
        assert state.sequence.sentence == ""

    def test_reset_is_idempotent(self, state):
        """Calling reset on already-empty state must not raise."""
        state.reset()
        state.reset()
        assert state.smoother.stable_letter is None

    def test_reset_releases_lock(self, state):
        """Lock must be available after reset (not left acquired)."""
        state.reset()
        acquired = state.lock.acquire(blocking=False)
        assert acquired, "Lock was still held after reset()"
        state.lock.release()
