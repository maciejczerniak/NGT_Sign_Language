"""Tests for the SequenceBuilder."""

from unittest.mock import patch

from sign_language.utils.sequence import SequenceBuilder


class TestSequenceBuilder:
    """Group all sequence builder tests together."""

    # Fixed timing values used across all tests so they never depend on
    # whatever the .env or settings singleton happens to contain.
    _HOLD = 1.0
    _COOLDOWN = 1.0
    _SPACE = 1.5
    _GRACE = 0.4

    @classmethod
    def _make_sb(cls, **overrides) -> SequenceBuilder:
        """Return a SequenceBuilder with deterministic, settings-independent timings."""
        return SequenceBuilder(
            letter_hold_sec=overrides.get("letter_hold_sec", cls._HOLD),
            cooldown_sec=overrides.get("cooldown_sec", cls._COOLDOWN),
            space_pause_sec=overrides.get("space_pause_sec", cls._SPACE),
            stable_grace_sec=overrides.get("stable_grace_sec", cls._GRACE),
        )

    def test_initial_state_is_empty(self):
        """Fresh builder should have empty sentence and word."""
        sb = self._make_sb()
        assert sb.sentence == ""
        assert sb.current_word == ""

    def test_no_commit_before_hold_time(self):
        """Letter should not commit if hold duration is not met."""
        sb = self._make_sb()
        result = sb.update("A", True)
        assert result["committed_letter"] is None
        assert result["current_word"] == ""

    def test_letter_commits_after_hold_time(self):
        """Letter should commit once held for letter_hold_sec."""
        sb = self._make_sb()

        # First call sets the letter and starts the timer
        with patch("sign_language.utils.sequence.time") as mock_time:
            mock_time.time.return_value = 100.0
            sb.update("A", True)

            # Same letter, still within hold time
            mock_time.time.return_value = 100.5
            result = sb.update("A", True)
            assert result["committed_letter"] is None

            # Same letter, past hold time
            mock_time.time.return_value = 101.1
            result = sb.update("A", True)
            assert result["committed_letter"] == "A"
            assert result["current_word"] == "A"

    def test_space_on_hand_removed(self):
        """Word should move to sentence when hand is absent long enough."""
        sb = self._make_sb()

        with patch("sign_language.utils.sequence.time") as mock_time:
            # Commit a letter
            mock_time.time.return_value = 100.0
            sb.update("H", True)
            mock_time.time.return_value = 101.1
            sb.update("H", True)
            assert sb.current_word == "H"

            # Hand disappears
            mock_time.time.return_value = 102.0
            sb.update(None, False)

            # Past space_pause_sec (1.5 s)
            mock_time.time.return_value = 103.6
            result = sb.update(None, False)
            assert result["current_word"] == ""
            assert "H" in result["sentence"]

    def test_multiple_letters_build_word(self):
        """Consecutive committed letters should build up current_word."""
        sb = self._make_sb()

        with patch("sign_language.utils.sequence.time") as mock_time:
            # Commit "H"
            mock_time.time.return_value = 100.0
            sb.update("H", True)
            mock_time.time.return_value = 101.1
            sb.update("H", True)

            # Commit "I" (after cooldown)
            mock_time.time.return_value = 102.2
            sb.update("I", True)
            mock_time.time.return_value = 103.3
            sb.update("I", True)

            assert sb.current_word == "HI"

    def test_clear_resets_everything(self):
        """After clear(), all state should be empty."""
        sb = self._make_sb()

        with patch("sign_language.utils.sequence.time") as mock_time:
            mock_time.time.return_value = 100.0
            sb.update("A", True)
            mock_time.time.return_value = 101.1
            sb.update("A", True)

        sb.clear()
        assert sb.sentence == ""
        assert sb.current_word == ""
        assert sb.last_letter is None
        assert sb.committed_letter is None

    def test_update_returns_correct_keys(self):
        """Return dict should have exactly the expected keys."""
        sb = self._make_sb()
        result = sb.update(None, False)
        assert set(result.keys()) == {"current_word", "sentence", "committed_letter"}

    def test_no_hand_resets_letter_tracking(self):
        """When hand disappears briefly, letter tracking should reset."""
        sb = self._make_sb()

        with patch("sign_language.utils.sequence.time") as mock_time:
            mock_time.time.return_value = 100.0
            sb.update("A", True)

            # Hand gone (but not long enough for space)
            mock_time.time.return_value = 100.5
            sb.update(None, False)

            assert sb.last_letter is None
            assert sb.letter_since is None

    def test_same_letter_after_commit_is_ignored(self):
        """Once a letter is committed, the same stable letter should not re-trigger."""
        sb = self._make_sb()

        with patch("sign_language.utils.sequence.time") as mock_time:
            # Commit "A"
            mock_time.time.return_value = 100.0
            sb.update("A", True)
            mock_time.time.return_value = 101.1
            result = sb.update("A", True)
            assert result["committed_letter"] == "A"

            # Same "A" again — should be ignored (no double commit)
            mock_time.time.return_value = 102.2
            result = sb.update("A", True)
            assert result["committed_letter"] is None
            assert sb.current_word == "A"  # still just one A

    def test_stable_none_resets_committed(self):
        """When stable_letter becomes None, committed tracking should reset."""
        sb = self._make_sb()

        with patch("sign_language.utils.sequence.time") as mock_time:
            # Commit "A"
            mock_time.time.return_value = 100.0
            sb.update("A", True)
            mock_time.time.return_value = 101.1
            sb.update("A", True)
            assert sb.current_word == "A"

            # Stable goes None
            mock_time.time.return_value = 102.0
            sb.update(None, True)
            assert sb.committed_letter is None
