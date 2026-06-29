"""Sequence builder.

Converts stable single-letter predictions into words and sentences by
tracking hold durations, cooldown periods, and pauses (hand removed from
view).

Grace period
------------
If ``stable_letter`` briefly drops to ``None`` for less than
``stable_grace_sec`` seconds the hold timer is *not* reset. This
prevents the 1-second commit window from restarting every time the
smoother has a noisy frame, which was the main cause of letters being
very hard to commit.
"""

import time
from typing import Optional

from sign_language.core.settings import settings


class SequenceBuilder:
    """Accumulate stable letter predictions into words and sentences.

    A letter is committed after it has been held for
    :attr:`letter_hold_sec` seconds. A space is inserted when no hand
    is detected for :attr:`space_pause_sec` seconds.

    Timing values default to the application settings but can be
    overridden per-instance for testing.
    """

    def __init__(
        self,
        letter_hold_sec: Optional[float] = None,
        cooldown_sec: Optional[float] = None,
        space_pause_sec: Optional[float] = None,
        stable_grace_sec: Optional[float] = None,
    ) -> None:
        """Initialise the sequence builder with optional timing overrides.

        :param letter_hold_sec: Seconds a stable letter must be held before
            it is committed. Defaults to ``settings.sequence_letter_hold_sec``.
        :param cooldown_sec: Minimum seconds between two consecutive letter
            commits. Defaults to ``settings.sequence_cooldown_sec``.
        :param space_pause_sec: Seconds of hand absence before a space is
            inserted into the sentence. Defaults to
            ``settings.sequence_space_pause_sec``.
        :param stable_grace_sec: Maximum seconds a stable letter may be
            absent before the hold timer is reset. Defaults to
            ``settings.sequence_stable_grace_sec``.
        """
        self.letter_hold_sec: float = (
            letter_hold_sec
            if letter_hold_sec is not None
            else settings.sequence_letter_hold_sec
        )
        self.cooldown_sec: float = (
            cooldown_sec if cooldown_sec is not None else settings.sequence_cooldown_sec
        )
        self.space_pause_sec: float = (
            space_pause_sec
            if space_pause_sec is not None
            else settings.sequence_space_pause_sec
        )
        self.stable_grace_sec: float = (
            stable_grace_sec
            if stable_grace_sec is not None
            else settings.sequence_stable_grace_sec
        )
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated sequence state.

        Resets the sentence, current word, hold timer, cooldown timer,
        no-hand timer, and grace-period tracker to their initial values.
        """
        self.sentence: str = ""
        self.current_word: str = ""
        self.last_letter: Optional[str] = None
        self.letter_since: Optional[float] = None
        self.last_committed: Optional[float] = None
        self.committed_letter: Optional[str] = None
        self.no_hand_since: Optional[float] = None
        self._stable_lost_since: Optional[float] = None

    def update(self, stable_letter: Optional[str], hand_detected: bool) -> dict:
        """Process one frame and return the current sequence state.

        Handles four cases in order:

        1. **No hand detected**: starts or extends the no-hand timer; inserts
           a space into the sentence if the hand has been absent for
           ``space_pause_sec`` seconds.
        2. **Stable letter briefly absent (grace period)**: if
           ``stable_letter`` is ``None`` but within ``stable_grace_sec`` of
           the last valid letter, the hold timer is preserved across the
           noisy frame.
        3. **Same letter as last committed**: hold timer is reset to prevent
           double-committing the same letter.
        4. **Hold timer**: starts or continues tracking the current letter;
           commits it to ``current_word`` once held for ``letter_hold_sec``
           seconds and the cooldown has elapsed.

        :param stable_letter: The smoothed stable letter from
            :class:`~sign_language.utils.smoothing.PredictionSmoother`,
            or ``None`` if no stable prediction is available.
        :param hand_detected: Whether a hand was detected in the current frame.
        :returns: Dict with keys ``current_word``, ``sentence``, and
            ``committed_letter``. ``committed_letter`` is the letter just
            committed in this frame, or ``None`` if no commit occurred.
        """
        now = time.time()
        committed_letter: Optional[str] = None

        # ── No hand ──────────────────────────────────────────────────────
        if not hand_detected:
            if self.no_hand_since is None:
                self.no_hand_since = now
            elif now - self.no_hand_since >= self.space_pause_sec:
                if self.current_word:
                    self.sentence += self.current_word + " "
                    self.current_word = ""
                    self.no_hand_since = now

            self.last_letter = None
            self.letter_since = None
            self._stable_lost_since = None
            self.committed_letter = None

            return {
                "current_word": self.current_word,
                "sentence": self.sentence.strip(),
                "committed_letter": None,
            }

        # ── Hand present ─────────────────────────────────────────────────
        self.no_hand_since = None

        # ── Grace period: stable_letter briefly None ─────────────────────
        if stable_letter is None:
            if self._stable_lost_since is None:
                self._stable_lost_since = now

            grace_expired = now - self._stable_lost_since > self.stable_grace_sec
            if not grace_expired and self.last_letter is not None:
                return {
                    "current_word": self.current_word,
                    "sentence": self.sentence.strip(),
                    "committed_letter": None,
                }

            self.last_letter = None
            self.letter_since = None
            self.committed_letter = None
            return {
                "current_word": self.current_word,
                "sentence": self.sentence.strip(),
                "committed_letter": None,
            }

        # Stable letter is present — clear the grace timer.
        self._stable_lost_since = None

        # ── Already committed this letter; wait for a change ─────────────
        if stable_letter == self.committed_letter:
            self.last_letter = None
            self.letter_since = None
            return {
                "current_word": self.current_word,
                "sentence": self.sentence.strip(),
                "committed_letter": None,
            }

        self.committed_letter = None

        # ── Start or continue tracking the hold timer ────────────────────
        if stable_letter != self.last_letter:
            self.last_letter = stable_letter
            self.letter_since = now
        else:
            held = now - self.letter_since  # type: ignore[operator]
            cooldown_ok = (
                self.last_committed is None
                or now - self.last_committed >= self.cooldown_sec
            )
            if held >= self.letter_hold_sec and cooldown_ok:
                self.current_word += stable_letter
                self.last_committed = now
                self.committed_letter = stable_letter
                committed_letter = stable_letter
                self.last_letter = None
                self.letter_since = None

        return {
            "current_word": self.current_word,
            "sentence": self.sentence.strip(),
            "committed_letter": committed_letter,
        }

    def clear(self) -> None:
        """Alias for :meth:`reset`.

        Provided for API consistency with other stateful helpers that
        expose a ``clear`` method.
        """
        self.reset()
