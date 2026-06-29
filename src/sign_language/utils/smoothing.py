"""Prediction smoother.

Buffers the last *N* frame-level predictions and emits a stable letter
only when a supermajority of recent frames agree with sufficient average
confidence. This eliminates the flickering that raw per-frame output
would produce.

Hysteresis
----------
Gaining stability requires ``acquire_threshold`` votes (strict).
Keeping stability only requires ``sticky_threshold`` votes (lenient).
This prevents a single noisy frame from instantly clearing a stable
reading that the window spent time building up.
"""

from collections import Counter, deque
from typing import Optional

from sign_language.core.settings import settings


class PredictionSmoother:
    """Sliding-window majority-vote smoother with hysteresis.

    Buffers the last ``window_size`` frame predictions and emits a stable
    letter only when a supermajority of frames agree with sufficient average
    confidence. Two separate vote thresholds implement hysteresis: a strict
    ``acquire_threshold`` to gain stability and a lenient ``sticky_threshold``
    to maintain it.

    All tuning parameters default to the application settings but can be
    overridden per-instance for testing.

    :ivar stable_letter: The currently agreed-upon letter, or ``None`` if
        no stable prediction has been reached.
    :ivar stable_confidence: Average confidence of the stable letter across
        its supporting frames in the current window. Zero when unstable.
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        acquire_threshold: Optional[int] = None,
        sticky_threshold: Optional[int] = None,
        min_confidence: Optional[float] = None,
    ) -> None:
        """Initialise the smoother with optional parameter overrides.

        :param window_size: Number of recent frames to keep in the sliding
            window. Defaults to ``settings.smoother_window_size``.
        :param acquire_threshold: Minimum vote count required to acquire a
            stable letter (strict bar). Defaults to
            ``settings.smoother_acquire_threshold``.
        :param sticky_threshold: Minimum vote count required to keep an
            already-stable letter stable (lenient bar). Defaults to
            ``settings.smoother_sticky_threshold``.
        :param min_confidence: Per-frame confidence floor below which frames
            are excluded from the average confidence calculation. Defaults to
            ``settings.smoother_min_confidence``.
        """
        self.window_size: int = (
            window_size if window_size is not None else settings.smoother_window_size
        )
        self.acquire_threshold: int = (
            acquire_threshold
            if acquire_threshold is not None
            else settings.smoother_acquire_threshold
        )
        self.sticky_threshold: int = (
            sticky_threshold
            if sticky_threshold is not None
            else settings.smoother_sticky_threshold
        )
        self.min_confidence: float = (
            min_confidence
            if min_confidence is not None
            else settings.smoother_min_confidence
        )
        self.buffer: deque[tuple[str, float]] = deque(maxlen=self.window_size)
        self.stable_letter: str | None = None
        self.stable_confidence: float = 0.0

    def update(self, letter: str, confidence: float) -> None:
        """Add a new frame prediction and recalculate the stable letter.

        Appends the prediction to the sliding window buffer. If the buffer
        is not yet full, returns immediately without updating stability.

        Once full, applies hysteresis: if a letter is already stable, checks
        whether it still has at least ``sticky_threshold`` votes with
        sufficient average confidence. If so, keeps it. If support has
        collapsed, falls through to the standard acquisition path which
        requires ``acquire_threshold`` votes to establish a new stable letter.

        :param letter: Predicted letter label for the current frame.
        :param confidence: Model confidence score for ``letter`` in [0, 1].
        """
        self.buffer.append((letter, confidence))

        if len(self.buffer) < self.window_size:
            return

        votes = Counter(ltr for ltr, _ in self.buffer)
        top_letter, top_count = votes.most_common(1)[0]

        # ── Hysteresis: use a lower bar to keep the current stable letter ──
        if self.stable_letter is not None:
            current_count = votes.get(self.stable_letter, 0)
            current_avg = (
                sum(c for ltr, c in self.buffer if ltr == self.stable_letter)
                / current_count
                if current_count > 0
                else 0.0
            )
            if (
                current_count >= self.sticky_threshold
                and current_avg >= self.min_confidence
            ):
                self.stable_confidence = current_avg
                return

        # ── Standard acquisition path ────────────────────────────────────
        if top_count >= self.acquire_threshold:
            avg_conf = sum(c for ltr, c in self.buffer if ltr == top_letter) / top_count
            if avg_conf >= self.min_confidence:
                self.stable_letter = top_letter
                self.stable_confidence = avg_conf
                return

        self.stable_letter = None
        self.stable_confidence = 0.0

    def clear(self) -> None:
        """Reset the prediction buffer and clear the stable letter state.

        After calling this method, ``stable_letter`` is ``None`` and
        ``stable_confidence`` is ``0.0``.
        """
        self.buffer.clear()
        self.stable_letter = None
        self.stable_confidence = 0.0
