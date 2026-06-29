"""Request/response models for the API."""

from typing import Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for the ``POST /api/predict`` and ``/ws/predict`` endpoints."""

    image: str = Field(..., description="Base64-encoded frame (data URL or raw).")


class TopKItem(BaseModel):
    """A single entry in the top-k prediction list.

    :param letter: The predicted class label (e.g. ``"A"``).
    :param confidence: The model's confidence score for this class, in [0, 1].
    """

    letter: str
    confidence: float


class PredictResponse(BaseModel):
    """Response body for a single-frame inference request.

    :param hand_detected: Whether a hand was detected in the input frame.
    :param predicted_letter: The top-1 predicted class label, or ``None``
        if no hand was detected.
    :param confidence: The model's confidence score for the top-1 prediction.
    :param top_3: The top-3 predicted letters with their confidence scores.
    :param stable_letter: The smoothed stable prediction from the smoother,
        or ``None`` if the smoother has not yet committed to a letter.
    :param stable_confidence: Confidence of the stable letter, or ``None``
        if no stable letter is available.
    :param current_word: The word currently being built from committed letters.
    :param sentence: The full sentence accumulated so far.
    :param committed_letter: The most recently committed letter, or ``None``
        if no letter was committed in this frame.
    """

    hand_detected: bool
    predicted_letter: Optional[str]
    confidence: float
    top_3: list[TopKItem]
    stable_letter: Optional[str]
    stable_confidence: float | None
    current_word: str
    sentence: str
    committed_letter: Optional[str]


class ResetResponse(BaseModel):
    """Response body for the ``POST /api/reset`` endpoint.

    :param ok: Always ``True``, confirming the reset was applied successfully.
    """

    ok: bool = True


class InfoResponse(BaseModel):
    """Response body for the ``GET /api/info`` endpoint.

    :param app_name: The application name from project settings.
    :param version: The application version string from project settings.
    :param device: The compute device in use, e.g. ``"cpu"`` or ``"cuda:0"``.
    :param num_classes: The number of output classes the classifier supports.
    :param class_names: Ordered list of class label strings.
    :param landmark_model_available: Whether the landmark MLP model is loaded.
    :param hand_detector_available: Whether the MediaPipe hand detector is loaded.
    """

    app_name: str
    version: str
    device: str
    num_classes: int
    class_names: list[str]
    landmark_model_available: bool
    hand_detector_available: bool


# ── User stats ───────────────────────────────────────────────────────
# Level reflects how much of the 22-letter NGT alphabet the user has
# "learned". A letter counts as learned once it has been signed correctly
# 3 times (tracking to be implemented in Phase 2). Points are a separate
# score that rewards practice and do NOT affect level.
TOTAL_LETTERS = 22

# Distinct letters the user is encouraged to practise each day (home-page goal).
DAILY_GOAL = 5

# (min_letters_learned, level_number, level_name) — checked high to low.
_LEVEL_BANDS = [
    (22, 5, "Alphabet Master"),
    (15, 4, "Advanced"),
    (10, 3, "Halfway"),
    (5, 2, "Learner"),
    (0, 1, "Beginner"),
]


def level_band(letters_learned: int) -> tuple[int, str]:
    """Map a letters-learned count to a (level number, level name).

    Bands (22-letter NGT alphabet):
        0–4  → 1 Beginner
        5–9  → 2 Learner
        10–14 → 3 Halfway
        15–21 → 4 Advanced
        22   → 5 Alphabet Master

    :param letters_learned: Number of letters the user has learned (0–22).
    :returns: A ``(level, level_name)`` tuple.
    """
    for threshold, level, name in _LEVEL_BANDS:
        if letters_learned >= threshold:
            return level, name
    # Unreachable in practice (the 0-threshold band always matches); kept as a
    # defensive fallback in case the bands are ever edited.
    return 1, "Beginner"


class StatsResponse(BaseModel):
    """Response body for the ``GET /api/stats`` endpoint.

    The per-user progress shown on the logged-in home page. ``level`` and
    ``level_name`` are derived from ``letters_learned`` (see
    :func:`level_band`); they are not stored.

    :param streak: Consecutive-day practice streak.
    :param letters_learned: Letters learned so far (signed correctly 3×), 0–22.
    :param total_letters: Total letters in the NGT alphabet (22) — lets the
        frontend show progress as ``letters_learned / total_letters``.
    :param points: Points earned through practice.
    :param level: Level number (1–5), derived from ``letters_learned``.
    :param level_name: Human-readable level name, e.g. ``"Beginner"``.
    :param last_played: Name of the last mode/activity, or ``None``.
    """

    streak: int
    letters_learned: int
    total_letters: int = TOTAL_LETTERS
    points: int
    level: int
    level_name: str
    last_played: Optional[str]
    practiced_today: int = 0
    daily_goal: int = DAILY_GOAL


# ── Collect mode ─────────────────────────────────────────────────────
class CollectRequest(BaseModel):
    """Request body for ``POST /api/collect`` — one contributed sample.

    Matches what the Collect-mode frontend produces per sample.

    :param image: Base64-encoded image (data URL or raw base64).
    :param letter: The letter the sample is labelled with, e.g. ``"A"``.
    :param source: How it was captured — ``"camera"``, ``"upload"``, or ``"auto"``.
    :param language: Sign language the sample belongs to. Defaults to ``"NGT"``.
    """

    image: str = Field(..., description="Base64-encoded image (data URL or raw).")
    letter: str = Field(..., min_length=1, max_length=10)
    source: str = Field(..., description='"camera", "upload", or "auto".')
    language: str = Field(default="NGT", max_length=10)


class CollectResponse(BaseModel):
    """Response body for ``POST /api/collect``.

    :param id: The stored sample's unique identifier.
    :param letter: The letter the sample was labelled with.
    :param stored: Always ``True`` on success, confirming the sample was saved.
    """

    id: str
    letter: str
    stored: bool = True


# ── Progress reporting ───────────────────────────────────────────────
# A letter counts as "learned" once signed correctly this many times.
LEARNED_THRESHOLD = 3


class ProgressRequest(BaseModel):
    """Request body for ``POST /api/stats/progress`` — one practice event.

    Sent by the game / Learn pages each time the user attempts a sign.

    :param letter: The letter that was attempted, e.g. ``"A"``.
    :param correct: Whether the attempt was correct.
    :param points: Points earned for this event (the frontend computes these,
        accounting for hints/bonuses). Ignored when ``correct`` is False.
    :param activity: Optional name of the activity, e.g. ``"Random Letters"``;
        stored as ``last_played`` when provided.
    """

    letter: str = Field(..., min_length=1, max_length=10)
    correct: bool = Field(...)
    points: int = Field(default=0, ge=0)
    activity: str | None = Field(default=None, max_length=50)


class LetterProgressItem(BaseModel):
    """One letter's practice progress, for the home-page progress list.

    :param letter: The letter, e.g. ``"A"``.
    :param correct_count: Times the user has signed it correctly.
    :param learned: True once ``correct_count`` has reached LEARNED_THRESHOLD.
    """

    letter: str
    correct_count: int
    learned: bool


class LetterProgressResponse(BaseModel):
    """Response for ``GET /api/stats/letters``.

    :param threshold: Correct-count needed to "learn" a letter (LEARNED_THRESHOLD).
    :param total_letters: Total letters in the NGT alphabet being tracked.
    :param letters: Per-letter progress, ordered most-practiced first.
    """

    threshold: int
    total_letters: int
    letters: list[LetterProgressItem]
