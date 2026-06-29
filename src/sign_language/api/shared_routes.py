"""DB-backed HTTP routes shared by every Sign Language API deployment.

These endpoints contain no inference or per-process state — they only read
and write the database and enforce auth. Both the in-process backend
(:mod:`sign_language.api.app`) and the Azure ML proxy
(:mod:`sign_language_azure_api.app`) mount this router under ``/api`` so the
user-facing surface stays identical across deployments.

Route groups
------------
- **Stats** — ``/stats``, ``/stats/progress``, ``/stats/letters``: the
  authenticated user's home-page progress (streak, points, per-letter
  mastery, daily goal).
- **Admin** — ``/admin/whoami``, ``/admin/metrics``,
  ``/admin/metrics/history``: superuser-only identity check and operational
  metrics read from ``monitoring_events``.

Inference, session, and collect routes are intentionally *not* here; each
app defines those itself because they differ between deployments.
"""

import logging
import uuid
from datetime import date, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from sign_language.auth.models import (
    DailyActivity,
    LetterProgress,
    User,
    UserStats,
)
from sign_language.auth.users import (
    current_active_user,
    current_admin,
)
from sign_language.db.engine import get_async_session

from .monitoring import get_metrics, get_metrics_history
from .schemas import (
    LEARNED_THRESHOLD,
    LetterProgressItem,
    LetterProgressResponse,
    ProgressRequest,
    StatsResponse,
    TOTAL_LETTERS,
    level_band,
)

logger = logging.getLogger(__name__)

shared_router = APIRouter()


# ── Admin identity ───────────────────────────────────────────────────
@shared_router.get("/admin/whoami")
def admin_whoami(admin: User = Depends(current_admin)) -> dict:
    """Return the identity of the authenticated admin user.

    Primarily a convenience endpoint for verifying that a JWT token
    belongs to a superuser account. Returns 403 for any token that does
    not have ``is_superuser=True``.

    Args:
        admin: Authenticated superuser injected by the ``current_admin``
            dependency. FastAPI returns 401/403 automatically if the
            token is missing, invalid, or not a superuser.

    Returns:
        Dict containing the user's ``id``, ``email``, and an
        ``is_admin`` flag set to ``True``.
    """
    return {"id": str(admin.id), "email": admin.email, "is_admin": True}


# ── User stats endpoint ──────────────────────────────────────────────
@shared_router.get("/stats", response_model=StatsResponse)
async def get_stats(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> StatsResponse:
    """Return the authenticated user's progress stats for the home page.

    Requires a valid bearer token (``401`` if missing/invalid). Loads the
    caller's :class:`~sign_language.auth.models.UserStats` row and returns
    it, with ``level``/``level_name`` derived from ``letters_learned``.

    Every user gets a stats row at registration, so one normally exists. As
    a defensive fallback (e.g. accounts created before this feature), a
    missing row is treated as all-zero rather than raising.

    :param user: The authenticated user, injected by ``current_active_user``.
    :param session: The active async SQLAlchemy session.
    :returns: A :class:`~sign_language.api.schemas.StatsResponse`.
    """
    result = await session.execute(
        select(UserStats).where(UserStats.user_id == user.id)
    )
    stats = result.scalar_one_or_none()

    if stats is None:
        # Defensive fallback: no row yet → report zeros, don't error.
        level, level_name = level_band(0)
        return StatsResponse(
            streak=0,
            letters_learned=0,
            points=0,
            level=level,
            level_name=level_name,
            last_played=None,
        )

    learned = await _count_letters_learned(session, user.id)
    practiced_today = await _practiced_today_count(session, user.id)
    level, level_name = level_band(learned)
    return StatsResponse(
        streak=stats.streak,
        letters_learned=learned,
        points=stats.points,
        level=level,
        level_name=level_name,
        last_played=stats.last_played,
        practiced_today=practiced_today,
    )


# ── Admin metrics endpoints ──────────────────────────────────────────
@shared_router.get("/admin/metrics")
async def admin_metrics(
    range: str = "all",
    date_from: str | None = None,
    date_to: str | None = None,
    admin: User = Depends(current_admin),
) -> dict:
    """Return aggregated operational metrics for the selected time range.

    Queries the ``monitoring_events`` table and returns a single snapshot
    of all key metrics — request counts, error rates, latency percentiles,
    prediction confidence, and Shannon entropy. Restricted to superuser
    accounts.

    Supports the same custom-date mode as the history endpoint: when both
    ``date_from`` and ``date_to`` are provided they take priority over
    ``range``, keeping the KPI cards aligned with the history charts.

    Args:
        range: Time window for the query. Accepted values are ``"1h"``,
            ``"6h"``, ``"1d"``, ``"7d"``, ``"30d"``, or ``"all"``
            (no time filter, returns full history). Ignored when
            ``date_from`` and ``date_to`` are both set.
        date_from: Start of a custom date range in ISO format, e.g.
            ``"2026-05-01"``. Optional.
        date_to: End of a custom date range in ISO format, e.g.
            ``"2026-05-30"``. Optional. Must be provided together with
            ``date_from``.
        admin: Authenticated superuser injected by the ``current_admin``
            dependency.

    Returns:
        Dict with keys: ``total_requests``, ``error_count``,
        ``error_rate``, ``avg_latency_ms``, ``p50_latency_ms``,
        ``p95_latency_ms``, ``avg_confidence``, ``avg_entropy``,
        ``total_predictions``, ``letter_counts``, ``error_breakdown``.
    """
    return await get_metrics(range_str=range, date_from=date_from, date_to=date_to)


@shared_router.get("/admin/metrics/history")
async def admin_metrics_history(
    range: str = "1h",
    date_from: str | None = None,
    date_to: str | None = None,
    admin: User = Depends(current_admin),
) -> list:
    """Return time-bucketed request and latency history for chart rendering.

    Supports two time-window modes: a preset range (e.g. ``"7d"``) or a
    custom date range specified by ``date_from`` and ``date_to``. When
    both custom date parameters are provided, ``range`` is ignored.
    Restricted to superuser accounts.

    Args:
        range: Preset time window. Accepted values are ``"1h"``,
            ``"6h"``, ``"1d"``, ``"7d"``, ``"30d"``, or ``"all"``.
            Ignored when ``date_from`` and ``date_to`` are both set.
        date_from: Start of a custom date range in ISO format, e.g.
            ``"2026-05-01"``. Optional.
        date_to: End of a custom date range in ISO format, e.g.
            ``"2026-05-30"``. Optional. Must be provided together with
            ``date_from``.
        admin: Authenticated superuser injected by the ``current_admin``
            dependency.

    Returns:
        List of time-bucket dicts ordered chronologically, each
        containing: ``timestamp``, ``request_count``, ``error_count``,
        ``avg_latency_ms``, ``p50_latency_ms``, ``p95_latency_ms``,
        ``avg_entropy``.
    """
    return await get_metrics_history(
        range_str=range, date_from=date_from, date_to=date_to
    )


# ── Progress helpers ─────────────────────────────────────────────────
async def _count_letters_learned(session: AsyncSession, user_id: uuid.UUID) -> int:
    """Count how many letters the user has "learned".

    A letter is learned once its ``correct_count`` reaches
    ``LEARNED_THRESHOLD``. Computed live from ``letter_progress`` so the
    aggregate can never drift from the per-letter counts.
    """
    result = await session.execute(
        select(func.count())
        .select_from(LetterProgress)
        .where(
            LetterProgress.user_id == user_id,
            LetterProgress.correct_count >= LEARNED_THRESHOLD,
        )
    )
    return int(result.scalar_one())


async def _practiced_today_count(session: AsyncSession, user_id: uuid.UUID) -> int:
    """Return how many distinct letters the user has practised today.

    Reads today's ``daily_activity`` row (if any) and returns the size of its
    letter set. Zero when the user hasn't practised today.
    """
    result = await session.execute(
        select(DailyActivity).where(
            DailyActivity.user_id == user_id,
            DailyActivity.activity_date == date.today(),
        )
    )
    row = result.scalar_one_or_none()
    return len(row.letters) if row is not None else 0


# ── Progress reporting endpoint ──────────────────────────────────────
@shared_router.post("/stats/progress", response_model=StatsResponse)
async def report_progress(
    payload: ProgressRequest,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> StatsResponse:
    """Record one practice event and return the user's updated stats.

    Called by the game / Learn pages each time the user attempts a sign.
    Requires authentication. On a correct attempt this adds the reported
    ``points`` and increments the user's per-letter ``correct_count`` (a
    letter becomes "learned" at ``LEARNED_THRESHOLD`` correct attempts). The
    daily streak is updated on every event based on ``last_active_date``:
    practising again the same day leaves it unchanged, the next day extends
    it, and a longer gap resets it to 1.

    :param payload: The practice event — letter, correct flag, points, activity.
    :param user: The authenticated user, injected by ``current_active_user``.
    :param session: The active async SQLAlchemy session.
    :returns: The updated :class:`~sign_language.api.schemas.StatsResponse`.
    """
    result = await session.execute(
        select(UserStats).where(UserStats.user_id == user.id)
    )
    stats = result.scalar_one_or_none()
    if stats is None:
        # Defensive: seed a row if somehow missing (e.g. pre-feature account).
        stats = UserStats(user_id=user.id)
        session.add(stats)

    # ── Streak, points, and per-letter progress (only on a correct attempt) ──
    # The streak counts days on which the user signed at least one letter
    # correctly, consistent with points and letter progress — a wrong attempt
    # does not advance it.
    if payload.correct:
        # Streak: compare last active day to today.
        today = date.today()
        last = stats.last_active_date
        if last is None:
            stats.streak = 1
        elif last == today:
            pass  # already practised today — streak unchanged
        elif last == today - timedelta(days=1):
            stats.streak = (stats.streak or 0) + 1  # consecutive day
        else:
            stats.streak = 1  # gap → restart
        stats.last_active_date = today

        stats.points = (stats.points or 0) + payload.points

        lp_result = await session.execute(
            select(LetterProgress).where(
                LetterProgress.user_id == user.id,
                LetterProgress.letter == payload.letter,
            )
        )
        lp = lp_result.scalar_one_or_none()
        if lp is None:
            lp = LetterProgress(user_id=user.id, letter=payload.letter, correct_count=1)
            session.add(lp)
        else:
            lp.correct_count = (lp.correct_count or 0) + 1

        # Record the letter toward today's daily goal (distinct letters today).
        # Fetch-or-create today's row and add the letter only if it's new.
        # Reassign the list (rather than .append) so SQLAlchemy detects the
        # ARRAY change and persists it.
        day_result = await session.execute(
            select(DailyActivity).where(
                DailyActivity.user_id == user.id,
                DailyActivity.activity_date == date.today(),
            )
        )
        day = day_result.scalar_one_or_none()
        if day is None:
            day = DailyActivity(
                user_id=user.id, activity_date=date.today(), letters=[payload.letter]
            )
            session.add(day)
        elif payload.letter not in day.letters:
            day.letters = day.letters + [payload.letter]

    if payload.activity is not None:
        stats.last_played = payload.activity

    await session.commit()

    learned = await _count_letters_learned(session, user.id)
    practiced_today = await _practiced_today_count(session, user.id)
    level, level_name = level_band(learned)
    logger.info(
        "Progress for user %s: letter=%s correct=%s points+=%s "
        "(total streak=%s, points=%s, learned=%s)",
        user.id,
        payload.letter,
        payload.correct,
        payload.points,
        stats.streak,
        stats.points,
        learned,
    )
    return StatsResponse(
        streak=stats.streak,
        letters_learned=learned,
        points=stats.points,
        level=level,
        level_name=level_name,
        last_played=stats.last_played,
        practiced_today=practiced_today,
    )


@shared_router.get("/stats/letters", response_model=LetterProgressResponse)
async def get_letter_progress(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> LetterProgressResponse:
    """Return the authenticated user's per-letter practice progress.

    Lists every letter the user has practiced at least once (i.e. has a
    ``letter_progress`` row), with its correct-count and whether it has been
    learned (count >= LEARNED_THRESHOLD). Ordered most-practiced first so the
    home-page progress list naturally surfaces nearly-learned letters near the
    top. The total alphabet size and the threshold are included so the UI can
    render "x / 3" and "of N mastered" without hardcoding them.

    :param user: The authenticated user, injected by ``current_active_user``.
    :param session: The active async SQLAlchemy session.
    :returns: A :class:`~sign_language.api.schemas.LetterProgressResponse`.
    """
    result = await session.execute(
        select(LetterProgress)
        .where(LetterProgress.user_id == user.id)
        .order_by(LetterProgress.correct_count.desc(), LetterProgress.letter.asc())
    )
    rows = result.scalars().all()

    items = [
        LetterProgressItem(
            letter=row.letter,
            correct_count=row.correct_count,
            learned=row.correct_count >= LEARNED_THRESHOLD,
        )
        for row in rows
    ]
    return LetterProgressResponse(
        threshold=LEARNED_THRESHOLD,
        total_letters=TOTAL_LETTERS,
        letters=items,
    )
