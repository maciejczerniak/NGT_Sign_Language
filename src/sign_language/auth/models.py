"""SQLAlchemy models — User, per-user stats, collected samples, monitoring."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import sqlalchemy as sa
from sqlalchemy import JSON
from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from fastapi_users_db_sqlalchemy.generics import GUID
from sqlalchemy.orm import Mapped, mapped_column

from sign_language.db.engine import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    """User table.

    Inherits from SQLAlchemyBaseUserTableUUID, which provides:
        id              : UUID primary key
        email           : str, unique
        hashed_password : str
        is_active       : bool (default True)
        is_superuser    : bool (default False)  ← used as "admin" flag
        is_verified     : bool (default False)

    Three privilege tiers:
        anonymous (no user)         — public endpoints + /predict, /ws/predict
        regular  (is_superuser=F)   — anonymous + /stats, /progress (future)
        admin    (is_superuser=T)   — everything, incl. /api/admin/*
    """


class UserStats(Base):
    """Per-user progress stats shown on the logged-in home page.

    One row per user, created when they register (see
    ``UserManager.on_after_register``). Holds the values the home dashboard
    displays. ``level`` is *not* stored — it is computed from ``points`` on
    read, so it can never drift out of sync with the points total.

    Columns
    -------
    user_id          : FK to ``user.id`` (UUID) — primary key, one row per user.
    streak           : Consecutive-day practice streak.
    points           : Points earned through practice.
    last_played      : Name of the last mode/activity, or NULL if never played.
    last_active_date : Date the user last practiced — used to compute the streak.
    updated_at       : UTC timestamp of the last update to this row.
    """

    __tablename__ = "user_stats"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        sa.ForeignKey("user.id", ondelete="CASCADE"),
        primary_key=True,
    )
    streak: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    points: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    last_played: Mapped[str | None] = mapped_column(sa.String(50), nullable=True)
    last_active_date: Mapped[date | None] = mapped_column(sa.Date, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class CollectedSample(Base):
    """One labelled fingerspelling sample contributed via Collect mode.

    The image bytes are written to disk under ``settings.collect_storage_dir``;
    this row stores the file path plus metadata, not the bytes themselves.
    This mirrors how object storage works, so a future move to Azure Blob only
    changes ``image_path`` from a local filesystem path to a blob URL — the
    table and the rest of the flow stay the same.

    Authentication is optional for contributing, so ``user_id`` is nullable:
    logged-in users are attributed, anonymous/guest contributions have NULL.

    Columns
    -------
    id           : UUID primary key — globally unique sample identifier.
    user_id      : FK to ``user.id`` (UUID), or NULL for anonymous/guest
                   contributions. ON DELETE SET NULL so deleting a user keeps
                   their contributed samples (just un-attributes them).
    letter       : The labelled letter the sample represents, e.g. ``"A"``.
    image_path   : Path to the stored image file (local path now, blob URL later).
    source       : How the sample was captured — ``"camera"``, ``"upload"``,
                   or ``"auto"``.
    language     : Sign language the sample belongs to. Defaults to ``"NGT"``.
    created_at   : UTC timestamp when the sample was contributed.
    """

    __tablename__ = "collected_samples"

    id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        GUID(),
        sa.ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    letter: Mapped[str] = mapped_column(sa.String(10), nullable=False, index=True)
    image_path: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    source: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    language: Mapped[str] = mapped_column(sa.String(10), nullable=False, default="NGT")
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class MonitoringEvent(Base):
    """Persistent store for one operational event (request or prediction).

    One row is written per HTTP request by ``MonitoringMiddleware``.
    Prediction-specific columns (``predicted_letter``, ``confidence``,
    ``endpoint_type``) are populated by ``track_prediction()`` and are
    ``NULL`` for non-inference requests.

    Columns
    -------
    id              : UUID primary key — globally unique event identifier.
    timestamp       : UTC timestamp of the event — used for time-range queries.
    path            : Request path, e.g. ``/api/predict``.
    method          : HTTP method, e.g. ``GET``, ``POST``.
    status_code     : HTTP response status code.
    latency_ms      : Wall-clock request duration in milliseconds.
    predicted_letter: Top-1 predicted letter, or NULL if no inference occurred.
    confidence      : Model confidence score (0–1), or NULL.
    endpoint_type   : ``"http"`` or ``"ws"`` — which interface produced the
                      prediction. NULL for non-inference requests.
    entropy         : Shannon entropy over the normalised top-3 probabilities,
                      or NULL when not applicable.
    """

    __tablename__ = "monitoring_events"

    id: Mapped[uuid.UUID] = mapped_column(
        sa.UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    timestamp: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    path: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    method: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    status_code: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(sa.Float, nullable=False)
    predicted_letter: Mapped[str | None] = mapped_column(sa.String(10), nullable=True)
    confidence: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    endpoint_type: Mapped[str | None] = mapped_column(sa.String(10), nullable=True)
    entropy: Mapped[float | None] = mapped_column(sa.Float, nullable=True)


class LetterProgress(Base):
    """Per-user, per-letter practice progress.

    One row per (user, letter) pair, tracking how many times the user has
    signed that letter correctly. A letter counts as "learned" once
    ``correct_count`` reaches ``LEARNED_THRESHOLD`` (3). The aggregate
    ``letters_learned`` shown to the user is *computed* by counting the rows
    at or above that threshold (see the stats endpoint) rather than stored, so
    it can never drift out of sync with the underlying counts.

    Columns
    -------
    user_id       : FK to ``user.id`` (UUID). Part of the composite PK.
    letter        : The letter, e.g. ``"A"``. Part of the composite PK.
    correct_count : Number of times the user has signed this letter correctly.
    updated_at    : UTC timestamp of the last update to this row.
    """

    __tablename__ = "letter_progress"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        sa.ForeignKey("user.id", ondelete="CASCADE"),
        primary_key=True,
    )
    letter: Mapped[str] = mapped_column(sa.String(10), primary_key=True)
    correct_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class DailyActivity(Base):
    """Distinct letters a user practiced on a given calendar day.

    Backs the home-page "daily goal" (practise N distinct letters today).
    One row per (user, day). ``letters`` holds the set of distinct letters
    practised that day — a letter is added only once no matter how many times
    it is signed, so ``len(letters)`` is the distinct-letter count the goal
    measures. Resets naturally each day because each day is a new row.

    Columns
    -------
    user_id       : FK to ``user.id`` (UUID). Part of the composite PK.
    activity_date : The calendar day (server local date). Part of the PK.
    letters       : Distinct letters practised that day, e.g. ``["A", "C"]``.
    """

    __tablename__ = "daily_activity"

    user_id: Mapped[uuid.UUID] = mapped_column(
        GUID(),
        sa.ForeignKey("user.id", ondelete="CASCADE"),
        primary_key=True,
    )
    activity_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    # ARRAY on Postgres (production); JSON on SQLite so the test suite,
    # which uses an in-memory SQLite DB, can create and query this table.
    letters: Mapped[list[str]] = mapped_column(
        sa.ARRAY(sa.String(10)).with_variant(JSON(), "sqlite"),
        nullable=False,
        default=list,
    )
