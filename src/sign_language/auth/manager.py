"""UserManager — business logic hooks for FastAPI-Users."""

import logging
import uuid
from typing import AsyncIterator, Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, UUIDIDMixin
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from sign_language.core.settings import settings
from sign_language.db.engine import get_async_session

from .models import User, UserStats

logger = logging.getLogger(__name__)


async def get_user_db(
    session: AsyncSession = Depends(get_async_session),
) -> AsyncIterator[SQLAlchemyUserDatabase]:
    """Yield a :class:`~fastapi_users_db_sqlalchemy.SQLAlchemyUserDatabase` instance.

    FastAPI dependency that wraps the active async session and the
    :class:`~sign_language.auth.models.User` model into a database adapter
    for FastAPI-Users.

    :param session: The active async SQLAlchemy session injected by
        :func:`~sign_language.db.engine.get_async_session`.
    :yields: A :class:`~fastapi_users_db_sqlalchemy.SQLAlchemyUserDatabase`
        bound to the current session.
    """
    yield SQLAlchemyUserDatabase(session, User)


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """Business logic hooks for FastAPI-Users user lifecycle events.

    Extends :class:`~fastapi_users.BaseUserManager` with UUID primary keys
    via :class:`~fastapi_users.UUIDIDMixin`. Token secrets are read from
    project settings.
    """

    reset_password_token_secret = settings.secret_key
    verification_token_secret = settings.secret_key

    async def on_after_register(
        self, user: User, request: Optional[Request] = None
    ) -> None:
        """Seed an empty stats row after a new user successfully registers.

        Creates a :class:`~sign_language.auth.models.UserStats` row (all
        counters at zero) so every user has stats from the moment they sign
        up, and the ``GET /api/stats`` endpoint always finds a row to return.

        :param user: The newly registered :class:`~sign_language.auth.models.User`.
        :param request: The originating :class:`~fastapi.Request`, if available.
        """
        logger.info("User registered: %s (%s)", user.email, user.id)

        # Seed a zeroed stats row for the new user, reusing the same async
        # session that created the user (held by the FastAPI-Users adapter).
        # self.user_db is the SQLAlchemy adapter, which exposes .session at
        # runtime; the BaseUserDatabase type doesn't declare it, hence the ignore.
        session: AsyncSession = self.user_db.session  # type: ignore[attr-defined]
        session.add(UserStats(user_id=user.id))
        await session.commit()
        logger.info("Seeded stats row for user %s", user.id)

    async def on_after_login(
        self,
        user: User,
        request: Optional[Request] = None,
        response: Optional[object] = None,
    ) -> None:
        """Log a message after a user successfully logs in.

        :param user: The :class:`~sign_language.auth.models.User` who logged in.
        :param request: The originating :class:`~fastapi.Request`, if available.
        :param response: The outgoing response object, if available.
        """
        logger.info("User logged in: %s", user.email)


async def get_user_manager(
    user_db: SQLAlchemyUserDatabase = Depends(get_user_db),
) -> AsyncIterator[UserManager]:
    """Yield a :class:`UserManager` instance as a FastAPI dependency.

    :param user_db: The database adapter injected by :func:`get_user_db`.
    :yields: A :class:`UserManager` bound to the provided database adapter.
    """
    yield UserManager(user_db)
