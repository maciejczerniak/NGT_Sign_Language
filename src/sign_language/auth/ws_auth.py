"""WebSocket auth helper.

FastAPI-Users has no built-in WebSocket dependency, so this module decodes
the JWT manually using the same :class:`~fastapi_users.authentication.JWTStrategy`
as the HTTP auth backend. Authentication failures are always silent — the
WebSocket connection continues as anonymous.
"""

import logging
import uuid
from typing import Optional

from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase

from sign_language.db.engine import AsyncSessionLocal

from .manager import UserManager
from .models import User
from .users import get_jwt_strategy

logger = logging.getLogger(__name__)


async def get_user_from_ws_token(token: Optional[str]) -> Optional[User]:
    """Decode a JWT passed as a WebSocket query parameter and return the user.

    Opens a short-lived database session to resolve the user from the token.
    Any failure — missing token, invalid or expired JWT, unknown user, or
    inactive account — returns ``None`` silently so that callers treat
    anonymous and failed-auth sessions identically.

    :param token: The JWT bearer token passed via the WebSocket ``token``
        query parameter, or ``None`` if not provided.
    :returns: The authenticated :class:`~sign_language.auth.models.User` if
        the token is valid and the user is active, otherwise ``None``.
    """
    if not token:
        return None

    strategy = get_jwt_strategy()
    try:
        async with AsyncSessionLocal() as session:
            user_db: SQLAlchemyUserDatabase[User, uuid.UUID] = SQLAlchemyUserDatabase(
                session, User
            )
            user_manager = UserManager(user_db)
            result: Optional[User] = await strategy.read_token(token, user_manager)
            if result and result.is_active:
                return result
    except Exception as exc:  # noqa: BLE001 — auth failure is non-fatal
        logger.debug("WS token rejected: %s", exc)

    return None
