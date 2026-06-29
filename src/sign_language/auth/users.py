"""FastAPI-Users instance, JWT backend, and reusable dependencies.

This module wires together the JWT authentication backend and exposes
three reusable FastAPI dependencies for use across route handlers:

- :data:`current_active_user` — requires a valid bearer token (401 if missing)
- :data:`current_active_user_optional` — returns ``None`` instead of 401
- :data:`current_admin` — requires ``is_superuser=True`` (403 otherwise)
"""

import uuid

from fastapi_users import FastAPIUsers
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)

from sign_language.core.settings import settings

from .manager import get_user_manager
from .models import User

bearer_transport = BearerTransport(tokenUrl="api/auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """Create and return a :class:`~fastapi_users.authentication.JWTStrategy`.

    Reads the secret key and token lifetime from project settings.

    :returns: A :class:`~fastapi_users.authentication.JWTStrategy` configured
        with ``settings.secret_key`` and ``settings.jwt_lifetime_seconds``.
    """
    return JWTStrategy(
        secret=settings.secret_key,
        lifetime_seconds=settings.jwt_lifetime_seconds,
    )


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

# ── Reusable dependencies ────────────────────────────────────────────
# Required: 401 if missing/invalid token
current_active_user = fastapi_users.current_user(active=True)

# Optional: returns None instead of 401 — used for /predict
current_active_user_optional = fastapi_users.current_user(active=True, optional=True)

# Admin only: 403 if user is not is_superuser
current_admin = fastapi_users.current_user(active=True, superuser=True)
