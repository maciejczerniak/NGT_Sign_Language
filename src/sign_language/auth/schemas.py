"""Pydantic schemas for user-facing auth payloads."""

import uuid

from fastapi_users import schemas


class UserRead(schemas.BaseUser[uuid.UUID]):
    """Public user representation (returned by /users/me etc.)."""


class UserCreate(schemas.BaseUserCreate):
    """Fields required to register a new user (email + password)."""


class UserUpdate(schemas.BaseUserUpdate):
    """Fields a user is allowed to update on themselves."""
