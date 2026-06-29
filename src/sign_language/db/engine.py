"""Async SQLAlchemy engine, session factory, and Base for ORM models."""

import ssl
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from sign_language.core.settings import settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models.

    All SQLAlchemy ORM models in the project should inherit from this class
    so their tables are registered on the shared metadata used by Alembic
    for migration autogeneration.
    """


connect_args = {}
if settings.require_ssl:
    ssl_ctx = ssl.create_default_context()
    connect_args["ssl"] = ssl_ctx

engine = create_async_engine(
    settings.database_url, echo=False, connect_args=connect_args
)

AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def get_async_session() -> AsyncIterator[AsyncSession]:
    """Yield an async SQLAlchemy session as a FastAPI dependency.

    Opens a new :class:`~sqlalchemy.ext.asyncio.AsyncSession` from
    :data:`AsyncSessionLocal` for the duration of a single request and
    closes it automatically on exit, whether the request succeeds or raises.

    :yields: An :class:`~sqlalchemy.ext.asyncio.AsyncSession` bound to the
        shared async engine.
    """
    async with AsyncSessionLocal() as session:
        yield session
