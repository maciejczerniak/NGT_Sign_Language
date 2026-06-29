"""Alembic environment — async-aware, reads URL from app settings.

This module is the Alembic entry point executed for every ``alembic``
CLI command (``upgrade``, ``downgrade``, ``revision --autogenerate``,
etc.).  It supports two execution modes:

Offline mode
    Generates SQL migration scripts without requiring a live database
    connection.  Useful for reviewing changes before applying them.

Online mode
    Connects to the live database and applies migrations directly.
    Uses an async SQLAlchemy engine to match the application's async
    stack.

Model registration
    Every ORM model must be imported here so that its table registers
    on ``Base.metadata``.  Without the import, ``--autogenerate`` will
    not detect the table and will silently skip it.  Currently registered
    models: :class:`~sign_language.auth.models.User`,
    :class:`~sign_language.auth.models.MonitoringEvent`.

Configuration
    The database URL is read from
    :attr:`~sign_language.core.settings.Settings.database_url` at
    runtime, overriding the blank ``sqlalchemy.url`` in ``alembic.ini``.
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from sign_language.auth.models import User, MonitoringEvent  # noqa: F401
from sign_language.core.settings import settings
from sign_language.db.engine import Base

# ── Import every model module so its tables register on Base.metadata.
# Autogenerate only sees what's imported here.
from sign_language.auth import models as _models  # noqa: F401,E402

# ── Alembic config ──────────────────────────────────────────────────
config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Generate SQL scripts without a live DB connection.

    Configures the Alembic context with the database URL directly so
    that migration scripts can be produced for review or manual
    execution without connecting to the database.
    """
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Apply migrations using an existing synchronous connection.

    Called from :func:`run_async_migrations` via
    ``connection.run_sync()`` so that the async engine can hand off to
    Alembic's synchronous migration runner.

    Args:
        connection: An active synchronous SQLAlchemy connection provided
            by the async engine bridge.
    """
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations against the live database.

    Uses ``NullPool`` to ensure the engine does not retain connections
    after the migration run completes — important for short-lived CLI
    processes.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    """Entry point for online migration mode.

    Wraps :func:`run_async_migrations` in ``asyncio.run()`` so that the
    async migration coroutine can be executed from the synchronous
    Alembic CLI context.
    """
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
