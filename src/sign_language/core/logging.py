"""Logging configuration and logger factory.

Provides a centralised :func:`get_logger` factory that configures file
and optional console handlers based on project settings. Handlers are
reused if already attached to a logger to avoid duplicate log entries.
"""

import logging
from pathlib import Path
from typing import Optional

from sign_language.core.settings import Settings, settings


def _get_log_formatter() -> logging.Formatter:
    """Create and return the standard log formatter.

    :returns: A :class:`~logging.Formatter` producing lines in the format
        ``asctime - name - levelname - message``.
    """
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def _get_log_path(settings: Settings) -> Path:
    """Resolve the log file path from settings and ensure its parent directory exists.

    :param settings: The :class:`~sign_language.core.settings.Settings` instance
        to read logging configuration from.
    :returns: The resolved :class:`~pathlib.Path` to the log file.
    :raises ValueError: If ``log_path`` is not defined in the logging config.
    """
    raw_path = settings.get_logging_config().get("path")
    if raw_path is None:
        raise ValueError("log_path must be defined")

    log_path = Path(raw_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def _get_existing_file_handler(
    logger: logging.Logger, log_path: Path
) -> Optional[logging.FileHandler]:
    """Return an existing file handler on ``logger`` pointing to ``log_path``, if any.

    Compares resolved absolute paths to avoid duplicate handlers when the
    logger is retrieved multiple times.

    :param logger: The :class:`~logging.Logger` instance to inspect.
    :param log_path: The target log file path to match against.
    :returns: The matching :class:`~logging.FileHandler` if found,
        otherwise ``None``.
    """
    resolved_log_path = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_filename = getattr(handler, "baseFilename", None)
            if (
                base_filename is not None
                and str(Path(base_filename).resolve()) == resolved_log_path
            ):
                return handler
    return None


def _get_existing_console_handler(
    logger: logging.Logger,
) -> Optional[logging.StreamHandler]:
    """Return an existing console (non-file) stream handler on ``logger``, if any.

    :param logger: The :class:`~logging.Logger` instance to inspect.
    :returns: The first :class:`~logging.StreamHandler` that is not a
        :class:`~logging.FileHandler`, or ``None`` if none exists.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            return handler
    return None


def get_logger(
    name: Optional[str] = None, settings: Settings = settings
) -> logging.Logger:
    """Create or retrieve a configured logger with file and optional console handlers.

    Configures the logger with the log level from settings. Attaches a file
    handler if one pointing to the configured log path is not already present.
    In development environments, also attaches a console handler if one is
    not already present.

    Existing handlers are reused and reconfigured rather than duplicated.

    :param name: Logger name passed to :func:`logging.getLogger`. If ``None``,
        returns the root logger.
    :param settings: The :class:`~sign_language.core.settings.Settings` instance
        used to resolve log level, log path, and environment info.
    :returns: A configured :class:`~logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.get_log_level())
    logger.propagate = False

    formatter = _get_log_formatter()
    log_path = _get_log_path(settings)

    file_handler = _get_existing_file_handler(logger, log_path)
    if file_handler is None:
        file_handler = get_log_file_handler(settings)
        logger.addHandler(file_handler)
    else:
        file_handler.setLevel(settings.get_log_level())
        file_handler.setFormatter(formatter)

    if settings.get_environment_info().get("is_development", False):
        console_handler = _get_existing_console_handler(logger)
        if console_handler is None:
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)
        console_handler.setLevel(settings.get_log_level())
        console_handler.setFormatter(formatter)

    return logger


def get_log_file_handler(settings: Settings = settings) -> logging.FileHandler:
    """Create and return a configured file handler for the log path from settings.

    :param settings: The :class:`~sign_language.core.settings.Settings` instance
        used to resolve the log file path and log level.
    :returns: A :class:`~logging.FileHandler` configured with the standard
        formatter and the log level from settings.
    """
    log_path = _get_log_path(settings)

    handler = logging.FileHandler(log_path)
    handler.setLevel(settings.get_log_level())
    handler.setFormatter(_get_log_formatter())
    return handler
