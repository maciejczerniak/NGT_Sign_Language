"""Tests for the logging configuration module."""

import logging
import pytest
from sign_language.core.logging import get_logger, get_log_file_handler
from sign_language.core.settings import Settings


@pytest.fixture
def dev_settings(tmp_path):
    """Settings instance in development mode with a temp log path."""
    return Settings(
        status="DEVELOPMENT",
        log_path=tmp_path / "logs" / "test.log",
        authors_email=[
            "242621@buas.nl",
            "243552@buas.nl",
            "242845@buas.nl",
            "241929@buas.nl",
            "241290@buas.nl",
        ],
    )


@pytest.fixture
def prod_settings(tmp_path):
    """Settings instance in production mode with a temp log path."""
    return Settings(
        status="PRODUCTION",
        debug=False,
        log_path=tmp_path / "logs" / "test.log",
        authors_email=[
            "242621@buas.nl",
            "243552@buas.nl",
            "242845@buas.nl",
            "241929@buas.nl",
            "241290@buas.nl",
        ],
    )


class TestGetLogFileHandler:
    def test_returns_file_handler(self, dev_settings):
        """Verify get log file handler returns file handler."""
        handler = get_log_file_handler(dev_settings)
        assert isinstance(handler, logging.FileHandler)
        handler.close()

    def test_creates_log_directory(self, dev_settings):
        """Verify get log file handler creates log directory."""
        handler = get_log_file_handler(dev_settings)
        assert dev_settings.log_path.parent.exists()
        handler.close()

    def test_raises_when_log_path_none(self, dev_settings, monkeypatch):
        # We patch the METHOD on the class, so the instance 'dev_settings'
        # will now return None when it calls it.
        """Verify get log file handler raises when log path none."""
        monkeypatch.setattr(Settings, "get_logging_config", lambda self: {"path": None})

        with pytest.raises(ValueError, match="log_path must be defined"):
            get_log_file_handler(dev_settings)


class TestGetLogger:
    def test_returns_logger(self, dev_settings):
        """Verify get logger returns logger."""
        logger = get_logger("test_logger", dev_settings)
        assert isinstance(logger, logging.Logger)

    def test_logger_has_file_handler(self, dev_settings):
        """Verify get logger logger has file handler."""
        logger = get_logger("test_file_handler", dev_settings)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in handler_types

    def test_development_adds_console_handler(self, dev_settings):
        """Verify get logger development adds console handler."""
        logger = get_logger("test_dev", dev_settings)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_production_no_console_handler(self, prod_settings):
        """Verify get logger production no console handler."""
        logger = get_logger("test_prod", prod_settings)
        stream_handlers = [
            h for h in logger.handlers if type(h) is logging.StreamHandler
        ]
        assert len(stream_handlers) == 0

    def test_logger_name(self, dev_settings):
        """Verify get logger logger name."""
        logger = get_logger("my_module", dev_settings)
        assert logger.name == "my_module"
