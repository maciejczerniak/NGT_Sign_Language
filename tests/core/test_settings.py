import logging
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_VALID = dict(
    app_name="Test App",
    version="1.0.0",
    authors=["Alice"],
    authors_email=["alice@example.com"],
    status="DEVELOPMENT",
    cors_origins=None,
    log_path=Path(tempfile.gettempdir()) / "sign_language_tests" / "app.log",
)


def make_settings(**overrides):
    """Import Settings fresh without loading the developer's repository .env."""
    from sign_language.core.settings import Settings

    return Settings(_env_file=None, **{**BASE_VALID, **overrides})


# ---------------------------------------------------------------------------
# Settings file discovery
# ---------------------------------------------------------------------------


def test_dotenv_path_points_to_repository_root():
    """Verify settings load .env from the repository root."""
    from sign_language.core.settings import BASE_DIR, DOTENV

    assert DOTENV == BASE_DIR / ".env"


# ---------------------------------------------------------------------------
# Debug validator
# ---------------------------------------------------------------------------


class TestDebugValidator:
    @pytest.mark.parametrize(
        ("raw_value", "expected"),
        [
            ("release", False),
            ("production", False),
            ("dev", True),
            ("on", True),
        ],
    )
    def test_debug_accepts_environment_aliases(self, raw_value, expected):
        """Verify debug validator debug accepts environment aliases."""
        s = make_settings(status="DEVELOPMENT", debug=raw_value)
        assert s.debug is expected

    def test_debug_defaults_true_in_development(self):
        """Verify debug validator debug defaults true in development."""
        s = make_settings(status="DEVELOPMENT", debug=None)
        assert s.debug is True

    def test_debug_defaults_false_in_production(self):
        """Verify debug validator debug defaults false in production."""
        s = make_settings(status="PRODUCTION", debug=None)
        assert s.debug is False

    def test_debug_explicit_false_in_production(self):
        """Verify debug validator debug explicit false in production."""
        s = make_settings(status="PRODUCTION", debug=False)
        assert s.debug is False

    def test_debug_true_in_development(self):
        """Verify debug validator debug true in development."""
        s = make_settings(status="DEVELOPMENT", debug=True)
        assert s.debug is True

    def test_debug_true_in_production_raises(self):
        """Verify debug validator debug true in production raises."""
        with pytest.raises(ValidationError, match="Debug mode cannot be enabled"):
            make_settings(status="PRODUCTION", debug=True)


# ---------------------------------------------------------------------------
# Log level validator
# ---------------------------------------------------------------------------


class TestLogLevelValidator:
    def test_defaults_to_debug_when_debug_true(self):
        """Verify log level validator defaults to debug when debug true."""
        s = make_settings(status="DEVELOPMENT", debug=True, log_level=None)
        assert s.log_level == "DEBUG"

    def test_defaults_to_info_when_debug_false(self):
        """Verify log level validator defaults to info when debug false."""
        s = make_settings(status="PRODUCTION", debug=False, log_level=None)
        assert s.log_level == "INFO"

    def test_explicit_valid_level(self):
        """Verify log level validator explicit valid level."""
        for level in [
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARNING",
            "WARN",
            "INFO",
            "DEBUG",
            "NOTSET",
        ]:
            s = make_settings(log_level=level)
            assert s.log_level == level

    def test_invalid_level_raises(self):
        """Verify log level validator invalid level raises."""
        with pytest.raises(ValidationError, match="Invalid log level"):
            make_settings(log_level="VERBOSE")


# ---------------------------------------------------------------------------
# Version validator
# ---------------------------------------------------------------------------


class TestVersionValidator:
    @pytest.mark.parametrize(
        "v", ["1.0.0", "0.1.0", "2.3.4-rc1", "1.0.0+build42", "1.0.0-alpha.1+001"]
    )
    def test_valid_versions(self, v):
        """Verify version validator valid versions."""
        s = make_settings(version=v)
        assert s.version == v

    @pytest.mark.parametrize("v", ["1.0", "v1.0.0", "1.0.0.0", "abc"])
    def test_invalid_versions_raise(self, v):
        """Verify version validator invalid versions raise."""
        with pytest.raises(ValidationError, match="semantic versioning"):
            make_settings(version=v)


# ---------------------------------------------------------------------------
# Authors validator
# ---------------------------------------------------------------------------


class TestAuthorsValidator:
    def test_valid_authors(self):
        """Verify authors validator valid authors."""
        s = make_settings(
            authors=["Alice", "Bob"], authors_email=["a@x.com", "b@x.com"]
        )
        assert len(s.authors) == 2

    def test_empty_authors_raises(self):
        """Verify authors validator empty authors raises."""
        with pytest.raises(ValidationError, match="At least one author"):
            make_settings(authors=[], authors_email=[])

    def test_blank_author_name_raises(self):
        """Verify authors validator blank author name raises."""
        with pytest.raises(ValidationError, match="Author names cannot be empty"):
            make_settings(authors=["  "], authors_email=["a@x.com"])


# ---------------------------------------------------------------------------
# Email validator
# ---------------------------------------------------------------------------


class TestEmailValidator:
    def test_mismatched_count_raises(self):
        """Verify email validator mismatched count raises."""
        with pytest.raises(ValidationError, match="must match number of authors"):
            make_settings(authors=["Alice", "Bob"], authors_email=["a@x.com"])

    def test_invalid_email_format_raises(self):
        """Verify email validator invalid email format raises."""
        with pytest.raises(ValidationError, match="Invalid email format"):
            make_settings(authors=["Alice"], authors_email=["not-an-email"])

    def test_valid_emails(self):
        """Verify email validator valid emails."""
        s = make_settings(authors=["Alice"], authors_email=["alice@domain.co.uk"])
        assert s.authors_email == ["alice@domain.co.uk"]


# ---------------------------------------------------------------------------
# Log path validator
# ---------------------------------------------------------------------------


class TestLogPathValidator:
    def test_does_not_create_directory_during_validation(self, tmp_path):
        """Verify validation has no filesystem side effects."""
        log_file = tmp_path / "nested" / "dir" / "app.log"
        s = make_settings(log_path=log_file)
        assert not log_file.parent.exists()
        assert s.log_path == log_file

    def test_existing_directory_is_fine(self, tmp_path):
        """Verify log path validator existing directory is fine."""
        log_file = tmp_path / "app.log"
        s = make_settings(log_path=log_file)
        assert s.log_path == log_file


# ---------------------------------------------------------------------------
# CORS origins validator
# ---------------------------------------------------------------------------


class TestCorsOriginsValidator:
    def test_none_returns_empty_list(self):
        """Verify cors origins validator none returns empty list."""
        s = make_settings(cors_origins=None)
        assert s.cors_origins == []

    def test_comma_separated_string(self):
        """Verify cors origins validator comma separated string."""
        s = make_settings(cors_origins="http://localhost:3000, http://localhost:5173")
        assert s.cors_origins == ["http://localhost:3000", "http://localhost:5173"]

    def test_list_passthrough(self):
        """Verify cors origins validator list passthrough."""
        origins = ["http://a.com", "http://b.com"]
        s = make_settings(cors_origins=origins)
        assert s.cors_origins == origins

    def test_json_array_string_decodes_to_list(self):
        """Verify cors origins validator json array string decodes to list."""
        s = make_settings(cors_origins='["http://a.com"]')
        assert s.cors_origins == ["http://a.com"]

    def test_invalid_json_array_string_falls_back_to_csv(self):
        """Verify cors origins validator invalid json array string falls back to csv."""
        s = make_settings(cors_origins="[not-json")
        assert s.cors_origins == ["[not-json"]

    def test_non_string_or_list_raises(self):
        """Verify cors origins validator non string or list raises."""
        with pytest.raises(ValidationError, match="cors_origins must be"):
            make_settings(cors_origins=123)


# ---------------------------------------------------------------------------
# Training validators
# ---------------------------------------------------------------------------


class TestTrainingValidators:
    @pytest.mark.parametrize(
        "field",
        [
            "training_img_size",
            "training_batch_size",
            "training_epochs",
            "training_patience",
            "training_expected_num_classes",
        ],
    )
    def test_positive_integer_training_settings_raise_for_zero(self, field):
        """Verify training validators positive integer training settings raise for zero."""
        with pytest.raises(ValidationError, match="must be positive"):
            make_settings(**{field: 0})

    @pytest.mark.parametrize("field", ["training_learning_rate"])
    def test_positive_float_training_settings_raise_for_zero(self, field):
        """Verify training validators positive float training settings raise for zero."""
        with pytest.raises(ValidationError, match="must be positive"):
            make_settings(**{field: 0.0})

    def test_training_eta_min_allows_zero(self):
        """Verify cosine annealing can decay fully to zero."""
        s = make_settings(training_eta_min=0.0)
        assert s.training_eta_min == 0.0

    def test_training_eta_min_rejects_negative(self):
        """Verify eta_min cannot be negative."""
        with pytest.raises(ValidationError, match="cannot be negative"):
            make_settings(training_eta_min=-0.1)

    @pytest.mark.parametrize(
        "field", ["training_val_split", "training_target_accuracy"]
    )
    def test_training_ratio_settings_must_be_between_zero_and_one(self, field):
        """Verify training validators training ratio settings must be between zero and one."""
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            make_settings(**{field: 1.0})

    def test_training_n_splits_must_remain_one(self):
        """Verify training validators training n splits must remain one."""
        with pytest.raises(ValidationError, match="Number of splits must be exactly 1"):
            make_settings(training_n_splits=2)

    def test_training_num_workers_cannot_be_negative(self):
        """Verify training validators training num workers cannot be negative."""
        with pytest.raises(
            ValidationError, match="Number of workers cannot be negative"
        ):
            make_settings(training_num_workers=-1)


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------


class TestGetters:
    def setup_method(self):
        self.s = make_settings(status="DEVELOPMENT", debug=True, log_level="DEBUG")

    def test_get_debug_state(self):
        """Verify getters get debug state."""
        assert self.s.get_debug_state() is True

    def test_get_log_level_returns_int(self):
        """Verify getters get log level returns int."""
        assert self.s.get_log_level() == logging.DEBUG

    def test_get_log_level_all_values(self):
        """Verify getters get log level all values."""
        expected = {
            "CRITICAL": logging.CRITICAL,
            "FATAL": logging.FATAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }
        for level_str, level_int in expected.items():
            s = make_settings(log_level=level_str)
            assert s.get_log_level() == level_int

    def test_get_log_level_fallback_none(self):
        # When log_level resolves to INFO via validator
        """Verify getters get log level fallback none."""
        s = make_settings(status="PRODUCTION", debug=False, log_level=None)
        assert s.get_log_level() == logging.INFO

    def test_get_app_info(self):
        """Verify getters get app info."""
        info = self.s.get_app_info()
        assert info["name"] == "Test App"
        assert info["version"] == "1.0.0"
        assert info["environment"] == "DEVELOPMENT"
        assert info["debug"] is True

    # Remove stale test that referenced a non-existent method
    # test_get_sign_language_info was generated incorrectly - covered by test_get_app_info above

    def test_get_authors_with_emails(self):
        """Verify getters get authors with emails."""
        result = self.s.get_authors_with_emails()
        assert result == [{"name": "Alice", "email": "alice@example.com"}]

    def test_get_logging_config(self):
        """Verify getters get logging config."""
        config = self.s.get_logging_config()
        assert config["level"] == "DEBUG"
        assert config["level_int"] == logging.DEBUG
        assert "app.log" in config["path"]

    def test_is_development(self):
        """Verify getters is development."""
        assert self.s.is_development() is True
        assert self.s.is_production() is False

    def test_is_production(self):
        """Verify getters is production."""
        s = make_settings(status="PRODUCTION", debug=False)
        assert s.is_production() is True
        assert s.is_development() is False

    def test_as_dict(self):
        """Verify getters as dict."""
        d = self.s.as_dict()
        assert d["app_name"] == "Test App"
        assert d["status"] == "DEVELOPMENT"
        assert d["training"]["batch_size"] == 16
        assert d["training"]["epochs"] == 30

    def test_training_settings_are_configurable(self):
        """Verify getters training settings are configurable."""
        s = make_settings(
            training_batch_size=8,
            training_epochs=2,
            training_learning_rate=0.001,
        )
        assert s.training_batch_size == 8
        assert s.training_epochs == 2
        assert s.get_training_config()["learning_rate"] == 0.001
        assert s.get_training_config()["expected_num_classes"] == 22

    def test_get_mlflow_config(self):
        """Verify getters get mlflow config."""
        s = make_settings(
            mlflow_enabled=True,
            mlflow_tracking_uri="file:./mlruns",
            mlflow_experiment_name="example-project",
            mlflow_run_name="example-run",
            mlflow_autolog=False,
            mlflow_log_artifacts=False,
        )
        assert s.get_mlflow_config() == {
            "enabled": True,
            "tracking_uri": "file:./mlruns",
            "experiment_name": "example-project",
            "run_name": "example-run",
            "autolog": False,
            "log_artifacts": False,
        }

    def test_get_environment_info(self):
        """Verify getters get environment info."""
        info = self.s.get_environment_info()
        assert info["is_development"] is True
        assert info["is_production"] is False
        assert "python_version" in info
        assert "platform" in info

    def test_get_device_prefers_cuda(self):
        """Verify getters get device prefers cuda."""
        with patch(
            "sign_language.core.settings.torch.cuda.is_available", return_value=True
        ):
            assert self.s.get_device().type == "cuda"

    def test_get_device_falls_back_to_mps(self):
        """Verify getters get device falls back to mps."""
        with (
            patch(
                "sign_language.core.settings.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "sign_language.core.settings.torch.backends.mps.is_available",
                return_value=True,
            ),
        ):
            assert self.s.get_device().type == "mps"

    def test_get_device_falls_back_to_cpu(self):
        """Verify getters get device falls back to cpu."""
        with (
            patch(
                "sign_language.core.settings.torch.cuda.is_available",
                return_value=False,
            ),
            patch(
                "sign_language.core.settings.torch.backends.mps.is_available",
                return_value=False,
            ),
        ):
            assert self.s.get_device().type == "cpu"


# ---------------------------------------------------------------------------
# Version components
# ---------------------------------------------------------------------------


class TestVersionComponents:
    def test_simple_version(self):
        """Verify version components simple version."""
        s = make_settings(version="1.2.3")
        c = s.get_version_components()
        assert c == {
            "major": "1",
            "minor": "2",
            "patch": "3",
            "prerelease": "",
            "build": "",
        }

    def test_version_with_prerelease(self):
        """Verify version components version with prerelease."""
        s = make_settings(version="1.2.3-rc1")
        c = s.get_version_components()
        assert c["prerelease"] == "rc1"
        assert c["build"] == ""

    def test_version_with_build(self):
        """Verify version components version with build."""
        s = make_settings(version="1.2.3+build99")
        c = s.get_version_components()
        assert c["build"] == "build99"

    def test_version_with_prerelease_and_build(self):
        """Verify version components version with prerelease and build."""
        s = make_settings(version="1.2.3-alpha+001")
        c = s.get_version_components()
        assert c["prerelease"] == "alpha"
        assert c["build"] == "001"

    def test_invalid_runtime_version_returns_empty_components(self):
        """Verify version components invalid runtime version returns empty components."""
        s = make_settings()
        object.__setattr__(s, "version", "invalid")

        assert s.get_version_components() == {
            "major": "",
            "minor": "",
            "patch": "",
            "prerelease": "",
            "build": "",
        }


# ---------------------------------------------------------------------------
# get_settings factory
# ---------------------------------------------------------------------------


class TestGetSettingsFactory:
    def test_default_returns_settings_instance(self):
        """Verify get settings factory default returns settings instance."""
        from sign_language.core.settings import Settings, get_settings

        s = get_settings()
        assert isinstance(s, Settings)

    def test_use_test_env_returns_test_settings(self, tmp_path):
        """Verify get settings factory use test env returns test settings."""
        from sign_language.core import settings as settings_module

        fake_default_env = tmp_path / ".env"
        fake_default_env.write_text(
            "APP_NAME=DefaultApp\nVERSION=1.0.0\nSTATUS=DEVELOPMENT\n"
            'AUTHORS=["Default Tester"]\nAUTHORS_EMAIL=["default@t.com"]\n'
        )
        fake_test_env = tmp_path / ".env.test"
        fake_test_env.write_text(
            "APP_NAME=TestApp\nVERSION=1.0.0\nSTATUS=DEVELOPMENT\n"
            'AUTHORS=["Tester"]\nAUTHORS_EMAIL=["t@t.com"]\n'
        )
        with patch.object(
            settings_module.Path,
            "resolve",
            return_value=fake_default_env,
        ):
            s = settings_module.get_settings(use_test_env=True)
        from sign_language.core.settings import Settings

        assert isinstance(s, Settings)
        assert s.app_name == "TestApp"


def test_settings_module_main_prints_settings(capsys):
    """Verify settings module main prints settings."""
    runpy.run_module("sign_language.core.settings", run_name="__main__")

    captured = capsys.readouterr()
    assert "app_name='Sign Language'" in captured.out
    assert "training_batch_size=16" in captured.out


# ---------------------------------------------------------------------------
# Server settings validators
# ---------------------------------------------------------------------------


class TestServerValidators:
    def test_default_host_is_all_interfaces(self):
        """Verify server validators default host is all interfaces."""
        s = make_settings()
        assert s.server_host == "0.0.0.0"

    def test_default_port_is_8000(self):
        """Verify server validators default port is 8000."""
        s = make_settings()
        assert s.server_port == 8000

    def test_custom_host_accepted(self):
        """Verify server validators custom host accepted."""
        s = make_settings(server_host="127.0.0.1")
        assert s.server_host == "127.0.0.1"

    def test_empty_host_raises(self):
        """Verify server validators empty host raises."""
        with pytest.raises(ValidationError, match="server_host cannot be empty"):
            make_settings(server_host="")

    def test_whitespace_only_host_raises(self):
        """Verify server validators whitespace only host raises."""
        with pytest.raises(ValidationError, match="server_host cannot be empty"):
            make_settings(server_host="   ")

    def test_custom_port_accepted(self):
        """Verify server validators custom port accepted."""
        s = make_settings(server_port=9000)
        assert s.server_port == 9000

    def test_port_zero_raises(self):
        """Verify server validators port zero raises."""
        with pytest.raises(ValidationError, match="between 1 and 65535"):
            make_settings(server_port=0)

    def test_port_too_high_raises(self):
        """Verify server validators port too high raises."""
        with pytest.raises(ValidationError, match="between 1 and 65535"):
            make_settings(server_port=65536)

    def test_port_boundary_low(self):
        """Verify server validators port boundary low."""
        s = make_settings(server_port=1)
        assert s.server_port == 1

    def test_port_boundary_high(self):
        """Verify server validators port boundary high."""
        s = make_settings(server_port=65535)
        assert s.server_port == 65535


# ---------------------------------------------------------------------------
# Inference threshold validators
# ---------------------------------------------------------------------------


class TestInferenceValidators:
    def test_default_efficientnet_threshold(self):
        """Verify inference validators default efficientnet threshold."""
        s = make_settings()
        assert s.efficientnet_confidence_threshold == pytest.approx(0.70)

    def test_default_landmark_override_threshold(self):
        """Verify inference validators default landmark override threshold."""
        s = make_settings()
        assert s.landmark_override_threshold == pytest.approx(0.90)

    @pytest.mark.parametrize(
        "field",
        ["efficientnet_confidence_threshold", "landmark_override_threshold"],
    )
    def test_zero_raises(self, field):
        """Verify inference validators zero raises."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            make_settings(**{field: 0.0})

    @pytest.mark.parametrize(
        "field",
        ["efficientnet_confidence_threshold", "landmark_override_threshold"],
    )
    def test_one_raises(self, field):
        """Verify inference validators one raises."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            make_settings(**{field: 1.0})

    @pytest.mark.parametrize(
        "field",
        ["efficientnet_confidence_threshold", "landmark_override_threshold"],
    )
    def test_valid_mid_range(self, field):
        """Verify inference validators valid mid range."""
        s = make_settings(**{field: 0.5})
        assert getattr(s, field) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Sequence builder timing validators
# ---------------------------------------------------------------------------


class TestSequenceValidators:
    def test_defaults(self):
        """Verify sequence validators defaults."""
        s = make_settings()
        assert s.sequence_letter_hold_sec == pytest.approx(1.0)
        assert s.sequence_cooldown_sec == pytest.approx(1.0)
        assert s.sequence_space_pause_sec == pytest.approx(1.5)
        assert s.sequence_stable_grace_sec == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "field",
        [
            "sequence_letter_hold_sec",
            "sequence_cooldown_sec",
            "sequence_space_pause_sec",
            "sequence_stable_grace_sec",
        ],
    )
    def test_zero_raises(self, field):
        """Verify sequence validators zero raises."""
        with pytest.raises(ValidationError, match="must be positive"):
            make_settings(**{field: 0.0})

    @pytest.mark.parametrize(
        "field",
        [
            "sequence_letter_hold_sec",
            "sequence_cooldown_sec",
            "sequence_space_pause_sec",
            "sequence_stable_grace_sec",
        ],
    )
    def test_negative_raises(self, field):
        """Verify sequence validators negative raises."""
        with pytest.raises(ValidationError, match="must be positive"):
            make_settings(**{field: -0.5})

    @pytest.mark.parametrize(
        "field",
        [
            "sequence_letter_hold_sec",
            "sequence_cooldown_sec",
            "sequence_space_pause_sec",
            "sequence_stable_grace_sec",
        ],
    )
    def test_positive_accepted(self, field):
        """Verify sequence validators positive accepted."""
        s = make_settings(**{field: 2.5})
        assert getattr(s, field) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Prediction smoother validators
# ---------------------------------------------------------------------------


class TestSmootherValidators:
    def test_defaults(self):
        """Verify smoother validators defaults."""
        s = make_settings()
        assert s.smoother_window_size == 15
        assert s.smoother_acquire_threshold == 10
        assert s.smoother_sticky_threshold == 7
        assert s.smoother_min_confidence == pytest.approx(0.55)

    @pytest.mark.parametrize(
        "field",
        [
            "smoother_window_size",
            "smoother_acquire_threshold",
            "smoother_sticky_threshold",
        ],
    )
    def test_zero_raises(self, field):
        """Verify smoother validators zero raises."""
        with pytest.raises(ValidationError, match="at least 1"):
            make_settings(**{field: 0})

    @pytest.mark.parametrize(
        "field",
        [
            "smoother_window_size",
            "smoother_acquire_threshold",
            "smoother_sticky_threshold",
        ],
    )
    def test_positive_accepted(self, field):
        """Verify smoother validators positive accepted."""
        s = make_settings(**{field: 5})
        assert getattr(s, field) == 5

    def test_min_confidence_zero_raises(self):
        """Verify smoother validators min confidence zero raises."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            make_settings(smoother_min_confidence=0.0)

    def test_min_confidence_one_raises(self):
        """Verify smoother validators min confidence one raises."""
        with pytest.raises(ValidationError, match="between 0 and 1"):
            make_settings(smoother_min_confidence=1.0)

    def test_min_confidence_valid(self):
        """Verify smoother validators min confidence valid."""
        s = make_settings(smoother_min_confidence=0.6)
        assert s.smoother_min_confidence == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# New getter methods
# ---------------------------------------------------------------------------


class TestNewGetters:
    def setup_method(self):
        self.s = make_settings()

    def test_get_server_config_keys(self):
        """Verify new getters get server config keys."""
        cfg = self.s.get_server_config()
        assert set(cfg.keys()) == {"host", "port", "reload"}

    def test_get_server_config_defaults(self):
        """Verify new getters get server config defaults."""
        cfg = self.s.get_server_config()
        assert cfg["host"] == "0.0.0.0"
        assert cfg["port"] == 8000

    def test_get_server_config_reload_matches_is_development(self):
        """Verify new getters get server config reload matches is development."""
        cfg = self.s.get_server_config()
        assert cfg["reload"] == self.s.is_development()

    def test_get_server_config_reload_false_in_production(self):
        """Verify new getters get server config reload false in production."""
        s = make_settings(status="PRODUCTION", debug=False)
        assert s.get_server_config()["reload"] is False

    def test_get_inference_config_keys(self):
        """Verify new getters get inference config keys."""
        cfg = self.s.get_inference_config()
        assert set(cfg.keys()) == {
            "efficientnet_confidence_threshold",
            "landmark_override_threshold",
        }

    def test_get_inference_config_values(self):
        """Verify new getters get inference config values."""
        s = make_settings(
            efficientnet_confidence_threshold=0.65,
            landmark_override_threshold=0.85,
        )
        cfg = s.get_inference_config()
        assert cfg["efficientnet_confidence_threshold"] == pytest.approx(0.65)
        assert cfg["landmark_override_threshold"] == pytest.approx(0.85)

    def test_get_sequence_config_keys(self):
        """Verify new getters get sequence config keys."""
        cfg = self.s.get_sequence_config()
        assert set(cfg.keys()) == {
            "letter_hold_sec",
            "cooldown_sec",
            "space_pause_sec",
            "stable_grace_sec",
        }

    def test_get_sequence_config_values(self):
        """Verify new getters get sequence config values."""
        s = make_settings(
            sequence_letter_hold_sec=0.8,
            sequence_cooldown_sec=1.2,
            sequence_space_pause_sec=2.0,
            sequence_stable_grace_sec=0.3,
        )
        cfg = s.get_sequence_config()
        assert cfg["letter_hold_sec"] == pytest.approx(0.8)
        assert cfg["cooldown_sec"] == pytest.approx(1.2)
        assert cfg["space_pause_sec"] == pytest.approx(2.0)
        assert cfg["stable_grace_sec"] == pytest.approx(0.3)

    def test_get_smoother_config_keys(self):
        """Verify new getters get smoother config keys."""
        cfg = self.s.get_smoother_config()
        assert set(cfg.keys()) == {
            "window_size",
            "acquire_threshold",
            "sticky_threshold",
            "min_confidence",
        }

    def test_get_smoother_config_values(self):
        """Verify new getters get smoother config values."""
        s = make_settings(
            smoother_window_size=20,
            smoother_acquire_threshold=14,
            smoother_sticky_threshold=9,
            smoother_min_confidence=0.60,
        )
        cfg = s.get_smoother_config()
        assert cfg["window_size"] == 20
        assert cfg["acquire_threshold"] == 14
        assert cfg["sticky_threshold"] == 9
        assert cfg["min_confidence"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Model registry settings
# ---------------------------------------------------------------------------


class TestDeployTarget:
    """Tests for the deploy_target field."""

    def test_default_is_local(self):
        """deploy_target should default to 'local'."""
        s = make_settings()
        assert s.deploy_target == "local"

    def test_accepts_onprem(self):
        """deploy_target should accept 'onprem'."""
        s = make_settings(deploy_target="onprem")
        assert s.deploy_target == "onprem"

    def test_accepts_azure(self):
        """deploy_target should accept 'azure'."""
        s = make_settings(deploy_target="azure")
        assert s.deploy_target == "azure"

    def test_rejects_unknown_target(self):
        """deploy_target should reject values outside the Literal."""
        with pytest.raises(ValidationError):
            make_settings(deploy_target="kubernetes")


class TestModelRegistrySettings:
    """Tests for shared model registry cache + name settings."""

    def test_model_cache_dir_has_default(self):
        """model_cache_dir should have a default path."""
        s = make_settings()
        assert s.model_cache_dir is not None
        assert isinstance(s.model_cache_dir, Path)

    def test_model_cache_dir_can_be_overridden(self, tmp_path):
        """model_cache_dir should accept a custom path."""
        s = make_settings(model_cache_dir=tmp_path / "cache")
        assert s.model_cache_dir == tmp_path / "cache"

    def test_model_registry_name_default(self):
        """model_registry_name should default to ngt-sign-language."""
        s = make_settings()
        assert s.model_registry_name == "ngt-sign-language"

    def test_model_registry_name_can_be_overridden(self):
        """model_registry_name should accept a custom name."""
        s = make_settings(model_registry_name="my-model")
        assert s.model_registry_name == "my-model"
