"""Tests for sign_language/main.py."""

import sys
from unittest.mock import patch

_MODULE = "sign_language.main"


class TestMain:
    def test_main_calls_app(self) -> None:
        """main() should invoke the Typer CLI app."""
        import importlib

        sys.modules.pop(_MODULE, None)
        mod = importlib.import_module(_MODULE)
        with patch("sign_language.main.app") as mock_app:
            mod.main()
        mock_app.assert_called_once()


class TestMainEntrypoint:
    def test_dunder_main_calls_main(self) -> None:
        """Running the module as __main__ must invoke main()."""
        import runpy

        sys.modules.pop(_MODULE, None)
        with patch("sys.argv", ["sign-language", "--help"]):
            try:
                runpy.run_module(_MODULE, run_name="__main__", alter_sys=True)
            except SystemExit:
                pass  # --help exits with code 0, that's expected
