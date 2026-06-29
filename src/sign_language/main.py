"""Entry point for the sign-language CLI command."""

from sign_language.cli import app


def main() -> None:
    """Main entry point registered in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
