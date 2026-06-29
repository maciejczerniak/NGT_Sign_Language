"""CLI entrypoint for the standalone Azure ML endpoint API."""

from __future__ import annotations

import typer
import uvicorn

from sign_language_azure_api.settings import settings


app = typer.Typer(add_completion=False)


@app.command()
def serve(
    host: str = typer.Option(None, "--host"),
    port: int = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Serve the standalone Azure ML endpoint API.

    Args:
        host: Optional bind host overriding application settings.
        port: Optional bind port overriding application settings.
        reload: Enable Uvicorn development reload mode.
    """
    uvicorn.run(
        "sign_language_azure_api.app:create_app",
        factory=True,
        host=host or settings.azure_api_host,
        port=port or settings.azure_api_port,
        reload=reload,
    )


def main() -> None:
    """Run the standalone Azure API command-line application.

    Returns:
        None.
    """
    app()


if __name__ == "__main__":
    main()
