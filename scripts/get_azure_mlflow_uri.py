"""Print the Azure ML workspace MLflow tracking URI.

Useful for verifying Azure ML connectivity and copying the tracking URI
into local MLflow configuration or environment variables.

Typical usage::

    poetry run python scripts/get_azure_mlflow_uri.py
"""

import typer

from sign_language_training.azure_config import get_client, get_mlflow_tracking_uri

app = typer.Typer(add_completion=False)


@app.command()
def main() -> None:
    """Fetch and print the MLflow tracking URI for the configured Azure ML workspace.

    Creates an Azure ML client from project settings, resolves the workspace
    MLflow tracking URI, and prints it to stdout.

    Raises:
        RuntimeError: If the tracking URI cannot be resolved from the workspace.
        ValueError: If any required Azure setting is missing from .env.
    """
    ml_client = get_client()
    tracking_uri = get_mlflow_tracking_uri(ml_client)
    print(tracking_uri)


if __name__ == "__main__":
    app()
