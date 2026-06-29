"""Invoke the Azure ML online endpoint using settings from .env."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import typer

from sign_language_azure_api.client import AzureMLEndpointClient
from sign_language_azure_api.settings import settings


app = typer.Typer(add_completion=False)


@app.command()
def main(
    image_path: Path = typer.Option(
        ...,
        "--image-path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Send one image to the configured Azure ML online endpoint."""
    endpoint_url = settings.azure_api_online_endpoint_url
    endpoint_key = settings.azure_api_online_endpoint_key
    deployment_name = settings.azure_api_default_deployment

    client = AzureMLEndpointClient(
        endpoint_url=endpoint_url,
        endpoint_key=endpoint_key,
        timeout_seconds=settings.azure_api_online_timeout_seconds,
    )
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    prediction = client.predict(image_data, deployment_name=deployment_name)

    typer.echo(
        json.dumps(
            {
                "predicted_letter": prediction.predicted_letter,
                "confidence": prediction.confidence,
                "top_3": prediction.top_3,
                "model_name": prediction.model_name,
                "model_version": prediction.model_version,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
