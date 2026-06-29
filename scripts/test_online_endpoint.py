"""Run synthetic end-to-end checks against an Azure ML online endpoint."""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

import typer
from PIL import Image

from sign_language_azure_api.settings import settings as azure_api_settings


app = typer.Typer(add_completion=False)


def _make_test_image() -> str:
    """Create a deterministic synthetic image for endpoint testing.

    Returns:
        Base64-encoded PNG image.
    """
    image = Image.new("RGB", (224, 224), color=(120, 80, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _post(
    endpoint_url: str,
    endpoint_key: str,
    payload: dict,
    deployment_name: str | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[int, dict, float]:
    """Send one request directly to an Azure ML online endpoint.

    Args:
        endpoint_url: Azure ML scoring URI.
        endpoint_key: Endpoint authentication key.
        payload: JSON request payload.
        deployment_name: Optional deployment targeted by the request.
        timeout_seconds: Maximum request duration.

    Returns:
        HTTP status, parsed response body, and elapsed seconds.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {endpoint_key}",
    }
    if deployment_name:
        headers["azureml-model-deployment"] = deployment_name

    request = urllib.request.Request(
        endpoint_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    elapsed = time.perf_counter() - start

    try:
        parsed = json.loads(body)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
    except json.JSONDecodeError:
        parsed = {"raw": body}
    return status, parsed, elapsed


@app.command()
def main(
    endpoint_url: str | None = typer.Option(None, "--endpoint-url"),
    endpoint_key: str | None = typer.Option(None, "--endpoint-key"),
    deployment_name: str | None = typer.Option(None, "--deployment-name"),
    max_latency_seconds: float = typer.Option(30.0, "--max-latency-seconds"),
) -> None:
    """Run valid, invalid, and missing-field checks."""
    resolved_endpoint_url = (
        endpoint_url or azure_api_settings.azure_api_online_endpoint_url
    )
    resolved_endpoint_key = (
        endpoint_key or azure_api_settings.azure_api_online_endpoint_key
    )
    resolved_deployment_name = (
        deployment_name or azure_api_settings.azure_api_default_deployment
    )

    missing_config = []
    if not resolved_endpoint_url:
        missing_config.append(
            "endpoint URL (--endpoint-url or AZURE_API_ONLINE_ENDPOINT_URL)"
        )
    if not resolved_endpoint_key:
        missing_config.append(
            "endpoint key (--endpoint-key or AZURE_API_ONLINE_ENDPOINT_KEY)"
        )
    if missing_config:
        typer.echo(
            "Skipping live Azure ML endpoint checks: missing "
            + " and ".join(missing_config)
            + "."
        )
        raise typer.Exit(code=0)

    status, body, elapsed = _post(
        resolved_endpoint_url,
        resolved_endpoint_key,
        {"image": _make_test_image()},
        resolved_deployment_name,
    )
    if status != 200:
        raise typer.BadParameter(f"Valid request failed with HTTP {status}: {body}")
    for key in ("predicted_letter", "confidence", "top_3"):
        if key not in body:
            raise typer.BadParameter(f"Valid response missing '{key}': {body}")
    if elapsed > max_latency_seconds:
        raise typer.BadParameter(
            f"Valid request latency {elapsed:.2f}s exceeded {max_latency_seconds:.2f}s"
        )

    _, invalid_body, _ = _post(
        resolved_endpoint_url,
        resolved_endpoint_key,
        {"image": "not-base64"},
        resolved_deployment_name,
    )
    if "error" not in invalid_body:
        raise typer.BadParameter(
            f"Invalid image did not return an error: {invalid_body}"
        )

    _, missing_body, _ = _post(
        resolved_endpoint_url,
        resolved_endpoint_key,
        {},
        resolved_deployment_name,
    )
    if "error" not in missing_body:
        raise typer.BadParameter(
            f"Missing image field did not return an error: {missing_body}"
        )

    typer.echo(
        f"Online endpoint tests passed in {elapsed:.2f}s. "
        f"Response saved model_version={body.get('model_version')}"
    )


if __name__ == "__main__":
    app()
