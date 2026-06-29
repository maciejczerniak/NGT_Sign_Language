"""Register the raw NGT ImageFolder dataset as one Azure ML data asset.

Expected local structure::

    data/raw/
        A/
            A_train_001.png
            ...
        B/
        C/
        ...

Do not physically split the dataset here. The training workflow already performs a
deterministic stratified train/validation split from the raw ImageFolder.

Typical usage::

    poetry run python scripts/register_raw_data.py

Expected output::

    azureml:ngt-raw:1
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import get_client, settings  # noqa: E402
from sign_language_training.orchestration.dataset_inventory import (  # noqa: E402
    build_dataset_inventory,
)
from sign_language_training.orchestration import training_state  # noqa: E402

app = typer.Typer(
    name="register-raw-data",
    help="Register the raw NGT ImageFolder dataset as an Azure ML data asset.",
    add_completion=False,
)


@app.command()
def main(
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        help=(
            "Local raw ImageFolder directory used for metadata counting. "
            "Also used as the registered asset path when --data-uri is omitted. "
            "Defaults to TRAINING_LOCAL_DATA_DIR from settings."
        ),
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    data_uri: str | None = typer.Option(
        None,
        "--data-uri",
        help=(
            "Azure ML datastore URI for an existing raw ImageFolder, e.g. "
            "azureml://datastores/workspaceblobstore/paths/datasets/ngt/raw/current/"
        ),
    ),
    asset_name: str | None = typer.Option(
        None,
        "--asset-name",
        help="Azure ML data asset name. Defaults to AZURE_RAW_DATA_ASSET_NAME.",
    ),
    version: str | None = typer.Option(
        None,
        "--version",
        help="Azure ML data asset version. Defaults to AZURE_RAW_DATA_ASSET_VERSION.",
    ),
    image_count: int | None = typer.Option(
        None,
        "--image-count",
        min=0,
        help=(
            "Total image count metadata override. By default this is computed "
            "from --data-dir or TRAINING_LOCAL_DATA_DIR."
        ),
    ),
    manifest_hash: str | None = typer.Option(
        None,
        "--manifest-hash",
        help=(
            "Dataset manifest hash to store as metadata. Computed automatically "
            "for local --data-dir registrations."
        ),
    ),
) -> None:
    """Register the configured raw ImageFolder as an Azure ML data asset.

    Args:
        data_dir: Optional local raw ImageFolder directory used to compute
            metadata. If ``data_uri`` is omitted, this is also the registered
            asset path.
        data_uri: Optional Azure ML datastore URI for data already present in
            workspace storage.
        asset_name: Optional Azure ML data asset name override.
        version: Optional Azure ML data asset version override.
        image_count: Optional total image count metadata override. Defaults to
            the count computed from ``data_dir``.
        manifest_hash: Optional manifest hash metadata override. For local
            registrations this is computed from the files when omitted.

    Returns:
        None. Registered asset details are written to standard output.

    Raises:
        typer.BadParameter: If the local raw-data directory is missing.
    """
    ml_client = get_client()

    local_data_path = (
        data_dir or (REPO_ROOT / settings.training_local_data_dir).resolve()
    )
    data_path = data_uri or str(local_data_path)
    resolved_image_count = image_count
    resolved_manifest_hash = manifest_hash

    if resolved_image_count is None or resolved_manifest_hash is None:
        if local_data_path.is_dir():
            inventory = build_dataset_inventory(local_data_path)
            if resolved_image_count is None:
                resolved_image_count = inventory.image_count
            if resolved_manifest_hash is None:
                resolved_manifest_hash = inventory.manifest_hash
        elif data_uri is None:
            raise typer.BadParameter(
                f"Raw data directory not found: {local_data_path}",
                param_hint="--data-dir",
            )
        else:
            typer.echo(
                f"Warning: local metadata directory not found: {local_data_path}. "
                "Registered asset will not include computed image_count or "
                "manifest_hash unless passed explicitly.",
                err=True,
            )

    if data_uri is None:
        if not local_data_path.is_dir():
            raise typer.BadParameter(
                f"Raw data directory not found: {local_data_path}",
                param_hint="--data-dir",
            )
        data_path = str(local_data_path)

    if data_uri is not None and not data_uri.startswith("azureml://"):
        raise typer.BadParameter(
            "--data-uri must be an Azure ML datastore URI starting with azureml://",
            param_hint="--data-uri",
        )

    resolved_asset_name = asset_name or settings.azure_raw_data_asset_name
    resolved_version = version or settings.azure_raw_data_asset_version

    tags = {
        "project": "sign-language",
        "dataset": "ngt",
        "format": "imagefolder",
        "split_strategy": "stratified_split_inside_training",
        "registered_at": training_state.utc_now_iso(),
    }
    if resolved_image_count is not None:
        tags["image_count"] = str(resolved_image_count)
    if resolved_manifest_hash:
        tags["manifest_hash"] = resolved_manifest_hash

    data_asset = Data(
        name=resolved_asset_name,
        version=resolved_version,
        description="Raw NGT alphabet ImageFolder dataset. Split is created inside training.",
        path=data_path,
        type=AssetTypes.URI_FOLDER,
        tags=tags,
    )

    registered = ml_client.data.create_or_update(data_asset)

    print("Registered raw data asset:")
    print(f"  name: {registered.name}")
    print(f"  version: {registered.version}")
    print(f"  type: {registered.type}")
    print(f"  path: {registered.path}")
    print(f"  image_count: {resolved_image_count}")
    print(f"  manifest_hash: {resolved_manifest_hash}")


if __name__ == "__main__":
    app()
