"""Register pre-split NGT ImageFolder data assets in Azure ML.

Registers the ``train``, ``test``, and ``val`` ImageFolder split directories
as versioned ``URI_FOLDER`` data assets in the Azure ML workspace. Each split
is registered under the name ``ngt-<split>`` with the specified version.

Typical usage::

    poetry run python scripts/register_data_splits.py --version 1
    poetry run python scripts/register_data_splits.py --data-root data/ngt_subset1 --version 2
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

from sign_language_training.azure_config import get_client  # noqa: E402

app = typer.Typer(
    name="register-data-splits",
    help="Register train/test/val ImageFolder splits as Azure ML data assets.",
    add_completion=False,
)


@app.command()
def main(
    data_root: Path = typer.Option(
        REPO_ROOT / "data" / "ngt_subset1",
        "--data-root",
        help="Directory containing train, test, and val split folders.",
    ),
    version: str = typer.Option(
        ...,
        "--version",
        help="Azure ML data asset version to register.",
    ),
) -> None:
    """Register the train, test, and val ImageFolder splits as Azure ML data assets.

    Iterates over the ``train``, ``test``, and ``val`` subdirectories under
    ``data_root``, validates that each exists, and registers it as a versioned
    ``URI_FOLDER`` data asset named ``ngt-<split>`` in the Azure ML workspace.

    Args:
        data_root: Root directory containing ``train``, ``test``, and ``val``
            subdirectories in ImageFolder format.
        version: Version string to assign to each registered data asset.
            Must be unique per asset name in the workspace, or the existing version
            will be updated.

    Raises:
        typer.BadParameter: If any of the expected split subdirectories
            does not exist under ``data_root``.
    """
    ml_client = get_client()
    splits = ("train", "test", "val")

    for split in splits:
        split_path = (data_root / split).resolve()
        if not split_path.is_dir():
            raise typer.BadParameter(
                f"Split directory not found: {split_path}",
                param_hint="--data-root",
            )

        data_asset = Data(
            name=f"ngt-{split}",
            version=version,
            description=f"{split.capitalize()} split for NGT alphabet gesture recognition",
            path=str(split_path),
            type=AssetTypes.URI_FOLDER,
            tags={"project": "ngt", "resolution": "224x224"},
        )

        ml_client.data.create_or_update(data_asset)
        typer.echo(f"Registered: {data_asset.name} (v{data_asset.version})")

    typer.echo("All splits processed.")


if __name__ == "__main__":
    app()
