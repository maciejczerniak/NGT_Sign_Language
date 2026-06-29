"""Upload a local ImageFolder dataset directory to MinIO as a zipfile.

Mirror of ``scripts/upload_to_mlflow.py``, but for the raw training data
rather than model checkpoints. Used to seed the ``training-data`` MinIO
bucket that ``docker-compose.data-seeder.yml`` reads from.

The script:
1. Zips the local ImageFolder root (one subdir per class).
2. Uploads the zip to the ``training-data`` bucket in MinIO as ``training.zip``.
3. Replaces any existing zip with the same name.

After running, deploy/restart the data-seeder stack in Portainer to copy
the new data into the training-data Docker volume on the Lambda host. Then
restart the training-pipeline container to actually train on it.

Typical usage::

    export MINIO_ENDPOINT_URL=http://194.171.191.227:2028
    export MINIO_ACCESS_KEY=<MINIO_ROOT_USER from Portainer>
    export MINIO_SECRET_KEY=<MINIO_ROOT_PASSWORD from Portainer>

    poetry run python scripts/upload_training_data.py \\
        --source-dir data/raw/training \\
        --bucket training-data \\
        --object-name training.zip

Defaults pick up sensible values for our Lambda host setup, so this often
shortens to::

    poetry run python scripts/upload_training_data.py
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _zip_directory(source_dir: Path, output_path: Path) -> None:
    """Zip ``source_dir`` into ``output_path``, preserving its top-level name.

    The archive contains entries like ``<source_dir.name>/A/img0001.jpg``,
    matching what the data-seeder script expects: it unzips into ``/tmp``
    and copies ``/tmp/<source_dir.name>/.`` into the volume.

    :param source_dir: ImageFolder root to compress.
    :param output_path: Where to write the .zip file.
    """
    logger.info("Zipping %s -> %s", source_dir, output_path)
    file_count = 0
    skipped_count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            # Skip macOS metadata that breaks the ImageFolder split downstream.
            if file_path.name == ".DS_Store" or file_path.name.startswith("._"):
                skipped_count += 1
                continue
            arcname = file_path.relative_to(source_dir.parent)
            zf.write(file_path, arcname=arcname)
            file_count += 1
            if file_count % 200 == 0:
                logger.info("  ... %d files added", file_count)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "Zip complete — %d files (%d junk skipped), %.1f MB",
        file_count,
        skipped_count,
        size_mb,
    )


def _upload_to_minio(
    local_path: Path,
    endpoint_url: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    object_name: str,
) -> None:
    """Upload ``local_path`` to MinIO at ``s3://bucket/object_name``.

    Creates the bucket if it does not already exist. Overwrites any
    existing object with the same name.

    :raises typer.Exit: If boto3 is not installed or credentials are wrong.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logger.error(
            "boto3 is not installed. Install with: poetry install --with training"
        )
        raise typer.Exit(1)

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        # MinIO uses path-style addressing, not virtual-hosted-style.
        config=boto3.session.Config(s3={"addressing_style": "path"}),
    )

    # Create bucket if needed.
    try:
        client.head_bucket(Bucket=bucket)
        logger.info("Bucket '%s' exists.", bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"404", "NoSuchBucket"}:
            logger.info("Creating bucket '%s' ...", bucket)
            client.create_bucket(Bucket=bucket)
        else:
            logger.error("Cannot access bucket '%s': %s", bucket, exc)
            raise typer.Exit(1) from exc

    logger.info(
        "Uploading %s (%.1f MB) -> s3://%s/%s",
        local_path,
        local_path.stat().st_size / (1024 * 1024),
        bucket,
        object_name,
    )
    client.upload_file(str(local_path), bucket, object_name)
    logger.info("Upload complete.")


@app.command()
def main(
    source_dir: Path = typer.Option(
        Path("data/raw/training"),
        "--source-dir",
        help="Local ImageFolder root to upload (one subdirectory per class).",
    ),
    bucket: str = typer.Option(
        "training-data",
        "--bucket",
        help="MinIO bucket name. Created if missing.",
    ),
    object_name: str = typer.Option(
        "training.zip",
        "--object-name",
        help="Name of the zipfile in the bucket. Overwrites any existing object.",
    ),
    endpoint_url: str = typer.Option(
        None,
        "--endpoint-url",
        help=(
            "MinIO S3 endpoint URL. Defaults to MINIO_ENDPOINT_URL env var or "
            "http://194.171.191.227:2028."
        ),
    ),
    keep_zip: bool = typer.Option(
        False,
        "--keep-zip",
        help="If set, write the zipfile to ./training.zip and don't delete it after upload.",
    ),
) -> None:
    """Zip a local dataset directory and upload it to MinIO.

    Reads MinIO credentials from environment variables:

    - ``MINIO_ACCESS_KEY`` (or ``MINIO_ROOT_USER`` as fallback)
    - ``MINIO_SECRET_KEY`` (or ``MINIO_ROOT_PASSWORD`` as fallback)
    - ``MINIO_ENDPOINT_URL`` (optional; falls back to the on-prem host)

    :raises typer.Exit: If the source directory doesn't exist or doesn't look
        like an ImageFolder root (no class subdirectories), or if credentials
        are missing.
    """
    source_dir = source_dir.resolve()
    if not source_dir.exists():
        logger.error("Source directory not found: %s", source_dir)
        raise typer.Exit(1)
    class_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        logger.error(
            "Source directory has no class subdirectories — does not look "
            "like an ImageFolder root: %s",
            source_dir,
        )
        raise typer.Exit(1)
    logger.info(
        "Source: %s (%d class directories: %s)",
        source_dir,
        len(class_dirs),
        ", ".join(sorted(d.name for d in class_dirs)),
    )

    # Credentials: prefer explicit env vars but accept the MLflow stack's
    # variable names as a fallback to save the user from re-exporting them.
    access_key = os.environ.get("MINIO_ACCESS_KEY") or os.environ.get("MINIO_ROOT_USER")
    secret_key = os.environ.get("MINIO_SECRET_KEY") or os.environ.get(
        "MINIO_ROOT_PASSWORD"
    )
    endpoint = (
        endpoint_url
        or os.environ.get("MINIO_ENDPOINT_URL")
        or "http://194.171.191.227:2028"
    )

    if not access_key or not secret_key:
        logger.error(
            "MinIO credentials missing. Set MINIO_ACCESS_KEY and MINIO_SECRET_KEY "
            "(or MINIO_ROOT_USER/MINIO_ROOT_PASSWORD) before running."
        )
        raise typer.Exit(1)

    # Zip into a temp file (or local file if --keep-zip).
    if keep_zip:
        zip_path = Path.cwd() / "training.zip"
        _zip_directory(source_dir, zip_path)
        _upload_to_minio(
            zip_path, endpoint, access_key, secret_key, bucket, object_name
        )
        logger.info("Local zip kept at %s (delete it when no longer needed).", zip_path)
    else:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = Path(tmp.name)
        try:
            _zip_directory(source_dir, zip_path)
            _upload_to_minio(
                zip_path, endpoint, access_key, secret_key, bucket, object_name
            )
        finally:
            zip_path.unlink(missing_ok=True)

    typer.echo(
        f"\nDone. The training-data MinIO bucket now holds {object_name}.\n"
        f"Next steps:\n"
        f"  1. In Portainer, redeploy the data-seeder stack to copy this "
        f"into the training-data Docker volume.\n"
        f"  2. Restart the training-pipeline container to retrain on the new data."
    )


if __name__ == "__main__":
    app()
