"""Promote a specific ngt-sign-language model version for API serving.

Sets ``promoted=true`` on the chosen version and clears it from all others
so exactly one version is live at a time.

Usage examples
--------------
Promote a specific version by number::

    poetry run python scripts/promote_model.py --version 4277446031

Promote the best version from a specific sweep (by f1_macro tag)::

    poetry run python scripts/promote_model.py --sweep-id polite_bottle_y5vz1dz...

Dry-run either of the above (no changes made)::

    poetry run python scripts/promote_model.py --version 4277446031 --dry-run
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from azure.ai.ml import MLClient

SCRIPTS_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import get_client  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _pick_best_from_sweep(ml_client: MLClient, model_name: str, sweep_id: str) -> str:
    """Return the version string with the highest f1_macro among a sweep's trials.

    Iterates over all registered versions of ``model_name`` and filters those
    whose ``sweep_id`` tag matches the given sweep ID, or whose ``run_id`` tag
    starts with it. Among the matching candidates, the version with the highest
    ``f1_macro`` tag value is selected, with ``accuracy`` used as a tiebreaker.

    Args:
        ml_client: Authenticated :class:`~azure.ai.ml.MLClient` instance.
        model_name: Name of the model in the Azure ML registry to search.
        sweep_id: Parent sweep run ID used to filter candidate versions.

    Returns:
        The version string of the best performing trial in the sweep.

    Raises:
        typer.Exit: If no model versions are found matching the given sweep ID.
    """
    candidates = []
    for version in ml_client.models.list(name=model_name):
        tags = version.tags or {}
        if tags.get("sweep_id") == sweep_id or tags.get("run_id", "").startswith(
            sweep_id
        ):
            candidates.append(version)

    if not candidates:
        logger.error(
            "No model versions found for sweep_id='%s'. "
            "Make sure AZUREML_ROOT_RUN_ID is being saved as the 'sweep_id' tag.",
            sweep_id,
        )
        raise typer.Exit(1)

    best = max(
        candidates,
        key=lambda v: (
            float(v.tags.get("f1_macro", 0)),
            float(v.tags.get("accuracy", 0)),
        ),
    )
    logger.info(
        "Best trial in sweep '%s': version=%s f1_macro=%s accuracy=%s run_id=%s",
        sweep_id,
        best.version,
        best.tags.get("f1_macro"),
        best.tags.get("accuracy"),
        best.tags.get("run_id"),
    )
    return str(best.version)


@app.command()
def main(
    model_name: str = typer.Option(
        "ngt-sign-language",
        "--model-name",
        help="Name of the model in Azure ML registry.",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Exact version string to promote.",
    ),
    sweep_id: Optional[str] = typer.Option(
        None,
        "--sweep-id",
        help="Parent sweep run ID; promotes the best trial by f1_macro.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would happen without making any changes.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Apply promotion without an interactive confirmation prompt.",
    ),
) -> None:
    """Promote one model version for API serving by setting the ``promoted=true`` tag.

    Exactly one version is live at a time. The target version receives
    ``promoted=true`` and any previously promoted version is demoted to
    ``promoted=false``. The target version can be specified directly via
    ``--version`` or resolved automatically from a sweep via ``--sweep-id``.

    Use ``--dry-run`` to preview planned tag changes without applying them.
    Use ``--yes`` to skip the interactive confirmation prompt in CI or
    automated contexts.

    Args:
        model_name: Name of the model in the Azure ML registry.
        version: Exact version string to promote. Mutually exclusive
            with ``sweep_id``.
        sweep_id: Parent sweep run ID. The trial with the highest
            ``f1_macro`` tag (with ``accuracy`` as tiebreaker) is promoted.
            Mutually exclusive with ``version``.
        dry_run: If ``True``, prints the planned tag changes without
            applying them. No Azure ML API writes are performed.
        yes: If ``True``, skips the interactive confirmation prompt
            and applies changes immediately.

    Raises:
        typer.Exit: If neither or both of ``--version`` and ``--sweep-id``
            are provided, if the target version does not exist in the registry,
            or if the user declines the confirmation prompt.
    """
    if not version and not sweep_id:
        logger.error("Provide either --version or --sweep-id.")
        raise typer.Exit(1)
    if version and sweep_id:
        logger.error("Provide --version OR --sweep-id, not both.")
        raise typer.Exit(1)

    ml_client = get_client()

    if version is not None:
        target_version = version
    else:
        if sweep_id is None:
            raise typer.Exit(1)
        target_version = _pick_best_from_sweep(ml_client, model_name, sweep_id)

    try:
        target = ml_client.models.get(name=model_name, version=target_version)
    except Exception as exc:
        logger.error(
            "Version '%s' not found for model '%s'.", target_version, model_name
        )
        raise typer.Exit(1) from exc

    logger.info(
        "Target: version=%s f1_macro=%s accuracy=%s run_id=%s",
        target.version,
        target.tags.get("f1_macro"),
        target.tags.get("accuracy"),
        target.tags.get("run_id"),
    )

    typer.echo("\nPlanned changes:")
    versions_to_update = []
    for v in ml_client.models.list(name=model_name):
        currently = (v.tags or {}).get("promoted", "false")
        if str(v.version) == target_version:
            typer.echo(f"  version {v.version}: promoted {currently!r} -> 'true'")
            versions_to_update.append(v)
        elif currently == "true":
            typer.echo(f"  version {v.version}: promoted 'true' -> 'false'  (demoted)")
            versions_to_update.append(v)

    if dry_run:
        return

    if not yes and not typer.confirm("Apply these production model tag changes?"):
        raise typer.Exit(1)

    promoted_count = 0
    demoted_count = 0

    for v in versions_to_update:
        tags = dict(v.tags or {})
        if str(v.version) == target_version:
            if tags.get("promoted") == "true":
                logger.info("Version %s already promoted — skipping.", v.version)
                continue
            v.tags = {**tags, "promoted": "true"}
            ml_client.models.create_or_update(v)
            logger.info("Promoted version %s.", v.version)
            promoted_count += 1
        else:
            if tags.get("promoted") == "true":
                v.tags = {**tags, "promoted": "false"}
                ml_client.models.create_or_update(v)
                logger.info("Demoted version %s.", v.version)
                demoted_count += 1

    typer.echo(
        f"\nDone. Promoted: {promoted_count}, Demoted: {demoted_count}. "
        f"Active version: {target_version}"
    )


if __name__ == "__main__":
    app()
