"""Promote an MLflow model version to ``@champion`` for API serving.

This is the MLflow / on-prem analogue of ``scripts/promote_model.py`` (which
operates against the Azure ML registry). The two scripts mirror each other:
both pick a specific version of ``ngt-sign-language``, mark it as the live
one, and demote the previous live version. The mechanism differs by registry:

- Azure ML: tag ``promoted=true`` (cleared on the previous version).
- MLflow:   move the ``@champion`` alias (set_registered_model_alias is
            atomic — moving the alias to v2 automatically removes it from v1).

The backend with ``DEPLOY_TARGET=onprem`` resolves ``models:/<name>@champion``
on startup, so promoting here is what actually swaps the live model.

Usage::

    poetry run python scripts/promote_mlflow_champion.py --version 5
    poetry run python scripts/promote_mlflow_champion.py --from-candidate
    poetry run python scripts/promote_mlflow_champion.py --version 5 --dry-run

The tracking URI comes from the ``MLFLOW_TRACKING_URI`` environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import typer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)

CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"


def _connect_client():
    """Return an ``MlflowClient`` configured from ``MLFLOW_TRACKING_URI``.

    :returns: An authenticated :class:`~mlflow.tracking.MlflowClient`.
    :raises typer.Exit: If MLflow is not installed or the env var is unset.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.error(
            "MLflow is not installed. Install with: poetry install --with training"
        )
        raise typer.Exit(1)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        logger.error(
            "MLFLOW_TRACKING_URI is not set. Export it before running:\n"
            "    export MLFLOW_TRACKING_URI=http://<host>:2027"
        )
        raise typer.Exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


@app.command()
def main(
    model_name: str = typer.Option(
        "ngt-sign-language",
        "--model-name",
        help="Name of the model in the MLflow registry.",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Exact version number to promote.",
    ),
    from_candidate: bool = typer.Option(
        False,
        "--from-candidate",
        help=(
            f"Promote whatever version currently holds the '@{CANDIDATE_ALIAS}' "
            "alias. Convenient after a training run produced a new candidate."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would happen without modifying any aliases.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip the interactive confirmation prompt.",
    ),
) -> None:
    """Promote a model version to ``@champion`` for backend serving.

    Exactly one version is live at a time. The target version receives the
    ``@champion`` alias; MLflow atomically moves the alias off whichever
    version held it previously (so an explicit demotion step is not needed).

    The target is selected by either ``--version`` (exact) or
    ``--from-candidate`` (whatever currently holds ``@candidate``).

    :raises typer.Exit: If neither or both selectors are provided, the
        target version doesn't exist, or the user declines confirmation.
    """
    if not version and not from_candidate:
        logger.error("Provide either --version or --from-candidate.")
        raise typer.Exit(1)
    if version and from_candidate:
        logger.error("Provide --version OR --from-candidate, not both.")
        raise typer.Exit(1)

    client = _connect_client()

    # Resolve target version.
    if from_candidate:
        try:
            candidate_mv = client.get_model_version_by_alias(
                name=model_name, alias=CANDIDATE_ALIAS
            )
        except Exception as exc:
            logger.error(
                "Could not resolve alias '@%s' for model '%s': %s",
                CANDIDATE_ALIAS,
                model_name,
                exc,
            )
            raise typer.Exit(1) from exc
        target_version = candidate_mv.version
        logger.info(
            "Target (from @%s): version=%s tags=%s",
            CANDIDATE_ALIAS,
            target_version,
            candidate_mv.tags,
        )
    else:
        assert version is not None  # narrowed by the `if not version` guard above
        try:
            mv = client.get_model_version(name=model_name, version=version)
        except Exception as exc:
            logger.error("Version '%s' not found for model '%s'.", version, model_name)
            raise typer.Exit(1) from exc
        target_version = mv.version
        logger.info("Target: version=%s tags=%s", target_version, mv.tags)

    # Find current champion (if any) for the dry-run diff and the message.
    try:
        current_champion_mv = client.get_model_version_by_alias(
            name=model_name, alias=CHAMPION_ALIAS
        )
        current_champion_version: Optional[str] = current_champion_mv.version
    except Exception:
        current_champion_version = None

    typer.echo("\nPlanned changes:")
    if current_champion_version == target_version:
        typer.echo(
            f"  version {target_version}: already holds @{CHAMPION_ALIAS} "
            f"— nothing to do."
        )
        return

    if current_champion_version is not None:
        typer.echo(
            f"  version {current_champion_version}: " f"@{CHAMPION_ALIAS} -> (removed)"
        )
    typer.echo(f"  version {target_version}: (none) -> @{CHAMPION_ALIAS}")

    if dry_run:
        return

    if not yes and not typer.confirm("Apply these changes?"):
        raise typer.Exit(1)

    # set_registered_model_alias is atomic — moving the alias to a new version
    # automatically removes it from the previous holder. No explicit demote.
    client.set_registered_model_alias(
        name=model_name, alias=CHAMPION_ALIAS, version=target_version
    )
    client.set_model_version_tag(
        name=model_name,
        version=target_version,
        key="promoted_from",
        value=CANDIDATE_ALIAS if from_candidate else "manual",
    )

    typer.echo(
        f"\nDone. @{CHAMPION_ALIAS} is now {model_name} v{target_version}. "
        f"Restart the backend stack in Portainer to pick up the new model."
    )


if __name__ == "__main__":
    app()
