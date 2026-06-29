"""Finalize completed Azure ML retraining sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COMPLETED_STATUSES = {"completed"}
FINAL_STATUSES = {"completed", "failed", "canceled", "cancelled"}


@dataclass(frozen=True)
class SweepFinalizationResult:
    """Outcome of finalizing one completed sweep.

    Args:
        sweep_id: Azure ML sweep job name.
        promoted_version: Version promoted from the sweep, or ``None``.
        archived_versions: Non-promoted sweep model versions archived.
        message: Human-readable summary.
    """

    sweep_id: str
    promoted_version: str | None
    archived_versions: list[str]
    message: str


def _metric(model: Any, name: str) -> float:
    """Read a numeric model metric tag.

    Args:
        model: Azure ML model asset object.
        name: Metric tag name.

    Returns:
        Parsed metric value, or ``0.0`` when missing/invalid.
    """
    try:
        return float((model.tags or {}).get(name, 0))
    except (TypeError, ValueError):
        return 0.0


def model_score(model: Any) -> tuple[float, float]:
    """Return the ordering score for model promotion.

    Args:
        model: Azure ML model asset object.

    Returns:
        ``(f1_macro, accuracy)`` score tuple.
    """
    return (_metric(model, "f1_macro"), _metric(model, "accuracy"))


def is_better_than_current(best: Any, current: Any | None) -> bool:
    """Return whether a sweep candidate beats the currently promoted model.

    Args:
        best: Best candidate model from the sweep.
        current: Currently promoted model, or ``None``.

    Returns:
        ``True`` when no current model exists or ``best`` has a higher
        ``(f1_macro, accuracy)`` score.
    """
    if current is None:
        return True
    return model_score(best) > model_score(current)


def get_models_for_sweep(ml_client: Any, model_name: str, sweep_id: str) -> list[Any]:
    """Return all model versions registered by a sweep.

    Args:
        ml_client: Authenticated Azure ML client.
        model_name: Azure ML model registry name.
        sweep_id: Parent sweep job name.

    Returns:
        Matching model versions.
    """
    matches = []
    for model in ml_client.models.list(name=model_name):
        tags = model.tags or {}
        if tags.get("sweep_id") == sweep_id or tags.get("run_id", "").startswith(
            sweep_id
        ):
            matches.append(model)
    return matches


def get_current_promoted_model(ml_client: Any, model_name: str) -> Any | None:
    """Return the currently promoted model version.

    Args:
        ml_client: Authenticated Azure ML client.
        model_name: Azure ML model registry name.

    Returns:
        Highest-scoring model with ``promoted=true``, or ``None``.
    """
    promoted = [
        model
        for model in ml_client.models.list(name=model_name)
        if (model.tags or {}).get("promoted") == "true"
    ]
    if not promoted:
        return None
    return max(promoted, key=model_score)


def mark_sweep_finalized(ml_client: Any, sweep: Any, status: str) -> None:
    """Mark a sweep job as finalized by updating its tags.

    Args:
        ml_client: Authenticated Azure ML client.
        sweep: Azure ML sweep job object.
        status: Finalization status tag value.
    """
    tags = dict(getattr(sweep, "tags", {}) or {})
    tags["finalization_status"] = status
    sweep.tags = tags
    ml_client.jobs.create_or_update(sweep)


def promote_model_version(ml_client: Any, model_name: str, version: str) -> None:
    """Promote one model version and demote any previously promoted versions.

    Args:
        ml_client: Authenticated Azure ML client.
        model_name: Azure ML model registry name.
        version: Model version to promote.
    """
    for model in ml_client.models.list(name=model_name):
        tags = dict(model.tags or {})
        if str(model.version) == str(version):
            model.tags = {**tags, "promoted": "true"}
            ml_client.models.create_or_update(model)
        elif tags.get("promoted") == "true":
            model.tags = {**tags, "promoted": "false"}
            ml_client.models.create_or_update(model)


def archive_model_version(ml_client: Any, model_name: str, version: str) -> None:
    """Archive one Azure ML model version.

    Args:
        ml_client: Authenticated Azure ML client.
        model_name: Azure ML model registry name.
        version: Model version to archive.
    """
    ml_client.models.archive(name=model_name, version=str(version))


def finalize_sweep(
    ml_client: Any,
    sweep: Any,
    model_name: str,
    archive_non_best: bool = True,
) -> SweepFinalizationResult:
    """Promote the best completed sweep model and archive the rest.

    Args:
        ml_client: Authenticated Azure ML client.
        sweep: Completed Azure ML sweep job.
        model_name: Azure ML model registry name.
        archive_non_best: If ``True``, archive non-promoted models from the
            sweep.

    Returns:
        Finalization result.
    """
    sweep_id = str(getattr(sweep, "name"))
    candidates = get_models_for_sweep(ml_client, model_name, sweep_id)
    if not candidates:
        mark_sweep_finalized(ml_client, sweep, "no_registered_models")
        return SweepFinalizationResult(
            sweep_id=sweep_id,
            promoted_version=None,
            archived_versions=[],
            message="No registered models found for completed sweep.",
        )

    best = max(candidates, key=model_score)
    current = get_current_promoted_model(ml_client, model_name)
    should_promote = is_better_than_current(best, current)

    promoted_version: str | None = None
    if should_promote:
        promoted_version = str(best.version)
        promote_model_version(ml_client, model_name, promoted_version)

    archived_versions: list[str] = []
    if archive_non_best:
        for model in candidates:
            if str(model.version) == str(best.version):
                continue
            archive_model_version(ml_client, model_name, str(model.version))
            archived_versions.append(str(model.version))

    mark_sweep_finalized(
        ml_client,
        sweep,
        "promoted" if should_promote else "not_better_than_current",
    )
    return SweepFinalizationResult(
        sweep_id=sweep_id,
        promoted_version=promoted_version,
        archived_versions=archived_versions,
        message=(
            f"Promoted version {promoted_version}."
            if promoted_version
            else "Best sweep model did not beat current promoted model."
        ),
    )


def finalize_completed_sweeps(
    ml_client: Any,
    experiment_name: str,
    model_name: str,
    archive_non_best: bool = True,
) -> list[SweepFinalizationResult]:
    """Finalize all completed pending retraining sweeps in an experiment.

    Args:
        ml_client: Authenticated Azure ML client.
        experiment_name: Azure ML experiment to inspect.
        model_name: Azure ML model registry name.
        archive_non_best: If ``True``, archive non-promoted sweep models.

    Returns:
        Finalization results for all completed pending sweeps.
    """
    results: list[SweepFinalizationResult] = []
    for job in ml_client.jobs.list():
        if getattr(job, "experiment_name", None) != experiment_name:
            continue
        tags = getattr(job, "tags", {}) or {}
        if tags.get("purpose") != "retraining-sweep":
            continue
        if tags.get("finalization_status") != "pending":
            continue
        status = str(getattr(job, "status", "")).strip().lower()
        if status in COMPLETED_STATUSES:
            results.append(
                finalize_sweep(
                    ml_client=ml_client,
                    sweep=job,
                    model_name=model_name,
                    archive_non_best=archive_non_best,
                )
            )
        elif status in FINAL_STATUSES:
            mark_sweep_finalized(ml_client, job, f"terminal_{status}")
    return results
