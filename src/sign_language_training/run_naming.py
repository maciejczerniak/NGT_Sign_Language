"""Automatic run-naming utilities for the training workflow.

Generates sequential run directory names of the form ``model_<N>`` under
a given output directory, incrementing the number based on existing runs.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_RUN_PREFIX = "model_"
_RUN_PATTERN = re.compile(rf"^{re.escape(_RUN_PREFIX)}(\d+)$")


def _find_existing_run_numbers(output_dir: Path) -> list[int]:
    """Return sorted run numbers already present in ``output_dir``.

    Scans ``output_dir`` for subdirectories whose names match the pattern
    ``model_<N>`` and returns their integer suffixes in ascending order.

    Args:
        output_dir: Directory to scan for existing run subdirectories.

    Returns:
        Sorted list of run numbers found. Returns an empty list if
            ``output_dir`` does not exist or contains no matching subdirectories.
    """
    if not output_dir.is_dir():
        return []

    numbers: list[int] = []
    for entry in output_dir.iterdir():
        if entry.is_dir():
            match = _RUN_PATTERN.match(entry.name)
            if match:
                numbers.append(int(match.group(1)))

    return sorted(numbers)


def generate_run_name(output_dir: Path) -> str:
    """Return the next available run name for ``output_dir``.

    Finds the highest existing ``model_<N>`` run number and returns
    ``model_<N+1>``. Returns ``model_1`` if no runs exist yet.

    Args:
        output_dir: Directory in which the new run subdirectory will
            be created.

    Returns:
        A run name string in the format ``model_<N>``.
    """
    existing = _find_existing_run_numbers(output_dir)
    next_number = (existing[-1] + 1) if existing else 1
    name = f"{_RUN_PREFIX}{next_number}"
    logger.info("Generated run name: %s", name)
    return name


def resolve_run_dir(output_dir: Path, run_name: str | None = None) -> Path:
    """Return the full path for a training run directory.

    If ``run_name`` is not provided, generates the next available run name
    via :func:`generate_run_name`.

    Args:
        output_dir: Parent directory under which the run directory lives.
        run_name: Optional explicit run name. If ``None``, a name is
            generated automatically.

    Returns:
        The full :class:`~pathlib.Path` for the run directory, i.e.
            ``output_dir / run_name``.
    """
    if run_name is None:
        run_name = generate_run_name(output_dir)
    return output_dir / run_name
