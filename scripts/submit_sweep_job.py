"""Submit an Azure ML SDK v2 hyperparameter sweep job.

This script submits a sweep over selected training hyperparameters for the
standalone sign_language_training package.

The training package handles:

- device detection: CUDA > MPS > CPU
- MLflow logging
- model gate
- optional Azure ML model registration

Recommended tiny validation sweep::

    poetry run python scripts/submit_sweep_job.py \\
      --max-total-trials 2 \\
      --max-concurrent-trials 1 \\
      --epochs 1 \\
      --batch-sizes 8,16 \\
      --patience-values 3,5

Recommended real sweep::

    poetry run python scripts/submit_sweep_job.py \\
      --max-total-trials 6 \\
      --max-concurrent-trials 2 \\
      --epochs 30 \\
      --batch-sizes 8,16,32 \\
      --patience-values 5,7,10
"""

from __future__ import annotations

import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Optional, cast

import typer
from azure.ai.ml import Input, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import JobResourceConfiguration
from azure.ai.ml.sweep import Choice, LogUniform, MedianStoppingPolicy

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sign_language_training.azure_config import (  # noqa: E402
    get_client,
    pretrained_checkpoint_reference_or_path,
    raw_data_asset_reference,
    resolve_compute_target,
    resolve_environment,
    resolve_instance_type,
)
from sign_language_training.orchestration.sweep_submitter import (  # noqa: E402
    DEFAULT_SWEEP_TIMEOUT_SECONDS,
)
from sign_language_training.settings import settings  # noqa: E402


def _parse_int_list(value: str, option_name: str) -> list[int]:
    """Parse a comma-separated string into a list of positive integers.

    Args:
        value: Comma-separated string of integer values, e.g. ``"8,16,32"``.
        option_name: CLI option name used in error messages, e.g. ``"--batch-sizes"``.

    Returns:
        List of parsed positive integers.

    Raises:
        typer.BadParameter: If any value is not a valid integer, if the
            list is empty, or if any value is not positive.
    """
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise typer.BadParameter(
            f"{option_name} must be a comma-separated list of integers."
        ) from exc

    if not parsed:
        raise typer.BadParameter(f"{option_name} must contain at least one value.")

    if any(item <= 0 for item in parsed):
        raise typer.BadParameter(f"{option_name} values must be positive.")

    return parsed


def _load_params_file(path: Path) -> dict[str, Any]:
    """Load sweep parameters from a JSON file.

    Keys must match the long option names with hyphens replaced by underscores,
    e.g. ``learning_rate_min``, ``batch_sizes``, ``max_total_trials``.
    For list parameters (``batch_sizes``, ``patience_values``) both a JSON
    array (``[8, 16]``) and a comma-separated string (``"8,16"``) are accepted.

    Example file::

        {
            "epochs": 30,
            "batch_sizes": [8, 16, 32],
            "patience_values": [5, 7, 10],
            "learning_rate_min": 1e-5,
            "learning_rate_max": 3e-4,
            "max_total_trials": 6,
            "max_concurrent_trials": 2
        }

    Args:
        path: Path to the JSON parameters file.

    Returns:
        Dictionary of parameter names to values. List parameters are
            normalised to comma-separated strings for compatibility with Typer options.

    Raises:
        typer.BadParameter: If the file does not exist, contains invalid
            JSON, or is not a JSON object.
    """
    if not path.exists():
        raise typer.BadParameter(
            f"Params file not found: {path}", param_hint="--params-file"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"Invalid JSON in {path}: {exc}", param_hint="--params-file"
        ) from exc
    if not isinstance(data, dict):
        raise typer.BadParameter(
            "Params file must be a JSON object.", param_hint="--params-file"
        )
    for list_key in ("batch_sizes", "patience_values"):
        if list_key in data and isinstance(data[list_key], list):
            data[list_key] = ",".join(str(v) for v in data[list_key])
    return data


def _resolve(
    ctx: typer.Context, file_params: dict[str, Any], name: str, cli_value: Any
) -> Any:
    """Return the effective parameter value, preferring explicit CLI input over file defaults.

    Uses Click's ``ParameterSource`` to detect whether the value was explicitly
    supplied on the CLI or is still at its default. If the parameter was not
    explicitly set on the CLI and a value exists in ``file_params``, the file
    value is returned. Otherwise the CLI value (default or explicit) is returned.

    Args:
        ctx: The active Typer/Click context, used to check parameter sources.
        file_params: Dictionary of parameter values loaded from a JSON params
            file via :func:`_load_params_file`.
        name: The parameter name as it appears in ``file_params``, matching
            the CLI option name with hyphens replaced by underscores.
        cli_value: The current value of the parameter from the CLI (may be
            the default or an explicitly supplied value).

    Returns:
        The file param value if it exists and the CLI value is still at
            its default; otherwise the CLI value.
    """
    from click.core import ParameterSource

    if (
        name in file_params
        and ctx.get_parameter_source(name) == ParameterSource.DEFAULT
    ):
        return file_params[name]
    return cli_value


def main(
    ctx: typer.Context,
    params_file: Optional[Path] = typer.Option(
        None,
        "--params-file",
        help=(
            "Path to a JSON file with default parameter values. "
            "Any flag supplied explicitly on the CLI overrides the file. "
            "See _load_params_file() docstring for the expected schema."
        ),
    ),
    job_name: Optional[str] = typer.Option(
        None,
        "--job-name",
        help="Optional Azure ML sweep job name. Leave empty to let Azure generate one.",
    ),
    experiment_name: str = typer.Option(
        "sign-language-sweep",
        "--experiment-name",
        help="Azure ML experiment name for the sweep.",
    ),
    display_name: str = typer.Option(
        "NGT sign-language hyperparameter sweep",
        "--display-name",
        help="Display name shown in Azure ML Studio.",
    ),
    description: str = typer.Option(
        "Hyperparameter sweep for the NGT sign-language classifier.",
        "--description",
        help="Azure ML sweep job description.",
    ),
    data_asset: str | None = typer.Option(
        None,
        "--data-asset",
        help="Azure ML raw data asset reference, e.g. azureml:ngt-raw:1.",
    ),
    pretrained_checkpoint: str | None = typer.Option(
        None,
        "--pretrained-checkpoint",
        help=(
            "Pretrained checkpoint path or Azure ML asset reference. "
            "Example: azureml:ngt-pretrained-checkpoint:1"
        ),
    ),
    instance_type: Optional[str] = typer.Option(
        None,
        "--instance-type",
        help=(
            "Azure ML Kubernetes instance type. "
            "Leave empty to use resolve_instance_type(), usually GPU if configured."
        ),
    ),
    epochs: int = typer.Option(
        settings.training_epochs,
        "--epochs",
        help="Maximum epochs per trial.",
        min=1,
    ),
    eta_min: float = typer.Option(
        settings.training_eta_min,
        "--eta-min",
        help=(
            "Minimum learning rate for cosine scheduler. Passed as TRAINING_ETA_MIN "
            "because the training CLI does not expose --eta-min."
        ),
        min=0.0,
    ),
    img_size: int = typer.Option(
        settings.training_img_size,
        "--img-size",
        help="Fixed image size. Keep fixed initially.",
        min=1,
    ),
    val_split: float = typer.Option(
        settings.training_val_split,
        "--val-split",
        help="Fixed validation split ratio. Keep fixed for fair comparison.",
        min=0.01,
        max=0.99,
    ),
    seed: int = typer.Option(
        settings.training_seed,
        "--seed",
        help="Fixed random seed. Keep fixed for fair comparison.",
    ),
    target_accuracy: float = typer.Option(
        settings.training_target_accuracy,
        "--target-accuracy",
        help="Accuracy threshold used by the model gate.",
        min=0.0,
        max=1.0,
    ),
    expected_num_classes: int = typer.Option(
        settings.training_expected_num_classes,
        "--expected-num-classes",
        help="Expected number of class folders.",
        min=1,
    ),
    f1_threshold: float = typer.Option(
        settings.training_f1_threshold,
        "--f1-threshold",
        help="F1 macro threshold used by the model gate.",
        min=0.0,
        max=1.0,
    ),
    num_workers: int = typer.Option(
        settings.training_num_workers,
        "--num-workers",
        help="DataLoader worker process count.",
        min=0,
    ),
    pin_memory: bool = typer.Option(
        settings.training_pin_memory,
        "--pin-memory/--no-pin-memory",
        help=(
            "Whether DataLoader should use pinned memory. Passed as TRAINING_PIN_MEMORY "
            "because the training CLI does not expose --pin-memory."
        ),
    ),
    learning_rate_min: float = typer.Option(
        1e-5,
        "--learning-rate-min",
        help="Minimum learning rate for log-uniform sweep.",
        min=0.0,
    ),
    learning_rate_max: float = typer.Option(
        3e-4,
        "--learning-rate-max",
        help="Maximum learning rate for log-uniform sweep.",
        min=0.0,
    ),
    batch_sizes: str = typer.Option(
        "8,16,32",
        "--batch-sizes",
        help="Comma-separated batch sizes to try.",
    ),
    patience_values: str = typer.Option(
        "5,7,10",
        "--patience-values",
        help="Comma-separated patience values to try.",
    ),
    primary_metric: str = typer.Option(
        "best_val_accuracy",
        "--primary-metric",
        help=(
            "Metric to optimize. Must match a metric logged by training. "
            "Use best_val_accuracy for simple sweeps, val_acc for median stopping."
        ),
    ),
    goal: str = typer.Option(
        "Maximize",
        "--goal",
        help="Sweep goal: Maximize or Minimize.",
    ),
    sampling_algorithm: str = typer.Option(
        "random",
        "--sampling-algorithm",
        help="Sweep sampling algorithm: random, grid, or bayesian.",
    ),
    max_total_trials: int = typer.Option(
        6,
        "--max-total-trials",
        help="Maximum number of trials in the sweep.",
        min=1,
    ),
    max_concurrent_trials: int = typer.Option(
        2,
        "--max-concurrent-trials",
        help="Maximum number of trials running at once.",
        min=1,
    ),
    timeout: int = typer.Option(
        DEFAULT_SWEEP_TIMEOUT_SECONDS,
        "--timeout",
        help="Maximum total sweep duration in seconds.",
        min=1,
    ),
    trial_timeout: int = typer.Option(
        7200,
        "--trial-timeout",
        help="Maximum duration of one trial in seconds.",
        min=1,
    ),
    use_median_stopping: bool = typer.Option(
        False,
        "--use-median-stopping/--no-median-stopping",
        help=(
            "Enable Azure median stopping. Best with per-epoch primary metric val_acc, "
            "not final-only best_val_accuracy."
        ),
    ),
    median_delay_evaluation: int = typer.Option(
        3,
        "--median-delay-evaluation",
        help="Number of intervals before median stopping starts.",
        min=0,
    ),
    median_evaluation_interval: int = typer.Option(
        1,
        "--median-evaluation-interval",
        help="Frequency for median stopping evaluation.",
        min=1,
    ),
    mlflow_enabled: bool = typer.Option(
        True,
        "--mlflow-enabled/--no-mlflow-enabled",
        help="Set MLFLOW_ENABLED for the training package.",
    ),
    mlflow_experiment_name: str = typer.Option(
        settings.mlflow_experiment_name,
        "--mlflow-experiment-name",
        help="MLflow experiment name used inside trials.",
    ),
    mlflow_autolog: bool = typer.Option(
        settings.mlflow_autolog,
        "--mlflow-autolog/--no-mlflow-autolog",
        help="Set MLFLOW_AUTOLOG for the training package.",
    ),
    mlflow_log_artifacts: bool = typer.Option(
        settings.mlflow_log_artifacts,
        "--mlflow-log-artifacts/--no-mlflow-log-artifacts",
        help="Set MLFLOW_LOG_ARTIFACTS for the training package.",
    ),
    model_registry_name: str = typer.Option(
        settings.model_registry_name,
        "--model-registry-name",
        help="Azure ML model name used by the training package model gate.",
    ),
) -> None:
    """Submit one Azure ML hyperparameter sweep job for the NGT sign-language classifier.

    Resolves data asset and checkpoint references from CLI options with fallback
    to project settings, then applies any JSON params file with CLI flags taking
    priority. Builds a trial command job and wraps it in an Azure ML sweep job
    with the configured search space, sampling algorithm, limits, and optional
    median stopping policy.

    The following hyperparameters are swept:

    - ``learning_rate``: log-uniform sample between ``learning_rate_min`` and
      ``learning_rate_max``
    - ``batch_size``: discrete choice from ``batch_sizes``
    - ``patience``: discrete choice from ``patience_values``

    All other training parameters are fixed across trials and passed as
    constants or environment variables.

    Prints a full summary of the submitted job configuration and the Azure ML
    Studio URL for monitoring.

    Args:
        ctx: Typer/Click context used by :func:`_resolve` to detect whether
            parameters were explicitly supplied on the CLI or are at their defaults.
        params_file: Optional path to a JSON file providing parameter
            defaults. Explicit CLI flags always override file values.
        job_name: Optional Azure ML job name. If omitted, Azure generates one.
        experiment_name: Azure ML experiment name to group sweep trials under.
        display_name: Human-readable display name shown in Azure ML Studio.
        description: Description text for the Azure ML sweep job.
        data_asset: Raw data asset reference in ``azureml:<name>:<version>``
            format. Defaults to :func:`~azure_config.raw_data_asset_reference`.
        pretrained_checkpoint: Pretrained checkpoint asset reference or local
            path. Defaults to :func:`~azure_config.pretrained_checkpoint_reference_or_path`.
        instance_type: Azure ML Kubernetes instance type override. Defaults
            to :func:`~azure_config.resolve_instance_type`.
        epochs: Maximum number of training epochs per trial.
        eta_min: Minimum learning rate for the cosine annealing scheduler,
            passed as the ``TRAINING_ETA_MIN`` environment variable.
        img_size: Fixed image size in pixels used across all trials.
        val_split: Fixed validation split ratio used across all trials.
        seed: Fixed random seed for reproducibility across trials.
        target_accuracy: Minimum validation accuracy required to pass the
            model gate after training.
        expected_num_classes: Expected number of output classes.
        f1_threshold: Minimum macro F1 score required to pass the model gate.
        num_workers: Number of DataLoader worker processes per trial.
        pin_memory: Whether to use pinned memory in the DataLoader, passed
            as the ``TRAINING_PIN_MEMORY`` environment variable.
        learning_rate_min: Lower bound of the log-uniform learning rate range.
        learning_rate_max: Upper bound of the log-uniform learning rate range.
        batch_sizes: Comma-separated batch sizes to include in the sweep
            search space.
        patience_values: Comma-separated early stopping patience values to
            include in the sweep search space.
        primary_metric: Name of the metric to optimise. Must match a metric
            logged by the training package.
        goal: Optimisation direction: ``Maximize`` or ``Minimize``.
        sampling_algorithm: Sweep sampling strategy: ``random``, ``grid``,
            or ``bayesian``.
        max_total_trials: Maximum number of trials to run in the sweep.
        max_concurrent_trials: Maximum number of trials running simultaneously.
        timeout: Maximum total sweep wall-clock duration in seconds.
        trial_timeout: Maximum wall-clock duration for a single trial in seconds.
        use_median_stopping: If ``True``, attaches a
            :class:`~azure.ai.ml.sweep.MedianStoppingPolicy` to the sweep. Works
            best when ``primary_metric`` is logged per epoch (e.g. ``val_acc``).
        median_delay_evaluation: Number of evaluation intervals to wait before
            the median stopping policy begins making decisions.
        median_evaluation_interval: Frequency at which the median stopping
            policy evaluates running trials.
        mlflow_enabled: If ``True``, sets ``MLFLOW_ENABLED=true`` in each
            trial's environment.
        mlflow_experiment_name: MLflow experiment name passed to each trial
            via the ``MLFLOW_EXPERIMENT_NAME`` environment variable.
        mlflow_autolog: Whether to enable MLflow autologging in each trial.
        mlflow_log_artifacts: Whether to log training artifacts to MLflow
            in each trial.
        model_registry_name: Azure ML model registry name passed to the
            training package model gate via the ``MODEL_REGISTRY_NAME`` environment
            variable.

    Raises:
        typer.BadParameter: If learning rate bounds are not positive, if
            ``learning_rate_min >= learning_rate_max``, or if batch sizes or patience
            values cannot be parsed as positive integers.
        ValueError: If required Azure or checkpoint settings are missing
            from ``.env``.
    """
    resolved_data_asset = data_asset or raw_data_asset_reference()
    resolved_pretrained_checkpoint = (
        pretrained_checkpoint or pretrained_checkpoint_reference_or_path()
    )

    file_params: dict[str, Any] = _load_params_file(params_file) if params_file else {}

    epochs = _resolve(ctx, file_params, "epochs", epochs)
    eta_min = _resolve(ctx, file_params, "eta_min", eta_min)
    img_size = _resolve(ctx, file_params, "img_size", img_size)
    val_split = _resolve(ctx, file_params, "val_split", val_split)
    seed = _resolve(ctx, file_params, "seed", seed)
    target_accuracy = _resolve(ctx, file_params, "target_accuracy", target_accuracy)
    expected_num_classes = _resolve(
        ctx, file_params, "expected_num_classes", expected_num_classes
    )
    f1_threshold = _resolve(ctx, file_params, "f1_threshold", f1_threshold)
    num_workers = _resolve(ctx, file_params, "num_workers", num_workers)
    pin_memory = _resolve(ctx, file_params, "pin_memory", pin_memory)
    learning_rate_min = _resolve(
        ctx, file_params, "learning_rate_min", learning_rate_min
    )
    learning_rate_max = _resolve(
        ctx, file_params, "learning_rate_max", learning_rate_max
    )
    batch_sizes = _resolve(ctx, file_params, "batch_sizes", batch_sizes)
    patience_values = _resolve(ctx, file_params, "patience_values", patience_values)
    primary_metric = _resolve(ctx, file_params, "primary_metric", primary_metric)
    goal = _resolve(ctx, file_params, "goal", goal)
    sampling_algorithm = _resolve(
        ctx, file_params, "sampling_algorithm", sampling_algorithm
    )
    max_total_trials = _resolve(ctx, file_params, "max_total_trials", max_total_trials)
    max_concurrent_trials = _resolve(
        ctx, file_params, "max_concurrent_trials", max_concurrent_trials
    )
    timeout = _resolve(ctx, file_params, "timeout", timeout)
    trial_timeout = _resolve(ctx, file_params, "trial_timeout", trial_timeout)
    use_median_stopping = _resolve(
        ctx, file_params, "use_median_stopping", use_median_stopping
    )
    mlflow_enabled = _resolve(ctx, file_params, "mlflow_enabled", mlflow_enabled)
    mlflow_experiment_name = _resolve(
        ctx, file_params, "mlflow_experiment_name", mlflow_experiment_name
    )
    mlflow_autolog = _resolve(ctx, file_params, "mlflow_autolog", mlflow_autolog)
    mlflow_log_artifacts = _resolve(
        ctx, file_params, "mlflow_log_artifacts", mlflow_log_artifacts
    )
    model_registry_name = _resolve(
        ctx, file_params, "model_registry_name", model_registry_name
    )

    if params_file:
        typer.echo(f"Loaded parameters from: {params_file}")

    if learning_rate_min <= 0 or learning_rate_max <= 0:
        raise typer.BadParameter("Learning-rate bounds must be positive.")

    if learning_rate_min >= learning_rate_max:
        raise typer.BadParameter(
            "learning_rate_min must be smaller than learning_rate_max."
        )

    parsed_batch_sizes = _parse_int_list(batch_sizes, "--batch-sizes")
    parsed_patience_values = _parse_int_list(patience_values, "--patience-values")

    if use_median_stopping and primary_metric == "best_val_accuracy":
        typer.echo(
            "Warning: median stopping works best with a metric logged each epoch, "
            "for example val_acc. best_val_accuracy is usually logged near the end.",
            err=True,
        )

    ml_client = get_client()
    compute_target = resolve_compute_target(ml_client)
    environment = resolve_environment(ml_client)
    resolved_instance_type = instance_type or resolve_instance_type()

    environment_variables = {
        "TRAINING_ETA_MIN": str(eta_min),
        "TRAINING_PIN_MEMORY": str(pin_memory).lower(),
        "MLFLOW_ENABLED": str(mlflow_enabled).lower(),
        "MLFLOW_EXPERIMENT_NAME": mlflow_experiment_name,
        "MLFLOW_AUTOLOG": str(mlflow_autolog).lower(),
        "MLFLOW_LOG_ARTIFACTS": str(mlflow_log_artifacts).lower(),
        "MODEL_REGISTRY_NAME": model_registry_name,
    }

    # NOTE:
    # Current train.py uses a Typer app with a `train` command.
    # If `python -m sign_language_training.train --help` shows options directly,
    # remove the extra `train` word below.
    trial_setup = [
        "set -e",
        f"export TRAINING_ETA_MIN={shlex.quote(str(eta_min))}",
        f"export TRAINING_PIN_MEMORY={str(pin_memory).lower()}",
        f"export MLFLOW_ENABLED={str(mlflow_enabled).lower()}",
        f"export MLFLOW_EXPERIMENT_NAME={shlex.quote(mlflow_experiment_name)}",
        f"export MLFLOW_AUTOLOG={str(mlflow_autolog).lower()}",
        f"export MLFLOW_LOG_ARTIFACTS={str(mlflow_log_artifacts).lower()}",
        f"export MODEL_REGISTRY_NAME={shlex.quote(model_registry_name)}",
    ]
    if "gpu" in resolved_instance_type.lower():
        trial_setup.append(
            'python -c "import torch; assert torch.cuda.is_available(), '
            "'GPU sweep trial received no CUDA device. Check the sweep node "
            "instance_type resource configuration.'\""
        )

    trial_command = (
        " && ".join(trial_setup)
        + """ &&
    set -e
    pip install -e src/sign_language_training/
    python -m sign_language_training.train \
        --data-dir ${{inputs.data}} \
        --pretrained-checkpoint ${{inputs.pretrained_checkpoint}} \
        --checkpoint-dir ${{outputs.checkpoints}} \
        --results-dir ${{outputs.results}} \
        --batch-size ${{inputs.batch_size}} \
        --epochs ${{inputs.epochs}} \
        --learning-rate ${{inputs.learning_rate}} \
        --img-size ${{inputs.img_size}} \
        --val-split ${{inputs.val_split}} \
        --seed ${{inputs.seed}} \
        --patience ${{inputs.patience}} \
        --target-accuracy ${{inputs.target_accuracy}} \
        --expected-num-classes ${{inputs.expected_num_classes}} \
        --f1-threshold ${{inputs.f1_threshold}} \
        --num-workers ${{inputs.num_workers}}
    """
    )

    trial_job = command(
        display_name="NGT sign-language sweep trial",
        description="One trial for NGT sign-language hyperparameter sweep.",
        experiment_name=experiment_name,
        code=str(REPO_ROOT),
        command=trial_command,
        inputs={
            "data": Input(
                type=AssetTypes.URI_FOLDER,
                path=resolved_data_asset,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "pretrained_checkpoint": Input(
                type=AssetTypes.URI_FILE,
                path=resolved_pretrained_checkpoint,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "learning_rate": LogUniform(
                min_value=math.log(learning_rate_min),
                max_value=math.log(learning_rate_max),
            ),
            "batch_size": Choice(values=cast(Any, parsed_batch_sizes)),
            "patience": Choice(values=cast(Any, parsed_patience_values)),
            "epochs": epochs,
            "img_size": img_size,
            "val_split": val_split,
            "seed": seed,
            "target_accuracy": target_accuracy,
            "expected_num_classes": expected_num_classes,
            "f1_threshold": f1_threshold,
            "num_workers": num_workers,
        },
        outputs={
            "checkpoints": {
                "type": AssetTypes.URI_FOLDER,
                "mode": "rw_mount",
            },
            "results": {
                "type": AssetTypes.URI_FOLDER,
                "mode": "rw_mount",
            },
        },
        environment=environment,
        compute=compute_target,
        resources=JobResourceConfiguration(
            instance_type=resolved_instance_type,
            instance_count=1,
        ),
        environment_variables=environment_variables,
    )

    early_termination_policy = None
    if use_median_stopping:
        early_termination_policy = MedianStoppingPolicy(
            delay_evaluation=median_delay_evaluation,
            evaluation_interval=median_evaluation_interval,
        )

    sweep_job = trial_job.sweep(
        sampling_algorithm=sampling_algorithm,
        primary_metric=primary_metric,
        goal=goal,
    )
    sweep_job.resources = JobResourceConfiguration(
        instance_type=resolved_instance_type,
        instance_count=1,
    )

    if early_termination_policy is not None:
        sweep_job.early_termination = early_termination_policy

    sweep_job.name = job_name
    sweep_job.display_name = display_name
    sweep_job.description = description
    sweep_job.experiment_name = experiment_name

    sweep_job.set_limits(
        max_total_trials=max_total_trials,
        max_concurrent_trials=max_concurrent_trials,
        timeout=timeout,
        trial_timeout=trial_timeout,
    )

    returned_job = ml_client.jobs.create_or_update(sweep_job)

    typer.echo("Submitted Azure ML sweep job:")
    typer.echo(f"  name: {returned_job.name}")
    typer.echo(f"  experiment: {experiment_name}")
    typer.echo(f"  compute: {compute_target}")
    typer.echo(f"  instance_type: {resolved_instance_type}")
    typer.echo(f"  environment: {environment}")
    typer.echo(f"  sampling_algorithm: {sampling_algorithm}")
    typer.echo(f"  primary_metric: {primary_metric}")
    typer.echo(f"  goal: {goal}")
    typer.echo(f"  max_total_trials: {max_total_trials}")
    typer.echo(f"  max_concurrent_trials: {max_concurrent_trials}")
    typer.echo(f"  learning_rate range: [{learning_rate_min}, {learning_rate_max}]")
    typer.echo(f"  batch_sizes: {parsed_batch_sizes}")
    typer.echo(f"  patience_values: {parsed_patience_values}")
    typer.echo(f"  mlflow_enabled: {mlflow_enabled}")
    typer.echo(f"  model_registry_name: {model_registry_name}")
    typer.echo(f"  studio_url: {returned_job.studio_url}")


if __name__ == "__main__":
    typer.run(main)
