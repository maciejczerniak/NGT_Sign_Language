"""NGT retraining DAG — on-prem orchestration of the trigger policy.

This DAG is the Airflow replacement for the supercronic ``signlang-trigger``
container. It reproduces the *exact* decision logic used by the Azure ML
``CronTrigger`` schedule and the on-prem checker: data-change first, scheduled
interval as fallback. The policy itself is not reimplemented here — it is
imported from
:mod:`sign_language_training.orchestration.mlflow_trigger_policy`, so Airflow,
the supercronic checker, and the Azure path all share one source of truth.

Structure (a branch, not a linear chain — this is what makes it a real DAG):

    decide ──► train_model ──► report
          └──► skip ─────────► report

- ``decide``: a PythonOperator that runs the policy *decision only* (build
  dataset inventory + read the newest MLflow registered-version timestamp).
  It does NOT train. It pushes the decision to XCom and returns the id of the
  next task to run. This task needs only stdlib + mlflow — no torch — so the
  Airflow image stays light.
- ``train_model``: a DockerOperator that runs the existing heavy
  ``signlang-training`` image exactly as the one-shot pipeline does, with
  ``--pretrain-from-mlflow --register-as-candidate``. Airflow conducts the
  container; it does not import the training dependencies itself.
- ``skip`` / ``report``: bookkeeping so every run ends cleanly and the Tree
  view in the UI shows why a run did or didn't train.

Why DockerOperator rather than importing the pipeline:
    Keeps Airflow's image free of torch/mediapipe/azureml (which fight
    Airflow's tight dependency pins), and demonstrates orchestration of a
    separate container image — the more honest "platform conducts the work"
    story for the portfolio.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
from pathlib import Path

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

logger = logging.getLogger(__name__)

# --- Configuration (overridable via Airflow Variables / env) ----------------
# These mirror the supercronic checker's CLI defaults so behaviour is identical.
TRAINING_IMAGE = os.environ.get(
    "NGT_TRAINING_IMAGE",
    "ghcr.io/bredauniversityadsai/2025-26d-fai2-adsai-group-researchgroup2/"
    "signlang-training:latest",
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("NGT_MODEL_NAME", "ngt-sign-language")
MIN_NEW_IMAGES = int(os.environ.get("NGT_MIN_NEW_IMAGES", "10"))
INTERVAL_DAYS = int(os.environ.get("NGT_INTERVAL_DAYS", "7"))
SHARED_NETWORK = os.environ.get("NGT_SHARED_NETWORK", "signlang-shared")

# Paths *inside the Airflow worker* for the decision step. The training data
# and trigger-state volumes are mounted into the Airflow container the same way
# they were mounted into the supercronic container.
DATA_DIR = Path(os.environ.get("NGT_DATA_DIR", "/data"))
STATE_PATH = Path(
    os.environ.get("NGT_STATE_PATH", "/app/state/training_trigger_state_local.json")
)

# Named volumes (declared in docker-compose.airflow.yml) for the train step.
DATA_VOLUME = os.environ.get("NGT_DATA_VOLUME", "training-data")
OUTPUTS_VOLUME = os.environ.get("NGT_OUTPUTS_VOLUME", "training-outputs")


default_args = {
    "owner": "adsai-r2",
    "retries": 0,  # training is expensive and non-idempotent; don't auto-retry
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ngt_retraining",
    description="On-prem NGT retraining trigger (data-change + interval policy).",
    schedule="0 7 * * *",  # daily 07:00 UTC — matches the Azure CronTrigger
    start_date=datetime(2026, 1, 1),
    catchup=False,  # never backfill missed days; we only care about "now"
    max_active_runs=1,  # never let two training runs overlap
    default_args=default_args,
    tags=["ngt", "mlops", "retraining"],
)
def ngt_retraining():
    """Define the retraining DAG."""

    @task.branch(task_id="decide")
    def decide() -> str:
        """Run the shared trigger policy and branch on its decision.

        Reuses :func:`evaluate_and_maybe_train` with a *no-op runner*: the
        runner is what would launch training, but in Airflow the training is a
        separate DockerOperator task, so here the runner does nothing and we
        only consume the policy's reasoning. The real
        ``evaluate_and_maybe_train`` would persist state after the runner; to
        keep state semantics identical to the supercronic path we let the
        downstream training task own the run and persist state via the same
        ``run_local_pipeline`` invocation.

        Returns:
            The ``task_id`` to run next: ``"train_model"`` or ``"skip"``.
        """
        from sign_language_training.orchestration.mlflow_trigger_policy import (
            MlflowTriggerConfig,
            build_decision,
        )

        config = MlflowTriggerConfig(
            data_dir=DATA_DIR,
            state_path=STATE_PATH,
            model_name=MODEL_NAME,
            min_new_images=MIN_NEW_IMAGES,
            interval_days=INTERVAL_DAYS,
        )
        decision = build_decision(config=config, tracking_uri=MLFLOW_TRACKING_URI)

        logger.info("Trigger decision: %s", decision.message)
        context = get_current_context()
        context["ti"].xcom_push(key="reason", value=decision.reason)
        context["ti"].xcom_push(key="new_images", value=decision.new_image_count)
        context["ti"].xcom_push(
            key="current_images", value=decision.current_image_count
        )

        return "train_model" if decision.should_train else "skip"

    # The training task: run the heavy image exactly like the one-shot pipeline.
    train_model = DockerOperator(
        task_id="train_model",
        image=TRAINING_IMAGE,
        # Same command the supercronic checker built, minus the checker wrapper.
        command=[
            "python",
            "scripts/run_local_pipeline.py",
            "--raw-data-dir",
            "/data",
            "--output-dir",
            "/outputs",
            "--pretrain-from-mlflow",
            "--register-as-candidate",
            "--mlflow",
            "--num-workers",
            "0",
            "--clean",
        ],
        environment={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "MLFLOW_EXPERIMENT_NAME": "ngt-training-runs",
            "PYTHONUNBUFFERED": "1",
        },
        mounts=[
            Mount(source=DATA_VOLUME, target="/data", type="volume", read_only=True),
            Mount(source=OUTPUTS_VOLUME, target="/outputs", type="volume"),
        ],
        # Reach the MLflow service on the shared network.
        network_mode=SHARED_NETWORK,
        # Talk to the host Docker daemon (socket mounted in compose).
        docker_url="unix://var/run/docker.sock",
        auto_remove="success",
        mount_tmp_dir=False,
    )

    @task(task_id="skip")
    def skip() -> None:
        """No-op task taken when the policy decides not to retrain."""
        context = get_current_context()
        reason = context["ti"].xcom_pull(task_ids="decide", key="reason")
        logger.info("No retraining this run (reason=%s).", reason)

    @task(task_id="report", trigger_rule="none_failed_min_one_success")
    def report() -> None:
        """Summarise the run regardless of which branch executed."""
        context = get_current_context()
        ti = context["ti"]
        reason = ti.xcom_pull(task_ids="decide", key="reason")
        new_images = ti.xcom_pull(task_ids="decide", key="new_images")
        current = ti.xcom_pull(task_ids="decide", key="current_images")
        logger.info(
            "NGT retraining run complete — reason=%s, new_images=%s, total=%s.",
            reason,
            new_images,
            current,
        )

    branch = decide()
    skip_task = skip()
    report_task = report()

    branch >> [train_model, skip_task]
    [train_model, skip_task] >> report_task


ngt_retraining()
