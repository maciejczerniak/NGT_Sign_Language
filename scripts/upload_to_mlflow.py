"""One-off: upload EfficientNet + Landmark MLP checkpoints to MLflow.

Run this ONCE per environment to seed the model registry. The backend stack
with ``DEPLOY_TARGET=onprem`` then resolves ``models:/ngt-sign-language@champion``
(primary) and ``models:/ngt-landmark-mlp@champion`` (fallback) on startup.

Usage:

    # 1. Point at the on-prem MLflow tracking server:
    export MLFLOW_TRACKING_URI=http://194.171.191.227:2027

    # 2. From the repo root:
    poetry run python scripts/upload_to_mlflow.py

For each model the script:
1. Logs the .pth file as a raw artifact under a fresh run.
2. Registers the artifact dir as a new model version via the low-level
   MlflowClient.create_model_version() — MLflow 3.x's high-level
   mlflow.register_model() now requires a "logged model" object produced by
   mlflow.<flavor>.log_model(), which doesn't fit our custom .pth format.
3. Sets the ``@champion`` alias to the new version.
4. Tags the version with val_acc and source path.

Re-running creates v2, v3, ... and moves @champion to the newest each time.
The landmark MLP step is optional — if its checkpoint is missing, the
script skips it with a warning rather than failing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import mlflow
from mlflow.tracking import MlflowClient


# ---------------------------------------------------------------------------
# Configuration — adjust if your paths/names differ
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_NAME = "ngt-registry-seeding"
ARTIFACT_SUBDIR = "data"  # placed under runs:/<id>/data
ALIAS = "champion"


@dataclass
class ModelSpec:
    """A checkpoint to upload + register."""

    label: str  # human-readable name for logs
    checkpoint_path: Path
    registry_name: str
    required: bool  # if False, missing checkpoint is a warning, not an error


MODELS = [
    ModelSpec(
        label="EfficientNet-B0 (primary)",
        checkpoint_path=REPO_ROOT / "models" / "best_ngt_model_v2.pth",
        registry_name="ngt-sign-language",
        required=True,
    ),
    ModelSpec(
        label="Landmark MLP (fallback)",
        checkpoint_path=REPO_ROOT / "models" / "best_landmark_mlp.pth",
        registry_name="ngt-landmark-mlp",
        required=False,
    ),
]


def upload_one(spec: ModelSpec, client: MlflowClient) -> Optional[str]:
    """Upload + register + alias one checkpoint. Return version string on success.

    Returns None if the checkpoint is missing and ``spec.required`` is False.
    Raises on hard failures (missing required checkpoint, registry errors).
    """
    print()
    print(f"=== {spec.label} ===")
    print(f"Checkpoint: {spec.checkpoint_path}")

    if not spec.checkpoint_path.exists():
        msg = f"Checkpoint not found: {spec.checkpoint_path}"
        if spec.required:
            sys.exit(msg)
        print(f"  skipping — {msg}")
        return None

    # Read metadata for tags. weights_only=False because the .pth contains a
    # dict with class_names + model_state, not just tensors.
    ckpt = torch.load(spec.checkpoint_path, map_location="cpu", weights_only=False)
    class_names: list[str] = ckpt.get("class_names", [])
    val_acc = float(ckpt.get("val_acc", 0.0))
    epoch = int(ckpt.get("epoch", -1))
    print(
        f"  metadata: {len(class_names)} classes, "
        f"val_acc={val_acc:.4f}, epoch={epoch}"
    )

    # Log artifact under a new run.
    with mlflow.start_run(run_name=f"seed-{spec.checkpoint_path.stem}") as run:
        run_id = run.info.run_id
        mlflow.log_param("num_classes", len(class_names))
        mlflow.log_metric("val_acc", val_acc)
        mlflow.log_metric("epoch", epoch)
        mlflow.log_artifact(str(spec.checkpoint_path), artifact_path=ARTIFACT_SUBDIR)

    # Register via the low-level API (see module docstring for rationale).
    try:
        client.create_registered_model(spec.registry_name)
        print(f"  created registered model '{spec.registry_name}'")
    except mlflow.exceptions.MlflowException as exc:
        msg = str(exc).lower()
        if "already exists" in msg or "resource_already_exists" in msg:
            print(f"  '{spec.registry_name}' already exists — adding new version")
        else:
            raise

    model_uri = f"runs:/{run_id}/{ARTIFACT_SUBDIR}"
    mv = client.create_model_version(
        name=spec.registry_name,
        source=model_uri,
        run_id=run_id,
    )
    print(f"  registered v{mv.version}")

    client.set_registered_model_alias(
        name=spec.registry_name, alias=ALIAS, version=mv.version
    )
    print(f"  alias '@{ALIAS}' → v{mv.version}")

    client.set_model_version_tag(
        name=spec.registry_name,
        version=mv.version,
        key="source_checkpoint",
        value=str(spec.checkpoint_path.relative_to(REPO_ROOT)),
    )
    client.set_model_version_tag(
        name=spec.registry_name,
        version=mv.version,
        key="val_acc",
        value=f"{val_acc:.4f}",
    )
    client.set_model_version_tag(
        name=spec.registry_name,
        version=mv.version,
        key="seeded",
        value="true",
    )

    return mv.version


def main() -> None:
    """Upload, register, and alias every configured checkpoint."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        sys.exit(
            "MLFLOW_TRACKING_URI is not set. Export it first:\n"
            "    export MLFLOW_TRACKING_URI=http://<host>:2027"
        )

    print(f"Tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)

    results: dict[str, Optional[str]] = {}
    for spec in MODELS:
        results[spec.registry_name] = upload_one(spec, client)

    print()
    print("=== Summary ===")
    for name, version in results.items():
        if version is None:
            print(f"  {name}: skipped")
        else:
            print(f"  {name}: v{version}")
            print(f"    {tracking_uri}/#/models/{name}/versions/{version}")


if __name__ == "__main__":
    main()
