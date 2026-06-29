# Training pipeline — `sign_language_training`

The Azure ML training pipeline package. Defines, orchestrates, and submits the four-stage gated ML training pipeline. Runs identically locally and on Azure ML using the same component functions.

---

## Contents

- [Overview](#overview)
- [Design rationale](#design-rationale)
- [Package structure](#package-structure)
- [Pipeline stages](#pipeline-stages)
- [Module table](#module-table)
  - [Pipeline components](#pipeline-components)
  - [Orchestration](#orchestration)
  - [Trigger API](#trigger-api)
  - [Configuration and utilities](#configuration-and-utilities)
- [Submission scripts](#submission-scripts)
- [Model registry](#model-registry)
- [Deployment scripts](#deployment-scripts)
- [Dependencies](#dependencies)
- [Diagrams](#diagrams)

---

## Overview

`sign_language_training` is the standalone training package. It defines a four-stage pipeline — preprocessing, training, evaluation/gating, and registration — and can run the same pipeline graph both locally and on Azure ML without any code changes. Metrics produced locally and on Azure ML are directly comparable because the same component functions execute in both environments.

The package also includes a lightweight trigger API (`trigger_api/`) that exposes an HTTP endpoint for triggering retraining programmatically, and a full suite of orchestration and submission scripts under `scripts/`.

---

## Design rationale

**Environment parity.** The pipeline runs identically locally (`scripts/run_local_pipeline.py`) and on Azure ML. The same component functions are used in both environments so there is no risk of local metrics diverging from cloud metrics.

**`--val-dir` flag.** `train.py` accepts a `--val-dir` flag that allows it to receive either a pre-split dataset (when running as a pipeline component, where preprocessing has already produced the split) or an unsplit dataset (when running as a standalone script). This means the training script does not need to be duplicated or modified between modes.

**Model gate.** No model reaches the registry without passing a configurable accuracy and F1 threshold. This is enforced programmatically in `model_evaluation.py` and cannot be bypassed by manual UI interaction — the gate is part of the pipeline code.

**Everything through code.** Data assets, dataset splits, and execution environments are registered through scripts rather than Azure ML UI clicks. This keeps all configuration under version control and makes the pipeline fully reproducible from a clean state.

**Orchestration separated from training.** Orchestration logic (pipeline submission, trigger policies, training state tracking) lives in `orchestration/` and is completely separate from model training code. The pipeline graph can be restructured without touching any model or training logic.

**Preprocessing cached.** The submission script checks whether a cached `ngt-augmented-train` asset already exists for the current raw data version. If it does, the preprocessing step is skipped and only training runs. This avoids redundant augmentation on unchanged data.

---

## Package structure

```
src/sign_language_training/
├── preprocessing.py
├── augmentation.py
├── data_loading.py
├── train.py
├── model_training.py
├── model_definitions.py
├── model_evaluation.py
├── model_registration.py
├── configuration.py
├── azure_config.py
├── run_naming.py
├── logging_utils.py
├── settings.py
├── runtime.py
├── orchestration/
│   ├── pipeline_submitter.py
│   ├── trigger_policy.py
│   ├── training_state.py
│   ├── dataset_inventory.py
│   └── __init__.py
├── trigger_api/
│   ├── app.py
│   ├── security.py
│   ├── schemas.py
│   ├── settings.py
│   └── __init__.py
└── __init__.py
```

---

## Pipeline stages

| Stage | Scripts | What it does |
|---|---|---|
| 1 — Preprocess & augment | `preprocessing.py`, `augmentation.py` | Stratified 80/10/10 split. Offline augmentation (×4 copies) applied to training partition only, so validation and test metrics remain stable across runs. Registers `ngt-augmented-train` as a versioned Azure ML data asset. |
| 2 — Train | `train.py`, `model_training.py` | Fine-tunes EfficientNet-B0 from a pretrained checkpoint. Supports `--val-dir` flag for pre-split (pipeline) or internal-split (standalone) inputs. |
| 3 — Evaluate & gate | `model_evaluation.py` | Computes accuracy and F1 on the held-out test set. Enforces configurable thresholds (accuracy ≥ 0.85, F1 ≥ 0.80). Blocks registration if the model does not pass. |
| 4 — Register | `model_registration.py` | Writes gated models to the appropriate registry (MLflow for on-premise, Azure ML for cloud) as `ngt-sign-language` with versioning derived deterministically from the run ID. |

---

## Module table

### Pipeline components

| Module | Responsibility | Depends on |
|---|---|---|
| `preprocessing.py` | Stratified train/val/test split; offline augmentation on training partition only | `augmentation.py`, `data_loading.py` |
| `augmentation.py` | Image augmentation transforms — applied only to training data | — |
| `data_loading.py` | Dataset loading, class mapping, and ImageFolder utilities | `configuration.py` |
| `train.py` | Training entry point; `--val-dir` flag for pre-split or internal-split mode | `model_training.py`, `model_definitions.py` |
| `model_training.py` | Core training loop: forward pass, loss computation, optimiser step, metric logging | `model_definitions.py`, `logging_utils.py` |
| `model_definitions.py` | EfficientNet-B0 architecture definition for training | — |
| `model_evaluation.py` | Computes accuracy and F1; enforces configurable gate thresholds before registration | `configuration.py` |
| `model_registration.py` | Registers gated models to MLflow or Azure ML with run-ID-derived versioning | `azure_config.py`, `run_naming.py` |

### Orchestration

| Module | Responsibility | Depends on |
|---|---|---|
| `orchestration/pipeline_submitter.py` | Builds and submits the Azure ML pipeline graph; wires component functions into the four-stage DAG | `azure_config.py`, `orchestration/trigger_policy.py` |
| `orchestration/trigger_policy.py` | Defines the conditions under which retraining is triggered (e.g. new data version, scheduled interval) | `orchestration/training_state.py` |
| `orchestration/training_state.py` | Tracks training run state across submissions to avoid duplicate runs | — |
| `orchestration/dataset_inventory.py` | Inventories available data assets in the Azure ML registry | `azure_config.py` |

### Trigger API

A lightweight FastAPI service that exposes an HTTP endpoint for triggering retraining programmatically — for example from a CI pipeline or a data ingestion workflow.

| Module | Responsibility | Depends on |
|---|---|---|
| `trigger_api/app.py` | FastAPI app exposing `POST /trigger` retraining endpoint | `trigger_api/security.py`, `trigger_api/schemas.py` |
| `trigger_api/security.py` | API key authentication for the trigger endpoint | — |
| `trigger_api/schemas.py` | Pydantic request and response models for the trigger endpoint | — |
| `trigger_api/settings.py` | Trigger API environment settings (port, API key) | — |

### Configuration and utilities

| Module | Responsibility | Depends on |
|---|---|---|
| `configuration.py` | Dataclass-based pipeline configuration (epochs, batch size, augmentation copies, gate thresholds) | — |
| `azure_config.py` | Azure ML workspace, compute target, and environment configuration — shared by training, registry, trigger, and deployment scripts | `settings.py` |
| `run_naming.py` | Deterministic run name generation derived from run ID | — |
| `logging_utils.py` | MLflow metric and artefact logging helpers | — |
| `settings.py` | Training environment settings loaded from `.env` | — |
| `runtime.py` | Runtime environment detection (local vs Azure ML) | — |

---

## Submission scripts

All scripts live under `scripts/` at the repo root and are run via `poetry run python scripts/<name>.py`.

| Script | Purpose |
|---|---|
| `run_local_pipeline.py` | Runs the full four-stage pipeline locally using the same component functions as Azure ML |
| `submit_pipeline.py` | Submits the full pipeline to Azure ML. Skips preprocessing if `ngt-augmented-train` is already cached for the current raw data version. Use `--force-preprocess` to override. |
| `submit_training_job.py` | Submits training only (no preprocessing) to Azure ML |
| `submit_sweep_job.py` | Submits a hyperparameter sweep job to Azure ML |
| `register_raw_data.py` | Registers a local raw ImageFolder as a versioned Azure ML data asset (`ngt-raw`) |
| `register_data_splits.py` | Registers pre-computed dataset splits as versioned Azure ML data assets |
| `register_env.py` | Registers a conda environment in Azure ML. Required when package versions in `train-env-gpu.yml` change. |
| `get_azure_mlflow_uri.py` | Reads and prints the Azure ML MLflow tracking URI for use in `.env` |
| `smoke_test.py` | Submits a small Azure ML smoke job to verify connectivity before running the full pipeline |

---

## Model registry

Trained models are registered in Azure ML as `ngt-sign-language`. Version numbers increment automatically on each passing training run.

The inference API loads the latest registered model at startup when `AZURE_USE_REGISTRY_MODEL=true` is set in `.env`. The downloaded model is cached locally under `src/sign_language/models/registry_cache/v<N>/model.pth`. Subsequent restarts reuse the cached version unless the registry version has changed.

For local development, leave `AZURE_USE_REGISTRY_MODEL=false` — the local `models/best_ngt_model_v2.pth` is used instead.

---

## Deployment scripts

After a model passes the gate and is registered, it can be deployed to an Azure ML managed online endpoint.

| Script | Purpose |
|---|---|
| `deploy_online_endpoint.py` | Deploys a registered model version to an Azure ML managed online endpoint |
| `rollout_blue_green.py` | Blue/green traffic rollout between two endpoint deployments. `--rollback` switches traffic back to the previous deployment. |
| `rollout_canary.py` | Canary rollout with configurable traffic steps (e.g. `--steps 10,25,50,100`). Gradually shifts traffic from the stable deployment to the candidate. `--rollback` returns all traffic to stable. |
| `test_online_endpoint.py` | Smoke-tests a deployed online endpoint |
| `invoke_online_endpoint.py` | Sends a single real image request to the online endpoint |

---

## Dependencies

**Key external dependencies:**

| Dependency | Purpose |
|---|---|
| `torch` / `torchvision` | Model training and image transforms |
| `azure-ai-ml` | Azure ML SDK v2 — pipeline submission, job management, registry |
| `mlflow` | Experiment tracking and model registry (local and Azure ML) |
| `mediapipe` | Hand landmark extraction during preprocessing |
| `scikit-learn` | Stratified split, F1 computation |
| `fastapi` / `uvicorn` | Trigger API service |
| `pydantic` | Configuration and schema validation |
