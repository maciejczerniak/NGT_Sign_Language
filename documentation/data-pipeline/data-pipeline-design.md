# Data pipeline design

Thought process, design decisions, and rationale behind the NGT sign language recognition training pipeline.

---

## Contents

- [Overview](#overview)
- [Pipeline stages](#pipeline-stages)
- [Design decisions](#design-decisions)
  - [Why offline augmentation on training data only](#why-offline-augmentation-on-training-data-only)
  - [Why a stratified split](#why-a-stratified-split)
  - [Why the model gate exists](#why-the-model-gate-exists)
  - [Why the pipeline runs identically locally and on Azure ML](#why-the-pipeline-runs-identically-locally-and-on-azure-ml)
  - [Why data assets and environments are registered through code](#why-data-assets-and-environments-are-registered-through-code)
  - [Why preprocessing is cached](#why-preprocessing-is-cached)
  - [Why versioning is derived from the run ID](#why-versioning-is-derived-from-the-run-id)
- [Tradeoffs considered](#tradeoffs-considered)
- [Brief requirements addressed](#brief-requirements-addressed)
- [Diagrams](#diagrams)

---

## Overview

The training pipeline takes raw NGT fingerspelling image data as input and produces a registered, gated model as output. It runs as a two-step Azure ML pipeline job — preprocessing and training — using the same component functions that can also be executed locally, ensuring full environment parity.

The pipeline was designed around four core requirements from the creative brief:

- The system must support retraining as new data becomes available
- Model quality must be enforced programmatically before deployment
- The pipeline must be reproducible and auditable
- The pipeline must run on cloud infrastructure regardless of hardware specifications

---

## Pipeline stages

The pipeline runs as two steps in Azure ML — `preprocess_step` and `train` — with evaluation and registration happening inside the train step.

**Inputs**
- `ngt-raw` — versioned raw image data asset
- `ngt-pretrained-checkpoint` — pretrained EfficientNet-B0 checkpoint (`best_ngt_model_v2.pth`)

**Step 1 — NGT offline augmentation** (`preprocess_step`)
Receives `ngt-raw` as input. Applies a stratified 80/10/10 split, then generates ×4 augmented copies of the training partition only. Outputs three data streams: `augmented_data` (training), `val_data`, and `test_data`. The augmented training data is also registered as the `ngt-augmented-train` versioned asset for caching and reproducibility.

**Step 2 — NGT EfficientNet-B0 fine-tuning** (`train`)
Receives `augmented_data`, `val_data`, `test_data`, and `ngt-pretrained-checkpoint` as inputs. Fine-tunes EfficientNet-B0, logs metrics to MLflow, evaluates against the gate thresholds (accuracy ≥ 0.85, F1 ≥ 0.80), and registers the model as `ngt-sign-language` in the Azure ML registry if the gate passes. If the gate does not pass, no registration occurs. Outputs: `checkpoints` and `results`.

For the visual pipeline graph captured from a live Azure ML run, see [`diagrams/pipeline_graph.png`](./diagrams/pipeline_graph.png).

---

## Design decisions

### Why offline augmentation on training data only

Augmentation is applied **before training**, generating ×4 copies of each training image and registering them as a separate `ngt-augmented-train` data asset. Validation and test data are never augmented.

This was a deliberate choice for two reasons:

**Metric stability.** If augmentation were applied online (i.e. randomly during training), validation and test metrics would vary slightly between runs depending on which augmented versions were seen. By applying augmentation offline and freezing the augmented dataset as a registered asset, every training run that uses the same `ngt-augmented-train` version sees exactly the same data. Validation and test metrics are stable and directly comparable across runs.

**Reproducibility.** A registered `ngt-augmented-train` asset is versioned and immutable. Any team member or CI run can reproduce a training job exactly by referencing the same asset version, regardless of when the job is submitted.

---

### Why a stratified split

The 80/10/10 train/validation/test split is stratified by class label. NGT fingerspelling has unequal letter frequencies in natural signing — a random split risks under-representing rare letters in validation or test sets, which would produce misleadingly optimistic metrics on common letters and undetected poor performance on rare ones.

Stratification ensures every letter is proportionally represented in all three partitions, making validation and test metrics reliable indicators of real-world performance across the full alphabet.

---

### Why the model gate exists

No model reaches the registry without passing programmatically enforced thresholds (accuracy ≥ 0.85, F1 ≥ 0.80). If a training run produces a model that does not meet these thresholds, the gate inside the train step blocks registration and the pipeline completes without writing anything to the model registry.

This was chosen over a manual review process for two reasons:

**Consistency.** A manual review introduces subjectivity — different team members might accept or reject the same model based on different criteria. The gate enforces the same standard on every run automatically.

**Safety for automated retraining.** As new data becomes available and the pipeline is re-triggered, there is no human in the loop to catch a regression. The gate acts as an automated quality control layer that prevents a degraded model from being deployed.

The thresholds themselves are configurable via `configuration.py` — they can be tightened as the model matures without changing any pipeline logic.

---

### Why the pipeline runs identically locally and on Azure ML

The same component functions (`preprocessing.py`, `train.py`, `model_evaluation.py`, `model_registration.py`) are used in both `scripts/run_local_pipeline.py` (local execution) and the Azure ML pipeline submission (`scripts/submit_pipeline.py`).

This was a deliberate design choice over having separate local and cloud implementations:

**Comparable metrics.** If the local and cloud implementations diverged even slightly — different preprocessing steps, different augmentation logic — metrics would not be directly comparable. The same functions guarantee identical data transformations and training logic in both environments.

**Easier debugging.** Issues can be reproduced and debugged locally without submitting an Azure ML job, which saves compute cost and iteration time. Once the local run passes, the cloud run is expected to produce the same result.

**Reduced maintenance.** A single codebase for both environments means a bug fix or improvement only needs to be made once.

---

### Why data assets and environments are registered through code

Data assets (`register_raw_data.py`, `register_data_splits.py`), conda environments (`register_env.py`), and pipeline submissions are all done through scripts rather than through the Azure ML Studio UI.

**Version control.** Scripts committed to the repository mean every data registration, environment change, and pipeline submission is recorded in Git history. The state of the system at any point in time can be reconstructed from the repository alone.

**Reproducibility.** Any team member with appropriate Azure credentials can reproduce the exact same environment and data assets by running the same scripts. There is no dependency on undocumented UI steps.

**Auditability.** The creative brief requires the system to be auditable. Code-based registration provides a clear, inspectable record of what data was used, which environment was active, and what pipeline configuration was submitted for any given training run.

---

### Why preprocessing is cached

The submission script (`submit_pipeline.py`) checks whether a `ngt-augmented-train` asset already exists for the current version of `ngt-raw`. If it does, the preprocessing step is skipped and only the training step runs.

Offline augmentation generates ×4 copies of the training data. For a dataset of meaningful size, this is computationally expensive. Rerunning it every time a training hyperparameter is tuned — when the raw data has not changed — wastes compute time and cost.

Caching the preprocessed asset means repeated training experiments on the same data are significantly faster. The cache is invalidated automatically when a new version of `ngt-raw` is registered, ensuring the augmented dataset stays in sync with the source data.

---

### Why versioning is derived from the run ID

Registered model versions are derived deterministically from the Azure ML run ID rather than being assigned sequentially by a counter.

Sequential counters introduce a race condition when multiple training runs complete in parallel — two runs finishing simultaneously could attempt to register as the same version number. Run-ID-derived versioning is unique by construction and ties the registered model back to the specific job that produced it, making the registry fully traceable.

---

## Tradeoffs considered

| Decision | Alternative considered | Why we chose what we chose |
|---|---|---|
| Offline augmentation | Online augmentation during training | Metric stability and reproducibility outweigh the flexibility of online augmentation for this use case |
| Stratified split | Random split | Unequal class frequencies in NGT make random splits unreliable for rare letters |
| Programmatic gate | Manual model review | Consistency and safety for automated retraining |
| Same functions locally and on Azure ML | Separate local and cloud implementations | Comparable metrics and reduced maintenance overhead |
| Code-based asset registration | Azure ML Studio UI | Version control, reproducibility, and auditability |
| Cached preprocessing | Re-run preprocessing on every job | Compute cost and iteration speed |
| Run-ID-derived versioning | Sequential counter | Race condition safety and traceability |

---

## Brief requirements addressed

| Requirement | How the pipeline addresses it |
|---|---|
| Retraining support | Pipeline can be re-triggered via `submit_pipeline.py` or the trigger API when new data is available |
| Deployable on any infrastructure | Runs locally and on Azure ML with the same code; no cloud-specific logic in component functions |
| Auditable | All data assets, environments, and submissions are registered and tracked through code and versioned in Git |
| Quality enforced | Model gate blocks registration of any model that does not meet accuracy and F1 thresholds |
| Reproducible | Offline augmentation, registered data assets, and code-based environment registration ensure any run can be reproduced exactly |

---

## Diagrams

See [`diagrams/pipeline_graph.png`](./data-pipeline-diagram.png) for the Azure ML pipeline graph captured from a completed pipeline run, showing the two executed steps (`preprocess_step` and `train`) with their data inputs and outputs.
