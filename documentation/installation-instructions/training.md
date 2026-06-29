# Training — Installation & Usage

The training package (`sign_language_training`) handles model fine-tuning, data augmentation, and pipeline submission to Azure ML. Local training and cloud training use the same underlying functions.

## Requirements

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation)
- Azure ML credentials (for cloud training only)

## Installation

Training is installed as part of the main project. From the repository root:

```bash
poetry install
```

To verify the training package is available:

```bash
poetry run python -c "import sign_language_training; print('OK')"
```

## Local Training

### Fine-tune on existing data

Run the NGT training workflow with a pretrained checkpoint:

```bash
poetry run sign-language train \
    --pretrained-checkpoint models/best_ngt_model_v2.pth
```

With overrides:

```bash
poetry run sign-language train \
    --pretrained-checkpoint models/best_ngt_model_v2.pth \
    --epochs 20 --batch-size 16 --learning-rate 0.0001
```

### Run the full local retraining pipeline

Runs the complete pipeline locally (stratified split → augmentation → training → evaluation → model gate check), replicating the Azure ML graph on your machine:

```bash
poetry run sign-language local-pipeline \
    --pretrained-checkpoint models/best_ngt_model_v2.pth
```

With overrides:

```bash
poetry run sign-language local-pipeline \
    --raw-data-dir data/sample \
    --pretrained-checkpoint models/best_ngt_model_v2.pth \
    --epochs 5 --batch-size 8
```

## Cloud Training (Azure ML)

Ensure your Azure ML credentials are configured first — see [Azure ML Configuration](#azure-ml-configuration) below.

### Submit a training job

```bash
poetry run python scripts/submit_training_job.py
```

### Submit a hyperparameter sweep job

```bash
poetry run python scripts/submit_sweep_job.py
```

### Submit the full pipeline

```bash
poetry run python scripts/submit_pipeline.py
```

### Promote a trained model to champion (Azure ML registry)

Promote a specific version by number:

```bash
poetry run python scripts/promote_model.py --version 4277446031
```

Promote the best trial from a sweep (by F1 macro):

```bash
poetry run python scripts/promote_model.py --sweep-id <sweep-run-id>
```

Preview changes without applying them:

```bash
poetry run python scripts/promote_model.py --version 4277446031 --dry-run
```

### Promote a trained model to champion (MLflow registry)

Promote a specific version:

```bash
poetry run python scripts/promote_mlflow_champion.py --version 5
```

Promote whatever version currently holds the `@candidate` alias:

```bash
poetry run python scripts/promote_mlflow_champion.py --from-candidate
```

Preview changes without applying them:

```bash
poetry run python scripts/promote_mlflow_champion.py --version 5 --dry-run
```

## MLflow Tracking

Training metrics (loss, accuracy, per-class precision/recall) are logged automatically via MLflow autologging. To view runs locally:

```bash
poetry run mlflow ui --backend-store-uri file:./logs/mlflow
```

Then open `http://localhost:5000` in your browser.

## Azure ML Configuration

Ensure the following variables are set in your `.env` file before submitting cloud jobs:

```text
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=your-resource-group
AZURE_WORKSPACE=your-workspace-name
```

## Full Reference

See the [Sphinx documentation](https://cautious-carnival-p365vjk.pages.github.io) for the complete `sign_language_training` API reference.
