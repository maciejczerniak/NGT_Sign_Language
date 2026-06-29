# Azure / MLflow script reference

This folder contains utility scripts for Azure ML training, retraining automation, model promotion, online endpoint deployment, endpoint testing, and the on-prem MLflow workflow.

Run commands from the repository root:

```bash
poetry run python scripts/<script_name>.py --help
```

Most Azure scripts use the project Azure configuration from `.env`, such as subscription, resource group, workspace, compute target, environment, raw data asset, checkpoint asset, and instance type.

## Common workflow

Typical Azure workflow:

```text
1. Register environment.
2. Register raw data.
3. Submit smoke test.
4. Submit training / pipeline / sweep.
5. Finalize completed sweeps and promote the best model.
6. Deploy promoted model to online endpoint.
7. Test endpoint.
8. Use scheduled trigger for automatic retraining.
```

---

## Azure ML setup scripts

### `register_env.py`

Registers an Azure ML environment asset from a conda YAML file. Use this before submitting Azure ML jobs when the environment does not exist yet.

Key flags:

| Flag | Meaning |
|---|---|
| `--env-name` | Azure ML environment name. Falls back to `AZURE_ENVIRONMENT_NAME`. |
| `--env-version` | Environment version. Falls back to `AZURE_ENVIRONMENT_VERSION`. |
| `--conda-file` | Path to the conda YAML file. |
| `--gpu/--cpu` | Use CUDA GPU base image or CPU base image. |

Example:

```bash
poetry run python scripts/register_env.py \
  --env-name sign-language-training-env-gpu \
  --env-version 7 \
  --conda-file src/sign_language_training/train-env-gpu.yml \
  --gpu
```

### `register_raw_data.py`

Registers the raw NGT ImageFolder dataset as one Azure ML `URI_FOLDER` data asset. It also stores metadata tags such as `image_count` and `manifest_hash`.

Key flags:

| Flag | Meaning |
|---|---|
| `--data-dir` | Local raw ImageFolder path. |
| `--data-uri` | Existing Azure ML datastore URI for data already uploaded. |
| `--asset-name` | Azure ML data asset name. |
| `--version` | Data asset version. |
| `--image-count` | Manual image count metadata override. |
| `--manifest-hash` | Manual dataset hash metadata override. |

Example:

```bash
poetry run python scripts/register_raw_data.py \
  --data-dir data/raw \
  --asset-name ngt-raw \
  --version 2
```

Example with datastore URI:

```bash
poetry run python scripts/register_raw_data.py \
  --data-uri azureml://datastores/workspaceblobstore/paths/datasets/ngt/raw/current/ \
  --asset-name ngt-raw \
  --version 3 \
  --image-count 1099
```

### Azure frontend collection Blob storage

The Azure frontend collection page sends labelled images to:

```text
POST /api/collect
```

The Azure API validates each image, converts it to JPEG, and uploads it to a
private Azure Blob container:

```text
signlang-r2-collected/
  pending/
    A/
      <uuid>.jpg
    B/
      <uuid>.jpg
```

Each uploaded blob includes metadata:

| Metadata | Meaning |
|---|---|
| `letter` | Submitted class label. |
| `source` | `camera`, `upload`, or `auto`. |
| `language` | Sign language identifier, currently `NGT`. |
| `review_status` | Initially `pending`. |
| `collected_at` | UTC collection timestamp. |

The pending collection container is separate from the approved `ngt-raw`
training data. Collected images must be reviewed before they are added to a new
training data asset version.

#### Create the private container

Authenticate first:

```powershell
az login
az account set --subscription "<subscription-id>"
```

Create the dedicated private container:

```powershell
az storage container create `
  --account-name staswegend46479454361583 `
  --name signlang-r2-collected `
  --auth-mode key
```

Verify that public access is disabled:

```powershell
az storage container show `
  --account-name staswegend46479454361583 `
  --name signlang-r2-collected `
  --auth-mode key `
  --query "{name:name,publicAccess:properties.publicAccess}" `
  --output table
```

`publicAccess` should be empty or `None`.

#### Generate the upload SAS

The deployed Azure API requires a container-scoped SAS token to create and
write collected images. Generate a one-year token:

```powershell
$expiry = (Get-Date).ToUniversalTime().AddYears(1).ToString("yyyy-MM-ddTHH:mmZ")

$SAS = az storage container generate-sas `
  --account-name staswegend46479454361583 `
  --name signlang-r2-collected `
  --permissions cw `
  --expiry $expiry `
  --https-only `
  --auth-mode key `
  --output tsv

$SAS
```

Store the printed value as the GitHub Actions repository secret:

```text
AZURE_COLLECT_SAS_TOKEN
```

Do not add the SAS token to `.env`, source control, logs, or documentation.
The deployment workflow stores it as the `collect-sas-token` Container App
secret and exposes it to the Azure API through
`AZURE_API_COLLECT_SAS_TOKEN`.

After adding or renewing the GitHub secret, rerun
`.github/workflows/deploy-azure.yml` so the Container App receives the current
token.

#### Verify collection uploads

The collection page marks a sample:

- `Cloud` when `/api/collect` successfully uploads it.
- `Local only` when the upload fails and the sample exists only in browser
  local storage.

`Local only` samples are not retried automatically.

List pending blobs:

```powershell
az storage blob list `
  --account-name staswegend46479454361583 `
  --container-name signlang-r2-collected `
  --prefix "pending/" `
  --auth-mode key `
  --query "[].{name:name,size:properties.contentLength}" `
  --output table
```

Inspect Azure API logs when uploads fail:

```powershell
az containerapp logs show `
  --name signlang-r2-api `
  --resource-group buas-y2 `
  --tail 200 `
  --format text
```

Common failures:

| Symptom | Likely cause |
|---|---|
| `POST /api/collect` returns `404` | An older Azure API image without the collection route is deployed. |
| `POST /api/collect` returns `503` | SAS is missing, expired, or does not permit Blob creation/writes. |
| Frontend displays `Local only` | The backend request failed; inspect Container App logs. |

#### Download pending samples for review

Download all pending samples:

```powershell
az storage blob download-batch `
  --account-name staswegend46479454361583 `
  --source signlang-r2-collected `
  --destination data/collected-review `
  --pattern "pending/*" `
  --auth-mode key
```

The local review directory will contain:

```text
data/collected-review/
  pending/
    A/
      <uuid>.jpg
    B/
      <uuid>.jpg
```

Reviewers should verify the label, hand visibility, image quality, duplicates,
and inappropriate or sensitive content. Copy only approved images into the
corresponding complete local dataset directory:

```text
data/raw/<LETTER>/
```

Delete a rejected pending blob only after the review decision is confirmed:

```powershell
az storage blob delete `
  --account-name staswegend46479454361583 `
  --container-name signlang-r2-collected `
  --name "pending/<LETTER>/<uuid>.jpg" `
  --auth-mode key
```

Then register the complete approved dataset as the next `ngt-raw` version:

```powershell
poetry run python scripts/register_raw_data.py `
  --data-dir data/raw `
  --asset-name ngt-raw `
  --version <next-version>
```

The scheduled checker selects the highest numeric `ngt-raw` version. It does
not combine multiple data asset versions, so every registered version used for
training must reference or contain the complete approved dataset.

#### SAS renewal and incident response

- Renew the SAS before its expiry date and update `AZURE_COLLECT_SAS_TOKEN`.
- Rerun the Azure deployment workflow after replacing the secret.
- Generate a replacement immediately if the SAS is exposed.
- Revoke existing SAS tokens by rotating the storage account key used to sign
  them. Key rotation may affect other users of that storage account and should
  be coordinated with its administrator.

### `register_data_splits.py`

Registers existing `train`, `test`, and `val` ImageFolder directories as separate Azure ML data assets: `ngt-train`, `ngt-test`, and `ngt-val`.

Key flags:

| Flag | Meaning |
|---|---|
| `--data-root` | Folder containing `train/`, `test/`, and `val/`. |
| `--version` | Version applied to all three registered assets. |

Example:

```bash
poetry run python scripts/register_data_splits.py \
  --data-root data/ngt_subset1 \
  --version 1
```

### `get_azure_mlflow_uri.py`

Prints the Azure ML workspace MLflow tracking URI.

Example:

```bash
poetry run python scripts/get_azure_mlflow_uri.py
```

### `smoke_test.py`

Submits a small Azure ML smoke test job. It validates Azure client access, compute/environment resolution, package installation, imports, CLI availability, and MLflow logging.

Key flag:

| Flag | Meaning |
|---|---|
| `--instance-type` | Optional Kubernetes instance type, for example `cpu-small`, `cpu-xl`, or `gpu`. |

Example:

```bash
poetry run python scripts/smoke_test.py --instance-type cpu-small
```

---

## Azure ML training and retraining scripts

### `submit_training_job.py`

Submits a single Azure ML command job for NGT training. It consumes a raw data asset and pretrained checkpoint asset, installs the training package, runs training, logs metrics, and writes checkpoints/results.

Key flags:

| Flag | Meaning |
|---|---|
| `--job-name` | Optional Azure ML job name. |
| `--experiment-name` | Azure ML experiment name. |
| `--data-asset` | Raw data asset reference, for example `azureml:ngt-raw:2`. |
| `--pretrained-checkpoint` | Pretrained checkpoint asset or path. |
| `--batch-size` | Training batch size. |
| `--epochs` | Number of training epochs. |
| `--learning-rate` | Optimizer learning rate. |
| `--img-size` | Input image size. |
| `--val-split` | Validation split ratio. |
| `--target-accuracy` | Accuracy gate threshold. |
| `--f1-threshold` | Macro-F1 gate threshold. |
| `--mlflow-enabled/--no-mlflow-enabled` | Enable or disable MLflow in the job. |

Example:

```bash
poetry run python scripts/submit_training_job.py \
  --experiment-name sign-language-training \
  --data-asset azureml:ngt-raw:2 \
  --epochs 5 \
  --batch-size 16 \
  --mlflow-enabled
```

### `submit_pipeline.py`

Submits the full Azure ML preprocessing + training pipeline. The pipeline performs stratified split and offline augmentation, then runs training and evaluation.

Key flags:

| Flag | Meaning |
|---|---|
| `--experiment-name` | Azure ML experiment name. |
| `--display-name` | Display name in Azure ML Studio. |
| `--data-asset` | Raw data asset reference. |
| `--ngt-raw-version` | Raw data version used for cache tracking. |
| `--pretrained-checkpoint` | Pretrained checkpoint asset reference. |
| `--augmented-asset-name` | Name used for cached augmented data. |
| `--augment-copies` | Number of augmented copies per image. |
| `--batch-size`, `--epochs`, `--learning-rate` | Training settings. |
| `--force-preprocess` | Re-run preprocessing even when cache exists. |

Example:

```bash
poetry run python scripts/submit_pipeline.py \
  --data-asset azureml:ngt-raw:2 \
  --ngt-raw-version 2 \
  --epochs 10 \
  --batch-size 16 \
  --force-preprocess
```

### `submit_sweep_job.py`

Submits an Azure ML hyperparameter sweep. The sweep tries different learning rates, batch sizes, and patience values.

Key flags:

| Flag | Meaning |
|---|---|
| `--params-file` | JSON file with sweep defaults. CLI flags override this file. |
| `--job-name` | Optional sweep job name. |
| `--experiment-name` | Azure ML experiment name. |
| `--data-asset` | Raw data asset reference. |
| `--pretrained-checkpoint` | Pretrained checkpoint reference. |
| `--instance-type` | Kubernetes instance type override. |
| `--epochs` | Epochs per trial. |
| `--learning-rate-min`, `--learning-rate-max` | Log-uniform learning-rate range. |
| `--batch-sizes` | Comma-separated batch sizes, for example `8,16,32`. |
| `--patience-values` | Comma-separated patience values. |
| `--primary-metric` | Metric optimized by the sweep, usually `best_val_accuracy`. |
| `--max-total-trials` | Total number of trials. |
| `--max-concurrent-trials` | Parallel trials. |
| `--use-median-stopping` | Enable Azure median early stopping. |

Tiny validation sweep:

```bash
poetry run python scripts/submit_sweep_job.py \
  --max-total-trials 2 \
  --max-concurrent-trials 1 \
  --epochs 1 \
  --batch-sizes 8,16 \
  --patience-values 3,5
```

Real sweep:

```bash
poetry run python scripts/submit_sweep_job.py \
  --max-total-trials 12 \
  --max-concurrent-trials 2 \
  --epochs 30 \
  --batch-sizes 8,16,32 \
  --patience-values 5,7,10
```

---

## Azure ML retraining automation

### `create_training_schedule.py`

Creates or updates an Azure ML schedule that runs the retraining checker. The checker evaluates data-change first and scheduled fallback second, then submits a retraining sweep only when policy conditions pass.

Key flags:

| Flag | Meaning |
|---|---|
| `--schedule-name` | Azure ML schedule name. |
| `--cron` | Cron expression. |
| `--time-zone` | Time zone used by the Azure ML cron trigger. |
| `--data-asset` | Raw data asset name to inspect. |
| `--min-new-images` | Minimum changed image count required. |
| `--interval-days` | Minimum fallback interval between retraining runs. |
| `--experiment-name` | Azure ML experiment name. |

Example:

```bash
poetry run python scripts/create_training_schedule.py \
  --schedule-name sign-language-training-trigger-daily \
  --cron "0 8 * * *" \
  --time-zone UTC \
  --min-new-images 10 \
  --interval-days 7
```

### `check_training_triggers.py`

Evaluates Azure ML retraining triggers using Azure ML metadata as the source of truth. It checks active retraining jobs, latest raw data asset metadata, last completed retraining job, data changes, and interval fallback. It also finalizes completed sweeps before deciding whether to submit a new job.

Key flags:

| Flag | Meaning |
|---|---|
| `--asset-name` | Azure ML raw data asset name, usually `ngt-raw`. |
| `--min-new-images` | Minimum new images for data-change retraining. |
| `--interval-days` | Minimum days between fallback retraining. |
| `--experiment-name` | Azure ML experiment name. |
| `--force` | Submit retraining regardless of policy. |
| `--submit-kind` | `sweep` or `train`. |
| `--model-name` | Model name used for sweep finalization. |
| `--archive-non-best/--keep-non-best` | Archive or keep non-best models after sweep finalization. |

Example:

```bash
poetry run python scripts/check_training_triggers.py \
  --asset-name ngt-raw \
  --min-new-images 10 \
  --interval-days 7 \
  --submit-kind sweep
```

Force retraining:

```bash
poetry run python scripts/check_training_triggers.py --force --submit-kind sweep
```

### `check_data_change_and_train.py`

Older/simple local-state based Azure retraining trigger. It checks local `data/raw`, compares with a local state JSON, and submits Azure ML retraining if enough new images are found.

Key flags:

| Flag | Meaning |
|---|---|
| `--data-dir` | Local ImageFolder dataset root. |
| `--state-path` | Local trigger state JSON. |
| `--min-new-images` | Minimum new image threshold. |
| `--force/--no-force` | Force retraining regardless of threshold. |

Example:

```bash
poetry run python scripts/check_data_change_and_train.py \
  --data-dir data/raw \
  --min-new-images 100
```

### `trigger_training.py`

General-purpose manual/data-change/scheduled trigger for Azure ML retraining. It uses the shared trigger policy and submits a preprocessing + training pipeline if allowed.

Key flags:

| Flag | Meaning |
|---|---|
| `--reason` | `manual`, `data_change`, or `scheduled`. |
| `--force/--no-force` | Force submission. |
| `--data-dir` | Local ImageFolder dataset root. |
| `--state-path` | Local state JSON. |
| `--min-new-images` | New image threshold. |
| `--interval-days` | Scheduled fallback interval. |

Examples:

```bash
poetry run python scripts/trigger_training.py --reason manual --force
poetry run python scripts/trigger_training.py --reason data_change --min-new-images 100
poetry run python scripts/trigger_training.py --reason scheduled --interval-days 7
```

### `finalize_completed_sweeps.py`

Finds completed retraining sweeps tagged as pending, selects the best registered model from each completed sweep by accuracy, promotes it, demotes the old promoted model, and marks the sweep finalized.

Expected job tags:

```text
purpose=retraining-sweep
finalization_status=pending
```

Key flags:

| Flag | Meaning |
|---|---|
| `--model-name` | Azure ML registered model name to promote. |
| `--purpose-tag` | Job tag used to identify retraining sweeps. |
| `--pending-status` | `finalization_status` value treated as pending. |
| `--limit` | Maximum pending jobs inspected. |
| `--mark-failed/--no-mark-failed` | Mark failed/cancelled sweeps as finalized failures. |
| `--dry-run` | Preview changes without modifying Azure ML. |
| `--yes` | Apply changes without confirmation. |

Examples:

```bash
poetry run python scripts/finalize_completed_sweeps.py --dry-run
poetry run python scripts/finalize_completed_sweeps.py --yes
```

---

## Azure model promotion and endpoint deployment

### `promote_model.py`

Promotes an Azure ML registered model version for serving. It sets `promoted=true` on the target version and clears it from other versions.

Key flags:

| Flag | Meaning |
|---|---|
| `--model-name` | Azure ML model name. |
| `--version` | Exact model version to promote. |
| `--sweep-id` | Promote best version from a sweep. |
| `--dry-run` | Preview tag changes. |
| `--yes` | Apply without confirmation. |

Examples:

```bash
poetry run python scripts/promote_model.py --version 2481833779 --yes
poetry run python scripts/promote_model.py --sweep-id <sweep_job_id> --yes
poetry run python scripts/promote_model.py --version 2481833779 --dry-run
```

### `deploy_online_endpoint.py`

Deploys a registered Azure ML model as a Kubernetes online endpoint deployment. It can deploy an explicit model version, the promoted version, or the latest version.

Key flags:

| Flag | Meaning |
|---|---|
| `--endpoint-name` | Online endpoint name. |
| `--deployment-name` | Deployment name inside the endpoint, for example `blue`. |
| `--model-name` | Registered model name. |
| `--model-version` | Explicit model version. |
| `--promoted` | Deploy the version tagged `promoted=true`. |
| `--latest` | Deploy Azure ML latest version. |
| `--traffic-percent` | Traffic assigned to this deployment. |
| `--instance-type` | Kubernetes serving instance type. |
| `--instance-count` | Number of serving instances. |
| `--auth-mode` | Endpoint auth mode, usually `key`. |
| `--source-revision` | Optional commit SHA tag for CI/CD traceability. |

Example:

```bash
poetry run python scripts/deploy_online_endpoint.py \
  --endpoint-name ngt-sign-language-blue-green \
  --deployment-name blue \
  --model-name ngt-sign-language \
  --promoted \
  --traffic-percent 100 \
  --instance-type gpu
```

### `rollout_blue_green.py`

Blue/green rollout helper. It deploys the inactive color (`blue` or `green`) and switches all endpoint traffic after deployment. It also supports rollback and staged activation.

Key flags:

| Flag | Meaning |
|---|---|
| `--endpoint-name` | Online endpoint name. |
| `--model-name` | Registered model name. |
| `--model-version` | Explicit model version. |
| `--promoted` | Use promoted model. |
| `--latest` | Use latest model. |
| `--instance-type` | Serving instance type. |
| `--instance-count` | Number of instances. |
| `--rollback` | Switch traffic back to inactive color. |
| `--stage-only` | Deploy inactive color at 0% traffic only. |
| `--activate-staged` | Activate already staged deployment. |
| `--reuse-running` | Skip rollout if active deployment is reusable. |
| `--source-revision` | Commit SHA / source revision tag. |

Examples:

```bash
poetry run python scripts/rollout_blue_green.py --promoted --instance-type gpu
poetry run python scripts/rollout_blue_green.py --promoted --stage-only
poetry run python scripts/rollout_blue_green.py --activate-staged
poetry run python scripts/rollout_blue_green.py --rollback
```

### `rollout_canary.py`

Canary rollout helper. It deploys a candidate deployment and gradually shifts traffic from the stable deployment to the candidate.

Key flags:

| Flag | Meaning |
|---|---|
| `--endpoint-name` | Online endpoint name. |
| `--stable-deployment` | Current live deployment name. |
| `--candidate-deployment` | Candidate deployment name. |
| `--model-name` | Registered model name. |
| `--model-version` | Explicit model version. |
| `--promoted` | Use promoted model. |
| `--latest` | Use latest model. |
| `--instance-type` | Serving instance type. |
| `--instance-count` | Number of candidate instances. |
| `--steps` | Comma-separated candidate traffic percentages. |
| `--wait-seconds` | Delay between traffic shifts. |
| `--rollback` | Restore traffic to stable deployment. |

Example:

```bash
poetry run python scripts/rollout_canary.py \
  --stable-deployment blue \
  --candidate-deployment green \
  --promoted \
  --steps 10,25,50,100 \
  --wait-seconds 60
```

Rollback:

```bash
poetry run python scripts/rollout_canary.py \
  --stable-deployment blue \
  --candidate-deployment green \
  --rollback
```

### `endpoint_common.py`

Shared helper module used by endpoint deployment scripts. It is not normally run directly.

It provides Azure ML client creation, model version resolution, endpoint environment resolution, endpoint compute resolution, endpoint instance type validation, model reference construction, deployment tag construction, and readable CLI output helpers.

Used by:

```text
deploy_online_endpoint.py
rollout_blue_green.py
rollout_canary.py
```

---

## Endpoint testing scripts

### `invoke_online_endpoint.py`

Sends one local image to the configured Azure ML online endpoint using endpoint URL/key from settings.

Key flag:

| Flag | Meaning |
|---|---|
| `--image-path` | Local image file to send. |

Example:

```bash
poetry run python scripts/invoke_online_endpoint.py \
  --image-path src/frontend/public/signs/a_photo.jpg
```

### `test_online_endpoint.py`

Runs synthetic tests directly against an Azure ML online endpoint. It sends a valid image, invalid base64, and missing-field payload.

Key flags:

| Flag | Meaning |
|---|---|
| `--endpoint-url` | Override endpoint scoring URL. |
| `--endpoint-key` | Override endpoint key. |
| `--deployment-name` | Optional deployment target. |
| `--max-latency-seconds` | Maximum accepted latency for valid request. |

Example:

```bash
poetry run python scripts/test_online_endpoint.py \
  --deployment-name blue \
  --max-latency-seconds 30
```

### `live_test.py`

Runs a webcam-based WebSocket test against the API. It captures frames, streams them to `/ws/predict`, and overlays predictions.

Key flags:

| Flag | Meaning |
|---|---|
| `--url` | WebSocket URL, default `ws://localhost:8000/ws/predict`. |
| `--camera` | OpenCV camera index. |
| `--fps` | Target frames per second. |

Examples:

```bash
poetry run python scripts/live_test.py \
  --url ws://localhost:8000/ws/predict
```

```bash
poetry run python scripts/live_test.py \
  --url wss://<container-app-fqdn>/ws/predict
```

Controls:

```text
Q = quit
R = reset sequence/smoother
S = pause/resume frame sending
D = toggle skeleton overlay
```

---

## On-prem / MLflow scripts

### `run_local_pipeline.py`

Runs the full preprocess → train workflow locally. It mirrors the Azure ML pipeline but runs on a local machine or inside the on-prem training container.

Key flags:

| Flag | Meaning |
|---|---|
| `--raw-data-dir` | Raw ImageFolder dataset. |
| `--output-dir` | Output root for preprocessed data, checkpoints, and results. |
| `--pretrained-checkpoint` | Local checkpoint path. |
| `--pretrain-from-mlflow` | Download current `@champion` checkpoint from MLflow. |
| `--register-as-candidate` | Register trained checkpoint as MLflow `@candidate`. |
| `--model-name` | MLflow registered model name. |
| `--augment-copies` | Augmented copies per image. |
| `--batch-size`, `--epochs`, `--learning-rate` | Training settings. |
| `--target-accuracy`, `--f1-threshold` | Gate thresholds. |
| `--mlflow/--no-mlflow` | Enable or disable MLflow logging. |
| `--skip-preprocess` | Reuse existing preprocessing output. |
| `--clean` | Clean output directory before running. |

Local example:

```bash
poetry run python scripts/run_local_pipeline.py \
  --raw-data-dir data/raw \
  --output-dir outputs/local_pipeline \
  --pretrained-checkpoint models/best_ngt_model_v2.pth \
  --epochs 5 \
  --batch-size 8 \
  --num-workers 0
```

On-prem MLflow example:

```bash
python scripts/run_local_pipeline.py \
  --raw-data-dir /data \
  --output-dir /outputs \
  --pretrain-from-mlflow \
  --register-as-candidate \
  --mlflow \
  --num-workers 0 \
  --clean
```

### `check_training_triggers_local.py`

On-prem analogue of `check_training_triggers.py`. It checks local ImageFolder changes and MLflow registered model timestamps. If training is due, it runs `run_local_pipeline.py` as a subprocess and registers a new `@candidate` if the gate passes.

Key flags:

| Flag | Meaning |
|---|---|
| `--data-dir` | Local ImageFolder dataset root. |
| `--output-dir` | Pipeline output root. |
| `--state-path` | Local trigger state JSON. |
| `--model-name` | MLflow registered model name. |
| `--min-new-images` | Minimum changed image count. |
| `--interval-days` | Fallback interval. |
| `--epochs` | Optional pipeline epoch override. |
| `--batch-size` | Optional batch-size override. |
| `--num-workers` | DataLoader workers. |
| `--force` | Force training. |

Example:

```bash
python scripts/check_training_triggers_local.py \
  --data-dir /data \
  --output-dir /outputs \
  --min-new-images 10 \
  --interval-days 7
```

### `promote_mlflow_champion.py`

Promotes an MLflow model version to the `@champion` alias for on-prem serving. The backend with `DEPLOY_TARGET=onprem` loads `models:/<name>@champion`.

Key flags:

| Flag | Meaning |
|---|---|
| `--model-name` | MLflow registered model name. |
| `--version` | Exact version to promote. |
| `--from-candidate` | Promote the current `@candidate`. |
| `--dry-run` | Preview alias change. |
| `--yes` | Apply without confirmation. |

Examples:

```bash
poetry run python scripts/promote_mlflow_champion.py --from-candidate --yes
poetry run python scripts/promote_mlflow_champion.py --version 5 --yes
```

### `upload_to_mlflow.py`

One-off script to seed the MLflow model registry with local checkpoint files. It uploads EfficientNet and optionally Landmark MLP checkpoints, registers them, and assigns `@champion`.

Required environment variable:

```bash
export MLFLOW_TRACKING_URI=http://<host>:2027
```

Example:

```bash
poetry run python scripts/upload_to_mlflow.py
```

### `upload_training_data_mlflow.py`

Uploads local ImageFolder training data to MinIO as a zip file for the on-prem training-data volume workflow.

Required environment variables:

```bash
export MINIO_ENDPOINT_URL=http://<host>:2028
export MINIO_ACCESS_KEY=<key>
export MINIO_SECRET_KEY=<secret>
```

Key flags:

| Flag | Meaning |
|---|---|
| `--source-dir` | Local ImageFolder root. |
| `--bucket` | MinIO bucket name. |
| `--object-name` | Object name in the bucket. |
| `--endpoint-url` | MinIO endpoint URL override. |
| `--keep-zip` | Keep the local zip file after upload. |

Example:

```bash
poetry run python scripts/upload_training_data_mlflow.py \
  --source-dir data/raw/training \
  --bucket training-data \
  --object-name training.zip
```

---

## Notes

- Azure ML scripts generally require Azure authentication and `.env` workspace settings.
- GitHub Actions / Azure jobs should use non-interactive auth.
- Local manual runs usually work with Azure CLI login or interactive browser auth, depending on `AZURE_AUTH_MODE`.
- Promotion and deployment are separate steps:
  - `promote_model.py` or `finalize_completed_sweeps.py` changes the model registry tag.
  - `deploy_online_endpoint.py`, `rollout_blue_green.py`, or GitHub Actions deploys the promoted version to the serving endpoint.
- `endpoint_common.py` is a helper module and should not normally be run directly.
