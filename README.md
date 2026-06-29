# Sign Language Project

This project is part of Block D (ADS-AI).
It uses Poetry for dependency management and virtual environments, and it follows a `src/`-based Python package structure.

The repository contains two packages:
- `src/sign_language/` — inference API (FastAPI + PyTorch + MediaPipe)
- `src/sign_language_training/` — standalone Azure ML training package

## Requirements

- Python 3.11
- Poetry
- Git + Git LFS
- Make (see below)
- Docker (for containerised deployment)
- Azure CLI (`az`) — required for Azure ML job submission and registry access

## Install Make

### Windows

```powershell
winget install GnuWin32.Make
[System.Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files (x86)\GnuWin32\bin", "User")
```

Restart PowerShell, then verify: `make --version`

**Alternative:** Use Git Bash — Make ships with Git for Windows.

### macOS

```bash
xcode-select --install
```

### Linux

```bash
sudo apt install make
```

## Install Poetry

https://python-poetry.org/docs/

## Install Azure CLI

https://learn.microsoft.com/en-us/cli/azure/install-azure-cli

After installation, log in:

```powershell
az login
```

## Install Git LFS

```powershell
winget install GitHub.GitLFS   # Windows
brew install git-lfs            # macOS
sudo apt install git-lfs        # Ubuntu/Debian
git lfs install                 # all platforms — run once per machine
```

### Tracked file types

| Pattern | Contents |
|---------|----------|
| `models/*.pth` | PyTorch model weights |
| `models/*.pt` | PyTorch model checkpoints |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-researchgroup2.git
cd 2025-26d-fai2-adsai-group-researchgroup2
```

### 2. Pull LFS files

```powershell
git lfs pull
```

### 3. Use Python 3.11

```powershell
poetry env use 3.11
```

### 4. Install the project and dependencies

**CPU (default — required for Docker and CI):**

```powershell
make install
```

**GPU (optional — NVIDIA GPU only):**

```powershell
make install-gpu
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

```powershell
copy .env.example .env
```

See [Environment Variables](#environment-variables) for a full reference.

---

## Usage

### CLI

#### Run inference on an image

```bash
poetry run sign-language predict --image path/to/hand.jpg
```

With custom model checkpoints:

```bash
poetry run sign-language predict --image path/to/hand.jpg \
    --model models/best_ngt_model_v2.pth \
    --lm-model models/best_landmark_mlp.pth \
    --landmarker hand_landmarker.task
```

Additional options:

```bash
poetry run sign-language predict --image hand.jpg --top-k 5 --verbose
```

#### Start the API server

Development mode:

```bash
poetry run sign-language serve --reload
```

Production mode:

```bash
poetry run sign-language serve --host 0.0.0.0 --port 8000 --workers 4
```

#### Run local training

```bash
poetry run sign-language train \
    --data-dir data/raw \
    --pretrained-checkpoint models/best_ngt_model_v2.pth
```

---

## Training Pipeline (Azure ML)

The training pipeline runs on Azure ML and consists of two steps:

1. **Preprocessing** — stratified 80/10/10 split + offline augmentation (×4 copies per image). Registers the augmented train split as `ngt-augmented-train` data asset.
2. **Training** — fine-tunes EfficientNet-B0, evaluates, gates on accuracy ≥ 0.85 and F1 ≥ 0.80, and registers the model as `ngt-sign-language` in Azure ML.

### Submit the full pipeline

```powershell
poetry run python scripts/submit_pipeline.py --epochs 60 --augment-copies 4
```

The script checks whether a cached `ngt-augmented-train` asset already exists for the current `ngt-raw` version. If it does, preprocessing is skipped and only the training step runs.

```powershell
# Force re-augmentation (e.g. after uploading new raw data as version 2)
poetry run python scripts/submit_pipeline.py --ngt-raw-version 2 --force-preprocess
```

### Submit training only (no preprocessing)

```powershell
poetry run python scripts/submit_training_job.py --epochs 60
```

### Register the Azure ML environment

Only needed when changing package versions in `train-env-gpu.yml`:

```powershell
poetry run python scripts/register_env.py \
  --env-name sign-language-training-env-gpu \
  --env-version 9 \
  --conda-file src/sign_language_training/train-env-gpu.yml \
  --gpu
```

Update `.env` to point to the new version:

```
AZURE_ENVIRONMENT_VERSION=9
```

---

## Model Registry

Trained models are registered in Azure ML as `ngt-sign-language` with version numbers incremented automatically on each passing training run.

### Load the latest registered model on API startup

Set in `.env`:

```
AZURE_USE_REGISTRY_MODEL=true
```

When enabled, the inference API downloads the latest registered model from Azure ML on startup and caches it locally under `src/sign_language/models/registry_cache/v<N>/model.pth`. Subsequent restarts reuse the cached version unless the registry version has changed.

Leave `AZURE_USE_REGISTRY_MODEL=false` (default) for local development — the local `models/best_ngt_model_v2.pth` is used instead.

---

## Cloud Deployment (Azure ML Endpoints)

Epic 1 deployment uses Azure ML endpoints for serving registered model versions.
The deployment scripts use the existing Azure ML SDK settings from `.env` and
prefer the dedicated inference environment variables
`AZURE_INFERENCE_ENVIRONMENT_NAME` / `AZURE_INFERENCE_ENVIRONMENT_VERSION`.
If those are empty, the scripts fall back to `AZURE_ENVIRONMENT_NAME` /
`AZURE_ENVIRONMENT_VERSION`.

The scripts follow the repository's existing Azure configuration convention:

```env
AZURE_SUBSCRIPTION_ID="<subscription-id>"
AZURE_RESOURCE_GROUP="<resource-group>"
AZURE_WORKSPACE="<azure-ml-workspace-name>"
```

Use `AZURE_WORKSPACE`, not `AZURE_ML_WORKSPACE_NAME`.
Endpoint names such as `ngt-sign-language-online` are passed as script
arguments, not read from `.env`.

### Deploy an online endpoint

```powershell
poetry run python scripts/deploy_online_endpoint.py `
  --endpoint-name ngt-sign-language-online `
  --deployment-name blue `
  --model-name ngt-sign-language `
  --promoted `
  --traffic-percent 100
```

Use `--model-version <version>` instead of `--promoted` when you need a pinned,
fully reproducible deployment.

For online endpoints, register and use an environment that contains
`azureml-inference-server-http`, for example:

```powershell
poetry run python scripts/register_env.py `
  --env-name sign-language-inference-env-gpu `
  --env-version 1 `
  --conda-file deployment/online/environment.yml `
  --gpu
```

Then set:

```env
AZURE_INFERENCE_ENVIRONMENT_NAME="sign-language-inference-env-gpu"
AZURE_INFERENCE_ENVIRONMENT_VERSION="1"
```

### Run the separate Azure endpoint API

Set these in `.env` after deploying the online endpoint:

```env
AZURE_API_ONLINE_ENDPOINT_URL="<scoring-uri>"
AZURE_API_ONLINE_ENDPOINT_KEY="<endpoint-key>"
AZURE_API_ONLINE_MODEL_VERSION="<served-model-version>"
AZURE_API_DEFAULT_DEPLOYMENT="blue"
```

For GitHub Actions or other CI/CD, store the same values as secrets rather than
plain environment variables:

```text
AZURE_API_ONLINE_ENDPOINT_URL
AZURE_API_ONLINE_ENDPOINT_KEY
AZURE_API_ONLINE_MODEL_VERSION
AZURE_API_DEFAULT_DEPLOYMENT
```

The Azure endpoint API is separate from the main frontend/backend API. Start it
on its own port:

```powershell
poetry run sign-language-azure-api --reload
```

It exposes:

- `GET /health`
- `GET /info`
- `POST /predict`

The main `sign_language.api` backend remains local-model based and is not wired
to Azure ML endpoint serving.

### Test the online endpoint

```powershell
poetry run python scripts/test_online_endpoint.py
```

The test script reads `AZURE_API_ONLINE_ENDPOINT_URL`,
`AZURE_API_ONLINE_ENDPOINT_KEY`, and `AZURE_API_DEFAULT_DEPLOYMENT` from `.env`.
You can still override them with `--endpoint-url`, `--endpoint-key`, and
`--deployment-name`. Add `--deployment-name green` to test a non-live deployment
directly before switching traffic.

### Live-test Azure integration

Run these only after `az login` and after setting the Azure ML SDK variables in
`.env` (`AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_WORKSPACE`, and
the environment/model/data settings needed by the command you run).

Start with a read-only connectivity check:

```powershell
poetry run python scripts/get_azure_mlflow_uri.py
```

Register or verify the Azure ML environment:

```powershell
poetry run python scripts/register_env.py `
  --env-name sign-language-training-env-gpu `
  --env-version <version> `
  --conda-file conda.yaml `
  --gpu
```

Optionally register the local raw ImageFolder data asset:

```powershell
poetry run python scripts/register_raw_data.py `
  --data-dir data/raw `
  --asset-name ngt-raw `
  --version <version>
```

Submit a small Azure ML smoke job before running the full training pipeline:

```powershell
poetry run python scripts/smoke_test.py --instance-type gpu
```

After a model version is registered or promoted, deploy it to an online
endpoint:

```powershell
poetry run python scripts/deploy_online_endpoint.py `
  --endpoint-name ngt-sign-language-online `
  --deployment-name blue `
  --model-name ngt-sign-language `
  --model-version <version> `
  --traffic-percent 100
```

Copy the endpoint scoring URI and key into `.env`, then run:

```powershell
poetry run python scripts/test_online_endpoint.py
```

For a single real image request:

```powershell
poetry run python scripts/invoke_online_endpoint.py --image-path path/to/image.png
```

### Blue/green rollout

```powershell
poetry run python scripts/rollout_blue_green.py `
  --endpoint-name ngt-sign-language-online `
  --model-name ngt-sign-language `
  --model-version <new-version>
```

Rollback switches traffic back to the previous color:

```powershell
poetry run python scripts/rollout_blue_green.py --endpoint-name ngt-sign-language-online --rollback
```

### Canary rollout

Canary is an alternative to blue/green, not an additional step on top of it.
Teams choosing canary should use separate `stable` and `canary` deployment
names. Initialize the canary strategy with a stable deployment:

```powershell
poetry run python scripts/deploy_online_endpoint.py `
  --endpoint-name ngt-sign-language-online `
  --deployment-name stable `
  --model-name ngt-sign-language `
  --model-version <current-production-version> `
  --traffic-percent 100
```

Then deploy a new model through the canary rollout:

```powershell
poetry run python scripts/rollout_canary.py `
  --endpoint-name ngt-sign-language-online `
  --stable-deployment stable `
  --candidate-deployment canary `
  --promoted `
  --steps 10,25,50,100 `
  --wait-seconds 60
```

The script validates the traffic steps, deploys the candidate with `0%` traffic,
then gradually shifts traffic from the stable deployment to the candidate. Use
`--model-version <version>` instead of `--promoted` to deploy a pinned version.
The initial `stable` deployment must already exist and receive live traffic.
The canary script does not call or depend on `rollout_blue_green.py`.
If blue/green and canary must be demonstrated simultaneously, give canary a
separate endpoint name such as `ngt-sign-language-canary` in both commands.

Rollback returns all traffic to the stable deployment:

```powershell
poetry run python scripts/rollout_canary.py `
  --endpoint-name ngt-sign-language-online `
  --stable-deployment stable `
  --candidate-deployment canary `
  --rollback
```

---

## Azure Container Apps CI/CD

The repository mirrors the on-prem CI/CD chain for Azure Container Apps:

```text
push to main -> Code Quality -> Docker Build & Push -> Deploy to Azure Container Apps
```

The on-prem workflow renders Compose and pushes a `deploy` branch for
Portainer. Azure Container Apps does not consume Docker Compose directly, so
`.github/workflows/deploy-azure.yml` keeps the same chained trigger but deploys
the already-built SHA-tagged GHCR images with Azure CLI.

Authentication uses GitHub OIDC through `azure/login@v2`; do not store an Azure
client secret in GitHub. The federated credential must trust the same branch as
the workflow trigger, currently `main`.

The Azure workflow automates the PDF course architecture:

```text
Frontend Container App
  -> endpoint-backed API Container App
      -> Azure ML Kubernetes online endpoint
```

The endpoint-backed API image is `signlang-azure-api`. It forwards `/predict`
requests to the Azure ML Kubernetes online endpoint using endpoint URL/key secrets.
The workflow creates the Container Apps when they do not exist, updates them
on later runs, deploys the Azure ML online endpoint, reads its scoring URI/key,
stores them as Container App secrets, and points the API at those secrets.

Required GitHub Actions variables:

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP
AZURE_CONTAINERAPP_ENVIRONMENT
AZURE_CONTAINERAPP_BACKEND_NAME
AZURE_CONTAINERAPP_FRONTEND_NAME
AZURE_ML_RESOURCE_GROUP
AZURE_WORKSPACE
```

For the 2025-2026 cohort, `AZURE_CONTAINERAPP_ENVIRONMENT` is:

```text
cae-y2d-2026-r2
```

Optional GitHub Actions variables:

```text
AZURE_ONLINE_ENDPOINT_NAME       # default: ngt-sign-language-blue-green
AZURE_ONLINE_MODEL_NAME          # default: ngt-sign-language
AZURE_ONLINE_MODEL_VERSION       # if empty, workflow deploys --promoted
AZURE_ONLINE_INSTANCE_COUNT      # default: 1
AZURE_INFERENCE_ENVIRONMENT_NAME
AZURE_INFERENCE_ENVIRONMENT_VERSION
```

Azure ML endpoint deployments use the Kubernetes instance profile resolved
from `AZURE_INSTANCE_TYPE`, defaulting to `gpu`.

Known endpoint-serving environment values in `team-R2-2026`:

```text
AZURE_INFERENCE_ENVIRONMENT_NAME=sign-language-inference-env-gpu
AZURE_INFERENCE_ENVIRONMENT_VERSION=2
```

CPU alternative:

```text
AZURE_INFERENCE_ENVIRONMENT_NAME=sign-language-inference-env-cpu
AZURE_INFERENCE_ENVIRONMENT_VERSION=3
```

Optional GitHub Actions secret:

```text
GHCR_PAT
```

Set `GHCR_PAT` only if the GHCR images are private. It must be a GitHub token
with `read:packages`; public packages do not need it.

The workflow stores these Container App secrets automatically after the Azure
ML endpoint deploys:

```text
azure-api-online-endpoint-url
azure-api-online-endpoint-key
```

Those map to:

```text
AZURE_API_ONLINE_ENDPOINT_URL
AZURE_API_ONLINE_ENDPOINT_KEY
```

The workflow sets `FRONTEND_BACKEND_ORIGIN` on the frontend Container App so the
frontend proxies API/WebSocket traffic to the endpoint-backed API Container App.

The workflow deploys the model with a blue/green rollout. The first run creates
the `blue` Kubernetes deployment. Later runs deploy the inactive color and then
switch all endpoint traffic to it. The endpoint-backed API does not pin a
deployment name, so requests follow the endpoint traffic allocation.

The workflow can also be started manually from GitHub Actions through
`workflow_dispatch`. Manual runs support redeploying a specific image tag,
skipping Azure ML endpoint deployment, or overriding the model version without
pushing a new commit.

After deployment, the workflow smoke-tests:

```text
backend /health
backend /info
frontend HTTP response
WebSocket /ws/predict landmark response
```

To roll back, activate a previous Container Apps revision from the Azure portal
or with Azure CLI:

```powershell
az containerapp revision list `
  --name <backend-container-app> `
  --resource-group <resource-group> `
  --output table

az containerapp revision activate `
  --name <backend-container-app> `
  --resource-group <resource-group> `
  --revision <previous-revision-name>
```

---

## Environment Variables

Copy `.env.example` to `.env`. All variables are optional unless marked required.

### Database and Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://signlang:signlang@localhost:5432/signlang` | Async PostgreSQL connection string |
| `SECRET_KEY` | _(required)_ | JWT signing secret — min 32 chars. Generate: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `JWT_LIFETIME_SECONDS` | `3600` | JWT token lifetime in seconds |
| `POSTGRES_USER` | `signlang` | PostgreSQL username (Docker Compose) |
| `POSTGRES_PASSWORD` | _(required)_ | PostgreSQL password (Docker Compose) |
| `POSTGRES_DB` | `signlang` | PostgreSQL database name (Docker Compose) |

### Inference API

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_USE_REGISTRY_MODEL` | `false` | Load model from Azure ML registry on startup |
| `AZURE_MODEL_DOWNLOAD_DIR` | `src/sign_language/models/registry_cache` | Local cache for downloaded registry models |

### Standalone Azure Endpoint API

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_API_ONLINE_ENDPOINT_URL` | _(empty)_ | Azure ML online endpoint scoring URI |
| `AZURE_API_ONLINE_ENDPOINT_KEY` | _(empty)_ | Azure ML online endpoint key |
| `AZURE_API_ONLINE_MODEL_VERSION` | _(empty)_ | Model version expected from the online endpoint |
| `AZURE_API_DEFAULT_DEPLOYMENT` | _(empty)_ | Optional deployment name, e.g. `blue` or `green` |
| `AZURE_API_PORT` | `8010` | Port for the separate Azure endpoint API |

### Azure ML SDK

| Variable | Example | Description |
|----------|---------|-------------|
| `AZURE_SUBSCRIPTION_ID` | `0a94de80-...` | Azure subscription ID |
| `AZURE_RESOURCE_GROUP` | `buas-y2` | Azure resource group |
| `AZURE_WORKSPACE` | `team-R2-2026` | Azure ML workspace name |
| `AZURE_ENVIRONMENT_NAME` | `sign-language-training-env-gpu` | Registered conda environment name |
| `AZURE_ENVIRONMENT_VERSION` | `8` | Environment version |
| `AZURE_INFERENCE_ENVIRONMENT_NAME` | `sign-language-inference-env-gpu` | Endpoint serving environment name |
| `AZURE_INFERENCE_ENVIRONMENT_VERSION` | `1` | Endpoint serving environment version |
| `AZURE_INSTANCE_TYPE` | `gpu` | Compute instance type |

These names are shared by the training, registry, trigger, and endpoint
deployment scripts through `sign_language_training.azure_config`.

### MLflow

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_ENABLED` | `false` | Enable MLflow tracking |
| `MLFLOW_TRACKING_URI` | _(empty)_ | Tracking URI (empty = local; set to Azure ML URI for remote) |
| `MLFLOW_EXPERIMENT_NAME` | `sign-language` | Experiment name |
| `MLFLOW_RUN_NAME` | _(empty)_ | Optional run name |
| `MLFLOW_AUTOLOG` | `true` | Enable MLflow autologging |
| `MLFLOW_LOG_ARTIFACTS` | `true` | Upload plots and reports as artifacts |

To get the Azure ML MLflow URI:

```powershell
poetry run python scripts/get_azure_mlflow_uri.py
```

---

## Docker

```bash
docker compose up -d        # start
docker compose logs -f backend  # logs
docker compose down         # stop
docker compose up -d --build    # rebuild after code changes
```

---

## Database

The API uses PostgreSQL for user authentication and future stats/progress storage. Schema is managed by Alembic.

### Start the database

```bash
docker compose up -d db
```

### Run migrations

Always run before starting the API for the first time or after pulling new migrations:

```bash
poetry run alembic upgrade head
```

Other useful Alembic commands:

```bash
poetry run alembic current        # check current schema revision
poetry run alembic history        # list all migrations
poetry run alembic downgrade -1   # roll back one revision
```

### Generate a new migration

After modifying a SQLAlchemy model:

```bash
poetry run alembic revision --autogenerate -m "describe your change"
```

Always inspect the generated file under `src/sign_language/alembic/versions/` before applying — autogenerate is not perfect.

---

## Authentication

The API uses JWT-based authentication via [FastAPI-Users](https://fastapi-users.github.io/fastapi-users/).

### Privilege tiers

| Tier | How | Access |
|------|-----|--------|
| Anonymous | No token | `/api/health`, `/api/info`, `/api/predict`, `/ws/predict` |
| User | Valid JWT | Anonymous + `/api/users/me`, future `/stats`, `/progress` |
| Admin | Valid JWT + `is_superuser=True` | Everything + `/api/admin/*` |

Prediction endpoints (`/api/predict` and `/ws/predict`) accept auth **optionally** — anonymous callers get full functionality, authenticated callers are attributed for future tracking.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/auth/register` | Register a new user |
| `POST` | `/api/auth/jwt/login` | Log in, receive JWT |
| `POST` | `/api/auth/jwt/logout` | Invalidate token |
| `GET` | `/api/users/me` | Current user info |
| `GET` | `/api/admin/whoami` | Admin check (admin only) |

### Register and log in

```bash
# Register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "Str0ngPassword!"}'

# Login — returns access_token
curl -X POST http://localhost:8000/api/auth/jwt/login \
  -d "username=you@example.com&password=Str0ngPassword!"
```

### Use token on predict

```bash
# HTTP
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>"}'

# WebSocket — token passed as query param
ws://localhost:8000/ws/predict?token=<token>
```

### Promote a user to admin

```bash
poetry run python -c "
import asyncio
from sqlalchemy import update
from sign_language.db.engine import AsyncSessionLocal
from sign_language.auth.models import User

async def promote(email: str) -> None:
    async with AsyncSessionLocal() as s:
        await s.execute(update(User).where(User.email == email).values(is_superuser=True))
        await s.commit()
        print(f'Promoted {email} to admin.')

asyncio.run(promote('you@example.com'))
"
```

---

## Code Quality

```powershell
make check       # Black + Flake8 + MyPy + pytest
make format      # Black
make lint        # Flake8
make typecheck   # MyPy
make test        # pytest + coverage
```

Pre-commit hooks (optional but recommended):

```powershell
poetry run pre-commit install
poetry run pre-commit run --all-files
poetry run pre-commit run
```

Use `run --all-files` before opening a pull request. Use plain `run` to check
only files changed for the next commit.

---

## Testing

A minimum coverage of **90%** is required.

```powershell
poetry run pytest
poetry run pytest --cov=src --cov-report=term-missing
```

---

## Documentation

```powershell
make docs-open   # build and open in browser
make docs        # build only
```

On Windows, open manually after building:

```powershell
start docs/_build/html/index.html
```

---

## Project Structure

| Directory / File | Contents |
|------------------|----------|
| `src/sign_language/` | Inference API package (FastAPI + PyTorch + MediaPipe) |
| `src/sign_language/auth/` | Authentication package (FastAPI-Users, JWT, User model) |
| `src/sign_language/db/` | SQLAlchemy async engine and session factory |
| `src/sign_language/alembic/` | Alembic migration scripts |
| `src/sign_language_training/` | Standalone Azure ML training package |
| `tests/` | pytest test suite |
| `scripts/` | Azure ML submission, environment registration, utilities |
| `models/` | Local model weights (Git LFS) |
| `data/` | Raw and processed datasets (DVC) |
| `notebooks/` | Exploratory notebooks |
| `docs/` | Sphinx documentation |
| `docker/` | Dockerfiles (backend, frontend) |
| `docker-compose.yml` | Multi-container orchestration |
| `.env` | Local environment variables (not committed) |

## Notes

- All commands should be run through `make` or `poetry run ...`.
- PyTorch is CPU-only by default (`make install`). Run `make install-gpu` for the CUDA build.
- Large binary files are versioned with Git LFS. Install it before cloning.
- Azure ML job submission requires `az login` and appropriate workspace permissions.
- `AZURE_USE_REGISTRY_MODEL=false` is the correct default for local development.
