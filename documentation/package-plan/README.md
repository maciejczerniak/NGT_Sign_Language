# System documentation

This folder contains the architectural and package-level documentation for the sign language recognition system. It covers all components of the monorepo — backend packages, frontend application, infrastructure, and deployment.

For setup, installation, and usage instructions, see the [root README](../README.md).

---

## Contents

- [System overview](#system-overview)
- [Repository structure](#repository-structure)
- [Component documents](#component-documents)
- [Deployment summary](#deployment-summary)
- [Diagrams](#diagrams)

---

## System overview

The system is a full-stack, real-time NGT fingerspelling recognition application. It consists of three independently installable Python backend packages, a Vue 3 frontend, and a shared Docker/CI infrastructure layer — all managed as a single monorepo.

The system was designed to be:
- **Modular** — each package has a single responsibility and can be installed, tested, and deployed independently
- **Platform-agnostic** — the same Docker image runs in all three deployment environments; only the Compose overlay and a single environment variable differ
- **Multi-user and secure** — a three-tier authentication layer (anonymous, user, admin) is isolated from the inference path
- **Auditable** — the training pipeline is gated, versioned, and fully reproducible across local and cloud environments

---

## Repository structure

```
.
├── src/
│   ├── sign_language/              ← inference API package
│   ├── sign_language_training/     ← Azure ML training pipeline package
│   ├── sign_language_azure_api/    ← Azure ML endpoint client package
│   └── frontend/                   ← Vue 3 frontend application
├── docker/
│   ├── dockerfiles/                ← one Dockerfile per service
│   └── compose/                    ← base + environment overlay files
├── scripts/                        ← Azure ML submission and deployment utilities
├── models/                         ← local model weights (Git LFS)
├── data/                           ← raw and processed datasets
├── tests/                          ← pytest test suite
├── docs/                           ← Sphinx API documentation
├── documentation/                  ← this folder — architectural documentation
│   ├── README.md                   ← you are here
│   ├── backend-api.md              ← sign_language package
│   ├── training.md                 ← sign_language_training package
│   ├── azure-api.md                ← sign_language_azure_api package
│   ├── frontend.md                 ← Vue frontend
│   └── diagrams/                   ← auto-generated package structure diagrams
├── pyproject.toml                  ← Poetry monorepo configuration
├── Makefile                        ← developer shortcuts
└── README.md                       ← setup and usage instructions
```

---

## Component documents

| Component | Description | Document |
|---|---|---|
| `sign_language` | FastAPI inference API — HTTP and WebSocket recognition endpoints, authentication, model loading | [backend-api.md](./backend-api.md) |
| `sign_language_training` | Azure ML training pipeline — four-stage gated pipeline, orchestration, model registration | [training.md](./training.md) |
| `sign_language_azure_api` | Azure ML endpoint client — proxy service forwarding requests to the cloud-hosted model | [azure-api.md](./azure-api.md) |
| Frontend | Vue 3 application — camera interface, WebSocket recognition, learning and game views | [frontend.md](./frontend.md) |

---

## Deployment summary

The same backend Docker image runs in all three environments. Only the Compose overlay and the `DEPLOY_TARGET` variable differ:

| Target | Model source | Compose overlay | Deployment method |
|---|---|---|---|
| Local | Local `.pth` file | `docker-compose.yml` (base only) | `docker compose up` |
| On-premise | MLflow registry `@champion` alias | `docker-compose.onprem.yml` | Portainer GitOps pull from `deploy` branch |
| Cloud | Azure ML managed online endpoint | `docker-compose.azure.yml` | Azure Container Apps via GitHub Actions OIDC |

The on-premise deployment uses a **GitOps pull model**: the CI pipeline renders the correct Compose overlay, pins image tags with `yq`, and force-pushes to a dedicated `deploy` branch. Portainer on the on-premise host polls that branch and applies changes. This avoids exposing deployment credentials to the CI runner and makes the deploy branch the single source of truth for what is currently running.

The cloud deployment uses **GitHub OIDC** (no stored secrets) to authenticate with Azure and deploy to Azure Container Apps. The workflow chain is:

```
push to main → code quality → Docker build & push → deploy to Azure Container Apps
```

---

## Diagrams

Auto-generated package dependency diagrams (produced with `pyreverse`) are available in the [`diagrams/`](./diagrams/) folder:

| Diagram | Contents |
|---|---|
| [`packages_sign_language.png`](./diagrams/packages_sign_language.png) | Module structure and import dependencies for the inference API package |
| [`packages_sign_language_training.png`](./diagrams/packages_sign_language_training.png) | Module structure and import dependencies for the training pipeline package |
| [`packages_sign_language_azure_api.png`](./diagrams/packages_sign_language_azure_api.png) | Module structure and import dependencies for the Azure endpoint client package |


To regenerate the diagrams after code changes:s
```powershell
poetry run pyreverse -o png -p sign_language src/sign_language
poetry run pyreverse -o png -p sign_language_training src/sign_language_training
poetry run pyreverse -o png -p sign_language_azure_api src/sign_language_azure_api
```
