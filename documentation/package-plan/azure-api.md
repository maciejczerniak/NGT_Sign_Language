# Azure endpoint client — `sign_language_azure_api`

The Azure ML online endpoint client package. Wraps the Azure ML managed online endpoint as a callable proxy service, allowing inference requests to be forwarded to the cloud-hosted model.

---

## Contents

- [Overview](#overview)
- [Design rationale](#design-rationale)
- [Package structure](#package-structure)
- [Module table](#module-table)
- [API endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Relationship to the main inference API](#relationship-to-the-main-inference-api)
- [Dependencies](#dependencies)

---

## Overview

`sign_language_azure_api` is a lightweight FastAPI proxy service that forwards prediction requests to an Azure ML managed online endpoint. It runs as a separate service on port 8010, independently of the main `sign_language` inference API.

It is used in the **cloud deployment target**, where the frontend routes through the Azure Container Apps stack:

```
Frontend Container App
  → Azure endpoint API Container App (this package)
      → Azure ML managed online endpoint
          → registered ngt-sign-language model
```

---

## Design rationale

**Isolated as a separate package.** The Azure ML SDK and endpoint authentication logic are kept entirely out of the main `sign_language` inference API. In local and on-premise deployments, this package is simply not installed — the main API loads models directly without any Azure dependency. This keeps the default deployment lightweight and avoids SDK version conflicts.

**Separate service, separate port.** Running as a distinct service on port 8010 means it can be deployed, scaled, and smoke-tested independently of the main backend. The frontend can be pointed at either service depending on the deployment target without any code changes.

**Environment-driven configuration.** All endpoint connection details (URL, key, model version, deployment name) are loaded from environment variables, making the package fully portable across different Azure ML workspaces and endpoint names.

---

## Package structure

```
src/sign_language_azure_api/
├── app.py
├── client.py
├── schemas.py
├── settings.py
├── main.py
└── __init__.py
```

---

## Module table

| Module | Responsibility | Depends on |
|---|---|---|
| `client.py` | Azure ML endpoint client — authenticates using the endpoint key and forwards inference requests to the managed online endpoint scoring URI | `settings.py`, `schemas.py` |
| `app.py` | FastAPI app wrapping the client — exposes `/health`, `/info`, and `/predict` routes | `client.py`, `schemas.py` |
| `schemas.py` | Pydantic request and response models for the proxy endpoints | — |
| `settings.py` | Loads endpoint URL, API key, model version, and deployment name from environment variables | — |
| `main.py` | Entry point — starts the Uvicorn server on `AZURE_API_PORT` (default 8010) | `app.py` |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — confirms the proxy service is running |
| `GET` | `/info` | Returns model version and deployment name from environment config |
| `POST` | `/predict` | Forwards the request to the Azure ML online endpoint and returns the prediction response |

---

## Configuration

All configuration is loaded from `.env` environment variables:

| Variable | Description |
|---|---|
| `AZURE_API_ONLINE_ENDPOINT_URL` | Azure ML online endpoint scoring URI |
| `AZURE_API_ONLINE_ENDPOINT_KEY` | Azure ML online endpoint authentication key |
| `AZURE_API_ONLINE_MODEL_VERSION` | Expected model version served by the endpoint |
| `AZURE_API_DEFAULT_DEPLOYMENT` | Deployment name to target (e.g. `blue`, `green`, `stable`) |
| `AZURE_API_PORT` | Port for the proxy service (default: `8010`) |

In the Azure Container Apps deployment, the endpoint URL and key are stored as Container App secrets and injected automatically by the CI/CD workflow — they do not need to be set manually.

---

## Relationship to the main inference API

The `sign_language_azure_api` package and the main `sign_language` inference API are **independent services** and do not call each other. They serve different deployment scenarios:

| Service | Deployment target | Model source |
|---|---|---|
| `sign_language` | Local, on-premise | Local `.pth` file or MLflow registry |
| `sign_language_azure_api` | Cloud (Azure Container Apps) | Azure ML managed online endpoint |

The frontend is pointed at whichever service is active for the current deployment target. Both services expose a `/predict` endpoint with the same request/response contract, making the switch transparent to the frontend.

---

## Dependencies

**Key external dependencies:**

| Dependency | Purpose |
|---|---|
| `fastapi` | Web framework and routing |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation and settings |
| `httpx` | Async HTTP client for forwarding requests to the Azure ML endpoint |
