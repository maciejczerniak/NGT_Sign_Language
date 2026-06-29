# Backend API — `sign_language`

The inference API package. Serves real-time NGT fingerspelling recognition over HTTP and WebSocket using FastAPI, PyTorch, and MediaPipe.

---

## Contents

- [Overview](#overview)
- [Design rationale](#design-rationale)
- [Package structure](#package-structure)
- [Module table](#module-table)
  - [api/](#api)
  - [auth/](#auth)
  - [core/](#core)
  - [models/](#models)
  - [features/](#features)
  - [utils/](#utils)
  - [db/](#db)
  - [alembic/](#alembic)
  - [Entry points](#entry-points)
- [Authentication tiers](#authentication-tiers)
- [API endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Diagrams](#diagrams)

---

## Overview

`sign_language` is the user-facing backend. It loads a trained EfficientNet-B0 model at startup and exposes two inference paths — a standard HTTP endpoint for single-frame prediction and a WebSocket endpoint for real-time frame-by-frame recognition. Authentication is optional on inference endpoints so anonymous users retain full functionality.

The package supports three deployment targets without any code changes:

| Target | Model source |
|---|---|
| Local | Local `.pth` file |
| On-premise | MLflow registry `@champion` alias |
| Cloud | Azure ML managed online endpoint |

The target is selected at runtime via the `DEPLOY_TARGET` environment variable.

---

## Design rationale

**Separation of concerns.** Each subpackage has a single responsibility — routing, authentication, inference, and data access are fully independent. Changes to the authentication layer do not touch inference logic and vice versa.

**Lifespan hook.** Models are loaded once at startup via FastAPI's `asynccontextmanager` lifespan hook in `app.py`, not per-request. This avoids cold-start latency on inference calls and ensures clean resource disposal at shutdown.

**AppState container.** All shared runtime objects (loaded models, smoother, sequence builder, threading lock) are bundled in a single `AppState` dataclass. This makes concurrent WebSocket access safe and keeps state management explicit rather than scattered across modules.

**Auth isolation.** The three-tier privilege model is implemented in its own subpackage using FastAPI-Users. The anonymous inference path is not disrupted by authentication — prediction endpoints accept tokens optionally.

**Database migrations under version control.** Alembic migration scripts live inside the package under `alembic/versions/`. Schema history is versioned alongside application code rather than managed through manual UI changes.

---

## Package structure

```
src/sign_language/
├── api/
│   ├── app.py
│   ├── routes.py
│   ├── ws.py
│   ├── schemas.py
│   ├── state.py
│   └── __init__.py
├── auth/
│   ├── users.py
│   ├── models.py
│   ├── manager.py
│   ├── ws_auth.py
│   ├── schemas.py
│   └── __init__.py
├── core/
│   ├── inference.py
│   ├── preprocessing.py
│   ├── hand_tracking.py
│   ├── image_transforms.py
│   ├── settings.py
│   ├── logging.py
│   └── __init__.py
├── models/
│   ├── loader.py
│   ├── architectures.py
│   └── __init__.py
├── features/
│   ├── landmarks.py
│   └── __init__.py
├── utils/
│   ├── sequence.py
│   ├── smoothing.py
│   └── __init__.py
├── db/
│   ├── engine.py
│   └── __init__.py
├── alembic/
│   └── versions/
├── main.py
└── cli.py
```

---

## Module table

### api/

| Module | Responsibility | Depends on |
|---|---|---|
| `app.py` | App factory; `asynccontextmanager` lifespan hook loads models at startup and disposes the async DB engine at shutdown | `core`, `auth`, `db` |
| `routes.py` | HTTP route handlers — `/api/predict`, `/api/health`, `/api/info` | `core.inference`, `api.schemas`, `api.state` |
| `ws.py` | WebSocket handler for real-time frame-by-frame recognition at `/ws/predict` | `core.inference`, `api.state`, `auth.ws_auth` |
| `schemas.py` | Pydantic request and response models for input validation | — |
| `state.py` | `AppState` container: loaded models, per-connection smoother, sequence builder, threading lock for concurrent access | `models.loader`, `utils.smoothing`, `utils.sequence` |

### auth/

| Module | Responsibility | Depends on |
|---|---|---|
| `users.py` | FastAPI-Users configuration; three-tier privilege model (anonymous, user, admin) | `auth.models`, `auth.manager`, `db.engine` |
| `models.py` | SQLAlchemy ORM `User` model with async SQLAlchemy | `db.engine` |
| `manager.py` | User manager: registration, verification, password hashing | `auth.models` |
| `ws_auth.py` | WebSocket-specific JWT authentication middleware | `auth.users` |
| `schemas.py` | Pydantic schemas for register, login, and user response | — |

### core/

| Module | Responsibility | Depends on |
|---|---|---|
| `inference.py` | Runs model forward pass; returns top prediction with confidence score | `models.loader`, `core.preprocessing`, `core.hand_tracking` |
| `preprocessing.py` | Frame normalisation and image transform pipeline | `core.image_transforms` |
| `hand_tracking.py` | MediaPipe hand landmark extraction from input frames | — |
| `image_transforms.py` | Reusable torchvision-style image transforms | — |
| `settings.py` | Pydantic settings loaded from `.env` environment variables | — |
| `logging.py` | Structured logging configuration | `core.settings` |

### models/

| Module | Responsibility | Depends on |
|---|---|---|
| `loader.py` | Loads model weights from local `.pth` path or MLflow registry `@champion` alias when `AZURE_USE_REGISTRY_MODEL=true`. Caches downloaded registry models under `models/registry_cache/v<N>/` | `models.architectures`, `core.settings` |
| `architectures.py` | EfficientNet-B0 model class definition | — |

### features/

| Module | Responsibility | Depends on |
|---|---|---|
| `landmarks.py` | MediaPipe landmark feature extraction utilities | — |

### utils/

| Module | Responsibility | Depends on |
|---|---|---|
| `sequence.py` | Builds letter sequences from consecutive per-frame predictions | — |
| `smoothing.py` | Per-connection prediction smoother to reduce frame-to-frame jitter | — |

### db/

| Module | Responsibility | Depends on |
|---|---|---|
| `engine.py` | Async SQLAlchemy engine and session factory (PostgreSQL via `asyncpg`) | `core.settings` |

### alembic/

| Path | Responsibility |
|---|---|
| `alembic/versions/` | Alembic DB migration scripts — full schema history under version control. Run `poetry run alembic upgrade head` before first startup. |

### Entry points

| Module | Responsibility | Depends on |
|---|---|---|
| `main.py` | ASGI entry point; exposes the FastAPI app for Uvicorn | `api.app` |
| `cli.py` | CLI entry point; exposes `predict`, `serve`, and `train` subcommands | `core.settings` |

---

## Authentication tiers

| Tier | How | Access |
|---|---|---|
| Anonymous | No token | `/api/health`, `/api/info`, `/api/predict`, `/ws/predict` |
| Authenticated user | Valid JWT | Anonymous + `/api/users/me`, future `/stats`, `/progress` |
| Admin | Valid JWT + `is_superuser=True` | Everything + `/api/admin/*` |

Prediction endpoints accept authentication **optionally** — anonymous callers receive full inference functionality. Authenticated callers are attributed for future tracking.

---

## API endpoints

| Method | Path | Auth required | Description |
|---|---|---|---|
| `POST` | `/api/auth/register` | No | Register a new user |
| `POST` | `/api/auth/jwt/login` | No | Log in, receive JWT |
| `POST` | `/api/auth/jwt/logout` | Yes | Invalidate token |
| `GET` | `/api/users/me` | Yes | Current user info |
| `GET` | `/api/admin/whoami` | Admin | Admin identity check |
| `POST` | `/api/predict` | Optional | Single-frame HTTP inference |
| `GET` | `/api/health` | No | Health check |
| `GET` | `/api/info` | No | Model and server info |
| `WS` | `/ws/predict` | Optional | Real-time WebSocket inference |

---

## Dependencies

**Key external dependencies:**

| Dependency | Purpose |
|---|---|
| `fastapi` | Web framework and routing |
| `uvicorn` | ASGI server |
| `pydantic` | Request/response validation and settings |
| `fastapi-users` | Authentication, JWT, user management |
| `sqlalchemy` (async) | ORM and database access |
| `alembic` | Database schema migrations |
| `asyncpg` | Async PostgreSQL driver |
| `torch` | Model inference |
| `torchvision` | Image transforms |
| `mediapipe` | Hand landmark detection |
