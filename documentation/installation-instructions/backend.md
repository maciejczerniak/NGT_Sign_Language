# Backend — Installation & Usage

The backend is a FastAPI application providing REST and WebSocket endpoints for real-time NGT sign language recognition. It uses an EfficientNet-B0 primary model and a MediaPipe landmark MLP as a fallback.

## Requirements

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation)
- Git & Git LFS

## Installation

```bash
git clone https://github.com/BredaUniversityADSAI/2025-26d-fai2-adsai-group-researchgroup2.git
cd 2025-26d-fai2-adsai-group-researchgroup2
poetry env use python3.11
poetry install
```

## Environment Configuration

Copy the example environment file and fill in the required values:

```bash
cp env.example .env
```

Key variables:

```text
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/signlang
SECRET_KEY=your-secret-key-at-least-32-characters

# MLflow (optional)
MLFLOW_ENABLED=false
MLFLOW_TRACKING_URI=file:./logs/mlflow

# Azure ML (optional, required for cloud training only)
AZURE_SUBSCRIPTION_ID=
AZURE_RESOURCE_GROUP=
AZURE_WORKSPACE=
```

## Database Setup

Run Alembic migrations before starting the server for the first time:

```bash
poetry run alembic upgrade head
```

## Starting the Server

```bash
poetry run uvicorn sign_language.api:create_app --factory --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.

Interactive API docs are available at `http://localhost:8000/docs`.

## Usage Examples

### Run a live camera inference test

With the server running, open a second terminal and run:

```bash
poetry run python scripts/live_test.py
```

This opens a WebSocket connection to `ws://localhost:8000/ws/predict` and streams predictions from your webcam in real time.

### CLI inference on an image

```bash
poetry run sign-language predict --image hand.jpg
```

With a custom model checkpoint:

```bash
poetry run sign-language predict --image hand.jpg --model models/best_ngt_model_v2.pth
```

### Start the server (development mode)

```bash
poetry run sign-language serve --reload
```

### Start the server (production)

```bash
poetry run sign-language serve --host 0.0.0.0 --port 8080 --workers 4
```

### Health check

```bash
curl http://localhost:8000/health
```

## Docker (Alternative)

To run the full stack (backend + frontend + supporting services) via Docker Compose:

```bash
docker compose up --build
```

> **Note:** If running the frontend separately in development mode, start the backend directly with the command above rather than via Docker.

## Full API Reference

See the [Sphinx documentation](https://cautious-carnival-p365vjk.pages.github.io) for complete endpoint and module reference.
