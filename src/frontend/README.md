# Sign Language Learning Frontend — Developer Documentation

This guide provides the technical instructions required to set up and run the Sign Language Learning Tool in a local or on-premise environment.

---

## Prerequisites

Ensure the following are installed on your system before proceeding:

| Dependency | Version | Download |
|---|---|---|
| Python | 3.11+ |  |
| Node.js & npm | ^20.19.0 or ≥22.12.0 | [Vue Quick Start](https://vuejs.org/guide/quick-start) |
| Docker & Docker Compose | Latest | [Docker](https://docs.docker.com/desktop/setup/install/windows-install/) |
| Git LFS | Latest | [git-lfs.com](https://git-lfs.com/) — required for model weight sync |

---

## Quick Start (Docker)

The recommended way to run the full stack. Ensures environment parity across the team by sharing the host's operating system while running services in isolation.

### 1. Sync Models

```bash
git lfs pull
```

### 2. Build & Run

From the project root:

```bash
docker-compose up --build
```

### 3. Access

| Service | URL |
|---|---|
| Frontend | `http://localhost:3000` |
| Backend API | `http://localhost:8000`
| API Documentation | `http://localhost:8000/docs`|

---

## Manual Setup (Development Mode)

Use this if you need to debug specific components outside of a container.

### Backend Setup (Inference API)

**1. Install Dependencies**

```bash
poetry install
```

**2. Start Server**

```powershell
uvicorn sign_language.main:app --reload
```

**3. Health Check**

Visit [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health). Expected response:

```json
{"status": "ok"}
```

### Frontend Setup (Vue.js)

**1. Install & Run**

```bash
cd src/frontend
npm install
npm run dev
```

The frontend connects to the relative WebSocket route `/ws/predict`.
In development, Vite proxies that route to the backend at `127.0.0.1:8000`.

**2. Access the Tool**

Open `http://localhost:5173/` in your browser.

---

## System Architecture & Data Flow

The application uses a containerised microservices architecture to ensure portability.

1. Camera (Browser):
- Captures video and sends Base64 frames every 100ms via WebSocket.

2. Backend (FastAPI):
- Extracts hand landmarks using MediaPipe.
- Classifies letters using EfficientNet.


3. Frontend (Vue.js):
- Renders a skeleton overlay from the coordinate array (X, Y, Z).
- Updates the prediction history.

---

## Troubleshooting

### Model Loading Failure
If the server crashes on startup with a `FileNotFoundError`, the `.pth` model files are missing. Run `git lfs pull` from the project root.

### Docker Connectivity
The frontend connects to `/ws/predict`. In Docker, Nginx forwards that request
from the frontend container to the backend service:

```bash
http://backend:8000/ws/predict
```

If predictions do not work, check that both containers are running and rebuild
the frontend image after changes to `docker/nginx.frontend.conf`.

### Port Conflicts
If port 5173 or 8000 is already in use, the Docker container will fail to start. Clear any existing instances with:
```bash
docker-compose down
```
