# Frontend — Installation & Usage

The frontend is a Vue 3 + Vite application that provides the user interface for real-time NGT sign language recognition. It communicates with the backend via a WebSocket proxy.

> **Important:** The backend must be running on `http://localhost:8000` before starting the frontend. See [backend.md](./backend.md) for setup instructions.

## Requirements

- Node.js `^20.19.0` or `>=22.12.0`
- npm (bundled with Node.js)

## Installation

From the repository root:

```bash
cd src/frontend
npm install
```

## Starting the Development Server

```bash
npm run dev
```

This starts the Vite dev server and automatically opens the app in your browser. The WebSocket connection to `ws://localhost:8000/ws/predict` is proxied automatically — no additional configuration is needed.

## Usage

Once both servers are running:

1. Open the app in your browser (Vite will display the local URL on startup, typically `http://localhost:5173`).
2. Allow camera access when prompted.
3. Hold a hand sign in front of your camera — predictions will appear in real time.

## Other Scripts

| Command | Description |
|---|---|
| `npm run build` | Compile and type-check for production |
| `npm run preview` | Preview the production build locally |
| `npm run lint` | Run oxlint and ESLint with auto-fix |
| `npm run format` | Format source files with Prettier |

## No Environment Variables Required

The WebSocket proxy target (`ws://127.0.0.1:8000`) is configured directly in `vite.config.ts`. No `.env` file is needed for local development.
