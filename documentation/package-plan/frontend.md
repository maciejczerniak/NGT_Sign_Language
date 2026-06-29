# Frontend — Vue 3 application

The browser-based user interface for NGT fingerspelling recognition. Built with Vue 3, TypeScript, Tailwind CSS, and Vite.

---

## Contents

- [Overview](#overview)
- [Design rationale](#design-rationale)
- [Application structure](#application-structure)
- [Module table](#module-table)
  - [Entry points](#entry-points)
  - [Router](#router)
  - [Views](#views)
  - [Composables](#composables)
  - [Constants](#constants)
  - [Static assets](#static-assets)
  - [Configuration files](#configuration-files)
- [Views and routing](#views-and-routing)
- [WebSocket communication](#websocket-communication)
- [Dependencies](#dependencies)

---

## Overview

The frontend is a single-page application (SPA) that provides two main user experiences — a **learning mode** where users can study NGT fingerspelling signs letter by letter, and a **game mode** where users practice recognition in real time using their camera. Real-time recognition is handled via a WebSocket connection to the backend inference API.

The frontend communicates with the backend over two channels:
- **HTTP** — `/api/predict` for single-frame prediction
- **WebSocket** — `/ws/predict` for real-time frame-by-frame recognition during gameplay

---

## Design rationale

**Vue 3 with Composition API.** The Composition API and composables pattern allow reusable stateful logic (camera access, WebSocket connection, frame capture) to be extracted into a single `useCamera` composable and shared across any view that needs it — without duplicating logic or creating tightly coupled components.

**Single composable for camera and WebSocket.** Camera stream access and the WebSocket connection to the backend are bundled in `useCamera.ts`. This keeps all real-time communication logic in one place, making it easy to test, replace, or extend independently of the views that consume it.

**Four focused views.** Each view is a self-contained component responsible for a single screen. Views do not share state directly — they communicate through the router and composables. This makes each screen independently maintainable.

**TypeScript throughout.** All application logic is written in TypeScript, catching type errors at build time and making the codebase easier to navigate and extend.

**Tailwind CSS for styling.** Utility-first CSS keeps styles co-located with markup and avoids the overhead of a separate stylesheet architecture for a project of this size.

**Vite dev proxy.** The Vite development server is configured to proxy API requests to the backend, so the frontend can be developed locally without CORS issues and without changing any backend configuration.

**Static sign assets.** Sign reference images (illustration + photo per NGT letter) are served as static files from `public/signs/`. Keeping them as static assets rather than fetching them from the backend avoids unnecessary API calls for content that never changes.

---

## Application structure

```
src/frontend/
├── src/
│   ├── composables/
│   │   └── useCamera.ts
│   ├── constants/
│   │   └── letters.ts
│   ├── router/
│   │   └── index.ts
│   ├── views/
│   │   ├── HomeView.vue
│   │   ├── LearnView.vue
│   │   ├── PlayView.vue
│   │   └── LevelEasyGame.vue
│   ├── App.vue
│   ├── main.ts
│   └── style.css
├── public/
│   ├── signs/
│   │   ├── a_illustration.png
│   │   ├── a_photo.jpg
│   │   └── ... (one illustration + one photo per NGT letter)
│   ├── favicon.ico
│   └── Otto_1.png
├── index.html
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
├── tsconfig.json
├── tsconfig.app.json
├── tsconfig.node.json
├── package.json
├── eslint.config.ts
└── .prettierrc.json
```

---

## Module table

### Entry points

| Module | Responsibility | Depends on |
|---|---|---|
| `main.ts` | Application entry point — creates the Vue app instance, registers the router plugin, and mounts the app to the DOM | `App.vue`, `router/index.ts` |
| `App.vue` | Root component — provides the `<RouterView>` outlet that renders the active view | `router/index.ts` |
| `style.css` | Global base styles and Tailwind CSS directives | — |

### Router

| Module | Responsibility | Depends on |
|---|---|---|
| `router/index.ts` | Vue Router configuration — defines four named routes and maps them to their view components | `views/` |

### Views

| Module | Responsibility | Depends on |
|---|---|---|
| `views/HomeView.vue` | Landing screen — entry point for the application, navigation to Learn and Play modes | — |
| `views/LearnView.vue` | Learning mode — displays the sign illustration and reference photo for each NGT letter, allowing users to study signs before practising | `constants/letters.ts` |
| `views/PlayView.vue` | Play mode entry screen — presents available difficulty levels and navigates to the selected game view | — |
| `views/LevelEasyGame.vue` | Easy game view — activates the camera, opens a WebSocket connection to the backend, displays real-time recognition results, and provides per-letter feedback to the user | `composables/useCamera.ts` |

### Composables

| Module | Responsibility | Depends on |
|---|---|---|
| `composables/useCamera.ts` | Reusable composable encapsulating: camera stream initialisation and teardown, per-frame capture and base64 encoding, WebSocket connection lifecycle to `/ws/predict`, and incoming prediction handling. Consumed by any view that requires live recognition. | `constants/letters.ts` |

### Constants

| Module | Responsibility | Depends on |
|---|---|---|
| `constants/letters.ts` | NGT fingerspelling letter definitions — letter list, display names, and metadata used by the learning view and the camera composable | — |

### Static assets

| Path | Contents |
|---|---|
| `public/signs/<letter>_illustration.png` | Hand shape illustration for each NGT fingerspelling letter (A–Y, excluding letters not in the NGT alphabet) |
| `public/signs/<letter>_photo.jpg` | Reference photograph for each NGT fingerspelling letter |
| `public/Otto_1.png` | Application mascot image |
| `public/favicon.ico` | Browser tab icon |

### Configuration files

| File | Purpose |
|---|---|
| `vite.config.ts` | Vite build configuration; configures the dev server proxy to forward `/api` and `/ws` requests to the backend, eliminating CORS issues during local development |
| `tailwind.config.js` | Tailwind CSS configuration — defines content paths for purging unused styles in production builds |
| `postcss.config.js` | PostCSS configuration for Tailwind CSS processing |
| `tsconfig.json` / `tsconfig.app.json` / `tsconfig.node.json` | TypeScript compiler configuration — split into app and Node contexts |
| `eslint.config.ts` | ESLint configuration for Vue 3 and TypeScript |
| `.prettierrc.json` | Prettier formatting configuration |
| `package.json` | npm dependency manifest and build/dev scripts |

---

## Views and routing

| Route | View | Description |
|---|---|---|
| `/` | `HomeView.vue` | Landing screen |
| `/learn` | `LearnView.vue` | Letter-by-letter sign learning interface |
| `/play` | `PlayView.vue` | Game mode selection screen |
| `/play/easy` | `LevelEasyGame.vue` | Real-time recognition game — easy level |

---

## WebSocket communication

Real-time recognition during gameplay uses a persistent WebSocket connection managed by `useCamera.ts`:

1. The view mounts and calls `useCamera` — the composable initialises the camera stream and opens a WebSocket connection to `/ws/predict`
2. On each animation frame, the composable captures a frame from the camera, encodes it as base64, and sends it over the WebSocket
3. The backend processes the frame, runs inference, and returns a prediction with a confidence score
4. The composable exposes the latest prediction as a reactive ref — the view renders it in real time without managing any WebSocket logic itself
5. On view unmount, the composable tears down the camera stream and closes the WebSocket connection cleanly

---

## Dependencies

**Key external dependencies:**

| Dependency | Purpose |
|---|---|
| `vue` | UI framework (Composition API) |
| `vue-router` | Client-side routing |
| `typescript` | Type safety across all application logic |
| `vite` | Build tool and development server |
| `tailwindcss` | Utility-first CSS framework |
| `eslint` + `prettier` | Code quality and formatting |
