# Person Detection Demo Frontend

React + Vite + TypeScript frontend for YOLO26s aerial person detection.

## Quick start

```bash
npm install
cp .env.example .env   # optional; enables Railway live-camera fallback
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`).

## Environment

| Variable | Purpose |
| --- | --- |
| `VITE_DETECTION_API_URL` | Railway API base URL for **live camera cloud fallback** when WebGPU is unavailable. Trailing slashes are normalized. |

Static upload never uses this URL — it always runs browser ONNX WASM locally.

## Scripts

- `npm run dev` — local development server
- `npm run build` — production build
- `npm run preview` — preview the production build

## Modes

- **Upload Image** — select an aerial image and run single-image detection with the browser WASM ONNX session at imgsz 1280
- **Live Camera** — stream the device camera and run continuous detection:
  - **WebGPU ready** → browser WebGPU at imgsz 960 (frames stay local)
  - **WebGPU unavailable** → Railway `POST /detect` cloud CPU at imgsz 1280 (sampled JPEG frames uploaded; quality 0.75)
  - Max one in-flight inference on either path; busy frames are dropped, never queued

Live camera needs HTTPS outside localhost and browser camera permission. Cloud fallback additionally needs `VITE_DETECTION_API_URL`.

### Developer diagnostics

Open with `?debug=onnx` to enable:

- Browser ONNX Caltech regression test
- Browser runtime benchmark matrix
- WebGPU camera probe

Open with `?debug=true` to enable live-camera diagnostics and test-session recording.

Force Railway fallback (even when WebGPU is ready):

- checkbox **Force cloud CPU fallback** in `?debug=true`, or
- `?forceLiveFallback=1`, or
- `window.__forceLiveRailwayFallback = true` in the console
