# Person Detection Demo Frontend

React + Vite + TypeScript frontend for YOLO26s aerial person detection.
Inference runs entirely in the browser with ONNX Runtime Web.

## Quick start

```bash
npm install
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`).

No backend API URL is required. Static upload and live camera both detect in the browser.

## Scripts

- `npm run dev` — local development server
- `npm run build` — production build
- `npm run preview` — preview the production build

## Modes

- **Upload Image** — select an aerial image and run single-image detection with the browser WASM ONNX session at imgsz 1280
- **Live Camera** — stream the device camera and run continuous WebGPU detection at imgsz 960 with max one in-flight inference; busy frames are dropped

Live camera requires WebGPU, HTTPS outside localhost, and browser camera permission.

### Developer diagnostics

Open with `?debug=onnx` to enable:

- Browser ONNX Caltech regression test
- Browser runtime benchmark matrix
- WebGPU camera probe

Open with `?debug=true` to enable live-camera diagnostics and test-session recording.
