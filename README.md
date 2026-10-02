# Person Detection Demo Frontend

Polished React + Vite + TypeScript frontend for the deployed YOLO26s aerial person-detection MVP.

## Quick start

```bash
npm install
cp .env.example .env
npm run dev
```

Open the local URL printed by Vite (usually `http://localhost:5173`).

## Environment

```env
VITE_DETECTION_API_URL=https://delta-o8cc.onrender.com
```

## Scripts

- `npm run dev` — local development server
- `npm run build` — production build
- `npm run preview` — preview the production build

## Modes

- **Upload Image** — select an aerial image and run a single detection
- **Live Camera** — stream the device camera and run continuous detection with a single in-flight `/detect` request at a time

Live camera requires HTTPS outside localhost and browser camera permission.

### Developer diagnostics

Open with `?debug=true` to enable:

- Live Diagnostics panel (latency breakdown, payload size, in-flight count)
- Capture width / JPEG quality controls
- Capture-width benchmark
- Test session recording + quality markers + JSON/CSV export

See `docs/live-camera-evaluation.md`.

**Note:** Capture width is the JPEG uploaded by the browser. Deployed YOLO `imgsz` remains **1280** unless the backend is extended.

