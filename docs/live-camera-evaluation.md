# Live Camera Evaluation Plan

This document describes how to collect trustworthy live-camera evidence for the
YOLO26s person-detection MVP after the browser WebGPU migration.

## Important distinctions

- **Live inference** runs in the browser on **WebGPU** at **imgsz 960**.
- Frames are **not uploaded**. Busy frames are dropped (max concurrent inference = 1).
- Static upload remains on the browser **WASM** session at **imgsz 1280**.
- Manual quality markers are **qualitative only**. Do not compute precision/recall from them.

## Prerequisites

1. Host the frontend over **HTTPS** (or use `localhost`).
2. Use desktop Chrome with WebGPU available.
3. Confirm header shows **Browser model ready** and **WebGPU ready**.
4. Switch to **Live Camera**.
5. Start the camera and allow permission (prefer rear / environment camera).

Optional live diagnostics / session recording:

```text
http://localhost:5173/?debug=true
```

Optional ONNX/WebGPU debug panels:

```text
http://localhost:5173/?debug=onnx
```

## Targets (diagnostic, not claims)

| Metric | Target |
|--------|--------|
| Camera preview | ~30 FPS |
| Detection update rate | interactive updates without queue growth |
| End-to-end latency | hardware-dependent |
| Inferences in flight | ≤ 1 |

## Per-scenario procedure (~10–20 seconds each)

For each scenario below:

1. Open with `?debug=true`.
2. Select the matching **Scenario** label in Test Session.
3. Click **Start Test Session**.
4. Point the camera and walk the scenario.
5. Use marker buttons for obvious events:
   - **Good**
   - **Missed Person**
   - **False Positive**
   - **Bad Box**
6. Click **Stop Test Session**.
7. Export **JSON** and **CSV**.
8. Optionally clear before the next scenario.

### Scenarios

1. `single_near` — one person close to camera
2. `single_far` — one person far away
3. `two_people` — two or more people
4. `walking` — person walking across frame
5. `crowded` — denser scene
6. `partial_occlusion` — person partially hidden
7. `person_enters_leaves` — enter/exit frame
8. `low_light` — darker scene
9. `empty_scene` — no people
10. `camera_motion` — moving camera
11. `custom` — anything else worth recording

## What to record

From Live Diagnostics / `window.__liveInferenceStats`:

- completed inferences
- dropped frames
- median inference ms
- median total ms
- effective inference FPS
- max concurrent inference
- session identity / session creations

Confirm Network shows:

- `POST /detect` = 0
- no image/video frame uploads
- model file is not fetched per frame
