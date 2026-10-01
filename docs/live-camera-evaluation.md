# Live Camera Evaluation Plan

This document describes how to collect trustworthy live-camera evidence for the
YOLO26s person-detection MVP.

## Important distinctions

- **Capture resolution** — JPEG width uploaded by the browser (`640 / 768 / 960 / 1280`).
- **Model inference resolution (`imgsz`)** — currently fixed at **1280** on the deployed Lightning backend.
- Changing capture width does **not** change YOLO `imgsz`.
- Manual quality markers are **qualitative only**. Do not compute precision/recall from them.

## Prerequisites

1. Host the frontend over **HTTPS** (or use `localhost`).
2. Open with developer tools enabled:

   ```text
   http://localhost:5173/?debug=true
   ```

3. Confirm header shows **Model Online**.
4. Switch to **Live Camera**.
5. Select capture width (default `960`) and JPEG quality (default `0.75`).
6. Start the camera and allow permission (prefer rear / environment camera).

## Targets (diagnostic, not claims)

| Metric | Target |
|--------|--------|
| Camera preview | ~30 FPS |
| Detection update rate | 3–5 FPS |
| End-to-end latency | < 500 ms |
| Requests in flight | ≤ 1 |

## Per-scenario procedure (~10–20 seconds each)

For each scenario below:

1. Select the matching **Scenario** label in Test Session.
2. Click **Start Test Session**.
3. Point the camera and walk the scenario.
4. Use marker buttons for obvious events:
   - **Good**
   - **Missed Person**
   - **False Positive**
   - **Bad Box**
5. Click **Stop Test Session**.
6. Export **JSON** and **CSV**.
7. Optionally clear before the next scenario.

### Scenarios

1. `single_near` — one person close to camera
2. `single_far` — one person far away
3. `two_people` — two or more people
4. `walking` — person walking across frame
5. `crowded` — denser scene
6. `partial_occlusion` — person partially hidden
7. `person_enters_leaves` — enter/exit frame
8. `low_light` — dim lighting
9. `empty_scene` — no people
10. `camera_motion` — moving / shaking camera

## Capture-width benchmark

1. Start camera.
2. Open `?debug=true`.
3. Click **Run Live Benchmark**.
4. Wait for sequential runs at capture widths `640`, `768`, `960`, `1280` (5 samples each).
5. Export benchmark JSON.
6. Remember: **model imgsz remains 1280** unless backend support is added later.

## Overlay sanity check

Before concluding the model is wrong:

1. Resize the browser window — boxes should stay aligned.
2. Rotate phone to portrait — boxes should stay aligned.
3. Confirm Live Diagnostics `Captured frame` matches API `image.width × image.height`.
4. Confirm `Requests in flight` never exceeds `1`.

## What not to do

- Do not retrain or swap checkpoints during this evaluation.
- Do not enable tracking/interpolation yet.
- Do not hammer the API with concurrent requests.
- Do not treat manual markers as ground truth labels.
