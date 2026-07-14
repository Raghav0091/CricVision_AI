# CricVision Pro Architecture v2

## Why the prototype remains, but is not the product architecture

Streamlit is valuable for model experiments, uploaded-video diagnostics, and rapid inspection. A professional live product needs browser-native camera ownership, predictable UI state, concurrent API requests, background analysis, reconnectable realtime status, and independently scalable compute. Keeping those concerns in one Streamlit process would couple camera lifecycle, rendering, model inference, and persistence.

The existing Streamlit application remains available and unchanged as the legacy/debug prototype. Existing Video Analysis and Live Session flows stay there until their replacements meet feature and reliability gates.

## Target system

- `apps/web`: Next.js, React, TypeScript, and Tailwind client. Owns camera permission, alignment UI, capture buffering, navigation, and result presentation.
- `services/api`: FastAPI control plane. Validates contracts, registers sessions and deliveries, stores calibration frames, and exposes job status.
- `services/worker`: Python execution boundary for calibration, detection, tracking, trajectory, and replay work.
- `packages/cricket_vision`: reusable CV contracts and geometry that depend on neither Streamlit nor FastAPI.
- local filesystem: development storage for frames, clips, and replays. Object and relational storage are later milestones.

## Data flow

1. The web client opens the rear camera and renders alignment guides.
2. Continue captures one frame and posts it with normalized guide boxes to `/calibration/solve`.
3. The API saves the frame and hands calibration to the worker when a dedicated stump detector is available.
4. A successful solve returns an `EnvironmentContext`; a missing detector or weak solve returns an explicit failure.
5. During capture, the browser will maintain a rolling buffer and save one delivery clip around a measured trigger.
6. The API registers the clip and queues worker analysis.
7. The worker will detect the moving ball, fit a confidence-gated trajectory, and render a replay.
8. WebSocket status will update the web client, which then requests and displays the replay/report.

## Responsibilities and boundaries

The frontend never claims calibration, tracking, or trajectory success on visual alignment alone. The API never performs heavy inference in a request handler. The worker never owns browser state. The `cricket_vision` package owns domain types and reusable CV logic, but not transport or UI.

Current local stores are process-local scaffolds. They make API contracts executable but are not durable or multi-worker safe. Generated calibration frames are written below `outputs/pro_v2/`, which must remain uncommitted.

## Migration plan

1. Stabilize health, session, delivery, and calibration contracts.
2. Validate the browser camera and responsive alignment experience on target phones/tablets.
3. Connect a dedicated stump detector and return measured environment context.
4. Add browser rolling-buffer delivery capture and local clip upload.
5. Connect existing CricVision detection/tracking modules behind worker adapters without moving risky code prematurely.
6. Add job status over WebSocket and replay delivery.
7. Retire individual Streamlit workflows only after parity, manual field testing, and rollback plans.

## Risk areas

- browser camera constraints, orientation changes, thermal throttling, and iOS permission behavior;
- six-stump visibility and detector generalization across grounds, lighting, and occlusion;
- camera movement invalidating calibration after setup;
- network upload latency and dropped connections during delivery handoff;
- false motion triggers, missed pre-roll, and overlapping cooldown windows;
- misleading trajectory output from sparse observations;
- local process stores losing data on restart or diverging across API workers.

There is no official Hawk-Eye claim, DRS/LBW decision, speed, swing, spin, or fabricated 3D result in this foundation.

## Local startup

```powershell
python -m uvicorn services.api.main:app --reload
cd apps/web
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL` only when the API is not at `http://localhost:8000`.
