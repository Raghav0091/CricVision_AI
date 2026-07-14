# Streamlit to CricVision Pro Migration Plan

## Architecture decision

CricVision has one current application and one future application architecture:

```text
Current: main.py -> Streamlit UI -> Backends/src
Future:  apps/web -> services/api -> services/worker -> packages/cricket_vision
```

Streamlit is not embedded into, proxied through, or presented inside the future
Next.js application. It remains available only while measured workflows are
replaced one at a time.

## 1. Streamlit's current role

`main.py` is the current working entry point. `Backends/src/ui/` provides the
Dashboard, Live Session, Video Analysis, and Session Results. The backend also
contains the currently tested model registry/loading, video IO, calibration,
tracking, reports, overlays, processed-video support, and local session storage.

Streamlit is classified `LEGACY ACTIVE`, not dead code. It remains the reference
implementation and rollback path until the new stack passes equivalent real-video
and live-camera tests.

## 2. Next.js future role

`apps/web/` becomes the product UI for mobile, tablet, and desktop browsers. It
owns camera permission/lifecycle, stump-alignment guidance, delivery-capture UI,
job status, replay presentation, and session navigation. It must not contain CV
inference or invent calibration/analysis success.

The current scaffold is runnable but incomplete. Its calibration request is the
only meaningful API integration today.

## 3. FastAPI future role

`services/api/` is the control plane. It validates transport contracts, registers
sessions and deliveries, accepts calibration evidence, creates analysis jobs, and
returns status/results. Heavy CV inference does not run inside request handlers.

Current stores are process-local and development-only. Persistent storage and a
real worker queue are later migration decisions.

## 4. Worker and shared-package future role

`services/worker/` will run calibration, detection, tracking, reporting, and
replay jobs outside the API process. It is currently an explicit unavailable
scaffold, not a running queue consumer.

`packages/cricket_vision/` holds framework-independent types and pure logic. It
must not import Streamlit or FastAPI. Current contents are deliberately limited
to useful calibration geometry, detection contracts/adapters, and report schemas.

## 5. Recommended migration order

1. **Freeze behavior with tests and fixtures.** Preserve current result keys,
   model-loading behavior, processed videos, calibration evidence, and session
   compatibility.
2. **Migrate pure contracts and geometry.** Move stable dataclasses, coordinate
   types, and deterministic geometry into `packages/cricket_vision` first.
3. **Migrate framework-independent tracking/report functions.** Move one tested
   module at a time; keep temporary compatibility imports in `Backends/` only
   while both paths are exercised.
4. **Build one real worker job.** Calibration solve is the safest first vertical
   slice because failure can be validated without claiming delivery analysis.
5. **Connect API job state.** Replace process-only job metadata with a durable,
   retryable boundary before long-running video analysis.
6. **Connect the Next.js workflow.** Complete calibration, then delivery capture,
   then analysis status/replay. Do not migrate UI before its backend slice is real.
7. **Retire Streamlit features individually.** Remove each legacy page only after
   the equivalent web/API/worker path passes manual field tests and automated
   regression checks.

## 6. First concrete migration slice

The next step should be a real stump-calibration adapter:

```text
apps/web live camera
    -> POST /calibration/solve
    -> worker calibration job/adapter
    -> packages/cricket_vision geometry and response contract
    -> evidence-based success or explicit failure
```

This establishes the complete frontend/API/worker/shared-package boundary without
moving the larger delivery-analysis pipeline prematurely.

## 7. Do not delete yet

- `main.py` and active `Backends/src/ui/` pages.
- `Backends/src/video_pipeline/`, `tracking/`, `models/`, analysis, agents,
  storage, calibration, replay, and physics modules.
- Session-result compatibility and processed-video support.
- Any model weight under `Models/` or root `yolov8n.pt`.
- Useful diagnostic, replay, smoke, performance, and planner scripts.
- Current/pro architecture docs, audits, roadmap, specifications, and PDF.
- `apps/web/`, `services/api/`, `services/worker/`, and the small shared package.

Streamlit can be removed only when the web stack has real camera calibration,
delivery capture, analysis, replay, sessions, and an operational rollback plan.
