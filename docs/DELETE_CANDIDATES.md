# Repository Delete Candidates

Audit date: 2026-07-14

Classification is based on `main.py` routing, `git grep` import/caller searches,
API router imports, Next.js imports, tests, and launch/build validation. Generated
content such as `node_modules`, `.next`, caches, `.tmp`, outputs, and videos is
ignored and is not source architecture.

## ACTIVE_CURRENT

Required by `streamlit run main.py` or its active workflows:

| Path | Evidence / reason |
|---|---|
| `main.py` | Current Streamlit entry point and page router. |
| `Backends/src/ui/dashboard.py` | Routed by `main.py`. |
| `Backends/src/ui/live_session.py` | Routed by `main.py`; owns current live workflow. |
| `Backends/src/ui/video_analysis.py` | Routed by `main.py`; owns uploaded-video orchestration. |
| `Backends/src/ui/results_page.py` | Routed by `main.py`; reads current session storage. |
| `Backends/src/video_pipeline/` | Imported by active UI, tracking diagnostics, and tests. |
| `Backends/src/models/` | Active registry and lazy model loading. |
| `Backends/src/tracking/` | Imported by Live Session, Video Analysis, scripts, and tests. |
| `Backends/src/physics_trajectory.py` | Imported by Video Analysis, annotations, Live Session results, and tests. |
| Current calibration modules | Each family has distinct active UI/live/tracking callers. |
| Current replay modules | Used by Video Analysis, Live Session, and replay tests. |
| `Backends/src/analysis/`, `agents/`, `storage/`, `config/` | Active reports, repair, persistence, and configuration. |

## ACTIVE_FUTURE

Required for runnable future architecture:

| Path | Evidence / reason |
|---|---|
| `apps/web/app/` | Next.js routes and live workflow. |
| `apps/web/components/` and `lib/` | Imported by the frontend pages. |
| `apps/web/package.json` and build configs | Required for dev/build commands. |
| `services/api/main.py` | FastAPI application entry point. |
| `services/api/routes/`, `schemas/`, `services/` | Imported by `services/api/main.py` and its routers. |

## KEEP_FOR_NOW

Not part of the current runtime, but retained for the selected migration plan:

| Path | Reason |
|---|---|
| `services/worker/` | Chosen future processing boundary; explicitly documented as unconnected. |
| `packages/cricket_vision/calibration/` | Tested pure geometry/context foundation. |
| `packages/cricket_vision/detection/` | Small model-independent detection contracts/adapter. |
| `packages/cricket_vision/reports/` | Small model-independent result schemas. |
| Dev-only Streamlit pages | Hidden but useful for current diagnostics and data inspection. |
| `Backends/src/detection/yolo_detector.py` | Legacy wrapper; default root-model reference and potential compatibility value. |
| Experimental model registry entries | Metadata only; no startup model load and planned future roles. |
| Historical audits/refactor notes | Explain current constraints and past decisions; clearly marked where superseded. |

## SAFE_DELETE

Caller searches found no runtime, script, service, frontend, or test imports:

| Item | Status | Evidence |
|---|---|---|
| `smart_pipeline.crop_frame_with_roi()` | Deleted this pass | Only definition and historical-doc references existed. |
| `smart_pipeline.restore_roi_boxes_to_full_frame()` | Deleted this pass | Only definition and historical-doc references existed. |
| `Backends/src/engine/` | Already deleted | Active UI did not call it; one helper dependency was relocated. |
| `tests/test_engine.py` | Already deleted | Tested only the removed engine architecture. |
| Shared-package placeholder tracking/replay/trajectory modules | Already deleted | Duplicated active backend modules and had no callers. |

No additional folder currently meets the SAFE_DELETE evidence threshold.

## DO_NOT_DELETE

- `main.py` and active Streamlit pages.
- `Models/`, all `.pt` weights, and root `yolov8n.pt`.
- `apps/web/`, `services/api/`, and their configuration files.
- Active video pipeline, model, tracking, calibration, physics, report, replay,
  processed-video, and session-storage modules.
- `scripts/debug_ball_tracking.py`, diagnostic replay, smoke, performance, and
  planner scripts.
- Tests for active behavior.
- `ARCHITECTURE_V2.md`, `LIVE_WORKFLOW.md`, `ROADMAP.md`, calibration/capture
  specs, AI context, cleanup audit, planner PDF, and useful historical audits.

## Remaining duplicate-looking areas

The backend still has multiple calibration, replay, and tracking modules. Caller
inspection shows they currently serve different workflows, so they are not safe
deletions. They should be consolidated only as part of tested worker/shared-package
migration—not as a filename cleanup.
