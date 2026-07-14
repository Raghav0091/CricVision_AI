# CricVision AI — Current State

## Product

- Current entry point: `main.py`.
- Current runtime: Python, Streamlit, OpenCV, and Ultralytics YOLO.
- Current pages: Dashboard, Live Session, Video Analysis, and Session Results.
- Dev pages remain hidden by `SHOW_DEV_PAGES = False`.
- Next.js, FastAPI, worker, and shared-package folders are future architecture scaffolds, not replacements yet.

## Implemented in the current Streamlit application

- Video Analysis, Live Session, Session Results, and processed-video preview.
- Visual Observer repair and observer timeline.
- Practice Environment and live-session calibration flows.
- Ball tracking, trajectory fitting/replay, delivery/impact/shot reports, shot direction, and outcome prediction.
- Local-first lazy model loading with current registry behavior.
- Clean overlays by default with optional debug overlays.

Uploaded Video Analysis currently owns its orchestration and calls shared
`video_pipeline/`, tracking, analysis, calibration, annotation, model, and
storage modules. There is no separate `Backends/src/engine/` layer.

## Current code organization

- `Backends/src/ui/` — Streamlit pages and current orchestration.
- `Backends/src/video_pipeline/` — video IO, detection helpers, reports, annotations, and timing.
- `Backends/src/tracking/` — active ball tracking and trajectory fitting/scoring.
- `Backends/src/calibration/` plus top-level live/session calibration modules — active measured context flows.
- `Backends/src/analysis/` and `agents/` — reports and observer logic.
- `Backends/src/models/` — registry and lazy model loading.
- `Backends/src/storage/` — local session persistence.
- `apps/web/`, `services/`, and `packages/cricket_vision/` — professional v2 foundations.

## Validation baseline

- `python -m compileall -q Backends services packages scripts`
- `python -m pytest -q`
- `python scripts/smoke_check.py`
- `python scripts/performance_check.py`
- `npm run build` from `apps/web/`

## Product direction

Keep Streamlit stable while the Next.js frontend, FastAPI control plane, worker,
and shared package mature beside it. Migrate one tested workflow at a time and
do not add another parallel orchestration layer.

## Next engineering tasks

1. Manually verify current Streamlit Video Analysis and Live Session modes.
2. Integrate a real stump detector behind the safe calibration API contract.
3. Field-test browser camera/alignment on mobile and tablet.
4. Connect delivery capture only after calibration is reliable.
