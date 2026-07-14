# CricVision Project Cleanup Audit

Audit date: 2026-07-14

Branch: `cleanup-project-structure`

This report was written before deleting or renaming project files. Generated folders (`node_modules`, `.next`, caches, outputs, and temporary files) are excluded from the architecture assessment.

## 1. Current entry points

### Current working application

- `main.py` is the active Streamlit entry point: `streamlit run main.py`.
- It routes to Dashboard, Live Session, Video Analysis, and Results under `Backends/src/ui/`.
- `Backends/` therefore remains the current/legacy-active implementation until the web migration reaches parity.

### Future web application

`apps/web/package.json` exposes:

- `npm run dev` — Next.js development server.
- `npm run build` — production build and TypeScript validation.
- `npm run start` — serve a production build.
- `npm run lint` — Next.js lint checks.

### Future API and worker

- `services/api/main.py` exposes the FastAPI application for `python -m uvicorn services.api.main:app --reload`.
- `services/worker/worker.py` is a future synchronous worker boundary. It is not connected to a queue or the Streamlit runtime.

### Development scripts

- `scripts/debug_ball_tracking.py` — run ball-tracking diagnostics on local clips.
- `scripts/replay_ball_tracking_from_csv.py` — replay recorded diagnostic candidates.
- `scripts/smoke_check.py` — lightweight import/runtime smoke checks.
- `scripts/performance_check.py` — lightweight performance checks.
- `scripts/generate_planner_pdf.py` — regenerate the project planner PDF.

## 2. Import audit

The requested `git grep` checks produced these decisions:

| Query | Finding |
|---|---|
| `Backends.src` | Extensively used by `main.py`, active Streamlit pages, scripts, and tests. `Backends/` cannot be removed. |
| `packages` / `cricket-vision` | The shared package is not imported by runtime code. Its hyphenated name is invalid as a normal Python import package. |
| `services.api` | Referenced by architecture documentation and used directly as the FastAPI launch module. It is independent of Streamlit. |
| `physics_trajectory` | `Backends/src/physics_trajectory.py` is used by Video Analysis, annotation writing, Live Session results, and tests. Keep it. The package adapter is unused placeholder duplication. |
| `session_calibration` | Used by Live Session, Video Analysis, live capture, stump validation, and overlays. Keep it. |
| `live_session` | Imported directly from `main.py` and covered by import tests. Keep it. |
| `engine` | `analyze_delivery_clip`, `EngineOptions`, `EngineResult`, and engine processors are not called by active UI/runtime code. One active scorer function imports only `normalize_ball_tracking_mode` from `engine_options.py`; that small normalization belongs with active tracking utilities. |

### Calibration modules

All current backend calibration families have distinct active callers:

- `Backends/src/calibration/pitch_calibration.py` and `stump_calibration.py` feed `calibration_context` and tests.
- `Backends/src/pitch_calibration.py` feeds the candidate tracker, session calibration, and Video Analysis ROI normalization.
- `Backends/src/session_calibration.py` feeds Live Session, live capture, stump validation, Video Analysis, and annotation overlays.
- `Backends/src/replay_calibration.py` feeds Video Analysis replay reporting.

They may need a later domain consolidation, but deleting or merging them in this structural cleanup would risk current behavior.

### Replay modules

- `Backends/src/trajectory_replay.py` is called by Video Analysis and tested.
- `Backends/src/replay3d/` is called by Video Analysis and tested.
- `Backends/src/virtual_pitch_overlay.py` is called by Live Session.

These are active and remain. The matching modules under the future shared package are unused placeholders.

### Tracking modules

- `Backends/src/ball_candidate_tracker.py` is called by Video Analysis.
- `Backends/src/tracking/` is called by Live Session, Video Analysis, diagnostic scripts, and extensive tests.

These are active and remain. The matching future-package tracker files are unused interface placeholders.

## 3. Folder classification

| Folder | Classification | Decision |
|---|---|---|
| `Backends/` | LEGACY ACTIVE | Keep. Current Streamlit implementation. |
| `apps/web/` | FUTURE SCAFFOLD | Keep. Runnable Next.js frontend; not feature-complete. |
| `services/api/` | FUTURE SCAFFOLD | Keep. Runnable FastAPI control plane with local/process-only storage. |
| `services/worker/` | FUTURE SCAFFOLD | Keep and label clearly. Not connected to a queue or ML pipeline. |
| `packages/cricket-vision/` | DUPLICATE / MERGE LATER | Rename to `packages/cricket_vision/`; retain only useful shared contracts/geometry and remove duplicate placeholders. |
| `Backends/src/engine/` | UNUSED / DELETE | Remove after relocating its one runtime-used normalization helper. Active UI directly owns orchestration. |
| `Models/` | ACTIVE ASSET | Keep all weights unchanged. |
| `scripts/` | ACTIVE DEVELOPMENT | Keep all current scripts. |
| `tests/` | ACTIVE DEVELOPMENT | Keep, except tests that solely validate the dead engine architecture. |
| `docs/` | ACTIVE DOCUMENTATION | Keep current/pro architecture docs and historical audits. |
| `benchmarks/` | ACTIVE DEVELOPMENT | Keep fixtures and notes. |

## 4. Confirmed deletion candidates

- `Backends/src/engine/`: unused orchestration architecture. The UI does not call `analyze_delivery_clip`; its single external helper dependency will move to active tracking utilities first.
- `tests/test_engine.py`: validates only the removed engine API.
- Engine-only tests in `tests/test_accuracy_mode.py`: remove or rewrite around active tracking/UI behavior.
- `packages/cricket_vision/tracking/`: unused placeholder duplication of active `Backends/src/tracking/`.
- `packages/cricket_vision/trajectory/`: unused placeholder duplication of active `Backends/src/physics_trajectory.py` and tracking fit code.
- `packages/cricket_vision/replay/`: unused placeholder duplication of active replay/overlay modules.

No active calibration, replay, tracking, model, script, PDF, or legacy documentation file is a safe deletion candidate in this pass.

## 5. Keep candidates

- `main.py` and all active `Backends/src/ui/` routes.
- Current backend calibration, replay, tracking, video pipeline, analysis, storage, and model-loading modules.
- `apps/web/`, because it builds and provides the selected future frontend direction.
- `services/api/`, because it starts and exposes the selected future API direction.
- `services/worker/`, clearly labelled as an unconnected future boundary.
- Useful shared-package calibration geometry, detection schemas/adapters, and report dataclasses after the package rename.
- All model files, including root `yolov8n.pt`: `Backends/src/detection/yolo_detector.py` uses that filename as its default model path.
- All current scripts and tests for active modules.
- `ARCHITECTURE_V2.md`, workflow/spec/roadmap docs, `docs/ai_context/`, historical audits, refactor notes, and the planner PDF.

## 6. Known maturity boundaries

- Streamlit is the current working product surface.
- Next.js and FastAPI are runnable foundations, not replacements yet.
- The FastAPI calibration endpoint fails honestly when the stump detector is unavailable.
- The worker is not connected to a queue, persistent database, or ML inference pipeline.
- The shared package is future-facing and is not yet imported by the active runtime.

## 7. Cleanup actions completed

- Renamed `packages/cricket-vision/` to import-safe `packages/cricket_vision/`.
- Removed unused `Backends/src/engine/` after relocating the tracking-mode normalizer.
- Removed placeholder shared-package tracking, trajectory, and replay modules that duplicated active backend code.
- Removed `tests/test_engine.py` and engine-only assertions from `tests/test_accuracy_mode.py`.
- Corrected an outdated UI cross-import assertion while retaining its model-load regression check.
- Added root and component README files that distinguish current runtime from future scaffolds.
- Updated current-state, architecture, refactor, and production-audit documentation.
- Kept all active calibration, replay, tracking, models, scripts, legacy docs, and PDFs.

## 8. Validation results

- `python -m compileall -q Backends services packages scripts` — passed.
- `python -m pytest -q` — passed (one expected skip).
- `python scripts/smoke_check.py` — passed.
- `python scripts/performance_check.py` — passed.
- Shared package import check — passed.
- Frontend production build — passed with lint and TypeScript checks.
- `git diff --check` — passed during final scope validation.
