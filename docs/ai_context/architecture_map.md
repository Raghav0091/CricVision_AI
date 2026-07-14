# CricVision AI — Architecture Map

## Current working application

### `main.py`

Streamlit entry point and page router. `SHOW_DEV_PAGES` gates development-only pages.

### `Backends/src/ui/`

- `dashboard.py`
- `video_analysis.py`
- `live_session.py`
- `results_page.py`
- `components.py`, `analysis_helpers.py`, and `theme.py`

Owns current Streamlit presentation and interaction. `video_analysis.py` and
`live_session.py` also retain legacy-active orchestration while calling shared
backend modules.

### Shared active backend modules

- `video_pipeline/`: video reading/writing, detection support, reports, annotations, and timing.
- `tracking/`: ball tracking, trajectory scoring, and trajectory fitting.
- `calibration/`, `pitch_calibration.py`, `session_calibration.py`, and `replay_calibration.py`: distinct current calibration contexts.
- `analysis/` and `agents/`: cricket reports and observer/repair logic.
- `models/`: local-first registry and lazy model loading.
- `storage/`: backward-compatible local session persistence.
- `replay3d/`, `trajectory_replay.py`, and `virtual_pitch_overlay.py`: active replay/overlay paths.

`Backends/src/engine/` was removed after caller audit showed that the active UI
did not use it. The single tracking-mode normalizer used outside that folder now
lives in `tracking/ball_tracking_utils.py`.

## Professional v2 foundation

### `apps/web/`

Future Next.js browser frontend. Currently provides the app shell, browser
camera, stump-alignment guides, and safe calibration request flow.

### `services/api/`

Future FastAPI control plane. Provides health, calibration, session, delivery,
and analysis-job contracts using local/process-only storage.

### `services/worker/`

Unconnected future processing boundary. It is not a queue consumer or current
ML pipeline yet.

### `packages/cricket_vision/`

Future framework-independent shared package. It currently contains useful
calibration geometry, detection contracts/adapters, and report schemas only.
Active tracking, trajectory, and replay remain under `Backends/` until a real
migration replaces them.

## Direction

```text
main.py + Backends/        current Streamlit implementation
apps/web/                  future browser frontend
services/api/              future FastAPI control plane
services/worker/           future background processing
packages/cricket_vision/   future shared CV contracts
```

Migrate measured workflows into the worker/shared package only when they are
actually connected. Do not create another unused orchestration layer.
