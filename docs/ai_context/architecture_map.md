# CricVision AI — Architecture Map

## Current Structure

### `main.py`

Streamlit entrypoint and page router. `SHOW_DEV_PAGES` gates non-production pages.

### `Backends/src/engine/`

- `analyze_delivery.py`
- `engine_options.py`
- `engine_result.py`
- `processors/delivery.py`
- `processors/batting.py`

Owns the reusable `analyze_delivery_clip(video_path, calibration_context,
options)` entrypoint, input validation, mode dispatch, processed-video
finalization, warnings/errors, the stable result contract, and both uploaded
video frame loops. Engine imports do not load models or import Streamlit/UI.

### `Backends/src/ui/`

- `dashboard.py`
- `video_analysis.py`
- `live_session.py`
- `results_page.py`
- `components.py`
- `analysis_helpers.py`
- `theme.py`

Owns presentation, widgets, upload lifecycle, report/session persistence, and
Streamlit interaction. Video Analysis calls the engine and contains no
detection/report frame loop.

### `Backends/src/video_pipeline/`

- `video_reader.py`
- `detection_pipeline.py`
- `report_pipeline.py`
- `annotation_writer.py`
- `performance_timer.py`

Reusable video processing, annotation, performance measurement, and report orchestration.

### `Backends/src/agents/`

- `observer_timeline.py`
- `vision_agent.py`
- `tracking_repair_agent.py`
- `visual_observer_agent.py`

Detection-quality review, deterministic 2D tracking repair, and agent-style analysis.

### `Backends/src/calibration/`

- `calibration_context.py`
- `stump_calibration.py`
- `pitch_calibration.py`

Practice-environment understanding: stumps, pitch corridor, camera view, handedness, and calibration quality.

### `Backends/src/analysis/`

- `frame_detection_utils.py`
- `smart_pipeline.py`
- `impact_detection.py`
- `shot_classification.py`
- `shot_direction.py`
- `outcome_prediction.py`
- `delivery_enrichment.py`
- `cricket_agent.py`

Cricket analysis logic.

### `Backends/src/models/`

- `model_registry.py`
- `model_loader.py`
- `remote_model_loader.py`

Local-first model paths, cached lazy loading, and Hugging Face fallback.

### `Backends/src/storage/`

- `session_store.py`

Session-result persistence and backward-compatible normalization.

### `Backends/src/config/`

- `constants.py`
- `paths.py`

Stable shared constants and project paths.

### `tests/`

Lightweight tests using synthetic or dummy data: no real model files, GPU, camera, `HF_TOKEN`, or internet.

### `scripts/`

- `smoke_check.py`
- `performance_check.py`

Fast contract and performance checks.

## Target Direction

Continue the framework-neutral engine:

```text
Backends/src/engine/
    analyze_delivery.py
    engine_options.py
    engine_result.py
    processors/
        delivery.py
        batting.py
    session_engine.py      # later

services/api/       # later
services/worker/    # later
```

The engine should expose:

```python
analyze_delivery_clip(video_path, calibration_context, options)
```

Streamlit calls this engine. Future FastAPI, worker, mobile, and glasses
clients should call the same engine. Do not move analysis logic back into
Streamlit pages.
