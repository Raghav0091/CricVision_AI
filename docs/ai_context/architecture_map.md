# CricVision AI — Architecture Map

## Current Structure

### `main.py`

Streamlit entrypoint and page router. `SHOW_DEV_PAGES` gates non-production pages.

### `Backends/src/ui/`

- `dashboard.py`
- `video_analysis.py`
- `live_session.py`
- `results_page.py`
- `components.py`
- `analysis_helpers.py`
- `theme.py`

Owns presentation and Streamlit interaction where possible. It should not become the whole product.

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

Add a framework-neutral engine:

```text
Backends/src/engine/
    analyze_delivery.py
    session_engine.py
    engine_options.py
    engine_result.py

services/api/       # later
services/worker/    # later
```

The engine should expose:

```python
analyze_delivery_clip(video_path, calibration_context, options)
```

Streamlit should call this engine. Future FastAPI, worker, mobile, and glasses clients should call the same engine. Avoid adding more heavy analysis logic directly to large Streamlit pages.
