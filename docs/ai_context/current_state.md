# CricVision AI — Current State

## Product

- **Stage:** Strong Streamlit MVP moving toward a continuous cricket training platform.
- **Entry point:** `main.py`
- **Runtime:** Python 3.11, Streamlit, OpenCV, and Ultralytics YOLO.
- **Production pages:** Dashboard, Video Analysis, Live Session, and Session Results.
- **Development pages:** Hidden by `SHOW_DEV_PAGES = False`.

## Implemented

- Video Analysis, Live Session, Session Results, and processed-video preview.
- Reusable `analyze_delivery_clip()` engine entrypoint with engine-owned
  delivery/batting processors, normalized options, and a stable result.
- Visual Observer Agent with deterministic 2D ball-tracking repair.
- Practice Environment Calibration and Calibration Context card.
- Observer Timeline and Delivery, Impact, and Shot reports.
- Shot Direction / Field Zone and Outcome Prediction.
- CricVision Agent Review.
- Clean overlays by default with optional debug overlays.

## Architecture

Core code is organized under:

- `Backends/src/video_pipeline/`
- `Backends/src/engine/`
- `Backends/src/agents/`
- `Backends/src/calibration/`
- `Backends/src/analysis/`
- `Backends/src/models/`
- `Backends/src/storage/`
- `Backends/src/ui/`
- `Backends/src/config/`

## Validation Baseline

- `python -m compileall -q Backends` passes.
- `python -m pytest -q` passes.
- `python scripts/smoke_check.py` passes.
- `python scripts/performance_check.py` passes.

## Product Direction

Pro Nets Capture → Continuous Batter Session Engine → Fast Analysis Mode → Coaching Agent → Club Dashboard → Bowling Action Analysis. A phone or AR-glasses app is a future client of the same engine.

## Next Engineering Tasks

1. Add a real-video benchmark pack for the extracted engine.
2. Manually verify all Video Analysis modes and overlays.
3. Build a Continuous Batter Session MVP with a manual delivery marker.
4. Add Fast Analysis Mode.
5. Add Coaching Agent v1.
