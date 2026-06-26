# CricVision AI Refactor Notes

Last updated: architecture cleanup pass after the smart video analysis pipeline.

## Active architecture

Production navigation (`main.py`, `SHOW_DEV_PAGES=False`):

- `Backends/src/ui/dashboard.py`
- `Backends/src/ui/live_session.py`
- `Backends/src/ui/video_analysis.py`
- `Backends/src/ui/results_page.py`
- Shared UI: `components.py`, `analysis_helpers.py`, `theme.py`

Analysis stack:

- Smart pipeline: `analysis/smart_pipeline.py`, `analysis/analysis_speed.py`
- Detection timeline helpers: `analysis/frame_detection_utils.py` (new)
- Reports: `impact_detection.py`, `shot_classification.py`, `shot_direction.py`, `outcome_prediction.py`, `delivery_enrichment.py`
- Agents: `agents/observer_timeline.py`, `agents/vision_agent.py`
- Models: `models/model_registry.py`, `model_loader.py`, `remote_model_loader.py`
- Storage: `storage/session_store.py`

Dev-only pages remain gated behind `SHOW_DEV_PAGES`: `field_map.py`, `datasets_page.py`, `training_page.py`.

## Removed from active path

| Item | Action |
|------|--------|
| `estimate_bounce_point()` in `video_analysis.py` | Removed (unused wrapper) |
| Old/External ball models in default model pickers | Removed from UI; files kept on disk |
| Duplicate `_normalize_frame_detections()` copies | Consolidated into `frame_detection_utils.py` |
| Duplicate session/report defaults | Consolidated into `ui/analysis_helpers.py` |
| Eager `import cv2` in `bat_detection.py` | Moved to lazy import inside draw helpers |
| `detection/yolo_detector.py` | Marked LEGACY; not imported |

## Marked legacy / inactive (kept on disk)

| Item | Reason kept |
|------|-------------|
| `Models/ball_detector/best.pt` | Legacy ensemble weight; file untouched |
| `Models/cricket_objects/best_external.pt` | Legacy ensemble weight; file untouched |
| `find_possible_impact_frame()` | Superseded by `detect_bat_ball_impact()` |
| `run_direction_and_agent_review()` | Superseded by `run_post_shot_pipeline()` |
| `smart_pipeline.crop_frame_with_roi()` | Placeholder for future safe ROI work |
| `interactive_field_map.draw_field_map()` | Matplotlib preview; dev/debug only |
| Registry keys `player_type`, `striker_segmentation`, `shot_classifier` | Registered + HF metadata; marked `experimental` / not wired |

## Performance improvements

- No YOLO/Keras loads at app startup; models load on first analysis action.
- `shot_classifier.keras` remains `lazy_only`; no startup TensorFlow import.
- `bat_detection.py` no longer imports OpenCV at module import time.
- Shared frame timeline normalization avoids repeated parsing logic across six modules.
- Smart video pipeline (previous commit) already ensures single-pass detection + shared timeline.
- Default model picker no longer exposes unused legacy weights unless ensemble mode is available.

## Intentionally kept

- Hugging Face remote loading (`remote_model_loader.py`, `HF_TOKEN` from secrets/env).
- CricShot10k registry entries and local path candidates.
- Field setup card (`interactive_field_map.render_field_setup_card`) for batter handedness / fielders context.
- Internal `generate_wagon_wheel_data()` for field-analysis history CSV only — **not rendered as a map in reports**.
- Ensemble mode (Advanced) when multiple ball weights exist.
- Processed video preview (optional), Observer Timeline, and all text-based reports.

## Not removed (risky / future work)

- `video_analysis.py` monolith (~3k lines): Live Session imports detection helpers from it. Splitting into `detection_pipeline.py` is deferred to avoid regressions.
- Duplicate `DETECTION_PRESETS` in Video Analysis and Live Session: low risk duplication; consolidate later.
- Path-based `load_yolo_model()` in `video_analysis.py` alongside registry loader: ensemble mode still uses both caches.
- `player_frame_stride` in smart settings: reserved for future player model wiring.

## Security checks

- No secrets printed in UI or logs from this refactor.
- `.env` / `.streamlit/secrets.toml` remain gitignored.
- Session JSON writes unchanged; still via `session_store.py`.
- No new external API keys added.

## Manual verification checklist

1. `python -m compileall -q Backends`
2. `streamlit run main.py`
3. Dashboard loads without heavy models
4. Video Analysis: upload clip, run Smart Balanced, confirm all reports render, no map outputs
5. Session Results: save + reopen old records
6. Live Session page loads without crash
