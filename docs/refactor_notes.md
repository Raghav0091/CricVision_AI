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

- Video pipeline: `video_pipeline/video_reader.py`, `detection_pipeline.py`,
  `report_pipeline.py`, `annotation_writer.py`, `performance_timer.py`
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
| `find_possible_impact_frame()` | Removed after caller/test audit; superseded by `detect_bat_ball_impact()` |
| `run_direction_and_agent_review()` | Removed after caller/test audit; superseded by `run_post_shot_pipeline()` |
| Duplicate page forwarding wrappers | Replaced with direct imports from `ui/analysis_helpers.py` |
| Duplicate paths and detection presets | Consolidated into import-safe `config/paths.py` and `config/constants.py` |
| Persistent uploaded-video temp file | Replaced with auto-cleaned `TemporaryDirectory` |
| Machine-specific pytest temp/cache failure | Use pytest/OS default temp paths; keep `.pytest_cache/` gitignored |

## Marked legacy / inactive (kept on disk)

| Item | Reason kept |
|------|-------------|
| `Models/ball_detector/best.pt` | Legacy ensemble weight; file untouched |
| `Models/cricket_objects/best_external.pt` | Legacy ensemble weight; file untouched |
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

- The two established frame loops remain in `video_analysis.py` for this safe
  pass. Shared model, ROI, annotation, report, video I/O, and timing helpers now
  resolve through `video_pipeline/`; moving the loops themselves is the next
  isolated refactor.
- Detection presets and stable project paths now come from import-safe config modules.
- The duplicate shared helper definitions were removed from `video_analysis.py`;
  its established frame loops now call `video_pipeline` implementations.
- `player_frame_stride` in smart settings: reserved for future player model wiring.

## Video pipeline extraction

- `detection_pipeline.py` owns lazy model selection, class mapping, ensemble
  loading, ROI detection, local recovery, calibration, and line/length helpers.
- `report_pipeline.py` normalizes one shared `frame_detections` timeline and
  builds observer, impact, shot, direction, outcome, and agent results without
  YOLO inference.
- `annotation_writer.py` owns labels, ROI overlays, review frames, browser MP4
  conversion, and optional impact-marker rewriting.
- `video_reader.py` owns safe opening, metadata, iteration, first-frame
  extraction, and recorded-frame writing.
- `performance_timer.py` owns the stable performance profile schema.
- Live Session no longer imports any helper from `ui/video_analysis.py`.
- Model files, local-first/Hugging Face fallback, Streamlit resource caching,
  and lazy-only `shot_classifier.keras` behavior are unchanged.
- Internal field context/history remains available; production map rendering was
  not reintroduced.

## Synthetic pipeline integration tests

- `tests/helpers/synthetic_video.py` creates a 24-frame, 320x240 cricket-like
  clip under pytest `tmp_path`, with codec-safe skipping.
- `video_reader.py` is covered for opening, metadata, iteration, and invalid
  paths.
- `report_pipeline.py` is covered with ball, bat, stump, impact, and post-impact
  dummy detections, including sparse-data safety.
- `annotation_writer.py` is covered for enabled output, disabled output, and
  missing detections.
- Pipeline import tests guard against YOLO/Keras imports, model-loader calls,
  Hugging Face downloads, and an active Streamlit runtime requirement.
- Smoke and performance scripts now exercise the complete report contract.
- Tests need no model files, GPU, camera, `HF_TOKEN`, or network access.
- Real YOLO accuracy and representative real-video codec checks remain
  manual/local only.

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

## Visual Observer Agent + 2D ball tracking repair

- `agents/tracking_repair_agent.py` extracts the ball path, flags missing or
  suspicious points, repairs only short bounded gaps, and returns a copied
  repaired timeline plus repair statistics.
- `agents/visual_observer_agent.py` turns those statistics into a stable
  confidence level, decision, and readable notes.
- Uploaded-video reports now consume repaired detections where safe and retain
  raw detections for comparison. Repair failure falls back to raw detections.
- Impossible jumps and low-confidence candidates are downgraded rather than
  silently trusted. Repairs are marked `source="observer_repair"` and
  `repaired=True`.
- Session persistence stores a compact repair summary while old JSON records
  continue to normalize with an empty repair block.
- No model loading behavior changed, `shot_classifier.keras` remains lazy-only,
  no map output was reintroduced, and no external service is involved.
- This is explicitly 2D repair. Live Session integration follows Video Analysis
  validation; pseudo-3D pitch calibration is later work; true multi-camera 3D
  tracking is future research.

## Practice Environment Calibration — Part 1

- Added `calibration_context.py`, `stump_calibration.py`, and
  `pitch_calibration.py`; all are deterministic, JSON-safe, and model-free.
- Video Analysis builds a confirmed provisional context from camera view and
  handedness, then refines it from existing stump detections after the frame
  loop.
- The context includes batter-end stumps, approximate crease, pitch corridor,
  batter/bowler pitch ends, stump-line references, frame dimensions, quality,
  and explanatory notes.
- The report contract and session schema now carry normalized
  `calibration_context`; missing old values become a disabled default.
- Calibration UI is text/card-based only. Existing field setup and report
  outputs remain intact, and no map renderer was added.
- No model loading, remote fallback, TensorFlow/Keras, or external API behavior
  changed.
- This establishes a future hook for AR Nets Mode, line/length improvements,
  virtual fielders, and ROI tuning. It is not true 3D or AR hardware support.

## Video Analysis result UI cleanup

- Video Analysis results now default to Processed Video Preview, Quick Result
  Summary, and Save Status; detailed cards live in Summary / Tracking Quality /
  Impact & Shot / Calibration / Technical Details tabs.
- Added `render_analysis_summary_card()` and `build_analysis_summary_data()` in
  `components.py` for a compact coach-oriented summary.
- Calibration quality is capped in `finalize_calibration_quality()` so
  estimated/default stumps cannot read as High.
- Processed video preview uses `validate_processed_video_path()` before
  `st.video()` and falls back to raw output when browser MP4 conversion fails.
- Clean overlay is the default; Debug overlay is optional via the Overlay detail
  control. No map outputs were reintroduced and model loading behavior is unchanged.
