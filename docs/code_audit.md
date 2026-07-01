# CricVision AI Code Audit

Last updated: testing suite and architecture audit pass.

## 1. Current active architecture

### Active pages (production)

| Sidebar | Module | Purpose |
|---------|--------|---------|
| Dashboard | `Backends/src/ui/dashboard.py` | Quick status, navigation |
| Live Session | `Backends/src/ui/live_session.py` | Recorded/live delivery analysis |
| Video Analysis | `Backends/src/ui/video_analysis.py` | Upload clip, smart pipeline, reports |
| Session Results | `Backends/src/ui/results_page.py` | Saved deliveries + analytics dashboard |

Dev-only (gated by `SHOW_DEV_PAGES=False` in `main.py`): Field Setup Lab, Datasets, Training Lab.

### `main.py` import graph

**Always loaded:** `Backends/src/ui/theme.py`

**Lazy per page:**

- Dashboard → `dashboard.py` → `model_registry.validate_model_paths`, `components`, `theme`
- Live Session → `live_session.py` → analysis, tracking, models, `interactive_field_map` (field setup card)
- Video Analysis → `video_analysis.py` → full analysis stack + `smart_pipeline` at analyze time
- Results → `results_page.py` → `session_store`, `components`

### Video Analysis module stack

| Layer | Modules |
|-------|---------|
| UI / orchestration | `video_analysis.py`, `analysis_helpers.py`, `components.py`, `interactive_field_map.py` (setup only) |
| Shared video pipeline | `video_pipeline/video_reader.py`, `detection_pipeline.py`, `report_pipeline.py`, `annotation_writer.py`, `performance_timer.py` |
| Smart pipeline | `smart_pipeline.py`, `analysis_speed.py` |
| Detection timeline | `frame_detection_utils.py`, `bat_detection.py`, ball/stump ROI helpers in `video_pipeline/detection_pipeline.py` |
| Tracking | `ball_tracking_utils.py` |
| Reports | `impact_detection.py`, `shot_classification.py`, `shot_direction.py`, `outcome_prediction.py`, `delivery_enrichment.py` |
| Agents | `observer_timeline.py`, `vision_agent.py` |
| Delivery text | `cricket_agent.py` |
| Models | `model_loader.py`, `model_registry.py`, `remote_model_loader.py` |
| Persistence | `session_store.py` |

Flow: **video → single read loop → shared `frame_detections` → observer timeline → impact/shot/direction/outcome/agent → session save → UI reports.**

### Live Session module stack

Shares: `cricket_agent`, `field_zones` (internal context only),
`ball_tracking_utils`, `model_loader`, `interactive_field_map`, and the reusable
`video_pipeline` modules. It no longer imports `ui/video_analysis.py`.

### Session Results module stack

`results_page.py` → `session_store.py` (`load_session_results`, `get_session_summary`, `normalize_session_result`) → `components.py` report renderers.

---

## 2. Efficiency audit

| Check | Status | Notes |
|-------|--------|-------|
| Dashboard loads heavy models? | **No** | Only `validate_model_paths()` (filesystem metadata) |
| Video processed once? | **Yes** | Main loop builds shared `frame_detections`; reports reuse it |
| Reports share `frame_detections`? | **Yes** | `run_post_shot_pipeline` + observer timeline use same timeline |
| Cached model loader? | **Yes** | `get_cached_yolo_model` + `@st.cache_resource` path loader in video analysis |
| `shot_classifier.keras` lazy-only? | **Yes** | Registry `lazy_only`; `get_cached_keras_model` never called in production path |
| Repeated video reads? | **Partial** | `refine_bat_detections_near_impact` re-opens video for impact±N frames only; optional impact video rewrite |
| Repeated YOLO passes? | **Reduced** | Stump lock, bat stride, single-pass same-model fusion; ball every frame in smart modes |
| Optional processed video? | **Yes** | UI checkbox + `generate_processed_video` flag |
| Stump reuse/lock? | **Yes** | `lock_static_stump_detection` + `apply_locked_stump` in smart pipeline |

**Repeated work still present (documented, not changed in this pass):**

- The mature frame loops still live in `video_analysis.py`; shared detection,
  report, video I/O, annotation, and timing behavior has moved to
  `video_pipeline/`. Loop extraction remains a focused follow-up.
- Ensemble mode may load multiple YOLO weights
- Path-based and registry-based caches can duplicate the same weights

---

## 3. Code quality audit

### Large files (split candidates)

| File | Lines (approx) | Recommendation |
|------|----------------|----------------|
| `ui/video_analysis.py` | ~1900 | Split frame-loop orchestration only after real-video regression coverage |
| `ui/live_session.py` | ~1360 | Keep stable webcam path; split only after regression coverage |
| `ui/interactive_field_map.py` | ~800 | Keep; field setup card is shared |

### Duplicate helpers (partially addressed)

- ✅ `normalize_frame_detections` consolidated → `frame_detection_utils.py`
- ✅ Session save + report defaults → `analysis_helpers.py`
- ✅ Detection presets, shared thresholds, and stable paths consolidated under `config/`
- ⚠️ Detection collection helpers differ (`collect_model_detections` vs `collect_detections`)
- ⚠️ Two YOLO loader entry points (registry vs path string)

### Dead / legacy active paths

- `detection/yolo_detector.py` — LEGACY, not imported
- Superseded `find_possible_impact_frame` and `run_direction_and_agent_review` wrappers removed after caller/test audit
- Registry models `player_type`, `striker_segmentation`, `shot_classifier` — registered, not wired
- `draw_field_map()` — dev/debug only; not in production reports

### Readability

- Smart pipeline settings are clear and mode-driven — good
- Rule-based shot/direction/outcome modules are testable — good
- Avoid over-shortening `process_video` loop until extracted to dedicated module

---

## 4. Security audit

| Check | Status |
|-------|--------|
| HF_TOKEN printed in UI/logs | **Not found** in active UI paths |
| `.env` / secrets exposed | **Not committed**; loaded via `remote_model_loader` only |
| Uploaded video paths | Auto-cleaned temporary directory; outputs under project-root `outputs/` |
| Output filenames | Timestamp-based names; no raw user path concatenation in session JSON |
| Debug logs expose secrets | Performance/observer notes are detection stats only |
| Model/output files committed | Outputs/remote models ignored; four legacy/current `.pt` weights remain intentionally tracked |

**Continue to avoid:** printing `get_hf_token()`, committing `.streamlit/secrets.toml`, logging full env in Streamlit.

---

## 5. Performance improvement ideas

### Safe (recommended next)

- Decouple Live Session from `video_analysis.py` imports via shared `detection_pipeline.py`
- Add model path/failure unit tests without loading weights
- Extend `best_detection_center` / timeline helpers (xyxy support added in test pass)
- Keep optional processed video off for batch analysis

### Risky (defer)

- Full ROI inference without regression tests on real clips
- Removing ensemble / legacy path loaders without migration
- Batching YOLO across frames without validated Ultralytics behavior

### Future

- Wire `player_type` / segmentation models with explicit opt-in
- Lazy `shot_classifier.keras` behind explicit user action only
- Optional real-video benchmark script with local path only

### Do not change yet

- Ball-every-frame policy in Smart Balanced / Smart Accurate
- Hugging Face remote fallback logic
- Observer timeline + vision agent thresholds without labeled validation data

---

## 6. Testing gaps (before → after)

| Area | Before | After |
|------|--------|-------|
| `frame_detection_utils` | None | `tests/test_frame_detection_utils.py` |
| `analysis_speed` box parsing | None | Covered in frame detection tests |
| Observer timeline | None | `tests/test_observer_timeline.py` |
| Impact detection | None | `tests/test_impact_detection.py` |
| Shot direction | None | `tests/test_shot_direction.py` |
| Outcome prediction | None | `tests/test_outcome_prediction.py` |
| Session store | None | `tests/test_session_store.py` |
| Module imports | None | `tests/test_imports.py` |
| Video reader | None | Tiny generated clip in `tests/test_video_pipeline_reader.py` |
| Report pipeline | Unit-only timeline | Synthetic delivery integration in `tests/test_video_pipeline_report_integration.py` |
| Annotation writer | None | Dummy frames/detections in `tests/test_video_pipeline_annotation_writer.py` |
| Pipeline import side effects | Basic loader spy | YOLO/Keras/Hugging Face import and download guards |
| End-to-end smoke | None | `scripts/smoke_check.py` |
| Performance helpers | None | Includes outcome and complete report-pipeline timing |
| CI | None | `.github/workflows/tests.yml` |
| Video Analysis UI | Manual only | Still manual (Streamlit) |
| Live Session / camera | Manual only | Still manual |
| Real YOLO inference | None | Intentionally excluded (no model files in CI) |

### Test run results (local)

```
python -m compileall -q Backends   → pass
python -m pytest -q                -> 79 passed, 1 skipped locally (optional OpenCV/imageio package absent)
python scripts/smoke_check.py      → Smoke checks passed
python scripts/performance_check.py -> pass, including the synthetic report pipeline
```

Synthetic videos are created only under pytest `tmp_path`; none are committed.
These tests require no YOLO/Keras weights, GPU, camera, `HF_TOKEN`, or Hugging
Face download. Real-model detection accuracy and real-video codec coverage
remain manual/local checks with the full Python 3.11 dependencies installed.

---

## 7. Commands

```bash
pip install -r requirements-dev.txt
python -m compileall -q Backends
python -m pytest -q
python scripts/smoke_check.py
python scripts/performance_check.py
streamlit run main.py
```

---

## 8. Risky areas left untouched

- The page modules still contain mature frame loops, but the Live Session to
  Video Analysis cross-import has been removed
- Internal `generate_wagon_wheel_data()` (not rendered as map; used for field history context)
- Bat refinement second video pass near impact
- Ensemble multi-model mode

---

## 9. Recommended next improvements

1. Extract `detection_pipeline.py` from `video_analysis.py` and point Live Session at it
2. Add integration tests with tiny synthetic video (no real weights) once pipeline is extracted
3. Add `pytest` to developer onboarding in README
4. Profile real 197-frame clip with performance expander; compare Smart Balanced vs Accurate
5. Consider marking slow UI import tests with `@pytest.mark.integration` if split grows

---

## 10. Visual Observer and 2D tracking repair

- Added a local, deterministic Visual Observer layer before uploaded-video
  impact, shot, direction, outcome, and agent reports.
- Short bounded ball gaps (up to four frames) are repaired with linear
  interpolation. Long or unbounded gaps are left unchanged.
- Low-confidence detections and impossible isolated jumps are marked untrusted
  and downgraded. Raw detections remain available for debugging.
- The report and Session Results UI show original versus repaired coverage,
  repaired frames, suspicious detections, confidence, and the observer decision.
- This is 2D frame-coordinate repair, not true 3D tracking. No models, external
  APIs, map outputs, eager Keras/TensorFlow imports, or download behavior were
  added or changed.
- Synthetic unit, report-pipeline, smoke, and performance checks cover the
  feature without YOLO weights, GPU, camera, video, or `HF_TOKEN`.

Future work remains deliberately separate: validate Video Analysis first, then
consider Live Session integration; pseudo-3D pitch calibration can come later;
real multi-camera 3D tracking remains research scope.

---

## 11. Practice Environment Calibration — Part 1

- Added an import-safe, model-free `calibration/` package with stable context
  normalization, validation, stump-reference estimation, pitch-corridor
  estimation, pitch-end/crease references, and handedness-aware stump-line
  references.
- Video Analysis now offers optional camera-view, batter-handedness,
  auto-estimation, and confirmation controls before analysis.
- The provisional context is refined from stump detections already produced by
  the normal frame loop. No extra YOLO pass or page-load model activity was
  added.
- `report_pipeline.py` accepts and returns normalized `calibration_context`
  while preserving every existing report.
- Video Analysis and Session Results render text-only calibration summaries;
  no pitch, field, bounce, wagon-wheel, or shot-placement map was introduced.
- Session JSON stores only the small JSON-safe context. Old records normalize
  to a disabled default.
- This is approximate 2D practice-environment context, not metric 3D tracking,
  AR glasses integration, or a virtual-field engine.
- The context prepares future AR Nets Mode, improved line/length rules, virtual
  fielder reasoning, and calibrated ROI optimization.
- Model registry, Hugging Face fallback, and lazy-only Keras behavior are
  unchanged.
