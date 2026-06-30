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
| `ui/video_analysis.py` | ~3000 | Split into `detection_pipeline.py`, `report_ui.py` when Live Session decoupled |
| `ui/live_session.py` | ~1300 | Extract shared pipeline from video analysis first |
| `ui/interactive_field_map.py` | ~800 | Keep; field setup card is shared |

### Duplicate helpers (partially addressed)

- ✅ `normalize_frame_detections` consolidated → `frame_detection_utils.py`
- ✅ Session save + report defaults → `analysis_helpers.py`
- ⚠️ `DETECTION_PRESETS` duplicated in Video Analysis + Live Session
- ⚠️ Detection collection helpers differ (`collect_model_detections` vs `collect_detections`)
- ⚠️ Two YOLO loader entry points (registry vs path string)

### Dead / legacy active paths

- `detection/yolo_detector.py` — LEGACY, not imported
- `find_possible_impact_frame`, `run_direction_and_agent_review` — superseded
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
| Uploaded video paths | Temp files via `tempfile`; outputs under `outputs/` |
| Output filenames | Timestamp-based names; no raw user path concatenation in session JSON |
| Debug logs expose secrets | Performance/observer notes are detection stats only |
| Model/output files committed | **Gitignored** (`Models/remote/`, `outputs/`, `data/`) |

**Continue to avoid:** printing `get_hf_token()`, committing `.streamlit/secrets.toml`, logging full env in Streamlit.

---

## 5. Performance improvement ideas

### Safe (recommended next)

- Decouple Live Session from `video_analysis.py` imports via shared `detection_pipeline.py`
- Unify `DETECTION_PRESETS` into one module
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
| End-to-end smoke | None | `scripts/smoke_check.py` |
| Performance helpers | None | `scripts/performance_check.py` |
| CI | None | `.github/workflows/tests.yml` |
| Video Analysis UI | Manual only | Still manual (Streamlit) |
| Live Session / camera | Manual only | Still manual |
| Real YOLO inference | None | Intentionally excluded (no model files in CI) |

### Test run results (local)

```
python -m compileall -q Backends   → pass
python -m pytest -q                → 39 passed, 1 skipped (video_analysis import if imageio_ffmpeg missing locally; installed in requirements.txt / CI)
python scripts/smoke_check.py      → Smoke checks passed
python scripts/performance_check.py --frames 200 → pass (~0.2–0.7 ms per helper on dummy data)
```

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
