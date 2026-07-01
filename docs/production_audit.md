# CricVision AI production audit

Audit date: 2026-07-01  
Runtime target: Python 3.11, Streamlit, OpenCV, Ultralytics YOLO  
Scope: public MVP (`main.py`) plus backend, dev-only pages, tests, scripts, and Git hygiene.

## 1. Executive summary

The project has a sound MVP boundary: page routing is small, model loading is
lazy, one shared detection timeline feeds reports, and deterministic observer
repair has no model or network dependency. The main production risks are not
algorithmic. They are maintenance and lifecycle risks:

- `ui/video_analysis.py` (about 1,900 lines) and `ui/live_session.py` (about
  1,360 lines) each combine UI, orchestration, frame loops, and result shaping.
- Output and model paths are repeated and mostly relative to the current
  process directory.
- Uploaded Video Analysis input is written with `delete=False` and is not
  removed after analysis.
- Exact detection-preset dictionaries and several thin UI forwarding wrappers
  are duplicated.
- There is no retention policy for generated videos, reports, review frames, or
  corrupt-session backups.
- Four model weights are tracked in Git (about 24 MB total). This is intentional
  current state and is not changed by this audit, but it should be an explicit
  repository policy.

The safe cleanup for this pass is intentionally narrow: central path and preset
constants, removal of proven-unused wrappers/imports, temporary upload cleanup,
one dev-only absolute-path fix, import-safety tests, and documentation. The
large page modules should not be split in the same pass.

## 2. Current architecture map

### Entrypoint and pages

`main.py` owns Streamlit page configuration and sidebar routing:

- Dashboard → `ui/dashboard.py`
- Live Session → `ui/live_session.py`
- Video Analysis → `ui/video_analysis.py`
- Results → `ui/results_page.py`

`SHOW_DEV_PAGES = False` gates Field Map, Datasets, and Training:

- `ui/field_map.py`
- `ui/datasets_page.py`
- `ui/training_page.py`
- `ui/interactive_field_map.py` supplies the production field-setup input card
  as well as dev-only visual helpers.

### Video pipeline

- `video_pipeline/video_reader.py`: safe OpenCV opening, metadata, iteration,
  first-frame extraction, and frame writing.
- `video_pipeline/detection_pipeline.py`: model selection, class mapping, ROI
  detection, local recovery, calibration helpers, and lazy YOLO construction.
- `video_pipeline/report_pipeline.py`: Visual Observer repair followed by
  observer timeline, impact, shot, direction, outcome, and agent reports.
- `video_pipeline/annotation_writer.py`: overlays, review-frame exports, video
  conversion, and impact markers.
- `video_pipeline/performance_timer.py`: stable timing/profile schema.

The mature frame loops still live in the two UI page modules. That is the
largest architecture debt, but moving them now would be a risky rewrite.

### Agents

- `agents/tracking_repair_agent.py`: deterministic 2D gap repair and anomaly
  downgrading.
- `agents/visual_observer_agent.py`: high-level repair decision and report.
- `agents/observer_timeline.py`: detection coverage/quality summary.
- `agents/vision_agent.py`: deterministic consistency and confidence review.

### Analysis

- Detection normalization and speed: `frame_detection_utils.py`,
  `analysis_speed.py`, `smart_pipeline.py`, `bat_detection.py`
- Cricket results: `impact_detection.py`, `shot_classification.py`,
  `shot_direction.py`, `outcome_prediction.py`, `delivery_enrichment.py`,
  `cricket_agent.py`
- Field context: `field_geometry.py`, `field_zones.py`
- Older trajectory utilities: `tracking/ball_tracking_utils.py`

### Models and storage

- `models/model_registry.py`: local-first registry and non-downloading status.
- `models/model_loader.py`: cached lazy YOLO/Keras construction.
- `models/remote_model_loader.py`: lazy Hugging Face fallback.
- `storage/session_store.py`: normalized, backward-compatible JSON records and
  summary views.

### Tests and scripts

There are 64 collected lightweight tests across configuration, normalization, impact,
direction, outcome, observer timeline, tracking repair, report integration,
session compatibility, import safety, and video helpers. OpenCV-dependent
modules may be skipped when the optional codec/runtime is absent.

- `scripts/smoke_check.py`: no-model report and import smoke path.
- `scripts/performance_check.py`: synthetic helper/report timings, including
  200-frame tracking repair.
- CI compiles Backends, runs pytest, and runs smoke checks on Python 3.11.
  The performance script and `git diff --check` are local/manual checks.

## 3. Import and dependency audit

### Heavy or runtime-sensitive imports

| Module | Import behavior | Assessment |
|---|---|---|
| `ui/*` | Imports Streamlit at module scope | Correct for UI modules |
| `ui/live_session.py` | Imports WebRTC/AV inside runtime functions | Lazy and acceptable |
| `ui/interactive_field_map.py` | Imports Matplotlib at module scope | Dev/field-setup module; relatively heavy |
| `video_pipeline/detection_pipeline.py` | Imports NumPy at module scope; Streamlit optional; OpenCV proxy | Import-safe for models, moderately heavy |
| `tracking/ball_tracking_utils.py` | Imports pandas at module scope | Only loaded by analysis pages, not Dashboard |
| `detection/yolo_detector.py` | Imports Ultralytics at module scope | Legacy and unsafe for general imports; not active |
| `models/model_loader.py` | Imports Ultralytics/TensorFlow only inside explicit load functions | Correct and lazy |
| `models/remote_model_loader.py` | Imports Hugging Face client only inside download | Correct and lazy |
| `utils/cv2_loader.py` | Defers OpenCV import until attribute access | Correct |

Dashboard imports only model-registry path checks and UI helpers. It does not
call a loader or remote download. `shot_classifier.keras` remains registered,
experimental, `lazy_only`, and has no active caller.

### Internal dependency direction

- UI modules import analysis, models, storage, and video-pipeline modules.
- No analysis, agent, model, storage, or video-pipeline module imports
  `Backends.src.ui`.
- Tests can import backend report and repair modules without starting
  Streamlit.
- Model modules optionally know how to display Streamlit warnings. This is a
  small boundary compromise, not an active cycle or model-loading trigger.

### Circular imports

Static inspection finds one localized cycle:

`analysis_speed` imports smart settings from `smart_pipeline`; the
`refine_bat_detections_near_impact` function in `smart_pipeline` lazily imports
resize/scale helpers from `analysis_speed`.

Because the reverse import occurs inside a function, import-time execution is
currently safe. It should be removed when frame-loop helpers are extracted, not
by moving code during this cleanup.

## 4. Large-file audit

| Approx. lines | File | Purpose and risk | Action |
|---:|---|---|---|
| 1,901 | `ui/video_analysis.py` | UI, upload lifecycle, two frame loops, calibration, outputs | Split later, after real-video regression fixture |
| 1,359 | `ui/live_session.py` | WebRTC state, recording, detection loop, reports, UI | Split later; webcam behavior is high risk |
| 1,056 | `ui/components.py` | All report, result, filter, and utility renderers | Optional domain-based UI split later |
| 791 | `ui/interactive_field_map.py` | Field input plus dev-only plotting/canvas code | Keep; separate input from plotting later |
| 722 | `storage/session_store.py` | Persistence, normalization, summaries, report views | Split schema/view helpers later only with migration tests |
| 694 | `video_pipeline/detection_pipeline.py` | Models, ROI, recovery, calibration, parsing | Cohesive enough for MVP; split by responsibility later |
| 648 | `analysis/field_zones.py` | Field geometry compatibility and persistence | Keep; dev/field context is mature and sensitive |
| 492 | `agents/tracking_repair_agent.py` | Defensive shape handling and repair | Keep now; consider helper reduction after validation |
| 421 | `ui/theme.py` | CSS and shared UI primitives | Keep |

No large file should be split in this pass. Tests do not yet exercise real
Video Analysis or Live Session frame loops deeply enough to make that safe.

## 5. Duplicate-code audit

| Duplication | Finding | Decision |
|---|---|---|
| Detection presets | Exact dictionary duplicated in Live Session and Video Analysis | Move to shared constants now |
| Output paths | Review/report/video paths repeated across UI, analysis, storage, and writers | Centralize stable directory constants now |
| Model paths | Three legacy/current weight paths repeated in pipeline and Live Session | Centralize path values now; preserve registry behavior |
| Low-confidence threshold | `0.35` repeated in observer, detection pipeline, and Live Session | Shared threshold is reasonable now |
| Detection parsing | Similar center/box/confidence parsing in frame utilities, observer, vision, and repair | Keep local now; semantics differ and consolidation is riskier |
| Agent coverage/center helpers | Exact private helpers duplicated in observer and vision agents | Defer; small and private |
| UI persistence/default wrappers | Identical forwarding wrappers in both production page modules | Replace with direct shared imports now |
| Session save logic | Central implementation already exists in `ui/analysis_helpers.py` and `storage/session_store.py` | Preserve |
| Video opening/writing | Shared helpers exist, but mature page loops still perform some direct OpenCV I/O | Defer until frame-loop extraction |
| UI card renderers | Centralized in `ui/components.py`; no active duplicate Visual Observer renderer | Preserve |

## 6. Hardcoded value and path audit

| Value | Classification | Action |
|---|---|---|
| `C:\Dataset` in dev Datasets page | Risky/local absolute path | Replace with project-relative dataset directory |
| `outputs/...` directories in production modules | Should move to path config | Centralize using project-root `Path` objects |
| `Models/...` paths in registry/pipeline/Live | Should move to path config where used as `Path` | Centralize without changing registry metadata |
| Hugging Face repo ID and remote filenames | OK as model configuration constants | Keep |
| Detection preset labels and values | Shared configuration | Centralize |
| Smart pipeline stride/resize settings | Domain-local configuration | Keep in `smart_pipeline.py` |
| Impact/shot/direction thresholds | Domain-local named constants | Keep local |
| Observer jump threshold (`180`) | Shared agent threshold | Centralize if both agents continue using it |
| Tracking repair defaults (`4`, `0.25`, `0.15`) | Public algorithm defaults | Keep with repair agent for API clarity |
| Recording limits and trajectory lengths | Live Session-local operational limits | Keep local |
| Repeated UI labels | Mostly page-specific and cheap | Keep; a label registry would add indirection |
| Secrets/tokens | Only key name `HF_TOKEN`; value read from secrets/env | Safe; never print token |

No production source contains an absolute CricVision workspace path. The one
absolute Windows dataset path is dev-only but still should be removed.

## 7. Dead-code and legacy audit

### SAFE TO REMOVE NOW

Each item has one definition/import occurrence, no caller, no test reference,
and no active documentation contract:

- Unused `draw_label` import in `video_pipeline/detection_pipeline.py`
- `analysis_speed.resolve_analysis_mode` forwarding wrapper
- `bat_detection.find_possible_impact_frame` (documented as superseded)
- `delivery_enrichment.run_direction_and_agent_review` (documented as superseded)
- Duplicate `_persist_result_to_session` and
  `ensure_delivery_report_fields` forwarding functions in Live Session and
  Video Analysis; direct shared imports preserve behavior

Removal must be recorded below after tests verify it.

### KEEP BUT MARK LEGACY

- `detection/yolo_detector.py`: early eager YOLO wrapper; already marked legacy.
- `interactive_field_map.draw_field_map`: dev/debug renderer, not a production
  report output; already marked legacy.
- Legacy field-coordinate compatibility in `field_zones.py`: needed for saved
  data/presets.

### KEEP FOR FUTURE FEATURE

- `smart_pipeline.crop_frame_with_roi` and
  `restore_roi_boxes_to_full_frame`
- `model_loader.get_cached_keras_model` and shot-classifier registry metadata
- Experimental player/segmentation registry entries
- Dev-only Field Map, Datasets, and Training pages
- Remote model status/registry helper functions
- Field setup card and internal field-history context

### UNSURE — DO NOT DELETE

- Public but currently uncalled component helpers such as `hero_section`,
  `primary_action_card`, `report_history_card`, and developer expanders
- Uncalled field label/geometry helpers
- Live Session-local `collect_detections`,
  `save_low_confidence_review_frame`, and `show_delivery_report`; these may be
  retained external/manual hooks around webcam work
- Model cache clearing and model listing APIs

There are no dead commented-out blocks. Uncertainty is resolved in favor of
keeping code.

## 8. Production-readiness audit

### Video upload and files

- File selection is extension-filtered, but there is no explicit upload-size
  cap or content signature validation.
- OpenCV returns friendly failures for unreadable videos, invalid dimensions,
  no frames, and writer creation failure.
- Browser conversion already has fallback behavior.
- Manual first-frame extraction cleans its temporary file.
- Main Video Analysis upload currently leaves its `delete=False` temporary file
  behind; fix in this pass with a standard-library temporary directory.
- Generated videos, reports, review frames, clips, field history, and corrupt
  JSON backups have no retention policy. Automatic deletion is unsafe because
  Session Results may reference generated clips.

### Models and remote fallback

- Local paths win; remote download occurs only on explicit model use.
- Missing models, missing `huggingface_hub`, missing token, offline downloads,
  invalid model types, and constructor errors return `None` with warnings.
- No model loads during Dashboard import/status checks.
- No TensorFlow import exists on the active path.
- Exceptions are reported without printing `HF_TOKEN`.

### Detection and reports

- Empty/no-ball/no-bat/no-stump timelines return conservative report results.
- Visual Observer repair catches failures and falls back to raw detections.
- Raw and repaired timelines remain available separately.
- Report algorithms are deterministic and do not call models or APIs.
- Unexpected exceptions in the full UI processing call are not consistently
  wrapped at the page boundary; a narrow user-friendly boundary is useful when
  adding temporary-file cleanup.

### Session Results

- Missing JSON returns an empty list.
- Malformed JSON is moved aside and treated as empty.
- Normalization supports old records and absent observer-repair keys.
- Writes use a temporary JSON followed by replace, but there is no file lock.
  Concurrent multi-user writes could race. Local JSON is acceptable for the
  MVP; revisit before true multi-user deployment.

## 9. Testing audit

Strong automated coverage:

- Frame detection normalization and invalid shapes
- Observer timeline and Visual Observer repair
- Impact, shot direction, and outcome fallbacks
- Shared report contract and repaired-timeline integration
- Old/missing/corrupt session storage
- Import safety and no model/Hugging Face activity
- Video reader and annotation writer with optional lightweight OpenCV fixtures
- No-model smoke and synthetic performance scripts

Important missing coverage:

- Shot classification has no dedicated test module.
- Model registry, local-first resolution, remote failure, and cache behavior
  have import guards but few direct unit tests.
- `analysis_speed`, `smart_pipeline`, `ball_tracking_utils`, field geometry,
  and field-zone persistence lack focused tests.
- Video Analysis and Live Session frame loops remain mostly manual.
- Upload cleanup and generated-output retention are not tested.
- Dashboard “no model load” is inferred through import tests, not a Streamlit
  AppTest.
- CI does not run the performance script or `git diff --check`.

Tests require no real YOLO/Keras files, GPU, camera, `HF_TOKEN`, internet, or
large cricket video.

## 10. Git hygiene audit

`.gitignore` correctly excludes environments, Python caches, Streamlit secrets,
 outputs, session JSON/clips, remote/CricShot models, archives, videos, Keras
weights, and common IDE files.

Findings:

- No tracked outputs, session JSON, remote models, secrets, caches, generated
  videos, or Keras files were found.
- Tracked weights: `yolov8n.pt`, `Models/ball_detector/best.pt`,
  `Models/cricket_objects/best.pt`, and
  `Models/cricket_objects/best_external.pt` (about 24 MB total).
  The task explicitly forbids deleting model files, so they remain untouched.
- Local untracked `.cursor/`, `AGENTS.md`, and `cls` exist. `cls` is captured
  terminal output. They are outside production code and must not be staged
  accidentally.
- The old root `.pytest_cache` is unreadable on this Windows workspace.
  `pytest.ini` now routes both temporary files and cache data under ignored
  `outputs/`, so the documented pytest command remains portable and clean.

## 11. Planned safe cleanup for this pass

1. Add small `Backends/src/config` modules for stable paths and truly shared
   presets/thresholds.
2. Preserve existing module-level constant names through imports/aliases.
3. Remove only the proven-safe items listed above.
4. Replace the dev-only `C:\Dataset` path.
5. Ensure uploaded Video Analysis temporary input is cleaned on success and
   failure, with a user-friendly unexpected-error boundary.
6. Add config/import and upload-helper tests only where they add a stable
   contract.
7. Re-run compile, full pytest, smoke, performance, and diff checks.

## 12. Manual Streamlit checklist

- [ ] Dashboard opens quickly.
- [ ] Dashboard import/status does not load a model.
- [ ] Video Analysis opens.
- [ ] A short video uploads and previews.
- [ ] Smart Balanced completes.
- [ ] Visual Observer Repair card appears.
- [ ] Impact, Shot, Direction, and Outcome reports appear.
- [ ] Session Results saves the new record.
- [ ] Old saved results still open.
- [ ] Live Session opens and webcam controls remain stable.
- [ ] Dev-only pages remain hidden with `SHOW_DEV_PAGES=False`.
- [ ] No bounce map, field map, shot-placement map, or wagon-wheel output is
  rendered in production reports.
- [ ] No secrets are printed in the UI or logs.

## 13. Cleanup record

No source was deleted before this Phase 1 audit was written. The following
cleanup was then performed:

| Removed or changed | Why safe | Verification |
|---|---|---|
| Unused `draw_label` import from `detection_pipeline.py` | No module/test caller; annotation writer remains source of the helper | Import tests and static search |
| `analysis_speed.resolve_analysis_mode` | No caller; active code uses smart settings directly | Static search and import tests |
| `bat_detection.find_possible_impact_frame` plus two private-only helpers | Documented superseded wrapper with no caller/test | Static search; impact/report tests |
| `delivery_enrichment.run_direction_and_agent_review` | Documented superseded wrapper with no caller/test | Static search; report integration tests |
| Four duplicate UI forwarding functions | Pages now import the shared implementations from `ui/analysis_helpers.py` | UI import and session tests |
| Duplicate detection presets, model/output paths, and shared thresholds | Exact values moved to import-safe config modules | Config and import-safety tests |
| Dev-only `C:\Dataset` | Machine-specific and not portable | Replaced with project-root `data/datasets` |
| Persistent uploaded-video temp file | File is not needed after processing returns | `TemporaryDirectory` cleans success/failure paths |
| Partial outputs from failed analysis/conversion | Failed results are not displayed or persisted | Only paths created for that failed run are unlinked |
| Pytest global temp/cache location | Permission-denied machine state broke the documented command | `pytest.ini` now uses ignored workspace output paths |

No files, models, pages, registry entries, remote-loading behavior, reports, or
session compatibility paths were removed.

## 14. Remaining technical debt and recommended next steps

1. Validate the Visual Observer on representative real videos before enabling
   it in Live Session.
2. Add a small real-cricket-video benchmark that stays local and optional.
3. Add model registry/resolution failure tests without requiring weight files.
4. Add an explicit retention policy for outputs and corrupt-session backups;
   do not delete files still referenced by saved results.
5. Add session-write locking or move storage only when real multi-user traffic
   makes local JSON insufficient.
6. Consider pseudo-3D pitch calibration later. Real multi-camera 3D tracking is
   research scope.
7. Split UI components or page frame loops only after real-video regression
   coverage exists.
8. Consider optional model-registry cleanup later; keep experimental entries
   and lazy Keras metadata until that decision is explicit.
