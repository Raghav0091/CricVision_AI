# Release Point V1 Architecture and Audit

Date: 2026-07-23

Scope: architecture/audit only. This document designs True Ball Release Point V1 without implementing the feature.

## 1. Active Pipeline Map

The active uploaded-video workflow is the Next.js + FastAPI persistent Video Analysis flow, not the Streamlit MVP and not the scaffolded worker queue.

```text
apps/web/app/video-analysis/page.tsx
  -> apps/web/lib/api.ts
  -> services/api/main.py
  -> services/api/routes/video_analysis.py
  -> services/api/services/video_analysis_service.py
  -> services/api/services/video_calibration_service.py
  -> services/api/services/video_ball_detection_service.py
  -> services/api/services/video_ball_tracking_service.py
  -> outputs/video_analysis/<analysis_id>/*
```

Current stage sequence:

1. Browser uploads one video through `prepareVideoAnalysis`.
2. FastAPI route `POST /video-analysis/prepare` calls `prepare_video`.
3. `prepare_video` creates an analysis directory, stores the original video, reads FPS/frame count/dimensions, and extracts an early clean reference frame.
4. Browser confirms scene calibration through the calibration panel.
5. `POST /video-analysis/{analysis_id}/ball-detection/start` creates a job and runs `run_video_ball_detection_job` as a FastAPI background task.
6. Ball detection opens the clean original video, runs selected YOLO model on every frame, stores every per-frame candidate in JSON/CSV, and writes a detection overlay video.
7. `POST /video-analysis/{analysis_id}/tracking/start` creates a job and runs `run_video_ball_tracking_job` as a FastAPI background task.
8. Tracking loads `detections.json`, builds the primary moving-ball track, recovers short gaps, estimates primary bounce, and writes tracking JSON/CSV/debug video/replay video.
9. Browser polls job endpoints and displays linked artifacts.

Important finding: `services/worker/worker.py` and `services/worker/pipeline/*` are not the active uploaded-video execution engine. They are scaffold/placeholder code.

## 2. File And Module Audit

### Active UI

- `apps/web/app/video-analysis/page.tsx`: orchestrates upload, calibration, detection, tracking, restore-by-analysis-id, and stage navigation.
- `apps/web/lib/api.ts`: typed client for video-analysis endpoints and result contracts.
- `apps/web/components/video-analysis/SceneCalibrationPanel.tsx`: scene calibration UI.
- `apps/web/components/video-analysis/BallDetectionPanel.tsx`: model selection and every-frame detection job polling. Defaults to E4C.
- `apps/web/components/video-analysis/BallTrackingPanel.tsx`: tracking job polling and replay/debug display.

### Active API And Persistence

- `services/api/main.py`: mounts routes and static output directories.
- `services/api/routes/video_analysis.py`: route layer. It should remain orchestration only.
- `services/api/schemas/video_analysis.py`: active Pydantic contracts for prepared analysis, calibration, detections, tracking, bounce, and job status.
- `services/api/services/video_analysis_service.py`: analysis ID creation, upload persistence, metadata, early reference frame extraction.
- `services/api/services/video_ball_detection_job_store.py`: in-memory/background job status store.
- `services/api/services/video_ball_tracking_job_store.py`: in-memory/background job status store.

### Active Scene Calibration

- `services/api/services/video_calibration_service.py`: confirmed legacy/2D scene calibration. Produces `calibration.json`, `calibration_overlay.jpg`, optional `scene_overlay.mp4`.
- `services/api/services/video_calibration_v2_service.py`: ground-plane homography and metric pitch landmarks. This is available but is calibration v2A, not a full 3D release solution.
- `services/api/services/video_camera_pose_service.py`: wicket-body landmark PnP camera pose. This is available but depends on explicit landmark quality and intrinsics assumptions.
- `packages/cricket_vision/calibration/*`: shared cricket geometry primitives used by calibration v2.

### Active Ball Detection

- `services/api/services/ball_detector_registry.py`: trusted model registry.
  - E2: `Models/Copy of ball_only_E2_1280_baseline.pt`
  - E3: `Models/ball_only_E3_1280_motion_blur.pt`
  - E4C: `Models/Copy of ball_only_E4C_1280_random_sampling_control.pt`
- `services/api/services/video_ball_detection_service.py`: every-frame YOLO inference and candidate persistence.
- `services/api/services/ball_detection_clip.py`: shared YOLO extraction/transcode helper used by detection services.

### Active Tracking, Trajectory Output, Replay

- `services/api/services/video_ball_tracking_service.py`: active Complete Delivery Tracking v2 for uploaded video analysis.
  - Loads all saved detections.
  - Flattens top-K frame candidates.
  - Scores moving candidates.
  - Builds a primary tracklet.
  - Recovers short gaps.
  - Rejects outliers/trims shot-like outgoing path.
  - Detects primary bounce.
  - Writes `tracking_result.json`, `tracking_points.csv`, `tracking_summary.json`, `tracking_debug.mp4`, and `delivery_replay.mp4`.

### Legacy Or Non-Active Areas To Avoid

- `main.py`, `Backends/src/ui/*`, and Streamlit paths are not the active Next.js/FastAPI uploaded-video pipeline for this audit.
- `services/worker/worker.py` and `services/worker/pipeline/*` are currently scaffold/placeholders.
- `Backends/src/tracking/trajectory_scorer.py`, `Backends/src/tracking/trajectory_fit.py`, and `Backends/src/physics_trajectory.py` contain useful legacy physics/trajectory heuristics and tests, but they are not directly called by `video_ball_tracking_service.py`.
- `packages/cricket_vision/README.md` explicitly says tracking, trajectory, and replay remain in the active backend until a real migration replaces them. Do not create placeholder duplicates there.

## 3. Existing Per-Frame Data

Prepared analysis metadata:

- `analysis_id`
- original filename and stored filename
- raw video URL
- reference frame URL/index
- `fps`
- `frame_count`
- `duration_seconds`
- `width`
- `height`
- codec
- calibration/detection/tracking status and artifact URLs

Detection document:

```json
{
  "analysis_id": "...",
  "detector": {"key": "e4c_best_overall", "name": "...", "model_file": "..."},
  "model_path_used": "...",
  "settings": {"frame_stride": 1, "imgsz": 960, "confidence_threshold": 0.15, "max_det": 20},
  "frames": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "processed": true,
      "detections": [
        {
          "candidate_id": "frame_000000_candidate_001",
          "class_id": 0,
          "class_name": "ball",
          "confidence": 0.73,
          "bbox_xyxy": [x1, y1, x2, y2],
          "bbox_normalized": {"x": 0.1, "y": 0.2, "width": 0.01, "height": 0.01},
          "center": {"x": 742.1, "y": 311.5},
          "center_normalized": {"x": 0.58, "y": 0.43},
          "width_pixels": 8.0,
          "height_pixels": 8.0,
          "area_pixels": 64.0,
          "inside_pitch_corridor": true
        }
      ]
    }
  ]
}
```

Top-K finding: top-K detections are preserved in `detections.json` and `detections.csv`; the web result only exposes candidate counts and links. Release V1 should load the JSON server-side rather than depending on the browser payload.

Tracking document:

- `primary_track`: final selected/recovered points.
- `raw_primary_track`: association before refinement/outlier trim.
- `candidate_diagnostics`: selected/rejected candidate scoring diagnostics.
- `bounce`: primary bounce result.

Each tracking point has:

- `frame_index`
- `timestamp_seconds`
- `source`
- `provenance`: `OBSERVED`, `TRACKER_RECOVERED`, `PHYSICS_RECONSTRUCTED`, or `PROJECTED`
- `candidate_id`
- pixel and normalized center
- confidence
- uncertainty
- velocity proxy `vx`, `vy`
- prediction error
- pitch-corridor flag

Clean original frames are available at:

```text
outputs/video_analysis/<analysis_id>/raw/<stored_filename>
```

Release pose estimation must read this raw video. Detection/tracking/replay overlays are display artifacts only and must never be used as CV input.

## 4. Proposed Release V1 Architecture

Add release logic as an engine module, separate from API and UI:

```text
services/api/services/video_release_point_service.py        # orchestration/persistence
services/api/schemas/release_point.py                      # stable API/document contracts
Backends/src/release_point/pose_provider.py                # provider interface
Backends/src/release_point/bowler_tracker.py               # bowler identity and pose sequence
Backends/src/release_point/release_engine.py                # candidate generation/fusion
Backends/src/release_point/features.py                      # feature extraction
Backends/src/release_point/validation.py                    # metrics/annotation helpers
```

This keeps the current deployment simple while preventing release logic from leaking into the UI or route layer. If/when a migration to `packages/cricket_vision` is warranted, move only stable contracts and pure geometry helpers after the first prototype proves itself.

### A. PoseProvider

Purpose: abstract human pose estimation from release logic.

Conceptual interface:

```python
class PoseProvider:
    provider_name: str
    model_name: str
    keypoint_schema: str

    def estimate_frame(self, frame_bgr, frame_index: int, timestamp_seconds: float) -> list[PosePerson]:
        ...

    def estimate_sequence(self, video_path, frame_window: range, fps: float) -> PoseSequence:
        ...
```

Output contract must normalize model-specific keypoints into a stable cricket pose schema:

- person box
- person confidence
- keypoints for shoulder, elbow, wrist, hip, knee, ankle
- optional hand keypoints when available
- per-keypoint confidence
- source provider/model
- frame index/time

### B. BowlerTracker

Purpose: select and track the bowler, not the batter/umpire/fielder.

Inputs:

- clean frames from raw video
- pose sequence
- scene calibration: 2D pitch corridor, bowler/striker wicket geometry, optional calibration v2/camera pose
- ball detections and primary track

Responsibilities:

- define a bowling-end ROI from calibration; initial assumption is non-striker/bowler end for behind-bowler videos, with metadata noting uncertainty
- select the person whose pose enters/operates from the bowling-end ROI
- maintain identity across frames by box/keypoint continuity
- reject stationary background persons and batter-side persons
- output one `BowlerPoseSequence`

### C. ReleaseCandidateGenerator

Purpose: reduce the search to plausible release frames.

Candidate sources:

- observed ball track start vicinity
- earliest reliable moving-ball segment
- frames where ball is near wrist/hand and then persistently separates
- wrist/arm velocity peak vicinity
- backward trajectory extrapolation into pre-track frames

Release V1 should search a configurable window around the primary track start, plus a limited backward window. It must not equate first detection with release.

### D. ReleaseFeatureExtractor

Features per candidate frame:

- ball-to-wrist distance in pixels
- normalized ball-to-wrist distance, scaled by shoulder width or torso size when available
- separation velocity
- persistence of separation over next N frames
- wrist velocity
- wrist acceleration proxy
- elbow/wrist/shoulder arm geometry
- bowling-arm extension angle proxy
- early trajectory direction consistency
- backward trajectory fit consistency
- forward free-flight confirmation
- detector confidence and candidate rank
- tracker confidence/provenance
- scene ROI consistency
- pose/keypoint confidence

### E. ReleaseEstimator

Purpose: fuse evidence into one stable release result.

Required methods:

- `observed_pose_ball_separation`: ball observed near hand, then separates persistently into tracked free flight.
- `trajectory_pose_inferred`: pose timing available but ball missing at exact hand separation; infer from backward/forward trajectory and arm motion.
- `fallback_trajectory_only`: no reliable pose; estimate from earliest reliable free-flight segment and mark low confidence.

The estimator should return both observed and inferred release classifications. It must expose evidence and uncertainty rather than a single magic frame.

### F. ReleaseResult

Stable result contract:

```json
{
  "schema_version": "1.0",
  "analysis_id": "analysis_...",
  "status": "ready",
  "release_frame": 43,
  "release_time_seconds": 1.433,
  "release_point_px": {"x": 742.1, "y": 311.5},
  "confidence": 0.89,
  "frame_uncertainty": {"start": 42, "end": 44},
  "method": "pose_ball_temporal_fusion",
  "release_type": "OBSERVED_RELEASE",
  "evidence": {
    "ball_candidate_id": "frame_000043_candidate_001",
    "bowler_person_id": "bowler_01",
    "wrist_keypoint": {"x": 735.2, "y": 309.8, "confidence": 0.82},
    "ball_wrist_distance_px": 7.1,
    "normalized_ball_wrist_distance": 0.18,
    "separation_persistence_frames": 4,
    "forward_track_points": 8,
    "backward_fit_error_px": 5.4,
    "detector_confidence": 0.76,
    "track_confidence": 0.81,
    "pose_confidence": 0.78
  },
  "quality_flags": [],
  "provenance": {
    "pose_provider": "rtmpose",
    "pose_model": "rtmw-x",
    "ball_detector_model_key": "e4c_best_overall",
    "tracking_version": "delivery_track_v2",
    "calibration_sources": ["legacy_2d", "calibration_v2_if_available"]
  }
}
```

Do not add metric 3D release position in V1 unless camera pose and depth evidence are genuinely reliable. Pixel-space release is the correct first contract.

## 5. Recommended PoseProvider Prototype

Primary prototype recommendation: RTMPose/RTMW through MMPose, wrapped behind `PoseProvider`.

Reasoning:

- better suited than MediaPipe for small/far athletes and sports frames
- stronger wrist/arm localization options and available whole-body variants
- can run on CUDA when available, CPU as fallback for small clips
- better benchmark path for a professional sports-analysis system
- open model ecosystem, but dependency weight and Windows install complexity must be measured before making it mandatory

Fallback: MediaPipe can be a secondary provider only after the interface exists. It is useful for quick CPU smoke tests, but should not be the architectural choice for release accuracy.

Prototype policy:

- Do not install MMPose in the main runtime until a small compatibility spike passes on Windows/Python/GPU and CPU.
- Keep provider dependencies optional.
- Save provider name/model/schema in every result.

## 6. Exact Shared Data Contracts

### ReleaseAnalysisInput

```json
{
  "analysis_id": "analysis_...",
  "raw_video_path": "outputs/video_analysis/.../raw/original_video.mp4",
  "fps": 30.0,
  "frame_count": 180,
  "width": 1280,
  "height": 720,
  "detections_path": "outputs/video_analysis/.../detections/detections.json",
  "tracking_path": "outputs/video_analysis/.../tracking/tracking_result.json",
  "calibration_path": "outputs/video_analysis/.../calibration/calibration.json",
  "calibration_v2_path": null,
  "camera_pose_path": null
}
```

### PosePerson

```json
{
  "person_id": "provider_local_3",
  "frame_index": 43,
  "timestamp_seconds": 1.433,
  "bbox_xyxy": [620.0, 120.0, 810.0, 610.0],
  "confidence": 0.86,
  "keypoints": {
    "right_shoulder": {"x": 720.0, "y": 210.0, "confidence": 0.88},
    "right_elbow": {"x": 745.0, "y": 250.0, "confidence": 0.81},
    "right_wrist": {"x": 735.2, "y": 309.8, "confidence": 0.82}
  },
  "provider": {"name": "rtmpose", "model": "rtmw-x", "schema": "coco_wholebody"}
}
```

### BowlerPoseSequence

```json
{
  "bowler_id": "bowler_01",
  "selection_confidence": 0.79,
  "frames": [43],
  "poses_by_frame": {"43": {}},
  "quality_flags": ["bowling_end_assignment_uncertain"]
}
```

### ReleaseResultDocument

```json
{
  "analysis_id": "analysis_...",
  "created_at": "2026-07-23T00:00:00Z",
  "completed_at": "2026-07-23T00:00:02Z",
  "result": {},
  "candidate_scores": [],
  "quality_summary": {},
  "message": "Release Point V1 completed."
}
```

Persistence path:

```text
outputs/video_analysis/<analysis_id>/reports/release_point_v1.json
```

Optional display artifacts later:

```text
outputs/video_analysis/<analysis_id>/replay/release_overlay.mp4
```

## 7. Parallel Implementation Boundaries

### Agent A: Pose + Bowler Tracking

Owns:

- `Backends/src/release_point/pose_provider.py`
- `Backends/src/release_point/bowler_tracker.py`
- pose provider spike scripts/tests
- pose/bowler portions of `tests/`

Does not edit simultaneously:

- `services/api/routes/video_analysis.py`
- `apps/web/*`
- active ball tracking internals

Shared contracts:

- `PosePerson`
- `PoseSequence`
- `BowlerPoseSequence`

### Agent B: Release Event Engine + Fusion

Owns:

- `Backends/src/release_point/release_engine.py`
- `Backends/src/release_point/features.py`
- `Backends/src/release_point/validation.py`
- `services/api/schemas/release_point.py`
- `services/api/services/video_release_point_service.py`
- release engine tests

Does not edit simultaneously:

- pose provider implementation internals owned by Agent A
- web UI components owned by Agent C
- YOLO model registry or weights

Shared contracts:

- `ReleaseAnalysisInput`
- `ReleaseResult`
- `ReleaseResultDocument`
- candidate feature schema

### Agent C: Release UI/UX

Owns:

- `apps/web/lib/api.ts` release client additions
- a new release panel/component under `apps/web/components/video-analysis/`
- minimal integration in `apps/web/app/video-analysis/page.tsx`

Does not edit simultaneously:

- release engine internals
- pose provider internals
- ball tracking service internals

Shared contracts:

- API response types generated/manually mirrored from `services/api/schemas/release_point.py`
- release artifact URLs and status strings

Merge-conflict minimization:

- Agent B creates API route and schema first.
- Agent C consumes only the route contract after it lands.
- Agent A exposes pose/bowler interfaces and fake/test provider before MMPose integration.

## 8. Integration Sequence

1. Add release schemas and pure interfaces with no heavy pose dependency.
2. Add `ReleaseAnalysisInput` loader that validates existing prepared analysis, detection, tracking, calibration, and raw video paths.
3. Add a deterministic fake/test `PoseProvider` for unit tests.
4. Implement `BowlerTracker` with calibration-aware ROI selection and identity continuity.
5. Implement feature extraction from pose + detections + primary track.
6. Implement release estimator with the three method classes and confidence/uncertainty.
7. Add API endpoints:
   - `POST /video-analysis/{analysis_id}/release-point/start`
   - `GET /video-analysis/{analysis_id}/release-point/job/{job_id}`
   - `GET /video-analysis/{analysis_id}/release-point`
8. Add minimal UI panel after tracking is complete.
9. Spike RTMPose/RTMW provider behind the interface.
10. Run validation against annotated clips and iterate thresholds.

## 9. Risks

- Bowler may be small, blurred, occluded, or partially outside frame at release.
- Wrist keypoint can be unreliable exactly when motion blur is highest.
- Ball may be invisible while still in hand, so observed release is not always possible.
- Calibration may identify wicket ends incorrectly for non-standard camera views.
- Current tracking may start after release; release engine must support backward inference.
- MMPose dependencies may be heavy on Windows and Streamlit/FastAPI deployment.
- In-memory job stores do not survive process restarts; acceptable for current scaffold, but release jobs inherit this limitation unless persistence is upgraded.
- Ground-plane calibration does not provide ball height; do not overclaim 3D.

## 10. Validation Plan

Lightweight annotation format:

```json
{
  "schema_version": "1.0",
  "video_id": "clip_001",
  "analysis_id": "analysis_...",
  "fps": 30.0,
  "annotations": {
    "release_frame": 43,
    "release_point_px": {"x": 742.1, "y": 311.5},
    "visibility": "visible",
    "bowling_arm": "right",
    "confidence": "high",
    "notes": ""
  }
}
```

Metrics:

- exact-frame accuracy
- within +/-1 frame
- within +/-2 frames
- mean absolute frame error
- catastrophic failure rate: no result or error greater than 5 frames
- confidence vs correctness calibration by confidence bins
- observed vs inferred split performance
- quality-flag frequency

Initial validation set:

- 20-30 short clips
- mixed camera angles and resolutions
- E4C detection/tracking already run
- manual release labels from slow-motion frame stepping

## 11. Concise Implementation Checklist

- Add release schemas.
- Add release service skeleton and path loader.
- Add pose provider interface and fake provider.
- Add bowler tracker skeleton.
- Add release feature extraction.
- Add estimator and confidence flags.
- Add tests using synthetic detections/tracks/poses.
- Add API start/job/result endpoints.
- Add minimal web release panel.
- Spike RTMPose/RTMW provider separately.
- Validate against annotated JSON clips.

