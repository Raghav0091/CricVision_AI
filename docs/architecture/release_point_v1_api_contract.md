# Release Point V1 API Contract

Date: 2026-07-23

Release Point V1 belongs to the active Video Analysis pipeline:

```text
apps/web -> services/api/routes/video_analysis.py
         -> services/api/services/video_release_point_service.py
         -> outputs/video_analysis/<analysis_id>/reports/release_point_v1.json
```

## Preconditions

Release Point V1 requires these persisted server-side artifacts:

```text
outputs/video_analysis/<analysis_id>/raw/<stored_filename>
outputs/video_analysis/<analysis_id>/detections/detections.json
outputs/video_analysis/<analysis_id>/tracking/tracking_result.json
outputs/video_analysis/<analysis_id>/calibration/calibration.json
```

Optional inputs are loaded when present:

```text
outputs/video_analysis/<analysis_id>/calibration/calibration_v2.json
outputs/video_analysis/<analysis_id>/calibration/camera_pose.json
```

The API never depends on browser-only detection payloads. It loads full
`detections.json`, including preserved top-K ball candidates, server-side.

## Start

```http
POST /video-analysis/{analysis_id}/release-point/start
```

Response `202`:

```json
{
  "success": true,
  "status": "queued",
  "analysis_id": "analysis_...",
  "job_id": "release_point_job_...",
  "progress": 0,
  "message": "Release Point V1 queued."
}
```

Conflict `409` means a required upstream artifact is missing or a release job is
already active for the analysis.

## Poll Job

```http
GET /video-analysis/{analysis_id}/release-point/job/{job_id}
```

Job status values:

```text
queued
loading_inputs
generating_candidates
scoring_candidates
saving_results
ready
unresolved
failed
```

Response:

```json
{
  "success": true,
  "status": "ready",
  "analysis_id": "analysis_...",
  "job_id": "release_point_job_...",
  "progress": 100,
  "created_at": "2026-07-23T00:00:00Z",
  "updated_at": "2026-07-23T00:00:02Z",
  "error_message": null,
  "result": {
    "release_json_url": "/static/video-analysis/analysis_.../reports/release_point_v1.json"
  },
  "message": "Release Point V1 completed."
}
```

For `unresolved`, `success` is `false`, `progress` is `100`, and the persisted
document still exists with explicit quality flags.

## Fetch Result

```http
GET /video-analysis/{analysis_id}/release-point
```

Response:

```json
{
  "success": true,
  "status": "ready",
  "analysis_id": "analysis_...",
  "release_json_url": "/static/video-analysis/analysis_.../reports/release_point_v1.json",
  "result": {
    "schema_version": "1.0",
    "analysis_id": "analysis_...",
    "status": "ready",
    "release_frame": 43,
    "release_time_seconds": 1.433333,
    "release_point_px": {"x": 742.1, "y": 311.5},
    "confidence": 0.89,
    "frame_uncertainty": {"start": 42, "end": 44},
    "method": "release_point_v1_heuristic_fusion",
    "evidence_mode": "observed_pose_ball_separation",
    "release_type": "OBSERVED_RELEASE",
    "evidence": {},
    "quality_flags": [],
    "provenance": {}
  },
  "candidate_scores": [],
  "quality_summary": {},
  "message": "Release Point V1 completed."
}
```

`result.release_point_px` is pixel-space only. V1 does not report metric 3D
release coordinates.

Possible `result.evidence_mode` values:

```text
observed_pose_ball_separation
trajectory_pose_inferred
fallback_trajectory_only
unresolved
```

Possible `result.release_type` values:

```text
OBSERVED_RELEASE
INFERRED_RELEASE
UNRESOLVED
```

## Persisted Document

The durable report is written to:

```text
outputs/video_analysis/<analysis_id>/reports/release_point_v1.json
```

Shape:

```json
{
  "schema_version": "1.0",
  "analysis_id": "analysis_...",
  "created_at": "2026-07-23T00:00:00Z",
  "completed_at": "2026-07-23T00:00:02Z",
  "result": {},
  "candidate_scores": [],
  "quality_summary": {},
  "message": "Release Point V1 completed."
}
```

