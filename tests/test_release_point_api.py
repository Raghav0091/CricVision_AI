from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.routes.video_analysis import router
from services.api.services import video_analysis_service
from services.api.services import video_release_point_service


ANALYSIS_ID = "analysis_20990101_000000_abcdef"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _candidate(frame: int, x: float, y: float):
    return {
        "candidate_id": f"frame_{frame:06d}_candidate_001",
        "class_id": 0,
        "class_name": "ball",
        "confidence": 0.82,
        "bbox_xyxy": [x - 4, y - 4, x + 4, y + 4],
        "bbox_normalized": {
            "x": (x - 4) / 1280,
            "y": (y - 4) / 720,
            "width": 8 / 1280,
            "height": 8 / 720,
        },
        "center": {"x": x, "y": y},
        "center_normalized": {"x": x / 1280, "y": y / 720},
        "width_pixels": 8,
        "height_pixels": 8,
        "area_pixels": 64,
        "inside_pitch_corridor": True,
    }


def _seed_analysis(root: Path) -> None:
    analysis_dir = root / ANALYSIS_ID
    for directory in ("raw", "calibration", "detections", "tracking", "reports"):
        (analysis_dir / directory).mkdir(parents=True, exist_ok=True)
    (analysis_dir / "raw" / "original_video.mp4").write_bytes(b"synthetic")
    (analysis_dir / "calibration" / "reference_frame.jpg").write_bytes(b"synthetic")

    _write_json(
        analysis_dir / "reports" / "analysis_metadata.json",
        {
            "success": True,
            "analysis_id": ANALYSIS_ID,
            "status": "calibrated",
            "original_filename": "clip.mp4",
            "stored_filename": "original_video.mp4",
            "file_size_bytes": 9,
            "created_at": "2026-07-23T00:00:00Z",
            "updated_at": "2026-07-23T00:00:00Z",
            "duration_seconds": 1.333,
            "fps": 30.0,
            "frame_count": 40,
            "width": 1280,
            "height": 720,
            "reference_frame_index": 0,
            "original_video_url": f"/static/video-analysis/{ANALYSIS_ID}/raw/original_video.mp4",
            "reference_frame_url": f"/static/video-analysis/{ANALYSIS_ID}/calibration/reference_frame.jpg",
            "calibration_status": "confirmed",
            "ball_detector_model_key": "e3_motion_blur",
            "ball_detector_model_name": "E3",
            "tracking_status": "tracking_complete",
            "message": "ready",
        },
    )
    _write_json(
        analysis_dir / "calibration" / "calibration.json",
        {
            "analysis_id": ANALYSIS_ID,
            "mode": "automatic_visual",
            "image_width": 1280,
            "image_height": 720,
        },
    )

    points = [(300, 250), (330, 252), (362, 255), (395, 258), (430, 261)]
    frames = []
    for frame in range(40):
        detections = []
        if 10 <= frame < 15:
            detections = [_candidate(frame, *points[frame - 10])]
        frames.append(
            {
                "frame_index": frame,
                "timestamp_seconds": frame / 30,
                "processed": True,
                "detections": detections,
            }
        )
    _write_json(
        analysis_dir / "detections" / "detections.json",
        {
            "analysis_id": ANALYSIS_ID,
            "detector": {
                "key": "e3_motion_blur",
                "name": "E3",
                "model_file": "e3.pt",
            },
            "model_path_used": "Models/e3.pt",
            "model_class_names": ["ball"],
            "settings": {
                "frame_stride": 1,
                "imgsz": 960,
                "confidence_threshold": 0.15,
                "max_det": 20,
            },
            "frames": frames,
        },
    )

    primary_track = []
    previous = None
    for offset, (x, y) in enumerate(points):
        frame = 10 + offset
        vx, vy = (0.0, 0.0) if previous is None else (x - previous[0], y - previous[1])
        previous = (x, y)
        primary_track.append(
            {
                "frame_index": frame,
                "timestamp_seconds": frame / 30,
                "source": "observed",
                "provenance": "OBSERVED",
                "candidate_id": f"frame_{frame:06d}_candidate_001",
                "x": x,
                "y": y,
                "normalized_x": x / 1280,
                "normalized_y": y / 720,
                "confidence": 0.82,
                "uncertainty": 0.05,
                "vx": vx / 1280,
                "vy": vy / 720,
                "inside_pitch_corridor": True,
            }
        )
    _write_json(
        analysis_dir / "tracking" / "tracking_result.json",
        {
            "analysis_id": ANALYSIS_ID,
            "status": "ready",
            "created_at": "2026-07-23T00:00:00Z",
            "completed_at": "2026-07-23T00:00:01Z",
            "settings": {
                "motion_model": "constant_velocity_recent_median",
                "max_recoverable_gap": 6,
                "minimum_observed_points": 3,
                "static_radius_normalized": 0.012,
                "base_gate_normalized": 0.025,
                "maximum_gate_normalized": 0.16,
                "history_points": 8,
                "beam_width": 4,
                "tracker_version": "delivery_track_v2",
            },
            "primary_track": primary_track,
            "raw_primary_track": primary_track,
            "candidate_diagnostics": [],
            "bounce": None,
            "message": "ready",
        },
    )


def test_release_point_api_start_job_result_lifecycle(tmp_path, monkeypatch):
    root = tmp_path / "video_analysis"
    _seed_analysis(root)
    monkeypatch.setattr(video_analysis_service, "VIDEO_ANALYSIS_ROOT", root)
    monkeypatch.setattr(video_release_point_service, "VIDEO_ANALYSIS_ROOT", root)
    monkeypatch.delenv(video_release_point_service.POSE_PROVIDER_ENV, raising=False)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    missing = client.get(f"/video-analysis/{ANALYSIS_ID}/release-point")
    assert missing.status_code == 404

    started = client.post(f"/video-analysis/{ANALYSIS_ID}/release-point/start")
    assert started.status_code == 202
    job_id = started.json()["job_id"]

    job = client.get(f"/video-analysis/{ANALYSIS_ID}/release-point/job/{job_id}")
    assert job.status_code == 200
    assert job.json()["status"] == "ready"

    result = client.get(f"/video-analysis/{ANALYSIS_ID}/release-point")
    assert result.status_code == 200
    body = result.json()
    assert body["result"]["evidence_mode"] == "fallback_trajectory_only"
    assert body["result"]["release_type"] == "INFERRED_RELEASE"
    assert body["result"]["provenance"]["ball_detector_model_key"] == "e3_motion_blur"
    assert "pose_not_run" in body["result"]["quality_flags"]
