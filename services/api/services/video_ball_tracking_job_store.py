"""Process-local background-job state for Video Analysis ball tracking."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any
from uuid import uuid4

from .video_ball_detection_job_store import utc_now


ACTIVE_TRACKING_STATUSES = {
    "queued",
    "loading_detections",
    "analysing_candidates",
    "building_track",
    "recovering_gaps",
    "fitting_physics",
    "rendering_video",
    "saving_results",
}


class VideoBallTrackingJobStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock:
            if any(
                record["analysis_id"] == analysis_id
                and record["status"] in ACTIVE_TRACKING_STATUSES
                for record in self._records.values()
            ):
                return None
            now = utc_now()
            stamp = now.strftime("%Y%m%d%H%M%S")
            job_id = f"tracking_job_{stamp}_{uuid4().hex[:8]}"
            record = {
                "success": True,
                "status": "queued",
                "analysis_id": analysis_id,
                "job_id": job_id,
                "progress": 0,
                "created_at": now,
                "updated_at": now,
                "error_message": None,
                "failure_code": None,
                "result": None,
                "message": "Moving-ball tracking queued.",
            }
            self._records[job_id] = record
            return deepcopy(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            return deepcopy(record) if record is not None else None

    def update(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            record.update(updates)
            record["updated_at"] = utc_now()
            return deepcopy(record)


video_ball_tracking_job_store = VideoBallTrackingJobStore()
