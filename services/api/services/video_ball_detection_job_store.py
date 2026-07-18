"""Process-local job state for persistent Video Analysis ball detections."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4


ACTIVE_STATUSES = {
    "queued",
    "loading_model",
    "processing",
    "writing_video",
    "saving_results",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class VideoBallDetectionJobStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(
        self,
        analysis_id: str,
        total_frames: int,
    ) -> dict[str, Any] | None:
        with self._lock:
            if any(
                record["analysis_id"] == analysis_id
                and record["status"] in ACTIVE_STATUSES
                for record in self._records.values()
            ):
                return None
            now = utc_now()
            stamp = now.strftime("%Y%m%d%H%M%S")
            job_id = f"video_ball_job_{stamp}_{uuid4().hex[:8]}"
            record = {
                "success": True,
                "status": "queued",
                "analysis_id": analysis_id,
                "job_id": job_id,
                "progress": 0,
                "current_frame": 0,
                "total_frames": total_frames,
                "created_at": now,
                "updated_at": now,
                "model_path_used": None,
                "error_message": None,
                "result": None,
                "message": "Every-frame ball detection queued.",
            }
            self._records[job_id] = record
            return deepcopy(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            return deepcopy(record) if record is not None else None

    def find_active(self, analysis_id: str) -> dict[str, Any] | None:
        with self._lock:
            for record in self._records.values():
                if (
                    record["analysis_id"] == analysis_id
                    and record["status"] in ACTIVE_STATUSES
                ):
                    return deepcopy(record)
        return None

    def update(self, job_id: str, **updates: Any) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            if record is None:
                return None
            record.update(updates)
            record["updated_at"] = utc_now()
            return deepcopy(record)


video_ball_detection_job_store = VideoBallDetectionJobStore()
