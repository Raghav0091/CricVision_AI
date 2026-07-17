from threading import Lock
from uuid import uuid4
from typing import Any

from ..schemas.analysis import AnalysisJob, AnalysisRequest
from ..schemas.session import utc_now


class JobStore:
    def __init__(self) -> None:
        self._records: dict[str, AnalysisJob] = {}
        self._lock = Lock()

    def create(self, request: AnalysisRequest) -> AnalysisJob:
        record = AnalysisJob(id=str(uuid4()), created_at=utc_now(), **request.model_dump())
        with self._lock:
            self._records[record.id] = record
        return record


class ClipJobStore:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(self, delivery_index: int, session_id: str | None = None) -> dict[str, Any]:
        job_id = str(uuid4())
        record = {
            "success": True,
            "status": "processing",
            "job_id": job_id,
            "progress": 0,
            "delivery_index": delivery_index,
            "session_id": session_id,
            "message": "Ball detection job started.",
        }
        with self._lock:
            self._records[job_id] = record
        return dict(record)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(job_id)
            return dict(record) if record else None

    def update(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            if job_id in self._records:
                self._records[job_id].update(updates)


job_store = JobStore()
clip_job_store = ClipJobStore()
