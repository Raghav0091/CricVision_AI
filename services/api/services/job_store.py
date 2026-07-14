from threading import Lock
from uuid import uuid4

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


job_store = JobStore()
