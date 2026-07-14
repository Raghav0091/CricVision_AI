from threading import Lock
from uuid import uuid4

from ..schemas.session import SessionCreate, SessionRecord, utc_now


class SessionStore:
    """Process-local store for the architecture scaffold."""

    def __init__(self) -> None:
        self._records: dict[str, SessionRecord] = {}
        self._lock = Lock()

    def create(self, request: SessionCreate) -> SessionRecord:
        record = SessionRecord(id=str(uuid4()), created_at=utc_now(), **request.model_dump())
        with self._lock:
            self._records[record.id] = record
        return record

    def get(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)


session_store = SessionStore()
