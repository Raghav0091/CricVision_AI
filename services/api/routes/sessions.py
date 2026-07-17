from fastapi import APIRouter, HTTPException

from ..schemas.session import SessionCreate, SessionRecord
from ..services.session_store import session_store
from ..services.job_store import clip_job_store


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionRecord, status_code=201)
def create_session(request: SessionCreate) -> SessionRecord:
    return session_store.create(request)


@router.get("", response_model=list[SessionRecord])
def list_sessions() -> list[SessionRecord]:
    records = []
    for record in session_store.list():
        synchronized = session_store.synchronize(record.id, clip_job_store.get)
        if synchronized is not None:
            records.append(synchronized)
    return records


@router.get("/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    record = session_store.synchronize(session_id, clip_job_store.get)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return record


@router.post("/{session_id}/capture-complete", response_model=SessionRecord)
def complete_session_capture(session_id: str) -> SessionRecord:
    record = session_store.complete_capture(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return record
