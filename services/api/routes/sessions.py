from fastapi import APIRouter, HTTPException

from ..schemas.session import SessionCreate, SessionRecord
from ..services.session_store import session_store


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionRecord, status_code=201)
def create_session(request: SessionCreate) -> SessionRecord:
    return session_store.create(request)


@router.get("/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    record = session_store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return record
