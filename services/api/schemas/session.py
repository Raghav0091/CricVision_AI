from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class SessionCreate(BaseModel):
    name: str = Field(default="Live bowling session", min_length=1, max_length=120)
    source: Literal["live", "upload"] = "live"


class SessionRecord(SessionCreate):
    id: str
    created_at: datetime
    status: Literal["created", "capturing", "complete"] = "created"
    delivery_ids: list[str] = Field(default_factory=list)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
