from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DeliveryCreate(BaseModel):
    session_id: str = Field(min_length=1)
    clip_path: str = Field(min_length=1)


class DeliveryRecord(DeliveryCreate):
    id: str
    created_at: datetime
    status: Literal["registered", "queued", "analysing", "complete", "failed"] = "registered"
