from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    delivery_id: str = Field(min_length=1)


class AnalysisJob(AnalysisRequest):
    id: str
    created_at: datetime
    status: Literal["queued"] = "queued"
    message: str = "Analysis worker integration is not available yet."
