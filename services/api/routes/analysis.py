from fastapi import APIRouter

from ..schemas.analysis import AnalysisJob, AnalysisRequest
from ..services.job_store import job_store


router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("", response_model=AnalysisJob, status_code=202)
def queue_analysis(request: AnalysisRequest) -> AnalysisJob:
    return job_store.create(request)
