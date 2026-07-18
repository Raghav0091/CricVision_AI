import logging
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..schemas.video_analysis import VideoAnalysisPreparedResponse
from ..services.video_analysis_service import (
    VideoAnalysisServiceError,
    load_video_analysis,
    prepare_video,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])


@router.post(
    "/prepare",
    response_model=VideoAnalysisPreparedResponse,
    status_code=201,
)
def prepare_video_analysis(
    video: Annotated[UploadFile, File()],
) -> VideoAnalysisPreparedResponse:
    try:
        record = prepare_video(video.file, video.filename)
        logger.info(
            "Prepared video analysis %s from %s",
            record.analysis_id,
            record.original_filename,
        )
        return record
    except VideoAnalysisServiceError as exc:
        logger.warning("Video analysis preparation rejected: %s", exc.message)
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}",
    response_model=VideoAnalysisPreparedResponse,
)
def get_video_analysis(
    analysis_id: str,
) -> VideoAnalysisPreparedResponse:
    try:
        return load_video_analysis(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
