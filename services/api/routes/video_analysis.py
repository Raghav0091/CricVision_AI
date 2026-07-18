import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ..schemas.video_analysis import (
    ConfirmedVideoCalibrationResponse,
    VideoAnalysisPreparedResponse,
    VideoBallDetectionJobResponse,
    VideoBallDetectionResultResponse,
    VideoBallDetectionStartResponse,
    VideoBallTrackingJobResponse,
    VideoBallTrackingResultResponse,
    VideoBallTrackingStartResponse,
    VideoCalibrationConfirmationRequest,
    VideoCalibrationDetectionResponse,
)
from ..services.video_analysis_service import (
    VideoAnalysisServiceError,
    load_video_analysis,
    prepare_video,
)
from ..services.video_ball_detection_job_store import (
    video_ball_detection_job_store,
)
from ..services.video_ball_detection_service import (
    VideoBallDetectionError,
    load_video_ball_detection_result,
    mark_video_ball_detection_queued,
    run_video_ball_detection_job,
)
from ..services.video_ball_tracking_job_store import (
    video_ball_tracking_job_store,
)
from ..services.video_ball_tracking_service import (
    VideoBallTrackingError,
    load_video_ball_tracking_result,
    mark_video_ball_tracking_queued,
    run_video_ball_tracking_job,
    validate_video_ball_tracking_input,
)
from ..services.video_calibration_service import (
    confirm_video_calibration,
    detect_video_calibration,
    load_video_calibration,
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


@router.post(
    "/{analysis_id}/ball-detection/start",
    response_model=VideoBallDetectionStartResponse,
    status_code=202,
)
def start_analysis_ball_detection(
    analysis_id: str,
    background_tasks: BackgroundTasks,
) -> VideoBallDetectionStartResponse:
    job = None
    try:
        analysis = load_video_analysis(analysis_id)
        job = video_ball_detection_job_store.create(
            analysis_id,
            analysis.frame_count,
        )
        if job is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An active every-frame ball-detection job already exists "
                    "for this analysis."
                ),
            )
        mark_video_ball_detection_queued(analysis_id, job["job_id"])
        background_tasks.add_task(
            run_video_ball_detection_job,
            analysis_id,
            job["job_id"],
        )
        logger.info(
            "Queued every-frame ball detection %s for %s",
            job["job_id"],
            analysis_id,
        )
        return VideoBallDetectionStartResponse.model_validate(job)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except VideoBallDetectionError as exc:
        if job is not None:
            video_ball_detection_job_store.update(
                job["job_id"],
                success=False,
                status="failed",
                error_message=exc.message,
                message=exc.message,
            )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/ball-detection/job/{job_id}",
    response_model=VideoBallDetectionJobResponse,
)
def get_analysis_ball_detection_job(
    analysis_id: str,
    job_id: str,
) -> VideoBallDetectionJobResponse:
    job = video_ball_detection_job_store.get(job_id)
    if job is None or job["analysis_id"] != analysis_id:
        raise HTTPException(
            status_code=404,
            detail="Every-frame ball-detection job not found.",
        )
    return VideoBallDetectionJobResponse.model_validate(job)


@router.get(
    "/{analysis_id}/ball-detection",
    response_model=VideoBallDetectionResultResponse,
)
def get_analysis_ball_detection(
    analysis_id: str,
) -> VideoBallDetectionResultResponse:
    try:
        return load_video_ball_detection_result(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    except VideoBallDetectionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/tracking/start",
    response_model=VideoBallTrackingStartResponse,
    status_code=202,
)
def start_analysis_ball_tracking(
    analysis_id: str,
    background_tasks: BackgroundTasks,
) -> VideoBallTrackingStartResponse:
    job = None
    try:
        validate_video_ball_tracking_input(analysis_id)
        job = video_ball_tracking_job_store.create(analysis_id)
        if job is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An active Moving Ball Tracker job already exists "
                    "for this analysis."
                ),
            )
        mark_video_ball_tracking_queued(analysis_id, job["job_id"])
        background_tasks.add_task(
            run_video_ball_tracking_job,
            analysis_id,
            job["job_id"],
        )
        logger.info(
            "Queued Moving Ball Tracker job %s for %s",
            job["job_id"],
            analysis_id,
        )
        return VideoBallTrackingStartResponse.model_validate(job)
    except (VideoAnalysisServiceError, VideoBallTrackingError) as exc:
        if job is not None:
            video_ball_tracking_job_store.update(
                job["job_id"],
                success=False,
                status="failed",
                error_message=exc.message,
                message=exc.message,
            )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/tracking/job/{job_id}",
    response_model=VideoBallTrackingJobResponse,
)
def get_analysis_ball_tracking_job(
    analysis_id: str,
    job_id: str,
) -> VideoBallTrackingJobResponse:
    job = video_ball_tracking_job_store.get(job_id)
    if job is None or job["analysis_id"] != analysis_id:
        raise HTTPException(
            status_code=404,
            detail="Moving Ball Tracker job not found.",
        )
    return VideoBallTrackingJobResponse.model_validate(job)


@router.get(
    "/{analysis_id}/tracking",
    response_model=VideoBallTrackingResultResponse,
)
def get_analysis_ball_tracking(
    analysis_id: str,
) -> VideoBallTrackingResultResponse:
    try:
        return load_video_ball_tracking_result(analysis_id)
    except (VideoAnalysisServiceError, VideoBallTrackingError) as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/calibration/detect",
    response_model=VideoCalibrationDetectionResponse,
)
def detect_analysis_calibration(
    analysis_id: str,
) -> VideoCalibrationDetectionResponse:
    try:
        return detect_video_calibration(analysis_id)
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Video calibration detection rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.put(
    "/{analysis_id}/calibration/confirm",
    response_model=ConfirmedVideoCalibrationResponse,
)
def confirm_analysis_calibration(
    analysis_id: str,
    request: VideoCalibrationConfirmationRequest,
) -> ConfirmedVideoCalibrationResponse:
    try:
        record = confirm_video_calibration(analysis_id, request)
        logger.info("Confirmed scene calibration for %s", analysis_id)
        return record
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Video calibration confirmation rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/calibration",
    response_model=ConfirmedVideoCalibrationResponse,
)
def get_analysis_calibration(
    analysis_id: str,
) -> ConfirmedVideoCalibrationResponse:
    try:
        return load_video_calibration(analysis_id)
    except VideoAnalysisServiceError as exc:
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
