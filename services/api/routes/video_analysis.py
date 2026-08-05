import logging
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, Body, File, HTTPException, Response, UploadFile

from ..schemas.camera_bridge import CameraBridgeResponse
from ..schemas.video_analysis import (
    BallDetectorModelOption,
    BallDetectorModelsResponse,
    VideoAnalysisPreparedResponse,
    VideoBallDetectionJobResponse,
    VideoBallDetectionResultResponse,
    VideoBallDetectionStartRequest,
    VideoBallDetectionStartResponse,
    VideoBallTrackingJobResponse,
    VideoBallTrackingResultResponse,
    VideoBallTrackingStartResponse,
)
from ..schemas.virtual_pitch import (
    SyntheticPitchPreviewResponse,
    VirtualPitchSpecification,
)
from ..schemas.replay_payload import ReplayPayloadV1
from ..schemas.wicket_box_calibration import (
    CalibrationResult,
    WicketBoxCalibrationAcceptRequest,
    WicketBoxCalibrationAcceptResponse,
    WicketBoxCalibrationDetectResponse,
    WicketBoxCalibrationRegisterRequest,
    WicketBoxCalibrationRegisterResponse,
)
from ..services.ball_detector_registry import (
    BallDetectorModelMissing,
    list_ball_detector_models,
    resolve_ball_detector_model,
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
from ..services.camera_bridge_service import (
    build_synthetic_camera_bridge,
    load_analysis_camera_bridge,
)
from ..services.virtual_pitch_service import (
    build_synthetic_preview,
    build_virtual_pitch_specification,
)
from ..services.replay_payload_service import load_replay_payload
from ..services.wicket_box_calibration_service import (
    accept_wicket_box_calibration,
    detect_wicket_box_calibration,
    load_wicket_box_calibration,
    register_wicket_box_calibration,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])


@router.get(
    "/virtual-pitch",
    response_model=VirtualPitchSpecification,
)
def get_virtual_pitch_specification() -> VirtualPitchSpecification:
    return build_virtual_pitch_specification()


@router.get(
    "/virtual-pitch/synthetic-projection",
    response_model=SyntheticPitchPreviewResponse,
)
def get_synthetic_virtual_pitch_projection(
    camera_name: str = "centred_bowler_end",
    profile: str = "analytical",
) -> SyntheticPitchPreviewResponse:
    try:
        return build_synthetic_preview(camera_name, profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/virtual-pitch/camera-bridge",
    response_model=CameraBridgeResponse,
)
def get_synthetic_camera_bridge(
    camera_name: str = "centred_bowler_end",
) -> CameraBridgeResponse:
    try:
        return build_synthetic_camera_bridge(camera_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/{analysis_id}/camera-bridge",
    response_model=CameraBridgeResponse,
)
def get_analysis_camera_bridge(
    analysis_id: str,
) -> CameraBridgeResponse:
    try:
        return load_analysis_camera_bridge(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/wicket-box-calibration",
    response_model=CalibrationResult,
)
def get_analysis_wicket_box_calibration(
    analysis_id: str,
) -> CalibrationResult:
    """Return persisted wicket-box calibration for session restore."""
    try:
        return load_wicket_box_calibration(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/wicket-box-calibration/detect",
    response_model=WicketBoxCalibrationDetectResponse,
)
def detect_analysis_wicket_box_calibration(
    analysis_id: str,
    request: WicketBoxCalibrationRegisterRequest,
) -> WicketBoxCalibrationDetectResponse:
    """Detect stump landmarks inside submitted NEAR/FAR wicket boxes."""
    try:
        return detect_wicket_box_calibration(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/wicket-box-calibration/register",
    response_model=WicketBoxCalibrationRegisterResponse,
)
def register_analysis_wicket_box_calibration(
    analysis_id: str,
    request: WicketBoxCalibrationRegisterRequest,
    assignment_hypothesis: Literal["A", "B"] | None = None,
) -> WicketBoxCalibrationRegisterResponse:
    """Register camera pose from wicket-box landmarks via PnP."""
    try:
        return register_wicket_box_calibration(
            analysis_id,
            request,
            assignment_hypothesis=assignment_hypothesis,
        )
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/wicket-box-calibration/accept",
    response_model=WicketBoxCalibrationAcceptResponse,
)
def accept_analysis_wicket_box_calibration(
    analysis_id: str,
    request: WicketBoxCalibrationAcceptRequest,
    assignment_hypothesis: Literal["A", "B"] | None = None,
) -> WicketBoxCalibrationAcceptResponse:
    """Accept registered wicket-box calibration and freeze pose."""
    try:
        return accept_wicket_box_calibration(
            analysis_id,
            request,
            assignment_hypothesis=assignment_hypothesis,
        )
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/replay-payload",
    response_model=ReplayPayloadV1,
)
def get_analysis_replay_payload(
    analysis_id: str,
    response: Response,
) -> ReplayPayloadV1:
    """Virtual Pitch Replay payload assembled from tracking and physics."""
    response.headers["Cache-Control"] = "no-store"
    try:
        return load_replay_payload(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/detector-models",
    response_model=BallDetectorModelsResponse,
)
def get_detector_models() -> BallDetectorModelsResponse:
    return BallDetectorModelsResponse(
        models=[
            BallDetectorModelOption(
                key=model.key,
                display_name=model.display_name,
                description=model.description,
                available=model.available,
            )
            for model in list_ball_detector_models()
        ]
    )


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
    request: VideoBallDetectionStartRequest = Body(
        default=VideoBallDetectionStartRequest()
    ),
) -> VideoBallDetectionStartResponse:
    job = None
    try:
        analysis = load_video_analysis(analysis_id)
        selected_model = resolve_ball_detector_model(
            request.ball_detector_model_key
        )
        job = video_ball_detection_job_store.create(
            analysis_id,
            analysis.frame_count,
            selected_model.model_key,
            selected_model.display_name,
        )
        if job is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "An active every-frame ball-detection job already exists "
                    "for this analysis."
                ),
            )
        mark_video_ball_detection_queued(
            analysis_id,
            job["job_id"],
            selected_model.model_key,
            selected_model.display_name,
        )
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
    except BallDetectorModelMissing as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    include_frames: bool = False,
) -> VideoBallDetectionResultResponse:
    try:
        return load_video_ball_detection_result(
            analysis_id,
            include_frames=include_frames,
        )
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
