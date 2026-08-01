import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, File, HTTPException, Response, UploadFile

from ..schemas.camera_bridge import CameraBridgeResponse
from ..schemas.preset_auto_registration import (
    CameraSetupPresetListResponse,
    PresetAutoRegistrationResult,
    PresetAutoRegistrationRunRequest,
    list_camera_setup_presets,
)
from ..schemas.video_analysis import (
    BallDetectorModelOption,
    BallDetectorModelsResponse,
    CalibrationV2ConfirmRequest,
    CalibrationV2InitialiseResponse,
    CalibrationV2Result,
    ConfirmedVideoCalibrationResponse,
    VideoAnalysisPreparedResponse,
    VideoBallDetectionJobResponse,
    VideoBallDetectionResultResponse,
    VideoBallDetectionStartRequest,
    VideoBallDetectionStartResponse,
    VideoBallTrackingJobResponse,
    VideoBallTrackingResultResponse,
    VideoBallTrackingStartResponse,
    VideoCalibrationConfirmationRequest,
    VideoCalibrationDetectionRequest,
    VideoCalibrationDetectionResponse,
    WicketCameraPoseInitialiseResponse,
    WicketCameraPoseResult,
    WicketCameraPoseSolveRequest,
)
from ..schemas.virtual_pitch import (
    SyntheticPitchPreviewResponse,
    VirtualPitchSpecification,
)
from ..schemas.real_pitch_registration import RealPitchRegistrationResult
from ..schemas.scene_calibration import (
    SceneCalibrationActionRequest,
    SceneCalibrationAnchorUpdateRequest,
    SceneCalibrationOrientationRequest,
    SceneCalibrationPresetRequest,
    SceneCalibrationPresetResponse,
    SceneCalibrationRefineRequest,
    SceneCalibrationResult,
)
from ..schemas.wicket_observation import WicketObservationResult
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
from ..services.video_calibration_service import (
    confirm_video_calibration,
    detect_video_calibration,
    load_video_calibration,
)
from ..services.video_calibration_v2_service import (
    confirm_video_calibration_v2,
    initialise_video_calibration_v2,
    load_video_calibration_v2,
)
from ..services.video_camera_pose_service import (
    initialise_wicket_camera_pose,
    load_wicket_camera_pose,
    solve_wicket_camera_pose,
)
from ..services.camera_bridge_service import (
    build_synthetic_camera_bridge,
    load_analysis_camera_bridge,
)
from ..services.real_pitch_registration_service import (
    load_real_pitch_registration,
    run_real_pitch_registration,
)
from ..services.preset_auto_registration import (
    clear_preset_auto_registration,
    load_preset_auto_registration,
    run_preset_auto_registration,
)
from ..services.scene_calibration_service import (
    accept_scene_calibration,
    apply_scene_calibration_orientation,
    apply_scene_calibration_preset,
    clear_scene_calibration_orientation,
    load_scene_calibration,
    load_orientation_presets_for_analysis,
    refine_scene_calibration,
    reject_scene_calibration,
    run_scene_calibration,
    update_scene_calibration_anchors,
    use_visual_overlay_only,
)
from ..services.virtual_pitch_service import (
    build_synthetic_preview,
    build_virtual_pitch_specification,
)
from ..services.wicket_observation_service import (
    load_wicket_observation,
    run_wicket_observation,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/video-analysis", tags=["video-analysis"])


@router.get(
    "/scene-calibration/presets",
    response_model=CameraSetupPresetListResponse,
)
def get_scene_calibration_presets() -> CameraSetupPresetListResponse:
    return CameraSetupPresetListResponse(presets=list_camera_setup_presets())


@router.post(
    "/{analysis_id}/scene-calibration/auto-register",
    response_model=PresetAutoRegistrationResult,
)
def run_analysis_preset_auto_registration(
    analysis_id: str,
    request: PresetAutoRegistrationRunRequest,
) -> PresetAutoRegistrationResult:
    try:
        return run_preset_auto_registration(
            analysis_id,
            preset_id=request.preset_id,
            reuse_existing_observations=request.reuse_existing_observations,
            force_redetect=request.force_redetect,
            development_diagnostics=request.development_diagnostics,
        )
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/scene-calibration/auto-registration",
    response_model=PresetAutoRegistrationResult,
)
def get_analysis_preset_auto_registration(
    analysis_id: str,
) -> PresetAutoRegistrationResult:
    try:
        return load_preset_auto_registration(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/auto-registration/clear",
    status_code=204,
)
def clear_analysis_preset_auto_registration(analysis_id: str) -> Response:
    try:
        clear_preset_auto_registration(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    return Response(status_code=204)


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


@router.post(
    "/{analysis_id}/wicket-observations/run",
    response_model=WicketObservationResult,
)
def run_analysis_wicket_observations(
    analysis_id: str,
) -> WicketObservationResult:
    """Run bounded real-frame observation without camera registration."""
    try:
        return run_wicket_observation(analysis_id)
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Wicket observation rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/wicket-observations",
    response_model=WicketObservationResult,
)
def get_analysis_wicket_observations(
    analysis_id: str,
) -> WicketObservationResult:
    try:
        return load_wicket_observation(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/pitch-registration/run",
    response_model=RealPitchRegistrationResult,
)
def run_analysis_pitch_registration(
    analysis_id: str,
) -> RealPitchRegistrationResult:
    try:
        return run_real_pitch_registration(analysis_id)
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Real pitch registration rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/pitch-registration",
    response_model=RealPitchRegistrationResult,
)
def get_analysis_pitch_registration(
    analysis_id: str,
) -> RealPitchRegistrationResult:
    try:
        return load_real_pitch_registration(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/run",
    response_model=SceneCalibrationResult,
)
def run_analysis_scene_calibration(
    analysis_id: str,
) -> SceneCalibrationResult:
    try:
        return run_scene_calibration(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/scene-calibration",
    response_model=SceneCalibrationResult,
)
def get_analysis_scene_calibration(
    analysis_id: str,
) -> SceneCalibrationResult:
    try:
        return load_scene_calibration(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/anchors",
    response_model=SceneCalibrationResult,
)
def update_analysis_scene_calibration_anchors(
    analysis_id: str,
    request: SceneCalibrationAnchorUpdateRequest,
) -> SceneCalibrationResult:
    try:
        return update_scene_calibration_anchors(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/refine",
    response_model=SceneCalibrationResult,
)
def refine_analysis_scene_calibration(
    analysis_id: str,
    request: SceneCalibrationRefineRequest,
) -> SceneCalibrationResult:
    try:
        return refine_scene_calibration(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/orientation",
    response_model=SceneCalibrationResult,
)
def confirm_analysis_scene_calibration_orientation(
    analysis_id: str,
    request: SceneCalibrationOrientationRequest,
) -> SceneCalibrationResult:
    try:
        return apply_scene_calibration_orientation(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/scene-calibration/preset",
    response_model=SceneCalibrationPresetResponse,
)
def get_analysis_scene_calibration_presets(
    analysis_id: str,
) -> SceneCalibrationPresetResponse:
    try:
        return load_orientation_presets_for_analysis(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/preset",
    response_model=SceneCalibrationResult,
)
def use_analysis_scene_calibration_preset(
    analysis_id: str,
    request: SceneCalibrationPresetRequest,
) -> SceneCalibrationResult:
    try:
        return apply_scene_calibration_preset(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/orientation/clear",
    response_model=SceneCalibrationResult,
)
def clear_analysis_scene_calibration_orientation(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    try:
        return clear_scene_calibration_orientation(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/accept",
    response_model=SceneCalibrationResult,
)
def accept_analysis_scene_calibration(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    try:
        return accept_scene_calibration(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/reject",
    response_model=SceneCalibrationResult,
)
def reject_analysis_scene_calibration(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    try:
        return reject_scene_calibration(analysis_id, request)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/scene-calibration/use-visual-only",
    response_model=SceneCalibrationResult,
)
def use_analysis_visual_calibration_only(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    try:
        return use_visual_overlay_only(analysis_id, request)
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
    "/{analysis_id}/calibration/v2/initialise",
    response_model=CalibrationV2InitialiseResponse,
)
def initialise_analysis_calibration_v2(
    analysis_id: str,
) -> CalibrationV2InitialiseResponse:
    try:
        return initialise_video_calibration_v2(analysis_id)
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Calibration v2 initialisation rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.put(
    "/{analysis_id}/calibration/v2/confirm",
    response_model=CalibrationV2Result,
)
def confirm_analysis_calibration_v2(
    analysis_id: str,
    request: CalibrationV2ConfirmRequest,
) -> CalibrationV2Result:
    try:
        result = confirm_video_calibration_v2(analysis_id, request)
        logger.info(
            "Saved Calibration v2 for %s with status %s",
            analysis_id,
            result.status,
        )
        return result
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Calibration v2 confirmation rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/calibration/v2",
    response_model=CalibrationV2Result,
)
def get_analysis_calibration_v2(
    analysis_id: str,
) -> CalibrationV2Result:
    try:
        return load_video_calibration_v2(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.post(
    "/{analysis_id}/calibration/v2/camera-pose/initialise",
    response_model=WicketCameraPoseInitialiseResponse,
)
def initialise_analysis_camera_pose(
    analysis_id: str,
) -> WicketCameraPoseInitialiseResponse:
    try:
        return initialise_wicket_camera_pose(analysis_id)
    except VideoAnalysisServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.put(
    "/{analysis_id}/calibration/v2/camera-pose/solve",
    response_model=WicketCameraPoseResult,
)
def solve_analysis_camera_pose(
    analysis_id: str,
    request: WicketCameraPoseSolveRequest,
) -> WicketCameraPoseResult:
    try:
        result = solve_wicket_camera_pose(analysis_id, request)
        logger.info(
            "Saved wicket-based camera pose for %s with status %s",
            analysis_id,
            result.status,
        )
        return result
    except VideoAnalysisServiceError as exc:
        logger.warning(
            "Camera-pose solve rejected for %s: %s",
            analysis_id,
            exc.message,
        )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc


@router.get(
    "/{analysis_id}/calibration/v2/camera-pose",
    response_model=WicketCameraPoseResult,
)
def get_analysis_camera_pose(
    analysis_id: str,
) -> WicketCameraPoseResult:
    try:
        return load_wicket_camera_pose(analysis_id)
    except VideoAnalysisServiceError as exc:
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
    refresh_early_reference: bool = False,
    body: Annotated[
        VideoCalibrationDetectionRequest | None, Body()
    ] = None,
) -> VideoCalibrationDetectionResponse:
    try:
        return detect_video_calibration(
            analysis_id,
            refresh_early_reference=refresh_early_reference,
            striker_guide=None if body is None else body.striker_guide,
            non_striker_guide=(
                None if body is None else body.non_striker_guide
            ),
        )
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
