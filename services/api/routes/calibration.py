from fastapi import APIRouter

from ..schemas.calibration import CalibrationRequest, CalibrationResponse
from ..services.calibration_store import save_calibration_frame
from ..services.stump_detector_service import (
    STUMP_MODEL_PATH,
    save_debug_overlay,
    solve_stump_calibration,
)


router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("/solve", response_model=CalibrationResponse)
def solve_calibration(request: CalibrationRequest) -> CalibrationResponse:
    """Persist one frame and solve both user-aligned stump regions."""
    try:
        frame_path, image = save_calibration_frame(request.frame_data_url)
    except ValueError as exc:
        return CalibrationResponse(
            success=False,
            status="invalid_calibration_frame",
            quality="Unavailable",
            reason="invalid_calibration_frame",
            message=str(exc),
        )

    box_layout = request.box_layout.model_dump()
    result = solve_stump_calibration(
        image,
        frame_width=request.frame_width,
        frame_height=request.frame_height,
        box_layout=box_layout,
    )
    debug_path = frame_path.with_name(f"{frame_path.stem}_debug.jpg")
    try:
        save_debug_overlay(
            image,
            box_layout=box_layout,
            detections=result.get("detections"),
            virtual_stumps=result.get("virtual_stumps"),
            pitch_overlay=result.get("pitch_overlay"),
            output_path=debug_path,
        )
        overlay_filename = debug_path.name
    except Exception:
        overlay_filename = None

    status = result["status"]
    quality = (
        "Good"
        if status == "setup_complete"
        else "Poor"
        if status == "stumps_not_found"
        else "Unavailable"
    )
    return CalibrationResponse(
        success=result["success"],
        status=status,
        quality=quality,
        reason=status,
        message=result["message"],
        calibration_frame_path=str(frame_path),
        model_path=str(STUMP_MODEL_PATH),
        detections=result.get("detections"),
        virtual_stumps=result.get("virtual_stumps"),
        pitch_overlay=result.get("pitch_overlay"),
        calibration_quality=result.get("calibration_quality"),
        environment_context=result.get("environment_context"),
        debug_files={
            "original": frame_path.name,
            "overlay": overlay_filename,
        },
    )
