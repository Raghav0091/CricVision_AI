from fastapi import APIRouter

from ..schemas.calibration import CalibrationRequest, CalibrationResponse
from ..services.calibration_store import save_calibration_frame


router = APIRouter(prefix="/calibration", tags=["calibration"])


@router.post("/solve", response_model=CalibrationResponse)
def solve_calibration(request: CalibrationRequest) -> CalibrationResponse:
    """Persist the frame, then fail safely until a dedicated detector exists."""
    try:
        frame_path = save_calibration_frame(request.frame_data_url)
    except ValueError as exc:
        return CalibrationResponse(
            success=False,
            quality="Unavailable",
            reason="invalid_calibration_frame",
            message=str(exc),
        )

    return CalibrationResponse(
        success=False,
        quality="Unavailable",
        reason="stump_detector_missing",
        message="Dedicated stump detector not available yet.",
        calibration_frame_path=str(frame_path),
    )
