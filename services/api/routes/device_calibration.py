"""Per-device lens calibration: solve once per phone, reuse everywhere.

The uploaded footage is deleted as soon as the profile exists. It is the
user's camera roll and it has no value once the intrinsics are solved.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from ..schemas.device_calibration import (
    CheckerboardSpec,
    DeviceCalibrationResponse,
    DeviceLensProfile,
)
from ..services.ball_detection_clip import transcode_browser_mp4
from ..services.device_calibration_service import (
    DeviceCalibrationError,
    calibrate_device_from_video,
    delete_device_profile,
    load_device_profile,
    save_device_profile,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/device-calibration", tags=["device-calibration"])


@router.post("/solve", response_model=DeviceCalibrationResponse)
def solve_device_calibration(
    video: Annotated[UploadFile, File()],
    device_id: Annotated[str, Form()],
    columns: Annotated[int, Form()] = 9,
    rows: Annotated[int, Form()] = 6,
    square_size_mm: Annotated[float, Form()] = 25.0,
    device_label: Annotated[str | None, Form()] = None,
) -> DeviceCalibrationResponse:
    try:
        spec = CheckerboardSpec(
            columns=columns,
            rows=rows,
            square_size_mm=square_size_mm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    suffix = Path(video.filename or "calibration.webm").suffix or ".webm"
    with tempfile.TemporaryDirectory(prefix="device_calibration_") as tmp_dir:
        raw_path = Path(tmp_dir) / f"raw{suffix}"
        with raw_path.open("wb") as raw_file:
            shutil.copyfileobj(video.file, raw_file)

        solve_path = raw_path
        if suffix.lower() != ".mp4":
            # Same MediaRecorder problem the Quick Test uploader hits: OpenCV
            # cannot reliably open a browser webm. An mp4 recording is already
            # readable, so only the webm case pays the transcode.
            transcoded_path = Path(tmp_dir) / "transcoded.mp4"
            try:
                transcode_browser_mp4(raw_path, transcoded_path, timeout_seconds=600)
                solve_path = transcoded_path
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not read the recorded clip: {type(exc).__name__}.",
                ) from exc

        try:
            profile = calibrate_device_from_video(
                solve_path,
                device_id,
                device_label,
                spec,
            )
        except DeviceCalibrationError as exc:
            logger.warning("Device calibration rejected for %s: %s", device_id, exc.message)
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.message,
            ) from exc
        except Exception as exc:
            # Anything unexpected from OpenCV or Pydantic would otherwise surface
            # as a bare 500 with the reason only visible in the server console.
            logger.exception("Device calibration failed for %s", device_id)
            raise HTTPException(
                status_code=500,
                detail=f"Calibration failed while solving: {type(exc).__name__}: {exc}",
            ) from exc

    try:
        save_device_profile(profile)
    except DeviceCalibrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except OSError as exc:
        logger.exception("Could not persist device profile for %s", device_id)
        raise HTTPException(
            status_code=500,
            detail=f"Solved the lens but could not save it: {exc}",
        ) from exc
    logger.info(
        "Calibrated device %s: fx=%.1f rms=%.3f views=%d",
        device_id,
        profile.focal_length_x_px,
        profile.quality.rms_reprojection_px,
        profile.quality.views_used,
    )
    return DeviceCalibrationResponse(
        success=True,
        status="CALIBRATED",
        profile=profile,
        message=profile.quality.advice,
    )


@router.get("/{device_id}", response_model=DeviceLensProfile)
def get_device_calibration(device_id: str) -> DeviceLensProfile:
    try:
        profile = load_device_profile(device_id)
    except DeviceCalibrationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    if profile is None:
        raise HTTPException(
            status_code=404,
            detail="This device has not been calibrated.",
        )
    return profile


@router.delete("/{device_id}", status_code=204)
def delete_device_calibration(device_id: str) -> Response:
    try:
        delete_device_profile(device_id)
    except DeviceCalibrationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc
    return Response(status_code=204)
