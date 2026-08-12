"""Solve and store a phone's lens intrinsics from checkerboard footage."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Iterator

import cv2
import numpy as np

from ..schemas.device_calibration import (
    MAX_PLAUSIBLE_DIAGONAL_FOV_DEG,
    MIN_PLAUSIBLE_DIAGONAL_FOV_DEG,
    MIN_VIEWS,
    RECOMMENDED_VIEWS,
    CalibrationQuality,
    CheckerboardSpec,
    DeviceLensProfile,
    quality_band,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVICE_CALIBRATION_ROOT = PROJECT_ROOT / "outputs" / "device_calibration"

# Consecutive video frames are near-duplicates and add no new constraints, so
# frames are sampled apart. A 30s clip at 30fps yields ~60 candidate views.
FRAME_SAMPLE_STRIDE = 15
MAX_CANDIDATE_FRAMES = 120

# Views whose corners are nearly identical to one already accepted contribute
# nothing but do bias the solve toward whatever pose was held longest. Measured
# as mean per-corner displacement, so this is a real on-screen distance.
MIN_VIEW_SEPARATION_PX = 20.0

_CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
_FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)


class DeviceCalibrationError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _safe_device_id(device_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", device_id).strip("._")
    if not cleaned:
        raise DeviceCalibrationError("device_id is not a usable identifier.")
    return cleaned


def _profile_path(device_id: str) -> Path:
    return DEVICE_CALIBRATION_ROOT / f"{_safe_device_id(device_id)}.json"


def _object_points(spec: CheckerboardSpec) -> np.ndarray:
    """Corner positions on the board, in millimetres, z=0 on the board plane."""
    grid = np.zeros((spec.rows * spec.columns, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0 : spec.columns, 0 : spec.rows].T.reshape(-1, 2)
    return grid * float(spec.square_size_mm)


def _iter_frames(video_path: Path) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise DeviceCalibrationError("The calibration video could not be opened.")
    try:
        index = 0
        emitted = 0
        while emitted < MAX_CANDIDATE_FRAMES:
            ok, frame = capture.read()
            if not ok:
                break
            if index % FRAME_SAMPLE_STRIDE == 0:
                emitted += 1
                yield frame
            index += 1
    finally:
        capture.release()


def _detect_corners(
    frame: np.ndarray,
    spec: CheckerboardSpec,
) -> np.ndarray | None:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        grey,
        (spec.columns, spec.rows),
        flags=_FIND_FLAGS,
    )
    if not found:
        return None
    # Sub-pixel refinement is what takes corner precision from ~1px to ~0.1px,
    # and corner precision is the ceiling on the whole calibration.
    return cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), _CORNER_CRITERIA)


def _is_distinct(candidate: np.ndarray, accepted: list[np.ndarray]) -> bool:
    """Mean per-corner displacement against every view already accepted.

    Must be the mean of per-corner distances, not a norm over the flattened
    array — the latter scales with sqrt(corner count) and silently rejects
    almost every view.
    """
    for existing in accepted:
        displacement = float(
            np.mean(np.linalg.norm(candidate - existing, axis=-1))
        )
        if displacement < MIN_VIEW_SEPARATION_PX:
            return False
    return True


def _diagonal_fov_degrees(fx: float, fy: float, width: int, height: int) -> float:
    diagonal = math.hypot(width, height)
    focal = (fx + fy) / 2.0
    return math.degrees(2.0 * math.atan(diagonal / (2.0 * focal)))


def _advice(band: str, views: int, fov_plausible: bool) -> str:
    if not fov_plausible:
        return (
            "The solved field of view is outside the range a phone main camera "
            "produces. Re-film with more tilt and with the board reaching the "
            "edges of the frame."
        )
    if band == "POOR":
        return (
            "Reprojection error is high. Re-film more slowly, keep the board "
            "flat, and include tilts of 20-45 degrees."
        )
    if views < RECOMMENDED_VIEWS:
        return (
            f"Usable, but only {views} distinct views were found. Filming more "
            "angles would tighten the result."
        )
    return "Calibration is good. This phone will not need calibrating again."


def calibrate_device_from_video(
    video_path: Path,
    device_id: str,
    device_label: str | None,
    spec: CheckerboardSpec,
) -> DeviceLensProfile:
    object_grid = _object_points(spec)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    accepted: list[np.ndarray] = []
    submitted = 0
    size: tuple[int, int] | None = None

    for frame in _iter_frames(video_path):
        submitted += 1
        if size is None:
            size = (frame.shape[1], frame.shape[0])
        corners = _detect_corners(frame, spec)
        if corners is None:
            continue
        if not _is_distinct(corners, accepted):
            continue
        accepted.append(corners)
        image_points.append(corners)
        object_points.append(object_grid)

    if size is None:
        raise DeviceCalibrationError("The calibration video contained no frames.")
    if len(image_points) < MIN_VIEWS:
        raise DeviceCalibrationError(
            f"Only {len(image_points)} usable views were found; {MIN_VIEWS} are "
            "required. Check the inner-corner counts match the printed board, "
            "keep the whole board in frame, and move slowly enough to avoid blur.",
            status_code=422,
        )

    # cv2 raises on degenerate input — for example every view being effectively
    # the same plane, which happens when the board is filmed flat-on throughout.
    # That is a user-fixable filming problem, not a server fault.
    try:
        rms, camera_matrix, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            size,
            None,
            None,
        )
    except cv2.error as exc:
        raise DeviceCalibrationError(
            "The views could not be solved into a single lens. This usually "
            "means the board was filmed at too similar an angle throughout — "
            "re-film with tilts of 20-45 degrees in different directions.",
            status_code=422,
        ) from exc

    width, height = size
    fx = float(camera_matrix[0][0])
    fy = float(camera_matrix[1][1])
    cx = float(camera_matrix[0][2])
    cy = float(camera_matrix[1][2])
    # DeviceLensProfile constrains these to be positive and finite. A degenerate
    # solve can produce a negative principal point or a non-finite focal length,
    # and letting that reach Pydantic turns a bad calibration into a 500.
    solved = (fx, fy, cx, cy, float(rms))
    if not all(math.isfinite(value) for value in solved) or fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
        raise DeviceCalibrationError(
            "The solve produced an impossible lens. Re-film with more varied "
            "angles and with the board reaching the edges of the frame.",
            status_code=422,
        )
    fov = _diagonal_fov_degrees(fx, fy, width, height)
    plausible = MIN_PLAUSIBLE_DIAGONAL_FOV_DEG <= fov <= MAX_PLAUSIBLE_DIAGONAL_FOV_DEG
    band = quality_band(float(rms))

    return DeviceLensProfile(
        device_id=device_id,
        device_label=device_label,
        calibrated_at=datetime.now(timezone.utc),
        image_width=width,
        image_height=height,
        focal_length_x_px=fx,
        focal_length_y_px=fy,
        principal_point_x_px=cx,
        principal_point_y_px=cy,
        distortion_coefficients=[float(value) for value in distortion.ravel()[:5]],
        checkerboard=spec,
        quality=CalibrationQuality(
            rms_reprojection_px=float(rms),
            band=band,
            views_used=len(image_points),
            views_submitted=submitted,
            diagonal_fov_degrees=fov,
            fov_plausible=plausible,
            advice=_advice(band, len(image_points), plausible),
        ),
    )


def save_device_profile(profile: DeviceLensProfile) -> Path:
    DEVICE_CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.device_id)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_device_profile(device_id: str) -> DeviceLensProfile | None:
    path = _profile_path(device_id)
    if not path.is_file():
        return None
    try:
        return DeviceLensProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def delete_device_profile(device_id: str) -> bool:
    """Discard a stored profile so a bad calibration can be redone."""
    path = _profile_path(device_id)
    if not path.is_file():
        return False
    path.unlink()
    return True


def load_device_profile_for_frame(
    device_id: str | None,
    width: int,
    height: int,
) -> DeviceLensProfile | None:
    """Profile rescaled to the frame in hand, or None when unusable.

    Returning None rather than raising keeps the caller free to fall back to
    the bounded-hypothesis sweep, which is still better than nothing.
    """
    if not device_id:
        return None
    profile = load_device_profile(device_id)
    if profile is None:
        return None
    if profile.image_width == width and profile.image_height == height:
        return profile
    try:
        return profile.scaled_to(width, height)
    except ValueError:
        return None
