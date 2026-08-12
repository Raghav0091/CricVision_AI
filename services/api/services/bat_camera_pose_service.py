"""Camera pose from two upright bats — a reduced-accuracy stand-in for the
twelve-landmark stump calibration when real stumps are unavailable.

Stump calibration (`video_camera_pose_service.py`) solves from twelve known
3D points (left/middle/right base+top at both ends), spread across three
different lateral (x) positions, which conditions the PnP solve well and
resolves camera roll unambiguously.

A single bat per end gives only two points (base + top of one vertical
line), and both bats sit on the same lateral centreline (x=0), so all four
points are coplanar. Coplanar point sets are a classically weaker PnP case
than a spread 3D point cloud — the solve is usable but noticeably less
robust than the stump path, especially for anything that depends on lateral
(swing/line) accuracy. This module grades its own confidence accordingly and
never reports HIGH confidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CRICVISION_PITCH_V1,
    PITCH_LENGTH_M,
)

from ..schemas.delivery_physics import CameraCalibration
from .bat_detector_service import bat_anchor_points
from .video_analysis_service import ANALYSIS_ID_PATTERN, VIDEO_ANALYSIS_ROOT


BAT_POSE_SCHEMA_VERSION = "bat_camera_pose_v1"
BAT_POSE_FILENAME = "bat_camera_pose.json"


# Standard full-size adult bat length. Not measured per-session; see warnings
# on every result produced from this module.
DEFAULT_BAT_HEIGHT_M = 0.86

ASSUMED_HORIZONTAL_FOV_DEGREES = 60.0
MAX_ACCEPTABLE_RMSE_PX = 25.0


def _estimated_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    horizontal_fov_radians = math.radians(ASSUMED_HORIZONTAL_FOV_DEGREES)
    focal_length = image_width / (2 * math.tan(horizontal_fov_radians / 2))
    return np.asarray(
        [
            [focal_length, 0.0, image_width / 2],
            [0.0, focal_length, image_height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _bat_world_points(bat_height_m: float) -> np.ndarray:
    """Four world points: near/far base (z=0) and near/far top (z=bat_height_m).

    Matches the physics engine's canonical world axes directly (x lateral,
    y longitudinal 0..PITCH_LENGTH_M, z vertical) — no coordinate swap needed
    for CRICVISION_PITCH_V1, unlike the CALIBRATION_V2 stump path.
    """
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, bat_height_m],
            [0.0, PITCH_LENGTH_M, 0.0],
            [0.0, PITCH_LENGTH_M, bat_height_m],
        ],
        dtype=np.float64,
    )


def solve_camera_pose_from_bats(
    *,
    near_detection: dict[str, Any],
    far_detection: dict[str, Any],
    image_width: int,
    image_height: int,
    bat_height_m: float = DEFAULT_BAT_HEIGHT_M,
) -> CameraCalibration:
    """Solve camera pose from one bat detection at each end.

    `near_detection`/`far_detection` are single-end results from
    `bat_detector_service.detect_bats_guided(...)["detections"]`
    (non_striker=near, striker=far).
    """
    near_anchor = bat_anchor_points(near_detection, bat_height_m=bat_height_m)
    far_anchor = bat_anchor_points(far_detection, bat_height_m=bat_height_m)
    if near_anchor is None or far_anchor is None:
        return CameraCalibration(
            mode="IMAGE_SPACE_ONLY",
            confidence="UNAVAILABLE",
            image_width=image_width,
            image_height=image_height,
            calibration_confidence=0.0,
            failure_reason="Both bats were not detected at both ends.",
            warnings=["Bat-based calibration requires a detected bat at each end."],
        )

    image_points = np.asarray(
        [
            [near_anchor["base_px"]["x"], near_anchor["base_px"]["y"]],
            [near_anchor["top_px"]["x"], near_anchor["top_px"]["y"]],
            [far_anchor["base_px"]["x"], far_anchor["base_px"]["y"]],
            [far_anchor["top_px"]["x"], far_anchor["top_px"]["y"]],
        ],
        dtype=np.float64,
    )
    world_points = _bat_world_points(bat_height_m)
    camera_matrix = _estimated_camera_matrix(image_width, image_height)
    distortion = np.zeros(5, dtype=np.float64)

    # SQPNP (not EPNP) — EPNP is documented as poorly conditioned for exactly
    # four coplanar points, which is what two centreline bats produce; SQPNP
    # is a globally-optimal solver that handles this case correctly
    # (verified: recovers a synthetic ground-truth pose to near-zero
    # reprojection error where EPNP left 50+ px of residual error).
    success, rotation_vector, translation_vector = cv2.solvePnP(
        world_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    warnings = [
        f"Calibrated from two bats assumed to be {bat_height_m:.3f}m long "
        "(standard adult length, not individually measured). Wrong bat "
        "length will bias every 3D measurement.",
        "Both bat anchors sit on the pitch centreline (coplanar points); "
        "this is a materially weaker-conditioned pose solve than stump "
        "calibration and lateral (swing/line) accuracy is reduced.",
        "Camera intrinsics are estimated from an assumed 60 degree "
        "horizontal field of view, not calibrated.",
    ]
    if not success:
        return CameraCalibration(
            mode="IMAGE_SPACE_ONLY",
            confidence="UNAVAILABLE",
            image_width=image_width,
            image_height=image_height,
            calibration_confidence=0.0,
            failure_reason="solvePnP did not converge on the bat correspondences.",
            warnings=warnings,
        )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projected, _ = cv2.projectPoints(
        world_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    errors_px = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rmse_px = float(np.sqrt(np.mean(np.square(errors_px))))

    if rmse_px > MAX_ACCEPTABLE_RMSE_PX:
        return CameraCalibration(
            mode="IMAGE_SPACE_ONLY",
            confidence="UNAVAILABLE",
            image_width=image_width,
            image_height=image_height,
            calibration_confidence=0.0,
            reprojection_error_px=round(rmse_px, 3),
            failure_reason=(
                f"Bat-pose reprojection RMSE ({rmse_px:.1f}px) exceeds the "
                f"{MAX_ACCEPTABLE_RMSE_PX:.0f}px acceptance threshold."
            ),
            warnings=warnings,
        )

    # Confidence is capped well below the stump path's ceiling: even a
    # perfect fit on four coplanar points is a weaker pose than twelve
    # spread points, so this never reports HIGH.
    confidence_score = max(0.0, min(0.55, 0.55 * (1.0 - rmse_px / MAX_ACCEPTABLE_RMSE_PX)))
    projection_matrix = camera_matrix @ np.hstack([rotation_matrix, translation_vector])

    return CameraCalibration(
        mode="METRIC_3D",
        confidence="MEDIUM" if confidence_score >= 0.35 else "LOW",
        world_coordinate_system=CRICVISION_PITCH_V1,
        image_width=image_width,
        image_height=image_height,
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=distortion.tolist(),
        rotation_vector=rotation_vector.reshape(-1).tolist(),
        rotation_matrix=rotation_matrix.tolist(),
        translation_vector=translation_vector.reshape(-1).tolist(),
        projection_matrix=projection_matrix.tolist(),
        correspondences_used=4,
        reprojection_error_px=round(rmse_px, 3),
        calibration_confidence=round(confidence_score, 6),
        warnings=warnings,
    )


def _require_analysis_id(analysis_id: str) -> None:
    if not analysis_id or not ANALYSIS_ID_PATTERN.fullmatch(analysis_id):
        raise ValueError("analysis_id is invalid.")


def _bat_pose_path(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / BAT_POSE_FILENAME


def save_bat_camera_pose(
    analysis_id: str,
    calibration: CameraCalibration,
) -> Path:
    """Persist a solved bat-pose calibration so it survives process restarts."""
    _require_analysis_id(analysis_id)
    path = _bat_pose_path(analysis_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": BAT_POSE_SCHEMA_VERSION,
        "analysis_id": analysis_id,
        "calibration": calibration.model_dump(mode="json"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_bat_camera_pose(analysis_id: str) -> CameraCalibration | None:
    """Return a previously solved bat-pose calibration, or None if absent/unusable."""
    _require_analysis_id(analysis_id)
    path = _bat_pose_path(analysis_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("schema_version") != BAT_POSE_SCHEMA_VERSION:
        return None
    if payload.get("analysis_id") != analysis_id:
        return None
    try:
        calibration = CameraCalibration.model_validate(payload["calibration"])
    except (KeyError, TypeError, ValueError):
        return None
    if calibration.mode != "METRIC_3D":
        return None
    return calibration


def _manual_detection(bbox: dict[str, float], source_box: str) -> dict[str, Any]:
    return {
        "found": True,
        "confidence": 1.0,
        "bbox": {
            "x": int(round(bbox["x"])),
            "y": int(round(bbox["y"])),
            "width": max(1, int(round(bbox["width"]))),
            "height": max(1, int(round(bbox["height"]))),
        },
        "source_box": source_box,
        "class_name": "manual",
    }


def run_bat_pose_calibration(
    analysis_id: str,
    *,
    frame_data_url: str,
    frame_width: int,
    frame_height: int,
    guides: dict[str, dict[str, float]] | None = None,
    bat_height_m: float = DEFAULT_BAT_HEIGHT_M,
    manual_boxes: dict[str, dict[str, float]] | None = None,
    bat_detector_model_key: str = "cricshot10k",
) -> dict[str, Any]:
    """End-to-end: decode a submitted frame, get both bat boxes, solve, persist.

    `manual_boxes` (pixel-space `{x, y, width, height}` per end) skips the ML
    detector entirely and uses the caller-supplied box directly — useful when
    the detector is unreliable for a given setup (e.g. a bat standing
    upright and static looks nothing like the batting-shot footage the
    CricShot10k bat model was trained on).

    Mirrors the shape of `routes/calibration.py`'s stump `/solve` endpoint so
    the frontend can reuse the same "capture frame -> POST -> render result"
    pattern for the bat fallback path.
    """
    from .calibration_store import save_calibration_frame

    try:
        _frame_path, image = save_calibration_frame(frame_data_url)
    except ValueError as exc:
        return {
            "success": False,
            "status": "invalid_calibration_frame",
            "message": str(exc),
            "detections": None,
            "calibration": None,
        }

    if manual_boxes is not None:
        detections = {
            "striker": _manual_detection(manual_boxes["striker"], "striker"),
            "non_striker": _manual_detection(manual_boxes["non_striker"], "non_striker"),
        }
    else:
        from .bat_detector_service import detect_bats_guided

        detection_result = detect_bats_guided(image, guides, model_key=bat_detector_model_key)
        if detection_result["status"] in ("bat_detector_missing", "bat_detector_error"):
            return {
                "success": False,
                "status": detection_result["status"],
                "message": detection_result["message"],
                "detections": None,
                "calibration": None,
            }

        detections = detection_result["detections"]
        if not detection_result["success"]:
            return {
                "success": False,
                "status": "bats_not_found",
                "message": detection_result["message"],
                "detections": detections,
                "calibration": None,
            }

    calibration = register_and_persist_bat_camera_pose(
        analysis_id,
        near_detection=detections["non_striker"],
        far_detection=detections["striker"],
        image_width=frame_width,
        image_height=frame_height,
        bat_height_m=bat_height_m,
    )
    if calibration.mode != "METRIC_3D":
        return {
            "success": False,
            "status": "solve_failed",
            "message": calibration.failure_reason or "Bat-pose calibration failed.",
            "detections": detections,
            "calibration": calibration,
        }

    return {
        "success": True,
        "status": "calibrated",
        "message": "Camera pose solved from two bats and saved.",
        "detections": detections,
        "calibration": calibration,
    }


def register_and_persist_bat_camera_pose(
    analysis_id: str,
    *,
    near_detection: dict[str, Any],
    far_detection: dict[str, Any],
    image_width: int,
    image_height: int,
    bat_height_m: float = DEFAULT_BAT_HEIGHT_M,
) -> CameraCalibration:
    """Solve a bat-pose calibration and persist it if usable. Always returns the
    result (including UNAVAILABLE ones) but only writes usable METRIC_3D results."""
    calibration = solve_camera_pose_from_bats(
        near_detection=near_detection,
        far_detection=far_detection,
        image_width=image_width,
        image_height=image_height,
        bat_height_m=bat_height_m,
    )
    if calibration.mode == "METRIC_3D":
        save_bat_camera_pose(analysis_id, calibration)
    return calibration
