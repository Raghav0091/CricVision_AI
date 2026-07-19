"""Calibration v2B: camera pose from explicit wicket-body landmarks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CricketPitchDimensions,
    LEFT_RIGHT_CONVENTION,
    wicket_landmark_world_points,
)

from ..schemas.video_analysis import (
    CalibrationCoordinateSystem,
    CameraIntrinsics,
    CameraPoseQualityComponents,
    CameraPoseReprojectionDiagnostic,
    CameraPoseSolution,
    CricketPitchGeometry,
    WicketCameraPoseInitialiseResponse,
    WicketCameraPoseResult,
    WicketCameraPoseSolveRequest,
    WicketPoseLandmark,
    WicketPoseLandmarkInput,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_calibration_v2_service import (
    initialise_video_calibration_v2,
    load_video_calibration_v2,
)


CAMERA_POSE_FILENAME = "camera_pose.json"
CAMERA_POSE_OVERLAY_FILENAME = "camera_pose_overlay.jpg"
ASSUMED_HORIZONTAL_FOV_DEGREES = 60.0
PNP_RANSAC_REPROJECTION_THRESHOLD_PX = 6.0
PNP_RANSAC_ITERATIONS = 500
PNP_RANSAC_CONFIDENCE = 0.999
MIN_USABLE_LANDMARKS = 8
MIN_ACCEPTED_INLIER_RATIO = 0.65

STUMP_POSITIONS = ("left", "middle", "right")
WICKET_ENDS = ("bowler", "striker")
POINT_TYPES = ("base", "top")
WICKET_LANDMARK_IDS = tuple(
    f"{wicket_end}_{stump_position}_stump_{point_type}"
    for wicket_end in WICKET_ENDS
    for point_type in POINT_TYPES
    for stump_position in STUMP_POSITIONS
)


def initialise_wicket_camera_pose(
    analysis_id: str,
) -> WicketCameraPoseInitialiseResponse:
    analysis = load_video_analysis(analysis_id)
    image = _open_reference_image(_reference_path(analysis_id))
    image_width, image_height = image.size
    pitch_geometry = CricketPitchGeometry()
    saved_ground = None
    try:
        saved_ground = load_video_calibration_v2(analysis_id)
        pitch_geometry = saved_ground.pitch_geometry
    except VideoAnalysisServiceError:
        pass
    geometry = _shared_geometry(pitch_geometry)
    world_points = wicket_landmark_world_points(geometry)
    ground_initialisation = (
        initialise_video_calibration_v2(analysis_id)
        if saved_ground is None
        else None
    )
    base_landmarks = (
        saved_ground.landmark_set.primary_stump_bases
        if saved_ground is not None
        else ground_initialisation.landmarks
    )
    bases = {landmark.id: landmark for landmark in base_landmarks}

    landmarks: list[WicketPoseLandmark] = []
    for landmark_id in WICKET_LANDMARK_IDS:
        wicket_end, stump_position, _, point_type = landmark_id.split("_")
        base = bases[f"{wicket_end}_{stump_position}_stump_base"]
        normalized_y = (
            base.normalized_y
            if point_type == "base"
            else _top_guess_y(wicket_end, base.normalized_y)
        )
        world_x, world_y, world_z = world_points[landmark_id]
        landmarks.append(
            WicketPoseLandmark(
                id=landmark_id,
                label=_landmark_label(
                    wicket_end,
                    stump_position,
                    point_type,
                ),
                wicket_end=wicket_end,
                stump_position=stump_position,
                point_type=point_type,
                normalized_x=base.normalized_x,
                normalized_y=normalized_y,
                pixel_x=round(base.normalized_x * image_width, 6),
                pixel_y=round(normalized_y * image_height, 6),
                source="inferred",
                confidence=None,
                visibility="visible",
                world_x_m=world_x,
                world_y_m=world_y,
                world_z_m=world_z,
            )
        )

    return WicketCameraPoseInitialiseResponse(
        success=True,
        status="initialised",
        analysis_id=analysis_id,
        reference_frame_url=analysis.reference_frame_url,
        image_width=image_width,
        image_height=image_height,
        pitch_geometry=pitch_geometry,
        landmarks=landmarks,
        camera_intrinsics=_estimated_intrinsics(image_width, image_height),
        warnings=[
            (
                "All twelve markers are low-confidence guesses inferred from "
                "wicket boxes. Correct them before solving."
            ),
            (
                "Place BASE markers at stump-ground contact and TOP markers "
                "at the top of the stump body, excluding bails."
            ),
            (
                "Camera intrinsics are estimated from a declared 60 degree "
                "horizontal field-of-view assumption; lens distortion is not "
                "calibrated."
            ),
            *(
                [
                    (
                        "Six base starters were restored from Calibration "
                        "v2A; top points remain inferred guesses."
                    )
                ]
                if saved_ground is not None
                else []
            ),
        ],
        message="Wicket-based camera-pose landmarks initialised.",
    )


def solve_wicket_camera_pose(
    analysis_id: str,
    request: WicketCameraPoseSolveRequest,
) -> WicketCameraPoseResult:
    if request.analysis_id != analysis_id:
        raise VideoAnalysisServiceError(
            "Camera-pose analysis ID does not match the URL.",
            status_code=400,
        )
    analysis = load_video_analysis(analysis_id)
    image = _open_reference_image(_reference_path(analysis_id))
    image_width, image_height = image.size
    if (
        request.camera_intrinsics.image_width != image_width
        or request.camera_intrinsics.image_height != image_height
    ):
        raise VideoAnalysisServiceError(
            "Camera intrinsics resolution does not match the reference frame.",
            status_code=422,
        )
    geometry = _shared_geometry(request.pitch_geometry)
    landmarks = _confirmed_landmarks(
        request.landmarks,
        geometry,
        image_width,
        image_height,
    )
    camera_matrix, distortion = _validated_intrinsics(
        request.camera_intrinsics,
    )
    solution, quality, status = _calculate_pose(
        landmarks,
        request.camera_intrinsics,
        camera_matrix,
        distortion,
        geometry,
        image_width,
        image_height,
        request.landmark_semantics_confirmed,
    )

    calibration_dir = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration"
    calibration_path = calibration_dir / CAMERA_POSE_FILENAME
    overlay_path = calibration_dir / CAMERA_POSE_OVERLAY_FILENAME
    now = datetime.now(timezone.utc)
    created_at = _existing_created_at(calibration_path) or now
    camera_pose_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{CAMERA_POSE_FILENAME}"
    )
    overlay_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{CAMERA_POSE_OVERLAY_FILENAME}"
    )
    messages = {
        "ready": "Wicket-based camera pose is ready using calibrated intrinsics.",
        "usable": (
            "Wicket-based camera pose is usable, but estimated camera "
            "intrinsics limit metric confidence."
        ),
        "weak": (
            "A camera pose was calculated, but the evidence is too weak for "
            "reliable world mapping."
        ),
        "unstable": (
            "The wicket correspondences produced an unstable camera pose."
        ),
        "insufficient_landmarks": (
            "Not enough reliable, distributed wicket landmarks are available."
        ),
        "solver_failed": "OpenCV could not solve a camera pose from these landmarks.",
        "implausible_pose": (
            "A numerical pose was found but rejected by physical plausibility checks."
        ),
    }
    result = WicketCameraPoseResult(
        success=status in {"ready", "usable"},
        status=status,
        schema_version="2.2",
        analysis_id=analysis_id,
        calibration_mode="wicket_camera_pose",
        coordinate_system=CalibrationCoordinateSystem(
            units="metres",
            origin="bowler_wicket_centre",
            x_axis="toward_striker",
            y_axis="lateral",
            z_axis="up",
            left_right_convention=LEFT_RIGHT_CONVENTION,
            image_left_right_convention="image_left_is_world_left",
        ),
        pitch_geometry=request.pitch_geometry,
        stump_top_definition="top_of_stump_body_excluding_bails",
        landmarks=landmarks,
        camera_intrinsics=request.camera_intrinsics,
        camera_pose=solution,
        quality=quality,
        camera_pose_url=camera_pose_url,
        camera_pose_overlay_url=overlay_url,
        reference_frame_url=analysis.reference_frame_url,
        image_width=image_width,
        image_height=image_height,
        landmark_semantics_confirmed=request.landmark_semantics_confirmed,
        created_at=created_at,
        updated_at=now,
        user_note=request.user_note,
        message=messages[status],
    )
    _save_overlay(image, result, camera_matrix, distortion, overlay_path)
    _write_json(calibration_path, result.model_dump(mode="json"))
    _update_analysis_metadata(analysis_id, result, now)
    return result


def load_wicket_camera_pose(analysis_id: str) -> WicketCameraPoseResult:
    load_video_analysis(analysis_id)
    calibration_dir = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration"
    calibration_path = calibration_dir / CAMERA_POSE_FILENAME
    overlay_path = calibration_dir / CAMERA_POSE_OVERLAY_FILENAME
    if not calibration_path.is_file():
        raise VideoAnalysisServiceError(
            "Wicket-based camera pose has not been solved.",
            status_code=404,
        )
    try:
        result = WicketCameraPoseResult.model_validate(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored wicket-based camera pose is unavailable.",
            status_code=500,
        ) from exc
    if (
        result.analysis_id != analysis_id
        or not _reference_path(analysis_id).is_file()
        or not overlay_path.is_file()
    ):
        raise VideoAnalysisServiceError(
            "Stored wicket-based camera-pose files are incomplete.",
            status_code=404,
        )
    return result


def _calculate_pose(
    landmarks: list[WicketPoseLandmark],
    intrinsics: CameraIntrinsics,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    geometry: CricketPitchDimensions,
    image_width: int,
    image_height: int,
    semantics_confirmed: bool,
) -> tuple[CameraPoseSolution, CameraPoseQualityComponents, str]:
    unavailable = [
        landmark.id
        for landmark in landmarks
        if landmark.visibility in {"occluded", "unavailable"}
    ]
    used = [
        landmark
        for landmark in landmarks
        if landmark.visibility in {"visible", "uncertain"}
    ]
    warnings: list[str] = []
    rejection_reasons: list[str] = []
    if not semantics_confirmed:
        rejection_reasons.append("wicket_landmark_semantics_not_confirmed")
    coverage_issues = _coverage_issues(used)
    rejection_reasons.extend(coverage_issues)
    if any(landmark.visibility == "uncertain" for landmark in used):
        warnings.append("Uncertain landmarks were included in the pose solve.")
    if intrinsics.source in {"heuristic_estimated", "metadata_estimated"}:
        warnings.append("pose_low_confidence_due_to_intrinsics")
    if intrinsics.distortion_model_source == "not_calibrated":
        warnings.append("Lens distortion is assumed zero because it is not calibrated.")
    warnings.append(
        (
            "The wicket target is long and narrow rather than a generic 3D "
            "calibration object; reprojection and physical checks remain required."
        )
    )
    geometry_score = _geometry_condition_score(used)

    if rejection_reasons:
        quality = _quality_components(
            used,
            intrinsics,
            geometry_score,
            reprojection_quality=0.0,
            plausibility=0.0,
        )
        return (
            _empty_solution(
                "not_run",
                used,
                unavailable,
                warnings,
                rejection_reasons,
            ),
            quality,
            "insufficient_landmarks",
        )

    object_points = np.asarray(
        [[item.world_x_m, item.world_y_m, item.world_z_m] for item in used],
        dtype=np.float64,
    )
    image_points = np.asarray(
        [[item.pixel_x, item.pixel_y] for item in used],
        dtype=np.float64,
    )
    try:
        solved, rotation_vector, translation_vector, inlier_indices = (
            cv2.solvePnPRansac(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_EPNP,
                iterationsCount=PNP_RANSAC_ITERATIONS,
                reprojectionError=PNP_RANSAC_REPROJECTION_THRESHOLD_PX,
                confidence=PNP_RANSAC_CONFIDENCE,
            )
        )
    except cv2.error:
        solved = False
        rotation_vector = None
        translation_vector = None
        inlier_indices = None
    if not solved or rotation_vector is None or translation_vector is None:
        rejection_reasons.append("solvePnPRansac_failed")
        quality = _quality_components(
            used,
            intrinsics,
            geometry_score,
            reprojection_quality=0.0,
            plausibility=0.0,
        )
        return (
            _empty_solution(
                "solvePnPRansac_EPNP",
                used,
                unavailable,
                warnings,
                rejection_reasons,
            ),
            quality,
            "solver_failed",
        )

    inlier_positions = (
        [int(value) for value in inlier_indices.reshape(-1)]
        if inlier_indices is not None
        else list(range(len(used)))
    )
    inlier_ids = [used[index].id for index in inlier_positions]
    refinement_method: str | None = None
    if len(inlier_positions) >= 6:
        try:
            rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                object_points[inlier_positions],
                image_points[inlier_positions],
                camera_matrix,
                distortion,
                rotation_vector,
                translation_vector,
            )
            refinement_method = "solvePnPRefineLM_on_ransac_inliers"
        except cv2.error:
            warnings.append("Levenberg-Marquardt pose refinement failed.")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    projected_points = projected.reshape(-1, 2)
    camera_points = (
        rotation_matrix @ object_points.T + translation_vector.reshape(3, 1)
    ).T
    inlier_set = set(inlier_ids)
    diagnostics: list[CameraPoseReprojectionDiagnostic] = []
    errors: list[float] = []
    for landmark, observed, predicted, camera_point in zip(
        used,
        image_points,
        projected_points,
        camera_points,
        strict=True,
    ):
        residual = float(np.linalg.norm(observed - predicted))
        errors.append(residual)
        diagnostics.append(
            CameraPoseReprojectionDiagnostic(
                landmark_id=landmark.id,
                observed_pixel_x=round(float(observed[0]), 6),
                observed_pixel_y=round(float(observed[1]), 6),
                projected_pixel_x=round(float(predicted[0]), 6),
                projected_pixel_y=round(float(predicted[1]), 6),
                residual_px=round(residual, 6),
                camera_depth_m=round(float(camera_point[2]), 6),
                ransac_inlier=landmark.id in inlier_set,
            )
        )
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    median_error = float(np.median(np.asarray(errors)))
    maximum_error = max(errors)
    diagonal = math.hypot(image_width, image_height)
    normalized_rmse = rmse / diagonal
    camera_position = -rotation_matrix.T @ translation_vector.reshape(3, 1)
    camera_position = camera_position.reshape(3)
    camera_forward = rotation_matrix.T @ np.array([0.0, 0.0, 1.0])
    camera_forward /= max(float(np.linalg.norm(camera_forward)), 1e-12)
    positive_depth = bool(np.all(camera_points[:, 2] > 0))
    bowler_depth = _world_depth(
        rotation_matrix,
        translation_vector,
        np.array([0.0, 0.0, 0.0]),
    )
    striker_depth = _world_depth(
        rotation_matrix,
        translation_vector,
        np.array([geometry.pitch_length_m, 0.0, 0.0]),
    )
    both_wickets_in_front = bowler_depth > 0 and striker_depth > 0
    vector_to_pitch = np.array(
        [geometry.pitch_length_m / 2, 0.0, 0.0]
    ) - camera_position
    vector_to_pitch /= max(float(np.linalg.norm(vector_to_pitch)), 1e-12)
    camera_faces_pitch = float(camera_forward @ vector_to_pitch) > 0.2
    if camera_position[0] < 0:
        wicket_order_plausible = bowler_depth < striker_depth
    elif camera_position[0] > geometry.pitch_length_m:
        wicket_order_plausible = striker_depth < bowler_depth
    else:
        # A camera alongside the pitch can legitimately have either end closer.
        wicket_order_plausible = abs(striker_depth - bowler_depth) > 1e-4
    camera_height = float(camera_position[2])
    camera_distance = float(np.linalg.norm(camera_position))
    finite_pose = bool(
        np.isfinite(rotation_matrix).all()
        and np.isfinite(translation_vector).all()
        and np.isfinite(camera_position).all()
    )
    plausibility_checks = [
        finite_pose,
        abs(float(np.linalg.det(rotation_matrix)) - 1.0) < 1e-3,
        positive_depth,
        both_wickets_in_front,
        camera_faces_pitch,
        0.2 <= camera_height <= 20.0,
        camera_distance <= 150.0,
    ]
    plausibility = sum(plausibility_checks) / len(plausibility_checks)
    if not positive_depth:
        rejection_reasons.append("one_or_more_landmarks_behind_camera")
    if not both_wickets_in_front:
        rejection_reasons.append("one_or_both_wickets_behind_camera")
    if not camera_faces_pitch:
        rejection_reasons.append("camera_optical_axis_does_not_face_pitch")
    if not 0.2 <= camera_height <= 20.0:
        rejection_reasons.append("camera_height_outside_broad_plausible_range")
    if camera_distance > 150.0:
        rejection_reasons.append("camera_position_over_150m_from_world_origin")
    if not wicket_order_plausible:
        warnings.append("Bowler/striker depth ordering is numerically ambiguous.")
    if not finite_pose:
        rejection_reasons.append("pose_contains_non_finite_values")

    inlier_ratio = len(inlier_ids) / len(used)
    if inlier_ratio < MIN_ACCEPTED_INLIER_RATIO:
        warnings.append(
            f"RANSAC retained only {len(inlier_ids)}/{len(used)} landmarks."
        )
    reprojection_quality = max(
        0.0,
        min(1.0, 1.0 - normalized_rmse / 0.008),
    )
    quality = _quality_components(
        used,
        intrinsics,
        geometry_score,
        reprojection_quality,
        plausibility,
    )
    if rejection_reasons:
        status = "implausible_pose"
    elif inlier_ratio < MIN_ACCEPTED_INLIER_RATIO or normalized_rmse > 0.015:
        status = "unstable"
    elif quality.landmark_quality < 0.7:
        warnings.append(
            "Low-confidence inferred landmarks prevent pose acceptance."
        )
        status = "weak"
    elif quality.overall_pose_quality < 0.48:
        status = "weak"
    elif intrinsics.source == "calibrated_device_profile":
        status = "ready"
    else:
        status = "usable"
    accepted = status in {"ready", "usable"}
    outlier_ids = [item.id for item in used if item.id not in inlier_set]
    solution = CameraPoseSolution(
        solved=True,
        accepted=accepted,
        solver_method="solvePnPRansac_EPNP",
        refinement_method=refinement_method,
        rotation_vector=_vector_list(rotation_vector),
        rotation_matrix=_matrix_list(rotation_matrix),
        translation_vector=_vector_list(translation_vector),
        camera_position_world=_vector_list(camera_position),
        camera_forward_direction_world=_vector_list(camera_forward),
        camera_height_m=round(camera_height, 6),
        landmark_count=len(used),
        used_landmark_ids=[item.id for item in used],
        unavailable_landmark_ids=unavailable,
        ransac_inlier_ids=inlier_ids,
        ransac_outlier_ids=outlier_ids,
        reprojection_rmse_px=round(rmse, 6),
        reprojection_median_px=round(median_error, 6),
        reprojection_max_px=round(maximum_error, 6),
        normalized_reprojection_rmse=round(normalized_rmse, 9),
        reprojection_diagnostics=diagnostics,
        positive_depth_for_all_used_landmarks=positive_depth,
        both_wickets_in_front=both_wickets_in_front,
        camera_faces_pitch=camera_faces_pitch,
        wicket_order_plausible=wicket_order_plausible,
        warnings=_deduplicate(warnings),
        rejection_reasons=_deduplicate(rejection_reasons),
    )
    return solution, quality, status


def _confirmed_landmarks(
    inputs: list[WicketPoseLandmarkInput],
    geometry: CricketPitchDimensions,
    image_width: int,
    image_height: int,
) -> list[WicketPoseLandmark]:
    ids = [item.id for item in inputs]
    if len(ids) != len(set(ids)) or set(ids) != set(WICKET_LANDMARK_IDS):
        raise VideoAnalysisServiceError(
            "Camera-pose calibration requires the exact twelve wicket landmark IDs.",
            status_code=422,
        )
    by_id = {item.id: item for item in inputs}
    world_points = wicket_landmark_world_points(geometry)
    result: list[WicketPoseLandmark] = []
    for landmark_id in WICKET_LANDMARK_IDS:
        item = by_id[landmark_id]
        expected_end, expected_position, _, expected_type = landmark_id.split("_")
        if (
            item.wicket_end != expected_end
            or item.stump_position != expected_position
            or item.point_type != expected_type
        ):
            raise VideoAnalysisServiceError(
                f"Landmark {landmark_id} has inconsistent semantics.",
                status_code=422,
            )
        world_x, world_y, world_z = world_points[landmark_id]
        result.append(
            WicketPoseLandmark(
                **item.model_dump(
                    exclude={
                        "pixel_x",
                        "pixel_y",
                        "world_x_m",
                        "world_y_m",
                        "world_z_m",
                    }
                ),
                pixel_x=round(item.normalized_x * image_width, 6),
                pixel_y=round(item.normalized_y * image_height, 6),
                world_x_m=world_x,
                world_y_m=world_y,
                world_z_m=world_z,
            )
        )
    return result


def _coverage_issues(used: list[WicketPoseLandmark]) -> list[str]:
    issues: list[str] = []
    if len(used) < MIN_USABLE_LANDMARKS:
        issues.append(f"at_least_{MIN_USABLE_LANDMARKS}_usable_landmarks_required")
    for wicket_end in WICKET_ENDS:
        end_landmarks = [item for item in used if item.wicket_end == wicket_end]
        if len(end_landmarks) < 4:
            issues.append(f"{wicket_end}_wicket_has_fewer_than_4_usable_landmarks")
        for point_type in POINT_TYPES:
            count = sum(item.point_type == point_type for item in end_landmarks)
            if count < 2:
                issues.append(
                    f"{wicket_end}_wicket_requires_2_{point_type}_landmarks"
                )
    return issues


def _geometry_condition_score(landmarks: list[WicketPoseLandmark]) -> float:
    if len(landmarks) < 4:
        return 0.0
    points = np.asarray(
        [[item.world_x_m, item.world_y_m, item.world_z_m] for item in landmarks],
        dtype=np.float64,
    )
    centred = points - points.mean(axis=0)
    scale = centred.std(axis=0)
    if np.any(scale < 1e-9):
        return 0.0
    singular_values = np.linalg.svd(centred / scale, compute_uv=False)
    rank_score = min(1.0, float(singular_values[-1] / singular_values[0]) * 2)
    # The target is deliberately long and narrow, so never label it perfectly
    # conditioned merely because normalization gives full algebraic rank.
    return round(min(0.7, rank_score), 6)


def _quality_components(
    landmarks: list[WicketPoseLandmark],
    intrinsics: CameraIntrinsics,
    geometry_score: float,
    reprojection_quality: float,
    plausibility: float,
) -> CameraPoseQualityComponents:
    source_scores = {
        "manual": 1.0,
        "manually_adjusted": 0.9,
        "detected": 0.65,
        "inferred": 0.3,
    }
    landmark_quality = (
        sum(
            source_scores[item.source]
            * (0.6 if item.visibility == "uncertain" else 1.0)
            for item in landmarks
        )
        / len(landmarks)
        if landmarks
        else 0.0
    )
    landmark_coverage = min(1.0, len(landmarks) / 12)
    if not all(
        any(
            item.wicket_end == wicket_end and item.point_type == point_type
            for item in landmarks
        )
        for wicket_end in WICKET_ENDS
        for point_type in POINT_TYPES
    ):
        landmark_coverage *= 0.5
    intrinsics_quality = {
        "calibrated_device_profile": 1.0,
        "manually_provided": 0.7,
        "metadata_estimated": 0.5,
        "heuristic_estimated": 0.35,
    }[intrinsics.source]
    overall = (
        landmark_quality * 0.2
        + landmark_coverage * 0.15
        + reprojection_quality * 0.25
        + intrinsics_quality * 0.15
        + geometry_score * 0.1
        + plausibility * 0.15
    )
    return CameraPoseQualityComponents(
        landmark_quality=round(landmark_quality, 6),
        landmark_coverage=round(landmark_coverage, 6),
        reprojection_quality=round(reprojection_quality, 6),
        intrinsics_quality=round(intrinsics_quality, 6),
        geometry_condition=round(geometry_score, 6),
        pose_plausibility=round(plausibility, 6),
        overall_pose_quality=round(overall, 6),
    )


def _empty_solution(
    solver: str,
    used: list[WicketPoseLandmark],
    unavailable: list[str],
    warnings: list[str],
    rejection_reasons: list[str],
) -> CameraPoseSolution:
    return CameraPoseSolution(
        solved=False,
        accepted=False,
        solver_method=solver,
        landmark_count=len(used),
        used_landmark_ids=[item.id for item in used],
        unavailable_landmark_ids=unavailable,
        warnings=_deduplicate(warnings),
        rejection_reasons=_deduplicate(rejection_reasons),
    )


def _estimated_intrinsics(
    image_width: int,
    image_height: int,
) -> CameraIntrinsics:
    horizontal_fov_radians = math.radians(ASSUMED_HORIZONTAL_FOV_DEGREES)
    focal_length = image_width / (2 * math.tan(horizontal_fov_radians / 2))
    cx = image_width / 2
    cy = image_height / 2
    return CameraIntrinsics(
        image_width=image_width,
        image_height=image_height,
        fx=round(focal_length, 6),
        fy=round(focal_length, 6),
        cx=round(cx, 6),
        cy=round(cy, 6),
        intrinsic_matrix=[
            [round(focal_length, 6), 0.0, round(cx, 6)],
            [0.0, round(focal_length, 6), round(cy, 6)],
            [0.0, 0.0, 1.0],
        ],
        distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        source="heuristic_estimated",
        quality="low",
        assumed_horizontal_fov_degrees=ASSUMED_HORIZONTAL_FOV_DEGREES,
        distortion_model_source="not_calibrated",
        assumptions=[
            "Square pixels: fx equals fy.",
            "Principal point is the image centre.",
            "Horizontal field of view is assumed to be 60 degrees.",
            "Lens distortion is assumed zero, not calibrated.",
        ],
    )


def _validated_intrinsics(
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.asarray(intrinsics.intrinsic_matrix, dtype=np.float64)
    expected = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(
        intrinsics.distortion_coefficients,
        dtype=np.float64,
    )
    if (
        camera_matrix.shape != (3, 3)
        or not np.isfinite(camera_matrix).all()
        or not np.allclose(camera_matrix, expected, atol=1e-6)
        or not np.isfinite(distortion).all()
    ):
        raise VideoAnalysisServiceError(
            "Camera intrinsics are internally inconsistent.",
            status_code=422,
        )
    if (
        intrinsics.distortion_model_source == "not_calibrated"
        and not np.allclose(distortion, 0.0, atol=1e-12)
    ):
        raise VideoAnalysisServiceError(
            "Uncalibrated lens distortion must use the explicit zero assumption.",
            status_code=422,
        )
    if (
        intrinsics.source == "calibrated_device_profile"
        and not intrinsics.device_profile_id
    ):
        raise VideoAnalysisServiceError(
            "A calibrated camera profile requires a device profile ID.",
            status_code=422,
        )
    return camera_matrix, distortion


def _save_overlay(
    image: Image.Image,
    result: WicketCameraPoseResult,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    output_path: Path,
) -> None:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    base_color = (80, 220, 255, 245)
    top_color = (255, 202, 104, 245)
    projected_color = (213, 255, 107, 245)
    outlier_color = (255, 90, 110, 245)
    radius = max(5, round(min(image.size) * 0.006))
    diagnostic_by_id = {
        item.landmark_id: item
        for item in result.camera_pose.reprojection_diagnostics
    }
    for landmark in result.landmarks:
        if landmark.visibility in {"occluded", "unavailable"}:
            continue
        color = base_color if landmark.point_type == "base" else top_color
        x, y = landmark.pixel_x, landmark.pixel_y
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=max(2, radius // 3),
        )
        draw.text((x + radius + 2, y - radius), _short_label(landmark), fill=color)
        diagnostic = diagnostic_by_id.get(landmark.id)
        if diagnostic is not None:
            projected = (
                diagnostic.projected_pixel_x,
                diagnostic.projected_pixel_y,
            )
            reprojection_color = (
                projected_color if diagnostic.ransac_inlier else outlier_color
            )
            draw.line((x, y, *projected), fill=reprojection_color, width=2)
            px, py = projected
            draw.line(
                (px - radius, py, px + radius, py),
                fill=reprojection_color,
                width=2,
            )
            draw.line(
                (px, py - radius, px, py + radius),
                fill=reprojection_color,
                width=2,
            )

    if result.camera_pose.solved:
        _draw_projected_wicket_geometry(
            draw,
            result,
            camera_matrix,
            distortion,
            projected_color,
        )

    reliable = result.status in {"ready", "usable"}
    if reliable and result.camera_pose.rotation_vector is not None:
        _draw_pitch_centreline(
            draw,
            result,
            camera_matrix,
            distortion,
        )
    banner_height = max(72, round(image.height * 0.095))
    draw.rectangle((0, 0, image.width, banner_height), fill=(8, 12, 15, 215))
    rmse = result.camera_pose.reprojection_rmse_px
    draw.text(
        (16, 12),
        (
            f"CAMERA POSE: {result.status.upper()}  |  "
            f"RMSE: {rmse:.2f}px"
            if rmse is not None
            else f"CAMERA POSE: {result.status.upper()}  |  RMSE: unavailable"
        ),
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (16, 36),
        (
            f"Intrinsics: {result.camera_intrinsics.source}  |  "
            "circles=observed crosses=reprojected"
        ),
        fill=(210, 220, 230, 255),
    )
    if not reliable:
        draw.text(
            (16, 57),
            "POSE NOT RELIABLE FOR WORLD MAPPING",
            fill=(255, 170, 120, 255),
        )
    try:
        overlay.convert("RGB").save(output_path, format="JPEG", quality=93)
    except OSError as exc:
        raise VideoAnalysisServiceError(
            "Camera-pose overlay could not be saved.",
            status_code=500,
        ) from exc


def _draw_projected_wicket_geometry(
    draw: ImageDraw.ImageDraw,
    result: WicketCameraPoseResult,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    color: tuple[int, int, int, int],
) -> None:
    assert result.camera_pose.rotation_vector is not None
    assert result.camera_pose.translation_vector is not None
    rotation = np.asarray(result.camera_pose.rotation_vector, dtype=np.float64)
    translation = np.asarray(result.camera_pose.translation_vector, dtype=np.float64)
    by_id = {item.id: item for item in result.landmarks}
    for wicket_end in WICKET_ENDS:
        for stump_position in STUMP_POSITIONS:
            base = by_id[f"{wicket_end}_{stump_position}_stump_base"]
            top = by_id[f"{wicket_end}_{stump_position}_stump_top"]
            projected, _ = cv2.projectPoints(
                np.asarray(
                    [
                        [base.world_x_m, base.world_y_m, base.world_z_m],
                        [top.world_x_m, top.world_y_m, top.world_z_m],
                    ],
                    dtype=np.float64,
                ),
                rotation,
                translation,
                camera_matrix,
                distortion,
            )
            points = [tuple(map(float, point)) for point in projected.reshape(-1, 2)]
            draw.line(points, fill=color, width=2)
        for point_type in POINT_TYPES:
            objects = [
                by_id[f"{wicket_end}_{position}_stump_{point_type}"]
                for position in STUMP_POSITIONS
            ]
            projected, _ = cv2.projectPoints(
                np.asarray(
                    [
                        [item.world_x_m, item.world_y_m, item.world_z_m]
                        for item in objects
                    ],
                    dtype=np.float64,
                ),
                rotation,
                translation,
                camera_matrix,
                distortion,
            )
            points = [tuple(map(float, point)) for point in projected.reshape(-1, 2)]
            draw.line(points, fill=color, width=2)


def _draw_pitch_centreline(
    draw: ImageDraw.ImageDraw,
    result: WicketCameraPoseResult,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> None:
    projected, _ = cv2.projectPoints(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [result.pitch_geometry.pitch_length_m, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(result.camera_pose.rotation_vector, dtype=np.float64),
        np.asarray(result.camera_pose.translation_vector, dtype=np.float64),
        camera_matrix,
        distortion,
    )
    points = [tuple(map(float, point)) for point in projected.reshape(-1, 2)]
    draw.line(points, fill=(255, 255, 255, 170), width=2)


def _top_guess_y(wicket_end: str, base_y: float) -> float:
    # ponytail: only a visible starter; there is no wicket keypoint detector.
    height = 0.20 if wicket_end == "bowler" else 0.10
    return max(0.0, min(1.0, base_y - height))


def _landmark_label(
    wicket_end: str,
    stump_position: str,
    point_type: str,
) -> str:
    return (
        f"{wicket_end.title()} {stump_position.title()} Stump "
        f"{'Ground Contact' if point_type == 'base' else 'Top of Body'}"
    )


def _short_label(landmark: WicketPoseLandmark) -> str:
    return (
        f"{landmark.wicket_end[0].upper()}"
        f"{'B' if landmark.point_type == 'base' else 'T'}-"
        f"{landmark.stump_position[0].upper()}"
    )


def _world_depth(
    rotation: np.ndarray,
    translation: np.ndarray,
    point: np.ndarray,
) -> float:
    return float((rotation @ point.reshape(3, 1) + translation.reshape(3, 1))[2])


def _shared_geometry(
    geometry: CricketPitchGeometry,
) -> CricketPitchDimensions:
    configured = CricketPitchDimensions(
        pitch_length_m=geometry.pitch_length_m,
        wicket_width_m=geometry.wicket_width_m,
        wicket_height_m=geometry.wicket_height_m,
        stump_diameter_m=geometry.stump_diameter_m,
        pitch_width_m=geometry.pitch_width_m,
        popping_crease_distance_m=geometry.popping_crease_distance_m,
    )
    try:
        configured.validate()
    except ValueError as exc:
        raise VideoAnalysisServiceError(str(exc), status_code=422) from exc
    return configured


def _reference_path(analysis_id: str) -> Path:
    return (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "calibration"
        / "reference_frame.jpg"
    )


def _open_reference_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise VideoAnalysisServiceError(
            "Calibration reference frame is missing.",
            status_code=404,
        )
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
            return image
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Calibration reference frame could not be opened.",
            status_code=500,
        ) from exc


def _existing_created_at(path: Path) -> datetime | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("created_at")
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoAnalysisServiceError(
            "Camera-pose JSON could not be saved.",
            status_code=500,
        ) from exc


def _update_analysis_metadata(
    analysis_id: str,
    result: WicketCameraPoseResult,
    updated_at: datetime,
) -> None:
    metadata_path = (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "reports"
        / "analysis_metadata.json"
    )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "camera_pose_status": result.status,
                "camera_pose_quality": (
                    result.quality.overall_pose_quality
                ),
                "camera_pose_url": result.camera_pose_url,
                "camera_pose_overlay_url": result.camera_pose_overlay_url,
                "camera_intrinsics_source": result.camera_intrinsics.source,
                "camera_pose_reprojection_rmse_px": (
                    result.camera_pose.reprojection_rmse_px
                ),
                "calibration_mode_used": result.calibration_mode,
                "updated_at": _iso(updated_at),
            }
        )
        temporary_path = metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoAnalysisServiceError(
            "Analysis metadata could not be updated for camera pose.",
            status_code=500,
        ) from exc


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [
        [round(float(value), 12) for value in row]
        for row in matrix.tolist()
    ]


def _vector_list(vector: np.ndarray) -> list[float]:
    return [
        round(float(value), 12)
        for value in np.asarray(vector).reshape(-1).tolist()
    ]


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
