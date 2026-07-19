"""Calibration v2A: cricket-world ground-plane geometry and homography."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CricketPitchDimensions,
    LEFT_RIGHT_CONVENTION,
    standard_ground_reference_world_points,
    stump_base_world_points,
    virtual_pitch_ground_lines,
)

from ..schemas.video_analysis import (
    CalibrationCoordinateSystem,
    CalibrationLandmark,
    CalibrationLandmarkInput,
    CalibrationLandmarkSet,
    CalibrationQualityV2,
    CalibrationV2ConfirmRequest,
    CalibrationV2FutureCameraPoseFields,
    CalibrationV2InitialiseResponse,
    CalibrationV2Result,
    CricketPitchGeometry,
    GroundHomographyResult,
    GroundPoint2D,
    ImagePixelPoint,
    ProjectedPitchLine,
    ReprojectionDiagnostic,
    VirtualPitchOverlayGeometry,
    WicketCalibration,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .video_calibration_service import (
    detect_video_calibration,
    load_video_calibration,
)


CALIBRATION_V2_FILENAME = "calibration_v2.json"
CALIBRATION_V2_OVERLAY_FILENAME = "calibration_v2_overlay.jpg"
PRIMARY_LANDMARK_IDS = (
    "bowler_left_stump_base",
    "bowler_middle_stump_base",
    "bowler_right_stump_base",
    "striker_left_stump_base",
    "striker_middle_stump_base",
    "striker_right_stump_base",
)
LANDMARK_LABELS = {
    "bowler_left_stump_base": "Bowler Left Stump Base",
    "bowler_middle_stump_base": "Bowler Middle Stump Base",
    "bowler_right_stump_base": "Bowler Right Stump Base",
    "striker_left_stump_base": "Striker Left Stump Base",
    "striker_middle_stump_base": "Striker Middle Stump Base",
    "striker_right_stump_base": "Striker Right Stump Base",
}
MIN_IMAGE_HULL_COVERAGE = 0.0005
MIN_WICKET_SEPARATION_NORMALIZED = 0.04
MIN_STUMP_SEPARATION_NORMALIZED = 0.002
# Initial conservative full-pitch gate: require at least four wide, known
# metric ground controls (the UI supplies standard crease/edge references),
# meaningful hull coverage in both coordinate systems, and reject a projection
# when too many defined pitch vertices leave the image. These are inspectable
# v2A.1 policy thresholds, not accuracy guarantees.
MIN_ADDITIONAL_METRIC_REFERENCES_FOR_FULL_PITCH = 4
MIN_WORLD_COVERAGE_FOR_FULL_PITCH = 0.35
MIN_IMAGE_COVERAGE_FOR_FULL_PITCH = 0.02
MAX_PROJECTION_OUTSIDE_FRACTION = 0.4
# RANSAC is reserved for the overdetermined wide-reference case. Four pixels is
# an initial sub-percent threshold for the current reference-frame resolutions.
RANSAC_MIN_CORRESPONDENCES = 8
RANSAC_REPROJECTION_THRESHOLD_PX = 4.0
MIN_RANSAC_INLIER_RATIO = 0.75
IMAGE_DUPLICATE_TOLERANCE_PX = 0.75
WORLD_DUPLICATE_TOLERANCE_M = 0.005
MAX_IMAGE_ROUND_TRIP_RMSE_PX = 0.05
MAX_GROUND_ROUND_TRIP_RMSE_M = 0.0001
WELL_CONDITIONED_DLT_LIMIT = 100.0
WEAK_DLT_LIMIT = 1000.0
# Resolution-relative reprojection thresholds. Reprojection alone is not a
# proof of camera quality because these same points fit the homography.
EXCELLENT_RMSE_DIAGONAL_RATIO = 0.003
GOOD_RMSE_DIAGONAL_RATIO = 0.008
USABLE_RMSE_DIAGONAL_RATIO = 0.02


def initialise_video_calibration_v2(
    analysis_id: str,
) -> CalibrationV2InitialiseResponse:
    analysis = load_video_analysis(analysis_id)
    reference_path = _reference_path(analysis_id)
    image = _open_reference_image(reference_path)
    image_width, image_height = image.size
    pitch_geometry = CricketPitchGeometry()
    geometry = _shared_geometry(pitch_geometry)
    warnings = [
        (
            "Automatic landmarks are approximate. Adjust all six stump-base "
            "markers before confirming."
        )
    ]

    bowler_wicket: WicketCalibration | None = None
    striker_wicket: WicketCalibration | None = None
    try:
        legacy = load_video_calibration(analysis_id)
        # Legacy non-striker is the bowler end in the established behind-bowler
        # camera convention. The UI can swap ends when the camera is reversed.
        bowler_wicket = legacy.non_striker_wicket
        striker_wicket = legacy.striker_wicket
        warnings.append(
            "Initial guesses were inferred from the confirmed legacy wicket boxes."
        )
    except VideoAnalysisServiceError:
        try:
            detected = detect_video_calibration(analysis_id)
            bowler_wicket = detected.provisional_non_striker_wicket
            striker_wicket = detected.provisional_striker_wicket
            if bowler_wicket is not None or striker_wicket is not None:
                warnings.append(
                    "Initial guesses were inferred from current wicket detections."
                )
        except VideoAnalysisServiceError:
            pass

    if bowler_wicket is None:
        warnings.append(
            "Bowler-end wicket was not available; a manual starter layout was used."
        )
    if striker_wicket is None:
        warnings.append(
            "Striker-end wicket was not available; a manual starter layout was used."
        )
    bowler_box = (
        bowler_wicket.box
        if bowler_wicket is not None
        else _starter_wicket_box("bowler")
    )
    striker_box = (
        striker_wicket.box
        if striker_wicket is not None
        else _starter_wicket_box("striker")
    )
    world_points = stump_base_world_points(geometry)
    landmarks = [
        *_landmarks_from_wicket_box(
            "bowler",
            bowler_box.x,
            bowler_box.y,
            bowler_box.width,
            bowler_box.height,
            image_width,
            image_height,
            world_points,
        ),
        *_landmarks_from_wicket_box(
            "striker",
            striker_box.x,
            striker_box.y,
            striker_box.width,
            striker_box.height,
            image_width,
            image_height,
            world_points,
        ),
    ]
    return CalibrationV2InitialiseResponse(
        success=True,
        status="initialised",
        analysis_id=analysis_id,
        reference_frame_url=analysis.reference_frame_url,
        image_width=image_width,
        image_height=image_height,
        pitch_geometry=pitch_geometry,
        landmarks=landmarks,
        image_left_right_convention="image_left_is_world_left",
        warnings=warnings,
        message="Calibration v2 stump-base landmarks initialised.",
    )


def confirm_video_calibration_v2(
    analysis_id: str,
    request: CalibrationV2ConfirmRequest,
) -> CalibrationV2Result:
    if request.analysis_id != analysis_id:
        raise VideoAnalysisServiceError(
            "Calibration v2 analysis ID does not match the URL.",
            status_code=400,
        )
    analysis = load_video_analysis(analysis_id)
    reference_path = _reference_path(analysis_id)
    image = _open_reference_image(reference_path)
    image_width, image_height = image.size
    geometry = _shared_geometry(request.pitch_geometry)
    primary_inputs, optional_inputs = _validate_landmark_inputs(
        request.landmarks,
        request.image_left_right_convention,
    )
    if request.ground_reference_mode == "skip":
        optional_inputs = []
    world_points = stump_base_world_points(geometry)
    primary_landmarks = [
        _confirmed_primary_landmark(
            landmark,
            world_points,
            image_width,
            image_height,
        )
        for landmark in primary_inputs
    ]
    optional_landmarks = [
        _confirmed_optional_landmark(
            landmark,
            geometry,
            image_width,
            image_height,
        )
        for landmark in optional_inputs
    ]
    all_landmarks = [*primary_landmarks, *optional_landmarks]
    geometry_warnings, wicket_order_valid = _geometry_consistency_warnings(
        primary_landmarks,
        request.image_left_right_convention,
        image_width,
        image_height,
    )
    homography, quality = _calculate_ground_homography(
        all_landmarks,
        primary_landmarks,
        image_width,
        image_height,
        geometry,
        wicket_order_valid,
        geometry_warnings,
        request.landmark_semantics_confirmed,
    )
    if not homography.transform_available:
        status = (
            "unstable"
            if quality.geometry_condition == "unstable"
            else "insufficient_geometry"
        )
        projection_mode = "landmarks_only"
    elif quality.full_pitch_projection_allowed:
        status = "ready"
        projection_mode = "full_pitch"
    else:
        status = "weak"
        projection_mode = "local_debug"
    projected_geometry = _project_virtual_pitch_geometry(
        homography.ground_to_image_homography,
        geometry,
        projection_mode,
    )
    calibration_dir = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration"
    calibration_path = calibration_dir / CALIBRATION_V2_FILENAME
    overlay_path = calibration_dir / CALIBRATION_V2_OVERLAY_FILENAME
    now = datetime.now(timezone.utc)
    created_at = _existing_created_at(calibration_path) or now
    calibration_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{CALIBRATION_V2_FILENAME}"
    )
    overlay_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{CALIBRATION_V2_OVERLAY_FILENAME}"
    )
    messages = {
        "ready": (
            "Calibration v2A.1 ground-plane mapping is ready for validated "
            "world-geometry projection."
        ),
        "weak": (
            "A numerical ground transform exists, but the metric landmark "
            "spread is too weak for a trustworthy full-pitch projection."
        ),
        "unstable": (
            "Calibration landmarks were saved, but the ground transform is "
            "unstable and was rejected."
        ),
        "insufficient_geometry": (
            "Calibration landmarks were saved, but known metric geometry is "
            "insufficient for a reliable ground transform."
        ),
    }
    result = CalibrationV2Result(
        success=status == "ready",
        status=status,
        schema_version="2.1",
        analysis_id=analysis_id,
        calibration_mode="ground_plane",
        coordinate_system=CalibrationCoordinateSystem(
            units="metres",
            origin="bowler_wicket_centre",
            x_axis="toward_striker",
            y_axis="lateral",
            z_axis="up",
            left_right_convention=LEFT_RIGHT_CONVENTION,
            image_left_right_convention=request.image_left_right_convention,
        ),
        pitch_geometry=request.pitch_geometry,
        landmark_set=CalibrationLandmarkSet(
            primary_stump_bases=primary_landmarks,
            optional_ground_landmarks=optional_landmarks,
        ),
        homography=homography,
        quality=quality,
        virtual_pitch_overlay_geometry=projected_geometry,
        calibration_v2_url=calibration_url,
        calibration_v2_overlay_url=overlay_url,
        reference_frame_url=analysis.reference_frame_url,
        image_width=image_width,
        image_height=image_height,
        landmark_semantics_confirmed=request.landmark_semantics_confirmed,
        ground_reference_mode=request.ground_reference_mode,
        ground_transform_reason=(
            "trusted_metric_ground_references_not_available"
            if request.ground_reference_mode == "skip"
            else None
        ),
        created_at=created_at,
        updated_at=now,
        user_note=request.user_note,
        future_camera_pose=CalibrationV2FutureCameraPoseFields(),
        message=messages[status],
    )
    _save_calibration_v2_overlay(image, result, overlay_path)
    _write_json(calibration_path, result.model_dump(mode="json"))
    _update_analysis_metadata(
        analysis_id,
        result,
        now,
    )
    return result


def load_video_calibration_v2(analysis_id: str) -> CalibrationV2Result:
    load_video_analysis(analysis_id)
    calibration_dir = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration"
    calibration_path = calibration_dir / CALIBRATION_V2_FILENAME
    overlay_path = calibration_dir / CALIBRATION_V2_OVERLAY_FILENAME
    if not calibration_path.is_file():
        raise VideoAnalysisServiceError(
            "Calibration v2 has not been confirmed.",
            status_code=404,
        )
    try:
        result = CalibrationV2Result.model_validate(
            json.loads(calibration_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored Calibration v2 is unavailable.",
            status_code=500,
        ) from exc
    if (
        result.analysis_id != analysis_id
        or not _reference_path(analysis_id).is_file()
        or not overlay_path.is_file()
    ):
        raise VideoAnalysisServiceError(
            "Stored Calibration v2 files are incomplete.",
            status_code=404,
        )
    return result


def image_point_to_pitch_ground(
    image_to_ground_homography: Sequence[Sequence[float]],
    u: float,
    v: float,
) -> tuple[float, float]:
    return _transform_homogeneous_point(
        image_to_ground_homography,
        u,
        v,
    )


def pitch_ground_point_to_image(
    ground_to_image_homography: Sequence[Sequence[float]],
    x_m: float,
    y_m: float,
) -> tuple[float, float]:
    return _transform_homogeneous_point(
        ground_to_image_homography,
        x_m,
        y_m,
    )


def _validate_landmark_inputs(
    landmarks: list[CalibrationLandmarkInput],
    image_convention: str,
) -> tuple[list[CalibrationLandmarkInput], list[CalibrationLandmarkInput]]:
    ids = [landmark.id for landmark in landmarks]
    if len(ids) != len(set(ids)):
        raise VideoAnalysisServiceError(
            "Calibration landmarks contain duplicate IDs.",
            status_code=422,
        )
    primary_by_id = {
        landmark.id: landmark
        for landmark in landmarks
        if landmark.id in PRIMARY_LANDMARK_IDS
    }
    if set(primary_by_id) != set(PRIMARY_LANDMARK_IDS):
        raise VideoAnalysisServiceError(
            "Calibration v2 requires all six stump-base landmarks.",
            status_code=422,
        )
    primary = [primary_by_id[landmark_id] for landmark_id in PRIMARY_LANDMARK_IDS]
    for landmark in primary:
        expected_end = "bowler" if landmark.id.startswith("bowler_") else "striker"
        if (
            landmark.wicket_end != expected_end
            or landmark.landmark_type != "stump_base"
        ):
            raise VideoAnalysisServiceError(
                f"Landmark {landmark.id} has an invalid wicket assignment.",
                status_code=422,
            )
    optional = [
        landmark
        for landmark in landmarks
        if landmark.id not in PRIMARY_LANDMARK_IDS
    ]
    for landmark in optional:
        if (
            landmark.landmark_type != "ground_control"
            or landmark.world_x_m is None
            or landmark.world_y_m is None
            or landmark.world_z_m not in (None, 0, 0.0)
        ):
            raise VideoAnalysisServiceError(
                (
                    f"Optional landmark {landmark.id} requires an explicit "
                    "ground-plane X/Y correspondence with Z = 0."
                ),
                status_code=422,
            )
    if image_convention not in {
        "image_left_is_world_left",
        "image_left_is_world_right",
    }:
        raise VideoAnalysisServiceError(
            "Image left/right convention is invalid.",
            status_code=422,
        )
    return primary, optional


def _confirmed_primary_landmark(
    landmark: CalibrationLandmarkInput,
    world_points: dict[str, tuple[float, float, float]],
    image_width: int,
    image_height: int,
) -> CalibrationLandmark:
    world_x, world_y, world_z = world_points[landmark.id]
    return CalibrationLandmark(
        **landmark.model_dump(
            exclude={"world_x_m", "world_y_m", "world_z_m"}
        ),
        pixel_x=round(landmark.normalized_x * image_width, 6),
        pixel_y=round(landmark.normalized_y * image_height, 6),
        world_x_m=world_x,
        world_y_m=world_y,
        world_z_m=world_z,
    )


def _confirmed_optional_landmark(
    landmark: CalibrationLandmarkInput,
    geometry: CricketPitchDimensions,
    image_width: int,
    image_height: int,
) -> CalibrationLandmark:
    world_x = float(landmark.world_x_m)
    world_y = float(landmark.world_y_m)
    standard_references = standard_ground_reference_world_points(geometry)
    if landmark.id in standard_references:
        expected_x, expected_y, _ = standard_references[landmark.id]
        if (
            abs(world_x - expected_x) > 1e-6
            or abs(world_y - expected_y) > 1e-6
        ):
            raise VideoAnalysisServiceError(
                (
                    f"Ground reference {landmark.id} does not match its "
                    "configured crease/pitch-edge world coordinate."
                ),
                status_code=422,
            )
        world_x, world_y = expected_x, expected_y
    if (
        world_x < -5
        or world_x > geometry.pitch_length_m + 5
        or abs(world_y) > geometry.pitch_width_m * 2
    ):
        raise VideoAnalysisServiceError(
            f"Optional landmark {landmark.id} has implausible world coordinates.",
            status_code=422,
        )
    return CalibrationLandmark(
        **landmark.model_dump(
            exclude={"world_x_m", "world_y_m", "world_z_m"}
        ),
        pixel_x=round(landmark.normalized_x * image_width, 6),
        pixel_y=round(landmark.normalized_y * image_height, 6),
        world_x_m=world_x,
        world_y_m=world_y,
        world_z_m=0.0,
    )


def _calculate_ground_homography(
    landmarks: list[CalibrationLandmark],
    primary_landmarks: list[CalibrationLandmark],
    image_width: int,
    image_height: int,
    geometry: CricketPitchDimensions,
    wicket_order_valid: bool,
    initial_warnings: list[str],
    landmark_semantics_confirmed: bool,
) -> tuple[GroundHomographyResult, CalibrationQualityV2]:
    warnings = [*initial_warnings]
    optional_landmarks = [
        landmark
        for landmark in landmarks
        if landmark.id not in PRIMARY_LANDMARK_IDS
    ]
    landmark_sources = _landmark_source_counts(landmarks)
    used_landmark_ids = [landmark.id for landmark in landmarks]
    source = np.array(
        [[landmark.pixel_x, landmark.pixel_y] for landmark in landmarks],
        dtype=np.float64,
    )
    destination = np.array(
        [[landmark.world_x_m, landmark.world_y_m] for landmark in landmarks],
        dtype=np.float64,
    )
    diagonal = math.hypot(image_width, image_height)
    image_coverage = _convex_hull_coverage(source, image_width, image_height)
    world_coverage = _world_hull_coverage(destination, geometry)
    landmark_spread_score = _landmark_spread_score(
        image_coverage,
        world_coverage,
    )
    landmark_coverage = min(1.0, len(primary_landmarks) / 6)
    condition_number, design_rank = _normalized_dlt_condition(
        landmarks,
        image_width,
        image_height,
        geometry,
    )
    geometry_condition = _geometry_condition(condition_number, design_rank)
    end_separation = _wicket_end_separation(primary_landmarks)
    minimum_wicket_width = _minimum_wicket_image_width(primary_landmarks)
    insufficient_reasons: list[str] = []
    if not landmark_semantics_confirmed:
        insufficient_reasons.append(
            (
                "Landmark semantics were not confirmed. Confirm that stump "
                "markers are ground-contact points and any ground references "
                "are the named crease/pitch-edge intersections."
            )
        )
    if len(landmarks) < 4:
        insufficient_reasons.append("At least four ground correspondences are required.")
    duplicate_image_pairs = _duplicate_point_pairs(
        landmarks,
        coordinate_getter=lambda item: (item.pixel_x, item.pixel_y),
        tolerance=IMAGE_DUPLICATE_TOLERANCE_PX,
    )
    if duplicate_image_pairs:
        insufficient_reasons.append(
            "Duplicate or overlapping image correspondences: "
            + ", ".join(duplicate_image_pairs)
            + "."
        )
    duplicate_world_pairs = _duplicate_point_pairs(
        landmarks,
        coordinate_getter=lambda item: (
            item.world_x_m,
            item.world_y_m,
        ),
        tolerance=WORLD_DUPLICATE_TOLERANCE_M,
    )
    if duplicate_world_pairs:
        insufficient_reasons.append(
            "Duplicate metric world correspondences: "
            + ", ".join(duplicate_world_pairs)
            + "."
        )
    if image_coverage < MIN_IMAGE_HULL_COVERAGE:
        insufficient_reasons.append(
            "Landmarks cover too little of the source image for a stable transform."
        )
    if end_separation < MIN_WICKET_SEPARATION_NORMALIZED:
        insufficient_reasons.append(
            "Bowler and striker wicket landmarks are too close in the image."
        )
    if minimum_wicket_width < MIN_STUMP_SEPARATION_NORMALIZED * 2:
        insufficient_reasons.append(
            "At least one wicket has near-overlapping stump landmarks."
        )
    if design_rank < 8:
        insufficient_reasons.append(
            "Ground correspondences are mathematically degenerate."
        )
    if geometry_condition == "unstable":
        insufficient_reasons.append(
            "The landmark arrangement is poorly conditioned; add wider ground controls if available."
        )
    warnings.extend(insufficient_reasons)
    if insufficient_reasons:
        return _unavailable_homography(), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            geometry_condition,
            warnings,
            primary_landmarks,
            used_landmark_ids,
            landmark_sources,
        )

    use_ransac = (
        len(landmarks) >= RANSAC_MIN_CORRESPONDENCES
        and len(optional_landmarks)
        >= MIN_ADDITIONAL_METRIC_REFERENCES_FOR_FULL_PITCH
    )
    estimation_method = "ransac" if use_ransac else "direct"
    try:
        ground_to_image, inlier_mask = cv2.findHomography(
            destination,
            source,
            method=cv2.RANSAC if use_ransac else 0,
            ransacReprojThreshold=RANSAC_REPROJECTION_THRESHOLD_PX,
        )
    except cv2.error:
        ground_to_image = None
        inlier_mask = None
    if ground_to_image is None or ground_to_image.shape != (3, 3):
        warnings.append("OpenCV could not calculate a ground homography.")
        return _unavailable_homography(), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            "insufficient",
            warnings,
            primary_landmarks,
            used_landmark_ids,
            landmark_sources,
        )
    ground_to_image = ground_to_image.astype(np.float64)
    if abs(ground_to_image[2, 2]) <= 1e-12:
        warnings.append("Calculated homography has an invalid scale.")
        return _unavailable_homography(), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            "unstable",
            warnings,
            primary_landmarks,
            used_landmark_ids,
            landmark_sources,
        )
    ground_to_image /= ground_to_image[2, 2]
    ground_determinant = float(np.linalg.det(ground_to_image))
    if not math.isfinite(ground_determinant) or abs(ground_determinant) < 1e-12:
        warnings.append("Calculated homography is singular or near-singular.")
        return _unavailable_homography(), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            "unstable",
            warnings,
            primary_landmarks,
            used_landmark_ids,
            landmark_sources,
        )
    try:
        image_to_ground = np.linalg.inv(ground_to_image)
        image_to_ground /= image_to_ground[2, 2]
    except (np.linalg.LinAlgError, FloatingPointError):
        warnings.append("Calculated homography could not be inverted.")
        return _unavailable_homography(), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            "unstable",
            warnings,
            primary_landmarks,
            used_landmark_ids,
            landmark_sources,
        )
    if not np.isfinite(image_to_ground).all() or not np.isfinite(ground_to_image).all():
        warnings.append("Calculated homography contains invalid values.")
        return _unavailable_homography(), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            "unstable",
            warnings,
            primary_landmarks,
            used_landmark_ids,
            landmark_sources,
        )

    inlier_flags = (
        [bool(value) for value in inlier_mask.reshape(-1)]
        if inlier_mask is not None
        else [True] * len(landmarks)
    )
    inlier_ids = [
        landmark.id
        for landmark, is_inlier in zip(
            landmarks,
            inlier_flags,
            strict=True,
        )
        if is_inlier
    ]
    inlier_count = len(inlier_ids)
    inlier_ratio = inlier_count / len(landmarks)
    diagnostics = _reprojection_diagnostics(
        landmarks,
        ground_to_image,
        dict(zip(used_landmark_ids, inlier_flags, strict=True)),
    )
    if use_ransac and (
        inlier_count < 4 or inlier_ratio < MIN_RANSAC_INLIER_RATIO
    ):
        warnings.append(
            (
                "RANSAC rejected too many metric correspondences "
                f"({inlier_count}/{len(landmarks)} inliers)."
            )
        )
        outlier_ids = [
            landmark.id
            for landmark, is_inlier in zip(
                landmarks,
                inlier_flags,
                strict=True,
            )
            if not is_inlier
        ]
        return _rejected_ransac_homography(
            condition_number,
            inlier_ids,
        ), _quality_without_transform(
            landmark_coverage,
            len(landmarks),
            len(optional_landmarks),
            image_coverage,
            world_coverage,
            landmark_spread_score,
            wicket_order_valid,
            condition_number,
            "unstable",
            warnings,
            primary_landmarks,
            inlier_ids,
            landmark_sources,
            diagnostics=diagnostics,
            ignored_landmark_ids=outlier_ids,
        )

    errors = [
        diagnostic.error_px
        for diagnostic in diagnostics
        if diagnostic.ransac_inlier is not False
    ]
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    maximum_error = max(errors)
    median_error = float(np.median(np.asarray(errors, dtype=np.float64)))
    normalized_rmse = rmse / diagonal
    image_round_trip_rmse, ground_round_trip_rmse = _round_trip_errors(
        landmarks,
        image_to_ground,
        ground_to_image,
    )
    projection_outside_fraction = _full_projection_outside_fraction(
        ground_to_image,
        geometry,
        image_width,
        image_height,
    )
    if not wicket_order_valid:
        warnings.append(
            "Left/middle/right stump ordering does not match the selected image convention."
        )
    if normalized_rmse > USABLE_RMSE_DIAGONAL_RATIO:
        warnings.append(
            "Reprojection error is excessive for this image resolution."
        )
    full_pitch_projection_allowed = all(
        (
            len(optional_landmarks)
            >= MIN_ADDITIONAL_METRIC_REFERENCES_FOR_FULL_PITCH,
            world_coverage >= MIN_WORLD_COVERAGE_FOR_FULL_PITCH,
            image_coverage >= MIN_IMAGE_COVERAGE_FOR_FULL_PITCH,
            geometry_condition == "well_conditioned",
            wicket_order_valid,
            normalized_rmse <= GOOD_RMSE_DIAGONAL_RATIO,
            image_round_trip_rmse <= MAX_IMAGE_ROUND_TRIP_RMSE_PX,
            ground_round_trip_rmse <= MAX_GROUND_ROUND_TRIP_RMSE_M,
            projection_outside_fraction
            <= MAX_PROJECTION_OUTSIDE_FRACTION,
            not use_ransac or inlier_ratio >= MIN_RANSAC_INLIER_RATIO,
        )
    )
    if (
        len(optional_landmarks)
        < MIN_ADDITIONAL_METRIC_REFERENCES_FOR_FULL_PITCH
    ):
        warnings.append(
            (
                "Six stump bases constrain only two narrow wicket lines. Add "
                "four confirmed crease/pitch-edge intersections before using "
                "a full-pitch projection."
            )
        )
    if world_coverage < MIN_WORLD_COVERAGE_FOR_FULL_PITCH:
        warnings.append(
            (
                "Metric correspondences cover too little of the configured "
                "pitch width for a trustworthy full-pitch projection."
            )
        )
    if projection_outside_fraction > MAX_PROJECTION_OUTSIDE_FRACTION:
        warnings.append(
            (
                "Too much projected pitch geometry falls outside the source "
                "image; check ground references and pitch dimensions."
            )
        )
    if not full_pitch_projection_allowed:
        warnings.append(
            (
                "Full-pitch geometry is withheld. Only local low-confidence "
                "debug geometry may be shown."
            )
        )
    warnings.extend(
        [
            (
                "Reprojection error measures the fitted correspondences; it "
                "is not independent proof of real-world accuracy."
            ),
            (
                "Numerical round-trip consistency does not prove that manual "
                "landmark semantics are correct."
            ),
            (
                "Homography maps only the pitch ground plane (Z = 0), not "
                "airborne ball height."
            ),
        ]
    )
    quality_grade = (
        _quality_grade(
            normalized_rmse,
            geometry_condition,
            wicket_order_valid,
        )
        if full_pitch_projection_allowed
        else "weak"
    )
    confidence = _overall_confidence(
        normalized_rmse,
        image_coverage,
        world_coverage,
        geometry_condition,
        wicket_order_valid,
        full_pitch_projection_allowed,
    )
    determinant = float(np.linalg.det(image_to_ground))
    homography = GroundHomographyResult(
        transform_available=True,
        image_to_ground_homography=_matrix_list(image_to_ground),
        ground_to_image_homography=_matrix_list(ground_to_image),
        determinant=round(determinant, 12),
        condition_number=round(condition_number, 6),
        estimation_method=estimation_method,
        ransac_reprojection_threshold_px=(
            RANSAC_REPROJECTION_THRESHOLD_PX if use_ransac else None
        ),
        ransac_inlier_count=inlier_count,
        ransac_inlier_landmark_ids=inlier_ids,
        round_trip_image_rmse_px=round(image_round_trip_rmse, 9),
        round_trip_ground_rmse_m=round(ground_round_trip_rmse, 12),
        image_convention="pixel_uv",
        ground_convention="pitch_xy_metres_z0",
    )
    quality = CalibrationQualityV2(
        landmark_coverage=landmark_coverage,
        usable_landmarks=len(landmarks),
        metric_correspondence_count=len(landmarks),
        additional_metric_ground_landmark_count=len(optional_landmarks),
        landmark_spread_score=round(landmark_spread_score, 6),
        world_coverage=round(world_coverage, 6),
        reprojection_rmse_px=round(rmse, 6),
        max_reprojection_error_px=round(maximum_error, 6),
        median_reprojection_error_px=round(median_error, 6),
        normalized_reprojection_rmse=round(normalized_rmse, 9),
        geometry_condition=geometry_condition,
        homography_condition_number=round(condition_number, 6),
        image_coverage=round(image_coverage, 6),
        wicket_order_valid=wicket_order_valid,
        transform_available=True,
        full_pitch_projection_allowed=full_pitch_projection_allowed,
        projection_outside_fraction=round(
            projection_outside_fraction,
            6,
        ),
        manual_adjustment_count=sum(
            landmark.source in {"manual", "manually_adjusted"}
            for landmark in primary_landmarks
        ),
        used_landmark_ids=used_landmark_ids,
        ignored_landmark_ids=[],
        landmark_sources=landmark_sources,
        warnings=_deduplicate(warnings),
        quality_grade=quality_grade,
        overall_confidence=confidence,
        reprojection_diagnostics=diagnostics,
    )
    return homography, quality


def _quality_without_transform(
    landmark_coverage: float,
    usable_landmarks: int,
    additional_metric_ground_landmark_count: int,
    image_coverage: float,
    world_coverage: float,
    landmark_spread_score: float,
    wicket_order_valid: bool,
    condition_number: float,
    geometry_condition: str,
    warnings: list[str],
    primary_landmarks: list[CalibrationLandmark],
    used_landmark_ids: list[str],
    landmark_sources: dict[str, int],
    diagnostics: list[ReprojectionDiagnostic] | None = None,
    ignored_landmark_ids: list[str] | None = None,
) -> CalibrationQualityV2:
    diagnostics = diagnostics or []
    errors = [diagnostic.error_px for diagnostic in diagnostics]
    rmse = (
        math.sqrt(sum(error * error for error in errors) / len(errors))
        if errors
        else None
    )
    return CalibrationQualityV2(
        landmark_coverage=landmark_coverage,
        usable_landmarks=usable_landmarks,
        metric_correspondence_count=usable_landmarks,
        additional_metric_ground_landmark_count=(
            additional_metric_ground_landmark_count
        ),
        landmark_spread_score=round(landmark_spread_score, 6),
        world_coverage=round(world_coverage, 6),
        reprojection_rmse_px=round(rmse, 6) if rmse is not None else None,
        max_reprojection_error_px=(
            round(max(errors), 6) if errors else None
        ),
        median_reprojection_error_px=(
            round(float(np.median(np.asarray(errors))), 6)
            if errors
            else None
        ),
        normalized_reprojection_rmse=None,
        geometry_condition=(
            geometry_condition
            if geometry_condition in {"weak", "unstable"}
            else "insufficient"
        ),
        homography_condition_number=(
            round(condition_number, 6)
            if math.isfinite(condition_number)
            else None
        ),
        image_coverage=round(image_coverage, 6),
        wicket_order_valid=wicket_order_valid,
        transform_available=False,
        full_pitch_projection_allowed=False,
        projection_outside_fraction=None,
        manual_adjustment_count=sum(
            landmark.source in {"manual", "manually_adjusted"}
            for landmark in primary_landmarks
        ),
        used_landmark_ids=used_landmark_ids,
        ignored_landmark_ids=ignored_landmark_ids or [],
        landmark_sources=landmark_sources,
        warnings=_deduplicate(warnings),
        quality_grade="insufficient_geometry",
        overall_confidence=0.0,
        reprojection_diagnostics=diagnostics,
    )


def _project_virtual_pitch_geometry(
    ground_to_image_homography: list[list[float]] | None,
    geometry: CricketPitchDimensions,
    projection_mode: str,
) -> VirtualPitchOverlayGeometry:
    if ground_to_image_homography is None:
        return VirtualPitchOverlayGeometry(
            projected_lines=[],
            projection_mode="landmarks_only",
        )
    labels = {
        "pitch_outline": "Pitch Ground Outline",
        "pitch_centreline": "Pitch Centreline",
        "bowler_wicket_width": "Bowler Wicket Width",
        "striker_wicket_width": "Striker Wicket Width",
        "bowler_popping_crease": "Bowler Popping Crease",
        "striker_popping_crease": "Striker Popping Crease",
    }
    allowed_line_ids = (
        set(labels)
        if projection_mode == "full_pitch"
        else {
            "pitch_centreline",
            "bowler_wicket_width",
            "striker_wicket_width",
        }
    )
    lines: list[ProjectedPitchLine] = []
    for line_id, ground_points in virtual_pitch_ground_lines(geometry).items():
        if line_id not in allowed_line_ids:
            continue
        try:
            image_points = [
                pitch_ground_point_to_image(
                    ground_to_image_homography,
                    x_m,
                    y_m,
                )
                for x_m, y_m in ground_points
            ]
        except VideoAnalysisServiceError:
            continue
        if not all(
            math.isfinite(x) and math.isfinite(y)
            and abs(x) < 10_000_000 and abs(y) < 10_000_000
            for x, y in image_points
        ):
            continue
        lines.append(
            ProjectedPitchLine(
                id=line_id,
                label=labels[line_id],
                ground_points=[
                    GroundPoint2D(x_m=x_m, y_m=y_m)
                    for x_m, y_m in ground_points
                ],
                image_points=[
                    ImagePixelPoint(x=x, y=y)
                    for x, y in image_points
                ],
            )
        )
    return VirtualPitchOverlayGeometry(
        projected_lines=lines,
        projection_mode=projection_mode,
    )


def _geometry_consistency_warnings(
    primary: list[CalibrationLandmark],
    image_convention: str,
    image_width: int,
    image_height: int,
) -> tuple[list[str], bool]:
    warnings: list[str] = []
    by_id = {landmark.id: landmark for landmark in primary}
    order_valid = True
    for wicket_end in ("bowler", "striker"):
        left = by_id[f"{wicket_end}_left_stump_base"]
        middle = by_id[f"{wicket_end}_middle_stump_base"]
        right = by_id[f"{wicket_end}_right_stump_base"]
        if image_convention == "image_left_is_world_left":
            current_valid = (
                left.normalized_x < middle.normalized_x < right.normalized_x
            )
        else:
            current_valid = (
                left.normalized_x > middle.normalized_x > right.normalized_x
            )
        order_valid = order_valid and current_valid
        minimum_distance = min(
            _normalized_distance(left, middle),
            _normalized_distance(middle, right),
            _normalized_distance(left, right),
        )
        if minimum_distance < MIN_STUMP_SEPARATION_NORMALIZED:
            warnings.append(
                f"{wicket_end.title()} stump-base landmarks overlap or are too close."
            )
        vertical_spread = max(
            left.normalized_y,
            middle.normalized_y,
            right.normalized_y,
        ) - min(
            left.normalized_y,
            middle.normalized_y,
            right.normalized_y,
        )
        if vertical_spread > 0.08:
            warnings.append(
                f"{wicket_end.title()} stump bases have an unusually large vertical spread."
            )
    bowler_middle = by_id["bowler_middle_stump_base"]
    striker_middle = by_id["striker_middle_stump_base"]
    if _normalized_distance(bowler_middle, striker_middle) < MIN_WICKET_SEPARATION_NORMALIZED:
        warnings.append("Bowler and striker wicket centres are too close.")
    if not order_valid:
        warnings.append(
            "Use Swap Left/Right or adjust markers to match the selected convention."
        )
    if image_width <= 0 or image_height <= 0:
        warnings.append("Reference image dimensions are invalid.")
    return warnings, order_valid


def _normalized_dlt_condition(
    landmarks: list[CalibrationLandmark],
    image_width: int,
    image_height: int,
    geometry: CricketPitchDimensions,
) -> tuple[float, int]:
    rows: list[list[float]] = []
    for landmark in landmarks:
        u = landmark.pixel_x / image_width
        v = landmark.pixel_y / image_height
        x = landmark.world_x_m / geometry.pitch_length_m
        y = landmark.world_y_m / geometry.pitch_width_m
        rows.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        rows.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[0] < 8:
        return math.inf, int(np.linalg.matrix_rank(matrix))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix, tol=1e-10))
    non_null_minimum = singular_values[-2] if len(singular_values) >= 2 else 0
    condition = (
        float(singular_values[0] / non_null_minimum)
        if non_null_minimum > 1e-12
        else math.inf
    )
    return condition, rank


def _geometry_condition(condition_number: float, rank: int) -> str:
    if rank < 8 or not math.isfinite(condition_number):
        return "insufficient"
    if condition_number <= WELL_CONDITIONED_DLT_LIMIT:
        return "well_conditioned"
    if condition_number <= WEAK_DLT_LIMIT:
        return "weak"
    return "unstable"


def _quality_grade(
    normalized_rmse: float,
    geometry_condition: str,
    wicket_order_valid: bool,
) -> str:
    if normalized_rmse <= EXCELLENT_RMSE_DIAGONAL_RATIO:
        grade = "excellent"
    elif normalized_rmse <= GOOD_RMSE_DIAGONAL_RATIO:
        grade = "good"
    elif normalized_rmse <= USABLE_RMSE_DIAGONAL_RATIO:
        grade = "usable"
    else:
        grade = "poor"
    order = ["poor", "usable", "good", "excellent"]
    if geometry_condition == "weak":
        grade = order[min(order.index(grade), order.index("usable"))]
    elif geometry_condition == "unstable" or not wicket_order_valid:
        grade = "poor"
    return grade


def _overall_confidence(
    normalized_rmse: float,
    image_coverage: float,
    world_coverage: float,
    geometry_condition: str,
    wicket_order_valid: bool,
    full_pitch_projection_allowed: bool,
) -> float:
    reprojection = max(
        0.0,
        1.0 - normalized_rmse / USABLE_RMSE_DIAGONAL_RATIO,
    )
    image_spread = min(1.0, image_coverage / 0.05)
    world_spread = min(
        1.0,
        world_coverage / MIN_WORLD_COVERAGE_FOR_FULL_PITCH,
    )
    condition = {
        "well_conditioned": 1.0,
        "weak": 0.6,
        "unstable": 0.2,
        "insufficient": 0.0,
    }[geometry_condition]
    confidence = (
        0.2 * reprojection
        + 0.2 * image_spread
        + 0.2 * world_spread
        + 0.2 * condition
        + 0.1 * float(wicket_order_valid)
        + 0.1 * float(full_pitch_projection_allowed)
    )
    return round(max(0.0, min(1.0, confidence)), 2)


def _reprojection_diagnostics(
    landmarks: list[CalibrationLandmark],
    ground_to_image: np.ndarray,
    inlier_flags: dict[str, bool],
) -> list[ReprojectionDiagnostic]:
    diagnostics: list[ReprojectionDiagnostic] = []
    matrix = _matrix_list(ground_to_image)
    for landmark in landmarks:
        projected_x, projected_y = pitch_ground_point_to_image(
            matrix,
            landmark.world_x_m,
            landmark.world_y_m,
        )
        error = math.hypot(
            projected_x - landmark.pixel_x,
            projected_y - landmark.pixel_y,
        )
        diagnostics.append(
            ReprojectionDiagnostic(
                landmark_id=landmark.id,
                landmark_source=landmark.source,
                used_for_homography=True,
                ransac_inlier=inlier_flags.get(landmark.id),
                observed_pixel_x=landmark.pixel_x,
                observed_pixel_y=landmark.pixel_y,
                reprojected_pixel_x=round(projected_x, 6),
                reprojected_pixel_y=round(projected_y, 6),
                error_px=round(error, 6),
            )
        )
    return diagnostics


def _round_trip_errors(
    landmarks: list[CalibrationLandmark],
    image_to_ground: np.ndarray,
    ground_to_image: np.ndarray,
) -> tuple[float, float]:
    image_to_ground_values = _matrix_list(image_to_ground)
    ground_to_image_values = _matrix_list(ground_to_image)
    image_errors: list[float] = []
    ground_errors: list[float] = []
    for landmark in landmarks:
        ground_x, ground_y = image_point_to_pitch_ground(
            image_to_ground_values,
            landmark.pixel_x,
            landmark.pixel_y,
        )
        image_x, image_y = pitch_ground_point_to_image(
            ground_to_image_values,
            ground_x,
            ground_y,
        )
        image_errors.append(
            math.hypot(
                image_x - landmark.pixel_x,
                image_y - landmark.pixel_y,
            )
        )
        projected_x, projected_y = pitch_ground_point_to_image(
            ground_to_image_values,
            landmark.world_x_m,
            landmark.world_y_m,
        )
        round_trip_x, round_trip_y = image_point_to_pitch_ground(
            image_to_ground_values,
            projected_x,
            projected_y,
        )
        ground_errors.append(
            math.hypot(
                round_trip_x - landmark.world_x_m,
                round_trip_y - landmark.world_y_m,
            )
        )
    return (
        math.sqrt(
            sum(error * error for error in image_errors)
            / len(image_errors)
        ),
        math.sqrt(
            sum(error * error for error in ground_errors)
            / len(ground_errors)
        ),
    )


def _save_calibration_v2_overlay(
    image: Image.Image,
    result: CalibrationV2Result,
    output_path: Path,
) -> None:
    try:
        overlay = image.convert("RGBA")
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        line_colors = {
            "pitch_outline": (213, 255, 107, 210),
            "pitch_centreline": (255, 255, 255, 230),
            "bowler_wicket_width": (80, 220, 255, 235),
            "striker_wicket_width": (255, 190, 70, 235),
            "bowler_popping_crease": (80, 220, 255, 170),
            "striker_popping_crease": (255, 190, 70, 170),
        }
        line_width = max(2, round(image.width / 420))
        local_debug = (
            result.virtual_pitch_overlay_geometry.projection_mode
            == "local_debug"
        )
        for line in result.virtual_pitch_overlay_geometry.projected_lines:
            points = [(round(point.x), round(point.y)) for point in line.image_points]
            if len(points) >= 2:
                color = line_colors.get(
                    line.id,
                    (213, 255, 107, 200),
                )
                if local_debug:
                    _draw_dashed_polyline(
                        draw,
                        points,
                        fill=(*color[:3], 150),
                        width=line_width,
                    )
                else:
                    draw.line(points, fill=color, width=line_width)
        for landmark in result.landmark_set.primary_stump_bases:
            color = (
                (80, 220, 255, 255)
                if landmark.wicket_end == "bowler"
                else (255, 190, 70, 255)
            )
            radius = max(5, round(image.width / 120))
            x = round(landmark.pixel_x)
            y = round(landmark.pixel_y)
            inferred = landmark.source == "inferred"
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(
                    (8, 12, 16, 110)
                    if inferred
                    else (8, 12, 16, 235)
                ),
                outline=color,
                width=max(1 if inferred else 2, radius // 3),
            )
            short_label = (
                "B" if landmark.wicket_end == "bowler" else "S"
            ) + landmark.id.split("_")[1][0].upper()
            if inferred:
                short_label = f"~{short_label}"
            draw.text((x + radius + 2, y - radius), short_label, fill=color)
        for landmark in result.landmark_set.optional_ground_landmarks:
            x = round(landmark.pixel_x)
            y = round(landmark.pixel_y)
            radius = max(6, round(image.width / 105))
            color = (255, 94, 190, 255)
            draw.polygon(
                (
                    (x, y - radius),
                    (x + radius, y),
                    (x, y + radius),
                    (x - radius, y),
                ),
                fill=(8, 12, 16, 235),
                outline=color,
            )
            reference_label = (
                "B" if landmark.id.startswith("bowler_") else "S"
            ) + ("L" if "_left_" in landmark.id else "R")
            draw.text(
                (x + radius + 2, y - radius),
                reference_label,
                fill=color,
            )
        header_width = min(image.width - 20, 390)
        draw.rectangle((10, 10, header_width, 82), fill=(8, 12, 16, 225))
        draw.text(
            (18, 17),
            "Calibration v2A.1 - Ground Plane",
            fill=(255, 255, 255, 255),
        )
        draw.text(
            (18, 35),
            (
                f"Status: {result.status} | "
                f"Quality: {result.quality.quality_grade}"
            ),
            fill=(
                (213, 255, 107, 255)
                if result.status == "ready"
                else (255, 202, 104, 255)
            ),
        )
        rmse = result.quality.reprojection_rmse_px
        draw.text(
            (18, 53),
            (
                f"Metric points: {result.quality.metric_correspondence_count} "
                f"| RMSE: {rmse:.2f}px"
                if rmse is not None
                else (
                    "Metric transform unavailable | "
                    f"points: {result.quality.metric_correspondence_count}"
                )
            ),
            fill=(220, 225, 230, 255),
        )
        if result.status != "ready":
            warning_text = (
                "LOW-CONFIDENCE DEBUG - FULL PITCH HIDDEN"
                if result.status == "weak"
                else "GROUND CALIBRATION REJECTED - LANDMARKS ONLY"
            )
            draw.rectangle(
                (10, 88, min(image.width - 20, 370), 110),
                fill=(90, 25, 15, 225),
            )
            draw.text(
                (18, 94),
                warning_text,
                fill=(255, 220, 180, 255),
            )
        composed = Image.alpha_composite(overlay, layer)
        composed.convert("RGB").save(output_path, format="JPEG", quality=93)
    except Exception as exc:
        raise VideoAnalysisServiceError(
            f"Calibration v2 overlay could not be saved: {type(exc).__name__}.",
            status_code=500,
        ) from exc


def _draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    dash_length = max(6, width * 4)
    gap_length = max(4, width * 2)
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        distance = 0.0
        while distance < length:
            dash_end = min(length, distance + dash_length)
            draw.line(
                (
                    (
                        round(start[0] + dx * distance / length),
                        round(start[1] + dy * distance / length),
                    ),
                    (
                        round(start[0] + dx * dash_end / length),
                        round(start[1] + dy * dash_end / length),
                    ),
                ),
                fill=fill,
                width=width,
            )
            distance += dash_length + gap_length


def _landmarks_from_wicket_box(
    wicket_end: str,
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    world_points: dict[str, tuple[float, float, float]],
) -> list[CalibrationLandmark]:
    positions = (
        ("left", x + width * 0.18),
        ("middle", x + width * 0.5),
        ("right", x + width * 0.82),
    )
    normalized_y = max(0.0, min(1.0, y + height))
    landmarks: list[CalibrationLandmark] = []
    for side, normalized_x in positions:
        landmark_id = f"{wicket_end}_{side}_stump_base"
        world_x, world_y, world_z = world_points[landmark_id]
        normalized_x = max(0.0, min(1.0, normalized_x))
        landmarks.append(
            CalibrationLandmark(
                id=landmark_id,
                label=LANDMARK_LABELS[landmark_id],
                wicket_end=wicket_end,
                landmark_type="stump_base",
                normalized_x=normalized_x,
                normalized_y=normalized_y,
                pixel_x=round(normalized_x * image_width, 6),
                pixel_y=round(normalized_y * image_height, 6),
                source="inferred",
                # A wicket-box score does not measure stump-base precision.
                confidence=None,
                world_x_m=world_x,
                world_y_m=world_y,
                world_z_m=world_z,
            )
        )
    return landmarks


def _starter_wicket_box(wicket_end: str):
    from ..schemas.video_analysis import NormalizedBox

    if wicket_end == "bowler":
        return NormalizedBox(x=0.35, y=0.62, width=0.30, height=0.28)
    return NormalizedBox(x=0.43, y=0.22, width=0.14, height=0.24)


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


def _transform_homogeneous_point(
    matrix: Sequence[Sequence[float]],
    first: float,
    second: float,
) -> tuple[float, float]:
    array = np.asarray(matrix, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise VideoAnalysisServiceError(
            "Homography matrix is invalid.",
            status_code=422,
        )
    projected = array @ np.array([first, second, 1.0], dtype=np.float64)
    denominator = float(projected[2])
    if not math.isfinite(denominator) or abs(denominator) < 1e-12:
        raise VideoAnalysisServiceError(
            "Point cannot be normalized through this homography.",
            status_code=422,
        )
    x = float(projected[0] / denominator)
    y = float(projected[1] / denominator)
    if not math.isfinite(x) or not math.isfinite(y):
        raise VideoAnalysisServiceError(
            "Homography produced an invalid point.",
            status_code=422,
        )
    return x, y


def _convex_hull_coverage(
    points: np.ndarray,
    image_width: int,
    image_height: int,
) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    area = abs(float(cv2.contourArea(hull)))
    return max(0.0, min(1.0, area / (image_width * image_height)))


def _world_hull_coverage(
    points: np.ndarray,
    geometry: CricketPitchDimensions,
) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    area = abs(float(cv2.contourArea(hull)))
    pitch_area = geometry.pitch_length_m * geometry.pitch_width_m
    return max(0.0, min(1.0, area / pitch_area))


def _landmark_spread_score(
    image_coverage: float,
    world_coverage: float,
) -> float:
    """Combined spread diagnostic, not a calibration-accuracy probability."""
    normalized_image_spread = min(
        1.0,
        image_coverage / MIN_IMAGE_COVERAGE_FOR_FULL_PITCH,
    )
    normalized_world_spread = min(
        1.0,
        world_coverage / MIN_WORLD_COVERAGE_FOR_FULL_PITCH,
    )
    return math.sqrt(normalized_image_spread * normalized_world_spread)


def _duplicate_point_pairs(
    landmarks: list[CalibrationLandmark],
    coordinate_getter,
    tolerance: float,
) -> list[str]:
    duplicates: list[str] = []
    for index, first in enumerate(landmarks):
        first_x, first_y = coordinate_getter(first)
        for second in landmarks[index + 1 :]:
            second_x, second_y = coordinate_getter(second)
            if math.hypot(second_x - first_x, second_y - first_y) <= tolerance:
                duplicates.append(f"{first.id}/{second.id}")
    return duplicates


def _landmark_source_counts(
    landmarks: list[CalibrationLandmark],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for landmark in landmarks:
        counts[landmark.source] = counts.get(landmark.source, 0) + 1
    return counts


def _full_projection_outside_fraction(
    ground_to_image: np.ndarray,
    geometry: CricketPitchDimensions,
    image_width: int,
    image_height: int,
) -> float:
    matrix = _matrix_list(ground_to_image)
    points: list[tuple[float, float]] = []
    for ground_points in virtual_pitch_ground_lines(geometry).values():
        for x_m, y_m in ground_points:
            try:
                points.append(
                    pitch_ground_point_to_image(matrix, x_m, y_m)
                )
            except VideoAnalysisServiceError:
                return 1.0
    if not points:
        return 1.0
    outside = sum(
        not (
            0 <= x < image_width
            and 0 <= y < image_height
            and math.isfinite(x)
            and math.isfinite(y)
        )
        for x, y in points
    )
    return outside / len(points)


def _wicket_end_separation(
    landmarks: list[CalibrationLandmark],
) -> float:
    by_id = {landmark.id: landmark for landmark in landmarks}
    bowler = by_id["bowler_middle_stump_base"]
    striker = by_id["striker_middle_stump_base"]
    return _normalized_distance(bowler, striker)


def _minimum_wicket_image_width(
    landmarks: list[CalibrationLandmark],
) -> float:
    by_id = {landmark.id: landmark for landmark in landmarks}
    return min(
        _normalized_distance(
            by_id[f"{end}_left_stump_base"],
            by_id[f"{end}_right_stump_base"],
        )
        for end in ("bowler", "striker")
    )


def _normalized_distance(
    first: CalibrationLandmark,
    second: CalibrationLandmark,
) -> float:
    return math.hypot(
        second.normalized_x - first.normalized_x,
        second.normalized_y - first.normalized_y,
    )


def _unavailable_homography() -> GroundHomographyResult:
    return GroundHomographyResult(
        transform_available=False,
        image_to_ground_homography=None,
        ground_to_image_homography=None,
        determinant=None,
        condition_number=None,
        image_convention="pixel_uv",
        ground_convention="pitch_xy_metres_z0",
    )


def _rejected_ransac_homography(
    condition_number: float,
    inlier_ids: list[str],
) -> GroundHomographyResult:
    """Keep RANSAC evidence while withholding the rejected transform."""
    return GroundHomographyResult(
        transform_available=False,
        image_to_ground_homography=None,
        ground_to_image_homography=None,
        determinant=None,
        condition_number=(
            round(condition_number, 6)
            if math.isfinite(condition_number)
            else None
        ),
        estimation_method="ransac",
        ransac_reprojection_threshold_px=RANSAC_REPROJECTION_THRESHOLD_PX,
        ransac_inlier_count=len(inlier_ids),
        ransac_inlier_landmark_ids=inlier_ids,
        image_convention="pixel_uv",
        ground_convention="pitch_xy_metres_z0",
    )


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [
        [round(float(value), 12) for value in row]
        for row in matrix.tolist()
    ]


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
            "Calibration v2 JSON could not be saved.",
            status_code=500,
        ) from exc


def _update_analysis_metadata(
    analysis_id: str,
    result: CalibrationV2Result,
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
                "calibration_v2_status": result.status,
                "calibration_v2_url": result.calibration_v2_url,
                "calibration_v2_overlay_url": (
                    result.calibration_v2_overlay_url
                ),
                "calibration_v2_quality_grade": result.quality.quality_grade,
                "calibration_v2_reprojection_rmse_px": (
                    result.quality.reprojection_rmse_px
                ),
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
            "Analysis metadata could not be updated for Calibration v2.",
            status_code=500,
        ) from exc


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
