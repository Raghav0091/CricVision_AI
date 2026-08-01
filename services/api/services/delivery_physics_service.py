"""Physics Engine V1 for the existing offline Video Analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CANONICAL_COORDINATE_SYSTEM,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    calibration_to_canonical_world,
    canonical_to_calibration_world,
)

from ..schemas.delivery_physics import (
    BouncePhysicsResult,
    CameraCalibration,
    ConfidenceGrade,
    DeliveryInterval,
    DeliveryPhysicsResult,
    FitDiagnostics,
    FittedTrajectoryParameters,
    LateralMovementResult,
    LineLengthResult,
    MetricAvailability,
    ObservedBallPoint,
    PostBounceMovementResult,
    RejectedObservation,
    SpeedAnalytics,
    TrajectorySample,
)
from ..schemas.video_analysis import (
    PrimaryBounceResult,
    TrackingPoint,
    VideoBallDetectionsDocument,
)
from .video_calibration_v2_service import (
    image_point_to_pitch_ground,
    load_video_calibration_v2,
    pitch_ground_point_to_image,
)
from .video_camera_pose_service import load_wicket_camera_pose


PHYSICS_ENGINE_VERSION = "v1"
GRAVITY_MPS2 = 9.81
MIN_METRIC_OBSERVATIONS = 6
MIN_IMAGE_OBSERVATIONS = 3
MAX_PROJECTION_SECONDS = 0.35
MAX_REASONABLE_SPEED_MPS = 50.0
MIN_REASONABLE_SPEED_MPS = 4.0
MAX_LATERAL_ACCELERATION_MPS2 = 20.0
MAX_FORWARD_DECELERATION_MPS2 = 15.0
OUTLIER_FLOOR_PX = 8.0

# CricVision coaching categories. These are not official MCC definitions.
LENGTH_BANDS_M = (
    (2.0, "yorker"),
    (5.0, "full"),
    (8.0, "good length"),
    (11.0, "short"),
)
LINE_BANDS_M = (
    (-0.34, "outside pitch left"),
    (-0.12, "pitch-left channel"),
    (0.12, "middle"),
    (0.34, "pitch-right channel"),
)

COORDINATE_SYSTEM_DESCRIPTION = CANONICAL_COORDINATE_SYSTEM


@dataclass(frozen=True)
class _MetricFit:
    model_name: str
    parameters: np.ndarray
    inlier_indexes: np.ndarray
    errors_px: np.ndarray
    rmse_px: float
    median_px: float
    max_inlier_px: float
    optimizer_status: str
    iterations: int
    bounds_reached: list[str]


@dataclass(frozen=True)
class _PostBounceFit:
    parameters: np.ndarray
    observation_indexes: np.ndarray
    errors_px: np.ndarray


@dataclass(frozen=True)
class _ImageFitSegment:
    coefficients_x: np.ndarray
    coefficients_y: np.ndarray
    origin_timestamp_seconds: float
    first_frame: int
    last_frame: int
    global_inlier_indexes: np.ndarray
    global_errors_px: np.ndarray


def analyse_delivery_physics(
    *,
    analysis_id: str,
    primary_track: list[TrackingPoint],
    detections: VideoBallDetectionsDocument,
    tracker_bounce: PrimaryBounceResult | None,
    fps: float,
    width: int,
    height: int,
    total_frames: int,
) -> DeliveryPhysicsResult:
    """Analyse one persisted tracker result without rerunning detection."""
    started = time.perf_counter()
    calibration = load_physics_calibration(analysis_id, width, height)
    accepted, rejected = canonical_observations(primary_track, detections, fps)

    if len(accepted) < MIN_IMAGE_OBSERVATIONS:
        return insufficient_physics_result(
            analysis_id=analysis_id,
            calibration=calibration,
            accepted=accepted,
            rejected=rejected,
            reason="Fewer than three reliable timestamped observations are available.",
            processing_seconds=time.perf_counter() - started,
        )

    if calibration.mode == "METRIC_3D" and len(accepted) >= MIN_METRIC_OBSERVATIONS:
        try:
            return _analyse_metric_3d(
                analysis_id=analysis_id,
                observations=accepted,
                initially_rejected=rejected,
                calibration=calibration,
                tracker_bounce=tracker_bounce,
                fps=fps,
                total_frames=total_frames,
                processing_started=started,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            calibration = calibration.model_copy(
                update={
                    "mode": "IMAGE_SPACE_ONLY",
                    "confidence": "LOW",
                    "failure_reason": f"Metric fit rejected: {type(exc).__name__}.",
                    "warnings": [
                        *calibration.warnings,
                        "Metric optimisation failed; image-space trajectory retained.",
                    ],
                }
            )

    return _analyse_non_3d(
        analysis_id=analysis_id,
        observations=accepted,
        rejected=rejected,
        calibration=calibration,
        tracker_bounce=tracker_bounce,
        fps=fps,
        total_frames=total_frames,
        processing_started=started,
    )


def load_physics_calibration(
    analysis_id: str,
    width: int,
    height: int,
) -> CameraCalibration:
    """Load the strongest already-confirmed calibration without fabricating it."""
    try:
        from .scene_calibration_service import (
            load_scene_calibration,
            load_active_accepted_scene_calibration,
        )

        assisted = load_scene_calibration(analysis_id)
        if assisted.accepted_calibration is not None:
            accepted = load_active_accepted_scene_calibration(analysis_id)
            confidence_score = _clamp(
                1.0 - accepted.reprojection_rmse_px / 20.0
            )
            if accepted.calibration_level == "METRIC_3D_READY":
                return CameraCalibration(
                    mode="METRIC_3D",
                    confidence=_grade(confidence_score),
                    image_width=width,
                    image_height=height,
                    camera_matrix=accepted.camera_matrix,
                    distortion_coefficients=accepted.distortion_coefficients,
                    rotation_vector=accepted.rotation_vector,
                    rotation_matrix=accepted.rotation_matrix,
                    translation_vector=accepted.translation_vector,
                    projection_matrix=accepted.projection_matrix,
                    image_to_pitch_homography=(
                        accepted.image_to_pitch_homography
                    ),
                    pitch_to_image_homography=(
                        accepted.pitch_to_image_homography
                    ),
                    correspondences_used=accepted.correspondence_count,
                    reprojection_error_px=accepted.reprojection_rmse_px,
                    calibration_confidence=confidence_score,
                    warnings=[
                        "Using accepted assisted calibration revision "
                        f"{accepted.revision}."
                    ],
                )
            return CameraCalibration(
                mode="METRIC_GROUND_PLANE",
                confidence=_grade(confidence_score),
                image_width=width,
                image_height=height,
                image_to_pitch_homography=(
                    accepted.image_to_pitch_homography
                ),
                pitch_to_image_homography=(
                    accepted.pitch_to_image_homography
                ),
                correspondences_used=accepted.correspondence_count,
                reprojection_error_px=accepted.reprojection_rmse_px,
                calibration_confidence=confidence_score,
                failure_reason=(
                    "Accepted assisted calibration supports ground-plane "
                    "metrics only."
                ),
                warnings=[
                    "Using accepted assisted calibration revision "
                    f"{accepted.revision}.",
                    "Airborne speed, height, and metric swing remain unavailable.",
                ],
            )
        if assisted.stage != "NOT_STARTED":
            return CameraCalibration(
                mode="IMAGE_SPACE_ONLY",
                confidence="LOW",
                image_width=width,
                image_height=height,
                failure_reason=(
                    "Assisted scene calibration has not passed acceptance."
                ),
                warnings=[
                    "Legacy visual geometry is not used to unlock metric physics."
                ],
            )
    except Exception:
        try:
            from .scene_calibration_service import load_scene_calibration

            assisted = load_scene_calibration(analysis_id)
            if assisted.stage != "NOT_STARTED":
                return CameraCalibration(
                    mode="IMAGE_SPACE_ONLY",
                    confidence="LOW",
                    image_width=width,
                    image_height=height,
                    failure_reason=(
                        "Accepted assisted calibration is unavailable or invalid."
                    ),
                    warnings=[
                        "Metric physics remains locked; image-space analysis is preserved."
                    ],
                )
        except Exception:
            pass

    ground = None
    ground_warning: list[str] = []
    try:
        ground = load_video_calibration_v2(analysis_id)
    except Exception:
        ground_warning.append("Confirmed metric ground-plane calibration is unavailable.")

    try:
        pose = load_wicket_camera_pose(analysis_id)
        solution = pose.camera_pose
        if (
            pose.success
            and solution.accepted
            and solution.rotation_vector is not None
            and solution.translation_vector is not None
        ):
            camera_matrix = np.asarray(
                pose.camera_intrinsics.intrinsic_matrix,
                dtype=np.float64,
            )
            rotation_vector = np.asarray(
                solution.rotation_vector,
                dtype=np.float64,
            ).reshape(3, 1)
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            translation = np.asarray(
                solution.translation_vector,
                dtype=np.float64,
            ).reshape(3, 1)
            projection = camera_matrix @ np.hstack(
                [rotation_matrix, translation]
            )
            confidence_score = float(
                pose.quality.overall_pose_quality
            )
            return CameraCalibration(
                mode="METRIC_3D",
                confidence=_grade(confidence_score),
                image_width=width,
                image_height=height,
                camera_matrix=camera_matrix.tolist(),
                distortion_coefficients=(
                    pose.camera_intrinsics.distortion_coefficients
                ),
                rotation_vector=rotation_vector.reshape(-1).tolist(),
                rotation_matrix=rotation_matrix.tolist(),
                translation_vector=translation.reshape(-1).tolist(),
                projection_matrix=projection.tolist(),
                image_to_pitch_homography=(
                    ground.homography.image_to_ground_homography
                    if ground is not None and ground.homography.transform_available
                    else None
                ),
                pitch_to_image_homography=(
                    ground.homography.ground_to_image_homography
                    if ground is not None and ground.homography.transform_available
                    else None
                ),
                correspondences_used=solution.landmark_count,
                reprojection_error_px=solution.reprojection_rmse_px,
                calibration_confidence=confidence_score,
                warnings=[*solution.warnings, *ground_warning],
            )
    except Exception:
        pass

    if (
        ground is not None
        and ground.success
        and ground.homography.transform_available
        and ground.homography.image_to_ground_homography is not None
        and ground.homography.ground_to_image_homography is not None
    ):
        confidence_score = float(ground.quality.overall_confidence)
        return CameraCalibration(
            mode="METRIC_GROUND_PLANE",
            confidence=_grade(confidence_score),
            image_width=width,
            image_height=height,
            image_to_pitch_homography=(
                ground.homography.image_to_ground_homography
            ),
            pitch_to_image_homography=(
                ground.homography.ground_to_image_homography
            ),
            correspondences_used=ground.quality.metric_correspondence_count,
            reprojection_error_px=ground.quality.reprojection_rmse_px,
            calibration_confidence=confidence_score,
            failure_reason=(
                "Full camera pose unavailable; only ground-plane metrics are valid."
            ),
            warnings=[
                *ground.quality.warnings,
                "Airborne height and 3D speed are unavailable in ground-plane mode.",
            ],
        )

    return CameraCalibration(
        mode="IMAGE_SPACE_ONLY",
        confidence="UNAVAILABLE",
        image_width=width,
        image_height=height,
        calibration_confidence=0.0,
        failure_reason="No reliable metric camera calibration is available.",
        warnings=[
            "Trajectory is reported in image space; metric values are unavailable."
        ],
    )


def canonical_observations(
    primary_track: list[TrackingPoint],
    detections: VideoBallDetectionsDocument,
    fps: float,
) -> tuple[list[ObservedBallPoint], list[RejectedObservation]]:
    candidate_by_id = {
        candidate.candidate_id: candidate
        for frame in detections.frames
        for candidate in frame.detections
    }
    accepted: list[ObservedBallPoint] = []
    rejected: list[RejectedObservation] = []
    previous: ObservedBallPoint | None = None
    seen_frames: set[int] = set()

    for point in sorted(primary_track, key=lambda item: item.frame_index):
        if point.provenance != "OBSERVED":
            continue
        if point.frame_index in seen_frames:
            rejected.append(
                RejectedObservation(
                    frame_index=point.frame_index,
                    candidate_id=point.candidate_id,
                    reason="duplicate_observation_frame",
                )
            )
            continue
        expected_timestamp = point.frame_index / fps
        if (
            not math.isfinite(point.timestamp_seconds)
            or point.timestamp_seconds < 0
            or abs(point.timestamp_seconds - expected_timestamp) > max(0.05, 2 / fps)
        ):
            rejected.append(
                RejectedObservation(
                    frame_index=point.frame_index,
                    candidate_id=point.candidate_id,
                    reason="invalid_or_inconsistent_timestamp",
                )
            )
            continue
        candidate = candidate_by_id.get(point.candidate_id or "")
        detector_confidence = (
            float(candidate.confidence)
            if candidate is not None
            else float(point.confidence)
        )
        if min(detector_confidence, point.confidence) < 0.08:
            rejected.append(
                RejectedObservation(
                    frame_index=point.frame_index,
                    candidate_id=point.candidate_id,
                    reason="very_low_combined_confidence",
                )
            )
            continue
        observation = ObservedBallPoint(
            frame_index=point.frame_index,
            timestamp_seconds=point.timestamp_seconds,
            pixel_x=point.x,
            pixel_y=point.y,
            detector_confidence=detector_confidence,
            tracker_confidence=point.confidence,
            source=point.provenance,
            candidate_id=point.candidate_id,
            bbox_xyxy=(candidate.bbox_xyxy if candidate is not None else None),
        )
        if previous is not None:
            dt = observation.timestamp_seconds - previous.timestamp_seconds
            distance = math.hypot(
                observation.pixel_x - previous.pixel_x,
                observation.pixel_y - previous.pixel_y,
            )
            diagonal = math.hypot(
                detections.frames[0].detections[0].center.x
                if detections.frames and detections.frames[0].detections
                else 1,
                detections.frames[0].detections[0].center.y
                if detections.frames and detections.frames[0].detections
                else 1,
            )
            # Pixel gate is deliberately generous; robust reprojection handles
            # subtler false positives without deleting fast motion.
            if dt <= 0 or distance / max(dt, 1e-6) > max(5000.0, diagonal * 8):
                rejected.append(
                    RejectedObservation(
                        frame_index=point.frame_index,
                        candidate_id=point.candidate_id,
                        reason="impossible_image_space_jump",
                    )
                )
                continue
        accepted.append(observation)
        seen_frames.add(point.frame_index)
        previous = observation
    return accepted, rejected


def _analyse_metric_3d(
    *,
    analysis_id: str,
    observations: list[ObservedBallPoint],
    initially_rejected: list[RejectedObservation],
    calibration: CameraCalibration,
    tracker_bounce: PrimaryBounceResult | None,
    fps: float,
    total_frames: int,
    processing_started: float,
) -> DeliveryPhysicsResult:
    pre_indexes = np.arange(len(observations))
    if (
        tracker_bounce is not None
        and tracker_bounce.bounce_frame is not None
    ):
        before = np.array(
            [
                index
                for index, observation in enumerate(observations)
                if observation.frame_index <= tracker_bounce.bounce_frame
            ],
            dtype=int,
        )
        if len(before) >= MIN_METRIC_OBSERVATIONS:
            pre_indexes = before
    fit = fit_metric_trajectory(
        [observations[index] for index in pre_indexes],
        calibration,
    )
    accepted_indexes = pre_indexes[fit.inlier_indexes]
    metric_observations = [observations[index] for index in accepted_indexes]
    outlier_indexes = sorted(set(pre_indexes) - set(accepted_indexes))
    rejected = [
        *initially_rejected,
        *[
            RejectedObservation(
                frame_index=observations[index].frame_index,
                candidate_id=observations[index].candidate_id,
                reason="reprojection_outlier",
                residual_px=float(fit.errors_px[list(pre_indexes).index(index)]),
            )
            for index in outlier_indexes
        ],
    ]
    params = fit.parameters
    bounce = _metric_bounce(
        params,
        fit.model_name,
        calibration,
        observations,
        tracker_bounce,
        fps,
    )
    post_fit = _fit_post_bounce(
        observations,
        calibration,
        params,
        fit.model_name,
        bounce,
    )
    samples, interval = _metric_samples(
        observations=observations,
        accepted=metric_observations,
        calibration=calibration,
        parameters=params,
        model_name=fit.model_name,
        post_fit=post_fit,
        bounce=bounce,
        fps=fps,
        total_frames=total_frames,
        fit_rmse=fit.rmse_px,
    )
    speed = _speed_analytics(samples, bounce, metric_observations)
    lateral = _lateral_movement(
        params,
        fit.model_name,
        bounce,
        metric_observations,
    )
    post_movement = _post_bounce_movement(
        params,
        fit.model_name,
        post_fit,
        bounce,
        observations,
    )
    line_length = _line_and_length(bounce)
    observed_fraction = len(metric_observations) / max(1, len(samples))
    fit_score = max(0.0, 1.0 - fit.rmse_px / 30.0)
    confidence_score = _clamp(
        calibration.calibration_confidence * 0.35
        + min(1.0, len(metric_observations) / 12) * 0.20
        + observed_fraction * 0.15
        + fit_score * 0.30
    )
    warnings: list[str] = []
    warnings.append(
        "Line side labels follow the confirmed calibration convention; "
        "batter handedness is not independently inferred."
    )
    if post_fit is None:
        warnings.append(
            "Post-bounce movement is projected or unavailable because fewer "
            "than four reliable post-bounce observations exist."
        )
    if speed.earliest_measured_speed_kmh is None:
        warnings.append("Metric speed was rejected as physically implausible.")

    return DeliveryPhysicsResult(
        status=(
            "SUCCESS"
            if bounce.status != "INSUFFICIENT_EVIDENCE"
            and confidence_score >= 0.5
            else "PARTIAL"
        ),
        analysis_id=analysis_id,
        coordinate_system=COORDINATE_SYSTEM_DESCRIPTION,
        calibration=calibration,
        fitted_parameters=_parameter_contract(
            params,
            fit.model_name,
            observations[0].timestamp_seconds,
            fit.bounds_reached,
            post_fit,
        ),
        trajectory_samples=samples,
        accepted_observations=metric_observations,
        rejected_observations=rejected,
        delivery_interval=interval,
        bounce=bounce,
        speed=speed,
        pre_bounce_lateral_movement=lateral,
        post_bounce_movement=post_movement,
        line_and_length=line_length,
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model=fit.model_name,
            optimizer_status=fit.optimizer_status,
            iterations=fit.iterations,
            inlier_frames=[
                observation.frame_index for observation in metric_observations
            ],
            outlier_frames=[
                observations[index].frame_index for index in outlier_indexes
            ],
            weighted_reprojection_rmse_px=fit.rmse_px,
            median_reprojection_error_px=fit.median_px,
            maximum_inlier_error_px=fit.max_inlier_px,
            parameter_bounds_reached=fit.bounds_reached,
            processing_duration_seconds=round(
                time.perf_counter() - processing_started,
                6,
            ),
        ),
        confidence=_grade(confidence_score),
        confidence_score=round(confidence_score, 6),
        uncertainty_method="deterministic residual and bounded-parameter perturbation",
        unavailable_metrics=_unavailable_metrics(
            speed,
            lateral,
            post_movement,
            bounce,
        ),
        warnings=warnings,
    )


def fit_metric_trajectory(
    observations: list[ObservedBallPoint],
    calibration: CameraCalibration,
) -> _MetricFit:
    """Confidence-weighted robust reprojection fit with simple model selection."""
    if len(observations) < MIN_METRIC_OBSERVATIONS:
        raise ValueError("Metric fitting requires at least six observations.")
    if calibration.mode != "METRIC_3D":
        raise ValueError("Metric fitting requires an accepted 3D camera pose.")
    timestamps = np.asarray(
        [item.timestamp_seconds for item in observations],
        dtype=np.float64,
    )
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("Observation timestamps must increase.")

    fits: list[_MetricFit] = []
    for model_name in (
        "BALLISTIC",
        "BALLISTIC_LATERAL",
        "BALLISTIC_LATERAL_DECELERATION",
    ):
        candidate = _fit_model(observations, calibration, model_name)
        if not fits:
            fits.append(candidate)
            continue
        previous = fits[-1]
        improvement = (
            (previous.rmse_px - candidate.rmse_px)
            / max(previous.rmse_px, 1e-6)
        )
        required = 0.08 if model_name == "BALLISTIC_LATERAL" else 0.10
        if improvement >= required and not candidate.bounds_reached:
            fits.append(candidate)
        else:
            break
    return fits[-1]


def _fit_model(
    observations: list[ObservedBallPoint],
    calibration: CameraCalibration,
    model_name: str,
) -> _MetricFit:
    initial = _initial_parameters(observations, calibration, model_name)
    lower, upper, names = _parameter_bounds(model_name)
    weights = np.sqrt(
        np.asarray(
            [
                max(0.05, item.detector_confidence * item.tracker_confidence)
                for item in observations
            ],
            dtype=np.float64,
        )
    )

    def residual(parameters: np.ndarray, indexes: np.ndarray) -> np.ndarray:
        selected = [observations[index] for index in indexes]
        projected = _project_observations(
            parameters,
            model_name,
            selected,
            calibration,
        )
        observed_pixels = np.asarray(
            [[item.pixel_x, item.pixel_y] for item in selected],
            dtype=np.float64,
        )
        pixel_residual = (
            projected - observed_pixels
        ) * weights[indexes, None]
        regularization = _physical_regularization(parameters, model_name)
        return np.concatenate([pixel_residual.reshape(-1), regularization])

    indexes = np.arange(len(observations), dtype=int)
    result = least_squares(
        lambda parameters: residual(parameters, indexes),
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=500,
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    errors = _pixel_errors(
        result.x,
        model_name,
        observations,
        calibration,
    )
    threshold = _outlier_threshold(errors)
    inliers = np.where(errors <= threshold)[0]
    if len(inliers) >= MIN_METRIC_OBSERVATIONS and len(inliers) < len(indexes):
        result = least_squares(
            lambda parameters: residual(parameters, inliers),
            result.x,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=350,
        )
        errors = _pixel_errors(
            result.x,
            model_name,
            observations,
            calibration,
        )
        threshold = _outlier_threshold(errors[inliers])
        inliers = np.where(errors <= threshold)[0]
    if len(inliers) < MIN_METRIC_OBSERVATIONS or not result.success:
        raise ValueError("Robust metric fit did not retain enough inliers.")

    inlier_errors = errors[inliers]
    bounds_reached = [
        name
        for name, value, low, high in zip(names, result.x, lower, upper)
        if abs(value - low) <= max(1e-4, abs(low) * 1e-3)
        or abs(value - high) <= max(1e-4, abs(high) * 1e-3)
    ]
    return _MetricFit(
        model_name=model_name,
        parameters=result.x,
        inlier_indexes=inliers,
        errors_px=errors,
        rmse_px=float(np.sqrt(np.mean(np.square(inlier_errors)))),
        median_px=float(np.median(inlier_errors)),
        max_inlier_px=float(np.max(inlier_errors)),
        optimizer_status=result.message,
        iterations=int(result.nfev),
        bounds_reached=bounds_reached,
    )


def _initial_parameters(
    observations: list[ObservedBallPoint],
    calibration: CameraCalibration,
    model_name: str,
) -> np.ndarray:
    duration = max(
        0.08,
        observations[-1].timestamp_seconds
        - observations[0].timestamp_seconds,
    )
    ground_points = [
        _image_to_physics_ground(
            calibration.image_to_pitch_homography,
            item.pixel_x,
            item.pixel_y,
        )
        for item in observations
    ]
    valid = [item for item in ground_points if item is not None]
    if len(valid) >= 2:
        x0, y0 = valid[0]
        x1, y1 = valid[-1]
    else:
        x0, y0 = 0.0, 2.0
        x1, y1 = 0.0, min(18.0, 2.0 + 24.0 * duration)
    vy = _clamp((y1 - y0) / duration, 8.0, 42.0)
    vx = _clamp((x1 - x0) / duration, -6.0, 6.0)
    base = [x0, y0, 1.8, vx, vy, 1.0]
    if model_name in {"BALLISTIC_LATERAL", "BALLISTIC_LATERAL_DECELERATION"}:
        base.append(0.0)
    if model_name == "BALLISTIC_LATERAL_DECELERATION":
        base.append(-1.0)
    lower, upper, _ = _parameter_bounds(model_name)
    return np.clip(np.asarray(base, dtype=np.float64), lower + 1e-6, upper - 1e-6)


def _parameter_bounds(model_name: str):
    names = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
    lower = [-2.5, -1.0, 0.15, -10.0, 4.0, -12.0]
    upper = [2.5, 21.0, 3.5, 10.0, 50.0, 15.0]
    if model_name in {"BALLISTIC_LATERAL", "BALLISTIC_LATERAL_DECELERATION"}:
        names.append("ax")
        lower.append(-MAX_LATERAL_ACCELERATION_MPS2)
        upper.append(MAX_LATERAL_ACCELERATION_MPS2)
    if model_name == "BALLISTIC_LATERAL_DECELERATION":
        names.append("ay")
        lower.append(-MAX_FORWARD_DECELERATION_MPS2)
        upper.append(0.0)
    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        names,
    )


def _world_at(
    parameters: Sequence[float],
    model_name: str,
    elapsed: np.ndarray | float,
) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.float64)
    t = np.asarray(elapsed, dtype=np.float64)
    ax = p[6] if len(p) >= 7 else 0.0
    ay = p[7] if len(p) >= 8 else 0.0
    x = p[0] + p[3] * t + 0.5 * ax * t * t
    y = p[1] + p[4] * t + 0.5 * ay * t * t
    z = p[2] + p[5] * t - 0.5 * GRAVITY_MPS2 * t * t
    return np.stack([x, y, z], axis=-1)


def _velocity_at(
    parameters: Sequence[float],
    elapsed: float,
) -> np.ndarray:
    p = np.asarray(parameters, dtype=np.float64)
    ax = p[6] if len(p) >= 7 else 0.0
    ay = p[7] if len(p) >= 8 else 0.0
    return np.asarray(
        [
            p[3] + ax * elapsed,
            p[4] + ay * elapsed,
            p[5] - GRAVITY_MPS2 * elapsed,
        ],
        dtype=np.float64,
    )


def _project_observations(
    parameters: Sequence[float],
    model_name: str,
    observations: list[ObservedBallPoint],
    calibration: CameraCalibration,
) -> np.ndarray:
    origin = observations[0].timestamp_seconds
    elapsed = np.asarray(
        [item.timestamp_seconds - origin for item in observations],
        dtype=np.float64,
    )
    world = _world_at(parameters, model_name, elapsed)
    return _project_physics_world(world, calibration)


def _project_physics_world(
    world: np.ndarray,
    calibration: CameraCalibration,
) -> np.ndarray:
    # Explicit compatibility adapter: Calibration V2 stores
    # longitudinal/lateral/up while the permanent model is
    # lateral/longitudinal/up.
    calibration_world = np.column_stack(
        [world[:, 1], world[:, 0], world[:, 2]]
    ).astype(np.float64)
    projected, _ = cv2.projectPoints(
        calibration_world,
        np.asarray(calibration.rotation_vector, dtype=np.float64),
        np.asarray(calibration.translation_vector, dtype=np.float64),
        np.asarray(calibration.camera_matrix, dtype=np.float64),
        np.asarray(calibration.distortion_coefficients, dtype=np.float64),
    )
    return projected.reshape(-1, 2)


def _pixel_errors(
    parameters: np.ndarray,
    model_name: str,
    observations: list[ObservedBallPoint],
    calibration: CameraCalibration,
) -> np.ndarray:
    projected = _project_observations(
        parameters,
        model_name,
        observations,
        calibration,
    )
    observed = np.asarray(
        [[item.pixel_x, item.pixel_y] for item in observations],
        dtype=np.float64,
    )
    return np.linalg.norm(projected - observed, axis=1)


def _physical_regularization(
    parameters: np.ndarray,
    model_name: str,
) -> np.ndarray:
    penalties = [
        (parameters[2] - 1.8) / 1.5,
        max(0.0, abs(parameters[3]) - 7.0) / 2.0,
    ]
    if len(parameters) >= 7:
        penalties.append(parameters[6] / 12.0)
    if len(parameters) >= 8:
        penalties.append(parameters[7] / 8.0)
    return np.asarray(penalties, dtype=np.float64)


def _outlier_threshold(errors: np.ndarray) -> float:
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    return max(OUTLIER_FLOOR_PX, median + 3.5 * max(mad, 1.0))


def _metric_bounce(
    parameters: np.ndarray,
    model_name: str,
    calibration: CameraCalibration,
    observations: list[ObservedBallPoint],
    tracker_bounce: PrimaryBounceResult | None,
    fps: float,
) -> BouncePhysicsResult:
    roots = np.roots(
        [-0.5 * GRAVITY_MPS2, parameters[5], parameters[2]]
    )
    positive = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-8 and root.real > 0
    )
    if not positive:
        return _unavailable_bounce()
    elapsed = positive[0]
    world = _world_at(parameters, model_name, elapsed).reshape(3)
    if (
        world[1] < -0.5
        or world[1] > PITCH_LENGTH_M + 0.5
        or abs(world[0]) > PITCH_WIDTH_M / 2 + 0.5
    ):
        return _unavailable_bounce()
    origin = observations[0].timestamp_seconds
    timestamp = origin + elapsed
    frame = int(round(timestamp * fps))
    pixel = _project_physics_world(world.reshape(1, 3), calibration)[0]
    directly_supported = any(
        abs(item.frame_index - frame) <= 2 for item in observations
    )
    tracker_support = (
        tracker_bounce is not None
        and tracker_bounce.bounce_frame is not None
        and abs(tracker_bounce.bounce_frame - frame) <= 3
    )
    score = 0.45 + 0.25 * directly_supported + 0.20 * tracker_support
    evidence = ["ballistic_height_crosses_pitch_plane"]
    if directly_supported:
        evidence.append("nearby_observed_ball_point")
    if tracker_support:
        evidence.append("tracker_slope_transition_agrees")
    return BouncePhysicsResult(
        status="DETECTED" if directly_supported and tracker_support else "ESTIMATED",
        frame_index=max(0, frame),
        timestamp_seconds=round(timestamp, 6),
        world_x_m=round(float(world[0]), 6),
        world_y_m=round(float(world[1]), 6),
        pixel_x=round(float(pixel[0]), 3),
        pixel_y=round(float(pixel[1]), 3),
        distance_from_striker_wicket_m=round(
            max(0.0, PITCH_LENGTH_M - float(world[1])),
            6,
        ),
        lateral_offset_m=round(float(world[0]), 6),
        confidence=_grade(score),
        confidence_score=round(_clamp(score), 6),
        uncertainty_frames=2 if directly_supported else 4,
        uncertainty_m=round(0.15 + 0.35 * (1.0 - score), 3),
        directly_supported=directly_supported,
        evidence=evidence,
    )


def _fit_post_bounce(
    observations: list[ObservedBallPoint],
    calibration: CameraCalibration,
    pre_parameters: np.ndarray,
    model_name: str,
    bounce: BouncePhysicsResult,
) -> _PostBounceFit | None:
    if bounce.timestamp_seconds is None:
        return None
    indexes = np.asarray(
        [
            index
            for index, item in enumerate(observations)
            if item.timestamp_seconds > bounce.timestamp_seconds + 1e-6
        ],
        dtype=int,
    )
    if len(indexes) < 4:
        return None
    pre_elapsed = (
        bounce.timestamp_seconds - observations[0].timestamp_seconds
    )
    bounce_world = _world_at(
        pre_parameters,
        model_name,
        pre_elapsed,
    ).reshape(3)
    pre_velocity = _velocity_at(pre_parameters, pre_elapsed)
    initial = np.asarray(
        [
            pre_velocity[0] * 0.8,
            max(2.0, pre_velocity[1] * 0.75),
            max(1.0, -pre_velocity[2] * 0.45),
        ],
        dtype=np.float64,
    )
    lower = np.asarray([-15.0, 2.0, 0.5])
    upper = np.asarray([15.0, 45.0, 18.0])
    selected = [observations[index] for index in indexes]
    elapsed = np.asarray(
        [
            item.timestamp_seconds - bounce.timestamp_seconds
            for item in selected
        ],
        dtype=np.float64,
    )
    observed = np.asarray(
        [[item.pixel_x, item.pixel_y] for item in selected],
        dtype=np.float64,
    )
    weights = np.sqrt(
        np.asarray(
            [
                max(0.05, item.detector_confidence * item.tracker_confidence)
                for item in selected
            ]
        )
    )

    def post_world(parameters: np.ndarray) -> np.ndarray:
        x = bounce_world[0] + parameters[0] * elapsed
        y = bounce_world[1] + parameters[1] * elapsed
        z = parameters[2] * elapsed - 0.5 * GRAVITY_MPS2 * elapsed**2
        return np.column_stack([x, y, z])

    def residual(parameters: np.ndarray) -> np.ndarray:
        projected = _project_physics_world(post_world(parameters), calibration)
        return ((projected - observed) * weights[:, None]).reshape(-1)

    result = least_squares(
        residual,
        np.clip(initial, lower + 1e-6, upper - 1e-6),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=4.0,
        max_nfev=300,
    )
    errors = np.linalg.norm(
        _project_physics_world(post_world(result.x), calibration) - observed,
        axis=1,
    )
    if not result.success or float(np.median(errors)) > 20.0:
        return None
    return _PostBounceFit(
        parameters=result.x,
        observation_indexes=indexes,
        errors_px=errors,
    )


def _metric_samples(
    *,
    observations: list[ObservedBallPoint],
    accepted: list[ObservedBallPoint],
    calibration: CameraCalibration,
    parameters: np.ndarray,
    model_name: str,
    post_fit: _PostBounceFit | None,
    bounce: BouncePhysicsResult,
    fps: float,
    total_frames: int,
    fit_rmse: float,
) -> tuple[list[TrajectorySample], DeliveryInterval]:
    first_frame = accepted[0].frame_index
    last_observed = accepted[-1].frame_index
    maximum_terminal = min(
        total_frames - 1,
        last_observed + max(1, int(round(MAX_PROJECTION_SECONDS * fps))),
    )
    accepted_by_frame = {item.frame_index: item for item in accepted}
    accepted_frames = np.asarray(sorted(accepted_by_frame), dtype=int)
    samples: list[TrajectorySample] = []
    origin = observations[0].timestamp_seconds
    terminal_reason = "maximum_projection_horizon"

    for frame in range(first_frame, maximum_terminal + 1):
        timestamp = frame / fps
        elapsed = timestamp - origin
        world = _world_at(parameters, model_name, elapsed).reshape(3)
        velocity = _velocity_at(parameters, elapsed)
        if (
            bounce.timestamp_seconds is not None
            and timestamp > bounce.timestamp_seconds
        ):
            post_elapsed = timestamp - bounce.timestamp_seconds
            bounce_elapsed = bounce.timestamp_seconds - origin
            bounce_world = _world_at(
                parameters,
                model_name,
                bounce_elapsed,
            ).reshape(3)
            if post_fit is not None:
                pv = post_fit.parameters
            else:
                before = _velocity_at(parameters, bounce_elapsed)
                pv = np.asarray(
                    [before[0] * 0.8, before[1] * 0.75, max(0.5, -before[2] * 0.4)]
                )
            world = np.asarray(
                [
                    bounce_world[0] + pv[0] * post_elapsed,
                    bounce_world[1] + pv[1] * post_elapsed,
                    max(0.0, pv[2] * post_elapsed - 0.5 * GRAVITY_MPS2 * post_elapsed**2),
                ]
            )
            velocity = np.asarray(
                [pv[0], pv[1], pv[2] - GRAVITY_MPS2 * post_elapsed]
            )
        if world[1] >= PITCH_LENGTH_M:
            terminal_reason = "striker_wicket_plane"
        pixel = _project_physics_world(world.reshape(1, 3), calibration)[0]
        nearest_index = int(np.argmin(np.abs(accepted_frames - frame)))
        nearest_frame = int(accepted_frames[nearest_index])
        delta_frames = abs(frame - nearest_frame)
        if frame in accepted_by_frame:
            provenance = "OBSERVED"
        elif frame <= last_observed:
            provenance = "RECONSTRUCTED"
        else:
            provenance = "PROJECTED"
        base_confidence = max(0.1, 1.0 - fit_rmse / 35.0)
        confidence = base_confidence * math.exp(
            -delta_frames / max(2.0, fps * 0.18)
        )
        speed = float(np.linalg.norm(velocity))
        samples.append(
            TrajectorySample(
                frame_index=frame,
                timestamp_seconds=round(timestamp, 6),
                world_x_m=round(float(world[0]), 6),
                world_y_m=round(float(world[1]), 6),
                world_z_m=round(max(0.0, float(world[2])), 6),
                pixel_x=round(float(pixel[0]), 3),
                pixel_y=round(float(pixel[1]), 3),
                velocity_x_mps=round(float(velocity[0]), 6),
                velocity_y_mps=round(float(velocity[1]), 6),
                velocity_z_mps=round(float(velocity[2]), 6),
                speed_mps=round(speed, 6),
                provenance=provenance,
                confidence=round(_clamp(confidence), 6),
                nearest_observation_frame=nearest_frame,
                nearest_observation_delta_frames=delta_frames,
                nearest_observation_delta_seconds=round(delta_frames / fps, 6),
            )
        )
        if terminal_reason == "striker_wicket_plane":
            break

    return samples, DeliveryInterval(
        start_frame=samples[0].frame_index if samples else None,
        end_frame=samples[-1].frame_index if samples else None,
        first_observed_frame=accepted[0].frame_index,
        last_observed_frame=last_observed,
        terminal_reason=terminal_reason,
    )


def _analyse_non_3d(
    *,
    analysis_id: str,
    observations: list[ObservedBallPoint],
    rejected: list[RejectedObservation],
    calibration: CameraCalibration,
    tracker_bounce: PrimaryBounceResult | None,
    fps: float,
    total_frames: int,
    processing_started: float,
) -> DeliveryPhysicsResult:
    image_segments = _fit_image_segments(
        observations,
        (
            tracker_bounce.bounce_frame
            if tracker_bounce is not None
            and tracker_bounce.bounce_detected is True
            else None
        ),
    )
    inliers = np.asarray(
        sorted(
            {
                int(index)
                for segment in image_segments
                for index in segment.global_inlier_indexes
            }
        ),
        dtype=int,
    )
    errors = np.full(len(observations), np.nan, dtype=np.float64)
    for segment in image_segments:
        finite = np.isfinite(segment.global_errors_px)
        errors[finite] = segment.global_errors_px[finite]
    accepted = [observations[index] for index in inliers]
    rejected_indexes = sorted(set(range(len(observations))) - set(inliers))
    rejected = [
        *rejected,
        *[
            RejectedObservation(
                frame_index=observations[index].frame_index,
                candidate_id=observations[index].candidate_id,
                reason="image_trajectory_outlier",
                residual_px=float(errors[index]),
            )
            for index in rejected_indexes
        ],
    ]
    samples, interval = _image_samples(
        accepted,
        image_segments,
        fps,
        total_frames,
        float(np.sqrt(np.mean(np.square(errors[inliers])))),
    )
    bounce = _ground_or_image_bounce(
        tracker_bounce,
        calibration,
        fps,
    )
    line_length = _line_and_length(bounce)
    mode_status = (
        "PARTIAL"
        if calibration.mode == "METRIC_GROUND_PLANE"
        else "IMAGE_SPACE_ONLY"
    )
    rmse = float(np.sqrt(np.mean(np.square(errors[inliers]))))
    score = _clamp(
        min(1.0, len(accepted) / 10) * 0.45
        + max(0.0, 1.0 - rmse / 30.0) * 0.35
        + calibration.calibration_confidence * 0.20
    )
    unavailable_reason = (
        "A reliable 3D camera pose is required for metric speed."
    )
    speed = SpeedAnalytics(
        confidence="INSUFFICIENT_EVIDENCE",
        observed_temporal_span_seconds=round(
            accepted[-1].timestamp_seconds - accepted[0].timestamp_seconds,
            6,
        ),
        unavailable_reason=unavailable_reason,
    )
    lateral = LateralMovementResult(
        direction="unavailable",
        observed_fraction=len(accepted) / max(1, len(samples)),
        confidence="INSUFFICIENT_EVIDENCE",
        unavailable_reason=(
            "Metric pre-bounce lateral movement requires a reliable 3D camera pose."
        ),
    )
    post = PostBounceMovementResult(
        status="UNAVAILABLE",
        confidence="INSUFFICIENT_EVIDENCE",
        unavailable_reason=(
            "Observed post-bounce 3D trajectory evidence is unavailable."
        ),
    )
    return DeliveryPhysicsResult(
        status=mode_status,
        analysis_id=analysis_id,
        coordinate_system=COORDINATE_SYSTEM_DESCRIPTION,
        calibration=calibration,
        fitted_parameters=FittedTrajectoryParameters(
            selected_model="IMAGE_QUADRATIC",
            origin_timestamp_seconds=accepted[0].timestamp_seconds,
        ),
        trajectory_samples=samples,
        accepted_observations=accepted,
        rejected_observations=rejected,
        delivery_interval=interval,
        bounce=bounce,
        speed=speed,
        pre_bounce_lateral_movement=lateral,
        post_bounce_movement=post,
        line_and_length=line_length,
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model="IMAGE_QUADRATIC",
            optimizer_status=(
                "deterministic_piecewise_weighted_polynomial_fit"
                if len(image_segments) > 1
                else "deterministic_weighted_polynomial_fit"
            ),
            iterations=2 * len(image_segments),
            inlier_frames=[item.frame_index for item in accepted],
            outlier_frames=[
                observations[index].frame_index for index in rejected_indexes
            ],
            weighted_reprojection_rmse_px=rmse,
            median_reprojection_error_px=float(np.median(errors[inliers])),
            maximum_inlier_error_px=float(np.max(errors[inliers])),
            processing_duration_seconds=round(
                time.perf_counter() - processing_started,
                6,
            ),
        ),
        confidence=_grade(score),
        confidence_score=round(score, 6),
        uncertainty_method="deterministic image residual envelope",
        unavailable_metrics=_unavailable_metrics(speed, lateral, post, bounce),
        warnings=[
            *calibration.warnings,
            "Projected image-space paths are estimates, not metric measurements.",
        ],
    )


def _fit_image_trajectory(
    observations: list[ObservedBallPoint],
):
    origin = observations[0].timestamp_seconds
    times = np.asarray(
        [item.timestamp_seconds - origin for item in observations],
        dtype=np.float64,
    )
    x = np.asarray([item.pixel_x for item in observations], dtype=np.float64)
    y = np.asarray([item.pixel_y for item in observations], dtype=np.float64)
    weights = np.asarray(
        [
            max(0.05, item.detector_confidence * item.tracker_confidence)
            for item in observations
        ],
        dtype=np.float64,
    )
    degree = 2 if len(observations) >= 4 else 1
    inliers = np.arange(len(observations))
    for _ in range(2):
        cx = np.polyfit(times[inliers], x[inliers], degree, w=weights[inliers])
        cy = np.polyfit(times[inliers], y[inliers], degree, w=weights[inliers])
        predicted_x = np.polyval(cx, times)
        predicted_y = np.polyval(cy, times)
        errors = np.hypot(predicted_x - x, predicted_y - y)
        threshold = _outlier_threshold(errors[inliers])
        next_inliers = np.where(errors <= threshold)[0]
        if len(next_inliers) < MIN_IMAGE_OBSERVATIONS:
            break
        inliers = next_inliers
    return cx, cy, inliers, errors


def _fit_image_segments(
    observations: list[ObservedBallPoint],
    bounce_frame: int | None,
) -> list[_ImageFitSegment]:
    all_indexes = np.arange(len(observations), dtype=int)
    groups = [all_indexes]
    if bounce_frame is not None:
        pre = np.asarray(
            [
                index
                for index, item in enumerate(observations)
                if item.frame_index <= bounce_frame
            ],
            dtype=int,
        )
        post = np.asarray(
            [
                index
                for index, item in enumerate(observations)
                if item.frame_index > bounce_frame
            ],
            dtype=int,
        )
        if len(pre) >= MIN_IMAGE_OBSERVATIONS and len(post) >= MIN_IMAGE_OBSERVATIONS:
            groups = [pre, post]

    segments: list[_ImageFitSegment] = []
    for group in groups:
        selected = [observations[index] for index in group]
        coefficients_x, coefficients_y, local_inliers, local_errors = (
            _fit_image_trajectory(selected)
        )
        global_errors = np.full(len(observations), np.nan, dtype=np.float64)
        global_errors[group] = local_errors
        segments.append(
            _ImageFitSegment(
                coefficients_x=coefficients_x,
                coefficients_y=coefficients_y,
                origin_timestamp_seconds=selected[0].timestamp_seconds,
                first_frame=selected[0].frame_index,
                last_frame=selected[-1].frame_index,
                global_inlier_indexes=group[local_inliers],
                global_errors_px=global_errors,
            )
        )
    return segments


def _image_samples(
    accepted: list[ObservedBallPoint],
    segments: list[_ImageFitSegment],
    fps: float,
    total_frames: int,
    rmse: float,
):
    first = accepted[0].frame_index
    last = accepted[-1].frame_index
    end = min(
        total_frames - 1,
        last + max(1, int(round(MAX_PROJECTION_SECONDS * fps))),
    )
    observed_by_frame = {item.frame_index: item for item in accepted}
    observed_frames = np.asarray(sorted(observed_by_frame), dtype=int)
    samples: list[TrajectorySample] = []
    for frame in range(first, end + 1):
        timestamp = frame / fps
        segment = next(
            (
                item
                for item in segments
                if item.first_frame <= frame <= item.last_frame
            ),
            segments[-1] if frame > segments[-1].last_frame else segments[0],
        )
        elapsed = timestamp - segment.origin_timestamp_seconds
        nearest_frame = int(
            observed_frames[np.argmin(np.abs(observed_frames - frame))]
        )
        delta = abs(frame - nearest_frame)
        provenance = (
            "OBSERVED"
            if frame in observed_by_frame
            else "RECONSTRUCTED"
            if frame <= last
            else "PROJECTED"
        )
        confidence = max(0.08, 1.0 - rmse / 35.0) * math.exp(
            -delta / max(2.0, fps * 0.18)
        )
        samples.append(
            TrajectorySample(
                frame_index=frame,
                timestamp_seconds=round(timestamp, 6),
                pixel_x=round(
                    float(np.polyval(segment.coefficients_x, elapsed)),
                    3,
                ),
                pixel_y=round(
                    float(np.polyval(segment.coefficients_y, elapsed)),
                    3,
                ),
                provenance=provenance,
                confidence=round(_clamp(confidence), 6),
                nearest_observation_frame=nearest_frame,
                nearest_observation_delta_frames=delta,
                nearest_observation_delta_seconds=round(delta / fps, 6),
            )
        )
    return samples, DeliveryInterval(
        start_frame=first,
        end_frame=end,
        first_observed_frame=first,
        last_observed_frame=last,
        terminal_reason="maximum_projection_horizon",
    )


def _ground_or_image_bounce(
    tracker_bounce: PrimaryBounceResult | None,
    calibration: CameraCalibration,
    fps: float,
) -> BouncePhysicsResult:
    if (
        tracker_bounce is None
        or tracker_bounce.bounce_detected is not True
        or tracker_bounce.bounce_frame is None
        or tracker_bounce.bounce_x is None
        or tracker_bounce.bounce_y is None
    ):
        return _unavailable_bounce()
    world = _image_to_physics_ground(
        calibration.image_to_pitch_homography,
        tracker_bounce.bounce_x,
        tracker_bounce.bounce_y,
    )
    if world is None:
        return BouncePhysicsResult(
            status="ESTIMATED",
            frame_index=tracker_bounce.bounce_frame,
            timestamp_seconds=(
                tracker_bounce.bounce_timestamp_seconds
                or tracker_bounce.bounce_frame / fps
            ),
            pixel_x=tracker_bounce.bounce_x,
            pixel_y=tracker_bounce.bounce_y,
            confidence="LOW",
            confidence_score=tracker_bounce.confidence,
            uncertainty_frames=3,
            directly_supported=True,
            evidence=["tracker_image_space_slope_transition"],
        )
    x, y = world
    if not (-PITCH_WIDTH_M <= x <= PITCH_WIDTH_M and -1 <= y <= PITCH_LENGTH_M + 1):
        return _unavailable_bounce()
    return BouncePhysicsResult(
        status="DETECTED",
        frame_index=tracker_bounce.bounce_frame,
        timestamp_seconds=(
            tracker_bounce.bounce_timestamp_seconds
            or tracker_bounce.bounce_frame / fps
        ),
        world_x_m=round(x, 6),
        world_y_m=round(y, 6),
        pixel_x=tracker_bounce.bounce_x,
        pixel_y=tracker_bounce.bounce_y,
        distance_from_striker_wicket_m=round(max(0.0, PITCH_LENGTH_M - y), 6),
        lateral_offset_m=round(x, 6),
        confidence=_grade(tracker_bounce.confidence),
        confidence_score=tracker_bounce.confidence,
        uncertainty_frames=3,
        uncertainty_m=0.35,
        directly_supported=True,
        evidence=["tracker_slope_transition", "metric_ground_plane_projection"],
    )


def _speed_analytics(
    samples: list[TrajectorySample],
    bounce: BouncePhysicsResult,
    observations: list[ObservedBallPoint],
) -> SpeedAnalytics:
    pre = [
        sample
        for sample in samples
        if sample.speed_mps is not None
        and (
            bounce.timestamp_seconds is None
            or sample.timestamp_seconds <= bounce.timestamp_seconds
        )
    ]
    if not pre:
        return SpeedAnalytics(
            confidence="INSUFFICIENT_EVIDENCE",
            unavailable_reason="No fitted metric pre-bounce velocity is available.",
        )
    speeds = [sample.speed_mps for sample in pre if sample.speed_mps is not None]
    if (
        not speeds
        or min(speeds) < MIN_REASONABLE_SPEED_MPS
        or max(speeds) > MAX_REASONABLE_SPEED_MPS
    ):
        return SpeedAnalytics(
            confidence="INSUFFICIENT_EVIDENCE",
            observed_temporal_span_seconds=round(
                observations[-1].timestamp_seconds
                - observations[0].timestamp_seconds,
                6,
            ),
            unavailable_reason="Fitted speed lies outside configured plausible bounds.",
        )
    observed_frames = {item.frame_index for item in observations}
    observed_pre = [item for item in pre if item.frame_index in observed_frames]
    earliest = observed_pre[0] if observed_pre else pre[0]
    at_bounce = min(
        pre,
        key=lambda item: abs(
            item.timestamp_seconds
            - (bounce.timestamp_seconds or pre[-1].timestamp_seconds)
        ),
    )
    post_speeds = [
        sample.speed_mps
        for sample in samples
        if sample.speed_mps is not None
        and bounce.timestamp_seconds is not None
        and sample.timestamp_seconds > bounce.timestamp_seconds
        and sample.provenance != "PROJECTED"
    ]
    uncertainty = max(2.0, (max(speeds) - min(speeds)) * 3.6 * 0.15)
    return SpeedAnalytics(
        earliest_measured_speed_kmh=round(earliest.speed_mps * 3.6, 2),
        average_pre_bounce_speed_kmh=round(float(np.mean(speeds)) * 3.6, 2),
        speed_at_bounce_kmh=round(at_bounce.speed_mps * 3.6, 2),
        average_post_bounce_speed_kmh=(
            round(float(np.mean(post_speeds)) * 3.6, 2)
            if post_speeds
            else None
        ),
        observed_temporal_span_seconds=round(
            observations[-1].timestamp_seconds
            - observations[0].timestamp_seconds,
            6,
        ),
        confidence="MEDIUM",
        uncertainty_kmh=round(uncertainty, 2),
    )


def _lateral_movement(
    parameters: np.ndarray,
    model_name: str,
    bounce: BouncePhysicsResult,
    observations: list[ObservedBallPoint],
) -> LateralMovementResult:
    if len(parameters) < 7:
        return LateralMovementResult(
            movement_m=0.0,
            movement_cm=0.0,
            direction="negligible",
            lateral_acceleration_mps2=0.0,
            observed_fraction=1.0,
            confidence="MEDIUM",
            uncertainty_m=0.05,
        )
    end_time = (
        (bounce.timestamp_seconds - observations[0].timestamp_seconds)
        if bounce.timestamp_seconds is not None
        else observations[-1].timestamp_seconds
        - observations[0].timestamp_seconds
    )
    end_time = max(0.0, end_time)
    movement = 0.5 * parameters[6] * end_time**2
    direction = (
        "negligible"
        if abs(movement) < 0.02
        else "toward_positive_x"
        if movement > 0
        else "toward_negative_x"
    )
    return LateralMovementResult(
        movement_m=round(float(movement), 6),
        movement_cm=round(float(movement * 100), 2),
        direction=direction,
        lateral_acceleration_mps2=round(float(parameters[6]), 6),
        observed_fraction=1.0,
        confidence="MEDIUM",
        uncertainty_m=round(max(0.03, abs(movement) * 0.25), 3),
    )


def _post_bounce_movement(
    parameters: np.ndarray,
    model_name: str,
    post_fit: _PostBounceFit | None,
    bounce: BouncePhysicsResult,
    observations: list[ObservedBallPoint],
) -> PostBounceMovementResult:
    if post_fit is None or bounce.timestamp_seconds is None:
        return PostBounceMovementResult(
            status="PROJECTED" if bounce.timestamp_seconds is not None else "UNAVAILABLE",
            confidence="INSUFFICIENT_EVIDENCE",
            unavailable_reason=(
                "At least four reliable post-bounce observations are required."
            ),
        )
    elapsed = bounce.timestamp_seconds - observations[0].timestamp_seconds
    before = _velocity_at(parameters, elapsed)
    after = post_fit.parameters
    turn = float(after[0] - before[0])
    last = observations[int(post_fit.observation_indexes[-1])]
    duration = last.timestamp_seconds - bounce.timestamp_seconds
    before_angle = math.degrees(math.atan2(-before[2], max(before[1], 1e-6)))
    after_angle = math.degrees(math.atan2(after[2], max(after[1], 1e-6)))
    speed_loss = max(0.0, np.linalg.norm(before) - np.linalg.norm(after)) * 3.6
    return PostBounceMovementResult(
        status="MEASURED",
        lateral_turn_mps=round(turn, 6),
        lateral_turn_cm_at_last_observation=round(turn * duration * 100, 2),
        speed_loss_kmh=round(float(speed_loss), 2),
        bounce_angle_change_degrees=round(after_angle - before_angle, 2),
        observed_points=len(post_fit.observation_indexes),
        confidence="MEDIUM",
    )


def _line_and_length(bounce: BouncePhysicsResult) -> LineLengthResult:
    if (
        bounce.world_x_m is None
        or bounce.distance_from_striker_wicket_m is None
    ):
        return LineLengthResult(
            line="unavailable",
            length="unavailable",
            unavailable_reason="A metric bounce point is unavailable.",
        )
    x = bounce.world_x_m
    line = "outside pitch right"
    for upper, label in LINE_BANDS_M:
        if x < upper:
            line = label
            break
    distance = bounce.distance_from_striker_wicket_m
    length = "very short"
    for upper, label in LENGTH_BANDS_M:
        if distance < upper:
            length = label
            break
    return LineLengthResult(
        line=line,
        length=length,
        bounce_distance_from_striker_m=distance,
        lateral_offset_from_middle_m=x,
    )


def _parameter_contract(
    parameters: np.ndarray,
    model_name: str,
    origin_timestamp: float,
    bounds_reached: list[str],
    post_fit: _PostBounceFit | None,
) -> FittedTrajectoryParameters:
    return FittedTrajectoryParameters(
        selected_model=model_name,
        origin_timestamp_seconds=origin_timestamp,
        x0_m=float(parameters[0]),
        y0_m=float(parameters[1]),
        z0_m=float(parameters[2]),
        vx0_mps=float(parameters[3]),
        vy0_mps=float(parameters[4]),
        vz0_mps=float(parameters[5]),
        lateral_acceleration_mps2=(
            float(parameters[6]) if len(parameters) >= 7 else 0.0
        ),
        forward_acceleration_mps2=(
            float(parameters[7]) if len(parameters) >= 8 else 0.0
        ),
        post_bounce_vx_mps=(
            float(post_fit.parameters[0]) if post_fit is not None else None
        ),
        post_bounce_vy_mps=(
            float(post_fit.parameters[1]) if post_fit is not None else None
        ),
        post_bounce_vz_mps=(
            float(post_fit.parameters[2]) if post_fit is not None else None
        ),
        parameter_bounds_reached=bounds_reached,
    )


def insufficient_physics_result(
    *,
    analysis_id: str,
    calibration: CameraCalibration,
    accepted: list[ObservedBallPoint],
    rejected: list[RejectedObservation],
    reason: str,
    processing_seconds: float,
) -> DeliveryPhysicsResult:
    bounce = _unavailable_bounce()
    speed = SpeedAnalytics(
        confidence="INSUFFICIENT_EVIDENCE",
        unavailable_reason=reason,
    )
    lateral = LateralMovementResult(
        direction="unavailable",
        confidence="INSUFFICIENT_EVIDENCE",
        unavailable_reason=reason,
    )
    post = PostBounceMovementResult(
        status="UNAVAILABLE",
        confidence="INSUFFICIENT_EVIDENCE",
        unavailable_reason=reason,
    )
    return DeliveryPhysicsResult(
        status="INSUFFICIENT_EVIDENCE",
        analysis_id=analysis_id,
        coordinate_system=COORDINATE_SYSTEM_DESCRIPTION,
        calibration=calibration,
        accepted_observations=accepted,
        rejected_observations=rejected,
        delivery_interval=DeliveryInterval(terminal_reason=reason),
        bounce=bounce,
        speed=speed,
        pre_bounce_lateral_movement=lateral,
        post_bounce_movement=post,
        line_and_length=_line_and_length(bounce),
        fit_diagnostics=FitDiagnostics(
            converged=False,
            selected_model="none",
            optimizer_status=reason,
            processing_duration_seconds=round(processing_seconds, 6),
        ),
        confidence="INSUFFICIENT_EVIDENCE",
        confidence_score=0.0,
        uncertainty_method="unavailable",
        unavailable_metrics=_unavailable_metrics(speed, lateral, post, bounce),
        warnings=[reason],
    )


def failed_physics_result(
    analysis_id: str,
    width: int,
    height: int,
    reason: str,
) -> DeliveryPhysicsResult:
    calibration = CameraCalibration(
        mode="IMAGE_SPACE_ONLY",
        confidence="UNAVAILABLE",
        image_width=width,
        image_height=height,
        failure_reason=reason,
    )
    result = insufficient_physics_result(
        analysis_id=analysis_id,
        calibration=calibration,
        accepted=[],
        rejected=[],
        reason=reason,
        processing_seconds=0.0,
    )
    return result.model_copy(update={"status": "FAILED"})


def _unavailable_bounce() -> BouncePhysicsResult:
    return BouncePhysicsResult(
        status="INSUFFICIENT_EVIDENCE",
        confidence="INSUFFICIENT_EVIDENCE",
        confidence_score=0.0,
        evidence=[],
    )


def _unavailable_metrics(
    speed: SpeedAnalytics,
    lateral: LateralMovementResult,
    post: PostBounceMovementResult,
    bounce: BouncePhysicsResult,
) -> list[MetricAvailability]:
    return [
        MetricAvailability(
            metric="metric_speed",
            available=speed.earliest_measured_speed_kmh is not None,
            reason=speed.unavailable_reason,
        ),
        MetricAvailability(
            metric="metric_bounce",
            available=bounce.world_y_m is not None,
            reason=(
                None
                if bounce.world_y_m is not None
                else "A reliable metric bounce point is unavailable."
            ),
        ),
        MetricAvailability(
            metric="pre_bounce_lateral_movement",
            available=lateral.movement_m is not None,
            reason=lateral.unavailable_reason,
        ),
        MetricAvailability(
            metric="post_bounce_turn",
            available=post.status == "MEASURED",
            reason=post.unavailable_reason,
        ),
        MetricAvailability(
            metric="exact_spin_rpm",
            available=False,
            reason="Ball surface rotation is not directly observable.",
        ),
        MetricAvailability(
            metric="exact_seam_angle",
            available=False,
            reason="Seam orientation is not directly observable.",
        ),
    ]


def _image_to_physics_ground(
    homography: Sequence[Sequence[float]] | None,
    pixel_x: float,
    pixel_y: float,
) -> tuple[float, float] | None:
    if homography is None:
        return None
    try:
        longitudinal, lateral = image_point_to_pitch_ground(
            homography,
            pixel_x,
            pixel_y,
        )
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
        return None
    if not math.isfinite(longitudinal) or not math.isfinite(lateral):
        return None
    canonical = calibration_to_canonical_world(
        float(longitudinal),
        float(lateral),
        0.0,
    )
    return canonical[0], canonical[1]


def physics_ground_to_image(
    homography: Sequence[Sequence[float]],
    lateral_x_m: float,
    longitudinal_y_m: float,
) -> tuple[float, float]:
    longitudinal, lateral, _ = canonical_to_calibration_world(
        lateral_x_m,
        longitudinal_y_m,
        0.0,
    )
    return pitch_ground_point_to_image(homography, longitudinal, lateral)


def _grade(score: float) -> ConfidenceGrade:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.5:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "INSUFFICIENT_EVIDENCE"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))
