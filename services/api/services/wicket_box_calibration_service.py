"""Two-wicket box calibration for Virtual Pitch Replay scene setup."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2
import numpy as np
from PIL import Image

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CRICVISION_PITCH_V1,
    CricketPitchDimensions,
)

from ..schemas.real_pitch_registration import (
    CameraIntrinsicsCandidate,
    CameraPoseCandidate,
    PoseUncertainty,
    RealPitchRegistrationResult,
    RegistrationCorrespondence,
    RegistrationDiagnostics,
)
from ..schemas.wicket_box_calibration import (
    CalibrationResult,
    PixelPoint,
    ReprojectionDiagnostic,
    StumpLandmark,
    WicketBox,
    WicketBoxCalibrationAcceptRequest,
    WicketBoxCalibrationAcceptResponse,
    WicketBoxCalibrationCandidateSummary,
    WicketBoxCalibrationDetectResponse,
    WicketBoxCalibrationRegisterRequest,
    WicketBoxCalibrationRegisterResponse,
    WicketBoxPairValidationResult,
    WicketBoxRegistrationSummary,
    WicketBoxRole,
    validate_wicket_box_pair,
)
from ..schemas.delivery_physics import CameraCalibration
from ..schemas.device_calibration import DeviceLensProfile
from ..schemas.wicket_observation import (
    AssignmentHypothesis,
    PixelBox,
    PixelPoint as ObservationPixelPoint,
    RoiMetadata,
    SetupFrameCandidate,
    WicketLandmarkObservation,
    WicketObservation,
    WicketObservationDiagnostics,
    WicketObservationResult,
    WicketRegionObservation,
)
from .device_calibration_service import load_device_profile_for_frame
from .frame_timing import resolve_frame_timestamp
from .real_pitch_registration_service import (
    AMBIGUITY_SCORE_GAP,
    MIN_ASSISTED_POINTLIKE_ANCHORS,
    _perturbation_uncertainty,
    _read_setup_frame,
    build_intrinsics_candidates,
    build_registration_correspondences,
    solve_pose_candidate,
)
from .stump_detector_service import (
    STUMP_MODEL_PATH,
    _detect_end,
    _load_model,
    _virtual_stumps_from_bbox,
)
from .video_analysis_service import (
    ANALYSIS_ID_PATTERN,
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .wicket_observation_service import extract_wicket_landmarks


RESULT_FILENAME = "wicket_box_calibration_v1.json"
ACCEPTED_FILENAME = "accepted_wicket_box_calibration_v1.json"
DEBUG_REPORT_FILENAME = "wicket_box_landmark_debug_v1.json"
LOW_LANDMARK_CONFIDENCE = 0.35
EXPECTED_STUMPS_PER_WICKET = 3
MIN_BOX_WIDTH_PX = 24.0
MIN_BOX_HEIGHT_PX = 32.0
MAX_ACCEPTANCE_RMSE_PX = 8.0
MAX_WICKET_RMSE_PX = 12.0
# Both constants above are flat pixel counts tuned against 720x1280 phone
# clips. A 4K frame is over five times wider, and its stump features are
# correspondingly larger, so the same physical accuracy shows up as a much
# bigger pixel number — 25.9px at 3840 wide is 0.67% of the frame, against
# 1.67% for 12px at 720 wide. Judging both by one pixel count rejects better
# calibrations at higher resolution than it accepts at lower.
RMSE_GATE_FRACTION_OF_WIDTH = 0.012
REFERENCE_FRAME_WIDTH_PX = 720.0
ORIENTATION_AMBIGUITY_GAP = AMBIGUITY_SCORE_GAP
MANDATORY_PHYSICAL_CHECKS = frozenset(
    {
        "finite_pose",
        "camera_above_pitch",
        "all_anchors_in_front",
        "camera_faces_pitch",
        "focal_length_plausible",
        "near_far_projected_size_order",
        "projected_wickets_reasonable",
    }
)

AssignmentChoice = Literal["A", "B"]


def detect_wicket_box_calibration(
    analysis_id: str,
    request: WicketBoxCalibrationRegisterRequest,
) -> WicketBoxCalibrationDetectResponse:
    _require_analysis_id(analysis_id)
    _require_matching_analysis_id(analysis_id, request.analysis_id)
    validation = _validate_boxes(request.near_wicket_box, request.far_wicket_box)
    if not validation.valid:
        return WicketBoxCalibrationDetectResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            message="; ".join(validation.messages) or "Wicket boxes are invalid.",
        )

    load_video_analysis(analysis_id)
    frame, _ = _read_setup_frame(analysis_id, request.calibration_frame_index)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = _detect_landmarks_in_boxes(
        image,
        frame,
        request.near_wicket_box,
        request.far_wicket_box,
        fps=float(load_video_analysis(analysis_id).fps),
    )
    result = CalibrationResult(
        status="DETECTED",
        analysis_id=analysis_id,
        calibration_frame_index=request.calibration_frame_index,
        source_image_width=request.source_image_width,
        source_image_height=request.source_image_height,
        near_wicket_box=request.near_wicket_box,
        far_wicket_box=request.far_wicket_box,
        stump_landmarks=landmarks,
        automatic_stump_landmarks=list(landmarks),
        validation_status=validation.validation_status,
        message="Automatic stump landmarks extracted inside submitted wicket boxes.",
    )
    _write_result(analysis_id, result)
    return WicketBoxCalibrationDetectResponse(
        success=True,
        status="READY",
        analysis_id=analysis_id,
        calibration_frame_index=request.calibration_frame_index,
        source_image_width=request.source_image_width,
        source_image_height=request.source_image_height,
        near_wicket_box=request.near_wicket_box,
        far_wicket_box=request.far_wicket_box,
        stump_landmarks=landmarks,
        message=result.message,
    )


def register_wicket_box_calibration(
    analysis_id: str,
    request: WicketBoxCalibrationRegisterRequest,
    *,
    assignment_hypothesis: AssignmentChoice | None = None,
) -> WicketBoxCalibrationRegisterResponse:
    _require_analysis_id(analysis_id)
    _require_matching_analysis_id(analysis_id, request.analysis_id)
    validation = _validate_boxes(request.near_wicket_box, request.far_wicket_box)
    if not validation.valid:
        return WicketBoxCalibrationRegisterResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            validation=validation,
            calibration=None,
            message="; ".join(validation.messages) or "Wicket boxes are invalid.",
        )

    load_video_analysis(analysis_id)
    frame, analysis = _read_setup_frame(analysis_id, request.calibration_frame_index)
    fps = float(analysis.fps)
    landmarks = request.stump_landmarks
    automatic_landmarks: list[StumpLandmark] = []
    if landmarks and any(item.provenance == "USER_CORRECTED" for item in landmarks):
        prior = load_wicket_box_calibration(analysis_id)
        if prior.automatic_stump_landmarks:
            automatic_landmarks = list(prior.automatic_stump_landmarks)
        elif prior.stump_landmarks:
            automatic_landmarks = [
                item.model_copy(update={"provenance": "AUTOMATIC"})
                for item in prior.stump_landmarks
            ]
    if not landmarks:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = _detect_landmarks_in_boxes(
            image,
            frame,
            request.near_wicket_box,
            request.far_wicket_box,
            fps=fps,
        )

    landmark_reasons = _landmark_validation_reasons(landmarks)
    if landmark_reasons:
        calibration = CalibrationResult(
            status="FAILED",
            analysis_id=analysis_id,
            calibration_frame_index=request.calibration_frame_index,
            source_image_width=request.source_image_width,
            source_image_height=request.source_image_height,
            device_id=request.device_id,
            pitch_geometry=request.pitch_geometry,
            near_wicket_box=request.near_wicket_box,
            far_wicket_box=request.far_wicket_box,
            stump_landmarks=landmarks,
            automatic_stump_landmarks=automatic_landmarks or [
                item for item in landmarks if item.provenance == "AUTOMATIC"
            ],
            validation_status=validation.validation_status,
            warnings=landmark_reasons,
            message="; ".join(landmark_reasons),
        )
        _write_result(analysis_id, calibration)
        return WicketBoxCalibrationRegisterResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            validation=validation,
            calibration=calibration,
            message=calibration.message,
        )

    observation = _observation_from_landmarks(
        analysis_id=analysis_id,
        frame_index=request.calibration_frame_index,
        fps=fps,
        near_box=request.near_wicket_box,
        far_box=request.far_wicket_box,
        landmarks=landmarks,
    )
    registration, warnings, orientation_required, registration_summary = _register_from_observation(
        analysis_id,
        observation,
        frame,
        assignment_hypothesis=assignment_hypothesis,
        device_id=request.device_id,
        dimensions=(
            request.pitch_geometry.to_dimensions()
            if request.pitch_geometry is not None
            else None
        ),
    )
    calibration = _calibration_from_registration(
        analysis_id=analysis_id,
        request=request,
        landmarks=landmarks,
        automatic_landmarks=automatic_landmarks,
        validation=validation,
        registration=registration,
        warnings=warnings,
        registration_summary=registration_summary,
    )
    if orientation_required and assignment_hypothesis is None:
        calibration.message = (
            "Registration complete. Review the recommended calibration or choose "
            "the alternative if both are physically valid."
        )
        calibration.warnings = [
            warning
            for warning in calibration.warnings
            if warning != "orientation_hypothesis_required"
        ]

    selected = registration.selected_candidate
    debug_filename = _write_landmark_debug_report(
        analysis_id,
        landmarks=landmarks,
        correspondences=registration.correspondences,
        candidate=selected,
        frame=frame,
        width=request.source_image_width,
        height=request.source_image_height,
    )
    if selected is None or registration.status == "REGISTRATION_FAILED":
        calibration.status = "FAILED"
        calibration.message = registration.message
        calibration.warnings = [*calibration.warnings, f"debug_report:{debug_filename}"]
        _write_result(analysis_id, calibration)
        return WicketBoxCalibrationRegisterResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            validation=validation,
            calibration=calibration,
            message=calibration.message,
        )

    reject_reasons = _registration_rejection_reasons(selected, registration)
    calibration.warnings = [*calibration.warnings, f"debug_report:{debug_filename}"]
    if reject_reasons:
        calibration.status = "FAILED"
        calibration.message = "; ".join(reject_reasons)
        calibration.warnings = [*calibration.warnings, *reject_reasons]
        _write_result(analysis_id, calibration)
        return WicketBoxCalibrationRegisterResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            validation=validation,
            calibration=calibration,
            message=calibration.message,
        )

    calibration.status = "REGISTERED"
    calibration.message = "Camera pose registered from wicket-box landmarks."
    _write_result(analysis_id, calibration)
    return WicketBoxCalibrationRegisterResponse(
        success=True,
        status="READY",
        analysis_id=analysis_id,
        validation=validation,
        calibration=calibration,
        message=calibration.message,
    )


def accept_wicket_box_calibration(
    analysis_id: str,
    request: WicketBoxCalibrationAcceptRequest,
    *,
    assignment_hypothesis: AssignmentChoice | None = None,
) -> WicketBoxCalibrationAcceptResponse:
    _require_analysis_id(analysis_id)
    _require_matching_analysis_id(analysis_id, request.analysis_id)
    if not request.accept_registered_calibration:
        return WicketBoxCalibrationAcceptResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            calibration=None,
            message="accept_registered_calibration must be true to accept.",
        )

    stored = load_wicket_box_calibration(analysis_id)
    if stored.status not in {"REGISTERED", "ACCEPTED"}:
        return WicketBoxCalibrationAcceptResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            calibration=stored,
            message="Run successful registration before acceptance.",
        )

    if stored.near_wicket_box is None or stored.far_wicket_box is None:
        return WicketBoxCalibrationAcceptResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            calibration=stored,
            message="Registered wicket boxes are missing.",
        )

    if assignment_hypothesis and (
        "orientation_hypothesis_required" in stored.warnings
        or "orientation_ambiguous" in stored.warnings
    ):
        register_request = WicketBoxCalibrationRegisterRequest(
            analysis_id=analysis_id,
            calibration_frame_index=stored.calibration_frame_index,
            source_image_width=stored.source_image_width,
            source_image_height=stored.source_image_height,
            near_wicket_box=stored.near_wicket_box,
            far_wicket_box=stored.far_wicket_box,
            stump_landmarks=stored.stump_landmarks,
            # Re-solving for a forced orientation must stay on the same pitch
            # the operator declared, or the retry falls back to regulation.
            pitch_geometry=stored.pitch_geometry,
        )
        registered = register_wicket_box_calibration(
            analysis_id,
            register_request,
            assignment_hypothesis=assignment_hypothesis,
        )
        if not registered.success or registered.calibration is None:
            return WicketBoxCalibrationAcceptResponse(
                success=False,
                status="FAILED",
                analysis_id=analysis_id,
                calibration=registered.calibration,
                message=registered.message,
            )
        stored = registered.calibration

    reject_reasons = _acceptance_rejection_reasons(stored)
    if reject_reasons:
        failed = stored.model_copy(
            update={
                "status": "FAILED",
                "message": "; ".join(reject_reasons),
                "warnings": [*stored.warnings, *reject_reasons],
            }
        )
        _write_result(analysis_id, failed)
        return WicketBoxCalibrationAcceptResponse(
            success=False,
            status="FAILED",
            analysis_id=analysis_id,
            calibration=failed,
            message=failed.message,
        )

    accepted = stored.model_copy(
        update={
            "status": "ACCEPTED",
            "accepted_at": datetime.now(timezone.utc),
            "message": "Wicket-box calibration accepted and frozen.",
            "warnings": [
                warning
                for warning in stored.warnings
                if warning != "orientation_hypothesis_required"
            ],
        }
    )
    _write_result(analysis_id, accepted)
    _write_accepted_snapshot(analysis_id, accepted)
    from .video_ball_tracking_service import (
        invalidate_tracking_after_calibration_change,
    )

    invalidate_tracking_after_calibration_change(analysis_id)
    return WicketBoxCalibrationAcceptResponse(
        success=True,
        status="READY",
        analysis_id=analysis_id,
        calibration=accepted,
        message=accepted.message,
    )


_ACCEPTED_SNAPSHOT_REQUIRED_FIELDS = (
    "camera_matrix",
    "rotation_matrix",
    "translation_vector",
)


def _describe_registration_failure(
    candidates: list[CameraPoseCandidate],
) -> str:
    """Say which gate rejected the best attempt, not just that all failed.

    Every candidate already records its reprojection error, its failure reasons
    and which plausibility checks it failed. Collapsing all of that into
    "Registration failed for all pose candidates" leaves the operator guessing
    between a mis-declared pitch, mis-placed stump points and a bad lens — which
    need completely different fixes.
    """
    if not candidates:
        return (
            "Registration failed: no pose candidates were attempted. The stump "
            "points did not reach the solver."
        )

    scored = [item for item in candidates if item.reprojection_rmse_px is not None]
    best = (
        min(scored, key=lambda item: item.reprojection_rmse_px or float("inf"))
        if scored
        else max(candidates, key=lambda item: item.score)
    )

    parts: list[str] = [f"Registration failed for all {len(candidates)} pose candidates."]
    if best.reprojection_rmse_px is not None:
        parts.append(f"Best attempt was {best.reprojection_rmse_px:.1f}px off.")

    failed_checks = [
        check.check_id
        for check in best.plausibility_checks
        if not check.passed
    ]
    if failed_checks:
        parts.append(f"Failed: {', '.join(failed_checks)}.")
    elif best.reprojection_rmse_px is not None and best.reprojection_rmse_px > 12:
        # All physical checks pass and only the error gate rejects it — the
        # signature of a declared pitch that does not match the real one.
        parts.append(
            "All physical checks passed, so the placed points are probably fine "
            "but the declared pitch size does not match the real rig."
        )
    if best.failure_reasons:
        parts.append(" ".join(best.failure_reasons[:2]))

    return " ".join(parts)


def load_active_accepted_wicket_box_calibration(
    analysis_id: str,
) -> dict[str, Any] | None:
    """Return frozen accepted snapshot with bridge extrinsics, or None if absent."""
    _require_analysis_id(analysis_id)
    path = _accepted_path(analysis_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("schema_version") != "wicket_box_calibration_accepted_v1":
        return None
    if not payload.get("frozen"):
        return None
    if payload.get("analysis_id") != analysis_id:
        return None
    if any(payload.get(field) is None for field in _ACCEPTED_SNAPSHOT_REQUIRED_FIELDS):
        return None
    rotation = np.asarray(payload["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(payload["translation_vector"], dtype=np.float64).reshape(
        3, 1
    )
    rotation_vector, _ = cv2.Rodrigues(rotation)
    payload["rotation_vector"] = rotation_vector.reshape(-1).tolist()
    payload["camera_world_position"] = (-rotation.T @ translation).reshape(-1).tolist()
    return payload


def build_camera_calibration_from_pose_candidate(
    candidate: CameraPoseCandidate,
    width: int,
    height: int,
) -> CameraCalibration:
    """Build physics CameraCalibration from a registration pose candidate."""
    if (
        not candidate.solver_success
        or candidate.rotation_vector is None
        or candidate.translation_vector is None
        or candidate.rotation_matrix is None
    ):
        raise ValueError("Pose candidate is missing a solved camera pose.")
    camera_matrix = np.asarray(
        _camera_matrix_from_intrinsics(
            candidate.intrinsics.focal_length_x_px,
            width,
            height,
        ),
        dtype=np.float64,
    )
    rotation_matrix = np.asarray(candidate.rotation_matrix, dtype=np.float64)
    rotation_vector = np.asarray(candidate.rotation_vector, dtype=np.float64).reshape(
        3, 1
    )
    translation = np.asarray(candidate.translation_vector, dtype=np.float64).reshape(
        3, 1
    )
    projection = camera_matrix @ np.hstack([rotation_matrix, translation])
    rmse = candidate.reprojection_rmse_px
    confidence_score = (
        max(0.0, min(1.0, 1.0 - float(rmse) / 20.0))
        if rmse is not None
        else 0.7
    )
    distortion = list(candidate.intrinsics.distortion_coefficients)
    return CameraCalibration(
        mode="METRIC_3D",
        confidence="HIGH" if confidence_score >= 0.75 else "MEDIUM",
        world_coordinate_system=CRICVISION_PITCH_V1,
        image_width=width,
        image_height=height,
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=[float(value) for value in distortion],
        rotation_vector=rotation_vector.reshape(-1).tolist(),
        rotation_matrix=rotation_matrix.tolist(),
        translation_vector=translation.reshape(-1).tolist(),
        projection_matrix=projection.tolist(),
        correspondences_used=len(candidate.inlier_correspondence_ids),
        reprojection_error_px=float(rmse) if rmse is not None else None,
        calibration_confidence=confidence_score,
        warnings=["Track-refined wicket-box pose candidate."],
    )


def _intrinsics_from_focal(
    focal_px: float,
    width: int,
    height: int,
) -> CameraIntrinsicsCandidate:
    lower = width * 0.35
    upper = width * 3.5
    fov = math.degrees(2.0 * math.atan(width / (2.0 * focal_px)))
    return CameraIntrinsicsCandidate(
        candidate_id=f"focal_refine_{int(round(focal_px))}",
        focal_length_x_px=focal_px,
        focal_length_y_px=focal_px,
        principal_point_x_px=width / 2.0,
        principal_point_y_px=height / 2.0,
        distortion_coefficients=[0.0] * 5,
        source="bounded_image_hypothesis",
        confidence="LOW",
        horizontal_fov_degrees=fov,
        lower_focal_bound_px=lower,
        upper_focal_bound_px=upper,
        distortion_assumption=(
            "Zero distortion; track-guided focal refinement around accepted pose."
        ),
    )


def _intrinsics_from_device_profile(
    profile: DeviceLensProfile,
) -> CameraIntrinsicsCandidate:
    """The one candidate worth trying when the lens has actually been measured.

    Focal bounds are held to +/-2% of the measured value: refinement is free to
    absorb a little cropping or stabilisation, but not to re-open the focal
    length that the checkerboard already settled.
    """
    focal = profile.focal_length_x_px
    fov = math.degrees(
        2.0 * math.atan(profile.image_width / (2.0 * focal))
    )
    return CameraIntrinsicsCandidate(
        candidate_id="device_lens",
        focal_length_x_px=focal,
        focal_length_y_px=profile.focal_length_y_px,
        principal_point_x_px=profile.principal_point_x_px,
        principal_point_y_px=profile.principal_point_y_px,
        distortion_coefficients=list(profile.distortion_coefficients),
        source="device_calibration",
        confidence="HIGH",
        horizontal_fov_degrees=fov,
        lower_focal_bound_px=focal * 0.98,
        upper_focal_bound_px=focal * 1.02,
        distortion_assumption=(
            "Measured distortion from this device's checkerboard calibration "
            f"(RMS {profile.quality.rms_reprojection_px:.2f} px, "
            f"{profile.quality.views_used} views)."
        ),
    )


def _intrinsics_candidates_for_frame(
    device_id: str | None,
    width: int,
    height: int,
) -> list[CameraIntrinsicsCandidate]:
    profile = load_device_profile_for_frame(device_id, width, height)
    if profile is None:
        return build_intrinsics_candidates(width, height)
    return [_intrinsics_from_device_profile(profile)]


def _anchor_candidate_for_accepted(
    stored: CalibrationResult,
    candidates: list[CameraPoseCandidate],
) -> CameraPoseCandidate | None:
    if not stored.camera_matrix:
        return None
    fx = float(stored.camera_matrix[0][0])
    valid = [
        candidate
        for candidate in candidates
        if candidate.solver_success
        and _mandatory_physical_validation_passed(candidate)[0]
    ]
    pool = valid or [candidate for candidate in candidates if candidate.solver_success]
    if not pool:
        return None
    return min(
        pool,
        key=lambda candidate: abs(candidate.intrinsics.focal_length_x_px - fx),
    )


def _stored_pitch_dimensions(
    stored: CalibrationResult,
) -> CricketPitchDimensions | None:
    """Re-solve against the pitch the stored pose was originally solved on."""
    if stored.pitch_geometry is None:
        return None
    return stored.pitch_geometry.to_dimensions()


def list_dense_focal_calibration_candidates(
    analysis_id: str,
    *,
    center_focal_px: float,
    assignment_hypothesis: str,
    lateral_mapping: str,
    relative_span: float = 0.22,
    steps: int = 17,
) -> list[CameraPoseCandidate]:
    """Sweep focal length around the accepted value and re-solve stump pose."""
    stored = load_wicket_box_calibration(analysis_id)
    if (
        stored.status not in {"REGISTERED", "ACCEPTED"}
        or not stored.stump_landmarks
        or stored.near_wicket_box is None
        or stored.far_wicket_box is None
    ):
        return []
    width = stored.source_image_width
    height = stored.source_image_height
    lower = max(width * 0.35, center_focal_px * (1.0 - relative_span))
    upper = min(width * 3.5, center_focal_px * (1.0 + relative_span))
    if upper <= lower:
        return []

    load_video_analysis(analysis_id)
    frame, analysis = _read_setup_frame(analysis_id, stored.calibration_frame_index)
    observation = _observation_from_landmarks(
        analysis_id=analysis_id,
        frame_index=stored.calibration_frame_index,
        fps=float(analysis.fps),
        near_box=stored.near_wicket_box,
        far_box=stored.far_wicket_box,
        landmarks=stored.stump_landmarks,
    )
    dimensions = _stored_pitch_dimensions(stored)
    correspondences = build_registration_correspondences(
        observation,
        assignment_hypothesis=assignment_hypothesis,
        lateral_mapping=lateral_mapping,
        dimensions=dimensions,
    )
    candidates: list[CameraPoseCandidate] = []
    for focal_px in np.linspace(lower, upper, steps):
        intrinsics = _intrinsics_from_focal(float(focal_px), width, height)
        candidate_id = (
            f"dense:{assignment_hypothesis}:{lateral_mapping}:"
            f"{intrinsics.candidate_id}"
        )
        candidate = solve_pose_candidate(
            candidate_id,
            assignment_hypothesis,
            lateral_mapping,
            stored.calibration_frame_index,
            intrinsics,
            correspondences,
            frame,
            observation,
            minimum_pointlike_anchors=MIN_ASSISTED_POINTLIKE_ANCHORS,
            dimensions=dimensions,
        )
        if not candidate.solver_success:
            continue
        uncertainty = _perturbation_uncertainty(
            candidate,
            correspondences,
            width,
            height,
            minimum_pointlike_anchors=MIN_ASSISTED_POINTLIKE_ANCHORS,
            dimensions=dimensions,
        )
        candidate = candidate.model_copy(update={"uncertainty": uncertainty})
        if _mandatory_physical_validation_passed(candidate)[0]:
            candidates.append(candidate)
    return candidates


def list_track_refineable_calibration_candidates(
    analysis_id: str,
) -> list[CameraPoseCandidate]:
    """Re-enumerate physically valid registration candidates from stored landmarks."""
    stored = load_wicket_box_calibration(analysis_id)
    if stored.status not in {"REGISTERED", "ACCEPTED"}:
        return []
    if (
        not stored.stump_landmarks
        or stored.near_wicket_box is None
        or stored.far_wicket_box is None
    ):
        return []
    load_video_analysis(analysis_id)
    frame, analysis = _read_setup_frame(analysis_id, stored.calibration_frame_index)
    observation = _observation_from_landmarks(
        analysis_id=analysis_id,
        frame_index=stored.calibration_frame_index,
        fps=float(analysis.fps),
        near_box=stored.near_wicket_box,
        far_box=stored.far_wicket_box,
        landmarks=stored.stump_landmarks,
    )
    registration, _, _, _ = _register_from_observation(
        analysis_id,
        observation,
        frame,
        assignment_hypothesis=None,
        device_id=stored.device_id,
        dimensions=_stored_pitch_dimensions(stored),
    )
    candidates = [
        candidate
        for candidate in registration.candidates
        if candidate.solver_success
        and _mandatory_physical_validation_passed(candidate)[0]
    ]
    anchor = _anchor_candidate_for_accepted(stored, candidates)
    if anchor is None:
        return candidates
    dense = list_dense_focal_calibration_candidates(
        analysis_id,
        center_focal_px=anchor.intrinsics.focal_length_x_px,
        assignment_hypothesis=anchor.assignment_hypothesis,
        lateral_mapping=anchor.lateral_mapping,
    )
    seen = {candidate.candidate_id for candidate in candidates}
    candidates.extend(
        candidate for candidate in dense if candidate.candidate_id not in seen
    )
    return candidates


def persist_pose_candidate_as_accepted(
    analysis_id: str,
    candidate: CameraPoseCandidate,
    stored: CalibrationResult,
) -> None:
    """Persist a track-scored pose candidate into the accepted calibration snapshot."""
    if (
        candidate.rotation_matrix is None
        or candidate.translation_vector is None
    ):
        raise ValueError("Pose candidate is missing extrinsics.")
    diagnostics: list[ReprojectionDiagnostic] = []
    for residual in candidate.reprojection_residuals:
        if residual.observed_pixel is None:
            continue
        diagnostics.append(
            ReprojectionDiagnostic(
                landmark_id=residual.correspondence_id,
                observed_pixel_x=residual.observed_pixel.x,
                observed_pixel_y=residual.observed_pixel.y,
                reprojected_pixel_x=residual.projected_pixel.x,
                reprojected_pixel_y=residual.projected_pixel.y,
                error_px=residual.residual_px,
            )
        )
    accepted = stored.model_copy(
        update={
            "status": "ACCEPTED",
            "accepted_at": datetime.now(timezone.utc),
            "camera_matrix": _camera_matrix_from_intrinsics(
                candidate.intrinsics.focal_length_x_px,
                stored.source_image_width,
                stored.source_image_height,
            ),
            "rotation_matrix": candidate.rotation_matrix,
            "translation_vector": candidate.translation_vector,
            "distortion_coefficients": list(
                candidate.intrinsics.distortion_coefficients
            ),
            "reprojection_rmse_px": candidate.reprojection_rmse_px,
            "reprojection_diagnostics": diagnostics,
            "message": "Track-refined wicket-box calibration accepted.",
        }
    )
    _write_result(analysis_id, accepted)
    _write_accepted_snapshot(analysis_id, accepted)


def load_wicket_box_calibration(analysis_id: str) -> CalibrationResult:
    _require_analysis_id(analysis_id)
    path = _result_path(analysis_id)
    if not path.is_file():
        return CalibrationResult(
            status="NOT_STARTED",
            analysis_id=analysis_id,
            calibration_frame_index=0,
            source_image_width=1,
            source_image_height=1,
            message="Wicket-box calibration has not started.",
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationResult.model_validate(payload)


def _validate_boxes(
    near_box: WicketBox,
    far_box: WicketBox,
) -> WicketBoxPairValidationResult:
    result = validate_wicket_box_pair(near_box, far_box)
    messages = list(result.messages)
    for label, box in (("NEAR", near_box), ("FAR", far_box)):
        if box.width < MIN_BOX_WIDTH_PX or box.height < MIN_BOX_HEIGHT_PX:
            messages.append(
                f"{label} wicket box is too small "
                f"({box.width:.1f}x{box.height:.1f}px)."
            )
    if messages and result.validation_status == "VALID":
        validation_status = "INVALID_DIMENSIONS"
    else:
        validation_status = result.validation_status
    return WicketBoxPairValidationResult(
        valid=not messages,
        near_box=near_box,
        far_box=far_box,
        validation_status=validation_status,
        overlap_fraction=result.overlap_fraction,
        role_order_valid=result.role_order_valid,
        messages=messages,
    )


def _detect_landmarks_in_boxes(
    image: Image.Image,
    frame: np.ndarray,
    near_box: WicketBox,
    far_box: WicketBox,
    *,
    fps: float | None = None,
) -> list[StumpLandmark]:
    landmarks: list[StumpLandmark] = []
    yolo_detections: dict[str, dict[str, Any] | None] = {"NEAR": None, "FAR": None}
    if STUMP_MODEL_PATH.is_file():
        try:
            model = _load_model(str(STUMP_MODEL_PATH))
            for role, box in (("NEAR", near_box), ("FAR", far_box)):
                normalized = _box_to_normalized(box)
                legacy_key = "non_striker" if role == "NEAR" else "striker"
                yolo_detections[role] = _detect_end(
                    model,
                    image,
                    legacy_key,
                    normalized,
                    box.source_image_width,
                    box.source_image_height,
                )
        except Exception:
            yolo_detections = {"NEAR": None, "FAR": None}

    for role, box in (("NEAR", near_box), ("FAR", far_box)):
        roi_image, roi, region = _roi_from_box(frame, box, fps=fps)
        extracted, _, _, _ = extract_wicket_landmarks(
            roi_image,
            roi,
            region=region,
        )
        yolo = yolo_detections.get(role)
        yolo_stumps = None
        if yolo and yolo.get("found") and yolo.get("bbox"):
            yolo_stumps = _virtual_stumps_from_bbox(
                yolo["bbox"],
                frame_width=box.source_image_width,
                frame_height=box.source_image_height,
            )["stumps"]
        landmarks.extend(
            _stump_landmarks_for_wicket(
                role,
                extracted,
                yolo_stumps=yolo_stumps,
            )
        )
    return landmarks


def _stump_landmarks_for_wicket(
    role: Literal["NEAR", "FAR"],
    extracted: list[Any],
    *,
    yolo_stumps: list[dict[str, Any]] | None,
) -> list[StumpLandmark]:
    by_id = {
        item.semantic_id: item
        for item in extracted
        if item.geometry_type == "POINT"
    }
    yolo_by_name = {
        item["name"]: item
        for item in (yolo_stumps or [])
    }
    output: list[StumpLandmark] = []
    for identity, detailed_base, coarse_base_id, coarse_top_id in (
        ("LEFT", "left_stump_base", "wicket_outer_left_base", "wicket_outer_left_top"),
        ("MIDDLE", "middle_stump_base", "wicket_base_center", "wicket_top_center"),
        ("RIGHT", "right_stump_base", "wicket_outer_right_base", "wicket_outer_right_top"),
    ):
        base_item = by_id.get(detailed_base)
        top_item = by_id.get(detailed_base.replace("_base", "_top"))
        yolo_item = yolo_by_name.get(identity.lower())
        base_point = _landmark_point(base_item) or _yolo_point(yolo_item, "base")
        top_point = _landmark_point(top_item) or _yolo_point(yolo_item, "top")
        if base_point is None and coarse_base_id in by_id:
            base_point = _landmark_point(by_id[coarse_base_id])
        if top_point is None and coarse_top_id in by_id:
            top_point = _landmark_point(by_id[coarse_top_id])
        if base_point is None:
            continue
        if top_point is None:
            top_point = PixelPoint(x=base_point.x, y=max(0.0, base_point.y - 8.0))
        centre = PixelPoint(
            x=(base_point.x + top_point.x) / 2,
            y=(base_point.y + top_point.y) / 2,
        )
        confidence = _landmark_confidence(base_item, top_item, yolo_item)
        visibility = "VISIBLE"
        if base_item and base_item.status != "AVAILABLE":
            visibility = "PARTIAL"
        if top_item and top_item.status != "AVAILABLE":
            visibility = "PARTIAL" if visibility == "VISIBLE" else "OCCLUDED"
        output.append(
            StumpLandmark(
                wicket_role=role,
                stump_identity=identity,
                base=base_point,
                top=top_point,
                centre=centre,
                visibility=visibility,
                confidence=confidence,
                provenance="AUTOMATIC",
            )
        )
    return output


def _observation_from_landmarks(
    *,
    analysis_id: str,
    frame_index: int,
    fps: float,
    near_box: WicketBox,
    far_box: WicketBox,
    landmarks: list[StumpLandmark],
) -> WicketObservationResult:
    setup = SetupFrameCandidate(
        frame_index=frame_index,
        timestamp_seconds=resolve_frame_timestamp(frame_index, fps=fps)[0],
        image_width=near_box.source_image_width,
        image_height=near_box.source_image_height,
        score=0.9,
        sharpness=100.0,
        brightness=120.0,
        wicket_detection_count=2,
        mean_detector_confidence=0.9,
        detection_stability=0.9,
        obstruction_score=0.0,
        selected=True,
    )
    near = _wicket_observation_from_landmarks("near", near_box, landmarks, frame_index, fps)
    far = _wicket_observation_from_landmarks("far", far_box, landmarks, frame_index, fps)
    return WicketObservationResult(
        analysis_id=analysis_id,
        status="READY_FOR_REGISTRATION_EXPERIMENT",
        setup_frame=setup,
        supporting_frames=[setup],
        near_wicket=near,
        far_wicket=far,
        assignment_hypotheses=[
            AssignmentHypothesis(
                hypothesis_id="A",
                near_semantic_end="bowler",
                far_semantic_end="striker",
                confidence=0.5,
                evidence=["wicket_box_calibration"],
            ),
            AssignmentHypothesis(
                hypothesis_id="B",
                near_semantic_end="striker",
                far_semantic_end="bowler",
                confidence=0.5,
                evidence=["wicket_box_calibration"],
            ),
        ],
        diagnostics=WicketObservationDiagnostics(
            detector_model_path="wicket_box_calibration",
            detector_class_labels=["stump"],
            clean_source_video="wicket_box_calibration",
            sampled_frame_ids=[frame_index],
            raw_detections=[],
        ),
        future_registration_readiness="READY_FOR_REGISTRATION_EXPERIMENT",
        message="Synthetic wicket observation built from wicket-box landmarks.",
    )


def _wicket_observation_from_landmarks(
    role: Literal["near", "far"],
    box: WicketBox,
    landmarks: list[StumpLandmark],
    frame_index: int,
    fps: float,
) -> WicketObservation:
    wicket_role = "NEAR" if role == "near" else "FAR"
    role_landmarks = [item for item in landmarks if item.wicket_role == wicket_role]
    pixel_box = PixelBox(x=box.x, y=box.y, width=box.width, height=box.height)
    region = WicketRegionObservation(
        frame_index=frame_index,
        timestamp_seconds=resolve_frame_timestamp(frame_index, fps=fps)[0],
        bbox=pixel_box,
        centre=ObservationPixelPoint(x=box.x + box.width / 2, y=box.y + box.height / 2),
        width=box.width,
        height=box.height,
        detector_confidence=max((item.confidence for item in role_landmarks), default=0.5),
        detector_model="wicket_box_calibration",
        source="wicket_box_calibration",
        temporal_support=1,
        supporting_frame_ids=[frame_index],
        centre_variation_px=0.0,
        size_variation_ratio=0.0,
        confidence_variation=0.0,
        perspective_role=(
            "NEAR_WICKET_CANDIDATE" if role == "near" else "FAR_WICKET_CANDIDATE"
        ),
        stability="STABLE",
        quality="HIGH",
        uncertainty_px=2.0,
    )
    roi = RoiMetadata(
        source_frame_width=box.source_image_width,
        source_frame_height=box.source_image_height,
        x=max(0, int(box.x)),
        y=max(0, int(box.y)),
        width=max(1, int(box.width)),
        height=max(1, int(box.height)),
        padding_x=0,
        padding_y=0,
    )
    coarse: list[WicketLandmarkObservation] = []
    detailed: list[WicketLandmarkObservation] = []
    for item in role_landmarks:
        semantic_base = f"{item.stump_identity.lower()}_stump_base"
        semantic_top = f"{item.stump_identity.lower()}_stump_top"
        detailed.extend(
            [
                _anchor_landmark(semantic_base, item.base, item.confidence, item.provenance),
                _anchor_landmark(semantic_top, item.top, item.confidence, item.provenance),
            ]
        )
        if item.stump_identity == "LEFT":
            coarse.append(
                _anchor_landmark(
                    "wicket_outer_left_base",
                    item.base,
                    item.confidence,
                    item.provenance,
                )
            )
            coarse.append(
                _anchor_landmark(
                    "wicket_outer_left_top",
                    item.top,
                    item.confidence,
                    item.provenance,
                )
            )
        elif item.stump_identity == "RIGHT":
            coarse.append(
                _anchor_landmark(
                    "wicket_outer_right_base",
                    item.base,
                    item.confidence,
                    item.provenance,
                )
            )
            coarse.append(
                _anchor_landmark(
                    "wicket_outer_right_top",
                    item.top,
                    item.confidence,
                    item.provenance,
                )
            )
        else:
            coarse.append(
                _anchor_landmark(
                    "wicket_base_center",
                    item.base,
                    item.confidence,
                    item.provenance,
                )
            )
            coarse.append(
                _anchor_landmark(
                    "wicket_top_center",
                    item.top,
                    item.confidence,
                    item.provenance,
                )
            )
    return WicketObservation(
        region=region,
        roi=roi,
        coarse_landmarks=coarse,
        detailed_landmarks=detailed,
        detailed_landmarks_status="AVAILABLE",
        quality_score=max((item.confidence for item in role_landmarks), default=0.5),
        quality_factors={"frame_edge_clipping": 1.0},
    )


def _landmark_endpoint_count(landmarks: list[StumpLandmark], role: WicketBoxRole) -> int:
    return sum(
        2
        for item in landmarks
        if item.wicket_role == role
    )


def _landmark_validation_reasons(landmarks: list[StumpLandmark]) -> list[str]:
    reasons: list[str] = []
    for role in ("NEAR", "FAR"):
        role_landmarks = [item for item in landmarks if item.wicket_role == role]
        endpoint_count = _landmark_endpoint_count(landmarks, role)
        if endpoint_count < EXPECTED_STUMPS_PER_WICKET * 2:
            reasons.append(
                f"Only {endpoint_count} of six {role} endpoints were detected"
            )
        centre = next(
            (item for item in role_landmarks if item.stump_identity == "MIDDLE"),
            None,
        )
        if centre is not None and centre.confidence < LOW_LANDMARK_CONFIDENCE:
            reasons.append(f"{role} centre stump base has low confidence")
        ordered = sorted(
            role_landmarks,
            key=lambda item: item.base.x,
        )
        if len(ordered) >= 3:
            left, middle, right = ordered[0], ordered[1], ordered[2]
            if not (left.base.x <= middle.base.x <= right.base.x):
                reasons.append("Landmark correspondence is inconsistent")
            if any(item.top.y > item.base.y + 1.0 for item in role_landmarks):
                reasons.append("Landmark correspondence is inconsistent")
    return sorted(set(reasons))


def _perturbation_stability_score(
    uncertainty: PoseUncertainty | None,
) -> float | None:
    if uncertainty is None:
        return None
    if uncertainty.perturbation_count == 0:
        return 0.0
    spreads = (
        uncertainty.camera_position_spread_m,
        uncertainty.rotation_spread_degrees,
        uncertainty.maximum_overlay_movement_px,
    )
    if any(value is not None and not math.isfinite(value) for value in spreads):
        return 0.0
    if uncertainty.stable_for_future_metric_use:
        position = uncertainty.camera_position_spread_m or 0.0
        rotation = uncertainty.rotation_spread_degrees or 0.0
        overlay = uncertainty.maximum_overlay_movement_px or 0.0
        return max(
            0.55,
            min(
                1.0,
                0.4 * max(0.0, 1.0 - position / 1.5)
                + 0.3 * max(0.0, 1.0 - rotation / 3.0)
                + 0.3 * max(0.0, 1.0 - overlay / 12.0),
            ),
        )
    position = uncertainty.camera_position_spread_m or 999.0
    rotation = uncertainty.rotation_spread_degrees or 999.0
    overlay = uncertainty.maximum_overlay_movement_px or 999.0
    return max(
        0.0,
        min(
            0.54,
            0.4 * max(0.0, 1.0 - position / 1.5)
            + 0.3 * max(0.0, 1.0 - rotation / 3.0)
            + 0.3 * max(0.0, 1.0 - overlay / 12.0),
        ),
    )


def _stability_failure_reason(
    candidate: CameraPoseCandidate,
) -> str | None:
    uncertainty = candidate.uncertainty
    if uncertainty is None:
        return (
            "Automatic stump landmarks are valid, but camera-stability "
            "validation encountered an internal error."
        )
    if uncertainty.perturbation_count == 0:
        return "PnP perturbation trials failed"
    spreads = (
        uncertainty.camera_position_spread_m,
        uncertainty.rotation_spread_degrees,
        uncertainty.maximum_overlay_movement_px,
    )
    if any(value is not None and not math.isfinite(value) for value in spreads):
        return "Stability calculation produced a non-finite value"
    if not uncertainty.stable_for_future_metric_use:
        return "Camera pose changes excessively under a one-pixel perturbation"
    return None


def _stability_diagnostics(
    candidate: CameraPoseCandidate,
    correspondences: Sequence[RegistrationCorrespondence],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    temporal = candidate.temporal_validation
    uncertainty = candidate.uncertainty
    zero_reason = None
    if temporal is not None and temporal.stability_score == 0.0:
        if not temporal.evaluated_frame_ids:
            zero_reason = "no_supporting_frame_raw_detections_for_temporal_iou"
        elif temporal.warnings:
            zero_reason = temporal.warnings[0]
    return {
        "candidate_id": candidate.candidate_id,
        "baseline": {
            "rotation_vector": candidate.rotation_vector,
            "translation_vector": candidate.translation_vector,
            "focal_length_px": candidate.intrinsics.focal_length_x_px,
            "reprojection_rmse_px": candidate.reprojection_rmse_px,
        },
        "perturbation": (
            uncertainty.model_dump(mode="json") if uncertainty is not None else None
        ),
        "perturbation_stability_score": _perturbation_stability_score(uncertainty),
        "temporal_stability_score": (
            temporal.stability_score if temporal is not None else None
        ),
        "temporal_zero_reason": zero_reason,
        "stability_failure_reason": _stability_failure_reason(candidate),
        "correspondence_count": len(
            [item for item in correspondences if item.status == "USED"]
        ),
    }


def _write_landmark_debug_report(
    analysis_id: str,
    *,
    landmarks: list[StumpLandmark],
    correspondences: Sequence[RegistrationCorrespondence],
    candidate: CameraPoseCandidate | None,
    frame: np.ndarray,
    width: int,
    height: int,
) -> str:
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / DEBUG_REPORT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    endpoints = [
        {
            "wicket_role": item.wicket_role,
            "stump_identity": item.stump_identity,
            "endpoint": endpoint,
            "x": getattr(point, "x"),
            "y": getattr(point, "y"),
            "confidence": item.confidence,
            "provenance": item.provenance,
        }
        for item in landmarks
        for endpoint, point in (("base", item.base), ("top", item.top))
    ]
    payload = {
        "endpoints": endpoints,
        "correspondences": [
            item.model_dump(mode="json") for item in correspondences
        ],
        "stability": (
            _stability_diagnostics(
                candidate,
                correspondences,
                width=width,
                height=height,
            )
            if candidate is not None
            else None
        ),
        "source_image_width": width,
        "source_image_height": height,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return DEBUG_REPORT_FILENAME


def _rmse_gate_px(candidate: CameraPoseCandidate, base_px: float) -> float:
    """Scale a pixel gate to the frame it is judging.

    Frame width is taken from the principal point, which is seeded at the image
    centre, so no signature change is needed. The flat constant stays as a floor
    so nothing that passed at 720x1280 starts failing.
    """
    estimated_width = float(candidate.intrinsics.principal_point_x_px) * 2.0
    if estimated_width <= 0:
        return base_px
    scaled = estimated_width * RMSE_GATE_FRACTION_OF_WIDTH
    return max(base_px, scaled)


def _mandatory_physical_validation_passed(
    candidate: CameraPoseCandidate,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    failed = {
        check.check_id
        for check in candidate.plausibility_checks
        if check.check_id in MANDATORY_PHYSICAL_CHECKS and not check.passed
    }
    for check in candidate.plausibility_checks:
        if check.check_id in failed and check.reason:
            reasons.append(check.reason)
    if candidate.reprojection_rmse_px is None:
        reasons.append("Registration reprojection error is unavailable.")
    else:
        gate = _rmse_gate_px(candidate, MAX_WICKET_RMSE_PX)
        if candidate.reprojection_rmse_px > gate:
            reasons.append(
                "Per-wicket reprojection error too high "
                f"({candidate.reprojection_rmse_px:.2f}px against a {gate:.1f}px limit "
                "for this frame size)."
            )
    stability_reason = _stability_failure_reason(candidate)
    if stability_reason is not None:
        reasons.append(stability_reason)
    return len(reasons) == 0, sorted(set(reasons))


def _wicket_reprojection_errors(
    candidate: CameraPoseCandidate,
) -> tuple[float | None, float | None]:
    near_errors: list[float] = []
    far_errors: list[float] = []
    for residual in candidate.reprojection_residuals:
        if not residual.inlier:
            continue
        correspondence_id = residual.correspondence_id.lower()
        if correspondence_id.startswith("near:"):
            near_errors.append(residual.residual_px)
        elif correspondence_id.startswith("far:"):
            far_errors.append(residual.residual_px)
    near = (
        float(math.sqrt(sum(value * value for value in near_errors) / len(near_errors)))
        if near_errors
        else None
    )
    far = (
        float(math.sqrt(sum(value * value for value in far_errors) / len(far_errors)))
        if far_errors
        else None
    )
    return near, far


def _candidate_summary(
    candidate: CameraPoseCandidate,
    *,
    physically_valid: bool,
    rejection_reasons: list[str],
) -> WicketBoxCalibrationCandidateSummary:
    near_error, far_error = _wicket_reprojection_errors(candidate)
    camera_height = None
    for check in candidate.plausibility_checks:
        if check.check_id == "camera_above_pitch" and check.value is not None:
            camera_height = float(check.value)
            break
    return WicketBoxCalibrationCandidateSummary(
        candidate_id=candidate.candidate_id,
        assignment_hypothesis=candidate.assignment_hypothesis,
        near_semantic_end=candidate.near_semantic_end,
        far_semantic_end=candidate.far_semantic_end,
        reprojection_rmse_px=candidate.reprojection_rmse_px,
        near_wicket_error_px=near_error,
        far_wicket_error_px=far_error,
        camera_height_m=camera_height,
        focal_length_px=candidate.intrinsics.focal_length_x_px,
        stability_score=_perturbation_stability_score(candidate.uncertainty),
        physically_valid=physically_valid,
        rejection_reasons=rejection_reasons,
    )


def _registration_summary_from_candidates(
    candidates: list[CameraPoseCandidate],
    *,
    selected: CameraPoseCandidate | None,
    alternative: CameraPoseCandidate | None,
    orientation_ambiguous: bool,
) -> WicketBoxRegistrationSummary:
    summaries: dict[str, WicketBoxCalibrationCandidateSummary] = {}
    rejected: list[WicketBoxCalibrationCandidateSummary] = []
    best_by_assignment: dict[str, CameraPoseCandidate] = {}
    for candidate in candidates:
        if not candidate.eligible_for_selection:
            continue
        current = best_by_assignment.get(candidate.assignment_hypothesis)
        if current is None or candidate.score > current.score:
            best_by_assignment[candidate.assignment_hypothesis] = candidate

    for candidate in best_by_assignment.values():
        physically_valid, reasons = _mandatory_physical_validation_passed(candidate)
        summary = _candidate_summary(
            candidate,
            physically_valid=physically_valid,
            rejection_reasons=reasons,
        )
        summaries[candidate.assignment_hypothesis] = summary
        if not physically_valid:
            rejected.append(summary)

    recommended_summary = (
        _candidate_summary(
            selected,
            physically_valid=True,
            rejection_reasons=[],
        )
        if selected is not None
        else None
    )
    alternative_summary = (
        _candidate_summary(
            alternative,
            physically_valid=True,
            rejection_reasons=[],
        )
        if alternative is not None
        else None
    )
    if recommended_summary is None:
        user_message = (
            "Calibration could not be accepted. Correct the stump points or adjust "
            "the wicket boxes and try again."
        )
    elif orientation_ambiguous and alternative_summary is not None:
        user_message = (
            "CricVision tested multiple possible camera positions and selected the "
            "solution that best matches the wicket positions and physical pitch "
            "geometry. An alternative calibration is available if needed."
        )
    else:
        user_message = (
            "CricVision tested multiple possible camera positions and selected the "
            "solution that best matches the wicket positions and physical pitch "
            "geometry."
        )
    return WicketBoxRegistrationSummary(
        recommended=recommended_summary,
        alternative=alternative_summary,
        rejected=rejected,
        auto_selected=recommended_summary is not None and not orientation_ambiguous,
        orientation_ambiguous=orientation_ambiguous,
        user_message=user_message,
    )


def _register_from_observation(
    analysis_id: str,
    observation: WicketObservationResult,
    frame: np.ndarray,
    *,
    assignment_hypothesis: AssignmentChoice | None,
    device_id: str | None = None,
    dimensions: CricketPitchDimensions | None = None,
) -> tuple[RealPitchRegistrationResult, list[str], bool]:
    assert observation.setup_frame is not None
    intrinsics_candidates = _intrinsics_candidates_for_frame(
        device_id,
        frame.shape[1],
        frame.shape[0],
    )
    candidates: list[CameraPoseCandidate] = []
    correspondences_by_candidate: dict[str, list[RegistrationCorrespondence]] = {}
    assignments = (
        [assignment_hypothesis]
        if assignment_hypothesis is not None
        else ["A", "B"]
    )
    for assignment in assignments:
        for lateral_mapping in (
            "image_left_to_world_left",
            "image_left_to_world_right",
        ):
            correspondences = build_registration_correspondences(
                observation,
                assignment_hypothesis=assignment,
                lateral_mapping=lateral_mapping,
                dimensions=dimensions,
            )
            for intrinsics in intrinsics_candidates:
                candidate_id = (
                    f"{assignment}:{lateral_mapping}:{intrinsics.candidate_id}"
                )
                correspondences_by_candidate[candidate_id] = correspondences
                candidate = solve_pose_candidate(
                    candidate_id,
                    assignment,
                    lateral_mapping,
                    observation.setup_frame.frame_index,
                    intrinsics,
                    correspondences,
                    frame,
                    observation,
                    minimum_pointlike_anchors=MIN_ASSISTED_POINTLIKE_ANCHORS,
                    dimensions=dimensions,
                )
                candidates.append(candidate)

    enriched: list[CameraPoseCandidate] = []
    for candidate in candidates:
        if not candidate.solver_success:
            enriched.append(candidate)
            continue
        correspondences = correspondences_by_candidate[candidate.candidate_id]
        uncertainty = _perturbation_uncertainty(
            candidate,
            correspondences,
            frame.shape[1],
            frame.shape[0],
            minimum_pointlike_anchors=MIN_ASSISTED_POINTLIKE_ANCHORS,
            dimensions=dimensions,
        )
        enriched.append(candidate.model_copy(update={"uncertainty": uncertainty}))
    candidates = enriched

    ranked = sorted(
        [item for item in candidates if item.eligible_for_selection],
        key=lambda item: (-item.score, item.candidate_id),
    )
    physically_valid_ranked = [
        item
        for item in ranked
        if _mandatory_physical_validation_passed(item)[0]
    ]
    selected = physically_valid_ranked[0] if physically_valid_ranked else None
    alternative = None
    competing = physically_valid_ranked[1] if len(physically_valid_ranked) > 1 else None
    ambiguity = 0.0
    orientation_required = False
    orientation_ambiguous = False
    if selected and competing:
        ambiguity = abs(selected.score - competing.score)
        if selected.assignment_hypothesis != competing.assignment_hypothesis:
            if ambiguity <= ORIENTATION_AMBIGUITY_GAP:
                orientation_ambiguous = True
                alternative = competing
                if assignment_hypothesis is None:
                    orientation_required = True
            elif competing.score > selected.score * 0.92:
                alternative = competing
    elif selected and assignment_hypothesis is None:
        for item in ranked:
            if (
                item.candidate_id != selected.candidate_id
                and item.assignment_hypothesis != selected.assignment_hypothesis
                and _mandatory_physical_validation_passed(item)[0]
            ):
                alternative = item
                break

    if assignment_hypothesis is not None:
        forced = next(
            (
                item
                for item in physically_valid_ranked
                if item.assignment_hypothesis == assignment_hypothesis
            ),
            None,
        )
        if forced is not None:
            if (
                selected is not None
                and forced.candidate_id != selected.candidate_id
            ):
                alternative = selected
            selected = forced
            orientation_required = False
            orientation_ambiguous = False

    status = "REGISTRATION_FAILED"
    if selected is not None:
        status = selected.classification
    result = RealPitchRegistrationResult(
        analysis_id=analysis_id,
        status=status,
        attempted=True,
        setup_frame=observation.setup_frame,
        supporting_frames=observation.supporting_frames,
        wicket_observation_source="wicket_box_calibration",
        correspondences=(
            build_registration_correspondences(
                observation,
                assignment_hypothesis=selected.assignment_hypothesis,
                lateral_mapping=selected.lateral_mapping,
                dimensions=dimensions,
            )
            if selected is not None
            else []
        ),
        candidates=candidates,
        selected_candidate=selected,
        competing_candidate=alternative,
        ambiguity_score=ambiguity,
        diagnostics=RegistrationDiagnostics(),
        message=(
            "Registration complete."
            if selected is not None
            else _describe_registration_failure(candidates)
        ),
    )
    registration_summary = _registration_summary_from_candidates(
        candidates,
        selected=selected,
        alternative=alternative,
        orientation_ambiguous=orientation_ambiguous,
    )
    warnings: list[str] = []
    if orientation_required:
        warnings.append("orientation_ambiguous")
    return result, warnings, orientation_required, registration_summary


def _calibration_from_registration(
    *,
    analysis_id: str,
    request: WicketBoxCalibrationRegisterRequest,
    landmarks: list[StumpLandmark],
    automatic_landmarks: list[StumpLandmark] | None = None,
    validation: WicketBoxPairValidationResult,
    registration: RealPitchRegistrationResult,
    warnings: list[str],
    registration_summary: WicketBoxRegistrationSummary | None = None,
) -> CalibrationResult:
    selected = registration.selected_candidate
    diagnostics: list[ReprojectionDiagnostic] = []
    rmse: float | None = None
    camera_matrix = None
    rotation_matrix = None
    translation_vector = None
    distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
    if selected is not None:
        rmse = selected.reprojection_rmse_px
        if selected.intrinsics.source == "device_calibration":
            # A measured lens knows where its optical centre is and how much it
            # bends straight lines. Re-centring and zeroing that would discard
            # the reason for calibrating in the first place.
            camera_matrix = [
                [selected.intrinsics.focal_length_x_px, 0.0, selected.intrinsics.principal_point_x_px],
                [0.0, selected.intrinsics.focal_length_y_px, selected.intrinsics.principal_point_y_px],
                [0.0, 0.0, 1.0],
            ]
            distortion = list(selected.intrinsics.distortion_coefficients)
        else:
            camera_matrix = _camera_matrix_from_intrinsics(
                selected.intrinsics.focal_length_x_px,
                request.source_image_width,
                request.source_image_height,
            )
        rotation_matrix = selected.rotation_matrix
        translation_vector = selected.translation_vector
        for residual in selected.reprojection_residuals:
            if residual.observed_pixel is None:
                continue
            diagnostics.append(
                ReprojectionDiagnostic(
                    landmark_id=residual.correspondence_id,
                    observed_pixel_x=residual.observed_pixel.x,
                    observed_pixel_y=residual.observed_pixel.y,
                    reprojected_pixel_x=residual.projected_pixel.x,
                    reprojected_pixel_y=residual.projected_pixel.y,
                    error_px=residual.residual_px,
                )
            )
        warnings = [
            *warnings,
            *[
                check.reason
                for check in selected.plausibility_checks
                if not check.passed and check.reason
            ],
        ]
    preserved_automatic = automatic_landmarks or [
        item for item in landmarks if item.provenance == "AUTOMATIC"
    ]
    return CalibrationResult(
        status="REGISTERED" if selected is not None else "FAILED",
        analysis_id=analysis_id,
        calibration_frame_index=request.calibration_frame_index,
        source_image_width=request.source_image_width,
        source_image_height=request.source_image_height,
        device_id=request.device_id,
        pitch_geometry=request.pitch_geometry,
        near_wicket_box=request.near_wicket_box,
        far_wicket_box=request.far_wicket_box,
        stump_landmarks=landmarks,
        automatic_stump_landmarks=preserved_automatic,
        camera_matrix=camera_matrix,
        rotation_matrix=rotation_matrix,
        translation_vector=translation_vector,
        distortion_coefficients=distortion,
        reprojection_rmse_px=rmse,
        reprojection_diagnostics=diagnostics,
        validation_status=validation.validation_status,
        warnings=warnings,
        registration_summary=registration_summary,
        message=registration.message,
    )


def _registration_rejection_reasons(
    candidate: CameraPoseCandidate,
    registration: RealPitchRegistrationResult,
) -> list[str]:
    reasons: list[str] = []
    failed = {
        check.check_id
        for check in candidate.plausibility_checks
        if not check.passed
    }
    if "camera_above_pitch" in failed:
        reasons.append("Camera appears below ground or at implausible height.")
    if "focal_length_plausible" in failed:
        reasons.append("Recovered focal length is implausible for this frame.")
    if "near_far_projected_size_order" in failed:
        reasons.append("FAR wicket projects larger/closer than NEAR wicket.")
    if candidate.reprojection_rmse_px > _rmse_gate_px(candidate, MAX_WICKET_RMSE_PX):
        reasons.append(
            f"Per-wicket reprojection error too high ({candidate.reprojection_rmse_px:.2f}px)."
        )
    stability_reason = _stability_failure_reason(candidate)
    if stability_reason is not None:
        reasons.append(stability_reason)
    return reasons


def _acceptance_rejection_reasons(calibration: CalibrationResult) -> list[str]:
    reasons: list[str] = []
    # Accepting is stricter than registering, and scales with the frame for the
    # same reason: a fixed pixel budget punishes higher-resolution capture.
    width = float(calibration.source_image_width or 0)
    gate = max(MAX_ACCEPTANCE_RMSE_PX, width * RMSE_GATE_FRACTION_OF_WIDTH * 0.8)
    if calibration.reprojection_rmse_px is None:
        reasons.append("Registration reprojection error is unavailable.")
    elif calibration.reprojection_rmse_px > gate:
        reasons.append(
            f"Acceptance reprojection RMSE exceeds limit "
            f"({calibration.reprojection_rmse_px:.2f}px against {gate:.1f}px)."
        )
    if "orientation_hypothesis_required" in calibration.warnings:
        reasons.append("Orientation hypothesis A/B must be resolved before acceptance.")
    if "orientation_ambiguous" in calibration.warnings:
        pass
    for warning in calibration.warnings:
        if warning.startswith("Camera appears below ground"):
            reasons.append(warning)
        if warning.startswith("Recovered focal length"):
            reasons.append(warning)
        if warning.startswith("FAR wicket projects"):
            reasons.append(warning)
        if warning.startswith("Mirrored pitch orientation"):
            reasons.append(warning)
    return sorted(set(reasons))


def _anchor_landmark(
    semantic_id: str,
    point: PixelPoint,
    confidence: float,
    provenance: str,
) -> WicketLandmarkObservation:
    return WicketLandmarkObservation(
        semantic_id=semantic_id,
        geometry_type="POINT",
        pixel_x=point.x,
        pixel_y=point.y,
        confidence=confidence,
        uncertainty_px=max(1.0, (1.0 - confidence) * 6.0),
        extraction_method=(
            "user_corrected_wicket_box"
            if provenance == "USER_CORRECTED"
            else "wicket_box_automatic"
        ),
        supporting_evidence=["wicket_box_calibration"],
        supporting_frames=[],
        registration_role="PRIMARY_ANCHOR",
        quality="HIGH",
        status="AVAILABLE",
    )


def _roi_from_box(
    frame: np.ndarray,
    box: WicketBox,
    *,
    fps: float | None = None,
) -> tuple[np.ndarray, RoiMetadata, WicketRegionObservation]:
    x1 = max(0, int(box.x))
    y1 = max(0, int(box.y))
    x2 = min(frame.shape[1], int(round(box.x + box.width)))
    y2 = min(frame.shape[0], int(round(box.y + box.height)))
    roi_image = frame[y1:y2, x1:x2]
    roi = RoiMetadata(
        source_frame_width=box.source_image_width,
        source_frame_height=box.source_image_height,
        x=x1,
        y=y1,
        width=max(1, x2 - x1),
        height=max(1, y2 - y1),
        padding_x=0,
        padding_y=0,
    )
    region = WicketRegionObservation(
        frame_index=box.calibration_frame_index,
        timestamp_seconds=resolve_frame_timestamp(
            box.calibration_frame_index,
            fps=fps,
        )[0],
        bbox=PixelBox(x=box.x, y=box.y, width=box.width, height=box.height),
        centre=ObservationPixelPoint(x=box.x + box.width / 2, y=box.y + box.height / 2),
        width=box.width,
        height=box.height,
        detector_confidence=0.8,
        detector_model="wicket_box_calibration",
        source="wicket_box_calibration",
        temporal_support=1,
        supporting_frame_ids=[box.calibration_frame_index],
        centre_variation_px=0.0,
        size_variation_ratio=0.0,
        confidence_variation=0.0,
        perspective_role=(
            "NEAR_WICKET_CANDIDATE"
            if box.role == "NEAR"
            else "FAR_WICKET_CANDIDATE"
        ),
        stability="STABLE",
        quality="HIGH",
        uncertainty_px=2.0,
    )
    return roi_image, roi, region


def _box_to_normalized(box: WicketBox) -> dict[str, float]:
    return {
        "x": box.x / box.source_image_width,
        "y": box.y / box.source_image_height,
        "width": box.width / box.source_image_width,
        "height": box.height / box.source_image_height,
    }


def _landmark_point(item: Any | None) -> PixelPoint | None:
    if item is None or item.status != "AVAILABLE":
        return None
    if item.pixel_x is None or item.pixel_y is None:
        return None
    return PixelPoint(x=float(item.pixel_x), y=float(item.pixel_y))


def _yolo_point(
    item: dict[str, Any] | None,
    key: Literal["base", "top"],
) -> PixelPoint | None:
    if item is None:
        return None
    point = item.get(key)
    if not isinstance(point, dict):
        return None
    return PixelPoint(x=float(point["x"]), y=float(point["y"]))


def _landmark_confidence(
    base_item: Any | None,
    top_item: Any | None,
    yolo_item: dict[str, Any] | None,
) -> float:
    values: list[float] = []
    for item in (base_item, top_item):
        if item is not None and item.status == "AVAILABLE":
            values.append(float(item.confidence))
    if yolo_item is not None:
        values.append(0.75)
    if not values:
        return 0.35
    return max(0.0, min(1.0, sum(values) / len(values)))


def _camera_matrix_from_intrinsics(
    focal: float,
    width: int,
    height: int,
) -> list[list[float]]:
    cx = width / 2.0
    cy = height / 2.0
    return [
        [float(focal), 0.0, cx],
        [0.0, float(focal), cy],
        [0.0, 0.0, 1.0],
    ]


def _result_path(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME


def _accepted_path(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / ACCEPTED_FILENAME


def _write_result(analysis_id: str, result: CalibrationResult) -> None:
    path = _result_path(analysis_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


def _write_accepted_snapshot(analysis_id: str, result: CalibrationResult) -> None:
    path = _accepted_path(analysis_id)
    snapshot = {
        "schema_version": "wicket_box_calibration_accepted_v1",
        "analysis_id": analysis_id,
        "accepted_at": result.accepted_at.isoformat() if result.accepted_at else None,
        "calibration_frame_index": result.calibration_frame_index,
        "source_image_width": result.source_image_width,
        "source_image_height": result.source_image_height,
        "camera_matrix": result.camera_matrix,
        "rotation_matrix": result.rotation_matrix,
        "translation_vector": result.translation_vector,
        "distortion_coefficients": result.distortion_coefficients or [],
        "reprojection_rmse_px": result.reprojection_rmse_px,
        "near_wicket_box": (
            result.near_wicket_box.model_dump(mode="json")
            if result.near_wicket_box
            else None
        ),
        "far_wicket_box": (
            result.far_wicket_box.model_dump(mode="json")
            if result.far_wicket_box
            else None
        ),
        "stump_landmarks": [
            item.model_dump(mode="json") for item in result.stump_landmarks
        ],
        # The pitch this pose was solved against. Physics and replay read it
        # back so a short rig is never measured against regulation.
        "pitch_geometry": (
            result.pitch_geometry.model_dump(mode="json")
            if result.pitch_geometry
            else None
        ),
        "frozen": True,
    }
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def _require_analysis_id(analysis_id: str) -> None:
    if not analysis_id or not analysis_id.strip():
        raise VideoAnalysisServiceError(
            status_code=400,
            message="analysis_id is required.",
        )
    if not ANALYSIS_ID_PATTERN.fullmatch(analysis_id.strip()):
        raise VideoAnalysisServiceError(
            "Invalid analysis ID.",
            status_code=404,
        )


def _require_matching_analysis_id(analysis_id: str, body_id: str) -> None:
    if body_id != analysis_id:
        raise VideoAnalysisServiceError(
            status_code=400,
            message="analysis_id in path and body must match.",
        )
