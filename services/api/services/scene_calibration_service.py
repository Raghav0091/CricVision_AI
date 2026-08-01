"""Unified assisted scene-calibration orchestration and acceptance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..schemas.real_pitch_registration import (
    CameraPoseCandidate,
    RealPitchRegistrationResult,
)
from ..schemas.scene_calibration import (
    AcceptedCalibrationSummary,
    AcceptedSceneCalibrationSnapshot,
    CalibrationThresholdResult,
    SceneCalibrationActionRequest,
    SceneCalibrationAnchor,
    SceneCalibrationAnchorInput,
    SceneCalibrationAnchorUpdateRequest,
    SceneCalibrationDetectionSummary,
    SceneCalibrationOrientationRequest,
    SceneCalibrationObservationSummary,
    SceneCalibrationPresetRequest,
    SceneCalibrationPresetResponse,
    SceneCalibrationRefineRequest,
    SceneCalibrationRegistrationSummary,
    SceneCalibrationResult,
    SceneCalibrationStage,
    SceneCalibrationStageEvent,
    SceneCalibrationValidation,
    CameraOrientationPreset,
    ImageLeftMapping,
    OrientationEvidence,
    OrientationResolution,
)
from ..schemas.wicket_observation import PixelPoint, WicketObservationResult
from .real_pitch_registration_service import (
    check_registration_eligibility,
    run_real_pitch_registration,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .wicket_observation_service import (
    load_wicket_observation,
    run_wicket_observation,
)


RESULT_FILENAME = "scene_calibration_v1.json"
REFINED_REGISTRATION_FILENAME = "real_pitch_registration_v1_refined.json"
REFINED_DEBUG_DIRECTORY = "real_pitch_registration_v1_refined"
ACCEPTED_FILENAME = "accepted_scene_calibration_v1.json"
ORIENTATION_PRESETS_FILENAME = "camera_orientation_presets_v1.json"

# All acceptance thresholds are centralized here and reported per check.
METRIC_THRESHOLDS = {
    "rmse_px": 5.0,
    "median_px": 6.0,
    "maximum_px": 12.0,
    "wicket_envelope": 0.30,
    "temporal_stability": 0.50,
    "independent_scene": 0.40,
    "optional_crease_support": 0.30,
    "ambiguity": 0.25,
    "position_spread_m": 0.50,
    "rotation_spread_deg": 2.0,
    "overlay_sensitivity_px": 8.0,
}
GROUND_THRESHOLDS = {
    "rmse_px": 9.0,
    "median_px": 10.0,
    "maximum_px": 18.0,
    "wicket_envelope": 0.20,
    "temporal_stability": 0.25,
    "independent_scene": 0.25,
    "optional_crease_support": 0.20,
    "ambiguity": 0.35,
    "position_spread_m": 1.00,
    "rotation_spread_deg": 4.0,
    "overlay_sensitivity_px": 15.0,
}

WICKET_ANCHOR_IDS = (
    "near_left_base",
    "near_right_base",
    "near_top_center",
    "far_left_base",
    "far_right_base",
    "far_top_center",
)
CREASE_ANCHOR_IDS = (
    "near_popping_crease_left",
    "near_popping_crease_right",
    "far_popping_crease_left",
    "far_popping_crease_right",
)
PITCH_EDGE_ANCHOR_IDS = (
    "pitch_left_edge_reference",
    "pitch_right_edge_reference",
)
SEMANTIC_ORIENTATION_ANCHOR_IDS = (*CREASE_ANCHOR_IDS, *PITCH_EDGE_ANCHOR_IDS)
SYMMETRIC_EVIDENCE_INSUFFICIENT = [
    "two_symmetric_wickets",
    "wicket_centres",
    "wicket_widths",
    "unlabelled_outer_wicket_anchors",
    "centreline",
    "symmetric_pitch_boundaries",
    "generic_popping_crease_lines",
    "generic_bowling_crease_lines",
    "ball_trajectory_without_semantic_side_reference",
    "camera_height",
    "camera_distance",
    "focal_length",
    "near_far_scale_alone",
]
AUTOMATIC_LANDMARK_IDS = {
    "near_left_base": ("near", "wicket_outer_left_base"),
    "near_right_base": ("near", "wicket_outer_right_base"),
    "near_top_center": ("near", "wicket_top_center"),
    "far_left_base": ("far", "wicket_outer_left_base"),
    "far_right_base": ("far", "wicket_outer_right_base"),
    "far_top_center": ("far", "wicket_top_center"),
}

METRIC_3D_METRICS = [
    "earliest_measured_speed",
    "average_pre_bounce_speed",
    "speed_at_bounce",
    "metric_bounce_position",
    "line_and_length",
    "lateral_movement",
    "post_bounce_turn_when_supported",
]
GROUND_PLANE_METRICS = [
    "metric_bounce_position",
    "line_and_length",
    "top_down_ground_replay",
]

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "NOT_STARTED": {"DETECTING_WICKETS"},
    "DETECTING_WICKETS": {"OBSERVING_WICKETS", "FAILED"},
    "OBSERVING_WICKETS": {
        "GENERATING_POSE",
        "INSUFFICIENT_EVIDENCE",
        "FAILED",
    },
    "GENERATING_POSE": {
        "NEEDS_ADJUSTMENT",
        "ORIENTATION_REQUIRED",
        "GROUND_PLANE_READY",
        "METRIC_3D_READY",
        "INSUFFICIENT_EVIDENCE",
        "FAILED",
    },
    "NEEDS_ADJUSTMENT": {
        "GENERATING_POSE",
        "DETECTING_WICKETS",
        "NEEDS_ADJUSTMENT",
        "ORIENTATION_REQUIRED",
        "GROUND_PLANE_READY",
        "METRIC_3D_READY",
    },
    "ORIENTATION_REQUIRED": {
        "GENERATING_POSE",
        "DETECTING_WICKETS",
        "NEEDS_ADJUSTMENT",
        "ORIENTATION_REQUIRED",
        "GROUND_PLANE_READY",
        "METRIC_3D_READY",
    },
    "GROUND_PLANE_READY": {
        "GENERATING_POSE",
        "DETECTING_WICKETS",
        "NEEDS_ADJUSTMENT",
        "GROUND_PLANE_READY",
    },
    "METRIC_3D_READY": {
        "GENERATING_POSE",
        "DETECTING_WICKETS",
        "NEEDS_ADJUSTMENT",
        "METRIC_3D_READY",
    },
    "INSUFFICIENT_EVIDENCE": {"DETECTING_WICKETS"},
    "FAILED": {"DETECTING_WICKETS"},
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _result_path(analysis_id: str) -> Path:
    return VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (
        value.model_dump(mode="json")  # type: ignore[attr-defined]
        if hasattr(value, "model_dump")
        else value
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _persist(result: SceneCalibrationResult) -> SceneCalibrationResult:
    _write_json(_result_path(result.analysis_id), result)
    return result


def _empty_result(analysis_id: str) -> SceneCalibrationResult:
    load_video_analysis(analysis_id)
    return SceneCalibrationResult(
        analysis_id=analysis_id,
        stage="NOT_STARTED",
        updated_at=_now(),
        metrics_locked_reasons=["Scene calibration has not been run."],
        message="Detect wickets to begin scene calibration.",
    )


def load_scene_calibration(analysis_id: str) -> SceneCalibrationResult:
    load_video_analysis(analysis_id)
    path = _result_path(analysis_id)
    if not path.is_file():
        return _empty_result(analysis_id)
    try:
        result = SceneCalibrationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if (
            result.setup_frame is not None
            and result.setup_frame_image_url is None
        ):
            try:
                observation = load_wicket_observation(analysis_id)
                result = result.model_copy(
                    update={
                        "setup_frame_image_url": (
                            observation.diagnostics.setup_frame_image_url
                        ),
                        "raw_wicket_overlay_url": (
                            observation.diagnostics.raw_detection_overlay_url
                        ),
                    }
                )
            except Exception:
                pass
        return result
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored scene calibration is unavailable.",
            status_code=500,
        ) from exc


def transition_scene_calibration(
    result: SceneCalibrationResult,
    stage: SceneCalibrationStage,
    message: str,
) -> SceneCalibrationResult:
    if stage != result.stage and stage not in ALLOWED_TRANSITIONS[result.stage]:
        raise VideoAnalysisServiceError(
            f"Invalid scene-calibration transition: {result.stage} -> {stage}.",
            status_code=409,
        )
    now = _now()
    event = SceneCalibrationStageEvent(stage=stage, at=now, message=message)
    return result.model_copy(
        update={
            "stage": stage,
            "updated_at": now,
            "stage_history": [*result.stage_history, event],
            "message": message,
        }
    )


def _observation_summary(
    analysis_id: str,
    observation: WicketObservationResult,
) -> SceneCalibrationObservationSummary:
    available = sum(
        item.status == "AVAILABLE"
        for wicket in (observation.near_wicket, observation.far_wicket)
        if wicket is not None
        for item in wicket.coarse_landmarks + wicket.detailed_landmarks
    )
    return SceneCalibrationObservationSummary(
        status=observation.status,
        setup_frame_index=(
            observation.setup_frame.frame_index
            if observation.setup_frame is not None
            else None
        ),
        supporting_frame_count=len(observation.supporting_frames),
        near_wicket_available=observation.near_wicket is not None,
        far_wicket_available=observation.far_wicket is not None,
        available_anchor_count=available,
        result_url=(
            f"/static/video-analysis/{analysis_id}/reports/"
            "wicket_observations_v1.json"
        ),
    )


def _detection_summary(
    observation: WicketObservationResult,
    *,
    reused: bool,
) -> SceneCalibrationDetectionSummary:
    diagnostics = observation.diagnostics
    return SceneCalibrationDetectionSummary(
        detector_model=diagnostics.detector_model_path,
        sampled_frame_count=len(diagnostics.sampled_frame_ids),
        raw_detection_count=len(diagnostics.raw_detections),
        rejected_detection_count=len(diagnostics.rejected_detections),
        reused_persisted_result=reused,
    )


def _registration_summary(
    analysis_id: str,
    registration: RealPitchRegistrationResult,
    *,
    refined: bool = False,
) -> SceneCalibrationRegistrationSummary:
    selected = registration.selected_candidate
    independent = selected.independent_validation if selected else None
    temporal = selected.temporal_validation if selected else None
    return SceneCalibrationRegistrationSummary(
        status=registration.status,
        attempted=registration.attempted,
        selected_candidate_id=selected.candidate_id if selected else None,
        assignment_hypothesis=(
            selected.assignment_hypothesis if selected else None
        ),
        focal_length_px=(
            selected.intrinsics.focal_length_x_px if selected else None
        ),
        reprojection_rmse_px=(
            selected.reprojection_rmse_px if selected else None
        ),
        median_reprojection_error_px=(
            selected.median_reprojection_error_px if selected else None
        ),
        maximum_inlier_error_px=(
            selected.maximum_inlier_error_px if selected else None
        ),
        inlier_count=len(selected.inlier_correspondence_ids) if selected else 0,
        outlier_count=(
            len(selected.outlier_correspondence_ids) if selected else 0
        ),
        wicket_envelope_score=(
            independent.projected_wicket_envelope_score
            if independent
            else None
        ),
        temporal_stability_score=(
            temporal.stability_score if temporal else None
        ),
        independent_scene_score=(
            independent.independent_scene_score if independent else None
        ),
        ambiguity_score=registration.ambiguity_score,
        result_url=(
            f"/static/video-analysis/{analysis_id}/reports/"
            f"{REFINED_REGISTRATION_FILENAME if refined else 'real_pitch_registration_v1.json'}"
        ),
    )


def _preset_store_path() -> Path:
    return VIDEO_ANALYSIS_ROOT / "_calibration" / ORIENTATION_PRESETS_FILENAME


def _candidate_mapping(candidate: CameraPoseCandidate | None) -> ImageLeftMapping | None:
    if candidate is None:
        return None
    return (
        "IMAGE_LEFT_IS_PITCH_LEFT"
        if candidate.lateral_mapping == "image_left_to_world_left"
        else "IMAGE_LEFT_IS_PITCH_RIGHT"
    )


def _registration_lateral_mapping(mapping: ImageLeftMapping) -> str:
    return (
        "image_left_to_world_left"
        if mapping == "IMAGE_LEFT_IS_PITCH_LEFT"
        else "image_left_to_world_right"
    )


def _mirror_ambiguous(registration: RealPitchRegistrationResult) -> bool:
    selected = registration.selected_candidate
    competing = registration.competing_candidate
    return bool(
        selected is not None
        and competing is not None
        and selected.lateral_mapping != competing.lateral_mapping
        and registration.ambiguity_score > GROUND_THRESHOLDS["ambiguity"]
    )


def _orientation_stage(
    registration: RealPitchRegistrationResult,
    validation: SceneCalibrationValidation | None = None,
) -> SceneCalibrationStage:
    if _mirror_ambiguous(registration):
        return "ORIENTATION_REQUIRED"
    if validation and validation.eligible_level in {
        "GROUND_PLANE_READY",
        "METRIC_3D_READY",
    }:
        return validation.eligible_level
    return _candidate_stage(registration)


def _orientation_resolution(
    *,
    registration: RealPitchRegistrationResult,
    evidence: list[OrientationEvidence],
    ambiguity_before: float | None = None,
    image_left_mapping: ImageLeftMapping | None = None,
    camera_end: str | None = None,
) -> OrientationResolution:
    selected = registration.selected_candidate
    consistent = [
        item.candidate_id
        for item in registration.candidates
        if item.eligible_for_selection
        and (
            image_left_mapping is None
            or _candidate_mapping(item) == image_left_mapping
        )
    ]
    rejected = [
        item.candidate_id
        for item in registration.candidates
        if image_left_mapping is not None
        and _candidate_mapping(item) != image_left_mapping
    ]
    return OrientationResolution(
        required=_mirror_ambiguous(registration) or bool(image_left_mapping),
        resolved=bool(
            image_left_mapping
            and selected is not None
            and _candidate_mapping(selected) == image_left_mapping
            and registration.ambiguity_score <= GROUND_THRESHOLDS["ambiguity"]
        ),
        image_left_mapping=image_left_mapping,
        camera_end=camera_end if camera_end in {"bowler", "striker", "unknown"} else None,
        ambiguity_before=(
            registration.ambiguity_score if ambiguity_before is None else ambiguity_before
        ),
        ambiguity_after=registration.ambiguity_score,
        selected_candidate_id=selected.candidate_id if selected else None,
        rejected_candidate_ids=rejected,
        consistent_candidate_ids=consistent,
        evidence_applied=evidence,
        symmetric_evidence_insufficient=SYMMETRIC_EVIDENCE_INSUFFICIENT,
        remaining_failures=registration.failure_reasons,
    )


def _orientation_evidence_id(*parts: object) -> str:
    digest = hashlib.sha1(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()[:12]
    return f"orientation_{digest}"


def _user_orientation_evidence(
    *,
    mapping: ImageLeftMapping,
    camera_end: str,
    registration: RealPitchRegistrationResult,
) -> OrientationEvidence:
    supported = [
        item.candidate_id
        for item in registration.candidates
        if _candidate_mapping(item) == mapping
    ]
    rejected = [
        item.candidate_id
        for item in registration.candidates
        if _candidate_mapping(item) != mapping
    ]
    return OrientationEvidence(
        evidence_id=_orientation_evidence_id(mapping, camera_end, _now().isoformat()),
        evidence_type="USER_CONFIRMED_LATERAL_ORIENTATION",
        source="user",
        semantic_label=mapping,
        confidence=0.95,
        uncertainty=0.05,
        authoritative=True,
        supports_candidate_ids=supported,
        rejects_candidate_ids=rejected,
        explanation=(
            "User confirmed the native-video image-left to pitch-left/right "
            "mapping. This resolves semantic lateral orientation only."
        ),
        created_at=_now(),
        user_confirmed=True,
    )


def _load_presets() -> list[CameraOrientationPreset]:
    path = _preset_store_path()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            CameraOrientationPreset.model_validate(item)
            for item in payload.get("presets", [])
        ]
    except (OSError, ValueError, TypeError):
        return []


def _write_presets(presets: list[CameraOrientationPreset]) -> None:
    _write_json(
        _preset_store_path(),
        {"presets": [item.model_dump(mode="json") for item in presets]},
    )


def _preset_compatibility(
    preset: CameraOrientationPreset,
    *,
    analysis_id: str,
    width: int,
    height: int,
    camera_end: str,
) -> CameraOrientationPreset:
    reasons: list[str] = []
    aspect = width / max(height, 1)
    preset_aspect = preset.native_width / max(preset.native_height, 1)
    if abs(aspect - preset_aspect) > 0.02:
        reasons.append("aspect_ratio_mismatch")
    if abs(width - preset.native_width) / max(preset.native_width, 1) > 0.05:
        reasons.append("resolution_width_mismatch")
    if abs(height - preset.native_height) / max(preset.native_height, 1) > 0.05:
        reasons.append("resolution_height_mismatch")
    if preset.camera_end != "unknown" and camera_end != "unknown" and preset.camera_end != camera_end:
        reasons.append("camera_end_mismatch")
    if preset.source_analysis_id == analysis_id:
        reasons.append("same_analysis_source")
    return preset.model_copy(
        update={
            "compatible": not reasons,
            "compatibility_reasons": (
                ["compatible_but_requires_user_confirmation"]
                if not reasons
                else reasons
            ),
        }
    )


def load_orientation_presets_for_analysis(
    analysis_id: str,
) -> SceneCalibrationPresetResponse:
    result = load_scene_calibration(analysis_id)
    if result.setup_frame is None:
        return SceneCalibrationPresetResponse(analysis_id=analysis_id)
    width = result.setup_frame.image_width
    height = result.setup_frame.image_height
    checked = [
        _preset_compatibility(
            preset,
            analysis_id=analysis_id,
            width=width,
            height=height,
            camera_end=result.camera_end or "unknown",
        )
        for preset in _load_presets()
    ]
    return SceneCalibrationPresetResponse(
        analysis_id=analysis_id,
        compatible_presets=[item for item in checked if item.compatible],
        rejected_presets=[item for item in checked if not item.compatible],
    )


def _mapping_from_pair(
    left: SceneCalibrationAnchor,
    right: SceneCalibrationAnchor,
) -> ImageLeftMapping | None:
    if not (
        left.valid
        and right.valid
        and left.video_point is not None
        and right.video_point is not None
    ):
        return None
    return (
        "IMAGE_LEFT_IS_PITCH_LEFT"
        if left.video_point.x < right.video_point.x
        else "IMAGE_LEFT_IS_PITCH_RIGHT"
    )


def _infer_mapping_from_semantic_anchors(
    anchors: list[SceneCalibrationAnchor],
) -> tuple[ImageLeftMapping | None, list[tuple[str, PixelPoint]]]:
    by_id = {item.semantic_id: item for item in anchors}
    inferred: list[ImageLeftMapping] = []
    evidence_points: list[tuple[str, PixelPoint]] = []
    pairs = [
        ("near_popping_crease", "near_popping_crease_left", "near_popping_crease_right"),
        ("far_popping_crease", "far_popping_crease_left", "far_popping_crease_right"),
        ("pitch_edge", "pitch_left_edge_reference", "pitch_right_edge_reference"),
    ]
    for label, left_id, right_id in pairs:
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        if left is None or right is None:
            continue
        mapping = _mapping_from_pair(left, right)
        if mapping is None:
            continue
        inferred.append(mapping)
        if left.video_point is not None:
            evidence_points.append((f"{label}:left", left.video_point))
        if right.video_point is not None:
            evidence_points.append((f"{label}:right", right.video_point))
    if len(set(inferred)) > 1:
        raise VideoAnalysisServiceError(
            "Contradictory semantic orientation anchors.",
            status_code=422,
        )
    return (inferred[0], evidence_points) if inferred else (None, [])


def _semantic_anchor_evidence(
    *,
    mapping: ImageLeftMapping,
    points: list[tuple[str, PixelPoint]],
    frame_index: int,
    registration: RealPitchRegistrationResult,
) -> list[OrientationEvidence]:
    supported = [
        item.candidate_id
        for item in registration.candidates
        if _candidate_mapping(item) == mapping
    ]
    rejected = [
        item.candidate_id
        for item in registration.candidates
        if _candidate_mapping(item) != mapping
    ]
    evidence: list[OrientationEvidence] = []
    for label, point in points:
        evidence.append(
            OrientationEvidence(
                evidence_id=_orientation_evidence_id(label, point.x, point.y),
                evidence_type=(
                    "SEMANTIC_PITCH_EDGE_POINT"
                    if label.startswith("pitch_edge")
                    else "SEMANTIC_CREASE_ENDPOINT"
                ),
                source="manual_anchor",
                frame_index=frame_index,
                native_pixel_coordinate=point,
                semantic_label=label,
                confidence=0.82,
                uncertainty=4.0,
                authoritative=False,
                supports_candidate_ids=supported,
                rejects_candidate_ids=rejected,
                explanation=(
                    "A labelled left/right scene anchor pair supports the "
                    f"{mapping} orientation in native video coordinates."
                ),
                created_at=_now(),
                user_confirmed=True,
            )
        )
    return evidence


def _automatic_anchor(
    observation: WicketObservationResult,
    semantic_id: str,
) -> SceneCalibrationAnchor:
    role, landmark_id = AUTOMATIC_LANDMARK_IDS[semantic_id]
    wicket = (
        observation.near_wicket if role == "near" else observation.far_wicket
    )
    landmark = next(
        (
            item
            for item in (
                wicket.coarse_landmarks + wicket.detailed_landmarks
                if wicket is not None
                else []
            )
            if item.semantic_id == landmark_id and item.status == "AVAILABLE"
        ),
        None,
    )
    point = (
        PixelPoint(x=landmark.pixel_x, y=landmark.pixel_y)
        if landmark is not None
        and landmark.pixel_x is not None
        and landmark.pixel_y is not None
        else None
    )
    return SceneCalibrationAnchor(
        semantic_id=semantic_id,
        kind="wicket",
        wicket_role=role,
        video_point=point,
        source="automatic",
        original_automatic_point=point,
        confidence=landmark.confidence if landmark else 0.0,
        uncertainty_px=landmark.uncertainty_px if landmark else 0.0,
        adjustment_distance_px=0.0,
        frame_index=(
            observation.setup_frame.frame_index
            if observation.setup_frame is not None
            else 0
        ),
        valid=point is not None,
        used_for_refinement=point is not None,
        used_for_validation=True,
        validation_messages=(
            [] if point is not None else ["Automatic anchor unavailable."]
        ),
    )


def initialise_scene_anchors(
    observation: WicketObservationResult,
) -> list[SceneCalibrationAnchor]:
    return [
        _automatic_anchor(observation, semantic_id)
        for semantic_id in WICKET_ANCHOR_IDS
    ]


def _candidate_stage(registration: RealPitchRegistrationResult) -> SceneCalibrationStage:
    if registration.status in {"NOT_ATTEMPTED"}:
        return "INSUFFICIENT_EVIDENCE"
    if registration.status == "REGISTRATION_FAILED":
        return "FAILED"
    return "NEEDS_ADJUSTMENT"


def run_scene_calibration(analysis_id: str) -> SceneCalibrationResult:
    previous = load_scene_calibration(analysis_id)
    result = _empty_result(analysis_id).model_copy(
        update={
            "started_at": _now(),
            "accepted_calibration": None,
            "stage_history": previous.stage_history,
            "warnings": [
                "Legacy visual calibration remains available as a fallback.",
                "Metric analytics remain locked until strict acceptance.",
            ],
        }
    )
    result = transition_scene_calibration(
        result, "DETECTING_WICKETS", "Detecting wickets."
    )
    _persist(result)
    try:
        observation = run_wicket_observation(analysis_id)
        result = result.model_copy(
            update={
                "raw_stump_detection_summary": _detection_summary(
                    observation, reused=False
                )
            }
        )
        result = transition_scene_calibration(
            result,
            "OBSERVING_WICKETS",
            "Stabilising wicket observations and extracting anchors.",
        )
        _persist(result)
        result = result.model_copy(
            update={
                "setup_frame": observation.setup_frame,
                "supporting_frames": observation.supporting_frames,
                "setup_frame_image_url": (
                    observation.diagnostics.setup_frame_image_url
                ),
                "raw_wicket_overlay_url": (
                    observation.diagnostics.raw_detection_overlay_url
                ),
                "wicket_observation_summary": _observation_summary(
                    analysis_id, observation
                ),
                "current_anchor_set": initialise_scene_anchors(observation),
                "anchor_version": 1,
                "developer_diagnostics_available": True,
            }
        )
        eligibility = check_registration_eligibility(observation)
        if not eligibility.eligible:
            result = transition_scene_calibration(
                result,
                "INSUFFICIENT_EVIDENCE",
                "Wicket evidence is insufficient for camera registration.",
            )
            return _persist(
                result.model_copy(
                    update={
                        "completed_at": _now(),
                        "calibration_level": "UNAVAILABLE",
                        "failure_reasons": eligibility.reasons,
                        "metrics_locked_reasons": [
                            "Camera registration was not eligible.",
                            *eligibility.reasons,
                        ],
                    }
                )
            )
        result = transition_scene_calibration(
            result, "GENERATING_POSE", "Aligning the virtual pitch."
        )
        _persist(result)
        registration = run_real_pitch_registration(analysis_id)
        orientation_resolution = _orientation_resolution(
            registration=registration,
            evidence=[],
        )
        stage = _orientation_stage(registration)
        result = transition_scene_calibration(
            result,
            stage,
            (
                "Pitch alignment proposed. Fine adjustment is available."
                if stage == "NEEDS_ADJUSTMENT"
                else "Camera registration could not produce a usable candidate."
            ),
        )
        return _persist(
            result.model_copy(
                update={
                    "completed_at": _now(),
                    "automatic_registration_summary": _registration_summary(
                        analysis_id, registration
                    ),
                    "selected_candidate": registration.selected_candidate,
                    "competing_candidate": registration.competing_candidate,
                    "projected_pitch_geometry": (
                        registration.projected_pitch_geometry
                    ),
                    "competing_projected_pitch_geometry": (
                        registration.competing_projected_pitch_geometry
                    ),
                    "orientation_required": stage == "ORIENTATION_REQUIRED",
                    "orientation_resolution": orientation_resolution,
                    "calibration_level": (
                        "VISUAL_ONLY"
                        if registration.selected_candidate is not None
                        else "UNAVAILABLE"
                    ),
                    "failure_reasons": registration.failure_reasons,
                    "metrics_locked_reasons": [
                        "Calibration has not passed backend acceptance.",
                        *registration.failure_reasons,
                    ],
                }
            )
        )
    except Exception as exc:
        result = transition_scene_calibration(
            result, "FAILED", "Scene calibration failed safely."
        )
        return _persist(
            result.model_copy(
                update={
                    "completed_at": _now(),
                    "failure_reasons": [f"{type(exc).__name__}: {exc}"],
                    "metrics_locked_reasons": [
                        "Scene calibration failed; image-space analysis remains available."
                    ],
                }
            )
        )


def _distance(first: PixelPoint | None, second: PixelPoint | None) -> float:
    if first is None or second is None:
        return 0.0
    return math.hypot(first.x - second.x, first.y - second.y)


def _anchor_from_input(
    current: SceneCalibrationAnchor | None,
    value: SceneCalibrationAnchorInput,
    *,
    frame_index: int,
) -> SceneCalibrationAnchor:
    original = current.original_automatic_point if current else None
    point = value.video_point
    kind = (
        "crease"
        if value.semantic_id in CREASE_ANCHOR_IDS
        else "pitch_edge"
        if value.semantic_id in PITCH_EDGE_ANCHOR_IDS
        else "wicket"
    )
    return SceneCalibrationAnchor(
        semantic_id=value.semantic_id,
        kind=kind,
        wicket_role=(
            "near"
            if value.semantic_id.startswith("near_")
            else "far"
            if value.semantic_id.startswith("far_")
            else None
        ),
        video_point=point,
        source=value.source,
        original_automatic_point=original,
        confidence=(
            0.90 if value.source == "manually_adjusted" else 0.72
        ),
        uncertainty_px=2.0 if value.source == "manually_adjusted" else 4.0,
        adjustment_distance_px=_distance(original, point),
        frame_index=frame_index,
        valid=point is not None,
        used_for_refinement=value.used_for_refinement,
        used_for_validation=value.used_for_validation,
    )


def _point_map(
    anchors: Iterable[SceneCalibrationAnchor],
) -> dict[str, PixelPoint]:
    return {
        item.semantic_id: item.video_point
        for item in anchors
        if item.video_point is not None and item.valid
    }


def validate_scene_anchors(
    anchors: list[SceneCalibrationAnchor],
    *,
    image_width: int,
    image_height: int,
) -> list[SceneCalibrationAnchor]:
    by_id = {item.semantic_id: item for item in anchors}
    messages: dict[str, list[str]] = {
        item.semantic_id: [] for item in anchors
    }
    for item in anchors:
        point = item.video_point
        if point is None:
            messages[item.semantic_id].append("Anchor is unavailable.")
        elif not (0 <= point.x < image_width and 0 <= point.y < image_height):
            messages[item.semantic_id].append(
                "Anchor must remain inside the video frame."
            )
        if item.adjustment_distance_px > math.hypot(
            image_width, image_height
        ) * 0.25:
            messages[item.semantic_id].append(
                "Anchor movement is too large for one refinement."
            )
    for role in ("near", "far"):
        left = by_id.get(f"{role}_left_base")
        right = by_id.get(f"{role}_right_base")
        top = by_id.get(f"{role}_top_center")
        if left and right and left.video_point and right.video_point:
            width = right.video_point.x - left.video_point.x
            if width <= 2:
                text = "Left and right wicket bases are reversed or too close."
                messages[left.semantic_id].append(text)
                messages[right.semantic_id].append(text)
            if width > image_width * 0.45:
                text = "Wicket width is implausibly large."
                messages[left.semantic_id].append(text)
                messages[right.semantic_id].append(text)
        if left and right and top and all(
            item.video_point is not None for item in (left, right, top)
        ):
            base_y = (left.video_point.y + right.video_point.y) / 2
            if top.video_point.y >= base_y - 2:
                messages[top.semantic_id].append(
                    "Wicket top must be above its bases."
                )
    near = [by_id.get(f"near_{name}") for name in (
        "left_base", "right_base", "top_center"
    )]
    far = [by_id.get(f"far_{name}") for name in (
        "left_base", "right_base", "top_center"
    )]
    if all(item is not None and item.video_point is not None for item in near + far):
        near_points = [item.video_point for item in near]
        far_points = [item.video_point for item in far]
        near_width = near_points[1].x - near_points[0].x
        far_width = far_points[1].x - far_points[0].x
        if near_width < far_width * 0.75:
            text = "Near wicket is implausibly smaller than the far wicket."
            for item in near + far:
                messages[item.semantic_id].append(text)
        near_centre = PixelPoint(
            x=(near_points[0].x + near_points[1].x) / 2,
            y=(near_points[0].y + near_points[1].y) / 2,
        )
        far_centre = PixelPoint(
            x=(far_points[0].x + far_points[1].x) / 2,
            y=(far_points[0].y + far_points[1].y) / 2,
        )
        if _distance(near_centre, far_centre) < math.hypot(
            image_width, image_height
        ) * 0.03:
            text = "Near and far wickets overlap or imply a degenerate pitch."
            for item in near + far:
                messages[item.semantic_id].append(text)
    for role in ("near", "far"):
        left = by_id.get(f"{role}_popping_crease_left")
        right = by_id.get(f"{role}_popping_crease_right")
        if left and right and left.video_point and right.video_point:
            if _distance(left.video_point, right.video_point) < image_width * 0.02:
                text = "Crease left/right anchors are too close."
                messages[left.semantic_id].append(text)
                messages[right.semantic_id].append(text)
    left_edge = by_id.get("pitch_left_edge_reference")
    right_edge = by_id.get("pitch_right_edge_reference")
    if (
        left_edge
        and right_edge
        and left_edge.video_point
        and right_edge.video_point
        and _distance(left_edge.video_point, right_edge.video_point)
        < image_width * 0.02
    ):
        text = "Pitch-edge references are too close."
        messages[left_edge.semantic_id].append(text)
        messages[right_edge.semantic_id].append(text)
    return [
        item.model_copy(
            update={
                "valid": item.video_point is not None
                and not messages[item.semantic_id],
                "validation_messages": messages[item.semantic_id],
            }
        )
        for item in anchors
    ]


def update_scene_calibration_anchors(
    analysis_id: str,
    request: SceneCalibrationAnchorUpdateRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload calibration before editing.",
            status_code=409,
        )
    if result.setup_frame is None:
        raise VideoAnalysisServiceError(
            "No setup frame is available for anchor editing.",
            status_code=422,
        )
    current = {
        item.semantic_id: item
        for item in [
            *result.current_anchor_set,
            *result.optional_crease_anchors,
        ]
    }
    for value in request.anchors:
        if value.semantic_id not in {
            *WICKET_ANCHOR_IDS,
            *SEMANTIC_ORIENTATION_ANCHOR_IDS,
        }:
            raise VideoAnalysisServiceError(
                f"Unsupported calibration anchor: {value.semantic_id}.",
                status_code=422,
            )
        existing = current.get(value.semantic_id)
        if value.source == "manually_added" and existing is not None and (
            existing.original_automatic_point is not None
        ):
            raise VideoAnalysisServiceError(
                f"{value.semantic_id} already has an automatic origin.",
                status_code=422,
            )
        current[value.semantic_id] = _anchor_from_input(
            existing,
            value,
            frame_index=result.setup_frame.frame_index,
        )
    validated = validate_scene_anchors(
        list(current.values()),
        image_width=result.setup_frame.image_width,
        image_height=result.setup_frame.image_height,
    )
    wickets = [
        item for item in validated if item.semantic_id in WICKET_ANCHOR_IDS
    ]
    creases = [
        item for item in validated if item.semantic_id in SEMANTIC_ORIENTATION_ANCHOR_IDS
    ]
    return _persist(
        result.model_copy(
            update={
                "current_anchor_set": wickets,
                "optional_crease_anchors": creases,
                "anchor_version": result.anchor_version + 1,
                "accepted_calibration": None,
                "metrics_unlocked": [],
                "metrics_locked_reasons": [
                    "Edited anchors require backend recalculation and acceptance."
                ],
                "updated_at": _now(),
                "stage": "NEEDS_ADJUSTMENT",
                "message": "Anchor edits saved. Recalculate alignment.",
            }
        )
    )


def _check(
    check_id: str,
    value: float | int | bool | str | None,
    passed: bool,
    requirement: str,
    failure: str,
) -> CalibrationThresholdResult:
    return CalibrationThresholdResult(
        threshold_id=check_id,
        passed=passed,
        value=value,
        requirement=requirement,
        reason="Passed." if passed else failure,
    )


def evaluate_calibration_candidate(
    candidate: CameraPoseCandidate | None,
    *,
    ambiguity_score: float,
    anchors: list[SceneCalibrationAnchor],
    orientation_resolution: OrientationResolution | None = None,
) -> SceneCalibrationValidation:
    adjusted = sum(item.source == "manually_adjusted" for item in anchors)
    added = sum(item.source == "manually_added" for item in anchors)
    valid_wickets = [
        item
        for item in anchors
        if item.kind == "wicket" and item.valid and item.video_point is not None
    ]
    if candidate is None or not candidate.solver_success:
        return SceneCalibrationValidation(
            eligible_level="UNAVAILABLE",
            accepted_anchor_count=len(valid_wickets),
            manually_adjusted_anchor_count=adjusted,
            manually_added_anchor_count=added,
            all_required_checks_passed=False,
            failure_reasons=["No solved camera candidate is available."],
        )
    independent = candidate.independent_validation
    temporal = candidate.temporal_validation
    uncertainty = candidate.uncertainty
    hard_checks_pass = all(
        item.passed for item in candidate.plausibility_checks
    )
    has_validation_crease = any(
        item.kind == "crease"
        and item.valid
        and item.video_point is not None
        and item.used_for_validation
        for item in anchors
    )
    focal_bound = candidate.intrinsics.focal_bound_reached or bool(
        candidate.refinement.parameters_reaching_bounds
    )
    lateral_resolved = bool(
        orientation_resolution is None
        or not orientation_resolution.required
        or orientation_resolution.resolved
    )

    def checks_for(level: str, thresholds: dict[str, float]):
        classification_ok = (
            candidate.classification == "METRIC_3D_CANDIDATE"
            if level == "METRIC_3D_READY"
            else candidate.classification
            in {"METRIC_3D_CANDIDATE", "GROUND_PLANE_CANDIDATE"}
        )
        values = {
            "rmse_px": candidate.reprojection_rmse_px,
            "median_px": candidate.median_reprojection_error_px,
            "maximum_px": candidate.maximum_inlier_error_px,
            "wicket_envelope": (
                independent.projected_wicket_envelope_score
                if independent
                else None
            ),
            "temporal_stability": (
                temporal.stability_score if temporal else None
            ),
            "independent_scene": (
                independent.independent_scene_score if independent else None
            ),
            "optional_crease_support": (
                independent.crease_edge_support_score
                if independent and has_validation_crease
                else None
            ),
            "ambiguity": ambiguity_score,
            "position_spread_m": (
                uncertainty.camera_position_spread_m if uncertainty else None
            ),
            "rotation_spread_deg": (
                uncertainty.rotation_spread_degrees if uncertainty else None
            ),
            "overlay_sensitivity_px": (
                uncertainty.maximum_overlay_movement_px
                if uncertainty
                else None
            ),
        }
        output = [
            _check(
                "candidate_classification",
                candidate.classification,
                classification_ok,
                f"Candidate supports {level}.",
                "Candidate classification is too weak.",
            ),
            _check(
                "accepted_wicket_anchors",
                len(valid_wickets),
                len(valid_wickets) == 6,
                "All six wicket anchors are valid.",
                "All six wicket anchors are required.",
            ),
            _check(
                "physical_plausibility",
                hard_checks_pass,
                hard_checks_pass,
                "All physical plausibility checks pass.",
                "One or more physical plausibility checks failed.",
            ),
            _check(
                "optimisation_bounds",
                focal_bound,
                not focal_bound,
                "No optimized parameter reaches a bound.",
                "An optimized parameter reached a configured bound.",
            ),
            _check(
                "perturbation_stability",
                (
                    uncertainty.stable_for_future_metric_use
                    if uncertainty
                    else False
                ),
                bool(
                    uncertainty
                    and uncertainty.stable_for_future_metric_use
                ),
                "Deterministic perturbation is stable.",
                "Pose perturbation stability is insufficient.",
            ),
            _check(
                "lateral_orientation_resolved",
                lateral_resolved,
                lateral_resolved,
                "Pitch-left/right mapping is resolved in native-video orientation.",
                "Pitch orientation requires explicit semantic evidence.",
            ),
            *[
                _check(
                    f"physical_{item.check_id}",
                    item.value,
                    item.passed,
                    item.reason,
                    item.reason,
                )
                for item in candidate.plausibility_checks
            ],
        ]
        for key, maximum in thresholds.items():
            if key == "optional_crease_support" and not has_validation_crease:
                continue
            value = values[key]
            minimum_metric = key in {
                "wicket_envelope",
                "temporal_stability",
                "independent_scene",
                "optional_crease_support",
            }
            passed = value is not None and (
                value >= maximum if minimum_metric else value <= maximum
            )
            comparator = ">=" if minimum_metric else "<="
            output.append(
                _check(
                    key,
                    value,
                    passed,
                    f"{key} {comparator} {maximum}.",
                    f"{key} did not satisfy {comparator} {maximum}.",
                )
            )
        return output

    metric_checks = checks_for("METRIC_3D_READY", METRIC_THRESHOLDS)
    ground_checks = checks_for("GROUND_PLANE_READY", GROUND_THRESHOLDS)
    if all(item.passed for item in metric_checks):
        level = "METRIC_3D_READY"
        checks = metric_checks
    elif all(item.passed for item in ground_checks):
        level = "GROUND_PLANE_READY"
        checks = ground_checks
    else:
        level = "VISUAL_ONLY"
        checks = ground_checks
    failures = [item.reason for item in checks if not item.passed]
    return SceneCalibrationValidation(
        eligible_level=level,
        checks=checks,
        accepted_anchor_count=len(valid_wickets),
        manually_adjusted_anchor_count=adjusted,
        manually_added_anchor_count=added,
        all_required_checks_passed=not failures,
        failure_reasons=failures,
    )


def refine_scene_calibration(
    analysis_id: str,
    request: SceneCalibrationRefineRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if request.anchor_version != result.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload calibration before refinement.",
            status_code=409,
        )
    if result.setup_frame is None:
        raise VideoAnalysisServiceError(
            "No setup frame is available for refinement.",
            status_code=422,
        )
    all_anchors = [
        *result.current_anchor_set,
        *result.optional_crease_anchors,
    ]
    validated = validate_scene_anchors(
        all_anchors,
        image_width=result.setup_frame.image_width,
        image_height=result.setup_frame.image_height,
    )
    invalid_required = [
        item
        for item in validated
        if item.semantic_id in WICKET_ANCHOR_IDS and not item.valid
    ]
    if invalid_required:
        messages = [
            f"{item.semantic_id}: {message}"
            for item in invalid_required
            for message in item.validation_messages
        ]
        raise VideoAnalysisServiceError(
            "Invalid wicket anchors. " + " ".join(messages),
            status_code=422,
        )
    stage_result = transition_scene_calibration(
        result, "GENERATING_POSE", "Recalculating camera alignment."
    )
    _persist(stage_result)
    wickets = [
        item for item in validated if item.semantic_id in WICKET_ANCHOR_IDS
    ]
    creases = [
        item for item in validated if item.semantic_id in SEMANTIC_ORIENTATION_ANCHOR_IDS
    ]
    inferred_mapping, inferred_points = _infer_mapping_from_semantic_anchors(
        creases
    )
    mapping = result.image_left_mapping or inferred_mapping
    if (
        result.image_left_mapping is not None
        and inferred_mapping is not None
        and result.image_left_mapping != inferred_mapping
    ):
        raise VideoAnalysisServiceError(
            "Semantic anchors contradict the confirmed orientation.",
            status_code=422,
        )
    registration = run_real_pitch_registration(
        analysis_id,
        point_overrides=_point_map(wickets),
        crease_overrides=_point_map(
            item
            for item in creases
            if item.semantic_id in CREASE_ANCHOR_IDS and item.used_for_refinement
        ),
        manual_override_ids={
            item.semantic_id
            for item in wickets
            if item.source in {"manually_adjusted", "manually_added"}
        },
        required_lateral_mapping=(
            _registration_lateral_mapping(mapping)
            if mapping is not None
            else None
        ),
        result_filename=REFINED_REGISTRATION_FILENAME,
        debug_directory=REFINED_DEBUG_DIRECTORY,
    )
    semantic_evidence = (
        _semantic_anchor_evidence(
            mapping=inferred_mapping,
            points=inferred_points,
            frame_index=result.setup_frame.frame_index,
            registration=registration,
        )
        if inferred_mapping is not None
        else []
    )
    evidence = [*result.orientation_evidence, *semantic_evidence]
    orientation_resolution = _orientation_resolution(
        registration=registration,
        evidence=evidence,
        ambiguity_before=(
            result.orientation_resolution.ambiguity_before
            if result.orientation_resolution is not None
            else registration.ambiguity_score
        ),
        image_left_mapping=mapping,
        camera_end=result.camera_end,
    )
    validation = evaluate_calibration_candidate(
        registration.selected_candidate,
        ambiguity_score=registration.ambiguity_score,
        anchors=validated,
        orientation_resolution=orientation_resolution,
    )
    stage: SceneCalibrationStage = _orientation_stage(registration, validation)
    refined = transition_scene_calibration(
        stage_result,
        stage,
        (
            "Alignment passed backend validation and is ready for confirmation."
            if stage in {"GROUND_PLANE_READY", "METRIC_3D_READY"}
            else "Alignment remains visual only. Further adjustment is required."
        ),
    )
    return _persist(
        refined.model_copy(
            update={
                "completed_at": _now(),
                "current_anchor_set": wickets,
                "optional_crease_anchors": creases,
                "refined_registration_summary": _registration_summary(
                    analysis_id, registration, refined=True
                ),
                "selected_candidate": registration.selected_candidate,
                "competing_candidate": registration.competing_candidate,
                "projected_pitch_geometry": (
                    registration.projected_pitch_geometry
                ),
                "competing_projected_pitch_geometry": (
                    registration.competing_projected_pitch_geometry
                ),
                "orientation_required": stage == "ORIENTATION_REQUIRED",
                "orientation_resolution": orientation_resolution,
                "image_left_mapping": mapping,
                "orientation_evidence": evidence,
                "validation": validation,
                "calibration_level": validation.eligible_level,
                "accepted_calibration": None,
                "metrics_unlocked": [],
                "metrics_locked_reasons": (
                    validation.failure_reasons
                    or ["User acceptance is still required."]
                ),
            }
        )
    )


def _validated_scene_anchors_for_result(
    result: SceneCalibrationResult,
) -> tuple[list[SceneCalibrationAnchor], list[SceneCalibrationAnchor]]:
    if result.setup_frame is None:
        raise VideoAnalysisServiceError(
            "No setup frame is available for calibration.",
            status_code=422,
        )
    validated = validate_scene_anchors(
        [*result.current_anchor_set, *result.optional_crease_anchors],
        image_width=result.setup_frame.image_width,
        image_height=result.setup_frame.image_height,
    )
    wickets = [
        item for item in validated if item.semantic_id in WICKET_ANCHOR_IDS
    ]
    semantic = [
        item
        for item in validated
        if item.semantic_id in SEMANTIC_ORIENTATION_ANCHOR_IDS
    ]
    return wickets, semantic


def _create_orientation_preset(
    *,
    result: SceneCalibrationResult,
    mapping: ImageLeftMapping,
    camera_end: str,
    preset_name: str | None,
) -> CameraOrientationPreset | None:
    if result.setup_frame is None:
        return None
    now = _now()
    seed = (
        f"{result.analysis_id}:{result.setup_frame.image_width}:"
        f"{result.setup_frame.image_height}:{mapping}:{camera_end}"
    )
    preset = CameraOrientationPreset(
        preset_id=f"orientation_preset_{hashlib.sha1(seed.encode('utf-8')).hexdigest()[:12]}",
        preset_name=preset_name or f"Fixed camera {mapping.lower()}",
        created_at=now,
        updated_at=now,
        source_analysis_id=result.analysis_id,
        native_width=result.setup_frame.image_width,
        native_height=result.setup_frame.image_height,
        rotation_metadata=None,
        camera_end=camera_end if camera_end in {"bowler", "striker"} else "unknown",
        image_left_mapping=mapping,
        confidence=0.9,
        user_confirmed=True,
        compatible=True,
        compatibility_reasons=["created_from_current_user_confirmation"],
    )
    presets = [
        item for item in _load_presets() if item.preset_id != preset.preset_id
    ]
    _write_presets([*presets, preset])
    return preset


def apply_scene_calibration_orientation(
    analysis_id: str,
    request: SceneCalibrationOrientationRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload before confirming orientation.",
            status_code=409,
        )
    if result.selected_candidate is None and result.setup_frame is None:
        raise VideoAnalysisServiceError(
            "Run scene calibration before confirming orientation.",
            status_code=422,
        )
    if request.image_left_mapping == "NOT_SURE":
        resolution = OrientationResolution(
            required=True,
            resolved=False,
            camera_end=request.camera_end,
            ambiguity_before=(
                result.orientation_resolution.ambiguity_before
                if result.orientation_resolution is not None
                else 1.0
            ),
            ambiguity_after=(
                result.orientation_resolution.ambiguity_after
                if result.orientation_resolution is not None
                else 1.0
            ),
            selected_candidate_id=(
                result.selected_candidate.candidate_id
                if result.selected_candidate is not None
                else None
            ),
            symmetric_evidence_insufficient=SYMMETRIC_EVIDENCE_INSUFFICIENT,
            remaining_failures=[
                "User selected not sure; lateral orientation remains unresolved."
            ],
        )
        return _persist(
            result.model_copy(
                update={
                    "stage": "ORIENTATION_REQUIRED",
                    "orientation_required": True,
                    "orientation_resolution": resolution,
                    "metrics_unlocked": [],
                    "metrics_locked_reasons": [
                        "Pitch orientation remains unresolved."
                    ],
                    "updated_at": _now(),
                    "message": "Pitch orientation still needs confirmation.",
                }
            )
        )
    mapping: ImageLeftMapping = request.image_left_mapping
    wickets, semantic = _validated_scene_anchors_for_result(result)
    invalid_required = [
        item for item in wickets if item.semantic_id in WICKET_ANCHOR_IDS and not item.valid
    ]
    if invalid_required:
        raise VideoAnalysisServiceError(
            "Invalid wicket anchors must be corrected before orientation acceptance.",
            status_code=422,
        )
    inferred_mapping, inferred_points = _infer_mapping_from_semantic_anchors(
        semantic
    )
    if inferred_mapping is not None and inferred_mapping != mapping:
        raise VideoAnalysisServiceError(
            "Semantic anchors contradict the selected orientation.",
            status_code=422,
        )
    stage_result = transition_scene_calibration(
        result, "GENERATING_POSE", "Applying orientation evidence."
    )
    _persist(stage_result)
    registration = run_real_pitch_registration(
        analysis_id,
        point_overrides=_point_map(wickets),
        crease_overrides=_point_map(
            item
            for item in semantic
            if item.semantic_id in CREASE_ANCHOR_IDS and item.used_for_refinement
        ),
        manual_override_ids={
            item.semantic_id
            for item in wickets
            if item.source in {"manually_adjusted", "manually_added"}
        },
        required_lateral_mapping=_registration_lateral_mapping(mapping),
        result_filename=REFINED_REGISTRATION_FILENAME,
        debug_directory=REFINED_DEBUG_DIRECTORY,
    )
    evidence = [
        *result.orientation_evidence,
        _user_orientation_evidence(
            mapping=mapping,
            camera_end=request.camera_end,
            registration=registration,
        ),
    ]
    if inferred_mapping is not None:
        evidence.extend(
            _semantic_anchor_evidence(
                mapping=inferred_mapping,
                points=inferred_points,
                frame_index=result.setup_frame.frame_index,
                registration=registration,
            )
        )
    preset = (
        _create_orientation_preset(
            result=result,
            mapping=mapping,
            camera_end=request.camera_end,
            preset_name=request.preset_name,
        )
        if request.create_preset
        and request.user_confirmed_same_fixed_setup
        else None
    )
    if preset is not None:
        evidence.append(
            OrientationEvidence(
                evidence_id=_orientation_evidence_id(preset.preset_id),
                evidence_type="SAVED_CAMERA_ORIENTATION_PRESET",
                source="saved_preset",
                semantic_label=preset.image_left_mapping,
                confidence=preset.confidence,
                uncertainty=0.1,
                authoritative=True,
                supports_candidate_ids=[
                    item.candidate_id
                    for item in registration.candidates
                    if _candidate_mapping(item) == mapping
                ],
                rejects_candidate_ids=[
                    item.candidate_id
                    for item in registration.candidates
                    if _candidate_mapping(item) != mapping
                ],
                explanation=(
                    "Orientation preset saved for explicit future reuse; it "
                    "does not accept the current camera pose."
                ),
                created_at=_now(),
                user_confirmed=True,
            )
        )
    ambiguity_before = (
        result.orientation_resolution.ambiguity_before
        if result.orientation_resolution is not None
        else result.refined_registration_summary.ambiguity_score
        if result.refined_registration_summary is not None
        else result.automatic_registration_summary.ambiguity_score
        if result.automatic_registration_summary is not None
        else registration.ambiguity_score
    )
    orientation_resolution = _orientation_resolution(
        registration=registration,
        evidence=evidence,
        ambiguity_before=ambiguity_before,
        image_left_mapping=mapping,
        camera_end=request.camera_end,
    )
    validation = evaluate_calibration_candidate(
        registration.selected_candidate,
        ambiguity_score=registration.ambiguity_score,
        anchors=[*wickets, *semantic],
        orientation_resolution=orientation_resolution,
    )
    stage = _orientation_stage(registration, validation)
    updated = transition_scene_calibration(
        stage_result,
        stage,
        (
            "Orientation resolved. Review calibration quality before accepting."
            if orientation_resolution.resolved
            else "Orientation evidence applied but ambiguity remains."
        ),
    )
    return _persist(
        updated.model_copy(
            update={
                "completed_at": _now(),
                "current_anchor_set": wickets,
                "optional_crease_anchors": semantic,
                "refined_registration_summary": _registration_summary(
                    analysis_id, registration, refined=True
                ),
                "selected_candidate": registration.selected_candidate,
                "competing_candidate": registration.competing_candidate,
                "projected_pitch_geometry": registration.projected_pitch_geometry,
                "competing_projected_pitch_geometry": (
                    registration.competing_projected_pitch_geometry
                ),
                "orientation_required": stage == "ORIENTATION_REQUIRED",
                "image_left_mapping": mapping,
                "camera_end": request.camera_end,
                "orientation_evidence": evidence,
                "orientation_resolution": orientation_resolution,
                "orientation_preset_id": preset.preset_id if preset else result.orientation_preset_id,
                "validation": validation,
                "calibration_level": validation.eligible_level,
                "accepted_calibration": None,
                "metrics_unlocked": [],
                "metrics_locked_reasons": (
                    validation.failure_reasons
                    or ["User acceptance is still required."]
                ),
            }
        )
    )


def clear_scene_calibration_orientation(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload before clearing orientation.",
            status_code=409,
        )
    return _persist(
        result.model_copy(
            update={
                "image_left_mapping": None,
                "camera_end": None,
                "orientation_evidence": [],
                "orientation_resolution": None,
                "orientation_preset_id": None,
                "orientation_required": bool(result.selected_candidate),
                "accepted_calibration": None,
                "metrics_unlocked": [],
                "metrics_locked_reasons": [
                    "Pitch orientation has been cleared."
                ],
                "stage": (
                    "ORIENTATION_REQUIRED"
                    if result.selected_candidate is not None
                    else result.stage
                ),
                "updated_at": _now(),
                "message": "Pitch orientation cleared.",
            }
        )
    )


def apply_scene_calibration_preset(
    analysis_id: str,
    request: SceneCalibrationPresetRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload before using the preset.",
            status_code=409,
        )
    if not request.user_confirmed_same_fixed_setup:
        raise VideoAnalysisServiceError(
            "Preset reuse requires explicit same fixed setup confirmation.",
            status_code=422,
        )
    presets = load_orientation_presets_for_analysis(analysis_id)
    preset = next(
        (
            item
            for item in presets.compatible_presets
            if item.preset_id == request.preset_id
        ),
        None,
    )
    if preset is None:
        raise VideoAnalysisServiceError(
            "Orientation preset is not compatible with this video.",
            status_code=422,
        )
    return apply_scene_calibration_orientation(
        analysis_id,
        SceneCalibrationOrientationRequest(
            anchor_version=request.anchor_version,
            image_left_mapping=preset.image_left_mapping,
            camera_end=preset.camera_end,
            create_preset=False,
            user_confirmed_same_fixed_setup=True,
        ),
    )


def _camera_arrays(candidate: CameraPoseCandidate):
    if (
        candidate.rotation_vector is None
        or candidate.translation_vector is None
        or candidate.rotation_matrix is None
        or candidate.camera_world_position is None
    ):
        raise VideoAnalysisServiceError(
            "Selected camera candidate is incomplete.",
            status_code=422,
        )
    camera_matrix = np.asarray(
        [
            [
                candidate.intrinsics.focal_length_x_px,
                0.0,
                candidate.intrinsics.principal_point_x_px,
            ],
            [
                0.0,
                candidate.intrinsics.focal_length_y_px,
                candidate.intrinsics.principal_point_y_px,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotation = np.asarray(candidate.rotation_matrix, dtype=np.float64)
    translation = np.asarray(
        candidate.translation_vector, dtype=np.float64
    ).reshape(3, 1)
    projection = camera_matrix @ np.hstack([rotation, translation])
    ground_to_image = camera_matrix @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation.reshape(3)]
    )
    try:
        image_to_ground = np.linalg.inv(ground_to_image)
    except np.linalg.LinAlgError as exc:
        raise VideoAnalysisServiceError(
            "Accepted camera has a singular ground-plane transform.",
            status_code=422,
        ) from exc
    ground_to_image /= ground_to_image[2, 2]
    image_to_ground /= image_to_ground[2, 2]
    return camera_matrix, projection, image_to_ground, ground_to_image


def _next_revision(analysis_id: str) -> tuple[int, Path]:
    reports = VIDEO_ANALYSIS_ROOT / analysis_id / "reports"
    first = reports / ACCEPTED_FILENAME
    if not first.exists():
        return 1, first
    revision = 2
    while (reports / f"accepted_scene_calibration_v1_r{revision}.json").exists():
        revision += 1
    return revision, reports / f"accepted_scene_calibration_v1_r{revision}.json"


def _snapshot_url(analysis_id: str, path: Path) -> str:
    return f"/static/video-analysis/{analysis_id}/reports/{path.name}"


def _rerun_physics_if_available(analysis_id: str) -> str | None:
    try:
        from .video_ball_tracking_service import (
            PHYSICS_RESULT_FILENAME,
            TRACKING_RESULT_FILENAME,
            _load_detection_document,
        )
        from ..schemas.video_analysis import VideoBallTrackingDocument
        from .delivery_physics_service import analyse_delivery_physics

        analysis = load_video_analysis(analysis_id)
        tracking_dir = VIDEO_ANALYSIS_ROOT / analysis_id / "tracking"
        tracking_path = tracking_dir / TRACKING_RESULT_FILENAME
        if not tracking_path.is_file():
            return "Tracking has not completed; Physics V1 was not rerun."
        tracking = VideoBallTrackingDocument.model_validate_json(
            tracking_path.read_text(encoding="utf-8")
        )
        detections = _load_detection_document(
            analysis_id, analysis.frame_count
        )
        physics = analyse_delivery_physics(
            analysis_id=analysis_id,
            primary_track=tracking.primary_track,
            detections=detections,
            tracker_bounce=tracking.bounce,
            fps=analysis.fps,
            width=analysis.width,
            height=analysis.height,
            total_frames=analysis.frame_count,
        ).model_copy(
            update={
                "physics_result_url": (
                    f"/static/video-analysis/{analysis_id}/tracking/"
                    f"{PHYSICS_RESULT_FILENAME}"
                )
            }
        )
        updated_tracking = tracking.model_copy(update={"physics": physics})
        _write_json(tracking_path, updated_tracking)
        _write_json(tracking_dir / PHYSICS_RESULT_FILENAME, physics)
        return None
    except Exception as exc:
        return (
            "Accepted calibration was saved, but Physics V1 rerun failed "
            f"without changing tracking: {type(exc).__name__}."
        )


def accept_scene_calibration(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload before acceptance.",
            status_code=409,
        )
    candidate = result.selected_candidate
    validation = result.validation
    if (
        candidate is None
        or validation is None
        or validation.eligible_level
        not in {"GROUND_PLANE_READY", "METRIC_3D_READY"}
        or not validation.all_required_checks_passed
    ):
        raise VideoAnalysisServiceError(
            "Backend validation does not permit metric calibration acceptance.",
            status_code=422,
        )
    if request.candidate_id and request.candidate_id != candidate.candidate_id:
        raise VideoAnalysisServiceError(
            "Selected candidate changed. Reload before acceptance.",
            status_code=409,
        )
    if result.setup_frame is None:
        raise VideoAnalysisServiceError(
            "Accepted calibration requires a setup frame.",
            status_code=422,
        )
    (
        camera_matrix,
        projection,
        image_to_ground,
        ground_to_image,
    ) = _camera_arrays(candidate)
    revision, path = _next_revision(analysis_id)
    snapshot = AcceptedSceneCalibrationSnapshot(
        analysis_id=analysis_id,
        revision=revision,
        accepted_at=_now(),
        calibration_level=validation.eligible_level,
        candidate_id=candidate.candidate_id,
        setup_frame=result.setup_frame,
        supporting_frames=result.supporting_frames,
        setup_frame_image_url=result.setup_frame_image_url,
        raw_wicket_overlay_url=result.raw_wicket_overlay_url,
        anchors=result.current_anchor_set,
        optional_crease_anchors=result.optional_crease_anchors,
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=candidate.intrinsics.distortion_coefficients,
        rotation_vector=candidate.rotation_vector,
        rotation_matrix=candidate.rotation_matrix,
        translation_vector=candidate.translation_vector,
        projection_matrix=projection.tolist(),
        camera_world_position=candidate.camera_world_position,
        image_to_pitch_homography=image_to_ground.tolist(),
        pitch_to_image_homography=ground_to_image.tolist(),
        end_assignment=candidate.assignment_hypothesis,
        reprojection_rmse_px=candidate.reprojection_rmse_px or 0.0,
        median_reprojection_error_px=(
            candidate.median_reprojection_error_px or 0.0
        ),
        maximum_inlier_error_px=candidate.maximum_inlier_error_px or 0.0,
        correspondence_count=len(candidate.inlier_correspondence_ids),
        uncertainty=(
            candidate.uncertainty.model_dump(mode="json")
            if candidate.uncertainty is not None
            else {}
        ),
        validation=validation,
        orientation_evidence=result.orientation_evidence,
        image_left_mapping=result.image_left_mapping,
        camera_end=result.camera_end,
        ambiguity_before_resolution=(
            result.orientation_resolution.ambiguity_before
            if result.orientation_resolution is not None
            else None
        ),
        ambiguity_after_resolution=(
            result.orientation_resolution.ambiguity_after
            if result.orientation_resolution is not None
            else None
        ),
        selected_mirror_candidate=(
            result.orientation_resolution.selected_candidate_id
            if result.orientation_resolution is not None
            else candidate.candidate_id
        ),
        rejected_mirror_candidate=(
            result.orientation_resolution.rejected_candidate_ids[0]
            if result.orientation_resolution is not None
            and result.orientation_resolution.rejected_candidate_ids
            else None
        ),
        orientation_preset_id=result.orientation_preset_id,
        semantic_anchor_version=result.anchor_version,
        user_confirmation_timestamp=(
            result.orientation_evidence[-1].created_at
            if result.orientation_evidence
            else None
        ),
    )
    _write_json(path, snapshot)
    accepted_at = snapshot.accepted_at
    summary = AcceptedCalibrationSummary(
        revision=revision,
        accepted_at=accepted_at,
        accepted_level=validation.eligible_level,
        accepted_candidate_id=candidate.candidate_id,
        anchor_version=result.anchor_version,
        snapshot_url=_snapshot_url(analysis_id, path),
        image_left_mapping=result.image_left_mapping,
        orientation_preset_id=result.orientation_preset_id,
    )
    metrics = (
        METRIC_3D_METRICS
        if validation.eligible_level == "METRIC_3D_READY"
        else GROUND_PLANE_METRICS
    )
    warning = _rerun_physics_if_available(analysis_id)
    warnings = [*result.warnings, *([warning] if warning else [])]
    return _persist(
        result.model_copy(
            update={
                "accepted_calibration": summary,
                "calibration_level": validation.eligible_level,
                "stage": validation.eligible_level,
                "metrics_unlocked": metrics,
                "metrics_locked_reasons": (
                    []
                    if validation.eligible_level == "METRIC_3D_READY"
                    else [
                        "Airborne speed, height, and metric swing require METRIC_3D_READY."
                    ]
                ),
                "warnings": warnings,
                "updated_at": accepted_at,
                "message": (
                    f"Calibration revision {revision} accepted as "
                    f"{validation.eligible_level}."
                ),
            }
        )
    )


def reject_scene_calibration(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload before rejection.",
            status_code=409,
        )
    return _persist(
        result.model_copy(
            update={
                "stage": "NEEDS_ADJUSTMENT",
                "accepted_calibration": None,
                "metrics_unlocked": [],
                "metrics_locked_reasons": [
                    "Calibration was rejected by the user."
                ],
                "updated_at": _now(),
                "message": "Calibration rejected. Diagnostics were preserved.",
            }
        )
    )


def use_visual_overlay_only(
    analysis_id: str,
    request: SceneCalibrationActionRequest,
) -> SceneCalibrationResult:
    result = load_scene_calibration(analysis_id)
    if result.anchor_version != request.anchor_version:
        raise VideoAnalysisServiceError(
            "Anchor version is stale. Reload before selecting visual overlay.",
            status_code=409,
        )
    if result.selected_candidate is None:
        raise VideoAnalysisServiceError(
            "No visual calibration candidate is available.",
            status_code=422,
        )
    return _persist(
        result.model_copy(
            update={
                "stage": "NEEDS_ADJUSTMENT",
                "calibration_level": "VISUAL_ONLY",
                "visual_overlay_enabled": True,
                "accepted_calibration": None,
                "metrics_unlocked": [],
                "metrics_locked_reasons": [
                    "Visual overlay mode never unlocks metric analytics."
                ],
                "updated_at": _now(),
                "message": "Visual overlay enabled. Metric analytics remain locked.",
            }
        )
    )


def load_active_accepted_scene_calibration(
    analysis_id: str,
) -> AcceptedSceneCalibrationSnapshot:
    result = load_scene_calibration(analysis_id)
    accepted = result.accepted_calibration
    if accepted is None:
        raise VideoAnalysisServiceError(
            "No active accepted assisted calibration exists.",
            status_code=404,
        )
    filename = Path(accepted.snapshot_url).name
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / filename
    try:
        snapshot = AcceptedSceneCalibrationSnapshot.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Accepted assisted calibration snapshot is unavailable.",
            status_code=500,
        ) from exc
    if (
        snapshot.revision != accepted.revision
        or snapshot.candidate_id != accepted.accepted_candidate_id
    ):
        raise VideoAnalysisServiceError(
            "Accepted assisted calibration snapshot does not match active state.",
            status_code=500,
        )
    return snapshot
