from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from services.api.schemas.real_pitch_registration import (
    RealPitchRegistrationResult,
    RegistrationDiagnostics,
)
from services.api.schemas.wicket_observation import (
    AssignmentHypothesis,
    PixelBox,
    PixelPoint,
    RawWicketDetection,
    RoiMetadata,
    SetupFrameCandidate,
    WicketLandmarkObservation,
    WicketLineObservation,
    WicketObservation,
    WicketObservationDiagnostics,
    WicketObservationResult,
    WicketRegionObservation,
)
from services.api.services.real_pitch_registration_service import (
    MIN_ASSISTED_POINTLIKE_ANCHORS,
    MIN_POINTLIKE_ANCHORS,
    PERTURBATION_SEED,
    _perturbation_uncertainty,
    apply_assisted_anchor_overrides,
    build_intrinsics_candidates,
    build_registration_correspondences,
    check_registration_eligibility,
    classify_pose_candidate,
    resolve_registration_status,
    solve_pose_candidate,
)
from services.api.services.virtual_pitch_service import (
    build_synthetic_camera,
    project_virtual_pitch,
)


def _setup_frame(index: int = 4, *, selected: bool = False) -> SetupFrameCandidate:
    return SetupFrameCandidate(
        frame_index=index,
        timestamp_seconds=index / 30,
        image_width=1280,
        image_height=720,
        score=0.9,
        sharpness=100,
        brightness=120,
        wicket_detection_count=2,
        mean_detector_confidence=0.9,
        detection_stability=0.9,
        obstruction_score=0.0,
        selected=selected,
    )


def _synthetic_wicket(end: str, role: str, *, noise_px: float = 0.0) -> WicketObservation:
    projection = project_virtual_pitch(build_synthetic_camera("centred_bowler_end"))
    pixels = {
        item.semantic_id: item.pixel_point
        for item in projection.projected_landmarks
        if item.pixel_point is not None
    }
    ids = {
        "wicket_outer_left_base": f"{end}_left_stump_base",
        "wicket_outer_right_base": f"{end}_right_stump_base",
        "wicket_base_center": f"{end}_wicket_center_base",
        "wicket_outer_left_top": f"{end}_left_stump_top",
        "wicket_outer_right_top": f"{end}_right_stump_top",
        "wicket_top_center": f"{end}_middle_stump_top",
        "wicket_center": f"{end}_middle_stump_top",
    }
    landmarks: list[WicketLandmarkObservation] = []
    for index, (observed_id, virtual_id) in enumerate(ids.items()):
        point = pixels[virtual_id]
        y = (
            (point.y + pixels[f"{end}_middle_stump_base"].y) / 2
            if observed_id == "wicket_center"
            else point.y
        )
        landmarks.append(
            WicketLandmarkObservation(
                semantic_id=observed_id,
                geometry_type="POINT",
                pixel_x=point.x + noise_px * ((index % 3) - 1),
                pixel_y=y + noise_px * ((index % 2) - 0.5),
                confidence=0.95,
                uncertainty_px=max(1.0, noise_px * 2),
                extraction_method="synthetic_projection",
                supporting_evidence=["canonical_virtual_pitch"],
                supporting_frames=[4, 5, 6],
                registration_role=(
                    "SECONDARY_ANCHOR"
                    if observed_id == "wicket_center"
                    else "PRIMARY_ANCHOR"
                ),
                quality="HIGH",
                status="AVAILABLE",
            )
        )
    line_pairs = {
        "base_line": ("left_stump_base", "right_stump_base"),
        "top_or_bail_line": ("left_stump_top", "right_stump_top"),
        "left_outer_axis": ("left_stump_base", "left_stump_top"),
        "right_outer_axis": ("right_stump_base", "right_stump_top"),
    }
    for observed_id, (start_id, end_id) in line_pairs.items():
        start = pixels[f"{end}_{start_id}"]
        finish = pixels[f"{end}_{end_id}"]
        landmarks.append(
            WicketLandmarkObservation(
                semantic_id=observed_id,
                geometry_type="LINE",
                line=WicketLineObservation(
                    start=PixelPoint(x=start.x, y=start.y),
                    end=PixelPoint(x=finish.x, y=finish.y),
                ),
                confidence=0.9,
                uncertainty_px=1.5,
                extraction_method="synthetic_projection",
                supporting_frames=[4, 5, 6],
                registration_role="SECONDARY_ANCHOR",
                quality="HIGH",
                status="AVAILABLE",
            )
        )
    stump_points = [
        pixels[f"{end}_{side}_stump_{level}"]
        for side in ("left", "middle", "right")
        for level in ("base", "top")
    ]
    left = min(point.x for point in stump_points) - 2
    top = min(point.y for point in stump_points) - 2
    right = max(point.x for point in stump_points) + 2
    bottom = max(point.y for point in stump_points) + 2
    box = PixelBox(x=left, y=top, width=right - left, height=bottom - top)
    region = WicketRegionObservation(
        frame_index=4,
        timestamp_seconds=4 / 30,
        bbox=box,
        centre=PixelPoint(x=(left + right) / 2, y=(top + bottom) / 2),
        width=box.width,
        height=box.height,
        detector_confidence=0.94,
        detector_model="synthetic",
        source="test",
        temporal_support=3,
        supporting_frame_ids=[4, 5, 6],
        centre_variation_px=0.5,
        size_variation_ratio=0.01,
        confidence_variation=0.01,
        perspective_role=(
            "NEAR_WICKET_CANDIDATE" if role == "near" else "FAR_WICKET_CANDIDATE"
        ),
        stability="STABLE",
        quality="HIGH",
        uncertainty_px=1.5,
    )
    return WicketObservation(
        region=region,
        roi=RoiMetadata(
            source_frame_width=1280,
            source_frame_height=720,
            x=max(0, int(left) - 5),
            y=max(0, int(top) - 5),
            width=max(1, int(box.width) + 10),
            height=max(1, int(box.height) + 10),
            padding_x=5,
            padding_y=5,
        ),
        coarse_landmarks=landmarks,
        detailed_landmarks=[],
        detailed_landmarks_status="INSUFFICIENT_EVIDENCE",
        quality_score=0.95,
        quality_factors={"frame_edge_clipping": 1.0},
    )


def _synthetic_observation(*, noise_px: float = 0.0) -> WicketObservationResult:
    near = _synthetic_wicket("bowler", "near", noise_px=noise_px)
    far = _synthetic_wicket("striker", "far", noise_px=noise_px)
    raw = []
    for frame_index in (4, 5, 6):
        for wicket in (near, far):
            raw.append(
                RawWicketDetection(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / 30,
                    bbox=wicket.region.bbox,
                    confidence=0.94,
                    class_name="stump_set",
                    source="test",
                    detector_model="synthetic",
                    perspective_role=wicket.region.perspective_role,
                )
            )
    return WicketObservationResult(
        analysis_id="analysis_synthetic_registration",
        status="READY_FOR_REGISTRATION_EXPERIMENT",
        setup_frame=_setup_frame(selected=True),
        supporting_frames=[_setup_frame(5), _setup_frame(6)],
        frame_candidates=[_setup_frame(selected=True), _setup_frame(5), _setup_frame(6)],
        near_wicket=near,
        far_wicket=far,
        assignment_hypotheses=[
            AssignmentHypothesis(
                hypothesis_id="A",
                near_semantic_end="bowler",
                far_semantic_end="striker",
                confidence=0.5,
            ),
            AssignmentHypothesis(
                hypothesis_id="B",
                near_semantic_end="striker",
                far_semantic_end="bowler",
                confidence=0.5,
            ),
        ],
        diagnostics=WicketObservationDiagnostics(
            detector_model_path="synthetic",
            detector_class_labels=["stump_set"],
            clean_source_video="/static/raw.mp4",
            sampled_frame_ids=[4, 5, 6],
            raw_detections=raw,
        ),
        future_registration_readiness="READY_FOR_REGISTRATION_EXPERIMENT",
        message="synthetic",
    )


def _solve(
    observation: WicketObservationResult,
    *,
    assignment: str = "A",
    lateral_mapping: str = "image_left_to_world_left",
):
    intrinsics = build_intrinsics_candidates(1280, 720)[1]
    correspondences = build_registration_correspondences(
        observation,
        assignment_hypothesis=assignment,
        lateral_mapping=lateral_mapping,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    candidate = solve_pose_candidate(
        "synthetic",
        assignment,
        lateral_mapping,
        4,
        intrinsics,
        correspondences,
        frame,
        observation,
    )
    return candidate, correspondences


def test_eligibility_requires_two_stable_unclipped_wickets() -> None:
    observation = _synthetic_observation()
    assert check_registration_eligibility(observation).eligible

    clipped = observation.model_copy(
        update={
            "near_wicket": observation.near_wicket.model_copy(
                update={"quality_factors": {"frame_edge_clipping": 0.4}}
            )
        }
    )
    result = check_registration_eligibility(clipped)
    assert not result.eligible
    assert "near_wicket_severely_clipped" in result.reasons


def test_eligibility_rejects_source_dimension_mismatch() -> None:
    observation = _synthetic_observation()
    observation.near_wicket.roi.source_frame_width = 1920
    result = check_registration_eligibility(observation)
    assert not result.eligible
    assert "near_wicket_source_dimensions_mismatch" in result.reasons


def test_correspondence_semantics_keep_boxes_and_lines_soft() -> None:
    correspondences = build_registration_correspondences(
        _synthetic_observation(),
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
    )
    pointlike = [item for item in correspondences if item.exactness == "POINTLIKE"]
    soft = [item for item in correspondences if item.exactness == "SOFT"]
    assert len([item for item in pointlike if item.status == "USED"]) >= MIN_POINTLIKE_ANCHORS
    assert all(item.constraint_category == "SOFT_GEOMETRIC_CONSTRAINT" for item in soft)
    assert all(
        item.exactness != "EXACT"
        for item in correspondences
        if item.observed_bbox is not None
    )


def test_uncertainty_reduces_correspondence_weight() -> None:
    observation = _synthetic_observation()
    normal = build_registration_correspondences(
        observation,
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
    )
    landmark = observation.near_wicket.coarse_landmarks[0]
    observation.near_wicket.coarse_landmarks[0] = landmark.model_copy(
        update={"uncertainty_px": 30.0}
    )
    uncertain = build_registration_correspondences(
        observation,
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
    )
    assert uncertain[0].registration_weight < normal[0].registration_weight


def test_complete_manual_wicket_anchors_supersede_stale_role_constraints() -> None:
    observation = _synthetic_observation()
    correspondences = build_registration_correspondences(
        observation,
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
    )
    overrides = {
        "near_left_base": PixelPoint(x=200, y=600),
        "near_right_base": PixelPoint(x=320, y=600),
        "near_top_center": PixelPoint(x=260, y=450),
    }
    updated = apply_assisted_anchor_overrides(
        correspondences,
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
        point_overrides=overrides,
        manual_override_ids=set(overrides),
    )
    manual_ids = {
        "near:wicket_outer_left_base",
        "near:wicket_outer_right_base",
        "near:wicket_top_center",
    }
    assert all(
        item.status == "USED" and item.exactness == "EXACT"
        for item in updated
        if item.correspondence_id in manual_ids
    )
    assert all(
        item.status == "REJECTED"
        and item.rejection_reason
        == "superseded_by_complete_manual_wicket_anchors"
        for item in updated
        if item.observed_wicket_role == "near"
        and item.correspondence_id not in manual_ids
        and item.observed_pixel is not None
    )
    assert any(
        item.status == "USED" and item.observed_wicket_role == "far"
        for item in updated
    )


def test_six_complete_manual_anchors_can_seed_assisted_pose() -> None:
    observation = _synthetic_observation(noise_px=0.2)
    correspondences = build_registration_correspondences(
        observation,
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
    )
    semantic_to_correspondence = {
        "near_left_base": "near:wicket_outer_left_base",
        "near_right_base": "near:wicket_outer_right_base",
        "near_top_center": "near:wicket_top_center",
        "far_left_base": "far:wicket_outer_left_base",
        "far_right_base": "far:wicket_outer_right_base",
        "far_top_center": "far:wicket_top_center",
    }
    by_id = {item.correspondence_id: item for item in correspondences}
    overrides = {
        semantic_id: by_id[correspondence_id].observed_pixel
        for semantic_id, correspondence_id in semantic_to_correspondence.items()
    }
    assisted = apply_assisted_anchor_overrides(
        correspondences,
        assignment_hypothesis="A",
        lateral_mapping="image_left_to_world_left",
        point_overrides=overrides,
        manual_override_ids=set(overrides),
    )
    candidate = solve_pose_candidate(
        "assisted-six",
        "A",
        "image_left_to_world_left",
        4,
        build_intrinsics_candidates(1280, 720)[1],
        assisted,
        np.zeros((720, 1280, 3), dtype=np.uint8),
        observation,
        minimum_pointlike_anchors=MIN_ASSISTED_POINTLIKE_ANCHORS,
    )
    assert candidate.solver_success
    expected_fov = math.degrees(
        2.0
        * math.atan(
            1280.0 / (2.0 * candidate.intrinsics.focal_length_x_px)
        )
    )
    assert candidate.intrinsics.horizontal_fov_degrees == pytest.approx(
        expected_fov
    )


def test_intrinsics_are_bounded_centred_and_zero_distortion() -> None:
    candidates = build_intrinsics_candidates(1280, 720)
    assert len(candidates) == 3
    assert [item.horizontal_fov_degrees for item in candidates] == [45, 60, 75]
    assert all(item.principal_point_x_px == 640 for item in candidates)
    assert all(item.principal_point_y_px == 360 for item in candidates)
    assert all(item.distortion_coefficients == [0.0] * 5 for item in candidates)
    assert all(not item.principal_point_refined for item in candidates)


def test_correct_assignment_solves_and_wrong_end_assignment_is_not_preferred() -> None:
    observation = _synthetic_observation(noise_px=0.2)
    correct, _ = _solve(observation, assignment="A")
    wrong, _ = _solve(observation, assignment="B")
    assert correct.solver_success
    assert correct.eligible_for_selection
    assert correct.reprojection_rmse_px is not None
    assert wrong.score < correct.score or not wrong.eligible_for_selection


def test_robust_solver_handles_noise_and_marks_a_large_outlier() -> None:
    observation = _synthetic_observation(noise_px=0.5)
    target = observation.near_wicket.coarse_landmarks[0]
    observation.near_wicket.coarse_landmarks[0] = target.model_copy(
        update={"pixel_x": target.pixel_x + 100, "pixel_y": target.pixel_y - 80}
    )
    candidate, _ = _solve(observation)
    assert candidate.solver_success
    assert candidate.refinement.attempted
    assert candidate.refinement.robust_loss == "soft_l1"
    assert "near:wicket_outer_left_base" in candidate.outlier_correspondence_ids


def test_insufficient_and_degenerate_correspondences_fail_explicitly() -> None:
    observation = _synthetic_observation()
    for wicket in (observation.near_wicket, observation.far_wicket):
        for item in wicket.coarse_landmarks[3:]:
            item.status = "UNAVAILABLE"
    candidate, _ = _solve(observation)
    assert not candidate.solver_success
    assert "insufficient_pointlike_anchors" in candidate.failure_reasons


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"failed_hard_checks": ["camera_above_pitch"]}, "REGISTRATION_FAILED"),
        (
            {
                "score": 0.8,
                "reprojection_rmse_px": 4,
                "exact_anchor_count": 6,
                "vertical_anchor_count": 4,
                "temporal_stability_score": 0.8,
                "independent_scene_score": 0.6,
            },
            "METRIC_3D_CANDIDATE",
        ),
        (
            {
                "score": 0.6,
                "reprojection_rmse_px": 7,
                "wicket_envelope_score": 0.5,
            },
            "GROUND_PLANE_CANDIDATE",
        ),
        ({"score": 0.3}, "VISUAL_ONLY"),
    ],
)
def test_candidate_classification_boundaries(arguments: dict[str, object], expected: str) -> None:
    defaults = {
        "failed_hard_checks": [],
        "score": 0.3,
        "reprojection_rmse_px": 20.0,
        "exact_anchor_count": 0,
        "vertical_anchor_count": 0,
        "temporal_stability_score": 0.0,
        "independent_scene_score": 0.0,
        "wicket_envelope_score": 0.0,
    }
    assert classify_pose_candidate(**(defaults | arguments)) == expected


def test_result_status_exposes_ambiguity_and_uncertainty_downgrade() -> None:
    ambiguous, ambiguity = resolve_registration_status(
        selected_classification="GROUND_PLANE_CANDIDATE",
        selected_score=0.61,
        competing_score=0.57,
        uncertainty_stable=True,
    )
    unstable, _ = resolve_registration_status(
        selected_classification="METRIC_3D_CANDIDATE",
        selected_score=0.85,
        competing_score=None,
        uncertainty_stable=False,
    )
    assert ambiguous == "AMBIGUOUS"
    assert ambiguity == pytest.approx(0.5)
    assert unstable == "GROUND_PLANE_CANDIDATE"


def test_uncertainty_is_deterministic_and_candidate_only() -> None:
    candidate, correspondences = _solve(_synthetic_observation(noise_px=0.2))
    first = _perturbation_uncertainty(candidate, correspondences, 1280, 720)
    second = _perturbation_uncertainty(candidate, correspondences, 1280, 720)
    assert first == second
    assert first.deterministic_seed == PERTURBATION_SEED
    assert first.perturbation_count > 0
    assert candidate.classification != "METRIC_3D_CANDIDATE"


def _locked_result() -> RealPitchRegistrationResult:
    return RealPitchRegistrationResult(
        analysis_id="analysis_api_test",
        status="NOT_ATTEMPTED",
        attempted=False,
        wicket_observation_source="/static/wicket.json",
        ambiguity_score=0.0,
        failure_reasons=["insufficient_evidence"],
        diagnostics=RegistrationDiagnostics(),
        message="not attempted",
    )


def test_registration_service_cannot_unlock_legacy_physics() -> None:
    source = Path(
        "services/api/services/real_pitch_registration_service.py"
    ).read_text(encoding="utf-8")
    assert "delivery_physics_service" not in source
    assert "video_camera_pose_service" not in source
    assert "\"accepted\"" not in source
    assert "METRIC_3D_READY" not in source


def test_result_schema_round_trip_is_strict_and_metric_locked() -> None:
    payload = _locked_result()
    encoded = json.dumps(payload.model_dump(mode="json"), sort_keys=True)
    restored = RealPitchRegistrationResult.model_validate_json(encoded)
    assert restored == payload
    with pytest.raises(ValueError):
        RealPitchRegistrationResult.model_validate(
            payload.model_dump(mode="json") | {"metrics_locked": False}
        )
