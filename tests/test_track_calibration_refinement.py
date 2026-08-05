"""Post-track wicket-box calibration refinement for metric 3D replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from packages.cricket_vision.calibration.cricket_pitch_geometry import CRICVISION_PITCH_V1
from services.api.schemas.delivery_physics import (
    BouncePhysicsResult,
    DeliveryInterval,
    DeliveryPhysicsResult,
    FitDiagnostics,
    GeometryValidationResult,
    LateralMovementResult,
    LineLengthResult,
    PostBounceMovementResult,
    SpeedAnalytics,
)
from services.api.schemas.video_analysis import (
    TrackingPoint,
    VideoBallDetectionsDocument,
)
from services.api.schemas.wicket_box_calibration import CalibrationResult
from services.api.services.delivery_physics_service import (
    _geometry_validation_rank,
    refine_metric_calibration_with_track,
)
from services.api.services.wicket_box_calibration_service import (
    ACCEPTED_FILENAME,
    RESULT_FILENAME,
    _intrinsics_from_focal,
    build_camera_calibration_from_pose_candidate,
    list_track_refineable_calibration_candidates,
    persist_pose_candidate_as_accepted,
)
from tests.test_delivery_physics import synthetic_calibration, synthetic_observations
from tests.test_real_pitch_registration import _solve, _synthetic_observation
from tests.test_wicket_box_calibration import (
    ANALYSIS_ID,
    _register_request,
    _synthetic_landmarks,
    isolated_wicket_box,
)


def _tracking_points_from_calibration(
    *,
    count: int = 10,
) -> list[TrackingPoint]:
    parameters = np.array([0.1, 2.0, 1.8, 0.5, 28.0, 2.0])
    observations = synthetic_observations(parameters, "BALLISTIC", count=count)
    return [
        TrackingPoint(
            frame_index=obs.frame_index,
            timestamp_seconds=obs.timestamp_seconds,
            source="observed",
            provenance="OBSERVED",
            candidate_id=obs.candidate_id or f"candidate-{obs.frame_index}",
            x=obs.pixel_x,
            y=obs.pixel_y,
            normalized_x=obs.pixel_x / 1280.0,
            normalized_y=obs.pixel_y / 720.0,
            confidence=0.9,
            vx=100.0,
            vy=75.0,
        )
        for obs in observations
    ]


def _detections_document(track: list[TrackingPoint]) -> VideoBallDetectionsDocument:
    frames = []
    for point in track:
        half = 6.0
        frames.append(
            {
                "frame_index": point.frame_index,
                "timestamp_seconds": point.timestamp_seconds,
                "processed": True,
                "detections": [
                    {
                        "candidate_id": point.candidate_id,
                        "class_id": 0,
                        "class_name": "ball",
                        "confidence": point.confidence,
                        "bbox_xyxy": [
                            point.x - half,
                            point.y - half,
                            point.x + half,
                            point.y + half,
                        ],
                        "bbox_normalized": {
                            "x": point.normalized_x - 0.01,
                            "y": point.normalized_y - 0.01,
                            "width": 0.02,
                            "height": 0.02,
                        },
                        "center": {"x": point.x, "y": point.y},
                        "center_normalized": {
                            "x": point.normalized_x,
                            "y": point.normalized_y,
                        },
                        "width_pixels": half * 2,
                        "height_pixels": half * 2,
                        "area_pixels": (half * 2) ** 2,
                        "inside_pitch_corridor": True,
                    }
                ],
            }
        )
    return VideoBallDetectionsDocument.model_validate(
        {
            "analysis_id": ANALYSIS_ID,
            "model_path_used": "test-model.pt",
            "model_class_names": ["ball"],
            "settings": {
                "frame_stride": 1,
                "imgsz": 960,
                "confidence_threshold": 0.15,
                "max_det": 20,
            },
            "frames": frames,
        }
    )


def test_geometry_validation_rank_prefers_valid_metric_3d() -> None:
    valid = GeometryValidationResult(
        validity="VALID_METRIC_3D",
        median_reprojection_px=8.0,
        in_pitch_fraction=0.95,
    )
    invalid = GeometryValidationResult(
        validity="OUTSIDE_PITCH_GEOMETRY",
        median_reprojection_px=2.0,
        in_pitch_fraction=0.99,
    )
    assert _geometry_validation_rank(valid) > _geometry_validation_rank(invalid)


def test_build_camera_calibration_from_pose_candidate() -> None:
    observation = _synthetic_observation(noise_px=0.2)
    candidate, _ = _solve(observation)
    calibration = build_camera_calibration_from_pose_candidate(candidate, 1280, 720)
    assert calibration.mode == "METRIC_3D"
    assert calibration.camera_matrix is not None
    assert calibration.rotation_vector is not None


def test_refine_returns_none_when_geometry_already_valid() -> None:
    track = _tracking_points_from_calibration()
    detections = _detections_document(track)
    current = DeliveryPhysicsResult(
        status="SUCCESS",
        analysis_id=ANALYSIS_ID,
        coordinate_system="cricvision_pitch_v1",
        calibration=synthetic_calibration(),
        geometry_validation=GeometryValidationResult(
            validity="VALID_METRIC_3D",
            median_reprojection_px=4.0,
            in_pitch_fraction=1.0,
        ),
        delivery_interval=DeliveryInterval(terminal_reason="test"),
        bounce=BouncePhysicsResult(
            status="INSUFFICIENT_EVIDENCE",
            confidence="INSUFFICIENT_EVIDENCE",
            confidence_score=0.0,
        ),
        speed=SpeedAnalytics(confidence="INSUFFICIENT_EVIDENCE"),
        pre_bounce_lateral_movement=LateralMovementResult(
            direction="unavailable",
            confidence="INSUFFICIENT_EVIDENCE",
        ),
        post_bounce_movement=PostBounceMovementResult(
            status="UNAVAILABLE",
            confidence="INSUFFICIENT_EVIDENCE",
        ),
        line_and_length=LineLengthResult(line="unavailable", length="unavailable"),
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model="BALLISTIC",
            optimizer_status="test",
        ),
        confidence="HIGH",
        confidence_score=0.9,
        uncertainty_method="test",
    )
    refined = refine_metric_calibration_with_track(
        analysis_id=ANALYSIS_ID,
        primary_track=track,
        detections=detections,
        tracker_bounce=None,
        fps=30.0,
        width=1280,
        height=720,
        total_frames=120,
        current_physics=current,
    )
    assert refined is None


def test_refine_metric_calibration_with_track_picks_better_candidate(
    isolated_wicket_box: Path,
) -> None:
    good_calibration = synthetic_calibration()
    bad_calibration = good_calibration.model_copy(
        update={"world_coordinate_system": CRICVISION_PITCH_V1},
    )
    bad_candidate = SimpleNamespace(candidate_id="bad")
    good_candidate = SimpleNamespace(candidate_id="good")
    track = _tracking_points_from_calibration()
    detections = _detections_document(track)
    current = DeliveryPhysicsResult(
        status="PARTIAL",
        analysis_id=ANALYSIS_ID,
        coordinate_system="cricvision_pitch_v1",
        calibration=bad_calibration,
        geometry_validation=GeometryValidationResult(
            validity="OUTSIDE_PITCH_GEOMETRY",
            median_reprojection_px=20.0,
            in_pitch_fraction=0.2,
            reason="seed bad geometry",
        ),
        delivery_interval=DeliveryInterval(terminal_reason="test"),
        bounce=BouncePhysicsResult(
            status="INSUFFICIENT_EVIDENCE",
            confidence="INSUFFICIENT_EVIDENCE",
            confidence_score=0.0,
        ),
        speed=SpeedAnalytics(confidence="INSUFFICIENT_EVIDENCE"),
        pre_bounce_lateral_movement=LateralMovementResult(
            direction="unavailable",
            confidence="INSUFFICIENT_EVIDENCE",
        ),
        post_bounce_movement=PostBounceMovementResult(
            status="UNAVAILABLE",
            confidence="INSUFFICIENT_EVIDENCE",
        ),
        line_and_length=LineLengthResult(line="unavailable", length="unavailable"),
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model="BALLISTIC",
            optimizer_status="test",
        ),
        confidence="LOW",
        confidence_score=0.4,
        uncertainty_method="test",
    )

    stored = CalibrationResult(
        status="ACCEPTED",
        analysis_id=ANALYSIS_ID,
        calibration_frame_index=4,
        source_image_width=1280,
        source_image_height=720,
        stump_landmarks=_synthetic_landmarks(),
        near_wicket_box=_register_request().near_wicket_box,
        far_wicket_box=_register_request().far_wicket_box,
        message="seed",
    )
    result_path = isolated_wicket_box / ANALYSIS_ID / "reports" / RESULT_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(stored.model_dump_json(), encoding="utf-8")

    def _calibration_for_candidate(candidate, width, height):
        if candidate.candidate_id == "good":
            return good_calibration
        return bad_calibration

    def _physics_for_calibration(*, calibration, **kwargs):
        if calibration.world_coordinate_system == CRICVISION_PITCH_V1:
            return current
        return current.model_copy(
            update={
                "calibration": good_calibration,
                "geometry_validation": GeometryValidationResult(
                    validity="VALID_METRIC_3D",
                    median_reprojection_px=3.0,
                    in_pitch_fraction=1.0,
                ),
            }
        )

    observation = _synthetic_observation(noise_px=0.2)
    persist_candidate, _ = _solve(observation)

    with patch(
        "services.api.services.wicket_box_calibration_service.list_track_refineable_calibration_candidates",
        return_value=[bad_candidate, good_candidate],
    ), patch(
        "services.api.services.wicket_box_calibration_service.build_camera_calibration_from_pose_candidate",
        side_effect=_calibration_for_candidate,
    ), patch(
        "services.api.services.delivery_physics_service._analyse_metric_3d",
        side_effect=_physics_for_calibration,
    ), patch(
        "services.api.services.wicket_box_calibration_service.persist_pose_candidate_as_accepted",
        side_effect=lambda analysis_id, candidate, stored: persist_pose_candidate_as_accepted(
            analysis_id,
            persist_candidate,
            stored,
        ),
    ):
        refined = refine_metric_calibration_with_track(
            analysis_id=ANALYSIS_ID,
            primary_track=track,
            detections=detections,
            tracker_bounce=None,
            fps=30.0,
            width=1280,
            height=720,
            total_frames=120,
            current_physics=current,
        )

    assert refined is not None
    assert refined.geometry_validation is not None
    assert refined.geometry_validation.validity == "VALID_METRIC_3D"
    accepted_path = isolated_wicket_box / ANALYSIS_ID / "reports" / ACCEPTED_FILENAME
    assert accepted_path.is_file()


def test_intrinsics_from_focal_is_bounded() -> None:
    intrinsics = _intrinsics_from_focal(1539.95, 1280, 720)
    assert intrinsics.focal_length_x_px == pytest.approx(1539.95)
    assert intrinsics.lower_focal_bound_px == pytest.approx(1280 * 0.35)
    assert intrinsics.upper_focal_bound_px == pytest.approx(1280 * 3.5)


def test_list_track_refineable_calibration_candidates(
    isolated_wicket_box: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    landmarks = _synthetic_landmarks()
    request = _register_request(landmarks=landmarks)
    stored = CalibrationResult(
        status="ACCEPTED",
        analysis_id=ANALYSIS_ID,
        calibration_frame_index=request.calibration_frame_index,
        source_image_width=request.source_image_width,
        source_image_height=request.source_image_height,
        near_wicket_box=request.near_wicket_box,
        far_wicket_box=request.far_wicket_box,
        stump_landmarks=landmarks,
        message="seed",
    )
    result_path = isolated_wicket_box / ANALYSIS_ID / "reports" / RESULT_FILENAME
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(stored.model_dump_json(), encoding="utf-8")

    candidates = list_track_refineable_calibration_candidates(ANALYSIS_ID)
    assert candidates
    assert all(item.solver_success for item in candidates)


def test_persist_pose_candidate_as_accepted_writes_snapshot(
    isolated_wicket_box: Path,
) -> None:
    observation = _synthetic_observation(noise_px=0.2)
    candidate, _ = _solve(observation)
    stored = CalibrationResult(
        status="REGISTERED",
        analysis_id=ANALYSIS_ID,
        calibration_frame_index=4,
        source_image_width=1280,
        source_image_height=720,
        stump_landmarks=_synthetic_landmarks(),
        near_wicket_box=_register_request().near_wicket_box,
        far_wicket_box=_register_request().far_wicket_box,
        message="registered",
    )
    persist_pose_candidate_as_accepted(ANALYSIS_ID, candidate, stored)
    accepted_path = isolated_wicket_box / ANALYSIS_ID / "reports" / ACCEPTED_FILENAME
    assert accepted_path.is_file()
    payload = json.loads(accepted_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "wicket_box_calibration_accepted_v1"
    assert payload["reprojection_rmse_px"] == candidate.reprojection_rmse_px
