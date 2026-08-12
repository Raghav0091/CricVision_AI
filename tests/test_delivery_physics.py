from __future__ import annotations

import cv2
import numpy as np
import pytest

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CALIBRATION_V2_WORLD_ORDER,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    CricketPitchDimensions,
)
from services.api.schemas.delivery_physics import (
    BouncePhysicsResult,
    CameraCalibration,
    ObservedBallPoint,
    PhysicsPitchGeometry,
    TrajectorySample,
)
from services.api.schemas.video_analysis import (
    PrimaryBounceResult,
    TrackingPoint,
    VideoBallDetectionsDocument,
)
from services.api.services.delivery_physics_service import (
    _analyse_non_3d,
    _overall_stump_to_stump_speed,
    _pitch_length_m,
    _world_point_in_pitch,
    _fit_post_bounce,
    _line_and_length,
    _metric_bounce,
    _metric_samples,
    _post_bounce_movement,
    _project_physics_world,
    _speed_analytics,
    _world_at,
    canonical_observations,
    failed_physics_result,
    fit_metric_trajectory,
    insufficient_physics_result,
)
from services.api.services.video_ball_tracking_service import (
    _physics_replay_points,
)


def test_physics_pitch_geometry_uses_authoritative_dimensions() -> None:
    geometry = PhysicsPitchGeometry()

    assert geometry.pitch_length_m == PITCH_LENGTH_M
    assert geometry.pitch_width_m == PITCH_WIDTH_M


def synthetic_calibration() -> CameraCalibration:
    camera_matrix = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]]
    )
    camera_position = np.array([-8.0, 0.0, 6.0])
    target = np.array([10.0, 0.0, 1.0])
    forward = target - camera_position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ camera_position
    rotation_vector, _ = cv2.Rodrigues(rotation)
    return CameraCalibration(
        mode="METRIC_3D",
        confidence="HIGH",
        world_coordinate_system=CALIBRATION_V2_WORLD_ORDER,
        image_width=1280,
        image_height=720,
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=[0.0] * 5,
        rotation_vector=rotation_vector.reshape(-1).tolist(),
        rotation_matrix=rotation.tolist(),
        translation_vector=translation.tolist(),
        projection_matrix=(
            camera_matrix
            @ np.hstack([rotation, translation.reshape(3, 1)])
        ).tolist(),
        correspondences_used=12,
        reprojection_error_px=0.8,
        calibration_confidence=0.9,
    )


def synthetic_observations(
    parameters: np.ndarray,
    model_name: str,
    *,
    count: int = 10,
    interval: float = 0.04,
    noise_px: float = 0.0,
) -> list[ObservedBallPoint]:
    calibration = synthetic_calibration()
    times = np.arange(count, dtype=np.float64) * interval
    pixels = _project_physics_world(
        _world_at(parameters, model_name, times),
        calibration,
    )
    if noise_px:
        pattern = np.array(
            [[-0.5, 0.25], [0.4, -0.2], [0.1, 0.3], [-0.2, -0.35]]
        )
        pixels = pixels + np.vstack(
            [pattern[index % len(pattern)] for index in range(count)]
        ) * noise_px
    return [
        ObservedBallPoint(
            frame_index=index,
            timestamp_seconds=float(timestamp),
            pixel_x=float(pixel[0]),
            pixel_y=float(pixel[1]),
            detector_confidence=0.9,
            tracker_confidence=0.9,
            source="OBSERVED",
            candidate_id=f"candidate-{index}",
        )
        for index, (timestamp, pixel) in enumerate(zip(times, pixels))
    ]


def test_ballistic_fit_recovers_gravity_constrained_delivery() -> None:
    expected = np.array([0.1, 2.0, 1.8, 0.5, 28.0, 2.0])

    fit = fit_metric_trajectory(
        synthetic_observations(expected, "BALLISTIC"),
        synthetic_calibration(),
    )

    assert fit.model_name == "BALLISTIC"
    assert fit.rmse_px < 0.5
    assert fit.parameters[4] == pytest.approx(28.0, abs=1.0)
    assert fit.parameters[2] == pytest.approx(1.8, abs=0.15)

    repeated = fit_metric_trajectory(
        synthetic_observations(expected, "BALLISTIC"),
        synthetic_calibration(),
    )
    assert repeated.parameters == pytest.approx(fit.parameters, abs=1e-9)


def test_lateral_model_recovers_bounded_effective_acceleration() -> None:
    expected = np.array([0.1, 2.0, 1.8, 0.5, 28.0, 2.0, 4.0])

    fit = fit_metric_trajectory(
        synthetic_observations(expected, "BALLISTIC_LATERAL"),
        synthetic_calibration(),
    )

    assert fit.model_name == "BALLISTIC_LATERAL"
    assert fit.parameters[6] == pytest.approx(4.0, abs=0.8)
    assert abs(fit.parameters[6]) <= 20.0


def test_forward_deceleration_model_remains_non_accelerating() -> None:
    expected = np.array([0.0, 2.0, 3.0, 0.2, 30.0, 5.0, 2.0, -10.0])

    fit = fit_metric_trajectory(
        synthetic_observations(
            expected,
            "BALLISTIC_LATERAL_DECELERATION",
            count=20,
            interval=0.035,
        ),
        synthetic_calibration(),
    )

    assert fit.model_name == "BALLISTIC_LATERAL_DECELERATION"
    assert -15.0 <= fit.parameters[7] <= 0.0
    assert fit.parameters[7] < -0.1


def test_robust_fit_rejects_false_detector_outlier() -> None:
    parameters = np.array([0.0, 2.0, 1.8, 0.3, 27.0, 2.0])
    observations = synthetic_observations(parameters, "BALLISTIC", count=11)
    observations[5] = observations[5].model_copy(
        update={
            "pixel_x": observations[5].pixel_x + 180,
            "pixel_y": observations[5].pixel_y - 130,
        }
    )

    fit = fit_metric_trajectory(observations, synthetic_calibration())

    assert 5 not in fit.inlier_indexes
    assert len(fit.inlier_indexes) == 10
    assert fit.rmse_px < 1.0


def test_known_ballistic_bounce_is_recovered_on_pitch() -> None:
    parameters = np.array([0.0, 5.0, 1.5, 0.0, 20.0, -1.0])
    observations = synthetic_observations(parameters, "BALLISTIC", count=8)

    bounce = _metric_bounce(
        parameters,
        "BALLISTIC",
        synthetic_calibration(),
        observations,
        None,
        25.0,
    )

    assert bounce.status in {"DETECTED", "ESTIMATED"}
    assert bounce.world_y_m is not None
    assert 5.0 < bounce.world_y_m < 20.12
    assert bounce.distance_from_striker_wicket_m == pytest.approx(
        20.12 - bounce.world_y_m,
        abs=1e-6,
    )


def test_line_and_length_use_metric_bounce_coordinates() -> None:
    parameters = np.array([0.05, 11.0, 1.0, 0.0, 18.0, -2.0])
    observations = synthetic_observations(parameters, "BALLISTIC", count=8)
    bounce = _metric_bounce(
        parameters,
        "BALLISTIC",
        synthetic_calibration(),
        observations,
        None,
        25.0,
    )

    result = _line_and_length(bounce)

    assert result.line == "middle"
    assert result.length in {"full", "good length", "short", "very short"}
    assert result.bounce_distance_from_striker_m is not None


def test_metric_samples_mark_gaps_and_terminal_projection_honestly() -> None:
    parameters = np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0])
    observations = synthetic_observations(parameters, "BALLISTIC", count=8)
    accepted = [item for index, item in enumerate(observations) if index != 3]
    bounce = _metric_bounce(
        parameters,
        "BALLISTIC",
        synthetic_calibration(),
        observations,
        None,
        25.0,
    )

    samples, _ = _metric_samples(
        observations=observations,
        accepted=accepted,
        calibration=synthetic_calibration(),
        parameters=parameters,
        model_name="BALLISTIC",
        post_fit=None,
        bounce=bounce,
        fps=25.0,
        total_frames=30,
        fit_rmse=1.0,
    )

    by_frame = {sample.frame_index: sample for sample in samples}
    assert by_frame[0].provenance == "OBSERVED"
    assert by_frame[3].provenance == "RECONSTRUCTED"
    assert any(sample.provenance == "PROJECTED" for sample in samples)
    projected = [
        sample for sample in samples if sample.provenance == "PROJECTED"
    ]
    assert projected[-1].confidence <= projected[0].confidence


def test_speed_uses_fitted_velocity_and_kmh_conversion() -> None:
    samples = [
        TrajectorySample(
            frame_index=index,
            timestamp_seconds=index / 25,
            world_x_m=0,
            world_y_m=index,
            world_z_m=1,
            pixel_x=10,
            pixel_y=10,
            velocity_x_mps=0,
            velocity_y_mps=25,
            velocity_z_mps=0,
            speed_mps=25,
            provenance="OBSERVED",
            confidence=0.9,
        )
        for index in range(6)
    ]
    observations = synthetic_observations(
        np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0]),
        "BALLISTIC",
        count=6,
    )

    result = _speed_analytics(
        samples,
        _metric_bounce(
            np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0]),
            "BALLISTIC",
            synthetic_calibration(),
            observations,
            None,
            25.0,
        ),
        observations,
    )

    assert result.earliest_measured_speed_kmh == pytest.approx(90.0)
    assert result.average_pre_bounce_speed_kmh == pytest.approx(90.0)


def test_image_space_fallback_does_not_fabricate_metric_values() -> None:
    observations = synthetic_observations(
        np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0]),
        "BALLISTIC",
        count=7,
        noise_px=0.4,
    )
    calibration = CameraCalibration(
        mode="IMAGE_SPACE_ONLY",
        confidence="UNAVAILABLE",
        image_width=1280,
        image_height=720,
        failure_reason="No calibration.",
    )

    result = _analyse_non_3d(
        analysis_id="analysis_test",
        observations=observations,
        rejected=[],
        calibration=calibration,
        tracker_bounce=None,
        fps=25.0,
        total_frames=30,
        processing_started=0.0,
    )

    assert result.status == "IMAGE_SPACE_ONLY"
    assert result.speed.earliest_measured_speed_kmh is None
    assert result.bounce.world_y_m is None
    assert result.pre_bounce_lateral_movement.movement_m is None


def test_image_fallback_fits_separate_segments_across_bounce() -> None:
    observations = [
        ObservedBallPoint(
            frame_index=frame,
            timestamp_seconds=frame / 25,
            pixel_x=100 + 8 * frame,
            pixel_y=120 + 2 * frame + 1.5 * frame**2,
            detector_confidence=0.9,
            tracker_confidence=0.9,
            source="OBSERVED",
        )
        for frame in range(5)
    ]
    observations.extend(
        ObservedBallPoint(
            frame_index=frame,
            timestamp_seconds=frame / 25,
            pixel_x=132 + 5 * (frame - 4),
            pixel_y=152 - 10 * (frame - 4) + 1.2 * (frame - 4) ** 2,
            detector_confidence=0.9,
            tracker_confidence=0.9,
            source="OBSERVED",
        )
        for frame in range(5, 10)
    )
    calibration = CameraCalibration(
        mode="IMAGE_SPACE_ONLY",
        confidence="UNAVAILABLE",
        image_width=1280,
        image_height=720,
    )
    tracker_bounce = PrimaryBounceResult(
        bounce_detected=True,
        bounce_frame=4,
        bounce_timestamp_seconds=4 / 25,
        bounce_x=132,
        bounce_y=152,
        confidence=0.8,
    )

    result = _analyse_non_3d(
        analysis_id="analysis_piecewise",
        observations=observations,
        rejected=[],
        calibration=calibration,
        tracker_bounce=tracker_bounce,
        fps=25,
        total_frames=20,
        processing_started=0.0,
    )

    assert "piecewise" in result.fit_diagnostics.optimizer_status
    assert result.fit_diagnostics.weighted_reprojection_rmse_px < 0.1
    assert result.bounce.status == "ESTIMATED"


def test_post_bounce_fit_requires_and_uses_observed_second_segment() -> None:
    calibration = synthetic_calibration()
    pre_parameters = np.array([0.0, 4.0, 1.0, 0.2, 24.0, -1.0])
    roots = np.roots([-4.905, pre_parameters[5], pre_parameters[2]])
    bounce_elapsed = min(float(root.real) for root in roots if root.real > 0)
    bounce_world = _world_at(
        pre_parameters,
        "BALLISTIC",
        bounce_elapsed,
    ).reshape(3)
    post_velocity = np.array([1.4, 18.0, 4.2])
    pre_times = np.arange(0.0, bounce_elapsed, 0.04)
    post_elapsed = np.arange(0.04, 0.25, 0.04)
    pre_world = _world_at(pre_parameters, "BALLISTIC", pre_times)
    post_world = np.column_stack(
        [
            bounce_world[0] + post_velocity[0] * post_elapsed,
            bounce_world[1] + post_velocity[1] * post_elapsed,
            post_velocity[2] * post_elapsed - 4.905 * post_elapsed**2,
        ]
    )
    timestamps = np.concatenate([pre_times, bounce_elapsed + post_elapsed])
    pixels = _project_physics_world(
        np.vstack([pre_world, post_world]),
        calibration,
    )
    observations = [
        ObservedBallPoint(
            frame_index=index,
            timestamp_seconds=float(timestamp),
            pixel_x=float(pixel[0]),
            pixel_y=float(pixel[1]),
            detector_confidence=0.95,
            tracker_confidence=0.95,
            source="OBSERVED",
        )
        for index, (timestamp, pixel) in enumerate(zip(timestamps, pixels))
    ]
    bounce = BouncePhysicsResult(
        status="DETECTED",
        frame_index=int(round(bounce_elapsed * 25)),
        timestamp_seconds=bounce_elapsed,
        world_x_m=float(bounce_world[0]),
        world_y_m=float(bounce_world[1]),
        pixel_x=300,
        pixel_y=300,
        distance_from_striker_wicket_m=20.12 - float(bounce_world[1]),
        lateral_offset_m=float(bounce_world[0]),
        confidence="HIGH",
        confidence_score=0.9,
    )

    post_fit = _fit_post_bounce(
        observations,
        calibration,
        pre_parameters,
        "BALLISTIC",
        bounce,
    )
    movement = _post_bounce_movement(
        pre_parameters,
        "BALLISTIC",
        post_fit,
        bounce,
        observations,
    )

    assert post_fit is not None
    assert post_fit.parameters == pytest.approx(post_velocity, abs=0.05)
    assert movement.status == "MEASURED"
    assert movement.observed_points >= 4


def test_canonical_observations_reject_inconsistent_timestamps() -> None:
    detections = VideoBallDetectionsDocument.model_validate(
        {
            "analysis_id": "analysis_timestamp",
            "model_path_used": "model.pt",
            "model_class_names": ["ball"],
            "settings": {
                "frame_stride": 1,
                "imgsz": 960,
                "confidence_threshold": 0.15,
                "max_det": 20,
            },
            "frames": [],
        }
    )
    points = [
        TrackingPoint(
            frame_index=0,
            timestamp_seconds=0.0,
            source="observed",
            provenance="OBSERVED",
            x=100,
            y=100,
            normalized_x=0.1,
            normalized_y=0.1,
            confidence=0.9,
            vx=0,
            vy=0,
        ),
        TrackingPoint(
            frame_index=1,
            timestamp_seconds=1.0,
            source="observed",
            provenance="OBSERVED",
            x=110,
            y=105,
            normalized_x=0.11,
            normalized_y=0.105,
            confidence=0.9,
            vx=250,
            vy=125,
        ),
    ]

    accepted, rejected = canonical_observations(points, detections, fps=25)

    assert [item.frame_index for item in accepted] == [0]
    assert rejected[0].reason == "invalid_or_inconsistent_timestamp"


def test_failed_result_round_trips_without_affecting_track_contract() -> None:
    result = failed_physics_result(
        "analysis_failure",
        1280,
        720,
        "Synthetic optimizer failure.",
    )

    restored = type(result).model_validate_json(result.model_dump_json())

    assert restored.status == "FAILED"
    assert restored.trajectory_samples == []
    assert "Synthetic optimizer failure." in restored.warnings


def test_physics_samples_adapt_to_existing_replay_provenance() -> None:
    observations = synthetic_observations(
        np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0]),
        "BALLISTIC",
        count=6,
    )
    result = _analyse_non_3d(
        analysis_id="analysis_replay",
        observations=[item for index, item in enumerate(observations) if index != 3],
        rejected=[],
        calibration=CameraCalibration(
            mode="IMAGE_SPACE_ONLY",
            confidence="UNAVAILABLE",
            image_width=1280,
            image_height=720,
        ),
        tracker_bounce=None,
        fps=25,
        total_frames=16,
        processing_started=0.0,
    )

    replay = _physics_replay_points(result, 1280, 720)

    assert any(item.provenance == "OBSERVED" for item in replay)
    assert any(item.provenance == "PHYSICS_RECONSTRUCTED" for item in replay)
    assert any(item.provenance == "PROJECTED" for item in replay)
    assert all(0 <= item.normalized_x <= 1 for item in replay)
    assert all(0 <= item.normalized_y <= 1 for item in replay)


def test_insufficient_evidence_marks_spin_rpm_unavailable() -> None:
    calibration = CameraCalibration(
        mode="IMAGE_SPACE_ONLY",
        confidence="UNAVAILABLE",
        image_width=1280,
        image_height=720,
    )

    result = insufficient_physics_result(
        analysis_id="analysis_test",
        calibration=calibration,
        accepted=[],
        rejected=[],
        reason="Too few points.",
        processing_seconds=0.01,
    )

    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.exact_spin_rpm is None
    assert "not directly observable" in result.exact_spin_rpm_unavailable_reason
    assert result.line_and_length.line == "unavailable"


def _declared_pitch_samples(pitch_length_m: float) -> list[TrajectorySample]:
    """A track spanning exactly the declared pitch at a constant 20 m/s."""
    speed_mps = 20.0
    count = 11
    return [
        TrajectorySample(
            frame_index=index,
            timestamp_seconds=(index * pitch_length_m / (count - 1)) / speed_mps,
            world_x_m=0.0,
            world_y_m=index * pitch_length_m / (count - 1),
            world_z_m=1.0,
            pixel_x=10.0,
            pixel_y=10.0,
            speed_mps=speed_mps,
            provenance="OBSERVED",
            confidence=0.9,
        )
        for index in range(count)
    ]


def test_physics_uses_declared_geometry() -> None:
    """Stump-to-stump speed must span the declared wickets, not 20.12 m."""
    samples = _declared_pitch_samples(4.0)

    declared = _overall_stump_to_stump_speed(samples, 4.0)

    assert declared.status == "MEASURED"
    assert declared.travelled_distance_m == pytest.approx(4.0, abs=1e-6)
    assert declared.speed_mps == pytest.approx(20.0, abs=0.05)
    assert declared.speed_kph == pytest.approx(72.0, abs=0.5)
    assert declared.end_crossing.world_y_m == pytest.approx(4.0, abs=1e-6)


def test_declared_geometry_speed_is_not_computed_against_regulation() -> None:
    """The regulation plane sits past the rig, so it must not silently report."""
    samples = _declared_pitch_samples(4.0)

    regulation = _overall_stump_to_stump_speed(samples, PITCH_LENGTH_M)

    assert regulation.status == "UNAVAILABLE"
    assert regulation.travelled_distance_m is None
    assert regulation.speed_mps is None
    assert regulation.speed_kph is None


def test_calibration_carries_declared_geometry_to_pitch_bounds() -> None:
    """Pitch bounds follow the declared pitch rather than the constant."""
    declared = synthetic_calibration().model_copy(
        update={
            "pitch_geometry": CricketPitchDimensions(
                pitch_length_m=4.0,
                popping_crease_distance_m=1.0,
            )
        }
    )
    regulation = synthetic_calibration()

    assert _pitch_length_m(declared) == 4.0
    assert _pitch_length_m(regulation) == PITCH_LENGTH_M

    # A point 10 m down a 4 m pitch is off the rig, but on a regulation pitch.
    assert not _world_point_in_pitch(0.0, 10.0, 0.5, pitch_length_m=4.0)
    assert _world_point_in_pitch(0.0, 10.0, 0.5, pitch_length_m=PITCH_LENGTH_M)
    assert _world_point_in_pitch(0.0, 3.5, 0.5, pitch_length_m=4.0)
