"""Virtual Replay coordinate contract and geometry validation regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    CALIBRATION_V2_WORLD_ORDER,
    CRICVISION_PITCH_V1,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    calibration_to_canonical_world,
    canonical_to_calibration_world,
)
from services.api.schemas.delivery_physics import (
    CameraCalibration,
    DeliveryPhysicsResult,
    GeometryValidationResult,
    ObservedBallPoint,
    TrajectorySample,
)
from services.api.schemas.replay_payload import ReplayPayloadV1
from services.api.services.delivery_physics_service import (
    _calibration_world_from_physics,
    _project_physics_world,
    _world_at,
    fit_metric_trajectory,
    validate_metric_geometry,
)
from services.api.services.replay_payload_service import assemble_replay_payload


def _synthetic_calibration(
    *,
    world_order: str = CALIBRATION_V2_WORLD_ORDER,
) -> CameraCalibration:
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
        world_coordinate_system=world_order,
        image_width=1280,
        image_height=720,
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=[0.0] * 5,
        rotation_vector=rotation_vector.reshape(-1).tolist(),
        rotation_matrix=rotation.tolist(),
        translation_vector=translation.tolist(),
        calibration_confidence=0.9,
    )


def _synthetic_observations(
    parameters: np.ndarray,
    model_name: str,
    calibration: CameraCalibration,
    *,
    count: int = 10,
) -> list[ObservedBallPoint]:
    times = np.arange(count, dtype=np.float64) * 0.04
    pixels = _project_physics_world(
        _world_at(parameters, model_name, times),
        calibration,
    )
    return [
        ObservedBallPoint(
            frame_index=index,
            timestamp_seconds=float(timestamp),
            pixel_x=float(pixel[0]),
            pixel_y=float(pixel[1]),
            detector_confidence=0.9,
            tracker_confidence=0.9,
            source="OBSERVED",
        )
        for index, (timestamp, pixel) in enumerate(zip(times, pixels))
    ]


def test_canonical_axis_definitions_match_replay_contract() -> None:
    lateral, longitudinal, _ = calibration_to_canonical_world(4.0, -0.2, 0.0)
    assert lateral == pytest.approx(-0.2)
    assert longitudinal == pytest.approx(4.0)
    back = canonical_to_calibration_world(lateral, longitudinal, 0.0)
    assert back == pytest.approx((4.0, -0.2, 0.0))


def test_calibration_v2_adapter_swaps_axes_for_legacy_pose() -> None:
    world = np.array([[0.2, 8.0, 1.5]])
    adapted = _calibration_world_from_physics(
        world,
        _synthetic_calibration(world_order=CALIBRATION_V2_WORLD_ORDER),
    )
    assert adapted[0].tolist() == pytest.approx([8.0, 0.2, 1.5])


def test_cricvision_pitch_v1_projection_does_not_swap_axes() -> None:
    world = np.array([[0.2, 8.0, 1.5]])
    adapted = _calibration_world_from_physics(
        world,
        _synthetic_calibration(world_order=CRICVISION_PITCH_V1),
    )
    assert adapted[0].tolist() == pytest.approx([0.2, 8.0, 1.5])


def test_image_to_world_to_image_round_trip_for_canonical_pose() -> None:
    calibration = _synthetic_calibration(world_order=CALIBRATION_V2_WORLD_ORDER)
    parameters = np.array([0.1, 2.0, 1.8, 0.5, 28.0, 2.0])
    observations = _synthetic_observations(parameters, "BALLISTIC", calibration)
    fit = fit_metric_trajectory(observations, calibration)
    assert fit.rmse_px < 1.0

    samples = [
        TrajectorySample(
            frame_index=obs.frame_index,
            timestamp_seconds=obs.timestamp_seconds,
            world_x_m=float(world[0]),
            world_y_m=float(world[1]),
            world_z_m=float(world[2]),
            pixel_x=obs.pixel_x,
            pixel_y=obs.pixel_y,
            provenance="OBSERVED",
            confidence=0.9,
        )
        for obs, world in zip(
            observations,
            _world_at(
                fit.parameters,
                fit.model_name,
                np.array(
                    [item.timestamp_seconds - observations[0].timestamp_seconds for item in observations]
                ),
            ),
        )
    ]
    validation = validate_metric_geometry(samples, observations, calibration)
    assert validation.validity == "VALID_METRIC_3D"
    assert validation.mean_reprojection_px is not None
    assert validation.mean_reprojection_px < 2.0


def test_lateral_coordinates_remain_inside_pitch_corridor() -> None:
    calibration = _synthetic_calibration(world_order=CALIBRATION_V2_WORLD_ORDER)
    parameters = np.array([0.15, 3.0, 1.6, 0.2, 26.0, 1.5])
    observations = _synthetic_observations(parameters, "BALLISTIC", calibration, count=12)
    fit = fit_metric_trajectory(observations, calibration)
    half = PITCH_WIDTH_M / 2
    for elapsed in np.linspace(0.0, 0.35, 8):
        x, y, _ = _world_at(fit.parameters, fit.model_name, elapsed).reshape(3)
        assert abs(x) <= half + 0.2
        assert 0.0 <= y <= PITCH_LENGTH_M + 0.2


def test_axis_swap_produces_outside_pitch_geometry() -> None:
    calibration = _synthetic_calibration(world_order=CALIBRATION_V2_WORLD_ORDER)
    wrong_calibration = calibration.model_copy(
        update={"world_coordinate_system": CRICVISION_PITCH_V1},
    )
    parameters = np.array([0.1, 2.0, 1.8, 0.5, 28.0, 2.0])
    observations = _synthetic_observations(parameters, "BALLISTIC", calibration)
    fit = fit_metric_trajectory(observations, wrong_calibration)
    samples = [
        TrajectorySample(
            frame_index=obs.frame_index,
            timestamp_seconds=obs.timestamp_seconds,
            world_x_m=float(world[0]),
            world_y_m=float(world[1]),
            world_z_m=float(world[2]),
            pixel_x=obs.pixel_x,
            pixel_y=obs.pixel_y,
            provenance="OBSERVED",
            confidence=0.9,
        )
        for obs, world in zip(
            observations,
            _world_at(
                fit.parameters,
                fit.model_name,
                np.array(
                    [item.timestamp_seconds - observations[0].timestamp_seconds for item in observations]
                ),
            ),
        )
    ]
    validation = validate_metric_geometry(samples, observations, wrong_calibration)
    assert validation.validity in {"INVALID_REPROJECTION", "OUTSIDE_PITCH_GEOMETRY"}


def test_high_reprojection_cannot_be_valid_metric_3d() -> None:
    calibration = _synthetic_calibration(world_order=CALIBRATION_V2_WORLD_ORDER)
    observations = _synthetic_observations(
        np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0]),
        "BALLISTIC",
        calibration,
    )
    samples = [
        TrajectorySample(
            frame_index=obs.frame_index,
            timestamp_seconds=obs.timestamp_seconds,
            world_x_m=12.0,
            world_y_m=12.0,
            world_z_m=2.0,
            pixel_x=obs.pixel_x,
            pixel_y=obs.pixel_y,
            provenance="OBSERVED",
            confidence=0.9,
        )
        for obs in observations
    ]
    validation = validate_metric_geometry(samples, observations, calibration)
    assert validation.validity != "VALID_METRIC_3D"
    assert validation.p95_reprojection_px is not None
    assert validation.p95_reprojection_px > (validation.threshold_px or 0.0)


def test_replay_payload_uses_finalized_track_id_when_present() -> None:
    from services.api.services import replay_payload_service as module

    physics = DeliveryPhysicsResult.model_validate(
        {
            "status": "SUCCESS",
            "analysis_id": "analysis_20260803_213801_b1c2d3",
            "coordinate_system": "canonical",
            "calibration": {
                "mode": "IMAGE_SPACE_ONLY",
                "confidence": "LOW",
                "image_width": 720,
                "image_height": 1280,
            },
            "geometry_validation": {
                "validity": "IMAGE_SPACE_ONLY",
                "reason": "test",
            },
            "delivery_interval": {"terminal_reason": "test"},
            "bounce": {"status": "INSUFFICIENT_EVIDENCE", "confidence": "LOW", "confidence_score": 0.0},
            "speed": {"confidence": "INSUFFICIENT_EVIDENCE"},
            "pre_bounce_lateral_movement": {"direction": "unavailable", "confidence": "INSUFFICIENT_EVIDENCE"},
            "post_bounce_movement": {"status": "UNAVAILABLE", "confidence": "INSUFFICIENT_EVIDENCE"},
            "line_and_length": {"line": "unavailable", "length": "unavailable"},
            "fit_diagnostics": {"converged": True, "selected_model": "none", "optimizer_status": "test"},
            "confidence": "LOW",
            "confidence_score": 0.1,
            "uncertainty_method": "test",
        }
    )
    payload = module._build_diagnostics(
        measurement_validity="IMAGE_SPACE_ONLY",
        physics=physics,
        tracking_status="ready",
        trajectory=[],
        bridge_response=None,
        camera=module.ReplayCamera(source="UNAVAILABLE", visualization_only=False),
        source_track_id="tracking_job_abc123",
        tracking_job_id="tracking_job_abc123",
        consistency_errors=[],
    )
    assert payload.source_track_id == "tracking_job_abc123"
    assert payload.tracking_job_id == "tracking_job_abc123"
    assert payload.source_track_id != "rebuild_test"


def test_world_points_map_to_scene_inside_pitch_frustum() -> None:
    # Mirror replayCoordinates.ts: Three x=lateral, y=height, z=-longitudinal
    lateral, longitudinal, height = 0.1, 8.0, 1.2
    scene_x, scene_y, scene_z = lateral, height, -longitudinal
    half = PITCH_WIDTH_M / 2
    assert abs(scene_x) <= half + 0.01
    assert 0.0 <= -scene_z <= PITCH_LENGTH_M + 0.01
    assert scene_y == pytest.approx(1.2)


def test_image_space_only_payload_has_no_world_positions() -> None:
    payload = ReplayPayloadV1.model_validate(
        {
            **{
                field: None
                for field in ()
            },
            "analysis_id": "analysis_image_only",
            "measurement_validity": "IMAGE_SPACE_ONLY",
            "camera": {"source": "UNAVAILABLE", "visualization_only": False},
            "playback": {},
            "trajectory": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "image_position": {"x": 100.0, "y": 200.0},
                    "provenance": "OBSERVED",
                    "confidence": 0.8,
                }
            ],
            "bounce": {"status": "UNAVAILABLE", "unavailable_reason": "test"},
            "metrics": {
                "release_speed_kmh": {
                    "value": None,
                    "unit": "km/h",
                    "status": "UNAVAILABLE",
                    "unavailable_reason": "test",
                },
                "average_pre_bounce_speed_kmh": {
                    "value": None,
                    "unit": "km/h",
                    "status": "UNAVAILABLE",
                    "unavailable_reason": "test",
                },
                "speed_at_bounce_kmh": {
                    "value": None,
                    "unit": "km/h",
                    "status": "UNAVAILABLE",
                    "unavailable_reason": "test",
                },
                "overall_stump_to_stump_speed_kmh": {
                    "value": None,
                    "unit": "km/h",
                    "status": "UNAVAILABLE",
                    "unavailable_reason": "test",
                },
                "delivery_length_m": {
                    "value": None,
                    "unit": "m",
                    "status": "UNAVAILABLE",
                    "unavailable_reason": "test",
                },
                "estimated_lateral_deviation_m": {
                    "value": None,
                    "unit": "m",
                    "status": "UNAVAILABLE",
                    "unavailable_reason": "test",
                },
            },
            "diagnostics": {
                "status": "READY",
                "measurement_validity": "IMAGE_SPACE_ONLY",
                "geometry_validity": "INVALID_REPROJECTION",
                "warnings": [],
                "unavailable_reason": "Geometry failed",
            },
        }
    )
    assert payload.trajectory[0].world_position is None
    assert payload.measurement_validity == "IMAGE_SPACE_ONLY"


def test_speed_remains_available_when_geometry_validation_passes() -> None:
    calibration = _synthetic_calibration(world_order=CALIBRATION_V2_WORLD_ORDER)
    parameters = np.array([0.0, 2.0, 1.8, 0.0, 25.0, 1.0])
    observations = _synthetic_observations(parameters, "BALLISTIC", calibration, count=8)
    fit = fit_metric_trajectory(observations, calibration)
    assert fit.parameters[4] == pytest.approx(25.0, abs=1.5)


def test_geometry_validation_result_serializes() -> None:
    result = GeometryValidationResult(
        validity="VALID_METRIC_3D",
        mean_reprojection_px=1.2,
        threshold_px=12.0,
    )
    restored = GeometryValidationResult.model_validate_json(result.model_dump_json())
    assert restored.validity == "VALID_METRIC_3D"
