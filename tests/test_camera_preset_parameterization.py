from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from services.api.schemas.preset_auto_registration import get_camera_setup_preset
from services.api.services.camera_preset_parameterization import (
    PARAMETER_NAMES,
    PresetCameraParameters,
    build_opencv_camera_from_preset_parameters,
    decompose_opencv_camera_to_preset_parameters,
    denormalize_parameters,
    known_camera_diagnostic,
    normalize_parameters,
    pack_parameters,
    round_trip_projection_diagnostic,
    unpack_parameters,
    validate_rotation_matrix,
)
from services.api.services.virtual_pitch_service import build_virtual_pitch_specification


@pytest.mark.parametrize("camera_end", ["bowler", "striker"])
@pytest.mark.parametrize(
    "mapping", ["IMAGE_LEFT_IS_PITCH_LEFT", "IMAGE_LEFT_IS_PITCH_RIGHT"]
)
@pytest.mark.parametrize(
    "parameters",
    [
        PresetCameraParameters(0.0, 8.0, 1.5, 0.0, -4.0, 0.0, 45.0),
        PresetCameraParameters(1.4, 5.5, 2.2, 8.0, -11.0, 5.0, 61.0, 17.5, -9.25),
        PresetCameraParameters(-1.1, 12.0, 2.8, -12.0, 6.0, -7.0, 33.0, -21.0, 14.0, 1.08, 0.75),
    ],
)
def test_camera_parameter_round_trip_is_numerically_exact(camera_end, mapping, parameters):
    original = build_opencv_camera_from_preset_parameters(
        parameters, image_width=1280, image_height=720,
        camera_end=camera_end, image_left_mapping=mapping,
    )
    decomposition = decompose_opencv_camera_to_preset_parameters(
        camera_matrix=original.camera_matrix,
        rotation_matrix=original.rotation_matrix,
        translation_vector=original.translation_vector,
        image_width=1280, image_height=720,
        camera_end=camera_end, image_left_mapping=mapping,
    )
    rebuilt = build_opencv_camera_from_preset_parameters(
        decomposition.parameters, image_width=1280, image_height=720,
        camera_end=camera_end, image_left_mapping=mapping,
    )
    np.testing.assert_allclose(pack_parameters(decomposition.parameters), pack_parameters(parameters), atol=1e-10)
    np.testing.assert_allclose(rebuilt.camera_matrix, original.camera_matrix, atol=1e-10)
    np.testing.assert_allclose(rebuilt.rotation_matrix, original.rotation_matrix, atol=1e-10)
    np.testing.assert_allclose(rebuilt.translation_vector, original.translation_vector, atol=1e-10)
    diagnostic = round_trip_projection_diagnostic(original, rebuilt)
    assert diagnostic["point_count"] == len(build_virtual_pitch_specification().landmarks) == 36
    assert diagnostic["projection_maximum_error_px"] < 1e-6
    assert diagnostic["camera_position_difference_m"] < 1e-8
    assert diagnostic["positive_depth_mismatch_count"] == 0
    assert not diagnostic["mirror_warning"]
    assert not diagnostic["bowler_striker_reversal_warning"]


def test_rotation_vector_input_and_intrinsic_diagnostics_are_supported():
    parameters = PresetCameraParameters(0.4, 7.0, 1.8, 3.0, -5.0, 2.0, 52.0, 11.0, -7.0, 1.12, 1.5)
    camera = build_opencv_camera_from_preset_parameters(
        parameters, image_width=1000, image_height=800,
        camera_end="bowler", image_left_mapping="image_left_to_world_left",
    )
    result = decompose_opencv_camera_to_preset_parameters(
        camera_matrix=camera.camera_matrix,
        rotation_vector=camera.rotation_vector,
        translation_vector=camera.translation_vector,
        image_width=1000, image_height=800,
        camera_end="bowler", image_left_mapping="image_left_to_world_left",
    )
    assert result.diagnostics["unequal_focal_lengths"] is True
    assert result.parameters.focal_y_over_x == pytest.approx(1.12)
    assert result.parameters.skew_px == pytest.approx(1.5)
    assert result.parameters.principal_point_offset_x_px == pytest.approx(11.0)
    assert result.parameters.principal_point_offset_y_px == pytest.approx(-7.0)


@pytest.mark.parametrize(
    "rotation",
    [np.diag([1.0, 1.0, -1.0]), np.asarray([[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])],
)
def test_invalid_rotations_are_rejected(rotation):
    with pytest.raises(ValueError, match="orthonormal|determinant"):
        validate_rotation_matrix(rotation)


def test_parameter_packing_and_normalized_variables_round_trip():
    parameters = PresetCameraParameters(0.2, 8.0, 1.7, 2.0, -4.0, 1.0, 48.0, 3.0, -2.0, 1.03, 0.4)
    packed = pack_parameters(parameters)
    lower = packed - np.arange(1, len(PARAMETER_NAMES) + 1)
    upper = packed + np.arange(1, len(PARAMETER_NAMES) + 1)
    normalized = normalize_parameters(packed, lower, upper)
    np.testing.assert_allclose(normalized, 0.0)
    np.testing.assert_allclose(denormalize_parameters(normalized, lower, upper), packed)
    assert unpack_parameters(packed) == parameters


def test_known_assisted_camera_round_trip_and_bounds():
    source = Path("outputs/video_analysis/analysis_20260728_120858_762989/reports/real_pitch_registration_v1_refined.json")
    if not source.exists():
        pytest.skip("Persisted assisted-camera fixture is not available.")
    preset = get_camera_setup_preset("STANDARD_REAR_WICKET_NET_V1")
    diagnostic = known_camera_diagnostic(
        source, camera_end="bowler",
        image_left_mapping="IMAGE_LEFT_IS_PITCH_LEFT", preset=preset,
    )
    assert diagnostic["candidate_id"] == "A:image_left_to_world_left:fov_45"
    assert diagnostic["camera_position_world"] == pytest.approx([0.052879489363866, -7.737436770932896, 1.390531725580209])
    expected = {
        "lateral_offset_m": 0.052879489363866,
        "distance_behind_wicket_m": 7.737436770932896,
        "camera_height_m": 1.390531725580209,
        "yaw_deg": -0.4301474771469393,
        "pitch_deg": -1.3284435479232324,
        "roll_deg": 0.389549446369039,
    }
    for name, value in expected.items():
        assert diagnostic["parameters"][name] == pytest.approx(value, abs=1e-10)
    assert diagnostic["reported_horizontal_fov_deg"] == 45.0
    assert diagnostic["effective_horizontal_fov_deg"] == pytest.approx(21.960861554515162)
    assert diagnostic["reported_fov_matches_camera_matrix"] is False
    outside = [item["parameter_name"] for item in diagnostic["parameter_bounds"] if not item["inside_bounds"]]
    assert outside == ["horizontal_fov_deg"]

    payload = json.loads(source.read_text(encoding="utf-8"))
    candidate = payload["selected_candidate"]
    rebuilt = build_opencv_camera_from_preset_parameters(
        PresetCameraParameters(**diagnostic["parameters"]),
        image_width=720,
        image_height=1280,
        camera_end="bowler",
        image_left_mapping="IMAGE_LEFT_IS_PITCH_LEFT",
    )
    expected_k = np.asarray(diagnostic["camera_matrix"])
    np.testing.assert_allclose(rebuilt.camera_matrix, expected_k, atol=1e-10)
    np.testing.assert_allclose(rebuilt.rotation_matrix, candidate["rotation_matrix"], atol=1e-10)
    np.testing.assert_allclose(rebuilt.translation_vector, candidate["translation_vector"], atol=1e-10)


def test_wrong_end_decomposition_exposes_longitudinal_reversal():
    camera = build_opencv_camera_from_preset_parameters(
        PresetCameraParameters(0.0, 8.0, 1.5, 0.0, -4.0, 0.0, 45.0),
        image_width=1280, image_height=720,
        camera_end="bowler", image_left_mapping="IMAGE_LEFT_IS_PITCH_LEFT",
    )
    wrong = decompose_opencv_camera_to_preset_parameters(
        camera_matrix=camera.camera_matrix, rotation_matrix=camera.rotation_matrix,
        translation_vector=camera.translation_vector, image_width=1280, image_height=720,
        camera_end="striker", image_left_mapping="IMAGE_LEFT_IS_PITCH_RIGHT",
    )
    assert wrong.parameters.distance_behind_wicket_m < 0
    assert abs(wrong.parameters.yaw_deg) == pytest.approx(180.0)
