from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from packages.cricket_vision.calibration.cricket_pitch_geometry import (
    BOWLING_CREASE_LENGTH_M,
    CREASE_LINE_WIDTH_M,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
    POPPING_CREASE_OFFSET_M,
    RETURN_CREASE_OFFSET_M,
    STUMP_DIAMETER_MAX_M,
    STUMP_DIAMETER_MIN_M,
    STUMP_HEIGHT_M,
    WICKET_WIDTH_M,
    CricketPitchDimensions,
    calibration_to_canonical_world,
    canonical_to_calibration_world,
)
from services.api.main import app
from services.api.schemas.virtual_pitch import PixelPoint2D
from services.api.services.video_analysis_service import (
    VideoAnalysisServiceError,
)
from services.api.services.virtual_pitch_service import (
    build_synthetic_camera,
    build_synthetic_preview,
    build_virtual_pitch_specification,
    map_native_pixel_to_contained_display,
    project_virtual_pitch,
    projected_landmark_observations,
    recover_synthetic_camera_pose,
    synthetic_camera_names,
)


def landmark_map():
    return {
        item.semantic_id: item
        for item in build_virtual_pitch_specification().landmarks
    }


def line_map():
    return {
        item.primitive_id: item
        for item in build_virtual_pitch_specification().line_segments
    }


def test_official_dimensions_have_one_exact_metric_contract() -> None:
    dimensions = build_virtual_pitch_specification().dimensions

    assert dimensions.pitch_length_m == PITCH_LENGTH_M == 20.12
    assert dimensions.pitch_width_m == PITCH_WIDTH_M == 3.05
    assert dimensions.wicket_width_m == WICKET_WIDTH_M == 0.2286
    assert dimensions.stump_height_m == STUMP_HEIGHT_M == 0.7112
    assert dimensions.stump_diameter_min_m == STUMP_DIAMETER_MIN_M == 0.035
    assert dimensions.stump_diameter_max_m == STUMP_DIAMETER_MAX_M == 0.0381
    assert dimensions.bowling_crease_length_m == BOWLING_CREASE_LENGTH_M == 2.64
    assert dimensions.popping_crease_offset_m == POPPING_CREASE_OFFSET_M == 1.22
    assert dimensions.return_crease_offset_m == RETURN_CREASE_OFFSET_M == 1.32
    assert CREASE_LINE_WIDTH_M == 0.0254


def test_coordinate_system_is_right_handed_and_camera_neutral() -> None:
    coordinate = build_virtual_pitch_specification().coordinate_system

    assert coordinate.handedness == "right_handed"
    assert coordinate.origin == "bowler_end_middle_stump_base"
    assert coordinate.x_axis == "lateral_camera_neutral_right"
    assert coordinate.y_axis == "bowler_to_striker"
    assert coordinate.z_axis == "up"
    assert coordinate.off_leg_assignment == "not_assigned"


def test_legacy_calibration_adapter_round_trips_canonical_world() -> None:
    canonical = (0.42, 13.6, 1.2)

    legacy = canonical_to_calibration_world(*canonical)
    restored = calibration_to_canonical_world(*legacy)

    assert legacy == (13.6, 0.42, 1.2)
    assert restored == canonical


def test_wickets_share_one_specification_and_correct_spacing() -> None:
    landmarks = landmark_map()
    outer_spacing = WICKET_WIDTH_M - STUMP_DIAMETER_MAX_M

    for end, y in (("bowler", 0.0), ("striker", PITCH_LENGTH_M)):
        left = landmarks[f"{end}_left_stump_base"].point
        middle = landmarks[f"{end}_middle_stump_base"].point
        right = landmarks[f"{end}_right_stump_base"].point
        assert middle.y == y
        assert right.x - left.x == pytest.approx(outer_spacing)
        assert middle.x - left.x == pytest.approx(outer_spacing / 2)
        assert right.x - middle.x == pytest.approx(outer_spacing / 2)
        assert landmarks[f"{end}_middle_stump_top"].point.z == STUMP_HEIGHT_M

    assert (
        landmarks["striker_middle_stump_base"].point.y
        - landmarks["bowler_middle_stump_base"].point.y
        == PITCH_LENGTH_M
    )


def test_pitch_corners_centerline_and_lbw_corridor() -> None:
    specification = build_virtual_pitch_specification()
    landmarks = landmark_map()
    corridor = next(
        item
        for item in specification.polygons
        if item.primitive_id == "lbw_stump_to_stump_corridor"
    )

    assert landmarks["bowler_left_pitch_corner"].point.x == -PITCH_WIDTH_M / 2
    assert landmarks["striker_right_pitch_corner"].point.x == PITCH_WIDTH_M / 2
    assert landmarks["pitch_centerline_bowler_endpoint"].point.y == 0
    assert landmarks["pitch_centerline_striker_endpoint"].point.y == PITCH_LENGTH_M
    assert {point.x for point in corridor.vertices} == {
        -WICKET_WIDTH_M / 2,
        WICKET_WIDTH_M / 2,
    }
    assert {point.y for point in corridor.vertices} == {0.0, PITCH_LENGTH_M}


def test_both_crease_systems_are_generated_from_offsets() -> None:
    lines = line_map()

    for end, wicket_y, popping_y in (
        ("bowler", 0.0, POPPING_CREASE_OFFSET_M),
        ("striker", PITCH_LENGTH_M, PITCH_LENGTH_M - POPPING_CREASE_OFFSET_M),
    ):
        bowling = lines[f"{end}_bowling_crease"]
        popping = lines[f"{end}_popping_crease"]
        assert bowling.start.y == bowling.end_point.y == wicket_y
        assert bowling.end_point.x - bowling.start.x == BOWLING_CREASE_LENGTH_M
        assert popping.start.y == popping.end_point.y == popping_y
        assert {popping.start.x, popping.end_point.x} == {
            -RETURN_CREASE_OFFSET_M,
            RETURN_CREASE_OFFSET_M,
        }
        for side in ("left", "right"):
            return_line = lines[
                f"{end}_{side}_return_crease_registration_span"
            ]
            assert abs(return_line.start.x) == RETURN_CREASE_OFFSET_M
            assert {return_line.start.y, return_line.end_point.y} == {
                wicket_y,
                popping_y,
            }


def test_landmark_and_primitive_ids_are_stable_and_unique() -> None:
    specification = build_virtual_pitch_specification()
    landmark_ids = [item.semantic_id for item in specification.landmarks]
    primitive_ids = [
        *[item.primitive_id for item in specification.stumps],
        *[item.primitive_id for item in specification.bails],
        *[item.primitive_id for item in specification.line_segments],
        *[item.primitive_id for item in specification.polygons],
    ]

    assert len(landmark_ids) == len(set(landmark_ids))
    assert len(primitive_ids) == len(set(primitive_ids))
    assert all(item.description for item in specification.landmarks)
    assert all(
        item.calibration_anchor
        for item in specification.landmarks
        if "stump" in item.semantic_id and "center" not in item.semantic_id
    )


def test_specification_defaults_to_regulation() -> None:
    """Regression guard for every caller that builds without an argument.

    The builder now reads a ``CricketPitchDimensions`` instead of the module
    constants, so this derives the expected geometry from the constants to
    catch any drift between the two.
    """
    specification = build_virtual_pitch_specification()
    dimensions = specification.dimensions

    assert dimensions.pitch_length_m == PITCH_LENGTH_M == 20.12
    assert specification == build_virtual_pitch_specification(
        CricketPitchDimensions()
    )

    outer_stump_x = (WICKET_WIDTH_M - STUMP_DIAMETER_MAX_M) / 2
    half_pitch = PITCH_WIDTH_M / 2
    half_bowling_crease = BOWLING_CREASE_LENGTH_M / 2
    expected: dict[str, tuple[float, float, float]] = {}
    for end, y, direction in (
        ("bowler", 0.0, 1.0),
        ("striker", PITCH_LENGTH_M, -1.0),
    ):
        popping_y = y + direction * POPPING_CREASE_OFFSET_M
        expected[f"{end}_wicket_center_base"] = (0.0, y, 0.0)
        for side, x in (
            ("left", -outer_stump_x),
            ("middle", 0.0),
            ("right", outer_stump_x),
        ):
            expected[f"{end}_{side}_stump_base"] = (x, y, 0.0)
            expected[f"{end}_{side}_stump_top"] = (x, y, STUMP_HEIGHT_M)
        expected[f"{end}_left_pitch_corner"] = (-half_pitch, y, 0.0)
        expected[f"{end}_right_pitch_corner"] = (half_pitch, y, 0.0)
        expected[f"{end}_bowling_crease_left_endpoint"] = (
            -half_bowling_crease,
            y,
            0.0,
        )
        expected[f"{end}_bowling_crease_right_endpoint"] = (
            half_bowling_crease,
            y,
            0.0,
        )
        expected[f"{end}_popping_crease_left_endpoint"] = (
            -RETURN_CREASE_OFFSET_M,
            popping_y,
            0.0,
        )
        expected[f"{end}_popping_crease_right_endpoint"] = (
            RETURN_CREASE_OFFSET_M,
            popping_y,
            0.0,
        )
        expected[f"{end}_left_return_bowling_intersection"] = (
            -RETURN_CREASE_OFFSET_M,
            y,
            0.0,
        )
        expected[f"{end}_right_return_bowling_intersection"] = (
            RETURN_CREASE_OFFSET_M,
            y,
            0.0,
        )
        expected[f"{end}_left_return_popping_intersection"] = (
            -RETURN_CREASE_OFFSET_M,
            popping_y,
            0.0,
        )
        expected[f"{end}_right_return_popping_intersection"] = (
            RETURN_CREASE_OFFSET_M,
            popping_y,
            0.0,
        )
        expected[f"pitch_centerline_{end}_endpoint"] = (0.0, y, 0.0)

    landmarks = landmark_map()
    assert set(expected).issubset(set(landmarks))
    for semantic_id, (x, y, z) in expected.items():
        point = landmarks[semantic_id].point
        assert (point.x, point.y, point.z) == (x, y, z), semantic_id


def test_specification_honours_declared_length() -> None:
    specification = build_virtual_pitch_specification(
        CricketPitchDimensions(pitch_length_m=4.0, popping_crease_distance_m=1.0)
    )
    landmarks = {
        item.semantic_id: item.point for item in specification.landmarks
    }

    assert specification.dimensions.pitch_length_m == 4.0
    assert landmarks["striker_wicket_center_base"].y == 4.0
    assert landmarks["striker_middle_stump_base"].y == 4.0
    assert landmarks["bowler_middle_stump_base"].y == 0.0
    # The striker popping crease is measured back from the striker wicket.
    assert landmarks["striker_popping_crease_left_endpoint"].y == 3.0
    assert landmarks["bowler_popping_crease_left_endpoint"].y == 1.0
    assert all(
        item.centre.y in {0.0, 4.0} for item in specification.stumps
    )


def test_specification_honours_declared_wicket_size() -> None:
    specification = build_virtual_pitch_specification(
        CricketPitchDimensions(
            pitch_length_m=4.0,
            wicket_width_m=0.12,
            wicket_height_m=0.4,
            stump_diameter_m=0.02,
            popping_crease_distance_m=1.0,
        )
    )
    landmarks = {
        item.semantic_id: item.point for item in specification.landmarks
    }

    assert specification.dimensions.stump_height_m == 0.4
    assert landmarks["striker_middle_stump_top"].z == 0.4
    assert landmarks["bowler_right_stump_base"].x == (0.12 - 0.02) / 2
    # min must never exceed max once a thinner stump is declared.
    assert specification.dimensions.stump_diameter_min_m == 0.02
    assert specification.dimensions.stump_diameter_max_m == 0.02


@pytest.mark.parametrize(
    "dimensions",
    [
        CricketPitchDimensions(pitch_length_m=0.0),
        CricketPitchDimensions(pitch_length_m=-4.0),
        # A popping crease at or beyond half the pitch length is degenerate.
        CricketPitchDimensions(
            pitch_length_m=2.0, popping_crease_distance_m=1.22
        ),
        CricketPitchDimensions(wicket_width_m=0.0),
        CricketPitchDimensions(stump_diameter_m=0.2),
    ],
)
def test_invalid_dimensions_rejected(
    dimensions: CricketPitchDimensions,
) -> None:
    with pytest.raises(VideoAnalysisServiceError) as excinfo:
        build_virtual_pitch_specification(dimensions)

    assert excinfo.value.status_code == 422


def test_cache_keyed_on_dimensions() -> None:
    """A cache that ignored the argument would silently return regulation."""
    short = build_virtual_pitch_specification(
        CricketPitchDimensions(pitch_length_m=4.0, popping_crease_distance_m=1.0)
    )
    regulation = build_virtual_pitch_specification()
    short_again = build_virtual_pitch_specification(
        CricketPitchDimensions(pitch_length_m=4.0, popping_crease_distance_m=1.0)
    )

    assert short.dimensions.pitch_length_m == 4.0
    assert regulation.dimensions.pitch_length_m == PITCH_LENGTH_M
    assert short_again.dimensions.pitch_length_m == 4.0
    assert short != regulation


def test_schema_round_trip_and_json_are_deterministic() -> None:
    specification = build_virtual_pitch_specification()
    first = specification.model_dump_json()
    second = build_virtual_pitch_specification().model_dump_json()

    assert first == second
    restored = type(specification).model_validate_json(first)
    assert restored == specification
    assert json.loads(first)["virtual_pitch_model_version"] == "v1"


@pytest.mark.parametrize("camera_name", synthetic_camera_names())
def test_every_synthetic_camera_projects_the_same_model(camera_name: str) -> None:
    projection = build_synthetic_preview(camera_name).projection

    assert projection.virtual_pitch_model_version == "v1"
    assert projection.source_camera.name == camera_name
    assert projection.diagnostics.valid_landmark_count > 0
    assert projection.diagnostics.behind_camera_count == 0
    assert projection.diagnostics.perspective_order_valid
    assert len(projection.projected_stumps) == 6
    assert len(projection.projected_bails) == 4


def test_near_wicket_appears_larger_for_both_end_views() -> None:
    bowler = build_synthetic_preview("centred_bowler_end").projection.diagnostics
    striker = build_synthetic_preview("centred_striker_end").projection.diagnostics

    assert bowler.nearer_wicket == "bowler"
    assert bowler.bowler_wicket_mean_height_px > bowler.striker_wicket_mean_height_px
    assert striker.nearer_wicket == "striker"
    assert striker.striker_wicket_mean_height_px > striker.bowler_wicket_mean_height_px


def test_narrower_focal_length_projects_larger_wickets() -> None:
    narrow = build_synthetic_preview("narrow_focal_length").projection.diagnostics
    wide = build_synthetic_preview("wide_focal_length").projection.diagnostics

    assert narrow.bowler_wicket_mean_height_px > wide.bowler_wicket_mean_height_px
    assert narrow.striker_wicket_mean_height_px > wide.striker_wicket_mean_height_px


def test_points_behind_camera_are_invalid_not_clamped() -> None:
    camera = build_synthetic_camera("centred_bowler_end").model_copy(
        update={"translation_vector": [0.0, 0.0, -100.0]}
    )

    projection = project_virtual_pitch(camera)

    assert projection.diagnostics.behind_camera_count > 0
    assert any(
        item.pixel_point is None and not item.projection_valid
        for item in projection.projected_landmarks
    )


def test_out_of_frame_geometry_is_marked_safely() -> None:
    projection = build_synthetic_preview("elevated_camera").projection

    assert projection.diagnostics.out_of_frame_count > 0
    assert any(
        item.projection_valid and not item.in_frame
        for item in projection.projected_landmarks
    )


def test_exact_synthetic_pnp_round_trip() -> None:
    camera = build_synthetic_camera("centred_bowler_end")
    observations = projected_landmark_observations(project_virtual_pitch(camera))

    result = recover_synthetic_camera_pose(camera, observations)

    assert result.success
    assert result.rotation_error_degrees < 0.001
    assert result.translation_error_m < 0.001
    assert result.reprojection_rmse_px < 0.01
    assert result.inlier_count >= 12


def test_noisy_synthetic_pnp_recovery_stays_within_tolerance() -> None:
    camera = build_synthetic_camera("left_of_centre")
    observations = projected_landmark_observations(project_virtual_pitch(camera))
    noisy = [
        item.model_copy(
            update={
                "pixel_point": PixelPoint2D(
                    x=item.pixel_point.x + ((index % 5) - 2) * 0.25,
                    y=item.pixel_point.y + ((index % 3) - 1) * 0.20,
                )
            }
        )
        for index, item in enumerate(observations)
    ]

    result = recover_synthetic_camera_pose(camera, noisy)

    assert result.success
    assert result.rotation_error_degrees < 0.05
    assert result.translation_error_m < 0.05
    assert result.reprojection_rmse_px < 0.75


def test_synthetic_pnp_rejects_bad_correspondence_as_outlier() -> None:
    camera = build_synthetic_camera("right_of_centre")
    observations = projected_landmark_observations(project_virtual_pitch(camera))
    bad = list(observations)
    bad[7] = bad[7].model_copy(
        update={
            "pixel_point": PixelPoint2D(
                x=bad[7].pixel_point.x + 120,
                y=bad[7].pixel_point.y - 100,
            )
        }
    )

    result = recover_synthetic_camera_pose(camera, bad)

    assert result.success
    assert bad[7].semantic_id in result.outlier_landmark_ids
    assert result.reprojection_rmse_px < 0.1


def test_synthetic_pnp_fails_for_insufficient_and_degenerate_anchors() -> None:
    camera = build_synthetic_camera("centred_bowler_end")
    projection = project_virtual_pitch(camera)
    observations = projected_landmark_observations(
        projection,
        calibration_anchors_only=False,
    )
    by_id = {item.semantic_id: item for item in observations}
    degenerate_ids = [
        "bowler_wicket_center_base",
        "bowler_middle_stump_base",
        "pitch_centerline_bowler_endpoint",
        "striker_wicket_center_base",
        "striker_middle_stump_base",
        "pitch_centerline_striker_endpoint",
    ]

    insufficient = recover_synthetic_camera_pose(camera, observations[:4])
    degenerate = recover_synthetic_camera_pose(
        camera,
        [by_id[semantic_id] for semantic_id in degenerate_ids],
    )

    assert insufficient.failure_reason == "insufficient_correspondences"
    assert degenerate.failure_reason == "degenerate_world_geometry"


def test_object_contain_mapping_handles_landscape_and_portrait_letterboxing() -> None:
    landscape = map_native_pixel_to_contained_display(
        640,
        360,
        native_width=1280,
        native_height=720,
        container_width=390,
        container_height=400,
    )
    portrait = map_native_pixel_to_contained_display(
        360,
        640,
        native_width=720,
        native_height=1280,
        container_width=600,
        container_height=400,
    )

    assert landscape == pytest.approx((195, 200))
    assert portrait == pytest.approx((300, 200))


def test_virtual_pitch_api_is_lightweight_and_projection_is_developer_only() -> None:
    client = TestClient(app)

    specification_response = client.get("/video-analysis/virtual-pitch")
    projection_response = client.get(
        "/video-analysis/virtual-pitch/synthetic-projection",
        params={"camera_name": "portrait_bowler_end"},
    )

    assert specification_response.status_code == 200
    assert specification_response.json()["virtual_pitch_model_version"] == "v1"
    assert projection_response.status_code == 200
    payload = projection_response.json()
    assert payload["developer_only"] is True
    assert payload["registration_status"] == "not_registered_to_video"
    assert payload["projection"]["source_camera"]["image_width"] == 720
    assert "metric analytics remain locked" in payload["message"]


def test_unknown_synthetic_camera_is_rejected() -> None:
    client = TestClient(app)

    response = client.get(
        "/video-analysis/virtual-pitch/synthetic-projection",
        params={"camera_name": "not_a_camera"},
    )

    assert response.status_code == 400
