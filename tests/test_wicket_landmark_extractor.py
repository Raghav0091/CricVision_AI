from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from services.api.services.wicket_landmark_extractor import (
    extract_aligned_wicket_landmarks,
    extract_wicket_landmarks,
)
from services.api.services.wicket_line_geometry import (
    LineSegment,
    Point2D,
    angular_distance_degrees,
    line_intersection,
    normalized_line_equation,
)


def _wicket(
    *,
    shape: tuple[int, int] = (180, 140),
    xs: tuple[int, int, int] = (38, 70, 102),
    top: bool = True,
    base: bool = True,
    thickness: int = 3,
) -> np.ndarray:
    height, width = shape
    top_y, base_y = int(height * 0.14), int(height * 0.86)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in xs:
        cv2.line(image, (x, top_y), (x, base_y), (245, 245, 245), thickness)
    if top:
        cv2.line(image, (xs[0] - 3, top_y), (xs[-1] + 3, top_y), (245, 245, 245), 2)
    if base:
        cv2.line(image, (xs[0] - 3, base_y), (xs[-1] + 3, base_y), (245, 245, 245), 2)
    return image


def test_line_geometry_is_normalized_and_intersects() -> None:
    vertical = LineSegment(Point2D(10, 0), Point2D(10, 20))
    horizontal = LineSegment(Point2D(0, 7), Point2D(20, 7))
    a, b, _ = normalized_line_equation(vertical)
    assert np.hypot(a, b) == pytest.approx(1.0)
    assert angular_distance_degrees(vertical.angle_degrees, 90) == pytest.approx(0)
    assert line_intersection(vertical, horizontal) == Point2D(10, 7)


def test_near_wicket_extracts_three_physical_axes_and_endpoints() -> None:
    result = extract_wicket_landmarks(_wicket(), role="near", frame_id=12)
    assert result.status == "AVAILABLE"
    assert [item.semantic_id for item in result.axes] == [
        "left_stump_axis", "middle_stump_axis", "right_stump_axis"
    ]
    assert all(item.status == "AVAILABLE" for item in result.axes)
    assert sum(item.status == "AVAILABLE" for item in result.points) == 6
    assert all(item.line is not None for item in result.lines)
    assert result.diagnostics["spacing_ratio"] > 0.8


def test_far_wicket_allows_narrower_separation_with_larger_uncertainty() -> None:
    far = extract_wicket_landmarks(
        _wicket(shape=(80, 52), xs=(20, 25, 30), thickness=1),
        role="far",
        frame_id=3,
    )
    assert far.status == "AVAILABLE"
    assert far.diagnostics["minimum_axis_separation_px"] < 2.0
    assert far.uncertainty_px is not None and far.uncertainty_px >= 0.9


def test_low_resolution_temporal_far_axes_survive_without_transverse_lines() -> None:
    image = np.zeros((66, 42, 3), np.uint8)
    for x in (15, 18, 22):
        cv2.line(image, (x, 10), (x, 58), (245, 245, 245), 1)
    result = extract_wicket_landmarks(
        image,
        role="far",
        frame_id=14,
        supporting_frame_ids=tuple(range(10, 18)),
    )
    assert result.status == "AVAILABLE"
    assert result.diagnostics["group_score"] >= 0.80
    assert (
        result.diagnostics["observed_minimum_axis_separation_px"]
        >= result.diagnostics["minimum_axis_separation_px"]
    )
    assert result.diagnostics["temporal_support_count"] == 8
    assert all(item.status == "AVAILABLE" for item in result.axes)
    assert all(item.status == "UNAVAILABLE" for item in result.points + result.lines)
    assert result.uncertainty_px is not None and result.uncertainty_px >= 1.5


def test_missing_base_is_partial_and_never_invented() -> None:
    result = extract_wicket_landmarks(_wicket(base=False), role="near", frame_id=8)
    assert result.status == "AVAILABLE"
    assert result.evidence_by_id()["base_line"].status == "UNAVAILABLE"
    assert all(
        item.status == "UNAVAILABLE"
        for item in result.points
        if item.semantic_id.endswith("_base")
    )


def test_internal_mat_lines_do_not_fabricate_endpoints_on_temporal_near_crop() -> None:
    # Regression geometry from the strongest 91x126 temporal median crop.
    image = np.zeros((126, 91, 3), np.uint8)
    for x in (22, 43, 66):
        cv2.line(image, (x, 26), (x, 124), (245, 245, 245), 2)
    cv2.line(image, (4, 11), (86, 11), (220, 220, 220), 2)  # false crop/mat top
    cv2.line(image, (3, 74), (87, 74), (220, 220, 220), 2)  # internal mat boundary

    result = extract_wicket_landmarks(
        image,
        role="near",
        frame_id=20,
        native_origin=(316, 731),
        supporting_frame_ids=(18, 19, 20, 21, 22),
    )

    assert result.status == "AVAILABLE"
    assert [item.line.midpoint.x for item in result.axes if item.line] == pytest.approx(
        [337.5, 359.0, 382.0], abs=2.0
    )
    assert all(item.status == "UNAVAILABLE" for item in result.lines)
    assert all(item.status == "UNAVAILABLE" for item in result.points)
    assert result.diagnostics["selected_top_line_y"] is None
    assert result.diagnostics["selected_base_line_y"] is None
    assert "axes_only_transverse_support_unavailable" in result.warnings


def test_horizontal_line_outside_axis_endpoint_zone_is_unavailable() -> None:
    image = _wicket(top=False, base=False)
    cv2.line(image, (10, 5), (130, 5), (255, 255, 255), 3)
    result = extract_wicket_landmarks(
        image,
        role="near",
        frame_id=30,
        supporting_frame_ids=(28, 29, 30, 31),
    )
    assert result.status == "AVAILABLE"
    assert result.evidence_by_id()["bail_line"].status == "UNAVAILABLE"
    assert result.evidence_by_id()["base_line"].status == "UNAVAILABLE"
    assert all(item.status == "UNAVAILABLE" for item in result.points)


def test_blank_and_small_rois_are_explicitly_unavailable() -> None:
    blank = extract_wicket_landmarks(np.zeros((100, 80, 3), np.uint8), role="near", frame_id=1)
    small = extract_wicket_landmarks(_wicket(shape=(30, 20), xs=(5, 10, 15)), role="near", frame_id=2)
    assert blank.status == small.status == "UNAVAILABLE"
    assert all(item.line is None for item in blank.axes)
    assert all(item.point is None for item in small.points)


def test_net_poles_without_bails_or_ground_support_are_rejected() -> None:
    poles = _wicket(top=False, base=False)
    result = extract_wicket_landmarks(poles, role="near", frame_id=5)
    assert result.status == "UNAVAILABLE"
    assert "axes_lack_wicket_top_or_base_support" in result.warnings


def test_unrelated_diagonal_lines_do_not_become_stump_axes() -> None:
    image = np.zeros((180, 140, 3), np.uint8)
    cv2.line(image, (5, 170), (130, 20), (255, 255, 255), 4)
    cv2.line(image, (8, 20), (125, 160), (255, 255, 255), 4)
    result = extract_wicket_landmarks(image, role="near", frame_id=9)
    assert result.status == "UNAVAILABLE"
    assert all(item.status == "UNAVAILABLE" for item in result.axes)


def test_dense_vertical_player_or_net_clutter_is_rejected() -> None:
    image = np.zeros((180, 180, 3), np.uint8)
    for x in range(20, 161, 12):
        cv2.line(image, (x, 15), (x, 165), (255, 255, 255), 1)
    cv2.line(image, (10, 15), (170, 15), (255, 255, 255), 2)
    cv2.line(image, (10, 165), (170, 165), (255, 255, 255), 2)
    result = extract_wicket_landmarks(image, role="near", frame_id=10)
    assert result.status == "UNAVAILABLE"
    assert result.diagnostics["vertical_cluster_count"] > 10
    assert "vertical_clutter_or_player_occlusion" in result.warnings


@dataclass
class _AlignedCrop:
    consensus_image: np.ndarray
    role: str
    frame_index: int
    native_roi: dict[str, float]


def test_aligned_crop_adapter_maps_evidence_to_native_coordinates() -> None:
    source = _AlignedCrop(_wicket(), "near", 44, {"x": 300.0, "y": 120.0})
    result = extract_aligned_wicket_landmarks(source)
    assert result.frame_id == 44
    assert min(item.line.midpoint.x for item in result.axes if item.line) > 300
    assert min(item.line.midpoint.y for item in result.axes if item.line) > 120


def test_mapping_adapter_accepts_numpy_consensus_image() -> None:
    result = extract_aligned_wicket_landmarks({
        "consensus_image": _wicket(),
        "wicket_role": "near",
        "frame_id": 6,
        "native_origin": (10, 20),
    })
    assert result.status == "AVAILABLE"
    assert result.as_mapping()["axes"][0]["normalized_line_equation"] is not None
    contract = result.axes[0].as_contract_mapping()
    assert contract["semantic_type"] == "LINE"
    assert contract["start_x_px"] is not None


def test_adapter_accepts_numpy_directly_and_never_uses_roi_edges_as_axes() -> None:
    image = _wicket()
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), (255, 255, 255), 2)
    result = extract_aligned_wicket_landmarks(image, role="near", frame_id=7)
    assert result.status == "AVAILABLE"
    assert all(
        image.shape[1] * 0.04 < item.line.midpoint.x < image.shape[1] * 0.96
        for item in result.axes
        if item.line is not None
    )
