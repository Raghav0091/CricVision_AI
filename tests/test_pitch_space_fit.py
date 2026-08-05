from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from services.api.schemas.pitch_space_analysis import StableWicketBox
from services.api.services.two_wicket_pitch_fit_service import fit_two_wicket_pitch
from services.api.services.virtual_pitch_service import build_virtual_pitch_specification


def _box(role: str, x: float, bottom: float, width: float, height: float, support: int = 4) -> StableWicketBox:
    return StableWicketBox(
        perspective_role=role,
        x=x,
        y=bottom - height,
        width=width,
        height=height,
        confidence=0.9,
        frame_support=support,
        supporting_frame_indices=[0, 5, 10, 15][:support],
        centre_spread_px=1,
        size_spread_ratio=0.02,
        clipped=False,
        source="persisted",
    )


def _fit(width: int = 1280, height: int = 720):
    near = _box("NEAR", width * 0.38, height * 0.88, width * 0.20, height * 0.18)
    far = _box("FAR", width * 0.48, height * 0.48, width * 0.04, height * 0.07)
    return fit_two_wicket_pitch(near, far, image_width=width, image_height=height)


@pytest.mark.parametrize("size", [(1280, 720), (478, 850)])
def test_landscape_and_portrait_fit_is_finite_and_invertible(size: tuple[int, int]) -> None:
    result = _fit(*size)
    assert result.status == "READY"
    image_to_pitch = np.asarray(result.image_to_pitch_homography)
    pitch_to_image = np.asarray(result.pitch_to_image_homography)
    assert np.isfinite(image_to_pitch).all()
    assert image_to_pitch @ pitch_to_image == pytest.approx(np.eye(3), abs=1e-6)
    assert result.projected_pitch_area_px2 > 10


def test_four_correspondences_round_trip_exactly() -> None:
    result = _fit()
    image_to_pitch = np.asarray(result.image_to_pitch_homography)
    points = np.asarray(
        [[item.image_point.x, item.image_point.y] for item in result.correspondences],
        dtype=np.float64,
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, image_to_pitch).reshape(-1, 2)
    expected = np.asarray([[item.pitch_point.x_m, item.pitch_point.y_m] for item in result.correspondences])
    assert mapped == pytest.approx(expected, abs=1e-5)
    assert result.reprojection_rmse_px < 1e-3


def test_fit_uses_virtual_pitch_v1_dimensions() -> None:
    result = _fit()
    specification = build_virtual_pitch_specification()
    ys = {item.pitch_point.y_m for item in result.correspondences}
    xs = {abs(item.pitch_point.x_m) for item in result.correspondences}
    assert ys == {0.0, specification.dimensions.pitch_length_m}
    assert xs == {specification.dimensions.wicket_width_m / 2}


def test_orientation_hypothesis_is_explicit_and_deterministic() -> None:
    first = _fit()
    second = _fit()
    assert first.selected_hypothesis == second.selected_hypothesis
    assert first.near_semantic_end == "bowler"
    assert first.image_left_is_pitch_left is True
    assert "Box-only evidence cannot independently resolve semantic end" in " ".join(first.warnings)


def test_invalid_or_collapsed_boxes_fail_honestly() -> None:
    near = _box("NEAR", 400, 600, 100, 100)
    far = _box("FAR", 400, 600, 100, 100)
    result = fit_two_wicket_pitch(near, far, image_width=1280, image_height=720)
    assert result.status == "PITCH_FIT_FAILED"
    assert result.image_to_pitch_homography is None


def test_missing_wicket_fails_without_manual_fallback() -> None:
    result = fit_two_wicket_pitch(None, _box("FAR", 500, 350, 40, 50), image_width=1280, image_height=720)
    assert result.status == "PITCH_FIT_FAILED"
    assert result.confidence == 0


def test_projected_geometry_reuses_canonical_ids() -> None:
    result = _fit()
    ids = {item.primitive_id for item in result.projected_pitch}
    assert "pitch_surface" in ids
    assert "pitch_centerline" in ids
    assert "bowler_wicket_ground_extent" in ids
    assert "striker_wicket_ground_extent" in ids


def test_ground_homography_does_not_claim_vertical_stump_projection() -> None:
    result = _fit()
    assert not any("stump_top" in item.primitive_id for item in result.projected_pitch)
    assert "airborne height" in " ".join(result.warnings)


def test_no_pitch_dimensions_are_duplicated_in_fit_source() -> None:
    source = Path("services/api/services/two_wicket_pitch_fit_service.py").read_text(encoding="utf-8")
    assert "20.12" not in source
    assert "3.05" not in source
    assert "build_virtual_pitch_specification" in source

