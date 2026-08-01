from __future__ import annotations

import cv2
import numpy as np
import pytest

from services.api.schemas.wicket_observation import PixelBox, SetupFrameCandidate
from services.api.services.wicket_landmark_frame_service import (
    NativeFrame,
    apply_orientation_once,
    balance_role_support,
    extract_wicket_roi,
    map_detector_box_to_native,
    rank_native_frames,
    write_analysis_debug_crops,
)


def _candidate(
    frame_index: int,
    *,
    confidence: float = 0.8,
    stability: float = 0.8,
    obstruction: float = 0.1,
) -> SetupFrameCandidate:
    return SetupFrameCandidate(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 25,
        image_width=160,
        image_height=120,
        score=0.8,
        sharpness=100,
        brightness=120,
        wicket_detection_count=2,
        mean_detector_confidence=confidence,
        detection_stability=stability,
        obstruction_score=obstruction,
        selected=False,
    )


def _sharp_frame() -> np.ndarray:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, ::4] = 255
    image[::5, :] = 180
    return image


def _native(frame: np.ndarray, index: int = 1) -> NativeFrame:
    candidate = _candidate(index)
    selected = rank_native_frames([(frame, candidate)], maximum=1)[0]
    return selected


def test_resized_detector_box_maps_to_native_landscape() -> None:
    result = map_detector_box_to_native(
        PixelBox(x=32, y=18, width=64, height=36),
        detector_width=320,
        detector_height=180,
        native_width=1280,
        native_height=720,
    )
    assert result.model_dump() == {"x": 128.0, "y": 72.0, "width": 256.0, "height": 144.0}


def test_portrait_rotation_maps_box_and_orientation_once() -> None:
    frame = np.arange(4 * 6, dtype=np.uint8).reshape(4, 6)
    rotated = apply_orientation_once(frame, 90)
    assert rotated.shape == (6, 4)
    result = map_detector_box_to_native(
        PixelBox(x=1, y=1, width=2, height=1),
        detector_width=6,
        detector_height=4,
        native_width=6,
        native_height=4,
        rotation_degrees=90,
    )
    assert result.model_dump() == {"x": 2.0, "y": 1.0, "width": 1.0, "height": 2.0}
    with pytest.raises(ValueError, match="already"):
        apply_orientation_once(frame, 90, orientation_already_applied=True)


def test_sharp_frame_selected_and_blurred_frame_rejected() -> None:
    sharp = _sharp_frame()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    ranked = rank_native_frames(
        [(blurred, _candidate(2)), (sharp, _candidate(7))], maximum=2
    )
    assert [item.frame_index for item in ranked] == [7]


def test_deterministic_order_uses_earlier_frame_for_equal_quality() -> None:
    frame = _sharp_frame()
    ranked = rank_native_frames(
        [(frame.copy(), _candidate(8)), (frame.copy(), _candidate(3))], maximum=2
    )
    assert [item.frame_index for item in ranked] == [3, 8]


def test_selection_reserves_three_frames_for_each_persisted_role() -> None:
    frame = _sharp_frame()
    ranked = rank_native_frames(
        [(frame.copy(), _candidate(index, confidence=0.95 - index * 0.01)) for index in range(9)],
        maximum=9,
    )
    selected = balance_role_support(
        ranked,
        near_support=(6, 7, 8),
        far_support=(0, 1, 2, 3, 4, 5),
        maximum=6,
    )
    ids = {item.frame_index for item in selected}
    assert {6, 7, 8}.issubset(ids)
    assert len(ids.intersection({0, 1, 2, 3, 4, 5})) == 3


def test_occlusion_downgrades_frame_score() -> None:
    frame = _sharp_frame()
    clear = rank_native_frames([(frame, _candidate(1, obstruction=0.1))])[0]
    obscured = rank_native_frames([(frame, _candidate(2, obstruction=0.7))])[0]
    assert clear.quality_score > obscured.quality_score


def test_role_specific_padding_and_crop_to_native_round_trip() -> None:
    frame = _native(_sharp_frame())
    bbox = PixelBox(x=60, y=35, width=24, height=48)
    near = extract_wicket_roi(frame, bbox, role="near")
    far = extract_wicket_roi(frame, bbox, role="far")
    assert near.image.shape[0] > far.image.shape[0]
    assert near.image.shape[1] > far.image.shape[1]
    point = (7.25, 11.5)
    native = near.transform.crop_to_native(point)
    assert near.transform.native_to_crop(native) == pytest.approx(point)
    assert near.accepted


def test_heavily_clipped_crop_is_rejected() -> None:
    frame = _native(_sharp_frame())
    crop = extract_wicket_roi(
        frame,
        PixelBox(x=0, y=0, width=45, height=70),
        role="near",
    )
    assert not crop.accepted
    assert "severe_frame_bound_clipping" in crop.rejection_reasons


def test_tiny_far_crop_is_rejected_without_resizing() -> None:
    frame = _native(_sharp_frame())
    crop = extract_wicket_roi(
        frame,
        PixelBox(x=80, y=50, width=4, height=8),
        role="far",
    )
    assert crop.image.shape[:2] != (28, 18)
    assert "native_roi_resolution_insufficient" in crop.rejection_reasons


def test_debug_writer_rejects_non_analysis_identifier() -> None:
    with pytest.raises(ValueError, match="invalid analysis ID"):
        write_analysis_debug_crops("../../outside", [])
