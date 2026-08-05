from __future__ import annotations

import cv2
import numpy as np
import pytest

from services.api.services.wicket_landmark_frame_service import (
    CropToNativeTransform,
    WicketCrop,
)
from services.api.services.wicket_roi_alignment import (
    align_crop_to_reference,
    align_wicket_crops,
    aligned_point_to_native,
    moving_native_point_to_aligned,
)


def _image() -> np.ndarray:
    image = np.zeros((96, 80, 3), dtype=np.uint8)
    cv2.line(image, (23, 15), (25, 82), (255, 255, 255), 3)
    cv2.line(image, (42, 14), (44, 83), (180, 220, 255), 2)
    cv2.circle(image, (60, 35), 7, (80, 180, 90), -1)
    return image


def _crop(frame_id: int, image: np.ndarray, quality: float = 0.8) -> WicketCrop:
    return WicketCrop(
        frame_index=frame_id,
        role="near",
        image=image,
        transform=CropToNativeTransform(
            x=100,
            y=200,
            width=image.shape[1],
            height=image.shape[0],
            native_width=1280,
            native_height=720,
        ),
        requested_box=(100, 200, 180, 296),
        clipping_fraction=0,
        quality_score=quality,
        quality_factors={},
        accepted=True,
    )


def test_exact_translation_recovery() -> None:
    reference = _image()
    moving = cv2.warpAffine(
        reference,
        np.array([[1, 0, 5], [0, 1, -3]], dtype=np.float32),
        (reference.shape[1], reference.shape[0]),
    )
    result = align_crop_to_reference(reference, moving)
    assert result.accepted
    assert result.moving_to_reference[0, 2] == pytest.approx(-5, abs=0.25)
    assert result.moving_to_reference[1, 2] == pytest.approx(3, abs=0.25)
    assert result.residual < 0.04


def test_small_affine_recovery_when_enabled() -> None:
    reference = _image()
    centre = (reference.shape[1] / 2, reference.shape[0] / 2)
    transform = cv2.getRotationMatrix2D(centre, 1.2, 1.01)
    transform[:, 2] += (2.0, -1.0)
    moving = cv2.warpAffine(reference, transform, (80, 96))
    result = align_crop_to_reference(reference, moving, allow_affine=True)
    assert result.method == "ECC_AFFINE"
    assert result.accepted
    assert result.residual < 0.06


def test_excessive_translation_is_rejected() -> None:
    reference = _image()
    moving = cv2.warpAffine(
        reference,
        np.array([[1, 0, 25], [0, 1, 0]], dtype=np.float32),
        (80, 96),
    )
    result = align_crop_to_reference(reference, moving)
    assert not result.accepted
    assert "excessive_translation" in result.rejection_reasons


def test_low_texture_alignment_is_rejected() -> None:
    reference = np.full((96, 80, 3), 127, dtype=np.uint8)
    moving = reference.copy()
    result = align_crop_to_reference(reference, moving)
    assert not result.accepted
    assert "low_alignment_confidence" in result.rejection_reasons


def test_temporal_products_exclude_rejected_frames() -> None:
    reference = _image()
    shifted = cv2.warpAffine(
        reference,
        np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float32),
        (80, 96),
    )
    rejected = _crop(3, reference.copy())
    rejected = WicketCrop(**{**rejected.__dict__, "accepted": False})
    result = align_wicket_crops(
        [_crop(2, shifted, 0.7), rejected, _crop(1, reference, 0.9)]
    )
    assert result.reference_frame_index == 1
    assert result.accepted_frame_ids == (1, 2)
    assert result.rejected_frame_ids == (3,)
    assert result.alignments[-1].method == "REJECTED"
    assert "crop_rejected_before_alignment" in result.alignments[-1].rejection_reasons
    assert result.temporal_median.shape == reference.shape
    assert result.temporal_maximum_edge.shape == reference.shape[:2]
    assert result.stability_map.shape == reference.shape[:2]


def test_aligned_and_crop_coordinates_round_trip_to_native() -> None:
    crop = _crop(1, _image())
    stack = align_wicket_crops([crop])
    native = aligned_point_to_native((12.5, 20.0), crop)
    aligned = moving_native_point_to_aligned(
        native, crop, stack.alignments[0]
    )
    assert native == pytest.approx((112.5, 220.0))
    assert aligned == pytest.approx((12.5, 20.0))
