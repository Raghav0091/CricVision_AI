"""Bounded temporal alignment for native-coordinate wicket crops."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Sequence

import cv2
import numpy as np

from .wicket_landmark_frame_service import WicketCrop


AlignmentMethod = Literal[
    "REFERENCE", "PHASE_TRANSLATION", "ECC_AFFINE", "REJECTED"
]


@dataclass(frozen=True)
class CropAlignment:
    frame_index: int
    method: AlignmentMethod
    moving_to_reference: np.ndarray = field(repr=False, compare=False)
    aligned_image: np.ndarray = field(repr=False, compare=False)
    residual: float
    confidence: float
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemporalAlignedStack:
    reference_frame_index: int
    reference_crop: WicketCrop
    alignments: tuple[CropAlignment, ...]
    accepted_frame_ids: tuple[int, ...]
    rejected_frame_ids: tuple[int, ...]
    temporal_median: np.ndarray = field(repr=False, compare=False)
    temporal_maximum_edge: np.ndarray = field(repr=False, compare=False)
    stability_map: np.ndarray = field(repr=False, compare=False)


def _gray_float(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return np.asarray(gray, dtype=np.float32)


def _warp(image: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(
        image,
        matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _residual(reference: np.ndarray, aligned: np.ndarray) -> float:
    first, second = _gray_float(reference), _gray_float(aligned)
    return float(np.mean(np.abs(first - second)) / 255.0)


def _validate_transform(
    matrix: np.ndarray,
    *,
    width: int,
    height: int,
    maximum_translation_fraction: float,
    maximum_rotation_degrees: float,
    maximum_scale_change: float,
) -> tuple[str, ...]:
    linear = matrix[:, :2]
    sx = math.hypot(float(linear[0, 0]), float(linear[1, 0]))
    sy = math.hypot(float(linear[0, 1]), float(linear[1, 1]))
    rotation = math.degrees(math.atan2(float(linear[1, 0]), float(linear[0, 0])))
    tx, ty = abs(float(matrix[0, 2])), abs(float(matrix[1, 2]))
    reasons: list[str] = []
    if tx > width * maximum_translation_fraction or ty > height * maximum_translation_fraction:
        reasons.append("excessive_translation")
    if abs(rotation) > maximum_rotation_degrees:
        reasons.append("excessive_rotation")
    if abs(sx - 1.0) > maximum_scale_change or abs(sy - 1.0) > maximum_scale_change:
        reasons.append("excessive_scale_change")
    if abs(float(np.linalg.det(linear))) < 0.5:
        reasons.append("degenerate_alignment")
    return tuple(reasons)


def align_crop_to_reference(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    allow_affine: bool = False,
    minimum_confidence: float = 0.12,
    maximum_translation_fraction: float = 0.18,
    maximum_rotation_degrees: float = 2.5,
    maximum_scale_change: float = 0.035,
) -> CropAlignment:
    """Align equal-sized crops using phase translation and optional bounded ECC."""
    if reference.shape != moving.shape or reference.size == 0:
        raise ValueError("reference and moving crops must be non-empty and equal-sized")
    height, width = reference.shape[:2]
    ref_gray, moving_gray = _gray_float(reference), _gray_float(moving)
    texture_insufficient = (
        float(ref_gray.std()) < 2.0
        or float(moving_gray.std()) < 2.0
        or float(cv2.Laplacian(ref_gray, cv2.CV_32F).var()) < 1.0
        or float(cv2.Laplacian(moving_gray, cv2.CV_32F).var()) < 1.0
    )
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(ref_gray, moving_gray, window)
    phase_matrix = np.array(
        [[1.0, 0.0, -shift[0]], [0.0, 1.0, -shift[1]]], dtype=np.float32
    )
    matrix = phase_matrix
    method: AlignmentMethod = "PHASE_TRANSLATION"
    confidence = max(0.0, min(1.0, float(response)))
    if allow_affine:
        template_to_input = np.array(
            [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32
        )
        try:
            ecc, template_to_input = cv2.findTransformECC(
                ref_gray / 255.0,
                moving_gray / 255.0,
                template_to_input,
                cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6),
                None,
                3,
            )
            matrix = cv2.invertAffineTransform(template_to_input)
            method = "ECC_AFFINE"
            confidence = max(0.0, min(1.0, float(ecc)))
        except cv2.error:
            pass
    reasons = list(
        _validate_transform(
            matrix,
            width=width,
            height=height,
            maximum_translation_fraction=maximum_translation_fraction,
            maximum_rotation_degrees=maximum_rotation_degrees,
            maximum_scale_change=maximum_scale_change,
        )
    )
    if texture_insufficient:
        reasons.append("low_alignment_confidence")
    if confidence < minimum_confidence:
        reasons.append("low_alignment_confidence")
    aligned = _warp(moving, matrix, (width, height))
    residual = _residual(reference, aligned)
    if residual > 0.34:
        reasons.append("high_alignment_residual")
    return CropAlignment(
        frame_index=-1,
        method=method,
        moving_to_reference=matrix,
        aligned_image=aligned,
        residual=round(residual, 6),
        confidence=round(confidence, 6),
        accepted=not reasons,
        rejection_reasons=tuple(sorted(set(reasons))),
    )


def _identity(crop: WicketCrop) -> CropAlignment:
    return CropAlignment(
        frame_index=crop.frame_index,
        method="REFERENCE",
        moving_to_reference=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        aligned_image=crop.image.copy(),
        residual=0.0,
        confidence=1.0,
        accepted=True,
    )


def _rejected(crop: WicketCrop, reason: str) -> CropAlignment:
    reasons = tuple(sorted(set((*crop.rejection_reasons, reason))))
    return CropAlignment(
        frame_index=crop.frame_index,
        method="REJECTED",
        moving_to_reference=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        ),
        aligned_image=crop.image.copy(),
        residual=1.0,
        confidence=0.0,
        accepted=False,
        rejection_reasons=reasons,
    )


def align_wicket_crops(
    crops: Sequence[WicketCrop],
    *,
    allow_affine: bool = False,
) -> TemporalAlignedStack:
    """Align accepted crops to the best deterministic reference and fuse them."""
    usable = [item for item in crops if item.accepted]
    if not usable:
        raise ValueError("at least one accepted wicket crop is required")
    reference = min(usable, key=lambda item: (-item.quality_score, item.frame_index))
    alignments: list[CropAlignment] = []
    for crop in sorted(crops, key=lambda item: item.frame_index):
        if not crop.accepted:
            alignment = _rejected(crop, "crop_rejected_before_alignment")
        elif crop.image.shape != reference.image.shape:
            alignment = _rejected(crop, "crop_shape_mismatch")
        elif crop.frame_index == reference.frame_index:
            alignment = _identity(crop)
        else:
            result = align_crop_to_reference(
                reference.image, crop.image, allow_affine=allow_affine
            )
            alignment = CropAlignment(
                frame_index=crop.frame_index,
                method=result.method,
                moving_to_reference=result.moving_to_reference,
                aligned_image=result.aligned_image,
                residual=result.residual,
                confidence=result.confidence,
                accepted=result.accepted,
                rejection_reasons=result.rejection_reasons,
            )
        alignments.append(alignment)
    accepted = [item for item in alignments if item.accepted]
    stack = np.stack([item.aligned_image for item in accepted], axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)
    edges = [cv2.Canny(_gray_float(item).astype(np.uint8), 45, 135) for item in stack]
    maximum_edge = np.maximum.reduce(edges)
    gray_stack = np.stack([_gray_float(item) for item in stack], axis=0)
    stability = np.clip(255.0 - np.std(gray_stack, axis=0) * 4.0, 0, 255).astype(
        np.uint8
    )
    accepted_ids = tuple(item.frame_index for item in accepted)
    rejected_ids = tuple(
        sorted(
            {item.frame_index for item in crops}
            - set(accepted_ids)
        )
    )
    return TemporalAlignedStack(
        reference_frame_index=reference.frame_index,
        reference_crop=reference,
        alignments=tuple(alignments),
        accepted_frame_ids=accepted_ids,
        rejected_frame_ids=rejected_ids,
        temporal_median=median,
        temporal_maximum_edge=maximum_edge,
        stability_map=stability,
    )


def aligned_point_to_native(
    point: tuple[float, float],
    reference_crop: WicketCrop,
) -> tuple[float, float]:
    """Map a reference-aligned crop point back to native image coordinates."""
    return reference_crop.transform.crop_to_native(point)


def moving_native_point_to_aligned(
    point: tuple[float, float],
    moving_crop: WicketCrop,
    alignment: CropAlignment,
) -> tuple[float, float]:
    """Map a moving-frame native point into reference-aligned crop space."""
    crop_x, crop_y = moving_crop.transform.native_to_crop(point)
    homogeneous = np.array([crop_x, crop_y, 1.0], dtype=np.float64)
    output = alignment.moving_to_reference @ homogeneous
    return float(output[0]), float(output[1])
