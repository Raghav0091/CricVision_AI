"""Two-wicket ground-plane fit using the canonical Virtual Pitch V1 model."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..schemas.pitch_space_analysis import (
    ImagePoint,
    PitchFitCorrespondence,
    PitchFitResult,
    PitchPoint,
    ProjectedPitchPrimitive,
    StableWicketBox,
)
from .virtual_pitch_service import build_virtual_pitch_specification


def _transform(points: list[tuple[float, float]], matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(source, matrix).reshape(-1, 2)


def _polygon_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def _projected_geometry(matrix: np.ndarray) -> list[ProjectedPitchPrimitive]:
    specification = build_virtual_pitch_specification()
    output: list[ProjectedPitchPrimitive] = []
    for item in specification.line_segments:
        if abs(item.start.z) > 0.01 or abs(item.end_point.z) > 0.01:
            continue
        projected = _transform(
            [(item.start.x, item.start.y), (item.end_point.x, item.end_point.y)],
            matrix,
        )
        if np.isfinite(projected).all():
            output.append(
                ProjectedPitchPrimitive(
                    primitive_id=item.primitive_id,
                    primitive_type="LINE",
                    image_points=[ImagePoint(x=x, y=y) for x, y in projected],
                )
            )
    for item in specification.polygons:
        if any(abs(point.z) > 0.01 for point in item.vertices):
            continue
        projected = _transform([(point.x, point.y) for point in item.vertices], matrix)
        if np.isfinite(projected).all():
            output.append(
                ProjectedPitchPrimitive(
                    primitive_id=item.primitive_id,
                    primitive_type="POLYGON",
                    image_points=[ImagePoint(x=x, y=y) for x, y in projected],
                )
            )
    dimensions = specification.dimensions
    for end, y in (("bowler", 0.0), ("striker", dimensions.pitch_length_m)):
        projected = _transform(
            [(-dimensions.wicket_width_m / 2, y), (dimensions.wicket_width_m / 2, y)],
            matrix,
        )
        output.append(
            ProjectedPitchPrimitive(
                primitive_id=f"{end}_wicket_ground_extent",
                primitive_type="WICKET_BASE",
                image_points=[ImagePoint(x=x, y=y_value) for x, y_value in projected],
            )
        )
    return output


def fit_two_wicket_pitch(
    near: StableWicketBox | None,
    far: StableWicketBox | None,
    *,
    image_width: int,
    image_height: int,
) -> PitchFitResult:
    if near is None or far is None or image_width <= 0 or image_height <= 0:
        return PitchFitResult(
            status="PITCH_FIT_FAILED",
            fit_score=0,
            confidence=0,
            warnings=["Two valid stabilized wickets and native dimensions are required."],
        )
    specification = build_virtual_pitch_specification()
    length = specification.dimensions.pitch_length_m
    half_wicket = specification.dimensions.wicket_width_m / 2
    image_points = np.float32(
        [
            [near.x, near.y + near.height],
            [near.x + near.width, near.y + near.height],
            [far.x, far.y + far.height],
            [far.x + far.width, far.y + far.height],
        ]
    )
    hypotheses: list[tuple[float, str, str, bool, np.ndarray, np.ndarray, float, float]] = []
    for near_end in ("bowler", "striker"):
        near_y, far_y = ((0.0, length) if near_end == "bowler" else (length, 0.0))
        for image_left_is_pitch_left in (True, False):
            left_x, right_x = (
                (-half_wicket, half_wicket)
                if image_left_is_pitch_left
                else (half_wicket, -half_wicket)
            )
            pitch_points = np.float32(
                [[left_x, near_y], [right_x, near_y], [left_x, far_y], [right_x, far_y]]
            )
            image_to_pitch = cv2.getPerspectiveTransform(image_points, pitch_points)
            if not np.isfinite(image_to_pitch).all():
                continue
            determinant = float(np.linalg.det(image_to_pitch))
            condition = float(np.linalg.cond(image_to_pitch))
            if abs(determinant) < 1e-12 or not math.isfinite(condition) or condition > 1e12:
                continue
            try:
                pitch_to_image = np.linalg.inv(image_to_pitch)
            except np.linalg.LinAlgError:
                continue
            round_trip = _transform([tuple(point) for point in pitch_points], pitch_to_image)
            rmse = float(np.sqrt(np.mean(np.square(round_trip - image_points))))
            pitch_corners = _transform(
                [
                    (-specification.dimensions.pitch_width_m / 2, 0),
                    (specification.dimensions.pitch_width_m / 2, 0),
                    (specification.dimensions.pitch_width_m / 2, length),
                    (-specification.dimensions.pitch_width_m / 2, length),
                ],
                pitch_to_image,
            )
            area = _polygon_area(pitch_corners)
            if not np.isfinite(pitch_corners).all() or area < 10:
                continue
            in_margin = np.mean(
                (pitch_corners[:, 0] >= -image_width)
                & (pitch_corners[:, 0] <= image_width * 2)
                & (pitch_corners[:, 1] >= -image_height)
                & (pitch_corners[:, 1] <= image_height * 2)
            )
            perspective = min(1.0, (near.width * near.height) / max(far.width * far.height, 1.0))
            score = float(max(0, min(1, 0.45 * in_margin + 0.35 * min(1, area / (image_width * image_height)) + 0.20 * perspective)))
            hypothesis_id = f"near_{near_end}__image_left_{'pitch_left' if image_left_is_pitch_left else 'pitch_right'}"
            hypotheses.append((score, hypothesis_id, near_end, image_left_is_pitch_left, image_to_pitch, pitch_to_image, rmse, area))
    if not hypotheses:
        return PitchFitResult(
            status="PITCH_FIT_FAILED",
            fit_score=0,
            confidence=0,
            warnings=["All deterministic two-wicket homography hypotheses were invalid."],
        )
    # With box-only evidence semantic end and mirror can be tied. Preserve a
    # deterministic camera-behind-bowler convention and report the ambiguity.
    selected = max(
        hypotheses,
        key=lambda item: (
            item[0],
            item[2] == "bowler",
            item[3],
        ),
    )
    score, hypothesis_id, near_end, image_left_is_pitch_left, image_to_pitch, pitch_to_image, rmse, area = selected
    near_y, far_y = ((0.0, length) if near_end == "bowler" else (length, 0.0))
    left_x, right_x = ((-half_wicket, half_wicket) if image_left_is_pitch_left else (half_wicket, -half_wicket))
    semantics = ("near_left_base", "near_right_base", "far_left_base", "far_right_base")
    pitch_values = ((left_x, near_y), (right_x, near_y), (left_x, far_y), (right_x, far_y))
    correspondences = [
        PitchFitCorrespondence(
            semantic_id=semantic,
            image_point=ImagePoint(x=float(image[0]), y=float(image[1])),
            pitch_point=PitchPoint(x_m=float(pitch[0]), y_m=float(pitch[1])),
        )
        for semantic, image, pitch in zip(semantics, image_points, pitch_values)
    ]
    support = min(1.0, min(near.frame_support, far.frame_support) / 3)
    confidence = score * ((near.confidence + far.confidence) / 2) * (0.65 + 0.35 * support)
    return PitchFitResult(
        status="READY",
        selected_hypothesis=hypothesis_id,
        near_semantic_end=near_end,
        image_left_is_pitch_left=image_left_is_pitch_left,
        image_to_pitch_homography=image_to_pitch.tolist(),
        pitch_to_image_homography=pitch_to_image.tolist(),
        correspondences=correspondences,
        projected_pitch=_projected_geometry(pitch_to_image),
        determinant=float(np.linalg.det(image_to_pitch)),
        condition_number=float(np.linalg.cond(image_to_pitch)),
        reprojection_rmse_px=rmse,
        projected_pitch_area_px2=area,
        fit_score=round(score, 6),
        confidence=round(max(0, min(1, confidence)), 6),
        warnings=[
            "Box bottoms are approximate wicket-base extents, not independently detected stump keypoints.",
            "Box-only evidence cannot independently resolve semantic end or pitch-left mirroring.",
            "Ground homography projects z=0 geometry only; airborne height and vertical stump projection are unavailable.",
        ],
    )
