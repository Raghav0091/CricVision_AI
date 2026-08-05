"""Classical, conservative wicket landmark extraction from native ROI crops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np

from .wicket_line_geometry import (
    LineSegment,
    Point2D,
    is_horizontal,
    is_vertical,
    merge_collinear_segments,
    normalized_line_equation,
    translate_line,
)

EvidenceStatus = Literal["AVAILABLE", "UNAVAILABLE", "REJECTED"]
WicketRole = Literal["near", "far", "unresolved"]

AXIS_IDS = ("left_stump_axis", "middle_stump_axis", "right_stump_axis")
POINT_IDS = tuple(
    f"{side}_stump_{level}"
    for side in ("left", "middle", "right")
    for level in ("top", "base")
)
LINE_IDS = ("bail_line", "base_line")


@dataclass(frozen=True)
class LandmarkEvidence:
    semantic_id: str
    geometry_type: Literal["POINT", "LINE"]
    status: EvidenceStatus
    confidence: float = 0.0
    point: Point2D | None = None
    line: LineSegment | None = None
    uncertainty_x_px: float | None = None
    uncertainty_y_px: float | None = None
    angular_uncertainty_deg: float | None = None
    perpendicular_uncertainty_px: float | None = None
    supporting_frame_ids: tuple[int, ...] = ()
    extraction_method: str = "classical_line_geometry_v1"
    reason: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, Any]:
        equation = normalized_line_equation(self.line) if self.line is not None else None
        return {
            "semantic_id": self.semantic_id,
            "geometry_type": self.geometry_type,
            "status": self.status,
            "confidence": float(self.confidence),
            "point": self.point.as_mapping() if self.point else None,
            "line": self.line.as_mapping() if self.line else None,
            "normalized_line_equation": list(equation) if equation else None,
            "uncertainty_x_px": self.uncertainty_x_px,
            "uncertainty_y_px": self.uncertainty_y_px,
            "angular_uncertainty_deg": self.angular_uncertainty_deg,
            "perpendicular_uncertainty_px": self.perpendicular_uncertainty_px,
            "supporting_frame_count": len(self.supporting_frame_ids),
            "supporting_frame_ids": list(self.supporting_frame_ids),
            "extraction_method": self.extraction_method,
            "reason": self.reason,
            "attributes": dict(self.attributes),
        }

    def as_contract_mapping(self) -> dict[str, Any]:
        """Return flat fields matching the v1 evidence contract vocabulary."""

        semantic_type = (
            "UNAVAILABLE"
            if self.status != "AVAILABLE"
            else "POINTLIKE"
            if self.geometry_type == "POINT"
            else "LINE"
        )
        common = {
            "semantic_id": self.semantic_id,
            "confidence": float(self.confidence),
            "supporting_frame_count": len(self.supporting_frame_ids),
            "supporting_frame_ids": list(self.supporting_frame_ids),
            "extraction_method": self.extraction_method,
            "semantic_type": semantic_type,
            "status": self.status,
            "correlation_family": self.attributes.get("correlation_family", self.semantic_id),
            "rejection_reason": self.reason,
        }
        if self.geometry_type == "POINT":
            return {
                **common,
                "x_px": self.point.x if self.point else None,
                "y_px": self.point.y if self.point else None,
                "uncertainty_x_px": self.uncertainty_x_px,
                "uncertainty_y_px": self.uncertainty_y_px,
            }
        equation = normalized_line_equation(self.line) if self.line else None
        return {
            **common,
            "start_x_px": self.line.start.x if self.line else None,
            "start_y_px": self.line.start.y if self.line else None,
            "end_x_px": self.line.end.x if self.line else None,
            "end_y_px": self.line.end.y if self.line else None,
            "normalized_line_equation": (
                {"a": equation[0], "b": equation[1], "c": equation[2]}
                if equation else None
            ),
            "angular_uncertainty_deg": self.angular_uncertainty_deg,
            "perpendicular_uncertainty_px": self.perpendicular_uncertainty_px,
        }


@dataclass(frozen=True)
class WicketLandmarkExtraction:
    role: WicketRole
    frame_id: int
    status: EvidenceStatus
    axes: tuple[LandmarkEvidence, ...]
    points: tuple[LandmarkEvidence, ...]
    lines: tuple[LandmarkEvidence, ...]
    outer_envelope: Mapping[str, float] | None
    confidence: float
    uncertainty_px: float | None
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...] = ()

    def evidence_by_id(self) -> dict[str, LandmarkEvidence]:
        return {item.semantic_id: item for item in self.axes + self.points + self.lines}

    def as_mapping(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "frame_id": self.frame_id,
            "status": self.status,
            "axes": [item.as_mapping() for item in self.axes],
            "points": [item.as_mapping() for item in self.points],
            "lines": [item.as_mapping() for item in self.lines],
            "outer_envelope": dict(self.outer_envelope) if self.outer_envelope else None,
            "confidence": float(self.confidence),
            "uncertainty_px": self.uncertainty_px,
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
        }


def _unavailable(semantic_id: str, geometry_type: Literal["POINT", "LINE"], reason: str, frame_id: int) -> LandmarkEvidence:
    return LandmarkEvidence(
        semantic_id=semantic_id,
        geometry_type=geometry_type,
        status="UNAVAILABLE",
        supporting_frame_ids=(frame_id,),
        reason=reason,
    )


def _empty(role: WicketRole, frame_id: int, reason: str, diagnostics: Mapping[str, Any]) -> WicketLandmarkExtraction:
    return WicketLandmarkExtraction(
        role=role,
        frame_id=frame_id,
        status="UNAVAILABLE",
        axes=tuple(_unavailable(item, "LINE", reason, frame_id) for item in AXIS_IDS),
        points=tuple(_unavailable(item, "POINT", reason, frame_id) for item in POINT_IDS),
        lines=tuple(_unavailable(item, "LINE", reason, frame_id) for item in LINE_IDS),
        outer_envelope=None,
        confidence=0.0,
        uncertainty_px=None,
        diagnostics=diagnostics,
        warnings=(reason,),
    )


def _hough_segments(gray: np.ndarray) -> list[LineSegment]:
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    edges = cv2.Canny(contrast, 45, 135, L2gradient=True)
    height, width = gray.shape
    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(10, min(height, width) // 7),
        minLineLength=max(8, int(height * 0.22)),
        maxLineGap=max(3, int(height * 0.05)),
    )
    if detected is None:
        return []
    return [
        LineSegment(Point2D(float(x1), float(y1)), Point2D(float(x2), float(y2)))
        for x1, y1, x2, y2 in np.asarray(detected).reshape(-1, 4)
    ]


def _best_horizontal(
    lines: Sequence[LineSegment],
    *,
    top: bool,
    axes: Sequence[LineSegment],
) -> LineSegment | None:
    if not axes:
        return None
    xs = [item.midpoint.x for item in axes]
    left, right = min(xs), max(xs)
    centre = (left + right) / 2.0
    target_y = float(np.median([
        item.ordered_by_y().start.y if top else item.ordered_by_y().end.y
        for item in axes
    ]))
    axis_span = float(np.median([item.length for item in axes]))
    endpoint_tolerance = max(4.0, axis_span * 0.08)
    candidates = []
    for line in lines:
        ordered = line.ordered_by_x()
        if ordered.end.x < left - 0.25 * (right - left) or ordered.start.x > right + 0.25 * (right - left):
            continue
        endpoint_distance = abs(ordered.midpoint.y - target_y)
        if endpoint_distance > endpoint_tolerance:
            continue
        coverage = max(0.0, min(ordered.end.x, right) - max(ordered.start.x, left)) / max(1.0, right - left)
        crossing_count = sum(ordered.start.x <= x <= ordered.end.x for x in xs)
        if coverage < 0.55 or crossing_count < 2:
            continue
        endpoint_score = 1.0 - endpoint_distance / endpoint_tolerance
        centre_support = 1.0 if ordered.start.x <= centre <= ordered.end.x else 0.0
        score = 0.45 * coverage + 0.35 * endpoint_score + 0.10 * centre_support + 0.10 * min(1.0, crossing_count / 3.0)
        candidates.append((score, ordered))
    score, selected = max(candidates, default=(0.0, None), key=lambda item: item[0])
    return selected if score >= 0.55 else None


def _axis_group(
    vertical: Sequence[LineSegment],
    width: int,
    height: int,
    *,
    role: WicketRole,
) -> tuple[list[LineSegment], dict[str, float]]:
    viable = [
        line.ordered_by_y()
        for line in vertical
        if line.length >= height * 0.34
        and width * 0.04 <= line.midpoint.x <= width * 0.96
    ]
    best: tuple[float, list[LineSegment], dict[str, float]] | None = None
    for index in range(max(0, len(viable) - 2)):
        group = viable[index:index + 3]
        xs = [item.midpoint.x for item in group]
        spacing = (xs[1] - xs[0], xs[2] - xs[1])
        minimum_separation = max(1.5, width * 0.018) if role == "far" else max(2.5, width * 0.025)
        if min(spacing) < minimum_separation:
            continue
        spacing_ratio = min(spacing) / max(spacing)
        span_overlap = max(0.0, min(item.end.y for item in group) - max(item.start.y for item in group))
        span_overlap /= max(1.0, max(item.end.y for item in group) - min(item.start.y for item in group))
        boundary_margin = min(xs[0], width - xs[-1]) / max(1.0, width)
        score = 0.55 * spacing_ratio + 0.35 * span_overlap + 0.10 * min(1.0, boundary_margin / 0.08)
        details = {
            "spacing_ratio": spacing_ratio,
            "span_overlap": span_overlap,
            "group_score": score,
            "minimum_axis_separation_px": minimum_separation,
            "observed_minimum_axis_separation_px": min(spacing),
        }
        if spacing_ratio >= 0.58 and span_overlap >= 0.48 and (best is None or score > best[0]):
            best = (score, group, details)
    return (best[1], best[2]) if best else ([], {"spacing_ratio": 0.0, "span_overlap": 0.0, "group_score": 0.0})


def extract_wicket_landmarks(
    image: np.ndarray,
    *,
    role: WicketRole,
    frame_id: int,
    native_origin: tuple[float, float] = (0.0, 0.0),
    supporting_frame_ids: Sequence[int] | None = None,
) -> WicketLandmarkExtraction:
    """Extract one conservative evidence set from an aligned native-resolution ROI."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        return _empty(role, frame_id, "invalid_or_empty_roi", {"image_valid": False})
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    height, width = gray.shape
    diagnostics: dict[str, Any] = {"image_valid": True, "roi_width": width, "roi_height": height}
    minimum_width = 28 if role == "near" else 16
    minimum_height = 38 if role == "near" else 24
    if width < minimum_width or height < minimum_height:
        return _empty(role, frame_id, "roi_resolution_insufficient", diagnostics)
    if float(np.std(gray)) < 7.0:
        return _empty(role, frame_id, "local_contrast_insufficient", diagnostics)

    raw = _hough_segments(gray)
    vertical = merge_collinear_segments(
        [item for item in raw if is_vertical(item)],
        orientation="vertical",
        position_tolerance_px=(
            max(1.25, width * 0.014) if role == "far" else max(5.5, width * 0.035)
        ),
    )
    horizontal = merge_collinear_segments(
        [item for item in raw if is_horizontal(item)],
        orientation="horizontal",
        position_tolerance_px=max(2.0, height * 0.018),
    )
    vertical.sort(key=lambda item: item.midpoint.x)
    diagnostics.update(raw_line_count=len(raw), vertical_cluster_count=len(vertical), horizontal_cluster_count=len(horizontal))
    if len(vertical) > 10:
        return _empty(role, frame_id, "vertical_clutter_or_player_occlusion", diagnostics)

    group, group_details = _axis_group(vertical, width, height, role=role)
    diagnostics.update(group_details)
    if not group:
        return _empty(role, frame_id, "three_stump_geometry_not_resolved", diagnostics)

    top_target = float(np.median([item.ordered_by_y().start.y for item in group]))
    base_target = float(np.median([item.ordered_by_y().end.y for item in group]))
    endpoint_tolerance = max(4.0, float(np.median([item.length for item in group])) * 0.08)
    top = _best_horizontal(horizontal, top=True, axes=group)
    base = _best_horizontal(horizontal, top=False, axes=group)
    if top is not None and base is not None and abs(top.midpoint.y - base.midpoint.y) < height * 0.20:
        if top.midpoint.y < height / 2.0:
            base = None
        else:
            top = None
    horizontal_support = int(top is not None) + int(base is not None)
    support_ids = tuple(sorted(set(supporting_frame_ids or (frame_id,))))
    diagnostics.update(
        horizontal_support_count=horizontal_support,
        top_endpoint_target_y=top_target,
        base_endpoint_target_y=base_target,
        endpoint_zone_tolerance_px=endpoint_tolerance,
        selected_top_line_y=top.midpoint.y if top else None,
        selected_base_line_y=base.midpoint.y if base else None,
        temporal_support_count=len(support_ids),
    )
    # Three long, evenly spaced axes without any transverse support are commonly net poles.
    if horizontal_support == 0:
        minimum_temporal_support = 5 if role == "far" else 3
        minimum_group_score = 0.80 if role == "far" else 0.82
        diagnostics.update(
            axes_only_minimum_temporal_support=minimum_temporal_support,
            axes_only_minimum_group_score=minimum_group_score,
        )
        strong_temporal_axes = (
            len(support_ids) >= minimum_temporal_support
            and group_details["group_score"] >= minimum_group_score
            and group_details["span_overlap"] >= 0.70
            and group_details["spacing_ratio"] >= 0.70
        )
        if not strong_temporal_axes:
            return _empty(role, frame_id, "axes_lack_wicket_top_or_base_support", diagnostics)

    confidence = float(np.clip(
        0.28 + 0.34 * group_details["group_score"] + 0.16 * horizontal_support + 0.12 * min(1.0, np.std(gray) / 48.0),
        0.0,
        0.96,
    ))
    separation = group_details["observed_minimum_axis_separation_px"]
    resolution_penalty = 1.0 + (1.5 / max(1.5, separation) if role == "far" else 0.0)
    uncertainty = max(0.75, 3.5 * (1.0 - confidence), width * 0.008) * resolution_penalty
    dx, dy = native_origin
    translated_axes = [translate_line(item, dx, dy) for item in group]
    top_native = translate_line(top, dx, dy) if top else None
    base_native = translate_line(base, dx, dy) if base else None

    axes = tuple(
        LandmarkEvidence(
            semantic_id=semantic_id,
            geometry_type="LINE",
            status="AVAILABLE",
            confidence=confidence,
            line=line,
            angular_uncertainty_deg=max(0.5, 4.0 * (1.0 - confidence)),
            perpendicular_uncertainty_px=uncertainty,
            supporting_frame_ids=support_ids,
        )
        for semantic_id, line in zip(AXIS_IDS, translated_axes)
    )

    points: list[LandmarkEvidence] = []
    for side, axis in zip(("left", "middle", "right"), translated_axes):
        for level, support_line in (("top", top_native), ("base", base_native)):
            semantic_id = f"{side}_stump_{level}"
            if support_line is None:
                points.append(_unavailable(semantic_id, "POINT", f"{level}_line_unavailable", frame_id))
                continue
            y = support_line.midpoint.y
            points.append(LandmarkEvidence(
                semantic_id=semantic_id,
                geometry_type="POINT",
                status="AVAILABLE",
                confidence=max(0.0, confidence - 0.04),
                point=Point2D(axis.midpoint.x, y),
                uncertainty_x_px=uncertainty,
                uncertainty_y_px=uncertainty * 1.25,
                supporting_frame_ids=support_ids,
                attributes={"intersection_of": [f"{side}_stump_axis", f"{level}_line"]},
            ))
    lines = tuple(
        LandmarkEvidence(
            semantic_id=semantic_id,
            geometry_type="LINE",
            status="AVAILABLE" if line else "UNAVAILABLE",
            confidence=max(0.0, confidence - 0.03) if line else 0.0,
            line=line,
            angular_uncertainty_deg=max(0.6, 4.5 * (1.0 - confidence)) if line else None,
            perpendicular_uncertainty_px=uncertainty * 1.2 if line else None,
            supporting_frame_ids=support_ids,
            reason=None if line else f"{semantic_id}_not_supported",
        )
        for semantic_id, line in (("bail_line", top_native), ("base_line", base_native))
    )
    y_values = [value for line in translated_axes for value in (line.start.y, line.end.y)]
    envelope = {
        "left": min(item.midpoint.x for item in translated_axes),
        "right": max(item.midpoint.x for item in translated_axes),
        "top": top_native.midpoint.y if top_native else min(y_values),
        "base": base_native.midpoint.y if base_native else max(y_values),
    }
    status: EvidenceStatus = "AVAILABLE"
    warnings = (
        ()
        if horizontal_support == 2
        else ("axes_only_transverse_support_unavailable",)
        if horizontal_support == 0
        else ("partial_top_or_base_support",)
    )
    return WicketLandmarkExtraction(
        role=role,
        frame_id=frame_id,
        status=status,
        axes=axes,
        points=tuple(points),
        lines=lines,
        outer_envelope=envelope,
        confidence=confidence,
        uncertainty_px=uncertainty,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def extract_aligned_wicket_landmarks(
    source: object,
    *,
    role: WicketRole | None = None,
    frame_id: int | None = None,
) -> WicketLandmarkExtraction:
    """Adapt an Agent-1-style aligned crop mapping/dataclass to the NumPy API.

    The adapter intentionally uses structural fields instead of importing the frame
    service. Supported image names are ``consensus_image``, ``aligned_image``,
    ``image`` and ``crop``. Native origin may be ``native_origin`` or an ROI with
    ``x`` and ``y`` fields.
    """

    def read(name: str, default: Any = None) -> Any:
        if isinstance(source, Mapping):
            return source.get(name, default)
        return getattr(source, name, default)

    image = source if isinstance(source, np.ndarray) else next(
        (value for name in ("consensus_image", "aligned_image", "image", "crop") if isinstance((value := read(name)), np.ndarray)),
        None,
    )
    resolved_role = role or read("role") or read("wicket_role") or "unresolved"
    if resolved_role not in ("near", "far", "unresolved"):
        resolved_role = "unresolved"
    resolved_frame = frame_id
    if resolved_frame is None:
        resolved_frame = read("frame_id", read("frame_index", 0))
    origin = read("native_origin")
    if origin is None:
        roi = read("native_roi", read("roi"))
        if roi is not None:
            roi_read = roi.get if isinstance(roi, Mapping) else lambda name, default=0.0: getattr(roi, name, default)
            origin = (float(roi_read("x", 0.0)), float(roi_read("y", 0.0)))
    if origin is None:
        origin = (0.0, 0.0)
    accepted_frame_ids = read("accepted_frame_ids")
    if accepted_frame_ids is None:
        aligned_stack = read("aligned_stack")
        if aligned_stack is not None:
            accepted_frame_ids = (
                aligned_stack.get("accepted_frame_ids")
                if isinstance(aligned_stack, Mapping)
                else getattr(aligned_stack, "accepted_frame_ids", None)
            )
    return extract_wicket_landmarks(
        image if image is not None else np.asarray([], dtype=np.uint8),
        role=resolved_role,
        frame_id=int(resolved_frame),
        native_origin=(float(origin[0]), float(origin[1])),
        supporting_frame_ids=accepted_frame_ids,
    )
