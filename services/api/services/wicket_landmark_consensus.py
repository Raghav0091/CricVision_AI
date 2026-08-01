"""Robust temporal fusion for independently extracted wicket landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .wicket_landmark_extractor import LandmarkEvidence, WicketLandmarkExtraction
from .wicket_line_geometry import LineSegment, Point2D, inlier_indices, robust_location


@dataclass(frozen=True)
class WicketLandmarkConsensus:
    role: str
    status: str
    axes: tuple[LandmarkEvidence, ...]
    points: tuple[LandmarkEvidence, ...]
    lines: tuple[LandmarkEvidence, ...]
    supporting_frame_ids: tuple[int, ...]
    confidence: float
    uncertainty_px: float | None
    diagnostics: Mapping[str, Any]
    warnings: tuple[str, ...] = ()

    def as_mapping(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status,
            "axes": [item.as_mapping() for item in self.axes],
            "points": [item.as_mapping() for item in self.points],
            "lines": [item.as_mapping() for item in self.lines],
            "supporting_frame_ids": list(self.supporting_frame_ids),
            "confidence": self.confidence,
            "uncertainty_px": self.uncertainty_px,
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
        }


def _fuse_point(items: Sequence[LandmarkEvidence], minimum_support: int) -> LandmarkEvidence:
    available = [item for item in items if item.status == "AVAILABLE" and item.point is not None]
    semantic_id = items[0].semantic_id
    if len(available) < minimum_support:
        return LandmarkEvidence(semantic_id, "POINT", "UNAVAILABLE", reason="temporal_support_insufficient")
    x_indices = set(inlier_indices([item.point.x for item in available]))
    y_indices = set(inlier_indices([item.point.y for item in available]))
    inliers = [item for index, item in enumerate(available) if index in x_indices & y_indices]
    if len(inliers) < minimum_support:
        return LandmarkEvidence(semantic_id, "POINT", "UNAVAILABLE", reason="temporal_consensus_rejected_outliers")
    x, sx = robust_location([item.point.x for item in inliers])
    y, sy = robust_location([item.point.y for item in inliers])
    frames = tuple(sorted({frame for item in inliers for frame in item.supporting_frame_ids}))
    confidence = float(np.clip(np.median([item.confidence for item in inliers]) * min(1.0, len(inliers) / max(3, minimum_support)), 0, 1))
    return LandmarkEvidence(
        semantic_id, "POINT", "AVAILABLE", confidence=confidence,
        point=Point2D(x, y), uncertainty_x_px=max(0.5, sx), uncertainty_y_px=max(0.5, sy),
        supporting_frame_ids=frames, extraction_method="temporal_median_mad_consensus_v1",
    )


def _fuse_line(items: Sequence[LandmarkEvidence], minimum_support: int) -> LandmarkEvidence:
    available = [item for item in items if item.status == "AVAILABLE" and item.line is not None]
    semantic_id = items[0].semantic_id
    if len(available) < minimum_support:
        return LandmarkEvidence(semantic_id, "LINE", "UNAVAILABLE", reason="temporal_support_insufficient")
    horizontal = "bail" in semantic_id or "base_line" in semantic_id
    positions = [item.line.midpoint.y if horizontal else item.line.midpoint.x for item in available]
    indices = inlier_indices(positions)
    inliers = [available[index] for index in indices]
    if len(inliers) < minimum_support:
        return LandmarkEvidence(semantic_id, "LINE", "UNAVAILABLE", reason="temporal_consensus_rejected_outliers")
    if horizontal:
        position, spread = robust_location([item.line.midpoint.y for item in inliers])
        start = Point2D(float(np.median([item.line.ordered_by_x().start.x for item in inliers])), position)
        end = Point2D(float(np.median([item.line.ordered_by_x().end.x for item in inliers])), position)
    else:
        position, spread = robust_location([item.line.midpoint.x for item in inliers])
        start = Point2D(position, float(np.median([item.line.ordered_by_y().start.y for item in inliers])))
        end = Point2D(position, float(np.median([item.line.ordered_by_y().end.y for item in inliers])))
    frames = tuple(sorted({frame for item in inliers for frame in item.supporting_frame_ids}))
    confidence = float(np.clip(np.median([item.confidence for item in inliers]) * min(1.0, len(inliers) / max(3, minimum_support)), 0, 1))
    return LandmarkEvidence(
        semantic_id, "LINE", "AVAILABLE", confidence=confidence, line=LineSegment(start, end),
        angular_uncertainty_deg=max(0.4, float(np.median([item.angular_uncertainty_deg or 2.0 for item in inliers]))),
        perpendicular_uncertainty_px=max(0.5, spread), supporting_frame_ids=frames,
        extraction_method="temporal_median_mad_consensus_v1",
    )


def build_wicket_landmark_consensus(
    extractions: Sequence[WicketLandmarkExtraction],
    *,
    minimum_support: int = 3,
) -> WicketLandmarkConsensus:
    """Fuse aligned native-coordinate extractions while withholding weak evidence."""

    if minimum_support < 2:
        raise ValueError("minimum_support must be at least 2")
    if not extractions:
        return WicketLandmarkConsensus("unresolved", "UNAVAILABLE", (), (), (), (), 0.0, None, {"input_count": 0}, ("no_extractions",))
    role = extractions[0].role
    compatible = [item for item in extractions if item.role == role]
    available = [item for item in compatible if item.status == "AVAILABLE"]
    ids = sorted({evidence.semantic_id for item in compatible for evidence in item.axes + item.points + item.lines})
    by_id = {semantic_id: [evidence for item in compatible for evidence in item.axes + item.points + item.lines if evidence.semantic_id == semantic_id] for semantic_id in ids}
    axes = tuple(_fuse_line(by_id[item], minimum_support) for item in ids if "axis" in item)
    points = tuple(_fuse_point(by_id[item], minimum_support) for item in ids if "stump_top" in item or "stump_base" in item)
    lines = tuple(_fuse_line(by_id[item], minimum_support) for item in ids if item in ("bail_line", "base_line"))
    published = [item for item in axes + points + lines if item.status == "AVAILABLE"]
    frame_ids = tuple(sorted({frame for item in published for frame in item.supporting_frame_ids}))
    status = "AVAILABLE" if published else "UNAVAILABLE"
    confidence = float(np.median([item.confidence for item in published])) if published else 0.0
    uncertainties = [item.perpendicular_uncertainty_px for item in axes + lines if item.perpendicular_uncertainty_px is not None]
    uncertainties += [max(item.uncertainty_x_px or 0, item.uncertainty_y_px or 0) for item in points if item.status == "AVAILABLE"]
    uncertainty = float(np.median(uncertainties)) if uncertainties else None
    diagnostics = {
        "input_count": len(extractions),
        "compatible_role_count": len(compatible),
        "available_extraction_count": len(available),
        "minimum_support": minimum_support,
        "published_evidence_count": len(published),
    }
    warnings = () if status == "AVAILABLE" else ("temporal_evidence_unavailable",)
    return WicketLandmarkConsensus(role, status, axes, points, lines, frame_ids, confidence, uncertainty, diagnostics, warnings)
