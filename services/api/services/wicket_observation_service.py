"""Real-video wicket region consensus and evidence-backed 2D landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..schemas.wicket_observation import (
    AssignmentHypothesis,
    PixelBox,
    PixelPoint,
    RawWicketDetection,
    RoiMetadata,
    SetupFrameCandidate,
    WicketLandmarkObservation,
    WicketLineObservation,
    WicketObservation,
    WicketObservationDiagnostics,
    WicketObservationResult,
    WicketRegionObservation,
)
from .stump_detector_service import (
    STUMP_MODEL_RELATIVE_PATH,
    detect_wickets_robust,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)


RESULT_FILENAME = "wicket_observations_v1.json"
DEBUG_DIRECTORY = "wicket_observations_v1"
SAMPLE_LIMIT = 12
SAMPLE_WINDOW_FRAMES = 36
MIN_SHARPNESS = 18.0
MIN_BRIGHTNESS = 14.0
MAX_BRIGHTNESS = 246.0
COARSE_POINT_IDS = (
    "wicket_outer_left_base",
    "wicket_outer_right_base",
    "wicket_base_center",
    "wicket_outer_left_top",
    "wicket_outer_right_top",
    "wicket_top_center",
    "wicket_center",
)
COARSE_LINE_IDS = (
    "base_line",
    "top_or_bail_line",
    "left_outer_axis",
    "right_outer_axis",
)
DETAILED_POINT_IDS = tuple(
    f"{side}_stump_{level}"
    for level in ("base", "top")
    for side in ("left", "middle", "right")
)


@dataclass(frozen=True)
class FrameEvidence:
    index: int
    timestamp: float
    frame: np.ndarray
    sharpness: float
    brightness: float
    obstruction: float
    detector_result: dict[str, Any]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _box_dict(candidate: dict[str, Any]) -> dict[str, float]:
    box = candidate["bbox"]
    return {
        "x": float(box["x"]),
        "y": float(box["y"]),
        "width": float(box["width"]),
        "height": float(box["height"]),
    }


def _box_iou(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = (
        first["width"] * first["height"]
        + second["width"] * second["height"]
        - intersection
    )
    return intersection / max(union, 1e-6)


def _intersection_over_smaller(
    first: dict[str, float], second: dict[str, float]
) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    smaller = min(
        first["width"] * first["height"],
        second["width"] * second["height"],
    )
    return intersection / max(smaller, 1e-6)


def _role_candidate(
    detector_result: dict[str, Any], role: str
) -> dict[str, Any] | None:
    # The legacy robust detector uses striker=far and non_striker=near. These
    # labels are not trusted as cricket-end semantics here.
    legacy_key = "non_striker" if role == "near" else "striker"
    selected = detector_result.get("selected") or {}
    candidate = selected.get(legacy_key)
    return candidate if isinstance(candidate, dict) else None


def sample_frame_indices(
    frame_count: int,
    *,
    window_frames: int = SAMPLE_WINDOW_FRAMES,
    sample_limit: int = SAMPLE_LIMIT,
) -> list[int]:
    """Return deterministic, bounded early-frame indices."""
    window = max(0, min(int(frame_count), int(window_frames)))
    if window <= 0:
        return []
    count = min(window, max(1, int(sample_limit)))
    if count == 1:
        return [0]
    return sorted(
        {
            int(round(index * (window - 1) / (count - 1)))
            for index in range(count)
        }
    )


def frame_quality_metrics(frame: np.ndarray) -> tuple[float, float, float]:
    """Return sharpness, brightness and a conservative obstruction heuristic."""
    if frame.size == 0:
        return 0.0, 0.0, 1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    # Entropy-like edge occupancy is only a clutter/obstruction proxy. It is
    # deliberately reported as a heuristic, not person segmentation.
    edges = cv2.Canny(gray, 80, 180)
    edge_fraction = float(np.count_nonzero(edges)) / max(edges.size, 1)
    obstruction = _clamp((edge_fraction - 0.10) / 0.28)
    return sharpness, brightness, obstruction


def _candidate_detection_stability(
    evidence: Sequence[FrameEvidence], index: int
) -> float:
    current = evidence[index]
    scores: list[float] = []
    for role in ("near", "far"):
        box = _role_candidate(current.detector_result, role)
        if box is None:
            continue
        overlaps: list[float] = []
        for neighbour_index in (index - 1, index + 1):
            if neighbour_index < 0 or neighbour_index >= len(evidence):
                continue
            neighbour = _role_candidate(
                evidence[neighbour_index].detector_result, role
            )
            if neighbour is not None:
                overlaps.append(_box_iou(_box_dict(box), _box_dict(neighbour)))
        if overlaps:
            scores.append(max(overlaps))
    return float(sum(scores) / len(scores)) if scores else 0.0


def score_setup_frames(
    evidence: Sequence[FrameEvidence],
) -> list[SetupFrameCandidate]:
    candidates: list[SetupFrameCandidate] = []
    for position, item in enumerate(evidence):
        accepted = item.detector_result.get("candidates") or []
        confidences = [
            float(candidate.get("confidence") or 0.0) for candidate in accepted
        ]
        detection_count = len(
            [
                role
                for role in ("near", "far")
                if _role_candidate(item.detector_result, role) is not None
            ]
        )
        stability = _candidate_detection_stability(evidence, position)
        sharpness_score = _clamp(
            math.log1p(item.sharpness) / math.log1p(500.0)
        )
        brightness_score = _clamp(
            1.0 - abs(item.brightness - 128.0) / 128.0
        )
        detector_score = detection_count / 2.0
        mean_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        score = _clamp(
            0.34 * detector_score
            + 0.19 * mean_confidence
            + 0.18 * stability
            + 0.16 * sharpness_score
            + 0.09 * brightness_score
            + 0.04 * (1.0 - item.obstruction)
        )
        reasons: list[str] = []
        if item.sharpness < MIN_SHARPNESS:
            reasons.append("motion_blur_or_low_detail")
        if not MIN_BRIGHTNESS <= item.brightness <= MAX_BRIGHTNESS:
            reasons.append("brightness_out_of_range")
        if detection_count == 0:
            reasons.append("no_wicket_regions_detected")
        if item.obstruction > 0.7:
            reasons.append("high_visual_clutter_or_obstruction_heuristic")
        candidates.append(
            SetupFrameCandidate(
                frame_index=item.index,
                timestamp_seconds=item.timestamp,
                image_width=int(item.frame.shape[1]),
                image_height=int(item.frame.shape[0]),
                score=round(score, 6),
                sharpness=round(item.sharpness, 3),
                brightness=round(item.brightness, 3),
                wicket_detection_count=detection_count,
                mean_detector_confidence=round(mean_confidence, 6),
                detection_stability=round(stability, 6),
                obstruction_score=round(item.obstruction, 6),
                selected=False,
                rejection_reasons=reasons,
            )
        )
    if candidates:
        # Stable tie break: highest score, then earliest frame.
        selected = max(candidates, key=lambda item: (item.score, -item.frame_index))
        selected.selected = True
    return candidates


def build_consensus_region(
    detections: Sequence[RawWicketDetection],
    *,
    perspective_role: str,
    selected_frame_index: int,
    fps: float,
) -> WicketRegionObservation | None:
    """Build a robust median region without combining incompatible boxes."""
    if not detections:
        return None
    reference = min(
        detections,
        key=lambda item: abs(item.frame_index - selected_frame_index),
    )
    reference_box = reference.bbox.model_dump()
    compatible = [
        item
        for item in detections
        if _box_iou(reference_box, item.bbox.model_dump()) >= 0.2
        or math.hypot(
            item.bbox.x + item.bbox.width / 2
            - (reference.bbox.x + reference.bbox.width / 2),
            item.bbox.y + item.bbox.height / 2
            - (reference.bbox.y + reference.bbox.height / 2),
        )
        <= 0.65 * max(reference.bbox.width, reference.bbox.height)
    ]
    if not compatible:
        return None
    xs = [item.bbox.x for item in compatible]
    ys = [item.bbox.y for item in compatible]
    widths = [item.bbox.width for item in compatible]
    heights = [item.bbox.height for item in compatible]
    confidences = [item.confidence for item in compatible]
    x, y = median(xs), median(ys)
    width, height = median(widths), median(heights)
    centres = [
        (item.bbox.x + item.bbox.width / 2, item.bbox.y + item.bbox.height / 2)
        for item in compatible
    ]
    centre_x, centre_y = x + width / 2, y + height / 2
    centre_variation = median(
        [math.hypot(cx - centre_x, cy - centre_y) for cx, cy in centres]
    )
    size_variation = median(
        [
            abs(item.bbox.width - width) / max(width, 1)
            + abs(item.bbox.height - height) / max(height, 1)
            for item in compatible
        ]
    ) / 2
    confidence_variation = median(
        [abs(confidence - median(confidences)) for confidence in confidences]
    )
    support = len({item.frame_index for item in compatible})
    jump_ratio = centre_variation / max(width, height, 1)
    if support >= 3 and jump_ratio <= 0.2 and size_variation <= 0.25:
        stability, quality = "STABLE", "HIGH"
    elif support >= 2 and jump_ratio <= 0.45 and size_variation <= 0.5:
        stability, quality = "PARTIALLY_STABLE", "MEDIUM"
    else:
        stability, quality = "UNSTABLE", "LOW"
    uncertainty = max(1.5, centre_variation + max(width, height) * size_variation)
    return WicketRegionObservation(
        frame_index=selected_frame_index,
        timestamp_seconds=selected_frame_index / max(fps, 1e-6),
        bbox=PixelBox(x=x, y=y, width=width, height=height),
        centre=PixelPoint(x=centre_x, y=centre_y),
        width=width,
        height=height,
        detector_confidence=median(confidences),
        detector_model=STUMP_MODEL_RELATIVE_PATH,
        source="temporal_median_consensus",
        temporal_support=support,
        supporting_frame_ids=sorted({item.frame_index for item in compatible}),
        centre_variation_px=round(centre_variation, 4),
        size_variation_ratio=round(size_variation, 6),
        confidence_variation=round(confidence_variation, 6),
        perspective_role=perspective_role,
        stability=stability,
        quality=quality,
        uncertainty_px=round(uncertainty, 4),
        rejection_reason=(
            None if support >= 2 else "isolated_detection_has_no_temporal_consensus"
        ),
    )


def build_temporal_region_candidates(
    detections: Sequence[RawWicketDetection],
    *,
    selected_frame_index: int,
    fps: float,
) -> list[WicketRegionObservation]:
    """Associate detector boxes before assigning perspective roles."""
    clusters: list[list[RawWicketDetection]] = []
    for detection in sorted(
        detections, key=lambda item: (item.frame_index, -item.confidence)
    ):
        best_cluster: list[RawWicketDetection] | None = None
        best_overlap = 0.0
        for cluster in clusters:
            if any(item.frame_index == detection.frame_index for item in cluster):
                continue
            overlap = _box_iou(
                detection.bbox.model_dump(), cluster[-1].bbox.model_dump()
            )
            if overlap > best_overlap and overlap >= 0.22:
                best_cluster = cluster
                best_overlap = overlap
        if best_cluster is None:
            clusters.append([detection])
        else:
            best_cluster.append(detection)

    regions: list[WicketRegionObservation] = []
    for cluster in clusters:
        region = build_consensus_region(
            cluster,
            perspective_role="UNRESOLVED_WICKET",
            selected_frame_index=selected_frame_index,
            fps=fps,
        )
        if region is not None:
            regions.append(region)
    return regions


def select_near_far_regions(
    regions: Sequence[WicketRegionObservation],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[
    WicketRegionObservation | None,
    WicketRegionObservation | None,
    list[WicketRegionObservation],
]:
    """Choose complementary lower/larger and upper/smaller region tracks."""
    candidates = [
        item
        for item in regions
        if item.temporal_support >= 2
        and item.stability != "UNSTABLE"
        and item.bbox.width * item.bbox.height
        <= frame_width * frame_height * 0.18
    ]
    best_pair: tuple[WicketRegionObservation, WicketRegionObservation] | None = None
    best_score = -1.0
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            first_bottom = (first.bbox.y + first.bbox.height) / frame_height
            second_bottom = (second.bbox.y + second.bbox.height) / frame_height
            near, far = (
                (first, second)
                if first_bottom >= second_bottom
                else (second, first)
            )
            near_bottom = (near.bbox.y + near.bbox.height) / frame_height
            far_bottom = (far.bbox.y + far.bbox.height) / frame_height
            vertical_separation = near_bottom - far_bottom
            if vertical_separation < 0.035:
                continue
            near_area = near.bbox.width * near.bbox.height
            far_area = far.bbox.width * far.bbox.height
            if near_area < 0.55 * far_area:
                continue
            near_box = near.bbox.model_dump()
            far_box = far.bbox.model_dump()
            if (
                _box_iou(near_box, far_box) > 0.25
                or _intersection_over_smaller(near_box, far_box) > 0.55
            ):
                continue
            score = (
                0.22 * _clamp(near.temporal_support / 5)
                + 0.22 * _clamp(far.temporal_support / 5)
                + 0.16 * near.detector_confidence
                + 0.16 * far.detector_confidence
                + 0.12 * _clamp(vertical_separation / 0.25)
                + 0.07 * _clamp(near_area / max(far_area * 2.5, 1))
                + 0.05
                * _clamp(
                    1.0
                    - far_area / max(frame_width * frame_height * 0.06, 1)
                )
            )
            if score > best_score:
                best_pair = near, far
                best_score = score

    if best_pair is None:
        return None, None, list(regions)
    near, far = best_pair
    near.perspective_role = "NEAR_WICKET_CANDIDATE"
    far.perspective_role = "FAR_WICKET_CANDIDATE"
    unresolved = [item for item in regions if item is not near and item is not far]
    return near, far, unresolved


def build_native_roi(
    frame: np.ndarray,
    bbox: PixelBox,
    *,
    padding_fraction_x: float = 0.14,
    padding_fraction_y: float = 0.10,
) -> tuple[np.ndarray, RoiMetadata]:
    frame_height, frame_width = frame.shape[:2]
    padding_x = max(2, int(round(bbox.width * padding_fraction_x)))
    padding_y = max(2, int(round(bbox.height * padding_fraction_y)))
    x1 = max(0, int(math.floor(bbox.x)) - padding_x)
    y1 = max(0, int(math.floor(bbox.y)) - padding_y)
    x2 = min(frame_width, int(math.ceil(bbox.x + bbox.width)) + padding_x)
    y2 = min(frame_height, int(math.ceil(bbox.y + bbox.height)) + padding_y)
    roi = frame[y1:y2, x1:x2].copy()
    metadata = RoiMetadata(
        source_frame_width=frame_width,
        source_frame_height=frame_height,
        x=x1,
        y=y1,
        width=max(1, x2 - x1),
        height=max(1, y2 - y1),
        padding_x=padding_x,
        padding_y=padding_y,
        processing_variants=[
            "grayscale",
            "clahe_contrast",
            "canny_edges",
            "vertical_gradient",
        ],
    )
    return roi, metadata


def preprocess_roi(roi: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(contrast, 45, 135)
    vertical_gradient = cv2.convertScaleAbs(
        cv2.Sobel(contrast, cv2.CV_32F, 1, 0, ksize=3)
    )
    return {
        "grayscale": gray,
        "clahe_contrast": contrast,
        "canny_edges": edges,
        "vertical_gradient": vertical_gradient,
    }


def roi_to_native(point: tuple[float, float], roi: RoiMetadata) -> PixelPoint:
    return PixelPoint(x=roi.x + point[0], y=roi.y + point[1])


def _vertical_clusters(
    edges: np.ndarray,
) -> tuple[list[float], list[tuple[int, int, int, int]], float]:
    height, width = edges.shape
    min_line = max(5, int(height * 0.32))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(8, int(height * 0.12)),
        minLineLength=min_line,
        maxLineGap=max(3, int(height * 0.12)),
    )
    accepted: list[tuple[int, int, int, int]] = []
    if lines is not None:
        for raw in lines[:, 0]:
            x1, y1, x2, y2 = (int(value) for value in raw)
            dy, dx = abs(y2 - y1), abs(x2 - x1)
            if dy >= min_line and dx <= max(3, int(0.22 * dy)):
                accepted.append((x1, y1, x2, y2))
    centres = sorted((line[0] + line[2]) / 2 for line in accepted)
    clusters: list[list[float]] = []
    tolerance = max(3.0, width * 0.045)
    for centre in centres:
        if not clusters or centre - median(clusters[-1]) > tolerance:
            clusters.append([centre])
        else:
            clusters[-1].append(centre)
    cluster_centres = [float(median(cluster)) for cluster in clusters]
    strength = _clamp(
        sum(math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in accepted)
        / max(height * 3.0, 1)
    )
    return cluster_centres, accepted, strength


def _horizontal_boundary(
    edges: np.ndarray, *, upper: bool
) -> tuple[float | None, float]:
    height, width = edges.shape
    start, end = ((0, max(1, height // 2)) if upper else (height // 2, height))
    row_strength = np.count_nonzero(edges[start:end], axis=1) / max(width, 1)
    if row_strength.size == 0:
        return None, 0.0
    relative = int(np.argmax(row_strength))
    strength = float(row_strength[relative])
    if strength < 0.06:
        return None, strength
    return float(start + relative), _clamp(strength / 0.35)


def _quality_level(score: float) -> str:
    if score >= 0.72:
        return "HIGH"
    if score >= 0.48:
        return "MEDIUM"
    if score > 0:
        return "LOW"
    return "UNAVAILABLE"


def _registration_role(quality: str, status: str) -> str:
    if status != "AVAILABLE":
        return "DO_NOT_USE"
    return {
        "HIGH": "PRIMARY_ANCHOR",
        "MEDIUM": "SECONDARY_ANCHOR",
        "LOW": "VALIDATION_ONLY",
        "UNAVAILABLE": "DO_NOT_USE",
    }[quality]


def _available_point(
    semantic_id: str,
    point: PixelPoint,
    *,
    confidence: float,
    uncertainty: float,
    method: str,
    evidence: list[str],
    frames: list[int],
) -> WicketLandmarkObservation:
    quality = _quality_level(confidence)
    return WicketLandmarkObservation(
        semantic_id=semantic_id,
        geometry_type="POINT",
        pixel_x=point.x,
        pixel_y=point.y,
        confidence=confidence,
        uncertainty_px=uncertainty,
        extraction_method=method,
        supporting_evidence=evidence,
        supporting_frames=frames,
        registration_role=_registration_role(quality, "AVAILABLE"),
        quality=quality,
        status="AVAILABLE",
    )


def _available_line(
    semantic_id: str,
    start: PixelPoint,
    end: PixelPoint,
    *,
    confidence: float,
    uncertainty: float,
    method: str,
    evidence: list[str],
    frames: list[int],
) -> WicketLandmarkObservation:
    quality = _quality_level(confidence)
    return WicketLandmarkObservation(
        semantic_id=semantic_id,
        geometry_type="LINE",
        line=WicketLineObservation(start=start, end=end),
        confidence=confidence,
        uncertainty_px=uncertainty,
        extraction_method=method,
        supporting_evidence=evidence,
        supporting_frames=frames,
        registration_role=_registration_role(quality, "AVAILABLE"),
        quality=quality,
        status="AVAILABLE",
    )


def _unavailable(
    semantic_id: str, *, geometry_type: str, reason: str
) -> WicketLandmarkObservation:
    return WicketLandmarkObservation(
        semantic_id=semantic_id,
        geometry_type=geometry_type,
        confidence=0,
        uncertainty_px=0,
        extraction_method="not_extracted",
        registration_role="DO_NOT_USE",
        quality="UNAVAILABLE",
        status="UNAVAILABLE",
        rejection_reason=reason,
    )


def extract_wicket_landmarks(
    roi_image: np.ndarray,
    roi: RoiMetadata,
    *,
    region: WicketRegionObservation,
) -> tuple[
    list[WicketLandmarkObservation],
    list[WicketLandmarkObservation],
    str,
    dict[str, float],
]:
    """Extract line-supported landmarks; missing evidence remains unavailable."""
    variants = preprocess_roi(roi_image)
    edges = variants["canny_edges"]
    x_clusters, vertical_lines, vertical_strength = _vertical_clusters(edges)
    top_y, top_strength = _horizontal_boundary(edges, upper=True)
    base_y, base_strength = _horizontal_boundary(edges, upper=False)
    roi_height, roi_width = edges.shape
    usable_clusters = [
        x for x in x_clusters if 0.04 * roi_width <= x <= 0.96 * roi_width
    ]
    coarse: list[WicketLandmarkObservation] = []
    frame_ids = region.supporting_frame_ids
    resolution_score = _clamp(min(roi_width / 80.0, roi_height / 100.0))
    temporal_score = _clamp(region.temporal_support / 4)
    vertical_clutter_score = (
        1.0
        if len(usable_clusters) <= 8
        else _clamp(1.0 - (len(usable_clusters) - 8) / 6)
    )
    region_clipped = (
        region.bbox.x <= 1
        or region.bbox.y <= 1
        or region.bbox.x + region.bbox.width
        >= roi.source_frame_width - 1
        or region.bbox.y + region.bbox.height
        >= roi.source_frame_height - 1
    )
    clipping_score = (
        0.45
        if region_clipped
        else 1.0
    )
    uncertainty_score = _clamp(
        1.0
        - region.uncertainty_px / max(region.width, region.height, 1)
    )
    confidence = _clamp(
        0.30 * region.detector_confidence
        + 0.22 * temporal_score
        + 0.18 * resolution_score
        + 0.10 * vertical_strength
        + 0.06 * top_strength
        + 0.06 * base_strength
        + 0.04 * vertical_clutter_score
        + 0.02 * clipping_score
        + 0.02 * uncertainty_score
    )
    if region_clipped or uncertainty_score < 0.75:
        confidence = min(confidence, 0.47)
    uncertainty = max(
        1.5,
        region.uncertainty_px + (1.0 - resolution_score) * 8.0,
    )
    # Three stump edges commonly produce six lines; bails and duplicate edge
    # responses can add two more. Beyond that, scene structure dominates.
    has_axes = 2 <= len(usable_clusters) <= 8
    has_top = has_axes and top_y is not None
    has_base = has_axes and base_y is not None
    evidence = [
        f"vertical_line_clusters={len(usable_clusters)}",
        f"vertical_strength={vertical_strength:.3f}",
        f"temporal_support={region.temporal_support}",
        f"native_roi={roi_width}x{roi_height}",
    ]
    left_x = usable_clusters[0] if has_axes else None
    right_x = usable_clusters[-1] if has_axes else None

    point_values: dict[str, PixelPoint | None] = {
        "wicket_outer_left_base": (
            roi_to_native((left_x, base_y), roi)
            if left_x is not None and base_y is not None
            else None
        ),
        "wicket_outer_right_base": (
            roi_to_native((right_x, base_y), roi)
            if right_x is not None and base_y is not None
            else None
        ),
        "wicket_base_center": (
            roi_to_native(((left_x + right_x) / 2, base_y), roi)
            if has_base
            else None
        ),
        "wicket_outer_left_top": (
            roi_to_native((left_x, top_y), roi)
            if left_x is not None and top_y is not None
            else None
        ),
        "wicket_outer_right_top": (
            roi_to_native((right_x, top_y), roi)
            if right_x is not None and top_y is not None
            else None
        ),
        "wicket_top_center": (
            roi_to_native(((left_x + right_x) / 2, top_y), roi)
            if has_top
            else None
        ),
        "wicket_center": (
            roi_to_native(
                ((left_x + right_x) / 2, (top_y + base_y) / 2), roi
            )
            if has_top and has_base
            else None
        ),
    }
    for semantic_id in COARSE_POINT_IDS:
        point = point_values[semantic_id]
        if point is None:
            missing = (
                "insufficient_vertical_axis_evidence"
                if not has_axes
                else "top_boundary_unavailable"
                if "top" in semantic_id
                else "base_boundary_unavailable"
            )
            coarse.append(
                _unavailable(semantic_id, geometry_type="POINT", reason=missing)
            )
        else:
            boundary_strength = (
                top_strength if "top" in semantic_id else base_strength
            )
            coarse.append(
                _available_point(
                    semantic_id,
                    point,
                    confidence=_clamp(confidence * (0.75 + 0.25 * boundary_strength)),
                    uncertainty=uncertainty,
                    method="roi_hough_vertical_axes_and_edge_boundary",
                    evidence=evidence,
                    frames=frame_ids,
                )
            )

    line_values: dict[
        str, tuple[PixelPoint, PixelPoint, float] | None
    ] = {
        "base_line": (
            roi_to_native((left_x, base_y), roi),
            roi_to_native((right_x, base_y), roi),
            base_strength,
        )
        if has_base
        else None,
        "top_or_bail_line": (
            roi_to_native((left_x, top_y), roi),
            roi_to_native((right_x, top_y), roi),
            top_strength,
        )
        if has_top
        else None,
        "left_outer_axis": (
            roi_to_native((left_x, top_y), roi),
            roi_to_native((left_x, base_y), roi),
            vertical_strength,
        )
        if has_top and has_base
        else None,
        "right_outer_axis": (
            roi_to_native((right_x, top_y), roi),
            roi_to_native((right_x, base_y), roi),
            vertical_strength,
        )
        if has_top and has_base
        else None,
    }
    for semantic_id in COARSE_LINE_IDS:
        value = line_values[semantic_id]
        if value is None:
            coarse.append(
                _unavailable(
                    semantic_id,
                    geometry_type="LINE",
                    reason="required_axes_or_boundary_unavailable",
                )
            )
        else:
            start, end, support = value
            coarse.append(
                _available_line(
                    semantic_id,
                    start,
                    end,
                    confidence=_clamp(confidence * (0.7 + 0.3 * support)),
                    uncertainty=uncertainty,
                    method="roi_hough_line_consensus",
                    evidence=evidence,
                    frames=frame_ids,
                )
            )

    detailed: list[WicketLandmarkObservation] = []
    exactly_three = len(usable_clusters) == 3
    spacings = (
        [
            usable_clusters[1] - usable_clusters[0],
            usable_clusters[2] - usable_clusters[1],
        ]
        if exactly_three
        else []
    )
    spacing_ratio = (
        min(spacings) / max(spacings) if spacings and max(spacings) > 0 else 0
    )
    detailed_supported = (
        exactly_three
        and spacing_ratio >= 0.55
        and has_top
        and has_base
        and roi_width >= 36
        and roi_height >= 54
        and vertical_strength >= 0.30
    )
    if detailed_supported:
        for level, y_value in (("base", base_y), ("top", top_y)):
            for side, x_value in zip(
                ("left", "middle", "right"), usable_clusters
            ):
                detailed.append(
                    _available_point(
                        f"{side}_stump_{level}",
                        roi_to_native((x_value, y_value), roi),
                        confidence=_clamp(confidence * spacing_ratio),
                        uncertainty=uncertainty + 1,
                        method="three_parallel_axis_spacing_consensus",
                        evidence=evidence
                        + [f"adjacent_spacing_ratio={spacing_ratio:.3f}"],
                        frames=frame_ids,
                    )
                )
        detailed_status = "AVAILABLE"
    else:
        reason = (
            "native_roi_resolution_insufficient"
            if roi_width < 36 or roi_height < 54
            else "three_parallel_stump_axes_not_supported"
            if not exactly_three
            else "stump_spacing_or_boundary_evidence_insufficient"
        )
        detailed = [
            _unavailable(item, geometry_type="POINT", reason=reason)
            for item in DETAILED_POINT_IDS
        ]
        detailed_status = "INSUFFICIENT_EVIDENCE"
    factors = {
        "detector_confidence": round(region.detector_confidence, 6),
        "temporal_stability": round(temporal_score, 6),
        "roi_resolution": round(resolution_score, 6),
        "vertical_line_consistency": round(vertical_strength, 6),
        "vertical_clutter": round(vertical_clutter_score, 6),
        "top_support": round(top_strength, 6),
        "base_support": round(base_strength, 6),
        "spacing_plausibility": round(spacing_ratio, 6),
        "frame_edge_clipping": round(clipping_score, 6),
        "consensus_uncertainty": round(uncertainty_score, 6),
    }
    return coarse, detailed, detailed_status, factors


def _save_variant(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _draw_overlay(
    frame: np.ndarray,
    *,
    detections: Sequence[RawWicketDetection],
    observations: Sequence[WicketObservation],
    output_path: Path,
) -> None:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    for item in detections:
        box = item.bbox
        draw.rectangle(
            (box.x, box.y, box.x + box.width, box.y + box.height),
            outline=(210, 210, 210, 180),
            width=2,
        )
    for observation in observations:
        region = observation.region
        colour = (
            (55, 210, 255, 255)
            if region.perspective_role == "NEAR_WICKET_CANDIDATE"
            else (255, 190, 70, 255)
        )
        box = region.bbox
        draw.rectangle(
            (box.x, box.y, box.x + box.width, box.y + box.height),
            outline=colour,
            width=4,
        )
        draw.text(
            (box.x + 3, max(0, box.y - 15)),
            f"{region.perspective_role} {region.stability}",
            fill=colour,
        )
        for landmark in observation.coarse_landmarks + observation.detailed_landmarks:
            if landmark.status != "AVAILABLE":
                continue
            anchor_colour = {
                "PRIMARY_ANCHOR": (120, 235, 110, 255),
                "SECONDARY_ANCHOR": (255, 210, 80, 255),
                "VALIDATION_ONLY": (255, 140, 80, 255),
                "DO_NOT_USE": (220, 80, 80, 255),
            }[landmark.registration_role]
            if landmark.geometry_type == "POINT":
                x, y = float(landmark.pixel_x), float(landmark.pixel_y)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=anchor_colour)
            elif landmark.line is not None:
                draw.line(
                    (
                        landmark.line.start.x,
                        landmark.line.start.y,
                        landmark.line.end.x,
                        landmark.line.end.y,
                    ),
                    fill=anchor_colour,
                    width=2,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path, "JPEG", quality=92)


def _read_sampled_frames(
    video_path: Path, frame_count: int, fps: float
) -> list[FrameEvidence]:
    indices = sample_frame_indices(frame_count)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoAnalysisServiceError(
            "Could not open the clean source video for wicket observation.",
            status_code=422,
        )
    evidence: list[FrameEvidence] = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                continue
            sharpness, brightness, obstruction = frame_quality_metrics(frame)
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detector_result = detect_wickets_robust(rgb, enable_roi=True)
            evidence.append(
                FrameEvidence(
                    index=index,
                    timestamp=index / max(fps, 1e-6),
                    frame=frame,
                    sharpness=sharpness,
                    brightness=brightness,
                    obstruction=obstruction,
                    detector_result=detector_result,
                )
            )
    finally:
        capture.release()
    return evidence


def _assignment_hypotheses() -> list[AssignmentHypothesis]:
    shared = [
        "Perspective near/far is inferred from apparent position and size only.",
        "No trusted delivery-direction or camera-setup metadata finalises the ends.",
    ]
    return [
        AssignmentHypothesis(
            hypothesis_id="A",
            near_semantic_end="bowler",
            far_semantic_end="striker",
            confidence=0.5,
            evidence=shared,
        ),
        AssignmentHypothesis(
            hypothesis_id="B",
            near_semantic_end="striker",
            far_semantic_end="bowler",
            confidence=0.5,
            evidence=shared,
        ),
    ]


def _detector_labels(evidence: Sequence[FrameEvidence]) -> list[str]:
    return sorted(
        {
            str(candidate.get("class_name"))
            for item in evidence
            for candidate in item.detector_result.get("candidates") or []
            if candidate.get("class_name")
        }
    )


def _raw_detections(
    evidence: Sequence[FrameEvidence],
) -> list[RawWicketDetection]:
    output: list[RawWicketDetection] = []
    for item in evidence:
        for candidate in item.detector_result.get("candidates") or []:
            output.append(
                RawWicketDetection(
                    frame_index=item.index,
                    timestamp_seconds=item.timestamp,
                    bbox=PixelBox(**_box_dict(candidate)),
                    confidence=float(candidate.get("confidence") or 0),
                    class_name=str(candidate.get("class_name") or "unknown"),
                    source=str(candidate.get("source") or "unknown"),
                    detector_model=STUMP_MODEL_RELATIVE_PATH,
                    perspective_role="UNRESOLVED_WICKET",
                )
            )
    return output


def _observation_for_region(
    selected_frame: np.ndarray,
    region: WicketRegionObservation,
    *,
    debug_dir: Path,
    analysis_id: str,
    short_role: str,
) -> WicketObservation:
    roi_image, roi = build_native_roi(selected_frame, region.bbox)
    variants = preprocess_roi(roi_image)
    debug_urls: dict[str, str] = {}
    for name, image in variants.items():
        filename = f"{short_role}_roi_{name}.png"
        _save_variant(debug_dir / filename, image)
        debug_urls[name] = (
            f"/static/video-analysis/{analysis_id}/calibration/"
            f"{DEBUG_DIRECTORY}/{filename}"
        )
    coarse, detailed, detailed_status, factors = extract_wicket_landmarks(
        roi_image, roi, region=region
    )
    available = [
        item
        for item in coarse + detailed
        if item.status == "AVAILABLE"
    ]
    quality_score = (
        sum(item.confidence for item in available) / len(available)
        if available
        else 0.0
    )
    warnings: list[str] = []
    if detailed_status != "AVAILABLE":
        warnings.append(
            "Individual stump landmarks were withheld because image evidence "
            "did not support three distinct stump axes."
        )
    if region.stability == "UNSTABLE":
        warnings.append("Region is not temporally stable.")
    return WicketObservation(
        region=region,
        roi=roi,
        coarse_landmarks=coarse,
        detailed_landmarks=detailed,
        detailed_landmarks_status=detailed_status,
        quality_score=quality_score,
        quality_factors=factors,
        warnings=warnings,
        roi_debug_urls=debug_urls,
    )


def _result_status(
    near: WicketObservation | None,
    far: WicketObservation | None,
) -> str:
    observations = [item for item in (near, far) if item is not None]
    if not observations:
        return "INSUFFICIENT_WICKETS"
    if len(observations) < 2:
        return "PARTIAL"
    if any(item.region.stability == "UNSTABLE" for item in observations):
        return "UNSTABLE"
    usable_by_wicket = [
        [
            landmark
            for landmark in item.coarse_landmarks
            if landmark.status == "AVAILABLE"
            and landmark.registration_role
            in ("PRIMARY_ANCHOR", "SECONDARY_ANCHOR")
        ]
        for item in observations
    ]
    if any(
        len(usable) < 3 or item.quality_score < 0.48
        for item, usable in zip(observations, usable_by_wicket)
    ):
        return "INSUFFICIENT_LANDMARKS"
    return "READY_FOR_REGISTRATION_EXPERIMENT"


def run_wicket_observation(analysis_id: str) -> WicketObservationResult:
    analysis = load_video_analysis(analysis_id)
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    raw_path = analysis_dir / "raw" / analysis.stored_filename
    evidence = _read_sampled_frames(raw_path, analysis.frame_count, analysis.fps)
    frame_candidates = score_setup_frames(evidence)
    selected_candidate = next(
        (item for item in frame_candidates if item.selected), None
    )
    debug_dir = analysis_dir / "calibration" / DEBUG_DIRECTORY
    reports_dir = analysis_dir / "reports"
    debug_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw = _raw_detections(evidence)
    rejected = [
        {
            "frame_index": item.index,
            **rejection,
        }
        for item in evidence
        for rejection in (
            item.detector_result.get("diagnostics", {}).get("rejected") or []
        )
    ]
    base_diagnostics = dict(
        detector_model_path=STUMP_MODEL_RELATIVE_PATH,
        detector_class_labels=_detector_labels(evidence),
        clean_source_video=(
            f"/static/video-analysis/{analysis_id}/raw/{analysis.stored_filename}"
        ),
        sampled_frame_ids=[item.index for item in evidence],
        raw_detections=raw,
        rejected_detections=rejected,
        result_json_url=(
            f"/static/video-analysis/{analysis_id}/reports/{RESULT_FILENAME}"
        ),
    )
    if selected_candidate is None:
        result = WicketObservationResult(
            analysis_id=analysis_id,
            status="FAILED",
            setup_frame=None,
            frame_candidates=frame_candidates,
            assignment_hypotheses=_assignment_hypotheses(),
            warnings=["No readable setup frame was available."],
            diagnostics=WicketObservationDiagnostics(**base_diagnostics),
            future_registration_readiness="FAILED",
            message="Real wicket observation failed before landmark extraction.",
        )
        _write_result(reports_dir, result)
        return result

    selected_evidence = next(
        item for item in evidence if item.index == selected_candidate.frame_index
    )
    setup_filename = f"setup_frame_{selected_candidate.frame_index:06d}.jpg"
    cv2.imwrite(str(debug_dir / setup_filename), selected_evidence.frame)
    setup_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{DEBUG_DIRECTORY}/{setup_filename}"
    )
    region_candidates = build_temporal_region_candidates(
        raw,
        selected_frame_index=selected_candidate.frame_index,
        fps=analysis.fps,
    )
    near_region, far_region, unresolved_regions = select_near_far_regions(
        region_candidates,
        frame_width=selected_candidate.image_width,
        frame_height=selected_candidate.image_height,
    )
    near = (
        _observation_for_region(
            selected_evidence.frame,
            near_region,
            debug_dir=debug_dir,
            analysis_id=analysis_id,
            short_role="near",
        )
        if near_region
        else None
    )
    far = (
        _observation_for_region(
            selected_evidence.frame,
            far_region,
            debug_dir=debug_dir,
            analysis_id=analysis_id,
            short_role="far",
        )
        if far_region
        else None
    )
    observations = [item for item in (near, far) if item is not None]
    overlay_filename = "landmark_overlay.jpg"
    selected_raw = [
        item for item in raw if item.frame_index == selected_candidate.frame_index
    ]
    _draw_overlay(
        selected_evidence.frame,
        detections=selected_raw,
        observations=observations,
        output_path=debug_dir / overlay_filename,
    )
    overlay_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{DEBUG_DIRECTORY}/{overlay_filename}"
    )
    status = _result_status(near, far)
    warnings = [
        "Near/far is perspective-only; bowler/striker end assignment is unresolved.",
        "Obstruction is a visual-clutter heuristic, not person segmentation.",
    ]
    if near is None or far is None:
        warnings.append("Only one or no wicket regions reached temporal consensus.")
    supporting = sorted(
        [
            item
            for item in frame_candidates
            if not item.selected
            and item.wicket_detection_count > 0
            and abs(item.frame_index - selected_candidate.frame_index) <= 9
        ],
        key=lambda item: (-item.score, item.frame_index),
    )[:4]
    result = WicketObservationResult(
        analysis_id=analysis_id,
        status=status,
        setup_frame=selected_candidate,
        supporting_frames=supporting,
        frame_candidates=frame_candidates,
        near_wicket=near,
        far_wicket=far,
        unresolved_regions=unresolved_regions,
        assignment_hypotheses=_assignment_hypotheses(),
        warnings=warnings,
        diagnostics=WicketObservationDiagnostics(
            **base_diagnostics,
            setup_frame_image_url=setup_url,
            raw_detection_overlay_url=overlay_url,
            landmark_overlay_url=overlay_url,
        ),
        future_registration_readiness=status,
        message=(
            "Real wicket observations are ready for a future registration experiment."
            if status == "READY_FOR_REGISTRATION_EXPERIMENT"
            else "Real wicket observations are partial or insufficient for registration."
        ),
    )
    _write_result(reports_dir, result)
    return result


def _write_result(reports_dir: Path, result: WicketObservationResult) -> None:
    destination = reports_dir / RESULT_FILENAME
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_wicket_observation(analysis_id: str) -> WicketObservationResult:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.is_file():
        raise VideoAnalysisServiceError(
            "Wicket observations have not been generated for this analysis.",
            status_code=404,
        )
    try:
        return WicketObservationResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored wicket observations are unavailable.",
            status_code=500,
        ) from exc
