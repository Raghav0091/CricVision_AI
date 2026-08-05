from __future__ import annotations

import cv2
import numpy as np

from services.api.services.wicket_landmark_consensus import build_wicket_landmark_consensus
from services.api.services.wicket_landmark_extractor import extract_wicket_landmarks


def _wicket(offset_x: int = 0, *, base: bool = True) -> np.ndarray:
    image = np.zeros((180, 160, 3), np.uint8)
    xs = (48 + offset_x, 80 + offset_x, 112 + offset_x)
    for x in xs:
        cv2.line(image, (x, 25), (x, 155), (255, 255, 255), 3)
    cv2.line(image, (xs[0] - 3, 25), (xs[-1] + 3, 25), (255, 255, 255), 2)
    if base:
        cv2.line(image, (xs[0] - 3, 155), (xs[-1] + 3, 155), (255, 255, 255), 2)
    return image


def _extract(frame: int, offset: int = 0, *, base: bool = True):
    return extract_wicket_landmarks(_wicket(offset, base=base), role="near", frame_id=frame)


def test_temporal_consensus_fuses_jitter_and_reports_support() -> None:
    result = build_wicket_landmark_consensus([
        _extract(10, -1), _extract(11, 0), _extract(12, 1), _extract(13, 0)
    ])
    assert result.status == "AVAILABLE"
    middle = next(item for item in result.axes if item.semantic_id == "middle_stump_axis")
    assert middle.status == "AVAILABLE"
    assert middle.line is not None and abs(middle.line.midpoint.x - 80) < 2
    assert middle.supporting_frame_ids == (10, 11, 12, 13)
    assert middle.perpendicular_uncertainty_px is not None


def test_large_spatial_outlier_is_rejected_without_moving_consensus() -> None:
    result = build_wicket_landmark_consensus([
        _extract(1, 0), _extract(2, 1), _extract(3, -1), _extract(4, 25)
    ])
    middle = next(item for item in result.axes if item.semantic_id == "middle_stump_axis")
    assert middle.line is not None and abs(middle.line.midpoint.x - 80) < 2
    assert 4 not in middle.supporting_frame_ids


def test_missing_frames_do_not_create_temporal_evidence() -> None:
    unavailable = extract_wicket_landmarks(np.zeros((180, 160, 3), np.uint8), role="near", frame_id=2)
    result = build_wicket_landmark_consensus([_extract(1), unavailable], minimum_support=2)
    assert result.status == "UNAVAILABLE"
    assert result.supporting_frame_ids == ()


def test_partial_base_evidence_remains_unavailable_when_support_is_too_low() -> None:
    result = build_wicket_landmark_consensus([
        _extract(1, base=True), _extract(2, base=False), _extract(3, base=False)
    ], minimum_support=2)
    base = next(item for item in result.lines if item.semantic_id == "base_line")
    bail = next(item for item in result.lines if item.semantic_id == "bail_line")
    assert base.status == "UNAVAILABLE"
    assert bail.status == "AVAILABLE"


def test_consensus_keeps_near_and_far_roles_separate() -> None:
    near = _extract(1)
    far = extract_wicket_landmarks(_wicket(), role="far", frame_id=2)
    result = build_wicket_landmark_consensus([near, far], minimum_support=2)
    assert result.role == "near"
    assert result.status == "UNAVAILABLE"
    assert result.diagnostics["compatible_role_count"] == 1


def test_empty_consensus_is_honestly_unavailable() -> None:
    result = build_wicket_landmark_consensus([])
    assert result.status == "UNAVAILABLE"
    assert result.confidence == 0
    assert result.as_mapping()["supporting_frame_ids"] == []
