from __future__ import annotations

import pytest

from services.api.schemas.video_analysis import TrackingPoint
from services.api.services.pitch_space_track_service import (
    AIRBORNE_WARNING,
    convert_track_to_pitch_space,
    project_image_point,
)


def tracked(frame: int, x: float, y: float, provenance: str = "OBSERVED") -> TrackingPoint:
    source = "recovered" if provenance == "TRACKER_RECOVERED" else provenance.lower()
    return TrackingPoint(
        frame_index=frame, timestamp_seconds=frame / 25, source=source,
        provenance=provenance, x=x, y=y, normalized_x=x / 1280, normalized_y=y / 720,
        confidence=0.81, uncertainty=0.05, vx=0, vy=0,
    )


def test_homography_conversion_preserves_image_evidence_and_provenance() -> None:
    points = convert_track_to_pitch_space(
        [tracked(2, 100, 200), tracked(3, 110, 210, "TRACKER_RECOVERED")],
        [[0.01, 0, -1], [0, 0.02, -4], [0, 0, 1]],
        pitch_fit_confidence=0.64, bounce_frame=3,
    )
    assert points[0].pitch_x_m == pytest.approx(0.0)
    assert points[0].pitch_y_m == pytest.approx(0.0)
    assert points[1].pitch_x_m == pytest.approx(0.1)
    assert points[1].pitch_y_m == pytest.approx(0.2)
    assert points[0].image_x_px == 100
    assert points[0].detection_confidence == pytest.approx(0.81)
    assert points[0].combined_confidence == pytest.approx(0.72)
    assert points[1].provenance == "TRACKER_RECOVERED"
    assert points[1].bounce_phase == "BOUNCE"
    assert AIRBORNE_WARNING in points[0].warnings


def test_mapping_input_and_invalid_points_are_handled_without_detector_work() -> None:
    points = convert_track_to_pitch_space(
        [
            {"frame_index": 1, "timestamp_seconds": 0.04, "x": 2, "y": 3,
             "confidence": 0.5, "provenance": "PROJECTED"},
            {"frame_index": 2, "timestamp_seconds": 0.08, "x": 2, "y": 3,
             "confidence": 0.5, "provenance": "OBSERVED", "track_valid": False},
        ],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], pitch_fit_confidence=1,
    )
    assert len(points) == 1
    assert points[0].provenance == "PROJECTED"
    assert any("not directly detected" in warning for warning in points[0].warnings)


def test_singular_or_malformed_projection_is_unavailable() -> None:
    assert project_image_point(1, 2, [[1, 0], [0, 1]]) is None
    assert project_image_point(1, 2, [[1, 0, 0], [0, 1, 0], [0, 0, 0]]) is None
    assert convert_track_to_pitch_space([tracked(1, 2, 3)], [[1, 0], [0, 1]], pitch_fit_confidence=1) == []


def test_camera_instability_downgrades_only_affected_points() -> None:
    points = convert_track_to_pitch_space(
        [tracked(4, 0, 1), tracked(5, 0, 2)],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]], pitch_fit_confidence=0.9,
        camera_stability="UNSTABLE_CAMERA", unstable_after_frame=5,
    )
    assert points[0].pitch_fit_confidence == pytest.approx(0.9)
    assert points[1].pitch_fit_confidence == 0
    assert any("instability" in warning for warning in points[1].warnings)
