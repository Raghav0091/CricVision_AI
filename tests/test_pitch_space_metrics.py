from __future__ import annotations

import pytest

from services.api.schemas.video_analysis import PrimaryBounceResult
from services.api.services.pitch_space_bounce_service import estimate_pitch_space_bounce
from services.api.services.pitch_space_metrics_service import (
    calculate_line_and_length,
    estimate_lateral_movement,
    estimate_planar_speed,
)


def point(frame: int, x: float, y: float, *, confidence: float = 0.9, provenance: str = "OBSERVED") -> dict:
    return {
        "frame_index": frame, "timestamp_seconds": frame / 25, "image_x_px": 300 + frame,
        "image_y_px": 200 + frame, "pitch_x_m": x, "pitch_y_m": y,
        "combined_confidence": confidence, "provenance": provenance,
    }


def test_existing_bounce_is_reused_and_mapped_to_pitch_space() -> None:
    track = [point(i, 0.02, i * 0.8) for i in range(8)]
    bounce = PrimaryBounceResult(
        bounce_detected=True, bounce_frame=4, bounce_timestamp_seconds=0.16,
        bounce_x=304, bounce_y=204, confidence=0.8, evidence=["existing_image_bounce"],
    )
    result = estimate_pitch_space_bounce(track, existing_bounce=bounce)
    assert result.bounce_frame == 4
    assert result.pitch_y_m == pytest.approx(3.2)
    assert "source:existing_bounce" in result.evidence


def test_bounce_is_not_forced_without_existing_or_strong_local_evidence() -> None:
    result = estimate_pitch_space_bounce([point(i, 0, i) for i in range(7)])
    assert result.status == "UNAVAILABLE"
    assert result.bounce_frame is None


def test_line_and_length_use_virtual_pitch_dimensions() -> None:
    result = calculate_line_and_length({"status": "DETECTED", "pitch_x_m": -0.2,
                                        "pitch_y_m": 14.12, "confidence": 0.8})
    assert result.line == "PITCH_LEFT"
    assert result.length == "GOOD_LENGTH"
    assert result.distance_from_striker_wicket_m == pytest.approx(6.0)
    assert result.distance_from_striker_popping_crease_m == pytest.approx(4.78)


def test_planar_speed_uses_robust_multi_point_regression() -> None:
    track = [point(i, 0.0, 1.0 + 25.0 * (i / 25)) for i in range(8)]
    track[3]["pitch_y_m"] += 2.5
    result = estimate_planar_speed(track)
    assert result.status == "AVAILABLE"
    assert result.speed_mps == pytest.approx(25.0, abs=0.1)
    assert result.speed_kmh == pytest.approx(90.0, abs=0.4)
    assert len(result.frames_used) >= 7
    assert result.label == "ESTIMATED_PLANAR_SPEED"


def test_speed_rejects_single_pair_and_projected_only_tracks() -> None:
    assert estimate_planar_speed([point(0, 0, 0), point(1, 0, 1)]).status == "UNAVAILABLE"
    projected = [point(i, 0, i, provenance="PROJECTED") for i in range(8)]
    assert estimate_planar_speed(projected).status == "UNAVAILABLE"


def test_lateral_movement_uses_early_direction_and_rejects_outlier() -> None:
    track = []
    for i in range(10):
        y = float(i)
        deviation = 0.0 if i < 5 else 0.04 * (i - 4)
        track.append(point(i, 0.01 * y + deviation, y))
    track[6]["pitch_x_m"] = 2.0
    result = estimate_lateral_movement(track)
    assert result.status == "AVAILABLE"
    assert result.direction == "PITCH_RIGHT"
    assert result.movement_m == pytest.approx(0.18, abs=0.08)
    assert 6 not in result.frames_used
    assert result.label == "ESTIMATED_LATERAL_MOVEMENT"


def test_partial_unavailable_metrics_remain_independent() -> None:
    assert calculate_line_and_length(None).status == "UNAVAILABLE"
    assert estimate_lateral_movement([point(i, 0, i) for i in range(4)]).status == "UNAVAILABLE"

