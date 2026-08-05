"""Tests for frame timestamp resolution."""

from __future__ import annotations

import pytest

from services.api.services.frame_timing import resolve_frame_timestamp


@pytest.mark.parametrize(
    ("fps", "frame_index", "expected"),
    [
        (25.0, 50, 2.0),
        (30.0, 90, 3.0),
        (60.0, 120, 2.0),
        (120.0, 240, 2.0),
    ],
)
def test_container_fps_timestamp(fps: float, frame_index: int, expected: float) -> None:
    timestamp, method = resolve_frame_timestamp(frame_index, fps=fps)
    assert method == "CONTAINER_FPS"
    assert timestamp == pytest.approx(expected)


def test_frame_timestamps_have_first_precedence() -> None:
    timestamps = [0.0, 0.04, 0.09]
    timestamp, method = resolve_frame_timestamp(2, fps=30.0, frame_timestamps=timestamps)
    assert method == "FRAME_TIMESTAMPS"
    assert timestamp == 0.09


def test_time_base_precedes_container_fps() -> None:
    timestamp, method = resolve_frame_timestamp(10, fps=30.0, time_base=(1.0, 0.04))
    assert method == "TIME_BASE"
    assert timestamp == pytest.approx(1.4)


def test_nominal_fps_fallback_exposes_method() -> None:
    timestamp, method = resolve_frame_timestamp(30, fps=0.0)
    assert method == "NOMINAL_FPS_FALLBACK"
    assert timestamp == pytest.approx(1.0)
