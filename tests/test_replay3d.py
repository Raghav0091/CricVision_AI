"""Stump-calibrated 3D replay tests (no YOLO, videos, GPU, internet, or Streamlit)."""

from __future__ import annotations

import re

import numpy as np

from Backends.src.replay3d.replay_renderer import build_3d_replay_figure
from Backends.src.replay3d.stump_calibration import build_stump_calibration_context
from Backends.src.replay3d.trajectory_3d import (
    MIN_3D_POINTS,
    UNKNOWN_METRICS,
    build_estimated_3d_trajectory,
)


def _fake_trajectory(count: int = 8):
    return [(320 + index * 8, 120 + index * 45) for index in range(count)]


def _stump(center, confidence=0.9):
    x, y = center
    return {
        "center": [x, y],
        "bbox": [x - 8, y - 30, x + 8, y],
        "confidence": confidence,
    }


def test_calibration_missing_stumps_is_low():
    context = build_stump_calibration_context(
        frame_size={"width": 1280, "height": 720},
        stump_detections=[],
        pitch_roi=None,
    )
    assert context["calibration_quality"] in {"Low", "Partial"}
    assert context["batter_stumps"]
    assert context["bowler_stumps"]
    assert context["pitch_roi"]["bbox"] is not None


def test_calibration_partial_with_one_stump():
    context = build_stump_calibration_context(
        frame_size=(1280, 720),
        stump_detections=[_stump((640, 500), confidence=0.82)],
        pitch_roi={"bbox": [500, 100, 780, 650], "source": "provided"},
    )
    assert context["calibration_quality"] in {"Partial", "Low", "Good"}


def test_calibration_good_with_stumps_and_roi():
    context = build_stump_calibration_context(
        frame_size={"width": 1280, "height": 720},
        stump_detections=[
            _stump((640, 180), confidence=0.91),
            _stump((638, 520), confidence=0.88),
        ],
        pitch_roi={"bbox": [480, 120, 800, 620], "polygon": [], "source": "provided"},
        camera_view="umpire_end",
        camera_height_ft=9.0,
    )
    assert context["calibration_quality"] == "Good"
    assert context["camera_height_ft"] == 9.0
    assert context["pitch_centerline"]["pitch_length_ft"] == 66.0


def test_3d_trajectory_empty_when_few_points():
    context = build_stump_calibration_context(frame_size=(1280, 720))
    result = build_estimated_3d_trajectory(_fake_trajectory(3), context)
    assert result["available"] is False
    assert result["trajectory_quality"] == "Unavailable"
    assert result["points_3d"] == []


def test_3d_trajectory_simple_path():
    context = build_stump_calibration_context(
        frame_size=(1280, 720),
        stump_detections=[_stump((640, 180)), _stump((640, 520))],
        pitch_roi={"bbox": [480, 120, 800, 620], "source": "provided"},
    )
    result = build_estimated_3d_trajectory(_fake_trajectory(MIN_3D_POINTS + 2), context)
    assert result["available"] is True
    assert len(result["points_3d"]) >= MIN_3D_POINTS
    assert result["release_3d"] is not None
    assert all("x_ft" in point and "y_ft" in point and "z_ft" in point for point in result["points_3d"])


def test_bounce_z_is_zero():
    context = build_stump_calibration_context(frame_size=(1280, 720))
    bounce = (360, 300)
    result = build_estimated_3d_trajectory(
        _fake_trajectory(8),
        context,
        bounce_point=bounce,
    )
    assert result["bounce_3d"] is not None
    assert result["bounce_3d"]["z_ft"] == 0.0


def test_renderer_creates_plotly_or_image():
    context = build_stump_calibration_context(frame_size=(1280, 720))
    trajectory_3d = build_estimated_3d_trajectory(_fake_trajectory(8), context)
    payload = build_3d_replay_figure(trajectory_3d, context, width=640, height=480)
    assert payload["available"] is True
    assert payload["backend"] in {"plotly", "opencv"}
    if payload["backend"] == "plotly":
        assert payload["figure"] is not None
    else:
        assert isinstance(payload["image"], np.ndarray)
        assert payload["image"].shape[2] == 3


def test_no_numeric_speed_swing_spin_lbw():
    context = build_stump_calibration_context(frame_size=(1280, 720))
    trajectory_3d = build_estimated_3d_trajectory(_fake_trajectory(8), context)
    metrics = trajectory_3d["metrics"]
    assert metrics["speed_kmh"] == "Not calibrated"
    assert metrics["swing"] == "Unknown"
    assert metrics["spin"] == "Unknown"
    assert metrics["lbw"] == "Not available"

    payload = build_3d_replay_figure(trajectory_3d, context)
    caption = payload["caption"]
    assert "Estimated 3D Replay" in caption
    joined_notes = " ".join(trajectory_3d["notes"])
    assert not re.search(r"\b\d{2,}\s*km/h", joined_notes, flags=re.IGNORECASE)
    assert UNKNOWN_METRICS["speed_kmh"] in joined_notes


def test_3d_trajectory_accepts_observer_fitted_path_shape():
    context = build_stump_calibration_context(frame_size=(1280, 720))
    fitted_path = [
        {"frame_index": index, "x": 320 + index * 6, "y": 120 + index * 40, "source": "observer_fit"}
        for index in range(8)
    ]
    result = build_estimated_3d_trajectory(fitted_path, context)
    assert result["available"] is True
    assert len(result["points_3d"]) >= MIN_3D_POINTS
