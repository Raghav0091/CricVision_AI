"""Lightweight tests for Accuracy / Small Ball ball-tracking mode."""

import importlib
import inspect
import tempfile
from pathlib import Path

import numpy as np
import pytest

from Backends.src.engine.engine_options import (
    ACCURACY_BALL_CONFIDENCE,
    ACCURACY_BALL_IMGSZ,
    EngineOptions,
    resolve_ball_tracking_settings,
)
from Backends.src.tracking.trajectory_scorer import (
    TrajectoryBallSelector,
    resolve_delivery_tracking_quality,
    should_enable_online_best_tracklet,
)
from Backends.src.video_pipeline.annotation_writer import (
    add_delivery_trajectory_overlay_to_video,
    delivery_overlay_metrics,
    draw_fitted_trajectory_overlay,
)

pytest.importorskip("cv2")


def _ball(center, confidence=0.8):
    x, y = center
    return {
        "center": center,
        "confidence": confidence,
        "box": (x - 4, y - 4, x + 4, y + 4),
        "class_name": "ball",
    }


def _moving_track(selector, *, count=10, start_frame=0):
    previous = None
    for index in range(count):
        center = (100 + index * 12, 120 + index * 8)
        chosen = selector.select(
            [_ball(center, 0.8)],
            previous_center=previous,
            frame_index=start_frame + index,
        )
        if chosen is not None:
            previous = chosen["center"]
    return previous


def test_accuracy_mode_options_reach_delivery_processor(monkeypatch):
    engine_module = importlib.import_module(
        "Backends.src.engine.analyze_delivery"
    )
    calls = []

    class FakeCapture:
        def release(self):
            return None

    def fake_processor(path, context, options, output_path):
        calls.append(options)
        return {
            "success": True,
            "processed_video_generated": False,
            "ball_tracking_mode": options.ball_tracking_mode,
        }

    monkeypatch.setattr(engine_module, "open_video", lambda path: FakeCapture())
    monkeypatch.setattr(engine_module, "_run_processor", fake_processor)

    with tempfile.TemporaryDirectory(
        dir=Path(__file__).resolve().parent
    ) as temp_dir:
        video_path = Path(temp_dir) / "delivery.mp4"
        video_path.write_bytes(b"synthetic placeholder")
        result = engine_module.analyze_delivery_clip(
            video_path,
            options=EngineOptions(
                ball_tracking_mode="Accuracy / Small Ball",
                processed_video_enabled=False,
            ),
        )

    assert result["success"] is True
    assert len(calls) == 1
    assert calls[0].ball_tracking_mode == "Accuracy / Small Ball"

def test_accuracy_mode_uses_lower_confidence_and_larger_imgsz():
    settings = resolve_ball_tracking_settings(
        "Accuracy / Small Ball",
        calibration_context={"enabled": True, "calibration_quality": "Good"},
        preset_confidence=0.25,
        preset_image_size=640,
        speed_use_roi=True,
    )

    assert settings["confidence_threshold"] == ACCURACY_BALL_CONFIDENCE
    assert settings["image_size"] == ACCURACY_BALL_IMGSZ
    assert settings["ball_candidate_confidence"] == ACCURACY_BALL_CONFIDENCE


def test_accuracy_mode_avoids_roi_when_calibration_weak_or_disabled():
    disabled = resolve_ball_tracking_settings(
        "Accuracy / Small Ball",
        calibration_context={
            "enabled": False,
            "calibration_quality": "Disabled",
        },
        speed_use_roi=True,
    )
    low = resolve_ball_tracking_settings(
        "Accuracy / Small Ball",
        calibration_context={
            "enabled": True,
            "calibration_quality": "Low",
        },
        speed_use_roi=True,
    )

    assert disabled["use_roi"] is False
    assert disabled["full_frame_roi_mode"] == "full_frame_no_roi"
    assert low["use_roi"] is False
    assert low["full_frame_roi_mode"] == "full_frame_no_roi"


def test_accuracy_mode_keeps_roi_when_calibration_is_strong():
    settings = resolve_ball_tracking_settings(
        "Accuracy / Small Ball",
        calibration_context={
            "enabled": True,
            "calibration_quality": "Good",
        },
        speed_use_roi=True,
    )

    assert settings["use_roi"] is True
    assert settings["full_frame_roi_mode"] == "roi_enabled"


def test_short_valid_track_reports_partial():
    selector = TrajectoryBallSelector(640, 360)
    _moving_track(selector, count=10, start_frame=62)

    quality, suppress = resolve_delivery_tracking_quality(selector)
    summary = selector.debug_summary(quality)

    assert quality == "Partial"
    assert suppress is True
    assert summary["short_track_reason"] is not None


def test_balanced_mode_remains_compatible():
    settings = resolve_ball_tracking_settings(
        "Balanced",
        calibration_context={"enabled": False, "calibration_quality": "Disabled"},
        preset_confidence=0.25,
        preset_image_size=768,
        speed_use_roi=True,
    )
    options = EngineOptions()

    assert settings["ball_tracking_mode"] == "Balanced"
    assert settings["confidence_threshold"] == 0.25
    assert settings["image_size"] == 768
    assert settings["ball_candidate_confidence"] is None
    assert options.ball_tracking_mode == "Balanced"


def test_accuracy_mode_enables_online_best_tracklet_ranking():
    assert should_enable_online_best_tracklet(
        ball_tracking_mode="Accuracy / Small Ball",
        speed_mode="Smart Balanced",
    )
    assert not should_enable_online_best_tracklet(
        ball_tracking_mode="Balanced",
        speed_mode="Smart Balanced",
    )


def test_video_analysis_wires_preset_settings_into_process_video():
    # ponytail: UI owns process_video directly; presets supply confidence/imgsz
    # instead of EngineOptions(ball_tracking_mode=...).
    video_analysis = importlib.import_module("Backends.src.ui.video_analysis")
    source = inspect.getsource(video_analysis.show_video_analysis_page)
    assert "result = process_video(" in source
    assert "confidence=confidence" in source
    assert "imgsz=image_size" in source
    assert "DETECTION_PRESETS" in inspect.getsource(video_analysis)


def test_delivery_result_tracking_fields_present():
    delivery = importlib.import_module(
        "Backends.src.engine.processors.delivery"
    )
    source = inspect.getsource(delivery.process_delivery_video)
    for field_name in (
        "best_tracklet_applied",
        "best_segment_start_frame",
        "best_segment_end_frame",
        "best_segment_point_count",
        "selected_ball_points",
        "trajectory_fit_quality",
        "trajectory_visualization_mode",
        "tracking_quality",
        "extension_applied",
        "extension_fallback_reason",
        "ball_tracking_mode",
    ):
        assert field_name in source


def test_delivery_overlay_metrics_unknown_when_calibration_disabled():
    line, length, bounce = delivery_overlay_metrics(
        {"enabled": False, "calibration_quality": "Disabled"},
        line="Middle",
        length="Good Length",
        bounce_point=(100, 200),
    )
    assert line == "Unknown"
    assert length == "Unknown"
    assert bounce == "Unknown"


def test_clean_overlay_does_not_draw_rejection_labels():
    frame = np.zeros((120, 220, 3), dtype=np.uint8)
    draw_fitted_trajectory_overlay(
        frame,
        observed_points=[(20, 70), (35, 65)],
        fitted_points=[(20, 70), (35, 65), (50, 60)],
        visualization_mode="full_fit",
        trajectory_quality="Good",
        fit_quality="Good",
        tracking_quality="Partial",
        line="Unknown",
        length="Unknown",
        calibration_context={"enabled": False},
    )
    assert b"rejected" not in frame.tobytes()
    assert b"static" not in frame.tobytes()


def test_add_delivery_trajectory_overlay_rewrites_video(tmp_path):
    output_path = tmp_path / "delivery.mp4"
    from Backends.src.video_pipeline.annotation_writer import write_annotated_video

    result = write_annotated_video(
        [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(3)],
        output_path,
        fps=12,
    )
    if result is None:
        pytest.skip("OpenCV mp4v writer is unavailable.")

    add_delivery_trajectory_overlay_to_video(
        output_path,
        trajectory_fit_result={
            "fitted_trajectory_points": [(20, 60), (40, 64), (60, 68)],
            "observed_trajectory_points": [(20, 60), (40, 64)],
            "trajectory_visualization_mode": "full_fit",
            "trajectory_fit_quality": "Good",
        },
        overall_tracking_quality="Partial",
        estimated_line="Unknown",
        estimated_length="Unknown",
        calibration_context={"enabled": False},
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
