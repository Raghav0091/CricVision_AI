"""Lightweight tests for Accuracy / Small Ball ball-tracking mode."""

import importlib
import tempfile
from pathlib import Path

from Backends.src.engine.engine_options import (
    ACCURACY_BALL_CONFIDENCE,
    ACCURACY_BALL_IMGSZ,
    EngineOptions,
    resolve_ball_tracking_settings,
)
from Backends.src.tracking.trajectory_scorer import (
    TrajectoryBallSelector,
    resolve_delivery_tracking_quality,
)


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
