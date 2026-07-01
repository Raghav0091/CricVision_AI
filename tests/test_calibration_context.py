"""Calibration context tests are model-free and JSON-only."""

import json

from Backends.src.calibration.calibration_context import (
    calibration_quality_label,
    default_calibration_context,
    normalize_calibration_context,
)


def test_default_calibration_context_is_json_safe():
    context = default_calibration_context()

    assert context["enabled"] is False
    assert context["calibration_quality"] == "Low"
    assert json.loads(json.dumps(context)) == context


def test_normalize_calibration_context_handles_none():
    context = normalize_calibration_context(None)

    assert context == default_calibration_context()


def test_normalize_calibration_context_handles_partial_old_shape():
    context = normalize_calibration_context(
        {
            "enabled": True,
            "camera_view": "Umpire End",
            "batter_handedness": "Right-handed",
            "calibration_score": 0.75,
            "stumps": {
                "bbox": [10, 20, 30, 60],
                "confidence": 0.7,
                "source": "auto",
            },
        }
    )

    assert context["camera_view"] == "umpire_end"
    assert context["batter_handedness"] == "right"
    assert context["calibration_quality"] == "Good"
    assert context["stumps"]["batter_end"]["center"] == [20.0, 40.0]


def test_calibration_quality_label_maps_safe_ranges():
    assert calibration_quality_label(None) == "Low"
    assert calibration_quality_label(-1) == "Low"
    assert calibration_quality_label(0.4) == "Medium"
    assert calibration_quality_label(0.7) == "Good"
    assert calibration_quality_label(0.85) == "High"
    assert calibration_quality_label("bad") == "Low"


def test_normalize_preserves_legacy_quality_label_without_score():
    context = normalize_calibration_context(
        {"enabled": True, "calibration_quality": "Good"}
    )

    assert context["calibration_quality"] == "Good"
