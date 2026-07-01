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
    assert context["calibration_quality"] == "Disabled"
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
    assert context["calibration_quality"] == "Medium"
    assert context["stumps"]["batter_end"]["center"] == [20.0, 40.0]


def test_calibration_quality_label_maps_safe_ranges():
    assert calibration_quality_label(None) == "Low"
    assert calibration_quality_label(-1) == "Low"
    assert calibration_quality_label(0.4) == "Medium"
    assert calibration_quality_label(0.7) == "Good"
    assert calibration_quality_label(0.85) == "High"
    assert calibration_quality_label("bad") == "Low"


def test_finalize_calibration_quality_caps_estimated_stump_to_medium_or_lower():
    from Backends.src.calibration.calibration_context import finalize_calibration_quality

    context = finalize_calibration_quality(
        {
            "enabled": True,
            "calibration_version": 1,
            "camera_view": "umpire_end",
            "batter_handedness": "right",
            "calibration_score": 0.9,
            "calibration_quality": "High",
            "stumps": {
                "batter_end": {
                    "bbox": [100, 200, 140, 300],
                    "center": [120, 250],
                    "confidence": 0.2,
                    "source": "estimated",
                    "status": "estimated",
                }
            },
            "pitch_corridor": {
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "bbox": [0, 0, 1, 1],
                "source": "estimated",
                "status": "estimated",
                "confidence": 0.2,
            },
            "notes": [],
        }
    )

    assert context["calibration_quality"] in {"Low", "Medium"}
    assert context["calibration_quality"] != "High"
    assert any("no usable stump detection" in note.lower() for note in context["notes"])


def test_finalize_calibration_quality_allows_high_for_detected_stumps():
    from Backends.src.calibration.calibration_context import finalize_calibration_quality

    context = finalize_calibration_quality(
        {
            "enabled": True,
            "calibration_version": 1,
            "camera_view": "umpire_end",
            "batter_handedness": "right",
            "calibration_score": 0.9,
            "stumps": {
                "batter_end": {
                    "bbox": [100, 200, 140, 300],
                    "center": [120, 250],
                    "confidence": 0.8,
                    "source": "auto",
                    "status": "detected",
                }
            },
            "pitch_corridor": {
                "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "bbox": [0, 0, 1, 1],
                "source": "estimated",
                "status": "estimated",
                "confidence": 0.7,
            },
            "notes": [],
        }
    )

    assert context["calibration_quality"] == "Medium"
    assert context["calibration_quality"] != "High"


def test_normalize_preserves_legacy_quality_label_without_score():
    context = normalize_calibration_context(
        {"enabled": True, "calibration_quality": "Good"}
    )

    assert context["calibration_quality"] in {"Low", "Medium", "Good"}
