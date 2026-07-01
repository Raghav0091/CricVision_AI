"""Analysis summary helper tests are Streamlit-free."""

from Backends.src.ui.components import build_analysis_summary_data


def test_build_analysis_summary_handles_missing_impact_and_shot():
    summary = build_analysis_summary_data(
        {
            "estimated_line": "Leg Side",
            "estimated_length": "Yorker",
            "overall_tracking_quality": "Good",
            "calibration_context": {
                "enabled": True,
                "calibration_quality": "Medium",
                "stumps": {
                    "batter_end": {
                        "source": "estimated",
                        "status": "estimated",
                        "confidence": 0.2,
                    }
                },
            },
            "impact_info": {
                "impact_detected": False,
                "impact_frame": None,
            },
            "shot_info": {"shot_type": "Unknown"},
            "direction_info": {"field_zone": "Unknown"},
            "outcome_info": {"predicted_outcome": "Unknown"},
            "observer_timeline": {"bat_detection_coverage": 0.0},
            "visual_observer_repair": {"repair_confidence": "Medium"},
        }
    )

    assert summary["line"] == "Leg Side"
    assert summary["length"] == "Yorker"
    assert summary["impact_status"] == "Not Detected"
    assert summary["shot_type"] == "Unavailable"
    assert summary["predicted_outcome"] == "Unavailable"
    assert "bat was not detected" in summary["coach_note"].lower()
    assert summary["calibration_quality"].startswith("Estimated /")
