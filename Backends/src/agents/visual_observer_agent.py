"""High-level Visual Observer wrapper for deterministic ball-track repair."""

from __future__ import annotations

from copy import deepcopy

from Backends.src.agents.tracking_repair_agent import repair_ball_tracking


def run_visual_observer_repair(
    frame_detections,
    frame_width=None,
    frame_height=None,
    fps=None,
    context=None,
):
    """Repair a 2D ball timeline and return a concise observer decision."""
    del fps, context  # Reserved for later video-specific tuning.
    try:
        result = repair_ball_tracking(
            frame_detections,
            frame_width=frame_width,
            frame_height=frame_height,
        )
    except Exception as error:
        raw = deepcopy(frame_detections)
        return {
            "frame_detections": raw,
            "raw_frame_detections": raw,
            "repair_report": {
                "original_coverage": 0.0,
                "repaired_coverage": 0.0,
                "missing_frames": 0,
                "repaired_frames": 0,
                "removed_or_downgraded_frames": 0,
                "suspicious_detections": 0,
                "false_detection_candidates": 0,
                "repair_confidence": "Low",
                "agent_decision": "Tracking repair failed. Reports are using raw detections.",
                "notes": [f"Visual Observer fallback: {error}"],
            },
        }

    report = dict(result["repair_report"])
    report["repair_confidence"] = _overall_repair_confidence(report)
    report["agent_decision"] = _agent_decision(report)
    report["notes"] = _observer_notes(report)
    return {
        "frame_detections": result["repaired_frame_detections"],
        "raw_frame_detections": result["raw_frame_detections"],
        "repair_report": report,
    }


def _overall_repair_confidence(report):
    total_frames = int(report.get("total_frames") or 0)
    repaired_coverage = float(report.get("repaired_coverage") or 0)
    suspicious = int(report.get("suspicious_detections") or 0)
    repaired = int(report.get("repaired_frames") or 0)
    missing = int(report.get("missing_frames") or 0)

    if total_frames == 0 or repaired_coverage < 30:
        return "Low"
    if suspicious > 0 or repaired > 0 or missing > 0 or repaired_coverage < 65:
        return "Medium"
    return "High"


def _agent_decision(report):
    if int(report.get("total_frames") or 0) == 0:
        return "Tracking quality is too weak. Reports should be treated as low confidence."
    if float(report.get("repaired_coverage") or 0) < 30:
        return "Tracking quality is too weak. Reports should be treated as low confidence."
    if int(report.get("suspicious_detections") or 0):
        return "Suspicious jumps were downgraded and reports should use medium confidence."
    if int(report.get("repaired_frames") or 0):
        return "Short gaps were repaired using trajectory interpolation."
    if int(report.get("missing_frames") or 0):
        return "Long or unbounded gaps were left unrepaired; reports should use medium confidence."
    return "No repair needed. Ball tracking was stable."


def _observer_notes(report):
    notes = []
    repaired = int(report.get("repaired_frames") or 0)
    downgraded = int(report.get("removed_or_downgraded_frames") or 0)
    missing = int(report.get("missing_frames") or 0)

    if repaired:
        notes.append(
            f"Repaired {repaired} frame(s) with linear interpolation between trusted detections."
        )
    if downgraded:
        notes.append(
            f"Downgraded {downgraded} suspicious detection frame(s) before report generation."
        )
    if missing > repaired:
        notes.append("Long or unbounded gaps were left unchanged rather than guessed.")
    if not notes:
        notes.append("No short gaps or suspicious ball jumps required repair.")
    notes.append("This is 2D tracking repair, not 3D ball tracking.")
    return notes
