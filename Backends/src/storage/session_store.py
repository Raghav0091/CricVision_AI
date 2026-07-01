"""Local JSON persistence for analyzed delivery session results."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Backends.src.config.paths import (
    DATA_DIR,
    SESSION_CLIPS_DIR,
    SESSION_RESULTS_FILE,
)

SESSION_DATA_DIR = DATA_DIR

STRING_DEFAULT = "Unknown"
TEXT_DEFAULT = "N/A"


def load_session_results() -> list[dict]:
    """Load all saved session results. Return an empty list if the file does not exist."""
    if not SESSION_RESULTS_FILE.exists():
        return []

    try:
        with open(SESSION_RESULTS_FILE, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        corrupt_path = SESSION_RESULTS_FILE.with_suffix(
            f".corrupt.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            SESSION_RESULTS_FILE.rename(corrupt_path)
        except OSError:
            pass
        return []

    if isinstance(payload, dict):
        results = payload.get("results", [])
    elif isinstance(payload, list):
        results = payload
    else:
        return []

    return [normalize_session_result(item) for item in results if isinstance(item, dict)]


def save_session_result(result: dict) -> dict:
    """Append a new analysis result to local JSON storage."""
    SESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    normalized = normalize_session_result(result)

    if not normalized.get("id"):
        normalized["id"] = str(uuid.uuid4())
    if not normalized.get("created_at"):
        normalized["created_at"] = datetime.now(timezone.utc).isoformat()

    existing = []
    if SESSION_RESULTS_FILE.exists():
        try:
            with open(SESSION_RESULTS_FILE, encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                existing = payload.get("results", [])
            elif isinstance(payload, list):
                existing = payload
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            existing = []

    existing = [normalize_session_result(item) for item in existing if isinstance(item, dict)]
    existing.append(normalized)

    temp_path = SESSION_RESULTS_FILE.with_suffix(".tmp.json")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump({"results": existing}, handle, indent=2)
    temp_path.replace(SESSION_RESULTS_FILE)
    return normalized


def normalize_session_result(result: dict | None) -> dict:
    """Ensure old and new saved results have a safe consistent shape."""
    result = result or {}
    impact = result.get("impact_info") if isinstance(result.get("impact_info"), dict) else {}
    shot = result.get("shot_info") if isinstance(result.get("shot_info"), dict) else {}
    outcome = result.get("outcome_info") if isinstance(result.get("outcome_info"), dict) else {}
    direction = result.get("direction_info") if isinstance(result.get("direction_info"), dict) else {}

    bounce_point = result.get("bounce_point", result.get("estimated_bounce_point"))
    if bounce_point is not None and not isinstance(bounce_point, (list, tuple)):
        bounce_point = None

    coach_feedback = result.get("delivery_coach_feedback", [])
    if isinstance(coach_feedback, str):
        coach_feedback = [coach_feedback] if coach_feedback.strip() else []
    elif not isinstance(coach_feedback, list):
        coach_feedback = []

    review_flags = result.get("review_flags", [])
    if not isinstance(review_flags, list):
        review_flags = []

    observer = result.get("observer_timeline") if isinstance(result.get("observer_timeline"), dict) else {}
    repair = (
        result.get("visual_observer_repair")
        if isinstance(result.get("visual_observer_repair"), dict)
        else {}
    )

    return {
        "id": result.get("id") or "",
        "created_at": result.get("created_at") or "",
        "source_type": result.get("source_type") or result.get("analysis_mode") or STRING_DEFAULT,
        "video_name": result.get("video_name") or result.get("processed_file_name") or TEXT_DEFAULT,
        "processed_video_path": _safe_path(result.get("processed_video_path")),
        "impact_frame_image_path": _safe_path(
            result.get("impact_frame_image_path") or impact.get("impact_frame_image_path")
        ),
        "line": result.get("line", result.get("estimated_line", STRING_DEFAULT)) or STRING_DEFAULT,
        "length": result.get("length", result.get("estimated_length", STRING_DEFAULT)) or STRING_DEFAULT,
        "bounce_point": list(bounce_point) if bounce_point else None,
        "bounce_distance": result.get("bounce_distance", TEXT_DEFAULT),
        "ball_tracking_quality": result.get(
            "ball_tracking_quality",
            result.get("overall_tracking_quality", STRING_DEFAULT),
        )
        or STRING_DEFAULT,
        "detected_objects": result.get("detected_objects") or TEXT_DEFAULT,
        "delivery_result_summary": result.get("delivery_result_summary") or TEXT_DEFAULT,
        "delivery_coach_feedback": coach_feedback,
        "impact_detected": bool(
            result.get("impact_detected", impact.get("impact_detected", impact.get("impact_frame") is not None))
        ),
        "impact_frame": result.get("impact_frame", impact.get("impact_frame")),
        "impact_time_sec": result.get("impact_time_sec", impact.get("impact_time_sec")),
        "min_ball_bat_distance_px": result.get(
            "min_ball_bat_distance_px",
            impact.get("min_ball_bat_distance_px", impact.get("min_distance")),
        ),
        "impact_confidence": result.get("impact_confidence", impact.get("impact_confidence", STRING_DEFAULT))
        or STRING_DEFAULT,
        "impact_reason": result.get("impact_reason", impact.get("reason", impact.get("impact_reason", ""))) or "",
        "shot_type": result.get("shot_type", shot.get("shot_type", STRING_DEFAULT)) or STRING_DEFAULT,
        "shot_confidence": result.get("shot_confidence", shot.get("shot_confidence", STRING_DEFAULT))
        or STRING_DEFAULT,
        "shot_height": result.get("shot_height", shot.get("shot_height", STRING_DEFAULT)) or STRING_DEFAULT,
        "shot_reason": result.get("shot_reason", shot.get("reason", shot.get("shot_reason", ""))) or "",
        "shot_direction": result.get(
            "shot_direction",
            direction.get("shot_direction", result.get("direction_shot_category", STRING_DEFAULT)),
        )
        or STRING_DEFAULT,
        "field_zone": result.get("field_zone", direction.get("field_zone", STRING_DEFAULT)) or STRING_DEFAULT,
        "zone_confidence": result.get("zone_confidence", direction.get("zone_confidence", STRING_DEFAULT))
        or STRING_DEFAULT,
        "direction_angle_degrees": result.get(
            "direction_angle_degrees",
            direction.get("direction_angle_degrees"),
        ),
        "movement_dx": result.get("movement_dx", direction.get("movement_dx")),
        "movement_dy": result.get("movement_dy", direction.get("movement_dy")),
        "direction_reason": result.get(
            "direction_reason",
            direction.get("reason", direction.get("direction_reason", "")),
        )
        or "",
        "predicted_outcome": result.get(
            "predicted_outcome",
            outcome.get("predicted_outcome", STRING_DEFAULT),
        )
        or STRING_DEFAULT,
        "outcome_confidence": result.get(
            "outcome_confidence",
            outcome.get("outcome_confidence", STRING_DEFAULT),
        )
        or STRING_DEFAULT,
        "run_estimate": result.get("run_estimate", outcome.get("run_estimate")),
        "dismissal_risk": result.get("dismissal_risk", outcome.get("dismissal_risk", STRING_DEFAULT))
        or STRING_DEFAULT,
        "boundary_chance": result.get("boundary_chance", outcome.get("boundary_chance", STRING_DEFAULT))
        or STRING_DEFAULT,
        "outcome_reason": result.get(
            "outcome_reason",
            outcome.get("reason", outcome.get("outcome_reason", "")),
        )
        or "",
        "agent_quality": result.get("agent_quality", STRING_DEFAULT) or STRING_DEFAULT,
        "agent_confidence": result.get("agent_confidence", STRING_DEFAULT) or STRING_DEFAULT,
        "ball_tracking_coverage": result.get("ball_tracking_coverage"),
        "bat_detection_coverage": result.get("bat_detection_coverage"),
        "stump_detection_coverage": result.get("stump_detection_coverage"),
        "missing_ball_frames": int(result.get("missing_ball_frames") or 0),
        "possible_false_ball_detections": int(result.get("possible_false_ball_detections") or 0),
        "analysis_consistency": result.get("analysis_consistency", STRING_DEFAULT) or STRING_DEFAULT,
        "review_flags": review_flags,
        "agent_notes": result.get("agent_notes") or "",
        "total_frames": result.get("total_frames", observer.get("total_frames")),
        "processed_frames": result.get("processed_frames", observer.get("processed_frames")),
        "low_confidence_ball_frames": int(
            result.get("low_confidence_ball_frames", observer.get("low_confidence_ball_frames")) or 0
        ),
        "detection_quality": result.get("detection_quality", observer.get("detection_quality", STRING_DEFAULT))
        or STRING_DEFAULT,
        "observer_notes": result.get("observer_notes", observer.get("observer_notes", "")) or "",
        "visual_observer_repair": _normalize_visual_observer_repair(repair),
        "smart_analysis_mode": result.get(
            "smart_analysis_mode",
            result.get("speed_mode", (result.get("performance_profile") or {}).get("speed_mode", STRING_DEFAULT)),
        )
        or STRING_DEFAULT,
        "total_analysis_time_sec": result.get(
            "total_analysis_time_sec",
            (result.get("performance_profile") or {}).get("total_analysis_time_sec"),
        ),
        "frames_processed": result.get(
            "frames_processed",
            (result.get("performance_profile") or {}).get("frames_processed"),
        ),
        "avg_time_per_frame_sec": result.get(
            "avg_time_per_frame_sec",
            (result.get("performance_profile") or {}).get("avg_time_per_frame_sec"),
        ),
        "processed_video_generated": result.get(
            "processed_video_generated",
            (result.get("performance_profile") or {}).get("processed_video_generated"),
        ),
        "smart_pipeline_used": bool(
            result.get(
                "smart_pipeline_used",
                (result.get("performance_profile") or {}).get("smart_pipeline_used", False),
            )
        ),
    }


def get_session_summary(results: list[dict] | None) -> dict:
    """Return dashboard summary metrics from saved results."""
    results = [normalize_session_result(item) for item in (results or []) if isinstance(item, dict)]
    if not results:
        return {
            "total_deliveries": 0,
            "total_predicted_runs": 0,
            "most_common_shot_type": TEXT_DEFAULT,
            "most_common_field_zone": TEXT_DEFAULT,
            "most_common_outcome": TEXT_DEFAULT,
            "most_common_length": TEXT_DEFAULT,
            "most_common_line": TEXT_DEFAULT,
            "average_ball_tracking_coverage": None,
            "average_agent_quality": TEXT_DEFAULT,
            "agent_quality_distribution": {},
            "shot_type_distribution": {},
            "outcome_distribution": {},
            "field_zone_distribution": {},
            "length_distribution": {},
            "line_distribution": {},
            "insights": [],
        }

    shot_types = _counter_values(results, "shot_type")
    field_zones = _counter_values(results, "field_zone")
    outcomes = _counter_values(results, "predicted_outcome")
    lengths = _counter_values(results, "length")
    lines = _counter_values(results, "line")
    agent_qualities = _counter_values(results, "agent_quality")

    coverages = [
        float(item["ball_tracking_coverage"])
        for item in results
        if item.get("ball_tracking_coverage") is not None
    ]
    avg_coverage = round(sum(coverages) / len(coverages), 1) if coverages else None

    total_runs = 0
    for item in results:
        total_runs += _parse_run_estimate(item.get("run_estimate"))

    most_shot = shot_types.most_common(1)[0][0] if shot_types else TEXT_DEFAULT
    most_zone = field_zones.most_common(1)[0][0] if field_zones else TEXT_DEFAULT
    most_outcome = outcomes.most_common(1)[0][0] if outcomes else TEXT_DEFAULT
    most_length = lengths.most_common(1)[0][0] if lengths else TEXT_DEFAULT
    most_line = lines.most_common(1)[0][0] if lines else TEXT_DEFAULT
    avg_agent = _average_agent_quality(agent_qualities)

    insights = []
    if most_shot not in {TEXT_DEFAULT, STRING_DEFAULT, ""}:
        insights.append(f"Most common shot: {most_shot}")
    if most_zone not in {TEXT_DEFAULT, STRING_DEFAULT, ""}:
        insights.append(f"Most common scoring zone: {most_zone}")
    if most_outcome not in {TEXT_DEFAULT, STRING_DEFAULT, ""}:
        insights.append(f"Most common predicted outcome: {most_outcome}")
    if avg_coverage is not None:
        insights.append(f"Average ball tracking coverage: {avg_coverage:.1f}%")
    if avg_agent not in {TEXT_DEFAULT, STRING_DEFAULT, ""}:
        insights.append(f"Agent quality is mostly {avg_agent}")

    return {
        "total_deliveries": len(results),
        "total_predicted_runs": total_runs,
        "most_common_shot_type": most_shot,
        "most_common_field_zone": most_zone,
        "most_common_outcome": most_outcome,
        "most_common_length": most_length,
        "most_common_line": most_line,
        "average_ball_tracking_coverage": avg_coverage,
        "average_agent_quality": avg_agent,
        "agent_quality_distribution": dict(agent_qualities),
        "shot_type_distribution": dict(shot_types),
        "outcome_distribution": dict(outcomes),
        "field_zone_distribution": dict(field_zones),
        "length_distribution": dict(lengths),
        "line_distribution": dict(lines),
        "insights": insights,
    }


def persist_analysis_to_session(
    result: dict,
    source_type: str,
    video_name: str | None = None,
) -> dict:
    """Build a lightweight session record from an analysis result and save it."""
    record = build_session_record_from_analysis(result, source_type=source_type, video_name=video_name)
    return save_session_result(record)


def build_session_record_from_analysis(
    result: dict,
    source_type: str,
    video_name: str | None = None,
) -> dict:
    """Convert a full in-memory analysis result into a lightweight persisted record."""
    result = result or {}
    impact = result.get("impact_info") or {}
    record_id = str(uuid.uuid4())

    processed_video_path = _resolve_processed_video_path(result, record_id)
    impact_image_path = _safe_path(impact.get("impact_frame_image_path"))

    bounce_point = result.get("estimated_bounce_point")
    bounce_distance = _format_bounce_distance(result)

    report_view = session_record_to_report_view(
        normalize_session_result(
            {
                **result,
                "source_type": source_type,
                "video_name": video_name or _default_video_name(result, source_type),
                "processed_video_path": processed_video_path,
                "impact_frame_image_path": impact_image_path,
                "line": result.get("estimated_line"),
                "length": result.get("estimated_length"),
                "bounce_point": bounce_point,
                "bounce_distance": bounce_distance,
            }
        )
    )
    summary, feedback = _build_delivery_text_fields(report_view)
    observer = result.get("observer_timeline") or {}
    profile = result.get("performance_profile") or {}

    return normalize_session_result(
        {
            "id": record_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_type": source_type,
            "video_name": video_name or _default_video_name(result, source_type),
            "processed_video_path": processed_video_path,
            "impact_frame_image_path": impact_image_path,
            "line": result.get("estimated_line"),
            "length": result.get("estimated_length"),
            "bounce_point": list(bounce_point) if bounce_point else None,
            "bounce_distance": bounce_distance,
            "ball_tracking_quality": result.get("overall_tracking_quality"),
            "detected_objects": _format_detected_objects_summary(result),
            "delivery_result_summary": summary,
            "delivery_coach_feedback": feedback,
            "smart_analysis_mode": result.get("speed_mode") or profile.get("speed_mode"),
            "total_analysis_time_sec": profile.get("total_analysis_time_sec"),
            "frames_processed": profile.get("frames_processed", result.get("total_frames")),
            "avg_time_per_frame_sec": profile.get("avg_time_per_frame_sec"),
            "processed_video_generated": result.get(
                "processed_video_generated",
                profile.get("processed_video_generated"),
            ),
            "smart_pipeline_used": result.get(
                "smart_pipeline_used",
                profile.get("smart_pipeline_used", True),
            ),
            "impact_detected": impact.get("impact_detected", impact.get("impact_frame") is not None),
            "impact_frame": impact.get("impact_frame"),
            "impact_time_sec": impact.get("impact_time_sec"),
            "min_ball_bat_distance_px": impact.get("min_ball_bat_distance_px", impact.get("min_distance")),
            "impact_confidence": impact.get("impact_confidence"),
            "impact_reason": impact.get("reason", impact.get("impact_reason", "")),
            "shot_type": result.get("shot_type") or (result.get("shot_info") or {}).get("shot_type"),
            "shot_confidence": result.get("shot_confidence")
            or (result.get("shot_info") or {}).get("shot_confidence"),
            "shot_height": result.get("shot_height") or (result.get("shot_info") or {}).get("shot_height"),
            "shot_reason": result.get("shot_reason")
            or (result.get("shot_info") or {}).get("reason")
            or (result.get("shot_info") or {}).get("shot_reason"),
            "shot_direction": result.get("direction_shot_category")
            or (result.get("direction_info") or {}).get("shot_direction"),
            "field_zone": result.get("field_zone"),
            "zone_confidence": result.get("zone_confidence"),
            "direction_angle_degrees": result.get("direction_angle_degrees"),
            "movement_dx": result.get("movement_dx"),
            "movement_dy": result.get("movement_dy"),
            "direction_reason": result.get("direction_reason")
            or (result.get("direction_info") or {}).get("reason"),
            "predicted_outcome": result.get("predicted_outcome")
            or (result.get("outcome_info") or {}).get("predicted_outcome"),
            "outcome_confidence": result.get("outcome_confidence")
            or (result.get("outcome_info") or {}).get("outcome_confidence"),
            "run_estimate": result.get("run_estimate") or (result.get("outcome_info") or {}).get("run_estimate"),
            "dismissal_risk": result.get("dismissal_risk")
            or (result.get("outcome_info") or {}).get("dismissal_risk"),
            "boundary_chance": result.get("boundary_chance")
            or (result.get("outcome_info") or {}).get("boundary_chance"),
            "outcome_reason": result.get("outcome_reason")
            or (result.get("outcome_info") or {}).get("reason")
            or (result.get("outcome_info") or {}).get("outcome_reason"),
            "agent_quality": result.get("agent_quality"),
            "agent_confidence": result.get("agent_confidence"),
            "analysis_consistency": result.get("analysis_consistency"),
            "review_flags": result.get("review_flags") or [],
            "agent_notes": result.get("agent_notes"),
            "total_frames": result.get("total_frames", observer.get("total_frames")),
            "processed_frames": observer.get("processed_frames", result.get("processed_frames")),
            "low_confidence_ball_frames": result.get(
                "low_confidence_ball_frames",
                observer.get("low_confidence_ball_frames"),
            ),
            "detection_quality": observer.get("detection_quality"),
            "observer_notes": observer.get("observer_notes"),
            "visual_observer_repair": result.get("visual_observer_repair") or {},
            "ball_tracking_coverage": result.get("ball_tracking_coverage")
            or observer.get("ball_tracking_coverage"),
            "bat_detection_coverage": result.get("bat_detection_coverage")
            or observer.get("bat_detection_coverage"),
            "stump_detection_coverage": result.get("stump_detection_coverage")
            or observer.get("stump_detection_coverage"),
            "missing_ball_frames": result.get("missing_ball_frames", observer.get("missing_ball_frames")),
            "possible_false_ball_detections": result.get(
                "possible_false_ball_detections",
                observer.get("possible_false_ball_detections"),
            ),
        }
    )


def session_record_to_report_view(record: dict) -> dict:
    """Convert a normalized session record into the dict shape used by render_* components."""
    record = normalize_session_result(record)
    bounce_point = record.get("bounce_point")
    if isinstance(bounce_point, list) and len(bounce_point) >= 2:
        bounce_tuple = (bounce_point[0], bounce_point[1])
    else:
        bounce_tuple = None

    ball_frames = None
    bat_frames = None
    stump_frames = None
    detected_objects = record.get("detected_objects", "")
    if isinstance(detected_objects, str):
        if "Ball" in detected_objects:
            ball_frames = 1
        if "Bat" in detected_objects:
            bat_frames = 1
        if "Stumps" in detected_objects:
            stump_frames = 1

    return {
        "success": True,
        "source_type": record.get("source_type"),
        "video_name": record.get("video_name"),
        "processed_video_path": record.get("processed_video_path"),
        "output_path": record.get("processed_video_path"),
        "estimated_line": record.get("line"),
        "estimated_length": record.get("length"),
        "estimated_bounce_point": bounce_tuple,
        "bounce_distance": record.get("bounce_distance"),
        "overall_tracking_quality": record.get("ball_tracking_quality"),
        "detected_objects": record.get("detected_objects"),
        "delivery_result_summary": record.get("delivery_result_summary"),
        "delivery_coach_feedback": record.get("delivery_coach_feedback"),
        "ball_detected_frames": ball_frames,
        "bat_detected_frames": bat_frames,
        "stump_detected_frames": stump_frames,
        "delivery_result_summary": record.get("delivery_result_summary"),
        "delivery_coach_feedback": record.get("delivery_coach_feedback"),
        "impact_info": {
            "impact_detected": record.get("impact_detected"),
            "impact_frame": record.get("impact_frame"),
            "impact_time_sec": record.get("impact_time_sec"),
            "min_ball_bat_distance_px": record.get("min_ball_bat_distance_px"),
            "impact_confidence": record.get("impact_confidence"),
            "reason": record.get("impact_reason"),
            "impact_reason": record.get("impact_reason"),
            "impact_frame_image_path": record.get("impact_frame_image_path"),
        },
        "shot_info": {
            "shot_type": record.get("shot_type"),
            "shot_confidence": record.get("shot_confidence"),
            "shot_direction": record.get("shot_direction"),
            "shot_height": record.get("shot_height"),
            "reason": record.get("shot_reason"),
            "shot_reason": record.get("shot_reason"),
        },
        "direction_info": {
            "shot_direction": record.get("shot_direction"),
            "field_zone": record.get("field_zone"),
            "zone_confidence": record.get("zone_confidence"),
            "direction_angle_degrees": record.get("direction_angle_degrees"),
            "movement_dx": record.get("movement_dx"),
            "movement_dy": record.get("movement_dy"),
            "reason": record.get("direction_reason"),
            "direction_reason": record.get("direction_reason"),
        },
        "field_zone": record.get("field_zone"),
        "zone_confidence": record.get("zone_confidence"),
        "direction_angle_degrees": record.get("direction_angle_degrees"),
        "direction_reason": record.get("direction_reason"),
        "movement_dx": record.get("movement_dx"),
        "movement_dy": record.get("movement_dy"),
        "direction_shot_category": record.get("shot_direction"),
        "outcome_info": {
            "predicted_outcome": record.get("predicted_outcome"),
            "outcome_confidence": record.get("outcome_confidence"),
            "run_estimate": record.get("run_estimate"),
            "dismissal_risk": record.get("dismissal_risk"),
            "boundary_chance": record.get("boundary_chance"),
            "reason": record.get("outcome_reason"),
            "outcome_reason": record.get("outcome_reason"),
        },
        "predicted_outcome": record.get("predicted_outcome"),
        "outcome_confidence": record.get("outcome_confidence"),
        "run_estimate": record.get("run_estimate"),
        "agent_info": {
            "agent_quality": record.get("agent_quality"),
            "agent_confidence": record.get("agent_confidence"),
            "ball_tracking_coverage": record.get("ball_tracking_coverage"),
            "bat_detection_coverage": record.get("bat_detection_coverage"),
            "stump_detection_coverage": record.get("stump_detection_coverage"),
            "missing_ball_frames": record.get("missing_ball_frames"),
            "possible_false_ball_detections": record.get("possible_false_ball_detections"),
            "analysis_consistency": record.get("analysis_consistency"),
            "review_flags": record.get("review_flags"),
            "agent_notes": record.get("agent_notes"),
        },
        "agent_quality": record.get("agent_quality"),
        "agent_confidence": record.get("agent_confidence"),
        "ball_tracking_coverage": record.get("ball_tracking_coverage"),
        "session_saved": True,
        "session_result_id": record.get("id"),
        "observer_timeline": {
            "total_frames": record.get("total_frames"),
            "processed_frames": record.get("processed_frames"),
            "ball_tracking_coverage": record.get("ball_tracking_coverage"),
            "bat_detection_coverage": record.get("bat_detection_coverage"),
            "stump_detection_coverage": record.get("stump_detection_coverage"),
            "missing_ball_frames": record.get("missing_ball_frames"),
            "low_confidence_ball_frames": record.get("low_confidence_ball_frames"),
            "possible_false_ball_detections": record.get("possible_false_ball_detections"),
            "detection_quality": record.get("detection_quality"),
            "observer_notes": record.get("observer_notes"),
        },
        "visual_observer_repair": record.get("visual_observer_repair") or {},
    }


def _safe_path(value: Any) -> str | None:
    if value in {None, "", TEXT_DEFAULT}:
        return None
    return str(value)


def _normalize_visual_observer_repair(repair: dict) -> dict:
    if not repair:
        return {}
    notes = repair.get("notes") or []
    if isinstance(notes, str):
        notes = [notes]
    elif not isinstance(notes, list):
        notes = []
    return {
        "original_coverage": repair.get("original_coverage"),
        "repaired_coverage": repair.get("repaired_coverage"),
        "missing_frames": int(repair.get("missing_frames") or 0),
        "repaired_frames": int(repair.get("repaired_frames") or 0),
        "suspicious_detections": int(
            repair.get(
                "suspicious_detections",
                repair.get("removed_or_downgraded_frames"),
            )
            or 0
        ),
        "repair_confidence": repair.get("repair_confidence") or STRING_DEFAULT,
        "agent_decision": repair.get("agent_decision") or "",
        "notes": [str(note) for note in notes if str(note).strip()],
    }


def _resolve_processed_video_path(result: dict, record_id: str) -> str | None:
    output_path = result.get("output_path") or result.get("processed_video_path")
    if output_path:
        return str(output_path)

    video_bytes = result.get("processed_video_bytes")
    if not video_bytes:
        return None

    SESSION_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    clip_path = SESSION_CLIPS_DIR / f"{record_id}.mp4"
    try:
        clip_path.write_bytes(video_bytes)
        return str(clip_path)
    except OSError:
        return None


def _default_video_name(result: dict, source_type: str) -> str:
    if result.get("processed_file_name"):
        return str(result["processed_file_name"])
    output_path = result.get("output_path")
    if output_path:
        return Path(str(output_path)).name
    return f"{source_type} delivery"


def _format_bounce_distance(result: dict) -> str:
    normalized = result.get("pitch_normalized_bounce_point")
    if normalized is not None:
        return str(normalized)
    bounce_point = result.get("estimated_bounce_point")
    if bounce_point is None:
        return TEXT_DEFAULT
    try:
        bx, by = bounce_point
        return f"({int(bx)}, {int(by)})"
    except (TypeError, ValueError):
        return TEXT_DEFAULT


def _format_detected_objects_summary(result: dict) -> str:
    items = []
    if result.get("ball_detected_frames"):
        items.append(f"Ball ({int(result['ball_detected_frames'])} frames)")
    if result.get("bat_detected_frames"):
        items.append(f"Bat ({int(result['bat_detected_frames'])} frames)")
    if result.get("stump_detected_frames"):
        items.append(f"Stumps ({int(result['stump_detected_frames'])} frames)")
    return ", ".join(items) if items else TEXT_DEFAULT


def _build_delivery_text_fields(report_view: dict) -> tuple[str, list[str]]:
    try:
        from Backends.src.analysis.cricket_agent import (
            generate_coaching_feedback,
            generate_delivery_report,
        )

        summary = generate_delivery_report(report_view)
        feedback = generate_coaching_feedback(report_view)
        if not isinstance(feedback, list):
            feedback = [str(feedback)]
        return str(summary), [str(item) for item in feedback if str(item).strip()]
    except Exception:
        line = report_view.get("estimated_line", STRING_DEFAULT)
        length = report_view.get("estimated_length", STRING_DEFAULT)
        tracking = report_view.get("overall_tracking_quality", STRING_DEFAULT)
        return (
            f"Line {line}, length {length}, tracking quality {tracking}.",
            [],
        )


def _counter_values(results: list[dict], field: str) -> Counter:
    counter: Counter = Counter()
    for item in results:
        value = item.get(field)
        if value in {None, "", TEXT_DEFAULT, STRING_DEFAULT}:
            continue
        counter[str(value)] += 1
    return counter


def _parse_run_estimate(value: Any) -> int:
    if value in {None, "", TEXT_DEFAULT, STRING_DEFAULT}:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    mapping = {
        "dot ball": 0,
        "dot": 0,
        "single": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "six": 6,
    }
    return mapping.get(text, 0)


def _average_agent_quality(counter: Counter) -> str:
    if not counter:
        return TEXT_DEFAULT
    weights = {"High": 3, "Medium": 2, "Low": 1, "Unknown": 0}
    total_weight = 0
    total_count = 0
    for label, count in counter.items():
        total_weight += weights.get(label, 0) * count
        total_count += count
    if total_count == 0:
        return TEXT_DEFAULT
    average = total_weight / total_count
    if average >= 2.5:
        return "High"
    if average >= 1.5:
        return "Medium"
    return "Low"
