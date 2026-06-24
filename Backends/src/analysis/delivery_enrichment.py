"""Post-shot direction estimation and vision agent review helpers."""

from Backends.src.agents.vision_agent import run_vision_agent
from Backends.src.analysis.outcome_prediction import predict_shot_outcome
from Backends.src.analysis.shot_direction import estimate_shot_direction_zone


def run_post_shot_pipeline(
    impact_frame_detections,
    impact_info,
    shot_info,
    batter_handedness,
    fps,
    delivery_report=None,
):
    """Run direction estimation, outcome prediction, and vision agent review."""
    direction_info = estimate_shot_direction_zone(
        impact_frame_detections,
        impact_info,
        shot_result=shot_info,
        batter_handedness=batter_handedness,
        fps=fps,
    )
    outcome_info = predict_shot_outcome(
        impact_frame_detections,
        impact_info,
        shot_info,
        fps=fps,
        batter_handedness=batter_handedness,
        direction_result=direction_info,
    )
    agent_info = run_vision_agent(
        impact_frame_detections,
        delivery_report=delivery_report,
        impact_result=impact_info,
        shot_result=shot_info,
        direction_result=direction_info,
        outcome_result=outcome_info,
        fps=fps,
    )
    enrichment = merge_direction_and_agent_fields(direction_info, agent_info)
    return direction_info, outcome_info, agent_info, enrichment


def run_direction_and_agent_review(
    frame_detections,
    impact_info,
    shot_info,
    outcome_info,
    batter_handedness=None,
    fps=None,
    delivery_report=None,
):
    """Estimate field zone and run the vision agent after shot/outcome analysis."""
    direction_info = estimate_shot_direction_zone(
        frame_detections,
        impact_info,
        shot_result=shot_info,
        batter_handedness=batter_handedness,
        fps=fps,
    )
    agent_info = run_vision_agent(
        frame_detections,
        delivery_report=delivery_report,
        impact_result=impact_info,
        shot_result=shot_info,
        direction_result=direction_info,
        outcome_result=outcome_info,
        fps=fps,
    )
    return direction_info, agent_info


def merge_direction_and_agent_fields(direction_info, agent_info):
    """Flatten direction and agent dicts for session persistence and UI."""
    direction_info = direction_info or {}
    agent_info = agent_info or {}

    return {
        "direction_info": direction_info,
        "agent_info": agent_info,
        "field_zone": direction_info.get("field_zone", "Unknown"),
        "zone_confidence": direction_info.get("zone_confidence", "Unknown"),
        "direction_angle_degrees": direction_info.get("direction_angle_degrees"),
        "direction_reason": direction_info.get("reason", ""),
        "movement_dx": direction_info.get("movement_dx"),
        "movement_dy": direction_info.get("movement_dy"),
        "direction_shot_category": direction_info.get("shot_direction", "Unknown"),
        "agent_quality": agent_info.get("agent_quality", "Unknown"),
        "agent_confidence": agent_info.get("agent_confidence", "Unknown"),
        "ball_tracking_coverage": agent_info.get("ball_tracking_coverage"),
        "bat_detection_coverage": agent_info.get("bat_detection_coverage"),
        "stump_detection_coverage": agent_info.get("stump_detection_coverage"),
        "missing_ball_frames": agent_info.get("missing_ball_frames", 0),
        "possible_false_ball_detections": agent_info.get("possible_false_ball_detections", 0),
        "analysis_consistency": agent_info.get("analysis_consistency", "Unknown"),
        "review_flags": list(agent_info.get("review_flags") or []),
        "agent_notes": agent_info.get("agent_notes", ""),
        "review_frames_recommended": agent_info.get("review_frames_recommended", False),
        "review_reason": agent_info.get("review_reason", ""),
        "direction_agent_available": True,
    }
