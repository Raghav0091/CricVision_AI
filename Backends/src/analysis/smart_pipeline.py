"""Smart accurate video analysis helpers for CricVision."""

from __future__ import annotations

import copy
from typing import Any

SMART_MODES = ("Smart Balanced", "Smart Accurate", "Debug Full Frame")

LEGACY_MODE_MAP = {
    "fast": "Smart Balanced",
    "balanced": "Smart Balanced",
    "accurate": "Smart Accurate",
    "smart balanced": "Smart Balanced",
    "smart accurate": "Smart Accurate",
    "debug full frame": "Debug Full Frame",
}


def resolve_smart_mode(mode: str | None) -> str:
    """Map UI or legacy mode names to a smart pipeline mode."""
    normalized = (mode or "Smart Balanced").strip()
    key = normalized.lower()
    if key in LEGACY_MODE_MAP:
        return LEGACY_MODE_MAP[key]
    if normalized in SMART_MODES:
        return normalized
    return "Smart Balanced"


def get_smart_analysis_settings(mode: str) -> dict[str, Any]:
    """Return analysis settings for Smart Balanced, Smart Accurate, or Debug Full Frame."""
    resolved = resolve_smart_mode(mode)
    if resolved == "Smart Accurate":
        return {
            "mode": resolved,
            "ball_frame_stride": 1,
            "bat_frame_stride": 1,
            "stump_detect_initial_frames": 20,
            "stump_verify_every_n_frames": 30,
            "player_frame_stride": 3,
            "resize_width": 960,
            "yolo_imgsz": 960,
            "light_annotation": False,
            "use_roi": True,
            "generate_processed_video": True,
            "enable_local_redetection": True,
            "skip_impact_video_rewrite": False,
            "single_pass_same_model": False,
            "refine_bat_near_impact": True,
            "impact_window_radius": 8,
            "smart_pipeline_used": True,
        }
    if resolved == "Debug Full Frame":
        return {
            "mode": resolved,
            "ball_frame_stride": 1,
            "bat_frame_stride": 1,
            "stump_detect_initial_frames": None,
            "stump_verify_every_n_frames": 1,
            "player_frame_stride": 1,
            "resize_width": None,
            "yolo_imgsz": 960,
            "light_annotation": False,
            "use_roi": False,
            "generate_processed_video": True,
            "enable_local_redetection": True,
            "skip_impact_video_rewrite": False,
            "single_pass_same_model": False,
            "refine_bat_near_impact": False,
            "impact_window_radius": 8,
            "smart_pipeline_used": True,
        }
    return {
        "mode": "Smart Balanced",
        "ball_frame_stride": 1,
        "bat_frame_stride": 2,
        "stump_detect_initial_frames": 20,
        "stump_verify_every_n_frames": 0,
        "player_frame_stride": 5,
        "resize_width": 854,
        "yolo_imgsz": 640,
        "light_annotation": True,
        "use_roi": True,
        "generate_processed_video": True,
        "enable_local_redetection": True,
        "skip_impact_video_rewrite": True,
        "single_pass_same_model": True,
        "refine_bat_near_impact": True,
        "impact_window_radius": 8,
        "smart_pipeline_used": True,
    }


def should_detect_ball(frame_index: int, settings: dict) -> bool:
    stride = max(1, int(settings.get("ball_frame_stride", 1)))
    return frame_index % stride == 0


def should_detect_bat(
    frame_index: int,
    settings: dict,
    rough_impact_frame: int | None = None,
) -> bool:
    stride = max(1, int(settings.get("bat_frame_stride", 1)))
    if stride == 1:
        return True
    radius = int(settings.get("impact_window_radius", 8))
    if rough_impact_frame is not None and abs(frame_index - rough_impact_frame) <= radius:
        return True
    return frame_index % stride == 0


def should_detect_stump(
    frame_index: int,
    settings: dict,
    locked_stump: dict | None,
) -> bool:
    initial = settings.get("stump_detect_initial_frames")
    verify_every = int(settings.get("stump_verify_every_n_frames") or 0)
    if initial is None:
        return True
    if frame_index < int(initial):
        return True
    if locked_stump is None and frame_index < int(initial) * 2:
        return True
    if verify_every > 0 and frame_index % verify_every == 0:
        return True
    return False


def lock_static_stump_detection(frame_detections, max_initial_frames: int = 20):
    """Find stable stump detection from early frames."""
    if not frame_detections:
        return None

    candidates = []
    for item in frame_detections:
        frame_index = int(item.get("frame_index", 0))
        if frame_index >= max_initial_frames:
            break
        for detection in item.get("stump_detections") or []:
            if isinstance(detection, dict):
                candidates.append(detection)

    if not candidates:
        return None

    def score(detection):
        try:
            confidence = float(detection.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0
        box = detection.get("box") or detection.get("bbox")
        area = 0
        if box and len(box) >= 4:
            area = abs(float(box[2]) - float(box[0])) * abs(float(box[3]) - float(box[1]))
        return confidence * 1000 + area

    best = max(candidates, key=score)
    locked = copy.deepcopy(best)
    locked["stump_locked"] = True
    return locked


def apply_locked_stump(stump_detections, locked_stump):
    """Reuse locked stump when the current frame has no stump detection."""
    if stump_detections:
        return stump_detections
    if locked_stump is None:
        return []
    return [copy.deepcopy(locked_stump)]


def update_rough_impact_frame(frame_index, ball_detections, bat_detections, current_best):
    """Track the frame with the smallest ball-bat separation."""
    if not ball_detections or not bat_detections:
        return current_best

    from Backends.src.analysis.bat_detection import calculate_distance

    best_distance = None
    ball_center = ball_detections[0].get("center")
    if ball_center is None:
        return current_best
    for bat in bat_detections:
        bat_center = bat.get("center")
        if bat_center is None:
            continue
        distance = calculate_distance(ball_center, bat_center)
        if best_distance is None or distance < best_distance:
            best_distance = distance

    if best_distance is None:
        return current_best
    if current_best is None or best_distance < current_best[0]:
        return best_distance, frame_index
    return current_best


def refine_bat_detections_near_impact(
    video_path,
    frame_detections,
    impact_frame,
    bat_model,
    confidence,
    resize_width,
    radius=8,
    stats=None,
):
    """Re-run bat detection every frame in the impact window."""
    if bat_model is None or impact_frame is None or not frame_detections:
        return frame_detections

    from Backends.src.analysis.analysis_speed import resize_frame_for_inference, scale_detections_to_original
    from Backends.src.analysis.bat_detection import detect_bat_in_frame
    from Backends.src.utils.cv2_loader import cv2

    try:
        impact_frame = int(impact_frame)
    except (TypeError, ValueError):
        return frame_detections

    frame_map = {int(item["frame_index"]): item for item in frame_detections if "frame_index" in item}
    start = max(0, impact_frame - radius)
    end = impact_frame + radius

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frame_detections

    for frame_index in range(start, end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            continue
        inference_frame, detection_scale = resize_frame_for_inference(frame, resize_width)
        bat_detections = detect_bat_in_frame(inference_frame, bat_model, confidence)
        bat_detections = scale_detections_to_original(bat_detections, detection_scale, stats=stats)
        if frame_index in frame_map:
            frame_map[frame_index]["bat_detections"] = bat_detections

    cap.release()
    return sorted(frame_map.values(), key=lambda item: item["frame_index"])
