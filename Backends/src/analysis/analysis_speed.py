"""Analysis speed/quality presets for uploaded video processing."""

from __future__ import annotations

from typing import Any


def get_analysis_mode_settings(mode: str) -> dict[str, Any]:
    """Return frame stride, resize, and inference settings for a speed mode."""
    normalized = (mode or "Balanced").strip().title()
    if normalized == "Fast":
        return {
            "frame_stride": 3,
            "resize_width": 640,
            "process_every_frame": False,
            "max_frames": None,
            "yolo_imgsz": 512,
            "enable_local_redetection": False,
            "skip_impact_video_rewrite": True,
            "single_pass_same_model": True,
        }
    if normalized == "Accurate":
        return {
            "frame_stride": 1,
            "resize_width": None,
            "process_every_frame": True,
            "max_frames": None,
            "yolo_imgsz": 960,
            "enable_local_redetection": True,
            "skip_impact_video_rewrite": False,
            "single_pass_same_model": False,
        }
    return {
        "frame_stride": 2,
        "resize_width": 768,
        "process_every_frame": False,
        "max_frames": None,
        "yolo_imgsz": 640,
        "enable_local_redetection": True,
        "skip_impact_video_rewrite": False,
        "single_pass_same_model": True,
    }


def resolve_frame_limit(enabled: bool, choice) -> int | None:
    """Convert UI frame-limit selection to an integer cap or None."""
    if not enabled:
        return None
    if choice in {None, "", "All frames"}:
        return None
    try:
        return max(1, int(choice))
    except (TypeError, ValueError):
        return None


def resize_frame_for_inference(frame, target_width: int | None):
    """Resize a frame for inference and return the frame plus inverse scale."""
    if target_width is None:
        return frame, 1.0

    height, width = frame.shape[:2]
    if width <= target_width:
        return frame, 1.0

    scale = target_width / float(width)
    new_height = max(1, int(height * scale))
    from Backends.src.utils.cv2_loader import cv2

    resized = cv2.resize(frame, (target_width, new_height))
    return resized, scale


def scale_detections_to_original(detections, scale: float):
    """Scale detection boxes/centers from resized inference space back to source frame."""
    if scale == 1.0 or not detections:
        return detections

    scaled = []
    inverse = 1.0 / scale
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        scaled.append(
            {
                **detection,
                "box": (
                    int(x1 * inverse),
                    int(y1 * inverse),
                    int(x2 * inverse),
                    int(y2 * inverse),
                ),
                "center": (int(center_x * inverse), int(center_y * inverse)),
            }
        )
    return scaled
