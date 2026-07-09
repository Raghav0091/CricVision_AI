"""Live delivery event detection + short clip capture (v1).

No Streamlit/YOLO imports. No model loading.
Simple motion + optional ball detections + optional pitch corridor.
Does not invent fake ball points or official DRS/LBW outcomes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from Backends.src.config.paths import OUTPUTS_DIR
from Backends.src.session_calibration import point_inside_live_pitch_corridor

# ponytail: short clip window is enough for later Video Analysis; not full live inference.
DEFAULT_CLIP_FRAMES = 90
DEFAULT_MIN_CLIP_FRAMES = 24
DEFAULT_COOLDOWN_FRAMES = 45
DEFAULT_INACTIVE_STOP_FRAMES = 18
DEFAULT_MOTION_START = 12.0
DEFAULT_MOTION_ACTIVE = 6.0
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "live_deliveries"


def create_delivery_capture_state() -> dict[str, Any]:
    return {
        "recording": False,
        "frames": [],
        "delivery_count": 0,
        "last_delivery_time": None,
        "cooldown_frames": 0,
        "inactive_frames": 0,
        "prev_gray_small": None,
        "notes": [],
    }


def _ensure_state(state: Any) -> dict[str, Any]:
    if isinstance(state, dict):
        base = create_delivery_capture_state()
        base.update(state)
        if not isinstance(base.get("frames"), list):
            base["frames"] = []
        if not isinstance(base.get("notes"), list):
            base["notes"] = []
        return base
    return create_delivery_capture_state()


def _downsample_gray(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    try:
        arr = np.asarray(frame)
    except Exception:
        return None
    if arr.ndim < 2:
        return None
    # BGR/RGB → luminance; already-gray stays 2D.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = (
            0.114 * arr[:, :, 0].astype(np.float32)
            + 0.587 * arr[:, :, 1].astype(np.float32)
            + 0.299 * arr[:, :, 2].astype(np.float32)
        )
    else:
        gray = arr.astype(np.float32)
    return gray[::8, ::8]


def estimate_frame_motion_score(frame: Any, prev_gray_small: Any) -> tuple[float, np.ndarray | None]:
    """Mean absolute difference on a downsampled grayscale frame."""
    current = _downsample_gray(frame)
    if current is None:
        return 0.0, None
    if prev_gray_small is None:
        return 0.0, current
    try:
        prev = np.asarray(prev_gray_small, dtype=np.float32)
    except Exception:
        return 0.0, current
    if prev.shape != current.shape:
        return 0.0, current
    return float(np.mean(np.abs(current - prev))), current


def _extract_ball_points(detections: Any) -> list[Any]:
    """Pull real ball centers from optional detections; never invent points."""
    if not detections:
        return []
    items = detections
    if isinstance(detections, dict):
        items = (
            detections.get("ball_detections")
            or detections.get("balls")
            or detections.get("detections")
            or []
        )
    points: list[Any] = []
    if not isinstance(items, (list, tuple)):
        return points
    for item in items:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name") or item.get("label") or "ball").lower()
        if class_name not in {"ball", "sports ball", "cricket_ball"}:
            # Allow unlabeled center/box payloads from lightweight callers.
            if "center" not in item and "box" not in item and "xyxy" not in item:
                continue
            if class_name not in {"ball", "sports ball", "cricket_ball", ""}:
                continue
        center = item.get("center")
        if isinstance(center, dict) and "x" in center and "y" in center:
            points.append(center)
            continue
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            points.append({"x": center[0], "y": center[1]})
            continue
        box = item.get("box") or item.get("xyxy")
        if isinstance(box, dict):
            try:
                points.append(
                    {
                        "x": (float(box["x1"]) + float(box["x2"])) / 2.0,
                        "y": (float(box["y1"]) + float(box["y2"])) / 2.0,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        elif isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                points.append(
                    {
                        "x": (float(box[0]) + float(box[2])) / 2.0,
                        "y": (float(box[1]) + float(box[3])) / 2.0,
                    }
                )
            except (TypeError, ValueError):
                continue
    return points


def _ball_supports_delivery(detections: Any, calibration: Any) -> bool:
    points = _extract_ball_points(detections)
    if not points:
        return False
    if not isinstance(calibration, dict) or not calibration.get("available"):
        return True
    for point in points:
        scored = point_inside_live_pitch_corridor(point, calibration, margin=50)
        if scored.get("inside") or scored.get("near"):
            return True
    return False


def update_delivery_capture_state(
    frame: Any,
    detections: Any = None,
    motion_score: float | None = None,
    calibration: Any = None,
    state: Any = None,
    *,
    clip_frames: int = DEFAULT_CLIP_FRAMES,
    min_clip_frames: int = DEFAULT_MIN_CLIP_FRAMES,
    cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    inactive_stop_frames: int = DEFAULT_INACTIVE_STOP_FRAMES,
    motion_start: float = DEFAULT_MOTION_START,
    motion_active: float = DEFAULT_MOTION_ACTIVE,
) -> dict[str, Any]:
    """Frame-by-frame delivery capture update. Defensive; no heavy analysis."""
    notes: list[str] = []
    state = _ensure_state(state)
    completed_clip = None
    delivery_detected = False

    if frame is None:
        notes.append("No frame provided.")
        return {
            "state": state,
            "delivery_detected": False,
            "recording": bool(state.get("recording")),
            "completed_clip": None,
            "notes": notes,
        }

    computed_motion, gray_small = estimate_frame_motion_score(
        frame,
        state.get("prev_gray_small"),
    )
    if gray_small is not None:
        state["prev_gray_small"] = gray_small
    if motion_score is None:
        motion_score = computed_motion
    else:
        try:
            motion_score = float(motion_score)
        except (TypeError, ValueError):
            motion_score = computed_motion

    cooldown = int(state.get("cooldown_frames") or 0)
    if cooldown > 0:
        state["cooldown_frames"] = cooldown - 1
        state["notes"] = notes
        return {
            "state": state,
            "delivery_detected": False,
            "recording": bool(state.get("recording")),
            "completed_clip": None,
            "notes": notes,
        }

    ball_hint = _ball_supports_delivery(detections, calibration)
    active_motion = motion_score >= motion_active
    start_motion = motion_score >= motion_start

    if not state.get("recording"):
        if start_motion or ball_hint:
            state["recording"] = True
            state["frames"] = [frame.copy() if hasattr(frame, "copy") else frame]
            state["inactive_frames"] = 0
            delivery_detected = True
            notes.append("Delivery detected — recording started.")
        state["notes"] = notes
        return {
            "state": state,
            "delivery_detected": delivery_detected,
            "recording": bool(state.get("recording")),
            "completed_clip": None,
            "notes": notes,
        }

    # Recording path.
    try:
        state["frames"].append(frame.copy() if hasattr(frame, "copy") else frame)
    except Exception:
        state["frames"].append(frame)

    if active_motion or ball_hint:
        state["inactive_frames"] = 0
    else:
        state["inactive_frames"] = int(state.get("inactive_frames") or 0) + 1

    frame_count = len(state["frames"])
    stop_for_length = frame_count >= max(1, int(clip_frames))
    stop_for_idle = (
        frame_count >= max(1, int(min_clip_frames))
        and int(state.get("inactive_frames") or 0) >= max(1, int(inactive_stop_frames))
    )

    if stop_for_length or stop_for_idle:
        completed_clip = list(state["frames"])
        state["frames"] = []
        state["recording"] = False
        state["inactive_frames"] = 0
        state["cooldown_frames"] = max(0, int(cooldown_frames))
        state["delivery_count"] = int(state.get("delivery_count") or 0) + 1
        state["last_delivery_time"] = datetime.now().isoformat(timespec="seconds")
        reason = "max frames" if stop_for_length else "inactivity"
        notes.append(f"Delivery clip completed ({reason}, {len(completed_clip)} frames).")

    state["notes"] = notes
    return {
        "state": state,
        "delivery_detected": delivery_detected,
        "recording": bool(state.get("recording")),
        "completed_clip": completed_clip,
        "notes": notes,
    }


def save_delivery_clip(
    frames: Any,
    output_dir: Any = None,
    fps: float = 30,
    prefix: str = "delivery",
) -> dict[str, Any]:
    """Save recorded delivery frames as mp4 under outputs/live_deliveries/."""
    notes: list[str] = []
    if not frames:
        notes.append("No frames to save.")
        return {"saved": False, "path": None, "frame_count": 0, "notes": notes}

    try:
        from Backends.src.utils.cv2_loader import cv2
    except Exception as error:
        notes.append(f"OpenCV unavailable: {error}")
        return {"saved": False, "path": None, "frame_count": 0, "notes": notes}

    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        notes.append(f"Could not create output dir: {error}")
        return {"saved": False, "path": None, "frame_count": len(frames), "notes": notes}

    first = frames[0]
    try:
        height, width = int(first.shape[0]), int(first.shape[1])
    except Exception:
        notes.append("Frames missing valid shape.")
        return {"saved": False, "path": None, "frame_count": len(frames), "notes": notes}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{prefix}_{timestamp}.mp4"
    try:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps) if fps and fps > 0 else 30.0,
            (width, height),
        )
    except Exception as error:
        notes.append(f"VideoWriter failed: {error}")
        return {"saved": False, "path": None, "frame_count": len(frames), "notes": notes}

    if not writer.isOpened():
        notes.append("VideoWriter could not open output file.")
        writer.release()
        return {"saved": False, "path": None, "frame_count": len(frames), "notes": notes}

    written = 0
    try:
        for frame in frames:
            if frame is None:
                continue
            try:
                if frame.shape[0] != height or frame.shape[1] != width:
                    frame = cv2.resize(frame, (width, height))
            except Exception:
                continue
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written <= 0:
        notes.append("No frames were written.")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"saved": False, "path": None, "frame_count": 0, "notes": notes}

    notes.append(f"Saved delivery clip with {written} frames.")
    return {
        "saved": True,
        "path": str(path),
        "frame_count": written,
        "notes": notes,
    }
