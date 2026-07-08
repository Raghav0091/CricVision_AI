"""Detection health diagnostics built from existing analysis outputs (visibility only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Backends.src.analysis.frame_detection_utils import normalize_frame_detections
from Backends.src.utils.cv2_loader import cv2

REQUIRED_HEALTH_KEYS = (
    "model_path",
    "model_exists",
    "model_name",
    "speed_mode",
    "detection_preset",
    "confidence_threshold",
    "imgsz",
    "roi_enabled",
    "total_frames",
    "processed_frames",
    "frames_with_raw_ball",
    "raw_ball_detections",
    "ball_detection_rate",
    "selected_ball_points",
    "ball_tracking_rate",
    "kalman_predicted_frames",
    "interpolated_ball_frames",
    "tracker_recoveries",
    "low_confidence_ball_frames",
    "overall_tracking_quality",
    "review_flags",
    "visual_observer_summary",
    "failure_type",
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_tracking_rate(rate: Any) -> float | None:
    """Return tracking rate as a 0-1 fraction."""
    value = _safe_float(rate)
    if value is None:
        return None
    if value > 1.0:
        return value / 100.0
    return value


def _count_frames_with_ball(frame_detections: Any) -> int:
    frames = normalize_frame_detections(frame_detections or [])
    return sum(1 for item in frames if item.get("ball_detections"))


def _count_raw_ball_detections(frame_detections: Any) -> int:
    frames = normalize_frame_detections(frame_detections or [])
    return sum(len(item.get("ball_detections") or []) for item in frames)


def _derive_selected_ball_points(result: dict[str, Any]) -> int:
    explicit = result.get("selected_ball_points")
    if explicit is not None:
        return _safe_int(explicit, 0)

    ball_positions = result.get("ball_positions")
    if isinstance(ball_positions, (list, tuple)):
        return sum(1 for point in ball_positions if point is not None)

    detected_frames = result.get("ball_detected_frames")
    if detected_frames is not None:
        return _safe_int(detected_frames, 0)

    total_frames = _safe_int(result.get("total_frames"), 0)
    tracking_rate = _normalize_tracking_rate(result.get("ball_tracking_rate"))
    interpolated = _safe_int(result.get("interpolated_ball_frames"), 0)
    if total_frames > 0 and tracking_rate is not None:
        usable = int(round(tracking_rate * total_frames))
        return max(0, usable - interpolated)

    return 0


def classify_failure_type(
    *,
    raw_ball_detections: int,
    selected_ball_points: int,
    ball_tracking_rate: Any,
    total_frames: int | None = None,
) -> str:
    """Classify detection/tracking health without touching analysis behavior."""
    if total_frames is not None and total_frames <= 0:
        return "unknown"

    rate = _normalize_tracking_rate(ball_tracking_rate)
    if raw_ball_detections == 0 and rate is None and selected_ball_points == 0:
        if total_frames in {None, 0}:
            return "unknown"

    if raw_ball_detections == 0:
        return "detector_failure"
    if selected_ball_points < 5:
        return "tracker_failure"
    if rate is None:
        return "unknown"
    if rate >= 0.45:
        return "good_track"
    if selected_ball_points >= 5:
        return "partial_track"
    return "unknown"


def _build_visual_observer_summary(repair: dict[str, Any] | None) -> str:
    repair = repair or {}
    repaired = _safe_int(repair.get("repaired_frames"), 0)
    removed = _safe_int(repair.get("removed_or_downgraded_frames"), 0)
    suspicious = _safe_int(repair.get("suspicious_detections"), 0)
    confidence = repair.get("repair_confidence") or "Unknown"
    decision = repair.get("agent_decision") or ""
    parts = [f"Repair confidence: {confidence}."]
    if repaired:
        parts.append(f"Repaired {repaired} frame(s).")
    if removed:
        parts.append(f"Removed/downgraded {removed} detection(s).")
    if suspicious:
        parts.append(f"Flagged {suspicious} suspicious detection(s).")
    if decision:
        parts.append(str(decision))
    return " ".join(parts).strip() or "Visual Observer repair summary unavailable."


def build_detection_health(
    analysis_result: dict[str, Any] | None,
    *,
    model_path: str | Path | None = None,
    model_key: str | None = None,
    model_name: str | None = None,
    speed_mode: str | None = None,
    detection_preset: str | None = None,
    confidence_threshold: float | None = None,
    imgsz: int | None = None,
    roi_enabled: bool | None = None,
) -> dict[str, Any]:
    """Build a defensive detection-health summary from an analysis result dict."""
    result = analysis_result or {}
    observer = result.get("observer_timeline") or {}
    repair = result.get("visual_observer_repair") or {}
    agent = result.get("agent_info") or {}

    resolved_model_path = model_path or result.get("model_path")
    if resolved_model_path is None and model_key:
        try:
            from Backends.src.models.model_registry import get_model_path

            candidate = get_model_path(model_key)
            if candidate is not None:
                resolved_model_path = str(candidate)
        except Exception:
            resolved_model_path = None

    path_obj = Path(str(resolved_model_path)) if resolved_model_path else None
    model_exists = bool(path_obj and path_obj.is_file())

    resolved_model_name = (
        model_name
        or result.get("ball_model_used")
        or result.get("active_model")
        or model_key
        or "Unknown"
    )

    total_frames = result.get("total_frames", observer.get("total_frames"))
    processed_frames = result.get(
        "processed_frames",
        observer.get("processed_frames", total_frames),
    )

    raw_frame_detections = result.get("raw_frame_detections")
    frames_with_raw_ball = result.get("ball_detected_frames")
    if frames_with_raw_ball is None:
        frames_with_raw_ball = _count_frames_with_ball(raw_frame_detections)

    raw_ball_detections = result.get("total_ball_detections")
    if raw_ball_detections is None:
        raw_ball_detections = _count_raw_ball_detections(raw_frame_detections)
    raw_ball_detections = _safe_int(raw_ball_detections, 0)
    selected_ball_points = _derive_selected_ball_points(result)

    review_flags = result.get("review_flags")
    if review_flags is None:
        review_flags = agent.get("review_flags")
    if not isinstance(review_flags, list):
        review_flags = []

    if roi_enabled is None:
        roi_frames = _safe_int(result.get("roi_detected_frames"), 0)
        if roi_frames > 0:
            roi_enabled = True
        elif total_frames not in {None, 0}:
            roi_enabled = False

    health = {
        "model_path": str(resolved_model_path) if resolved_model_path else "Unknown",
        "model_exists": model_exists,
        "model_name": resolved_model_name,
        "speed_mode": speed_mode or result.get("speed_mode") or "Unknown",
        "detection_preset": detection_preset or result.get("active_preset") or "Unknown",
        "confidence_threshold": confidence_threshold,
        "imgsz": imgsz,
        "roi_enabled": roi_enabled,
        "total_frames": _safe_int(total_frames, 0),
        "processed_frames": _safe_int(processed_frames, 0),
        "frames_with_raw_ball": _safe_int(frames_with_raw_ball, 0),
        "raw_ball_detections": raw_ball_detections,
        "ball_detection_rate": result.get("ball_detection_rate"),
        "selected_ball_points": selected_ball_points,
        "ball_tracking_rate": result.get("ball_tracking_rate"),
        "kalman_predicted_frames": _safe_int(result.get("kalman_predicted_frames"), 0),
        "interpolated_ball_frames": _safe_int(result.get("interpolated_ball_frames"), 0),
        "tracker_recoveries": _safe_int(result.get("tracker_recoveries"), 0),
        "low_confidence_ball_frames": _safe_int(
            result.get("low_confidence_ball_frames"),
            _safe_int(observer.get("low_confidence_ball_frames"), 0),
        ),
        "overall_tracking_quality": result.get("overall_tracking_quality") or "Unknown",
        "review_flags": review_flags,
        "visual_observer_summary": _build_visual_observer_summary(repair),
        "failure_type": classify_failure_type(
            raw_ball_detections=raw_ball_detections,
            selected_ball_points=selected_ball_points,
            ball_tracking_rate=result.get("ball_tracking_rate"),
            total_frames=_safe_int(total_frames, 0),
        ),
    }
    return health


def _sample_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    sample_count = max(6, min(12, int(sample_count)))
    if total_frames <= 0:
        return []
    if total_frames <= sample_count:
        return list(range(total_frames))
    step = max(1, total_frames // sample_count)
    indices = list(range(0, total_frames, step))[:sample_count]
    if indices[-1] != total_frames - 1 and len(indices) < sample_count:
        indices.append(total_frames - 1)
    return indices[:sample_count]


def _parse_ball_detections_from_result(
    result: Any,
    class_names: dict[int, str],
) -> list[dict[str, Any]]:
    """Extract ball detections from a single YOLO predict result (no scaling)."""
    detections: list[dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].cpu().numpy())
        class_name = class_names.get(class_id)
        if class_name != "ball":
            continue
        detection_confidence = float(box.conf[0].cpu().numpy())
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().numpy())
        bbox = (x1, y1, x2, y2)
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": detection_confidence,
                "bbox": bbox,
                "box": bbox,
                "center": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            }
        )
    return detections


def run_raw_detection_preview(
    video_path: str | Path,
    *,
    model_key: str | None = "current_best",
    model_path: str | Path | None = None,
    use_ensemble: bool = False,
    confidence: float | None = None,
    imgsz: int | None = None,
    sample_count: int = 8,
    speed_mode: str | None = None,
    detection_preset: str | None = None,
) -> dict[str, Any]:
    """Run YOLO on sampled frames only; no tracking, reports, or processed video."""
    from Backends.src.analysis.analysis_speed import get_analysis_mode_settings
    from Backends.src.video_pipeline import annotation_writer as annotations
    from Backends.src.video_pipeline.detection_pipeline import (
        load_detection_model,
        load_ensemble_models,
        map_model_classes,
    )

    video_path = Path(video_path)
    if not video_path.is_file():
        return {"success": False, "error": f"Video not found: {video_path}"}

    speed_settings = get_analysis_mode_settings(speed_mode or "Smart Balanced")
    inference_imgsz = int(imgsz or speed_settings.get("yolo_imgsz", 768))
    confidence_value = 0.25 if confidence is None else float(confidence)

    ensemble_models: list[dict[str, Any]] = []
    ball_model = None
    class_names: dict[int, str] = {}

    if use_ensemble:
        ensemble_models = load_ensemble_models()
        if not ensemble_models:
            return {"success": False, "error": "No ensemble models available for raw preview."}
    else:
        ball_model = load_detection_model(model_key=model_key, model_path=model_path)
        if ball_model is None:
            return {"success": False, "error": "Selected detection model is unavailable."}
        class_names = map_model_classes(ball_model)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"success": False, "error": "Could not open uploaded video for raw preview."}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        target_indices = set(_sample_frame_indices(total_frames, sample_count))
        stride = None
    else:
        target_indices = None
        stride = 30

    preview_frames = []
    raw_ball_total = 0
    frames_with_ball = 0
    confidence_values: list[float] = []
    frame_index = 0
    max_samples = max(6, min(12, int(sample_count)))

    while True:
        success, frame = cap.read()
        if not success:
            break
        should_sample = (
            frame_index in target_indices
            if target_indices is not None
            else frame_index % stride == 0
        )
        if not should_sample:
            frame_index += 1
            continue

        ball_detections: list[dict[str, Any]] = []
        if use_ensemble:
            for entry in ensemble_models:
                predict_result = entry["model"].predict(
                    source=frame,
                    conf=confidence_value,
                    imgsz=inference_imgsz,
                    verbose=False,
                )[0]
                ball_detections.extend(
                    _parse_ball_detections_from_result(predict_result, entry["class_names"])
                )
        else:
            predict_result = ball_model.predict(
                source=frame,
                conf=confidence_value,
                imgsz=inference_imgsz,
                verbose=False,
            )[0]
            ball_detections = _parse_ball_detections_from_result(predict_result, class_names)

        annotated = frame.copy()
        annotations.draw_ball_detections(annotated, ball_detections)
        for detection in ball_detections:
            center_x, center_y = detection["center"]
            cv2.putText(
                annotated,
                f"F{frame_index}",
                (max(0, center_x - 20), max(20, center_y - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        if ball_detections:
            frames_with_ball += 1
            raw_ball_total += len(ball_detections)
            confidence_values.extend(
                float(item.get("confidence", 0.0)) for item in ball_detections
            )

        preview_frames.append(
            {
                "frame_index": frame_index,
                "image_bgr": annotated,
                "ball_detections": ball_detections,
            }
        )
        frame_index += 1
        if len(preview_frames) >= max_samples:
            break

    cap.release()

    average_confidence = None
    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)

    resolved_path = model_path
    if resolved_path is None and model_key:
        try:
            from Backends.src.models.model_registry import get_model_path

            candidate = get_model_path(model_key)
            if candidate is not None:
                resolved_path = candidate
        except Exception:
            resolved_path = None

    return {
        "success": True,
        "sampled_frames": len(preview_frames),
        "frames_with_ball": frames_with_ball,
        "raw_ball_detections": raw_ball_total,
        "average_confidence": average_confidence,
        "model_path": str(resolved_path) if resolved_path else "Unknown",
        "model_name": model_key or "Unknown",
        "detection_preset": detection_preset or "Unknown",
        "confidence_threshold": confidence_value,
        "imgsz": inference_imgsz,
        "speed_mode": speed_mode or speed_settings.get("mode", "Smart Balanced"),
        "frames": preview_frames,
    }
