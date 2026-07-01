"""Frame-level bat/ball detection and possible-impact estimation helpers."""

from Backends.src.analysis.frame_detection_utils import calculate_point_distance


def _class_names(model) -> dict:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return names
    if isinstance(names, (list, tuple)):
        return dict(enumerate(names))
    return {}


def _detect_class(frame, model, target_class: str, conf: float) -> list[dict]:
    if frame is None or model is None:
        return []
    try:
        results = model.predict(source=frame, conf=conf, verbose=False)
    except Exception:
        return []
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = _class_names(model)
    detections = []
    for box in boxes:
        try:
            class_id = int(box.cls[0].item())
            class_name = str(names.get(class_id, target_class)).lower()
            # Single-class exported models are valid even if their label is generic.
            if target_class not in class_name and len(names) > 1:
                continue
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
            }
        )
    return detections


def detect_bat_in_frame(frame, bat_model, conf: float = 0.25) -> list[dict]:
    return _detect_class(frame, bat_model, "bat", conf)


def detect_ball_in_frame(frame, ball_model, conf: float = 0.25) -> list[dict]:
    return _detect_class(frame, ball_model, "ball", conf)


def calculate_distance(point1, point2) -> float:
    """Euclidean distance between two points."""
    value = calculate_point_distance(point1, point2)
    return value if value is not None else 0.0


def draw_bat_detections(frame, bat_detections):
    from Backends.src.utils.cv2_loader import cv2

    for detection in bat_detections or []:
        x1, y1, x2, y2 = detection.get("bbox", detection.get("box"))
        confidence = detection.get("confidence", 0.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(
            frame,
            f"bat {confidence:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
        )
    return frame


def draw_impact_marker(frame, impact_info, current_frame_index):
    from Backends.src.analysis.impact_detection import draw_impact_marker as draw_marker

    return draw_marker(frame, impact_info, current_frame_index)
