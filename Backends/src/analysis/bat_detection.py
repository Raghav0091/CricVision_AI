"""Frame-level bat/ball detection and possible-impact estimation helpers."""

from math import hypot

import cv2


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
    return hypot(float(point1[0]) - float(point2[0]), float(point1[1]) - float(point2[1]))


def _distance_to_bbox(point, bbox) -> float:
    x1, y1, x2, y2 = bbox
    closest_x = min(max(float(point[0]), x1), x2)
    closest_y = min(max(float(point[1]), y1), y2)
    return calculate_distance(point, (closest_x, closest_y))


def _ball_center_for_frame(ball_tracks, frame_index):
    if isinstance(ball_tracks, dict):
        value = ball_tracks.get(frame_index, ball_tracks.get(str(frame_index)))
    elif frame_index < len(ball_tracks):
        value = ball_tracks[frame_index]
    else:
        value = None
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("center") or value.get("ball_center")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[:2]
    return None


def find_possible_impact_frame(ball_tracks, bat_detections_by_frame, max_distance: float = 80) -> dict:
    from Backends.src.analysis.impact_detection import detect_bat_ball_impact

    frame_detections = []
    max_frame_count = len(ball_tracks or [])
    detections_by_frame = bat_detections_by_frame or {}
    if isinstance(detections_by_frame, dict) and detections_by_frame:
        max_frame_count = max(max_frame_count, max(int(key) for key in detections_by_frame) + 1)

    for frame_index in range(max_frame_count):
        ball_center = _ball_center_for_frame(ball_tracks or [], frame_index)
        ball_detections = []
        if ball_center is not None:
            ball_detections.append({"center": ball_center})
        if isinstance(detections_by_frame, dict):
            bat_detections = detections_by_frame.get(frame_index, detections_by_frame.get(str(frame_index), []))
        else:
            bat_detections = detections_by_frame[frame_index] if frame_index < len(detections_by_frame) else []
        frame_detections.append(
            {
                "frame_index": frame_index,
                "ball_detections": ball_detections,
                "bat_detections": bat_detections,
            }
        )

    return detect_bat_ball_impact(frame_detections)


def draw_bat_detections(frame, bat_detections):
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
