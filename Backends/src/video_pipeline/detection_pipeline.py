"""Shared ball, stump, ROI, and model-selection helpers.

This module is import-safe: YOLO weights are loaded only by
``load_detection_model`` or ``load_ensemble_models`` at analysis runtime.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st

from Backends.src.analysis.field_zones import normalize_handedness
from Backends.src.models.model_loader import get_cached_yolo_model
from Backends.src.models.model_registry import get_model_path
from Backends.src.utils.cv2_loader import cv2
from Backends.src.video_pipeline.annotation_writer import draw_label

BALL_MODEL_PATH = Path("Models/ball_detector/best.pt")
CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
EXTERNAL_BALL_MODEL_PATH = Path("Models/cricket_objects/best_external.pt")
ENSEMBLE_MODEL_NAME = "Ensemble: All Ball Models + Stumps"
LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35


@st.cache_resource
def load_yolo_model(model_path_str):
    model_path = Path(model_path_str)
    if not model_path.exists():
        return None
    from ultralytics import YOLO

    return YOLO(str(model_path))


def get_ensemble_model_configs():
    return [
        {
            "name": "Ball + Stump Detector",
            "path": CRICKET_OBJECTS_MODEL_PATH,
            "model_key": "current_best",
            "use_ball": True,
            "use_stump": True,
        },
        {
            "name": "Old Ball Detector",
            "path": BALL_MODEL_PATH,
            "model_key": None,
            "use_ball": True,
            "use_stump": False,
        },
        {
            "name": "External Ball Model",
            "path": EXTERNAL_BALL_MODEL_PATH,
            "model_key": None,
            "use_ball": True,
            "use_stump": False,
        },
    ]


def get_available_ensemble_model_names():
    available = []
    for config in get_ensemble_model_configs():
        model_key = config.get("model_key")
        if model_key:
            path = get_model_path(model_key)
            if path is not None and path.is_file():
                available.append(config["name"])
        elif config["path"].exists():
            available.append(config["name"])
    return available


def get_model_options():
    options = {
        "Ball + Stump Detector": {
            "path": CRICKET_OBJECTS_MODEL_PATH,
            "model_key": "current_best",
        },
    }
    if len(get_available_ensemble_model_names()) >= 2:
        options[ENSEMBLE_MODEL_NAME] = {
            "path": None,
            "model_key": None,
            "ensemble": True,
        }
    return options


def get_model_names(model):
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return names
    if isinstance(names, (list, tuple)):
        return {index: name for index, name in enumerate(names)}
    return {}


def map_model_classes(model):
    class_names = {}
    for class_id, raw_name in get_model_names(model).items():
        normalized_name = str(raw_name).lower()
        if "ball" in normalized_name:
            class_names[int(class_id)] = "ball"
        elif any(name in normalized_name for name in ("stump", "stumps", "wicket")):
            class_names[int(class_id)] = "stump"
    return class_names


def load_detection_model(model_key=None, model_path=None):
    if model_key:
        return get_cached_yolo_model(model_key)
    if model_path is None:
        return None
    return load_yolo_model(str(model_path))


def load_ensemble_models():
    models = []
    for config in get_ensemble_model_configs():
        model = load_detection_model(
            model_key=config.get("model_key"),
            model_path=config.get("path"),
        )
        if model is not None:
            models.append(
                {
                    **config,
                    "model": model,
                    "class_names": map_model_classes(model),
                }
            )
    return models


def _resolve_ensemble_models(model_paths=None):
    if model_paths is None:
        return load_ensemble_models()

    models = []
    for config in get_ensemble_model_configs():
        if config["path"] not in model_paths and str(config["path"]) not in model_paths:
            continue
        model = load_detection_model(
            model_key=config.get("model_key"),
            model_path=config.get("path"),
        )
        if model is not None:
            models.append(
                {
                    **config,
                    "model": model,
                    "class_names": map_model_classes(model),
                }
            )
    return models


def get_box_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def choose_main_ball(ball_detections, previous_center=None):
    if not ball_detections:
        return None
    if previous_center is None:
        return max(ball_detections, key=lambda item: item["confidence"])

    previous_x, previous_y = previous_center

    def distance_from_previous(item):
        center_x, center_y = item["center"]
        return ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5

    return min(ball_detections, key=distance_from_previous)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    return intersection / union if union > 0 else 0


def non_max_suppression(detections, iou_threshold=0.4):
    pending = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept = []
    while pending:
        best = pending.pop(0)
        kept.append(best)
        pending = [
            detection
            for detection in pending
            if calculate_iou(best["box"], detection["box"]) < iou_threshold
        ]
    return kept


def _run_single_model_detection(frame, model, confidence, imgsz):
    return model.predict(
        source=frame,
        conf=confidence,
        imgsz=imgsz,
        verbose=False,
    )[0]


def _collect_model_detections(
    result,
    class_names,
    confidence,
    model_name,
    ball_confidence=None,
):
    ball_detections = []
    stump_detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return ball_detections, stump_detections

    for box in result.boxes:
        class_id = int(box.cls[0].cpu().numpy())
        class_name = class_names.get(class_id)
        if class_name is None:
            continue
        detection_confidence = float(box.conf[0].cpu().numpy())
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().numpy())
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": detection_confidence,
            "box": (x1, y1, x2, y2),
            "center": get_box_center(x1, y1, x2, y2),
            "model_name": model_name,
        }
        if class_name == "ball" and detection_confidence >= (ball_confidence or confidence):
            ball_detections.append(detection)
        elif class_name == "stump" and detection_confidence >= confidence:
            stump_detections.append(detection)
    return ball_detections, stump_detections


def _collect_low_confidence_balls(result, class_names, model_name):
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    for box in result.boxes:
        class_id = int(box.cls[0].cpu().numpy())
        if class_names.get(class_id) != "ball":
            continue
        confidence = float(box.conf[0].cpu().numpy())
        if not 0.10 <= confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD:
            continue
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].cpu().numpy())
        detections.append(
            {
                "class_id": class_id,
                "class_name": "ball",
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": get_box_center(x1, y1, x2, y2),
                "model_name": model_name,
            }
        )
    return detections


def _offset_detections(detections, offset_x, offset_y):
    adjusted = []
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        adjusted.append(
            {
                **detection,
                "box": (
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ),
                "center": (center_x + offset_x, center_y + offset_y),
            }
        )
    return adjusted


def estimate_pitch_roi(frame_shape, stump_detections, previous_roi=None):
    height, width = frame_shape[:2]
    if not stump_detections:
        return previous_roi
    main_stump = max(
        stump_detections,
        key=lambda item: (item["box"][2] - item["box"][0])
        * (item["box"][3] - item["box"][1]),
    )
    x1, y1, x2, y2 = main_stump["box"]
    stump_center_x = int((x1 + x2) / 2)
    stump_width = max(x2 - x1, 1)
    stump_height = max(y2 - y1, 1)
    corridor_width = int(
        min(width * 0.90, max(stump_width * 10, stump_height * 4, width * 0.38))
    )
    roi_x1 = max(0, stump_center_x - corridor_width // 2)
    roi_x2 = min(width, stump_center_x + corridor_width // 2)
    if roi_x2 - roi_x1 < width * 0.25:
        extra = int((width * 0.25 - (roi_x2 - roi_x1)) / 2)
        roi_x1 = max(0, roi_x1 - extra)
        roi_x2 = min(width, roi_x2 + extra)
    roi_y1 = max(0, int(min(y1 - height * 0.18, height * 0.20)))
    roi_y2 = height
    if roi_y2 - roi_y1 < height * 0.45:
        roi_y1 = max(0, int(height * 0.35))
    return int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2)


def _crop_frame(frame, roi_box):
    if roi_box is None:
        return frame, (0, 0), None
    x1, y1, x2, y2 = roi_box
    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0), None
    return frame[y1:y2, x1:x2], (x1, y1), roi_box


def _create_local_search_roi(frame_shape, center, missing_frames=1):
    if center is None:
        return None
    height, width = frame_shape[:2]
    center_x, center_y = center
    window_size = int(
        max(96, min(width, height) * 0.18)
        * (1 + min(missing_frames, 6) * 0.12)
    )
    half_window = window_size // 2
    x1 = max(0, int(center_x - half_window))
    y1 = max(0, int(center_y - half_window))
    x2 = min(width, int(center_x + half_window))
    y2 = min(height, int(center_y + half_window))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def run_ensemble_detection(frame, models, confidence, imgsz):
    if models is None or not models or not isinstance(models[0], dict):
        models = _resolve_ensemble_models(models)
    balls = []
    stumps = []
    low_confidence_balls = []
    for config in models:
        result = _run_single_model_detection(
            frame,
            config["model"],
            min(confidence, 0.10),
            imgsz,
        )
        model_balls, model_stumps = _collect_model_detections(
            result,
            config["class_names"],
            confidence,
            config["name"],
        )
        low_confidence_balls.extend(
            _collect_low_confidence_balls(
                result,
                config["class_names"],
                config["name"],
            )
        )
        if config["use_ball"]:
            balls.extend(model_balls)
        if config["use_stump"]:
            stumps.extend(model_stumps)
    return {
        "ball_detections": non_max_suppression(balls),
        "stump_detections": stumps,
        "low_confidence_ball_detections": non_max_suppression(low_confidence_balls),
    }


def run_pitch_roi_detection(
    frame,
    stump_model,
    stump_class_names,
    confidence,
    imgsz,
    previous_roi=None,
    ball_model=None,
    ball_class_names=None,
    ensemble_models=None,
    use_ensemble=False,
    ball_confidence=None,
    speed_settings=None,
    detect_stump=True,
    locked_stump_detections=None,
    use_roi=True,
):
    speed_settings = speed_settings or {}
    stump_detections = list(locked_stump_detections or [])
    same_model = (
        not use_ensemble
        and ball_model is not None
        and stump_model is not None
        and ball_model is stump_model
    )
    if detect_stump and same_model and speed_settings.get("single_pass_same_model", False):
        started_at = time.perf_counter()
        result = _run_single_model_detection(
            frame,
            stump_model,
            min(confidence, 0.10),
            imgsz,
        )
        balls, stumps = _collect_model_detections(
            result,
            stump_class_names,
            confidence,
            "Ball + Stump Detector",
            ball_confidence=ball_confidence,
        )
        low_confidence = _collect_low_confidence_balls(
            result,
            stump_class_names,
            "Ball + Stump Detector",
        )
        return {
            "ball_detections": non_max_suppression(balls),
            "stump_detections": stumps,
            "low_confidence_ball_detections": non_max_suppression(low_confidence),
            "roi_box": estimate_pitch_roi(frame.shape, stumps, previous_roi),
            "full_frame_time_ms": (time.perf_counter() - started_at) * 1000,
            "roi_time_ms": 0,
            "used_roi": False,
            "single_pass": True,
        }

    full_frame_time_ms = 0.0
    if detect_stump:
        started_at = time.perf_counter()
        result = _run_single_model_detection(frame, stump_model, confidence, imgsz)
        _, stump_detections = _collect_model_detections(
            result,
            stump_class_names,
            confidence,
            "Ball + Stump Detector",
        )
        full_frame_time_ms = (time.perf_counter() - started_at) * 1000

    roi_box = (
        estimate_pitch_roi(frame.shape, stump_detections, previous_roi)
        if use_roi
        else None
    )
    if use_roi and roi_box is not None:
        roi_frame, (offset_x, offset_y), active_roi = _crop_frame(frame, roi_box)
    else:
        roi_frame, offset_x, offset_y, active_roi = frame, 0, 0, None
    used_roi = active_roi is not None and use_roi
    started_at = time.perf_counter()

    if use_ensemble:
        detection_result = run_ensemble_detection(
            roi_frame,
            ensemble_models,
            confidence,
            imgsz,
        )
        balls = detection_result["ball_detections"]
        low_confidence = detection_result.get("low_confidence_ball_detections", [])
    else:
        result = _run_single_model_detection(
            roi_frame,
            ball_model,
            min(confidence, 0.10),
            imgsz,
        )
        balls, _ = _collect_model_detections(
            result,
            ball_class_names,
            confidence,
            "Selected Model",
            ball_confidence=ball_confidence,
        )
        low_confidence = _collect_low_confidence_balls(
            result,
            ball_class_names,
            "Selected Model",
        )

    detection_time_ms = (time.perf_counter() - started_at) * 1000
    if used_roi:
        balls = _offset_detections(balls, offset_x, offset_y)
        low_confidence = _offset_detections(low_confidence, offset_x, offset_y)
        roi_time_ms = detection_time_ms
    else:
        full_frame_time_ms += detection_time_ms
        roi_time_ms = 0
    return {
        "ball_detections": non_max_suppression(balls),
        "stump_detections": stump_detections,
        "low_confidence_ball_detections": non_max_suppression(low_confidence),
        "roi_box": active_roi,
        "full_frame_time_ms": full_frame_time_ms,
        "roi_time_ms": roi_time_ms,
        "used_roi": used_roi,
    }


def run_local_redetection(
    frame,
    search_center,
    confidence,
    imgsz,
    missing_frames,
    ball_model=None,
    ball_class_names=None,
    ensemble_models=None,
    use_ensemble=False,
):
    search_roi = _create_local_search_roi(frame.shape, search_center, missing_frames)
    if search_roi is None:
        return {"ball_detections": [], "search_roi": None, "recovered": False}

    search_frame, (offset_x, offset_y), active_roi = _crop_frame(frame, search_roi)
    recovery_confidence = max(0.08, min(confidence, 0.18))
    if use_ensemble:
        balls = run_ensemble_detection(
            search_frame,
            ensemble_models,
            recovery_confidence,
            max(imgsz, 960),
        )["ball_detections"]
    else:
        result = _run_single_model_detection(
            search_frame,
            ball_model,
            recovery_confidence,
            max(imgsz, 960),
        )
        balls, _ = _collect_model_detections(
            result,
            ball_class_names,
            recovery_confidence,
            "Local Re-detection",
            ball_confidence=recovery_confidence,
        )
    balls = _offset_detections(balls, offset_x, offset_y)
    return {
        "ball_detections": non_max_suppression(balls),
        "search_roi": active_roi,
        "recovered": bool(balls),
    }


def estimate_auto_pitch_corners(frame_shape, stump_detections):
    if not stump_detections:
        return None
    height, width = frame_shape[:2]
    main_stump = max(
        stump_detections,
        key=lambda item: (item["box"][2] - item["box"][0])
        * (item["box"][3] - item["box"][1]),
    )
    x1, _, x2, y2 = main_stump["box"]
    center_x = int((x1 + x2) / 2)
    stump_width = max(x2 - x1, 1)
    top_half = max(stump_width * 2.2, width * 0.10)
    bottom_half = max(stump_width * 6.0, width * 0.30)
    top_y = max(0, int(y2 + height * 0.02))
    return [
        (max(0, int(center_x - top_half)), top_y),
        (min(width - 1, int(center_x + top_half)), top_y),
        (max(0, int(center_x - bottom_half)), height - 1),
        (min(width - 1, int(center_x + bottom_half)), height - 1),
    ]


def compute_pitch_homography(pitch_points):
    if pitch_points is None or len(pitch_points) != 4:
        return None
    source = np.array(pitch_points, dtype="float32")
    target = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype="float32",
    )
    return cv2.getPerspectiveTransform(source, target)


def transform_point_to_pitch(point, homography):
    if point is None or homography is None:
        return None
    points = np.array([[[float(point[0]), float(point[1])]]], dtype="float32")
    transformed = cv2.perspectiveTransform(points, homography)[0][0]
    return (
        min(max(float(transformed[0]), 0.0), 1.0),
        min(max(float(transformed[1]), 0.0), 1.0),
    )


def estimate_line_from_pitch_x(pitch_x, batter_handedness="right"):
    if pitch_x is None:
        return "Unknown"
    is_left_handed = normalize_handedness(batter_handedness) == "left"
    left_label = "Leg side" if is_left_handed else "Off side"
    right_label = "Off side" if is_left_handed else "Leg side"
    if pitch_x < 0.38:
        return left_label
    if pitch_x > 0.62:
        return right_label
    return "Middle"


def estimate_length_from_pitch_y(pitch_y):
    if pitch_y is None:
        return "Unknown"
    if pitch_y >= 0.84:
        return "Yorker"
    if pitch_y >= 0.68:
        return "Full"
    if pitch_y >= 0.45:
        return "Good Length"
    return "Short"


def estimate_line_from_stumps(bounce_point, stump_detections, batter_handedness="right"):
    if bounce_point is None or not stump_detections:
        return "Unknown"
    bounce_x, _ = bounce_point
    main_stump = max(
        stump_detections,
        key=lambda item: item["box"][2] - item["box"][0],
    )
    x1, _, x2, _ = main_stump["box"]
    margin = int((x2 - x1) * 0.4)
    is_left_handed = normalize_handedness(batter_handedness) == "left"
    if bounce_x < x1 - margin:
        return "Leg side" if is_left_handed else "Off side"
    if bounce_x > x2 + margin:
        return "Off side" if is_left_handed else "Leg side"
    return "Middle"


def estimate_length_from_bounce(bounce_point, frame_height):
    if bounce_point is None or frame_height <= 0:
        return "Unknown"
    bounce_ratio = bounce_point[1] / frame_height
    if bounce_ratio >= 0.82:
        return "Yorker"
    if bounce_ratio >= 0.68:
        return "Full"
    if bounce_ratio >= 0.48:
        return "Good Length"
    return "Short"


def get_nearest_stump_detections(stump_detections_by_frame, frame_index):
    if frame_index is None or not stump_detections_by_frame:
        return []
    frame_index = min(frame_index, len(stump_detections_by_frame) - 1)
    for detections in reversed(stump_detections_by_frame[: frame_index + 1]):
        if detections:
            return detections
    for detections in stump_detections_by_frame[frame_index + 1 :]:
        if detections:
            return detections
    return []


def has_enough_ball_movement(trajectory_points, min_distance=40):
    if len(trajectory_points) < 2:
        return False
    start_x, start_y = trajectory_points[0]
    end_x, end_y = trajectory_points[-1]
    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
    return distance >= min_distance
