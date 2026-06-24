import tempfile
import subprocess
import csv
import json
import zipfile
import time
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import streamlit as st

from Backends.src.utils.cv2_loader import cv2

from Backends.src.analysis.cricket_agent import (
    calculate_detection_quality,
    detect_analysis_warnings,
    generate_coaching_feedback,
    generate_delivery_report,
)
from Backends.src.analysis.field_zones import generate_wagon_wheel_data
from Backends.src.analysis.field_zones import (
    find_nearest_fielder,
    normalize_handedness,
    save_field_analysis_history,
    save_field_setup,
    suggest_field_adjustment,
)
from Backends.src.tracking.ball_tracking_utils import (
    BallKalmanTracker,
    calculate_tracking_quality,
    detect_bounce_by_direction_change,
    get_tracking_quality_label,
    interpolate_missing_positions,
    smooth_trajectory,
)
from Backends.src.models.model_loader import get_cached_yolo_model
from Backends.src.models.model_registry import (
    get_model_info,
    get_model_path,
    validate_model_paths,
)


BALL_MODEL_PATH = Path("Models/ball_detector/best.pt")
CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
EXTERNAL_BALL_MODEL_PATH = Path("Models/cricket_objects/best_external.pt")
OUTPUT_DIR = Path("outputs/video_analysis")
PROCESSED_VIDEO_DIR = Path("outputs/processed_videos")
REPORTS_DIR = Path("outputs/reports")
REVIEW_FRAMES_DIR = Path("outputs/review_frames")
REVIEW_FRAMES_CSV = REVIEW_FRAMES_DIR / "review_frames.csv"
FIELD_ZONE_CORRECTIONS_CSV = Path("outputs/field_zone_corrections.csv")
ENSEMBLE_MODEL_NAME = "Ensemble: All Ball Models + Stumps"
LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.35
MAX_REVIEW_FRAMES_PER_ANALYSIS = 80
DETECTION_PRESETS = {
    "Fast Bowling Mode": {
        "imgsz": 960,
        "confidence": 0.15,
    },
    "Balanced Mode": {
        "imgsz": 768,
        "confidence": 0.25,
    },
    "High Precision Mode": {
        "imgsz": 960,
        "confidence": 0.35,
    },
}


@st.cache_resource
def load_yolo_model(model_path_str):
    model_path = Path(model_path_str)

    if not model_path.exists():
        return None

    from ultralytics import YOLO

    return YOLO(str(model_path))


def get_model_options():
    return {
        "Ball + Stump Detector": {
            "path": CRICKET_OBJECTS_MODEL_PATH,
            "model_key": "current_best",
        },
        "Old Ball Detector": {
            "path": BALL_MODEL_PATH,
            "model_key": None,
        },
        "External Ball Model": {
            "path": EXTERNAL_BALL_MODEL_PATH,
            "model_key": None,
        },
        ENSEMBLE_MODEL_NAME: {
            "path": None,
            "model_key": None,
            "ensemble": True,
        },
    }


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
        elif (
            "stump" in normalized_name
            or "stumps" in normalized_name
            or "wicket" in normalized_name
        ):
            class_names[int(class_id)] = "stump"

    return class_names


def load_detection_model(model_key=None, model_path=None):
    """Load a YOLO detection model from a registry key or legacy path."""
    if model_key:
        return get_cached_yolo_model(model_key)
    if model_path is None:
        return None
    return load_yolo_model(str(model_path))


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    if intersection_area <= 0:
        return 0

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0

    return intersection_area / union_area


def non_max_suppression_custom(detections, iou_threshold=0.4):
    kept_detections = []
    sorted_detections = sorted(
        detections,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    while sorted_detections:
        best_detection = sorted_detections.pop(0)
        kept_detections.append(best_detection)
        sorted_detections = [
            detection
            for detection in sorted_detections
            if calculate_iou(best_detection["box"], detection["box"]) < iou_threshold
        ]

    return kept_detections


def run_single_model_detection(frame, model, confidence, imgsz):
    return model.predict(
        source=frame,
        conf=confidence,
        imgsz=imgsz,
        verbose=False,
    )[0]


def collect_model_detections(
    result,
    class_names,
    confidence,
    model_name,
    get_box_center,
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

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": detection_confidence,
            "box": (x1, y1, x2, y2),
            "center": get_box_center(x1, y1, x2, y2),
            "model_name": model_name,
        }

        if class_name == "ball":
            if detection_confidence >= (ball_confidence or confidence):
                ball_detections.append(detection)
        elif class_name == "stump":
            if detection_confidence >= confidence:
                stump_detections.append(detection)

    return ball_detections, stump_detections


def collect_low_confidence_ball_detections(result, class_names, model_name, get_box_center):
    low_confidence_detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return low_confidence_detections

    for box in result.boxes:
        class_id = int(box.cls[0].cpu().numpy())

        if class_names.get(class_id) != "ball":
            continue

        confidence = float(box.conf[0].cpu().numpy())

        if not 0.10 <= confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        low_confidence_detections.append(
            {
                "class_id": class_id,
                "class_name": "ball",
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "center": get_box_center(x1, y1, x2, y2),
                "model_name": model_name,
            }
        )

    return low_confidence_detections


def offset_detection_coordinates(detections, offset_x, offset_y):
    adjusted_detections = []

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        adjusted_detection = {
            **detection,
            "box": (
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y,
            ),
            "center": (center_x + offset_x, center_y + offset_y),
        }
        adjusted_detections.append(adjusted_detection)

    return adjusted_detections


def estimate_pitch_roi(frame_shape, stump_detections, previous_roi=None):
    height, width = frame_shape[:2]

    if not stump_detections:
        return previous_roi

    main_stump = max(
        stump_detections,
        key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]),
    )
    x1, y1, x2, y2 = main_stump["box"]
    stump_center_x = int((x1 + x2) / 2)
    stump_width = max(x2 - x1, 1)
    stump_height = max(y2 - y1, 1)

    corridor_width = int(
        min(
            width * 0.90,
            max(stump_width * 10, stump_height * 4, width * 0.38),
        )
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


def crop_frame_to_roi(frame, roi_box):
    if roi_box is None:
        return frame, (0, 0), None

    x1, y1, x2, y2 = roi_box

    if x2 <= x1 or y2 <= y1:
        return frame, (0, 0), None

    return frame[y1:y2, x1:x2], (x1, y1), roi_box


def create_local_search_roi(frame_shape, center, missing_frames=1):
    if center is None:
        return None

    height, width = frame_shape[:2]
    center_x, center_y = center
    window_size = int(max(96, min(width, height) * 0.18) * (1 + min(missing_frames, 6) * 0.12))
    half_window = window_size // 2

    x1 = max(0, int(center_x - half_window))
    y1 = max(0, int(center_y - half_window))
    x2 = min(width, int(center_x + half_window))
    y2 = min(height, int(center_y + half_window))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def draw_pitch_roi(frame, roi_box):
    if roi_box is None:
        return

    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
    draw_label(frame, "Pitch ROI", x1, y1, (40, 160, 40))


def draw_search_roi(frame, roi_box):
    if roi_box is None:
        return

    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 220), 2)
    draw_label(frame, "Recovery ROI", x1, y1, (150, 40, 130))


def estimate_auto_pitch_corners(frame_shape, stump_detections):
    if not stump_detections:
        return None

    height, width = frame_shape[:2]
    main_stump = max(
        stump_detections,
        key=lambda item: (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1]),
    )
    x1, y1, x2, y2 = main_stump["box"]
    stump_center_x = int((x1 + x2) / 2)
    stump_width = max(x2 - x1, 1)

    top_half_width = max(stump_width * 2.2, width * 0.10)
    bottom_half_width = max(stump_width * 6.0, width * 0.30)
    top_y = max(0, int(y2 + height * 0.02))
    bottom_y = height - 1

    return [
        (max(0, int(stump_center_x - top_half_width)), top_y),
        (min(width - 1, int(stump_center_x + top_half_width)), top_y),
        (max(0, int(stump_center_x - bottom_half_width)), bottom_y),
        (min(width - 1, int(stump_center_x + bottom_half_width)), bottom_y),
    ]


def compute_pitch_homography(pitch_points):
    if pitch_points is None or len(pitch_points) != 4:
        return None

    source_points = np.array(pitch_points, dtype="float32")
    target_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype="float32",
    )

    return cv2.getPerspectiveTransform(source_points, target_points)


def transform_point_to_pitch(point, homography):
    if point is None or homography is None:
        return None

    points = np.array([[[float(point[0]), float(point[1])]]], dtype="float32")
    transformed = cv2.perspectiveTransform(points, homography)[0][0]
    x = min(max(float(transformed[0]), 0.0), 1.0)
    y = min(max(float(transformed[1]), 0.0), 1.0)
    return x, y


def estimate_line_from_pitch_x(pitch_x, batter_handedness="right"):
    if pitch_x is None:
        return "Unknown"
    left_label = "Leg side" if normalize_handedness(batter_handedness) == "left" else "Off side"
    right_label = "Off side" if normalize_handedness(batter_handedness) == "left" else "Leg side"
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


def load_ensemble_models():
    models = []

    for config in get_ensemble_model_configs():
        model = load_detection_model(
            model_key=config.get("model_key"),
            model_path=config.get("path"),
        )

        if model is None:
            continue

        models.append(
            {
                **config,
                "model": model,
                "class_names": map_model_classes(model),
            }
        )

    return models


def resolve_ensemble_models(model_paths=None):
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
        if model is None:
            continue

        models.append(
            {
                **config,
                "model": model,
                "class_names": map_model_classes(model),
            }
        )

    return models


def run_ensemble_detection(frame, model_paths, confidence, imgsz):
    models = model_paths

    if models is None or not models or not isinstance(models[0], dict):
        models = resolve_ensemble_models(models)

    combined_ball_detections = []
    combined_stump_detections = []
    low_confidence_ball_detections = []

    for model_config in models:
        result = run_single_model_detection(
            frame,
            model_config["model"],
            min(confidence, 0.10),
            imgsz,
        )
        ball_detections, stump_detections = collect_model_detections(
            result,
            model_config["class_names"],
            confidence,
            model_config["name"],
            get_box_center,
        )
        low_confidence_ball_detections.extend(
            collect_low_confidence_ball_detections(
                result,
                model_config["class_names"],
                model_config["name"],
                get_box_center,
            )
        )

        if model_config["use_ball"]:
            combined_ball_detections.extend(ball_detections)

        if model_config["use_stump"]:
            combined_stump_detections.extend(stump_detections)

    return {
        "ball_detections": non_max_suppression_custom(combined_ball_detections),
        "stump_detections": combined_stump_detections,
        "low_confidence_ball_detections": non_max_suppression_custom(
            low_confidence_ball_detections
        ),
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
):
    speed_settings = speed_settings or {}
    same_model = (
        not use_ensemble
        and ball_model is not None
        and stump_model is not None
        and ball_model is stump_model
    )
    if same_model and speed_settings.get("single_pass_same_model", False):
        full_frame_start = time.perf_counter()
        combined_result = run_single_model_detection(
            frame,
            stump_model,
            min(confidence, 0.10),
            imgsz,
        )
        ball_detections, stump_detections = collect_model_detections(
            combined_result,
            stump_class_names,
            confidence,
            "Ball + Stump Detector",
            get_box_center,
            ball_confidence=ball_confidence,
        )
        low_confidence_ball_detections = collect_low_confidence_ball_detections(
            combined_result,
            stump_class_names,
            "Ball + Stump Detector",
            get_box_center,
        )
        full_frame_time_ms = (time.perf_counter() - full_frame_start) * 1000
        roi_box = estimate_pitch_roi(frame.shape, stump_detections, previous_roi)
        return {
            "ball_detections": non_max_suppression_custom(ball_detections),
            "stump_detections": stump_detections,
            "low_confidence_ball_detections": non_max_suppression_custom(
                low_confidence_ball_detections
            ),
            "roi_box": roi_box,
            "full_frame_time_ms": full_frame_time_ms,
            "roi_time_ms": 0,
            "used_roi": False,
            "single_pass": True,
        }

    full_frame_start = time.perf_counter()
    stump_result = run_single_model_detection(
        frame,
        stump_model,
        confidence,
        imgsz,
    )
    _, stump_detections = collect_model_detections(
        stump_result,
        stump_class_names,
        confidence,
        "Ball + Stump Detector",
        get_box_center,
    )
    full_frame_time_ms = (time.perf_counter() - full_frame_start) * 1000

    roi_box = estimate_pitch_roi(frame.shape, stump_detections, previous_roi)
    roi_frame, (offset_x, offset_y), active_roi_box = crop_frame_to_roi(frame, roi_box)
    used_roi = active_roi_box is not None

    roi_start = time.perf_counter()

    if use_ensemble:
        detection_result = run_ensemble_detection(
            roi_frame,
            ensemble_models,
            confidence,
            imgsz,
        )
        ball_detections = detection_result["ball_detections"]
        low_confidence_ball_detections = detection_result.get(
            "low_confidence_ball_detections",
            [],
        )
    else:
        ball_result = run_single_model_detection(
            roi_frame,
            ball_model,
            min(confidence, 0.10),
            imgsz,
        )
        ball_detections, _ = collect_model_detections(
            ball_result,
            ball_class_names,
            confidence,
            "Selected Model",
            get_box_center,
            ball_confidence=ball_confidence,
        )
        low_confidence_ball_detections = collect_low_confidence_ball_detections(
            ball_result,
            ball_class_names,
            "Selected Model",
            get_box_center,
        )

    detection_time_ms = (time.perf_counter() - roi_start) * 1000

    if used_roi:
        ball_detections = offset_detection_coordinates(
            ball_detections,
            offset_x,
            offset_y,
        )
        low_confidence_ball_detections = offset_detection_coordinates(
            low_confidence_ball_detections,
            offset_x,
            offset_y,
        )
        roi_time_ms = detection_time_ms
    else:
        full_frame_time_ms += detection_time_ms
        roi_time_ms = 0

    return {
        "ball_detections": non_max_suppression_custom(ball_detections),
        "stump_detections": stump_detections,
        "low_confidence_ball_detections": non_max_suppression_custom(
            low_confidence_ball_detections
        ),
        "roi_box": active_roi_box,
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
    search_roi = create_local_search_roi(frame.shape, search_center, missing_frames)

    if search_roi is None:
        return {
            "ball_detections": [],
            "search_roi": None,
            "recovered": False,
        }

    search_frame, (offset_x, offset_y), active_search_roi = crop_frame_to_roi(frame, search_roi)
    recovery_confidence = max(0.08, min(confidence, 0.18))

    if use_ensemble:
        detection_result = run_ensemble_detection(
            search_frame,
            ensemble_models,
            recovery_confidence,
            max(imgsz, 960),
        )
        ball_detections = detection_result["ball_detections"]
    else:
        result = run_single_model_detection(
            search_frame,
            ball_model,
            recovery_confidence,
            max(imgsz, 960),
        )
        ball_detections, _ = collect_model_detections(
            result,
            ball_class_names,
            recovery_confidence,
            "Local Re-detection",
            get_box_center,
            ball_confidence=recovery_confidence,
        )

    ball_detections = offset_detection_coordinates(ball_detections, offset_x, offset_y)

    return {
        "ball_detections": non_max_suppression_custom(ball_detections),
        "search_roi": active_search_roi,
        "recovered": bool(ball_detections),
    }


def write_review_metadata(row):
    REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "source",
        "frame_index",
        "reason",
        "file_name",
        "model_name",
        "confidence",
        "bbox",
        "note",
    ]
    write_header = not REVIEW_FRAMES_CSV.exists()

    with open(REVIEW_FRAMES_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_field_zone_correction(row):
    FIELD_ZONE_CORRECTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "estimated_zone",
        "corrected_zone",
        "shot_angle",
        "confidence",
        "mode",
        "source",
    ]
    write_header = not FIELD_ZONE_CORRECTIONS_CSV.exists()

    with open(FIELD_ZONE_CORRECTIONS_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def save_review_frame(
    frame,
    timestamp,
    frame_index,
    reason,
    detections=None,
    source="video_analysis",
    note="",
):
    REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    safe_reason = reason.replace(" ", "_").lower()
    file_name = f"{safe_reason}_{timestamp}_{frame_index:04d}.jpg"
    output_path = REVIEW_FRAMES_DIR / file_name
    review_frame = frame.copy()

    for detection in detections or []:
        x1, y1, x2, y2 = detection["box"]
        center_x, center_y = detection["center"]
        confidence = detection.get("confidence", 0)
        cv2.rectangle(review_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.circle(review_frame, (center_x, center_y), 4, (0, 165, 255), -1)
        draw_label(review_frame, f"{reason} {confidence:.2f}", x1, y1, (0, 120, 220))

    cv2.imwrite(str(output_path), review_frame)

    metadata_detections = detections or [None]
    for detection in metadata_detections:
        write_review_metadata(
            {
                "timestamp": timestamp,
                "source": source,
                "frame_index": frame_index,
                "reason": reason,
                "file_name": file_name,
                "model_name": "" if detection is None else detection.get("model_name", ""),
                "confidence": "" if detection is None else f"{detection.get('confidence', 0):.4f}",
                "bbox": "" if detection is None else ",".join(map(str, detection.get("box", ""))),
                "note": note,
            }
        )

    return output_path


def create_review_frames_zip():
    REVIEW_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = REVIEW_FRAMES_DIR / f"cricvision_review_frames_{timestamp}.zip"
    files_to_zip = [
        path
        for path in REVIEW_FRAMES_DIR.iterdir()
        if path.is_file()
        and path.name != zip_path.name
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".csv"}
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for path in files_to_zip:
            zip_file.write(path, arcname=path.name)

    return zip_path, len(files_to_zip)


def draw_label(frame, text, x, y, color):
    y = max(y, 25)
    label_width = len(text) * 10 + 10

    cv2.rectangle(
        frame,
        (x, y - 24),
        (x + label_width, y),
        color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def get_box_center(x1, y1, x2, y2):
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


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


def estimate_bounce_point(trajectory_points, min_points=6):
    if len([point for point in trajectory_points if point is not None]) < min_points:
        return None

    bounce_result = detect_bounce_by_direction_change(trajectory_points)

    if bounce_result is None:
        return None

    return bounce_result["point"]


def convert_to_browser_mp4(input_path, output_path):
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    return output_path


def extract_first_video_frame(uploaded_video):
    uploaded_video.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_video.read())
        temp_path = Path(temp_input.name)

    cap = cv2.VideoCapture(str(temp_path))
    success, frame = cap.read()
    cap.release()
    uploaded_video.seek(0)

    if not success:
        return None

    return frame


def _draw_ball_detections(frame, ball_detections):
    for detection in ball_detections or []:
        x1, y1, x2, y2 = detection["bbox"]
        center_x, center_y = detection["center"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
        draw_label(frame, f"ball {detection['confidence']:.2f}", x1, y1, (0, 180, 180))


def _add_impact_marker_to_video(video_path, impact_info):
    """Rewrite an analyzed video once so its retrospectively chosen impact frame is marked."""
    from Backends.src.analysis.impact_detection import draw_impact_marker

    if not impact_info or impact_info.get("impact_frame") is None:
        return Path(video_path)

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return video_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_path = video_path.with_name(f"{video_path.stem}_impact{video_path.suffix}")
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        return video_path

    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        draw_impact_marker(frame, impact_info, frame_index)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    temp_path.replace(video_path)
    return video_path


def _persist_result_to_session(result, source_type, video_name=None):
    """Save analysis result to local session storage without crashing the UI."""
    try:
        from Backends.src.storage.session_store import persist_analysis_to_session

        saved = persist_analysis_to_session(result, source_type, video_name=video_name)
        result["session_saved"] = True
        result["session_result_id"] = saved.get("id")
        result["session_save_error"] = None
    except Exception as error:
        result["session_saved"] = False
        result["session_save_error"] = str(error)
    return result


def save_batting_report(result, analysis_mode):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = REPORTS_DIR / f"batting_analysis_{timestamp}.json"
    impact = result.get("impact_info", {})
    report = {
        "analysis_mode": analysis_mode,
        "ball_detected": bool(result.get("ball_detected_frames", 0)),
        "bat_detected": bool(result.get("bat_detected_frames", 0)),
        "impact_detected": impact.get("impact_detected", False),
        "possible_impact_frame": impact.get("impact_frame"),
        "impact_time_sec": impact.get("impact_time_sec"),
        "min_ball_bat_distance_px": impact.get("min_ball_bat_distance_px"),
        "impact_confidence": impact.get("impact_confidence", "Unknown"),
        "impact_reason": impact.get("reason", impact.get("impact_reason", "")),
        "impact_frame_image_path": str(impact.get("impact_frame_image_path") or ""),
        "shot_type": result.get("shot_info", {}).get("shot_type", "Unknown"),
        "shot_confidence": result.get("shot_info", {}).get("shot_confidence", "Unknown"),
        "shot_direction": result.get("shot_info", {}).get("shot_direction", "Unknown"),
        "shot_height": result.get("shot_info", {}).get("shot_height", "Unknown"),
        "shot_reason": result.get("shot_info", {}).get("reason", result.get("shot_info", {}).get("shot_reason", "")),
        "predicted_outcome": result.get("outcome_info", {}).get("predicted_outcome", "Unknown"),
        "outcome_confidence": result.get("outcome_info", {}).get("outcome_confidence", "Unknown"),
        "run_estimate": result.get("outcome_info", {}).get("run_estimate"),
        "dismissal_risk": result.get("outcome_info", {}).get("dismissal_risk", "Unknown"),
        "boundary_chance": result.get("outcome_info", {}).get("boundary_chance", "Unknown"),
        "outcome_reason": result.get("outcome_info", {}).get(
            "reason",
            result.get("outcome_info", {}).get("outcome_reason", ""),
        ),
        "field_zone": result.get("field_zone", "Unknown"),
        "zone_confidence": result.get("zone_confidence", "Unknown"),
        "direction_angle_degrees": result.get("direction_angle_degrees"),
        "direction_reason": result.get("direction_reason", ""),
        "movement_dx": result.get("movement_dx"),
        "movement_dy": result.get("movement_dy"),
        "direction_shot_category": result.get("direction_shot_category", "Unknown"),
        "agent_quality": result.get("agent_quality", "Unknown"),
        "agent_confidence": result.get("agent_confidence", "Unknown"),
        "ball_tracking_coverage": result.get("ball_tracking_coverage"),
        "bat_detection_coverage": result.get("bat_detection_coverage"),
        "stump_detection_coverage": result.get("stump_detection_coverage"),
        "missing_ball_frames": result.get("missing_ball_frames", 0),
        "possible_false_ball_detections": result.get("possible_false_ball_detections", 0),
        "analysis_consistency": result.get("analysis_consistency", "Unknown"),
        "review_flags": list(result.get("review_flags") or []),
        "agent_notes": result.get("agent_notes", ""),
        "minimum_ball_bat_distance": impact.get(
            "min_distance",
            impact.get("min_ball_bat_distance_px"),
        ),
        "ball_model_used": result.get("ball_model_used", "Unknown"),
        "bat_model_used": result.get("bat_model_used", "Unknown"),
        "processed_video_path": str(result.get("output_path", "")),
    }
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    result["report_path"] = report_path
    result["batting_report"] = report
    return report_path


def _run_post_shot_pipeline(
    impact_frame_detections,
    impact_info,
    shot_info,
    batter_handedness,
    fps,
    delivery_report=None,
):
    from Backends.src.analysis.delivery_enrichment import run_post_shot_pipeline

    return run_post_shot_pipeline(
        impact_frame_detections,
        impact_info,
        shot_info,
        batter_handedness,
        fps,
        delivery_report=delivery_report,
    )


def process_batting_video(
    video_path,
    output_path,
    ball_model_key="current_best",
    bat_model_key="cricshot_bat",
    confidence=0.25,
    speed_mode="Balanced",
    max_frames=None,
):
    """Process a clip with only the models needed for batting intelligence."""
    from Backends.src.agents.observer_timeline import build_observer_timeline
    from Backends.src.analysis.analysis_speed import (
        get_analysis_mode_settings,
        resize_frame_for_inference,
        scale_detections_to_original,
    )
    from Backends.src.analysis.bat_detection import (
        detect_ball_in_frame,
        detect_bat_in_frame,
        draw_bat_detections,
    )
    from Backends.src.analysis.impact_detection import (
        detect_bat_ball_impact,
        save_impact_frame_preview,
    )
    from Backends.src.analysis.shot_classification import classify_shot_type

    ball_model = get_cached_yolo_model(ball_model_key)
    bat_model = get_cached_yolo_model(bat_model_key)
    bat_unavailable_reason = ""
    if ball_model is None:
        return {"success": False, "error": "The selected ball model is unavailable."}
    if bat_model is None:
        bat_unavailable_reason = "Impact not detected: bat detection unavailable."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"success": False, "error": "Could not open uploaded video."}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    speed_settings = get_analysis_mode_settings(speed_mode)
    frame_stride = max(1, int(speed_settings.get("frame_stride", 1)))
    resize_width = speed_settings.get("resize_width")
    performance = _empty_performance_profile()
    performance["speed_mode"] = speed_mode
    analysis_started = time.perf_counter()
    processed_detection_frames = 0
    if width <= 0 or height <= 0:
        cap.release()
        return {"success": False, "error": "Could not read video width/height."}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        return {"success": False, "error": "Could not create output video writer."}

    frame_index = 0
    ball_detected_frames = 0
    bat_detected_frames = 0
    ball_tracks = []
    bat_detections_by_frame = {}
    impact_frame_detections = []
    frame_detections = impact_frame_detections
    impact_frame_candidates = {}
    trajectory = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    while True:
        read_started = time.perf_counter()
        success, frame = cap.read()
        if not success:
            break
        if max_frames is not None and frame_index >= max_frames:
            break
        performance["video_read_time_sec"] += time.perf_counter() - read_started
        performance["frames_read"] += 1

        annotated_frame = frame.copy()
        ball_detections = []
        bat_detections = []
        if frame_index % frame_stride == 0:
            inference_started = time.perf_counter()
            inference_frame, detection_scale = resize_frame_for_inference(frame, resize_width)
            ball_detections = detect_ball_in_frame(inference_frame, ball_model, confidence)
            ball_detections = scale_detections_to_original(ball_detections, detection_scale)
            bat_detections = (
                detect_bat_in_frame(inference_frame, bat_model, confidence)
                if bat_model
                else []
            )
            bat_detections = scale_detections_to_original(bat_detections, detection_scale)
            performance["model_inference_time_sec"] += time.perf_counter() - inference_started
            processed_detection_frames += 1
        if ball_detections:
            ball_detected_frames += 1
            main_ball = max(ball_detections, key=lambda item: item["confidence"])
            ball_tracks.append(main_ball["center"])
            trajectory.append(tuple(main_ball["center"]))
        else:
            ball_tracks.append(None)
        if bat_detections:
            bat_detected_frames += 1
            bat_detections_by_frame[frame_index] = bat_detections
        impact_frame_detections.append(
            {
                "frame_index": frame_index,
                "ball_detections": ball_detections,
                "bat_detections": bat_detections,
                "stump_detections": [],
            }
        )
        if ball_detections and bat_detections:
            impact_frame_candidates[frame_index] = frame.copy()

        _draw_ball_detections(annotated_frame, ball_detections)
        draw_bat_detections(annotated_frame, bat_detections)
        for index in range(1, len(trajectory[-35:])):
            recent = trajectory[-35:]
            cv2.line(annotated_frame, recent[index - 1], recent[index], (0, 255, 255), 3)
        annotation_started = time.perf_counter()
        writer.write(annotated_frame)
        performance["annotation_write_time_sec"] += time.perf_counter() - annotation_started
        frame_index += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_index / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_index}/{total_frames}")

    cap.release()
    writer.release()
    progress_bar.empty()
    status_text.empty()
    if frame_index == 0:
        return {"success": False, "error": "No video frames were processed."}

    impact_info = detect_bat_ball_impact(impact_frame_detections, fps=fps)
    if bat_unavailable_reason:
        impact_info["reason"] = bat_unavailable_reason
        impact_info["impact_reason"] = bat_unavailable_reason
    impact_frame = impact_info.get("impact_frame")
    if impact_frame is not None:
        preview_path = save_impact_frame_preview(
            impact_frame_candidates.get(impact_frame),
            impact_info,
            prefix=f"batting_impact_{Path(output_path).stem}",
        )
        if preview_path is not None:
            impact_info["impact_frame_image_path"] = str(preview_path)
    _add_impact_marker_to_video(output_path, impact_info)
    shot_info = classify_shot_type(
        frame_detections,
        impact_info,
        batter_handedness=None,
        fps=fps,
    )
    report_started = time.perf_counter()
    direction_info, outcome_info, agent_info, enrichment = _run_post_shot_pipeline(
        frame_detections,
        impact_info,
        shot_info,
        batter_handedness=None,
        fps=fps,
    )
    observer_timeline = build_observer_timeline(frame_detections, total_frames=frame_index, fps=fps)
    performance["report_generation_time_sec"] = time.perf_counter() - report_started
    performance["frames_processed"] = processed_detection_frames
    performance["total_analysis_time_sec"] = time.perf_counter() - analysis_started
    if processed_detection_frames > 0:
        performance["average_ms_per_processed_frame"] = round(
            (performance["model_inference_time_sec"] / processed_detection_frames) * 1000,
            2,
        )
    ball_info = get_model_info(ball_model_key) or {}
    bat_info = get_model_info(bat_model_key) or {}
    return {
        "success": True,
        "analysis_mode": "Batting Analysis",
        "output_path": Path(output_path),
        "total_frames": frame_index,
        "ball_detected_frames": ball_detected_frames,
        "bat_detected_frames": bat_detected_frames,
        "ball_detection_rate": (ball_detected_frames / frame_index) * 100,
        "bat_detection_rate": (bat_detected_frames / frame_index) * 100,
        "impact_info": impact_info,
        "frame_detections": frame_detections,
        "impact_frame_detections": frame_detections,
        "observer_timeline": observer_timeline,
        "performance_profile": performance,
        "speed_mode": speed_mode,
        "shot_info": shot_info,
        "direction_info": direction_info,
        "agent_info": agent_info,
        "shot_type": shot_info.get("shot_type", "Unknown"),
        "shot_confidence": shot_info.get("shot_confidence", "Unknown"),
        "shot_direction": shot_info.get("shot_direction", "Unknown"),
        "shot_height": shot_info.get("shot_height", "Unknown"),
        "shot_reason": shot_info.get("reason", shot_info.get("shot_reason", "")),
        "outcome_info": outcome_info,
        "predicted_outcome": outcome_info.get("predicted_outcome", "Unknown"),
        "outcome_confidence": outcome_info.get("outcome_confidence", "Unknown"),
        "run_estimate": outcome_info.get("run_estimate"),
        "dismissal_risk": outcome_info.get("dismissal_risk", "Unknown"),
        "boundary_chance": outcome_info.get("boundary_chance", "Unknown"),
        "outcome_reason": outcome_info.get("reason", outcome_info.get("outcome_reason", "")),
        "ball_model_used": ball_info.get("name", ball_model_key),
        "bat_model_used": bat_info.get("name", bat_model_key) if bat_model else "Unavailable",
        **enrichment,
    }


def get_default_pitch_points(frame):
    height, width = frame.shape[:2]
    return {
        "top_left": (int(width * 0.40), int(height * 0.28)),
        "top_right": (int(width * 0.60), int(height * 0.28)),
        "bottom_left": (int(width * 0.22), int(height * 0.95)),
        "bottom_right": (int(width * 0.78), int(height * 0.95)),
    }


def show_manual_pitch_point_inputs(frame):
    defaults = get_default_pitch_points(frame)
    height, width = frame.shape[:2]
    preview = frame.copy()

    for label, point in defaults.items():
        cv2.circle(preview, point, 7, (0, 255, 0), -1)
        draw_label(preview, label.replace("_", " "), point[0], point[1], (40, 160, 40))

    st.image(
        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
        caption="First frame with default pitch-point guide. Adjust the coordinates below.",
        use_container_width=True,
    )

    points = []
    labels = [
        ("Top-left pitch corner", "top_left"),
        ("Top-right pitch corner", "top_right"),
        ("Bottom-left pitch corner", "bottom_left"),
        ("Bottom-right pitch corner", "bottom_right"),
    ]

    for label, key in labels:
        default_x, default_y = defaults[key]
        col_x, col_y = st.columns(2)

        with col_x:
            point_x = st.number_input(
                f"{label} X",
                min_value=0,
                max_value=max(width - 1, 0),
                value=default_x,
                step=1,
                key=f"manual_pitch_{key}_x",
            )

        with col_y:
            point_y = st.number_input(
                f"{label} Y",
                min_value=0,
                max_value=max(height - 1, 0),
                value=default_y,
                step=1,
                key=f"manual_pitch_{key}_y",
            )

        points.append((int(point_x), int(point_y)))

    return points


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


def ensure_delivery_report_fields(result):
    result.setdefault("ball_tracking_rate", result.get("ball_detection_rate", 0))
    result.setdefault("interpolated_ball_frames", 0)
    result.setdefault("estimated_line", "Unknown")
    result.setdefault("estimated_length", "Unknown")
    result.setdefault("estimated_bounce_point", None)
    result.setdefault("average_ball_confidence", 0)
    result.setdefault("kalman_predicted_frames", 0)
    result.setdefault("tracker_recoveries", 0)
    result.setdefault("overall_tracking_quality", "Poor")
    result.setdefault("pitch_normalized_bounce_point", None)
    result.setdefault("calibration_status", "Not calibrated")
    result.setdefault("calibration_source", "None")
    result.setdefault("calibration_warning", "Confidence warning: pitch calibration is missing.")
    result.setdefault("wagon_wheel", {})


def show_cricket_delivery_report(result):
    ensure_delivery_report_fields(result)

    quality = calculate_detection_quality(result)
    report = generate_delivery_report(result)
    feedback_items = generate_coaching_feedback(result)
    warnings = detect_analysis_warnings(result)

    st.subheader("🏏 Delivery Report")
    st.metric(
        "Analysis Quality",
        f"{quality['quality_score']}/100",
        quality["quality_label"],
    )
    st.info(report)

    st.markdown("**Coaching Feedback**")
    for feedback_item in feedback_items:
        st.markdown(f"- {feedback_item}")

    if warnings:
        st.warning("Warnings:\n" + "\n".join(f"- {warning}" for warning in warnings))


DEBUG_PERFORMANCE = False


def _empty_performance_profile():
    return {
        "video_read_time_sec": 0.0,
        "model_inference_time_sec": 0.0,
        "annotation_write_time_sec": 0.0,
        "report_generation_time_sec": 0.0,
        "total_analysis_time_sec": 0.0,
        "frames_processed": 0,
        "frames_read": 0,
        "average_ms_per_processed_frame": None,
        "speed_mode": "Balanced",
    }


def process_video(
    video_path,
    output_path,
    model_path,
    model_key=None,
    class_names=None,
    confidence=0.25,
    imgsz=640,
    use_ensemble=False,
    show_pitch_roi=False,
    calibration_mode="Auto calibration using detected stumps",
    manual_pitch_points=None,
    shot_trajectory_mode="Use last part of trajectory",
    manual_contact_frame=None,
    field_setup=None,
    bat_model_key=None,
    speed_mode="Balanced",
    max_frames=None,
):
    from Backends.src.agents.observer_timeline import build_observer_timeline
    from Backends.src.analysis.analysis_speed import (
        get_analysis_mode_settings,
        resize_frame_for_inference,
        scale_detections_to_original,
    )
    from Backends.src.analysis.bat_detection import (
        detect_bat_in_frame,
        draw_bat_detections,
    )
    from Backends.src.analysis.impact_detection import (
        detect_bat_ball_impact,
        save_impact_frame_preview,
    )
    from Backends.src.analysis.shot_classification import classify_shot_type

    model = None
    ensemble_models = []
    bat_model = get_cached_yolo_model(bat_model_key) if bat_model_key else None
    bat_unavailable_reason = ""

    if bat_model_key and bat_model is None:
        bat_unavailable_reason = "Impact not detected: bat detection unavailable."
    stump_model = get_cached_yolo_model("current_best")

    if stump_model is None:
        return {
            "success": False,
            "error": "Current Best Ball + Stump Model is unavailable.",
        }

    stump_class_names = map_model_classes(stump_model)

    if use_ensemble:
        ensemble_models = load_ensemble_models()

        if not ensemble_models:
            return {
                "success": False,
                "error": "No ensemble models were found. Add at least one configured model file.",
            }
    else:
        model = load_detection_model(model_key=model_key, model_path=model_path)

        if model is None:
            model_label = (get_model_info(model_key) or {}).get("name") if model_key else None
            return {
                "success": False,
                "error": f"Model not found: {model_label or model_path}",
            }

        class_names = map_model_classes(model)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "success": False,
            "error": "Could not open uploaded video.",
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Could not read video width/height.",
        }

    speed_settings = get_analysis_mode_settings(speed_mode)
    frame_stride = max(1, int(speed_settings.get("frame_stride", 1)))
    inference_imgsz = int(speed_settings.get("yolo_imgsz", imgsz))
    resize_width = speed_settings.get("resize_width")
    performance = _empty_performance_profile()
    performance["speed_mode"] = speed_mode
    analysis_started = time.perf_counter()
    processed_detection_frames = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        return {
            "success": False,
            "error": "Could not create output video writer.",
        }

    frame_index = 0

    ball_detected_frames = 0
    stump_detected_frames = 0

    total_ball_detections = 0
    total_stump_detections = 0

    confidence_values = []
    low_confidence_ball_frames = 0
    review_frame_count = 0
    full_frame_detection_time_total = 0
    roi_detection_time_total = 0
    roi_detected_frames = 0
    tracker_recoveries = 0
    kalman_predicted_frames = 0
    last_roi_size = "Full frame"

    trajectory_points = []
    ball_positions = []
    bat_detections_by_frame = {}
    bat_detected_frames = 0
    impact_frame_detections = []
    frame_detections = impact_frame_detections
    impact_frame_candidates = {}
    stump_detections_by_frame = []
    last_raw_frame = None
    previous_roi_box = None
    max_trajectory_points = 35
    pitch_homography = None
    calibration_status = "Not calibrated"
    calibration_source = "None"
    calibration_warning = "Confidence warning: pitch calibration is missing; using image-space fallback."
    pitch_normalized_bounce_point = None

    if calibration_mode.startswith("Manual"):
        pitch_homography = compute_pitch_homography(manual_pitch_points)

        if pitch_homography is not None:
            calibration_status = "Calibrated"
            calibration_source = "Manual"
            calibration_warning = ""

    previous_ball_center = None
    kalman_tracker = BallKalmanTracker(max_missing_frames=10)

    missing_ball_frames = 0
    max_missing_ball_frames = 12

    estimated_bounce_point = None
    estimated_bounce_frame = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    estimated_bounce_point = None
    estimated_bounce_frame = None
    estimated_line = "Unknown"
    estimated_length = "Unknown"
    field_setup = field_setup or {}
    batter_handedness = normalize_handedness(field_setup.get("batter_handedness", "right"))
    bowler_arm = field_setup.get("bowler_arm", "Right-arm bowler")
    camera_view = field_setup.get("camera_view", "Behind bowler")
    fielders = field_setup.get("fielders", [])
    field_preset = field_setup.get("preset", "Custom")

    min_track_points_for_bounce = 8
    min_movement_distance = 40
    min_ball_confidence_for_tracking = 0.35 

    while True:
        read_started = time.perf_counter()
        success, frame = cap.read()

        if not success:
            break

        if max_frames is not None and frame_index >= max_frames:
            break

        performance["video_read_time_sec"] += time.perf_counter() - read_started
        performance["frames_read"] += 1

        last_raw_frame = frame.copy()
        annotated_frame = frame.copy()
        run_detection = frame_index % frame_stride == 0
        low_confidence_ball_detections = []
        ball_detections = []
        stump_detections = []
        bat_detections = []

        if run_detection:
            inference_started = time.perf_counter()
            inference_frame, detection_scale = resize_frame_for_inference(frame, resize_width)
            bat_detections = (
                detect_bat_in_frame(inference_frame, bat_model, confidence)
                if bat_model
                else []
            )
            bat_detections = scale_detections_to_original(bat_detections, detection_scale)

            detection_result = run_pitch_roi_detection(
                inference_frame,
                stump_model=stump_model,
                stump_class_names=stump_class_names,
                confidence=confidence,
                imgsz=inference_imgsz,
                previous_roi=previous_roi_box,
                ball_model=model,
                ball_class_names=class_names,
                ensemble_models=ensemble_models,
                use_ensemble=use_ensemble,
                ball_confidence=min_ball_confidence_for_tracking,
                speed_settings=speed_settings,
            )
            ball_detections = scale_detections_to_original(
                detection_result["ball_detections"],
                detection_scale,
            )
            stump_detections = scale_detections_to_original(
                detection_result["stump_detections"],
                detection_scale,
            )
            low_confidence_ball_detections = scale_detections_to_original(
                detection_result.get("low_confidence_ball_detections", []),
                detection_scale,
            )
            performance["model_inference_time_sec"] += time.perf_counter() - inference_started
            processed_detection_frames += 1
            full_frame_detection_time_total += detection_result["full_frame_time_ms"]
            roi_detection_time_total += detection_result["roi_time_ms"]

            if detection_result.get("used_roi"):
                previous_roi_box = detection_result["roi_box"]
                roi_detected_frames += 1
                roi_x1, roi_y1, roi_x2, roi_y2 = detection_result["roi_box"]
                last_roi_size = f"{roi_x2 - roi_x1}x{roi_y2 - roi_y1}"

            if show_pitch_roi:
                draw_pitch_roi(annotated_frame, detection_result.get("roi_box"))

            confidence_values.extend(item["confidence"] for item in ball_detections)

            if low_confidence_ball_detections:
                low_confidence_ball_frames += 1

                if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "low_confidence",
                        low_confidence_ball_detections,
                        source="video_analysis",
                    )
                    review_frame_count += 1

            if ball_detections:
                ball_detected_frames += 1
                total_ball_detections += len(ball_detections)
            elif speed_settings.get("enable_local_redetection", True):
                search_center = previous_ball_center or kalman_tracker.last_prediction
                recovery_result = run_local_redetection(
                    inference_frame,
                    search_center,
                    confidence,
                    inference_imgsz,
                    missing_ball_frames + 1,
                    ball_model=model,
                    ball_class_names=class_names,
                    ensemble_models=ensemble_models,
                    use_ensemble=use_ensemble,
                )

                if show_pitch_roi:
                    draw_search_roi(annotated_frame, recovery_result.get("search_roi"))

                if recovery_result["recovered"]:
                    ball_detections = scale_detections_to_original(
                        recovery_result["ball_detections"],
                        detection_scale,
                    )
                    tracker_recoveries += 1
                    ball_detected_frames += 1
                    total_ball_detections += len(ball_detections)
                    confidence_values.extend(item["confidence"] for item in ball_detections)
                elif review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "missed_ball",
                        source="video_analysis",
                        note="No ball detection passed the selected confidence threshold.",
                    )
                    review_frame_count += 1
            elif review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                save_review_frame(
                    frame,
                    timestamp,
                    frame_index,
                    "missed_ball",
                    source="video_analysis",
                    note="No ball detection passed the selected confidence threshold.",
                )
                review_frame_count += 1

            if stump_detections:
                stump_detected_frames += 1
                total_stump_detections += len(stump_detections)

                if calibration_mode.startswith("Auto") and pitch_homography is None:
                    auto_pitch_points = estimate_auto_pitch_corners(frame.shape, stump_detections)
                    pitch_homography = compute_pitch_homography(auto_pitch_points)

                    if pitch_homography is not None:
                        calibration_status = "Calibrated"
                        calibration_source = "Auto"
                        calibration_warning = ""

        if bat_detections:
            bat_detected_frames += 1
            bat_detections_by_frame[frame_index] = bat_detections
        draw_bat_detections(annotated_frame, bat_detections)

        stump_detections_by_frame.append(stump_detections)
        impact_frame_detections.append(
            {
                "frame_index": frame_index,
                "ball_detections": ball_detections,
                "bat_detections": bat_detections,
                "stump_detections": stump_detections,
            }
        )
        if ball_detections and bat_detections:
            impact_frame_candidates[frame_index] = frame.copy()

        for detection in ball_detections:
            x1, y1, x2, y2 = detection["box"]
            conf = detection["confidence"]
            center_x, center_y = detection["center"]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2,
            )

            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (0, 255, 255),
                -1,
            )

            draw_label(
                annotated_frame,
                f"ball {conf:.2f}",
                x1,
                y1,
                (0, 180, 180),
            )

        for detection in stump_detections:
            x1, y1, x2, y2 = detection["box"]
            conf = detection["confidence"]

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (255, 100, 0),
                2,
            )

            draw_label(
                annotated_frame,
                f"stump {conf:.2f}",
                x1,
                y1,
                (255, 100, 0),
            )

        main_ball = choose_main_ball(ball_detections, previous_ball_center)

        if main_ball is not None:
            missing_ball_frames = 0

            previous_ball_center = main_ball["center"]
            kalman_tracker.update(previous_ball_center)
            ball_positions.append(previous_ball_center)

            smoothed_positions = smooth_trajectory(ball_positions)
            display_trajectory_points = []

            for point in reversed(smoothed_positions):
                if point is None:
                    if display_trajectory_points:
                        break
                    continue
                display_trajectory_points.append(point)

            trajectory_points = list(reversed(display_trajectory_points))[-max_trajectory_points:]
            interpolated_positions = interpolate_missing_positions(ball_positions)
            usable_trajectory_points = [
                point for point in interpolated_positions if point is not None
            ]
            bounce_result = None

            if (
                len(usable_trajectory_points) >= min_track_points_for_bounce
                and has_enough_ball_movement(usable_trajectory_points, min_movement_distance)
            ):
                bounce_result = detect_bounce_by_direction_change(ball_positions)

            if bounce_result is not None and estimated_bounce_point is None:
                
                estimated_bounce_point = bounce_result["point"]
                
                estimated_bounce_frame = bounce_result["frame_index"]
                bounce_stump_detections = get_nearest_stump_detections(
                    stump_detections_by_frame,
                    estimated_bounce_frame,
                )

                estimated_line = estimate_line_from_stumps(
                    estimated_bounce_point,
                    bounce_stump_detections,
                    batter_handedness,
                )

                estimated_length = estimate_length_from_bounce(
                    estimated_bounce_point,
                    height 
                )

                pitch_normalized_bounce_point = transform_point_to_pitch(
                    estimated_bounce_point,
                    pitch_homography,
                )

                if pitch_normalized_bounce_point is not None:
                    pitch_x, pitch_y = pitch_normalized_bounce_point
                    estimated_line = estimate_line_from_pitch_x(pitch_x, batter_handedness)
                    estimated_length = estimate_length_from_pitch_y(pitch_y)


        else:
            missing_ball_frames += 1
            predicted_center = kalman_tracker.predict()

            if predicted_center is not None and missing_ball_frames <= 10:
                ball_positions.append(predicted_center)
                previous_ball_center = predicted_center
                kalman_predicted_frames += 1
            else:
                ball_positions.append(None)

            if missing_ball_frames >= max_missing_ball_frames:
                kalman_tracker.reset()
                if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
                    save_review_frame(
                        frame,
                        timestamp,
                        frame_index,
                        "poor_tracking",
                        source="video_analysis",
                        note=f"Ball missing for {missing_ball_frames} consecutive frames.",
                    )
                    review_frame_count += 1
                trajectory_points.clear()
                previous_ball_center = None
            else:
                smoothed_positions = smooth_trajectory(ball_positions)
                display_trajectory_points = []

                for point in reversed(smoothed_positions):
                    if point is None:
                        if display_trajectory_points:
                            break
                        continue
                    display_trajectory_points.append(point)

                trajectory_points = list(reversed(display_trajectory_points))[-max_trajectory_points:]

        for i in range(1, len(trajectory_points)):
            cv2.line(
                annotated_frame,
                trajectory_points[i - 1],
                trajectory_points[i],
                (0, 255, 255),
                3,
            )

        if estimated_bounce_point is not None:
            bx, by = estimated_bounce_point

            cv2.circle(
                annotated_frame,
                (bx, by),
                10,
                (0, 0, 255),
                -1,
            )

            cv2.circle(
                annotated_frame,
                (bx, by),
                16,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Bounce Frame: {estimated_bounce_frame}",
                (bx + 15, by - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            cv2.rectangle(
                annotated_frame,
                (15, 15),
                (470, 240),
                (0, 0, 0),
                -1,
            )
        
        

        bounce_text = "Not found"
        if estimated_bounce_frame is not None:
            bounce_text = f"Frame {estimated_bounce_frame}"

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_index}/{source_total_frames or frame_index}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Balls in frame: {len(ball_detections)}",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Stumps in frame: {len(stump_detections)}",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 160, 0),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Trajectory: {len(trajectory_points)} | Missing: {missing_ball_frames}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            annotated_frame,
            f"Bounce: {bounce_text}",
            (30, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
        )
        
        cv2.putText(
            annotated_frame,
            f"Line: {estimated_line}",
            (30, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        
        cv2.putText(
            annotated_frame,
            f"Length: {estimated_length}",
            (30, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )

        annotation_started = time.perf_counter()
        writer.write(annotated_frame)
        performance["annotation_write_time_sec"] += time.perf_counter() - annotation_started

        frame_index += 1

        if source_total_frames > 0:
            progress = min(frame_index / source_total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_index}/{source_total_frames}")
        else:
            status_text.text(f"Processing frame {frame_index}")

    cap.release()
    writer.release()

    progress_bar.empty()
    status_text.empty()

    if frame_index == 0:
        return {
            "success": False,
            "error": "No frames were processed. The uploaded video may be corrupted or unsupported.",
        }

    impact_info = detect_bat_ball_impact(impact_frame_detections, fps=fps)
    if bat_unavailable_reason:
        impact_info["reason"] = bat_unavailable_reason
        impact_info["impact_reason"] = bat_unavailable_reason
    impact_frame = impact_info.get("impact_frame")
    if impact_frame is not None:
        preview_path = save_impact_frame_preview(
            impact_frame_candidates.get(impact_frame),
            impact_info,
            prefix=f"video_impact_{Path(output_path).stem}",
        )
        if preview_path is not None:
            impact_info["impact_frame_image_path"] = str(preview_path)
        if not speed_settings.get("skip_impact_video_rewrite"):
            _add_impact_marker_to_video(output_path, impact_info)

    report_started = time.perf_counter()
    stump_detection_rate = 0

    if frame_index > 0:
        ball_detection_rate = (ball_detected_frames / frame_index) * 100
        stump_detection_rate = (stump_detected_frames / frame_index) * 100

    average_confidence = 0

    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)

    tracking_quality = calculate_tracking_quality(ball_positions, frame_index)
    overall_tracking_quality = get_tracking_quality_label(
        tracking_quality["tracking_rate"],
        tracking_quality["interpolated_frames"],
        kalman_predicted_frames,
    )
    shot_info = classify_shot_type(
        frame_detections,
        impact_info,
        batter_handedness=batter_handedness,
        fps=fps,
    )
    delivery_report = {
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "ball_detection_rate": ball_detection_rate,
        "overall_tracking_quality": overall_tracking_quality,
    }
    direction_info, outcome_info, agent_info, enrichment = _run_post_shot_pipeline(
        frame_detections,
        impact_info,
        shot_info,
        batter_handedness=batter_handedness,
        fps=fps,
        delivery_report=delivery_report,
    )
    observer_timeline = build_observer_timeline(
        frame_detections,
        total_frames=frame_index,
        fps=fps,
    )
    performance["report_generation_time_sec"] += time.perf_counter() - report_started
    performance["frames_processed"] = processed_detection_frames
    performance["total_analysis_time_sec"] = time.perf_counter() - analysis_started
    if processed_detection_frames > 0:
        performance["average_ms_per_processed_frame"] = round(
            (performance["model_inference_time_sec"] / processed_detection_frames) * 1000,
            2,
        )
    wagon_wheel = generate_wagon_wheel_data(
        ball_positions,
        batter_handedness=batter_handedness,
        mode=shot_trajectory_mode,
        manual_contact_frame=manual_contact_frame,
    )
    wagon_wheel["mode"] = shot_trajectory_mode
    nearest_fielder = find_nearest_fielder(
        wagon_wheel.get("shot_angle"), fielders, batter_handedness
    )
    wagon_wheel["nearest_fielder"] = nearest_fielder
    wagon_wheel["suggested_adjustment"] = suggest_field_adjustment(wagon_wheel, nearest_fielder)

    if field_setup:
        save_field_setup(field_setup)
        save_field_analysis_history(
            {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "source": "video_analysis",
                "batter_handedness": batter_handedness,
                "bowler_arm": bowler_arm,
                "camera_view": camera_view,
                "preset": field_preset,
                "simple_zone": wagon_wheel.get("simple_zone", "Unknown"),
                "detailed_zone": wagon_wheel.get("detailed_zone", "Unknown"),
                "shot_angle": "" if wagon_wheel.get("shot_angle") is None else f"{wagon_wheel['shot_angle']:.2f}",
                "nearest_fielder": "" if nearest_fielder is None else nearest_fielder.get("name", ""),
                "confidence": wagon_wheel.get("confidence", "Low"),
                "corrected_zone": "",
            }
        )
    average_full_frame_detection_time = 0
    average_roi_detection_time = 0

    if frame_index > 0:
        average_full_frame_detection_time = full_frame_detection_time_total / frame_index

    if roi_detected_frames > 0:
        average_roi_detection_time = roi_detection_time_total / roi_detected_frames

    if last_raw_frame is not None and estimated_bounce_point is None:
        if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
            save_review_frame(
                last_raw_frame,
                timestamp,
                max(frame_index - 1, 0),
                "bounce_unknown",
                source="video_analysis",
                note="Analysis finished without a bounce estimate.",
            )
            review_frame_count += 1

    if last_raw_frame is not None and (estimated_line == "Unknown" or estimated_length == "Unknown"):
        if review_frame_count < MAX_REVIEW_FRAMES_PER_ANALYSIS:
            save_review_frame(
                last_raw_frame,
                timestamp,
                max(frame_index - 1, 0),
                "line_length_unknown",
                source="video_analysis",
                note=f"Line={estimated_line}; Length={estimated_length}.",
            )
            review_frame_count += 1

    return {
        "success": True,
        "output_path": output_path,
        "total_frames": frame_index,
        "ball_detected_frames": ball_detected_frames,
        "stump_detected_frames": stump_detected_frames,
        "bat_detected_frames": bat_detected_frames,
        "total_ball_detections": total_ball_detections,
        "low_confidence_ball_frames": low_confidence_ball_frames,
        "total_stump_detections": total_stump_detections,
        "ball_detection_rate": ball_detection_rate,
        "ball_tracking_rate": tracking_quality["tracking_rate"],
        "interpolated_ball_frames": tracking_quality["interpolated_frames"],
        "kalman_predicted_frames": kalman_predicted_frames,
        "tracker_recoveries": tracker_recoveries,
        "overall_tracking_quality": overall_tracking_quality,
        "stump_detection_rate": stump_detection_rate,
        "average_ball_confidence": average_confidence,
        "full_frame_detection_time_ms": average_full_frame_detection_time,
        "roi_detection_time_ms": average_roi_detection_time,
        "roi_detected_frames": roi_detected_frames,
        "last_roi_size": last_roi_size,
        "estimated_bounce_point": estimated_bounce_point,
        "estimated_bounce_frame": estimated_bounce_frame,
        "pitch_normalized_bounce_point": pitch_normalized_bounce_point,
        "calibration_status": calibration_status,
        "calibration_source": calibration_source,
        "calibration_warning": calibration_warning,
        "estimated_line": estimated_line,
        "estimated_length": estimated_length,
        "wagon_wheel": wagon_wheel,
        "field_setup": field_setup,
        "batter_handedness": batter_handedness,
        "bowler_arm": bowler_arm,
        "camera_view": camera_view,
        "review_frame_count": review_frame_count,
        "review_frames_dir": REVIEW_FRAMES_DIR,
        "frame_detections": frame_detections,
        "impact_frame_detections": frame_detections,
        "observer_timeline": observer_timeline,
        "performance_profile": performance,
        "speed_mode": speed_mode,
        "impact_info": impact_info,
        "shot_info": shot_info,
        "direction_info": direction_info,
        "agent_info": agent_info,
        "shot_type": shot_info.get("shot_type", "Unknown"),
        "shot_confidence": shot_info.get("shot_confidence", "Unknown"),
        "shot_direction": shot_info.get("shot_direction", "Unknown"),
        "shot_height": shot_info.get("shot_height", "Unknown"),
        "shot_reason": shot_info.get("reason", shot_info.get("shot_reason", "")),
        "outcome_info": outcome_info,
        "predicted_outcome": outcome_info.get("predicted_outcome", "Unknown"),
        "outcome_confidence": outcome_info.get("outcome_confidence", "Unknown"),
        "run_estimate": outcome_info.get("run_estimate"),
        "dismissal_risk": outcome_info.get("dismissal_risk", "Unknown"),
        "boundary_chance": outcome_info.get("boundary_chance", "Unknown"),
        "outcome_reason": outcome_info.get("reason", outcome_info.get("outcome_reason", "")),
        "ball_model_used": "Current Best Ball + Stump Model",
        "bat_model_used": (
            (get_model_info(bat_model_key) or {}).get("name", bat_model_key)
            if bat_model_key and bat_model is not None
            else ("Unavailable" if bat_model_key else "Not used")
        ),
        **enrichment,
    }


def show_batting_analysis_results(result):
    from Backends.src.ui.components import (
        render_delivery_report,
        render_impact_frame_preview,
        render_impact_report,
        render_observer_timeline_report,
        render_outcome_prediction,
        render_performance_details,
        render_save_status,
        render_shot_direction_report,
        render_shot_report,
        render_vision_agent_report,
        video_preview_card,
    )

    video_preview_card("Processed Video Preview")
    output_path = result.get("output_path")
    if output_path:
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)
        with open(output_path, "rb") as video_file:
            st.download_button(
                "Download Processed Video",
                data=video_file,
                file_name=Path(output_path).name,
                mime="video/mp4",
                use_container_width=True,
                key="download_batting_processed_video",
            )
    else:
        st.warning("Processed video preview is not available for this result.")

    render_observer_timeline_report(result)
    render_delivery_report(result)
    render_impact_report(result)
    render_impact_frame_preview(result)
    render_shot_report(result)
    render_shot_direction_report(result)
    render_outcome_prediction(result)
    render_vision_agent_report(result)
    render_performance_details(result)
    render_save_status(result, "Video Analysis")


def show_video_analysis_results(result, selected_model_name, preset_name, show_pitch_roi):
    from Backends.src.ui.components import (
        render_delivery_report,
        render_impact_frame_preview,
        render_impact_report,
        render_observer_timeline_report,
        render_outcome_prediction,
        render_performance_details,
        render_save_status,
        render_shot_direction_report,
        render_shot_report,
        render_vision_agent_report,
        video_preview_card,
    )
    from Backends.src.ui.theme import render_status_pill

    st.markdown(
        f'<div style="margin:0.75rem 0 1rem 0;">{render_status_pill("Analysis Complete", "success")} '
        f'{render_status_pill(result.get("analysis_mode", "Full Delivery Analysis"), "gold")}</div>',
        unsafe_allow_html=True,
    )

    video_preview_card("Processed Video Preview")
    output_path = result.get("output_path")
    if output_path:
        with open(output_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)

        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="cricvision_processed_video.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
    else:
        st.warning("Processed video preview is not available for this result.")

    render_observer_timeline_report(result)
    render_delivery_report(result)
    render_impact_report(result)
    render_impact_frame_preview(result)
    render_shot_report(result)
    render_shot_direction_report(result)
    render_outcome_prediction(result)
    render_vision_agent_report(result)
    render_performance_details(result)
    render_save_status(result, "Video Analysis")


def show_video_analysis_page():
    from Backends.src.ui.components import clean_upload_box
    from Backends.src.ui.theme import render_empty_state, render_page_header

    render_page_header(
        "Analyze",
        "Upload a delivery clip. CricVision uses smart defaults and generates a processed video plus professional report.",
    )

    if "video_analysis_result" not in st.session_state:
        st.session_state.video_analysis_result = None
    if "video_analysis_settings" not in st.session_state:
        st.session_state.video_analysis_settings = {}

    model_options = get_model_options()

    from Backends.src.ui.interactive_field_map import render_field_setup_card

    field_setup = render_field_setup_card(key_prefix="video_analysis_field", compact=True, default_preset="Balanced")

    clean_upload_box("Upload cricket video")
    uploaded_video = st.file_uploader(
        "Upload delivery video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_analysis_upload",
        label_visibility="collapsed",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

    analyze_clicked = st.button(
        "Analyze Delivery",
        type="primary",
        use_container_width=True,
        disabled=uploaded_video is None,
        key="analyze_video_button",
    )

    speed_mode = st.selectbox(
        "Analysis Mode",
        ["Fast", "Balanced", "Accurate"],
        index=1,
        key="video_analysis_speed_mode",
        help="Fast samples fewer frames for quicker testing. Balanced is recommended.",
    )

    with st.expander("Advanced Settings", expanded=False):
        analysis_mode = st.selectbox(
            "Analysis mode",
            ["Bowling Analysis", "Batting Analysis", "Full Delivery Analysis"],
            index=2,
            key="video_analysis_mode",
        )
        selected_bat_model_key = None
        selected_ball_model_key = "current_best"
        selected_model_key = "current_best"

        if analysis_mode == "Batting Analysis":
            batting_ball_options = {
                "Current Best Ball + Stump Model": "current_best",
                "CricShot10k Ball Detector": "cricshot_ball",
            }
            selected_model_name = st.selectbox(
                "Ball model",
                list(batting_ball_options),
                key="video_analysis_batting_ball_model",
            )
            selected_ball_model_key = batting_ball_options[selected_model_name]
            selected_model_path = get_model_path(selected_ball_model_key)
            use_ensemble = False
            selected_bat_model_key = "cricshot_bat"
            st.selectbox("Bat model", ["CricShot10k Bat Detector"], key="video_analysis_bat_model")
        else:
            selected_model_name = st.selectbox(
                "Detection model",
                list(model_options.keys()),
                key="video_analysis_model",
            )
            selected_model = model_options[selected_model_name]
            selected_model_path = selected_model["path"]
            selected_model_key = selected_model.get("model_key")
            use_ensemble = selected_model.get("ensemble", False)
            if analysis_mode == "Full Delivery Analysis":
                selected_bat_model_key = "cricshot_bat"
                st.selectbox("Bat model", ["CricShot10k Bat Detector"], key="video_analysis_full_bat_model")

        preset_name = st.selectbox(
            "Detection preset",
            list(DETECTION_PRESETS.keys()),
            index=1,
            key="video_analysis_preset",
        )
        active_preset = DETECTION_PRESETS[preset_name]
        confidence = active_preset["confidence"]
        image_size = active_preset["imgsz"]

        show_pitch_roi = st.checkbox("Show pitch ROI overlay", value=False, key="video_analysis_show_roi")
        calibration_mode = st.radio(
            "Pitch calibration",
            [
                "Auto calibration using detected stumps",
                "Manual calibration using 4 pitch corner points",
            ],
            index=0,
            key="video_analysis_calibration_mode",
        )
        shot_trajectory_mode = st.radio(
            "Shot direction trajectory",
            [
                "Use full trajectory",
                "Use last part of trajectory",
                "Manually mark bat contact frame",
            ],
            index=1,
            key="video_analysis_shot_trajectory_mode",
        )
        manual_contact_frame = None
        if shot_trajectory_mode == "Manually mark bat contact frame":
            manual_contact_frame = st.number_input(
                "Bat contact frame",
                min_value=0,
                value=0,
                step=1,
                key="video_analysis_bat_contact_frame",
            )

        with st.expander("Model Status", expanded=False):
            for status in validate_model_paths().values():
                st.write(f"{status['status']}: {status['name']}")

        with st.expander("Advanced analysis settings", expanded=False):
            limit_frames_enabled = st.checkbox(
                "Limit frames for testing",
                value=False,
                key="video_analysis_limit_frames_enabled",
            )
            frame_limit_choice = st.selectbox(
                "Frame limit",
                [50, 100, 200, "All frames"],
                index=3,
                key="video_analysis_frame_limit_choice",
                disabled=not limit_frames_enabled,
            )
            st.checkbox(
                "Show performance details",
                value=False,
                key="video_analysis_show_performance",
            )

    analysis_mode = st.session_state.get("video_analysis_mode", "Full Delivery Analysis")
    preset_name = st.session_state.get("video_analysis_preset", "Balanced Mode")
    active_preset = DETECTION_PRESETS[preset_name]
    confidence = active_preset["confidence"]
    image_size = active_preset["imgsz"]
    show_pitch_roi = st.session_state.get("video_analysis_show_roi", False)
    calibration_mode = st.session_state.get(
        "video_analysis_calibration_mode",
        "Auto calibration using detected stumps",
    )
    shot_trajectory_mode = st.session_state.get(
        "video_analysis_shot_trajectory_mode",
        "Use last part of trajectory",
    )
    manual_contact_frame = st.session_state.get("video_analysis_bat_contact_frame", 0)
    if shot_trajectory_mode != "Manually mark bat contact frame":
        manual_contact_frame = None

    from Backends.src.analysis.analysis_speed import resolve_frame_limit

    speed_mode = st.session_state.get("video_analysis_speed_mode", "Balanced")
    max_frames = resolve_frame_limit(
        st.session_state.get("video_analysis_limit_frames_enabled", False),
        st.session_state.get("video_analysis_frame_limit_choice", "All frames"),
    )
    show_performance = st.session_state.get("video_analysis_show_performance", False)

    if analysis_mode == "Batting Analysis":
        batting_ball_options = {
            "Current Best Ball + Stump Model": "current_best",
            "CricShot10k Ball Detector": "cricshot_ball",
        }
        selected_model_name = st.session_state.get(
            "video_analysis_batting_ball_model",
            "Current Best Ball + Stump Model",
        )
        selected_ball_model_key = batting_ball_options.get(selected_model_name, "current_best")
        selected_model_path = get_model_path(selected_ball_model_key)
        selected_model_key = selected_ball_model_key
        use_ensemble = False
        selected_bat_model_key = "cricshot_bat"
    else:
        selected_model_name = st.session_state.get(
            "video_analysis_model",
            list(model_options.keys())[0],
        )
        selected_model = model_options.get(
            selected_model_name,
            list(model_options.values())[0],
        )
        selected_model_path = selected_model["path"]
        selected_model_key = selected_model.get("model_key")
        use_ensemble = selected_model.get("ensemble", False)
        selected_bat_model_key = "cricshot_bat" if analysis_mode == "Full Delivery Analysis" else None

    manual_pitch_points = None
    if analyze_clicked and uploaded_video is not None:
        if calibration_mode.startswith("Manual"):
            first_frame = extract_first_video_frame(uploaded_video)
            if first_frame is None:
                st.warning("Could not read the first frame for manual calibration.")
            else:
                with st.expander("Manual Pitch Calibration", expanded=True):
                    manual_pitch_points = show_manual_pitch_point_inputs(first_frame)

        uploaded_video.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
            temp_input.write(uploaded_video.read())
            input_video_path = Path(temp_input.name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        PROCESSED_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        raw_output_path = PROCESSED_VIDEO_DIR / f"raw_cricvision_analysis_{timestamp}.mp4"
        browser_output_path = PROCESSED_VIDEO_DIR / f"cricvision_analysis_{timestamp}.mp4"

        with st.spinner("Analyzing delivery..."):
            if analysis_mode == "Batting Analysis":
                result = process_batting_video(
                    video_path=input_video_path,
                    output_path=raw_output_path,
                    ball_model_key=selected_ball_model_key,
                    bat_model_key=selected_bat_model_key,
                    confidence=confidence,
                    speed_mode=speed_mode,
                    max_frames=max_frames,
                )
            else:
                result = process_video(
                    video_path=input_video_path,
                    output_path=raw_output_path,
                    model_path=selected_model_path,
                    model_key=selected_model_key,
                    confidence=confidence,
                    imgsz=image_size,
                    use_ensemble=use_ensemble,
                    show_pitch_roi=show_pitch_roi,
                    calibration_mode=calibration_mode,
                    manual_pitch_points=manual_pitch_points,
                    shot_trajectory_mode=shot_trajectory_mode,
                    manual_contact_frame=manual_contact_frame,
                    field_setup=field_setup,
                    bat_model_key=selected_bat_model_key,
                    speed_mode=speed_mode,
                    max_frames=max_frames,
                )
            result["analysis_mode"] = analysis_mode
            result["active_preset"] = preset_name
            result["active_model"] = selected_model_name
            result["ball_model_used"] = selected_model_name
            result["show_performance_details"] = show_performance

        if not result["success"]:
            st.error(result["error"])
            st.session_state.video_analysis_result = None
        else:
            try:
                final_video_path = convert_to_browser_mp4(
                    input_path=result["output_path"],
                    output_path=browser_output_path,
                )
                result["output_path"] = final_video_path
                if analysis_mode in {"Batting Analysis", "Full Delivery Analysis"}:
                    save_batting_report(result, analysis_mode)
                video_name = uploaded_video.name if uploaded_video is not None else None
                _persist_result_to_session(result, "Video Analysis", video_name=video_name)
                st.session_state.video_analysis_result = result
                st.session_state.video_analysis_settings = {
                    "analysis_mode": analysis_mode,
                    "selected_model_name": selected_model_name,
                    "preset_name": preset_name,
                    "show_pitch_roi": show_pitch_roi,
                    "shot_trajectory_mode": shot_trajectory_mode,
                    "speed_mode": speed_mode,
                    "show_performance_details": show_performance,
                }
                st.success("Analysis complete.")
            except Exception as error:
                st.error(f"Video conversion failed: {error}")
                st.session_state.video_analysis_result = None

    result = st.session_state.video_analysis_result
    settings = st.session_state.video_analysis_settings

    if result is None or not result.get("success"):
        render_empty_state(
            "No analysis yet",
            "Upload a clip and click Analyze Delivery to generate a processed video and report.",
            action_label="Smart defaults are applied automatically",
        )
    elif result.get("analysis_mode") == "Batting Analysis":
        show_batting_analysis_results(result)
    else:
        show_video_analysis_results(
            result=result,
            selected_model_name=settings.get("selected_model_name", result.get("active_model", "Unknown")),
            preset_name=settings.get("preset_name", result.get("active_preset", "Balanced Mode")),
            show_pitch_roi=settings.get("show_pitch_roi", False),
        )
                
def estimate_line_from_stumps(bounce_point, stump_detections, batter_handedness="right"):
    """
    Estimate cricket line using bounce point and stump position.

    Simple version:
    - left of stumps = off side for right-handed batters, leg side for left
    - inside stump width = middle
    - right of stumps = leg side for right-handed batters, off side for left
    """

    if bounce_point is None or not stump_detections:
        return "Unknown"

    bx, by = bounce_point

    # Use widest stump box if multiple stump boxes are detected
    main_stump = max(
        stump_detections,
        key=lambda item: item["box"][2] - item["box"][0]
    )

    x1, y1, x2, y2 = main_stump["box"]

    stump_width = x2 - x1
    margin = int(stump_width * 0.4)
    left_label = "Leg side" if normalize_handedness(batter_handedness) == "left" else "Off side"
    right_label = "Off side" if normalize_handedness(batter_handedness) == "left" else "Leg side"

    if bx < x1 - margin:
        return left_label
    elif bx > x2 + margin:
        return right_label
    else:
        return "Middle"
    

def estimate_length_from_bounce(bounce_point, frame_height):
    if bounce_point is None or frame_height <= 0:
        return "Unknown"

    bx, by = bounce_point

    bounce_ratio = by / frame_height

    if bounce_ratio >= 0.82:
        return "Yorker"
    elif bounce_ratio >= 0.68:
        return "Full"
    elif bounce_ratio >= 0.48:
        return "Good Length"
    else:
        return "Short"
    
def has_enough_ball_movement(trajectory_points, min_distance=40):
    if len(trajectory_points) < 2:
        return False

    start_x, start_y = trajectory_points[0]
    end_x, end_y = trajectory_points[-1]

    distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5

    return distance >= min_distance
