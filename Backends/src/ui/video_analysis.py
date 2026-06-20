import tempfile
import subprocess
import csv
import zipfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import streamlit as st
from ultralytics import YOLO

from Backends.src.analysis.cricket_agent import (
    calculate_detection_quality,
    detect_analysis_warnings,
    generate_coaching_feedback,
    generate_delivery_report,
)
from Backends.src.analysis.field_zones import FIELD_ZONES, generate_wagon_wheel_data
from Backends.src.analysis.field_zones import (
    DETAILED_FIELD_ZONES,
    find_nearest_fielder,
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


BALL_MODEL_PATH = Path("Models/ball_detector/best.pt")
CRICKET_OBJECTS_MODEL_PATH = Path("Models/cricket_objects/best.pt")
EXTERNAL_BALL_MODEL_PATH = Path("Models/cricket_objects/best_external.pt")
OUTPUT_DIR = Path("outputs/video_analysis")
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

    return YOLO(str(model_path))


def get_model_options():
    return {
        "Ball + Stump Detector": {
            "path": CRICKET_OBJECTS_MODEL_PATH,
        },
        "Old Ball Detector": {
            "path": BALL_MODEL_PATH,
        },
        "External Ball Model": {
            "path": EXTERNAL_BALL_MODEL_PATH,
        },
        ENSEMBLE_MODEL_NAME: {
            "path": None,
            "ensemble": True,
        },
    }


def get_ensemble_model_configs():
    return [
        {
            "name": "Ball + Stump Detector",
            "path": CRICKET_OBJECTS_MODEL_PATH,
            "use_ball": True,
            "use_stump": True,
        },
        {
            "name": "Old Ball Detector",
            "path": BALL_MODEL_PATH,
            "use_ball": True,
            "use_stump": False,
        },
        {
            "name": "External Ball Model",
            "path": EXTERNAL_BALL_MODEL_PATH,
            "use_ball": True,
            "use_stump": False,
        },
    ]


def get_available_ensemble_model_names():
    return [
        config["name"]
        for config in get_ensemble_model_configs()
        if config["path"].exists()
    ]


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


def estimate_line_from_pitch_x(pitch_x):
    if pitch_x is None:
        return "Unknown"
    if pitch_x < 0.38:
        return "Off side"
    if pitch_x > 0.62:
        return "Leg side"
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


def draw_pitch_map(pitch_point, line_label="Unknown", length_label="Unknown"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_facecolor("#d6b47a")
    fig.patch.set_facecolor("white")

    length_zones = [
        (0.00, 0.45, "Short", "#f6c2c2"),
        (0.45, 0.68, "Good Length", "#fff1a8"),
        (0.68, 0.84, "Full", "#bfe7c2"),
        (0.84, 1.00, "Yorker", "#acd7ff"),
    ]

    for y1, y2, label, color in length_zones:
        ax.axhspan(y1, y2, color=color, alpha=0.75)
        ax.text(0.03, (y1 + y2) / 2, label, va="center", ha="left", fontsize=9)

    ax.axvline(0.38, color="#444444", linestyle="--", linewidth=1)
    ax.axvline(0.62, color="#444444", linestyle="--", linewidth=1)
    ax.text(0.19, 1.04, "Off", ha="center", fontsize=9)
    ax.text(0.50, 1.04, "Middle", ha="center", fontsize=9)
    ax.text(0.81, 1.04, "Leg", ha="center", fontsize=9)

    if pitch_point is not None:
        ax.scatter(
            [pitch_point[0]],
            [pitch_point[1]],
            s=90,
            color="#d71920",
            edgecolor="white",
            linewidth=1.5,
            zorder=5,
        )
        ax.text(
            pitch_point[0],
            max(0, pitch_point[1] - 0.04),
            f"{line_label} / {length_label}",
            ha="center",
            fontsize=8,
            color="#111111",
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Pitch Map", fontsize=12)
    fig.tight_layout()
    return fig


def load_ensemble_models():
    models = []

    for config in get_ensemble_model_configs():
        if not config["path"].exists():
            continue

        model = load_yolo_model(str(config["path"]))

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

        if not config["path"].exists():
            continue

        model = load_yolo_model(str(config["path"]))

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
):
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


def process_video(
    video_path,
    output_path,
    model_path,
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
):
    model = None
    ensemble_models = []
    stump_model = load_yolo_model(str(CRICKET_OBJECTS_MODEL_PATH))

    if stump_model is None:
        return {
            "success": False,
            "error": f"Stump model not found: {CRICKET_OBJECTS_MODEL_PATH}",
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
        model = load_yolo_model(str(model_path))

        if model is None:
            return {
                "success": False,
                "error": f"Model not found: {model_path}",
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        cap.release()
        return {
            "success": False,
            "error": "Could not read video width/height.",
        }

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
    
    valid_ball_track_started = False
    min_track_points_for_bounce = 8
    min_movement_distance = 40
    min_ball_confidence_for_tracking = 0.35 

    while True:
        success, frame = cap.read()

        if not success:
            break

        last_raw_frame = frame.copy()
        annotated_frame = frame.copy()
        low_confidence_ball_detections = []

        detection_result = run_pitch_roi_detection(
            frame,
            stump_model=stump_model,
            stump_class_names=stump_class_names,
            confidence=confidence,
            imgsz=imgsz,
            previous_roi=previous_roi_box,
            ball_model=model,
            ball_class_names=class_names,
            ensemble_models=ensemble_models,
            use_ensemble=use_ensemble,
            ball_confidence=min_ball_confidence_for_tracking,
        )
        ball_detections = detection_result["ball_detections"]
        stump_detections = detection_result["stump_detections"]
        low_confidence_ball_detections = detection_result.get(
            "low_confidence_ball_detections",
            [],
        )
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
        else:
            search_center = previous_ball_center or kalman_tracker.last_prediction
            recovery_result = run_local_redetection(
                frame,
                search_center,
                confidence,
                imgsz,
                missing_ball_frames + 1,
                ball_model=model,
                ball_class_names=class_names,
                ensemble_models=ensemble_models,
                use_ensemble=use_ensemble,
            )

            if show_pitch_roi:
                draw_search_roi(annotated_frame, recovery_result.get("search_roi"))

            if recovery_result["recovered"]:
                ball_detections = recovery_result["ball_detections"]
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

        stump_detections_by_frame.append(stump_detections)

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
                    bounce_stump_detections
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
                    estimated_line = estimate_line_from_pitch_x(pitch_x)
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
            f"Frame: {frame_index}/{total_frames}",
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

        writer.write(annotated_frame)

        frame_index += 1

        if total_frames > 0:
            progress = min(frame_index / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_index}/{total_frames}")
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

    ball_detection_rate = 0
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
    field_setup = field_setup or {}
    batter_handedness = field_setup.get("batter_handedness", "Right-hand batter")
    bowler_arm = field_setup.get("bowler_arm", "Right-arm bowler")
    camera_view = field_setup.get("camera_view", "Behind bowler")
    fielders = field_setup.get("fielders", [])
    field_preset = field_setup.get("preset", "Custom")

    wagon_wheel = generate_wagon_wheel_data(
        ball_positions,
        batter_handedness=batter_handedness,
        mode=shot_trajectory_mode,
        manual_contact_frame=manual_contact_frame,
    )
    wagon_wheel["mode"] = shot_trajectory_mode
    nearest_fielder = find_nearest_fielder(wagon_wheel.get("shot_angle"), fielders)
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
    }


def show_video_analysis_results(result, selected_model_name, preset_name, show_pitch_roi):
    from Backends.src.ui.ui_components import badge_row, info_panel, metric_card, section_header, status_badge
    from Backends.src.ui.field_map import draw_field_map

    tracking_quality = result.get("overall_tracking_quality", "Poor")
    tracking_tone = "green" if tracking_quality in {"Excellent", "Good"} else "amber"

    badge_row(
        [
            status_badge(f"Model: {result.get('active_model', selected_model_name)}", "cyan"),
            status_badge(f"Preset: {result.get('active_preset', preset_name)}", "blue"),
            status_badge(
                f"ROI: {'Enabled' if show_pitch_roi else 'Hidden'}",
                "green" if show_pitch_roi else "muted",
            ),
            status_badge(f"Tracking: {tracking_quality}", tracking_tone),
        ]
    )

    section_header("Processed Video")
    with open(result["output_path"], "rb") as video_file:
        video_bytes = video_file.read()
    st.video(video_bytes)

    section_header("Analysis Stats")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        metric_card("Total Frames", str(result["total_frames"]), "Processed clip length")
    with stat_cols[1]:
        metric_card("Ball Frames", str(result["ball_detected_frames"]), "Frames with ball detections")
    with stat_cols[2]:
        metric_card("Stump Frames", str(result["stump_detected_frames"]), "Frames with stump detections")
    with stat_cols[3]:
        metric_card("Ball Detection Rate", f"{result['ball_detection_rate']:.1f}%", "Detection coverage")

    stat_cols_2 = st.columns(4)
    with stat_cols_2[0]:
        metric_card("Ball Tracking Rate", f"{result.get('ball_tracking_rate', 0):.1f}%", "Continuous tracking")
    with stat_cols_2[1]:
        metric_card("Avg Confidence", f"{result['average_ball_confidence']:.2f}", "Mean ball confidence")
    with stat_cols_2[2]:
        metric_card("Recoveries", str(result.get("tracker_recoveries", 0)), "Local re-detection events")
    with stat_cols_2[3]:
        metric_card("Review Frames", str(result.get("review_frame_count", 0)), "Saved for training review")

    timing_cols = st.columns(3)
    with timing_cols[0]:
        metric_card(
            "Full Frame Time",
            f"{result.get('full_frame_detection_time_ms', 0):.1f} ms",
            "Average full-frame detection",
        )
    with timing_cols[1]:
        metric_card(
            "ROI Time",
            f"{result.get('roi_detection_time_ms', 0):.1f} ms",
            "Average ROI detection",
        )
    with timing_cols[2]:
        metric_card("ROI Frames", str(result.get("roi_detected_frames", 0)), result.get("last_roi_size", "Full frame"))

    section_header("Bounce / Pitch Estimate")
    calib_col1, calib_col2 = st.columns(2)
    calib_col1.metric("Calibration", result.get("calibration_status", "Not calibrated"))
    calib_col2.metric("Calibration Mode", result.get("calibration_source", "None"))

    if result.get("calibration_warning"):
        st.warning(result["calibration_warning"])

    if result["estimated_bounce_point"] is not None:
        bx, by = result["estimated_bounce_point"]
        bounce_cols = st.columns(5)
        bounce_cols[0].metric("Bounce Frame", result["estimated_bounce_frame"])
        bounce_cols[1].metric("Bounce X", bx)
        bounce_cols[2].metric("Bounce Y", by)
        bounce_cols[3].metric("Estimated Line", result["estimated_line"])
        bounce_cols[4].metric("Estimated Length", result["estimated_length"])

        normalized_bounce = result.get("pitch_normalized_bounce_point")
        if normalized_bounce is not None:
            pitch_x, pitch_y = normalized_bounce
            norm_col1, norm_col2 = st.columns(2)
            norm_col1.metric("Pitch X", f"{pitch_x:.2f}")
            norm_col2.metric("Pitch Y", f"{pitch_y:.2f}")

        st.success("Estimated bounce/pitch point found.")
    else:
        st.warning("Bounce/pitch point was not found. Try a clearer or longer clip.")

    section_header("Pitch Map")
    pitch_map_fig = draw_pitch_map(
        result.get("pitch_normalized_bounce_point"),
        result.get("estimated_line", "Unknown"),
        result.get("estimated_length", "Unknown"),
    )
    st.pyplot(pitch_map_fig)

    section_header("Batting Direction")
    wagon_wheel = result.get("wagon_wheel", {})
    shot_angle = wagon_wheel.get("shot_angle")
    shot_zone = wagon_wheel.get("estimated_zone", "Unknown")
    simple_zone = wagon_wheel.get("simple_zone", shot_zone)
    detailed_zone = wagon_wheel.get("detailed_zone", "Unknown")
    nearest_fielder = wagon_wheel.get("nearest_fielder")
    nearest_fielder_name = "Unknown" if nearest_fielder is None else nearest_fielder.get("name", "Unknown")
    shot_confidence = wagon_wheel.get("confidence", "Low")

    shot_cols = st.columns(4)
    shot_cols[0].metric("Estimated Shot Zone", simple_zone)
    shot_cols[1].metric("Shot Angle", "Unknown" if shot_angle is None else f"{shot_angle:.1f} deg")
    shot_cols[2].metric("Detailed Zone", detailed_zone)
    shot_cols[3].metric("Confidence", shot_confidence)
    context_cols = st.columns(3)
    context_cols[0].metric("Batter", result.get("batter_handedness", "Unknown"))
    context_cols[1].metric("Bowler Arm", result.get("bowler_arm", "Unknown"))
    context_cols[2].metric("Nearest Fielder", nearest_fielder_name)

    info_panel(wagon_wheel.get("suggested_adjustment", "No field adjustment suggestion available."))

    if not wagon_wheel.get("success"):
        st.warning(wagon_wheel.get("message", "Shot direction is uncertain."))

    st.pyplot(
        draw_field_map(
            shot_angle=shot_angle,
            selected_zone=detailed_zone,
            fielders=result.get("field_setup", {}).get("fielders", []),
        )
    )

    correction_col1, correction_col2 = st.columns([2, 1])
    with correction_col1:
        corrected_zone = st.selectbox(
            "Manual correction: actual field zone",
            ["No correction"] + DETAILED_FIELD_ZONES,
            key="video_analysis_field_zone_correction",
        )

    with correction_col2:
        save_correction = st.button("Save Zone Correction", key="save_field_zone_correction")

    if save_correction:
        if corrected_zone == "No correction":
            st.warning("Choose an actual zone before saving a correction.")
        else:
            save_field_zone_correction(
                {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "estimated_zone": shot_zone,
                    "corrected_zone": corrected_zone,
                    "shot_angle": "" if shot_angle is None else f"{shot_angle:.2f}",
                    "confidence": shot_confidence,
                    "mode": wagon_wheel.get("mode", ""),
                    "source": "video_analysis",
                }
            )
            save_field_analysis_history(
                {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "source": "video_analysis_correction",
                    "batter_handedness": result.get("batter_handedness", ""),
                    "bowler_arm": result.get("bowler_arm", ""),
                    "camera_view": result.get("camera_view", ""),
                    "preset": result.get("field_setup", {}).get("preset", "Custom"),
                    "simple_zone": simple_zone,
                    "detailed_zone": detailed_zone,
                    "shot_angle": "" if shot_angle is None else f"{shot_angle:.2f}",
                    "nearest_fielder": nearest_fielder_name,
                    "confidence": shot_confidence,
                    "corrected_zone": corrected_zone,
                }
            )
            result["wagon_wheel"]["corrected_zone"] = corrected_zone
            st.session_state.video_analysis_result = result
            st.success("Field-zone correction saved for future review.")

    show_cricket_delivery_report(result)
    st.caption(f"Saved output: {result['output_path']}")

    with open(result["output_path"], "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="cricvision_processed_video.mp4",
            mime="video/mp4",
        )

    if st.button("Export Review Frames for Training", key="export_review_frames_video"):
        zip_path, file_count = create_review_frames_zip()

        if file_count == 0:
            st.warning("No review frames are available yet.")
        else:
            with open(zip_path, "rb") as zip_file:
                st.download_button(
                    label="Download Review Frames ZIP",
                    data=zip_file,
                    file_name=zip_path.name,
                    mime="application/zip",
                    key="download_review_frames_zip_video",
                )


def show_video_analysis_page():
    from Backends.src.ui.ui_components import badge_row, card, info_panel, page_header, section_header, status_badge
    from Backends.src.ui.field_map import draw_field_map, field_setup_editor

    page_header(
        "Video Analysis",
        "Upload a bowling clip. CricVision AI will detect balls, stumps, draw the ball trajectory, and estimate the bounce/pitch point.",
    )

    if "video_analysis_result" not in st.session_state:
        st.session_state.video_analysis_result = None
    if "video_analysis_settings" not in st.session_state:
        st.session_state.video_analysis_settings = {}

    model_options = get_model_options()
    tab_upload, tab_model, tab_debug, tab_results = st.tabs(
        ["Upload", "Model Settings", "Advanced Debug", "Results"]
    )

    with tab_model:
        section_header("Detection Model")
        selected_model_name = st.selectbox(
            "Choose detection model",
            list(model_options.keys()),
            key="video_analysis_model",
        )

        selected_model = model_options[selected_model_name]
        selected_model_path = selected_model["path"]
        use_ensemble = selected_model.get("ensemble", False)

        if not use_ensemble and not selected_model_path.exists():
            st.error(f"Model not found: {selected_model_path}")
            info_panel("Make sure your model file is inside the correct Models folder.")
            st.stop()

        badge_row([status_badge(f"Model: {selected_model_name}", "cyan")])

        if use_ensemble:
            active_model_names = get_available_ensemble_model_names()
            if active_model_names:
                info_panel("Active ensemble models: " + ", ".join(active_model_names))
            else:
                st.warning("No configured ensemble model files were found.")
        else:
            st.caption(f"Model path: {selected_model_path}")

        info_panel(
            "If the selected model does not include stumps, line detection may remain Unknown."
        )

        section_header("Detection Preset")
        preset_name = st.selectbox(
            "Detection preset",
            list(DETECTION_PRESETS.keys()),
            index=1,
            key="video_analysis_preset",
        )
        active_preset = DETECTION_PRESETS[preset_name]
        confidence = active_preset["confidence"]
        image_size = active_preset["imgsz"]
        badge_row([status_badge(f"Preset: {preset_name}", "blue")])

        st.caption(
            f"Active preset: {preset_name} | imgsz={image_size} | confidence={confidence:.2f}"
        )

    with tab_debug:
        section_header("Debug & Calibration")
        show_pitch_roi = st.checkbox("Show Pitch ROI", value=False, key="video_analysis_show_roi")
        calibration_mode = st.radio(
            "Pitch calibration mode",
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

        badge_row(
            [
                status_badge(f"ROI Overlay: {'On' if show_pitch_roi else 'Off'}", "green" if show_pitch_roi else "muted"),
                status_badge(
                    "Calibration: Auto" if calibration_mode.startswith("Auto") else "Calibration: Manual",
                    "blue",
                ),
                status_badge(f"Shot: {shot_trajectory_mode}", "cyan"),
            ]
        )

        with st.expander("Runtime Debug Panel", expanded=False):
            result = st.session_state.video_analysis_result
            if result and result.get("success"):
                st.write(f"Active model: {result.get('active_model', selected_model_name)}")
                st.write(f"Active preset: {result.get('active_preset', preset_name)}")
                st.write(f"ROI size: {result.get('last_roi_size', 'Full frame')}")
                st.write(f"Ball detections: {result.get('total_ball_detections', 0)}")
                st.write(f"Tracker recoveries: {result.get('tracker_recoveries', 0)}")
                st.write(f"Average confidence: {result.get('average_ball_confidence', 0):.2f}")
                st.write(
                    f"Calibration: {result.get('calibration_status', 'Not calibrated')} "
                    f"({result.get('calibration_source', 'None')})"
                )
            else:
                st.caption("Run an analysis to populate debug metrics.")

        with st.expander("Set Field Before Delivery", expanded=False):
            field_setup = field_setup_editor("video_analysis", default_preset="Attacking Field")
            st.pyplot(
                draw_field_map(
                    shot_angle=None,
                    selected_zone="Unknown",
                    fielders=field_setup["fielders"],
                )
            )

    with tab_upload:
        section_header("Upload Delivery Clip")
        info_panel(
            "For phone videos, use landscape mode if possible. Good lighting and a stable camera will improve tracking."
        )
        uploaded_video = st.file_uploader(
            "Upload bowling video from phone or camera",
            type=["mp4", "mov", "avi", "mkv"],
            key="video_analysis_upload",
        )

        manual_pitch_points = None
        if uploaded_video is not None:
            section_header("Original Video")
            st.video(uploaded_video)

            if calibration_mode.startswith("Manual"):
                section_header("Manual Pitch Calibration")
                first_frame = extract_first_video_frame(uploaded_video)

                if first_frame is None:
                    st.warning("Could not read the first frame for manual calibration.")
                else:
                    manual_pitch_points = show_manual_pitch_point_inputs(first_frame)

            if st.button("Analyze Video", type="primary", key="analyze_video_button"):
                uploaded_video.seek(0)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
                    temp_input.write(uploaded_video.read())
                    input_video_path = Path(temp_input.name)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                raw_output_path = OUTPUT_DIR / f"raw_cricvision_analysis_{timestamp}.mp4"
                browser_output_path = OUTPUT_DIR / f"cricvision_analysis_{timestamp}.mp4"

                with st.spinner("Analyzing video..."):
                    result = process_video(
                        video_path=input_video_path,
                        output_path=raw_output_path,
                        model_path=selected_model_path,
                        confidence=confidence,
                        imgsz=image_size,
                        use_ensemble=use_ensemble,
                        show_pitch_roi=show_pitch_roi,
                        calibration_mode=calibration_mode,
                        manual_pitch_points=manual_pitch_points,
                        shot_trajectory_mode=shot_trajectory_mode,
                        manual_contact_frame=manual_contact_frame,
                        field_setup=field_setup,
                    )
                    result["active_preset"] = preset_name
                    result["active_model"] = selected_model_name

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
                        st.session_state.video_analysis_result = result
                        st.session_state.video_analysis_settings = {
                            "selected_model_name": selected_model_name,
                            "preset_name": preset_name,
                            "show_pitch_roi": show_pitch_roi,
                            "shot_trajectory_mode": shot_trajectory_mode,
                        }
                        st.success("Video analysis completed. Open the Results tab to review.")
                    except Exception as error:
                        st.error(f"Video conversion failed: {error}")
                        info_panel(
                            "The analysis worked, but the final video could not be converted for browser playback."
                        )
                        st.session_state.video_analysis_result = None

    with tab_results:
        result = st.session_state.video_analysis_result
        settings = st.session_state.video_analysis_settings

        if result is None or not result.get("success"):
            card(
                title="No Results Yet",
                content_html=(
                    "Upload a clip in the <strong>Upload</strong> tab, configure model settings, "
                    "then click <strong>Analyze Video</strong> to generate processed video and reports."
                ),
            )
        else:
            show_video_analysis_results(
                result=result,
                selected_model_name=settings.get("selected_model_name", result.get("active_model", "Unknown")),
                preset_name=settings.get("active_preset", settings.get("preset_name", result.get("active_preset", "Unknown"))),
                show_pitch_roi=settings.get("show_pitch_roi", False),
            )
                
def estimate_line_from_stumps(bounce_point, stump_detections):
    """
    Estimate cricket line using bounce point and stump position.

    Simple version:
    - left of stumps = off side
    - inside stump width = middle
    - right of stumps = leg side

    This assumes right-handed batter view for now.
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

    if bx < x1 - margin:
        return "Off side"
    elif bx > x2 + margin:
        return "Leg side"
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
