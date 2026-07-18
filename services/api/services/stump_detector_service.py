"""Lazy, box-cropped stump detection for live calibration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUMP_MODEL_PATH = PROJECT_ROOT / "Models" / "stump_detector" / "best.pt"
STUMP_MODEL_RELATIVE_PATH = "Models/stump_detector/best.pt"
STUMP_CLASS_FRAGMENTS = ("stump", "wicket")


def solve_stump_calibration(
    image: Image.Image,
    *,
    frame_width: int,
    frame_height: int,
    box_layout: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Detect one stump set inside each alignment box and build estimated geometry."""
    if not STUMP_MODEL_PATH.is_file():
        return {
            "success": False,
            "status": "stump_detector_missing",
            "message": (
                "Stump detector model is missing. Add a model at "
                "Models/stump_detector/best.pt."
            ),
            "detections": None,
            "virtual_stumps": None,
            "pitch_overlay": None,
            "calibration_quality": None,
            "environment_context": None,
        }

    try:
        model = _load_model(str(STUMP_MODEL_PATH))
    except Exception as exc:
        return {
            "success": False,
            "status": "stump_detector_error",
            "message": f"Stump detector could not be loaded: {type(exc).__name__}.",
            "detections": None,
            "virtual_stumps": None,
            "pitch_overlay": None,
            "calibration_quality": None,
            "environment_context": None,
        }

    actual_width, actual_height = image.size
    # ponytail: browser dimensions are retained in the contract; decoded image
    # dimensions are authoritative for crop and overlay coordinates.
    _ = frame_width, frame_height
    try:
        detections = {
            end: _detect_end(
                model,
                image,
                end,
                box_layout[end],
                actual_width,
                actual_height,
            )
            for end in ("striker", "non_striker")
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "stump_detector_error",
            "message": f"Stump detection failed: {type(exc).__name__}.",
            "detections": None,
            "virtual_stumps": None,
            "pitch_overlay": None,
            "calibration_quality": None,
            "environment_context": None,
        }

    virtual_stumps = {
        end: (
            _virtual_stumps_from_bbox(
                detection["bbox"],
                frame_width=actual_width,
                frame_height=actual_height,
            )
            if detection["found"] and detection["bbox"]
            else None
        )
        for end, detection in detections.items()
    }
    if not all(item["found"] for item in detections.values()):
        return {
            "success": False,
            "status": "stumps_not_found",
            "message": (
                "Could not detect both stump sets. Make sure both stump sets "
                "are inside the red boxes and try again."
            ),
            "detections": detections,
            "virtual_stumps": (
                virtual_stumps if any(virtual_stumps.values()) else None
            ),
            "pitch_overlay": None,
            "calibration_quality": None,
            "environment_context": None,
        }

    pitch_overlay = _build_pitch_overlay(
        detections,
        virtual_stumps,
        frame_width=actual_width,
        frame_height=actual_height,
    )
    confidence_score = round(
        sum(item["confidence"] for item in detections.values()) / 2,
        4,
    )
    return {
        "success": True,
        "status": "setup_complete",
        "message": "Both stump sets detected. Pitch setup is ready.",
        "detections": detections,
        "virtual_stumps": virtual_stumps,
        "pitch_overlay": pitch_overlay,
        "calibration_quality": {
            "status": "good",
            "score": confidence_score,
        },
        # Kept for clients using the previous calibration response.
        "environment_context": pitch_overlay,
    }


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    # Lazy by design: Dashboard/API import and missing-model calibration do not
    # import Ultralytics or construct YOLO.
    from ultralytics import YOLO

    return YOLO(model_path)


def detect_stump_candidates(image: Image.Image) -> dict[str, Any]:
    """Run the shared cached stump model once over a complete reference frame."""
    if not STUMP_MODEL_PATH.is_file():
        return {
            "success": False,
            "status": "stump_detector_missing",
            "message": (
                "Stump detector model not found at "
                f"{STUMP_MODEL_RELATIVE_PATH}"
            ),
            "candidates": [],
        }

    try:
        model = _load_model(str(STUMP_MODEL_PATH))
        results = model.predict(source=image, verbose=False)
    except Exception as exc:
        return {
            "success": False,
            "status": "stump_detector_error",
            "message": f"Stump detector inference failed: {type(exc).__name__}.",
            "candidates": [],
        }

    frame_width, frame_height = image.size
    candidates: list[dict[str, Any]] = []
    try:
        for result in results or []:
            names = getattr(result, "names", None) or getattr(model, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            classes = _to_list(getattr(boxes, "cls", []))
            confidences = _to_list(getattr(boxes, "conf", []))
            coordinates = _to_list(getattr(boxes, "xyxy", []))
            for class_id, confidence, xyxy in zip(
                classes,
                confidences,
                coordinates,
            ):
                class_name = _class_name(names, int(class_id))
                if not _is_stump_class(class_name, names):
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy[:4])
                x1 = max(0.0, min(float(frame_width), x1))
                y1 = max(0.0, min(float(frame_height), y1))
                x2 = max(x1, min(float(frame_width), x2))
                y2 = max(y1, min(float(frame_height), y2))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                candidates.append(
                    {
                        "confidence": round(float(confidence), 4),
                        "class_name": class_name,
                        "bbox": {
                            "x": x1,
                            "y": y1,
                            "width": x2 - x1,
                            "height": y2 - y1,
                        },
                    }
                )
    except Exception as exc:
        return {
            "success": False,
            "status": "stump_detector_error",
            "message": f"Stump detector results could not be read: {type(exc).__name__}.",
            "candidates": [],
        }

    candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
    return {
        "success": True,
        "status": "candidates_ready",
        "message": "Stump detection completed.",
        "candidates": candidates,
    }


def _detect_end(
    model,
    image: Image.Image,
    source_box: str,
    normalized_box: dict[str, float],
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    crop_bounds = _normalized_box_to_pixels(
        normalized_box,
        frame_width,
        frame_height,
    )
    x1, y1, x2, y2 = crop_bounds
    crop = image.crop(crop_bounds)
    results = model.predict(source=crop, verbose=False)

    best: dict[str, Any] | None = None
    for result in results or []:
        names = getattr(result, "names", None) or getattr(model, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        classes = _to_list(getattr(boxes, "cls", []))
        confidences = _to_list(getattr(boxes, "conf", []))
        coordinates = _to_list(getattr(boxes, "xyxy", []))
        for class_id, confidence, xyxy in zip(classes, confidences, coordinates):
            class_name = _class_name(names, int(class_id))
            if not _is_stump_class(class_name, names):
                continue
            confidence = float(confidence)
            if best is not None and confidence <= best["confidence"]:
                continue
            bx1, by1, bx2, by2 = (float(value) for value in xyxy[:4])
            full_x1 = max(0, min(frame_width, round(x1 + bx1)))
            full_y1 = max(0, min(frame_height, round(y1 + by1)))
            full_x2 = max(full_x1, min(frame_width, round(x1 + bx2)))
            full_y2 = max(full_y1, min(frame_height, round(y1 + by2)))
            best = {
                "found": True,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x": full_x1,
                    "y": full_y1,
                    "width": full_x2 - full_x1,
                    "height": full_y2 - full_y1,
                },
                "source_box": source_box,
                "class_name": class_name,
            }

    return best or {
        "found": False,
        "confidence": 0.0,
        "bbox": None,
        "source_box": source_box,
    }


def _normalized_box_to_pixels(
    box: dict[str, float], frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(frame_width - 1, round(box["x"] * frame_width)))
    y1 = max(0, min(frame_height - 1, round(box["y"] * frame_height)))
    x2 = max(x1 + 1, min(frame_width, round((box["x"] + box["width"]) * frame_width)))
    y2 = max(y1 + 1, min(frame_height, round((box["y"] + box["height"]) * frame_height)))
    return x1, y1, x2, y2


def _virtual_stumps_from_bbox(
    bbox: dict[str, int],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    x = max(0, min(frame_width - 1, bbox["x"]))
    y = max(0, min(frame_height - 1, bbox["y"]))
    right = max(x, min(frame_width - 1, bbox["x"] + bbox["width"]))
    bottom = max(y, min(frame_height - 1, bbox["y"] + bbox["height"]))
    width = right - x
    height = bottom - y
    top_y = round(y + height * 0.08)
    base_y = bottom
    return {
        "geometry_type": "estimated_from_bbox",
        "stumps": [
            {
                "name": name,
                "top": {"x": round(x + width * fraction), "y": top_y},
                "base": {"x": round(x + width * fraction), "y": base_y},
            }
            for name, fraction in (("left", 0.2), ("middle", 0.5), ("right", 0.8))
        ],
        "bails": [
            {
                "name": "left_bail",
                "start": {"x": round(x + width * 0.14), "y": top_y},
                "end": {"x": round(x + width * 0.5), "y": top_y},
            },
            {
                "name": "right_bail",
                "start": {"x": round(x + width * 0.5), "y": top_y},
                "end": {"x": round(x + width * 0.86), "y": top_y},
            },
        ],
    }


def _build_pitch_overlay(
    detections: dict[str, dict[str, Any]],
    virtual_stumps: dict[str, Any],
    *,
    frame_width: int,
    frame_height: int,
) -> dict[str, Any]:
    striker = detections["striker"]["bbox"]
    non_striker = detections["non_striker"]["bbox"]

    def center(box: dict[str, int]) -> dict[str, int]:
        return {
            "x": max(
                0,
                min(frame_width - 1, round(box["x"] + box["width"] / 2)),
            ),
            "y": max(
                0,
                min(frame_height - 1, round(box["y"] + box["height"])),
            ),
        }

    striker_center = center(striker)
    non_striker_center = center(non_striker)
    corridor = [
        {"x": max(0, non_striker["x"]), "y": non_striker_center["y"]},
        {"x": max(0, striker["x"]), "y": striker_center["y"]},
        {
            "x": min(frame_width - 1, striker["x"] + striker["width"]),
            "y": striker_center["y"],
        },
        {
            "x": min(
                frame_width - 1,
                non_striker["x"] + non_striker["width"],
            ),
            "y": non_striker_center["y"],
        },
    ]

    def crease(box: dict[str, int]) -> list[dict[str, int]]:
        extension = round(box["width"] * 0.45)
        y = min(frame_height - 1, box["y"] + box["height"])
        return [
            {"x": max(0, box["x"] - extension), "y": y},
            {
                "x": min(
                    frame_width - 1,
                    box["x"] + box["width"] + extension,
                ),
                "y": y,
            },
        ]

    return {
        "geometry_type": "estimated_from_stump_bboxes",
        "pitch_axis": {
            "start": non_striker_center,
            "end": striker_center,
        },
        "pitch_corridor": corridor,
        "center_line": [non_striker_center, striker_center],
        "wickets": virtual_stumps,
        "crease_guides": {
            "striker": crease(striker),
            "non_striker": crease(non_striker),
        },
    }


def save_debug_overlay(
    image: Image.Image,
    *,
    box_layout: dict[str, dict[str, float]],
    detections: dict[str, dict[str, Any]] | None,
    virtual_stumps: dict[str, Any] | None,
    pitch_overlay: dict[str, Any] | None,
    output_path: Path,
) -> Path:
    debug = image.copy()
    draw = ImageDraw.Draw(debug, "RGBA")
    width, height = debug.size

    corridor = (pitch_overlay or {}).get("pitch_corridor") or []
    if len(corridor) == 4:
        polygon = [(point["x"], point["y"]) for point in corridor]
        draw.polygon(polygon, fill=(255, 216, 92, 35), outline=(255, 216, 92, 210), width=3)
    center_line = (pitch_overlay or {}).get("center_line") or []
    if len(center_line) == 2:
        draw.line(
            (
                center_line[0]["x"],
                center_line[0]["y"],
                center_line[1]["x"],
                center_line[1]["y"],
            ),
            fill=(255, 255, 255, 210),
            width=2,
        )
    for guide in ((pitch_overlay or {}).get("crease_guides") or {}).values():
        if len(guide) == 2:
            draw.line(
                (guide[0]["x"], guide[0]["y"], guide[1]["x"], guide[1]["y"]),
                fill=(255, 255, 255, 190),
                width=2,
            )

    for end in ("striker", "non_striker"):
        x1, y1, x2, y2 = _normalized_box_to_pixels(
            box_layout[end], width, height
        )
        draw.rectangle((x1, y1, x2, y2), outline="red", width=4)
        draw.text((x1 + 5, y1 + 5), end, fill="red")

        detection = (detections or {}).get(end) or {}
        bbox = detection.get("bbox")
        if detection.get("found") and bbox:
            bx1, by1 = bbox["x"], bbox["y"]
            bx2 = bx1 + bbox["width"]
            by2 = by1 + bbox["height"]
            draw.rectangle((bx1, by1, bx2, by2), outline="lime", width=4)
            draw.text(
                (bx1 + 5, max(0, by1 - 14)),
                f"{end} {detection['confidence']:.2f}",
                fill="lime",
            )

        end_geometry = (virtual_stumps or {}).get(end) or {}
        for stump in end_geometry.get("stumps", []):
            top = stump["top"]
            base = stump["base"]
            draw.line((top["x"], top["y"], base["x"], base["y"]), fill="yellow", width=3)
        for bail in end_geometry.get("bails", []):
            start = bail["start"]
            end_point = bail["end"]
            draw.line(
                (start["x"], start["y"], end_point["x"], end_point["y"]),
                fill="yellow",
                width=4,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug.save(output_path, format="JPEG", quality=92)
    return output_path


def _to_list(value) -> list:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value or [])


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _is_stump_class(class_name: str, names) -> bool:
    normalized = class_name.strip().lower()
    if any(fragment in normalized for fragment in STUMP_CLASS_FRAGMENTS):
        return True
    # A single-purpose detector may use an arbitrary class label.
    return isinstance(names, (dict, list, tuple)) and len(names) == 1
