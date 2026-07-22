"""Lazy stump detection for live box-crop calibration and video auto-cal."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUMP_MODEL_PATH = PROJECT_ROOT / "Models" / "stump_detector" / "best.pt"
STUMP_MODEL_RELATIVE_PATH = "Models/stump_detector/best.pt"
STUMP_CLASS_FRAGMENTS = ("stump", "wicket")

# ponytail: rear-camera nets — far stump sits mid-upper; near stump is large
# lower-centre and often touches the bottom. Broader than Live alignment boxes.
FAR_WICKET_ROI = {"x": 0.18, "y": 0.15, "width": 0.64, "height": 0.50}
NEAR_WICKET_ROI = {"x": 0.05, "y": 0.45, "width": 0.90, "height": 0.55}
ROI_LAYOUTS = {
    "far_roi": FAR_WICKET_ROI,
    "near_roi": NEAR_WICKET_ROI,
}
# Keep Ultralytics default-ish floor; do not globally drop conf to near-zero.
DEFAULT_DETECT_CONF = 0.25
ROI_DETECT_CONF = 0.20
ROI_HIGHRES_IMGSZ = 960
DEDUPE_IOU = 0.45
MIN_BOX_PX = 4.0
# Reject flat junk (e.g. crease lines) — not large/bottom/plastic near wickets.
MIN_ASPECT_HEIGHT_OVER_WIDTH = 0.35
MAX_AREA_FRACTION = 0.55


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
    result = detect_wickets_robust(image, enable_roi=False)
    return {
        "success": result["success"],
        "status": result["status"],
        "message": result["message"],
        "candidates": [
            {
                "confidence": item["confidence"],
                "class_name": item["class_name"],
                "bbox": item["bbox"],
            }
            for item in result.get("candidates") or []
        ],
    }


# ponytail: sensible rear-camera defaults — far upper/central, near lower/central larger.
DEFAULT_STRIKER_GUIDE = {"x": 0.34, "y": 0.16, "width": 0.32, "height": 0.36}
DEFAULT_NON_STRIKER_GUIDE = {"x": 0.22, "y": 0.48, "width": 0.56, "height": 0.48}


def default_wicket_guides() -> dict[str, dict[str, float]]:
    return {
        "striker": dict(DEFAULT_STRIKER_GUIDE),
        "non_striker": dict(DEFAULT_NON_STRIKER_GUIDE),
    }


def detect_wickets_guided(
    image: Image.Image,
    guides: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Detect wickets inside user guide boxes with robust role corroboration.

    Role follows guide labels (striker guide → striker). Does not fabricate boxes.
    Reuses `_detect_end` (same crop path as Upload Image) and `detect_wickets_robust`.
    """
    if not STUMP_MODEL_PATH.is_file():
        return {
            "success": False,
            "status": "stump_detector_missing",
            "message": (
                "Stump detector model not found at "
                f"{STUMP_MODEL_RELATIVE_PATH}"
            ),
            "candidates": [],
            "selected": {"striker": None, "non_striker": None},
            "rois": dict(guides),
            "diagnostics": {
                "passes": [],
                "rejected": [],
                "selected": None,
                "pass_count": 0,
            },
        }

    try:
        model = _load_model(str(STUMP_MODEL_PATH))
    except Exception as exc:
        return {
            "success": False,
            "status": "stump_detector_error",
            "message": f"Stump detector could not be loaded: {type(exc).__name__}.",
            "candidates": [],
            "selected": {"striker": None, "non_striker": None},
            "rois": dict(guides),
            "diagnostics": {
                "passes": [],
                "rejected": [],
                "selected": None,
                "pass_count": 0,
            },
        }

    frame_width, frame_height = image.size
    diagnostics: dict[str, Any] = {
        "passes": [],
        "rejected": [],
        "selected": None,
        "pass_count": 0,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "mode": "guided",
    }
    selected: dict[str, dict[str, Any] | None] = {
        "striker": None,
        "non_striker": None,
    }
    candidates: list[dict[str, Any]] = []

    for end in ("striker", "non_striker"):
        guide = guides[end]
        detection = _detect_end(
            model,
            image,
            end,
            guide,
            frame_width,
            frame_height,
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        if detection.get("found") and detection.get("bbox"):
            candidate = {
                "confidence": float(detection["confidence"]),
                "class_name": str(detection.get("class_name") or "stump"),
                "bbox": {
                    "x": float(detection["bbox"]["x"]),
                    "y": float(detection["bbox"]["y"]),
                    "width": float(detection["bbox"]["width"]),
                    "height": float(detection["bbox"]["height"]),
                },
                "source": "guide_roi",
                "guide_end": end,
            }
            accepted.append(candidate)
            candidates.append(candidate)
            selected[end] = candidate
        else:
            rejected.append(
                {
                    "source": "guide_roi",
                    "reason": "no_stump_in_guide",
                    "guide_end": end,
                    "confidence": None,
                    "class_name": None,
                    "bbox": None,
                }
            )
        diagnostics["passes"].append(
            _pass_summary(f"guide_{end}", accepted, rejected, roi=guide)
        )
        diagnostics["rejected"].extend(rejected)

    # ponytail: guides define search regions; the existing robust far/near
    # assignment decides semantic roles when its candidate is inside that guide.
    # This avoids a higher-confidence net/post crop replacing the known near wicket.
    robust = detect_wickets_robust(image, enable_roi=True)
    robust_diagnostics = robust.get("diagnostics") or {}
    diagnostics["passes"].extend(robust_diagnostics.get("passes") or [])
    diagnostics["rejected"].extend(robust_diagnostics.get("rejected") or [])
    for item in robust.get("candidates") or []:
        candidates = _merge_candidates(candidates, [item])

    robust_selected = robust.get("selected") or {}
    for end in ("striker", "non_striker"):
        role_candidate = robust_selected.get(end)
        if role_candidate is not None and _bbox_center_in_guide(
            role_candidate["bbox"],
            guides[end],
            frame_width,
            frame_height,
        ):
            selected[end] = role_candidate

    missing = [end for end, item in selected.items() if item is None]
    if missing:

        used_boxes = [
            item["bbox"] for item in selected.values() if item is not None
        ]
        for end in missing:
            guide = guides[end]
            best = _best_candidate_in_guide(
                candidates,
                guide,
                frame_width,
                frame_height,
                exclude_boxes=used_boxes,
            )
            if best is None:
                # Last resort: robust role assignment for this end only.
                fallback = robust_selected.get(end)
                if fallback is not None and _bbox_center_in_guide(
                    fallback["bbox"], guide, frame_width, frame_height
                ):
                    best = fallback
                elif fallback is not None and end == "non_striker":
                    # ponytail: purple near-wicket often sits near frame bottom;
                    # allow robust near pick when guide was close but empty.
                    best = fallback
            if best is not None:
                selected[end] = best
                used_boxes.append(best["bbox"])

    diagnostics["pass_count"] = len(diagnostics["passes"])
    diagnostics["selected"] = {
        "striker": _candidate_debug(selected["striker"]),
        "non_striker": _candidate_debug(selected["non_striker"]),
    }
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    both = selected["striker"] is not None and selected["non_striker"] is not None
    return {
        "success": True,
        "status": "candidates_ready" if both else "detection_incomplete",
        "message": (
            "Guided stump detection found both wickets."
            if both
            else "Guided stump detection could not lock both wickets."
        ),
        "candidates": candidates,
        "selected": selected,
        "rois": dict(guides),
        "diagnostics": diagnostics,
    }


def _best_candidate_in_guide(
    candidates: list[dict[str, Any]],
    guide: dict[str, float],
    frame_width: int,
    frame_height: int,
    *,
    exclude_boxes: list[dict[str, float]],
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_conf = -1.0
    for candidate in candidates:
        bbox = candidate.get("bbox") or {}
        if any(_bbox_iou(bbox, other) >= DEDUPE_IOU for other in exclude_boxes):
            continue
        if not _bbox_center_in_guide(bbox, guide, frame_width, frame_height):
            continue
        confidence = float(candidate.get("confidence") or 0.0)
        if confidence > best_conf:
            best = candidate
            best_conf = confidence
    return best


def _bbox_center_in_guide(
    bbox: dict[str, float],
    guide: dict[str, float],
    frame_width: int,
    frame_height: int,
) -> bool:
    try:
        cx = (float(bbox["x"]) + float(bbox["width"]) / 2.0) / frame_width
        cy = (float(bbox["y"]) + float(bbox["height"]) / 2.0) / frame_height
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        guide["x"] <= cx <= guide["x"] + guide["width"]
        and guide["y"] <= cy <= guide["y"] + guide["height"]
    )


def build_scene_overlay_rgba(
    *,
    frame_width: int,
    frame_height: int,
    striker_bbox: dict[str, float],
    non_striker_bbox: dict[str, float],
) -> Image.Image:
    """Transparent Upload-Image-style scene overlay (virtual wickets + corridor).

    Shared by calibration JPEG preview and full-video scene compositing.
    """
    detections = {
        "striker": {
            "found": True,
            "confidence": 1.0,
            "bbox": _bbox_to_int(striker_bbox),
        },
        "non_striker": {
            "found": True,
            "confidence": 1.0,
            "bbox": _bbox_to_int(non_striker_bbox),
        },
    }
    virtual_stumps = {
        end: _virtual_stumps_from_bbox(
            detections[end]["bbox"],
            frame_width=frame_width,
            frame_height=frame_height,
        )
        for end in ("striker", "non_striker")
    }
    pitch_overlay = _build_pitch_overlay(
        detections,
        virtual_stumps,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    overlay = Image.new("RGBA", (frame_width, frame_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    _draw_upload_style_scene(
        draw,
        pitch_overlay=pitch_overlay,
        detections=detections,
        virtual_stumps=virtual_stumps,
        frame_width=frame_width,
        include_detection_boxes=True,
    )
    return overlay


def compose_scene_overlay_image(
    image: Image.Image,
    *,
    striker_bbox: dict[str, float],
    non_striker_bbox: dict[str, float],
) -> Image.Image:
    """Composite Upload-Image-style scene geometry onto a clean RGB frame."""
    width, height = image.size
    overlay = build_scene_overlay_rgba(
        frame_width=width,
        frame_height=height,
        striker_bbox=striker_bbox,
        non_striker_bbox=non_striker_bbox,
    )
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _bbox_to_int(bbox: dict[str, float]) -> dict[str, int]:
    return {
        "x": int(round(float(bbox["x"]))),
        "y": int(round(float(bbox["y"]))),
        "width": max(1, int(round(float(bbox["width"])))),
        "height": max(1, int(round(float(bbox["height"])))),
    }


def _draw_upload_style_scene(
    draw: ImageDraw.ImageDraw,
    *,
    pitch_overlay: dict[str, Any],
    detections: dict[str, dict[str, Any]],
    virtual_stumps: dict[str, Any],
    frame_width: int,
    include_detection_boxes: bool,
) -> None:
    """Thin perspective corridor + dashed centreline + virtual stumps (Upload style)."""
    stroke = max(2, round(frame_width / 640))
    stump_width = max(3, round(frame_width / 420))
    bail_width = max(3, round(frame_width / 380))

    corridor = pitch_overlay.get("pitch_corridor") or []
    if len(corridor) == 4:
        polygon = [(point["x"], point["y"]) for point in corridor]
        draw.polygon(
            polygon,
            fill=(226, 183, 72, 40),
            outline=(255, 225, 132, 200),
            width=stroke,
        )

    center_line = pitch_overlay.get("center_line") or []
    if len(center_line) == 2:
        _draw_dashed_line(
            draw,
            (
                center_line[0]["x"],
                center_line[0]["y"],
                center_line[1]["x"],
                center_line[1]["y"],
            ),
            fill=(255, 255, 255, 210),
            width=stroke,
            dash=max(8, round(frame_width / 80)),
            gap=max(6, round(frame_width / 100)),
        )

    for guide in (pitch_overlay.get("crease_guides") or {}).values():
        if len(guide) == 2:
            draw.line(
                (guide[0]["x"], guide[0]["y"], guide[1]["x"], guide[1]["y"]),
                fill=(255, 255, 255, 185),
                width=max(1, stroke - 1),
            )

    if include_detection_boxes:
        for end in ("striker", "non_striker"):
            detection = detections.get(end) or {}
            bbox = detection.get("bbox")
            if not detection.get("found") or not bbox:
                continue
            bx1, by1 = bbox["x"], bbox["y"]
            bx2 = bx1 + bbox["width"]
            by2 = by1 + bbox["height"]
            draw.rectangle(
                (bx1, by1, bx2, by2),
                outline=(183, 243, 75, 230),
                width=stroke,
            )
            label = f"{end} {(detection.get('confidence') or 0) * 100:.0f}%"
            draw.text((bx1 + 4, max(0, by1 - 14)), label, fill=(183, 243, 75, 255))

    for end in ("striker", "non_striker"):
        end_geometry = virtual_stumps.get(end) or {}
        for stump in end_geometry.get("stumps", []):
            top = stump["top"]
            base = stump["base"]
            draw.line(
                (top["x"], top["y"], base["x"], base["y"]),
                fill=(246, 207, 98, 255),
                width=stump_width,
            )
        for bail in end_geometry.get("bails", []):
            start = bail["start"]
            end_point = bail["end"]
            draw.line(
                (start["x"], start["y"], end_point["x"], end_point["y"]),
                fill=(255, 223, 126, 255),
                width=bail_width,
            )


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    x1, y1, x2, y2 = xy
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    drawn = 0.0
    while drawn < length:
        seg_end = min(length, drawn + dash)
        sx = round(x1 + ux * drawn)
        sy = round(y1 + uy * drawn)
        ex = round(x1 + ux * seg_end)
        ey = round(y1 + uy * seg_end)
        draw.line((sx, sy, ex, ey), fill=fill, width=width)
        drawn = seg_end + gap


def detect_wickets_robust(
    image: Image.Image,
    *,
    enable_roi: bool = True,
) -> dict[str, Any]:
    """Multi-pass stump detection: full frame, then far/near ROI if needed.

    Shared service for Video Analysis auto-cal (and future Live Session reuse).
    Does not invent wickets — only returns detector boxes with pass metadata.
    """
    if not STUMP_MODEL_PATH.is_file():
        return {
            "success": False,
            "status": "stump_detector_missing",
            "message": (
                "Stump detector model not found at "
                f"{STUMP_MODEL_RELATIVE_PATH}"
            ),
            "candidates": [],
            "selected": {"striker": None, "non_striker": None},
            "rois": dict(ROI_LAYOUTS),
            "diagnostics": {
                "passes": [],
                "rejected": [],
                "selected": None,
                "pass_count": 0,
            },
        }

    try:
        model = _load_model(str(STUMP_MODEL_PATH))
    except Exception as exc:
        return {
            "success": False,
            "status": "stump_detector_error",
            "message": f"Stump detector could not be loaded: {type(exc).__name__}.",
            "candidates": [],
            "selected": {"striker": None, "non_striker": None},
            "rois": dict(ROI_LAYOUTS),
            "diagnostics": {
                "passes": [],
                "rejected": [],
                "selected": None,
                "pass_count": 0,
            },
        }

    frame_width, frame_height = image.size
    diagnostics: dict[str, Any] = {
        "passes": [],
        "rejected": [],
        "selected": None,
        "pass_count": 0,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }
    merged: list[dict[str, Any]] = []

    full_raw, full_rejected = _run_detection_pass(
        model,
        image,
        source="full_frame",
        offset_x=0.0,
        offset_y=0.0,
        frame_width=frame_width,
        frame_height=frame_height,
        conf=DEFAULT_DETECT_CONF,
        imgsz=None,
    )
    diagnostics["passes"].append(
        _pass_summary("full_frame", full_raw, full_rejected)
    )
    diagnostics["rejected"].extend(full_rejected)
    merged = _merge_candidates(merged, full_raw)

    need_roi = enable_roi and not _has_two_plausible_ends(
        merged, frame_width, frame_height
    )
    if need_roi:
        for source, roi in ROI_LAYOUTS.items():
            crop_bounds = _normalized_box_to_pixels(
                roi, frame_width, frame_height
            )
            crop = image.crop(crop_bounds)
            raw, rejected = _run_detection_pass(
                model,
                crop,
                source=source,
                offset_x=float(crop_bounds[0]),
                offset_y=float(crop_bounds[1]),
                frame_width=frame_width,
                frame_height=frame_height,
                conf=ROI_DETECT_CONF,
                imgsz=None,
            )
            diagnostics["passes"].append(
                _pass_summary(source, raw, rejected, roi=roi)
            )
            diagnostics["rejected"].extend(rejected)
            merged = _merge_candidates(merged, raw)

    # Optional higher-res on the missing end only (max two extra predicts).
    if enable_roi and not _has_two_plausible_ends(
        merged, frame_width, frame_height
    ):
        missing = _missing_end_sources(merged, frame_width, frame_height)
        for source in missing:
            roi = ROI_LAYOUTS[source]
            crop_bounds = _normalized_box_to_pixels(
                roi, frame_width, frame_height
            )
            crop = image.crop(crop_bounds)
            pass_name = f"{source}_imgsz{ROI_HIGHRES_IMGSZ}"
            raw, rejected = _run_detection_pass(
                model,
                crop,
                source=source,
                offset_x=float(crop_bounds[0]),
                offset_y=float(crop_bounds[1]),
                frame_width=frame_width,
                frame_height=frame_height,
                conf=ROI_DETECT_CONF,
                imgsz=ROI_HIGHRES_IMGSZ,
            )
            diagnostics["passes"].append(
                _pass_summary(pass_name, raw, rejected, roi=roi)
            )
            diagnostics["rejected"].extend(rejected)
            merged = _merge_candidates(merged, raw)

    diagnostics["pass_count"] = len(diagnostics["passes"])
    selected = _select_far_near_ends(merged, frame_width, frame_height)
    diagnostics["selected"] = {
        "striker": _candidate_debug(selected["striker"]),
        "non_striker": _candidate_debug(selected["non_striker"]),
    }
    merged.sort(key=lambda item: item["confidence"], reverse=True)
    both = selected["striker"] is not None and selected["non_striker"] is not None
    return {
        "success": True,
        "status": "candidates_ready" if both else "detection_incomplete",
        "message": (
            "Robust stump detection found both wickets."
            if both
            else "Robust stump detection could not lock both wickets."
        ),
        "candidates": merged,
        "selected": selected,
        "rois": dict(ROI_LAYOUTS),
        "diagnostics": diagnostics,
    }


def save_robust_detection_overlay(
    image: Image.Image,
    *,
    candidates: list[dict[str, Any]],
    selected: dict[str, dict[str, Any] | None],
    rois: dict[str, dict[str, float]],
    output_path: Path,
) -> Path:
    """Developer debug overlay: raw boxes, ROI bounds, selected ends."""
    debug = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(debug, "RGBA")
    width, height = debug.size

    for source, roi in rois.items():
        x1, y1, x2, y2 = _normalized_box_to_pixels(roi, width, height)
        color = (255, 170, 40, 220) if source == "far_roi" else (120, 200, 255, 220)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), source, fill=color)

    for candidate in candidates:
        bbox = candidate["bbox"]
        x1 = round(bbox["x"])
        y1 = round(bbox["y"])
        x2 = round(bbox["x"] + bbox["width"])
        y2 = round(bbox["y"] + bbox["height"])
        draw.rectangle((x1, y1, x2, y2), outline=(220, 220, 220, 200), width=2)
        draw.text(
            (x1 + 3, max(0, y1 - 14)),
            f"{candidate.get('source', '?')} {candidate['confidence']:.2f}",
            fill=(230, 230, 230, 230),
        )

    for end, color, label in (
        ("striker", (255, 190, 70, 255), "SELECTED STRIKER"),
        ("non_striker", (80, 220, 255, 255), "SELECTED NON-STRIKER"),
    ):
        item = selected.get(end)
        if not item:
            continue
        bbox = item["bbox"]
        x1 = round(bbox["x"])
        y1 = round(bbox["y"])
        x2 = round(bbox["x"] + bbox["width"])
        y2 = round(bbox["y"] + bbox["height"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1 + 4, max(0, y1 - 16)), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug.convert("RGB").save(output_path, format="JPEG", quality=92)
    return output_path


def _run_detection_pass(
    model,
    image: Image.Image,
    *,
    source: str,
    offset_x: float,
    offset_y: float,
    frame_width: int,
    frame_height: int,
    conf: float,
    imgsz: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predict_kwargs: dict[str, Any] = {
        "source": image,
        "verbose": False,
        "conf": conf,
    }
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    try:
        results = model.predict(**predict_kwargs)
    except Exception:
        return [], [
            {
                "source": source,
                "reason": "inference_error",
                "confidence": None,
                "class_name": None,
                "bbox": None,
            }
        ]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
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
                classes, confidences, coordinates
            ):
                class_name = _class_name(names, int(class_id))
                if not _is_stump_class(class_name, names):
                    rejected.append(
                        {
                            "source": source,
                            "reason": f"class_filtered:{class_name}",
                            "confidence": round(float(confidence), 4),
                            "class_name": class_name,
                            "bbox": None,
                        }
                    )
                    continue
                x1, y1, x2, y2 = (float(value) for value in xyxy[:4])
                full_x1 = max(0.0, min(float(frame_width), offset_x + x1))
                full_y1 = max(0.0, min(float(frame_height), offset_y + y1))
                full_x2 = max(full_x1, min(float(frame_width), offset_x + x2))
                full_y2 = max(full_y1, min(float(frame_height), offset_y + y2))
                width = full_x2 - full_x1
                height = full_y2 - full_y1
                bbox = {
                    "x": full_x1,
                    "y": full_y1,
                    "width": width,
                    "height": height,
                }
                reason = _reject_reason(
                    bbox, frame_width=frame_width, frame_height=frame_height
                )
                if reason is not None:
                    rejected.append(
                        {
                            "source": source,
                            "reason": reason,
                            "confidence": round(float(confidence), 4),
                            "class_name": class_name,
                            "bbox": {
                                "x": round(full_x1, 2),
                                "y": round(full_y1, 2),
                                "width": round(width, 2),
                                "height": round(height, 2),
                            },
                        }
                    )
                    continue
                accepted.append(
                    {
                        "confidence": round(float(confidence), 4),
                        "class_name": class_name,
                        "source": source,
                        "bbox": bbox,
                    }
                )
    except Exception:
        return [], [
            {
                "source": source,
                "reason": "result_parse_error",
                "confidence": None,
                "class_name": None,
                "bbox": None,
            }
        ]
    accepted.sort(key=lambda item: item["confidence"], reverse=True)
    return accepted, rejected


def _reject_reason(
    bbox: dict[str, float],
    *,
    frame_width: int,
    frame_height: int,
) -> str | None:
    width = float(bbox["width"])
    height = float(bbox["height"])
    if width < MIN_BOX_PX or height < MIN_BOX_PX:
        return "too_small_px"
    if height / max(width, 1e-6) < MIN_ASPECT_HEIGHT_OVER_WIDTH:
        return "aspect_too_flat"
    area_fraction = (width * height) / max(frame_width * frame_height, 1)
    if area_fraction > MAX_AREA_FRACTION:
        return "area_too_large"
    # Explicit non-rejects for near purple / edge cases (documented in diagnostics).
    return None


def _merge_candidates(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing)
    for candidate in incoming:
        duplicate_index = None
        for index, other in enumerate(merged):
            if _bbox_iou(candidate["bbox"], other["bbox"]) < DEDUPE_IOU:
                continue
            duplicate_index = index
            break
        if duplicate_index is None:
            merged.append(candidate)
            continue
        # Keep higher confidence; prefer keeping original source metadata when tied.
        current = merged[duplicate_index]
        if candidate["confidence"] > current["confidence"]:
            merged[duplicate_index] = candidate
    return merged


def _has_two_plausible_ends(
    candidates: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> bool:
    selected = _select_far_near_ends(candidates, frame_width, frame_height)
    return (
        selected["striker"] is not None and selected["non_striker"] is not None
    )


def _missing_end_sources(
    candidates: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> list[str]:
    selected = _select_far_near_ends(candidates, frame_width, frame_height)
    missing: list[str] = []
    if selected["striker"] is None:
        missing.append("far_roi")
    if selected["non_striker"] is None:
        missing.append("near_roi")
    return missing


def _select_far_near_ends(
    candidates: list[dict[str, Any]],
    frame_width: int,
    frame_height: int,
) -> dict[str, dict[str, Any] | None]:
    """Pick FAR=higher/smaller and NEAR=lower/larger by explicit scores."""
    if len(candidates) < 2:
        only = candidates[0] if candidates else None
        if only is None:
            return {"striker": None, "non_striker": None}
        if _near_affinity(only, frame_width, frame_height) >= 0.55:
            return {"striker": None, "non_striker": only}
        return {"striker": only, "non_striker": None}

    best_pair: tuple[dict[str, Any], dict[str, Any]] | None = None
    best_score = -1e9
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if _bbox_iou(first["bbox"], second["bbox"]) > DEDUPE_IOU:
                continue
            near, far = _order_near_far(
                first, second, frame_width, frame_height
            )
            separation = _center_separation_norm(
                near, far, frame_width, frame_height
            )
            if separation < 0.04:
                continue
            near_aff = _near_affinity(near, frame_width, frame_height)
            far_aff = _far_affinity(far, frame_width, frame_height)
            # Require complementary roles — avoid two near-ish or two far-ish.
            if near_aff < 0.35 or far_aff < 0.25:
                continue
            score = (
                near_aff
                + far_aff
                + min(separation, 0.55)
                + 0.35 * (near["confidence"] + far["confidence"])
            )
            if score > best_score:
                best_pair = (far, near)
                best_score = score

    if best_pair is None:
        # Fallback: strongest vertical+size contrast pair without inventing boxes.
        ranked = sorted(
            candidates,
            key=lambda item: _near_affinity(item, frame_width, frame_height),
        )
        far = ranked[0]
        near = ranked[-1]
        if (
            far is near
            or _bbox_iou(far["bbox"], near["bbox"]) > DEDUPE_IOU
            or _center_separation_norm(near, far, frame_width, frame_height)
            < 0.04
        ):
            return {"striker": None, "non_striker": None}
        return {"striker": far, "non_striker": near}

    far, near = best_pair
    return {"striker": far, "non_striker": near}


def _order_near_far(
    first: dict[str, Any],
    second: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if _near_affinity(first, frame_width, frame_height) >= _near_affinity(
        second, frame_width, frame_height
    ):
        return first, second
    return second, first


def _near_affinity(
    candidate: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> float:
    bbox = candidate["bbox"]
    bottom_y = (bbox["y"] + bbox["height"]) / max(frame_height, 1)
    area = (bbox["width"] * bbox["height"]) / max(
        frame_width * frame_height, 1
    )
    # Lower + larger ≈ non-striker / near end for rear-camera views.
    return (
        0.55 * _clamp01(bottom_y)
        + 0.30 * _clamp01(area / 0.12)
        + 0.15 * _clamp01(candidate["confidence"])
    )


def _far_affinity(
    candidate: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> float:
    bbox = candidate["bbox"]
    center_y = (bbox["y"] + bbox["height"] / 2) / max(frame_height, 1)
    area = (bbox["width"] * bbox["height"]) / max(
        frame_width * frame_height, 1
    )
    # Higher + smaller ≈ striker / far end.
    return (
        0.55 * _clamp01(1.0 - center_y)
        + 0.30 * _clamp01(1.0 - area / 0.08)
        + 0.15 * _clamp01(candidate["confidence"])
    )


def _center_separation_norm(
    first: dict[str, Any],
    second: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> float:
    fx = first["bbox"]["x"] + first["bbox"]["width"] / 2
    fy = first["bbox"]["y"] + first["bbox"]["height"]
    sx = second["bbox"]["x"] + second["bbox"]["width"] / 2
    sy = second["bbox"]["y"] + second["bbox"]["height"]
    dx = (fx - sx) / max(frame_width, 1)
    dy = (fy - sy) / max(frame_height, 1)
    return (dx * dx + dy * dy) ** 0.5


def _bbox_iou(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(
        first["x"] + first["width"],
        second["x"] + second["width"],
    )
    bottom = min(
        first["y"] + first["height"],
        second["y"] + second["height"],
    )
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = first["width"] * first["height"]
    second_area = second["width"] * second["height"]
    return intersection / max(first_area + second_area - intersection, 1e-6)


def _pass_summary(
    name: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    roi: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "raw_count": len(accepted) + len(rejected),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "roi": roi,
        "accepted": [_candidate_debug(item) for item in accepted],
        "rejected_reasons": [item.get("reason") for item in rejected],
    }


def _candidate_debug(candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    bbox = candidate["bbox"]
    return {
        "confidence": candidate.get("confidence"),
        "class_name": candidate.get("class_name"),
        "source": candidate.get("source"),
        "bbox": {
            "x": round(float(bbox["x"]), 2),
            "y": round(float(bbox["y"]), 2),
            "width": round(float(bbox["width"]), 2),
            "height": round(float(bbox["height"]), 2),
        },
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
