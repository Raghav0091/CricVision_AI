"""Persistent scene calibration for prepared Video Analysis reference frames."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ..schemas.video_analysis import (
    ConfirmedVideoCalibrationResponse,
    NormalizedBox,
    NormalizedPoint,
    PitchGeometry,
    VideoCalibrationConfirmationRequest,
    VideoCalibrationDetectionResponse,
    WicketCalibration,
    WicketCalibrationInput,
    WicketCandidate,
)
from .stump_detector_service import (
    STUMP_MODEL_PATH,
    STUMP_MODEL_RELATIVE_PATH,
    detect_stump_candidates,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)


CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_OVERLAY_FILENAME = "calibration_overlay.jpg"
WICKET_PROXIMITY_WARNING = (
    "The two wicket locations appear too close together. Check the calibration."
)


def detect_video_calibration(
    analysis_id: str,
) -> VideoCalibrationDetectionResponse:
    analysis = load_video_analysis(analysis_id)
    reference_path = _reference_path(analysis_id)
    image = _open_reference_image(reference_path)
    image_width, image_height = image.size
    detection_result = detect_stump_candidates(image)
    raw_candidates = detection_result.get("candidates") or []
    candidates: list[WicketCandidate] = []
    for index, candidate in enumerate(raw_candidates, start=1):
        try:
            candidates.append(
                _candidate_from_detection(
                    candidate,
                    index=index,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    striker: WicketCalibration | None = None
    non_striker: WicketCalibration | None = None
    if len(candidates) >= 2:
        pair = _choose_provisional_pair(candidates)
        if pair is not None:
            striker, non_striker = _assign_provisional_ends(*pair)
    elif len(candidates) == 1:
        only = candidates[0]
        if _near_score(only) >= 0.65:
            non_striker = _calibration_from_candidate(only, "non_striker")
        else:
            striker = _calibration_from_candidate(only, "striker")

    pitch_geometry = (
        calculate_pitch_geometry(striker, non_striker, 1.0)
        if striker is not None and non_striker is not None
        else None
    )
    warning = (
        _wicket_proximity_warning(striker, non_striker)
        if striker is not None and non_striker is not None
        else None
    )

    if not detection_result["success"]:
        status = detection_result["status"]
        message = detection_result["message"]
        success = False
    elif not candidates:
        status = "manual_required"
        message = "No wicket detections found. Place both wicket boxes manually."
        success = True
    elif striker is None or non_striker is None:
        status = "manual_required"
        message = (
            "Only one usable wicket location was found. "
            "Place the missing wicket box manually."
        )
        success = True
    else:
        status = "candidates_ready"
        message = "Review and adjust both wicket locations before confirming."
        success = True

    return VideoCalibrationDetectionResponse(
        success=success,
        status=status,
        analysis_id=analysis_id,
        reference_frame_url=analysis.reference_frame_url,
        image_width=image_width,
        image_height=image_height,
        candidates=candidates,
        provisional_striker_wicket=striker,
        provisional_non_striker_wicket=non_striker,
        pitch_geometry=pitch_geometry,
        model_path_used=STUMP_MODEL_RELATIVE_PATH,
        warning=warning,
        message=message,
    )


def confirm_video_calibration(
    analysis_id: str,
    request: VideoCalibrationConfirmationRequest,
) -> ConfirmedVideoCalibrationResponse:
    if request.analysis_id != analysis_id:
        raise VideoAnalysisServiceError(
            "Calibration analysis ID does not match the URL.",
            status_code=400,
        )

    analysis = load_video_analysis(analysis_id)
    reference_path = _reference_path(analysis_id)
    image = _open_reference_image(reference_path)
    image_width, image_height = image.size
    striker = _calibration_from_input(request.striker_wicket, "striker")
    non_striker = _calibration_from_input(
        request.non_striker_wicket,
        "non_striker",
    )
    warning = _wicket_proximity_warning(striker, non_striker)
    if warning:
        raise VideoAnalysisServiceError(warning, status_code=422)

    pitch_geometry = calculate_pitch_geometry(
        striker,
        non_striker,
        request.corridor_width_multiplier,
    )
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    calibration_dir = analysis_dir / "calibration"
    calibration_path = calibration_dir / CALIBRATION_FILENAME
    overlay_path = calibration_dir / CALIBRATION_OVERLAY_FILENAME
    now = datetime.now(timezone.utc)
    created_at = _existing_created_at(calibration_path) or now
    calibration_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{CALIBRATION_FILENAME}"
    )
    overlay_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{CALIBRATION_OVERLAY_FILENAME}"
    )
    model_path_used = (
        STUMP_MODEL_RELATIVE_PATH
        if STUMP_MODEL_PATH.is_file()
        and (
            striker.source != "manual"
            or non_striker.source != "manual"
        )
        else None
    )
    confirmed = ConfirmedVideoCalibrationResponse(
        success=True,
        status="calibrated",
        analysis_id=analysis_id,
        created_at=created_at,
        updated_at=now,
        reference_frame_index=analysis.reference_frame_index,
        reference_frame_url=analysis.reference_frame_url,
        calibration_url=calibration_url,
        calibration_overlay_url=overlay_url,
        image_width=image_width,
        image_height=image_height,
        model_path_used=model_path_used,
        striker_wicket=striker,
        non_striker_wicket=non_striker,
        pitch_geometry=pitch_geometry,
        user_note=request.user_note,
        message="Scene calibration confirmed.",
    )

    _save_calibration_overlay(image, confirmed, overlay_path)
    _write_json(calibration_path, confirmed.model_dump(mode="json"))
    _update_analysis_metadata(
        analysis_dir,
        updated_at=now,
        calibration_url=calibration_url,
        overlay_url=overlay_url,
    )
    return confirmed


def load_video_calibration(
    analysis_id: str,
) -> ConfirmedVideoCalibrationResponse:
    load_video_analysis(analysis_id)
    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    calibration_path = analysis_dir / "calibration" / CALIBRATION_FILENAME
    if not calibration_path.is_file():
        raise VideoAnalysisServiceError(
            "Scene calibration has not been confirmed.",
            status_code=404,
        )
    try:
        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        confirmed = ConfirmedVideoCalibrationResponse.model_validate(data)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored scene calibration is unavailable.",
            status_code=500,
        ) from exc

    if confirmed.analysis_id != analysis_id:
        raise VideoAnalysisServiceError(
            "Stored scene calibration is invalid.",
            status_code=500,
        )
    overlay_path = (
        analysis_dir / "calibration" / CALIBRATION_OVERLAY_FILENAME
    )
    if not _reference_path(analysis_id).is_file() or not overlay_path.is_file():
        raise VideoAnalysisServiceError(
            "Stored scene calibration files are missing.",
            status_code=404,
        )
    return confirmed


def calculate_pitch_geometry(
    striker: WicketCalibration,
    non_striker: WicketCalibration,
    corridor_width_multiplier: float,
) -> PitchGeometry:
    ends = {
        "striker": striker,
        "non_striker": non_striker,
    }
    near_label = max(ends, key=lambda label: _near_score(ends[label]))
    far_label = "non_striker" if near_label == "striker" else "striker"
    near = ends[near_label]
    far = ends[far_label]
    dx = near.bottom_center.x - far.bottom_center.x
    dy = near.bottom_center.y - far.bottom_center.y
    length = math.hypot(dx, dy)
    if length < 0.000001:
        raise VideoAnalysisServiceError(
            WICKET_PROXIMITY_WARNING,
            status_code=422,
        )
    perpendicular_x = -dy / length
    perpendicular_y = dx / length

    raw_near_half_width = max(near.box.width * 1.35, 0.02)
    raw_far_half_width = max(far.box.width * 1.35, 0.014)
    near_half_width = max(
        raw_near_half_width,
        raw_far_half_width * 1.15,
    ) * corridor_width_multiplier
    far_half_width = min(
        raw_far_half_width * corridor_width_multiplier,
        near_half_width * 0.86,
    )
    corridor = [
        _offset_point(
            near.bottom_center,
            perpendicular_x,
            perpendicular_y,
            near_half_width,
        ),
        _offset_point(
            far.bottom_center,
            perpendicular_x,
            perpendicular_y,
            far_half_width,
        ),
        _offset_point(
            far.bottom_center,
            perpendicular_x,
            perpendicular_y,
            -far_half_width,
        ),
        _offset_point(
            near.bottom_center,
            perpendicular_x,
            perpendicular_y,
            -near_half_width,
        ),
    ]
    return PitchGeometry(
        axis_start=striker.bottom_center,
        axis_end=non_striker.bottom_center,
        corridor=corridor,
        near_end_label=near_label,
        far_end_label=far_label,
        geometry_type="approximate_2d",
        corridor_width_multiplier=corridor_width_multiplier,
    )


def _reference_path(analysis_id: str) -> Path:
    return (
        VIDEO_ANALYSIS_ROOT
        / analysis_id
        / "calibration"
        / "reference_frame.jpg"
    )


def _open_reference_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise VideoAnalysisServiceError(
            "Calibration reference frame is missing.",
            status_code=404,
        )
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.load()
            return image
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Calibration reference frame could not be opened.",
            status_code=500,
        ) from exc


def _candidate_from_detection(
    candidate: dict[str, Any],
    *,
    index: int,
    image_width: int,
    image_height: int,
) -> WicketCandidate:
    bbox = candidate["bbox"]
    x = _clamp(float(bbox["x"]) / image_width)
    y = _clamp(float(bbox["y"]) / image_height)
    right = _clamp(
        (float(bbox["x"]) + float(bbox["width"])) / image_width
    )
    bottom = _clamp(
        (float(bbox["y"]) + float(bbox["height"])) / image_height
    )
    box = NormalizedBox(
        x=x,
        y=y,
        width=max(0.000001, right - x),
        height=max(0.000001, bottom - y),
    )
    center, bottom_center = _box_points(box)
    return WicketCandidate(
        candidate_id=f"wicket_{index}",
        confidence=float(candidate["confidence"]),
        class_name=str(candidate["class_name"]),
        box=box,
        center=center,
        bottom_center=bottom_center,
    )


def _choose_provisional_pair(
    candidates: list[WicketCandidate],
) -> tuple[WicketCandidate, WicketCandidate] | None:
    best_pair: tuple[WicketCandidate, WicketCandidate] | None = None
    best_score = -1.0
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if _intersection_over_union(first.box, second.box) > 0.45:
                continue
            separation = math.hypot(
                first.bottom_center.x - second.bottom_center.x,
                first.bottom_center.y - second.bottom_center.y,
            )
            if separation < 0.04:
                continue
            score = (
                first.confidence
                + second.confidence
                + min(separation, 0.5)
            )
            if score > best_score:
                best_pair = (first, second)
                best_score = score
    return best_pair


def _assign_provisional_ends(
    first: WicketCandidate,
    second: WicketCandidate,
) -> tuple[WicketCalibration, WicketCalibration]:
    near, far = (
        (first, second)
        if _near_score(first) >= _near_score(second)
        else (second, first)
    )
    # ponytail: the Video Analysis camera convention follows the existing
    # behind-non-striker setup; labels remain explicitly swappable in the UI.
    return (
        _calibration_from_candidate(far, "striker"),
        _calibration_from_candidate(near, "non_striker"),
    )


def _calibration_from_candidate(
    candidate: WicketCandidate,
    label: str,
) -> WicketCalibration:
    return WicketCalibration(
        label=label,
        source="detected",
        confidence=candidate.confidence,
        box=candidate.box,
        center=candidate.center,
        bottom_center=candidate.bottom_center,
    )


def _calibration_from_input(
    wicket: WicketCalibrationInput,
    expected_label: str,
) -> WicketCalibration:
    if wicket.label != expected_label:
        raise VideoAnalysisServiceError(
            f"Expected the {expected_label.replace('_', '-')} wicket.",
            status_code=422,
        )
    center, bottom_center = _box_points(wicket.box)
    return WicketCalibration(
        label=expected_label,
        source=wicket.source,
        confidence=wicket.confidence,
        box=wicket.box,
        center=center,
        bottom_center=bottom_center,
    )


def _box_points(
    box: NormalizedBox,
) -> tuple[NormalizedPoint, NormalizedPoint]:
    center_x = _clamp(box.x + box.width / 2)
    return (
        NormalizedPoint(
            x=center_x,
            y=_clamp(box.y + box.height / 2),
        ),
        NormalizedPoint(
            x=center_x,
            y=_clamp(box.y + box.height),
        ),
    )


def _near_score(wicket: WicketCandidate | WicketCalibration) -> float:
    return (
        wicket.bottom_center.y
        + wicket.box.width * 0.35
        + wicket.box.height * 0.1
    )


def _wicket_proximity_warning(
    striker: WicketCalibration,
    non_striker: WicketCalibration,
) -> str | None:
    distance = math.hypot(
        striker.bottom_center.x - non_striker.bottom_center.x,
        striker.bottom_center.y - non_striker.bottom_center.y,
    )
    if (
        distance < 0.055
        or _intersection_over_union(striker.box, non_striker.box) > 0.3
    ):
        return WICKET_PROXIMITY_WARNING
    return None


def _intersection_over_union(
    first: NormalizedBox,
    second: NormalizedBox,
) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = first.width * first.height
    second_area = second.width * second.height
    return intersection / max(first_area + second_area - intersection, 0.000001)


def _offset_point(
    point: NormalizedPoint,
    perpendicular_x: float,
    perpendicular_y: float,
    distance: float,
) -> NormalizedPoint:
    return NormalizedPoint(
        x=_clamp(point.x + perpendicular_x * distance),
        y=_clamp(point.y + perpendicular_y * distance),
    )


def _save_calibration_overlay(
    image: Image.Image,
    calibration: ConfirmedVideoCalibrationResponse,
    output_path: Path,
) -> None:
    try:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        width, height = image.size

        corridor = [
            (round(point.x * width), round(point.y * height))
            for point in calibration.pitch_geometry.corridor
        ]
        draw.polygon(
            corridor,
            fill=(213, 255, 107, 42),
            outline=(213, 255, 107, 220),
        )
        axis_start = calibration.pitch_geometry.axis_start
        axis_end = calibration.pitch_geometry.axis_end
        draw.line(
            (
                round(axis_start.x * width),
                round(axis_start.y * height),
                round(axis_end.x * width),
                round(axis_end.y * height),
            ),
            fill=(255, 255, 255, 235),
            width=max(2, round(width / 640)),
        )

        for wicket, color, label in (
            (calibration.striker_wicket, (255, 190, 70, 255), "Striker Wicket"),
            (
                calibration.non_striker_wicket,
                (80, 220, 255, 255),
                "Non-Striker Wicket",
            ),
        ):
            x1 = round(wicket.box.x * width)
            y1 = round(wicket.box.y * height)
            x2 = round((wicket.box.x + wicket.box.width) * width)
            y2 = round((wicket.box.y + wicket.box.height) * height)
            draw.rectangle(
                (x1, y1, x2, y2),
                outline=color,
                width=max(3, round(width / 480)),
            )
            label_top = max(0, y1 - 20)
            draw.rectangle(
                (x1, label_top, min(width, x1 + 145), y1),
                fill=(8, 12, 16, 220),
            )
            draw.text((x1 + 4, label_top + 3), label, fill=color)
            point_x = round(wicket.bottom_center.x * width)
            point_y = round(wicket.bottom_center.y * height)
            radius = max(4, round(width / 320))
            draw.ellipse(
                (
                    point_x - radius,
                    point_y - radius,
                    point_x + radius,
                    point_y + radius,
                ),
                fill=color,
                outline=(0, 0, 0, 255),
            )

        draw.rectangle((12, 12, 205, 36), fill=(8, 12, 16, 210))
        draw.text(
            (18, 18),
            "Approximate 2D calibration",
            fill=(255, 255, 255, 255),
        )
        composed = Image.alpha_composite(image.convert("RGBA"), overlay)
        composed.convert("RGB").save(output_path, format="JPEG", quality=92)
    except Exception as exc:
        raise VideoAnalysisServiceError(
            f"Calibration overlay could not be saved: {type(exc).__name__}.",
            status_code=500,
        ) from exc


def _existing_created_at(path: Path) -> datetime | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("created_at")
        return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        return None


def _update_analysis_metadata(
    analysis_dir: Path,
    *,
    updated_at: datetime,
    calibration_url: str,
    overlay_url: str,
) -> None:
    metadata_path = analysis_dir / "reports" / "analysis_metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoAnalysisServiceError(
            "Analysis metadata could not be updated.",
            status_code=500,
        ) from exc
    metadata.update(
        {
            "status": "calibrated",
            "calibration_status": "confirmed",
            "calibration_url": calibration_url,
            "calibration_overlay_url": overlay_url,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
        }
    )
    _write_json(metadata_path, metadata)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise VideoAnalysisServiceError(
            f"{path.name} could not be saved.",
            status_code=500,
        ) from exc


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
