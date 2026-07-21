"""Persistent scene calibration for prepared Video Analysis reference frames."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from ..schemas.video_analysis import (
    ConfirmedVideoCalibrationResponse,
    NormalizedBox,
    NormalizedPoint,
    PitchGeometry,
    VideoCalibrationConfirmationRequest,
    VideoCalibrationDetectionResponse,
    VisualCalibrationDetectionDebug,
    VisualCalibrationQuality,
    WicketCalibration,
    WicketCalibrationInput,
    WicketCandidate,
)
from .ball_detection_clip import transcode_browser_mp4
from .stump_detector_service import (
    STUMP_MODEL_PATH,
    STUMP_MODEL_RELATIVE_PATH,
    build_scene_overlay_rgba,
    compose_scene_overlay_image,
    default_wicket_guides,
    detect_wickets_guided,
    detect_wickets_robust,
    save_robust_detection_overlay,
)
from .video_analysis_service import (
    EARLY_REFERENCE_WINDOW,
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
    replace_reference_frame,
    _reference_frame_quality,
)


CALIBRATION_FILENAME = "calibration.json"
CALIBRATION_OVERLAY_FILENAME = "calibration_overlay.jpg"
SCENE_OVERLAY_FILENAME = "scene_overlay.mp4"
DETECTION_DEBUG_JSON = "wicket_detection_debug.json"
DETECTION_DEBUG_OVERLAY = "wicket_detection_debug.jpg"
EARLY_FALLBACK_ATTEMPTS = 3
WICKET_PROXIMITY_WARNING = (
    "The two wicket locations appear too close together. Check the calibration."
)


def detect_video_calibration(
    analysis_id: str,
    *,
    refresh_early_reference: bool = False,
    striker_guide: NormalizedBox | None = None,
    non_striker_guide: NormalizedBox | None = None,
) -> VideoCalibrationDetectionResponse:
    analysis = load_video_analysis(analysis_id)
    if refresh_early_reference:
        _try_refresh_early_reference(analysis_id, analysis)
        analysis = load_video_analysis(analysis_id)

    guides = _resolve_guides(striker_guide, non_striker_guide)
    response = _detect_on_current_reference(analysis_id, analysis, guides=guides)
    if (
        response.provisional_striker_wicket is not None
        and response.provisional_non_striker_wicket is not None
    ):
        return response

    # Guided redetect keeps the same frame; only unguided legacy path falls back.
    if striker_guide is None and non_striker_guide is None:
        if _try_early_frame_fallback(analysis_id, analysis):
            analysis = load_video_analysis(analysis_id)
            return _detect_on_current_reference(
                analysis_id, analysis, guides=guides
            )
    return response


def _detect_on_current_reference(
    analysis_id: str,
    analysis: Any,
    *,
    guides: dict[str, dict[str, float]],
) -> VideoCalibrationDetectionResponse:
    reference_path = _reference_path(analysis_id)
    image = _open_reference_image(reference_path)
    image_width, image_height = image.size
    detection_result = detect_wickets_guided(image, guides)
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
    assignment_warning: str | None = None
    selected = detection_result.get("selected") or {}
    selected_striker = selected.get("striker")
    selected_non_striker = selected.get("non_striker")
    if selected_striker is not None:
        striker_candidate = _match_selected_candidate(
            candidates, selected_striker, image_width, image_height
        )
        if striker_candidate is not None:
            striker = _calibration_from_candidate(striker_candidate, "striker")
    if selected_non_striker is not None:
        non_striker_candidate = _match_selected_candidate(
            candidates, selected_non_striker, image_width, image_height
        )
        if non_striker_candidate is not None:
            non_striker = _calibration_from_candidate(
                non_striker_candidate, "non_striker"
            )
    if striker is not None and non_striker is not None:
        assignment_warning = _assignment_uncertainty_warning(striker, non_striker)
    elif len(candidates) == 1 and striker is None and non_striker is None:
        only = candidates[0]
        if _near_score(only) >= 0.65:
            non_striker = _calibration_from_candidate(only, "non_striker")
        else:
            striker = _calibration_from_candidate(only, "striker")

    failed_ends: list[str] = []
    if striker is None:
        failed_ends.append("striker")
    if non_striker is None:
        failed_ends.append("non_striker")

    pitch_geometry = (
        calculate_pitch_geometry(striker, non_striker, 1.0)
        if striker is not None and non_striker is not None
        else None
    )
    proximity_warning = (
        _wicket_proximity_warning(striker, non_striker)
        if striker is not None and non_striker is not None
        else None
    )
    quality, quality_reasons = assess_visual_calibration_quality(
        striker=striker,
        non_striker=non_striker,
        pitch_geometry=pitch_geometry,
        assignment_warning=assignment_warning,
        proximity_warning=proximity_warning,
        detector_success=bool(detection_result.get("success")),
    )

    detection_debug = _persist_detection_debug(
        analysis_id,
        image=image,
        detection_result=detection_result,
    )

    guide_models = {
        "striker": NormalizedBox(**guides["striker"]),
        "non_striker": NormalizedBox(**guides["non_striker"]),
    }

    if not detection_result["success"]:
        status = detection_result["status"]
        message = detection_result["message"]
        success = False
    elif striker is None or non_striker is None:
        status = "detection_incomplete"
        failed_label = " and ".join(
            end.replace("_", "-") for end in failed_ends
        ) or "wickets"
        message = (
            f"Guided detection could not lock the {failed_label} end(s) "
            "inside the current guide boxes. Reposition the guide(s) and "
            "press Detect Wickets again."
        )
        success = True
    else:
        status = "candidates_ready"
        message = (
            "Guided visual calibration detected both wickets. "
            "Review the overlay, then Accept, Redetect, or Swap Wicket Ends."
        )
        success = True

    warning = proximity_warning or assignment_warning
    return VideoCalibrationDetectionResponse(
        success=success,
        status=status,
        analysis_id=analysis_id,
        reference_frame_index=analysis.reference_frame_index,
        reference_frame_url=analysis.reference_frame_url,
        image_width=image_width,
        image_height=image_height,
        candidates=candidates,
        provisional_striker_wicket=striker,
        provisional_non_striker_wicket=non_striker,
        pitch_geometry=pitch_geometry,
        striker_guide=guide_models["striker"],
        non_striker_guide=guide_models["non_striker"],
        failed_ends=failed_ends,  # type: ignore[arg-type]
        model_path_used=STUMP_MODEL_RELATIVE_PATH,
        mode="automatic_visual",
        quality=quality,
        quality_reasons=quality_reasons,
        assignment_warning=assignment_warning,
        warning=warning,
        message=message,
        detection_debug=detection_debug,
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
    assignment_warning = _assignment_uncertainty_warning(striker, non_striker)
    quality, quality_reasons = assess_visual_calibration_quality(
        striker=striker,
        non_striker=non_striker,
        pitch_geometry=pitch_geometry,
        assignment_warning=assignment_warning,
        proximity_warning=None,
        detector_success=True,
    )
    if quality == "FAILED":
        raise VideoAnalysisServiceError(
            "Automatic visual calibration quality is FAILED. "
            "Redetect before accepting.",
            status_code=422,
        )

    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    calibration_dir = analysis_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / CALIBRATION_FILENAME
    overlay_path = calibration_dir / CALIBRATION_OVERLAY_FILENAME
    scene_overlay_path = calibration_dir / SCENE_OVERLAY_FILENAME
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
    scene_overlay_url = (
        f"/static/video-analysis/{analysis_id}/calibration/"
        f"{SCENE_OVERLAY_FILENAME}"
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
        scene_overlay_url=None,
        scene_overlay_status=None,
        image_width=image_width,
        image_height=image_height,
        model_path_used=model_path_used,
        mode="automatic_visual",
        quality=quality,
        quality_reasons=quality_reasons,
        assignment_warning=assignment_warning,
        striker_wicket=striker,
        non_striker_wicket=non_striker,
        pitch_geometry=pitch_geometry,
        striker_guide=request.striker_guide,
        non_striker_guide=request.non_striker_guide,
        user_note=request.user_note,
        message=(
            "Guided scene overlay locked. "
            "Approximate 2D scene context only — not metric 3D."
        ),
    )

    _save_calibration_overlay(image, confirmed, overlay_path)
    # ponytail: scene_overlay.mp4 is a separate display artefact; never feed
    # composited frames into ball detection / YOLO.
    scene_overlay_status: str = "failed"
    scene_overlay_url_value: str | None = None
    try:
        _render_scene_overlay_video(
            analysis_id,
            analysis,
            confirmed,
            scene_overlay_path,
        )
        scene_overlay_status = "ready"
        scene_overlay_url_value = scene_overlay_url
    except Exception:
        scene_overlay_path.unlink(missing_ok=True)
        scene_overlay_status = "failed"
        scene_overlay_url_value = None

    confirmed = confirmed.model_copy(
        update={
            "scene_overlay_url": scene_overlay_url_value,
            "scene_overlay_status": scene_overlay_status,
        }
    )
    _write_json(calibration_path, confirmed.model_dump(mode="json"))
    _update_analysis_metadata(
        analysis_dir,
        updated_at=now,
        calibration_url=calibration_url,
        overlay_url=overlay_url,
        quality=quality,
        scene_overlay_url=scene_overlay_url_value,
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
    detection_pass = candidate.get("source")
    if detection_pass not in {
        "full_frame",
        "far_roi",
        "near_roi",
        "guide_roi",
    }:
        detection_pass = None
    return WicketCandidate(
        candidate_id=f"wicket_{index}",
        confidence=float(candidate["confidence"]),
        class_name=str(candidate["class_name"]),
        box=box,
        center=center,
        bottom_center=bottom_center,
        detection_pass=detection_pass,
    )


def _match_selected_candidate(
    candidates: list[WicketCandidate],
    selected: dict[str, Any],
    image_width: int,
    image_height: int,
) -> WicketCandidate | None:
    selected_bbox = selected.get("bbox") or {}
    try:
        target = NormalizedBox(
            x=_clamp(float(selected_bbox["x"]) / image_width),
            y=_clamp(float(selected_bbox["y"]) / image_height),
            width=max(
                0.000001,
                _clamp(
                    (float(selected_bbox["x"]) + float(selected_bbox["width"]))
                    / image_width
                )
                - _clamp(float(selected_bbox["x"]) / image_width),
            ),
            height=max(
                0.000001,
                _clamp(
                    (
                        float(selected_bbox["y"])
                        + float(selected_bbox["height"])
                    )
                    / image_height
                )
                - _clamp(float(selected_bbox["y"]) / image_height),
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None
    best: WicketCandidate | None = None
    best_iou = 0.0
    for candidate in candidates:
        iou = _intersection_over_union(candidate.box, target)
        if iou > best_iou:
            best = candidate
            best_iou = iou
    return best if best_iou >= 0.2 else None


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
        approximate_wicket_base_reference=candidate.bottom_center,
        detection_pass=candidate.detection_pass,
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
        approximate_wicket_base_reference=bottom_center,
        detection_pass=wicket.detection_pass,
    )


def _persist_detection_debug(
    analysis_id: str,
    *,
    image: Image.Image,
    detection_result: dict[str, Any],
) -> VisualCalibrationDetectionDebug | None:
    calibration_dir = VIDEO_ANALYSIS_ROOT / analysis_id / "calibration"
    diagnostics = detection_result.get("diagnostics") or {}
    rois_raw = detection_result.get("rois") or {}
    rois: dict[str, NormalizedBox] = {}
    for name, roi in rois_raw.items():
        try:
            rois[name] = NormalizedBox(
                x=float(roi["x"]),
                y=float(roi["y"]),
                width=float(roi["width"]),
                height=float(roi["height"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

    overlay_path = calibration_dir / DETECTION_DEBUG_OVERLAY
    json_path = calibration_dir / DETECTION_DEBUG_JSON
    try:
        save_robust_detection_overlay(
            image,
            candidates=detection_result.get("candidates") or [],
            selected=detection_result.get("selected") or {},
            rois=rois_raw,
            output_path=overlay_path,
        )
        payload = {
            "analysis_id": analysis_id,
            "diagnostics": diagnostics,
            "rois": rois_raw,
            "selected": detection_result.get("selected"),
            "candidates": [
                {
                    "confidence": item.get("confidence"),
                    "class_name": item.get("class_name"),
                    "source": item.get("source"),
                    "bbox": item.get("bbox"),
                }
                for item in (detection_result.get("candidates") or [])
            ],
        }
        _write_json(json_path, payload)
    except Exception:
        # ponytail: debug artefacts must never block calibration.
        return VisualCalibrationDetectionDebug(
            pass_count=int(diagnostics.get("pass_count") or 0),
            passes=list(diagnostics.get("passes") or []),
            rejected=list(diagnostics.get("rejected") or []),
            rois=rois,
            selected=diagnostics.get("selected"),
            debug_overlay_url=None,
            debug_json_url=None,
        )

    return VisualCalibrationDetectionDebug(
        pass_count=int(diagnostics.get("pass_count") or 0),
        passes=list(diagnostics.get("passes") or []),
        rejected=list(diagnostics.get("rejected") or []),
        rois=rois,
        selected=diagnostics.get("selected"),
        debug_overlay_url=(
            f"/static/video-analysis/{analysis_id}/calibration/"
            f"{DETECTION_DEBUG_OVERLAY}"
        ),
        debug_json_url=(
            f"/static/video-analysis/{analysis_id}/calibration/"
            f"{DETECTION_DEBUG_JSON}"
        ),
    )


def _try_early_frame_fallback(analysis_id: str, analysis: Any) -> bool:
    """If current early frame fails two-wicket lock, try a few other early ones."""
    import cv2

    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    video_path = analysis_dir / "raw" / analysis.stored_filename
    if not video_path.is_file():
        return False

    current_index = int(analysis.reference_frame_index)
    window = max(1, min(EARLY_REFERENCE_WINDOW, int(analysis.frame_count)))
    capture = cv2.VideoCapture(str(video_path))
    tried = 0
    try:
        if not capture.isOpened():
            return False
        for index in range(window):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index == current_index:
                continue
            usable, score = _reference_frame_quality(frame)
            if not usable:
                continue
            # Quick robust check before replacing the stored reference.
            rgb = Image.fromarray(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            probe = detect_wickets_robust(rgb, enable_roi=True)
            selected = probe.get("selected") or {}
            if selected.get("striker") is None or selected.get("non_striker") is None:
                tried += 1
                if tried >= EARLY_FALLBACK_ATTEMPTS:
                    return False
                continue
            replace_reference_frame(
                analysis_id,
                frame_index=index,
                frame=frame,
                selection={
                    "strategy": "earliest_clean_stable",
                    "window_scanned": index + 1,
                    "window_limit": window,
                    "selected_index": index,
                    "score": round(float(score), 3),
                    "reason": "two_wicket_early_fallback",
                },
            )
            return True
    finally:
        capture.release()
    return False


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


def assess_visual_calibration_quality(
    *,
    striker: WicketCalibration | None,
    non_striker: WicketCalibration | None,
    pitch_geometry: PitchGeometry | None,
    assignment_warning: str | None,
    proximity_warning: str | None,
    detector_success: bool,
) -> tuple[VisualCalibrationQuality, list[str]]:
    reasons: list[str] = []
    if not detector_success:
        return "FAILED", ["Stump detector did not succeed."]
    if striker is None or non_striker is None:
        return "FAILED", ["Both striker and non-striker wickets are required."]
    if pitch_geometry is None:
        return "FAILED", ["Approximate pitch corridor could not be built."]

    distance = math.hypot(
        striker.bottom_center.x - non_striker.bottom_center.x,
        striker.bottom_center.y - non_striker.bottom_center.y,
    )
    if distance < 0.055:
        reasons.append("Wicket separation is too small.")
    if _intersection_over_union(striker.box, non_striker.box) > 0.3:
        reasons.append("Wicket boxes overlap too much.")
    if proximity_warning:
        reasons.append(proximity_warning)

    for wicket, name in (
        (striker, "striker"),
        (non_striker, "non_striker"),
    ):
        if wicket.box.width < 0.012 or wicket.box.height < 0.02:
            reasons.append(f"{name} wicket box is unusually small.")
        if (
            wicket.box.x <= 0.005
            or wicket.box.y <= 0.005
            or wicket.box.x + wicket.box.width >= 0.995
            or wicket.box.y + wicket.box.height >= 0.995
        ):
            reasons.append(f"{name} wicket is clipped near the frame edge.")
        if wicket.confidence is not None and wicket.confidence < 0.25:
            reasons.append(f"{name} detection confidence is low.")

    # Corridor inverted / collapsed: near end should be larger than far end.
    near = (
        striker
        if pitch_geometry.near_end_label == "striker"
        else non_striker
    )
    far = (
        non_striker
        if pitch_geometry.near_end_label == "striker"
        else striker
    )
    if near.box.width + near.box.height < far.box.width + far.box.height:
        reasons.append(
            "Perspective ordering looks inverted for a rear-camera view."
        )

    if any(
        "too small" in reason.lower()
        or "overlap" in reason.lower()
        or "separation" in reason.lower()
        for reason in reasons
    ):
        return "FAILED", reasons

    if assignment_warning:
        reasons.append(assignment_warning)
    if reasons:
        return "WEAK", reasons
    return "READY", ["Both wickets detected with usable approximate geometry."]


def _assignment_uncertainty_warning(
    striker: WicketCalibration,
    non_striker: WicketCalibration,
) -> str | None:
    score_gap = abs(_near_score(striker) - _near_score(non_striker))
    size_ratio = max(striker.box.width, non_striker.box.width) / max(
        min(striker.box.width, non_striker.box.width),
        0.000001,
    )
    if score_gap < 0.08 or size_ratio < 1.12:
        return (
            "Wicket-end assignment is uncertain for this camera view. "
            "Use Swap Wicket Ends if striker/non-striker labels look reversed."
        )
    return None


def _try_refresh_early_reference(analysis_id: str, analysis: Any) -> None:
    """On Redetect, try another early clean frame if the current one is weak."""
    import cv2

    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    video_path = analysis_dir / "raw" / analysis.stored_filename
    if not video_path.is_file():
        return

    current_index = int(analysis.reference_frame_index)
    window = max(1, min(EARLY_REFERENCE_WINDOW, int(analysis.frame_count)))
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return
        for index in range(window):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if index == current_index:
                continue
            usable, score = _reference_frame_quality(frame)
            if not usable:
                continue
            replace_reference_frame(
                analysis_id,
                frame_index=index,
                frame=frame,
                selection={
                    "strategy": "earliest_clean_stable",
                    "window_scanned": index + 1,
                    "window_limit": window,
                    "selected_index": index,
                    "score": round(float(score), 3),
                    "reason": "redetect_early_alternate",
                },
            )
            return
    finally:
        capture.release()


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


def _resolve_guides(
    striker_guide: NormalizedBox | None,
    non_striker_guide: NormalizedBox | None,
) -> dict[str, dict[str, float]]:
    defaults = default_wicket_guides()
    return {
        "striker": (
            striker_guide.model_dump()
            if striker_guide is not None
            else defaults["striker"]
        ),
        "non_striker": (
            non_striker_guide.model_dump()
            if non_striker_guide is not None
            else defaults["non_striker"]
        ),
    }


def _normalized_wicket_bbox_pixels(
    wicket: WicketCalibration,
    width: int,
    height: int,
) -> dict[str, float]:
    return {
        "x": float(wicket.box.x) * width,
        "y": float(wicket.box.y) * height,
        "width": float(wicket.box.width) * width,
        "height": float(wicket.box.height) * height,
    }


def _save_calibration_overlay(
    image: Image.Image,
    calibration: ConfirmedVideoCalibrationResponse,
    output_path: Path,
) -> None:
    try:
        width, height = image.size
        composed = compose_scene_overlay_image(
            image,
            striker_bbox=_normalized_wicket_bbox_pixels(
                calibration.striker_wicket, width, height
            ),
            non_striker_bbox=_normalized_wicket_bbox_pixels(
                calibration.non_striker_wicket, width, height
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composed.save(output_path, format="JPEG", quality=92)
    except Exception as exc:
        raise VideoAnalysisServiceError(
            f"Calibration overlay could not be saved: {type(exc).__name__}.",
            status_code=500,
        ) from exc


def _render_scene_overlay_video(
    analysis_id: str,
    analysis: Any,
    calibration: ConfirmedVideoCalibrationResponse,
    output_path: Path,
) -> None:
    """Composite a prebuilt RGBA scene overlay onto clean original frames."""
    import cv2
    import numpy as np

    analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
    video_path = analysis_dir / "raw" / analysis.stored_filename
    if not video_path.is_file():
        raise VideoAnalysisServiceError(
            "Original analysis video is missing.",
            status_code=404,
        )

    total_frames = int(analysis.frame_count)
    if total_frames <= 0:
        raise VideoAnalysisServiceError(
            "Original analysis video contains zero frames.",
            status_code=500,
        )

    intermediate_path = output_path.with_name("scene_overlay_intermediate.avi")
    encoded_path = output_path.with_name("scene_overlay_encoded.mp4")
    for stale_path in (output_path, intermediate_path, encoded_path):
        stale_path.unlink(missing_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not capture.isOpened():
            raise VideoAnalysisServiceError(
                "OpenCV could not open the original video.",
                status_code=500,
            )
        input_fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(input_fps) or input_fps <= 0:
            raise VideoAnalysisServiceError(
                "Original analysis video has an invalid FPS value.",
                status_code=500,
            )
        if width <= 0 or height <= 0:
            raise VideoAnalysisServiceError(
                "Original analysis video has invalid dimensions.",
                status_code=500,
            )
        if capture_frame_count > 0 and capture_frame_count != total_frames:
            raise VideoAnalysisServiceError(
                "Stored frame count no longer matches the original video.",
                status_code=500,
            )

        # Prebuild once; composite onto clean frames only (separate output).
        overlay_rgba = build_scene_overlay_rgba(
            frame_width=width,
            frame_height=height,
            striker_bbox=_normalized_wicket_bbox_pixels(
                calibration.striker_wicket, width, height
            ),
            non_striker_bbox=_normalized_wicket_bbox_pixels(
                calibration.non_striker_wicket, width, height
            ),
        )
        overlay_np = np.asarray(overlay_rgba)
        alpha = overlay_np[:, :, 3:4].astype(np.float32) / 255.0
        overlay_bgr = cv2.cvtColor(overlay_np[:, :, :3], cv2.COLOR_RGB2BGR)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(intermediate_path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            input_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise VideoAnalysisServiceError(
                "Could not create the scene-overlay intermediate video.",
                status_code=500,
            )

        for frame_index in range(total_frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise VideoAnalysisServiceError(
                    f"Video decoding stopped at frame {frame_index} "
                    f"of {total_frames}.",
                    status_code=500,
                )
            composed = (
                frame.astype(np.float32) * (1.0 - alpha)
                + overlay_bgr.astype(np.float32) * alpha
            ).astype(np.uint8)
            writer.write(composed)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    try:
        transcode_browser_mp4(
            intermediate_path,
            encoded_path,
            timeout_seconds=600,
        )
        encoded_path.replace(output_path)
    except Exception as exc:
        encoded_path.unlink(missing_ok=True)
        raise VideoAnalysisServiceError(
            f"Could not encode scene overlay video: {type(exc).__name__}.",
            status_code=500,
        ) from exc
    finally:
        intermediate_path.unlink(missing_ok=True)


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
    quality: VisualCalibrationQuality,
    scene_overlay_url: str | None = None,
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
            "visual_calibration_mode": "automatic_visual",
            "visual_calibration_quality": quality,
            "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
        }
    )
    if scene_overlay_url:
        metadata["scene_overlay_url"] = scene_overlay_url
        metadata["scene_overlay_status"] = "ready"
    else:
        metadata["scene_overlay_url"] = None
        metadata["scene_overlay_status"] = "failed"
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
