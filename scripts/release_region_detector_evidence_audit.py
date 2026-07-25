"""Offline Release-Region Detector Evidence Audit V1.

Human release frames and optional ball locations are evaluation inputs only.
This module never changes production detector, persistence, or tracking settings.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.api.services.ball_detection_clip import extract_ball_candidates  # noqa: E402
from services.api.services.ball_detector_registry import (  # noqa: E402
    get_ball_detector_model,
    load_ball_detector_model,
)


VALIDATION_ROOT = ROOT / "outputs" / "release_validation"
ANALYSES_ROOT = ROOT / "outputs" / "video_analysis"
ANNOTATIONS_PATH = VALIDATION_ROOT / "release_annotations.json"
OUTPUT_ROOT = VALIDATION_ROOT / "detector_evidence_audit"
AUDIT_POINTS_PATH = OUTPUT_ROOT / "true_ball_audit_annotations.json"
RAW_CACHE_PATH = OUTPUT_ROOT / "raw_yolo_evidence.json"
THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.20)
DIAGNOSTIC_FLOOR = 0.01
PRODUCTION_THRESHOLD = 0.15
PRODUCTION_MAX_DETECTIONS = 20
INFERENCE_SIZE = 960
WINDOW_RADIUS = 10
MATCH_RADIUS_PX = 18.0
KEY_CLIPS = {"rv1_002", "rv1_008", "rv1_011", "rv1_012", "rv1_013"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("template", "infer", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    annotations = _release_annotations()
    if args.command in {"template", "all"}:
        write_annotation_template(annotations)
    if args.command in {"infer", "all"}:
        run_raw_inference(
            annotations,
            device=args.device,
            force=args.force_inference,
        )
    if args.command in {"report", "all"}:
        build_report(annotations)
    return 0


def write_annotation_template(release_annotations: list[dict[str, Any]]) -> None:
    existing = _read_json(AUDIT_POINTS_PATH) if AUDIT_POINTS_PATH.is_file() else {}
    existing_clips = {
        item["validation_id"]: item for item in existing.get("clips", [])
    }
    clips = []
    for item in release_annotations:
        validation_id = item["validation_id"]
        prior = existing_clips.get(validation_id, {})
        clips.append(
            {
                "validation_id": validation_id,
                "analysis_id": item["analysis_id"],
                "human_release_frame": item["human_release_frame"],
                "annotation_status": prior.get("annotation_status", "pending"),
                "frames": prior.get("frames", []),
                "notes": prior.get("notes", ""),
            }
        )
    _write_json(
        AUDIT_POINTS_PATH,
        {
            "schema_version": "1.0",
            "purpose": "Offline spatial true-ball matching; never a production input.",
            "match_policy": {
                "default_radius_px": MATCH_RADIUS_PX,
                "frame_fields": {
                    "frame_index": "zero-based clean-video frame",
                    "ball_visible": "true, false, or null when uncertain",
                    "point": "[x, y] center or null",
                    "bbox_xyxy": "optional [x1, y1, x2, y2]",
                    "false_positive_labels": "optional candidate-id to visual category map",
                },
            },
            "clips": clips,
        },
    )


def run_raw_inference(
    release_annotations: list[dict[str, Any]],
    *,
    device: str | None,
    force: bool,
) -> dict[str, Any]:
    if RAW_CACHE_PATH.is_file() and not force:
        return _read_json(RAW_CACHE_PATH)

    selected_model, model = load_ball_detector_model("e4c_best_overall")
    started = perf_counter()
    clips: list[dict[str, Any]] = []
    for clip in release_annotations:
        analysis_dir = ANALYSES_ROOT / clip["analysis_id"]
        raw_path = _raw_video_path(analysis_dir, clip)
        human_frame = int(clip["human_release_frame"])
        start = max(0, human_frame - WINDOW_RADIUS)
        end = human_frame + WINDOW_RADIUS
        clip_result: dict[str, Any] = {
            "validation_id": clip["validation_id"],
            "analysis_id": clip["analysis_id"],
            "human_release_frame": human_frame,
            "window_start_frame": start,
            "window_end_frame": end,
            "raw_video_path": str(raw_path) if raw_path else None,
            "status": "ready" if raw_path else "raw_video_missing",
            "frames": [],
        }
        if raw_path is None:
            clips.append(clip_result)
            continue

        capture = cv2.VideoCapture(str(raw_path))
        if not capture.isOpened():
            clip_result["status"] = "raw_video_unreadable"
            clips.append(clip_result)
            continue
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        end = min(end, max(0, total - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            for frame_index in range(start, end + 1):
                ok, frame = capture.read()
                if not ok:
                    clip_result["status"] = "frame_decode_failed"
                    break
                results = model.predict(
                    source=frame,
                    imgsz=INFERENCE_SIZE,
                    conf=DIAGNOSTIC_FLOOR,
                    max_det=300,
                    device=device,
                    verbose=False,
                )
                candidates = extract_ball_candidates(
                    results,
                    getattr(model, "names", {}),
                    strict=True,
                )
                candidates.sort(key=lambda row: row["confidence"], reverse=True)
                clip_result["frames"].append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": (
                            round(frame_index / fps, 6) if fps > 0 else None
                        ),
                        "candidates": [
                            _raw_candidate(candidate, rank)
                            for rank, candidate in enumerate(candidates, 1)
                        ],
                    }
                )
        finally:
            capture.release()
        clips.append(clip_result)

    document = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "audit_only": True,
        "model": {
            "key": selected_model.key,
            "name": selected_model.display_name,
            "path": str(selected_model.path),
        },
        "diagnostic_settings": {
            "confidence_floor": DIAGNOSTIC_FLOOR,
            "inference_size": INFERENCE_SIZE,
            "max_detections": 300,
            "window_radius_frames": WINDOW_RADIUS,
        },
        "production_reference_settings": {
            "confidence_threshold": PRODUCTION_THRESHOLD,
            "inference_size": INFERENCE_SIZE,
            "max_detections": PRODUCTION_MAX_DETECTIONS,
        },
        "processing_seconds": round(perf_counter() - started, 3),
        "clips": clips,
    }
    _write_json(RAW_CACHE_PATH, document)
    return document


def build_report(release_annotations: list[dict[str, Any]]) -> dict[str, Any]:
    raw_doc = _read_json(RAW_CACHE_PATH)
    audit_annotations = _read_json(AUDIT_POINTS_PATH)
    audit_by_id = {
        item["validation_id"]: item for item in audit_annotations.get("clips", [])
    }
    raw_by_id = {item["validation_id"]: item for item in raw_doc["clips"]}
    frozen_compatibility = _frozen_release_input_compatibility()
    rows: list[dict[str, Any]] = []
    clips: list[dict[str, Any]] = []

    for release in release_annotations:
        validation_id = release["validation_id"]
        analysis_dir = ANALYSES_ROOT / release["analysis_id"]
        raw_clip = raw_by_id[validation_id]
        truth_clip = audit_by_id.get(validation_id, {})
        truth_by_frame = {
            int(row["frame_index"]): row for row in truth_clip.get("frames", [])
        }
        persisted_doc = _read_optional_json(
            analysis_dir / "detections" / "detections.json"
        )
        tracking_doc = _read_optional_json(
            analysis_dir / "tracking" / "tracking_result.json"
        )
        persisted_by_frame = _persisted_by_frame(persisted_doc)
        selected_by_frame = _selected_track_by_frame(tracking_doc)
        compatibility = _compatibility(
            analysis_dir,
            persisted_doc,
            tracking_doc,
            frozen_compatibility.get(validation_id),
        )
        frame_rows = []
        for raw_frame in raw_clip.get("frames", []):
            frame_index = int(raw_frame["frame_index"])
            truth = truth_by_frame.get(frame_index)
            persisted = persisted_by_frame.get(frame_index, [])
            selected = selected_by_frame.get(frame_index)
            frame_row = _evaluate_frame(
                validation_id=validation_id,
                analysis_id=release["analysis_id"],
                frame_index=frame_index,
                timestamp=raw_frame.get("timestamp_seconds"),
                truth=truth,
                raw_candidates=raw_frame.get("candidates", []),
                persisted=persisted,
                selected=selected,
                persistence_comparable=compatibility[
                    "raw_e4c_to_persisted_model_comparable"
                ],
            )
            rows.append(frame_row)
            frame_rows.append(frame_row)

        clip_summary = _summarize_clip(
            release,
            raw_clip,
            truth_clip,
            compatibility,
            frame_rows,
        )
        clips.append(clip_summary)
        if validation_id in KEY_CLIPS:
            _write_debug_frames(
                release=release,
                raw_clip=raw_clip,
                truth_by_frame=truth_by_frame,
                persisted_by_frame=persisted_by_frame,
                selected_by_frame=selected_by_frame,
                frame_rows=frame_rows,
            )

    aggregate = _aggregate(clips, rows, raw_doc)
    _write_json(
        OUTPUT_ROOT / "detector_evidence_audit.json",
        {
            "schema_version": "1.0",
            "created_at": _utc_now(),
            "method": _method_document(),
            "clips": clips,
            "frames": rows,
        },
    )
    _write_csv(OUTPUT_ROOT / "detector_evidence_audit.csv", rows)
    _write_json(
        OUTPUT_ROOT / "aggregate_detector_evidence_summary.json",
        aggregate,
    )
    (OUTPUT_ROOT / "aggregate_detector_evidence_summary.md").write_text(
        _summary_markdown(aggregate, clips),
        encoding="utf-8",
    )
    return aggregate


def _evaluate_frame(
    *,
    validation_id: str,
    analysis_id: str,
    frame_index: int,
    timestamp: float | None,
    truth: dict[str, Any] | None,
    raw_candidates: list[dict[str, Any]],
    persisted: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    persistence_comparable: bool = True,
) -> dict[str, Any]:
    visible = None if truth is None else truth.get("ball_visible")
    point = None if truth is None else truth.get("point")
    bbox = None if truth is None else truth.get("bbox_xyxy")
    matching_raw = _best_match(raw_candidates, point, bbox)
    matching_persisted = _best_match(persisted, point, bbox)
    selected_match = _candidate_matches(selected, point, bbox)
    raw_rank = None if matching_raw is None else matching_raw.get("rank")
    confidence = None if matching_raw is None else matching_raw.get("confidence")
    disappearance = (
        _disappearance_reason(matching_raw, matching_persisted, selected_match)
        if persistence_comparable
        else (
            "historical_persisted_detector_model_differs_from_raw_audit_model"
            if matching_raw is not None
            else "not_produced_by_e4c_above_diagnostic_floor"
        )
    )
    if visible is True and matching_raw is None:
        comparison = "NO_RAW_BALL_EVIDENCE"
    elif matching_raw is not None and matching_persisted is None:
        comparison = "RAW_NOT_PERSISTED"
    elif matching_persisted is not None and not selected_match:
        comparison = "PERSISTED_NOT_SELECTED"
    elif selected_match:
        comparison = "SELECTED_PRIMARY"
    elif matching_raw is not None:
        comparison = "RAW_AND_PERSISTED"
    else:
        comparison = "UNAVAILABLE"
    return {
        "validation_id": validation_id,
        "analysis_id": analysis_id,
        "frame_index": frame_index,
        "timestamp_seconds": timestamp,
        "spatial_ground_truth_available": bool(point or bbox),
        "ball_visible": visible,
        "true_ball_point": point,
        "true_ball_bbox_xyxy": bbox,
        "raw_candidate_count_at_0_01": len(raw_candidates),
        "raw_candidates": raw_candidates,
        "matching_raw_candidate": matching_raw,
        "true_ball_raw_confidence": confidence,
        "true_ball_raw_rank": raw_rank,
        "threshold_presence": {
            str(value): bool(
                matching_raw is not None and confidence is not None and confidence >= value
            )
            for value in (*THRESHOLDS, PRODUCTION_THRESHOLD)
        },
        "persisted_candidate_count": len(persisted),
        "matching_persisted_candidate": matching_persisted,
        "raw_to_persisted_model_comparable": persistence_comparable,
        "primary_track_candidate": selected,
        "primary_track_matches_true_ball": selected_match,
        "comparison_category": comparison,
        "true_ball_disappearance_reason": disappearance,
        "false_positive_labels": (
            {} if truth is None else truth.get("false_positive_labels", {})
        ),
    }


def _summarize_clip(
    release: dict[str, Any],
    raw_clip: dict[str, Any],
    truth_clip: dict[str, Any],
    compatibility: dict[str, Any],
    frame_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluated = [
        row
        for row in frame_rows
        if row["ball_visible"] is True and row["spatial_ground_truth_available"]
    ]
    raw_hits = sum(row["matching_raw_candidate"] is not None for row in evaluated)
    persisted_hits = sum(
        row["matching_persisted_candidate"] is not None for row in evaluated
    )
    track_hits = sum(row["primary_track_matches_true_ball"] for row in evaluated)
    category = _classify_clip(compatibility, evaluated)
    return {
        "validation_id": release["validation_id"],
        "analysis_id": release["analysis_id"],
        "human_release_frame": release["human_release_frame"],
        "diagnostic_status": raw_clip.get("status"),
        "audit_annotation_status": truth_clip.get("annotation_status", "missing"),
        "spatially_evaluated_visible_frames": len(evaluated),
        "raw_true_ball_hits": raw_hits,
        "persisted_true_ball_hits": persisted_hits,
        "primary_track_true_ball_hits": track_hits,
        "raw_detector_recall": _ratio(raw_hits, len(evaluated)),
        "persisted_candidate_recall": _ratio(persisted_hits, len(evaluated)),
        "primary_track_recall": _ratio(track_hits, len(evaluated)),
        "true_ball_confidences": [
            row["true_ball_raw_confidence"]
            for row in evaluated
            if row["true_ball_raw_confidence"] is not None
        ],
        "true_ball_ranks": [
            row["true_ball_raw_rank"]
            for row in evaluated
            if row["true_ball_raw_rank"] is not None
        ],
        "true_ball_detections_lost_by_persistence": (
            sum(
            row["matching_raw_candidate"] is not None
            and row["matching_persisted_candidate"] is None
            for row in evaluated
            )
            if all(row["raw_to_persisted_model_comparable"] for row in evaluated)
            else None
        ),
        "true_ball_frames_never_detected": sum(
            row["matching_raw_candidate"] is None for row in evaluated
        ),
        "input_compatibility": compatibility,
        "dominant_evidence_bottleneck": category,
        "classification_explanation": _classification_explanation(
            category, evaluated, compatibility
        ),
    }


def _classify_clip(
    compatibility: dict[str, Any],
    evaluated: list[dict[str, Any]],
) -> str:
    if (
        not compatibility["raw_original_video_available"]
        or compatibility.get("frozen_release_input_compatible") is False
    ):
        return "HISTORICAL_INPUT_INCOMPATIBLE"
    if not evaluated:
        return (
            "HISTORICAL_INPUT_INCOMPATIBLE"
            if not compatibility["modern_detection_json_available"]
            or not compatibility["primary_tracking_valid"]
            else "UNKNOWN"
        )
    raw_hits = [row for row in evaluated if row["matching_raw_candidate"]]
    if not raw_hits:
        return "DETECTOR_NO_TRUE_BALL_OUTPUT"
    persisted_hits = [row for row in evaluated if row["matching_persisted_candidate"]]
    selected_hits = [row for row in evaluated if row["primary_track_matches_true_ball"]]
    if len(raw_hits) < len(evaluated):
        return "DETECTOR_NO_TRUE_BALL_OUTPUT"
    if any(row["true_ball_raw_confidence"] < PRODUCTION_THRESHOLD for row in raw_hits):
        return "DETECTOR_TRUE_BALL_LOW_CONFIDENCE"
    if any((row["true_ball_raw_rank"] or 0) > PRODUCTION_MAX_DETECTIONS for row in raw_hits):
        return "TRUE_BALL_RANKED_TOO_LOW"
    if len(persisted_hits) < len(raw_hits):
        return "TRUE_BALL_LOST_DURING_PERSISTENCE"
    if len(selected_hits) < len(persisted_hits):
        return "TRUE_BALL_PERSISTED_BUT_TRACKER_REJECTED"
    return "SUFFICIENT_NEAR_RELEASE_EVIDENCE"


def _aggregate(
    clips: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    raw_doc: dict[str, Any],
) -> dict[str, Any]:
    evaluated = [
        row
        for row in rows
        if row["ball_visible"] is True and row["spatial_ground_truth_available"]
    ]
    raw_hits = [row for row in evaluated if row["matching_raw_candidate"]]
    persisted_hits = [row for row in evaluated if row["matching_persisted_candidate"]]
    track_hits = [row for row in evaluated if row["primary_track_matches_true_ball"]]
    confidences = [row["true_ball_raw_confidence"] for row in raw_hits]
    ranks = [int(row["true_ball_raw_rank"]) for row in raw_hits]
    rank_distribution = Counter(
        "rank_1"
        if rank == 1
        else "rank_2"
        if rank == 2
        else "rank_3"
        if rank == 3
        else "rank_4_to_10"
        if rank <= 10
        else "over_10"
        for rank in ranks
    )
    false_positive_types = Counter(
        label
        for row in evaluated
        for label in row.get("false_positive_labels", {}).values()
        if label
    )
    return {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "audit_scope": {
            "clip_count": len(clips),
            "window_radius_frames": WINDOW_RADIUS,
            "diagnostic_confidence_floor": DIAGNOSTIC_FLOOR,
            "production_confidence_threshold": PRODUCTION_THRESHOLD,
            "model": raw_doc["model"],
        },
        "evaluable_clip_count": sum(
            clip["spatially_evaluated_visible_frames"] > 0 for clip in clips
        ),
        "spatially_evaluated_visible_frame_count": len(evaluated),
        "raw_yolo_true_ball_recall": _ratio(len(raw_hits), len(evaluated)),
        "persisted_candidate_true_ball_recall": _ratio(
            len(persisted_hits), len(evaluated)
        ),
        "primary_track_true_ball_recall": _ratio(len(track_hits), len(evaluated)),
        "true_ball_confidence_distribution": {
            "count": len(confidences),
            "minimum": min(confidences, default=None),
            "maximum": max(confidences, default=None),
            "mean": (
                round(sum(confidences) / len(confidences), 6)
                if confidences
                else None
            ),
            "below_production_threshold_count": sum(
                value < PRODUCTION_THRESHOLD for value in confidences
            ),
            "threshold_recall": {
                str(threshold): _ratio(
                    sum(value >= threshold for value in confidences),
                    len(evaluated),
                )
                for threshold in (*THRESHOLDS, PRODUCTION_THRESHOLD)
            },
        },
        "true_ball_rank_distribution": {
            **{
                key: rank_distribution.get(key, 0)
                for key in ("rank_1", "rank_2", "rank_3", "rank_4_to_10", "over_10")
            },
            "not_detected": len(evaluated) - len(raw_hits),
        },
        "real_ball_detections_lost_by_persistence": (
            sum(
                row["matching_raw_candidate"] is not None
                and row["matching_persisted_candidate"] is None
                for row in evaluated
            )
            if evaluated
            and all(row["raw_to_persisted_model_comparable"] for row in evaluated)
            else None
        ),
        "persistence_loss_attribution_available": bool(
            evaluated
            and all(row["raw_to_persisted_model_comparable"] for row in evaluated)
        ),
        "real_ball_frames_never_detected_by_yolo": len(evaluated) - len(raw_hits),
        "dominant_false_positive_types": dict(false_positive_types.most_common()),
        "failure_category_counts": dict(
            Counter(clip["dominant_evidence_bottleneck"] for clip in clips)
        ),
        "input_compatibility_counts": {
            "raw_original_video_available": sum(
                clip["input_compatibility"]["raw_original_video_available"]
                for clip in clips
            ),
            "modern_detection_json_available": sum(
                clip["input_compatibility"]["modern_detection_json_available"]
                for clip in clips
            ),
            "top_k_persistence_available": sum(
                clip["input_compatibility"]["top_k_persistence_available"]
                for clip in clips
            ),
            "primary_tracking_valid": sum(
                clip["input_compatibility"]["primary_tracking_valid"]
                for clip in clips
            ),
            "raw_e4c_to_persisted_model_comparable": sum(
                clip["input_compatibility"][
                    "raw_e4c_to_persisted_model_comparable"
                ]
                for clip in clips
            ),
        },
        "clips": clips,
        "measurement_limitation": (
            "Recall denominators include only frames explicitly annotated with a "
            "visible true-ball point or box. Unannotated frames are unavailable, "
            "not detector misses."
        ),
    }


def _persisted_by_frame(document: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    if not document:
        return result
    for frame in document.get("frames", []):
        frame_index = frame.get("frame_index")
        if frame_index is None:
            continue
        candidates = []
        for rank, candidate in enumerate(frame.get("detections", []), 1):
            box = candidate.get("bbox_xyxy") or candidate.get("bbox")
            center = candidate.get("center")
            if isinstance(center, dict):
                center = [center.get("x"), center.get("y")]
            candidates.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "confidence": candidate.get("confidence"),
                    "bbox_xyxy": box,
                    "center": center or _center(box),
                    "rank": rank,
                }
            )
        result[int(frame_index)] = candidates
    return result


def _selected_track_by_frame(document: dict[str, Any] | None) -> dict[int, dict[str, Any]]:
    if not document:
        return {}
    points = (
        document.get("primary_track")
        or document.get("trajectory")
        or document.get("track_points")
        or []
    )
    if isinstance(points, dict):
        points = points.get("points", [])
    result = {}
    for point in points:
        frame_index = point.get("frame_index")
        if frame_index is None:
            continue
        x = point.get("x")
        y = point.get("y")
        center = point.get("center")
        if isinstance(center, dict):
            x, y = center.get("x"), center.get("y")
        elif isinstance(center, list) and len(center) >= 2:
            x, y = center[:2]
        result[int(frame_index)] = {
            "candidate_id": point.get("candidate_id"),
            "confidence": point.get("confidence"),
            "center": [x, y] if x is not None and y is not None else None,
            "bbox_xyxy": point.get("bbox_xyxy") or point.get("bbox"),
            "provenance": point.get("provenance"),
        }
    return result


def _compatibility(
    analysis_dir: Path,
    detections: dict[str, Any] | None,
    tracking: dict[str, Any] | None,
    frozen_release_class: str | None = None,
) -> dict[str, Any]:
    raw = next((analysis_dir / "raw").glob("*"), None) if (analysis_dir / "raw").is_dir() else None
    modern_detection = bool(
        detections
        and isinstance(detections.get("frames"), list)
        and (detections.get("settings") or {}).get("confidence_threshold") is not None
    )
    top_k = bool(
        modern_detection
        and any(frame.get("detections") for frame in detections.get("frames", []))
    )
    selected = _selected_track_by_frame(tracking)
    persisted_model_path = str((detections or {}).get("model_path_used") or "")
    e4c_model = get_ball_detector_model("e4c_best_overall")
    model_comparable = Path(persisted_model_path).name == e4c_model.model_file
    return {
        "raw_original_video_available": bool(raw and raw.is_file()),
        "modern_detection_json_available": modern_detection,
        "top_k_persistence_available": top_k,
        "primary_tracking_valid": bool(selected),
        "persisted_detector_model_path": persisted_model_path or None,
        "raw_audit_detector_model_path": str(e4c_model.path),
        "raw_e4c_to_persisted_model_comparable": model_comparable,
        "frozen_release_input_class": frozen_release_class,
        "frozen_release_input_compatible": (
            frozen_release_class
            != "malformed_or_incompatible_historical_tracking_input"
        ),
    }


def _frozen_release_input_compatibility() -> dict[str, str]:
    path = (
        VALIDATION_ROOT
        / "release_v1_3_vs_observation_recovery_comparison.json"
    )
    document = _read_optional_json(path) or {}
    return {
        row["validation_id"]: row.get("input_compatibility_class")
        for row in document.get("per_delivery_comparison", [])
        if row.get("validation_id")
    }


def _best_match(
    candidates: list[dict[str, Any]],
    point: list[float] | None,
    bbox: list[float] | None,
) -> dict[str, Any] | None:
    matched = [
        candidate
        for candidate in candidates
        if _candidate_matches(candidate, point, bbox)
    ]
    if not matched:
        return None
    return min(matched, key=lambda row: _match_distance(row, point, bbox))


def _candidate_matches(
    candidate: dict[str, Any] | None,
    point: list[float] | None,
    bbox: list[float] | None,
) -> bool:
    if not candidate or (not point and not bbox):
        return False
    candidate_box = candidate.get("bbox_xyxy")
    candidate_center = candidate.get("center") or _center(candidate_box)
    if bbox and candidate_box and _iou(candidate_box, bbox) >= 0.1:
        return True
    if point and candidate_box and _point_in_box(point, _expand(candidate_box, 4)):
        return True
    if point and candidate_center:
        return math.dist(map(float, point), map(float, candidate_center)) <= MATCH_RADIUS_PX
    return False


def _match_distance(
    candidate: dict[str, Any],
    point: list[float] | None,
    bbox: list[float] | None,
) -> float:
    target = point or _center(bbox)
    center = candidate.get("center") or _center(candidate.get("bbox_xyxy"))
    return math.dist(map(float, target), map(float, center)) if target and center else math.inf


def _disappearance_reason(
    raw: dict[str, Any] | None,
    persisted: dict[str, Any] | None,
    selected: bool,
) -> str | None:
    if raw is None:
        return "not_produced_by_yolo_above_diagnostic_floor"
    if persisted is None:
        confidence = float(raw.get("confidence") or 0)
        rank = int(raw.get("rank") or 0)
        if confidence < PRODUCTION_THRESHOLD:
            return "production_confidence_threshold"
        if rank > PRODUCTION_MAX_DETECTIONS:
            return "production_max_det_or_ranking"
        return "nms_class_geometry_or_serialization_boundary"
    if not selected:
        return "primary_tracker_rejected_or_not_associated"
    return None


def _write_debug_frames(
    *,
    release: dict[str, Any],
    raw_clip: dict[str, Any],
    truth_by_frame: dict[int, dict[str, Any]],
    persisted_by_frame: dict[int, list[dict[str, Any]]],
    selected_by_frame: dict[int, dict[str, Any]],
    frame_rows: list[dict[str, Any]],
) -> None:
    raw_path = Path(raw_clip["raw_video_path"])
    output = OUTPUT_ROOT / release["validation_id"] / "debug_frames"
    output.mkdir(parents=True, exist_ok=True)
    row_by_frame = {row["frame_index"]: row for row in frame_rows}
    capture = cv2.VideoCapture(str(raw_path))
    try:
        for raw_frame in raw_clip["frames"]:
            frame_index = int(raw_frame["frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, image = capture.read()
            if not ok:
                continue
            for candidate in raw_frame["candidates"]:
                _draw_box(image, candidate, (180, 180, 180), f"R{candidate['rank']} {candidate['confidence']:.3f}")
            for candidate in persisted_by_frame.get(frame_index, []):
                _draw_box(image, candidate, (0, 180, 255), f"P{candidate['rank']}")
            selected = selected_by_frame.get(frame_index)
            if selected:
                _draw_box(image, selected, (0, 255, 0), "TRACK")
            truth = truth_by_frame.get(frame_index)
            if truth and truth.get("point"):
                x, y = map(round, truth["point"])
                cv2.drawMarker(image, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(image, "TRUE BALL", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            row = row_by_frame[frame_index]
            label = f"{release['validation_id']} f{frame_index} {row['comparison_category']}"
            cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 900), 30), (0, 0, 0), -1)
            cv2.putText(image, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(output / f"frame_{frame_index:06d}_audit.jpg"), image)
    finally:
        capture.release()


def _draw_box(
    image: Any,
    candidate: dict[str, Any],
    color: tuple[int, int, int],
    label: str,
) -> None:
    box = candidate.get("bbox_xyxy")
    if not box:
        center = candidate.get("center")
        if not center or None in center:
            return
        x, y = map(round, center)
        box = [x - 4, y - 4, x + 4, y + 4]
    x1, y1, x2, y2 = map(round, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _raw_candidate(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    box = [round(float(value), 3) for value in candidate["bbox_xyxy"]]
    return {
        "candidate_id": f"raw_rank_{rank:03d}",
        "rank": rank,
        "class_id": candidate["class_id"],
        "class_name": candidate["class_name"],
        "confidence": round(float(candidate["confidence"]), 6),
        "bbox_xyxy": box,
        "center": [round(value, 3) for value in _center(box)],
    }


def _raw_video_path(
    analysis_dir: Path, release: dict[str, Any]
) -> Path | None:
    reference = release.get("video_reference")
    if reference and Path(reference).is_file():
        return Path(reference)
    raw_dir = analysis_dir / "raw"
    return next((path for path in raw_dir.glob("*") if path.is_file()), None)


def _release_annotations() -> list[dict[str, Any]]:
    document = _read_json(ANNOTATIONS_PATH)
    return [
        row
        for row in document.get("annotations", [])
        if row.get("annotation_status") == "labeled"
        and row.get("human_release_frame") is not None
    ]


def _method_document() -> dict[str, Any]:
    return {
        "diagnostic_window": f"human release frame +/- {WINDOW_RADIUS} frames",
        "raw_inference": {
            "model": get_ball_detector_model("e4c_best_overall").model_file,
            "confidence_floor": DIAGNOSTIC_FLOOR,
            "inference_size": INFERENCE_SIZE,
            "max_detections": 300,
        },
        "production_reference": {
            "confidence_threshold": PRODUCTION_THRESHOLD,
            "inference_size": INFERENCE_SIZE,
            "max_detections": PRODUCTION_MAX_DETECTIONS,
        },
        "spatial_match": (
            f"IoU >= 0.1, point inside box expanded by 4 px, or center within "
            f"{MATCH_RADIUS_PX:g} px"
        ),
        "thresholds": [*THRESHOLDS, PRODUCTION_THRESHOLD],
    }


def _classification_explanation(
    category: str,
    evaluated: list[dict[str, Any]],
    compatibility: dict[str, Any],
) -> str:
    if category == "UNKNOWN":
        return "No explicit spatial true-ball annotations are available."
    if category == "HISTORICAL_INPUT_INCOMPATIBLE":
        return f"Historical artifacts are incomplete or incompatible: {compatibility}."
    counts = Counter(row["comparison_category"] for row in evaluated)
    return f"{len(evaluated)} visible spatially annotated frames; comparison counts={dict(counts)}."


def _summary_markdown(
    aggregate: dict[str, Any], clips: list[dict[str, Any]]
) -> str:
    lines = [
        "# Release-Region Detector Evidence Audit V1",
        "",
        "This is an offline diagnostic. Human release frames and ball locations are not production inputs.",
        "",
        "## Aggregate",
        "",
        f"- Clips: {aggregate['audit_scope']['clip_count']}",
        f"- Evaluable clips: {aggregate['evaluable_clip_count']}",
        f"- Spatially evaluated visible frames: {aggregate['spatially_evaluated_visible_frame_count']}",
        f"- Raw YOLO true-ball recall: {_pct(aggregate['raw_yolo_true_ball_recall'])}",
        f"- Persisted-candidate true-ball recall: {_pct(aggregate['persisted_candidate_true_ball_recall'])}",
        f"- Primary-track true-ball recall: {_pct(aggregate['primary_track_true_ball_recall'])}",
        f"- Raw detections lost during persistence: "
        f"{aggregate['real_ball_detections_lost_by_persistence'] if aggregate['persistence_loss_attribution_available'] else 'unavailable (historical persisted model differs)'}",
        f"- Visible true-ball frames never detected: {aggregate['real_ball_frames_never_detected_by_yolo']}",
        "",
        "## Classification",
        "",
    ]
    for category, count in aggregate["failure_category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Per Clip", ""])
    for clip in clips:
        lines.append(
            f"- {clip['validation_id']}: {clip['dominant_evidence_bottleneck']} "
            f"(evaluated frames={clip['spatially_evaluated_visible_frames']}, "
            f"raw={_pct(clip['raw_detector_recall'])}, "
            f"persisted={_pct(clip['persisted_candidate_recall'])}, "
            f"track={_pct(clip['primary_track_recall'])})"
        )
    lines.extend(["", f"> {aggregate['measurement_limitation']}", ""])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "validation_id", "analysis_id", "frame_index", "timestamp_seconds",
        "spatial_ground_truth_available", "ball_visible", "true_ball_point",
        "raw_candidate_count_at_0_01", "true_ball_raw_confidence",
        "true_ball_raw_rank", "persisted_candidate_count",
        "primary_track_matches_true_ball", "comparison_category",
        "true_ball_disappearance_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key)
                    for key in fields
                }
            )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        return _read_json(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _center(box: list[float] | None) -> list[float] | None:
    if not box or len(box) < 4:
        return None
    return [(float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2]


def _expand(box: list[float], pixels: float) -> list[float]:
    return [box[0] - pixels, box[1] - pixels, box[2] + pixels, box[3] + pixels]


def _point_in_box(point: list[float], box: list[float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    area_right = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _pct(value: float | None) -> str:
    return "unavailable" if value is None else f"{value * 100:.2f}%"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
