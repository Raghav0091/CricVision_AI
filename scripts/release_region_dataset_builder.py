"""Build and summarize the local CricVision release-region dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ANALYSIS_ROOT = ROOT / "outputs" / "video_analysis"
DATASET_ROOT = ROOT / "datasets" / "release_region_v1"
MANIFEST_PATH = DATASET_ROOT / "manifest.json"
ANNOTATIONS_PATH = DATASET_ROOT / "annotations.json"
SUMMARY_PATH = DATASET_ROOT / "quality_summary.json"
V13_PATH = ROOT / "outputs" / "release_validation" / "release_v1_3_results.json"

FULL_DIR = DATASET_ROOT / "full_frames"
BOWLER_DIR = DATASET_ROOT / "bowler_rois"
HAND_DIR = DATASET_ROOT / "hand_rois"
SOURCE_DIR = DATASET_ROOT / "sources"
META_DIR = DATASET_ROOT / "metadata"

ANNOTATION_ENUMS = {
    "ball_visible": ["unlabeled", "yes", "no", "uncertain"],
    "visibility_confidence": ["unlabeled", "high", "medium", "low"],
    "blur": ["unlabeled", "none", "mild", "moderate", "severe"],
    "hand_relationship": [
        "unlabeled", "in_hand", "overlapping_hand", "emerging_from_hand",
        "separated", "unknown",
    ],
    "occlusion": ["unlabeled", "none", "partial", "heavy"],
    "contrast": ["unlabeled", "high", "medium", "low"],
    "background_difficulty": ["unlabeled", "easy", "moderate", "hard"],
    "hard_negative_label": [
        "unlabeled", "pitch_highlight", "shoe", "sock", "stump", "clothing",
        "watermark", "static_background", "other", "unknown",
    ],
    "release_relative_phase": [
        "unlabeled", "before_release", "likely_release", "early_free_flight",
        "later_free_flight", "unknown",
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--max-sources", type=int)
    build.add_argument("--force", action="store_true")
    suggestions = sub.add_parser("suggestions")
    suggestions.add_argument("--models", default="e4c")
    suggestions.add_argument("--device", default=None)
    sub.add_parser("summary")
    sub.add_parser("audit")
    args = parser.parse_args()
    if args.command == "audit":
        manifest = discover_sources()
        _write_json(MANIFEST_PATH, manifest)
        print(json.dumps(manifest["summary"], indent=2))
    elif args.command == "build":
        manifest = build_dataset(max_sources=args.max_sources, force=args.force)
        print(json.dumps(manifest["summary"], indent=2))
    elif args.command == "suggestions":
        print(
            json.dumps(
                add_detector_suggestions(
                    [item.strip() for item in args.models.split(",") if item.strip()],
                    device=args.device,
                ),
                indent=2,
            )
        )
    else:
        print(json.dumps(write_quality_summary(), indent=2))
    return 0


def discover_sources() -> dict[str, Any]:
    v13 = _v13_by_analysis()
    records = []
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for analysis_dir in sorted(ANALYSIS_ROOT.glob("analysis_*")):
        raw_dir = analysis_dir / "raw"
        raw = next((p for p in raw_dir.glob("*") if p.is_file()), None)
        if raw is None:
            continue
        video = _video_info(raw)
        digest = sha256_file(raw)
        hash_groups[digest].append(analysis_dir.name)
        release = _release_evidence(analysis_dir, video, v13.get(analysis_dir.name))
        pose = _pose_doc(analysis_dir)
        tracking = _read_optional(analysis_dir / "tracking" / "tracking_result.json")
        detections = _read_optional(analysis_dir / "detections" / "detections.json")
        records.append(
            {
                "analysis_id": analysis_dir.name,
                "original_video_path": str(raw),
                "sha256": digest,
                **video,
                "release_region": release,
                "pose_available": bool(pose and _poses(pose)),
                "tracking_available": bool(_track_points(tracking)),
                "detector_model_originally_used": (
                    (detections or {}).get("model_path_used")
                    or ((detections or {}).get("detector") or {}).get("model_file")
                ),
            }
        )
    records_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_hash[record["sha256"]].append(record)
    canonical_by_hash = {
        digest: max(group, key=_canonical_source_score)["analysis_id"]
        for digest, group in records_by_hash.items()
    }
    for record in records:
        canonical = canonical_by_hash[record["sha256"]]
        record["canonical_analysis_id"] = canonical
        record["is_duplicate"] = record["analysis_id"] != canonical
        record["duplicate_group"] = hash_groups[record["sha256"]]
    duplicate_groups = [ids for ids in hash_groups.values() if len(ids) > 1]
    return {
        "schema_version": "1.0",
        "created_at": _now(),
        "source_policy": "Clean files under outputs/video_analysis/<analysis_id>/raw only.",
        "summary": {
            "source_videos_discovered": len(records),
            "independent_unique_deliveries": len(hash_groups),
            "duplicate_source_count": sum(len(ids) - 1 for ids in duplicate_groups),
            "duplicate_group_count": len(duplicate_groups),
        },
        "duplicate_groups": duplicate_groups,
        "sources": records,
        "sequences": [],
        "frames": [],
    }


def _canonical_source_score(record: dict[str, Any]) -> tuple[int, int, int, str]:
    release_score = {
        "release_estimator_v1_3": 3,
        "saved_release_result": 2,
        "first_track_frame_lead": 1,
        "peak_bowling_wrist_motion": 1,
        "video_midpoint_fallback": 0,
    }.get(record["release_region"]["source"], 0)
    return (
        int(record["pose_available"]),
        int(record["tracking_available"]),
        release_score,
        record["analysis_id"],
    )


def build_dataset(*, max_sources: int | None = None, force: bool = False) -> dict[str, Any]:
    manifest = discover_sources()
    for directory in (FULL_DIR, BOWLER_DIR, HAND_DIR, SOURCE_DIR, META_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    unique = [row for row in manifest["sources"] if not row["is_duplicate"]]
    if max_sources is not None:
        unique = unique[:max_sources]
    annotations = _load_annotations()
    annotation_by_id = {
        row["sample_id"]: row for row in annotations.get("annotations", [])
    }
    extracted = []
    sequences = []
    for source in unique:
        sequence_id = f"seq_{source['analysis_id']}"
        indices = sample_window(
            source["release_region"]["center_frame"],
            source["fps"],
            source["frame_count"],
            source["release_region"]["confidence"],
        )
        sequences.append(
            {
                "sequence_id": sequence_id,
                "analysis_id": source["analysis_id"],
                "frame_indices": [item["frame_index"] for item in indices],
                "release_region": source["release_region"],
            }
        )
        pose_doc = _pose_doc(ANALYSIS_ROOT / source["analysis_id"])
        capture = cv2.VideoCapture(source["original_video_path"])
        try:
            for sampled in indices:
                frame_index = sampled["frame_index"]
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                sample_id = f"{source['analysis_id']}_f{frame_index:06d}"
                full_path = FULL_DIR / f"{sample_id}.jpg"
                if force or not full_path.is_file():
                    cv2.imwrite(str(full_path), frame)
                bowler_box, bowler_source = bowler_crop_box(
                    pose_doc, frame_index, source["width"], source["height"]
                )
                bowler_path = BOWLER_DIR / f"{sample_id}.jpg"
                if force or not bowler_path.is_file():
                    cv2.imwrite(str(bowler_path), crop_image(frame, bowler_box))
                hand_box, wrist = hand_crop_box(
                    pose_doc, frame_index, source["width"], source["height"], bowler_box
                )
                hand_path = None
                if hand_box is not None:
                    hand_path = HAND_DIR / f"{sample_id}.jpg"
                    if force or not hand_path.is_file():
                        cv2.imwrite(str(hand_path), crop_image(frame, hand_box))
                suggestions = _persisted_suggestions(source["analysis_id"], frame_index)
                phase = phase_suggestion(
                    frame_index, source["release_region"]["center_frame"], source["fps"]
                )
                metadata = {
                    "sample_id": sample_id,
                    "sequence_id": sequence_id,
                    "analysis_id": source["analysis_id"],
                    "source_frame_index": frame_index,
                    "timestamp_seconds": round(frame_index / source["fps"], 6),
                    "sampling_mode": sampled["sampling_mode"],
                    "original_dimensions": [source["width"], source["height"]],
                    "full_frame_path": str(full_path.relative_to(ROOT)),
                    "bowler_roi_path": str(bowler_path.relative_to(ROOT)),
                    "bowler_crop_xyxy": bowler_box,
                    "bowler_crop_source": bowler_source,
                    "hand_roi_path": (
                        str(hand_path.relative_to(ROOT)) if hand_path else None
                    ),
                    "hand_crop_xyxy": hand_box,
                    "hand_roi_available": hand_box is not None,
                    "wrist_evidence": wrist,
                    "automatic_suggestions": {
                        "release_relative_phase": phase,
                        "historical_persisted_detector_candidates": suggestions,
                        "offline_detector_candidates": [],
                        "hard_negative_candidates": [
                            {**item, "hard_negative_candidate": True}
                            for item in suggestions
                        ],
                    },
                    "ground_truth": None,
                }
                _write_json(META_DIR / f"{sample_id}.json", metadata)
                extracted.append(metadata)
                annotation_by_id.setdefault(sample_id, annotation_template(metadata))
        finally:
            capture.release()
        _write_json(
            SOURCE_DIR / f"{source['analysis_id']}.json",
            source,
        )
    current_ids = {row["sample_id"] for row in extracted}
    annotations["annotations"] = sorted(
        (
            annotation_by_id[sample_id]
            for sample_id in current_ids
        ),
        key=lambda row: row["sample_id"],
    )
    annotations["updated_at"] = _now()
    atomic_write_json(ANNOTATIONS_PATH, annotations)
    manifest["sequences"] = sequences
    manifest["frames"] = extracted
    manifest["summary"].update(
        {
            "extracted_sequence_count": len(sequences),
            "extracted_frame_count": len(extracted),
            "full_frame_count": len(extracted),
            "bowler_roi_count": sum(bool(row["bowler_roi_path"]) for row in extracted),
            "hand_roi_count": sum(row["hand_roi_available"] for row in extracted),
        }
    )
    if force:
        _prune_generated_files(current_ids)
    _write_json(MANIFEST_PATH, manifest)
    write_quality_summary()
    return manifest


def add_detector_suggestions(
    model_names: list[str], *, device: str | None = None
) -> dict[str, Any]:
    from services.api.services.ball_detection_clip import extract_ball_candidates
    from services.api.services.ball_detector_registry import load_ball_detector_model

    aliases = {
        "e4c": "e4c_best_overall",
        "e3": "e3_motion_blur",
    }
    manifest = _read_optional(MANIFEST_PATH)
    annotations = _load_annotations()
    if not manifest:
        raise RuntimeError("Build the dataset before generating suggestions.")
    annotations_by_id = {
        row["sample_id"]: row for row in annotations["annotations"]
    }
    processed = Counter()
    for alias in model_names:
        model_key = aliases.get(alias)
        if model_key is None:
            raise ValueError(f"Unsupported suggestion model: {alias}")
        selected, model = load_ball_detector_model(model_key)
        for metadata in manifest["frames"]:
            image = cv2.imread(str(ROOT / metadata["full_frame_path"]))
            results = model.predict(
                source=image,
                imgsz=960,
                conf=0.01,
                max_det=100,
                device=device,
                verbose=False,
            )
            candidates = extract_ball_candidates(
                results, getattr(model, "names", {}), strict=True
            )
            candidates.sort(key=lambda row: row["confidence"], reverse=True)
            suggestions = [
                {
                    "model_key": model_key,
                    "model_path": str(selected.path),
                    "confidence": round(float(candidate["confidence"]), 6),
                    "bbox_xyxy": [
                        round(float(value), 3)
                        for value in candidate["bbox_xyxy"]
                    ],
                    "rank": rank,
                    "suggestion_only": True,
                }
                for rank, candidate in enumerate(candidates, 1)
            ]
            automatic = metadata["automatic_suggestions"]
            prior = [
                row
                for row in automatic.get("offline_detector_candidates", [])
                if row.get("model_key") != model_key
            ]
            automatic["offline_detector_candidates"] = prior + suggestions
            automatic["hard_negative_candidates"] = [
                {
                    **row,
                    "hard_negative_candidate": True,
                    "manual_confirmation_required": True,
                }
                for row in automatic["offline_detector_candidates"]
            ]
            _write_json(META_DIR / f"{metadata['sample_id']}.json", metadata)
            annotation = annotations_by_id[metadata["sample_id"]]
            annotation["automatic_suggestions"] = automatic
            processed[alias] += 1
    manifest["detector_suggestion_settings"] = {
        "diagnostic_only": True,
        "confidence_floor": 0.01,
        "imgsz": 960,
        "models": model_names,
    }
    _write_json(MANIFEST_PATH, manifest)
    atomic_write_json(ANNOTATIONS_PATH, annotations)
    return {
        "frames_processed_by_model": dict(processed),
        "ground_truth_labels_modified": 0,
    }


def _prune_generated_files(current_ids: set[str]) -> None:
    for directory, suffix in (
        (FULL_DIR, ".jpg"),
        (BOWLER_DIR, ".jpg"),
        (HAND_DIR, ".jpg"),
        (META_DIR, ".json"),
    ):
        for path in directory.glob(f"*{suffix}"):
            if path.stem not in current_ids:
                path.unlink()


def sample_window(center: int, fps: float, frame_count: int, confidence: str) -> list[dict[str, Any]]:
    dense_radius = max(5, min(15, round(fps * (0.25 if confidence == "low" else 0.18))))
    context_radius = max(12, min(36, round(fps * (0.65 if confidence == "low" else 0.45))))
    sparse_step = max(2, round(fps * 0.1))
    start, end = max(0, center - context_radius), min(frame_count - 1, center + context_radius)
    dense_start, dense_end = max(start, center - dense_radius), min(end, center + dense_radius)
    selected = {
        frame: "dense_release_window"
        for frame in range(dense_start, dense_end + 1)
    }
    for frame in range(start, end + 1, sparse_step):
        selected.setdefault(frame, "sparse_context_window")
    selected.setdefault(start, "sparse_context_window")
    selected.setdefault(end, "sparse_context_window")
    return [
        {"frame_index": frame, "sampling_mode": mode}
        for frame, mode in sorted(selected.items())
    ]


def bowler_crop_box(
    pose_doc: dict[str, Any] | None,
    frame_index: int,
    width: int,
    height: int,
) -> tuple[list[int], str]:
    pose = _pose_at(pose_doc, frame_index)
    box = (pose or {}).get("bbox_xyxy")
    if box:
        x1, y1, x2, y2 = box
        margin_x, margin_y = (x2 - x1) * 0.35, (y2 - y1) * 0.2
        return clamp_box([x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y], width, height), "pose_bbox"
    return [0, 0, width, height], "fallback_full_frame_action_roi"


def hand_crop_box(
    pose_doc: dict[str, Any] | None,
    frame_index: int,
    width: int,
    height: int,
    bowler_box: list[int],
) -> tuple[list[int] | None, dict[str, Any] | None]:
    pose = _pose_at(pose_doc, frame_index)
    arm = ((pose_doc or {}).get("bowling_arm") or {}).get("bowling_arm")
    if arm not in {"left", "right"} or not pose:
        return None, None
    wrist = (pose.get("keypoints") or {}).get(f"{arm}_wrist")
    if not wrist or float(wrist.get("confidence") or 0) < 0.15:
        return None, wrist
    side = max(128, min(320, round(max(bowler_box[2] - bowler_box[0], bowler_box[3] - bowler_box[1]) * 0.55)))
    x, y = float(wrist["x"]), float(wrist["y"])
    return clamp_box([x - side / 2, y - side / 2, x + side / 2, y + side / 2], width, height), {
        "keypoint": f"{arm}_wrist", "x": x, "y": y,
        "confidence": float(wrist["confidence"]), "bowling_arm": arm,
    }


def clamp_box(box: list[float], width: int, height: int) -> list[int]:
    x1 = max(0, min(width - 1, math.floor(box[0])))
    y1 = max(0, min(height - 1, math.floor(box[1])))
    x2 = max(x1 + 1, min(width, math.ceil(box[2])))
    y2 = max(y1 + 1, min(height, math.ceil(box[3])))
    return [x1, y1, x2, y2]


def crop_image(frame: Any, box: list[int]) -> Any:
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2]


def annotation_template(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": metadata["sample_id"],
        "sequence_id": metadata["sequence_id"],
        "analysis_id": metadata["analysis_id"],
        "source_frame_index": metadata["source_frame_index"],
        "status": "unlabeled",
        "ball_visible": "unlabeled",
        "ball_bbox_xyxy": None,
        "ball_center": None,
        "visibility_confidence": "unlabeled",
        "ball_size_px": None,
        "blur": "unlabeled",
        "hand_relationship": "unlabeled",
        "occlusion": "unlabeled",
        "contrast": "unlabeled",
        "background_difficulty": "unlabeled",
        "hard_negative_label": "unlabeled",
        "bowling_arm": (
            (metadata.get("wrist_evidence") or {}).get("bowling_arm") or "unknown"
        ),
        "release_relative_phase": "unlabeled",
        "notes": "",
        "automatic_suggestions": metadata["automatic_suggestions"],
        "annotated_at": None,
    }


def validate_annotation(annotation: dict[str, Any]) -> None:
    for field, values in ANNOTATION_ENUMS.items():
        if annotation.get(field) not in values:
            raise ValueError(f"Invalid {field}: {annotation.get(field)}")
    box = annotation.get("ball_bbox_xyxy")
    if box is not None and (len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError("ball_bbox_xyxy must be a positive xyxy box")
    if annotation["ball_visible"] in {"no", "uncertain"} and box is not None:
        raise ValueError("No-ball/uncertain annotations cannot carry ground-truth boxes")


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_quality_summary() -> dict[str, Any]:
    manifest = _read_optional(MANIFEST_PATH) or {"summary": {}, "frames": []}
    annotations = (_read_optional(ANNOTATIONS_PATH) or {}).get("annotations", [])
    labeled = [row for row in annotations if row.get("status") == "complete"]
    summary = {
        "schema_version": "1.0",
        "created_at": _now(),
        **manifest.get("summary", {}),
        "annotation_counts": dict(Counter(row.get("ball_visible", "unlabeled") for row in annotations)),
        "completed_annotations": len(labeled),
        "distributions": {
            field: dict(Counter(row.get(field, "unlabeled") for row in labeled))
            for field in (
                "blur", "hand_relationship", "occlusion", "contrast",
                "background_difficulty", "bowling_arm", "release_relative_phase",
            )
        },
        "ball_size_buckets": dict(Counter(_ball_size_bucket(row.get("ball_size_px")) for row in labeled if row.get("ball_size_px") is not None)),
        "source_delivery_counts": dict(Counter(row["analysis_id"] for row in labeled)),
        "gaps": _dataset_gaps(labeled),
        "collection_target": {
            "independent_deliveries": "100-200",
            "useful_targeted_frames": "1000-4000",
            "priority": [
                "independent left/right arm and fast/spin deliveries",
                "severe blur plus low contrast and hand overlap",
                "lighting, camera, pitch, ball, clothing, and background diversity",
                "intentional hard negatives",
            ],
        },
    }
    _write_json(SUMMARY_PATH, summary)
    return summary


def phase_suggestion(frame: int, center: int, fps: float) -> str:
    delta = frame - center
    if delta < -max(1, round(fps * 0.05)):
        return "before_release"
    if abs(delta) <= max(1, round(fps * 0.05)):
        return "likely_release"
    if delta <= max(2, round(fps * 0.2)):
        return "early_free_flight"
    return "later_free_flight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_evidence(analysis_dir: Path, video: dict[str, Any], v13: dict[str, Any] | None) -> dict[str, Any]:
    if v13 and v13.get("prediction_status") == "ready":
        return {"center_frame": int(v13["predicted_release_frame"]), "source": "release_estimator_v1_3", "confidence": "high"}
    release = _read_optional(analysis_dir / "reports" / "release_point_v1.json")
    result = (release or {}).get("result") or {}
    if result.get("release_frame") is not None:
        return {"center_frame": int(result["release_frame"]), "source": "saved_release_result", "confidence": "medium"}
    track = _track_points(_read_optional(analysis_dir / "tracking" / "tracking_result.json"))
    if track:
        first = int(track[0]["frame_index"])
        return {"center_frame": max(0, first - round(video["fps"] * 0.12)), "source": "first_track_frame_lead", "confidence": "low"}
    pose = _pose_doc(analysis_dir)
    action = _peak_wrist_frame(pose)
    if action is not None:
        return {"center_frame": action, "source": "peak_bowling_wrist_motion", "confidence": "low"}
    return {"center_frame": video["frame_count"] // 2, "source": "video_midpoint_fallback", "confidence": "low"}


def _peak_wrist_frame(pose_doc: dict[str, Any] | None) -> int | None:
    arm = ((pose_doc or {}).get("bowling_arm") or {}).get("bowling_arm")
    if arm not in {"left", "right"}:
        return None
    points = []
    for key, pose in _poses(pose_doc).items():
        wrist = (pose.get("keypoints") or {}).get(f"{arm}_wrist")
        if wrist and float(wrist.get("confidence") or 0) >= 0.15:
            points.append((int(key), float(wrist["x"]), float(wrist["y"])))
    if len(points) < 2:
        return None
    return max(
        range(1, len(points)),
        key=lambda i: math.dist(points[i - 1][1:], points[i][1:]),
    ) and points[max(range(1, len(points)), key=lambda i: math.dist(points[i - 1][1:], points[i][1:]))][0]


def _pose_doc(analysis_dir: Path) -> dict[str, Any] | None:
    return _read_optional(analysis_dir / "reports" / "rtmpose_validation.json")


def _poses(document: dict[str, Any] | None) -> dict[str, Any]:
    return ((document or {}).get("bowler") or {}).get("poses_by_frame") or {}


def _pose_at(document: dict[str, Any] | None, frame: int) -> dict[str, Any] | None:
    poses = _poses(document)
    exact = poses.get(str(frame))
    if exact:
        return exact
    if not poses:
        return None
    nearest = min((int(key) for key in poses), key=lambda value: abs(value - frame))
    return poses.get(str(nearest)) if abs(nearest - frame) <= 2 else None


def _track_points(document: dict[str, Any] | None) -> list[dict[str, Any]]:
    points = (document or {}).get("primary_track") or []
    return sorted(points, key=lambda row: row.get("frame_index", 0)) if isinstance(points, list) else []


def _persisted_suggestions(analysis_id: str, frame: int) -> list[dict[str, Any]]:
    document = _read_optional(ANALYSIS_ROOT / analysis_id / "detections" / "detections.json") or {}
    record = next((row for row in document.get("frames", []) if row.get("frame_index") == frame), None)
    return [
        {
            "model": document.get("model_path_used"),
            "confidence": candidate.get("confidence"),
            "bbox_xyxy": candidate.get("bbox_xyxy"),
            "rank": rank,
            "candidate_id": candidate.get("candidate_id"),
            "suggestion_only": True,
        }
        for rank, candidate in enumerate((record or {}).get("detections", []), 1)
    ]


def _video_info(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    return {"fps": round(fps, 6), "width": width, "height": height, "frame_count": frames, "duration_seconds": round(frames / fps, 6) if fps else None}


def _v13_by_analysis() -> dict[str, dict[str, Any]]:
    doc = _read_optional(V13_PATH) or {}
    return {row["analysis_id"]: row for row in doc.get("records", []) if row.get("analysis_id")}


def _load_annotations() -> dict[str, Any]:
    return _read_optional(ANNOTATIONS_PATH) or {
        "schema_version": "1.0",
        "annotation_enums": ANNOTATION_ENUMS,
        "annotations": [],
    }


def _dataset_gaps(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No completed manual annotations yet."]
    gaps = []
    for field, value in (("bowling_arm", "left"), ("blur", "severe"), ("hand_relationship", "overlapping_hand")):
        if sum(row.get(field) == value for row in rows) < 10:
            gaps.append(f"Fewer than 10 completed examples for {field}={value}.")
    return gaps


def _ball_size_bucket(size: float) -> str:
    if size <= 6: return "<=6"
    if size <= 8: return "6-8"
    if size <= 10: return "8-10"
    if size <= 12: return "10-12"
    if size <= 16: return "12-16"
    return ">16"


def _read_optional(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
