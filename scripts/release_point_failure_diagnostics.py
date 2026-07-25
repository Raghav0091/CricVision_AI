"""Generate Release Point V1 failure diagnostics without changing estimation logic."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Backends.src.release_point.features import (  # noqa: E402
    Keypoint,
    ReleaseCandidate,
    ReleasePointConfig,
    arm_geometry,
    body_scale_px,
    extract_release_features,
    keypoint_velocity,
    parse_bowler_pose_sequence,
    parse_detection_observations,
    parse_track_observations,
)


DEFAULT_ANALYSIS_ID = "analysis_20260718_065149_af258b"
DEFAULT_OUTPUT = Path("outputs/release_validation/failure_analysis/rv1_002")
FRAME_START = 45
FRAME_END = 70
TRUE_RELEASE_FRAME = 52
PREDICTED_RELEASE_FRAME = 63
UNAVAILABLE = "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a per-frame Release Point V1 diagnostic package."
    )
    parser.add_argument(
        "--all-clips",
        action="store_true",
        help="Classify all manually labelled Release V1 validation clips.",
    )
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID)
    parser.add_argument("--true-frame", type=int, default=TRUE_RELEASE_FRAME)
    parser.add_argument("--predicted-frame", type=int, default=PREDICTED_RELEASE_FRAME)
    parser.add_argument("--start-frame", type=int, default=FRAME_START)
    parser.add_argument("--end-frame", type=int, default=FRAME_END)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--classification-output-dir",
        type=Path,
        default=Path("outputs/release_validation/failure_analysis/all_clips"),
    )
    args = parser.parse_args()
    if args.all_clips:
        return classify_all_clips(args.classification_output_dir)

    analysis_dir = REPO_ROOT / "outputs" / "video_analysis" / args.analysis_id
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(analysis_dir / "reports" / "analysis_metadata.json")
    detections_doc = _read_json(analysis_dir / "detections" / "detections.json")
    tracking_doc = _read_json(analysis_dir / "tracking" / "tracking_result.json")
    release_doc = _read_json(analysis_dir / "reports" / "release_point_v1.json")
    rtmpose_doc = _read_json(analysis_dir / "reports" / "rtmpose_validation.json")
    calibration_doc = _read_optional_json(analysis_dir / "calibration" / "calibration.json")
    calibration_v2_doc = _read_optional_json(analysis_dir / "calibration" / "calibration_v2.json")

    bowler_doc = dict(rtmpose_doc.get("bowler") or {})
    if "provider" not in bowler_doc:
        bowler_doc["provider"] = rtmpose_doc.get("pose_provider") or {}
    if "quality_flags" not in bowler_doc:
        bowler_doc["quality_flags"] = list(
            release_doc.get("result", {}).get("quality_flags") or []
        )
    pose_sequence = parse_bowler_pose_sequence(bowler_doc)
    detections_by_frame = parse_detection_observations(detections_doc)
    primary_track = parse_track_observations(tracking_doc)
    track_by_frame = {point.frame_index: point for point in primary_track}
    score_by_frame = {
        int(score.get("frame_index")): score
        for score in release_doc.get("candidate_scores", [])
        if score.get("frame_index") is not None
    }
    raw_pose_by_frame = {
        int(frame): pose
        for frame, pose in (bowler_doc.get("poses_by_frame") or {}).items()
    }
    tracking_diagnostic_by_candidate = {
        item.get("candidate_id"): item
        for item in tracking_doc.get("candidate_diagnostics", [])
        if item.get("candidate_id")
    }

    config = ReleasePointConfig()
    rows: list[dict[str, Any]] = []
    for frame_index in range(args.start_frame, args.end_frame + 1):
        score = score_by_frame.get(frame_index)
        source = score.get("source") if score else "diagnostic_window"
        features = extract_release_features(
            ReleaseCandidate(frame_index=frame_index, source=source),
            detections_by_frame=detections_by_frame,
            primary_track=primary_track,
            pose_sequence=pose_sequence,
            config=config,
        )
        raw_pose = raw_pose_by_frame.get(frame_index) or {}
        pose = None if pose_sequence is None else pose_sequence.poses_by_frame.get(frame_index)
        scale = body_scale_px(pose)
        left_wrist = _keypoint_dict(None if pose is None else pose.keypoints.get("left_wrist"))
        right_wrist = _keypoint_dict(None if pose is None else pose.keypoints.get("right_wrist"))
        selected_ball = _selected_ball(features)
        exact_track = track_by_frame.get(frame_index)
        nearest_track = _nearest_track_dict(features, primary_track)
        all_candidates = [
            _ball_observation_dict(candidate, tracking_diagnostic_by_candidate)
            for candidate in detections_by_frame.get(frame_index, [])
        ]
        selected_wrist = (
            left_wrist
            if features.wrist_keypoint_name == "left_wrist"
            else right_wrist
            if features.wrist_keypoint_name == "right_wrist"
            else None
        )
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": _timestamp(frame_index, metadata),
                "is_human_true_release_frame": frame_index == args.true_frame,
                "is_predicted_release_frame": frame_index == args.predicted_frame,
                "all_ball_candidates": all_candidates,
                "selected_ball_candidate": selected_ball,
                "selected_ball_detector_confidence": features.detector_confidence
                if selected_ball
                else None,
                "selected_ball_candidate_rank": features.candidate_rank,
                "selected_ball_tracking_provenance": features.tracker_provenance,
                "exact_primary_track_point": _track_observation_dict(exact_track),
                "nearest_primary_track_point_used_by_features": nearest_track,
                "left_wrist": left_wrist,
                "right_wrist": right_wrist,
                "bowler_pose_person_id": raw_pose.get("person_id"),
                "bowler_pose_bbox_xyxy": raw_pose.get("bbox_xyxy"),
                "bowler_pose_keypoints": raw_pose.get("keypoints"),
                "selected_bowling_arm": _selected_bowling_arm(rtmpose_doc),
                "release_engine_selected_wrist": features.wrist_keypoint_name,
                "ball_to_left_wrist_distance_px": _distance(selected_ball, left_wrist),
                "ball_to_right_wrist_distance_px": _distance(selected_ball, right_wrist),
                "ball_to_selected_wrist_distance_px": features.ball_wrist_distance_px,
                "normalized_ball_to_left_wrist_distance": _normalized_distance(
                    selected_ball, left_wrist, scale
                ),
                "normalized_ball_to_right_wrist_distance": _normalized_distance(
                    selected_ball, right_wrist, scale
                ),
                "normalized_ball_to_selected_wrist_distance": (
                    features.normalized_ball_wrist_distance
                ),
                "separation_velocity": features.separation_velocity,
                "separation_persistence_frames": features.separation_persistence_frames,
                "selected_wrist_velocity": features.wrist_velocity,
                "selected_wrist_acceleration_proxy": features.wrist_acceleration_proxy,
                "left_wrist_velocity": _kp_velocity(pose_sequence, frame_index, "left_wrist"),
                "right_wrist_velocity": _kp_velocity(pose_sequence, frame_index, "right_wrist"),
                "arm_geometry_evidence": _arm_geometry_evidence(pose),
                "selected_arm_angle_degrees": features.arm_angle_degrees,
                "selected_arm_extension_proxy": features.bowling_arm_extension_proxy,
                "pose_confidence": None if pose is None else pose.confidence,
                "pose_keypoint_confidence": features.pose_keypoint_confidence,
                "forward_free_flight_points": features.forward_free_flight_points,
                "forward_free_flight_confirmation": features.forward_free_flight_confirmation,
                "backward_trajectory_fit_error_px": features.backward_trajectory_fit_error_px,
                "scene_roi_evidence": _scene_roi_evidence(
                    features,
                    detections_by_frame.get(frame_index, []),
                    exact_track,
                    calibration_doc,
                    calibration_v2_doc,
                ),
                "release_candidate_score": None if score is None else score.get("score"),
                "release_candidate_method": None if score is None else score.get("method"),
                "release_candidate_source": None if score is None else score.get("source"),
                "release_candidate_release_type": None
                if score is None
                else score.get("release_type"),
                "release_candidate_observed": None if score is None else score.get("observed"),
                "release_feature_scores": None
                if score is None
                else score.get("score_components"),
                "release_candidate_quality_flags": None
                if score is None
                else score.get("quality_flags"),
                "unavailable_evidence": _unavailable_evidence(
                    selected_ball=selected_ball,
                    exact_track=exact_track,
                    pose=pose,
                    score=score,
                    features=features,
                ),
            }
        )

    summary = _build_summary(
        args=args,
        analysis_dir=analysis_dir,
        metadata=metadata,
        detections_by_frame=detections_by_frame,
        tracking_doc=tracking_doc,
        release_doc=release_doc,
        rtmpose_doc=rtmpose_doc,
        rows=rows,
    )

    _write_json(output_dir / "frame_diagnostics.json", rows)
    _write_csv(output_dir / "frame_diagnostics.csv", rows)
    _write_json(output_dir / "diagnostic_summary.json", summary)
    visual = _write_visual_debug(
        analysis_dir=analysis_dir,
        metadata=metadata,
        rows=rows,
        output_dir=output_dir,
        true_frame=args.true_frame,
        predicted_frame=args.predicted_frame,
    )
    summary["visual_debug"] = visual
    _write_json(output_dir / "diagnostic_summary.json", summary)
    print(f"Wrote diagnostics to {output_dir}")
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_jsonable(data), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_index",
        "timestamp_seconds",
        "is_human_true_release_frame",
        "is_predicted_release_frame",
        "ball_candidate_count",
        "ball_candidate_ids",
        "selected_ball_candidate_id",
        "selected_ball_x",
        "selected_ball_y",
        "selected_ball_detector_confidence",
        "selected_ball_candidate_rank",
        "selected_ball_tracking_provenance",
        "exact_primary_track_provenance",
        "exact_primary_track_candidate_id",
        "left_wrist_x",
        "left_wrist_y",
        "left_wrist_confidence",
        "right_wrist_x",
        "right_wrist_y",
        "right_wrist_confidence",
        "selected_bowling_arm",
        "release_engine_selected_wrist",
        "ball_to_left_wrist_distance_px",
        "ball_to_right_wrist_distance_px",
        "ball_to_selected_wrist_distance_px",
        "normalized_ball_to_selected_wrist_distance",
        "separation_velocity",
        "separation_persistence_frames",
        "selected_wrist_velocity",
        "selected_wrist_acceleration_proxy",
        "selected_arm_angle_degrees",
        "selected_arm_extension_proxy",
        "pose_confidence",
        "pose_keypoint_confidence",
        "forward_free_flight_points",
        "forward_free_flight_confirmation",
        "backward_trajectory_fit_error_px",
        "scene_roi_consistency",
        "release_candidate_score",
        "release_candidate_method",
        "release_candidate_source",
        "release_feature_scores",
        "release_candidate_quality_flags",
        "unavailable_evidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            selected = row.get("selected_ball_candidate") or {}
            exact_track = row.get("exact_primary_track_point") or {}
            scene_roi = row.get("scene_roi_evidence") or {}
            writer.writerow(
                {
                    "frame_index": row.get("frame_index"),
                    "timestamp_seconds": row.get("timestamp_seconds"),
                    "is_human_true_release_frame": row.get(
                        "is_human_true_release_frame"
                    ),
                    "is_predicted_release_frame": row.get(
                        "is_predicted_release_frame"
                    ),
                    "ball_candidate_count": len(row.get("all_ball_candidates") or []),
                    "ball_candidate_ids": _csv_value(
                        [
                            item.get("candidate_id")
                            for item in row.get("all_ball_candidates") or []
                        ]
                    ),
                    "selected_ball_candidate_id": selected.get("candidate_id"),
                    "selected_ball_x": selected.get("x"),
                    "selected_ball_y": selected.get("y"),
                    "selected_ball_detector_confidence": row.get(
                        "selected_ball_detector_confidence"
                    ),
                    "selected_ball_candidate_rank": row.get(
                        "selected_ball_candidate_rank"
                    ),
                    "selected_ball_tracking_provenance": row.get(
                        "selected_ball_tracking_provenance"
                    ),
                    "exact_primary_track_provenance": exact_track.get("provenance"),
                    "exact_primary_track_candidate_id": exact_track.get("candidate_id"),
                    "left_wrist_x": (row.get("left_wrist") or {}).get("x"),
                    "left_wrist_y": (row.get("left_wrist") or {}).get("y"),
                    "left_wrist_confidence": (row.get("left_wrist") or {}).get(
                        "confidence"
                    ),
                    "right_wrist_x": (row.get("right_wrist") or {}).get("x"),
                    "right_wrist_y": (row.get("right_wrist") or {}).get("y"),
                    "right_wrist_confidence": (row.get("right_wrist") or {}).get(
                        "confidence"
                    ),
                    "selected_bowling_arm": _csv_value(
                        row.get("selected_bowling_arm")
                    ),
                    "release_engine_selected_wrist": row.get(
                        "release_engine_selected_wrist"
                    ),
                    "ball_to_left_wrist_distance_px": row.get(
                        "ball_to_left_wrist_distance_px"
                    ),
                    "ball_to_right_wrist_distance_px": row.get(
                        "ball_to_right_wrist_distance_px"
                    ),
                    "ball_to_selected_wrist_distance_px": row.get(
                        "ball_to_selected_wrist_distance_px"
                    ),
                    "normalized_ball_to_selected_wrist_distance": row.get(
                        "normalized_ball_to_selected_wrist_distance"
                    ),
                    "separation_velocity": row.get("separation_velocity"),
                    "separation_persistence_frames": row.get(
                        "separation_persistence_frames"
                    ),
                    "selected_wrist_velocity": row.get("selected_wrist_velocity"),
                    "selected_wrist_acceleration_proxy": row.get(
                        "selected_wrist_acceleration_proxy"
                    ),
                    "selected_arm_angle_degrees": row.get(
                        "selected_arm_angle_degrees"
                    ),
                    "selected_arm_extension_proxy": row.get(
                        "selected_arm_extension_proxy"
                    ),
                    "pose_confidence": row.get("pose_confidence"),
                    "pose_keypoint_confidence": row.get("pose_keypoint_confidence"),
                    "forward_free_flight_points": row.get(
                        "forward_free_flight_points"
                    ),
                    "forward_free_flight_confirmation": row.get(
                        "forward_free_flight_confirmation"
                    ),
                    "backward_trajectory_fit_error_px": row.get(
                        "backward_trajectory_fit_error_px"
                    ),
                    "scene_roi_consistency": scene_roi.get("scene_roi_consistency"),
                    "release_candidate_score": row.get("release_candidate_score"),
                    "release_candidate_method": row.get("release_candidate_method"),
                    "release_candidate_source": row.get("release_candidate_source"),
                    "release_feature_scores": _csv_value(
                        row.get("release_feature_scores")
                    ),
                    "release_candidate_quality_flags": _csv_value(
                        row.get("release_candidate_quality_flags")
                    ),
                    "unavailable_evidence": _csv_value(row.get("unavailable_evidence")),
                }
            )


def classify_all_clips(output_dir: Path) -> int:
    output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_doc = _read_json(
        REPO_ROOT / "outputs" / "release_validation" / "release_annotations.json"
    )
    baseline_doc = _read_json(
        REPO_ROOT / "outputs" / "release_validation" / "baseline_release_v1_results.json"
    )
    metrics_doc = _read_json(
        REPO_ROOT / "outputs" / "release_validation" / "release_point_v1_metrics.json"
    )
    baseline_by_analysis = {
        record.get("analysis_id"): record
        for record in baseline_doc.get("records", [])
        if record.get("analysis_id")
    }
    records = [
        _classify_clip(annotation, baseline_by_analysis.get(annotation.get("analysis_id")))
        for annotation in annotations_doc.get("annotations", [])
    ]
    summary = _aggregate_classification(records, metrics_doc)
    _write_json(output_dir / "failure_classification.json", records)
    _write_classification_csv(output_dir / "failure_classification.csv", records)
    _write_json(output_dir / "aggregate_failure_summary.json", summary)
    (output_dir / "aggregate_failure_summary.md").write_text(
        _summary_markdown(summary, records),
        encoding="utf-8",
    )
    print(f"Wrote all-clips failure classification to {output_dir}")
    return 0


def _classify_clip(
    annotation: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline = baseline or {}
    analysis_id = str(annotation.get("analysis_id"))
    validation_id = str(annotation.get("validation_id"))
    true_frame = _optional_int(annotation.get("human_release_frame"))
    predicted_frame = _optional_int(
        annotation.get("predicted_release_frame")
        if annotation.get("predicted_release_frame") is not None
        else baseline.get("predicted_release_frame")
    )
    analysis_dir = REPO_ROOT / "outputs" / "video_analysis" / analysis_id
    detections_doc = _read_optional_json(analysis_dir / "detections" / "detections.json") or {}
    tracking_doc = _read_optional_json(analysis_dir / "tracking" / "tracking_result.json") or {}
    release_doc = _read_optional_json(analysis_dir / "reports" / "release_point_v1.json") or {}
    rtmpose_doc = _read_optional_json(analysis_dir / "reports" / "rtmpose_validation.json") or {}

    detections_by_frame = parse_detection_observations(detections_doc)
    primary_track_raw = tracking_doc.get("primary_track") or []
    first_track_frame = _first_track_frame(primary_track_raw)
    tracking_status = tracking_doc.get("status")
    baseline_error = baseline.get("baseline_collection_error")
    baseline_status = baseline.get("baseline_collection_status")
    tracking_malformed = _tracking_malformed_for_release(
        tracking_doc=tracking_doc,
        baseline_error=baseline_error,
    )
    candidate_window = _candidate_window_summary(detections_by_frame, true_frame)
    rejected = _rejected_candidates_near_true(
        tracking_doc=tracking_doc,
        true_frame=true_frame,
        first_track_frame=first_track_frame,
    )
    track_window = _track_window_summary(primary_track_raw, true_frame)
    release_details = _release_details(
        release_doc=release_doc,
        rtmpose_doc=rtmpose_doc,
        true_frame=true_frame,
        predicted_frame=predicted_frame,
        detections_by_frame=detections_by_frame,
        tracking_doc=tracking_doc,
    )

    prediction_available = predicted_frame is not None
    absolute_error = (
        None
        if true_frame is None or predicted_frame is None
        else abs(predicted_frame - true_frame)
    )
    signed_error = (
        None
        if true_frame is None or predicted_frame is None
        else predicted_frame - true_frame
    )
    frames_to_track_start = (
        None
        if true_frame is None or first_track_frame is None
        else first_track_frame - true_frame
    )
    quality_flags = list(
        annotation.get("prediction_quality_flags")
        or baseline.get("quality_flags")
        or (release_doc.get("result", {}) or {}).get("quality_flags")
        or []
    )
    pose_status = (
        ((release_doc.get("result", {}) or {}).get("provenance") or {}).get("pose_status")
        or baseline.get("pose_status")
        or rtmpose_doc.get("status")
    )
    pose_available = bool(
        ((release_doc.get("result", {}) or {}).get("provenance") or {}).get(
            "pose_evidence_real"
        )
        or baseline.get("pose_status") == "ran"
        or rtmpose_doc.get("status") == "ready"
    )
    bowler_selection_confidence = _bowler_selection_confidence(rtmpose_doc)
    inferred_arm = release_details.get("inferred_bowling_arm")
    wrist_used = release_details.get("wrist_used_by_release_features")
    wrist_inconsistency = _wrist_inconsistency(
        inferred_arm=inferred_arm,
        wrist_used=wrist_used,
        quality_flags=quality_flags,
    )

    evidence_flags = _evidence_flags(
        true_frame=true_frame,
        prediction_available=prediction_available,
        absolute_error=absolute_error,
        signed_error=signed_error,
        tracking_malformed=tracking_malformed,
        tracking_status=tracking_status,
        primary_track_raw=primary_track_raw,
        frames_to_track_start=frames_to_track_start,
        candidate_window=candidate_window,
        rejected=rejected,
        pose_available=pose_available,
        pose_status=pose_status,
        quality_flags=quality_flags,
        release_details=release_details,
        baseline_error=baseline_error,
    )
    primary_category, secondary_categories = _choose_categories(evidence_flags)
    explanation = _classification_explanation(
        primary_category=primary_category,
        secondary_categories=secondary_categories,
        prediction_available=prediction_available,
        signed_error=signed_error,
        candidate_window=candidate_window,
        first_track_frame=first_track_frame,
        frames_to_track_start=frames_to_track_start,
        rejected=rejected,
        pose_available=pose_available,
        pose_status=pose_status,
        release_details=release_details,
        tracking_status=tracking_status,
        baseline_status=baseline_status,
        baseline_error=baseline_error,
    )
    return {
        "validation_id": validation_id,
        "analysis_id": analysis_id,
        "human_release_frame": true_frame,
        "predicted_release_frame": predicted_frame,
        "absolute_error": absolute_error,
        "signed_error": signed_error,
        "prediction_available": prediction_available,
        "prediction_status": baseline.get("prediction_status"),
        "baseline_collection_status": baseline_status,
        "baseline_collection_error": baseline_error,
        "tracking_status": tracking_status,
        "tracking_artifact_malformed_or_incompatible": tracking_malformed,
        "detector_candidates_near_true_release": candidate_window,
        "candidate_at_or_near_true_release": candidate_window[
            "candidate_at_or_near_true_release"
        ],
        "first_usable_ball_candidate_near_release": candidate_window[
            "first_candidate_near_release"
        ],
        "first_primary_track_frame": first_track_frame,
        "frames_between_true_release_and_track_start": frames_to_track_start,
        "pre_track_candidate_rejected": rejected["has_rejected_pre_track_candidate"],
        "rejected_pre_track_candidates": rejected["candidates"],
        "track_provenance_around_release": track_window,
        "pose_available": pose_available,
        "pose_status": pose_status,
        "bowler_selection_confidence": bowler_selection_confidence,
        "inferred_bowling_arm": inferred_arm,
        "wrist_used_by_release_features": wrist_used,
        "wrist_side_inconsistency": wrist_inconsistency,
        "ball_to_wrist_evidence": release_details.get("ball_to_wrist_evidence"),
        "search_window_position": release_details.get("search_window_position"),
        "forward_flight_evidence": release_details.get("forward_flight_evidence"),
        "backward_fit_availability": release_details.get("backward_fit_availability"),
        "release_feature_snapshot": release_details.get("release_feature_snapshot"),
        "primary_failure_category": primary_category,
        "secondary_failure_categories": secondary_categories,
        "concise_evidence_based_explanation": explanation,
    }


def _candidate_window_summary(
    detections_by_frame: dict[int, list[Any]],
    true_frame: int | None,
) -> dict[str, Any]:
    if true_frame is None:
        return {
            "window_start": None,
            "window_end": None,
            "total_candidates": 0,
            "frames_with_candidates": [],
            "candidate_at_exact_frame": False,
            "candidate_at_or_near_true_release": False,
            "first_candidate_near_release": None,
        }
    start = max(0, true_frame - 10)
    end = true_frame + 10
    frames = []
    first_candidate = None
    for frame_index in range(start, end + 1):
        candidates = detections_by_frame.get(frame_index, [])
        if not candidates:
            continue
        frame_record = {
            "frame_index": frame_index,
            "offset_from_true": frame_index - true_frame,
            "count": len(candidates),
            "top_candidate_id": candidates[0].candidate_id,
            "top_confidence": candidates[0].confidence,
            "top_rank": candidates[0].rank,
        }
        frames.append(frame_record)
        if first_candidate is None:
            first_candidate = frame_record
    near_frames = [
        item for item in frames if abs(int(item["offset_from_true"])) <= 2
    ]
    exact = any(item["frame_index"] == true_frame for item in frames)
    return {
        "window_start": start,
        "window_end": end,
        "total_candidates": sum(item["count"] for item in frames),
        "frames_with_candidates": frames,
        "candidate_at_exact_frame": exact,
        "candidate_at_or_near_true_release": bool(near_frames),
        "near_true_candidate_frames": near_frames,
        "first_candidate_near_release": first_candidate,
        "true_ball_present_in_top_k_proxy": bool(near_frames),
        "note": (
            "Top-K proxy is based on persisted detector candidates within true frame +/-2; "
            "visual identity of the true ball is not inferred."
        ),
    }


def _rejected_candidates_near_true(
    *,
    tracking_doc: dict[str, Any],
    true_frame: int | None,
    first_track_frame: int | None,
) -> dict[str, Any]:
    if true_frame is None:
        return {"has_rejected_pre_track_candidate": False, "candidates": []}
    start = max(0, true_frame - 10)
    end = true_frame + 10
    candidates = []
    for item in tracking_doc.get("candidate_diagnostics", []) or []:
        frame_index = _optional_int(item.get("frame_index"))
        if frame_index is None or not (start <= frame_index <= end):
            continue
        if first_track_frame is not None and frame_index >= first_track_frame:
            continue
        if item.get("selected") is not False:
            continue
        candidates.append(
            {
                "frame_index": frame_index,
                "candidate_id": item.get("candidate_id"),
                "selection_reason": item.get("selection_reason"),
                "static_likelihood": item.get("static_likelihood"),
                "score_components": item.get("score_components"),
            }
        )
    return {
        "has_rejected_pre_track_candidate": bool(candidates),
        "candidates": candidates,
    }


def _track_window_summary(
    primary_track_raw: list[dict[str, Any]],
    true_frame: int | None,
) -> list[dict[str, Any]]:
    if true_frame is None:
        return []
    start = max(0, true_frame - 5)
    end = true_frame + 10
    rows = []
    for point in primary_track_raw:
        frame_index = _optional_int(point.get("frame_index"))
        if frame_index is None or not (start <= frame_index <= end):
            continue
        rows.append(
            {
                "frame_index": frame_index,
                "candidate_id": point.get("candidate_id"),
                "provenance": point.get("provenance") or point.get("source"),
                "confidence": point.get("confidence"),
            }
        )
    return rows


def _release_details(
    *,
    release_doc: dict[str, Any],
    rtmpose_doc: dict[str, Any],
    true_frame: int | None,
    predicted_frame: int | None,
    detections_by_frame: dict[int, list[Any]],
    tracking_doc: dict[str, Any],
) -> dict[str, Any]:
    result = release_doc.get("result") or {}
    candidate_scores = release_doc.get("candidate_scores") or []
    score_by_frame = {
        _optional_int(score.get("frame_index")): score
        for score in candidate_scores
        if _optional_int(score.get("frame_index")) is not None
    }
    predicted_score = score_by_frame.get(predicted_frame)
    true_score = score_by_frame.get(true_frame)
    evidence = result.get("evidence") or {}
    wrist_used = None
    inferred_arm = _extract_inferred_arm(rtmpose_doc)
    if rtmpose_doc and predicted_frame is not None:
        wrist_used = _wrist_used_from_persisted_pose(
            release_doc=release_doc,
            rtmpose_doc=rtmpose_doc,
            predicted_frame=predicted_frame,
            detections_by_frame=detections_by_frame,
            tracking_doc=tracking_doc,
        )
    candidate_frames = sorted(
        frame for frame in score_by_frame if frame is not None
    )
    return {
        "inferred_bowling_arm": inferred_arm,
        "wrist_used_by_release_features": wrist_used,
        "ball_to_wrist_evidence": {
            "predicted_ball_wrist_distance_px": evidence.get("ball_wrist_distance_px"),
            "predicted_normalized_ball_wrist_distance": evidence.get(
                "normalized_ball_wrist_distance"
            ),
            "true_frame_ball_wrist_distance_px": (true_score or {})
            .get("features", {})
            .get("ball_wrist_distance_px"),
            "true_frame_normalized_ball_wrist_distance": (true_score or {})
            .get("features", {})
            .get("normalized_ball_wrist_distance"),
        },
        "search_window_position": {
            "candidate_score_frame_min": min(candidate_frames) if candidate_frames else None,
            "candidate_score_frame_max": max(candidate_frames) if candidate_frames else None,
            "human_true_frame_in_candidate_scores": true_frame in score_by_frame,
            "predicted_frame_in_candidate_scores": predicted_frame in score_by_frame,
        },
        "forward_flight_evidence": {
            "predicted_forward_free_flight_points": evidence.get(
                "forward_free_flight_points"
            ),
            "predicted_forward_free_flight_confirmation": evidence.get(
                "forward_free_flight_confirmation"
            ),
            "true_frame_forward_free_flight_points": (true_score or {})
            .get("features", {})
            .get("forward_free_flight_points"),
            "true_frame_forward_free_flight_confirmation": (true_score or {})
            .get("features", {})
            .get("forward_free_flight_confirmation"),
        },
        "backward_fit_availability": {
            "predicted_backward_trajectory_fit_error_px": evidence.get(
                "backward_trajectory_fit_error_px"
            ),
            "true_frame_backward_trajectory_fit_error_px": (true_score or {})
            .get("features", {})
            .get("backward_trajectory_fit_error_px"),
            "predicted_backward_fit_available": evidence.get(
                "backward_trajectory_fit_error_px"
            )
            is not None,
            "true_frame_backward_fit_available": (true_score or {})
            .get("features", {})
            .get("backward_trajectory_fit_error_px")
            is not None,
        },
        "release_feature_snapshot": {
            "predicted_score": None if predicted_score is None else predicted_score.get("score"),
            "predicted_method": None if predicted_score is None else predicted_score.get("method"),
            "predicted_score_components": None
            if predicted_score is None
            else predicted_score.get("score_components"),
            "true_frame_score": None if true_score is None else true_score.get("score"),
            "true_frame_method": None if true_score is None else true_score.get("method"),
            "true_frame_score_components": None
            if true_score is None
            else true_score.get("score_components"),
        },
    }


def _wrist_used_from_persisted_pose(
    *,
    release_doc: dict[str, Any],
    rtmpose_doc: dict[str, Any],
    predicted_frame: int,
    detections_by_frame: dict[int, list[Any]],
    tracking_doc: dict[str, Any],
) -> str | None:
    bowler_doc = dict(rtmpose_doc.get("bowler") or {})
    if "provider" not in bowler_doc:
        bowler_doc["provider"] = rtmpose_doc.get("pose_provider") or {}
    pose_sequence = parse_bowler_pose_sequence(bowler_doc)
    if pose_sequence is None:
        return None
    try:
        features = extract_release_features(
            ReleaseCandidate(frame_index=predicted_frame, source="diagnostic"),
            detections_by_frame=detections_by_frame,
            primary_track=parse_track_observations(tracking_doc),
            pose_sequence=pose_sequence,
            config=ReleasePointConfig(),
        )
    except Exception:
        return None
    return features.wrist_keypoint_name


def _extract_inferred_arm(rtmpose_doc: dict[str, Any]) -> str | None:
    arm = rtmpose_doc.get("bowling_arm") or (rtmpose_doc.get("bowler") or {}).get(
        "bowling_arm"
    )
    if isinstance(arm, dict):
        return arm.get("bowling_arm")
    return None


def _evidence_flags(
    *,
    true_frame: int | None,
    prediction_available: bool,
    absolute_error: int | None,
    signed_error: int | None,
    tracking_malformed: bool,
    tracking_status: str | None,
    primary_track_raw: list[dict[str, Any]],
    frames_to_track_start: int | None,
    candidate_window: dict[str, Any],
    rejected: dict[str, Any],
    pose_available: bool,
    pose_status: str | None,
    quality_flags: list[str],
    release_details: dict[str, Any],
    baseline_error: str | None,
) -> dict[str, bool]:
    _ = true_frame, absolute_error, tracking_status, pose_status, baseline_error
    detector_gap = not candidate_window["candidate_at_or_near_true_release"]
    no_primary_track = not primary_track_raw
    primary_track_late = (
        frames_to_track_start is not None and frames_to_track_start > 2
    )
    pose_bad = (not pose_available) or any(
        flag in quality_flags
        for flag in {
            "pose_unavailable_or_unreliable",
            "pose_insufficient",
            "insufficient_bowler_evidence",
        }
    )
    late_bias = bool(
        prediction_available
        and signed_error is not None
        and signed_error > 2
        and (
            (
                release_details.get("forward_flight_evidence", {}).get(
                    "predicted_forward_free_flight_confirmation"
                )
                or 0
            )
            >= 0.8
        )
    )
    bad_backward = "bad_backward_trajectory_fit" in quality_flags or not (
        release_details.get("backward_fit_availability", {}).get(
            "predicted_backward_fit_available"
        )
        if prediction_available
        else True
    )
    search_window = bool(
        prediction_available
        and not release_details.get("search_window_position", {}).get(
            "human_true_frame_in_candidate_scores"
        )
    )
    return {
        "malformed_or_incompatible_tracking_input": tracking_malformed,
        "detector_missing_near_release": detector_gap,
        "candidate_present_tracker_rejected": rejected[
            "has_rejected_pre_track_candidate"
        ],
        "primary_track_starts_too_late": primary_track_late,
        "no_primary_track": no_primary_track,
        "bowler_selection_failure": any(
            flag in quality_flags
            for flag in {"bowler_selection_uncertain", "insufficient_bowler_evidence"}
        ),
        "bowling_arm_or_wrist_mismatch": _wrist_inconsistency(
            inferred_arm=release_details.get("inferred_bowling_arm"),
            wrist_used=release_details.get("wrist_used_by_release_features"),
            quality_flags=quality_flags,
        ),
        "pose_unavailable_or_unreliable": pose_bad,
        "search_window_failure": search_window,
        "late_free_flight_bias": late_bias,
        "backward_reconstruction_insufficient": bad_backward,
        "feature_fusion_failure": bool(
            prediction_available and absolute_error is not None and absolute_error > 2
        ),
    }


def _choose_categories(evidence_flags: dict[str, bool]) -> tuple[str, list[str]]:
    if evidence_flags["malformed_or_incompatible_tracking_input"]:
        primary = "malformed_or_incompatible_tracking_input"
    elif evidence_flags["detector_missing_near_release"]:
        primary = "detector_missing_near_release"
    elif evidence_flags["candidate_present_tracker_rejected"] and evidence_flags["no_primary_track"]:
        primary = "candidate_present_tracker_rejected"
    elif evidence_flags["primary_track_starts_too_late"]:
        primary = "primary_track_starts_too_late"
    elif evidence_flags["pose_unavailable_or_unreliable"]:
        primary = "pose_unavailable_or_unreliable"
    elif evidence_flags["late_free_flight_bias"]:
        primary = "late_free_flight_bias"
    elif evidence_flags["backward_reconstruction_insufficient"]:
        primary = "backward_reconstruction_insufficient"
    elif evidence_flags["feature_fusion_failure"]:
        primary = "feature_fusion_failure"
    else:
        primary = "no_failure_detected"
    secondary = [
        category
        for category, active in evidence_flags.items()
        if active and category != primary and category != "no_primary_track"
    ]
    if not secondary and primary == "no_failure_detected":
        return primary, []
    return primary, secondary


def _classification_explanation(
    *,
    primary_category: str,
    secondary_categories: list[str],
    prediction_available: bool,
    signed_error: int | None,
    candidate_window: dict[str, Any],
    first_track_frame: int | None,
    frames_to_track_start: int | None,
    rejected: dict[str, Any],
    pose_available: bool,
    pose_status: str | None,
    release_details: dict[str, Any],
    tracking_status: str | None,
    baseline_status: str | None,
    baseline_error: str | None,
) -> str:
    pieces = [f"Primary={primary_category}."]
    if prediction_available and signed_error is not None:
        direction = "late" if signed_error > 0 else "early" if signed_error < 0 else "exact"
        pieces.append(f"Prediction was {abs(signed_error)} frames {direction}.")
    else:
        pieces.append(
            f"No ready prediction; baseline_status={baseline_status}, tracking_status={tracking_status}."
        )
    pieces.append(
        f"Detector window had {candidate_window['total_candidates']} candidates; "
        f"near/exact release candidate={candidate_window['candidate_at_or_near_true_release']}."
    )
    if first_track_frame is not None:
        pieces.append(
            f"First primary-track frame={first_track_frame} "
            f"({frames_to_track_start:+d} from true release)."
        )
    else:
        pieces.append("No primary-track frame was available.")
    if rejected["has_rejected_pre_track_candidate"]:
        first_rejected = rejected["candidates"][0]
        pieces.append(
            f"Pre-track candidate rejected at frame {first_rejected['frame_index']} "
            f"({first_rejected.get('selection_reason')})."
        )
    pieces.append(f"Pose available={pose_available} status={pose_status}.")
    forward = release_details.get("forward_flight_evidence") or {}
    backward = release_details.get("backward_fit_availability") or {}
    if forward.get("predicted_forward_free_flight_confirmation") is not None:
        pieces.append(
            "Predicted forward-flight confirmation="
            f"{forward.get('predicted_forward_free_flight_confirmation')}."
        )
    if backward.get("predicted_backward_fit_available") is not None:
        pieces.append(
            "Predicted backward fit available="
            f"{backward.get('predicted_backward_fit_available')}."
        )
    if secondary_categories:
        pieces.append(f"Secondary={', '.join(secondary_categories)}.")
    if baseline_error:
        pieces.append(f"Baseline error: {baseline_error}")
    return " ".join(pieces)


def _aggregate_classification(
    records: list[dict[str, Any]],
    metrics_doc: dict[str, Any],
) -> dict[str, Any]:
    primary_counts = _count_values(record["primary_failure_category"] for record in records)
    secondary_counts = _count_values(
        category
        for record in records
        for category in record.get("secondary_failure_categories", [])
    )
    delays = [
        record["frames_between_true_release_and_track_start"]
        for record in records
        if record["frames_between_true_release_and_track_start"] is not None
    ]
    after_delays = [delay for delay in delays if delay > 0]
    signed_errors = [
        record["signed_error"]
        for record in records
        if record["prediction_available"] and record["signed_error"] is not None
    ]
    return {
        "schema_version": "1.0",
        "clip_count": len(records),
        "metrics_snapshot": {
            key: metrics_doc.get(key)
            for key in [
                "prediction_coverage",
                "unresolved_rate",
                "exact_frame_accuracy",
                "within_1_frame_accuracy",
                "within_2_frame_accuracy",
                "mean_absolute_frame_error",
                "median_absolute_frame_error",
            ]
        },
        "primary_failure_category_counts": primary_counts,
        "secondary_failure_category_counts": secondary_counts,
        "detector_gap_near_release_count": sum(
            1 for record in records if not record["candidate_at_or_near_true_release"]
        ),
        "candidate_present_tracker_rejected_count": sum(
            1 for record in records if record["pre_track_candidate_rejected"]
        ),
        "primary_tracks_begin_after_human_release_count": sum(
            1
            for record in records
            if (
                record["frames_between_true_release_and_track_start"] is not None
                and record["frames_between_true_release_and_track_start"] > 0
            )
        ),
        "average_delay_between_true_release_and_first_primary_track": _mean(delays),
        "average_positive_delay_when_track_starts_after_release": _mean(after_delays),
        "pose_or_wrist_inconsistency_count": sum(
            1 for record in records if record["wrist_side_inconsistency"]
        ),
        "malformed_tracking_artifact_count": sum(
            1
            for record in records
            if record["primary_failure_category"]
            == "malformed_or_incompatible_tracking_input"
        ),
        "prediction_bias_counts": {
            "early": sum(1 for error in signed_errors if error < 0),
            "late": sum(1 for error in signed_errors if error > 0),
            "exact": sum(1 for error in signed_errors if error == 0),
            "no_prediction": sum(1 for record in records if not record["prediction_available"]),
        },
        "rv1_002_pattern_commonality": _rv1_002_commonality(records),
        "dominant_primary_failure_category": _dominant_primary_category(primary_counts),
        "dominant_algorithmic_root_cause_excluding_malformed_artifacts": (
            _dominant_root_cause(records)
        ),
        "dominant_root_cause": (
            "Malformed/incompatible historical tracking artifacts are the largest "
            "single primary bucket, but among clips with usable current artifacts "
            "the dominant actionable engine issue is pose/wrist unreliability plus "
            "pre-track candidate handling."
        ),
        "recommended_first_release_v1_1_correction": (
            "Build pre-track release-region reconstruction/classification from "
            "persisted detector candidates and backward trajectory projection, "
            "then require it to compete with late free-flight frames before "
            "tuning scoring weights."
        ),
        "corrections_to_defer": [
            "Detector threshold or model changes until candidate/track failure modes are separated.",
            "Full ReleaseEstimator score tuning until malformed legacy artifacts are excluded from the validation denominator.",
            "UI/API changes until Release V1.1 has a stable diagnostic-backed engine correction.",
            "Metric 3D release reconstruction and physics-heavy reconstruction.",
        ],
    }


def _write_classification_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "validation_id",
        "analysis_id",
        "human_release_frame",
        "predicted_release_frame",
        "absolute_error",
        "signed_error",
        "prediction_available",
        "detector_candidates_near_true_release",
        "candidate_at_or_near_true_release",
        "first_usable_ball_candidate_near_release",
        "first_primary_track_frame",
        "frames_between_true_release_and_track_start",
        "pose_available",
        "bowler_selection_confidence",
        "inferred_bowling_arm",
        "wrist_used_by_release_features",
        "primary_failure_category",
        "secondary_failure_categories",
        "concise_evidence_based_explanation",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: _csv_value(record.get(key))
                    for key in fieldnames
                }
            )


def _summary_markdown(
    summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Release Point V1 Failure Classification",
        "",
        f"- Clip count: {summary['clip_count']}",
        f"- Dominant root cause: {summary['dominant_root_cause']}",
        f"- Detector gaps near release: {summary['detector_gap_near_release_count']}",
        f"- Candidate present but tracker rejected: {summary['candidate_present_tracker_rejected_count']}",
        f"- Primary tracks begin after human release: {summary['primary_tracks_begin_after_human_release_count']}",
        f"- Average delay true release to first track: {summary['average_delay_between_true_release_and_first_primary_track']}",
        f"- Pose/wrist inconsistency count: {summary['pose_or_wrist_inconsistency_count']}",
        f"- Malformed tracking artifacts: {summary['malformed_tracking_artifact_count']}",
        "",
        "## Primary Category Counts",
        "",
    ]
    for category, count in summary["primary_failure_category_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Prediction Bias", ""])
    for category, count in summary["prediction_bias_counts"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "## Clip Classifications", ""])
    for record in records:
        lines.append(
            f"- {record['validation_id']} `{record['analysis_id']}`: "
            f"{record['primary_failure_category']} "
            f"(secondary: {', '.join(record['secondary_failure_categories']) or 'none'}). "
            f"{record['concise_evidence_based_explanation']}"
        )
    lines.extend(
        [
            "",
            "## Recommended First Correction",
            "",
            summary["recommended_first_release_v1_1_correction"],
            "",
            "## Corrections To Defer",
            "",
        ]
    )
    for item in summary["corrections_to_defer"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_track_frame(primary_track_raw: list[dict[str, Any]]) -> int | None:
    frames = [
        _optional_int(point.get("frame_index"))
        for point in primary_track_raw
        if _optional_int(point.get("frame_index")) is not None
    ]
    return min(frames) if frames else None


def _tracking_malformed_for_release(
    *,
    tracking_doc: dict[str, Any],
    baseline_error: str | None,
) -> bool:
    if baseline_error and "tracking_result.json is malformed" in baseline_error:
        return True
    if tracking_doc.get("status") != "ready":
        return False
    primary_track = tracking_doc.get("primary_track") or []
    if not primary_track:
        return False
    required_v2_fields = {"provenance", "uncertainty"}
    return any(
        not required_v2_fields.issubset(set(point.keys()))
        for point in primary_track[: min(3, len(primary_track))]
    )


def _bowler_selection_confidence(rtmpose_doc: dict[str, Any]) -> float | None:
    bowler = rtmpose_doc.get("bowler") or {}
    value = bowler.get("selection_confidence")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _wrist_inconsistency(
    *,
    inferred_arm: str | None,
    wrist_used: str | None,
    quality_flags: list[str],
) -> bool:
    if inferred_arm and wrist_used:
        return not wrist_used.startswith(f"{inferred_arm}_")
    return any(
        flag in quality_flags
        for flag in {"bowling_arm_ambiguous", "low_confidence_wrist"}
    )


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _mean(values: list[int]) -> float | None:
    return None if not values else round(sum(values) / len(values), 6)


def _dominant_root_cause(records: list[dict[str, Any]]) -> str:
    non_artifact = [
        record
        for record in records
        if record["primary_failure_category"]
        != "malformed_or_incompatible_tracking_input"
    ]
    counts = _count_values(record["primary_failure_category"] for record in non_artifact)
    if not counts:
        return "malformed_or_incompatible_tracking_input"
    category, count = next(iter(counts.items()))
    return f"{category} ({count}/{len(records)} clips; malformed artifacts counted separately)"


def _dominant_primary_category(counts: dict[str, int]) -> str | None:
    if not counts:
        return None
    category, count = next(iter(counts.items()))
    return f"{category} ({count} clips)"


def _rv1_002_commonality(records: list[dict[str, Any]]) -> dict[str, Any]:
    rv1_002 = next(
        (record for record in records if record["validation_id"] == "rv1_002"),
        None,
    )
    if rv1_002 is None:
        return {"status": "unknown", "reason": "rv1_002 was not in the classification set."}
    same_primary = [
        record["validation_id"]
        for record in records
        if record["primary_failure_category"] == rv1_002["primary_failure_category"]
    ]
    shared_detector_gap_and_late_track = [
        record["validation_id"]
        for record in records
        if (
            not record["candidate_at_or_near_true_release"]
            and record["frames_between_true_release_and_track_start"] is not None
            and record["frames_between_true_release_and_track_start"] > 2
        )
    ]
    return {
        "rv1_002_primary_failure_category": rv1_002["primary_failure_category"],
        "same_primary_category_validation_ids": same_primary,
        "same_primary_category_count": len(same_primary),
        "shared_detector_gap_and_late_track_validation_ids": shared_detector_gap_and_late_track,
        "shared_detector_gap_and_late_track_count": len(
            shared_detector_gap_and_late_track
        ),
        "assessment": (
            "common_pattern"
            if len(shared_detector_gap_and_late_track) >= 3
            else "partially_shared_but_not_dominant"
        ),
    }


def _timestamp(frame_index: int, metadata: dict[str, Any]) -> float:
    fps = float(metadata.get("fps") or 0.0)
    return round(frame_index / fps, 6) if fps > 0 else 0.0


def _keypoint_dict(keypoint: Keypoint | None) -> dict[str, float] | None:
    if keypoint is None:
        return None
    return {
        "x": keypoint.x,
        "y": keypoint.y,
        "confidence": keypoint.confidence,
    }


def _selected_ball(features: Any) -> dict[str, Any] | None:
    if features.ball_candidate_id is None or features.ball_x is None or features.ball_y is None:
        return None
    return {
        "candidate_id": features.ball_candidate_id,
        "x": features.ball_x,
        "y": features.ball_y,
        "detector_confidence": features.detector_confidence,
        "rank": features.candidate_rank,
    }


def _nearest_track_dict(features: Any, primary_track: list[Any]) -> dict[str, Any] | None:
    point = None
    for candidate in primary_track:
        if (
            candidate.provenance == features.tracker_provenance
            and abs(candidate.confidence - features.track_confidence) < 0.000001
        ):
            if point is None or abs(candidate.frame_index - features.frame_index) < abs(
                point.frame_index - features.frame_index
            ):
                point = candidate
    return _track_observation_dict(point)


def _ball_observation_dict(
    candidate: Any,
    tracking_diagnostic_by_candidate: dict[str, Any],
) -> dict[str, Any]:
    tracking_diagnostic = tracking_diagnostic_by_candidate.get(candidate.candidate_id)
    return {
        "candidate_id": candidate.candidate_id,
        "x": candidate.x,
        "y": candidate.y,
        "normalized_x": candidate.normalized_x,
        "normalized_y": candidate.normalized_y,
        "confidence": candidate.confidence,
        "rank": candidate.rank,
        "inside_pitch_corridor": candidate.inside_pitch_corridor,
        "tracking_selected": None
        if tracking_diagnostic is None
        else tracking_diagnostic.get("selected"),
        "tracking_selection_reason": None
        if tracking_diagnostic is None
        else tracking_diagnostic.get("selection_reason"),
        "tracking_score_components": None
        if tracking_diagnostic is None
        else tracking_diagnostic.get("score_components"),
    }


def _track_observation_dict(point: Any | None) -> dict[str, Any] | None:
    if point is None:
        return None
    return asdict(point)


def _selected_bowling_arm(rtmpose_doc: dict[str, Any]) -> dict[str, Any] | None:
    arm = rtmpose_doc.get("bowling_arm")
    if arm is not None:
        return arm
    bowler = rtmpose_doc.get("bowler") or {}
    return bowler.get("bowling_arm")


def _distance(ball: dict[str, Any] | None, wrist: dict[str, Any] | None) -> float | None:
    if not ball or not wrist:
        return None
    return math.hypot(float(ball["x"]) - float(wrist["x"]), float(ball["y"]) - float(wrist["y"]))


def _normalized_distance(
    ball: dict[str, Any] | None,
    wrist: dict[str, Any] | None,
    scale: float | None,
) -> float | None:
    distance = _distance(ball, wrist)
    if distance is None or scale is None:
        return None
    return distance / max(1.0, scale)


def _kp_velocity(
    pose_sequence: Any | None,
    frame_index: int,
    name: str,
) -> float | None:
    return keypoint_velocity(pose_sequence, frame_index, name, step=1)


def _arm_geometry_evidence(pose: Any | None) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for side in ("left", "right"):
        wrist_name = f"{side}_wrist"
        angle, extension = arm_geometry(pose, wrist_name)
        evidence[side] = {
            "angle_degrees": angle,
            "extension_proxy": extension,
        }
    return evidence


def _scene_roi_evidence(
    features: Any,
    detections: list[Any],
    exact_track: Any | None,
    calibration_doc: dict[str, Any] | None,
    calibration_v2_doc: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "scene_roi_consistency": features.scene_roi_consistency,
        "candidate_inside_pitch_corridor_values": [
            candidate.inside_pitch_corridor for candidate in detections
        ],
        "exact_track_inside_pitch_corridor": None
        if exact_track is None
        else exact_track.inside_pitch_corridor,
        "legacy_calibration_status": None
        if calibration_doc is None
        else calibration_doc.get("status"),
        "calibration_v2_status": None
        if calibration_v2_doc is None
        else calibration_v2_doc.get("status"),
        "note": (
            "Release V1 feature extraction uses detection/track inside_pitch_corridor "
            "booleans for this score."
        ),
    }


def _unavailable_evidence(
    *,
    selected_ball: dict[str, Any] | None,
    exact_track: Any | None,
    pose: Any | None,
    score: dict[str, Any] | None,
    features: Any,
) -> list[str]:
    unavailable = []
    if selected_ball is None:
        unavailable.append("selected_ball_candidate")
        unavailable.append("ball_pixel_coordinates")
        unavailable.append("ball_to_wrist_distances")
        unavailable.append("normalized_ball_to_wrist_distances")
        unavailable.append("detector_confidence_for_selected_ball")
    if exact_track is None:
        unavailable.append("exact_primary_track_point")
    if pose is None:
        unavailable.append("bowler_pose")
        unavailable.append("wrist_coordinates")
    if features.separation_velocity is None:
        unavailable.append("separation_velocity")
    if features.backward_trajectory_fit_error_px is None:
        unavailable.append("backward_trajectory_fit_error_px")
    if score is None:
        unavailable.append("release_candidate_score")
        unavailable.append("release_feature_scores")
    return unavailable


def _build_summary(
    *,
    args: argparse.Namespace,
    analysis_dir: Path,
    metadata: dict[str, Any],
    detections_by_frame: dict[int, list[Any]],
    tracking_doc: dict[str, Any],
    release_doc: dict[str, Any],
    rtmpose_doc: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result = release_doc.get("result") or {}
    candidate_scores = release_doc.get("candidate_scores") or []
    sorted_scores = sorted(
        candidate_scores, key=lambda item: float(item.get("score", 0.0)), reverse=True
    )
    true_row = next(row for row in rows if row["frame_index"] == args.true_frame)
    predicted_row = next(row for row in rows if row["frame_index"] == args.predicted_frame)
    true_detections = detections_by_frame.get(args.true_frame, [])
    primary_track = tracking_doc.get("primary_track") or []
    first_track_frame = min((int(item["frame_index"]) for item in primary_track), default=None)
    frame_48_detections = detections_by_frame.get(48, [])
    return {
        "schema_version": "1.0",
        "analysis_id": args.analysis_id,
        "raw_video_path": str(analysis_dir / "raw" / metadata.get("stored_filename", "")),
        "frame_window": {"start": args.start_frame, "end": args.end_frame},
        "human_true_release_frame": args.true_frame,
        "cricvision_predicted_release_frame": args.predicted_frame,
        "absolute_error_frames": abs(args.predicted_frame - args.true_frame),
        "reported_confidence": result.get("confidence"),
        "reported_method": result.get("evidence_mode"),
        "reported_quality_flags": result.get("quality_flags"),
        "pose_provider": rtmpose_doc.get("pose_provider"),
        "bowler_selection": {
            "bowler_id": (rtmpose_doc.get("bowler") or {}).get("bowler_id"),
            "selection_confidence": (rtmpose_doc.get("bowler") or {}).get(
                "selection_confidence"
            ),
            "selected_bowling_arm": _selected_bowling_arm(rtmpose_doc),
            "frame_count": len((rtmpose_doc.get("bowler") or {}).get("frames") or []),
        },
        "release_score_ranking": [
            {
                "rank": index + 1,
                "frame_index": item.get("frame_index"),
                "score": item.get("score"),
                "method": item.get("method"),
                "source": item.get("source"),
                "quality_flags": item.get("quality_flags"),
            }
            for index, item in enumerate(sorted_scores[:8])
        ],
        "true_frame_detection": {
            "frame_index": args.true_frame,
            "detection_count": len(true_detections),
            "candidate_ids": [item.candidate_id for item in true_detections],
            "valid_ball_detection": len(true_detections) > 0,
            "present_in_top_k": len(true_detections) > 0,
        },
        "tracking_preservation": {
            "first_primary_track_frame": first_track_frame,
            "primary_track_frames": [item.get("frame_index") for item in primary_track],
            "true_frame_in_primary_track": any(
                int(item.get("frame_index", -1)) == args.true_frame
                for item in primary_track
            ),
            "frame_48_candidate": [
                _ball_observation_dict(item, {}) for item in frame_48_detections
            ],
            "frame_48_tracking_diagnostic": [
                item
                for item in tracking_doc.get("candidate_diagnostics", [])
                if int(item.get("frame_index", -1)) == 48
            ],
        },
        "why_predicted_frame_won": {
            "predicted_frame": args.predicted_frame,
            "predicted_score": predicted_row.get("release_candidate_score"),
            "predicted_feature_scores": predicted_row.get("release_feature_scores"),
            "predicted_evidence": {
                "selected_ball_candidate": predicted_row.get("selected_ball_candidate"),
                "forward_free_flight_points": predicted_row.get(
                    "forward_free_flight_points"
                ),
                "forward_free_flight_confirmation": predicted_row.get(
                    "forward_free_flight_confirmation"
                ),
                "backward_trajectory_fit_error_px": predicted_row.get(
                    "backward_trajectory_fit_error_px"
                ),
                "scene_roi": predicted_row.get("scene_roi_evidence"),
                "normalized_ball_to_selected_wrist_distance": predicted_row.get(
                    "normalized_ball_to_selected_wrist_distance"
                ),
                "separation_velocity": predicted_row.get("separation_velocity"),
                "separation_persistence_frames": predicted_row.get(
                    "separation_persistence_frames"
                ),
            },
            "true_frame_score": true_row.get("release_candidate_score"),
            "true_frame_feature_scores": true_row.get("release_feature_scores"),
            "true_frame_evidence": {
                "selected_ball_candidate": true_row.get("selected_ball_candidate"),
                "forward_free_flight_points": true_row.get(
                    "forward_free_flight_points"
                ),
                "forward_free_flight_confirmation": true_row.get(
                    "forward_free_flight_confirmation"
                ),
                "backward_trajectory_fit_error_px": true_row.get(
                    "backward_trajectory_fit_error_px"
                ),
                "selected_wrist_velocity": true_row.get("selected_wrist_velocity"),
                "selected_wrist_acceleration_proxy": true_row.get(
                    "selected_wrist_acceleration_proxy"
                ),
            },
            "diagnosis": (
                "Frame 63 won because it had an observed primary-track ball with "
                "strong detector/track confidence, excellent forward free-flight "
                "confirmation, and excellent backward trajectory fit. Frame 52 had "
                "pose and wrist-motion evidence but no persisted ball detection, no "
                "selected ball coordinates, no backward fit, and zero forward "
                "free-flight points within the configured forward window."
            ),
        },
        "root_cause_assessment": {
            "true_ball_detected_at_frame_52": len(true_detections) > 0,
            "true_ball_in_top_k_at_frame_52": len(true_detections) > 0,
            "tracking_preserved_true_frame": False,
            "correct_bowler_selected": "likely_but_uncertain",
            "correct_bowling_wrist_selected": "likely_right_wrist_but_not_ball-confirmed",
            "primary_failure_component": "detector_tracker_coverage_gap",
            "secondary_failure_components": [
                "feature_fusion_overweights_later_confirmed_free_flight_when release-frame ball evidence is missing",
                "scene_roi_score_is_zero_for_the_whole_selected_track_and_does_not_discriminate",
                "bowler_selection_confidence_is_low",
            ],
        },
        "evidence_availability_policy": (
            "Null values mean the current persisted architecture did not contain "
            "that signal for the frame; they are not inferred."
        ),
    }


def _write_visual_debug(
    *,
    analysis_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
    true_frame: int,
    predicted_frame: int,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {
            "status": UNAVAILABLE,
            "reason": "opencv-python is not available in this environment.",
        }

    raw_path = analysis_dir / "raw" / str(metadata.get("stored_filename", ""))
    if not raw_path.is_file():
        return {"status": UNAVAILABLE, "reason": "raw original video is missing."}
    frames_dir = output_dir / "debug_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        return {"status": UNAVAILABLE, "reason": "OpenCV could not open raw video."}

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(metadata.get("width") or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(metadata.get("height") or 0)
    fps = float(metadata.get("fps") or 60.0)
    video_path = output_dir / "diagnostic_debug.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        min(12.0, max(4.0, fps / 6.0)),
        (width, height),
    )
    saved_frames = []
    for row in rows:
        frame_index = int(row["frame_index"])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            continue
        _draw_frame_overlay(cv2, frame, row, true_frame, predicted_frame)
        frame_path = frames_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_frames.append(str(frame_path))
        if writer.isOpened():
            writer.write(frame)
    capture.release()
    if writer.isOpened():
        writer.release()
        video_status = "ready"
    else:
        video_status = UNAVAILABLE
    return {
        "status": "ready",
        "debug_video_status": video_status,
        "debug_video_path": str(video_path) if video_status == "ready" else None,
        "debug_frame_dir": str(frames_dir),
        "debug_frame_count": len(saved_frames),
    }


def _draw_frame_overlay(
    cv2: Any,
    frame: Any,
    row: dict[str, Any],
    true_frame: int,
    predicted_frame: int,
) -> None:
    frame_index = int(row["frame_index"])
    height, width = frame.shape[:2]
    banner = (32, 32, 32)
    if frame_index == true_frame:
        banner = (0, 0, 220)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), banner, 8)
    if frame_index == predicted_frame:
        banner = (0, 150, 0)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), banner, 8)
    cv2.rectangle(frame, (0, 0), (width, 74), banner, -1)
    label = f"frame {frame_index}  t={row['timestamp_seconds']:.3f}s"
    if frame_index == true_frame:
        label += "  HUMAN RELEASE"
    if frame_index == predicted_frame:
        label += "  CRICVISION PREDICTED"
    cv2.putText(frame, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    score = row.get("release_candidate_score")
    cv2.putText(
        frame,
        f"score={score if score is not None else 'null'}  candidates={len(row.get('all_ball_candidates') or [])}",
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
    )
    for candidate in row.get("all_ball_candidates") or []:
        x, y = int(round(candidate["x"])), int(round(candidate["y"]))
        cv2.circle(frame, (x, y), 7, (0, 215, 255), 2)
        cv2.putText(
            frame,
            f"#{candidate.get('rank')} {candidate.get('confidence'):.2f}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 215, 255),
            1,
        )
    selected = row.get("selected_ball_candidate")
    if selected:
        x, y = int(round(selected["x"])), int(round(selected["y"]))
        cv2.circle(frame, (x, y), 11, (0, 255, 0), 3)
        cv2.putText(frame, "selected ball", (x + 12, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
    _draw_pose(cv2, frame, row)


def _draw_pose(cv2: Any, frame: Any, row: dict[str, Any]) -> None:
    bbox = row.get("bowler_pose_bbox_xyxy")
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 255), 2)
        cv2.putText(
            frame,
            str(row.get("bowler_pose_person_id") or "bowler"),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 255),
            1,
        )

    raw_keypoints = row.get("bowler_pose_keypoints") or {}
    keypoints = {
        name: (int(round(point["x"])), int(round(point["y"])))
        for name, point in raw_keypoints.items()
        if point and point.get("confidence", 0.0) is not None
    }
    skeleton = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]
    for first, second in skeleton:
        if first in keypoints and second in keypoints:
            cv2.line(frame, keypoints[first], keypoints[second], (220, 170, 80), 2)
    for name, point in raw_keypoints.items():
        if not point:
            continue
        x, y = int(round(point["x"])), int(round(point["y"]))
        color = (220, 170, 80)
        radius = 3
        if name == "left_wrist":
            color = (255, 255, 0)
            radius = 7
        elif name == "right_wrist":
            color = (255, 0, 255)
            radius = 7
        cv2.circle(frame, (x, y), radius, color, -1)

    wrists = {
        "left_wrist": row.get("left_wrist"),
        "right_wrist": row.get("right_wrist"),
    }
    for name, point in wrists.items():
        if not point:
            continue
        x, y = int(round(point["x"])), int(round(point["y"]))
        color = (255, 255, 0) if name == "left_wrist" else (255, 0, 255)
        cv2.putText(
            frame,
            f"{name} {point.get('confidence'):.2f}",
            (x + 8, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
        )
    selected = row.get("release_engine_selected_wrist")
    if selected in keypoints:
        cv2.circle(frame, keypoints[selected], 13, (255, 255, 255), 2)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
