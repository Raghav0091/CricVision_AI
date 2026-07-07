"""Offline ball-tracking replay from an existing debug CSV (no YOLO / no video)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Backends.src.tracking.trajectory_scorer import (  # noqa: E402
    TrajectoryBallSelector,
    default_max_link_distance_px,
    extend_ranked_tracklet,
    rank_candidate_tracklets,
)


@dataclass(slots=True)
class ReplayConfig:
    min_conf: float = 0.01
    max_frame_gap: int = 5
    max_link_distance_px: float | None = None
    min_tracklet_points: int = 3
    min_total_movement_px: float = 12.0
    min_average_movement_px: float = 0.5
    top_k: int = 10


def _parse_float(value, default=None):
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default=None):
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_ball_candidates_from_csv(
    csv_path: Path,
    *,
    min_conf: float = 0.01,
) -> tuple[dict[int, list[dict]], dict]:
    """Read debug CSV and group raw ball detections by frame_index.

    ponytail: intentionally ignores online tracker columns such as
    candidate_selected, candidate_rejected, rejection_reason, and
    tracking_quality_so_far. Replay only uses raw ball geometry + confidence.
    """
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    frame_width = _parse_int(rows[0].get("frame_width"), 1280) or 1280
    frame_height = _parse_int(rows[0].get("frame_height"), 720) or 720
    fps = _estimate_fps(rows)

    frame_candidates: dict[int, list[dict]] = {}
    total_ball_rows = 0
    filtered_out_by_confidence_count = 0
    frames_40_76_candidate_count = 0

    for row in rows:
        if row.get("class_name") != "ball":
            continue

        center_x = _parse_int(row.get("center_x"))
        center_y = _parse_int(row.get("center_y"))
        if center_x is None or center_y is None:
            continue

        confidence = _parse_float(row.get("confidence"), 0.0) or 0.0
        if confidence < min_conf:
            filtered_out_by_confidence_count += 1
            continue

        frame_index = _parse_int(row.get("frame_index"))
        if frame_index is None:
            continue

        detection = {
            "center": (center_x, center_y),
            "confidence": confidence,
            "box": (
                _parse_int(row.get("bbox_x1"), center_x - 5) or center_x - 5,
                _parse_int(row.get("bbox_y1"), center_y - 5) or center_y - 5,
                _parse_int(row.get("bbox_x2"), center_x + 5) or center_x + 5,
                _parse_int(row.get("bbox_y2"), center_y + 5) or center_y + 5,
            ),
            "class_name": "ball",
            "candidate_id": row.get("candidate_id"),
        }
        frame_candidates.setdefault(frame_index, []).append(detection)
        total_ball_rows += 1
        if 40 <= frame_index <= 76:
            frames_40_76_candidate_count += 1

    metadata = {
        "csv_path": str(csv_path),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": fps,
        "min_conf": min_conf,
        "total_csv_rows": len(rows),
        "raw_candidate_count_used_for_replay": total_ball_rows,
        "filtered_out_by_confidence_count": filtered_out_by_confidence_count,
        "candidate_frames_used": len(frame_candidates),
        "frame_index_min": min(frame_candidates) if frame_candidates else None,
        "frame_index_max": max(frame_candidates) if frame_candidates else None,
        "frames_40_76_candidate_count": frames_40_76_candidate_count,
    }
    return frame_candidates, metadata


def _estimate_fps(rows: list[dict]) -> float:
    by_frame: dict[int, float] = {}
    for row in rows:
        frame_index = _parse_int(row.get("frame_index"))
        timestamp = _parse_float(row.get("timestamp_sec"))
        if frame_index is None or timestamp is None:
            continue
        by_frame[frame_index] = timestamp

    frames = sorted(by_frame)
    if len(frames) < 2:
        return 25.0

    deltas = []
    for index in range(1, len(frames)):
        prev_frame = frames[index - 1]
        current_frame = frames[index]
        if current_frame - prev_frame != 1:
            continue
        delta = by_frame[current_frame] - by_frame[prev_frame]
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return 25.0
    avg_delta = sum(deltas) / len(deltas)
    return round(1.0 / avg_delta, 3) if avg_delta > 0 else 25.0


def replay_online_selector(
    frame_candidates: dict[int, list[dict]],
    *,
    frame_width: int,
    frame_height: int,
    fps: float | None,
) -> dict:
    """Replay the online selector for comparison (shows termination side effects)."""
    selector = TrajectoryBallSelector(frame_width, frame_height)
    rejection_reasons: Counter[str] = Counter()
    termination_frame = None

    all_frames = sorted(frame_candidates.keys())
    if all_frames:
        min_frame = all_frames[0]
        max_frame = all_frames[-1]
        for frame_index in range(min_frame, max_frame + 1):
            detections = frame_candidates.get(frame_index, [])
            selector.select(detections, frame_index=frame_index)
            if selector._track_terminated and termination_frame is None:
                termination_frame = frame_index

    for reason, count in selector.rejection_reasons.items():
        rejection_reasons[reason] = count

    accepted = [
        {
            "frame_index": frame_index,
            "x": position[0],
            "y": position[1],
        }
        for frame_index, position in zip(
            selector._accepted_frame_indices,
            selector.accepted_positions,
        )
    ]

    return {
        "online_selected_point_count": len(accepted),
        "online_selected_start_frame": accepted[0]["frame_index"] if accepted else None,
        "online_selected_end_frame": accepted[-1]["frame_index"] if accepted else None,
        "online_track_terminated": selector._track_terminated,
        "online_termination_frame": termination_frame,
        "online_rejection_reasons": dict(rejection_reasons.most_common(10)),
        "after_track_terminated_count": rejection_reasons.get("after_track_terminated", 0),
        "selector_debug": selector.debug_summary("Replay", fps=fps),
        "accepted_points": accepted,
    }


def write_ranked_segments_csv(path: Path, segments: list[dict]) -> None:
    fieldnames = [
        "rank",
        "start_frame",
        "end_frame",
        "point_count",
        "frame_span",
        "duration_sec",
        "total_movement",
        "average_movement_per_frame",
        "static_penalty",
        "edge_penalty",
        "smoothness_score",
        "confidence_score",
        "final_segment_score",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for segment in segments:
            writer.writerow(segment)


def run_replay(csv_path: Path, config: ReplayConfig | None = None) -> dict:
    config = config or ReplayConfig()
    frame_candidates, metadata = load_ball_candidates_from_csv(
        csv_path,
        min_conf=config.min_conf,
    )

    max_link = config.max_link_distance_px
    if max_link is None:
        max_link = default_max_link_distance_px(
            metadata["frame_width"],
            metadata["frame_height"],
        )

    ranking = rank_candidate_tracklets(
        frame_candidates,
        metadata["frame_width"],
        metadata["frame_height"],
        metadata["fps"],
        top_n=config.top_k,
        max_frame_gap=config.max_frame_gap,
        max_link_distance_px=max_link,
        min_tracklet_points=config.min_tracklet_points,
        min_total_movement_px=config.min_total_movement_px,
        min_average_movement_px=config.min_average_movement_px,
    )
    online = replay_online_selector(
        frame_candidates,
        frame_width=metadata["frame_width"],
        frame_height=metadata["frame_height"],
        fps=metadata["fps"],
    )

    winner = ranking.get("winner") or {}
    extension = {}
    if winner.get("tracklet_points"):
        extension = extend_ranked_tracklet(
            winner["tracklet_points"],
            frame_candidates,
            metadata["frame_width"],
            metadata["frame_height"],
            metadata["fps"],
        )
    online_start = online.get("online_selected_start_frame")
    online_end = online.get("online_selected_end_frame")

    export_segments = ranking["top_segments"] or ranking.get("nearest_failed_segments", [])

    summary = {
        **metadata,
        "replay_config": {
            "min_conf": config.min_conf,
            "max_frame_gap": config.max_frame_gap,
            "max_link_distance_px": round(max_link, 2),
            "min_tracklet_points": config.min_tracklet_points,
            "min_total_movement_px": config.min_total_movement_px,
            "min_average_movement_px": config.min_average_movement_px,
            "top_k": config.top_k,
        },
        "ranking": {
            "candidate_segment_count": ranking["candidate_segment_count"],
            "rejected_static_segment_count": ranking["rejected_static_segment_count"],
            "total_valid_tracklets": ranking["total_valid_tracklets"],
            "top_segment_count": len(ranking["top_segments"]),
            "winner_start_frame": winner.get("start_frame"),
            "winner_end_frame": winner.get("end_frame"),
            "winner_point_count": winner.get("point_count"),
            "winner_total_movement": winner.get("total_movement"),
            "winner_final_segment_score": winner.get("final_segment_score"),
            "winner_reason": winner.get("reason"),
            "extension_enabled": extension.get("extension_enabled", False),
            "extension_applied": extension.get("extension_applied", False),
            "backward_extension_points": extension.get("backward_extension_points", 0),
            "forward_extension_points": extension.get("forward_extension_points", 0),
            "extended_segment_start_frame": extension.get(
                "extended_segment_start_frame"
            ),
            "extended_segment_end_frame": extension.get("extended_segment_end_frame"),
            "extended_segment_point_count": extension.get(
                "extended_segment_point_count",
                winner.get("point_count"),
            ),
            "extension_rejection_reasons": extension.get(
                "extension_rejection_reasons",
                {},
            ),
            "extension_fallback_reason": extension.get("extension_fallback_reason"),
            "extension_preserved_original_segment": extension.get(
                "extension_preserved_original_segment",
                not extension.get("extension_applied", False),
            ),
            "extension_fit_delta": extension.get("extension_fit_delta", 0),
            "trajectory_fit_quality_after_extension": extension.get(
                "trajectory_fit_quality_after_extension"
            ),
            "frames_40_76_in_any_ranked_segment": ranking[
                "frames_40_76_in_any_ranked_segment"
            ],
            "ranking_applies_after_track_terminated": ranking[
                "ranking_applies_after_track_terminated"
            ],
            "why_no_segments": ranking.get("why_no_segments"),
            "static_rejection_reason_counts": (
                (ranking.get("why_no_segments") or {}).get("static_rejection_reason_counts")
                or _static_rejection_reason_counts(ranking.get("rejected_static_segments") or [])
            ),
            "nearest_failed_segment_count": len(
                ranking.get("nearest_failed_segments") or []
            ),
        },
        "online_selector": {
            "selected_start_frame": online_start,
            "selected_end_frame": online_end,
            "selected_point_count": online.get("online_selected_point_count"),
            "track_terminated": online.get("online_track_terminated"),
            "termination_frame": online.get("online_termination_frame"),
            "after_track_terminated_count": online.get("after_track_terminated_count"),
            "main_rejection_reasons": online.get("online_rejection_reasons"),
        },
        "comparison": {
            "offline_winner_differs_from_online": (
                winner.get("start_frame") != online_start
                or winner.get("end_frame") != online_end
            ),
            "why_online_may_differ": _why_online_differs(winner, online, ranking),
        },
    }

    stem = csv_path.with_suffix("")
    summary_path = Path(f"{stem}_replay_summary.json")
    ranked_json_path = Path(f"{stem}_ranked_segments.json")
    ranked_csv_path = Path(f"{stem}_ranked_segments.csv")
    failed_json_path = Path(f"{stem}_nearest_failed_segments.json")
    rejected_static_path = Path(f"{stem}_rejected_static_segments.json")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ranked_json_path.write_text(json.dumps(export_segments, indent=2), encoding="utf-8")
    write_ranked_segments_csv(ranked_csv_path, export_segments)
    failed_json_path.write_text(
        json.dumps(ranking.get("nearest_failed_segments") or [], indent=2),
        encoding="utf-8",
    )
    rejected_static_path.write_text(
        json.dumps(ranking.get("rejected_static_segments") or [], indent=2),
        encoding="utf-8",
    )

    return {
        "summary": summary,
        "ranking": ranking,
        "online": online,
        "output_files": {
            "summary_json": str(summary_path),
            "ranked_segments_json": str(ranked_json_path),
            "ranked_segments_csv": str(ranked_csv_path),
            "nearest_failed_segments_json": str(failed_json_path),
            "rejected_static_segments_json": str(rejected_static_path),
        },
    }


def _static_rejection_reason_counts(rejected_segments: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for segment in rejected_segments:
        reason = segment.get("rejection_reason", "near_static_segment")
        counts[reason] += 1
    return dict(counts.most_common())


def _why_online_differs(winner: dict, online: dict, ranking: dict) -> str:
    parts = []
    if online.get("after_track_terminated_count", 0) > 0:
        parts.append(
            "online_selector_rejects_late_candidates_after_track_terminated"
        )
    if winner and online.get("online_selected_start_frame") == winner.get("start_frame"):
        parts.append("same_start_but_online_may_end_early")
    elif winner:
        parts.append("offline_ranking_considers_all_frames_before_termination")
    if ranking.get("frames_40_76_in_any_ranked_segment"):
        parts.append("frames_40_76_present_in_offline_ranking")
    else:
        parts.append("frames_40_76_not_in_any_valid_offline_tracklet")
    return ";".join(parts) if parts else "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay ball tracking segment ranking from a debug CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to ball_tracking_debug.csv")
    parser.add_argument("--min-conf", type=float, default=0.01, help="Min ball confidence")
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=5,
        help="Max frame gap when linking tracklet points",
    )
    parser.add_argument(
        "--max-link-distance-px",
        type=float,
        default=None,
        help="Max pixel distance between linked points (default: 12%% of min frame side)",
    )
    parser.add_argument(
        "--min-tracklet-points",
        type=int,
        default=3,
        help="Minimum points required for a ranked tracklet",
    )
    parser.add_argument(
        "--min-total-movement-px",
        type=float,
        default=12.0,
        help="Reject tracklets with endpoint displacement below this",
    )
    parser.add_argument(
        "--min-average-movement-px",
        type=float,
        default=0.5,
        help="Reject tracklets with average step movement below this",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top segments to export")
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        print(f"CSV not found: {args.csv_path}", file=sys.stderr)
        return 1

    config = ReplayConfig(
        min_conf=args.min_conf,
        max_frame_gap=args.max_frame_gap,
        max_link_distance_px=args.max_link_distance_px,
        min_tracklet_points=args.min_tracklet_points,
        min_total_movement_px=args.min_total_movement_px,
        min_average_movement_px=args.min_average_movement_px,
        top_k=args.top_k,
    )
    result = run_replay(args.csv_path, config=config)
    print(json.dumps(result["summary"], indent=2))
    print("\nOutput files:")
    for label, path in result["output_files"].items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
