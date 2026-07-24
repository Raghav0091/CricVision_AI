"""Run the real RTMPose provider on a prepared CricVision analysis.

This is a developer-only spike runner. It reads only the clean raw uploaded
video and writes pose debug artifacts under the analysis reports folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Backends.src.release_point.bowler_tracker import BowlerTracker
from Backends.src.release_point.pose_provider import Keypoint, PosePerson
from Backends.src.release_point.rtmpose_provider import (
    RTMPoseProvider,
    RTMPoseProviderConfig,
    RTMPoseProviderUnavailable,
    infer_bowling_arm,
)
from services.api.services import video_release_point_service as release_service
from services.api.services.video_analysis_service import VIDEO_ANALYSIS_ROOT


SKELETON = (
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
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_id")
    parser.add_argument("--pose-model", default="human")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-debug-frames", type=int, default=12)
    args = parser.parse_args()

    release_input = release_service.validate_video_release_point_input(args.analysis_id)
    tracking = release_service._load_tracking_document(release_input)
    calibration = release_service._read_json(
        Path(release_input.calibration_path),
        "calibration.json",
    )
    calibration_v2 = (
        release_service._read_json(Path(release_input.calibration_v2_path), "calibration_v2.json")
        if release_input.calibration_v2_path
        else None
    )
    frame_window = release_service._pose_frame_window(
        tracking,
        release_input.frame_count,
    )
    provider = RTMPoseProvider(
        RTMPoseProviderConfig(
            pose2d=args.pose_model,
            model_name=args.pose_model,
            device=args.device,
        )
    )
    report_dir = VIDEO_ANALYSIS_ROOT / args.analysis_id / "reports"
    debug_dir = report_dir / "rtmpose_debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        pose_sequence = provider.estimate_sequence(
            release_input.raw_video_path,
            frame_window,
            fps=release_input.fps,
        )
    except RTMPoseProviderUnavailable as exc:
        _write_json(
            report_dir / "rtmpose_validation.json",
            {
                "status": "unavailable",
                "analysis_id": args.analysis_id,
                "pose_provider": provider.provider_info.to_dict(),
                "message": str(exc),
            },
        )
        return 2

    bowler = BowlerTracker().track(
        pose_sequence,
        scene_calibration=calibration,
        pitch_context=calibration_v2,
        ball_track=tracking.get("primary_track", []),
    )
    arm = infer_bowling_arm(bowler.poses_by_frame)
    timings = {
        "total_seconds": round(time.perf_counter() - started, 4),
        "frames_requested": len(frame_window),
        "frames_with_pose": len(pose_sequence.frames),
        "seconds_per_requested_frame": round(
            (time.perf_counter() - started) / max(1, len(frame_window)),
            4,
        ),
    }
    debug_files = _write_debug_frames(
        Path(release_input.raw_video_path),
        debug_dir,
        bowler.poses_by_frame,
        max_frames=args.max_debug_frames,
    )
    report = {
        "status": "ready" if bowler.poses_by_frame else "pose_insufficient",
        "analysis_id": args.analysis_id,
        "raw_video_path": release_input.raw_video_path,
        "pose_provider": provider.provider_info.to_dict(),
        "frame_window": {
            "start": min(frame_window) if frame_window else None,
            "end": max(frame_window) if frame_window else None,
            "count": len(frame_window),
        },
        "bowler": bowler.to_dict(),
        "bowling_arm": arm,
        "timings": timings,
        "debug_frames": debug_files,
    }
    _write_json(report_dir / "rtmpose_validation.json", report)
    print(json.dumps(report, indent=2))
    return 0 if bowler.poses_by_frame else 3


def _write_debug_frames(
    raw_video_path: Path,
    debug_dir: Path,
    poses_by_frame: dict[int, PosePerson],
    *,
    max_frames: int,
) -> list[str]:
    from Backends.src.utils.cv2_loader import cv2

    paths: list[str] = []
    capture = cv2.VideoCapture(str(raw_video_path))
    try:
        for frame_index in sorted(poses_by_frame)[:max_frames]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            _draw_pose(frame, poses_by_frame[frame_index])
            out = debug_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            paths.append(str(out))
    finally:
        capture.release()
    return paths


def _draw_pose(frame: Any, pose: PosePerson) -> None:
    from Backends.src.utils.cv2_loader import cv2

    x1, y1, x2, y2 = [int(value) for value in pose.bbox_xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (195, 255, 83), 2)
    cv2.putText(
        frame,
        f"frame {pose.frame_index} conf {pose.confidence:.2f}",
        (max(0, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (195, 255, 83),
        2,
        cv2.LINE_AA,
    )
    for a, b in SKELETON:
        first = pose.keypoint(a)
        second = pose.keypoint(b)
        if _usable(first) and _usable(second):
            cv2.line(
                frame,
                (int(first.x), int(first.y)),
                (int(second.x), int(second.y)),
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )
    for name, point in pose.keypoints.items():
        if not _usable(point):
            continue
        color = (80, 220, 255)
        radius = 4
        if name.endswith("_wrist"):
            color = (90, 120, 255)
            radius = 6
            cv2.putText(
                frame,
                f"{name} {point.confidence:.2f}",
                (int(point.x) + 7, int(point.y) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.circle(frame, (int(point.x), int(point.y)), radius, color, -1, cv2.LINE_AA)


def _usable(point: Keypoint | None) -> bool:
    return point is not None and point.confidence >= 0.2


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
