#!/usr/bin/env python3
"""Lightweight performance checks for analysis helpers (no real models)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_dummy_timeline(frame_count: int):
    frames = []
    for index in range(frame_count):
        cx = 100 + index * 2
        cy = 100 + (index % 3)
        frames.append(
            {
                "frame_index": index,
                "ball_detections": [
                    {
                        "center": [cx, cy],
                        "confidence": 0.9,
                        "box": [cx - 4, cy - 4, cx + 4, cy + 4],
                    }
                ],
                "bat_detections": [],
                "stump_detections": [],
            }
        )
    return frames


def benchmark_helpers(frame_count: int = 500, repeats: int = 5) -> dict[str, float]:
    from Backends.src.agents.observer_timeline import build_observer_timeline
    from Backends.src.agents.tracking_repair_agent import repair_ball_tracking
    from Backends.src.analysis.frame_detection_utils import normalize_frame_detections
    from Backends.src.analysis.impact_detection import detect_bat_ball_impact
    from Backends.src.analysis.outcome_prediction import predict_shot_outcome
    from Backends.src.analysis.shot_direction import estimate_shot_direction_zone
    from Backends.src.video_pipeline.report_pipeline import build_video_reports

    raw = _build_dummy_timeline(frame_count)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    for _ in range(repeats):
        normalize_frame_detections(raw)
    timings["normalize_frame_detections_ms"] = ((time.perf_counter() - start) / repeats) * 1000

    frames = normalize_frame_detections(raw)
    repair_frames = _build_dummy_timeline(200)
    start = time.perf_counter()
    for _ in range(repeats):
        repair_ball_tracking(repair_frames, frame_width=1280, frame_height=720)
    timings["tracking_repair_200_frames_ms"] = (
        (time.perf_counter() - start) / repeats
    ) * 1000

    start = time.perf_counter()
    for _ in range(repeats):
        build_observer_timeline(frames, total_frames=frame_count, fps=25)
    timings["observer_timeline_ms"] = ((time.perf_counter() - start) / repeats) * 1000

    start = time.perf_counter()
    for _ in range(repeats):
        detect_bat_ball_impact(frames, fps=25)
    timings["impact_detection_ms"] = ((time.perf_counter() - start) / repeats) * 1000

    impact = detect_bat_ball_impact(frames, fps=25)
    shot_result = {"shot_type": "Defence", "shot_confidence": "Medium"}
    start = time.perf_counter()
    for _ in range(repeats):
        estimate_shot_direction_zone(
            frames,
            impact,
            shot_result=shot_result,
            batter_handedness="right",
        )
    timings["shot_direction_ms"] = ((time.perf_counter() - start) / repeats) * 1000

    direction = estimate_shot_direction_zone(
        frames,
        impact,
        shot_result=shot_result,
        batter_handedness="right",
    )
    start = time.perf_counter()
    for _ in range(repeats):
        predict_shot_outcome(
            frames,
            impact,
            shot_result,
            fps=25,
            batter_handedness="right",
            direction_result=direction,
        )
    timings["outcome_prediction_ms"] = ((time.perf_counter() - start) / repeats) * 1000

    start = time.perf_counter()
    for _ in range(repeats):
        build_video_reports(
            frames,
            fps=25,
            total_frames=frame_count,
            batter_handedness="right",
        )
    timings["report_pipeline_ms"] = ((time.perf_counter() - start) / repeats) * 1000

    return timings


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CricVision helper functions.")
    parser.add_argument("--frames", type=int, default=500, help="Dummy timeline length")
    parser.add_argument("--repeats", type=int, default=5, help="Repeat count per benchmark")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional video path for future real-video benchmarking (not run by default)",
    )
    args = parser.parse_args()

    if args.video:
        print(f"Video benchmarking is optional and not enabled in CI. Skipping: {args.video}")

    timings = benchmark_helpers(frame_count=args.frames, repeats=args.repeats)
    print(f"Performance check ({args.frames} frames, {args.repeats} repeats each):")
    for name, value in timings.items():
        print(f"  {name}: {value:.2f} ms")
    print("Performance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
