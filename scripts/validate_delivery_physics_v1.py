"""Run Physics Engine V1 on persisted Video Analysis tracker outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.api.schemas.delivery_physics import DeliveryPhysicsResult  # noqa: E402
from services.api.schemas.video_analysis import (  # noqa: E402
    VideoBallDetectionsDocument,
    VideoBallTrackingDocument,
)
from services.api.services.delivery_physics_service import (  # noqa: E402
    analyse_delivery_physics,
)
from services.api.services.video_analysis_service import (  # noqa: E402
    VIDEO_ANALYSIS_ROOT,
    load_video_analysis,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "physics_validation_v1"


def _write_contact_sheet(
    *,
    analysis: Any,
    result: DeliveryPhysicsResult,
    destination: Path,
) -> str | None:
    samples = result.trajectory_samples
    if not samples:
        return None
    source = (
        VIDEO_ANALYSIS_ROOT
        / result.analysis_id
        / "raw"
        / analysis.stored_filename
    )
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        return None

    count = min(12, len(samples))
    selected_indexes = np.linspace(0, len(samples) - 1, count, dtype=int)
    selected = [samples[int(index)] for index in selected_indexes]
    observed = {
        item.frame_index: item
        for item in result.accepted_observations
    }
    colours = {
        "OBSERVED": (80, 224, 120),
        "RECONSTRUCTED": (104, 202, 255),
        "PROJECTED": (173, 165, 154),
    }
    tiles: list[np.ndarray] = []
    try:
        for sample in selected:
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            detector_point = observed.get(sample.frame_index)
            if detector_point is not None:
                cv2.circle(
                    frame,
                    (
                        int(round(detector_point.pixel_x)),
                        int(round(detector_point.pixel_y)),
                    ),
                    8,
                    (255, 255, 255),
                    2,
                )
            colour = colours[sample.provenance]
            cv2.drawMarker(
                frame,
                (int(round(sample.pixel_x)), int(round(sample.pixel_y))),
                colour,
                cv2.MARKER_CROSS,
                16,
                2,
            )
            if (
                result.bounce.frame_index == sample.frame_index
                and result.bounce.pixel_x is not None
                and result.bounce.pixel_y is not None
            ):
                cv2.circle(
                    frame,
                    (
                        int(round(result.bounce.pixel_x)),
                        int(round(result.bounce.pixel_y)),
                    ),
                    16,
                    (40, 110, 255),
                    3,
                )
            cv2.rectangle(frame, (0, 0), (360, 32), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"F{sample.frame_index} {sample.provenance}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colour,
                2,
                cv2.LINE_AA,
            )
            scale = 320 / frame.shape[1]
            tiles.append(
                cv2.resize(
                    frame,
                    (320, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            )
    finally:
        capture.release()

    if not tiles:
        return None
    columns = 4
    tile_height = max(tile.shape[0] for tile in tiles)
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * 320, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[
            row * tile_height:row * tile_height + tile.shape[0],
            column * 320:(column + 1) * 320,
        ] = tile
    path = destination / "diagnostic_contact_sheet.jpg"
    if not cv2.imwrite(str(path), sheet):
        return None
    return str(path)


def validate_analysis(analysis_id: str, output_root: Path) -> dict[str, Any]:
    analysis_root = VIDEO_ANALYSIS_ROOT / analysis_id
    destination = output_root / analysis_id
    destination.mkdir(parents=True, exist_ok=True)
    try:
        analysis = load_video_analysis(analysis_id)
        detections = VideoBallDetectionsDocument.model_validate_json(
            (analysis_root / "detections" / "detections.json").read_text(
                encoding="utf-8"
            )
        )
        tracking = VideoBallTrackingDocument.model_validate_json(
            (analysis_root / "tracking" / "tracking_result.json").read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            detail = (
                f"{exc.error_count()} schema validation errors; "
                f"first error: {exc.errors()[0]['loc']} "
                f"{exc.errors()[0]['msg']}"
            )
        else:
            detail = str(exc)
        summary = {
            "analysis_id": analysis_id,
            "status": "INCOMPATIBLE_HISTORICAL_INPUT",
            "failure_reason": detail,
            "warnings": [
                "Persisted detector/tracker artifacts are missing or do not "
                "match the current production schema."
            ],
        }
        (destination / "diagnostic_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary

    result = analyse_delivery_physics(
        analysis_id=analysis_id,
        primary_track=tracking.primary_track,
        detections=detections,
        tracker_bounce=tracking.bounce,
        fps=analysis.fps,
        width=analysis.width,
        height=analysis.height,
        total_frames=analysis.frame_count,
    )
    (destination / "physics_result.json").write_text(
        json.dumps(result.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    contact_sheet = _write_contact_sheet(
        analysis=analysis,
        result=result,
        destination=destination,
    )
    summary = {
        "analysis_id": analysis_id,
        "status": result.status,
        "calibration_mode": result.calibration.mode,
        "calibration_confidence": result.calibration.confidence,
        "accepted_observations": len(result.accepted_observations),
        "rejected_observations": len(result.rejected_observations),
        "trajectory_samples": len(result.trajectory_samples),
        "observed_samples": sum(
            item.provenance == "OBSERVED"
            for item in result.trajectory_samples
        ),
        "reconstructed_samples": sum(
            item.provenance == "RECONSTRUCTED"
            for item in result.trajectory_samples
        ),
        "projected_samples": sum(
            item.provenance == "PROJECTED"
            for item in result.trajectory_samples
        ),
        "fit_rmse_px": result.fit_diagnostics.weighted_reprojection_rmse_px,
        "bounce_status": result.bounce.status,
        "bounce_frame": result.bounce.frame_index,
        "earliest_measured_speed_kmh": (
            result.speed.earliest_measured_speed_kmh
        ),
        "line": result.line_and_length.line,
        "length": result.line_and_length.length,
        "confidence": result.confidence,
        "diagnostic_contact_sheet": contact_sheet,
        "warnings": result.warnings,
    }
    (destination / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis_ids", nargs="+")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    summaries = [
        validate_analysis(analysis_id, args.output_root)
        for analysis_id in args.analysis_ids
    ]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
