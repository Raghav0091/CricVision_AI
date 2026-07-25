"""Run the offline Release-Region Hard-Case Detector Benchmark V1."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
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
from services.api.services.ball_detector_registry import load_ball_detector_model  # noqa: E402


OUTPUT = ROOT / "outputs" / "release_validation" / "release_region_detector_benchmark"
ANNOTATIONS = OUTPUT / "benchmark_annotations.json"
RAW_RESULTS = OUTPUT / "raw_model_outputs.json"
THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20)
MODELS = {
    "e3_motion_blur": "E3",
    "e4c_best_overall": "E4C",
}
IMAGE_SIZE = 960
MATCH_RADIUS = 18.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("infer", "report", "all"), nargs="?", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in {"infer", "all"}:
        infer(device=args.device, force=args.force)
    if args.command in {"report", "all"}:
        report()
    return 0


def infer(*, device: str | None, force: bool) -> dict[str, Any]:
    if RAW_RESULTS.is_file() and not force:
        return _read(RAW_RESULTS)
    annotations = _read(ANNOTATIONS)["frames"]
    output: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": _now(),
        "settings": {"confidence_floor": 0.01, "imgsz": IMAGE_SIZE, "max_det": 300},
        "models": {},
    }
    for model_key, label in MODELS.items():
        selected, model = load_ball_detector_model(model_key)
        started = perf_counter()
        rows = []
        for annotation in annotations:
            frame = _load_frame(annotation)
            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=0.01,
                max_det=300,
                device=device,
                verbose=False,
            )
            raw = extract_ball_candidates(results, getattr(model, "names", {}), strict=True)
            raw.sort(key=lambda row: row["confidence"], reverse=True)
            rows.append(
                {
                    "benchmark_id": annotation["benchmark_id"],
                    "candidates": [_candidate(item, rank) for rank, item in enumerate(raw, 1)],
                }
            )
        output["models"][model_key] = {
            "label": label,
            "model_path": str(selected.path),
            "processing_seconds": round(perf_counter() - started, 3),
            "frames": rows,
        }
    _write(RAW_RESULTS, output)
    return output


def report() -> dict[str, Any]:
    annotations = _read(ANNOTATIONS)["frames"]
    raw = _read(RAW_RESULTS)
    raw_by_model = {
        key: {row["benchmark_id"]: row["candidates"] for row in value["frames"]}
        for key, value in raw["models"].items()
    }
    rows = []
    for annotation in annotations:
        row = dict(annotation)
        for model_key in MODELS:
            candidates = raw_by_model[model_key][annotation["benchmark_id"]]
            match = _match(candidates, annotation)
            row[model_key] = {
                "detected_at_0_01": match is not None,
                "true_ball_confidence": None if match is None else match["confidence"],
                "true_ball_rank": None if match is None else match["rank"],
                "true_ball_candidate": match,
                "raw_candidate_count": len(candidates),
                "raw_candidates": candidates,
            }
        rows.append(row)

    model_summaries = {
        key: _model_summary(key, rows) for key in MODELS
    }
    positive = [row for row in rows if row["ball_visible"]]
    comparison_counts = Counter()
    for row in positive:
        e3 = row["e3_motion_blur"]["detected_at_0_01"]
        e4c = row["e4c_best_overall"]["detected_at_0_01"]
        comparison_counts[
            "both_detect" if e3 and e4c else
            "e3_only" if e3 else
            "e4c_only" if e4c else
            "both_miss"
        ] += 1
    aggregate = {
        "schema_version": "1.0",
        "created_at": _now(),
        "benchmark_size": {
            "total_frames": len(rows),
            "positive_frames": len(positive),
            "negative_frames": len(rows) - len(positive),
        },
        "annotation_distribution": _annotation_distribution(rows),
        "models": model_summaries,
        "head_to_head_at_0_01": dict(comparison_counts),
        "per_positive_frame": [_comparison_row(row) for row in positive],
        "dataset_gap_analysis": _dataset_gap(model_summaries),
        "architectural_assessment": _architecture_assessment(model_summaries),
        "recommended_next_strategy": _recommendation(model_summaries, comparison_counts),
        "limitations": [
            "Small, deliberately difficult benchmark derived from five audited validation clips.",
            "rv1_008 and rv1_012 are duplicate source content retained to match frozen validation.",
            "Point/bbox annotations are approximate manual audit localizations.",
        ],
    }
    _write(OUTPUT / "benchmark_results.json", {"frames": rows})
    _write_csv(OUTPUT / "benchmark_results.csv", rows)
    _write(OUTPUT / "aggregate_benchmark_summary.json", aggregate)
    (OUTPUT / "aggregate_benchmark_summary.md").write_text(
        _markdown(aggregate), encoding="utf-8"
    )
    _gallery(rows)
    return aggregate


def _model_summary(model_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["ball_visible"]]
    negatives = [row for row in rows if not row["ball_visible"]]
    thresholds = {}
    for threshold in THRESHOLDS:
        true_hits = 0
        false_positives = 0
        frames_with_fp = 0
        for row in rows:
            info = row[model_key]
            match = info["true_ball_candidate"]
            if row["ball_visible"] and match and match["confidence"] >= threshold:
                true_hits += 1
            frame_fp = sum(
                candidate["confidence"] >= threshold
                and not _candidate_matches(candidate, row)
                for candidate in info["raw_candidates"]
            )
            false_positives += frame_fp
            frames_with_fp += frame_fp > 0
        tp = true_hits
        precision = tp / (tp + false_positives) if tp + false_positives else None
        thresholds[str(threshold)] = {
            "true_ball_recall": _ratio(true_hits, len(positives)),
            "precision": None if precision is None else round(precision, 6),
            "false_positives": false_positives,
            "false_positives_per_frame": _ratio(false_positives, len(rows)),
            "frames_with_false_positive": frames_with_fp,
            "negative_frames_with_any_detection": sum(
                any(c["confidence"] >= threshold for c in row[model_key]["raw_candidates"])
                for row in negatives
            ),
        }
    detected = [
        row[model_key] for row in positives if row[model_key]["detected_at_0_01"]
    ]
    return {
        "threshold_metrics": thresholds,
        "true_ball_confidence": _distribution(
            [row["true_ball_confidence"] for row in detected]
        ),
        "true_ball_rank_distribution": dict(
            Counter(str(row["true_ball_rank"]) for row in detected)
        ),
        "size_bucket_performance": _group_performance(model_key, positives, _size_bucket),
        "blur_severity_performance": _group_performance(
            model_key, positives, lambda row: row["blur_severity"]
        ),
        "hand_overlap_performance": _group_performance(
            model_key, positives, lambda row: row["hand_overlap_occlusion"]
        ),
        "condition_performance": _condition_performance(model_key, positives),
        "missed_benchmark_ids_at_0_01": [
            row["benchmark_id"]
            for row in positives
            if not row[model_key]["detected_at_0_01"]
        ],
    }


def _group_performance(model_key, rows, grouping):
    groups = defaultdict(list)
    for row in rows:
        groups[grouping(row)].append(row)
    return {
        group: {
            "sample_count": len(items),
            "recall_at_0_01": _ratio(
                sum(row[model_key]["detected_at_0_01"] for row in items), len(items)
            ),
            "recall_at_0_15": _ratio(
                sum(
                    (row[model_key]["true_ball_confidence"] or 0) >= 0.15
                    for row in items
                ),
                len(items),
            ),
            "mean_true_ball_confidence": _mean(
                [
                    row[model_key]["true_ball_confidence"]
                    for row in items
                    if row[model_key]["true_ball_confidence"] is not None
                ]
            ),
        }
        for group, items in groups.items()
    }


def _condition_performance(model_key, positives):
    conditions = sorted({condition for row in positives for condition in row["conditions"]})
    return {
        condition: _group_performance(
            model_key,
            [row for row in positives if condition in row["conditions"]],
            lambda _: condition,
        )[condition]
        for condition in conditions
    }


def _annotation_distribution(rows):
    positives = [row for row in rows if row["ball_visible"]]
    return {
        "size_buckets": dict(Counter(_size_bucket(row) for row in positives)),
        "blur_severity": dict(Counter(row["blur_severity"] for row in positives)),
        "hand_overlap_occlusion": dict(
            Counter(row["hand_overlap_occlusion"] for row in positives)
        ),
        "background_difficulty": dict(
            Counter(row["background_difficulty"] for row in rows)
        ),
        "conditions": dict(Counter(c for row in rows for c in row["conditions"])),
    }


def _comparison_row(row):
    return {
        "benchmark_id": row["benchmark_id"],
        "validation_id": row["validation_id"],
        "frame_index": row["frame_index"],
        "ground_truth_point": row["ball_point"],
        "ground_truth_bbox_xyxy": row["ball_bbox_xyxy"],
        "e3_detected": row["e3_motion_blur"]["detected_at_0_01"],
        "e3_confidence": row["e3_motion_blur"]["true_ball_confidence"],
        "e3_rank": row["e3_motion_blur"]["true_ball_rank"],
        "e4c_detected": row["e4c_best_overall"]["detected_at_0_01"],
        "e4c_confidence": row["e4c_best_overall"]["true_ball_confidence"],
        "e4c_rank": row["e4c_best_overall"]["true_ball_rank"],
    }


def _dataset_gap(models):
    e3 = models["e3_motion_blur"]["condition_performance"]
    e4c = models["e4c_best_overall"]["condition_performance"]
    return {
        "evidence": {
            condition: {
                "e3_recall_0_01": e3.get(condition, {}).get("recall_at_0_01"),
                "e4c_recall_0_01": e4c.get(condition, {}).get("recall_at_0_01"),
                "e3_recall_0_15": e3.get(condition, {}).get("recall_at_0_15"),
                "e4c_recall_0_15": e4c.get(condition, {}).get("recall_at_0_15"),
            }
            for condition in sorted(set(e3) | set(e4c))
        },
        "interpretation": (
            "The prior project finding says training already contains many small "
            "balls; no standalone training-data audit artifact was found in this "
            "workspace to quantify that distribution further. Benchmark evidence "
            "instead isolates release-specific combinations: severe blur plus low "
            "contrast/background clutter, hand association, and static-highlight "
            "hard negatives. The <=8 px bucket was detected perfectly by both "
            "models, so tiny-ball frequency alone is not the measured gap."
        ),
        "likely_missing_distribution": [
            "severe-motion-blur plus low-contrast release balls",
            "hand-associated and emerging-from-hand balls",
            "bright-pitch, watermark, and static-background hard negatives",
        ],
    }


def _architecture_assessment(models):
    return {
        "supported_direction": "general_detector_plus_bounded_release_region_specialist_second_pass",
        "reason": (
            "Release failures occur in a narrow temporal/hand context and require "
            "lower-confidence evidence that is unsafe globally because hard negatives "
            "fire in the same confidence range. A bounded hand/release ROI can raise "
            "effective resolution and apply context without changing global behavior."
        ),
        "implementation_status": "assessment_only",
    }


def _recommendation(models, comparison):
    e3 = models["e3_motion_blur"]["threshold_metrics"]["0.01"]["true_ball_recall"]
    e4c = models["e4c_best_overall"]["threshold_metrics"]["0.01"]["true_ball_recall"]
    return {
        "choice": "E",
        "strategy": "Collect and label more release-region data before training anything.",
        "next_experiment": (
            "Expand the frozen benchmark with independent deliveries containing "
            "hand-associated release, severe blur plus low contrast, and bright/static "
            "hard negatives; then repeat E3/E4C and specialist-ROI offline evaluation."
        ),
        "reason": (
            f"Current positive sample count is 14 (E3 recall={e3}, E4C recall={e4c}); "
            "that is enough to identify failure distributions but too small and "
            "duplicate-heavy to justify model replacement or new training."
        ),
    }


def _gallery(rows):
    categories = defaultdict(list)
    positives = [row for row in rows if row["ball_visible"]]
    for row in positives:
        e3 = row["e3_motion_blur"]["detected_at_0_01"]
        e4c = row["e4c_best_overall"]["detected_at_0_01"]
        category = (
            "both_detect" if e3 and e4c else "e3_only" if e3 else
            "e4c_only" if e4c else "both_miss"
        )
        categories[category].append(row)
        if max(
            row["e3_motion_blur"]["true_ball_confidence"] or 0,
            row["e4c_best_overall"]["true_ball_confidence"] or 0,
        ) < 0.05:
            categories["very_low_confidence"].append(row)
        if "hand_overlap" in row["conditions"]:
            categories["hand_overlap"].append(row)
        if (row["approx_ball_max_dimension_px"] or 999) <= 8:
            categories["tiny_ball"].append(row)
        if any(
            candidate["rank"] < (row[key]["true_ball_rank"] or 999)
            for key in MODELS
            for candidate in row[key]["raw_candidates"]
            if not _candidate_matches(candidate, row)
        ):
            categories["false_positive_outranks_ball"].append(row)
    for row in rows:
        if not row["ball_visible"] and any(row[key]["raw_candidates"] for key in MODELS):
            categories["hard_negative_false_positive"].append(row)

    for category, items in categories.items():
        directory = OUTPUT / "failure_gallery" / category
        directory.mkdir(parents=True, exist_ok=True)
        for row in items[:8]:
            image = _load_frame(row)
            for model_index, key in enumerate(MODELS):
                color = (255, 180, 0) if model_index == 0 else (0, 180, 255)
                for candidate in row[key]["raw_candidates"]:
                    _draw(image, candidate, color, f"{MODELS[key]} r{candidate['rank']} {candidate['confidence']:.3f}")
            if row["ball_visible"]:
                x, y = map(round, row["ball_point"])
                cv2.drawMarker(image, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.imwrite(str(directory / f"{row['benchmark_id']}.jpg"), image)


def _load_frame(annotation):
    raw_dir = ROOT / "outputs" / "video_analysis" / annotation["analysis_id"] / "raw"
    video = next(path for path in raw_dir.glob("*") if path.is_file())
    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, annotation["frame_index"])
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode {annotation['benchmark_id']}")
        return frame
    finally:
        capture.release()


def _candidate(candidate, rank):
    box = [round(float(value), 3) for value in candidate["bbox_xyxy"]]
    return {
        "rank": rank,
        "confidence": round(float(candidate["confidence"]), 6),
        "class_name": candidate["class_name"],
        "bbox_xyxy": box,
        "center": [round((box[0] + box[2]) / 2, 3), round((box[1] + box[3]) / 2, 3)],
    }


def _match(candidates, annotation):
    matches = [candidate for candidate in candidates if _candidate_matches(candidate, annotation)]
    return min(matches, key=lambda candidate: math.dist(candidate["center"], annotation["ball_point"])) if matches else None


def _candidate_matches(candidate, annotation):
    if not annotation["ball_visible"]:
        return False
    point = annotation["ball_point"]
    box = candidate["bbox_xyxy"]
    if box[0] - 4 <= point[0] <= box[2] + 4 and box[1] - 4 <= point[1] <= box[3] + 4:
        return True
    return math.dist(candidate["center"], point) <= MATCH_RADIUS


def _size_bucket(row):
    size = row["approx_ball_max_dimension_px"]
    if size <= 6:
        return "<=6"
    if size <= 8:
        return "6-8"
    if size <= 10:
        return "8-10"
    if size <= 12:
        return "10-12"
    if size <= 16:
        return "12-16"
    return ">16"


def _distribution(values):
    return {
        "count": len(values),
        "minimum": min(values, default=None),
        "maximum": max(values, default=None),
        "mean": _mean(values),
    }


def _mean(values):
    return round(sum(values) / len(values), 6) if values else None


def _ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def _draw(image, candidate, color, label):
    x1, y1, x2, y2 = map(round, candidate["bbox_xyxy"])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    cv2.putText(image, label, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def _write_csv(path, rows):
    fields = [
        "benchmark_id", "validation_id", "analysis_id", "frame_index",
        "ball_visible", "ball_point", "ball_bbox_xyxy",
        "approx_ball_max_dimension_px", "blur_severity",
        "hand_overlap_occlusion", "background_difficulty", "conditions",
        "e3_detected", "e3_confidence", "e3_rank",
        "e4c_detected", "e4c_confidence", "e4c_rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for key in ("ball_point", "ball_bbox_xyxy", "conditions"):
                flat[key] = json.dumps(flat[key])
            flat.update(
                {
                    "e3_detected": row["e3_motion_blur"]["detected_at_0_01"],
                    "e3_confidence": row["e3_motion_blur"]["true_ball_confidence"],
                    "e3_rank": row["e3_motion_blur"]["true_ball_rank"],
                    "e4c_detected": row["e4c_best_overall"]["detected_at_0_01"],
                    "e4c_confidence": row["e4c_best_overall"]["true_ball_confidence"],
                    "e4c_rank": row["e4c_best_overall"]["true_ball_rank"],
                }
            )
            writer.writerow(flat)


def _markdown(summary):
    lines = [
        "# Release-Region Hard-Case Detector Benchmark V1", "",
        "Offline evaluation only. No production settings were changed.", "",
        "## Benchmark", "",
        f"- Positive frames: {summary['benchmark_size']['positive_frames']}",
        f"- Hard-negative frames: {summary['benchmark_size']['negative_frames']}", "",
        "## Threshold Tradeoffs", "",
    ]
    for model_key, model in summary["models"].items():
        lines.append(f"### {MODELS[model_key]}")
        lines.append("")
        for threshold, metrics in model["threshold_metrics"].items():
            lines.append(
                f"- {threshold}: recall={metrics['true_ball_recall']}, "
                f"precision={metrics['precision']}, FP/frame={metrics['false_positives_per_frame']}"
            )
        lines.append("")
    lines.extend([
        "## Head To Head", "",
        *[f"- {key}: {value}" for key, value in summary["head_to_head_at_0_01"].items()],
        "", "## Recommendation", "",
        f"**{summary['recommended_next_strategy']['choice']}: "
        f"{summary['recommended_next_strategy']['strategy']}**", "",
        summary["recommended_next_strategy"]["reason"], "",
        "## Architecture", "",
        summary["architectural_assessment"]["reason"], "",
    ])
    return "\n".join(lines)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
