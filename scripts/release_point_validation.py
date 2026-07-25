"""Release Point V1 validation tooling.

This is validation-only scaffolding: it reads active Video Analysis outputs,
freezes Release V1 predictions separately, creates lightweight frame-review
packages, and computes metrics from human labels.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
VIDEO_ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "video_analysis"
VALIDATION_ROOT = PROJECT_ROOT / "outputs" / "release_validation"
DOCS_VALIDATION_ROOT = PROJECT_ROOT / "docs" / "validation"
MANIFEST_PATH = VALIDATION_ROOT / "validation_manifest.json"
MANIFEST_CSV_PATH = VALIDATION_ROOT / "validation_manifest.csv"
BASELINE_PATH = VALIDATION_ROOT / "baseline_release_v1_results.json"
ANNOTATIONS_PATH = VALIDATION_ROOT / "release_annotations.json"
METRICS_PATH = VALIDATION_ROOT / "release_point_v1_metrics.json"
V1_1_RESULTS_PATH = VALIDATION_ROOT / "release_v1_1_results.json"
V1_1_METRICS_PATH = VALIDATION_ROOT / "release_v1_1_metrics.json"
V1_2_RESULTS_PATH = VALIDATION_ROOT / "release_v1_2_results.json"
V1_2_METRICS_PATH = VALIDATION_ROOT / "release_v1_2_metrics.json"
V1_3_RESULTS_PATH = VALIDATION_ROOT / "release_v1_3_results.json"
V1_3_METRICS_PATH = VALIDATION_ROOT / "release_v1_3_metrics.json"
V1_3_RECOVERY_RESULTS_PATH = VALIDATION_ROOT / "release_v1_3_observation_recovery_results.json"
V1_3_RECOVERY_METRICS_PATH = VALIDATION_ROOT / "release_v1_3_observation_recovery_metrics.json"
V1_COMPARE_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_comparison.json"
V1_COMPARE_CSV_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_comparison.csv"
V1_COMPARE_MD_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_comparison.md"
V1_2_COMPARE_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_vs_v1_2_comparison.json"
V1_2_COMPARE_CSV_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_vs_v1_2_comparison.csv"
V1_2_COMPARE_MD_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_vs_v1_2_comparison.md"
V1_3_COMPARE_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_vs_v1_2_vs_v1_3_comparison.json"
V1_3_COMPARE_CSV_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_vs_v1_2_vs_v1_3_comparison.csv"
V1_3_COMPARE_MD_PATH = VALIDATION_ROOT / "release_v1_vs_v1_1_vs_v1_2_vs_v1_3_comparison.md"
V1_3_RECOVERY_COMPARE_PATH = VALIDATION_ROOT / "release_v1_3_vs_observation_recovery_comparison.json"
V1_3_RECOVERY_COMPARE_CSV_PATH = VALIDATION_ROOT / "release_v1_3_vs_observation_recovery_comparison.csv"
V1_3_RECOVERY_COMPARE_MD_PATH = VALIDATION_ROOT / "release_v1_3_vs_observation_recovery_comparison.md"
REPORT_PATH = DOCS_VALIDATION_ROOT / "release_point_v1_validation.md"
GUIDELINE_PATH = DOCS_VALIDATION_ROOT / "release_point_v1_annotation_guideline.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Build validation manifest.")
    audit.add_argument("--max-clips", type=int, default=30)

    baseline = subparsers.add_parser("baseline", help="Freeze Release V1 baseline.")
    baseline.add_argument(
        "--run-missing",
        action="store_true",
        help="Run current Release V1 for manifest items without saved results.",
    )
    baseline.add_argument(
        "--force-rerun",
        action="store_true",
        help="Rerun Release V1 even when reports/release_point_v1.json exists.",
    )
    baseline.add_argument("--pose-provider", default="rtmpose")
    baseline.add_argument("--pose-device", default="cpu")

    annotations = subparsers.add_parser(
        "annotations", help="Create annotation templates and frame packages."
    )
    annotations.add_argument("--window", type=int, default=5)
    annotations.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing annotation template values.",
    )

    subparsers.add_parser("metrics", help="Compute metrics from annotations.")
    v1_1 = subparsers.add_parser("v1_1", help="Run V1.1 validation comparison.")
    v1_1.add_argument("--pose-provider", default="rtmpose")
    v1_1.add_argument("--pose-device", default="cpu")
    v1_2 = subparsers.add_parser("v1_2", help="Run V1.2 validation comparison.")
    v1_2.add_argument("--pose-provider", default="rtmpose")
    v1_2.add_argument("--pose-device", default="cpu")
    v1_3 = subparsers.add_parser("v1_3", help="Run V1.3 validation comparison.")
    v1_3.add_argument("--pose-provider", default="rtmpose")
    v1_3.add_argument("--pose-device", default="cpu")
    v1_3_recovery = subparsers.add_parser(
        "v1_3_recovery",
        help="Run V1.3 with Release-Region Observation Recovery V1.",
    )
    v1_3_recovery.add_argument("--pose-provider", default="rtmpose")
    v1_3_recovery.add_argument("--pose-device", default="cpu")
    subparsers.add_parser("guideline", help="Write annotation guideline.")
    subparsers.add_parser("report", help="Write validation report.")

    all_cmd = subparsers.add_parser("all", help="Audit, baseline, annotations, metrics, report.")
    all_cmd.add_argument("--max-clips", type=int, default=30)
    all_cmd.add_argument("--window", type=int, default=5)

    args = parser.parse_args()
    if args.command == "audit":
        command_audit(args.max_clips)
    elif args.command == "baseline":
        command_baseline(
            run_missing=args.run_missing,
            force_rerun=args.force_rerun,
            pose_provider=args.pose_provider,
            pose_device=args.pose_device,
        )
    elif args.command == "annotations":
        command_annotations(window=args.window, overwrite=args.overwrite)
    elif args.command == "metrics":
        command_metrics()
    elif args.command == "v1_1":
        command_v1_1(
            pose_provider=args.pose_provider,
            pose_device=args.pose_device,
        )
    elif args.command == "v1_2":
        command_v1_2(
            pose_provider=args.pose_provider,
            pose_device=args.pose_device,
        )
    elif args.command == "v1_3":
        command_v1_3(
            pose_provider=args.pose_provider,
            pose_device=args.pose_device,
        )
    elif args.command == "v1_3_recovery":
        command_v1_3_recovery(
            pose_provider=args.pose_provider,
            pose_device=args.pose_device,
        )
    elif args.command == "guideline":
        write_guideline()
    elif args.command == "report":
        command_report()
    elif args.command == "all":
        command_audit(args.max_clips)
        command_baseline(run_missing=False, force_rerun=False)
        command_annotations(window=args.window, overwrite=False)
        command_metrics()
        write_guideline()
        command_report()
    return 0


def command_audit(max_clips: int) -> dict[str, Any]:
    rows = [_analysis_record(path) for path in sorted(VIDEO_ANALYSIS_ROOT.glob("analysis_*"))]
    complete = [row for row in rows if row["usable_for_release_v1_baseline"]]
    selected = complete[:max_clips]
    manifest = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "source_root": str(VIDEO_ANALYSIS_ROOT),
        "selection_policy": (
            "All analyses with raw video, detections.json, tracking_result.json, "
            "calibration.json, and metadata; capped by --max-clips."
        ),
        "requested_initial_clip_count": "20-30",
        "available_analysis_count": len(rows),
        "usable_release_v1_input_count": len(complete),
        "selected_count": len(selected),
        "clips": [
            {
                **row,
                "validation_id": f"rv1_{index:03d}",
                "camera_notes": _camera_notes(row),
                "bowler_handedness": "unknown",
                "quality_notes": _quality_notes(row),
            }
            for index, row in enumerate(selected, start=1)
        ],
        "excluded": [row for row in rows if not row["usable_for_release_v1_baseline"]],
    }
    _write_json(MANIFEST_PATH, manifest)
    _write_manifest_csv(manifest["clips"])
    return manifest


def command_baseline(
    *,
    run_missing: bool,
    force_rerun: bool,
    pose_provider: str = "rtmpose",
    pose_device: str = "cpu",
) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    records = []
    for clip in manifest.get("clips", []):
        analysis_id = clip["analysis_id"]
        analysis_dir = VIDEO_ANALYSIS_ROOT / analysis_id
        release_path = analysis_dir / "reports" / "release_point_v1.json"
        started = perf_counter()
        run_status = "read_existing"
        error_message = None
        document = None

        if force_rerun or (run_missing and not release_path.is_file()):
            run_status = "ran_current_algorithm"
            try:
                _run_current_release_v1(
                    analysis_id,
                    pose_provider=pose_provider,
                    pose_device=pose_device,
                )
            except Exception as exc:
                run_status = "failed"
                error_message = f"{type(exc).__name__}: {exc}"

        if release_path.is_file():
            try:
                document = _read_json(release_path)
            except Exception as exc:
                run_status = "failed"
                error_message = f"Could not read release result: {type(exc).__name__}: {exc}"
        elif error_message is None:
            run_status = "missing_release_result"
            error_message = "No saved Release V1 result. Use baseline --run-missing to execute."

        elapsed = perf_counter() - started
        records.append(_baseline_record(clip, document, run_status, error_message, elapsed))

    baseline = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "algorithm_freeze": "Release Point V1 current repository state at collection time.",
        "run_missing_requested": run_missing,
        "force_rerun_requested": force_rerun,
        "clip_count": len(records),
        "ready_prediction_count": sum(1 for item in records if item["prediction_status"] == "ready"),
        "unresolved_or_failed_count": sum(1 for item in records if item["prediction_status"] != "ready"),
        "records": records,
    }
    _write_json(BASELINE_PATH, baseline)
    return baseline


def command_annotations(*, window: int, overwrite: bool) -> dict[str, Any]:
    baseline = _read_json(BASELINE_PATH)
    existing_by_id = {}
    if ANNOTATIONS_PATH.is_file() and not overwrite:
        existing = _read_json(ANNOTATIONS_PATH)
        existing_by_id = {
            item["validation_id"]: item for item in existing.get("annotations", [])
        }

    annotations = []
    for record in baseline.get("records", []):
        validation_id = record["validation_id"]
        template = _annotation_template(record)
        if validation_id in existing_by_id:
            template = _preserve_human_annotation_fields(
                template,
                existing_by_id[validation_id],
            )
        frame_package = _write_frame_package(record, window=window)
        template["frame_package"] = frame_package
        annotations.append(template)

    document = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "annotation_guideline": str(GUIDELINE_PATH),
        "primary_task": "Select true release frame, uncertain, or not_visible.",
        "annotations": annotations,
    }
    _write_json(ANNOTATIONS_PATH, document)
    return document


def command_metrics() -> dict[str, Any]:
    baseline = _read_json(BASELINE_PATH)
    annotations = _read_json(ANNOTATIONS_PATH) if ANNOTATIONS_PATH.is_file() else {
        "annotations": []
    }
    metrics = _metrics_for_records(
        records=baseline.get("records", []),
        annotations=annotations,
        output_path=METRICS_PATH,
    )
    return metrics


def command_v1_1(*, pose_provider: str, pose_device: str) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    baseline = _read_json(BASELINE_PATH)
    annotations = _read_json(ANNOTATIONS_PATH)
    records = []
    for clip in manifest.get("clips", []):
        started = perf_counter()
        document = None
        error_message = None
        run_status = "ran_v1_1"
        try:
            document = _run_release_v1_1_for_validation(
                clip["analysis_id"],
                pose_provider=pose_provider,
                pose_device=pose_device,
            )
        except Exception as exc:
            run_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter() - started
        record = _baseline_record(clip, document, run_status, error_message, elapsed)
        record["algorithm_version"] = "release_point_v1_1"
        record["release_result_path"] = (
            str(VALIDATION_ROOT / "release_v1_1_documents" / f"{clip['validation_id']}.json")
            if document
            else None
        )
        records.append(record)
        if document:
            document_dir = VALIDATION_ROOT / "release_v1_1_documents"
            document_dir.mkdir(parents=True, exist_ok=True)
            _write_json(document_dir / f"{clip['validation_id']}.json", document)

    v1_1 = {
        "schema_version": "1.1",
        "created_at": _utc_now(),
        "algorithm": "Release Point V1.1 pre-track reconstruction",
        "baseline_reference": str(BASELINE_PATH),
        "human_annotations_reference": str(ANNOTATIONS_PATH),
        "clip_count": len(records),
        "ready_prediction_count": sum(
            1 for item in records if item["prediction_status"] == "ready"
        ),
        "unresolved_or_failed_count": sum(
            1 for item in records if item["prediction_status"] != "ready"
        ),
        "records": records,
    }
    _write_json(V1_1_RESULTS_PATH, v1_1)
    v1_1_metrics = _metrics_for_records(
        records=records,
        annotations=annotations,
        output_path=V1_1_METRICS_PATH,
    )
    v1_metrics = _read_json(METRICS_PATH)
    comparison = _comparison_document(
        v1_records=baseline.get("records", []),
        v1_1_records=records,
        annotations=annotations,
        v1_metrics=v1_metrics,
        v1_1_metrics=v1_1_metrics,
    )
    _write_json(V1_COMPARE_PATH, comparison)
    _write_comparison_csv(comparison["per_delivery_comparison"])
    V1_COMPARE_MD_PATH.write_text(_comparison_markdown(comparison), encoding="utf-8")
    return v1_1


def command_v1_2(*, pose_provider: str, pose_device: str) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    baseline = _read_json(BASELINE_PATH)
    v1_1 = _read_json(V1_1_RESULTS_PATH)
    annotations = _read_json(ANNOTATIONS_PATH)
    records = []
    document_dir = VALIDATION_ROOT / "release_v1_2_documents"
    for clip in manifest.get("clips", []):
        started = perf_counter()
        document = None
        error_message = None
        run_status = "ran_v1_2"
        try:
            document = _run_release_v1_2_for_validation(
                clip["analysis_id"],
                pose_provider=pose_provider,
                pose_device=pose_device,
            )
        except Exception as exc:
            run_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter() - started
        record = _baseline_record(clip, document, run_status, error_message, elapsed)
        record["algorithm_version"] = "release_point_v1_2"
        record["release_result_path"] = (
            str(document_dir / f"{clip['validation_id']}.json") if document else None
        )
        records.append(record)
        if document:
            document_dir.mkdir(parents=True, exist_ok=True)
            _write_json(document_dir / f"{clip['validation_id']}.json", document)

    v1_2 = {
        "schema_version": "1.2",
        "created_at": _utc_now(),
        "algorithm": "Release Point V1.2 hypothesis arbitration",
        "baseline_reference": str(BASELINE_PATH),
        "v1_1_reference": str(V1_1_RESULTS_PATH),
        "human_annotations_reference": str(ANNOTATIONS_PATH),
        "clip_count": len(records),
        "ready_prediction_count": sum(
            1 for item in records if item["prediction_status"] == "ready"
        ),
        "unresolved_or_failed_count": sum(
            1 for item in records if item["prediction_status"] != "ready"
        ),
        "records": records,
    }
    _write_json(V1_2_RESULTS_PATH, v1_2)
    v1_2_metrics = _metrics_for_records(
        records=records,
        annotations=annotations,
        output_path=V1_2_METRICS_PATH,
    )
    v1_metrics = _read_json(METRICS_PATH)
    v1_1_metrics = _read_json(V1_1_METRICS_PATH)
    comparison = _comparison_document_v1_v1_1_v1_2(
        v1_records=baseline.get("records", []),
        v1_1_records=v1_1.get("records", []),
        v1_2_records=records,
        annotations=annotations,
        v1_metrics=v1_metrics,
        v1_1_metrics=v1_1_metrics,
        v1_2_metrics=v1_2_metrics,
    )
    _write_json(V1_2_COMPARE_PATH, comparison)
    _write_comparison_v1_2_csv(comparison["per_delivery_comparison"])
    V1_2_COMPARE_MD_PATH.write_text(
        _comparison_v1_2_markdown(comparison),
        encoding="utf-8",
    )
    return v1_2


def command_v1_3(*, pose_provider: str, pose_device: str) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    baseline = _read_json(BASELINE_PATH)
    v1_1 = _read_json(V1_1_RESULTS_PATH)
    v1_2 = _read_json(V1_2_RESULTS_PATH)
    annotations = _read_json(ANNOTATIONS_PATH)
    records = []
    document_dir = VALIDATION_ROOT / "release_v1_3_documents"
    for clip in manifest.get("clips", []):
        started = perf_counter()
        document = None
        error_message = None
        run_status = "ran_v1_3"
        try:
            document = _run_release_v1_3_for_validation(
                clip["analysis_id"],
                pose_provider=pose_provider,
                pose_device=pose_device,
            )
        except Exception as exc:
            run_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter() - started
        record = _baseline_record(clip, document, run_status, error_message, elapsed)
        record["algorithm_version"] = "release_point_v1_3"
        record["release_result_path"] = (
            str(document_dir / f"{clip['validation_id']}.json") if document else None
        )
        records.append(record)
        if document:
            document_dir.mkdir(parents=True, exist_ok=True)
            _write_json(document_dir / f"{clip['validation_id']}.json", document)

    v1_3 = {
        "schema_version": "1.3",
        "created_at": _utc_now(),
        "algorithm": "Release Point V1.3 late free-flight bias guard",
        "baseline_reference": str(BASELINE_PATH),
        "v1_1_reference": str(V1_1_RESULTS_PATH),
        "v1_2_reference": str(V1_2_RESULTS_PATH),
        "human_annotations_reference": str(ANNOTATIONS_PATH),
        "clip_count": len(records),
        "ready_prediction_count": sum(
            1 for item in records if item["prediction_status"] == "ready"
        ),
        "unresolved_or_failed_count": sum(
            1 for item in records if item["prediction_status"] != "ready"
        ),
        "records": records,
    }
    _write_json(V1_3_RESULTS_PATH, v1_3)
    v1_3_metrics = _metrics_for_records(
        records=records,
        annotations=annotations,
        output_path=V1_3_METRICS_PATH,
    )
    comparison = _comparison_document_v1_v1_1_v1_2_v1_3(
        v1_records=baseline.get("records", []),
        v1_1_records=v1_1.get("records", []),
        v1_2_records=v1_2.get("records", []),
        v1_3_records=records,
        annotations=annotations,
        v1_metrics=_read_json(METRICS_PATH),
        v1_1_metrics=_read_json(V1_1_METRICS_PATH),
        v1_2_metrics=_read_json(V1_2_METRICS_PATH),
        v1_3_metrics=v1_3_metrics,
    )
    _write_json(V1_3_COMPARE_PATH, comparison)
    _write_comparison_v1_3_csv(comparison["per_delivery_comparison"])
    V1_3_COMPARE_MD_PATH.write_text(
        _comparison_v1_3_markdown(comparison),
        encoding="utf-8",
    )
    return v1_3


def command_v1_3_recovery(*, pose_provider: str, pose_device: str) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    v1_3 = _read_json(V1_3_RESULTS_PATH)
    annotations = _read_json(ANNOTATIONS_PATH)
    records = []
    document_dir = VALIDATION_ROOT / "release_v1_3_observation_recovery_documents"
    for clip in manifest.get("clips", []):
        started = perf_counter()
        document = None
        error_message = None
        run_status = "ran_v1_3_observation_recovery"
        try:
            document = _run_release_v1_3_recovery_for_validation(
                clip["analysis_id"],
                pose_provider=pose_provider,
                pose_device=pose_device,
            )
        except Exception as exc:
            run_status = "failed"
            error_message = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter() - started
        record = _baseline_record(clip, document, run_status, error_message, elapsed)
        record["algorithm_version"] = "release_pipeline_v1_3_observation_recovery_v1"
        record["input_compatibility_class"] = _input_compatibility_class(
            run_status,
            error_message,
            document,
        )
        recovery = (document or {}).get("release_region_observation_recovery") or {}
        record["recovery_status"] = recovery.get("status")
        record["recovered_observation_count"] = len(
            recovery.get("recovered_observed_points") or []
        )
        record["recovery_path_score"] = recovery.get("path_score")
        record["release_result_path"] = (
            str(document_dir / f"{clip['validation_id']}.json") if document else None
        )
        records.append(record)
        if document:
            document_dir.mkdir(parents=True, exist_ok=True)
            _write_json(document_dir / f"{clip['validation_id']}.json", document)

    output = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "algorithm": "Release Pipeline V1.3 + Observation Recovery V1",
        "v1_3_reference": str(V1_3_RESULTS_PATH),
        "human_annotations_reference": str(ANNOTATIONS_PATH),
        "clip_count": len(records),
        "ready_prediction_count": sum(
            1 for item in records if item["prediction_status"] == "ready"
        ),
        "unresolved_or_failed_count": sum(
            1 for item in records if item["prediction_status"] != "ready"
        ),
        "records": records,
    }
    _write_json(V1_3_RECOVERY_RESULTS_PATH, output)
    recovery_metrics = _metrics_for_records(
        records=records,
        annotations=annotations,
        output_path=V1_3_RECOVERY_METRICS_PATH,
    )
    comparison = _comparison_document_v1_3_recovery(
        v1_3_records=v1_3.get("records", []),
        recovery_records=records,
        annotations=annotations,
        v1_3_metrics=_read_json(V1_3_METRICS_PATH),
        recovery_metrics=recovery_metrics,
    )
    _write_json(V1_3_RECOVERY_COMPARE_PATH, comparison)
    _write_recovery_comparison_csv(comparison["per_delivery_comparison"])
    V1_3_RECOVERY_COMPARE_MD_PATH.write_text(
        _recovery_comparison_markdown(comparison),
        encoding="utf-8",
    )
    return output


def _metrics_for_records(
    *,
    records: list[dict[str, Any]],
    annotations: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    annotation_by_id = {
        item["validation_id"]: item for item in annotations.get("annotations", [])
    }
    evaluated = []
    for record in records:
        annotation = annotation_by_id.get(record["validation_id"])
        if not annotation:
            continue
        human_frame = annotation.get("human_release_frame")
        status = annotation.get("annotation_status")
        if status != "labeled" or not isinstance(human_frame, int):
            continue
        predicted = record.get("predicted_release_frame")
        error = None if predicted is None else abs(int(predicted) - human_frame)
        evaluated.append({"record": record, "annotation": annotation, "error": error})

    errors = [item["error"] for item in evaluated if item["error"] is not None]
    metrics = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "dataset_clip_count": len(records),
        "human_annotation_count": len(annotation_by_id),
        "valid_labeled_count": len(evaluated),
        "not_visible_or_uncertain_count": sum(
            1
            for item in annotation_by_id.values()
            if item.get("annotation_status") in {"uncertain", "not_visible"}
        ),
        "prediction_coverage": _ratio(
            sum(1 for record in records if record.get("predicted_release_frame") is not None),
            len(records),
        ),
        "unresolved_rate": _ratio(
            sum(1 for record in records if record.get("prediction_status") != "ready"),
            len(records),
        ),
        "exact_frame_accuracy": _accuracy(errors, 0),
        "within_1_frame_accuracy": _accuracy(errors, 1),
        "within_2_frame_accuracy": _accuracy(errors, 2),
        "mean_absolute_frame_error": statistics.fmean(errors) if errors else None,
        "median_absolute_frame_error": statistics.median(errors) if errors else None,
        "catastrophic_failure_rate": _ratio(
            sum(1 for error in errors if error > 5),
            len(evaluated),
        ),
        "confidence_bins": _confidence_bins(evaluated),
        "method_breakdown": _method_breakdown(evaluated, records),
        "pose_quality_breakdown": _pose_quality_breakdown(evaluated, records),
        "failure_category_counts": _failure_category_counts(annotation_by_id.values()),
        "labelled_records": [
            {
                "validation_id": item["record"]["validation_id"],
                "analysis_id": item["record"]["analysis_id"],
                "predicted_release_frame": item["record"].get("predicted_release_frame"),
                "human_release_frame": item["annotation"].get("human_release_frame"),
                "absolute_frame_error": item["error"],
                "method": item["record"].get("evidence_mode"),
                "confidence": item["record"].get("confidence"),
                "quality_flags": item["record"].get("quality_flags", []),
            }
            for item in evaluated
        ],
        "prediction_bias_counts": _prediction_bias_counts(evaluated, records),
    }
    _write_json(output_path, metrics)
    return metrics


def command_report() -> None:
    manifest = _read_json(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None
    baseline = _read_json(BASELINE_PATH) if BASELINE_PATH.is_file() else None
    metrics = _read_json(METRICS_PATH) if METRICS_PATH.is_file() else None
    decision = _completion_decision(metrics)
    lines = [
        "# Release Point V1 Validation",
        "",
        f"Generated: {_utc_now()}",
        "",
        "## Scope",
        "",
        "This report validates the current Release Point V1 baseline only. It does not tune thresholds, retrain models, change pose providers, or implement Milestone 2.",
        "",
        "## Dataset Audit",
        "",
    ]
    if manifest is None:
        lines.append("No validation manifest has been generated yet.")
    else:
        lines.extend(
            [
                f"- Available analysis folders: {manifest['available_analysis_count']}",
                f"- Usable Release V1 inputs: {manifest['usable_release_v1_input_count']}",
                f"- Selected validation clips: {manifest['selected_count']}",
                "- Requested initial target: 20-30 usable real deliveries",
            ]
        )
        if manifest["selected_count"] < 20:
            lines.append("- Finding: existing complete inputs are below the requested validation set size.")
    lines.extend(["", "## Annotation Methodology", ""])
    lines.extend(
        [
            "Annotators inspect frame packages around the predicted release frame and label the first frame where the ball has physically separated from the hand and begins independent free flight.",
            "If separation occurs between low-FPS frames, annotators choose the most defensible frame and may record an uncertainty interval.",
            "Annotations are stored separately at `outputs/release_validation/release_annotations.json`; predictions are not overwritten.",
        ]
    )

    lines.extend(["", "## Baseline Prediction Freeze", ""])
    if baseline is None:
        lines.append("No baseline file has been generated yet.")
    else:
        failed_records = [
            record for record in baseline.get("records", [])
            if record.get("baseline_collection_status") == "failed"
        ]
        lines.extend(
            [
                f"- Baseline clip count: {baseline['clip_count']}",
                f"- Ready predictions: {baseline['ready_prediction_count']}",
                f"- Unresolved/failed/missing predictions: {baseline['unresolved_or_failed_count']}",
                "- Baseline file: `outputs/release_validation/baseline_release_v1_results.json`",
            ]
        )
        statuses = _count_by(baseline.get("records", []), "baseline_collection_status")
        lines.append(f"- Collection statuses: `{json.dumps(statuses, sort_keys=True)}`")
        if failed_records:
            lines.append("- Failed baseline cases:")
            for record in failed_records:
                lines.append(
                    f"  - `{record['validation_id']}` `{record['analysis_id']}`: "
                    f"{record.get('baseline_collection_error')}"
                )

    lines.extend(["", "## Objective Metrics", ""])
    if metrics is None:
        lines.append("No metrics file has been generated yet.")
    else:
        lines.extend(
            [
                f"- Dataset/clip count: {metrics['dataset_clip_count']}",
                f"- Human annotation count: {metrics['human_annotation_count']}",
                f"- Valid labelled count: {metrics['valid_labeled_count']}",
                f"- Exact-frame accuracy: {_fmt_metric(metrics['exact_frame_accuracy'])}",
                f"- Within +/-1 frame: {_fmt_metric(metrics['within_1_frame_accuracy'])}",
                f"- Within +/-2 frames: {_fmt_metric(metrics['within_2_frame_accuracy'])}",
                f"- MAE: {_fmt_metric(metrics['mean_absolute_frame_error'])}",
                f"- Median absolute error: {_fmt_metric(metrics['median_absolute_frame_error'])}",
                f"- Catastrophic error rate: {_fmt_metric(metrics['catastrophic_failure_rate'])}",
                f"- Unresolved rate: {_fmt_metric(metrics['unresolved_rate'])}",
                f"- Prediction coverage: {_fmt_metric(metrics['prediction_coverage'])}",
            ]
        )

    lines.extend(["", "## Confidence And Method Analysis", ""])
    if metrics is None:
        lines.append("Pending human labels.")
    else:
        lines.append("Confidence bins are descriptive only; Release V1 confidence is not treated as calibrated probability.")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(metrics.get("confidence_bins", {}), indent=2))
        lines.append("```")
        lines.append("")
        lines.append("Method breakdown:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(metrics.get("method_breakdown", {}), indent=2))
        lines.append("```")

    lines.extend(["", "## Pose-Quality Analysis", ""])
    if metrics is None:
        lines.append("Pending human labels.")
    else:
        lines.append("```json")
        lines.append(json.dumps(metrics.get("pose_quality_breakdown", {}), indent=2))
        lines.append("```")

    lines.extend(["", "## Failure Categories", ""])
    if metrics is None:
        lines.append("Pending human labels.")
    else:
        lines.append("```json")
        lines.append(json.dumps(metrics.get("failure_category_counts", {}), indent=2))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Latency",
            "",
            _latency_summary(baseline),
            "",
            "## Limitations",
            "",
            "- No release accuracy can be claimed until manually labelled real deliveries exist.",
            "- Existing complete Release V1 input set is smaller than the requested 20-30 clips.",
            "- Failure-package categories require human review of incorrect/unresolved examples.",
            "- RTMPose-m sufficiency cannot be judged until wrist/pose errors are correlated with labelled frame errors.",
            "",
            "## Recommendations Before Tuning",
            "",
            "1. Add enough complete processed deliveries to reach at least 20-30 validation clips.",
            "2. Human-label the generated frame packages before changing any thresholds.",
            "3. Repair or exclude malformed/non-ready tracking cases before interpreting model accuracy.",
            "4. Use the weak flags analysis to decide whether bowler selection, wrist confidence, or trajectory recovery is the limiting factor.",
            "",
            "## Completion Decision",
            "",
            f"MILESTONE 1 RELEASE POINT V1 = {decision}",
            "",
            "Proceed to MILESTONE 2 - CANONICAL COMPLETE DELIVERY TRACK: not recommended until Release Point V1 has labelled baseline metrics.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_guideline() -> None:
    text = """# Release Point V1 Annotation Guideline

## Target Event

Select the first frame where the ball has physically separated from the bowler's hand and begins independent free flight.

## Frame Choice Rule

- If the ball is visibly touching or still hidden in the hand, do not mark that frame as release.
- If the next frame shows clear separation and independent motion, select that separated frame.
- If separation likely occurred between two frames, select the most defensible frame and record `uncertainty_start_frame` and `uncertainty_end_frame`.
- If the hand or ball is too blurred, occluded, or outside frame, mark `annotation_status` as `uncertain` or `not_visible`.
- Do not use model confidence as ground truth. Inspect the clean frame package.

## Required Annotation Fields

- `annotation_status`: `labeled`, `uncertain`, or `not_visible`
- `human_release_frame`: integer frame number when `annotation_status` is `labeled`
- `human_annotation_confidence`: `high`, `medium`, or `low`
- `release_visibility`: `visible`, `partially_visible`, or `not_visible`
- `uncertainty_start_frame` and `uncertainty_end_frame`: optional inclusive interval
- `failure_categories`: optional list for incorrect/unresolved cases
- `notes`: short human note

## Failure Categories

- `wrong_bowler_selected`
- `pose_wrist_inaccurate`
- `ball_invisible_near_release`
- `ball_detector_late`
- `tracker_begins_too_late`
- `backward_trajectory_inaccurate`
- `wrong_bowling_arm`
- `calibration_bowling_end_issue`
- `low_fps_ambiguity`
- `other`
"""
    GUIDELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GUIDELINE_PATH.write_text(text, encoding="utf-8")


def _analysis_record(path: Path) -> dict[str, Any]:
    metadata_path = path / "reports" / "analysis_metadata.json"
    metadata = _read_json(metadata_path) if metadata_path.is_file() else {}
    stored_filename = metadata.get("stored_filename") or "original_video.mp4"
    raw_path = path / "raw" / stored_filename
    detections_path = path / "detections" / "detections.json"
    tracking_path = path / "tracking" / "tracking_result.json"
    calibration_path = path / "calibration" / "calibration.json"
    release_path = path / "reports" / "release_point_v1.json"
    missing = []
    for label, candidate in (
        ("raw_video", raw_path),
        ("detections_json", detections_path),
        ("tracking_result_json", tracking_path),
        ("calibration_json", calibration_path),
    ):
        if not candidate.is_file():
            missing.append(label)
    return {
        "analysis_id": path.name,
        "video_path": str(raw_path),
        "fps": metadata.get("fps"),
        "frame_count": metadata.get("frame_count"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "codec": metadata.get("codec"),
        "original_filename": metadata.get("original_filename"),
        "stored_filename": stored_filename,
        "created_at": metadata.get("created_at"),
        "has_raw_video": raw_path.is_file(),
        "has_detections": detections_path.is_file(),
        "has_tracking": tracking_path.is_file(),
        "has_calibration": calibration_path.is_file(),
        "has_release_result": release_path.is_file(),
        "missing_requirements": missing,
        "usable_for_release_v1_baseline": not missing,
    }


def _baseline_record(
    clip: dict[str, Any],
    document: dict[str, Any] | None,
    run_status: str,
    error_message: str | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    result = (document or {}).get("result") or {}
    evidence = result.get("evidence") or {}
    arbitration = evidence.get("arbitration") or {}
    provenance = result.get("provenance") or {}
    arm = provenance.get("bowling_arm") or {}
    frame_uncertainty = result.get("frame_uncertainty")
    return {
        "validation_id": clip["validation_id"],
        "analysis_id": clip["analysis_id"],
        "video_path": clip["video_path"],
        "fps": clip.get("fps"),
        "frame_count": clip.get("frame_count"),
        "width": clip.get("width"),
        "height": clip.get("height"),
        "baseline_collection_status": run_status,
        "baseline_collection_error": error_message,
        "baseline_collection_seconds": round(elapsed_seconds, 3),
        "prediction_status": result.get("status") or ("failed" if error_message else "missing"),
        "predicted_release_frame": result.get("release_frame"),
        "predicted_release_time_seconds": result.get("release_time_seconds"),
        "release_point_px": result.get("release_point_px"),
        "method": result.get("method"),
        "evidence_mode": result.get("evidence_mode"),
        "release_type": result.get("release_type"),
        "confidence": result.get("confidence"),
        "frame_uncertainty": frame_uncertainty,
        "pose_provider": provenance.get("pose_provider"),
        "pose_model": provenance.get("pose_model"),
        "pose_status": provenance.get("pose_status"),
        "pose_evidence_real": provenance.get("pose_evidence_real"),
        "bowler_id": provenance.get("bowler_id"),
        "bowler_selection_confidence": evidence.get("bowler_selection_confidence"),
        "bowling_arm": arm.get("bowling_arm"),
        "bowling_arm_confidence": arm.get("confidence"),
        "pose_confidence": evidence.get("pose_confidence"),
        "pose_keypoint_confidence": evidence.get("pose_keypoint_confidence"),
        "wrist_confidence": evidence.get("wrist_confidence"),
        "wrist_used": evidence.get("wrist_used"),
        "arbitration_candidate_type": arbitration.get("selected_candidate_type"),
        "arbitration_reason_codes": arbitration.get("selected_reason_codes") or [],
        "arbitration_score": arbitration.get("selected_arbitration_score"),
        "confidence_disagreement_penalty": arbitration.get(
            "confidence_disagreement_penalty"
        ),
        "detector_model": provenance.get("ball_detector_model_key"),
        "tracking_provenance": evidence.get("tracker_provenance"),
        "tracking_version": provenance.get("tracking_version"),
        "quality_flags": result.get("quality_flags") or [],
        "release_result_path": (
            str(VIDEO_ANALYSIS_ROOT / clip["analysis_id"] / "reports" / "release_point_v1.json")
            if document
            else None
        ),
    }


def _input_compatibility_class(
    run_status: str,
    error_message: str | None,
    document: dict[str, Any] | None,
) -> str:
    if error_message:
        lowered = error_message.lower()
        if (
            "malformed" in lowered
            or "incompatible" in lowered
            or "ready primary ball track" in lowered
            or "primary ball track is required" in lowered
        ):
            return "malformed_or_incompatible_historical_tracking_input"
        if "missing" in lowered or "detections" in lowered:
            return "missing_required_persisted_candidate_data"
        return "failed_unknown"
    recovery = (document or {}).get("release_region_observation_recovery") or {}
    if recovery.get("status") == "NO_CREDIBLE_RELEASE_REGION_CHAIN":
        return "valid_modern_inputs_insufficient_observation_evidence"
    if recovery.get("status") == "ready":
        return "valid_modern_inputs_recovery_ready"
    if run_status == "failed":
        return "failed_unknown"
    return "valid_modern_inputs_no_recovery_needed_or_skipped"


def _run_current_release_v1(
    analysis_id: str,
    *,
    pose_provider: str,
    pose_device: str,
) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    os.environ.setdefault("CRICVISION_RELEASE_POSE_PROVIDER", pose_provider)
    os.environ.setdefault("CRICVISION_RELEASE_POSE_DEVICE", pose_device)
    from services.api.services.video_release_point_service import (
        _process_video_release_point,
    )

    _process_video_release_point(
        analysis_id,
        job_id=f"validation_baseline_{analysis_id}",
        bowler_pose_sequence=None,
    )


def _run_release_v1_1_for_validation(
    analysis_id: str,
    *,
    pose_provider: str,
    pose_device: str,
    schema_version: str = "1.1",
    algorithm: str = "Release Point V1.1 pre-track reconstruction",
) -> dict[str, Any]:
    os.environ.setdefault("CRICVISION_RELEASE_POSE_PROVIDER", pose_provider)
    os.environ.setdefault("CRICVISION_RELEASE_POSE_DEVICE", pose_device)
    from Backends.src.release_point.release_engine import (
        ReleaseEstimator,
        candidate_score_to_dict,
    )
    from services.api.services.video_release_point_service import (
        _load_detection_document,
        _load_tracking_document,
        _provenance,
        _read_json as _service_read_json,
        _resolve_pose_context,
        load_release_analysis_input,
    )

    release_input = load_release_analysis_input(analysis_id)
    detections = _load_detection_document(release_input)
    tracking = _load_tracking_document(release_input)
    calibration = _service_read_json(Path(release_input.calibration_path), "calibration.json")
    calibration_v2 = (
        _service_read_json(Path(release_input.calibration_v2_path), "calibration_v2.json")
        if release_input.calibration_v2_path
        else None
    )
    camera_pose = (
        _service_read_json(Path(release_input.camera_pose_path), "camera_pose.json")
        if release_input.camera_pose_path
        else None
    )
    pose_context = _resolve_pose_context(
        release_input,
        calibration=calibration,
        calibration_v2=calibration_v2,
        camera_pose=camera_pose,
        tracking=tracking,
        bowler_pose_sequence=_persisted_bowler_pose_sequence(analysis_id),
    )
    started = _utc_now()
    estimate = ReleaseEstimator().estimate(
        analysis_id=analysis_id,
        fps=release_input.fps,
        detections_document=detections,
        tracking_document=tracking,
        bowler_pose_sequence=pose_context.bowler_pose_sequence,
        provenance=_provenance(
            detections,
            tracking,
            calibration,
            calibration_v2,
            camera_pose,
            pose_context,
        ),
    )
    return {
        "schema_version": schema_version,
        "analysis_id": analysis_id,
        "created_at": started,
        "completed_at": _utc_now(),
        "algorithm": algorithm,
        "result": estimate.result,
        "candidate_scores": [
            candidate_score_to_dict(score) for score in estimate.candidate_scores
        ],
        "quality_summary": estimate.quality_summary,
        "message": estimate.message,
    }


def _run_release_v1_2_for_validation(
    analysis_id: str,
    *,
    pose_provider: str,
    pose_device: str,
) -> dict[str, Any]:
    return _run_release_v1_1_for_validation(
        analysis_id,
        pose_provider=pose_provider,
        pose_device=pose_device,
        schema_version="1.2",
        algorithm="Release Point V1.2 hypothesis arbitration",
    )


def _run_release_v1_3_for_validation(
    analysis_id: str,
    *,
    pose_provider: str,
    pose_device: str,
) -> dict[str, Any]:
    return _run_release_v1_1_for_validation(
        analysis_id,
        pose_provider=pose_provider,
        pose_device=pose_device,
        schema_version="1.3",
        algorithm="Release Point V1.3 late free-flight bias guard",
    )


def _run_release_v1_3_recovery_for_validation(
    analysis_id: str,
    *,
    pose_provider: str,
    pose_device: str,
) -> dict[str, Any]:
    os.environ.setdefault("CRICVISION_RELEASE_POSE_PROVIDER", pose_provider)
    os.environ.setdefault("CRICVISION_RELEASE_POSE_DEVICE", pose_device)
    from Backends.src.release_point.features import (
        ReleasePointConfig,
        parse_bowler_pose_sequence,
        parse_detection_observations,
        parse_track_observations,
    )
    from Backends.src.release_point.release_engine import (
        ReleaseEstimator,
        candidate_score_to_dict,
    )
    from Backends.src.release_point.release_region_recovery import (
        augment_tracking_with_recovery,
        recover_release_region_observations,
    )
    from services.api.services.video_release_point_service import (
        _load_detection_document,
        _load_tracking_document,
        _provenance,
        _read_json as _service_read_json,
        _resolve_pose_context,
        load_release_analysis_input,
    )

    release_input = load_release_analysis_input(analysis_id)
    detections = _load_detection_document(release_input)
    tracking = _load_tracking_document(release_input)
    calibration = _service_read_json(Path(release_input.calibration_path), "calibration.json")
    calibration_v2 = (
        _service_read_json(Path(release_input.calibration_v2_path), "calibration_v2.json")
        if release_input.calibration_v2_path
        else None
    )
    camera_pose = (
        _service_read_json(Path(release_input.camera_pose_path), "camera_pose.json")
        if release_input.camera_pose_path
        else None
    )
    pose_context = _resolve_pose_context(
        release_input,
        calibration=calibration,
        calibration_v2=calibration_v2,
        camera_pose=camera_pose,
        tracking=tracking,
        bowler_pose_sequence=_persisted_bowler_pose_sequence(analysis_id),
    )
    config = ReleasePointConfig()
    recovery = recover_release_region_observations(
        detections_by_frame=parse_detection_observations(detections),
        primary_track=parse_track_observations(tracking),
        pose_sequence=parse_bowler_pose_sequence(pose_context.bowler_pose_sequence),
        config=config,
    )
    augmented_tracking = augment_tracking_with_recovery(tracking, recovery)
    provenance = _provenance(
        detections,
        augmented_tracking,
        calibration,
        calibration_v2,
        camera_pose,
        pose_context,
    )
    provenance["release_region_observation_recovery"] = {
        "status": recovery.status,
        "recovered_observation_count": len(recovery.recovered_observations),
        "path_score": recovery.path_score,
    }
    started = _utc_now()
    estimate = ReleaseEstimator().estimate(
        analysis_id=analysis_id,
        fps=release_input.fps,
        detections_document=detections,
        tracking_document=augmented_tracking,
        bowler_pose_sequence=pose_context.bowler_pose_sequence,
        provenance=provenance,
    )
    return {
        "schema_version": "1.0",
        "analysis_id": analysis_id,
        "created_at": started,
        "completed_at": _utc_now(),
        "algorithm": "Release Pipeline V1.3 + Observation Recovery V1",
        "result": estimate.result,
        "candidate_scores": [
            candidate_score_to_dict(score) for score in estimate.candidate_scores
        ],
        "quality_summary": estimate.quality_summary,
        "release_region_observation_recovery": recovery.diagnostics(),
        "message": estimate.message,
    }


def _persisted_bowler_pose_sequence(analysis_id: str) -> dict[str, Any] | None:
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / "rtmpose_validation.json"
    if not path.is_file():
        return None
    document = _read_json(path)
    if document.get("status") != "ready":
        return None
    bowler = dict(document.get("bowler") or {})
    if not bowler:
        return None
    bowler.setdefault("provider", document.get("pose_provider") or {})
    if document.get("bowling_arm") and not bowler.get("bowling_arm"):
        bowler["bowling_arm"] = document.get("bowling_arm")
    return bowler


def _annotation_template(record: dict[str, Any]) -> dict[str, Any]:
    predicted = record.get("predicted_release_frame")
    uncertainty = record.get("frame_uncertainty") or {}
    return {
        "validation_id": record["validation_id"],
        "analysis_id": record["analysis_id"],
        "video_reference": record["video_path"],
        "fps": record.get("fps"),
        "predicted_release_frame": predicted,
        "predicted_release_time_seconds": record.get("predicted_release_time_seconds"),
        "prediction_confidence": record.get("confidence"),
        "prediction_method": record.get("evidence_mode"),
        "prediction_quality_flags": record.get("quality_flags", []),
        "model_uncertainty_start_frame": uncertainty.get("start"),
        "model_uncertainty_end_frame": uncertainty.get("end"),
        "annotation_status": "unlabeled",
        "human_release_frame": None,
        "human_uncertainty_start": None,
        "human_uncertainty_end": None,
        "human_annotation_confidence": None,
        "release_visibility": None,
        "uncertainty_start_frame": None,
        "uncertainty_end_frame": None,
        "cannot_determine": False,
        "true_release_point_px": None,
        "failure_categories": [],
        "notes": "",
        "annotated_at": None,
    }


def _preserve_human_annotation_fields(
    refreshed: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    human_fields = (
        "annotation_status",
        "human_release_frame",
        "human_uncertainty_start",
        "human_uncertainty_end",
        "human_annotation_confidence",
        "release_visibility",
        "uncertainty_start_frame",
        "uncertainty_end_frame",
        "cannot_determine",
        "true_release_point_px",
        "failure_categories",
        "notes",
        "annotated_at",
    )
    for field in human_fields:
        if field in existing:
            refreshed[field] = existing[field]
    return refreshed


def _write_frame_package(record: dict[str, Any], *, window: int) -> dict[str, Any]:
    predicted = record.get("predicted_release_frame")
    center = predicted if isinstance(predicted, int) else _fallback_center_frame(record)
    if center is None:
        return {"status": "not_generated", "reason": "no predicted or fallback frame"}
    start = max(0, int(center) - window)
    end = min(int(record.get("frame_count") or center), int(center) + window)
    output_dir = VALIDATION_ROOT / "frame_packages" / record["validation_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(record["video_path"])
    if not video_path.is_file():
        return {"status": "not_generated", "reason": "raw video missing"}
    try:
        from Backends.src.utils.cv2_loader import cv2
    except Exception as exc:
        return {"status": "not_generated", "reason": f"cv2 unavailable: {exc}"}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"status": "not_generated", "reason": "could not open raw video"}

    files = []
    try:
        for frame_index in range(start, end + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            label = "predicted" if frame_index == predicted else "context"
            path = output_dir / f"frame_{frame_index:06d}_{label}.jpg"
            cv2.imwrite(str(path), frame)
            files.append(str(path))
    finally:
        capture.release()
    return {
        "status": "generated",
        "center_frame": center,
        "start_frame": start,
        "end_frame": end,
        "directory": str(output_dir),
        "frames": files,
    }


def _fallback_center_frame(record: dict[str, Any]) -> int | None:
    tracking_path = VIDEO_ANALYSIS_ROOT / record["analysis_id"] / "tracking" / "tracking_result.json"
    if not tracking_path.is_file():
        return None
    tracking = _read_json(tracking_path)
    primary_track = tracking.get("primary_track") or []
    if not primary_track:
        return None
    return int(primary_track[0].get("frame_index"))


def _write_manifest_csv(clips: list[dict[str, Any]]) -> None:
    VALIDATION_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "validation_id",
        "analysis_id",
        "video_path",
        "fps",
        "frame_count",
        "width",
        "height",
        "camera_notes",
        "bowler_handedness",
        "quality_notes",
    ]
    with MANIFEST_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for clip in clips:
            writer.writerow({key: clip.get(key) for key in fieldnames})


def _confidence_bins(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    bins = {
        "0.00-0.39": [],
        "0.40-0.59": [],
        "0.60-0.79": [],
        "0.80-1.00": [],
    }
    for item in evaluated:
        confidence = item["record"].get("confidence")
        if confidence is None or item["error"] is None:
            continue
        if confidence < 0.4:
            key = "0.00-0.39"
        elif confidence < 0.6:
            key = "0.40-0.59"
        elif confidence < 0.8:
            key = "0.60-0.79"
        else:
            key = "0.80-1.00"
        bins[key].append(item["error"])
    return {
        key: {
            "count": len(errors),
            "within_2_rate": _accuracy(errors, 2),
            "mae": statistics.fmean(errors) if errors else None,
        }
        for key, errors in bins.items()
    }


def _prediction_bias_counts(
    evaluated: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, int]:
    signed_errors = []
    for item in evaluated:
        predicted = item["record"].get("predicted_release_frame")
        human = item["annotation"].get("human_release_frame")
        if predicted is None or not isinstance(human, int):
            continue
        signed_errors.append(int(predicted) - human)
    return {
        "early": sum(1 for error in signed_errors if error < 0),
        "late": sum(1 for error in signed_errors if error > 0),
        "exact": sum(1 for error in signed_errors if error == 0),
        "no_prediction": sum(
            1 for record in records if record.get("predicted_release_frame") is None
        ),
    }


def _comparison_document(
    *,
    v1_records: list[dict[str, Any]],
    v1_1_records: list[dict[str, Any]],
    annotations: dict[str, Any],
    v1_metrics: dict[str, Any],
    v1_1_metrics: dict[str, Any],
) -> dict[str, Any]:
    annotation_by_id = {
        item["validation_id"]: item for item in annotations.get("annotations", [])
    }
    v1_by_id = {record["validation_id"]: record for record in v1_records}
    v1_1_by_id = {record["validation_id"]: record for record in v1_1_records}
    metric_keys = [
        "prediction_coverage",
        "unresolved_rate",
        "exact_frame_accuracy",
        "within_1_frame_accuracy",
        "within_2_frame_accuracy",
        "mean_absolute_frame_error",
        "median_absolute_frame_error",
        "catastrophic_failure_rate",
    ]
    metric_comparison = [
        {
            "metric": key,
            "v1": v1_metrics.get(key),
            "v1_1": v1_1_metrics.get(key),
            "difference": _metric_difference(v1_metrics.get(key), v1_1_metrics.get(key)),
        }
        for key in metric_keys
    ]
    metric_comparison.append(
        {
            "metric": "early_prediction_count",
            "v1": (v1_metrics.get("prediction_bias_counts") or {}).get("early"),
            "v1_1": (v1_1_metrics.get("prediction_bias_counts") or {}).get("early"),
            "difference": _metric_difference(
                (v1_metrics.get("prediction_bias_counts") or {}).get("early"),
                (v1_1_metrics.get("prediction_bias_counts") or {}).get("early"),
            ),
        }
    )
    metric_comparison.append(
        {
            "metric": "late_prediction_count",
            "v1": (v1_metrics.get("prediction_bias_counts") or {}).get("late"),
            "v1_1": (v1_1_metrics.get("prediction_bias_counts") or {}).get("late"),
            "difference": _metric_difference(
                (v1_metrics.get("prediction_bias_counts") or {}).get("late"),
                (v1_1_metrics.get("prediction_bias_counts") or {}).get("late"),
            ),
        }
    )
    per_delivery = []
    for validation_id in sorted(v1_by_id):
        annotation = annotation_by_id.get(validation_id, {})
        human = annotation.get("human_release_frame")
        v1 = v1_by_id[validation_id]
        v1_1 = v1_1_by_id.get(validation_id, {})
        v1_error = _absolute_error(v1.get("predicted_release_frame"), human)
        v1_1_error = _absolute_error(v1_1.get("predicted_release_frame"), human)
        per_delivery.append(
            {
                "validation_id": validation_id,
                "analysis_id": v1.get("analysis_id"),
                "human_frame": human,
                "v1_prediction": v1.get("predicted_release_frame"),
                "v1_error": v1_error,
                "v1_1_prediction": v1_1.get("predicted_release_frame"),
                "v1_1_error": v1_1_error,
                "v1_1_status": v1_1.get("prediction_status"),
                "v1_1_confidence": v1_1.get("confidence"),
                "change": _delivery_change(v1_error, v1_1_error),
                "v1_1_quality_flags": v1_1.get("quality_flags", []),
            }
        )
    return {
        "schema_version": "1.1",
        "created_at": _utc_now(),
        "metric_comparison": metric_comparison,
        "per_delivery_comparison": per_delivery,
        "improved_validation_ids": [
            item["validation_id"] for item in per_delivery if item["change"] == "improved"
        ],
        "regressed_validation_ids": [
            item["validation_id"] for item in per_delivery if item["change"] == "regressed"
        ],
        "unchanged_validation_ids": [
            item["validation_id"] for item in per_delivery if item["change"] == "unchanged"
        ],
        "new_predictions": [
            item["validation_id"]
            for item in per_delivery
            if item["v1_prediction"] is None and item["v1_1_prediction"] is not None
        ],
        "lost_predictions": [
            item["validation_id"]
            for item in per_delivery
            if item["v1_prediction"] is not None and item["v1_1_prediction"] is None
        ],
    }


def _comparison_document_v1_v1_1_v1_2(
    *,
    v1_records: list[dict[str, Any]],
    v1_1_records: list[dict[str, Any]],
    v1_2_records: list[dict[str, Any]],
    annotations: dict[str, Any],
    v1_metrics: dict[str, Any],
    v1_1_metrics: dict[str, Any],
    v1_2_metrics: dict[str, Any],
) -> dict[str, Any]:
    annotation_by_id = {
        item["validation_id"]: item for item in annotations.get("annotations", [])
    }
    v1_by_id = {record["validation_id"]: record for record in v1_records}
    v1_1_by_id = {record["validation_id"]: record for record in v1_1_records}
    v1_2_by_id = {record["validation_id"]: record for record in v1_2_records}
    metric_keys = [
        "prediction_coverage",
        "unresolved_rate",
        "exact_frame_accuracy",
        "within_1_frame_accuracy",
        "within_2_frame_accuracy",
        "mean_absolute_frame_error",
        "median_absolute_frame_error",
        "catastrophic_failure_rate",
    ]
    metric_comparison = [
        {
            "metric": key,
            "v1": v1_metrics.get(key),
            "v1_1": v1_1_metrics.get(key),
            "v1_2": v1_2_metrics.get(key),
            "v1_2_change_from_v1": _metric_difference(
                v1_metrics.get(key), v1_2_metrics.get(key)
            ),
            "v1_2_change_from_v1_1": _metric_difference(
                v1_1_metrics.get(key), v1_2_metrics.get(key)
            ),
        }
        for key in metric_keys
    ]
    for bias_key in ("early", "late", "exact", "no_prediction"):
        v1_bias = (v1_metrics.get("prediction_bias_counts") or {}).get(bias_key)
        v1_1_bias = (v1_1_metrics.get("prediction_bias_counts") or {}).get(bias_key)
        v1_2_bias = (v1_2_metrics.get("prediction_bias_counts") or {}).get(bias_key)
        metric_comparison.append(
            {
                "metric": f"{bias_key}_prediction_count",
                "v1": v1_bias,
                "v1_1": v1_1_bias,
                "v1_2": v1_2_bias,
                "v1_2_change_from_v1": _metric_difference(v1_bias, v1_2_bias),
                "v1_2_change_from_v1_1": _metric_difference(v1_1_bias, v1_2_bias),
            }
        )

    per_delivery = []
    for validation_id in sorted(v1_by_id):
        annotation = annotation_by_id.get(validation_id, {})
        human = annotation.get("human_release_frame")
        v1 = v1_by_id[validation_id]
        v1_1 = v1_1_by_id.get(validation_id, {})
        v1_2 = v1_2_by_id.get(validation_id, {})
        v1_error = _absolute_error(v1.get("predicted_release_frame"), human)
        v1_1_error = _absolute_error(v1_1.get("predicted_release_frame"), human)
        v1_2_error = _absolute_error(v1_2.get("predicted_release_frame"), human)
        per_delivery.append(
            {
                "validation_id": validation_id,
                "analysis_id": v1.get("analysis_id"),
                "human_frame": human,
                "v1_prediction": v1.get("predicted_release_frame"),
                "v1_error": v1_error,
                "v1_1_prediction": v1_1.get("predicted_release_frame"),
                "v1_1_error": v1_1_error,
                "v1_2_prediction": v1_2.get("predicted_release_frame"),
                "v1_2_error": v1_2_error,
                "v1_2_status": v1_2.get("prediction_status"),
                "v1_2_confidence": v1_2.get("confidence"),
                "change_from_v1": _delivery_change(v1_error, v1_2_error),
                "change_from_v1_1": _delivery_change(v1_1_error, v1_2_error),
                "v1_2_arbitration_candidate_type": v1_2.get(
                    "arbitration_candidate_type"
                ),
                "v1_2_arbitration_reason_codes": v1_2.get(
                    "arbitration_reason_codes",
                    [],
                ),
                "v1_2_quality_flags": v1_2.get("quality_flags", []),
            }
        )
    return {
        "schema_version": "1.2",
        "created_at": _utc_now(),
        "metric_comparison": metric_comparison,
        "per_delivery_comparison": per_delivery,
        "improved_from_v1_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1"] == "improved"
        ],
        "regressed_from_v1_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1"] == "regressed"
        ],
        "improved_from_v1_1_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1_1"] == "improved"
        ],
        "regressed_from_v1_1_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1_1"] == "regressed"
        ],
        "unresolved_v1_2_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["v1_2_prediction"] is None
        ],
    }


def _comparison_document_v1_v1_1_v1_2_v1_3(
    *,
    v1_records: list[dict[str, Any]],
    v1_1_records: list[dict[str, Any]],
    v1_2_records: list[dict[str, Any]],
    v1_3_records: list[dict[str, Any]],
    annotations: dict[str, Any],
    v1_metrics: dict[str, Any],
    v1_1_metrics: dict[str, Any],
    v1_2_metrics: dict[str, Any],
    v1_3_metrics: dict[str, Any],
) -> dict[str, Any]:
    annotation_by_id = {
        item["validation_id"]: item for item in annotations.get("annotations", [])
    }
    by_version = {
        "v1": {record["validation_id"]: record for record in v1_records},
        "v1_1": {record["validation_id"]: record for record in v1_1_records},
        "v1_2": {record["validation_id"]: record for record in v1_2_records},
        "v1_3": {record["validation_id"]: record for record in v1_3_records},
    }
    metric_sources = {
        "v1": v1_metrics,
        "v1_1": v1_1_metrics,
        "v1_2": v1_2_metrics,
        "v1_3": v1_3_metrics,
    }
    metric_keys = [
        "prediction_coverage",
        "unresolved_rate",
        "exact_frame_accuracy",
        "within_1_frame_accuracy",
        "within_2_frame_accuracy",
        "mean_absolute_frame_error",
        "median_absolute_frame_error",
        "catastrophic_failure_rate",
    ]
    metric_comparison = []
    for key in metric_keys:
        row = {"metric": key}
        for version, metrics in metric_sources.items():
            row[version] = metrics.get(key)
        row["v1_3_change_from_v1"] = _metric_difference(row["v1"], row["v1_3"])
        row["v1_3_change_from_v1_2"] = _metric_difference(row["v1_2"], row["v1_3"])
        metric_comparison.append(row)
    for bias_key in ("early", "late", "exact", "no_prediction"):
        row = {"metric": f"{bias_key}_prediction_count"}
        for version, metrics in metric_sources.items():
            row[version] = (metrics.get("prediction_bias_counts") or {}).get(bias_key)
        row["v1_3_change_from_v1"] = _metric_difference(row["v1"], row["v1_3"])
        row["v1_3_change_from_v1_2"] = _metric_difference(row["v1_2"], row["v1_3"])
        metric_comparison.append(row)

    per_delivery = []
    for validation_id in sorted(by_version["v1"]):
        annotation = annotation_by_id.get(validation_id, {})
        human = annotation.get("human_release_frame")
        v1 = by_version["v1"][validation_id]
        v1_1 = by_version["v1_1"].get(validation_id, {})
        v1_2 = by_version["v1_2"].get(validation_id, {})
        v1_3 = by_version["v1_3"].get(validation_id, {})
        v1_error = _absolute_error(v1.get("predicted_release_frame"), human)
        v1_1_error = _absolute_error(v1_1.get("predicted_release_frame"), human)
        v1_2_error = _absolute_error(v1_2.get("predicted_release_frame"), human)
        v1_3_error = _absolute_error(v1_3.get("predicted_release_frame"), human)
        per_delivery.append(
            {
                "validation_id": validation_id,
                "analysis_id": v1.get("analysis_id"),
                "human_frame": human,
                "v1_prediction": v1.get("predicted_release_frame"),
                "v1_error": v1_error,
                "v1_1_prediction": v1_1.get("predicted_release_frame"),
                "v1_1_error": v1_1_error,
                "v1_2_prediction": v1_2.get("predicted_release_frame"),
                "v1_2_error": v1_2_error,
                "v1_3_prediction": v1_3.get("predicted_release_frame"),
                "v1_3_error": v1_3_error,
                "v1_3_status": v1_3.get("prediction_status"),
                "v1_3_confidence": v1_3.get("confidence"),
                "change_from_v1": _delivery_change(v1_error, v1_3_error),
                "change_from_v1_2": _delivery_change(v1_2_error, v1_3_error),
                "v1_3_arbitration_candidate_type": v1_3.get(
                    "arbitration_candidate_type"
                ),
                "v1_3_arbitration_reason_codes": v1_3.get(
                    "arbitration_reason_codes",
                    [],
                ),
                "v1_3_quality_flags": v1_3.get("quality_flags", []),
            }
        )
    return {
        "schema_version": "1.3",
        "created_at": _utc_now(),
        "metric_comparison": metric_comparison,
        "per_delivery_comparison": per_delivery,
        "improved_from_v1_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1"] == "improved"
        ],
        "regressed_from_v1_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1"] == "regressed"
        ],
        "improved_from_v1_2_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1_2"] == "improved"
        ],
        "regressed_from_v1_2_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["change_from_v1_2"] == "regressed"
        ],
        "unresolved_v1_3_validation_ids": [
            item["validation_id"]
            for item in per_delivery
            if item["v1_3_prediction"] is None
        ],
    }


def _comparison_document_v1_3_recovery(
    *,
    v1_3_records: list[dict[str, Any]],
    recovery_records: list[dict[str, Any]],
    annotations: dict[str, Any],
    v1_3_metrics: dict[str, Any],
    recovery_metrics: dict[str, Any],
) -> dict[str, Any]:
    annotation_by_id = {
        item["validation_id"]: item for item in annotations.get("annotations", [])
    }
    v1_3_by_id = {record["validation_id"]: record for record in v1_3_records}
    recovery_by_id = {record["validation_id"]: record for record in recovery_records}
    metric_keys = [
        "prediction_coverage",
        "unresolved_rate",
        "exact_frame_accuracy",
        "within_1_frame_accuracy",
        "within_2_frame_accuracy",
        "mean_absolute_frame_error",
        "median_absolute_frame_error",
        "catastrophic_failure_rate",
    ]
    metric_comparison = []
    for key in metric_keys:
        metric_comparison.append(
            {
                "metric": key,
                "v1_3": v1_3_metrics.get(key),
                "v1_3_observation_recovery": recovery_metrics.get(key),
                "difference": _metric_difference(
                    v1_3_metrics.get(key),
                    recovery_metrics.get(key),
                ),
            }
        )
    for bias_key in ("early", "late", "exact", "no_prediction"):
        v1_3_value = (v1_3_metrics.get("prediction_bias_counts") or {}).get(bias_key)
        recovery_value = (recovery_metrics.get("prediction_bias_counts") or {}).get(bias_key)
        metric_comparison.append(
            {
                "metric": f"{bias_key}_prediction_count",
                "v1_3": v1_3_value,
                "v1_3_observation_recovery": recovery_value,
                "difference": _metric_difference(v1_3_value, recovery_value),
            }
        )

    per_delivery = []
    for validation_id in sorted(v1_3_by_id):
        annotation = annotation_by_id.get(validation_id, {})
        human = annotation.get("human_release_frame")
        v1_3 = v1_3_by_id[validation_id]
        recovery = recovery_by_id.get(validation_id, {})
        v1_3_error = _absolute_error(v1_3.get("predicted_release_frame"), human)
        recovery_error = _absolute_error(recovery.get("predicted_release_frame"), human)
        prediction_changed = (
            v1_3.get("predicted_release_frame")
            != recovery.get("predicted_release_frame")
        )
        per_delivery.append(
            {
                "validation_id": validation_id,
                "analysis_id": v1_3.get("analysis_id"),
                "human_frame": human,
                "v1_3_prediction": v1_3.get("predicted_release_frame"),
                "v1_3_error": v1_3_error,
                "recovery_prediction": recovery.get("predicted_release_frame"),
                "recovery_error": recovery_error,
                "prediction_changed": prediction_changed,
                "change": _delivery_change(v1_3_error, recovery_error),
                "recovery_status": recovery.get("recovery_status"),
                "recovered_observation_count": recovery.get("recovered_observation_count"),
                "recovery_path_score": recovery.get("recovery_path_score"),
                "input_compatibility_class": recovery.get("input_compatibility_class"),
                "recovery_quality_flags": recovery.get("quality_flags", []),
            }
        )

    changed = [item for item in per_delivery if item["prediction_changed"]]
    return {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "metric_comparison": metric_comparison,
        "observation_recovery_summary": {
            "clips_with_recovered_candidates": sum(
                1 for item in per_delivery if (item.get("recovered_observation_count") or 0) > 0
            ),
            "total_recovered_observations": sum(
                item.get("recovered_observation_count") or 0 for item in per_delivery
            ),
            "paths_connected_to_primary_free_flight": sum(
                1 for item in per_delivery if item.get("recovery_status") == "ready"
            ),
            "prediction_changed_count": len(changed),
            "changed_predictions_improved": sum(
                1 for item in changed if item["change"] == "improved"
            ),
            "changed_predictions_regressed": sum(
                1 for item in changed if item["change"] == "regressed"
            ),
            "changed_predictions_unchanged_error": sum(
                1 for item in changed if item["change"] == "unchanged"
            ),
            "input_compatibility_counts": _count_by(
                per_delivery,
                "input_compatibility_class",
            ),
        },
        "per_delivery_comparison": per_delivery,
        "improved_validation_ids": [
            item["validation_id"] for item in per_delivery if item["change"] == "improved"
        ],
        "regressed_validation_ids": [
            item["validation_id"] for item in per_delivery if item["change"] == "regressed"
        ],
        "unchanged_validation_ids": [
            item["validation_id"] for item in per_delivery if item["change"] == "unchanged"
        ],
    }


def _write_comparison_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "validation_id",
        "analysis_id",
        "human_frame",
        "v1_prediction",
        "v1_error",
        "v1_1_prediction",
        "v1_1_error",
        "v1_1_status",
        "v1_1_confidence",
        "change",
        "v1_1_quality_flags",
    ]
    with V1_COMPARE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key)
                    for key in fieldnames
                }
            )


def _write_comparison_v1_2_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "validation_id",
        "analysis_id",
        "human_frame",
        "v1_prediction",
        "v1_error",
        "v1_1_prediction",
        "v1_1_error",
        "v1_2_prediction",
        "v1_2_error",
        "v1_2_status",
        "v1_2_confidence",
        "change_from_v1",
        "change_from_v1_1",
        "v1_2_arbitration_candidate_type",
        "v1_2_arbitration_reason_codes",
        "v1_2_quality_flags",
    ]
    with V1_2_COMPARE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key)
                    for key in fieldnames
                }
            )


def _write_comparison_v1_3_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "validation_id",
        "analysis_id",
        "human_frame",
        "v1_prediction",
        "v1_error",
        "v1_1_prediction",
        "v1_1_error",
        "v1_2_prediction",
        "v1_2_error",
        "v1_3_prediction",
        "v1_3_error",
        "v1_3_status",
        "v1_3_confidence",
        "change_from_v1",
        "change_from_v1_2",
        "v1_3_arbitration_candidate_type",
        "v1_3_arbitration_reason_codes",
        "v1_3_quality_flags",
    ]
    with V1_3_COMPARE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key)
                    for key in fieldnames
                }
            )


def _write_recovery_comparison_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "validation_id",
        "analysis_id",
        "human_frame",
        "v1_3_prediction",
        "v1_3_error",
        "recovery_prediction",
        "recovery_error",
        "prediction_changed",
        "change",
        "recovery_status",
        "recovered_observation_count",
        "recovery_path_score",
        "input_compatibility_class",
        "recovery_quality_flags",
    ]
    with V1_3_RECOVERY_COMPARE_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key)) if isinstance(row.get(key), list) else row.get(key)
                    for key in fieldnames
                }
            )


def _comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Release Point V1 vs V1.1 Validation",
        "",
        f"Generated: {comparison['created_at']}",
        "",
        "## Metrics",
        "",
        "| Metric | V1 | V1.1 | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["metric_comparison"]:
        lines.append(
            f"| {row['metric']} | {_fmt_md(row['v1'])} | {_fmt_md(row['v1_1'])} | {_fmt_md(row['difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Per Delivery",
            "",
            "| Validation | Human | V1 Pred | V1 Err | V1.1 Pred | V1.1 Err | Change |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in comparison["per_delivery_comparison"]:
        lines.append(
            "| {validation_id} | {human_frame} | {v1_prediction} | {v1_error} | "
            "{v1_1_prediction} | {v1_1_error} | {change} |".format(
                **{key: _fmt_md(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            f"- Improved: {', '.join(comparison['improved_validation_ids']) or 'none'}",
            f"- Regressed: {', '.join(comparison['regressed_validation_ids']) or 'none'}",
            f"- New predictions: {', '.join(comparison['new_predictions']) or 'none'}",
            f"- Lost predictions: {', '.join(comparison['lost_predictions']) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_v1_2_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Release Point V1 vs V1.1 vs V1.2 Validation",
        "",
        f"Generated: {comparison['created_at']}",
        "",
        "## Metrics",
        "",
        "| Metric | V1 | V1.1 | V1.2 | V1.2 - V1 | V1.2 - V1.1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["metric_comparison"]:
        lines.append(
            "| {metric} | {v1} | {v1_1} | {v1_2} | {from_v1} | {from_v1_1} |".format(
                metric=row["metric"],
                v1=_fmt_md(row["v1"]),
                v1_1=_fmt_md(row["v1_1"]),
                v1_2=_fmt_md(row["v1_2"]),
                from_v1=_fmt_md(row["v1_2_change_from_v1"]),
                from_v1_1=_fmt_md(row["v1_2_change_from_v1_1"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per Delivery",
            "",
            "| Validation | Human | V1 Pred | V1 Err | V1.1 Pred | V1.1 Err | V1.2 Pred | V1.2 Err | From V1.1 | Type |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in comparison["per_delivery_comparison"]:
        lines.append(
            "| {validation_id} | {human_frame} | {v1_prediction} | {v1_error} | "
            "{v1_1_prediction} | {v1_1_error} | {v1_2_prediction} | {v1_2_error} | "
            "{change_from_v1_1} | {v1_2_arbitration_candidate_type} |".format(
                **{key: _fmt_md(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            f"- Improved from V1: {', '.join(comparison['improved_from_v1_validation_ids']) or 'none'}",
            f"- Regressed from V1: {', '.join(comparison['regressed_from_v1_validation_ids']) or 'none'}",
            f"- Improved from V1.1: {', '.join(comparison['improved_from_v1_1_validation_ids']) or 'none'}",
            f"- Regressed from V1.1: {', '.join(comparison['regressed_from_v1_1_validation_ids']) or 'none'}",
            f"- V1.2 unresolved: {', '.join(comparison['unresolved_v1_2_validation_ids']) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def _comparison_v1_3_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# Release Point V1 vs V1.1 vs V1.2 vs V1.3 Validation",
        "",
        f"Generated: {comparison['created_at']}",
        "",
        "## Metrics",
        "",
        "| Metric | V1 | V1.1 | V1.2 | V1.3 | V1.3 - V1 | V1.3 - V1.2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison["metric_comparison"]:
        lines.append(
            "| {metric} | {v1} | {v1_1} | {v1_2} | {v1_3} | {from_v1} | {from_v1_2} |".format(
                metric=row["metric"],
                v1=_fmt_md(row["v1"]),
                v1_1=_fmt_md(row["v1_1"]),
                v1_2=_fmt_md(row["v1_2"]),
                v1_3=_fmt_md(row["v1_3"]),
                from_v1=_fmt_md(row["v1_3_change_from_v1"]),
                from_v1_2=_fmt_md(row["v1_3_change_from_v1_2"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per Delivery",
            "",
            "| Validation | Human | V1 | V1 Err | V1.1 | V1.1 Err | V1.2 | V1.2 Err | V1.3 | V1.3 Err | From V1.2 | Type |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in comparison["per_delivery_comparison"]:
        lines.append(
            "| {validation_id} | {human_frame} | {v1_prediction} | {v1_error} | "
            "{v1_1_prediction} | {v1_1_error} | {v1_2_prediction} | {v1_2_error} | "
            "{v1_3_prediction} | {v1_3_error} | {change_from_v1_2} | "
            "{v1_3_arbitration_candidate_type} |".format(
                **{key: _fmt_md(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            f"- Improved from V1: {', '.join(comparison['improved_from_v1_validation_ids']) or 'none'}",
            f"- Regressed from V1: {', '.join(comparison['regressed_from_v1_validation_ids']) or 'none'}",
            f"- Improved from V1.2: {', '.join(comparison['improved_from_v1_2_validation_ids']) or 'none'}",
            f"- Regressed from V1.2: {', '.join(comparison['regressed_from_v1_2_validation_ids']) or 'none'}",
            f"- V1.3 unresolved: {', '.join(comparison['unresolved_v1_3_validation_ids']) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def _recovery_comparison_markdown(comparison: dict[str, Any]) -> str:
    summary = comparison["observation_recovery_summary"]
    lines = [
        "# Release Point V1.3 vs Observation Recovery V1",
        "",
        f"Generated: {comparison['created_at']}",
        "",
        "## Metrics",
        "",
        "| Metric | V1.3 | V1.3 + Recovery | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["metric_comparison"]:
        lines.append(
            f"| {row['metric']} | {_fmt_md(row['v1_3'])} | {_fmt_md(row['v1_3_observation_recovery'])} | {_fmt_md(row['difference'])} |"
        )
    lines.extend(
        [
            "",
            "## Observation Recovery",
            "",
            f"- Clips with recovered candidates: {summary['clips_with_recovered_candidates']}",
            f"- Total recovered observations: {summary['total_recovered_observations']}",
            f"- Paths connected to primary free flight: {summary['paths_connected_to_primary_free_flight']}",
            f"- Predictions changed: {summary['prediction_changed_count']}",
            f"- Changed improved/regressed/unchanged: {summary['changed_predictions_improved']}/{summary['changed_predictions_regressed']}/{summary['changed_predictions_unchanged_error']}",
            f"- Input compatibility counts: {summary['input_compatibility_counts']}",
            "",
            "## Per Delivery",
            "",
            "| Validation | Human | V1.3 | V1.3 Err | Recovery | Recovery Err | Change | Recovery Status | Recovered Obs |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in comparison["per_delivery_comparison"]:
        lines.append(
            "| {validation_id} | {human_frame} | {v1_3_prediction} | {v1_3_error} | "
            "{recovery_prediction} | {recovery_error} | {change} | {recovery_status} | "
            "{recovered_observation_count} |".format(
                **{key: _fmt_md(value) for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            f"- Improved: {', '.join(comparison['improved_validation_ids']) or 'none'}",
            f"- Regressed: {', '.join(comparison['regressed_validation_ids']) or 'none'}",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_difference(v1: Any, v1_1: Any) -> float | int | None:
    if isinstance(v1, (int, float)) and isinstance(v1_1, (int, float)):
        return round(v1_1 - v1, 6)
    return None


def _absolute_error(predicted: Any, human: Any) -> int | None:
    if predicted is None or not isinstance(human, int):
        return None
    return abs(int(predicted) - human)


def _delivery_change(v1_error: int | None, v1_1_error: int | None) -> str:
    if v1_error is None and v1_1_error is None:
        return "unchanged"
    if v1_error is None and v1_1_error is not None:
        return "new_prediction"
    if v1_error is not None and v1_1_error is None:
        return "regressed"
    if v1_1_error < v1_error:
        return "improved"
    if v1_1_error > v1_error:
        return "regressed"
    return "unchanged"


def _fmt_md(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _method_breakdown(
    evaluated: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
) -> dict[str, Any]:
    methods = sorted({record.get("evidence_mode") or "missing" for record in all_records})
    output = {}
    for method in methods:
        method_eval = [
            item for item in evaluated if (item["record"].get("evidence_mode") or "missing") == method
        ]
        errors = [item["error"] for item in method_eval if item["error"] is not None]
        output[method] = {
            "baseline_count": sum(
                1 for record in all_records if (record.get("evidence_mode") or "missing") == method
            ),
            "labelled_count": len(method_eval),
            "within_2_rate": _accuracy(errors, 2),
            "mae": statistics.fmean(errors) if errors else None,
        }
    return output


def _pose_quality_breakdown(
    evaluated: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
) -> dict[str, Any]:
    keys = [
        "bowling_end_assignment_uncertain",
        "bowler_selection_uncertain",
        "low_confidence_wrist",
        "pose_unavailable_or_unreliable",
        "trajectory_only_estimate",
    ]
    output = {}
    for key in keys:
        baseline_records = [
            record for record in all_records if key in (record.get("quality_flags") or [])
        ]
        labelled = [
            item for item in evaluated if key in (item["record"].get("quality_flags") or [])
        ]
        errors = [item["error"] for item in labelled if item["error"] is not None]
        output[key] = {
            "baseline_count": len(baseline_records),
            "labelled_count": len(labelled),
            "within_2_rate": _accuracy(errors, 2),
            "mae": statistics.fmean(errors) if errors else None,
        }
    return output


def _failure_category_counts(annotations: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for annotation in annotations:
        for category in annotation.get("failure_categories") or []:
            counts[str(category)] = counts.get(str(category), 0) + 1
    return counts


def _accuracy(errors: list[int | None], tolerance: int) -> float | None:
    valid = [error for error in errors if error is not None]
    return _ratio(sum(1 for error in valid if error <= tolerance), len(valid))


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _camera_notes(row: dict[str, Any]) -> str:
    width = row.get("width")
    height = row.get("height")
    fps = row.get("fps")
    orientation = "unknown"
    if isinstance(width, int) and isinstance(height, int):
        orientation = "portrait" if height > width else "landscape"
    return f"{orientation}, {width}x{height}, {fps} fps"


def _quality_notes(row: dict[str, Any]) -> str:
    notes = []
    if row.get("fps") and float(row["fps"]) < 30:
        notes.append("low_fps")
    if row.get("width") and row.get("height") and min(int(row["width"]), int(row["height"])) < 500:
        notes.append("small_frame_dimension")
    if row.get("has_release_result"):
        notes.append("existing_release_baseline")
    return ", ".join(notes) if notes else "unreviewed"


def _count_by(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _completion_decision(metrics: dict[str, Any] | None) -> str:
    if not metrics or metrics.get("valid_labeled_count", 0) == 0:
        return "INSUFFICIENT DATA"
    if metrics.get("valid_labeled_count", 0) < 20:
        return "INSUFFICIENT DATA"
    within_2 = metrics.get("within_2_frame_accuracy")
    catastrophic = metrics.get("catastrophic_failure_rate")
    if within_2 is not None and within_2 >= 0.8 and (catastrophic or 0) <= 0.1:
        return "VALIDATED"
    return "NEEDS IMPROVEMENT"


def _latency_summary(baseline: dict[str, Any] | None) -> str:
    if baseline is None:
        return "No baseline timings are available yet."
    records = baseline.get("records", [])
    ran = [
        record["baseline_collection_seconds"]
        for record in records
        if record.get("baseline_collection_status") == "ran_current_algorithm"
    ]
    failed = [
        record["baseline_collection_seconds"]
        for record in records
        if record.get("baseline_collection_status") == "failed"
    ]
    parts = [
        f"Total baseline collection wall time recorded across clips: {sum(record.get('baseline_collection_seconds') or 0 for record in records):.3f}s.",
    ]
    if ran:
        parts.append(
            f"Successful current-algorithm runs: {len(ran)} clips, mean {statistics.fmean(ran):.3f}s per delivery, median {statistics.median(ran):.3f}s."
        )
    if failed:
        parts.append(
            f"Failed current-algorithm attempts: {len(failed)} clips, mean {statistics.fmean(failed):.3f}s before failure."
        )
    parts.append(
        "Pose inference time per frame is not separately emitted by Release V1 yet; use these as delivery-level CPU baseline timings."
    )
    return " ".join(parts)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
