"""Orchestrate and persist native-pixel wicket landmark evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from ..schemas.wicket_landmark_evidence import (
    FrameSelectionSummary,
    NativeRoi,
    TemporalAlignmentSummary,
    WicketEvidenceLine,
    WicketEvidencePoint,
    WicketEvidenceQuality,
    WicketLandmarkDebugMedia,
    WicketLandmarkEvidenceResult,
    WicketLandmarkEvidenceRunRequest,
    WicketLandmarkSet,
)
from ..schemas.wicket_observation import WicketObservationResult
from .wicket_landmark_consensus import build_wicket_landmark_consensus
from .wicket_landmark_extractor import (
    LandmarkEvidence,
    extract_aligned_wicket_landmarks,
)
from .wicket_landmark_frame_service import (
    AlignedRoleFrames,
    WicketLandmarkFrameBundle,
    prepare_wicket_landmark_frames,
)
from .video_analysis_service import (
    VIDEO_ANALYSIS_ROOT,
    VideoAnalysisServiceError,
    load_video_analysis,
)
from .wicket_observation_service import (
    load_wicket_observation,
    run_wicket_observation,
)


RESULT_FILENAME = "wicket_landmark_evidence_v1.json"


def _write_role_debug_media(
    analysis_id: str,
    role_frames: AlignedRoleFrames,
    evidence: WicketLandmarkSet,
) -> WicketLandmarkDebugMedia:
    analysis_root = (VIDEO_ANALYSIS_ROOT / analysis_id).resolve()
    debug_directory = (analysis_root / "calibration" / "wicket_landmarks_v1").resolve()
    if analysis_root not in debug_directory.parents:
        raise VideoAnalysisServiceError("Invalid analysis-owned debug path.", status_code=400)
    debug_directory.mkdir(parents=True, exist_ok=True)
    role = role_frames.role
    consensus_name = f"{role}_temporal_consensus.png"
    overlay_name = f"{role}_accepted_overlay.png"
    consensus = role_frames.consensus_image
    overlay = cv2.cvtColor(consensus, cv2.COLOR_GRAY2BGR) if consensus.ndim == 2 else consensus.copy()
    origin_x, origin_y = role_frames.native_origin
    for line in [*evidence.axes, *evidence.lines]:
        if line.status != "AVAILABLE":
            continue
        start = (int(round((line.start_x_px or 0) - origin_x)), int(round((line.start_y_px or 0) - origin_y)))
        end = (int(round((line.end_x_px or 0) - origin_x)), int(round((line.end_y_px or 0) - origin_y)))
        cv2.line(overlay, start, end, (20, 220, 255), 1, cv2.LINE_AA)
    for point in evidence.points:
        if point.status != "AVAILABLE":
            continue
        centre = (int(round((point.x_px or 0) - origin_x)), int(round((point.y_px or 0) - origin_y)))
        cv2.drawMarker(overlay, centre, (255, 80, 40), cv2.MARKER_CROSS, 7, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(debug_directory / consensus_name), consensus):
        raise VideoAnalysisServiceError("Could not write temporal consensus debug image.", status_code=500)
    if not cv2.imwrite(str(debug_directory / overlay_name), overlay):
        raise VideoAnalysisServiceError("Could not write landmark overlay debug image.", status_code=500)
    prefix = f"/static/video-analysis/{analysis_id}/calibration/wicket_landmarks_v1"
    return WicketLandmarkDebugMedia(
        native_roi_image_url=f"{prefix}/{role}_frame_{role_frames.frame_index:06d}.png",
        temporal_consensus_image_url=f"{prefix}/{consensus_name}",
        accepted_evidence_overlay_url=f"{prefix}/{overlay_name}",
    )


def _contract_set(
    *,
    analysis_id: str,
    role_frames: AlignedRoleFrames,
    source_wicket: object,
    write_debug_media: bool,
) -> WicketLandmarkSet:
    stack = role_frames.aligned_stack
    origin = role_frames.native_origin
    extractions = [
        extract_aligned_wicket_landmarks(
            {
                "aligned_image": alignment.aligned_image,
                "role": role_frames.role,
                "frame_id": alignment.frame_index,
                "native_origin": origin,
            }
        )
        for alignment in stack.alignments
        if alignment.accepted
    ]
    per_frame = build_wicket_landmark_consensus(extractions, minimum_support=3)
    accepted_ids = tuple(sorted(stack.accepted_frame_ids))
    alignment_quality = float(
        np.mean([item.confidence for item in stack.alignments if item.accepted])
    )
    median_extraction = extract_aligned_wicket_landmarks(role_frames)
    median_items = [
        replace(
            item,
            confidence=min(item.confidence, alignment_quality),
            supporting_frame_ids=accepted_ids,
            extraction_method="aligned_temporal_median_classical_v1",
        )
        if item.status == "AVAILABLE"
        else item
        for item in [
            *median_extraction.axes,
            *median_extraction.points,
            *median_extraction.lines,
        ]
    ]
    per_frame_items = {
        item.semantic_id: item
        for item in [*per_frame.axes, *per_frame.points, *per_frame.lines]
    }
    median_by_id = {item.semantic_id: item for item in median_items}

    def merged(semantic_id: str, geometry_type: str) -> LandmarkEvidence:
        temporal = per_frame_items.get(semantic_id)
        median = median_by_id.get(semantic_id)
        if temporal is not None and temporal.status == "AVAILABLE":
            return temporal
        if median is not None and median.status == "AVAILABLE":
            return median
        if temporal is not None:
            return temporal
        if median is not None:
            return median
        return LandmarkEvidence(
            semantic_id=semantic_id,
            geometry_type=geometry_type,
            status="UNAVAILABLE",
            reason="landmark_not_emitted_by_specialist",
        )

    point_items = [
        merged(f"{side}_stump_{level}", "POINT")
        for side in ("left", "middle", "right")
        for level in ("top", "base")
    ]
    axis_items = [merged(f"{side}_stump_axis", "LINE") for side in ("left", "middle", "right")]
    line_items = [merged(semantic_id, "LINE") for semantic_id in ("bail_line", "base_line")]
    points = [WicketEvidencePoint.model_validate(item.as_contract_mapping()) for item in point_items]
    axes = [WicketEvidenceLine.model_validate(item.as_contract_mapping()) for item in axis_items]
    lines = [WicketEvidenceLine.model_validate(item.as_contract_mapping()) for item in line_items]
    available_axes = {item.semantic_id for item in axes if item.status == "AVAILABLE"}
    available_lines = {item.semantic_id for item in lines if item.status == "AVAILABLE"}
    independent = set(available_axes) | set(available_lines)
    for point in points:
        if point.status != "AVAILABLE":
            continue
        side, _, level = point.semantic_id.partition("_stump_")
        axis_id = f"{side}_stump_axis"
        line_id = "bail_line" if level == "top" else "base_line"
        if axis_id not in available_axes or line_id not in available_lines:
            independent.add(point.semantic_id)
    published = [
        item
        for item in [*points, *axes, *lines]
        if item.status == "AVAILABLE"
    ]
    supporting_frame_ids = sorted(
        {frame for item in published for frame in item.supporting_frame_ids}
    )
    uncertainties = [
        max(item.uncertainty_x_px or 0.0, item.uncertainty_y_px or 0.0)
        for item in points
        if item.status == "AVAILABLE"
    ] + [
        item.perpendicular_uncertainty_px or 0.0
        for item in [*axes, *lines]
        if item.status == "AVAILABLE"
    ]
    grade = (
        "DETAILED"
        if len(available_axes) == 3 and sum(item.status == "AVAILABLE" for item in points) >= 4
        else "PARTIAL"
        if published
        else "INSUFFICIENT"
    )
    reference = stack.reference_crop
    transform = reference.transform
    source_box = source_wicket.region.bbox
    crop_quality = float(np.mean([item.quality_score for item in role_frames.crops]))
    median_fallback_ids = sorted(
        item.semantic_id
        for item in [*point_items, *axis_items, *line_items]
        if item.status == "AVAILABLE"
        and item.extraction_method == "aligned_temporal_median_classical_v1"
    )
    result = WicketLandmarkSet(
        role=role_frames.role,
        source_consensus_box=source_box,
        native_roi=NativeRoi(
            box={
                "x": transform.x,
                "y": transform.y,
                "width": transform.width,
                "height": transform.height,
            },
            clipped=any(item.clipping_fraction > 0 for item in role_frames.crops),
        ),
        supporting_frame_ids=supporting_frame_ids,
        crop_quality=crop_quality,
        alignment_quality=alignment_quality,
        axes=axes,
        points=points,
        lines=lines,
        outer_envelope=None,
        evidence_completeness=WicketEvidenceQuality(
            detailed_axis_count=len(available_axes),
            top_point_count=sum(item.status == "AVAILABLE" and item.semantic_id.endswith("_top") for item in points),
            base_point_count=sum(item.status == "AVAILABLE" and item.semantic_id.endswith("_base") for item in points),
            line_count=len(available_axes) + len(available_lines),
            independent_constraint_count=len(independent),
            temporal_support=len(supporting_frame_ids),
            mean_confidence=float(np.mean([item.confidence for item in published])) if published else 0.0,
            median_uncertainty_px=float(np.median(uncertainties)) if uncertainties else None,
            severe_clipping=any(item.clipping_fraction > 0.22 for item in role_frames.crops),
            false_line_risk=max(
                0.0,
                1.0
                - (float(np.median([item.confidence for item in published])) if published else 0.0),
            ),
            evidence_grade=grade,
        ),
        confidence=float(np.median([item.confidence for item in published])) if published else 0.0,
        uncertainty_px=float(np.median(uncertainties)) if uncertainties else None,
        clipping=any(item.clipping_fraction > 0 for item in role_frames.crops),
        warnings=(
            ["Temporal-median evidence used for: " + ", ".join(median_fallback_ids)]
            if median_fallback_ids
            else []
        ),
    )
    if write_debug_media:
        result = result.model_copy(
            update={
                "debug_media": _write_role_debug_media(
                    analysis_id, role_frames, result
                )
            }
        )
    return result


def _build_result(
    analysis_id: str,
    observation: WicketObservationResult,
    bundle: WicketLandmarkFrameBundle,
    *,
    detector_reused: bool,
    write_debug_media: bool = False,
) -> WicketLandmarkEvidenceResult:
    setup = observation.setup_frame
    if setup is None:
        raise VideoAnalysisServiceError(
            "Wicket landmark extraction requires a native setup frame.", status_code=422
        )
    if (bundle.selection.native_width, bundle.selection.native_height) != (
        setup.image_width,
        setup.image_height,
    ):
        raise VideoAnalysisServiceError(
            "Decoded frames do not match the persisted native coordinate space.",
            status_code=422,
        )
    near = (
        _contract_set(
            analysis_id=analysis_id,
            role_frames=bundle.near,
            source_wicket=observation.near_wicket,
            write_debug_media=write_debug_media,
        )
        if bundle.near is not None and observation.near_wicket is not None
        else None
    )
    far = (
        _contract_set(
            analysis_id=analysis_id,
            role_frames=bundle.far,
            source_wicket=observation.far_wicket,
            write_debug_media=write_debug_media,
        )
        if bundle.far is not None and observation.far_wicket is not None
        else None
    )
    usable = [item for item in (near, far) if item is not None and item.evidence_completeness.evidence_grade != "INSUFFICIENT"]
    status = "READY" if len(usable) == 2 else "PARTIAL" if usable else "INSUFFICIENT_EVIDENCE"
    alignments = [
        alignment
        for role in (bundle.near, bundle.far)
        if role is not None
        for alignment in role.aligned_stack.alignments
    ]
    accepted = [item for item in alignments if item.accepted]
    return WicketLandmarkEvidenceResult(
        analysis_id=analysis_id,
        source_observation_version=observation.version,
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,
        native_image_width=bundle.selection.native_width,
        native_image_height=bundle.selection.native_height,
        rotation_applied=bundle.selection.rotation_applied_degrees,
        near_wicket=near,
        far_wicket=far,
        supporting_frames=list(bundle.supporting_frames),
        frame_selection=FrameSelectionSummary(
            frames_considered=len(bundle.selection.considered_frame_ids),
            frames_selected=len(bundle.selection.selected_frames),
            minimum_required=3,
            selection_method="persisted_roi_native_quality_rank_v1",
        ),
        temporal_alignment=TemporalAlignmentSummary(
            method="bounded_reference_translation_v1",
            frames_attempted=len(alignments),
            frames_aligned=len(accepted),
            frames_rejected=len(alignments) - len(accepted),
            median_normalized_residual=(
                float(np.median([item.residual for item in accepted]))
                if accepted
                else None
            ),
        ),
        extraction_diagnostics={
            "near_available": near is not None,
            "far_available": far is not None,
            "frames_rejected": len(bundle.selection.rejected_frame_ids),
        },
        warnings=["Optional scene-line extraction is not part of the current specialist output."],
        failure_reasons=[] if usable else ["independent_landmark_evidence_insufficient"],
        detector_reused=detector_reused,
        production_accepted=False,
        metrics_unlocked=[],
    )


def persist_wicket_landmark_evidence(
    result: WicketLandmarkEvidenceResult,
    *,
    reports_directory: Path | None = None,
) -> Path:
    if result.production_accepted is not False or result.metrics_unlocked:
        raise ValueError("Landmark evidence cannot accept calibration or unlock metrics.")
    destination = (
        reports_directory
        or VIDEO_ANALYSIS_ROOT / result.analysis_id / "reports"
    ) / RESULT_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def run_wicket_landmark_evidence(
    analysis_id: str,
    request: WicketLandmarkEvidenceRunRequest | None = None,
) -> WicketLandmarkEvidenceResult:
    """Reuse persisted detector ROIs unless redetection is explicitly requested."""
    workflow_started = perf_counter()
    stage_started = perf_counter()
    load_video_analysis(analysis_id)
    video_analysis_load_ms = (perf_counter() - stage_started) * 1000.0
    options = request or WicketLandmarkEvidenceRunRequest()
    stage_started = perf_counter()
    if options.force_redetect:
        observation = run_wicket_observation(analysis_id)
        detector_reused = False
    else:
        observation = load_wicket_observation(analysis_id)
        detector_reused = True
    observation_load_ms = (perf_counter() - stage_started) * 1000.0
    stage_started = perf_counter()
    frame_bundle = prepare_wicket_landmark_frames(
        analysis_id,
        observation,
        write_debug=options.write_debug_media,
    )
    frame_preparation_ms = (perf_counter() - stage_started) * 1000.0
    stage_started = perf_counter()
    result = _build_result(
        analysis_id,
        observation,
        frame_bundle,
        detector_reused=detector_reused,
        write_debug_media=options.write_debug_media,
    )
    extraction_consensus_ms = (perf_counter() - stage_started) * 1000.0
    result = result.model_copy(
        update={
            "extraction_diagnostics": {
                **result.extraction_diagnostics,
                "video_analysis_load_ms": round(video_analysis_load_ms, 3),
                "observation_load_ms": round(observation_load_ms, 3),
                "frame_preparation_composite_ms": round(frame_preparation_ms, 3),
                "frame_loading_scoring_ms": None,
                "roi_construction_ms": None,
                "temporal_alignment_ms": None,
                "preprocessing_ms": None,
                "landmark_extraction_consensus_ms": round(extraction_consensus_ms, 3),
                "serialization_ms": None,
                "auto_registration_ms": None,
                "timing_note": (
                    "Frame loading, scoring, ROI construction, alignment, and "
                    "preprocessing are currently measured as one composite stage."
                ),
            }
        }
    )
    stage_started = perf_counter()
    persist_wicket_landmark_evidence(result)
    serialization_ms = (perf_counter() - stage_started) * 1000.0
    auto_registration_ms: float | None = None
    if options.rerun_auto_registration:
        from .preset_auto_registration import (
            run_preset_auto_registration_with_landmark_evidence,
        )

        stage_started = perf_counter()
        run_preset_auto_registration_with_landmark_evidence(
            analysis_id,
            preset_id=options.preset_id,
            landmark_evidence=result,
            include_optional_scene_evidence=options.include_optional_scene_evidence,
        )
        auto_registration_ms = (perf_counter() - stage_started) * 1000.0
    total_ms = (perf_counter() - workflow_started) * 1000.0
    result = result.model_copy(
        update={
            "extraction_diagnostics": {
                **result.extraction_diagnostics,
                "serialization_ms": round(serialization_ms, 3),
                "auto_registration_ms": (
                    round(auto_registration_ms, 3)
                    if auto_registration_ms is not None
                    else None
                ),
                "total_ms": round(total_ms, 3),
            }
        }
    )
    persist_wicket_landmark_evidence(result)
    return result


def load_wicket_landmark_evidence(analysis_id: str) -> WicketLandmarkEvidenceResult:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.is_file():
        raise VideoAnalysisServiceError(
            "Wicket landmark evidence has not been generated for this analysis.",
            status_code=404,
        )
    try:
        return WicketLandmarkEvidenceResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise VideoAnalysisServiceError(
            "Stored wicket landmark evidence is unavailable.", status_code=500
        ) from exc


def clear_wicket_landmark_evidence(analysis_id: str) -> bool:
    load_video_analysis(analysis_id)
    path = VIDEO_ANALYSIS_ROOT / analysis_id / "reports" / RESULT_FILENAME
    if not path.exists():
        return False
    path.unlink()
    return True
