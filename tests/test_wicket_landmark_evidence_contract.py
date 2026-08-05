from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from services.api.schemas.wicket_landmark_evidence import (
    FrameSelectionSummary,
    NativeRoi,
    NormalizedLineEquation,
    TemporalAlignmentSummary,
    WicketEvidenceLine,
    WicketEvidencePoint,
    WicketEvidenceQuality,
    WicketLandmarkEvidenceResult,
    WicketLandmarkEvidenceRunRequest,
    WicketLandmarkDebugMedia,
    WicketLandmarkSet,
)
from services.api.services.wicket_landmark_evidence_service import (
    RESULT_FILENAME,
    persist_wicket_landmark_evidence,
)


def _point(**changes):
    payload = {
        "semantic_id": "left_stump_base",
        "x_px": 320.0,
        "y_px": 840.0,
        "confidence": 0.8,
        "uncertainty_x_px": 1.5,
        "uncertainty_y_px": 2.0,
        "supporting_frame_count": 3,
        "supporting_frame_ids": [2, 4, 6],
        "extraction_method": "temporal_consensus_v1",
        "semantic_type": "POINTLIKE",
        "status": "AVAILABLE",
        "correlation_family": "left_stump_axis_base",
    }
    payload.update(changes)
    return WicketEvidencePoint(**payload)


def _line(**changes):
    payload = {
        "semantic_id": "left_stump_axis",
        "start_x_px": 320.0,
        "start_y_px": 700.0,
        "end_x_px": 320.0,
        "end_y_px": 840.0,
        "normalized_line_equation": {"a": 1.0, "b": 0.0, "c": -320.0},
        "confidence": 0.82,
        "angular_uncertainty_deg": 1.0,
        "perpendicular_uncertainty_px": 1.8,
        "supporting_frame_count": 3,
        "supporting_frame_ids": [2, 4, 6],
        "extraction_method": "temporal_consensus_v1",
        "semantic_type": "LINE",
        "status": "AVAILABLE",
        "correlation_family": "left_stump_axis",
    }
    payload.update(changes)
    return WicketEvidenceLine(**payload)


def _quality(independent: int = 2) -> WicketEvidenceQuality:
    return WicketEvidenceQuality(
        detailed_axis_count=1,
        top_point_count=0,
        base_point_count=1,
        line_count=1,
        independent_constraint_count=independent,
        temporal_support=3,
        mean_confidence=0.81,
        median_uncertainty_px=1.9,
        severe_clipping=False,
        false_line_risk=0.1,
        evidence_grade="PARTIAL",
    )


def _set(independent: int = 2) -> WicketLandmarkSet:
    return WicketLandmarkSet(
        role="near",
        source_consensus_box={"x": 300, "y": 690, "width": 60, "height": 170},
        native_roi=NativeRoi(box={"x": 280, "y": 660, "width": 100, "height": 220}),
        supporting_frame_ids=[2, 4, 6],
        crop_quality=0.8,
        alignment_quality=0.9,
        axes=[_line()],
        points=[_point()],
        evidence_completeness=_quality(independent),
        confidence=0.8,
        uncertainty_px=1.9,
        clipping=False,
    )


def _result() -> WicketLandmarkEvidenceResult:
    return WicketLandmarkEvidenceResult(
        analysis_id="analysis_test",
        source_observation_version="wicket_observations_v1",
        created_at="2026-08-02T00:00:00+00:00",
        status="PARTIAL",
        native_image_width=720,
        native_image_height=1280,
        near_wicket=_set(),
        frame_selection=FrameSelectionSummary(
            frames_considered=5,
            frames_selected=3,
            minimum_required=3,
            selection_method="test",
        ),
        temporal_alignment=TemporalAlignmentSummary(
            method="test", frames_attempted=3, frames_aligned=3, frames_rejected=0
        ),
        detector_reused=True,
    )


def test_contract_is_strict_versioned_and_native_pixel_based() -> None:
    payload = _result().model_dump(mode="json")
    assert payload["wicket_landmark_evidence_version"] == "v1"
    assert payload["coordinate_space"] == "NATIVE_ORIENTED_PIXELS"
    assert payload["production_accepted"] is False
    assert payload["metrics_unlocked"] == []
    with pytest.raises(ValidationError):
        WicketLandmarkEvidenceResult(**{**payload, "unexpected": True})


def test_unavailable_evidence_never_contains_fake_coordinates() -> None:
    unavailable = _point(
        x_px=None,
        y_px=None,
        uncertainty_x_px=None,
        uncertainty_y_px=None,
        supporting_frame_count=0,
        supporting_frame_ids=[],
        confidence=0,
        semantic_type="UNAVAILABLE",
        status="UNAVAILABLE",
    )
    assert unavailable.x_px is None and unavailable.y_px is None
    with pytest.raises(ValidationError):
        _point(status="UNAVAILABLE")


def test_line_equation_and_support_counts_are_validated() -> None:
    assert NormalizedLineEquation(a=0.0, b=1.0, c=-10.0).b == 1.0
    with pytest.raises(ValidationError):
        _line(normalized_line_equation={"a": 2.0, "b": 0.0, "c": 0.0})
    with pytest.raises(ValidationError):
        _point(supporting_frame_count=4)


def test_correlated_derivatives_cannot_inflate_independent_count() -> None:
    with pytest.raises(ValidationError):
        _set(independent=3)


def test_redetection_requires_explicit_non_reuse_request() -> None:
    with pytest.raises(ValidationError):
        WicketLandmarkEvidenceRunRequest(force_redetect=True)
    request = WicketLandmarkEvidenceRunRequest(
        reuse_existing_observations=False, force_redetect=True
    )
    assert request.force_redetect is True


def test_debug_media_rejects_filesystem_and_cross_surface_paths() -> None:
    media = WicketLandmarkDebugMedia(
        native_roi_image_url=(
            "/static/video-analysis/analysis_test/calibration/"
            "wicket_landmarks_v1/near_frame_000010.png"
        )
    )
    assert media.native_roi_image_url.startswith("/static/video-analysis/")
    with pytest.raises(ValidationError):
        WicketLandmarkDebugMedia(native_roi_image_url="C:\\private\\near.png")


def test_report_persistence_is_atomic_and_keeps_locks(tmp_path) -> None:
    destination = persist_wicket_landmark_evidence(
        _result(), reports_directory=tmp_path
    )
    assert destination == tmp_path / RESULT_FILENAME
    assert not destination.with_suffix(".tmp").exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["production_accepted"] is False
    assert payload["metrics_unlocked"] == []
