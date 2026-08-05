from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from services.api.routes.video_analysis import router
from services.api.schemas.preset_auto_registration import STANDARD_REAR_WICKET_NET_V1
from services.api.schemas.wicket_landmark_evidence import (
    FrameSelectionSummary,
    NativeRoi,
    TemporalAlignmentSummary,
    WicketEvidenceLine,
    WicketEvidencePoint,
    WicketEvidenceQuality,
    WicketLandmarkEvidenceResult,
    WicketLandmarkEvidenceRunRequest,
    WicketLandmarkSet,
)
from services.api.schemas.wicket_observation import WicketObservationResult
from services.api.services import preset_auto_registration as solver
from services.api.services import wicket_landmark_evidence_service as evidence_service
from services.api.services.wicket_landmark_frame_service import (
    NativeFrameSelection,
    WicketLandmarkFrameBundle,
)


def _candidate(frame_index: int, *, selected: bool = False) -> dict:
    return {
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 30,
        "image_width": 720,
        "image_height": 1280,
        "score": 0.9,
        "sharpness": 100,
        "brightness": 128,
        "wicket_detection_count": 2,
        "mean_detector_confidence": 0.9,
        "detection_stability": 0.9,
        "obstruction_score": 0.1,
        "selected": selected,
    }


def _wicket(role: str, x: float, y: float, width: float, height: float) -> dict:
    return {
        "region": {
            "frame_index": 10,
            "timestamp_seconds": 1 / 3,
            "bbox": {"x": x, "y": y, "width": width, "height": height},
            "centre": {"x": x + width / 2, "y": y + height / 2},
            "width": width,
            "height": height,
            "detector_confidence": 0.9,
            "detector_model": "persisted-stump-set",
            "source": "persisted",
            "temporal_support": 3,
            "supporting_frame_ids": [10, 11, 12],
            "centre_variation_px": 1,
            "size_variation_ratio": 0.02,
            "confidence_variation": 0.01,
            "perspective_role": f"{role.upper()}_WICKET_CANDIDATE",
            "stability": "STABLE",
            "quality": "HIGH",
            "uncertainty_px": 2,
        },
        "roi": {
            "source_frame_width": 720,
            "source_frame_height": 1280,
            "x": int(x),
            "y": int(y),
            "width": int(width),
            "height": int(height),
            "padding_x": 0,
            "padding_y": 0,
        },
        "coarse_landmarks": [],
        "detailed_landmarks": [],
        "detailed_landmarks_status": "INSUFFICIENT_EVIDENCE",
        "quality_score": 0.85,
        "quality_factors": {"frame_edge_clipping": 1.0},
    }


def _observation() -> WicketObservationResult:
    setup = _candidate(10, selected=True)
    support = [_candidate(11), _candidate(12), _candidate(13)]
    return WicketObservationResult(
        analysis_id="analysis_test",
        status="READY_FOR_REGISTRATION_EXPERIMENT",
        setup_frame=setup,
        supporting_frames=support,
        frame_candidates=[setup, *support],
        near_wicket=_wicket("near", 300, 700, 100, 200),
        far_wicket=_wicket("far", 340, 300, 40, 80),
        assignment_hypotheses=[
            {
                "hypothesis_id": "A",
                "near_semantic_end": "bowler",
                "far_semantic_end": "striker",
                "confidence": 0.5,
            },
            {
                "hypothesis_id": "B",
                "near_semantic_end": "striker",
                "far_semantic_end": "bowler",
                "confidence": 0.5,
            },
        ],
        diagnostics={
            "detector_model_path": "models/stumps.pt",
            "detector_class_labels": ["stumps"],
            "clean_source_video": "/static/video-analysis/analysis_test/raw/test.mp4",
            "sampled_frame_ids": [10, 11, 12, 13],
            "raw_detections": [],
        },
        future_registration_readiness="READY_FOR_REGISTRATION_EXPERIMENT",
        message="test",
    )


def _point(semantic_id: str, x: float, y: float, uncertainty: float = 2) -> WicketEvidencePoint:
    return WicketEvidencePoint(
        semantic_id=semantic_id,
        x_px=x,
        y_px=y,
        confidence=0.85,
        uncertainty_x_px=uncertainty,
        uncertainty_y_px=uncertainty,
        supporting_frame_count=3,
        supporting_frame_ids=[10, 11, 12],
        extraction_method="classical_temporal_v1",
        semantic_type="POINTLIKE",
        status="AVAILABLE",
    )


def _line(semantic_id: str, x: float) -> WicketEvidenceLine:
    return WicketEvidenceLine(
        semantic_id=semantic_id,
        start_x_px=x,
        start_y_px=700,
        end_x_px=x,
        end_y_px=900,
        normalized_line_equation={"a": 1, "b": 0, "c": -x},
        confidence=0.85,
        angular_uncertainty_deg=1,
        perpendicular_uncertainty_px=2,
        supporting_frame_count=3,
        supporting_frame_ids=[10, 11, 12],
        extraction_method="classical_temporal_v1",
        semantic_type="LINE",
        status="AVAILABLE",
    )


def _landmark_set(role: str, *, uncertainty: float = 2) -> WicketLandmarkSet:
    y0 = 700 if role == "near" else 300
    xs = (320.0, 350.0, 380.0) if role == "near" else (350.0, 360.0, 370.0)
    points = [
        _point(f"{side}_stump_{level}", x, y0 + (180 if level == "base" else 0), uncertainty)
        for side, x in zip(("left", "middle", "right"), xs)
        for level in ("top", "base")
    ]
    axes = [_line(f"{side}_stump_axis", x) for side, x in zip(("left", "middle", "right"), xs)]
    return WicketLandmarkSet(
        role=role,
        source_consensus_box={"x": min(xs), "y": y0, "width": max(xs) - min(xs), "height": 180},
        native_roi=NativeRoi(box={"x": min(xs) - 10, "y": y0 - 10, "width": max(xs) - min(xs) + 20, "height": 200}),
        supporting_frame_ids=[10, 11, 12],
        crop_quality=0.8,
        alignment_quality=0.9,
        axes=axes,
        points=points,
        evidence_completeness=WicketEvidenceQuality(
            detailed_axis_count=3,
            top_point_count=3,
            base_point_count=3,
            line_count=3,
            independent_constraint_count=9,
            temporal_support=3,
            mean_confidence=0.85,
            median_uncertainty_px=uncertainty,
            severe_clipping=False,
            false_line_risk=0.1,
            evidence_grade="DETAILED",
        ),
        confidence=0.85,
        uncertainty_px=uncertainty,
        clipping=False,
    )


def _evidence(*, uncertainty: float = 2) -> WicketLandmarkEvidenceResult:
    return WicketLandmarkEvidenceResult(
        analysis_id="analysis_test",
        source_observation_version="wicket_observations_v1",
        created_at="2026-08-02T00:00:00+00:00",
        status="READY",
        native_image_width=720,
        native_image_height=1280,
        near_wicket=_landmark_set("near", uncertainty=uncertainty),
        far_wicket=_landmark_set("far", uncertainty=uncertainty),
        frame_selection=FrameSelectionSummary(frames_considered=4, frames_selected=3, minimum_required=3, selection_method="test"),
        temporal_alignment=TemporalAlignmentSummary(method="test", frames_attempted=6, frames_aligned=6, frames_rejected=0),
        detector_reused=True,
    )


def test_classical_endpoints_remain_pointlike_not_exact() -> None:
    correspondences = solver.build_landmark_registration_correspondences(
        _observation(),
        _evidence(),
        assignment_hypothesis="A",
        lateral_mapping="IMAGE_LEFT_IS_PITCH_LEFT",
    )
    detailed = [
        item
        for item in correspondences
        if item.mapping_type == "DETAILED_EXACT_POINT" and item.status == "USED"
    ]
    assert len(detailed) == 12
    assert all(item.exactness == "POINTLIKE" for item in detailed)
    assert not any(
        item.mapping_type == "DETAILED_EXACT_POINT" and item.exactness == "EXACT"
        for item in correspondences
    )


def test_axis_lines_enter_existing_soft_correspondence_path() -> None:
    correspondences = solver.build_landmark_registration_correspondences(
        _observation(), _evidence(), assignment_hypothesis="A", lateral_mapping="IMAGE_LEFT_IS_PITCH_LEFT"
    )
    axes = [item for item in correspondences if item.mapping_type == "STUMP_AXIS" and item.status == "SOFT_ONLY"]
    assert len(axes) == 6
    assert all(item.exactness == "SOFT" for item in axes)
    assert all(item.angular_uncertainty_deg is not None for item in axes)


def test_stump_axis_uses_orientation_and_infinite_line_residuals() -> None:
    preset = solver._normalise_preset(STANDARD_REAR_WICKET_NET_V1)
    evidence = solver._evidence(
        _observation(),
        preset,
        landmark_evidence=_evidence(),
        evidence_mode="WICKET_LANDMARKS",
    )
    parameters = preset.nominal.copy()
    rotation, translation, _, _, _, camera_matrix = solver._camera_arrays(
        parameters, preset, evidence
    )
    axis = next(item for item in evidence.correspondences if item.mapping_type == "STUMP_AXIS")
    residuals = solver._stump_axis_residuals(axis, rotation, translation, camera_matrix)
    assert len(residuals) == 3
    assert all(np.isfinite(value) for value in residuals)
    diagnostics = solver._semantic_objective_diagnostics(
        parameters, preset, evidence, solver._NORMAL_OBJECTIVE
    )
    assert any(item["mapping_type"] == "STUMP_AXIS" for item in diagnostics["lines"])


def test_uncertainty_reduces_landmark_registration_weight() -> None:
    def first_weight(value):
        items = solver.build_landmark_registration_correspondences(
            _observation(), value, assignment_hypothesis="A", lateral_mapping="IMAGE_LEFT_IS_PITCH_LEFT"
        )
        return next(
            item.registration_weight
            for item in items
            if item.mapping_type == "DETAILED_EXACT_POINT" and item.status == "USED"
        )
    assert first_weight(_evidence(uncertainty=2)) > first_weight(_evidence(uncertainty=12))


def test_legacy_coarse_mode_remains_default() -> None:
    preset = solver._normalise_preset(STANDARD_REAR_WICKET_NET_V1)
    legacy = solver._evidence(_observation(), preset)
    improved = solver._evidence(
        _observation(), preset, landmark_evidence=_evidence(), evidence_mode="WICKET_LANDMARKS"
    )
    assert legacy.evidence_mode == "LEGACY_COARSE"
    assert not any(
        item.mapping_type == "DETAILED_EXACT_POINT" and item.status == "USED"
        for item in legacy.correspondences
    )
    assert any(
        item.mapping_type == "DETAILED_EXACT_POINT"
        and item.exactness == "POINTLIKE"
        and item.status == "USED"
        for item in improved.correspondences
    )


def test_native_dimension_mismatch_is_rejected() -> None:
    wrong = _evidence().model_copy(update={"native_image_width": 1280})
    with pytest.raises(ValueError, match="native dimensions"):
        solver.adapt_wicket_landmark_evidence(_observation(), wrong)


def test_landmark_solver_wrapper_reuses_detector_and_preserves_mode(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(solver, "run_preset_auto_registration", lambda *args, **kwargs: captured.update(kwargs) or "result")
    assert solver.run_preset_auto_registration_with_landmark_evidence(
        "analysis_test", landmark_evidence=_evidence()
    ) == "result"
    assert captured["force_redetect"] is False
    assert captured["reuse_existing_observations"] is True
    assert captured["evidence_mode"] == "WICKET_LANDMARKS"


def test_orchestration_reuses_persisted_observation_by_default(monkeypatch) -> None:
    observation = _observation()
    selection = NativeFrameSelection(
        considered_frame_ids=(10, 11, 12),
        selected_frames=(),
        rejected_frame_ids=(10, 11, 12),
        native_width=720,
        native_height=1280,
        rotation_applied_degrees=0,
    )
    bundle = WicketLandmarkFrameBundle(selection=selection, supporting_frames=(), near=None, far=None)
    monkeypatch.setattr(evidence_service, "load_video_analysis", lambda analysis_id: SimpleNamespace())
    monkeypatch.setattr(evidence_service, "load_wicket_observation", lambda analysis_id: observation)
    monkeypatch.setattr(evidence_service, "run_wicket_observation", lambda analysis_id: pytest.fail("detector reran"))
    monkeypatch.setattr(evidence_service, "prepare_wicket_landmark_frames", lambda analysis_id, item, **kwargs: bundle)
    monkeypatch.setattr(evidence_service, "persist_wicket_landmark_evidence", lambda result: None)
    result = evidence_service.run_wicket_landmark_evidence("analysis_test")
    assert result.detector_reused is True
    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.production_accepted is False and result.metrics_unlocked == []


def test_routes_are_analysis_owned_and_safe() -> None:
    paths = {route.path: route.methods for route in router.routes}
    base = "/video-analysis/{analysis_id}/wicket-landmark-evidence"
    assert paths[f"{base}/run"] == {"POST"}
    assert paths[base] == {"GET"}
    assert paths[f"{base}/clear"] == {"POST"}
