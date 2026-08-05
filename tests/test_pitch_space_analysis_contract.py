from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from services.api.schemas.pitch_space_analysis import (
    PitchSpaceDeliveryAnalysisV1,
    SetupFrameDecision,
)
from services.api.schemas.wicket_observation import (
    PixelBox,
    RawWicketDetection,
    SetupFrameCandidate,
)
from services.api.services.setup_frame_selection_service import (
    deterministic_early_frame_indices,
    select_setup_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def _candidate(frame: int, *, score: float = 0.7, sharpness: float = 80.0):
    return SetupFrameCandidate(
        frame_index=frame,
        timestamp_seconds=frame / 25,
        image_width=1280,
        image_height=720,
        score=score,
        sharpness=sharpness,
        brightness=110,
        wicket_detection_count=2,
        mean_detector_confidence=0.8,
        detection_stability=0.8,
        obstruction_score=0.1,
        selected=False,
    )


def _detections(frame: int, *, confidence: float = 0.8):
    common = {
        "frame_index": frame,
        "timestamp_seconds": frame / 25,
        "confidence": confidence,
        "class_name": "stump",
        "source": "persisted",
        "detector_model": "existing-stump-detector",
        "perspective_role": "UNRESOLVED_WICKET",
    }
    return [
        RawWicketDetection(
            **common,
            bbox=PixelBox(x=540, y=520, width=90, height=150),
        ),
        RawWicketDetection(
            **common,
            bbox=PixelBox(x=585, y=245, width=35, height=70),
        ),
    ]


def test_result_contract_is_strict_versioned_and_complete() -> None:
    fields = PitchSpaceDeliveryAnalysisV1.model_fields
    assert fields["version"].default == "pitch_space_delivery_analysis_v1"
    assert PitchSpaceDeliveryAnalysisV1.model_config["extra"] == "forbid"
    assert {
        "setup_frame_decision",
        "stable_near_wicket",
        "stable_far_wicket",
        "pitch_fit",
        "camera_stability",
        "image_space_track",
        "pitch_space_track",
        "bounce",
        "line",
        "length",
        "estimated_planar_speed",
        "estimated_lateral_movement",
        "unavailable_metrics",
        "stage_timings",
    } <= fields.keys()


def test_result_contract_supports_independent_partial_failures() -> None:
    status_annotation = str(PitchSpaceDeliveryAnalysisV1.model_fields["status"].annotation)
    for status in (
        "NO_VIDEO",
        "UPLOAD_FAILED",
        "FRAME_ZERO_UNUSABLE",
        "INSUFFICIENT_WICKETS",
        "PITCH_FIT_FAILED",
        "UNSTABLE_CAMERA",
        "BALL_TRACK_UNAVAILABLE",
        "BOUNCE_UNAVAILABLE",
        "SPEED_UNAVAILABLE",
        "MOVEMENT_UNAVAILABLE",
        "PARTIAL",
        "COMPLETE",
    ):
        assert status in status_annotation


def test_development_result_cannot_accept_production_or_unlock_3d() -> None:
    fields = PitchSpaceDeliveryAnalysisV1.model_fields
    assert fields["production_accepted"].default is False
    assert fields["metrics_unlocked"].default_factory() == []
    assert fields["airborne_3d_available"].default is False

    schema = PitchSpaceDeliveryAnalysisV1.model_json_schema()
    assert schema["properties"]["production_accepted"].get("const") is False
    assert schema["properties"]["airborne_3d_available"].get("const") is False


def test_setup_decision_always_preserves_frame_zero_policy() -> None:
    with pytest.raises(ValidationError):
        SetupFrameDecision(
            preferred_frame_attempted=False,
            preferred_frame_index=5,
            preferred_frame_passed=False,
            fallback_used=False,
            quality_score=0,
        )


def test_frame_zero_wins_even_when_later_frame_scores_higher() -> None:
    candidates = [_candidate(0, score=0.55), _candidate(5, score=0.99)]
    detections = [*_detections(0, confidence=0.6), *_detections(5, confidence=0.99)]

    decision = select_setup_frame(candidates, detections)

    assert decision.preferred_frame_attempted is True
    assert decision.preferred_frame_passed is True
    assert decision.selected_frame_index == 0
    assert decision.fallback_used is False


def test_fallback_is_repeatable_and_uses_stable_earliest_tie_break() -> None:
    candidates = [
        _candidate(0, score=0.9, sharpness=2),
        _candidate(5, score=0.8),
        _candidate(10, score=0.8),
    ]
    detections = [*_detections(0), *_detections(5), *_detections(10)]

    decisions = [select_setup_frame(candidates, detections) for _ in range(10)]

    assert {item.selected_frame_index for item in decisions} == {5}
    assert all(item.fallback_used for item in decisions)
    assert all(item.model_dump() == decisions[0].model_dump() for item in decisions)


def test_early_frame_policy_is_bounded_sorted_and_contains_no_randomness() -> None:
    assert deterministic_early_frame_indices([20, 10, 5, 0, 15, 99, 5]) == [
        0,
        5,
        10,
        15,
        20,
    ]

    source_path = ROOT / "services/api/services/setup_frame_selection_service.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "random" not in imported_roots


def test_pitch_fit_reuses_virtual_pitch_and_has_no_manual_calibration_path() -> None:
    path = ROOT / "services/api/services/two_wicket_pitch_fit_service.py"
    if not path.exists():
        pytest.skip("Two-wicket fit implementation has not landed yet.")
    source = path.read_text(encoding="utf-8")
    assert "build_virtual_pitch_specification" in source
    assert "20.12" not in source
    assert "3.05" not in source
    lowered = source.lower()
    assert "manual_anchor" not in lowered
    assert "camera_height_input" not in lowered
    assert "fov_input" not in lowered
