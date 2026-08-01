from __future__ import annotations

from services.api.schemas.wicket_observation import (
    PixelBox,
    RawWicketDetection,
    SetupFrameCandidate,
)
from services.api.services.setup_frame_selection_service import (
    deterministic_early_frame_indices,
    select_setup_frame,
)
from services.api.services.wicket_box_stabilization_service import (
    assess_camera_stability,
    stabilize_wicket_boxes,
)
from fastapi.testclient import TestClient
from services.api.main import app


def _candidate(index: int, score: float = 0.7, *, width: int = 1280, height: int = 720, sharpness: float = 80) -> SetupFrameCandidate:
    return SetupFrameCandidate(
        frame_index=index,
        timestamp_seconds=index / 25,
        image_width=width,
        image_height=height,
        score=score,
        sharpness=sharpness,
        brightness=120,
        wicket_detection_count=2,
        mean_detector_confidence=0.8,
        detection_stability=0.8,
        obstruction_score=0,
        selected=False,
    )


def _detections(index: int, *, width: int = 1280, height: int = 720, clipped: bool = False) -> list[RawWicketDetection]:
    near_x = 0 if clipped else width * 0.42
    values = [
        (near_x, height * 0.70, width * 0.10, height * 0.18, 0.88),
        (width * 0.47, height * 0.42, width * 0.035, height * 0.08, 0.76),
    ]
    return [
        RawWicketDetection(
            frame_index=index,
            timestamp_seconds=index / 25,
            bbox=PixelBox(x=x, y=y, width=w, height=h),
            confidence=confidence,
            class_name="stump_set",
            source="existing_detector",
            detector_model="Models/stump_detector/best.pt",
            perspective_role="UNRESOLVED_WICKET",
        )
        for x, y, w, h, confidence in values
    ]


def test_frame_zero_passes_and_remains_selected() -> None:
    candidates = [_candidate(0, 0.7), _candidate(5, 0.99)]
    decision = select_setup_frame(candidates, [*_detections(0), *_detections(5)])
    assert decision.preferred_frame_attempted
    assert decision.preferred_frame_passed
    assert decision.selected_frame_index == 0
    assert not decision.fallback_used


def test_frame_zero_fails_and_frame_five_is_selected() -> None:
    candidates = [_candidate(0), _candidate(5)]
    decision = select_setup_frame(candidates, _detections(5))
    assert not decision.preferred_frame_passed
    assert decision.selected_frame_index == 5
    assert decision.fallback_used


def test_best_early_fallback_has_deterministic_tie_break() -> None:
    candidates = [_candidate(0), _candidate(5, 0.6), _candidate(10, 0.9)]
    detections = [*_detections(5), *_detections(10)]
    first = select_setup_frame(candidates, detections)
    second = select_setup_frame(candidates, detections)
    assert first == second
    assert first.selected_frame_index == 10


def test_persisted_sampling_maps_to_fixed_early_targets() -> None:
    assert deterministic_early_frame_indices([0, 3, 6, 10, 13, 16, 19, 23]) == [0, 6, 10, 16, 19]


def test_blurred_frame_zero_uses_fallback() -> None:
    candidates = [_candidate(0, sharpness=2), _candidate(5)]
    decision = select_setup_frame(candidates, [*_detections(0), *_detections(5)])
    assert decision.selected_frame_index == 5
    assert "frame_blurred_or_low_detail" in decision.evaluations[0].reasons


def test_clipped_wicket_rejects_frame() -> None:
    decision = select_setup_frame([_candidate(0)], _detections(0, clipped=True))
    assert decision.selected_frame_index is None
    assert "near_wicket_severely_clipped" in decision.evaluations[0].reasons


def test_missing_far_wicket_is_insufficient() -> None:
    decision = select_setup_frame([_candidate(0)], _detections(0)[:1])
    assert decision.selected_frame_index is None
    assert "both_wickets_not_detected" in decision.evaluations[0].reasons


def test_portrait_and_landscape_native_dimensions_are_preserved() -> None:
    landscape = select_setup_frame([_candidate(0)], _detections(0))
    portrait = select_setup_frame(
        [_candidate(0, width=478, height=850)],
        _detections(0, width=478, height=850),
    )
    assert (landscape.evaluations[0].image_width, landscape.evaluations[0].image_height) == (1280, 720)
    assert (portrait.evaluations[0].image_width, portrait.evaluations[0].image_height) == (478, 850)


def test_stabilization_reduces_jitter_and_rejects_outlier() -> None:
    candidates = [_candidate(index) for index in (0, 5, 10, 15)]
    detections = []
    for index, shift in ((0, 0), (5, 2), (10, -2), (15, 300)):
        frame = _detections(index)
        for item in frame:
            item.bbox.x += shift
        detections.extend(frame)
    decision = select_setup_frame(candidates, detections)
    near, far = stabilize_wicket_boxes(decision.evaluations, source="persisted")
    assert near is not None and far is not None
    assert near.frame_support == 3
    assert 15 not in near.supporting_frame_indices
    assert near.source == "persisted"


def test_camera_stability_is_based_on_temporal_box_support() -> None:
    candidates = [_candidate(index) for index in (0, 5, 10)]
    detections = [item for index in (0, 5, 10) for item in _detections(index)]
    decision = select_setup_frame(candidates, detections)
    near, far = stabilize_wicket_boxes(decision.evaluations, source="persisted")
    stability = assess_camera_stability(near, far)
    assert stability.status == "FIXED_CAMERA"
    assert stability.frames_checked == [0, 5, 10]


def test_pitch_space_router_is_registered_and_rejects_unknown_analysis() -> None:
    response = TestClient(app).get("/pitch-space-analysis/analysis_20000101_000000_000000")
    assert response.status_code == 404


def test_recent_route_is_not_shadowed_by_analysis_id_route() -> None:
    response = TestClient(app).get("/pitch-space-analysis/recent", params={"limit": 1})
    assert response.status_code == 200
    assert "items" in response.json()
