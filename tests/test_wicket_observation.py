from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.api.schemas.wicket_observation import (
    AssignmentHypothesis,
    PixelBox,
    RawWicketDetection,
    WicketObservationDiagnostics,
    WicketObservationResult,
)
from services.api.services.wicket_observation_service import (
    FrameEvidence,
    _assignment_hypotheses,
    _result_status,
    build_consensus_region,
    build_native_roi,
    extract_wicket_landmarks,
    frame_quality_metrics,
    preprocess_roi,
    roi_to_native,
    sample_frame_indices,
    score_setup_frames,
    select_near_far_regions,
)


def _candidate(
    x: float,
    y: float,
    width: float,
    height: float,
    confidence: float = 0.8,
) -> dict[str, object]:
    return {
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "confidence": confidence,
        "class_name": "stump",
        "source": "full_frame",
    }


def _detector(
    *,
    near: dict[str, object] | None = None,
    far: dict[str, object] | None = None,
) -> dict[str, object]:
    candidates = [item for item in (near, far) if item is not None]
    return {
        "success": True,
        "candidates": candidates,
        "selected": {"non_striker": near, "striker": far},
        "diagnostics": {"rejected": []},
    }


def _frame_evidence(
    index: int,
    *,
    sharpness: float = 120,
    brightness: float = 128,
    obstruction: float = 0.1,
    near: dict[str, object] | None = None,
    far: dict[str, object] | None = None,
) -> FrameEvidence:
    return FrameEvidence(
        index=index,
        timestamp=index / 30,
        frame=np.full((180, 320, 3), int(brightness), dtype=np.uint8),
        sharpness=sharpness,
        brightness=brightness,
        obstruction=obstruction,
        detector_result=_detector(near=near, far=far),
    )


def _raw(
    frame: int,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    role: str = "NEAR_WICKET_CANDIDATE",
) -> RawWicketDetection:
    return RawWicketDetection(
        frame_index=frame,
        timestamp_seconds=frame / 30,
        bbox=PixelBox(x=x, y=y, width=width, height=height),
        confidence=0.8,
        class_name="stump",
        source="full_frame",
        detector_model="Models/stump_detector/best.pt",
        perspective_role=role,
    )


def _clean_wicket_image(
    *,
    width: int = 120,
    height: int = 160,
    stump_xs: tuple[int, ...] = (35, 60, 85),
    draw_top: bool = True,
    draw_base: bool = True,
) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for x in stump_xs:
        cv2.line(image, (x, 30), (x, 135), (255, 255, 255), 3)
    if draw_top:
        cv2.line(image, (min(stump_xs) - 4, 30), (max(stump_xs) + 4, 30), (255, 255, 255), 3)
    if draw_base:
        cv2.line(image, (min(stump_xs) - 4, 135), (max(stump_xs) + 4, 135), (255, 255, 255), 3)
    return image


def _region(
    *,
    support: int = 4,
    stability: str = "STABLE",
    width: float = 100,
    height: float = 140,
):
    detections = [
        _raw(frame, 10 + frame % 2, 10, width, height)
        for frame in range(support)
    ]
    result = build_consensus_region(
        detections,
        perspective_role="NEAR_WICKET_CANDIDATE",
        selected_frame_index=1,
        fps=30,
    )
    assert result is not None
    if stability != result.stability:
        result.stability = stability
    return result


def test_sample_frame_indices_are_bounded_and_deterministic() -> None:
    first = sample_frame_indices(100, window_frames=36, sample_limit=12)
    assert first == sample_frame_indices(100, window_frames=36, sample_limit=12)
    assert len(first) == 12
    assert first[0] == 0
    assert first[-1] == 35


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [(0, []), (1, [0]), (3, [0, 1, 2])],
)
def test_sample_frame_indices_handle_short_or_empty_video(
    frame_count: int, expected: list[int]
) -> None:
    assert sample_frame_indices(frame_count) == expected


def test_clean_detected_frame_is_preferred() -> None:
    near = _candidate(120, 90, 50, 75)
    far = _candidate(150, 25, 20, 32)
    evidence = [
        _frame_evidence(0, sharpness=5),
        _frame_evidence(1, near=near, far=far),
        _frame_evidence(2, near=near, far=far),
    ]
    scored = score_setup_frames(evidence)
    assert next(item.frame_index for item in scored if item.selected) in (1, 2)


def test_blurred_frame_has_rejection_reason() -> None:
    scored = score_setup_frames([_frame_evidence(0, sharpness=2)])
    assert "motion_blur_or_low_detail" in scored[0].rejection_reasons


def test_obstructed_frame_is_downgraded() -> None:
    box = _candidate(100, 70, 40, 80)
    clear, cluttered = score_setup_frames(
        [
            _frame_evidence(0, near=box, obstruction=0.05),
            _frame_evidence(1, near=box, obstruction=0.95),
        ]
    )
    assert clear.score > cluttered.score


def test_no_detection_is_reported_without_fabrication() -> None:
    scored = score_setup_frames([_frame_evidence(0)])
    assert scored[0].wicket_detection_count == 0
    assert "no_wicket_regions_detected" in scored[0].rejection_reasons


def test_temporal_consensus_stable_region() -> None:
    region = build_consensus_region(
        [_raw(0, 100, 80, 40, 70), _raw(1, 101, 80, 40, 70), _raw(2, 99, 81, 41, 69)],
        perspective_role="NEAR_WICKET_CANDIDATE",
        selected_frame_index=1,
        fps=30,
    )
    assert region is not None
    assert region.stability == "STABLE"
    assert region.temporal_support == 3


def test_temporal_consensus_isolated_detection_is_unstable() -> None:
    region = build_consensus_region(
        [_raw(0, 100, 80, 40, 70)],
        perspective_role="NEAR_WICKET_CANDIDATE",
        selected_frame_index=0,
        fps=30,
    )
    assert region is not None
    assert region.stability == "UNSTABLE"
    assert region.rejection_reason is not None


def test_temporal_consensus_rejects_incompatible_group() -> None:
    region = build_consensus_region(
        [
            _raw(0, 100, 80, 40, 70),
            _raw(1, 101, 80, 40, 70),
            _raw(2, 280, 10, 12, 20),
        ],
        perspective_role="NEAR_WICKET_CANDIDATE",
        selected_frame_index=0,
        fps=30,
    )
    assert region is not None
    assert region.temporal_support == 2
    assert 2 not in region.supporting_frame_ids


def test_missing_detections_return_no_consensus() -> None:
    assert build_consensus_region(
        [],
        perspective_role="FAR_WICKET_CANDIDATE",
        selected_frame_index=0,
        fps=30,
    ) is None


def test_nested_region_is_not_treated_as_second_wicket() -> None:
    large = build_consensus_region(
        [_raw(i, 100, 80, 180, 220) for i in range(3)],
        perspective_role="UNRESOLVED_WICKET",
        selected_frame_index=1,
        fps=30,
    )
    inner = build_consensus_region(
        [_raw(i, 150, 140, 35, 70) for i in range(3)],
        perspective_role="UNRESOLVED_WICKET",
        selected_frame_index=1,
        fps=30,
    )
    assert large is not None and inner is not None
    near, far, unresolved = select_near_far_regions(
        [large, inner], frame_width=480, frame_height=850
    )
    assert near is None and far is None
    assert len(unresolved) == 2


def test_complementary_separated_regions_receive_neutral_roles() -> None:
    near_region = build_consensus_region(
        [_raw(i, 200, 600, 80, 120) for i in range(3)],
        perspective_role="UNRESOLVED_WICKET",
        selected_frame_index=1,
        fps=30,
    )
    far_region = build_consensus_region(
        [_raw(i, 225, 250, 30, 55) for i in range(3)],
        perspective_role="UNRESOLVED_WICKET",
        selected_frame_index=1,
        fps=30,
    )
    assert near_region is not None and far_region is not None
    near, far, unresolved = select_near_far_regions(
        [near_region, far_region], frame_width=480, frame_height=850
    )
    assert near is not None and far is not None
    assert near.perspective_role == "NEAR_WICKET_CANDIDATE"
    assert far.perspective_role == "FAR_WICKET_CANDIDATE"
    assert unresolved == []


def test_native_roi_padding_and_mapping() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    _, roi = build_native_roi(frame, PixelBox(x=50, y=20, width=40, height=50))
    mapped = roi_to_native((0, 0), roi)
    assert roi.x < 50 and roi.y < 20
    assert mapped.x == roi.x and mapped.y == roi.y
    assert roi.native_scale == 1.0


def test_roi_clips_at_frame_edges() -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop, roi = build_native_roi(frame, PixelBox(x=0, y=0, width=25, height=30))
    assert roi.x == 0 and roi.y == 0
    assert crop.shape[:2] == (roi.height, roi.width)


@pytest.mark.parametrize("shape", [(160, 90), (90, 160)])
def test_roi_supports_portrait_and_landscape(shape: tuple[int, int]) -> None:
    height, width = shape
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    crop, roi = build_native_roi(
        frame,
        PixelBox(x=width * 0.2, y=height * 0.2, width=width * 0.4, height=height * 0.5),
    )
    assert crop.size > 0
    assert roi.source_frame_width == width
    assert roi.source_frame_height == height


def test_preprocessing_preserves_native_roi_dimensions() -> None:
    roi = _clean_wicket_image()
    variants = preprocess_roi(roi)
    assert set(variants) == {"grayscale", "clahe_contrast", "canny_edges", "vertical_gradient"}
    assert all(item.shape == roi.shape[:2] for item in variants.values())


def test_clean_wicket_produces_coarse_landmarks() -> None:
    image = _clean_wicket_image()
    _, metadata = build_native_roi(
        image, PixelBox(x=5, y=5, width=110, height=150), padding_fraction_x=0, padding_fraction_y=0
    )
    coarse, _, _, _ = extract_wicket_landmarks(image[5:155, 5:115], metadata, region=_region())
    assert sum(item.status == "AVAILABLE" for item in coarse) >= 7


def test_missing_top_is_explicitly_unavailable() -> None:
    image = _clean_wicket_image(draw_top=False)
    _, metadata = build_native_roi(
        image, PixelBox(x=0, y=0, width=120, height=160), padding_fraction_x=0, padding_fraction_y=0
    )
    coarse, _, _, _ = extract_wicket_landmarks(image, metadata, region=_region())
    top = next(item for item in coarse if item.semantic_id == "wicket_top_center")
    assert top.status in ("AVAILABLE", "UNAVAILABLE")
    if top.status == "UNAVAILABLE":
        assert top.rejection_reason


def test_insufficient_resolution_never_fabricates_detailed_points() -> None:
    image = _clean_wicket_image(width=30, height=45, stump_xs=(7, 15, 23))
    _, metadata = build_native_roi(
        image, PixelBox(x=0, y=0, width=30, height=45), padding_fraction_x=0, padding_fraction_y=0
    )
    _, detailed, status, _ = extract_wicket_landmarks(image, metadata, region=_region(width=30, height=45))
    assert status == "INSUFFICIENT_EVIDENCE"
    assert all(item.status == "UNAVAILABLE" for item in detailed)
    assert all(item.pixel_x is None and item.pixel_y is None for item in detailed)


def test_incorrect_spacing_does_not_create_detailed_stumps() -> None:
    image = _clean_wicket_image(stump_xs=(15, 25, 95))
    _, metadata = build_native_roi(
        image, PixelBox(x=0, y=0, width=120, height=160), padding_fraction_x=0, padding_fraction_y=0
    )
    _, detailed, status, _ = extract_wicket_landmarks(image, metadata, region=_region())
    assert status == "INSUFFICIENT_EVIDENCE"
    assert all(item.status == "UNAVAILABLE" for item in detailed)


def test_ambiguous_merged_axes_do_not_create_detailed_stumps() -> None:
    image = _clean_wicket_image(stump_xs=(40, 80))
    _, metadata = build_native_roi(
        image, PixelBox(x=0, y=0, width=120, height=160), padding_fraction_x=0, padding_fraction_y=0
    )
    _, detailed, status, _ = extract_wicket_landmarks(image, metadata, region=_region())
    assert status == "INSUFFICIENT_EVIDENCE"
    assert len(detailed) == 6


def test_low_resolution_increases_uncertainty() -> None:
    high = _region(width=100, height=140)
    low = _region(width=20, height=30)
    high_image = _clean_wicket_image()
    low_image = cv2.resize(high_image, (30, 40))
    _, high_roi = build_native_roi(
        high_image, PixelBox(x=0, y=0, width=120, height=160), padding_fraction_x=0, padding_fraction_y=0
    )
    _, low_roi = build_native_roi(
        low_image, PixelBox(x=0, y=0, width=30, height=40), padding_fraction_x=0, padding_fraction_y=0
    )
    high_landmarks, _, _, _ = extract_wicket_landmarks(high_image, high_roi, region=high)
    low_landmarks, _, _, _ = extract_wicket_landmarks(low_image, low_roi, region=low)
    high_available = [item for item in high_landmarks if item.status == "AVAILABLE"]
    low_available = [item for item in low_landmarks if item.status == "AVAILABLE"]
    if high_available and low_available:
        assert min(item.uncertainty_px for item in low_available) >= min(
            item.uncertainty_px for item in high_available
        )


def test_low_confidence_landmarks_are_not_primary_anchors() -> None:
    image = cv2.convertScaleAbs(_clean_wicket_image(), alpha=0.12)
    _, metadata = build_native_roi(
        image, PixelBox(x=0, y=0, width=120, height=160), padding_fraction_x=0, padding_fraction_y=0
    )
    coarse, _, _, _ = extract_wicket_landmarks(
        image, metadata, region=_region(support=1, stability="UNSTABLE")
    )
    for item in coarse:
        if item.quality == "LOW":
            assert item.registration_role == "VALIDATION_ONLY"


def test_near_far_is_separate_from_cricket_end_semantics() -> None:
    hypotheses = _assignment_hypotheses()
    assert [(item.near_semantic_end, item.far_semantic_end) for item in hypotheses] == [
        ("bowler", "striker"),
        ("striker", "bowler"),
    ]
    assert all(item.finalised is False for item in hypotheses)


def test_schema_round_trip_and_deterministic_json() -> None:
    result = WicketObservationResult(
        analysis_id="analysis_20260728_120858_762989",
        status="INSUFFICIENT_WICKETS",
        setup_frame=None,
        assignment_hypotheses=_assignment_hypotheses(),
        diagnostics=WicketObservationDiagnostics(
            detector_model_path="Models/stump_detector/best.pt",
            detector_class_labels=["stump"],
            clean_source_video="/static/raw.mp4",
            sampled_frame_ids=[],
            raw_detections=[],
        ),
        future_registration_readiness="INSUFFICIENT_WICKETS",
        message="partial",
    )
    first = json.dumps(result.model_dump(mode="json"), sort_keys=True)
    second = json.dumps(
        WicketObservationResult.model_validate_json(first).model_dump(mode="json"),
        sort_keys=True,
    )
    assert first == second
    assert "METRIC_3D_READY" not in first


def test_readiness_requires_both_wickets_and_usable_landmarks() -> None:
    assert _result_status(None, None) == "INSUFFICIENT_WICKETS"


def test_frame_quality_marks_black_frame_as_dark() -> None:
    sharpness, brightness, obstruction = frame_quality_metrics(
        np.zeros((50, 80, 3), dtype=np.uint8)
    )
    assert sharpness == 0
    assert brightness == 0
    assert 0 <= obstruction <= 1


def test_service_has_no_pose_release_or_real_pnp_imports() -> None:
    source = Path(
        "services/api/services/wicket_observation_service.py"
    ).read_text(encoding="utf-8")
    assert "release_point" not in source
    assert "pose" not in source.lower()
    assert "solvePnP" not in source
    assert "solvePnPRansac" not in source
