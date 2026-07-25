from scripts.release_region_detector_evidence_audit import (
    _classify_clip,
    _disappearance_reason,
    _evaluate_frame,
)


COMPATIBLE = {
    "raw_original_video_available": True,
    "modern_detection_json_available": True,
    "top_k_persistence_available": True,
    "primary_tracking_valid": True,
}


def _raw(confidence=0.2, rank=1, center=(100, 100)):
    return {
        "candidate_id": "raw",
        "confidence": confidence,
        "rank": rank,
        "center": list(center),
        "bbox_xyxy": [center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3],
    }


def _frame(raw=None, persisted=None, selected=None):
    return _evaluate_frame(
        validation_id="rv1_test",
        analysis_id="analysis_test",
        frame_index=10,
        timestamp=0.4,
        truth={"ball_visible": True, "point": [100, 100]},
        raw_candidates=[] if raw is None else [raw],
        persisted=[] if persisted is None else [persisted],
        selected=selected,
    )


def test_no_raw_detection_is_explicit():
    row = _frame()
    assert row["comparison_category"] == "NO_RAW_BALL_EVIDENCE"
    assert row["true_ball_disappearance_reason"] == "not_produced_by_yolo_above_diagnostic_floor"


def test_low_confidence_raw_detection_records_threshold_loss():
    row = _frame(raw=_raw(confidence=0.08))
    assert row["comparison_category"] == "RAW_NOT_PERSISTED"
    assert row["true_ball_disappearance_reason"] == "production_confidence_threshold"
    assert _classify_clip(COMPATIBLE, [row]) == "DETECTOR_TRUE_BALL_LOW_CONFIDENCE"


def test_persisted_true_ball_rejected_by_tracker():
    candidate = _raw()
    row = _frame(raw=candidate, persisted=candidate)
    assert row["comparison_category"] == "PERSISTED_NOT_SELECTED"
    assert _classify_clip(COMPATIBLE, [row]) == "TRUE_BALL_PERSISTED_BUT_TRACKER_REJECTED"


def test_selected_primary_true_ball_is_sufficient():
    candidate = _raw()
    row = _frame(raw=candidate, persisted=candidate, selected=candidate)
    assert row["comparison_category"] == "SELECTED_PRIMARY"
    assert _classify_clip(COMPATIBLE, [row]) == "SUFFICIENT_NEAR_RELEASE_EVIDENCE"


def test_unannotated_frames_do_not_become_detector_misses():
    row = _evaluate_frame(
        validation_id="rv1_test",
        analysis_id="analysis_test",
        frame_index=10,
        timestamp=0.4,
        truth=None,
        raw_candidates=[],
        persisted=[],
        selected=None,
    )
    assert row["ball_visible"] is None
    assert row["comparison_category"] == "UNAVAILABLE"


def test_rank_over_production_max_is_distinct():
    raw = _raw(confidence=0.3, rank=21)
    assert _disappearance_reason(raw, None, False) == "production_max_det_or_ranking"
    row = _frame(raw=raw)
    assert _classify_clip(COMPATIBLE, [row]) == "TRUE_BALL_RANKED_TOO_LOW"
