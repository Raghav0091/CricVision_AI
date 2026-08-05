"""Regression tests for persisted detection -> primary track selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from services.api.services.video_analysis_service import load_video_analysis
from services.api.services.video_ball_tracking_service import (
    _assign_static_likelihoods,
    _build_primary_track,
    _build_tracking_points,
    _flatten_candidates,
    _is_reliable_track,
    _load_detection_document,
    _refine_primary_track,
    _reject_outliers,
    _trim_outgoing_shot,
)


ANALYSIS_ID = "analysis_20260803_132522_b2d939"


def _saved_track(root: Path) -> list[dict]:
    path = root / ANALYSIS_ID / "tracking" / "tracking_result.json"
    if not path.is_file():
        pytest.skip(f"fixture analysis missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["primary_track"]


def _rerun_track(root: Path) -> list:
    analysis = load_video_analysis(ANALYSIS_ID)
    document = _load_detection_document(ANALYSIS_ID, analysis.frame_count)
    candidates = _flatten_candidates(document)
    _assign_static_likelihoods(candidates)
    primary, _, _ = _build_primary_track(candidates)
    assert primary is not None
    primary = _refine_primary_track(primary, candidates)
    primary = _reject_outliers(primary)
    primary = _trim_outgoing_shot(primary)
    assert _is_reliable_track(primary)
    return _build_tracking_points(primary, analysis.fps)


def test_primary_track_observed_points_match_fixture_analysis() -> None:
    root = Path("outputs/video_analysis")
    saved = _saved_track(root)
    rerun = _rerun_track(root)
    saved_observed = [
        point for point in saved if point.get("provenance") == "OBSERVED"
    ]
    rerun_observed = [point for point in rerun if point.provenance == "OBSERVED"]
    assert len(rerun_observed) == len(saved_observed)
    for left, right in zip(rerun_observed, saved_observed):
        assert left.frame_index == right["frame_index"]
        assert left.candidate_id == right.get("candidate_id")
        assert abs(left.x - right["x"]) <= 1.0
        assert abs(left.y - right["y"]) <= 1.0


def test_tracking_result_exposes_candidate_diagnostics() -> None:
    from services.api.services.video_ball_tracking_service import (
        load_video_ball_tracking_result,
    )

    root = Path("outputs/video_analysis")
    if not (root / ANALYSIS_ID / "tracking" / "tracking_result.json").is_file():
        pytest.skip("fixture analysis missing")
    result = load_video_ball_tracking_result(ANALYSIS_ID)
    assert result.candidate_diagnostics
    assert result.raw_primary_track is not None
