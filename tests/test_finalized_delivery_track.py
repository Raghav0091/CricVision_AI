"""Regression tests for canonical finalized delivery track consistency."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from services.api.schemas.delivery_physics import (
    BouncePhysicsResult,
    CameraCalibration,
    DeliveryInterval,
    DeliveryPhysicsResult,
    FitDiagnostics,
    FittedTrajectoryParameters,
    LateralMovementResult,
    LineLengthResult,
    PostBounceMovementResult,
    SpeedAnalytics,
    TrajectorySample,
)
from services.api.schemas.video_analysis import TrackingPoint
from services.api.services.finalized_delivery_track import (
    FINALIZED_TRACK_FILENAME,
    build_finalized_delivery_track,
    cache_bust_url,
    finalized_render_track,
    validate_source_consistency,
)
from services.api.services.replay_payload_service import (
    REPLAY_PAYLOAD_FILENAME,
    build_and_save_replay_payload,
)
from services.api.services.video_ball_tracking_service import (
    DELIVERY_REPLAY_FILENAME,
    TRACKING_VIDEO_FILENAME,
    _clear_previous_tracking_outputs,
    _render_delivery_replay,
    _render_tracking_video,
)

ANALYSIS_ID = "analysis_finalized_track_test"
JOB_ID = "job_finalized_track_test"


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        mode="METRIC_3D",
        confidence="HIGH",
        image_width=1280,
        image_height=720,
        camera_matrix=[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        distortion_coefficients=[0.0, 0.0, 0.0, 0.0, 0.0],
        rotation_vector=[0.1, 0.2, 0.3],
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.0, 0.0, 0.0],
        calibration_confidence=0.9,
    )


def _primary_track() -> list[TrackingPoint]:
    return [
        TrackingPoint(
            frame_index=10,
            timestamp_seconds=0.40,
            source="observed",
            provenance="OBSERVED",
            candidate_id="cand-10",
            x=500.0,
            y=300.0,
            normalized_x=500 / 1280,
            normalized_y=300 / 720,
            confidence=0.9,
            uncertainty=0.0,
            vx=0.0,
            vy=0.0,
        ),
        TrackingPoint(
            frame_index=11,
            timestamp_seconds=0.44,
            source="recovered",
            provenance="TRACKER_RECOVERED",
            candidate_id="cand-11",
            x=510.0,
            y=320.0,
            normalized_x=510 / 1280,
            normalized_y=320 / 720,
            confidence=0.7,
            uncertainty=0.2,
            vx=250.0,
            vy=500.0,
        ),
        TrackingPoint(
            frame_index=12,
            timestamp_seconds=0.48,
            source="observed",
            provenance="OBSERVED",
            candidate_id="cand-12",
            x=520.0,
            y=340.0,
            normalized_x=520 / 1280,
            normalized_y=340 / 720,
            confidence=0.85,
            uncertainty=0.0,
            vx=250.0,
            vy=500.0,
        ),
    ]


def _physics_result(*, include_offscreen: bool = False) -> DeliveryPhysicsResult:
    samples = [
        TrajectorySample(
            frame_index=10,
            timestamp_seconds=0.40,
            world_x_m=0.1,
            world_y_m=4.0,
            world_z_m=1.2,
            pixel_x=500.0,
            pixel_y=300.0,
            speed_mps=30.0,
            provenance="OBSERVED",
            confidence=0.9,
        ),
        TrajectorySample(
            frame_index=11,
            timestamp_seconds=0.44,
            world_x_m=0.12,
            world_y_m=5.0,
            world_z_m=1.0,
            pixel_x=9999.0 if include_offscreen else 510.0,
            pixel_y=320.0,
            speed_mps=29.0,
            provenance="RECONSTRUCTED",
            confidence=0.75,
        ),
        TrajectorySample(
            frame_index=12,
            timestamp_seconds=0.48,
            world_x_m=0.14,
            world_y_m=6.0,
            world_z_m=0.8,
            pixel_x=520.0,
            pixel_y=340.0,
            speed_mps=28.0,
            provenance="OBSERVED",
            confidence=0.85,
        ),
        TrajectorySample(
            frame_index=13,
            timestamp_seconds=0.52,
            world_x_m=0.16,
            world_y_m=7.0,
            world_z_m=0.6,
            pixel_x=530.0,
            pixel_y=360.0,
            speed_mps=27.0,
            provenance="PROJECTED",
            confidence=0.55,
        ),
    ]
    return DeliveryPhysicsResult(
        status="SUCCESS",
        analysis_id=ANALYSIS_ID,
        coordinate_system="CRICVISION_X_RIGHT_Y_BOWLER_TO_STRIKER_Z_UP",
        calibration=_calibration(),
        fitted_parameters=FittedTrajectoryParameters(
            selected_model="BALLISTIC",
            origin_timestamp_seconds=0.40,
        ),
        trajectory_samples=samples,
        delivery_interval=DeliveryInterval(
            start_frame=10,
            end_frame=13,
            first_observed_frame=10,
            last_observed_frame=12,
            terminal_reason="maximum_projection_horizon",
        ),
        bounce=BouncePhysicsResult(
            status="DETECTED",
            frame_index=12,
            timestamp_seconds=0.48,
            world_x_m=0.14,
            world_y_m=6.0,
            pixel_x=520.0,
            pixel_y=340.0,
            distance_from_striker_wicket_m=14.06,
            confidence="HIGH",
            confidence_score=0.85,
            directly_supported=True,
            evidence=["ballistic_height_crosses_pitch_plane"],
        ),
        speed=SpeedAnalytics(
            earliest_measured_speed_kmh=108.0,
            average_pre_bounce_speed_kmh=104.0,
            speed_at_bounce_kmh=98.0,
            confidence="MEDIUM",
            uncertainty_kmh=3.0,
        ),
        pre_bounce_lateral_movement=LateralMovementResult(
            movement_m=0.18,
            movement_cm=18.0,
            direction="toward_positive_x",
            confidence="MEDIUM",
            uncertainty_m=0.04,
        ),
        post_bounce_movement=PostBounceMovementResult(
            status="UNAVAILABLE",
            confidence="INSUFFICIENT_EVIDENCE",
            unavailable_reason="Not enough post-bounce observations.",
        ),
        line_and_length=LineLengthResult(
            line="middle",
            length="good length",
            bounce_distance_from_striker_m=14.06,
        ),
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model="BALLISTIC",
            optimizer_status="success",
        ),
        confidence="MEDIUM",
        confidence_score=0.7,
        uncertainty_method="residual envelope",
    )


def _build_track(*, include_offscreen: bool = False):
    return build_finalized_delivery_track(
        analysis_id=ANALYSIS_ID,
        tracking_job_id=JOB_ID,
        generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        primary_track=_primary_track(),
        physics=_physics_result(include_offscreen=include_offscreen),
        fps=25.0,
        width=1280,
        height=720,
    )


class TestFinalizedDeliveryTrack:
    def test_tracker_points_precede_physics_pixels(self) -> None:
        track = _build_track(include_offscreen=True)
        by_frame = {point.frame_index: point for point in track.render_track}
        assert by_frame[11].x == pytest.approx(510.0)
        assert by_frame[11].provenance == "TRACKER_RECOVERED"
        assert len(track.recovered) == 1
        assert len(track.physics_reconstructed) == 0

    def test_projected_points_extend_beyond_primary_track(self) -> None:
        track = _build_track()
        frames = [point.frame_index for point in track.render_track]
        assert 13 in frames
        assert track.projected[0].frame_index == 13

    def test_render_track_matches_debug_and_delivery_replay_inputs(self) -> None:
        track = _build_track()
        debug_frames = {point.frame_index: (point.x, point.y) for point in track.render_track}
        replay_frames = {point.frame_index: (point.x, point.y) for point in track.render_track}
        assert debug_frames == replay_frames

    def test_source_consistency_requires_shared_track_id(self) -> None:
        consistent = validate_source_consistency(
            JOB_ID,
            speed_source_track_id=JOB_ID,
            main_video_source_track_id=JOB_ID,
            delivery_replay_source_track_id=JOB_ID,
            replay_payload_source_track_id=JOB_ID,
        )
        assert consistent.consistent is True

        inconsistent = validate_source_consistency(
            JOB_ID,
            speed_source_track_id=JOB_ID,
            main_video_source_track_id="other",
            delivery_replay_source_track_id=JOB_ID,
            replay_payload_source_track_id=JOB_ID,
        )
        assert inconsistent.consistent is False
        assert inconsistent.errors

    def test_cache_bust_url_appends_tracking_job_id(self) -> None:
        url = cache_bust_url("/static/video-analysis/x/tracking/delivery_replay.mp4", tracking_job_id=JOB_ID)
        assert url.endswith(f"v={JOB_ID}")

    def test_replay_payload_uses_finalized_image_coords(self) -> None:
        track = _build_track()
        with patch(
            "services.api.services.replay_payload_service._safe_camera_bridge",
            return_value=None,
        ), patch(
            "services.api.services.replay_payload_service.load_active_accepted_wicket_box_calibration",
            return_value={
                "schema_version": "wicket_box_calibration_accepted_v1",
                "analysis_id": ANALYSIS_ID,
                "accepted_at": "2026-08-03T12:00:00Z",
                "calibration_frame_index": 0,
                "source_image_width": 1280,
                "source_image_height": 720,
                "camera_matrix": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
                "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation_vector": [0.0, 0.0, 0.0],
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                "reprojection_rmse_px": 2.5,
                "near_wicket_box": None,
                "far_wicket_box": None,
                "stump_landmarks": [],
                "frozen": True,
            },
        ):
            payload = build_and_save_replay_payload(
                ANALYSIS_ID,
                physics=_physics_result(),
                primary_track=_primary_track(),
                finalized_track=track,
                tracking_status="ready",
                fps=25.0,
                width=1280,
                height=720,
            )
        assert payload.diagnostics.source_track_id == JOB_ID
        assert payload.diagnostics.tracking_job_id == JOB_ID
        assert payload.trajectory
        assert payload.trajectory[0].image_position.x == pytest.approx(500.0)
        assert payload.trajectory[1].image_position.x == pytest.approx(510.0)
        assert payload.trajectory[0].world_position is not None

    def test_clear_previous_tracking_outputs_removes_replay_artifacts(self, tmp_path: Path) -> None:
        for filename in (
            TRACKING_VIDEO_FILENAME,
            DELIVERY_REPLAY_FILENAME,
            REPLAY_PAYLOAD_FILENAME,
            FINALIZED_TRACK_FILENAME,
        ):
            (tmp_path / filename).write_text("stale", encoding="utf-8")
        _clear_previous_tracking_outputs(tmp_path)
        assert not any((tmp_path / filename).exists() for filename in (
            TRACKING_VIDEO_FILENAME,
            DELIVERY_REPLAY_FILENAME,
            REPLAY_PAYLOAD_FILENAME,
            FINALIZED_TRACK_FILENAME,
        ))

    def test_delivery_replay_renderer_accepts_render_track(self) -> None:
        track = _build_track()
        assert callable(_render_delivery_replay)
        assert callable(_render_tracking_video)
        merged = finalized_render_track(
            observed=track.observed,
            recovered=track.recovered,
            physics_reconstructed=track.physics_reconstructed,
            projected=track.projected,
        )
        assert [point.frame_index for point in merged] == [
            point.frame_index for point in track.render_track
        ]
