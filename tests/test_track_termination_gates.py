"""Tests for track termination gates and stump-to-stump speed."""

from __future__ import annotations

from datetime import datetime, timezone

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
from services.api.services.delivery_physics_service import _overall_stump_to_stump_speed
from services.api.services.finalized_delivery_track import (
    FinalizedTrackPoint,
    build_finalized_delivery_track,
    gate_physics_extensions,
)
from services.api.services.video_ball_tracking_service import _terminate_tracking_points


def _physics_with_bad_projection() -> DeliveryPhysicsResult:
    samples = [
        TrajectorySample(
            frame_index=10,
            timestamp_seconds=0.40,
            world_x_m=0.0,
            world_y_m=0.5,
            world_z_m=1.5,
            pixel_x=300.0,
            pixel_y=400.0,
            speed_mps=30.0,
            provenance="OBSERVED",
            confidence=0.9,
        ),
        TrajectorySample(
            frame_index=11,
            timestamp_seconds=0.44,
            world_x_m=0.05,
            world_y_m=2.0,
            world_z_m=1.3,
            pixel_x=310.0,
            pixel_y=390.0,
            speed_mps=29.0,
            provenance="OBSERVED",
            confidence=0.85,
        ),
        TrajectorySample(
            frame_index=12,
            timestamp_seconds=0.48,
            world_x_m=0.1,
            world_y_m=3.0,
            world_z_m=1.0,
            pixel_x=320.0,
            pixel_y=380.0,
            speed_mps=28.0,
            provenance="OBSERVED",
            confidence=0.8,
        ),
        TrajectorySample(
            frame_index=13,
            timestamp_seconds=0.52,
            world_x_m=0.2,
            world_y_m=18.0,
            world_z_m=0.5,
            pixel_x=700.0,
            pixel_y=750.0,
            speed_mps=27.0,
            provenance="PROJECTED",
            confidence=0.4,
        ),
    ]
    return DeliveryPhysicsResult(
        status="SUCCESS",
        analysis_id="analysis_gate_test",
        coordinate_system="CRICVISION_X_RIGHT_Y_BOWLER_TO_STRIKER_Z_UP",
        calibration=CameraCalibration(
            mode="METRIC_3D",
            confidence="HIGH",
            image_width=720,
            image_height=1280,
        ),
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
            status="INSUFFICIENT_EVIDENCE",
            confidence="INSUFFICIENT_EVIDENCE",
            confidence_score=0.0,
            directly_supported=False,
        ),
        speed=SpeedAnalytics(confidence="MEDIUM"),
        pre_bounce_lateral_movement=LateralMovementResult(
            direction="negligible",
            confidence="MEDIUM",
        ),
        post_bounce_movement=PostBounceMovementResult(
            status="UNAVAILABLE",
            confidence="INSUFFICIENT_EVIDENCE",
        ),
        line_and_length=LineLengthResult(line="middle", length="good length"),
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model="BALLISTIC",
            optimizer_status="success",
        ),
        confidence="MEDIUM",
        confidence_score=0.6,
        uncertainty_method="residual envelope",
    )


class TestPhysicsExtensionGating:
    def test_rejects_discontinuous_projected_pixel_jump(self) -> None:
        physics = _physics_with_bad_projection()
        primary = [
            TrackingPoint(
                frame_index=10,
                timestamp_seconds=0.40,
                source="observed",
                provenance="OBSERVED",
                candidate_id="c10",
                x=300.0,
                y=400.0,
                normalized_x=300 / 720,
                normalized_y=400 / 1280,
                confidence=0.9,
                uncertainty=0.0,
                vx=0.0,
                vy=0.0,
            ),
            TrackingPoint(
                frame_index=11,
                timestamp_seconds=0.44,
                source="observed",
                provenance="OBSERVED",
                candidate_id="c11",
                x=310.0,
                y=390.0,
                normalized_x=310 / 720,
                normalized_y=390 / 1280,
                confidence=0.85,
                uncertainty=0.0,
                vx=250.0,
                vy=-250.0,
            ),
            TrackingPoint(
                frame_index=12,
                timestamp_seconds=0.48,
                source="observed",
                provenance="OBSERVED",
                candidate_id="c12",
                x=320.0,
                y=380.0,
                normalized_x=320 / 720,
                normalized_y=380 / 1280,
                confidence=0.8,
                uncertainty=0.0,
                vx=250.0,
                vy=-250.0,
            ),
        ]
        track = build_finalized_delivery_track(
            analysis_id="analysis_gate_test",
            tracking_job_id="job_gate_test",
            generated_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
            primary_track=primary,
            physics=physics,
            fps=25.0,
            width=720,
            height=1280,
        )
        assert track.projected == []
        assert track.render_track[-1].frame_index == 12
        assert track.termination.first_invalid_frame == 13
        assert track.termination.first_invalid_reason == "pixel_jump_exceeds_gate"

    def test_gate_physics_extensions_stops_after_first_invalid(self) -> None:
        observed = [
            FinalizedTrackPoint(
                frame_index=1,
                timestamp_seconds=0.04,
                x=100.0,
                y=200.0,
                normalized_x=0.1,
                normalized_y=0.2,
                provenance="OBSERVED",
                confidence=0.9,
                source="observed",
            )
        ]
        projected = [
            FinalizedTrackPoint(
                frame_index=2,
                timestamp_seconds=0.08,
                x=500.0,
                y=800.0,
                normalized_x=0.7,
                normalized_y=0.8,
                provenance="PROJECTED",
                confidence=0.2,
                source="predicted",
            )
        ]
        _, accepted_projected, termination = gate_physics_extensions(
            observed=observed,
            recovered=[],
            physics_reconstructed=[],
            projected=projected,
            width=720,
            height=1280,
            fps=25.0,
        )
        assert accepted_projected == []
        assert termination.first_invalid_frame == 2


class TestTrackingTailTermination:
    def test_trims_late_false_detection(self) -> None:
        points = [
            TrackingPoint(
                frame_index=index,
                timestamp_seconds=index / 25.0,
                source="observed",
                provenance="OBSERVED",
                candidate_id=f"c{index}",
                x=100.0 + index * 5,
                y=400.0 - index * 3,
                normalized_x=(100.0 + index * 5) / 720,
                normalized_y=(400.0 - index * 3) / 1280,
                confidence=0.8,
                uncertainty=0.0,
                vx=125.0,
                vy=-75.0,
            )
            for index in range(5)
        ]
        points.append(
            points[-1].model_copy(
                update={
                    "frame_index": 5,
                    "timestamp_seconds": 0.2,
                    "x": points[-1].x + 400,
                    "y": points[-1].y + 300,
                    "candidate_id": "c5",
                }
            )
        )
        trimmed = _terminate_tracking_points(points, fps=25.0)
        assert len(trimmed) == 5
        assert trimmed[-1].frame_index == 4


class TestOverallStumpToStumpSpeed:
    def test_computes_path_average_between_wicket_planes(self) -> None:
        ys = [-0.5, 5.0, 10.0, 15.0, 21.0]
        samples = [
            TrajectorySample(
                frame_index=index,
                timestamp_seconds=index * 0.25,
                world_x_m=0.0,
                world_y_m=y,
                world_z_m=1.5 - index * 0.1,
                pixel_x=100.0,
                pixel_y=200.0,
                speed_mps=22.0,
                provenance="OBSERVED" if index < 4 else "PROJECTED",
                confidence=0.9,
            )
            for index, y in enumerate(ys)
        ]
        result = _overall_stump_to_stump_speed(samples)
        assert result.status in {"MEASURED", "PARTIALLY_PROJECTED"}
        assert result.speed_kph is not None
        assert result.travelled_distance_m is not None
        assert result.start_crossing is not None
        assert result.end_crossing is not None

    def test_unavailable_when_planes_not_crossed(self) -> None:
        samples = [
            TrajectorySample(
                frame_index=0,
                timestamp_seconds=0.0,
                world_x_m=0.0,
                world_y_m=2.0,
                world_z_m=1.0,
                pixel_x=10.0,
                pixel_y=10.0,
                speed_mps=20.0,
                provenance="OBSERVED",
                confidence=0.9,
            )
        ]
        result = _overall_stump_to_stump_speed(samples)
        assert result.status == "UNAVAILABLE"
