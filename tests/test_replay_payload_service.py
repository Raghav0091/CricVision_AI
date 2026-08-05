"""Stage 4 Virtual Pitch Replay payload assembly tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from services.api.schemas.camera_bridge import (
    CameraBridgeDistortion,
    CameraBridgeInput,
    CameraBridgeResponse,
)
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
from services.api.schemas.replay_payload import ReplayPayloadV1
from services.api.schemas.video_analysis import TrackingPoint
from services.api.services import replay_payload_service as replay_service
from services.api.services.finalized_delivery_track import build_finalized_delivery_track
from services.api.services.replay_payload_service import (
    REPLAY_PAYLOAD_FILENAME,
    build_and_save_replay_payload,
    load_replay_payload,
)
from services.api.services.video_ball_tracking_service import (
    DELIVERY_REPLAY_FILENAME,
    _render_delivery_replay,
)


ANALYSIS_ID = "analysis_20260803_213801_b1c2d3"


def _calibration(
    *,
    mode: str = "METRIC_3D",
) -> CameraCalibration:
    return CameraCalibration(
        mode=mode,
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


def _physics_result(
    *,
    status: str = "SUCCESS",
    calibration: CameraCalibration | None = None,
    samples: list[TrajectorySample] | None = None,
    bounce: BouncePhysicsResult | None = None,
    speed: SpeedAnalytics | None = None,
    lateral: LateralMovementResult | None = None,
) -> DeliveryPhysicsResult:
    calibration = calibration or _calibration()
    samples = samples or [
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
            pixel_x=510.0,
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
            provenance="PROJECTED",
            confidence=0.55,
        ),
    ]
    bounce = bounce or BouncePhysicsResult(
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
    )
    speed = speed or SpeedAnalytics(
        earliest_measured_speed_kmh=108.0,
        average_pre_bounce_speed_kmh=104.0,
        speed_at_bounce_kmh=98.0,
        confidence="MEDIUM",
        uncertainty_kmh=3.0,
    )
    lateral = lateral or LateralMovementResult(
        movement_m=0.18,
        movement_cm=18.0,
        direction="toward_positive_x",
        confidence="MEDIUM",
        uncertainty_m=0.04,
    )
    return DeliveryPhysicsResult(
        status=status,
        analysis_id=ANALYSIS_ID,
        coordinate_system="CRICVISION_X_RIGHT_Y_BOWLER_TO_STRIKER_Z_UP",
        calibration=calibration,
        fitted_parameters=FittedTrajectoryParameters(
            selected_model="BALLISTIC",
            origin_timestamp_seconds=0.40,
        ),
        trajectory_samples=samples,
        delivery_interval=DeliveryInterval(
            start_frame=10,
            end_frame=12,
            first_observed_frame=10,
            last_observed_frame=12,
            terminal_reason="maximum_projection_horizon",
        ),
        bounce=bounce,
        speed=speed,
        pre_bounce_lateral_movement=lateral,
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


def _tracking_points() -> list[TrackingPoint]:
    return [
        TrackingPoint(
            frame_index=10,
            timestamp_seconds=0.40,
            source="observed",
            provenance="OBSERVED",
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
            x=510.0,
            y=320.0,
            normalized_x=510 / 1280,
            normalized_y=320 / 720,
            confidence=0.7,
            uncertainty=0.2,
            vx=250.0,
            vy=500.0,
        ),
    ]


def _finalized_track() -> object:
    return build_finalized_delivery_track(
        analysis_id=ANALYSIS_ID,
        tracking_job_id="job_replay_payload_test",
        generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
        primary_track=_tracking_points(),
        physics=_physics_result(),
        fps=25.0,
        width=1280,
        height=720,
    )


def _accepted_bridge() -> CameraBridgeResponse:
    distortion = CameraBridgeDistortion(
        mode="ZERO_DISTORTION",
        coefficients=[0.0, 0.0, 0.0, 0.0],
        coefficient_order="OpenCV",
        maximum_absolute_coefficient=0.0,
        frame_preundistorted=False,
        exact_pinhole_rendering_supported=True,
    )
    camera = CameraBridgeInput(
        source="ACCEPTED_SCENE_CALIBRATION",
        source_version="scene_calibration_v1_revision_1",
        analysis_id=ANALYSIS_ID,
        candidate_id="candidate-1",
        accepted=True,
        classification="METRIC_3D_READY",
        image_width=1280,
        image_height=720,
        camera_matrix=[[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        fx=1000.0,
        fy=1000.0,
        cx=640.0,
        cy=360.0,
        skew=0.0,
        distortion=distortion,
        rotation_vector=[0.1, 0.2, 0.3],
        rotation_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_vector=[0.0, 0.0, 0.0],
        camera_world_position=[-8.0, 0.0, 6.0],
        near_m=0.01,
        far_m=250.0,
    )
    return CameraBridgeResponse(
        status="AVAILABLE",
        camera=camera,
        message="Accepted calibration.",
    )


def _wicket_box_bridge() -> CameraBridgeResponse:
    camera = _accepted_bridge().camera.model_copy(
        update={
            "source": "ACCEPTED_WICKET_BOX_CALIBRATION",
            "source_version": "wicket_box_calibration_accepted_v1",
            "candidate_id": "wicket-box-accepted",
        }
    )
    return CameraBridgeResponse(
        status="AVAILABLE",
        camera=camera,
        message="Accepted wicket-box calibration.",
    )


def _preset_bridge() -> CameraBridgeResponse:
    camera = _accepted_bridge().camera.model_copy(
        update={
            "source": "SYNTHETIC_VIRTUAL_PITCH",
            "accepted": False,
            "classification": "SYNTHETIC_EXACT",
        }
    )
    return CameraBridgeResponse(
        status="AVAILABLE",
        camera=camera,
        message="Synthetic preset.",
    )


class TestReplayPayloadAssembly:
    def test_provenance_and_timestamp_ordering(self) -> None:
        physics = _physics_result()
        trajectory = replay_service._build_trajectory(
            physics=physics,
            primary_track=_tracking_points(),
            finalized_track=_finalized_track(),
            measurement_validity="CALIBRATED",
        )
        assert [item.provenance for item in trajectory] == [
            "OBSERVED",
            "RECOVERED",
            "PHYSICS_FITTED",
        ]
        timestamps = [item.timestamp_seconds for item in trajectory]
        assert timestamps == sorted(timestamps)

    def test_image_space_only_suppresses_world_and_metrics(self) -> None:
        physics = _physics_result(
            calibration=_calibration(mode="IMAGE_SPACE_ONLY"),
            status="IMAGE_SPACE_ONLY",
        )
        trajectory = replay_service._build_trajectory(
            physics=physics,
            primary_track=_tracking_points(),
            finalized_track=_finalized_track(),
            measurement_validity="IMAGE_SPACE_ONLY",
        )
        assert all(item.world_position is None for item in trajectory)
        assert all(item.image_position is not None for item in trajectory)

        metrics = replay_service._build_metrics(
            physics,
            "IMAGE_SPACE_ONLY",
            bounce=replay_service._build_bounce(physics, "IMAGE_SPACE_ONLY"),
        )
        assert metrics.release_speed_kmh.value is None
        assert metrics.delivery_length_m.value is None
        assert metrics.estimated_lateral_deviation_m.value is None

        payload = build_and_save_replay_payload(
            ANALYSIS_ID,
            physics=physics,
            primary_track=_tracking_points(),
            finalized_track=_finalized_track(),
            tracking_status="ready",
            fps=25.0,
            width=1280,
            height=720,
        )
        assert payload.measurement_validity == "IMAGE_SPACE_ONLY"

    def test_visualization_only_restricts_metrics(self) -> None:
        camera, accepted = replay_service._build_replay_camera(
            _preset_bridge(),
            _physics_result(),
            1280,
            720,
        )
        assert accepted is False
        assert camera.visualization_only is True
        validity = replay_service._resolve_measurement_validity(
            has_accepted_calibration=False,
            camera=camera,
            physics=_physics_result(),
            tracking_status="ready",
            primary_track=_tracking_points(),
        )
        assert validity == "VISUALIZATION_ONLY"

        with patch.object(
            replay_service,
            "_safe_camera_bridge",
            return_value=_preset_bridge(),
        ):
            payload = build_and_save_replay_payload(
                ANALYSIS_ID,
                physics=_physics_result(),
                primary_track=_tracking_points(),
                finalized_track=_finalized_track(),
                tracking_status="ready",
                fps=25.0,
                width=1280,
                height=720,
            )
        assert payload.measurement_validity == "VISUALIZATION_ONLY"
        assert payload.metrics.release_speed_kmh.value is None
        assert payload.camera.visualization_only is True

    def test_no_bounce_means_no_length(self) -> None:
        bounce = BouncePhysicsResult(
            status="INSUFFICIENT_EVIDENCE",
            confidence="INSUFFICIENT_EVIDENCE",
            confidence_score=0.0,
            evidence=[],
        )
        physics = _physics_result(bounce=bounce)
        metrics = replay_service._build_metrics(
            physics,
            "CALIBRATED",
            bounce=replay_service._build_bounce(physics, "CALIBRATED"),
        )
        assert metrics.delivery_length_m.value is None
        assert metrics.delivery_length_m.status == "UNAVAILABLE"

    def test_insufficient_evidence_suppresses_speed_and_deviation(self) -> None:
        physics = _physics_result(
            status="INSUFFICIENT_EVIDENCE",
            speed=SpeedAnalytics(
                confidence="INSUFFICIENT_EVIDENCE",
                unavailable_reason="Too few observations.",
            ),
            lateral=LateralMovementResult(
                direction="unavailable",
                confidence="INSUFFICIENT_EVIDENCE",
                unavailable_reason="Too few observations.",
            ),
        )
        metrics = replay_service._build_metrics(
            physics,
            "INSUFFICIENT_EVIDENCE",
            bounce=replay_service._build_bounce(
                physics,
                "INSUFFICIENT_EVIDENCE",
            ),
        )
        assert metrics.release_speed_kmh.value is None
        assert metrics.estimated_lateral_deviation_m.value is None

    def test_wicket_box_accepted_bridge_is_calibrated(self) -> None:
        camera, accepted = replay_service._build_replay_camera(
            _wicket_box_bridge(),
            _physics_result(),
            1280,
            720,
        )
        assert accepted is True
        assert camera.source == "CALIBRATED"
        assert camera.visualization_only is False

    def test_resolve_replay_camera_uses_accepted_snapshot_without_setup_frame(
        self,
    ) -> None:
        snapshot = {
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
        }
        physics = _physics_result()
        with patch.object(
            replay_service,
            "_safe_camera_bridge",
            return_value=None,
        ), patch.object(
            replay_service,
            "load_active_accepted_wicket_box_calibration",
            return_value=snapshot,
        ):
            camera, accepted = replay_service._resolve_replay_camera(
                ANALYSIS_ID,
                physics,
                1280,
                720,
            )
            validity = replay_service._resolve_measurement_validity(
                has_accepted_calibration=accepted,
                camera=camera,
                physics=physics,
                tracking_status="ready",
                primary_track=_tracking_points(),
            )
            trajectory = replay_service._build_trajectory(
                physics=physics,
                primary_track=_tracking_points(),
                finalized_track=_finalized_track(),
                measurement_validity=validity,
            )
        assert accepted is True
        assert camera.source == "CALIBRATED"
        assert validity == "CALIBRATED"
        assert trajectory[0].world_position is not None

    def test_calibrated_payload_serializes_through_schema(self) -> None:
        with patch.object(
            replay_service,
            "_safe_camera_bridge",
            return_value=_accepted_bridge(),
        ):
            payload = build_and_save_replay_payload(
                ANALYSIS_ID,
                physics=_physics_result(),
                primary_track=_tracking_points(),
                finalized_track=_finalized_track(),
                tracking_status="ready",
                fps=25.0,
                width=1280,
                height=720,
            )
        restored = ReplayPayloadV1.model_validate_json(payload.model_dump_json())
        assert restored.measurement_validity == "CALIBRATED"
        assert restored.diagnostics.status == "READY"
        assert restored.metrics.release_speed_kmh.value == 108.0
        assert restored.metrics.estimated_lateral_deviation_m.value == 0.18
        assert restored.metrics.delivery_length_m.value == 14.06
        assert restored.trajectory[0].world_position is not None

    def test_saved_payload_loads_from_disk(self) -> None:
        payload = build_and_save_replay_payload(
            ANALYSIS_ID,
            physics=_physics_result(),
            primary_track=_tracking_points(),
            finalized_track=_finalized_track(),
            tracking_status="ready",
            fps=25.0,
            width=1280,
            height=720,
        )
        target = (
            Path(__file__).resolve().parent
            / "_replay_payload_fixture"
            / ANALYSIS_ID
            / "tracking"
            / REPLAY_PAYLOAD_FILENAME
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
        try:
            with patch.object(
                replay_service,
                "_replay_payload_path",
                return_value=target,
            ), patch.object(
                replay_service,
                "_saved_replay_payload_is_current",
                return_value=True,
            ):
                loaded = load_replay_payload(ANALYSIS_ID)
            assert loaded.analysis_id == ANALYSIS_ID
            assert loaded.diagnostics.status == payload.diagnostics.status
        finally:
            if target.is_file():
                target.unlink()
            for parent in (
                target.parent,
                target.parent.parent,
                target.parent.parent.parent,
            ):
                if parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()


class TestLegacyReplayRegression:
    def test_delivery_replay_renderer_still_present(self) -> None:
        assert callable(_render_delivery_replay)
        assert DELIVERY_REPLAY_FILENAME == "delivery_replay.mp4"

    def test_tracking_service_still_references_delivery_replay(self) -> None:
        source = Path(
            "services/api/services/video_ball_tracking_service.py"
        ).read_text(encoding="utf-8")
        assert "_render_delivery_replay" in source
        assert DELIVERY_REPLAY_FILENAME in source
