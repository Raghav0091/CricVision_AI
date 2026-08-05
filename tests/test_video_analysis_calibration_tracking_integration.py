"""E2E integration: accepted wicket-box calibration → physics → replay payload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.schemas.delivery_physics import (
    BouncePhysicsResult,
    DeliveryInterval,
    DeliveryPhysicsResult,
    FitDiagnostics,
    FittedTrajectoryParameters,
    GeometryValidationResult,
    LateralMovementResult,
    LineLengthResult,
    PostBounceMovementResult,
    SpeedAnalytics,
    TrajectorySample,
)
from services.api.schemas.replay_payload import ReplayPayloadV1
from services.api.schemas.video_analysis import (
    PrimaryBounceResult,
    TrackingPoint,
    VideoBallDetectionsDocument,
)
from services.api.schemas.wicket_box_calibration import (
    WicketBoxCalibrationAcceptRequest,
    WicketBoxCalibrationRegisterRequest,
)
from services.api.services.delivery_physics_service import (
    analyse_delivery_physics,
    load_physics_calibration,
)
from services.api.services.finalized_delivery_track import build_finalized_delivery_track
from services.api.services.replay_payload_service import (
    REPLAY_PAYLOAD_FILENAME,
    build_and_save_replay_payload,
    load_replay_payload,
)
from services.api.services.video_ball_tracking_service import (
    TRACKING_RESULT_FILENAME,
    TRACKING_SUMMARY_FILENAME,
    invalidate_tracking_after_calibration_change,
)
from services.api.services.wicket_box_calibration_service import (
    ACCEPTED_FILENAME,
    _observation_from_landmarks,
    accept_wicket_box_calibration,
    register_wicket_box_calibration,
)
from tests.test_real_pitch_registration import _solve
from tests.test_delivery_physics import synthetic_calibration, synthetic_observations
from tests.test_replay_payload_service import _wicket_box_bridge
from tests.test_wicket_box_calibration import (
    ANALYSIS_ID,
    _empty_registration_summary,
    _register_request,
    _registration,
    _synthetic_landmarks,
    isolated_wicket_box,
)


client = TestClient(app)


def _accepted_wicket_box_snapshot() -> dict[str, object]:
    calibration = synthetic_calibration()
    return {
        "schema_version": "wicket_box_calibration_accepted_v1",
        "analysis_id": ANALYSIS_ID,
        "accepted_at": "2026-08-03T12:00:00Z",
        "calibration_frame_index": 4,
        "source_image_width": 1280,
        "source_image_height": 720,
        "camera_matrix": calibration.camera_matrix,
        "rotation_matrix": calibration.rotation_matrix,
        "translation_vector": calibration.translation_vector,
        "distortion_coefficients": calibration.distortion_coefficients,
        "reprojection_rmse_px": 2.5,
        "near_wicket_box": None,
        "far_wicket_box": None,
        "stump_landmarks": [],
        "frozen": True,
    }


def _write_accepted_snapshot(root: Path) -> None:
    path = root / ANALYSIS_ID / "reports" / ACCEPTED_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_accepted_wicket_box_snapshot(), indent=2), encoding="utf-8")


def _primary_track(count: int = 10) -> list[TrackingPoint]:
    return [
        TrackingPoint(
            frame_index=index,
            timestamp_seconds=index * 0.04,
            source="observed",
            provenance="OBSERVED",
            candidate_id=f"candidate-{index}",
            x=500.0 + index * 4.0,
            y=300.0 + index * 3.0,
            normalized_x=(500.0 + index * 4.0) / 1280.0,
            normalized_y=(300.0 + index * 3.0) / 720.0,
            confidence=0.9,
            vx=100.0,
            vy=75.0,
        )
        for index in range(count)
    ]


def _detections_document(track: list[TrackingPoint]) -> VideoBallDetectionsDocument:
    frames = []
    for point in track:
        half = 6.0
        frames.append(
            {
                "frame_index": point.frame_index,
                "timestamp_seconds": point.timestamp_seconds,
                "processed": True,
                "detections": [
                    {
                        "candidate_id": point.candidate_id,
                        "class_id": 0,
                        "class_name": "ball",
                        "confidence": point.confidence,
                        "bbox_xyxy": [
                            point.x - half,
                            point.y - half,
                            point.x + half,
                            point.y + half,
                        ],
                        "bbox_normalized": {
                            "x": point.normalized_x - 0.01,
                            "y": point.normalized_y - 0.01,
                            "width": 0.02,
                            "height": 0.02,
                        },
                        "center": {"x": point.x, "y": point.y},
                        "center_normalized": {
                            "x": point.normalized_x,
                            "y": point.normalized_y,
                        },
                        "width_pixels": half * 2,
                        "height_pixels": half * 2,
                        "area_pixels": (half * 2) ** 2,
                        "inside_pitch_corridor": True,
                    }
                ],
            }
        )
    return VideoBallDetectionsDocument.model_validate(
        {
            "analysis_id": ANALYSIS_ID,
            "model_path_used": "test-model.pt",
            "model_class_names": ["ball"],
            "settings": {
                "frame_stride": 1,
                "imgsz": 960,
                "confidence_threshold": 0.15,
                "max_det": 20,
            },
            "frames": frames,
        }
    )


def _metric_physics_result() -> DeliveryPhysicsResult:
    calibration = synthetic_calibration()
    return DeliveryPhysicsResult(
        status="SUCCESS",
        analysis_id=ANALYSIS_ID,
        coordinate_system="cricvision_pitch_v1",
        calibration=calibration,
        fitted_parameters=FittedTrajectoryParameters(
            selected_model="BALLISTIC",
            origin_timestamp_seconds=0.0,
        ),
        trajectory_samples=[
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
        ],
        accepted_observations=[],
        rejected_observations=[],
        delivery_interval=DeliveryInterval(start_frame=10, end_frame=11),
        bounce=BouncePhysicsResult(
            status="DETECTED",
            frame_index=11,
            timestamp_seconds=0.44,
            distance_from_striker_wicket_m=12.0,
            lateral_offset_m=0.1,
            confidence="HIGH",
            confidence_score=0.8,
            pixel_x=510.0,
            pixel_y=320.0,
        ),
        speed=SpeedAnalytics(
            earliest_measured_speed_kmh=108.0,
            average_pre_bounce_speed_kmh=105.0,
            speed_at_bounce_kmh=104.0,
            confidence="HIGH",
        ),
        pre_bounce_lateral_movement=LateralMovementResult(
            movement_m=0.12,
            movement_cm=12.0,
            direction="leg",
            confidence="MEDIUM",
        ),
        post_bounce_movement=PostBounceMovementResult(
            status="MEASURED",
            lateral_turn_cm_at_last_observation=4.0,
            confidence="LOW",
        ),
        line_and_length=LineLengthResult(line="middle", length="good length"),
        fit_diagnostics=FitDiagnostics(
            converged=True,
            selected_model="BALLISTIC",
            weighted_reprojection_rmse_px=2.75,
            inlier_frames=[10, 11],
            outlier_frames=[],
            processing_duration_seconds=0.1,
        ),
        confidence="HIGH",
        confidence_score=0.85,
        exact_spin_rpm=None,
        exact_spin_rpm_unavailable_reason="not measured",
        warnings=[],
    )


@pytest.fixture
def integrated_analysis(
    isolated_wicket_box: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = isolated_wicket_box
    for module_path in (
        "services.api.services.replay_payload_service.VIDEO_ANALYSIS_ROOT",
        "services.api.services.video_ball_tracking_service.VIDEO_ANALYSIS_ROOT",
        "services.api.services.camera_bridge_service.VIDEO_ANALYSIS_ROOT",
    ):
        monkeypatch.setattr(module_path, root)

    metadata_dir = root / ANALYSIS_ID / "reports"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.joinpath("analysis_metadata.json").write_text(
        json.dumps(
            {
                "analysis_id": ANALYSIS_ID,
                "width": 1280,
                "height": 720,
                "fps": 30.0,
                "frame_count": 120,
                "tracking_status": "tracking_complete",
            }
        ),
        encoding="utf-8",
    )
    tracking_dir = root / ANALYSIS_ID / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.joinpath(TRACKING_SUMMARY_FILENAME).write_text(
        json.dumps({"status": "ready", "analysis_id": ANALYSIS_ID}),
        encoding="utf-8",
    )
    tracking_dir.joinpath(TRACKING_RESULT_FILENAME).write_text(
        json.dumps({"primary_track": [], "analysis_id": ANALYSIS_ID}),
        encoding="utf-8",
    )
    return root


class TestAcceptedWicketBoxPhysicsIntegration:
    def test_load_physics_calibration_uses_accepted_wicket_box(
        self,
        integrated_analysis: Path,
    ) -> None:
        _write_accepted_snapshot(integrated_analysis)
        calibration = load_physics_calibration(ANALYSIS_ID, 1280, 720)
        assert calibration.mode == "METRIC_3D"
        assert calibration.reprojection_error_px == pytest.approx(2.5)
        assert "wicket-box" in calibration.warnings[0].lower()

    def test_accept_clears_stale_tracking_outputs(
        self,
        integrated_analysis: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tracking_dir = integrated_analysis / ANALYSIS_ID / "tracking"
        stale = tracking_dir / REPLAY_PAYLOAD_FILENAME
        stale.write_text("{}", encoding="utf-8")
        observation = _observation_from_landmarks(
            analysis_id=ANALYSIS_ID,
            frame_index=4,
            fps=30.0,
            near_box=_register_request().near_wicket_box,
            far_box=_register_request().far_wicket_box,
            landmarks=_synthetic_landmarks(),
        )
        candidate, _ = _solve(observation)
        registration = _registration(observation).model_copy(
            update={"selected_candidate": candidate, "status": candidate.classification}
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._register_from_observation",
            lambda *args, **kwargs: (
                registration,
                [],
                False,
                _empty_registration_summary(),
            ),
        )
        monkeypatch.setattr(
            "services.api.services.wicket_box_calibration_service._registration_rejection_reasons",
            lambda candidate, registration: [],
        )
        registered = register_wicket_box_calibration(
            ANALYSIS_ID,
            _register_request(landmarks=_synthetic_landmarks()),
        )
        assert registered.success is True
        accepted = accept_wicket_box_calibration(
            ANALYSIS_ID,
            WicketBoxCalibrationAcceptRequest(
                analysis_id=ANALYSIS_ID,
                accept_registered_calibration=True,
            ),
        )
        assert accepted.success is True
        assert not stale.is_file()
        metadata = json.loads(
            (
                integrated_analysis
                / ANALYSIS_ID
                / "reports"
                / "analysis_metadata.json"
            ).read_text(encoding="utf-8")
        )
        assert metadata.get("tracking_status") is None

    def test_metric_physics_and_replay_payload_are_calibrated(
        self,
        integrated_analysis: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_accepted_snapshot(integrated_analysis)
        calibration = load_physics_calibration(ANALYSIS_ID, 1280, 720)
        assert calibration.mode == "METRIC_3D"
        monkeypatch.setattr(
            "services.api.services.delivery_physics_service.validate_metric_geometry",
            lambda samples, observations, calibration: GeometryValidationResult(
                validity="VALID_METRIC_3D",
                mean_reprojection_px=1.0,
                median_reprojection_px=1.0,
                p95_reprojection_px=2.0,
                max_reprojection_px=3.0,
                in_pitch_fraction=1.0,
                threshold_px=25.0,
            ),
        )

        expected = np.array([0.1, 2.0, 1.8, 0.5, 28.0, 2.0])
        observations = synthetic_observations(expected, "BALLISTIC", count=10)
        track = [
            TrackingPoint(
                frame_index=observation.frame_index,
                timestamp_seconds=observation.timestamp_seconds,
                source="observed",
                provenance="OBSERVED",
                candidate_id=observation.candidate_id,
                x=observation.pixel_x,
                y=observation.pixel_y,
                normalized_x=observation.pixel_x / 1280.0,
                normalized_y=observation.pixel_y / 720.0,
                confidence=observation.detector_confidence,
                vx=100.0,
                vy=75.0,
            )
            for observation in observations
        ]
        detections = _detections_document(track)

        physics = analyse_delivery_physics(
            analysis_id=ANALYSIS_ID,
            primary_track=track,
            detections=detections,
            tracker_bounce=PrimaryBounceResult(
                bounce_detected=True,
                bounce_frame=track[8].frame_index,
                bounce_timestamp_seconds=track[8].timestamp_seconds,
                bounce_x=track[8].x,
                bounce_y=track[8].y,
                bounce_normalized_x=track[8].normalized_x,
                bounce_normalized_y=track[8].normalized_y,
                confidence=0.8,
                evidence=["test"],
                warnings=[],
            ),
            fps=25.0,
            width=1280,
            height=720,
            total_frames=120,
        )

        assert physics.calibration.mode == "METRIC_3D"
        assert physics.status in {"SUCCESS", "PARTIAL"}
        assert physics.trajectory_samples
        assert any(
            sample.world_x_m is not None and sample.world_y_m is not None
            for sample in physics.trajectory_samples
        )

        with patch.object(
            __import__(
                "services.api.services.replay_payload_service",
                fromlist=["replay_payload_service"],
            ),
            "load_analysis_camera_bridge",
            return_value=_wicket_box_bridge(),
        ):
            payload = build_and_save_replay_payload(
                ANALYSIS_ID,
                physics=physics,
                primary_track=track,
                finalized_track=build_finalized_delivery_track(
                    analysis_id=ANALYSIS_ID,
                    tracking_job_id="job_integration_test",
                    generated_at=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc),
                    primary_track=track,
                    physics=physics,
                    fps=25.0,
                    width=1280,
                    height=720,
                ),
                tracking_status="ready",
                fps=25.0,
                width=1280,
                height=720,
            )

        assert payload.measurement_validity == "CALIBRATED"
        assert payload.camera.source == "CALIBRATED"
        assert payload.trajectory
        assert payload.trajectory[0].world_position is not None
        assert payload.metrics.release_speed_kmh.status == "AVAILABLE"

        saved = integrated_analysis / ANALYSIS_ID / "tracking" / REPLAY_PAYLOAD_FILENAME
        assert saved.is_file()
        restored = ReplayPayloadV1.model_validate(
            json.loads(saved.read_text(encoding="utf-8"))
        )
        assert restored.measurement_validity == "CALIBRATED"

    def test_invalidate_tracking_helper_clears_outputs(
        self,
        integrated_analysis: Path,
    ) -> None:
        tracking_dir = integrated_analysis / ANALYSIS_ID / "tracking"
        stale = tracking_dir / REPLAY_PAYLOAD_FILENAME
        stale.write_text("{}", encoding="utf-8")
        invalidate_tracking_after_calibration_change(ANALYSIS_ID)
        assert not stale.is_file()
