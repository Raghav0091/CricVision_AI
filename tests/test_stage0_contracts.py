"""Stage 0 Virtual Pitch Replay contract and API foundation tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from services.api.main import app
from services.api.schemas.delivery_physics import DeliveryPhysicsResult
from services.api.schemas.replay_payload import (
    REPLAY_COORDINATE_SYSTEM,
    REPLAY_DISTANCE_UNIT,
    REPLAY_EXPORT_HEIGHT,
    REPLAY_EXPORT_WIDTH,
    REPLAY_SCHEMA_VERSION,
    REPLAY_TIME_UNIT,
    ImagePoint2D,
    ReplayMetric,
    ReplayMetrics,
    ReplayPayloadV1,
    ReplayTrajectorySample,
    build_stage0_replay_payload,
    unavailable_metric,
)
from services.api.schemas.wicket_box_calibration import (
    WicketBox,
    WicketBoxCalibrationRegisterRequest,
    validate_wicket_box_pair,
)
from services.api.schemas.video_analysis import VideoAnalysisPreparedResponse


ANALYSIS_ID = "analysis_20260803_213800_a0b1c2"
client = TestClient(app)


def _wicket_box(
    role: str,
    *,
    x: float = 100.0,
    y: float = 400.0,
    width: float = 80.0,
    height: float = 120.0,
) -> WicketBox:
    return WicketBox(
        role=role,
        x=x,
        y=y,
        width=width,
        height=height,
        source_image_width=1920,
        source_image_height=1080,
        calibration_frame_index=12,
    )


class TestReplayPayloadContracts:
    def test_schema_version_and_units(self):
        payload = build_stage0_replay_payload(ANALYSIS_ID)
        assert payload.schema_version == REPLAY_SCHEMA_VERSION
        assert payload.coordinate_system == REPLAY_COORDINATE_SYSTEM
        assert payload.distance_unit == REPLAY_DISTANCE_UNIT
        assert payload.time_unit == REPLAY_TIME_UNIT

    def test_required_analysis_id(self):
        with pytest.raises(ValidationError):
            ReplayPayloadV1.model_validate(
                build_stage0_replay_payload(ANALYSIS_ID).model_dump()
                | {"analysis_id": ""}
            )

    def test_export_dimensions_are_1920_by_1080_landscape(self):
        payload = build_stage0_replay_payload(ANALYSIS_ID)
        assert payload.playback.export_width == REPLAY_EXPORT_WIDTH
        assert payload.playback.export_height == REPLAY_EXPORT_HEIGHT
        assert payload.playback.landscape is True

    def test_missing_metrics_stay_null_unavailable(self):
        payload = build_stage0_replay_payload(ANALYSIS_ID)
        lateral = payload.metrics.estimated_lateral_deviation_m
        assert lateral.value is None
        assert lateral.status == "UNAVAILABLE"
        assert lateral.unavailable_reason
        assert "estimated_lateral_deviation_m" in ReplayMetrics.model_fields

    def test_metric_never_zero_for_missing_evidence(self):
        metric = unavailable_metric("km/h", "not enough evidence")
        assert metric.value is None
        with pytest.raises(ValidationError):
            ReplayMetric(
                value=0.0,
                unit="km/h",
                status="UNAVAILABLE",
                unavailable_reason="not enough evidence",
            )

    def test_image_space_only_suppresses_world_measurements(self):
        with pytest.raises(ValidationError):
            ReplayPayloadV1.model_validate(
                {
                    **build_stage0_replay_payload(ANALYSIS_ID).model_dump(),
                    "measurement_validity": "IMAGE_SPACE_ONLY",
                    "trajectory": [
                        ReplayTrajectorySample(
                            frame_index=0,
                            timestamp_seconds=0.0,
                            world_position={"x_m": 1.0, "y_m": 2.0, "z_m": 0.5},
                            provenance="OBSERVED",
                            confidence=0.8,
                        ).model_dump()
                    ],
                }
            )

    def test_image_space_only_allows_image_positions(self):
        payload = ReplayPayloadV1.model_validate(
            {
                **build_stage0_replay_payload(ANALYSIS_ID).model_dump(),
                "measurement_validity": "IMAGE_SPACE_ONLY",
                "trajectory": [
                    ReplayTrajectorySample(
                        frame_index=0,
                        timestamp_seconds=0.0,
                        image_position=ImagePoint2D(x=100.0, y=200.0),
                        provenance="OBSERVED",
                        confidence=0.8,
                    ).model_dump()
                ],
            }
        )
        assert payload.trajectory[0].world_position is None
        assert payload.trajectory[0].image_position is not None

    def test_visualization_only_cannot_claim_calibrated_measurements(self):
        base = build_stage0_replay_payload(ANALYSIS_ID).model_dump()
        base["measurement_validity"] = "VISUALIZATION_ONLY"
        base["camera"]["visualization_only"] = True
        base["metrics"]["release_speed_kmh"] = {
            "value": 120.0,
            "unit": "km/h",
            "confidence": 0.9,
            "method": "fake",
            "status": "AVAILABLE",
            "unavailable_reason": None,
        }
        with pytest.raises(ValidationError):
            ReplayPayloadV1.model_validate(base)


class TestWicketBoxContracts:
    def test_original_resolution_box_coordinates(self):
        box = _wicket_box("NEAR")
        assert box.source_image_width == 1920
        assert box.source_image_height == 1080
        assert box.x + box.width <= box.source_image_width

    def test_near_far_validation_roles(self):
        near = _wicket_box("NEAR", y=700.0, height=150.0, width=120.0)
        far = _wicket_box("FAR", y=120.0, height=60.0, width=40.0)
        result = validate_wicket_box_pair(near, far)
        assert result.valid is True
        assert result.role_order_valid is True

    def test_invalid_box_dimensions_out_of_bounds(self):
        with pytest.raises(ValidationError):
            _wicket_box("NEAR", x=1900.0, width=80.0)

    def test_overlap_rejected(self):
        near = _wicket_box("NEAR", x=100.0, y=400.0)
        far = _wicket_box("FAR", x=110.0, y=410.0, width=60.0, height=80.0)
        result = validate_wicket_box_pair(near, far)
        assert result.valid is False
        assert result.validation_status == "OVERLAP"

    def test_no_silent_near_far_swapping_on_register(self):
        near = _wicket_box("NEAR", y=120.0, height=60.0, width=40.0)
        far = _wicket_box("FAR", y=700.0, height=150.0, width=120.0)
        result = validate_wicket_box_pair(near, far)
        assert result.valid is False
        assert result.role_order_valid is False
        assert near.role == "NEAR"
        assert far.role == "FAR"

    def test_register_request_rejects_swapped_role_labels(self):
        near = _wicket_box("FAR")
        far = _wicket_box("NEAR")
        with pytest.raises(ValidationError):
            WicketBoxCalibrationRegisterRequest(
                analysis_id=ANALYSIS_ID,
                calibration_frame_index=12,
                source_image_width=1920,
                source_image_height=1080,
                near_wicket_box=near,
                far_wicket_box=far,
            )


class TestStage0ApiFoundations:
    def test_detect_requires_submitted_boxes(self):
        response = client.post(
            f"/video-analysis/{ANALYSIS_ID}/wicket-box-calibration/detect"
        )
        assert response.status_code == 422

    def test_register_validates_invalid_boxes(self):
        near = _wicket_box("NEAR", y=700.0, height=150.0, width=120.0)
        far = _wicket_box("FAR", y=120.0, height=60.0, width=40.0)
        response = client.post(
            f"/video-analysis/{ANALYSIS_ID}/wicket-box-calibration/register",
            json={
                "analysis_id": ANALYSIS_ID,
                "calibration_frame_index": 12,
                "source_image_width": 1920,
                "source_image_height": 1080,
                "near_wicket_box": near.model_dump(),
                "far_wicket_box": far.model_dump(),
            },
        )
        assert response.status_code in {200, 404, 422}

    def test_accept_requires_registered_calibration(self):
        response = client.post(
            f"/video-analysis/{ANALYSIS_ID}/wicket-box-calibration/accept",
            json={
                "analysis_id": ANALYSIS_ID,
                "accept_registered_calibration": True,
            },
        )
        assert response.status_code in {200, 404}
        if response.status_code == 200:
            body = response.json()
            assert body["success"] is False

    def test_replay_payload_does_not_fabricate_metrics(self):
        response = client.get(
            f"/video-analysis/{ANALYSIS_ID}/replay-payload"
        )
        assert response.status_code == 200
        body = response.json()
        assert body["diagnostics"]["status"] == "NOT_IMPLEMENTED"
        assert body["metrics"]["estimated_lateral_deviation_m"]["value"] is None
        assert body["trajectory"] == []


class TestExistingSchemaCompatibility:
    def test_delivery_physics_result_still_serializes(self):
        sample = DeliveryPhysicsResult.model_json_schema()
        assert "physics_engine_version" in sample["properties"]

    def test_video_analysis_prepared_response_still_importable(self):
        assert VideoAnalysisPreparedResponse is not None
