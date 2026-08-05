from types import SimpleNamespace

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.routes import video_analysis as routes
from services.api.schemas.video_analysis import (
    TrackingPoint,
    VideoBallTrackingStartRequest,
)
from services.api.services import video_ball_tracking_service as tracking
from services.api.services.video_ball_tracking_job_store import (
    VideoBallTrackingJobStore,
)


ANALYSIS_ID = "analysis_ball_pipeline_test"


def test_tracking_point_normalises_legacy_native_pixel_fields() -> None:
    point = TrackingPoint(
        frame_index=12,
        timestamp_seconds=0.48,
        source="observed",
        provenance="OBSERVED",
        candidate_id="frame_12_ball_0",
        x=321.5,
        y=456.25,
        normalized_x=0.502344,
        normalized_y=0.633681,
        confidence=0.82,
        vx=0.0,
        vy=0.0,
    )

    assert point.image_x_px == point.x
    assert point.image_y_px == point.y
    assert point.detector_confidence == 0.82
    assert point.tracking_confidence == 0.82
    assert point.valid is True


def test_complete_delivery_v2_points_preserve_boxes_and_recovery_provenance() -> None:
    first = tracking.RawTrackingCandidate(
        frame_index=10,
        timestamp_seconds=0.4,
        candidate_id="first",
        confidence=0.9,
        bounding_box=[95.0, 195.0, 105.0, 205.0],
        x=100.0,
        y=200.0,
        normalized_x=0.15625,
        normalized_y=0.416667,
        width_pixels=10.0,
        height_pixels=10.0,
        area_pixels=100.0,
        inside_pitch_corridor=True,
    )
    second = tracking.RawTrackingCandidate(
        frame_index=12,
        timestamp_seconds=0.48,
        candidate_id="second",
        confidence=0.8,
        bounding_box=[115.0, 205.0, 125.0, 215.0],
        x=120.0,
        y=210.0,
        normalized_x=0.1875,
        normalized_y=0.4375,
        width_pixels=10.0,
        height_pixels=10.0,
        area_pixels=100.0,
        inside_pitch_corridor=True,
    )

    points = tracking._build_tracking_points(
        tracking.Tracklet(observations=[first, second]),
        fps=25.0,
    )

    observed = points[0]
    recovered = points[1]
    assert observed.bounding_box == first.bounding_box
    assert observed.detector_confidence == first.confidence
    assert recovered.provenance == "TRACKER_RECOVERED"
    assert recovered.bounding_box is None
    assert recovered.detector_confidence == 0.0
    assert recovered.tracking_confidence == recovered.confidence
    assert recovered.image_x_px == recovered.x


def test_tracker_delivery_analysis_flag_is_opt_in_compatible(
    monkeypatch,
) -> None:
    calls: list[bool] = []

    def fail_after_capture(
        analysis_id: str,
        job_id: str,
        *,
        include_delivery_analysis: bool,
    ):
        calls.append(include_delivery_analysis)
        raise tracking.VideoBallTrackingError("stop")

    monkeypatch.setattr(tracking, "_process_video_ball_tracking", fail_after_capture)
    monkeypatch.setattr(tracking, "_mark_job_failed", lambda *args, **kwargs: None)

    tracking.run_video_ball_tracking_job(
        ANALYSIS_ID,
        "tracking_without_delivery_analysis",
        include_delivery_analysis=False,
    )
    tracking.run_video_ball_tracking_job(
        ANALYSIS_ID,
        "legacy_tracking_default",
    )

    assert calls == [False, True]


def test_scoped_tracker_skips_bounce_physics_and_delivery_replay(
    tmp_path,
    monkeypatch,
) -> None:
    analysis = SimpleNamespace(
        fps=25.0,
        frame_count=10,
        width=640,
        height=360,
        stored_filename="source.mp4",
    )
    monkeypatch.setattr(tracking, "load_video_analysis", lambda _: analysis)
    monkeypatch.setattr(tracking, "validate_video_ball_tracking_input", lambda _: None)
    monkeypatch.setattr(tracking, "_tracking_output_dir", lambda _: tmp_path)
    monkeypatch.setattr(tracking, "_clear_previous_tracking_outputs", lambda _: None)
    monkeypatch.setattr(tracking, "_update_analysis_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(tracking, "_update_job", lambda *args, **kwargs: None)
    monkeypatch.setattr(tracking, "_load_detection_document", lambda *args: object())
    monkeypatch.setattr(tracking, "_flatten_candidates", lambda _: [])
    monkeypatch.setattr(tracking, "_assign_static_likelihoods", lambda _: None)
    monkeypatch.setattr(tracking, "_build_primary_track", lambda _: (None, {}, None))
    monkeypatch.setattr(
        tracking,
        "_render_tracking_video",
        lambda **kwargs: tmp_path / tracking.TRACKING_VIDEO_FILENAME,
    )
    monkeypatch.setattr(tracking, "_verify_output_video", lambda _: (10, 25.0))
    monkeypatch.setattr(
        tracking,
        "_detect_primary_bounce",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("bounce analysis must be disabled")
        ),
    )
    monkeypatch.setattr(
        tracking,
        "analyse_delivery_physics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("physics must be disabled")
        ),
    )
    monkeypatch.setattr(
        tracking,
        "_render_delivery_replay",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("delivery replay must be disabled")
        ),
    )

    summary, primary_track = tracking._process_video_ball_tracking(
        ANALYSIS_ID,
        "scoped_tracking_job",
        include_delivery_analysis=False,
    )

    assert summary.status == "no_reliable_track"
    assert summary.physics_result_url is None
    assert summary.physics_engine_version is None
    assert summary.physics_status is None
    assert summary.delivery_replay_url is None
    assert primary_track == []
    assert not (tmp_path / tracking.PHYSICS_RESULT_FILENAME).exists()


def test_existing_tracking_start_endpoint_queues_scoped_tracker(
    monkeypatch,
) -> None:
    job_store = VideoBallTrackingJobStore()
    monkeypatch.setattr(routes, "validate_video_ball_tracking_input", lambda _: None)
    monkeypatch.setattr(routes, "video_ball_tracking_job_store", job_store)
    monkeypatch.setattr(routes, "mark_video_ball_tracking_queued", lambda *args: None)
    background_tasks = BackgroundTasks()

    response = routes.start_analysis_ball_tracking(
        ANALYSIS_ID,
        background_tasks,
        VideoBallTrackingStartRequest(include_delivery_analysis=False),
    )

    assert response.status == "queued"
    assert len(background_tasks.tasks) == 1
    task = background_tasks.tasks[0]
    assert task.func is routes.run_video_ball_tracking_job
    assert task.kwargs == {"include_delivery_analysis": False}


def test_tracking_start_request_preserves_legacy_delivery_analysis_default() -> None:
    assert VideoBallTrackingStartRequest().include_delivery_analysis is True


def test_tracking_result_response_includes_review_diagnostics(
    tmp_path,
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    from services.api.schemas.video_analysis import (
        TrackingCandidateDiagnostic,
        TrackingCandidateScoreComponents,
        TrackingPoint,
        VideoBallTrackingDocument,
        VideoBallTrackingSummary,
    )
    from services.api.services.video_ball_tracking_service import (
        TRACKING_CSV_FILENAME,
        TRACKING_RESULT_FILENAME,
        TRACKING_SUMMARY_FILENAME,
        TRACKING_VIDEO_FILENAME,
        load_video_ball_tracking_result,
    )

    analysis_id = "analysis_review_diagnostics"
    monkeypatch.setattr(
        tracking,
        "load_video_analysis",
        lambda _: SimpleNamespace(frame_count=10),
    )
    monkeypatch.setattr(tracking, "_tracking_output_dir", lambda _: tmp_path)

    summary = VideoBallTrackingSummary(
        analysis_id=analysis_id,
        status="ready",
        total_video_frames=10,
        raw_candidate_count=2,
        candidate_frames=2,
        track_start_frame=1,
        track_end_frame=2,
        track_duration_frames=2,
        track_duration_seconds=0.08,
        observed_track_points=1,
        predicted_points=0,
        recovered_points=1,
        rejected_candidates=1,
        longest_gap_frames=0,
        average_observed_confidence=0.8,
        track_confidence=0.7,
        track_quality="medium",
        approximate_direction="down",
        possible_bounce_transition_detected=False,
        tracking_video_url="/static/video-analysis/x/tracking/tracking_debug.mp4",
        tracking_json_url="/static/video-analysis/x/tracking/tracking_result.json",
        tracking_csv_url="/static/video-analysis/x/tracking/tracking.csv",
        tracking_summary_url="/static/video-analysis/x/tracking/tracking_summary.json",
        processing_duration_seconds=1.0,
        message="Complete Delivery Tracking v2 completed.",
    )
    document = VideoBallTrackingDocument(
        analysis_id=analysis_id,
        status="ready",
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        settings={
            "motion_model": "constant_velocity_recent_median",
            "max_recoverable_gap": 3,
            "minimum_observed_points": 4,
            "static_radius_normalized": 0.02,
            "base_gate_normalized": 0.08,
            "maximum_gate_normalized": 0.2,
            "history_points": 4,
            "beam_width": 4,
            "tracker_version": "delivery_track_v2",
        },
        primary_track=[
            TrackingPoint(
                frame_index=1,
                timestamp_seconds=0.04,
                source="observed",
                provenance="OBSERVED",
                x=10.0,
                y=20.0,
                normalized_x=0.1,
                normalized_y=0.2,
                confidence=0.8,
                vx=0.0,
                vy=0.0,
            )
        ],
        raw_primary_track=[],
        candidate_diagnostics=[
            TrackingCandidateDiagnostic(
                frame_index=1,
                candidate_id="frame_000001_candidate_001",
                selected=True,
                selection_reason="Selected as a coherent primary-track observation.",
                static_likelihood=0.1,
                score_components=TrackingCandidateScoreComponents(
                    detector_confidence=0.8,
                    motion=0.5,
                    prediction_proximity=0.4,
                    direction=0.3,
                    size_consistency=0.2,
                    corridor=0.1,
                    static_penalty=0.0,
                    jump_penalty=0.0,
                    total=0.7,
                ),
            )
        ],
        message="Complete Delivery Tracking v2 completed.",
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / TRACKING_SUMMARY_FILENAME).write_text(
        summary.model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / TRACKING_RESULT_FILENAME).write_text(
        document.model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / TRACKING_CSV_FILENAME).write_text("frame_index\n", encoding="utf-8")
    (tmp_path / TRACKING_VIDEO_FILENAME).write_bytes(b"\x00")

    result = load_video_ball_tracking_result(analysis_id)

    assert len(result.candidate_diagnostics) == 1
    assert result.candidate_diagnostics[0].candidate_id == "frame_000001_candidate_001"
    assert result.raw_primary_track == []


def test_detection_result_can_include_frame_records(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timezone

    from services.api.schemas.video_analysis import (
        BallCandidate,
        FrameDetectionRecord,
        VideoBallDetectionsDocument,
        VideoBallDetectionSettings,
        VideoBallDetectionSummary,
    )
    from services.api.services.video_ball_detection_service import (
        DETECTIONS_CSV_FILENAME,
        DETECTIONS_JSON_FILENAME,
        DETECTION_OVERLAY_FILENAME,
        DETECTION_SUMMARY_FILENAME,
        load_video_ball_detection_result,
    )
    from services.api.services import video_ball_detection_service as detection

    analysis_id = "analysis_detection_frames"
    monkeypatch.setattr(detection, "load_video_analysis", lambda _: object())
    monkeypatch.setattr(detection, "_detection_output_dir", lambda _: tmp_path)

    summary = VideoBallDetectionSummary(
        analysis_id=analysis_id,
        status="ready",
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        original_video_url="/static/video-analysis/x/raw/source.mp4",
        processed_video_url="/static/video-analysis/x/detection/detection_overlay.mp4",
        detections_json_url="/static/video-analysis/x/detection/detections.json",
        detections_csv_url="/static/video-analysis/x/detection/detections.csv",
        detection_summary_url="/static/video-analysis/x/detection/detection_summary.json",
        model_path_used="models/ball.pt",
        model_class_names=["ball"],
        device_used="cpu",
        imgsz=960,
        confidence_threshold=0.15,
        frame_stride=1,
        max_det=20,
        total_frames=1,
        frames_processed=1,
        frames_with_candidates=1,
        frames_without_candidates=0,
        total_candidates=1,
        frames_with_multiple_candidates=0,
        candidates_inside_pitch_corridor=1,
        candidates_outside_pitch_corridor=0,
        candidates_without_corridor_information=0,
        best_confidence=0.9,
        average_confidence=0.9,
        average_candidates_per_detected_frame=1.0,
        processing_duration_seconds=1.0,
        output_video_frame_count=1,
        input_fps=25.0,
        output_fps=25.0,
        input_duration_seconds=0.04,
        output_duration_seconds=0.04,
        message="Every-frame ball detection completed.",
    )
    document = VideoBallDetectionsDocument(
        analysis_id=analysis_id,
        model_path_used="models/ball.pt",
        model_class_names=["ball"],
        settings=VideoBallDetectionSettings(
            frame_stride=1,
            imgsz=960,
            confidence_threshold=0.15,
            max_det=20,
        ),
        frames=[
            FrameDetectionRecord(
                frame_index=0,
                timestamp_seconds=0.0,
                processed=True,
                detections=[
                    BallCandidate(
                        candidate_id="frame_000000_candidate_001",
                        class_id=0,
                        class_name="ball",
                        confidence=0.9,
                        bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
                        bbox_normalized={
                            "x": 0.1,
                            "y": 0.2,
                            "width": 0.01,
                            "height": 0.01,
                        },
                        center={"x": 2.0, "y": 3.0},
                        center_normalized={"x": 0.15, "y": 0.25},
                        width_pixels=2.0,
                        height_pixels=2.0,
                        area_pixels=4.0,
                        inside_pitch_corridor=True,
                    )
                ],
            )
        ],
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / DETECTION_SUMMARY_FILENAME).write_text(
        summary.model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / DETECTIONS_JSON_FILENAME).write_text(
        document.model_dump_json(),
        encoding="utf-8",
    )
    (tmp_path / DETECTIONS_CSV_FILENAME).write_text("frame_index\n", encoding="utf-8")
    (tmp_path / DETECTION_OVERLAY_FILENAME).write_bytes(b"\x00")

    without_frames = load_video_ball_detection_result(analysis_id)
    with_frames = load_video_ball_detection_result(analysis_id, include_frames=True)

    assert without_frames.frames is None
    assert with_frames.frames is not None
    assert with_frames.frames[0].detections[0].candidate_id == "frame_000000_candidate_001"


def test_tracking_start_http_contract_preserves_default_and_accepts_scoped_mode(
    monkeypatch,
) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(routes, "validate_video_ball_tracking_input", lambda _: None)
    monkeypatch.setattr(
        routes,
        "video_ball_tracking_job_store",
        VideoBallTrackingJobStore(),
    )
    monkeypatch.setattr(routes, "mark_video_ball_tracking_queued", lambda *args: None)
    monkeypatch.setattr(
        routes,
        "run_video_ball_tracking_job",
        lambda analysis_id, job_id, *, include_delivery_analysis: calls.append(
            (analysis_id, include_delivery_analysis)
        ),
    )
    client = TestClient(app)

    legacy = client.post(
        "/video-analysis/analysis_legacy/tracking/start",
    )
    scoped = client.post(
        "/video-analysis/analysis_scoped/tracking/start",
        json={"include_delivery_analysis": False},
    )

    assert legacy.status_code == 202
    assert scoped.status_code == 202
    assert calls == [
        ("analysis_legacy", True),
        ("analysis_scoped", False),
    ]
