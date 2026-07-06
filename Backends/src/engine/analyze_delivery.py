"""Framework-neutral entrypoint for analysis of one delivery clip."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from Backends.src.calibration.calibration_context import (
    normalize_calibration_context,
)
from Backends.src.config.paths import PROCESSED_VIDEO_DIR
from Backends.src.engine.engine_options import EngineOptions
from Backends.src.engine.engine_result import EngineResult
from Backends.src.video_pipeline.video_reader import open_video


def analyze_delivery_clip(video_path, calibration_context=None, options=None):
    """Analyze one delivery clip and return a stable, backward-compatible dict.

    Model loading remains lazy: the input is checked before the established
    processor is imported or called.
    """
    try:
        engine_options = EngineOptions.from_value(options)
    except (TypeError, ValueError) as error:
        return EngineResult.failure(
            str(error),
            calibration_context=calibration_context,
        ).to_dict()

    context = normalize_calibration_context(calibration_context)
    input_path = _input_path(video_path)
    if input_path is None:
        return EngineResult.failure(
            "A video path is required.",
            calibration_context=context,
        ).to_dict()
    if not input_path.is_file():
        return EngineResult.failure(
            f"Video path does not exist: {input_path}",
            calibration_context=context,
        ).to_dict()

    capture = open_video(input_path)
    if capture is None:
        return EngineResult.failure(
            "Could not open the video. Check that it uses a supported codec.",
            calibration_context=context,
        ).to_dict()
    capture.release()

    raw_output_path, browser_output_path = _output_paths(engine_options)
    try:
        pipeline_result = _run_processor(
            input_path,
            context,
            engine_options,
            raw_output_path,
        )
        if not isinstance(pipeline_result, dict):
            pipeline_result = {
                "success": False,
                "error": "Delivery processor returned an invalid result.",
            }

        pipeline_result.setdefault(
            "analysis_mode",
            engine_options.analysis_mode,
        )
        pipeline_result.setdefault(
            "active_preset",
            engine_options.active_preset,
        )
        pipeline_result.setdefault(
            "active_model",
            engine_options.model_name or engine_options.model_key,
        )
        if engine_options.model_name:
            pipeline_result["ball_model_used"] = engine_options.model_name
        else:
            pipeline_result.setdefault(
                "ball_model_used",
                engine_options.ball_model_key or engine_options.model_key,
            )
        pipeline_result["show_performance_details"] = (
            engine_options.show_performance_details
        )

        if pipeline_result.get("success"):
            _finalize_processed_video(
                pipeline_result,
                raw_output_path,
                browser_output_path,
            )
            _add_analysis_warnings(pipeline_result)

        return EngineResult.from_pipeline_result(
            pipeline_result,
            calibration_context=context,
        ).to_dict()
    except Exception as error:
        return EngineResult.failure(
            f"Video analysis failed: {type(error).__name__}: {error}",
            calibration_context=context,
        ).to_dict()


def _run_processor(
    video_path,
    calibration_context,
    options,
    output_path,
):
    """Dispatch to the engine-owned processor for the selected mode."""

    if options.analysis_mode == "Batting Analysis":
        from Backends.src.engine.processors.batting import (
            process_batting_video,
        )

        return process_batting_video(
            video_path=video_path,
            output_path=output_path,
            ball_model_key=options.ball_model_key or options.model_key,
            bat_model_key=options.bat_model_key or "cricshot_bat",
            confidence=options.confidence_threshold,
            speed_mode=options.smart_mode,
            max_frames=options.max_frames,
            generate_processed_video=options.processed_video_enabled,
            calibration_context=calibration_context,
            overlay_detail=options.overlay_detail,
            progress_callback=options.progress_callback,
        )

    from Backends.src.engine.processors.delivery import (
        process_delivery_video,
    )

    model_path = options.model_path
    if model_path is None and options.model_key:
        from Backends.src.models.model_registry import get_model_path

        model_path = get_model_path(options.model_key)

    bat_model_key = options.bat_model_key
    if (
        bat_model_key is None
        and options.analysis_mode == "Full Delivery Analysis"
    ):
        bat_model_key = "cricshot_bat"

    return process_delivery_video(
        video_path=video_path,
        output_path=output_path,
        model_path=model_path,
        model_key=options.model_key,
        class_names=options.class_names,
        confidence=options.confidence_threshold,
        imgsz=options.image_size,
        use_ensemble=options.use_ensemble,
        show_pitch_roi=options.show_pitch_roi,
        calibration_mode=options.calibration_mode,
        manual_pitch_points=options.manual_pitch_points,
        shot_trajectory_mode=options.shot_trajectory_mode,
        manual_contact_frame=options.manual_contact_frame,
        field_setup=options.field_setup,
        bat_model_key=bat_model_key,
        speed_mode=options.smart_mode,
        max_frames=options.max_frames,
        generate_processed_video=options.processed_video_enabled,
        calibration_context=calibration_context,
        overlay_detail=options.overlay_detail,
        progress_callback=options.progress_callback,
    )


def _finalize_processed_video(
    result,
    raw_output_path,
    browser_output_path,
):
    from Backends.src.video_pipeline.annotation_writer import (
        convert_to_browser_mp4,
        validate_processed_video_path,
    )

    generated = bool(result.get("processed_video_generated"))
    raw_path = Path(result.get("output_path") or raw_output_path)
    if not generated:
        result["output_path"] = None
        result["processed_video_path"] = None
        result["raw_output_path"] = None
        result["processed_video_validation"] = {}
        return

    result["raw_output_path"] = str(raw_path) if raw_path.is_file() else None
    final_path = raw_path
    try:
        final_path = convert_to_browser_mp4(
            input_path=raw_path,
            output_path=browser_output_path,
        )
        result["processed_video_conversion"] = "converted"
    except Exception as error:
        result["processed_video_conversion"] = "failed"
        result["processed_video_conversion_error"] = str(error)
        _append_warning(
            result,
            "Processed video preview conversion failed; the raw video remains available.",
        )

    result["output_path"] = str(final_path) if final_path.is_file() else None
    result["processed_video_path"] = result["output_path"]
    result["processed_video_validation"] = validate_processed_video_path(
        final_path
    )


def _add_analysis_warnings(result):
    try:
        from Backends.src.analysis.cricket_agent import (
            detect_analysis_warnings,
        )

        for warning in detect_analysis_warnings(result):
            _append_warning(result, warning)
    except Exception:
        # Warnings are supplementary and must not invalidate a completed run.
        return


def _append_warning(result, message):
    warnings = result.setdefault("warnings", [])
    if isinstance(warnings, str):
        warnings = [warnings]
        result["warnings"] = warnings
    if message not in warnings:
        warnings.append(message)


def _output_paths(options):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    raw_path = (
        Path(options.output_path)
        if options.output_path
        else PROCESSED_VIDEO_DIR
        / f"raw_cricvision_analysis_{timestamp}.mp4"
    )
    browser_path = (
        Path(options.browser_output_path)
        if options.browser_output_path
        else PROCESSED_VIDEO_DIR / f"cricvision_analysis_{timestamp}.mp4"
    )
    if options.processed_video_enabled:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        browser_path.parent.mkdir(parents=True, exist_ok=True)
    return raw_path, browser_path


def _input_path(video_path):
    if video_path is None:
        return None
    try:
        return Path(video_path).expanduser()
    except TypeError:
        return None
