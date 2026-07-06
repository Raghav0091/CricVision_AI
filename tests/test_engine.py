"""Lightweight tests for the reusable delivery-analysis engine."""

import importlib
import inspect
import sys

from Backends.src.engine.engine_options import EngineOptions
from Backends.src.engine.engine_result import EngineResult


def test_engine_modules_import_without_loading_models(monkeypatch):
    model_calls = []

    def fake_get_cached_yolo_model(model_key):
        model_calls.append(model_key)
        return None

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "Backends.src.models.model_loader.get_cached_yolo_model",
        fake_get_cached_yolo_model,
    )
    sys.modules.pop("Backends.src.ui.video_analysis", None)

    for module_name in (
        "Backends.src.engine.engine_options",
        "Backends.src.engine.engine_result",
        "Backends.src.engine.processors.batting",
        "Backends.src.engine.processors.delivery",
        "Backends.src.engine.processors",
        "Backends.src.engine.analyze_delivery",
        "Backends.src.engine",
    ):
        module = importlib.import_module(module_name)
        importlib.reload(module)
        assert module is not None

    assert model_calls == []
    assert "Backends.src.ui.video_analysis" not in sys.modules

    analyze_module = importlib.import_module(
        "Backends.src.engine.analyze_delivery"
    )
    assert "Backends.src.ui" not in inspect.getsource(analyze_module)


def test_engine_options_defaults_and_legacy_aliases():
    options = EngineOptions()

    assert options.analysis_mode == "Full Delivery Analysis"
    assert options.smart_mode == "Smart Balanced"
    assert options.processed_video_enabled is True
    assert options.overlay_detail == "Clean"
    assert options.confidence_threshold == 0.25

    aliased = EngineOptions.from_value(
        {
            "speed_mode": "Smart Accurate",
            "generate_processed_video": False,
            "confidence": 0.4,
            "imgsz": 960,
        }
    )
    assert aliased.smart_mode == "Smart Accurate"
    assert aliased.processed_video_enabled is False
    assert aliased.confidence_threshold == 0.4
    assert aliased.image_size == 960


def test_analyze_delivery_clip_handles_missing_video_without_models(
    tmp_path,
    monkeypatch,
):
    engine_module = importlib.import_module(
        "Backends.src.engine.analyze_delivery"
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("processor must not run for a missing path")

    monkeypatch.setattr(
        engine_module,
        "_run_processor",
        fail_if_called,
    )

    result = engine_module.analyze_delivery_clip(tmp_path / "missing.mp4")

    assert result["success"] is False
    assert "does not exist" in result["error"]
    assert result["errors"]
    assert result["report_result"]["impact_report"] == {}
    assert result["processed_video_path"] is None


def test_engine_result_normalizes_missing_optional_sections():
    result = EngineResult.from_pipeline_result({"success": True}).to_dict()

    assert result["success"] is True
    assert result["report_result"]["delivery_report"] == {}
    assert result["processed_video_validation"] == {}
    assert result["calibration_context"]["enabled"] is False
    assert result["visual_observer_summary"] == {}
    assert result["delivery_report"] == {}
    assert result["impact_report"] == {}
    assert result["shot_report"] == {}
    assert result["outcome_prediction"] == {}
    assert result["timings"] == {}
    assert result["warnings"] == []
    assert result["errors"] == []


def test_analyze_delivery_clip_dispatches_without_ui_or_models(
    tmp_path,
    monkeypatch,
):
    engine_module = importlib.import_module(
        "Backends.src.engine.analyze_delivery"
    )
    video_path = tmp_path / "delivery.mp4"
    video_path.write_bytes(b"synthetic placeholder")
    calls = []

    class FakeCapture:
        def release(self):
            return None

    def fake_processor(path, context, options, output_path):
        calls.append((path, context, options, output_path))
        return {
            "success": True,
            "processed_video_generated": False,
            "impact_info": {"impact_detected": False},
        }

    monkeypatch.setattr(engine_module, "open_video", lambda path: FakeCapture())
    monkeypatch.setattr(engine_module, "_run_processor", fake_processor)

    result = engine_module.analyze_delivery_clip(
        video_path,
        options={"generate_processed_video": False},
    )

    assert result["success"] is True
    assert len(calls) == 1
    assert calls[0][0] == video_path
    assert result["processed_video_path"] is None
    assert result["impact_report"] == {"impact_detected": False}


def test_video_analysis_ui_does_not_define_processors():
    video_analysis = importlib.import_module(
        "Backends.src.ui.video_analysis"
    )
    source = inspect.getsource(video_analysis)

    assert "def process_video(" not in source
    assert "def process_batting_video(" not in source
    assert "analyze_delivery_clip(" in source
