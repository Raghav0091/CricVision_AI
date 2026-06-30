"""Video pipeline modules import without triggering model loading."""

import importlib
import inspect


PIPELINE_MODULES = [
    "Backends.src.video_pipeline.video_reader",
    "Backends.src.video_pipeline.detection_pipeline",
    "Backends.src.video_pipeline.report_pipeline",
    "Backends.src.video_pipeline.annotation_writer",
    "Backends.src.video_pipeline.performance_timer",
]


def test_video_pipeline_imports_do_not_load_models(monkeypatch):
    calls = []

    def fake_get_cached_yolo_model(model_key):
        calls.append(model_key)
        return None

    monkeypatch.setattr(
        "Backends.src.models.model_loader.get_cached_yolo_model",
        fake_get_cached_yolo_model,
    )

    for module_name in PIPELINE_MODULES:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        assert module is not None

    assert calls == []


def test_ui_modules_no_longer_cross_import(monkeypatch):
    calls = []

    def fake_get_cached_yolo_model(model_key):
        calls.append(model_key)
        return None

    monkeypatch.setattr(
        "Backends.src.models.model_loader.get_cached_yolo_model",
        fake_get_cached_yolo_model,
    )

    video_analysis = importlib.import_module("Backends.src.ui.video_analysis")
    live_session = importlib.import_module("Backends.src.ui.live_session")
    importlib.reload(video_analysis)
    importlib.reload(live_session)

    assert calls == []
    assert "Backends.src.ui.video_analysis" not in inspect.getsource(live_session)
