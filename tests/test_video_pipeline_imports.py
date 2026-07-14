"""Video pipeline modules import without triggering model loading."""

import importlib
import sys


PIPELINE_MODULES = [
    "Backends.src.video_pipeline.video_reader",
    "Backends.src.video_pipeline.detection_pipeline",
    "Backends.src.video_pipeline.report_pipeline",
    "Backends.src.video_pipeline.annotation_writer",
    "Backends.src.video_pipeline.performance_timer",
]


def test_video_pipeline_imports_do_not_load_models(monkeypatch):
    model_calls = []
    download_calls = []

    def fake_get_cached_yolo_model(model_key):
        model_calls.append(model_key)
        return None

    def fake_download_remote_model(model_key, force_download=False):
        download_calls.append((model_key, force_download))
        return None

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(
        "Backends.src.models.model_loader.get_cached_yolo_model",
        fake_get_cached_yolo_model,
    )
    monkeypatch.setattr(
        "Backends.src.models.model_loader.download_remote_model",
        fake_download_remote_model,
    )
    modules_before = set(sys.modules)

    for module_name in PIPELINE_MODULES:
        module = importlib.import_module(module_name)
        importlib.reload(module)
        assert module is not None

    imported_during_test = set(sys.modules) - modules_before
    forbidden_roots = {"ultralytics", "tensorflow", "keras", "huggingface_hub"}
    assert not {
        name
        for name in imported_during_test
        if name.split(".", 1)[0] in forbidden_roots
    }
    assert model_calls == []
    assert download_calls == []


def test_detection_pipeline_imports_without_streamlit_runtime(monkeypatch):
    module_name = "Backends.src.video_pipeline.detection_pipeline"
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "streamlit", None)

    module = importlib.import_module(module_name)

    assert module.st is None


def test_ui_module_imports_do_not_load_models(monkeypatch):
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
