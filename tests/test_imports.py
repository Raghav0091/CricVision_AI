"""Import smoke tests — modules should import without loading YOLO/Keras weights."""

import importlib
import sys

import pytest


MODULES = [
    "Backends.src.config.constants",
    "Backends.src.config.paths",
    "Backends.src.ui.dashboard",
    "Backends.src.ui.results_page",
    "Backends.src.ui.analysis_helpers",
    "Backends.src.models.model_registry",
    "Backends.src.models.model_loader",
    "Backends.src.analysis.frame_detection_utils",
    "Backends.src.agents.observer_timeline",
    "Backends.src.agents.tracking_repair_agent",
    "Backends.src.agents.visual_observer_agent",
    "Backends.src.analysis.impact_detection",
    "Backends.src.analysis.shot_direction",
    "Backends.src.analysis.outcome_prediction",
    "Backends.src.agents.vision_agent",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_core_module_imports(module_name):
    if module_name in sys.modules:
        module = sys.modules[module_name]
        importlib.reload(module)
    else:
        module = importlib.import_module(module_name)
    assert module is not None


def test_video_analysis_imports_without_model_load(monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    calls = []

    def fake_get_cached_yolo_model(model_key):
        calls.append(model_key)
        return None

    monkeypatch.setattr(
        "Backends.src.models.model_loader.get_cached_yolo_model",
        fake_get_cached_yolo_model,
    )
    import Backends.src.ui.video_analysis as video_analysis

    importlib.reload(video_analysis)
    assert calls == []


def test_live_session_imports_without_model_load(monkeypatch):
    calls = []

    def fake_get_cached_yolo_model(model_key):
        calls.append(model_key)
        return None

    monkeypatch.setattr(
        "Backends.src.models.model_loader.get_cached_yolo_model",
        fake_get_cached_yolo_model,
    )
    import Backends.src.ui.live_session as live_session

    importlib.reload(live_session)
    assert calls == []
