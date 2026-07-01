"""Shared configuration stays import-safe and independent of the working directory."""

from pathlib import Path

from Backends.src.config.constants import DETECTION_PRESETS
from Backends.src.config.paths import (
    DATASETS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
)


def test_project_paths_are_absolute_and_rooted_in_repository(monkeypatch, tmp_path):
    expected_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)

    assert PROJECT_ROOT == expected_root
    assert PROJECT_ROOT.is_absolute()
    for path in (DATASETS_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR):
        assert path.is_absolute()
        assert path.is_relative_to(PROJECT_ROOT)


def test_shared_detection_presets_keep_production_defaults():
    assert list(DETECTION_PRESETS) == [
        "Fast Bowling Mode",
        "Balanced Mode",
        "High Precision Mode",
    ]
    assert DETECTION_PRESETS["Balanced Mode"] == {
        "imgsz": 768,
        "confidence": 0.25,
    }
