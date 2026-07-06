"""Options for the reusable delivery-analysis engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class EngineOptions:
    """Normalized options for one delivery clip.

    The first five fields are the stable public surface. The remaining fields
    preserve current Video Analysis controls and adapter callbacks.
    """

    analysis_mode: str = "Full Delivery Analysis"
    smart_mode: str = "Smart Balanced"
    processed_video_enabled: bool = True
    overlay_detail: str = "Clean"
    confidence_threshold: float = 0.25

    output_path: str | Path | None = None
    browser_output_path: str | Path | None = None
    model_path: str | Path | None = None
    model_key: str | None = "current_best"
    model_name: str | None = None
    ball_model_key: str | None = "current_best"
    bat_model_key: str | None = None
    class_names: Any = None
    image_size: int = 640
    use_ensemble: bool = False
    show_pitch_roi: bool = False
    calibration_mode: str = "Auto calibration using detected stumps"
    manual_pitch_points: Any = None
    shot_trajectory_mode: str = "Use last part of trajectory"
    manual_contact_frame: int | None = None
    field_setup: dict = field(default_factory=dict)
    max_frames: int | None = None
    active_preset: str | None = None
    show_performance_details: bool = False
    progress_callback: Any = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        valid_analysis_modes = {
            "Bowling Analysis",
            "Batting Analysis",
            "Full Delivery Analysis",
        }
        if self.analysis_mode not in valid_analysis_modes:
            self.analysis_mode = "Full Delivery Analysis"

        self.smart_mode = str(self.smart_mode or "Smart Balanced")
        self.overlay_detail = (
            "Debug"
            if str(self.overlay_detail or "").strip().lower() == "debug"
            else "Clean"
        )
        self.processed_video_enabled = bool(self.processed_video_enabled)
        self.confidence_threshold = _bounded_float(
            self.confidence_threshold,
            default=0.25,
        )
        self.image_size = _positive_int(self.image_size, default=640)
        self.max_frames = _optional_positive_int(self.max_frames)
        if not isinstance(self.field_setup, dict):
            self.field_setup = {}
        if not callable(self.progress_callback):
            self.progress_callback = None

    @classmethod
    def from_value(cls, value=None) -> "EngineOptions":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("options must be EngineOptions, a mapping, or None")

        normalized = dict(value)
        aliases = {
            "speed_mode": "smart_mode",
            "generate_processed_video": "processed_video_enabled",
            "confidence": "confidence_threshold",
            "imgsz": "image_size",
            "preset_name": "active_preset",
            "selected_model_name": "model_name",
        }
        for old_name, new_name in aliases.items():
            if old_name in normalized and new_name not in normalized:
                normalized[new_name] = normalized.pop(old_name)

        accepted = {item.name for item in fields(cls)}
        unknown = sorted(set(normalized) - accepted)
        if unknown:
            names = ", ".join(unknown)
            raise TypeError(f"Unknown engine option(s): {names}")
        return cls(**normalized)

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("progress_callback", None)
        return result


def _bounded_float(value, *, default):
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _positive_int(value, *, default):
    try:
        value = int(value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _optional_positive_int(value):
    if value is None:
        return None
    try:
        value = int(value)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None
