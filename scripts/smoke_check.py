#!/usr/bin/env python3
"""Lightweight smoke checks for the FastAPI video analysis stack."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    importlib.import_module("services.api.main")
    importlib.import_module("services.api.services.delivery_physics_service")
    importlib.import_module("services.api.services.video_ball_tracking_service")
    importlib.import_module("packages.cricket_vision.calibration.cricket_pitch_geometry")

    from packages.cricket_vision.calibration.cricket_pitch_geometry import (
        CRICVISION_PITCH_V1,
    )

    assert CRICVISION_PITCH_V1
    print("Smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
