"""Resolve frame timestamps without hardcoded FPS assumptions."""

from __future__ import annotations

from typing import Literal, Sequence

TimestampMethod = Literal[
    "FRAME_TIMESTAMPS",
    "TIME_BASE",
    "CONTAINER_FPS",
    "NOMINAL_FPS_FALLBACK",
]

NOMINAL_FPS = 30.0


def resolve_frame_timestamp(
    frame_index: int,
    *,
    fps: float | None = None,
    frame_timestamps: Sequence[float] | None = None,
    time_base: tuple[float, float] | None = None,
    nominal_fps: float = NOMINAL_FPS,
) -> tuple[float, TimestampMethod]:
    """Return (timestamp_seconds, method) using the first applicable source."""
    if frame_timestamps is not None and 0 <= frame_index < len(frame_timestamps):
        timestamp = float(frame_timestamps[frame_index])
        if timestamp >= 0:
            return timestamp, "FRAME_TIMESTAMPS"

    if time_base is not None:
        offset, scale = time_base
        return offset + frame_index * scale, "TIME_BASE"

    if fps is not None and fps > 0:
        return frame_index / fps, "CONTAINER_FPS"

    # ponytail: nominal fallback keeps legacy analyses usable when container FPS is missing.
    return frame_index / max(nominal_fps, 1.0), "NOMINAL_FPS_FALLBACK"
