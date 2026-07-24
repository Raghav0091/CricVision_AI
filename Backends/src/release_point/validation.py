"""Small validation helpers for Release Point V1 tests and future labels."""

from __future__ import annotations


def frame_error(estimated_frame: int | None, labelled_frame: int) -> int | None:
    if estimated_frame is None:
        return None
    return abs(int(estimated_frame) - int(labelled_frame))


def within_tolerance(
    estimated_frame: int | None,
    labelled_frame: int,
    *,
    tolerance_frames: int,
) -> bool:
    error = frame_error(estimated_frame, labelled_frame)
    return error is not None and error <= tolerance_frames

