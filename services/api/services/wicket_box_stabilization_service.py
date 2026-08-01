"""Robust early-frame stabilization for near/far wicket boxes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..schemas.pitch_space_analysis import (
    CameraStabilityResult,
    FrameWicketBox,
    SetupFrameEvaluation,
    StableWicketBox,
)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    threshold = sum(weights) / 2
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= threshold:
            return float(value)
    return float(ordered[-1][0])


def _stabilize_role(
    evaluations: Sequence[SetupFrameEvaluation],
    *,
    role: str,
    source: str,
) -> StableWicketBox | None:
    observations: list[tuple[int, FrameWicketBox]] = []
    for evaluation in evaluations:
        box = evaluation.near_wicket if role == "NEAR" else evaluation.far_wicket
        if box is not None and not box.clipped:
            observations.append((evaluation.frame_index, box))
    if not observations:
        return None
    centres_x = [box.x + box.width / 2 for _, box in observations]
    centres_y = [box.y + box.height / 2 for _, box in observations]
    widths = [box.width for _, box in observations]
    heights = [box.height for _, box in observations]
    weights = [max(0.05, box.confidence) for _, box in observations]
    median_x = _weighted_median(centres_x, weights)
    median_y = _weighted_median(centres_y, weights)
    median_w = _weighted_median(widths, weights)
    median_h = _weighted_median(heights, weights)
    compatible: list[tuple[int, FrameWicketBox]] = []
    for frame_index, box in observations:
        centre_distance = math.hypot(
            box.x + box.width / 2 - median_x,
            box.y + box.height / 2 - median_y,
        )
        size_error = abs(box.width - median_w) / median_w + abs(box.height - median_h) / median_h
        if centre_distance <= max(median_w, median_h) * 0.55 and size_error <= 0.8:
            compatible.append((frame_index, box))
    if not compatible:
        compatible = [max(observations, key=lambda item: item[1].confidence)]
    weights = [max(0.05, box.confidence) for _, box in compatible]
    xs = [box.x for _, box in compatible]
    ys = [box.y for _, box in compatible]
    widths = [box.width for _, box in compatible]
    heights = [box.height for _, box in compatible]
    x = _weighted_median(xs, weights)
    y = _weighted_median(ys, weights)
    width = _weighted_median(widths, weights)
    height = _weighted_median(heights, weights)
    centre_x, centre_y = x + width / 2, y + height / 2
    centre_spread = _weighted_median(
        [
            math.hypot(
                box.x + box.width / 2 - centre_x,
                box.y + box.height / 2 - centre_y,
            )
            for _, box in compatible
        ],
        weights,
    )
    size_spread = _weighted_median(
        [
            (abs(box.width - width) / width + abs(box.height - height) / height) / 2
            for _, box in compatible
        ],
        weights,
    )
    return StableWicketBox(
        perspective_role=role,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=min(1.0, _weighted_median([box.confidence for _, box in compatible], weights) * min(1.0, 0.6 + len(compatible) * 0.12)),
        frame_support=len(compatible),
        supporting_frame_indices=sorted(frame for frame, _ in compatible),
        centre_spread_px=round(centre_spread, 4),
        size_spread_ratio=round(size_spread, 6),
        clipped=False,
        source=source,
    )


def stabilize_wicket_boxes(
    evaluations: Sequence[SetupFrameEvaluation],
    *,
    source: str,
) -> tuple[StableWicketBox | None, StableWicketBox | None]:
    """Stabilize boxes without changing the selected visual reference frame."""
    return (
        _stabilize_role(evaluations, role="NEAR", source=source),
        _stabilize_role(evaluations, role="FAR", source=source),
    )


def assess_camera_stability(
    near: StableWicketBox | None,
    far: StableWicketBox | None,
) -> CameraStabilityResult:
    if near is None or far is None:
        return CameraStabilityResult(
            status="UNAVAILABLE",
            confidence=0,
            warnings=["Two stable wicket tracks are required for camera stability."],
        )
    frames = sorted(set(near.supporting_frame_indices + far.supporting_frame_indices))
    maximum_centre = max(
        near.centre_spread_px / max(near.width, near.height),
        far.centre_spread_px / max(far.width, far.height),
    )
    maximum_scale = max(near.size_spread_ratio, far.size_spread_ratio)
    if maximum_centre <= 0.12 and maximum_scale <= 0.16:
        status = "FIXED_CAMERA"
    elif maximum_centre <= 0.25 and maximum_scale <= 0.30:
        status = "MINOR_DRIFT"
    else:
        status = "UNSTABLE_CAMERA"
    support_confidence = min(1.0, min(near.frame_support, far.frame_support) / 3)
    return CameraStabilityResult(
        status=status,
        frames_checked=frames,
        maximum_centre_drift_ratio=round(maximum_centre, 6),
        maximum_scale_change_ratio=round(maximum_scale, 6),
        confidence=round(support_confidence * max(0.0, 1 - maximum_centre), 6),
        reliable_until_frame=max(frames) if status != "UNSTABLE_CAMERA" and frames else None,
        warnings=(
            ["Only early persisted wicket evidence was available; periodic full-video validation remains pending."]
            if frames
            else []
        ),
    )

