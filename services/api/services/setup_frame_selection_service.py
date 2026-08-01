"""Deterministic Frame-0-first setup-frame selection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ..schemas.pitch_space_analysis import (
    FrameWicketBox,
    SetupFrameDecision,
    SetupFrameEvaluation,
)
from ..schemas.wicket_observation import (
    RawWicketDetection,
    SetupFrameCandidate,
)


EARLY_FRAME_TARGETS = (0, 5, 10, 15, 20)
MIN_WICKET_CONFIDENCE = 0.18


def deterministic_early_frame_indices(
    available_indices: Sequence[int],
) -> list[int]:
    """Map the fixed early policy onto frames already inspected by V1."""
    available = sorted({int(index) for index in available_indices if 0 <= index <= 20})
    if not available:
        return [0]
    selected: list[int] = []
    for target in EARLY_FRAME_TARGETS:
        nearest = min(available, key=lambda value: (abs(value - target), value))
        if nearest not in selected:
            selected.append(nearest)
    return selected


def _clipped(box: FrameWicketBox, width: int, height: int) -> bool:
    margin = 1.0
    return (
        box.x <= margin
        or box.y <= margin
        or box.x + box.width >= width - margin
        or box.y + box.height >= height - margin
    )


def _pair_wickets(
    detections: Sequence[RawWicketDetection], width: int, height: int
) -> tuple[FrameWicketBox | None, FrameWicketBox | None, list[str]]:
    accepted = [item for item in detections if item.confidence >= MIN_WICKET_CONFIDENCE]
    if len(accepted) < 2:
        return None, None, ["both_wickets_not_detected"]
    pairs: list[tuple[float, RawWicketDetection, RawWicketDetection]] = []
    for index, first in enumerate(accepted):
        for second in accepted[index + 1 :]:
            first_bottom = first.bbox.y + first.bbox.height
            second_bottom = second.bbox.y + second.bbox.height
            near, far = (first, second) if first_bottom >= second_bottom else (second, first)
            vertical = (near.bbox.y + near.bbox.height - far.bbox.y - far.bbox.height) / max(height, 1)
            near_area = near.bbox.width * near.bbox.height
            far_area = far.bbox.width * far.bbox.height
            if vertical < 0.025 or near_area < far_area * 0.45:
                continue
            overlap_x = max(
                0.0,
                min(near.bbox.x + near.bbox.width, far.bbox.x + far.bbox.width)
                - max(near.bbox.x, far.bbox.x),
            )
            overlap_y = max(
                0.0,
                min(near.bbox.y + near.bbox.height, far.bbox.y + far.bbox.height)
                - max(near.bbox.y, far.bbox.y),
            )
            intersection = overlap_x * overlap_y
            if intersection / max(min(near_area, far_area), 1.0) > 0.5:
                continue
            score = (
                near.confidence
                + far.confidence
                + min(1.0, vertical / 0.2)
                + min(1.0, near_area / max(far_area, 1.0)) * 0.25
            )
            pairs.append((score, near, far))
    if not pairs:
        return None, None, ["wicket_pair_geometry_implausible"]
    _, near_raw, far_raw = max(pairs, key=lambda item: item[0])

    def convert(item: RawWicketDetection) -> FrameWicketBox:
        box = FrameWicketBox(
            x=item.bbox.x,
            y=item.bbox.y,
            width=item.bbox.width,
            height=item.bbox.height,
            confidence=item.confidence,
            source=item.source,
        )
        return box.model_copy(update={"clipped": _clipped(box, width, height)})

    return convert(near_raw), convert(far_raw), []


def evaluate_setup_frames(
    frame_candidates: Sequence[SetupFrameCandidate],
    detections: Sequence[RawWicketDetection],
) -> list[SetupFrameEvaluation]:
    candidates = {item.frame_index: item for item in frame_candidates}
    grouped: dict[int, list[RawWicketDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.frame_index].append(detection)
    indices = deterministic_early_frame_indices([*candidates, *grouped])
    evaluations: list[SetupFrameEvaluation] = []
    for frame_index in indices:
        candidate = candidates.get(frame_index)
        if candidate is None:
            evaluations.append(
                SetupFrameEvaluation(
                    frame_index=frame_index,
                    timestamp_seconds=0,
                    decoded=False,
                    image_width=0,
                    image_height=0,
                    sharpness=0,
                    brightness=0,
                    suitable=False,
                    quality_score=0,
                    reasons=["frame_not_present_in_persisted_early_evidence"],
                )
            )
            continue
        near, far, reasons = _pair_wickets(
            grouped.get(frame_index, []), candidate.image_width, candidate.image_height
        )
        if candidate.sharpness < 18:
            reasons.append("frame_blurred_or_low_detail")
        if not 14 <= candidate.brightness <= 246:
            reasons.append("frame_exposure_out_of_range")
        if near and near.clipped:
            reasons.append("near_wicket_severely_clipped")
        if far and far.clipped:
            reasons.append("far_wicket_severely_clipped")
        pair_confidence = (
            (near.confidence + far.confidence) / 2 if near and far else 0.0
        )
        quality = max(
            0.0,
            min(1.0, 0.65 * candidate.score + 0.35 * pair_confidence),
        )
        evaluations.append(
            SetupFrameEvaluation(
                frame_index=frame_index,
                timestamp_seconds=candidate.timestamp_seconds,
                decoded=True,
                image_width=candidate.image_width,
                image_height=candidate.image_height,
                sharpness=candidate.sharpness,
                brightness=candidate.brightness,
                near_wicket=near,
                far_wicket=far,
                suitable=not reasons and near is not None and far is not None,
                quality_score=round(quality, 6),
                reasons=reasons,
            )
        )
    return evaluations


def select_setup_frame(
    frame_candidates: Sequence[SetupFrameCandidate],
    detections: Sequence[RawWicketDetection],
) -> SetupFrameDecision:
    evaluations = evaluate_setup_frames(frame_candidates, detections)
    frame_zero = next((item for item in evaluations if item.frame_index == 0), None)
    if frame_zero and frame_zero.suitable:
        return SetupFrameDecision(
            preferred_frame_passed=True,
            selected_frame_index=0,
            selected_timestamp_seconds=frame_zero.timestamp_seconds,
            fallback_used=False,
            fallback_candidates=[item.frame_index for item in evaluations if item.frame_index != 0],
            evaluations=evaluations,
            selection_reasons=["FRAME_ZERO_USABLE_AND_RETAINED"],
            quality_score=frame_zero.quality_score,
        )
    suitable = [item for item in evaluations if item.frame_index != 0 and item.suitable]
    selected = max(suitable, key=lambda item: (item.quality_score, -item.frame_index), default=None)
    reasons = ["FRAME_ZERO_UNUSABLE"]
    if selected:
        reasons.append("DETERMINISTIC_EARLY_FALLBACK_SELECTED")
    else:
        reasons.append("NO_EARLY_FRAME_WITH_TWO_USABLE_WICKETS")
    return SetupFrameDecision(
        preferred_frame_passed=False,
        selected_frame_index=selected.frame_index if selected else None,
        selected_timestamp_seconds=selected.timestamp_seconds if selected else None,
        fallback_used=selected is not None,
        fallback_candidates=[item.frame_index for item in evaluations if item.frame_index != 0],
        evaluations=evaluations,
        selection_reasons=reasons,
        quality_score=selected.quality_score if selected else 0,
    )
