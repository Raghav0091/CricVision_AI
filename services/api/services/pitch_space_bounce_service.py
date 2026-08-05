"""Conservative pitch-space bounce selection over an existing ball track."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping


class _Record(dict):
    __getattr__ = dict.__getitem__


def _value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _result(values: dict[str, Any]) -> Any:
    try:
        from ..schemas import pitch_space_analysis as schema

        model = getattr(schema, "PitchSpaceBounceResult", None)
        return model.model_validate(values) if model is not None else _Record(values)
    except (ImportError, AttributeError, TypeError, ValueError):
        return _Record(values)


def _candidate_payload(candidate: Any) -> tuple[int | None, float, list[str]]:
    frame = _value(candidate, "bounce_frame", "frame_index")
    confidence = _clamp(float(_value(candidate, "confidence", "confidence_score", default=0.0)))
    evidence = list(_value(candidate, "evidence", default=[]) or [])
    try:
        return (int(frame) if frame is not None else None), confidence, evidence
    except (TypeError, ValueError):
        return None, confidence, evidence


def _local_evidence(points: list[Any], index: int) -> tuple[float, list[str]]:
    if index < 2 or index + 2 >= len(points):
        return 0.0, []
    p2, p1, point, n1, n2 = points[index - 2 : index + 3]
    score = 0.0
    evidence: list[str] = []
    try:
        image_before = float(_value(point, "image_y_px")) - float(_value(p2, "image_y_px"))
        image_after = float(_value(n2, "image_y_px")) - float(_value(point, "image_y_px"))
        if image_before > 0 and image_after < 0:
            score += 0.35
            evidence.append("image_vertical_motion_reversal")
        v1 = (
            float(_value(point, "pitch_x_m")) - float(_value(p1, "pitch_x_m")),
            float(_value(point, "pitch_y_m")) - float(_value(p1, "pitch_y_m")),
        )
        v2 = (
            float(_value(n1, "pitch_x_m")) - float(_value(point, "pitch_x_m")),
            float(_value(n1, "pitch_y_m")) - float(_value(point, "pitch_y_m")),
        )
        norm = math.hypot(*v1) * math.hypot(*v2)
        if norm > 1e-9:
            turn = 1.0 - max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / norm))
            if turn > 0.08:
                score += min(0.25, turn * 0.5)
                evidence.append("local_pitch_path_direction_change")
        if all(str(_value(p, "provenance", default="")).upper() == "OBSERVED" for p in (p1, point, n1)):
            score += 0.2
            evidence.append("observed_local_continuity")
        confidence = float(_value(point, "combined_confidence", default=0.0))
        if confidence >= 0.5:
            score += 0.15
            evidence.append("confident_pitch_space_point")
    except (TypeError, ValueError):
        return 0.0, []
    return _clamp(score), evidence


def estimate_pitch_space_bounce(
    pitch_space_track: Iterable[Any],
    *,
    existing_bounce: Any | None = None,
    existing_candidates: Iterable[Any] | None = None,
) -> Any:
    """Map existing bounce evidence first; only infer when evidence is strong."""
    points = sorted(list(pitch_space_track), key=lambda item: int(_value(item, "frame_index", default=0)))
    if len(points) < 3:
        return _result({
            "status": "UNAVAILABLE", "bounce_frame": None, "bounce_timestamp_seconds": None,
            "pitch_x_m": None, "pitch_y_m": None, "confidence": 0.0,
            "evidence": [], "alternative_candidates": [],
            "warnings": ["Insufficient pitch-space track points for bounce estimation."],
        })

    candidates: list[tuple[int, float, list[str], str]] = []
    detected = _value(existing_bounce, "bounce_detected", "status")
    frame, confidence, evidence = _candidate_payload(existing_bounce) if existing_bounce is not None else (None, 0.0, [])
    if frame is not None and detected in (True, "DETECTED", "ESTIMATED", "uncertain", "UNCERTAIN"):
        candidates.append((frame, confidence, evidence, "existing_bounce"))
    for candidate in existing_candidates or []:
        frame, confidence, evidence = _candidate_payload(candidate)
        if frame is not None:
            candidates.append((frame, confidence, evidence, "existing_candidate"))

    by_frame = {int(_value(point, "frame_index")): (index, point) for index, point in enumerate(points)}
    scored: list[dict[str, Any]] = []
    for frame, prior, prior_evidence, source in candidates:
        match = by_frame.get(frame)
        if match is None:
            continue
        index, point = match
        local_score, local_evidence = _local_evidence(points, index)
        combined = _clamp(0.65 * prior + 0.35 * local_score)
        scored.append({"frame": frame, "point": point, "confidence": combined,
                       "evidence": [f"source:{source}", *prior_evidence, *local_evidence]})

    if not scored:
        for index in range(2, len(points) - 2):
            score, evidence = _local_evidence(points, index)
            if score >= 0.72 and len(evidence) >= 3:
                scored.append({"frame": int(_value(points[index], "frame_index")), "point": points[index],
                               "confidence": score * 0.75, "evidence": evidence})
    scored.sort(key=lambda item: (-item["confidence"], item["frame"]))
    if not scored or scored[0]["confidence"] < 0.35:
        alternatives = [{"frame_index": item["frame"], "confidence": round(item["confidence"], 6)} for item in scored[:3]]
        return _result({
            "status": "UNAVAILABLE", "bounce_frame": None, "bounce_timestamp_seconds": None,
            "pitch_x_m": None, "pitch_y_m": None, "confidence": 0.0,
            "evidence": [], "alternative_candidates": alternatives,
            "warnings": ["No bounce candidate has sufficient independent evidence."],
        })

    best = scored[0]
    point = best["point"]
    alternatives = [
        {"frame_index": item["frame"], "confidence": round(item["confidence"], 6)}
        for item in scored[1:4]
    ]
    return _result({
        "status": "DETECTED" if best["confidence"] >= 0.5 else "UNCERTAIN",
        "bounce_frame": best["frame"],
        "bounce_timestamp_seconds": float(_value(point, "timestamp_seconds")),
        "pitch_x_m": float(_value(point, "pitch_x_m")),
        "pitch_y_m": float(_value(point, "pitch_y_m")),
        "confidence": round(best["confidence"], 6),
        "evidence": list(dict.fromkeys(best["evidence"])),
        "alternative_candidates": alternatives,
        "warnings": ["Bounce is a ground-plane estimate; true airborne height is unavailable."],
    })


estimate_bounce = estimate_pitch_space_bounce

