"""Live session pitch overlays — alignment boxes and estimated virtual stumps.

Visual helpers only. No official LBW/DRS lines.
"""

from __future__ import annotations

from typing import Any

from Backends.src.utils.cv2_loader import cv2


def _frame_line_thickness(frame, base=2):
    height, width = frame.shape[:2]
    return max(base, min(4, int(round(min(width, height) / 360))))


def _draw_dashed_rect(frame, x1, y1, x2, y2, color, thickness=2, dash=12, gap=8):
    """Dashed rectangle; falls back cleanly if segment math fails."""
    try:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        segments = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (sx, sy), (ex, ey) in segments:
            length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
            if length < 1:
                continue
            dx = (ex - sx) / length
            dy = (ey - sy) / length
            pos = 0.0
            draw = True
            while pos < length:
                step = dash if draw else gap
                next_pos = min(pos + step, length)
                if draw:
                    p1 = (int(sx + dx * pos), int(sy + dy * pos))
                    p2 = (int(sx + dx * next_pos), int(sy + dy * next_pos))
                    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
                pos = next_pos
                draw = not draw
    except (TypeError, ValueError):
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)


def _draw_white_black_label(frame, text, x, y):
    """Small white-bg / black-text label that stays inside the frame."""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.52, width / 1600.0))
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x, pad_y = 5, 3
    box_w = text_w + pad_x * 2
    box_h = text_h + baseline + pad_y * 2

    lx = max(2, min(int(x), width - box_w - 2))
    ly = int(y) - box_h - 2
    if ly < 2:
        ly = max(2, min(int(y) + 2, height - box_h - 2))

    cv2.rectangle(frame, (lx, ly), (lx + box_w, ly + box_h), (255, 255, 255), -1)
    cv2.putText(
        frame,
        text,
        (lx + pad_x, ly + box_h - pad_y - baseline),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_alignment_boxes(frame, box_layout=None, validation_result=None):
    """align_stumps: dashed boxes + small labels. Green when validation marks a side found."""
    layout = box_layout if isinstance(box_layout, dict) else {}
    validation = validation_result if isinstance(validation_result, dict) else {}
    striker = layout.get("striker_stumps_box") or {}
    non_striker = layout.get("non_striker_stumps_box") or {}
    thickness = _frame_line_thickness(frame, base=2)
    drawn = False
    red = (0, 0, 255)
    green = (0, 200, 80)

    try:
        if all(k in striker for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (
                int(striker["x1"]),
                int(striker["y1"]),
                int(striker["x2"]),
                int(striker["y2"]),
            )
            striker_found = bool((validation.get("striker") or {}).get("found"))
            color = green if striker_found else red
            _draw_dashed_rect(frame, x1, y1, x2, y2, color, thickness)
            _draw_white_black_label(frame, "Striker", x1, y1)
            drawn = True
        if all(k in non_striker for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (
                int(non_striker["x1"]),
                int(non_striker["y1"]),
                int(non_striker["x2"]),
                int(non_striker["y2"]),
            )
            non_found = bool((validation.get("non_striker") or {}).get("found"))
            color = green if non_found else red
            _draw_dashed_rect(frame, x1, y1, x2, y2, color, thickness)
            _draw_white_black_label(frame, "Non-Striker", x1, y1)
            drawn = True
    except (TypeError, ValueError):
        pass
    return drawn


def _parse_corridor_points(corridor: Any) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for point in corridor or []:
        if isinstance(point, dict) and "x" in point and "y" in point:
            try:
                points.append((int(point["x"]), int(point["y"])))
            except (TypeError, ValueError):
                continue
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                points.append((int(point[0]), int(point[1])))
            except (TypeError, ValueError):
                continue
    return points


def _draw_virtual_stump_lines(frame, stump_lines, color, thickness):
    drawn = False
    for line in stump_lines or []:
        if not isinstance(line, dict):
            continue
        try:
            x1 = int(line.get("x1", line.get("x")))
            y1 = int(line["y1"])
            x2 = int(line.get("x2", line.get("x")))
            y2 = int(line["y2"])
            cv2.line(frame, (x1, y1), (x2, y2), color, max(1, thickness), cv2.LINE_AA)
            drawn = True
        except (TypeError, ValueError, KeyError):
            continue
    return drawn


def draw_virtual_stumps_overlay(frame, calibration_result=None):
    """setup_complete / live_capture: estimated virtual stumps + subtle pitch corridor."""
    result = calibration_result if isinstance(calibration_result, dict) else {}
    env = result.get("environment_context") if isinstance(result.get("environment_context"), dict) else {}
    virtual = result.get("virtual_stumps") if isinstance(result.get("virtual_stumps"), dict) else {}
    thickness = _frame_line_thickness(frame, base=2)
    drawn = False

    near_lines = virtual.get("near_virtual_stumps") or virtual.get("non_striker_virtual_stumps") or []
    far_lines = virtual.get("far_virtual_stumps") or virtual.get("striker_virtual_stumps") or []
    if _draw_virtual_stump_lines(frame, near_lines, (180, 200, 120), thickness):
        drawn = True
    if _draw_virtual_stump_lines(frame, far_lines, (140, 180, 220), thickness):
        drawn = True

    corridor = (
        env.get("pitch_corridor")
        or result.get("pitch_corridor")
        or []
    )
    points = _parse_corridor_points(corridor)
    if len(points) >= 3:
        for index, point in enumerate(points):
            cv2.line(
                frame,
                point,
                points[(index + 1) % len(points)],
                (80, 140, 255),
                max(1, thickness - 1),
                cv2.LINE_AA,
            )
        drawn = True

    if drawn and virtual.get("available"):
        _draw_white_black_label(frame, "Est. stumps", 8, 24)

    return drawn


# Backward-compat aliases used by older live_session imports.
def draw_stump_alignment_overlay(frame, box_layout, validation_result=None):
    """Legacy alias — alignment stage ignores validation colouring."""
    _ = validation_result
    return draw_alignment_boxes(frame, box_layout)


def draw_alignment_overlay(frame, box_layout, validation=None):
    return draw_stump_alignment_overlay(frame, box_layout, validation_result=validation)


def draw_environment_preview_overlay(
    frame,
    environment_context=None,
    box_layout=None,
    calibration=None,
    show_pitch_axis=False,
):
    """After setup_complete: virtual stumps + corridor. No blue LBW line by default."""
    _ = box_layout
    report = calibration if isinstance(calibration, dict) else {}
    merged = dict(report)
    if isinstance(environment_context, dict) and environment_context:
        merged.setdefault("environment_context", environment_context)
        merged.setdefault("pitch_corridor", environment_context.get("pitch_corridor"))
    drawn = draw_virtual_stumps_overlay(frame, merged)
    if show_pitch_axis:
        env = environment_context if isinstance(environment_context, dict) else {}
        pitch_axis = env.get("pitch_axis") or env.get("stump_line") or report.get("stump_line") or {}
        start = pitch_axis.get("start") or {}
        end = pitch_axis.get("end") or {}
        try:
            if "x" in start and "y" in start and "x" in end and "y" in end:
                p1 = (int(start["x"]), int(start["y"]))
                p2 = (int(end["x"]), int(end["y"]))
                thickness = _frame_line_thickness(frame, base=2)
                cv2.line(frame, p1, p2, (80, 180, 255), thickness, cv2.LINE_AA)
                _draw_white_black_label(frame, "Pitch axis preview", p1[0], p1[1])
                drawn = True
        except (TypeError, ValueError):
            pass
    return drawn


def draw_calibrated_overlay(frame, calibration, box_layout=None):
    _ = box_layout
    env = calibration.get("environment_context") if isinstance(calibration, dict) else None
    return draw_environment_preview_overlay(
        frame,
        environment_context=env,
        calibration=calibration,
        show_pitch_axis=False,
    )


def draw_setup_complete_overlay(frame, calibration_result=None, environment_context=None):
    """setup_complete: virtual stumps + subtle corridor — no LBW line."""
    report = calibration_result if isinstance(calibration_result, dict) else {}
    merged = dict(report)
    if isinstance(environment_context, dict) and environment_context:
        merged.setdefault("environment_context", environment_context)
        merged.setdefault("pitch_corridor", environment_context.get("pitch_corridor"))
    return draw_virtual_stumps_overlay(frame, merged)
