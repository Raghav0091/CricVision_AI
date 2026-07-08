"""Render estimated 3D trajectory replay (Plotly preferred, OpenCV fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np

from Backends.src.utils.cv2_loader import cv2

PITCH_LENGTH_FT = 66.0
PITCH_WIDTH_FT = 10.0
TRAJECTORY_COLOR = (30, 30, 220)
BOUNCE_COLOR = (0, 165, 255)
IMPACT_COLOR = (0, 220, 120)
STUMP_COLOR = (220, 220, 220)


def build_3d_replay_figure(
    trajectory_3d: dict[str, Any],
    calibration_context: dict[str, Any] | None = None,
    *,
    width: int = 720,
    height: int = 520,
) -> dict[str, Any]:
    """Return a render payload with either a Plotly figure or static image."""
    calibration_context = calibration_context or {}
    if not trajectory_3d.get("available"):
        return {
            "available": False,
            "backend": "none",
            "figure": None,
            "image": None,
            "caption": "Estimated 3D replay unavailable.",
        }

    points = trajectory_3d.get("points_3d") or []
    if len(points) < 2:
        return {
            "available": False,
            "backend": "none",
            "figure": None,
            "image": None,
            "caption": "Estimated 3D replay unavailable.",
        }

    caption = (
        "Estimated 3D Replay — based on tracked video points and stump/pitch calibration."
    )

    plotly_figure = _build_plotly_figure(
        trajectory_3d,
        calibration_context,
        width=width,
        height=height,
        caption=caption,
    )
    if plotly_figure is not None:
        return {
            "available": True,
            "backend": "plotly",
            "figure": plotly_figure,
            "image": None,
            "caption": caption,
        }

    image = _build_opencv_fallback(
        trajectory_3d,
        calibration_context,
        width=width,
        height=height,
        caption=caption,
    )
    return {
        "available": image is not None,
        "backend": "opencv",
        "figure": None,
        "image": image,
        "caption": caption,
    }


def _build_plotly_figure(
    trajectory_3d: dict[str, Any],
    calibration_context: dict[str, Any],
    *,
    width: int,
    height: int,
    caption: str,
) -> Any | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    points = trajectory_3d.get("points_3d") or []
    xs = [point["x_ft"] for point in points]
    ys = [point["y_ft"] for point in points]
    zs = [point["z_ft"] for point in points]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            line={"color": "red", "width": 6},
            marker={"size": 3, "color": "red"},
            name="Ball path",
        )
    )

    release = trajectory_3d.get("release_3d")
    if release:
        fig.add_trace(
            go.Scatter3d(
                x=[release["x_ft"]],
                y=[release["y_ft"]],
                z=[release["z_ft"]],
                mode="markers",
                marker={"size": 6, "color": "yellow", "symbol": "circle"},
                name="Release",
            )
        )

    bounce = trajectory_3d.get("bounce_3d")
    if bounce:
        fig.add_trace(
            go.Scatter3d(
                x=[bounce["x_ft"]],
                y=[bounce["y_ft"]],
                z=[0.0],
                mode="markers",
                marker={"size": 7, "color": "orange", "symbol": "diamond"},
                name="Bounce",
            )
        )

    impact = trajectory_3d.get("impact_3d")
    if impact:
        fig.add_trace(
            go.Scatter3d(
                x=[impact["x_ft"]],
                y=[impact["y_ft"]],
                z=[impact["z_ft"]],
                mode="markers",
                marker={"size": 7, "color": "lime", "symbol": "square"},
                name="Impact",
            )
        )

    _add_pitch_and_stumps_plotly(fig, calibration_context)

    quality = trajectory_3d.get("trajectory_quality", "Unknown")
    cal_quality = calibration_context.get("calibration_quality", "Unknown")
    fig.update_layout(
        title={
            "text": (
                "Estimated 3D Replay — based on tracked video points "
                "and stump/pitch calibration."
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        scene={
            "xaxis": {"title": "Lateral (ft)", "range": [-8, 8]},
            "yaxis": {"title": "Pitch length (ft)", "range": [-2, PITCH_LENGTH_FT + 4]},
            "zaxis": {"title": "Height (ft)", "range": [0, 12]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 3, "z": 0.6},
        },
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        width=width,
        height=height,
        showlegend=True,
        annotations=[
            {
                "text": (
                    f"Trajectory quality: {quality} | "
                    f"Calibration: {cal_quality} | "
                    "Speed/Swing/Spin/LBW: not available"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.02,
                "showarrow": False,
                "font": {"size": 11},
            }
        ],
    )
    fig.layout.meta = {"caption": caption}
    return fig


def _add_pitch_and_stumps_plotly(fig: Any, calibration_context: dict[str, Any]) -> None:
    import plotly.graph_objects as go

    half_width = PITCH_WIDTH_FT / 2.0
    pitch_x = [-half_width, half_width, half_width, -half_width, -half_width]
    pitch_y = [0, 0, PITCH_LENGTH_FT, PITCH_LENGTH_FT, 0]
    pitch_z = [0, 0, 0, 0, 0]
    fig.add_trace(
        go.Scatter3d(
            x=pitch_x,
            y=pitch_y,
            z=pitch_z,
            mode="lines",
            line={"color": "green", "width": 4},
            name="Pitch",
        )
    )

    for end_name, y_pos in (("Bowler stumps", 0.0), ("Batter stumps", PITCH_LENGTH_FT)):
        stump_x = [-0.4, 0.0, 0.4]
        stump_y = [y_pos, y_pos, y_pos]
        stump_z = [0.0, 2.3, 0.0]
        fig.add_trace(
            go.Scatter3d(
                x=stump_x,
                y=stump_y,
                z=stump_z,
                mode="lines+markers",
                line={"color": "white", "width": 5},
                marker={"size": 2, "color": "white"},
                name=end_name,
                showlegend=True,
            )
        )


def _build_opencv_fallback(
    trajectory_3d: dict[str, Any],
    calibration_context: dict[str, Any],
    *,
    width: int,
    height: int,
    caption: str,
) -> np.ndarray | None:
    points = trajectory_3d.get("points_3d") or []
    if len(points) < 2:
        return None

    width = max(int(width), 1)
    height = max(int(height), 1)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (36, 70, 36)

    margin = 50
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin - 80

    def project(point: dict[str, float]) -> tuple[int, int]:
        x_ratio = (point["x_ft"] + 6.0) / 12.0
        y_ratio = point["y_ft"] / PITCH_LENGTH_FT
        px = margin + int(x_ratio * plot_w)
        py = margin + int((1.0 - y_ratio) * plot_h)
        return px, py

    pitch_corners = [
        project({"x_ft": -PITCH_WIDTH_FT / 2, "y_ft": 0.0, "z_ft": 0.0}),
        project({"x_ft": PITCH_WIDTH_FT / 2, "y_ft": 0.0, "z_ft": 0.0}),
        project({"x_ft": PITCH_WIDTH_FT / 2, "y_ft": PITCH_LENGTH_FT, "z_ft": 0.0}),
        project({"x_ft": -PITCH_WIDTH_FT / 2, "y_ft": PITCH_LENGTH_FT, "z_ft": 0.0}),
    ]
    cv2.fillPoly(canvas, [np.array(pitch_corners, dtype=np.int32)], (90, 150, 70))

    mapped = [project(point) for point in points]
    if len(mapped) >= 2:
        cv2.polylines(
            canvas,
            [np.array(mapped, dtype=np.int32)],
            False,
            TRAJECTORY_COLOR,
            3,
            cv2.LINE_AA,
        )

    bounce = trajectory_3d.get("bounce_3d")
    if bounce:
        cv2.circle(canvas, project(bounce), 8, BOUNCE_COLOR, -1, lineType=cv2.LINE_AA)

    impact = trajectory_3d.get("impact_3d")
    if impact:
        cv2.circle(canvas, project(impact), 8, IMPACT_COLOR, -1, lineType=cv2.LINE_AA)

    for stump_y in (0.0, PITCH_LENGTH_FT):
        center = project({"x_ft": 0.0, "y_ft": stump_y, "z_ft": 0.0})
        _draw_stumps_2d(canvas, center)

    cv2.putText(
        canvas,
        caption[:70],
        (margin, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Static fallback (Plotly unavailable)",
        (margin, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _draw_stumps_2d(canvas: np.ndarray, center: tuple[int, int]) -> None:
    cx, cy = center
    for offset in (-6, 0, 6):
        cv2.line(canvas, (cx + offset, cy), (cx + offset, cy - 16), STUMP_COLOR, 2, cv2.LINE_AA)
