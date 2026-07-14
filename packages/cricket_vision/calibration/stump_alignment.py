from typing import Mapping


Box = dict[str, float]
BoxLayout = dict[str, Box]


def build_alignment_boxes(frame_width: int, frame_height: int) -> BoxLayout:
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    return {
        "striker": {
            "x": frame_width * 0.43,
            "y": frame_height * 0.22,
            "width": frame_width * 0.14,
            "height": frame_height * 0.24,
        },
        "non_striker": {
            "x": frame_width * 0.35,
            "y": frame_height * 0.66,
            "width": frame_width * 0.30,
            "height": frame_height * 0.28,
        },
    }


def validate_box_layout(box_layout: Mapping[str, Mapping[str, float]]) -> bool:
    if set(box_layout) != {"striker", "non_striker"}:
        return False
    for box in box_layout.values():
        if not all(key in box for key in ("x", "y", "width", "height")):
            return False
        if box["x"] < 0 or box["y"] < 0 or box["width"] <= 0 or box["height"] <= 0:
            return False
    return True


def calculate_box_centers(box_layout: Mapping[str, Mapping[str, float]]) -> dict[str, tuple[float, float]]:
    if not validate_box_layout(box_layout):
        raise ValueError("Invalid stump alignment box layout.")
    return {
        name: (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        for name, box in box_layout.items()
    }
