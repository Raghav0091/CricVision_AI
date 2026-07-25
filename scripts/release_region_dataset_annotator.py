"""Standalone Tkinter annotator for datasets/release_region_v1.

Run:
    python scripts/release_region_dataset_annotator.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "release_region_v1"
ANNOTATIONS = DATASET / "annotations.json"
MANIFEST = DATASET / "manifest.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.release_region_dataset_builder import (  # noqa: E402
    ANNOTATION_ENUMS,
    atomic_write_json,
    validate_annotation,
)


def save_annotation_document(path: Path, document: dict[str, Any]) -> None:
    for annotation in document.get("annotations", []):
        validate_annotation(annotation)
    atomic_write_json(path, document)


def copy_previous_bbox(
    annotations: list[dict[str, Any]], index: int
) -> list[float] | None:
    sequence = annotations[index]["sequence_id"]
    for previous in reversed(annotations[:index]):
        if previous["sequence_id"] == sequence and previous.get("ball_bbox_xyxy"):
            return list(previous["ball_bbox_xyxy"])
    return None


class Annotator:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.document = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
        self.annotations = self.document["annotations"]
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.metadata = {row["sample_id"]: row for row in self.manifest["frames"]}
        self.index = next(
            (i for i, row in enumerate(self.annotations) if row["status"] != "complete"),
            0,
        )
        self.zoom = tk.DoubleVar(value=1.0)
        self.image_mode = tk.StringVar(value="full")
        self.variables: dict[str, tk.StringVar] = {}
        self.canvas = tk.Canvas(root, background="#111")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.panel = ttk.Frame(root, padding=10)
        self.panel.grid(row=1, column=1, sticky="ns")
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.start: tuple[float, float] | None = None
        self.box_id = None
        self.tk_image = None
        self.scale = 1.0
        self._toolbar()
        self._form()
        self.canvas.bind("<ButtonPress-1>", self._start_box)
        self.canvas.bind("<B1-Motion>", self._drag_box)
        self.canvas.bind("<ButtonRelease-1>", self._finish_box)
        root.bind("<Left>", lambda _: self.navigate(-1))
        root.bind("<Right>", lambda _: self.navigate(1))
        self.load()

    def _toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=6)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Button(bar, text="Previous", command=lambda: self.navigate(-1)).pack(side="left")
        ttk.Button(bar, text="Next", command=lambda: self.navigate(1)).pack(side="left")
        ttk.Button(bar, text="Copy Previous BBox", command=self.copy_bbox).pack(side="left", padx=8)
        ttk.Button(bar, text="Save", command=self.save).pack(side="left")
        ttk.Label(bar, text="View").pack(side="left", padx=(16, 2))
        ttk.Combobox(
            bar, textvariable=self.image_mode,
            values=["full", "bowler", "hand"], state="readonly", width=8,
        ).pack(side="left")
        ttk.Label(bar, text="Zoom").pack(side="left", padx=(16, 2))
        ttk.Scale(bar, variable=self.zoom, from_=0.5, to=3.0, command=lambda _: self.load_image()).pack(side="left")
        self.status_label = ttk.Label(bar)
        self.status_label.pack(side="right")
        self.image_mode.trace_add("write", lambda *_: self.load_image())

    def _form(self) -> None:
        for field, values in ANNOTATION_ENUMS.items():
            variable = tk.StringVar()
            self.variables[field] = variable
            ttk.Label(self.panel, text=field.replace("_", " ").title()).pack(anchor="w")
            ttk.Combobox(
                self.panel, textvariable=variable, values=values,
                state="readonly", width=28,
            ).pack(fill="x", pady=(0, 5))
        ttk.Label(self.panel, text="Notes").pack(anchor="w")
        self.notes = tk.Text(self.panel, width=32, height=5)
        self.notes.pack(fill="x")
        ttk.Button(self.panel, text="Mark Complete + Save", command=self.complete).pack(fill="x", pady=8)
        ttk.Button(self.panel, text="Clear BBox", command=self.clear_bbox).pack(fill="x")

    def load(self) -> None:
        row = self.annotations[self.index]
        for field, variable in self.variables.items():
            variable.set(str(row.get(field, "unlabeled")))
        self.notes.delete("1.0", "end")
        self.notes.insert("1.0", row.get("notes", ""))
        self.status_label.config(
            text=f"{self.index + 1}/{len(self.annotations)}  {row['sample_id']}"
        )
        self.load_image()

    def load_image(self) -> None:
        row = self.annotations[self.index]
        meta = self.metadata[row["sample_id"]]
        key = {
            "full": "full_frame_path",
            "bowler": "bowler_roi_path",
            "hand": "hand_roi_path",
        }[self.image_mode.get()]
        path = meta.get(key)
        if not path:
            self.canvas.delete("all")
            self.canvas.create_text(200, 100, text="ROI unavailable", fill="white")
            return
        image = Image.open(ROOT / path)
        max_width = max(320, self.root.winfo_width() - 380)
        max_height = max(320, self.root.winfo_height() - 100)
        base = min(max_width / image.width, max_height / image.height, 1.0)
        self.scale = base * self.zoom.get()
        image = image.resize(
            (max(1, round(image.width * self.scale)), max(1, round(image.height * self.scale)))
        )
        self.tk_image = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")
        self.canvas.config(scrollregion=(0, 0, image.width, image.height))
        if self.image_mode.get() == "full" and row.get("ball_bbox_xyxy"):
            self._draw_saved_box(row["ball_bbox_xyxy"])

    def _start_box(self, event) -> None:
        if self.image_mode.get() != "full":
            return
        self.start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _drag_box(self, event) -> None:
        if not self.start:
            return
        if self.box_id:
            self.canvas.delete(self.box_id)
        self.box_id = self.canvas.create_rectangle(
            *self.start, self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
            outline="#ff00ff", width=2,
        )

    def _finish_box(self, event) -> None:
        if not self.start:
            return
        x1, y1 = self.start
        x2, y2 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        box = [
            min(x1, x2) / self.scale, min(y1, y2) / self.scale,
            max(x1, x2) / self.scale, max(y1, y2) / self.scale,
        ]
        if box[2] - box[0] >= 2 and box[3] - box[1] >= 2:
            row = self.annotations[self.index]
            row["ball_bbox_xyxy"] = [round(value, 2) for value in box]
            row["ball_center"] = [
                round((box[0] + box[2]) / 2, 2),
                round((box[1] + box[3]) / 2, 2),
            ]
            row["ball_size_px"] = round(max(box[2] - box[0], box[3] - box[1]), 2)
            self.variables["ball_visible"].set("yes")
        self.start = None

    def _draw_saved_box(self, box) -> None:
        if self.box_id:
            self.canvas.delete(self.box_id)
        self.box_id = self.canvas.create_rectangle(
            *(value * self.scale for value in box), outline="#ff00ff", width=2
        )

    def clear_bbox(self) -> None:
        row = self.annotations[self.index]
        row["ball_bbox_xyxy"] = row["ball_center"] = row["ball_size_px"] = None
        self.load_image()

    def copy_bbox(self) -> None:
        box = copy_previous_bbox(self.annotations, self.index)
        if box is None:
            messagebox.showinfo("Copy BBox", "No earlier bbox exists in this sequence.")
            return
        self.annotations[self.index]["ball_bbox_xyxy"] = box
        self.annotations[self.index]["ball_center"] = [
            round((box[0] + box[2]) / 2, 2),
            round((box[1] + box[3]) / 2, 2),
        ]
        self.annotations[self.index]["ball_size_px"] = round(
            max(box[2] - box[0], box[3] - box[1]), 2
        )
        self.variables["ball_visible"].set("yes")
        self.load_image()

    def _sync(self) -> None:
        row = self.annotations[self.index]
        for field, variable in self.variables.items():
            row[field] = variable.get()
        row["notes"] = self.notes.get("1.0", "end").strip()
        if row["ball_visible"] in {"no", "uncertain"}:
            row["ball_bbox_xyxy"] = row["ball_center"] = row["ball_size_px"] = None

    def save(self) -> None:
        self._sync()
        self.document["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_annotation_document(ANNOTATIONS, self.document)

    def complete(self) -> None:
        self._sync()
        row = self.annotations[self.index]
        row["status"] = "complete"
        row["annotated_at"] = datetime.now(timezone.utc).isoformat()
        try:
            self.save()
        except ValueError as exc:
            messagebox.showerror("Invalid annotation", str(exc))
            return
        self.navigate(1)

    def navigate(self, delta: int) -> None:
        try:
            self.save()
        except ValueError as exc:
            messagebox.showerror("Invalid annotation", str(exc))
            return
        self.index = max(0, min(len(self.annotations) - 1, self.index + delta))
        self.load()


def main() -> int:
    if not ANNOTATIONS.is_file() or not MANIFEST.is_file():
        raise SystemExit(
            "Run: python scripts/release_region_dataset_builder.py build"
        )
    root = tk.Tk()
    root.title("CricVision Release-Region Dataset Annotator")
    root.geometry("1280x820")
    Annotator(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
