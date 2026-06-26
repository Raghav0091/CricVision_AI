"""LEGACY: early YOLO wrapper kept for reference. Not imported by the active app."""

from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0]
